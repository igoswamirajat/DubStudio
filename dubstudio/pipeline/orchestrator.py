from __future__ import annotations

import json
import logging

from dubstudio.jobs.states import require_transition, PIPELINE_STAGES
from dubstudio.jobs.store import store
from dubstudio.pipeline.checkpoint import clear_from, is_done, mark_done
from dubstudio.pipeline.diarization import run_diarization
from dubstudio.pipeline.export import run_export
from dubstudio.pipeline.media import run_extract
from dubstudio.pipeline.mixing import run_mixing
from dubstudio.pipeline.segments import words_to_segments
from dubstudio.pipeline.separation import run_separation
from dubstudio.pipeline.synthesis import run_synthesis
from dubstudio.pipeline.timing import run_timing
from dubstudio.pipeline.transcription import run_transcription
from dubstudio.pipeline.translation import run_translation
from dubstudio.pipeline.voice_enroll import run_enroll
from dubstudio.settings import settings

log = logging.getLogger("dubstudio.orchestrator")


def _advance(job_id: str, state: str, percent: int, message: str) -> dict:
    job = store.get(job_id)
    assert job
    require_transition(job["state"], state)
    job["state"] = state
    job["percent"] = percent
    job["message"] = message
    job["stage_total"] = len(PIPELINE_STAGES)
    job["stage_index"] = (PIPELINE_STAGES.index(state) + 1) if state in PIPELINE_STAGES else job.get("stage_index", 0)
    store.save(job)
    return job


def _cleanup_memory():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _load_transcript(job_dir) -> dict | None:
    path = job_dir / "asr" / "transcript.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def run_job(job_id: str) -> dict:
    """Run the analysis phase of the pipeline, then pause for voice selection."""
    job = store.get(job_id)
    if not job:
        raise KeyError(job_id)
    job_dir = settings.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)

    resume = bool(settings.enable_resume)

    def done(stage: str) -> bool:
        return resume and is_done(job_dir, stage)

    # --- ingest ---
    _advance(job_id, "ingesting", 5, "Validating upload")
    src_files = list((job_dir / "source").glob("*"))
    if not any(p.is_file() and p.name != "meta.json" for p in src_files):
        raise FileNotFoundError("upload missing")
    mark_done(job_dir, "ingesting")

    # --- extract audio ---
    _advance(job_id, "extracting", 12, "Extracting audio with FFmpeg")
    full_wav = job_dir / "audio" / "full.wav"
    if done("extracting") and full_wav.exists():
        duration_s = float(job.get("duration_s") or 8.0)
    else:
        media_info = run_extract(job_dir)
        duration_s = float(media_info.get("duration_s") or 8.0)
        job = store.get(job_id)
        job["duration_s"] = duration_s
        store.save(job)
        mark_done(job_dir, "extracting", {"duration_s": duration_s})

    # --- separation ---
    job = store.get(job_id)
    skip_sep = bool(job.get("skip_separation", False)) or settings.skip_separation
    _advance(job_id, "separating", 20, "Copying audio (separation skipped)" if skip_sep else "Separating dialogue / bed (Demucs)")
    vocals = job_dir / "audio" / "vocals.wav"
    bed = job_dir / "audio" / "bed.wav"
    if not (done("separating") and vocals.exists() and bed.exists()):
        run_separation(job_dir, skip=skip_sep)
        mark_done(job_dir, "separating")

    # --- transcription ---
    _advance(job_id, "transcribing", 32, "Transcribing speech")
    transcript = None
    if done("transcribing") and (job_dir / "asr" / "transcript.json").exists():
        transcript = _load_transcript(job_dir)
    if transcript is None:
        transcript = run_transcription(job_dir, model=settings.whisper_model)
        mark_done(job_dir, "transcribing")
    job = store.get(job_id)
    if not job.get("source_language"):
        job["source_language"] = transcript.get("language") or "en"
        store.save(job)

    # --- diarization ---
    _advance(job_id, "diarizing", 40, "Detecting speakers")
    if not done("diarizing"):
        run_diarization(job_dir, transcript)
        mark_done(job_dir, "diarizing")
    else:
        transcript = _load_transcript(job_dir) or transcript

    # --- segmenting ---
    _advance(job_id, "segmenting", 48, "Building dialogue segments")
    seg_dir = job_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    if not (done("segmenting") and (seg_dir / "segments.json").exists()):
        segs = words_to_segments(transcript.get("words", []), job_id=job_id)
        (seg_dir / "segments.json").write_text(json.dumps(segs, indent=2, ensure_ascii=False), encoding="utf-8")
        mark_done(job_dir, "segmenting")

    # --- translation ---
    job = store.get(job_id)
    _advance(job_id, "translating", 58, f"Translating to {job.get('target_language')}")
    if not done("translating"):
        job = store.get(job_id)
        run_translation(job_dir, job)
        mark_done(job_dir, "translating")

    # --- enroll voices ---
    _advance(job_id, "enrolling_voices", 65, "Enrolling speaker voices")
    if not (done("enrolling_voices") and (job_dir / "voices" / "speaker_map.json").exists()):
        speaker_map = run_enroll(job_dir, job=job)
        job = store.get(job_id)
        job["speakers"] = speaker_map.get("speakers", [])
        store.save(job)
        mark_done(job_dir, "enrolling_voices")

    # --- PAUSE: await voice selection ---
    # Pipeline pauses here. The job thread exits, GPU lock is released.
    # User selects voices in the UI, then POST /voice-options/apply triggers run_job_synthesis.
    _advance(job_id, "awaiting_voice_selection", 67,
             "Choose voices for each speaker to continue")
    log.info("Job %s paused at awaiting_voice_selection — waiting for user input", job_id)
    return store.get(job_id)


def _apply_voice_selections(job_dir: Path) -> None:
    """Re-read speaker_map.json (updated by user selections) and propagate voice_id to all segments."""
    map_path = job_dir / "voices" / "speaker_map.json"
    if not map_path.exists():
        return
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    speaker_voices = {s["speaker_id"]: s for s in payload.get("speakers", [])}

    seg_path = job_dir / "segments" / "segments.json"
    if not seg_path.exists():
        return
    segments = json.loads(seg_path.read_text(encoding="utf-8"))
    for seg in segments:
        sid = seg.get("speaker_id") or "S00"
        sp = speaker_voices.get(sid, {})
        if sp.get("voice_id"):
            seg["voice_id"] = sp["voice_id"]
        if sp.get("voice_mode"):
            seg["voice_mode"] = sp["voice_mode"]
    seg_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")


def run_job_synthesis(job_id: str) -> dict:
    """Resume pipeline after user selects voices — runs synthesis through export."""
    job = store.get(job_id)
    if not job:
        raise KeyError(job_id)
    job_dir = settings.jobs_dir / job_id

    resume = bool(settings.enable_resume)

    def done(stage: str) -> bool:
        return resume and is_done(job_dir, stage)

    # Clear downstream checkpoints to guarantee a fresh synthesis pass with selected voices
    clear_from(job_dir, "synthesizing")

    # Re-propagate user's voice selections to segments
    _apply_voice_selections(job_dir)

    # Re-read duration_s
    duration_s = float(job.get("duration_s") or 8.0)

    # --- synthesis ---
    _advance(job_id, "synthesizing", 75, "Generating target speech")
    job = store.get(job_id)
    job["tts_engine"] = job.get("tts_engine") or settings.tts_engine
    store.save(job)
    if not done("synthesizing"):
        _cleanup_memory()
        run_synthesis(job_dir, job)
        mark_done(job_dir, "synthesizing")

    # --- timing fit ---
    _advance(job_id, "fitting", 85, "Fitting timing")
    if not done("fitting"):
        run_timing(job_dir, job)
        mark_done(job_dir, "fitting")

    # --- mixing ---
    _advance(job_id, "mixing", 92, "Mixing dialogue with bed")
    if not done("mixing"):
        run_mixing(job_dir, duration_s)
        mark_done(job_dir, "mixing")

    # --- export (always, to publish artifact paths) ---
    _advance(job_id, "exporting", 97, "Remuxing final MP4")
    job = store.get(job_id)
    artifacts = run_export(job_dir, job)
    mark_done(job_dir, "exporting")

    job = store.get(job_id)
    require_transition(job["state"], "completed")
    job["state"] = "completed"
    job["percent"] = 100
    job["message"] = "Completed"
    job["error"] = None
    job["artifacts"] = artifacts
    store.save(job)
    return job
