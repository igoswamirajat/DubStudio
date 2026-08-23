from __future__ import annotations

import json
import logging

from dubstudio.jobs.states import require_transition
from dubstudio.jobs.store import store
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
    store.save(job)
    return job


def run_job(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise KeyError(job_id)
    job_dir = settings.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)

    _advance(job_id, "ingesting", 5, "Validating upload")
    src_files = list((job_dir / "source").glob("*"))
    if not any(p.is_file() and p.name != "meta.json" for p in src_files):
        raise FileNotFoundError("upload missing")

    _advance(job_id, "extracting", 12, "Extracting audio with FFmpeg")
    media_info = run_extract(job_dir)
    duration_s = float(media_info.get("duration_s") or 8.0)
    job = store.get(job_id)
    job["duration_s"] = duration_s
    store.save(job)

    job = store.get(job_id)
    skip_sep = bool(job.get("skip_separation", True)) or settings.skip_separation
    _advance(job_id, "separating", 20, "Skipping separation (Phase 1)" if skip_sep else "Separating dialogue / bed")
    run_separation(job_dir, skip=skip_sep)

    _advance(job_id, "transcribing", 32, "Transcribing speech")
    transcript = run_transcription(job_dir, model=settings.whisper_model)
    job = store.get(job_id)
    if not job.get("source_language"):
        job["source_language"] = transcript.get("language") or "en"
        store.save(job)

    _advance(job_id, "diarizing", 40, "Speaker labels")
    sp_raw = sorted({str(w.get("speaker", "S00")) for w in transcript.get("words", [])}) or ["S00"]
    speakers = {"speakers": [{"speaker_id": s if s.startswith("S") else f"S{i:02d}", "label": f"Speaker {i}"} for i, s in enumerate(sp_raw)]}
    (job_dir / "asr" / "speakers.json").write_text(json.dumps(speakers, indent=2), encoding="utf-8")

    _advance(job_id, "segmenting", 48, "Building dialogue segments")
    segs = words_to_segments(transcript.get("words", []), job_id=job_id)
    seg_dir = job_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "segments.json").write_text(json.dumps(segs, indent=2, ensure_ascii=False), encoding="utf-8")

    job = store.get(job_id)
    _advance(job_id, "translating", 58, f"Translating to {job.get('target_language')}")
    job = store.get(job_id)
    run_translation(job_dir, job)

    _advance(job_id, "enrolling_voices", 65, "Enrolling speaker voices")
    speaker_map = run_enroll(job_dir)
    job = store.get(job_id)
    job["speakers"] = speaker_map.get("speakers", [])
    store.save(job)

    _advance(job_id, "synthesizing", 75, "Generating target speech")
    job = store.get(job_id)
    job["tts_engine"] = settings.tts_engine
    store.save(job)
    run_synthesis(job_dir, job)

    _advance(job_id, "fitting", 85, "Fitting timing")
    run_timing(job_dir)

    _advance(job_id, "mixing", 92, "Mixing dialogue with bed")
    run_mixing(job_dir, duration_s)

    _advance(job_id, "exporting", 97, "Remuxing final MP4")
    job = store.get(job_id)
    artifacts = run_export(job_dir, job)

    job = store.get(job_id)
    require_transition(job["state"], "completed")
    job["state"] = "completed"
    job["percent"] = 100
    job["message"] = "Completed"
    job["error"] = None
    job["artifacts"] = artifacts
    store.save(job)
    return job
