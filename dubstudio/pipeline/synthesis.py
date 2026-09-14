from __future__ import annotations

import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any

from dubstudio.engines.base import SynthRequest
from dubstudio.engines.factory import get_voice_engine
from dubstudio.jobs.store import store

log = logging.getLogger("dubstudio.synthesis")

SEGMENT_TIMEOUT_S = 120
ATTEMPTS_PER_SEGMENT = 2          # retries on the speaker's OWN engine first
FALLBACK_ENGINE = "veena"
FALLBACK_VOICE_MODE = "standard"


def _speaker_lookup(job_dir: Path) -> dict[str, dict]:
    path = job_dir / "voices" / "speaker_map.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["speaker_id"]: s for s in data.get("speakers", []) if s.get("speaker_id")}


def _safe_save_job(job: dict) -> None:
    if not job or not job.get("job_id"):
        return
    try:
        store.save(job)
    except Exception:
        pass


class _EnginePool:
    """Lazily instantiates engines and keeps one instance per engine name."""

    def __init__(self) -> None:
        self._engines: dict[str, Any] = {}

    def get(self, name: str):
        if name not in self._engines:
            self._engines[name] = get_voice_engine(name)
        return self._engines[name]


def _preferred_engine(default_engine_name: str, voice_mode: str) -> str:
    if default_engine_name == "dummy":
        return "dummy"
    if voice_mode in {"clone", "design"}:
        return "omnivoice"
    return "veena"


def build_voice_plan(segments: list[dict], speakers: dict[str, dict], default_engine_name: str) -> dict[str, dict]:
    """Lock one engine + voice per speaker for the WHOLE job.

    A speaker's voice identity must never change part-way through a video, so
    the engine is decided once, up front, instead of being flipped globally by
    the first segment that happens to time out.
    """
    plan: dict[str, dict] = {}
    for seg in segments:
        sid = seg.get("speaker_id") or "S00"
        if sid in plan:
            continue
        sp = speakers.get(sid) or {}
        voice_mode = seg.get("voice_mode") or sp.get("voice_mode") or "clone"
        plan[sid] = {
            "engine": _preferred_engine(default_engine_name, voice_mode),
            "voice_mode": voice_mode,
            "voice_id": seg.get("voice_id") or sp.get("voice_id") or sid,
            "design_prompt": seg.get("design_prompt") or sp.get("design_prompt"),
            "ref_text": sp.get("ref_text"),
            "downgraded": False,
        }
    return plan


def _ref_wav(job_dir: Path, sid: str, voice_id: str) -> Path | None:
    for candidate in (job_dir / "voices" / voice_id / "ref.wav", job_dir / "voices" / sid / "ref.wav"):
        if candidate.exists():
            return candidate
    return None


def _generate_once(engine, seg: dict, entry: dict, lang: str, ref: Path | None, out: Path, timeout_s: int):
    request = SynthRequest(
        text=seg.get("translated_text") or seg.get("source_text") or "",
        language=lang,
        voice_id=entry["voice_id"],
        ref_wav=ref,
        voice_mode=entry["voice_mode"],
        ref_text=entry.get("ref_text"),
        instruct=entry.get("design_prompt"),
        target_duration_ms=seg.get("target_duration_ms"),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(engine.generate, request, out).result(timeout=timeout_s)


def _render_segment(pool: _EnginePool, job_dir: Path, seg: dict, entry: dict, lang: str,
                    synth_dir: Path, timeout_s: int) -> tuple[bool, str | None]:
    """Render one segment on the speaker's locked engine, retrying that same
    engine before giving up. Returns (ok, error)."""
    sid = seg.get("speaker_id") or "S00"
    out = synth_dir / f"{seg['segment_id']}.wav"
    ref = _ref_wav(job_dir, sid, entry["voice_id"])
    engine = pool.get(entry["engine"])
    last_error: str | None = None

    for attempt in range(1, ATTEMPTS_PER_SEGMENT + 1):
        try:
            result = _generate_once(engine, seg, entry, lang, ref, out, timeout_s)
        except concurrent.futures.TimeoutError:
            last_error = f"timeout after {timeout_s}s on {entry['engine']}"
            log.warning("Segment %s attempt %d/%d: %s", seg.get("segment_id"), attempt,
                        ATTEMPTS_PER_SEGMENT, last_error)
            continue
        except Exception as exc:  # noqa: BLE001 - engine errors are reported, not raised
            last_error = str(exc)
            log.warning("Segment %s attempt %d/%d failed on %s: %s", seg.get("segment_id"),
                        attempt, ATTEMPTS_PER_SEGMENT, entry["engine"], exc)
            continue

        seg["generated_wav"] = str(out.relative_to(job_dir))
        seg["generated_duration_ms"] = result.duration_ms
        seg["status"] = "synthesized"
        seg["tts_engine"] = result.engine
        seg["voice_mode"] = entry["voice_mode"]
        seg["voice_id"] = entry["voice_id"]
        seg.pop("error", None)
        return True, None

    return False, last_error


def run_synthesis(job_dir: Path, job: dict) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    synth_dir = job_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    lang = job.get("target_language") or "hi"
    timeout_s = int(job.get("synthesis_timeout_s") or SEGMENT_TIMEOUT_S)
    default_engine_name = job.get("tts_engine") or "veena"
    speakers = _speaker_lookup(job_dir)
    plan = build_voice_plan(segments, speakers, default_engine_name)
    pool = _EnginePool()
    total = len(segments)

    log.info("Starting synthesis for job %s: %d segments, voice plan: %s",
             job.get("job_id"), total,
             {sid: entry["engine"] for sid, entry in plan.items()})
    if job.get("job_id"):
        job["message"] = "Initializing neural voice synthesis..."
        _safe_save_job(job)

    downgraded: list[str] = []
    errors: dict[str, str] = {}

    # Pass 1 - every segment on its speaker's locked engine.
    for i, seg in enumerate(segments):
        sid = seg.get("speaker_id") or "S00"
        entry = plan[sid]
        if job.get("job_id") and total > 0:
            job["percent"] = min(75 + int((i / total) * 10), 84)
            job["message"] = f"Synthesizing speech {i + 1}/{total} ({sid})"
            _safe_save_job(job)

        ok, error = _render_segment(pool, job_dir, seg, entry, lang, synth_dir, timeout_s)
        if ok:
            continue

        errors[seg["segment_id"]] = error or "unknown synthesis error"
        if entry["engine"] != FALLBACK_ENGINE and not entry["downgraded"]:
            # Downgrade this SPEAKER only, and re-render all of that speaker's
            # segments in pass 2 so the voice stays identical end to end.
            log.warning("Speaker %s: %s failed, downgrading the whole speaker to %s",
                        sid, entry["engine"], FALLBACK_ENGINE)
            entry["engine"] = FALLBACK_ENGINE
            entry["voice_mode"] = FALLBACK_VOICE_MODE
            entry["downgraded"] = True
            if sid not in downgraded:
                downgraded.append(sid)
        else:
            seg["status"] = "failed"
            seg["error"] = errors[seg["segment_id"]]

    # Pass 2 - re-render every segment of each downgraded speaker on the
    # fallback engine, including the ones that already succeeded, so a single
    # bad segment can no longer change the speaker's voice mid-video.
    for sid in downgraded:
        entry = plan[sid]
        targets = [s for s in segments if (s.get("speaker_id") or "S00") == sid]
        log.warning("Re-rendering %d segment(s) for speaker %s on %s for voice consistency",
                    len(targets), sid, entry["engine"])
        if job.get("job_id"):
            job["message"] = f"Re-rendering {sid} on {entry['engine']} for a consistent voice"
            _safe_save_job(job)
        for seg in targets:
            ok, error = _render_segment(pool, job_dir, seg, entry, lang, synth_dir, timeout_s)
            if ok:
                errors.pop(seg["segment_id"], None)
            else:
                seg["status"] = "failed"
                seg["error"] = error or "unknown synthesis error"
                seg.pop("generated_wav", None)
                errors[seg["segment_id"]] = seg["error"]

    failed = [s["segment_id"] for s in segments if s.get("status") == "failed"]
    if failed:
        log.error("Synthesis finished with %d failed segment(s): %s", len(failed), failed)

    if job.get("job_id"):
        job["percent"] = 85
        job["failed_segments"] = failed
        job["voice_plan"] = {sid: entry["engine"] for sid, entry in plan.items()}
        job["message"] = (
            f"Completed synthesis of all {total} segments"
            if not failed
            else f"Synthesized {total - len(failed)}/{total} segments, {len(failed)} failed"
        )
        _safe_save_job(job)

    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
