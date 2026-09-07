from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logging

from dubstudio.engines.base import SynthRequest
from dubstudio.engines.factory import get_voice_engine
from dubstudio.jobs.store import store

log = logging.getLogger("dubstudio.synthesis")


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


def run_synthesis(job_dir: Path, job: dict) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    default_engine_name = job.get("tts_engine") or "veena"
    synth_dir = job_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    lang = job.get("target_language") or "hi"
    speakers = _speaker_lookup(job_dir)
    total = len(segments)

    _engines: dict[str, Any] = {}

    def _resolve_engine(v_mode: str, v_id: str):
        if default_engine_name == "dummy":
            target = "dummy"
        elif v_mode in {"clone", "design"}:
            target = "omnivoice"
        else:
            target = "veena"
        if target not in _engines:
            _engines[target] = get_voice_engine(target)
        return _engines[target]

    log.info("Starting synthesis for job %s: %d segments", job.get("job_id"), total)
    if job.get("job_id"):
        job["message"] = "Initializing neural voice synthesis..."
        _safe_save_job(job)

    import concurrent.futures
    import time

    # Track if we've already fallen back to Veena
    fallback_to_veena = False

    for i, seg in enumerate(segments):
        sid = seg.get("speaker_id") or "S00"
        if job.get("job_id") and total > 0:
            pct = 75 + int((i / total) * 10)
            job["percent"] = min(pct, 84)
            job["message"] = f"Synthesizing speech {i + 1}/{total} ({sid})"
            _safe_save_job(job)

        out = synth_dir / f"{seg['segment_id']}.wav"
        sp = speakers.get(sid) or {}
        voice_id = seg.get("voice_id") or sp.get("voice_id") or sid
        voice_mode = seg.get("voice_mode") or sp.get("voice_mode") or "clone"
        design_prompt = seg.get("design_prompt") or sp.get("design_prompt")
        ref_text = sp.get("ref_text")

        ref = None
        for candidate in (
            job_dir / "voices" / voice_id / "ref.wav",
            job_dir / "voices" / sid / "ref.wav",
        ):
            if candidate.exists():
                ref = candidate
                break

        text = seg.get("translated_text") or seg.get("source_text") or ""

        # If fallback triggered, force Veena for this segment
        if fallback_to_veena:
            engine = get_voice_engine("veena")
        else:
            engine = _resolve_engine(voice_mode, voice_id)

        log.info(
            "Generating segment %d/%d (%s) with %s: '%s'",
            i + 1,
            total,
            seg.get("segment_id"),
            engine.name,
            text[:40],
        )

        # Use a timeout for generation
        def _generate():
            return engine.generate(
                SynthRequest(
                    text=text,
                    language=lang,
                    voice_id=voice_id,
                    ref_wav=ref,
                    voice_mode=voice_mode,
                    ref_text=ref_text,
                    instruct=design_prompt,
                    target_duration_ms=seg.get("target_duration_ms"),
                ),
                out,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_generate)
                result = future.result(timeout=120)  # 120 seconds timeout
        except concurrent.futures.TimeoutError:
            log.error("Segment %d/%d timed out after 120s with %s", i+1, total, engine.name)
            # If we're using OmniVoice, fallback to Veena for this and subsequent segments
            if engine.name == "omnivoice" and not fallback_to_veena:
                log.warning("Falling back to Veena engine for remaining segments")
                fallback_to_veena = True
                # Retry this segment with Veena
                engine = get_voice_engine("veena")
                try:
                    result = engine.generate(
                        SynthRequest(
                            text=text,
                            language=lang,
                            voice_id=voice_id,
                            ref_wav=ref,
                            voice_mode="standard",  # Veena doesn't use clone mode
                            ref_text=ref_text,
                            instruct=design_prompt,
                            target_duration_ms=seg.get("target_duration_ms"),
                        ),
                        out,
                    )
                except Exception as e2:
                    log.error("Fallback Veena also failed for segment %d: %s", i+1, e2)
                    seg["status"] = "failed"
                    seg["error"] = str(e2)
                    continue
            else:
                log.error("No fallback available, marking segment as failed")
                seg["status"] = "failed"
                seg["error"] = "Timeout"
                continue
        except Exception as e:
            log.error("Segment %d/%d failed with %s: %s", i+1, total, engine.name, e)
            if engine.name == "omnivoice" and not fallback_to_veena:
                log.warning("Falling back to Veena engine for remaining segments")
                fallback_to_veena = True
                engine = get_voice_engine("veena")
                try:
                    result = engine.generate(
                        SynthRequest(
                            text=text,
                            language=lang,
                            voice_id=voice_id,
                            ref_wav=ref,
                            voice_mode="standard",
                            ref_text=ref_text,
                            instruct=design_prompt,
                            target_duration_ms=seg.get("target_duration_ms"),
                        ),
                        out,
                    )
                except Exception as e2:
                    log.error("Fallback Veena also failed: %s", e2)
                    seg["status"] = "failed"
                    seg["error"] = str(e2)
                    continue
            else:
                seg["status"] = "failed"
                seg["error"] = str(e)
                continue

        seg["generated_wav"] = str(out.relative_to(job_dir))
        seg["generated_duration_ms"] = result.duration_ms
        seg["status"] = "synthesized"
        seg["tts_engine"] = result.engine
        seg["voice_mode"] = voice_mode if not fallback_to_veena else "standard"
        log.info("Segment %d/%d generated (%d ms)", i + 1, total, result.duration_ms)

    if job.get("job_id"):
        job["percent"] = 85
        job["message"] = f"Completed synthesis of all {total} segments"
        _safe_save_job(job)

    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
