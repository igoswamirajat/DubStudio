"""Phase 8 — segment timeline editor API."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dubstudio.engines.base import SynthRequest
from dubstudio.engines.factory import get_voice_engine
from dubstudio.jobs.store import store
from dubstudio.pipeline.export import run_export
from dubstudio.pipeline.mixing import run_mixing
from dubstudio.pipeline.timing import run_timing
from dubstudio.settings import settings
from dubstudio.util.paths import rel_posix

router = APIRouter()


class SegmentPatch(BaseModel):
    source_text: str | None = None
    translated_text: str | None = None
    speaker_id: str | None = None
    voice_mode: str | None = None
    start: float | None = None
    end: float | None = None


class ResynthBody(BaseModel):
    text: str | None = Field(default=None, description="Override translated_text for this synth")


def _segs_path(job_id: str) -> Path:
    return settings.jobs_dir / job_id / "segments" / "segments.json"


def _load(job_id: str) -> list[dict]:
    path = _segs_path(job_id)
    if not path.exists():
        raise HTTPException(404, "segments not ready")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(job_id: str, segments: list[dict]) -> None:
    path = _segs_path(job_id)
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")


@router.get("/jobs/{job_id}/segments")
def list_segments(job_id: str):
    if not store.get(job_id):
        raise HTTPException(404, "job not found")
    path = _segs_path(job_id)
    if not path.exists():
        return {"segments": []}
    return {"segments": json.loads(path.read_text(encoding="utf-8"))}


@router.patch("/jobs/{job_id}/segments/{segment_id}")
def patch_segment(job_id: str, segment_id: str, body: SegmentPatch):
    if not store.get(job_id):
        raise HTTPException(404, "job not found")
    segments = _load(job_id)
    found = None
    for s in segments:
        if s.get("segment_id") == segment_id:
            found = s
            break
    if not found:
        raise HTTPException(404, f"segment {segment_id} not found")
    data = body.model_dump(exclude_none=True)
    found.update(data)
    if "start" in data or "end" in data:
        start = float(found.get("start") or 0)
        end = float(found.get("end") or start)
        found["duration_ms"] = int(max(0, (end - start) * 1000))
    _save(job_id, segments)
    return found


@router.post("/jobs/{job_id}/segments/{segment_id}/resynth")
def resynth_segment(job_id: str, segment_id: str, body: ResynthBody | None = None):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    segments = _load(job_id)
    found = None
    for s in segments:
        if s.get("segment_id") == segment_id:
            found = s
            break
    if not found:
        raise HTTPException(404, f"segment {segment_id} not found")

    text = (body.text if body and body.text else None) or found.get("translated_text") or found.get("source_text") or ""
    if body and body.text:
        found["translated_text"] = body.text

    job_dir = settings.jobs_dir / job_id
    engine = get_voice_engine(job.get("tts_engine"))
    synth_dir = job_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    out = synth_dir / f"{segment_id}.wav"
    voice_id = found.get("voice_id") or found.get("speaker_id") or "S00"
    ref = job_dir / "voices" / voice_id / "ref.wav"
    result = engine.generate(
        SynthRequest(
            text=text,
            language=job.get("target_language") or "hi",
            voice_id=voice_id,
            ref_wav=ref if ref.exists() else None,
            voice_mode=found.get("voice_mode") or "clone",
        ),
        out,
    )
    found["generated_wav"] = rel_posix(out, job_dir)
    found["generated_duration_ms"] = result.duration_ms
    found["status"] = "synthesized"
    _save(job_id, segments)

    run_timing(job_dir)
    duration_s = 8.0
    meta = job_dir / "source" / "meta.json"
    if meta.exists():
        try:
            duration_s = float(json.loads(meta.read_text()).get("format", {}).get("duration") or 8)
        except Exception:
            pass
    run_mixing(job_dir, duration_s)
    artifacts = run_export(job_dir, job)
    job = store.get(job_id)
    job["artifacts"] = artifacts
    store.save(job)

    segments = _load(job_id)
    found = next(s for s in segments if s.get("segment_id") == segment_id)
    return {"segment": found, "artifacts": artifacts}


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    from dubstudio.jobs.runner import enqueue

    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["state"] == "completed":
        return {"ok": True, "message": "already completed"}
    await enqueue(job_id)
    return {"ok": True, "job_id": job_id, "state": "queued"}
