"""Speaker cards API (Phase 2).

After diarization + enroll, UI shows one card per speaker:
  - preview ref clip
  - voice_mode: clone | design | fixed
  - design_prompt / voice_id overrides
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dubstudio.jobs.store import store
from dubstudio.settings import settings

router = APIRouter()


class SpeakerPatch(BaseModel):
    voice_mode: str | None = Field(default=None, description="clone | design | fixed | auto")
    voice_id: str | None = None
    label: str | None = None
    design_prompt: str | None = Field(
        default=None,
        description='OmniVoice instruct e.g. "female, young adult, hindi accent"',
    )
    ref_text: str | None = Field(default=None, description="Transcript of the ref clip (optional)")


def _map_path(job_id: str) -> Path:
    return settings.jobs_dir / job_id / "voices" / "speaker_map.json"


def _load_map(job_id: str) -> dict:
    path = _map_path(job_id)
    if not path.exists():
        raise HTTPException(404, {"error": {"code": "SPEAKERS_NOT_READY", "message": job_id}})
    return json.loads(path.read_text(encoding="utf-8"))


def _save_map(job_id: str, payload: dict) -> None:
    path = _map_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    # also mirror overrides for re-enroll / re-synth paths
    ov = path.parent / "overrides.json"
    overrides = {}
    for s in payload.get("speakers", []):
        overrides[s["speaker_id"]] = {
            "voice_mode": s.get("voice_mode", "clone"),
            "voice_id": s.get("voice_id", s["speaker_id"]),
            "label": s.get("label"),
            "design_prompt": s.get("design_prompt"),
            "ref_text": s.get("ref_text"),
        }
    ov.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")


@router.get("/jobs/{job_id}/speakers")
def list_speakers(job_id: str):
    if not store.get(job_id):
        raise HTTPException(404, "job not found")
    return _load_map(job_id)


@router.patch("/jobs/{job_id}/speakers/{speaker_id}")
def patch_speaker(job_id: str, speaker_id: str, body: SpeakerPatch):
    if not store.get(job_id):
        raise HTTPException(404, "job not found")
    payload = _load_map(job_id)
    found = None
    for s in payload.get("speakers", []):
        if s.get("speaker_id") == speaker_id:
            found = s
            break
    if not found:
        raise HTTPException(404, {"error": {"code": "SPEAKER_NOT_FOUND", "message": speaker_id}})

    data = body.model_dump(exclude_none=True)
    if "voice_mode" in data:
        mode = data["voice_mode"].lower()
        if mode not in {"clone", "design", "fixed", "auto"}:
            raise HTTPException(400, f"invalid voice_mode: {mode}")
        found["voice_mode"] = mode
    for key in ("voice_id", "label", "design_prompt", "ref_text"):
        if key in data:
            found[key] = data[key]

    _save_map(job_id, payload)

    # keep job.speakers in sync for UI
    job = store.get(job_id)
    if job is not None:
        job["speakers"] = payload.get("speakers", [])
        store.save(job)

    return found


@router.get("/jobs/{job_id}/speakers/{speaker_id}/ref")
def speaker_ref_audio(job_id: str, speaker_id: str):
    """Stream the short reference clip used for cloning."""
    if not store.get(job_id):
        raise HTTPException(404, "job not found")
    payload = _load_map(job_id)
    for s in payload.get("speakers", []):
        if s.get("speaker_id") == speaker_id:
            rel = s.get("ref_wav")
            if not rel:
                break
            path = settings.jobs_dir / job_id / rel
            if path.exists():
                return FileResponse(path, media_type="audio/wav", filename=f"{speaker_id}_ref.wav")
            break
    raise HTTPException(404, "ref audio not found")
