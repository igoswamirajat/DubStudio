"""Voice Selection API — human-in-the-loop voice picking.

After the analysis phase pauses at ``awaiting_voice_selection``, these
endpoints let the frontend fetch available voices, apply user selections,
and resume the synthesis pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dubstudio.jobs.runner import resume_synthesis
from dubstudio.jobs.store import store
from dubstudio.pipeline.voice_matcher import VEENA_VOICES
from dubstudio.settings import settings

router = APIRouter()


# ---------- schemas ----------

class VoiceSelection(BaseModel):
    voice_id: str
    voice_mode: str = "native"          # native | clone | design
    design_prompt: str | None = None    # only when voice_mode == "design"


class ApplySelectionsBody(BaseModel):
    selections: dict[str, VoiceSelection]   # speaker_id → choice


# ---------- helpers ----------

def _map_path(job_id: str) -> Path:
    return settings.jobs_dir / job_id / "voices" / "speaker_map.json"


def _load_map(job_id: str) -> dict:
    path = _map_path(job_id)
    if not path.exists():
        raise HTTPException(404, {"error": {"code": "SPEAKERS_NOT_READY", "message": job_id}})
    return json.loads(path.read_text(encoding="utf-8"))


_AVAILABLE_VOICES = [
    {
        "voice_id": "clone_original",
        "engine": "omnivoice",
        "label": "🔊 Clone Original Voice (Recommended)",
        "description": "Clones the original speaker's exact voice — 100% pitch & gender consistency with zero voice drift",
        "gender": None,
        "category": "clone",
    },
]

# Add Veena native voices from the voice catalog
for _vid, _vc in VEENA_VOICES.items():
    _AVAILABLE_VOICES.append({
        "voice_id": _vc.voice_id,
        "engine": "veena",
        "label": f"🎭 {_vc.voice_id.capitalize()} (Experimental)",
        "description": f"{_vc.description} (Experimental Maya Veena voice; may exhibit pitch drift on male dialogue)",
        "gender": _vc.gender,
        "category": "native",
    })

_AVAILABLE_VOICES.append({
    "voice_id": "design_custom",
    "engine": "omnivoice",
    "label": "✨ Design Custom Voice",
    "description": "Describe the voice you want in natural language (OmniVoice instruct prompt)",
    "gender": None,
    "category": "design",
})


# ---------- endpoints ----------

@router.get("/jobs/{job_id}/voice-options")
def get_voice_options(job_id: str):
    """Return detected speakers and available voice options for the job."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    payload = _load_map(job_id)
    speakers_out = []
    for s in payload.get("speakers", []):
        speakers_out.append({
            "speaker_id": s["speaker_id"],
            "label": s.get("label", s["speaker_id"]),
            "segment_count": s.get("segment_count", 0),
            "ref_audio_url": f"/api/v1/jobs/{job_id}/speakers/{s['speaker_id']}/ref",
            "auto_suggestion": {
                "voice_id": s.get("voice_id", "clone_original"),
                "confidence": s.get("auto_confidence", 0.7),
            },
            "current_voice_id": s.get("voice_id", "clone_original"),
            "current_voice_mode": s.get("voice_mode", "clone"),
            "gender": s.get("gender"),
            "f0_median": s.get("f0_median"),
        })

    return {
        "job_state": job.get("state"),
        "tts_engine": job.get("tts_engine", "veena"),
        "speakers": speakers_out,
        "available_voices": _AVAILABLE_VOICES,
    }


@router.post("/jobs/{job_id}/voice-options/apply")
async def apply_voice_selections(job_id: str, body: ApplySelectionsBody):
    """Apply user voice selections and resume the pipeline."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.get("state") != "awaiting_voice_selection":
        raise HTTPException(
            409,
            {
                "error": {
                    "code": "INVALID_STATE",
                    "message": (
                        f"Job is in state '{job.get('state')}', "
                        "expected 'awaiting_voice_selection'"
                    ),
                }
            },
        )

    # Update speaker_map.json with user's selections
    payload = _load_map(job_id)
    speakers = {s["speaker_id"]: s for s in payload.get("speakers", [])}

    for sid, sel in body.selections.items():
        sp = speakers.get(sid)
        if not sp:
            continue

        if sel.voice_id == "clone_original":
            # OmniVoice clone mode — use the speaker's own ref
            sp["voice_id"] = sid
            sp["voice_mode"] = "clone"
        elif sel.voice_id == "design_custom":
            sp["voice_id"] = sid
            sp["voice_mode"] = "design"
            sp["design_prompt"] = sel.design_prompt or sp.get("design_prompt", "")
        else:
            # Veena native voice
            sp["voice_id"] = sel.voice_id
            sp["voice_mode"] = sel.voice_mode or "native"

    payload["speakers"] = list(speakers.values())
    map_path = _map_path(job_id)
    map_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Also write overrides.json for consistency
    overrides = {}
    for s in payload["speakers"]:
        overrides[s["speaker_id"]] = {
            "voice_mode": s.get("voice_mode", "clone"),
            "voice_id": s.get("voice_id", s["speaker_id"]),
            "label": s.get("label"),
            "design_prompt": s.get("design_prompt"),
        }
    ov_path = settings.jobs_dir / job_id / "voices" / "overrides.json"
    ov_path.write_text(
        json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Sync speakers to job store
    job["speakers"] = payload["speakers"]
    store.save(job)

    # Resume pipeline → synthesizing
    await resume_synthesis(job_id)

    return {"ok": True, "state": "synthesizing", "job_id": job_id}


@router.post("/jobs/{job_id}/voice-options/auto-apply")
async def auto_apply_voice_selections(job_id: str):
    """Apply AI-suggested voice selections and resume the pipeline (one-click)."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.get("state") != "awaiting_voice_selection":
        raise HTTPException(
            409,
            {
                "error": {
                    "code": "INVALID_STATE",
                    "message": (
                        f"Job is in state '{job.get('state')}', "
                        "expected 'awaiting_voice_selection'"
                    ),
                }
            },
        )

    # Keep existing speaker_map.json as-is (already has auto-matched voices)
    # Just resume the pipeline
    await resume_synthesis(job_id)

    return {"ok": True, "state": "synthesizing", "job_id": job_id}
