"""Speaker map + voice_mode contract (no GPU / no OmniVoice required)."""

from __future__ import annotations

import json
from pathlib import Path

from dubstudio.engines.base import SynthRequest


def test_synth_request_phase2_fields():
    req = SynthRequest(
        text="namaste",
        language="hi",
        voice_id="S00",
        ref_wav=None,
        voice_mode="design",
        instruct="female, young adult, hindi accent",
        ref_text="hello",
    )
    assert req.voice_mode == "design"
    assert req.instruct is not None
    assert req.ref_text == "hello"


def test_speaker_map_roundtrip(tmp_path: Path):
    voices = tmp_path / "voices"
    voices.mkdir()
    payload = {
        "speakers": [
            {
                "speaker_id": "S00",
                "label": "Hero",
                "voice_id": "S00",
                "voice_mode": "clone",
                "design_prompt": None,
                "ref_text": "ye line original hai",
                "ref_wav": "voices/S00/ref.wav",
                "segment_count": 2,
            }
        ]
    }
    path = voices / "speaker_map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["speakers"][0]["voice_mode"] == "clone"

    # simulate PATCH design mode
    loaded["speakers"][0]["voice_mode"] = "design"
    loaded["speakers"][0]["design_prompt"] = "male, deep, hindi"
    path.write_text(json.dumps(loaded), encoding="utf-8")
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["speakers"][0]["voice_mode"] == "design"
    assert "hindi" in again["speakers"][0]["design_prompt"]
