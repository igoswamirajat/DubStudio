"""The new stages have to run on their own, without the orchestrator having to
know about them.

Synthesis groups the cues if nobody did it yet, and the fitting stage hands a
blocked job to the block aligner instead of stretching cues one by one. That
keeps resumed jobs and old checkpoints working.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from dubstudio.engines.base import SynthResult
from dubstudio.pipeline import synthesis as S
from dubstudio.pipeline import timing as T

SR = 48000


class _FakeEngine:
    def __init__(self, name="veena", wps=2.6):
        self.name = name
        self.calls = []
        self._wps = wps

    def warmup(self):
        pass

    def unload(self):
        pass

    def generate(self, request, out_path):
        self.calls.append(request)
        words = max(1, len((request.text or "").split()))
        dur = words / self._wps
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(round(dur * 24000)), dtype=np.float32) / 24000
        sf.write(str(out_path), (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 24000)
        return SynthResult(wav_path=out_path, duration_ms=int(round(dur * 1000)),
                           sample_rate=24000, engine=self.name)


def _tone(dur_s, freq=220.0, amp=0.3, sr=SR):
    t = np.arange(int(round(dur_s * sr)), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _seg(i, start, end, text="one two", **extra):
    seg = {
        "segment_id": f"seg_{i:04d}",
        "speaker_id": "S00",
        "voice_id": "S00",
        "start": start,
        "end": end,
        "slot_start": start,
        "slot_end": end,
        "source_text": "ORIGINAL LANGUAGE LINE",
        "translated_text": text,
        "target_duration_ms": int(round((end - start) * 1000)),
        "status": "translated",
        "generated_wav": None,
        "fitted_wav": None,
    }
    seg.update(extra)
    return seg


def _write_segments(job_dir: Path, segs) -> Path:
    (job_dir / "segments").mkdir(parents=True, exist_ok=True)
    (job_dir / "segments" / "segments.json").write_text(json.dumps(segs), encoding="utf-8")
    return job_dir


def test_synthesis_groups_a_job_that_was_never_blocked(tmp_path, monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(S, "get_voice_engine", lambda name: engine)
    job_dir = _write_segments(tmp_path / "job", [
        _seg(0, 0.5, 2.4, "one two"),
        _seg(1, 2.6, 4.4, "three four"),
        _seg(2, 4.6, 6.4, "five six"),
    ])

    segs = S.run_synthesis(job_dir, {"target_language": "hi", "tts_engine": "dummy"})

    assert (job_dir / "segments" / "blocks.json").exists()
    blocks = json.loads((job_dir / "segments" / "blocks.json").read_text(encoding="utf-8"))
    assert len(blocks) == 1
    assert len(engine.calls) == 1
    assert engine.calls[0].text == "one two three four five six"
    assert segs[0]["generated_wav"] == "synth/blk_0000.wav"
    assert [s["status"] for s in segs] == ["synthesized", "in_block", "in_block"]


def test_a_real_beat_still_splits_the_job_into_two_calls(tmp_path, monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(S, "get_voice_engine", lambda name: engine)
    job_dir = _write_segments(tmp_path / "job", [
        _seg(0, 0.5, 2.4, "one two"),
        _seg(1, 6.0, 7.8, "three four"),
    ])

    S.run_synthesis(job_dir, {"target_language": "hi", "tts_engine": "dummy"})
    assert [c.text for c in engine.calls] == ["one two", "three four"]


def test_the_fitting_stage_hands_a_blocked_job_to_the_block_aligner(tmp_path):
    job_dir = tmp_path / "job"
    segs = [
        _seg(0, 0.5, 2.4, block_id="blk_0000", block_role="lead", block_index=0, block_size=2,
             generated_wav="synth/blk_0000.wav", status="synthesized"),
        _seg(1, 2.6, 4.4, block_id="blk_0000", block_role="member", block_index=1, block_size=2,
             status="in_block"),
    ]
    _write_segments(job_dir, segs)
    (job_dir / "segments" / "blocks.json").write_text(json.dumps([{
        "block_id": "blk_0000",
        "speaker_id": "S00",
        "segment_ids": ["seg_0000", "seg_0001"],
        "start": 0.5, "end": 4.4, "slot_start": 0.5, "slot_end": 4.4,
        "target_duration_ms": 3900,
        "text": "one two three four",
        "parts": [],
        "generated_wav": "synth/blk_0000.wav",
        "status": "synthesized",
    }]), encoding="utf-8")
    (job_dir / "synth").mkdir(parents=True, exist_ok=True)
    sf.write(str(job_dir / "synth" / "blk_0000.wav"), _tone(3.9), SR)

    out = T.run_timing(job_dir)

    assert out[0]["fitted_wav"] == "timing/blk_0000.fitted.wav"
    assert out[1]["fitted_wav"] is None and out[1]["status"] == "in_block"
    assert out[0]["stretch_ratio"] == out[1]["stretch_ratio"]
    assert (job_dir / "timing" / "blk_0000.fitted.wav").exists()
    assert not (job_dir / "timing" / "seg_0000.fitted.wav").exists()


def test_the_fitting_stage_is_unchanged_for_a_job_with_no_blocks(tmp_path):
    job_dir = tmp_path / "job"
    _write_segments(job_dir, [_seg(0, 0.5, 2.4, generated_wav="synth/seg_0000.wav",
                                   status="synthesized")])
    (job_dir / "segments" / "blocks.json").write_text("[]", encoding="utf-8")
    (job_dir / "synth").mkdir(parents=True, exist_ok=True)
    sf.write(str(job_dir / "synth" / "seg_0000.wav"), _tone(1.9), SR)

    out = T.run_timing(job_dir)
    assert out[0]["fitted_wav"] == "timing/seg_0000.fitted.wav"
