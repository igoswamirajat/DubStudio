"""Block plumbing end to end: one TTS call per block, and a block laid into
the mix as ONE continuous piece.

The old pipeline rendered and mixed every cue on its own, so a five second run
of speech was six separate takes pasted at six ASR timestamps. These tests pin
the two places that used to break it: synthesis (one call per run) and mixing
(one render unit per run, not clamped to the first cue's slot).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from dubstudio.engines.base import SynthResult
from dubstudio.pipeline import mixing as M
from dubstudio.pipeline import synthesis as S

SR = 48000
DUB_F, BED_F, ORIG_F = 400.0, 110.0, 1200.0


def _tone(dur_s: float, freq: float, amp: float, mod: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(dur_s * SR)), dtype=np.float32) / SR
    env = 1.0 if mod == 0.0 else (0.6 + 0.4 * np.sin(2 * np.pi * mod * t))
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _frames_rms(x: np.ndarray, sr: int = SR, win_s: float = 0.05) -> np.ndarray:
    w = int(round(win_s * sr))
    k = len(x) // w
    return np.array([float(np.sqrt(np.mean(x[i * w:(i + 1) * w] ** 2))) for i in range(k)])


def _at(env: np.ndarray, t_s: float, win_s: float = 0.05) -> float:
    return float(env[int(t_s / win_s)])


# --- render units -----------------------------------------------------------

def test_render_units_collapse_a_block_and_keep_lone_cues_apart():
    segs = [
        {"segment_id": "a", "start": 0.5, "end": 2.4, "block_id": "blk_0000", "fitted_wav": "timing/blk_0000.fitted.wav"},
        {"segment_id": "b", "start": 2.6, "end": 4.4, "block_id": "blk_0000"},
        {"segment_id": "c", "start": 4.6, "end": 6.4, "block_id": "blk_0000"},
        {"segment_id": "d", "start": 9.0, "end": 10.0, "block_id": None, "fitted_wav": "timing/d.fitted.wav"},
        {"segment_id": "e", "start": 10.2, "end": 11.0, "block_id": None, "fitted_wav": "timing/e.fitted.wav"},
    ]
    units = M._render_units(segs)
    assert [u["id"] for u in units] == ["blk_0000", "d", "e"]
    assert units[0]["segment_ids"] == ["a", "b", "c"]
    assert units[0]["start"] == 0.5 and units[0]["end"] == 6.4
    assert units[0]["wav"] == "timing/blk_0000.fitted.wav"
    assert units[0]["label"] == "Block blk_0000"
    assert units[1]["label"] == "Segment d"


def test_every_cue_of_an_unrendered_block_is_reported_failed():
    segs = [
        {"segment_id": "a", "start": 0.5, "end": 2.4, "block_id": "blk_0000"},
        {"segment_id": "b", "start": 2.6, "end": 4.4, "block_id": "blk_0000"},
    ]
    units = M._render_units(segs)
    assert len(units) == 1 and units[0]["wav"] is None


# --- mixing -----------------------------------------------------------------

def _mix_job(job_dir: Path, specs, duration_s: float, *, block=None, block_len=None):
    """specs: list of (segment_id, start, end, dub_len_s or None).

    block: block_id to put every cue in. block_len: length of the single block
    render, written to timing/<block_id>.fitted.wav and owned by the lead cue.
    """
    for sub in ("audio", "synth", "timing", "segments"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)
    n = int(round(duration_s * SR))

    t = np.arange(n, dtype=np.float32) / SR
    stereo = np.column_stack([
        0.18 * np.sin(2 * np.pi * BED_F * t),
        0.18 * np.sin(2 * np.pi * BED_F * t + 0.4),
    ]).astype(np.float32)
    sf.write(str(job_dir / "audio" / "bed.wav"), stereo, SR)

    voc = np.zeros(n, dtype=np.float32)
    for _sid, start, end, _dub in specs:
        chunk = _tone(end - start, ORIG_F, 0.40, mod=5.0)
        i0 = int(round(start * SR))
        voc[i0:i0 + len(chunk)] = chunk[:max(0, min(len(chunk), n - i0))]
    sf.write(str(job_dir / "audio" / "vocals.wav"), voc, SR)

    rows = []
    for idx, (sid, start, end, dub_len) in enumerate(specs):
        row = {
            "segment_id": sid,
            "start": start,
            "end": end,
            "speaker_id": "S00",
            "target_duration_ms": int(round((end - start) * 1000)),
            "block_id": block,
            "block_role": ("lead" if idx == 0 else "member") if block else "solo",
            "status": "synthesized",
        }
        if block:
            row["fitted_wav"] = f"timing/{block}.fitted.wav" if idx == 0 else None
            row["block_wav"] = f"timing/{block}.fitted.wav"
            if idx > 0:
                row["status"] = "in_block"
        elif dub_len:
            sf.write(str(job_dir / "timing" / f"{sid}.fitted.wav"), _tone(dub_len, DUB_F, 0.30), SR)
            row["fitted_wav"] = f"timing/{sid}.fitted.wav"
        else:
            row["status"] = "failed"
            row["error"] = "simulated TTS failure"
        rows.append(row)

    if block:
        sf.write(str(job_dir / "timing" / f"{block}.fitted.wav"), _tone(block_len, DUB_F, 0.30), SR)

    (job_dir / "segments" / "segments.json").write_text(json.dumps(rows), encoding="utf-8")
    return job_dir


_RUN = [("seg0", 0.50, 2.40, 1.90), ("seg1", 2.60, 4.40, 1.80), ("seg2", 4.60, 6.40, 1.80)]


def test_a_block_is_laid_down_as_one_continuous_piece(tmp_path):
    job_dir = _mix_job(tmp_path / "job", _RUN, 8.0, block="blk_0000", block_len=5.90)
    M.run_mixing(job_dir, 8.0)
    qc = json.loads((job_dir / "mix" / "qc.json").read_text(encoding="utf-8"))

    assert qc["failed_segments"] == []
    assert qc["segments_mixed"] == 1          # one render unit, not three
    assert qc["joins_welded"] == 0            # nothing to weld inside a block
    assert qc["coverage"] >= 0.99
    assert qc["holes"] == []                  # no uncovered speech inside the run

    dialogue, sr = sf.read(str(job_dir / "mix" / "dialogue.wav"))
    env = _frames_rms(np.asarray(dialogue, dtype=np.float32).reshape(-1), sr)
    inside = env[int(0.60 / 0.05):int(6.30 / 0.05)]
    assert float(inside.min()) > 0.02         # no silent frame between the old cues


def test_a_block_render_is_not_clamped_to_its_first_cue(tmp_path):
    """The regression. The chunk loop used to cut each render at the NEXT cue's
    ASR start, so a 5.9s block became 2.1s and the rest of the run went mute."""
    job_dir = _mix_job(tmp_path / "job", _RUN, 8.0, block="blk_0000", block_len=5.90)
    M.run_mixing(job_dir, 8.0)

    dialogue, sr = sf.read(str(job_dir / "mix" / "dialogue.wav"))
    env = _frames_rms(np.asarray(dialogue, dtype=np.float32).reshape(-1), sr)
    assert _at(env, 1.00) > 0.02
    assert _at(env, 3.50) > 0.02              # inside cue 2: was silent before
    assert _at(env, 5.50) > 0.02              # inside cue 3: was silent before
    assert _at(env, 7.00) < 0.01              # after the run, still silent


def test_cues_without_blocks_mix_exactly_as_before(tmp_path):
    job_dir = _mix_job(tmp_path / "job", _RUN, 8.0)
    M.run_mixing(job_dir, 8.0)
    qc = json.loads((job_dir / "mix" / "qc.json").read_text(encoding="utf-8"))
    assert qc["segments_mixed"] == 3
    assert qc["failed_segments"] == []


def test_a_lone_cue_with_no_audio_is_still_reported(tmp_path):
    specs = [("seg0", 0.50, 2.40, 1.90), ("seg1", 2.60, 4.40, None), ("seg2", 4.60, 6.40, 1.80)]
    job_dir = _mix_job(tmp_path / "job", specs, 8.0)
    M.run_mixing(job_dir, 8.0)
    qc = json.loads((job_dir / "mix" / "qc.json").read_text(encoding="utf-8"))
    assert qc["failed_segments"] == ["seg1"]
    assert qc["segments_mixed"] == 2


# --- synthesis --------------------------------------------------------------

class _FakeEngine:
    """Writes a tone whose length tracks the word count, like a real engine."""

    def __init__(self, name="veena", fail_on=None, wps=2.6):
        self.name = name
        self.calls = []
        self._fail_on = fail_on or (lambda req: False)
        self._wps = wps

    def warmup(self):
        pass

    def unload(self):
        pass

    def generate(self, request, out_path):
        self.calls.append(request)
        if self._fail_on(request):
            raise RuntimeError("engine blew up")
        words = max(1, len((request.text or "").split()))
        dur = words / self._wps
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(round(dur * 24000)), dtype=np.float32) / 24000
        sf.write(str(out_path), (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 24000)
        return SynthResult(wav_path=out_path, duration_ms=int(round(dur * 1000)),
                           sample_rate=24000, engine=self.name)


def _synth_job(job_dir: Path, *, texts, block_text=None):
    (job_dir / "segments").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(texts):
        rows.append({
            "segment_id": f"seg_{i:04d}",
            "speaker_id": "S00",
            "voice_id": "S00",
            "start": 0.5 + i * 2.0,
            "end": 2.3 + i * 2.0,
            "target_duration_ms": 1800,
            "source_text": "ORIGINAL LANGUAGE LINE",
            "translated_text": text,
            "status": "translated",
            "block_id": "blk_0000" if block_text else None,
            "block_index": i,
            "block_size": len(texts),
            "block_role": ("lead" if i == 0 else "member") if block_text else "solo",
        })
    (job_dir / "segments" / "segments.json").write_text(json.dumps(rows), encoding="utf-8")
    if block_text:
        block = {
            "block_id": "blk_0000",
            "speaker_id": "S00",
            "voice_id": "S00",
            "voice_mode": "clone",
            "segment_ids": [r["segment_id"] for r in rows],
            "start": rows[0]["start"],
            "end": rows[-1]["end"],
            "slot_start": rows[0]["start"],
            "slot_end": rows[-1]["end"],
            "target_duration_ms": int(round((rows[-1]["end"] - rows[0]["start"]) * 1000)),
            "text": block_text,
            "parts": [],
            "generated_wav": None,
            "status": "pending",
        }
        (job_dir / "segments" / "blocks.json").write_text(json.dumps([block]), encoding="utf-8")
    return job_dir


def test_a_whole_block_is_one_tts_call(tmp_path, monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(S, "get_voice_engine", lambda name: engine)
    job_dir = _synth_job(tmp_path / "job", texts=["one two", "three four", "five six"],
                         block_text="one two three four five six")

    segs = S.run_synthesis(job_dir, {"target_language": "hi", "tts_engine": "dummy"})

    assert len(engine.calls) == 1
    assert engine.calls[0].text == "one two three four five six"
    assert engine.calls[0].target_duration_ms == 5800
    assert segs[0]["generated_wav"] == "synth/blk_0000.wav"
    assert segs[0]["status"] == "synthesized"
    assert [s["status"] for s in segs[1:]] == ["in_block", "in_block"]
    assert all(s["generated_wav"] is None for s in segs[1:])
    assert all(s["block_wav"] == "synth/blk_0000.wav" for s in segs)
    assert (job_dir / "synth" / "blk_0000.wav").exists()

    blocks = json.loads((job_dir / "segments" / "blocks.json").read_text(encoding="utf-8"))
    assert blocks[0]["status"] == "synthesized"
    assert blocks[0]["generated_duration_ms"] > 0


def test_a_block_that_cannot_be_rendered_falls_back_to_cue_by_cue(tmp_path, monkeypatch):
    engine = _FakeEngine(fail_on=lambda req: len((req.text or "").split()) > 4)
    monkeypatch.setattr(S, "get_voice_engine", lambda name: engine)
    job_dir = _synth_job(tmp_path / "job", texts=["one two", "three four", "five six"],
                         block_text="one two three four five six")

    segs = S.run_synthesis(job_dir, {"target_language": "hi", "tts_engine": "dummy"})

    assert [s["status"] for s in segs] == ["synthesized"] * 3
    assert [s["generated_wav"] for s in segs] == [f"synth/seg_{i:04d}.wav" for i in range(3)]
    assert all(s["block_id"] is None and s["block_role"] == "solo" for s in segs)
    blocks = json.loads((job_dir / "segments" / "blocks.json").read_text(encoding="utf-8"))
    assert blocks[0]["status"] == "failed" and blocks[0]["generated_wav"] is None


def test_the_per_cue_path_still_renders_every_cue(tmp_path, monkeypatch):
    # The fallback path has to keep working on its own: one call per cue.
    engine = _FakeEngine()
    monkeypatch.setattr(S, "get_voice_engine", lambda name: engine)
    job_dir = _synth_job(tmp_path / "job", texts=["one two", "three four"])
    segs = json.loads((job_dir / "segments" / "segments.json").read_text(encoding="utf-8"))
    plan = S.build_voice_plan(segs, {}, "dummy")

    S._synthesize_segments(S._EnginePool(), job_dir, {}, segs, plan, "hi",
                           job_dir / "synth", 5, report_progress=False)

    assert len(engine.calls) == 2
    assert [s["generated_wav"] for s in segs] == ["synth/seg_0000.wav", "synth/seg_0001.wav"]


def test_an_untranslated_cue_is_never_dubbed_in_the_source_language(tmp_path, monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(S, "get_voice_engine", lambda name: engine)
    job_dir = _synth_job(tmp_path / "job", texts=["one two", ""])

    segs = S.run_synthesis(job_dir, {"target_language": "hi", "tts_engine": "dummy"})
    assert len(engine.calls) == 1
    assert "ORIGINAL" not in (engine.calls[0].text or "")
    assert segs[1]["status"] == "failed"
    assert segs[1]["error"] == "no translation to dub"
    assert segs[1]["generated_wav"] is None
