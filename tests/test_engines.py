"""VoiceEngine factory + contract tests."""

from __future__ import annotations

from pathlib import Path

from dubstudio.engines.base import SynthRequest
from dubstudio.engines.dummy import DummyEngine
from dubstudio.engines.factory import get_voice_engine


def test_default_engine_falls_back_to_dummy():
    eng = get_voice_engine()
    assert eng is not None
    assert eng.name in {"dummy", "omnivoice", "chatterbox"}


def test_explicit_dummy():
    eng = get_voice_engine("dummy")
    assert isinstance(eng, DummyEngine)
    assert eng.name == "dummy"


def test_dummy_generate_duration_tracks_text(tmp_path: Path):
    eng = DummyEngine()
    eng.warmup()
    out = tmp_path / "t.wav"
    short = eng.generate(
        SynthRequest(text="hi", language="en", voice_id="S00", ref_wav=None),
        out,
    )
    out2 = tmp_path / "t2.wav"
    long = eng.generate(
        SynthRequest(
            text="this is a much longer line for duration check",
            language="en",
            voice_id="S00",
            ref_wav=None,
        ),
        out2,
    )
    assert short.duration_ms > 0
    assert long.duration_ms > short.duration_ms
    assert out.exists() and out2.exists()


def test_omnivoice_request_falls_back_cleanly():
    eng = get_voice_engine("omnivoice")
    assert eng is not None
    assert eng.name in {"omnivoice", "dummy"}


def test_veena_request_resolves_or_falls_back():
    eng = get_voice_engine("veena")
    assert eng is not None
    assert eng.name in {"veena", "dummy"}
    if eng.name == "veena":
        # Check voice resolver
        from dubstudio.engines.veena_engine import VeenaEngine
        v_eng = VeenaEngine()
        assert v_eng._resolve_voice(SynthRequest(text="hi", language="hi", voice_id="S00", ref_wav=None)) == "kavya"
        assert v_eng._resolve_voice(SynthRequest(text="hi", language="hi", voice_id="S01", ref_wav=None)) == "agastya"
        assert v_eng._resolve_voice(SynthRequest(text="hi", language="hi", voice_id="custom", instruct="male narrator", ref_wav=None)) == "agastya"


def test_unknown_engine_falls_back_to_dummy():
    eng = get_voice_engine("not_a_real_engine_xyz")
    assert eng.name == "dummy"

