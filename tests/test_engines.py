"""VoiceEngine factory + contract tests."""

from __future__ import annotations

from pathlib import Path

from dubstudio.engines.base import SynthRequest
from dubstudio.engines.dummy import DummyEngine
from dubstudio.engines.factory import get_voice_engine


def test_default_engine_falls_back_to_dummy():
    eng = get_voice_engine()
    assert eng is not None
    assert eng.name in {"dummy", "omnivoice", "chatterbox", "veena"}


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


def test_veena_male_pitch_anchoring():
    import pytest
    try:
        import torch
        import torchaudio
    except (ImportError, OSError) as exc:
        pytest.skip(f"Torch/CUDA unavailable or memory limit: {exc}")

    import numpy as np
    from dubstudio.engines.veena_engine import VeenaEngine, VEENA_SR

    v_eng = VeenaEngine()
    # Create a synthetic 220 Hz sine tone (typical female register)
    duration = 0.5
    t = np.linspace(0, duration, int(VEENA_SR * duration), endpoint=False)
    high_pitch_audio = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    # When speaker is kavya (female), high pitch should remain untouched
    untouched = v_eng._anchor_pitch(high_pitch_audio, speaker="kavya", sr=VEENA_SR)
    assert np.array_equal(untouched, high_pitch_audio)


def test_unknown_engine_falls_back_to_dummy():
    eng = get_voice_engine("not_a_real_engine_xyz")
    assert eng.name == "dummy"


# --- OmniVoice pacing -------------------------------------------------------
# Regression for job_01M2JTG79S269HPHSZ5ETB64ED, where OmniVoice ignored
# target_duration_ms, spoke Hindi at 4.34 w/s against a 2.6 w/s budget, and left
# 5.6s of dead air in the dub.

def test_pacing_uses_slot_length_when_text_can_fill_it():
    from dubstudio.engines.omnivoice_engine import _pacing_duration_s

    # 60 Hindi words in a 20.64s slot: the engine's natural floor is
    # 60/1.8 = 33s, so the slot is the binding request.
    text = " ".join(["शब्द"] * 60)
    req = SynthRequest(text=text, language="hi", voice_id="S00", ref_wav=None,
                       target_duration_ms=20640)
    assert _pacing_duration_s(req) == 20.64


def test_pacing_never_asks_for_a_drawl():
    from dubstudio.engines.omnivoice_engine import _pacing_duration_s

    # Five words cannot plausibly fill a 20s slot; ask for a natural length
    # instead of stretching them out.
    req = SynthRequest(text=" ".join(["शब्द"] * 5), language="hi", voice_id="S00",
                       ref_wav=None, target_duration_ms=20000)
    pacing = _pacing_duration_s(req)
    assert pacing is not None
    assert pacing < 5.0


def test_pacing_absent_without_a_target():
    from dubstudio.engines.omnivoice_engine import _pacing_duration_s

    req = SynthRequest(text="कुछ शब्द", language="hi", voice_id="S00", ref_wav=None)
    assert _pacing_duration_s(req) is None

    req_zero = SynthRequest(text="कुछ शब्द", language="hi", voice_id="S00", ref_wav=None,
                            target_duration_ms=0)
    assert _pacing_duration_s(req_zero) is None


def test_pacing_ignores_degenerate_slots():
    from dubstudio.engines.omnivoice_engine import _pacing_duration_s

    req = SynthRequest(text="कुछ शब्द यहाँ", language="hi", voice_id="S00", ref_wav=None,
                       target_duration_ms=120)
    assert _pacing_duration_s(req) is None


def test_generate_passes_duration_to_the_model(monkeypatch, tmp_path: Path):
    """The engine must actually forward the pacing request to OmniVoice."""
    from dubstudio.engines import omnivoice_engine as mod

    captured: dict = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            import numpy as np
            return [np.zeros(24000, dtype=np.float32)]

    eng = mod.OmniVoiceEngine()
    eng._model = FakeModel()
    out = tmp_path / "paced.wav"
    req = SynthRequest(text=" ".join(["शब्द"] * 60), language="hi", voice_id="S00",
                       ref_wav=None, target_duration_ms=20640)
    eng.generate(req, out)
    assert captured.get("duration") == 20.64
    assert captured.get("language") == "hi"


def test_veena_snac_deinterleaving():
    import pytest
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"Torch unavailable or memory limit: {exc}")
    from dubstudio.engines.veena_engine import AUDIO_CODE_BASE_OFFSET, _decode_snac_tokens

    class MockSnac(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))
            self.hierarchical_codes = None

        def decode(self, codes):
            self.hierarchical_codes = codes
            return torch.zeros(1, 1, 100)

    # Prepare 7 tokens: indices 0..6 mapped to values 10, 20, 30, 40, 50, 60, 70
    offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]
    val_map = [10, 20, 30, 40, 50, 60, 70]
    tokens = [offsets[i] + val_map[i] for i in range(7)]

    mock_snac = MockSnac()
    result = _decode_snac_tokens(tokens, mock_snac)
    assert result is not None
    assert len(result) == 100

    codes = mock_snac.hierarchical_codes
    assert len(codes) == 3
    # Level 0: coarse (index 0 -> val_map[0]=10)
    assert codes[0].shape == (1, 1)
    assert codes[0][0, 0].item() == 10

    # Level 1: medium (index 1 -> val_map[1]=20, index 4 -> val_map[4]=50)
    assert codes[1].shape == (1, 2)
    assert codes[1][0, 0].item() == 20
    assert codes[1][0, 1].item() == 50

    # Level 2: fine (indices 2,3,5,6 -> val_map[2]=30, val_map[3]=40, val_map[5]=60, val_map[6]=70)
    assert codes[2].shape == (1, 4)
    assert codes[2][0, 0].item() == 30
    assert codes[2][0, 1].item() == 40
    assert codes[2][0, 2].item() == 60
    assert codes[2][0, 3].item() == 70


