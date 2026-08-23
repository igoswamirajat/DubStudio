from dubstudio.pipeline.timing import fit_segment
from dubstudio.util.audio import duration_ms, write_tone


def test_fit_accepts_close_duration(tmp_path):
    src = tmp_path / "a.wav"
    write_tone(src, 1.0)
    dest = tmp_path / "b.wav"
    info = fit_segment(src, dest, target_ms=1000)
    assert dest.exists()
    assert info["stretch_ratio"] == 1.0


def test_fit_stretches_long_audio(tmp_path):
    src = tmp_path / "long.wav"
    write_tone(src, 1.5)
    dest = tmp_path / "fit.wav"
    info = fit_segment(src, dest, target_ms=1000, max_stretch=0.12)
    assert dest.exists()
    assert duration_ms(dest) < duration_ms(src)
    assert 0.88 <= info["stretch_ratio"] <= 1.12
