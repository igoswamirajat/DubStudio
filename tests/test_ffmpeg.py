from dubstudio.util.ffmpeg import extract_wav, ffmpeg_ok, probe, remux_replace_audio


def test_ffmpeg_present():
    assert ffmpeg_ok()


def test_extract_and_remux(clip_8s, tmp_path):
    meta = probe(clip_8s)
    assert float(meta["format"]["duration"]) >= 7.5
    wav = tmp_path / "full.wav"
    extract_wav(clip_8s, wav)
    assert wav.exists() and wav.stat().st_size > 1000
    out = tmp_path / "out.mp4"
    remux_replace_audio(clip_8s, wav, out)
    assert out.exists() and out.stat().st_size > 1000
    meta2 = probe(out)
    assert float(meta2["format"]["duration"]) >= 7.0
