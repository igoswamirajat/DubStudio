from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dubstudio.util.audio import read_mono, write_wav


def run_mixing(job_dir: Path, duration_s: float) -> Path:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    sr = 48000
    n = max(1, int(duration_s * sr) + sr // 2)
    bed_path = job_dir / "audio" / "bed.wav"
    if bed_path.exists():
        bed, _ = read_mono(bed_path, target_sr=sr)
        bed = np.pad(bed, (0, max(0, n - len(bed))))[:n] if len(bed) < n else bed[:n]
    else:
        bed = np.zeros(n, dtype=np.float32)
    dialogue = np.zeros(n, dtype=np.float32)
    duck = np.ones(n, dtype=np.float32)
    for seg in segments:
        wav_rel = seg.get("fitted_wav") or seg.get("generated_wav")
        if not wav_rel:
            continue
        wav_path = job_dir / wav_rel
        if not wav_path.exists():
            continue
        audio, _ = read_mono(wav_path, target_sr=sr)
        start = float(seg["start"])
        i0 = int(start * sr)
        i1 = min(n, i0 + len(audio))
        if i0 >= n or i1 <= i0:
            continue
        dialogue[i0:i1] += audio[: i1 - i0]
        fade = int(0.05 * sr)
        duck[i0:i1] = 0.32
        if fade > 0:
            a0, a1 = max(0, i0 - fade), min(n, i1 + fade)
            if i0 > a0:
                duck[a0:i0] = np.linspace(1.0, 0.32, i0 - a0, dtype=np.float32)
            if a1 > i1:
                duck[i1:a1] = np.linspace(0.32, 1.0, a1 - i1, dtype=np.float32)
    mix = bed * duck + dialogue
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    mix_dir = job_dir / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)
    write_wav(mix_dir / "dialogue.wav", dialogue, sr)
    final_path = mix_dir / "final.wav"
    write_wav(final_path, mix, sr)
    return final_path
