from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from dubstudio.util.audio import duration_ms


def _atempo_chain(ratio: float) -> list[str]:
    factors: list[float] = []
    remaining = ratio
    while remaining < 0.5 or remaining > 2.0:
        if remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        else:
            factors.append(2.0)
            remaining /= 2.0
    factors.append(remaining)
    return factors


def fit_segment(src: Path, dest: Path, target_ms: int, *, max_stretch: float = 0.12) -> dict:
    gen_ms = duration_ms(src)
    if gen_ms <= 0:
        shutil.copy2(src, dest)
        return {"stretch_ratio": 1.0, "generated_duration_ms": gen_ms, "fitted_duration_ms": gen_ms}
    target_ms = max(1, target_ms)
    ratio = gen_ms / target_ms
    lo, hi = 1.0 - max_stretch, 1.0 + max_stretch
    if 0.92 <= ratio <= 1.08:
        shutil.copy2(src, dest)
        return {"stretch_ratio": 1.0, "generated_duration_ms": gen_ms, "fitted_duration_ms": gen_ms, "status": "ok"}
    applied = max(lo, min(hi, ratio))
    dest.parent.mkdir(parents=True, exist_ok=True)
    factors = _atempo_chain(applied)
    filt = ",".join(f"atempo={f:.6f}" for f in factors)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-filter:a", filt, str(dest)]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fitted = duration_ms(dest)
    return {"stretch_ratio": applied, "generated_duration_ms": gen_ms, "fitted_duration_ms": fitted, "status": "needs_stretch" if abs(1 - applied) > 0.01 else "ok"}


def run_timing(job_dir: Path) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    timing_dir = job_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        gen_rel = seg.get("generated_wav")
        if not gen_rel:
            continue
        src = job_dir / gen_rel
        out = timing_dir / f"{seg['segment_id']}.fitted.wav"
        info = fit_segment(src, out, int(seg.get("target_duration_ms") or 1000))
        seg["fitted_wav"] = str(out.relative_to(job_dir))
        seg["generated_duration_ms"] = info["generated_duration_ms"]
        seg["fitted_duration_ms"] = info["fitted_duration_ms"]
        seg["stretch_ratio"] = info["stretch_ratio"]
        seg["status"] = "ok"
        (timing_dir / f"{seg['segment_id']}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
