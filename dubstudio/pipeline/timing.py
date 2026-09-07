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


def fit_segment(src: Path, dest: Path, target_ms: int, *, max_stretch: float = 0.30) -> dict:
    gen_ms = duration_ms(src)
    if gen_ms <= 0:
        shutil.copy2(src, dest)
        return {"stretch_ratio": 1.0, "generated_duration_ms": gen_ms, "fitted_duration_ms": gen_ms}
    target_ms = max(1, target_ms)

    work_src = src
    temp_trimmed = None

    # If audio is longer than target, trim leading & trailing silence before applying any stretching
    if gen_ms > target_ms * 1.05:
        try:
            import numpy as np
            from dubstudio.util.audio import read_audio, write_wav

            data, sr = read_audio(src, target_sr=48000, mono=True)
            thresh = 10 ** (-42.0 / 20.0)  # -42 dB noise gate
            non_silent = np.where(np.abs(data) > thresh)[0]
            if len(non_silent) > 0:
                pad_start = int(sr * 0.05)  # 50ms lead-in padding
                pad_end = int(sr * 0.15)    # 150ms trailing padding preserves breath & decaying vowels
                start = max(0, non_silent[0] - pad_start)
                end = min(len(data), non_silent[-1] + pad_end)
                trimmed = data[start:end]
                trimmed_ms = int(round(len(trimmed) / sr * 1000))
                # Only use trimmed version if it trimmed dead air without cutting the line
                if trimmed_ms >= target_ms * 0.85 and trimmed_ms < gen_ms:
                    temp_trimmed = dest.parent / f"{dest.stem}.trimmed.wav"
                    temp_trimmed.parent.mkdir(parents=True, exist_ok=True)
                    write_wav(temp_trimmed, trimmed, sr)
                    work_src = temp_trimmed
                    gen_ms = trimmed_ms
        except Exception:
            pass

    ratio = gen_ms / target_ms
    lo, hi = 1.0 - max_stretch, 1.0 + max_stretch

    # If within 5%, leave natural rhythm untouched (no pitch/tempo artifacts)
    if 0.95 <= ratio <= 1.05:
        applied = 1.0
        shutil.copy2(work_src, dest)
    else:
        applied = max(lo, min(hi, ratio))
        dest.parent.mkdir(parents=True, exist_ok=True)
        factors = _atempo_chain(applied)
        filt = ",".join(f"atempo={f:.6f}" for f in factors)
        cmd = ["ffmpeg", "-y", "-i", str(work_src), "-filter:a", filt, str(dest)]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if temp_trimmed and temp_trimmed.exists():
        temp_trimmed.unlink(missing_ok=True)

    # Always apply smooth micro-fades and ensure audio doesn't overshoot target timeline slot
    fitted = duration_ms(dest)
    try:
        from dubstudio.util.audio import apply_micro_fades, read_audio, write_wav

        data, sr = read_audio(dest, target_sr=48000, mono=True)
        max_samples = int(round((target_ms + 40) / 1000.0 * sr))
        if len(data) > max_samples:
            data = data[:max_samples]
        data = apply_micro_fades(data, fade_ms=10.0, sample_rate=sr)
        write_wav(dest, data, sr)
        fitted = duration_ms(dest)
    except Exception:
        pass

    return {
        "stretch_ratio": applied,
        "generated_duration_ms": gen_ms,
        "fitted_duration_ms": fitted,
        "status": "needs_stretch" if abs(1 - applied) > 0.01 else "ok",
    }


def run_timing(job_dir: Path, job: dict | None = None) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    timing_dir = job_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    total = len(segments)
    for i, seg in enumerate(segments):
        if job and job.get("job_id") and total > 0:
            try:
                from dubstudio.jobs.store import store

                pct = 85 + int((i / total) * 7)
                job["percent"] = min(pct, 91)
                job["message"] = f"Fitting audio timing {i + 1}/{total}"
                store.save(job)
            except Exception:
                pass
        gen_rel = seg.get("generated_wav")
        if not gen_rel:
            continue
        src = job_dir / gen_rel
        out = timing_dir / f"{seg['segment_id']}.fitted.wav"
        info = fit_segment(src, out, int(seg.get("target_duration_ms") or 1000), max_stretch=0.30)
        seg["fitted_wav"] = str(out.relative_to(job_dir))
        seg["generated_duration_ms"] = info["generated_duration_ms"]
        seg["fitted_duration_ms"] = info["fitted_duration_ms"]
        seg["stretch_ratio"] = info["stretch_ratio"]
        seg["status"] = "ok"
        (timing_dir / f"{seg['segment_id']}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
