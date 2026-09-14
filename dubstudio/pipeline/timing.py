from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from dubstudio.util.audio import duration_ms

DEFAULT_MAX_STRETCH = 0.25


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


def fit_segment(src: Path, dest: Path, target_ms: int, *, max_stretch: float = DEFAULT_MAX_STRETCH) -> dict:
    gen_ms = duration_ms(src)
    if gen_ms <= 0:
        shutil.copy2(src, dest)
        return {"stretch_ratio": 1.0, "generated_duration_ms": gen_ms, "fitted_duration_ms": gen_ms}
    target_ms = max(1, target_ms)

    work_src = src
    temp_trimmed = None

    if gen_ms > target_ms * 1.05:
        try:
            import numpy as np
            from dubstudio.util.audio import read_audio, write_wav

            data, sr = read_audio(src, target_sr=48000, mono=True)
            thresh = 10 ** (-42.0 / 20.0)
            non_silent = np.where(np.abs(data) > thresh)[0]
            if len(non_silent) > 0:
                pad = int(sr * 0.05)
                start = max(0, non_silent[0] - pad)
                end = min(len(data), non_silent[-1] + pad)
                trimmed = data[start:end]
                trimmed_ms = int(round(len(trimmed) / sr * 1000))
                if trimmed_ms >= target_ms * 0.9 and trimmed_ms < gen_ms:
                    temp_trimmed = dest.parent / f"{dest.stem}.trimmed.wav"
                    temp_trimmed.parent.mkdir(parents=True, exist_ok=True)
                    write_wav(temp_trimmed, trimmed, sr)
                    work_src = temp_trimmed
                    gen_ms = trimmed_ms
        except Exception:
            pass

    ratio = gen_ms / target_ms
    lo, hi = 1.0 - max_stretch, 1.0 + max_stretch

    if 0.95 <= ratio <= 1.05:
        shutil.copy2(work_src, dest)
        if temp_trimmed and temp_trimmed.exists():
            temp_trimmed.unlink(missing_ok=True)
        return {"stretch_ratio": 1.0, "generated_duration_ms": gen_ms, "fitted_duration_ms": gen_ms,
                "overflow_ms": 0, "status": "ok"}

    applied = max(lo, min(hi, ratio))
    dest.parent.mkdir(parents=True, exist_ok=True)
    factors = _atempo_chain(applied)
    filt = ",".join(f"atempo={f:.6f}" for f in factors)
    cmd = ["ffmpeg", "-y", "-i", str(work_src), "-filter:a", filt, str(dest)]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if temp_trimmed and temp_trimmed.exists():
        temp_trimmed.unlink(missing_ok=True)

    fitted = duration_ms(dest)
    try:
        from dubstudio.util.audio import apply_micro_fades, read_audio, write_wav

        data, sr = read_audio(dest, target_sr=48000, mono=True)
        data = apply_micro_fades(data, fade_ms=5.0, sample_rate=sr)
        write_wav(dest, data, sr)
        fitted = duration_ms(dest)
    except Exception:
        pass

    overflow_ms = max(0, fitted - target_ms)
    status = "ok" if abs(1 - applied) <= 0.01 else "needs_stretch"
    if overflow_ms > 250:
        status = "overflow"
    return {"stretch_ratio": applied, "generated_duration_ms": gen_ms, "fitted_duration_ms": fitted,
            "overflow_ms": overflow_ms, "status": status}


def run_timing(job_dir: Path, job: dict | None = None, *, max_stretch: float = DEFAULT_MAX_STRETCH) -> list[dict]:
    from dubstudio.pipeline.blocks import load_blocks

    if load_blocks(job_dir):
        # This job was synthesised as whole blocks, so it has to be fitted as
        # whole blocks: one tempo ratio per run of speech instead of one per
        # cue, which is what used to make the speaking rate jump mid-sentence.
        from dubstudio.pipeline.align import run_align

        run_align(job_dir, job)
        return json.loads((job_dir / "segments" / "segments.json").read_text(encoding="utf-8"))

    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    timing_dir = job_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    total = len(segments)
    overflows: list[str] = []
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
        info = fit_segment(src, out, int(seg.get("target_duration_ms") or 1000), max_stretch=max_stretch)
        seg["fitted_wav"] = str(out.relative_to(job_dir))
        seg["generated_duration_ms"] = info["generated_duration_ms"]
        seg["fitted_duration_ms"] = info["fitted_duration_ms"]
        seg["stretch_ratio"] = info["stretch_ratio"]
        seg["overflow_ms"] = info.get("overflow_ms", 0)
        seg["status"] = "ok"
        if info.get("status") == "overflow":
            overflows.append(seg["segment_id"])
        (timing_dir / f"{seg['segment_id']}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if overflows and job is not None:
        job["timing_overflows"] = overflows
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
