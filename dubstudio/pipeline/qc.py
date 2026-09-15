"""Ship / no-ship gate for a finished mix.

`run_mixing` already measured itself into `mix/qc.json`, but nothing ever read
it: a job that lost a third of its speech still ended as "completed", and the
only way to find out was to watch the video. This module turns those numbers
(plus a fresh loudness / true-peak / bandwidth measurement of the rendered
files) into an explicit verdict in `mix/gate.json`:

    status: pass | warn | fail | skip
    checks: [{id, label, status, value, threshold, message}, ...]

`fail` means do not ship this, it has audible holes or it clips. `warn` means
shippable but something upstream is weak - most often a band-limited TTS voice
sitting under a full-bandwidth music bed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from dubstudio.pipeline import loudness as L
from dubstudio.util.audio import read_audio

log = logging.getLogger("dubstudio.qc")

MIN_COVERAGE = 0.95              # share of source speech that must be dubbed
MAX_LARGEST_HOLE_S = 1.0         # a gap this long reads as a dropout
WARN_LARGEST_HOLE_S = 0.35
TARGET_LUFS = L.MASTER_LUFS
LUFS_TOLERANCE_DB = 1.5
MAX_TRUE_PEAK_DBTP = L.MASTER_TRUE_PEAK_DBTP
TRUE_PEAK_SLACK_DB = 0.2
MIN_DIALOGUE_TOP_HZ = 12000.0    # below this the voice sounds muffled
MAX_BANDWIDTH_DEFICIT_HZ = 3000.0
MIN_DUCK_DEPTH_DB = 8.0

PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"
_RANK = {SKIP: 0, PASS: 1, WARN: 2, FAIL: 3}


def _num(data: dict, key: str):
    value = data.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add(checks: list, cid: str, label: str, status: str,
         value=None, threshold=None, message: str = "") -> None:
    checks.append({"id": cid, "label": label, "status": status,
                   "value": value, "threshold": threshold, "message": message})


def evaluate(qc: dict, measured: dict | None = None) -> dict:
    """Grade a qc.json payload. Missing numbers are skipped, never guessed."""
    data = dict(qc or {})
    data.update({k: v for k, v in (measured or {}).items() if v is not None})
    checks: list = []

    coverage = _num(data, "coverage")
    if coverage is None:
        _add(checks, "coverage", "Speech coverage", SKIP, message="not measured")
    else:
        _add(checks, "coverage", "Speech coverage",
             PASS if coverage >= MIN_COVERAGE else FAIL, round(coverage, 4), MIN_COVERAGE,
             "{:.1f}% of the source speech is dubbed".format(coverage * 100.0))

    holes_over_1s = _num(data, "holes_over_1s")
    if holes_over_1s is None:
        _add(checks, "holes_over_1s", "Dropouts over 1 s", SKIP, message="not measured")
    else:
        _add(checks, "holes_over_1s", "Dropouts over 1 s",
             PASS if holes_over_1s <= 0 else FAIL, int(holes_over_1s), 0,
             "{} gap(s) of a second or more".format(int(holes_over_1s)))

    largest = _num(data, "largest_hole_s")
    if largest is None:
        _add(checks, "largest_hole", "Largest gap", SKIP, message="not measured")
    else:
        status = PASS
        if largest > MAX_LARGEST_HOLE_S:
            status = FAIL
        elif largest > WARN_LARGEST_HOLE_S:
            status = WARN
        _add(checks, "largest_hole", "Largest gap", status, round(largest, 2),
             MAX_LARGEST_HOLE_S,
             "longest uncovered speech window is {:.2f}s".format(largest))

    failed = data.get("failed_segments")
    if failed is None:
        _add(checks, "failed_segments", "Undubbed lines", SKIP, message="not measured")
    else:
        count = len(failed) if isinstance(failed, (list, tuple)) else int(failed or 0)
        _add(checks, "failed_segments", "Undubbed lines",
             PASS if count == 0 else FAIL, count, 0,
             "{} line(s) have no dub audio".format(count))

    total = _num(data, "segments_total")
    mixed = _num(data, "segments_mixed")
    if total is None or mixed is None:
        _add(checks, "units_mixed", "Rendered units", SKIP, message="not measured")
    else:
        _add(checks, "units_mixed", "Rendered units",
             FAIL if total > 0 and mixed <= 0 else PASS, int(mixed), 1,
             "{} unit(s) mixed from {} cue(s)".format(int(mixed), int(total)))

    master_lufs = _num(data, "master_lufs")
    if master_lufs is None:
        _add(checks, "loudness", "Master loudness", SKIP, message="not measured")
    else:
        off = abs(master_lufs - TARGET_LUFS)
        _add(checks, "loudness", "Master loudness",
             PASS if off <= LUFS_TOLERANCE_DB else FAIL, round(master_lufs, 2), TARGET_LUFS,
             "{:.2f} LUFS ({:+.2f} LU vs target)".format(master_lufs, master_lufs - TARGET_LUFS))

    true_peak = _num(data, "true_peak_dbtp")
    if true_peak is None:
        _add(checks, "true_peak", "True peak", SKIP, message="not measured")
    else:
        _add(checks, "true_peak", "True peak",
             PASS if true_peak <= MAX_TRUE_PEAK_DBTP + TRUE_PEAK_SLACK_DB else FAIL,
             round(true_peak, 2), MAX_TRUE_PEAK_DBTP, "{:.2f} dBTP".format(true_peak))

    dia_top = _num(data, "dialogue_top_hz")
    if dia_top is None or dia_top <= 0.0:
        _add(checks, "dialogue_bandwidth", "Dialogue bandwidth", SKIP, message="not measured")
    else:
        _add(checks, "dialogue_bandwidth", "Dialogue bandwidth",
             PASS if dia_top >= MIN_DIALOGUE_TOP_HZ else WARN, round(dia_top, 1),
             MIN_DIALOGUE_TOP_HZ,
             "dub tops out at {:.0f} Hz; a 44.1 kHz voice would reach higher".format(dia_top))

        bed_top = _num(data, "bed_top_hz")
        if bed_top is None or bed_top <= 0.0:
            _add(checks, "bandwidth_match", "Dub vs bed bandwidth", SKIP, message="no bed")
        else:
            deficit = bed_top - dia_top
            _add(checks, "bandwidth_match", "Dub vs bed bandwidth",
                 PASS if deficit <= MAX_BANDWIDTH_DEFICIT_HZ else WARN, round(deficit, 1),
                 MAX_BANDWIDTH_DEFICIT_HZ, "bed reaches {:.0f} Hz above the dub".format(deficit))

    duck_depth = _num(data, "duck_depth_db")
    if duck_depth is None:
        _add(checks, "duck_depth", "Bed ducking", SKIP, message="not measured")
    else:
        _add(checks, "duck_depth", "Bed ducking",
             PASS if duck_depth <= -MIN_DUCK_DEPTH_DB else WARN, round(duck_depth, 2),
             -MIN_DUCK_DEPTH_DB,
             "bed sits {:.1f} dB under the dialogue".format(abs(duck_depth)))

    worst = max((_RANK[c["status"]] for c in checks), default=_RANK[SKIP])
    status = next(k for k, v in _RANK.items() if v == worst)
    return {
        # nothing measurable is not the same thing as nothing wrong
        "status": status,
        "checks": checks,
        "failures": [c["id"] for c in checks if c["status"] == FAIL],
        "warnings": [c["id"] for c in checks if c["status"] == WARN],
        "skipped": [c["id"] for c in checks if c["status"] == SKIP],
    }


def measure(job_dir: Path) -> dict:
    """Measure the rendered files instead of trusting what mixing reported."""
    job_dir = Path(job_dir)
    mix_dir = job_dir / "mix"
    out: dict = {}

    final = mix_dir / "final.wav"
    if final.exists():
        try:
            data, sr = read_audio(final, target_sr=48000, mono=False)
            out["master_lufs"] = round(L.lufs(data, sr), 2)
            out["true_peak_dbtp"] = round(L.true_peak_dbtp(data, sr), 2)
            out["sample_peak"] = round(float(np.max(np.abs(data))) if data.size else 0.0, 4)
        except Exception as exc:
            log.warning("could not measure %s: %s", final, exc)

    dialogue = mix_dir / "dialogue.wav"
    if dialogue.exists():
        try:
            data, sr = read_audio(dialogue, target_sr=48000, mono=True)
            out["dialogue_lufs"] = round(L.lufs(data, sr), 2)
            out["dialogue_top_hz"] = round(L.spectral_top_hz(data, sr), 1)
        except Exception as exc:
            log.warning("could not measure %s: %s", dialogue, exc)

    bed = job_dir / "audio" / "bed.wav"
    if bed.exists():
        try:
            data, sr = read_audio(bed, target_sr=48000, mono=True)
            out["bed_top_hz"] = round(L.spectral_top_hz(data, sr), 1)
        except Exception as exc:
            log.warning("could not measure %s: %s", bed, exc)

    return out


def run_qc(job_dir: Path, *, write: bool = True) -> dict:
    """Grade a finished job and (by default) write mix/gate.json."""
    job_dir = Path(job_dir)
    mix_dir = job_dir / "mix"
    qc_path = mix_dir / "qc.json"
    qc: dict = {}
    if qc_path.exists():
        try:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("unreadable qc.json: %s", exc)

    measured = measure(job_dir)
    gate = evaluate(qc, measured)
    gate["measured"] = measured

    if write:
        mix_dir.mkdir(parents=True, exist_ok=True)
        (mix_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    if gate["status"] == FAIL:
        log.error("QC gate FAILED: %s", ", ".join(
            "{}={}".format(c["id"], c["value"]) for c in gate["checks"] if c["status"] == FAIL))
    elif gate["warnings"]:
        log.warning("QC gate warnings: %s", ", ".join(gate["warnings"]))
    return gate
