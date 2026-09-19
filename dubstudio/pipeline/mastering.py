"""Broadcast-level finish for a rendered job, plus the QC verdict.

`run_mixing` renders and balances, but nothing normalised the result: one job
came out at -24 LUFS, another clipped between samples. This is the last stop
before a file is shipped, and it is deliberately independent of mixing so it
can also be run over a job that is already on disk:

    python -m dubstudio.pipeline.mastering data/jobs/<job_id>

The untouched render is kept as mix/final.raw.wav the first time, so mastering
is always reversible and never compounds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from dubstudio.pipeline import loudness as L
from dubstudio.pipeline import qc as Q
from dubstudio.util.audio import read_audio, write_wav

log = logging.getLogger("dubstudio.mastering")

RAW_NAME = "final.raw.wav"


def master_job(job_dir, *, target_lufs: float = L.MASTER_LUFS, run_gate: bool = True) -> dict:
    """Normalise mix/final.wav, then (optionally) grade the job.

    `run_gate=False` is for the pipeline, where export grades the job again once
    the mp4 exists - that second pass is the one that can see the A/V streams.
    """
    job_dir = Path(job_dir)
    mix_dir = job_dir / "mix"
    final = mix_dir / "final.wav"
    if not final.exists():
        log.warning("nothing to master: %s is missing", final)
        return {}

    audio, sr = read_audio(final, target_sr=48000, mono=False)
    raw = mix_dir / RAW_NAME
    if not raw.exists():
        write_wav(raw, np.asarray(audio, dtype=np.float32), sr)

    mastered, info = L.master(audio, sr, target_lufs=target_lufs)
    write_wav(final, np.asarray(mastered, dtype=np.float32), sr)
    log.info("mastered %s: %+.2f dB -> %.2f LUFS, %.2f dBTP",
             final.name, info["gain_db"], info["output_lufs"], info["true_peak_dbtp"])

    qc: dict = {}
    qc_path = mix_dir / "qc.json"
    if qc_path.exists():
        try:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("unreadable qc.json: %s", exc)

    qc.update({
        "master_lufs": info["output_lufs"],
        "master_gain_db": info["gain_db"],
        "true_peak_dbtp": info["true_peak_dbtp"],
        "master_limited": bool(info["limited"]),
    })
    measured = Q.measure(job_dir)
    for key in ("dialogue_lufs", "dialogue_top_hz", "bed_top_hz"):
        if key in measured:
            qc[key] = measured[key]
    mix_dir.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    return {"master": info, "gate": Q.run_qc(job_dir) if run_gate else {}}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Master a rendered job and print the QC verdict.")
    parser.add_argument("job_dir")
    parser.add_argument("--target-lufs", type=float, default=L.MASTER_LUFS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = master_job(args.job_dir, target_lufs=args.target_lufs)
    if not result:
        print("no mix/final.wav in {}".format(args.job_dir))
        return 1

    info, gate = result["master"], result["gate"]
    print("loudness {:+.2f} dB -> {:.2f} LUFS, true peak {:.2f} dBTP".format(
        info["gain_db"], info["output_lufs"], info["true_peak_dbtp"]))
    print("gate: {}".format(gate["status"].upper()))
    for check in gate["checks"]:
        if check["status"] in (Q.FAIL, Q.WARN):
            print("  [{}] {}: {}".format(check["status"].upper(), check["label"], check["message"]))
    return 1 if gate["status"] == Q.FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
