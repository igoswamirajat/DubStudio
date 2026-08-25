"""Phase 7 — job checkpointing and resume.

Each completed stage writes a marker under job_dir/checkpoints/{stage}.done
plus stage artifact paths. run_job can skip stages already done.
"""

from __future__ import annotations

import json
from pathlib import Path

STAGE_ORDER = [
    "ingesting",
    "extracting",
    "separating",
    "transcribing",
    "diarizing",
    "segmenting",
    "translating",
    "enrolling_voices",
    "synthesizing",
    "fitting",
    "mixing",
    "exporting",
]


def ckpt_dir(job_dir: Path) -> Path:
    d = job_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mark_done(job_dir: Path, stage: str, payload: dict | None = None) -> None:
    d = ckpt_dir(job_dir)
    (d / f"{stage}.done").write_text("ok", encoding="utf-8")
    if payload is not None:
        (d / f"{stage}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def is_done(job_dir: Path, stage: str) -> bool:
    return (ckpt_dir(job_dir) / f"{stage}.done").exists()


def last_completed(job_dir: Path) -> str | None:
    done = None
    for s in STAGE_ORDER:
        if is_done(job_dir, s):
            done = s
        else:
            break
    return done


def clear_from(job_dir: Path, stage: str) -> None:
    """Clear stage and all after it (for forced re-run)."""
    if stage not in STAGE_ORDER:
        return
    idx = STAGE_ORDER.index(stage)
    d = ckpt_dir(job_dir)
    for s in STAGE_ORDER[idx:]:
        for p in (d / f"{s}.done", d / f"{s}.json"):
            if p.exists():
                p.unlink()
