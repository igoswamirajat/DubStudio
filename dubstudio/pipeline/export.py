from __future__ import annotations

import json
import logging
from pathlib import Path

from dubstudio.pipeline.media import run_copy_source_name
from dubstudio.pipeline.qc import run_qc
from dubstudio.util.ffmpeg import remux_replace_audio
from dubstudio.util.srt import write_srt
from dubstudio.util.paths import rel_posix

log = logging.getLogger("dubstudio.export")


def run_export(job_dir: Path, job: dict) -> dict:
    export_dir = job_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    video = run_copy_source_name(job_dir)
    audio = job_dir / "mix" / "final.wav"
    if not audio.exists():
        raise FileNotFoundError("mix/final.wav missing")
    out_mp4 = export_dir / "output.mp4"
    remux_replace_audio(
        video, audio, out_mp4, language=(job.get("target_language") or None),
    )

    # Grade the finished job. The gate existed but nothing in the pipeline ever
    # called it, so a mix with audible holes still ended as "completed" with no
    # verdict anywhere. Advisory: a FAIL is recorded and logged, never raised -
    # a bad dub the user can inspect beats a job that throws the work away.
    gate: dict = {}
    try:
        gate = run_qc(job_dir)
    except Exception as exc:  # noqa: BLE001 - QC must never break an export
        log.warning("QC gate could not run: %s", exc)

    segs_path = job_dir / "segments" / "segments.json"
    segments = json.loads(segs_path.read_text(encoding="utf-8")) if segs_path.exists() else []
    out_srt = export_dir / "output.srt"
    write_srt(out_srt, segments)
    manifest = {
        "job_id": job["job_id"],
        "output_mp4": rel_posix(out_mp4, job_dir),
        "output_srt": rel_posix(out_srt, job_dir),
        "segment_count": len(segments),
        "target_language": job.get("target_language"),
        "qc_status": gate.get("status"),
        "qc_failures": gate.get("failures", []),
        "qc_warnings": gate.get("warnings", []),
    }
    (export_dir / "job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "output_mp4": rel_posix(out_mp4, job_dir),
        "output_srt": rel_posix(out_srt, job_dir),
        "manifest": manifest,
    }
