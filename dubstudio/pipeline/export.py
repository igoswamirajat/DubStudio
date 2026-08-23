from __future__ import annotations

import json
from pathlib import Path

from dubstudio.pipeline.media import run_copy_source_name
from dubstudio.util.ffmpeg import remux_replace_audio
from dubstudio.util.srt import write_srt


def run_export(job_dir: Path, job: dict) -> dict:
    export_dir = job_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    video = run_copy_source_name(job_dir)
    audio = job_dir / "mix" / "final.wav"
    if not audio.exists():
        raise FileNotFoundError("mix/final.wav missing")
    out_mp4 = export_dir / "output.mp4"
    remux_replace_audio(video, audio, out_mp4)
    segs_path = job_dir / "segments" / "segments.json"
    segments = json.loads(segs_path.read_text(encoding="utf-8")) if segs_path.exists() else []
    out_srt = export_dir / "output.srt"
    write_srt(out_srt, segments)
    manifest = {
        "job_id": job["job_id"],
        "output_mp4": str(out_mp4.relative_to(job_dir)),
        "output_srt": str(out_srt.relative_to(job_dir)),
        "segment_count": len(segments),
        "target_language": job.get("target_language"),
    }
    (export_dir / "job_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "output_mp4": str(out_mp4.relative_to(job_dir)),
        "output_srt": str(out_srt.relative_to(job_dir)),
        "manifest": manifest,
    }
