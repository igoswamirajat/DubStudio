from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def extract_wav(src: Path, dest: Path, sample_rate: int = 48000) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(dest),
    ]
    subprocess.check_call(cmd)


def video_duration_s(path: Path) -> float | None:
    """Duration of the first video stream in seconds, or None when unknown.

    The *video stream* duration, not the container's: they differ whenever the
    audio track is longer than the picture, and the picture is what the dub has
    to line up with.
    """
    try:
        meta = probe(path)
    except Exception:
        return None

    from_frames = None
    for st in meta.get("streams", []):
        if st.get("codec_type") != "video":
            continue
        try:
            dur = float(st.get("duration"))
            if dur > 0:
                return dur
        except (TypeError, ValueError):
            pass
        try:
            frames = float(st.get("nb_frames"))
            num, _, den = str(st.get("avg_frame_rate") or "").partition("/")
            rate = float(num) / float(den) if den else 0.0
            if frames > 0 and rate > 0:
                from_frames = frames / rate
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if from_frames:
        return from_frames
    try:
        dur = float(meta.get("format", {}).get("duration"))
        return dur if dur > 0 else None
    except (TypeError, ValueError):
        return None


def remux_replace_audio(
    video: Path,
    audio: Path,
    dest: Path,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    bitrate: str = "256k",
    language: str | None = None,
) -> None:
    """Replace a video's audio track without re-encoding a single video frame.

    The replacement audio is resampled, padded and trimmed to exactly the video
    stream's duration before muxing, so picture and sound are the same length by
    construction rather than by luck. Two measured problems this fixes (both on
    job_01M2JTG79S269HPHSZ5ETB64ED):

    * ``-shortest`` let whichever stream ended first decide the output length.
      The mix was 43.457s; the muxed audio came out 42.880s, so 0.58s of the
      mix was discarded with no warning anywhere. Had the last line of dialogue
      lived in that tail it would have been cut mid-word.
    * With no explicit ``-ar``/``-ac`` the AAC encoder picked its own rate and
      layout, and its priming samples shift the audio against the picture.
      Pinning ``first_pts=0`` stops a source offset becoming a lip-sync error.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = video_duration_s(video)

    # Resample first so the pad/trim arithmetic is in output samples. async=1
    # lets the resampler absorb drift instead of accumulating it; first_pts=0
    # pins the first sample to t=0.
    filters = [f"aresample={sample_rate}:async=1:first_pts=0"]
    if duration:
        filters.append("apad")
        filters.append(f"atrim=0:{duration:.6f}")
        filters.append("asetpts=N/SR/TB")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-i", str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", bitrate,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-af", ",".join(filters),
        "-movflags", "+faststart",
    ]
    if language:
        cmd += ["-metadata:s:a:0", f"language={language}"]
    cmd.append(str(dest))
    subprocess.check_call(cmd)
