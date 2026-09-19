"""Job-artifact paths as they appear in JSON.

Artifact locations are written into job JSON (segments.json, blocks.json,
speaker_map.json, the export manifest) and read back later - sometimes after the
job directory has been moved, and sometimes by the browser. They are stored with
forward slashes so the payload is portable: ``pathlib`` accepts ``/`` on Windows,
so reads are unaffected, whereas a Windows-style ``\\`` path is meaningless once
the file is read anywhere else and is awkward to concatenate into a URL.

Tests that assert on these strings should compare against ``as_posix()`` output
rather than hard-coding a separator.
"""
from __future__ import annotations

from pathlib import Path


def rel_posix(path: Path | str, base: Path | str) -> str:
    """Return `path` relative to `base`, always using forward slashes."""
    return Path(path).relative_to(Path(base)).as_posix()
