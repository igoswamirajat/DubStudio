from __future__ import annotations

import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any

from dubstudio.engines.base import SynthRequest
from dubstudio.engines.factory import get_voice_engine
from dubstudio.jobs.store import store
from dubstudio.pipeline.blocks import blocks_path, load_blocks, run_blocks

log = logging.getLogger("dubstudio.synthesis")

SEGMENT_TIMEOUT_S = 120
ATTEMPTS_PER_SEGMENT = 2          # retries on the speaker's OWN engine first
FALLBACK_ENGINE = "veena"
FALLBACK_VOICE_MODE = "standard"
# A block is a whole run of speech, so it needs proportionally more time than a
# single cue before we are allowed to call it a timeout.
BLOCK_TIMEOUT_PER_S = 6.0


def _speaker_lookup(job_dir: Path) -> dict[str, dict]:
    path = job_dir / "voices" / "speaker_map.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["speaker_id"]: s for s in data.get("speakers", []) if s.get("speaker_id")}


def _safe_save_job(job: dict) -> None:
    if not job or not job.get("job_id"):
        return
    try:
        store.save(job)
    except Exception:
        pass


class _EnginePool:
    """Lazily instantiates engines and keeps one instance per engine name."""

    def __init__(self) -> None:
        self._engines: dict[str, Any] = {}

    def get(self, name: str):
        if name not in self._engines:
            self._engines[name] = get_voice_engine(name)
        return self._engines[name]


def _preferred_engine(default_engine_name: str, voice_mode: str) -> str:
    if default_engine_name == "dummy":
        return "dummy"
    if voice_mode in {"clone", "design"}:
        return "omnivoice"
    return "veena"


def build_voice_plan(segments: list[dict], speakers: dict[str, dict], default_engine_name: str) -> dict[str, dict]:
    """Lock one engine + voice per speaker for the WHOLE job.

    A speaker's voice identity must never change part-way through a video, so
    the engine is decided once, up front, instead of being flipped globally by
    the first segment that happens to time out.
    """
    plan: dict[str, dict] = {}
    for seg in segments:
        sid = seg.get("speaker_id") or "S00"
        if sid in plan:
            continue
        sp = speakers.get(sid) or {}
        voice_mode = seg.get("voice_mode") or sp.get("voice_mode") or "clone"
        plan[sid] = {
            "engine": _preferred_engine(default_engine_name, voice_mode),
            "voice_mode": voice_mode,
            "voice_id": seg.get("voice_id") or sp.get("voice_id") or sid,
            "design_prompt": seg.get("design_prompt") or sp.get("design_prompt"),
            "ref_text": sp.get("ref_text"),
            "downgraded": False,
        }
    return plan


def _ref_wav(job_dir: Path, sid: str, voice_id: str) -> Path | None:
    for candidate in (job_dir / "voices" / voice_id / "ref.wav", job_dir / "voices" / sid / "ref.wav"):
        if candidate.exists():
            return candidate
    return None


def _generate_once(engine, seg: dict, entry: dict, lang: str, ref: Path | None, out: Path, timeout_s: int):
    request = SynthRequest(
        # Translated text ONLY. Falling back to the source text used to make
        # the dub repeat the original language - the loudest "nothing changed"
        # bug in the old pipeline.
        text=seg.get("translated_text") or "",
        language=lang,
        voice_id=entry["voice_id"],
        ref_wav=ref,
        voice_mode=entry["voice_mode"],
        ref_text=entry.get("ref_text"),
        instruct=entry.get("design_prompt"),
        target_duration_ms=seg.get("target_duration_ms"),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(engine.generate, request, out).result(timeout=timeout_s)


def _render_segment(pool: _EnginePool, job_dir: Path, seg: dict, entry: dict, lang: str,
                    synth_dir: Path, timeout_s: int) -> tuple[bool, str | None]:
    """Render one segment on the speaker's locked engine, retrying that same
    engine before giving up. Returns (ok, error)."""
    sid = seg.get("speaker_id") or "S00"
    if not (seg.get("translated_text") or "").strip():
        seg["status"] = "failed"
        seg["error"] = "no translation to dub"
        seg["generated_wav"] = None
        log.warning("Segment %s has no translation; it will not be dubbed", seg.get("segment_id"))
        return False, "no translation to dub"
    out = synth_dir / f"{seg['segment_id']}.wav"
    ref = _ref_wav(job_dir, sid, entry["voice_id"])
    engine = pool.get(entry["engine"])
    last_error: str | None = None

    for attempt in range(1, ATTEMPTS_PER_SEGMENT + 1):
        try:
            result = _generate_once(engine, seg, entry, lang, ref, out, timeout_s)
        except concurrent.futures.TimeoutError:
            last_error = f"timeout after {timeout_s}s on {entry['engine']}"
            log.warning("Segment %s attempt %d/%d: %s", seg.get("segment_id"), attempt,
                        ATTEMPTS_PER_SEGMENT, last_error)
            continue
        except Exception as exc:  # noqa: BLE001 - engine errors are reported, not raised
            last_error = str(exc)
            log.warning("Segment %s attempt %d/%d failed on %s: %s", seg.get("segment_id"),
                        attempt, ATTEMPTS_PER_SEGMENT, entry["engine"], exc)
            continue

        seg["generated_wav"] = str(out.relative_to(job_dir))
        seg["generated_duration_ms"] = result.duration_ms
        seg["status"] = "synthesized"
        seg["tts_engine"] = result.engine
        seg["voice_mode"] = entry["voice_mode"]
        seg["voice_id"] = entry["voice_id"]
        seg.pop("error", None)
        return True, None

    return False, last_error


# --- block rendering ------------------------------------------------------

def _block_timeout(block: dict, base_timeout_s: int) -> int:
    slot_s = max(1.0, float(block.get("target_duration_ms") or 1000) / 1000.0)
    return int(max(base_timeout_s, slot_s * BLOCK_TIMEOUT_PER_S))


def _render_block(pool: _EnginePool, job_dir: Path, block: dict, entry: dict, lang: str,
                  synth_dir: Path, timeout_s: int) -> tuple[bool, str | None]:
    """Render a whole block of speech in ONE call.

    This is the point of the block: the engine sees the full run of text, so it
    keeps one breath, one intonation arc and one speaking rate across it. Cue
    boundaries inside the block never become edit points.
    """
    sid = block.get("speaker_id") or "S00"
    out = synth_dir / f"{block['block_id']}.wav"
    ref = _ref_wav(job_dir, sid, entry["voice_id"])
    engine = pool.get(entry["engine"])
    budget = _block_timeout(block, timeout_s)
    last_error: str | None = None
    payload = {
        "segment_id": block.get("block_id"),
        "translated_text": block.get("text") or "",
        "target_duration_ms": block.get("target_duration_ms"),
    }

    for attempt in range(1, ATTEMPTS_PER_SEGMENT + 1):
        try:
            result = _generate_once(engine, payload, entry, lang, ref, out, budget)
        except concurrent.futures.TimeoutError:
            last_error = f"timeout after {budget}s on {entry['engine']}"
            log.warning("Block %s attempt %d/%d: %s", block.get("block_id"), attempt,
                        ATTEMPTS_PER_SEGMENT, last_error)
            continue
        except Exception as exc:  # noqa: BLE001 - engine errors are reported, not raised
            last_error = str(exc)
            log.warning("Block %s attempt %d/%d failed on %s: %s", block.get("block_id"),
                        attempt, ATTEMPTS_PER_SEGMENT, entry["engine"], exc)
            continue

        block["generated_wav"] = str(out.relative_to(job_dir))
        block["generated_duration_ms"] = result.duration_ms
        block["status"] = "synthesized"
        block["tts_engine"] = result.engine
        block["voice_mode"] = entry["voice_mode"]
        block["voice_id"] = entry["voice_id"]
        block.pop("error", None)
        return True, None

    return False, last_error


def _members(segments: list[dict], block: dict) -> list[dict]:
    by_id = {str(s.get("segment_id")): s for s in segments}
    return [by_id[str(i)] for i in (block.get("segment_ids") or []) if str(i) in by_id]


def _publish_block(block: dict, members: list[dict]) -> None:
    """Point the block's cues at the block's audio.

    The first cue carries it for the whole run; the rest are marked `in_block`
    so no later stage tries to fit or re-edit them on their own.
    """
    rel = block.get("generated_wav")
    for i, seg in enumerate(members):
        seg["block_wav"] = rel
        seg["tts_engine"] = block.get("tts_engine")
        seg["voice_id"] = block.get("voice_id") or seg.get("voice_id")
        seg["voice_mode"] = block.get("voice_mode") or seg.get("voice_mode")
        seg.pop("error", None)
        if i == 0:
            seg["generated_wav"] = rel
            seg["generated_duration_ms"] = block.get("generated_duration_ms")
            seg["status"] = "synthesized"
        else:
            seg["generated_wav"] = None
            seg["generated_duration_ms"] = None
            seg["status"] = "in_block"


def _unblock(members: list[dict]) -> None:
    """Detach cues from a block that could not be rendered.

    Invariant for the rest of the pipeline: a cue has a `block_id` only if that
    block really produced audio. Anything else is rendered cue by cue.
    """
    for seg in members:
        seg["block_id"] = None
        seg["block_index"] = 0
        seg["block_size"] = 1
        seg["block_role"] = "solo"
        seg["block_wav"] = None


def _synthesize_segments(pool: _EnginePool, job_dir: Path, job: dict, targets: list[dict],
                         plan: dict[str, dict], lang: str, synth_dir: Path, timeout_s: int,
                         *, report_progress: bool = True) -> dict[str, str]:
    """Cue-by-cue rendering: the original path, and the fallback for a block
    that would otherwise be lost entirely."""
    errors: dict[str, str] = {}
    downgraded: list[str] = []
    total = len(targets)

    for i, seg in enumerate(targets):
        sid = seg.get("speaker_id") or "S00"
        entry = plan[sid]
        if not (seg.get("translated_text") or "").strip():
            # Nothing to say in the target language. That is a translation
            # problem, not an engine problem, so it must NOT drag the whole
            # speaker onto the fallback voice.
            seg["status"] = "failed"
            seg["error"] = "no translation to dub"
            seg["generated_wav"] = None
            errors[seg["segment_id"]] = seg["error"]
            log.warning("Segment %s has no translation; it will not be dubbed",
                        seg.get("segment_id"))
            continue
        if report_progress and job.get("job_id") and total > 0:
            job["percent"] = min(75 + int((i / total) * 10), 84)
            job["message"] = f"Synthesizing speech {i + 1}/{total} ({sid})"
            _safe_save_job(job)

        ok, error = _render_segment(pool, job_dir, seg, entry, lang, synth_dir, timeout_s)
        if ok:
            continue

        errors[seg["segment_id"]] = error or "unknown synthesis error"
        if entry["engine"] != FALLBACK_ENGINE and not entry["downgraded"]:
            log.warning("Speaker %s: %s failed, downgrading the whole speaker to %s",
                        sid, entry["engine"], FALLBACK_ENGINE)
            entry["engine"] = FALLBACK_ENGINE
            entry["voice_mode"] = FALLBACK_VOICE_MODE
            entry["downgraded"] = True
            if sid not in downgraded:
                downgraded.append(sid)
        else:
            seg["status"] = "failed"
            seg["error"] = errors[seg["segment_id"]]

    # Re-render every cue of a downgraded speaker on the fallback engine,
    # including the ones that already worked, so one bad cue can no longer
    # change that speaker's voice mid-video.
    for sid in downgraded:
        entry = plan[sid]
        again = [
            s for s in targets
            if (s.get("speaker_id") or "S00") == sid and (s.get("translated_text") or "").strip()
        ]
        log.warning("Re-rendering %d cue(s) for speaker %s on %s for voice consistency",
                    len(again), sid, entry["engine"])
        if job.get("job_id"):
            job["message"] = f"Re-rendering {sid} on {entry['engine']} for a consistent voice"
            _safe_save_job(job)
        for seg in again:
            ok, error = _render_segment(pool, job_dir, seg, entry, lang, synth_dir, timeout_s)
            if ok:
                errors.pop(seg["segment_id"], None)
            else:
                seg["status"] = "failed"
                seg["error"] = error or "unknown synthesis error"
                seg.pop("generated_wav", None)
                errors[seg["segment_id"]] = seg["error"]
    return errors


def _synthesize_blocks(pool: _EnginePool, job_dir: Path, job: dict, segments: list[dict],
                       blocks: list[dict], plan: dict[str, dict], lang: str, synth_dir: Path,
                       timeout_s: int) -> dict[str, str]:
    errors: dict[str, str] = {}
    downgraded: list[str] = []
    total = len(blocks)

    for i, block in enumerate(blocks):
        sid = block.get("speaker_id") or "S00"
        entry = plan.setdefault(sid, {
            "engine": FALLBACK_ENGINE, "voice_mode": FALLBACK_VOICE_MODE,
            "voice_id": block.get("voice_id") or sid, "design_prompt": None,
            "ref_text": None, "downgraded": False,
        })
        if job.get("job_id") and total > 0:
            job["percent"] = min(75 + int((i / total) * 10), 84)
            job["message"] = f"Synthesizing speech block {i + 1}/{total} ({sid})"
            _safe_save_job(job)

        ok, error = _render_block(pool, job_dir, block, entry, lang, synth_dir, timeout_s)
        if ok:
            continue
        errors[block["block_id"]] = error or "unknown synthesis error"
        if entry["engine"] != FALLBACK_ENGINE and not entry["downgraded"]:
            log.warning("Speaker %s: %s failed, downgrading the whole speaker to %s",
                        sid, entry["engine"], FALLBACK_ENGINE)
            entry["engine"] = FALLBACK_ENGINE
            entry["voice_mode"] = FALLBACK_VOICE_MODE
            entry["downgraded"] = True
            if sid not in downgraded:
                downgraded.append(sid)

    for sid in downgraded:
        entry = plan[sid]
        again = [b for b in blocks if (b.get("speaker_id") or "S00") == sid]
        log.warning("Re-rendering %d block(s) for speaker %s on %s for voice consistency",
                    len(again), sid, entry["engine"])
        for block in again:
            ok, error = _render_block(pool, job_dir, block, entry, lang, synth_dir, timeout_s)
            if ok:
                errors.pop(block["block_id"], None)
            else:
                errors[block["block_id"]] = error or "unknown synthesis error"

    # Publish what worked, and drop back to cue-by-cue for what did not, so a
    # single bad block costs a few seconds of flow instead of the whole run.
    rescue: list[dict] = []
    for block in blocks:
        members = _members(segments, block)
        if block.get("status") == "synthesized" and block.get("generated_wav"):
            _publish_block(block, members)
            continue
        block["status"] = "failed"
        block["error"] = errors.get(block["block_id"], "unknown synthesis error")
        block["generated_wav"] = None
        log.error("Block %s failed (%s); falling back to %d separate cue(s)",
                  block.get("block_id"), block["error"], len(members))
        _unblock(members)
        rescue.extend(members)

    # Cues that never made it into a block (no translation, or already marked
    # failed upstream) still have to be resolved, so the mix QC can see them
    # instead of silently losing that piece of the video.
    in_block = {str(sid) for b in blocks for sid in (b.get("segment_ids") or [])}
    leftovers = [s for s in segments if str(s.get("segment_id")) not in in_block]
    todo = rescue + leftovers
    if todo:
        errors.update(_synthesize_segments(pool, job_dir, job, todo, plan, lang,
                                           synth_dir, timeout_s, report_progress=False))
    return errors


def run_synthesis(job_dir: Path, job: dict) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    synth_dir = job_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    lang = job.get("target_language") or "hi"
    timeout_s = int(job.get("synthesis_timeout_s") or SEGMENT_TIMEOUT_S)
    default_engine_name = job.get("tts_engine") or "veena"
    speakers = _speaker_lookup(job_dir)
    plan = build_voice_plan(segments, speakers, default_engine_name)
    pool = _EnginePool()
    blocks = load_blocks(job_dir)
    if not blocks:
        # A job that was translated before blocks existed, or resumed from an
        # old checkpoint, still gets grouped here: synthesis is ALWAYS one call
        # per run of speech.
        blocks = run_blocks(job_dir, job)
        segments = json.loads(path.read_text(encoding="utf-8"))
    total = len(segments)

    log.info("Starting synthesis for job %s: %d cue(s) in %d block(s), voice plan: %s",
             job.get("job_id"), total, len(blocks),
             {sid: entry["engine"] for sid, entry in plan.items()})
    if job.get("job_id"):
        job["message"] = "Initializing neural voice synthesis..."
        _safe_save_job(job)

    if blocks:
        _synthesize_blocks(pool, job_dir, job, segments, blocks, plan, lang, synth_dir, timeout_s)
        blocks_path(job_dir).write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        _synthesize_segments(pool, job_dir, job, segments, plan, lang, synth_dir, timeout_s)

    failed = [s["segment_id"] for s in segments if s.get("status") == "failed"]
    failed_blocks = [b["block_id"] for b in blocks if b.get("status") == "failed"]
    if failed:
        log.error("Synthesis finished with %d failed cue(s): %s", len(failed), failed)

    if job.get("job_id"):
        job["percent"] = 85
        job["failed_segments"] = failed
        job["failed_blocks"] = failed_blocks
        job["block_count"] = len(blocks)
        job["voice_plan"] = {sid: entry["engine"] for sid, entry in plan.items()}
        rendered = len(blocks) - len(failed_blocks) if blocks else total - len(failed)
        unit = "block" if blocks else "segment"
        job["message"] = (
            f"Completed synthesis of all {len(blocks) if blocks else total} {unit}s"
            if not failed
            else f"Synthesized {rendered} {unit}s, {len(failed)} cue(s) failed"
        )
        _safe_save_job(job)

    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
