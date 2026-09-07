from __future__ import annotations

import asyncio
import logging
import traceback

from dubstudio.jobs.states import require_transition
from dubstudio.jobs.store import store
from dubstudio.pipeline.orchestrator import run_job, run_job_synthesis
from dubstudio.settings import settings

log = logging.getLogger("dubstudio.runner")
_gpu_lock = asyncio.Lock()
_tasks: dict[str, asyncio.Task] = {}


def is_active(job_id: str) -> bool:
    t = _tasks.get(job_id)
    return t is not None and not t.done()


async def enqueue(job_id: str) -> None:
    job = store.get(job_id)
    if not job:
        raise KeyError(job_id)
    require_transition(job["state"], "queued")
    job["state"] = "queued"
    job["percent"] = 1
    job["message"] = "Queued"
    store.save(job)
    _tasks[job_id] = asyncio.create_task(_run(job_id))


async def _run(job_id: str) -> None:
    async with _gpu_lock:
        try:
            await asyncio.to_thread(run_job, job_id)
        except asyncio.CancelledError:
            job = store.get(job_id)
            if job and job["state"] not in {"completed", "failed"}:
                job["state"] = "canceled"
                job["message"] = "Canceled"
                store.save(job)
        except Exception as exc:
            job = store.get(job_id) or {"job_id": job_id, "state": "failed"}
            job["state"] = "failed"
            job["error"] = str(exc)
            job["message"] = "Failed"
            store.save(job)
            log_path = settings.jobs_dir / job_id / "logs" / "pipeline.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            log.exception("job %s failed", job_id)


async def resume_synthesis(job_id: str) -> None:
    """Resume pipeline from awaiting_voice_selection → synthesizing."""
    job = store.get(job_id)
    if not job:
        raise KeyError(job_id)
    if job.get("state") != "awaiting_voice_selection":
        raise ValueError(
            f"Job {job_id} is in state '{job.get('state')}', expected 'awaiting_voice_selection'"
        )
    _tasks[job_id] = asyncio.create_task(_run_synth(job_id))


async def _run_synth(job_id: str) -> None:
    """Run synthesis phase under GPU lock."""
    async with _gpu_lock:
        try:
            await asyncio.to_thread(run_job_synthesis, job_id)
        except asyncio.CancelledError:
            job = store.get(job_id)
            if job and job["state"] not in {"completed", "failed"}:
                job["state"] = "canceled"
                job["message"] = "Canceled"
                store.save(job)
        except Exception as exc:
            job = store.get(job_id) or {"job_id": job_id, "state": "failed"}
            job["state"] = "failed"
            job["error"] = str(exc)
            job["message"] = "Failed"
            store.save(job)
            log_path = settings.jobs_dir / job_id / "logs" / "pipeline.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            log.exception("job %s synthesis failed", job_id)


async def cancel(job_id: str) -> None:
    t = _tasks.get(job_id)
    if t:
        t.cancel()
    job = store.get(job_id)
    if job and job["state"] not in {"completed", "failed", "canceled"}:
        job["state"] = "canceled"
        job["message"] = "Canceled"
        store.save(job)
