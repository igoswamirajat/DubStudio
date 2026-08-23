from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from dubstudio.jobs.states import PIPELINE_STAGES, require_transition
from dubstudio.jobs.store import store

log = logging.getLogger("dubstudio.runner")
_gpu_lock = asyncio.Lock()
_tasks: dict[str, asyncio.Task] = {}


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
            await _pipeline(job_id)
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
            log.exception("job %s failed", job_id)


async def _pipeline(job_id: str) -> None:
    job = store.get(job_id)
    assert job
    require_transition(job["state"], "ingesting")
    total = len(PIPELINE_STAGES)
    for i, stage in enumerate(PIPELINE_STAGES, start=1):
        job = store.get(job_id)
        assert job
        if job["state"] == "canceled":
            return
        require_transition(job["state"] if i == 1 else PIPELINE_STAGES[i - 2], stage)
        job["state"] = stage
        job["stage_index"] = i
        job["stage_total"] = total
        job["percent"] = int(i / total * 95)
        job["message"] = f"{stage.replace('_', ' ').title()} (phase 0 stub)"
        store.save(job)
        await asyncio.sleep(0.35)
    job = store.get(job_id)
    assert job
    require_transition(job["state"], "completed")
    job["state"] = "completed"
    job["percent"] = 100
    job["message"] = "Completed (phase 0 stub — no media processed yet)"
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    store.save(job)


async def cancel(job_id: str) -> None:
    t = _tasks.get(job_id)
    if t:
        t.cancel()
    job = store.get(job_id)
    if job and job["state"] not in {"completed", "failed", "canceled"}:
        job["state"] = "canceled"
        job["message"] = "Canceled"
        store.save(job)
