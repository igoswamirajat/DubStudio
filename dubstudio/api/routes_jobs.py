from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from ulid import ULID

from dubstudio.jobs.runner import cancel, enqueue
from dubstudio.jobs.store import store
from dubstudio.settings import settings

router = APIRouter()


def _new_id() -> str:
    return f"job_{ULID()}"


@router.post("/jobs")
async def create_job(
    file: UploadFile | None = File(default=None),
    target_language: str = Form("hi"),
    source_language: str = Form(""),
    skip_separation: str = Form("false"),
    tts_engine: str = Form(""),
):
    job_id = _new_id()
    job_dir = settings.jobs_dir / job_id / "source"
    job_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename if file and file.filename else "none.bin"
    dest = job_dir / (filename or "upload.bin")
    if file is not None:
        max_bytes = settings.max_upload_mb * 1024 * 1024
        written = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        out.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(413, f"file too large (>{settings.max_upload_mb} MB)")
                    out.write(chunk)
        finally:
            await file.close()
    engine = (tts_engine or settings.tts_engine or "omnivoice").strip().lower()
    job = {
        "job_id": job_id,
        "state": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_filename": filename,
        "source_language": source_language or None,
        "target_language": target_language,
        "tts_engine": engine,
        "skip_separation": skip_separation.lower() == "true",
        "percent": 0,
        "stage_index": 0,
        "stage_total": 12,
        "message": "Created",
        "error": None,
        "speakers": [],
        "artifacts": {},
    }
    store.create(job)
    await enqueue(job_id)
    return JSONResponse({"job_id": job_id, "state": "queued", "tts_engine": engine}, status_code=201)


@router.get("/jobs")
def list_jobs():
    return {"jobs": store.list()}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, {"error": {"code": "JOB_NOT_FOUND", "message": job_id}})
    base = settings.jobs_dir / job_id
    has_output = (base / "export" / "output.mp4").is_file()
    src_dir = base / "source"
    has_source = any(p.is_file() and p.name != "meta.json" for p in src_dir.glob("*")) if src_dir.is_dir() else False
    job["has_output"] = has_output
    job["has_source"] = has_source
    return job


@router.get("/jobs/{job_id}/segments")
def get_segments(job_id: str):
    path = settings.jobs_dir / job_id / "segments" / "segments.json"
    if not path.exists():
        return {"segments": []}

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {"segments": raw if isinstance(raw, list) else raw.get("segments", [])}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.get("state") not in {"completed", "failed", "canceled", "created"}:
        await cancel(job_id)
    store.delete(job_id)
    job_dir = settings.jobs_dir / job_id
    if job_dir.is_dir():
        shutil.rmtree(job_dir, ignore_errors=True)
    return {"ok": True, "deleted": job_id}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if not store.get(job_id):
        raise HTTPException(404, "job not found")
    await cancel(job_id)
    return {"ok": True}


@router.get("/jobs/{job_id}/download")
def download(job_id: str, artifact: str = "json"):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    base = settings.jobs_dir / job_id
    mapping = {
        "json": base / "job.json",
        "mp4": base / "export" / "output.mp4",
        "srt": base / "export" / "output.srt",
    }
    path = mapping.get(artifact)
    if not path or not path.exists():
        raise HTTPException(404, "artifact not ready")
    return FileResponse(path)


@router.get("/jobs/{job_id}/media/{kind}")
def media(job_id: str, kind: str):
    """Serve source or dubbed video inline for in-browser preview."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    base = settings.jobs_dir / job_id
    if kind == "output":
        path = base / "export" / "output.mp4"
        if not path or not path.exists():
            raise HTTPException(404, {"error": {"code": "OUTPUT_NOT_READY", "message": "Dubbed output video is not ready yet."}})
    elif kind == "source":
        src_dir = base / "source"
        cands = [p for p in src_dir.glob("*") if p.is_file() and p.name != "meta.json"] if src_dir.is_dir() else []
        path = cands[0] if cands else None
        if not path or not path.exists():
            raise HTTPException(404, {"error": {"code": "SOURCE_NOT_FOUND", "message": "Source video file not found."}})
    else:
        raise HTTPException(400, "kind must be source or output")

    return FileResponse(
        path,
        media_type="video/mp4",
        content_disposition_type="inline",
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/jobs/{job_id}/events")
async def events(job_id: str):
    if not store.get(job_id):
        raise HTTPException(404, "job not found")

    async def gen():
        last = None
        while True:
            job = store.get(job_id)
            if not job:
                break
            snap = (job["state"], job.get("percent"), job.get("message"))
            if snap != last:
                last = snap
                kind = "progress"
                if job["state"] == "completed":
                    kind = "done"
                elif job["state"] == "failed":
                    kind = "error"
                yield {"event": kind, "data": json.dumps(job, default=str)}
                if kind in {"done", "error"}:
                    break
            await asyncio.sleep(0.25)

    return EventSourceResponse(gen())
