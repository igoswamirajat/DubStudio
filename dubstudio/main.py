from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dubstudio.api.routes_health import router as health_router
from dubstudio.api.routes_jobs import router as jobs_router
from dubstudio.api.routes_speakers import router as speakers_router
from dubstudio.api.routes_segments import router as segments_router
from dubstudio.api.routes_settings import router as settings_router
from dubstudio.settings import settings

_RESERVED = {"api", "docs", "redoc", "openapi.json", "assets"}


def create_app() -> FastAPI:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    # Route model caches to a configured location before any ML lib loads.
    if settings.hf_home:
        import os

        os.environ.setdefault("HF_HOME", settings.hf_home)
        os.environ["HF_HOME"] = settings.hf_home

    d_temp = Path("D:/temp")
    if d_temp.is_dir():
        import os, tempfile

        for env_var in ("TMP", "TEMP", "TMPDIR"):
            os.environ[env_var] = str(d_temp)
        tempfile.tempdir = str(d_temp)

    from dubstudio.jobs.store import store

    store.recover_interrupted()
    app = FastAPI(title="DubStudio", version="0.9.0")
    # Local-first single-user: same-origin UI is served by this app, so CORS is
    # only needed for the Vite dev server. Restrict to loopback origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(jobs_router, prefix="/api/v1")
    app.include_router(speakers_router, prefix="/api/v1")
    app.include_router(segments_router, prefix="/api/v1")
    app.include_router(settings_router, prefix="/api/v1")

    dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if dist.is_dir() and (dist / "index.html").exists():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/")
        def spa_index():
            return FileResponse(dist / "index.html")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            first = (full_path or "").split("/", 1)[0]
            if first in _RESERVED or full_path.startswith("api/"):
                raise HTTPException(404, "Not found")
            candidate = dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
