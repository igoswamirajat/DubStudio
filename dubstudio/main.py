from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dubstudio.api.routes_health import router as health_router
from dubstudio.api.routes_jobs import router as jobs_router
from dubstudio.api.routes_speakers import router as speakers_router
from dubstudio.api.routes_segments import router as segments_router
from dubstudio.settings import settings


def create_app() -> FastAPI:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="DubStudio", version="0.8.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(jobs_router, prefix="/api/v1")
    app.include_router(speakers_router, prefix="/api/v1")
    app.include_router(segments_router, prefix="/api/v1")
    return app


app = create_app()
