from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

from dubstudio.settings import settings


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[str] = mapped_column(String(32), default="created")
    payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class JobStore:
    def __init__(self, db_path: Path | None = None) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        path = db_path or settings.db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{path}",
            echo=False,
            connect_args={"timeout": 30, "check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)

    def create(self, job: dict[str, Any]) -> dict[str, Any]:
        with Session(self.engine) as s:
            row = JobRow(job_id=job["job_id"], state=job["state"], payload=json.dumps(job))
            s.add(row)
            s.commit()
        self._write_snapshot(job)
        return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as s:
            row = s.get(JobRow, job_id)
            if not row:
                return None
            return json.loads(row.payload)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            rows = s.scalars(select(JobRow).order_by(JobRow.created_at.desc()).limit(limit)).all()
            return [json.loads(r.payload) for r in rows]

    def save(self, job: dict[str, Any]) -> dict[str, Any]:
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        with Session(self.engine) as s:
            row = s.get(JobRow, job["job_id"])
            if row is None:
                raise KeyError(job["job_id"])
            row.state = job["state"]
            row.payload = json.dumps(job)
            row.updated_at = datetime.now(timezone.utc)
            s.commit()
        self._write_snapshot(job)
        return job

    def delete(self, job_id: str) -> bool:
        with Session(self.engine) as s:
            row = s.get(JobRow, job_id)
            if row is None:
                return False
            s.delete(row)
            s.commit()
        return True

    def recover_interrupted(self) -> int:
        """Mark jobs left mid-flight by a previous process as failed.

        In-flight tasks live only in memory, so any non-terminal job found at
        startup was interrupted and can never resume on its own.
        """
        terminal = {"completed", "failed", "canceled", "created"}
        recovered = 0
        for job in self.list(limit=1000):
            if job.get("state") not in terminal:
                job["state"] = "failed"
                job["error"] = "interrupted by server restart"
                job["message"] = "Interrupted (server restarted)"
                try:
                    self.save(job)
                    recovered += 1
                except KeyError:
                    pass
        return recovered

    def _write_snapshot(self, job: dict[str, Any]) -> None:
        d = settings.jobs_dir / job["job_id"]
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        dest = d / "job.json"
        tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        tmp.replace(dest)


store = JobStore()
