from __future__ import annotations

from datetime import datetime

from models.camera import Record
from models.jobs import Backlog


class LogLine(Record):
    at: datetime
    level: str
    message: str


class ProcessHealth(Record):
    role: str
    pid: int
    started_at: datetime
    written_at: datetime
    last_snapshot_at: datetime | None
    disk_bytes: int | None
    last_problem: LogLine | None


class ProcessStatus(Record):
    health: ProcessHealth
    running: bool
    heartbeat_age_s: float


class HealthReport(Record):
    checked_at: datetime
    processes: list[ProcessStatus]
    latest_frame_at: datetime | None
    backlog: list[Backlog]
    failing_jobs: int
    disk_bytes: int | None
    disk_budget_bytes: int
    disk_free_bytes: int
