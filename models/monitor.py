from __future__ import annotations

from datetime import datetime

from models.camera import Record
from models.health import HealthReport
from models.jobs import Backlog, Failure


class QueueStat(Record):
    name: str
    waiting: int
    workers: int
    max_inflight: int


class VectorStat(Record):
    owner: str
    model: str
    vectors: int
    oldest: datetime | None
    newest: datetime | None


class TableStat(Record):
    name: str
    rows: int
    bytes: int


class DiskPart(Record):
    name: str
    bytes: int


class ServiceStat(Record):
    url: str
    up: bool
    model: str | None
    device: str | None


class Counts(Record):
    events: int
    captioned: int
    changes: int
    frames: int
    frames_on_disk: int
    frames_in_video: int
    hour_summaries: int
    day_summaries: int
    events_last_hour: int
    events_last_day: int


class DatabaseStats(Record):
    database_bytes: int
    vector_index_bytes: int
    vectors: list[VectorStat]
    tables: list[TableStat]
    counts: Counts


class MonitorView(Record):
    generated_at: datetime
    embedding_model: str
    health: HealthReport
    queues: list[QueueStat]
    backlog: list[Backlog]
    failures: list[Failure]
    database: DatabaseStats
    disk: list[DiskPart]
    embedding_service: ServiceStat
