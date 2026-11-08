from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import Field

from models.camera import Record


class QueueName(StrEnum):
    EMBED = "embed"
    INDEX = "index"
    SERIAL = "serial"
    MAINTENANCE = "maintenance"


class IndexKind(StrEnum):
    CHANGE_VECTORS = "change_vectors"
    FRAME_VECTORS = "frame_vectors"
    CLIP_VECTORS = "clip_vectors"
    CAPTIONS = "captions"
    HOUR_SUMMARIES = "hour_summaries"
    DAY_SUMMARIES = "day_summaries"
    CROP_VECTORS = "crop_vectors"


class Period(StrEnum):
    HOUR = "hour"
    DAY = "day"


class QueueSettings(Record):
    workers: int = Field(ge=1)
    max_inflight: int = Field(ge=1)


class JobsConfig(Record):
    queue_file: Path
    queues: dict[QueueName, QueueSettings]
    dispatch_every_s: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    backoff_first_s: float = Field(gt=0)
    backoff_max_s: float = Field(gt=0)
    outage_pause_s: float = Field(gt=0)
    timezone: str
    vector_thin_every_s: float = Field(gt=0)
    vector_keep_days: int = Field(ge=1)

    @classmethod
    def from_yaml(cls, path: Path) -> JobsConfig:
        config = cls.model_validate_json(json.dumps(yaml.safe_load(path.read_text())))

        return config.model_copy(update={"queue_file": path.parent / config.queue_file})


class Job(Record):
    kind: str
    key: str


class Backoff(Record):
    max_attempts: int = Field(ge=1)
    first_s: float = Field(gt=0)
    max_s: float = Field(gt=0)


class Backlog(Record):
    kind: IndexKind
    pending: int
    oldest: datetime | None


class Failure(Record):
    kind: str
    key: str
    attempts: int
    last_error: str
    retry_at: datetime
    updated_at: datetime


class SummaryWork(Record):
    camera_id: str
    period: Period
    start: datetime

    @property
    def key(self) -> str:
        return f"{self.camera_id}|{self.period.value}|{self.start.isoformat()}"

    @classmethod
    def from_key(cls, key: str) -> SummaryWork:
        camera_id, period, start = key.split("|")

        return cls(camera_id=camera_id, period=Period(period), start=datetime.fromisoformat(start))


class SummaryInput(Record):
    at: datetime
    text: str
    events: int = Field(ge=1)
