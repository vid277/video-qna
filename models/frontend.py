from __future__ import annotations

from datetime import datetime

from pydantic import Field

from models.camera import Record
from models.health import HealthReport
from models.memory import ChangeKind


class AskRequest(Record):
    question: str = Field(min_length=1, max_length=500)


class Evidence(Record):
    ref: str
    label: str
    at: datetime
    image_url: str


class StreamDone(Record):
    seconds: float


class StreamError(Record):
    error: str


class ChangeView(Record):
    ref: str
    event_ref: str
    kind: ChangeKind
    at: datetime
    before_url: str
    after_url: str


class ActivityView(Record):
    ref: str
    started_at: datetime
    ended_at: datetime | None
    summary: str | None
    image_url: str
    changes: list[ChangeKind]


class ItemView(Record):
    ref: str
    label: str | None
    place: str | None
    since: datetime
    evidence: str
    image_url: str


class RoomView(Record):
    latest: Evidence | None
    state: list[ItemView]
    activity: list[ActivityView]
    changes: list[ChangeView]
    health: HealthReport
