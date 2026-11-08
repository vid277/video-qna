from __future__ import annotations

from datetime import datetime
from typing import Annotated
from enum import StrEnum

from pydantic import AfterValidator, Field

from models.camera import Record
from models.memory import ChangeKind


type LocalTime = Annotated[datetime, AfterValidator(lambda at: at if at.tzinfo else at.astimezone())]
"""A time from the model: one without an offset means local time, never the database's zone."""


class RoomKind(StrEnum):
    LABELS = "change_labels"
    LINKS = "item_links"


class VectorOwner(StrEnum):
    FRAME = "frame"
    CLIP = "clip"
    CHANGE = "change"
    CROP = "crop"
    FACT = "fact"


class TextKind(StrEnum):
    FACT_VECTORS = "fact_vectors"


class EmbeddingRow(Record):
    owner: VectorOwner
    owner_id: int
    part: int
    camera_id: str
    event_id: int
    captured_at: datetime
    vector: list[float]


class EventWork(Record):
    event_id: int
    camera_id: str
    captured_at: datetime


class StoredChange(Record):
    id: int
    event_id: int
    kind: ChangeKind
    before_path: str
    after_path: str
    changed_after: datetime
    detected_at: datetime


class SearchHit(Record):
    ref: str
    owner: VectorOwner
    event_id: int | None
    captured_at: datetime
    score: float


class ChangeDigest(Record):
    ref: str
    kind: ChangeKind


class EventDigest(Record):
    ref: str
    camera_id: str
    started_at: datetime
    ended_at: datetime | None
    summary: str | None
    new: list[str] | None
    key_frames: list[str]
    changes: list[ChangeDigest]


class Caption(Record):
    summary: str
    new: list[str]
    facts: list[str]
    objects: list[str]


class FactHit(Record):
    ref: str
    started_at: datetime
    text: str
    rank: float


class SearchArgs(Record):
    query: str = Field(description="Short phrase naming an object, person, pet or action, e.g. 'keys', 'cat eating'.")
    start: LocalTime | None = Field(default=None, description="ISO 8601 start of the time window, with offset.")
    end: LocalTime | None = Field(default=None, description="ISO 8601 end of the time window, with offset.")


class EventsArgs(Record):
    start: LocalTime = Field(description="ISO 8601 start of the time window, with offset.")
    end: LocalTime = Field(description="ISO 8601 end of the time window, with offset.")


class LookArgs(Record):
    refs: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Refs from search or events results: 'frame:ID', 'change:ID' or 'event:ID'.",
    )


class FactsArgs(Record):
    query: str = Field(description="What to look for in event descriptions, e.g. 'drinking water' or 'cat food'.")
    start: LocalTime | None = Field(default=None, description="ISO 8601 start of the time window, with offset.")
    end: LocalTime | None = Field(default=None, description="ISO 8601 end of the time window, with offset.")


class ToolCall(Record):
    id: str
    name: str
    arguments: str


class ToolResult(Record):
    id: str
    name: str
    summary: str
    refs: list[str]
    output: str


class AnswerDelta(Record):
    text: str


class Answer(Record):
    text: str
    refs: list[str]


class ChangeCategory(StrEnum):
    OBJECT = "object"
    PERSON = "person"
    LIGHTING = "lighting"
    SCREEN = "screen"
    FURNITURE = "furniture"
    OTHER = "other"


class ChangeLabel(Record):
    category: ChangeCategory
    label: str
    place: str


class ItemChange(Record):
    ref: str
    item_ref: str
    kind: ChangeKind
    label: str | None
    place: str | None
    changed_after: datetime
    detected_at: datetime


class RoomItem(Record):
    item_ref: str
    label: str | None
    place: str | None
    since: datetime
    evidence: str


class RoomStateArgs(Record):
    at: LocalTime | None = Field(default=None, description="ISO 8601 time with offset; omit for now.")


class HistoryArgs(Record):
    query: str = Field(description="Short phrase for the object or part of the room, e.g. 'book', 'AirPods', 'door'.")
