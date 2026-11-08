from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Computed, DateTime, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from models import ChangeKind, StoredChange, StoredFrame


EMBEDDING_DIM = 1024
EARLIEST = datetime.min.replace(tzinfo=UTC)
LATEST = datetime.max.replace(tzinfo=UTC)


def sqlalchemy_url(database_url: str) -> str:
    return make_url(database_url).set(drivername="postgresql+psycopg").render_as_string(hide_password=False)


class Base(DeclarativeBase):
    type_annotation_map = {datetime: DateTime(timezone=True), dict[str, Any]: JSONB, list[Any]: JSONB}


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime]
    ended_at: Mapped[datetime | None]
    summary: Mapped[str | None] = mapped_column(Text)
    new: Mapped[list[Any] | None]
    objects: Mapped[list[Any] | None]

    __table_args__ = (Index("events_by_time", "camera_id", "started_at"),)


class Frame(Base):
    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(Text)
    captured_at: Mapped[datetime]
    path: Mapped[str | None] = mapped_column(Text)
    video_path: Mapped[str | None] = mapped_column(Text)
    video_index: Mapped[int | None]
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id", ondelete="SET NULL"))
    is_key: Mapped[bool] = mapped_column(default=False)
    foreground: Mapped[list[Any]] = mapped_column(default=list)

    __table_args__ = (
        Index("frames_by_time", "camera_id", "captured_at"),
        Index("frames_by_event", "event_id"),
        Index("frames_by_video", "video_path"),
    )

    def stored(self) -> StoredFrame:
        return StoredFrame(
            id=self.id,
            captured_at=self.captured_at,
            path=self.path,
            is_key=self.is_key,
            video_path=self.video_path,
            video_index=self.video_index,
            event_id=self.event_id,
        )


class SceneChange(Base):
    __tablename__ = "changes"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(Text)
    box: Mapped[dict[str, Any]]
    source: Mapped[dict[str, Any] | None]
    before_path: Mapped[str] = mapped_column(Text)
    after_path: Mapped[str] = mapped_column(Text)
    changed_after: Mapped[datetime]
    detected_at: Mapped[datetime]
    category: Mapped[str | None] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(Text)
    place: Mapped[str | None] = mapped_column(Text)
    item_id: Mapped[int | None]

    __table_args__ = (Index("changes_by_event", "event_id"), Index("changes_by_item", "item_id"), Index("changes_by_time", "detected_at"))

    def stored(self) -> StoredChange:
        return StoredChange(
            id=self.id,
            event_id=self.event_id,
            kind=ChangeKind(self.kind),
            before_path=self.before_path,
            after_path=self.after_path,
            changed_after=self.changed_after,
            detected_at=self.detected_at,
        )


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[int]
    part: Mapped[int]
    camera_id: Mapped[str] = mapped_column(Text)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    captured_at: Mapped[datetime]
    model: Mapped[str] = mapped_column(Text)
    vector: Mapped[list[float]] = mapped_column(HALFVEC(EMBEDDING_DIM))

    __table_args__ = (
        Index("embeddings_unique", "owner", "owner_id", "part", "model", unique=True),
        Index("embeddings_by_owner", "owner", "owner_id"),
        Index("embeddings_by_event", "event_id", "owner"),
        Index("embeddings_by_time", "camera_id", "captured_at"),
        Index(
            "embeddings_search",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": "halfvec_cosine_ops"},
        ),
    )


class Fact(Base):
    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text)
    search: Mapped[str] = mapped_column(TSVECTOR, Computed("to_tsvector('english', text)", persisted=True))

    __table_args__ = (
        Index("facts_by_event", "event_id"),
        Index("facts_search", "search", postgresql_using="gin"),
    )


class JobFailure(Base):
    __tablename__ = "job_failures"

    kind: Mapped[str] = mapped_column(Text, primary_key=True)
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    attempts: Mapped[int]
    last_error: Mapped[str] = mapped_column(Text)
    retry_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(Text)
    period: Mapped[str] = mapped_column(Text)
    start: Mapped[datetime]
    end: Mapped[datetime]
    text: Mapped[str] = mapped_column(Text)
    events: Mapped[int]
    updated_at: Mapped[datetime]

    __table_args__ = (Index("summaries_unique", "camera_id", "period", "start", unique=True),)
