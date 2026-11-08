from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    ColumnElement,
    Select,
    Text,
    and_,
    cast,
    delete,
    exists,
    func,
    literal,
    or_,
    select,
    text,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import InstrumentedAttribute, aliased

from models import (
    Box,
    Caption,
    Change,
    ChangeDigest,
    ChangeKind,
    EmbeddingRow,
    EventDigest,
    EventWork,
    FactHit,
    SearchHit,
    StoredChange,
    StoredFrame,
    VectorOwner,
)
from models.jobs import (
    Backlog,
    Backoff,
    Failure,
    IndexKind,
    Period,
    SummaryInput,
    SummaryWork,
)
from models.monitor import Counts, DatabaseStats, TableStat, VectorStat
from store.db import (
    EARLIEST,
    LATEST,
    Embedding,
    Event,
    Fact,
    Frame,
    JobFailure,
    SceneChange,
    Summary,
    sqlalchemy_url,
)

REF_PREFIX = {
    VectorOwner.FRAME: "frame",
    VectorOwner.CROP: "frame",
    VectorOwner.CHANGE: "change",
    VectorOwner.CLIP: "event",
}
SEARCH_OVERFETCH = 4
NEVER = datetime(9000, 1, 1, tzinfo=UTC)
EVENTS_LIMIT = 50
EVENT_KINDS = [IndexKind.CHANGE_VECTORS, IndexKind.FRAME_VECTORS, IndexKind.CLIP_VECTORS, IndexKind.CAPTIONS, IndexKind.CROP_VECTORS]
PERIOD_STEP = {Period.HOUR: timedelta(hours=1), Period.DAY: timedelta(days=1)}
MONITORED_TABLES = ["events", "frames", "changes", "embeddings", "facts", "summaries", "job_failures"]


class Store:
    def __init__(self, database_url: str, embedding_model: str):
        self._engine = create_async_engine(sqlalchemy_url(database_url))
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._model = embedding_model

    async def close(self) -> None:
        await self._engine.dispose()

    async def open_event(self, camera_id: str, at: datetime) -> int:
        async with self._session.begin() as s:
            event = Event(camera_id=camera_id, started_at=at)
            s.add(event)
            await s.flush()

            return event.id

    async def open_events(self, camera_id: str) -> list[tuple[int, datetime]]:
        """Events a previous run left open, with the time of their last frame."""
        last_frame = select(func.max(Frame.captured_at)).where(Frame.event_id == Event.id).scalar_subquery()
        query = select(Event.id, func.coalesce(last_frame, Event.started_at)).where(Event.camera_id == camera_id, Event.ended_at.is_(None))

        async with self._session() as s:
            return [(event_id, at) for event_id, at in (await s.execute(query)).all()]

    async def close_event(self, event_id: int, at: datetime) -> None:
        async with self._session.begin() as s:
            await s.execute(update(Event).where(Event.id == event_id).values(ended_at=at))

    async def add_frame(
        self, camera_id: str, at: datetime, path: str, event_id: int | None, foreground: list[Box]
    ) -> None:
        async with self._session.begin() as s:
            s.add(
                Frame(
                    camera_id=camera_id,
                    captured_at=at,
                    path=path,
                    event_id=event_id,
                    foreground=[b.model_dump() for b in foreground],
                )
            )

    async def mark_key_frames(self, event_id: int, count: int) -> None:
        async with self._session.begin() as s:
            query = select(Frame.id).where(Frame.event_id == event_id).order_by(Frame.captured_at)
            ids = (await s.scalars(query)).all()

            step = max(len(ids) - 1, 0) / (count - 1)
            keys = {ids[round(i * step)] for i in range(count * bool(ids))}

            await s.execute(update(Frame).where(Frame.id.in_(keys)).values(is_key=True))

    async def add_change(
        self,
        event_id: int,
        change: Change,
        before_path: str,
        after_path: str,
        changed_after: datetime,
        detected_at: datetime,
    ) -> None:
        async with self._session.begin() as s:
            s.add(
                SceneChange(
                    event_id=event_id,
                    kind=change.kind.value,
                    box=change.box.model_dump(),
                    source=change.source.model_dump() if change.source else None,
                    before_path=before_path,
                    after_path=after_path,
                    changed_after=changed_after,
                    detected_at=detected_at,
                )
            )

    async def thinning_candidates(self, camera_id: str, cutoff: datetime) -> list[StoredFrame]:
        async with self._session() as s:
            rows = await s.scalars(
                select(Frame)
                .where(Frame.camera_id == camera_id, Frame.captured_at < cutoff)
                .where(Frame.path.is_not(None), ~Frame.is_key)
                .order_by(Frame.captured_at)
            )

            return [f.stored() for f in rows]

    async def replace_with_video(self, kept: Sequence[StoredFrame], dropped: Iterable[int], video_path: str) -> None:
        async with self._session.begin() as s:
            for index, frame in enumerate(kept):
                await s.execute(
                    update(Frame).where(Frame.id == frame.id).values(path=None, video_path=video_path, video_index=index)
                )

            await s.execute(delete(Frame).where(Frame.id.in_(list(dropped)), ~Frame.is_key))

    async def drop_video(self, video_path: str) -> None:
        async with self._session.begin() as s:
            await s.execute(delete(Frame).where(Frame.video_path == video_path, ~Frame.is_key))

    async def key_frames(self, event_id: int) -> list[StoredFrame]:
        async with self._session() as s:
            rows = await s.scalars(select(Frame).where(Frame.event_id == event_id, Frame.is_key).order_by(Frame.captured_at))

            return [f.stored() for f in rows]

    async def event_changes(self, event_id: int) -> list[StoredChange]:
        return await self._changes(SceneChange.event_id == event_id)

    async def frames(self, frame_ids: Iterable[int]) -> list[StoredFrame]:
        async with self._session() as s:
            rows = await s.scalars(select(Frame).where(Frame.id.in_(list(frame_ids))).order_by(Frame.captured_at))

            return [f.stored() for f in rows]

    async def changes(self, change_ids: Iterable[int]) -> list[StoredChange]:
        return await self._changes(SceneChange.id.in_(list(change_ids)))

    async def replace_embeddings(self, rows: Sequence[EmbeddingRow]) -> None:
        owners = {(r.owner.value, r.owner_id) for r in rows}

        async with self._session.begin() as s:
            await s.execute(delete(Embedding).where(tuple_(Embedding.owner, Embedding.owner_id).in_(owners)))
            s.add_all(Embedding(**r.model_dump(exclude={"owner"}), owner=r.owner.value, model=self._model) for r in rows)

    async def search(self, vector: list[float], start: datetime | None, end: datetime | None, limit: int) -> list[SearchHit]:
        distance = Embedding.vector.cosine_distance(vector)
        query = (
            select(Embedding.owner, Embedding.owner_id, Embedding.event_id, Embedding.captured_at, distance)
            .where(Embedding.model == self._model, Embedding.owner != VectorOwner.FACT.value)
            .where(Embedding.captured_at.between(start or EARLIEST, end or LATEST))
            .order_by(distance)
            .limit(limit * SEARCH_OVERFETCH)
        )

        async with self._session.begin() as s:
            await s.execute(text("set local hnsw.iterative_scan = relaxed_order"))
            rows = (await s.execute(query)).all()

        hits = [
            SearchHit(
                ref=f"{REF_PREFIX[VectorOwner(owner)]}:{owner_id}",
                owner=VectorOwner(owner),
                event_id=event_id,
                captured_at=captured_at,
                score=round(1 - float(d), 4),
            )
            for owner, owner_id, event_id, captured_at, d in rows
        ]
        best = {h.ref: h for h in reversed(hits)}

        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:limit]

    async def events_between(self, start: datetime, end: datetime) -> list[EventDigest]:
        query = (
            select(Event)
            .where(Event.started_at <= end, (Event.ended_at >= start) | Event.ended_at.is_(None))
            .order_by(Event.started_at.desc())
            .limit(EVENTS_LIMIT)
        )

        return list(reversed(await self._digests(query)))

    async def recent_events(self, limit: int) -> list[EventDigest]:
        return await self._digests(select(Event).order_by(Event.started_at.desc()).limit(limit))

    async def _digests(self, query: Select[Event]) -> list[EventDigest]:
        async with self._session() as s:
            events = (await s.scalars(query)).all()
            ids = [e.id for e in events]
            keys = (await s.execute(select(Frame.event_id, Frame.id).where(Frame.event_id.in_(ids), Frame.is_key))).all()
            changes = (await s.execute(select(SceneChange.event_id, SceneChange.id, SceneChange.kind).where(SceneChange.event_id.in_(ids)))).all()

        return [
            EventDigest(
                ref=f"event:{e.id}",
                camera_id=e.camera_id,
                started_at=e.started_at,
                ended_at=e.ended_at,
                summary=e.summary,
                new=e.new,
                key_frames=[f"frame:{f}" for event_id, f in keys if event_id == e.id],
                changes=[ChangeDigest(ref=f"change:{c}", kind=ChangeKind(kind)) for event_id, c, kind in changes if event_id == e.id],
            )
            for e in events
        ]

    async def set_caption(self, event_id: int, caption: Caption) -> None:
        facts = list(dict.fromkeys([*caption.facts, *caption.new]))
        values = {"summary": caption.summary, "new": caption.new, "objects": caption.objects}

        async with self._session.begin() as s:
            await s.execute(update(Event).where(Event.id == event_id).values(**values))
            await s.execute(delete(Fact).where(Fact.event_id == event_id))
            s.add_all(Fact(event_id=event_id, text=fact) for fact in facts)

    async def previous_summary(self, event_id: int) -> str | None:
        camera = select(Event.camera_id).where(Event.id == event_id).scalar_subquery()
        query = (
            select(Event.summary)
            .where(Event.camera_id == camera, Event.id < event_id, Event.summary.is_not(None))
            .order_by(Event.id.desc())
            .limit(1)
        )

        async with self._session() as s:
            return await s.scalar(query)

    async def search_facts(self, words: str, start: datetime | None, end: datetime | None, limit: int) -> list[FactHit]:
        terms = func.websearch_to_tsquery("english", " or ".join(words.split()))
        rank = func.ts_rank(Fact.search, terms)
        query = (
            select(Fact.text, Event.id, Event.started_at, rank)
            .join(Event, Event.id == Fact.event_id)
            .where(Fact.search.op("@@")(terms), Event.started_at.between(start or EARLIEST, end or LATEST))
            .order_by(rank.desc(), Event.started_at.desc())
            .limit(limit)
        )

        async with self._session() as s:
            rows = (await s.execute(query)).all()

        return [FactHit(ref=f"event:{event_id}", started_at=at, text=fact, rank=round(float(r), 4)) for fact, event_id, at, r in rows]

    async def frames_with_crops(self, frame_ids: Iterable[int]) -> set[int]:
        query = select(Embedding.owner_id).where(
            Embedding.owner == VectorOwner.CROP.value, Embedding.model == self._model, Embedding.owner_id.in_(list(frame_ids))
        )

        async with self._session() as s:
            return set((await s.scalars(query.distinct())).all())

    async def event_work(self, event_id: int) -> EventWork:
        query = select(Event.camera_id, Event.started_at, Event.ended_at).where(Event.id == event_id)

        async with self._session() as s:
            row = (await s.execute(query)).one()

        return EventWork(event_id=event_id, camera_id=row.camera_id, captured_at=row.ended_at or row.started_at)

    async def camera_ids(self) -> list[str]:
        async with self._session() as s:
            return list((await s.scalars(select(Event.camera_id).distinct())).all())

    async def pending_events(self, kind: IndexKind, limit: int, newer_than: datetime = EARLIEST) -> list[int]:
        query = (
            select(Event.id)
            .where(*self._pending(kind), Event.ended_at > newer_than)
            .order_by(Event.ended_at.desc())
            .limit(limit)
        )

        async with self._session() as s:
            return list((await s.scalars(query)).all())

    async def index_backlog(self) -> list[Backlog]:
        async with self._session() as s:
            rows = [(k, (await s.execute(select(func.count(), func.min(Event.ended_at)).where(*self._pending(k)))).one()) for k in EVENT_KINDS]

        return [Backlog(kind=k, pending=pending, oldest=oldest) for k, (pending, oldest) in rows]

    async def record_failure(self, kind: str, key: str, error: str, backoff: Backoff) -> None:
        now = datetime.now(UTC)

        async with self._session.begin() as s:
            attempts = (await s.scalar(select(JobFailure.attempts).where(JobFailure.kind == kind, JobFailure.key == key)) or 0) + 1
            delay = timedelta(seconds=min(backoff.first_s * 2 ** (attempts - 1), backoff.max_s))
            retry_at = NEVER if attempts >= backoff.max_attempts else now + delay
            values = {"attempts": attempts, "last_error": error, "retry_at": retry_at, "updated_at": now}

            await s.execute(
                insert(JobFailure)
                .values(kind=kind, key=key, **values)
                .on_conflict_do_update(index_elements=[JobFailure.kind, JobFailure.key], set_=values)
            )

    async def clear_failure(self, kind: str, key: str) -> None:
        async with self._session.begin() as s:
            await s.execute(delete(JobFailure).where(JobFailure.kind == kind, JobFailure.key == key))

    async def failures(self) -> list[Failure]:
        async with self._session() as s:
            rows = await s.scalars(select(JobFailure).order_by(JobFailure.updated_at.desc()))

            return [
                Failure(kind=f.kind, key=f.key, attempts=f.attempts, last_error=f.last_error, retry_at=f.retry_at, updated_at=f.updated_at)
                for f in rows
            ]

    async def clear_failures(self) -> int:
        async with self._session.begin() as s:
            return len((await s.scalars(delete(JobFailure).returning(JobFailure.key))).all())

    async def pending_summaries(self, period: Period, timezone: str, kind: str, limit: int) -> list[SummaryWork]:
        counts = PERIOD_COUNTS[period](timezone).subquery()
        done = aliased(Summary)
        query = (
            select(counts.c.camera_id, counts.c.start)
            .outerjoin(done, and_(done.camera_id == counts.c.camera_id, done.period == period.value, done.start == counts.c.start))
            .where(counts.c.start + literal(PERIOD_STEP[period]) <= func.now(), or_(done.id.is_(None), done.events != counts.c.events))
            .order_by(counts.c.start.desc())
        )
        blocking = select(JobFailure.key).where(JobFailure.kind == kind, JobFailure.retry_at > func.now())

        async with self._session() as s:
            blocked = set((await s.scalars(blocking)).all())
            rows = (await s.execute(query.limit(limit + len(blocked)))).all()

        works = [SummaryWork(camera_id=camera_id, period=period, start=start) for camera_id, start in rows]

        return [w for w in works if w.key not in blocked][:limit]

    async def summary_inputs(self, work: SummaryWork, end: datetime) -> list[SummaryInput]:
        async with self._session() as s:
            rows = (await s.execute(SUMMARY_INPUTS[work.period](work, end))).all()

        return [SummaryInput(at=at, text=body, events=events) for at, body, events in rows]

    async def set_summary(self, work: SummaryWork, end: datetime, body: str, events: int) -> None:
        values = {"end": end, "text": body, "events": events, "updated_at": datetime.now(UTC)}

        async with self._session.begin() as s:
            await s.execute(
                insert(Summary)
                .values(camera_id=work.camera_id, period=work.period.value, start=work.start, **values)
                .on_conflict_do_update(index_elements=[Summary.camera_id, Summary.period, Summary.start], set_=values)
            )

    async def database_stats(self) -> DatabaseStats:
        now = datetime.now(UTC)
        tables = text(
            "select relname, n_live_tup, pg_total_relation_size(relid) from pg_stat_user_tables"
            " where relname = any(:names) order by pg_total_relation_size(relid) desc"
        )
        sizes = text("select pg_database_size(current_database()), coalesce(pg_relation_size(to_regclass('embeddings_search')), 0)")
        vectors = (
            select(Embedding.owner, Embedding.model, func.count(), func.min(Embedding.captured_at), func.max(Embedding.captured_at))
            .group_by(Embedding.owner, Embedding.model)
            .order_by(Embedding.model, Embedding.owner)
        )
        events = select(
            func.count(),
            func.count().filter(Event.summary.is_not(None)),
            func.count().filter(Event.started_at > now - timedelta(hours=1)),
            func.count().filter(Event.started_at > now - timedelta(days=1)),
        ).select_from(Event)
        frames = select(func.count(), func.count().filter(Frame.path.is_not(None)), func.count().filter(Frame.video_path.is_not(None)))
        summaries = select(
            func.count().filter(Summary.period == Period.HOUR.value), func.count().filter(Summary.period == Period.DAY.value)
        ).select_from(Summary)

        async with self._session() as s:
            table_rows = (await s.execute(tables, {"names": MONITORED_TABLES})).all()
            database_bytes, index_bytes = (await s.execute(sizes)).one()
            vector_rows = (await s.execute(vectors)).all()
            event_count, captioned, last_hour, last_day = (await s.execute(events)).one()
            frame_count, on_disk, in_video = (await s.execute(frames.select_from(Frame))).one()
            change_count = await s.scalar(select(func.count()).select_from(SceneChange)) or 0
            hours, days = (await s.execute(summaries)).one()

        return DatabaseStats(
            database_bytes=database_bytes,
            vector_index_bytes=index_bytes,
            vectors=[VectorStat(owner=o, model=m, vectors=n, oldest=lo, newest=hi) for o, m, n, lo, hi in vector_rows],
            tables=[TableStat(name=name, rows=rows, bytes=size) for name, rows, size in table_rows],
            counts=Counts(
                events=event_count,
                captioned=captioned,
                changes=change_count,
                frames=frame_count,
                frames_on_disk=on_disk,
                frames_in_video=in_video,
                hour_summaries=hours,
                day_summaries=days,
                events_last_hour=last_hour,
                events_last_day=last_day,
            ),
        )

    async def thin_vectors(self, cutoff: datetime) -> int:
        minute = func.date_trunc("minute", Embedding.captured_at)
        ranked = (
            select(Embedding.id, func.row_number().over(partition_by=(Embedding.camera_id, Embedding.model, minute), order_by=Embedding.captured_at).label("n"))
            .where(Embedding.owner == VectorOwner.FRAME.value, Embedding.captured_at < cutoff)
            .subquery()
        )

        crops = delete(Embedding).where(Embedding.owner == VectorOwner.CROP.value, Embedding.captured_at < cutoff)
        frames = delete(Embedding).where(Embedding.id.in_(select(ranked.c.id).where(ranked.c.n > 1)))

        async with self._session.begin() as s:
            deleted = [len((await s.scalars(q.returning(Embedding.id))).all()) for q in (crops, frames)]

        return sum(deleted)

    def _pending(self, kind: IndexKind) -> list[ColumnElement[bool]]:
        has_key_frames = exists().where(Frame.event_id == Event.id, Frame.is_key)
        blocked = exists().where(JobFailure.kind == kind.value, JobFailure.key == cast(Event.id, Text), JobFailure.retry_at > func.now())
        unembedded_change = exists().where(SceneChange.event_id == Event.id, ~self._has_vector(VectorOwner.CHANGE, SceneChange.id))
        conditions = {
            IndexKind.CHANGE_VECTORS: [unembedded_change],
            IndexKind.FRAME_VECTORS: [has_key_frames, ~self._event_has(VectorOwner.FRAME)],
            IndexKind.CLIP_VECTORS: [has_key_frames, ~self._event_has(VectorOwner.CLIP)],
            IndexKind.CAPTIONS: [has_key_frames, Event.summary.is_(None), ~self._earlier_uncaptioned()],
            IndexKind.CROP_VECTORS: [has_key_frames, self._event_has(VectorOwner.CLIP), ~self._event_has(VectorOwner.CROP)],
        }

        return [Event.ended_at.is_not(None), ~blocked, *conditions[kind]]

    def _earlier_uncaptioned(self) -> ColumnElement[bool]:
        earlier = aliased(Event)
        has_key_frames = exists().where(Frame.event_id == earlier.id, Frame.is_key)
        skipped = exists().where(
            JobFailure.kind == IndexKind.CAPTIONS.value, JobFailure.key == cast(earlier.id, Text), JobFailure.retry_at > func.now()
        )

        return (
            exists()
            .where(earlier.camera_id == Event.camera_id, earlier.id < Event.id, earlier.ended_at.is_not(None), earlier.summary.is_(None))
            .where(has_key_frames, ~skipped)
        )

    def _has_vector(self, owner: VectorOwner, owner_id: InstrumentedAttribute[int]) -> ColumnElement[bool]:
        return exists().where(Embedding.owner == owner.value, Embedding.owner_id == owner_id, Embedding.model == self._model)

    def _event_has(self, owner: VectorOwner) -> ColumnElement[bool]:
        return exists().where(Embedding.event_id == Event.id, Embedding.owner == owner.value, Embedding.model == self._model)

    async def _changes(self, condition: ColumnElement[bool]) -> list[StoredChange]:
        async with self._session() as s:
            rows = await s.scalars(select(SceneChange).where(condition).order_by(SceneChange.detected_at))

            return [c.stored() for c in rows]

    async def latest_frame(self) -> StoredFrame | None:
        async with self._session() as s:
            frame = await s.scalar(select(Frame).order_by(Frame.id.desc()).limit(1))

        return frame.stored() if frame else None


def _hour_counts(timezone: str) -> Select[str, datetime, int]:
    start = func.date_trunc("hour", Event.started_at, timezone)

    return (
        select(Event.camera_id.label("camera_id"), start.label("start"), func.count().label("events"))
        .where(Event.summary.is_not(None), Event.ended_at.is_not(None))
        .group_by(Event.camera_id, start)
    )


def _day_counts(timezone: str) -> Select[str, datetime, int]:
    start = func.date_trunc("day", Summary.start, timezone)

    return (
        select(Summary.camera_id.label("camera_id"), start.label("start"), func.sum(Summary.events).label("events"))
        .where(Summary.period == Period.HOUR.value)
        .group_by(Summary.camera_id, start)
    )


def _hour_inputs(work: SummaryWork, end: datetime) -> Select[datetime, str | None, int]:
    return (
        select(Event.started_at, Event.summary, literal(1))
        .where(Event.camera_id == work.camera_id, Event.summary.is_not(None), Event.ended_at.is_not(None))
        .where(Event.started_at >= work.start, Event.started_at < end)
        .order_by(Event.started_at)
    )


def _day_inputs(work: SummaryWork, end: datetime) -> Select[datetime, str, int]:
    return (
        select(Summary.start, Summary.text, Summary.events)
        .where(Summary.camera_id == work.camera_id, Summary.period == Period.HOUR.value)
        .where(Summary.start >= work.start, Summary.start < end)
        .order_by(Summary.start)
    )


PERIOD_COUNTS: dict[Period, Callable[[str], Select[str, datetime, int]]] = {
    Period.HOUR: _hour_counts,
    Period.DAY: _day_counts,
}
SUMMARY_INPUTS: dict[Period, Callable[[SummaryWork, datetime], Select]] = {
    Period.HOUR: _hour_inputs,
    Period.DAY: _day_inputs,
}

