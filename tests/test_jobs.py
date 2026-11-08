from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp
import pytest
from huey import Huey, MemoryHuey
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from jobs.scheduler import JobKind, Scheduled, Scheduler
from models import (
    Box,
    Caption,
    Change,
    ChangeKind,
    EmbeddingRow,
    IngestConfig,
    VectorOwner,
)
from models.jobs import (
    Backoff,
    IndexKind,
    Job,
    JobsConfig,
    Period,
    QueueName,
    QueueSettings,
    SummaryWork,
)
from openrouter import OpenRouterError
from store.archive import FrameArchive
from store.db import Embedding, Fact
from store.store import Store
from tests.factory import CAMERA, T0, add_frame, mark_key

VECTOR = [0.1] * 1024
BACKOFF = Backoff(max_attempts=2, first_s=60, max_s=3600)


def jobs_config(tmp_path: Path, **overrides: object) -> JobsConfig:
    config = JobsConfig(
        queue_file=tmp_path / "jobs.db",
        queues={
            QueueName.EMBED: QueueSettings(workers=1, max_inflight=3),
            QueueName.INDEX: QueueSettings(workers=2, max_inflight=3),
            QueueName.SERIAL: QueueSettings(workers=1, max_inflight=1),
            QueueName.MAINTENANCE: QueueSettings(workers=1, max_inflight=1),
        },
        dispatch_every_s=0.05,
        max_attempts=3,
        backoff_first_s=60,
        backoff_max_s=3600,
        outage_pause_s=60,
        timezone="UTC",
        vector_thin_every_s=3600,
        vector_keep_days=30,
    )

    return config.model_copy(update=overrides)


def memory_queues() -> dict[QueueName, Huey]:
    return {name: MemoryHuey(name.value, results=False) for name in QueueName}


async def closed_event(store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession], start: datetime) -> int:
    event_id = await store.open_event(CAMERA, start)

    for i in range(2):
        await add_frame(store, archive, start + timedelta(seconds=i), i, event_id)
        await mark_key(db, start + timedelta(seconds=i))

    await store.add_change(event_id, Change(kind=ChangeKind.ADDED, box=Box(x=1, y=1, w=4, h=4)), "b.jpg", "a.jpg", start, start)
    await store.close_event(event_id, start + timedelta(seconds=5))

    return event_id


def row(owner: VectorOwner, owner_id: int, event_id: int, at: datetime, part: int = 0) -> EmbeddingRow:
    return EmbeddingRow(owner=owner, owner_id=owner_id, part=part, camera_id=CAMERA, event_id=event_id, captured_at=at, vector=VECTOR)


async def vector_count(db: async_sessionmaker[AsyncSession], owner: VectorOwner) -> int:
    async with db() as s:
        return int(await s.scalar(select(func.count()).where(Embedding.owner == owner.value)) or 0)


async def pending(store: Store) -> dict[IndexKind, list[int]]:
    kinds = [IndexKind.CHANGE_VECTORS, IndexKind.FRAME_VECTORS, IndexKind.CLIP_VECTORS, IndexKind.CAPTIONS, IndexKind.CROP_VECTORS]

    return {k: await store.pending_events(k, 10) for k in kinds}


async def test_pending_work_follows_what_is_missing(store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession]) -> None:
    event_id = await closed_event(store, archive, db, T0)
    await store.open_event(CAMERA, T0 + timedelta(minutes=5))

    before = await pending(store)
    assert before == {
        IndexKind.CHANGE_VECTORS: [event_id],
        IndexKind.FRAME_VECTORS: [event_id],
        IndexKind.CLIP_VECTORS: [event_id],
        IndexKind.CAPTIONS: [event_id],
        IndexKind.CROP_VECTORS: [],
    }

    change = (await store.event_changes(event_id))[0]
    frames = await store.key_frames(event_id)
    await store.replace_embeddings([row(VectorOwner.CHANGE, change.id, event_id, T0)])
    await store.replace_embeddings([row(VectorOwner.FRAME, f.id, event_id, T0) for f in frames] + [row(VectorOwner.CLIP, event_id, event_id, T0)])
    await store.set_caption(event_id, Caption(summary="a book was put down", new=["put a book down"], facts=["a book is on the desk"], objects=["book"]))

    after = await pending(store)
    assert after == {
        IndexKind.CHANGE_VECTORS: [],
        IndexKind.FRAME_VECTORS: [],
        IndexKind.CLIP_VECTORS: [],
        IndexKind.CAPTIONS: [],
        IndexKind.CROP_VECTORS: [event_id],
    }

    backlog = {b.kind: b.pending for b in await store.index_backlog()}
    assert backlog == {k: int(k is IndexKind.CROP_VECTORS) for k in after}


async def test_captions_wait_for_the_previous_event(store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession]) -> None:
    first = await closed_event(store, archive, db, T0)
    second = await closed_event(store, archive, db, T0 + timedelta(minutes=2))
    third = await closed_event(store, archive, db, T0 + timedelta(minutes=4))

    assert await store.pending_events(IndexKind.CAPTIONS, 10) == [first]

    await store.set_caption(first, Caption(summary="typing at the desk", new=["sat down at the desk"], facts=[], objects=[]))
    assert await store.pending_events(IndexKind.CAPTIONS, 10) == [second]

    await store.record_failure(IndexKind.CAPTIONS.value, str(second), "ValueError('bad sheet')", Backoff(max_attempts=1, first_s=60, max_s=60))
    assert await store.pending_events(IndexKind.CAPTIONS, 10) == [third]


async def test_caption_deltas_are_stored_searchable_and_shown(store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession]) -> None:
    event_id = await closed_event(store, archive, db, T0)
    caption = Caption(
        summary="a person keeps typing at the desk",
        new=["drank water from a glass", "put phone on the desk"],
        facts=["a person is typing at the desk", "put phone on the desk"],
        objects=["glass", "phone"],
    )

    await store.set_caption(event_id, caption)

    [digest] = await store.events_between(T0 - timedelta(minutes=1), T0 + timedelta(minutes=1))
    assert digest.new == ["drank water from a glass", "put phone on the desk"]

    hits = await store.search_facts("drank water", None, None, 10)
    assert [h.text for h in hits] == ["drank water from a glass"]

    async with db() as s:
        facts = (await s.scalars(select(Fact.text).where(Fact.event_id == event_id))).all()
    assert sorted(facts) == ["a person is typing at the desk", "drank water from a glass", "put phone on the desk"]


async def test_pending_events_are_newest_first(store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession]) -> None:
    older = await closed_event(store, archive, db, T0)
    newer = await closed_event(store, archive, db, T0 + timedelta(hours=1))

    assert await store.pending_events(IndexKind.FRAME_VECTORS, 10) == [newer, older]
    assert await store.pending_events(IndexKind.FRAME_VECTORS, 10, newer_than=T0 + timedelta(minutes=30)) == [newer]


async def test_replacing_embeddings_never_double_counts(
    store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession], config: IngestConfig
) -> None:
    event_id = await closed_event(store, archive, db, T0)
    rows = [row(VectorOwner.CLIP, event_id, event_id, T0)]

    await store.replace_embeddings(rows)
    await store.replace_embeddings(rows)
    assert await vector_count(db, VectorOwner.CLIP) == 1

    switched = Store(config.database_url, "other/model")
    try:
        assert event_id in await switched.pending_events(IndexKind.CLIP_VECTORS, 10)

        await switched.replace_embeddings(rows)
        assert await vector_count(db, VectorOwner.CLIP) == 1
        assert event_id not in await switched.pending_events(IndexKind.CLIP_VECTORS, 10)
        assert event_id in await store.pending_events(IndexKind.CLIP_VECTORS, 10)
    finally:
        await switched.close()


async def test_failing_events_back_off_then_give_up(store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession]) -> None:
    event_id = await closed_event(store, archive, db, T0)
    kind, key = IndexKind.FRAME_VECTORS, str(event_id)

    await store.record_failure(kind.value, key, "ValueError('bad image')", BACKOFF)
    first = (await store.failures())[0]
    assert first.attempts == 1 and first.retry_at < datetime.now(UTC) + timedelta(minutes=2)
    assert await store.pending_events(kind, 10) == []
    assert await store.pending_events(IndexKind.CAPTIONS, 10) == [event_id]

    await store.record_failure(kind.value, key, "ValueError('bad image')", BACKOFF)
    assert (await store.failures())[0].retry_at.astimezone(UTC).year == 9000

    await store.clear_failure(kind.value, key)
    assert await store.pending_events(kind, 10) == [event_id]


async def test_thin_vectors_keeps_clips_changes_and_one_frame_a_minute(
    store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession]
) -> None:
    event_id = await closed_event(store, archive, db, T0)
    old = [T0, T0 + timedelta(seconds=10), T0 + timedelta(seconds=20), T0 + timedelta(minutes=1)]
    recent = T0 + timedelta(days=60)

    await store.replace_embeddings(
        [row(VectorOwner.FRAME, i, event_id, at) for i, at in enumerate(old)]
        + [row(VectorOwner.FRAME, 99, event_id, recent), row(VectorOwner.CROP, 1, event_id, T0, part=1)]
        + [row(VectorOwner.CLIP, event_id, event_id, T0), row(VectorOwner.CHANGE, 1, event_id, T0)]
    )

    deleted = await store.thin_vectors(T0 + timedelta(days=30))

    assert deleted == 3
    assert await vector_count(db, VectorOwner.FRAME) == 3
    assert await vector_count(db, VectorOwner.CROP) == 0
    assert await vector_count(db, VectorOwner.CLIP) == 1
    assert await vector_count(db, VectorOwner.CHANGE) == 1


async def test_hour_and_day_summaries_follow_captions(store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession]) -> None:
    event_id = await closed_event(store, archive, db, T0)
    kind = IndexKind.HOUR_SUMMARIES.value
    assert await store.pending_summaries(Period.HOUR, "UTC", kind, 10) == []

    await store.set_caption(event_id, Caption(summary="a book was put on the desk", new=[], facts=[], objects=["book"]))
    hour = SummaryWork(camera_id=CAMERA, period=Period.HOUR, start=T0)
    assert await store.pending_summaries(Period.HOUR, "UTC", kind, 10) == [hour]

    inputs = await store.summary_inputs(hour, T0 + timedelta(hours=1))
    assert [i.text for i in inputs] == ["a book was put on the desk"]

    await store.set_summary(hour, T0 + timedelta(hours=1), "book left on desk", 1)
    assert await store.pending_summaries(Period.HOUR, "UTC", kind, 10) == []

    day = SummaryWork(camera_id=CAMERA, period=Period.DAY, start=T0.replace(hour=0))
    assert await store.pending_summaries(Period.DAY, "UTC", IndexKind.DAY_SUMMARIES.value, 10) == [day]

    other = await closed_event(store, archive, db, T0 + timedelta(minutes=20))
    await store.set_caption(other, Caption(summary="the book was picked up", new=[], facts=[], objects=["book"]))
    assert await store.pending_summaries(Period.HOUR, "UTC", kind, 10) == [hour]


async def test_dispatch_takes_highest_priority_first_and_never_duplicates(tmp_path: Path) -> None:
    async def low(_limit: int) -> list[str]:
        return ["1", "2", "3"]

    async def high(_limit: int) -> list[str]:
        return ["x"]

    async def run(_key: str) -> None: ...

    queues = memory_queues()
    kinds = [JobKind("low", QueueName.INDEX, 10, low, run), JobKind("high", QueueName.INDEX, 20, high, run)]
    scheduler = Scheduler(jobs_config(tmp_path), queues, kinds, [], FakeFailures(), lambda _e: False)

    await scheduler.dispatch_once()
    await scheduler.dispatch_once()

    queued = [(t.args[0], t.priority) for t in queues[QueueName.INDEX].pending()]
    assert sorted(queued, key=lambda q: -q[1]) == [
        (Job(kind="high", key="x"), 20),
        (Job(kind="low", key="1"), 10),
        (Job(kind="low", key="2"), 10),
    ]


async def test_scheduler_records_failures_pauses_on_outages_and_runs_schedules(tmp_path: Path) -> None:
    done: list[str] = []
    failures = FakeFailures()

    async def one(_limit: int) -> list[str]:
        return ["1"]

    async def unfinished(_limit: int) -> list[str]:
        return [] if "1" in done else ["1"]

    async def unrecorded(_limit: int) -> list[str]:
        return [] if failures.recorded else ["1"]

    async def works(key: str) -> None:
        done.append(key)

    async def broken(_key: str) -> None:
        raise ValueError("corrupt image")

    async def offline(_key: str) -> None:
        raise aiohttp.ClientConnectionError("openrouter unreachable")

    async def tick() -> None:
        done.append("tick")

    kinds = [
        JobKind("works", QueueName.EMBED, 30, unfinished, works),
        JobKind("broken", QueueName.INDEX, 20, unrecorded, broken),
        JobKind("offline", QueueName.INDEX, 10, one, offline),
    ]

    def is_outage(e: BaseException) -> bool:
        return isinstance(e, aiohttp.ClientError)

    scheduler = Scheduler(jobs_config(tmp_path), memory_queues(), kinds, [Scheduled("tick", 3600, tick)], failures, is_outage)

    task = asyncio.create_task(scheduler.run())
    try:
        async with asyncio.timeout(10):
            while not ({"1", "tick"} <= set(done) and failures.recorded and scheduler._paused_until):
                await asyncio.sleep(0.05)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert done.count("tick") == 1
    assert failures.recorded == [("broken", "1")]
    assert ("works", "1") in failures.cleared
    assert not [c for c in failures.cleared if c[0] != "works"]


async def test_idle_only_kinds_wait_until_their_queue_is_empty(tmp_path: Path) -> None:
    urgent: list[str] = ["c1"]

    async def changes(_limit: int) -> list[str]:
        return list(urgent)

    async def crops(_limit: int) -> list[str]:
        return ["e1", "e2"]

    async def captions(_limit: int) -> list[str]:
        return ["e9"]

    async def run(_key: str) -> None: ...

    queues = memory_queues()
    kinds = [
        JobKind("changes", QueueName.EMBED, 60, changes, run),
        JobKind("crops", QueueName.EMBED, 10, crops, run, idle_only=True),
        JobKind("captions", QueueName.INDEX, 40, captions, run),
    ]
    scheduler = Scheduler(jobs_config(tmp_path), queues, kinds, [], FakeFailures(), lambda _e: False)

    await scheduler.dispatch_once()
    assert {j.kind for j in scheduler.inflight[QueueName.EMBED]} == {"changes"}
    assert {j.kind for j in scheduler.inflight[QueueName.INDEX]} == {"captions"}

    urgent.clear()
    scheduler._inflight[QueueName.EMBED].clear()
    await scheduler.dispatch_once()
    assert sorted((j.kind, j.key) for j in scheduler.inflight[QueueName.EMBED]) == [("crops", "e1"), ("crops", "e2")]


async def test_fast_serial_jobs_run_back_to_back_not_once_per_tick(tmp_path: Path) -> None:
    order: list[str] = []
    backlog = [str(i) for i in range(20)]

    async def oldest(limit: int) -> list[str]:
        return [k for k in backlog if k not in order][:limit]

    async def link(key: str) -> None:
        order.append(key)

    config = jobs_config(tmp_path, dispatch_every_s=5.0)
    scheduler = Scheduler(config, memory_queues(), [JobKind("links", QueueName.SERIAL, 50, oldest, link)], [], FakeFailures(), lambda _e: False)

    task = asyncio.create_task(scheduler.run())
    try:
        async with asyncio.timeout(4):
            while len(order) < len(backlog):
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert order == backlog


class FakeFailures:
    def __init__(self) -> None:
        self.recorded: list[tuple[str, str]] = []
        self.cleared: list[tuple[str, str]] = []

    async def record_failure(self, kind: str, key: str, error: str, backoff: Backoff) -> None:
        self.recorded.append((kind, key))

    async def clear_failure(self, kind: str, key: str) -> None:
        self.cleared.append((kind, key))


@pytest.fixture(autouse=True)
def _quiet_huey(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="huey")


@pytest.mark.parametrize(
    ("error", "outage"),
    [
        (OpenRouterError(503, "unavailable"), True),
        (OpenRouterError(429, "rate limited"), True),
        (OpenRouterError(500, "bad input"), False),
        (OpenRouterError(400, "bad request"), False),
        (asyncio.TimeoutError(), True),
        (ValueError("bad image"), False),
    ],
)
def test_only_unreachable_or_busy_services_count_as_outages(error: BaseException, outage: bool) -> None:
    from background.tasks import is_outage

    assert is_outage(error) is outage
