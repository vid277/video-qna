from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import aiohttp
from sqlalchemy.exc import InterfaceError, OperationalError

from ingest.captioner import Captioner
from ingest.embedder import Embedder
from ingest.retention import apply_retention
from ingest.facts import facts
from ingest.room import room
from ingest.runtime import config, jobs_config, store
from ingest.summarizer import Summarizer
from jobs.scheduler import JobKind, Scheduled
from models import RoomKind, TextKind
from models.jobs import IndexKind, Period, QueueName
from openrouter import OpenRouter, OpenRouterError

OUTAGES = (aiohttp.ClientConnectionError, asyncio.TimeoutError, ConnectionError, OperationalError, InterfaceError)
UNAVAILABLE = {429, 502, 503, 529}

log = logging.getLogger(__name__)


def is_outage(error: BaseException) -> bool:
    """Only a service that is unreachable or busy pauses work; other errors count toward max_attempts."""
    return isinstance(error, OUTAGES) or (isinstance(error, OpenRouterError) and error.status in UNAVAILABLE)


def index_kinds(client: OpenRouter) -> list[JobKind]:
    embedder, captioner, summarizer = Embedder(client), Captioner(client), Summarizer(client)

    async def caption(event_id: int) -> None:
        await captioner.caption_event(await store.event_work(event_id))

    return [
        _event_kind(IndexKind.CHANGE_VECTORS, QueueName.EMBED, 60, embedder.embed_changes),
        _event_kind(IndexKind.FRAME_VECTORS, QueueName.EMBED, 50, embedder.embed_frames),
        _event_kind(IndexKind.CLIP_VECTORS, QueueName.EMBED, 40, embedder.embed_clip),
        _event_kind(IndexKind.CROP_VECTORS, QueueName.EMBED, 10, embedder.embed_crops, newer_than=_vector_cutoff, idle_only=True),
        _event_kind(IndexKind.CAPTIONS, QueueName.INDEX, 40, caption),
        _summary_kind(IndexKind.HOUR_SUMMARIES, 30, Period.HOUR, summarizer),
        _summary_kind(IndexKind.DAY_SUMMARIES, 20, Period.DAY, summarizer),
        _room_kind(RoomKind.LABELS, QueueName.INDEX, 55, lambda key: room.label(int(key), client)),
        _room_kind(RoomKind.LINKS, QueueName.SERIAL, 50, lambda key: room.link(int(key))),
        JobKind(
            name=TextKind.FACT_VECTORS.value,
            queue=QueueName.EMBED,
            priority=45,
            find=facts.pending,
            run=lambda key: facts.embed(int(key), client),
        ),
    ]


def scheduled() -> list[Scheduled]:
    return [
        Scheduled("retention", config.retention_every_s, _retain),
        Scheduled("vector_thinning", jobs_config.vector_thin_every_s, _thin_vectors),
    ]


def _event_kind(
    kind: IndexKind,
    queue: QueueName,
    priority: int,
    run: Callable[[int], Awaitable[None]],
    newer_than: Callable[[], datetime] = lambda: datetime.min.replace(tzinfo=UTC),
    idle_only: bool = False,
) -> JobKind:
    async def find(limit: int) -> list[str]:
        return [str(event_id) for event_id in await store.pending_events(kind, limit, newer_than())]

    async def handle(key: str) -> None:
        await run(int(key))

    return JobKind(name=kind.value, queue=queue, priority=priority, find=find, run=handle, idle_only=idle_only)


def _summary_kind(kind: IndexKind, priority: int, period: Period, summarizer: Summarizer) -> JobKind:
    async def find(limit: int) -> list[str]:
        return [w.key for w in await store.pending_summaries(period, jobs_config.timezone, kind.value, limit)]

    return JobKind(name=kind.value, queue=QueueName.INDEX, priority=priority, find=find, run=summarizer.summarize)


def _room_kind(kind: RoomKind, queue: QueueName, priority: int, run: Callable[[str], Awaitable[None]]) -> JobKind:
    async def find(limit: int) -> list[str]:
        return await room.pending(kind, limit)

    return JobKind(name=kind.value, queue=queue, priority=priority, find=find, run=run)


def _vector_cutoff() -> datetime:
    return datetime.now(UTC) - timedelta(days=jobs_config.vector_keep_days)


async def _retain() -> None:
    for camera_id in await store.camera_ids():
        await apply_retention(camera_id, datetime.now())


async def _thin_vectors() -> None:
    deleted = await store.thin_vectors(_vector_cutoff())

    log.info("thinned %d vectors older than %d days", deleted, jobs_config.vector_keep_days)
