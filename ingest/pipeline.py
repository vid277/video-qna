from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from sqlalchemy.exc import InterfaceError, OperationalError

from ingest import images
from ingest.changes import ChangeTracker, SceneWatcher
from ingest.people import PersonDetector
from ingest.runtime import archive, config, store
from models import Box, Change, Episode, Snapshot

type Detected = tuple[Change, bytes, bytes, float, float]

RETRY_FIRST_S = 1.0
RETRY_MAX_S = 30.0

log = logging.getLogger(__name__)


async def persist[**P, T](call: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
    delay = RETRY_FIRST_S

    while True:
        try:
            return await call(*args, **kwargs)
        except (OperationalError, InterfaceError) as e:
            log.warning("database unavailable, retrying %s in %.0fs: %s", call.__name__, delay, str(e.orig or e).splitlines()[0])

        await asyncio.sleep(delay)
        delay = min(delay * 2, RETRY_MAX_S)


class IngestPipeline:
    def __init__(self, camera_id: str):
        self._camera_id = camera_id
        self._watcher = SceneWatcher(config)
        self._tracker = ChangeTracker(config)
        self._people = PersonDetector(config.person_score)
        self._episode: Episode | None = None
        self._last_event_id: int | None = None
        self._last_kept: datetime | None = None
        self._last_check: datetime | None = None

    async def recover(self) -> None:
        for event_id, last_frame_at in await persist(store.open_events, self._camera_id):
            await self._close(event_id, last_frame_at)
            log.info("closed event %s left open by a previous run", event_id)

    async def ingest(self, snapshot: Snapshot) -> None:
        at = snapshot.captured_at.astimezone()
        preview = await asyncio.to_thread(images.decode_preview, snapshot.image)
        seen = self._watcher.observe(preview, at.timestamp())

        if seen.started:
            self._episode = await self._open(at)

        if self._episode and (at - self._episode.started_at).total_seconds() >= config.max_event_s:
            await self._close(self._episode.event_id, at)
            self._episode = await self._open(at)

        if self._episode or _due(self._last_kept, at, config.still_keep_s):
            await self._keep(snapshot.image, at, seen.foreground)

        if _due(self._last_check, at, config.change_check_s):
            await self._check(snapshot.image, at, seen.foreground)

        if seen.settled and self._episode:
            await self._close(self._episode.event_id, at)
            self._episode = None

    async def _open(self, at: datetime) -> Episode:
        event_id = await persist(store.open_event, self._camera_id, at)
        self._last_event_id = event_id

        return Episode(event_id=event_id, started_at=at)

    async def _close(self, event_id: int, at: datetime) -> None:
        await persist(store.close_event, event_id, at)
        await persist(store.mark_key_frames, event_id, config.key_frames)

    async def _check(self, jpeg: bytes, at: datetime, foreground: list[Box]) -> None:
        self._last_check = at
        found = await asyncio.to_thread(self._detect, jpeg, at.timestamp(), foreground)
        event_id = await self._event_for(at) if found else 0

        for index, (change, before, after, changed_after, changed_by) in enumerate(found):
            before_path = archive.change_path(self._camera_id, at, event_id, index, "before")
            after_path = archive.change_path(self._camera_id, at, event_id, index, "after")
            await asyncio.to_thread(archive.write, before_path, before)
            await asyncio.to_thread(archive.write, after_path, after)

            window = (datetime.fromtimestamp(changed_after).astimezone(), datetime.fromtimestamp(changed_by).astimezone())
            await persist(store.add_change, event_id, change, before_path, after_path, *window)

    def _detect(self, jpeg: bytes, at: float, foreground: list[Box]) -> list[Detected]:
        frame = images.decode(jpeg)
        people = self._people.detect(frame)
        found = self._tracker.check(frame, at, foreground, people, self._watcher.volatile_mask(frame.shape))

        return [(change, images.encode(before), images.encode(after), since, by) for change, before, after, since, by in found]

    async def _event_for(self, at: datetime) -> int:
        current = self._episode.event_id if self._episode else self._last_event_id

        return current if current is not None else await self._standalone_event(at)

    async def _standalone_event(self, at: datetime) -> int:
        episode = await self._open(at)
        await self._close(episode.event_id, at)

        return episode.event_id

    async def _keep(self, jpeg: bytes, at: datetime, foreground: list[Box]) -> None:
        path = archive.frame_path(self._camera_id, at)
        await asyncio.to_thread(archive.write, path, jpeg)

        event_id = self._episode.event_id if self._episode else None
        await persist(store.add_frame, self._camera_id, at, path, event_id, foreground)
        self._last_kept = at


def _due(last: datetime | None, at: datetime, every_s: float) -> bool:
    return last is None or (at - last).total_seconds() >= every_s
