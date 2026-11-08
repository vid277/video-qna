from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from store.archive import FrameArchive
from store.store import Store
from tests.factory import CAMERA, T0, add_frame, frames


async def event_with_frames(store: Store, archive: FrameArchive, count: int, offset_s: int = 0) -> int:
    event_id = await store.open_event(CAMERA, T0)

    for i in range(count):
        await add_frame(store, archive, T0 + timedelta(seconds=offset_s + i), i, event_id)

    return event_id


@pytest.mark.parametrize(("count", "keys"), [(20, 8), (10, 4), (6, 3), (5, 8), (2, 8), (1, 8)])
async def test_mark_key_frames_includes_first_and_last(
    store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession], count: int, keys: int
) -> None:
    event_id = await event_with_frames(store, archive, count)
    other = await event_with_frames(store, archive, count, offset_s=1000)

    await store.mark_key_frames(event_id, keys)

    rows = [f for f in await frames(db) if f.event_id == event_id]
    marked = [i for i, f in enumerate(rows) if f.is_key]
    assert marked[0] == 0 and marked[-1] == count - 1
    assert len(marked) == min(count, keys)
    assert not [f for f in await frames(db) if f.event_id == other and f.is_key]


async def test_mark_key_frames_on_event_without_frames(
    store: Store, archive: FrameArchive, db: async_sessionmaker[AsyncSession]
) -> None:
    empty = await event_with_frames(store, archive, 0)
    await event_with_frames(store, archive, 3)

    await store.mark_key_frames(empty, 8)

    assert not [f for f in await frames(db) if f.is_key]
