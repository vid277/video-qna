from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import numpy as np
import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ingest.db import EMBEDDING_DIM, SceneChange
from ingest.room import RoomLog
from ingest.store import Store
from models import Box, Change, ChangeKind, EmbeddingRow, IngestConfig, RoomKind, VectorOwner
from tests.factory import CAMERA, T0

COUCH = Box(x=100, y=100, w=40, h=20)
TABLE = Box(x=600, y=300, w=40, h=20)
FLOOR = Box(x=300, y=500, w=12, h=8)


def vector(seed: int) -> list[float]:
    v = np.random.default_rng(seed).normal(size=EMBEDDING_DIM)

    return list(v / np.linalg.norm(v))


def near(seed: int) -> list[float]:
    base = np.asarray(vector(seed)) + 0.2 * np.random.default_rng(seed + 1000).normal(size=EMBEDDING_DIM) / np.sqrt(EMBEDDING_DIM)

    return list(base / np.linalg.norm(base))


@pytest.fixture
async def room(config: IngestConfig, db: async_sessionmaker[AsyncSession]) -> AsyncIterator[RoomLog]:
    log = RoomLog(config.database_url, config.embedding_model)

    yield log

    await log.close()


async def change(
    store: Store, db: async_sessionmaker[AsyncSession], kind: ChangeKind, box: Box, minute: int, label: str, look: list[float]
) -> int:
    at = T0 + timedelta(minutes=minute)
    event_id = await store.open_event(CAMERA, at)
    await store.add_change(event_id, Change(kind=kind, box=box), "before.jpg", "after.jpg", at - timedelta(seconds=30), at)

    async with db.begin() as s:
        change_id = (await s.execute(update(SceneChange).where(SceneChange.event_id == event_id).values(category="object", label=label, place="here").returning(SceneChange.id))).scalar_one()

    row = EmbeddingRow(owner=VectorOwner.CHANGE, owner_id=change_id, part=0, camera_id=CAMERA, event_id=event_id, captured_at=at, vector=look)
    await store.replace_embeddings([row])

    return change_id


async def link_all(room: RoomLog) -> None:
    for key in await room.pending(RoomKind.LINKS, 100):
        await room.link(int(key))


async def test_items_follow_objects_through_moves_and_pickups(store: Store, db: async_sessionmaker[AsyncSession], room: RoomLog) -> None:
    book_on_couch = await change(store, db, ChangeKind.ADDED, COUCH, 0, "red book", vector(1))
    airpods = await change(store, db, ChangeKind.ADDED, FLOOR, 5, "AirPods case", vector(2))
    book_taken = await change(store, db, ChangeKind.REMOVED, COUCH, 10, "red book", near(1))
    airpods_taken = await change(store, db, ChangeKind.REMOVED, FLOOR, 20, "AirPods case", near(2))
    book_on_table = await change(store, db, ChangeKind.ADDED, TABLE, 30, "red book", near(1))

    await link_all(room)

    book = f"item:{book_on_couch}"
    history = await room.history([book_on_couch])
    assert [(c.ref, c.kind) for c in history] == [
        (f"change:{book_on_couch}", ChangeKind.ADDED),
        (f"change:{book_taken}", ChangeKind.REMOVED),
        (f"change:{book_on_table}", ChangeKind.ADDED),
    ]
    assert {c.item_ref for c in await room.history([airpods])} == {f"item:{airpods}"}
    assert airpods_taken not in {int(c.ref.split(":")[1]) for c in history}

    at = {minute: [i.item_ref for i in await room.state(T0 + timedelta(minutes=minute, seconds=1))] for minute in (6, 15, 25, 31)}
    assert sorted(at[6]) == sorted([book, f"item:{airpods}"])
    assert at[15] == [f"item:{airpods}"]
    assert at[25] == []
    assert at[31] == [book]

    assert await room.items_labelled("book", 5) == [book_on_couch]


async def test_a_different_object_gets_its_own_item(store: Store, db: async_sessionmaker[AsyncSession], room: RoomLog) -> None:
    first = await change(store, db, ChangeKind.ADDED, COUCH, 0, "red book", vector(1))
    await change(store, db, ChangeKind.REMOVED, COUCH, 10, "red book", near(1))
    other = await change(store, db, ChangeKind.ADDED, TABLE, 20, "mug", vector(9))

    await link_all(room)

    assert {i.item_ref for i in await room.state(None)} == {f"item:{other}"}
    assert other != first


def test_label_replies_with_plain_string_enums_are_accepted() -> None:
    from ingest.room import parse_label

    label = parse_label({"category": "object", "label": "sheet of paper", "place": "on the desk"})

    assert label.category.value == "object"


async def test_links_wait_for_earlier_changes_to_be_labelled(store: Store, db: async_sessionmaker[AsyncSession], room: RoomLog) -> None:
    added = await change(store, db, ChangeKind.ADDED, FLOOR, 0, "AirPods case", vector(2))
    removed = await change(store, db, ChangeKind.REMOVED, FLOOR, 10, "AirPods case", near(2))

    async with db.begin() as s:
        await s.execute(update(SceneChange).where(SceneChange.id == added).values(category=None, label=None))

    assert await room.pending(RoomKind.LINKS, 10) == []

    async with db.begin() as s:
        await s.execute(update(SceneChange).where(SceneChange.id == added).values(category="object", label="AirPods case"))
    await link_all(room)

    assert {c.item_ref for c in await room.history([added])} == {f"item:{added}"}
    assert [c.ref for c in await room.history([added])] == [f"change:{added}", f"change:{removed}"]
