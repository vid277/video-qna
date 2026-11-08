from __future__ import annotations

import math
import zlib
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ingest.db import EMBEDDING_DIM, SceneChange
from ingest.room import RoomLog
from ingest.store import Store
from models import Box, Change, ChangeKind, EmbeddingRow, Record, RoomKind, VectorOwner
from tests.sim.scenario import DAY_S, PLACES, SETTLE_AFTER_S, Expectation, Scenario, ordered, state_at
from tests.sim.score import Found, Report, clock

CAMERA = "sim"
EPOCH = datetime(2026, 10, 1, tzinfo=UTC)
NEAR_PX = 25
UNKNOWN = "unknown thing"
LINK_BATCH = 200


class Checkpoint(Record):
    at: float
    truth: dict[str, str | None]
    seen: dict[str, str | None]

    @property
    def wrong(self) -> list[str]:
        return [f"{item}: saw {self.seen[item] or 'gone'}, truly {self.truth[item] or 'gone'}" for item in self.truth if self.seen[item] != self.truth[item]]


class RoomReport(Record):
    scenario: str
    checkpoints: list[Checkpoint]

    @property
    def accuracy(self) -> float:
        answers = [c.seen[item] == c.truth[item] for c in self.checkpoints for item in c.truth]

        return sum(answers) / len(answers) if answers else 1.0

    def describe(self) -> list[str]:
        return [
            f"{self.scenario} room state: {self.accuracy:.2f} of item answers right over {len(self.checkpoints)} checkpoints",
            *[f"  {clock(c.at)}  {w}" for c in self.checkpoints for w in c.wrong],
        ]


def when(at: float) -> datetime:
    return EPOCH + timedelta(seconds=at)


def perfect_tracker(expected: list[Expectation]) -> list[Found]:
    kinds = {True: ChangeKind.ADDED, False: ChangeKind.REMOVED}

    return [
        Found(kind=kinds[ChangeKind.ADDED in e.kinds], box=e.box, source=None, changed_after=e.at - 5, changed_by=e.at, detected_at=e.at + 25 + i * 1e-3)
        for i, e in enumerate(expected)
        if e.item
    ]


async def check_room(
    scenario: Scenario, report: Report, found: list[Found], store: Store, room: RoomLog, db: async_sessionmaker[AsyncSession]
) -> RoomReport:
    items = {h.found: h.expectation.item for h in report.hits if h.found and h.expectation.item}

    for f in sorted(found, key=lambda f: f.detected_at):
        place = _place_of(f.box, state_at(scenario, f.detected_at).coords)
        await _record(store, db, f, items.get(f) or UNKNOWN, place)

    for _ in range(math.ceil(len(found) / LINK_BATCH) + 1):
        for key in await room.pending(RoomKind.LINKS, LINK_BATCH):
            await room.link(int(key))

    names = [i.name for i in scenario.items]
    checkpoints = [await _checkpoint(scenario, room, names, at) for at in checkpoint_times(scenario)]

    return RoomReport(scenario=scenario.name, checkpoints=checkpoints)


def checkpoint_times(scenario: Scenario) -> list[float]:
    after_actions = [a.at + SETTLE_AFTER_S + 60 for a in ordered(scenario)]
    day_ends = [(d + 1) * DAY_S - 1 for d in range(math.ceil(scenario.days))]

    return sorted({t for t in after_actions + day_ends if t <= scenario.days * DAY_S})


async def _checkpoint(scenario: Scenario, room: RoomLog, names: list[str], at: float) -> Checkpoint:
    truth = state_at(scenario, at)
    present = await room.state(when(at))
    places = {name: sorted(i.place or "unplaced" for i in present if i.label == name) for name in names}

    return Checkpoint(at=at, truth=truth.items, seen={name: _answer(found) for name, found in places.items()})


def _answer(places: list[str]) -> str | None:
    match places:
        case []:
            return None
        case [place]:
            return place
        case _:
            return f"several: {', '.join(places)}"


def _place_of(box: Box, camera: tuple[int, int]) -> str:
    bottom = (box.x + box.w / 2 - camera[0], box.y + box.h - camera[1])
    distance, place = min((math.dist(bottom, (p.x, p.y)), p.name) for p in PLACES.values())

    return place if distance <= NEAR_PX else "elsewhere"


async def _record(store: Store, db: async_sessionmaker[AsyncSession], found: Found, label: str, place: str) -> int:
    event_id = await store.open_event(CAMERA, when(found.changed_after))
    change = Change(kind=found.kind, box=found.box, source=found.source)
    await store.add_change(event_id, change, "before.jpg", "after.jpg", when(found.changed_after), when(found.detected_at))

    async with db.begin() as s:
        query = update(SceneChange).where(SceneChange.event_id == event_id).values(category="object", label=label, place=place).returning(SceneChange.id)
        change_id = (await s.execute(query)).scalar_one()

    row = EmbeddingRow(
        owner=VectorOwner.CHANGE,
        owner_id=change_id,
        part=0,
        camera_id=CAMERA,
        event_id=event_id,
        captured_at=when(found.detected_at),
        vector=_look(label, change_id),
    )
    await store.replace_embeddings([row])

    return change_id


def _look(label: str, change_id: int) -> list[float]:
    seed = zlib.crc32(label.encode()) if label != UNKNOWN else 1_000_000 + change_id
    base = np.random.default_rng(seed).normal(size=EMBEDDING_DIM)
    jitter = np.random.default_rng(seed * 7 + change_id).normal(size=EMBEDDING_DIM) * 0.15
    vector = base / np.linalg.norm(base) + jitter / np.sqrt(EMBEDDING_DIM)

    return list(vector / np.linalg.norm(vector))
