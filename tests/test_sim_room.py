from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ingest.room import RoomLog
from ingest.store import Store
from models import IngestConfig
from tests.sim import library
from tests.sim.room_check import check_room, perfect_tracker
from tests.sim.run import SCENARIOS, trace
from tests.sim.scenario import Scenario, expectations, samples
from tests.sim.score import score

ROOM_ACCURACY = 0.8


@pytest.fixture
async def room(config: IngestConfig, db: async_sessionmaker[AsyncSession]) -> AsyncIterator[RoomLog]:
    log = RoomLog(config.database_url, config.embedding_model)

    yield log

    await log.close()


async def room_accuracy(scenario: Scenario, perfect: bool, store: Store, room: RoomLog, db: async_sessionmaker[AsyncSession]) -> tuple[float, list[str]]:
    expected = expectations(scenario)
    found = perfect_tracker(expected) if perfect else trace(scenario)
    report = score(scenario.name, scenario.days, len(samples(scenario)), expected, found)
    result = await check_room(scenario, report, found, store, room, db)

    return result.accuracy, result.describe()




@pytest.mark.parametrize(
    "make",
    [
        library.drop_and_pick_up,
        library.night_ir,
        library.party,
        library.bump,
        library.two_weeks_steady,
        library.pan,
        library.small_item_after_bump,
        library.two_weeks,
    ],
)
async def test_room_state_matches_truth_given_perfect_changes(
    make: object, store: Store, room: RoomLog, db: async_sessionmaker[AsyncSession]
) -> None:
    scenario = make()  # type: ignore[operator]
    accuracy, lines = await room_accuracy(scenario, True, store, room, db)

    assert accuracy == 1.0, "\n".join(lines)


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n in SCENARIOS if n != "quiet_day"])
async def test_room_state_from_the_tracker(name: str, store: Store, room: RoomLog, db: async_sessionmaker[AsyncSession]) -> None:
    accuracy, lines = await room_accuracy(SCENARIOS[name](), False, store, room, db)

    assert accuracy >= ROOM_ACCURACY, "\n".join(lines)
