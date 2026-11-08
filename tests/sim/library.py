from __future__ import annotations

import numpy as np

from tests.sim.scenario import (
    DAY_S,
    HOUR_S,
    PLACES,
    Action,
    Bump,
    Door,
    Drop,
    Furniture,
    Item,
    Linger,
    Move,
    Party,
    PickUp,
    Routine,
    Scenario,
    Thresholds,
)

ITEMS = [
    Item(name="book", w=30, h=11, color=(40, 45, 175)),
    Item(name="airpods", w=9, h=7, color=(238, 238, 238)),
    Item(name="keys", w=13, h=6, color=(80, 165, 205)),
    Item(name="mug", w=12, h=15, color=(170, 90, 40)),
    Item(name="bag", w=40, h=32, color=(45, 125, 60)),
    Item(name="bottle", w=8, h=22, color=(40, 100, 40)),
]
DAILY_ITEMS = ["book", "airpods", "keys", "mug"]
DAILY_PLACES = [p for p in PLACES if p != "floor"]
CLEARANCE_S = 600
STRICT = Thresholds(min_recall=1.0, max_false_per_day=0, max_distractor_per_day=2)
LOOSE = Thresholds(min_recall=0.75, max_false_per_day=1, max_distractor_per_day=2)


def h(day: float, hour: float) -> float:
    return day * DAY_S + hour * HOUR_S


def scenario(name: str, actions: list[Action], thresholds: Thresholds, days: float = 1, seed: int = 11) -> Scenario:
    return Scenario(name=name, seed=seed, days=days, routine=Routine(), items=ITEMS, actions=actions, thresholds=thresholds)


def quiet_day() -> Scenario:
    return scenario("quiet_day", [], STRICT)


def drop_and_pick_up() -> Scenario:
    return scenario(
        "drop_and_pick_up",
        [
            Drop(at=h(0, 10), item="book", place="desk left"),
            Drop(at=h(0, 13), item="airpods", place="couch"),
            Drop(at=h(0, 15), item="keys", place="side table"),
            Move(at=h(0, 17), item="book", place="shelf top"),
            PickUp(at=h(0, 20), item="airpods"),
            PickUp(at=h(0, 22), item="keys"),
        ],
        STRICT,
    )


def night_ir() -> Scenario:
    return scenario(
        "night_ir",
        [
            Drop(at=h(0, 2), item="airpods", place="desk middle"),
            Drop(at=h(0, 3.5), item="book", place="couch"),
            PickUp(at=h(0, 5), item="airpods"),
            Drop(at=h(0, 23.8), item="keys", place="side table"),
        ],
        STRICT,
    )


def party() -> Scenario:
    return scenario(
        "party",
        [
            Drop(at=h(0, 12), item="bottle", place="side table"),
            Party(at=h(0, 19), until=h(0, 22), guests=6),
            Drop(at=h(0, 20.2), item="bag", place="floor"),
            PickUp(at=h(0, 21), item="bottle"),
        ],
        LOOSE,
    )


def linger() -> Scenario:
    return scenario(
        "linger",
        [
            Drop(at=h(0, 11), item="book", place="desk middle"),
            Linger(at=h(0, 14), until=h(0, 14.2), place="desk middle"),
            Linger(at=h(0, 16), until=h(0, 16.5), place="couch"),
        ],
        STRICT,
    )


def furniture_and_door() -> Scenario:
    return scenario(
        "furniture_and_door",
        [
            Door(at=h(0, 9), open=True),
            Door(at=h(0, 9.7), open=False),
            Furniture(at=h(0, 11), dx=130, dy=10),
            Door(at=h(0, 18), open=True),
            Door(at=h(0, 23), open=False),
        ],
        STRICT,
    )


def bump() -> Scenario:
    return scenario(
        "bump",
        [
            Drop(at=h(0, 9), item="book", place="desk left"),
            Bump(at=h(0, 12), dx=14, dy=9),
            Drop(at=h(0, 14), item="keys", place="desk right"),
            PickUp(at=h(0, 16), item="book"),
        ],
        STRICT,
    )


def pan() -> Scenario:
    return scenario(
        "pan",
        [
            Drop(at=h(0, 9), item="airpods", place="side table"),
            Bump(at=h(0, 12), dx=40, dy=25),
            PickUp(at=h(0, 14), item="airpods"),
            Drop(at=h(0, 15), item="keys", place="desk right"),
        ],
        STRICT,
    )


def two_weeks(seed: int = 5) -> Scenario:
    return scenario("two_weeks", daily_life(14, seed), LOOSE, days=14, seed=seed)


def two_weeks_steady(seed: int = 5) -> Scenario:
    return scenario("two_weeks_steady", daily_life(14, seed, bumped=False), LOOSE, days=14, seed=seed)


def small_item_after_bump() -> Scenario:
    return scenario(
        "small_item_after_bump",
        [
            Drop(at=h(0, 9), item="airpods", place="side table"),
            Bump(at=h(0, 12), dx=12, dy=8),
            PickUp(at=h(0, 14), item="airpods"),
        ],
        STRICT,
    )


def daily_life(days: int, seed: int, bumped: bool = True) -> list[Action]:
    rng = np.random.default_rng(seed)
    where: dict[str, str | None] = dict.fromkeys(DAILY_ITEMS)
    actions: list[Action] = []

    for day in range(days):
        hours = sorted(rng.choice(np.arange(8.0, 23.0, 0.25), size=int(rng.integers(2, 5)), replace=False))

        for hour in hours:
            item = str(rng.choice(DAILY_ITEMS))
            place = str(rng.choice([p for p in DAILY_PLACES if p not in where.values()]))
            action = _daily_action(day, float(hour), item, place, where.get(item))
            where[item] = place if not isinstance(action, PickUp) else None
            actions.append(action)

        opened = h(day, float(rng.choice(np.arange(7.0, 22.0, 0.5))) + 0.1)
        door = [Door(at=opened, open=True), Door(at=opened + float(rng.uniform(0.3, 1.5)) * HOUR_S, open=False)]
        clear = all(abs(d.at - h(day, float(hour))) >= CLEARANCE_S for d in door for hour in hours)
        actions += door * clear

    return actions + [
        Linger(at=h(3, 15.1), until=h(3, 15.3), place="desk middle"),
        Party(at=h(6, 19.05), until=h(6, 22.05), guests=5),
        Drop(at=h(6, 20.3), item="bag", place="floor"),
        Furniture(at=h(9, 10.05), dx=130, dy=10),
        *[Bump(at=h(11, 12.05), dx=12, dy=8)] * bumped,
    ]


def _daily_action(day: int, hour: float, item: str, place: str, current: str | None) -> Action:
    at = h(day, hour)
    choices: dict[bool, Action] = {
        False: Drop(at=at, item=item, place=place),
        True: PickUp(at=at, item=item) if hour > 20 else Move(at=at, item=item, place=place),
    }

    return choices[current is not None]
