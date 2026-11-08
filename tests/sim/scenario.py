from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import Field

from ingest.changes import MAX_ALIGN_SHIFT
from models import Box, ChangeKind, Record

DAY_S = 86_400.0
HOUR_S = 3_600.0
QUIET_STEP_S = 300.0
WALK_S = 12.0
SETTLE_AFTER_S = 45.0
MATCH_SLACK_S = 90.0


class Item(Record):
    name: str
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    color: tuple[int, int, int]


class Place(Record):
    name: str
    x: int
    y: int


class Routine(Record):
    sunrise_h: float = 6.5
    sunset_h: float = 18.5
    lamp_on_h: float = 18.0
    lamp_off_h: float = 23.5
    tv_on_h: float = 19.0
    tv_off_h: float = 22.5
    curtain_open_h: float = 7.5
    curtain_close_h: float = 21.0
    cloudiness: float = Field(default=0.25, ge=0, le=1)
    noise: float = Field(default=2.5, ge=0)


class Drop(Record):
    kind: Literal["drop"] = "drop"
    at: float
    item: str
    place: str


class PickUp(Record):
    kind: Literal["pick_up"] = "pick_up"
    at: float
    item: str


class Move(Record):
    kind: Literal["move"] = "move"
    at: float
    item: str
    place: str


class Door(Record):
    kind: Literal["door"] = "door"
    at: float
    open: bool


class Furniture(Record):
    kind: Literal["furniture"] = "furniture"
    at: float
    dx: int
    dy: int


class Bump(Record):
    kind: Literal["bump"] = "bump"
    at: float
    dx: int
    dy: int


class Linger(Record):
    kind: Literal["linger"] = "linger"
    at: float
    until: float
    place: str


class Party(Record):
    kind: Literal["party"] = "party"
    at: float
    until: float
    guests: int = Field(ge=1, le=8)


type Action = Annotated[Drop | PickUp | Move | Door | Furniture | Bump | Linger | Party, Field(discriminator="kind")]


class Thresholds(Record):
    min_recall: float = Field(ge=0, le=1)
    max_false_per_day: float = Field(ge=0)
    max_distractor_per_day: float = Field(ge=0)


class Scenario(Record):
    name: str
    seed: int
    days: float = Field(gt=0)
    routine: Routine
    items: list[Item]
    actions: list[Action]
    thresholds: Thresholds


class Expectation(Record):
    label: str
    kinds: frozenset[ChangeKind]
    box: Box
    at: float
    item: str | None = None


class Zone(Record):
    label: str
    box: Box


class RoomState(Record):
    at: float
    items: dict[str, str | None]
    door_open: bool
    chair: tuple[int, int]
    camera: tuple[int, int]
    coords: tuple[int, int]


ANY_KIND = frozenset(ChangeKind)
ARRIVED = frozenset({ChangeKind.ADDED, ChangeKind.CHANGED, ChangeKind.MOVED})
LEFT = frozenset({ChangeKind.REMOVED, ChangeKind.CHANGED, ChangeKind.MOVED})

PLACES = {
    p.name: p
    for p in [
        Place(name="desk left", x=290, y=352),
        Place(name="desk middle", x=390, y=352),
        Place(name="desk right", x=500, y=352),
        Place(name="shelf top", x=878, y=150),
        Place(name="shelf middle", x=878, y=218),
        Place(name="couch", x=720, y=372),
        Place(name="floor", x=430, y=470),
        Place(name="side table", x=560, y=430),
    ]
}
DOOR_BOX = Box(x=40, y=70, w=110, h=262)
CHAIR = Box(x=170, y=360, w=70, h=110)
TV_BOX = Box(x=330, y=90, w=140, h=80)
WINDOW_BOX = Box(x=600, y=40, w=220, h=190)
DISTRACTORS = [Zone(label="tv", box=TV_BOX), Zone(label="window and curtain", box=WINDOW_BOX)]


def item_box(item: Item, place: str, camera: tuple[int, int] = (0, 0)) -> Box:
    spot = PLACES[place]

    return Box(x=spot.x - item.w // 2 + camera[0], y=spot.y - item.h + camera[1], w=item.w, h=item.h)


def shifted(box: Box, camera: tuple[int, int]) -> Box:
    return Box(x=max(0, box.x + camera[0]), y=max(0, box.y + camera[1]), w=box.w, h=box.h)


def ordered(scenario: Scenario) -> list[Action]:
    return sorted(scenario.actions, key=lambda a: a.at)


def state_at(scenario: Scenario, at: float) -> RoomState:
    state = RoomState(at=at, items={i.name: None for i in scenario.items}, door_open=False, chair=(0, 0), camera=(0, 0), coords=(0, 0))

    for action in ordered(scenario):
        state = _apply(state, action) if action.at <= at else state

    return state


def _apply(state: RoomState, action: Action) -> RoomState:
    match action:
        case Drop() | Move():
            return state.model_copy(update={"items": state.items | {action.item: action.place}})
        case PickUp():
            return state.model_copy(update={"items": state.items | {action.item: None}})
        case Door():
            return state.model_copy(update={"door_open": action.open})
        case Furniture():
            return state.model_copy(update={"chair": (state.chair[0] + action.dx, state.chair[1] + action.dy)})
        case Bump():
            camera = (state.camera[0] + action.dx, state.camera[1] + action.dy)
            realigned = max(abs(action.dx), abs(action.dy)) <= MAX_ALIGN_SHIFT

            return state.model_copy(update={"camera": camera, "coords": state.coords if realigned else camera})
        case Linger() | Party():
            return state


def expectations(scenario: Scenario) -> list[Expectation]:
    items = {i.name: i for i in scenario.items}
    found: list[Expectation] = []

    for action in ordered(scenario):
        before = state_at(scenario, action.at - 0.001)
        found += list(_expect(action, before, items))

    return found


def _expect(action: Action, before: RoomState, items: dict[str, Item]) -> Iterator[Expectation]:
    camera = before.coords

    match action:
        case Drop():
            yield Expectation(label=f"{action.item} dropped on {action.place}", kinds=ARRIVED, box=item_box(items[action.item], action.place, camera), at=action.at, item=action.item)
        case PickUp():
            place = before.items[action.item]
            assert place, f"{action.item} picked up at {action.at} but it is not in the room"
            yield Expectation(label=f"{action.item} picked up from {place}", kinds=LEFT, box=item_box(items[action.item], place, camera), at=action.at, item=action.item)
        case Move():
            place = before.items[action.item]
            assert place, f"{action.item} moved at {action.at} but it is not in the room"
            yield Expectation(label=f"{action.item} moved from {place}", kinds=LEFT, box=item_box(items[action.item], place, camera), at=action.at, item=action.item)
            yield Expectation(label=f"{action.item} moved to {action.place}", kinds=ARRIVED, box=item_box(items[action.item], action.place, camera), at=action.at, item=action.item)
        case Door():
            yield Expectation(label=f"door {'opened' if action.open else 'closed'}", kinds=ANY_KIND, box=shifted(DOOR_BOX, camera), at=action.at)
        case Furniture():
            old = Box(x=CHAIR.x + before.chair[0], y=CHAIR.y + before.chair[1], w=CHAIR.w, h=CHAIR.h)
            new = Box(x=old.x + action.dx, y=old.y + action.dy, w=CHAIR.w, h=CHAIR.h)
            yield Expectation(label="chair moved away", kinds=ANY_KIND, box=shifted(old, camera), at=action.at)
            yield Expectation(label="chair moved here", kinds=ANY_KIND, box=shifted(new, camera), at=action.at)
        case Bump() | Linger() | Party():
            return


def samples(scenario: Scenario) -> list[float]:
    end = scenario.days * DAY_S
    quiet = [i * QUIET_STEP_S for i in range(int(end // QUIET_STEP_S) + 1)]
    dense = [t for action in scenario.actions for t in _window(action)]

    return sorted({round(t * 2) / 2 for t in quiet + dense if 0 <= t <= end})


def _window(action: Action) -> list[float]:
    match action:
        case Linger():
            return _steps(action.at - WALK_S - 5, action.until + WALK_S + SETTLE_AFTER_S, 2.0)
        case Party():
            return _steps(action.at - 5, action.until + SETTLE_AFTER_S, 3.0)
        case _:
            return _steps(action.at - WALK_S - 3, action.at + WALK_S + SETTLE_AFTER_S, 1.0)


def _steps(start: float, end: float, step: float) -> list[float]:
    return [start + i * step for i in range(int((end - start) // step) + 1)]
