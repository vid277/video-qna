from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from ingest.changes import ChangeTracker, SceneWatcher
from models import Box, ChangeKind, IngestConfig
from store import images

CONFIG = IngestConfig.from_yaml(Path(__file__).with_name("ingest.yaml"))
W, H = 960, 540

type Frame = NDArray[np.uint8]
type Shot = tuple[Frame, list[Box]]
type Scenario = Callable[[np.random.Generator], Iterator[Shot]]
type Expected = list[tuple[ChangeKind, Box, float]]
type Seen = list[tuple[ChangeKind, Box, float, float]]

ROOM_SEED = 3
PHONE = Box(x=600, y=345, w=22, h=12)
KEYS = Box(x=400, y=350, w=20, h=10)
BOOK = Box(x=300, y=340, w=50, h=20)
BAG = Box(x=250, y=335, w=40, h=30)
TIME_SLACK_S = 30


def room() -> Frame:
    rng = np.random.default_rng(ROOM_SEED)
    base = np.asarray(cv2.GaussianBlur(rng.integers(70, 190, (H, W, 3), dtype=np.uint8), (0, 0), 2), np.uint8)
    cv2.rectangle(base, (150, 330), (750, 380), (60, 90, 130), -1)

    return base


ROOM = room()


def thing(img: Frame, box: Box, color: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (box.x, box.y), (box.x + box.w, box.y + box.h), color, -1)
    cv2.rectangle(img, (box.x + 2, box.y + 2), (box.x + box.w - 3, box.y + box.h - 3), tuple(255 - c for c in color), 1)


def person(img: Frame, x: int, y: int = 120) -> Box:
    cv2.rectangle(img, (x, y), (x + 90, y + 360), (50, 50, 140), -1)
    cv2.circle(img, (x + 45, y - 25), 28, (140, 160, 200), -1)

    return Box(x=max(0, x), y=y - 53, w=90, h=413)


def capture(img: Frame, rng: np.random.Generator, gain: float = 1.0, people: list[Box] | None = None) -> Shot:
    noisy = np.clip(img.astype(np.float32) * gain + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)

    return np.asarray(images.decode(images.encode(noisy)), np.uint8), people or []


def scene(rng: np.random.Generator, extras: list[tuple[Box, tuple[int, int, int]]], people: list[int]) -> Shot:
    img = ROOM.copy()
    thing(img, PHONE, (30, 30, 30))
    for box, color in extras:
        thing(img, box, color)

    return capture(img, rng, people=[person(img, x) for x in people])


def quiet_drop(rng: np.random.Generator) -> Iterator[Shot]:
    for t in range(300):
        walking = [100 + 60 * (t - 100)] if 100 <= t < 110 else []
        yield scene(rng, [(KEYS, (20, 200, 230))] * (t >= 105), walking)


def pick_up(rng: np.random.Generator) -> Iterator[Shot]:
    for t in range(300):
        walking = [700 - 60 * (t - 100)] if 100 <= t < 110 else []
        img = ROOM.copy()
        thing(img, PHONE, (30, 30, 30)) if t < 105 else None
        yield capture(img, rng, people=[person(img, x) for x in walking])


def linger(rng: np.random.Generator) -> Iterator[Shot]:
    for t in range(400):
        arriving = [100 + 40 * (t - 100)] if 100 <= t < 105 else []
        staying = [360 + int(3 * np.sin(t / 3))] if 105 <= t < 225 else []
        leaving = [360 + 60 * (t - 225)] if 225 <= t < 235 else []
        yield scene(rng, [(BOOK, (200, 80, 40))] * (t >= 105), arriving + staying + leaving)


def party(rng: np.random.Generator) -> Iterator[Shot]:
    xs = rng.integers(500, W - 90, 3)
    for t in range(900):
        xs = np.clip(xs + rng.integers(-30, 31, 3), 450, W - 90)
        guests = [int(x) for x in xs] if 60 <= t < 780 else []
        img = ROOM.copy()
        thing(img, PHONE, (30, 30, 30)) if t < 360 else None
        thing(img, BAG, (40, 160, 60)) if t >= 180 else None
        dropping = [180] if 170 <= t < 180 else []
        yield capture(img, rng, people=[person(img, x) for x in guests + dropping])


DESK_PHONE = Box(x=180, y=342, w=26, h=14)
DESK_PHONE_MOVED = Box(x=390, y=342, w=26, h=14)
DESK_BOOK = Box(x=260, y=338, w=60, h=24)


def seated(img: Frame, t: int) -> Box:
    sway = int(6 * np.sin(t / 4))
    cv2.rectangle(img, (520 + sway, 40), (920 + sway, 320), (70, 60, 120), -1)
    cv2.circle(img, (720 + sway, 20), 60, (140, 160, 200), -1)

    return Box(x=520 + sway, y=0, w=400, h=320)


def desk(rng: np.random.Generator) -> Iterator[Shot]:
    for t in range(400):
        img = ROOM.copy()
        thing(img, DESK_PHONE if t < 200 else DESK_PHONE_MOVED, (30, 30, 30))
        thing(img, DESK_BOOK, (200, 80, 40)) if t >= 100 else None
        reaching = [person(img, 300)] if 95 <= t < 100 or 195 <= t < 200 else []
        yield capture(img, rng, people=[seated(img, t), *reaching])


def noise(rng: np.random.Generator) -> Iterator[Shot]:
    for _ in range(600):
        img = ROOM.copy()
        thing(img, PHONE, (30, 30, 30))
        img[40:140, 700:900] = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
        flicker = rng.uniform(0.7, 1.3) if rng.random() < 0.3 else 1.0
        img[20:250, 20:120] = np.clip(img[20:250, 20:120] * flicker, 0, 255).astype(np.uint8)
        yield capture(img, rng)


def day_night(rng: np.random.Generator) -> Iterator[Shot]:
    for t in range(900):
        img = ROOM.copy()
        thing(img, PHONE, (30, 30, 30))
        thing(img, Box(x=420, y=345, w=24, h=14), (220, 220, 220)) if t >= 450 else None
        people = [person(img, 300 + 20 * (t - 440))] if 440 <= t < 450 else []
        night = 300 <= t < 600
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        yield capture(np.asarray(gray if night else img, np.uint8), rng, 0.6 if night else 1.0, people)


def bump(rng: np.random.Generator) -> Iterator[Shot]:
    shift = np.array([[1, 0, 40], [0, 1, 25]], np.float32)
    for t in range(500):
        img = ROOM.copy()
        thing(img, PHONE, (30, 30, 30))
        thing(img, KEYS, (20, 200, 230)) if t >= 300 else None
        people = [person(img, 300 + 30 * (t - 290))] if 290 <= t < 300 else []
        moved = cv2.warpAffine(img, shift, (W, H), borderMode=cv2.BORDER_REFLECT) if t >= 150 else img
        offset = [Box(x=b.x + 40, y=b.y + 25, w=b.w, h=b.h) for b in people]
        yield capture(np.asarray(moved, np.uint8), rng, people=offset)


def drift(rng: np.random.Generator) -> Iterator[Shot]:
    for t in range(900):
        img = ROOM.copy()
        thing(img, PHONE, (30, 30, 30))
        yield capture(img, rng, 1.0 - 0.35 * t / 900)


def run(scenario: Scenario) -> Seen:
    rng = np.random.default_rng(7)
    watcher, tracker = SceneWatcher(CONFIG), ChangeTracker(CONFIG)
    found: Seen = []

    for t, (frame, people) in enumerate(scenario(rng)):
        preview = images.decode_preview(images.encode(frame))
        seen = watcher.observe(preview, float(t))
        if t % int(CONFIG.change_check_s):
            continue

        mask = watcher.volatile_mask(frame.shape)
        found += [(c.kind, c.box, after, by) for c, _, _, after, by in tracker.check(frame, float(t), seen.foreground, people, mask)]

    return found


def overlaps(a: Box, b: Box) -> bool:
    return a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h


def matches(found: Seen, expected: Expected) -> tuple[list[str], list[str]]:
    missing = [
        f"{kind.value} {box} ~{at:.0f}s"
        for kind, box, at in expected
        if not any(k is kind and overlaps(b, box) and after - TIME_SLACK_S <= at <= by + TIME_SLACK_S for k, b, after, by in found)
    ]
    extra = [
        f"{k.value} {b} between {after:.0f}s and {by:.0f}s"
        for k, b, after, by in found
        if not any(overlaps(b, box) for _, box, _ in expected)
    ]

    return missing, extra


SHIFTED_KEYS = Box(x=KEYS.x + 40, y=KEYS.y + 25, w=KEYS.w, h=KEYS.h)

CASES: dict[str, tuple[Scenario, Expected]] = {
    "quiet_drop": (quiet_drop, [(ChangeKind.ADDED, KEYS, 106)]),
    "pick_up": (pick_up, [(ChangeKind.REMOVED, PHONE, 106)]),
    "linger": (linger, [(ChangeKind.ADDED, BOOK, 106)]),
    "party": (party, [(ChangeKind.ADDED, BAG, 180), (ChangeKind.REMOVED, PHONE, 360)]),
    "desk": (
        desk,
        [(ChangeKind.ADDED, DESK_BOOK, 100), (ChangeKind.MOVED, DESK_PHONE_MOVED, 200)],
    ),
    "noise": (noise, []),
    "day_night": (day_night, [(ChangeKind.ADDED, Box(x=420, y=345, w=24, h=14), 450)]),
    "bump": (bump, [(ChangeKind.ADDED, SHIFTED_KEYS, 300)]),
    "drift": (drift, []),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", CASES)
def test_tracker_scenario(name: str) -> None:
    scenario, expected = CASES[name]
    missing, extra = matches(run(scenario), expected)

    assert not missing and not extra, f"missing {missing}, unexpected {extra}"


PHOTO = Path(cv2.__file__).parents[1] / "ultralytics" / "assets" / "bus.jpg"


@pytest.mark.skipif(not PHOTO.exists(), reason="sample photo not installed")
def test_person_detector_finds_people() -> None:
    from ingest.people import PersonDetector

    people = PersonDetector(CONFIG.person_score).detect(np.asarray(cv2.imread(str(PHOTO)), np.uint8))

    assert len(people) >= 3
    assert all(b.h > b.w for b in people)
