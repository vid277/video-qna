from __future__ import annotations

import bisect
import math
import zlib

import cv2
import numpy as np
from numpy.typing import NDArray

from ingest import images
from models import Box
from tests.sim.scenario import (
    CHAIR,
    DAY_S,
    DOOR_BOX,
    HOUR_S,
    PLACES,
    TV_BOX,
    WALK_S,
    WINDOW_BOX,
    Door,
    Drop,
    Furniture,
    Linger,
    Move,
    Party,
    PickUp,
    RoomState,
    Scenario,
    item_box,
    ordered,
    state_at,
)

W, H = 960, 540
DOOR_X = DOOR_BOX.x + DOOR_BOX.w // 2
PERSON_W, PERSON_H = 70, 290
STAND_OFFSET = 48
IR_LEVEL = 0.2
LAMP_AT = (205, 190)

type Image = NDArray[np.uint8]
type Shot = tuple[Image, list[Box]]

_ROWS = np.arange(H, dtype=np.float32)[:, None]
_COLS = np.arange(W, dtype=np.float32)[None, :]


def _hash(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def _falloff(cx: float, cy: float, sx: float, sy: float) -> NDArray[np.float32]:
    return np.exp(-(((_COLS - cx) / sx) ** 2) - ((_ROWS - cy) / sy) ** 2).astype(np.float32)


WINDOW_LIGHT = _falloff(WINDOW_BOX.x + WINDOW_BOX.w / 2, 260, 420, 300)
LAMP_LIGHT = _falloff(*LAMP_AT, 380, 300)
IR_VIGNETTE = 0.45 + 0.55 * _falloff(W / 2, H * 0.55, 620, 420)


class Renderer:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self._items = {i.name: i for i in scenario.items}
        self._actions = ordered(scenario)
        self._times = [a.at for a in self._actions]
        self._states = [state_at(scenario, a.at) for a in self._actions]
        self._initial = state_at(scenario, -1.0)
        self._room = _room(scenario.seed)

    def state(self, at: float) -> RoomState:
        index = bisect.bisect_right(self._times, at)

        return self._states[index - 1] if index else self._initial

    def frame(self, at: float) -> Shot:
        state = self.state(at)
        light = self.light(at)
        scene = self._room.copy()

        self._draw_window(scene, light)
        self._draw_curtain(scene, at)
        self._draw_door(scene, state.door_open)
        self._draw_tv(scene, at)
        _chair(scene, state.chair)

        for name, place in [(n, p) for n, p in state.items.items() if p]:
            _thing(scene, item_box(self._items[name], place), self._items[name].color)

        people = [_person(scene, x, y, who) for x, y, who in self._people(at)]
        lit = self._light(scene, at, light)
        moved = _shift(lit, state.camera)
        grain = np.random.default_rng(_hash(self.scenario.seed, round(at * 2))).normal(0, self._noise(light), moved.shape)
        noisy = np.clip(moved + grain, 0, 255).astype(np.uint8)
        camera = state.camera

        return np.asarray(images.decode(images.encode(noisy)), np.uint8), [_clip(Box(x=b.x + camera[0], y=b.y + camera[1], w=b.w, h=b.h)) for b in people]

    def light(self, at: float) -> float:
        routine = self.scenario.routine
        hour = (at % DAY_S) / HOUR_S
        daylight = _daylight(hour, routine.sunrise_h, routine.sunset_h) * self._clouds(at)
        lamp = float(routine.lamp_on_h <= hour < routine.lamp_off_h)

        return 0.06 + 0.9 * daylight * (0.3 + 0.7 * self.curtain(at)) + 0.5 * lamp

    def night(self, at: float) -> bool:
        return self.light(at) < IR_LEVEL

    def curtain(self, at: float) -> float:
        routine = self.scenario.routine
        hour = (at % DAY_S) / HOUR_S

        return float(routine.curtain_open_h <= hour < routine.curtain_close_h)

    def tv_on(self, at: float) -> bool:
        routine = self.scenario.routine
        hour = (at % DAY_S) / HOUR_S

        return routine.tv_on_h <= hour < routine.tv_off_h

    def _clouds(self, at: float) -> float:
        phase = self.scenario.seed % 97
        wave = 0.5 + 0.5 * math.sin(at / 1700 + phase) * math.sin(at / 4100 + 2 * phase)

        return 1.0 - self.scenario.routine.cloudiness * wave

    def _noise(self, light: float) -> float:
        return self.scenario.routine.noise * (2.4 if light < IR_LEVEL else 1.0)

    def _light(self, scene: Image, at: float, light: float) -> NDArray[np.float32]:
        if light < IR_LEVEL:
            gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY).astype(np.float32) * IR_VIGNETTE * 0.95
            return np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)

        routine = self.scenario.routine
        hour = (at % DAY_S) / HOUR_S
        daylight = _daylight(hour, routine.sunrise_h, routine.sunset_h) * self._clouds(at) * (0.3 + 0.7 * self.curtain(at))
        lamp = float(routine.lamp_on_h <= hour < routine.lamp_off_h)
        sun_x = W * (0.9 - 0.75 * min(1.0, max(0.0, (hour - routine.sunrise_h) / (routine.sunset_h - routine.sunrise_h))))
        sun = _falloff(sun_x, 450, 150, 60) * 0.35 * daylight
        exposure = min(2.2, max(0.8, 0.75 / math.sqrt(max(light, 0.2))))
        gain = (0.25 + 0.6 * daylight * WINDOW_LIGHT + 0.35 * daylight + sun + 0.55 * lamp * LAMP_LIGHT) * exposure
        tint = np.array([1.0 - 0.12 * lamp, 1.0 - 0.04 * lamp, 1.0 + 0.06 * lamp], np.float32)

        return (scene.astype(np.float32) * gain[:, :, None] * tint).astype(np.float32)

    def _draw_window(self, scene: Image, light: float) -> None:
        b = WINDOW_BOX
        sky = np.array([235, 215, 190], np.float32) * min(1.0, 0.1 + light) + np.array([40, 25, 20], np.float32) * (light < IR_LEVEL)
        scene[b.y + 6 : b.y + b.h - 6, b.x + 6 : b.x + b.w - 6] = np.clip(sky, 0, 255).astype(np.uint8)
        cv2.line(scene, (b.x + b.w // 2, b.y), (b.x + b.w // 2, b.y + b.h), (70, 70, 70), 4)

    def _draw_curtain(self, scene: Image, at: float) -> None:
        b = WINDOW_BOX
        sway = int(4 * math.sin(at / 7) + 2 * math.sin(at / 3.1))
        width = int(28 + (1 - self.curtain(at)) * (b.w / 2 - 28)) + sway

        for x0, x1 in ((b.x - 10, b.x + width), (b.x + b.w - width, b.x + b.w + 10)):
            cv2.rectangle(scene, (x0, b.y - 12), (x1, b.y + b.h + 30), (110, 140, 170), -1)
            for fold in range(x0 + 6, x1, 12):
                cv2.line(scene, (fold, b.y - 12), (fold + sway // 2, b.y + b.h + 30), (85, 110, 140), 2)

    def _draw_door(self, scene: Image, open_: bool) -> None:
        b = DOOR_BOX
        cv2.rectangle(scene, (b.x, b.y), (b.x + b.w, b.y + b.h), DOOR_COLORS[open_], -1)
        DOOR_DETAILS[open_](scene)

    def _draw_tv(self, scene: Image, at: float) -> None:
        b = TV_BOX
        screen = scene[b.y + 5 : b.y + b.h - 5, b.x + 5 : b.x + b.w - 5]

        if not self.tv_on(at):
            screen[:] = (18, 18, 20)
            return

        cells = np.random.default_rng(_hash("tv", int(at))).integers(30, 255, (4, 6, 3), dtype=np.uint8)
        screen[:] = cv2.resize(cells, (screen.shape[1], screen.shape[0]), interpolation=cv2.INTER_LINEAR)

    def _people(self, at: float) -> list[tuple[int, int, int]]:
        return [p for i, action in enumerate(self._actions) for p in self._actors(action, i, at)]

    def _actors(self, action: object, index: int, at: float) -> list[tuple[int, int, int]]:
        match action:
            case Drop() | Move():
                return _walker(at, action.at, PLACES[action.place].x + STAND_OFFSET, index)
            case PickUp():
                place = self.state(action.at - 0.001).items[action.item]
                return _walker(at, action.at, PLACES[str(place)].x + STAND_OFFSET, index)
            case Door():
                return _walker(at, action.at, DOOR_BOX.x + DOOR_BOX.w + 10, index)
            case Furniture():
                return _walker(at, action.at, CHAIR.x + CHAIR.w + 20 + self.state(action.at).chair[0], index)
            case Linger():
                return _linger(at, action, index)
            case Party():
                return _guests(at, action, index)
            case _:
                return []


DOOR_COLORS = {True: (28, 28, 34), False: (60, 85, 125)}
DOOR_DETAILS = {
    True: lambda scene: cv2.rectangle(scene, (DOOR_BOX.x, DOOR_BOX.y), (DOOR_BOX.x + 16, DOOR_BOX.y + DOOR_BOX.h), (75, 105, 150), -1),
    False: lambda scene: cv2.circle(scene, (DOOR_BOX.x + DOOR_BOX.w - 16, DOOR_BOX.y + 140), 5, (60, 170, 200), -1),
}


def _daylight(hour: float, sunrise: float, sunset: float) -> float:
    inside = sunrise < hour < sunset

    return math.sin(math.pi * (hour - sunrise) / (sunset - sunrise)) ** 0.6 if inside else 0.0


def _walker(at: float, action_at: float, target_x: int, who: int) -> list[tuple[int, int, int]]:
    progress = 1 - abs(at - action_at) / WALK_S
    x = int(DOOR_X + (target_x - DOOR_X) * min(1.0, progress * 1.3))

    return [(x, 150, who)] if progress > 0 else []


def _linger(at: float, action: Linger, who: int) -> list[tuple[int, int, int]]:
    target = PLACES[action.place].x + STAND_OFFSET
    arriving = _walker(at, action.at, target, who) if at < action.at else []
    leaving = _walker(at, action.until, target, who) if at > action.until else []
    staying = [(target + int(3 * math.sin(at / 3)), 150, who)] if action.at <= at <= action.until else []

    return arriving + staying + leaving


def _guests(at: float, action: Party, index: int) -> list[tuple[int, int, int]]:
    inside = action.at <= at <= action.until

    return [
        (int(np.clip(260 + 90 * g + 160 * math.sin(at / 97 + g * 1.7) + 50 * math.sin(at / 23 + g), 160, W - PERSON_W - 20)), 135 + 12 * (g % 3), index * 10 + g)
        for g in range(action.guests * inside)
    ]


def _person(scene: Image, x: int, y: int, who: int) -> Box:
    shade = _hash("person", who)
    shirt = (40 + shade % 160, 40 + (shade >> 8) % 160, 40 + (shade >> 16) % 160)
    cv2.rectangle(scene, (x, y + 40), (x + PERSON_W, y + 170), shirt, -1)
    cv2.rectangle(scene, (x + 8, y + 170), (x + PERSON_W - 8, y + PERSON_H), (60, 50, 45), -1)
    cv2.circle(scene, (x + PERSON_W // 2, y + 20), 22, (120, 150, 190), -1)

    return _clip(Box(x=x, y=y - 2, w=PERSON_W, h=PERSON_H + 2))


def _thing(scene: Image, box: Box, color: tuple[int, int, int]) -> None:
    edge = tuple(int(c * 0.55) for c in color)
    cv2.rectangle(scene, (box.x, box.y), (box.x + box.w - 1, box.y + box.h - 1), color, -1)
    cv2.rectangle(scene, (box.x, box.y), (box.x + box.w - 1, box.y + box.h - 1), edge, 1)


def _chair(scene: Image, offset: tuple[int, int]) -> None:
    x, y = CHAIR.x + offset[0], CHAIR.y + offset[1]
    cv2.rectangle(scene, (x, y), (x + CHAIR.w, y + 50), (45, 70, 105), -1)
    cv2.rectangle(scene, (x, y + 50), (x + CHAIR.w, y + 62), (35, 55, 85), -1)
    cv2.rectangle(scene, (x + 4, y + 62), (x + 12, y + CHAIR.h), (30, 40, 60), -1)
    cv2.rectangle(scene, (x + CHAIR.w - 12, y + 62), (x + CHAIR.w - 4, y + CHAIR.h), (30, 40, 60), -1)


def _shift(image: NDArray[np.float32], camera: tuple[int, int]) -> NDArray[np.float32]:
    matrix = np.array([[1, 0, camera[0]], [0, 1, camera[1]]], np.float32)

    return image if camera == (0, 0) else np.asarray(cv2.warpAffine(image, matrix, (W, H), borderMode=cv2.BORDER_REFLECT), np.float32)


def _clip(box: Box) -> Box:
    x, y = max(0, min(W - 1, box.x)), max(0, min(H - 1, box.y))

    return Box(x=x, y=y, w=max(1, min(W - x, box.w)), h=max(1, min(H - y, box.h)))


def _room(seed: int) -> Image:
    rng = np.random.default_rng(seed)
    wall = np.full((H, W, 3), (196, 202, 208), np.float32) + rng.normal(0, 4, (H, W, 1))
    room = cv2.GaussianBlur(wall, (0, 0), 1.5)

    for i, y in enumerate(range(300, H, 24)):
        room[y : y + 24] = np.array([68, 104, 142], np.float32) * (0.9 + 0.2 * ((i * 37) % 7) / 7)
        room[y : y + 1] = (45, 70, 95)

    room = np.clip(room + rng.normal(0, 3, (H, W, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(room, (WINDOW_BOX.x, WINDOW_BOX.y), (WINDOW_BOX.x + WINDOW_BOX.w, WINDOW_BOX.y + WINDOW_BOX.h), (70, 70, 70), -1)
    cv2.rectangle(room, (DOOR_BOX.x - 8, DOOR_BOX.y - 8), (DOOR_BOX.x + DOOR_BOX.w + 8, DOOR_BOX.y + DOOR_BOX.h), (150, 160, 170), -1)
    cv2.rectangle(room, (TV_BOX.x, TV_BOX.y), (TV_BOX.x + TV_BOX.w, TV_BOX.y + TV_BOX.h), (20, 20, 22), -1)
    cv2.rectangle(room, (250, 330), (560, 352), (80, 120, 160), -1)
    cv2.rectangle(room, (250, 352), (560, 392), (55, 85, 120), -1)
    for x in (258, 540):
        cv2.rectangle(room, (x, 392), (x + 12, 470), (45, 70, 100), -1)
    cv2.rectangle(room, (860, 110), (944, 300), (50, 75, 105), -1)
    for y in (150, 218, 286):
        cv2.rectangle(room, (856, y), (948, y + 6), (70, 100, 135), -1)
    cv2.rectangle(room, (600, 318), (900, 372), (95, 85, 80), -1)
    cv2.rectangle(room, (600, 372), (900, 440), (110, 100, 92), -1)
    for x in (700, 800):
        cv2.line(room, (x, 372), (x, 440), (85, 76, 70), 2)
    cv2.rectangle(room, (530, 430), (592, 438), (70, 100, 135), -1)
    cv2.rectangle(room, (556, 438), (566, 480), (55, 80, 110), -1)
    cv2.rectangle(room, (LAMP_AT[0] - 3, LAMP_AT[1]), (LAMP_AT[0] + 3, 330), (40, 40, 40), -1)
    cv2.rectangle(room, (LAMP_AT[0] - 26, LAMP_AT[1] - 30), (LAMP_AT[0] + 26, LAMP_AT[1]), (170, 200, 220), -1)

    return room
