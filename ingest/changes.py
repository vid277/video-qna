from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from models import Box, Change, ChangeKind, IngestConfig, LightMode, Observation
from store.images import PREVIEW_SCALE, Image, Mask, gray

MOTION_HISTORY = 120
MOTION_RATE = 0.05
FLICKER_DELTA = 20
VOLATILE_LEVEL = 0.15
VOLATILE_FILL = np.ones((7, 7), np.uint8)
VOLATILE_MARGIN = np.ones((5, 5), np.uint8)
MIN_ACTIVE_FRAMES = 2
MIN_FOREGROUND_PX = 4
PREVIEW_LEVEL = 128.0

ALIGN_SCALE = 2
MIN_ALIGN_SHIFT = 3
MAX_ALIGN_SHIFT = 20
MIN_ALIGN_RESPONSE = 0.2
COLOR_DELTA = 20
PATCH_PX = 9
MIN_CORRELATION = 0.5
TEXTURE_VARIANCE = (40.0, 12.0, 12.0)
TEXTURE_RATIO = 6.0
EDGE_THRESHOLDS = ((50, 150), (20, 60), (20, 60))
GLOBAL_CHANGE = 0.3
EDGE_RATIO = 1.5
MOVE_SIMILARITY = 0.6
HISTOGRAM_BINS = 32
NIGHT_SATURATION = 12
MOTION_MARGIN = 0.25
MAX_MARGIN_PX = 40
MERGE_GAP_PX = 24
AGENT_REACH_PX = 24
AGENT_SLACK_S = 4.0
ABSORB_MARGIN = 8
YOUNG_TOLERANCE = 0.2
MIN_STATIC_FRACTION = 0.2
LIGHTING_SCALE = 1 / 24
LIGHTING_SHIFT = 0.04

type Found = tuple[Change, Image, Image, float, float]


@dataclass
class Reference:
    """One lighting mode's view of the room at rest, and per pixel since when it has differed and when it last looked normal."""

    image: Image
    known: Mask
    since: Image
    clear_at: Image


@dataclass(frozen=True)
class Diff:
    old: Image
    new: Image
    old_edges: Image
    new_edges: Image
    changed: Mask
    lighting_shift: float


class SceneWatcher:
    def __init__(self, config: IngestConfig):
        self._settle_s = config.settle_s
        self._activity_fraction = config.activity_fraction
        self._subtractor = cv2.createBackgroundSubtractorMOG2(history=MOTION_HISTORY, detectShadows=False)
        self._volatility: Image = np.zeros((1, 1), np.float32)
        self._warmup_frames = config.volatility_warmup_s * config.fps
        self._frames_seen = 0
        self._fps = config.fps
        self._window_s = config.volatility_window_s
        self._last_at: float | None = None
        self._previous: Image = np.zeros((1, 1), np.uint8)
        self._streak = 0
        self._last_motion: float | None = None

    @property
    def in_episode(self) -> bool:
        return self._last_motion is not None

    def observe(self, preview: Image, at: float) -> Observation:
        steady = cv2.convertScaleAbs(preview, alpha=PREVIEW_LEVEL / max(float(np.median(preview)), 1.0))
        dt = at - self._last_at if self._last_at is not None else 1 / self._fps
        self._last_at = at

        motion_rate = 1 - (1 - MOTION_RATE) ** max(dt * self._fps, 1.0)
        moving = (self._subtractor.apply(steady, learningRate=motion_rate) > 0).astype(np.uint8)

        self._learn_volatility(steady, moving, dt)
        moving[self._volatile()] = 0

        motion = bool(moving.mean() > self._activity_fraction)
        self._streak = self._streak + 1 if motion else 0

        last = self._last_motion
        started = last is None and self._streak >= MIN_ACTIVE_FRAMES
        active = motion and (last is not None or started)
        settled = not active and last is not None and at - last >= self._settle_s
        self._last_motion = at if active else None if settled else last

        return Observation(
            active=active,
            started=started,
            settled=settled,
            foreground=_boxes(moving, MIN_FOREGROUND_PX, PREVIEW_SCALE),
        )

    def volatile_mask(self, shape: tuple[int, ...]) -> Mask:
        mask = self._volatile().astype(np.uint8)

        return np.asarray(cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST), dtype=bool)

    def _volatile(self) -> Mask:
        mask = cv2.morphologyEx((self._volatility > VOLATILE_LEVEL).astype(np.uint8), cv2.MORPH_CLOSE, VOLATILE_FILL)

        return np.asarray(cv2.dilate(mask, VOLATILE_MARGIN), dtype=bool)

    def _learn_volatility(self, steady: Image, moving: Image, dt: float) -> None:
        same_size = self._previous.shape == steady.shape
        flicker = cv2.absdiff(steady, self._previous) > FLICKER_DELTA if same_size else np.zeros(steady.shape, bool)
        changing = flicker | moving.astype(bool)
        self._previous = steady
        self._frames_seen += 1

        learning = self._frames_seen <= self._warmup_frames or not self.in_episode
        rate = min(1.0, max(1 / self._frames_seen, dt / self._window_s)) * learning
        volatility = self._volatility if same_size else np.zeros(steady.shape, np.float32)
        cv2.accumulateWeighted(changing.astype(np.float32), volatility, rate)
        self._volatility = volatility


class ChangeTracker:
    def __init__(self, config: IngestConfig):
        self._persist_s = config.change_persist_s
        self._min_px = config.min_change_px
        self._drift_s = config.reference_drift_s
        self._fast_drift_s = config.reference_fast_drift_s
        self._last_check: float | None = None
        self._references: dict[LightMode, Reference] = {}
        self._stale: dict[LightMode, list[Box]] = {mode: [] for mode in LightMode}
        self._busy_at: Image = np.zeros((1, 1), np.float32)
        self._width = config.change_width

    def check(self, frame: Image, at: float, foreground: list[Box], people: list[Box], ignore: Mask) -> list[Found]:
        """Compares at change_width for speed and stable thresholds; boxes and after-crops come back at full size."""
        scale = min(1.0, self._width / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        small_ignore = np.asarray(cv2.resize(ignore.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST), bool)

        found = self._check(small, at, _scale_all(foreground, scale), _scale_all(people, scale), small_ignore)

        return [
            (_scale_change(c, 1 / scale), before, _crop(frame, _scale(c.box, 1 / scale)), after_s, by)
            for c, before, _, after_s, by in found
        ]

    def _check(self, frame: Image, at: float, foreground: list[Box], people: list[Box], ignore: Mask) -> list[Found]:
        mode = light_mode(frame)
        ref = self._references.get(mode)
        busy = _region_mask(frame.shape, foreground) | _region_mask(frame.shape, people)

        if ref is None or ref.image.shape != frame.shape:
            return self._rebase(mode, frame, busy, at)

        aligned, moved = _align(ref.image, frame, busy | ~ref.known)
        if moved:
            return self._rebase(mode, frame, busy, at)

        recently_busy = self._mark_busy(busy, at)
        diff = _compare(ref.image, aligned, static=~(recently_busy | ignore), valid=~ignore & ref.known)
        if diff.changed.mean() > GLOBAL_CHANGE:
            return self._rebase(mode, aligned, busy, at)

        changed = diff.changed
        ref.since = np.where(busy, ref.since, np.where(changed, np.minimum(ref.since, at), np.inf))
        ref.clear_at = np.where(changed | busy, ref.clear_at, np.float32(at)).astype(np.float32)
        regions = _merge(_ready_regions(changed & ~busy, recently_busy, ref.since, at, self._persist_s, self._min_px), MERGE_GAP_PX)
        caused = [r for r in regions if _touched(self._busy_at, _changed_after(ref.clear_at, changed, r), r)]
        fresh = self._unreported(mode, caused)

        found = _report(ref, aligned, diff, fresh, at)
        self._learn(ref, aligned, diff, busy, recently_busy, regions + [c.source for c, *_ in found if c.source], at)

        return found

    def _mark_busy(self, busy: Mask, at: float) -> Mask:
        """Where people or motion are now or were within change_persist_s."""
        busy_at = self._busy_at if self._busy_at.shape == busy.shape else np.full(busy.shape, -np.inf, np.float32)
        self._busy_at = np.where(busy, np.float32(at), busy_at).astype(np.float32)

        return (at - self._busy_at) < self._persist_s

    def _unreported(self, mode: LightMode, caused: list[Box]) -> list[Box]:
        """A change reported in one lighting mode is old news when the other mode sees it later."""
        stale = [r for r in caused if any(_near(r, s, 0) for s in self._stale[mode])]
        fresh = [r for r in caused if r not in stale]
        self._stale = {
            m: [s for s in boxes if not any(_near(s, r, 0) for r in stale)] + (fresh if m is not mode else [])
            for m, boxes in self._stale.items()
        }

        return fresh

    def _learn(self, ref: Reference, aligned: Image, diff: Diff, busy: Mask, recently_busy: Mask, settled: list[Box], at: float) -> None:
        """Reported regions become the new normal at once; the rest drifts toward the frame, faster while the light shifts."""
        drift_s = self._fast_drift_s if diff.lighting_shift > LIGHTING_SHIFT else self._drift_s
        drift = 1 - np.exp(-(at - (self._last_check if self._last_check is not None else at)) / drift_s)
        self._last_check = at
        revealed = ~ref.known & ~recently_busy
        absorbed = _absorb(ref.image, aligned, settled, diff.changed | busy, float(drift))
        ref.image = np.where(_channels(revealed, absorbed), aligned, absorbed).astype(ref.image.dtype)
        ref.known = ref.known | revealed
        ref.since = _fill(ref.since, settled, np.inf)
        ref.clear_at = _fill(ref.clear_at, settled, at)

    def _rebase(self, mode: LightMode, frame: Image, busy: Mask, at: float) -> list[Found]:
        """Start over from this frame; whatever people cover now is unknown until they move away."""
        since = np.full(frame.shape[:2], np.inf, np.float32)
        self._references[mode] = Reference(image=frame, known=~busy, since=since, clear_at=np.full(frame.shape[:2], np.float32(at), np.float32))

        return []


def _compare(reference: Image, aligned: Image, static: Mask, valid: Mask) -> Diff:
    """Changed pixels after removing exposure and room-scale lighting differences, measured on static parts of the room."""
    old_lab, raw_lab = _lab(reference), _lab(aligned)
    new_lab = _match_lighting(old_lab, _match_exposure(old_lab, raw_lab, static), static)
    old_edges, new_edges = _edges(old_lab), _edges(new_lab)
    changed = _changed_pixels(old_lab, new_lab, old_edges, new_edges) & valid
    lighting_shift = abs(np.log(_gain(old_lab, raw_lab, static)))

    return Diff(old_lab[:, :, 0], new_lab[:, :, 0], old_edges, new_edges, changed, float(lighting_shift))


def _report(ref: Reference, aligned: Image, diff: Diff, fresh: list[Box], at: float) -> list[Found]:
    changes = _pair_moves([Change(kind=_kind(r, diff.old_edges, diff.new_edges), box=r) for r in fresh], diff.old, diff.new)
    window = {r: (_changed_after(ref.clear_at, diff.changed, r), _changed_by(ref.since, diff.changed, r, at)) for r in fresh}

    return [(c, _crop(ref.image, c.source or c.box), _crop(aligned, c.box), *window.get(c.box, (at, at))) for c in changes]


def _merge(boxes: list[Box], gap: int) -> list[Box]:
    """Pieces of one change found in the same check are reported once."""
    merged = list(boxes)
    joined = True
    while joined:
        joined = False
        for i, a in enumerate(merged):
            near = next((b for b in merged[i + 1 :] if _near(a, b, gap)), None)
            if near is not None:
                merged = [m for m in merged if m is not a and m is not near] + [_union(a, near)]
                joined = True
                break

    return merged


def _near(a: Box, b: Box, gap: int) -> bool:
    return a.x - gap < b.x + b.w and b.x - gap < a.x + a.w and a.y - gap < b.y + b.h and b.y - gap < a.y + a.h


def _union(a: Box, b: Box) -> Box:
    x, y = min(a.x, b.x), min(a.y, b.y)

    return Box(x=x, y=y, w=max(a.x + a.w, b.x + b.w) - x, h=max(a.y + a.h, b.y + b.h) - y)


def _scale(box: Box, factor: float) -> Box:
    return Box(
        x=int(box.x * factor), y=int(box.y * factor), w=max(1, round(box.w * factor)), h=max(1, round(box.h * factor))
    )


def _scale_all(boxes: list[Box], factor: float) -> list[Box]:
    return [_scale(b, factor) for b in boxes]


def _scale_change(change: Change, factor: float) -> Change:
    source = _scale(change.source, factor) if change.source else None

    return Change(kind=change.kind, box=_scale(change.box, factor), source=source)


def _channels(mask: Mask, image: Image) -> Mask:
    return mask[..., None] if image.ndim == 3 else mask


def light_mode(frame: Image) -> LightMode:
    saturation = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) if frame.ndim == 3 else 0.0

    return LightMode.NIGHT if saturation < NIGHT_SATURATION else LightMode.DAY


def _align(reference: Image, frame: Image, busy: Mask) -> tuple[Image, bool]:
    still = ~busy
    small_ref, small_new = (_masked_small(gray(i), still) for i in (reference, frame))
    (dx, dy), response = cv2.phaseCorrelate(small_ref, small_new)
    dx, dy = dx * ALIGN_SCALE, dy * ALIGN_SCALE
    shift = max(abs(dx), abs(dy))

    moved = shift > MAX_ALIGN_SHIFT or response < MIN_ALIGN_RESPONSE
    dx, dy = (round(dx), round(dy)) if MIN_ALIGN_SHIFT <= shift <= MAX_ALIGN_SHIFT else (0, 0)
    matrix = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    return aligned, moved


def _masked_small(image: Image, keep: Mask) -> Image:
    filled = np.where(keep, image, np.float32(np.median(image[keep]) if keep.any() else 0)).astype(np.float32)

    return cv2.resize(filled, None, fx=1 / ALIGN_SCALE, fy=1 / ALIGN_SCALE, interpolation=cv2.INTER_AREA)


def _lab(image: Image) -> Image:
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return cv2.cvtColor(cv2.GaussianBlur(color, (5, 5), 0), cv2.COLOR_BGR2LAB)


def _edges(lab: Image) -> Image:
    channels = [cv2.Canny(lab[:, :, c], *EDGE_THRESHOLDS[c]) for c in range(3)]

    return cv2.bitwise_or(cv2.bitwise_or(channels[0], channels[1]), channels[2])


def _gain(reference: Image, image: Image, static: Mask) -> float:
    sample = static if static.mean() > MIN_STATIC_FRACTION else np.ones(static.shape, bool)

    return float(np.median(reference[:, :, 0][sample]) + 1) / float(np.median(image[:, :, 0][sample]) + 1)


def _match_exposure(reference: Image, image: Image, static: Mask) -> Image:
    matched = image.copy()
    matched[:, :, 0] = cv2.convertScaleAbs(image[:, :, 0], alpha=_gain(reference, image, static))

    return matched


def _match_lighting(reference: Image, image: Image, static: Mask) -> Image:
    """Removes smooth, room-scale lighting differences; objects are small and sharp, so they survive."""
    sigma = max(reference.shape[:2]) * LIGHTING_SCALE
    weight = cv2.GaussianBlur(static.astype(np.float32), (0, 0), sigma) + 1e-3
    smooth = lambda channel: cv2.GaussianBlur(channel * static, (0, 0), sigma) / weight  # noqa: E731

    ref, new = reference.astype(np.float32), image.astype(np.float32)
    lightness = new[:, :, 0] * (smooth(ref[:, :, 0]) + 1) / (smooth(new[:, :, 0]) + 1)
    colour = [new[:, :, c] + smooth(ref[:, :, c]) - smooth(new[:, :, c]) for c in (1, 2)]

    return np.clip(np.dstack([lightness, *colour]), 0, 255).astype(np.uint8)


def _changed_pixels(old: Image, new: Image, old_edges: Image, new_edges: Image) -> Mask:
    """A patch changed if its pattern changed; light makes patterns brighter or darker but keeps them."""
    structure = np.zeros(old.shape[:2], bool)
    for channel, texture in enumerate(TEXTURE_VARIANCE):
        structure |= _pattern_changed(old[:, :, channel].astype(np.float32), new[:, :, channel].astype(np.float32), texture)

    changed = structure.astype(np.uint8)
    closed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    return np.asarray(cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), dtype=bool)


def _pattern_changed(old: Image, new: Image, texture: float) -> Mask:
    window = (PATCH_PX, PATCH_PX)
    mean_old, mean_new = cv2.boxFilter(old, -1, window), cv2.boxFilter(new, -1, window)
    var_old = cv2.boxFilter(old * old, -1, window) - mean_old * mean_old
    var_new = cv2.boxFilter(new * new, -1, window) - mean_new * mean_new
    covariance = cv2.boxFilter(old * new, -1, window) - mean_old * mean_new

    correlation = covariance / np.sqrt(np.maximum(var_old * var_new, 1e-6))
    textured = (var_old > texture) | (var_new > texture)
    lopsided = np.maximum(var_old, var_new) > TEXTURE_RATIO * np.maximum(np.minimum(var_old, var_new), texture)

    return textured & ((correlation < MIN_CORRELATION) | lopsided)


def _region_mask(shape: tuple[int, ...], boxes: list[Box]) -> Mask:
    mask = np.zeros(shape[:2], bool)
    for b in boxes:
        pad_x, pad_y = min(int(b.w * MOTION_MARGIN), MAX_MARGIN_PX), min(int(b.h * MOTION_MARGIN), MAX_MARGIN_PX)
        mask[max(0, b.y - pad_y) : b.y + b.h + pad_y, max(0, b.x - pad_x) : b.x + b.w + pad_x] = True

    return mask


def _ready_regions(visible: Mask, busy: Mask, since: Image, at: float, persist_s: float, min_px: int) -> list[Box]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(visible.astype(np.uint8), connectivity=8)
    labels = np.asarray(labels, np.int64)
    young = np.bincount(labels[visible & (at - since < persist_s)], minlength=count)
    near_busy = visible & cv2.dilate(busy.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    touching = np.bincount(labels[near_busy], minlength=count)

    return [
        Box(x=int(x), y=int(y), w=int(w), h=int(h))
        for (x, y, w, h, area), fresh, touch in zip(stats[1:count], young[1:count], touching[1:count], strict=True)
        if area >= min_px and touch == 0 and fresh <= YOUNG_TOLERANCE * area
    ]


def _touched(busy_at: Image, clear: float, box: Box) -> bool:
    """Objects do not move on their own: something must have been at the spot since it last looked normal."""
    pad = AGENT_REACH_PX
    window = busy_at[max(0, box.y - pad) : box.y + box.h + pad, max(0, box.x - pad) : box.x + box.w + pad]

    return bool(window.size) and float(window.max()) >= clear - AGENT_SLACK_S


def _changed_after(clear_at: Image, changed: Mask, box: Box) -> float:
    window = (slice(box.y, box.y + box.h), slice(box.x, box.x + box.w))
    times = clear_at[window][changed[window]]

    return float(np.median(times)) if times.size else float(clear_at[window].max())


def _changed_by(since: Image, changed: Mask, box: Box, at: float) -> float:
    window = (slice(box.y, box.y + box.h), slice(box.x, box.x + box.w))
    times = since[window][changed[window]]
    seen = times[np.isfinite(times)]

    return float(np.median(seen)) if seen.size else at


def _fill(values: Image, boxes: list[Box], value: float) -> Image:
    for b in boxes:
        values[b.y : b.y + b.h, b.x : b.x + b.w] = value

    return values


def _absorb(reference: Image, aligned: Image, ready: list[Box], held: Mask, drift: float) -> Image:
    drifted = cv2.addWeighted(reference, 1 - drift, aligned, drift, 0)
    updated = np.where(held[..., None] if reference.ndim == 3 else held, reference, drifted).astype(reference.dtype)

    for b in ready:
        y0, x0 = max(0, b.y - ABSORB_MARGIN), max(0, b.x - ABSORB_MARGIN)
        updated[y0 : b.y + b.h + ABSORB_MARGIN, x0 : b.x + b.w + ABSORB_MARGIN] = aligned[
            y0 : b.y + b.h + ABSORB_MARGIN, x0 : b.x + b.w + ABSORB_MARGIN
        ]

    return updated


def _crop(image: Image, box: Box, margin: int = 32) -> Image:
    h, w = image.shape[:2]
    pad = max(margin, box.w // 2, box.h // 2)

    return image[max(0, box.y - pad) : min(h, box.y + box.h + pad), max(0, box.x - pad) : min(w, box.x + box.w + pad)]


def _boxes(mask: Image | Mask, min_px: int, scale: int) -> list[Box]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    return [
        Box(x=int(x) * scale, y=int(y) * scale, w=int(w) * scale, h=int(h) * scale)
        for x, y, w, h, area in stats[1:count]
        if area >= min_px
    ]


def _kind(box: Box, old_edges: Image, new_edges: Image) -> ChangeKind:
    before, after = _edge_count(old_edges, box), _edge_count(new_edges, box)
    kinds = {(True, False): ChangeKind.ADDED, (False, True): ChangeKind.REMOVED}

    return kinds.get((after > before * EDGE_RATIO, before > after * EDGE_RATIO), ChangeKind.CHANGED)


def _edge_count(edges: Image, box: Box) -> int:
    return int(np.count_nonzero(edges[box.y : box.y + box.h, box.x : box.x + box.w]))


def _pair_moves(changes: list[Change], old: Image, new: Image) -> list[Change]:
    """A removal and an addition that look alike are one move, best matches first, each change used once."""
    paired: set[int] = set()
    moves: list[Change] = []

    for r, a in _move_candidates(changes, old, new):
        if not {id(r), id(a)} & paired:
            paired |= {id(r), id(a)}
            moves.append(Change(kind=ChangeKind.MOVED, box=a.box, source=r.box))

    return [c for c in changes if id(c) not in paired] + moves


def _move_candidates(changes: list[Change], old: Image, new: Image) -> list[tuple[Change, Change]]:
    removed = [c for c in changes if c.kind is ChangeKind.REMOVED]
    added = [c for c in changes if c.kind is ChangeKind.ADDED]
    scored = [(_similarity(old, r.box, new, a.box), r, a) for r in removed for a in added if _similar_size(r.box, a.box)]

    return [(r, a) for score, r, a in sorted(scored, key=lambda t: t[0], reverse=True) if score >= MOVE_SIMILARITY]


def _similar_size(a: Box, b: Box) -> bool:
    return max(a.area, b.area) <= 2 * min(a.area, b.area)


def _similarity(old: Image, a: Box, new: Image, b: Box) -> float:
    return float(cv2.compareHist(_histogram(old, a), _histogram(new, b), cv2.HISTCMP_CORREL))


def _histogram(image: Image, box: Box) -> Image:
    patch = image[box.y : box.y + box.h, box.x : box.x + box.w]

    return cv2.calcHist([patch], [0], None, [HISTOGRAM_BINS], [0, 256])
