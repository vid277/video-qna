from __future__ import annotations

import statistics

from models import Box, ChangeKind, Record
from tests.sim.scenario import DAY_S, DISTRACTORS, MATCH_SLACK_S, Expectation, Thresholds

LATEST_DETECTION_S = 4 * 3600.0
EARLY_DETECTION_S = 5.0
PAD = 6
CLAIMS = {kind: 1 for kind in ChangeKind} | {ChangeKind.MOVED: 2}


class Found(Record):
    kind: ChangeKind
    box: Box
    source: Box | None
    changed_after: float
    changed_by: float
    detected_at: float


class Hit(Record):
    expectation: Expectation
    found: Found | None


class Report(Record):
    scenario: str
    days: float
    frames: int
    hits: list[Hit]
    false: list[Found]
    distractor: list[Found]
    duplicates: int

    @property
    def recall(self) -> float:
        return sum(h.found is not None for h in self.hits) / len(self.hits) if self.hits else 1.0

    @property
    def false_per_day(self) -> float:
        return len(self.false) / self.days

    @property
    def distractor_per_day(self) -> float:
        return len(self.distractor) / self.days

    @property
    def latency_s(self) -> float:
        delays = [h.found.detected_at - h.expectation.at for h in self.hits if h.found]

        return statistics.median(delays) if delays else 0.0

    def failures(self, thresholds: Thresholds) -> list[str]:
        checks = {
            f"recall {self.recall:.2f} < {thresholds.min_recall}": self.recall < thresholds.min_recall,
            f"{self.false_per_day:.2f} false changes/day > {thresholds.max_false_per_day}": self.false_per_day > thresholds.max_false_per_day,
            f"{self.distractor_per_day:.2f} TV/curtain changes/day > {thresholds.max_distractor_per_day}": self.distractor_per_day > thresholds.max_distractor_per_day,
        }

        return [message for message, failed in checks.items() if failed]

    def describe(self) -> list[str]:
        missed = [h.expectation for h in self.hits if h.found is None]

        return [
            f"{self.scenario}: {self.days:g} days, {self.frames} frames, recall {self.recall:.2f} "
            f"({len(self.hits) - len(missed)}/{len(self.hits)}), {len(self.false)} false, {len(self.distractor)} TV/curtain, "
            f"{self.duplicates} duplicates, median detection delay {self.latency_s:.0f}s",
            *[f"  missed   {e.label} at {clock(e.at)} box {_box(e.box)}" for e in missed],
            *[f"  false    {f.kind.value} box {_box(f.box)} detected {clock(f.detected_at)}" for f in self.false],
            *[f"  tv/curt  {f.kind.value} box {_box(f.box)} detected {clock(f.detected_at)}" for f in self.distractor],
        ]


def score(name: str, days: float, frames: int, expected: list[Expectation], found: list[Found]) -> Report:
    matched: dict[int, Found] = {}
    false: list[Found] = []
    distractor: list[Found] = []
    duplicates = 0

    for f in sorted(found, key=lambda f: f.detected_at):
        candidates = [i for i, e in enumerate(expected) if _matches(f, e)]
        fresh = [i for i in candidates if i not in matched]
        matched |= {i: f for i in fresh[: CLAIMS[f.kind]]}
        duplicates += bool(candidates) and not fresh
        distracting = not candidates and any(_overlap(f.box, z.box) for z in DISTRACTORS)
        distractor += [f] * distracting
        false += [f] * (not candidates and not distracting)

    hits = [Hit(expectation=e, found=matched.get(i)) for i, e in enumerate(expected)]

    return Report(scenario=name, days=days, frames=frames, hits=hits, false=false, distractor=distractor, duplicates=duplicates)


def _matches(found: Found, expected: Expectation) -> bool:
    boxes = [found.box] + ([found.source] if found.source else [])
    placed = any(_overlap(b, expected.box) for b in boxes)
    timely = expected.at - EARLY_DETECTION_S <= found.detected_at <= expected.at + LATEST_DETECTION_S
    bracketed = found.changed_after - MATCH_SLACK_S <= expected.at <= found.changed_by + MATCH_SLACK_S

    return placed and timely and bracketed and found.kind in expected.kinds


def _overlap(a: Box, b: Box) -> bool:
    return a.x - PAD < b.x + b.w and b.x - PAD < a.x + a.w and a.y - PAD < b.y + b.h and b.y - PAD < a.y + a.h


def clock(at: float) -> str:
    day, rest = divmod(at, DAY_S)

    return f"day {int(day)} {int(rest // 3600):02d}:{int(rest % 3600 // 60):02d}:{int(rest % 60):02d}"


def _box(box: Box) -> str:
    return f"({box.x},{box.y} {box.w}x{box.h})"
