from __future__ import annotations

import numpy as np
import pytest

from models import Box, ChangeKind
from tests.sim import library
from tests.sim.render import Renderer
from tests.sim.run import SCENARIOS, run
from tests.sim.scenario import ARRIVED, LEFT, Bump, Door, Drop, Furniture, Party, expectations, samples, state_at
from tests.sim.score import Found, score

BOOK_ON_DESK = Box(x=275, y=341, w=30, h=11)


def found(kind: ChangeKind, box: Box, at: float, source: Box | None = None) -> Found:
    return Found(kind=kind, box=box, source=source, changed_after=at - 5, changed_by=at, detected_at=at + 20)


def test_expectations_follow_the_actions() -> None:
    expected = expectations(library.drop_and_pick_up())

    assert [(e.label, e.kinds) for e in expected] == [
        ("book dropped on desk left", ARRIVED),
        ("airpods dropped on couch", ARRIVED),
        ("keys dropped on side table", ARRIVED),
        ("book moved from desk left", LEFT),
        ("book moved to shelf top", ARRIVED),
        ("airpods picked up from couch", LEFT),
        ("keys picked up from side table", LEFT),
    ]
    assert expected[0].box == BOOK_ON_DESK


def test_small_bumps_keep_coordinates_and_pans_move_them() -> None:
    small, pan = expectations(library.bump()), expectations(library.pan())

    assert (small[-1].label, small[-1].box.x - small[0].box.x, small[-1].box.y - small[0].box.y) == ("book picked up from desk left", 0, 0)
    assert (pan[1].label, pan[1].box.x - pan[0].box.x, pan[1].box.y - pan[0].box.y) == ("airpods picked up from side table", 40, 25)


def test_room_state_replays_actions() -> None:
    scenario = library.drop_and_pick_up()

    assert state_at(scenario, library.h(0, 9)).items["book"] is None
    assert state_at(scenario, library.h(0, 16)).items == {"book": "desk left", "airpods": "couch", "keys": "side table", "mug": None, "bag": None, "bottle": None}
    assert state_at(scenario, library.h(0, 23)).items["book"] == "shelf top"


def test_time_is_dense_around_actions_and_sparse_between() -> None:
    times = np.array(samples(library.drop_and_pick_up()))
    gaps = np.diff(times)
    near_drop = times[(times > library.h(0, 10) - 10) & (times < library.h(0, 10) + 30)]

    assert gaps.max() == 300 and np.all(np.diff(near_drop) <= 1)


def test_night_is_infrared_grayscale_and_day_is_colour() -> None:
    renderer = Renderer(library.night_ir())
    night, _ = renderer.frame(library.h(0, 2.5))
    day, _ = renderer.frame(library.h(0, 12))

    assert renderer.night(library.h(0, 2.5)) and not renderer.night(library.h(0, 12))
    assert np.abs(night[:, :, 0].astype(int) - night[:, :, 2]).mean() < 3
    assert np.abs(day[:, :, 0].astype(int) - day[:, :, 2]).mean() > 10


def test_people_are_only_reported_while_someone_is_in_the_room() -> None:
    renderer = Renderer(library.party())

    assert renderer.frame(library.h(0, 12))[1]
    assert not renderer.frame(library.h(0, 12) + 120)[1]
    assert len(renderer.frame(library.h(0, 20))[1]) == 6


def test_score_counts_hits_moves_duplicates_false_and_distractors() -> None:
    scenario = library.drop_and_pick_up()
    expected = expectations(scenario)
    book_move = [e for e in expected if e.label.startswith("book moved")]
    reports = [
        found(ChangeKind.ADDED, BOOK_ON_DESK, library.h(0, 10)),
        found(ChangeKind.ADDED, BOOK_ON_DESK, library.h(0, 10) + 30),
        found(ChangeKind.MOVED, book_move[1].box, library.h(0, 17), source=book_move[0].box),
        found(ChangeKind.REMOVED, BOOK_ON_DESK, library.h(0, 13)),
        found(ChangeKind.CHANGED, Box(x=380, y=100, w=20, h=20), library.h(0, 14)),
        found(ChangeKind.ADDED, Box(x=10, y=500, w=20, h=20), library.h(0, 15)),
    ]

    report = score(scenario.name, 1, 100, expected, reports)

    assert [h.found is not None for h in report.hits] == [True, False, False, True, True, False, False]
    assert (report.duplicates, len(report.distractor), len(report.false)) == (1, 1, 2)
    assert report.failures(scenario.thresholds) == ["recall 0.43 < 1.0", "2.00 false changes/day > 0.0"]


def test_two_weeks_covers_every_situation_consistently() -> None:
    scenario = library.two_weeks()
    kinds = {type(a) for a in scenario.actions}
    expected = expectations(scenario)

    assert {Drop, Door, Party, Furniture, Bump} <= kinds
    assert len(expected) > 60
    assert all(0 <= e.at <= scenario.days * 86_400 for e in expected)


@pytest.mark.slow
@pytest.mark.parametrize("name", SCENARIOS)
def test_simulated_room(name: str) -> None:
    scenario = SCENARIOS[name]()
    report = run(scenario)

    assert not report.failures(scenario.thresholds), "\n".join(report.describe())
