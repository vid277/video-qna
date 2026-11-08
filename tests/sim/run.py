from __future__ import annotations

import argparse
import time
from pathlib import Path

from ingest.changes import ChangeTracker, SceneWatcher
from models import IngestConfig
from store import images
from tests.sim import library
from tests.sim.render import Renderer
from tests.sim.scenario import Scenario, expectations, samples
from tests.sim.score import Found, Report, score

CONFIG = IngestConfig.from_yaml(Path(__file__).parents[1] / "ingest.yaml")
SCENARIOS = {
    s.__name__: s
    for s in (
        library.quiet_day,
        library.drop_and_pick_up,
        library.night_ir,
        library.party,
        library.linger,
        library.furniture_and_door,
        library.bump,
        library.pan,
        library.two_weeks,
    )
}


def run(scenario: Scenario, config: IngestConfig = CONFIG) -> Report:
    found = trace(scenario, config)

    return score(scenario.name, scenario.days, len(samples(scenario)), expectations(scenario), found)


def trace(scenario: Scenario, config: IngestConfig = CONFIG) -> list[Found]:
    renderer = Renderer(scenario)
    watcher, tracker = SceneWatcher(config), ChangeTracker(config)
    times = samples(scenario)
    found: list[Found] = []
    next_check = 0.0

    for at in times:
        frame, people = renderer.frame(at)
        seen = watcher.observe(images.decode_preview(images.encode(frame)), at)

        if at < next_check:
            continue

        next_check = at + config.change_check_s
        checked = tracker.check(frame, at, seen.foreground, people, watcher.volatile_mask(frame.shape))
        found += [
            Found(kind=c.kind, box=c.box, source=c.source, changed_after=after, changed_by=by, detected_at=at)
            for c, _, _, after, by in checked
        ]

    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulated rooms through the change tracker and score the result.")
    parser.add_argument("names", nargs="*", help=f"scenarios to run (default: all): {', '.join(SCENARIOS)}")
    args = parser.parse_args()

    for name in args.names or list(SCENARIOS):
        scenario = SCENARIOS[name]()
        started = time.monotonic()
        report = run(scenario)
        failures = report.failures(scenario.thresholds)

        print("\n".join(report.describe()[:40]))
        print(f"  {'FAIL: ' + '; '.join(failures) if failures else 'PASS'} ({time.monotonic() - started:.0f}s)\n", flush=True)


if __name__ == "__main__":
    main()
