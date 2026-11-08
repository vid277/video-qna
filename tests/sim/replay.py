"""Replay stored live frames through the tracker and count what it would report, e.g. `python -m tests.sim.replay 17:00 17:20`."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime

from sqlalchemy import select

from ingest import images
from ingest.changes import ChangeTracker, SceneWatcher
from ingest.db import Frame
from ingest.people import PersonDetector
from ingest.runtime import archive, config, store


async def replay(start: datetime, end: datetime) -> Counter[str]:
    query = select(Frame).where(Frame.captured_at.between(start, end), Frame.path.is_not(None)).order_by(Frame.captured_at)

    async with store._session() as s:
        frames = [f for f in (await s.scalars(query)).all() if archive.exists(str(f.path))]

    watcher, tracker, detector = SceneWatcher(config), ChangeTracker(config), PersonDetector(config.person_score)
    counts: Counter[str] = Counter()
    last: float | None = None

    for frame in frames:
        jpeg = archive.read(str(frame.path))
        at = frame.captured_at.timestamp()
        seen = watcher.observe(images.decode_preview(jpeg), at)
        if last is not None and at - last < config.change_check_s:
            continue

        last = at
        image = images.decode(jpeg)
        found = tracker.check(image, at, seen.foreground, detector.detect(image), watcher.volatile_mask(image.shape))
        counts.update(c.kind.value for c, *_ in found)

    counts["frames"] = len(frames)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", help="local time HH:MM today")
    parser.add_argument("end", help="local time HH:MM today")
    args = parser.parse_args()

    today = datetime.now().astimezone()
    at = lambda hm: today.replace(hour=int(hm[:2]), minute=int(hm[3:]), second=0, microsecond=0)  # noqa: E731
    print(dict(asyncio.run(replay(at(args.start), at(args.end)))))


if __name__ == "__main__":
    main()
