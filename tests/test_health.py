from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta

import pytest

from ingest import health
from ingest.health import Heartbeat, ProblemRecorder
from models import IngestConfig
from models.health import ProcessHealth

pytestmark = pytest.mark.usefixtures("runtime")


async def test_heartbeat_writes_file_while_running_and_removes_it_on_stop(config: IngestConfig) -> None:
    heartbeat = Heartbeat("ingest")
    heartbeat.saw_snapshot(heartbeat.sample().written_at)
    task = asyncio.create_task(heartbeat.run())
    beats: list[ProcessHealth] = []

    for _ in range(50):
        await asyncio.sleep(0.1)
        beats = health.read_heartbeats()
        if beats:
            break

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert [(b.role, b.pid, b.last_snapshot_at is not None) for b in beats] == [("ingest", os.getpid(), True)]
    assert health.read_heartbeats() == []


def test_problem_recorder_keeps_last_warning_first_line() -> None:
    recorder = ProblemRecorder()
    logger = logging.getLogger("tests.health")
    logger.addHandler(recorder)

    logger.info("fine")
    logger.warning("database unavailable\nsecond line")

    try:
        raise ValueError("bad frame")
    except ValueError:
        logger.exception("dropped snapshot")

    logger.removeHandler(recorder)

    assert recorder.last is not None
    assert (recorder.last.level, recorder.last.message) == ("ERROR", "dropped snapshot (ValueError('bad frame'))")


async def test_report_marks_stale_and_dead_processes_not_running(config: IngestConfig) -> None:
    fresh = Heartbeat("ingest").sample()
    stale = fresh.model_copy(update={"role": "worker", "written_at": fresh.written_at - timedelta(minutes=5)})
    dead = fresh.model_copy(update={"role": "zombie", "pid": 999_999})

    for beat in (fresh, stale, dead):
        (config.data_dir / "health").mkdir(exist_ok=True)
        (config.data_dir / "health" / f"{beat.role}.json").write_text(beat.model_dump_json())

    report = await health.report()
    lines = health.describe(report)

    assert {p.health.role: p.running for p in report.processes} == {"ingest": True, "worker": False, "zombie": False}
    assert report.latest_frame_at is None and report.failing_jobs == 0
    assert any(line.startswith("worker") and "NOT RUNNING" in line for line in lines)


def test_times_are_shown_in_local_time_whatever_their_timezone() -> None:
    now = datetime.now().astimezone()
    utc = (now - timedelta(minutes=5)).astimezone(UTC)
    naive = (now - timedelta(minutes=5)).replace(tzinfo=None)

    assert health._ago(utc, now) == f"{utc.astimezone():%Y-%m-%d %H:%M:%S} (5m ago)"
    assert health._ago(naive, now) == f"{naive:%Y-%m-%d %H:%M:%S} (5m ago)"
    assert health._ago(None, now) == "never"
