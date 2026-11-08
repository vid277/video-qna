from __future__ import annotations

import asyncio
import logging
import os
import shutil
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from models.health import HealthReport, LogLine, ProcessHealth, ProcessStatus
from store.runtime import archive, config, store

DISK_EVERY_S = 300.0
STALE_BEATS = 3
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MESSAGE_CHARS = 300

log = logging.getLogger(__name__)


def health_dir() -> Path:
    return config.data_dir / "health"


class ProblemRecorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.last: LogLine | None = None

    def emit(self, record: logging.LogRecord) -> None:
        cause = f" ({record.exc_info[1]!r})" if record.exc_info and record.exc_info[1] else ""
        message = (record.getMessage().splitlines() or [""])[0] + cause

        self.last = LogLine(
            at=datetime.fromtimestamp(record.created).astimezone(),
            level=record.levelname,
            message=message[:MESSAGE_CHARS],
        )


class Heartbeat:
    def __init__(self, role: str):
        self.role = role
        self.problems = ProblemRecorder()
        self._started_at = datetime.now().astimezone()
        self._last_snapshot: datetime | None = None
        self._disk_bytes: int | None = None

    @property
    def path(self) -> Path:
        return health_dir() / f"{self.role}.json"

    def saw_snapshot(self, at: datetime) -> None:
        self._last_snapshot = at

    def configure_logging(self) -> None:
        logs = config.data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        rotating = RotatingFileHandler(
            logs / f"{self.role}.log", maxBytes=int(config.log_max_mb * 1e6), backupCount=config.log_backups
        )

        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler(), rotating, self.problems])

    def sample(self) -> ProcessHealth:
        return ProcessHealth(
            role=self.role,
            pid=os.getpid(),
            started_at=self._started_at,
            written_at=datetime.now().astimezone(),
            last_snapshot_at=self._last_snapshot,
            disk_bytes=self._disk_bytes,
            last_problem=self.problems.last,
        )

    async def run(self) -> None:
        try:
            await asyncio.gather(self._beat(), self._measure_disk())
        finally:
            self.path.unlink(missing_ok=True)

    async def _beat(self) -> None:
        while True:
            try:
                await asyncio.to_thread(_write_atomic, self.path, self.sample().model_dump_json())
            except OSError:
                log.exception("heartbeat write failed")

            await asyncio.sleep(config.heartbeat_every_s)

    async def _measure_disk(self) -> None:
        while True:
            try:
                self._disk_bytes = await asyncio.to_thread(archive.usage_bytes)
            except OSError:
                log.exception("disk usage scan failed")

            await asyncio.sleep(DISK_EVERY_S)


def read_heartbeats() -> list[ProcessHealth]:
    files = sorted(health_dir().glob("*.json")) if health_dir().is_dir() else []

    return [ProcessHealth.model_validate_json(f.read_text()) for f in files]


async def report() -> HealthReport:
    now = datetime.now().astimezone()
    beats = await asyncio.to_thread(read_heartbeats)
    latest = await store.latest_frame()
    backlog = await store.index_backlog()
    failures = await store.failures()
    sizes = [b.disk_bytes for b in beats if b.disk_bytes is not None]
    config.data_dir.mkdir(parents=True, exist_ok=True)

    return HealthReport(
        checked_at=now,
        processes=[_status(b, now) for b in beats],
        latest_frame_at=latest.captured_at if latest else None,
        backlog=backlog,
        failing_jobs=len(failures),
        disk_bytes=max(sizes, default=None),
        disk_budget_bytes=int(config.disk_budget_gb * 1e9),
        disk_free_bytes=shutil.disk_usage(config.data_dir).free,
    )


def describe(health: HealthReport) -> list[str]:
    now = health.checked_at
    lines = [_describe_process(p, now) for p in health.processes] or ["processes  none running (no heartbeat files)"]
    used = f"{_gb(health.disk_bytes)} of {_gb(health.disk_budget_bytes)} budget" if health.disk_bytes is not None else "not measured yet"

    return [
        *lines,
        f"last frame {_ago(health.latest_frame_at, now)}",
        *[f"backlog    {b.kind.value}: {b.pending} pending, oldest {_ago(b.oldest, now)}" for b in health.backlog if b.pending],
        f"jobs       {health.failing_jobs} failing (see `ghostcam.py jobs`)",
        f"disk       {used}, {_gb(health.disk_free_bytes)} free on volume",
        *[f"problem    {p.health.role}: {p.health.last_problem.at.astimezone():%Y-%m-%d %H:%M:%S} {p.health.last_problem.level} "
          f"{p.health.last_problem.message}" for p in health.processes if p.health.last_problem],
    ]


def _status(beat: ProcessHealth, now: datetime) -> ProcessStatus:
    age = (now - beat.written_at).total_seconds()
    fresh = age < STALE_BEATS * config.heartbeat_every_s

    return ProcessStatus(health=beat, running=fresh and _alive(beat.pid), heartbeat_age_s=round(age, 1))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True


def _describe_process(p: ProcessStatus, now: datetime) -> str:
    state = {True: "running", False: "NOT RUNNING (stale heartbeat)"}[p.running]
    snapshot = f", last snapshot {_ago(p.health.last_snapshot_at, now)}" if p.health.last_snapshot_at else ""

    return f"{p.health.role:10} {state}, pid {p.health.pid}, heartbeat {p.heartbeat_age_s:.0f}s ago{snapshot}"


def _ago(at: datetime | None, now: datetime) -> str:
    local = at.astimezone() if at else None
    seconds = (now.astimezone() - local).total_seconds() if local else None

    return "never" if local is None or seconds is None else f"{local:%Y-%m-%d %H:%M:%S} ({_duration(seconds)} ago)"


def _duration(seconds: float) -> str:
    units = [(86400, "d"), (3600, "h"), (60, "m"), (1, "s")]
    size, suffix = next(((s, u) for s, u in units if seconds >= s), (1, "s"))

    return f"{seconds / size:.0f}{suffix}"


def _gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.tmp")
    partial.write_text(text)
    partial.replace(path)
