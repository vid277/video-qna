from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from huey import Huey

from jobs.queue import consumer
from models.jobs import Backoff, Job, JobsConfig, QueueName

ERROR_CHARS = 2000

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobKind:
    name: str
    queue: QueueName
    priority: int
    find: Callable[[int], Awaitable[list[str]]]
    run: Callable[[str], Awaitable[None]]
    idle_only: bool = False


@dataclass(frozen=True)
class Scheduled:
    name: str
    every_s: float
    run: Callable[[], Awaitable[None]]


class FailureLog(Protocol):
    async def record_failure(self, kind: str, key: str, error: str, backoff: Backoff) -> None: ...

    async def clear_failure(self, kind: str, key: str) -> None: ...


class Scheduler:
    def __init__(
        self,
        config: JobsConfig,
        queues: dict[QueueName, Huey],
        kinds: Sequence[JobKind],
        scheduled: Sequence[Scheduled],
        failures: FailureLog,
        is_outage: Callable[[BaseException], bool],
    ):
        self._config = config
        self._queues = queues
        self._kinds = sorted(kinds, key=lambda k: -k.priority)
        self._scheduled = scheduled
        self._failures = failures
        self._is_outage = is_outage
        self._backoff = Backoff(max_attempts=config.max_attempts, first_s=config.backoff_first_s, max_s=config.backoff_max_s)

        self._handlers = {k.name: self._indexed(k) for k in kinds} | {s.name: self._timed(s) for s in scheduled}
        self._tasks = {name: q.task(name="run")(self._execute) for name, q in queues.items()}
        self._inflight: dict[QueueName, set[Job]] = {name: set() for name in QueueName}
        self._next_run = {s.name: 0.0 for s in scheduled}
        self._work_queues = [q for q in QueueName if any(k.queue == q for k in kinds)]
        self._paused_until = 0.0
        self._finished = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def inflight(self) -> dict[QueueName, frozenset[Job]]:
        return {name: frozenset(jobs) for name, jobs in self._inflight.items()}

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        consumers = [consumer(q, self._config.queues[name].workers) for name, q in self._queues.items()]

        for q in self._queues.values():
            q.flush()
        for c in consumers:
            c.start()

        try:
            while True:
                await self._dispatch_safely()
                await self._wait_for_work()
        finally:
            await asyncio.gather(*(asyncio.to_thread(c.stop, True) for c in consumers))

    async def dispatch_once(self) -> None:
        now = asyncio.get_running_loop().time()
        self._enqueue_due(now)

        for queue in self._work_queues if now >= self._paused_until else []:
            await self._fill(queue)

    def _enqueue_due(self, now: float) -> None:
        timed = self._inflight[QueueName.MAINTENANCE]
        due = [s for s in self._scheduled if now >= self._next_run[s.name] and Job(kind=s.name, key="") not in timed]

        self._next_run |= {s.name: now + s.every_s for s in due}
        self._enqueue_all(QueueName.MAINTENANCE, [Job(kind=s.name, key="") for s in due], 0)

    async def _fill(self, queue: QueueName) -> None:
        inflight = self._inflight[queue]
        room = self._config.queues[queue].max_inflight - len(inflight)
        busy = bool(inflight)

        for kind in [k for k in self._kinds if k.queue == queue]:
            allowed = 0 if kind.idle_only and busy else room
            found = await kind.find(allowed + len(inflight)) if allowed > 0 else []
            fresh = [j for j in (Job(kind=kind.name, key=k) for k in found) if j not in inflight][:allowed]

            self._enqueue_all(queue, fresh, kind.priority)
            room -= len(fresh)
            busy = busy or bool(found)

    async def _dispatch_safely(self) -> None:
        try:
            await self.dispatch_once()
        except Exception as e:
            self._pause(e)

    async def _wait_for_work(self) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(self._config.dispatch_every_s):
                await self._finished.wait()

        self._finished.clear()

    def _enqueue_all(self, queue: QueueName, jobs: list[Job], priority: int) -> None:
        for job in jobs:
            self._inflight[queue].add(job)
            self._tasks[queue](job, priority=priority)

    def _execute(self, job: Job) -> None:
        assert self._loop is not None, "Scheduler.run() must be running"

        asyncio.run_coroutine_threadsafe(self._attempt(job), self._loop).result()

    async def _attempt(self, job: Job) -> None:
        try:
            await self._handlers[job.kind](job.key)
        finally:
            for jobs in self._inflight.values():
                jobs.discard(job)

            self._finished.set()

    def _indexed(self, kind: JobKind) -> Callable[[str], Awaitable[None]]:
        async def handle(key: str) -> None:
            try:
                await kind.run(key)
            except Exception as e:
                await (self._pause_for(e) if self._is_outage(e) else self._record(kind.name, key, e))
                return

            await self._failures.clear_failure(kind.name, key)

        return handle

    def _timed(self, scheduled: Scheduled) -> Callable[[str], Awaitable[None]]:
        async def handle(_key: str) -> None:
            try:
                await scheduled.run()
            except Exception:
                log.exception("scheduled job %s failed", scheduled.name)

        return handle

    async def _record(self, kind: str, key: str, error: Exception) -> None:
        log.error("job %s %s failed: %r", kind, key, error, exc_info=error)

        try:
            await self._failures.record_failure(kind, key, repr(error)[:ERROR_CHARS], self._backoff)
        except Exception as e:
            self._pause(e)

    async def _pause_for(self, error: BaseException) -> None:
        self._pause(error)

    def _pause(self, error: BaseException) -> None:
        self._paused_until = asyncio.get_running_loop().time() + self._config.outage_pause_s
        log.warning("pausing new work for %.0fs: %r", self._config.outage_pause_s, error)
