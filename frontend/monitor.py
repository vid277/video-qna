from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import web
from yarl import URL

from ingest import health
from ingest.runtime import config, jobs_config, store
from jobs.queue import open_queues
from models.monitor import DiskPart, MonitorView, QueueStat, ServiceStat

DISK_CACHE_S = 60
SERVICE_TIMEOUT_S = 2
STATIC = Path(__file__).with_name("static")


class Monitor:
    def __init__(self) -> None:
        self._queues = open_queues(jobs_config)
        self._disk: list[DiskPart] = []
        self._disk_measured = 0.0

    def routes(self) -> list[web.RouteDef]:
        return [web.get("/monitor", self.page), web.get("/api/monitor", self.stats)]

    async def page(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "monitor.html")

    async def stats(self, _request: web.Request) -> web.Response:
        report, database, failures, service, disk = await asyncio.gather(
            health.report(), store.database_stats(), store.failures(), _embedding_service(), self._disk_usage()
        )
        view = MonitorView(
            generated_at=datetime.now().astimezone(),
            embedding_model=config.embedding_model,
            health=report,
            queues=await asyncio.to_thread(self._queue_stats),
            backlog=report.backlog,
            failures=failures,
            database=database,
            disk=disk,
            embedding_service=service,
        )

        return web.Response(text=view.model_dump_json(), content_type="application/json")

    def _queue_stats(self) -> list[QueueStat]:
        return [
            QueueStat(name=name.value, waiting=q.pending_count(), workers=s.workers, max_inflight=s.max_inflight)
            for name, q in self._queues.items()
            for s in [jobs_config.queues[name]]
        ]

    async def _disk_usage(self) -> list[DiskPart]:
        stale = time.monotonic() - self._disk_measured > DISK_CACHE_S
        self._disk = await asyncio.to_thread(_measure, config.data_dir) if stale else self._disk
        self._disk_measured = time.monotonic() if stale else self._disk_measured

        return self._disk


def _measure(root: Path) -> list[DiskPart]:
    entries = sorted(root.iterdir()) if root.is_dir() else []
    parts = [DiskPart(name=e.name, bytes=_size(e)) for e in entries]

    return sorted(parts, key=lambda p: -p.bytes)


def _size(path: Path) -> int:
    files = path.rglob("*") if path.is_dir() else [path]

    return sum(f.stat().st_size for f in files if f.is_file())


async def _embedding_service() -> ServiceStat:
    url = str(URL(config.embedding_url).with_path("/health"))

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=SERVICE_TIMEOUT_S)) as session:
            async with session.get(url) as response:
                body = await response.json()
    except (aiohttp.ClientError, TimeoutError, ValueError):
        return ServiceStat(url=url, up=False, model=None, device=None)

    return ServiceStat(url=url, up=True, model=body.get("model"), device=body.get("device"))
