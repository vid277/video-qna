from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path

from aiohttp import web
from pydantic import ValidationError

from background.room import room
from frontend.monitor import Monitor
from models import EventDigest, Record, StoredFrame
from models.frontend import (
    ActivityView,
    AskRequest,
    ChangeView,
    Evidence,
    ItemView,
    RoomView,
    StreamDone,
    StreamError,
)
from monitoring import health
from openrouter import OpenRouter
from recall.agent import Agent
from store.runtime import archive, config, store

STATIC = Path(__file__).with_name("static")
RECENT = timedelta(hours=6)
MAX_CHANGES = 30
RECENT_ACTIVITY = 5
ACTIVITY_LOOKBACK = 60
MAX_EVIDENCE = 12
EVENT_FRAMES = 4

CAMEL = re.compile(r"(?<!^)(?=[A-Z])")

type Stream = Callable[[str], AsyncIterator[Record]]

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
CROSS_SITE = {"cross-site", "same-site"}
MAX_CONCURRENT_ASKS = 2

log = logging.getLogger(__name__)


@web.middleware
async def local_only(request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> web.StreamResponse:
    """Only this machine's browser, on this page, may use the camera archive or start paid agent runs."""
    host = request.host.rsplit(":", 1)[0] if not request.host.endswith("]") else request.host
    allowed = host in LOCAL_HOSTS and request.headers.get("Sec-Fetch-Site", "same-origin") not in CROSS_SITE

    return await handler(request) if allowed else web.json_response({"error": "local requests only"}, status=403)


class Site:
    def __init__(self, stream: Stream):
        self._stream = stream
        self._asking = asyncio.Semaphore(MAX_CONCURRENT_ASKS)
        self._evidence: dict[str, Callable[[int], Awaitable[list[Evidence]]]] = {
            "frame": self._frame_evidence,
            "change": self._change_evidence,
            "event": self._event_evidence,
        }

    def app(self) -> web.Application:
        app = web.Application(middlewares=[local_only])
        app.add_routes(
            [
                *Monitor().routes(),
                web.get("/", self.index),
                web.get("/api/ask", self.ask),
                web.get("/api/room", self.room),
                web.get(r"/img/frame/{id:\d+}", self.frame_image),
                web.get(r"/img/change/{id:\d+}/{side:before|after}", self.change_image),
                web.get(r"/img/event/{id:\d+}", self.event_image),
            ]
        )

        return app

    async def index(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "index.html")

    async def ask(self, request: web.Request) -> web.StreamResponse:
        try:
            question = AskRequest(question=request.query.get("q", "")).question
        except ValidationError as e:
            return web.json_response({"error": e.errors()[0]["msg"]}, status=400)

        if self._asking.locked():
            return web.json_response({"error": "busy answering other questions, try again shortly"}, status=429)

        async with self._asking:
            return await self._answer(request, question)

    async def _answer(self, request: web.Request, question: str) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)
        started = time.monotonic()

        try:
            async for event in self._stream(question):
                await self._send(response, event)
        except ConnectionResetError:
            return response
        except Exception as e:
            log.exception("ask failed")
            await self._send(response, StreamError(error=f"the agent failed: {e}"))

        await self._send(response, StreamDone(seconds=round(time.monotonic() - started, 1)))

        return response

    async def _send(self, response: web.StreamResponse, event: Record) -> None:
        payload = event.model_dump(mode="json")
        evidence = await self.evidence(payload.get("refs", []))
        payload["evidence"] = [e.model_dump(mode="json") for e in evidence]
        name = CAMEL.sub("_", type(event).__name__).lower()

        await response.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())

    async def evidence(self, refs: list[str]) -> list[Evidence]:
        parsed = [ref.partition(":") for ref in dict.fromkeys(refs)]
        found = [e for kind, _, raw in parsed if kind in self._evidence and raw.isdigit() for e in await self._evidence[kind](int(raw))]

        return list({e.image_url: e for e in found}.values())[:MAX_EVIDENCE]

    async def room(self, _request: web.Request) -> web.Response:
        now = datetime.now().astimezone()
        latest = await store.latest_frame()
        events = await store.events_between(now - RECENT, now)
        change_refs = {int(c.ref.partition(":")[2]): e.ref for e in events for c in e.changes}
        changes = await store.changes(change_refs)

        recent = await store.recent_events(ACTIVITY_LOOKBACK)
        items = await room.state(None)

        view = RoomView(
            latest=_frame_evidence(latest, "Latest frame") if latest else None,
            state=[
                ItemView(
                    ref=i.item_ref,
                    label=i.label,
                    place=i.place,
                    since=i.since,
                    evidence=i.evidence,
                    image_url=f"/img/change/{i.evidence.partition(':')[2]}/after",
                )
                for i in sorted(items, key=lambda i: i.since, reverse=True)
            ],
            activity=[_activity(chain) for chain in _continuous(recent)][:RECENT_ACTIVITY],
            changes=[
                ChangeView(
                    ref=f"change:{c.id}",
                    event_ref=change_refs[c.id],
                    kind=c.kind,
                    at=c.detected_at,
                    before_url=f"/img/change/{c.id}/before",
                    after_url=f"/img/change/{c.id}/after",
                )
                for c in sorted(changes, key=lambda c: c.detected_at, reverse=True)[:MAX_CHANGES]
            ],
            health=await health.report(),
        )

        return web.Response(text=view.model_dump_json(), content_type="application/json")

    async def frame_image(self, request: web.Request) -> web.Response:
        frames = await store.frames([int(request.match_info["id"])])
        jpegs = await asyncio.to_thread(lambda: [archive.frame_image(f) for f in frames])

        return _jpeg(jpegs)

    async def event_image(self, request: web.Request) -> web.Response:
        frames = await store.key_frames(int(request.match_info["id"]))
        middle = frames[len(frames) // 2 : len(frames) // 2 + 1]
        jpegs = await asyncio.to_thread(lambda: [archive.frame_image(f) for f in middle])

        return _jpeg(jpegs)

    async def change_image(self, request: web.Request) -> web.Response:
        changes = await store.changes([int(request.match_info["id"])])
        paths = [{"before": c.before_path, "after": c.after_path}[request.match_info["side"]] for c in changes]
        jpegs = await asyncio.to_thread(lambda: [archive.read(p) for p in paths])

        return _jpeg(jpegs)

    async def _frame_evidence(self, frame_id: int) -> list[Evidence]:
        return [_frame_evidence(f, f"Frame {f.id}") for f in await store.frames([frame_id])]

    async def _change_evidence(self, change_id: int) -> list[Evidence]:
        return [
            Evidence(ref=f"change:{c.id}", label=f"{c.kind.value.capitalize()}, {side}", at=c.detected_at, image_url=f"/img/change/{c.id}/{side}")
            for c in await store.changes([change_id])
            for side in ("before", "after")
        ]

    async def _event_evidence(self, event_id: int) -> list[Evidence]:
        frames = await store.key_frames(event_id)
        picks = frames[:: max(1, -(-len(frames) // EVENT_FRAMES))]

        return [_frame_evidence(f, f"Event {event_id}") for f in picks]


def _frame_evidence(frame: StoredFrame, label: str) -> Evidence:
    return Evidence(ref=f"frame:{frame.id}", label=label, at=frame.captured_at, image_url=f"/img/frame/{frame.id}")


def _jpeg(jpegs: list[bytes]) -> web.Response:
    responses = [web.Response(body=j, content_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"}) for j in jpegs]

    return (responses or [web.Response(status=404, text="not found")])[0]


async def serve(host: str) -> None:
    async with OpenRouter(config.openrouter_api_key.get_secret_value(), config.embedding_url) as client:
        runner = web.AppRunner(Site(Agent(client).stream).app())
        await runner.setup()
        await web.TCPSite(runner, host, config.web_port).start()
        print(f"serving on http://{host}:{config.web_port} (Ctrl+C to stop)", flush=True)

        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()


def _continuous(newest_first: list[EventDigest]) -> list[list[EventDigest]]:
    """Events cut at max_event_s start the moment the previous one ends; those belong to one activity."""
    chains: list[list[EventDigest]] = []
    for event in newest_first:
        if chains and chains[-1][-1].started_at == event.ended_at:
            chains[-1].append(event)
        else:
            chains.append([event])

    return chains


def _activity(chain: list[EventDigest]) -> ActivityView:
    newest, oldest = chain[0], chain[-1]
    captioned = next((e for e in chain if e.summary), None)

    return ActivityView(
        ref=newest.ref,
        started_at=oldest.started_at,
        ended_at=newest.ended_at,
        summary=captioned.summary if captioned else None,
        image_url=f"/img/event/{(captioned or newest).ref.partition(':')[2]}",
        changes=[c.kind for e in reversed(chain) for c in e.changes],
    )
