from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from aiohttp.test_utils import TestClient, TestServer

from ingest import images
from ingest.archive import FrameArchive
from ingest.health import Heartbeat
from ingest.db import SceneChange
from ingest.store import Store
from models import Box, Change, ChangeKind, Record
from tests.factory import CAMERA, add_frame, picture
from frontend.server import Site

pytestmark = pytest.mark.usefixtures("runtime")


class ToolCall(Record):
    id: str
    name: str
    arguments: str


class ToolResult(Record):
    id: str
    name: str
    summary: str
    refs: list[str]


class Answer(Record):
    text: str
    refs: list[str]


class Seeded(Record):
    frame_ids: list[int]
    event_id: int
    change_id: int


async def seed(store: Store, archive: FrameArchive) -> Seeded:
    start = datetime.now().astimezone() - timedelta(minutes=10)
    event_id = await store.open_event(CAMERA, start)

    for i in range(3):
        await add_frame(store, archive, start + timedelta(seconds=i), i, event_id)

    before, after = "changes/b.jpg", "changes/a.jpg"
    archive.write(before, images.encode(picture(100)))
    archive.write(after, images.encode(picture(101)))
    change = Change(kind=ChangeKind.ADDED, box=Box(x=1, y=1, w=10, h=10))
    await store.add_change(event_id, change, before, after, start + timedelta(seconds=1), start + timedelta(seconds=2))
    await store.close_event(event_id, start + timedelta(seconds=3))
    await store.mark_key_frames(event_id, 8)

    frames = await store.key_frames(event_id)
    changes = await store.event_changes(event_id)

    return Seeded(frame_ids=[f.id for f in frames], event_id=event_id, change_id=changes[0].id)


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    blocks = [b for b in body.split("\n\n") if b.strip()]
    lines = [dict(line.split(": ", 1) for line in b.splitlines()) for b in blocks]

    return [(fields["event"], json.loads(fields["data"])) for fields in lines]


@pytest.fixture
async def client(store: Store, archive: FrameArchive) -> AsyncIterator[tuple[TestClient, Seeded]]:
    seeded = await seed(store, archive)

    async def stream(question: str) -> AsyncIterator[Record]:
        yield ToolCall(id="c1", name="search", arguments=json.dumps({"query": question}))
        yield ToolResult(id="c1", name="search", summary="2 hits", refs=[f"frame:{seeded.frame_ids[0]}", "bogus:1", "frame:999999"])
        yield Answer(text="On the shelf.", refs=[f"change:{seeded.change_id}", f"event:{seeded.event_id}"])

    async with TestClient(TestServer(Site(stream).app())) as c:
        yield c, seeded


async def test_ask_streams_steps_with_evidence(client: tuple[TestClient, Seeded]) -> None:
    c, seeded = client

    response = await c.get("/api/ask", params={"q": "book"})
    events = parse_sse(await response.text())

    assert response.headers["Content-Type"] == "text/event-stream"
    assert [name for name, _ in events] == ["tool_call", "tool_result", "answer", "stream_done"]
    assert json.loads(events[0][1]["arguments"]) == {"query": "book"}
    assert [e["image_url"] for e in events[1][1]["evidence"]] == [f"/img/frame/{seeded.frame_ids[0]}"]

    answer = events[2][1]
    urls = [e["image_url"] for e in answer["evidence"]]
    assert answer["text"] == "On the shelf."
    assert urls[:2] == [f"/img/change/{seeded.change_id}/before", f"/img/change/{seeded.change_id}/after"]
    assert set(urls[2:]) <= {f"/img/frame/{i}" for i in seeded.frame_ids} and len(urls) > 2


async def test_ask_reports_agent_failure_in_stream(store: Store, archive: FrameArchive) -> None:
    async def failing(question: str) -> AsyncIterator[Record]:
        yield ToolCall(id="c1", name="search", arguments="{}")
        raise RuntimeError("openrouter down")

    async with TestClient(TestServer(Site(failing).app())) as c:
        events = parse_sse(await (await c.get("/api/ask", params={"q": "book"})).text())

    assert [name for name, _ in events] == ["tool_call", "stream_error", "stream_done"]
    assert "openrouter down" in events[1][1]["error"]


async def test_ask_rejects_empty_question(client: tuple[TestClient, Seeded]) -> None:
    c, _ = client

    response = await c.get("/api/ask", params={"q": ""})

    assert response.status == 400


async def test_room_shows_latest_frame_changes_and_health(client: tuple[TestClient, Seeded], db: async_sessionmaker[AsyncSession]) -> None:
    c, seeded = client

    async with db.begin() as s:
        linked = {"label": "red book", "place": "on the desk", "category": "object", "item_id": seeded.change_id}
        await s.execute(update(SceneChange).where(SceneChange.id == seeded.change_id).values(**linked))

    heartbeat = Heartbeat("ingest")
    heartbeat.path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.path.write_text(heartbeat.sample().model_dump_json())

    room = await (await c.get("/api/room")).json()

    assert room["latest"]["ref"] == f"frame:{max(seeded.frame_ids)}"
    assert [ch["ref"] for ch in room["changes"]] == [f"change:{seeded.change_id}"]
    assert room["changes"][0]["event_ref"] == f"event:{seeded.event_id}"
    assert [(i["label"], i["place"], i["image_url"]) for i in room["state"]] == [("red book", "on the desk", f"/img/change/{seeded.change_id}/after")]
    assert [(p["health"]["role"], p["running"]) for p in room["health"]["processes"]] == [("ingest", True)]


async def test_images_are_served_by_ref(client: tuple[TestClient, Seeded], archive: FrameArchive) -> None:
    c, seeded = client

    frame = await c.get(f"/img/frame/{seeded.frame_ids[0]}")
    after = await c.get(f"/img/change/{seeded.change_id}/after")
    missing = await c.get("/img/frame/999999")

    assert (frame.status, frame.content_type) == (200, "image/jpeg")
    assert await after.read() == archive.read("changes/a.jpg")
    assert missing.status == 404


async def test_index_page_is_served(client: tuple[TestClient, Seeded]) -> None:
    c, _ = client

    page = await (await c.get("/")).text()

    assert "<title>Ghostcam</title>" in page and "/api/ask" in page


@pytest.mark.parametrize(
    "headers",
    [{"Host": "evil.example:8765"}, {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}],
)
async def test_only_local_same_origin_requests_are_served(client: tuple[TestClient, Seeded], headers: dict[str, str]) -> None:
    c, _ = client

    blocked = await c.get("/api/room", headers=headers)
    allowed = await c.get("/api/room", headers={"Sec-Fetch-Site": "same-origin"})

    assert blocked.status == 403
    assert allowed.status == 200
