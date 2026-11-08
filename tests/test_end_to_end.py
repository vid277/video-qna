"""Without a camera or model access: synthetic frames go through ingest, every index job and the agent's tools."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import background.room
import background.tasks
import embeddings.facts
import ingest.pipeline
import store.runtime as runtime
from background.room import RoomLog
from embeddings.facts import FactIndex
from ingest.pipeline import IngestPipeline
from models import Answer, Box, ChangeKind, IngestConfig, Snapshot, ToolResult
from models.camera import StreamQuality
from openrouter import Message, OpenRouter, Part
from recall.agent import Agent
from store import images
from store.archive import FrameArchive
from store.db import Event, SceneChange
from store.store import Store
from tests.test_tracker import KEYS, quiet_drop

WIRED_PACKAGES = {"store", "ingest", "embeddings", "background", "monitoring", "recall", "frontend"}
LABEL = {"category": "object", "label": "yellow keys", "place": "on the desk"}
CAPTION = {"summary": "Someone put yellow keys on the desk.", "new": ["yellow keys on the desk"], "facts": ["yellow keys placed on the desk"], "objects": ["keys"]}


class StubModels:
    """Stands in for OpenRouter and the local embedding server with fixed, deterministic replies."""

    def __init__(self) -> None:
        self.chat_requests: list[list[Message]] = []

    async def embed(
        self, model: str, inputs: Sequence[Sequence[Part]], dimensions: int, input_type: str | None = None, interactive: bool = False
    ) -> list[list[float]]:
        return [_vector(json.dumps(list(parts)), dimensions) for parts in inputs]

    async def complete_json(self, model: str, parts: list[Part], name: str, schema: dict[str, Any]) -> dict[str, Any]:
        replies = {"event_caption": CAPTION, "change_label": LABEL, "period_summary": {"summary": "Keys were put on the desk."}}

        return dict(replies[name])

    async def chat_stream(self, model: str, messages: list[Message], tools: list[dict[str, Any]]) -> AsyncIterator[str | Message]:
        self.chat_requests.append(list(messages))
        reply = _agent_turn(len(self.chat_requests), messages)

        if isinstance(reply.get("content"), str):
            yield reply["content"]
        yield reply


def _vector(text: str, dimensions: int) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8])
    vector = np.random.default_rng(seed).normal(size=dimensions)

    return (vector / np.linalg.norm(vector)).tolist()


def _call(index: int, name: str, arguments: dict[str, Any]) -> Message:
    return {"id": f"c{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _agent_turn(turn: int, messages: list[Message]) -> Message:
    """Uses every tool once, looks at the changes history returned, then answers."""
    now = datetime.now().astimezone()
    if turn == 1:
        calls = [
            _call(1, "room_state", {}),
            _call(2, "history", {"query": "keys"}),
            _call(3, "facts", {"query": "keys on the desk"}),
            _call(4, "search", {"query": "keys"}),
            _call(5, "events", {"start": (now - timedelta(hours=6)).isoformat(), "end": now.isoformat()}),
        ]
        return {"role": "assistant", "content": None, "tool_calls": calls}

    if turn == 2:
        tool_text = " ".join(m["content"] for m in messages if m.get("role") == "tool")
        refs = list(dict.fromkeys(re.findall(r"change:\d+", tool_text)))[:2]
        return {"role": "assistant", "content": None, "tool_calls": [_call(6, "look", {"refs": refs})]}

    return {"role": "assistant", "content": "Your keys are on the desk."}


class ScriptedPeople:
    """The person detector, told where the synthetic scene drew people."""

    boxes: list[Box] = []

    def __init__(self, score: float):
        pass

    def detect(self, frame: Any) -> list[Box]:
        return ScriptedPeople.boxes


@pytest.fixture
async def wired(monkeypatch: pytest.MonkeyPatch, config: IngestConfig, store: Store, archive: FrameArchive, room_log: RoomLog) -> AsyncIterator[None]:
    """Points every module's shared config, store, archive, room log and fact index at the test database."""
    fact_index = FactIndex(config.database_url, config.embedding_model)
    swap = {
        id(runtime.config): config,
        id(runtime.store): store,
        id(runtime.archive): archive,
        id(background.room.room): room_log,
        id(embeddings.facts.facts): fact_index,
    }

    for module in [m for name, m in list(sys.modules.items()) if name.split(".")[0] in WIRED_PACKAGES]:
        for name, value in list(vars(module).items()):
            if id(value) in swap:
                monkeypatch.setattr(module, name, swap[id(value)])

    monkeypatch.setattr(ingest.pipeline, "PersonDetector", ScriptedPeople)

    yield

    await fact_index.close()


async def run_jobs(client: OpenRouter) -> dict[str, int]:
    """What the worker does, minus the queue: run every kind's pending work until none is left."""
    kinds, ran = background.tasks.index_kinds(client), {}

    for _ in range(10):
        pending = [(kind, key) for kind in kinds for key in await kind.find(100)]
        if not pending:
            return ran

        for kind, key in pending:
            await kind.run(key)
            ran[kind.name] = ran.get(kind.name, 0) + 1

    raise AssertionError(f"jobs never finished: {ran}")


async def test_dropped_keys_are_found_by_the_agent(wired: None, db: async_sessionmaker[AsyncSession]) -> None:
    pipeline = IngestPipeline("cam1")
    start = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=3)

    for t, (frame, people) in enumerate(quiet_drop(np.random.default_rng(7))):
        ScriptedPeople.boxes = people
        await pipeline.ingest(Snapshot(camera_id="cam1", captured_at=start + timedelta(seconds=t), quality=StreamQuality.MAIN, image=images.encode(frame)))

    async with db() as s:
        changes = list(await s.scalars(select(SceneChange)))
        events = list(await s.scalars(select(Event)))

    added = [c for c in changes if c.kind == ChangeKind.ADDED.value]
    boxes = [Box.model_validate(c.box) for c in added]
    assert boxes and all(abs(b.x - KEYS.x) < 40 and abs(b.y - KEYS.y) < 40 for b in boxes)
    assert events and all(e.ended_at is not None for e in events)

    client = cast(OpenRouter, StubModels())
    ran = await run_jobs(client)
    assert {"change_vectors", "frame_vectors", "captions", "change_labels", "item_links", "fact_vectors"} <= ran.keys()

    steps = [s async for s in Agent(client).stream("where are my keys?")]
    results = {s.name: s for s in steps if isinstance(s, ToolResult)}

    assert not [r.output for r in results.values() if r.output.startswith("error")]
    assert "yellow keys" in results["room_state"].output and "on the desk" in results["room_state"].output
    assert '"kind": "added"' in results["history"].output
    assert "yellow keys placed on the desk" in results["facts"].output
    assert results["search"].refs and results["events"].summary != "0 events"
    assert results["look"].summary != "looked at 0 images"
    assert steps[-1] == Answer(text="Your keys are on the desk.", refs=results["look"].refs)
