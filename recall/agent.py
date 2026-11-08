from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError

from background.room import room
from embeddings.facts import facts
from models import (
    Answer,
    AnswerDelta,
    EventsArgs,
    FactsArgs,
    HistoryArgs,
    LookArgs,
    Record,
    RoomStateArgs,
    SearchArgs,
    SearchHit,
    StoredFrame,
    ToolCall,
    ToolResult,
    VectorOwner,
)
from openrouter import Message, OpenRouter, Part, image_part, text_part
from prompts import render
from store.db import EMBEDDING_DIM
from store.runtime import archive, config, store
from store.views import frame_crops, frame_image

MAX_LOOK_IMAGES = 8
MAX_TOOL_CALLS = 5
MAX_TOOL_OUTPUT = 20_000

type View = tuple[str, bytes]
type Outcome = tuple[str, list[Part]]

REF = re.compile(r"\b(?:frame|change|event):\d+\b")

log = logging.getLogger(__name__)


def tool(name: str, description: str, args: type[Record]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": args.model_json_schema()}}


TOOLS = [
    tool("search", "Find frames, events and scene changes that look like a short phrase.", SearchArgs),
    tool("facts", "Search written descriptions of events by meaning and keyword.", FactsArgs),
    tool("events", "List events in a time window with their summaries, key frames and scene changes.", EventsArgs),
    tool("room_state", "Items in the room at a time, with label, place and since when.", RoomStateArgs),
    tool("history", "Every change to an object or part of the room, in time order.", HistoryArgs),
    tool("look", "View frames, change crops or an event's key frames.", LookArgs),
]


class Agent:
    def __init__(self, client: OpenRouter):
        self._client = client
        self._tools: dict[str, Callable[[str], Awaitable[Outcome]]] = {
            "search": self._search,
            "facts": self._facts,
            "events": self._events,
            "room_state": self._room_state,
            "history": self._history,
            "look": self._look,
        }
        self._views: dict[str, Callable[[list[int]], Awaitable[list[View]]]] = {
            "frame": self._frame_views,
            "change": self._change_views,
            "event": self._event_views,
        }

    async def ask(self, question: str) -> Answer:
        answer = Answer(text="", refs=[])
        async for step in self.stream(question):
            answer = step if isinstance(step, Answer) else answer

        return answer

    async def stream(self, question: str) -> AsyncIterator[ToolCall | ToolResult | AnswerDelta | Answer]:
        now = datetime.now().astimezone()
        messages: list[Message] = [
            {"role": "system", "content": render("agent", now=f"{now:%A %Y-%m-%d %I:%M %p} (UTC{now:%z})", max_tool_calls=MAX_TOOL_CALLS)},
            {"role": "user", "content": question},
        ]
        looked: list[str] = []

        for step in range(config.agent_max_steps + 1):
            last = step == config.agent_max_steps
            messages += [{"role": "user", "content": "Answer now with what you have."}] if last else []
            reply: Message = {}

            async for piece in self._client.chat_stream(config.agent_model, messages, [] if last else TOOLS):
                if isinstance(piece, str):
                    yield AnswerDelta(text=piece)
                else:
                    reply = piece

            messages.append(reply)
            calls = reply.get("tool_calls")
            if last or not calls:
                yield _answer(reply.get("content") or "", looked)
                return

            async for event in self._call_tools(calls, messages, looked):
                yield event

    async def _call_tools(self, calls: list[Message], messages: list[Message], looked: list[str]) -> AsyncIterator[ToolCall | ToolResult]:
        """Run each requested tool, adding its output, and any images it fetched, to the conversation."""
        images: list[Part] = []

        for call in calls:
            step = ToolCall(id=call["id"], name=call["function"]["name"], arguments=call["function"]["arguments"] or "{}")
            yield step

            text, parts = await self._run(step.name, step.arguments)
            result = _result(step, text, parts)
            looked.extend(result.refs if step.name == "look" else [])
            yield result

            messages.append({"role": "tool", "tool_call_id": step.id, "content": text})
            images += parts

        if images:
            messages.append({"role": "user", "content": [text_part("Images you asked to look at, in order:"), *images]})

    async def _run(self, name: str, arguments: str) -> Outcome:
        try:
            return await self._tools[name](arguments)
        except (KeyError, ValueError, ValidationError, DBAPIError) as e:
            return f"error: {e}", []

    async def _search(self, arguments: str) -> Outcome:
        args = SearchArgs.model_validate_json(arguments)
        query = (await self._embed([[text_part(args.query)]]))[0]
        hits = await store.search(query, args.start, args.end, config.search_limit)
        ranked = await self._rerank_with_crops(query, hits)

        return _dump(ranked), []

    async def _rerank_with_crops(self, query: list[float], hits: list[SearchHit]) -> list[SearchHit]:
        frames = await self._uncropped_frames(hits)
        crops = await asyncio.to_thread(lambda: [frame_crops(f) for f in frames])
        vectors = await self._embed([[image_part(c)] for per_frame in crops for c in per_frame])
        scores = _cosine(vectors, query).reshape(len(frames), -1).max(axis=1) if frames else np.zeros(0)

        crop_hits = [_crop_hit(f, float(s)) for f, s in zip(frames, scores, strict=True)]
        best = {h.ref: h for h in sorted(hits + crop_hits, key=lambda h: h.score)}

        return sorted(best.values(), key=lambda h: h.score, reverse=True)[: config.search_limit]

    async def _uncropped_frames(self, hits: list[SearchHit]) -> list[StoredFrame]:
        """Matched frames and matched clips' key frames whose crops were never embedded at ingest."""
        frame_ids = [int(h.ref.partition(":")[2]) for h in hits if h.ref.startswith("frame:")]
        clips = [h.event_id for h in hits if h.owner is VectorOwner.CLIP and h.event_id]
        event_frames = [f.id for event_id in clips for f in await store.key_frames(event_id)]
        candidates = list(dict.fromkeys(frame_ids + event_frames))
        covered = await store.frames_with_crops(candidates)

        return await store.frames([i for i in candidates if i not in covered][: config.query_crop_frames])

    async def _facts(self, arguments: str) -> Outcome:
        args = FactsArgs.model_validate_json(arguments)
        query = await self._client.embed(config.embedding_model, [[text_part(args.query)]], EMBEDDING_DIM, "query")
        by_meaning = await facts.search(query[0], args.start, args.end, config.search_limit)
        by_words = await store.search_facts(args.query, args.start, args.end, config.search_limit)
        best = {(h.ref, h.text): h for h in sorted(by_words + by_meaning, key=lambda h: h.rank)}
        hits = sorted(best.values(), key=lambda h: h.rank, reverse=True)[: config.search_limit]

        return _dump(hits), []

    async def _events(self, arguments: str) -> Outcome:
        args = EventsArgs.model_validate_json(arguments)
        events = await store.events_between(args.start, args.end)

        return _dump(events), []

    async def _room_state(self, arguments: str) -> Outcome:
        args = RoomStateArgs.model_validate_json(arguments)
        items = await room.state(args.at)

        return _dump(items), []

    async def _history(self, arguments: str) -> Outcome:
        args = HistoryArgs.model_validate_json(arguments)
        query = (await self._embed([[text_part(args.query)]]))[0]
        hits = await store.search(query, None, None, config.search_limit)
        changes = [int(h.ref.partition(":")[2]) for h in hits if h.owner is VectorOwner.CHANGE]
        by_label = await room.items_labelled(args.query, config.search_limit)
        items = list(dict.fromkeys(by_label + await room.items_for_changes(changes)))[: config.search_limit]

        return _dump(await room.history(items)), []

    async def _look(self, arguments: str) -> Outcome:
        args = LookArgs.model_validate_json(arguments)
        refs = [ref.partition(":") for ref in args.refs]
        views = [v for kind, _, raw in refs for v in await self._views[kind]([int(raw)])][:MAX_LOOK_IMAGES]
        labels = "\n".join(f"image {i}: {label}" for i, (label, _) in enumerate(views, start=1))

        return labels or "nothing found for those refs", [image_part(jpeg) for _, jpeg in views]

    async def _frame_views(self, frame_ids: list[int]) -> list[View]:
        frames = [f for f in await store.frames(frame_ids) if archive.has_frame(f)]
        jpegs = await asyncio.to_thread(lambda: [frame_image(f) for f in frames])

        return [(f"frame:{f.id} at {f.captured_at.astimezone():%Y-%m-%d %H:%M:%S}", j) for f, j in zip(frames, jpegs, strict=True)]

    async def _change_views(self, change_ids: list[int]) -> list[View]:
        changes = await store.changes(change_ids)
        views: list[View] = []

        for c in changes:
            label = f"change:{c.id} {c.kind.value} by {c.detected_at.astimezone():%Y-%m-%d %H:%M:%S}"
            wide = await room.context(c.id)
            views += [(f"{label}, wide view with the spot outlined", wide)] if wide else []
            crops = (("before", c.before_path), ("after", c.after_path))
            views += [(f"{label}, {side}", archive.read(path)) for side, path in crops if archive.exists(path)]

        return views

    async def _event_views(self, event_ids: list[int]) -> list[View]:
        frames = [f for event_id in event_ids for f in await store.key_frames(event_id)]

        return await self._frame_views([f.id for f in frames])

    async def _embed(self, inputs: list[list[Part]]) -> list[list[float]]:
        return await self._client.embed(config.embedding_model, inputs, EMBEDDING_DIM, interactive=True) if inputs else []


def _cosine(vectors: list[list[float]], query: list[float]) -> np.ndarray:
    matrix, target = np.asarray(vectors, np.float32), np.asarray(query, np.float32)

    return matrix @ target / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(target) + 1e-9)


def _crop_hit(frame: StoredFrame, score: float) -> SearchHit:
    return SearchHit(
        ref=f"frame:{frame.id}",
        owner=VectorOwner.CROP,
        event_id=frame.event_id,
        captured_at=frame.captured_at,
        score=round(score, 4),
    )


def _result(call: ToolCall, text: str, images: list[Part]) -> ToolResult:
    refs = list(dict.fromkeys(REF.findall(text)))
    summaries = {
        "look": f"looked at {len(images)} images",
        "search": f"{len(refs)} matches",
        "facts": f"{len(refs)} matching descriptions",
        "events": f"{text.count('\"ref\": \"event:')} events",
        "room_state": f"{text.count('item_ref')} items",
        "history": f"{text.count('item_ref')} changes",
    }

    summary = summaries.get(call.name, text[:80])

    return ToolResult(id=call.id, name=call.name, summary=summary, refs=refs, output=text[:MAX_TOOL_OUTPUT])


def _answer(text: str, looked: list[str]) -> Answer:
    cited = REF.findall(text)

    return Answer(text=text, refs=list(dict.fromkeys(cited + looked)))


def _dump(records: Iterable[Record]) -> str:
    """Tool results in the viewer's local time, whatever the database session's zone is."""
    return json.dumps([r.model_dump() for r in records], default=_local)


def _local(value: object) -> str:
    return value.astimezone().isoformat(timespec="seconds") if isinstance(value, datetime) else str(value)
