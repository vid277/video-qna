from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import openrouter
from openrouter import Message, OpenRouter, OpenRouterError, text_part

type Handler = Callable[[web.Request], Any]


def sse(*chunks: dict[str, Any] | str) -> bytes:
    lines = [f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks]

    return (": OPENROUTER PROCESSING\n\n" + "".join(lines)).encode()


def delta(**fields: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": fields}]}


@contextlib.asynccontextmanager
async def serve(monkeypatch: pytest.MonkeyPatch, routes: dict[str, Handler]) -> AsyncIterator[OpenRouter]:
    app = web.Application()
    app.add_routes([web.post(path, handler) for path, handler in routes.items()])

    async with TestServer(app) as server:
        monkeypatch.setattr(openrouter, "BASE_URL", str(server.make_url("/api/v1/")))

        async with OpenRouter("key", str(server.make_url("/local/v1"))) as client:
            yield client


async def collect(client: OpenRouter, tools: list[dict[str, Any]] | None = None) -> tuple[list[str], Message]:
    pieces: list[str] = []
    final: Message = {}

    async for piece in client.chat_stream("m", [{"role": "user", "content": "hi"}], tools or []):
        match piece:
            case str():
                pieces.append(piece)
            case _:
                final = piece

    return pieces, final


def streaming(body: bytes) -> Handler:
    async def handler(request: web.Request) -> web.Response:
        assert (await request.json())["stream"] is True

        return web.Response(body=body, content_type="text/event-stream")

    return handler


async def test_chat_stream_yields_text_pieces_then_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    body = sse(delta(role="assistant", content=""), delta(content="On the "), delta(content="shelf."), "[DONE]")

    async with serve(monkeypatch, {"/api/v1/chat/completions": streaming(body)}) as client:
        pieces, final = await collect(client)

        assert pieces == ["On the ", "shelf."]
        assert final == {"role": "assistant", "content": "On the shelf."}


async def test_chat_stream_merges_tool_call_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    body = sse(
        delta(content="Let me look."),
        delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function", "function": {"name": "search", "arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"query":'}}]),
        delta(tool_calls=[{"index": 1, "id": "call_2", "function": {"name": "events", "arguments": "{}"}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": ' "book"}'}}]),
        "[DONE]",
    )

    async with serve(monkeypatch, {"/api/v1/chat/completions": streaming(body)}) as client:
        _, final = await collect(client)

        assert final["content"] == "Let me look."
        assert [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in final["tool_calls"]] == [
            ("call_1", "search", {"query": "book"}),
            ("call_2", "events", {}),
        ]


async def test_chat_stream_retries_before_streaming_and_raises_stream_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    async def flaky(request: web.Request) -> web.Response:
        attempts.append(1)
        body = sse(delta(content="ok"), {"error": {"code": 502, "message": "upstream died"}})

        return web.Response(status=503, text="busy") if len(attempts) == 1 else web.Response(body=body, content_type="text/event-stream")

    async with serve(monkeypatch, {"/api/v1/chat/completions": flaky}) as client:
        with pytest.raises(OpenRouterError, match="upstream died"):
            await collect(client)

        assert len(attempts) == 2


async def test_embeddings_go_to_the_embedding_url(monkeypatch: pytest.MonkeyPatch) -> None:
    async def local(request: web.Request) -> web.Response:
        inputs = (await request.json())["input"]

        return web.json_response({"data": [{"index": i, "embedding": [float(i)]} for i in range(len(inputs))]})

    async def remote(request: web.Request) -> web.Response:
        raise AssertionError("embeddings must not go to OpenRouter")

    async with serve(monkeypatch, {"/local/v1/embeddings": local, "/api/v1/embeddings": remote}) as client:
        vectors = await client.embed("m", [[text_part("a")], [text_part("b")]], 1)

        assert vectors == [[0.0], [1.0]]
