from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import aiohttp

BASE_URL = "https://openrouter.ai/api/v1/"
PROVIDER = {"data_collection": "deny"}
EMBED_BATCH = 32
RETRIES = 3
PRIORITY_HEADER = "X-Embedding-Priority"
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 524, 529}

type Part = dict[str, Any]
type Message = dict[str, Any]


class OpenRouterError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"OpenRouter {status}: {body}")
        self.status = status


def text_part(text: str) -> Part:
    return {"type": "text", "text": text}


def image_part(jpeg: bytes) -> Part:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


class OpenRouter:
    def __init__(self, api_key: str, embedding_url: str):
        self._api_key = api_key
        self._embedding_url = f"{embedding_url.rstrip('/')}/"
        self._sessions: dict[str, aiohttp.ClientSession] = {}

    async def __aenter__(self) -> OpenRouter:
        auth = {"Authorization": f"Bearer {self._api_key}"}
        self._sessions = {"chat": self._open(BASE_URL, auth), "embeddings": self._open(self._embedding_url, {})}

        return self

    async def __aexit__(self, *_exc: object) -> None:
        await asyncio.gather(*(s.close() for s in self._sessions.values()))

    def _open(self, base_url: str, headers: dict[str, str]) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            base_url=base_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=180),
        )

    def _http(self, purpose: str) -> aiohttp.ClientSession:
        assert self._sessions, "use OpenRouter as an async context manager"

        return self._sessions[purpose]

    async def embed(
        self, model: str, inputs: Sequence[Sequence[Part]], dimensions: int, input_type: str | None = None, interactive: bool = False
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        kind = {"input_type": input_type} if input_type else {}
        headers = {PRIORITY_HEADER: "interactive"} if interactive else {}

        for start in range(0, len(inputs), EMBED_BATCH):
            body = {
                "model": model,
                "input": [{"content": list(parts)} for parts in inputs[start : start + EMBED_BATCH]],
                "dimensions": dimensions,
                "encoding_format": "float",
                "provider": PROVIDER,
                **kind,
            }
            data = await self._post("embeddings", body, "embeddings", headers)
            vectors += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

        return vectors

    async def chat_stream(self, model: str, messages: list[Message], tools: list[dict[str, Any]]) -> AsyncIterator[str | Message]:
        body = {"model": model, "messages": messages, "provider": PROVIDER, "stream": True, **({"tools": tools} if tools else {})}
        content: list[str] = []
        calls: dict[int, Message] = {}

        async for data in self._events("chat/completions", body):
            delta = (data.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            content.append(piece)

            _merge_calls(calls, delta.get("tool_calls") or [])

            if piece:
                yield piece

        tool_calls = {"tool_calls": [calls[i] for i in sorted(calls)]} if calls else {}

        yield {"role": "assistant", "content": "".join(content) or None, **tool_calls}

    async def complete_json(self, model: str, parts: list[Part], name: str, schema: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": parts}],
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
            "provider": PROVIDER,
        }
        data = await self._post("chat/completions", body)

        return json.loads(data["choices"][0]["message"]["content"])

    async def _events(self, path: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async with self._ok(path, body) as resp:
            async for line in resp.content:
                data = _sse_data(line)

                if "error" in data:
                    raise OpenRouterError(resp.status, json.dumps(data["error"]))
                if data:
                    yield data

    async def _post(self, path: str, body: dict[str, Any], purpose: str = "chat", headers: dict[str, str] | None = None) -> dict[str, Any]:
        async with self._ok(path, body, purpose, headers) as resp:
            return json.loads(await resp.text())

    @contextlib.asynccontextmanager
    async def _ok(
        self, path: str, body: dict[str, Any], purpose: str = "chat", headers: dict[str, str] | None = None
    ) -> AsyncIterator[aiohttp.ClientResponse]:
        """A 200 response, retrying busy or failing upstreams with backoff before anything is read."""
        for attempt in range(RETRIES):
            async with self._http(purpose).post(path, json=body, headers=headers) as resp:
                if resp.status == 200:
                    yield resp
                    return

                status, payload = resp.status, await resp.text()

            if status not in RETRY_STATUSES or attempt == RETRIES - 1:
                raise OpenRouterError(status, payload)

            await asyncio.sleep(2**attempt)

def _sse_data(line: bytes) -> dict[str, Any]:
    text = line.decode().strip()
    payload = text.removeprefix("data:").strip() if text.startswith("data:") else ""

    return json.loads(payload) if payload and payload != "[DONE]" else {}


def _merge_calls(calls: dict[int, Message], fragments: list[dict[str, Any]]) -> None:
    """Streamed tool calls arrive in pieces keyed by index; join each piece onto its call."""
    for fragment in fragments:
        call = calls.setdefault(fragment["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        function = fragment.get("function") or {}
        call["id"] = fragment.get("id") or call["id"]
        call["function"]["name"] += function.get("name") or ""
        call["function"]["arguments"] += function.get("arguments") or ""
