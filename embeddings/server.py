#!/usr/bin/env python3
"""Local Qwen3-VL-Embedding service speaking the OpenRouter /v1/embeddings API."""

from __future__ import annotations

import asyncio
import base64
import io
import itertools
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
import yaml
from aiohttp import web
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

sys.path.insert(0, str(Path(__file__).with_name("vendor")))

from qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

CONFIG_PATH = Path(__file__).with_name("embeddings.yaml")
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

log = logging.getLogger("embeddings")


class Record(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class Settings(Record):
    model: str
    host: str
    port: int = Field(gt=0, lt=65536)
    device: str
    dtype: Literal["float16", "bfloat16", "float32"]
    max_batch: int = Field(ge=1)
    query_instruction: str

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        return cls.model_validate_json(json.dumps(yaml.safe_load(path.read_text())))


class ImageUrl(Record):
    url: str


class Part(Record):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class EmbedInput(Record):
    content: list[Part] = Field(min_length=1)


class EmbedRequest(Record):
    model: str
    input: list[EmbedInput] = Field(min_length=1)
    dimensions: int | None = Field(default=None, ge=32)
    encoding_format: Literal["float"] = "float"
    input_type: Literal["query", "document"] | None = None
    provider: dict[str, Any] | None = None


QUERY, INTERACTIVE, DOCUMENT = 0, 1, 2
PRIORITY_HEADER = "X-Embedding-Priority"

type Job = tuple[int, int, Callable[[], list[list[float]]], asyncio.Future[list[list[float]]]]


class Scheduler:
    """Runs one model batch at a time, always taking queries before queued image batches."""

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue()
        self._order = itertools.count()

    async def submit(self, priority: int, work: Callable[[], list[list[float]]]) -> list[list[float]]:
        done: asyncio.Future[list[list[float]]] = asyncio.get_running_loop().create_future()
        await self._queue.put((priority, next(self._order), work, done))

        return await done

    async def run(self) -> None:
        while True:
            _, _, work, done = await self._queue.get()

            if done.cancelled():
                continue

            try:
                done.set_result(await asyncio.to_thread(work))
            except Exception as e:
                done.set_exception(e)


class Embedder:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = Qwen3VLEmbedder(settings.model, torch_dtype=DTYPES[settings.dtype])
        self._model.model.to(settings.device)
        self.scheduler = Scheduler()

    async def embed(self, request: EmbedRequest, interactive: bool = False) -> list[list[float]]:
        inputs = [self._model_input(i, request.input_type) for i in request.input]
        batches = [inputs[i : i + self._settings.max_batch] for i in range(0, len(inputs), self._settings.max_batch)]
        priority = _priority(request, interactive)
        results = await asyncio.gather(
            *(self.scheduler.submit(priority, lambda b=b: self._run(b, request.dimensions)) for b in batches)
        )

        return [v for batch in results for v in batch]

    def _run(self, batch: list[dict[str, Any]], dimensions: int | None) -> list[list[float]]:
        raw = self._model.process(batch, normalize=False)

        return F.normalize(raw[:, :dimensions].float(), p=2, dim=-1).cpu().tolist()

    def _model_input(self, item: EmbedInput, input_type: str | None) -> dict[str, Any]:
        texts = [p.text for p in item.content if p.type == "text" and p.text]
        images = [_image(p.image_url.url) for p in item.content if p.image_url]
        visual = {0: {}, 1: {"image": images[:1]}}.get(len(images), {"video": images})
        is_query = input_type == "query" or (input_type is None and not images)
        query = {"instruction": self._settings.query_instruction} if is_query else {}

        text = {"text": " ".join(texts)} if texts else {}

        return text | visual | query


def _priority(request: EmbedRequest, interactive: bool) -> int:
    text = all(_text_only(i) for i in request.input)

    return {(True, True): QUERY, (True, False): QUERY, (False, True): INTERACTIVE}.get((text, interactive), DOCUMENT)


def _text_only(item: EmbedInput) -> bool:
    return all(p.type == "text" for p in item.content)


def _image(url: str) -> Image.Image:
    _, _, data = url.partition("base64,")

    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


class Service:
    def __init__(self, settings: Settings, embedder: Embedder):
        self._settings = settings
        self._embedder = embedder

    def app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024**2)
        app.add_routes([web.post("/v1/embeddings", self.embeddings), web.get("/health", self.health)])
        app.cleanup_ctx.append(self._scheduling)

        return app

    async def _scheduling(self, _app: web.Application):
        runner = asyncio.create_task(self._embedder.scheduler.run())

        yield

        runner.cancel()

    async def embeddings(self, request: web.Request) -> web.Response:
        try:
            body = EmbedRequest.model_validate_json(await request.text())
        except ValidationError as e:
            return web.json_response({"error": {"code": 400, "message": str(e)}}, status=400)

        started = time.perf_counter()
        interactive = request.headers.get(PRIORITY_HEADER) == "interactive"
        vectors = await self._embedder.embed(body, interactive)
        kind = {QUERY: "query", INTERACTIVE: "interactive", DOCUMENT: "background"}[_priority(body, interactive)]
        log.info("embedded %d %s inputs in %.2fs", len(vectors), kind, time.perf_counter() - started)

        return web.json_response(
            {
                "object": "list",
                "model": self._settings.model,
                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
            }
        )

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"model": self._settings.model, "device": self._settings.device})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_yaml(CONFIG_PATH)
    log.info("loading %s on %s", settings.model, settings.device)

    service = Service(settings, Embedder(settings))
    web.run_app(service.app(), host=settings.host, port=settings.port, print=None)


if __name__ == "__main__":
    main()
