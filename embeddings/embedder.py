from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from models import EmbeddingRow, EventWork, StoredChange, StoredFrame, VectorOwner
from openrouter import OpenRouter, Part, image_part
from store import views
from store.db import EMBEDDING_DIM
from store.runtime import archive, config, store

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, client: OpenRouter):
        self._client = client

    async def embed_changes(self, event_id: int) -> None:
        work = await store.event_work(event_id)
        changes = await store.event_changes(event_id)
        jpegs = await asyncio.to_thread(_change_images, changes)
        vectors = await self._embed([[image_part(j)] for j in jpegs])

        await store.replace_embeddings(
            [_row(work, VectorOwner.CHANGE, c.id, 0, c.detected_at, v) for c, v in zip(changes, vectors, strict=True)]
        )
        log.info("embedded %d changes for event %s", len(changes), event_id)

    async def embed_frames(self, event_id: int) -> None:
        work = await store.event_work(event_id)
        frames = await store.key_frames(event_id)
        jpegs = await asyncio.to_thread(_frame_images, frames)
        vectors = await self._embed([[image_part(j)] for j in jpegs])

        await store.replace_embeddings(
            [_row(work, VectorOwner.FRAME, f.id, 0, f.captured_at, v) for f, v in zip(frames, vectors, strict=True)]
        )
        log.info("embedded %d frames for event %s", len(frames), event_id)

    async def embed_clip(self, event_id: int) -> None:
        work = await store.event_work(event_id)
        jpegs = await asyncio.to_thread(_frame_images, await store.key_frames(event_id))
        [clip] = await self._embed([[image_part(j) for j in jpegs]])

        await store.replace_embeddings([_row(work, VectorOwner.CLIP, event_id, 0, work.captured_at, clip)])
        log.info("embedded the clip for event %s", event_id)

    async def embed_crops(self, event_id: int) -> None:
        work = await store.event_work(event_id)
        frames = await store.key_frames(event_id)
        crops = await asyncio.to_thread(_crop_images, frames)
        vectors = await self._embed([[image_part(c)] for per_frame in crops for c in per_frame])
        parts = [(f, part) for f, per_frame in zip(frames, crops, strict=True) for part in range(1, len(per_frame) + 1)]

        await store.replace_embeddings(
            [_row(work, VectorOwner.CROP, f.id, part, f.captured_at, v) for (f, part), v in zip(parts, vectors, strict=True)]
        )
        log.info("embedded crops for event %s", event_id)

    async def _embed(self, inputs: list[list[Part]]) -> list[list[float]]:
        return await self._client.embed(config.embedding_model, inputs, EMBEDDING_DIM)


def _frame_images(frames: list[StoredFrame]) -> list[bytes]:
    return [views.frame_image(f) for f in frames]


def _change_images(changes: list[StoredChange]) -> list[bytes]:
    return [archive.read(views.object_view(c)) for c in changes]


def _crop_images(frames: list[StoredFrame]) -> list[list[bytes]]:
    return [views.frame_crops(f) for f in frames]


def _row(work: EventWork, owner: VectorOwner, owner_id: int, part: int, at: datetime, vector: list[float]) -> EmbeddingRow:
    return EmbeddingRow(
        owner=owner,
        owner_id=owner_id,
        part=part,
        camera_id=work.camera_id,
        event_id=work.event_id,
        captured_at=at,
        vector=vector,
    )
