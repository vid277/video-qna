from __future__ import annotations

import asyncio
import logging

from models import Caption, EventWork, StoredChange, StoredFrame
from openrouter import OpenRouter, image_part, text_part
from prompts import render
from store import images
from store.runtime import archive, config, store
from store.views import object_view

MAX_CHANGE_IMAGES = 4

SCHEMA = Caption.model_json_schema() | {"additionalProperties": False}

log = logging.getLogger(__name__)


class Captioner:
    def __init__(self, client: OpenRouter):
        self._client = client

    async def caption_event(self, work: EventWork) -> None:
        frames = await store.key_frames(work.event_id)
        changes = await store.event_changes(work.event_id)
        sheet = await asyncio.to_thread(_contact_sheet, frames)
        close_ups = [archive.read(object_view(c)) for c in changes[:MAX_CHANGE_IMAGES]]
        previous = await store.previous_summary(work.event_id)

        prompt = render("caption", changes=_describe(changes), previous=previous or "none")
        parts = [text_part(prompt), image_part(sheet), *map(image_part, close_ups)]
        data = await self._client.complete_json(config.caption_model, parts, "event_caption", SCHEMA)

        await store.set_caption(work.event_id, Caption.model_validate(data))
        log.info("captioned event %s", work.event_id)


def _contact_sheet(frames: list[StoredFrame]) -> bytes:
    decoded = [images.decode(archive.frame_image(f)) for f in frames]

    return images.encode(images.contact_sheet(decoded, config.model_image_width))


def _describe(changes: list[StoredChange]) -> str:
    lines = [f"Image {i}: an object that was {c.kind.value}." for i, c in enumerate(changes[:MAX_CHANGE_IMAGES], start=2)]

    return "\n".join(lines) or "No objects were detected being added, removed or moved."
