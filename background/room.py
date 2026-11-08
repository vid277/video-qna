from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import cv2
import numpy as np
from sqlalchemy import ColumnElement, Text, cast, exists, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import aliased

from models import (
    Box,
    ChangeCategory,
    ChangeKind,
    ChangeLabel,
    ItemChange,
    RoomItem,
    RoomKind,
    StoredFrame,
    VectorOwner,
)
from openrouter import OpenRouter, image_part, text_part
from prompts import render
from store import images
from store.db import Embedding, Event, Frame, JobFailure, SceneChange, sqlalchemy_url
from store.runtime import archive, config
from store.views import model_image

PRESENT = {ChangeKind.ADDED.value, ChangeKind.MOVED.value, ChangeKind.CHANGED.value}
OUTLINE = (0, 0, 255)
LABELLED_KINDS = [ChangeKind.ADDED.value, ChangeKind.REMOVED.value, ChangeKind.MOVED.value]
CONTEXT_SCALE = 5
CONTEXT_MIN_PX = 480

SCHEMA = ChangeLabel.model_json_schema() | {"additionalProperties": False}

log = logging.getLogger(__name__)


class RoomLog:
    def __init__(self, database_url: str, embedding_model: str):
        self._engine = create_async_engine(sqlalchemy_url(database_url))
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._model = embedding_model

    async def close(self) -> None:
        await self._engine.dispose()

    async def pending(self, kind: RoomKind, limit: int) -> list[str]:
        blocked = exists().where(
            JobFailure.kind == kind.value, JobFailure.key == cast(SceneChange.id, Text), JobFailure.retry_at > func.now()
        )
        earlier = aliased(SceneChange)
        awaiting_label = exists().where(earlier.detected_at < SceneChange.detected_at, *_needs_label(earlier))
        conditions: dict[RoomKind, list[ColumnElement[bool]]] = {
            RoomKind.LABELS: _needs_label(SceneChange),
            RoomKind.LINKS: [
                SceneChange.category == ChangeCategory.OBJECT.value,
                SceneChange.item_id.is_(None),
                self._has_vector(),
                ~awaiting_label,
            ],
        }
        order = {RoomKind.LABELS: SceneChange.detected_at.desc(), RoomKind.LINKS: SceneChange.detected_at.asc()}
        query = select(SceneChange.id).where(~blocked, *conditions[kind]).order_by(order[kind]).limit(limit)

        async with self._session() as s:
            return [str(i) for i in (await s.scalars(query)).all()]

    async def label(self, change_id: int, client: OpenRouter) -> None:
        row, camera_id = await self._change(change_id)
        change = row.stored()
        box = Box.model_validate(row.box)

        frame = await self._frame_after(camera_id, change.detected_at)
        crops = [model_image(archive.read(p)) for p in (change.before_path, change.after_path) if archive.exists(p)]
        views = await asyncio.to_thread(_views, frame, box) if frame and archive.has_frame(frame) else []

        prompt = render("change_label", kind=change.kind.value)
        parts = [text_part(prompt), *map(image_part, views[:1] + crops + views[1:])]
        result = parse_label(await client.complete_json(config.caption_model, parts, "change_label", SCHEMA))
        values = {"category": result.category.value, "label": result.label, "place": result.place}

        async with self._session.begin() as s:
            await s.execute(update(SceneChange).where(SceneChange.id == change_id).values(**values))

        log.info("change %s: %s %s, %s", change_id, result.category.value, result.label, result.place)

    async def context(self, change_id: int) -> bytes | None:
        """A wide view around a change, at the moment it was seen."""
        row, camera_id = await self._change(change_id)
        frame = await self._frame_after(camera_id, row.detected_at)

        return (await asyncio.to_thread(_views, frame, Box.model_validate(row.box)))[0] if frame and archive.has_frame(frame) else None

    async def link(self, change_id: int) -> None:
        row, camera_id = await self._change(change_id)
        kind, box = ChangeKind(row.kind), Box.model_validate(row.box)
        source = Box.model_validate(row.source) if row.source else box
        latest = await self._latest(camera_id, row.detected_at)
        present = [r for r in latest if r.kind in PRESENT]
        gone = [r for r in latest if r.kind not in PRESENT]

        at_spot = {ChangeKind.REMOVED: box, ChangeKind.CHANGED: box, ChangeKind.MOVED: source}
        candidates = present if kind in at_spot else gone
        spot = at_spot.get(kind)
        item_id = (_overlapping(candidates, spot) if spot else None) or await self._most_similar(change_id, candidates)

        async with self._session.begin() as s:
            await s.execute(update(SceneChange).where(SceneChange.id == change_id).values(item_id=item_id or change_id))

    async def state(self, at: datetime | None) -> list[RoomItem]:
        latest = await self._latest(None, at or datetime.now(UTC))

        return [
            RoomItem(item_ref=f"item:{r.item_id}", label=r.label, place=r.place, since=r.changed_after, evidence=f"change:{r.id}")
            for r in latest
            if r.kind in PRESENT
        ]

    async def history(self, item_ids: list[int]) -> list[ItemChange]:
        query = select(SceneChange).where(SceneChange.item_id.in_(item_ids)).order_by(SceneChange.detected_at)

        async with self._session() as s:
            return [_item_change(c) for c in (await s.scalars(query)).all()]

    async def items_labelled(self, words: str, limit: int) -> list[int]:
        matches = func.to_tsvector("english", SceneChange.label).op("@@")(func.plainto_tsquery("english", words))
        query = (
            select(SceneChange.item_id)
            .where(SceneChange.item_id.is_not(None), matches)
            .group_by(SceneChange.item_id)
            .order_by(func.max(SceneChange.detected_at).desc())
            .limit(limit)
        )

        async with self._session() as s:
            return [i for i in (await s.scalars(query)).all() if i is not None]

    async def items_for_changes(self, change_ids: list[int]) -> list[int]:
        query = select(SceneChange.item_id).where(SceneChange.id.in_(change_ids), SceneChange.item_id.is_not(None))

        async with self._session() as s:
            return list(dict.fromkeys(i for i in (await s.scalars(query)).all() if i is not None))

    async def _change(self, change_id: int) -> tuple[SceneChange, str]:
        query = select(SceneChange, Event.camera_id).join(Event, Event.id == SceneChange.event_id).where(SceneChange.id == change_id)

        async with self._session() as s:
            row, camera_id = (await s.execute(query)).one()

        return row, camera_id

    async def _frame_after(self, camera_id: str, at: datetime) -> StoredFrame | None:
        query = select(Frame).where(Frame.camera_id == camera_id, Frame.captured_at >= at).order_by(Frame.captured_at).limit(1)

        async with self._session() as s:
            frame = await s.scalar(query)

        return frame.stored() if frame else None

    async def _latest(self, camera_id: str | None, before: datetime) -> list[SceneChange]:
        cameras = [Event.camera_id == camera_id] if camera_id else []
        newest = func.row_number().over(partition_by=SceneChange.item_id, order_by=SceneChange.detected_at.desc())
        ranked = (
            select(SceneChange.id, newest.label("n"))
            .join(Event, Event.id == SceneChange.event_id)
            .where(SceneChange.item_id.is_not(None), SceneChange.detected_at <= before, *cameras)
            .subquery()
        )
        query = select(SceneChange).join(ranked, ranked.c.id == SceneChange.id).where(ranked.c.n == 1).order_by(SceneChange.detected_at.desc())

        async with self._session() as s:
            return list((await s.scalars(query)).all())

    async def _most_similar(self, change_id: int, candidates: list[SceneChange]) -> int | None:
        vectors = await self._vectors([change_id, *[c.id for c in candidates]])
        mine = vectors.get(change_id)
        scored = [(float(mine @ vectors[c.id]), c.item_id) for c in candidates if mine is not None and c.id in vectors]
        similarity, item_id = max(scored, default=(0.0, None), key=lambda s: s[0])

        return item_id if similarity >= config.item_link_similarity else None

    async def _vectors(self, change_ids: list[int]) -> dict[int, np.ndarray]:
        query = select(Embedding.owner_id, Embedding.vector).where(
            Embedding.owner == VectorOwner.CHANGE.value, Embedding.model == self._model, Embedding.owner_id.in_(change_ids)
        )

        async with self._session() as s:
            rows = (await s.execute(query)).all()

        return {owner_id: _unit(np.asarray(vector, np.float32)) for owner_id, vector in rows}

    def _has_vector(self) -> ColumnElement[bool]:
        return exists().where(
            Embedding.owner == VectorOwner.CHANGE.value, Embedding.owner_id == SceneChange.id, Embedding.model == self._model
        )


def parse_label(reply: dict[str, object]) -> ChangeLabel:
    """Model replies carry enums as plain strings, which strict models accept only from JSON."""
    return ChangeLabel.model_validate_json(json.dumps(reply))


def _views(frame: StoredFrame, box: Box) -> list[bytes]:
    """A wide crop around the box, then the whole frame, both with the box outlined."""
    full = images.decode(archive.frame_image(frame))
    cv2.rectangle(full, (box.x, box.y), (box.x + box.w, box.y + box.h), OUTLINE, max(2, full.shape[1] // 300))

    h, w = full.shape[:2]
    half = max(CONTEXT_MIN_PX, CONTEXT_SCALE * max(box.w, box.h)) // 2
    cx, cy = box.x + box.w // 2, box.y + box.h // 2
    wide = full[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]

    return [model_image(images.encode(wide)), model_image(images.encode(full))]


def _needs_label(change: type[SceneChange]) -> list[ColumnElement[bool]]:
    """Unlabelled add/remove/move changes, unless their event changed so much it is lighting; links wait for these."""
    siblings = aliased(SceneChange)
    count = select(func.count()).where(siblings.event_id == change.event_id, siblings.kind.in_(LABELLED_KINDS)).scalar_subquery()
    blocked = exists().where(
        JobFailure.kind == RoomKind.LABELS.value, JobFailure.key == cast(change.id, Text), JobFailure.retry_at > func.now()
    )

    return [change.label.is_(None), change.kind.in_(LABELLED_KINDS), count <= config.label_max_event_changes, ~blocked]


def _overlapping(rows: list[SceneChange], box: Box) -> int | None:
    scored = [(_overlap(Box.model_validate(r.box), box), r.item_id) for r in rows]
    overlap, item_id = max(scored, default=(0.0, None), key=lambda s: s[0])

    return item_id if overlap > 0 else None


def _overlap(a: Box, b: Box) -> float:
    w = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    h = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))

    return w * h / min(a.area, b.area)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _item_change(c: SceneChange) -> ItemChange:
    return ItemChange(
        ref=f"change:{c.id}",
        item_ref=f"item:{c.item_id}",
        kind=ChangeKind(c.kind),
        label=c.label,
        place=c.place,
        changed_after=c.changed_after,
        detected_at=c.detected_at,
    )


room = RoomLog(config.database_url, config.embedding_model)
