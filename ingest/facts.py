from __future__ import annotations

from datetime import datetime

from sqlalchemy import Text, cast, exists, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ingest.db import EARLIEST, EMBEDDING_DIM, LATEST, Embedding, Event, Fact, JobFailure, sqlalchemy_url
from ingest.runtime import config, store
from models import EmbeddingRow, FactHit, TextKind, VectorOwner
from openrouter import OpenRouter, text_part

class FactIndex:
    """Caption facts as text vectors, so a question finds descriptions by meaning, not exact words."""

    def __init__(self, database_url: str, embedding_model: str):
        self._engine = create_async_engine(sqlalchemy_url(database_url))
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._model = embedding_model

    async def close(self) -> None:
        await self._engine.dispose()

    async def pending(self, limit: int) -> list[str]:
        has_facts = exists().where(Fact.event_id == Event.id)
        has_vectors = exists().where(
            Embedding.owner == VectorOwner.FACT.value, Embedding.owner_id == Event.id, Embedding.model == self._model
        )
        blocked = exists().where(
            JobFailure.kind == TextKind.FACT_VECTORS.value,
            JobFailure.key == cast(Event.id, Text),
            JobFailure.retry_at > func.now(),
        )
        query = select(Event.id).where(has_facts, ~has_vectors, ~blocked).order_by(Event.started_at.desc()).limit(limit)

        async with self._session() as s:
            return [str(i) for i in (await s.scalars(query)).all()]

    async def embed(self, event_id: int, client: OpenRouter) -> None:
        async with self._session() as s:
            event = await s.get_one(Event, event_id)
            facts = list((await s.scalars(select(Fact.text).where(Fact.event_id == event_id).order_by(Fact.id))).all())

        vectors = await client.embed(config.embedding_model, [[text_part(f)] for f in facts], EMBEDDING_DIM, "document")
        rows = [
            EmbeddingRow(
                owner=VectorOwner.FACT,
                owner_id=event_id,
                part=part,
                camera_id=event.camera_id,
                event_id=event_id,
                captured_at=event.started_at,
                vector=vector,
            )
            for part, vector in enumerate(vectors)
        ]

        await store.replace_embeddings(rows)

    async def search(self, vector: list[float], start: datetime | None, end: datetime | None, limit: int) -> list[FactHit]:
        similarity = 1 - Embedding.vector.cosine_distance(vector)
        ranked = (
            select(Embedding.event_id, Embedding.part, Embedding.captured_at, similarity.label("score"))
            .where(Embedding.owner == VectorOwner.FACT.value, Embedding.model == self._model)
            .where(Embedding.captured_at.between(start or EARLIEST, end or LATEST))
            .order_by(similarity.desc())
            .limit(limit)
            .subquery()
        )
        numbered = select(Fact.event_id, Fact.text, (func.row_number().over(partition_by=Fact.event_id, order_by=Fact.id) - 1).label("part")).subquery()
        query = (
            select(ranked.c.event_id, ranked.c.captured_at, numbered.c.text, ranked.c.score)
            .join(numbered, (numbered.c.event_id == ranked.c.event_id) & (numbered.c.part == ranked.c.part))
            .order_by(ranked.c.score.desc())
        )

        async with self._session() as s:
            rows = (await s.execute(query)).all()

        return [FactHit(ref=f"event:{e}", started_at=at, text=text, rank=round(float(score), 4)) for e, at, text, score in rows]


facts = FactIndex(config.database_url, config.embedding_model)
