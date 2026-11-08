from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ingest import retention
from ingest.archive import FrameArchive
from ingest.db import Frame
from ingest.store import Store
from models import IngestConfig, VideoEncoder
from tests.factory import (
    CAMERA,
    T0,
    add_frame,
    available_encoders,
    frame_at,
    frame_count,
    frames,
    mark_key,
    picture,
    psnr,
)

pytestmark = pytest.mark.usefixtures("runtime")

ENCODERS = [
    pytest.param(e, marks=pytest.mark.skipif(e.value not in available_encoders(), reason=f"ffmpeg lacks {e.value}"))
    for e in VideoEncoder
]


async def test_thin_frames_keeps_key_frames_and_encodes_one_video_per_hour(
    store: Store, archive: FrameArchive, config: IngestConfig, db: async_sessionmaker[AsyncSession]
) -> None:
    first_hour = [T0 + timedelta(seconds=s) for s in range(30)]
    second_hour = [T0 + timedelta(hours=1, seconds=s) for s in range(10)]
    key, recent = T0 + timedelta(seconds=7), T0 + timedelta(days=2)

    for seed, at in enumerate([*first_hour, *second_hour, recent]):
        await add_frame(store, archive, at, seed)

    await mark_key(db, key)
    before = {f.captured_at: f.path for f in await frames(db)}

    await retention.thin_frames(CAMERA, cutoff=T0 + timedelta(days=1))

    rows = {f.captured_at: f for f in await frames(db)}
    kept = [first_hour[0:30:5], second_hour[0:10:5]]
    assert set(rows) == {*kept[0], *kept[1], key, recent}

    for at in (key, recent):
        assert (rows[at].path, rows[at].video_path) == (before[at], None)
        assert (config.data_dir / str(before[at])).exists()

    videos = [{rows[at].video_path for at in hour} for hour in kept]
    assert [len(v) for v in videos] == [1, 1] and videos[0] != videos[1]

    for hour, (video,) in zip(kept, videos):
        assert [(rows[at].path, rows[at].video_index) for at in hour] == [(None, i) for i in range(len(hour))]
        assert frame_count(config.data_dir / str(video)) == len(hour)

    removed = set(before) - {key, recent}
    assert not [at for at in removed if (config.data_dir / str(before[at])).exists()]


@pytest.mark.parametrize("encoder", ENCODERS)
async def test_video_index_is_the_frame_at_that_second(
    monkeypatch: pytest.MonkeyPatch,
    encoder: VideoEncoder,
    store: Store,
    archive: FrameArchive,
    config: IngestConfig,
    db: async_sessionmaker[AsyncSession],
) -> None:
    monkeypatch.setattr(retention, "config", config.model_copy(update={"video_encoder": encoder}))
    seeds = {T0 + timedelta(seconds=s): s for s in range(60)}

    for at, seed in seeds.items():
        await add_frame(store, archive, at, seed)

    await retention.thin_frames(CAMERA, cutoff=T0 + timedelta(hours=1))

    rows = [f for f in await frames(db) if f.video_path]
    sources = [picture(seeds[f.captured_at]) for f in rows]
    assert [f.video_index for f in rows] == list(range(12))

    for row in rows:
        decoded = frame_at(config.data_dir / str(row.video_path), int(str(row.video_index)))
        scores = [psnr(decoded, source) for source in sources]

        assert scores.index(max(scores)) == row.video_index
        assert max(scores) > 25


async def test_enforce_disk_budget_deletes_oldest_videos_first(
    monkeypatch: pytest.MonkeyPatch,
    store: Store,
    archive: FrameArchive,
    config: IngestConfig,
    db: async_sessionmaker[AsyncSession],
) -> None:
    videos = [archive.video_path(CAMERA, T0 + timedelta(hours=h)) for h in range(3)]
    key_jpeg = await add_frame(store, archive, T0 - timedelta(days=1), 0)
    await mark_key(db, T0 - timedelta(days=1))

    async with db.begin() as s:
        for h, video in enumerate(videos):
            archive.write(video, bytes(1000))
            s.add(Frame(camera_id=CAMERA, captured_at=T0 + timedelta(hours=h), video_path=video, video_index=0))

        s.add(Frame(camera_id=CAMERA, captured_at=T0, video_path=videos[0], video_index=1, is_key=True))

    budget_gb = (archive.usage_bytes() - 1500) / 1e9
    monkeypatch.setattr(retention, "config", config.model_copy(update={"disk_budget_gb": budget_gb}))

    await retention.enforce_disk_budget()

    assert [(config.data_dir / v).exists() for v in videos] == [False, False, True]
    assert [(f.video_path, f.is_key) for f in await frames(db)] == [(None, True), (videos[0], True), (videos[2], False)]
    assert (config.data_dir / key_jpeg).exists()


async def test_enforce_disk_budget_never_deletes_key_frames(
    monkeypatch: pytest.MonkeyPatch,
    store: Store,
    archive: FrameArchive,
    config: IngestConfig,
    db: async_sessionmaker[AsyncSession],
) -> None:
    keys = [T0 + timedelta(minutes=m) for m in range(3)]

    for seed, at in enumerate(keys):
        await add_frame(store, archive, at, seed)
        await mark_key(db, at)

    archive.write(archive.video_path(CAMERA, T0), bytes(1000))
    monkeypatch.setattr(retention, "config", config.model_copy(update={"disk_budget_gb": 1e-9}))

    await retention.enforce_disk_budget()

    rows = await frames(db)
    assert [f.captured_at for f in rows] == keys
    assert all((config.data_dir / str(f.path)).exists() for f in rows)
    assert archive.videos_oldest_first() == []
