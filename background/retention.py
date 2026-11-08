from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from itertools import accumulate, groupby

from models import StoredFrame
from store.runtime import archive, config, store


async def apply_retention(camera_id: str, now: datetime) -> None:
    midnight = now.astimezone().replace(hour=0, minute=0, second=0, microsecond=0)

    await thin_frames(camera_id, cutoff=midnight - timedelta(days=config.full_res_days))
    await enforce_disk_budget()


async def thin_frames(camera_id: str, cutoff: datetime) -> None:
    frames = await store.thinning_candidates(camera_id, cutoff)

    for hour, group in groupby(frames, key=lambda f: f.captured_at.replace(minute=0, second=0, microsecond=0)):
        kept, dropped = _thin(list(group), config.thin_every_s)
        video = archive.video_path(camera_id, hour)

        await archive.encode_video(_paths(kept), video, config.thin_height, config.video_encoder)
        await store.replace_with_video(kept, [f.id for f in dropped], video)

        archive.delete(_paths(kept + dropped))


async def enforce_disk_budget() -> None:
    excess = await asyncio.to_thread(archive.usage_bytes) - config.disk_budget_gb * 1e9
    videos = archive.videos_oldest_first()
    freed_before = accumulate((archive.size(v) for v in videos), initial=0)

    for video in [v for v, freed in zip(videos, freed_before) if freed < excess]:
        await store.drop_video(video)
        archive.delete([video])


def _thin(frames: list[StoredFrame], every_s: int) -> tuple[list[StoredFrame], list[StoredFrame]]:
    buckets = [list(g) for _, g in groupby(frames, key=lambda f: int(f.captured_at.timestamp() // every_s))]

    return [b[0] for b in buckets], [f for b in buckets for f in b[1:]]


def _paths(frames: list[StoredFrame]) -> list[str]:
    return [f.path for f in frames if f.path]
