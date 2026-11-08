from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from store import images
from store.archive import FrameArchive
from store.db import Frame
from store.store import Store

CAMERA = "cam1"
T0 = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
WIDTH, HEIGHT = 320, 180
GRID = (8, 4)

type Gray = NDArray[np.uint8]


def picture(seed: int) -> Gray:
    cells = np.random.default_rng(seed).integers(0, 256, (GRID[1], GRID[0]), dtype=np.uint8)

    return np.asarray(cv2.resize(cells, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST), dtype=np.uint8)


async def add_frame(store: Store, archive: FrameArchive, at: datetime, seed: int, event_id: int | None = None) -> str:
    path = archive.frame_path(CAMERA, at)
    archive.write(path, images.encode(picture(seed)))
    await store.add_frame(CAMERA, at, path, event_id, [])

    return path


async def frames(db: async_sessionmaker[AsyncSession]) -> list[Frame]:
    async with db() as s:
        return list(await s.scalars(select(Frame).order_by(Frame.captured_at)))


async def mark_key(db: async_sessionmaker[AsyncSession], at: datetime) -> None:
    async with db.begin() as s:
        await s.execute(update(Frame).where(Frame.captured_at == at).values(is_key=True))


def frame_at(video: Path, second: int) -> Gray:
    raw = subprocess.run(
        [
            *("ffmpeg", "-loglevel", "error", "-ss", str(second), "-i", str(video)),
            *("-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"),
        ],
        capture_output=True,
        check=True,
    ).stdout

    return np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH)


def frame_count(video: Path) -> int:
    out = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0"),
            *("-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout

    return int(out)


def psnr(a: Gray, b: Gray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

    return 10 * np.log10(255**2 / max(mse, 1e-9))


def available_encoders() -> set[str]:
    listing = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout

    return {line.split()[1] for line in listing.split("------\n", 1)[-1].splitlines()}
