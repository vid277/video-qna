from __future__ import annotations

import asyncio
import subprocess
from datetime import timedelta

import cv2
import numpy as np
import pytest

from ingest import images
from ingest.archive import FrameArchive
from models import VideoEncoder
from tests.factory import T0


def test_videos_oldest_first_orders_by_hour_across_cameras(archive: FrameArchive) -> None:
    order = [("b", 0), ("a", 1), ("b", 2), ("a", 25)]

    for camera, hours in reversed(order):
        archive.write(archive.video_path(camera, T0 + timedelta(hours=hours)), b"x")

    assert archive.videos_oldest_first() == [archive.video_path(c, T0 + timedelta(hours=h)) for c, h in order]


async def test_cancelled_encode_kills_ffmpeg_and_leaves_no_file(archive: FrameArchive) -> None:
    noise = np.random.default_rng(0).integers(0, 256, (720, 1280), dtype=np.uint8)
    frames = [f"frames/{i}.jpg" for i in range(300)]

    for i, path in enumerate(frames):
        archive.write(path, images.encode(np.asarray(np.roll(cv2.GaussianBlur(noise, (5, 5), 0), i, axis=1))))

    out = "videos/cam/cancelled.mp4"
    encode = asyncio.create_task(archive.encode_video(frames, out, 720, VideoEncoder.LIBX265))
    running = False

    for _ in range(50):
        await asyncio.sleep(0.1)
        running = subprocess.run(["pgrep", "-f", f"{out}.part"], capture_output=True).returncode == 0
        if running:
            break

    encode.cancel()

    with pytest.raises(asyncio.CancelledError):
        await encode

    assert running
    assert subprocess.run(["pgrep", "-f", f"{out}.part"], capture_output=True).returncode == 1
    assert sorted(p.name for p in (archive.root / "videos/cam").iterdir()) == []


async def test_failed_encode_raises_and_leaves_no_file(archive: FrameArchive) -> None:
    archive.write("frames/bad.jpg", b"not a jpeg")

    with pytest.raises(subprocess.CalledProcessError):
        await archive.encode_video(["frames/bad.jpg"], "videos/cam/bad.mp4", 180, VideoEncoder.LIBX265)

    assert list((archive.root / "videos/cam").iterdir()) == []
