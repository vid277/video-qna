from __future__ import annotations

import asyncio
import contextlib
import subprocess
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from subprocess import DEVNULL, PIPE

from models import StoredFrame, VideoEncoder

ENCODER_ARGS: dict[VideoEncoder, tuple[str, ...]] = {
    VideoEncoder.LIBX265: ("-crf", "28", "-x265-params", "log-level=error"),
    VideoEncoder.VIDEOTOOLBOX: ("-q:v", "60"),
    VideoEncoder.NVENC: ("-preset", "p4", "-rc", "vbr", "-cq", "30", "-b:v", "0"),
    VideoEncoder.QSV: ("-preset", "faster", "-global_quality", "28"),
}


class FrameArchive:
    def __init__(self, root: Path):
        self.root = root

    def frame_path(self, camera_id: str, at: datetime) -> str:
        return f"frames/{camera_id}/{at:%Y-%m-%d/%H/%H%M%S}_{at.microsecond // 1000:03d}.jpg"

    def change_path(self, camera_id: str, at: datetime, event_id: int, index: int, side: str) -> str:
        return f"changes/{camera_id}/{at:%Y-%m-%d}/{at:%H%M%S}_{event_id}_{index}_{side}.jpg"

    def video_path(self, camera_id: str, hour: datetime) -> str:
        return f"videos/{camera_id}/{hour:%Y-%m-%d/%H}.mp4"

    def write(self, path: str, data: bytes) -> None:
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)

    def delete(self, paths: Iterable[str]) -> None:
        for path in paths:
            (self.root / path).unlink(missing_ok=True)

    def read(self, path: str) -> bytes:
        return (self.root / path).read_bytes()

    def exists(self, path: str) -> bool:
        return (self.root / path).is_file()

    def has_frame(self, frame: StoredFrame) -> bool:
        return self.exists(frame.path or frame.video_path or "")

    def frame_image(self, frame: StoredFrame) -> bytes:
        return self.read(frame.path) if frame.path else self._video_frame(str(frame.video_path), frame.video_index or 0)

    def _video_frame(self, video_path: str, index: int) -> bytes:
        return subprocess.run(
            [
                *("ffmpeg", "-loglevel", "error", "-ss", str(index), "-i", str(self.root / video_path)),
                *("-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-"),
            ],
            capture_output=True,
            check=True,
        ).stdout

    def size(self, path: str) -> int:
        return (self.root / path).stat().st_size

    def usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    def videos_oldest_first(self) -> list[str]:
        videos = [f.relative_to(self.root) for f in (self.root / "videos").rglob("*.mp4")]

        return [str(v) for v in sorted(videos, key=lambda v: (v.parts[2:], v.parts[1]))]

    async def encode_video(self, frames: list[str], out: str, height: int, encoder: VideoEncoder) -> None:
        target = self.root / out
        partial = target.with_name(f"{target.name}.part")
        target.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", suffix=".txt") as playlist:
            playlist.write("ffconcat version 1.0\n")
            playlist.writelines(f"file '{(self.root / f).resolve()}'\nduration 1\n" for f in frames)
            playlist.flush()

            command = [
                *("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", playlist.name),
                *("-vf", f"scale=-2:{height},format=yuv420p", "-r", "1"),
                *("-c:v", encoder.value, *ENCODER_ARGS[encoder], "-tag:v", "hvc1", "-f", "mp4", str(partial)),
            ]
            ffmpeg = await asyncio.create_subprocess_exec(*command, stdout=DEVNULL, stderr=PIPE)

            try:
                _, stderr = await ffmpeg.communicate()
            except BaseException:
                with contextlib.suppress(ProcessLookupError):
                    ffmpeg.kill()

                await ffmpeg.wait()
                partial.unlink(missing_ok=True)
                raise

        if ffmpeg.returncode:
            partial.unlink(missing_ok=True)
            raise subprocess.CalledProcessError(int(ffmpeg.returncode), command, stderr=stderr)

        partial.replace(target)
