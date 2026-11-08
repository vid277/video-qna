from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from models import CameraConfig, CameraEvent, CameraInfo, Direction, QualitySettings, Snapshot, StreamQuality


class Camera(ABC):
    def __init__(self, config: CameraConfig):
        self.config = config

    async def __aenter__(self) -> Camera:
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def info(self) -> CameraInfo: ...

    @abstractmethod
    async def snapshot(self, quality: StreamQuality) -> Snapshot: ...

    @abstractmethod
    def stream_url(self, quality: StreamQuality) -> str: ...

    @abstractmethod
    async def get_quality(self) -> QualitySettings: ...

    @abstractmethod
    async def set_quality(self, quality: QualitySettings) -> None: ...

    async def move(self, direction: Direction, seconds: float) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support PTZ")

    def events(self) -> AsyncIterator[CameraEvent]:
        raise NotImplementedError(f"{type(self).__name__} does not support events")
