from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class StreamQuality(StrEnum):
    MAIN = "main"
    SUB = "sub"


class Direction(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class EventKind(StrEnum):
    MOTION = "motion"
    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"


class Record(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class StreamSettings(Record):
    frame_rate: int = Field(gt=0)
    bit_rate: int = Field(gt=0)


class QualitySettings(Record):
    main: StreamSettings
    sub: StreamSettings


class PtzSettings(Record):
    speed: int = Field(gt=0, le=64)
    seconds: float = Field(gt=0)


class CameraConfig(Record):
    name: str
    host: str
    username: str
    password: SecretStr
    channel: int = Field(ge=0)
    rtsp_port: int = Field(gt=0, lt=65536)
    snapshot_quality: StreamQuality
    stream_quality: StreamQuality
    ptz: PtzSettings
    quality: QualitySettings


class CamerasFile(Record):
    cameras: list[CameraConfig] = Field(min_length=1)

    @classmethod
    def from_yaml(cls, path: Path) -> CamerasFile:
        return cls.model_validate_json(json.dumps(yaml.safe_load(path.read_text())))


class CameraInfo(Record):
    camera_id: str
    name: str
    vendor: str
    model: str
    hardware: str
    firmware: str
    host: str


class CameraEvent(Record):
    camera_id: str
    kind: EventKind
    active: bool
    occurred_at: datetime


class Snapshot(Record):
    camera_id: str
    captured_at: datetime
    quality: StreamQuality
    image: bytes


class CameraPort(Record):
    camera_id: str
    protocol: str
    port: int = Field(gt=0, lt=65536)
    enabled: bool
