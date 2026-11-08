from __future__ import annotations

import io
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml
from dotenv import dotenv_values
from pydantic import Field, SecretStr

from models.camera import Record


class LightMode(StrEnum):
    DAY = "day"
    NIGHT = "night"


class ChangeKind(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    CHANGED = "changed"


class VideoEncoder(StrEnum):
    LIBX265 = "libx265"
    VIDEOTOOLBOX = "hevc_videotoolbox"
    NVENC = "hevc_nvenc"
    QSV = "hevc_qsv"


class Box(Record):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    @property
    def area(self) -> int:
        return self.w * self.h


class Change(Record):
    kind: ChangeKind
    box: Box
    source: Box | None = None


class Observation(Record):
    active: bool
    started: bool
    settled: bool
    foreground: list[Box]


class Episode(Record):
    event_id: int
    started_at: datetime


class StoredFrame(Record):
    id: int
    captured_at: datetime
    path: str | None
    is_key: bool
    video_path: str | None = None
    video_index: int | None = None
    event_id: int | None = None


class IngestConfig(Record):
    env_file: Path
    database_url: str
    data_dir: Path
    fps: float = Field(gt=0)
    snapshot_timeout_s: float = Field(gt=0)
    still_keep_s: float = Field(gt=0)
    settle_s: float = Field(gt=0)
    max_event_s: float = Field(gt=0)
    activity_fraction: float = Field(gt=0, lt=1)
    volatility_window_s: float = Field(gt=0)
    volatility_warmup_s: float = Field(ge=0)
    min_change_px: int = Field(gt=0)
    change_width: int = Field(gt=0)
    person_score: float = Field(gt=0, le=1)
    item_link_similarity: float = Field(gt=0, le=1)
    label_max_event_changes: int = Field(ge=1)
    change_check_s: float = Field(gt=0)
    change_persist_s: float = Field(gt=0)
    reference_drift_s: float = Field(gt=0)
    reference_fast_drift_s: float = Field(gt=0)
    key_frames: int = Field(ge=2)
    full_res_days: int = Field(ge=1)
    thin_every_s: int = Field(ge=1)
    thin_height: int = Field(gt=0)
    video_encoder: VideoEncoder
    disk_budget_gb: float = Field(gt=0)
    retention_every_s: float = Field(gt=0)
    openrouter_api_key: SecretStr
    embedding_model: str
    embedding_url: str
    agent_model: str
    caption_model: str
    model_image_width: int = Field(gt=0)
    agent_max_steps: int = Field(ge=1)
    search_limit: int = Field(ge=1)
    query_crop_frames: int = Field(ge=0)
    heartbeat_every_s: float = Field(gt=0)
    log_max_mb: float = Field(gt=0)
    log_backups: int = Field(ge=0)
    web_port: int = Field(gt=0, lt=65536)

    @classmethod
    def from_yaml(cls, path: Path) -> IngestConfig:
        data = yaml.safe_load(path.read_text())
        env = dotenv_values(stream=io.StringIO((path.parent / data["env_file"]).read_text()))
        secrets = {key.lower(): value for key, value in env.items()}

        config = cls.model_validate_json(json.dumps(data | secrets))

        return config.model_copy(
            update={"data_dir": path.parent / config.data_dir, "env_file": path.parent / config.env_file}
        )
