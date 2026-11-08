from __future__ import annotations

from pathlib import Path

from models import IngestConfig
from models.jobs import JobsConfig
from store.archive import FrameArchive
from store.store import Store

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "ingest.yaml"
JOBS_CONFIG_PATH = ROOT / "jobs.yaml"

config = IngestConfig.from_yaml(CONFIG_PATH)
jobs_config = JobsConfig.from_yaml(JOBS_CONFIG_PATH)
store = Store(config.database_url, config.embedding_model)
archive = FrameArchive(config.data_dir)
