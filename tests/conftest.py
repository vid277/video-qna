from __future__ import annotations

import getpass
import shutil
import socket
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from background import retention
from background.room import RoomLog
from frontend import server
from models import IngestConfig
from monitoring import health
from store.archive import FrameArchive
from store.db import Base, sqlalchemy_url
from store.store import Store

TEST_CONFIG = Path(__file__).with_name("ingest.yaml")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))

        return s.getsockname()[1]


@pytest.fixture(scope="session")
def database_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root, port = tmp_path_factory.mktemp("postgres"), free_port()
    url = make_url(IngestConfig.from_yaml(TEST_CONFIG).database_url).set(host="localhost", port=port)

    subprocess.run(["initdb", "-D", root / "data", "-U", getpass.getuser(), "--auth=trust"], capture_output=True, check=True)
    subprocess.run(
        ["pg_ctl", "-D", root / "data", "-o", f"-p {port} -k ''", "-l", root / "log", "-w", "start"], capture_output=True, check=True
    )
    subprocess.run(["createdb", "-h", "localhost", "-p", str(port), str(url.database)], check=True)

    yield url.render_as_string(hide_password=False)

    subprocess.run(["pg_ctl", "-D", root / "data", "-m", "immediate", "stop"], capture_output=True, check=True)
    shutil.rmtree(root)


@pytest.fixture
def config(tmp_path: Path, database_url: str) -> IngestConfig:
    return IngestConfig.from_yaml(TEST_CONFIG).model_copy(update={"data_dir": tmp_path, "database_url": database_url})


@pytest.fixture
async def db(database_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(sqlalchemy_url(database_url))

    async with engine.begin() as conn:
        await conn.execute(text("create extension if not exists vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield async_sessionmaker(engine, expire_on_commit=False)

    await engine.dispose()


@pytest.fixture
async def store(config: IngestConfig, db: async_sessionmaker[AsyncSession]) -> AsyncIterator[Store]:
    store = Store(config.database_url, config.embedding_model)

    yield store

    await store.close()


@pytest.fixture
def archive(config: IngestConfig) -> FrameArchive:
    return FrameArchive(config.data_dir)


@pytest.fixture
async def room_log(config: IngestConfig, db: async_sessionmaker[AsyncSession]) -> AsyncIterator[RoomLog]:
    log = RoomLog(config.database_url, config.embedding_model)

    yield log

    await log.close()


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch, config: IngestConfig, store: Store, archive: FrameArchive, room_log: RoomLog) -> None:
    values = {"config": config, "store": store, "archive": archive, "room": room_log}

    for module in (retention, health, server):
        for name in values.keys() & vars(module).keys():
            monkeypatch.setattr(module, name, values[name])
