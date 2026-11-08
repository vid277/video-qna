from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from store.db import Base, sqlalchemy_url

ROOT = Path(__file__).resolve().parents[1]


def test_migrations_build_the_schema_the_models_expect(database_url: str) -> None:
    url = make_url(database_url).set(database="migrations_check")
    subprocess.run(["createdb", "-h", str(url.host), "-p", str(url.port), str(url.database)], check=True)
    target = url.render_as_string(hide_password=False)

    subprocess.run([sys.executable, "-m", "alembic", "-x", f"url={target}", "upgrade", "head"], cwd=ROOT, check=True, capture_output=True)

    engine = create_engine(sqlalchemy_url(target))
    try:
        with engine.connect() as conn:
            drift = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    finally:
        engine.dispose()

    assert drift == []
