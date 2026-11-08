from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

ENVIRONMENT = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    undefined=StrictUndefined,
    keep_trailing_newline=False,
    autoescape=False,
)


def render(name: str, **values: Any) -> str:
    return ENVIRONMENT.get_template(f"{name}.md.j2").render(**values).strip()
