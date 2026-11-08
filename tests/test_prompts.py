from __future__ import annotations

import pytest
from jinja2 import UndefinedError

from prompts import render

VALUES = {
    "agent": {"now": "Saturday 2026-11-07 04:10 PM", "max_tool_calls": 5},
    "caption": {"changes": "No objects were detected being added, removed or moved.", "previous": "none"},
    "change_label": {"kind": "added"},
    "summary": {"period": "hour", "start": "16:00", "end": "17:00", "notes": "16:10 drank water"},
}


@pytest.mark.parametrize("name", VALUES)
def test_every_prompt_renders_with_its_values(name: str) -> None:
    text = render(name, **VALUES[name])

    assert text and "{{" not in text and "}}" not in text


@pytest.mark.parametrize("name", VALUES)
def test_a_missing_value_fails_loudly(name: str) -> None:
    with pytest.raises(UndefinedError):
        render(name)


def test_agent_prompt_caps_tool_calls() -> None:
    assert "at most 5 tool calls" in render("agent", **VALUES["agent"])
