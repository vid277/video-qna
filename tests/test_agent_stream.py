from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from models import Answer, AnswerDelta, ToolCall, ToolResult
from openrouter import Message, OpenRouter
from recall import agent
from recall.agent import Agent


class ScriptedClient:
    def __init__(self, replies: list[list[str | Message]]):
        self.replies = replies
        self.requests: list[tuple[list[Message], int]] = []

    async def chat_stream(self, model: str, messages: list[Message], tools: list[dict[str, Any]]) -> AsyncIterator[str | Message]:
        self.requests.append((list(messages), len(tools)))
        for piece in self.replies.pop(0):
            yield piece


async def test_agent_streams_answer_text_before_the_final_answer() -> None:
    client = ScriptedClient([["Not ", "observed."] + [{"role": "assistant", "content": "Not observed."}]])

    steps = [s async for s in Agent(cast(OpenRouter, client)).stream("where is my book?")]

    assert steps == [AnswerDelta(text="Not "), AnswerDelta(text="observed."), Answer(text="Not observed.", refs=[])]


async def test_agent_reports_tool_errors_then_forces_an_answer_after_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "config", agent.config.model_copy(update={"agent_max_steps": 2}))
    call = {"id": "c1", "type": "function", "function": {"name": "unknown", "arguments": ""}}
    asks: list[str | Message] = [{"role": "assistant", "content": None, "tool_calls": [call]}]
    client = ScriptedClient([asks, asks, ["Done."] + [{"role": "assistant", "content": "Done."}]])

    steps = [s async for s in Agent(cast(OpenRouter, client)).stream("where is my book?")]

    assert [type(s) for s in steps] == [ToolCall, ToolResult, ToolCall, ToolResult, AnswerDelta, Answer]
    assert steps[0] == ToolCall(id="c1", name="unknown", arguments="{}")
    assert isinstance(steps[1], ToolResult) and steps[1].output.startswith("error:")
    assert steps[-1] == Answer(text="Done.", refs=[])
    assert [tools for _, tools in client.requests] == [len(agent.TOOLS), len(agent.TOOLS), 0]
    final_messages = client.requests[-1][0]
    assert final_messages[-1] == {"role": "user", "content": "Answer now with what you have."}
    assert {"role": "tool", "tool_call_id": "c1", "content": steps[1].output} in final_messages
