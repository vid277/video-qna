from __future__ import annotations

import logging
from zoneinfo import ZoneInfo

from ingest.runtime import config, jobs_config, store
from ingest.store import PERIOD_STEP
from models.jobs import Period, SummaryInput, SummaryWork
from openrouter import OpenRouter, text_part
from prompts import render

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

TIME_FORMAT = {Period.HOUR: "%H:%M", Period.DAY: "%a %d %b %H:%M"}

log = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, client: OpenRouter):
        self._client = client
        self._zone = ZoneInfo(jobs_config.timezone)

    async def summarize(self, key: str) -> None:
        work = SummaryWork.from_key(key)
        start = work.start.astimezone(self._zone)
        end = start + PERIOD_STEP[work.period]
        inputs = await store.summary_inputs(work, end)

        prompt = render("summary", period=work.period.value, start=start.isoformat(), end=end.isoformat(), notes=self._notes(work, inputs))
        data = await self._client.complete_json(config.caption_model, [text_part(prompt)], "period_summary", SCHEMA)

        await store.set_summary(work, end, data["summary"], sum(i.events for i in inputs))
        log.info("summarized %s %s (%d notes)", work.period.value, start.isoformat(), len(inputs))

    def _notes(self, work: SummaryWork, inputs: list[SummaryInput]) -> str:
        return "\n".join(f"- {i.at.astimezone(self._zone).strftime(TIME_FORMAT[work.period])}: {i.text}" for i in inputs)
