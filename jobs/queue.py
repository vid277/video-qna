from __future__ import annotations

from huey import Huey, SqliteHuey
from huey.consumer import Consumer

from models.jobs import JobsConfig, QueueName

POLL_MAX_S = 1.0


class QuietConsumer(Consumer):
    def _set_signal_handlers(self) -> None:
        pass


def open_queues(config: JobsConfig) -> dict[QueueName, Huey]:
    config.queue_file.parent.mkdir(parents=True, exist_ok=True)

    return {name: SqliteHuey(name.value, filename=str(config.queue_file), results=False, utc=True) for name in QueueName}


def consumer(queue: Huey, workers: int) -> QuietConsumer:
    return QuietConsumer(queue, workers=workers, worker_type="thread", periodic=False, max_delay=POLL_MAX_S)
