#!/usr/bin/env python3
"""Command-line tool for the cameras defined in cameras.yaml."""

import argparse
import asyncio
import contextlib
import logging
import signal
import subprocess
from collections.abc import AsyncGenerator
from pathlib import Path

from ingest import runtime, tasks
from ingest.health import Heartbeat, describe, report
from ingest.pipeline import IngestPipeline
from jobs.queue import open_queues
from jobs.scheduler import Scheduler
from models import CamerasFile, Direction, QualitySettings, Snapshot, StreamQuality
from openrouter import OpenRouter
from recall.agent import Agent
from reolink import ReolinkCamera
from frontend.server import serve

DEFAULT_CONFIG = Path(__file__).with_name("cameras.yaml")

log = logging.getLogger("ghostcam")


def make_camera(config_path: Path, name: str | None) -> ReolinkCamera:
    cameras = {c.name: c for c in CamerasFile.from_yaml(config_path).cameras}

    return ReolinkCamera(cameras[name or next(iter(cameras))])


def print_quality(quality: QualitySettings) -> None:
    for stream, s in quality.model_dump().items():
        print(f"  {stream:5} {s['frame_rate']:>3} fps  {s['bit_rate']:>5} kbps")


async def cmd_info(cam: ReolinkCamera, _args: argparse.Namespace) -> None:
    for key, value in (await cam.info()).model_dump().items():
        print(f"{key + ':':11} {value}")

    print("quality:")
    print_quality(await cam.get_quality())

    print("ports:")
    for p in await cam.ports():
        print(f"  {p.protocol:6} {p.port:>5}  {'ON' if p.enabled else 'off'}")


async def cmd_snapshot(cam: ReolinkCamera, args: argparse.Namespace) -> None:
    snap = await cam.snapshot(args.quality or cam.config.snapshot_quality)
    out = args.out or f"{cam.config.name}_{snap.captured_at:%Y%m%d_%H%M%S}.jpg"

    with open(out, "wb") as f:
        f.write(snap.image)

    print(f"saved {out} ({len(snap.image)} bytes, {snap.quality} stream)")


async def cmd_ptz(cam: ReolinkCamera, args: argparse.Namespace) -> None:
    seconds = args.seconds or cam.config.ptz.seconds
    await cam.move(args.direction, seconds)

    print(f"moved {args.direction} for {seconds}s")


async def cmd_watch(cam: ReolinkCamera, _args: argparse.Namespace) -> None:
    print("watching for events (Ctrl+C to stop)...", flush=True)

    async for ev in cam.events():
        print(f"{ev.occurred_at:%H:%M:%S}  {ev.kind:8} {'start' if ev.active else 'end'}", flush=True)


async def cmd_set_quality(cam: ReolinkCamera, _args: argparse.Namespace) -> None:
    await cam.set_quality(cam.config.quality)

    print("quality now:")
    print_quality(await cam.get_quality())


async def cmd_enable_rtsp(cam: ReolinkCamera, _args: argparse.Namespace) -> None:
    await cam.set_rtsp(True)

    rtsp = next(p for p in await cam.ports() if p.protocol == "rtsp")
    print(f"rtsp enabled: {rtsp.enabled} (port {rtsp.port})")


async def cmd_stream(cam: ReolinkCamera, args: argparse.Namespace) -> None:
    url = cam.stream_url(args.quality or cam.config.stream_quality)
    print(f"opening {url.split('@')[-1]}")

    await asyncio.to_thread(subprocess.run, ["ffplay", "-rtsp_transport", "tcp", "-fflags", "nobuffer", url])


async def cmd_ingest(cam: ReolinkCamera, _args: argparse.Namespace) -> None:
    heartbeat = Heartbeat("ingest")
    heartbeat.configure_logging()
    pipeline = IngestPipeline(cam.camera_id)
    await pipeline.recover()
    beating = asyncio.create_task(heartbeat.run())
    print(f"ingesting into {runtime.config.data_dir} (Ctrl+C to stop)...", flush=True)

    try:
        async with contextlib.aclosing(snapshots(cam, runtime.config.fps, runtime.config.snapshot_timeout_s)) as snaps:
            async for snap in snaps:
                heartbeat.saw_snapshot(snap.captured_at)
                await ingest_one(pipeline, snap)
    finally:
        beating.cancel()
        await asyncio.gather(beating, return_exceptions=True)
        await runtime.store.close()
        print("stopped", flush=True)


async def cmd_worker(_args: argparse.Namespace) -> None:
    heartbeat = Heartbeat("worker")
    heartbeat.configure_logging()
    print(f"worker using {runtime.jobs_config.queue_file} (Ctrl+C to stop)...", flush=True)

    async with OpenRouter(runtime.config.openrouter_api_key.get_secret_value(), runtime.config.embedding_url) as client:
        queues = open_queues(runtime.jobs_config)
        scheduler = Scheduler(runtime.jobs_config, queues, tasks.index_kinds(client), tasks.scheduled(), runtime.store, tasks.is_outage)
        beating = asyncio.create_task(heartbeat.run())

        try:
            await scheduler.run()
        finally:
            beating.cancel()
            await asyncio.gather(beating, return_exceptions=True)
            await runtime.store.close()
            print("stopped", flush=True)


async def cmd_jobs(args: argparse.Namespace) -> None:
    try:
        cleared = await runtime.store.clear_failures() if args.retry else 0
        backlog = await runtime.store.index_backlog()
        failures = await runtime.store.failures()
    finally:
        await runtime.store.close()

    print(f"cleared {cleared} failures" if args.retry else "pending:")
    for b in backlog:
        print(f"  {b.kind.value:15} {b.pending:>6}  oldest {b.oldest or '-'}")

    print(f"failures: {len(failures)}")
    for f in failures:
        print(f"  {f.kind:15} {f.key:>12}  attempts {f.attempts}  retry {f.retry_at:%Y-%m-%d %H:%M}  {f.last_error[:80]}")


async def cmd_ask(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="  %(message)s")

    async with OpenRouter(runtime.config.openrouter_api_key.get_secret_value(), runtime.config.embedding_url) as client:
        try:
            print((await Agent(client).ask(args.question)).text)
        finally:
            await runtime.store.close()


async def cmd_status(_args: argparse.Namespace) -> None:
    try:
        print("\n".join(describe(await report())))
    finally:
        await runtime.store.close()


async def cmd_serve(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        await serve(args.host)
    finally:
        await runtime.store.close()


async def ingest_one(pipeline: IngestPipeline, snap: Snapshot) -> None:
    try:
        await pipeline.ingest(snap)
    except Exception:
        log.exception("dropped snapshot from %s", snap.captured_at)


async def snapshots(cam: ReolinkCamera, fps: float, timeout_s: float) -> AsyncGenerator[Snapshot]:
    loop = asyncio.get_running_loop()
    due = loop.time()

    while True:
        try:
            async with asyncio.timeout(timeout_s):
                snap = await cam.snapshot(cam.config.snapshot_quality)
        except Exception as e:
            log.warning("snapshot failed: %r", e)
        else:
            yield snap

        due = max(due + 1 / fps, loop.time())
        await asyncio.sleep(due - loop.time())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="cameras YAML file")
    parser.add_argument("--camera", help="camera name from the config (default: first one)")

    sub = parser.add_subparsers(required=True)

    sub.add_parser("info", help="camera details, quality and port states").set_defaults(func=cmd_info)

    p = sub.add_parser("snapshot", help="save a JPEG")
    p.add_argument("out", nargs="?")
    p.add_argument("--quality", type=StreamQuality, choices=list(StreamQuality))
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("ptz", help="pan/tilt the camera")
    p.add_argument("direction", type=Direction, choices=list(Direction))
    p.add_argument("seconds", type=float, nargs="?")
    p.set_defaults(func=cmd_ptz)

    sub.add_parser("watch", help="print motion / detection events").set_defaults(func=cmd_watch)

    sub.add_parser("set-quality", help="apply quality from the config").set_defaults(func=cmd_set_quality)

    sub.add_parser("enable-rtsp", help="turn on RTSP").set_defaults(func=cmd_enable_rtsp)

    p = sub.add_parser("stream", help="open the RTSP stream in ffplay")
    p.add_argument("--quality", type=StreamQuality, choices=list(StreamQuality))
    p.set_defaults(func=cmd_stream)

    sub.add_parser("ingest", help="record frames, events and scene changes").set_defaults(func=cmd_ingest)

    p = sub.add_parser("worker", help="embed, caption, summarize and clean up in the background")
    p.set_defaults(func=cmd_worker, needs_camera=False)

    p = sub.add_parser("jobs", help="show background work still pending and jobs that keep failing")
    p.add_argument("--retry", action="store_true", help="clear recorded failures so they are retried")
    p.set_defaults(func=cmd_jobs, needs_camera=False)

    p = sub.add_parser("ask", help="answer a question about what the camera saw")
    p.add_argument("question")
    p.add_argument("--verbose", action="store_true", help="print each tool call")
    p.set_defaults(func=cmd_ask, needs_camera=False)

    sub.add_parser("status", help="ingest health: heartbeats, last frame, backlog, disk, last problem").set_defaults(
        func=cmd_status, needs_camera=False
    )

    p = sub.add_parser("serve", help="local web page to ask questions and watch the room")
    p.add_argument("--host", default="127.0.0.1", help="interface to listen on (port comes from ingest.yaml)")
    p.set_defaults(func=cmd_serve, needs_camera=False)

    parser.set_defaults(needs_camera=True)

    return parser


async def with_camera(args: argparse.Namespace) -> None:
    async with make_camera(args.config, args.camera) as cam:
        await args.func(cam, args)


async def without_camera(args: argparse.Namespace) -> None:
    await args.func(args)


async def main(args: argparse.Namespace) -> None:
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    assert task is not None

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)

    await {True: with_camera, False: without_camera}[args.needs_camera](args)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        asyncio.run(main(build_parser().parse_args()))
