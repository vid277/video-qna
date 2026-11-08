from __future__ import annotations

import asyncio
import xml.etree.ElementTree as XML
from collections.abc import AsyncIterator
from datetime import datetime

from reolink_aio.api import Host
from reolink_aio.baichuan.baichuan import Baichuan
from reolink_aio.baichuan.util import PortType

from camera import Camera
from models import (
    CameraEvent,
    CameraInfo,
    CameraPort,
    Direction,
    EventKind,
    QualitySettings,
    Snapshot,
    StreamQuality,
)

AI_KINDS = {"people": EventKind.PERSON, "vehicle": EventKind.VEHICLE, "dog_cat": EventKind.ANIMAL}
GET_ENCODING = 56


class ReolinkCamera(Camera):
    _api: Host
    camera_id: str

    @property
    def _bc(self) -> Baichuan:
        return self._api.baichuan

    async def connect(self) -> None:
        cfg = self.config
        self._api = Host(cfg.host, cfg.username, cfg.password.get_secret_value())

        await self._bc.login()
        self.camera_id = (await self.info()).camera_id

    async def close(self) -> None:
        await self._bc.logout()
        await self._api.logout()

    async def info(self) -> CameraInfo:
        raw = await self._bc.get_info()

        return CameraInfo(
            camera_id=raw["serialNumber"],
            name=raw["name"],
            vendor="Reolink",
            model=raw["type"],
            hardware=raw["hardwareVersion"],
            firmware=raw["firmwareVersion"],
            host=self.config.host,
        )

    async def snapshot(self, quality: StreamQuality) -> Snapshot:
        image = await self._bc.snapshot(self.config.channel, snapType=quality.value)

        return Snapshot(camera_id=self.camera_id, captured_at=datetime.now().astimezone(), quality=quality, image=image)

    def stream_url(self, quality: StreamQuality) -> str:
        cfg = self.config

        return (
            f"rtsp://{cfg.username}:{cfg.password.get_secret_value()}@{cfg.host}:{cfg.rtsp_port}"
            f"/h264Preview_{cfg.channel + 1:02d}_{quality.value}"
        )

    async def get_quality(self) -> QualitySettings:
        root = XML.fromstring(await self._bc.send(cmd_id=GET_ENCODING, channel=self.config.channel))

        return QualitySettings.model_validate(
            {
                q.value: {
                    "frame_rate": int(root.findtext(f".//{q.value}Stream/frame", "0")),
                    "bit_rate": int(root.findtext(f".//{q.value}Stream/bitRate", "0")),
                }
                for q in StreamQuality
            }
        )

    async def set_quality(self, quality: QualitySettings) -> None:
        enc = {
            "mainStream": {"frameRate": quality.main.frame_rate, "bitRate": quality.main.bit_rate},
            "subStream": {"frameRate": quality.sub.frame_rate, "bitRate": quality.sub.bit_rate},
        }

        await self._bc.SetEnc(channel=self.config.channel, Enc=enc)

    async def move(self, direction: Direction, seconds: float) -> None:
        cfg = self.config
        await self._bc.set_ptz_command(cfg.channel, direction.value.capitalize(), speed=cfg.ptz.speed)

        try:
            await asyncio.sleep(seconds)
        finally:
            await self._bc.set_ptz_command(cfg.channel, "Stop")

    async def events(self) -> AsyncIterator[CameraEvent]:
        await self._api.get_host_data()

        pushes: asyncio.Queue[dict[EventKind, bool]] = asyncio.Queue()
        callback_id = f"camera_events_{id(self)}"
        self._bc.register_callback(callback_id, lambda: pushes.put_nowait(self._state()))
        await self._bc.subscribe_events()

        try:
            last = dict.fromkeys(EventKind, False)

            while True:
                state = await pushes.get()
                now = datetime.now()

                for kind, active in state.items() - last.items():
                    yield CameraEvent(camera_id=self.camera_id, kind=kind, active=active, occurred_at=now)

                last |= state
        finally:
            self._bc.unregister_callback(callback_id)
            await self._bc.unsubscribe_events()

    async def ports(self) -> list[CameraPort]:
        raw = await self._bc.get_ports()

        return [
            CameraPort.model_validate(
                {"camera_id": self.camera_id, "protocol": name, "port": p["port"], "enabled": p["enable"]}
            )
            for name, p in raw.items()
        ]

    async def set_rtsp(self, enabled: bool) -> None:
        await self._bc.set_port_enabled(PortType.rtsp, enabled)

    def _state(self) -> dict[EventKind, bool]:
        ch = self.config.channel
        supported = AI_KINDS.keys() & set(self._api.ai_supported_types(ch))

        return {
            EventKind.MOTION: self._api.motion_detected(ch),
            **{AI_KINDS[t]: self._api.ai_detected(ch, t) for t in supported},
        }
