from __future__ import annotations

import cv2
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from models import Box
from store.images import Image

COCO_PERSON = 1
DETECT_WIDTH = 640
CPU_THREADS = 2


class PersonDetector:
    def __init__(self, min_score: float):
        torch.set_num_threads(CPU_THREADS)
        self._min_score = min_score
        self._model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).eval()

    def detect(self, frame: Image) -> list[Box]:
        scale = min(1.0, DETECT_WIDTH / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255

        with torch.inference_mode():
            found = self._model([tensor])[0]

        people = [
            box.tolist()
            for box, label, score in zip(found["boxes"], found["labels"], found["scores"], strict=True)
            if int(label) == COCO_PERSON and float(score) >= self._min_score
        ]

        return [_box(x0 / scale, y0 / scale, x1 / scale, y1 / scale) for x0, y0, x1, y1 in people]


def _box(x0: float, y0: float, x1: float, y1: float) -> Box:
    return Box(x=max(0, int(x0)), y=max(0, int(y0)), w=max(1, int(x1 - x0)), h=max(1, int(y1 - y0)))
