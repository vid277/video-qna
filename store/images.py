from __future__ import annotations

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

type Image = MatLike
type Mask = NDArray[np.bool_]

PREVIEW_SCALE = 4
JPEG_QUALITY = 90
CROP_RATIO = 0.6


def decode(jpeg: bytes) -> Image:
    return _imdecode(jpeg, cv2.IMREAD_COLOR)


def decode_preview(jpeg: bytes) -> Image:
    return _imdecode(jpeg, cv2.IMREAD_REDUCED_GRAYSCALE_4)


def encode(image: Image) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")

    return buf.tobytes()


def gray(image: Image) -> Image:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def fit_width(image: Image, width: int) -> Image:
    h, w = image.shape[:2]
    scale = min(1.0, width / w)

    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def five_crops(image: Image) -> list[Image]:
    h, w = image.shape[:2]
    ch, cw = round(h * CROP_RATIO), round(w * CROP_RATIO)
    origins = [((h - ch) // 2, (w - cw) // 2), (0, 0), (0, w - cw), (h - ch, 0), (h - ch, w - cw)]

    return [image[y : y + ch, x : x + cw] for y, x in origins]


def contact_sheet(frames: list[Image], width: int, cols: int = 2) -> Image:
    cols = min(cols, len(frames))
    cell = width // cols
    tiles = [fit_width(f, cell) for f in frames]
    cell_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, cell_h - t.shape[0], 0, cell - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)

    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def _imdecode(jpeg: bytes, flags: int) -> Image:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), flags)
    if image is None:
        raise ValueError("not a decodable JPEG")

    return image
