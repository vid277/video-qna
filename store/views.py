from __future__ import annotations

from models import ChangeKind, StoredChange, StoredFrame
from store import images
from store.runtime import archive, config


def model_image(jpeg: bytes) -> bytes:
    return images.encode(images.fit_width(images.decode(jpeg), config.model_image_width))


def frame_image(frame: StoredFrame) -> bytes:
    return model_image(archive.frame_image(frame))


def frame_crops(frame: StoredFrame) -> list[bytes]:
    crops = images.five_crops(images.decode(archive.frame_image(frame)))

    return [images.encode(images.fit_width(c, config.model_image_width)) for c in crops]


def object_view(change: StoredChange) -> str:
    return change.before_path if change.kind is ChangeKind.REMOVED else change.after_path
