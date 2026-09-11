"""Decode upstream DUDE's display-oriented tagged image into map array rows."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..mask_io import enforce_room_mask_contract


DUDE_OUTPUT_COORDINATE_CONTRACT = "native_flipud_to_source_rows_before_metric_clip_v1"


def tagged_image_to_source_labels(tagged_image: np.ndarray) -> np.ndarray:
    """Undo draw_stable_contour's cv::flip(..., 0), including RGB fallbacks.

    The upstream OccupancyGrid callback consumes row-major input unchanged, but
    draw_stable_contour publishes a vertically flipped image. Undo that display
    transform BEFORE clipping; clipping first irreversibly loses predictions.
    """
    image = np.flipud(np.asarray(tagged_image))
    if image.ndim == 2:
        if not np.all(np.isfinite(image)):
            raise ValueError("DUDE tagged image contains non-finite labels")
        labels = image.astype(np.int32, copy=True)
        labels[labels < 0] = 0
        return labels
    if image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = np.asarray(image[..., :3], dtype=np.uint8)
        flat = rgb.reshape(-1, 3)
        colors, inverse = np.unique(flat, axis=0, return_inverse=True)
        lookup = np.zeros(len(colors), dtype=np.int32)
        next_label = 1
        for index, color in enumerate(colors):
            if tuple(int(v) for v in color) in {(0, 0, 0), (208, 208, 208), (255, 255, 255)}:
                continue
            lookup[index] = next_label
            next_label += 1
        return lookup[inverse].reshape(rgb.shape[:2])
    raise ValueError("unsupported DUDE tagged image shape: %s" % (image.shape,))


def labels_from_tagged_image(tagged_image: np.ndarray, source_arrays: Mapping[str, Any]) -> np.ndarray:
    return enforce_room_mask_contract(
        tagged_image_to_source_labels(tagged_image), source_arrays, clip_to_eval_domain=True
    )
