from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.baselines.offline.fallback_voronoi import build_voronoi_ipa_input_image
from voxroom_online.isaac_runtime.baselines.ros_grid_io import snapshot_to_ipa_image


def test_ipa_input_image_uses_mono8_free_255_inaccessible_0() -> None:
    free = np.zeros((4, 5), dtype=bool)
    free[1:3, 1:4] = True
    obstacle = np.zeros_like(free)
    obstacle[2, 2] = True
    unknown = np.zeros_like(free)
    unknown[1, 1] = True
    arrays = {
        "occupancy_map": obstacle,
        "navigation_free_room_domain": free,
        "observed_free_mask": free,
        "obstacle_mask": obstacle,
        "unknown_mask": unknown,
    }
    image = snapshot_to_ipa_image(arrays)
    assert image.dtype == np.uint8
    assert image.shape == free.shape
    assert image[1, 2] == 255
    assert image[0, 0] == 0
    assert image[1, 1] == 0
    assert image[2, 2] == 0


def test_voronoi_input_skips_empty_navigation_free_source() -> None:
    observed = np.zeros((4, 5), dtype=bool)
    observed[1:3, 1:4] = True
    arrays = {
        "occupancy_map": np.zeros_like(observed),
        "navigation_free_room_domain": np.zeros_like(observed),
        "observed_free_mask": observed,
        "obstacle_mask": np.zeros_like(observed),
        "unknown_mask": np.zeros_like(observed),
    }
    image, free, metadata = build_voronoi_ipa_input_image(arrays)
    assert metadata["free_source"] == "observed_free_mask"
    assert int(np.count_nonzero(free)) == int(np.count_nonzero(observed))
    assert int(np.count_nonzero(image == 255)) == int(np.count_nonzero(observed))
