from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.baselines.topology_active.rasterize import rasterize_topomap
from voxroom_online.isaac_runtime.baselines.topology_active.schema import RoomNode
from voxroom_online.isaac_runtime.baselines.topology_active.topomap import TopologicalRoomMap


def test_active_topology_mask_rasterization():
    topomap = TopologicalRoomMap()
    shape = (8, 12)
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    left[1:7, 1:6] = True
    right[1:7, 6:11] = True
    topomap.nodes[1] = RoomNode(1, "Explored", (3, 2), left, 1, 1)
    topomap.nodes[2] = RoomNode(2, "Explored", (3, 9), right, 1, 1)
    topomap.current_node_id = 1
    domain = left | right
    labels, debug = rasterize_topomap(
        topomap,
        {
            "occupancy_map": np.zeros(shape, dtype=bool),
            "observed_free_mask": domain,
            "obstacle_mask": np.zeros(shape, dtype=bool),
            "unknown_mask": ~domain,
            "navigation_free_room_domain": domain,
        },
    )
    assert set(np.unique(labels).tolist()) == {0, 1, 2}
    assert labels[3, 2] == 1
    assert labels[3, 9] == 2
    assert debug["topology_current_node_id"].shape == ()

