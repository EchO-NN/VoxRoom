from voxroom_online.isaac_runtime.scripts import replay_voxel_roomseg_snapshots as replay


def test_compact_replay_npz_preserves_voronoi_source_inputs() -> None:
    required = {
        "occupancy_map",
        "observed_free_mask",
        "obstacle_mask",
        "unknown_mask",
        "voxel_nav_free_xy",
        "voxel_nav_occupied_xy",
        "voxel_nav_unknown_xy",
        "voxel_vertical_free_xy",
        "voxel_wall_after_step1_map",
        "voxel_door_cut_mask",
        "voxel_current_door_cut_mask",
        "voxel_door_partition_cut_mask",
        "voxel_door_topology_effective_cut_mask",
    }
    assert required.issubset(set(replay.REPLAY_NPZ_KEYS))
