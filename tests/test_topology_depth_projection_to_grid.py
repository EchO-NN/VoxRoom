from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.baselines.data_contract import MapInfo
from voxroom_online.isaac_runtime.baselines.topology_active.adapter import ActiveRoomSegmentationBaseline
from voxroom_online.isaac_runtime.baselines.topology_active.depth_projection import project_door_bbox_to_grid_rc
from voxroom_online.isaac_runtime.baselines.topology_active.detector import (
    CameraIntrinsics,
    DoorDetection2D,
    DoorDetector,
    project_door_bbox_to_grid,
)


class _FakeOriginalDetrDetector(DoorDetector):
    name = "original_detr"
    available = True
    detector_adapter_verified = True
    active_repo_commit = "fake_commit"
    checkpoint_sha256 = "fake_sha"
    checkpoint_path = "/tmp/fake_model.pth"
    repo_dir = "/tmp/fake_repo"

    def detect(self, rgb: np.ndarray) -> list[DoorDetection2D]:
        _ = rgb
        return [DoorDetection2D(bbox_xyxy=(45.0, 40.0, 55.0, 60.0), score=0.9)]


def test_topology_depth_projection_to_grid_identity_camera() -> None:
    detection = DoorDetection2D(bbox_xyxy=(45.0, 40.0, 55.0, 60.0), score=0.9)
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    map_info = MapInfo(resolution_m=0.1, min_x=-1.0, max_x=5.0, min_y=-1.0, max_y=5.0, width=60, height=60)
    result = project_door_bbox_to_grid_rc(
        detection=detection,
        depth=depth,
        camera_intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100),
        camera_pose_world=np.eye(4, dtype=np.float32),
        map_info=map_info,
        sample_radius_px=0,
    )
    assert result is not None
    # Bottom-near sample is u=50, v=58. With z=2m, y_world=(58-50)*2/100=0.16.
    # Runtime grid rows decrease as world y increases.
    assert result.rc == (48, 10)
    assert np.allclose(result.world_xyz, (0.0, 0.16, 2.0), atol=1e-6)


def test_topology_depth_projection_accepts_runtime_xyzyaw_pose_forward() -> None:
    detection = DoorDetection2D(bbox_xyxy=(45.0, 40.0, 55.0, 60.0), score=0.9)
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    map_info = MapInfo(resolution_m=0.1, min_x=-1.0, max_x=5.0, min_y=-1.0, max_y=5.0, width=60, height=60)
    result = project_door_bbox_to_grid_rc(
        detection=detection,
        depth=depth,
        camera_intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100),
        camera_pose_world=np.asarray([0.0, 0.0, 1.2, 0.0], dtype=np.float32),
        map_info=map_info,
        sample_radius_px=0,
    )
    assert result is not None
    assert result.rc == (50, 30)
    assert np.allclose(result.world_xyz, (2.0, 0.0, 1.04), atol=1e-6)


def test_topology_depth_projection_accepts_runtime_xyzyaw_pose_yaw90() -> None:
    detection = DoorDetection2D(bbox_xyxy=(45.0, 40.0, 55.0, 60.0), score=0.9)
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    map_info = MapInfo(resolution_m=0.1, min_x=-1.0, max_x=5.0, min_y=-1.0, max_y=5.0, width=60, height=60)
    result = project_door_bbox_to_grid_rc(
        detection=detection,
        depth=depth,
        camera_intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100),
        camera_pose_world=np.asarray([0.0, 0.0, 1.2, np.pi * 0.5], dtype=np.float32),
        map_info=map_info,
        sample_radius_px=0,
    )
    assert result is not None
    assert result.rc == (30, 9)
    assert np.allclose(result.world_xyz, (0.0, 2.0, 1.04), atol=1e-6)


def test_topology_depth_projection_falls_back_to_bbox_depth_when_sample_is_empty() -> None:
    detection = DoorDetection2D(bbox_xyxy=(45.0, 40.0, 55.0, 60.0), score=0.9)
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[50:61, 45:56] = 2.0
    depth[58, 50] = 0.0
    map_info = MapInfo(resolution_m=0.1, min_x=-1.0, max_x=5.0, min_y=-1.0, max_y=5.0, width=60, height=60)
    result = project_door_bbox_to_grid_rc(
        detection=detection,
        depth=depth,
        camera_intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100),
        camera_pose_world=np.eye(4, dtype=np.float32),
        map_info=map_info,
        sample_radius_px=0,
    )
    assert result is not None
    assert result.rc == (48, 10)


def test_detector_projection_wrapper_is_not_placeholder() -> None:
    candidate = project_door_bbox_to_grid(
        DoorDetection2D(bbox_xyxy=(45.0, 40.0, 55.0, 60.0), score=0.9),
        depth=np.full((100, 100), 2.0, dtype=np.float32),
        camera_intrinsics=np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        camera_pose_world=np.eye(4, dtype=np.float32),
        map_info=MapInfo(resolution_m=0.1, min_x=-1.0, max_x=5.0, min_y=-1.0, max_y=5.0, width=60, height=60),
    )
    assert candidate is not None
    assert candidate.rc == (48, 10)
    assert candidate.metadata["projection_status"] == "pinhole_depth_projected"


def test_topology_adapter_uses_projected_original_detr_detections(tmp_path) -> None:
    shape = (60, 60)
    baseline = ActiveRoomSegmentationBaseline(
        output_dir=tmp_path / "baselines" / "topology_visual_active",
        detector=_FakeOriginalDetrDetector(),
        map_info=MapInfo(resolution_m=0.1, min_x=-1.0, max_x=5.0, min_y=-1.0, max_y=5.0, width=60, height=60),
        save_stream=False,
    )
    free = np.ones(shape, dtype=bool)
    baseline.update(
        step=1,
        obs={
            "has_rgb": True,
            "has_depth": True,
            "rgb": np.zeros((100, 100, 3), dtype=np.uint8),
            "depth": np.full((100, 100), 2.0, dtype=np.float32),
            "camera_pose_world": np.eye(4, dtype=np.float32),
        },
        map_state={
            "occupancy": ~free,
            "free": free,
            "observed": np.ones(shape, dtype=bool),
            "current_grid": (12, 10),
        },
        frontier_map=np.zeros(shape, dtype=bool),
        camera_intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100),
    )
    metadata = baseline.latest_metadata
    assert metadata["vision_num_detections"] == 1
    assert metadata["vision_num_candidates"] == 1
    assert metadata["vision_projection_attempted"] is True
    assert metadata["projection_verified"] is True
    assert metadata["main_experiment_allowed"] is True


def test_topology_adapter_does_not_verify_projection_without_candidate(tmp_path) -> None:
    shape = (60, 60)
    baseline = ActiveRoomSegmentationBaseline(
        output_dir=tmp_path / "baselines" / "topology_visual_active",
        detector=_FakeOriginalDetrDetector(),
        map_info=MapInfo(resolution_m=0.1, min_x=-1.0, max_x=5.0, min_y=-1.0, max_y=5.0, width=60, height=60),
        save_stream=False,
    )
    free = np.ones(shape, dtype=bool)
    baseline.update(
        step=1,
        obs={
            "has_rgb": True,
            "has_depth": True,
            "rgb": np.zeros((100, 100, 3), dtype=np.uint8),
            "depth": np.zeros((100, 100), dtype=np.float32),
            "camera_pose_world": np.eye(4, dtype=np.float32),
        },
        map_state={
            "occupancy": ~free,
            "free": free,
            "observed": np.ones(shape, dtype=bool),
            "current_grid": (12, 10),
        },
        frontier_map=np.zeros(shape, dtype=bool),
        camera_intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100),
    )
    metadata = baseline.latest_metadata
    assert metadata["vision_num_detections"] == 1
    assert metadata["vision_projection_attempted"] is True
    assert metadata["vision_num_candidates"] == 0
    assert metadata["projection_verified"] is False
    assert metadata["main_experiment_allowed"] is False
