from __future__ import annotations

import numpy as np

from voxroom_online.isaac_runtime.env.isaac_process import IsaacSimServer


class _DepthOnlyCamera:
    def get_current_frame(self) -> dict:
        return {}

    def get_depth(self, device: str = "cpu") -> np.ndarray:
        _ = device
        return np.ones((4, 4), dtype=np.float32)


def test_camera_get_depth_fallback_is_explicit_image_plane_z() -> None:
    server = IsaacSimServer(width=4, height=4, enable_depth=True, verbose=False)
    server.camera = _DepthOnlyCamera()

    obs = server.get_observation(read_rgb=False, read_depth=True)

    assert obs["has_depth"] is True
    assert isinstance(obs["depth"], np.ndarray)
    assert obs["depth_source"] == "camera_get_depth"
    assert obs["depth_semantics"] == "image_plane_z"
