from __future__ import annotations

from voxroom_online.isaac_runtime.perception.detector_base import DryRunDetector


def build_detector(name: str, *args, **kwargs):
    """Build the detector requested by the runtime.

    VoxRoom-Online evaluates room segmentation from geometry. Object detectors
    are intentionally disabled in the public default pipeline.
    """
    detector_name = str(name or "none").strip().lower()
    if detector_name in {"none", "false", "0", "", "dry_run"}:
        return DryRunDetector()
    raise RuntimeError(
        "VoxRoom-Online disables open-vocabulary object detectors. "
        "Run with --detector none for the room-segmentation benchmark."
    )
