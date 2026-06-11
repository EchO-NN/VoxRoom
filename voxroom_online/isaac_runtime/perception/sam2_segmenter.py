from __future__ import annotations


def build_sam2_segmenter(mode: str = "none", *args, **kwargs):
    """Return no segmenter for the VoxRoom geometry-only pipeline."""
    segmenter_name = str(mode or "none").strip().lower()
    if segmenter_name in {"none", "false", "0", ""}:
        return None
    raise RuntimeError(
        "VoxRoom-Online disables SAM-style object segmentation. "
        "Run with --segmenter none for the room-segmentation benchmark."
    )
