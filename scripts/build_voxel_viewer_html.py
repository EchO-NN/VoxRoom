#!/usr/bin/env python3
"""Embed an unchanged three-state voxel snapshot in a standalone WebGL viewer."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.snapshot, allow_pickle=False) as arrays:
        state = arrays["voxel_occupancy_state_zyx"]
        z = arrays["voxel_occupancy_z_centers_m"].astype(float)
        resolution = float(arrays["map_resolution_m"])
        origin = [float(arrays["map_min_x_m"]), float(arrays["map_min_y_m"]),
                  float(z[0] - resolution / 2)]
        coverage = float(arrays["roomseg_eval_coverage_ratio"])
    if state.ndim != 3 or not set(np.unique(state)).issubset({0, 1, 2}):
        raise ValueError("Expected a ZYX volume with states 0=unknown, 1=free, 2=occupied")
    if not np.allclose(np.diff(z), resolution, atol=1e-6):
        raise ValueError("Expected cubic source voxels")
    flat = state.ravel(order="C")
    starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1]
    lengths = np.diff(np.r_[starts, flat.size])
    packed = ((lengths.astype(np.uint32) << 2) | flat[starts]).astype("<u4")
    # Verify that the portable run-length stream round-trips every voxel.
    reconstructed = np.repeat((packed & 3).astype(np.uint8), packed >> 2)
    if not np.array_equal(reconstructed, flat):
        raise ValueError("Voxel stream failed its lossless round-trip check")
    known = np.argwhere(np.any(state != 0, axis=0))
    lo, hi = known.min(axis=0), known.max(axis=0) + 1
    nz, ny, nx = state.shape
    metadata = {
        "title": "0003 · 70% 三维体素",
        "dimensions": [nx, ny, nz],
        "origin": origin,
        "voxelSize": resolution,
        "coverage": coverage,
        "knownBounds": [int(lo[1]), int(hi[1]), int(lo[0]), int(hi[0]), 0, nz],
        "sourceFile": str(args.snapshot.resolve()),
        "sourceSha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "stateCounts": {str(int(i)): int(n) for i, n in zip(*np.unique(state, return_counts=True))},
        "encoding": "little-endian uint32 runs: count << 2 | voxel state; C-order ZYX",
        "originalVoxels": int(state.size),
        "faceColors": {"occupied": "#E4979F", "unknown": "#D9D9D9"},
        "edgeColors": {"occupied": "#916C72", "unknown": "#888785"},
    }
    template = Path(__file__).with_name("assets").joinpath("voxel_viewer.html").read_text()
    html = template.replace("__VOXEL_METADATA__", json.dumps(metadata, ensure_ascii=False))
    html = html.replace("__VOXEL_RLE_BASE64__", base64.b64encode(packed.tobytes()).decode("ascii"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"html": str(args.output.resolve()), "bytes": args.output.stat().st_size,
                      "voxels": int(flat.size), "lossless_check": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
