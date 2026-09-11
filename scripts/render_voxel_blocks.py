#!/usr/bin/env python3
"""Render exact saved voxel cells as edged cubes using an orthographic camera.

No meshing, downsampling, invented geometry, or 2-D label maps are used. Faces
internal to the opaque occupied/unknown volume and back-facing faces are culled.
Each remaining quad is exactly one face of one original voxel. A height cutoff
is a documented visualization section; the source NPZ is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np


FACE_COLORS = np.array([[217, 217, 217], [255, 255, 255], [228, 151, 159]], dtype=np.uint8)
EDGE_COLORS = np.array([[136, 135, 133], [255, 255, 255], [145, 108, 114]], dtype=np.uint8)


def camera_basis(azimuth: float, elevation: float) -> np.ndarray:
    az, el = np.deg2rad([azimuth, elevation])
    toward_camera = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    return np.stack([right, np.cross(toward_camera, right), toward_camera], axis=1)


def visible_faces(state: np.ndarray, xyz_origin: np.ndarray, size: np.ndarray,
                  basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    solid = state != 1
    vertices, states = [], []
    for axis in range(3):  # Cartesian x, y, z versus storage z, y, x.
        dot = basis[axis, 2]
        if abs(dot) < 1e-8:
            continue
        sign = 1 if dot > 0 else -1
        storage_axis = 2 - axis
        neighbor = np.zeros_like(solid)
        dst, src = [slice(None)] * 3, [slice(None)] * 3
        dst[storage_axis] = slice(None, -1) if sign > 0 else slice(1, None)
        src[storage_axis] = slice(1, None) if sign > 0 else slice(None, -1)
        neighbor[tuple(dst)] = solid[tuple(src)]
        zz, yy, xx = np.nonzero(solid & ~neighbor)
        base = np.column_stack([xx, yy, zz]).astype(np.float32) * size + xyz_origin
        offsets = np.zeros((4, 3), dtype=np.float32)
        offsets[:, axis] = float(sign > 0)
        other = [i for i in range(3) if i != axis]
        offsets[:, other[0]] = [0, 1, 1, 0]
        offsets[:, other[1]] = [0, 0, 1, 1]
        vertices.append(base[:, None, :] + offsets[None, :, :] * size)
        states.append(state[zz, yy, xx])
    points = np.concatenate(vertices)
    face_states = np.concatenate(states)
    # The common lattice makes a back-to-front ordering possible without
    # interpolating a mesh across cells or resampling any voxel dimensions.
    projected = points @ basis
    order = np.argsort(projected[:, :, 2].mean(axis=1), kind="stable")
    return projected[order, :, :2], face_states[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--z-min", type=float)
    parser.add_argument("--detail-center-rc", nargs=2, type=int)
    parser.add_argument("--full-xy-grid", action="store_true",
                        help="Keep the complete original XY grid, including all outer unknown cells")
    parser.add_argument("--detail-width-m", type=float, default=3.0)
    parser.add_argument("--azimuth", type=float, default=-55.0)
    parser.add_argument("--elevation", type=float, default=62.0)
    parser.add_argument("--width-px", type=int, default=4200)
    parser.add_argument("--edge-width", type=float, default=0.12)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    with np.load(args.snapshot, allow_pickle=False) as data:
        raw = data["voxel_occupancy_state_zyx"]
        z = data["voxel_occupancy_z_centers_m"].astype(float)
        resolution = float(data["map_resolution_m"])
        origin_xy = np.array([float(data["map_min_x_m"]), float(data["map_min_y_m"])])
        coverage = float(data["roomseg_eval_coverage_ratio"])
    if not set(np.unique(raw)).issubset({0, 1, 2}):
        raise ValueError("Snapshot contains unsupported voxel states")
    dz = float(np.median(np.diff(z)))
    if not np.allclose(np.diff(z), resolution, rtol=1e-4, atol=1e-6):
        raise ValueError("Expected cubic voxels at the source map resolution")
    coords = np.argwhere(np.any(raw != 0, axis=0))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    if args.full_xy_grid:
        y0, x0 = 0, 0
        y1, x1 = raw.shape[1:]
    if args.full_xy_grid and args.detail_center_rc:
        raise ValueError("Full-grid and local-detail views are mutually exclusive")
    if args.detail_center_rc:
        cy, cx = args.detail_center_rc
        half = int(round(args.detail_width_m / resolution / 2))
        y0, y1 = max(0, cy - half), min(raw.shape[1], cy + half)
        x0, x1 = max(0, cx - half), min(raw.shape[2], cx + half)
    indices = np.flatnonzero(
        ((z >= args.z_min) if args.z_min is not None else np.ones(z.size, bool))
        & ((z <= args.z_max) if args.z_max is not None else np.ones(z.size, bool))
    )
    if not indices.size:
        raise ValueError("Height section does not include any voxel centers")
    z0, z1 = int(indices[0]), int(indices[-1]) + 1
    state = raw[z0:z1, y0:y1, x0:x1]
    xyz_origin = np.r_[origin_xy + np.array([x0, y0]) * resolution, z[z0] - dz / 2]
    voxel_size = np.array([resolution, resolution, dz])
    basis = camera_basis(args.azimuth, args.elevation)
    polygons, face_states = visible_faces(state, xyz_origin, voxel_size, basis)
    print(f"{args.output_prefix.name}: grid {state.shape}, {len(polygons):,} individual voxel faces", flush=True)

    flat = polygons.reshape(-1, 2)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    padding = max(hi - lo) * 0.025
    lo -= padding
    hi += padding
    height_px = int(round(args.width_px * (hi[1] - lo[1]) / (hi[0] - lo[0])))
    fig = plt.figure(figsize=(args.width_px / args.dpi, height_px / args.dpi), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.axis("off")
    faces = PolyCollection(polygons, facecolors=FACE_COLORS[face_states] / 255,
                           edgecolors=EDGE_COLORS[face_states] / 255,
                           linewidths=args.edge_width, antialiaseds=True,
                           closed=True, rasterized=True)
    ax.add_collection(faces)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output_prefix.with_suffix(suffix), dpi=args.dpi, facecolor="white")
    plt.close(fig)
    # Preserve provenance and explicit view limitations beside each figure.
    metadata = {
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "actual_coverage": coverage,
        "source_array": "voxel_occupancy_state_zyx",
        "source_shape_zyx": list(raw.shape),
        "source_state_counts": {str(int(i)): int(n) for i, n in zip(*np.unique(raw, return_counts=True))},
        "voxel_size_xyz_m": voxel_size.tolist(),
        "crop_zyx_exclusive": [[z0, z1], [int(y0), int(y1)], [int(x0), int(x1)]],
        "crop_policy": ("complete original XY grid" if args.full_xy_grid else
                        "explicit local detail window" if args.detail_center_rc else
                        "xy bounding box of all known 3-D cells"),
        "z_cut_m": {"min": args.z_min, "max": args.z_max},
        "actual_z_edges_m": [float(z[z0] - dz / 2), float(z[z1 - 1] + dz / 2)],
        "rendered_volume_counts": {str(int(i)): int(n) for i, n in zip(*np.unique(state, return_counts=True))},
        "visible_facing_surface_quads": len(polygons),
        "colors": {"occupied_face": "#E4979F", "occupied_edge": "#916C72", "unknown_face": "#D9D9D9", "unknown_edge": "#888785"},
        "free_display": "empty/transparent",
        "background": "#FFFFFF",
        "camera_degrees": {"azimuth": args.azimuth, "elevation": args.elevation},
        "original_state_values_unchanged": True,
        "voxel_downsampling": False,
        "unknown_is_solid": True,
        "cube_gap_m": 0,
        "png_size_px": [args.width_px, height_px],
        "pdf_surface_layer": "rasterized at specified DPI to keep file size practical",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved PNG / PDF / JSON: {args.output_prefix}", flush=True)


if __name__ == "__main__":
    main()
