#!/usr/bin/env python3
"""Render the saved three-state Vertical-Free map without door overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage


OCCUPIED_RGB = (228, 151, 159)  # #E4979F
FREE_RGB = (255, 255, 255)  # #FFFFFF
UNKNOWN_RGB = (217, 217, 217)  # #D9D9D9
DEFAULT_SEED_RGB = (215, 208, 239)  # #D7D0EF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--padding-cells", type=int, default=22)
    parser.add_argument(
        "--occupied-key",
        default="voxel_wall_xy",
        help="saved 2-D mask to render as occupied",
    )
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument(
        "--seed-key",
        action="append",
        default=[],
        help="saved seed mask to overlay; repeat to render their union",
    )
    parser.add_argument("--seed-color", default="#D7D0EF")
    parser.add_argument("--kept-seed-key")
    parser.add_argument("--kept-seed-color", default="#9B73BE")
    parser.add_argument(
        "--do-not-draw-kept-seeds",
        action="store_true",
        help="load kept seeds for filtering/association but omit their raster layer",
    )
    parser.add_argument("--rejected-seed-key")
    parser.add_argument("--rejected-seed-color", default="#AE0035")
    parser.add_argument(
        "--hide-kept-components-contained-in-room-label",
        action="append",
        default=[],
        type=int,
        help="presentation-only: hide kept-seed components wholly inside this room label",
    )
    parser.add_argument(
        "--fitted-line-key",
        action="append",
        default=[],
        help="saved fitted-line mask to overlay; repeat to render their union",
    )
    parser.add_argument("--fitted-line-color", default="#9B73BE")
    parser.add_argument(
        "--erase-below-row",
        type=int,
        help="Presentation-only cleanup: render cells at and below this row as unknown.",
    )
    parser.add_argument(
        "--hide-fitted-components-touching-hidden-kept",
        action="store_true",
    )
    return parser.parse_args()


def _parse_hex_rgb(value: str) -> tuple[int, int, int]:
    text = str(value).strip().lstrip("#")
    if len(text) != 6:
        raise ValueError("color must use six hexadecimal digits")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    output = args.output.expanduser().resolve()
    pdf_output = args.pdf_output.expanduser().resolve() if args.pdf_output else None

    with np.load(snapshot, allow_pickle=False) as arrays:
        required = (
            "voxel_vertical_free_xy",
            str(args.occupied_key),
            *(str(key) for key in args.seed_key),
            *([str(args.kept_seed_key)] if args.kept_seed_key else []),
            *([str(args.rejected_seed_key)] if args.rejected_seed_key else []),
            *(str(key) for key in args.fitted_line_key),
            *(
                ["voxel_final_room_label_map"]
                if args.hide_kept_components_contained_in_room_label
                else []
            ),
        )
        missing = [key for key in required if key not in arrays.files]
        if missing:
            raise KeyError("snapshot is missing: " + ", ".join(missing))
        free = np.asarray(arrays["voxel_vertical_free_xy"], dtype=bool)
        occupied = np.asarray(arrays[str(args.occupied_key)], dtype=bool)
        saved_unknown = (
            np.asarray(arrays["voxel_unknown_xy"], dtype=bool)
            if "voxel_unknown_xy" in arrays.files
            else np.zeros(free.shape, dtype=bool)
        )
        vertical_observed = (
            np.asarray(arrays["voxel_vertical_observed_xy"], dtype=bool)
            if "voxel_vertical_observed_xy" in arrays.files
            else free | occupied
        )
        coverage = (
            float(np.asarray(arrays["roomseg_eval_coverage_ratio"]).reshape(()))
            if "roomseg_eval_coverage_ratio" in arrays.files
            else None
        )
        seed_masks = [
            np.asarray(arrays[str(key)], dtype=bool) for key in args.seed_key
        ]
        kept_seeds = (
            np.asarray(arrays[str(args.kept_seed_key)], dtype=bool)
            if args.kept_seed_key
            else np.zeros(free.shape, dtype=bool)
        )
        rejected_seeds = (
            np.asarray(arrays[str(args.rejected_seed_key)], dtype=bool)
            if args.rejected_seed_key
            else np.zeros(free.shape, dtype=bool)
        )
        room_labels = (
            np.asarray(arrays["voxel_final_room_label_map"], dtype=np.int32)
            if args.hide_kept_components_contained_in_room_label
            else np.zeros(free.shape, dtype=np.int32)
        )
        fitted_line_masks = [
            np.asarray(arrays[str(key)], dtype=bool)
            for key in args.fitted_line_key
        ]

    if not (free.shape == occupied.shape == saved_unknown.shape == vertical_observed.shape):
        raise ValueError("Vertical-Free arrays do not share one grid shape")
    if any(
        mask.shape != free.shape
        for mask in [*seed_masks, kept_seeds, rejected_seeds, *fitted_line_masks]
    ):
        raise ValueError("seed masks do not match the Vertical-Free grid shape")
    if free.ndim != 2:
        raise ValueError("Vertical-Free arrays must be two-dimensional")
    occupied_free_overlap = free & occupied
    effective_free = free & ~occupied
    seeds = np.logical_or.reduce(seed_masks) if seed_masks else np.zeros(free.shape, dtype=bool)
    seed_rgb = _parse_hex_rgb(args.seed_color) if seed_masks else DEFAULT_SEED_RGB
    kept_seed_rgb = _parse_hex_rgb(args.kept_seed_color)
    rejected_seed_rgb = _parse_hex_rgb(args.rejected_seed_color)
    fitted_line_rgb = _parse_hex_rgb(args.fitted_line_color)
    fitted_lines = (
        np.logical_or.reduce(fitted_line_masks)
        if fitted_line_masks
        else np.zeros(free.shape, dtype=bool)
    )
    hidden_kept_seeds = np.zeros(free.shape, dtype=bool)
    if args.hide_kept_components_contained_in_room_label:
        kept_components, kept_component_count = ndimage.label(
            kept_seeds, structure=np.ones((3, 3), dtype=bool)
        )
        target_room_labels = {
            int(label) for label in args.hide_kept_components_contained_in_room_label
        }
        for component_id in range(1, int(kept_component_count) + 1):
            component = kept_components == component_id
            labels_under_component = set(
                int(value) for value in np.unique(room_labels[component])
            )
            if len(labels_under_component) == 1 and labels_under_component <= target_room_labels:
                hidden_kept_seeds |= component
        kept_seeds = kept_seeds & ~hidden_kept_seeds
    hidden_fitted_lines = np.zeros(free.shape, dtype=bool)
    if args.hide_fitted_components_touching_hidden_kept and np.any(hidden_kept_seeds):
        fitted_components, fitted_component_count = ndimage.label(
            fitted_lines, structure=np.ones((3, 3), dtype=bool)
        )
        touched_component_ids = np.unique(fitted_components[hidden_kept_seeds])
        touched_component_ids = touched_component_ids[touched_component_ids > 0]
        hidden_fitted_lines = np.isin(fitted_components, touched_component_ids)
        fitted_lines = fitted_lines & ~hidden_fitted_lines

    # The saved debug snapshot contains a small outside-boundary residue that
    # is in none of the three exported masks.  It is unknown by definition in
    # the rendered three-state map.  This also guarantees exhaustive classes.
    unknown = ~(effective_free | occupied)
    drawn_kept_seeds = (
        np.zeros(free.shape, dtype=bool)
        if args.do_not_draw_kept_seeds
        else kept_seeds
    )
    known_content = (
        vertical_observed
        | effective_free
        | occupied
        | seeds
        | drawn_kept_seeds
        | rejected_seeds
        | fitted_lines
    )
    presentation_erased = np.zeros(free.shape, dtype=bool)
    if args.erase_below_row is not None:
        erase_row = int(args.erase_below_row)
        if not 0 <= erase_row <= free.shape[0]:
            raise ValueError("--erase-below-row lies outside the saved grid")
        presentation_erased[erase_row:, :] = True
        known_content &= ~presentation_erased
    rows, cols = np.nonzero(known_content)
    if rows.size == 0:
        raise ValueError("snapshot contains no observed Vertical-Free content")
    pad = max(0, int(args.padding_cells))
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(free.shape[0], int(rows.max()) + pad + 1)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(free.shape[1], int(cols.max()) + pad + 1)

    canvas = np.full((*free.shape, 3), UNKNOWN_RGB, dtype=np.uint8)
    canvas[effective_free] = FREE_RGB
    canvas[occupied] = OCCUPIED_RGB
    canvas[seeds] = seed_rgb
    canvas[rejected_seeds] = rejected_seed_rgb
    canvas[drawn_kept_seeds] = kept_seed_rgb
    canvas[fitted_lines] = fitted_line_rgb
    canvas[presentation_erased] = UNKNOWN_RGB
    crop = canvas[r0:r1, c0:c1]

    background = tuple(channel / 255.0 for channel in UNKNOWN_RGB)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.imshow(crop, interpolation="nearest", origin="upper")
    ax.set_aspect("equal")
    ax.axis("off")
    if not args.no_legend:
        handles = [
                Patch(facecolor=np.asarray(OCCUPIED_RGB) / 255.0, edgecolor="black", linewidth=0.8, label="occupied"),
                Patch(facecolor=np.asarray(FREE_RGB) / 255.0, edgecolor="black", linewidth=0.8, label="free"),
                Patch(facecolor=np.asarray(UNKNOWN_RGB) / 255.0, edgecolor="black", linewidth=0.8, label="unknown"),
            ]
        if seed_masks:
            handles.append(
                Patch(facecolor=np.asarray(seed_rgb) / 255.0, edgecolor="black", linewidth=0.8, label="raw seed")
            )
        if args.kept_seed_key and not args.do_not_draw_kept_seeds:
            handles.append(
                Patch(facecolor=np.asarray(kept_seed_rgb) / 255.0, edgecolor="black", linewidth=0.8, label="kept seed")
            )
        if args.rejected_seed_key:
            handles.append(
                Patch(facecolor=np.asarray(rejected_seed_rgb) / 255.0, edgecolor="black", linewidth=0.8, label="rejected seed")
            )
        if fitted_line_masks:
            handles.append(
                Patch(facecolor=np.asarray(fitted_line_rgb) / 255.0, edgecolor="black", linewidth=0.8, label="fitted line")
            )
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            fontsize=11,
            handlelength=1.1,
            handleheight=0.9,
            handletextpad=0.45,
            borderpad=0.15,
            labelspacing=0.28,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(args.dpi), facecolor=background, bbox_inches="tight", pad_inches=0.04)
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, facecolor=background, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    metadata = {
        "schema": "vertical_free_map_figure_v1",
        "source_snapshot": str(snapshot),
        "output_png": str(output),
        "output_pdf": str(pdf_output) if pdf_output is not None else None,
        "coverage_ratio": coverage,
        "grid_shape": list(free.shape),
        "crop_rc_exclusive": [r0, r1, c0, c1],
        "class_sources": {
            "free": "voxel_vertical_free_xy minus occupied mask",
            "occupied": str(args.occupied_key),
            "unknown": "complement of free union occupied",
        },
        "class_cells_full_grid": {
            "occupied": int(np.count_nonzero(occupied)),
            "free": int(np.count_nonzero(effective_free)),
            "unknown": int(np.count_nonzero(unknown)),
        },
        "saved_unknown_cells": int(np.count_nonzero(saved_unknown)),
        "raw_vertical_free_cells": int(np.count_nonzero(free)),
        "occupied_cells_overriding_raw_vertical_free": int(
            np.count_nonzero(occupied_free_overlap)
        ),
        "exported_mask_residue_rendered_unknown": int(
            np.count_nonzero(~(free | occupied | saved_unknown))
        ),
        "colors": {
            "occupied": "#E4979F",
            "free": "#FFFFFF",
            "unknown": "#D9D9D9",
        },
        "door_seed_drawn": bool(
            seed_masks
            or np.any(drawn_kept_seeds)
            or np.any(rejected_seeds)
        ),
        "door_seed_sources": [str(key) for key in args.seed_key],
        "door_seed_cells": int(np.count_nonzero(seeds)),
        "door_seed_color": "#%02X%02X%02X" % seed_rgb,
        "kept_seed_source": str(args.kept_seed_key) if args.kept_seed_key else None,
        "kept_seed_cells": int(np.count_nonzero(kept_seeds)),
        "kept_seed_cells_drawn": int(np.count_nonzero(drawn_kept_seeds)),
        "kept_seed_color": "#%02X%02X%02X" % kept_seed_rgb,
        "rejected_seed_source": str(args.rejected_seed_key) if args.rejected_seed_key else None,
        "rejected_seed_cells": int(np.count_nonzero(rejected_seeds)),
        "rejected_seed_color": "#%02X%02X%02X" % rejected_seed_rgb,
        "kept_rejected_seed_overlap_cells": int(
            np.count_nonzero(kept_seeds & rejected_seeds)
        ),
        "presentation_only_hidden_kept_seed_room_labels": [
            int(label)
            for label in args.hide_kept_components_contained_in_room_label
        ],
        "presentation_only_hidden_kept_seed_cells": int(
            np.count_nonzero(hidden_kept_seeds)
        ),
        "fitted_line_sources": [str(key) for key in args.fitted_line_key],
        "fitted_line_cells": int(np.count_nonzero(fitted_lines)),
        "fitted_line_color": "#%02X%02X%02X" % fitted_line_rgb,
        "presentation_only_hidden_fitted_line_cells": int(
            np.count_nonzero(hidden_fitted_lines)
        ),
        "door_line_drawn": bool(fitted_line_masks),
        "room_labels_drawn": False,
        "legend_drawn": not bool(args.no_legend),
        "presentation_cleanup": {
            "erase_below_grid_row_inclusive": args.erase_below_row,
            "displayed_cells_erased": int(
                np.count_nonzero(
                    (
                        vertical_observed
                        | effective_free
                        | occupied
                        | seeds
                        | drawn_kept_seeds
                        | rejected_seeds
                        | fitted_lines
                    )
                    & presentation_erased
                )
            ),
            "source_snapshot_modified": False,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
