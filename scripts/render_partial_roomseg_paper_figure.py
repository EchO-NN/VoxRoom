#!/usr/bin/env python3
"""Render a compact paper figure from a saved room-segmentation snapshot."""

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


ROOM_COLORS = np.asarray(
    [
        (152, 198, 225),
        (244, 195, 130),
        (149, 206, 169),
        (189, 170, 215),
        (222, 211, 120),
        (154, 200, 202),
        (234, 153, 158),
        (244, 168, 203),
        (205, 180, 158),
    ],
    dtype=np.uint8,
)

# Used only when a method splits one reference room into several predicted
# regions.  These colors deliberately do not replace the shared reference
# palette; they make the additional local fragments visible instead of letting
# them visually merge into their dominant sibling.
SPLIT_ROOM_COLORS = np.asarray(
    [
        (127, 184, 224),
        (244, 171, 113),
        (122, 190, 143),
        (170, 139, 207),
        (237, 199, 101),
        (126, 185, 190),
        (222, 128, 139),
        (224, 133, 180),
    ],
    dtype=np.uint8,
)
DISPLAY_ROOM_COLORS = np.vstack((ROOM_COLORS, SPLIT_ROOM_COLORS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--padding-cells", type=int, default=22)
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument(
        "--method-name",
        default="voxroom",
        help="Method name recorded in the sidecar metadata.",
    )
    parser.add_argument(
        "--method-map-npz",
        type=Path,
        help="Optional NPZ carrying an alternative room-label and door-line map.",
    )
    parser.add_argument(
        "--label-key",
        default="voxel_final_room_label_map",
        help="Room-label array key in --method-map-npz.",
    )
    parser.add_argument(
        "--door-line-key",
        default="voxel_door_stable_cut_mask",
        help="Door-line array key in --method-map-npz.",
    )
    parser.add_argument(
        "--color-reference-npz",
        type=Path,
        help=(
            "Optional room-label NPZ used only to align the room palette. Each "
            "predicted room receives the color of its largest-overlap reference "
            "room; geometry and labels themselves are not changed."
        ),
    )
    parser.add_argument(
        "--color-reference-label-key",
        default="voxel_final_room_label_map",
        help="Room-label array key in --color-reference-npz.",
    )
    parser.add_argument(
        "--disambiguate-reference-splits",
        action="store_true",
        help=(
            "When multiple predicted rooms have the same best-overlap reference "
            "room, keep the largest one reference-colored and give every smaller "
            "split fragment a distinct supplementary color."
        ),
    )
    parser.add_argument(
        "--no-door-room-fill",
        action="store_true",
        help="Do not use VoxRoom accepted-seed support bands for presentation fill.",
    )
    parser.add_argument(
        "--derive-label-separator-lines",
        action="store_true",
        help=(
            "Render narrow occupied bands lying between two different predicted "
            "room labels as white separator lines. Useful for methods that emit "
            "room labels but no explicit separator mask."
        ),
    )
    parser.add_argument(
        "--separator-dilation-cells",
        type=int,
        default=6,
        help=(
            "Maximum label-to-separator distance in grid cells when deriving "
            "white separator lines (default: 6)."
        ),
    )
    parser.add_argument(
        "--erase-below-row",
        type=int,
        help="Presentation-only cleanup: hide all displayed cells at and below this grid row.",
    )
    return parser.parse_args()


def _label_boundary(labels: np.ndarray, visible: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(labels, shift=(dr, dc), axis=(0, 1))
        shifted_visible = np.roll(visible, shift=(dr, dc), axis=(0, 1))
        boundary |= visible & ((shifted != labels) | ~shifted_visible)
    return boundary


def _separator_bands_between_labels(
    labels: np.ndarray,
    candidate_cells: np.ndarray,
    dilation_cells: int,
) -> np.ndarray:
    """Find narrow candidate bands bordered by at least two room labels.

    SysNav publishes only a discrete room label image.  Its watershed borders
    remain as occupied pixels in the navigation map rather than being exported
    as a separate line mask.  This derives that *inter-room* subset only:
    exterior walls and furniture, which face at most one predicted room, are
    deliberately left untouched.
    """
    if dilation_cells < 1:
        raise ValueError("--separator-dilation-cells must be at least one")
    separator_support_count = np.zeros(labels.shape, dtype=np.uint8)
    structure = np.ones((3, 3), dtype=bool)
    for label in (int(value) for value in np.unique(labels) if value > 0):
        expanded = ndimage.binary_dilation(
            labels == label,
            structure=structure,
            iterations=dilation_cells,
        )
        separator_support_count += expanded.astype(np.uint8)
    return candidate_cells & (separator_support_count >= 2)


def _palette_labels_from_reference(
    labels: np.ndarray,
    reference_labels: np.ndarray | None,
    valid_overlap_domain: np.ndarray,
    disambiguate_splits: bool,
) -> tuple[dict[int, int], dict[int, dict[str, int]]]:
    """Match each predicted room to its largest-overlap reference room.

    This is intentionally a palette-only correspondence. It never remaps room
    IDs or alters masks. A merged prediction can only take the color of its
    largest-overlap reference room. By default, split predictions share their
    reference color too; the optional split-disambiguation mode instead keeps
    the largest one aligned and gives each smaller fragment a distinct extra
    color, so over-segmentation remains visible in a comparison panel.
    """
    predicted_ids = sorted(int(value) for value in np.unique(labels) if value > 0)
    palette_labels = {label: label for label in predicted_ids}
    correspondence: dict[int, dict[str, int]] = {}
    if reference_labels is None:
        return palette_labels, correspondence

    if reference_labels.shape != labels.shape:
        raise ValueError("color-reference labels do not share the render grid")
    for label in predicted_ids:
        overlap = reference_labels[
            (labels == label) & valid_overlap_domain & (reference_labels > 0)
        ]
        if overlap.size == 0:
            correspondence[label] = {
                "reference_label": 0,
                "overlap_cells": 0,
            }
            continue
        reference_ids, counts = np.unique(overlap, return_counts=True)
        best_index = int(np.argmax(counts))
        matched_label = int(reference_ids[best_index])
        palette_labels[label] = matched_label
        correspondence[label] = {
            "reference_label": matched_label,
            "overlap_cells": int(counts[best_index]),
        }
    if disambiguate_splits:
        groups: dict[int, list[tuple[int, int]]] = {}
        for predicted_label, values in correspondence.items():
            reference_label = values["reference_label"]
            if reference_label > 0:
                groups.setdefault(reference_label, []).append(
                    (predicted_label, values["overlap_cells"])
                )
        split_palette_index = len(ROOM_COLORS) + 1
        for group in groups.values():
            # The largest-overlap prediction is the shared visual counterpart;
            # all remaining fragments receive different supplemental colors.
            for predicted_label, _ in sorted(group, key=lambda item: (-item[1], item[0]))[1:]:
                palette_labels[predicted_label] = split_palette_index
                correspondence[predicted_label]["supplementary_split_color"] = (
                    split_palette_index
                )
                split_palette_index += 1
    return palette_labels, correspondence


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    output = args.output.expanduser().resolve()
    pdf_output = args.pdf_output.expanduser().resolve() if args.pdf_output else None

    with np.load(snapshot, allow_pickle=False) as arrays:
        required = (
            "roomseg_eval_reference_explorable_mask",
            "roomseg_eval_explored_reference_mask",
            "voxel_nav_occupied_xy",
            "voxel_nav_free_xy",
            "voxel_nav_observed_xy",
            "voxel_final_room_label_map",
            "voxel_door_stable_cut_mask",
            "voxel_door_seed_model_keep_mask",
            "roomseg_eval_coverage_ratio",
            "roomseg_eval_threshold",
            "roomseg_eval_explored_cells",
            "roomseg_eval_total_explorable_cells",
        )
        missing = [key for key in required if key not in arrays.files]
        if missing:
            raise KeyError("snapshot is missing: " + ", ".join(missing))

        reference = np.asarray(
            arrays["roomseg_eval_reference_explorable_mask"], dtype=bool
        )
        explored = np.asarray(
            arrays["roomseg_eval_explored_reference_mask"], dtype=bool
        )
        occupied = np.asarray(arrays["voxel_nav_occupied_xy"], dtype=bool)
        nav_free = np.asarray(arrays["voxel_nav_free_xy"], dtype=bool)
        observed = np.asarray(arrays["voxel_nav_observed_xy"], dtype=bool)
        labels = np.asarray(arrays["voxel_final_room_label_map"], dtype=np.int32)
        door_lines = np.asarray(arrays["voxel_door_stable_cut_mask"], dtype=bool)
        accepted_door_seeds = np.asarray(
            arrays["voxel_door_seed_model_keep_mask"], dtype=bool
        )
        coverage = float(np.asarray(arrays["roomseg_eval_coverage_ratio"]).reshape(()))
        threshold = float(np.asarray(arrays["roomseg_eval_threshold"]).reshape(()))
        explored_cells = int(
            np.asarray(arrays["roomseg_eval_explored_cells"]).reshape(())
        )
        total_cells = int(
            np.asarray(arrays["roomseg_eval_total_explorable_cells"]).reshape(())
        )

    method_map_npz = (
        args.method_map_npz.expanduser().resolve()
        if args.method_map_npz is not None
        else snapshot
    )
    if args.method_map_npz is not None:
        with np.load(method_map_npz, allow_pickle=False) as method_arrays:
            if args.label_key not in method_arrays.files:
                raise KeyError(
                    f"method map is missing label key {args.label_key}: {method_map_npz}"
                )
            if args.door_line_key not in method_arrays.files:
                raise KeyError(
                    "method map is missing door-line key "
                    f"{args.door_line_key}: {method_map_npz}"
                )
            labels = np.asarray(method_arrays[args.label_key], dtype=np.int32)
            door_lines = np.asarray(method_arrays[args.door_line_key], dtype=bool)

    color_reference_path = (
        args.color_reference_npz.expanduser().resolve()
        if args.color_reference_npz is not None
        else None
    )
    color_reference_labels: np.ndarray | None = None
    if color_reference_path is not None:
        with np.load(color_reference_path, allow_pickle=False) as reference_arrays:
            if args.color_reference_label_key not in reference_arrays.files:
                raise KeyError(
                    "color reference is missing label key "
                    f"{args.color_reference_label_key}: {color_reference_path}"
                )
            color_reference_labels = np.asarray(
                reference_arrays[args.color_reference_label_key], dtype=np.int32
            )

    if not (
        reference.shape
        == explored.shape
        == occupied.shape
        == nav_free.shape
        == observed.shape
        == labels.shape
        == door_lines.shape
        == accepted_door_seeds.shape
    ):
        raise ValueError("paper-figure arrays do not share one grid shape")
    if (
        color_reference_labels is not None
        and color_reference_labels.shape != labels.shape
    ):
        raise ValueError("color-reference labels do not share the render grid")

    explored &= reference
    observed_occupied = occupied & observed
    displayed = observed | explored
    presentation_erased = np.zeros(displayed.shape, dtype=bool)
    if args.erase_below_row is not None:
        erase_row = int(args.erase_below_row)
        if not 0 <= erase_row <= displayed.shape[0]:
            raise ValueError("--erase-below-row lies outside the saved grid")
        presentation_erased[erase_row:, :] = True
        displayed &= ~presentation_erased
        observed_occupied &= ~presentation_erased
    if args.derive_label_separator_lines:
        derived_separator_lines = _separator_bands_between_labels(
            labels,
            observed_occupied & displayed,
            int(args.separator_dilation_cells),
        )
        door_lines |= derived_separator_lines
    else:
        derived_separator_lines = np.zeros(labels.shape, dtype=bool)
    if args.no_door_room_fill:
        door_connected_seed_ids = np.empty((0,), dtype=np.int32)
        door_seed_visual = np.zeros(displayed.shape, dtype=bool)
        door_line_room_band = np.zeros(displayed.shape, dtype=bool)
    else:
        seed_components, _ = ndimage.label(
            accepted_door_seeds, structure=np.ones((3, 3), dtype=bool)
        )
        door_connected_seed_ids = np.unique(
            seed_components[door_lines & (seed_components > 0)]
        )
        door_seed_visual = (
            np.isin(seed_components, door_connected_seed_ids)
            & accepted_door_seeds
            & displayed
        )
        door_line_room_band = (
            ndimage.binary_dilation(
                door_lines, structure=np.ones((3, 3), dtype=bool), iterations=2
            )
            & displayed
            & ~door_lines
        )
    door_seed_room_mask = (door_seed_visual | door_line_room_band) & ~door_lines
    display_labels = labels.copy()
    _, nearest_label_indices = ndimage.distance_transform_edt(
        display_labels <= 0, return_indices=True
    )
    unlabeled_door_seed_room_mask = door_seed_room_mask & (display_labels <= 0)
    display_labels[unlabeled_door_seed_room_mask] = display_labels[
        nearest_label_indices[0][unlabeled_door_seed_room_mask],
        nearest_label_indices[1][unlabeled_door_seed_room_mask],
    ]
    # The evaluation reference mask is only the denominator/domain used to
    # compute exploration coverage.  It is not the visualization visibility
    # mask: valid, observed room labels can lie outside that reference domain
    # (most noticeably in door/clearance bands).  Restricting labels with
    # ``explored`` therefore paints already-observed free room cells dark and
    # can visually split one connected room.  Use the saved observation mask
    # for rendering instead; unobserved space remains untouched.
    observed_room_labels = (display_labels > 0) & observed & displayed
    visible_labels = observed_room_labels | (
        door_seed_room_mask & (display_labels > 0)
    )
    palette_labels, palette_correspondence = _palette_labels_from_reference(
        display_labels,
        color_reference_labels,
        nav_free & observed & displayed,
        bool(args.disambiguate_reference_splits),
    )

    # Exact dominant colors sampled from the supplied reference figure.
    # Unobserved/unknown map cells use the same paper color as the staged
    # Vertical-Free figures so cross-method panels share one semantic palette.
    background_color = (217, 217, 217)  # #D9D9D9
    explored_color = (58, 58, 58)
    canvas = np.full((*reference.shape, 3), background_color, dtype=np.uint8)
    canvas[displayed] = explored_color
    canvas[observed_occupied] = explored_color
    for label in sorted(
        int(value) for value in np.unique(display_labels[visible_labels])
    ):
        canvas[visible_labels & (display_labels == label)] = DISPLAY_ROOM_COLORS[
            (palette_labels.get(label, label) - 1) % len(DISPLAY_ROOM_COLORS)
        ]

    room_edges = _label_boundary(display_labels, visible_labels)
    # Do not replace room-mask cells with a full-grid-cell black outline.
    # Such an outline erodes one-cell and two-cell necks and can visually split
    # one connected room into multiple colored pieces. Observed walls already
    # provide the dark structural outline used by the reference rendering.
    canvas[door_lines & displayed] = (255, 255, 255)
    hidden_observed_room_labels = (
        (labels > 0)
        & observed
        & ~presentation_erased
        & ~door_lines
        & ~visible_labels
    )
    if np.any(hidden_observed_room_labels):
        raise RuntimeError("observed room-label cells were hidden by the renderer")
    remaining_dark_door_band = door_line_room_band & np.all(
        canvas == np.asarray(explored_color, dtype=np.uint8), axis=2
    )
    remaining_black_door_band = door_line_room_band & np.all(
        canvas == np.asarray((0, 0, 0), dtype=np.uint8), axis=2
    )
    if np.any(remaining_dark_door_band | remaining_black_door_band):
        raise RuntimeError("door-line support band still contains dark or black cells")

    content = displayed
    rows, cols = np.nonzero(content)
    if rows.size == 0:
        raise ValueError("snapshot contains no displayable map cells")
    pad = max(0, int(args.padding_cells))
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(reference.shape[0], int(rows.max()) + pad + 1)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(reference.shape[1], int(cols.max()) + pad + 1)
    crop = canvas[r0:r1, c0:c1]

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    background_rgb = tuple(channel / 255 for channel in background_color)
    fig.patch.set_facecolor(background_rgb)
    ax.set_facecolor(background_rgb)
    ax.imshow(crop, interpolation="nearest", origin="upper")
    ax.set_aspect("equal")
    ax.axis("off")
    if not args.no_legend:
        legend = ax.legend(
            handles=[
                Patch(
                    facecolor=background_rgb,
                    edgecolor="black",
                    linewidth=0.8,
                    label="unexplored cell",
                ),
                Patch(
                    facecolor=tuple(channel / 255 for channel in explored_color),
                    edgecolor="black",
                    linewidth=0.8,
                    label="explored cell",
                ),
            ],
            loc="upper right",
            frameon=False,
            fontsize=11,
            handlelength=1.1,
            handleheight=0.9,
            handletextpad=0.45,
            borderpad=0.15,
            labelspacing=0.28,
        )
        for text in legend.get_texts():
            text.set_color("black")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(args.dpi), facecolor=background_rgb, bbox_inches="tight", pad_inches=0.04)
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output, facecolor=background_rgb, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    metadata = {
        "schema": "partial_roomseg_paper_figure_v1",
        "source_snapshot": str(snapshot),
        "method_name": str(args.method_name),
        "method_map_npz": str(method_map_npz),
        "method_label_key": str(args.label_key),
        "method_door_line_key": str(args.door_line_key),
        "color_reference_npz": (
            str(color_reference_path) if color_reference_path is not None else None
        ),
        "color_reference_label_key": (
            str(args.color_reference_label_key)
            if color_reference_path is not None
            else None
        ),
        "palette_disambiguate_reference_splits": bool(
            args.disambiguate_reference_splits
        ),
        "palette_correspondence_largest_overlap": {
            str(label): values for label, values in palette_correspondence.items()
        },
        "derived_label_separator_lines": bool(args.derive_label_separator_lines),
        "derived_label_separator_dilation_cells": (
            int(args.separator_dilation_cells)
            if args.derive_label_separator_lines
            else None
        ),
        "derived_label_separator_cell_count": int(
            np.count_nonzero(derived_separator_lines & displayed)
        ),
        "door_room_presentation_fill_enabled": not bool(args.no_door_room_fill),
        "output_png": str(output),
        "output_pdf": str(pdf_output) if pdf_output is not None else None,
        "requested_threshold": threshold,
        "actual_coverage_ratio": coverage,
        "explored_cells": explored_cells,
        "total_explorable_cells": total_cells,
        "room_count_visible": int(np.unique(display_labels[visible_labels]).size),
        "navigation_free_source": "voxel_nav_free_xy",
        "navigation_clearance_applied": False,
        "navigation_free_cells": int(np.count_nonzero(nav_free)),
        "room_label_cells_outside_navigation_free": int(
            np.count_nonzero((labels > 0) & ~nav_free)
        ),
        "door_connected_accepted_seed_cells_rendered_as_room_mask": int(
            np.count_nonzero(door_seed_visual & ~door_lines)
        ),
        "two_cell_door_line_support_band_rendered_as_room_mask": int(
            np.count_nonzero(door_line_room_band)
        ),
        "combined_door_room_fill_cells": int(
            np.count_nonzero(door_seed_room_mask)
        ),
        "previously_unlabelled_door_seed_cells_filled_from_nearest_room": int(
            np.count_nonzero(unlabeled_door_seed_room_mask)
        ),
        "observed_room_label_cells_outside_evaluation_explored_mask_rendered": int(
            np.count_nonzero(visible_labels & observed & ~explored)
        ),
        "hidden_observed_room_label_cells": int(
            np.count_nonzero(hidden_observed_room_labels)
        ),
        "room_boundary_cells_not_raster_filled_black": int(
            np.count_nonzero(room_edges)
        ),
        "remaining_dark_or_black_cells_within_two_cells_of_door_lines": int(
            np.count_nonzero(remaining_dark_door_band | remaining_black_door_band)
        ),
        "door_connected_accepted_seed_component_count": int(
            door_connected_seed_ids.size
        ),
        "final_stable_door_line_cells": int(np.count_nonzero(door_lines & displayed)),
        "legend_drawn": not bool(args.no_legend),
        "presentation_cleanup": {
            "erase_below_grid_row_inclusive": args.erase_below_row,
            "displayed_cells_erased": int(
                np.count_nonzero((observed | explored) & presentation_erased)
            ),
            "source_snapshot_modified": False,
        },
        "crop_rc_exclusive": [r0, r1, c0, c1],
        "render_colors_rgb": {
            "unknown_unexplored_and_background": list(background_color),
            "explored": list(explored_color),
            "room_palette": ROOM_COLORS.tolist(),
            "supplementary_split_palette": SPLIT_ROOM_COLORS.tolist(),
            "final_stable_door_line": [255, 255, 255],
        },
        "legend_semantics": {
            "unknown_unexplored_cell": "not outlined; represented by #D9D9D9",
            "explored_cell": "explored but unlabelled cell or observed occupied cell",
            "pastel_colors": f"{args.method_name} room labels at this checkpoint",
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
