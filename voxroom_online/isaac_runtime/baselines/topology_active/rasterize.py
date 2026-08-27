from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mask_io import build_metric_domain_from_source, relabel_consecutive
from ..offline.fallback_utils import wavefront_fill_unlabeled
from .geometry import bresenham_cells
from .topomap import TopologicalRoomMap


@dataclass(frozen=True)
class RasterizationResult:
    final_room_label_map: np.ndarray
    debug_arrays: dict[str, np.ndarray]


def rasterize_topomap(topomap: TopologicalRoomMap, source_arrays: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    domain = build_metric_domain_from_source(source_arrays)
    labels = np.zeros(domain.shape, dtype=np.int32)
    raw_node_labels = np.zeros(domain.shape, dtype=np.int32)
    for node_id in sorted(topomap.nodes):
        node = topomap.nodes[node_id]
        mask = np.asarray(node.mcl_mask, dtype=bool)
        if mask.shape != domain.shape:
            continue
        claim = mask & domain & (labels == 0)
        labels[claim] = int(node_id)
        raw_node_labels[mask & domain] = int(node_id)
    labels = wavefront_fill_unlabeled(labels, domain=domain)
    labels = relabel_consecutive(labels)
    debug = {
        "topology_node_label_map": raw_node_labels.astype(np.int32),
        "topology_mcl_label_map": labels.astype(np.int32),
        "topology_current_node_id": np.asarray(-1 if topomap.current_node_id is None else int(topomap.current_node_id), dtype=np.int32),
    }
    return labels.astype(np.int32), debug


def rasterize_topomap_to_label_map(
    *,
    topomap: TopologicalRoomMap,
    domain: np.ndarray,
    shape: tuple[int, int],
    door_pairs=(),
    frontier_clusters=(),
    current_node_id: int | None = None,
    policy_target_rc: tuple[int, int] | None = None,
) -> RasterizationResult:
    domain_bool = np.asarray(domain, dtype=bool)
    if domain_bool.shape != tuple(shape):
        domain_bool = np.zeros(tuple(shape), dtype=bool)
    labels = np.zeros(tuple(shape), dtype=np.int32)
    raw_node_labels = np.zeros(tuple(shape), dtype=np.int32)
    for node_id in sorted(topomap.nodes):
        node = topomap.nodes[node_id]
        mask = np.asarray(node.mcl_mask, dtype=bool)
        if mask.shape != tuple(shape):
            continue
        claim = mask & domain_bool & (labels == 0)
        labels[claim] = int(node_id)
        raw_node_labels[mask & domain_bool] = int(node_id)
    labels = wavefront_fill_unlabeled(labels, domain=domain_bool)
    labels = relabel_consecutive(labels)
    door_endpoint = np.zeros(tuple(shape), dtype=np.int32)
    waypoint = np.zeros(tuple(shape), dtype=np.int32)
    door_line = np.zeros(tuple(shape), dtype=np.int32)
    for pair in door_pairs:
        for rc in (pair.p0_rc, pair.p1_rc):
            r, c = int(rc[0]), int(rc[1])
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                door_endpoint[r, c] = int(pair.door_id)
        for rc in (pair.waypoint_a_rc, pair.waypoint_b_rc):
            r, c = int(rc[0]), int(rc[1])
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                waypoint[r, c] = int(pair.door_id)
        for r, c in bresenham_cells(pair.p0_rc, pair.p1_rc):
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                door_line[r, c] = int(pair.door_id)
    frontier_map = np.zeros(tuple(shape), dtype=np.int32)
    for cluster in frontier_clusters:
        mask = getattr(cluster, "mask", None)
        if mask is not None and np.asarray(mask).shape == tuple(shape):
            frontier_map[np.asarray(mask, dtype=bool)] = int(cluster.cluster_id)
    debug = {
        "topology_node_label_map": raw_node_labels.astype(np.int32),
        "topology_mcl_label_map": labels.astype(np.int32),
        "topology_door_endpoint_map": door_endpoint,
        "topology_waypoint_map": waypoint,
        "topology_door_pair_line_map": door_line,
        "topology_frontier_cluster_map": frontier_map,
        "topology_current_node_id": np.asarray(
            -1 if current_node_id is None else int(current_node_id),
            dtype=np.int32,
        ),
        "topology_baseline_policy_target_rc": np.asarray(
            [-1, -1] if policy_target_rc is None else [int(policy_target_rc[0]), int(policy_target_rc[1])],
            dtype=np.int32,
        ),
    }
    return RasterizationResult(final_room_label_map=labels.astype(np.int32), debug_arrays=debug)
