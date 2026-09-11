from __future__ import annotations

from typing import Any, Mapping


BAD_MAIN_EXPERIMENT_WORDS = ("fallback", "smoke", "placeholder", "debug_only", "scaffold", "failure")


def assert_main_experiment_metadata(metadata: Mapping[str, Any], method: str) -> None:
    """Raise if baseline metadata is not valid for paper/main comparison metrics."""
    if str(metadata.get("method", metadata.get("baseline_name", method))) != str(method):
        raise ValueError("metadata method mismatch for %s: %s" % (method, metadata.get("method")))
    if metadata.get("main_experiment_allowed") is not True:
        raise ValueError("%s main_experiment_allowed is not true" % method)
    if bool(metadata.get("failed", False)):
        raise ValueError("%s metadata has failed=true" % method)
    for key in ("runner_type", "fallback_scope", "allowed_usage", "approximation_note"):
        value = str(metadata.get(key, "")).lower()
        bad = [word for word in BAD_MAIN_EXPERIMENT_WORDS if word in value]
        if bad:
            raise ValueError("%s metadata %s contains non-main marker %s: %s" % (method, key, bad[0], value))
    source_unavailable_reproduction = bool(
        method == "gomez_incremental"
        and metadata.get("source_code_available") is False
        and metadata.get("implementation_reference")
    )
    if not (
        metadata.get("original_repo")
        or metadata.get("original_repo_url")
        or method == "voxroom"
        or source_unavailable_reproduction
    ):
        raise ValueError("%s metadata missing original repo" % method)
    if (
        method != "voxroom"
        and not source_unavailable_reproduction
        and not metadata.get("original_repo_commit")
    ):
        raise ValueError("%s metadata missing original_repo_commit" % method)
    if not (metadata.get("map_resolution_m") or metadata.get("map_info") or method == "voxroom"):
        raise ValueError("%s metadata missing map resolution" % method)

    if method == "topology_visual_active":
        _require(metadata, "runner_type", {"original_active_room_segmentation", "original_repo_adapter"}, method)
        _require(metadata, "door_detector", {"original_detr"}, method)
        _require_true(metadata, "door_detector_available", method)
        _require_true(metadata, "detector_adapter_verified", method)
        _require_true(metadata, "projection_verified", method)
        _require_true(metadata, "topology_state_machine_verified", method)
        _require_present(metadata, "checkpoint_sha256", method)
        _require(metadata, "policy_control", {"never"}, method)
        _require_min_int(metadata, "panorama_views_saved", 12, method)
    elif method == "dude_incremental":
        _require(metadata, "runner_type", {"original_ros"}, method)
        _require(metadata, "input_topic", {"/map"}, method)
        _require(metadata, "output_topic", {"/tagged_image"}, method)
        _require_true(metadata, "incremental_state_reset_per_scene", method)
        _require_true(metadata, "incremental_order_enforced", method)
    elif method == "dude_offline":
        _require(metadata, "runner_type", {"original_ros"}, method)
        _require(metadata, "input_topic", {"/map"}, method)
        _require(metadata, "output_topic", {"/tagged_image"}, method)
        _require_true(metadata, "offline_state_reset_per_snapshot", method)
        _require(metadata, "state_reset_scope", {"snapshot"}, method)
    elif method == "gomez_incremental":
        _require(metadata, "runner_type", {"paper_reproduction"}, method)
        _require_true(metadata, "source_unavailable_reported_by_tvars_paper", method)
        _require_true(metadata, "incremental_state_reset_per_scene", method)
        _require_true(metadata, "incremental_order_enforced", method)
        _require_true(metadata, "door_lines_persistent", method)
        _require(
            metadata,
            "raw_seed_source",
            {"saved_tvars_original_hough_door_seed_map"},
            method,
        )
        if metadata.get("checkpoint_virtual_laser_recomputed") is not False:
            raise ValueError(
                "%s metadata checkpoint_virtual_laser_recomputed is not false"
                % method
            )
        _require_present(metadata, "raw_seed_source_snapshot", method)
        _require_true(metadata, "voxel_hough_confirmation_enabled", method)
        _require(
            metadata,
            "hough_confirmation_source",
            {"voxel_occupancy_state_zyx_vertical_cross_section"},
            method,
        )
        _require_present(metadata, "implementation_reference", method)
    elif method == "rose2":
        _require(metadata, "runner_type", {"original_ros"}, method)
        _require(metadata, "input_topic", {"/map"}, method)
        _require_any(metadata, ("output_source", "output_topic"), {"/rooms", "/features_ROSE2", "/ROSE2Srv"}, method)
        _require_true(metadata, "output_topic_confirmed_from_source", method)
    elif method in {"morphological", "distance_transform", "voronoi"}:
        _require(metadata, "runner_type", {"original_ros_action"}, method)
        _require(metadata, "original_package", {"ipa_room_segmentation"}, method)
        _require(metadata, "input_image_encoding", {"mono8"}, method)
        _require_true(metadata, "algorithm_id_verified", method)
        _require_present(metadata, "action_type", method)
        _require_present(metadata, "action_server", method)
        if int(metadata.get("input_free_value", -1)) != 255:
            raise ValueError("%s metadata input_free_value must be 255" % method)
        if int(metadata.get("input_occupied_value", -1)) != 0:
            raise ValueError("%s metadata input_occupied_value must be 0" % method)
        if not isinstance(metadata.get("room_segmentation_algorithm"), int):
            raise ValueError("%s metadata room_segmentation_algorithm must be int" % method)
        expected_algorithm_id = {
            "morphological": 1,
            "distance_transform": 2,
            "voronoi": 3,
        }[method]
        if int(metadata.get("room_segmentation_algorithm")) != expected_algorithm_id:
            raise ValueError(
                "%s metadata room_segmentation_algorithm=%s != expected %d"
                % (method, metadata.get("room_segmentation_algorithm"), expected_algorithm_id)
            )

    if method in {
        "dude_incremental",
        "dude_offline",
        "gomez_incremental",
        "rose2",
        "morphological",
        "distance_transform",
        "voronoi",
    }:
        _require_selected_segmentation_source(metadata, method)


def _require(metadata: Mapping[str, Any], key: str, allowed: set[str], method: str) -> None:
    value = str(metadata.get(key, ""))
    if value not in allowed:
        raise ValueError("%s metadata %s=%s not in %s" % (method, key, value, sorted(allowed)))


def _require_any(metadata: Mapping[str, Any], keys: tuple[str, ...], allowed: set[str], method: str) -> None:
    for key in keys:
        if str(metadata.get(key, "")) in allowed:
            return
    values = {key: metadata.get(key) for key in keys}
    raise ValueError("%s metadata none of %s is in %s" % (method, values, sorted(allowed)))


def _require_true(metadata: Mapping[str, Any], key: str, method: str) -> None:
    if metadata.get(key) is not True:
        raise ValueError("%s metadata %s is not true" % (method, key))


def _require_present(metadata: Mapping[str, Any], key: str, method: str) -> None:
    if not metadata.get(key):
        raise ValueError("%s metadata missing %s" % (method, key))


def _require_min_int(metadata: Mapping[str, Any], key: str, minimum: int, method: str) -> None:
    try:
        value = int(metadata.get(key))
    except Exception as exc:
        raise ValueError("%s metadata %s must be int >= %d" % (method, key, minimum)) from exc
    if value < int(minimum):
        raise ValueError("%s metadata %s=%d < %d" % (method, key, value, minimum))


def _require_selected_segmentation_source(
    metadata: Mapping[str, Any], method: str
) -> None:
    source: Any = metadata.get("segmentation_source")
    if method == "rose2":
        input_grid = metadata.get("input_grid")
        if isinstance(input_grid, Mapping):
            source = input_grid.get("free_source_key")
    elif method in {"morphological", "distance_transform", "voronoi"}:
        source = metadata.get("input_free_definition")
    mode = str(metadata.get("segmentation_input_mode") or "raw_vertical_free")
    allowed_by_mode = {
        "raw_vertical_free": {
            "voxel_vertical_free_xy",
            "height_profile_vertical_free_xy",
            "vertical_free_room_domain",
        },
        "raw_nav_free_no_clearance": {
            "voxel_nav_free_xy",
            "navigation_free_room_domain",
        },
    }
    allowed = allowed_by_mode.get(mode)
    if allowed is None:
        raise ValueError(
            "%s metadata has unsupported segmentation_input_mode=%s"
            % (method, mode)
        )
    if str(source) not in allowed:
        raise ValueError(
            "%s metadata segmentation source=%s is incompatible with %s; expected one of %s"
            % (method, source, mode, sorted(allowed))
        )
