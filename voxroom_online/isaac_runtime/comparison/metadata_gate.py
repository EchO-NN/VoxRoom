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
    if not (metadata.get("original_repo") or metadata.get("original_repo_url") or method == "voxroom"):
        raise ValueError("%s metadata missing original repo" % method)
    if method != "voxroom" and not metadata.get("original_repo_commit"):
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
