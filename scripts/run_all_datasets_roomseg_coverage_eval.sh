#!/usr/bin/env bash
set -uo pipefail

VOXROOM_ROOT="${VOXROOM_ROOT:-$HOME/VoxRoom}"
ACTIVE_ROOM_ROOT="${ACTIVE_ROOM_ROOT:-$HOME/Active_room_segmentation}"
ACTIVE_ROOM_PYTHON="${ACTIVE_ROOM_PYTHON:-$HOME/.conda/envs/active-room-seg/bin/python}"
BATCH_TAG="${BATCH_TAG:-all_datasets_approved23_union_fixed0p5_max7000_$(date +%Y%m%d_%H%M%S)}"
BATCH_ROOT="${BATCH_ROOT:-$HOME/roomseg_evaluation/$BATCH_TAG}"
MAX_STEPS="${MAX_STEPS:-7000}"
COVERAGE_MILESTONES="${COVERAGE_MILESTONES:-20,40,60,70,80,90}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MIN_AVAILABLE_RAM_MB="${MIN_AVAILABLE_RAM_MB:-16000}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-12000}"
RESOURCE_POLL_SECONDS="${RESOURCE_POLL_SECONDS:-30}"
MAX_PARALLEL_SCENES="${MAX_PARALLEL_SCENES:-2}"
PARALLEL_LAUNCH_STAGGER_SECONDS="${PARALLEL_LAUNCH_STAGGER_SECONDS:-10}"
CONTINUE_ON_VALIDATION_BLOCKED="${CONTINUE_ON_VALIDATION_BLOCKED:-1}"
DATASETS="${DATASETS:-habitat,interioragent,grscene}"
IFS=',' read -r -a DATASET_ORDER <<<"$DATASETS"
LIVE_BASELINE_POLICY_CONTROL="${LIVE_BASELINE_POLICY_CONTROL:-original_topology}"
NAVIGATION_PLANNER_BACKEND="${NAVIGATION_PLANNER_BACKEND:-astar}"
TRANSLATION_STEP_SCALE="${TRANSLATION_STEP_SCALE:-18.0}"
TRANSLATION_COMMAND_SCALE="${TRANSLATION_COMMAND_SCALE:-6.0}"
BASE_MAX_VX_MPS="${BASE_MAX_VX_MPS:-0.15}"
BASE_MAX_VY_MPS="${BASE_MAX_VY_MPS:-0.0}"
MAX_VX_MPS="${MAX_VX_MPS:-2.70}"
MAX_VY_MPS="${MAX_VY_MPS:-0.0}"
CONTROL_DT="${CONTROL_DT:-0.2}"
TURN_STEP_DEG="${TURN_STEP_DEG:-30.0}"
MAX_WZ_RADPS="${MAX_WZ_RADPS:-2.6179938779914944}"
FORWARD_HEADING_TOLERANCE_DEG="${FORWARD_HEADING_TOLERANCE_DEG:-75.0}"
ROBOT_RADIUS_M="${ROBOT_RADIUS_M:-0.15}"
RUNTIME_PLANNING_CLEARANCE_M="${RUNTIME_PLANNING_CLEARANCE_M:-0.01}"
SIMULATOR_COLLISION_REFERENCE_RADIUS_M="${SIMULATOR_COLLISION_REFERENCE_RADIUS_M:-0.05}"
ASTAR_CLEARANCE_DESIRED_M="${ASTAR_CLEARANCE_DESIRED_M:-0.20}"
ASTAR_CLEARANCE_HARD_MIN_M="${ASTAR_CLEARANCE_HARD_MIN_M:-0.05}"
LOOKAHEAD_MIN_CLEARANCE_M="${LOOKAHEAD_MIN_CLEARANCE_M:-$ASTAR_CLEARANCE_HARD_MIN_M}"
GUARD_MIN_CLEARANCE_M="${GUARD_MIN_CLEARANCE_M:-$ASTAR_CLEARANCE_HARD_MIN_M}"
ASTAR_CLEARANCE_WEIGHT="${ASTAR_CLEARANCE_WEIGHT:-6.0}"
REPLAN_EVERY_STEPS="${REPLAN_EVERY_STEPS:-10}"
MAX_UNEXPLORED_COMPONENT_AREA_M2="${MAX_UNEXPLORED_COMPONENT_AREA_M2:-0.50}"
ISAAC_HEADLESS_FLAG="${ISAAC_HEADLESS_FLAG-}"

EXPECTED_ACTIVE_ROOM_COMMIT="${EXPECTED_ACTIVE_ROOM_COMMIT:-c6dbe92c55ea34f9710ddcc5b10d59144662fe68}"
EXPECTED_CONFIG_SHA256="${EXPECTED_CONFIG_SHA256:-ddc1f76daed49bac8d6d998c0a8a6ccbef21df1086af89762cef598d4793568c}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-941d0f613326630f61a5287579830a2212bce908ddbe8c190bca2125f489f732}"
EXPECTED_HABITAT_COUNT="${EXPECTED_HABITAT_COUNT:-14}"
EXPECTED_INTERIORAGENT_COUNT="${EXPECTED_INTERIORAGENT_COUNT:-25}"
EXPECTED_GRSCENE_COUNT="${EXPECTED_GRSCENE_COUNT:-69}"

VOXROOM_CONFIG="${VOXROOM_CONFIG:-$VOXROOM_ROOT/configs/voxroom_online.yaml}"
INTERIORAGENT_EPISODE_LIST="${INTERIORAGENT_EPISODE_LIST:-$VOXROOM_ROOT/data/interioragent_episodes/radius015_physical010_targetlocal_20260809/episode_files.txt}"
GRSCENE_EPISODE_LIST="${GRSCENE_EPISODE_LIST:-$VOXROOM_ROOT/data/grscene_home_episodes/radius015_physical010_targetlocal_20260809/episode_files.txt}"
GIBSON_VAL_DATASET="${GIBSON_VAL_DATASET:-$ACTIVE_ROOM_ROOT/data/datasets/pointnav/gibson/v1/val/val.json.gz}"

STATUS_PATH="$BATCH_ROOT/status.jsonl"
SUMMARY_PATH="$BATCH_ROOT/summary.json"
CURRENT_PATH="$BATCH_ROOT/current.json"
CONTRACT_PATH="$BATCH_ROOT/contract.json"
SOURCE_MANIFEST="$BATCH_ROOT/source_manifest.sha256"
MANIFEST_DIR="$BATCH_ROOT/manifests"
LOG_DIR="$BATCH_ROOT/logs"
PREPARED_DIR="$BATCH_ROOT/prepared"
ACTIVE_DIR="$BATCH_ROOT/active"
STATE_LOCK_PATH="$BATCH_ROOT/.state.lock"

die() {
  echo "[all-datasets] fatal: $*" >&2
  exit 1
}

validate_clearance_contract() {
  python3 - "$ASTAR_CLEARANCE_HARD_MIN_M" "$LOOKAHEAD_MIN_CLEARANCE_M" "$GUARD_MIN_CLEARANCE_M" <<'PY'
import math
import sys

astar_hard_min, lookahead_min, guard_min = map(float, sys.argv[1:])
if not math.isclose(lookahead_min, astar_hard_min, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("LOOKAHEAD_MIN_CLEARANCE_M must equal ASTAR_CLEARANCE_HARD_MIN_M")
if not math.isclose(guard_min, astar_hard_min, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("GUARD_MIN_CLEARANCE_M must equal ASTAR_CLEARANCE_HARD_MIN_M")
PY
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "missing executable: $1"
}

dataset_enabled() {
  local requested="$1" dataset
  for dataset in "${DATASET_ORDER[@]}"; do
    if [[ "$dataset" == "$requested" ]]; then
      return 0
    fi
  done
  return 1
}

validate_dataset_selection() {
  python3 - "$DATASETS" <<'PY'
import sys

selected = sys.argv[1].split(",")
allowed = {"habitat", "interioragent", "grscene"}
if not selected or any(not value for value in selected):
    raise SystemExit("DATASETS must be a non-empty comma-separated list")
if len(selected) != len(set(selected)):
    raise SystemExit(f"DATASETS contains duplicates: {selected}")
unknown = [value for value in selected if value not in allowed]
if unknown:
    raise SystemExit(f"DATASETS contains unsupported values: {unknown}")
PY
  if [[ "$LIVE_BASELINE_POLICY_CONTROL" != "original_topology" ]]; then
    echo "strict TVARS batch requires LIVE_BASELINE_POLICY_CONTROL=original_topology" >&2
    return 1
  fi
  if [[ "$NAVIGATION_PLANNER_BACKEND" != "astar" ]]; then
    echo "strict TVARS batch requires NAVIGATION_PLANNER_BACKEND=astar" >&2
    return 1
  fi
  if [[ -z "$ISAAC_HEADLESS_FLAG" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "strict visual batch requires DISPLAY or WAYLAND_DISPLAY" >&2
    return 1
  fi
  python3 - "$TRANSLATION_STEP_SCALE" "$TRANSLATION_COMMAND_SCALE" "$BASE_MAX_VX_MPS" "$BASE_MAX_VY_MPS" "$MAX_VX_MPS" "$MAX_VY_MPS" "$CONTROL_DT" "$TURN_STEP_DEG" "$MAX_WZ_RADPS" "$FORWARD_HEADING_TOLERANCE_DEG" <<'PY'
import math
import sys

(
    scale,
    command_scale,
    base_vx,
    base_vy,
    max_vx,
    max_vy,
    control_dt,
    turn_step_deg,
    max_wz,
    forward_tolerance_deg,
) = map(float, sys.argv[1:])
if not math.isclose(scale, 18.0, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("strict batch requires TRANSLATION_STEP_SCALE=18.0")
if not math.isclose(command_scale, 6.0, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("strict batch requires TRANSLATION_COMMAND_SCALE=6.0")
if not math.isclose(max_vx, base_vx * scale, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("MAX_VX_MPS does not match TRANSLATION_STEP_SCALE")
if not math.isclose(max_vy, base_vy * scale, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("MAX_VY_MPS does not match TRANSLATION_STEP_SCALE")
if not math.isclose(turn_step_deg, 30.0, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("strict TVARS control requires TURN_STEP_DEG=30.0")
if not math.isclose(forward_tolerance_deg, 75.0, rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("forward-only arc control requires FORWARD_HEADING_TOLERANCE_DEG=75.0")
if not math.isclose(max_wz * control_dt, math.radians(turn_step_deg), rel_tol=0.0, abs_tol=1.0e-9):
    raise SystemExit("MAX_WZ_RADPS and CONTROL_DT do not produce one TVARS turn step")
PY
  python3 - "$ROBOT_RADIUS_M" "$RUNTIME_PLANNING_CLEARANCE_M" "$SIMULATOR_COLLISION_REFERENCE_RADIUS_M" <<'PY'
import sys

robot_radius, runtime_margin, reference_radius = map(float, sys.argv[1:])
if robot_radius <= 0.0 or runtime_margin < 0.0 or reference_radius < 0.0:
    raise SystemExit("collision-clearance radii are invalid")
if robot_radius <= reference_radius:
    raise SystemExit("runtime collision clearance must exceed the static reference radius")
PY
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

checkpoint_path_from_config() {
  python3 - "$VOXROOM_CONFIG" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"^\s*checkpoint_path:\s*(\S+)\s*$", text, re.MULTILINE)
if match is None:
    raise SystemExit("checkpoint_path is missing from VoxRoom config")
print(match.group(1))
PY
}

checkpoint_absolute_path() {
  local configured
  configured="$(checkpoint_path_from_config)" || return 1
  if [[ "$configured" == /* ]]; then
    printf '%s\n' "$configured"
  else
    printf '%s\n' "$VOXROOM_ROOT/$configured"
  fi
}

check_fixed_contract() {
  local config_hash checkpoint_path checkpoint_hash active_commit
  config_hash="$(sha256_of "$VOXROOM_CONFIG")" || return 1
  if [[ "$config_hash" != "$EXPECTED_CONFIG_SHA256" ]]; then
    echo "[all-datasets] config drift: expected=$EXPECTED_CONFIG_SHA256 actual=$config_hash" >&2
    return 1
  fi
  checkpoint_path="$(checkpoint_absolute_path)" || return 1
  require_file "$checkpoint_path"
  checkpoint_hash="$(sha256_of "$checkpoint_path")" || return 1
  if [[ "$checkpoint_hash" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
    echo "[all-datasets] checkpoint drift: expected=$EXPECTED_CHECKPOINT_SHA256 actual=$checkpoint_hash" >&2
    return 1
  fi
  active_commit="$(git -C "$ACTIVE_ROOM_ROOT" rev-parse HEAD)" || return 1
  if [[ "$active_commit" != "$EXPECTED_ACTIVE_ROOM_COMMIT" ]]; then
    echo "[all-datasets] Active Room commit drift: expected=$EXPECTED_ACTIVE_ROOM_COMMIT actual=$active_commit" >&2
    return 1
  fi
  if [[ -n "$(git -C "$ACTIVE_ROOM_ROOT" status --porcelain)" ]]; then
    echo "[all-datasets] Active Room checkout is dirty" >&2
    return 1
  fi
  if [[ -f "$SOURCE_MANIFEST" ]] && ! sha256sum -c --quiet "$SOURCE_MANIFEST"; then
    echo "[all-datasets] VoxRoom source drift detected" >&2
    return 1
  fi
}

write_source_manifest() {
  if [[ -f "$SOURCE_MANIFEST" ]]; then
    return 0
  fi
  {
    find "$VOXROOM_ROOT/voxroom_online" -type f -name '*.py' -print0
    find "$VOXROOM_ROOT/scripts" -maxdepth 2 -type f -name '*.sh' -print0
    find "$VOXROOM_ROOT/configs" -type f -name '*.yaml' -print0
  } | sort -z | xargs -0 sha256sum >"$SOURCE_MANIFEST"
  chmod 0444 "$SOURCE_MANIFEST"
}

prepare_scene_lists() {
  local habitat_list="$MANIFEST_DIR/habitat_scenes.txt"
  local interior_list="$MANIFEST_DIR/interioragent_episodes.txt"
  local grscene_list="$MANIFEST_DIR/grscene_episodes.txt"
  if dataset_enabled habitat && [[ ! -f "$habitat_list" ]]; then
    "$ACTIVE_ROOM_PYTHON" - "$GIBSON_VAL_DATASET" >"$habitat_list" <<'PY'
import gzip
import json
import pathlib
import sys

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    payload = json.load(stream)
for scene in sorted({pathlib.Path(row["scene_id"]).stem for row in payload.get("episodes", [])}):
    print(scene)
PY
  fi
  if dataset_enabled interioragent && [[ ! -f "$interior_list" ]]; then
    install -m 0444 "$INTERIORAGENT_EPISODE_LIST" "$interior_list"
  fi
  if dataset_enabled grscene && [[ ! -f "$grscene_list" ]]; then
    install -m 0444 "$GRSCENE_EPISODE_LIST" "$grscene_list"
  fi
  python3 - "$VOXROOM_ROOT" "$DATASETS" "$habitat_list" "$interior_list" "$grscene_list" \
    "$EXPECTED_HABITAT_COUNT" "$EXPECTED_INTERIORAGENT_COUNT" "$EXPECTED_GRSCENE_COUNT" \
    "$ROBOT_RADIUS_M" "$RUNTIME_PLANNING_CLEARANCE_M" "$SIMULATOR_COLLISION_REFERENCE_RADIUS_M" <<'PY'
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
selected = set(sys.argv[2].split(","))
specs = [
    ("habitat", Path(sys.argv[3]), int(sys.argv[6]), False),
    ("interioragent", Path(sys.argv[4]), int(sys.argv[7]), True),
    ("grscene", Path(sys.argv[5]), int(sys.argv[8]), True),
]
robot_radius_m = float(sys.argv[9])
runtime_planning_clearance_m = float(sys.argv[10])
static_reference_radius_m = float(sys.argv[11])
expected_extra_clearance_m = max(
    0.0,
    robot_radius_m - static_reference_radius_m,
)
for name, path, expected, paths_are_files in specs:
    if name not in selected:
        continue
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected or len(set(rows)) != expected:
        raise SystemExit(f"{name} scene contract mismatch: expected={expected} rows={len(rows)} unique={len(set(rows))}")
    if paths_are_files:
        missing = []
        for row in rows:
            candidate = Path(row)
            if not candidate.is_absolute():
                candidate = root / candidate
            if not candidate.is_file():
                missing.append(str(candidate))
        if missing:
            raise SystemExit(f"{name} episode files missing: {missing[:3]}")
        for row in rows:
            candidate = Path(row)
            if not candidate.is_absolute():
                candidate = root / candidate
            records = [
                json.loads(line)
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(records) != 1:
                raise SystemExit(
                    f"{name} strict episode file must contain one record: {candidate}"
                )
            metadata = dict(records[0].get("metadata") or {})
            expected_values = {
                "robot_radius_m": robot_radius_m,
                "runtime_planning_clearance_m": runtime_planning_clearance_m,
                "static_reference_radius_m": static_reference_radius_m,
                "static_execution_extra_clearance_m": expected_extra_clearance_m,
            }
            for key, expected_value in expected_values.items():
                if key not in metadata or not math.isclose(
                    float(metadata[key]),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise SystemExit(
                        f"{name} episode execution contract mismatch: "
                        f"file={candidate} key={key} "
                        f"expected={expected_value} actual={metadata.get(key)}"
                    )
            if metadata.get("static_execution_contract") not in {
                "object_clearance_then_reference_radius_delta",
                "usd_visible_face_collision_then_reference_radius_delta",
            }:
                raise SystemExit(
                    f"{name} episode has an unexpected execution contract: {candidate}"
                )
PY
}

write_contract() {
  if [[ -f "$CONTRACT_PATH" ]]; then
    return 0
  fi
  local checkpoint_path
  checkpoint_path="$(checkpoint_absolute_path)" || return 1
  python3 - "$CONTRACT_PATH" "$BATCH_TAG" "$MAX_STEPS" "$COVERAGE_MILESTONES" "$DATASETS" "$LIVE_BASELINE_POLICY_CONTROL" "$NAVIGATION_PLANNER_BACKEND" \
    "$TRANSLATION_STEP_SCALE" "$TRANSLATION_COMMAND_SCALE" "$BASE_MAX_VX_MPS" "$BASE_MAX_VY_MPS" "$MAX_VX_MPS" "$MAX_VY_MPS" \
    "$CONTROL_DT" "$TURN_STEP_DEG" "$MAX_WZ_RADPS" "$FORWARD_HEADING_TOLERANCE_DEG" \
    "$ROBOT_RADIUS_M" "$RUNTIME_PLANNING_CLEARANCE_M" "$SIMULATOR_COLLISION_REFERENCE_RADIUS_M" \
    "$ASTAR_CLEARANCE_DESIRED_M" "$ASTAR_CLEARANCE_HARD_MIN_M" "$LOOKAHEAD_MIN_CLEARANCE_M" "$GUARD_MIN_CLEARANCE_M" "$ASTAR_CLEARANCE_WEIGHT" "$REPLAN_EVERY_STEPS" "$MAX_UNEXPLORED_COMPONENT_AREA_M2" \
    "$EXPECTED_HABITAT_COUNT" "$EXPECTED_INTERIORAGENT_COUNT" "$EXPECTED_GRSCENE_COUNT" \
    "$VOXROOM_CONFIG" "$EXPECTED_CONFIG_SHA256" "$checkpoint_path" "$EXPECTED_CHECKPOINT_SHA256" \
    "$EXPECTED_ACTIVE_ROOM_COMMIT" "$INTERIORAGENT_EPISODE_LIST" "$GRSCENE_EPISODE_LIST" \
    "$MAX_PARALLEL_SCENES" "$PARALLEL_LAUNCH_STAGGER_SECONDS" "$CONTINUE_ON_VALIDATION_BLOCKED" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

(
    output_path,
    batch_tag,
    max_steps,
    milestones,
    datasets,
    live_baseline_policy_control,
    navigation_planner_backend,
    translation_step_scale,
    translation_command_scale,
    base_max_vx_mps,
    base_max_vy_mps,
    max_vx_mps,
    max_vy_mps,
    control_dt,
    turn_step_deg,
    max_wz_radps,
    forward_heading_tolerance_deg,
    robot_radius_m,
    runtime_planning_clearance_m,
    simulator_collision_reference_radius_m,
    astar_clearance_desired_m,
    astar_clearance_hard_min_m,
    lookahead_min_clearance_m,
    guard_min_clearance_m,
    astar_clearance_weight,
    replan_every_steps,
    max_unexplored_component_area_m2,
    habitat_count,
    interior_count,
    grscene_count,
    config_path,
    config_sha,
    checkpoint_path,
    checkpoint_sha,
    active_commit,
    interior_list,
    grscene_list,
    max_parallel_scenes,
    parallel_launch_stagger_seconds,
    continue_on_validation_blocked,
) = sys.argv[1:]
selected_datasets = datasets.split(",")
all_scene_counts = {
    "habitat": int(habitat_count),
    "interioragent": int(interior_count),
    "grscene": int(grscene_count),
}
scene_counts = {
    dataset: all_scene_counts[dataset]
    for dataset in selected_datasets
}
payload = {
    "schema_version": "voxroom_all_datasets_roomseg_eval_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "batch_tag": batch_tag,
    "max_steps_per_scene": int(max_steps),
    "coverage_milestones_percent": [int(value) for value in milestones.split(",")],
    "coverage_reference_source": "preprocessed_scene_same_floor_per_room_reachable_navigable_before_runtime_clearance",
    "coverage_reference_component_filter_applied": True,
    "coverage_reference_component_filter": "largest_8_connected_navigable_component_per_room_polygon",
    "coverage_reference_component_filter_without_room_polygons": "largest_8_connected_navigable_component_global_no_room_polygons",
    "execution_order": selected_datasets,
    "max_parallel_scenes": int(max_parallel_scenes),
    "parallel_launch_stagger_seconds": int(parallel_launch_stagger_seconds),
    "continue_on_validation_blocked": bool(int(continue_on_validation_blocked)),
    "exploration_policy": "tvars_original_isaac",
    "tvars_exploration_order": (
        "persistent_original_door_current_room_vertical_free_frontiers_"
        "then_choose_door_then_global_vertical_free_stop_guard"
    ),
    "tvars_persistent_door_memory": True,
    "live_baseline_policy_control": live_baseline_policy_control,
    "navigation_planner_backend": navigation_planner_backend,
    "isaac_short_term_planner": "clearance_astar",
    "isaac_astar_path_planner_enabled": True,
    "simulator_collision_guard_enabled": True,
    "simulator_collision_guard_source": "preprocessed_usd_collision_navigable_execution_only",
    "execution_guard_source": "preprocessed_usd_collision_guard_with_runtime_contact_feedback",
    "online_planning_source": "voxroom_online_plus_observed_collision_feedback",
    "online_planning_uses_posterior_static_map": False,
    "episode_execution_contract": "strict_static_execution_contract",
    "robot_radius_m": float(robot_radius_m),
    "runtime_planning_clearance_m": float(runtime_planning_clearance_m),
    "astar_clearance_cost_enabled": True,
    "astar_clearance_desired_m": float(astar_clearance_desired_m),
    "astar_clearance_hard_min_m": float(astar_clearance_hard_min_m),
    "lookahead_min_clearance_m": float(lookahead_min_clearance_m),
    "guard_min_clearance_m": float(guard_min_clearance_m),
    "planner_execution_clearance_contract": "astar_lookahead_guard_equal",
    "astar_clearance_weight": float(astar_clearance_weight),
    "astar_clearance_power": 2.0,
    "astar_replan_every_steps": int(replan_every_steps),
    "max_unexplored_component_area_m2_at_policy_stop": float(
        max_unexplored_component_area_m2
    ),
    "simulator_collision_reference_radius_m": float(simulator_collision_reference_radius_m),
    "simulator_collision_extra_clearance_m": (
        float(robot_radius_m)
        - float(simulator_collision_reference_radius_m)
    ),
    "astar_collision_feedback_semantics": (
        "target_local_runtime_contact_center_cell_clear_on_target_terminal"
    ),
    "exact_tvars_target_nearby_recovery_enabled": False,
    "translation_step_scale": float(translation_step_scale),
    "translation_command_scale": float(translation_command_scale),
    "base_max_vx_mps": float(base_max_vx_mps),
    "base_max_vy_mps": float(base_max_vy_mps),
    "max_vx_mps": float(max_vx_mps),
    "max_vy_mps": float(max_vy_mps),
    "control_dt": float(control_dt),
    "turn_step_deg": float(turn_step_deg),
    "max_wz_radps": float(max_wz_radps),
    "forward_heading_tolerance_deg": float(forward_heading_tolerance_deg),
    "motion_controller_semantics": "forward_arc_with_bounded_in_place_turn",
    "max_noncollision_consecutive_rotation_only_steps": 6,
    "max_collision_recovery_consecutive_rotation_only_steps": 18,
    "max_consecutive_zero_motion_steps": 6,
    "max_tvars_empty_lookahead_replans": 3,
    "astar_collision_overlay_policy": "observed_contact_center_cell_only",
    "scene_counts": scene_counts,
    "total_scene_count": sum(scene_counts.values()),
    "voxroom_config": config_path,
    "voxroom_config_sha256": config_sha,
    "checkpoint_path": checkpoint_path,
    "checkpoint_sha256": checkpoint_sha,
    "checkpoint_keep_threshold": 0.5,
    "checkpoint_input_context": "vertical",
    "door_seed_raw_seed_source": "voxroom_tvars_vertical_union",
    "door_seed_tvars_input": "voxel_vertical_free_wall_unknown",
    "door_seed_union_scope": "episode_history",
    "active_room_commit": active_commit,
    "interioragent_episode_list": interior_list,
    "grscene_episode_list": grscene_list,
    "fallback_policy": "fail_scene_no_model_or_algorithm_fallback",
}
path = Path(output_path)
temporary = path.with_name("." + path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
  chmod 0444 "$CONTRACT_PATH"
}

write_current() {
  local dataset="$1" scene="$2" attempt="$3" phase="$4" run_dir="$5"
  (
  flock -x 8
  python3 - "$CURRENT_PATH" "$ACTIVE_DIR" "$dataset" "$scene" "$attempt" "$phase" "$run_dir" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
active_dir = Path(sys.argv[2])
payload = {
    "dataset": sys.argv[3],
    "scene": sys.argv[4],
    "attempt": int(sys.argv[5]),
    "phase": sys.argv[6],
    "run_dir": sys.argv[7],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name("." + path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
active_dir.mkdir(parents=True, exist_ok=True)
active_path = active_dir / f"{payload['dataset']}__{payload['scene']}.json"
if payload["phase"] == "running":
    active_temporary = active_path.with_name("." + active_path.name + ".tmp")
    active_temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    active_temporary.replace(active_path)
else:
    active_path.unlink(missing_ok=True)
PY
  ) 8>"$STATE_LOCK_PATH"
}

append_status() {
  local dataset="$1" scene="$2" attempt="$3" status="$4" exit_code="$5"
  local run_dir="$6" log_path="$7" started="$8" finished="$9" reason="${10}"
  (
  flock -x 8
  python3 - "$STATUS_PATH" "$SUMMARY_PATH" "$CONTRACT_PATH" "$dataset" "$scene" \
    "$attempt" "$status" "$exit_code" "$run_dir" "$log_path" "$started" "$finished" "$reason" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

(
    status_path,
    summary_path,
    contract_path,
    dataset,
    scene,
    attempt,
    status,
    exit_code,
    run_dir,
    log_path,
    started,
    finished,
    reason,
) = sys.argv[1:]
row = {
    "dataset": dataset,
    "scene": scene,
    "attempt": int(attempt),
    "status": status,
    "exit_code": int(exit_code),
    "run_dir": run_dir,
    "log_path": log_path,
    "started_at_unix": int(started),
    "finished_at_unix": int(finished),
    "elapsed_seconds": int(finished) - int(started),
    "reason": reason,
    "recorded_at": datetime.now(timezone.utc).isoformat(),
}
status_file = Path(status_path)
with status_file.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
rows = [
    json.loads(line)
    for line in status_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
manifest_dir = status_file.parent / "manifests"
contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
selected_datasets = set(contract["execution_order"])
valid_scene_keys = set()
if "habitat" in selected_datasets:
    for scene in (manifest_dir / "habitat_scenes.txt").read_text(encoding="utf-8").splitlines():
        if scene.strip():
            valid_scene_keys.add(("habitat", scene.strip()))
for dataset_name, file_name in (
    ("interioragent", "interioragent_episodes.txt"),
    ("grscene", "grscene_episodes.txt"),
):
    if dataset_name not in selected_datasets:
        continue
    for episode_path in (manifest_dir / file_name).read_text(encoding="utf-8").splitlines():
        if episode_path.strip():
            valid_scene_keys.add((dataset_name, Path(episode_path.strip()).stem))
latest = {}
for item in rows:
    key = (item["dataset"], item["scene"])
    if key in valid_scene_keys:
        latest[key] = item
summary = {
    "schema_version": "voxroom_all_datasets_roomseg_eval_summary_v1",
    "batch_tag": contract["batch_tag"],
    "expected_scene_count": contract["total_scene_count"],
    "recorded_scene_count": len(latest),
    "ignored_invalid_record_count": sum(
        (item["dataset"], item["scene"]) not in valid_scene_keys
        for item in rows
    ),
    "completed": sum(item["status"] == "completed" for item in latest.values()),
    "failed_or_exhausted": sum(item["status"] in {"failed", "exhausted"} for item in latest.values()),
    "latest": sorted(latest.values(), key=lambda item: (item["dataset"], item["scene"])),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
summary_file = Path(summary_path)
temporary = summary_file.with_name("." + summary_file.name + ".tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(summary_file)
PY
  ) 8>"$STATE_LOCK_PATH"
}

verify_run() {
  local dataset="$1" run_dir="$2"
  python3 - "$dataset" "$run_dir" "$MAX_STEPS" "$EXPECTED_CHECKPOINT_SHA256" "$LIVE_BASELINE_POLICY_CONTROL" "$NAVIGATION_PLANNER_BACKEND" "$TRANSLATION_STEP_SCALE" "$TRANSLATION_COMMAND_SCALE" "$MAX_VX_MPS" "$MAX_VY_MPS" "$CONTROL_DT" "$TURN_STEP_DEG" "$MAX_WZ_RADPS" "$FORWARD_HEADING_TOLERANCE_DEG" "$ROBOT_RADIUS_M" "$RUNTIME_PLANNING_CLEARANCE_M" "$SIMULATOR_COLLISION_REFERENCE_RADIUS_M" "$ASTAR_CLEARANCE_DESIRED_M" "$ASTAR_CLEARANCE_HARD_MIN_M" "$LOOKAHEAD_MIN_CLEARANCE_M" "$GUARD_MIN_CLEARANCE_M" "$ASTAR_CLEARANCE_WEIGHT" "$REPLAN_EVERY_STEPS" "$MAX_UNEXPLORED_COMPONENT_AREA_M2" <<'PY'
import math
import json
from pathlib import Path
import sys
import numpy as np

dataset = sys.argv[1]
run_dir = Path(sys.argv[2])
max_steps = int(sys.argv[3])
checkpoint_sha = sys.argv[4]
policy_control = sys.argv[5]
navigation_planner_backend = sys.argv[6]
translation_step_scale = float(sys.argv[7])
translation_command_scale = float(sys.argv[8])
expected_max_vx_mps = float(sys.argv[9])
expected_max_vy_mps = float(sys.argv[10])
control_dt = float(sys.argv[11])
turn_step_deg = float(sys.argv[12])
expected_max_wz_radps = float(sys.argv[13])
forward_heading_tolerance_deg = float(sys.argv[14])
robot_radius_m = float(sys.argv[15])
runtime_planning_clearance_m = float(sys.argv[16])
simulator_collision_reference_radius_m = float(sys.argv[17])
astar_clearance_desired_m = float(sys.argv[18])
astar_clearance_hard_min_m = float(sys.argv[19])
lookahead_min_clearance_m = float(sys.argv[20])
guard_min_clearance_m = float(sys.argv[21])
astar_clearance_weight = float(sys.argv[22])
replan_every_steps = int(sys.argv[23])
max_unexplored_component_area_m2 = float(sys.argv[24])

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def verify_manifest(path):
    manifest = load_json(path)
    events = manifest.get("events", [])
    if not events or events[-1].get("event_id") != "final" or events[-1].get("status") != "complete":
        raise ValueError("coverage manifest has no complete final event")
    for event in events:
        if event.get("status") != "complete":
            raise ValueError(f"incomplete coverage event: {event.get('event_id')}")
        artifacts = event.get("artifacts", {})
        for key in ("voxroom_snapshot_npz", "tvars_original_snapshot_npz"):
            artifact = artifacts.get(key)
            if not artifact or not Path(artifact).is_file():
                raise ValueError(f"missing {key} for {event.get('event_id')}")
    return manifest

if dataset == "habitat":
    result = load_json(run_dir / "result.json")
    if result.get("status") != "completed":
        raise ValueError("Habitat episode did not complete")
    if result.get("roomseg_coverage_eval") is not True:
        raise ValueError("Habitat result is not a coverage evaluation")
    if int(result.get("requested_max_episode_steps", -1)) != max_steps:
        raise ValueError("Habitat result used a different maximum step count")
    worker = load_json(run_dir / "voxroom" / "result.json")
    if worker.get("status") != "completed":
        raise ValueError("Habitat VoxRoom worker did not complete")
    if worker.get("roomseg_coverage_eval") is not True:
        raise ValueError("Habitat VoxRoom worker is not a coverage evaluation")
    if worker.get("door_seed_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Habitat worker used a different checkpoint")
    if int(worker.get("door_seed_model_fallback_count", -1)) != 0:
        raise ValueError("Habitat worker used a model fallback")
    verify_manifest(run_dir / "voxroom" / "roomseg_coverage_eval" / "manifest.json")
    for relative_path in (
        "room_mask_final.png",
        "visualization_final.png",
        "room_labels_final.npz",
        "voxroom/visualization_final.png",
    ):
        artifact = run_dir / relative_path
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"missing Habitat final artifact: {relative_path}")
else:
    lines = [line for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Isaac results.jsonl is empty")
    result = json.loads(lines[-1])
    if result.get("roomseg_coverage_eval") is not True:
        raise ValueError("Isaac result is not a coverage evaluation")
    if int(result.get("steps", max_steps + 1)) > max_steps:
        raise ValueError("Isaac result exceeded max steps")
    if result.get("fallbacks_used"):
        raise ValueError("Isaac result reports a fallback")
    if result.get("door_seed_raw_seed_source") != "voxroom_tvars_vertical_union":
        raise ValueError("Isaac result did not use the VoxRoom+TVARS raw-seed union")
    if result.get("door_seed_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Isaac result used a different door-seed checkpoint")
    if int(result.get("door_seed_model_fallback_count", -1)) != 0:
        raise ValueError("Isaac result used a learned door-seed fallback")
    if int(result.get("door_seed_raw_seed_union_decision_count", 0)) <= 0:
        raise ValueError("Isaac result never formed a VoxRoom+TVARS raw-seed union")
    if result.get("live_baseline_policy_control") != policy_control:
        raise ValueError("Isaac result used a different TVARS policy control")
    if result.get("planner") != navigation_planner_backend:
        raise ValueError("Isaac result used a different planner argument")
    if result.get("navigation_planner_backend") != "clearance_astar":
        raise ValueError("Isaac result did not use clearance A*")
    if result.get("astar_path_planner_executed") is not True:
        raise ValueError("Isaac result did not execute A* path planning")
    if result.get("planner_fallback_used") is not False:
        raise ValueError("Isaac result reports a planner fallback")
    if result.get("simulator_collision_guard_enabled") is not True:
        raise ValueError("Isaac result disabled the execution-only simulator collision guard")
    if result.get("simulator_collision_guard_source") != "preprocessed_usd_collision_navigable_execution_only":
        raise ValueError("Isaac result used an unexpected static collision source")
    episode_contract = dict(result.get("episode_static_execution_contract") or {})
    if (
        episode_contract.get("validated") is not True
        or episode_contract.get("contract") != "strict_static_execution_contract"
        or not math.isclose(
            float(episode_contract.get("required_clearance_m", -1.0)),
            robot_radius_m - simulator_collision_reference_radius_m,
        )
        or float(episode_contract.get("start_clearance_m", -1.0))
        < float(episode_contract.get("required_clearance_m", 0.0)) - 1.0e-9
    ):
        raise ValueError(
            "Isaac episode did not satisfy the strict static execution contract"
        )
    if result.get("simulator_collision_guard_object_clearance_enabled") is not False:
        raise ValueError("Isaac result unexpectedly enabled bbox object clearance")
    if int(result.get("simulator_collision_guard_blocked_steps", -1)) < 0:
        raise ValueError("Isaac result omitted static collision-guard blocking count")
    if int(result.get("simulator_collision_guard_clipped_steps", -1)) < 0:
        raise ValueError("Isaac result omitted static collision-guard clipping count")
    if int(result.get("simulator_collision_guard_violation_count", -1)) != 0:
        raise ValueError("Isaac result reports a simulator collision guard violation")
    if result.get("online_astar_clearance_cost_enabled") is not True:
        raise ValueError("Isaac result disabled A* clearance cost")
    if not math.isclose(
        float(result.get("online_astar_clearance_desired_m", -1.0)),
        astar_clearance_desired_m,
    ):
        raise ValueError("Isaac result used the wrong desired A* clearance")
    if not math.isclose(
        float(result.get("online_astar_clearance_hard_min_m", -1.0)),
        astar_clearance_hard_min_m,
    ):
        raise ValueError("Isaac result used the wrong hard A* clearance")
    if result.get("online_astar_static_execution_constraint_enabled") is not False:
        raise ValueError("Isaac A* leaked the posterior static execution map")
    if int(result.get("online_astar_static_execution_safe_cells", -1)) != 0:
        raise ValueError("Isaac A* reported posterior static execution cells")
    if result.get("online_astar_map_source") != "voxroom_online_plus_observed_collision_feedback":
        raise ValueError("Isaac A* source contract mismatch")
    if result.get("online_astar_uses_posterior_static_map") is not False:
        raise ValueError("Isaac A* posterior-map contract mismatch")
    if result.get("static_nearfield_map") is not False:
        raise ValueError("Isaac mapper used the posterior static-nearfield map")
    if result.get("mapping_source") != "depth_ray_online":
        raise ValueError("Isaac mapper source is not online depth-only")
    if not math.isclose(
        float(result.get("online_lookahead_min_clearance_m_effective", -1.0)),
        lookahead_min_clearance_m,
    ):
        raise ValueError("Isaac result used the wrong lookahead clearance")
    if not math.isclose(
        float(result.get("online_guard_min_clearance_m_effective", -1.0)),
        guard_min_clearance_m,
    ):
        raise ValueError("Isaac result used the wrong execution-guard clearance")
    if not math.isclose(
        float(result.get("online_astar_clearance_weight", -1.0)),
        astar_clearance_weight,
    ):
        raise ValueError("Isaac result used the wrong A* clearance weight")
    if int(result.get("online_astar_replan_every_steps", -1)) != replan_every_steps:
        raise ValueError("Isaac result used the wrong A* replan interval")
    if int(result.get("metric_pose_invalid_steps", -1)) != 0:
        raise ValueError("Isaac result left the static simulator navigable map")
    collision_feedback_semantics = result.get("original_collision_feedback_semantics")
    if collision_feedback_semantics not in {
        "target_local_runtime_contact_center_cell_clear_on_target_terminal",
        "upstream_collision_map_preserve_target_and_replan",
    }:
        raise ValueError("Isaac result used unexpected A* collision feedback")
    if int(result.get("original_astar_recovery_count", -1)) != 0:
        raise ValueError("Isaac strict TVARS run used nearby-target recovery")
    if int(result.get("original_astar_collision_feedback_count", -1)) < 0:
        raise ValueError("Isaac result omitted A* collision-feedback count")
    if int(result.get("original_astar_collision_overlay_cells", -1)) < 0:
        raise ValueError("Isaac result omitted A* collision-overlay size")
    if int(result.get("original_fmm_plan_count", -1)) != 0:
        raise ValueError("Isaac result unexpectedly executed FMM")
    steps = int(result.get("steps", -1))
    stop_reason = str(result.get("stop_reason", "") or "")
    failure_reason = result.get("failure_reason")
    reached_step_limit = (
        steps == max_steps
        and stop_reason == "max_control_steps"
        and failure_reason == "max_control_steps"
    )
    policy_completed = (
        result.get("stop_called") is True
        and result.get("policy_stop_confirmed") is True
        and result.get("exploration_complete") is True
        and failure_reason in {None, ""}
        and bool(
            dict(result.get("original_policy_final_decision") or {}).get(
                "stop", False
            )
        )
    )
    if not (reached_step_limit or policy_completed):
        raise ValueError(
            "Isaac scene terminated before TVARS completion or max steps: "
            f"steps={steps} stop_reason={stop_reason!r} "
            f"failure_reason={failure_reason!r}"
    )
    if policy_completed:
        final_policy_decision = dict(result.get("original_policy_final_decision") or {})
        local_frontier_count = int(
            final_policy_decision.get("frontier_count", -1)
        )
        global_component_count = int(
            final_policy_decision.get("runtime_global_frontier_component_count", -1)
        )
        global_unreachable_excluded_count = int(
            final_policy_decision.get(
                "runtime_global_frontier_astar_unreachable_excluded_count", -1
            )
        )
        global_selection_candidate_count = int(
            final_policy_decision.get(
                "runtime_global_frontier_selection_candidate_count", -1
            )
        )
        all_remaining_global_frontiers_astar_unreachable = (
            global_component_count > 0
            and global_unreachable_excluded_count >= global_component_count
            and global_selection_candidate_count == 0
            and final_policy_decision.get("reason")
            == "original_topomap_all_remaining_global_frontiers_astar_unreachable"
        )
        if int(result.get("original_terminal_frontier_target_count", -1)) != 0:
            raise ValueError("Isaac TVARS enabled the non-source terminal-frontier registry")
        if final_policy_decision.get(
            "runtime_reached_global_frontier_registry_used_for_exclusion"
        ) is not False:
            raise ValueError(
                "Isaac TVARS used reached-frontier history to hide a live Vertical Free frontier"
            )
        if local_frontier_count != 0:
            raise ValueError(
                "Isaac TVARS stopped while live frontiers remained: "
                f"frontier_count={local_frontier_count}"
            )
        if (
            global_component_count != 0
            and not all_remaining_global_frontiers_astar_unreachable
        ):
            raise ValueError(
                "Isaac TVARS stopped while global VoxRoom frontiers remained: "
                f"component_count={final_policy_decision.get('runtime_global_frontier_component_count')}"
            )
        if (
            int(result.get("runtime_global_frontier_reachable_cells", -1)) != 0
            and not all_remaining_global_frontiers_astar_unreachable
        ):
            raise ValueError(
                "Isaac TVARS stopped while globally reachable frontier cells remained"
            )
    if not math.isclose(float(result.get("max_vx_mps", -1.0)), expected_max_vx_mps):
        raise ValueError("Isaac result used a different maximum x velocity")
    if not math.isclose(float(result.get("max_vy_mps", -1.0)), expected_max_vy_mps):
        raise ValueError("Isaac result used a different maximum y velocity")
    if result.get("motion_controller") != "forward_only":
        raise ValueError("Isaac result did not use the forward-only controller")
    if result.get("motion_controller_semantics") != (
        "forward_arc_with_bounded_in_place_turn"
    ):
        raise ValueError("Isaac result used unexpected forward-only semantics")
    if result.get("control_mode") != "kinematic_forward_only":
        raise ValueError("Isaac result reported a non-forward-only control mode")
    if not math.isclose(
        float(result.get("max_wz_radps", -1.0)),
        expected_max_wz_radps,
    ):
        raise ValueError("Isaac result used a different maximum angular velocity")
    if not math.isclose(
        float(result.get("max_turn_step_deg", -1.0)),
        turn_step_deg,
    ):
        raise ValueError("Isaac result used a different per-step turn angle")
    if not math.isclose(
        math.degrees(float(result.get("forward_heading_tolerance_rad", -1.0))),
        forward_heading_tolerance_deg,
    ):
        raise ValueError("Isaac result used a different forward heading tolerance")
    if not math.isclose(
        float(result.get("translation_command_scale", -1.0)),
        translation_command_scale,
    ):
        raise ValueError("Isaac result used a different translation command scale")
    expected_translation_kp_xy = 1.2 * translation_command_scale
    if not math.isclose(
        float(result.get("translation_kp_xy", -1.0)),
        expected_translation_kp_xy,
    ):
        raise ValueError("Isaac result used a different translation gain")
    if not math.isclose(translation_step_scale, 18.0):
        raise ValueError("batch translation step scale changed")
    if not math.isclose(translation_command_scale, 6.0):
        raise ValueError("batch translation command scale changed")
    if policy_control == "original_topology":
        trace_path = result.get("original_policy_trace")
        if not trace_path or not Path(trace_path).is_file():
            raise ValueError("Isaac result has no TVARS original policy trace")
        policy_trace_rows = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        policy_decisions = [
            row for row in policy_trace_rows
            if row.get("event") == "policy_decision"
        ]
        room_frontier_decisions = [
            row for row in policy_decisions
            if row.get("target_kind") == "room_frontier"
        ]
        if any(
            row.get("reason") != "unified_vertical_free_current_room_frontier"
            or row.get("policy_frontier_source")
            != "voxel_vertical_free_current_room"
            or row.get("tvars_room_constraint_source")
            != "persistent_original_door_lines"
            or row.get("tvars_room_selection_order")
            != "current_room_frontiers_then_original_choose_door"
            for row in room_frontier_decisions
        ):
            raise ValueError(
                "Isaac TVARS selected a room frontier outside the persistent-door current-room contract"
            )
        door_transition_decisions = [
            row for row in policy_decisions
            if row.get("target_kind") == "topology_exit_waypoint"
            and row.get("reason") == "original_choose_door"
        ]
        no_room_target_reasons = {
            "original_frontier_list_empty",
            "original_last_goal_repeat_suppressed",
            "original_all_frontiers_repeat_last_goal",
            "original_frontier_astar_unreachable_excluded",
            "original_all_frontiers_astar_unreachable_excluded",
        }
        if any(
            row.get("frontier_selection_reason") not in no_room_target_reasons
            or row.get("tvars_room_constraint_source")
            != "persistent_original_door_lines"
            for row in door_transition_decisions
        ):
            raise ValueError(
                "Isaac TVARS chose a door while a current-room Vertical Free target remained selectable"
            )
        if any(
            row.get("target_kind") == "global_voxroom_frontier"
            and row.get("reason")
            != "original_topomap_complete_global_voxroom_frontier"
            for row in policy_decisions
        ):
            raise ValueError(
                "Isaac TVARS used a global frontier before the topology stop guard"
            )
        if int(result.get("original_policy_scan_frames", 0)) < 12:
            raise ValueError("Isaac result did not execute a complete TVARS scan")
        control_trace = run_dir / "runtime_control_trace.jsonl"
        control_rows = [
            json.loads(line)
            for line in control_trace.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tvars_rows = [
            row
            for row in control_rows
            if row.get("nav_mode") == "tvars_original"
            and int(row.get("target_cells_count", 0)) > 0
        ]
        if not tvars_rows:
            raise ValueError("Isaac control trace has no TVARS A* motion rows")
        if any(row.get("path_planner_backend") != "clearance_astar" for row in tvars_rows):
            raise ValueError("Isaac control trace contains a non-A* TVARS path")
        if any(row.get("astar_path_planner_executed") is not True for row in tvars_rows):
            raise ValueError("Isaac control trace did not execute A* in TVARS mode")
        if any(bool(row.get("planner_fallback_used")) for row in tvars_rows):
            raise ValueError("Isaac control trace reports a planner fallback")
        if any(
            row.get("astar_used_clearance_cost") is not True
            or not math.isclose(
                float(row.get("astar_clearance_desired_m", -1.0)),
                astar_clearance_desired_m,
            )
            for row in tvars_rows
        ):
            raise ValueError("Isaac control trace used a different A* clearance policy")
        if any(row.get("simulator_collision_guard_enabled") is not True for row in tvars_rows):
            raise ValueError("Isaac control trace disabled the execution-only static guard")
        if any(
            row.get("simulator_collision_guard_source")
            != "preprocessed_usd_collision_navigable_execution_only"
            for row in tvars_rows
        ):
            raise ValueError("Isaac control trace used an unexpected static collision source")
        if any(
            row.get("astar_static_execution_constraint_enabled") is not False
            or int(row.get("astar_static_execution_safe_cells", -1)) != 0
            or row.get("astar_map_source")
            != "voxroom_online_plus_observed_collision_feedback"
            or row.get("astar_uses_posterior_static_map") is not False
            for row in tvars_rows
        ):
            raise ValueError("Isaac control trace leaked posterior data into A*")
        if any("blocked_by_online_guard" not in row for row in tvars_rows):
            raise ValueError("Isaac control trace omitted the upstream online guard state")
        if any(float(row.get("raw_cmd", [0.0, 0.0])[0]) < 0.0 for row in tvars_rows):
            raise ValueError("Isaac control trace commanded reverse motion")
        if any(
            not math.isclose(float(row.get("raw_cmd", [0.0, 0.0])[1]), 0.0)
            for row in tvars_rows
        ):
            raise ValueError("Isaac control trace commanded lateral motion")
        if any(float(row.get("guarded_cmd", [0.0, 0.0])[0]) < 0.0 for row in tvars_rows):
            raise ValueError("Isaac final control command moved in reverse")
        if any(
            not math.isclose(float(row.get("guarded_cmd", [0.0, 0.0])[1]), 0.0)
            for row in tvars_rows
        ):
            raise ValueError("Isaac final control command moved laterally")
        max_turn_rad = math.radians(turn_step_deg)
        if any(
            abs(float(row.get("raw_cmd", [0.0, 0.0, 0.0])[2]))
            * control_dt
            > max_turn_rad + 1.0e-9
            for row in tvars_rows
        ):
            raise ValueError("Isaac control trace exceeded one TVARS turn step")
        rotation_only_runs = []
        current_rotation_only_run = 0
        current_rotation_only_run_has_collision = False
        previous_step = None
        for row in tvars_rows:
            command = row.get("guarded_cmd", [0.0, 0.0, 0.0])
            step = int(row.get("step", -1))
            rotation_only = (
                math.hypot(float(command[0]), float(command[1])) <= 1.0e-9
                and abs(float(command[2])) > 1.0e-9
            )
            if rotation_only:
                if (
                    current_rotation_only_run > 0
                    and previous_step is not None
                    and step == previous_step + 1
                ):
                    current_rotation_only_run += 1
                else:
                    if current_rotation_only_run > 0:
                        rotation_only_runs.append(
                            (
                                current_rotation_only_run,
                                current_rotation_only_run_has_collision,
                            )
                        )
                    current_rotation_only_run = 1
                    current_rotation_only_run_has_collision = False
                current_rotation_only_run_has_collision = bool(
                    current_rotation_only_run_has_collision
                    or row.get("blocked_by_simulator_collision_guard")
                )
            else:
                if current_rotation_only_run > 0:
                    rotation_only_runs.append(
                        (
                            current_rotation_only_run,
                            current_rotation_only_run_has_collision,
                        )
                    )
                current_rotation_only_run = 0
                current_rotation_only_run_has_collision = False
            previous_step = step
        if current_rotation_only_run > 0:
            rotation_only_runs.append(
                (
                    current_rotation_only_run,
                    current_rotation_only_run_has_collision,
                )
            )
        longest_noncollision_rotation_run = max(
            (length for length, collided in rotation_only_runs if not collided),
            default=0,
        )
        longest_collision_rotation_run = max(
            (length for length, collided in rotation_only_runs if collided),
            default=0,
        )
        if longest_noncollision_rotation_run > 6:
            raise ValueError(
                "Isaac control trace rotated in place too long without collision: "
                f"steps={longest_noncollision_rotation_run}"
            )
        if longest_collision_rotation_run > 18:
            raise ValueError(
                "Isaac collision recovery rotated in place too long: "
                f"steps={longest_collision_rotation_run}"
            )
        longest_zero_motion_run = 0
        current_zero_motion_run = 0
        previous_step = None
        for row in tvars_rows:
            command = row.get("guarded_cmd", [0.0, 0.0, 0.0])
            step = int(row.get("step", -1))
            zero_motion = all(abs(float(value)) <= 1.0e-9 for value in command)
            if zero_motion:
                current_zero_motion_run = (
                    current_zero_motion_run + 1
                    if previous_step is not None and step == previous_step + 1
                    else 1
                )
                longest_zero_motion_run = max(
                    longest_zero_motion_run,
                    current_zero_motion_run,
                )
            else:
                current_zero_motion_run = 0
            previous_step = step
        if longest_zero_motion_run > 6:
            raise ValueError(
                "Isaac control trace remained at zero motion too long: "
                f"steps={longest_zero_motion_run}"
            )
        decision_rows = [
            json.loads(line)
            for line in (
                run_dir / "runtime_decision_trace.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        empty_lookahead_rows = [
            row
            for row in decision_rows
            if row.get("zero_velocity_continue") is True
            and row.get("event_type")
            in {
                "tvars_original_path_exhausted_replan",
                "original_astar_target_rejected_unreachable",
                "original_topology_exit_astar_no_path_exhausted",
                "original_room_entry_return_astar_no_path_exhausted",
                "original_global_frontier_astar_target_rejected_unreachable",
            }
        ]
        longest_empty_lookahead_run = 0
        current_empty_lookahead_run = 0
        previous_step = None
        for row in empty_lookahead_rows:
            step = int(row.get("step", -1))
            current_empty_lookahead_run = (
                current_empty_lookahead_run + 1
                if previous_step is not None and step == previous_step + 1
                else 1
            )
            longest_empty_lookahead_run = max(
                longest_empty_lookahead_run,
                current_empty_lookahead_run,
            )
            previous_step = step
        if longest_empty_lookahead_run > 3:
            raise ValueError(
                "Isaac repeatedly retried an empty TVARS lookahead: "
                f"steps={longest_empty_lookahead_run}"
            )
        if int(result.get("original_collision_map_update_count", -1)) != 0:
            raise ValueError("A* run unexpectedly wrote the FMM collision map")
        collision_feedback_rows = [
            row
            for row in tvars_rows
            if row.get("simulator_collision_feedback_semantics")
            == collision_feedback_semantics
        ]
        if any(
            row.get("simulator_collision_tvars_target_completed") is not False
            or row.get("simulator_collision_astar_path_cleared") is not True
            for row in collision_feedback_rows
        ):
            raise ValueError(
                "A* collision feedback completed a TVARS target or retained a stale path"
            )
        if collision_feedback_semantics == (
            "target_local_runtime_contact_center_cell_clear_on_target_terminal"
        ):
            if any(
                not isinstance(row.get("collision_contact_cell"), list)
                or len(row.get("collision_contact_cell")) != 2
                or int(row.get("collision_feedback_new_cells", -1)) not in {0, 1}
                or "collision_band_cells" in row
                for row in collision_feedback_rows
            ):
                raise ValueError(
                    "A* collision feedback was not limited to the observed contact cell"
                )
        else:
            expected_collision_band_width_cells = 2 * int(
                math.ceil(
                    float(result.get("robot_radius_m", -1.0))
                    / float(result.get("online_map_resolution_m", -1.0))
                    - 1.0e-9
                )
            ) + 1
            if any(
                row.get("collision_band_width_policy")
                != "robot_footprint_diameter"
                or int(row.get("collision_band_max_width_cells", -1))
                != expected_collision_band_width_cells
                or int(row.get("collision_band_width_cells", -1))
                > expected_collision_band_width_cells
                for row in collision_feedback_rows
            ):
                raise ValueError(
                    "legacy A* collision feedback exceeded the robot-footprint width"
                )
        decision_trace_text = (
            run_dir / "runtime_decision_trace.jsonl"
        ).read_text(encoding="utf-8")
        if (
            "original_astar_upstream_voxroom_recovery_started"
            in decision_trace_text
            or "follow_reachable_approach_path" in decision_trace_text
            or "original_topology_exit_no_path_waiting" in decision_trace_text
        ):
            raise ValueError(
                "strict TVARS trace contains nearby-target recovery or idle no-path waiting"
            )
        if any(
            not math.isclose(
                float(row.get("translation_command_scale", -1.0)),
                translation_command_scale,
            )
            for row in tvars_rows
        ):
            raise ValueError("Isaac control trace used a different translation command scale")
        if any(
            not math.isclose(
                float(row.get("translation_kp_xy", -1.0)),
                expected_translation_kp_xy,
            )
            for row in tvars_rows
        ):
            raise ValueError("Isaac control trace used a different translation gain")
    coverage_manifest = verify_manifest(
        run_dir / "roomseg_coverage_eval" / "manifest.json"
    )
    if dataset != "habitat":
        expected_reference_source = (
            "preprocessed_scene_same_floor_global_reachable_navigable_before_runtime_clearance"
            if dataset == "grscene"
            else "preprocessed_scene_same_floor_per_room_reachable_navigable_before_runtime_clearance"
        )
        expected_component_filter = (
            "largest_8_connected_navigable_component_global_no_room_polygons"
            if dataset == "grscene"
            else "largest_8_connected_navigable_component_per_room_polygon"
        )
        if coverage_manifest.get("reference_source") != expected_reference_source:
            raise ValueError("Isaac coverage used an unexpected reference source")
        if coverage_manifest.get("reference_component_filter_applied") is not True:
            raise ValueError("Isaac coverage did not apply the required reference filter")
        if coverage_manifest.get("reference_component_filter") != expected_component_filter:
            raise ValueError("Isaac coverage used an unexpected reference component filter")
        same_floor_cells = int(
            coverage_manifest.get("reference_same_floor_navigable_cells", 0)
        )
        if same_floor_cells <= 0:
            raise ValueError("Isaac coverage has no same-floor navigable reference cells")
        final_event = coverage_manifest["events"][-1]
        if int(final_event.get("step", -1)) != int(result.get("steps", -2)):
            raise ValueError(
                "Isaac final coverage snapshot does not match result steps"
            )
        if int(final_event.get("total_explorable_cells", -1)) != same_floor_cells:
            raise ValueError("Isaac final coverage denominator does not match the same-floor reference")
        if policy_completed:
            final_snapshot = Path(
                final_event["artifacts"]["voxroom_snapshot_npz"]
            )
            with np.load(final_snapshot, allow_pickle=False) as data:
                reference = np.asarray(
                    data["roomseg_eval_reference_explorable_mask"], dtype=bool
                )
                explored = np.asarray(
                    data["roomseg_eval_explored_reference_mask"], dtype=bool
                )
            if reference.shape != explored.shape:
                raise ValueError("final coverage masks have different shapes")
            missed = reference & ~explored
            visited = np.zeros_like(missed, dtype=bool)
            largest_component_cells = 0
            for start_row, start_col in np.argwhere(missed):
                row = int(start_row)
                col = int(start_col)
                if visited[row, col]:
                    continue
                visited[row, col] = True
                stack = [(row, col)]
                component_cells = 0
                while stack:
                    current_row, current_col = stack.pop()
                    component_cells += 1
                    for delta_row in (-1, 0, 1):
                        for delta_col in (-1, 0, 1):
                            if delta_row == 0 and delta_col == 0:
                                continue
                            next_row = current_row + delta_row
                            next_col = current_col + delta_col
                            if not (
                                0 <= next_row < missed.shape[0]
                                and 0 <= next_col < missed.shape[1]
                            ):
                                continue
                            if missed[next_row, next_col] and not visited[next_row, next_col]:
                                visited[next_row, next_col] = True
                                stack.append((next_row, next_col))
                largest_component_cells = max(
                    largest_component_cells, component_cells
                )
            resolution_m = float(coverage_manifest["resolution_m"])
            largest_component_area_m2 = (
                largest_component_cells * resolution_m * resolution_m
            )
            if largest_component_area_m2 > max_unexplored_component_area_m2:
                raise ValueError(
                    "Isaac TVARS policy stopped with a large unexplored reference component: "
                    f"cells={largest_component_cells} "
                    f"area_m2={largest_component_area_m2:.3f} "
                    f"limit_m2={max_unexplored_component_area_m2:.3f}"
                )
PY
}

wait_for_resources() {
  local dataset="$1" scene="$2"
  local available_ram_kb available_ram_mb gpu_line gpu_total_mb gpu_used_mb gpu_free_mb
  while true; do
    available_ram_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    available_ram_mb=$((available_ram_kb / 1024))
    gpu_line="$(nvidia-smi --id=0 --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n 1)" || return 1
    IFS=',' read -r gpu_total_mb gpu_used_mb <<<"$gpu_line"
    gpu_total_mb="${gpu_total_mb// /}"
    gpu_used_mb="${gpu_used_mb// /}"
    gpu_free_mb=$((gpu_total_mb - gpu_used_mb))
    if [[ "$available_ram_mb" -ge "$MIN_AVAILABLE_RAM_MB" && "$gpu_free_mb" -ge "$MIN_FREE_GPU_MB" ]]; then
      echo "[all-datasets] resources ready dataset=$dataset scene=$scene ram_mb=$available_ram_mb gpu_free_mb=$gpu_free_mb"
      return 0
    fi
    echo "[all-datasets] waiting dataset=$dataset scene=$scene ram_mb=$available_ram_mb/$MIN_AVAILABLE_RAM_MB gpu_free_mb=$gpu_free_mb/$MIN_FREE_GPU_MB"
    sleep "$RESOURCE_POLL_SECONDS"
  done
}

run_habitat_scene() {
  local scene="$1" run_dir="$2" _requested_prep_dir="$3" log_path="$4" run_id prep_dir
  run_id="$(< /proc/sys/kernel/random/uuid)"
  prep_dir="$ACTIVE_ROOM_ROOT/data/gibson-visual-runs/$run_id"
  (
    flock 9
    "$ACTIVE_ROOM_PYTHON" "$ACTIVE_ROOM_ROOT/scripts/prepare_gibson_visual.py" \
      --repository-root "$ACTIVE_ROOM_ROOT" \
      --scene-id "$scene" \
      --episode-index 0 \
      --output-dataset "$prep_dir/input_dataset.json.gz" \
      --manifest-path "$prep_dir/input_manifest.json" || exit $?
    env \
      TASK_CONFIG=tasks/pointnav_gibson_visual.yaml \
      SPLIT=val \
      REQUIRE_TOPOLOGY_TRANSITION=0 \
      PAD_EPISODE_TO_MAX_STEPS=0 \
      ALLOW_EARLY_COMPLETION=1 \
      RUN_CONTEXT_REQUIRED=1 \
      RUN_CONTEXT_MANIFEST="$prep_dir/input_manifest.json" \
      RUN_CONTEXT_DATASET="$prep_dir/input_dataset.json.gz" \
      WINDOW_CHECKPOINT_EVERY_STEPS=100 \
      VISUALIZATION_FRAME_EVERY_STEPS=5 \
      VOXROOM_SIDECAR=1 \
      VOXROOM_ROOT="$VOXROOM_ROOT" \
      VOXROOM_CONFIG="$VOXROOM_CONFIG" \
      VOXROOM_ROOMSEG_EVERY_STEPS=50 \
      VOXROOM_VISUALIZATION_EVERY_STEPS=5 \
      ROOMSEG_COVERAGE_EVAL=1 \
      ROOMSEG_COVERAGE_MILESTONES="$COVERAGE_MILESTONES" \
      MAX_EPISODE_STEPS="$MAX_STEPS" \
      RUN_ID="$run_id" \
      RUN_TAG="${BATCH_TAG}_${scene}" \
      RUN_DIR="$run_dir" \
      "$ACTIVE_ROOM_ROOT/scripts/run_habitat_test.sh"
  ) 9>"$ACTIVE_ROOM_ROOT/data/.gibson_visual.lock" >"$log_path" 2>&1 </dev/null
}

run_isaac_scene() {
  local episode_file="$1" run_dir="$2" log_path="$3"
  if [[ "$episode_file" != /* ]]; then
    episode_file="$VOXROOM_ROOT/$episode_file"
  fi
  (
    cd "$VOXROOM_ROOT" || exit 1
    env \
      ACTIVE_ROOM_SEG_EXPECTED_COMMIT="$EXPECTED_ACTIVE_ROOM_COMMIT" \
      CONFIG="$VOXROOM_CONFIG" \
      RUN_DIR="$run_dir" \
      MAX_CONTROL_STEPS="$MAX_STEPS" \
      COVERAGE_MILESTONES="$COVERAGE_MILESTONES" \
      LIVE_BASELINE_POLICY_CONTROL="$LIVE_BASELINE_POLICY_CONTROL" \
      NAVIGATION_PLANNER_BACKEND="$NAVIGATION_PLANNER_BACKEND" \
      TRANSLATION_STEP_SCALE="$TRANSLATION_STEP_SCALE" \
      TRANSLATION_COMMAND_SCALE="$TRANSLATION_COMMAND_SCALE" \
      BASE_MAX_VX_MPS="$BASE_MAX_VX_MPS" \
      BASE_MAX_VY_MPS="$BASE_MAX_VY_MPS" \
      MAX_VX_MPS="$MAX_VX_MPS" \
      MAX_VY_MPS="$MAX_VY_MPS" \
      CONTROL_DT="$CONTROL_DT" \
      TURN_STEP_DEG="$TURN_STEP_DEG" \
      MAX_WZ_RADPS="$MAX_WZ_RADPS" \
      FORWARD_HEADING_TOLERANCE_DEG="$FORWARD_HEADING_TOLERANCE_DEG" \
      ROBOT_RADIUS_M="$ROBOT_RADIUS_M" \
      RUNTIME_PLANNING_CLEARANCE_M="$RUNTIME_PLANNING_CLEARANCE_M" \
      SIMULATOR_COLLISION_REFERENCE_RADIUS_M="$SIMULATOR_COLLISION_REFERENCE_RADIUS_M" \
      ASTAR_CLEARANCE_DESIRED_M="$ASTAR_CLEARANCE_DESIRED_M" \
      ASTAR_CLEARANCE_HARD_MIN_M="$ASTAR_CLEARANCE_HARD_MIN_M" \
      LOOKAHEAD_MIN_CLEARANCE_M="$LOOKAHEAD_MIN_CLEARANCE_M" \
      GUARD_MIN_CLEARANCE_M="$GUARD_MIN_CLEARANCE_M" \
      ASTAR_CLEARANCE_WEIGHT="$ASTAR_CLEARANCE_WEIGHT" \
      REPLAN_EVERY_STEPS="$REPLAN_EVERY_STEPS" \
      HEADLESS_FLAG="$ISAAC_HEADLESS_FLAG" \
      VOXROOM_NUMBA_THREADS=28 \
      "$VOXROOM_ROOT/scripts/run_roomseg_coverage_eval_isaac.sh" "$episode_file"
  ) >"$log_path" 2>&1 </dev/null
}

process_scene() {
  local dataset="$1" scene="$2" input_ref="$3"
  local scene_root="$BATCH_ROOT/$dataset/$scene"
  local existing attempt_count attempt attempt_marker blocked_marker run_dir prep_dir log_path validation_log
  local started finished status reason exit_code validation_error
  mkdir -p "$scene_root" "$LOG_DIR/$dataset"

  blocked_marker="$scene_root/validation_failure.blocked"
  if [[ -s "$blocked_marker" ]]; then
    echo "[all-datasets] blocked dataset=$dataset scene=$scene marker=$blocked_marker"
    write_current "$dataset" "$scene" 0 blocked "$scene_root"
    return 5
  fi

  while IFS= read -r existing; do
    [[ -z "$existing" ]] && continue
    if verify_run "$dataset" "$existing" >/dev/null 2>&1; then
      echo "[all-datasets] already completed dataset=$dataset scene=$scene run=$existing"
      return 0
    fi
  done < <(find "$scene_root" -mindepth 1 -maxdepth 1 -type d -name 'attempt_*' | sort)

  attempt_count="$(find "$scene_root" -mindepth 1 -maxdepth 1 -type f -name 'attempt_*.started' | wc -l)"
  if [[ "$attempt_count" -ge "$MAX_ATTEMPTS" ]]; then
    started="$(date +%s)"
    append_status "$dataset" "$scene" "$attempt_count" exhausted 1 "$scene_root" "" "$started" "$started" max_attempts_exhausted
    echo "[all-datasets] exhausted dataset=$dataset scene=$scene attempts=$attempt_count"
    write_current "$dataset" "$scene" "$attempt_count" blocked "$scene_root"
    return 5
  fi

  attempt=$((attempt_count + 1))
  run_dir="$scene_root/attempt_$(printf '%02d' "$attempt")"
  attempt_marker="$scene_root/attempt_$(printf '%02d' "$attempt").started"
  prep_dir="$PREPARED_DIR/$dataset/$scene/attempt_$(printf '%02d' "$attempt")"
  log_path="$LOG_DIR/$dataset/${scene}_attempt_$(printf '%02d' "$attempt").log"
  validation_log="$run_dir/validation.log"
  printf '%s\n' "$(date --iso-8601=seconds)" >"$attempt_marker"
  started="$(date +%s)"
  write_current "$dataset" "$scene" "$attempt" running "$run_dir"
  echo "[all-datasets] starting dataset=$dataset scene=$scene attempt=$attempt max_steps=$MAX_STEPS"

  if ! check_fixed_contract; then
    finished="$(date +%s)"
    append_status "$dataset" "$scene" "$attempt" blocked 2 "$run_dir" "$log_path" "$started" "$finished" contract_drift
    write_current "$dataset" "$scene" "$attempt" blocked "$run_dir"
    return 2
  fi
  if ! wait_for_resources "$dataset" "$scene"; then
    finished="$(date +%s)"
    append_status "$dataset" "$scene" "$attempt" failed 3 "$run_dir" "$log_path" "$started" "$finished" resource_query_failed
    return 0
  fi

  if [[ "$dataset" == "habitat" ]]; then
    run_habitat_scene "$scene" "$run_dir" "$prep_dir" "$log_path"
    exit_code=$?
  else
    run_isaac_scene "$input_ref" "$run_dir" "$log_path"
    exit_code=$?
  fi
  if [[ -s "$run_dir/runtime_exception.json" && "$exit_code" -eq 0 ]]; then
    exit_code=70
  fi
  finished="$(date +%s)"
  if verify_run "$dataset" "$run_dir" >"$validation_log" 2>&1; then
    status=completed
    if [[ "$exit_code" -eq 0 ]]; then
      reason=validated_artifact_closure
    else
      reason=core_artifact_closure_postrun_ui_validation_failed
    fi
  else
    status=blocked
    if [[ -s "$run_dir/runtime_exception.json" ]]; then
      reason=runtime_exception_before_artifact_closure
    else
      reason=deterministic_artifact_validation_failed
    fi
    validation_error="$(tail -n 1 "$validation_log" 2>/dev/null || true)"
    printf '%s\n' \
      "dataset=$dataset" \
      "scene=$scene" \
      "attempt=$attempt" \
      "run_dir=$run_dir" \
      "validation_log=$validation_log" \
      "validation_error=$validation_error" >"$blocked_marker"
  fi
  append_status "$dataset" "$scene" "$attempt" "$status" "$exit_code" "$run_dir" "$log_path" "$started" "$finished" "$reason"
  write_current "$dataset" "$scene" "$attempt" "$status" "$run_dir"
  echo "[all-datasets] finished dataset=$dataset scene=$scene attempt=$attempt status=$status exit=$exit_code"

  if [[ "$status" == "blocked" ]]; then
    echo "[all-datasets] validation blocked dataset=$dataset scene=$scene error=$validation_error"
    return 5
  fi
}

run_habitat_dataset() {
  local scene rc
  while IFS= read -r scene <&3; do
    [[ -z "$scene" ]] && continue
    process_scene habitat "$scene" "$scene"
    rc=$?
    if [[ "$rc" -eq 5 && "$CONTINUE_ON_VALIDATION_BLOCKED" -eq 1 ]]; then
      echo "[all-datasets] continuing after recorded validation block dataset=habitat scene=$scene"
      continue
    fi
    [[ "$rc" -eq 0 ]] || return "$rc"
  done 3<"$MANIFEST_DIR/habitat_scenes.txt"
}

run_isaac_dataset() {
  local dataset="$1" list_path="$2" episode_file scene pid completed_pid rc failed_rc=0
  local -a running_pids=()
  while IFS= read -r episode_file <&3; do
    [[ -z "$episode_file" ]] && continue
    scene="$(basename "$episode_file" .jsonl)"
    while [[ "${#running_pids[@]}" -ge "$MAX_PARALLEL_SCENES" ]]; do
      completed_pid=""
      wait -n -p completed_pid "${running_pids[@]}"
      rc=$?
      if [[ -n "$completed_pid" ]]; then
        local -a remaining_pids=()
        for pid in "${running_pids[@]}"; do
          [[ "$pid" == "$completed_pid" ]] || remaining_pids+=("$pid")
        done
        running_pids=("${remaining_pids[@]}")
      fi
      if [[ "$rc" -eq 5 && "$CONTINUE_ON_VALIDATION_BLOCKED" -eq 1 ]]; then
        echo "[all-datasets] continuing after recorded validation block dataset=$dataset"
        rc=0
      fi
      if [[ "$rc" -ne 0 && "$failed_rc" -eq 0 ]]; then
        failed_rc="$rc"
      fi
      if [[ "$failed_rc" -ne 0 ]]; then
        break
      fi
    done
    if [[ "$failed_rc" -ne 0 ]]; then
      break
    fi
    process_scene "$dataset" "$scene" "$episode_file" &
    running_pids+=("$!")
    if [[ "${#running_pids[@]}" -lt "$MAX_PARALLEL_SCENES" && "$PARALLEL_LAUNCH_STAGGER_SECONDS" -gt 0 ]]; then
      sleep "$PARALLEL_LAUNCH_STAGGER_SECONDS"
    fi
  done 3<"$list_path"
  for pid in "${running_pids[@]}"; do
    wait "$pid"
    rc=$?
    if [[ "$rc" -eq 5 && "$CONTINUE_ON_VALIDATION_BLOCKED" -eq 1 ]]; then
      echo "[all-datasets] continuing after recorded validation block dataset=$dataset"
      rc=0
    fi
    if [[ "$rc" -ne 0 && "$failed_rc" -eq 0 ]]; then
      failed_rc="$rc"
    fi
  done
  return "$failed_rc"
}

main() {
  [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || die "MAX_STEPS must be a positive integer"
  [[ "$MAX_STEPS" -eq 7000 ]] || die "this evaluation contract requires MAX_STEPS=7000"
  [[ "$MAX_PARALLEL_SCENES" =~ ^[1-9][0-9]*$ ]] || die "MAX_PARALLEL_SCENES must be a positive integer"
  [[ "$PARALLEL_LAUNCH_STAGGER_SECONDS" =~ ^[0-9]+$ ]] || die "PARALLEL_LAUNCH_STAGGER_SECONDS must be a non-negative integer"
  [[ "$CONTINUE_ON_VALIDATION_BLOCKED" =~ ^[01]$ ]] || die "CONTINUE_ON_VALIDATION_BLOCKED must be 0 or 1"
  validate_clearance_contract || die "planner and execution clearance thresholds differ"
  validate_dataset_selection || die "invalid DATASETS selection: $DATASETS"
  require_executable "$ACTIVE_ROOM_PYTHON"
  require_executable "$VOXROOM_ROOT/scripts/run_roomseg_coverage_eval_isaac.sh"
  require_file "$VOXROOM_CONFIG"
  if dataset_enabled habitat; then
    require_executable "$ACTIVE_ROOM_ROOT/scripts/run_habitat_test.sh"
    require_file "$GIBSON_VAL_DATASET"
  fi
  if dataset_enabled interioragent; then
    require_file "$INTERIORAGENT_EPISODE_LIST"
  fi
  if dataset_enabled grscene; then
    require_file "$GRSCENE_EPISODE_LIST"
  fi

  mkdir -p "$BATCH_ROOT" "$MANIFEST_DIR" "$LOG_DIR" "$PREPARED_DIR" "$ACTIVE_DIR"
  exec > >(tee -a "$BATCH_ROOT/controller.log") 2>&1
  echo "[all-datasets] batch=$BATCH_TAG root=$BATCH_ROOT datasets=$DATASETS max_steps=$MAX_STEPS parallel=$MAX_PARALLEL_SCENES launch_stagger_s=$PARALLEL_LAUNCH_STAGGER_SECONDS policy_control=$LIVE_BASELINE_POLICY_CONTROL planner=$NAVIGATION_PLANNER_BACKEND translation_step_scale=$TRANSLATION_STEP_SCALE translation_command_scale=$TRANSLATION_COMMAND_SCALE max_vx_mps=$MAX_VX_MPS max_vy_mps=$MAX_VY_MPS turn_step_deg=$TURN_STEP_DEG forward_heading_tolerance_deg=$FORWARD_HEADING_TOLERANCE_DEG"
  prepare_scene_lists || die "scene list validation failed"
  write_source_manifest || die "could not write source manifest"
  write_contract || die "could not write batch contract"
  if ! check_fixed_contract; then
    echo "[all-datasets] blocked before launch because the fixed contract changed"
    write_current contract contract 0 blocked "$BATCH_ROOT"
    return 0
  fi

  local dataset
  for dataset in "${DATASET_ORDER[@]}"; do
    case "$dataset" in
      habitat)
        run_habitat_dataset || return $?
        ;;
      interioragent)
        if ! run_isaac_dataset interioragent "$MANIFEST_DIR/interioragent_episodes.txt"; then
          echo "[all-datasets] stopped after an InteriorAgent scene failed before result closure"
          return 0
        fi
        ;;
      grscene)
        if ! run_isaac_dataset grscene "$MANIFEST_DIR/grscene_episodes.txt"; then
          echo "[all-datasets] stopped after a GRScene scene failed before result closure"
          return 0
        fi
        ;;
    esac
  done
  if python3 - "$SUMMARY_PATH" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(
    0
    if summary.get("completed") == summary.get("expected_scene_count")
    and summary.get("failed_or_exhausted") == 0
    else 1
)
PY
  then
    write_current batch batch 0 completed "$BATCH_ROOT"
    echo "[all-datasets] all scheduled scenes completed; see $SUMMARY_PATH"
  else
    write_current batch batch 0 incomplete "$BATCH_ROOT"
    echo "[all-datasets] schedule ended with incomplete scenes; see $SUMMARY_PATH"
  fi
}

main "$@"
