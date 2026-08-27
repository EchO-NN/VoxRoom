#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:?set DATASET to interioragent or grscene}"
TASK_ROOT="${TASK_ROOT:-/media/echo/data/voxroom_door_seed_union_voxel_milestones_20260811}"
SCENE_LIST="${SCENE_LIST:?set SCENE_LIST}"
EPISODE_DIR="${EPISODE_DIR:?set EPISODE_DIR}"
COLLECTION_ROOT="${COLLECTION_ROOT:-$TASK_ROOT/collection}"
RUN_ROOT="${RUN_ROOT:-$TASK_ROOT/runs/$DATASET}"
CONFIG="${CONFIG:-configs/voxroom_online.yaml}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-7000}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
SCENE_LIMIT="${SCENE_LIMIT:-0}"
RUN_NUMBA_NUM_THREADS="${RUN_NUMBA_NUM_THREADS:-20}"
SANITIZE_DISPLAY_PRIMVARS="${SANITIZE_DISPLAY_PRIMVARS:-1}"
VOXROOM_VIZ_EVERY_STEPS="${VOXROOM_VIZ_EVERY_STEPS:-10}"
FULL_VOXEL_COVERAGE_MILESTONES="${FULL_VOXEL_COVERAGE_MILESTONES:-30,40,50,60,70,80,90}"

if [[ ! -f "$SCENE_LIST" ]]; then
  echo "[raw-seed-union-batch] missing scene list: $SCENE_LIST" >&2
  exit 2
fi
if [[ "$PARALLEL_JOBS" -lt 1 ]]; then
  echo "[raw-seed-union-batch] PARALLEL_JOBS must be positive" >&2
  exit 2
fi
mapfile -t SCENES < <(sed '/^[[:space:]]*$/d' "$SCENE_LIST")
if [[ "$SCENE_LIMIT" -gt 0 && "$SCENE_LIMIT" -lt "${#SCENES[@]}" ]]; then
  SCENES=("${SCENES[@]:0:$SCENE_LIMIT}")
fi
if [[ "${#SCENES[@]}" -eq 0 ]]; then
  echo "[raw-seed-union-batch] empty scene list: $SCENE_LIST" >&2
  exit 2
fi

mkdir -p "$COLLECTION_ROOT" "$RUN_ROOT"
exec 9>"$RUN_ROOT/.batch.lock"
if ! flock -n 9; then
  echo "[raw-seed-union-batch] another $DATASET batch owns $RUN_ROOT" >&2
  exit 2
fi

is_collection_complete() {
  local scene_id="$1"
  python3 - "$COLLECTION_ROOT" "$scene_id" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
scene_id = sys.argv[2]
for path in sorted(root.glob("*/manifest.json")):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if str(manifest.get("scene_id")) != scene_id:
        continue
    finals = [item for item in manifest.get("snapshots", []) if bool(item.get("is_final"))]
    if len(finals) != 1:
        continue
    final_path = path.parent / str(finals[0].get("path", ""))
    collection = dict(manifest.get("collection", {}) or {})
    if (
        final_path.is_file()
        and collection.get("raw_seed_source") == "voxroom_tvars_vertical_union"
        and int(collection.get("collection_every_steps", 0)) == 5
        and bool(collection.get("persistent_final_raw_seed_union", False))
        and bool(collection.get("save_full_voxel_milestones", False))
        and collection.get("full_voxel_coverage_milestones")
        == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        and bool(finals[0].get("full_voxel_snapshot", False))
    ):
        print(path.parent)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

validate_episode() {
  local scene_id="$1"
  local episode_file="$EPISODE_DIR/$scene_id.jsonl"
  python3 - "$episode_file" "$scene_id" <<'PY'
import json
import math
import sys
from pathlib import Path

episode_path = Path(sys.argv[1])
expected_scene = sys.argv[2]
episode = json.loads(next(line for line in episode_path.read_text(encoding="utf-8").splitlines() if line.strip()))
if str(episode.get("scene_id")) != expected_scene:
    raise SystemExit("scene mismatch in %s" % episode_path)
for key in ("usd_path", "rooms_json_path", "preprocessed_scene_dir"):
    path = Path(str(episode[key])).expanduser()
    if not path.exists():
        raise SystemExit("missing %s for %s: %s" % (key, expected_scene, path))
metadata = dict(episode.get("metadata", {}) or {})
expected_collision_contract = {
    "robot_radius_m": 0.15,
    "static_reference_radius_m": 0.05,
    "runtime_planning_clearance_m": 0.01,
    "static_execution_extra_clearance_m": 0.10,
}
for key, expected in expected_collision_contract.items():
    actual = metadata.get(key)
    if actual is None or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise SystemExit(
            "stale collision contract for %s: %s expected=%s actual=%s"
            % (expected_scene, key, expected, actual)
        )
PY
}

run_scene() {
  local scene_id="$1"
  local episode_file="$EPISODE_DIR/$scene_id.jsonl"
  local complete_path=""
  if complete_path="$(is_collection_complete "$scene_id")"; then
    echo "[raw-seed-union-batch] skip complete $scene_id -> $complete_path"
    return 0
  fi
  validate_episode "$scene_id" || return $?

  local scene_run_dir="$RUN_ROOT/$scene_id"
  if [[ -e "$scene_run_dir/results.jsonl" ]]; then
    scene_run_dir="$RUN_ROOT/${scene_id}_retry_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p "$scene_run_dir"
  {
    echo "dataset=$DATASET"
    echo "scene_id=$scene_id"
    echo "episode_file=$episode_file"
    echo "collection_root=$COLLECTION_ROOT"
    echo "raw_seed_source=voxroom_tvars_vertical_union"
    echo "tvars_map_source=voxel_vertical_free_wall_unknown"
    echo "raw_seed_observation_every_steps=5"
    echo "full_voxel_coverage_milestones=$FULL_VOXEL_COVERAGE_MILESTONES,final"
    echo "persistent_final_raw_seed_union=true"
    echo "frontier_selection_mode=random"
    echo "max_control_steps=$MAX_CONTROL_STEPS"
    echo "robot_radius_m=0.15"
    echo "runtime_planning_clearance_m=0.01"
    echo "static_collision_reference_radius_m=0.05"
    echo "static_collision_extra_clearance_m=0.10"
  } > "$scene_run_dir/command.txt"
  echo "[raw-seed-union-batch] running dataset=$DATASET scene=$scene_id"
  VOXROOM_NUMBA_THREADS="$RUN_NUMBA_NUM_THREADS" \
  VOXROOM_ISAAC_SANITIZE_DISPLAY_PRIMVARS="$SANITIZE_DISPLAY_PRIMVARS" \
  NUMBA_NUM_THREADS="$RUN_NUMBA_NUM_THREADS" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    scripts/collect_door_seed_training_data.sh \
      --config "$CONFIG" \
      --episode-file "$episode_file" \
      --episode-index 0 \
      --sim-backend isaac \
      --planner astar \
      --detector none \
      --segmenter none \
      --no-llm-enabled \
      --no-vllm-frontier-scoring \
      --no-vllm-image-scoring \
      --no-allow-gt-goal-fallback \
      --no-frontier-allow-near-fallback \
      --frontier-selection-mode random \
      --frontier-random-seed 0 \
      --frontier-source voxel_vertical_free \
      --room-map-mode voxel_occupancy_door_wall_v33 \
      --roomseg-backend voxel_occupancy_door_wall_v33 \
      --no-strict-benchmark \
      --explore-until-no-frontiers \
      --max-control-steps "$MAX_CONTROL_STEPS" \
      --max-vx-mps 2.70 \
      --max-vy-mps 0.0 \
      --control-dt 0.2 \
      --max-wz-radps 2.6179938779914944 \
      --forward-heading-tolerance-deg 75.0 \
      --translation-command-scale 6.0 \
      --replan-every-steps 10 \
      --simulator-collision-guard \
      --simulator-collision-reference-radius-m 0.05 \
      --no-static-nearfield-map \
      --robot-radius-m 0.15 \
      --runtime-planning-clearance-m 0.01 \
      --astar-clearance-cost-enabled \
      --astar-clearance-desired-m 0.20 \
      --astar-clearance-hard-min-m 0.05 \
      --astar-clearance-weight 6.0 \
      --astar-goal-min-clearance-m 0.08 \
      --lookahead-min-clearance-m 0.05 \
      --guard-min-clearance-m 0.05 \
      --headless \
      --read-depth \
      --camera-annotator-device cpu \
      --isaac-width 256 \
      --isaac-height 256 \
      --camera-hfov-deg 90 \
      --camera-mast-height-m 1.25 \
      --panorama-steps 12 \
      --voxroom-viz \
      --voxroom-viz-every-steps "$VOXROOM_VIZ_EVERY_STEPS" \
      --door-seed-raw-source voxroom_tvars_vertical_union \
      --door-seed-collection-every-steps 5 \
      --door-seed-save-full-voxel-milestones \
      --door-seed-full-voxel-coverage-milestones "$FULL_VOXEL_COVERAGE_MILESTONES" \
      --door-seed-tvars-seed-width-cells 3 \
      --door-seed-local-voxel-patch-size 19 \
      --door-seed-context-patch-size 41 \
      --door-seed-persistent-final-raw-seed-union \
      --door-seed-collection-root "$COLLECTION_ROOT" \
      --output "$scene_run_dir/results.jsonl" \
      > "$scene_run_dir/run.log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    echo "[raw-seed-union-batch] finished dataset=$DATASET scene=$scene_id"
  else
    echo "[raw-seed-union-batch] failed dataset=$DATASET scene=$scene_id status=$status" | tee -a "$scene_run_dir/run.log"
  fi
  return "$status"
}

echo "[raw-seed-union-batch] dataset=$DATASET scenes=${#SCENES[@]} jobs=$PARALLEL_JOBS"
active_jobs=0
failure_count=0
for scene_id in "${SCENES[@]}"; do
  run_scene "$scene_id" &
  active_jobs=$((active_jobs + 1))
  if [[ "$active_jobs" -ge "$PARALLEL_JOBS" ]]; then
    if ! wait -n; then failure_count=$((failure_count + 1)); fi
    active_jobs=$((active_jobs - 1))
  fi
done
while [[ "$active_jobs" -gt 0 ]]; do
  if ! wait -n; then failure_count=$((failure_count + 1)); fi
  active_jobs=$((active_jobs - 1))
done
echo "[raw-seed-union-batch] complete dataset=$DATASET failures=$failure_count"
[[ "$failure_count" -eq 0 ]]
