#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPLIT_FILE="${SPLIT_FILE:-data/door_seed_learning/seed_20260714/scene_split.json}"
ROLE="${ROLE:-development_collection}"
EPISODE_DIR="${EPISODE_DIR:-data/door_seed_learning/seed_20260714/episodes}"
COLLECTION_ROOT="${COLLECTION_ROOT:-outputs/door_seed_collection_6000}"
RUN_ROOT="${RUN_ROOT:-outputs/door_seed_collection_runs/seed_20260714_development}"
CONFIG="${CONFIG:-configs/voxroom_online.yaml}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-6000}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
RUN_NUMBA_NUM_THREADS="${RUN_NUMBA_NUM_THREADS:-28}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
ALLOW_HELDOUT_COLLECTION="${ALLOW_HELDOUT_COLLECTION:-0}"
DRY_RUN="${DRY_RUN:-0}"
VISUALIZE="${VISUALIZE:-0}"
VOXROOM_VIZ_EVERY_STEPS="${VOXROOM_VIZ_EVERY_STEPS:-1}"
SANITIZE_DISPLAY_PRIMVARS="${SANITIZE_DISPLAY_PRIMVARS:-1}"

VISUALIZATION_ARGS=(--headless --no-voxroom-viz)
if [[ "$VISUALIZE" == "1" ]]; then
  VISUALIZATION_ARGS=(
    --headless
    --voxroom-viz
    --voxroom-viz-every-steps "$VOXROOM_VIZ_EVERY_STEPS"
  )
fi

if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "[door-seed-batch] missing split file: $SPLIT_FILE" >&2
  exit 2
fi
if [[ "$ROLE" =~ ^(heldout_validation|test)$ && "$ALLOW_HELDOUT_COLLECTION" != "1" ]]; then
  echo "[door-seed-batch] heldout validation is frozen; set ALLOW_HELDOUT_COLLECTION=1 only after the checkpoint is frozen" >&2
  exit 2
fi
if [[ "$PARALLEL_JOBS" -lt 1 ]]; then
  echo "[door-seed-batch] PARALLEL_JOBS must be positive" >&2
  exit 2
fi

mapfile -t SCENES < <(python - "$SPLIT_FILE" "$ROLE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
role = sys.argv[2]
if role in {"development_collection", "heldout_validation"}:
    values = payload["roles"][role]
elif role in {"train", "val", "test"}:
    values = payload[role]
else:
    raise SystemExit("unsupported ROLE: %s" % role)
for scene in values:
    print(scene)
PY
)
if [[ "${#SCENES[@]}" -eq 0 ]]; then
  echo "[door-seed-batch] no scenes selected for role=$ROLE" >&2
  exit 2
fi

is_collection_complete() {
  local scene_id="$1"
  python - "$COLLECTION_ROOT" "$scene_id" <<'PY'
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
    snapshots = list(manifest.get("snapshots", []))
    finals = [item for item in snapshots if bool(item.get("is_final"))]
    if len(finals) == 1 and (path.parent / str(finals[0].get("path", ""))).is_file():
        print(path.parent)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

validate_episode() {
  local scene_id="$1"
  local episode_file="$EPISODE_DIR/$scene_id.jsonl"
  local report_file="$EPISODE_DIR/$scene_id.report.json"
  python - "$episode_file" "$report_file" "$scene_id" <<'PY'
import json
import sys
from pathlib import Path

episode_path, report_path, expected_scene = map(Path, sys.argv[1:])
if not episode_path.is_file() or not report_path.is_file():
    raise SystemExit("missing strict episode/report for %s" % expected_scene)
episode = json.loads(next(line for line in episode_path.read_text(encoding="utf-8").splitlines() if line.strip()))
report = json.loads(report_path.read_text(encoding="utf-8"))
if str(episode.get("scene_id")) != str(expected_scene):
    raise SystemExit("episode scene mismatch for %s" % expected_scene)
if list(report.get("selection_fallback_errors", [])):
    raise SystemExit("relaxed episode fallback is forbidden for %s" % expected_scene)
profile = dict(report.get("selection_profile", {}) or {})
if profile.get("name") != "requested":
    raise SystemExit("episode did not use requested constraints for %s" % expected_scene)
PY
}

mkdir -p "$COLLECTION_ROOT" "$RUN_ROOT"
LOCK_DIR="$RUN_ROOT/.run_lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  old_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[door-seed-batch] batch already running as pid $old_pid" >&2
    exit 1
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

run_scene() {
  local scene_id="$1"
  local episode_file="$EPISODE_DIR/$scene_id.jsonl"
  local scene_run_dir="$RUN_ROOT/$scene_id"
  local complete_path=""
  if complete_path="$(is_collection_complete "$scene_id")"; then
    echo "[door-seed-batch] skip complete $scene_id -> $complete_path"
    return 0
  fi
  validate_episode "$scene_id" || return $?
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[door-seed-batch] validated $scene_id"
    return 0
  fi
  mkdir -p "$scene_run_dir"
  rm -f "$scene_run_dir/results.jsonl"
  {
    echo "scene_id=$scene_id"
    echo "role=$ROLE"
    echo "split_file=$SPLIT_FILE"
    echo "episode_file=$episode_file"
    echo "collection_root=$COLLECTION_ROOT"
    echo "max_control_steps=$MAX_CONTROL_STEPS"
    echo "visualize=$VISUALIZE"
    echo "camera_pose_sync=five_app_updates_two_fresh_frames"
    echo "sanitize_display_primvars=$SANITIZE_DISPLAY_PRIMVARS"
    echo "segmenter=none"
    echo "roomseg_visual_overlays=deployment_default"
  } > "$scene_run_dir/command.txt"
  echo "[door-seed-batch] running $scene_id role=$ROLE max_steps=$MAX_CONTROL_STEPS"
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
      --frontier-selection-mode random \
      --frontier-random-seed 0 \
      --frontier-source voxel_vertical_free \
      --room-map-mode voxel_occupancy_door_wall_v33 \
      --roomseg-backend voxel_occupancy_door_wall_v33 \
      --no-strict-benchmark \
      --explore-until-no-frontiers \
      --max-control-steps "$MAX_CONTROL_STEPS" \
      --robot-radius-m 0.05 \
      --runtime-planning-clearance-m 0.05 \
      --astar-clearance-hard-min-m 0.06 \
      --astar-clearance-desired-m 0.18 \
      --astar-goal-min-clearance-m 0.12 \
      --lookahead-min-clearance-m 0.10 \
      --guard-min-clearance-m 0.10 \
      "${VISUALIZATION_ARGS[@]}" \
      --door-seed-collection-root "$COLLECTION_ROOT" \
      --output "$scene_run_dir/results.jsonl" \
      > "$scene_run_dir/run.log" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    echo "[door-seed-batch] finished $scene_id"
  else
    echo "[door-seed-batch] failed $scene_id status=$status" | tee -a "$scene_run_dir/run.log"
  fi
  return "$status"
}

echo "[door-seed-batch] role=$ROLE scenes=${#SCENES[@]} jobs=$PARALLEL_JOBS"
active_jobs=0
failure_count=0
for scene_id in "${SCENES[@]}"; do
  run_scene "$scene_id" &
  active_jobs=$((active_jobs + 1))
  if [[ "$active_jobs" -ge "$PARALLEL_JOBS" ]]; then
    if ! wait -n; then failure_count=$((failure_count + 1)); fi
    active_jobs=$((active_jobs - 1))
    if [[ "$CONTINUE_ON_ERROR" != "1" && "$failure_count" -gt 0 ]]; then
      wait
      exit 1
    fi
  fi
done
while [[ "$active_jobs" -gt 0 ]]; do
  if ! wait -n; then failure_count=$((failure_count + 1)); fi
  active_jobs=$((active_jobs - 1))
done

echo "[door-seed-batch] complete role=$ROLE failures=$failure_count"
if [[ "$failure_count" -gt 0 && "$CONTINUE_ON_ERROR" != "1" ]]; then
  exit 1
fi
