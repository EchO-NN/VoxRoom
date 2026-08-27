#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/voxroom_online.yaml}"
EPISODE_FILE="${1:-${EPISODE_FILE:-}}"
if [[ -z "$EPISODE_FILE" || ! -f "$EPISODE_FILE" ]]; then
  echo "[active-room-port] provide one InteriorAgent or GRScene episode JSONL" >&2
  exit 1
fi

SCENE_ID="$(basename "$EPISODE_FILE" .jsonl)"
RUN_DIR="${RUN_DIR:-outputs/active_room_original_closed_loop/$SCENE_ID}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$RUN_DIR/roomseg_snapshots}"
TVARS_DIR="${TVARS_DIR:-$RUN_DIR/baselines/tvars_original_isaac}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5000}"
HEADLESS_FLAG="${HEADLESS_FLAG---headless}"
MAX_VX_MPS="${MAX_VX_MPS:-0.15}"
MAX_VY_MPS="${MAX_VY_MPS:-0.0}"
MAX_WZ_RADPS="${MAX_WZ_RADPS:-0.35}"
NAVIGATION_PLANNER_BACKEND="${NAVIGATION_PLANNER_BACKEND:-astar}"

if [[ "$NAVIGATION_PLANNER_BACKEND" != "astar" ]]; then
  echo "[active-room-port] original topology control now requires clearance A*" >&2
  exit 1
fi

export ACTIVE_ROOM_SEG_ROOT="${ACTIVE_ROOM_SEG_ROOT:-$HOME/Active_room_segmentation}"
export TOPOLOGY_DOOR_DETR_PYTHON="${TOPOLOGY_DOOR_DETR_PYTHON:-$HOME/.conda/envs/active-room-seg/bin/python}"
export TOPOLOGY_DOOR_DETR_CHECKPOINT="${TOPOLOGY_DOOR_DETR_CHECKPOINT:-$ACTIVE_ROOM_SEG_ROOT/detr_door_detection/train_params/detr_resnet50_4/final_doors_dataset/model.pth}"
export ACTIVE_ROOM_SEG_EXPECTED_COMMIT="${ACTIVE_ROOM_SEG_EXPECTED_COMMIT:-c6dbe92c55ea34f9710ddcc5b10d59144662fe68}"

if [[ ! -d "$ACTIVE_ROOM_SEG_ROOT/.git" ]]; then
  echo "[active-room-port] source checkout missing: $ACTIVE_ROOM_SEG_ROOT" >&2
  exit 1
fi
actual_commit="$(git -C "$ACTIVE_ROOM_SEG_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$ACTIVE_ROOM_SEG_EXPECTED_COMMIT" ]]; then
  echo "[active-room-port] source commit mismatch: expected=$ACTIVE_ROOM_SEG_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$ACTIVE_ROOM_SEG_ROOT" status --porcelain)" ]]; then
  echo "[active-room-port] source checkout must be clean" >&2
  exit 1
fi
if [[ ! -x "$TOPOLOGY_DOOR_DETR_PYTHON" ]]; then
  echo "[active-room-port] DETR Python is not executable: $TOPOLOGY_DOOR_DETR_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$TOPOLOGY_DOOR_DETR_CHECKPOINT" ]]; then
  echo "[active-room-port] DETR checkpoint missing: $TOPOLOGY_DOOR_DETR_CHECKPOINT" >&2
  exit 1
fi

desktop_environment="$(systemctl --user show-environment)"
physical_display="$(
  awk -F= '$1 == "DISPLAY" {sub(/^[^=]*=/, ""); print; exit}' \
    <<<"$desktop_environment"
)"
physical_xauthority="$(
  awk -F= '$1 == "XAUTHORITY" {sub(/^[^=]*=/, ""); print; exit}' \
    <<<"$desktop_environment"
)"
if [[ ! "$physical_display" =~ ^:[0-9]+$ || ! -r "$physical_xauthority" ]]; then
  echo "[active-room-port] physical X11 display is unavailable" >&2
  exit 1
fi
if [[ ! -S "/tmp/.X11-unix/X${physical_display#:}" ]]; then
  echo "[active-room-port] physical X11 socket is unavailable: $physical_display" >&2
  exit 1
fi
export DISPLAY="$physical_display"
export XAUTHORITY="$physical_xauthority"

mkdir -p "$SNAPSHOT_DIR" "$TVARS_DIR"

echo "[active-room-port] scene=$SCENE_ID output=$RUN_DIR"
echo "[active-room-port] source=$ACTIVE_ROOM_SEG_ROOT commit=$actual_commit policy=original_topology planner=$NAVIGATION_PLANNER_BACKEND"
echo "[active-room-port] motion_limits_mps_radps=$MAX_VX_MPS,$MAX_VY_MPS,$MAX_WZ_RADPS control_dt=0.2"
VOXROOM_NUMBA_THREADS="${VOXROOM_NUMBA_THREADS:-28}" \
  scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/run_one_episode.py \
  --config "$CONFIG" \
  --episode-file "$EPISODE_FILE" \
  --episode-index 0 \
  --sim-backend isaac \
  --planner "$NAVIGATION_PLANNER_BACKEND" \
  --policy active_room_original_topology \
  --detector none \
  --segmenter none \
  --no-llm-enabled \
  --no-vllm-frontier-scoring \
  --no-vllm-image-scoring \
  --no-allow-gt-goal-fallback \
  --no-frontier-allow-near-fallback \
  --frontier-selection-mode tvars_original \
  --frontier-source voxel_vertical_free \
  --room-map-mode "${ROOM_MAP_MODE:-voxel_occupancy_door_wall_v33}" \
  --roomseg-backend "${ROOMSEG_BACKEND:-voxel_occupancy_door_wall_v33}" \
  --no-strict-benchmark \
  --explore-until-no-frontiers \
  --max-control-steps "$MAX_CONTROL_STEPS" \
  --max-vx-mps "$MAX_VX_MPS" \
  --max-vy-mps "$MAX_VY_MPS" \
  --max-wz-radps "$MAX_WZ_RADPS" \
  --robot-radius-m "${ROBOT_RADIUS_M:-0.15}" \
  --runtime-planning-clearance-m "${RUNTIME_PLANNING_CLEARANCE_M:-0.01}" \
  --astar-clearance-cost-enabled \
  --astar-clearance-desired-m "${ASTAR_CLEARANCE_DESIRED_M:-0.20}" \
  --astar-clearance-hard-min-m "${ASTAR_CLEARANCE_HARD_MIN_M:-0.05}" \
  --astar-clearance-weight "${ASTAR_CLEARANCE_WEIGHT:-6.0}" \
  --astar-goal-min-clearance-m "${ASTAR_GOAL_MIN_CLEARANCE_M:-0.08}" \
  --lookahead-min-clearance-m "${LOOKAHEAD_MIN_CLEARANCE_M:-0.06}" \
  --guard-min-clearance-m "${GUARD_MIN_CLEARANCE_M:-0.06}" \
  $HEADLESS_FLAG \
  --read-depth \
  --camera-annotator-device cpu \
  --isaac-width 256 \
  --isaac-height 256 \
  --camera-hfov-deg 90 \
  --camera-mast-height-m 1.25 \
  --panorama-steps 12 \
  --live-baseline-panorama-views 12 \
  --voxroom-viz \
  --voxroom-viz-every-steps 1 \
  --save-roomseg-snapshots \
  --roomseg-snapshot-dir "$SNAPSHOT_DIR" \
  --roomseg-snapshot-max-saves 100000 \
  --save-roomseg-voxel-evidence \
  --live-roomseg-baseline tvars_original_isaac \
  --live-baseline-output-dir "$TVARS_DIR" \
  --live-baseline-save-stream \
  --live-baseline-save-every-snapshot \
  --live-baseline-door-detector original_detr \
  --live-baseline-policy-control original_topology \
  --output "$RUN_DIR/results.jsonl"
