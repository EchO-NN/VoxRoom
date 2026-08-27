#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/voxroom_online.yaml}"
EPISODE_FILE="${1:-${EPISODE_FILE:-}}"
if [[ -z "$EPISODE_FILE" ]]; then
  EPISODE_FILE="$(find data/interioragent_episodes -name '*.jsonl' 2>/dev/null | sort | head -1 || true)"
fi
if [[ -z "$EPISODE_FILE" || ! -f "$EPISODE_FILE" ]]; then
  echo "[voxroom-tvars-original] provide an episode file" >&2
  echo "usage: scripts/run_collect_tvars_original_random_frontier.sh data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl" >&2
  exit 1
fi

SCENE_ID="$(basename "$EPISODE_FILE" .jsonl)"
RUN_DIR="${RUN_DIR:-result/single_scene_random_frontier_collect_tvars_original/$SCENE_ID}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$RUN_DIR/roomseg_snapshots}"
TVARS_DIR="${TVARS_DIR:-$RUN_DIR/baselines/tvars_original_isaac}"
ROOMSEG_SNAPSHOT_MAX_SAVES="${ROOMSEG_SNAPSHOT_MAX_SAVES:-100000}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5000}"
HEADLESS_FLAG="${HEADLESS_FLAG---headless}"

export ACTIVE_ROOM_SEG_ROOT="${ACTIVE_ROOM_SEG_ROOT:-$REPO_ROOT/external_baselines/Active_room_segmentation}"
export TOPOLOGY_DOOR_DETR_PYTHON="${TOPOLOGY_DOOR_DETR_PYTHON:-$HOME/.conda/envs/active-room-seg/bin/python}"
export TOPOLOGY_DOOR_DETR_CHECKPOINT="${TOPOLOGY_DOOR_DETR_CHECKPOINT:-$ACTIVE_ROOM_SEG_ROOT/detr_door_detection/train_params/detr_resnet50_4/final_doors_dataset/model.pth}"
if [[ ! -d "$ACTIVE_ROOM_SEG_ROOT/.git" ]]; then
  echo "[voxroom-tvars-original] Active Room git checkout not found: $ACTIVE_ROOM_SEG_ROOT" >&2
  exit 1
fi
if [[ -n "$(git -C "$ACTIVE_ROOM_SEG_ROOT" status --porcelain)" ]]; then
  echo "[voxroom-tvars-original] Active Room source tree must be clean" >&2
  exit 1
fi
if [[ -z "${ACTIVE_ROOM_SEG_EXPECTED_COMMIT:-}" ]]; then
  echo "[voxroom-tvars-original] ACTIVE_ROOM_SEG_EXPECTED_COMMIT must pin the validated source revision" >&2
  exit 1
fi
if [[ ! -x "$TOPOLOGY_DOOR_DETR_PYTHON" ]]; then
  echo "[voxroom-tvars-original] DETR Python is not executable: $TOPOLOGY_DOOR_DETR_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$TOPOLOGY_DOOR_DETR_CHECKPOINT" ]]; then
  echo "[voxroom-tvars-original] DETR checkpoint is missing: $TOPOLOGY_DOOR_DETR_CHECKPOINT" >&2
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
  echo "[voxroom-tvars-original] physical X11 display is unavailable" >&2
  exit 1
fi
if [[ ! -S "/tmp/.X11-unix/X${physical_display#:}" ]]; then
  echo "[voxroom-tvars-original] physical X11 socket is unavailable: $physical_display" >&2
  exit 1
fi
export DISPLAY="$physical_display"
export XAUTHORITY="$physical_xauthority"

mkdir -p "$SNAPSHOT_DIR" "$TVARS_DIR"

echo "[voxroom-tvars-original] scene=$SCENE_ID output=$RUN_DIR"
echo "[voxroom-tvars-original] habitat_runtime=disabled active_room_seg_root=$ACTIVE_ROOM_SEG_ROOT"
VOXROOM_NUMBA_THREADS="${VOXROOM_NUMBA_THREADS:-28}" \
  scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/run_one_episode.py \
  --config "$CONFIG" \
  --episode-file "$EPISODE_FILE" \
  --episode-index 0 \
  --sim-backend isaac \
  --planner astar \
  --detector none \
  --segmenter none \
  --no-llm-enabled \
  --no-vllm-frontier-scoring \
  --no-vllm-image-scoring \
  --frontier-selection-mode random \
  --frontier-source voxel_vertical_free \
  --room-map-mode "${ROOM_MAP_MODE:-voxel_occupancy_door_wall_v33}" \
  --roomseg-backend "${ROOMSEG_BACKEND:-voxel_occupancy_door_wall_v33}" \
  --no-strict-benchmark \
  --explore-until-no-frontiers \
  --max-control-steps "$MAX_CONTROL_STEPS" \
  --robot-radius-m "${ROBOT_RADIUS_M:-0.05}" \
  --runtime-planning-clearance-m "${RUNTIME_PLANNING_CLEARANCE_M:-0.01}" \
  --astar-clearance-desired-m "${ASTAR_CLEARANCE_DESIRED_M:-0.12}" \
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
  --live-baseline-panorama-views 12 \
  --voxroom-viz \
  --voxroom-viz-every-steps 1 \
  --save-roomseg-snapshots \
  --roomseg-snapshot-dir "$SNAPSHOT_DIR" \
  --roomseg-snapshot-max-saves "$ROOMSEG_SNAPSHOT_MAX_SAVES" \
  --save-roomseg-voxel-evidence \
  --live-roomseg-baseline tvars_original_isaac \
  --live-baseline-output-dir "$TVARS_DIR" \
  --live-baseline-save-stream \
  --live-baseline-save-every-snapshot \
  --live-baseline-door-detector original_detr \
  --live-baseline-policy-control never \
  --output "$RUN_DIR/results.jsonl"
