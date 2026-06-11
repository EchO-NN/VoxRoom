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
  echo "[voxroom-run] provide an episode file or generate episodes first" >&2
  echo "usage: scripts/run_one_scene_random_frontier.sh data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl" >&2
  exit 1
fi

SCENE_ID="$(basename "$EPISODE_FILE" .jsonl)"
RUN_DIR="${RUN_DIR:-result/single_scene_random_frontier/$SCENE_ID}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$RUN_DIR/roomseg_snapshots}"
VOXROOM_VIZ_SAVE_DIR="${VOXROOM_VIZ_SAVE_DIR:-$RUN_DIR/voxroom_viz_frames}"
VOXROOM_VIZ_SAVE_EVERY_STEPS="${VOXROOM_VIZ_SAVE_EVERY_STEPS:-1}"
ROOM_MAP_MODE="${ROOM_MAP_MODE:-voxel_occupancy_door_wall_v33_vlm}"
ROOMSEG_BACKEND="${ROOMSEG_BACKEND:-voxel_occupancy_door_wall_v33}"
ROBOT_RADIUS_M="${ROBOT_RADIUS_M:-0.05}"
RUNTIME_PLANNING_CLEARANCE_M="${RUNTIME_PLANNING_CLEARANCE_M:-0.01}"
ASTAR_CLEARANCE_DESIRED_M="${ASTAR_CLEARANCE_DESIRED_M:-0.12}"
ASTAR_GOAL_MIN_CLEARANCE_M="${ASTAR_GOAL_MIN_CLEARANCE_M:-0.08}"
LOOKAHEAD_MIN_CLEARANCE_M="${LOOKAHEAD_MIN_CLEARANCE_M:-0.06}"
GUARD_MIN_CLEARANCE_M="${GUARD_MIN_CLEARANCE_M:-0.06}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5000}"
ROOMSEG_SNAPSHOT_MAX_SAVES="${ROOMSEG_SNAPSHOT_MAX_SAVES:-100000}"
HEADLESS_FLAG="${HEADLESS_FLAG---headless}"

mkdir -p "$SNAPSHOT_DIR"

echo "[voxroom-run] scene=$SCENE_ID output=$RUN_DIR"
VOXROOM_NUMBA_THREADS="${VOXROOM_NUMBA_THREADS:-28}" \
VOXROOM_VOXEL_CPU_NUMBA_THREADS="${VOXROOM_VOXEL_CPU_NUMBA_THREADS:-${VOXROOM_NUMBA_THREADS:-28}}" \
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
  --room-map-mode "$ROOM_MAP_MODE" \
  --roomseg-backend "$ROOMSEG_BACKEND" \
  --no-strict-benchmark \
  --allow-debug-fallbacks \
  --explore-until-no-frontiers \
  --max-control-steps "$MAX_CONTROL_STEPS" \
  --robot-radius-m "$ROBOT_RADIUS_M" \
  --runtime-planning-clearance-m "$RUNTIME_PLANNING_CLEARANCE_M" \
  --astar-clearance-desired-m "$ASTAR_CLEARANCE_DESIRED_M" \
  --astar-goal-min-clearance-m "$ASTAR_GOAL_MIN_CLEARANCE_M" \
  --lookahead-min-clearance-m "$LOOKAHEAD_MIN_CLEARANCE_M" \
  --guard-min-clearance-m "$GUARD_MIN_CLEARANCE_M" \
  $HEADLESS_FLAG \
  --voxroom-viz \
  --voxroom-viz-every-steps 1 \
  --voxroom-viz-save-dir "$VOXROOM_VIZ_SAVE_DIR" \
  --voxroom-viz-save-every-steps "$VOXROOM_VIZ_SAVE_EVERY_STEPS" \
  --save-roomseg-snapshots \
  --roomseg-snapshot-dir "$SNAPSHOT_DIR" \
  --roomseg-snapshot-max-saves "$ROOMSEG_SNAPSHOT_MAX_SAVES" \
  --save-roomseg-voxel-evidence \
  --output "$RUN_DIR/results.jsonl"
