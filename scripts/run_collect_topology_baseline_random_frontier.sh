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
  echo "[voxroom-topology-collect] provide an episode file" >&2
  echo "usage: scripts/run_collect_topology_baseline_random_frontier.sh data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl" >&2
  exit 1
fi

SCENE_ID="$(basename "$EPISODE_FILE" .jsonl)"
RUN_DIR="${RUN_DIR:-result/single_scene_random_frontier_collect_topology/$SCENE_ID}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$RUN_DIR/roomseg_snapshots}"
TOPOLOGY_DIR="${TOPOLOGY_DIR:-$RUN_DIR/baselines/topology_visual_active}"
PANORAMA_STEPS="${PANORAMA_STEPS:-12}"
ROOMSEG_SNAPSHOT_MAX_SAVES="${ROOMSEG_SNAPSHOT_MAX_SAVES:-100000}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5000}"
HEADLESS_FLAG="${HEADLESS_FLAG---headless}"
STRICT_MAIN_COLLECTION="${STRICT_MAIN_COLLECTION:-0}"

# shellcheck disable=SC1091
source scripts/comparison/env.sh

RUN_ONE_EXTRA_ARGS=()
STRICT_BENCHMARK_ARGS=()
if [[ "$STRICT_MAIN_COLLECTION" == "1" ]]; then
  if [[ -z "$TOPOLOGY_DOOR_DETR_CHECKPOINT" || ! -f "$TOPOLOGY_DOOR_DETR_CHECKPOINT" ]]; then
    echo "[voxroom-topology-collect] strict collection requires TOPOLOGY_DOOR_DETR_CHECKPOINT" >&2
    exit 1
  fi
else
  RUN_ONE_EXTRA_ARGS+=(--allow-debug-fallbacks)
  STRICT_BENCHMARK_ARGS+=(--no-strict-benchmark)
fi

mkdir -p "$SNAPSHOT_DIR" "$TOPOLOGY_DIR"

echo "[voxroom-topology-collect] scene=$SCENE_ID output=$RUN_DIR"
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
  --room-map-mode "${ROOM_MAP_MODE:-voxel_occupancy_door_wall_v33_vlm}" \
  --roomseg-backend "${ROOMSEG_BACKEND:-voxel_occupancy_door_wall_v33}" \
  "${STRICT_BENCHMARK_ARGS[@]}" \
  "${RUN_ONE_EXTRA_ARGS[@]}" \
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
  --panorama-steps "$PANORAMA_STEPS" \
  --voxroom-viz \
  --voxroom-viz-every-steps 1 \
  --save-roomseg-snapshots \
  --roomseg-snapshot-dir "$SNAPSHOT_DIR" \
  --roomseg-snapshot-max-saves "$ROOMSEG_SNAPSHOT_MAX_SAVES" \
  --save-roomseg-voxel-evidence \
  --live-roomseg-baseline topology_visual_active \
  --live-baseline-output-dir "$TOPOLOGY_DIR" \
  --live-baseline-save-stream \
  --live-baseline-save-every-snapshot \
  --live-baseline-door-detector "${LIVE_BASELINE_DOOR_DETECTOR:-original_detr}" \
  --live-baseline-policy-control never \
  --live-baseline-panorama-views "$PANORAMA_STEPS" \
  --output "$RUN_DIR/results.jsonl"
