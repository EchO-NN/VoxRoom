#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EPISODE_FILE=""
RUN_ROOT="${RUN_ROOT:-result/comparison_random_frontier_v1}"
SCENE_ID=""
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5000}"
PANORAMA_VIEWS="${PANORAMA_VIEWS:-12}"
STRICT_MAIN_COLLECTION="${STRICT_MAIN_COLLECTION:-0}"

# shellcheck disable=SC1091
source scripts/comparison/env.sh

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episode-file) EPISODE_FILE="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --scene-id) SCENE_ID="$2"; shift 2 ;;
    --max-control-steps) MAX_CONTROL_STEPS="$2"; shift 2 ;;
    --panorama-views) PANORAMA_VIEWS="$2"; shift 2 ;;
    --strict-main-collection) STRICT_MAIN_COLLECTION=1; shift ;;
    --no-strict-main-collection) STRICT_MAIN_COLLECTION=0; shift ;;
    *) echo "[comparison-collect-one] unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$EPISODE_FILE" || ! -f "$EPISODE_FILE" ]]; then
  echo "[comparison-collect-one] --episode-file is required" >&2
  exit 1
fi
if [[ -z "$SCENE_ID" ]]; then
  SCENE_ID="$(basename "$EPISODE_FILE" .jsonl)"
fi

SCENE_ROOT="$RUN_ROOT/$SCENE_ID"
mkdir -p "$SCENE_ROOT" "$RUN_ROOT/commands"
{
  echo "episode_file=$EPISODE_FILE"
  echo "run_root=$RUN_ROOT"
  echo "scene_id=$SCENE_ID"
  echo "frontier_selection_mode=random"
  echo "frontier_source=voxel_vertical_free"
  echo "max_control_steps=$MAX_CONTROL_STEPS"
  echo "panorama_views=$PANORAMA_VIEWS"
  echo "strict_main_collection=$STRICT_MAIN_COLLECTION"
  echo "topology_door_detr_checkpoint=$TOPOLOGY_DOOR_DETR_CHECKPOINT"
} > "$SCENE_ROOT/command.txt"

RUN_DIR="$SCENE_ROOT" \
SNAPSHOT_DIR="$SCENE_ROOT/roomseg_snapshots" \
TOPOLOGY_DIR="$SCENE_ROOT/baselines/topology_visual_active" \
MAX_CONTROL_STEPS="$MAX_CONTROL_STEPS" \
PANORAMA_STEPS="$PANORAMA_VIEWS" \
STRICT_MAIN_COLLECTION="$STRICT_MAIN_COLLECTION" \
  bash scripts/run_collect_topology_baseline_random_frontier.sh "$EPISODE_FILE" \
  2>&1 | tee "$RUN_ROOT/commands/01_collect_${SCENE_ID}.sh.log"
