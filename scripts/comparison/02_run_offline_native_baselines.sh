#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-result/comparison_random_frontier_v1}"
SCENE_GLOB="${SCENE_GLOB:-kujiale_*}"
MAX_SNAPSHOTS_PER_SCENE=""
METHODS=(dude_incremental rose2 morphological distance_transform voronoi)
STRICT_NATIVE=0
MAP_RESOLUTION_M="${MAP_RESOLUTION_M:-0.05}"

# shellcheck disable=SC1091
source scripts/comparison/env.sh

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --scene-glob) SCENE_GLOB="$2"; shift 2 ;;
    --max-snapshots-per-scene) MAX_SNAPSHOTS_PER_SCENE="$2"; shift 2 ;;
    --map-resolution-m) MAP_RESOLUTION_M="$2"; shift 2 ;;
    --strict-native) STRICT_NATIVE=1; shift ;;
    --no-strict-native) STRICT_NATIVE=0; shift ;;
    --methods)
      shift
      METHODS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done
      ;;
    *) echo "[comparison-offline-native] unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT/commands"
ARGS=(
  --source-run-root "$RUN_ROOT"
  --output-run-root "$RUN_ROOT"
  --scene-glob "$SCENE_GLOB"
  --baselines "${METHODS[@]}"
  --overwrite
)
if [[ -n "$MAX_SNAPSHOTS_PER_SCENE" ]]; then
  ARGS+=(--max-snapshots-per-scene "$MAX_SNAPSHOTS_PER_SCENE")
fi
if [[ -n "$MAP_RESOLUTION_M" ]]; then
  ARGS+=(--map-resolution-m "$MAP_RESOLUTION_M")
fi
if [[ "$STRICT_NATIVE" == "1" ]]; then
  ARGS+=(--strict-native)
fi
ARGS+=(
  --ros-baseline-setup "$ROS_BASELINE_SETUP"
  --ros-baseline-python "$ROS_BASELINE_PYTHON"
  --dude-ws "$DUDE_WS"
  --rose2-ws "$ROSE2_WS"
  --ipa-ws "$IPA_WS"
  --active-room-seg-root "$ACTIVE_ROOM_SEG_ROOT"
  --dude-repo-root "$INCREMENTAL_DUDE_ROS_ROOT"
  --rose2-ros-workspace "$ROSE2_WS"
  --ipa-ros-workspace "$IPA_WS"
  --topology-checkpoint "$TOPOLOGY_DOOR_DETR_CHECKPOINT"
  --provenance-out "$RUN_ROOT/baseline_replay_provenance.json"
)

"$VOXROOM_PYTHON" -m voxroom_online.isaac_runtime.baselines.offline.run_saved_snapshots "${ARGS[@]}" \
  2>&1 | tee "$RUN_ROOT/commands/02_offline_replay_native.log"
