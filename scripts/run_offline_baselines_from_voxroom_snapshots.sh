#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-result/radius005_robot005_random_frontier_all_scenes_collect_topology}"
OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:-$SOURCE_RUN_ROOT}"
SCENE_GLOB="${SCENE_GLOB:-kujiale_*}"
BASELINES=()
FALLBACK_FLAG=()
STRICT_NATIVE=0

# shellcheck disable=SC1091
source scripts/comparison/env.sh

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-run-root) SOURCE_RUN_ROOT="$2"; shift 2 ;;
    --output-run-root) OUTPUT_RUN_ROOT="$2"; shift 2 ;;
    --scene-glob) SCENE_GLOB="$2"; shift 2 ;;
    --baselines)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        BASELINES+=("$1")
        shift
      done
      ;;
    --fallback-python) FALLBACK_FLAG=(--fallback-python); shift ;;
    --strict-native) STRICT_NATIVE=1; shift ;;
    --no-strict-native) STRICT_NATIVE=0; shift ;;
    *) echo "[voxroom-offline-baselines] unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "${#BASELINES[@]}" -eq 0 ]]; then
  BASELINES=(dude_incremental rose2 morphological distance_transform voronoi)
fi

STRICT_ARGS=()
if [[ "$STRICT_NATIVE" == "1" ]]; then
  STRICT_ARGS+=(--strict-native)
fi

python -m voxroom_online.isaac_runtime.baselines.offline.run_saved_snapshots \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --output-run-root "$OUTPUT_RUN_ROOT" \
  --scene-glob "$SCENE_GLOB" \
  --baselines "${BASELINES[@]}" \
  --ros-baseline-setup "$ROS_BASELINE_SETUP" \
  --ros-baseline-python "$ROS_BASELINE_PYTHON" \
  --dude-ws "$DUDE_WS" \
  --rose2-ws "$ROSE2_WS" \
  --ipa-ws "$IPA_WS" \
  --active-room-seg-root "$ACTIVE_ROOM_SEG_ROOT" \
  --dude-repo-root "$INCREMENTAL_DUDE_ROS_ROOT" \
  --rose2-ros-workspace "$ROSE2_WS" \
  --ipa-ros-workspace "$IPA_WS" \
  --topology-checkpoint "$TOPOLOGY_DOOR_DETR_CHECKPOINT" \
  "${STRICT_ARGS[@]}" \
  "${FALLBACK_FLAG[@]}"
