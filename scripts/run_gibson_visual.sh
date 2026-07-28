#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PREFIX="${TARGET_PREFIX:-$HOME/.conda/envs/active-room-seg}"
PYTHON="${PYTHON:-$TARGET_PREFIX/bin/python}"
SCENE_ID="${SCENE_ID:-Swormville}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"

"$PYTHON" "$ROOT_DIR/scripts/prepare_gibson_visual.py" \
    --repository-root "$ROOT_DIR" \
    --scene-id "$SCENE_ID" \
    --episode-index "$EPISODE_INDEX"

export TASK_CONFIG="tasks/pointnav_gibson_visual.yaml"
export SPLIT="val"
export REQUIRE_TOPOLOGY_TRANSITION="${REQUIRE_TOPOLOGY_TRANSITION:-1}"
export VISUALIZATION_FRAME_EVERY_STEPS="${VISUALIZATION_FRAME_EVERY_STEPS:-5}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-2500}"
export RUN_TAG="${RUN_TAG:-gibson_visual_${SCENE_ID}_$(date +%Y%m%d_%H%M%S)}"

exec "$ROOT_DIR/scripts/run_habitat_test.sh"
