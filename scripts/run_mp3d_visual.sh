#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PREFIX="${TARGET_PREFIX:-$HOME/.conda/envs/active-room-seg}"
PYTHON="${PYTHON:-$TARGET_PREFIX/bin/python}"
SCENE_ID="${SCENE_ID:-2azQ1b91cZZ}"

"$PYTHON" "$ROOT_DIR/scripts/prepare_mp3d_visual.py" \
    --repository-root "$ROOT_DIR" \
    --scene-id "$SCENE_ID"

export TASK_CONFIG="tasks/pointnav_mp3d_visual.yaml"
export SPLIT="val"
export REQUIRE_TOPOLOGY_TRANSITION="${REQUIRE_TOPOLOGY_TRANSITION:-1}"
export VISUALIZATION_FRAME_EVERY_STEPS="${VISUALIZATION_FRAME_EVERY_STEPS:-5}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
export RUN_TAG="${RUN_TAG:-mp3d_visual_${SCENE_ID}_$(date +%Y%m%d_%H%M%S)}"

exec "$ROOT_DIR/scripts/run_habitat_test.sh"
