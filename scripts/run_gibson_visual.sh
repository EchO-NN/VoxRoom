#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PREFIX="${TARGET_PREFIX:-$HOME/.conda/envs/active-room-seg}"
PYTHON="${PYTHON:-$TARGET_PREFIX/bin/python}"
SCENE_ID="${SCENE_ID:-Swormville}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
RUN_ID="${RUN_ID:-$(< /proc/sys/kernel/random/uuid)}"
RUN_TAG="${RUN_TAG:-gibson_visual_${SCENE_ID}_$(date +%Y%m%d_%H%M%S)}"
LOCK_PATH="$ROOT_DIR/data/.gibson_visual.lock"
PREPARED_DIR="$ROOT_DIR/data/gibson-visual-runs/$RUN_ID"
PREPARED_DATASET="$PREPARED_DIR/input_dataset.json.gz"
PREPARED_MANIFEST="$PREPARED_DIR/input_manifest.json"

if [[ "${REQUIRE_TOPOLOGY_TRANSITION:-1}" != "1" ]]; then
    echo "The Gibson visual entry point requires strict topology validation" >&2
    exit 1
fi

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "Another Gibson visual preparation or run is active" >&2
    exit 1
fi

"$PYTHON" "$ROOT_DIR/scripts/prepare_gibson_visual.py" \
    --repository-root "$ROOT_DIR" \
    --scene-id "$SCENE_ID" \
    --episode-index "$EPISODE_INDEX" \
    --output-dataset "$PREPARED_DATASET" \
    --manifest-path "$PREPARED_MANIFEST"

export TASK_CONFIG="tasks/pointnav_gibson_visual.yaml"
export SPLIT="val"
export REQUIRE_TOPOLOGY_TRANSITION=1
export PAD_EPISODE_TO_MAX_STEPS=0
export ALLOW_EARLY_COMPLETION=1
export VISUALIZATION_FRAME_EVERY_STEPS=5
export WINDOW_CHECKPOINT_EVERY_STEPS=100
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-2500}"
export RUN_CONTEXT_REQUIRED=1
export RUN_CONTEXT_MANIFEST="$PREPARED_MANIFEST"
export RUN_CONTEXT_DATASET="$PREPARED_DATASET"
export RUN_ID
export RUN_TAG

exec "$ROOT_DIR/scripts/run_habitat_test.sh"
