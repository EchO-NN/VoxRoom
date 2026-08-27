#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GRSCENE_ROOT="${GRSCENE_ROOT:-${GRSCENE_DATA_ROOT:-$HOME/GRScenes-100/home_scenes/scenes}}"
DATASET_ROOT="${DATASET_ROOT:-data/grscene_home_compat_all_render_safe}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-data/grscene_home_preprocessed_all_radius005}"
EPISODE_DIR="${EPISODE_DIR:-data/grscene_home_episodes/radius005_all_render_safe}"
RUN_ROOT="${RUN_ROOT:-outputs/grscene_all_roomseg_20260721}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-6000}"
VOXROOM_VIZ_SAVE_EVERY_STEPS="${VOXROOM_VIZ_SAVE_EVERY_STEPS:-50}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
FORCE_EPISODES="${FORCE_EPISODES:-0}"
MIN_AVAILABLE_RAM_MB="${MIN_AVAILABLE_RAM_MB:-28000}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-12000}"
RESOURCE_POLL_SECONDS="${RESOURCE_POLL_SECONDS:-30}"

scripts/run_voxroom_isaac_env.sh scripts/prepare_all_grscene_home.py \
  --grscene-root "$GRSCENE_ROOT"

if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$PREPROCESSED_DIR/index.json" ]]; then
  scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/preprocess_interioragent.py \
    --dataset-root "$DATASET_ROOT" \
    --scene-glob 'kujiale_grscene_*' \
    --out "$PREPROCESSED_DIR" \
    --resolution 0.05 \
    --robot-radius-m 0.05 \
    --inflation-radius-m 0.0
fi

if [[ "$FORCE_EPISODES" == "1" || ! -s "$EPISODE_DIR/episode_files.txt" ]]; then
  scripts/run_voxroom_isaac_env.sh scripts/generate_grscene_exploration_episodes.py \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --out-dir "$EPISODE_DIR" \
    --robot-spawn-height-m 0.05
fi

env -u DISPLAY -u WAYLAND_DISPLAY \
  DATASET_ROOT="$DATASET_ROOT" \
  PREPROCESSED_DIR="$PREPROCESSED_DIR" \
  EPISODE_DIR="$EPISODE_DIR" \
  RUN_ROOT="$RUN_ROOT" \
  SCENE_GLOB='kujiale_grscene_*' \
  PARALLEL_JOBS=1 \
  RUN_NUMBA_NUM_THREADS=28 \
  MAX_CONTROL_STEPS="$MAX_CONTROL_STEPS" \
  VOXROOM_VIZ_SAVE_EVERY_STEPS="$VOXROOM_VIZ_SAVE_EVERY_STEPS" \
  SKIP_COMPLETED_SCENES=1 \
  MIN_AVAILABLE_RAM_MB="$MIN_AVAILABLE_RAM_MB" \
  MIN_FREE_GPU_MB="$MIN_FREE_GPU_MB" \
  RESOURCE_POLL_SECONDS="$RESOURCE_POLL_SECONDS" \
  CONTINUE_ON_ERROR=1 \
  scripts/run_all_scenes_random_frontier.sh
