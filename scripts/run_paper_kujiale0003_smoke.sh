#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPISODE_FILE="${1:-data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl}"
RUN_DIR="${RUN_DIR:-result/kujiale_0003_voxroom}"

export RUN_DIR
export ROOMSEG_SNAPSHOT_MAX_SAVES="${ROOMSEG_SNAPSHOT_MAX_SAVES:-100000}"
export VOXROOM_NUMBA_THREADS="${VOXROOM_NUMBA_THREADS:-28}"
export VOXROOM_VOXEL_CPU_NUMBA_THREADS="${VOXROOM_VOXEL_CPU_NUMBA_THREADS:-28}"

scripts/run_one_scene_random_frontier.sh "$EPISODE_FILE"

