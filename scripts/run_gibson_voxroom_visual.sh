#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VOXROOM_SIDECAR=1
export VOXROOM_ROOT="${VOXROOM_ROOT:-/home/joey/VoxRoom}"
export VOXROOM_CONFIG="${VOXROOM_CONFIG:-$VOXROOM_ROOT/configs/voxroom_online.yaml}"
export VOXROOM_ROOMSEG_EVERY_STEPS="${VOXROOM_ROOMSEG_EVERY_STEPS:-50}"
export VOXROOM_VISUALIZATION_EVERY_STEPS="${VOXROOM_VISUALIZATION_EVERY_STEPS:-5}"
export VOXROOM_MAP_SIZE_M="${VOXROOM_MAP_SIZE_M:-48.0}"

exec "$ROOT_DIR/scripts/run_gibson_visual.sh"
