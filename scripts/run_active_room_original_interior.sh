#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPISODE_FILE="${1:-$REPO_ROOT/data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl}"
export RUN_DIR="${RUN_DIR:-$REPO_ROOT/outputs/active_room_original_interior_kujiale_0003}"
export ACTIVE_ROOM_SEG_ROOT="${ACTIVE_ROOM_SEG_ROOT:-$HOME/Active_room_segmentation}"
export TOPOLOGY_DOOR_DETR_PYTHON="${TOPOLOGY_DOOR_DETR_PYTHON:-$HOME/.conda/envs/active-room-seg/bin/python}"
export ACTIVE_ROOM_SEG_EXPECTED_COMMIT="${ACTIVE_ROOM_SEG_EXPECTED_COMMIT:-36af8cd9ba46f6d01373c1f106b57b875a31ea7a}"

exec "$REPO_ROOT/scripts/run_active_room_original_closed_loop.sh" "$EPISODE_FILE"
