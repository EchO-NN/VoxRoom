#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/scripts/run_voxroom_isaac_env.sh" \
  "$REPO_ROOT/voxroom_online/isaac_runtime/scripts/evaluate_door_seed_classifier.py" \
  "$@"
