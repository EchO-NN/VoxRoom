#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EPISODE_DIR="${EPISODE_DIR:-data/interioragent_episodes/radius005_all_scenes}"
SCENE_GLOB="${SCENE_GLOB:-kujiale_*.jsonl}"
RUN_ROOT="${RUN_ROOT:-result/comparison_random_frontier_v1}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5000}"
PANORAMA_VIEWS="${PANORAMA_VIEWS:-12}"
MAX_SCENES=""
STRICT_MAIN_COLLECTION="${STRICT_MAIN_COLLECTION:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episode-dir) EPISODE_DIR="$2"; shift 2 ;;
    --scene-glob) SCENE_GLOB="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --max-control-steps) MAX_CONTROL_STEPS="$2"; shift 2 ;;
    --panorama-views) PANORAMA_VIEWS="$2"; shift 2 ;;
    --max-scenes) MAX_SCENES="$2"; shift 2 ;;
    --strict-main-collection) STRICT_MAIN_COLLECTION=1; shift ;;
    --no-strict-main-collection) STRICT_MAIN_COLLECTION=0; shift ;;
    *) echo "[comparison-collect-all] unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
COLLECT_STRICT_ARGS=()
if [[ "$STRICT_MAIN_COLLECTION" == "1" ]]; then
  COLLECT_STRICT_ARGS+=(--strict-main-collection)
fi
count=0
while IFS= read -r episode_file; do
  [[ -z "$episode_file" ]] && continue
  count=$((count + 1))
  if [[ -n "$MAX_SCENES" && "$count" -gt "$MAX_SCENES" ]]; then
    break
  fi
  scene_id="$(basename "$episode_file" .jsonl)"
  bash scripts/comparison/01_collect_online_random_frontier.sh \
    --episode-file "$episode_file" \
    --run-root "$RUN_ROOT" \
    --scene-id "$scene_id" \
    --max-control-steps "$MAX_CONTROL_STEPS" \
    --panorama-views "$PANORAMA_VIEWS" \
    "${COLLECT_STRICT_ARGS[@]}"
done < <(find "$EPISODE_DIR" -maxdepth 1 -name "$SCENE_GLOB" -type f | sort)

echo "[comparison-collect-all] processed $count candidate scene(s)"
