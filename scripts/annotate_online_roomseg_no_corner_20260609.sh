#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ROOT="${RUN_ROOT:-对比实验/outputs/final_latest_20260609_voronoi_width_jump_axis2_interval_per_free_parallel020_150_no_corner}"
RESULT_ROOT="${RESULT_ROOT:-$RUN_ROOT/postprocessed_voronoi_width_jump_no_corner_snap}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval_online_roomseg}"
INDEX_SOURCE_ROOT="$EVAL_DIR/result_root_no_corner"
INDEX_PATH="$EVAL_DIR/index.json"
ANNOTATION_DIR="$EVAL_DIR/annotations"
GT_DIR="$EVAL_DIR/gt_final"
FIGURE_SCALE="${FIGURE_SCALE:-1.25}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"

if [[ ! -d "$RESULT_ROOT" ]]; then
  echo "missing RESULT_ROOT: $RESULT_ROOT" >&2
  exit 2
fi

mkdir -p "$EVAL_DIR"

if [[ "$REBUILD_INDEX" == "1" || ! -f "$INDEX_PATH" || ! -d "$INDEX_SOURCE_ROOT" ]]; then
  rm -rf "$INDEX_SOURCE_ROOT"
  mkdir -p "$INDEX_SOURCE_ROOT"

  for scene_dir in "$RESULT_ROOT"/kujiale_*; do
    [[ -d "$scene_dir" ]] || continue
    target=""
    for candidate in "$scene_dir"/baselines/*; do
      [[ -d "$candidate/roomseg_snapshots" ]] || continue
      target="$candidate"
      break
    done
    [[ -n "$target" ]] || continue
    ln -sfn "$(realpath "$target")" "$INDEX_SOURCE_ROOT/$(basename "$scene_dir")"
  done

  python -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli index \
    --result-root "$INDEX_SOURCE_ROOT" \
    --out "$INDEX_PATH" \
    --snapshot-policy all \
    --allow-missing-png \
    --no-validate-npz
fi

EPISODE_ARGS=()
for episode_uid in "$@"; do
  EPISODE_ARGS+=(--episode-uid "$episode_uid")
done

exec python -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli annotate-last \
  --index "$INDEX_PATH" \
  --annotation-dir "$ANNOTATION_DIR" \
  --gt-dir "$GT_DIR" \
  --line-width-cells 3 \
  --preclose-radius-cells 1 \
  --figure-scale "$FIGURE_SCALE" \
  --source-view segmentation \
  --skip-approved \
  --min-room-area-m2 0.3 \
  --cell-size-m 0.05 \
  "${EPISODE_ARGS[@]}"
