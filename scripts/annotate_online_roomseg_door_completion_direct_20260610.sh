#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ROOT="${RUN_ROOT:-对比实验/outputs/final_latest_20260610_voronoi_width_jump_axis2_interval_per_free_parallel020_150_no_corner_stateful_memory_freeze}"
RESULT_ROOT="${RESULT_ROOT:-$RUN_ROOT/stateful_replay}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval_online_roomseg_door_completion_direct}"
INDEX_PATH="$EVAL_DIR/index_all.json"
ANNOTATION_DIR="$EVAL_DIR/annotations"
GT_DIR="$EVAL_DIR/gt_final"
FIGURE_SCALE="${FIGURE_SCALE:-1.25}"
REBUILD_INDEX="${REBUILD_INDEX:-0}"

if [[ ! -d "$RESULT_ROOT" ]]; then
  echo "missing RESULT_ROOT: $RESULT_ROOT" >&2
  exit 2
fi

mkdir -p "$EVAL_DIR"

if [[ "$REBUILD_INDEX" == "1" || ! -f "$INDEX_PATH" ]]; then
  python - <<'PY'
from pathlib import Path
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import build_index, write_index

run_root = Path("对比实验/outputs/final_latest_20260610_voronoi_width_jump_axis2_interval_per_free_parallel020_150_no_corner_stateful_memory_freeze")
eval_root = run_root / "eval_online_roomseg_door_completion_direct"
result_root = Path(__import__("os").environ.get("RESULT_ROOT", str(run_root / "stateful_replay")))
index = build_index([result_root], snapshot_policy="all", allow_missing_png=True, validate_npz=False)
write_index(index, eval_root / "index_all.json")
print("indexed %d episode(s), %d snapshot(s) -> %s" % (
    len(index.get("episodes", [])),
    sum(len(ep.get("snapshots", [])) for ep in index.get("episodes", [])),
    eval_root / "index_all.json",
))
PY
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
