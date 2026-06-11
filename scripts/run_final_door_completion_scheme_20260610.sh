#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FINAL_SCHEME_NAME="endpoint1x1_snapshot_door_completion_check_and_raw_seed_views_no_clearance_parallel_prune_shorter_occ3_same_line_gap28_contig_gap2_unknown10_seedband2_surfaceocc1_ceiling095_stateful_memory_freeze_20260610_full"
FINAL_RUN_NAME="final_latest_20260610_voronoi_width_jump_axis2_interval_per_free_parallel020_150_no_corner_stateful_memory_freeze"

CONFIG="${CONFIG:-configs/voxroom_online.yaml}"
RAW_INPUT_ROOT="${RAW_INPUT_ROOT:-${INPUT_ROOT:-/home/echo/VoxRoom-Online-exp/radius005_robot005_endpoint1x1_7jobs4threads_all_scenes}}"
VIEW_ROOT="${VIEW_ROOT:-对比实验/outputs/$FINAL_SCHEME_NAME}"
RUN_ROOT="${RUN_ROOT:-对比实验/outputs/$FINAL_RUN_NAME}"
STATEFUL_REPLAY_ROOT="${STATEFUL_REPLAY_ROOT:-$RUN_ROOT/stateful_replay}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/eval_online_roomseg_door_completion_direct}"
INDEX_PATH="${INDEX_PATH:-$EVAL_DIR/index_all.json}"
WORKERS="${WORKERS:-4}"
RUN_METRICS="${RUN_METRICS:-0}"

if [[ ! -d "$RAW_INPUT_ROOT" ]]; then
  echo "[final-door-completion] missing RAW_INPUT_ROOT: $RAW_INPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$VIEW_ROOT" "$STATEFUL_REPLAY_ROOT" "$EVAL_DIR"

echo "[final-door-completion] final scheme: $FINAL_SCHEME_NAME"
echo "[final-door-completion] input snapshots: $RAW_INPUT_ROOT"
echo "[final-door-completion] check/raw-seed PNGs: $VIEW_ROOT"
echo "[final-door-completion] room-mask NPZ snapshots: $STATEFUL_REPLAY_ROOT"
echo "[final-door-completion] config: $CONFIG"

python scripts/render_final_door_completion_snapshots.py \
  --input-root "$RAW_INPUT_ROOT" \
  --output-root "$VIEW_ROOT" \
  --npz-output-root "$STATEFUL_REPLAY_ROOT" \
  --config "$CONFIG" \
  --stateful-by-scene \
  --workers "$WORKERS" \
  "$@"

python - <<'PY'
import os
from pathlib import Path
from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_index import build_index, write_index

result_root = Path(os.environ["STATEFUL_REPLAY_ROOT"])
eval_dir = Path(os.environ["EVAL_DIR"])
index_path = Path(os.environ["INDEX_PATH"])
index = build_index([result_root], snapshot_policy="all", allow_missing_png=True, validate_npz=False)
write_index(index, index_path)
print(
    "[final-door-completion] indexed %d episode(s), %d snapshot(s) -> %s"
    % (
        len(index.get("episodes", [])),
        sum(len(ep.get("snapshots", [])) for ep in index.get("episodes", [])),
        index_path,
    )
)
PY

if [[ "$RUN_METRICS" == "1" ]]; then
  GT_DIR="${GT_DIR:-$EVAL_DIR/gt_final}"
  STEP_GT_DIR="${STEP_GT_DIR:-$EVAL_DIR/gt_by_step}"
  METRICS_DIR="${METRICS_DIR:-$EVAL_DIR/metrics_geometric_only}"
  if [[ ! -d "$GT_DIR" ]]; then
    echo "[final-door-completion] RUN_METRICS=1 but GT_DIR is missing: $GT_DIR" >&2
    exit 3
  fi
  python -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli backproject \
    --index "$INDEX_PATH" \
    --gt-dir "$GT_DIR" \
    --step-gt-dir "$STEP_GT_DIR"
  python -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli compute \
    --index "$INDEX_PATH" \
    --step-gt-dir "$STEP_GT_DIR" \
    --out-dir "$METRICS_DIR" \
    --no-require-csr \
    --min-room-area-m2 0.3 \
    --cell-size-m 0.05
fi
