#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLECTION_ROOT="${COLLECTION_ROOT:-/media/echo/data/voxroom_door_seed_union_voxel_milestones_20260811/annotation_final_scenes}"
RUN_ROOT="${1:?usage: launch_approved_union_door_seed_training.sh RUN_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/echo/miniforge3/envs/sgnav-isaac/bin/python}"
DATASET_DIR="$RUN_ROOT/dataset"
MODEL_DIR="$RUN_ROOT/model_vertical"
HISTORY_PATH="$MODEL_DIR/training_history.json"
MONITOR_LOG="$RUN_ROOT/loss_monitor.log"
TRAIN_LOG="$RUN_ROOT/train_vertical.log"
BUILD_LOG="$RUN_ROOT/build_dataset.log"

mkdir -p "$RUN_ROOT" "$DATASET_DIR" "$MODEL_DIR"
cd "$REPO_ROOT"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "collection_root=$COLLECTION_ROOT"
  echo "dataset_dir=$DATASET_DIR"
  echo "model_dir=$MODEL_DIR"
  echo "context_source=vertical"
  echo "local_patch_size=19"
  echo "context_patch_size=41"
  echo "augmentation_rotations=0,90,180,270"
  echo "augmentation_mirror_lr_once=true"
  echo "sampler_order=snapshot_bucketed_grouped_coordinate"
  echo "snapshot_cache_size=8"
  echo "threshold_selection_mode=fixed"
  echo "fixed_keep_threshold=0.5"
  echo "precision=bfloat16"
  echo "batch_size=128"
} > "$RUN_ROOT/run_manifest.txt"

if [[ "${SKIP_DATASET_BUILD:-0}" != "1" ]]; then
  PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
    voxroom_online/isaac_runtime/scripts/build_door_seed_dataset.py \
    --collection-root "$COLLECTION_ROOT" \
    --out-dir "$DATASET_DIR" \
    --config configs/voxroom_online.yaml \
    > "$BUILD_LOG" 2>&1
else
  echo "dataset build skipped; reusing verified indexes" >> "$BUILD_LOG"
fi

env \
  DISPLAY="${DISPLAY:-:0}" \
  XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}" \
  MPLBACKEND=TkAgg \
  PYTHONPATH="$REPO_ROOT" \
  "$PYTHON_BIN" voxroom_online/isaac_runtime/scripts/monitor_door_seed_training.py \
  --history "$HISTORY_PATH" \
  --refresh-seconds 1 \
  > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!

cleanup_monitor() {
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup_monitor EXIT

set +e
PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" \
  voxroom_online/isaac_runtime/scripts/train_door_seed_classifier.py \
  --index "$DATASET_DIR/dataset_vertical.jsonl" \
  --out-dir "$MODEL_DIR" \
  --context-source vertical \
  --device cuda:0 \
  --precision bfloat16 \
  --batch-size 128 \
  --max-epochs 200 \
  --patience 10 \
  --early-stopping-metric validation_score \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --threshold-selection-mode fixed \
  --fixed-keep-threshold 0.5 \
  --snapshot-cache-size 8 \
  --seed 0 \
  --train-rotation-degrees 0,90,180,270 \
  --train-mirror-lr-once \
  > "$TRAIN_LOG" 2>&1
TRAIN_STATUS=$?
set -e

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "train_exit_status=$TRAIN_STATUS"
} >> "$RUN_ROOT/run_manifest.txt"
exit "$TRAIN_STATUS"
