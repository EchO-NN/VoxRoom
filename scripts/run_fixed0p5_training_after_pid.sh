#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?usage: run_fixed0p5_training_after_pid.sh WAIT_PID RUN_ROOT}"
RUN_ROOT="${2:?usage: run_fixed0p5_training_after_pid.sh WAIT_PID RUN_ROOT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/echo/miniforge3/envs/sgnav-isaac/bin/python}"
INDEX_PATH="$RUN_ROOT/dataset/dataset_vertical.jsonl"
MODEL_DIR="$RUN_ROOT/model_vertical_fixed0p5"
TRAIN_LOG="$RUN_ROOT/train_vertical_fixed0p5.log"
MONITOR_LOG="$RUN_ROOT/loss_monitor_fixed0p5.log"
STATUS_FILE="$RUN_ROOT/fixed0p5_followup_status.txt"

{
  echo "scheduled_at=$(date --iso-8601=seconds)"
  echo "waiting_for_pid=$WAIT_PID"
  echo "threshold_selection_mode=fixed"
  echo "fixed_keep_threshold=0.5"
  echo "target_recall_used=false"
} > "$STATUS_FILE"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

mkdir -p "$MODEL_DIR"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "model_dir=$MODEL_DIR"
} >> "$STATUS_FILE"

cd "$REPO_ROOT"
env \
  DISPLAY="${DISPLAY:-:0}" \
  XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}" \
  MPLBACKEND=TkAgg \
  PYTHONPATH="$REPO_ROOT" \
  "$PYTHON_BIN" voxroom_online/isaac_runtime/scripts/monitor_door_seed_training.py \
  --history "$MODEL_DIR/training_history.json" \
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
  --index "$INDEX_PATH" \
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
} >> "$STATUS_FILE"
exit "$TRAIN_STATUS"
