#!/usr/bin/env bash
set -Eeuo pipefail

VOXROOM_ROOT=${VOXROOM_ABLATION_ROOT:-/media/echo/data/voxroom_ablation_20260828}
VOXROOM_CODE=$VOXROOM_ROOT/code
VOXROOM_TRAIN=$VOXROOM_ROOT/training
VOXROOM_PYTHON=${VOXROOM_ABLATION_PYTHON:-/home/echo/miniforge3/envs/RoboTwin/bin/python}
VOXROOM_DATASET=${VOXROOM_NAV_DATASET:-/media/echo/data/voxroom_door_seed_union_training_runs/approved23_rot4_mirror1_vertical_20260813_1622/dataset/dataset_nav.jsonl}
VOXROOM_RUN=nav_no_clearance_full_f1_retrain
VOXROOM_OUTPUT=$VOXROOM_TRAIN/$VOXROOM_RUN
VOXROOM_ARCHIVE=$VOXROOM_TRAIN/${VOXROOM_RUN}_interrupted_epoch9_20260831_1533
VOXROOM_SERVICE=voxroom-nav-f1-retrain-recovery

test -d "$VOXROOM_CODE"
test -s "$VOXROOM_DATASET"
test -x "$VOXROOM_PYTHON"
if systemctl --user is-active --quiet "$VOXROOM_SERVICE.service"; then
  echo "training service is already active: $VOXROOM_SERVICE" >&2
  exit 1
fi
if [[ -e "$VOXROOM_ARCHIVE" ]]; then
  echo "interrupted-run archive already exists: $VOXROOM_ARCHIVE" >&2
  exit 1
fi
if [[ -d "$VOXROOM_OUTPUT" ]]; then
  mv "$VOXROOM_OUTPUT" "$VOXROOM_ARCHIVE"
fi
mkdir -p "$VOXROOM_OUTPUT"

systemd-run --user \
  --unit="$VOXROOM_SERVICE" \
  --collect \
  --property=MemoryHigh=28G \
  --property=MemoryMax=34G \
  --working-directory="$VOXROOM_CODE" \
  /bin/bash -lc "exec env \
PYTHONPATH='$VOXROOM_CODE' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
'$VOXROOM_PYTHON' -u \
'$VOXROOM_CODE/voxroom_online/isaac_runtime/scripts/train_door_seed_classifier.py' \
--index '$VOXROOM_DATASET' \
--out-dir '$VOXROOM_OUTPUT' \
--context-source nav \
--device cuda:0 \
--precision bfloat16 \
--batch-size 64 \
--num-workers 0 \
--max-epochs 50 \
--patience 10 \
--early-stopping-metric validation_score \
--checkpoint-selection-mode fixed_f1 \
--learning-rate 3e-4 \
--weight-decay 1e-4 \
--threshold-selection-mode fixed \
--fixed-keep-threshold 0.5 \
--max-pos-weight 10 \
--snapshot-cache-size 4 \
--seed 0 \
--train-rotation-degrees 0,90,180,270 \
--train-mirror-lr-once \
> '$VOXROOM_OUTPUT/train.log' 2>&1"

sleep 2
VOXROOM_TRAIN_PID=$(systemctl --user show "$VOXROOM_SERVICE.service" --property=MainPID --value)
if [[ ! "$VOXROOM_TRAIN_PID" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "$VOXROOM_TRAIN_PID" 2>/dev/null; then
  systemctl --user status "$VOXROOM_SERVICE.service" --no-pager >&2 || true
  exit 1
fi
printf '%s\n' "$VOXROOM_TRAIN_PID" > "$VOXROOM_OUTPUT/train.pid"

nohup env VOXROOM_ABLATION_ROOT="$VOXROOM_ROOT" VOXROOM_ABLATION_PYTHON="$VOXROOM_PYTHON" \
  bash "$VOXROOM_CODE/scripts/run_nav_f1_retrain_pipeline.sh" \
  > "$VOXROOM_ROOT/nav_f1_retrain_supervisor_recovery.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$VOXROOM_ROOT/nav_f1_retrain_supervisor_recovery.pid"

nohup env DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority MPLBACKEND=TkAgg \
  PYTHONPATH="$VOXROOM_CODE" \
  "$VOXROOM_PYTHON" -u \
  "$VOXROOM_CODE/voxroom_online/isaac_runtime/scripts/monitor_door_seed_training.py" \
  --history "$VOXROOM_OUTPUT/training_history.json" --refresh-seconds 1 \
  > "$VOXROOM_OUTPUT/loss_monitor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$VOXROOM_OUTPUT/loss_monitor.pid"

printf 'training_pid=%s\nsupervisor_pid=%s\nmonitor_pid=%s\narchive=%s\n' \
  "$VOXROOM_TRAIN_PID" \
  "$(<"$VOXROOM_ROOT/nav_f1_retrain_supervisor_recovery.pid")" \
  "$(<"$VOXROOM_OUTPUT/loss_monitor.pid")" \
  "$VOXROOM_ARCHIVE"
