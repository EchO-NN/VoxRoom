#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${VOXROOM_ABLATION_ROOT:-/media/echo/data/voxroom_ablation_20260828}
CODE=$ROOT/code
TRAIN=$ROOT/training
PY=${VOXROOM_ABLATION_PYTHON:-/home/echo/miniforge3/envs/RoboTwin/bin/python}
DATASET=/media/echo/data/voxroom_door_seed_union_training_runs/approved23_rot4_mirror1_vertical_20260813_1622/dataset/dataset_vertical.jsonl
RUN=vertical_full_retrain_f1_finetune_epoch14_lr1e4
OUTPUT=$TRAIN/$RUN
BASELINE=$TRAIN/vertical_full_retrain/best.pt
SERVICE=voxroom-vertical-f1-finetune

test -d "$CODE"
test -s "$DATASET"
test -s "$BASELINE"
test -x "$PY"
if systemctl --user is-active --quiet "$SERVICE.service"; then
  echo "training service is already active: $SERVICE" >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "fine-tune output already exists: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
cp --reflink=auto "$BASELINE" "$OUTPUT/baseline_epoch14.pt"

systemd-run --user \
  --unit="$SERVICE" \
  --collect \
  --property=MemoryHigh=28G \
  --property=MemoryMax=34G \
  --working-directory="$CODE" \
  /bin/bash -lc "exec env \
PYTHONPATH='$CODE' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
'$PY' -u \
'$CODE/voxroom_online/isaac_runtime/scripts/train_door_seed_classifier.py' \
--index '$DATASET' \
--out-dir '$OUTPUT' \
--init-checkpoint '$OUTPUT/baseline_epoch14.pt' \
--context-source vertical \
--device cuda:0 \
--precision bfloat16 \
--batch-size 64 \
--num-workers 0 \
--max-epochs 1000 \
--patience 10 \
--early-stopping-metric validation_score \
--checkpoint-selection-mode fixed_f1 \
--learning-rate 1e-4 \
--weight-decay 1e-4 \
--threshold-selection-mode fixed \
--fixed-keep-threshold 0.5 \
--max-pos-weight 10 \
--snapshot-cache-size 4 \
--seed 17 \
--train-rotation-degrees 0,90,180,270 \
--train-mirror-lr-once \
> '$OUTPUT/train.log' 2>&1"

sleep 2
TRAIN_PID=$(systemctl --user show "$SERVICE.service" --property=MainPID --value)
if [[ ! "$TRAIN_PID" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  systemctl --user status "$SERVICE.service" --no-pager >&2 || true
  exit 1
fi
printf '%s\n' "$TRAIN_PID" >"$OUTPUT/train.pid"

nohup env VOXROOM_ABLATION_ROOT="$ROOT" VOXROOM_ABLATION_PYTHON="$PY" \
  bash "$CODE/scripts/run_vertical_f1_finetune_pipeline.sh" \
  >"$ROOT/vertical_f1_finetune_supervisor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"$ROOT/vertical_f1_finetune_supervisor.pid"

nohup env DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority MPLBACKEND=TkAgg \
  PYTHONPATH="$CODE" \
  "$PY" -u \
  "$CODE/voxroom_online/isaac_runtime/scripts/monitor_door_seed_training.py" \
  --history "$OUTPUT/training_history.json" --refresh-seconds 1 \
  >"$OUTPUT/loss_monitor.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"$OUTPUT/loss_monitor.pid"

printf 'training_pid=%s\nsupervisor_pid=%s\nmonitor_pid=%s\noutput=%s\n' \
  "$TRAIN_PID" \
  "$(<"$ROOT/vertical_f1_finetune_supervisor.pid")" \
  "$(<"$OUTPUT/loss_monitor.pid")" \
  "$OUTPUT"
