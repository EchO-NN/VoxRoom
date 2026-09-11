#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${VOXROOM_ABLATION_ROOT:-/media/echo/data/voxroom_ablation_20260828}
CODE=$ROOT/code
TRAIN=$ROOT/training
EVAL=$ROOT/evaluation
PY=${VOXROOM_ABLATION_PYTHON:-/home/echo/miniforge3/envs/RoboTwin/bin/python}
IA=/media/echo/data/voxroom_roomseg_evaluation/interioragent_all_available_gt_20260828
GR=/media/echo/data/voxroom_roomseg_evaluation/grscene_all_available_gt_20260817
RUN=nav_no_clearance_full_f1_retrain
REPLAY_VARIANT=nav_no_clearance_full
STATUS=$ROOT/nav_f1_retrain_status.log

log_status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"
}

wait_for_training() {
  local directory=$TRAIN/$RUN
  local pid
  pid=$(<"$directory/train.pid")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  for name in training_summary.json training_history.json training_loss.png best.pt; do
    if [[ ! -s "$directory/$name" ]]; then
      log_status "ERROR incomplete training artifact: $name"
      return 1
    fi
  done
  log_status "training complete: $RUN"
}

launch_replay() {
  local dataset=$1
  local base out
  if [[ "$dataset" == interioragent ]]; then base=$IA; else base=$GR; fi
  out=$EVAL/${dataset}_${RUN}
  mkdir -p "$out"
  nohup env PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u "$CODE/scripts/replay_voxroom_ablation.py" \
    --index "$base/index_voxroom.json" --annotation-dir "$base/annotations" \
    --out-dir "$out" --variant "$REPLAY_VARIANT" \
    --config "$CODE/configs/voxroom_online.yaml" \
    --checkpoint "$TRAIN/$RUN/best.pt" --device cuda:0 --inference-batch-size 32 \
    >"$out/replay.log" 2>&1 < /dev/null &
  echo $! >"$out/replay.pid"
  log_status "launched replay: $dataset pid=$!"
}

wait_replay() {
  local dataset=$1
  local out=$EVAL/${dataset}_${RUN}
  local pid
  pid=$(<"$out/replay.pid")
  while kill -0 "$pid" 2>/dev/null; do sleep 20; done
  "$PY" - "$out/prediction_index.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
if not data.get("complete"):
    raise SystemExit("replay did not mark itself complete: " + sys.argv[1])
if int(data.get("processed", -1)) != int(data.get("snapshot_count", -2)):
    raise SystemExit("replay snapshot count mismatch: " + sys.argv[1])
PY
  log_status "replay complete: $dataset"
}

evaluate_one() {
  local dataset=$1
  local base out
  if [[ "$dataset" == interioragent ]]; then base=$IA; else base=$GR; fi
  out=$EVAL/${dataset}_${RUN}
  mkdir -p "$out/metrics"
  "$PY" -u "$CODE/scripts/evaluate_voxroom_ablation.py" \
    --source-index "$base/index_voxroom.json" \
    --prediction-index "$out/prediction_index.json" \
    --annotation-dir "$base/annotations" --gt-dir "$base/final_gt" \
    --exclusion-manifest "$base/training_scene_exclusion_manifest.json" \
    --out-dir "$out/metrics" --dataset "$dataset" >"$out/evaluate.log" 2>&1
  test -s "$out/metrics/summary.json"
  log_status "evaluated: $dataset"
}

log_status "Nav-no-clearance F1-selected retrain pipeline started"
wait_for_training
for dataset in interioragent grscene; do launch_replay "$dataset"; done
for dataset in interioragent grscene; do wait_replay "$dataset"; done
for dataset in interioragent grscene; do evaluate_one "$dataset"; done
log_status "pipeline complete"
