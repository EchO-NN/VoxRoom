#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${VOXROOM_ABLATION_ROOT:-/media/echo/data/voxroom_ablation_20260828}
CODE=$ROOT/code
TRAIN=$ROOT/training
EVAL=$ROOT/evaluation
PY=${VOXROOM_ABLATION_PYTHON:-/home/echo/miniforge3/envs/RoboTwin/bin/python}
IA=/media/echo/data/voxroom_roomseg_evaluation/interioragent_all_available_gt_20260828
GR=/media/echo/data/voxroom_roomseg_evaluation/grscene_all_available_gt_20260817
STATUS=$ROOT/vertical_retrain_dense_status.log

log_status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"
}

wait_for_training() {
  local variant=$1
  local directory=$TRAIN/$variant
  local pid_file=$directory/train.pid
  if [[ ! -f "$pid_file" ]]; then
    log_status "ERROR missing training pid: $variant"
    return 1
  fi
  local pid
  pid=$(<"$pid_file")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  for name in training_summary.json training_history.json training_loss.png best.pt; do
    if [[ ! -s "$directory/$name" ]]; then
      log_status "ERROR incomplete training artifact: $variant $name"
      return 1
    fi
  done
  log_status "training complete: $variant"
}

launch_replay() {
  local dataset=$1
  local variant=$2
  local checkpoint=$3
  local base out
  if [[ "$dataset" == interioragent ]]; then base=$IA; else base=$GR; fi
  out=$EVAL/${dataset}_${variant}
  if [[ -s "$out/metrics/summary.json" ]]; then
    log_status "reusing evaluated replay: $dataset $variant"
    return
  fi
  if [[ -f "$out/replay.pid" ]] && kill -0 "$(<"$out/replay.pid")" 2>/dev/null; then
    log_status "reusing active replay: $dataset $variant pid=$(<"$out/replay.pid")"
    return
  fi
  mkdir -p "$out"
  nohup env PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u "$CODE/scripts/replay_voxroom_ablation.py" \
    --index "$base/index_voxroom.json" --annotation-dir "$base/annotations" \
    --out-dir "$out" --variant "$variant" --config "$CODE/configs/voxroom_online.yaml" \
    --checkpoint "$checkpoint" --device cuda:0 --inference-batch-size 32 \
    >"$out/replay.log" 2>&1 < /dev/null &
  echo $! >"$out/replay.pid"
  log_status "launched replay: $dataset $variant pid=$!"
}

wait_replay() {
  local dataset=$1
  local variant=$2
  local out=$EVAL/${dataset}_${variant}
  if [[ -s "$out/metrics/summary.json" ]]; then
    return
  fi
  local pid
  pid=$(<"$out/replay.pid")
  while kill -0 "$pid" 2>/dev/null; do sleep 20; done
  "$PY" - "$out/prediction_index.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if not data.get("complete") or int(data.get("processed", -1)) != int(data.get("snapshot_count", -2)):
    raise SystemExit("incomplete replay: " + sys.argv[1])
PY
  log_status "replay complete: $dataset $variant"
}

evaluate_one() {
  local dataset=$1
  local variant=$2
  local base out
  if [[ "$dataset" == interioragent ]]; then base=$IA; else base=$GR; fi
  out=$EVAL/${dataset}_${variant}
  if [[ -s "$out/metrics/summary.json" ]]; then
    log_status "reusing metrics: $dataset $variant"
    return
  fi
  rm -rf "$out/metrics.tmp"
  "$PY" -u "$CODE/scripts/evaluate_voxroom_ablation.py" \
    --source-index "$base/index_voxroom.json" \
    --prediction-index "$out/prediction_index.json" \
    --annotation-dir "$base/annotations" --gt-dir "$base/final_gt" \
    --exclusion-manifest "$base/training_scene_exclusion_manifest.json" \
    --out-dir "$out/metrics.tmp" --dataset "$dataset" >"$out/evaluate.log" 2>&1
  rm -rf "$out/metrics"
  mv "$out/metrics.tmp" "$out/metrics"
  log_status "evaluated: $dataset $variant"
}

run_variant() {
  local variant=$1
  wait_for_training "$variant"
  for dataset in interioragent grscene; do
    launch_replay "$dataset" "$variant" "$TRAIN/$variant/best.pt"
  done
  for dataset in interioragent grscene; do
    wait_replay "$dataset" "$variant"
    evaluate_one "$dataset" "$variant"
  done
}

log_status "vertical retrain and all-free pipeline started"
run_variant vertical_full_retrain
run_variant all_vertical_free_cells

OUT=$ROOT/vertical_retrain_dense_report
rm -rf "$OUT.tmp"
"$PY" -u "$CODE/scripts/aggregate_vertical_retrain_dense.py" \
  --evaluation-root "$EVAL" --training-root "$TRAIN" --out-dir "$OUT.tmp" \
  >"$ROOT/vertical_retrain_dense_aggregate.log" 2>&1
rm -rf "$OUT"
mv "$OUT.tmp" "$OUT"
log_status "pipeline complete: $OUT/report.md"
