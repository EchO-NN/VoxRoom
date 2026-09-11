#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${VOXROOM_ABLATION_ROOT:-/media/echo/data/voxroom_ablation_20260828}
PREVIEW=${VOXROOM_ABLATION_PREVIEW:-$ROOT/previews/current_best_20260829_1600}
CODE=$ROOT/code
TRAIN=$ROOT/training
PY=${VOXROOM_ABLATION_PYTHON:-/home/echo/miniforge3/envs/RoboTwin/bin/python}
IA=/media/echo/data/voxroom_roomseg_evaluation/interioragent_all_available_gt_20260828
GR=/media/echo/data/voxroom_roomseg_evaluation/grscene_all_available_gt_20260817
STATUS=$PREVIEW/status.log

mkdir -p "$PREVIEW/checkpoints" "$PREVIEW/evaluation"

log_status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"
}

training_pids=()
for variant in nav_no_clearance_full vertical_3d_only; do
  pid=$(<"$TRAIN/$variant/train.pid")
  kill -0 "$pid"
  kill -STOP "$pid"
  training_pids+=("$pid")
  log_status "paused training: $variant pid=$pid"
done

resume_training() {
  local rc=$?
  trap - EXIT INT TERM
  for pid in "${training_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -CONT "$pid"
      log_status "resumed training: pid=$pid"
    fi
  done
  if ((rc == 0)); then
    log_status "preview complete: $PREVIEW"
  else
    log_status "preview failed: exit=$rc"
  fi
  exit "$rc"
}
trap resume_training EXIT INT TERM

for variant in nav_no_clearance_full vertical_2d_only vertical_3d_only; do
  cp --reflink=auto "$TRAIN/$variant/best.pt" "$PREVIEW/checkpoints/$variant.pt"
  sha256sum "$PREVIEW/checkpoints/$variant.pt" >>"$PREVIEW/checkpoints/SHA256SUMS"
done

launch_replay() {
  local dataset=$1
  local variant=$2
  local base checkpoint out
  if [[ "$dataset" == interioragent ]]; then
    base=$IA
  else
    base=$GR
  fi
  checkpoint=$PREVIEW/checkpoints/$variant.pt
  out=$PREVIEW/evaluation/${dataset}_${variant}
  mkdir -p "$out"
  env PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u "$CODE/scripts/replay_voxroom_ablation.py" \
    --index "$base/index_voxroom.json" --annotation-dir "$base/annotations" \
    --out-dir "$out" --variant "$variant" \
    --config "$CODE/configs/voxroom_online.yaml" \
    --checkpoint "$checkpoint" --device cuda:0 --inference-batch-size 32 \
    >"$out/replay.log" 2>&1 &
  local pid=$!
  echo "$pid" >"$out/replay.pid"
  replay_pids+=("$pid")
  replay_datasets+=("$dataset")
  replay_variants+=("$variant")
  log_status "launched replay: $dataset $variant pid=$pid"
}

evaluate_one() {
  local dataset=$1
  local variant=$2
  local base out
  if [[ "$dataset" == interioragent ]]; then
    base=$IA
  else
    base=$GR
  fi
  out=$PREVIEW/evaluation/${dataset}_${variant}
  "$PY" - "$out/prediction_index.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
if not data.get("complete") or int(data.get("processed", -1)) != int(data.get("snapshot_count", -2)):
    raise SystemExit("incomplete replay: " + sys.argv[1])
PY
  "$PY" -u "$CODE/scripts/evaluate_voxroom_ablation.py" \
    --source-index "$base/index_voxroom.json" \
    --prediction-index "$out/prediction_index.json" \
    --annotation-dir "$base/annotations" --gt-dir "$base/final_gt" \
    --exclusion-manifest "$base/training_scene_exclusion_manifest.json" \
    --out-dir "$out/metrics" --dataset "$dataset" \
    >"$out/evaluate.log" 2>&1
  log_status "evaluated: $dataset $variant"
}

replay_pids=()
replay_datasets=()
replay_variants=()
for variant in nav_no_clearance_full vertical_2d_only vertical_3d_only; do
  launch_replay interioragent "$variant"
  launch_replay grscene "$variant"
done

failed=0
for index in "${!replay_pids[@]}"; do
  pid=${replay_pids[$index]}
  dataset=${replay_datasets[$index]}
  variant=${replay_variants[$index]}
  if wait "$pid"; then
    if ! evaluate_one "$dataset" "$variant"; then
      failed=1
      log_status "evaluation failed: $dataset $variant"
    fi
  else
    failed=1
    log_status "replay failed: $dataset $variant"
  fi
done

exit "$failed"
