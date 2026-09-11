#!/usr/bin/env bash
set -euo pipefail

ROOT=${VOXROOM_ABLATION_ROOT:-/media/echo/data/voxroom_ablation_20260828}
CODE=$ROOT/code
TRAIN=$ROOT/training
EVAL=$ROOT/evaluation
PY=${VOXROOM_ABLATION_PYTHON:-/home/echo/miniforge3/envs/RoboTwin/bin/python}
IA=/media/echo/data/voxroom_roomseg_evaluation/interioragent_all_available_gt_20260828
GR=/media/echo/data/voxroom_roomseg_evaluation/grscene_all_available_gt_20260817
BASE_CKPT=/media/echo/data/voxroom_door_seed_union_training_runs/approved23_rot4_mirror1_vertical_20260813_1622/model_vertical_fixed0p5/best.pt
STATUS=$ROOT/pipeline_status.log

log_status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"
}

wait_for_pid_file() {
  local pid_file=$1
  local label=$2
  while true; do
    if [[ ! -f "$pid_file" ]]; then
      log_status "ERROR missing pid file: $label $pid_file"
      return 1
    fi
    local pid
    pid=$(<"$pid_file")
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 30
  done
  log_status "process exited: $label"
}

assert_replay_complete() {
  local out=$1
  "$PY" - "$out/prediction_index.json" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
if not d.get("complete") or int(d.get("processed",-1)) != int(d.get("snapshot_count",-2)):
    raise SystemExit("incomplete replay: %s" % p)
PY
}

evaluate_one() {
  local dataset=$1
  local variant=$2
  local base index annotations gt exclusion
  if [[ "$dataset" == interioragent ]]; then
    base=$IA
    index=$IA/index_voxroom.json
  else
    base=$GR
    index=$GR/index_voxroom.json
  fi
  annotations=$base/annotations
  gt=$base/final_gt
  exclusion=$base/training_scene_exclusion_manifest.json
  local out=$EVAL/${dataset}_${variant}
  assert_replay_complete "$out"
  rm -rf "$out/metrics.tmp"
  "$PY" -u "$CODE/scripts/evaluate_voxroom_ablation.py" \
    --source-index "$index" \
    --prediction-index "$out/prediction_index.json" \
    --annotation-dir "$annotations" \
    --gt-dir "$gt" \
    --exclusion-manifest "$exclusion" \
    --out-dir "$out/metrics.tmp" \
    --dataset "$dataset" >"$out/evaluate.log" 2>&1
  rm -rf "$out/metrics"
  mv "$out/metrics.tmp" "$out/metrics"
  log_status "evaluated: $dataset $variant"
}

launch_replay() {
  local dataset=$1
  local variant=$2
  local checkpoint=$3
  local batch=$4
  local base index annotations
  if [[ "$dataset" == interioragent ]]; then
    base=$IA
    index=$IA/index_voxroom.json
  else
    base=$GR
    index=$GR/index_voxroom.json
  fi
  annotations=$base/annotations
  local out=$EVAL/${dataset}_${variant}
  mkdir -p "$out"
  local checkpoint_args=()
  if [[ -n "$checkpoint" ]]; then
    checkpoint_args=(--checkpoint "$checkpoint")
  fi
  nohup env PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u "$CODE/scripts/replay_voxroom_ablation.py" \
    --index "$index" --annotation-dir "$annotations" --out-dir "$out" \
    --variant "$variant" --config "$CODE/configs/voxroom_online.yaml" \
    --inference-batch-size "$batch" "${checkpoint_args[@]}" \
    >"$out/replay.log" 2>&1 < /dev/null &
  echo $! >"$out/replay.pid"
  log_status "launched replay: $dataset $variant pid=$!"
}

wait_and_evaluate_pair() {
  local variant=$1
  wait_for_pid_file "$EVAL/interioragent_${variant}/replay.pid" "interioragent $variant"
  wait_for_pid_file "$EVAL/grscene_${variant}/replay.pid" "grscene $variant"
  evaluate_one interioragent "$variant"
  evaluate_one grscene "$variant"
}

log_status "pipeline supervisor started"

for dataset in interioragent grscene; do
  if [[ "$dataset" == interioragent ]]; then
    base=$IA
  else
    base=$GR
  fi
  original_out=$EVAL/${dataset}_production_full_model_original
  mkdir -p "$original_out"
  "$PY" -u "$CODE/scripts/index_original_voxroom_predictions.py" \
    --source-index "$base/index_voxroom.json" --annotation-dir "$base/annotations" \
    --out-dir "$original_out" >"$original_out/index.log" 2>&1
  evaluate_one "$dataset" production_full_model_original
done

for variant in no_tvars_raw_seed no_voxel_raw_seed no_neural_filter saved_full_model_control; do
  wait_and_evaluate_pair "$variant"
done

for variant in nav_no_clearance_full vertical_2d_only vertical_3d_only; do
  wait_for_pid_file "$TRAIN/$variant/train.pid" "training $variant"
  test -s "$TRAIN/$variant/training_summary.json"
  test -s "$TRAIN/$variant/best.pt"
  test -s "$TRAIN/$variant/training_loss.png"
  log_status "training complete: $variant"
done

for variant in nav_no_clearance_full vertical_2d_only vertical_3d_only; do
  launch_replay interioragent "$variant" "$TRAIN/$variant/best.pt" 32
  launch_replay grscene "$variant" "$TRAIN/$variant/best.pt" 32
done
DENSE_OUT=$TRAIN/all_vertical_free_cells
mkdir -p "$DENSE_OUT"
if [[ -f "$DENSE_OUT/train.pid" ]] && kill -0 "$(<"$DENSE_OUT/train.pid")" 2>/dev/null; then
  log_status "reusing active training: all_vertical_free_cells pid=$(<"$DENSE_OUT/train.pid")"
elif [[ -s "$DENSE_OUT/training_summary.json" && -s "$DENSE_OUT/best.pt" && -s "$DENSE_OUT/training_loss.png" ]]; then
  log_status "reusing completed training: all_vertical_free_cells"
else
  nohup env PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u "$CODE/voxroom_online/isaac_runtime/scripts/train_door_seed_classifier.py" \
    --index "$TRAIN/all_free_dataset/dataset_vertical_all_free_unique.jsonl" \
    --out-dir "$DENSE_OUT" --context-source vertical --device cuda:0 \
    --precision bfloat16 --batch-size 128 --num-workers 4 --max-epochs 50 \
    --patience 10 --checkpoint-selection-mode fixed_f1 \
    --threshold-selection-mode fixed --fixed-keep-threshold 0.5 \
    --snapshot-cache-size 2 --seed 0 --train-rotation-degrees 0,90,180,270 \
    --train-mirror-lr-once >"$DENSE_OUT/train.log" 2>&1 < /dev/null &
  echo $! >"$DENSE_OUT/train.pid"
  log_status "launched training: all_vertical_free_cells pid=$!"
fi

for variant in nav_no_clearance_full vertical_2d_only vertical_3d_only; do
  wait_and_evaluate_pair "$variant"
done

wait_for_pid_file "$DENSE_OUT/train.pid" "training all_vertical_free_cells"
test -s "$DENSE_OUT/training_summary.json"
test -s "$DENSE_OUT/best.pt"
test -s "$DENSE_OUT/training_loss.png"

launch_replay interioragent all_vertical_free_cells "$DENSE_OUT/best.pt" 32
launch_replay grscene all_vertical_free_cells "$DENSE_OUT/best.pt" 32
wait_and_evaluate_pair all_vertical_free_cells

FINAL=$ROOT/final_report
rm -rf "$FINAL.tmp"
"$PY" -u "$CODE/scripts/aggregate_voxroom_ablations.py" \
  --evaluation-root "$EVAL" --training-root "$TRAIN" --out-dir "$FINAL.tmp" \
  >"$ROOT/aggregate.log" 2>&1
rm -rf "$FINAL"
mv "$FINAL.tmp" "$FINAL"
log_status "pipeline complete: $FINAL/report.md"
