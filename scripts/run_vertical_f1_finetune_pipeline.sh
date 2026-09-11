#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${VOXROOM_ABLATION_ROOT:-/media/echo/data/voxroom_ablation_20260828}
CODE=$ROOT/code
TRAIN=$ROOT/training
EVAL=$ROOT/evaluation
PY=${VOXROOM_ABLATION_PYTHON:-/home/echo/miniforge3/envs/RoboTwin/bin/python}
IA=/media/echo/data/voxroom_roomseg_evaluation/interioragent_all_available_gt_20260828
GR=/media/echo/data/voxroom_roomseg_evaluation/grscene_all_available_gt_20260817
RUN=vertical_full_retrain_f1_finetune_epoch14_lr1e4
REPLAY_VARIANT=production_full_model_original
STATUS=$ROOT/vertical_f1_finetune_status.log

log_status() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$STATUS"
}

wait_for_training() {
  local directory=$TRAIN/$RUN
  local pid
  pid=$(<"$directory/train.pid")
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  for name in training_summary.json training_history.json training_loss.png best.pt baseline_epoch14.pt; do
    if [[ ! -s "$directory/$name" ]]; then
      log_status "ERROR incomplete training artifact: $name"
      return 1
    fi
  done
  log_status "training complete: $RUN"
}

select_checkpoint() {
  "$PY" - "$TRAIN/$RUN" <<'PY'
import json
import sys
from pathlib import Path

import torch

directory = Path(sys.argv[1])
summary = json.loads((directory / "training_summary.json").read_text(encoding="utf-8"))
load_kwargs = {"map_location": "cpu"}
try:
    baseline = torch.load(directory / "baseline_epoch14.pt", weights_only=True, **load_kwargs)
except TypeError:
    baseline = torch.load(directory / "baseline_epoch14.pt", **load_kwargs)
baseline_f1 = float(baseline["achieved_f1"])
finetuned_f1 = float(summary["selected_checkpoint_metrics"]["f1"])
winner = "best.pt" if finetuned_f1 > baseline_f1 else "baseline_epoch14.pt"
result = {
    "selection_metric": "validation_f1_at_fixed_threshold_0.5",
    "baseline_f1": baseline_f1,
    "finetuned_f1": finetuned_f1,
    "selected_checkpoint": str(directory / winner),
    "selected_source": "finetuned" if winner == "best.pt" else "baseline_epoch14",
}
(directory / "final_selection.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(result["selected_checkpoint"])
PY
}

launch_replay() {
  local dataset=$1 checkpoint=$2 base out
  if [[ "$dataset" == interioragent ]]; then base=$IA; else base=$GR; fi
  out=$EVAL/${dataset}_${RUN}
  mkdir -p "$out"
  nohup env PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u "$CODE/scripts/replay_voxroom_ablation.py" \
    --index "$base/index_voxroom.json" --annotation-dir "$base/annotations" \
    --out-dir "$out" --variant "$REPLAY_VARIANT" \
    --config "$CODE/configs/voxroom_online.yaml" \
    --checkpoint "$checkpoint" --device cuda:0 --inference-batch-size 32 \
    >"$out/replay.log" 2>&1 < /dev/null &
  echo $! >"$out/replay.pid"
  log_status "launched replay: $dataset pid=$! checkpoint=$checkpoint"
}

wait_replay() {
  local dataset=$1 out=$EVAL/${dataset}_${RUN} pid
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
  local dataset=$1 base out
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

log_status "Vertical-Free epoch-14 warm-start fine-tune pipeline started"
wait_for_training
CHECKPOINT=$(select_checkpoint)
log_status "selected checkpoint: $CHECKPOINT"
for dataset in interioragent grscene; do launch_replay "$dataset" "$CHECKPOINT"; done
for dataset in interioragent grscene; do wait_replay "$dataset"; done
for dataset in interioragent grscene; do evaluate_one "$dataset"; done
log_status "pipeline complete"
