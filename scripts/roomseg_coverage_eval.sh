#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

ACTION="${1:-}"
RESULT_ROOT="${2:-}"
EVAL_DIR="${3:-}"
if [[ ! "$ACTION" =~ ^(index|annotate|build-gt|compute)$ ]]; then
  echo "usage: $0 {index|annotate|build-gt|compute} RESULT_ROOT EVAL_DIR" >&2
  exit 1
fi
if [[ -z "$RESULT_ROOT" || ! -d "$RESULT_ROOT" || -z "$EVAL_DIR" ]]; then
  echo "usage: $0 {index|annotate|build-gt|compute} RESULT_ROOT EVAL_DIR" >&2
  exit 1
fi

VOXROOM_INDEX="$EVAL_DIR/index_voxroom.json"
TVARS_INDEX="$EVAL_DIR/index_tvars_original.json"
ANNOTATION_DIR="$EVAL_DIR/annotations"
GT_DIR="$EVAL_DIR/final_gt"
STEP_GT_DIR="$EVAL_DIR/step_gt"
MIN_ROOM_AREA_M2="${MIN_ROOM_AREA_M2:-0.0}"
CELL_SIZE_M="${CELL_SIZE_M:-0.05}"
mkdir -p "$EVAL_DIR"

"$PYTHON" -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli index-coverage \
  --result-root "$RESULT_ROOT" \
  --method voxroom \
  --snapshot-policy all \
  --out "$VOXROOM_INDEX"
"$PYTHON" -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli index-coverage \
  --result-root "$RESULT_ROOT" \
  --method tvars_original \
  --snapshot-policy all \
  --out "$TVARS_INDEX"

"$PYTHON" - "$VOXROOM_INDEX" "$TVARS_INDEX" <<'PY'
import json
import sys
from pathlib import Path

def contract(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(episode["episode_uid"]): [
            (str(item["coverage_event_id"]), Path(item["snapshot_path"]).stem)
            for item in episode["snapshots"]
        ]
        for episode in payload["episodes"]
    }

voxroom = contract(sys.argv[1])
tvars = contract(sys.argv[2])
if voxroom != tvars:
    raise SystemExit("VoxRoom and TVARS coverage indexes are not event-aligned")
print("paired coverage contract: {} episode(s)".format(len(voxroom)))
PY

case "$ACTION" in
  index)
    ;;
  annotate)
    exec "$PYTHON" -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli annotate-last \
      --index "$VOXROOM_INDEX" \
      --annotation-dir "$ANNOTATION_DIR" \
      --gt-dir "$GT_DIR" \
      --min-room-area-m2 "$MIN_ROOM_AREA_M2" \
      --cell-size-m "$CELL_SIZE_M" \
      --source-view segmentation
    ;;
  build-gt)
    "$PYTHON" -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli build-gt \
      --index "$VOXROOM_INDEX" \
      --annotation-dir "$ANNOTATION_DIR" \
      --gt-dir "$GT_DIR"
    "$PYTHON" -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli backproject \
      --index "$VOXROOM_INDEX" \
      --gt-dir "$GT_DIR" \
      --step-gt-dir "$STEP_GT_DIR" \
      --eval-snapshot-policy all
    ;;
  compute)
    for method in voxroom tvars_original; do
      index_var="$VOXROOM_INDEX"
      if [[ "$method" == "tvars_original" ]]; then
        index_var="$TVARS_INDEX"
      fi
      metrics_dir="$EVAL_DIR/metrics_$method"
      "$PYTHON" -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli compute \
        --index "$index_var" \
        --step-gt-dir "$STEP_GT_DIR" \
        --out-dir "$metrics_dir" \
        --strict-paper \
        --precision-recall-only \
        --min-room-area-m2 "$MIN_ROOM_AREA_M2" \
        --cell-size-m "$CELL_SIZE_M"
      "$PYTHON" -m voxroom_online.isaac_runtime.evaluation.online_roomseg.cli report \
        --metrics-dir "$metrics_dir" \
        --out "$metrics_dir/REPORT.md"
    done
    "$PYTHON" - "$EVAL_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = {
    method: json.loads(
        (root / ("metrics_" + method) / "summary_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    for method in ("voxroom", "tvars_original")
}
curves = {
    method: {row["event_id"]: row for row in payload["coverage_curve"]}
    for method, payload in summaries.items()
}
if set(curves["voxroom"]) != set(curves["tvars_original"]):
    raise SystemExit("coverage metric curves are not event-aligned")
event_order = [row["event_id"] for row in summaries["voxroom"]["coverage_curve"]]
rows = []
for event_id in event_order:
    voxroom = curves["voxroom"][event_id]
    tvars = curves["tvars_original"][event_id]
    if voxroom["snapshot_count"] != tvars["snapshot_count"]:
        raise SystemExit("coverage metric curves have different sample counts")
    rows.append(
        {
            "event_id": event_id,
            "threshold": voxroom["threshold"],
            "mean_measured_coverage": voxroom["mean_measured_coverage"],
            "snapshot_count": voxroom["snapshot_count"],
            "voxroom_precision": voxroom["Precision"],
            "voxroom_recall": voxroom["Recall"],
            "tvars_precision": tvars["Precision"],
            "tvars_recall": tvars["Recall"],
            "precision_delta_voxroom_minus_tvars": (
                voxroom["Precision"] - tvars["Precision"]
            ),
            "recall_delta_voxroom_minus_tvars": (
                voxroom["Recall"] - tvars["Recall"]
            ),
        }
    )
(root / "coverage_comparison.json").write_text(
    json.dumps({"rows": rows}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
with (root / "coverage_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# Coverage Precision / Recall Comparison",
    "",
    "| Event | Coverage | VoxRoom P | VoxRoom R | TVARS P | TVARS R | Delta P | Delta R |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {event_id} | {mean_measured_coverage:.4f} | {voxroom_precision:.4f} | "
        "{voxroom_recall:.4f} | {tvars_precision:.4f} | {tvars_recall:.4f} | "
        "{precision_delta_voxroom_minus_tvars:.4f} | "
        "{recall_delta_voxroom_minus_tvars:.4f} |".format(**row)
    )
(root / "COVERAGE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote paired coverage comparison -> {}".format(root / "COVERAGE_COMPARISON.md"))
PY
    ;;
esac
