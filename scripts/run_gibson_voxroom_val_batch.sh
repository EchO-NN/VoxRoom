#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PREFIX="${TARGET_PREFIX:-$HOME/.conda/envs/active-room-seg}"
PYTHON="${PYTHON:-$TARGET_PREFIX/bin/python}"
VAL_DATASET="$ROOT_DIR/data/datasets/pointnav/gibson/v1/val/val.json.gz"
BATCH_TAG="${BATCH_TAG:-gibson_voxroom_val_$(date +%Y%m%d_%H%M%S)}"
BATCH_DIR="${BATCH_DIR:-$ROOT_DIR/outputs/$BATCH_TAG}"
STATUS_PATH="$BATCH_DIR/batch_status.jsonl"
SUMMARY_PATH="$BATCH_DIR/batch_summary.json"
RUNNER="$ROOT_DIR/scripts/run_gibson_voxroom_visual.sh"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment is missing: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$VAL_DATASET" ]]; then
    echo "Official Gibson val dataset is missing: $VAL_DATASET" >&2
    exit 1
fi
if [[ ! -x "$RUNNER" ]]; then
    echo "Gibson VoxRoom runner is missing: $RUNNER" >&2
    exit 1
fi
if [[ -e "$BATCH_DIR" ]]; then
    echo "Batch output already exists: $BATCH_DIR" >&2
    exit 1
fi

mapfile -t scenes < <(
    "$PYTHON" - "$VAL_DATASET" <<'PY'
import gzip
import json
import pathlib
import sys

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    payload = json.load(stream)
scenes = sorted(
    {
        pathlib.Path(episode["scene_id"]).stem
        for episode in payload.get("episodes", [])
    }
)
for scene in scenes:
    print(scene)
PY
)
if [[ "${#scenes[@]}" -ne 14 ]]; then
    echo "Expected 14 scenes in the original Gibson val split, found ${#scenes[@]}" >&2
    exit 1
fi

mkdir -p "$BATCH_DIR/logs"
printf '%s\n' "${scenes[@]}" >"$BATCH_DIR/scenes.txt"
printf 'batch_tag=%s\nbatch_dir=%s\nval_dataset=%s\nscene_count=%s\n' \
    "$BATCH_TAG" "$BATCH_DIR" "$VAL_DATASET" "${#scenes[@]}" \
    >"$BATCH_DIR/batch_contract.txt"

completed=0
failed=0
for scene in "${scenes[@]}"; do
    run_id="$(< /proc/sys/kernel/random/uuid)"
    run_dir="$BATCH_DIR/$scene"
    log_path="$BATCH_DIR/logs/$scene.log"
    started_at="$(date +%s)"
    echo "[gibson-val-batch] starting scene=$scene run_id=$run_id"

    set +e
    SCENE_ID="$scene" \
    EPISODE_INDEX=0 \
    RUN_ID="$run_id" \
    RUN_TAG="${BATCH_TAG}_${scene}" \
    RUN_DIR="$run_dir" \
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-2500}" \
        "$RUNNER" 2>&1 | tee "$log_path"
    run_status="${PIPESTATUS[0]}"
    set -e

    finished_at="$(date +%s)"
    if [[ "$run_status" -eq 0 \
        && -f "$run_dir/validation.json" \
        && -f "$run_dir/room_mask_final.png" \
        && -f "$run_dir/voxroom/result.json" \
        && -f "$run_dir/voxroom/visualization_final.png" ]]; then
        scene_status="completed"
        completed=$((completed + 1))
    else
        scene_status="failed"
        failed=$((failed + 1))
    fi
    "$PYTHON" - "$STATUS_PATH" "$scene" "$scene_status" "$run_status" \
        "$run_id" "$run_dir" "$log_path" "$started_at" "$finished_at" <<'PY'
import json
import pathlib
import sys

(
    status_path,
    scene,
    status,
    exit_code,
    run_id,
    run_dir,
    log_path,
    started_at,
    finished_at,
) = sys.argv[1:]
row = {
    "scene": scene,
    "status": status,
    "exit_code": int(exit_code),
    "run_id": run_id,
    "run_dir": run_dir,
    "log_path": log_path,
    "started_at_unix": int(started_at),
    "finished_at_unix": int(finished_at),
    "elapsed_seconds": int(finished_at) - int(started_at),
}
with pathlib.Path(status_path).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
    echo "[gibson-val-batch] scene=$scene status=$scene_status exit=$run_status"
done

"$PYTHON" - "$STATUS_PATH" "$SUMMARY_PATH" "$BATCH_TAG" <<'PY'
import json
import pathlib
import sys

status_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
rows = [
    json.loads(line)
    for line in status_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
summary = {
    "batch_tag": sys.argv[3],
    "scene_count": len(rows),
    "completed": sum(row["status"] == "completed" for row in rows),
    "failed": sum(row["status"] == "failed" for row in rows),
    "scenes": rows,
}
temporary = summary_path.with_name("." + summary_path.name + ".tmp")
temporary.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(summary_path)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [[ "$failed" -ne 0 || "$completed" -ne "${#scenes[@]}" ]]; then
    exit 1
fi
