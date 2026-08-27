#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK_ROOT="${TASK_ROOT:-/media/echo/data/voxroom_door_seed_union_voxel_milestones_20260811}"
SELECTION_ROOT="${SELECTION_ROOT:-$REPO_ROOT/data/door_seed_learning/tvars_vertical_union_20260811}"
COLLECTION_ROOT="${COLLECTION_ROOT:-$TASK_ROOT/collection}"
POLL_SECONDS="${POLL_SECONDS:-30}"
ANNOTATOR="${ANNOTATOR:-joey}"
ANNOTATION_PYTHON="${ANNOTATION_PYTHON:-/home/echo/miniforge3/envs/sgnav-isaac/bin/python}"

mkdir -p "$TASK_ROOT/logs"
exec 9>"$TASK_ROOT/.annotation_finalizer.lock"
if ! flock -n 9; then
  echo "[raw-seed-union-finalizer] another finalizer owns $TASK_ROOT" >&2
  exit 2
fi

collection_complete() {
  python3 - "$COLLECTION_ROOT" \
    "$SELECTION_ROOT/interioragent_scenes.txt" \
    "$SELECTION_ROOT/grscene_scenes.txt" <<'PY'
import json
import sys
from pathlib import Path

collection_root = Path(sys.argv[1])
expected = set()
for path in map(Path, sys.argv[2:]):
    expected.update(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
complete = set()
for path in collection_root.glob("*/manifest.json"):
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    collection = dict(manifest.get("collection", {}) or {})
    finals = [item for item in manifest.get("snapshots", []) if item.get("is_final")]
    if (
        len(finals) == 1
        and (path.parent / str(finals[0].get("path", ""))).is_file()
        and collection.get("raw_seed_source") == "voxroom_tvars_vertical_union"
        and int(collection.get("collection_every_steps", 0)) == 5
        and bool(collection.get("save_full_voxel_milestones", False))
        and bool(finals[0].get("full_voxel_snapshot", False))
    ):
        complete.add(str(manifest.get("scene_id", "")))
missing = sorted(expected - complete)
if missing:
    print("[raw-seed-union-finalizer] waiting; missing=%d first=%s" % (
        len(missing), ",".join(missing[:4])
    ))
    raise SystemExit(1)
print("[raw-seed-union-finalizer] all %d scenes complete" % len(expected))
PY
}

while ! collection_complete; do
  sleep "$POLL_SECONDS"
done

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ANNOTATION_PYTHON" \
  voxroom_online/isaac_runtime/scripts/annotate_door_seed_labels.py \
  --collection-root "$COLLECTION_ROOT" \
  --annotator "$ANNOTATOR"
