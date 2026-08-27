#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK_ROOT="${TASK_ROOT:-/media/echo/data/voxroom_door_seed_union_voxel_milestones_20260811}"
SELECTION_ROOT="$REPO_ROOT/data/door_seed_learning/tvars_vertical_union_20260811"
EPISODE_ROOT="$TASK_ROOT/episodes"
COLLECTION_ROOT="$TASK_ROOT/collection"
LOG_ROOT="$TASK_ROOT/logs"
mkdir -p "$COLLECTION_ROOT" "$LOG_ROOT"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}"

DATASET=interioragent \
TASK_ROOT="$TASK_ROOT" \
SCENE_LIST="$SELECTION_ROOT/interioragent_scenes.txt" \
EPISODE_DIR="$EPISODE_ROOT/interioragent" \
COLLECTION_ROOT="$COLLECTION_ROOT" \
RUN_ROOT="$TASK_ROOT/runs/interioragent" \
PARALLEL_JOBS=1 \
  scripts/collect_voxroom_tvars_vertical_union_scene_list.sh \
  > "$LOG_ROOT/interioragent_batch.log" 2>&1 &
interioragent_pid=$!

DATASET=grscene \
TASK_ROOT="$TASK_ROOT" \
SCENE_LIST="$SELECTION_ROOT/grscene_scenes.txt" \
EPISODE_DIR="$EPISODE_ROOT/grscene" \
COLLECTION_ROOT="$COLLECTION_ROOT" \
RUN_ROOT="$TASK_ROOT/runs/grscene" \
PARALLEL_JOBS=1 \
  scripts/collect_voxroom_tvars_vertical_union_scene_list.sh \
  > "$LOG_ROOT/grscene_batch.log" 2>&1 &
grscene_pid=$!

interioragent_status=0
grscene_status=0
wait "$interioragent_pid" || interioragent_status=$?
wait "$grscene_pid" || grscene_status=$?
printf 'interioragent_status=%d\ngrscene_status=%d\n' \
  "$interioragent_status" "$grscene_status" > "$TASK_ROOT/collection_status.txt"
if [[ "$interioragent_status" -ne 0 || "$grscene_status" -ne 0 ]]; then
  echo "[raw-seed-union-controller] collection failed; annotation not started" >&2
  exit 1
fi

python3 - "$COLLECTION_ROOT" "$SELECTION_ROOT/interioragent_scenes.txt" "$SELECTION_ROOT/grscene_scenes.txt" <<'PY'
import json
import sys
from pathlib import Path

collection_root = Path(sys.argv[1])
expected = set()
for path in map(Path, sys.argv[2:]):
    expected.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
complete = set()
for path in collection_root.glob("*/manifest.json"):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    finals = [item for item in manifest.get("snapshots", []) if item.get("is_final")]
    if len(finals) == 1 and (path.parent / str(finals[0]["path"])).is_file():
        complete.add(str(manifest["scene_id"]))
missing = sorted(expected - complete)
if missing:
    raise SystemExit("missing complete collections: " + ", ".join(missing))
print("all %d selected scenes are complete" % len(expected))
PY

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec /home/echo/miniforge3/envs/sgnav-isaac/bin/python \
  voxroom_online/isaac_runtime/scripts/annotate_door_seed_labels.py \
  --collection-root "$COLLECTION_ROOT" \
  --annotator joey \
  > "$LOG_ROOT/annotation.log" 2>&1
