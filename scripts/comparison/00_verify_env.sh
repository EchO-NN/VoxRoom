#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source scripts/comparison/env.sh

RUN_ROOT="${COMPARISON_ROOT:-result/comparison_random_frontier_v1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    *) echo "[comparison-verify-env] unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT/commands"
LOG="$RUN_ROOT/commands/00_verify_env.sh.log"
export RUN_ROOT
STRICT_VERIFY="${STRICT_VERIFY:-0}"
export STRICT_VERIFY

{
  echo "repo_root=$REPO_ROOT"
  "$VOXROOM_PYTHON" -V
  "$VOXROOM_PYTHON" - <<'PY'
import importlib, json, os, platform, shutil, subprocess, sys
from pathlib import Path

def git_head(path):
    if path is None or not Path(path).exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def check_import(name):
    importlib.import_module(name)
    return True

payload = {
    "schema_version": 2,
    "strict_verify": os.environ.get("STRICT_VERIFY") == "1",
    "voxroom_python": sys.version,
    "platform": platform.platform(),
    "repo_commit": git_head(Path.cwd()),
    "commands": {name: shutil.which(name) for name in ["git", "roscore", "roslaunch", "rosrun", "catkin_make", "catkin"]},
    "voxroom_imports": {},
    "ros_imports": {},
    "ros_python_check": {"ok": False, "stdout": "", "stderr": "", "returncode": None},
    "external_repos": {},
    "build_artifacts": {},
}
for name in ["numpy", "scipy", "cv2", "voxroom_online"]:
    try:
        payload["voxroom_imports"][name] = check_import(name)
    except Exception as exc:
        raise SystemExit(f"missing python import {name}: {exc}")
ros_setup = os.environ.get("ROS_BASELINE_SETUP", "")
ros_python = os.environ.get("ROS_BASELINE_PYTHON", "python3")
ros_shell = f"""
set -euo pipefail
test -f {str(Path(ros_setup)).__repr__()}
set +u
source {str(Path(ros_setup)).__repr__()}
set -u
for setup in {str(Path(os.environ.get('DUDE_WS','')) / 'devel' / 'setup.bash').__repr__()} {str(Path(os.environ.get('ROSE2_WS','')) / 'devel' / 'setup.bash').__repr__()} {str(Path(os.environ.get('IPA_WS','')) / 'devel' / 'setup.bash').__repr__()}; do
  if [ -f "$setup" ]; then
    set +u
    source "$setup"
    set -u
  fi
done
{ros_python} - <<'PYROS'
import rospy
import actionlib
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
try:
    import jsk_recognition_msgs.msg
except Exception as exc:
    raise SystemExit('missing jsk_recognition_msgs: %s' % exc)
try:
    from ipa_building_msgs.msg import MapSegmentationAction, MapSegmentationGoal
except Exception as exc:
    raise SystemExit('missing ipa_building_msgs: %s' % exc)
print('ros python ok')
PYROS
"""
proc = subprocess.run(["bash", "-lc", ros_shell], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
payload["ros_python_check"] = {
    "ok": proc.returncode == 0,
    "stdout": proc.stdout,
    "stderr": proc.stderr,
    "returncode": proc.returncode,
    "ros_setup": ros_setup,
    "ros_python": ros_python,
}
for env_name in [
    "ACTIVE_ROOM_SEG_ROOT",
    "INCREMENTAL_DUDE_ROS_ROOT",
    "ROSE2_ROOT",
    "IPA_COVERAGE_ROOT",
]:
    raw = os.environ.get(env_name)
    path = Path(raw) if raw else None
    git_path = None
    if path is not None and path.exists():
        git_path = path if (path / ".git").exists() else path / "src"
    payload["external_repos"][env_name] = {
        "path": None if path is None else str(path),
        "exists": False if path is None else path.exists(),
        "commit": git_head(git_path),
    }
ckpt = os.environ.get("TOPOLOGY_DOOR_DETR_CHECKPOINT")
payload["topology_checkpoint"] = {"path": ckpt, "exists": bool(ckpt and Path(ckpt).exists())}
payload["build_artifacts"]["dude_executable"] = {
    "paths": [
        str(Path(os.environ["DUDE_WS"]) / "devel" / "lib" / "inc_dude" / "inc_dude"),
        str(Path(os.environ["INCREMENTAL_DUDE_ROS_ROOT"]) / "devel" / "lib" / "inc_dude" / "inc_dude"),
    ],
}
payload["build_artifacts"]["dude_executable"]["exists"] = any(Path(p).exists() and Path(p).is_file() for p in payload["build_artifacts"]["dude_executable"]["paths"])
Path(os.environ["RUN_ROOT"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["RUN_ROOT"]) / "environment.lock.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
if os.environ.get("STRICT_VERIFY") == "1":
    missing = []
    for cmd in ["roscore", "roslaunch", "rosrun"]:
        if not payload["commands"].get(cmd):
            missing.append(f"missing command {cmd}")
    if not payload["ros_python_check"]["ok"]:
        missing.append("ROS baseline python check failed")
    if not payload["topology_checkpoint"]["exists"]:
        missing.append("missing TOPOLOGY_DOOR_DETR_CHECKPOINT")
    for name, row in payload["external_repos"].items():
        if not row["exists"] or not row["commit"]:
            missing.append(f"missing external repo commit {name}")
    if not payload["build_artifacts"]["dude_executable"]["exists"]:
        missing.append("missing DUDE inc_dude executable")
    if missing:
        raise SystemExit("strict environment verification failed: " + "; ".join(missing))
PY
} 2>&1 | tee "$LOG"

echo "[comparison-verify-env] wrote $RUN_ROOT/environment.lock.json"
