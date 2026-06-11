#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source scripts/comparison/env.sh

mkdir -p external_baselines "$DUDE_WS/src" "$ROSE2_WS/src" "$IPA_WS/src" external_baselines/patches

clone_if_missing() {
  local url="$1"
  local dest="$2"
  local branch="${3:-}"
  if [[ -d "$dest/.git" ]]; then
    echo "[fetch-baselines] reusing $dest"
    return 0
  fi
  if [[ -n "$branch" ]]; then
    git clone -b "$branch" "$url" "$dest"
  else
    git clone "$url" "$dest"
  fi
}

clone_if_missing https://github.com/FreeformRobotics/Active_room_segmentation "$ACTIVE_ROOM_SEG_ROOT"
clone_if_missing https://github.com/lfermin77/Incremental_DuDe_ROS "$INCREMENTAL_DUDE_ROS_ROOT"
clone_if_missing https://github.com/aislabunimi/ROSE2 "$ROSE2_ROOT"
clone_if_missing https://github.com/ipa320/ipa_coverage_planning "$IPA_COVERAGE_ROOT" noetic_dev

"$VOXROOM_PYTHON" - <<'PY'
import json, os, subprocess
from pathlib import Path

def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

rows = {}
for name in ("ACTIVE_ROOM_SEG_ROOT", "INCREMENTAL_DUDE_ROS_ROOT", "ROSE2_ROOT", "IPA_COVERAGE_ROOT"):
    path = Path(os.environ[name])
    rows[name] = {
        "path": str(path),
        "exists": path.exists(),
        "commit": git_head(path),
    }
payload = {"schema_version": 2, "repos": rows}
for out in [Path("external_baselines/repos.lock.json"), Path(os.environ.get("RUN_ROOT", "result/comparison_random_frontier_t0")) / "repos.lock.json"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

if [[ ! -f "$ROS_BASELINE_SETUP" ]]; then
  echo "[fetch-baselines] missing ROS setup: $ROS_BASELINE_SETUP" >&2
  echo "[fetch-baselines] clone completed where possible, but catkin build cannot run without ROS." >&2
  exit 1
fi

COMMON_CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
if [[ -n "${ROS_BASELINE_CC:-}" && -n "${ROS_BASELINE_CXX:-}" ]]; then
  COMMON_CMAKE_ARGS="$COMMON_CMAKE_ARGS -DCMAKE_C_COMPILER=$ROS_BASELINE_CC -DCMAKE_CXX_COMPILER=$ROS_BASELINE_CXX"
fi

bash -lc "set -euo pipefail; export CC='${ROS_BASELINE_CC:-}' CXX='${ROS_BASELINE_CXX:-}'; set +u; source '$ROS_BASELINE_SETUP'; set -u; cd '$DUDE_WS'; catkin_make --cmake-args $COMMON_CMAKE_ARGS -DCGAL_DIR=/usr/lib/x86_64-linux-gnu/cmake/CGAL"
bash -lc "set -euo pipefail; export CC='${ROS_BASELINE_CC:-}' CXX='${ROS_BASELINE_CXX:-}'; set +u; source '$ROS_BASELINE_SETUP'; set -u; cd '$ROSE2_WS'; rosdep install --from-paths src --ignore-src -r -y || true; catkin_make --cmake-args $COMMON_CMAKE_ARGS"
bash -lc "set -euo pipefail; export CC='${ROS_BASELINE_CC:-}' CXX='${ROS_BASELINE_CXX:-}'; set +u; source '$ROS_BASELINE_SETUP'; set -u; cd '$IPA_WS'; rosdep install --from-paths src/ipa_coverage_planning/ipa_building_msgs src/ipa_coverage_planning/ipa_room_segmentation --ignore-src -r -y || true; catkin_make --cmake-args $COMMON_CMAKE_ARGS -DCATKIN_WHITELIST_PACKAGES='ipa_building_msgs;ipa_room_segmentation'"

STRICT_VERIFY=1 bash scripts/comparison/00_verify_env.sh --run-root "${RUN_ROOT:-result/comparison_random_frontier_t0}"
