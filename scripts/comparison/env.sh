#!/usr/bin/env bash
# Shared environment contract for the VoxRoom T0 comparison flow.
#
# This file intentionally does not guess or activate ROS workspaces beyond
# defining their expected locations. Strict verification happens in
# 00_verify_env.sh before any Isaac collection or native replay work starts.

VOXROOM_REPO_ROOT="${VOXROOM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export VOXROOM_REPO_ROOT

DEFAULT_VOXROOM_PYTHON="$(command -v python)"
export VOXROOM_PYTHON="${VOXROOM_PYTHON:-$DEFAULT_VOXROOM_PYTHON}"
DEFAULT_ROS_BASELINE_SETUP="$VOXROOM_REPO_ROOT/external_baselines/ros_noetic_env/setup.bash"
DEFAULT_ROS_BASELINE_PYTHON="$VOXROOM_REPO_ROOT/external_baselines/ros_noetic_env/bin/python"
if [[ -z "${ROS_BASELINE_SETUP:-}" && -f "$DEFAULT_ROS_BASELINE_SETUP" ]]; then
  export ROS_BASELINE_SETUP="$DEFAULT_ROS_BASELINE_SETUP"
else
  export ROS_BASELINE_SETUP="${ROS_BASELINE_SETUP:-/opt/ros/noetic/setup.bash}"
fi
if [[ -z "${ROS_BASELINE_PYTHON:-}" && -x "$DEFAULT_ROS_BASELINE_PYTHON" ]]; then
  export ROS_BASELINE_PYTHON="$DEFAULT_ROS_BASELINE_PYTHON"
else
  export ROS_BASELINE_PYTHON="${ROS_BASELINE_PYTHON:-python3}"
fi
if [[ -f "$ROS_BASELINE_SETUP" ]]; then
  ROS_BASELINE_PREFIX="$(cd "$(dirname "$ROS_BASELINE_SETUP")" && pwd)"
  export ROS_BASELINE_PREFIX
  if [[ -d "$ROS_BASELINE_PREFIX/bin" ]]; then
    export PATH="$ROS_BASELINE_PREFIX/bin:$PATH"
  fi
fi
if [[ -n "${ROS_BASELINE_PREFIX:-}" ]]; then
  DEFAULT_ROS_BASELINE_CC="$ROS_BASELINE_PREFIX/bin/x86_64-conda-linux-gnu-cc"
  DEFAULT_ROS_BASELINE_CXX="$ROS_BASELINE_PREFIX/bin/x86_64-conda-linux-gnu-c++"
  if [[ -z "${ROS_BASELINE_CC:-}" && -x "$DEFAULT_ROS_BASELINE_CC" ]]; then
    export ROS_BASELINE_CC="$DEFAULT_ROS_BASELINE_CC"
  else
    export ROS_BASELINE_CC="${ROS_BASELINE_CC:-}"
  fi
  if [[ -z "${ROS_BASELINE_CXX:-}" && -x "$DEFAULT_ROS_BASELINE_CXX" ]]; then
    export ROS_BASELINE_CXX="$DEFAULT_ROS_BASELINE_CXX"
  else
    export ROS_BASELINE_CXX="${ROS_BASELINE_CXX:-}"
  fi
fi

export DUDE_WS="${DUDE_WS:-$VOXROOM_REPO_ROOT/external_baselines/dude_ws}"
export ROSE2_WS="${ROSE2_WS:-$VOXROOM_REPO_ROOT/external_baselines/rose2_ws}"
export IPA_WS="${IPA_WS:-$VOXROOM_REPO_ROOT/external_baselines/ipa_ws}"

export ACTIVE_ROOM_SEG_ROOT="${ACTIVE_ROOM_SEG_ROOT:-$VOXROOM_REPO_ROOT/external_baselines/Active_room_segmentation}"
export INCREMENTAL_DUDE_ROS_ROOT="${INCREMENTAL_DUDE_ROS_ROOT:-$DUDE_WS/src/Incremental_DuDe_ROS}"
export ROSE2_ROOT="${ROSE2_ROOT:-$ROSE2_WS/src/ROSE2}"
export IPA_COVERAGE_ROOT="${IPA_COVERAGE_ROOT:-$IPA_WS/src/ipa_coverage_planning}"
DEFAULT_TOPOLOGY_DOOR_DETR_CHECKPOINT="$ACTIVE_ROOM_SEG_ROOT/detr_door_detection/train_params/detr_resnet50_4/final_doors_dataset/model.pth"
if [[ -z "${TOPOLOGY_DOOR_DETR_CHECKPOINT:-}" && -f "$DEFAULT_TOPOLOGY_DOOR_DETR_CHECKPOINT" ]]; then
  export TOPOLOGY_DOOR_DETR_CHECKPOINT="$DEFAULT_TOPOLOGY_DOOR_DETR_CHECKPOINT"
else
  export TOPOLOGY_DOOR_DETR_CHECKPOINT="${TOPOLOGY_DOOR_DETR_CHECKPOINT:-}"
fi
export DETR_TORCH_HUB_REPO="${DETR_TORCH_HUB_REPO:-$VOXROOM_REPO_ROOT/external_baselines/facebookresearch_detr}"
DEFAULT_TOPOLOGY_DOOR_DETR_PYTHON="$HOME/miniforge3/envs/openpi-dev/bin/python"
if [[ -z "${TOPOLOGY_DOOR_DETR_PYTHON:-}" && -x "$DEFAULT_TOPOLOGY_DOOR_DETR_PYTHON" ]]; then
  export TOPOLOGY_DOOR_DETR_PYTHON="$DEFAULT_TOPOLOGY_DOOR_DETR_PYTHON"
else
  export TOPOLOGY_DOOR_DETR_PYTHON="${TOPOLOGY_DOOR_DETR_PYTHON:-python}"
fi
