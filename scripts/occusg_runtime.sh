#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /work/upstream/install/setup.bash
export PYTHONPATH="/work/pydeps:/work/code:/work/upstream/src/scene_graph_core:/work/upstream/src/scene_graph_ros:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_LOCALHOST_ONLY=1
exec python3 -u "$@"
