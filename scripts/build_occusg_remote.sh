#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
export PYTHONPATH="/work/pydeps:${PYTHONPATH:-}"
export MAKEFLAGS=-j2
export CMAKE_BUILD_PARALLEL_LEVEL=2
cd /work/upstream
colcon build --executor sequential --packages-select mapconversion_msgs mapconversion incremental_dude_msgs incremental_dude_ros2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
g++ -std=c++17 -O2 /work/adapter/occusg_snapshot_to_octomap.cpp -loctomap -loctomath -o /work/adapter/snapshot_to_octomap
g++ -std=c++17 -O2 -I/usr/include/eigen3 -Isrc/mapconversion/mapconversion/include /work/adapter/check_occusg_native_projection.cpp src/mapconversion/mapconversion/src/MapConverter.cpp -o /work/adapter/check_projection
/work/adapter/check_projection
./build/incremental_dude_ros2/inc_decomp_regression_test
./build/incremental_dude_ros2/region_tracker_test
source install/setup.bash
export PYTHONPATH="/work/pydeps:/work/code:/work/upstream/src/scene_graph_core:/work/upstream/src/scene_graph_ros:${PYTHONPATH:-}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest /work/adapter/tests/test_occusg_adapter.py -q
python3 -c 'from scene_graph_ros.scene_graph_region import SceneGraphOrchestrator; print("upstream graph import OK")'
