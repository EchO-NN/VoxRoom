# VoxRoom real-robot deployment — release placeholder

**Status: deployment code and demonstration video pending release.**

This directory reserves the entry point for the real-robot FAST-LIO2, occupancy-mapping, and TensorRT FP16 implementation. It intentionally contains no fake launch files or nonfunctional inference stubs. The existing `voxroom_online/real_runtime/` contains a separate ZED-oriented runtime; it must not be presented as the deployment measured below.

Maintainer: **EchO-NN**, <2579947814@qq.com>.

## Platform

- LiDAR: PACECAT (蓝海光电) **LDS_M200_E**.
- Monitoring RGB camera: **DJI Osmo Action 4**, for viewpoint recording/monitoring.
- Robot: circular, **two-wheel differential-drive** base.
- Onboard computation: **Intel Core Ultra 9 275HX** and **NVIDIA GeForce RTX 5070 Laptop GPU**, carried on the mobile base.

## Planned release contents

When the actual implementation is provided, add its installation instructions, pinned external dependencies, sensor calibration and frame conventions, FAST-LIO2 / voxel integration launch files, TensorRT engine export/build instructions, verified model metadata, live/replay entry points, and repeatable latency measurement commands here. Do not substitute the simulation nvblox adapter for the robot's measured mapping stack.

The proposed signal path is LiDAR observations and pose estimation → occupancy voxel map → VoxRoom structural extraction and hybrid entry seeds → TensorRT FP16 verification → room segmentation. The RGB monitoring stream is separate. Sensor timestamps, map resolution, gravity alignment, coordinate transforms, and free/occupied/unknown semantics must be documented when code is released.

The real-world video section is reserved in the top-level README and `docs/index.html`. Add the actual recording using the instructions in `docs/assets/README.md`.

## Latency measurement notes

The source is the maintainer-provided measurement screenshot, accompanying hardware description, and clarification that FP16 verification takes **355.06 ms**, supplied on 2026-09-12. No raw profiler trace was supplied with it, so these are **reported measurements**, not independently rerun benchmarks from this repository.

| Stage | Time (ms) |
| --- | ---: |
| FAST-LIO2 + point cloud to voxel map, including publishing/finalization | 54.52 |
| Structural extraction, including voxel evidence / SFM / rule-based seeds and constraints | 136.95 |
| 2D ray casting + historical candidate merging | 42.48 |
| TensorRT FP16 verification, including data preparation and filtering | 355.06 |
| Post-verification segmentation and room-property processing | 286.93 |
| **VoxRoom total (sum of the four stages, excluding SLAM / mapping)** | **821.42** |

The SLAM value is an average over multiple live frames. VoxRoom stage values are measurements using one archived map. Ray casting runs **once per second**, while **42.48 ms is the time for one invocation**, not its update interval. The inference duration covers the stage, not one candidate.

The VoxRoom total excludes mapping: **136.95 + 42.48 + 355.06 + 286.93 = 821.42 ms**, including one ray-casting invocation but not the separate **54.52 ms** SLAM/mapping average.

OctoMap-to-dense-array conversion and external Nav projection preparation were not individually timed. The sum above is **not a measured synchronized end-to-end latency or FPS claim**. Do not add the screenshot's internal substeps a second time. Detailed profiler data and model weights are not uploaded.
