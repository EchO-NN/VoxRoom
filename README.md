# VoxRoom

VoxRoom is an online room-mask readout pipeline for embodied RGB-D exploration in
Isaac Sim InteriorAgent scenes. It builds a sensor-aware voxel memory with DDA
ray integration, projects columnar wall/free/entry-seed evidence, completes wall
anchors and entry separators, then extracts room masks directly from the online
map.

## Pipeline

<p align="center">
  <img src="assets/pipeline.png" alt="VoxRoom pipeline" width="100%">
</p>

## Online Example

The example below is an online Isaac Sim run on `kujiale_0003` with RGB, depth,
navigation room masks, vertical-free geometry, frontier targets, and A* path
overlays.

<p align="center">
  <img src="assets/kujiale_0003_demo.gif" alt="VoxRoom online demo on kujiale_0003" width="100%">
</p>

Full-resolution video: [assets/kujiale_0003_demo.mp4](assets/kujiale_0003_demo.mp4)

## Quick Start

```bash
cd /home/echo/VoxRoom
scripts/setup_voxroom_env.sh
source scripts/activate_voxroom_env.sh
python -m pip install -e .
```

Set your Isaac Sim and InteriorAgent paths in `configs/voxroom_online.yaml`, then
prepare episodes:

```bash
GENERATE_ONLY=1 scripts/run_all_scenes_random_frontier.sh
```

Run one online scene:

```bash
RUN_DIR=result/kujiale_0003_voxroom \
ROOMSEG_SNAPSHOT_MAX_SAVES=100000 \
VOXROOM_NUMBA_THREADS=28 \
VOXROOM_VOXEL_CPU_NUMBA_THREADS=28 \
scripts/run_one_scene_random_frontier.sh \
  data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl
```

The final geometry settings use 1x1 endpoint marking, no wall-line extension,
stateful door memory, no-clearance navigation-map entry seeds, and
`0.95 * ceiling_height` for entry-seed height selection.

## Evaluation

```bash
voxroom-roomseg-eval --help
```

Evaluation outputs include `summary_metrics.json`, `metrics_report.md`, and
room-mask galleries.
