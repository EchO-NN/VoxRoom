# Reproduction Guide

## One-Scene Isaac Run

```bash
cd /home/echo/VoxRoom
source scripts/activate_voxroom_env.sh

RUN_DIR=result/kujiale_0003_voxroom \
ROOMSEG_SNAPSHOT_MAX_SAVES=100000 \
VOXROOM_NUMBA_THREADS=28 \
VOXROOM_VOXEL_CPU_NUMBA_THREADS=28 \
scripts/run_one_scene_random_frontier.sh \
  data/interioragent_episodes/radius005_all_scenes/kujiale_0003.jsonl
```

## Expected Artifacts

```text
result/kujiale_0003_voxroom/
  results.jsonl
  roomseg_snapshots/
    roomseg_step_*.npz
```

## Snapshot Sanity Checks

The current final method should satisfy:

- `voxel_step2_extension_candidate_map` has no nonzero cells.
- `voxel_step2_extension_separator_map` has no nonzero cells.
- DDA endpoint splat config is zero in x/y and z.
- Navigation endpoint splat config is zero in x/y and z.

Example check:

```bash
python - <<'PY'
import glob
import os
import numpy as np

run_dir = "result/kujiale_0003_voxroom"
files = sorted(glob.glob(os.path.join(run_dir, "roomseg_snapshots", "*.npz")))
print("snapshots", len(files))
if files:
    d = np.load(files[-1], allow_pickle=True)
    for key in [
        "voxel_step2_extension_candidate_map",
        "voxel_step2_extension_separator_map",
    ]:
        if key in d.files:
            print(key, int(np.count_nonzero(d[key])))
PY
```

## Rendering Final Door-Completion Views

```bash
python scripts/render_final_door_completion_snapshots.py \
  --input-root result/kujiale_0003_voxroom \
  --output-root result/kujiale_0003_door_completion_views \
  --stateful-by-scene
```

## Full-Corpus Run

```bash
RUN_ROOT=result/radius005_robot005_endpoint1x1_all_scenes \
PARALLEL_JOBS=4 \
RUN_NUMBA_NUM_THREADS=7 \
scripts/run_all_scenes_random_frontier.sh
```

## Evaluation

```bash
voxroom-roomseg-eval --help
```

For formal comparison scripts, start from:

```bash
scripts/comparison/00_verify_env.sh
```

