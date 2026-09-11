# Running and reproducing VoxRoom

## Release boundaries

This repository contains source and documentation, not a turnkey simulator image. Detailed experiment data, scene assets, room/seed annotations, and model weights are **not uploaded**. Obtain those inputs separately with the appropriate permissions. The README's tables are transcribed from the manuscript, not evidence that running a demo once reproduces the full experimental protocol.

The simulation main method uses a 19×19 voxel branch and 41×41 **Vertical / Structural Free** context, fixed NN threshold 0.5, line correlation 0.95, and orthogonal variance 0.65 cells². Experimental L-shaped dual-arm handling is disabled. Column-encoding reuse is enabled.

## Environment

Use Linux, Python 3.11, Isaac Sim standalone 5.1.0, and an NVIDIA GPU. Create the environment with `scripts/setup_voxroom_env.sh`, activate it with `scripts/activate_voxroom_env.sh`, then install the package with `python -m pip install -e '.[door-seed-learning]'`.

The default `nvblox_fast_dda` mapper requires external `nvblox_torch` bindings exposing `Mapper`, `MapperParams`, `ProjectiveIntegratorParams`, `ProjectiveIntegratorType`, and `Sensor`. See [the adapter](../voxroom_online/isaac_runtime/mapping/voxel_nvblox_fast_dda_backend.py) for the exact interface. Isaac Sim and these bindings are not installed by `pip install -e .`. Match the PyTorch/CUDA build to the GPU and simulator environment; do not silently switch mapping backends to make a benchmark run.

```bash
export ISAAC_ROOT=/path/to/isaac-sim-standalone-5.1.0-linux-x86_64
source scripts/activate_voxroom_env.sh
bash scripts/run_voxroom_isaac_env.sh -c 'import torch, nvblox_torch; print(torch.cuda.is_available())'
```

`scripts/run_voxroom_isaac_env.sh` sources Isaac's `setup_conda_env.sh` and runs Python in the configured environment. The real-robot FAST-LIO2 / TensorRT stack is separate and pending release; see [real_robot](../real_robot/README.md).

## Local paths and checkpoint compatibility

```bash
cp configs/voxroom_online.yaml configs/voxroom.local.yaml
```

Edit at least `paths.interioragent_root`, `dataset.root`, `paths.isaac_sim_root`, and `mapping.room_segmentation.door_seed_learning.checkpoint_path`. Put generated data and run outputs outside the source tree when storage is limited. The local configuration pattern is ignored by Git.

Trained weights are **not included**. Provide your own compatible checkpoint through the local configuration. The inference engine checks architecture, preprocessing, branch context, patch sizes, height scale/layers, and metadata. Recent source/config changes can make older source-hash checks fail. Do not disable all checks blindly: use a matching experiment checkout, a newly trained compatible model, or explicitly audit and document a permitted metadata exception. Do not silently fall back to rule-only seeds and label that run as the neural main method.

## Scene preparation and a single-scene demo

Inspect the available preprocess/episode arguments before creating your own data:

```bash
bash scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/preprocess_interioragent.py --help
bash scripts/run_voxroom_isaac_env.sh voxroom_online/isaac_runtime/scripts/generate_episodes.py --help
```

Use the generated scene maps and episode JSONL files with consistent resolution, coordinate frames, and asset paths. A single-scene random-frontier run is:

```bash
CONFIG=configs/voxroom.local.yaml \
RUN_DIR=result/my_scene \
VOXROOM_NUMBA_THREADS=8 \
bash scripts/run_one_scene_random_frontier.sh /path/to/scene_episode.jsonl
```

The launcher enables the diagnostic popup and saves room-segmentation snapshots. A working display is required for an interactive popup. Its generated frame directory and simulator headless setting are separate: a headless simulator does not imply that GUI libraries can open a window without a display.

Random-frontier exploration is a collection/demo mode. The paper's controlled comparison instead requires matched trajectories, observations, coverage checkpoints, and evaluation domains across all methods. Changing trajectory or start pose produces a new experiment.

## Collection, annotation, and training

The collection configuration selects the TVARS-style/voxel raw-seed union and can save full voxel milestones. Configure the collection root and explicit train/validation/test scene split. Training labels are annotated on final accumulated candidate maps and back-projected into historical candidate positions; those labels must not enter inference.

```bash
bash scripts/annotate_door_seed_labels.sh --help
bash scripts/build_door_seed_dataset.sh --help
bash scripts/train_door_seed_classifier.sh --help
```

The README gives an example using fixed-threshold validation F1 selection. Synchronized 0°, 90°, 180°, and 270° rotations preserve correspondence between both branches. `--train-mirror-lr-once` adds one extra left-right mirror, not another four rotated mirrors. Freeze the scene split before augmentation. Training convergence, early stopping, and repeated-run statistics require actual training logs; they are not guaranteed by the example command.

## Snapshot inspection versus experimental replay

The snapshot utility provides visualization, stateless processing, and stateful replay:

```bash
python -m voxroom_online.isaac_runtime.scripts.replay_voxel_roomseg_snapshots --help
```

Its memory-source and raw-candidate policies matter. A stateless replay, a replay importing saved memory, and a fresh stateful history are different experimental conditions. Do not present one as another. Preserve saved source checksums, model metadata, method configuration, and per-scene checkpoint order. Avoid using final predicted room labels or GT as earlier inference input.

## Paired evaluation

For the established VoxRoom / TVARS workflow, inspect `scripts/run_roomseg_coverage_eval_isaac.sh` and [coverage evaluation](roomseg_coverage_evaluation.md). The strict TVARS adapter expects its original source checkout and DETR weights, separately from the VoxRoom verifier:

```bash
git worktree add external_baselines/Active_room_segmentation c6dbe92c55ea34f9710ddcc5b10d59144662fe68
```

Prepare external runtime paths and weights as required by the launcher. Native DUDE/IPA/ROSE² and other adapters similarly need their own dependencies; repository fallback implementations are not automatically equivalent to those native baselines.

Core evaluation entry points:

```bash
voxroom-roomseg-eval --help
python scripts/evaluate_approved_roomseg_paper_pr.py --help
```

Use the shared observed reference domain and approved final-room GT projected into each checkpoint. Exclude training/validation scenes. Set the room-area filter to 0.5 m² and the cell resolution to 0.05 m for this protocol. Never include unobserved future geometry in a method's inference just because the evaluation system has a full reference map.

P and R are macro room-overlap metrics. Compute F1 for each checkpoint before averaging. Room-mIoU uses one-to-one Hungarian assignment and is normalized by the number of GT rooms. Temporal SD/CV/Worst are computed within each scene first, then averaged; unavailable progress points are reported rather than replaced with zeros.

## Tests

After installing development/test dependencies:

```bash
python -m pytest -q tests/test_door_seed_column_reuse.py tests/test_voxroom_ablation_support.py
```

Native baseline, GUI, simulator, and real-sensor workflows require their corresponding external dependencies. Documentation checks and CPU model tests do not validate a full sensor deployment or re-run the paper benchmark.

## Historical material

August GRScene releases and source manifests in `repro/` describe earlier experiments. They do not override the current model, manuscript tables, or deployment status. The project README and [release notes](release_notes.md) distinguish these sources. No detailed data or trained weights are added by the paper-oriented publication update.
