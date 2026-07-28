# Modern GPU reproduction

This checkout keeps the original room-search, frontier, topology, and door
decision rules. The compatibility layer only fixes runtime assumptions that no
longer hold on a current NVIDIA GPU and Python stack.

Pinned components:

- Active Room Segmentation: `a941e3a5a1c8e16920a158fa0a9198c95be9c978`
- DETR: `29901c51d7fe8712168b8d0d64351170bc0f83e0`
- SG-Nav source containing Habitat-Lab: `d56863c96dea311aaa67fb0d39a1a8ccc3f0487f`
- Habitat-Lab API: 0.2.1
- Habitat-Sim: 0.2.4
- PyTorch: 2.7.1+cu128

Bootstrap on the wired peer:

```bash
cd ~/Active_room_segmentation
PROXY_URL=http://10.42.0.1:7890 scripts/bootstrap_modern.sh
```

The bootstrap clones the existing `SG_Nav` Conda environment into the dedicated
`~/.conda/envs/active-room-seg` prefix. Downloads use the wired host proxy. The
installer then checks exact source commits, package versions, scene references,
checkpoint SHA256, CUDA availability, and one real CUDA door-detector forward
pass. It does not modify the source `SG_Nav` environment.

Run one visual closed-loop episode on the official Habitat test assets:

```bash
cd ~/Active_room_segmentation
MAX_EPISODE_STEPS=120 scripts/run_habitat_test.sh
```

The runtime has networking disabled and requires CUDA. It selects the active,
local, non-remote `seat0` X11 session instead of accepting an SSH-forwarded
display. The main room-search window carries the exact run UUID, while the
Habitat worker window uses an explicit `- Habitat` suffix. A run is accepted
only from a clean committed source tree, when control and simulator step IDs are
exactly `1..N`, one unique run ID agrees across the process, progress, result,
and physical-window title, and two window captures separated by control steps
have different nonblank pixels. Results and the machine-readable validation
report are stored under `outputs/habitat_test_<timestamp>_<run-id>/`.

This Habitat test-scene run is a native compatibility and closed-loop smoke,
not the paper's Gibson benchmark. The upstream paper evaluates on Gibson.
Gibson scene meshes and the matching PointNav episode data remain subject to
their dataset terms and must be supplied by an authorized user; the bootstrap
does not download, redistribute, or replace them. Once both are present, the
original `tasks/pointnav_gibson.yaml` configuration remains the benchmark entry
point.

For an authorized local Gibson installation, run the deterministic visual
entry point:

```bash
cd ~/Active_room_segmentation
SCENE_ID=Swormville MAX_EPISODE_STEPS=2500 \
  scripts/run_gibson_visual.sh
```

Before launching, it verifies the accepted 10.83 GB archive SHA256, the complete
492-scene GLB/navmesh inventory, the 403 MB PointNav tree SHA256, every
referenced scene pair, and the selected episode's navigability and recorded
geodesic distance. The selected extracted GLB and navmesh must match their
archive entries byte for byte. It then writes a deterministic one-episode
dataset derived from the official validation split into an immutable run-ID
scope and copies that dataset plus its manifest into the run directory before
starting. Habitat reads that run-local dataset directly; there is no shared
mutable generated-dataset path in the execution chain. After Habitat loads the
episode, the runner recomputes a canonical contract digest over its start pose,
rotation, goals, navigation info, room and shortest-path fields.

`MAX_EPISODE_STEPS` is a hard upper bound. The Gibson entry point ends as soon
as topology exploration completes instead of padding the episode with repeated
turns. Strict door-crossing validation cannot be disabled for this entry point.
The validator binds the episode, scene and asset hashes to the result, checks
the complete dashboard-frame cadence, and monitors the process-owned physical
X11 client ID, PID and title until shutdown. The presence of the Gibson context
artifacts automatically enables strict validation: run and validator commits
must match, natural topology completion must be explicit, and aborted
exploration cannot be reported as success. A terminal dashboard render is
mandatory, including a cadence-aligned final frame. Physical-window pixels are
captured after application-rendered step 5, 10, and 30 handshakes, every 100
steps thereafter, and through an explicit terminal handshake. The validator
binds each stage capture to its saved render frame. The strict live canvas is
locked at 3200x1800 and cannot be resized; every captured X11 client image must
retain exactly the same dimensions. Saved cadence frames use a fixed 1600x900
canvas and the final image uses 1920x1080. All four dashboard panels must
remain nonblank and change from the first to terminal render. Strict success
also requires a confirmed crossing edge to survive in the final topology.
Runtime evidence pins the DETR
checkout, door-weight hash, Habitat/SG-Nav checkout and CUDA environment before
and after the episode.
The script never downloads,
substitutes, regenerates or falls back to different Gibson assets.
