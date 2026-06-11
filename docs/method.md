# VoxRoom Method Notes

This document summarizes the implementation path used by the current paper
repository.

## 1. Online Voxel Mapping

The robot receives RGB-D observations from Isaac Sim. Depth rays are integrated
into a 3D voxel grid using DDA traversal.

For the paper configuration, a depth hit marks only the endpoint voxel itself:

```yaml
endpoint_splat_xy_radius_cells: 0
endpoint_splat_z_radius_cells: 0
```

This means the endpoint update is 1x1 in the projected map. The older 3x3
endpoint expansion is disabled.

## 2. Navigation Projection

The voxel map is projected to 2D navigation layers:

- navigation free,
- navigation occupied,
- navigation unknown.

Door seed reasoning uses the no-clearance navigation free map rather than a
clearance-dilated traversibility map. This keeps thin doorway evidence visible.

The endpoint hysteresis projection also uses no splat:

```yaml
occupied_endpoint_xy_splat_radius_cells: 0
occupied_endpoint_z_splat_radius_cells: 0
```

## 3. Wall Evidence

The room-segmentation module derives wall-column evidence from voxel occupancy
and vertical structure. Small wall gaps can be completed before door reasoning.
This is a wall-column completion step, not the old wall-line extension stage.

## 4. Door Seed Detection

Door seeds are detected from narrow navigation-free bands with supporting
occupied or out-of-range upper evidence near the estimated ceiling. The current
height rule uses:

```text
door_seed_height = 0.95 * estimated_ceiling_height
```

The ceiling estimator selects a dominant occupied z layer above the lower
height cutoff, with a fallback only when the ceiling cannot be estimated.

## 5. Door Completion

Door seed clusters are grouped, bridged along the same line when allowed, and
checked with geometric line constraints. The accepted door completion lines are
then used as room separators.

The current method keeps the linear shape constraints and stateful door memory.
Once a door line has become a stable partition, later fragmented raw seeds do
not immediately delete or replace it. Stable door geometry is recomputed only
when the ceiling estimate changes enough to invalidate the seed-height test.

## 6. No Wall-Line Extension

The old second-stage wall-line extension is disabled:

```yaml
voxel_step2:
  enabled: false
```

This is important: room partitions are produced by completed door lines, not by
extending wall lines across free space.

## 7. Room Partition

Accepted door lines are rasterized into the room segmentation map. Connected
components in the explored free-space domain become room masks. The runtime
saves snapshot `.npz` files with both intermediate evidence layers and final
room labels.

