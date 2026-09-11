"""Pure, testable voxel/coordinate adapter for the aligned SysNav replay."""
from pathlib import Path
import hashlib
import numpy as np
import yaml

METHOD = 'sysnav_official_sim_aligned_snapshot_v1'
COMMIT = '0fa15cc8bc18be7409272fb65fa514a9a5bca6b0'
CONTRACT = 'observed_occupied_only__scan_free_state_segment__native_free_footprint_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def parameters(upstream):
    path = Path(upstream) / 'src/exploration_planner/tare_planner/config/matterport_sim.yaml'
    original = yaml.safe_load(path.read_text())['room_segmentation']['ros__parameters']
    result = dict(original)
    # Debug PNG output is disabled; all geometric parameters remain unchanged.
    result['isDebug'] = False
    # Explicitly record these upstream constructor defaults (not in the YAML).
    result.update(region_growing_radius=15.0, min_room_size=40, wall_thres_height_=0.1,
                  outward_distance_1=0.3, distance_angel_threshold=0.3,
                  angle_threshold_deg=6.0, **{'rolling_occupancy_grid.resolution_x': 0.2})
    return result, {'path': str(path), 'sha256': sha256(path), 'original': original}


def geometry(z):
    resolution = float(z['map_resolution_m'])
    shape = tuple(z['voxel_nav_free_xy'].shape)
    if 'map_bounds_xyxy_m' in z:
        bounds = np.asarray(z['map_bounds_xyxy_m'], dtype=np.float64)
    elif 'map_origin_xy_m' in z:
        origin = np.asarray(z['map_origin_xy_m'], dtype=np.float64)
        bounds = np.r_[origin, origin + np.array(shape[::-1]) * resolution]
    else:
        raise ValueError('World grid metadata missing')
    if len(bounds) != 4 or not np.allclose(bounds[2:] - bounds[:2], np.array(shape[::-1]) * resolution, atol=1e-4):
        raise ValueError('Grid bounds/shape/resolution disagree')
    return resolution, shape, bounds


def free_points(nav, bounds, resolution, robot_z, auxiliary_resolution=0.2):
    """Represent only fully known-free 0.2 m footprints, without expanding them.

    Native updateFreespace expands each input point FORWARD by 2x2 0.1 m cells.
    Its input must therefore reference the footprint's lower corner, not its
    centre. Occupied-stream coordinates separately follow its centre convention.
    This is a conservative saved-map substitute, NOT real viewpoint/LOS replay.
    """
    rows, cols = np.indices(nav.shape)
    xy = np.column_stack((bounds[0] + (cols.ravel()+.5)*resolution,
                          bounds[1] + (rows.ravel()+.5)*resolution))
    bins, inverse, counts = np.unique(np.floor(xy / auxiliary_resolution).astype(np.int32),
                                      axis=0, return_inverse=True, return_counts=True)
    free_counts = np.bincount(inverse, weights=nav.ravel(), minlength=len(bins))
    expected = round(auxiliary_resolution/resolution)**2
    valid = (counts == expected) & (free_counts == counts)
    result = np.zeros((np.count_nonzero(valid), 4), dtype=np.float32)
    result[:, :2] = bins[valid] * auxiliary_resolution
    result[:, 2] = robot_z
    result[:, 3] = 1
    return result


def load_clouds(snapshot, params):
    with np.load(snapshot, allow_pickle=False) as z:
        resolution, shape, bounds = geometry(z)
        nav = np.asarray(z['voxel_nav_free_xy'], dtype=bool)
        state = np.asarray(z['voxel_occupancy_state_zyx'], dtype=np.uint8)
        heights = np.asarray(z['voxel_occupancy_z_centers_m'], dtype=np.float64)
        pose_key = next((k for k in ('base_pose_world_xyzyaw', 'demo_pose_world') if k in z), None)
        if pose_key is None:
            raise ValueError('Saved robot pose required; do not silently use zero')
        pose = np.asarray(z[pose_key], dtype=np.float64).ravel()
    if state.shape[1:] != shape or heights.shape != (state.shape[0],) or len(pose) < 3 or not np.isfinite(pose).all():
        raise ValueError('Invalid voxel/pose geometry')
    zi, row, col = np.nonzero(state == 2)
    xyz = np.column_stack((bounds[0]+(col+.5)*resolution, bounds[1]+(row+.5)*resolution, heights[zi]))
    # Native callback uses precisely this ceiling cutoff. Do not quantize the
    # registered scan to coarse cell centres: upstream PCL computes centroids.
    ceiling = pose[2] + params['ceilingHeight_']
    keep = xyz[:, 2] <= ceiling
    xyz = xyz[keep]
    if len(xyz) < 5:
        raise ValueError('Insufficient observed occupied points for five scan chunks')
    native_res = params['room_resolution']
    dimensions = np.array([params['room_x'], params['room_y'], params['room_z']])
    idx = np.floor(xyz / native_res + dimensions/2)
    if np.any(idx < 0) or np.any(idx >= dimensions):
        raise ValueError('Observed input falls outside official grid; refusing silent clamping')
    scan = np.column_stack((xyz, np.ones(len(xyz)))).astype(np.float32)
    free = free_points(nav, bounds, resolution, pose[2])
    auxiliary_resolution = params['rolling_occupancy_grid.resolution_x']
    cells = np.unique(np.floor(xyz/auxiliary_resolution).astype(np.int32), axis=0)
    centres = (cells+.5)*auxiliary_resolution
    low, high = pose[2]-params['kViewPointCollisionMarginZMinus'], pose[2]+params['kViewPointCollisionMarginZPlus']
    centres = centres[(centres[:, 2] >= low) & (centres[:, 2] <= high)]
    occupied = np.column_stack((centres, np.zeros(len(centres)))).astype(np.float32)
    meta = {'method': METHOD, 'input_contract': CONTRACT, 'source_shape_yx': list(shape),
            'source_resolution_m': resolution, 'source_bounds_xyxy_m': bounds.tolist(),
            'base_pose_world_xyzyaw': pose.tolist(), 'pose_key': pose_key,
            'observed_occupied_point_count': len(scan), 'ceiling_filtered_point_count': int(np.count_nonzero(~keep)),
            'synthetic_floor_point_count': 0, 'freespace_point_count': len(free),
            'occupied_state_point_count': len(occupied), 'source_nav_free_cell_count': int(nav.sum()),
            'free_update_footprint': 'forward 2x2 native cells, lower-corner coordinates',
            'missing_native_streams': 'original sensor trajectory, per-frame LOS/viewpoint state not recoverable from snapshots'}
    return {'scan': scan, 'free': free, 'occupied': occupied}, meta, nav


def project_labels(labels_xy, meta, native_resolution):
    h, w = meta['source_shape_yx']
    bounds, res = meta['source_bounds_xyxy_m'], meta['source_resolution_m']
    x = bounds[0] + (np.arange(w)+.5)*res
    y = bounds[1] + (np.arange(h)+.5)*res
    ix = np.floor(x/native_resolution + labels_xy.shape[0]/2).astype(int)
    iy = np.floor(y/native_resolution + labels_xy.shape[1]/2).astype(int)
    if min(ix.min(), iy.min()) < 0 or ix.max() >= labels_xy.shape[0] or iy.max() >= labels_xy.shape[1]:
        raise ValueError('Projection exceeds native map')
    return labels_xy[np.ix_(ix, iy)].T.copy()


def project_doors(doors, meta, native_resolution):
    # Rasterize on the native grid then use exactly the same transform as labels.
    dimensions = meta['native_dimensions_xy']
    mask = np.zeros(dimensions, dtype=np.uint8)
    if doors.size:
        ij = np.rint(doors[:, :2]/native_resolution + np.array(dimensions)/2).astype(int)
        if np.any(ij < 0) or np.any(ij >= np.array(dimensions)):
            raise ValueError('Door output outside native grid')
        mask[ij[:, 0], ij[:, 1]] = 1
    return project_labels(mask, meta, native_resolution).astype(bool)
