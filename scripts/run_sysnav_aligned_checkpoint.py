#!/usr/bin/env python3
"""Immutable, deterministic official-SysNav saved-snapshot inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from sysnav_aligned_input import COMMIT, parameters, load_clouds, project_labels, project_doors, sha256


def run(snapshot, output, bridge, upstream, timeout=180, debug=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    params, profile = parameters(upstream)
    params['isDebug'] = debug
    clouds, meta, nav = load_clouds(snapshot, params)
    source_sha = sha256(snapshot)
    meta.update(snapshot=str(snapshot.resolve()), source_sha256=source_sha, sysnav_commit=COMMIT,
                parameters=params, official_profile=profile, bridge_sha256=sha256(bridge),
                native_dimensions_xy=[params['room_x'], params['room_y']])
    with tempfile.TemporaryDirectory(prefix='sysnav_aligned_', dir=output.parent) as scratch:
        temp = Path(scratch)
        for i, points in enumerate(np.array_split(clouds['scan'], 5)):
            points.astype('<f4').tofile(temp/f'scan_{i}.bin')
        clouds['free'].astype('<f4').tofile(temp/'free.bin')
        clouds['occupied'].astype('<f4').tofile(temp/'occupied.bin')
        np.savetxt(temp/'pose.txt', np.array(meta['base_pose_world_xyzyaw'][:3])[None], fmt='%.10f')
        command = [str(bridge.resolve()), str(temp), str(temp/'output'), '--ros-args']
        for key, value in params.items():
            command += ['-p', f'{key}:={str(value).lower() if isinstance(value,bool) else value}']
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = f'{upstream}/build/tare_planner:/opt/ros/jazzy/lib:' + env.get('LD_LIBRARY_PATH','')
        env['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
        env['OMP_NUM_THREADS'] = '1'
        with output.with_suffix('.native.log').open('w') as log:
            subprocess.run(command, cwd=temp, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
        audit = json.loads((temp/'output/native_audit.json').read_text())
        raw = np.fromfile(temp/'output/labels.i32', dtype='<i4').reshape(audit['rows'], audit['cols'])
        doors = np.fromfile(temp/'output/doors.f32', dtype='<f4').reshape(-1,3)
        labels = project_labels(raw, meta, params['room_resolution'])
        projected_doors = project_doors(doors, meta, params['room_resolution'])
        meta.update(native_audit=audit, native_room_count=int(np.unique(raw[raw>0]).size),
                    projected_before_clip_count=int(np.unique(labels[labels>0]).size))
        labels[~nav] = 0
        meta.update(projected_room_count=int(np.unique(labels[labels>0]).size),
                    projected_labeled_cell_count=int(np.count_nonzero(labels)),
                    runtime_seconds=time.monotonic()-started)
        if debug:
            import shutil
            for image in temp.glob('*.png'):
                target = output.parent / (output.stem + '_debug')
                target.mkdir(exist_ok=True)
                shutil.copy2(image, target/image.name)
        temporary = output.with_suffix('.partial.npz')
        np.savez_compressed(temporary, sysnav_room_label_map=labels, sysnav_door_line_map=projected_doors,
                            metadata_json=np.asarray(json.dumps(meta, sort_keys=True)))
        temporary.replace(output)
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('snapshot', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--bridge', type=Path, required=True)
    p.add_argument('--upstream', type=Path, default=Path('/home/joey/SysNav'))
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--debug', action='store_true')
    a=p.parse_args()
    print(json.dumps(run(a.snapshot,a.output,a.bridge,a.upstream,a.timeout,a.debug)))
