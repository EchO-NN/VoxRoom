#!/usr/bin/env python3
"""Append missing offline predictions using the frozen corrected DUDE runner.

Existing predictions are content-hashed and never overwritten. Only transport
timeouts change; native parameters, input grids and metric contracts do not.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time

import numpy as np

OLD = Path('/media/echo/data/voxroom_dude_corrected_20260910')
EXT = Path('/home/echo/VoxRoom-Online-exp/external_baselines')
INPUT_KEYS = {'step', 'occupancy_map', 'observed_free_mask', 'obstacle_mask', 'unknown_mask',
              'voxel_vertical_free_xy', 'height_profile_vertical_free_xy', 'vertical_free_room_domain',
              'voxel_wall_xy', 'structural_wall_clean', 'roomseg_sanitized_wall', 'navigation_free_room_domain',
              'roomseg_eval_reference_explorable_mask', 'roomseg_eval_explored_reference_mask'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def publish_exclusive(path, payload):
    """Atomic visibility with no overwrite, even if another writer races us."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.completion.tmp')
    with tmp.open('xb') as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = OLD / 'code'
    os.environ.update(PYTHONPATH=str(frozen), VOXROOM_REPO_ROOT=str(frozen),
                      ROS_MASTER_URI='http://127.0.0.1:11702', ROS_IP='127.0.0.1', ROS_HOSTNAME='localhost',
                      ROS_LOG_DIR=str(out / 'ros_logs'), TMPDIR=str(out / 'tmp'),
                      PATH=str(EXT / 'ros_noetic_env/bin') + ':' + os.environ['PATH'],
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for folder in ('ros_logs', 'tmp', 'native_logs'):
        (out / folder).mkdir(exist_ok=True)
    from voxroom_online.isaac_runtime.baselines.offline.dude_runner import DudeIncrementalRunner
    from voxroom_online.isaac_runtime.baselines.data_contract import resolve_map_info
    from voxroom_online.isaac_runtime.baselines.ros_grid_io import snapshot_to_ros_occupancy_grid
    from voxroom_online.isaac_runtime.baselines.mask_io import SEGMENTATION_INPUT_MODE_KEY, enforce_room_mask_contract, build_segmentation_domain_from_source
    from voxroom_online.isaac_runtime.baselines.offline.dude_image_io import DUDE_OUTPUT_COORDINATE_CONTRACT
    from voxroom_online.isaac_runtime.comparison.metadata_gate import assert_main_experiment_metadata

    records = []
    for dataset in ('interioragent', 'grscene'):
        audit = json.loads((OLD / dataset / 'input_audit.json').read_text())
        assert digest(EXT / 'dude_ws/devel/lib/inc_dude/inc_dude') == audit['native_binary_sha256']
        for relative, expected in audit['code_sha256'].items():
            if relative.startswith('voxroom_online/'):
                assert digest(frozen / relative) == expected, relative
        index = json.loads((OLD / dataset / 'inputs/index_voxroom.json').read_text())
        for ep in index['episodes']:
            assert ep['scene_id'] not in audit['excluded_scene_ids']
            for snap in ep['snapshots']:
                for method in ('dude_incremental', 'dude_offline'):
                    records.append({'dataset': dataset, 'method': method, 'scene_id': ep['scene_id'],
                                    'episode_uid': ep['episode_uid'], **snap,
                                    'prediction_path': str(OLD / dataset / 'replay/predictions' / method / ep['episode_uid'] / (snap['coverage_event_id'] + '.npz'))})
    assert len(records) == 924
    plan_path = out / 'plan.json'
    if not plan_path.exists():
        existing = {r['prediction_path']: digest(r['prediction_path']) for r in records if Path(r['prediction_path']).exists()}
        missing = [r for r in records if r['prediction_path'] not in existing]
        assert len(existing) == 728 and len(missing) == 196
        assert all(r['dataset'] == 'grscene' and r['method'] == 'dude_offline' for r in missing)
        save_json(plan_path, {'existing_sha256': existing, 'missing': missing, 'total_predictions': len(records),
                             'input': 'raw_vertical_free', 'concavity_threshold_m': 2.5,
                             'transport_timeouts_s': [900, 1800], 'frozen_code': str(frozen),
                             'driver_sha256': digest(__file__), 'created_at': time.time()})
    plan = json.loads(plan_path.read_text())
    skip_path = out / 'excluded_predictions.json'
    exclusions = json.loads(skip_path.read_text())['excluded_predictions'] if skip_path.exists() else []
    def record_key(row):
        return row['dataset'], row['method'], row['episode_uid'], row['coverage_event_id']
    excluded_keys = {record_key(r) for r in exclusions}
    assert len(excluded_keys) == len(exclusions)
    assert excluded_keys <= {record_key(r) for r in plan['missing']}
    assert all(not Path(r['prediction_path']).exists() for r in plan['missing'] if record_key(r) in excluded_keys)
    targets = [r for r in plan['missing'] if record_key(r) not in excluded_keys]
    for path, expected in plan['existing_sha256'].items():
        assert digest(path) == expected, ('existing prediction changed', path)

    def status(state, **extra):
        done = sum(Path(r['prediction_path']).is_file() for r in plan['missing'])
        value = {'status': state, 'completed_missing': done, 'total_missing': len(plan['missing']),
                 'user_excluded_predictions': len(exclusions), 'remaining_required_predictions': len(targets)-done,
                 'total_predictions_available': len(plan['existing_sha256']) + done,
                 'total_predictions_expected': len(records), 'time': time.time(), **extra}
        save_json(out / 'status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    class LoggedRunner(DudeIncrementalRunner):
        diagnostic_path = None

        def segment_snapshot(self, snapshot_path, arrays):
            # The native process can abort while the ROS bridge waits for its
            # message. Terminate only this runner's waiting bridge on that abort.
            stop = threading.Event()
            def watch_native_exit():
                while not stop.wait(.25):
                    native = self._node_process
                    if native is None or native.poll() is None:
                        continue
                    children = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().split()
                    for child in children:
                        try:
                            argv = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
                            module = b'voxroom_online.isaac_runtime.baselines.offline.ros_entrypoints.dude_ros_node'
                            if module in argv and b'--request' in argv:
                                request = Path(os.fsdecode(argv[argv.index(b'--request')+1])).resolve()
                                if (out / 'tmp').resolve() in request.parents:
                                    os.kill(int(child), signal.SIGTERM)
                                    return
                        except (FileNotFoundError, ProcessLookupError):
                            continue
            watcher = threading.Thread(target=watch_native_exit, daemon=True)
            watcher.start()
            try:
                return super().segment_snapshot(snapshot_path, arrays)
            finally:
                stop.set()
                watcher.join(timeout=2)

        def _stop_original_node(self, *, keep_roscore=False):
            if self._node_output is not None and self.diagnostic_path is not None:
                self._node_output.flush()
                self._node_output.seek(0)
                with self.diagnostic_path.open('wb') as stream:
                    shutil.copyfileobj(self._node_output, stream)
            return super()._stop_original_node(keep_roscore=keep_roscore)

    failures = []
    try:
        for attempt, timeout in enumerate((900., 1800.), 1):
            for row in targets:
                target = Path(row['prediction_path'])
                if target.exists():
                    continue
                uid, event = row['episode_uid'], row['coverage_event_id']
                source = Path(row['snapshot_path'])
                status('running', scene=row['scene_id'], event=event, attempt=attempt, timeout_s=timeout)
                runner = LoggedRunner(repo_root=EXT / 'dude_ws/src/Incremental_DuDe_ROS', dude_ws=EXT / 'dude_ws',
                                      concavity_threshold_m=2.5, use_incremental=False, fallback_python=False,
                                      map_resolution_m=.05, timeout_s=timeout,
                                      ros_setup=str(EXT / 'ros_noetic_env/setup.bash'),
                                      ros_python=str(EXT / 'ros_noetic_env/bin/python'))
                runner.diagnostic_path = out / 'native_logs' / (uid + '_' + event + '_attempt' + str(attempt) + '.log')
                start = time.monotonic()
                try:
                    with np.load(source, allow_pickle=False) as data:
                        arrays = {k: data[k] for k in data.files if k in INPUT_KEYS or k.startswith('map_')}
                    arrays[SEGMENTATION_INPUT_MODE_KEY] = np.asarray('raw_vertical_free')
                    grid = snapshot_to_ros_occupancy_grid(arrays)
                    anchor = OLD / row['dataset'] / 'replay/predictions/dude_incremental' / uid / (event + '.npz')
                    with np.load(anchor, allow_pickle=False) as existing:
                        np.testing.assert_array_equal(grid, existing['dude_ros_occupancy_grid'])
                        anchor_meta = json.loads(str(existing['baseline_metadata_json']))
                    map_info = resolve_map_info(snapshot_arrays=arrays, default_resolution_m=.05).to_metadata()
                    for field in ('resolution_m', 'min_x', 'min_y', 'max_x', 'max_y', 'height', 'width'):
                        assert map_info[field] == anchor_meta['map_info'][field]
                    runner.start_scene(uid)
                    result = runner.segment_snapshot(source, arrays)
                    metadata = {**result.metadata, 'segmentation_input_mode': 'raw_vertical_free',
                                'segmentation_input_source_key': build_segmentation_domain_from_source(arrays)[1],
                                'coverage_reference_used_as_segmentation_input': False,
                                'completion_run': str(out), 'completion_timeout_s': timeout,
                                'input_identical_to_saved_online': True, 'input_grid_sha256': hashlib.sha256(grid.tobytes()).hexdigest()}
                    assert_main_experiment_metadata(metadata, 'dude_offline')
                    assert metadata['parameters'] == {'concavity_threshold_m': 2.5, 'use_incremental': False}
                    assert metadata['output_coordinate_contract'] == DUDE_OUTPUT_COORDINATE_CONTRACT
                    np.testing.assert_array_equal(grid, result.debug_arrays['dude_ros_occupancy_grid'])
                    np.testing.assert_array_equal(np.flipud(result.debug_arrays['dude_tagged_image_native']).astype(np.int32), result.debug_arrays['dude_labels_source_frame'])
                    np.testing.assert_array_equal(result.label_map, enforce_room_mask_contract(result.debug_arrays['dude_labels_source_frame'], arrays, clip_to_eval_domain=True))
                    publish_exclusive(target, {'final_room_label_map': np.asarray(result.label_map, dtype=np.int32),
                                               'baseline_metadata_json': np.asarray(json.dumps(metadata)),
                                               'source_snapshot_path': np.asarray(str(source)),
                                               'coverage_event_id': np.asarray(event), 'step': np.asarray(int(row['step'])),
                                               **{k: v for k, v in result.debug_arrays.items() if k.startswith('dude_')}})
                    status('running', last_completed_scene=row['scene_id'], last_completed_event=event,
                           seconds=time.monotonic()-start, existing_predictions_untouched=True)
                except Exception as exc:
                    failures.append({'scene': row['scene_id'], 'event': event, 'attempt': attempt,
                                     'error': repr(exc), 'seconds': time.monotonic()-start})
                    save_json(out / 'attempt_errors.json', failures)
                    print(json.dumps(failures[-1]), flush=True)
                finally:
                    runner.end_scene()
        missing = [r for r in targets if not Path(r['prediction_path']).is_file()]
        for path, expected in plan['existing_sha256'].items():
            assert digest(path) == expected, ('existing prediction changed', path)
        if missing:
            status('incomplete_after_retries', remaining=[r['prediction_path'] for r in missing])
            raise RuntimeError(f'{len(missing)} predictions still missing; no empty/fallback predictions saved')
        save_json(out / 'completed_prediction_manifest.json', {'records': records, 'checkpoint_count': len(records)-len(exclusions),
                  'original_expected_checkpoint_count': len(records), 'excluded_predictions': exclusions,
                  'excluded_prediction_count': len(exclusions), 'requested_scope_complete': True,
                  'original_728_sha256_unchanged': True, 'new_offline_predictions': len(targets),
                  'new_prediction_sha256': {r['prediction_path']: digest(r['prediction_path']) for r in targets}})
        status('predictions_complete_with_exclusions' if exclusions else 'predictions_complete', existing_predictions_untouched=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
