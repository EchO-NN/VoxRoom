#!/usr/bin/env python3
"""Wait for the bounded completion job, then score, download and export CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'echo@fe80::630c:c787:c8e7:5fcb%enp129s0'
RBASE = '/media/echo/data/voxroom_dude_completion_20260910'
RFROZEN = '/media/echo/data/voxroom_dude_corrected_20260910'
RPYTHON = '/home/echo/miniforge3/envs/sgnav-isaac/bin/python'
SSH = ['ssh', '-6', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', REMOTE]
LOCAL = ROOT / 'results/dude_completion_20260911'
OUTPUT = ROOT / 'results/dude_occusg_csv_completed_20260911'


def state(value):
    value['time'] = time.time()
    LOCAL.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL / 'collector_status.json.tmp'
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(LOCAL / 'collector_status.json')
    print(json.dumps(value), flush=True)


def verify_archive(excluded_count=0):
    with (OUTPUT / 'files.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            path = OUTPUT / row['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            with path.open(encoding='utf-8-sig', newline='') as data:
                assert len(list(csv.DictReader(data))) == int(row['row_count'])
    with (OUTPUT / 'method_catalog.csv').open(encoding='utf-8-sig', newline='') as stream:
        methods = list(csv.DictReader(stream))
        assert len(methods) == 5
        for row in methods:
            omitted = excluded_count if row['method_id'] == 'DUDE_offline_vertical_tau2p5' else 0
            assert row['run_status'] == ('complete_with_user_exclusion' if omitted else 'complete')
            assert int(row['available_checkpoints']) == 462-omitted
            assert int(row['user_excluded_checkpoints']) == omitted
    with zipfile.ZipFile(OUTPUT.with_suffix('.zip')) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert archive.read(name) == (OUTPUT / name).read_bytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-hours', type=float, default=12)
    args = parser.parse_args()
    try:
        deadline = time.monotonic() + args.max_hours*3600
        while time.monotonic() < deadline:
            process = subprocess.run(SSH + ['cat ' + RBASE + '/status.json'], capture_output=True, text=True, timeout=30)
            if process.returncode != 0:
                state({'status': 'waiting_for_connection', 'error': process.stderr[-1000:]})
                time.sleep(30)
                continue
            remote = json.loads(process.stdout)
            state({'status': 'waiting_for_predictions', 'remote': remote})
            if remote['status'] in ('predictions_complete', 'predictions_complete_with_exclusions'):
                break
            if remote['status'] in ('failed', 'incomplete_after_retries'):
                raise RuntimeError(remote)
            time.sleep(30)
        else:
            raise TimeoutError('Collector deadline reached; remote predictions left untouched')
        excluded_count = int(remote.get('user_excluded_predictions', 0))
        state({'status': 'scoring_saved_predictions', 'expected_available': 924-excluded_count, 'user_excluded': excluded_count})
        command = (f'PYTHONPATH={RFROZEN}/code OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 '
                   f'{RPYTHON} -u {RBASE}/code/export_existing_corrected_dude_metrics.py '
                   f'--root {RFROZEN} --output {RBASE}/metrics_complete_20260911 --require-complete')
        if excluded_count:
            command += f' --allowed-missing-json {RBASE}/excluded_predictions.json'
        check = subprocess.run(SSH + [f'test -f {RBASE}/metrics_complete_20260911/export_audit.json'], timeout=30)
        if check.returncode != 0:
            subprocess.run(SSH + [command], check=True, timeout=7200)
        LOCAL.mkdir(parents=True, exist_ok=True)
        rsync = ['rsync', '-az', '--timeout=60', '-e', 'ssh -6 -o BatchMode=yes -o ConnectTimeout=8',
                 '--exclude=tmp/', '--exclude=ros_logs/',
                 'echo@[fe80::630c:c787:c8e7:5fcb%enp129s0]:' + RBASE + '/', str(LOCAL) + '/']
        subprocess.run(rsync, check=True, timeout=600)
        audit = json.loads((LOCAL / 'metrics_complete_20260911/export_audit.json').read_text())
        assert audit['requested_scope_complete'] and audit['unexpected_missing_predictions'] == 0
        assert audit['available_predictions'] == 924-excluded_count and audit['missing_predictions'] == excluded_count
        assert audit['authorized_exclusions'] == excluded_count
        state({'status': 'building_complete_csv_package'})
        if not (OUTPUT / 'files.csv').exists():
            organize = [sys.executable, str(ROOT / 'scripts/organize_dude_occusg_csv.py'),
                            '--dude-export', str(LOCAL / 'metrics_complete_20260911'), '--output', str(OUTPUT),
                            '--completion-audit', str(LOCAL / 'completed_prediction_manifest.json')]
            if excluded_count:
                organize += ['--excluded-predictions-json', str(LOCAL / 'excluded_predictions.json')]
            subprocess.run(organize, check=True, timeout=180)
        verify_archive(excluded_count)
        state({'status': 'complete_with_user_exclusion' if excluded_count else 'complete',
               'csv_directory': str(OUTPUT), 'archive': str(OUTPUT.with_suffix('.zip')),
               'new_offline_predictions': 196-excluded_count, 'total_dude_predictions': 924-excluded_count,
               'user_excluded_predictions': excluded_count, 'unexpected_missing_predictions': 0})
    except BaseException as exc:
        state({'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
