#!/usr/bin/env python3
"""Transcribe Tables I-III from the supplied VoxRoom draft, not raw run metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

METHODS = ['Gomez-reprod.', 'DUDE (Incremental)', 'DUDE (Snapshot)', 'ROSE2',
           'Morphological', 'Distance Transform', 'Voronoi', 'TVARS', 'OccuSG', 'SysNav', 'VoxRoom']
ALIASES = dict(zip(['Gomez-reprod.', 'D-Inc.', 'D-Off.', 'ROSE2', 'Morph.', 'Dist.',
                   'Voronoi', 'TVARS', 'OccuSG', 'SysNav', 'VoxRoom'], METHODS))
PROGRESS = ['20%', '40%', '60%', '70%', '80%', '90%', 'Final', 'Average']
METRICS = ['P', 'R', 'F1', 'mIoU']


def parse_tables(page7, page8):
    metric_rows = []
    for line in page7.splitlines():
        match = re.match(r'^\s*(P|R|F1|mIoU)\s+(.+)$', line)
        if match:
            values = re.findall(r'\d+\.\d+', match[2])
            if len(values) == 11:
                metric_rows.append((match[1], values))
    if len(metric_rows) != 32 or [m for m, _ in metric_rows] != METRICS * 8:
        raise ValueError('Unexpected Table I layout; inspect the manuscript before exporting')
    table1 = []
    for event_index, event in enumerate(PROGRESS):
        for method_index, method in enumerate(METHODS):
            table1.append({'progress': event, 'method': method, **{
                metric + '_percent': metric_rows[event_index * 4 + i][1][method_index]
                for i, metric in enumerate(METRICS)}})
    table2 = []
    columns = ['F1_mean_percent', 'F1_SD_pp', 'F1_CV_percent', 'F1_avg_worst_percent',
               'mIoU_mean_percent', 'mIoU_SD_pp', 'mIoU_CV_percent', 'mIoU_avg_worst_percent']
    for line in page7.splitlines():
        parts = line.split()
        if len(parts) == 9 and parts[0] in ALIASES and all(re.fullmatch(r'\d+\.\d+', v) for v in parts[1:]):
            table2.append({'method': ALIASES[parts[0]], **dict(zip(columns, parts[1:]))})
    if len(table2) != 11 or len({r['method'] for r in table2}) != 11:
        raise ValueError('Unexpected Table II layout')
    table3 = []
    for line in page8.splitlines():
        match = re.match(r'^\s*(Full Model|A[1-7]:[^\d]*?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        # Ablation labels contain "2D"/"3D", so use the first floating-point
        # field as the boundary rather than prohibiting digits in the label.
        if match is None:
            match = re.match(r'^\s*(A[1-7]:.*?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
        if match:
            table3.append({'setting': match[1].strip(), **dict(zip([m+'_percent' for m in METRICS], match.groups()[1:]))})
    if len(table3) != 8:
        raise ValueError('Unexpected Table III layout')
    return table1, table2, table3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('docs/results'))
    args = parser.parse_args()
    pages = [subprocess.check_output(['pdftotext', '-f', str(p), '-l', str(p), '-layout', str(args.paper), '-'], text=True)
             for p in (7, 8)]
    tables = parse_tables(*pages)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, rows in zip(('I', 'II', 'III'), tables):
        path = args.out_dir / f'paper_table_{name}.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files.append({'file': path.name, 'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    provenance = {'source_type': 'transcription of maintainer-supplied manuscript tables, not recomputed experiment metrics',
        'title': 'VoxRoom: Room Segmentation from Partial Observations during Robot Exploration',
        'paper_sha256': hashlib.sha256(args.paper.read_bytes()).hexdigest(),
        'table_pages': {'I': 7, 'II': 7, 'III': 8}, 'files': files,
        'notes': ['Values and precision are preserved exactly as printed in the tables.',
                  'Table I main Average and Table III Full Model differ; they are not forcibly reconciled.',
                  'Table III A2 mIoU is 56.3; the conflicting nearby narrative value is not substituted.',
                  'Raw local evaluation CSVs are not modified.',
                  'Real-world hardware/latency were provided separately and supersede draft hardware descriptions.']}
    (args.out_dir / 'paper_tables_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'rows': [len(t) for t in tables], 'output': str(args.out_dir)}))


if __name__ == '__main__':
    main()
