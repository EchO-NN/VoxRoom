#!/usr/bin/env python3
"""Build a deterministic caller without changing any upstream algorithm body.

Only one header access specifier is changed in an isolated build copy; main is
renamed by a compiler definition. Both translation units use that same header.
The upstream repository and its original executable are never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

COMMIT = '0fa15cc8bc18be7409272fb65fa514a9a5bca6b0'
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, default=Path('/home/joey/SysNav'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    up, out = args.upstream.resolve(), args.output.resolve()
    assert subprocess.check_output(['git', '-C', str(up), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(up), 'diff', 'HEAD', '--', 'src/exploration_planner/tare_planner'], text=True).strip()
    src = up / 'src/exploration_planner/tare_planner'
    build = up / 'build/tare_planner'
    cmake = build / 'CMakeFiles/room_segmentation.dir'
    out.mkdir(parents=True, exist_ok=True)
    original_header = src / 'include/room_segmentation/room_segmentation_node.h'
    header = original_header.read_text()
    assert header.count('\nprivate:\n') == 1
    copied_header = out / 'include/room_segmentation/room_segmentation_node.h'
    copied_header.parent.mkdir(parents=True, exist_ok=True)
    # Mechanical access-only adaptation; no fields/methods/algorithm code change.
    copied_header.write_text(header.replace('\nprivate:\n', '\npublic:\n'))
    flags = {}
    for line in (cmake / 'flags.make').read_text().splitlines():
        if line.startswith('CXX_'):
            name, value = line.split(' = ', 1)
            flags[name] = shlex.split(value)
    compile_flags = ['-I' + str(out / 'include')]
    for key in ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'):
        compile_flags += flags[key]
    source = src / 'src/room_segmentation/room_segmentation.cpp'
    caller = ROOT / 'scripts/sysnav_snapshot_bridge.cpp'
    commands = [
        ['/usr/bin/c++', *compile_flags, '-Dmain=sysnav_original_main', '-c', str(source), '-o', str(out / 'upstream.o')],
        ['/usr/bin/c++', *compile_flags, '-c', str(caller), '-o', str(out / 'bridge.o')],
    ]
    link = shlex.split((cmake / 'link.txt').read_text())
    obj = next(i for i, part in enumerate(link) if part.endswith('room_segmentation.cpp.o'))
    link[obj:obj+1] = [str(out / 'upstream.o'), str(out / 'bridge.o')]
    link[link.index('-o')+1] = str(out / 'sysnav_snapshot_bridge')
    commands.append(link)
    for i, command in enumerate(commands):
        print(f'Build stage {i+1}/3', flush=True)
        subprocess.run(command, cwd=build, check=True)
    (out / 'build_provenance.json').write_text(json.dumps({
        'upstream_commit': COMMIT, 'upstream_source_sha256': digest(source),
        'upstream_header_sha256': digest(original_header), 'adapted_header_sha256': digest(copied_header),
        'header_change': 'one private access specifier changed to public in build-only copy',
        'algorithm_body_changes': 0, 'caller_sha256': digest(caller),
        'binary_sha256': digest(out / 'sysnav_snapshot_bridge'),
        'commands': commands,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
