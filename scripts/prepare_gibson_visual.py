#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import habitat_sim
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from run_context_contract import episode_contract_sha256


EXPECTED_GIBSON_ARCHIVE_SHA256 = (
    "b8280c7fec1175794656bf274a94caa77a980e32df8bce6995aad41f35b910fe"
)
EXPECTED_GIBSON_ARCHIVE_SIZE = 10833075327
EXPECTED_GIBSON_SCENE_COUNT = 492
EXPECTED_GIBSON_ARCHIVE_ENTRY_COUNT = EXPECTED_GIBSON_SCENE_COUNT * 2
EXPECTED_POINTNAV_FILE_COUNT = 75
EXPECTED_POINTNAV_TREE_SHA256 = (
    "4da848fa38be405123092f3ba74c3e7503acd3ccc150614f5a5eceeae006966e"
)
EXPECTED_POINTNAV_REQUIRED_SCENE_COUNT = 86
EXPECTED_POINTNAV_COUNTS = {
    "train_scene_files": 72,
    "val": 994,
    "val_mini": 30,
}
SCENE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def dataset_tree_sha256(dataset_root):
    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(dataset_root).as_posix()
        digest.update(
            "{}\0{}\0{}\n".format(
                relative_path,
                path.stat().st_size,
                sha256(path),
            ).encode("utf-8")
        )
    return digest.hexdigest(), len(files)


def write_json_atomic(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_stream,
            mode="wb",
            mtime=0,
        ) as stream:
            stream.write(
                (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            )
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
    os.replace(temporary, path)


def load_episodes(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        raise RuntimeError("Invalid PointNav payload in {}".format(path))
    return episodes


def select_scene_episode(episodes, scene_name, episode_index):
    if not SCENE_NAME_PATTERN.fullmatch(scene_name):
        raise ValueError("Invalid Gibson scene name: {!r}".format(scene_name))
    matches = [
        episode
        for episode in episodes
        if Path(episode.get("scene_id", "")).stem == scene_name
    ]
    if not matches:
        raise RuntimeError("No PointNav episodes found for {}".format(scene_name))
    if episode_index < 0 or episode_index >= len(matches):
        raise IndexError(
            "Episode index {} is outside 0..{} for {}".format(
                episode_index,
                len(matches) - 1,
                scene_name,
            )
        )
    return matches[episode_index], len(matches)


def pointnav_inventory(dataset_root):
    tree_digest, file_count = dataset_tree_sha256(dataset_root)
    if file_count != EXPECTED_POINTNAV_FILE_COUNT:
        raise RuntimeError(
            "Unexpected Gibson PointNav file count: {}, expected {}".format(
                file_count,
                EXPECTED_POINTNAV_FILE_COUNT,
            )
        )
    if tree_digest != EXPECTED_POINTNAV_TREE_SHA256:
        raise RuntimeError(
            "Gibson PointNav tree SHA256 mismatch: {}, expected {}".format(
                tree_digest,
                EXPECTED_POINTNAV_TREE_SHA256,
            )
        )
    val_episodes = load_episodes(dataset_root / "val" / "val.json.gz")
    val_mini_episodes = load_episodes(
        dataset_root / "val_mini" / "val_mini.json.gz"
    )
    train_files = sorted((dataset_root / "train" / "content").glob("*.json.gz"))
    counts = {
        "train_scene_files": len(train_files),
        "val": len(val_episodes),
        "val_mini": len(val_mini_episodes),
    }
    if counts != EXPECTED_POINTNAV_COUNTS:
        raise RuntimeError(
            "Unexpected Gibson PointNav inventory: {}, expected {}".format(
                counts,
                EXPECTED_POINTNAV_COUNTS,
            )
        )

    referenced_scenes = {
        Path(episode["scene_id"]).name
        for episode in val_episodes + val_mini_episodes
    }
    for path in train_files:
        scene_name = path.name.removesuffix(".json.gz")
        if not SCENE_NAME_PATTERN.fullmatch(scene_name):
            raise RuntimeError(
                "Invalid Gibson PointNav content filename: {}".format(path)
            )
        referenced_scenes.add("{}.glb".format(scene_name))
    if len(referenced_scenes) != EXPECTED_POINTNAV_REQUIRED_SCENE_COUNT:
        raise RuntimeError(
            "Unexpected referenced Gibson scene count: {}, expected {}".format(
                len(referenced_scenes),
                EXPECTED_POINTNAV_REQUIRED_SCENE_COUNT,
            )
        )
    return counts, referenced_scenes, val_episodes, tree_digest, file_count


def validate_scene_assets(scene_root, referenced_scenes):
    missing = []
    for scene_file in sorted(referenced_scenes):
        scene_path = Path(scene_file)
        if (
            scene_path.name != scene_file
            or scene_path.suffix != ".glb"
            or not SCENE_NAME_PATTERN.fullmatch(scene_path.stem)
        ):
            raise RuntimeError(
                "Invalid PointNav Gibson scene reference: {!r}".format(scene_file)
            )
        glb_path = scene_root / scene_file
        navmesh_path = glb_path.with_suffix(".navmesh")
        if not glb_path.is_file() or not navmesh_path.is_file():
            missing.append(scene_file)
    if missing:
        raise FileNotFoundError(
            "PointNav references missing Gibson assets: {}".format(missing)
        )


def validate_archive(archive_path, scene_root):
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    archive_size = archive_path.stat().st_size
    if archive_size != EXPECTED_GIBSON_ARCHIVE_SIZE:
        raise RuntimeError(
            "Gibson archive size mismatch: {}, expected {}".format(
                archive_size,
                EXPECTED_GIBSON_ARCHIVE_SIZE,
            )
        )
    archive_digest = sha256(archive_path)
    if archive_digest != EXPECTED_GIBSON_ARCHIVE_SHA256:
        raise RuntimeError(
            "Gibson archive SHA256 mismatch: {}, expected {}".format(
                archive_digest,
                EXPECTED_GIBSON_ARCHIVE_SHA256,
            )
        )

    with zipfile.ZipFile(archive_path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(entries) != EXPECTED_GIBSON_ARCHIVE_ENTRY_COUNT:
            raise RuntimeError(
                "Unexpected Gibson archive entry count: {}, expected {}".format(
                    len(entries),
                    EXPECTED_GIBSON_ARCHIVE_ENTRY_COUNT,
                )
            )
        entry_by_name = {}
        glb_names = set()
        navmesh_names = set()
        for entry in entries:
            entry_path = Path(entry.filename)
            if (
                len(entry_path.parts) != 2
                or entry_path.parts[0] != "gibson"
                or entry_path.suffix not in {".glb", ".navmesh"}
                or not SCENE_NAME_PATTERN.fullmatch(entry_path.stem)
            ):
                raise RuntimeError(
                    "Unexpected Gibson archive entry: {}".format(entry.filename)
                )
            if entry.filename in entry_by_name:
                raise RuntimeError(
                    "Duplicate Gibson archive entry: {}".format(entry.filename)
                )
            entry_by_name[entry.filename] = entry
            if entry_path.suffix == ".glb":
                glb_names.add(entry_path.stem)
            else:
                navmesh_names.add(entry_path.stem)
        if (
            len(glb_names) != EXPECTED_GIBSON_SCENE_COUNT
            or glb_names != navmesh_names
        ):
            raise RuntimeError("Gibson archive GLB/navmesh inventory mismatch")

        extracted_glb_names = {path.stem for path in scene_root.glob("*.glb")}
        extracted_navmesh_names = {
            path.stem for path in scene_root.glob("*.navmesh")
        }
        if extracted_glb_names != glb_names or extracted_navmesh_names != glb_names:
            raise RuntimeError("Extracted Gibson GLB/navmesh inventory mismatch")
        for scene_name in sorted(glb_names):
            for suffix in (".glb", ".navmesh"):
                entry = entry_by_name["gibson/{}{}".format(scene_name, suffix)]
                extracted_path = scene_root / "{}{}".format(scene_name, suffix)
                if extracted_path.stat().st_size != entry.file_size:
                    raise RuntimeError(
                        "Extracted Gibson asset size mismatch: {}".format(
                            extracted_path
                        )
                    )
    return archive_digest, archive_size, entry_by_name


def validate_selected_asset_bytes(
    archive_path,
    entry_by_name,
    scene_root,
    scene_name,
):
    digests = {}
    with zipfile.ZipFile(archive_path) as archive:
        for suffix in (".glb", ".navmesh"):
            entry_name = "gibson/{}{}".format(scene_name, suffix)
            entry = entry_by_name[entry_name]
            with archive.open(entry) as stream:
                archived_digest = sha256_stream(stream)
            extracted_path = scene_root / "{}{}".format(scene_name, suffix)
            extracted_digest = sha256(extracted_path)
            if extracted_digest != archived_digest:
                raise RuntimeError(
                    "Extracted Gibson asset differs from archive: {}".format(
                        extracted_path
                    )
                )
            digests[suffix] = extracted_digest
    return digests


def validate_episode_navmesh(navmesh_path, episode):
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path)):
        raise RuntimeError("Habitat-Sim failed to load {}".format(navmesh_path))
    start = np.asarray(episode["start_position"], dtype=np.float32)
    goal = np.asarray(episode["goals"][0]["position"], dtype=np.float32)
    if not pathfinder.is_navigable(start):
        raise RuntimeError("Official episode start is not navigable")
    if not pathfinder.is_navigable(goal):
        raise RuntimeError("Official episode goal is not navigable")
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start
    shortest_path.requested_end = goal
    if not pathfinder.find_path(shortest_path):
        raise RuntimeError("Official episode start and goal are disconnected")
    measured = float(shortest_path.geodesic_distance)
    recorded = float(episode["info"]["geodesic_distance"])
    if abs(measured - recorded) > 0.02:
        raise RuntimeError(
            "Navmesh distance mismatch: measured {}, recorded {}".format(
                measured,
                recorded,
            )
        )
    return measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--scene-id", default="Swormville")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--manifest-path", required=True)
    args = parser.parse_args()

    repository_root = Path(args.repository_root).expanduser().resolve()
    dataset_root = (
        repository_root / "data" / "datasets" / "pointnav" / "gibson" / "v1"
    )
    scene_root = repository_root / "data" / "scene_datasets" / "gibson"
    archive_path = (
        repository_root
        / "data"
        / "downloads"
        / "gibson_habitat_trainval.zip"
    )
    prepared_root = (
        repository_root / "data" / "gibson-visual-runs"
    ).resolve()
    output_dataset = Path(args.output_dataset).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    if (
        output_dataset.parent != manifest_path.parent
        or output_dataset.parent.parent != prepared_root
        or output_dataset.name != "input_dataset.json.gz"
        or manifest_path.name != "input_manifest.json"
    ):
        raise RuntimeError(
            "Prepared outputs must use one run-scoped Gibson directory"
        )
    prepared_root.mkdir(parents=True, exist_ok=True)
    run_scope_dir = output_dataset.parent
    if run_scope_dir.exists():
        raise FileExistsError("Prepared Gibson run context already exists")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not scene_root.is_dir():
        raise FileNotFoundError(scene_root)

    (
        counts,
        referenced_scenes,
        val_episodes,
        pointnav_tree_digest,
        pointnav_file_count,
    ) = pointnav_inventory(dataset_root)
    archive_digest, archive_size, archive_entries = validate_archive(
        archive_path,
        scene_root,
    )
    validate_scene_assets(scene_root, referenced_scenes)
    episode, scene_episode_count = select_scene_episode(
        val_episodes,
        args.scene_id,
        args.episode_index,
    )
    scene_path = repository_root / episode["scene_id"]
    expected_scene_path = scene_root / "{}.glb".format(args.scene_id)
    if scene_path.resolve() != expected_scene_path.resolve():
        raise RuntimeError(
            "Selected episode resolves outside the expected scene: {}".format(
                scene_path
            )
        )
    navmesh_path = scene_path.with_suffix(".navmesh")
    selected_asset_digests = validate_selected_asset_bytes(
        archive_path,
        archive_entries,
        scene_root,
        args.scene_id,
    )
    measured_distance = validate_episode_navmesh(navmesh_path, episode)

    run_scope_dir.mkdir()
    write_gzip_json(output_dataset, {"episodes": [episode]})
    output_dataset_digest = sha256(output_dataset)

    manifest = {
        "format_version": 1,
        "status": "prepared",
        "source": "official_gibson_habitat_trainval",
        "archive_path": str(archive_path),
        "archive_size": archive_size,
        "archive_sha256": archive_digest,
        "archive_entry_count": len(archive_entries),
        "scene_id": args.scene_id,
        "scene_path": str(scene_path),
        "scene_size": scene_path.stat().st_size,
        "scene_sha256": selected_asset_digests[".glb"],
        "navmesh_path": str(navmesh_path),
        "navmesh_size": navmesh_path.stat().st_size,
        "navmesh_sha256": selected_asset_digests[".navmesh"],
        "pointnav_root": str(dataset_root),
        "pointnav_file_count": pointnav_file_count,
        "pointnav_tree_sha256": pointnav_tree_digest,
        "pointnav_counts": counts,
        "pointnav_required_scene_count": len(referenced_scenes),
        "available_scene_count": len(list(scene_root.glob("*.glb"))),
        "source_episode_count_for_scene": scene_episode_count,
        "selected_episode_index": args.episode_index,
        "selected_episode_id": str(episode["episode_id"]),
        "selected_episode_contract_sha256": episode_contract_sha256(episode),
        "selected_episode_geodesic_distance": measured_distance,
        "output_dataset": str(output_dataset),
        "output_dataset_sha256": output_dataset_digest,
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
