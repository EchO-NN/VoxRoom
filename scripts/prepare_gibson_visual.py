#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import habitat_sim
import numpy as np


EXPECTED_POINTNAV_COUNTS = {
    "train_scene_files": 72,
    "val": 994,
    "val_mini": 30,
}
SCENE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
    return counts, referenced_scenes, val_episodes


def validate_scene_assets(scene_root, referenced_scenes):
    missing = []
    for scene_file in sorted(referenced_scenes):
        glb_path = scene_root / scene_file
        navmesh_path = glb_path.with_suffix(".navmesh")
        if not glb_path.is_file() or not navmesh_path.is_file():
            missing.append(scene_file)
    if missing:
        raise FileNotFoundError(
            "PointNav references missing Gibson assets: {}".format(missing)
        )


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
    args = parser.parse_args()

    repository_root = Path(args.repository_root).expanduser().resolve()
    dataset_root = (
        repository_root / "data" / "datasets" / "pointnav" / "gibson" / "v1"
    )
    scene_root = repository_root / "data" / "scene_datasets" / "gibson"
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not scene_root.is_dir():
        raise FileNotFoundError(scene_root)

    counts, referenced_scenes, val_episodes = pointnav_inventory(dataset_root)
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
    measured_distance = validate_episode_navmesh(navmesh_path, episode)

    output_dataset = (
        repository_root
        / "data"
        / "datasets"
        / "pointnav"
        / "gibson-visual"
        / "v1"
        / "val"
        / "val.json.gz"
    )
    write_gzip_json(output_dataset, {"episodes": [episode]})

    manifest = {
        "status": "prepared",
        "source": "official_gibson_habitat_trainval",
        "scene_id": args.scene_id,
        "scene_path": str(scene_path),
        "scene_size": scene_path.stat().st_size,
        "scene_sha256": sha256(scene_path),
        "navmesh_path": str(navmesh_path),
        "navmesh_size": navmesh_path.stat().st_size,
        "navmesh_sha256": sha256(navmesh_path),
        "pointnav_counts": counts,
        "pointnav_required_scene_count": len(referenced_scenes),
        "available_scene_count": len(list(scene_root.glob("*.glb"))),
        "source_episode_count_for_scene": scene_episode_count,
        "selected_episode_index": args.episode_index,
        "selected_episode_id": str(episode["episode_id"]),
        "selected_episode_geodesic_distance": measured_distance,
        "output_dataset": str(output_dataset),
    }
    manifest_path = repository_root / "data" / "gibson_visual_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
