#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import habitat_sim
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_gzip_json(path, payload):
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as stream:
            stream.write(
                (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            )
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
    os.replace(temporary, path)


def load_source_episode(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes", [])
    if not episodes:
        raise RuntimeError("No episodes found in {}".format(path))
    episode = episodes[0]
    goal = episode.get("info", {}).get("best_viewpoint_position")
    if goal is None:
        raise RuntimeError("The source episode has no best viewpoint")
    return episode, goal


def measure_navmesh(navmesh_path, start, goal):
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path)):
        raise RuntimeError("Habitat-Sim failed to load {}".format(navmesh_path))
    start_array = np.asarray(start, dtype=np.float32)
    goal_array = np.asarray(goal, dtype=np.float32)
    if not pathfinder.is_navigable(start_array):
        raise RuntimeError("Source start position is not navigable")
    if not pathfinder.is_navigable(goal_array):
        raise RuntimeError("Source goal position is not navigable")
    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start_array
    shortest_path.requested_end = goal_array
    if not pathfinder.find_path(shortest_path):
        raise RuntimeError("Source start and goal are not connected")
    return float(shortest_path.geodesic_distance)


def build_navmesh(scene_path, navmesh_path, start, goal):
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene_path)
    simulator_config.enable_physics = False
    agent_config = habitat_sim.agent.AgentConfiguration()
    configuration = habitat_sim.Configuration(simulator_config, [agent_config])

    with habitat_sim.Simulator(configuration) as simulator:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = 1.5
        settings.agent_radius = 0.1
        settings.agent_max_climb = 0.2
        settings.agent_max_slope = 45.0
        if not simulator.recompute_navmesh(simulator.pathfinder, settings):
            raise RuntimeError("Habitat-Sim failed to build the MP3D navmesh")
        simulator.pathfinder.save_nav_mesh(str(navmesh_path))
    return measure_navmesh(navmesh_path, start, goal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--mp3d-root",
        default=os.environ.get(
            "MP3D_ROOT",
            str(Path.home() / "SG-Nav" / "MatterPort3D" / "mp3d"),
        ),
    )
    parser.add_argument(
        "--objectnav-root",
        default=os.environ.get(
            "MP3D_OBJECTNAV_ROOT",
            str(
                Path.home()
                / "SG-Nav"
                / "MatterPort3D"
                / "objectnav"
                / "mp3d"
                / "v1"
                / "val"
                / "content"
            ),
        ),
    )
    parser.add_argument("--scene-id", default="2azQ1b91cZZ")
    parser.add_argument("--force-navmesh", action="store_true")
    args = parser.parse_args()

    repository_root = Path(args.repository_root).expanduser().resolve()
    mp3d_root = Path(args.mp3d_root).expanduser().resolve()
    objectnav_root = Path(args.objectnav_root).expanduser().resolve()
    source_scene = mp3d_root / args.scene_id / "{}.glb".format(args.scene_id)
    source_episodes = objectnav_root / "{}.json.gz".format(args.scene_id)
    if not source_scene.is_file():
        raise FileNotFoundError(source_scene)
    if not source_episodes.is_file():
        raise FileNotFoundError(source_episodes)

    scene_dir = (
        repository_root
        / "data"
        / "scene_datasets"
        / "mp3d_visual"
        / args.scene_id
    )
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_link = scene_dir / "{}.glb".format(args.scene_id)
    if scene_link.exists() or scene_link.is_symlink():
        if not scene_link.is_symlink() or scene_link.resolve() != source_scene:
            raise RuntimeError("Unexpected scene link at {}".format(scene_link))
    else:
        scene_link.symlink_to(source_scene)

    source_episode, goal = load_source_episode(source_episodes)
    start = source_episode["start_position"]
    navmesh_path = scene_dir / "{}.navmesh".format(args.scene_id)
    if args.force_navmesh or not navmesh_path.is_file():
        geodesic_distance = build_navmesh(
            scene_link,
            navmesh_path,
            start,
            goal,
        )
    else:
        geodesic_distance = measure_navmesh(navmesh_path, start, goal)
    if not navmesh_path.is_file() or navmesh_path.stat().st_size == 0:
        raise RuntimeError("The generated navmesh is missing")

    scene_id = "data/scene_datasets/mp3d_visual/{0}/{0}.glb".format(
        args.scene_id
    )
    pointnav_episode = {
        "episode_id": "0",
        "scene_id": scene_id,
        "start_position": start,
        "start_rotation": source_episode["start_rotation"],
        "info": {
            "geodesic_distance": geodesic_distance,
            "source_dataset": "mp3d_objectnav_local",
            "source_episode_id": str(source_episode["episode_id"]),
        },
        "goals": [{"position": goal, "radius": None}],
        "shortest_paths": None,
        "start_room": None,
    }
    dataset_dir = (
        repository_root
        / "data"
        / "datasets"
        / "pointnav"
        / "mp3d-visual"
        / "v1"
        / "val"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "val.json.gz"
    write_gzip_json(dataset_path, {"episodes": [pointnav_episode]})

    manifest = {
        "status": "prepared",
        "scene_id": args.scene_id,
        "source_scene": str(source_scene),
        "source_scene_sha256": sha256(source_scene),
        "source_episode_file": str(source_episodes),
        "scene_link": str(scene_link),
        "navmesh": str(navmesh_path),
        "navmesh_size": navmesh_path.stat().st_size,
        "navmesh_sha256": sha256(navmesh_path),
        "dataset": str(dataset_path),
        "start_position": start,
        "goal_position": goal,
        "geodesic_distance": geodesic_distance,
    }
    manifest_path = repository_root / "data" / "mp3d_visual_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
