import gzip
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageOps

from arguments import canonical_task_config
from run_context_contract import (
    STRICT_LIVE_CANVAS_SIZE,
    STRICT_LIVE_FIGURE_DPI,
    STRICT_LIVE_FIGURE_SIZE_INCHES,
    STRICT_VISUAL_CAPTURE_STEPS,
    episode_contract_sha256,
)
from topology_contract import crossing_evidence_survives
from scripts.prepare_gibson_visual import (
    EXPECTED_GIBSON_ARCHIVE_ENTRY_COUNT,
    EXPECTED_GIBSON_ARCHIVE_SHA256,
    EXPECTED_GIBSON_ARCHIVE_SIZE,
    EXPECTED_GIBSON_SCENE_COUNT,
    EXPECTED_POINTNAV_FILE_COUNT,
    EXPECTED_POINTNAV_REQUIRED_SCENE_COUNT,
    EXPECTED_POINTNAV_TREE_SHA256,
    dataset_tree_sha256,
    select_scene_episode,
    validate_scene_assets,
    write_gzip_json,
)
from scripts.validate_run import (
    EXPECTED_GIBSON_CONTEXT,
    check_dashboard_panels,
    collect_artifact_hashes,
    detect_run_context,
    expected_physical_checkpoint_steps,
    expected_visualization_frames,
    require_fail_fast_sources,
    validate_capture_receipt,
)


class GibsonVisualPreparationTests(unittest.TestCase):
    def test_task_config_rejects_path_aliases(self):
        self.assertEqual(
            canonical_task_config("tasks/pointnav_gibson_visual.yaml"),
            "tasks/pointnav_gibson_visual.yaml",
        )
        with self.assertRaises(ValueError):
            canonical_task_config("./tasks/pointnav_gibson_visual.yaml")
        with self.assertRaises(ValueError):
            canonical_task_config(
                "tasks/../tasks/pointnav_gibson_visual.yaml"
            )

    def test_scene_selection_is_exact_and_indexed(self):
        episodes = [
            {"episode_id": 1, "scene_id": "data/scene_datasets/gibson/Foo.glb"},
            {"episode_id": 2, "scene_id": "data/scene_datasets/gibson/Foobar.glb"},
            {"episode_id": 3, "scene_id": "data/scene_datasets/gibson/Foo.glb"},
        ]

        selected, count = select_scene_episode(episodes, "Foo", 1)

        self.assertEqual(selected["episode_id"], 3)
        self.assertEqual(count, 2)

    def test_scene_selection_rejects_paths_and_bad_indices(self):
        episodes = [
            {"episode_id": 1, "scene_id": "data/scene_datasets/gibson/Foo.glb"}
        ]

        with self.assertRaises(ValueError):
            select_scene_episode(episodes, "../Foo", 0)
        with self.assertRaises(IndexError):
            select_scene_episode(episodes, "Foo", 1)

    def test_generated_dataset_is_deterministic(self):
        payload = {"episodes": [{"episode_id": 7}]}
        with tempfile.TemporaryDirectory() as temporary_dir:
            first = Path(temporary_dir) / "first.json.gz"
            second = Path(temporary_dir) / "second.json.gz"
            write_gzip_json(first, payload)
            write_gzip_json(second, payload)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            with gzip.open(first, "rt", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), payload)

    def test_dataset_tree_digest_tracks_paths_and_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "split").mkdir()
            payload = root / "split" / "episodes.json.gz"
            payload.write_bytes(b"first")

            first_digest, first_count = dataset_tree_sha256(root)
            payload.write_bytes(b"second")
            second_digest, second_count = dataset_tree_sha256(root)

            self.assertEqual(first_count, 1)
            self.assertEqual(second_count, 1)
            self.assertNotEqual(first_digest, second_digest)

    def test_scene_asset_validation_rejects_non_basename_references(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "Foo.glb").touch()
            (root / "Foo.navmesh").touch()

            validate_scene_assets(root, {"Foo.glb"})
            with self.assertRaises(RuntimeError):
                validate_scene_assets(root, {"../Foo.glb"})

    def test_strict_runtime_sources_have_no_broad_exception_recovery(self):
        repository_root = Path(__file__).resolve().parents[1]

        checked = require_fail_fast_sources(repository_root)

        self.assertEqual(
            set(checked),
            {
                "explorable_with_door_detection.py",
                "run_context_contract.py",
                "topology_contract.py",
                "topomap_construction.py",
                "frontier_detection.py",
                "door_detection.py",
                "env/habitat/__init__.py",
                "env/habitat/exploration_env.py",
                "env/habitat/hough_door_detection.py",
            },
        )

    def test_context_evidence_automatically_enables_strict_mode(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "input_manifest.json").write_text("{}")
            (run_dir / "input_dataset.json.gz").write_bytes(b"dataset")

            self.assertTrue(detect_run_context(run_dir))

    def test_partial_context_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            (run_dir / "input_manifest.json").write_text("{}")

            with self.assertRaises(RuntimeError):
                detect_run_context(run_dir)

    def test_frame_sequence_includes_cadence_aligned_final_render(self):
        self.assertEqual(
            expected_visualization_frames(5, 15),
            [
                "visualization_frames/frame_000005.png",
                "visualization_frames/frame_000010.png",
                "visualization_frames/frame_000015.png",
            ],
        )

    def test_physical_checkpoints_are_exact_and_include_aligned_terminal_step(self):
        self.assertEqual(
            expected_physical_checkpoint_steps(219, 100),
            [100, 200],
        )
        self.assertEqual(
            expected_physical_checkpoint_steps(200, 100),
            [100, 200],
        )

    def test_strict_visual_stages_are_fixed_render_milestones(self):
        self.assertEqual(
            STRICT_VISUAL_CAPTURE_STEPS,
            (("first", 5), ("later", 10), ("mid", 30)),
        )
        self.assertEqual(STRICT_LIVE_FIGURE_SIZE_INCHES, (32.0, 18.0))
        self.assertEqual(STRICT_LIVE_FIGURE_DPI, 100)
        self.assertEqual(STRICT_LIVE_CANVAS_SIZE, (3200, 1800))

    def test_artifact_hash_closure_excludes_only_its_own_report(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "result.json").write_text("result")
            (root / "frames").mkdir()
            (root / "frames" / "frame.png").write_bytes(b"frame")
            (root / "validation.json").write_text("old validation")

            hashes, sizes = collect_artifact_hashes(root)

            self.assertEqual(
                set(hashes),
                {"result.json", "frames/frame.png"},
            )
            self.assertEqual(sizes["result.json"], 6)

    def test_capture_receipt_binds_window_and_rendered_frame_pixels(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame_path = root / "frame.png"
            window_path = root / "window.png"
            receipt_path = root / "receipt.json"
            frame_path.write_bytes(b"rendered-frame")
            window_path.write_bytes(b"physical-window")
            expected = {
                "run_id": "run",
                "process_id": 123,
                "window_id": "0x123",
                "capture_label": "first",
                "step": 5,
                "render_step": 5,
                "frame_file": frame_path.name,
                "frame_sha256": hashlib.sha256(
                    frame_path.read_bytes()
                ).hexdigest(),
                "window_file": window_path.name,
            }
            receipt = {
                **expected,
                "window_sha256": hashlib.sha256(
                    window_path.read_bytes()
                ).hexdigest(),
                "window_size": [3200, 1842],
                "captured_at_unix": time.time(),
            }
            receipt_path.write_text(json.dumps(receipt))

            validated = validate_capture_receipt(
                receipt_path,
                expected=expected,
                run_dir=root,
                expected_window_size=[3200, 1842],
            )

            self.assertEqual(validated["step"], 5)
            window_path.write_bytes(b"changed-window")
            with self.assertRaises(RuntimeError):
                validate_capture_receipt(
                    receipt_path,
                    expected=expected,
                    run_dir=root,
                    expected_window_size=[3200, 1842],
                )

    def test_episode_contract_matches_loaded_object_and_detects_mutation(self):
        source = {
            "episode_id": 923,
            "scene_id": "data/scene_datasets/gibson/Swormville.glb",
            "start_position": [1.1, 0.2, -3.8],
            "start_rotation": [0, 0.75, 0, 0.65],
            "goals": [{"position": [0.7, 0.2, 0.5], "radius": None}],
            "info": {"geodesic_distance": 5.7},
            "start_room": None,
            "shortest_paths": None,
        }
        loaded = SimpleNamespace(
            **{
                **source,
                "goals": [SimpleNamespace(**source["goals"][0])],
            }
        )

        self.assertEqual(
            episode_contract_sha256(source),
            episode_contract_sha256(loaded),
        )
        mutated = {**source, "start_position": [9.0, 0.2, -3.8]}
        self.assertNotEqual(
            episode_contract_sha256(source),
            episode_contract_sha256(mutated),
        )

    def test_crossing_must_survive_in_the_final_topology(self):
        evidence = {
            "segments": [
                {
                    "confirmed": True,
                    "edge_id": 3,
                    "edge_stable_id": 103,
                    "source_node_id": 1,
                    "source_node_stable_id": 101,
                    "target_node_id": 2,
                    "target_node_stable_id": 102,
                    "target_waypoint": [40, 50],
                }
            ]
        }
        snapshot = {
            "edges": [
                {
                    "id": 3,
                    "stable_id": 103,
                    "source": 1,
                    "source_stable_id": 101,
                    "target": 2,
                    "target_stable_id": 102,
                    "way_point": [40, 50],
                }
            ]
        }

        self.assertTrue(crossing_evidence_survives(evidence, snapshot))
        snapshot["edges"][0]["target_stable_id"] = 100
        self.assertFalse(crossing_evidence_survives(evidence, snapshot))

    def test_panel_validation_rejects_title_only_changes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first_path = root / "first.png"
            terminal_path = root / "terminal.png"
            first = Image.new("RGB", (800, 600), "white")
            draw = ImageDraw.Draw(first)
            for y in range(40, 560, 40):
                draw.rectangle((20, y, 780, y + 20), fill=(30, 60, 90))
            first.save(first_path)
            terminal = first.copy()
            ImageDraw.Draw(terminal).rectangle((0, 0, 800, 30), fill="red")
            terminal.save(terminal_path)

            with self.assertRaises(RuntimeError):
                check_dashboard_panels(first_path, terminal_path)

            first.resize((1200, 900), Image.Resampling.BICUBIC).save(
                terminal_path
            )
            with self.assertRaises(RuntimeError):
                check_dashboard_panels(first_path, terminal_path)

            ImageOps.invert(first).resize((1200, 900)).save(terminal_path)
            metrics = check_dashboard_panels(first_path, terminal_path)
            self.assertEqual(set(metrics), {"rgb", "map", "voxroom"})

    def test_validator_uses_the_preparer_data_baseline(self):
        self.assertEqual(
            EXPECTED_GIBSON_CONTEXT,
            {
                "source": "official_gibson_habitat_trainval",
                "archive_sha256": EXPECTED_GIBSON_ARCHIVE_SHA256,
                "archive_size": EXPECTED_GIBSON_ARCHIVE_SIZE,
                "archive_entry_count": EXPECTED_GIBSON_ARCHIVE_ENTRY_COUNT,
                "pointnav_tree_sha256": EXPECTED_POINTNAV_TREE_SHA256,
                "pointnav_file_count": EXPECTED_POINTNAV_FILE_COUNT,
                "pointnav_required_scene_count": (
                    EXPECTED_POINTNAV_REQUIRED_SCENE_COUNT
                ),
                "available_scene_count": EXPECTED_GIBSON_SCENE_COUNT,
            },
        )


if __name__ == "__main__":
    unittest.main()
