import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageStat

from frontier_detection import Frontier_detection
from topomap_construction import Topomap_construction
from visualization import RuntimeDashboard, TopologyEventRecorder
from scripts.validate_run import check_capture_marker


class VisualReproductionTests(unittest.TestCase):
    def test_topology_snapshot_and_exit_events(self):
        events = []

        def collect(event_type, payload):
            events.append((event_type, payload))

        topology = Topomap_construction(map_size=32, event_callback=collect)
        topology._add_rooms(1)
        topology.g.vs[1]["room_status"] = "unexplored"
        topology.g.vs[1]["room_entry"] = [[20, 10]]
        topology.g.vs[1]["room_exp"] = []
        topology._add_edge(0, 1)
        topology._add_edge(1, 0)
        topology.g.es[0]["way_point"] = [20, 10]
        topology.g.es[1]["way_point"] = [10, 20]
        topology.v_num = topology.g.vcount()

        goals = topology.choose_door([10, 10])
        snapshot = topology.snapshot()

        self.assertEqual(goals, [[20, 10]])
        self.assertEqual(snapshot["room_count"], 2)
        self.assertEqual(snapshot["edge_count"], 2)
        self.assertEqual(snapshot["current_node_id"], 1)
        self.assertIn(
            "topology_exit_selected",
            [event_type for event_type, _ in events],
        )
        json.dumps(snapshot)

    def test_no_exit_is_recorded_without_claiming_transition(self):
        events = []
        topology = Topomap_construction(
            map_size=32,
            event_callback=lambda event_type, payload: events.append(event_type),
        )
        topology.g.vs[0]["room_status"] = "explored"

        self.assertEqual(topology.choose_door([10, 10]), [])
        self.assertIn("topology_no_exit_available", events)
        self.assertNotIn("door_crossing_confirmed", events)

    def test_transition_requires_a_complete_source_to_target_trajectory(self):
        events = []
        topology = Topomap_construction(
            map_size=32,
            event_callback=lambda event_type, payload: events.append(
                (event_type, payload)
            ),
        )
        topology._add_rooms(1)
        topology.g.vs[1]["room_status"] = "unexplored"
        topology.g.vs[1]["room_entry"] = [[20, 10]]
        topology.g.vs[1]["room_exp"] = []
        topology._add_edge(0, 1)
        topology._add_edge(1, 0)
        topology.g.es[0]["way_point"] = [20, 10]
        topology.g.es[1]["way_point"] = [10, 10]
        topology.v_num = topology.g.vcount()

        self.assertEqual(topology.choose_door([10, 10]), [[20, 10]])
        confirmed, evidence = topology.confirm_pending_transition(
            [[[10, 10], [14, 10], [16, 10], [20, 10]]],
            reached_exit_count=1,
        )

        self.assertTrue(confirmed)
        self.assertTrue(evidence["segments"][0]["confirmed"])
        self.assertEqual(topology.current_node_id, 1)
        confirmed_events = [
            payload
            for event_type, payload in events
            if event_type == "door_crossing_confirmed"
        ]
        self.assertEqual(len(confirmed_events), 1)
        self.assertEqual(
            confirmed_events[0]["method"],
            "trajectory_geometry",
        )

    def test_transition_rejects_target_only_trace_and_restores_source_room(self):
        events = []
        topology = Topomap_construction(
            map_size=32,
            event_callback=lambda event_type, payload: events.append(
                event_type
            ),
        )
        topology._add_rooms(1)
        topology.g.vs[1]["room_status"] = "unexplored"
        topology.g.vs[1]["room_entry"] = [[20, 10]]
        topology.g.vs[1]["room_exp"] = []
        topology._add_edge(0, 1)
        topology._add_edge(1, 0)
        topology.g.es[0]["way_point"] = [20, 10]
        topology.g.es[1]["way_point"] = [10, 10]
        topology.v_num = topology.g.vcount()

        topology.choose_door([10, 10])
        confirmed, evidence = topology.confirm_pending_transition(
            [[[16, 10], [20, 10]]],
            reached_exit_count=1,
        )

        self.assertFalse(confirmed)
        self.assertFalse(evidence["segments"][0]["confirmed"])
        self.assertEqual(topology.current_node_id, 0)
        self.assertEqual(topology.g.vs[0]["room_status"], "exploring")
        self.assertEqual(topology.g.vs[1]["room_status"], "unexplored")
        self.assertIn("door_crossing_geometry_rejected", events)
        self.assertNotIn("door_crossing_confirmed", events)

    def test_transition_rejection_resolves_rooms_by_stable_id(self):
        topology = Topomap_construction(map_size=32)
        topology._add_rooms(2)
        topology.g.vs[0]["room_status"] = "explored"
        topology.g.vs[1]["room_status"] = "exploring"
        topology.g.vs[1]["room_entry"] = [[10, 10]]
        topology.g.vs[1]["room_exp"] = []
        topology.g.vs[2]["room_status"] = "unexplored"
        topology.g.vs[2]["room_entry"] = [[20, 10]]
        topology.g.vs[2]["room_exp"] = []
        topology._add_edge(1, 2)
        topology._add_edge(2, 1)
        topology.g.es[0]["way_point"] = [20, 10]
        topology.g.es[1]["way_point"] = [10, 10]
        topology.current_node_id = 1
        topology.v_num = topology.g.vcount()

        topology.choose_door([10, 10])
        topology.g.delete_vertices(0)
        topology.v_num = topology.g.vcount()
        confirmed, _ = topology.confirm_pending_transition(
            [[[16, 10], [20, 10]]],
            reached_exit_count=1,
        )

        self.assertFalse(confirmed)
        self.assertEqual(topology.current_node_id, 0)
        self.assertEqual(topology.g.vs[0]["stable_id"], 1)
        self.assertEqual(topology.g.vs[0]["room_status"], "exploring")
        self.assertEqual(topology.g.vs[1]["stable_id"], 2)
        self.assertEqual(topology.g.vs[1]["room_status"], "unexplored")

    def test_missing_stable_id_fails_instead_of_being_backfilled(self):
        topology = Topomap_construction(map_size=32)
        topology.g.add_vertex()

        with self.assertRaises(RuntimeError):
            topology.snapshot()

    def test_controlled_room_creation_initializes_topology_contract(self):
        topology = Topomap_construction(map_size=32)

        topology._add_rooms(1)

        self.assertEqual(topology.g.vs[1]["room_status"], "unexplored")
        self.assertEqual(topology.g.vs[1]["room_entry"], [])
        self.assertEqual(topology.g.vs[1]["room_exp"], [])

    def test_room_label_map_uses_original_room_exp_pixels(self):
        topology = Topomap_construction(map_size=32)
        topology._add_rooms(1)
        topology.g.vs[0]["room_exp"] = [[2, 3], [2, 4], [3, 4]]
        topology.g.vs[1]["room_exp"] = [[3, 4], [8, 9]]

        labels = topology.room_label_map((12, 14))

        self.assertEqual(labels.dtype, np.uint16)
        self.assertEqual(labels[2, 3], 1)
        self.assertEqual(labels[3, 4], 2)
        self.assertEqual(labels[8, 9], 2)
        self.assertEqual(labels[0, 0], 0)

    def test_dashboard_room_labels_use_same_transpose_as_active_maps(self):
        explored_source = np.zeros((12, 12), dtype=np.float32)
        explored_source[1:4, 7:11] = 1.0
        labels_source = np.zeros_like(explored_source, dtype=np.uint16)
        labels_source[1:4, 7:11] = 1
        occupied_display = np.zeros_like(explored_source)
        explored_display = explored_source.transpose()

        with self.assertRaisesRegex(RuntimeError, "grossly misaligned"):
            RuntimeDashboard._validate_room_alignment(
                occupied_display,
                explored_display,
                labels_source,
            )

        RuntimeDashboard._validate_room_alignment(
            occupied_display,
            explored_display,
            labels_source.transpose(),
        )

    def test_original_door_barrier_stays_on_the_detected_segment(self):
        detector = Frontier_detection(map_size=24)
        detector.exp_map = np.zeros((24, 24), dtype=np.uint8)

        door_cells = detector.close_door([10, 6], [10, 8])

        self.assertIn([6, 10], door_cells)
        self.assertIn([8, 10], door_cells)
        self.assertNotIn([0, 10], door_cells)
        self.assertNotIn([23, 10], door_cells)

    def test_room_label_map_does_not_complete_unassigned_free_space(self):
        topology = Topomap_construction(map_size=20)
        topology._add_rooms(1)
        topology.g.vs[0]["room_exp"] = [[5, 3], [5, 4]]
        topology.g.vs[1]["room_exp"] = [[5, 15], [5, 16]]

        labels = topology.room_label_map((12, 20))

        self.assertEqual(labels[5, 3], 1)
        self.assertEqual(labels[5, 16], 2)
        self.assertEqual(labels[5, 9], 0)
        self.assertEqual(np.count_nonzero(labels), 4)

    def test_dashboard_clips_historical_room_labels_on_current_occupied(self):
        occupied = np.zeros((8, 9), dtype=np.float32)
        occupied[2, 6] = 1.0
        explored = np.zeros((8, 9), dtype=np.float32)
        explored[2, 5:7] = 1.0
        labels = np.zeros((8, 9), dtype=np.uint16)
        labels[2, 5:8] = 3

        clipped = RuntimeDashboard.navigation_clipped_room_labels(
            occupied,
            explored,
            labels,
        )

        self.assertEqual(clipped[2, 5], 3)
        self.assertEqual(clipped[2, 6], 0)
        self.assertEqual(clipped[2, 7], 0)
        self.assertEqual(labels[2, 6], 3)

    def test_dashboard_writes_frames_final_image_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            recorder = TopologyEventRecorder(
                root / "topology_events.jsonl",
                "test-run",
                123,
            )
            recorder.record("topology_initialized", 0)
            figure = plt.figure(figsize=(12, 7))
            dashboard = RuntimeDashboard(
                figure,
                root,
                capture_identity="test-run",
                frame_every_steps=1,
                refresh_seconds=0,
            )
            occupied = np.zeros((64, 64), dtype=np.float32)
            occupied[8:10, 8:50] = 1
            explored = np.zeros((64, 64), dtype=np.float32)
            explored[10:50, 10:50] = 1
            room_labels = np.zeros((64, 64), dtype=np.uint16)
            room_labels[10:30, 10:50] = 1
            room_labels[30:50, 10:50] = 2
            topology = {
                "current_node_id": 0,
                "room_count": 2,
                "edge_count": 0,
                "nodes": [
                    {
                        "id": 0,
                        "room_status": "exploring",
                        "room_entries": [],
                    }
                ],
                "edges": [],
            }
            for step in (1, 2):
                dashboard.render(
                    step=step,
                    rgb=np.full((64, 64, 3), 100 + step, dtype=np.uint8),
                    occupied=occupied,
                    explored=explored,
                    room_labels=room_labels,
                    agent_xy=(20 + step, 20),
                    heading_degrees=30,
                    goal_xy=(40, 40),
                    doors=[],
                    raw_doors=[],
                    frontiers=[[30, 30]],
                    trajectory_xy=[(20, 20), (20 + step, 20)],
                    topology_snapshot=topology,
                    coverage_history=[0.1, 0.2],
                    explored_ratio=0.2,
                    explored_area=12.0,
                    status={
                        "phase": "room_search_navigation",
                        "action": "forward",
                        "scene_name": "test-scene",
                        "door_count": 0,
                        "raw_door_count": 0,
                        "frontier_count": 1,
                    },
                    events=recorder.latest_events(),
                )
            manifest = dashboard.finalize(topology, recorder.summary())
            plt.close(figure)

            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["frame_size"], [1600, 900])
            self.assertEqual(manifest["final_image_size"], [1920, 1080])
            self.assertTrue((root / "visualization_final.png").is_file())
            self.assertTrue((root / "visualization_manifest.json").is_file())
            self.assertTrue((root / "room_mask_final.png").is_file())
            self.assertTrue((root / "room_labels_final.npz").is_file())
            self.assertEqual(manifest["room_label_ids"], [1, 2])
            self.assertEqual(
                manifest["room_pixel_counts"],
                {"1": 800, "2": 800},
            )
            with np.load(root / "room_labels_final.npz") as room_data:
                np.testing.assert_array_equal(
                    room_data["room_labels"],
                    room_labels,
                )
                np.testing.assert_array_equal(
                    room_data["occupied"],
                    occupied,
                )
                np.testing.assert_array_equal(
                    room_data["explored"],
                    explored,
                )
            with Image.open(root / "room_mask_final.png") as room_image:
                colors = np.unique(
                    np.asarray(room_image.convert("RGB")).reshape(-1, 3),
                    axis=0,
                )
                self.assertGreaterEqual(len(colors), 4)
            for step in (1, 2):
                frame_path = (
                    root
                    / "visualization_frames"
                    / "frame_{:06d}.png".format(step)
                )
                with Image.open(frame_path) as image:
                    self.assertEqual(image.size, (1600, 900))
                check_capture_marker(frame_path, "test-run", step)
            with Image.open(root / "visualization_final.png") as image:
                self.assertEqual(image.size, (1920, 1080))
                self.assertGreater(max(ImageStat.Stat(image.convert("RGB")).stddev), 2)
            check_capture_marker(
                root / "visualization_final.png",
                "test-run",
                2,
            )
            with self.assertRaises(RuntimeError):
                check_capture_marker(
                    root / "visualization_frames" / "frame_000001.png",
                    "test-run",
                    2,
                )

    def test_dashboard_rejects_transposed_room_labels(self):
        occupied = np.zeros((20, 20), dtype=np.float32)
        explored = np.zeros((20, 20), dtype=np.float32)
        explored[2:8, 11:18] = 1
        room_labels = np.zeros((20, 20), dtype=np.uint16)
        room_labels[11:18, 2:8] = 1

        with self.assertRaisesRegex(RuntimeError, "misaligned"):
            RuntimeDashboard._validate_room_alignment(
                occupied,
                explored,
                room_labels,
            )

    def test_live_alignment_allows_small_transient_occupancy_overlap(self):
        occupied = np.zeros((20, 20), dtype=np.float32)
        explored = np.ones((20, 20), dtype=np.float32)
        room_labels = np.ones((20, 20), dtype=np.uint16)
        occupied[0, :5] = 1

        RuntimeDashboard._validate_room_alignment(
            occupied,
            explored,
            room_labels,
        )

        occupied[0:2, :] = 1
        with self.assertRaisesRegex(RuntimeError, "too many occupied"):
            RuntimeDashboard._validate_room_alignment(
                occupied,
                explored,
                room_labels,
            )

    def test_fixed_render_is_independent_of_live_figure_reflow(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            figure = plt.figure(figsize=(8, 6))
            dashboard = RuntimeDashboard(
                figure,
                root,
                capture_identity="test-run",
                frame_every_steps=0,
                refresh_seconds=0,
            )
            figure.suptitle("fixed evidence")
            dashboard.rgb_axis.imshow(
                np.arange(64, dtype=np.float32).reshape(8, 8)
            )
            first_path = root / "first.png"
            second_path = root / "second.png"
            dashboard._save_fixed_render(first_path, dashboard.FRAME_DPI)
            figure.set_size_inches(12, 6, forward=False)
            figure.canvas.draw()
            dashboard._save_fixed_render(second_path, dashboard.FRAME_DPI)
            plt.close(figure)

            self.assertEqual(
                hashlib.sha256(first_path.read_bytes()).hexdigest(),
                hashlib.sha256(second_path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
