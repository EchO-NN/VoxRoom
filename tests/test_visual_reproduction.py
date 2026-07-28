import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageStat

from topomap_construction import Topomap_construction
from visualization import RuntimeDashboard, TopologyEventRecorder


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
                frame_every_steps=1,
                refresh_seconds=0,
            )
            occupied = np.zeros((64, 64), dtype=np.float32)
            occupied[8:10, 8:50] = 1
            explored = np.zeros((64, 64), dtype=np.float32)
            explored[10:50, 10:50] = 1
            topology = {
                "current_node_id": 0,
                "room_count": 1,
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
            self.assertTrue((root / "visualization_final.png").is_file())
            self.assertTrue((root / "visualization_manifest.json").is_file())
            with Image.open(root / "visualization_final.png") as image:
                self.assertGreater(max(ImageStat.Stat(image.convert("RGB")).stddev), 2)


if __name__ == "__main__":
    unittest.main()
