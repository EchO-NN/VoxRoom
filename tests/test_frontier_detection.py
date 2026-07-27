import unittest

import numpy as np

from frontier_detection import Frontier_detection


def reference_add(collections, data, list_name):
    if list_name == "MCL":
        if data not in collections["MCL"]:
            collections["MCL"].append(data)
        for other in ("MOL", "FOL", "FCL"):
            if data in collections[other]:
                collections[other].remove(data)
    if list_name == "MOL":
        if data not in collections["MOL"]:
            collections["MOL"].append(data)
        for other in ("MCL", "FOL", "FCL"):
            if data in collections[other]:
                collections[other].remove(data)
    if list_name == "FCL":
        if data not in collections["FCL"]:
            collections["FCL"].append(data)
        if data in collections["MOL"]:
            collections["MOL"].remove(data)
        if data in collections["FOL"]:
            collections["FOL"].remove(data)
    if list_name == "FOL":
        if data not in collections["FOL"]:
            collections["FOL"].append(data)
        for other in ("MOL", "MCL", "FCL"):
            if data in collections[other]:
                collections[other].remove(data)


class FrontierDetectionTests(unittest.TestCase):
    def test_ordered_point_state_matches_original_list_transitions(self):
        detector = Frontier_detection(map_size=32, vision_range=8)
        reference = {name: [] for name in ("MCL", "MOL", "FOL", "FCL")}
        operations = [
            ("MOL", [8, 8]),
            ("MOL", [8, 9]),
            ("FOL", [8, 8]),
            ("FCL", [8, 8]),
            ("MCL", [8, 9]),
            ("MOL", [8, 9]),
            ("MCL", [8, 8]),
            ("FCL", [8, 8]),
            ("FOL", [8, 8]),
            ("MCL", [8, 8]),
        ]

        for list_name, point in operations:
            reference_add(reference, point, list_name)
            detector.add_to_list(point, list_name)
            for name in reference:
                self.assertEqual(getattr(detector, name).copy(), reference[name])

    def test_repeated_detection_is_deterministic_and_profiled(self):
        size = 96
        occupied = np.zeros((size, size), dtype=np.uint8)
        explored = np.zeros((size, size), dtype=np.uint8)
        explored[8:-8, 8:-8] = 1
        occupied[40:56, 46:50] = 1
        detector = Frontier_detection(map_size=size, vision_range=20)
        arguments = (
            np.array([48, 32]),
            np.array([48, 32]),
            occupied,
            explored,
            np.array([0, size, 0, size]),
            [],
            None,
        )

        first = detector.frontier_detection(*arguments)
        first_profile = detector.last_profile.copy()
        second = detector.frontier_detection(*arguments)

        self.assertEqual(first[3], second[3])
        self.assertEqual(first[4], second[4])
        self.assertEqual(first[1], second[1])
        np.testing.assert_allclose(first[0], second[0])
        self.assertEqual(first_profile["closed_point_count"], len(first[3]))
        self.assertEqual(
            first_profile["frontier_cluster_count"],
            len(first[0]),
        )
        for key in (
            "preprocess_seconds",
            "wavefront_seconds",
            "cluster_seconds",
            "total_seconds",
        ):
            self.assertGreaterEqual(first_profile[key], 0.0)


if __name__ == "__main__":
    unittest.main()
