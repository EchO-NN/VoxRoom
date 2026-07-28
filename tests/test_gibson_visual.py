import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_gibson_visual import (
    select_scene_episode,
    write_gzip_json,
)


class GibsonVisualPreparationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
