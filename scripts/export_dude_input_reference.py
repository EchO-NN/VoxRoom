#!/usr/bin/env python3
"""Export only historical DUDE input grids/geometry, never prediction labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for dataset in ("interioragent", "grscene"):
        manifest = json.loads((args.source / dataset / "replay/manifests/dude_incremental.json").read_text())
        rows = []
        for record in manifest["rows"]:
            uid, event = record["episode_uid"], record["coverage_event_id"]
            original = Path(record["prediction_npz"])
            with np.load(original, allow_pickle=False) as data:
                grid = data["dude_ros_occupancy_grid"]
                metadata = json.loads(str(data["baseline_metadata_json"]))
            assert metadata["segmentation_input_mode"] == "raw_vertical_free"
            target = args.output / dataset / uid / (event + ".npz")
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists(), target
            np.savez_compressed(target, dude_ros_occupancy_grid=grid,
                                baseline_metadata_json=json.dumps({"map_info": metadata["map_info"]}))
            rows.append({"episode_uid": uid, "event": event, "source": str(original),
                         "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest()})
        summary[dataset] = rows
    assert len(summary["interioragent"]) == 64
    assert len(summary["grscene"]) == 398
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Exported 462 input-only references; no predicted room labels.")


if __name__ == "__main__":
    main()
