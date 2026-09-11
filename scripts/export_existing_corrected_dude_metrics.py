#!/usr/bin/env python3
"""Score existing corrected DUDE predictions only; never launch a predictor."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def write(path, rows, fieldnames=None):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--allowed-missing-json", type=Path)
    args = parser.parse_args()
    omissions = json.loads(args.allowed_missing_json.read_text())['excluded_predictions'] if args.allowed_missing_json else []
    def prediction_key(r):
        return r['dataset'], r['method'], r['episode_uid'], r['coverage_event_id']
    omitted_keys = {prediction_key(r) for r in omissions}
    assert len(omitted_keys) == len(omissions)
    if args.output.exists():
        raise FileExistsError("Choose a fresh export directory; predictions are never changed")
    args.output.mkdir(parents=True)
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.snapshot_io import load_snapshot_arrays
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.step_backprojection import backproject_final_gt_to_snapshot
    from voxroom_online.isaac_runtime.evaluation.online_roomseg.metrics import prepare_metric_label_maps, compute_snapshot_metrics
    from voxroom_online.isaac_runtime.baselines.mask_io import enforce_room_mask_contract
    from voxroom_online.isaac_runtime.baselines.offline.dude_image_io import DUDE_OUTPUT_COORDINATE_CONTRACT
    data = Path("/media/echo/data/voxroom_roomseg_evaluation")
    roots = {"interioragent": data / "interioragent_all_available_gt_20260828", "grscene": data / "grscene_all_available_gt_20260817"}
    rows, inventory, counts = [], [], {}
    for dataset, data_root in roots.items():
        audit = json.loads((args.root / dataset / "input_audit.json").read_text())
        for relative, digest in audit["code_sha256"].items():
            if relative.startswith("voxroom_online/"):
                assert hashlib.sha256((args.root / "code" / relative).read_bytes()).hexdigest() == digest, relative
        index = json.loads((args.root / dataset / "inputs/index_voxroom.json").read_text())
        expected_count = sum(len(e["snapshots"]) for e in index["episodes"])
        assert expected_count == audit["checkpoint_count"]
        original_metrics = args.root / dataset / "metrics/per_checkpoint_metrics_excluding_training.csv"
        cached = {}
        if original_metrics.exists():
            with original_metrics.open(encoding="utf-8-sig") as stream:
                cached = {(r["method"], r["episode_uid"], r["coverage_event_id"]): r for r in csv.DictReader(stream)}
        for episode in index["episodes"]:
            uid, scene = episode["episode_uid"], episode["scene_id"]
            assert scene not in audit["excluded_scene_ids"]
            gt = np.load(data_root / "final_gt" / uid / "last_step.gt_labels.npy")
            for record in episode["snapshots"]:
                event = record["coverage_event_id"]
                source = load_snapshot_arrays(Path(record["snapshot_path"]))
                with np.load(record["snapshot_path"], allow_pickle=False) as raw:
                    contract_arrays = {k: raw[k] for k in ("occupancy_map", "observed_free_mask", "obstacle_mask", "unknown_mask",
                        "roomseg_eval_reference_explorable_mask", "roomseg_eval_explored_reference_mask", "navigation_free_room_domain") if k in raw.files}
                projected_gt = backproject_final_gt_to_snapshot(gt, source, episode_uid=uid,
                                                               source_final_step=int(episode["last_snapshot_step"])).label_map
                input_grid = None
                for method in ("dude_incremental", "dude_offline"):
                    prediction_path = args.root / dataset / "replay/predictions" / method / uid / (event + ".npz")
                    common = {"dataset": dataset, "method": method, "scene_id": scene, "episode_uid": uid,
                              "coverage_event_id": event, "step": int(record["step"]),
                              "source_snapshot_path": record["snapshot_path"], "prediction_path": str(prediction_path)}
                    inventory.append({**common, "prediction_available": prediction_path.exists(),
                                      "authorized_exclusion": prediction_key(common) in omitted_keys,
                                      "missing_reason": "user_excluded_native_assertion_dude_cut_cpp_399" if prediction_key(common) in omitted_keys else ("" if prediction_path.exists() else "prediction_not_produced_before_original_replay_timeout")})
                    if prediction_key(common) in omitted_keys:
                        assert not prediction_path.exists(), 'Never discard an existing prediction using an omission manifest'
                    if not prediction_path.exists():
                        continue
                    with np.load(prediction_path, allow_pickle=False) as pred:
                        metadata = json.loads(str(pred["baseline_metadata_json"]))
                        assert metadata["method"] == method
                        assert metadata["parameters"]["concavity_threshold_m"] == 2.5
                        assert metadata["parameters"]["use_incremental"] == (method == "dude_incremental")
                        assert metadata["segmentation_input_mode"] == "raw_vertical_free"
                        assert metadata["runner_type"] == "original_ros"
                        assert metadata["coverage_reference_used_as_segmentation_input"] is False
                        assert metadata["output_coordinate_contract"] == DUDE_OUTPUT_COORDINATE_CONTRACT
                        assert Path(metadata["source_snapshot"]).resolve() == Path(record["snapshot_path"]).resolve()
                        assert int(pred["step"]) == int(record["step"])
                        source_frame = pred["dude_labels_source_frame"]
                        np.testing.assert_array_equal(np.flipud(pred["dude_tagged_image_native"]).astype(np.int32), source_frame)
                        labels = pred["final_room_label_map"]
                        np.testing.assert_array_equal(labels, enforce_room_mask_contract(source_frame, contract_arrays, clip_to_eval_domain=True))
                        grid = pred["dude_ros_occupancy_grid"]
                        if input_grid is None:
                            input_grid = grid
                        else:
                            np.testing.assert_array_equal(input_grid, grid)
                        grid_sha = hashlib.sha256(grid.tobytes()).hexdigest()
                    prepared = prepare_metric_label_maps(projected_gt, labels, min_room_area_cells=200)
                    scores = compute_snapshot_metrics(prepared.gt, prepared.pred)
                    p, r = float(scores["precision"]), float(scores["recall"])
                    row = {**common, "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.,
                           "miou_room": float(scores["miou_room"]), "n_gt": int(scores["n_gt"]),
                           "n_pred": int(scores["n_pred"]), "metric_domain_pixels": int(np.count_nonzero(prepared.metric_domain)),
                           "concavity_threshold_m": 2.5, "input_grid_sha256": grid_sha,
                           "output_coordinate_contract": metadata["output_coordinate_contract"]}
                    old = cached.get((method, uid, event))
                    if old is not None:
                        for metric in ("precision", "recall", "f1", "miou_room", "n_gt", "n_pred", "metric_domain_pixels"):
                            assert math.isclose(float(row[metric]), float(old[metric]), abs_tol=1e-10), (metric, row, old)
                    rows.append(row)
            print(f"Scored saved predictions: {dataset} {scene}; total {len(rows)}", flush=True)
        for method in ("dude_incremental", "dude_offline"):
            counts[dataset+"/"+method] = sum(r["dataset"] == dataset and r["method"] == method for r in rows)
    assert len(inventory) == 924 and len(rows) <= len(inventory), counts
    missing = [r for r in inventory if not r["prediction_available"]]
    assert omitted_keys <= {prediction_key(r) for r in missing}
    unexpected = [r for r in missing if prediction_key(r) not in omitted_keys]
    if args.require_complete:
        assert not unexpected, ("Unapproved missing predictions; refusing requested-scope complete export", counts)
    write(args.output / "per_checkpoint_metrics.csv", rows)
    write(args.output / "checkpoint_inventory.csv", inventory)
    write(args.output / "missing_predictions.csv", missing, fieldnames=list(inventory[0]))
    (args.output / "export_audit.json").write_text(json.dumps({"status": "export_complete", "prediction_run_complete": not missing,
        "requested_scope_complete": not unexpected, "authorized_exclusions": len(omissions),
        "unexpected_missing_predictions": len(unexpected), "eligible_predictions": len(inventory)-len(omissions),
        "available_predictions": len(rows), "expected_predictions": len(inventory), "missing_predictions": len(inventory)-len(rows),
        "counts": counts, "operation": "metrics from existing predictions only; no new segmentation or source overwrite",
        "cached_interioragent_metrics_verified": True, "coordinate_and_input_contracts_verified": True}, indent=2) + "\n")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
