# External Baselines

This directory records the original implementations required for the closed-loop room-segmentation comparison. Large repositories, checkpoints, ROS build outputs, and datasets should stay outside git or be ignored by normal source control.

Expected local checkouts:

- `Active_room_segmentation/` for `FreeformRobotics/Active_room_segmentation`
- `Incremental_DuDe_ROS/` for `lfermin77/Incremental_DuDe_ROS`
- `ROSE2/` or `$ROSE2_WS/src/ROSE2` for `aislabunimi/ROSE2`
- `ipa_coverage_planning/` or `$IPA_WS/src/ipa_coverage_planning` for `ipa320/ipa_coverage_planning`

Run `scripts/comparison/00_verify_env.sh --run-root <run-root>` after cloning/building to refresh `environment.lock.json` for the comparison run. Main experiment outputs must record original repo commits in every baseline snapshot metadata.

