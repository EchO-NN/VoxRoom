import numpy as np
import pytest

from voxroom_online.isaac_runtime.baselines.offline.dude_image_io import (
    labels_from_tagged_image, tagged_image_to_source_labels,
)
from voxroom_online.isaac_runtime.baselines.offline.dude_runner import (
    DudeIncrementalRunner, build_dude_rosrun_shell,
    labels_from_tagged_image as runner_decode,
)
from voxroom_online.isaac_runtime.baselines.offline.ros_entrypoints.dude_ros_node import (
    _labels_from_tagged_image as ros_decode,
)
from voxroom_online.isaac_runtime.baselines.ros_subprocess import RosSubprocessConfig


def test_native_flip_is_undone_before_clipping_without_mutation():
    free = np.zeros((80, 100), dtype=bool)
    free[8:25, 12:48] = True
    arrays = {"occupancy_map": np.zeros_like(free), "observed_free_mask": free}
    native = np.flipud(free.astype(np.float32) * 7).copy()
    original = native.copy()
    for decoder in (labels_from_tagged_image, runner_decode, ros_decode):
        actual = decoder(native, arrays)
        np.testing.assert_array_equal(actual, free.astype(np.int32))
    np.testing.assert_array_equal(native, original)
    assert not np.any((native > 0) & free), "old clip-first implementation loses all 612 cells"


def test_transform_preserves_ids_and_only_contract_relabels():
    source = np.zeros((15, 21), dtype=np.float32)
    source[2:4, 3:7] = 7
    source[6:9, 12:19] = 12
    np.testing.assert_array_equal(tagged_image_to_source_labels(np.flipud(source)), source)


@pytest.mark.parametrize("channels", [3, 4])
def test_color_images_use_same_orientation_and_background(channels):
    source = np.zeros((12, 15, channels), dtype=np.uint8)
    source[1:4, 2:8, :3] = [34, 72, 96]
    source[8, 5, :3] = 208
    source[9, 5, :3] = 255
    labels = tagged_image_to_source_labels(np.flipud(source))
    expected = np.zeros(source.shape[:2], dtype=np.int32)
    expected[1:4, 2:8] = 1
    np.testing.assert_array_equal(labels, expected)


def test_paper_threshold_is_not_rounded():
    assert DudeIncrementalRunner(fallback_python=True).concavity_threshold_m == 2.5
    shell = build_dude_rosrun_shell(RosSubprocessConfig(), concavity_threshold_m=2.5)
    assert shell.splitlines()[-1] == "exec rosrun inc_dude inc_dude 2.5"


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_threshold_rejected(value):
    with pytest.raises(ValueError):
        build_dude_rosrun_shell(RosSubprocessConfig(), concavity_threshold_m=value)
