import numpy as np
import pytest

from env.habitat.utils.supervision import HabitatMaps


class _State(object):
    position = np.asarray([1.0, 1.25, -2.0], dtype=np.float32)


class _PathFinder(object):
    is_loaded = True

    def get_island(self, position):
        assert np.allclose(position, _State.position)
        return 0

    def get_topdown_island_view(self, resolution_m, height_m, eps_m):
        assert resolution_m == pytest.approx(0.05)
        assert height_m == pytest.approx(1.25)
        assert eps_m == pytest.approx(0.5)
        return np.asarray(
            [
                [-1, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.int32,
        )

    def get_bounds(self):
        return (
            np.asarray([-2.0, -1.0, -4.0], dtype=np.float32),
            np.asarray([3.0, 2.0, 6.0], dtype=np.float32),
        )


class _Simulator(object):
    pathfinder = _PathFinder()

    def get_agent_state(self):
        return _State()


class _Environment(object):
    sim = _Simulator()


def test_floor_reference_uses_start_island_and_habitat_axis_order():
    floor_map = HabitatMaps(_Environment(), resolution=5)

    result = floor_map.get_map(125.0, -50.0, 50.0)

    assert result.shape == (3, 2)
    assert np.array_equal(
        result,
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert floor_map.floor_cell_count == 3
    assert floor_map.floor_island_index == 0
    assert floor_map.origin.tolist() == pytest.approx([-400.0, -200.0])
    assert floor_map.max.tolist() == pytest.approx([600.0, 300.0])


def test_floor_reference_rejects_height_or_slice_contract_changes():
    floor_map = HabitatMaps(_Environment(), resolution=5)

    with pytest.raises(ValueError, match="episode start height"):
        floor_map.get_map(100.0, -50.0, 50.0)
    with pytest.raises(ValueError, match="slice epsilon"):
        floor_map.get_map(125.0, -20.0, 20.0)
