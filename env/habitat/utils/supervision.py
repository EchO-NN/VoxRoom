import numpy as np


class HabitatMaps(object):
    def __init__(
        self,
        env,
        N=int(1e6),
        resolution=5,
        padding=0,
        floor_slice_eps_m=0.5,
    ):
        del N
        self.resolution = float(resolution)
        self.padding = float(padding)
        self.floor_slice_eps_m = float(floor_slice_eps_m)
        if self.resolution <= 0.0:
            raise ValueError("Habitat map resolution must be positive")
        if self.padding != 0.0:
            raise ValueError("PathFinder floor maps do not support padding")
        if self.floor_slice_eps_m <= 0.0:
            raise ValueError("floor slice epsilon must be positive")

        pathfinder = env.sim.pathfinder
        if not pathfinder.is_loaded:
            raise RuntimeError("Habitat PathFinder navmesh is not loaded")
        agent_position = np.asarray(
            env.sim.get_agent_state().position,
            dtype=np.float32,
        ).reshape(3)
        self.floor_height_m = float(agent_position[1])
        self.floor_island_index = int(pathfinder.get_island(agent_position))
        if self.floor_island_index < 0:
            raise RuntimeError("episode start does not belong to a navmesh island")

        island_view = np.asarray(
            pathfinder.get_topdown_island_view(
                self.resolution / 100.0,
                self.floor_height_m,
                self.floor_slice_eps_m,
            ),
            dtype=np.int32,
        )
        if island_view.ndim != 2 or island_view.size == 0:
            raise RuntimeError("PathFinder returned an invalid floor map")
        floor_map = island_view == self.floor_island_index
        if not np.any(floor_map):
            raise RuntimeError("episode start floor has no navigable cells")

        bounds_min, bounds_max = pathfinder.get_bounds()
        bounds_min = np.asarray(bounds_min, dtype=np.float64).reshape(3)
        bounds_max = np.asarray(bounds_max, dtype=np.float64).reshape(3)
        self.origin = np.asarray(
            [bounds_min[2], bounds_min[0]], dtype=np.float64
        ) * 100.0
        self.max = np.asarray(
            [bounds_max[2], bounds_max[0]], dtype=np.float64
        ) * 100.0
        self.size = np.asarray(island_view.shape, dtype=np.int64)
        self.floor_cell_count = int(np.count_nonzero(floor_map))
        # Habitat's legacy map uses x as rows and z as columns.
        self._floor_map = floor_map.T.astype(np.float32, copy=False)

    def get_map(self, y, lb, ub):
        requested_height_m = float(y) / 100.0
        lower_eps_m = -float(lb) / 100.0
        upper_eps_m = float(ub) / 100.0
        if not np.isclose(
            requested_height_m,
            self.floor_height_m,
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise ValueError("floor map height differs from the episode start height")
        if not (
            np.isclose(lower_eps_m, self.floor_slice_eps_m, atol=1.0e-9, rtol=0.0)
            and np.isclose(upper_eps_m, self.floor_slice_eps_m, atol=1.0e-9, rtol=0.0)
        ):
            raise ValueError("floor map bounds differ from the configured slice epsilon")
        return self._floor_map.copy()
