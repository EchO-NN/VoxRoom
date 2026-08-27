"""Real-sensor runtime adapters for VoxRoom.

The modules in this package deliberately do not import Isaac Sim.  They expose
VoxRoom's geometry mapper to physical RGB-D/visual-inertial sensors, beginning
with the ZED 2i.
"""

from .geometry import RigidTransform, matrix_to_planar_pose, quaternion_xyzw_to_matrix

__all__ = ["RigidTransform", "matrix_to_planar_pose", "quaternion_xyzw_to_matrix"]
