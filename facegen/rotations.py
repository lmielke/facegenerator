import math
import numpy as np

def rotate3d(vtx: np.ndarray, cumRads: tuple[float, float, float]) -> np.ndarray:
    """
    Rotate a single vertex array.

    Args:
        vtx:      (N,2) or (N,3) array of vertices
        cumRads:  (yaw, pitch, roll) in radians

    Returns:
        rotated:  (N,3) array of rotated vertices
    """
    yaw, pitch, roll = cumRads

    cy, sy = math.cos(yaw),   math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll),  math.sin(roll)

    # Rz (yaw), Ry (pitch), Rx (roll)
    Rz = np.array([[cz, -sz, 0.],
                   [sz,  cz, 0.],
                   [0.,  0., 1.]], dtype=np.float32)

    Ry = np.array([[ cy, 0., sy],
                   [ 0., 1., 0.],
                   [-sy, 0., cy]], dtype=np.float32)

    Rx = np.array([[1., 0.,  0.],
                   [0., cx, -sx],
                   [0., sx,  cx]], dtype=np.float32)

    R = (Rz @ (Ry @ Rx)).astype(np.float32)

    # Promote to 3D if necessary
    if vtx.shape[1] == 2:
        vtx3 = np.empty((vtx.shape[0], 3), dtype=np.float32)
        vtx3[:, 0:2] = vtx
        vtx3[:, 2] = 0.0
    else:
        vtx3 = vtx.astype(np.float32, copy=False)

    return vtx3 @ R
