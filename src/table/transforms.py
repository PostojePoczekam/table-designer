from __future__ import annotations

import math

import numpy as np


def rotation_matrix_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def rotation_matrix_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, 3] = [x, y, z]
    return t
