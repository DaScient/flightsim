"""
Skew‑symmetric matrix helper.

A skew‑symmetric matrix is useful when expressing cross products as
matrix multiplications.  Given a 3‑vector ``w``, the skew‑symmetric
matrix ``[w]_×`` satisfies ``[w]_× v = w × v`` for any 3‑vector ``v``.
"""

import numpy as np

def skew(w: np.ndarray) -> np.ndarray:
    """Return the skew‑symmetric matrix of a 3‑vector.

    Parameters
    ----------
    w : (3,) array_like
        3‑vector.

    Returns
    -------
    (3,3) ndarray
        Skew‑symmetric matrix such that ``skew(w) @ v == np.cross(w, v)``.
    """
    x, y, z = w
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


__all__ = ["skew"]