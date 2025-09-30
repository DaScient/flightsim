"""
Clohessy–Wiltshire (Hill's) equations for relative motion.

This module provides helper functions for computing the state
transition matrix for relative motion in a circular orbit using the
Clohessy–Wiltshire (CW) equations.  These equations describe the
relative dynamics of a chaser spacecraft with respect to a chief
spacecraft in a nearby circular orbit.

The CW equations are valid for small relative distances compared to
the orbit radius and assume the chief's orbit is circular.  They are
useful for preliminary analysis of rendezvous and formation‑flying
problems.
"""

from __future__ import annotations

import numpy as np

from ..dynamics.rigid_body import MU_EARTH


def mean_motion(a: float, mu: float = MU_EARTH) -> float:
    """Compute the mean motion (rad/s) for a circular orbit of semi‑major axis ``a``.

    Parameters
    ----------
    a : float
        Semi‑major axis [m].
    mu : float, optional
        Gravitational parameter [m³/s²].  Default is Earth's μ.

    Returns
    -------
    float
        Mean motion [rad/s].
    """
    return np.sqrt(mu / a ** 3)


def cw_state_transition(n: float, dt: float) -> np.ndarray:
    """Return the 6×6 Clohessy–Wiltshire state transition matrix.

    Parameters
    ----------
    n : float
        Mean motion [rad/s] of the reference circular orbit.
    dt : float
        Time step [s].

    Returns
    -------
    (6,6) ndarray
        State transition matrix that propagates the relative position
        and velocity `(x, y, z, ẋ, ẏ, ż)` over time `dt`.
    """
    nt = n * dt
    s = np.sin(nt)
    c = np.cos(nt)
    # Transition submatrices
    Phi_rr = np.array([
        [4 - 3 * c, 0.0, 0.0],
        [6 * (s - nt), 1.0, 0.0],
        [0.0, 0.0, c],
    ])
    Phi_rv = np.array([
        [s / n, 2 * (1 - c) / n, 0.0],
        [-2 * (1 - c) / n, (4 * s - 3 * nt) / n, 0.0],
        [0.0, 0.0, s / n],
    ])
    Phi_vr = np.array([
        [3 * n * s, 0.0, 0.0],
        [6 * n * (c - 1), 0.0, 0.0],
        [0.0, 0.0, -n * s],
    ])
    Phi_vv = np.array([
        [c, 2 * s, 0.0],
        [-2 * s, 4 * c - 3, 0.0],
        [0.0, 0.0, c],
    ])
    # Assemble full 6×6 matrix
    top = np.hstack((Phi_rr, Phi_rv))
    bottom = np.hstack((Phi_vr, Phi_vv))
    return np.vstack((top, bottom))


__all__ = ["mean_motion", "cw_state_transition"]