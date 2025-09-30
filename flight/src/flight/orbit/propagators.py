"""
Simple orbital propagation helpers.

These functions compute gravitational accelerations for Earth‑orbiting
spacecraft.  They mirror the private helpers used in
``flight.dynamics.rigid_body`` and are provided here for unit
tests and basic orbit simulations.
"""

from __future__ import annotations

import numpy as np

from ..dynamics.rigid_body import MU_EARTH, R_EARTH, J2_EARTH


def two_body_accel(r: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Compute the two‑body gravitational acceleration.

    Parameters
    ----------
    r : (3,) array_like
        Position vector in an inertial frame [m].
    mu : float, optional
        Gravitational parameter [m³/s²].  Defaults to Earth's μ.

    Returns
    -------
    (3,) ndarray
        The acceleration [m/s²].
    """
    rr = np.linalg.norm(r)
    return -mu * r / (rr ** 3 + 1e-18)


def j2_accel(r: np.ndarray, mu: float = MU_EARTH, R: float = R_EARTH, J2c: float = J2_EARTH) -> np.ndarray:
    """Compute the J2 perturbation acceleration.

    Parameters
    ----------
    r : (3,) array_like
        Position vector in the inertial frame [m].
    mu : float, optional
        Gravitational parameter [m³/s²].  Default is Earth's μ.
    R : float, optional
        Equatorial radius of the central body [m].  Default is Earth's radius.
    J2c : float, optional
        J2 coefficient.  Default is Earth's value.

    Returns
    -------
    (3,) ndarray
        J2 perturbation acceleration [m/s²].
    """
    x, y, z = r
    rr = np.linalg.norm(r) + 1e-18
    zx = z / rr
    factor = 1.5 * J2c * mu * (R ** 2) / (rr ** 5)
    ax = factor * x * (5 * zx ** 2 - 1)
    ay = factor * y * (5 * zx ** 2 - 1)
    az = factor * z * (5 * zx ** 2 - 3)
    return np.array([ax, ay, az])


__all__ = ["two_body_accel", "j2_accel"]