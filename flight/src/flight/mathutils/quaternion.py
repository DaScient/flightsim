"""
Quaternion math routines used throughout the flight library.

Quaternions here are represented as four‑element arrays ``[w, x, y, z]``
with a real scalar part ``w`` followed by a three‑vector ``x, y, z``.
All functions expect input quaternions to be floating‑point arrays and
return `numpy` arrays.  The functions provided here are intentionally
simple and self‑contained.  They do not depend on any external libraries
beyond NumPy.
"""

from __future__ import annotations

import numpy as np

def qnorm(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length.

    Parameters
    ----------
    q : (4,) array_like
        Input quaternion.  A zero quaternion will lead to a divide by zero.

    Returns
    -------
    (4,) ndarray
        Normalized quaternion of unit length.
    """
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions.

    Given two quaternions ``q1`` and ``q2`` (ordered ``[w, x, y, z]``),
    returns the quaternion representing their product ``q1 ⊗ q2``.

    Parameters
    ----------
    q1, q2 : (4,) array_like
        The left and right operands respectively.

    Returns
    -------
    (4,) ndarray
        The Hamilton product ``q1 ⊗ q2``.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a quaternion.

    The conjugate of ``[w, x, y, z]`` is ``[w, −x, −y, −z]``.

    Parameters
    ----------
    q : (4,) array_like
        Input quaternion.

    Returns
    -------
    (4,) ndarray
        Conjugated quaternion.
    """
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion.

    Given a unit quaternion ``q`` that rotates vectors from the body frame
    into the inertial frame, computes the rotated vector

    .. math:: v' = q ⊗ [0, v] ⊗ conj(q),

    and returns its vector part.  The input vector ``v`` is interpreted
    as a 3‑vector.

    Parameters
    ----------
    q : (4,) array_like
        Unit quaternion representing a rotation.
    v : (3,) array_like
        3‑vector to rotate.

    Returns
    -------
    (3,) ndarray
        The rotated vector.
    """
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    v_quat = np.concatenate(([0.0], v))
    return qmul(q, qmul(v_quat, qconj(q)))[1:]


def omega_to_qdot(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Quaternion derivative given body angular velocity.

    The quaternion derivative relates the rate of change of the
    body‑to‑inertial rotation to the body angular velocity ``w``.  Specifically,

    .. math:: \dot{q} = \tfrac{1}{2} q ⊗ [0, \omega].

    Parameters
    ----------
    q : (4,) array_like
        Unit quaternion representing the body→inertial orientation.
    w : (3,) array_like
        Angular velocity in the body frame [rad/s].

    Returns
    -------
    (4,) ndarray
        Quaternion derivative.
    """
    q = np.asarray(q, dtype=float)
    w = np.asarray(w, dtype=float)
    w_quat = np.concatenate(([0.0], w))
    return 0.5 * qmul(q, w_quat)


__all__ = ["qnorm", "qmul", "qconj", "qrot", "omega_to_qdot"]