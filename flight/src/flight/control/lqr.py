"""
Linear quadratic regulator (LQR) helper routines.

This module provides a discrete‑time LQR solver for time‑invariant
systems.  It solves the discrete algebraic Riccati equation (DARE)
using an iterative approach and returns the optimal feedback gain.

Example
-------

>>> import numpy as np
>>> from flight.control.lqr import dlqr
>>> A = np.array([[1.0, 0.1], [0.0, 1.0]])
>>> B = np.array([[0.0], [0.1]])
>>> Q = np.diag([1.0, 1.0])
>>> R = np.array([[0.01]])
>>> K, P = dlqr(A, B, Q, R)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def dlqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the discrete‑time LQR problem.

    The function solves the discrete algebraic Riccati equation (DARE)

    .. math:: P = A^T P A - A^T P B (B^T P B + R)^{-1} B^T P A + Q

    and computes the state feedback gain

    .. math:: K = (B^T P B + R)^{-1} B^T P A.

    Parameters
    ----------
    A : (n,n) ndarray
        State matrix of the discrete‑time system.
    B : (n,m) ndarray
        Input matrix of the discrete‑time system.
    Q : (n,n) ndarray
        State cost matrix (symmetric positive semi‑definite).
    R : (m,m) ndarray
        Control cost matrix (symmetric positive definite).
    tol : float, optional
        Convergence tolerance for the Riccati iteration.  Defaults to 1e‑9.
    max_iter : int, optional
        Maximum number of iterations.  Defaults to 1000.

    Returns
    -------
    (K, P)
        ``K`` is the optimal feedback gain matrix; ``P`` is the
        stabilizing solution to the DARE.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    P = Q.copy()
    for _ in range(max_iter):
        # Compute the gain at this iteration
        BT_P = B.T @ P
        S = BT_P @ B + R
        K = np.linalg.solve(S, BT_P @ A)
        # Riccati update
        P_next = A.T @ P @ (A - B @ K) + Q
        # Check convergence
        if np.linalg.norm(P_next - P, ord='fro') < tol:
            P = P_next
            break
        P = P_next
    # Compute final gain
    BT_P = B.T @ P
    S = BT_P @ B + R
    K = np.linalg.solve(S, BT_P @ A)
    return K, P


__all__ = ["dlqr"]