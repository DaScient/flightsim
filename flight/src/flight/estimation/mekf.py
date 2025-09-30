"""
Multiplicative extended Kalman filter (MEKF) for attitude estimation.

The `MEKF` class implements a minimal multiplicative extended Kalman
filter for estimating spacecraft attitude and gyroscope bias.  It is
configured with process and measurement noise covariance parameters and
provides `predict` and `update_vector` methods.  Only single vector
measurements (e.g. from a star tracker) are supported in the update.
"""

from __future__ import annotations

import numpy as np

from ..mathutils.quaternion import qnorm, qmul, qconj, qrot
from ..mathutils.skew import skew


class MEKF:
    """Minimal multiplicative EKF for attitude estimation.

    This filter tracks a unit quaternion `q` describing the body→inertial
    orientation and a three‑component gyro bias `bg`.  The error state
    consists of a small rotation vector `δθ` and the bias error.  The
    covariance matrix `P` is 6×6.
    """

    def __init__(
        self,
        *,
        Qg: float = 1e-7,
        Rv: float = 1e-4,
        dt: float = 1.0,
    ) -> None:
        """Initialize the filter.

        Parameters
        ----------
        Qg : float, optional
            Gyro process noise spectral density.  This value scales the
            continuous‑time process noise for both attitude and bias.
        Rv : float, optional
            Measurement noise covariance for a single reference vector.
        dt : float, optional
            Time step [s] used when propagating the covariance.
        """
        self.dt = dt
        self.bg = np.zeros(3)
        # Initial error covariance (attitude error, bias error)
        self.P = np.eye(6) * 1e-3
        # Process noise covariance.  Attitude and bias noise share the same
        # spectral density for simplicity.
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= Qg
        self.Q[3:6, 3:6] *= Qg
        # Measurement noise covariance (assuming unit vector measurement)
        self.R = np.eye(3) * Rv

    def predict(self, q: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Propagate the error covariance based on gyro measurement.

        The quaternion `q` itself is propagated outside of the filter (for
        example using `omega_to_qdot`).  The gyro measurement `gyro`
        is used to update the covariance matrix and the bias estimate.

        Parameters
        ----------
        q : (4,) array_like
            Current quaternion estimate (unused in this simple model).
        gyro : (3,) array_like
            Measured angular rate [rad/s].

        Returns
        -------
        (4,) ndarray
            Returns the input quaternion unchanged for convenience.
        """
        # Gyro residual: error between measurement and bias (not used directly)
        _ = np.asarray(gyro, dtype=float) - self.bg
        # Linearized error dynamics matrix F for small angle and bias
        F = np.zeros((6, 6))
        # Discrete covariance update (Euler discretization)
        self.P = self.P + (F @ self.P + self.P @ F.T + self.Q) * self.dt
        return q

    def update_vector(
        self,
        q: np.ndarray,
        v_b_meas: np.ndarray,
        v_i_ref: np.ndarray,
    ) -> np.ndarray:
        """Update the attitude using a single inertial reference vector.

        Parameters
        ----------
        q : (4,) array_like
            Current quaternion estimate.
        v_b_meas : (3,) array_like
            Measured direction of the reference vector in the body frame.
        v_i_ref : (3,) array_like
            True direction of the reference vector in the inertial frame.

        Returns
        -------
        (4,) ndarray
            Updated quaternion estimate.
        """
        # Normalize the reference vector to reduce sensitivity to magnitude
        v_i_ref = np.asarray(v_i_ref, dtype=float)
        v_i_ref = v_i_ref / (np.linalg.norm(v_i_ref) + 1e-18)
        q = np.asarray(q, dtype=float)
        v_b_meas = np.asarray(v_b_meas, dtype=float)
        # Predicted measurement in the body frame
        v_b_pred = qrot(q, v_i_ref)
        # Measurement residual
        y = v_b_meas - v_b_pred
        # Measurement sensitivity matrix H (3×6)
        H = np.hstack((-skew(v_b_pred), np.zeros((3, 3))))
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # Error state update
        dx = K @ y
        # Extract small angle error (first 3 elements) and bias error (last 3)
        dtheta = dx[0:3]
        dbias = dx[3:6]
        # Form an error quaternion and apply it multiplicatively
        dq = np.concatenate(([1.0], 0.5 * dtheta))
        dq = qnorm(dq)
        q_updated = qmul(q, dq)
        # Update bias estimate
        self.bg = self.bg + dbias
        # Joseph form covariance update for numerical stability
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        return q_updated


__all__ = ["MEKF"]