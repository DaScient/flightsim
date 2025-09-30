"""
Demonstration of the multiplicative extended Kalman filter (MEKF).

This example simulates a spacecraft rotating with a constant angular
rate while a star tracker observes a single inertial reference vector.
The gyroscope is biased and noisy, and the star tracker provides
noisy measurements of the inertial vector in the body frame.  The
MEKF estimates both the quaternion attitude and the gyro bias.  At
the end of the run the attitude error and estimated bias are
reported.

Run this script from the project root via::

    PYTHONPATH=src python examples/run_mekf_demo.py

The example does not require any optional dependencies beyond NumPy.
"""

from __future__ import annotations

import numpy as np

from flight.mathutils import qnorm, qmul, qconj, qrot, omega_to_qdot
from flight.estimation.mekf import MEKF


def integrate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate a unit quaternion using first‑order integration.

    Parameters
    ----------
    q : (4,) ndarray
        Current quaternion (body→inertial).
    omega : (3,) ndarray
        Angular rate in body frame [rad/s].
    dt : float
        Time step [s].

    Returns
    -------
    (4,) ndarray
        The updated quaternion, normalised to unit length.
    """
    return qnorm(q + omega_to_qdot(q, omega) * dt)


def simulate_measurements(
    q_true: np.ndarray,
    omega_true: np.ndarray,
    bg_true: np.ndarray,
    Rg: np.ndarray,
    Rv: np.ndarray,
    v_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic gyro and star‑tracker measurements.

    Parameters
    ----------
    q_true : (4,) ndarray
        The true quaternion (body→inertial).
    omega_true : (3,) ndarray
        The true body rates [rad/s].
    bg_true : (3,) ndarray
        The true constant gyro bias [rad/s].
    Rg : (3,3) ndarray
        Gyro noise covariance matrix [rad²/s²].
    Rv : (3,3) ndarray
        Star tracker noise covariance matrix.
    v_i : (3,) ndarray
        Reference inertial unit vector observed by the star tracker.

    Returns
    -------
    tuple
        ``(gyro_meas, v_b_meas)`` where ``gyro_meas`` is the measured
        body rate (biased and noisy) and ``v_b_meas`` is the measured
        reference vector in the body frame.
    """
    gyro_meas = omega_true + bg_true + np.random.multivariate_normal(np.zeros(3), Rg)
    v_b_true = qrot(q_true, v_i)
    v_b_meas = v_b_true + np.random.multivariate_normal(np.zeros(3), Rv)
    return gyro_meas, v_b_meas


def main() -> None:
    # Seed the RNG for repeatability
    np.random.seed(0)
    # True dynamics
    omega_true = np.array([0.015, -0.02, 0.01])  # rad/s
    q_true = np.array([1.0, 0.0, 0.0, 0.0])
    bg_true = np.array([0.002, -0.0015, 0.0010])
    # Reference inertial vector (unit length)
    v_i = np.array([0.3, -0.4, 0.8660254])
    v_i = v_i / np.linalg.norm(v_i)
    # Noise covariances
    Rg = np.diag([2e-5, 2e-5, 2e-5])  # gyro noise variance
    Rv = np.diag([1e-4, 1e-4, 1e-4])  # star tracker noise variance
    # Filter initialisation
    dt = 0.1
    mekf = MEKF(Qg=1e-7, Rv=1e-4, dt=dt)
    q_hat = np.array([1.0, 0.0, 0.0, 0.0])
    # Simulation horizon
    T = 60.0  # seconds
    steps = int(T / dt)
    errors_deg: list[float] = []
    for _ in range(steps):
        # Propagate the true quaternion
        q_true = integrate_quaternion(q_true, omega_true, dt)
        # Simulate sensor measurements
        gyro_meas, v_b_meas = simulate_measurements(q_true, omega_true, bg_true, Rg, Rv, v_i)
        # Propagate quaternion estimate using measured gyro minus current bias estimate
        q_hat = integrate_quaternion(q_hat, gyro_meas - mekf.bg, dt)
        # Propagate covariance (prediction step)
        q_hat = mekf.predict(q_hat, gyro_meas)
        # Measurement update with the star tracker
        q_hat = mekf.update_vector(q_hat, v_b_meas, v_i)
        # Compute attitude error (magnitude of rotation between estimate and truth)
        q_err = qmul(q_hat, qconj(q_true))
        cos_half_theta = np.clip(q_err[0], -1.0, 1.0)
        angle = 2.0 * np.degrees(np.arccos(cos_half_theta))
        errors_deg.append(angle)
    print(f"Final attitude estimation error: {errors_deg[-1]:.4f} deg")
    print(f"Estimated gyro bias: {mekf.bg}")


if __name__ == "__main__":
    main()