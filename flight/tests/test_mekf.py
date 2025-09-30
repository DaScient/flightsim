import numpy as np

from flight.estimation import MEKF
from flight.mathutils import qnorm, qmul, qconj


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute angle (deg) between two quaternions up to sign."""
    # Ensure unit quaternions
    q1 = qnorm(q1)
    q2 = qnorm(q2)
    dq = qmul(q1, qconj(q2))
    angle = 2.0 * np.degrees(np.arccos(np.clip(dq[0], -1.0, 1.0)))
    # wrap to [0, 180]
    return min(angle, 360.0 - angle)


def test_mekf_reduces_attitude_error():
    # True attitude quaternion
    q_true = qnorm(np.array([0.9, 0.1, -0.2, 0.1]))
    # Initial estimate with small error
    q_est = qnorm(np.array([0.92, 0.05, -0.18, 0.08]))
    # Define inertial reference vector
    v_i = np.array([0.3, -0.4, 0.8660254])
    # Simulate measurement in body frame using true quaternion
    from flight.mathutils import qrot
    v_b_meas = qrot(q_true, v_i)
    # Instantiate filter
    mekf = MEKF(Qg=1e-6, Rv=1e-4, dt=0.1)
    # Initial error
    err0 = quaternion_distance(q_est, q_true)
    # Run update
    q_est = mekf.predict(q_est, np.zeros(3))
    q_est = mekf.update_vector(q_est, v_b_meas, v_i)
    # New error should be smaller
    err1 = quaternion_distance(q_est, q_true)
    assert err1 < err0