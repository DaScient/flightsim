import numpy as np

from flight.mathutils import qnorm, qmul, qconj, qrot, omega_to_qdot


def test_quaternion_normalization_and_conjugate():
    # Random quaternion should be normalized and conjugated correctly
    q = np.array([0.2, -0.4, 0.1, 0.5])
    qn = qnorm(q)
    # Norm should be unity
    assert np.isclose(np.linalg.norm(qn), 1.0)
    # Conjugate should invert rotation
    v = np.array([1.0, -2.0, 0.5])
    v_rot = qrot(qn, v)
    v_back = qrot(qconj(qn), v_rot)
    assert np.allclose(v_back, v, atol=1e-12)


def test_quaternion_multiplication_associativity():
    # Quaternion multiplication is associative
    q1 = qnorm(np.array([1.0, 0.1, 0.2, -0.1]))
    q2 = qnorm(np.array([0.9, -0.3, 0.1, 0.2]))
    q3 = qnorm(np.array([1.1, 0.0, -0.5, 0.3]))
    left = qmul(qmul(q1, q2), q3)
    right = qmul(q1, qmul(q2, q3))
    assert np.allclose(left, right, atol=1e-12)


def test_omega_to_qdot_dimension():
    q = qnorm(np.array([1.0, 0.0, 0.0, 0.0]))
    w = np.array([0.1, -0.2, 0.3])
    qdot = omega_to_qdot(q, w)
    # qdot should have same shape as q
    assert qdot.shape == q.shape