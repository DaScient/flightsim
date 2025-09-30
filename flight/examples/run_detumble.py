"""
Example detumble simulation using a simple B‑dot controller.

This script demonstrates detumbling a spinning spacecraft using a
magnetorquer commanded by the B‑dot algorithm.  The spacecraft is
placed in a low Earth orbit with an initial random tumble.  A constant
geomagnetic field is assumed for simplicity.  The B‑dot control
produces a magnetic dipole proportional to the negative time
derivative of the measured magnetic field in the body frame.  The
magnetic dipole produces a torque ``τ = m × B`` applied as the control
input to the rigid body integrator.
"""

import numpy as np

from flight.dynamics import rigid_body_6dof_step
from flight.mathutils import qrot, qconj


def rotate_to_body(q: np.ndarray, v_i: np.ndarray) -> np.ndarray:
    """Rotate an inertial vector into the body frame."""
    return qrot(qconj(q), v_i)


def bdot_controller(B_b_hist: list[np.ndarray], dt: float, k: float = 5e-5) -> np.ndarray:
    """Compute the control torque from the B‑dot control law.

    The B‑dot law commands a dipole ``m = -k dB/dt``.  The torque
    applied to the spacecraft is ``τ = m × B``.  The finite difference
    derivative is computed from the history of measured fields.
    """
    if len(B_b_hist) < 2:
        return np.zeros(3)
    dB = (B_b_hist[-1] - B_b_hist[-2]) / dt
    m_dipole = -k * dB
    return np.cross(m_dipole, B_b_hist[-1])


def main() -> None:
    # Constant geomagnetic field in inertial frame (approx LEO)
    B_i = np.array([0.0, 0.0, 3.12e-5])  # Tesla
    # Spacecraft properties
    params = dict(
        mass=5.0,
        J=np.diag([0.05, 0.04, 0.06]),
        enable_gravity_gradient=False,
        enable_magnetic=False,  # external magnetic torque disabled; we inject via control input
        enable_aero_torque=False,
        enable_srp_torque=False,
    )
    # Initial orbit and attitude
    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, 7.5e3, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    # Start with a large tumble
    w = np.array([0.4, -0.3, 0.25])  # rad/s
    state = (r, v, q, w)
    dt = 0.1  # time step [s]
    T_total = 600.0  # simulate for 10 minutes
    steps = int(T_total / dt)
    B_b_hist: list[np.ndarray] = []
    omega_norms = []
    for _ in range(steps):
        # Measure magnetic field in body frame
        q_bi = state[2]
        # Magnetic field vector in body frame
        B_b = rotate_to_body(q_bi, B_i)
        B_b_hist.append(B_b)
        # Compute control torque
        tau_ctrl = bdot_controller(B_b_hist, dt)
        # Step integrator with control torque
        state = rigid_body_6dof_step(state, tau_ctrl, params, dt)
        # Record angular rate norm
        omega_norms.append(np.linalg.norm(state[3]))
    print(f"Initial |ω| = {np.linalg.norm(w):.3f} rad/s")
    print(f"Final   |ω| = {omega_norms[-1]:.3f} rad/s")


if __name__ == "__main__":
    main()