"""
Demonstration of the 6‑DOF rigid body integrator.

This script propagates a simple spacecraft for a few minutes in a
circular orbit while spinning with a constant angular rate.  No
external torques are applied, so the inertial angular momentum should
remain constant.  The simulation parameters can be adjusted to
exercise different perturbations.
"""

import numpy as np

from flight.dynamics import rigid_body_6dof_step
from flight.mathutils import qrot


def main() -> None:
    # Initial orbit: 700 km circular about Earth
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7.5e3, 0.0])
    # Start with identity quaternion (body aligned with inertial)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    # Impose a slight spin
    w0 = np.array([0.01, -0.02, 0.015])
    state = (r0, v0, q0, w0)
    # Spacecraft properties
    params = dict(
        mass=8.0,
        J=np.diag([0.1, 0.09, 0.11]),
        J2=1.08262668e-3,
        Cd=2.2,
        A_ref=0.02,
        rho=lambda r: 1e-12,
        v_atm_eci=np.array([0.0, 0.0, 0.0]),
        Cr=1.8,
        A_srp=0.01,
        r_sun_eci=np.array([1.5e11, 0.0, 0.0]),
        enable_gravity_gradient=True,
        enable_magnetic=False,
        enable_aero_torque=True,
        enable_srp_torque=True,
    )
    dt = 1.0  # 1‑second step
    T = 600.0  # total time [s]
    steps = int(T / dt)
    for _ in range(steps):
        # No control torque in this demo
        state = rigid_body_6dof_step(state, np.zeros(3), params, dt)
    r_f, v_f, q_f, w_f = state
    print(f"Final orbital radius: {np.linalg.norm(r_f)/1e3:.3f} km")
    print(f"Final inertial angular momentum: {qrot(q_f, params['J'] @ w_f)}")
    print(f"Final quaternion: {q_f}")


if __name__ == "__main__":
    main()