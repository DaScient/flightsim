import numpy as np

from flight.dynamics import rigid_body_6dof_step
from flight.mathutils import qrot


def test_energy_and_ang_momentum_stability_free_rotation():
    """Without external torques the rotational kinetic energy and |H| should be nearly conserved."""
    # Choose arbitrary initial spin and inertia
    J = np.diag([0.15, 0.13, 0.19])
    w = np.array([0.2, -0.1, 0.05])
    # Place spacecraft in orbit to exercise translation, but disable gravity‑gradient torque
    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, 7.5e3, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    state = (r, v, q, w)
    params = dict(
        mass=12.0,
        J=J,
        enable_gravity_gradient=False,
        enable_magnetic=False,
        enable_aero_torque=False,
        enable_srp_torque=False,
    )
    dt = 0.1
    # Compute initial inertial angular momentum and rotational kinetic energy
    H0_i = qrot(q, J @ w)
    E0 = 0.5 * w @ (J @ w)
    # Propagate
    for _ in range(100):
        state = rigid_body_6dof_step(state, np.zeros(3), params, dt)
    qf = state[2]
    wf = state[3]
    Hf_i = qrot(qf, J @ wf)
    Ef = 0.5 * wf @ (J @ wf)
    # Compare magnitudes
    assert np.isclose(np.linalg.norm(H0_i), np.linalg.norm(Hf_i), rtol=1e-3)
    assert np.isclose(E0, Ef, rtol=1e-3)