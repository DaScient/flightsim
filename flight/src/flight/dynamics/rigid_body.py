"""
Numerical integration of six‑degree‑of‑freedom rigid‑body dynamics.

The main function `rigid_body_6dof_step` advances a spacecraft state one
time step, propagating both translational and rotational dynamics.
It supports common perturbations such as J2 gravity, atmospheric drag
(cannonball model) and solar radiation pressure, and can inject
various torques including gravity‑gradient, magnetic and offset
torques from aerodynamic or SRP forces.  Each torque source can be
enabled or disabled at runtime via the ``params`` dictionary.

States are expressed as tuples ``(r_eci, v_eci, q_bi, w_b)`` where

* ``r_eci`` is the spacecraft position in an inertial frame [m],
* ``v_eci`` is the velocity in the inertial frame [m/s],
* ``q_bi`` is a unit quaternion mapping body→inertial,
* ``w_b`` is the body angular rate expressed in the body frame [rad/s].

Torques and control inputs are specified in the body frame.  Forces
such as drag are applied in the inertial frame.  See the docstring on
`rigid_body_6dof_step` for details of accepted parameters.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple

from ..mathutils.quaternion import qnorm, qrot, qconj, omega_to_qdot

# Physical constants (SI units)
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6_378_137.0       # m
J2_EARTH = 1.08262668e-3    # dimensionless


def _two_body_accel(r: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Return the two‑body gravitational acceleration on a point mass.

    Parameters
    ----------
    r : (3,) array_like
        Position vector in the inertial frame [m].
    mu : float, optional
        Gravitational parameter of the central body [m^3/s^2].  Default
        is Earth's gravitational parameter.

    Returns
    -------
    (3,) ndarray
        The acceleration [m/s^2].
    """
    rr = np.linalg.norm(r)
    return -mu * r / (rr ** 3 + 1e-18)


def _j2_accel(r: np.ndarray, mu: float = MU_EARTH, R: float = R_EARTH, J2c: float = J2_EARTH) -> np.ndarray:
    """Compute the J2 perturbation acceleration in the inertial frame.

    Parameters
    ----------
    r : (3,) array_like
        Position vector in the inertial frame [m].
    mu : float, optional
        Gravitational parameter of the central body [m^3/s^2].
    R : float, optional
        Equatorial radius of the central body [m].
    J2c : float, optional
        J2 coefficient (dimensionless).

    Returns
    -------
    (3,) ndarray
        The J2 acceleration contribution [m/s^2].  When J2c == 0 the
        return will be a zero vector.
    """
    x, y, z = r
    rr = np.linalg.norm(r) + 1e-18
    zx = z / rr
    factor = 1.5 * J2c * mu * (R ** 2) / (rr ** 5)
    ax = factor * x * (5 * zx ** 2 - 1)
    ay = factor * y * (5 * zx ** 2 - 1)
    az = factor * z * (5 * zx ** 2 - 3)
    return np.array([ax, ay, az])


def _drag_accel(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    m: float,
    Cd: float = 2.2,
    A_ref: float = 0.0,
    rho: Optional[Callable[[np.ndarray], float]] = None,
    v_atm_eci: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simple cannonball drag acceleration model.

    Parameters
    ----------
    r_eci : (3,) array_like
        Position in ECI [m].  Only used when ``rho`` is callable.
    v_eci : (3,) array_like
        Velocity in ECI [m/s].
    m : float
        Spacecraft mass [kg].
    Cd : float, optional
        Drag coefficient (dimensionless).  Default is 2.2.
    A_ref : float, optional
        Reference area [m²].  If zero or negative, drag is disabled.
    rho : callable or float, optional
        Atmospheric density [kg/m³] at the spacecraft position.  May be
        a constant float or a function of position.  If none, no drag.
    v_atm_eci : (3,) array_like, optional
        Velocity of the atmosphere in ECI [m/s] (e.g. due to Earth rotation).
        If None, assumed to be zero.

    Returns
    -------
    (3,) ndarray
        Drag acceleration [m/s²].
    """
    if A_ref <= 0.0:
        return np.zeros(3)
    # Determine density at current position
    if callable(rho):
        rho_val = float(rho(r_eci))
    elif rho is not None:
        rho_val = float(rho)
    else:
        rho_val = 0.0
    if rho_val <= 0.0:
        return np.zeros(3)
    v_rel = v_eci - (v_atm_eci if v_atm_eci is not None else 0.0)
    vmag = np.linalg.norm(v_rel)
    if vmag < 1e-12:
        return np.zeros(3)
    force = -0.5 * rho_val * Cd * A_ref * vmag * v_rel
    return force / (m + 1e-18)


def _srp_accel(
    r_sun_eci: Optional[np.ndarray],
    m: float,
    Cr: float = 1.8,
    A_srp: float = 0.0,
    P0: float = 4.56e-6,
) -> np.ndarray:
    """Compute solar radiation pressure acceleration (cannonball model).

    Parameters
    ----------
    r_sun_eci : (3,) array_like or None
        Vector from spacecraft to Sun in ECI [m].  If None, SRP is disabled.
    m : float
        Spacecraft mass [kg].
    Cr : float, optional
        Reflection coefficient.  Default 1.8.
    A_srp : float, optional
        Effective SRP area [m²].  If zero, SRP is disabled.
    P0 : float, optional
        Solar radiation pressure at 1 AU [N/m²].  Default 4.56×10⁻⁶.

    Returns
    -------
    (3,) ndarray
        Acceleration due to SRP [m/s²].
    """
    if A_srp <= 0.0 or r_sun_eci is None:
        return np.zeros(3)
    rnorm = np.linalg.norm(r_sun_eci)
    if rnorm < 1.0:
        return np.zeros(3)
    rhat = r_sun_eci / rnorm
    force = Cr * P0 * A_srp * rhat
    return force / (m + 1e-18)


def _gravity_gradient_tau(J: np.ndarray, r_b: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Return the gravity‑gradient torque in the body frame.

    The gravity‑gradient torque is given by
    ``τ = 3 μ / r³ * (n × (J n))``, where ``n = r_b/||r_b||``.
    """
    r = np.linalg.norm(r_b)
    if r < 1e-6:
        return np.zeros(3)
    n = r_b / r
    return 3.0 * mu / (r ** 3) * np.cross(n, J @ n)


def _magnetic_tau(m_dipole_b: Optional[np.ndarray], B_i: Optional[np.ndarray], q_bi: np.ndarray) -> np.ndarray:
    """Compute magnetic control or disturbance torque.

    Given a commanded magnetic dipole ``m_dipole_b`` in the body frame and
    the local magnetic field ``B_i`` in inertial coordinates, compute the
    torque ``τ = m × B`` (in the body frame).  If either the dipole or
    the magnetic field is absent, returns zero.
    """
    if m_dipole_b is None or B_i is None:
        return np.zeros(3)
    # rotate inertial magnetic field into body coordinates
    B_b = qrot(qconj(q_bi), B_i)
    return np.cross(m_dipole_b, B_b)


def _offset_torque_from_force(F_eci: Optional[np.ndarray], r_cp_b: Optional[np.ndarray], q_bi: np.ndarray) -> np.ndarray:
    """Compute torque from an ECI force applied at a body‑frame offset.

    Given a force ``F_eci`` applied at an offset ``r_cp_b`` (from the
    spacecraft centre of mass) expressed in the body frame, returns
    ``τ = r_cp × F`` in the body frame.  If either the force or
    the offset is absent, returns zero.
    """
    if r_cp_b is None or F_eci is None:
        return np.zeros(3)
    # Convert force to body frame
    F_b = qrot(qconj(q_bi), F_eci)
    return np.cross(r_cp_b, F_b)


def rigid_body_6dof_step(
    state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    u_b: np.ndarray,
    params: dict,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance a 6‑DOF spacecraft state one time step.

    Parameters
    ----------
    state : tuple
        A tuple ``(r_eci, v_eci, q_bi, w_b)`` representing the current
        spacecraft state.
    u_b : (3,) array_like
        Control torque applied in the body frame [N m].
    params : dict
        Simulation parameters.  Keys include:

        - ``mass`` (float): spacecraft mass [kg] **(required)**.
        - ``J`` (3×3 array): inertia matrix in the body frame [kg m²] **(required)**.
        - ``mu`` (float): gravitational parameter of central body.
        - ``R_earth`` (float): equatorial radius of central body for J2.
        - ``J2`` (float): J2 coefficient.
        - Drag parameters: ``Cd``, ``A_ref``, ``rho``, ``v_atm_eci``.
        - SRP parameters: ``Cr``, ``A_srp``, ``P0``, ``r_sun_eci``, ``r_cs_b``.
        - Magnetic parameters: ``m_dipole_b``, ``B_i``.
        - Offsets: ``r_cp_b`` (drag), ``r_cs_b`` (SRP).
        - Disturbances: ``f_dist_eci`` (external force), ``tau_dist_b`` (external torque).
        - Enable flags: ``enable_gravity_gradient``, ``enable_magnetic``,
          ``enable_aero_torque``, ``enable_srp_torque`` (booleans).

        Missing keys fall back to defaults defined in this module.
    dt : float
        Integration time step [s].

    Returns
    -------
    tuple
        Next state ``(r_eci_new, v_eci_new, q_bi_new, w_b_new)``.
    """
    # unpack state
    r, v, q, w = state

    # required parameters
    m = float(params["mass"])
    J = np.asarray(params["J"], dtype=float)
    Jinv = np.linalg.inv(J)

    # gravitational constants
    mu = params.get("mu", MU_EARTH)
    R = params.get("R_earth", R_EARTH)
    J2c = params.get("J2", J2_EARTH)

    # optional drag parameters
    Cd = params.get("Cd", 2.2)
    A_ref = params.get("A_ref", 0.0)
    rho = params.get("rho", 0.0)
    v_atm_eci = params.get("v_atm_eci", None)
    r_cp_b = params.get("r_cp_b", None)

    # optional SRP parameters
    Cr = params.get("Cr", 1.8)
    A_srp = params.get("A_srp", 0.0)
    P0 = params.get("P0", 4.56e-6)
    r_sun_eci = params.get("r_sun_eci", None)
    r_cs_b = params.get("r_cs_b", None)

    # magnetic parameters
    B_i = params.get("B_i", None)
    m_dipole_b = params.get("m_dipole_b", None)

    # disturbances
    f_dist_eci = params.get("f_dist_eci", None)
    tau_dist_b = params.get("tau_dist_b", None)

    # enable/disable flags
    enable_gravity_gradient = params.get("enable_gravity_gradient", True)
    enable_magnetic = params.get("enable_magnetic", True)
    enable_aero_torque = params.get("enable_aero_torque", True)
    enable_srp_torque = params.get("enable_srp_torque", True)

    # convenience: compute translational acceleration for given state
    def translational_accel(r_: np.ndarray, v_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # base two‑body
        a = _two_body_accel(r_, mu=mu)
        # J2
        if J2c and J2c != 0.0:
            a += _j2_accel(r_, mu=mu, R=R, J2c=J2c)
        # drag
        a += _drag_accel(r_, v_, m, Cd=Cd, A_ref=A_ref, rho=rho, v_atm_eci=v_atm_eci)
        # SRP
        a_srp = _srp_accel(r_sun_eci, m, Cr=Cr, A_srp=A_srp, P0=P0)
        a += a_srp
        # external force
        if f_dist_eci is not None:
            a += np.asarray(f_dist_eci, dtype=float) / (m + 1e-18)
        return a, a_srp

    # compute total torque in body frame given instantaneous state
    def total_torque_b(r_: np.ndarray, q_: np.ndarray, w_: np.ndarray) -> np.ndarray:
        tau_b = np.array(u_b, dtype=float)
        # gravity gradient
        if enable_gravity_gradient:
            r_b = qrot(qconj(q_), r_)
            tau_b += _gravity_gradient_tau(J, r_b, mu=mu)
        # magnetic torque
        if enable_magnetic:
            tau_b += _magnetic_tau(m_dipole_b, B_i, q_)
        # aerodynamic torque from drag force at CP
        if enable_aero_torque and A_ref > 0.0 and r_cp_b is not None:
            a_drag = _drag_accel(r_, v, m, Cd=Cd, A_ref=A_ref, rho=rho, v_atm_eci=v_atm_eci)
            F_drag_eci = a_drag * m
            tau_b += _offset_torque_from_force(F_drag_eci, r_cp_b, q_)
        # SRP torque from SRP force at CP
        if enable_srp_torque and A_srp > 0.0 and r_cs_b is not None and r_sun_eci is not None:
            a_srp = _srp_accel(r_sun_eci, m, Cr=Cr, A_srp=A_srp, P0=P0)
            F_srp_eci = a_srp * m
            tau_b += _offset_torque_from_force(F_srp_eci, r_cs_b, q_)
        # external disturbance torque
        if tau_dist_b is not None:
            tau_b += np.asarray(tau_dist_b, dtype=float)
        return tau_b

    # define derivative of full state for RK4
    def f_state(y: np.ndarray) -> np.ndarray:
        r_, v_, q_, w_ = y[:3], y[3:6], y[6:10], y[10:13]
        # translational acceleration and SRP acceleration (for torque later)
        a_eci, _ = translational_accel(r_, v_)
        rdot = v_
        vdot = a_eci
        # quaternion derivative
        qdot = omega_to_qdot(q_, w_)
        # torque
        tau_b = total_torque_b(r_, q_, w_)
        wdot = Jinv @ (tau_b - np.cross(w_, J @ w_))
        return np.concatenate([rdot, vdot, qdot, wdot])

    # build combined state vector
    y0 = np.concatenate([r, v, q, w])

    # RK4 integration
    k1 = f_state(y0)
    y1 = y0 + 0.5 * dt * k1
    y1[6:10] = qnorm(y1[6:10])  # renormalize quaternion
    k2 = f_state(y1)
    y2 = y0 + 0.5 * dt * k2
    y2[6:10] = qnorm(y2[6:10])
    k3 = f_state(y2)
    y3 = y0 + dt * k3
    y3[6:10] = qnorm(y3[6:10])
    k4 = f_state(y3)
    y_new = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    y_new[6:10] = qnorm(y_new[6:10])

    r_new = y_new[:3]
    v_new = y_new[3:6]
    q_new = y_new[6:10]
    w_new = y_new[10:13]
    return (r_new, v_new, q_new, w_new)


__all__ = ["rigid_body_6dof_step"]