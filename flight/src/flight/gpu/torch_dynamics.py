"""
GPU‑accelerated rigid‑body dynamics using PyTorch.

This module mirrors a subset of the functionality of the CPU
implementation in ``flight.dynamics.rigid_body``, but performs the
computations using tensors on a GPU when available.  It requires
PyTorch to be installed; if PyTorch is not installed or no suitable GPU
is available, an informative ``ImportError`` or ``RuntimeError`` is
raised.

The primary entry point is :func:`rigid_body_6dof_step_torch`, which
propagates a single time step of a spacecraft state under two‑body
gravity (with optional J2 perturbation) and basic rigid‑body
kinematics.  Additional perturbations such as drag and solar
radiation pressure are omitted here for clarity; they may be added in
future revisions.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

try:
    import torch  # type: ignore
except ImportError as exc:  # pragma: no cover - torch is optional
    raise ImportError(
        "The GPU extensions require PyTorch to be installed.\n"
        "Install PyTorch with MPS or CUDA support to use this module."
    ) from exc


def device_for_acceleration() -> "torch.device":
    """Return the best available torch device for acceleration.

    This helper chooses an appropriate device in the following order:

    1. If an MPS device is available (Apple Silicon), return ``'mps'``.
    2. Else if a CUDA device is available, return the default CUDA device.
    3. Otherwise return the CPU device.

    Users may override this selection by passing their own ``torch.device``
    to :func:`rigid_body_6dof_step_torch`.

    Returns
    -------
    torch.device
        The selected PyTorch device.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _omega_to_qdot(q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute quaternion derivative given angular rate.

    Parameters
    ----------
    q : (4,) torch.Tensor
        Unit quaternion ``[q0, q1, q2, q3]`` representing body→inertial
        rotation.
    w : (3,) torch.Tensor
        Angular rate vector in the body frame [rad/s].

    Returns
    -------
    (4,) torch.Tensor
        Time derivative of ``q``.
    """
    q0 = q[0]
    qv = q[1:]
    dq0dt = -0.5 * (qv @ w)
    dqvdt = 0.5 * (q0 * w + torch.cross(qv, w))
    return torch.cat((dq0dt.unsqueeze(0), dqvdt))


def _two_body_accel(r: torch.Tensor, mu: float) -> torch.Tensor:
    """Two‑body gravitational acceleration in torch tensors."""
    rr = torch.linalg.norm(r)
    return -mu * r / (rr ** 3 + 1e-18)


def _j2_accel(r: torch.Tensor, mu: float, R: float, J2c: float) -> torch.Tensor:
    """J2 perturbation acceleration computed on GPU."""
    rr = torch.linalg.norm(r) + 1e-18
    x, y, z = r
    zx = z / rr
    factor = 1.5 * J2c * mu * (R ** 2) / (rr ** 5)
    ax = factor * x * (5.0 * zx ** 2 - 1.0)
    ay = factor * y * (5.0 * zx ** 2 - 1.0)
    az = factor * z * (5.0 * zx ** 2 - 3.0)
    return torch.stack((ax, ay, az))


def rigid_body_6dof_step_torch(
    state: Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"],
    u_b: "torch.Tensor",
    params: Dict[str, Any],
    dt: float,
    *,
    device: "torch.device" | None = None,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Advance a rigid‑body state one time step on the GPU.

    Parameters
    ----------
    state : tuple
        Current spacecraft state ``(r, v, q, w)`` where each element is
        a 1‑D tensor residing on a PyTorch device.  ``r`` and ``v`` are
        position and velocity in the inertial frame [m] and [m/s],
        ``q`` is the body→inertial quaternion, and ``w`` is the angular
        rate in the body frame [rad/s].
    u_b : (3,) torch.Tensor
        Control torque in body frame [N·m].  If no control is desired,
        pass ``torch.zeros(3, device=device)``.
    params : dict
        Dictionary of parameters.  Recognized keys include:

        ``mass`` (float)
            Spacecraft mass [kg] (required).
        ``J`` ((3,3) tensor)
            Inertia matrix in body frame [kg·m²] (required).
        ``mu`` (float)
            Gravitational parameter [m³/s²].  Default is Earth's.
        ``J2`` (float)
            Second zonal harmonic coefficient.  If provided and
            non‑zero, J2 perturbation is applied.
        ``R_earth`` (float)
            Reference radius for J2 calculation [m].  Default is
            Earth’s mean equatorial radius.

    dt : float
        Integration time step [s].
    device : torch.device, optional
        Explicit device on which to perform the computation.  If
        ``None``, the device of the input state is used.  See
        :func:`device_for_acceleration` to select a default.

    Returns
    -------
    tuple of torch.Tensor
        New state ``(r, v, q, w)`` at ``t + dt``.  All tensors are
        allocated on the same device as the input state or the
        specified device.

    Notes
    -----
    This function performs a simple explicit Euler integration.  For
    high‑fidelity simulations you may wish to implement a higher order
    integrator such as Runge–Kutta 4; however Euler is sufficient to
    demonstrate GPU acceleration and is adequate for short time steps.
    """
    r, v, q, w = state
    if device is None:
        device = r.device
    # Cast parameters to floats/tensors on the correct device
    mu = float(params.get("mu", 3.986004418e14))
    mass = float(params["mass"])
    J = params["J"].to(device)
    J2c = float(params.get("J2", 0.0))
    R = float(params.get("R_earth", 6_378_137.0))
    # Compute accelerations
    a_eci = _two_body_accel(r, mu)
    if J2c != 0.0:
        a_eci = a_eci + _j2_accel(r, mu, R, J2c)
    # Update translational state using explicit Euler
    r_next = r + v * dt
    v_next = v + a_eci * dt
    # Rotational dynamics
    # Quaternion derivative
    qdot = _omega_to_qdot(q, w)
    q_next = q + qdot * dt
    q_next = q_next / torch.linalg.norm(q_next)
    # Angular acceleration from control torque only (neglect disturbance torques on GPU)
    tau = u_b
    wdot = torch.linalg.solve(J, tau - torch.cross(w, J @ w))
    w_next = w + wdot * dt
    return r_next, v_next, q_next, w_next


__all__ = ["device_for_acceleration", "rigid_body_6dof_step_torch"]