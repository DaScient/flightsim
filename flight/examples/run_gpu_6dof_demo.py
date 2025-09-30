"""
Run a simple six‑degree‑of‑freedom simulation on the GPU.

This example uses the GPU‑enabled dynamics integrator provided in
``flight.gpu`` to propagate a spacecraft state and plot the altitude
above Earth’s surface over time.  The simulation runs a circular
low‑Earth orbit for one hour with a ten‑second time step.  If a GPU
device is available (MPS on Apple Silicon or CUDA on an NVIDIA card)
and PyTorch is installed with the appropriate backend, the code
automatically selects it; otherwise it falls back to the CPU.

Usage::

    PYTHONPATH=src python examples/run_gpu_6dof_demo.py

Note that this script requires PyTorch; install a build with MPS or
CUDA support according to your platform.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This demo requires PyTorch. Please install PyTorch with MPS or CUDA support."
    ) from exc

from flight.gpu import rigid_body_6dof_step_torch, device_for_acceleration


def main() -> None:
    device = device_for_acceleration()
    print(f"Using device: {device}")

    # Physical constants
    mu = 3.986004418e14  # m^3/s^2
    R_earth = 6_378_137.0  # m

    # Initial conditions: 7000 km circular orbit with slight spin
    r0 = torch.tensor([7_000_000.0, 0.0, 0.0], dtype=torch.float64, device=device)
    v0 = torch.tensor([0.0, 7_546.05329, 0.0], dtype=torch.float64, device=device)
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
    w0 = torch.tensor([0.0, 0.0, 0.01], dtype=torch.float64, device=device)
    state = (r0, v0, q0, w0)

    # Spacecraft properties
    J = torch.diag(torch.tensor([0.12, 0.10, 0.20], dtype=torch.float64, device=device))
    params = {
        "mass": 10.0,
        "J": J,
        "mu": mu,
        "J2": 1.08262668e-3,
        "R_earth": R_earth,
    }
    # No control torques
    u = torch.zeros(3, dtype=torch.float64, device=device)

    # Simulation configuration
    dt = 10.0  # seconds
    duration = 3600.0  # one hour
    n_steps = int(duration / dt)

    times: list[float] = []
    altitudes: list[float] = []

    for step in range(n_steps):
        state = rigid_body_6dof_step_torch(state, u, params, dt, device=device)
        r, v, q, w = state
        times.append((step + 1) * dt)
        # Compute altitude (position magnitude minus Earth radius) on CPU
        alt = float(torch.linalg.norm(r).cpu().numpy()) - R_earth
        altitudes.append(alt)

    # Plot altitude vs. time
    plt.figure(figsize=(8, 4))
    plt.plot(times, altitudes)
    plt.title("Altitude vs. time (GPU simulation)")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude above Earth's surface [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()