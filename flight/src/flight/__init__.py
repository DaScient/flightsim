"""
Flight: A modular 6‑DOF simulation and GNC library.

The top‑level package exposes core functions and classes for rigid‑body dynamics,
guidance, navigation and control.  Submodules provide specific implementations
for simulation (``flight.dynamics``), control algorithms (``flight.control``),
estimation filters (``flight.estimation``), orbital propagators
(``flight.orbit``) and GPU acceleration (``flight.gpu``).

Example usage::

    from flight.dynamics import rigid_body
    from flight.control import pid

    # define initial state and parameters...
    next_state = rigid_body.rigid_body_6dof_step(state, control_torque, params, dt)

The package is intentionally simple to import and use.  Optional dependencies
such as PyTorch, fastplotlib and Streamlit extend the capabilities of
``flight`` without being required for the core functionality.

"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

__all__ = [
    "control",
    "dynamics",
    "estimation",
    "mathutils",
    "orbit",
    "optimization",
    "gpu",
    "web",
]