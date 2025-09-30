"""
Dynamics models for the flight library.

This subpackage currently contains a single six‑degree‑of‑freedom
rigid‑body integrator that can simulate spacecraft motion under
various perturbations.  Future versions may include simplified
models for other vehicle types.
"""

from .rigid_body import rigid_body_6dof_step, MU_EARTH, R_EARTH, J2_EARTH

__all__ = [
    "rigid_body_6dof_step",
    "MU_EARTH",
    "R_EARTH",
    "J2_EARTH",
]