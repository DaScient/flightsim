"""
GPU acceleration subpackage for the flight library.

This subpackage provides functions for running 6‑DOF dynamics on
PyTorch tensors.  To use these functions, install PyTorch with
either CUDA or Metal Performance Shaders support.  The GPU
implementation currently supports only two‑body gravity and J2
perturbation; additional effects can be added in future versions.
"""

from .torch_dynamics import device_for_acceleration, rigid_body_6dof_step_torch

__all__ = ["device_for_acceleration", "rigid_body_6dof_step_torch"]