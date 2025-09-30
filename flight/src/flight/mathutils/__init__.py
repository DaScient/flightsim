"""
Mathematical helper functions for the flight library.

This subpackage includes quaternion arithmetic and small matrix utilities.
"""

from .quaternion import qnorm, qmul, qconj, qrot, omega_to_qdot
from .skew import skew

__all__ = ["qnorm", "qmul", "qconj", "qrot", "omega_to_qdot", "skew"]