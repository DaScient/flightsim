"""
Control algorithms for the flight library.

This subpackage contains simple controllers such as PID and LQR.
Controllers are implemented as classes or helper functions and are
intentionally minimal; they can be extended or combined by users to
suit mission‑specific requirements.
"""

from .pid import PID
from .lqr import dlqr

__all__ = ["PID", "dlqr"]