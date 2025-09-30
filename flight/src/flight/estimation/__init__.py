"""
State estimation algorithms for the flight library.

This subpackage provides a minimal multiplicative extended Kalman
filter (MEKF) for attitude determination.  Additional filters (e.g.
orbit determination, unscented Kalman filters) may be added in future
releases.
"""

from .mekf import MEKF

__all__ = ["MEKF"]