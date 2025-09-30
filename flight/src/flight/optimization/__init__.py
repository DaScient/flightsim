"""
Optimisation and relative motion tools for the flight library.

Currently provides Clohessy–Wiltshire state transition matrix and
mean motion computation for rendezvous and formation flying.
"""

from .cw import mean_motion, cw_state_transition

__all__ = ["mean_motion", "cw_state_transition"]