"""
Proportional–integral–derivative (PID) controller implementation.

This PID controller operates componentwise on a vector error.  It
maintains an integral accumulator and uses finite differences to
approximate the derivative term.  Optional integral windup limiting
prevents the integral term from growing without bound.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class PID:
    """A simple vector PID controller.

    Parameters
    ----------
    kp : float or array_like
        Proportional gain.
    ki : float or array_like
        Integral gain.
    kd : float or array_like
        Derivative gain.
    dt : float
        Sampling interval [s].
    integral_limit : float or array_like, optional
        Maximum magnitude of the integral term.  If provided, the
        integral term will be clamped to ±limit for each component.
    """

    def __init__(
        self,
        kp: float | np.ndarray,
        ki: float | np.ndarray,
        kd: float | np.ndarray,
        dt: float,
        *,
        integral_limit: Optional[float | np.ndarray] = None,
    ) -> None:
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.dt = float(dt)
        self.integral_limit = None if integral_limit is None else np.asarray(integral_limit, dtype=float)
        self.integral = np.zeros_like(self.kp)
        self.prev_error = np.zeros_like(self.kp)

    def reset(self) -> None:
        """Reset the internal state of the controller."""
        self.integral = np.zeros_like(self.kp)
        self.prev_error = np.zeros_like(self.kp)

    def update(self, error: np.ndarray) -> np.ndarray:
        """Compute the control output for a given error.

        Parameters
        ----------
        error : array_like
            The error vector at the current time step.

        Returns
        -------
        ndarray
            The control output.
        """
        error = np.asarray(error, dtype=float)
        # Integral term
        self.integral += error * self.dt
        if self.integral_limit is not None:
            # Clamp each component of the integral term
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        # Compute control output
        return self.kp * error + self.ki * self.integral + self.kd * derivative


__all__ = ["PID"]