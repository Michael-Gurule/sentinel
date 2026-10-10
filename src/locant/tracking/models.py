"""Motion models.

State layout is ``[x, y, z, vx, vy, vz]`` (positions in m, velocities in m/s).
"""

from dataclasses import dataclass

import numpy as np

from locant.core.linalg import FloatArray


@dataclass(frozen=True)
class ConstantVelocity:
    """Nearly-constant-velocity model driven by continuous white-noise acceleration.

    Discretizing ẍ = w(t), E[w(t)w(τ)] = q δ(t-τ), over an interval Δt gives
    (Bar-Shalom, Li & Kirubarajan, §6.2.2)::

        F = [[I, Δt·I], [0, I]]
        Q = q · [[Δt³/3·I, Δt²/2·I], [Δt²/2·I, Δt·I]]

    Q is positive semidefinite for every Δt ≥ 0.

    Attributes:
        noise_intensity: Acceleration power spectral density q in m²/s³.
            A rule of thumb is q ≈ σₐ² · Δt for an expected acceleration
            standard deviation σₐ over a typical update interval.
        dim: Spatial dimensions (3 for x, y, z).
    """

    noise_intensity: float
    dim: int = 3

    def __post_init__(self) -> None:
        if self.noise_intensity < 0:
            raise ValueError("noise_intensity must be non-negative")
        if self.dim < 1:
            raise ValueError("dim must be >= 1")

    @property
    def state_dim(self) -> int:
        return 2 * self.dim

    def transition(self, dt: float) -> FloatArray:
        """State transition matrix F(Δt)."""
        _check_dt(dt)
        block = np.array([[1.0, dt], [0.0, 1.0]])
        return np.kron(block, np.eye(self.dim))

    def process_noise(self, dt: float) -> FloatArray:
        """Process noise covariance Q(Δt)."""
        _check_dt(dt)
        block = np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
        return np.asarray(self.noise_intensity * np.kron(block, np.eye(self.dim)))

    def position_matrix(self) -> FloatArray:
        """Measurement matrix selecting position from the state."""
        return np.hstack([np.eye(self.dim), np.zeros((self.dim, self.dim))])


def _check_dt(dt: float) -> None:
    if dt < 0:
        raise ValueError(f"time step must be non-negative, got {dt}")
