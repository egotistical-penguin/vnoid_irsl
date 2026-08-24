"""Continuous and discrete LIPCPM dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import expm

from .parameters import LIPCPMParameters

Array = np.ndarray
Discretization = Literal["zoh", "euler"]


@dataclass(frozen=True, slots=True)
class DiscreteSystem:
    A: Array
    B: Array
    dt: float

    @property
    def nx(self) -> int:
        return int(self.A.shape[0])

    @property
    def nu(self) -> int:
        return int(self.B.shape[1])


def lipcpm_continuous_matrices(params: LIPCPMParameters) -> tuple[Array, Array]:
    """Return Eq. (19) matrices for X=[x1, dx1, x2, dx2]."""
    m1 = params.robot_mass
    m2 = params.liquid_mass
    M = params.total_mass
    z0 = params.com_height
    g = params.gravity
    k = params.spring
    c = params.damping

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [M * g / (m1 * z0) - k / m1, -c / m1, k / m1, c / m1],
            [0.0, 0.0, 0.0, 1.0],
            [k / m2, c / m2, -k / m2, -c / m2],
        ],
        dtype=float,
    )
    B = np.array([[0.0], [1.0], [0.0], [0.0]], dtype=float)
    return A, B


def discretize(
    A: Array,
    B: Array,
    dt: float,
    method: Discretization = "zoh",
) -> DiscreteSystem:
    """Discretize a continuous LTI system.

    The paper only states a fixed control step and does not report the method.
    Exact zero-order hold is the default; Euler is available for MATLAB-code
    compatibility experiments.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if B.ndim == 1:
        B = B[:, None]
    if B.shape[0] != A.shape[0]:
        raise ValueError("A and B dimensions do not match")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    if method == "euler":
        Ad = np.eye(A.shape[0]) + dt * A
        Bd = dt * B
    elif method == "zoh":
        nx, nu = B.shape
        augmented = np.zeros((nx + nu, nx + nu), dtype=float)
        augmented[:nx, :nx] = A
        augmented[:nx, nx:] = B
        transition = expm(augmented * dt)
        Ad = transition[:nx, :nx]
        Bd = transition[:nx, nx:]
    else:
        raise ValueError(f"Unknown discretization method: {method}")
    return DiscreteSystem(A=Ad, B=Bd, dt=float(dt))


def relative_liquid_state(state: Array) -> tuple[float, float]:
    """Return liquid displacement and velocity relative to the robot body."""
    x = np.asarray(state, dtype=float).reshape(4)
    return float(x[2] - x[0]), float(x[3] - x[1])
