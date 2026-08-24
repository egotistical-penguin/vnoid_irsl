"""Parameters published in Kang et al. (IEEE Access, 2025)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


# Tables 2 and 3, proposed experimental-based parameters (c_tilde, k_tilde).
PAPER_CONTAINER_PARAMETERS: dict[int, dict[float, tuple[float, float]]] = {
    1: {
        3.0: (0.1528, 129.77),
        4.0: (0.1845, 221.46),
        5.0: (0.2155, 321.63),
        6.0: (0.2433, 433.62),
        7.0: (0.2691, 550.50),
        8.0: (0.2934, 669.26),
    },
    2: {
        2.0: (0.2217, 83.33),
        3.0: (0.2913, 163.46),
        4.0: (0.3552, 255.26),
        5.0: (0.4129, 356.81),
        6.0: (0.4660, 458.54),
        7.0: (0.5154, 562.97),
    },
}

PAPER_CONTAINER_LENGTHS: dict[int, float] = {1: 0.31, 2: 0.29}


@dataclass(frozen=True, slots=True)
class LIPCPMParameters:
    """Physical parameters for the paper's four-state LIPCPM.

    State order is [x_robot, dx_robot, x_liquid, dx_liquid].
    The control input has acceleration units because B=[0,1,0,0]^T.
    """

    robot_mass: float
    liquid_mass: float
    com_height: float
    spring: float
    damping: float
    gravity: float = 9.80665
    container_id: int | None = None

    PAPER_STATE_WEIGHTS: ClassVar[tuple[float, float, float, float]] = (
        1000.0,
        10.0,
        1000.0,
        50.0,
    )
    PAPER_INPUT_WEIGHT: ClassVar[float] = 1.0e-9

    def __post_init__(self) -> None:
        positive = {
            "robot_mass": self.robot_mass,
            "liquid_mass": self.liquid_mass,
            "com_height": self.com_height,
            "spring": self.spring,
            "gravity": self.gravity,
        }
        bad = [name for name, value in positive.items() if value <= 0.0]
        if bad:
            raise ValueError(f"Parameters must be positive: {', '.join(bad)}")
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")

    @property
    def total_mass(self) -> float:
        return self.robot_mass + self.liquid_mass

    @classmethod
    def from_paper_table(
        cls,
        *,
        container_id: int,
        liquid_mass: float,
        robot_mass: float,
        com_height: float,
        gravity: float = 9.80665,
    ) -> "LIPCPMParameters":
        try:
            damping, spring = PAPER_CONTAINER_PARAMETERS[container_id][
                float(liquid_mass)
            ]
        except KeyError as exc:
            available = PAPER_CONTAINER_PARAMETERS.get(container_id)
            if available is None:
                raise ValueError("container_id must be 1 or 2") from exc
            raise ValueError(
                f"No paper table row for container {container_id}, "
                f"liquid mass {liquid_mass}. Available: {sorted(available)}"
            ) from exc
        return cls(
            robot_mass=float(robot_mass),
            liquid_mass=float(liquid_mass),
            com_height=float(com_height),
            spring=float(spring),
            damping=float(damping),
            gravity=float(gravity),
            container_id=container_id,
        )
