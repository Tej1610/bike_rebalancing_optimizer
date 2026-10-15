"""Common optimizer interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from models.system import BikeSharingSystem


@dataclass(frozen=True)
class Transfer:
    """Move ``bikes`` bikes from station ``src`` to station ``dst`` (indices)."""

    src: int
    dst: int
    bikes: int


class ProblemTooLargeError(RuntimeError):
    """Raised when an exact algorithm would exceed its state/work budget."""


class Optimizer(ABC):
    name: str = "optimizer"

    @abstractmethod
    def plan(self, system: BikeSharingSystem) -> list[Transfer]:
        """Return transfers that rebalance ``system`` (system is not modified)."""


def transfer_cost(system: BikeSharingSystem, transfers: list[Transfer]) -> float:
    """Simplified transfer cost: sum of d_ij * x_ij (bike-km)."""
    return sum(system.distance(t.src, t.dst) * t.bikes for t in transfers)
