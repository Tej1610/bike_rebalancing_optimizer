"""Objective function shared by every algorithm.

score = alpha * D + beta * I_after + gamma * V

* D        total truck distance in km (depot -> stops -> depot, summed over trucks)
* I_after  remaining imbalance, sum_i |B_i - T_i|, in bikes
* V        number of trucks that were dispatched

Default weights: 1 point per km, 2 points per bike still out of place and
5 points per dispatched truck. Both algorithms are always scored with the same
weights on the same scenario so scores are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveWeights:
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 5.0

    def score(self, distance_km: float, imbalance_after: int, trucks_used: int) -> float:
        return self.alpha * distance_km + self.beta * imbalance_after + self.gamma * trucks_used
