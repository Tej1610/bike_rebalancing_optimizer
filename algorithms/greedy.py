"""Greedy baseline."""
from __future__ import annotations

from models.system import BikeSharingSystem
from services.demand_analyzer import DemandAnalyzer

from .base import Optimizer, Transfer


class GreedyOptimizer(Optimizer):
    """Serve the largest shortage first from its nearest station that has surplus.

    Each step is locally best (nearest source), which is fast but can lock in a
    poor overall assignment because later shortages get the leftover surplus.
    """

    name = "Greedy"

    def plan(self, system: BikeSharingSystem) -> list[Transfer]:
        report = DemandAnalyzer.analyze(system)
        supply = dict(report.surplus)
        need = dict(report.shortage)
        transfers: list[Transfer] = []
        while need and supply:
            dst = max(need, key=lambda j: (need[j], -j))
            src = min(supply, key=lambda i: (system.distance(i, dst), i))
            qty = min(supply[src], need[dst])
            transfers.append(Transfer(src, dst, qty))
            supply[src] -= qty
            need[dst] -= qty
            if supply[src] == 0:
                del supply[src]
            if need[dst] == 0:
                del need[dst]
        return transfers
