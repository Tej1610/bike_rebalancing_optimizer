"""Target inventory and imbalance analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass

from models.station import Station
from models.system import BikeSharingSystem


@dataclass
class ImbalanceReport:
    surplus: dict[int, int]  # station index -> bikes that can be given away
    shortage: dict[int, int]  # station index -> bikes needed
    total_imbalance: int  # sum_i |B_i - T_i|

    @property
    def total_surplus(self) -> int:
        return sum(self.surplus.values())

    @property
    def total_shortage(self) -> int:
        return sum(self.shortage.values())


class DemandAnalyzer:
    """Computes target inventories from predicted demand and classifies stations."""

    @staticmethod
    def compute_targets(stations: list[Station]) -> list[int]:
        """Integer targets, proportional to predicted demand.

        * The targets sum to the total number of bikes (bikes are conserved).
        * 0 <= target_i <= capacity_i (water-filling when a station would overflow).
        """
        n = len(stations)
        if n == 0:
            return []
        total = sum(s.bikes for s in stations)
        caps = [s.capacity for s in stations]
        w = [max(s.predicted_demand, 0.0) for s in stations]
        if sum(w) <= 0:
            w = [1.0] * n

        target = [0.0] * n
        free = set(range(n))
        remaining = float(total)
        while free and remaining > 1e-9:
            wsum = sum(w[i] for i in free)
            if wsum <= 0:
                for i in free:
                    w[i] = 1.0
                wsum = float(len(free))
            alloc = {i: remaining * w[i] / wsum for i in free}
            over = [i for i in free if alloc[i] > caps[i] + 1e-12]
            if not over:
                for i in free:
                    target[i] = alloc[i]
                remaining = 0.0
                break
            for i in over:
                target[i] = float(caps[i])
                remaining -= caps[i]
                free.discard(i)

        # Largest-remainder rounding that keeps the exact total and the caps.
        result = [min(int(math.floor(t + 1e-9)), caps[i]) for i, t in enumerate(target)]
        leftover = total - sum(result)
        order = sorted(range(n), key=lambda i: (-(target[i] - result[i]), i))
        while leftover > 0:
            progressed = False
            for i in order:
                if leftover == 0:
                    break
                if result[i] < caps[i]:
                    result[i] += 1
                    leftover -= 1
                    progressed = True
            if not progressed:  # cannot happen when total <= total capacity
                break
        return result

    def apply_targets(self, system: BikeSharingSystem) -> None:
        for s, t in zip(system.stations, self.compute_targets(system.stations)):
            s.target = t

    @staticmethod
    def analyze(system: BikeSharingSystem) -> ImbalanceReport:
        surplus = {s.index: s.surplus for s in system.stations if s.surplus > 0}
        shortage = {s.index: s.shortage for s in system.stations if s.shortage > 0}
        total = sum(s.imbalance for s in system.stations)
        return ImbalanceReport(surplus, shortage, total)
