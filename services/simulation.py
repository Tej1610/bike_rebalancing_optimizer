"""SimulationEngine: run an optimizer on a scenario and evaluate the result."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from algorithms.base import Optimizer, ProblemTooLargeError, Transfer, transfer_cost
from algorithms.dynamic_programming import DynamicProgrammingOptimizer
from algorithms.greedy import GreedyOptimizer
from models.system import BikeSharingSystem
from models.truck import Truck
from services.demand_analyzer import DemandAnalyzer
from services.objective import ObjectiveWeights
from services.route_planner import RoutePlanner

ALGORITHMS = {"Greedy": GreedyOptimizer, "Dynamic Programming": DynamicProgrammingOptimizer}


@dataclass
class RunResult:
    algorithm: str
    status: str  # "ok" or "skipped"
    message: str = ""
    imbalance_before: int = 0
    imbalance_after: int = 0
    bikes_moved: int = 0
    distance_km: float = 0.0
    transfer_cost: float = 0.0
    trucks_used: int = 0
    score: float = float("nan")
    optimize_time_s: float = 0.0
    total_time_s: float = 0.0
    transfers: list[Transfer] = field(default_factory=list)
    trucks: list[Truck] = field(default_factory=list)
    inventory_after: list[int] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        if self.imbalance_before == 0:
            return 0.0
        return (self.imbalance_before - self.imbalance_after) / self.imbalance_before * 100

    def metrics(self) -> dict:
        return {
            "Algorithm": self.algorithm,
            "Status": self.status,
            "Imbalance before": self.imbalance_before,
            "Imbalance after": self.imbalance_after,
            "Reduction %": round(self.reduction_pct, 1),
            "Bikes moved": self.bikes_moved,
            "Truck distance (km)": round(self.distance_km, 2),
            "Transfer cost (bike-km)": round(self.transfer_cost, 2),
            "Trucks used": self.trucks_used,
            "Objective score": round(self.score, 2),
            "Optimize time (s)": round(self.optimize_time_s, 4),
            "Total time (s)": round(self.total_time_s, 4),
        }


def make_optimizer(name: str) -> Optimizer:
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}")
    return ALGORITHMS[name]()


class SimulationEngine:
    def __init__(self, weights: ObjectiveWeights | None = None, balance_trucks: bool = True) -> None:
        self.weights = weights or ObjectiveWeights()
        self.planner = RoutePlanner(balance_trucks)

    def run(self, system: BikeSharingSystem, optimizer: Optimizer) -> RunResult:
        """Run one optimizer. ``system`` is never modified."""
        before = DemandAnalyzer.analyze(system).total_imbalance
        t0 = time.perf_counter()
        try:
            transfers = optimizer.plan(system.clone())
        except ProblemTooLargeError as exc:
            return RunResult(optimizer.name, "skipped", str(exc), imbalance_before=before,
                             imbalance_after=before, optimize_time_s=time.perf_counter() - t0,
                             total_time_s=time.perf_counter() - t0)
        t1 = time.perf_counter()
        trucks = self.planner.plan(system, transfers)
        inv_after = self.planner.execute(system, trucks)  # validates every constraint
        t2 = time.perf_counter()

        after = sum(abs(inv_after[s.index] - s.target) for s in system.stations)
        used = sum(1 for t in trucks if t.used)
        dist = sum(t.distance_km for t in trucks)
        moved = sum(t.bikes_moved for t in trucks)
        return RunResult(
            algorithm=optimizer.name,
            status="ok",
            imbalance_before=before,
            imbalance_after=after,
            bikes_moved=moved,
            distance_km=dist,
            transfer_cost=transfer_cost(system, transfers),
            trucks_used=used,
            score=self.weights.score(dist, after, used),
            optimize_time_s=t1 - t0,
            total_time_s=t2 - t0,
            transfers=transfers,
            trucks=trucks,
            inventory_after=inv_after,
        )

    def compare(self, system: BikeSharingSystem, names: list[str] | None = None) -> list[RunResult]:
        """Run several algorithms on the *same* scenario with the *same* objective."""
        return [self.run(system, make_optimizer(n)) for n in (names or list(ALGORITHMS))]
