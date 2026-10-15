"""Run independent scenarios concurrently with multiprocessing.

Each scenario is described by a small picklable ``ScenarioSpec``; the worker
rebuilds the system from its seed, so results are reproducible and no large
objects are sent between processes. Worker functions live at module level so
they can be imported by ``spawn``ed processes (required on Windows/macOS).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from services.data_manager import generate_system
from services.objective import ObjectiveWeights
from services.simulation import SimulationEngine, make_optimizer


@dataclass(frozen=True)
class ScenarioSpec:
    n_stations: int
    n_trucks: int
    truck_capacity: int
    seed: int
    algorithm: str
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 5.0


def run_scenario(spec: ScenarioSpec) -> dict:
    """Worker: build the scenario, run one algorithm, return flat metrics."""
    system = generate_system(spec.n_stations, spec.n_trucks, spec.truck_capacity, spec.seed)
    engine = SimulationEngine(ObjectiveWeights(spec.alpha, spec.beta, spec.gamma))
    res = engine.run(system, make_optimizer(spec.algorithm))
    row = res.metrics()
    row.update({"Seed": spec.seed, "Stations": spec.n_stations, "Message": res.message})
    return row


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


class ParallelRunner:
    def __init__(self, workers: int | None = None) -> None:
        self.workers = max(1, workers or default_workers())

    def run(self, specs: list[ScenarioSpec]) -> tuple[list[dict], float]:
        """Return (rows in the same order as ``specs``, wall-clock seconds)."""
        t0 = time.perf_counter()
        if self.workers == 1 or len(specs) <= 1:
            rows = [run_scenario(s) for s in specs]
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(self.workers, len(specs)), mp_context=ctx) as pool:
                rows = list(pool.map(run_scenario, specs, chunksize=1))
        return rows, time.perf_counter() - t0

    @staticmethod
    def make_specs(n_stations: int, n_trucks: int, truck_capacity: int, base_seed: int,
                   n_scenarios: int, algorithms: list[str],
                   weights: ObjectiveWeights | None = None) -> list[ScenarioSpec]:
        w = weights or ObjectiveWeights()
        return [
            ScenarioSpec(n_stations, n_trucks, truck_capacity, base_seed + k, alg, w.alpha, w.beta, w.gamma)
            for k in range(n_scenarios)
            for alg in algorithms
        ]
