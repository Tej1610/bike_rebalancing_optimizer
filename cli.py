"""Command-line interface (no dashboard needed).

Examples
  python cli.py --stations 12 --trucks 2 --capacity 10 --algo compare
  python cli.py --csv --trucks 2 --capacity 12
  python cli.py --stations 10 --scenarios 8 --workers 4
"""
from __future__ import annotations

import argparse

import pandas as pd

from services.data_manager import generate_system, load_system_from_csv
from services.objective import ObjectiveWeights
from services.parallel_runner import ParallelRunner
from services.simulation import ALGORITHMS, SimulationEngine, make_optimizer


def main() -> None:
    p = argparse.ArgumentParser(description="Bike-sharing station rebalancing optimizer")
    p.add_argument("--stations", type=int, default=12)
    p.add_argument("--trucks", type=int, default=2)
    p.add_argument("--capacity", type=int, default=10, help="truck capacity (bikes)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--algo", default="compare", choices=["greedy", "dp", "compare"])
    p.add_argument("--csv", action="store_true", help="use data/stations.csv and data/demand.csv")
    p.add_argument("--scenarios", type=int, default=1, help=">1 runs a seed batch in parallel")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=5.0)
    a = p.parse_args()

    names = {"greedy": ["Greedy"], "dp": ["Dynamic Programming"], "compare": list(ALGORITHMS)}[a.algo]
    weights = ObjectiveWeights(a.alpha, a.beta, a.gamma)
    pd.set_option("display.width", 200, "display.max_columns", 30)

    if a.scenarios > 1:
        specs = ParallelRunner.make_specs(a.stations, a.trucks, a.capacity, a.seed, a.scenarios, names, weights)
        rows, secs = ParallelRunner(a.workers).run(specs)
        df = pd.DataFrame(rows)
        print(df.groupby("Algorithm")[["Imbalance after", "Truck distance (km)", "Objective score",
                                       "Optimize time (s)"]].mean().round(3))
        print(f"\n{len(specs)} runs in {secs:.2f}s wall-clock with {ParallelRunner(a.workers).workers} worker(s)")
        return

    system = (load_system_from_csv(a.trucks, a.capacity) if a.csv
              else generate_system(a.stations, a.trucks, a.capacity, a.seed))
    engine = SimulationEngine(weights)
    results = [engine.run(system, make_optimizer(n)) for n in names]
    print(pd.DataFrame([r.metrics() for r in results]).set_index("Algorithm").T)
    for r in results:
        print(f"\n== {r.algorithm} routes ==" if r.status == "ok" else f"\n== {r.algorithm}: {r.message}")
        for t in r.trucks:
            if t.used:
                path = " -> ".join(
                    f"{system.stations[s.station].station_id}({'+' if s.action == 'pickup' else '-'}{s.bikes})"
                    for s in t.route)
                print(f"Truck {t.truck_id}: DEPOT -> {path} -> DEPOT  [{t.distance_km:.2f} km]")


if __name__ == "__main__":
    main()
