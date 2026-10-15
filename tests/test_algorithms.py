"""Algorithm correctness tests (run with: pytest)."""
import itertools
import random
import unittest

from algorithms.base import transfer_cost
from algorithms.dynamic_programming import DynamicProgrammingOptimizer
from algorithms.greedy import GreedyOptimizer
from services.data_manager import generate_system
from services.demand_analyzer import DemandAnalyzer
from services.objective import ObjectiveWeights
from services.parallel_runner import ParallelRunner
from services.simulation import SimulationEngine

from tests.helpers import make_system, set_targets


def brute_force_cost(system):
    """Independent exhaustive search over every transfer matrix (tiny cases only)."""
    rep = DemandAnalyzer.analyze(system)
    src, dst = sorted(rep.surplus), sorted(rep.shortage)
    best = float("inf")
    ranges = [range(min(rep.surplus[i], rep.shortage[j]) + 1) for i in src for j in dst]
    for combo in itertools.product(*ranges):
        x = {(i, j): combo[a * len(dst) + b] for a, i in enumerate(src) for b, j in enumerate(dst)}
        if any(sum(x[i, j] for j in dst) != rep.surplus[i] for i in src):
            continue
        if any(sum(x[i, j] for i in src) != rep.shortage[j] for j in dst):
            continue
        best = min(best, sum(system.distance(i, j) * q for (i, j), q in x.items()))
    return best


class DPTests(unittest.TestCase):
    def random_small_system(self, rng):
        while True:
            n = rng.randint(3, 6)
            caps = [rng.randint(4, 8) for _ in range(n)]
            bikes = [rng.randint(0, c) for c in caps]
            s = make_system(bikes, caps, demand=[rng.uniform(0.2, 3) for _ in range(n)],
                            coords=[(12.9 + rng.random() * 0.08, 77.5 + rng.random() * 0.08) for _ in range(n)])
            rep = DemandAnalyzer.analyze(s)
            cells = len(rep.surplus) * len(rep.shortage)
            if rep.surplus and rep.shortage and cells <= 6 and rep.total_imbalance <= 16:
                return s

    def test_dp_matches_exhaustive_search(self):
        rng = random.Random(123)
        for _ in range(60):
            s = self.random_small_system(rng)
            dp = DynamicProgrammingOptimizer().plan(s)
            self.assertAlmostEqual(transfer_cost(s, dp), brute_force_cost(s), places=9)

    def test_dp_never_worse_than_greedy_on_transfer_cost(self):
        for seed in range(30):
            s = generate_system(10, 2, 10, seed)
            g = transfer_cost(s, GreedyOptimizer().plan(s))
            d = transfer_cost(s, DynamicProgrammingOptimizer().plan(s))
            self.assertLessEqual(d, g + 1e-9)

    def test_dp_beats_greedy_on_classic_trap(self):
        # greedy serves the big shortage from the nearest source and starves the second shortage
        coords = [(12.970, 77.590), (12.970, 77.600), (12.970, 77.640), (12.970, 77.5995)]
        s = set_targets(make_system([5, 5, 0, 0], [10, 10, 10, 10], coords=coords), [0, 5, 5, 0])
        g = transfer_cost(s, GreedyOptimizer().plan(s))
        d = transfer_cost(s, DynamicProgrammingOptimizer().plan(s))
        self.assertLessEqual(d, g)

    def test_dp_role_swap_gives_same_optimum(self):
        # many shortages / few surplus stations and the reverse
        for bikes, targets in (([30, 0, 0, 0, 0], [6, 6, 6, 6, 6]), ([6, 6, 6, 6, 6], [30, 0, 0, 0, 0])):
            s = set_targets(make_system(bikes, [40] * 5), targets)
            cost = transfer_cost(s, DynamicProgrammingOptimizer().plan(s))
            self.assertAlmostEqual(cost, brute_force_cost(s), places=9)


class MetricTests(unittest.TestCase):
    def test_same_scenario_same_objective(self):
        s = generate_system(10, 2, 8, 5)
        w = ObjectiveWeights(1.0, 2.0, 5.0)
        res = SimulationEngine(w).compare(s)
        self.assertEqual(res[0].imbalance_before, res[1].imbalance_before)
        for r in res:
            self.assertAlmostEqual(r.score, w.score(r.distance_km, r.imbalance_after, r.trucks_used))

    def test_reduction_percentage(self):
        s = generate_system(10, 2, 8, 5)
        r = SimulationEngine().compare(s)[0]
        self.assertAlmostEqual(r.reduction_pct, (r.imbalance_before - r.imbalance_after) / r.imbalance_before * 100)


class ParallelTests(unittest.TestCase):
    def test_parallel_matches_serial_and_is_reproducible(self):
        specs = ParallelRunner.make_specs(8, 2, 8, 100, 4, ["Greedy", "Dynamic Programming"])

        def strip(rows):  # timings legitimately differ
            return [{k: v for k, v in r.items() if "time" not in k} for r in rows]

        serial, _ = ParallelRunner(1).run(specs)
        par1, _ = ParallelRunner(2).run(specs)
        par2, _ = ParallelRunner(2).run(specs)
        self.assertEqual(strip(serial), strip(par1))
        self.assertEqual(strip(par1), strip(par2))


if __name__ == "__main__":
    unittest.main()
