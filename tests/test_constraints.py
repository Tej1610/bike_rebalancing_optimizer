"""Constraint, conservation and edge-case tests (run with: pytest)."""
import unittest

from algorithms.greedy import GreedyOptimizer
from algorithms.dynamic_programming import DynamicProgrammingOptimizer
from models.truck import DELIVER, PICKUP, Truck
from services.data_manager import generate_system
from services.demand_analyzer import DemandAnalyzer
from services.route_planner import ConstraintViolation, RoutePlanner
from services.simulation import SimulationEngine, make_optimizer

from tests.helpers import make_system, set_targets

ALGOS = ["Greedy", "Dynamic Programming"]


class TargetTests(unittest.TestCase):
    def test_targets_conserve_bikes_and_respect_capacity(self):
        for seed in range(25):
            s = generate_system(12, 2, 8, seed)
            self.assertEqual(sum(x.target for x in s.stations), s.total_bikes)
            for x in s.stations:
                self.assertGreaterEqual(x.target, 0)
                self.assertLessEqual(x.target, x.capacity)

    def test_high_demand_station_is_capped(self):
        s = make_system([5, 5, 5], [6, 30, 30], demand=[100, 1, 1])
        self.assertEqual(s.stations[0].target, 6)
        self.assertEqual(sum(x.target for x in s.stations), 15)

    def test_zero_demand_everywhere_still_works(self):
        s = make_system([4, 0, 2], [10, 10, 10], demand=[0, 0, 0])
        self.assertEqual(sum(x.target for x in s.stations), 6)


class PlanConstraintTests(unittest.TestCase):
    def check(self, system, name):
        before_total = system.total_bikes
        before_inv = system.inventory()
        res = SimulationEngine().run(system, make_optimizer(name))
        self.assertEqual(res.status, "ok")
        # system untouched
        self.assertEqual(system.inventory(), before_inv)
        # conservation
        self.assertEqual(sum(res.inventory_after), before_total)
        for st, b in zip(system.stations, res.inventory_after):
            self.assertGreaterEqual(b, 0)
            self.assertLessEqual(b, st.capacity)
        # trucks
        self.assertLessEqual(len(res.trucks), system.n_trucks)
        for t in res.trucks:
            self.assertEqual(t.load, 0)
            self.assertGreaterEqual(t.distance_km, 0)
            load = 0
            for stop in t.route:
                load += stop.bikes if stop.action == PICKUP else -stop.bikes
                self.assertLessEqual(load, t.capacity)
                self.assertGreaterEqual(load, 0)
        # pickups == deliveries
        pick = sum(s.bikes for t in res.trucks for s in t.route if s.action == PICKUP)
        drop = sum(s.bikes for t in res.trucks for s in t.route if s.action == DELIVER)
        self.assertEqual(pick, drop)
        self.assertEqual(res.bikes_moved, drop)
        return res

    def test_random_scenarios_all_constraints(self):
        for name in ALGOS:
            for seed in range(8):
                for trucks, cap in ((1, 3), (3, 10), (5, 25)):
                    s = generate_system(9, trucks, cap, seed)
                    res = self.check(s, name)
                    self.assertEqual(res.imbalance_after, 0)

    def test_large_greedy_scenario(self):
        s = generate_system(80, 6, 15, 7)
        res = self.check(s, "Greedy")
        self.assertEqual(res.imbalance_after, 0)

    def test_transfer_larger_than_truck_is_split(self):
        s = set_targets(make_system([20, 0], [30, 30], truck_capacity=4), [10, 10])
        for name in ALGOS:
            res = self.check(s, name)
            self.assertEqual(res.bikes_moved, 10)
            for t in res.trucks:
                self.assertTrue(all(st.bikes <= 4 for st in t.route))


class EdgeCaseTests(unittest.TestCase):
    def test_already_balanced(self):
        s = set_targets(make_system([3, 4, 5], [10, 10, 10]), [3, 4, 5])
        for name in ALGOS:
            res = SimulationEngine().run(s, make_optimizer(name))
            self.assertEqual(res.bikes_moved, 0)
            self.assertEqual(res.trucks_used, 0)
            self.assertEqual(res.distance_km, 0)
            self.assertEqual(res.imbalance_after, 0)

    def test_empty_and_full_stations(self):
        s = set_targets(make_system([0, 10, 0, 10], [10, 10, 10, 10]), [5, 5, 5, 5])
        for name in ALGOS:
            res = SimulationEngine().run(s, make_optimizer(name))
            self.assertEqual(res.inventory_after, [5, 5, 5, 5])

    def test_no_surplus_only_shortage_and_vice_versa(self):
        # targets that cannot be met (bikes not conserved): planners must not crash
        s = set_targets(make_system([2, 2], [10, 10]), [6, 6])
        for name in ALGOS:
            res = SimulationEngine().run(s, make_optimizer(name))
            self.assertEqual(res.bikes_moved, 0)
        s = set_targets(make_system([8, 8], [10, 10]), [2, 2])
        for name in ALGOS:
            res = SimulationEngine().run(s, make_optimizer(name))
            self.assertEqual(res.bikes_moved, 0)

    def test_single_station(self):
        s = make_system([3], [10])
        for name in ALGOS:
            res = SimulationEngine().run(s, make_optimizer(name))
            self.assertEqual(res.bikes_moved, 0)

    def test_planner_rejects_bad_routes(self):
        s = set_targets(make_system([5, 0], [10, 10], truck_capacity=3), [2, 3])
        bad = Truck(1, 3)
        bad.route = []
        bad.add_stop(0, PICKUP, 3)
        with self.assertRaises(ValueError):
            bad.add_stop(0, PICKUP, 1)  # truck capacity
        with self.assertRaises(ConstraintViolation):
            RoutePlanner.execute(s, [bad])  # ends with bikes on board
        t = Truck(1, 99)
        t.add_stop(0, PICKUP, 9)
        t.add_stop(1, DELIVER, 9)
        with self.assertRaises(ConstraintViolation):
            RoutePlanner.execute(s, [t])  # station 0 has only 5 bikes

    def test_station_capacity_enforced_in_execution(self):
        s = set_targets(make_system([5, 8], [10, 10], truck_capacity=9), [5, 8])
        t = Truck(1, 9)
        t.add_stop(0, PICKUP, 5)
        t.add_stop(1, DELIVER, 5)  # 8 + 5 > 10
        with self.assertRaises(ConstraintViolation):
            RoutePlanner.execute(s, [t])

    def test_distances_non_negative_and_symmetric(self):
        s = generate_system(10, 2, 5, 3)
        for i in range(-1, 10):
            for j in range(-1, 10):
                self.assertGreaterEqual(s.distance(i, j), 0)
                self.assertAlmostEqual(s.distance(i, j), s.distance(j, i))
            self.assertEqual(s.distance(i, i), 0)

    def test_dp_work_budget(self):
        s = generate_system(15, 2, 10, 42)
        res = SimulationEngine().run(s, DynamicProgrammingOptimizer(max_work=50))
        self.assertEqual(res.status, "skipped")
        self.assertIn("budget", res.message)

    def test_imbalance_report(self):
        s = set_targets(make_system([8, 1, 4], [10, 10, 10]), [4, 5, 4])
        r = DemandAnalyzer.analyze(s)
        self.assertEqual(r.surplus, {0: 4})
        self.assertEqual(r.shortage, {1: 4})
        self.assertEqual(r.total_imbalance, 8)


if __name__ == "__main__":
    unittest.main()
