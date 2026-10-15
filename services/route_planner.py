"""Turn transfers into truck routes and validate/execute them."""
from __future__ import annotations

from dataclasses import dataclass

from algorithms.base import Transfer
from models.system import DEPOT, BikeSharingSystem
from models.truck import DELIVER, PICKUP, Truck


class ConstraintViolation(RuntimeError):
    pass


@dataclass
class Trip:
    src: int
    dst: int
    bikes: int


class RoutePlanner:
    """Assign trips to trucks.

    * A transfer larger than the truck capacity is split into several trips.
    * A trip is: go to ``src``, pick up ``bikes``, go to ``dst``, deliver them.
    * ``balance_trucks=True``  : each trip goes to the truck with the shortest finished route,
      which spreads work over the fleet (lower makespan, usually more km).
      ``balance_trucks=False`` : each trip goes to the truck whose route grows the least
      (fewest km, often a single truck does everything).
    * Routes end at the depot.
    * The fleet size is fixed, so trucks used <= available trucks.
    """

    def __init__(self, balance_trucks: bool = True) -> None:
        self.balance_trucks = balance_trucks

    @staticmethod
    def split_into_trips(transfers: list[Transfer], truck_capacity: int) -> list[Trip]:
        trips: list[Trip] = []
        for t in transfers:
            left = t.bikes
            while left > 0:
                q = min(left, truck_capacity)
                trips.append(Trip(t.src, t.dst, q))
                left -= q
        return trips

    def plan(self, system: BikeSharingSystem, transfers: list[Transfer]) -> list[Truck]:
        trucks = system.new_trucks()
        trips = self.split_into_trips(transfers, system.truck_capacity)
        # nearest-to-depot first gives compact tours
        trips.sort(key=lambda t: (system.distance(DEPOT, t.src), t.src, t.dst))
        for trip in trips:
            best_truck, best_delta = None, float("inf")
            for truck in trucks:
                here = truck.location
                delta = (
                    system.distance(here, trip.src)
                    + system.distance(trip.src, trip.dst)
                    + system.distance(trip.dst, DEPOT)
                    - system.distance(here, DEPOT)
                )
                closed_len = (truck.distance_km + system.distance(here, trip.src)
                              + system.distance(trip.src, trip.dst) + system.distance(trip.dst, DEPOT))
                key = closed_len if self.balance_trucks else delta
                if key < best_delta - 1e-12:
                    best_truck, best_delta = truck, key
            assert best_truck is not None
            best_truck.distance_km += (
                system.distance(best_truck.location, trip.src) + system.distance(trip.src, trip.dst)
            )
            best_truck.add_stop(trip.src, PICKUP, trip.bikes)
            best_truck.add_stop(trip.dst, DELIVER, trip.bikes)
        for truck in trucks:
            if truck.used:
                truck.distance_km += system.distance(truck.location, DEPOT)
        return trucks

    @staticmethod
    def execute(system: BikeSharingSystem, trucks: list[Truck]) -> list[int]:
        """Simulate the routes on a copy of the inventory; raise on any violation.

        Returns the inventory after all routes were executed.
        """
        if len(trucks) > system.n_trucks:
            raise ConstraintViolation("more trucks used than available")
        inv = system.inventory()
        caps = [s.capacity for s in system.stations]
        for truck in trucks:
            load = 0
            for stop in truck.route:
                if stop.station < 0 or stop.station >= system.n_stations:
                    raise ConstraintViolation("stop refers to an unknown station")
                if stop.action == PICKUP:
                    if inv[stop.station] < stop.bikes:
                        raise ConstraintViolation(f"station {stop.station} lacks bikes for pickup")
                    inv[stop.station] -= stop.bikes
                    load += stop.bikes
                    if load > truck.capacity:
                        raise ConstraintViolation(f"truck {truck.truck_id} over capacity")
                elif stop.action == DELIVER:
                    if load < stop.bikes:
                        raise ConstraintViolation(f"truck {truck.truck_id} delivers bikes it does not carry")
                    load -= stop.bikes
                    inv[stop.station] += stop.bikes
                    if inv[stop.station] > caps[stop.station]:
                        raise ConstraintViolation(f"station {stop.station} over capacity")
                else:
                    raise ConstraintViolation(f"unknown action {stop.action!r}")
            if load != 0:
                raise ConstraintViolation(f"truck {truck.truck_id} ends with bikes on board")
        return inv
