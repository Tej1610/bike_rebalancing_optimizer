"""Truck model."""
from __future__ import annotations

from dataclasses import dataclass, field

PICKUP = "pickup"
DELIVER = "deliver"


@dataclass
class Stop:
    station: int  # station index
    action: str  # PICKUP or DELIVER
    bikes: int


@dataclass
class Truck:
    truck_id: int
    capacity: int
    load: int = 0
    route: list[Stop] = field(default_factory=list)
    distance_km: float = 0.0
    location: int = -1  # -1 == depot

    @property
    def used(self) -> bool:
        return bool(self.route)

    @property
    def bikes_moved(self) -> int:
        return sum(s.bikes for s in self.route if s.action == DELIVER)

    def add_stop(self, station: int, action: str, bikes: int) -> None:
        """Append a stop and update the load, enforcing truck capacity."""
        if bikes <= 0:
            raise ValueError("stop must move a positive number of bikes")
        if action == PICKUP:
            if self.load + bikes > self.capacity:
                raise ValueError(f"truck {self.truck_id}: capacity {self.capacity} exceeded")
            self.load += bikes
        elif action == DELIVER:
            if bikes > self.load:
                raise ValueError(f"truck {self.truck_id}: cannot deliver more than it carries")
            self.load -= bikes
        else:
            raise ValueError(f"unknown action {action!r}")
        self.route.append(Stop(station, action, bikes))
        self.location = station
