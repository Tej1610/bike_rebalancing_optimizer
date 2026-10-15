"""BikeSharingSystem: stations, trucks, depot and distances."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .station import Station
from .truck import Truck

DEPOT = -1
EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    la = np.radians(lats)[:, None]
    lo = np.radians(lons)[:, None]
    dlat = la - la.T
    dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass
class BikeSharingSystem:
    stations: list[Station]
    n_trucks: int
    truck_capacity: int
    depot_lat: float
    depot_lon: float
    _dist: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_trucks < 1:
            raise ValueError("need at least one truck")
        if self.truck_capacity < 1:
            raise ValueError("truck capacity must be positive")
        for i, s in enumerate(self.stations):
            s.index = i
            s.validate()
        lats = np.array([s.lat for s in self.stations] + [self.depot_lat], dtype=float)
        lons = np.array([s.lon for s in self.stations] + [self.depot_lon], dtype=float)
        self._dist = haversine_matrix(lats, lons)

    # -- accessors --------------------------------------------------------
    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def total_bikes(self) -> int:
        return sum(s.bikes for s in self.stations)

    def distance(self, a: int, b: int) -> float:
        """Distance in km between station indices (``DEPOT`` = -1 is the depot)."""
        n = self.n_stations
        return float(self._dist[n if a == DEPOT else a, n if b == DEPOT else b])

    def new_trucks(self) -> list[Truck]:
        return [Truck(truck_id=k + 1, capacity=self.truck_capacity) for k in range(self.n_trucks)]

    def inventory(self) -> list[int]:
        return [s.bikes for s in self.stations]

    def clone(self) -> "BikeSharingSystem":
        return copy.deepcopy(self)
