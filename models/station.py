"""Station model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Station:
    """A bike-sharing station.

    ``index`` is the station's position in the system (used by all algorithms),
    ``station_id`` is the external identifier (e.g. from a CSV file).
    """

    index: int
    station_id: str
    name: str
    lat: float
    lon: float
    capacity: int
    bikes: int
    predicted_demand: float = 1.0
    target: int = 0

    @property
    def surplus(self) -> int:
        return max(self.bikes - self.target, 0)

    @property
    def shortage(self) -> int:
        return max(self.target - self.bikes, 0)

    @property
    def imbalance(self) -> int:
        return abs(self.bikes - self.target)

    @property
    def status(self) -> str:
        if self.bikes > self.target:
            return "surplus"
        if self.bikes < self.target:
            return "shortage"
        return "balanced"

    def validate(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"{self.station_id}: negative capacity")
        if not 0 <= self.bikes <= self.capacity:
            raise ValueError(
                f"{self.station_id}: bikes ({self.bikes}) must be within 0..capacity ({self.capacity})"
            )
