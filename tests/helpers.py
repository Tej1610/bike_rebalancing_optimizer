"""Helpers to build small hand-made systems for tests."""
from __future__ import annotations

import pandas as pd

from models.system import BikeSharingSystem
from services.data_manager import build_system


def make_system(bikes, caps, demand=None, n_trucks=2, truck_capacity=5, coords=None):
    n = len(bikes)
    coords = coords or [(12.97 + 0.01 * (i % 4), 77.59 + 0.012 * (i // 4) + 0.003 * i) for i in range(n)]
    stations = pd.DataFrame({
        "station_id": [f"A{i}" for i in range(n)],
        "lat": [c[0] for c in coords],
        "lon": [c[1] for c in coords],
        "capacity": caps,
        "current_bikes": bikes,
    })
    dem = None
    if demand is not None:
        dem = pd.DataFrame({"station_id": stations.station_id, "predicted_demand": demand})
    return build_system(stations, dem, n_trucks, truck_capacity)


def set_targets(system: BikeSharingSystem, targets):
    """Override targets (must sum to total bikes for a conserving scenario)."""
    for s, t in zip(system.stations, targets):
        s.target = t
    return system
