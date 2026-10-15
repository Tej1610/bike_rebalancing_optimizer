"""Load or generate scenarios (stations, demand, depot, trucks)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.station import Station
from models.system import BikeSharingSystem
from services.demand_analyzer import DemandAnalyzer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CITY_CENTER = (12.9716, 77.5946)  # synthetic city centre (lat, lon)


def generate_stations_frame(n_stations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic stations.csv-like and demand.csv-like frames (deterministic per seed)."""
    rng = np.random.default_rng(seed)
    # ~ +-4 km box around the centre; 1 deg lat ~ 111 km
    lat = CITY_CENTER[0] + rng.uniform(-0.036, 0.036, n_stations)
    lon = CITY_CENTER[1] + rng.uniform(-0.037, 0.037, n_stations)
    capacity = rng.integers(15, 31, n_stations)
    # start with roughly half-full stations but randomly skewed -> real imbalances
    fill = np.clip(rng.beta(1.2, 1.2, n_stations), 0.0, 1.0)
    bikes = np.minimum(np.rint(fill * capacity).astype(int), capacity)
    # a few hot-spots with much higher predicted demand
    demand = rng.uniform(0.3, 1.0, n_stations)
    hot = rng.random(n_stations) < 0.25
    demand[hot] *= rng.uniform(2.0, 3.5, hot.sum())
    ids = [f"S{i + 1:02d}" for i in range(n_stations)]
    stations = pd.DataFrame(
        {
            "station_id": ids,
            "name": [f"Station {i + 1}" for i in range(n_stations)],
            "lat": lat.round(6),
            "lon": lon.round(6),
            "capacity": capacity,
            "current_bikes": bikes,
        }
    )
    demand_df = pd.DataFrame({"station_id": ids, "predicted_demand": demand.round(3)})
    return stations, demand_df


def build_system(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame | None,
    n_trucks: int,
    truck_capacity: int,
) -> BikeSharingSystem:
    """Create a ready-to-optimise system (targets already computed)."""
    required = {"station_id", "lat", "lon", "capacity", "current_bikes"}
    missing = required - set(stations_df.columns)
    if missing:
        raise ValueError(f"stations data is missing columns: {sorted(missing)}")
    if stations_df.empty:
        raise ValueError("stations data is empty")
    df = stations_df.copy()
    df["station_id"] = df["station_id"].astype(str)
    if df["station_id"].duplicated().any():
        raise ValueError("duplicate station_id values")
    if "name" not in df.columns:
        df["name"] = "Station " + df["station_id"]
    if demand_df is not None and "predicted_demand" in demand_df.columns:
        d = demand_df.copy()
        d["station_id"] = d["station_id"].astype(str)
        df = df.merge(d[["station_id", "predicted_demand"]], on="station_id", how="left")
        df["predicted_demand"] = df["predicted_demand"].fillna(df["predicted_demand"].mean())
    elif "predicted_demand" not in df.columns:
        df["predicted_demand"] = 1.0
    df["predicted_demand"] = df["predicted_demand"].fillna(1.0)

    stations = [
        Station(
            index=i,
            station_id=str(r.station_id),
            name=str(r.name),
            lat=float(r.lat),
            lon=float(r.lon),
            capacity=int(r.capacity),
            bikes=int(r.current_bikes),
            predicted_demand=float(r.predicted_demand),
        )
        for i, r in enumerate(df.itertuples(index=False))
    ]
    system = BikeSharingSystem(
        stations=stations,
        n_trucks=int(n_trucks),
        truck_capacity=int(truck_capacity),
        depot_lat=float(df["lat"].mean()),
        depot_lon=float(df["lon"].mean()),
    )
    DemandAnalyzer().apply_targets(system)
    return system


def generate_system(n_stations: int, n_trucks: int, truck_capacity: int, seed: int) -> BikeSharingSystem:
    s, d = generate_stations_frame(n_stations, seed)
    return build_system(s, d, n_trucks, truck_capacity)


def load_csv_frames(
    stations_path: str | Path | None = None, demand_path: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    stations_path = Path(stations_path or DATA_DIR / "stations.csv")
    demand_path = Path(demand_path or DATA_DIR / "demand.csv")
    stations = pd.read_csv(stations_path)
    demand = pd.read_csv(demand_path) if demand_path.exists() else None
    return stations, demand


def load_system_from_csv(
    n_trucks: int,
    truck_capacity: int,
    stations_path: str | Path | None = None,
    demand_path: str | Path | None = None,
) -> BikeSharingSystem:
    s, d = load_csv_frames(stations_path, demand_path)
    return build_system(s, d, n_trucks, truck_capacity)
