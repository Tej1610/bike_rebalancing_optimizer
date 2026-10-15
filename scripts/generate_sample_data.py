"""Regenerate data/stations.csv and data/demand.csv.

Usage:  python scripts/generate_sample_data.py [n_stations] [seed]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.data_manager import DATA_DIR, generate_stations_frame  # noqa: E402

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    stations, demand = generate_stations_frame(n, seed)
    DATA_DIR.mkdir(exist_ok=True)
    stations.to_csv(DATA_DIR / "stations.csv", index=False)
    demand.to_csv(DATA_DIR / "demand.csv", index=False)
    print(f"Wrote {n} stations to {DATA_DIR}")
