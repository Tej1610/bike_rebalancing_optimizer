"""Streamlit dashboard.  Run with:  streamlit run app.py"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from models.truck import PICKUP
from services.data_manager import build_system, generate_system, load_csv_frames
from services.objective import ObjectiveWeights
from services.parallel_runner import ParallelRunner
from services.simulation import ALGORITHMS, SimulationEngine
from visualization.charts import ResultsVisualizer

st.set_page_config(page_title="Bike Rebalancing Optimizer", page_icon="🚲", layout="wide")

_V = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
_STRETCH = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}


def show_plot(fig, key: str) -> None:
    st.plotly_chart(fig, key=key, **_STRETCH)


# --------------------------------------------------------------------- sidebar
st.title("🚲 Bike-Sharing Station Rebalancing Optimizer")
st.caption("Plan truck routes that move bikes from surplus stations to shortage stations. "
           "Compare Greedy vs Dynamic Programming on identical scenarios.")

with st.sidebar.form("config"):
    st.header("Configuration")
    source = st.radio("Dataset", ["Synthetic", "Sample CSV (data/)", "Upload CSV"])
    n_stations = st.slider("Stations (synthetic)", 4, 60, 10,
                           help="Dynamic Programming is exact but limited to smaller problems (~up to 15).")
    seed = st.number_input("Random seed", 0, 10_000_000, 42)
    n_trucks = st.slider("Trucks", 1, 10, 3)
    capacity = st.slider("Truck capacity (bikes)", 2, 40, 10)
    algo_choice = st.selectbox("Algorithm", ["Compare Both", "Greedy", "Dynamic Programming"])
    balance = st.checkbox("Spread work across trucks", value=True,
                          help="Off = fewest km (often one truck does everything).")
    up_s = up_d = None
    if source == "Upload CSV":
        up_s = st.file_uploader("stations.csv", type="csv",
                                help="columns: station_id, lat, lon, capacity, current_bikes [, name]")
        up_d = st.file_uploader("demand.csv (optional)", type="csv", help="columns: station_id, predicted_demand")
    with st.expander("Objective weights"):
        st.caption("score = α·distance(km) + β·remaining imbalance(bikes) + γ·trucks used")
        alpha = st.number_input("α  per km", 0.0, 1000.0, 1.0)
        beta = st.number_input("β  per bike out of place", 0.0, 1000.0, 2.0)
        gamma = st.number_input("γ  per truck used", 0.0, 1000.0, 5.0)
    submitted = st.form_submit_button("Run optimization", type="primary")

weights = ObjectiveWeights(alpha, beta, gamma)
names = list(ALGORITHMS) if algo_choice == "Compare Both" else [algo_choice]


def build_scenario():
    if source == "Synthetic":
        return generate_system(n_stations, n_trucks, capacity, int(seed))
    if source == "Sample CSV (data/)":
        s, d = load_csv_frames()
        return build_system(s, d, n_trucks, capacity)
    if up_s is None:
        raise ValueError("Upload a stations.csv file first (or pick another dataset).")
    return build_system(pd.read_csv(up_s), pd.read_csv(up_d) if up_d is not None else None, n_trucks, capacity)


if submitted or "run" not in st.session_state:
    try:
        system = build_scenario()
        with st.spinner("Optimizing..."):
            results = SimulationEngine(weights, balance).compare(system, names)
        st.session_state["run"] = {"system": system, "results": results}
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        st.session_state.pop("run", None)
        st.error(f"Could not build the scenario: {exc}")

run = st.session_state.get("run")
if not run:
    st.stop()
system, results = run["system"], run["results"]
ok = [r for r in results if r.status == "ok"]
for r in results:
    if r.status != "ok":
        st.warning(f"{r.algorithm} was skipped: {r.message}")

tab_map, tab_inv, tab_cmp, tab_batch = st.tabs(
    ["🗺️ Map & routes", "📦 Inventory", "⚖️ Comparison", "⚙️ Parallel scenario batch"])

# ------------------------------------------------------------------ map & routes
with tab_map:
    if not ok:
        st.info("No successful run to display.")
    else:
        shown = st.radio("Show routes of", [r.algorithm for r in ok], horizontal=True) if len(ok) > 1 else ok[0].algorithm
        res = next(r for r in ok if r.algorithm == shown)
        c = st.columns(5)
        c[0].metric("Imbalance", f"{res.imbalance_after}", f"{res.imbalance_after - res.imbalance_before}",
                    delta_color="inverse")
        c[1].metric("Reduction", f"{res.reduction_pct:.0f}%")
        c[2].metric("Bikes moved", res.bikes_moved)
        c[3].metric("Truck distance", f"{res.distance_km:.1f} km")
        c[4].metric("Trucks used", f"{res.trucks_used} / {system.n_trucks}")
        show_plot(ResultsVisualizer.station_map(system, res), "map")

        st.subheader("Truck routes")
        summary, steps = [], []
        for t in res.trucks:
            if not t.used:
                continue
            summary.append({"Truck": t.truck_id, "Stops": len(t.route), "Bikes delivered": t.bikes_moved,
                            "Distance (km)": round(t.distance_km, 2)})
            for k, stop in enumerate(t.route, start=1):
                stn = system.stations[stop.station]
                steps.append({"Truck": t.truck_id, "Step": k, "Station": stn.station_id, "Name": stn.name,
                              "Action": "Pick up" if stop.action == PICKUP else "Deliver", "Bikes": stop.bikes})
        if summary:
            st.dataframe(pd.DataFrame(summary), hide_index=True)
            st.dataframe(pd.DataFrame(steps), hide_index=True)
        else:
            st.success("Nothing to move: every station is already at its target.")

# --------------------------------------------------------------------- inventory
with tab_inv:
    rows = []
    for s in system.stations:
        row = {"Station": s.station_id, "Name": s.name, "Capacity": s.capacity, "Predicted demand": s.predicted_demand,
               "Current": s.bikes, "Target": s.target, "Status": s.status, "Imbalance": s.imbalance}
        for r in ok:
            row[f"After ({r.algorithm})"] = r.inventory_after[s.index]
        rows.append(row)
    inv_df = pd.DataFrame(rows)
    show_plot(ResultsVisualizer.inventory_chart(system, results), "inv")
    st.dataframe(inv_df, hide_index=True)
    st.download_button("Download station table (CSV)", inv_df.to_csv(index=False), "stations_result.csv", "text/csv")

# -------------------------------------------------------------------- comparison
with tab_cmp:
    if ok:
        table = pd.DataFrame([r.metrics() for r in results]).set_index("Algorithm").T
        st.dataframe(table)
        st.download_button("Download metrics (CSV)", table.to_csv(), "metrics.csv", "text/csv")
        if len(ok) > 1:
            show_plot(ResultsVisualizer.comparison_chart(results), "cmp")
            g, d = ok[0], ok[1]
            st.markdown(
                f"**Same scenario, same objective.** Dynamic Programming minimizes the simplified transfer cost "
                f"Σ dᵢⱼ·xᵢⱼ exactly ({d.transfer_cost:.1f} vs {g.transfer_cost:.1f} bike-km for Greedy). "
                "Real truck distance also depends on stop order and truck count, so the route distance of the "
                "DP plan is not guaranteed to be shorter.")
    else:
        st.info("No successful run to compare.")

# ------------------------------------------------------------------------- batch
with tab_batch:
    st.write("Run many independent synthetic scenarios (seeds *seed, seed+1, …*) in parallel worker processes, "
             "using the stations / trucks / capacity / weights from the sidebar.")
    cpu = os.cpu_count() or 2
    b1, b2, b3 = st.columns(3)
    n_scen = b1.slider("Scenarios", 2, 40, 12)
    workers = b2.slider("Worker processes", 1, max(cpu, 2), min(max(cpu - 1, 1), 4))
    batch_algos = b3.multiselect("Algorithms", list(ALGORITHMS), default=names)
    bench = st.checkbox("Also run with 1 worker to measure speed-up", value=True)
    if st.button("Run batch"):
        if not batch_algos:
            st.error("Pick at least one algorithm.")
        else:
            specs = ParallelRunner.make_specs(n_stations, n_trucks, capacity, int(seed), n_scen, batch_algos, weights)
            with st.spinner(f"Running {len(specs)} runs on {workers} worker(s)..."):
                rows_p, t_par = ParallelRunner(workers).run(specs)
                t_ser = ParallelRunner(1).run(specs)[1] if bench and workers > 1 else None
            st.session_state["batch"] = {"rows": rows_p, "t_par": t_par, "t_ser": t_ser, "workers": workers}
    b = st.session_state.get("batch")
    if b:
        msg = f"{len(b['rows'])} runs finished in **{b['t_par']:.2f} s** with {b['workers']} worker(s)."
        if b["t_ser"]:
            msg += (f" Single process: {b['t_ser']:.2f} s → speed-up ×{b['t_ser'] / b['t_par']:.2f}. "
                    "(Small workloads can be slower in parallel because of process start-up cost.)")
        st.info(msg)
        df = pd.DataFrame(b["rows"])
        okdf = df[df["Status"] == "ok"]
        if okdf.empty:
            st.warning("All runs were skipped.")
        else:
            agg = okdf.groupby("Algorithm")[["Imbalance after", "Truck distance (km)", "Transfer cost (bike-km)",
                                             "Objective score", "Optimize time (s)"]].mean().round(3)
            st.subheader("Average over scenarios")
            st.dataframe(agg)
            metric = st.selectbox("Chart metric", ["Objective score", "Truck distance (km)",
                                                   "Transfer cost (bike-km)", "Optimize time (s)"])
            show_plot(ResultsVisualizer.batch_chart(b["rows"], metric), "batch")
        skipped = df[df["Status"] != "ok"]
        if len(skipped):
            st.warning(f"{len(skipped)} run(s) skipped (problem too large for the DP budget).")
        st.download_button("Download batch results (CSV)", df.to_csv(index=False), "batch_results.csv", "text/csv")
