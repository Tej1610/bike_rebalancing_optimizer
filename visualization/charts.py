"""Plotly figures for the dashboard (ResultsVisualizer)."""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.system import DEPOT, BikeSharingSystem
from services.simulation import RunResult

STATUS_COLORS = {"surplus": "#2a6fdb", "shortage": "#e8710a", "balanced": "#8a8f98"}
ALGO_COLORS = {"Greedy": "#2a6fdb", "Dynamic Programming": "#1b9e77"}
TRUCK_COLORS = ["#d62728", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#ff7f0e", "#1f77b4", "#7f7f7f"]


class ResultsVisualizer:
    @staticmethod
    def station_map(system: BikeSharingSystem, result: RunResult | None = None) -> go.Figure:
        """Stations coloured by status (before rebalancing) with each truck's route."""
        fig = go.Figure()

        if result is not None and result.status == "ok":
            for t in result.trucks:
                if not t.used:
                    continue
                pts = [(system.depot_lon, system.depot_lat)]
                for stop in t.route:
                    st = system.stations[stop.station]
                    p = (st.lon, st.lat)
                    if p != pts[-1]:
                        pts.append(p)
                pts.append((system.depot_lon, system.depot_lat))
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                    line=dict(color=TRUCK_COLORS[(t.truck_id - 1) % len(TRUCK_COLORS)], width=2.5),
                    name=f"Truck {t.truck_id} ({t.distance_km:.1f} km)", hoverinfo="name",
                ))

        for status, color in STATUS_COLORS.items():
            sts = [s for s in system.stations if s.status == status]
            if not sts:
                continue
            fig.add_trace(go.Scatter(
                x=[s.lon for s in sts], y=[s.lat for s in sts], mode="markers+text",
                text=[s.station_id for s in sts], textposition="top center",
                marker=dict(color=color, size=[min(10 + 2.5 * s.imbalance, 34) for s in sts],
                            line=dict(color="white", width=1)),
                customdata=[[s.name, s.bikes, s.target, s.capacity] for s in sts],
                hovertemplate="<b>%{customdata[0]}</b><br>bikes %{customdata[1]} / target %{customdata[2]}"
                              " / capacity %{customdata[3]}<extra></extra>",
                name=status.capitalize(),
            ))
        fig.add_trace(go.Scatter(
            x=[system.depot_lon], y=[system.depot_lat], mode="markers+text", text=["Depot"],
            textposition="bottom center",
            marker=dict(symbol="star", size=18, color="#222222", line=dict(color="white", width=1)),
            name="Depot", hoverinfo="name",
        ))
        fig.update_layout(
            height=560, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Longitude", showgrid=True, zeroline=False),
            yaxis=dict(title="Latitude", showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1.0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            title="Station status before rebalancing (marker size = imbalance)",
        )
        return fig

    @staticmethod
    def inventory_chart(system: BikeSharingSystem, results: list[RunResult]) -> go.Figure:
        ids = [s.station_id for s in system.stations]
        fig = go.Figure()
        fig.add_bar(x=ids, y=[s.bikes for s in system.stations], name="Current", marker_color="#8a8f98")
        fig.add_bar(x=ids, y=[s.target for s in system.stations], name="Target", marker_color="#222222")
        for r in results:
            if r.status == "ok":
                fig.add_bar(x=ids, y=r.inventory_after, name=f"After: {r.algorithm}",
                            marker_color=ALGO_COLORS.get(r.algorithm))
        fig.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=40, b=10),
                          title="Bikes per station: current vs target vs after rebalancing",
                          yaxis_title="Bikes", xaxis_title="Station",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        return fig

    @staticmethod
    def comparison_chart(results: list[RunResult]) -> go.Figure:
        ok = [r for r in results if r.status == "ok"]
        panels = [
            ("Imbalance after (bikes)", lambda r: r.imbalance_after),
            ("Truck distance (km)", lambda r: r.distance_km),
            ("Objective score (lower = better)", lambda r: r.score),
            ("Optimize time (s)", lambda r: r.optimize_time_s),
        ]
        fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
        for c, (_, fn) in enumerate(panels, start=1):
            for r in ok:
                fig.add_bar(x=[r.algorithm], y=[fn(r)], name=r.algorithm, showlegend=(c == 1),
                            marker_color=ALGO_COLORS.get(r.algorithm), row=1, col=c,
                            text=[f"{fn(r):.3g}"], textposition="outside")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.15))
        return fig

    @staticmethod
    def batch_chart(rows: list[dict], metric: str) -> go.Figure:
        fig = go.Figure()
        for alg in sorted({r["Algorithm"] for r in rows}):
            sub = [r for r in rows if r["Algorithm"] == alg and r["Status"] == "ok"]
            fig.add_bar(x=[f"seed {r['Seed']}" for r in sub], y=[r[metric] for r in sub], name=alg,
                        marker_color=ALGO_COLORS.get(alg))
        fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=40, b=10),
                          title=f"{metric} per scenario", yaxis_title=metric,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        return fig
