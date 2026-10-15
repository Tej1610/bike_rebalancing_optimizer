"""Exact Dynamic Programming optimizer.

Formulation (transportation problem solved by DP)
-------------------------------------------------
* Surplus stations i = 1..m with supplies s_i, shortage stations j = 1..k with needs n_j.
* Stages go over one side of the problem (stage "rows"); the state stores the
  remaining amounts of the other side ("columns"). The side whose state space
  prod(amount + 1) is smaller is used as the state, which keeps the DP small.
* Inside a stage the decision is made column by column (a knapsack-style layer):
      state  = (row j, column i, bikes of row j still to place, remaining column amounts)
      value  = min over q of  d_ij * q + value(next state)
* The memo table stores the best value of each state, so every sub-problem is solved once.

This minimizes exactly the simplified transfer cost  sum d_ij * x_ij  (bike-km) while
fully meeting every shortage, so it is optimal for that model. It is not a full
vehicle-routing optimum: actual truck distance depends on stop order and truck
count, which are handled afterwards by the RoutePlanner for both algorithms.

The number of states still grows quickly with the amounts, so a work budget is
enforced and ProblemTooLargeError is raised when it is exceeded.
"""
from __future__ import annotations

import math

from models.system import BikeSharingSystem
from services.demand_analyzer import DemandAnalyzer

from .base import Optimizer, ProblemTooLargeError, Transfer


def solve_transport(
    rows: list[int], cols: list[int], cost: list[list[float]], max_work: int
) -> tuple[list[list[int]], int]:
    """Min-cost transport: sum_c x[r][c] == rows[r], sum_r x[r][c] <= cols[c].

    Returns (x matrix, number of memoised states). Raises ProblemTooLargeError.
    """
    k, m = len(rows), len(cols)
    memo: dict[tuple, tuple[float, int]] = {}
    work = 0
    inf = float("inf")

    def solve(r: int, c: int, left: int, rem: tuple[int, ...]) -> float:
        nonlocal work
        if r == k:
            return 0.0
        if c == m:
            if left != 0:
                return inf
            return 0.0 if r + 1 == k else solve(r + 1, 0, rows[r + 1], rem)
        key = (r, c, left, rem)
        hit = memo.get(key)
        if hit is not None:
            return hit[0]
        later = sum(rem[c + 1:])
        best, best_q = inf, -1
        for q in range(max(0, left - later), min(rem[c], left) + 1):
            work += 1
            if work > max_work:
                raise ProblemTooLargeError(
                    f"DP exceeded its work budget ({max_work:,} transitions); "
                    "use fewer stations or choose Greedy for this size."
                )
            nxt = rem[:c] + (rem[c] - q,) + rem[c + 1:]
            total = cost[r][c] * q + solve(r, c + 1, left - q, nxt)
            if total < best - 1e-12:
                best, best_q = total, q
        memo[key] = (best, best_q)
        return best

    if k == 0:
        return [], 0
    solve(0, 0, rows[0], tuple(cols))

    x = [[0] * m for _ in range(k)]
    rem = tuple(cols)
    for r in range(k):
        left = rows[r]
        for c in range(m):
            q = memo[(r, c, left, rem)][1] if (r, c, left, rem) in memo else 0
            if q > 0:
                x[r][c] = q
                left -= q
                rem = rem[:c] + (rem[c] - q,) + rem[c + 1:]
    return x, len(memo)


class DynamicProgrammingOptimizer(Optimizer):
    name = "Dynamic Programming"

    def __init__(self, max_work: int = 3_000_000) -> None:
        self.max_work = max_work
        self.last_states = 0

    def plan(self, system: BikeSharingSystem) -> list[Transfer]:
        report = DemandAnalyzer.analyze(system)
        src_ids = sorted(report.surplus)
        dst_ids = sorted(report.shortage)
        supplies = [report.surplus[i] for i in src_ids]
        needs = [report.shortage[j] for j in dst_ids]
        if not src_ids or not dst_ids:
            return []

        # Targets conserve bikes so supply == demand; trim defensively otherwise.
        excess = sum(needs) - sum(supplies)
        while excess > 0:
            j = max(range(len(needs)), key=lambda q: needs[q])
            cut = min(needs[j], excess)
            needs[j] -= cut
            excess -= cut
        keep = [q for q, n in enumerate(needs) if n > 0]
        dst_ids = [dst_ids[q] for q in keep]
        needs = [needs[q] for q in keep]
        if not dst_ids:
            return []

        d = [[system.distance(i, j) for i in src_ids] for j in dst_ids]  # d[j][i]
        space_sup = math.prod(s + 1 for s in supplies)
        space_need = math.prod(n + 1 for n in needs)

        transfers: list[Transfer] = []
        balanced = sum(supplies) == sum(needs)  # always true for conserved targets
        if space_sup <= space_need or not balanced:
            # stages = shortage stations, state = remaining supplies
            x, self.last_states = solve_transport(needs, supplies, d, self.max_work)
            for jj, row in enumerate(x):
                for ii, q in enumerate(row):
                    if q:
                        transfers.append(Transfer(src_ids[ii], dst_ids[jj], q))
        else:
            # balanced problem: stages = surplus stations, state = remaining needs
            dt = [[d[j][i] for j in range(len(dst_ids))] for i in range(len(src_ids))]  # dt[i][j]
            x, self.last_states = solve_transport(supplies, needs, dt, self.max_work)
            for ii, row in enumerate(x):
                for jj, q in enumerate(row):
                    if q:
                        transfers.append(Transfer(src_ids[ii], dst_ids[jj], q))
        return transfers
