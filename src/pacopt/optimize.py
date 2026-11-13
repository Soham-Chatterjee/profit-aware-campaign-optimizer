"""Engine 3: Optimization Intelligence. Which combination of customers should be targeted (Q4)?

Problem (plan, Milestone 4):

    maximise   sum_i x_i * EP_i                      x_i in {0, 1}, eligible customers only
    subject to sum_i x_i * C_i            <= budget
               sum_i x_i                  <= max_contacts
               sum_{i in channel g} x_i   <= capacity_g
               min_s <= share of segment s in the list <= max_s
               share of protected (75+) customers      <= protected_max

Strategies, all checked by the same `check_constraints()`:
    Random (50 seeds), Propensity (highest P first), Greedy (highest EP/C first),
    Rule-based (highest balance first), Genetic Algorithm, and the exact MILP optimum.

Every heuristic strategy is a *priority order* fed through one shared constraint engine
(`_Engine`): take customers in order, skip any who would break a limit, then repair the
segment shares. The GA uses the same engine as its repair operator (D-036).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from pacopt import config as cfg

EPS = 1e-9


# ============================================================================= problem
@dataclass
class Problem:
    """Eligible customers as arrays (row i = customer i) plus the constraint set."""
    table_index: np.ndarray
    customer_id: np.ndarray
    ep: np.ndarray
    cost: np.ndarray
    propensity: np.ndarray
    balance: np.ndarray
    group: np.ndarray          # int code into `groups`
    groups: tuple
    segment: np.ndarray        # int code into `segments`
    segments: tuple
    protected: np.ndarray      # bool: aged 75+
    cc: cfg.CampaignConstraints = field(default_factory=cfg.CampaignConstraints)

    @property
    def n(self) -> int:
        return len(self.ep)

    def subset(self, mask: np.ndarray) -> "Problem":
        m = np.asarray(mask, dtype=bool)
        arrays = {k: getattr(self, k)[m] for k in ("table_index", "customer_id", "ep", "cost", "propensity",
                                                    "balance", "group", "segment", "protected")}
        return Problem(**arrays, groups=self.groups, segments=self.segments, cc=self.cc)

    # constraint vectors aligned with groups / segments
    @property
    def capacity(self) -> np.ndarray:
        return np.array([self.cc.channel_capacity.get(g, np.inf) for g in self.groups], dtype=float)

    @property
    def seg_max(self) -> np.ndarray:
        return np.array([self.cc.segment_max_share.get(s, 1.0) for s in self.segments], dtype=float)

    @property
    def seg_min(self) -> np.ndarray:
        return np.array([self.cc.segment_min_share.get(s, 0.0) for s in self.segments], dtype=float)


def build_problem(table: pd.DataFrame, cc: cfg.CampaignConstraints = cfg.CONSTRAINTS) -> Problem:
    """Eligible customers only: ineligible customers never enter the chromosome (D-007)."""
    e = table[table["eligible"] == 1]
    groups = tuple(dict.fromkeys(list(cc.channel_capacity) + sorted(e["channel_group"].unique())))
    segments = tuple(cfg.SEGMENTS)
    return Problem(
        table_index=e.index.to_numpy(),
        customer_id=e["customer_id"].to_numpy(),
        ep=e["expected_profit"].to_numpy(dtype=float),
        cost=e["contact_cost"].to_numpy(dtype=float),
        propensity=e["propensity"].to_numpy(dtype=float),
        balance=e["balance"].to_numpy(dtype=float),
        group=e["channel_group"].map({g: k for k, g in enumerate(groups)}).to_numpy(dtype=int),
        groups=groups,
        segment=e["segment"].astype(str).map({s: k for k, s in enumerate(segments)}).to_numpy(dtype=int),
        segments=segments,
        protected=e["flag_age_75plus"].to_numpy(dtype=bool),
        cc=cc,
    )


def fingerprint(table: pd.DataFrame, cc: cfg.CampaignConstraints = cfg.CONSTRAINTS) -> str:
    """Identifies an optimisation problem: the eligible customers' EP, cost, channel, segment and
    protected flag, plus the constraints. Saved GA/MILP results are reused only when this matches."""
    pr = build_problem(table, cc)
    h = hashlib.sha256("|".join(pr.customer_id.astype(str)).encode())
    for a in (np.round(pr.ep, 6), pr.cost, pr.group, pr.segment, pr.protected):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps(asdict(cc), sort_keys=True).encode())
    return h.hexdigest()[:16]


# ============================================================================= shared constraint engine
def _top(mask: np.ndarray, limit, order: np.ndarray) -> np.ndarray:
    """Per row, keep the first `limit` True entries of `mask` when read in `order`."""
    m = mask[:, order]
    keep = m & (np.cumsum(m, axis=1, dtype=np.int32) <= np.asarray(limit, dtype=float).reshape(-1, 1))
    out = np.zeros_like(mask)
    out[:, order] = keep
    return out


class _Engine:
    """Vectorised constraint handling for a population X (rows = solutions, columns = customers).

    budget_order: priority used where money is the limit (fill, budget repair).
    count_order:  priority used where a head-count is the limit (contacts, channel, segment shares).
    pool:         customers this strategy may select at all (any of them may be added to meet a
                  segment minimum).
    fill_pool:    customers used to top up spare capacity (Greedy and GA: positive EP only).
    Lower position in an order = more valuable = kept first / added first.
    """

    def __init__(self, pr: Problem, budget_order: np.ndarray, count_order: np.ndarray | None = None,
                 pool: np.ndarray | None = None, fill_pool: np.ndarray | None = None):
        self.pr, self.cc = pr, pr.cc
        self.ob = np.asarray(budget_order)
        self.oc = self.ob if count_order is None else np.asarray(count_order)
        self.pool = np.ones(pr.n, bool) if pool is None else np.asarray(pool, bool)
        self.fill_pool = self.pool if fill_pool is None else (np.asarray(fill_pool, bool) & self.pool)
        self.G = np.stack([pr.group == k for k in range(len(pr.groups))])
        self.S = np.stack([pr.segment == k for k in range(len(pr.segments))])
        self.cap, self.smax, self.smin = pr.capacity, pr.seg_max, pr.seg_min
        self.min_mask = self.S[self.smin > 0].any(axis=0) if (self.smin > 0).any() else np.zeros(pr.n, bool)
        self.penalty = 10.0 * max(float(np.abs(pr.ep).max(initial=1.0)), 1.0)

    # ------------------------------------------------------------------ hard limits
    def caps(self, X: np.ndarray, lock: np.ndarray | None = None) -> np.ndarray:
        """Channel capacity, max contacts and budget. Drops the least valuable unlocked selections."""
        cost, cc = self.pr.cost, self.cc
        lock = np.zeros_like(X) if lock is None else (lock & X)
        for k, g in enumerate(self.G):
            if not np.isfinite(self.cap[k]):
                continue
            sel = X & g
            lk = sel & lock
            X = (X & ~g) | lk | _top(sel & ~lock, self.cap[k] - lk.sum(1), self.oc)
        lk = X & lock
        X = lk | _top(X & ~lock, cc.max_contacts - lk.sum(1), self.oc)
        lk = X & lock
        free = (X & ~lock)[:, self.ob]
        room = cc.budget - (lk * cost).sum(1)
        keep = free & (np.cumsum(free * cost[self.ob], axis=1) <= room[:, None] + EPS)
        out = np.zeros_like(X)
        out[:, self.ob] = keep
        return lk | out

    # ------------------------------------------------------------------ segment shares
    def _drop_to_share(self, X: np.ndarray, member: np.ndarray, max_share: float) -> np.ndarray:
        N = X.sum(1)
        ns = (X & member).sum(1)
        drop = np.maximum(np.ceil((ns - max_share * N) / (1 - max_share) - EPS), 0)
        if drop.any():
            X = (X & ~member) | _top(X & member, ns - drop, self.oc)
        return X

    def max_shares(self, X: np.ndarray) -> np.ndarray:
        for _ in range(2):
            for k in np.flatnonzero(self.smax < 1):
                X = self._drop_to_share(X, self.S[k], self.smax[k])
            if self.cc.protected_max_share < 1:
                X = self._drop_to_share(X, self.pr.protected, self.cc.protected_max_share)
        return X

    def min_shares(self, X: np.ndarray) -> np.ndarray:
        """Add the best unselected members of under-represented segments (from the whole pool, so
        a negative-EP member can be added when a minimum requires it), restore the hard limits by
        dropping only from segments without a minimum, and, if a segment has no members left to
        add, shrink the rest of the list until its minimum share holds."""
        if not (self.smin > 0).any():
            return X
        ks = np.flatnonzero(self.smin > 0)
        # protected (75+) customers are never added just to meet a segment minimum
        addable = self.pool & ~self.pr.protected if self.cc.protected_max_share < 1 else self.pool
        for _ in range(3):
            changed = False
            for k in ks:
                N, ns = X.sum(1), (X & self.S[k]).sum(1)
                need = np.maximum(np.ceil((self.smin[k] * N - ns) / (1 - self.smin[k]) - EPS), 0)
                if need.any():
                    X = X | _top(addable & self.S[k] & ~X, need, self.oc)
                    changed = True
            if not changed:
                break
            X = self.caps(X, lock=X & self.min_mask)
        # members exhausted: largest list size each minimum still allows, then trim the others
        allowed = np.min([np.floor((X & self.S[k]).sum(1) / self.smin[k] + EPS) for k in ks], axis=0)
        excess = X.sum(1) - allowed
        if (excess > 0).any():
            rows = excess > 0
            Xr, al = X[rows], allowed[rows]
            other = Xr & ~self.min_mask
            caps = [(self.S[k], self.smax[k]) for k in np.flatnonzero(self.smax < 1) if self.smin[k] == 0]
            if self.cc.protected_max_share < 1:
                caps.append((self.pr.protected, self.cc.protected_max_share))
            for m, share in caps:            # max shares measured against the reduced list size
                other = (other & ~m) | _top(other & m, np.floor(share * al + EPS), self.oc)
            keep = np.maximum(al - (Xr & self.min_mask).sum(1), 0)
            X[rows] = (Xr & self.min_mask) | _top(other, keep, self.oc)
        return X

    # ------------------------------------------------------------------ fill unused capacity
    @property
    def n_ref(self) -> float:
        """Expected final list size: the reference for share caps while a list is still being filled
        (from an empty list, the first customer would otherwise be '100% of the list')."""
        cc, pool_cost = self.cc, np.sort(self.pr.cost[self.fill_pool])
        by_budget = int((np.cumsum(pool_cost) <= cc.budget + EPS).sum())
        return float(max(min(cc.max_contacts, by_budget, self.fill_pool.sum()), 1))

    def _blocked(self, X: np.ndarray) -> np.ndarray:
        """Customers whose segment (or protected group) is already at its maximum share of the
        larger of (current list + 1) and the expected final size; max_shares() trims any overshoot."""
        N = np.maximum(X.sum(1) + 1, self.n_ref)
        block = np.zeros_like(X)
        members = [(self.S[k], self.smax[k]) for k in np.flatnonzero(self.smax < 1)]
        if self.cc.protected_max_share < 1:
            members.append((self.pr.protected, self.cc.protected_max_share))
        for m, share in members:
            full = (X & m).sum(1) + 1 > share * N + EPS
            block |= m[None, :] & full[:, None]
        return block

    def fill(self, X: np.ndarray) -> np.ndarray:
        """Add customers in budget priority, skipping any who would break a limit (exact skip
        semantics: a bulk prefix, then single additions while anything still fits)."""
        cost, cc = self.pr.cost, self.cc
        cand = self.fill_pool & ~X & ~self._blocked(X)
        add = np.zeros_like(X)
        for k, g in enumerate(self.G):
            room = self.cap[k] - (X & g).sum(1) if np.isfinite(self.cap[k]) else np.full(len(X), np.inf)
            add |= _top(cand & g, room, self.ob)
        nref = self.n_ref
        members = [(self.S[k], self.smax[k]) for k in np.flatnonzero(self.smax < 1)]
        if cc.protected_max_share < 1:
            members.append((self.pr.protected, cc.protected_max_share))
        for m, share in members:                     # bulk step respects share caps of the final size
            room = np.floor(share * nref + EPS) - (X & m).sum(1)
            add = (add & ~m) | _top(add & m, np.maximum(room, 0), self.ob)
        add = _top(add, cc.max_contacts - X.sum(1), self.ob)
        ao = add[:, self.ob]
        rem = cc.budget - (X * cost).sum(1)
        keep = ao & (np.cumsum(ao * cost[self.ob], axis=1) <= rem[:, None] + EPS)
        add = np.zeros_like(X)
        add[:, self.ob] = keep
        X = X | add
        # single additions while anything still fits (only rows with budget left for the cheapest)
        cost_o, cmin = cost[self.ob], cost[self.fill_pool].min(initial=np.inf)
        active = np.flatnonzero((cc.budget - (X * cost).sum(1) >= cmin - EPS) & (X.sum(1) < cc.max_contacts))
        for _ in range(256):
            if not len(active):
                break
            Xa = X[active]
            rem = cc.budget - (Xa * cost).sum(1)
            room_ok = np.ones_like(Xa)
            for k, g in enumerate(self.G):
                if np.isfinite(self.cap[k]):
                    full = (Xa & g).sum(1) >= self.cap[k]
                    room_ok &= ~(g[None, :] & full[:, None])
            ok = (self.fill_pool & ~Xa & ~self._blocked(Xa) & room_ok & (Xa.sum(1) < cc.max_contacts)[:, None])[:, self.ob]
            ok &= cost_o[None, :] <= rem[:, None] + EPS
            rows = ok.any(1)
            first = self.ob[np.argmax(ok, axis=1)]
            X[active[rows], first[rows]] = True
            active = active[rows]
        return X

    # ------------------------------------------------------------------ repair / fitness
    def repair(self, X: np.ndarray, fill: bool = True) -> np.ndarray:
        """Hard limits -> segment maxima -> segment minima -> fill spare capacity -> re-check shares.
        Not guaranteed feasible in every corner case; the fitness penalty is the fallback (D-011)."""
        X = np.atleast_2d(X).astype(bool) & self.pool
        X = self.min_shares(self.max_shares(self.caps(X)))
        if fill:
            X = self.min_shares(self.max_shares(self.fill(X)))
        for _ in range(4):                    # interacting share limits: iterate until all rows hold
            bad = self.violation(X) > 0
            if not bad.any():
                break
            X[bad] = self.min_shares(self.max_shares(self.caps(X[bad])))
        return X

    def violation(self, X: np.ndarray) -> np.ndarray:
        """Total constraint violation per row, in customer-equivalent units (0 = feasible)."""
        pr, cc = self.pr, self.cc
        N = X.sum(1).astype(float)
        v = np.maximum((X * pr.cost).sum(1) - cc.budget, 0) / pr.cost.min() + np.maximum(N - cc.max_contacts, 0)
        for k, g in enumerate(self.G):
            v += np.maximum((X & g).sum(1) - self.cap[k], 0)
        for k, s in enumerate(self.S):
            ns = (X & s).sum(1)
            v += np.maximum(ns - self.smax[k] * N, 0) + np.maximum(self.smin[k] * N - ns, 0)
        v += np.maximum((X & pr.protected).sum(1) - cc.protected_max_share * N, 0)
        return np.where(v < 1e-7, 0.0, v)

    def fitness(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = self.violation(X)
        return (X * self.pr.ep).sum(1) - self.penalty * v, v


# ============================================================================= heuristic strategies
def _orders(pr: Problem) -> dict[str, np.ndarray]:
    return {
        "ep_per_cost": np.argsort(-pr.ep / pr.cost, kind="stable"),
        "ep": np.argsort(-pr.ep, kind="stable"),
        "propensity": np.argsort(-pr.propensity, kind="stable"),
        "balance": np.argsort(-pr.balance, kind="stable"),
    }


def priority_selection(pr: Problem, order: np.ndarray, fill_pool: np.ndarray | None = None,
                       count_order: np.ndarray | None = None) -> np.ndarray:
    """Take customers in `order`, skip any who would break a limit, then repair the segment shares."""
    eng = _Engine(pr, order, count_order, fill_pool=fill_pool)
    return eng.repair(np.zeros((1, pr.n), bool))[0]


def greedy(pr: Problem) -> np.ndarray:
    """Highest EP per £ first; tops up with positive-EP customers only."""
    return priority_selection(pr, _orders(pr)["ep_per_cost"], fill_pool=pr.ep > 0)


def propensity_only(pr: Problem) -> np.ndarray:
    return priority_selection(pr, _orders(pr)["propensity"])


def rule_based(pr: Problem) -> np.ndarray:
    """A typical business rule: highest balance first."""
    return priority_selection(pr, _orders(pr)["balance"])


def random_selection(pr: Problem, seed: int) -> np.ndarray:
    """Random eligible customers until a limit is reached (proxy for an untargeted campaign)."""
    return priority_selection(pr, np.random.default_rng(seed).permutation(pr.n))


# ============================================================================= genetic algorithm
@dataclass
class GAResult:
    x: np.ndarray                  # bool over the Problem's customers
    expected_profit: float
    feasible: bool
    generations: int
    runtime_s: float
    history: pd.DataFrame          # generation, best, mean (feasible rows), feasible_share
    seeded: bool


def run_ga(pr: Problem, ga: cfg.GAConfig = cfg.GA, progress=None) -> GAResult:
    """Binary-chromosome GA (D-010, D-035, D-036): tournament selection, uniform crossover,
    asymmetric bit-flip mutation, elitism, and a repair operator that restores feasibility.

    Genes = eligible customers with positive EP, plus every member of a segment that has a minimum
    share (a negative-EP customer can only lower the objective unless a minimum needs it; the MILP
    over *all* eligible customers verifies nothing is lost). Spare capacity is filled with
    positive-EP customers only."""
    t0 = time.perf_counter()
    min_segs = [k for k, s in enumerate(pr.segments) if pr.cc.segment_min_share.get(s, 0) > 0]
    genes = (pr.ep > 0) | np.isin(pr.segment, min_segs)
    sub = pr.subset(genes)
    n, pop = sub.n, ga.population
    o = _orders(sub)
    pos = sub.ep > 0
    eng = _Engine(sub, o["ep_per_cost"], o["ep"], fill_pool=pos)   # money -> EP/C, head-count -> EP
    rng = np.random.default_rng(ga.random_state)

    target = min(sub.cc.max_contacts, sub.cc.budget / sub.cost.mean(), n)
    P = (rng.random((pop, n)) < target / max(pos.sum(), 1)) & pos   # density matched to the budget
    if ga.greedy_seeds:
        # the Greedy baseline itself plus perturbed copies, so any gain over Greedy is found by the GA
        g0 = _Engine(sub, o["ep_per_cost"], fill_pool=pos).repair(np.zeros((1, n), bool))[0]
        seeds = [g0] + [(g0 & ~(rng.random(n) < 0.02)) | ((rng.random(n) < 0.002) & pos)
                        for _ in range(min(ga.greedy_seeds, pop) - 1)]
        P[: len(seeds)] = np.stack(seeds)
    P = eng.repair(P, fill=ga.fill)
    fit, viol = eng.fitness(P)

    n_elite = max(1, int(round(ga.elite_frac * pop)))
    n_child = pop - n_elite
    best_x, best_f = None, -np.inf
    history, stale, g = [], 0, 0

    def track(gen):
        nonlocal best_x, best_f, stale
        feas = viol == 0
        improved = False
        if feas.any():
            i = int(np.argmax(np.where(feas, fit, -np.inf)))
            if fit[i] > best_f + 1e-6:
                best_f, best_x, improved = float(fit[i]), P[i].copy(), True
        stale = 0 if improved else stale + 1
        history.append({"generation": gen, "best": best_f if np.isfinite(best_f) else np.nan,
                        "mean": float(fit[feas].mean()) if feas.any() else np.nan,
                        "feasible_share": float(feas.mean())})

    track(0)
    for g in range(1, ga.generations + 1):
        elite = np.argsort(-fit, kind="stable")[:n_elite]
        cand = rng.integers(pop, size=(n_child, 2, ga.tournament_k))
        win = np.take_along_axis(cand, np.argmax(fit[cand], axis=2)[..., None], axis=2)[..., 0]
        p1, p2 = P[win[:, 0]], P[win[:, 1]]
        cross = (rng.random(n_child) < ga.crossover_rate)[:, None] & (rng.random((n_child, n)) < 0.5)
        child = np.where(cross, p2, p1)
        n_on = child.sum(1)
        n_free = np.maximum((pos & ~child).sum(1), 1)
        add = pos & ~child & (rng.random((n_child, n)) < (ga.mutation_adds / n_free)[:, None])
        drop = child & (rng.random((n_child, n)) < (ga.mutation_drops / np.maximum(n_on, 1))[:, None])
        child = eng.repair((child | add) & ~drop, fill=ga.fill)
        cf, cv = eng.fitness(child)
        P = np.vstack([P[elite], child])
        fit, viol = np.concatenate([fit[elite], cf]), np.concatenate([viol[elite], cv])
        track(g)
        if progress is not None:
            progress(g, best_f)
        if stale >= ga.patience:
            break
        if ga.time_limit_s is not None and time.perf_counter() - t0 > ga.time_limit_s:
            break

    x = np.zeros(pr.n, bool)
    feasible = best_x is not None
    if feasible:
        x[np.flatnonzero(genes)[best_x]] = True
    return GAResult(x=x, expected_profit=float(pr.ep[x].sum()), feasible=feasible, generations=g,
                    runtime_s=time.perf_counter() - t0, history=pd.DataFrame(history),
                    seeded=bool(ga.greedy_seeds))


# ============================================================================= exact optimum
@dataclass
class MILPResult:
    x: np.ndarray
    expected_profit: float
    status: int
    message: str
    mip_gap: float
    dual_bound: float
    runtime_s: float

    @property
    def optimal(self) -> bool:
        return self.status == 0


def milp_optimum(pr: Problem, oc: cfg.OptimConfig = cfg.OPTIM) -> MILPResult:
    """Exact optimum over ALL eligible customers with scipy/HiGHS (the reference ceiling)."""
    t0 = time.perf_counter()
    cc, n = pr.cc, pr.n
    rows, lo, hi = [pr.cost, np.ones(n)], [0.0, 0.0], [cc.budget, cc.max_contacts]
    for k, cap in enumerate(pr.capacity):
        if np.isfinite(cap):
            rows.append((pr.group == k).astype(float)); lo.append(0.0); hi.append(cap)
    for k, (smax, smin) in enumerate(zip(pr.seg_max, pr.seg_min)):
        member = (pr.segment == k).astype(float)
        if smax < 1:
            rows.append(member - smax); lo.append(-np.inf); hi.append(0.0)
        if smin > 0:
            rows.append(member - smin); lo.append(0.0); hi.append(np.inf)
    if cc.protected_max_share < 1:
        rows.append(pr.protected.astype(float) - cc.protected_max_share); lo.append(-np.inf); hi.append(0.0)
    A = sparse.csr_array(np.vstack(rows))
    res = milp(-pr.ep, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=LinearConstraint(A, lo, hi),
               options={"time_limit": oc.milp_time_limit, "mip_rel_gap": oc.milp_rel_gap})
    x = (res.x > 0.5) if res.x is not None else np.zeros(n, bool)
    return MILPResult(x=x, expected_profit=float(pr.ep[x].sum()), status=int(res.status), message=str(res.message),
                      mip_gap=float(getattr(res, "mip_gap", np.nan) or 0.0),
                      dual_bound=float(-getattr(res, "mip_dual_bound", np.nan)),
                      runtime_s=time.perf_counter() - t0)


# ============================================================================= independent constraint check
def check_constraints(table: pd.DataFrame, selected_ids, cc: cfg.CampaignConstraints = cfg.CONSTRAINTS) -> pd.DataFrame:
    """One row per constraint: value, limit, ok. Works from the campaign table and customer IDs only,
    independently of the optimiser internals, so it can audit every strategy (plan 4.3)."""
    ids = pd.Index(list(selected_ids))
    s = table[table["customer_id"].isin(ids)]
    N = len(s)
    spend = float(s["contact_cost"].sum())
    rows = [("Unique customers", len(ids) - ids.nunique(), 0, ids.is_unique),
            ("Known customers", len(ids) - N, 0, N == len(ids)),
            ("Eligible customers only", int((s["eligible"] != 1).sum()), 0, bool((s["eligible"] == 1).all())),
            ("Budget (£)", spend, cc.budget, spend <= cc.budget + 1e-6),
            ("Contacts", N, cc.max_contacts, N <= cc.max_contacts)]
    for g, cap in cc.channel_capacity.items():
        ng = int((s["channel_group"] == g).sum())
        rows.append((f"Channel capacity: {g}", ng, cap, ng <= cap))
    share = (lambda m: float(m.sum()) / N) if N else (lambda m: 0.0)
    for seg_name in cfg.SEGMENTS:
        sh = share(s["segment"].astype(str) == seg_name)
        if seg_name in cc.segment_max_share:
            mx = cc.segment_max_share[seg_name]
            rows.append((f"Max share: {seg_name}", sh, mx, sh <= mx + 1e-9))
        if seg_name in cc.segment_min_share:
            mn = cc.segment_min_share[seg_name]
            rows.append((f"Min share: {seg_name}", sh, mn, N == 0 or sh >= mn - 1e-9))   # shares need a list
    ps = share(s["flag_age_75plus"] == 1)
    rows.append(("Max share: protected 75+", ps, cc.protected_max_share, ps <= cc.protected_max_share + 1e-9))
    return pd.DataFrame(rows, columns=["constraint", "value", "limit", "ok"])


def is_feasible(table: pd.DataFrame, selected_ids, cc: cfg.CampaignConstraints = cfg.CONSTRAINTS) -> bool:
    return bool(check_constraints(table, selected_ids, cc)["ok"].all())


# ============================================================================= comparison (Q5)
STRATEGY_ORDER = ("Random", "Propensity", "Greedy", "Genetic Algorithm", "Rule-based", "MILP optimum")
CHART_STRATEGIES = ("Random", "Propensity", "Greedy", "Genetic Algorithm")


def selection_metrics(pr: Problem, x: np.ndarray) -> dict:
    x = np.asarray(x, bool)
    spend = float(pr.cost[x].sum())
    ep = float(pr.ep[x].sum())
    return {"contacts": int(x.sum()), "spend": spend,
            "expected_conversions": float(pr.propensity[x].sum()),
            "expected_contribution": ep,
            "roi": ep / spend if spend else np.nan,
            **{f"contacts_{g}": int((x & (pr.group == k)).sum()) for k, g in enumerate(pr.groups)}}


@dataclass
class Comparison:
    table: pd.DataFrame                    # one row per strategy
    selections: dict                       # strategy -> bool array over problem (Random: seed 0 example)
    problem: Problem
    ga: GAResult | None                    # None until the GA has been run (dashboard: on demand)
    milp: MILPResult | None                # None until the exact optimum has been solved
    random_runs: pd.DataFrame              # metrics for every random seed

    def selected_ids(self, strategy: str) -> np.ndarray:
        return self.problem.customer_id[self.selections[strategy]]

    @property
    def has_ga(self) -> bool:
        return self.ga is not None

    @property
    def has_milp(self) -> bool:
        return self.milp is not None and self.milp.x.any()


def _summarise(table: pd.DataFrame, pr: Problem, sel: dict, runs: pd.DataFrame,
               cc: cfg.CampaignConstraints, milp_ep: float | None) -> pd.DataFrame:
    rows = []
    metric_cols = [c for c in runs.columns if c not in ("seed", "feasible")]
    for name in STRATEGY_ORDER:
        if name not in sel:
            continue
        if name == "Random":
            row = {"strategy": name, **runs[metric_cols].mean().to_dict(),
                   "expected_contribution_std": float(runs["expected_contribution"].std()),
                   "feasible": bool(runs["feasible"].all())}
            row["contacts"] = int(round(row["contacts"]))
        else:
            row = {"strategy": name, **selection_metrics(pr, sel[name]),
                   "feasible": is_feasible(table, pr.customer_id[sel[name]], cc)}
        rows.append(row)
    out = pd.DataFrame(rows).set_index("strategy")
    rnd = out.loc["Random", "expected_contribution"]
    out["incremental_vs_random"] = out["expected_contribution"] - rnd
    out["uplift_vs_random"] = out["expected_contribution"] / rnd - 1 if rnd else np.nan
    if milp_ep:
        out["pct_of_optimum"] = out["expected_contribution"] / milp_ep
    return out


def compare_strategies(table: pd.DataFrame, cc: cfg.CampaignConstraints = cfg.CONSTRAINTS,
                       ga: cfg.GAConfig = cfg.GA, oc: cfg.OptimConfig = cfg.OPTIM,
                       run_milp: bool = True, progress=None, run_genetic: bool = True) -> Comparison:
    """All strategies on one campaign table. `run_genetic=False, run_milp=False` gives the instant
    baselines (Random, Propensity, Greedy, Rule-based); `with_optimisers()` adds the GA and MILP later."""
    pr = build_problem(table, cc)
    runs = []
    sel = {}
    for seed in range(oc.random_runs):
        x = random_selection(pr, seed)
        runs.append({"seed": seed, **selection_metrics(pr, x),
                     "feasible": is_feasible(table, pr.customer_id[x], cc)})
        if seed == 0:
            sel["Random"] = x
    runs = pd.DataFrame(runs)
    sel["Propensity"] = propensity_only(pr)
    sel["Greedy"] = greedy(pr)
    sel["Rule-based"] = rule_based(pr)
    base = Comparison(_summarise(table, pr, sel, runs, cc, None), sel, pr, None, None, runs)
    if not (run_genetic or run_milp):
        return base
    ga_res = run_ga(pr, ga, progress) if run_genetic else None
    mres = milp_optimum(pr, oc) if run_milp else None
    return with_optimisers(table, base, ga_res, mres)


def with_optimisers(table: pd.DataFrame, base: Comparison, ga_res: GAResult | None,
                    mres: MILPResult | None) -> Comparison:
    """Add GA and/or MILP results (run now, or loaded from artifacts) to a baseline comparison."""
    sel = {k: v for k, v in base.selections.items() if k not in ("Genetic Algorithm", "MILP optimum")}
    if ga_res is not None:
        sel["Genetic Algorithm"] = ga_res.x
    if mres is not None and mres.x.any():
        sel["MILP optimum"] = mres.x
    milp_ep = mres.expected_profit if (mres is not None and mres.x.any()) else None
    tb = _summarise(table, base.problem, sel, base.random_runs, base.problem.cc, milp_ep)
    return Comparison(tb, sel, base.problem, ga_res, mres, base.random_runs)


def selection_mix(table: pd.DataFrame, comparison: Comparison, by: str = "segment") -> pd.DataFrame:
    """Share of each strategy's list by segment or channel group (Random = seed 0 example)."""
    pr = comparison.problem
    cols = {}
    for name, x in comparison.selections.items():
        s = table.loc[pr.table_index[x], by].astype(str)
        cols[name] = s.value_counts(normalize=True)
    order = list(cfg.SEGMENTS) if by == "segment" else None
    mix = pd.DataFrame(cols).fillna(0.0)
    return mix.reindex(order).fillna(0.0) if order else mix


def target_list(table: pd.DataFrame, comparison: Comparison, strategy: str = "Genetic Algorithm") -> pd.DataFrame:
    """The selected customers, best expected profit first (downloadable in the dashboard)."""
    cols = ["customer_id", "segment", "channel", "propensity", "value", "contact_cost", "break_even_cost",
            "expected_profit", "ep_per_cost", "age", "flag_age_75plus"]
    pr = comparison.problem
    t = table.loc[pr.table_index[comparison.selections[strategy]], cols]
    return t.sort_values("expected_profit", ascending=False).reset_index(drop=True)
