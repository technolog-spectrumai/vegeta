"""The searches that tune the MuJoCo autopilot of one course: a particle swarm (global), then a generalized pattern
search (local polish). Both are resumable state machines: ``step()`` runs one iteration, ``to_dict()`` /
``from_dict()`` save and restore everything needed to continue exactly where they stopped.

**Score** (lower is better): a finished lap = lap time + 2 s per collision; a DNF = 1000 − 20 × zones passed (passing
more zones is progress before any lap finishes); a crash of the simulation = 2000.

**PSO** (Kennedy & Eberhart 1995; Clerc & Kennedy 2002, *IEEE Trans. Evol. Comput.* 6(1)): v ← χ[v + c₁r₁(p − x) +
c₂r₂(g − x)], χ = 0.7298, c₁ = c₂ = 1.49618, positions clipped to [0, 1]^n, |v| ≤ 0.2. One particle starts at the
stored settings, the others on a seeded Latin hypercube. It ends on the first of: ``iterations`` (a hard maximum —
it always hands over to the polish then, even while still improving), ``stall`` iterations without a better global
best, the swarm radius below ``radius_tol``, or its share of the lap budget.

**GPS** (Torczon 1997; Audet & Dennis 2003): the 2n points x ± Δ·eᵢ are polled; the poll is flown completely (in
parallel) and then read in a fixed order, the first improvement taken (an opportunistic poll, so the decisions do not
depend on how many laps fly side by side). Success: move, Δ ← min(2Δ, Δ_max); failure (a mesh local optimum):
Δ ← Δ/2. Converged when Δ < Δ_min: a mesh local optimum at that resolution — not a global optimum.

The swarm never cuts laps short (each particle compares with its own best, so a cut could change its path); the
polish does (``cut=True``): it only compares with the incumbent, which a lap slower than it can never beat."""
from __future__ import annotations

DEFAULT = {"safety": 0.6, "lookahead_s": 0.25, "omega_pos": 2.5, "omega_att": 18.0, "tilt_max_deg": 60.0, "v_cap": 25.0}
BOUNDS = {"safety": (0.2, 0.95), "lookahead_s": (0.1, 1.0), "omega_pos": (1.5, 10.0), "omega_att": (10.0, 40.0), "tilt_max_deg": (40.0, 75.0), "v_cap": (2.0, 25.0)}
NO_CAP = 25.0                                                  # v_cap at the top of its range = no cap (run_mujoco's None)
KEYS = list(BOUNDS)


def score(st: dict) -> float:
    if st.get("status") == "finished":
        return float(st["lap time [s]"]) + 2.0 * float(st.get("collisions") or 0)
    if st.get("status") == "error":
        return 2000.0
    passed = int(str(st.get("zones passed") or "0 of").split(" ")[0] or 0)
    return 1000.0 - 20.0 * passed


def to_x(s: dict) -> dict:
    s = {**DEFAULT, **{k: v for k, v in s.items() if k in BOUNDS}}
    s["v_cap"] = NO_CAP if s.get("v_cap") is None else s["v_cap"]
    return {k: min(1.0, max(0.0, (s[k] - lo) / (hi - lo))) for k, (lo, hi) in BOUNDS.items()}


def to_settings(x: dict) -> dict:
    s = {k: round(lo + float(x[k]) * (hi - lo), 4) for k, (lo, hi) in BOUNDS.items()}
    if s["v_cap"] >= NO_CAP - 1e-9:
        s["v_cap"] = None
    return s


def _xd(v) -> dict:
    return {k: round(float(a), 6) for k, a in zip(KEYS, v)}


class PSO:
    """The swarm. ``laps`` must offer ``prefetch(settings_list, stage, iteration, cut)`` (returns the number of laps
    newly flown) and ``score(settings)``."""

    def __init__(self, start: dict, *, particles: int = 12, iterations: int = 15, seed: int = 0, stall: int = 4,
                 radius_tol: float = 0.05, budget: int = 280):
        self.cfg = dict(start=start, particles=particles, iterations=iterations, seed=seed, stall=stall, radius_tol=radius_tol, budget=budget)
        self.it, self.laps_used, self.since, self.done, self.reason = 0, 0, 0, False, ""
        self.X = self.V = self.P = self.Pf = self.g = None
        self.gf = None
        self.rng_state = None

    def _rng(self):
        import numpy as np
        r = np.random.default_rng(self.cfg["seed"])
        if self.rng_state is not None:
            r.bit_generator.state = self.rng_state
        return r

    def step(self, laps, log=print):
        import numpy as np
        rng = self._rng()
        n, m = len(KEYS), self.cfg["particles"]
        chi, c1, c2, vmax = 0.7298, 1.49618, 1.49618, 0.2
        if self.X is None:                                         # iteration 0: the initial swarm
            X = (rng.permuted(np.tile(np.arange(m), (n, 1)), axis=1).T + rng.random((m, n))) / m
            X[0] = np.array([to_x(self.cfg["start"])[k] for k in KEYS])
            V = rng.uniform(-vmax, vmax, (m, n)) * 0.5
            self.laps_used += laps.prefetch([to_settings(_xd(x)) for x in X], "pso", 0, cut=False)
            F = np.array([laps.score(to_settings(_xd(x))) for x in X])
            self.X, self.V, self.P, self.Pf = X, V, X.copy(), F
            self.g, self.gf = X[F.argmin()].copy(), float(F.min())
            log(f"pso iteration 0 (initial swarm): global best {self.gf:.2f}, laps {self.laps_used}")
        else:
            X, V, P, Pf, g = (np.array(a) for a in (self.X, self.V, self.P, self.Pf, self.g))
            if self.laps_used + m > self.cfg["budget"]:
                self.done, self.reason = True, f"PSO lap budget ({self.cfg['budget']}) reached"
                self.rng_state = rng.bit_generator.state
                return
            self.it += 1
            r1, r2 = rng.random((m, n)), rng.random((m, n))
            V = np.clip(chi * (V + c1 * r1 * (P - X) + c2 * r2 * (g - X)), -vmax, vmax)
            X = np.clip(X + V, 0.0, 1.0)
            self.laps_used += laps.prefetch([to_settings(_xd(x)) for x in X], "pso", self.it, cut=False)
            F = np.array([laps.score(to_settings(_xd(x))) for x in X])
            better = F < Pf
            P[better], Pf[better] = X[better], F[better]
            improved = Pf.min() < self.gf - 1e-9
            if improved:
                g, self.gf, self.since = P[Pf.argmin()].copy(), float(Pf.min()), 0
            else:
                self.since += 1
            self.X, self.V, self.P, self.Pf, self.g = X, V, P, Pf, g
            radius = float(np.max(np.linalg.norm(P - g, axis=1)))
            log(f"pso iteration {self.it}: global best {self.gf:.2f}{' (improved)' if improved else ''}, swarm radius {radius:.3f}, laps {self.laps_used}")
            if self.it >= self.cfg["iterations"]:
                self.done = True
                self.reason = (f"PSO hard limit of {self.cfg['iterations']} iterations reached"
                               + ("; it was still improving — switching to polishing" if self.since == 0 else ""))
            elif self.since >= self.cfg["stall"]:
                self.done, self.reason = True, f"no better global best in {self.cfg['stall']} iterations"
            elif radius < self.cfg["radius_tol"]:
                self.done, self.reason = True, f"swarm radius {radius:.3f} < {self.cfg['radius_tol']}"
        self.rng_state = rng.bit_generator.state
        if self.done:
            log(f"pso: stopped — {self.reason}")

    def best(self) -> dict:
        return to_settings(_xd(self.g))

    def to_dict(self) -> dict:
        lst = lambda a: None if a is None else [list(map(float, r)) for r in a] if getattr(a, "ndim", 1) == 2 else list(map(float, a))
        return {"cfg": self.cfg, "it": self.it, "laps_used": self.laps_used, "since": self.since, "done": self.done, "reason": self.reason,
                "X": lst(self.X), "V": lst(self.V), "P": lst(self.P), "Pf": lst(self.Pf), "g": lst(self.g), "gf": self.gf, "rng_state": self.rng_state}

    @classmethod
    def from_dict(cls, d: dict) -> "PSO":
        import numpy as np
        p = cls(**d["cfg"])
        p.it, p.laps_used, p.since, p.done, p.reason, p.gf, p.rng_state = d["it"], d["laps_used"], d["since"], d["done"], d["reason"], d["gf"], d["rng_state"]
        for k in ("X", "V", "P", "Pf", "g"):
            setattr(p, k, None if d[k] is None else np.array(d[k], dtype=float))
        return p


class GPS:
    """The pattern-search polish from ``start`` (settings)."""

    def __init__(self, start: dict, *, delta0: float = 0.125, delta_min: float = 1 / 64, delta_max: float = 0.25,
                 budget: int = 120, patience: int = 0, tol: float = 0.2):
        self.cfg = dict(start=start, delta0=delta0, delta_min=delta_min, delta_max=delta_max, budget=budget, patience=patience, tol=tol)
        self.x = to_x(start)
        self.delta, self.it, self.laps_used, self.stale, self.done, self.reason = delta0, 0, 0, 0, False, ""
        self.best_sc = None

    def step(self, laps, log=print):
        c = self.cfg
        if self.best_sc is None:                                  # the start point's own score (never cut)
            self.laps_used += laps.prefetch([to_settings(self.x)], "gps", 0, cut=False)
            self.best_sc = laps.score(to_settings(self.x))
            log(f"gps start: score {self.best_sc:.2f}")
            return
        poll = [(k, d) for k in KEYS for d in (+self.delta, -self.delta) if 0.0 <= self.x[k] + d <= 1.0]
        if self.laps_used + len(poll) > c["budget"]:
            self.done, self.reason = True, f"polishing lap budget ({c['budget']}) reached at Δ = {self.delta:g} (not yet below {c['delta_min']:g})"
            log(f"gps: stopped — {self.reason}")
            return
        self.it += 1
        before = self.best_sc
        ys = [{**self.x, k: self.x[k] + d} for k, d in poll]      # the poll points, flown exactly as stored
        pts = [to_settings(y) for y in ys]
        self.laps_used += laps.prefetch(pts, "gps", self.it, cut=True)
        moved = False
        for y, s in zip(ys, pts):                                 # opportunistic, in the fixed axis order
            sc = laps.score(s)
            if sc < self.best_sc - 1e-9:
                self.x, self.best_sc, moved = y, sc, True
                break
        self.delta = min(2 * self.delta, c["delta_max"]) if moved else self.delta / 2
        self.stale = self.stale + 1 if before - self.best_sc < c["tol"] else 0
        log(f"gps iteration {self.it}: {'success' if moved else 'mesh local optimum'}, score {self.best_sc:.2f}, Δ → {self.delta:g}, laps {self.laps_used}")
        if self.delta < c["delta_min"]:
            self.done = True
            self.reason = f"converged: Δ = {self.delta:g} < Δ_min = {c['delta_min']:g} — a mesh local optimum at 1/{round(1 / c['delta_min'])} of every knob's range, not a proven global optimum"
        elif c["patience"] and self.stale >= c["patience"]:
            self.done, self.reason = True, f"early stop (heuristic): {c['patience']} iterations gaining less than {c['tol']} s"
        if self.done:
            log(f"gps: stopped — {self.reason}")

    def best(self) -> dict:
        return to_settings(self.x)

    def to_dict(self) -> dict:
        return {"cfg": self.cfg, "x": self.x, "delta": self.delta, "it": self.it, "laps_used": self.laps_used, "stale": self.stale,
                "done": self.done, "reason": self.reason, "best_sc": self.best_sc}

    @classmethod
    def from_dict(cls, d: dict) -> "GPS":
        g = cls(**d["cfg"])
        for k in ("x", "delta", "it", "laps_used", "stale", "done", "reason", "best_sc"):
            setattr(g, k, d[k])
        return g
