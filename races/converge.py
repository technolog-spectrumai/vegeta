"""Generalized pattern search (GPS; Torczon 1997, Audet & Dennis 2003) on the MuJoCo autopilot of one course.

The knobs are scaled to x ∈ [0, 1]^n (``BOUNDS``). Each iteration *polls* the 2n points x ± Δ·e_i (a positive spanning
set) on the current mesh, in a fixed order, and takes the first that lowers the score (opportunistic poll):

- success: move there, Δ ← min(2Δ, Δ_max) (mesh expansion);
- failure (no poll point better, x is a *mesh local optimum*): Δ ← Δ/2 (mesh refinement).

Stopping (the convergence criterion): Δ < Δ_min. For a score that is continuously differentiable near the limit the
refining subsequence of mesh local optima converges to a stationary point (Torczon's theorem; for a merely Lipschitz
score, to a Clarke-stationary point in the directions polled, Audet & Dennis). The lap score is piecewise: a lap that
turns DNF jumps, so the honest guarantee here is a **mesh local optimum at resolution Δ_min** (1/64 of every knob's
range by default), not a global optimum. ``max_laps`` is a safety budget only; ``patience`` (off by default) is an
optional heuristic early stop.

Score (lower is better): a finished lap = lap time + 2 s per collision; a DNF = 1000 − 20 × zones passed; a crash of
the simulation = 2000. MuJoCo is deterministic, so a setting is flown once and remembered."""
from __future__ import annotations

import time

DEFAULT = {"safety": 0.6, "lookahead_s": 0.25, "omega_pos": 2.5, "omega_att": 18.0, "tilt_max_deg": 60.0, "v_cap": 25.0}
BOUNDS = {"safety": (0.2, 0.95), "lookahead_s": (0.1, 1.0), "omega_pos": (1.5, 10.0), "omega_att": (10.0, 40.0), "tilt_max_deg": (40.0, 75.0), "v_cap": (2.0, 25.0)}
NO_CAP = 25.0                                                  # v_cap at the top of its range = no cap (run_mujoco's None)


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
    s = {k: round(lo + x[k] * (hi - lo), 4) for k, (lo, hi) in BOUNDS.items()}
    if s["v_cap"] >= NO_CAP - 1e-9:
        s["v_cap"] = None
    return s


def search(course, fly, start: dict | None = None, *, delta0: float = 0.125, delta_min: float = 1 / 64, delta_max: float = 0.25,
           max_laps: int = 400, patience: int = 0, tol: float = 0.2, log=print):
    """``fly(settings) -> (stats, events)``. Returns ``(best settings, best stats, rows, reason)``."""
    seen, rows = {}, []
    it = 0

    def lap(x):
        x = {k: round(v, 6) for k, v in x.items()}
        key = tuple(sorted(x.items()))
        if key in seen:
            return seen[key]
        s = to_settings(x)
        t0 = time.time()
        st, ev = fly(s)
        sc = score(st)
        zones = [e[1] for e in ev if isinstance(e, (list, tuple)) and len(e) > 1 and str(e[1]).startswith("zone")]
        hits = [e for e in ev if isinstance(e, (list, tuple)) and len(e) > 1 and "collision" in str(e[1]).lower()]
        rows.append({"iteration": it, **s, "score": round(sc, 3), "status": st.get("status"), "reason": st.get("reason", ""), "lap time [s]": st.get("lap time [s]"),
                     "collisions": st.get("collisions"), "zones passed": st.get("zones passed"), "last zone reached": zones[-1] if zones else "",
                     "collisions at": "; ".join(f"{e[0]:.1f} s {e[1]}" for e in hits[:6]), "wall [s]": round(time.time() - t0, 1)})
        seen[key] = (sc, st)
        log(f"  lap {len(seen):3d}  {s}  → {st.get('status')} {st.get('lap time [s]') or ''} {st.get('zones passed') or ''}, score {sc:.2f}")
        return seen[key]

    x = to_x(start or {})
    best_sc, best_st = lap(x)
    delta, stale, reason = delta0, 0, ""
    while True:
        it += 1
        before, moved = best_sc, False
        for k in BOUNDS:                                           # poll: ±Δ along every axis, first improvement taken
            for d in (+delta, -delta):
                y = {**x, k: x[k] + d}
                if not 0.0 <= y[k] <= 1.0:
                    continue
                if len(seen) >= max_laps:
                    reason = f"budget: {max_laps} laps (Δ = {delta:g} not yet below {delta_min:g})"
                    break
                sc, st = lap(y)
                if sc < best_sc - 1e-9:
                    x, best_sc, best_st, moved = y, sc, st, True
                    break
            if moved or reason:
                break
        if reason:
            break
        delta = min(2 * delta, delta_max) if moved else delta / 2
        stale = stale + 1 if before - best_sc < tol else 0
        log(f"iteration {it}: {'success' if moved else 'mesh local optimum'}, score {best_sc:.2f}, Δ → {delta:g}, laps {len(seen)}")
        if delta < delta_min:
            reason = f"converged: Δ = {delta:g} < Δ_min = {delta_min:g} (a mesh local optimum at 1/{round(1 / delta_min)} of every knob's range)"
            break
        if patience and stale >= patience and best_st.get("status") == "finished":
            reason = f"early stop (heuristic): {patience} iterations gaining less than {tol} s"
            break
    return to_settings(x), best_st, rows, reason


# ================================================================================================= global stage: PSO
def pso(course, fly, start: dict | None = None, *, particles: int = 12, iterations: int = 15, seed: int = 0,
        stall: int = 4, radius_tol: float = 0.05, max_laps: int = 400, log=print):
    """Particle swarm optimisation with constriction (Kennedy & Eberhart 1995; Clerc & Kennedy 2002, *IEEE Trans.
    Evol. Comput.* 6(1)): v ← χ[v + c₁r₁(p − x) + c₂r₂(g − x)], χ = 0.7298, c₁ = c₂ = 1.49618 (φ = 4.1, the
    parameters with the proven convergence of the swarm's trajectories), positions clipped to [0, 1]^n, velocity
    limited to 0.2 per step. The swarm samples the whole box, which is what guards against the local minima a pattern
    search alone would settle into. One particle starts at ``start`` (the stored settings); the others are a Latin
    hypercube. Stops on the first of: ``iterations``; no improvement of the global best in ``stall`` iterations; the
    swarm's radius (largest distance of a particle's best from the global best) below ``radius_tol``. Returns
    ``(best x, rows)`` for the GPS polish."""
    import numpy as np
    rng = np.random.default_rng(seed)
    keys = list(BOUNDS)
    n = len(keys)
    chi, c1, c2, vmax = 0.7298, 1.49618, 1.49618, 0.2
    lhs = (rng.permuted(np.tile(np.arange(particles), (n, 1)), axis=1).T + rng.random((particles, n))) / particles
    X = lhs
    X[0] = np.array([to_x(start or {})[k] for k in keys])
    V = rng.uniform(-vmax, vmax, (particles, n)) * 0.5
    seen, rows = {}, []

    def f(xv, it):
        xd = {k: round(float(v), 6) for k, v in zip(keys, xv)}
        key = tuple(sorted(xd.items()))
        if key not in seen:
            s = to_settings(xd)
            st, ev = fly(s)
            sc = score(st)
            seen[key] = sc
            rows.append({"stage": "pso", "iteration": it, **s, "score": round(sc, 3), "status": st.get("status"), "lap time [s]": st.get("lap time [s]"),
                         "collisions": st.get("collisions"), "zones passed": st.get("zones passed")})
            log(f"  pso lap {len(seen):3d}  it {it}  {s}  → {st.get('status')} {st.get('lap time [s]') or ''} {st.get('zones passed') or ''}, score {sc:.2f}")
        return seen[key]

    P = X.copy()
    Pf = np.array([f(x, 0) for x in X])
    g = P[Pf.argmin()].copy()
    gf, since = Pf.min(), 0
    for it in range(1, iterations + 1):
        r1, r2 = rng.random((particles, n)), rng.random((particles, n))
        V = np.clip(chi * (V + c1 * r1 * (P - X) + c2 * r2 * (g - X)), -vmax, vmax)
        X = np.clip(X + V, 0.0, 1.0)
        for i in range(particles):
            if len(seen) >= max_laps:
                break
            fx = f(X[i], it)
            if fx < Pf[i]:
                P[i], Pf[i] = X[i].copy(), fx
        if Pf.min() < gf - 1e-9:
            g, gf, since = P[Pf.argmin()].copy(), Pf.min(), 0
        else:
            since += 1
        radius = float(np.max(np.linalg.norm(P - g, axis=1)))
        log(f"pso iteration {it}: global best {gf:.2f}, swarm radius {radius:.3f}, laps {len(seen)}")
        if since >= stall:
            log(f"pso: stopped — no improvement in {stall} iterations"); break
        if radius < radius_tol:
            log(f"pso: stopped — swarm radius {radius:.3f} < {radius_tol}"); break
        if len(seen) >= max_laps:
            log("pso: stopped — lap budget"); break
    return {k: float(v) for k, v in zip(keys, g)}, rows
