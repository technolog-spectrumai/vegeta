"""ASTOR, the goshawk's way: experiments on what the bird's mechanisms would buy the propeller dart (``astor``).
Exploration, not a design: each mechanism is modelled as simply as it can be and compared with the fixed dart on the
same simulator. Run:  python labs/aero/designs/astor_hawk.py   (minutes; writes data/astor_hawk.yaml and a figure).

The goshawk's skills, and how each is modelled here:

1. **A wing that spreads and tucks.** Three wing states on ASTOR's body (``STATES``): *spread* (1.0 m, little
   sweep, 0.22 m2), *mid* (the fixed dart: 0.8 m, 0.158 m2) and *tucked* (0.56 m, 35 deg of sweep, 0.10 m2 — the
   outer panels slide over the inner ones, as feathers overlap: the wetted area falls, which is what lowers drag at
   these speeds; sweep alone would not). Each state gets its own lattice coefficients; the zero-lift drag is the
   body's fixed drag area plus the wing's by its area (HAND split). ``SimMorph`` blends the states by airspeed
   (spread below 28 m/s, tucked above 50), at a limited rate. HAND: +120 g for the mechanism.
2. **A tail that fans into a brake.** A drag area added when the autopilot has the throttle closed above 22 m/s
   (``BRAKE_CDA``, HAND: the fanned tail and the spread wing's spoiling).
3. **The hawk's steering.** Harris's hawks chasing a jinking target are described by a mixed law (Brighton & Taylor
   2019): turn rate = N x line-of-sight rate + K x the angle off the target, with N ~ 0.7 and K ~ 1.2 /s — less
   anticipation than a missile's proportional navigation (N 3-5), more pointing at the target; it tolerates
   delay and suits short, close chases. Flown here in place of the simulator's PN (N = 4).
4. **Catching with the feet.** A reaching capture device forgives a miss of its reach but not a high closing speed.
   Here: the closing speed at the first pass inside 5 m, and the closest approach.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor as A
import astor_model as M
import astor_mujoco as SM

WING_CD = 0.010                    # HAND: the wing's zero-lift drag on its own area (+-0.003)
MECHANISM_G = 120.0                # HAND: pivots, actuator, sliding panels (+-40)
BRAKE_CDA = 0.030                  # HAND: the fanned tail + spoiling, drag area [m2] (+-0.010)
V_SPREAD, V_TUCK = 28.0, 50.0      # the schedule: spread below, tucked above
MORPH_RATE = 2.0                   # full travel in 0.5 s (a goshawk tucks in ~0.1-0.2 s)
STATES = {  # wing parameters over ASTOR's
    "tucked": dict(span=560.0, root_chord=250.0, taper=0.45, le_sweep_deg=35.0),   # 35 deg: the CAD parameter's limit
    "mid": dict(),
    "spread": dict(span=1000.0, root_chord=295.0, taper=0.50, le_sweep_deg=12.0),
}
HAWK_N, HAWK_K = 0.7, 1.2


def state_coefficients(name: str, mechanism: bool = True) -> dict:
    """ASTOR's coefficients with the wing in a state: the lattice for that planform, the drag by the wetted area."""
    base = M.get()
    body_cda = (M.CD0 - WING_CD) * base["S"]
    m = base["mass"] + (MECHANISM_G / 1000.0 if mechanism else 0.0)
    wx = A.variant()["wing_x"]
    for _ in range(4):                 # the pivot is placed so the static margin stays ASTOR's 10 % in every state:
        q = A.variant(**STATES[name], wing_x=wx)      # spreading moves the lift forward, so the spread wing sits further aft
        p = A.resolve(q)
        L = A.Astor.layout(p)
        S = L["S_mm2"] / 1e6
        C = am.build(q, p, L, "astor", "pusher", mass=m, CD0=(body_cda + WING_CD * S) / S, thrust=base["thrust"])
        # the CG does not move with the wing: the moments are taken about ASTOR's CG
        dx = base["x_cg_m"] - C["x_cg_m"]
        C["Cm0"] += C["CL0"] * dx / C["c"]
        C["Cma"] += C["CLa"] * dx / C["c"]
        C["x_cg_m"] = base["x_cg_m"]
        C["static_margin"] = -C["Cma"] / C["CLa"]
        if abs(C["static_margin"] - 0.10) < 0.005:
            break
        wx += (0.10 - C["static_margin"]) * C["c"] * 1000.0
    C["wing_x_mm"] = round(wx, 1)
    C["CL_MAX"] *= M.THIN_CLMAX
    C["alpha_stall"] = (C["CL_MAX"] - C["CL0"]) / C["CLa"]
    C.update({"aircraft": f"ASTOR {name}", "mass_budget": base["mass_budget"], "pack_kind": "astor", "equipment": base["equipment"]})
    return C


def state_numbers(C) -> dict:
    W, S = C["mass"] * am.G, C["S"]
    Vs = math.sqrt(W / (0.5 * am.RHO * S * C["CL_MAX"]))
    out = {"S_m2": round(S, 3), "span_m": round(C["b"], 2), "AR": round(C["AR"], 1), "CD0": round(C["CD0"], 4),
           "drag_area_cm2": round(C["CD0"] * S * 1e4, 1), "stall_m_s": round(Vs, 1), "top_speed_m_s": round(M.top_speed(C), 1),
           "static_margin": round(C["static_margin"], 3), "wing_x_mm": C.get("wing_x_mm"), "roll_damping_Clp": round(C["Clp"], 3)}
    for V in (20.0, 30.0, 45.0, 60.0):
        n = min(0.5 * am.RHO * V * V * S * C["CL_MAX"] / W, 8.0)
        out[f"turn_{V:.0f}_deg_s"] = round(math.degrees(am.G * math.sqrt(max(n * n - 1, 0)) / V), 0) if n > 1 else 0.0
        out[f"gust_g_per_m_s_{V:.0f}"] = round(0.5 * am.RHO * V * S * C["CLa"] / W, 2)
    return out


class SimMorph(SM.SimAstor):
    """ASTOR with the wing blended between the states by airspeed, and the tail's brake."""

    def __init__(self, bandit=False, brake=True, mesh=None):
        self.states = [state_coefficients(k) for k in ("tucked", "mid", "spread")]
        super().__init__(bandit=bandit, C=dict(self.states[1]), mesh=mesh)
        self.s_morph = 0.5
        self.use_brake = brake
        self.braking = 0.0
        self._keys = [k for k, v in self.states[1].items() if isinstance(v, float) and isinstance(self.states[0].get(k), float)]

    def step(self, n=1):
        V = self.state()["V"]
        want = min(1.0, max(0.0, (V_TUCK - V) / (V_TUCK - V_SPREAD)))
        d = MORPH_RATE * amj.DT * n
        self.s_morph += max(-d, min(d, want - self.s_morph))
        lo, hi, f = (self.states[0], self.states[1], self.s_morph / 0.5) if self.s_morph <= 0.5 else (self.states[1], self.states[2], (self.s_morph - 0.5) / 0.5)
        for k in self._keys:
            self.C[k] = lo[k] + f * (hi[k] - lo[k])
        self.C["inertia"] = tuple(a + f * (b - a) for a, b in zip(lo["inertia"], hi["inertia"]))
        want_b = 1.0 if (self.use_brake and self.controls[3] < 0.02 and V > 22.0) else 0.0
        self.braking += max(-4 * d, min(4 * d, want_b - self.braking))
        self.C["CD0"] += self.braking * BRAKE_CDA / self.C["S"]
        super().step(n)


def hawk_command(sim, b, vb):
    """The mixed law in the horizontal plane (see the module's notes); the height as the simulator's PN does."""
    s = sim.state()
    r = b - s["pos"]
    vr = vb - s["v_world"]
    rh2 = max(r[0] ** 2 + r[1] ** 2, 1e-6)
    lam_dot = (r[0] * vr[1] - r[1] * vr[0]) / rh2
    delta = (math.atan2(r[1], r[0]) - math.atan2(s["v_world"][1], s["v_world"][0]) + math.pi) % (2 * math.pi) - math.pi
    V = max(s["V"], 3.0)
    psi_dot = HAWK_N * lam_dot + HAWK_K * delta
    phi = math.atan(-V * psi_dot / am.G)
    return max(-1.3, min(1.3, phi)), max(-0.35, min(0.35, math.atan2(r[2], math.sqrt(rh2))))


# the scenarios: ASTOR arrives fast; the target is fast (the requirement) or slow (the end game a dart overshoots)
SCENARIOS = {
    "fast_50": ("else", dict(Vb=50.0, h=120.0, gap=300.0, T=60.0, V=60.0, what="a 50 m/s fixed wing, gentle weave, 300 m ahead")),
    "dash_to_zigzag": ("zigzag", dict(Vb=14.0, h=60.0, gap=400.0, T=40.0, V=60.0, what="arriving at 60 m/s on a 14 m/s weaving quad, 400 m ahead")),
    "dash_to_urban": ("urban", dict(Vb=10.0, h=20.0, gap=400.0, T=45.0, V=60.0, what="arriving at 60 m/s on a 10 m/s quad turning hard at 20 m, 400 m ahead")),
}
STARTS = ((0.0, 0.0, 0.0), (8.0, 0.0, 0.0), (-8.0, 0.0, 0.0), (0.0, 6.0, 2.0), (0.0, -6.0, -2.0))


def fly(sim_factory, scenario, guidance="pn") -> dict:
    """A scenario from the perturbed starts: first time inside 5 m, the closing speed then, the closest approach, the
    share of time inside 10 m, the peak load factor."""
    path, c = SCENARIOS[scenario]
    key = path if path in ("zigzag", "urban") else "_hawk_tmp"
    old_c, old_pn = amj.CHASES.get(key), amj.pn_command
    amj.CHASES[key] = c
    if guidance == "hawk":
        amj.pn_command = hawk_command
    runs = []
    try:
        t_b, bx, by, bz, bvx, bvy, bvz = amj.bandit_path(key)
        for hd, y0, dv in STARTS:
            h, s = amj.chase(sim_factory(), key, V=c["V"] + dv, heading_deg=hd, y0=y0)
            t = np.asarray(h["t"])
            B = np.stack([np.interp(t, t_b, bx), np.interp(t, t_b, by), np.interp(t, t_b, bz)], 1)
            Pp = np.stack([h["x"], h["y"], h["z"]], 1)
            d = np.linalg.norm(B - Pp, axis=1)
            vp = np.gradient(Pp, t, axis=0)
            vb = np.stack([np.interp(t, t_b, bvx), np.interp(t, t_b, bvy), np.interp(t, t_b, bvz)], 1)
            k5 = np.where(d < 5.0)[0]
            i = int(k5[0]) if len(k5) else int(np.argmin(d))
            runs.append({"time_to_5m_s": round(float(t[k5[0]]), 2) if len(k5) else None,
                         "closing_speed_m_s": round(float(np.linalg.norm(vp[i] - vb[i])), 1),
                         "closest_m": round(float(d.min()), 2), "share_within_10m": round(float(np.mean(d < 10.0)), 3),
                         "peak_g": round(float(np.max(h["n"])), 1), "crashed": bool(s["crashed"])})
    finally:
        amj.pn_command = old_pn
        if old_c is None:
            amj.CHASES.pop(key, None)
        else:
            amj.CHASES[key] = old_c
    times = [r["time_to_5m_s"] if r["time_to_5m_s"] is not None else math.inf for r in runs]
    med = float(np.median(times))
    return {"what": c["what"], "time_to_5m_s": round(med, 2) if math.isfinite(med) else None,
            "catch_rate": round(sum(math.isfinite(x) for x in times) / len(runs), 2),
            "closing_speed_m_s": round(float(np.median([r["closing_speed_m_s"] for r in runs])), 1),
            "closest_m": round(float(np.median([r["closest_m"] for r in runs])), 2),
            "share_within_10m": round(float(np.mean([r["share_within_10m"] for r in runs])), 3),
            "peak_g": round(float(np.max([r["peak_g"] for r in runs])), 1),
            "crash_rate": round(sum(r["crashed"] for r in runs) / len(runs), 2), "runs": runs}


def braking(C, brake_cda=0.0, V0=65.0, V1=20.0) -> dict:
    """Level, throttle closed, from V0 to V1 (point mass): the time and the distance."""
    W, S, m = C["mass"] * am.G, C["S"], C["mass"]
    V, t, x, dt = V0, 0.0, 0.0, 0.01
    while V > V1 and t < 120:
        q = 0.5 * am.RHO * V * V
        D = q * (S * (C["CD0"] + C["k"] * (W / (q * S)) ** 2) + brake_cda)
        V -= D / m * dt
        x += V * dt
        t += dt
    return {"time_s": round(t, 1), "distance_m": round(x, 0)}


def explore() -> dict:
    out = {"states": {k: state_numbers(state_coefficients(k)) for k in STATES},
           "fixed_dart": state_numbers(M.get())}
    Ct, Cs = state_coefficients("tucked"), state_coefficients("spread")
    out["braking_65_to_20_m_s"] = {"fixed dart": braking(M.get()), "tucked, clean": braking(Ct),
                                   "spread, clean": braking(Cs), "spread + fanned tail": braking(Cs, BRAKE_CDA)}
    cfgs = {"fixed dart": lambda: SM.SimAstor(bandit=True),
            "morphing wing": lambda: SimMorph(bandit=True, brake=False),
            "morphing wing + tail brake": lambda: SimMorph(bandit=True, brake=True)}
    out["chases"] = {}
    for sc in SCENARIOS:
        out["chases"][sc] = {}
        for cn, sf in cfgs.items():
            for g in ("pn", "hawk"):
                out["chases"][sc][f"{cn} | {g}"] = fly(sf, sc, g)
                r = out["chases"][sc][f"{cn} | {g}"]
                print(f"{sc:16s} {cn:28s} {g:5s} t5 {str(r['time_to_5m_s']):>6s} catch {r['catch_rate']:.1f} closing {r['closing_speed_m_s']:5.1f} m/s "
                      f"closest {r['closest_m']:5.2f} m within10 {r['share_within_10m']:.3f} peak {r['peak_g']} g crash {r['crash_rate']}", flush=True)
    return out


if __name__ == "__main__":
    import json
    import yaml
    res = explore()
    p = Path(__file__).resolve().parent / "data" / "astor_hawk.yaml"
    p.write_text(yaml.safe_dump(json.loads(json.dumps(res, default=float)), sort_keys=False, width=140))
    print("->", p)


def with_latency(law, tau_s: float, sigma_m: float, seed: int = 0):
    """A guidance law fed the target as a camera gives it: ``tau_s`` late, its position with ``sigma_m`` of noise
    (independent each 20 ms; the velocity from the noisy positions over 0.1 s)."""
    rng = np.random.default_rng(seed)
    buf = []

    def cmd(sim, b, vb):
        t = sim.d.time
        buf.append((t, np.asarray(b, float) + rng.normal(0.0, sigma_m, 3)))
        while len(buf) > 2 and buf[1][0] <= t - tau_s - 0.1:
            buf.pop(0)
        old = [x for x in buf if x[0] <= t - tau_s]
        if len(old) < 2:
            return law(sim, b, vb)
        (t0, b0), (t1, b1) = old[0], old[-1]
        v = (b1 - b0) / max(t1 - t0, 1e-3)
        return law(sim, b1, v)
    return cmd


def latency_study(taus=(0.0, 0.15, 0.3), sigma_m=0.5) -> dict:
    """The fixed dart on two scenarios under PN and under the hawk's law, with the camera's delay and noise."""
    out = {}
    base_pn = amj.pn_command
    for sc in ("fast_50", "dash_to_zigzag"):
        for tau in taus:
            for g, law in (("pn", base_pn), ("hawk", hawk_command)):
                amj.pn_command = with_latency(law, tau, sigma_m if tau > 0 else 0.0)
                try:
                    r = fly(lambda: SM.SimAstor(bandit=True), sc, "pn")      # "pn": fly() must not swap the law again
                finally:
                    amj.pn_command = base_pn
                out[f"{sc} | delay {tau:.2f} s | {g}"] = {k: v for k, v in r.items() if k != "runs"}
                print(f"{sc:16s} delay {tau:.2f} {g:5s} t5 {str(r['time_to_5m_s']):>6s} catch {r['catch_rate']:.1f} closest {r['closest_m']:5.2f} m "
                      f"within10 {r['share_within_10m']:.3f} peak {r['peak_g']} g", flush=True)
    return out
