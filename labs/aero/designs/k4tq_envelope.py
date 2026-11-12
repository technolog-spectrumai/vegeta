"""K4TQ — the flight envelope: climbing, diving / descending and manoeuvring, for K4TQ-A2 (Cathy autonomous) and K4TQ-F2 (Freddy
autonomous) — any variant on request.

On the design modules' models (``k4tq_sizing``: the drive maps, the tandem lattice, each wing's NeuralFoil polar, the
stall rule, the transition corridor; the CAD masses and inertia). Labels: ESTIMATE (the strip and rigid-body
formulas), CALC (the drive maps and the corridor; MOCK where the BEMT is coarsened under the mocks).

- **climb**: on the wings — the rate and angle of climb against speed from the thrust left over the drag (full and 80 %
  of the drive's current-limited thrust); vertically (tilt 90) — the climb rate where the four drives' thrust at that
  axial inflow meets the weight and the fuselage's broadside drag; the mission's climb to 120 m both ways;
- **descent**: the glide with every motor stopped (blades folded) — best glide ratio, least sink; the steepest unpowered
  dive under V_ne; the powered descent through the corridor (tilt, thrust, power), stepped down from level flight to
  the steepest balanced path at each speed; the vertical descent's
  vortex-ring band and the time and energy to come down 120 m against a spiral on the wings;
- **manoeuvre**: the sustained (thrust-limited) and instantaneous (CL max-limited) turn, capped at the limit load factor;
  roll by the wings' differential incidence (strip theory: the steady rate p = δ V Σ∫c y dy / Σ∫c y² dy, the roll
  mode's time constant from the damping), pitch by the front/rear incidence, yaw by differential thrust; in the hover
  roll / pitch from the thrust margin, yaw from the wings' differential tilt (the thrust vectors turned).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import k4tq
import k4tq_sizing as sz

G, RHO = k4tq.G, k4tq.RHO
V_NE = 32.0                       # m/s: the never-exceed speed (the structure and the actuators sized to 30 m/s, +7 %)
ALT_CLIMB = 120.0
ROLL_DELTA_DEG = 5.0              # the wings' differential incidence for the roll (the front wing's stall margin at the cruise cl)
PITCH_DELTA_DEG = 3.0
YAW_DT = 0.30                     # ± of the trim thrust, differential left / right (cruise)
HOVER_TILT_DELTA_DEG = 10.0       # the wings' differential tilt for the hover's yaw
DESIGNS = ("A2", "F2")


# ================================================================================================= helpers
def drag(veh: dict, V: float, CL: float, *, n_running: int = 4) -> float:
    """The drag [N] at airspeed V and (vertical) CL: each wing's polar at its share of the lift, the rest of the build-up,
    the lattice's induced drag; ``n_running`` motors turning (the others' blades folded)."""
    L = veh["L"]
    d = sz.drag_buildup(L, n_running=n_running)
    prof = sz.profile_model(veh)
    q = 0.5 * RHO * V * V
    CD = prof(V, CL, d) if prof else d["Cd0"] + sz.k_total(veh) * CL ** 2
    return q * L["S_ref"] * CD


def thrust_available(veh: dict, V: float, fraction: float = 1.0) -> float:
    return 4 * fraction * sz.max_thrust(veh["drive"], V)


def _cl_max(veh):
    return veh["stall"]["CL max (aircraft)"]


# ================================================================================================= climb
def climb(veh: dict, speeds=np.arange(18.0, 30.1, 1.0), fractions=(1.0, 0.8)) -> pd.DataFrame:
    """Wing-borne climb: at each speed (above 1.15 × the stall) the excess thrust T − D gives sin γ = (T − D)/W and the
    rate V sin γ (lift = W cos γ)."""
    W, S = veh["W"], veh["L"]["S_ref"]
    rows = []
    for V in speeds:
        q = 0.5 * RHO * V * V
        if W / (q * S) > _cl_max(veh) / 1.15 ** 2:
            continue
        for fr in fractions:
            gam = 0.0
            for _ in range(20):                                       # lift = W cos γ
                CL = W * math.cos(gam) / (q * S)
                sg = float(np.clip((thrust_available(veh, V, fr) - drag(veh, V, CL)) / W, -1.0, 1.0))
                gam = math.asin(sg)
            rows.append({"V [m/s]": V, "thrust": f"{fr:.0%}", "climb angle [deg]": math.degrees(gam), "rate of climb [m/s]": V * math.sin(gam)})
    df = pd.DataFrame(rows)
    df.attrs["label"] = sz.label("CALC")
    return df


def vertical_climb(veh: dict, fraction: float = 1.0) -> dict:
    """Tilt 90, the fuselage level: the climb rate Vc where 4 × T_max(Vc) × fraction = W + the fuselage's and the legs'
    broadside drag (the wings edge-on to the flow); the power at a 3 m/s climb."""
    L, W, p = veh["L"], veh["W"], veh["p"]
    CdA = p["fuselage_length"] * p["fuselage_width"] * 1.0 + 0.05
    m = veh["drive"]
    lo, hi = 0.0, 25.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if thrust_available(veh, mid, fraction) - W * sz.DOWNLOAD - 0.5 * RHO * mid * mid * CdA > 0:
            lo = mid
        else:
            hi = mid
    T3 = (W * sz.DOWNLOAD + 0.5 * RHO * 9.0 * CdA) / 4
    dp = sz.drive_point(m, 3.0, T3)
    P3 = 4 * dp["electrical_W"] + k4tq.avionics_power(p) if dp["feasible"] else math.inf
    return {"max vertical climb [m/s]": lo, "at 80 % thrust [m/s]": vertical_climb(veh, 0.8)["max vertical climb [m/s]"] if fraction == 1.0 else None,
            "power climbing at 3 m/s [W]": P3, "label": sz.label("CALC")}


def mission_climb(veh: dict, cl: pd.DataFrame | None = None) -> dict:
    """120 m: vertically at 3 m/s (tilt 90), against on the wings at the best rate (80 % thrust) — time and energy."""
    h = vertical_climb(veh)
    t_v = ALT_CLIMB / 3.0
    E_v = h["power climbing at 3 m/s [W]"] * t_v / 3600
    cl = cl if cl is not None else climb(veh)
    c8 = cl[cl["thrust"] == "80%"]
    best = c8.loc[c8["rate of climb [m/s]"].idxmax()]
    V, roc = float(best["V [m/s]"]), float(best["rate of climb [m/s]"])
    t_w = ALT_CLIMB / roc
    T = 0.8 * thrust_available(veh, V)
    dp = sz.drive_point(veh["drive"], V, T / 4)
    E_w = (4 * dp["electrical_W"] + k4tq.avionics_power(veh["p"])) * t_w / 3600
    return {"vertical: time [s]": t_v, "vertical: energy [Wh]": E_v, "on the wings: best rate [m/s]": roc, "at [m/s]": V, "on the wings: time [s]": t_w,
            "on the wings: energy [Wh]": E_w, "label": sz.label("CALC")}


# ================================================================================================= descent
def glide(veh: dict, speeds=np.arange(18.0, 33.1, 1.0)) -> pd.DataFrame:
    """Every motor stopped, the blades folded: γ from D / L, the sink rate, the glide ratio."""
    W, S = veh["W"], veh["L"]["S_ref"]
    rows = []
    for V in speeds:
        q = 0.5 * RHO * V * V
        if W / (q * S) > _cl_max(veh) / 1.15 ** 2:
            continue
        gam = 0.05
        for _ in range(20):
            CL = W * math.cos(gam) / (q * S)
            gam = math.asin(min(drag(veh, V, CL, n_running=0) / W, 1.0))
        rows.append({"V [m/s]": V, "glide angle [deg]": math.degrees(gam), "sink [m/s]": V * math.sin(gam), "glide ratio": 1 / math.tan(gam)})
    df = pd.DataFrame(rows).set_index("V [m/s]")
    df.attrs["label"] = sz.label("CALC")
    return df


def steepest_dive(veh: dict, V_ne: float = V_NE) -> dict:
    """The steepest unpowered dive that holds V_ne: D(V_ne) = W sin γ with lift W cos γ (blades folded)."""
    W, S = veh["W"], veh["L"]["S_ref"]
    q = 0.5 * RHO * V_ne ** 2
    gam = 0.1
    for _ in range(30):
        gam = math.asin(min(drag(veh, V_ne, W * math.cos(gam) / (q * S), n_running=0) / W, 1.0))
    return {"V_ne [m/s]": V_ne, "steepest dive [deg]": math.degrees(gam), "sink [m/s]": V_ne * math.sin(gam),
            "note": "steeper needs a brake: the wings tilted up (the corridor's descent) or the propellers' drag", "label": sz.label("CALC")}


def powered_descent(veh: dict, speeds=(12.0, 18.0, 24.0, 28.0), step: float = 1.0, gamma_min: float = -20.0) -> pd.DataFrame:
    """The corridor's balance in a descent (``k4tq_sizing.corridor_point`` with γ), by continuation: each speed starts
    from its level-flight state and steps the path down by ``step`` degrees, each γ from the last balanced one, until no
    balanced state is left (the thrust has reached zero: steeper needs a brake the model has not got — windmilling
    props — or a lower speed, where the tilted wings and the props carry the weight)."""
    rows = []
    for V in speeds:
        x0 = None
        g = 0.0
        while g >= gamma_min - 1e-9:
            r = sz.corridor_point(veh, V, g, x0=x0)
            xr = r.pop("_x", None)
            ok = bool(r["feasible"])
            rows.append({"V [m/s]": V, "γ [deg]": g, "feasible": ok, "tilt [deg]": r["tilt [deg]"] if ok else math.nan,
                         "rear differential [deg]": r["rear differential [deg]"] if ok else math.nan,
                         "thrust total [N]": r["thrust total [N]"] if ok else math.nan, "electrical [W]": r["electrical [W]"] if ok else math.nan,
                         "sink [m/s]": -V * math.sin(math.radians(g)), "stalled span front": r["stalled span front"] if ok else math.nan})
            if not ok:
                break
            x0 = xr
            g -= step
    df = pd.DataFrame(rows).set_index(["V [m/s]", "γ [deg]"])
    df.attrs["label"] = sz.label("CALC")
    return df


def steepest_powered(pdsc: pd.DataFrame) -> pd.Series:
    """The steepest balanced path [deg] at each speed of ``powered_descent``."""
    ok = pdsc[pdsc["feasible"]].reset_index()
    return ok.groupby("V [m/s]")["γ [deg]"].min()


def vertical_descent(veh: dict, gl: pd.DataFrame | None = None) -> dict:
    """The hover's vertical descent: the vortex-ring band (0.3–1.5 vᵢ: thrust and control lost) — so come down at ≤ 0.3 vᵢ
    or with forward speed; 120 m that way on hover power, against a spiral on the wings at the least sink, motors at idle."""
    h = sz.hover(veh)
    vi = h["induced velocity [m/s]"]
    v_safe = 0.3 * vi
    t_v = ALT_CLIMB / v_safe
    E_v = h["electrical [W]"] * t_v / 3600
    gl = gl if gl is not None else glide(veh)
    ms = gl["sink [m/s]"].min()
    t_g = ALT_CLIMB / ms
    E_g = k4tq.avionics_power(veh["p"]) * t_g / 3600
    return {"induced velocity [m/s]": vi, "vortex-ring band [m/s]": (0.3 * vi, 1.5 * vi), "safe vertical descent [m/s]": v_safe,
            "120 m vertically: time [s]": t_v, "energy [Wh]": E_v, "120 m spiralling on the wings: time [s]": t_g, "energy (avionics only) [Wh]": E_g,
            "label": sz.label("CALC")}


# ================================================================================================= manoeuvre
def turns(veh: dict, speeds=np.arange(20.0, 30.1, 1.0), fraction: float = 1.0) -> pd.DataFrame:
    """Sustained (thrust = drag at load factor n) and instantaneous (CL max) turns, both capped at ``N_LIMIT``: the load
    factor, the turn rate, the radius."""
    W, S = veh["W"], veh["L"]["S_ref"]
    rows = []
    for V in speeds:
        q = 0.5 * RHO * V * V
        if W / (q * S) > _cl_max(veh) / 1.15 ** 2:
            continue
        n_inst = min(q * S * _cl_max(veh) / W, sz.N_LIMIT)
        T = thrust_available(veh, V, fraction)
        lo, hi = 1.0, n_inst
        if drag(veh, V, W / (q * S)) > T:
            n_s = math.nan
        else:
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if drag(veh, V, mid * W / (q * S)) < T:
                    lo = mid
                else:
                    hi = mid
            n_s = lo
        def rate(n):
            return math.degrees(G * math.sqrt(max(n * n - 1, 0.0)) / V) if n == n else math.nan
        def radius(n):
            return V * V / (G * math.sqrt(n * n - 1)) if n == n and n > 1.0 else math.inf
        rows.append({"V [m/s]": V, "sustained n": n_s, "sustained rate [deg/s]": rate(n_s), "sustained radius [m]": radius(n_s), "instantaneous n": n_inst,
                     "instantaneous rate [deg/s]": rate(n_inst), "instantaneous radius [m]": radius(n_inst)})
    df = pd.DataFrame(rows).set_index("V [m/s]")
    df.attrs["label"] = sz.label("CALC")
    return df


def _strips(L: dict, which: str, n: int = 40):
    w = L["wings"][which]
    y = np.linspace(w["y0"], w["y1"], n)
    return y, np.array([w["c_at"](yy) for yy in y])


def roll(veh: dict, V: float | None = None, delta_deg: float = ROLL_DELTA_DEG) -> dict:
    """Roll by the wings' differential incidence (left +δ, right −δ on both pairs; strip theory, the lift slope cancels in
    the steady rate): p = δ V Σ∫c y dy / Σ∫c y² dy; the roll mode's time constant τ = Ixx / |L_p| with
    L_p = −ρ V a Σ∫c y² dy; the time to bank 60° ≈ τ + 60° / p."""
    L = veh["L"]
    V = V or sz.best_speeds(sz.cruise(veh, n_running=2))["best range V [m/s]"]
    a = veh["aero"]["CLa_one_pair"]
    I1 = I2 = 0.0
    for w in ("front", "rear"):
        y, c = _strips(L, w)
        I1 += 2 * np.trapezoid(c * y, y)
        I2 += 2 * np.trapezoid(c * y * y, y)
    d = math.radians(delta_deg)
    p = d * V * I1 / I2
    Lp = RHO * V * a * I2 / 2
    tau = veh["mass"]["Ixx"] / Lp
    return {"V [m/s]": V, "δ [deg]": delta_deg, "steady roll rate [deg/s]": math.degrees(p), "roll time constant [s]": tau,
            "time to 60° [s]": tau + 60.0 / math.degrees(p), "label": "ESTIMATE"}


def pitch(veh: dict, V: float | None = None, delta_deg: float = PITCH_DELTA_DEG) -> dict:
    """Pitch by the front/rear incidence (+δ front, −δ rear): the moment about the CG from each pair's lift change at its
    MAC quarter chord, the pitch acceleration (Iyy), the load factor it can add at once (the lift change of the larger
    pair alone)."""
    L, W = veh["L"], veh["W"]
    V = V or sz.best_speeds(sz.cruise(veh, n_running=2))["best range V [m/s]"]
    q = 0.5 * RHO * V * V
    a = veh["aero"]["CLa_one_pair"]
    d = math.radians(delta_deg)
    x_cg = veh["mass"]["x_cg"]
    wf, wr = L["wings"]["front"], L["wings"]["rear"]
    dLf = q * wf["S_pair"] * a * veh["aero"]["slope_front"] * d
    dLr = q * wr["S_pair"] * a * veh["aero"]["slope_rear"] * d
    M = dLf * (x_cg - wf["x_ac_mac"]) + dLr * (wr["x_ac_mac"] - x_cg)
    return {"V [m/s]": V, "δ [deg]": delta_deg, "pitch moment [N·m]": M, "pitch acceleration [deg/s²]": math.degrees(M / veh["mass"]["Iyy"]),
            "load factor per δ (rear pair up) [g]": (dLr) / W, "label": "ESTIMATE"}


def yaw(veh: dict, V: float | None = None, dt: float = YAW_DT) -> dict:
    """Yaw by differential thrust in cruise (±dt of the trim thrust on the left / right motors)."""
    L, W = veh["L"], veh["W"]
    V = V or sz.best_speeds(sz.cruise(veh, n_running=2))["best range V [m/s]"]
    q = 0.5 * RHO * V * V
    T = drag(veh, V, W / (q * L["S_ref"])) / 4
    N = 4 * dt * T * veh["p"]["motor_y"]
    return {"V [m/s]": V, "yaw moment [N·m]": N, "yaw acceleration [deg/s²]": math.degrees(N / veh["mass"]["Izz"]), "label": "ESTIMATE"}


def hover_agility(veh: dict, tilt_delta_deg: float = HOVER_TILT_DELTA_DEG) -> dict:
    """Hover: roll and pitch accelerations from the thrust margin (each motor up to its maximum on one side, as much down
    on the other: the total thrust kept), yaw from the wings' differential tilt (left pair forward, right pair aft: the
    thrust vectors turned) and the rotor torques' margin."""
    L, W = veh["L"], veh["W"]
    h = sz.hover(veh)
    m = veh["drive"]
    Tmax = sz.max_thrust(m, 0.0)
    Tf, Tr = h["thrust front each [N]"], h["thrust rear each [N]"]
    dT = min(Tmax - max(Tf, Tr), min(Tf, Tr))
    y = veh["p"]["motor_y"]
    xf, xr = L["thrust_x"]["front"], L["thrust_x"]["rear"]
    Mroll = 4 * dT * y
    Mpitch = 2 * dT * (xr - xf)
    T_mean = W * sz.DOWNLOAD / 4
    Myaw = 4 * T_mean * math.sin(math.radians(tilt_delta_deg)) * y
    I = veh["mass"]
    return {"thrust margin per motor [N]": dT, "roll acceleration [deg/s²]": math.degrees(Mroll / I["Ixx"]),
            "pitch acceleration [deg/s²]": math.degrees(Mpitch / I["Iyy"]), f"yaw acceleration at ±{tilt_delta_deg:.0f}° tilt [deg/s²]": math.degrees(Myaw / I["Izz"]),
            "label": "ESTIMATE"}


# ================================================================================================= all of it
def study(designs=DESIGNS) -> dict:
    out = {}
    for v in designs:
        veh = sz.vehicle(v)
        cl, gl = climb(veh), glide(veh)
        out[v] = {"veh": veh, "climb": cl, "vertical climb": vertical_climb(veh), "mission climb": mission_climb(veh, cl), "glide": gl,
                  "steepest dive": steepest_dive(veh), "powered descent": powered_descent(veh), "vertical descent": vertical_descent(veh, gl),
                  "turns": turns(veh), "roll": roll(veh), "pitch": pitch(veh), "yaw": yaw(veh), "hover agility": hover_agility(veh)}
    return out


def summary(res: dict) -> pd.DataFrame:
    rows = {}
    for v, r in res.items():
        cl8 = r["climb"][r["climb"]["thrust"] == "80%"]
        gl = r["glide"]
        pdsc = r["powered descent"]
        steep = steepest_powered(pdsc)
        tu = r["turns"]
        rows[f"K4TQ-{v}" + (f" ({k4tq.NICKNAME[v]})" if k4tq.NICKNAME.get(v) else "")] = {
            "mass [kg]": r["veh"]["mass"]["mass_kg"],
            "best climb rate on the wings (80 %) [m/s]": float(cl8["rate of climb [m/s]"].max()),
            "best climb angle (80 %) [deg]": float(cl8["climb angle [deg]"].max()),
            "max vertical climb [m/s]": r["vertical climb"]["max vertical climb [m/s]"],
            "120 m vertically / on the wings [Wh]": f"{r['mission climb']['vertical: energy [Wh]']:.1f} / {r['mission climb']['on the wings: energy [Wh]']:.1f}",
            "best glide ratio (motors off)": float(gl["glide ratio"].max()), "least sink [m/s]": float(gl["sink [m/s]"].min()),
            "steepest unpowered dive at V_ne [deg]": r["steepest dive"]["steepest dive [deg]"],
            **{f"steepest powered descent at {V:.0f} m/s [deg]": float(steep.get(V, math.nan)) for V in pdsc.index.get_level_values(0).unique()},
            "safe vertical descent [m/s]": r["vertical descent"]["safe vertical descent [m/s]"],
            "best sustained turn [deg/s]": float(tu["sustained rate [deg/s]"].max()), "tightest sustained radius [m]": float(tu["sustained radius [m]"].min()),
            "best instantaneous turn [deg/s]": float(tu["instantaneous rate [deg/s]"].max()),
            "roll rate (±5°) [deg/s]": r["roll"]["steady roll rate [deg/s]"], "time to 60° bank [s]": r["roll"]["time to 60° [s]"],
            "pitch acceleration (±3°) [deg/s²]": r["pitch"]["pitch acceleration [deg/s²]"], "yaw acceleration (±30 % thrust) [deg/s²]": r["yaw"]["yaw acceleration [deg/s²]"],
            "hover roll / pitch / yaw [deg/s²]": "{:.0f} / {:.0f} / {:.0f}".format(*[r["hover agility"][k] for k in r["hover agility"] if "acceleration" in k]),
        }
    df = pd.DataFrame(rows).T
    df.attrs["label"] = sz.label("CALC")
    return df


def figure(res: dict, path) -> object:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), dpi=110)
    for v, r in res.items():
        lab = f"K4TQ-{v}" + (f" ({k4tq.NICKNAME[v]})" if k4tq.NICKNAME.get(v) else "")
        c = r["climb"][r["climb"]["thrust"] == "80%"]
        ax[0, 0].plot(c["V [m/s]"], c["rate of climb [m/s]"], "o-", ms=3, label=lab)
        ax[0, 1].plot(r["glide"].index, r["glide"]["sink [m/s]"], "o-", ms=3, label=lab)
        t = r["turns"]
        ax[1, 0].plot(t.index, t["sustained rate [deg/s]"], "o-", ms=3, label=f"{lab} sustained")
        ax[1, 0].plot(t.index, t["instantaneous rate [deg/s]"], "--", lw=1)
        pdsc = r["powered descent"]
        ok = pdsc[pdsc["feasible"]].reset_index()
        for V, g in ok.groupby("V [m/s]"):
            ax[1, 1].plot(g["γ [deg]"], g["thrust total [N]"], "o-", ms=3, label=f"{lab} at {V:.0f} m/s")
    ax[0, 0].set(xlabel="airspeed [m/s]", ylabel="rate of climb [m/s]", title="climb on the wings, 80 % thrust")
    ax[0, 1].set(xlabel="airspeed [m/s]", ylabel="sink [m/s]", title="glide, every motor stopped (blades folded)")
    ax[1, 0].set(xlabel="airspeed [m/s]", ylabel="turn rate [deg/s]", title="turns: sustained (full thrust) — and instantaneous (dashed)")
    ax[1, 1].set(xlabel="flight path γ [deg]", ylabel="thrust total [N]", title="powered descent through the corridor: the thrust left to hold the speed")
    for a in ax.flat:
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
