"""kestrel_fast — compare Kestrel designs in a minute: Zephyr sections, Boreas points, rules of thumb for the structure.

A **standalone** comparator for the early design loop (notebook ``44_kestrel_fast``): a list of ``Candidate`` designs
(span, chord, gap, section, propeller, battery, bracing, spar) is turned into one row of numbers each — mass, wing
loading, stall and cruise speeds, hover power and thrust-to-weight, cruise power, endurance and range, the
thrust-limited top speed, the static margin, the spar's stress and deflection at the manoeuvre load, the posts'
buckling, the first bending mode — with nothing slower than a NeuralFoil polar (~2 s per section and Reynolds number,
cached) and a handful of blade-element points. **No CFD, no FEA, no transition integration**: the numbers are
*estimates* (the methods are named in ``METHODS``), good for ranking designs against each other, not for certifying
one. Where the full study disagrees with it (the CFD's CL max against NeuralFoil's, say) the full study wins.

The structure is closed form: the spar as a beam under the elliptic lift at ``n`` g (root moment, bending stress,
tip deflection) for a carbon tube or a printed box, the posts as Euler columns under the lift share they carry, the
wing pair's first bending mode as a cantilever estimate — and ``kestrel_structure.frame`` (the plane frame) beside it
where the design is the biplane with the standard tubes. A "very crude FEA" would not be faster than these and would
not be more right at this stage; the real FEA is the ``structure_fea`` workflow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

import kestrel
import kestrel_flight as kf
import kestrel_structure as kst
import kestrel_systems as ks

G, RHO, NU = ks.G, ks.RHO0, kf.NU
V_CRUISE = kf.V_CRUISE
N_MANOEUVRE = 4.0
ULTIMATE = 1.5
AVIONICS_W = 13.0
BATTERY_WH_PER_KG = 4 * 3.7 * 3.3 / 0.360            # the 4S 3300's specific energy, for other capacities

METHODS = {
    "mass": "kestrel_systems.mass_table on the candidate's layout (component data + assumed densities), the battery and propellers scaled",
    "section": "Zephyr / NeuralFoil at the candidate's chord Reynolds number, n_crit 6; cl_max × 0.9 for a built-up wing",
    "lift and margin": "the vortex lattice on the two wings (kestrel_flight.aero) and the neutral point with the pod and the propellers",
    "drag": "kestrel_flight.drag_buildup: skin friction with form factors, the struts, 10 % interference; the section's drag on the wings' share",
    "propulsion": "Boreas BEMT (generic blade section, prop_lab's calibration when on file) + the motor's first-order model at a few points",
    "structure": "beam theory on the spar (elliptic lift at n g, ultimate × 1.5), Euler buckling of the posts, a cantilever first-mode estimate; the plane frame beside it",
}

SPARS = {"carbon 10x8 tube": {"kind": "tube", "od": 0.010, "id": 0.008, "E": 80e9, "allow": 400e6 * 0.8, "rho": 1550.0},
         "carbon 12x10 tube": {"kind": "tube", "od": 0.012, "id": 0.010, "E": 80e9, "allow": 400e6 * 0.8, "rho": 1550.0},
         "printed PETG box 20x14x2": {"kind": "box", "b": 0.020, "h": 0.014, "t": 0.002, "E": 2.0e9, "allow": 45e6 * 0.7, "rho": 1250.0},
         "printed PETG box 24x18x2.4": {"kind": "box", "b": 0.024, "h": 0.018, "t": 0.0024, "E": 2.0e9, "allow": 45e6 * 0.7, "rho": 1250.0}}
PROPS = {"9x6": (9.0, 6.0, 0.014), "10x6": (10.0, 6.0, 0.018), "10x7": (10.0, 7.0, 0.018), "11x7": (11.0, 7.0, 0.022)}
MOTORS = {"2216 880KV": dict(kv=880.0, R_ohm=0.10, I0_A=0.5, I_max_A=25.0, mass_kg=0.066),
          "2814 1000KV": dict(kv=1000.0, R_ohm=0.06, I0_A=0.8, I_max_A=40.0, mass_kg=0.105)}


@dataclass
class Candidate:
    name: str
    span: float = 1.0                 # m
    chord: float = 0.14               # m
    gap: float = 0.28                 # m
    section: str = "NACA 23112"
    reflex_k21: float | None = 0.03
    prop: str = "9x6"
    motor: str = "2216 880KV"
    battery_ah: float = 3.3
    cells: int = 4
    bracing: str = "posts"            # x | posts | none
    strut_fairing: bool = True
    elevons: bool = True
    spar: str = "carbon 10x8 tube"
    payload_kg: float = 0.15
    note: str = ""

    def params(self) -> dict:
        """Kestrel CAD parameters for the candidate: the motors as far out as the tips allow, the gap at least the
        propeller's diameter plus the clearance, the feet behind the pod."""
        d_in, _, _ = PROPS[self.prop]
        D = d_in * 25.4
        gap = max(self.gap * 1e3, D + 30.0)
        span = self.span * 1e3
        motor_y = span / 2 - 34.0 / 2 - 5.0
        p = dict(span=span, chord=self.chord * 1e3, gap=gap, prop_diameter=D, motor_y=motor_y, bracing=self.bracing, strut_fairing=self.strut_fairing,
                 elevons=self.elevons, elevon_inner_y=max(250.0, motor_y - D / 2 - 20.0) if self.elevons else 275.0,
                 prop_x=max(125.0, 0.5 * D - 35.0 + 50.0), x_feet=max(540.0, 200.0 + self.chord * 1e3 + 200.0))
        p["pod_length"] = max(470.0, 200.0 + self.chord * 1e3 + 130.0)
        return kestrel.resolve(p)


# ================================================================================================= the pieces
@lru_cache(maxsize=64)
def section_polar(section: str, reflex_k21, Re_k: int, n_crit: float = 6.0) -> dict:
    """NeuralFoil's summary of the section at Re ≈ Re_k × 1000 (cached: the slow step)."""
    from vegeta import zephyr
    digits = section.replace("NACA", "").strip()
    s = zephyr.naca(digits, reflex_k21=reflex_k21) if reflex_k21 is not None else zephyr.naca(digits)
    p = zephyr.analyze(s, np.arange(-6.0, 20.0, 1.0), Re_k * 1000.0, n_crit=n_crit)
    return p.summary(0.35)


def propulsion(c: Candidate):
    from vegeta import boreas
    d_in, p_in, m_prop = PROPS[c.prop]
    d, pitch = boreas.inches(d_in, p_in)
    k = d_in / 9.0
    prop = boreas.Propeller.from_pitch(c.prop, d, pitch, blades=2, chord_root_m=0.014 * k, chord_max_m=0.021 * k, chord_tip_m=0.008 * k, mass_kg=m_prop, rotor_mass_kg=m_prop + 0.030)
    m = MOTORS[c.motor]
    motor = boreas.Motor(c.motor, kv_rpm_per_volt=m["kv"], resistance_ohm=m["R_ohm"], no_load_current_a=m["I0_A"], max_current_a=m["I_max_A"], mass_kg=m["mass_kg"])
    bat = boreas.Battery(f"{c.cells}S {c.battery_ah:g} Ah", cells=c.cells, capacity_ah=c.battery_ah, usable_fraction=0.8, mass_kg=c.battery_ah * c.cells * 3.7 / BATTERY_WH_PER_KG)
    return boreas.Propulsion(prop, ks.blade_airfoil(), motor, bat, rho=RHO)


def _full(sys_, V: float):
    """Full-throttle thrust at inflow V, held at the motor's current limit when it would exceed it."""
    full = sys_.at_throttle(1.0, V)
    if full.current_limited:
        lo, hi = 0.3, 1.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if sys_.at_throttle(mid, V).current > sys_.motor.max_current_a:
                hi = mid
            else:
                lo = mid
        full = sys_.at_throttle(lo, V)
    return full


def spar_rules(c: Candidate, *, W: float, n: float = N_MANOEUVRE) -> dict:
    """The spar as a beam: each wing carries half the lift, elliptic along the span; the root bending moment of one
    half-wing of an elliptic load with total L/2 over the semi-span b/2 is M = (L/2)·(4/3π)·(b/2)·(1/2)·… — exactly
    M_root = L_half · ȳ with ȳ = 4b/(3π·2)·… we use the elliptic centroid 4/(3π) of the semi-span. Stress σ = M c / I,
    tip deflection of a cantilever under the elliptic load ≈ 0.0725 · q_0 · (b/2)⁴ / (E I)·… taken as the uniform
    equivalent 5/384 ÷ cantilever form: δ ≈ L_half (b/2)³ / (8 E I) × 1.2. Ultimate × 1.5 on the stress."""
    s = SPARS[c.spar]
    b2 = c.span / 2
    L_half = n * W / 2 / 2                          # one wing's half: a quarter of n·W
    M = L_half * (4.0 / (3.0 * math.pi)) * b2        # the elliptic half-load's centroid from the root
    if s["kind"] == "tube":
        I = math.pi / 64 * (s["od"] ** 4 - s["id"] ** 4)
        cmax = s["od"] / 2
        area = math.pi / 4 * (s["od"] ** 2 - s["id"] ** 2)
    else:
        b, h, t = s["b"], s["h"], s["t"]
        I = (b * h ** 3 - (b - 2 * t) * (h - 2 * t) ** 3) / 12
        cmax = h / 2
        area = b * h - (b - 2 * t) * (h - 2 * t)
    sigma_u = ULTIMATE * M * cmax / I
    delta = 1.2 * L_half * b2 ** 3 / (8 * s["E"] * I)
    m_spar = area * s["rho"] * c.span
    # the pair's first bending mode: a cantilever semi-span with the wing's mass per length and the motor at ~0.9 b/2
    mw = (ks.WING_AREAL_KG_M2 * c.chord + area * s["rho"]) * 1.0
    m_tip = ks.COMPONENTS["motor"]["mass_kg"] + PROPS[c.prop][2] + ks.NACELLE_KG + ks.COMPONENTS["esc"]["mass_kg"]
    k_tip = 3 * s["E"] * I / b2 ** 3
    f1 = math.sqrt(k_tip / (m_tip + 0.24 * mw * b2)) / (2 * math.pi)
    return {"spar": c.spar, "root moment, limit [N·m]": M, "spar stress, ultimate [MPa]": sigma_u / 1e6, "spar allowable [MPa]": s["allow"] / 1e6,
            "spar reserve (ultimate)": s["allow"] / sigma_u, "tip deflection, limit [mm]": delta * 1e3, "tip deflection / semi-span [%]": delta / b2 * 100,
            "spar mass, both wings [kg]": 2 * m_spar, "first bending mode, one wing cantilever [Hz]": f1}


def post_rules(c: Candidate, *, W: float, n: float = N_MANOEUVRE) -> dict:
    """The posts as Euler columns: in the 75 % wing case the post carries the lift the two wings do not share,
    ~0.25 · n W / 2 per side; pin-ended, length = gap."""
    od, idd = kst.TUBES["post"]
    I = math.pi / 64 * (od ** 4 - idd ** 4)
    L = max(c.gap, (PROPS[c.prop][0] * 25.4 + 30.0) / 1e3)
    P_cr = math.pi ** 2 * kst.CARBON["E"] * I / L ** 2
    F = 0.25 * n * W / 2
    return {"post length [m]": L, "post load, 75 % wing, limit [N]": F, "post Euler load [N]": P_cr, "post buckling margin": P_cr / (ULTIMATE * F) if c.bracing != "none" else float("nan")}


# ================================================================================================= the evaluation
def evaluate(c: Candidate, *, n_crit: float = 6.0) -> dict:
    p = c.params()
    rules = kestrel.Kestrel4Q.checks(p)
    L = kestrel.layout(p)
    # mass: the standard table on this layout, then the battery, the propellers and the spar corrected
    table = ks.mass_table(p)
    ci = ks.cg_inertia(table)
    m_bat0 = ks.COMPONENTS["battery"]["mass_kg"]
    m_bat = c.battery_ah * c.cells * 3.7 / BATTERY_WH_PER_KG
    m_prop0, m_prop = ks.COMPONENTS["propeller"]["mass_kg"], PROPS[c.prop][2]
    m_motor0, m_motor = ks.COMPONENTS["motor"]["mass_kg"], MOTORS[c.motor]["mass_kg"]
    m_spar0 = 2 * ks.SPAR_KG_M * c.span
    sr = spar_rules(c, W=1.0)                      # for the spar mass only (W-independent)
    m = ci["mass_kg"] + (m_bat - m_bat0) + 4 * (m_prop - m_prop0) + 4 * (m_motor - m_motor0) + (sr["spar mass, both wings [kg]"] - m_spar0) + (c.payload_kg - 0.15)
    W = m * G
    S = L["S_ref"]
    # section and polar
    Re = V_CRUISE * c.chord / NU
    sec = section_polar(c.section, c.reflex_k21, int(round(Re / 1000)), n_crit)
    base = section_polar("NACA 0012", None, int(round(Re / 1000)), n_crit)
    a = kf.aero(p)
    d = kf.drag_buildup(p, V=V_CRUISE, faired=bool(c.strut_fairing))
    pol0 = kf.polar(p, a=a, d=d)
    cl_max = 0.9 * sec["cl_max"]
    cd0 = d["Cd0"] - d["Cd0_wings"] + d["Cd0_wings"] * sec["cd_design"] / base["cd_design"]
    k = pol0.k
    np_ = kf.neutral_point(p, x_cg=ci["x_cg"], a=a)
    Vs = math.sqrt(2 * W / (RHO * S * cl_max))
    q16 = 0.5 * RHO * V_CRUISE ** 2
    CL16 = W / (q16 * S)
    D16 = q16 * S * (cd0 + k * CL16 ** 2)
    # propulsion points
    sys_ = propulsion(c)
    hov = sys_.for_thrust(W / 4, 0.0)
    T_static = _full(sys_, 0.0).thrust
    cru = sys_.for_thrust(max(D16 / 4, 0.3), V_CRUISE)
    P16 = 4 * cru.electrical_power + AVIONICS_W
    Ewh = c.cells * 3.7 * c.battery_ah * 0.8
    # top speed: full thrust against drag on a coarse speed grid (the drag with CL = W/qS)
    V_max = float("nan")
    Vg = np.arange(14.0, 45.0, 2.0)
    prev = None
    for V in Vg:
        q = 0.5 * RHO * V * V
        CL = W / (q * S)
        Dv = q * S * (cd0 + k * CL ** 2)
        Tv = 4 * _full(sys_, V).thrust
        if prev is not None and (prev[0] - prev[1]) > 0 >= (Tv - Dv):
            V_max = prev[2] + (prev[0] - prev[1]) / ((prev[0] - prev[1]) - (Tv - Dv)) * 2.0
            break
        prev = (Tv, Dv, V)
    # structure
    sr = spar_rules(c, W=W)
    pr = post_rules(c, W=W)
    frame = None
    try:
        if c.spar == "carbon 10x8 tube":
            f = kst.frame(p, n=N_MANOEUVRE, mass_kg=m, bracing={"x": "braced", "posts": "posts", "none": "none"}[c.bracing], upper_share=0.75)
            frame = {"plane frame spar stress, limit [MPa]": f["spar stress [MPa]"], "plane frame spar reserve": f["spar reserve"]}
    except Exception:
        frame = None
    out = {"candidate": c.name, "note": c.note, "layout rules broken": len(rules),
           "span [m]": c.span, "chord [m]": c.chord, "gap [m]": L["gap"], "wing area S_ref [m²]": S, "aspect ratio (one wing)": c.span / c.chord,
           "section": sec["section"], "propeller": c.prop, "motor": c.motor, "battery": f"{c.cells}S {c.battery_ah:g} Ah", "bracing": c.bracing, "struts": "faired" if c.strut_fairing else "round",
           "mass [kg]": m, "wing loading [N/m²]": W / S, "CG x [mm]": ci["x_cg"] * 1e3, "static margin [chords]": np_["static_margin"],
           "Re (chord, cruise)": Re, "section cl_max (NeuralFoil)": sec["cl_max"], "section cd at cl 0.35": sec["cd_design"], "section cm0": sec["cm0"], "section confidence": sec["confidence_min_attached"],
           "CL max (×0.9)": cl_max, "stall speed [m/s]": Vs, "cruise / stall": V_CRUISE / Vs, "Cd0": cd0, "k (induced)": k, "cruise CL": CL16, "L/D at cruise": W / D16,
           "hover thrust per rotor [N]": W / 4, "hover rpm": hov.rpm, "hover electrical [W]": 4 * hov.electrical_power + AVIONICS_W, "hover current per motor [A]": hov.current,
           "static thrust per rotor [N]": T_static, "hover T/W": 4 * T_static / W, "cruise thrust per rotor [N]": D16 / 4, "cruise electrical [W]": P16,
           "usable energy [Wh]": Ewh, "endurance at cruise [min]": Ewh / P16 * 60, "range at cruise [km]": Ewh / P16 * V_CRUISE * 3.6, "hover time [min]": Ewh / (4 * hov.electrical_power + AVIONICS_W) * 60,
           "V max, thrust-limited [m/s]": V_max, "pitch speed [m/s]": PROPS[c.prop][1] * 0.0254 * MOTORS[c.motor]["kv"] * c.cells * 3.7 / 60,
           **sr, **pr}
    if frame:
        out.update(frame)
    return out


def compare(cands: list, *, progress: bool = True, **kw):
    """One row per candidate. ``progress``: a tqdm bar over the candidates (the first ones are the slow ones: each new
    section and Reynolds number is a NeuralFoil polar, cached after that)."""
    import pandas as pd
    it = cands
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(cands, desc="candidates", unit="candidate")
    rows = []
    for c in it:
        if progress:
            it.set_postfix_str(f"{c.name}: {c.section}", refresh=True)
        rows.append(evaluate(c, **kw))
    df = pd.DataFrame(rows).set_index("candidate")
    return df


KEY_COLUMNS = ["mass [kg]", "wing loading [N/m²]", "static margin [chords]", "CL max (×0.9)", "stall speed [m/s]", "cruise / stall", "Cd0", "L/D at cruise",
               "hover T/W", "hover electrical [W]", "cruise electrical [W]", "endurance at cruise [min]", "range at cruise [km]", "V max, thrust-limited [m/s]",
               "spar reserve (ultimate)", "tip deflection / semi-span [%]", "post buckling margin", "first bending mode, one wing cantilever [Hz]"]
BETTER = {"mass [kg]": "low", "wing loading [N/m²]": "low", "stall speed [m/s]": "low", "cruise / stall": "high", "Cd0": "low", "L/D at cruise": "high", "hover T/W": "high",
          "hover electrical [W]": "low", "cruise electrical [W]": "low", "endurance at cruise [min]": "high", "range at cruise [km]": "high", "V max, thrust-limited [m/s]": "high",
          "spar reserve (ultimate)": "high", "tip deflection / semi-span [%]": "low", "post buckling margin": "high", "first bending mode, one wing cantilever [Hz]": "high"}
TARGETS = {"mass [kg]": ("≤", 2.0), "hover T/W": ("≥", 2.0), "static margin [chords]": ("between", (0.05, 0.15)), "cruise / stall": ("≥", 1.3), "endurance at cruise [min]": ("≥", 30.0),
           "spar reserve (ultimate)": ("≥", 1.5), "post buckling margin": ("≥", 2.0), "tip deflection / semi-span [%]": ("≤", 5.0), "hover current per motor [A]": ("≤", 20.0), "layout rules broken": ("≤", 0)}


def ratios(df, baseline: str):
    """Every key metric as a ratio to the baseline candidate (1 = the same)."""
    cols = [c for c in KEY_COLUMNS if c in df.columns]
    r = df[cols].astype(float).div(df.loc[baseline, cols].astype(float))
    return r


def verdicts(df):
    """Which targets each candidate meets (✔ / ✘ per target, and the count)."""
    import pandas as pd
    rows = {}
    for name, r in df.iterrows():
        row = {}
        for k, (op, t) in TARGETS.items():
            v = float(r[k])
            ok = (v <= t if op == "≤" else v >= t if op == "≥" else t[0] <= v <= t[1])
            row[k] = "✔" if ok else "✘"
        row["targets met"] = sum(1 for v in row.values() if v == "✔")
        rows[name] = row
    return pd.DataFrame(rows).T


def plot_ratios(df, baseline: str, ax=None):
    import matplotlib.pyplot as plt
    r = ratios(df, baseline)
    cols = [c for c in r.columns if c != "static margin [chords]"]
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(cols))
    n = len(r)
    w = 0.8 / n
    for i, (name, row) in enumerate(r.iterrows()):
        ax.bar(x + (i - (n - 1) / 2) * w, row[cols].values, w, label=name)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c + ("  ↓" if BETTER.get(c) == "low" else "  ↑") for c in cols], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"ratio to {baseline}")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("every metric relative to the baseline (↑ higher is better, ↓ lower is better)")
    return ax


def default_candidates() -> list:
    return [
        Candidate("4Q-2 baseline", note="span 1.0, 9x6, 4S 3300: the aircraft of notebook 43"),
        Candidate("4Q-2, 5000 mAh", battery_ah=5.0, note="the same, a heavier battery"),
        Candidate("4Q-2, NACA 2412", section="NACA 2412", reflex_k21=None, note="cambered, not reflexed: trim by elevon"),
        Candidate("4Q-3 brief: 1.2 m, 10x7", span=1.2, chord=0.16, gap=0.30, prop="10x7", note="kestrel3.md: wing area +37 %, 10x7 for speed"),
        Candidate("4Q-3, 10x6", span=1.2, chord=0.16, gap=0.30, prop="10x6", note="the brief with the safer pitch"),
        Candidate("1.1 m, 0.15 chord, 10x6", span=1.1, chord=0.15, gap=0.30, prop="10x6", note="the fallback if CL max ≥ 0.75"),
        Candidate("4Q-3, printed box spar", span=1.2, chord=0.16, gap=0.30, prop="10x7", spar="printed PETG box 24x18x2.4", note="no carbon at all"),
        Candidate("4Q-3, 2814 motors, 11x7", span=1.2, chord=0.16, gap=0.32, prop="11x7", motor="2814 1000KV", note="more motor: the hover margin and the top speed"),
    ]
