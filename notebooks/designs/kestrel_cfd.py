"""Kestrel-4Q — the transitions in CFD (Aeromant / OpenFOAM) and the CFD fed back into the transition model (notebook 38).

The transition model (``kestrel_transition``) flies the aircraft through the stall on an estimated polar: attached flow
blended into a flat plate, the slipstream from momentum theory, the wings' parts treated one by one. What it cannot
know is how the four slipstreams, the pod, the bracing and the stalled wings really interact at 20–80°. The CFD
answers that at the points the aircraft actually flies:

- ``points``: from the simulated transitions (forward and back), the first time the free-stream angle of attack passes
  each of ``TRANSITION_ALPHAS`` — the airspeed, the attitude and the thrust of each rotor pair there — plus the hover
  (the wings edge-on in their own slipstream).
- ``case``: the airframe (``kestrel.Kestrel4Q`` part ``airframe``: everything but the propellers) turned to the angle of
  attack about the CG, in steady RANS k-ω SST with the **four propellers as rotor disks** (Aeromant's
  ``aircraft_quad_rotor_disks``: each disk at its own rpm — Boreas's rpm for that pair's thrust at the axial inflow —
  the blade rows and section polar from Boreas, the diagonal pairs counter-rotating). The forces on the airframe are
  in the slipstream: that is the question.
- ``sweep_specs``: the power-off airframe from 0° to 90° (``rans_ksst_external``): the bare stall, the reference the
  blown cases are read against.
- ``table`` / ``not_run_table``: the coefficients per case (CL, CD on S_ref; Cm about the CG on the chord), or the status
  ``diverged`` / ``FAILED`` / ``NOT RUN`` — a failure is a result, carried in the table, never fed to the model.
- ``summary``, ``sweep_comparison`` / ``sweep_numbers``: how the campaign went, and the bare sweep against the model's
  polar (lift slope, CL max, Cd0, static margin); ``save_records`` / ``load_records``: the solved cases to and from
  ``data/kestrel_cfd_results.json`` — the test notebook (40) writes it, the design notebook (38) plugs it in.
- ``CFDAirframe``: the CFD's coefficients minus the model's at the same points (α, the slipstream ratio μ = V / (V + Δv)),
  interpolated (linear inside the points, nearest outside) and added to the model's forces — the transition model then
  flies on the CFD where the CFD has been run and on the model elsewhere. ``kestrel_transition.make_vehicle(airframe=...)``.

Steady RANS through and beyond the stall is a time-averaged answer at best: read the residuals and the force scatter
(``Cd scatter``) before trusting a stalled case; an unsteady (URANS / DES) check of the worst point is on the to-do
list. Not run in the notebook when OpenFOAM is missing (``VEGETA_SKIP_OPENFOAM=1``): the cases are prepared and listed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import kestrel
import kestrel_control as kc
import kestrel_systems as ks

RHO, NU = ks.RHO0, 1.46e-5
TRANSITION_ALPHAS = (80.0, 60.0, 45.0, 30.0, 20.0, 15.0, 10.0)
SWEEP_ALPHAS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0)
SWEEP_ALPHAS_FINE = (0.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0)   # the campaign's: the stall's peak resolved
SWEEP_V = 10.0
RESULTS_JSON = Path(__file__).resolve().parent / "data" / "kestrel_cfd_results.json"     # the test notebook (40) writes it
QUALITY = {
    "screening": dict(cells_per_length=2.0, surface_level=4, near_level=3, wake_level=2, iterations=400, residual_target=1e-4),
    "fine": dict(cells_per_length=3.0, surface_level=5, near_level=4, wake_level=3, iterations=800, residual_target=1e-5),
}


def l_ref(p=None) -> float:
    """The template's reference length (it also sizes the domain and the mesh): a quarter of the span, as FALCO's."""
    return kestrel.layout(p)["span"] / 4


# ------------------------------------------------------------------------------------------------- the points
def points(episodes, *, alphas=TRANSITION_ALPHAS, mass_kg: float | None = None) -> list[dict]:
    """The CFD points from simulated transitions (``kestrel_transition.Episode``): for each episode and each α, the
    first sample in the transition with the free-stream α at or past it (decreasing for the forward transition,
    increasing for the back) and some airspeed (> 1.5 m/s); plus the hover if ``mass_kg`` is given."""
    out = []
    for ep in episodes:
        g = ep.log
        tr = (g["mode"] == "transition") & (g["V"] > 1.5)
        a = np.degrees(np.abs(g["alpha"]))
        used = set()
        for target in alphas:
            hit = np.where(tr & ((a <= target) if ep.kind == "forward" else (a >= target)))[0]
            if not len(hit) or int(hit[0]) in used:          # one sample past two targets: keep it once
                continue
            i = int(hit[0])
            used.add(i)
            out.append({"source": f"{ep.kind} {ep.t_tr:g} s", "kind": ep.kind, "alpha_deg": float(np.degrees(g["alpha"][i])),
                        "V": float(g["V"][i]), "T_up": float(g["T_up"][i]), "T_lo": float(g["T_lo"][i]), "t": float(g["t"][i]),
                        "target_deg": target})
    if mass_kg is not None:
        out.append({"source": "hover", "kind": "hover", "alpha_deg": 90.0, "V": 1.0, "T_up": mass_kg * ks.G / 4,
                    "T_lo": mass_kg * ks.G / 4, "t": 0.0, "target_deg": 90.0})
    return out


def flight_points(veh, *, V_cruise: float = 16.0, stall_factor: float = 1.05) -> list[dict]:
    """The two level-flight points the campaign adds to the transitions: the **cruise** (``V_cruise``, trimmed: the
    angle of attack that lifts the weight, the thrust that holds the speed, split as the trim asks) and the **stall**
    (``stall_factor`` × the stall speed at the polar's stall angle, the thrust of level flight there)."""
    import kestrel_transition as kt
    W, S = veh.weight, veh.L["S_ref"]
    pol = veh.pol
    out = []
    for source, V in (("cruise", V_cruise), ("stall", stall_factor * math.sqrt(2 * W / (RHO * S * pol.cl_max)))):
        a = kt._alpha_trim(veh, V)
        q = 0.5 * RHO * V * V
        CL = W / (q * S)
        D = q * S * (pol.cd0 + pol.k * CL ** 2)
        T = max(D / 4, veh.drive.T_min)
        out.append({"source": source, "kind": source, "alpha_deg": float(math.degrees(a)), "V": float(V), "T_up": float(T), "T_lo": float(T), "t": 0.0,
                    "target_deg": float(math.degrees(a))})
    return out


def point_name(pt: dict) -> str:
    return f"{pt['source']}: α {pt['alpha_deg']:.0f}°, {pt['V']:.1f} m/s, T {pt['T_up']:.2f}/{pt['T_lo']:.2f} N"


# ------------------------------------------------------------------------------------------------- the cases
def export_stl(workdir, alpha_deg: float, x_cg_m: float, p=None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"kestrel_airframe_a{alpha_deg:+05.1f}.stl"
    if not path.exists():
        q = kestrel.overrides(p)
        g = kestrel.Kestrel4Q().generate(**q, part="airframe", angle_of_attack_deg=float(alpha_deg), rotation_x=x_cg_m * 1000)
        g.export_stl(str(path), tolerance=0.3)
    return path


def disk_setup(pt: dict, x_cg_m: float, p=None, *, spins=kc.BASELINE_SPIN) -> dict:
    """The four disks' centres (the flow frame, metres), the common thrust axis, each disk's rpm (Boreas, for that
    pair's thrust at the axial inflow V cos α) and sense of rotation about the axis (the template's: about the thrust
    direction, so the opposite of ``kestrel_control.SPINS``'s sense about +x)."""
    L = kestrel.layout(p)
    a = pt["alpha_deg"]
    out = {}
    s = kc.SPINS[spins]
    v_ax = pt["V"] * math.cos(math.radians(a))
    rpm_up, rpm_lo = ks.rpm_for(pt["T_up"], v_ax), ks.rpm_for(pt["T_lo"], v_ax)
    for i, (mname, m) in enumerate(L["motors"].items(), start=1):
        c = kestrel.rotate((m["x"] * 1000, m["y"] * 1000, m["z"] * 1000), a, x_cg_m * 1000)
        out[f"disk{i}_center"] = [c[0] / 1000, c[1] / 1000, c[2] / 1000]
        out[f"rpm{i}"] = rpm_up if m["z"] > 0 else rpm_lo
        out[f"rotation{i}"] = float(-s[i - 1])
    ar = math.radians(a)
    out["disk_axis"] = [-math.cos(ar), 0.0, math.sin(ar)]
    return out


def blade_rows():
    """Boreas's blade (radius, twist, chord) and its section polar over −180..180° for the rotor-disk sources."""
    sys_ = ks.propulsion()
    pr, af = sys_.prop, sys_.airfoil
    alphas = np.unique(np.r_[np.arange(-180, -30, 10), np.arange(-30, 31, 1), np.arange(40, 181, 10)]).astype(float)
    cl, cd = af.coefficients(np.radians(alphas))
    return pr, [[r, b, c] for r, b, c in zip(pr.r, pr.beta_deg, pr.chord)], [[float(x), float(d), float(l)] for x, l, d in zip(alphas, cl, cd)]


def case(workdir, pt: dict, *, x_cg_m: float, z_cg_m: float = 0.0, quality: str = "screening", props: bool = True, env=None, p=None):
    """One Aeromant case (prepared, not run): the airframe at the point's α, the free stream at its airspeed, the four
    rotor disks at their rpm (``props=False``: the bare airframe, ``rans_ksst_external``)."""
    from vegeta import aeromant
    L = kestrel.layout(p)
    a = pt["alpha_deg"]
    stl = export_stl(Path(workdir) / "stl", a, x_cg_m, p)
    cor = kestrel.rotate((x_cg_m * 1000, 0.0, z_cg_m * 1000), a, x_cg_m * 1000)
    params = dict(velocity=max(pt["V"], 1.0), kinematic_viscosity=NU, density=RHO, reference_area=L["S_ref"], reference_length=l_ref(p),
                  center_of_rotation=(cor[0] / 1000, 0.0, cor[2] / 1000), **QUALITY[quality])
    template = "rans_ksst_external"
    if props:
        pr, blade, polar = blade_rows()
        params.update(disk_setup(pt, x_cg_m, p), diameter=pr.diameter, blades=pr.blades, blade=blade, polar=polar, disk_level=5)
        template = "aircraft_quad_rotor_disks"
    env = env or aeromant.OpenFOAMEnvironment.detect()
    tag = f"{pt['source'].replace(' ', '_')}_a{a:+05.1f}_V{pt['V']:.1f}" + ("" if props else "_bare") + f"_{quality}"
    return aeromant.CFDCase(template, str(stl), params, workdir=str(Path(workdir) / tag), geometry_units="mm", environment=env)


def sweep_specs(*, V: float = SWEEP_V, alphas=SWEEP_ALPHAS) -> dict:
    return {f"bare airframe, α {a:.0f}°, {V:g} m/s": {"source": "sweep", "kind": "sweep", "alpha_deg": a, "V": V, "T_up": 0.0,
                                                       "T_lo": 0.0, "t": 0.0, "target_deg": a} for a in alphas}


def specs(pts: list[dict], *, sweep: bool = True, sweep_alphas=SWEEP_ALPHAS) -> dict:
    """The cases by name and their point (``props`` False for the bare sweep)."""
    out = {point_name(pt): dict(pt, props=True) for pt in pts}
    if sweep:
        out.update({k: dict(v, props=False) for k, v in sweep_specs(alphas=sweep_alphas).items()})
    return out


def plan(workdir, pts: list[dict], *, x_cg_m: float, z_cg_m: float = 0.0, quality: str = "screening", sweep: bool = True, env=None, p=None,
         sweep_alphas=SWEEP_ALPHAS) -> dict:
    return {name: case(workdir, {k: v for k, v in s.items() if k != "props"}, x_cg_m=x_cg_m, z_cg_m=z_cg_m, quality=quality,
                       props=s["props"], env=env, p=p)
            for name, s in specs(pts, sweep=sweep, sweep_alphas=sweep_alphas).items()}


# ------------------------------------------------------------------------------------------------- the results
SANE = 10.0                      # |CL|, |CD|, |Cm| beyond this: the steady solver has diverged, not solved


def _note(r) -> str:
    msgs = [m for m in getattr(r, "messages", []) if m]
    return (msgs[-1][:160] if msgs else "")


def table(results: dict, spec: dict, p=None) -> "pd.DataFrame":
    """One row per case: the point, CL and CD on S_ref (wind axes), Cm about the CG on the chord, convergence, and the
    status — ``solved``, ``diverged`` (the solver ran but its coefficients are not finite or beyond ``SANE``: a blown
    case at a low free stream can run its iterations into 1e94), ``FAILED`` (a step failed; ``note`` says which) or
    ``NOT RUN``. A failure is a result: the table carries it and ``CFDAirframe`` / ``sweep_comparison`` use the solved
    rows only."""
    import pandas as pd
    L = kestrel.layout(p)
    k_cm = l_ref(p) / L["chord"]
    rows = {}
    for name, s in spec.items():
        r = results.get(name)
        base = {"α [deg]": s["alpha_deg"], "V [m/s]": s["V"], "T upper [N]": s["T_up"], "T lower [N]": s["T_lo"],
                "rotor disks": bool(s["props"]), "source": s["source"]}
        if r is None or not getattr(r, "ok", False):
            msg = "NOT RUN" if r is None or any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED"
            rows[name] = dict(base, status=msg, note="" if r is None else _note(r))
            continue
        m = r.metrics
        cl, cd, cm = m.get("Cl"), m.get("Cd"), m.get("Cm")
        coefs = [np.nan if v is None else float(v) for v in (cl, cd, cm)]
        sane = all(np.isfinite(v) and abs(v) <= SANE for v in coefs)
        note = "" if sane else f"diverged: |coefficients| up to {np.nanmax(np.abs(coefs)):.3g}"
        if sane and m.get("converged") is False:
            note = f"not converged in {m.get('iterations')} iterations (the residual target was not reached)"
        rows[name] = dict(base, status="solved" if sane else "diverged", CL=coefs[0], CD=coefs[1], **{"Cm (CG, chord)": coefs[2] * k_cm},
                          iterations=m.get("iterations"), converged=m.get("converged"), **{"Cd scatter (last 50)": m.get("Cd_std_lastN")},
                          cells=m.get("mesh_cells"), note=note)
    return pd.DataFrame(rows).T


def summary(cfd_table) -> dict:
    """How the campaign went: the count per status, split by rotor disks / bare, and one line to print."""
    st = cfd_table["status"].astype(str)
    disks = cfd_table["rotor disks"].astype(bool) if "rotor disks" in cfd_table.columns else st.map(lambda _: False)
    out = {"solved": int((st == "solved").sum()), "diverged": int((st == "diverged").sum()), "FAILED": int((st == "FAILED").sum()),
           "NOT RUN": int((st == "NOT RUN").sum()), "n": int(len(st)),
           "solved with rotor disks": int(((st == "solved") & disks).sum()), "solved bare": int(((st == "solved") & ~disks).sum()),
           "with rotor disks": int(disks.sum())}
    out["line"] = (f"{out['solved']} of {out['n']} cases solved ({out['solved with rotor disks']} of {out['with rotor disks']} with rotor "
                   f"disks, {out['solved bare']} bare), {out['diverged']} diverged, {out['FAILED']} failed, {out['NOT RUN']} not run")
    return out


def usable(cfd_table):
    """The solved rows (finite, sane coefficients)."""
    return cfd_table[cfd_table["status"].astype(str) == "solved"]


def not_run_table(spec: dict) -> "pd.DataFrame":
    """The planned cases with their air, rotor speeds and template, status NOT RUN."""
    import pandas as pd
    rows = {}
    for name, s in spec.items():
        v_ax = s["V"] * math.cos(math.radians(s["alpha_deg"]))
        rows[name] = {"status": "NOT RUN", "α [deg]": s["alpha_deg"], "V [m/s]": max(s["V"], 1.0), "source": s["source"], "rotor disks": bool(s["props"]),
                      "template": "aircraft_quad_rotor_disks" if s["props"] else "rans_ksst_external",
                      "rpm upper": ks.rpm_for(s["T_up"], v_ax) if s["props"] else "", "rpm lower": ks.rpm_for(s["T_lo"], v_ax) if s["props"] else "",
                      "Re (chord)": max(s["V"], 1.0) * kestrel.layout()["chord"] / NU}
    return pd.DataFrame(rows).T


def model_coefficients(veh, pt: dict) -> dict:
    """The transition model's airframe coefficients at a CFD point (wings with their slipstream and the pod; not the
    propellers' normal force — the CFD's force coefficients are the airframe's only): CL, CD on S_ref in wind axes,
    Cm about the CG on the chord, and the slipstream ratio μ."""
    import kestrel_transition as kt
    a = math.radians(pt["alpha_deg"])
    V = max(pt["V"], 1.0)
    u, w = V * math.cos(a), V * math.sin(a)
    f = kt.airframe_forces(veh, u, w, pt["T_up"], pt["T_lo"], props_normal=False)
    q = 0.5 * RHO * V * V
    S, c = veh.L["S_ref"], veh.L["chord"]
    drag = (f["Fx"] * u + f["Fz"] * w) / V                      # along the relative air (body x aft, z up)
    lift = (-f["Fx"] * w + f["Fz"] * u) / V
    dv = 0.5 * (f["slip_up"]["dv"] + f["slip_lo"]["dv"])
    return {"CL": lift / (q * S), "CD": drag / (q * S), "Cm": f["M"] / (q * S * c), "mu": V / (V + dv)}


class CFDAirframe:
    """The CFD's correction to the transition model: ΔCL, ΔCD, ΔCm (CFD − model) at the solved points, interpolated
    in (α, μ) — linear inside the points, the nearest point outside; from the bare sweep alone (μ = 1 at every point)
    linear in α — and added to the model's airframe forces."""

    def __init__(self, samples: list[dict], source: str = "disks"):
        if not samples:
            raise ValueError("no solved CFD case to build a correction from")
        self.samples = samples
        self.source = source
        X = np.array([[s["alpha"], s["mu"]] for s in samples])
        self._X = X
        self._D = np.array([[s["dCL"], s["dCD"], s["dCm"]] for s in samples])
        self._lin = None
        self._by_alpha = np.ptp(X[:, 1]) <= 1e-3 and len(samples) >= 2
        if self._by_alpha:
            o = np.argsort(X[:, 0])
            self._X, self._D = X[o], self._D[o]
        elif len(samples) >= 4 and np.ptp(X[:, 0]) > 1e-3:
            from scipy.interpolate import LinearNDInterpolator
            try:
                self._lin = LinearNDInterpolator(X, self._D)
            except Exception:                          # degenerate (collinear) points
                self._lin = None

    @staticmethod
    def records(cfd_table) -> list[dict]:
        """The solved rows as plain records (α, V, the thrusts, CL, CD, Cm, rotor disks): what ``save_records`` keeps
        and ``from_records`` rebuilds a correction from, with no dependence on the case names."""
        out = []
        for name, r in usable(cfd_table).iterrows():
            out.append({"case": str(name), "alpha_deg": float(r["α [deg]"]), "V": float(r["V [m/s]"]), "T_up": float(r["T upper [N]"]),
                        "T_lo": float(r["T lower [N]"]), "rotor_disks": bool(r["rotor disks"]), "source": str(r.get("source", "")),
                        "CL": float(r["CL"]), "CD": float(r["CD"]), "Cm": float(r["Cm (CG, chord)"]),
                        "converged": None if r.get("converged") is None or r.get("converged") != r.get("converged") else bool(r["converged"]),
                        "iterations": None if r.get("iterations") is None or r.get("iterations") != r.get("iterations") else int(r["iterations"])})
        return out

    @classmethod
    def from_records(cls, records: list[dict], veh, *, source: str = "disks"):
        """``source`` ``disks``: the blown cases (the correction in α and μ); ``sweep``: the bare airframe (in α, at
        μ = 1 — it corrects the model's un-blown polar where no blown case solved). ``None`` when no record of that
        kind is usable: the caller keeps flying on the model."""
        want = source == "disks"
        samples = []
        for r in records:
            if bool(r.get("rotor_disks")) != want:
                continue
            vals = [r.get("CL"), r.get("CD"), r.get("Cm")]
            if any(v is None or not np.isfinite(v) or abs(v) > SANE for v in vals):
                continue
            pt = {"alpha_deg": float(r["alpha_deg"]), "V": float(r["V"]), "T_up": float(r["T_up"]), "T_lo": float(r["T_lo"])}
            mc = model_coefficients(veh, pt)
            samples.append({"alpha": math.radians(pt["alpha_deg"]), "mu": mc["mu"] if want else 1.0, "dCL": float(vals[0]) - mc["CL"],
                            "dCD": float(vals[1]) - mc["CD"], "dCm": float(vals[2]) - mc["Cm"], "case": r.get("case", "")})
        return cls(samples, source) if samples else None

    @classmethod
    def from_results(cls, cfd_table, veh, *, source: str = "disks"):
        """From ``table``'s solved rows: ``source="disks"`` the rotor-disk cases (the bare sweep is their reference),
        ``"sweep"`` the bare airframe. ``None`` when nothing usable solved."""
        return cls.from_records(cls.records(cfd_table), veh, source=source)

    def correction(self, alpha: float, mu: float) -> np.ndarray:
        x = np.array([alpha, mu])
        if self._by_alpha:
            a = float(np.clip(alpha, self._X[0, 0], self._X[-1, 0]))
            return np.array([np.interp(a, self._X[:, 0], self._D[:, j]) for j in range(3)])
        if self._lin is not None:
            v = self._lin(x[None])[0]
            if np.all(np.isfinite(v)):
                return v
        d = ((self._X - x) / np.array([math.radians(10.0), 0.2])) ** 2
        return self._D[int(np.argmin(d.sum(1)))]


def sweep_comparison(cfd_table, veh) -> "pd.DataFrame":
    """The bare sweep against the model's airframe at the same points: CL, CD, Cm (CG, chord) from the CFD and from
    the model, and the differences. Empty when no bare case solved."""
    import pandas as pd
    rows = {}
    for name, r in usable(cfd_table).iterrows():
        if bool(r["rotor disks"]):
            continue
        pt = {"alpha_deg": float(r["α [deg]"]), "V": float(r["V [m/s]"]), "T_up": 0.0, "T_lo": 0.0}
        mc = model_coefficients(veh, pt)
        rows[name] = {"α [deg]": pt["alpha_deg"], "CL CFD": float(r["CL"]), "CL model": mc["CL"], "ΔCL": float(r["CL"]) - mc["CL"],
                      "CD CFD": float(r["CD"]), "CD model": mc["CD"], "ΔCD": float(r["CD"]) - mc["CD"],
                      "Cm CFD": float(r["Cm (CG, chord)"]), "Cm model": mc["Cm"], "ΔCm": float(r["Cm (CG, chord)"]) - mc["Cm"],
                      "converged": r.get("converged"), "note": r.get("note", "")}
    return pd.DataFrame(rows).T


def sweep_numbers(cmp) -> dict:
    """What the bare sweep says in a few numbers each for the CFD and the model: the lift slope (per rad, fitted on
    the points at or below 10°), CL max over the sweep and its α, CL max below 20° (the stall's peak — on the sweep's
    angles only: a 5° grid can miss it, ``SWEEP_ALPHAS_FINE`` resolves it) and its α, CD at the lowest α, and Cm's
    slope (per rad: −slope / CLα is the static margin in chords)."""
    out = {}
    if cmp is None or len(cmp) == 0:
        return out
    a = cmp["α [deg]"].astype(float).to_numpy()
    for who in ("CFD", "model"):
        cl, cd, cm = (cmp[f"{k} {who}"].astype(float).to_numpy() for k in ("CL", "CD", "Cm"))
        lin = a <= 10.0 + 1e-9
        ar = np.radians(a)
        if lin.sum() >= 2:
            cla = float(np.polyfit(ar[lin], cl[lin], 1)[0])
            cma = float(np.polyfit(ar[lin], cm[lin], 1)[0])
        else:
            cla = cma = math.nan
        i = int(np.nanargmax(cl))
        pre = a < 20.0
        j = int(np.flatnonzero(pre)[np.nanargmax(cl[pre])]) if pre.any() else i
        out[who] = {"CLα [1/rad]": cla, "CL max": float(cl[i]), "α at CL max [deg]": float(a[i]), "CL max below 20°": float(cl[j]), "α of it [deg]": float(a[j]),
                    "CD at the lowest α": float(cd[int(np.argmin(a))]),
                    "Cmα [1/rad]": cma, "static margin from Cmα [chords]": (-cma / cla) if cla and np.isfinite(cla) and cla != 0 else math.nan}
    return out


def save_records(cfd_table, path, *, key: str = "4Q-1", meta: dict | None = None, keep_previous: bool = True) -> list[dict]:
    """The campaign ``key`` (``"4Q-1"``, ``"4Q-2"``, ``"sections"``) into the JSON the design notebook plugs in
    (``load_records``): the solved cases (``CFDAirframe.records``), the whole table's statuses, the summary. Other
    campaigns in the file are kept. With ``keep_previous`` a campaign that solved nothing (a run without OpenFOAM) does
    not overwrite one that did — a NOT RUN pass never erases a result."""
    import json
    recs = CFDAirframe.records(cfd_table) if "CL" in cfd_table.columns else []
    st = {str(k): {"status": str(v["status"]), "note": str(v.get("note", "") or ""), "rotor disks": bool(v.get("rotor disks", False)),
                   "α [deg]": float(v["α [deg]"]), "V [m/s]": float(v["V [m/s]"])} for k, v in cfd_table.to_dict("index").items()}
    camp = {"meta": meta or {}, "summary": {k: v for k, v in summary(cfd_table).items() if k != "line"}, "records": recs, "statuses": st}
    path = Path(path)
    doc = load_records(path) or {"campaigns": {}}
    doc.setdefault("campaigns", {})
    prev = doc["campaigns"].get(key)
    if keep_previous and not recs and prev and prev.get("records"):
        return prev["records"]
    doc["campaigns"][key] = camp
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))
    return recs


def load_records(path, key: str | None = None) -> dict | None:
    """The JSON of ``save_records`` (``key`` None: the whole document, else that campaign), or ``None`` when the file
    or the campaign does not exist."""
    import json
    path = Path(path)
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if "campaigns" not in doc:                              # the first layout: one campaign, unkeyed
        doc = {"campaigns": {"4Q-1": doc}}
    return doc if key is None else doc["campaigns"].get(key)


def plug_in(veh, path=RESULTS_JSON, key: str = "4Q-1") -> dict:
    """What the design notebook does with a campaign: the correction from the blown cases if any solved, else from
    the bare sweep, else none; with the campaign's summary and the records. ``{"airframe": CFDAirframe | None,
    "source": "disks" | "sweep" | None, "campaign": dict | None}``."""
    camp = load_records(path, key)
    if not camp or not camp.get("records"):
        return {"airframe": None, "source": None, "campaign": camp}
    for source in ("disks", "sweep"):
        corr = CFDAirframe.from_records(camp["records"], veh, source=source)
        if corr is not None:
            return {"airframe": corr, "source": source, "campaign": camp}
    return {"airframe": None, "source": None, "campaign": camp}


def records_table(records: list[dict]):
    """The records of a campaign as a frame in ``table``'s columns (for ``sweep_comparison`` and the polar figure)."""
    import pandas as pd
    rows = {}
    for r in records:
        rows[r.get("case", f"α {r['alpha_deg']:g}°, {r['V']:g} m/s")] = {
            "α [deg]": r["alpha_deg"], "V [m/s]": r["V"], "T upper [N]": r["T_up"], "T lower [N]": r["T_lo"], "rotor disks": bool(r["rotor_disks"]),
            "source": r.get("source", ""), "status": "solved", "CL": r["CL"], "CD": r["CD"], "Cm (CG, chord)": r["Cm"],
            "iterations": r.get("iterations"), "converged": r.get("converged"), "note": ""}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------- the section check (Part II)
def section_stl(workdir, name: str, reflex_k21, alpha_deg: float, *, chord_m: float = 0.14, AR: float = 8.0) -> Path:
    """An extruded slab of a Zephyr section (chord ``chord_m``, span AR × chord, square tips) turned to ``alpha_deg``
    about its quarter chord, as an STL in millimetres: the CFD check of the surrogate's polar."""
    import cadquery as cq
    from vegeta import zephyr
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    s = zephyr.naca(name, reflex_k21=reflex_k21)
    tag = s.name.replace(" ", "_").replace("/", "").replace("(", "").replace(")", "")
    path = workdir / f"section_{tag}_a{alpha_deg:+05.1f}.stl"
    if not path.exists():
        c = chord_m * 1000
        pts = [(x * c, z * c) for x, z in s.coordinates(80, te_cut=0.995)]
        solid = cq.Workplane("XZ").polyline(pts).close().extrude(AR * c / 2, both=True)
        solid = solid.rotate((0.25 * c, 0, 0), (0.25 * c, 1, 0), alpha_deg)
        cq.exporters.export(solid, str(path), tolerance=0.2)
    return path


def section_case(workdir, spec: dict, *, quality: str = "screening", env=None):
    """One Aeromant ``rans_ksst_external`` case of a section slab (``kestrel_section.cfd_specs`` row): the free stream at
    the cruise speed, the reference area the slab's planform, the moment about the quarter chord (on L_ref = span / 4:
    multiply the case's Cm by L_ref / chord for the section's)."""
    from vegeta import aeromant
    c, AR, a = spec["chord_m"], spec["AR"], spec["alpha_deg"]
    stl = section_stl(Path(workdir).parent / "stl", spec["name"], spec["reflex_k21"], a, chord_m=c, AR=AR)
    params = dict(velocity=spec["V"], kinematic_viscosity=NU, density=RHO, reference_area=AR * c * c, reference_length=AR * c / 4,
                  center_of_rotation=(0.25 * c, 0.0, 0.0), **QUALITY[quality])        # L_ref a quarter of the span (it sizes the domain); Cm × L_ref / c for the chord
    env = env or aeromant.OpenFOAMEnvironment.detect()
    tag = stl.stem + f"_V{spec['V']:g}_{quality}"
    return aeromant.CFDCase("rans_ksst_external", str(stl), params, workdir=str(Path(workdir) / tag), geometry_units="mm", environment=env)


def section_table(results: dict, specs: dict) -> "pd.DataFrame":
    """One row per slab case: the status (solved / diverged / FAILED / NOT RUN), CL and CD on the slab's planform, Cm
    about the quarter chord on the chord (the template's reference length is AR·c/4), the section estimate of the lift
    coefficient (lifting line: CL·(AR + 2)/AR) to read against Zephyr's polar, convergence and a note."""
    import pandas as pd
    rows = {}
    for k, v in specs.items():
        r = results.get(k)
        label = v["name"] + (f", k₂/k₁ {v['reflex_k21']:g}" if v.get("reflex_k21") else "")
        base = {"α [deg]": v["alpha_deg"], "V [m/s]": v["V"], "Re (chord)": v["V"] * v["chord_m"] / NU, "AR": v["AR"], "section": label}
        if r is None or not getattr(r, "ok", False):
            msg = "NOT RUN" if r is None or any("NOT RUN" in m for m in getattr(r, "messages", [])) else "FAILED"
            rows[k] = dict(base, status=msg, note="" if r is None else _note(r))
            continue
        m = r.metrics
        coefs = [np.nan if m.get(c) is None else float(m.get(c)) for c in ("Cl", "Cd", "Cm")]
        sane = all(np.isfinite(x) and abs(x) <= SANE for x in coefs)
        note = "" if sane else f"diverged: |coefficients| up to {np.nanmax(np.abs(coefs)):.3g}"
        if sane and m.get("converged") is False:
            note = f"not converged in {m.get('iterations')} iterations"
        rows[k] = dict(base, status="solved" if sane else "diverged", CL=coefs[0], CD=coefs[1], **{"Cm (c/4, chord)": coefs[2] * v["AR"] / 4},
                       **{"cl section estimate": coefs[0] * (v["AR"] + 2) / v["AR"]}, iterations=m.get("iterations"), converged=m.get("converged"),
                       cells=m.get("mesh_cells"), note=note)
    return pd.DataFrame(rows).T


def save_section_table(table, path=RESULTS_JSON, *, key: str = "sections", meta: dict | None = None) -> dict:
    """The slab campaign into the results JSON (``campaigns[key]``: the rows as plain dicts, the status counts)."""
    import json
    rows = {str(k): {kk: (None if vv is None or (isinstance(vv, float) and not np.isfinite(vv)) else (vv.item() if hasattr(vv, "item") else vv))
                     for kk, vv in r.items()} for k, r in table.to_dict("index").items()}
    st = table["status"].astype(str)
    camp = {"meta": meta or {}, "summary": {k: int((st == k).sum()) for k in ("solved", "diverged", "FAILED", "NOT RUN")}, "rows": rows}
    path = Path(path)
    doc = load_records(path) or {"campaigns": {}}
    doc.setdefault("campaigns", {})
    prev = doc["campaigns"].get(key)
    if camp["summary"]["solved"] == 0 and prev and prev.get("summary", {}).get("solved"):
        return prev
    doc["campaigns"][key] = camp
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))
    return camp


def section_not_run_table(specs: dict) -> "pd.DataFrame":
    import pandas as pd
    rows = {k: {"status": "NOT RUN", "α [deg]": v["alpha_deg"], "V [m/s]": v["V"], "Re (chord)": v["V"] * v["chord_m"] / NU, "AR": v["AR"],
                "template": "rans_ksst_external", "reads against": f"the slab's lift slope ≈ AR/(AR+2) = {v['AR'] / (v['AR'] + 2):.2f} of the section's"}
            for k, v in specs.items()}
    return pd.DataFrame(rows).T
