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
- ``table`` / ``not_run_table``: the coefficients per case (CL, CD on S_ref; Cm about the CG on the chord), or NOT RUN.
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
SWEEP_V = 10.0
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
    stl = export_stl(Path(workdir).parent / "stl", a, x_cg_m, p)
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


def specs(pts: list[dict], *, sweep: bool = True) -> dict:
    """The cases by name and their point (``props`` False for the bare sweep)."""
    out = {point_name(pt): dict(pt, props=True) for pt in pts}
    if sweep:
        out.update({k: dict(v, props=False) for k, v in sweep_specs().items()})
    return out


def plan(workdir, pts: list[dict], *, x_cg_m: float, z_cg_m: float = 0.0, quality: str = "screening", sweep: bool = True, env=None, p=None) -> dict:
    return {name: case(workdir, {k: v for k, v in s.items() if k != "props"}, x_cg_m=x_cg_m, z_cg_m=z_cg_m, quality=quality,
                       props=s["props"], env=env, p=p)
            for name, s in specs(pts, sweep=sweep).items()}


# ------------------------------------------------------------------------------------------------- the results
def table(results: dict, spec: dict, p=None) -> "pd.DataFrame":
    """One row per case: the point, CL and CD on S_ref (wind axes), Cm about the CG on the chord, convergence — or NOT RUN."""
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
            rows[name] = dict(base, status=msg)
            continue
        m = r.metrics
        rows[name] = dict(base, status="solved", CL=m.get("Cl"), CD=m.get("Cd"), **{"Cm (CG, chord)": (m.get("Cm") or np.nan) * k_cm},
                          iterations=m.get("iterations"), converged=m.get("converged"), **{"Cd scatter (last 50)": m.get("Cd_std_lastN")},
                          cells=m.get("mesh_cells"))
    return pd.DataFrame(rows).T


def not_run_table(spec: dict) -> "pd.DataFrame":
    """The planned cases with their air, rotor speeds and template, status NOT RUN."""
    import pandas as pd
    rows = {}
    for name, s in spec.items():
        v_ax = s["V"] * math.cos(math.radians(s["alpha_deg"]))
        rows[name] = {"status": "NOT RUN", "α [deg]": s["alpha_deg"], "V [m/s]": max(s["V"], 1.0), "source": s["source"],
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
    in (α, μ) — linear inside the points, the nearest point outside — and added to the model's airframe forces."""

    def __init__(self, samples: list[dict]):
        if not samples:
            raise ValueError("no solved CFD case to build a correction from")
        self.samples = samples
        X = np.array([[s["alpha"], s["mu"]] for s in samples])
        self._X = X
        self._D = np.array([[s["dCL"], s["dCD"], s["dCm"]] for s in samples])
        self._lin = None
        if len(samples) >= 4 and np.ptp(X[:, 1]) > 1e-3 and np.ptp(X[:, 0]) > 1e-3:
            from scipy.interpolate import LinearNDInterpolator
            try:
                self._lin = LinearNDInterpolator(X, self._D)
            except Exception:                          # degenerate (collinear) points
                self._lin = None

    @classmethod
    def from_results(cls, cfd_table, veh) -> "CFDAirframe":
        """From ``table``'s solved rows with rotor disks (the bare sweep is the reference, not the correction)."""
        samples = []
        for _, r in cfd_table.iterrows():
            if r.get("status") != "solved" or not bool(r.get("rotor disks")):
                continue
            pt = {"alpha_deg": float(r["α [deg]"]), "V": float(r["V [m/s]"]), "T_up": float(r["T upper [N]"]), "T_lo": float(r["T lower [N]"])}
            mc = model_coefficients(veh, pt)
            samples.append({"alpha": math.radians(pt["alpha_deg"]), "mu": mc["mu"], "dCL": float(r["CL"]) - mc["CL"],
                            "dCD": float(r["CD"]) - mc["CD"], "dCm": float(r["Cm (CG, chord)"]) - mc["Cm"]})
        return cls(samples)

    def correction(self, alpha: float, mu: float) -> np.ndarray:
        x = np.array([alpha, mu])
        if self._lin is not None:
            v = self._lin(x[None])[0]
            if np.all(np.isfinite(v)):
                return v
        d = ((self._X - x) / np.array([math.radians(10.0), 0.2])) ** 2
        return self._D[int(np.argmin(d.sum(1)))]
