"""K4TQ — the wing section, phase by phase: which airfoil for the cruise, the speed, the hover and the transition?

A tilt wing asks four different things of its section, at four Reynolds numbers:
- **cruise** (the most energy): low drag at the wing's own design lift coefficient — and in a tandem the two wings fly at
  different ones (the CG sets the lift split: the front wing at ~1.5 × the aircraft's CL, the rear at ~0.7 ×);
- **speed** (the dash): low drag at a low lift coefficient;
- **hover**: the wing stands edge-on in its own slipstream: its drag there is download, and a cambered section's lift at
  zero incidence pulls the aircraft sideways in the chord (trimmed by the tilt, Stage 1) — symmetric is best;
- **transition**: a high cl_max at the low Reynolds number and a gentle stall (lift kept past it);
plus the tilt actuator (|cm| about the quarter chord: the hinge moment) and the spar (a Ø16 tube with its skins needs
~18 mm of section at the spar station, 30 % chord, of the tip chord).

Polars from NeuralFoil through ``vegeta.zephyr`` (``method="neuralfoil"``, n_crit 6: a built-up small-UAV wing) —
cheap (milliseconds), so they run for real even in the early phases; they are cached in ``data/k4tq/sections_neuralfoil.json``
with their source. Under the mocks Zephyr would hand back the thin-airfoil stand-in: ``study`` refuses it and uses the
cache. The CFD check of the leaders (Aeromant, a slab of the section) is prepared by ``cfd_specs`` / ``cfd_case`` and
run from ``exp/K4TQ_airfoils.ipynb``.

Labels: NeuralFoil results are CALC (a surrogate of XFOIL-class physics, with its own confidence); the CFD is SOLVER
when it ran, MOCK when mocked.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import k4tq

NU = 1.46e-5
DATA = Path(__file__).resolve().parent / "data" / "k4tq"
AIRFOILS = DATA / "airfoils"
CACHE = DATA / "sections_neuralfoil.json"
N_CRIT = 6.0
ALPHAS = np.arange(-8.0, 20.01, 0.5)
SPAR_DEPTH_M = 0.018              # Ø16 spar + 2 × 1 mm skins at the spar station
SPAR_X = 0.30

# name → how to build it: NACA through Zephyr, the UIUC low-Reynolds sections from coordinates (written once from
# AeroSandbox's bundled database into data/k4tq/airfoils/: the UIUC site is outside this environment's network policy)
CANDIDATES = {
    "NACA 0012": ("naca", "0012"), "NACA 0015": ("naca", "0015"), "NACA 2412": ("naca", "2412"), "NACA 2415": ("naca", "2415"),
    "NACA 4412": ("naca", "4412"), "NACA 4415": ("naca", "4415"), "NACA 23012": ("naca", "23012"), "NACA 23015": ("naca", "23015"),
    "SD7062": ("dat", "sd7062"), "E387": ("dat", "e387"), "S1223": ("dat", "s1223"), "Clark Y": ("dat", "clarky"), "MH32": ("dat", "mh32"),
    "SD7037": ("dat", "sd7037"),
}


def write_uiuc(names=("sd7062", "e387", "s1223", "clarky", "mh32", "sd7037")) -> list[Path]:
    """Write the UIUC coordinates from AeroSandbox's bundled database as Selig .dat files (once; the files are committed)."""
    import aerosandbox as asb
    AIRFOILS.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        a = asb.Airfoil(n)
        f = AIRFOILS / f"{n}.dat"
        f.write_text(n.upper() + "\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in a.coordinates) + "\n")
        out.append(f)
    return out


def section(name: str):
    from vegeta import zephyr
    kind, key = CANDIDATES[name]
    if kind == "naca":
        s = zephyr.naca(key)
    else:
        f = AIRFOILS / f"{key}.dat"
        if not f.exists():
            write_uiuc()
        s = zephyr.from_dat(f)
    import dataclasses
    return dataclasses.replace(s, name=name)


def thickness_at(name: str, x: float = SPAR_X) -> float:
    """The section's thickness at chord station ``x`` (upper − lower), on the unit chord."""
    xy = np.asarray(section(name).coordinates(200))
    i = int(np.argmin(xy[:, 0]))
    up, lo = xy[: i + 1][::-1], xy[i:]
    return float(np.interp(x, up[:, 0], up[:, 1]) - np.interp(x, lo[:, 0], lo[:, 1]))


# ================================================================================================= the flight conditions
def conditions(veh: dict) -> dict:
    """Each phase's speed, and each wing's Reynolds number (its MAC) and lift coefficient there: the tandem's lift split
    from the stall rule's loads (cl_wing / CL); the hover in the slipstream (1.5 vᵢ at the wing, zero incidence); the
    transition at 12 m/s."""
    import k4tq_sizing as sz
    L, W = veh["L"], veh["W"]
    st = veh["stall"]
    c2 = sz.cruise(veh, n_running=2)
    b = sz.best_speeds(c2)
    Vc, Vd = b["best range V [m/s]"], min(30.0, b["max level V [m/s]"])
    h = sz.hover(veh)
    v_slip = 1.5 * h["induced velocity [m/s]"]
    out = {}
    for which, load in (("front", st["cl front / CL"]), ("rear", st["cl rear / CL"])):
        c = L["wings"][which]["mac"]
        CLc = W / (0.5 * k4tq.RHO * Vc ** 2 * L["S_ref"])
        CLd = W / (0.5 * k4tq.RHO * Vd ** 2 * L["S_ref"])
        out[which] = {"chord [m]": c,
                      "cruise": {"V": Vc, "Re": Vc * c / NU, "cl": load * CLc},
                      "speed": {"V": Vd, "Re": Vd * c / NU, "cl": load * CLd},
                      "hover": {"V": v_slip, "Re": v_slip * c / NU, "cl": 0.0},
                      "transition": {"V": 12.0, "Re": 12.0 * c / NU},
                      "tip chord [m]": L["wings"][which]["tip_chord"]}
    return out


# ================================================================================================= polars (NeuralFoil, cached)
def _key(name: str, Re: float) -> str:
    return f"{name}|{round(Re, -4):.0f}|{N_CRIT:g}"


def polar(name: str, Re: float, *, cache: dict | None = None):
    """The NeuralFoil polar of ``name`` at ``Re`` (rounded to 10⁴), from the cache or computed. Refuses the mocks'
    thin-airfoil stand-in (it is not NeuralFoil)."""
    from vegeta import zephyr
    from vegeta.zephyr.polar import Polar
    Re = round(Re, -4)
    cache = cache if cache is not None else load_cache()
    k = _key(name, Re)
    if k in cache:
        return Polar.from_dict(cache[k])
    from vegeta import mock
    near = [(abs(float(kk.split("|")[1]) - Re), kk) for kk in cache if kk.split("|")[0] == name and kk.split("|")[2] == f"{N_CRIT:g}"]
    if mock.installed() and near and min(near)[0] <= 0.15 * Re:           # the mocked sizing moves the Reynolds numbers a little: the nearest cached
        return Polar.from_dict(cache[min(near)[1]])
    if mock.installed():
        raise RuntimeError(f"{name} at Re {Re:.0f} is not in {CACHE.name} and the mocks are on: run k4tq_sections.study() once with the mocks off "
                           "(NeuralFoil takes milliseconds)")
    p = zephyr.analyze(section(name), ALPHAS, Re=Re, method="neuralfoil", n_crit=N_CRIT)
    cache[k] = p.to_dict()
    return p


def load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))


# ================================================================================================= the metrics
def metrics(name: str, cond: dict, *, cache: dict) -> dict:
    """One section on one wing: cruise cd and L/D at the wing's cruise cl, speed cd at its dash cl, hover cd and cl at
    zero incidence in the slipstream, transition cl_max / stall angle / the lift kept 4° past it, the moment, the spar
    depth on the tip chord, NeuralFoil's least confidence on the attached branch."""
    pc = polar(name, cond["cruise"]["Re"], cache=cache)
    ps = polar(name, cond["speed"]["Re"], cache=cache)
    ph = polar(name, cond["hover"]["Re"], cache=cache)
    pt = polar(name, cond["transition"]["Re"], cache=cache)
    sc, st = pc.summary(), pt.summary()
    i0 = int(np.argmin(np.abs(ph.alpha_deg)))
    t30 = thickness_at(name)
    cl_c = cond["cruise"]["cl"]
    reach = cl_c <= 0.92 * float(np.max(pc.cl))
    cd_c = pc.at_cl(cl_c, "cd") if reach else math.inf
    cm_c = pc.at_cl(cl_c, "cm") if reach else math.nan
    return {"cruise cd": cd_c, "cruise L/D": cl_c / cd_c, "cruise cm": cm_c, "speed cd": ps.at_cl(cond["speed"]["cl"], "cd"),
            "hover cd (α 0)": float(ph.cd[i0]), "hover cl (α 0)": float(ph.cl[i0]), "transition cl_max": st["cl_max"],
            "stall α [deg]": st["alpha_cl_max_deg"], "lift kept 4° past the stall": st["cl_4deg_past_stall_ratio"],
            "best L/D (cruise Re)": sc["LD_max"], "t/c at 30 %": t30, "spar depth on the tip [mm]": 1e3 * t30 * cond["tip chord [m]"],
            "spar fits": t30 * cond["tip chord [m]"] >= SPAR_DEPTH_M, "least confidence": float(np.min(pc.confidence[pc.attached()])),
            "cruise cl in reach": reach}


WEIGHTS = {"cruise": 0.60, "speed": 0.10, "hover": 0.15, "transition": 0.15}    # the mission's energy and risk shares


def study(veh: dict | None = None, names=None) -> dict:
    """Every candidate on both wings; the phase scores (each metric scaled 0–1 across the candidates, best = 1) and the
    weighted total (``WEIGHTS``); the spar-depth rule as a hard filter; the recommendation per wing."""
    import k4tq_sizing as sz
    veh = veh or sz.vehicle(k4tq.MAIN)
    names = list(names or CANDIDATES)
    cond = conditions(veh)
    cache = load_cache()
    tables = {}
    for which in ("front", "rear"):
        rows = {n: metrics(n, cond[which], cache=cache) for n in names}
        df = pd.DataFrame(rows).T
        def sc(col, lower_better=True):
            v = df[col].astype(float).replace([np.inf, -np.inf], np.nan)
            lo, hi = np.nanmin(v), np.nanmax(v)
            s = (hi - v) / (hi - lo) if lower_better else (v - lo) / (hi - lo)
            return s.fillna(0.0)
        df["score cruise"] = sc("cruise cd")
        df["score speed"] = sc("speed cd")
        side_pull = df["hover cl (α 0)"].astype(float).abs()                  # the sideways pull in the slipstream: 0 for a symmetric section
        df["score hover"] = 0.5 * sc("hover cd (α 0)") + 0.5 * (1 - side_pull / max(side_pull.max(), 1e-9))
        df["score transition"] = 0.6 * sc("transition cl_max", False) + 0.4 * sc("lift kept 4° past the stall", False)
        df["total"] = sum(WEIGHTS[k] * df[f"score {k}"] for k in WEIGHTS)
        df.loc[~df["spar fits"].astype(bool), "total"] *= 0.0
        df.loc[~df["cruise cl in reach"].astype(bool), "total"] *= 0.0
        tables[which] = df.sort_values("total", ascending=False)
    save_cache(cache)
    rec = {}
    for w, df in tables.items():                                  # a tie (within 0.02) goes to the smaller hinge moment (the tilt actuator, the hover side pull)
        near = df[df["total"].astype(float) >= float(df["total"].iloc[0]) - 0.02]
        rec[w] = near["cruise cm"].astype(float).abs().idxmin()
    return {"conditions": cond, "tables": tables, "recommendation": rec, "weights": dict(WEIGHTS)}


# ================================================================================================= the CFD check (Aeromant)
QUALITY = {"quick": dict(cells_per_length=1.5, surface_level=4, near_level=2, wake_level=1, iterations=300, residual_target=2e-4),
           "screening": dict(cells_per_length=2.0, surface_level=4, near_level=3, wake_level=2, iterations=400, residual_target=1e-4)}


def cfd_specs(names, cond: dict, *, alphas=(2.0, 6.0, 10.0), which: str = "front", AR: float = 4.0) -> dict:
    """The slab cases for the CFD check: each section at a few angles at the wing's cruise speed and chord."""
    c, V = cond[which]["chord [m]"], cond[which]["cruise"]["V"]
    return {f"{n} α{a:+.0f}": {"name": n, "alpha_deg": a, "V": V, "chord_m": c, "AR": AR} for n in names for a in alphas}


def slab_stl(workdir: Path, name: str, alpha_deg: float, *, chord_m: float, AR: float) -> Path:
    """An extruded slab of the section (span AR × chord, square tips) turned to ``alpha_deg`` about its quarter chord,
    STL in millimetres (as ``kestrel_cfd.section_stl``, for any Zephyr section)."""
    import cadquery as cq
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"slab_{name.replace(' ', '_')}_c{chord_m * 1000:.0f}_a{alpha_deg:+05.1f}.stl"
    if not path.exists():
        c = chord_m * 1000
        pts = [(x * c, z * c) for x, z in section(name).coordinates(80, te_cut=0.995)]
        solid = cq.Workplane("XZ").polyline(pts).close().extrude(AR * c / 2, both=True)
        solid = solid.rotate((0.25 * c, 0, 0), (0.25 * c, 1, 0), alpha_deg)
        cq.exporters.export(solid, str(path), tolerance=0.2)
    return path


def cfd_case(workdir, spec: dict, *, quality: str = "quick", env=None):
    """One Aeromant ``rans_ksst_external`` case of a slab: free stream at the cruise speed, reference area the slab's
    planform, the moment about the quarter chord on L_ref = span / 3.5 (× L_ref / c for the section's)."""
    from vegeta import aeromant
    c, AR, a = spec["chord_m"], spec["AR"], spec["alpha_deg"]
    stl = slab_stl(Path(workdir) / "stl", spec["name"], a, chord_m=c, AR=AR)
    params = dict(velocity=spec["V"], kinematic_viscosity=NU, density=k4tq.RHO, reference_area=AR * c * c, reference_length=AR * c / 3.5,   # the span / 3.5: inside the template's 4 × L_ref body limit
                  center_of_rotation=(0.25 * c, 0.0, 0.0), **QUALITY[quality])
    env = env or aeromant.OpenFOAMEnvironment.detect()
    return aeromant.CFDCase("rans_ksst_external", str(stl), params, workdir=str(Path(workdir) / (stl.stem + f"_V{spec['V']:.0f}_{quality}")),
                            geometry_units="mm", environment=env)


def cfd_table(results: dict, specs: dict, cache: dict | None = None) -> pd.DataFrame:
    """CFD against NeuralFoil: the slab's CL, CD (its planform), the section estimate cl ≈ CL (AR + 2) / AR and
    NeuralFoil's cl and cd at the same α and Reynolds number; the status (solved / MOCK / NOT RUN / FAILED)."""
    from vegeta import mock
    cache = cache if cache is not None else load_cache()
    rows = {}
    for k, v in specs.items():
        Re = v["V"] * v["chord_m"] / NU
        p = polar(v["name"], Re, cache=cache)
        nf_cl, nf_cd = float(np.interp(v["alpha_deg"], p.alpha_deg, p.cl)), float(np.interp(v["alpha_deg"], p.alpha_deg, p.cd))
        r = results.get(k)
        base = {"section": v["name"], "α [deg]": v["alpha_deg"], "Re": Re, "NeuralFoil cl": nf_cl, "NeuralFoil cd": nf_cd}
        if r is None or not getattr(r, "ok", False):
            rows[k] = dict(base, status="NOT RUN" if r is None else "FAILED")
            continue
        m = r.metrics
        cl, cd = float(m.get("Cl", math.nan)), float(m.get("Cd", math.nan))
        rows[k] = dict(base, status="MOCK" if mock.installed() else "SOLVER", **{"CFD CL (slab)": cl, "CFD CD (slab)": cd,
                       "CFD cl (section est.)": cl * (v["AR"] + 2) / v["AR"], "Δcl vs NeuralFoil": cl * (v["AR"] + 2) / v["AR"] - nf_cl})
    return pd.DataFrame(rows).T


# ================================================================================================= figures and the report
def figure(res: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache = load_cache()
    cond = res["conditions"]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5), dpi=110)
    top = {w: list(res["tables"][w].index[:5]) for w in ("front", "rear")}
    for row, w in enumerate(("front", "rear")):
        ax = axs[row, 0]
        for n in top[w]:
            p = polar(n, cond[w]["cruise"]["Re"], cache=cache)
            s = p.attached()
            ax.plot(p.cd[s], p.cl[s], label=n)
        ax.axhline(cond[w]["cruise"]["cl"], color="k", lw=0.8, ls="--")
        ax.set(xlabel="cd", ylabel="cl", title=f"{w} wing: drag polar at the cruise Re {cond[w]['cruise']['Re']:.2g} (losses vs lift)", xlim=(0, 0.04))
        ax.legend(fontsize=8)
        ax = axs[row, 1]
        for n in top[w]:
            p = polar(n, cond[w]["transition"]["Re"], cache=cache)
            ax.plot(p.alpha_deg, p.cl, label=n)
        ax.set(xlabel="α [deg]", ylabel="cl", title=f"{w}: lift through the stall at the transition Re {cond[w]['transition']['Re']:.2g}")
        ax = axs[row, 2]
        df = res["tables"][w].head(8)
        idx = np.arange(len(df))
        for j, k in enumerate(("cruise", "speed", "hover", "transition")):
            ax.bar(idx + 0.2 * j - 0.3, df[f"score {k}"].astype(float), width=0.2, label=k)
        ax.plot(idx, df["total"].astype(float), "ko-", label="weighted total")
        ax.set_xticks(idx, df.index, rotation=35, ha="right", fontsize=8)
        ax.set(title=f"{w}: phase scores (1 = best of the candidates)", ylim=(0, 1.05))
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def report(res: dict, out: Path = DATA) -> Path:
    cond = res["conditions"]
    cols = ["cruise cd", "cruise L/D", "speed cd", "hover cd (α 0)", "hover cl (α 0)", "transition cl_max", "stall α [deg]", "lift kept 4° past the stall",
            "cruise cm", "t/c at 30 %", "spar depth on the tip [mm]", "least confidence", "total"]
    md = ["# K4TQ — the wing section by flight phase (NeuralFoil)\n",
          f"Generated by `labs/aero/designs/k4tq_sections.py`. Polars: NeuralFoil through `vegeta.zephyr` (n_crit {N_CRIT:g}), **CALC** (a surrogate, its "
          "confidence in the table), cached in `sections_neuralfoil.json`. The CFD check of the leaders is in `exp/K4TQ_airfoils.ipynb`.\n",
          "## The conditions (K4TQ-A, Cathy)\n",
          "| wing | chord | cruise V / Re / cl | speed V / Re / cl | hover slipstream V / Re | transition Re |\n|---|---|---|---|---|---|"]
    for w in ("front", "rear"):
        c = cond[w]
        md.append(f"| {w} | {c['chord [m]'] * 1e3:.0f} mm | {c['cruise']['V']:.0f} m/s, {c['cruise']['Re']:.2g}, {c['cruise']['cl']:.2f} | "
                  f"{c['speed']['V']:.0f} m/s, {c['speed']['Re']:.2g}, {c['speed']['cl']:.2f} | {c['hover']['V']:.1f} m/s, {c['hover']['Re']:.2g} | {c['transition']['Re']:.2g} |")
    md.append(f"\nWeights: {', '.join(f'{k} {v:.0%}' for k, v in res['weights'].items())}; a section whose spar depth at 30 % of the tip chord is under "
              f"{SPAR_DEPTH_M * 1e3:.0f} mm, or that cannot reach the cruise cl, scores 0.\n")
    for w in ("front", "rear"):
        df = res["tables"][w][cols]
        md.append(f"\n## {w.capitalize()} wing — recommendation: **{res['recommendation'][w]}**\n")
        md.append("| section | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
        for n, r in df.iterrows():
            md.append(f"| {n} | " + " | ".join(("yes" if v is True else "no" if v is False else f"{float(v):.4g}") for v in r.values) + " |")
    md.append("\n![phase comparison](figures/k4tq_sections.png)\n")
    f = out / "sections.md"
    f.write_text("\n".join(md))
    return f
