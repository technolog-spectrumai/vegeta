"""kestrel_compare — the other Vegeta machines scaled to Kestrel's size, and Kestrel as a monoplane: what the biplane
buys and costs against a conventional wing, analytically (seconds, no solver).

Every machine is reduced to the numbers that survive scaling — aspect ratio, wing loading, Cd0, span efficiency,
CL max — read from its design file under ``notebooks/designs/data/`` (the notebooks 26, 28, 33, 35 write them), then
rebuilt at **Kestrel's mass** and flown at **Kestrel's cruise speed** with one common propulsive efficiency, so the
airframes are compared and not the batteries. Kestrel itself enters three times: the biplane as built (its lattice
and drag build-up), the same span and area in one wing (chord 2c), and the same span and chord in one wing (half
the area). The Reynolds number does not scale with the mass: Cd0 is kept as each design found it and the note says
how far its chord Reynolds number moves.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import kestrel
import kestrel_flight as kf
import kestrel_systems as ks

DATA = Path(__file__).resolve().parent / "data"
G, RHO, NU = kf.G, kf.RHO, kf.NU
ETA_PROP = 0.60                  # one propulsive efficiency for every machine at the cruise: the airframes are compared, not the drives
CL_MAX_DEFAULT = 1.1


def _load(name: str) -> dict | None:
    f = DATA / f"{name}_design.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def _trapezoid(span_mm: float, root_mm: float, taper: float) -> tuple[float, float]:
    """(S [m²], AR) of a straight-tapered wing from its CAD parameters in millimetres."""
    b, cr = span_mm / 1000, root_mm / 1000
    S = b * cr * (1 + taper) / 2
    return S, b * b / S


def machines() -> dict:
    """``{name: {"type", "S", "AR", "mass", "Cd0", "e", "CLmax", "CLa", "L/D max (file)", "source", "note"}}`` for every design
    file on disk; what a file does not carry is filled from the plain conventions and said so in ``note``."""
    out = {}
    for name, label in (("falco", "FALCO (35): tractor monoplane, twin-boom tail"), ("nisus_plus", "NISUS+ (33): tractor monoplane, mountain survey")):
        d = _load(name)
        if not d:
            continue
        lay, der = d.get("layout", {}), d.get("derivatives", {})
        masses = d.get("mass", {})
        m = next(iter(masses.values()), {}) if masses else {}
        S = float(lay.get("S_ref", float("nan")))
        AR = float(lay.get("AR", float("nan")))
        env0 = (d.get("envelope") or {}).get("0 m", {})
        out[name] = {"type": label, "S": S, "AR": AR, "mass": float(m.get("mass_kg", float("nan"))), "Cd0": float(der.get("CD0", float("nan"))),
                     "e": float(der.get("e", der.get("oswald", 0.85))), "CLmax": float(der.get("CLmax", d.get("cl_max", CL_MAX_DEFAULT))), "CLa": float(der.get("CLa", float("nan"))),
                     "L/D max (file)": float(env0.get("L/D max", float("nan"))), "stall (file) [m/s]": float(env0.get("stall TAS [m/s]", float("nan"))),
                     "source": d.get("source", name), "note": ("e 0.85 assumed" if "e" not in der and "oswald" not in der else "") + ("; CL max 1.1 assumed" if "CLmax" not in der else "")}
    d = _load("merlin")
    if d:
        p = d.get("merlin_params", {})
        S, AR = _trapezoid(p.get("span", 1400.0), p.get("root_chord", 270.0), p.get("taper", 0.6))
        variants = d.get("variants") or {}
        pref = "tractor propeller" if "tractor propeller" in variants else (next(iter(variants)) if variants else "tractor propeller")
        v = variants.get(pref, {})
        out["merlin"] = {"type": f"MERLIN (26): {pref} monoplane, V-tail", "S": float(v.get("wing area [m²]", S)), "AR": float(v.get("aspect ratio", AR)),
                         "mass": float(v.get("all-up mass [kg]", float("nan"))), "Cd0": float(v.get("Cd0 with allowance", v.get("Cd0 build-up", float("nan")))),
                         "e": float(d.get("oswald", 0.8)), "CLmax": float(d.get("cl_max", CL_MAX_DEFAULT)), "CLa": float("nan"), "L/D max (file)": float("nan"),
                         "stall (file) [m/s]": float(v.get("stall speed (Cl_max 1.2) [m/s]", float("nan"))), "source": d.get("source", "26_merlin"), "note": "Cd0 the build-up with its allowance"}
    d = _load("peregrine")
    if d:
        p = (d.get("params") or {}).get("A", {})
        S, AR = _trapezoid(p.get("span", 1100.0), p.get("root_chord", 250.0), p.get("taper", 0.4))
        env = d.get("envelope") or {}
        row = {}
        try:
            df = pd.DataFrame(env["data"], index=env["index"], columns=env["columns"])
            row = df.iloc[:, 0].to_dict()
        except (KeyError, ValueError):
            pass
        cd0 = d.get("cfd_cd0") or {}
        cd0v = next((float(v) for v in cd0.values() if isinstance(v, (int, float))), float("nan")) if isinstance(cd0, dict) else float("nan")
        mass = (d.get("mass_g") or {}).get("A", float("nan")) / 1000 if isinstance(d.get("mass_g"), dict) else float("nan")
        ld = float(row.get("L/D max", float("nan")))
        # Cd0 from the file's L/D max when the CFD gave none: L/D max = ½ √(π AR e / Cd0)
        e = 0.85
        if not np.isfinite(cd0v) and np.isfinite(ld):
            cd0v = math.pi * AR * e / (4 * ld * ld)
        out["peregrine"] = {"type": "PEREGRINE A (28): tractor monoplane, folding wing (unfolded)", "S": S, "AR": AR, "mass": float(mass), "Cd0": float(cd0v), "e": e,
                            "CLmax": CL_MAX_DEFAULT, "CLa": float("nan"), "L/D max (file)": ld, "stall (file) [m/s]": float(row.get("stall [m/s]", float("nan"))),
                            "source": d.get("source", "28_peregrine"), "note": "Cd0 back-figured from the file's L/D max with e 0.85; CL max 1.1 assumed"}
    return out


def kestrel_rows(p=None, *, mass_kg: float | None = None, a: dict | None = None, d: dict | None = None, cl_max: float | None = None, faired: bool | None = None) -> dict:
    """Kestrel three ways, from its own model: the biplane (lattice k on the pair, the drag build-up's Cd0), the
    same span and area as one wing (chord 2c: the lattice monoplane's k, the build-up's Cd0 less the bracing and one
    junction set), the same span and chord as one wing (half the area, one wing's k)."""
    a = a or kf.aero(p)
    if faired is None:
        faired = bool(kestrel.resolve(kestrel.overrides(p)).get("strut_fairing", False))      # the variant's own struts: faired posts (4Q-2) or round tubes (4Q-1)
    d = d or kf.drag_buildup(p, V=kf.V_CRUISE, faired=faired)
    L = kestrel.layout(p)
    pol = kf.polar(p, a=a, d=d)
    clmax = cl_max if cl_max is not None else pol.cl_max
    m = mass_kg if mass_kg is not None else ks.cg_inertia(ks.mass_table(p))["mass_kg"]
    S2, b, c = L["S_ref"], L["span"], L["chord"]
    cd0_bi = d["Cd0"]
    cd0_mono = cd0_bi * (1 - d.get("bracing_share", 0.0))                       # no struts or posts; the pod and nacelles stay
    rows = {
        "Kestrel biplane (as built)": {"type": "zero-stagger biplane, quad rotor tailsitter", "S": S2, "AR": b * b / S2, "AR per wing": b / c, "mass": m, "Cd0": cd0_bi,
                                       "k": a["biplane"]["k"], "e": a["biplane"]["e"], "CLmax": clmax, "CLa": a["biplane"]["CLa"], "note": "lattice k of the pair; build-up Cd0 with the bracing"},
        "Kestrel as a monoplane, same span and area (chord 2c)": {"type": "monoplane, AR halved", "S": S2, "AR": b * b / S2, "AR per wing": b * b / S2, "mass": m, "Cd0": cd0_mono,
                                                                  "k": a["monoplane, same span and area"]["k"], "e": a["monoplane, same span and area"]["e"], "CLmax": clmax, "CLa": a["monoplane, same span and area"]["CLa"],
                                                                  "note": "no bracing drag; one thick 2c wing would also need its own spar (not costed)"},
        "Kestrel as a monoplane, same span and chord (one wing)": {"type": "monoplane, half the area", "S": S2 / 2, "AR": b / c, "AR per wing": b / c, "mass": m, "Cd0": cd0_mono * S2 / (S2 / 2) - d.get("Cd0_wings", 0.0),
                                                                   "k": a["one wing alone"]["k"], "e": a["one wing alone"]["e"], "CLmax": clmax, "CLa": a["one wing alone"]["CLa"],
                                                                   "note": "the body's drag area on half the wing area; wing loading doubles"},
    }
    return rows


def scaled(machine: dict, *, mass_kg: float, V: float = kf.V_CRUISE, eta: float = ETA_PROP) -> dict:
    """One machine rebuilt at ``mass_kg`` with its wing loading, AR, Cd0, e and CL max kept, flown level at ``V``."""
    W = mass_kg * G
    WS = machine["mass"] * G / machine["S"] if np.isfinite(machine.get("mass", float("nan"))) else float("nan")
    S = W / WS if np.isfinite(WS) else machine["S"]
    AR = machine["AR"]
    b = math.sqrt(AR * S)
    k = machine.get("k") or 1 / (math.pi * AR * machine["e"])
    q = 0.5 * RHO * V * V
    CL = W / (q * S)
    CD = machine["Cd0"] + k * CL * CL
    D = q * S * CD
    LDmax = 0.5 / math.sqrt(machine["Cd0"] * k)
    Vs = math.sqrt(2 * W / (RHO * S * machine["CLmax"]))
    Vmd = math.sqrt(2 * W / (RHO * S)) / (machine["Cd0"] / k) ** 0.25
    chord = S / b
    Re_scaled = V * chord / NU
    Re_orig = V * (machine["S"] / math.sqrt(AR * machine["S"])) / NU if np.isfinite(machine.get("S", float("nan"))) else float("nan")
    return {"type": machine["type"], "span scaled [m]": b, "area scaled [m²]": S, "AR": AR, "wing loading [N/m²]": W / S, "Cd0": machine["Cd0"], "k = CDi/CL²": k,
            "CL at cruise": CL, "L/D at cruise": W / D, "L/D max": LDmax, "best-glide speed [m/s]": Vmd, "stall speed [m/s]": Vs, "cruise / stall": V / Vs,
            "drag at cruise [N]": D, "shaft power at cruise [W]": D * V / eta, "chord Re at cruise": Re_scaled, "chord Re of the original": Re_orig,
            "Re moved by": (Re_scaled / Re_orig - 1) if np.isfinite(Re_orig) else float("nan"), "note": machine.get("note", "")}


def compare(p=None, *, mass_kg: float | None = None, V: float = kf.V_CRUISE, a: dict | None = None, d: dict | None = None, cl_max: float | None = None,
            faired: bool | None = None) -> pd.DataFrame:
    """The table: Kestrel three ways and every other machine on disk, all at Kestrel's mass and ``V``. ``faired``: the
    struts in the drag build-up (default: the parameter set's ``strut_fairing``)."""
    rows = kestrel_rows(p, mass_kg=mass_kg, a=a, d=d, cl_max=cl_max, faired=faired)
    m = next(iter(rows.values()))["mass"]
    out = {name: scaled(r, mass_kg=m, V=V) for name, r in rows.items()}
    for name, mc in machines().items():
        if np.isfinite(mc["S"]) and np.isfinite(mc["Cd0"]) and np.isfinite(mc["mass"]):
            out[mc["type"].split(":")[0]] = scaled(mc, mass_kg=m, V=V)
    df = pd.DataFrame(out).T
    num = [c for c in df.columns if c not in ("type", "note")]
    df[num] = df[num].astype(float)
    return df


def verdict(df: pd.DataFrame) -> list[str]:
    """What the table says about biplane against monoplane, in sentences."""
    bi = df.iloc[0]
    out = []
    m2c = df.index[1]
    m1 = df.index[2]
    out.append(f"At the same span and area, the biplane has {bi['L/D at cruise'] / df.loc[m2c, 'L/D at cruise']:.2f}× the monoplane's L/D at {kf.V_CRUISE:g} m/s "
               f"(induced drag {bi['k = CDi/CL²'] / df.loc[m2c, 'k = CDi/CL²']:.2f}× through the per-wing aspect ratio; Cd0 {bi['Cd0'] / df.loc[m2c, 'Cd0']:.2f}× through the bracing).")
    out.append(f"At the same span and chord, one wing alone stalls at {df.loc[m1, 'stall speed [m/s]']:.1f} m/s against the pair's {bi['stall speed [m/s]']:.1f} m/s: the second wing is "
               f"what keeps the stall speed and the hover-to-cruise transition in reach at this span.")
    others = df.iloc[3:]
    if len(others):
        best = others["L/D at cruise"].idxmax()
        out.append(f"Scaled to Kestrel's mass, the best conventional monoplane here ({best}) flies {others.loc[best, 'L/D at cruise']:.1f} L/D against the biplane's {bi['L/D at cruise']:.1f} "
                   f"with a span of {others.loc[best, 'span scaled [m]']:.2f} m against {bi['span scaled [m]']:.2f} m — the span, not the second wing, is what the monoplanes buy their efficiency with; "
                   f"Kestrel's span is bounded by the hover footprint and the capsule.")
    out.append("Cd0 and e are each design's own (the Re moves with the chord, see the column); the propulsive efficiency is the same 0.60 for all, so the power column compares airframes only.")
    return out
