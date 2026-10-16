"""Kestrel-4Q — the design loop's bookkeeping (notebook 38): the envelope estimate before any tool runs, the check
tables each iteration is measured by, and the log of how the numbers moved from iteration to iteration.

A design study is a loop, not a pipeline: estimate (``envelope``: the spec's numbers and the layout's lengths, nothing
else), plug the tools in (the CAD's mass table, Boreas, the lattice, the transition model), **check** every number
against its target (``check`` / ``check_table``), change what fails (the battery's station, the effectors, the entry
speed, the struts, the section, the bracing) and go round again; then plug the CFD in where it has been run and go
round once more. ``IterationLog`` keeps one row per pass through the loop so the last table of the notebook shows the
whole path from the envelope to Kestrel-4Q-2.

``TARGETS`` are the study's own (the specification's open decisions, § 9, closed with first numbers): take-off mass
≤ 2.0 kg, hover thrust-to-weight ≥ 2, endurance at the cruise ≥ 20 min, static margin 5–15 % of the chord, the
transitions' criteria of ``kestrel_transition.CRITERIA``.
"""
from __future__ import annotations

import math

import numpy as np

import kestrel
import kestrel_flight as kf
import kestrel_systems as ks

TARGETS = {
    "take-off mass [kg]": ("≤", 2.0),
    "hover thrust / weight": ("≥", 2.0),
    "endurance at the cruise [min]": ("≥", 20.0),
    "static margin [chords]": ("between", (0.05, 0.15)),
    "stall speed / cruise speed": ("≤", 0.8),
    "cruise CL / CL max": ("≤", 0.6),
    "height loss in a transition [m]": ("≤", 3.0),
    "transitions passing": ("≥", 1.0),
}

ENVELOPE_ASSUMPTIONS = {
    "CL max of the wings": 0.9,              # a built-up NACA 0012 wing at Re 1e5, 3-D
    "L/D at the cruise": 6.0,                # a braced biplane with round struts
    "figure of merit in the hover": 0.6,     # a 9-inch propeller
    "propulsive efficiency in cruise": 0.55, # motor × propeller at a light load
    "avionics and payload power [W]": 13.0,
}


def envelope(p=None, *, mass_kg: float = 2.0, V: float = kf.V_CRUISE, assume: dict | None = None) -> dict:
    """Iteration 0: the aircraft from the spec's targets, the layout's lengths and five assumptions — no CAD, no BEMT,
    no lattice. Wing loading, the cruise CL, the stall speed, the hover thrust and power from momentum theory, the
    cruise power from L/D, the endurance from the battery, the static margin with the CG at ``ks.X_CG_TARGET_FRAC`` and
    the neutral point at the wings' quarter chord less an allowance for the propellers and the pod."""
    a = dict(ENVELOPE_ASSUMPTIONS, **(assume or {}))
    L = kestrel.layout(p)
    pr = kestrel.resolve(p)
    rho, g = ks.RHO0, ks.G
    W = mass_kg * g
    S = L["S_ref"]
    q = 0.5 * rho * V * V
    CL = W / (q * S)
    Vs = math.sqrt(2 * W / (rho * S * a["CL max of the wings"]))
    T_rotor = W / 4
    A = L["A_disc"]
    P_hover_ideal = 4 * T_rotor * math.sqrt(T_rotor / (2 * rho * A))
    P_hover = P_hover_ideal / a["figure of merit in the hover"]
    D = W / a["L/D at the cruise"]
    P_cruise = D * V / a["propulsive efficiency in cruise"] + a["avionics and payload power [W]"]
    b = ks.COMPONENTS["battery"]
    Ewh = b["cells"] * 3.7 * b["capacity_ah"] * b["usable"]
    c = L["chord"]
    x_cg = L["x_le"] + ks.X_CG_TARGET_FRAC * c
    x_np = L["x_le"] + 0.25 * c - 0.03 * c            # the propellers ahead of the CG and the pod cost ~3 % (Part I, § 5)
    sm = (x_np - x_cg) / c
    T_W_needed = TARGETS["hover thrust / weight"][1]
    return {"take-off mass [kg]": mass_kg, "wing area [m²]": S, "wing loading [N/m²]": W / S, "cruise CL": CL,
            "CL max assumed": a["CL max of the wings"], "cruise CL / CL max": CL / a["CL max of the wings"],
            "stall speed [m/s]": Vs, "stall speed / cruise speed": Vs / V,
            "hover thrust per rotor [N]": T_rotor, "full thrust per rotor needed [N]": T_W_needed * T_rotor,
            "disk loading [N/m²]": T_rotor / A, "hover power, momentum theory [W]": P_hover,
            "cruise drag [N]": D, "cruise power [W]": P_cruise, "battery usable [Wh]": Ewh,
            "endurance at the cruise [min]": Ewh / P_cruise * 60, "hover time [min]": Ewh / P_hover * 60,
            "static margin [chords]": sm, "x_cg [m]": x_cg, "x_np [m]": x_np, "assumptions": a, "pod length [mm]": pr["pod_length"]}


# ------------------------------------------------------------------------------------------------- the checks
def _holds(op, value, target) -> bool:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return False
    if op == "≤":
        return value <= target
    if op == "≥":
        return value >= target
    if op == "between":
        return target[0] <= value <= target[1]
    if op == "==":
        return value == target
    raise ValueError(op)


def _text(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return f"{v:g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)


def _target_text(op, target) -> str:
    if op == "between":
        return f"{target[0]:g} – {target[1]:g}"
    return f"{op} {_text(target)}"


def check(name: str, value, op: str | None = None, target=None, *, unit: str = "", note: str = "", fmt: str = "{:.3g}") -> dict:
    """One row: ``value`` against ``op target`` (``≤``, ``≥``, ``between``, ``==``); without them the target comes from
    ``TARGETS[name]``."""
    if op is None:
        op, target = TARGETS[name]
    ok = _holds(op, value, target)
    if isinstance(value, (bool, np.bool_)):
        text = _text(value)
    elif isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(float(value)):
        text = fmt.format(value)
    else:
        text = str(value)
    return {"check": name, "value": text + (f" {unit}" if unit else ""), "target": _target_text(op, target), "holds": ok, "note": note}


def check_table(checks: list[dict], title: str | None = None):
    """The rows as a frame, ✔ / ✘ in the last column; ``attrs['passes']`` / ``attrs['of']`` count them."""
    import pandas as pd
    df = pd.DataFrame(checks).set_index("check")
    df["holds"] = df["holds"].map(lambda b: "✔" if b else "✘")
    df.attrs["passes"] = int(sum(bool(c["holds"]) for c in checks))
    df.attrs["of"] = len(checks)
    if title:
        df.columns.name = title
    return df


def all_hold(checks: list[dict]) -> bool:
    return all(bool(c["holds"]) for c in checks)


def failing(checks: list[dict]) -> list[str]:
    return [c["check"] for c in checks if not bool(c["holds"])]


# ------------------------------------------------------------------------------------------------- the log
class IterationLog:
    """One row per pass through the loop: the key numbers and what changed. ``frame()`` is the notebook's last table."""

    COLUMNS = ("take-off mass [kg]", "static margin [chords]", "hover thrust / weight", "stall speed [m/s]", "Cd0", "L/D at the cruise",
               "cruise power [W]", "endurance at the cruise [min]", "transitions passing", "checks holding")

    def __init__(self):
        self.rows = []

    def add(self, name: str, *, changed: str = "", source: str = "", checks: list[dict] | None = None, **numbers) -> dict:
        row = {"iteration": name, "what changed": changed, "source": source}
        row.update(numbers)
        if checks is not None:
            row["checks holding"] = f"{sum(bool(c['holds']) for c in checks)} / {len(checks)}"
            row["failing"] = ", ".join(failing(checks)) or "—"
        self.rows.append(row)
        return row

    def frame(self, columns=None):
        import pandas as pd
        df = pd.DataFrame(self.rows).set_index("iteration")
        cols = [c for c in ("what changed", "source") + tuple(columns or self.COLUMNS) + ("failing",) if c in df.columns]
        return df[cols]

    def to_dict(self) -> list[dict]:
        return [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in self.rows]
