"""Weighted trade matrix for the three variants. Quantitative criteria come from the calculations (``mission``,
``variants``); the judgement criteria (maturity, mechanism risk) are ASSUMED scores 1-5 and say so. The weights are
ASSUMED too, so the matrix also reports how often each variant wins when the weights are perturbed (Dirichlet)."""
from __future__ import annotations

import numpy as np

from . import mission as M
from .variants import VARIANTS, budget

WEIGHTS = {"uncooperative_capture": 0.25, "removals_per_life_250kg": 0.20, "propellant_per_removal": 0.10, "mission_time": 0.10,
           "mass_margin": 0.10, "maturity": 0.15, "mechanism_risk": 0.10}          # ASSUMED
JUDGEMENT = {"maturity": {"A": 4, "B": 4, "C": 3},                                  # ASSUMED 1-5, 5 best
             "mechanism_risk": {"A": 2, "B": 4, "C": 3}}


def raw() -> dict:
    out = {}
    for k, v in VARIANTS.items():
        c = M.campaign(v, M.DEBRIS[250])
        c500 = M.campaign(v, M.DEBRIS[500])
        b = budget(v)
        r = c.get("first", {})
        out[k] = {"uncooperative_capture": 0 if v.cooperative_only else 1,
                  "removals_per_life_250kg": c["removals"], "removals_per_life_500kg": c500["removals"],
                  "propellant_per_removal": r.get("propellant", float("nan")), "mission_time": r.get("days", float("nan")),
                  "mass_margin": 500.0 - b["wet"], **{k2: JUDGEMENT[k2][k] for k2 in JUDGEMENT}}
    return out


def scores(r: dict) -> dict:
    """0-1 scores: higher is better; quantities are min-max scaled across the variants."""
    def scale(key, invert=False):
        vals = {k: r[k][key] for k in r}
        lo, hi = np.nanmin(list(vals.values())), np.nanmax(list(vals.values()))
        return {k: (0.0 if hi == lo else ((hi - x) if invert else (x - lo)) / (hi - lo)) if not np.isnan(x) else 0.0 for k, x in vals.items()}
    s = {"uncooperative_capture": {k: float(r[k]["uncooperative_capture"]) for k in r},
         "removals_per_life_250kg": scale("removals_per_life_250kg"), "propellant_per_removal": scale("propellant_per_removal", True),
         "mission_time": scale("mission_time", True), "mass_margin": scale("mass_margin"),
         "maturity": {k: (r[k]["maturity"] - 1) / 4 for k in r}, "mechanism_risk": {k: (r[k]["mechanism_risk"] - 1) / 4 for k in r}}
    return s


def matrix(weights: dict = WEIGHTS) -> dict:
    r = raw()
    s = scores(r)
    total = {k: sum(weights[c] * s[c][k] for c in weights) for k in r}
    return {"raw": r, "scores": s, "total": total, "winner": max(total, key=total.get)}


def weight_sensitivity(n: int = 2000, seed: int = 0, concentration: float = 40.0) -> dict:
    """Share of random weight vectors (Dirichlet around ``WEIGHTS``) for which each variant wins."""
    rng = np.random.default_rng(seed)
    keys = list(WEIGHTS)
    base = np.array([WEIGHTS[k] for k in keys])
    s = scores(raw())
    wins = {k: 0 for k in VARIANTS}
    for w in rng.dirichlet(base * concentration, n):
        tot = {k: sum(wi * s[c][k] for wi, c in zip(w, keys)) for k in VARIANTS}
        wins[max(tot, key=tot.get)] += 1
    return {k: v / n for k, v in wins.items()}
