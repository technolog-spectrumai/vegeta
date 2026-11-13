"""FALCO's wing and tail built up in balsa for the laser cutter (notebook 36): NISUS+'s module through the design
seam, with FALCO's tail (the single fin on its 10/8 rod, the stabiliser on its 12/10 spar), FALCO's load cases for the
torsion check and FALCO's mass table for the comparison with the foam wing."""
from __future__ import annotations

import pandas as pd

import falco
import falco_structure as fst
import falco_systems as fsy
import nisus_plus_balsa as nb
from nisus_plus_balsa import BALSA, MATERIAL, cut_plan, cut_summary, compare_with_foam  # noqa: F401

STAB_SPAR = dict(od=fsy.STAB_SPAR["od"], x_frac=0.30, half_length=falco.resolve()["tail_span"] / 2)
FIN_ROD = dict(od=fsy.FIN_ROD["od"], x_frac=0.45, length=fsy.FIN_ROD["length"])


def _d():
    return falco.Falco()


def rib_stations(p=None, **kw) -> list[dict]:
    return nb.rib_stations(falco.resolve(p), design=_d(), **kw)


def station_table(p=None, **kw) -> pd.DataFrame:
    return nb.station_table(falco.resolve(p), design=_d(), **kw)


def wing_parts(p=None, **kw) -> list:
    return nb.wing_parts(falco.resolve(p), design=_d(), **kw)


def tail_parts(p=None, **kw) -> list:
    kw.setdefault("stab_spar", STAB_SPAR)
    kw.setdefault("fin_rod", FIN_ROD)
    return nb.tail_parts(falco.resolve(p), design=_d(), **kw)


def sheeting_blanks(p=None, **kw) -> pd.DataFrame:
    return nb.sheeting_blanks(falco.resolve(p), design=_d(), **kw)


def stock_list(p=None, **kw) -> pd.DataFrame:
    return nb.stock_list(falco.resolve(p), design=_d(), **kw)


def mass_estimate(p=None, **kw) -> pd.DataFrame:
    p = falco.resolve(p)
    kw.setdefault("tail", tail_parts(p, b=kw.get("b")))
    return nb.mass_estimate(p, design=_d(), **kw)


def foam_rows(p=None) -> dict[str, float]:
    """The rows of FALCO's mass table that the built-up wing replaces: the foam core, the balsa D-box line, the foam
    tail (the stabiliser and the fin plates)."""
    p = falco.resolve(p)
    out = {}
    for item in fsy.structure_items(p):
        name = item[0]
        if name.startswith("wing: XPS core") or name.startswith("wing: balsa D-box") or name.startswith("tail: foam"):
            out[name] = float(item[2])
    return out


def mass_comparison(p=None, balsa: pd.DataFrame = None) -> pd.DataFrame:
    p = falco.resolve(p)
    return compare_with_foam(foam_rows(p), balsa if balsa is not None else mass_estimate(p))


def torsion_check(p=None, cases=None, **kw) -> pd.DataFrame:
    p = falco.resolve(p)
    return nb.torsion_check(p, cases if cases is not None else fst.load_cases(p), design=_d(), **kw)
