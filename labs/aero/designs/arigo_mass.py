"""The ARIGO family's mass budgets (notebook 39): every item, its mass and where the number comes from, and the
battery options. The flight models take their mass from here (empty + the default pack), so the speeds, turns,
endurance and range all carry the pack's weight; ``arigo_perf`` also trades the pack size against endurance.

Sources, marked per line:
- ``catalogue``: ogma.batteries (the Li-ion 3S2P P42A pack: 450 g measured by the seller);
- ``cell``: Molicel P42A 21700 cell, 70 g, 4.2 Ah, 3.6 V nominal (the cell's datasheet class value);
- ``HAND``: a class value for the part (typical catalogue masses of the type), with its margin;
- ``geometry``: an areal or linear density (HAND, stated) times the CAD's areas and lengths.

The CG is the design point the CFD ran at (``cg_x``): the pack is placed to hold it, its mass does not move it.
"""
from __future__ import annotations

import math

# ---- structure densities (HAND): moulded EPO/EPP skins on a carbon spar, glass-reinforced fuselage
WING_KG_M2 = 1.55        # +-0.3: wing incl. spar, servos' bays, skin
TAIL_KG_M2 = 0.90        # +-0.2: stabiliser and fins (thinner, no spar)
FUSE_KG_M = 0.28         # +-0.06 per metre of fuselage at ~78 mm diameter
CELL_G, CELL_AH, CELL_V = 70.0, 4.2, 3.6     # cell: Molicel P42A
PACK_OVERHEAD = 1.07                         # HAND: nickel strip, wrap, BMS-less balance lead, connector (+-0.03)

COMMON = [  # (item, grams, source)
    ("flight controller + GPS/compass", 40.0, "HAND (+-10)"),
    ("vision computer (Jetson Orin Nano class, module + light carrier, heatsink)", 125.0, "HAND (+-25)"),
    ("observation camera (global shutter, M12 lens)", 30.0, "HAND (+-10)"),
    ("radio + telemetry + antennas", 25.0, "HAND (+-10)"),
    ("wiring, connectors, power module", 55.0, "HAND (+-20)"),
    ("belly skid + launch hook", 20.0, "HAND (+-10)"),
]
PROPULSION = {
    "single": [("motor (2814-class, ~900 W)", 110.0, "HAND (+-20)"), ("ESC 60 A", 45.0, "HAND (+-10)"),
               ("folding propeller 10x6 + spinner", 35.0, "HAND (+-10)")],
    "twin": [("2 motors (2212-class, ~450 W each)", 2 * 60.0, "HAND (+-20)"), ("2 ESC 35 A", 2 * 28.0, "HAND (+-10)"),
             ("2 folding propellers 8x6 + spinners", 2 * 22.0, "HAND (+-8)"), ("2 nacelles", 2 * 22.0, "HAND (+-10)")],
    "edf": [("2 ducted fan units 70 mm, 12 blades, motor included", 2 * 175.0, "HAND (+-30)"),
            ("2 ESC 80 A", 2 * 60.0, "HAND (+-15)"), ("2 pylons + nacelle fairings", 2 * 35.0, "HAND (+-15)")],
}


def packs(kind: str) -> dict:
    """The battery options: name -> (grams, Wh nominal, max continuous A, source)."""
    if kind == "edf":   # 6S LiPo (the fans want ~2 x 60 A): class values, ~150 g per Ah at 6S (HAND)
        return {f"6S {mah} mAh LiPo": (round(0.150 * mah), 22.2 * mah / 1000, 30 * mah / 1000, "HAND (+-10 %)")
                for mah in (3000, 4000, 5000, 6000)}
    out = {}
    for p in (1, 2, 3, 4):
        if p == 2:
            out["Li-ion 3S2P P42A"] = (450.0, 3 * CELL_V * 2 * CELL_AH, 90.0, "catalogue (ogma.batteries liion-3s2p-p42a)")
        else:
            out[f"Li-ion 3S{p}P P42A"] = (round(3 * p * CELL_G * PACK_OVERHEAD), 3 * CELL_V * p * CELL_AH, 45.0 * p, "cell x count + 7 % (HAND)")
    return out


DEFAULT_PACK = {"single": "Li-ion 3S2P P42A", "twin": "Li-ion 3S2P P42A", "edf": "6S 4000 mAh LiPo"}


def kind_of(propulsion: str) -> str:
    return {"twin": "twin", "edf": "edf"}.get(propulsion, "single")


def budget(propulsion: str, p: dict, L: dict, pack: str | None = None) -> dict:
    """The items [g] for an airframe (``p`` resolved parameters, ``L`` its layout) with ``pack``."""
    k = kind_of(propulsion)
    S_w, S_h, S_v = L["S_mm2"] / 1e6, L["S_h_mm2"] / 1e6, L["S_v_mm2"] / 1e6
    items = [("wing", 1000 * WING_KG_M2 * S_w, f"geometry: {S_w:.3f} m2 x {WING_KG_M2} kg/m2 (HAND +-20 %)"),
             ("tail surfaces", 1000 * TAIL_KG_M2 * (S_h + S_v), f"geometry: {S_h + S_v:.3f} m2 x {TAIL_KG_M2} kg/m2 (HAND +-20 %)"),
             ("fuselage", 1000 * FUSE_KG_M * p["fuselage_length"] / 1000, f"geometry: {p['fuselage_length']:.0f} mm x {FUSE_KG_M} kg/m (HAND +-20 %)"),
             ("servos", 12.0 * (5 if k == "edf" else 4), f"{5 if k == 'edf' else 4} x 12 g digital servos (HAND +-3 each)")]
    if propulsion == "tractor_lowcam":
        items.append(("camera blister", 15.0, "HAND (+-5)"))
    if propulsion == "tractor":
        items.append(("camera chin strut + turret", 35.0, "HAND (+-10)"))
    items += [(n, g, s) for n, g, s in PROPULSION[k]] + COMMON
    items.append(("growth margin (10 % of the items above)", 0.10 * sum(x[1] for x in items), "HAND: a preliminary budget's usual allowance"))
    pk = pack or DEFAULT_PACK[k]
    g, wh, amps, src = packs(k)[pk]
    empty = sum(x[1] for x in items)
    return {"items": [{"item": n, "g": round(v, 1), "source": s} for n, v, s in items],
            "empty_g": round(empty, 1), "pack": {"name": pk, "g": g, "Wh_nominal": round(wh, 1), "max_A": amps, "source": src},
            "total_g": round(empty + g, 1), "pack_share": round(g / (empty + g), 3)}


def total_kg(propulsion: str, p: dict, L: dict, pack: str | None = None) -> float:
    return budget(propulsion, p, L, pack)["total_g"] / 1000.0
