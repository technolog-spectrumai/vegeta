"""og.batteries reproduces Vegeta's packs and cells (golden/batteries.json, Vegeta e918c97)."""
import json
from pathlib import Path

import numpy as np

from ogma import batteries as ob

G = json.loads((Path(__file__).with_name("golden") / "batteries.json").read_text())


def test_packs():
    packs = [ob.LiIonPack("t", ob.CELLS[c], s, p) for c in ob.CELLS for s, p in ((6, 3), (6, 4), (8, 3))]
    for b, g in zip(packs, G["packs"]):
        got = {"mass_g": b.mass_g, "Ah": b.capacity_ah, "V": b.voltage, "Wh": b.energy_wh, "Imax": b.max_current_a, "Ichg": b.max_charge_a,
               "size": list(b.size_mm), "R": [b.resistance(T) for T in (25, 5, -10, -20)], "ocv": [float(b.ocv(x)) for x in (0, 0.15, 0.5, 0.93, 1)],
               "vt": [float(b.terminal_v(I, soc, T)) for I in (-10, 20, 70) for soc in (0.2, 0.8) for T in (25, -10)],
               "pmax": [b.max_power_w(soc, T) for soc in (0.3, 0.9) for T in (25, -10)],
               "usable": [b.usable_wh(T, I) for T in (25, 5, -10) for I in (10, 40)],
               "chg": [b.charge_limit_a(T, soc) for T in (0, 10) for soc in (0.5, 0.97)]}
        for k in g:
            assert np.allclose(got[k], g[k], rtol=1e-13, atol=0), k


def test_cold_factor_and_bought_packs():
    assert np.allclose([ob.CELLS["P45B"].capacity_factor(T) for T in (-25, -15, 0, 10, 30)], G["cap_factor"])
    assert [{"key": b.key, "V": b.voltage, "Wh": b.energy_wh, "A": b.max_current_a} for b in ob.BATTERIES] == G["batteries"]
    assert ob.Cell is ob.LiIonCell and ob.Pack is ob.LiIonPack
