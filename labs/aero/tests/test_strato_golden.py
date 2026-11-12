"""Strato regression goldens (Stage V8.15): four fixed reference designs (15, 30, 50, 80 m) through the feasibility
gate, July and December at 50°N. A change in any model moves these numbers; if the change is intended, record the
new goldens with ``STRATO_GOLDEN=update pytest labs/aero/tests/test_strato_golden.py`` and say why in the commit."""
import json
import math
import os
from pathlib import Path

import pytest

import strato_design as sd
import strato_feasibility as sf

GOLDEN = Path(__file__).with_name("golden") / "strato.json"
KEYS = ("system_mass", "net_lift", "surplus_lift", "gas_mass", "plies", "dp_min", "dp_max", "superheat_day",
        "availability", "energy_solar_wh", "v_max", "v_energy", "gas_life_days", "max_float_altitude", "feasible")


def designs() -> dict:
    """Fixed designs (not optimised: the goldens must not depend on the optimiser)."""
    base = dict(fineness=3.5, laminate="zylon", fin_scale=0.7, night_superpressure=150.0)
    return {
        "15m": sd.Design(length=15.0, fineness=15 / 3.5, payload_mass=1.0, battery_energy=500.0, propulsion_power=300.0),
        "30m": sd.Design(length=30.0, payload_mass=5.0, battery_energy=3000.0, propulsion_power=2500.0, **base),
        "50m": sd.Design(length=50.0, payload_mass=20.0, battery_energy=12000.0, propulsion_power=9000.0, **base),
        "80m": sd.Design(length=80.0, payload_mass=50.0, battery_energy=40000.0, propulsion_power=30000.0, **base),
    }


def compute() -> dict:
    out = {}
    for name, d in designs().items():
        for month in (7, 12):
            r = sf.evaluate(d, sd.Mission(month=month), n_days=500)
            out[f"{name}/{month}"] = {k: (None if isinstance(r[k], float) and not math.isfinite(r[k]) else r[k]) for k in KEYS}
    return out


def test_goldens():
    got = compute()
    if os.environ.get("STRATO_GOLDEN") == "update" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True))
        pytest.skip(f"goldens written to {GOLDEN}")
    want = json.loads(GOLDEN.read_text())
    assert got.keys() == want.keys()
    for case, row in want.items():
        for k, v in row.items():
            g = got[case][k]
            if v is None or isinstance(v, bool):
                assert g == v, (case, k)
            else:
                assert g == pytest.approx(v, rel=1e-6, abs=1e-6), (case, k)


def test_golden_story():
    """What the goldens say, in words: the 15 m ship never floats at 20 km; the 80 m one does, and is on station
    far more in July than in December."""
    want = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else compute()
    assert not want["15m/7"]["feasible"] and want["15m/7"]["surplus_lift"] < 0
    assert want["80m/7"]["feasible"]
    assert want["80m/7"]["availability"] > want["80m/12"]["availability"]
