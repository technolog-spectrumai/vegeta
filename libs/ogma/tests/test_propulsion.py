"""og.propulsion reproduces Vegeta's propulsion tables on their real data (golden/propulsion.json, Vegeta bbaf3bc):
Nisus's PropulsionMap, Nisus+'s drive (ElectricDrive), Kestrel's DriveTable, Merlin's Unit."""
import json
from pathlib import Path

import numpy as np

from ogma import propulsion as op

G = json.loads((Path(__file__).with_name("golden") / "propulsion.json").read_text())


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def same(a, b):
    a, b = plain(a), plain(b)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, float) or isinstance(b, float):
        assert np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True), (a, b)
    else:
        assert a == b


def test_propulsion_map():
    pm, g = op.PropulsionMap.from_json(G["pm_json"]), G["pm"]
    same([pm.at(V, t) for V in (0.0, 12.0, 25.0) for t in (0.3, 0.8, 1.0)], g["at"])
    same([pm.max_thrust(V) for V in (0, 10, 20)], g["max"])
    same([pm.throttle_for_thrust(V, T) for V in (0, 15) for T in (2.0, 6.0)], g["thr"])
    same([pm.electrical_for_thrust(V, T) for V in (0, 15) for T in (2.0, 6.0)], g["el"])
    same(pm.to_json(), G["pm_json"])


def test_electric_drive():
    dr, g = op.ElectricDrive.from_json(G["dr_json"]), G["dr"]
    same([dr.at(V, t, rho) for V in (0.0, 18.0) for t in (0.4, 1.0) for rho in (1.225, 0.8)], g["at"])
    same([dr.freewheel(V) for V in (10.0, 25.0)], g["free"])
    same([dr.locked_drag(V) for V in (10.0, 25.0)], g["locked"])
    same([dr.brake(V, b, i_charge_max=20.0) for V in (15.0, 25.0) for b in (0.3, 1.0)], g["brake"])
    same([dr.max_thrust(V, rho) for V in (0, 20) for rho in (1.225, 0.8)], g["max"])
    same([dr.throttle_for_thrust(V, T) for V in (0, 20) for T in (10.0, 30.0)], g["thr"])
    same(dr.map_at(0.9).to_json(), g["map"])


def test_drive_table():
    d = G["dt"]
    dt = op.DriveTable(**{k: (tuple(map(tuple, v)) if k == "P" else tuple(v) if isinstance(v, list) else v) for k, v in d.items()})
    g = G["dt_eval"]
    same([dt.t_max(v) for v in (0, 5, 15)], g["tmax"])
    same([dt.kap(v) for v in (0, 5, 15)], g["kap"])
    same([dt.power(T, v) for T in (1.0, 5.0) for v in (0, 10)], g["P"])


def test_unit():
    a = G["unit_args"]
    u = op.Unit("t", *(np.array(a[k]) for k in ("V", "rpm", "thrust", "power")), 0.3, 800.0, 0.85)
    same([u.full(V) for V in (0, 15, 30)], G["unit"]["full"])
    same([u.electrical_power(V, T) for V in (0, 15) for T in (5.0, 20.0)], G["unit"]["el"])
