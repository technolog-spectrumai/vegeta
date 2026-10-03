"""AGUYA (notebook 29): the CAD builds and splits into jet_external's three STLs; the turbojet table, the mission with
fuel burn, the fuel and tank sizing.

Run: cd /home/user/vegeta && .venv/bin/python -m pytest -q notebooks/designs/tests/test_aguya.py   (~15 s)
"""
import math

import numpy as np
import pytest

import aguya_flight as F
from vegeta.boreas import microjet as mj


@pytest.fixture(scope="module")
def engine():
    return mj.from_catalogue("140 N class")


@pytest.fixture(scope="module")
def unit(engine):
    return F.jet_unit(engine)


@pytest.fixture(scope="module")
def plane(engine, unit):
    import aguya as ag
    return F.airframe(ag.for_engine(engine), unit)


def test_jet_unit_matches_the_cycle(engine, unit):
    assert unit.full(0.0)[0] == pytest.approx(142.0, rel=1e-3)
    assert unit.full(0.0)[1] == pytest.approx(450.0 / 60e3, rel=1e-3)
    T = 60.0
    ff, rpm, ok = unit.for_thrust(100.0, T)
    assert ok and engine.rpm_idle < rpm < engine.rpm_max
    assert mj.solve(engine, rpm, 100.0).thrust == pytest.approx(T, rel=0.05)
    assert unit.for_thrust(100.0, 1e4)[2] is False


def test_top_speed_and_drag(plane, unit):
    m = plane.dry_mass_kg + 1.0
    v = F.top_speed(plane, unit, m)
    assert 140 < v < 200
    assert unit.full(v)[0] == pytest.approx(plane.drag(v, m), rel=1e-3)


def test_mission_burns_the_fuel_and_keeps_the_reserve(plane, unit):
    f = F.fuel_for(plane, unit, 10000.0)
    assert f.feasible and 0 < f.fuel_kg < plane.tank_kg
    assert f.fuel_left_kg == pytest.approx(0.15 * f.fuel_kg, abs=0.01)
    burned = sum(v[1] for v in f.phases.values())
    assert burned == pytest.approx(f.fuel_kg - f.fuel_left_kg, rel=1e-6)
    assert f.track["mass"][-1] == pytest.approx(plane.dry_mass_kg + f.fuel_left_kg, rel=1e-6)
    assert np.all(np.diff(f.track["mass"]) <= 1e-12)
    assert f.phases["sample"][0] == pytest.approx(180.0)
    assert 10000.0 / f.dash_speed < f.time_to_fire_s < 10000.0 / f.dash_speed + 40
    far = F.fuel_for(plane, unit, 20000.0)
    assert far.fuel_kg > f.fuel_kg and far.time_to_fire_s > f.time_to_fire_s


def test_out_of_reach_and_tank_sizing(engine, plane, unit):
    import aguya as ag
    f = F.fuel_for(plane, unit, 60000.0)
    assert not f.feasible
    q, a, g = F.size_for(ag.for_engine(engine), unit, 30000.0)
    assert g.feasible and a.tank_kg >= g.fuel_kg and q["fuselage_diameter"] > 120


def test_cad_and_cfd_surfaces(engine, tmp_path):
    pytest.importorskip("cadquery")
    import aguya as ag
    from vegeta.aeromant.stl import Surface, read_stl
    p = ag.for_engine(engine)
    for part in ("aircraft", "wing", "nacelle"):
        g = ag.Aguya().generate(**dict(p, part=part))
        assert g.shape.isValid() and len(g.shape.Solids()) == 1
    out = ag.cfd_surfaces(p, tmp_path)
    assert out["faces"]["intake"] == 1 and out["faces"]["exhaust"] == 1
    s = {k: read_stl(v) for k, v in out["paths"].items()}
    assert s["exhaust"].area == pytest.approx(out["nozzle_area_m2"] * 1e6, rel=0.02)
    assert s["intake"].area == pytest.approx(out["intake_area_m2"] * 1e6, rel=0.02)
    assert Surface(np.vstack([x.triangles for x in s.values()])).volume > 0
    assert p["nozzle_diameter"] == pytest.approx(2 * math.sqrt(engine.a8 / math.pi) * 1000)
