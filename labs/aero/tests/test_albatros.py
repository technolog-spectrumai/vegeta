"""ALBATROS (notebook 50): the bay's multirotors, the scaled turbojets, the three layouts sized and flown, the gust and
deck numbers, the choice, and the CAD.

Run: cd /home/user/vegeta/labs/aero && ../../.venv/bin/python -m pytest -q tests/test_albatros.py   (~2 min: four sizings)
"""
import math

import numpy as np
import pytest

import albatros_bay as B
import albatros_flight as F
import albatros_gust as Gu

REQ = F.Requirements(cargo_kg=5.0, radius_km=150.0)          # a short mission: the sizing is the slow part


@pytest.fixture(scope="module")
def option():
    return B.options(REQ.cargo_kg)["hexacopter"]


@pytest.fixture(scope="module")
def designs(option):
    return {k: F.size(k, REQ, option) for k in ("catapult", "vectored", "lift_cruise")}


# ------------------------------------------------------------------------------------------------ the bay
def test_multirotor_mass_closes_and_grows_with_the_cargo():
    h = B.size_multirotor(10.0, 6)
    assert sum(h.budget.values()) == pytest.approx(h.mass_kg, abs=1e-3)
    assert 4.0 < h.empty_kg < 10.0 and h.rotor_out_ok and 0.4 < h.prop_diameter_m < 0.8
    assert B.size_multirotor(5.0, 6).empty_kg < h.empty_kg
    assert not B.size_multirotor(3.0, 4).rotor_out_ok
    with pytest.raises(ValueError):
        B.size_multirotor(1.0, 2)


def test_one_hexacopter_against_three_quadcopters():
    o = B.options(10.0)
    one, three, cams = o["hexacopter"], o["three quadcopters"], o["three camera quadcopters"]
    assert one.cargo_kg == pytest.approx(10.0) and three.cargo_kg == pytest.approx(10.0)
    assert three.dropped_kg > one.dropped_kg                       # the fixed equipment is paid three times
    assert three.largest_item_kg == pytest.approx(10.0 / 3)
    assert cams.cargo_kg < 1.0                                     # camera quadcopters do not carry cargo
    b1, b3 = one.bay(), three.bay()
    assert b3["length_m"] > b1["length_m"] and b3["fuselage_width_m"] < b1["fuselage_width_m"]
    d1, d3 = B.delivery(one), B.delivery(three)
    assert d1["all of the cargo"] > d3["all of the cargo"] and d3["nothing"] < d1["nothing"]


# ------------------------------------------------------------------------------------------------ the engines
def test_scaled_engines_follow_the_cycle():
    e, u0, uh = F.units("400 N class", 6000.0)
    es, s0, sh = F.scaled("400 N class", 1.5, 6000.0)
    assert s0.full(0.0)[0] == pytest.approx(1.5 * u0.full(0.0)[0]) and s0.full(0.0)[1] == pytest.approx(1.5 * u0.full(0.0)[1])
    assert es.mass_kg == pytest.approx(1.5 * e.mass_kg) and es.tip_speed(es.rpm_max) == pytest.approx(e.tip_speed(e.rpm_max))
    assert uh.full(100.0)[0] < 0.7 * u0.full(100.0)[0]             # thin air at 6000 m
    (name, scale), lift = F.pick_engines(F.CONCEPTS["vectored"], 2000.0, F.Requirements())
    assert lift is not None and 0.6 < scale < 1.5 and 0.6 < lift[1] < 1.5
    assert F.pick_engines(F.CONCEPTS["catapult"], 2000.0, F.Requirements())[1] is None


# ------------------------------------------------------------------------------------------------ sized and flown
def test_the_take_off_mass_closes(designs, option):
    for d in designs.values():
        f = d.flight
        assert d.feasible, d.notes
        assert sum(d.budget.values()) <= d.mtow_kg + 1e-6 and sum(d.budget.values()) == pytest.approx(d.mtow_kg, abs=0.5)
        assert f.fuel_left_kg == pytest.approx(REQ.reserve * f.fuel_kg, abs=0.1)
        burned = sum(v["fuel"] for v in f.phases.values())
        assert burned == pytest.approx(f.fuel_kg - f.fuel_left_kg, rel=1e-6)
        assert f.landing_mass_kg == pytest.approx(d.mtow_kg - burned - option.dropped_kg, rel=1e-6)
        out = sum(v["dist"] for k, v in f.phases.items() if k in ("launch", "transition out", "climb out", "cruise out", "descent to the vessel"))
        assert out == pytest.approx(REQ.radius_km * 1000, rel=1e-3)
        assert np.all(np.diff(f.track["t"]) > 0)


def test_what_vertical_lift_costs(designs):
    cat, vec, lc = designs["catapult"], designs["vectored"], designs["lift_cruise"]
    assert cat.mtow_kg < vec.mtow_kg and cat.flight.fuel_kg < vec.flight.fuel_kg   # nothing carried for a hover
    assert cat.mtow_kg < lc.mtow_kg and cat.battery_kg == 0 and lc.battery_kg > 0 and lc.flight.battery_wh > 0
    assert "launch" in cat.flight.phases and "slow pass and release" in cat.flight.phases
    assert not any("hover" in k or "transition" in k for k in cat.flight.phases)
    thrust_borne = lambda f: sum(v["fuel"] for k, v in f.phases.items() if "hover" in k or "take-off" in k or "landing" in k)   # noqa: E731
    assert thrust_borne(vec.flight) > 5 * thrust_borne(lc.flight)                   # jets holding the weight
    assert "restart" in vec.flight.phases and "restart" not in cat.flight.phases
    assert lc.airframe.cd0 * lc.wing_area_m2 > vec.airframe.cd0 * vec.wing_area_m2  # the booms and the stopped rotors
    la = F.launch(cat)
    assert la["launch_speed"] == pytest.approx(1.2 * la["stall_speed"]) and 25 < la["stall_speed"] < 40
    assert la["stroke_with_10ms_wind_m"] < la["catapult_stroke_m"] < 15 and la["approach_speed"] < la["launch_speed"]
    assert 50 < la["runway_roll_m"] < 2000


def test_a_longer_trip_needs_a_bigger_aircraft(designs, option):
    near = designs["vectored"]
    far = F.size("vectored", F.Requirements(cargo_kg=5.0, radius_km=220.0), option)
    assert far.mtow_kg > near.mtow_kg and far.flight.fuel_kg > near.flight.fuel_kg


# ------------------------------------------------------------------------------------------------ gusts, deck, water
def test_the_wing_in_a_gust():
    assert Gu.lift_slope(10, 20) < Gu.lift_slope(10, 0) < 2 * math.pi
    assert Gu.lift_slope(6, 0) < Gu.lift_slope(12, 0)
    light = Gu.gust_load(60.0, 100.0, chord=0.3, aspect_ratio=10, sweep_deg=20)
    heavy = Gu.gust_load(250.0, 100.0, chord=0.3, aspect_ratio=10, sweep_deg=20)
    assert heavy["dn"] < 0.3 * light["dn"] and 0.7 < heavy["K_g"] < 0.88
    a = Gu.gust_alleviation(250.0, 100.0, chord=0.3, aspect_ratio=10, sweep_deg=20)
    assert a["dn_passive"] == pytest.approx(heavy["dn"], rel=0.1) and a["alleviation"] > 0.3 and a["flap_peak_deg"] <= 15.0
    v = Gu.vectoring_authority(400.0, 1.5, 0.25, 1.6)
    assert 20 < v["crossover_speed"] < 80


def test_hover_deck_and_water(designs):
    lc, vec = designs["lift_cruise"], designs["vectored"]
    w = Gu.afloat(designs["catapult"])
    assert w["floats"] and w["GM_m"] > 0 and w["rotor_above_water_m"] is None
    for d in (lc, vec):
        m = d.flight.landing_mass_kg
        g = Gu.hover_gust(d, mass_kg=m)
        assert g["peak_excursion_m"] < 1.0 and abs(g["x"][-1]) < 0.05 and g["wind_limit"] > 25
        h = Gu.deck_heave(d, mass_kg=m)
        assert h["rms_error_m"] < 0.25 and np.all(np.isfinite(h["error"]))
        w = Gu.afloat(d, m)
        assert w["floats"] and w["GM_m"] > 0 and 0 < w["draft_m"] < d.params["fuselage_height"] / 1000
        assert Gu.deck_stand(d)["static_tip_deg"] > 30
    assert Gu.hover_controls(lc)["vertical"][-1][1] < Gu.hover_controls(vec)["vertical"][-1][1]
    assert Gu.afloat(lc, 1e5)["floats"] is False


def test_the_choice(designs):
    two = {k: designs[k] for k in ("catapult", "vectored")}
    c = F.choose(two)
    assert c["choice"] == "catapult" and c["by_fuel"][0] == "catapult" and not c["failed"]["catapult"]
    assert c["notes"]["vectored"] and c["notes"]["catapult"]
    t = Gu.scorecard(two)
    assert t.loc["catapult end speed [m/s]"].iloc[1] == "-" and t.loc["hover wind limit [m/s]"].iloc[0] == "-"


# ------------------------------------------------------------------------------------------------ the CAD
def test_cad(designs):
    pytest.importorskip("cadquery")
    import albatros as al
    lc, cat = designs["lift_cruise"], designs["catapult"]
    g = al.Albatros().generate(**cat.params)
    assert g.shape.isValid() and len(g.shape.Solids()) == 1 and cat.params["lift_rotors"] == 0
    assert len(al.outline(cat.params)["rotors"]) == 0
    p = lc.params
    vol = {}
    for part in ("aircraft", "aircraft_open", "wing", "nacelle", "multirotor"):
        g = al.Albatros().generate(**dict(p, part=part))
        assert g.shape.isValid() and len(g.shape.Solids()) == 1, part
        vol[part] = g.shape.Volume()
    assert vol["aircraft_open"] < vol["aircraft"]                                   # the bay is cut out
    L = al.Albatros.layout(p)
    assert L["bay_x0"] < L["x_ac"] < L["bay_x1"]                                    # the bay's load leaves at the centre of gravity
    assert p["span"] ** 2 / (lc.wing_area_m2 * 1e6) == pytest.approx(REQ.aspect_ratio, rel=1e-6)
    db = al.drag_buildup(p, 110.0, mach=0.33)
    assert "stopped lift rotors" in db["parts_m2"] and db["cd0"] == pytest.approx(lc.airframe.cd0, rel=1e-3)
    assert "stopped lift rotors" not in al.drag_buildup(designs["vectored"].params, 110.0)["parts_m2"]
    g = al.Albatros().generate(**designs["vectored"].params)
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    with pytest.raises(ValueError):
        al.Albatros.layout(al.Albatros().resolve(**dict(p, bay_width=p["fuselage_width"])))
    o = al.outline(p)
    assert len(o["rotors"]) == 4 and len(o["side"]) >= 4
