"""ALBATROS over a flooded town (notebook 50, sims/albatros_flood.py): the Émerillon carrier and its bay, the town's
model, the MuJoCo run with saddle bags and with single boxes, the judge, the cold chain.

Run: cd /home/user/vegeta/labs/aero && ../../.venv/bin/python -m pytest -q tests/test_albatros_flood.py   (~1 min: two runs)
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
import albatros_flood as fl  # noqa: E402


@pytest.fixture(scope="module")
def saddle():
    return fl.run(fl.Scene())


def test_the_carrier_and_its_bay():
    car = fl.carrier()
    assert 1.0 < car.mass_kg < 1.8 and car.gross_kg == pytest.approx(car.mass_kg + 0.5)
    assert car.thrust_max(0.0) > 2.5 * car.gross_kg * 9.81 and car.thrust_max(40.0) < car.thrust_max(0.0)
    assert 150 < car.hover_power() < 500 and car.hover_power(car.mass_kg) < car.hover_power()
    opt = fl.bay_option(car, 6, 2)
    b = opt.bay()
    assert opt.count == 6 and opt.cargo_kg == pytest.approx(3.0) and opt.dropped_kg == pytest.approx(6 * car.gross_kg)
    assert b["length_m"] < fl.bay_option(car, 6, 1).bay()["length_m"] and b["width_m"] > fl.bay_option(car, 6, 1).bay()["width_m"]


def test_the_town_is_the_same_town_every_time():
    a, b = fl.Scene(seed=7), fl.Scene(seed=7)
    assert [(q.kind, q.x, q.y) for q in a.buildings] == [(q.kind, q.x, q.y) for q in b.buildings]
    assert {q.kind for q in a.buildings} == {"house", "school", "block", "church", "tower"}
    r = a.recipients()
    assert len(r) == 6 and sum(x["mode"] == "drop" for x in r) == 5 and r[-1]["mode"] == "land"
    for x in r:
        q = x["building"]
        assert q.people >= 2 and x["target"][2] >= q.wall_h > a.flood_m          # the roofs are above the water
    m = mujoco.MjModel.from_xml_string(fl.mjcf(a))
    assert m.nbody > 40 and m.opt.timestep == pytest.approx(0.002)
    with pytest.raises(ValueError):
        fl.Scene(payload="parachute")


def test_saddle_bags_reach_every_roof(saddle):
    s, d = fl.summary(saddle), saddle["deliveries"]
    assert s["delivered"] == s["roofs"] == 6 and s["people reached"] == s["people waiting"] and s["carriers landed"] == 6
    drops = [r for r in d if r["mode"] == "saddle-bag drop"]
    assert len(drops) == 5 and all(r["outcome"] == "astride the ridge" for r in drops)
    assert all(r["from the people [m]"] <= saddle["scene"].reach_m for r in drops)
    assert all(0 < r["shock [g]"] <= fl.SHOCK_LIMIT_G["vaccine"] for r in drops)
    assert all(5 < r["after the release from Albatros [s]"] < 40 for r in d)
    assert d[-1]["outcome"] == "standing on the roof"
    car = saddle["scene"].carrier
    assert all(0 < r["energy used [Wh]"] < 0.5 * car.pack_wh for r in d)
    for pl in saddle["pilots"]:
        ev = pl.events
        assert ev["released"] < ev["transit"] < ev["approach"] < ev["hover"] < ev["landed"] and pl.mode == "done"
    for tr in saddle["track"].values():
        assert np.all(np.isfinite(tr["p"])) and tr["p"][:, 2].min() > saddle["scene"].flood_m      # nobody flew into the water
        assert tr["v"].max() < 45.0
    assert fl.table(saddle).shape[0] == 6


def test_a_single_box_slides_off_a_pitched_roof():
    box = fl.run(fl.Scene(payload="box", drones=4))
    d = box["deliveries"]
    drops = [r for r in d if r["mode"] == "box drop"]
    assert len(drops) == 3 and sum(r["outcome"] == "in the water" for r in drops) >= 2
    assert fl.summary(box)["delivered"] < len(d)


def test_the_cold_chain():
    c = fl.cold_chain()
    assert c["in_range_until_roof"] and 2.0 <= c["min_c"] and 1.0 < c["hours_on_roof_in_range"] < 12.0
    bare = fl.cold_chain(pcm_kg=0.0)
    assert not bare["in_range_until_roof"] or bare["hours_on_roof_in_range"] < c["hours_on_roof_in_range"]
    assert bare["min_c"] < c["min_c"]                                               # without the pack the cold bay gets to the vials
