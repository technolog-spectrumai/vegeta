"""The drones Montezuma flies on real rotor dynamics (``tools/consumers/montezuma.json``): the published
``exports/quad.yx2`` and ``exports/drako.yx2`` carry a ``propulsion`` block whose reduced constants hold the machine
in hover within the rotors' maximum thrust, with margin to manoeuvre. Only Yax and the published files are needed."""
import json
from pathlib import Path

import numpy as np
import pytest

yax = pytest.importorskip("yax")
from yax import tags as tg  # noqa: E402
ROOT = Path(__file__).resolve().parents[2]
G = 9.81
DRONES = {"quad": 4, "drako": 6}


@pytest.fixture(scope="module", params=sorted(DRONES))
def drone(request):
    m, _ = yax.read(ROOT / "exports" / f"{request.param}.yx2")
    return request.param, m


def test_rotors_and_layers(drone):
    name, m = drone
    blk = m.data["propulsion"]
    assert len(blk["rotors"]) == DRONES[name] and sorted(blk["reduced"]) == [r["name"] for r in blk["rotors"]]
    assert sum(r["spin"] for r in blk["rotors"]) == 0                              # yaw balanced at equal thrust
    assert blk["propellers"] and blk["motors"] and blk["battery"]
    for r in blk["rotors"]:
        assert r["propeller"] in blk["propellers"] and r["motor"] in blk["motors"]
        assert np.allclose(r["axis"], [0.0, 0.0, 1.0])
    for r, o in zip(blk["rotors"], m.data["ogma"]["rotors"]):
        red = blk["reduced"][r["name"]]
        assert red["kF"] * red["w_max"] ** 2 == pytest.approx(o["T_max"]) and red["kM"] == o["kM"]
        assert red["tau_down"] >= red["tau_up"] > 0


def test_hover_within_max_thrust(drone):
    """kF ω² = m g / n at an ω below ω_max, and the rotors' T_max together exceed 1.5 m g."""
    name, m = drone
    blk = m.data["propulsion"]
    weight = sum(lk.mass or 0.0 for lk in m.links) * G
    reds = list(blk["reduced"].values())
    t_max = sum(r["kF"] * r["w_max"] ** 2 for r in reds)
    assert t_max > 1.5 * weight
    for r in reds:
        w_hover = np.sqrt(weight / len(reds) / r["kF"])
        assert 0 < w_hover < r["w_max"]
        assert r["kF"] * w_hover ** 2 * len(reds) == pytest.approx(weight)


def test_drako_hover_power_is_the_sizing():
    """Drako's reduced constants at hover (shaft power kM T ω per rotor) through the sizing's drive efficiency give the
    sizing's empty hover power (drako.Drako.sizing: 2334 W at 23.03 kg, momentum theory with the BEMT's FoM)."""
    m, _ = yax.read(ROOT / "exports" / "drako.yx2")
    reds = list(m.data["propulsion"]["reduced"].values())
    mass = sum(lk.mass or 0.0 for lk in m.links)
    assert mass == pytest.approx(23.03, abs=0.01)
    T = mass * G / len(reds)
    shaft = sum(r["kM"] * T * np.sqrt(T / r["kF"]) for r in reds)
    assert shaft / 0.85 == pytest.approx(2334.0, rel=0.03)


def test_montezuma_is_served_and_reads_propulsion():
    req_path = ROOT / "tools" / "consumers" / "montezuma.json"
    req = tg.Requirements.load(req_path)
    assert "00prop" in (*req.required, *req.optional)
    wants = json.loads(req_path.read_text())["machines"]
    man = json.loads((ROOT / "exports" / "MANIFEST.json").read_text())
    for name in DRONES:
        assert name in wants and man["machines"][name]["consumers"]["montezuma"]
        m, _ = yax.read(ROOT / "exports" / f"{name}.yx2")
        assert tg.check(m, req, "yx2").ok
