"""The sea game's vessels as rigid bodies for the Yax export: their masses are the sizings', Barracuda's centre of
gravity is the one its planing model assumes, the registry names them."""
import pytest

import sea_robots as sr


def test_masses_are_the_sizings():
    import zarco
    from barracuda import Barracuda
    from esmeralda import Esmeralda
    assert sr.zarco().total_mass() == pytest.approx(1000 * (sum(zarco.LIGHTSHIP.values()) + sum(zarco.DEADWEIGHT.values())))
    assert sr.esmeralda().total_mass() == pytest.approx(1000 * Esmeralda.sizing()["mass_t"], abs=10)
    assert sr.barracuda().total_mass() == pytest.approx(Barracuda.sizing()["mass_half_kg"], abs=2)


def test_barracuda_lcg_is_the_planing_models():
    _, x_tank, cg = sr.barracuda_weights()
    assert cg["lcg_from_transom_m"] == pytest.approx(cg["design_lcg_m"], abs=1e-6)
    assert 0.3 < x_tank < 5.0                                   # the tank lies inside the hull
    assert -0.5 < cg["vcg_above_wl_m"] < 1.0


def test_registered():
    from vegeta.labkit import machines
    import os
    machines.discover(os.path.join(os.path.dirname(sr.__file__)))
    assert {"zarco", "esmeralda", "barracuda", "squilla", "resident_dock", "jacket"} <= set(machines.names())


def test_site_bodies_and_the_ketos_card():
    import json
    assert sr.squilla().total_mass() == pytest.approx(json.loads((sr.KETOS_CARDS / "squilla.card.json").read_text())["mass_kg"], abs=0.1)
    assert sr.resident_dock().total_mass() == 1800.0 and sr.jacket().total_mass() > 1e6

    class M:
        pass
    part = sr.ketos_card("barracuda")(M())
    assert part["ketos"]["card"]["name"] == "barracuda" and part["tags"] == {"data:ketos*": "ketos"}
