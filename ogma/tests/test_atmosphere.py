"""og.atmosphere reproduces Vegeta's three ISA copies (golden/atmosphere.json, recorded at Vegeta 385738d)."""
import json
from pathlib import Path

import numpy as np
import pytest

from ogma import atmosphere as oa

G = json.loads((Path(__file__).with_name("golden") / "atmosphere.json").read_text())


def test_isa_is_boreas():
    got = [[oa.isa(h, d).pressure, oa.isa(h, d).temperature, oa.isa(h, d).density, oa.isa(h, d).speed_of_sound]
           for h in G["H"] for d in (0.0, 20.0)]
    assert np.allclose(got, G["isa"], rtol=1e-13)


def test_atmosphere_is_nisus_plus():
    got = [oa.atmosphere(h, d) for h in G["H"] for d in (0.0, -10.0, 20.0)]
    for a, b in zip(got, G["atmosphere"]):
        assert a.keys() == b.keys() and all(a[k] == pytest.approx(b[k], rel=1e-13) for k in a)
    assert np.allclose([oa.density_altitude(r) for r in (1.225, 1.0, 0.8, 0.6)], G["density_altitude"], rtol=1e-13)


def test_density_is_velutina_and_close_to_isa():
    assert np.allclose(oa.density(G["H"]), G["isa_density"], rtol=1e-13)
    assert np.allclose(oa.density(G["H"]), [oa.isa(h).density for h in G["H"]], rtol=1e-3)
