"""K4TQ wing sections: the NeuralFoil cache covers both wings' phases, the spar rule and the recommendation, the CFD
specs and the slab geometry (on the mocks: the polars come from the committed cache)."""
import numpy as np
import pytest

import k4tq
import k4tq_sections as ks
import k4tq_sizing as sz


@pytest.fixture(scope="module")
def RES():
    return ks.study(sz.vehicle(k4tq.MAIN))


def test_conditions_tandem_split(RES):
    c = RES["conditions"]
    assert c["front"]["cruise"]["cl"] > 1.5 * c["rear"]["cruise"]["cl"]          # the CG loads the front wing
    assert c["rear"]["cruise"]["Re"] > c["front"]["cruise"]["Re"]                # the larger rear chord


def test_thickness_and_spar_rule(RES):
    assert abs(ks.thickness_at("NACA 0012") - 0.12 * 0.99) < 0.01 and abs(ks.thickness_at("NACA 4415") - 0.15) < 0.01
    front = RES["tables"]["front"]
    assert (front.loc[~front["spar fits"].astype(bool), "total"].astype(float) == 0).all()


def test_recommendation_is_sound(RES):
    rec = RES["recommendation"]
    assert set(rec) == {"front", "rear"}
    for w, n in rec.items():
        row = RES["tables"][w].loc[n]
        assert bool(row["spar fits"]) and float(row["least confidence"]) > 0.9
    assert abs(float(RES["tables"]["front"].loc["NACA 0012", "hover cl (α 0)"])) < 1e-3                     # symmetric: no side pull


def test_cfd_specs_and_slab(RES, tmp_path):
    pytest.importorskip("cadquery")
    specs = ks.cfd_specs([RES["recommendation"]["front"]], RES["conditions"], alphas=(4.0,))
    (k, v), = specs.items()
    stl = ks.slab_stl(tmp_path, v["name"], v["alpha_deg"], chord_m=v["chord_m"], AR=v["AR"])
    assert stl.exists() and stl.stat().st_size > 1000
