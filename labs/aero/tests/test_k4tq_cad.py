"""K4TQ Stage 2: the CAD builds in its three configurations, nothing interferes and nothing internal leaves the skin,
the CAD places the parts with the sizing's own transform, the fold path is clear, the CAD masses are current and fed
back, the Dedalus design and the exports work. On the mocks. The exact Boolean checks (every pair in three configurations,
the fold path) are marked slow: ``python tools/lab.py test aero -- -m slow -k k4tq`` runs them (minutes)."""
import json
import math

import numpy as np
import pytest

pytest.importorskip("cadquery")

import k4tq
import k4tq_cad as c
import k4tq_sizing as s


@pytest.fixture(scope="module")
def VEH():
    return s.vehicle(k4tq.MAIN)


@pytest.fixture(scope="module")
def CONFIGS(VEH):
    p, it = VEH["p"], VEH["items"]
    return {k: c.assembly(p, t, f, items=it) for k, (t, f) in {"cruise": (0.0, 0.0), "hover": (90.0, 0.0), "folded": (90.0, p["fold_transport_deg"])}.items()}


def test_every_part_is_a_valid_solid(CONFIGS):
    for name, parts in CONFIGS.items():
        assert len(parts) > 90, name
        bad = [k for k, v in parts.items() if not v.solid.isValid() or v.solid.Volume() <= 0]
        assert bad == [], (name, bad)
        for which in ("front", "rear"):
            for side in ("L", "R"):
                for part in ("stub shaft", "bearing 1", "bearing 2", "gear", "tilt servo", "inner knuckle", "outer knuckle", "hinge pin", "spar", "wing panel"):
                    assert f"{part} {which} {side}" in parts
    assert "locking pin front L" in CONFIGS["cruise"] and "locking pin front L" not in CONFIGS["folded"]      # pulled to fold


@pytest.mark.slow
def test_no_interference_and_nothing_outside_the_skin(CONFIGS):
    for name, parts in CONFIGS.items():
        assert c.interferences(parts) == [], name
    assert c.outside_skin(CONFIGS["cruise"]) == []


def test_cad_uses_the_sizings_transform(VEH, CONFIGS):
    p, L = VEH["p"], VEH["L"]
    for cfg, (tilt, fold) in {"cruise": (0.0, 0.0), "hover": (math.pi / 2, 0.0)}.items():
        for which in ("front", "rear"):
            for side, sname in ((-1, "L"), (1, "R")):
                pose = k4tq.prop_pose(L, which, side, tilt, fold)
                centre = CONFIGS[cfg][f"spinner {which} {sname}"].solid.Center()
                expect = (pose["hub"] + 0.010 * pose["axis"]) * c.MM
                assert np.allclose([centre.x, centre.y, centre.z], expect, atol=0.05), (cfg, which, sname)


def test_symmetry_and_hover_discs(CONFIGS):
    hv = CONFIGS["hover"]
    for which in ("front", "rear"):
        l, r = hv[f"wing panel {which} L"].solid.Center(), hv[f"wing panel {which} R"].solid.Center()
        assert abs(l.x - r.x) < 1e-6 and abs(l.y + r.y) < 1e-6 and abs(l.z - r.z) < 1e-6
        bb = hv[f"disc {which} R"].solid.BoundingBox()
        assert bb.zlen < 5.0                                                  # the disc lies flat: thrust vertical


@pytest.mark.slow
def test_fold_path_is_clear_in_steps(VEH):
    p, it = VEH["p"], VEH["items"]
    for f in (20.0, 45.0, 70.0):
        assert c.interferences(c.assembly(p, 90.0, f, items=it)) == [], f


def test_cad_masses_are_current_and_fed_back(VEH):
    doc = json.loads(k4tq.CAD_MASSES.read_text())
    assert doc["fingerprint"] == k4tq.geometry_fingerprint(VEH["p"]) and VEH["p"]["cad_mass_status"] == "CAD"
    roots = [i for i in VEH["items"] if i["item"].startswith("wing root")]
    assert len(roots) == 4 and all(abs(i["mass_kg"] - doc["cad_mass"]["wing root (tilt + fold)"]) < 1e-12 for i in roots)
    rows = c.masses(VEH["p"], items=VEH["items"])
    assert abs(sum(r["mass_kg"] for r in rows) - c.mass_comparison(rows, VEH["items"])["total"]["CAD [kg]"]) < 1e-9
    root_now = c.cad_mass_params(VEH["p"], VEH["items"])["wing root (tilt + fold)"]
    assert abs(root_now - doc["cad_mass"]["wing root (tilt + fold)"]) < 0.005                  # the JSON matches the CAD


def test_packing_puts_the_cg_on_target(VEH):
    p = VEH["p"]
    assert not p["packing_clipped"] and abs(VEH["mass"]["x_cg"] - VEH["x_cg_target"]) < 1e-3
    xa, xb = (x * p["fuselage_length"] for x in p["battery_x"])
    assert xa < VEH["L"]["wings"]["front"]["x_pivot"] < xb < p["cargo_station"] * p["fuselage_length"] < VEH["L"]["wings"]["rear"]["x_pivot"]


def test_dedalus_design_and_tube(VEH, CONFIGS):
    g = c.K4TQ().generate(what="oml", tilt_deg=0.0)
    assert g.measure()["volume"] > 0
    t = c.tube_fit(CONFIGS["folded"])
    assert 250.0 < t["tube inner diameter [mm]"] < 600.0 and t["length [mm]"] > 1000.0 * VEH["p"]["fuselage_length"]
