"""og.mass reproduces Vegeta's two inertia calculators on their real tables (golden/mass.json, recorded from
Vegeta's code — the source of truth): kestrel_systems.cg_inertia (point masses + rods) and nisus_systems.cg_inertia
(point masses + plate wing and tail, ellipsoidal pod shell, boom rods keyed on the rows' names)."""
import json
from pathlib import Path

import numpy as np
import pytest

from ogma import mass as om

G = json.loads((Path(__file__).with_name("golden") / "mass.json").read_text())


def close(props, g, rtol=1e-12):
    assert props.mass == pytest.approx(g["mass_kg"], rel=rtol)
    assert np.allclose(props.com, g["cg"], rtol=rtol, atol=1e-15)
    assert np.allclose(props.inertia, g["I"], rtol=rtol, atol=1e-15)


def test_kestrel_budget():
    g = G["kestrel"]
    b = om.Budget([om.Item(i["item"], i["mass_kg"], (i["x"], i["y"], i["z"]),
                           om.Rod(*i["rod"]) if i["rod"] else None) for i in g["items"]])
    close(b.properties(), g)


def nisus_shape(name, p, mac_mm):
    """The shape rules of nisus_systems.cg_inertia, as Ogma shapes (mm in the design, m here)."""
    if name.startswith("wing:"):
        return om.Plate(a=mac_mm / 1000, b=p["span"] / 1000, axes="xy")
    if name.startswith("pod:") and "shell" in name:
        return om.Shell(p["pod_length"] / 2000, p["pod_width"] / 2000, p["pod_height"] / 2000)
    if name.startswith("booms:"):
        return om.Rod("x", p["boom_length"] / 1000)
    if name.startswith("tail:"):
        return om.Plate(a=p["tail_chord"] / 1000, b=p["tail_span"] / 1000, axes="xy")
    return None


@pytest.mark.parametrize("variant", ["Zero", "OBS"])
def test_nisus_budget(variant):
    g = G[f"nisus_{variant}"]
    b = om.Budget([om.Item(r["item"], r["mass_g"] / 1000, (r["x_mm"] / 1000, r["y_mm"] / 1000, r["z_mm"] / 1000),
                           nisus_shape(r["item"], g["p"], g["mac"])) for r in g["rows"]])
    close(b.properties(), g)


def test_solid_shapes_and_algebra():
    m = 2.0
    assert np.allclose(om.Box(0.2, 0.4, 0.6).inertia(m), np.diag([m * (0.16 + 0.36) / 12, m * (0.04 + 0.36) / 12, m * (0.04 + 0.16) / 12]))
    assert np.allclose(om.Cylinder(0.1, 0.3, "z").inertia(m), np.diag([m * (3 * 0.01 + 0.09) / 12] * 2 + [0.5 * m * 0.01]))
    assert np.allclose(om.Sphere(0.1).inertia(m), 0.4 * m * 0.01 * np.eye(3))
    a = om.Budget([om.Item("a", 1.0, (0, 0, 0), om.Box(0.1, 0.1, 0.1))]).properties()
    b = om.Budget([om.Item("b", 3.0, (1, 0, 0))]).properties()
    both = om.Budget([om.Item("a", 1.0, (0, 0, 0), om.Box(0.1, 0.1, 0.1)), om.Item("b", 3.0, (1, 0, 0))]).properties()
    s = a + b
    assert s.mass == both.mass and np.allclose(s.com, both.com) and np.allclose(s.inertia, both.inertia)


def test_place_for_cg_is_kestrels_battery_algebra():
    g = G["kestrel"]
    items = [om.Item(i["item"], i["mass_kg"], (i["x"], i["y"], i["z"])) for i in g["items"] if i["item"] != "battery"]
    batt = next(i for i in g["items"] if i["item"] == "battery")
    x, clipped = om.place_for_cg(om.Budget(items), batt["mass_kg"], target=0.0, bay=(batt["x"], batt["x"] + 0.1))
    assert clipped and x == batt["x"]
    free, c2 = om.place_for_cg(om.Budget(items), batt["mass_kg"], target=0.2, bay=(-10, 10))
    full = om.Budget(items + [om.Item("battery", batt["mass_kg"], (free, 0, 0))]).properties()
    assert not c2 and full.com[0] == pytest.approx(0.2)
