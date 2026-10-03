"""(Copied from notebooks/designs/tests with only the imports changed: the promoted copy must pass the same tests.)
Electric ducted fan (notebook 25): the rotor and the housing as single solids, the shroud and the convex lip, the
nozzle (its exit area as built, and the exit as its throat), the stators' place behind the rotor and their roots at any
stagger, the movie outline, and the rotor running free in the housing with the axial gap it asks for.

Run: cd /home/user/vegeta && xvfb-run -a .venv/bin/python -m pytest -q notebooks/designs/tests/test_ducted_fan.py
"""
import math

import cadquery as cq
import numpy as np
import pytest

from assemblies.components import ducted_fan as df


@pytest.fixture(scope="module")
def rotor():
    return df.EDFRotor().generate().shape                     # 12 blades: ~7 s, built once


@pytest.fixture(scope="module")
def housing():
    return df.EDFHousing().generate().shape


def _inside(shape, x, r, phi_deg=90.0 / 7):
    """Is the point at axial x, radius r, angle phi (from +y towards +z; between vanes 0 and 1 by default) inside?"""
    a = math.radians(phi_deg)
    return shape.isInside(cq.Vector(x, r * math.cos(a), r * math.sin(a)), 1e-4)


def _starts_at(shape, x, r, solid_above: bool, d=0.01):
    """Does the solid start (solid_above: hollow below r, solid above) or end at radius r, within +-d?"""
    return _inside(shape, x, r - d) != solid_above and _inside(shape, x, r + d) == solid_above


def test_rotor_is_one_solid_with_the_spinner_upstream(rotor):
    bb = rotor.BoundingBox()
    assert rotor.isValid() and len(rotor.Solids()) == 1
    for lo, hi in ((bb.xmin, bb.xmax), (bb.ymin, bb.ymax)):
        assert lo == pytest.approx(-45.0, abs=0.01) and hi == pytest.approx(45.0, abs=0.01)   # tips trimmed to R
    assert bb.zmin == pytest.approx(-7.0 - 22.0, abs=0.01)                                     # spinner on -Z
    assert not rotor.isInside(cq.Vector(0, 0, 5.0)) and rotor.isInside(cq.Vector(0, 0, -8.0))  # bore closed upstream
    assert bb.zmax == pytest.approx(df.rotor_trailing_edge(df.EDFHousing().resolve())["te_x"], abs=0.01)


def test_housing_is_one_solid_with_the_shroud_over_the_rotor(housing):
    g = df.housing_geometry({})
    bb = housing.BoundingBox()
    assert housing.isValid() and len(housing.Solids()) == 1
    assert bb.xmin == pytest.approx(-31.5, abs=0.01) and bb.xmax == pytest.approx(g["tail_end_x"], abs=0.01)
    assert g["shroud_radius"] == pytest.approx(45.8) and g["max_radius"] == pytest.approx(45.8 + 2.5)
    assert bb.ymax == pytest.approx(g["max_radius"], abs=0.01) and bb.zmin == pytest.approx(-g["max_radius"], abs=0.01)
    for x in (-20.0, -7.0, 0.0, 8.0, 30.0):                                  # the lip's end to past the stators
        assert _starts_at(housing, x, 45.8, True) and not _inside(housing, x, 48.4)   # the wall: 2.5 mm thick


def test_lip_is_convex_and_stands_no_higher_than_the_nacelle():
    p = df.EDFHousing().resolve()
    dp = df.EDFHousing.duct_profile(p)
    g = df.housing_geometry({})
    nacelle = g["nacelle_radius"]
    assert np.all(dp["outer"][:, 1] <= nacelle + 1e-9) and np.all(dp["nose"][:, 1] <= nacelle + 1e-9)
    assert g["highlight_radius"] == pytest.approx(45.8 + 2.5 - 0.75) and dp["nose"][-1, 0] == pytest.approx(-31.5)
    # round the lip, from the nose's top over the highlight and down the bell-mouth to the shroud: every turn the same
    # way (convex: no flare, no waist), the tangent continuous (no corner sharper than the sampling's)
    x_s = df.EDFHousing.layout(p)["x_s"]
    inner = dp["inner"]
    lip = np.vstack([dp["nose"], inner[1:np.searchsorted(inner[:, 0], x_s + 1e-9)]])
    assert lip[-1] == pytest.approx([x_s, 45.8])
    e = np.diff(lip, axis=0)
    cross = e[:-1, 0] * e[1:, 1] - e[:-1, 1] * e[1:, 0]
    turn = np.degrees(np.arctan2(cross, np.sum(e[:-1] * e[1:], axis=1)))
    assert np.all(cross > 0) and turn.max() < 12.0
    assert np.degrees(np.arctan2(*e[-1][::-1])) == pytest.approx(0.0, abs=4.0)          # level onto the shroud


def _annulus(over):
    """The annulus between the duct's and the centre body's profiles over the nozzle, sampled (x, area mm^2)."""
    p = df.EDFHousing().resolve(**over)
    L = df.EDFHousing.layout(p)
    inner, body = df.EDFHousing.duct_profile(p)["inner"], df.EDFHousing.body_profile(p)
    x = np.linspace(L["x_t0"], L["x_e"], 1501)
    noz = inner[inner[:, 0] >= L["x_t0"] - 1e-9]
    return x, math.pi * (np.interp(x, noz[:, 0], noz[:, 1]) ** 2 - np.interp(x, body[1:, 0], body[1:, 1], right=0.0) ** 2)


def test_nozzle_exit_area_is_as_requested(housing):
    g = df.housing_geometry({})
    assert g["exit_area_ratio"] == pytest.approx(0.9, rel=1e-9)
    assert g["fan_area"] == pytest.approx(math.pi * (45.0 ** 2 - 20.0 ** 2))
    x, R_e, r_t = g["exit_x"] - 0.01, g["exit_radius"], g["tail_radius_at_exit"]    # on the solid, just inside the exit
    assert _starts_at(housing, x, R_e, True) and _starts_at(housing, x, r_t, False, d=0.05)
    assert math.pi * ((R_e - 0.01) ** 2 - (r_t + 0.05) ** 2) / g["fan_area"] == pytest.approx(0.9, rel=0.03)
    p = df.EDFHousing().resolve()                                             # mid-nozzle, the CAD has the law's radius
    xm = 0.5 * (g["nozzle_start_x"] + g["exit_x"])
    assert _starts_at(housing, xm, float(df.EDFHousing.nozzle_radius(p, xm)[0]), True, d=0.02)
    g2 = df.housing_geometry({"exit_area_ratio": 0.75, "tail_length": 20.0})    # tail cone ends ahead of the exit
    assert g2["tail_radius_at_exit"] == 0.0 and g2["exit_area_ratio"] == pytest.approx(0.75, rel=1e-9)


@pytest.mark.parametrize("over", [{}, {"exit_area_ratio": 0.8}, {"exit_area_ratio": 1.0}, {"exit_area_ratio": 0.3},
                                  {"exit_area_ratio": 0.75, "tail_length": 20.0}, {"tail_length": 26.0}])
def test_nozzle_converges_to_its_exit(over):
    """The flow area falls all the way to the exit: the exit is the throat, the requested area its smallest (the
    polylines the CAD revolves ripple about the law by < 1e-4 of the fan area between their vertices)."""
    g = df.housing_geometry(over)
    x, a = _annulus(over)
    assert np.all(np.diff(a) <= 1e-4 * g["fan_area"]) and a.min() == pytest.approx(a[-1], rel=1e-12)
    assert g["throat_x"] == pytest.approx(g["exit_x"]) and g["throat_area"] == pytest.approx(g["exit_area"], rel=1e-9)
    assert g["exit_area_ratio"] == pytest.approx(over.get("exit_area_ratio", 0.9), rel=1e-6)
    p = df.EDFHousing().resolve(**over)
    assert df.EDFHousing.nozzle_radius(p, g["exit_x"])[1] < 0.0                # the duct wall converges at the exit


def test_wide_nozzle_and_max_radius():
    """Past the shroud's annulus the nozzle diffuses: the throat is its start, and the widest point is its outer wall."""
    over = {"exit_area_ratio": 1.5, "lip_radius": 3.0}
    g = df.housing_geometry(over)
    dp = df.EDFHousing.duct_profile(df.EDFHousing().resolve(**over))
    assert g["exit_area_ratio"] == pytest.approx(1.5) and g["max_radius"] == pytest.approx(dp["closed"][:, 1].max())
    assert g["max_radius"] == pytest.approx(dp["outer"][:, 1].max()) and g["max_radius"] > g["nacelle_radius"] + 2.0
    assert g["throat_x"] == pytest.approx(g["nozzle_start_x"]) and g["throat_area_ratio"] == pytest.approx(g["nozzle_start_area_ratio"])


def test_stators_stand_stator_gap_behind_the_rotor(housing):
    g = df.housing_geometry({})
    assert g["stator_le_x"] - g["rotor_te_x"] == pytest.approx(14.0)
    assert g["stator_te_x"] - g["stator_le_x"] == pytest.approx(16.0)
    root = df.rotor_trailing_edge(df.EDFHousing().resolve())["root_te_x"]      # the blade-root stub behind the hub face
    assert root > 7.0 and g["body_start_x"] == pytest.approx(root + 1.0) and g["body_clearance"] == pytest.approx(1.0)
    r = 0.5 * (g["stator_root_r"] + g["stator_tip_r"])                       # vane 0 stands along +y (phi = 0)
    assert _inside(housing, g["stator_le_x"] + 0.05, r, 0.0) and not _inside(housing, g["stator_le_x"] - 0.05, r, 0.0)
    assert _inside(housing, g["stator_te_x"] - 0.2, r, 0.0) and not _inside(housing, g["stator_te_x"] + 0.05, r, 0.0)
    assert sum(_inside(housing, 0.5 * (g["stator_le_x"] + g["stator_te_x"]), r, 360.0 * k / 7) for k in range(7)) == 7


@pytest.mark.parametrize("stagger", [60.0])
def test_staggered_vanes_stay_attached(stagger):
    """At any stagger the vanes run from inside the centre body into the duct wall (no slot at the root, nothing
    through the nacelle): along the radius through points of the chord line, everything is solid."""
    sh = df.EDFHousing().generate(stagger_deg=stagger).shape
    g = df.housing_geometry({"stagger_deg": stagger})
    assert sh.isValid() and len(sh.Solids()) == 1 and sh.BoundingBox().ymax == pytest.approx(g["nacelle_radius"], abs=0.01)
    a = math.radians(stagger)
    for u in (0.1, 0.95):                                    # near the leading and the trailing edge: root, mid, tip
        x, z = g["stator_le_x"] + 16.0 * u * math.cos(a), 16.0 * u * math.sin(a)
        for r in (19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.5, 33.0, 45.0, 45.5, 45.8, 46.1, 46.5):
            assert sh.isInside(cq.Vector(x, math.sqrt(r * r - z * z), z), 1e-4), (u, r)
        assert not sh.isInside(cq.Vector(x, math.sqrt(48.4 ** 2 - z * z), z), 1e-4)
    with pytest.raises(ValueError, match="wider"):                          # a section wider than the body: refused
        df.housing_geometry({"stagger_deg": stagger, "stator_chord": 30.0})
    with pytest.raises(ValueError, match="lip"):                            # a nose too round for the wall: refused
        df.housing_geometry({"lip_nose_radius": 1.3})


def test_outline_polygons():
    o = df.outline({})
    assert len(o["side"]) == 4 and len(o["axial"]) == 2 + 7                  # duct x2, body, the vane at phi = 0
    up, down = o["side"][0], o["side"][1]
    r_e = df.housing_geometry({})["exit_radius"] / 1000                      # the duct walls leave the axis free
    assert up[:, 1].min() == pytest.approx(r_e) and down[:, 1].max() == pytest.approx(-r_e)
    assert up[:, 0].min() == pytest.approx(-0.0315) and up[:, 0].max() == pytest.approx(0.0675)
    ring = o["axial"][0]
    rr = np.hypot(ring[:, 0], ring[:, 1])
    assert rr.min() == pytest.approx(0.0458) and rr.max() == pytest.approx(0.0483)
    assert all(len(poly) >= 3 and np.all(np.isfinite(poly)) for key in o for poly in o[key])
    assert len(df.outline({"stator_vanes": 6})["side"]) == 5                # an even count has a vane below too


def test_rotor_runs_free_in_the_housing(rotor, housing):
    rx = rotor.rotate((0, 0, 0), (0, 1, 0), 90)                               # Z -> X: the flow along +x
    bb = rx.BoundingBox()
    assert bb.xmin == pytest.approx(-29.0, abs=0.01)
    assert bb.xmax == pytest.approx(df.housing_geometry({})["body_start_x"] - 1.0, abs=0.01)   # axial_gap, free
    assert rx.intersect(housing).Volume() == pytest.approx(0.0, abs=1e-6)
    assert rx.translate(cq.Vector(0, 1.0, 0)).intersect(housing).Volume() > 0.1    # 1 mm off centre: the tips touch
    # pushed back along the axis (the shroud is a cylinder there) the blade roots meet the centre body after axial_gap
    assert rx.translate(cq.Vector(0.95, 0, 0)).intersect(housing).Volume() == pytest.approx(0.0, abs=1e-6)
    assert rx.translate(cq.Vector(1.05, 0, 0)).intersect(housing).Volume() > 1e-3
