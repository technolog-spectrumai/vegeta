"""Kestrel-Q4T, the tandem: the layout rules, Munk's theorem in the lattice, the front-first stall rule, the hover
lever, the masses and the spar, the vehicle through both transitions, the CAD."""
import math

import pytest

import kestrel_flight as kf
import kestrel_systems as ks
import kestrel_tandem as ktd


@pytest.fixture(scope="module")
def LAY():
    return {v: ktd.layout(v) for v in ktd.VARIANTS}


def test_layout_rules_and_the_hover_lever(LAY):
    for v, L in LAY.items():
        rules = ktd.checks(L)
        if v.startswith("Q4T1-10"):
            assert rules and "no vertical lever" in rules[0] and L["hover_pitch_lever"] == 0.0
        else:
            assert rules == [], (v, rules)
            assert abs(L["hover_pitch_lever"] - (L["gap"] - ktd.PYLON)) < 1e-9
        assert L["prop_ground_clearance"] >= 0.03 and L["stagger"] == 3 * L["chord"]
    n, p = LAY["Q4T1N-10"], LAY["Q4T1P-10"]
    assert n["z_front"] < n["z_rear"] and p["z_front"] > p["z_rear"] and n["pylon"]["front"] > 0 and p["pylon"]["rear"] > 0


def test_munk_in_the_lattice(LAY):
    """Stagger buys no induced drag, the gap does: flat / staggered ≈ Prandtl's 1/((1+σ)/2); staggered ≈ the biplane."""
    a_flat, a_n, a_p = ktd.aero(LAY["Q4T1-10 (flat)"]), ktd.aero(LAY["Q4T1N-10"]), ktd.aero(LAY["Q4T1P-10"])
    ratio = a_flat["k"] / a_n["k"]
    assert abs(ratio - 2 / (1 + kf.prandtl_sigma(0.28))) < 0.12 * ratio
    assert abs(a_n["k"] - kf.aero()["biplane"]["k"]) < 0.08 * a_n["k"]
    assert abs(a_n["k"] - a_p["k"]) < 1e-9                                      # the straight-wake lattice cannot tell P from N
    assert 0.5 < a_flat["slope_rear"] < a_n["slope_rear"] < 0.9 and a_n["slope_front"] > 0.95      # the rear wing in the downwash, the front almost alone
    assert ktd.aero(LAY["Q4T1N-12"])["k"] < a_n["k"]


def test_front_wing_stalls_first_and_the_cl_max_rule(LAY):
    L = LAY["Q4T1N-10"]
    a = ktd.aero(L)
    ci = ks.cg_inertia(ktd.mass_table(L))
    st = ktd.stall(L, a, ci["x_cg"])
    assert st["stalls first"] == "front" and 0.6 < st["front share at trim"] < 0.72 and st["nose-down at the stall"]
    assert abs(st["CL max (aircraft)"] - ktd.WING_CL_MAX * 0.5 / st["front share at trim"]) < 1e-9
    assert 0.9 < st["CL max (aircraft)"] < 1.05 and 0.25 < st["static margin [c]"] < 0.35
    assert abs((a["x_np"] - ci["x_cg"]) / L["stagger"] - ktd.STATIC_MARGIN) < 0.01            # the battery found the CG the trim wants
    # a CG behind the pair's neutral point would make the rear wing stall first
    assert ktd.stall(L, a, a["x_np"] + 0.3 * L["stagger"])["stalls first"] == "rear"


def test_masses_spar_and_drag(LAY):
    for v, L in LAY.items():
        ci = ks.cg_inertia(ktd.mass_table(L))
        assert 1.6 < ci["mass_kg"] < 2.0, v
        sp = ktd.cantilever(L, mass_kg=ci["mass_kg"])
        assert sp["spar reserve"] > 3 and sp["tip deflection / half-span [%]"] < 4 and sp["first bending, with the motor [Hz]"] > 12
        d = ktd.drag_buildup(L)
        assert 0.015 < d["Cd0"] < 0.03 and "pod" in d["parts_m2"]
    assert ks.cg_inertia(ktd.mass_table(LAY["Q4T1N-12"]))["mass_kg"] > ks.cg_inertia(ktd.mass_table(LAY["Q4T1N-10"]))["mass_kg"]


def test_vehicle_cruise_takeoff_and_transitions():
    import kestrel_transition as kt
    veh = ktd.make_vehicle("Q4T1N-10")
    assert 0.95 < veh.aircraft_cl_max < 1.02 and len(veh.panels) == 4 and {p["wing"] for p in veh.panels} == {"front", "rear"}
    assert all(p["rotor"] == ("up" if p["wing"] == "rear" else "lo") for p in veh.panels)      # N: the rear wing is the high one
    c = ktd.cruise(veh)
    r16 = c.loc[16.0]
    assert r16["feasible"] and 10 < r16["L/D"] < 15 and 40 < r16["electrical [W]"] < 90 and abs(r16["elevator trim [deg]"]) < 10
    assert not c.loc[10.0]["feasible"]                                           # below the stall
    to = ktd.takeoff_roll(veh)
    assert to["lifts off"] and to["ground roll [m]"] < 15 and 9 < to["stall speed [m/s]"] < 11
    fwd = kt.simulate(veh, "forward", t_tr=3.0, dt=0.004).metrics
    back = kt.simulate(veh, "back", t_tr=3.0, dt=0.004).metrics
    assert fwd["pass"] and fwd["height loss [m]"] < 3 and math.isnan(fwd["diverged at [s]"])
    assert math.isnan(back["diverged at [s]"]) and back["pitch error max [deg]"] < 10 and back["hover reached at [s]"] > 0
    flat = ktd.make_vehicle("Q4T1-10 (flat)")
    assert flat.z_rotor == 0.0
    m = kt.simulate(flat, "back", t_tr=3.0, dt=0.004).metrics
    assert math.isnan(m["diverged at [s]"])                                       # the elevons alone keep it integrable


def test_elevator_power_and_hover_allocation(LAY):
    import kestrel_control as kc
    L = LAY["Q4T1N-10"]
    ci = ks.cg_inertia(ktd.mass_table(L))
    ep = ktd.elevator_power(L, ktd.aero(L), ci["x_cg"])
    assert ep["Cm_delta rear pair [1/rad]"] < -0.2 and 2 < ep["rear arm [c]"] < 2.6
    rk = kc.rank_table(kappa=0.012, cg=ci, L=L)
    assert (rk["rank"] == 4).any()
    rk_flat = kc.rank_table(kappa=0.012, cg=ks.cg_inertia(ktd.mass_table(LAY["Q4T1-10 (flat)"])), L=LAY["Q4T1-10 (flat)"])
    assert (rk_flat["rank"] < 4).all()                                            # no pitch by thrust in one plane


def test_cad_builds_and_exports(tmp_path):
    g = ktd.KestrelQ4T().generate(**ktd.cad_params("Q4T1N-10"), part="airframe")
    bb = g.shape.BoundingBox()
    assert 990 < bb.ylen < 1010 and 800 < bb.xmax < 900 and bb.zmin < -250 < -200 < bb.zmax
    f = ktd.export_stl(tmp_path, "Q4T1N-10", 4.0, 0.375)
    assert f.exists() and f.stat().st_size > 100_000
