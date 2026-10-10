"""FALCO (notebook 33): the geometry and its layout, the transport pieces, the atmosphere, the mass and CG, the Li-ion
pack, the drive against its published point and over altitude (the brake region's signs), the climb and descent
targets, the crow increments, the derivative signs, the structural load cases and hand margins, the Chiron model over
the mountains (mass and inertia, the density at the altitude, the height field), a short flight and (slow) a whole
mission and a bird hunt.

Run: cd /home/user/vegeta && .venv/bin/python -m pytest -q notebooks/designs/tests/test_falco.py   (add -m "not slow")
"""
import math

import numpy as np
import pytest

import falco
import falco_flight as ff
import falco_structure as fst
import falco_systems as fs


@pytest.fixture(scope="module")
def a():
    return ff.aero()


@pytest.fixture(scope="module")
def dr():
    return fs.drive()


@pytest.fixture(scope="module")
def deriv(a):
    ci = fs.cg_inertia(fs.mass_table())
    return ff.derivatives(x_cg_m=ci["x_cg_m"], z_cg_m=ci["z_cg_m"], a=a)


# ----------------------------------------------------------------------------------------------- geometry
def test_layout_meets_the_brief():
    p = falco.resolve()
    L = falco.Falco.layout(p)
    assert L["S_ref"] == pytest.approx(0.60, rel=1e-9) and L["AR"] == pytest.approx(9.6, rel=1e-9)
    assert 0.45 < L["V_h"] < 0.65 and 0.05 < L["V_v"] < 0.09
    assert L["prop_clearance_boom"] > 40.0 and L["prop_ground_margin_resting"] > 20.0
    assert L["tail_le"] > L["prop_x"] + 200
    assert p["flap_y0"] > p["boom_y"] and p["flap_y1"] < p["wing_joint_y"] < L["y_aileron0"]
    assert falco.transport_check(p)["fits"] and L["longest_piece_mm"] <= 1150.0


def test_planform_split_tags_the_surfaces():
    pl = falco.planform_split()
    tags = pl["wing_tags"]
    assert tags.count("flap") == 2 and tags.count("aileron") == 2 and tags.count("centre") == 2
    area = sum(0.5 * np.linalg.norm(np.cross(q[2] - q[0], q[3] - q[1])) for q in pl["wing"])
    assert area == pytest.approx(falco.Falco.layout(falco.resolve())["S_ref"], rel=0.02)


def test_build_refuses_a_flap_across_the_joint():
    pytest.importorskip("cadquery")
    with pytest.raises(Exception):
        falco.Falco().generate(flap_y1=600.0)


def test_cad_volumes_match_a_rebuild():
    pytest.importorskip("cadquery")
    d = falco.Falco()
    for part in ("spar", "boom_fitting", "spar_joiner"):
        assert d.generate(part=part).measure()["volume"] == pytest.approx(fs.CAD[part], rel=0.01)


@pytest.mark.slow
def test_crow_pose_is_one_valid_solid():
    pytest.importorskip("cadquery")
    m = falco.Falco().generate(flap_deg=55.0, aileron_deg=-25.0).measure()
    assert m["valid"] and m["n_solids"] == 1 and m["dimensions"][1] == pytest.approx(2400.0, abs=1.0)


# ----------------------------------------------------------------------------------------------- the air, the masses, the pack
def test_atmosphere_is_the_isa():
    assert fs.atmosphere(0.0)["rho"] == pytest.approx(1.225, rel=1e-3)
    assert fs.atmosphere(4500.0)["rho"] == pytest.approx(0.777, rel=0.005)
    assert fs.density_altitude(fs.atmosphere(3000.0)["rho"]) == pytest.approx(3000.0, abs=5.0)
    hot = fs.atmosphere(3000.0, 20.0)
    assert hot["density_altitude_m"] > 3500.0
    assert fs.atmosphere(4000.0)["nu"] > 1.35 * fs.atmosphere(0.0)["nu"]          # thinner air: lower Reynolds numbers


def test_mass_and_cg():
    for k in ("6s3p-p45b", "6s4p-p45b"):
        t = fs.mass_table(k)
        ci = fs.cg_inertia(t)
        assert t.index.is_unique
        if k == fs.DEFAULT_PACK:                                   # the build's pack slides to the 28 % MAC target
            assert t.attrs["ballast_note"] == "" and ci["x_cg_frac_mac"] == pytest.approx(0.28, abs=1e-6)
        else:                                                      # the lighter pack at the bay's front: inside the CG range
            r = ff.cg_range(None, ci["mass_kg"])
            assert r["fwd_frac_mac"] + 0.05 < ci["x_cg_frac_mac"] < r["aft_frac_mac"] - 0.05
        assert ci["mass_kg"] == pytest.approx(t["mass [g]"].sum() / 1000)
        assert 4.3 < ci["mass_kg"] < 5.5
        assert ci["Izz"] > ci["Ixx"] > 0 and ci["Iyy"] > 0
        assert fs.battery_fits(fs.pack(k))["fits"]


def test_pack_sags_and_refuses_cold_charging():
    b = fs.pack("6s4p-p45b")
    assert b.energy_wh == pytest.approx(6 * 3.6 * 18.0)
    assert b.terminal_v(70.0, 0.5, -10.0) < b.terminal_v(70.0, 0.5, 25.0) < b.ocv(0.5)
    assert b.usable_wh(-10.0) < b.usable_wh(25.0) <= b.energy_wh
    assert b.charge_limit_a(0.0) == 0.0 and b.charge_limit_a(20.0, 0.5) > 0 and b.charge_limit_a(20.0, 0.99) == 0.0


# ----------------------------------------------------------------------------------------------- the drive
def test_drive_reproduces_its_published_point(dr):
    st = dr.at(0.0, 21.36 / dr.battery_v)
    assert st["rpm"] == pytest.approx(fs.PUBLISHED_STATIC["rpm"], rel=0.02)
    assert st["thrust"] == pytest.approx(fs.PUBLISHED_STATIC["thrust_n"], rel=0.03)
    assert st["current"] == pytest.approx(fs.PUBLISHED_STATIC["current"], rel=0.05)


def test_drive_scales_with_density_at_fixed_rpm(dr):
    s = fs.atmosphere(4500.0)["rho"] / fs.RHO0
    r0, r1 = dr.rows(15.0, fs.RHO0), dr.rows(15.0, fs.atmosphere(4500.0)["rho"])
    assert np.allclose(r1["thrust"], s * r0["thrust"]) and np.allclose(r1["torque"], s * r0["torque"])
    assert dr.max_thrust(15.0, fs.atmosphere(4500.0)["rho"]) < dr.max_thrust(15.0)


def test_brake_region_regenerates_and_drags(dr):
    fw = dr.freewheel(25.0)
    b = dr.brake(25.0, 1.0)
    assert fw["thrust"] < 0 and fw["electrical"] == 0.0
    assert b["thrust"] < fw["thrust"] - 3.0                        # the braked propeller drags more than the free one
    assert b["current"] < 0 and b["electrical"] < 0               # and charges the pack
    cold = dr.brake(25.0, 1.0, i_charge_max=0.0)
    assert cold["electrical"] == 0.0


# ----------------------------------------------------------------------------------------------- flight
def test_derivative_signs_and_trim(deriv):
    c = ff.coefficients(deriv)
    assert c["CLa"] > 4 and c["Cma"] < 0 and c["Cmq"] < 0 and c["Cmde"] < 0 and c["CLde"] > 0
    assert c["Cnb"] > 0 and c["Clb"] < 0 and c["Clp"] < 0 and c["Cnr"] < 0 and c["Clda"] > 0 and c["Cndr"] < 0
    assert c["CLdf"] > 0 and c["dCD_crow"] > 0.04
    assert abs(c["dCL_crow"]) < 0.1                                # crow: drag without losing the lift
    assert 0.05 < c["static_margin"] < 0.3
    t = ff.trim(deriv, 5.0, 20.0)
    assert -4 < t["alpha_deg"] < 4 and abs(t["elevator_deg"]) < 10


def test_climb_targets_met_at_sea_level_and_4500_m(dr):
    af = ff.airframe()
    assert ff.envelope(af, dr, 0.0)["roc_max"] >= 8.0
    assert ff.envelope(af, dr, 4500.0)["roc_max"] >= 5.0
    esc = ff.downdraft_escape(af, dr, heights=(4500.0,), downdrafts=(4.0,))
    assert esc.iloc[0, 1] > 1.0


def test_crow_and_brake_descend_ten_metres_a_second(dr, deriv):
    af = ff.airframe()
    t = ff.descent_table(af, dr, deriv, 3000.0)
    best = ff.best_descent(t)
    assert best.loc["crow + propeller brake", "sink [m/s]"] >= 10.0
    assert best.loc["crow + propeller brake", "EAS [m/s]"] <= ff.V_FE_EAS + 1e-9
    assert best.loc["crow", "sink [m/s]"] > 8.0


def test_regeneration_does_not_pay_in_energy(dr, deriv):
    af = ff.airframe()
    r = ff.regen_table(af, dr, deriv)
    warm = r.loc["warm pack"]
    assert warm["recovered [Wh]"].max() > 0 and warm["share of the pack [%]"].max() < 3.0
    assert (r.loc["cold pack (< 5 °C: no charging)"]["recovered [Wh]"] == 0).all()


# ----------------------------------------------------------------------------------------------- structure
def test_load_cases_and_hand_margins():
    c = fst.load_cases()
    assert c.attrs["n_limit"] >= ff.N_STRUCTURAL and c.attrs["n_ult"] == pytest.approx(1.5 * c.attrs["n_limit"])
    for df in (fst.wing_hand(cases=c), fst.joints_hand(cases=c)):
        margins = df[df.index.str.contains("margin")]["value"].astype(float)
        assert (margins > 0).all(), margins[margins <= 0]


def test_spar_segments_carry_the_root_moment():
    p = falco.resolve()
    L = falco.Falco.layout(p)
    segs = fst.spar_segment_forces(p, 1000.0)
    _, M = fst.moment_at(p, 1000.0, L["yc"] / 1000)
    assert sum(s["F"] * (s["y_bar"] - L["yc"]) / 1000 for s in segs) == pytest.approx(M, rel=0.02)


# ----------------------------------------------------------------------------------------------- the simulation
mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def lab():
    import falco_scenario as fsc
    return fsc.make_lab(fsc.Scenario(), log_geoms=False)


def test_chiron_model_matches_the_mass_table(lab):
    import falco_robot as fr
    ic = fr.inertia_check(lab, lab.robot.mass_table)
    assert abs(ic["mass_error"]) < 1e-6
    assert ic["mujoco_cg_x_mm"] == pytest.approx(ic["table_cg_x_mm"], abs=1.0)
    for k in ("Ixx_ratio", "Iyy_ratio", "Izz_ratio"):
        assert 0.9 < ic[k] < 1.1
    assert lab.model.opt.density == 0.0 and lab.model.opt.viscosity == 0.0


def test_height_field_is_the_massif(lab):
    hf = lab._tinfo
    assert hf["type"] == "hfield"
    assert hf["zmin"] == pytest.approx(float(lab.massif.height(*np.meshgrid(np.linspace(*lab.massif.extent[:2], 50), np.linspace(*lab.massif.extent[2:], 50))).min()), abs=30.0)
    assert lab.terrain.height(0.0, 0.0) == pytest.approx(1200.0, abs=1.0)


def test_hook_density_is_the_isa_at_the_altitude(lab):
    rho, sig = lab.aero.air(3000.0)
    assert rho == pytest.approx(fs.atmosphere(3000.0)["rho"])


def test_short_flight_launches_and_climbs():
    import falco_scenario as fsc
    scn = fsc.Scenario()
    lab_ = fsc.make_lab(scn, log_geoms=False)
    ep = fsc.run(lab_, scn, duration=60.0)
    d = fsc.timeseries(ep)
    assert d["agl"].iloc[-1] > 250.0 and d["vz"].iloc[-200:].mean() > 5.0
    phases = [n for _, n, note in ep.log["mission"] if note == "start"]
    assert phases[:3] == ["prelaunch", "launch", "climb"]
    assert d["rho"].iloc[-1] < d["rho"].iloc[0]
    ground = float(d["E_used_Wh"].iloc[0])
    assert ep.log["energy"]["E_used_wh"] == pytest.approx(ground + np.trapezoid(d["P_el_W"], d["t"]) / 3600, rel=0.03)


@pytest.mark.slow
def test_calm_mission_lands_on_the_meadow_with_the_reserve():
    import falco_scenario as fsc
    scn = fsc.Scenario()
    ep = fsc.run(fsc.make_lab(scn, log_geoms=False), scn)
    assert ep.outcome["success"], ep.outcome
    assert ep.log["energy"]["regen_wh"] > 0
    pt = fsc.phase_table(ep)
    assert pt.loc["descent", "max sink [m/s]"] > 10.0 and pt.loc["climb", "mean climb [m/s]"] > 6.0


def test_bird_tracker_assumes_the_planned_species_size():
    import falco_birds as fb
    bs = fb.BirdScenario()
    assert bs.size_prior() == pytest.approx(2.0) and bs.config().tracker.size_prior_m == pytest.approx(2.0)


@pytest.mark.slow
def test_bird_hunt_photographs_an_eagle():
    import falco_birds as fb
    ep = fb.run(fb.BirdScenario(), log_geoms=False)
    s = fb.summary(ep)
    assert s["photos"] > 0 and s["approaches < 10 m"] == 0
    assert ep.outcome["success"], ep.outcome
