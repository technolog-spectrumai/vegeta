"""NISUS (notebook 31): the geometry and its layout, the mass and CG arithmetic, the electrical and mission energy
bookkeeping, the propulsion map against its published point, the lattice-based derivatives' signs, the structural
load cases, the Chiron model (mass and inertia against the mass table, the fluid model off, the force signs), a short
flight in MuJoCo and (slow) a whole mission with the energy-triggered return.

Run: cd /home/user/vegeta && .venv/bin/python -m pytest -q notebooks/designs/tests/test_nisus.py   (add -m "not slow")
"""
import math

import numpy as np
import pytest

import nisus
import nisus_flight as nf
import nisus_structure as nst
import nisus_systems as ns


# ----------------------------------------------------------------------------------------------- geometry
def test_layout_meets_the_brief_where_it_was_kept():
    p = nisus.resolve()
    L = nisus.Nisus.layout(p)
    assert L["S_ref"] == pytest.approx(0.28, rel=1e-9) and L["AR"] == pytest.approx(7.0, rel=1e-9)
    assert L["S_h"] == pytest.approx(0.056, rel=1e-9)
    assert 2 * p["boom_y"] == 320.0 and p["prop_diameter"] == pytest.approx(228.6)
    assert L["prop_clearance_boom"] > 35.0                           # the disc clears the booms
    assert L["prop_ground_margin_resting"] > 10.0                    # and the ground, resting on the keel and the bumpers
    assert L["prop_clearance_wing_te"] > 40.0 and L["tail_le"] > L["prop_x"] + 100
    assert 0.45 < L["V_h"] < 0.6 and 0.05 < L["V_v"] < 0.09
    assert 900 < L["overall_length"] < 1050


def test_brief_clearance_reproduced_at_the_starting_thrust_line():
    """The brief's 40.7 mm: 320 mm spacing, 10 mm booms, 9-inch disc in line with the booms."""
    L = nisus.Nisus.layout(nisus.resolve(motor_z=-20.0))
    assert L["prop_clearance_boom"] == pytest.approx(160 - 5 - 114.3, abs=1e-9)


def test_design_refuses_a_propeller_that_touches_a_boom():
    pytest.importorskip("cadquery")
    from vegeta.dedalus.design import BuildError
    with pytest.raises(BuildError):
        nisus.Nisus().generate(part="boom", boom_y=110.0)


def test_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    d = nisus.Nisus()
    for part in ("boom", "rear_spar", "motor_mount", "battery_tray", "boom_fitting"):
        assert d.generate(part=part).measure()["volume"] == pytest.approx(ns.CAD[part], rel=0.01), part


@pytest.mark.slow
def test_aircraft_is_one_valid_solid():
    pytest.importorskip("cadquery")
    g = nisus.Nisus().generate()
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    assert g.dimensions[1] == pytest.approx(1400.0, abs=0.5)


# ----------------------------------------------------------------------------------------------- mass, power, energy
@pytest.mark.parametrize("variant", ["OBS", "Zero"])
def test_mass_table_balances_and_counts_each_item_once(variant):
    t = ns.mass_table(variant)
    assert t.index.is_unique
    ci = ns.cg_inertia(t)
    assert 0.25 < ci["x_cg_frac_mac"] < 0.35                          # trimmed by the battery's place
    a = nf.aero()
    L = nisus.Nisus.layout(nisus.resolve())
    assert (a["x_np"] * 1000 - L["x_mac_le"]) / L["mac"] > ci["x_cg_frac_mac"] + 0.05     # statically stable
    assert ci["mass_kg"] == pytest.approx(t["mass [g]"].sum() / 1000)
    assert ci["Ixx"] > 0 and ci["Iyy"] > 0 and ci["Izz"] > ci["Ixx"]


def test_zero_carries_the_computer_and_obs_does_not():
    obs, zero = ns.mass_table("OBS"), ns.mass_table("Zero")
    assert "computer" not in set(obs["group"]) and "computer" in set(zero["group"])
    assert any("camera" in i for i in obs.index) and any("camera" in i for i in zero.index)     # neither is camera-free
    assert 200 < (zero["mass [g]"].sum() - obs["mass [g]"].sum()) < 330


def test_reference_energy_numbers_of_the_brief():
    b = ns.battery("gens-ace-3s-2200")
    assert b.energy_wh == pytest.approx(24.42)
    assert ns.JETSON_INSTALLATION_W * 600 / 3600 == pytest.approx(3.0)
    assert 3.0 / b.energy_wh == pytest.approx(0.123, abs=5e-4)
    assert 0.2 * b.energy_wh == pytest.approx(4.884) and 0.3 * b.energy_wh == pytest.approx(7.326)


def test_jetson_installation_is_counted_once():
    e = ns.electrical_loads("Zero")
    assert sum("Jetson installation" in i for i in e.index) == 1
    assert not any(i.startswith("Jetson Nano") or i.startswith("vision camera") for i in e.index)   # inside the 18 W line
    assert e.loc["TOTAL (battery side)", "battery-side average [W]"] - ns.electrical_loads("OBS").loc["TOTAL (battery side)", "battery-side average [W]"] \
        == pytest.approx(18.0 + 2.0 / 0.9, abs=1e-9)


def test_propulsion_map_reproduces_the_published_static_point():
    pm = ns.propulsion_map()
    st = pm.at(0.0, 1.0)
    assert st["thrust"] == pytest.approx(ns.PUBLISHED_STATIC["thrust_n"], rel=0.05)
    assert st["electrical"] == pytest.approx(ns.PUBLISHED_STATIC["power_w"], rel=0.15)
    assert pm.max_thrust(16.0) < st["thrust"]                       # in flight the propeller unloads


def test_mission_energy_phases_add_up_and_the_reserve_is_kept():
    pm = ns.propulsion_map()
    b = ns.battery("gens-ace-3s-2200")
    for v in ("OBS", "Zero"):
        m = ns.cg_inertia(ns.mass_table(v))["mass_kg"]
        r = ns.mission_energy(v, b, pm, {"mass_kg": m, "cd0": 0.031, "AR": 7.0, "oswald": 0.9, "S": 0.28})
        ph = r["phases"]
        assert ph["energy [Wh]"].sum() == pytest.approx(r["E_used_wh"])
        assert (ph["propulsion [Wh]"] + ph["electronics [Wh]"]).sum() == pytest.approx(r["E_used_wh"])
        assert ph["time [s]"].drop("ground operation (prelaunch)").sum() == pytest.approx(600.0)
        assert r["reserve_kept"] and r["trigger_wh"] > r["E_reserve_wh"]


# ----------------------------------------------------------------------------------------------- aerodynamics and structure
def test_derivative_signs():
    ci = ns.cg_inertia(ns.mass_table("Zero"))
    c = nf.coefficients(nf.derivatives(x_cg_m=ci["x_cg_m"], z_cg_m=ci["z_cg_m"]))
    assert c["CLa"] > 4 and c["Cma"] < 0 and c["Cmq"] < 0 and c["Cmde"] < 0 and c["CLde"] > 0
    assert c["Cnb"] > 0 and c["Clb"] < 0 and c["Clp"] < 0 and c["Cnr"] < 0 and c["Clda"] > 0 and c["Cndr"] < 0
    t = nf.trim(nf.derivatives(x_cg_m=ci["x_cg_m"]), ci["mass_kg"], 16.0)
    assert -4 < t["alpha_deg"] < 4 and abs(t["elevator_deg"]) < 10


def test_load_cases_and_hand_margins_are_positive():
    c = nst.load_cases()
    assert c.attrs["n_limit"] >= nf.N_STRUCTURAL and c.attrs["n_ult"] == pytest.approx(1.5 * c.attrs["n_limit"])
    w, j = nst.wing_hand(cases=c), nst.joints_hand(cases=c)
    margins = [float(v) for k, v in list(w["value"].items()) + list(j["value"].items()) if "margin" in k]
    assert margins and min(margins) > 0.0


def test_spar_forces_carry_the_whole_wing_lift_moment():
    """The segment forces' moment about the saddle edge equals the Schrenk root moment of one half (moment-equivalent)."""
    p = nisus.resolve()
    L = nisus.Nisus.layout(p)
    total = 100.0
    segs = nst.spar_segment_forces(p, total)
    y, l = nst.schrenk(p)
    yc = L["yc"] / 1000
    k = y >= yc
    M = total * np.trapezoid(l[k] * (y[k] - yc), y[k])
    M_seg = sum(s["F"] * (s["y_bar"] / 1000 - yc) for s in segs)
    assert M_seg == pytest.approx(M, rel=0.02)


# ----------------------------------------------------------------------------------------------- the simulation
mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def lab():
    import nisus_scenario as nsc
    return nsc.make_lab(nsc.Scenario("Zero"))


def test_chiron_model_matches_the_mass_table_and_has_no_fluid_model(lab):
    import nisus_robot as nr
    ic = nr.inertia_check(lab, lab.robot.mass_table)
    assert abs(ic["mass_error"]) < 1e-6
    assert ic["mujoco_cg_x_mm"] == pytest.approx(ic["table_cg_x_mm"], abs=1.0)
    assert ic["mujoco_cg_z_mm"] == pytest.approx(ic["table_cg_z_mm"], abs=1.0)
    for k in ("Ixx_ratio", "Iyy_ratio", "Izz_ratio"):
        assert 0.9 < ic[k] < 1.1
    assert lab.model.opt.density == 0.0 and lab.model.opt.viscosity == 0.0


def test_aero_hook_balances_weight_at_the_linear_trim(lab):
    c = lab.coeff
    t = nf.trim(lab.deriv, lab.ci["mass_kg"], 16.0)
    CL, CD, CY, Cl, Cm, Cn, st = lab.aero.coefficients(math.radians(t["alpha_deg"]), 0.0, 0.0, 0.0, 0.0,
                                                       np.array([0.0, math.radians(t["elevator_deg"]), 0.0]), 16.0)
    qS = 0.5 * 1.225 * 16 ** 2 * c["S"]
    assert qS * CL == pytest.approx(lab.ci["mass_kg"] * 9.81, rel=0.03)
    assert abs(Cm) < 0.01 and not st and CY == 0.0


def test_short_flight_launches_climbs_and_counts_energy():
    import nisus_scenario as nsc
    scn = nsc.Scenario("Zero", duration=40.0)
    lab_ = nsc.make_lab(scn, log_geoms=False)
    ep = nsc.run(lab_, scn, duration=40.0)
    d = nsc.timeseries(ep)
    assert d["h"].max() > 30.0 and d["V"].iloc[-1] > 12.0             # launched and climbing
    phases = [n for _, n, note in ep.log["mission"] if note == "start"]
    assert phases[:3] == ["prelaunch", "launch", "climb"]
    ground = float(d["E_used_Wh"].iloc[0])
    assert ep.log["energy"]["E_used_wh"] == pytest.approx(ground + np.trapezoid(d["P_el_W"], d["t"]) / 3600, rel=0.02)


@pytest.mark.slow
def test_calm_mission_returns_on_energy_and_lands_with_the_reserve():
    import nisus_scenario as nsc
    scn = nsc.Scenario("Zero", "calm")
    ep = nsc.run(nsc.make_lab(scn, log_geoms=False), scn)
    assert ep.log["return_reason"] == "energy"
    assert ep.outcome["success"], ep.outcome
