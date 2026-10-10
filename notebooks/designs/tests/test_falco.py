"""FALCO (notebook 35): the tractor's layout and the parked propeller, the CAD, the masses and the CG, the sourced
motor's fit and the drive, the derivatives with one fin, the hand margins, the FEA models, the NISUS+ comparison, the
parking in MuJoCo; slow: the calm mission and a bird hunt."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import falco
import falco_systems as fsy
import falco_flight as fl
import falco_structure as fst_
import nisus_plus as npl
import nisus_plus_flight as ff
import nisus_plus_systems as fs


@pytest.fixture(scope="module")
def P():
    return falco.resolve()


@pytest.fixture(scope="module")
def L(P):
    return falco.Falco.layout(P)


@pytest.fixture(scope="module")
def A(P):
    return fl.aero(P)


@pytest.fixture(scope="module")
def DR():
    return fsy.drive()


# ------------------------------------------------------------------------------------------------ layout
def test_tractor_layout_and_the_parked_propeller(P, L):
    assert L["prop_x"] < L["x_nose"] + 60.0 < 0.0                                   # the propeller's plane at the nose, ahead of the wing
    assert L["prop_ground_margin_parked"] >= P["park_clearance_min"]                # parked horizontal: the spinner clears the keel's bottom line
    assert L["prop_ground_margin"] < 0.0 and L["prop_ground_margin_resting"] < 0.0  # a blade straight down is in the ground: hence the parking
    assert L["n_fins"] == 1 and L["S_v"] == pytest.approx(P["fin_height"] * P["fin_chord"] * 1e-6)
    Lp = npl.NisusPlus.layout(npl.resolve())
    assert L["l_t"] == pytest.approx(Lp["l_t"], rel=0.05) and L["V_h"] == pytest.approx(Lp["V_h"], rel=0.05)   # NISUS+'s tail arm and volume kept
    assert L["longest_piece_mm"] <= 1150.0
    assert L["x_tube0"] > L["spar_x_root"] + 40.0 and L["x_pod_end"] > P["root_chord"]
    tc = falco.transport_check(P)
    assert tc["fits"]


def test_bays_do_not_overlap_along_the_fuselage(P, L):
    b = falco.Falco.bays(P)
    assert b["battery bay"][1] == L["x_tube0"]
    assert b["computer bay (the Orin)"][1] <= b["battery bay"][0]
    assert b["ESC and intake (behind the firewall)"][0] >= L["x_firewall"]
    assert fsy.battery_fits(fsy.pack(), P)["fits"]


# ------------------------------------------------------------------------------------------------ CAD
@pytest.mark.parametrize("part", ["aircraft", "pod", "tail_tube", "tail_socket", "tail_fitting", "tail", "motor_mount", "skid", "tail_tube_fea"])
def test_parts_are_one_valid_solid(part):
    r = falco.Falco().generate(part=part).measure()
    assert r["valid"] and r["n_solids"] == 1


def test_crow_pose_and_the_marking():
    d = falco.Falco()
    assert d.generate(part="aircraft", flap_deg=55.0, aileron_deg=-25.0).measure()["valid"]
    assert falco.resolve()["marking"] == "FALCO-109"
    plain = d.generate(part="pod", marking="").measure()["volume"]
    assert d.generate(part="pod").measure()["volume"] > plain                         # the raised letters add material


# ------------------------------------------------------------------------------------------------ mass and CG
def test_mass_table_balances_the_nose_motor_with_the_pack(P):
    t = fsy.mass_table(p=P)
    ci = fsy.cg_inertia(t, P)
    assert t.attrs["ballast_note"] == ""                                              # the pack slides to 28 % MAC inside its bay
    assert ci["x_cg_frac_mac"] == pytest.approx(0.28, abs=0.01)
    assert 5.0 < ci["mass_kg"] < 5.8
    assert "motor T-Motor AT5220-A KV220 (long shaft)" in t.index and t.loc["motor T-Motor AT5220-A KV220 (long shaft)", "x [mm]"] < -200.0
    assert not any("boom" in i for i in t.index if i.startswith("tail:") or i.startswith("booms"))


def test_packs_share_the_box():
    a, b = fsy.pack("8s3p-p45b"), fs.pack("6s4p-p45b")
    assert a.size_mm == b.size_mm and a.energy_wh == pytest.approx(b.energy_wh) and a.voltage > b.voltage


# ------------------------------------------------------------------------------------------------ the drive
def test_motor_fit_reproduces_the_published_point():
    m, fit = fsy.motor_model()
    P_ = fsy.PUBLISHED_STATIC
    assert fit["kt_check"] == pytest.approx(1.0, abs=0.02)                            # kt (I − I0) against the published torque
    assert fit["motor_efficiency_at_point"] == pytest.approx(0.82, abs=0.02)
    assert fit["check_part_load_torque_nm"] == pytest.approx(P_["part_load"]["torque_nm"], rel=0.10)
    assert 0.08 < fit["resistance_ohm"] < 0.2 and m.max_current_a == 70.0


def test_drive_options_pick_the_8s_pack():
    d = fsy.drive_options()
    sel = d.attrs["selected"]
    assert sel in d.index and d.loc[sel, "motor current [A]"] < 70.0 and d.loc[sel, "static thrust [N]"] > d.attrs["T_req_N"]
    six = "AT5220-A KV220 / 6S4P / 20x13"
    assert d.loc[six, "shaft power [W]"] < 0.6 * d.loc[sel, "shaft power [W]"]       # the KV220 on 6S has not the volts


def test_drive_beats_nisus_plus_statically_and_brakes(DR):
    base = fs.drive()
    assert DR.max_thrust(0.0) > base.max_thrust(0.0)
    assert DR.battery_v == pytest.approx(fsy.BATTERY_V)
    assert DR.brake(25.0, 1.0)["thrust"] < -3.0
    assert DR.locked_drag(15.0) > 0.0


# ------------------------------------------------------------------------------------------------ aerodynamics
def test_derivatives_with_one_fin(P, A):
    ci = fsy.cg_inertia(fsy.mass_table(p=P), P)
    d = fl.derivatives(P, x_cg_m=ci["x_cg_m"], a=A, h_m=3000.0)
    v = d["value"]
    assert A["n_fins"] == 1
    assert v["CLa"] > 4.5 and v["Cma"] < 0 and v["Cnb"] > 0.03 and v["Clb"] < 0 and v["Cmde"] < 0 and v["Cndr"] < 0
    assert 0.08 < v["static_margin"] < 0.25
    assert v["dCD_crow"] > 0.05


def test_installation_effects_are_small_for_a_tractor(P):
    r = fl.installation(20.0, 5.0, 3000.0, P)
    assert 0.0 <= r["w"] < 0.08 and 0.0 <= r["t"] < 0.08
    af = fl.airframe(p=P, h_m=3000.0)
    assert af.cd0 > af.cd0_buildup and af.cd0_installation < 0.004


def test_compare_with_nisus_plus_shows_the_gain(DR, A):
    af_f = fl.airframe(h_m=1200.0, a=A)
    c = fl.compare_with_nisus_plus(heights=(1200.0,), af_f=af_f, dr_f=DR)
    g = c["FALCO vs NISUS+ [%]"]
    assert g["best climb at 1200 m [m/s]"] > 5.0
    assert g["mission energy used [Wh]"] < -2.0
    assert g["drag area Cd0·S [m²]"] < 0.0


# ------------------------------------------------------------------------------------------------ structure
@pytest.fixture(scope="module")
def LC(P, DR, A):
    return fst_.load_cases(P, dr=DR, a=A)


def test_load_cases_carry_the_tractor_rows(LC):
    assert LC.attrs["T_fin"] > 0 and LC.attrs["M_gyro"] > 0 and LC.attrs["a_strike"] > 50.0
    assert LC.loc["blade-down clearance at rest [mm]", "value"] < 0 < LC.loc["parked propeller clearance at rest [mm]", "value"]


def test_hand_margins_positive(P, LC):
    w = fst_.wing_hand(P, LC)
    j = fst_.joints_hand(P, LC)
    for df in (w, j):
        m = df[df.index.str.contains("margin")]["value"].astype(float)
        assert (m > 0).all(), m[m <= 0]
    assert j.loc["tail tube: fin twist under its limit side load [deg] (G 20 GPa roll-wrapped)", "value"] < 3.0


def test_fea_models_build(tmp_path, P, LC):
    ms = fst_.models(tmp_path, P, LC)
    assert [m.name for m in ms] == ["spar_pullup", "spar_joiner", "tail_socket", "motor_mount_flight", "motor_mount_brake", "motor_mount_landing", "battery_tray"]
    tt = fst_.tail_tube_cases(tmp_path, P, LC)
    assert [c.name for c in tt] == ["tail_tube_A", "tail_tube_B", "tail_tube_modes"] and tt[0].frame.kind == "single"


# ------------------------------------------------------------------------------------------------ MuJoCo
def test_robot_matches_the_mass_table_and_parks():
    import falco_scenario as fsn
    import falco_robot as frb
    scn = fsn.scenario("calm")
    lab = fsn.make_lab(scn, log_geoms=False)
    chk = frb.inertia_check(lab, lab.aero.robot.mass_table)
    assert abs(chk["mass_error"]) < 1e-6 and abs(chk["mujoco_cg_x_mm"] - chk["table_cg_x_mm"]) < 5.0
    aero = lab.aero
    aero.reset(lab)
    assert len(aero.command) == 7 and not aero.parked
    aero.command = np.array([0, 0, 0, 0.0, 0, 0, 1.0])

    class T:                                                                         # a stand-in lab clock for the propulsor
        time = 0.0
    aero.lab = T()
    r0 = aero._propulsor(15.0, 0.0, 0.0, fs.RHO0)
    assert r0["mode"] == "parking" and not aero.parked
    T.time = aero.park_time + 0.1
    r1 = aero._propulsor(15.0, 0.0, 0.0, fs.RHO0)
    assert aero.parked and r1["rpm"] == 0.0 and r1["thrust"] < 0.0
    aero.command[6] = 0.0
    r2 = aero._propulsor(15.0, 0.6, 0.0, fs.RHO0)
    assert not aero.parked and r2["thrust"] > 0.0


@pytest.mark.slow
def test_calm_mission_lands_parked():
    import falco_scenario as fsn
    scn = fsn.scenario("calm")
    lab = fsn.make_lab(scn, log_geoms=False)
    ep = fsn.run(lab, scn)
    assert ep.outcome["success"], ep.outcome
    assert ep.log["parked_at_touchdown"] is True
    assert ep.outcome["park_time_s"] < 5.0


@pytest.mark.slow
def test_eagle_hunt_lands():
    import falco_birds as fb
    ep = fb.run(fb.bird_scenarios()[0], log_geoms=False)
    assert ep.outcome["success"], ep.outcome
    assert fb.summary(ep)["photos"] >= 1
