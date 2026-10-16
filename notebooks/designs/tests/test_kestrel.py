"""Kestrel-4Q (notebook 38): the zero-stagger layout and its rules, the CAD, the masses and the CG, the hold point (the
allocation's rank for each sense of rotation, the attainable moments, a motor out), the biplane's lattice against
Prandtl's biplane factor, the drag of the bracing, the full-range polar, the slipstream, the transitions both ways
through the stall (and what fails without elevons), the CFD cases with four rotor disks and the CFD fed back into the
transition model, the truss and the ground stance.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_kestrel.py      (add -m "not slow")
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kestrel
import kestrel_control as kc
import kestrel_flight as kf
import kestrel_structure as kst
import kestrel_systems as ks
import kestrel_transition as kt


@pytest.fixture(scope="module")
def P():
    return kestrel.resolve()


@pytest.fixture(scope="module")
def L(P):
    return kestrel.Kestrel4Q.layout(P)


@pytest.fixture(scope="module")
def CI():
    return ks.cg_inertia(ks.mass_table())


@pytest.fixture(scope="module")
def DR():
    return ks.drive_table()


@pytest.fixture(scope="module")
def VEH():
    return kt.make_vehicle()


# ------------------------------------------------------------------------------------------------ layout and CAD
def test_zero_stagger_equal_wings_and_the_rules(P, L):
    assert "stagger" not in {p.name for p in kestrel.Kestrel4Q.parameters}             # zero stagger is built in, not a setting
    assert L["S_ref"] == pytest.approx(2 * L["S_wing"]) and L["gap"] == pytest.approx(P["gap"] / 1000)
    assert kestrel.Kestrel4Q.checks(P) == []
    assert L["prop_gap"] * 1000 >= P["prop_clearance"] and L["prop_to_pod"] * 1000 >= P["prop_clearance"]
    zs = sorted(m["z"] for m in L["motors"].values())
    assert zs == pytest.approx([-L["h"], -L["h"], L["h"], L["h"]])                     # two vertically separated pairs
    assert len({m["x"] for m in L["motors"].values()}) == 1                            # all four in one propeller plane
    assert sum(m["y"] for m in L["motors"].values()) == pytest.approx(0.0)             # mirror symmetry


def test_checks_refuse_overlapping_propellers(P):
    bad = dict(P, gap=240.0)
    assert any("clear each other" in m for m in kestrel.Kestrel4Q.checks(bad))
    assert any("pod" in m for m in kestrel.Kestrel4Q.checks(dict(P, motor_y=150.0)))


def test_rotate_matches_the_cad_convention():
    x, y, z = kestrel.rotate((0.0, 5.0, 0.0), 90.0, 200.0)                             # the nose tip, pitched up 90° about x = 200
    assert (x, y, z) == pytest.approx((200.0, 5.0, 200.0))


def test_cad_builds_symmetric_with_both_wings_at_one_station(P):
    pytest.importorskip("cadquery")
    d = kestrel.Kestrel4Q()
    wing = d.generate(part="wing").measure()
    assert wing["bbox_min"][0] == pytest.approx(P["wing_le_x"], abs=0.5)
    assert wing["bbox_min"][2] > 0 and wing["center_of_mass"][2] == pytest.approx(P["gap"] / 2, abs=0.5)
    whole = d.generate(part="airframe").measure()
    assert whole["valid"] and abs(whole["center_of_mass"][1]) < 1.0                    # mirror-symmetric
    assert whole["dimensions"][1] == pytest.approx(P["span"], abs=2.0)
    with pytest.raises(Exception):
        d.generate(gap=200.0)


# ------------------------------------------------------------------------------------------------ masses and drive
def test_mass_table_and_cg(L, CI):
    assert 1.4 < CI["mass_kg"] < 2.0
    assert CI["x_cg"] == pytest.approx(L["x_le"] + ks.X_CG_TARGET_FRAC * L["chord"], abs=1e-6)
    assert not CI["battery_clipped"] and abs(CI["y_cg"]) < 1e-9
    assert CI["Ixx"] > CI["Iyy"] and CI["Izz"] > CI["Iyy"]                              # the span dominates roll and yaw


def test_drive_table_is_consistent(DR):
    assert DR.T_max[0] > 8.0 and np.all(np.diff(DR.T_max) < 0)                          # full thrust falls with the inflow
    assert 0.008 < DR.kappa[0] < 0.03                                                    # Q/T of a 9-inch propeller [m]
    assert DR.power(4.0, 0.0) > DR.power(2.0, 0.0) > 0.0


@pytest.mark.slow
def test_stored_drive_table_matches_boreas(DR):
    fresh = ks.drive_table(recompute=True)
    assert np.allclose(fresh.T_max, DR.T_max, rtol=0.01) and np.allclose(fresh.kappa, DR.kappa, rtol=0.01)


# ------------------------------------------------------------------------------------------------ the hold point
def test_only_the_diagonal_pairing_controls_the_hover(CI, DR):
    tab = kc.rank_table(kappa=float(DR.kap(0.0)), cg=CI)
    assert tab.loc[kc.BASELINE_SPIN, "rank"] == 4
    for name in kc.SPINS:
        if name != kc.BASELINE_SPIN:
            assert tab.loc[name, "rank"] == 3, name                                     # one axis lost
    assert bool(tab.loc["side columns alike (M1, M3 | M2, M4)", "mirror-symmetric"])    # symmetric, and still uncontrollable


def test_fore_aft_offsets_carry_no_lever(P, CI):
    B = kc.allocation(P, kappa=0.015, cg=CI)
    moved = dict(CI, x_cg=CI["x_cg"] + 0.1)                                              # move the CG along the thrust axis
    assert np.allclose(kc.allocation(P, kappa=0.015, cg=moved), B)


def test_attainable_moments_in_the_hover(CI, DR):
    B = kc.allocation(kappa=float(DR.kap(0.0)), cg=CI)
    att = kc.attainable(B, CI["mass_kg"] * ks.G, DR.T_min, DR.T_max[0])
    for ax in ("L", "M", "N"):
        assert att[ax][0] < 0 < att[ax][1]
    assert abs(att["N"][0]) > abs(att["M"][0]) > 5 * abs(att["L"][0])                   # the heading lever is the weak one
    h = kestrel.layout()["h"]
    assert att["M"][1] <= 4 * h * (DR.T_max[0] - DR.T_min) / 2 + 1e-6


def test_a_motor_out_leaves_an_unbalanced_torque(CI, DR):
    tab = kc.motor_out(mass_kg=CI["mass_kg"], cg=CI)
    assert (tab["unbalanced torque about the axis [N·m]"].abs() > 0.1).all()


# ------------------------------------------------------------------------------------------------ aerodynamics
@pytest.fixture(scope="module")
def A():
    pytest.importorskip("cadquery")                                                     # peregrine_flight's lattice imports the CAD module
    return kf.aero()


def test_biplane_lattice_against_prandtl(A, L):
    # Prandtl's biplane: induced drag factor (1 + σ) / 2 of the monoplane, σ ≈ 0.35 at gap/span 0.28 (tables: 0.33-0.37)
    f = A["factors"]["induced drag vs the monoplane"]
    assert 0.64 < f < 0.74
    assert 0.85 < A["factors"]["lift slope vs two isolated wings"] < 0.95                # the gap's mutual downwash
    assert A["biplane"]["upper share"] == pytest.approx(0.5, abs=0.01)                  # zero stagger, no decalage: equal shares
    assert A["biplane"]["x_np"] == pytest.approx(L["x_ac"], abs=0.01 * L["chord"] + 2e-3)


def test_neutral_point_and_the_propellers(A, CI):
    npd = kf.neutral_point(x_cg=CI["x_cg"], a=A)
    assert npd["Cma_props"] > 0 and npd["Cma_pod"] > 0                                  # both destabilising
    assert 0.05 < npd["static_margin"] < 0.15
    bound = kf.prop_normal_force_slope(16.0, 0.8, bound=True)
    assert bound > 5 * kf.prop_normal_force_slope(16.0, 0.8)                            # the momentum bound is far above the estimate


def test_bracing_drag_and_fairings():
    round_ = kf.drag_buildup()
    faired = kf.drag_buildup(faired=True)
    assert round_["bracing_share"] > 0.5 > faired["bracing_share"]
    assert faired["Cd0"] < 0.6 * round_["Cd0"]


def test_full_range_polar_shape():
    pol = kf.Polar(cl_alpha=4.1, cd0=0.03, k=0.07, cl_max=0.81, cd90=1.24)
    a = np.radians([0.0, 5.0, 45.0, 90.0, 180.0])
    cl, cd, cm = pol.coefficients(a)
    assert cl[0] == pytest.approx(0.0) and cl[1] == pytest.approx(4.1 * math.sin(a[1]) * math.cos(a[1]), rel=0.02)
    assert cd[3] == pytest.approx(0.03 + 1.24, rel=0.01) and abs(cl[3]) < 1e-6           # broadside: a flat plate
    assert cl[2] == pytest.approx(0.62, abs=0.03)                                        # CD90 / 2 at 45°
    assert cm[3] < 0 and cm[0] == pytest.approx(0.0)                                     # mid-chord pressure at 90°: nose down


def test_glauert_induced_velocity():
    A = kestrel.layout()["A_disc"]
    vh = math.sqrt(4.0 / (2 * kf.RHO * A))
    assert kf.induced_velocity(4.0, 0.0, 0.0, A) == pytest.approx(vh, rel=1e-3)
    V = 15.0                                                                             # axial climb: vᵢ = −V/2 + √((V/2)² + vₕ²)
    assert kf.induced_velocity(4.0, V, 0.0, A) == pytest.approx(-V / 2 + math.sqrt((V / 2) ** 2 + vh ** 2), rel=1e-3)
    vi = kf.induced_velocity(4.0, 10.0, math.radians(90.0), A)                            # edgewise: vᵢ √(V² + vᵢ²) = vₕ²
    assert vi * math.sqrt(100.0 + vi ** 2) == pytest.approx(vh ** 2, rel=1e-3)


# ------------------------------------------------------------------------------------------------ the transitions
def test_slipstream_lowers_the_blown_angle(VEH):
    V, a = 4.0, math.radians(40.0)
    s = kt.slipstream(VEH, 5.0, V * math.cos(a), V * math.sin(a))
    blown = math.atan2(V * math.sin(a), V * math.cos(a) + s["dv"])
    assert math.degrees(blown) < 15.0 and 0.0 < s["blown_span"] < VEH.L["span"]


def test_forward_transition_passes_through_the_stall(VEH):
    ep = kt.simulate(VEH, "forward", t_tr=3.0)
    m = ep.metrics
    assert m["pass"], m
    assert m["time stalled, free stream [s]"] > 0.3                                     # it does cross the stall...
    assert m["α blown max while stalled [deg]"] < math.degrees(VEH.pol.alpha_stall) + 5  # ...with the blown span attached
    assert m["speed at the end [m/s]"] > 14.0


def test_back_transition_reaches_the_hover(VEH):
    ep = kt.simulate(VEH, "back", t_tr=3.0)
    m = ep.metrics
    assert m["pass"], m
    assert m["hover reached at [s]"] < ep.log["t"][-1] and m["speed at the end [m/s]"] < 0.5


def test_without_elevons_the_roll_check_fails(VEH):
    bare = kt.make_vehicle(elevons=False)
    ep = kt.simulate(bare, "back", t_tr=3.0)
    assert ep.metrics["roll demand / rotors' authority"] > 1.0 and not ep.metrics["roll holds"]


# ------------------------------------------------------------------------------------------------ CFD
def test_cfd_points_and_cases_prepare(VEH, tmp_path):
    pytest.importorskip("cadquery")
    import kestrel_cfd as kcfd
    from vegeta import aeromant
    eps = [kt.simulate(VEH, k, t_tr=3.0) for k in ("forward", "back")]
    pts = kcfd.points(eps, mass_kg=VEH.mass)
    assert {pt["kind"] for pt in pts} == {"forward", "back", "hover"}
    assert len({(pt["source"], round(pt["t"], 4)) for pt in pts}) == len(pts)              # no sample twice
    spec = kcfd.specs(pts)
    assert (kcfd.not_run_table(spec)["status"] == "NOT RUN").all()
    env = aeromant.OpenFOAMEnvironment(env={"PATH": str(tmp_path)})
    name = next(n for n, s in spec.items() if s["kind"] == "back")
    s = spec[name]
    c = kcfd.case(tmp_path / "cfd", {k: v for k, v in s.items() if k != "props"}, x_cg_m=VEH.x_cg, env=env)
    assert c.prepare().ok
    src = (Path(c.workdir) / "system/fvOptions").read_text()
    assert src.count("type            rotorDisk;") == 4
    ds = kcfd.disk_setup(s, VEH.x_cg)
    assert ds["rpm1"] < ds["rpm3"]                                                      # back transition: the lower pair pushes harder (nose up)
    assert np.linalg.norm(ds["disk_axis"]) == pytest.approx(1.0)


def test_cfd_correction_feeds_the_model(VEH):
    import pandas as pd
    import kestrel_cfd as kcfd
    pt = {"alpha_deg": 30.0, "V": 8.0, "T_up": 3.0, "T_lo": 3.5}
    mc = kcfd.model_coefficients(VEH, pt)
    rows = {}
    for da in (0.0, 15.0, 30.0):                                                        # a "CFD" 0.1 above the model in CL everywhere
        q = dict(pt, alpha_deg=pt["alpha_deg"] + da)
        m_ = kcfd.model_coefficients(VEH, q)
        rows[str(da)] = {"status": "solved", "rotor disks": True, "α [deg]": q["alpha_deg"], "V [m/s]": q["V"], "T upper [N]": q["T_up"],
                         "T lower [N]": q["T_lo"], "CL": m_["CL"] + 0.1, "CD": m_["CD"], "Cm (CG, chord)": m_["Cm"]}
    corr = kcfd.CFDAirframe.from_results(pd.DataFrame(rows).T, VEH)
    assert corr.correction(math.radians(30.0), mc["mu"])[0] == pytest.approx(0.1, abs=1e-6)
    veh2 = kt.make_vehicle(airframe=corr)
    a = math.radians(30.0)
    f0 = kt.airframe_forces(VEH, 8 * math.cos(a), 8 * math.sin(a), 3.0, 3.5)
    f1 = kt.airframe_forces(veh2, 8 * math.cos(a), 8 * math.sin(a), 3.0, 3.5)
    lift = lambda f: -f["Fx"] * math.sin(a) + f["Fz"] * math.cos(a)
    assert lift(f1) - lift(f0) == pytest.approx(0.1 * 0.5 * kf.RHO * 64 * VEH.L["S_ref"], rel=1e-6)


# ------------------------------------------------------------------------------------------------ structure and ground
def test_truss_stiffens_the_wing_pair(CI):
    tab = kst.compare_bracing(mass_kg=CI["mass_kg"])
    braced, bare = tab.loc["posts + X diagonals"], tab.loc["no truss (two cantilevers)"]
    assert braced["tip deflection, upper [mm]"] < 0.2 * bare["tip deflection, upper [mm]"]
    assert braced["spar stress [MPa]"] < bare["spar stress [MPa]"] and bare["spar reserve"] > 2.0
    uneven = kst.frame(mass_kg=CI["mass_kg"], bracing="posts", upper_share=0.75)
    assert abs(uneven["bars"][0]["force [N]"]) > 1.0                                    # the post works when the wings load unevenly


def test_cantilever_against_beam_theory():
    """No truss, no inertia, a uniform load: the frame's spar against w = q l⁴ / 8 E I and M = q l² / 2 at the root."""
    L = kestrel.layout()
    _, I_s = kst._section(*kst.TUBES["spar"])
    E = kst.CARBON["E"]
    r = kst.frame(mass_kg=1.0, n=1.0, bracing="none", distribution="uniform", inertia=False)
    l = L["span"] / 2
    q = 1.0 * kf.G / 2 / 2 / l                                                           # per wing, per metre of the half span
    assert r["tip deflection, upper [mm]"] / 1e3 == pytest.approx(q * l ** 4 / (8 * E * I_s), rel=0.03)
    assert r["spar peak moment [N·m]"] == pytest.approx(q * l ** 2 / 2, rel=0.03)


def test_ground_stance(CI):
    g = kst.ground_stance(mass_kg=CI["mass_kg"], x_cg=CI["x_cg"], z_cg=CI["z_cg"])
    assert g["least tip-over angle [deg]"] > 30.0
    assert g["wind that tips it, broadside [m/s]"] < g["wind that tips it, edge-on [m/s]"]
    assert g["foot rod margin, two feet first"] > 2.0
