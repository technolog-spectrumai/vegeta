"""Deterministic reference cases through the engineering tools, compared with golden.json (tight tolerances: these
are pure computations; a change means the numbers changed)."""
import pytest


def test_dedalus_component_measurements(golden):
    from components.quad_frame import QuadFrame
    m = QuadFrame().generate(wheelbase=180.0).measure()
    golden("dedalus.quad_frame_180", {k: m[k] for k in ("volume", "surface_area", "center_of_mass", "bbox_min",
                                                          "bbox_max", "n_faces")}, rel=1e-7, abs=1e-6)


def test_boreas_static_point(golden):
    from vegeta import boreas
    d, p = boreas.inches(10, 4.7)
    prop = boreas.Propeller.from_pitch("APC 10x4.7 SF", d, p, chord_root_m=0.018, chord_max_m=0.026,
                                       chord_tip_m=0.008, mass_kg=0.012, rotor_mass_kg=0.040)
    op = boreas.solve(prop, boreas.Airfoil(), 6000.0)
    assert op.converged
    golden("boreas.apc10x47_6000rpm", {"ct": op.ct, "cp": op.cp, "fom": op.figure_of_merit, "tip_mach": op.tip_mach},
           rel=1e-7)


@pytest.mark.parametrize("designation", ["2412", "23012"])
def test_zephyr_thin_airfoil(designation, golden):
    from vegeta.zephyr import naca, thin_airfoil
    ta = thin_airfoil(naca(designation))
    golden(f"zephyr.thin_airfoil_{designation}", {k: ta[k] for k in ("alpha0_deg", "cm_c4", "cl_alpha_per_rad")})


def test_chronos_rainflow_and_life(golden):
    from vegeta import chronos
    cycles = chronos.rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    curve = chronos.SNCurve("t", sigma_f=100.0, b=-0.1, ultimate=200.0)
    spec = chronos.LoadSpectrum("x", 60.0, [chronos.Block("a", 0.0, 10.0, 100.0, "one")], ["a"])
    golden("chronos.rainflow_astm", [list(c) if not isinstance(c, (int, float)) else c for c in cycles])
    golden("chronos.hotspot_damage", chronos.hotspot_damage(spec, {"a": 5.0}, curve))


def test_talos_stress_post_processing(golden):
    """CalculiX itself is not run here (test.sh's working run checks a cantilever against beam theory); the
    post-processing that every Talos result goes through is."""
    import numpy as np
    from vegeta.talos import von_mises
    s = np.array([[100.0, 0, 0, 0, 0, 0], [0, 0, 0, 50.0, 0, 0], [120.0, -30.0, 10.0, 15.0, -5.0, 20.0]])
    vm = von_mises(s)
    assert vm[0] == 100.0 and abs(vm[1] - 50.0 * 3 ** 0.5) < 1e-12
    golden("talos.von_mises", vm)
