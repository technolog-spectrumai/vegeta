"""Transonic carbon-fibre rotors for a 600 km/h (166.7 m/s) electric racer, against notebook 25's subsonic ones.

Two designs, both in carbon fibre (CFRP: unidirectional plies root-to-tip, +-45 skins):

- ``TPROP``: a 2-blade, 254 mm (10 in) high-pitch tractor propeller. At 166.7 m/s the blade tips run at a helical Mach
  number of ~0.95 (transonic): thin low-camber sections (6 % thick, 2 % camber) and a swept (skewed) tip relieve the
  compressibility drag. Sized with Boreas's blade-element solver on a Mach-aware section (``MachAirfoil``:
  Prandtl-Glauert lift, Korn drag rise).
- ``TFAN``: a 75 mm, 14-blade single-stage transonic ducted fan (90 000 rpm, tip 353 m/s, relative tip Mach ~1.15,
  pressure ratio ~1.3) with 11 stators and a converging nozzle — sized by a mean-line (Euler) estimate.

For comparison, notebook 25's subsonic parts: ``OLD_PROP`` (10x6, PA6-GF30) and ``OLD_EDF`` (90 mm, 12 blades).

``python notebooks/designs/transonic.py`` writes the CAD (STEP + STL) and a comparison picture into
``notebooks/designs/data/transonic/`` and prints the sizing. FEA, vibration and CFD: ``scripts/transonic/``.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "transonic"
RHO, A0, T0, P0, CP, GAM, R_AIR = 1.225, 340.3, 288.15, 101325.0, 1005.0, 1.4, 287.05
G = 9.81
V_MAX = 600 / 3.6                                    # 166.67 m/s

# --- the racer the propeller pulls (assumed: a sleek 3.5 kg pylon-type racer)
RACER = dict(mass_kg=3.5, cda_m2=0.0035, wing_area_m2=0.18, aspect_ratio=6.0, oswald=0.8)

# --- carbon fibre: a spanwise-dominated laminate (60 % 0 deg, 40 % +-45) as an equivalent isotropic solid for the FEA
CFRP = dict(name="CFRP, 60 % UD 0 / 40 % +-45 (equivalent isotropic)", E=75000.0, nu=0.30, rho=1.55e-9,
            strength=600.0,          # MPa, first-ply-failure allowable along the span (knocked down from ~1200 UD)
            ilss=50.0,               # MPa, interlaminar shear (root joints, bonded inserts)
            fatigue_1e7=300.0, damping=0.008)

# ------------------------------------------------------------------------------------------------------------ designs
TPROP = dict(diameter_mm=254.0, pitch_mm=530.0, blades=2, chord_root_mm=20.0, chord_max_mm=24.0, chord_tip_mm=11.0,
             thickness=0.06, thickness_root=0.16, camber=0.02, skew_deg=8.0, hub_diameter=26.0, hub_height=14.0, bore=6.0)
TFAN = dict(diameter_mm=75.0, blades=14, hub_diameter_mm=34.0, hub_height_mm=14.0, pitch_mm=115.0,
            chord_root_mm=12.0, chord_max_mm=13.0, chord_tip_mm=11.0, thickness=0.05, thickness_root=0.11, camber=0.03, skew_deg=15.0,
            tip_clearance_mm=0.25, exit_area_ratio=0.72, stator_vanes=11, stator_chord_mm=12.0, stator_thickness=0.07,
            stator_gap_mm=8.0, rpm=90000.0, pressure_ratio=1.30, eta_poly=0.88)
OLD_PROP = dict(diameter_mm=254.0, pitch_mm=152.4, blades=2, chord_root_mm=18.0, chord_max_mm=26.0, chord_tip_mm=10.0,
                thickness=0.10, camber=0.04, skew_deg=0.0, hub_diameter=20.0, hub_height=10.0, bore=5.0)
OLD_EDF = dict(diameter_mm=90.0, blades=12, hub_diameter_mm=40.0, hub_height_mm=14.0, pitch_mm=160.0,
               chord_root_mm=14.0, chord_max_mm=15.0, chord_tip_mm=12.0, thickness=0.08, camber=0.04, skew_deg=0.0,
               tip_clearance_mm=0.8, exit_area_ratio=0.9, stator_vanes=7, stator_chord_mm=16.0, stator_thickness=0.08,
               stator_gap_mm=14.0)


# ------------------------------------------------------------------------------------------------ the transonic section
class MachAirfoil:
    """A thin section's (cl, cd) with compressibility, one Mach number per blade station (Boreas calls
    ``coefficients(alpha)`` with one angle per station): Prandtl-Glauert lift slope (held above M 0.75), stall cl
    trimmed past M 0.6, and Korn's drag divergence M_dd = kappa - t/c - cl/10 with Lock's rise
    cd_w = 20 (M - M_crit)^4, M_crit = M_dd - (0.1/80)^(1/3), capped at 0.06 past M 1."""

    def __init__(self, mach, tc, camber=0.02, cd0=0.010, k=0.03, kappa=0.90):
        self.mach, self.tc = np.asarray(mach, float), np.asarray(tc, float)
        self.cl_alpha, self.alpha0 = 2 * math.pi * 0.92, -math.radians(100 * camber)   # ~ -1 deg per % camber
        self.cd0, self.k, self.kappa = cd0, k, kappa
        self.cl_max = 1.0

    def coefficients(self, alpha):
        a = np.asarray(alpha, float)
        M = self.mach if self.mach.shape == a.shape else np.full_like(a, float(np.mean(self.mach)))
        tc = self.tc if self.tc.shape == a.shape else np.full_like(a, float(np.mean(self.tc)))
        pg = 1 / np.sqrt(1 - np.minimum(M, 0.75) ** 2)
        lin = self.cl_alpha * pg * (a - self.alpha0)
        clm = self.cl_max * np.clip(1 - 0.6 * (M - 0.6), 0.55, 1.0)
        cl = np.clip(lin, -clm, clm)
        cd = self.cd0 + self.k * cl ** 2 + np.where(np.abs(lin) > clm, 0.5 * (np.abs(a - self.alpha0) - clm / (self.cl_alpha * pg)), 0)
        m_dd = self.kappa - tc - np.abs(cl) / 10
        m_cr = m_dd - (0.1 / 80) ** (1 / 3)
        return cl, cd + np.minimum(20 * np.maximum(M - m_cr, 0) ** 4, 0.06)


def boreas_prop(d: dict):
    from vegeta import boreas
    return boreas.Propeller.from_pitch(f"{d['diameter_mm']:.0f} mm x {d['pitch_mm']:.0f} pitch, {d['blades']} blades",
                                       d["diameter_mm"] / 1000, d["pitch_mm"] / 1000, blades=d["blades"],
                                       chord_root_m=d["chord_root_mm"] / 1000, chord_max_m=d["chord_max_mm"] / 1000,
                                       chord_tip_m=d["chord_tip_mm"] / 1000, hub_radius_m=d["hub_diameter"] / 2000, n_stations=20)


N_ST = 30


def solve_prop(d: dict, rpm: float, V: float):
    from vegeta import boreas
    pr = boreas_prop(d)
    r, _, _, _ = pr.stations(N_ST)
    mach = np.hypot(V, rpm * 2 * math.pi / 60 * r) / A0
    tc = np.interp(r / pr.radius, [d["hub_diameter"] / d["diameter_mm"] * 0.9, 1.0], [d.get("thickness_root") or d["thickness"], d["thickness"]])
    return boreas.solve(pr, MachAirfoil(mach, tc, d["camber"]), rpm, V, RHO, n_stations=N_ST, speed_of_sound=A0)


def prop_for_thrust(d: dict, T: float, V: float, rpm_max: float = 40000.0):
    lo, hi = 1000.0, rpm_max
    if solve_prop(d, hi, V).thrust < T:
        raise ValueError(f"{T:.1f} N not reachable at {V:.0f} m/s below {rpm_max:.0f} rpm")
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if solve_prop(d, mid, V).thrust < T else (lo, mid)
    return solve_prop(d, hi, V)


def racer_thrust(V: float) -> float:
    a = RACER
    q = 0.5 * RHO * V * V
    W = a["mass_kg"] * G
    cl = W / (q * a["wing_area_m2"])
    return q * a["cda_m2"] + q * a["wing_area_m2"] * cl ** 2 / (math.pi * a["aspect_ratio"] * a["oswald"])


def prop_sizing(d: dict = TPROP, speeds=(60.0, 100.0, 140.0, V_MAX)) -> list[dict]:
    rows = []
    for V in speeds:
        T = racer_thrust(V)
        op = prop_for_thrust(d, T, V)
        tip = math.hypot(V, op.rpm * 2 * math.pi / 60 * d["diameter_mm"] / 2000) / A0
        rows.append(dict(speed_m_s=round(V, 2), speed_kmh=round(V * 3.6), thrust_N=round(T, 2), rpm=round(op.rpm),
                         shaft_kW=round(op.power / 1000, 2), efficiency=round(op.efficiency, 3), tip_helical_mach=round(tip, 3),
                         torque_Nm=round(op.torque, 3), advance_ratio=round(op.advance_ratio, 3)))
    return rows


def fan_meanline(d: dict = TFAN, V: float = 0.0) -> dict:
    """Mean-line estimate of the transonic fan at flight speed ``V``: mass flow at an axial Mach of 0.5 at the face
    (the intake's ram raises the face's total pressure and density in flight), the work from the pressure ratio at
    ``eta_poly``, Euler's equation at the mean radius, a 3 % stator + duct loss, the nozzle's jet and the net thrust."""
    Rt, Rh = d["diameter_mm"] / 2000, d["hub_diameter_mm"] / 2000
    A = math.pi * (Rt ** 2 - Rh ** 2)
    Tt1 = T0 + V * V / (2 * CP)                                    # ram (intake loss-free)
    Pt1 = P0 * (Tt1 / T0) ** 3.5
    Ma = 0.5
    T_s = Tt1 / (1 + 0.2 * Ma ** 2)
    p_s = Pt1 * (T_s / Tt1) ** 3.5
    ca = Ma * math.sqrt(GAM * R_AIR * T_s)
    mdot = p_s / (R_AIR * T_s) * ca * A
    PR = d["pressure_ratio"]
    dT0 = Tt1 * (PR ** ((GAM - 1) / (GAM * d["eta_poly"])) - 1)
    work = CP * dT0
    om = d["rpm"] * 2 * math.pi / 60
    Um, Ut = om * (Rt + Rh) / 2, om * Rt
    T02, P02 = Tt1 + dT0, Pt1 * PR * 0.97
    npr = P02 / P0
    if npr > 1.893:                                                 # a converging nozzle chokes: sonic exit + pressure thrust
        Te = T02 / 1.2
        Vj = math.sqrt(GAM * R_AIR * Te)
        pe = P02 / 1.893
        Ae = mdot * R_AIR * Te / (pe * Vj)
        F = mdot * (Vj - V) + (pe - P0) * Ae
    else:
        Vj = math.sqrt(2 * CP * T02 * (1 - npr ** (-(GAM - 1) / GAM)))
        F = mdot * (Vj - V)
    Mj = Vj / math.sqrt(GAM * R_AIR * (T02 - Vj ** 2 / (2 * CP)))
    return dict(flight_m_s=round(V, 2), mass_flow_kg_s=round(mdot, 4), axial_m_s=round(ca, 1), tip_speed_m_s=round(Ut, 1),
                tip_relative_mach=round(math.hypot(Ut, ca) / math.sqrt(GAM * R_AIR * T_s), 3),
                mean_swirl_m_s=round(work / Um, 1), work_kJ_kg=round(work / 1000, 2), shaft_kW=round(mdot * work / 1000, 2),
                nozzle_pressure_ratio=round(npr, 3), jet_m_s=round(Vj, 1), jet_mach=round(Mj, 3), net_thrust_N=round(F, 1),
                propulsive_efficiency=round(F * V / (mdot * work), 3) if V > 0 else None,
                note="mean-line estimate (losses lumped); the CFD script is the check")


# ------------------------------------------------------------------------------------------------------------- the CAD
def prop_cad(d: dict, blades: int | None = None):
    from vegeta.dedalus.examples import Propeller
    kw = dict(diameter=d["diameter_mm"], pitch=d["pitch_mm"], blades=blades or d["blades"], chord_root=d["chord_root_mm"],
              chord_max=d["chord_max_mm"], chord_tip=d["chord_tip_mm"], thickness=d["thickness"], camber=d["camber"],
              stations=10, hub_diameter=d["hub_diameter"], hub_height=d["hub_height"], bore=d["bore"])
    if d.get("skew_deg"):
        kw["skew_deg"] = d["skew_deg"]
    if d.get("thickness_root"):
        kw["thickness_root"] = d["thickness_root"]
    return Propeller().generate(**kw)


def _edf_p(d: dict) -> tuple[dict, dict]:
    rotor = dict(diameter=d["diameter_mm"], hub_diameter=d["hub_diameter_mm"], hub_height=d["hub_height_mm"], pitch=d["pitch_mm"],
                 chord_root=d["chord_root_mm"], chord_max=d["chord_max_mm"], chord_tip=d["chord_tip_mm"], thickness=d["thickness"],
                 camber=d["camber"])
    housing = dict(rotor, tip_clearance=d["tip_clearance_mm"], exit_area_ratio=d["exit_area_ratio"], stator_vanes=d["stator_vanes"],
                   axial_gap=2.0, stator_chord=d["stator_chord_mm"], stator_thickness=d["stator_thickness"], stator_gap=d["stator_gap_mm"])
    s = d["diameter_mm"] / 90.0                                   # the housing's other lengths scale with the diameter
    housing.update(lip_length=31.5 * s, duct_length=67.5 * s, lip_radius=6.0 * s, tail_length=36.0 * s,
                   wall_thickness=max(1.2, 2.5 * s))
    return rotor, housing


def edf_cad(d: dict):
    import ducted_fan as df
    rp, hp = _edf_p(d)
    kw = dict(blades=d["blades"], **rp)
    if d.get("skew_deg"):
        kw["skew_deg"] = d["skew_deg"]
    kw["spinner_length"] = 22.0 * d["diameter_mm"] / 90.0
    if d.get("thickness_root"):
        kw["thickness_root"] = d["thickness_root"]
    return df.EDFRotor().generate(**kw), df.EDFHousing().generate(**hp)


PARTS = {"transonic_prop": ("prop", TPROP), "subsonic_prop_nb25": ("prop", OLD_PROP),
         "transonic_fan": ("edf", TFAN), "subsonic_edf_nb25": ("edf", OLD_EDF)}


def export_all(out: Path = OUT) -> dict:
    """STEP + STL of every part (fans: rotor and housing separately), and the meshes for the picture."""
    out.mkdir(parents=True, exist_ok=True)
    files, shapes = {}, {}
    for name, (kind, d) in PARTS.items():
        if kind == "prop":
            g = prop_cad(d)
            files[name] = g.export(out / name, formats=("step", "stl"), stl_tolerance=0.02).artifacts
            shapes[name] = [(g, "#2b2b2b" if "transonic" in name else "#c8a25a")]
        else:
            rot, hou = edf_cad(d)
            files[name + "_rotor"] = rot.export(out / f"{name}_rotor", formats=("step", "stl"), stl_tolerance=0.02).artifacts
            files[name + "_housing"] = hou.export(out / f"{name}_housing", formats=("step", "stl"), stl_tolerance=0.05).artifacts
            shapes[name] = [(rot, "#2b2b2b" if "transonic" in name else "#3b6ea5"), (hou, "#c9c9c9")]
    return {k: {f: str(Path(p).relative_to(out)) for f, p in v.items()} for k, v in files.items()}, shapes


def picture(shapes: dict, path: Path):
    """Three rows: the propellers; the fan rotors seen from ahead; the fans cut through the axis (rotor in place)."""
    import pyvista as pv
    from vegeta.dedalus import viz as dviz
    pv.OFF_SCREEN = True
    T, O = "TRANSONIC carbon", "old subsonic (notebook 25)"
    panels = [("transonic_prop", "prop", f"{T} prop 10 in, 530 mm pitch, 8 deg swept tips, 16 % root -> 6 % tip"),
              ("subsonic_prop_nb25", "prop", f"{O} 10x6 prop, 10 % thick"),
              ("transonic_fan", "rotor", f"{T} fan rotor 75 mm, 14 blades, skewed"),
              ("subsonic_edf_nb25", "rotor", f"{O} EDF rotor 90 mm, 12 blades"),
              ("transonic_fan", "cut", f"{T} fan cut: lip, rotor, 11 stators, nozzle (exit ratio 0.72)"),
              ("subsonic_edf_nb25", "cut", f"{O} EDF cut: 7 stators, nozzle (exit ratio 0.9)")]
    pl = pv.Plotter(shape=(3, 2), window_size=(1800, 1900), off_screen=True)
    for k, (name, mode, title) in enumerate(panels):
        pl.subplot(k // 2, k % 2)
        items = shapes[name]
        if mode == "prop":
            pl.add_mesh(dviz.to_pyvista(items[0][0], 0.05), color=items[0][1], smooth_shading=True)
            pl.view_isometric()
        elif mode == "rotor":
            pl.add_mesh(dviz.to_pyvista(items[0][0], 0.03), color=items[0][1], smooth_shading=True)
            pl.view_vector((0.35, -0.35, -1.0), viewup=(0, 1, 0))           # from ahead and a little aside
        else:
            rot = dviz.to_pyvista(items[0][0], 0.03).rotate_y(90, inplace=False)
            hou = dviz.to_pyvista(items[1][0], 0.05).clip(normal="z", origin=(0, 0, 0))
            pl.add_mesh(rot, color=items[0][1], smooth_shading=True)
            pl.add_mesh(hou, color="#bdbdbd", smooth_shading=True)
            pl.view_vector((0.15, 0.25, 1.0), viewup=(0, 1, 0))
        pl.add_text(title, font_size=10)
        pl.reset_camera()
    pl.screenshot(str(path))
    pl.close()
    return path


def main():
    sys.path.insert(0, str(HERE))
    rows = prop_sizing()
    fan = {"static": fan_meanline(TFAN, 0.0), "600_kmh": fan_meanline(TFAN, V_MAX)}
    print("transonic prop on the racer:")
    for r in rows:
        print("  ", r)
    print("transonic fan (mean line):", fan)
    files, shapes = export_all()
    pic = picture(shapes, OUT / "comparison.png")
    (OUT / "design.json").write_text(json.dumps({"racer": RACER, "material": CFRP, "transonic_prop": TPROP, "transonic_fan": TFAN,
                                                 "old_prop": OLD_PROP, "old_edf": OLD_EDF, "prop_sizing": rows, "fan_meanline": fan,
                                                 "files": files}, indent=1))
    print("wrote", pic, "and", OUT / "design.json")


if __name__ == "__main__":
    main()
