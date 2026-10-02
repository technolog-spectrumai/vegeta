"""The Sikarian Lobster's tail thruster in water (notebook 24 §5): two Aeromant cases (OpenFOAM), fast by default.

* ``prop_case`` — the bare 76 mm propeller at bollard (``rotor_mrf_static``: a rotating frame, steady k-ω SST, a small
  residual inflow): thrust and torque against Boreas blade element theory (``bemt_bollard``), and the particle
  movie (``aeromant.movie.make_movie``). The shroud is not in this case: an MRF zone cannot keep a duct stationary
  around a turning propeller; its gain at bollard is momentum theory (``duct_gain``), the ``DUCT_GAIN`` the MuJoCo
  thruster uses;
* ``tail_case`` — the whole body swimming (the ``cfd`` part: shell, tail segments, shroud, posed) with the tail
  **yawed** to deflect the jet, the propeller as a rotor disk (``hull_rotor_disk``: blade-element source terms from
  the Boreas blade and section, along the deflected shroud axis). What it adds to the simple picture — the thrust
  vector at the hub — is the jet scrubbing the shell and the shroud's own lift: the forces on the body
  (``tail_forces``), and a particle movie (``aeromant.viz.animate_particles``).

  **A calibrated disk.** At the swim the propeller is close to bollard (the jet ~5× the stream): the rotor disk with
  the local cell velocity (induction included, the rotorDisk default) diverges within a few iterations on any mesh
  tried. The case uses the template's ``fixed_inflow=1`` (the blade elements see the free stream: a steady source)
  and, since that leaves out the induction and over-predicts the thrust, runs the disk at the rpm whose
  no-induction thrust (``disk_thrust_fixed``, the same blade-element sum the solver does) equals the BEMT thrust
  of the open propeller at the swim (``disk_calibration``). The disk delivers the right thrust; its radial load
  shape is the no-induction one.

The CFD frame of ``tail_case``: the free stream along +x, so the body faces −x (the robot turned 180° about z), the
shell's centre at the origin. ``to_robot`` turns a CFD vector back to the robot frame (x forward, y left, z up).
``QUALITY`` holds the presets: ``fast`` (the default: coarse, a few minutes on 4 cores, forces within ~20-30 %) and
``fine``.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

import lobster_robot as lr
from lobster import SikarianLobster

__all__ = ["WATER", "QUALITY", "SWIM", "prop_stl", "prop_case", "bemt_bollard", "duct_gain", "body_stl", "disk_tables",
           "disk_geometry", "disk_thrust_fixed", "disk_calibration", "tail_case", "tail_forces", "disk_log_forces", "to_robot", "jet_side_force"]

#: Fresh water at 15 °C (the MuJoCo scene's): density [kg/m³], kinematic viscosity [m²/s].
WATER = {"density": lr.WATER["density"], "kinematic_viscosity": lr.WATER["viscosity"] / lr.WATER["density"]}
#: The swim the tail case is run at (the MuJoCo swim reaches ~0.8 m/s at 2600 rpm; 0.6 m/s is the cruise).
SWIM = {"speed_m_s": 0.6, "rpm": 2600.0}

#: Mesh and solver presets. ``fast``: background cells ~D/5 (prop) or L/3 (body), two refinement levels fewer, a
#: smaller domain, 250-300 iterations — minutes on 4 cores; ``fine``: the notebooks 13/14 settings, ~20-40 min.
QUALITY = {
    "fast": {
        "prop": dict(cells_per_diameter=5.0, surface_level=3, near_level=2, rotor_level=2, wake_level=1,
                     upstream=2.0, downstream=4.0, lateral=2.0, iterations=250, residual_target=1e-3),
        "tail": dict(cells_per_length=3.0, surface_level=4, near_level=2, wake_level=1, disk_level=4,
                     disk_thickness=0.15, upstream=2.0, downstream=4.0, lateral=2.0, wake_length=2.0,
                     iterations=300, residual_target=1e-3),
    },
    "fine": {
        "prop": dict(cells_per_diameter=6.0, surface_level=4, near_level=3, rotor_level=3, wake_level=2,
                     iterations=500, residual_target=1e-4),
        "tail": dict(cells_per_length=4.0, surface_level=5, near_level=3, wake_level=2, disk_level=5,
                     disk_thickness=0.10, iterations=700, residual_target=1e-4),
    },
}


def _params(p=None) -> dict:
    return lr.design_params() if p is None else dict(p)


# ----------------------------------------------------------------------------------------------- the propeller
def prop_stl(outdir, p=None) -> Path:
    """The propeller (axis +x, hub at the origin), as an STL in mm."""
    from vegeta import dedalus

    p = _params(p)
    g = SikarianLobster().generate(**dict(p, part="propeller"))
    geo = dedalus.Geometry.from_cadquery(g.shape, name="lobster_prop")
    return geo.export_stl(Path(outdir) / "lobster_prop_axis_x.stl", tolerance=0.02)


def prop_case(stl, workdir, p=None, *, rpm: float | None = None, quality: str = "fast", environment=None, **overrides):
    """``rotor_mrf_static`` on the bare propeller at ``rpm`` (default the thruster's maximum)."""
    from vegeta import aeromant

    p = _params(p)
    params = dict(rpm=float(rpm or lr.THRUSTER["rpm_max"]), diameter=p["prop_diameter"] / 1000.0, rotation=1,
                  kinematic_viscosity=WATER["kinematic_viscosity"], density=WATER["density"])
    params.update(QUALITY[quality]["prop"])
    params.update(overrides)
    return aeromant.CFDCase("rotor_mrf_static", stl, params, workdir=workdir, geometry_units="mm", environment=environment)


def bemt_bollard(rpm: float | None = None, p=None) -> dict:
    """Boreas blade element theory at bollard (open propeller, in water): thrust [N], torque [N·m], power [W]."""
    from vegeta import boreas

    p = _params(p)
    rpm = float(rpm or lr.THRUSTER["rpm_max"])
    op = boreas.solve(lr._prop(p), lr.THRUSTER["section"], rpm, 0.0, WATER["density"], speed_of_sound=1500.0)
    return {"rpm": rpm, "thrust_N": op.thrust, "torque_Nm": op.torque, "power_W": op.torque * rpm * 2 * math.pi / 60}


def duct_gain(expansion: float = 1.2) -> float:
    """Thrust of a ducted propeller over the open one at the same shaft power and disc area, bollard, momentum theory:
    open T³ = 2ρA P², ducted (the jet leaves the duct at area σA, σ the exit-to-disc ratio) T³ = 4σρA P² ⇒
    T_duct / T_open = (2σ)^(1/3) — 1.34 at σ = 1.2. With the duct's losses (a short, thick-lipped duct, the strut) the
    MuJoCo thruster takes 1.2 (``lobster_robot.DUCT_GAIN``)."""
    return (2.0 * expansion) ** (1.0 / 3.0)


# ----------------------------------------------------------------------------------------------- the vectored jet
def _shell_centre(p) -> np.ndarray:
    return np.array([0.0, 0.0, p["shell_bottom"] + p["shell_height"] / 2])


def to_robot(v) -> np.ndarray:
    """A CFD-frame vector (the body facing −x) in the robot frame (180° about z: x, y change sign)."""
    v = np.asarray(v, dtype=float)
    return np.array([-v[0], -v[1], v[2]])


def _tail_pose(deflection_deg: float, pitch_deg: float = 0.0) -> dict:
    """The jet deflected sideways by ``deflection_deg`` (yaw, split over the two tail joints)."""
    return {"tail_yaw_deg": deflection_deg / 2.0, "tail_pitch_deg": pitch_deg / 2.0}


def body_stl(outdir, deflection_deg: float = 0.0, p=None, pitch_deg: float = 0.0) -> Path:
    """The ``cfd`` part with the tail posed, in the CFD frame (mm): turned 180° about z, the shell's centre at the
    origin."""
    from vegeta import dedalus

    p = dict(_params(p), **_tail_pose(deflection_deg, pitch_deg))
    g = SikarianLobster().generate(**dict(p, part="cfd"))
    c = _shell_centre(p)
    shape = g.shape.translate(tuple(-c)).rotate((0, 0, 0), (0, 0, 1), 180)
    geo = dedalus.Geometry.from_cadquery(shape, name=f"lobster_cfd_{deflection_deg:g}deg")
    return geo.export_stl(Path(outdir) / f"lobster_cfd_tail_{deflection_deg:g}deg.stl", tolerance=0.2)


def disk_geometry(deflection_deg: float = 0.0, p=None, pitch_deg: float = 0.0) -> tuple:
    """(disk centre [m], thrust axis) in the CFD frame: the propeller plane in the shroud, the force on the robot."""
    p = dict(_params(p), **_tail_pose(deflection_deg, pitch_deg))
    _, (hub, R) = SikarianLobster.tail_frames(p)
    prop = hub + R @ np.array([-p["shroud_length"] * 0.15, 0.0, 0.0])          # as the CAD places it
    c = (prop - _shell_centre(p)) / 1000.0
    axis = R @ np.array([1.0, 0.0, 0.0])
    return to_robot(c), to_robot(axis)


def disk_tables(p=None) -> tuple:
    """(blade rows (r [m], twist [deg], chord [m]), polar rows (α [deg], Cd, Cl)) for the rotor disk, from the
    Boreas propeller and section the MuJoCo thruster uses."""
    p = _params(p)
    prop, sec = lr._prop(p), lr.THRUSTER["section"]
    blade = [[float(r), float(b), float(c)] for r, b, c in zip(prop.r, prop.beta_deg, prop.chord)]
    alphas = np.unique(np.r_[np.arange(-180, -30, 10), np.arange(-30, 31, 1), np.arange(40, 181, 10)]).astype(float)
    cl, cd = sec.coefficients(np.radians(alphas))
    polar = [[float(a), float(d), float(l)] for a, l, d in zip(alphas, cl, cd)]
    return blade, polar


def disk_thrust_fixed(rpm: float, v_axial: float, p=None, tip_effect: float = 0.97) -> float:
    """Thrust [N] of the rotor disk with ``fixed_inflow`` (no induction): the blade elements at ω r and the axial
    inflow ``v_axial``, the polar of ``disk_tables``, no lift beyond ``tip_effect`` R — what the solver sums."""
    blade, polar = (np.asarray(t, dtype=float) for t in disk_tables(p))
    p = _params(p)
    w, R = rpm * 2 * math.pi / 60, blade[-1, 0]
    r = np.linspace(blade[0, 0], R, 400)
    c, th = np.interp(r, blade[:, 0], blade[:, 2]), np.interp(r, blade[:, 0], blade[:, 1])
    vt = w * r
    phi = np.arctan2(v_axial, vt)
    a = th - np.degrees(phi)
    cd, cl = np.interp(a, polar[:, 0], polar[:, 1]), np.interp(a, polar[:, 0], polar[:, 2])
    cl = np.where(r / R > tip_effect, 0.0, cl)
    q = 0.5 * WATER["density"] * (v_axial ** 2 + vt ** 2) * c
    return float(int(p["prop_blades"]) * np.trapezoid(q * (cl * np.cos(phi) - cd * np.sin(phi)), r))


def disk_calibration(deflection_deg: float = 0.0, p=None, *, speed: float | None = None, rpm: float | None = None,
                     pitch_deg: float = 0.0) -> dict:
    """The disk rpm whose no-induction thrust equals the BEMT thrust of the open propeller at ``rpm`` and the axial
    inflow (``speed`` along the deflected axis)."""
    p = _params(p)
    speed, rpm = float(speed or SWIM["speed_m_s"]), float(rpm or SWIM["rpm"])
    _, axis = disk_geometry(deflection_deg, p, pitch_deg)
    v_ax = speed * abs(float(axis[0]))
    target = lr.thrust(lr.thruster_table(p), rpm, v_ax)[0] / lr.DUCT_GAIN          # open propeller (the disk has no duct)
    lo, hi = 0.2 * rpm, 1.5 * rpm
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if disk_thrust_fixed(mid, v_ax, p) < target:
            lo = mid
        else:
            hi = mid
    return {"rpm": rpm, "axial_inflow_m_s": v_ax, "target_thrust_N": target, "disk_rpm": 0.5 * (lo + hi),
            "uncalibrated_thrust_N": disk_thrust_fixed(rpm, v_ax, p)}


def tail_case(stl, workdir, deflection_deg: float = 0.0, p=None, *, speed: float | None = None, rpm: float | None = None,
              quality: str = "fast", environment=None, pitch_deg: float = 0.0, **overrides):
    """``hull_rotor_disk``: the body (``body_stl`` at the same deflection) swimming at ``speed`` with the propeller a
    rotor disk along the deflected shroud axis, calibrated (``disk_calibration``) to the BEMT thrust at ``rpm``.
    Forces on the body about the shell's centre."""
    from vegeta import aeromant

    p = _params(p)
    centre, axis = disk_geometry(deflection_deg, p, pitch_deg)
    blade, polar = disk_tables(p)
    cal = disk_calibration(deflection_deg, p, speed=speed, rpm=rpm, pitch_deg=pitch_deg)
    params = dict(velocity=float(speed or SWIM["speed_m_s"]), kinematic_viscosity=WATER["kinematic_viscosity"],
                  density=WATER["density"], reference_area=p["shell_width"] * p["shell_height"] * 1e-6,
                  reference_length=p["shell_length"] / 1000.0, center_of_rotation=(0.0, 0.0, 0.0),
                  disk1_center=centre.tolist(), disk_axis=axis.tolist(), diameter=p["prop_diameter"] / 1000.0,
                  rpm=cal["disk_rpm"], blades=int(p["prop_blades"]), blade=blade, polar=polar, rotation1=1,
                  fixed_inflow=1.0)
    params.update(QUALITY[quality]["tail"])
    params.update(overrides)
    return aeromant.CFDCase("hull_rotor_disk", stl, params, workdir=workdir, geometry_units="mm", environment=environment)


def disk_log_forces(case) -> dict:
    """What the rotorDisk source prints in the solver log at the last iteration (``Effective drag`` / ``lift``, the
    disk's blade forces summed: lift ≈ the thrust), or {} when the log has none."""
    wd = Path(case.workdir if hasattr(case, "workdir") else case)
    log = wd / "log.solver"
    if not log.is_file():
        return {}
    text = log.read_text(errors="replace")
    out = {}
    for key in ("drag", "lift"):
        hits = re.findall(rf"Effective {key}\s*=\s*([-+0-9.eE]+)", text)
        if hits:
            out[f"effective_{key}"] = float(hits[-1])
    return out


def tail_forces(case, average_window: int = 50) -> dict:
    """The forces on the body (shell, tail, shroud — not the disk) from the forceCoeffs history, in N and N·m, in
    the CFD frame and the robot frame: drag along the stream, side force along the CFD +y (lift × drag directions),
    lift along +z; the yaw moment about the shell's centre."""
    from vegeta.aeromant.results import find_coefficient_files, read_coefficients

    wd = Path(case.workdir if hasattr(case, "workdir") else case)
    files = find_coefficient_files(wd)
    if not files:
        return {}
    hist = read_coefficients(files)
    pr = case.parameters
    q = 0.5 * pr["density"] * pr["velocity"] ** 2
    A, L = pr["reference_area"], pr["reference_length"]
    n = max(1, min(average_window, len(hist.data)))

    def mean(c):
        return float(np.mean(hist[c][-n:])) if c in hist else float("nan")
    f_cfd = np.array([mean("Cd"), mean("Cs"), mean("Cl")]) * q * A
    out = {"drag_N": f_cfd[0], "side_N": f_cfd[1], "lift_N": f_cfd[2], "yaw_moment_Nm": mean("CmYaw") * q * A * L,
           "force_robot_N": to_robot(f_cfd).tolist(), "averaging_window": n}
    out.update(disk_log_forces(case))
    return out


def jet_side_force(deflection_deg: float, thrust_N: float) -> dict:
    """The simple picture: the thrust vector turned by the deflection: forward and side components [N]."""
    d = math.radians(deflection_deg)
    return {"forward_N": thrust_N * math.cos(d), "side_N": thrust_N * math.sin(d)}
