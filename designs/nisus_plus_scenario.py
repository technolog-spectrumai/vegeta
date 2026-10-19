"""NISUS+ missions in MuJoCo over the mountains: the scenarios, the runs, the judge, the tables and the movies
(notebook 33, ``scenarios/nisus_plus_mission.py``).

A ``Scenario`` names the conditions (the synoptic wind, the turbulence, a gust, the ISA offset, the pack's temperature,
a storm warning, the Jetson's failure) and the mission: the climb over the meadow to ``climb_alt``, the survey legs on
the north face at ``survey_agl`` above the slope, the return, the crow descent and the landing. ``make_lab`` builds the
aircraft (``nisus_plus_robot``) on the ``Massif`` height field with the ``NisusPlusAero`` hook; ``run`` flies it with the
``NisusPlusController``, the launch a scheduled impulse (a light bungee). ``outcome`` judges from the physics: a landing on
the meadow (within ``land_radius`` of home) at a gentle sink with the reserve kept, no terrain strike — or the first
failure. ``energy_table`` by phase (Wh, kWh, % of nominal; the regeneration in its own row); ``render_movie`` the chase
view over the terrain, the synthetic onboard camera, the map, the overlays (altitude, height above the ground, the
density altitude, EAS and TAS, crow and brake, the regeneration, the pack).

    import nisus_plus_scenario as fsc
    scn = fsc.Scenario()                              # calm
    lab = fsc.make_lab(scn); ep = fsc.run(lab, scn)
    fsc.outcome(ep), fsc.energy_table(ep), fsc.render_movie(ep, scn, "nisus_plus_calm.mp4")
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from vegeta import chiron as ch

import nisus_plus
import nisus_plus_controller as fc_
import nisus_plus_flight as ff
import nisus_plus_robot as fr
import nisus_plus_systems as fs

__all__ = ["Scenario", "CONDITIONS", "standard_scenarios", "make_lab", "controller", "run", "outcome", "timeseries", "phase_table", "energy_table",
           "summary", "compare", "render_movie"]


@dataclass
class Scenario:
    name: str = "calm"
    craft: str = "Nisus+ Zero"                # the aircraft's name in labels; ``slug_prefix`` in file names
    slug_prefix: str = "nisus_plus"
    battery_key: str = fs.DEFAULT_PACK
    wind: tuple = (0.0, 0.0, 0.0)            # the synoptic wind [m/s] (x east, y north)
    sigma: float = 0.0
    gust: tuple | None = None
    dT: float = 0.0                           # ISA offset [K]
    pack_T_C: float = 15.0
    weather_at: float | None = None
    jetson_failure: tuple | None = None
    climb_alt: float = 3650.0
    survey_x: tuple = (1500.0, 3000.0)
    survey_y: tuple = (2800.0, 2400.0, 2000.0)          # from the top down: every change of leg goes downhill
    survey_agl: float = 150.0
    survey_repeats: int = 1
    land_radius: float = 150.0
    duration: float = 2400.0
    seed: int = 0
    plan: fs.MissionPlan = field(default_factory=fs.MissionPlan)

    def __post_init__(self):
        self.plan = fs.MissionPlan(**{**asdict(self.plan), "dT": self.dT, "pack_T_C": self.pack_T_C, "wind_m_s": float(np.linalg.norm(np.asarray(self.wind[:2])))})

    @property
    def label(self):
        return f"{self.craft} — {self.name}"

    @property
    def slug(self):
        return self.slug_prefix + "_" + re.sub(r"[^A-Za-z0-9]+", "_", self.name.replace("m/s", "ms").replace("°", "")).strip("_")

    @property
    def launch_heading_deg(self):
        """The valley runs east-west: launch and land along it, into the wind's valley component (east in calm air)."""
        return 180.0 if self.wind[0] > 0.5 else 0.0

    def survey_waypoints(self, massif: fr.Massif):
        return fc_.slope_survey(massif, self.survey_x[0], self.survey_x[1], self.survey_y, self.survey_agl, ff.V_CRUISE_EAS)


CONDITIONS = {
    "calm": dict(),
    "ridge lift (south wind 8 m/s)": dict(wind=(0.0, 8.0, 0.0), sigma=0.8),
    "lee downdraft (north wind 10 m/s)": dict(wind=(0.0, -10.0, 0.0), sigma=1.0),
    "hot day (ISA +20 °C)": dict(dT=20.0, pack_T_C=30.0),
    "storm: weather escape": dict(weather_at=700.0, survey_repeats=4, wind=(-4.0, 0.0, 0.0), sigma=1.0),
    "Jetson failure": dict(jetson_failure=(650.0, 690.0)),
}


def standard_scenarios() -> list:
    return [Scenario(name, **kw) for name, kw in CONDITIONS.items()]


# ------------------------------------------------------------------------------------------------- the lab and the run
@dataclass
class Sim:
    """What ``make_lab`` builds the aircraft from — NISUS+'s pieces by default; FALCO passes its own (``falco_scenario.SIM``):
    the resolver, the robot builder, the drive, the lattice and derivative functions, the Aero hook class, the controller
    class, the CG arithmetic, the electronics and the pack, the aircraft's name and the MuJoCo body."""
    resolve: object = nisus_plus.resolve
    robot: object = fr.nisus_plus_robot
    drive: object = fs.drive
    aero: object = ff.aero
    derivatives: object = ff.derivatives
    cg_inertia: object = fs.cg_inertia
    aero_cls: type = fr.NisusPlusAero
    controller_cls: type = fc_.NisusPlusController
    phase_power: object = fs.phase_power
    pack: object = fs.pack
    jetson_w: float = fs.JETSON_INSTALLATION_W
    craft: str = "Nisus+ Zero"
    body: str = fr.BODY
    massif_cls: type = fr.Massif
    wind_cls: type = fr.MountainWind


SIM = Sim()


def make_lab(scn: Scenario, *, p=None, massif: fr.Massif | None = None, drive: fs.NisusPlusDrive | None = None, sim: Sim | None = None, **kwargs):
    """The aircraft of the scenario on the mountains in ChironLab with the ``NisusPlusAero`` hook (``lab.aero``); ``sim``:
    another aircraft's pieces (``Sim``)."""
    sim = sim or SIM
    p = sim.resolve(p)
    massif = massif or sim.massif_cls()
    robot = sim.robot(scn.battery_key, p)
    drive = drive or sim.drive()
    ci = sim.cg_inertia(robot.mass_table, p)
    a = sim.aero(p)
    deriv = sim.derivatives(p, x_cg_m=ci["x_cg_m"], z_cg_m=ci["z_cg_m"], a=a, h_m=2500.0)
    coeff = ff.coefficients(deriv)
    opts = fr.lab_options(massif, **kwargs)
    lab = ch.ChironLab(robot, massif.as_chiron(), **opts)
    wind = sim.wind_cls(steady=tuple(scn.wind), sigma=scn.sigma, gust=scn.gust, seed=scn.seed, massif=massif)
    el = sim.phase_power()["electronics battery-side [W]"].to_dict()
    pk = sim.pack(scn.battery_key)
    lab.aero = sim.aero_cls(robot, coeff, drive, pk, massif, wind=wind, electronics_w=el["prelaunch"], derating=scn.plan.derating, dT=scn.dT,
                            pack_T_C=scn.pack_T_C)
    lab.add_hook(lab.aero)
    lab.coeff, lab.deriv, lab.drive, lab.ci, lab.el, lab.massif, lab.a = coeff, deriv, drive, ci, el, massif, a
    lab.airframe = {"mass_kg": ci["mass_kg"], "cd0": coeff["CD0"], "AR": a["AR"], "oswald": 0.9 * a["e"], "S": a["S_ref"]}
    lab.scenario, lab.sim, lab.body = scn, sim, sim.body
    return lab


def controller(lab, scn: Scenario, **kw):
    sim = getattr(lab, "sim", SIM)
    em = fc_.EnergyManager(lab.aero, lab.drive, lab.airframe, scn.plan, lab.el, home_alt=float(lab.massif.height(0.0, 0.0)))
    wps = scn.survey_waypoints(lab.massif)
    return sim.controller_cls(lab.aero, lab.coeff, lab.airframe, scn.plan, lab.massif, launch_heading_deg=scn.launch_heading_deg, climb_alt=scn.climb_alt,
                              mode="auto", mission=fc_.AutoMission(wps, repeats=scn.survey_repeats), survey_wps=wps, energy=em, electronics_phase_w=lab.el,
                              jetson_w=sim.jetson_w, jetson_failure=scn.jetson_failure, weather_at=scn.weather_at,
                              wind_estimate=tuple(scn.wind[:2]), ground_s=scn.plan.ground_s, name=f"{sim.craft} autopilot ({scn.name})", **kw)


class TimeBar:
    """A tqdm bar over the simulation's time as a ChironLab hook (``run(..., progress=True)``): its postfix the
    controller's phase; closed with the outcome."""

    def __init__(self, total: float, desc: str, fc=None, every: int = 25):
        self.total, self.desc, self.fc, self.every = float(total), desc, fc, every
        self.bar, self.n = None, 0

    def reset(self, lab):
        from tqdm.auto import tqdm
        if self.bar is not None:
            self.bar.close()
        self.n = 0
        self.bar = tqdm(total=int(round(self.total)), desc=self.desc, unit="s", bar_format="{desc}: {n_fmt}/{total_fmt} s |{bar}| {elapsed} {postfix}")

    def __call__(self, lab):
        self.n += 1
        if self.bar is not None and self.n % self.every == 0:
            self.bar.n = int(min(lab.time, self.total))
            self.bar.set_postfix_str(getattr(self.fc, "phase", ""))
            self.bar.refresh()

    def close(self, note: str = ""):
        if self.bar is not None:
            self.bar.set_postfix_str(note)
            self.bar.refresh()
            self.bar.close()
            self.bar = None


def run(lab, scn: Scenario, *, duration: float | None = None, launch_speed: float = 17.0, launch_pitch_deg: float = 8.0, fc=None,
        progress: bool = False) -> ch.Episode:
    """The mission: the launch from the meadow — a light bungee (an impulse of ``m x launch_speed`` over 0.3 s along the
    launch heading; 17 m/s: a hand throw of 11-12 m/s leaves the 5.2 kg aircraft at its stall at 1200 m and it sinks
    into the meadow: the simulation's finding, ``nisus_plus_flight.launch_check`` needs the take-off flap for it) — then the
    controller's phases until it has landed and stopped (or ``duration``). ``progress``: a tqdm bar over the simulation's
    time (``TimeBar``)."""
    fc = fc or controller(lab, scn)
    tb = None
    if progress:
        tb = TimeBar(duration or scn.duration, scn.label, fc)
        lab.add_hook(tb)
    body = getattr(lab, "body", fr.BODY)
    m = float(lab.model.body_subtreemass[lab._body_id(body)])
    a, e = math.radians(scn.launch_heading_deg), math.radians(launch_pitch_deg)
    lab._disturbances = [d for d in lab._disturbances if d.body != body]
    lab.add_disturbance(ch.Disturbance(body, t_start=fc.prelaunch_sim_s, duration=0.30, impulse=m * launch_speed,
                                       direction=(math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))))
    z0 = float(lab.massif.height(0.0, 0.0)) + lab.nominal_base_height + 1.2           # the bungee's release height
    th = -math.radians(launch_pitch_deg + 2.0)                         # held nose-up in the thrower's hand (about the body's y axis)
    qy, qz = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0), (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
    quat = (qz[0] * qy[0] - qz[3] * qy[3], qz[0] * qy[1] - qz[3] * qy[2], qz[0] * qy[2] + qz[3] * qy[1], qz[3] * qy[0] + qz[0] * qy[3])
    ep = lab.run(fc, duration=duration or scn.duration, rules=None, settle=0.0, seed=scn.seed, base_pos=(0.0, 0.0, z0), base_quat=quat,
                 info={"controller": fc.name, "treatment": f"Zero:{scn.name}"}, stop_when=lambda lab_: fc.finished and lab_.time > fc.t_phase + 12.0)
    _finish(ep, lab, scn, fc, m, launch_speed)
    if tb is not None:
        tb.close(f"{ep.outcome['reason']} at {ep.outcome['t_end']:.0f} s")
        lab._hooks = [h for h in lab._hooks if h is not tb]
    return ep


def _finish(ep, lab, scn, fc, m, launch_speed):
    aero = lab.aero
    ep.log["aero"] = np.asarray(aero.history, float).reshape(-1, len(type(aero).COLUMNS))
    ep.log["aero_columns"] = list(type(aero).COLUMNS)
    ep.log["body"] = getattr(lab, "body", fr.BODY)
    ep.log["mission"] = [list(x) for x in fc.log]
    ep.log["events"] = sorted(list(ep.log.get("events", [])) + [list(x) for x in fc.events], key=lambda r: r[0])
    ep.log["touchdown"] = fc.touchdown
    ep.log["return_reason"] = fc.return_reason
    ep.log["scenario"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(scn).items() if k != "plan"}
    ep.log["plan"] = asdict(scn.plan)
    ep.log["home_alt"] = fc.home_alt
    ep.log["energy"] = {"E_nominal_wh": aero.E_nominal_Wh, "E_available_wh": aero.E_available_Wh, "E_used_wh": aero.E_used_Wh,
                        "E_remaining_wh": aero.E_remaining_Wh, "reserve_wh": fc.energy.E_reserve if fc.energy else None, "regen_wh": aero.regen_Wh,
                        "launch_impulse_Ns": m * launch_speed, "launch_speed": launch_speed}
    ep.log["mass_kg"] = m
    ep.log["mission_finished"] = fc.finished
    ep.outcome = outcome(ep, scn.land_radius)
    ep.controller = fc


# ------------------------------------------------------------------------------------------------- the judge and the tables
def timeseries(ep) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(ep.log["aero"]), columns=list(ep.log["aero_columns"]))


def _phase_at(ep, t):
    starts = [(tt, name) for tt, name, note in ep.log["mission"] if note == "start"]
    return next((name for tt, name in reversed(starts) if tt <= t + 1e-9), "prelaunch")


def outcome(ep, land_radius: float = 150.0) -> dict:
    """Success: landed on the meadow within ``land_radius`` of home, touchdown sink ≤ 2.5 m/s, no terrain strike in flight
    (pod, wing or tail on the ground before the touchdown), the reserve kept. Otherwise the first failure found."""
    log = ep.log
    ts = timeseries(ep)
    t = np.asarray(log["t"])
    bodies = list(log["bodies"])
    belly = np.asarray(log["belly_contact"])[:, bodies.index(log.get("body", fr.BODY))] if "belly_contact" in log else np.zeros(len(t), bool)
    E = log["energy"]
    reserve_ok = E["reserve_wh"] is None or E["E_remaining_wh"] >= E["reserve_wh"] - 1e-6
    phases = [name for _, name, note in log["mission"] if note == "start"]
    td = log.get("touchdown")
    t_launch = next((tt for tt, name, note in log["mission"] if name == "launch" and note == "start"), 0.0)
    t_td = td if td is not None else t[-1]
    flying = (t > t_launch + 1.0) & (t < t_td - 0.2)
    strike = bool(np.any(belly & flying))
    home_d = float(np.hypot(ts["x"].iloc[-1], ts["y"].iloc[-1]))
    sink = None
    if td is not None:
        i = int(np.searchsorted(ts["t"].to_numpy(), td))
        sink = float(-ts["vz"].iloc[max(i - 3, 0):i + 1].min())
    if strike:
        i = int(np.argmax(belly & flying))
        return {"success": False, "reason": "terrain strike", "t_end": float(t[i]), "detail": f"pod, wing or tail touched the ground at t = {t[i]:.1f} s in phase {_phase_at(ep, t[i])}"}
    if not reserve_ok:
        return {"success": False, "reason": "battery reserve", "t_end": float(t[-1]), "detail": f"remaining {E['E_remaining_wh']:.1f} Wh below the reserve {E['reserve_wh']:.1f} Wh"}
    if "landed" not in phases:
        return {"success": False, "reason": "not landed", "t_end": float(t[-1]), "detail": f"ended in phase {phases[-1]} after {t[-1]:.0f} s"}
    if home_d > land_radius:
        return {"success": False, "reason": "landed off the meadow", "t_end": float(td), "detail": f"{home_d:.0f} m from home"}
    if sink is not None and sink > 2.5:
        return {"success": False, "reason": "hard landing", "t_end": float(td), "detail": f"sink {sink:.1f} m/s at touchdown"}
    return {"success": True, "reason": "landed", "t_end": float(td), "distance_m": home_d,
            "detail": f"touchdown {home_d:.0f} m from home, sink {sink:.1f} m/s, {E['E_remaining_wh']:.0f} Wh left of {E['E_available_wh']:.0f}, regenerated {E['regen_wh']:.1f} Wh"}


def phase_table(ep) -> pd.DataFrame:
    ts = timeseries(ep)
    starts = [(tt, name) for tt, name, note in ep.log["mission"] if note == "start"]
    rows = {}
    for i, (t0, name) in enumerate(starts):
        t1 = starts[i + 1][0] if i + 1 < len(starts) else ts["t"].iloc[-1]
        s = ts[(ts["t"] >= t0) & (ts["t"] < t1)]
        if len(s) < 2:
            continue
        dtm = float(np.median(np.diff(s["t"])))
        rows[name] = {"start [s]": t0, "duration [s]": t1 - t0, "distance [m]": float(np.sum(np.hypot(np.diff(s["x"]), np.diff(s["y"])))),
                      "altitude from / to [m]": f"{s['z_asl'].iloc[0]:.0f} / {s['z_asl'].iloc[-1]:.0f}", "min agl [m]": float(s["agl"].min()),
                      "mean EAS [m/s]": float(s["EAS"].mean()), "mean climb [m/s]": float((s["z_asl"].iloc[-1] - s["z_asl"].iloc[0]) / max(t1 - t0, 1e-9)),
                      "max sink [m/s]": float(-s["vz"].min()), "mean power [W]": float(s["P_el_W"].mean()),
                      "energy [Wh]": float(s["E_used_Wh"].iloc[-1] - s["E_used_Wh"].iloc[0]),
                      "electronics [Wh]": float((s["P_electronics_W"] * dtm).sum() / 3600), "regen [Wh]": float(s["regen_Wh"].iloc[-1] - s["regen_Wh"].iloc[0]),
                      "mean crow": float(s["crow"].mean()), "mean brake": float(s["brake"].mean()), "max |bank| [deg]": float(s["roll_deg"].abs().max()),
                      "max n_z": float(s["n_z"].max()), "stalled [s]": float(s["stalled"].sum() * dtm)}
    return pd.DataFrame(rows).T


def energy_table(ep) -> pd.DataFrame:
    """The energy as flown, by phase [Wh, kWh, % of nominal], the electronics' share, the regeneration (already inside
    each phase's energy, shown on its own), the prelaunch ground time, the totals and the reserve."""
    E = ep.log["energy"]
    pt = phase_table(ep)
    nom = E["E_nominal_wh"]
    df = pt[["duration [s]", "energy [Wh]", "electronics [Wh]", "regen [Wh]"]].copy().astype(float)
    ground = next((float(d.split("charged: ")[1].split(" Wh")[0]) for _, s, d in ep.log["events"] if s == "energy" and "charged" in d), 0.0)
    if "prelaunch" in df.index:
        df.loc["prelaunch", "energy [Wh]"] += ground
        df.loc["prelaunch", "electronics [Wh]"] = df.loc["prelaunch", "energy [Wh]"]
    df["energy [kWh]"] = df["energy [Wh]"] / 1000
    df["% of nominal"] = 100 * df["energy [Wh]"] / nom
    total = df["energy [Wh]"].sum()
    df.loc["TOTAL"] = [df["duration [s]"].sum(), total, df["electronics [Wh]"].sum(), df["regen [Wh]"].sum(), total / 1000, 100 * total / nom]
    df.loc["remaining (of available)"] = [np.nan, E["E_remaining_wh"], np.nan, np.nan, E["E_remaining_wh"] / 1000, 100 * E["E_remaining_wh"] / nom]
    df.loc["reserve required"] = [np.nan, E["reserve_wh"], np.nan, np.nan, (E["reserve_wh"] or 0) / 1000, 100 * (E["reserve_wh"] or 0) / nom]
    return df


def summary(ep) -> dict:
    ts = timeseries(ep)
    E = ep.log["energy"]
    o = ep.outcome
    t_l = next((tt for tt, name, note in ep.log["mission"] if name == "climb" and note == "start"), 0.0)
    fly = ts[(ts["agl"] > 5.0) & (ts["t"] > t_l)]
    dtm = float(np.median(np.diff(ts["t"])))
    cl = ts[(ts["t"] > t_l) & (ts["t"] < next((tt for tt, name, note in ep.log["mission"] if name == "transit" and note == "start"), ts["t"].iloc[-1]))]
    return {"condition": ep.log["scenario"]["name"], "success": o["success"], "reason": o["reason"], "airborne [s]": float(len(fly) * dtm),
            "return reason": ep.log.get("return_reason"), "max altitude [m]": float(ts["z_asl"].max()),
            "max density altitude [m]": fs.density_altitude(float(ts["rho"].min())), "mean climb in the climb [m/s]": float((cl["z_asl"].iloc[-1] - cl["z_asl"].iloc[0]) / max(cl["t"].iloc[-1] - cl["t"].iloc[0], 1)) if len(cl) > 2 else np.nan,
            "max sink [m/s]": float(-fly["vz"].min()) if len(fly) else np.nan, "min agl in flight [m]": float(fly["agl"].min()) if len(fly) else np.nan,
            "max EAS [m/s]": float(ts["EAS"].max()), "max n_z": float(fly["n_z"].max()) if len(fly) else np.nan, "stall time [s]": float(fly["stalled"].sum() * dtm),
            "energy used [Wh]": E["E_used_wh"], "regenerated [Wh]": E["regen_wh"], "remaining [% available]": 100 * E["E_remaining_wh"] / E["E_available_wh"],
            "detail": o["detail"]}


def compare(eps: list) -> pd.DataFrame:
    return pd.DataFrame([summary(e) for e in eps]).set_index("condition")


# ------------------------------------------------------------------------------------------------- the movie
def _cameras(ep, idx):
    log = ep.log
    com = np.asarray(log["body_pos"])[:, 0]
    quat = np.asarray(log["body_quat"])[:, 0]
    vel = np.asarray(log["body_linvel"])[:, 0]
    t = np.asarray(log["t"])
    starts = [(tt, name) for tt, name, note in log["mission"] if note == "start"]
    chase, onboard, ground = [], [], []
    head = None
    home = np.array([0.0, 0.0, float(log.get("home_alt", 1200.0))])
    tripod = home + np.array([-15.0, 25.0, 1.7])
    for i in idx:
        c, v = com[i], vel[i]
        if head is None:
            head = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(v) > 2.0:
            d = v / np.linalg.norm(v)
            head = head + 0.15 * (d - head); head /= np.linalg.norm(head)
        chase.append((tuple(c - 11.0 * head + np.array([0, 0, 3.0])), tuple(c + 3.0 * head), (0, 0, 1)))
        R = fr.quat_to_R(quat[i])
        fwd, up = R[:, 0], R[:, 2]
        look = math.cos(math.radians(20)) * fwd - math.sin(math.radians(20)) * up
        eye = c + 0.45 * fwd - 0.05 * up
        onboard.append((tuple(eye), tuple(eye + 10.0 * look), tuple(up)))
        phase = next((name for tt, name in reversed(starts) if tt <= t[i] + 1e-9), "prelaunch")
        near = np.linalg.norm(c[:2] - home[:2]) < 250.0 and c[2] - home[2] < 120.0 and phase in ("approach", "landed", "prelaunch", "launch")
        ground.append((tuple(tripod), tuple(c), (0, 0, 1)) if near else None)
    return chase, onboard, ground


def _cropped(ep, margin=1200.0):
    """A shallow copy of the log whose terrain grid is cropped around the flight (a 10 km mesh stretches the renderer's
    clipping range: the chase camera's near plane would cut the aircraft)."""
    log = dict(ep.log)
    gp = dict(log["geom_pose"])
    Z = np.asarray(gp["terrain_z"])
    x0, x1, y0, y1 = np.asarray(gp["terrain_extent"], float)
    xs, ys = np.linspace(x0, x1, Z.shape[1]), np.linspace(y0, y1, Z.shape[0])
    com = np.asarray(log["com"])
    lo, hi = com[:, :2].min(0) - margin, com[:, :2].max(0) + margin
    ci = (xs >= lo[0]) & (xs <= hi[0])
    ri = (ys >= lo[1]) & (ys <= hi[1])
    gp["terrain_z"] = Z[np.ix_(ri, ci)]
    gp["terrain_extent"] = np.array([xs[ci][0], xs[ci][-1], ys[ri][0], ys[ri][-1]])
    log["geom_pose"] = gp
    return log


def _overlay(img, lines, title):
    import nisus_scenario as nsc
    return nsc._overlay(img, lines, title)


def _map(ep, i_now, massif: fr.Massif, scn: Scenario, size=(300, 240)):
    """A top-down map: the terrain shaded by height with contours, the survey legs, home, the track, the aircraft, the wind."""
    import cv2
    w, h = size
    com = np.asarray(ep.log["body_pos"])[:, 0]
    pts = np.vstack([com[:, :2], [[0, 0]], [[scn.survey_x[0], scn.survey_y[0]]], [[scn.survey_x[1], scn.survey_y[-1]]]])
    lo, hi = pts.min(0) - 500, pts.max(0) + 500
    span = float(max(hi - lo))
    cen = 0.5 * (lo + hi)
    xs = cen[0] + (np.arange(w) - w / 2) / (w - 20) * span
    ys = cen[1] - (np.arange(h) - h / 2) / (w - 20) * span
    X, Y = np.meshgrid(xs, ys)
    Z = massif.height(X, Y)
    zn = (Z - 1100.0) / 3300.0
    img = np.stack([60 + 120 * zn, 90 + 110 * zn, 50 + 150 * zn], -1).clip(0, 255).astype(np.uint8)
    for lev in np.arange(1500, 4400, 500):
        m = (np.abs(Z - lev) < span / w * 0.6 * 1.2).astype(np.uint8)
        img[m > 0] = (img[m > 0] * 0.55).astype(np.uint8)

    def px(xy):
        return (int(w / 2 + (xy[0] - cen[0]) / span * (w - 20)), int(h / 2 - (xy[1] - cen[1]) / span * (w - 20)))

    for y in scn.survey_y:
        cv2.line(img, px((scn.survey_x[0], y)), px((scn.survey_x[1], y)), (240, 210, 120), 1, cv2.LINE_AA)
    track = com[:i_now + 1, :2]
    for j in range(1, len(track), 2):
        cv2.line(img, px(track[j - 1]), px(track[j]), (255, 240, 160), 1, cv2.LINE_AA)
    cv2.circle(img, px((0, 0)), 4, (240, 240, 240), -1)
    cv2.putText(img, "meadow", (px((0, 0))[0] + 6, px((0, 0))[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (250, 250, 250), 1, cv2.LINE_AA)
    cv2.circle(img, px(com[i_now, :2]), 4, (80, 200, 255), -1)
    wv = np.asarray(scn.wind[:2], float)
    if np.linalg.norm(wv) > 0.3:
        o = (w - 40, 30); e = (int(o[0] + 18 * wv[0] / np.linalg.norm(wv)), int(o[1] - 18 * wv[1] / np.linalg.norm(wv)))
        cv2.arrowedLine(img, o, e, (255, 255, 255), 2, tipLength=0.4)
        cv2.putText(img, f"wind {np.linalg.norm(wv):.0f} m/s", (w - 95, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"map {span / 1000:.1f} km across, contours 500 m", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def _profile(ep, i_now, size=(300, 110)):
    """The altitude profile so far: the aircraft (white) over the ground below it (brown)."""
    import cv2
    w, h = size
    ts = timeseries(ep)
    t = ts["t"].to_numpy()
    z, g = ts["z_asl"].to_numpy(), (ts["z_asl"] - ts["agl"]).to_numpy()
    img = np.full((h, w, 3), (35, 40, 50), np.uint8)
    t_end = t[-1]
    zmin, zmax = min(g.min(), z.min()) - 50, max(z.max(), g.max()) + 100
    X = lambda tt: int(5 + (w - 10) * tt / max(t_end, 1))
    Yv = lambda zz: int(h - 12 - (h - 24) * (zz - zmin) / (zmax - zmin))
    t_now = float(np.asarray(ep.log["t"])[i_now])
    k = int(np.searchsorted(t, t_now))
    for j in range(1, max(k, 1), 3):
        cv2.line(img, (X(t[j - 3 if j >= 3 else 0]), Yv(g[j - 3 if j >= 3 else 0])), (X(t[j]), Yv(g[j])), (60, 110, 160), 1)
        cv2.line(img, (X(t[j - 3 if j >= 3 else 0]), Yv(z[j - 3 if j >= 3 else 0])), (X(t[j]), Yv(z[j])), (240, 240, 240), 1)
    cv2.putText(img, f"altitude {zmin:.0f}-{zmax:.0f} m", (6, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def render_movie(ep, scn: Scenario, path, *, speed: float = 30.0, fps: int = 20, size=(960, 540), onboard=True, massif: fr.Massif | None = None,
                 progress: bool = False):
    """The mission as an MP4 (NISUS's movie over the mountains): the chase camera (the meadow's tripod for the launch
    and the landing), the synthetic onboard view, the map with the terrain, the altitude profile, the overlays.
    ``progress``: a tqdm bar (the chase and onboard renders, then a step per composed frame, the write)."""
    import cv2
    from vegeta.aeromant._watermark import watermark
    from vegeta.chiron import viz
    from pekari_controller import end_card

    massif = massif or fr.Massif()
    log = ep.log
    t = np.asarray(log["t"])
    dt = float(t[1] - t[0])
    every = max(1, int(round(speed / (fps * dt))))
    t_end = (log.get("touchdown") or t[-1]) + 6.0
    stop = min(int(np.searchsorted(t, t_end)) + 1, len(t))
    idx = list(range(0, stop, every))
    chase, onboard_cams, ground = _cameras(ep, idx)
    main_cams = [g if g is not None else c for c, g in zip(chase, ground)]
    it_main = iter(main_cams)
    clog = _cropped(ep)
    common = dict(every=every, stop=idx[-1] + 1, show_time=False, scenery_range=5000.0, ground_color="#7f9a68", background="#b9d3ee")
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(idx) + 3, desc=f"movie {scn.slug}", bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {elapsed} {postfix}")
        bar.set_postfix_str(f"chase camera: {len(idx)} frames")
    imgs = viz.frames(clog, camera=lambda com: next(it_main), size=size, **common)
    if bar:
        bar.update(1); bar.set_postfix_str("onboard camera")
    if onboard:
        it_on = iter(onboard_cams)
        small = viz.frames(clog, camera=lambda com: next(it_on), size=(size[0] // 3 // 2 * 2, size[1] // 3 // 2 * 2), **common)
    if bar:
        bar.update(1); bar.set_postfix_str("overlays, map, profile")
    ts = timeseries(ep)
    ta = ts["t"].to_numpy()
    starts = [(tt, name) for tt, name, note in log["mission"] if note == "start"]
    events = sorted(log.get("events", []), key=lambda e: e[0])
    E = log["energy"]
    title = f"{scn.label}   ({speed:g}x; MuJoCo/Chiron, ISA density at the altitude, forces from the coefficient table)"
    out = []
    for k, i in enumerate(idx[:len(imgs)]):
        ti = t[i]
        j = min(int(np.searchsorted(ta, ti)), len(ts) - 1)
        row = ts.iloc[j]
        phase = next((name for tt, name in reversed(starts) if tt <= ti + 1e-9), "prelaunch")
        img = np.ascontiguousarray(imgs[k])
        lines = [(f"t = {ti:6.1f} s   phase: {phase}", (255, 225, 120)),
                 (f"altitude {row['z_asl']:5.0f} m   above ground {row['agl']:5.0f} m   density alt. {fs.density_altitude(row['rho']):5.0f} m", (210, 230, 255)),
                 (f"EAS {row['EAS']:4.1f} m/s  TAS {row['V']:4.1f} m/s   climb {row['vz']:+5.1f} m/s   air {row['w_air']:+4.1f} m/s", (210, 230, 255)),
                 (f"bank {row['roll_deg']:+5.0f}  pitch {row['pitch_deg']:+5.1f} deg   n {row['n_z']:4.2f}   crow {100 * row['crow']:3.0f} %   brake {100 * row['brake']:3.0f} %", (210, 230, 255)),
                 (f"throttle {100 * row['throttle']:3.0f} %  thrust {row['thrust_N']:+5.1f} N  propeller {row['P_prop_W']:+6.0f} W  pack {row['V_batt']:4.1f} V", (210, 230, 255)),
                 (f"battery used {row['E_used_Wh']:5.1f} Wh of {E['E_available_wh']:.0f}   regenerated {row['regen_Wh']:4.2f} Wh   reserve {E['reserve_wh'] or 0:.0f} Wh", (180, 255, 180))]
        if row["stalled"] > 0.5:
            lines.append(("STALL", (255, 110, 110)))
        recent = [e for e in events if 0 <= ti - e[0] < 7.0 and e[1] != "energy"]
        if recent:
            lines.append((f"> {recent[-1][1]}: {recent[-1][2][:95]}", (255, 200, 90)))
        img = _overlay(img, lines, title)
        if onboard:
            ins = np.ascontiguousarray(small[k])
            cv2.rectangle(ins, (0, 0), (ins.shape[1] - 1, 20), (25, 30, 40), -1)
            cv2.putText(ins, "onboard camera (SYNTHETIC render)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
            top = int(36 * min(size[0] / 960.0, 1.0))
            img[top:top + ins.shape[0], size[0] - ins.shape[1] - 10:size[0] - 10] = ins
        mp = _map(ep, i, massif, scn, size=(int(size[0] * 0.3), int(size[1] * 0.38)))
        pr = _profile(ep, i, size=(mp.shape[1], int(size[1] * 0.14)))
        y1 = size[1] - mp.shape[0] - 10
        img[y1:y1 + mp.shape[0], size[0] - mp.shape[1] - 10:size[0] - 10] = mp
        img[y1 - pr.shape[0] - 4:y1 - 4, size[0] - pr.shape[1] - 10:size[0] - 10] = pr
        out.append(img)
        if bar:
            bar.update(1)
    out = end_card(out, ep, fps=fps, hold_s=2.5)
    if bar:
        bar.set_postfix_str(f"writing {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = out[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    try:
        for img in out:
            vw.write(watermark(cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)))
    finally:
        vw.release()
    if bar:
        bar.update(1); bar.set_postfix_str("done"); bar.close()
    return path
