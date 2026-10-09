"""NISUS missions in MuJoCo: the scene, the runs, the judge, the tables and the movies (notebook 31,
``scenarios/nisus_mission.py``).

A ``Scenario`` names the variant (OBS or Zero), the pack, the wind (steady vector, turbulence, a discrete gust), the
Jetson failure and the survey (a field ``radius_m`` from the launch point, a lawnmower of legs). ``make_lab`` builds the
aircraft (``nisus_robot``) on flat ground with the launch pad, the field and some scenery as props and installs the
``Aero`` hook; ``run`` flies it with the ``FlightController`` (OBS: the scripted pilot; Zero: the computer's waypoints),
the launch as a scheduled impulse. ``outcome`` judges from the physics: a landing on the skid near home with a gentle
sink and the reserve kept is a success; a nose or wing strike, a touchdown far from home or a battery below the
reserve is a failure with its reason. ``energy_table`` splits the energy by phase (Wh, kWh, % of nominal, the
electronics' share) and compares with the plan of ``nisus_systems.mission_energy``; ``compare`` lines the scenarios up.
``render_movie`` draws the flight with Chiron's renderer: a chase view, a synthetic onboard-camera inset, a map inset,
the overlays (airspeed, height, bank and load factor, time, battery energy and the return margin), the end card.

    import nisus_scenario as nsc
    scn = nsc.Scenario("Zero")                       # calm air
    lab = nsc.make_lab(scn); ep = nsc.run(lab, scn)
    nsc.outcome(ep), nsc.energy_table(ep), nsc.render_movie(ep, scn, "zero_calm.mp4")
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from vegeta import chiron as ch

import nisus
import nisus_controller as nc
import nisus_flight as nf
import nisus_robot as nr
import nisus_systems as ns

__all__ = ["Scenario", "CONDITIONS", "standard_scenarios", "make_lab", "run", "outcome", "timeseries", "phase_table", "energy_table", "compare",
           "render_movie", "scenery"]


@dataclass
class Scenario:
    variant: str = "Zero"
    name: str = "calm"
    battery_key: str = "gens-ace-3s-2200"
    wind: tuple = (0.0, 0.0, 0.0)            # the air's velocity [m/s], world (x east, y north)
    sigma: float = 0.0                        # turbulence
    gust: tuple | None = None                 # (amplitude, t_start, duration, direction)
    jetson_failure: tuple | None = None       # (t_fail, t_reboot)
    radius_m: float = 600.0
    legs: int = 3
    leg_length: float = 300.0
    spacing: float = 60.0
    alt_m: float = 80.0
    launch_heading_deg: float = 0.0           # into the wind when there is one
    duration: float = 1800.0
    seed: int = 0
    plan: ns.MissionPlan = field(default_factory=ns.MissionPlan)

    def __post_init__(self):
        w = np.asarray(self.wind[:2], float)
        if np.linalg.norm(w) > 0.3:
            self.launch_heading_deg = math.degrees(math.atan2(-w[1], -w[0]))
        self.plan = ns.MissionPlan(**{**asdict(self.plan), "radius_m": self.radius_m, "climb_alt_m": self.alt_m, "wind_m_s": float(np.linalg.norm(w))})

    @property
    def label(self):
        return f"Nisus-{self.variant} — {self.name}"

    @property
    def slug(self):
        """A file-name-safe name: ``nisus_<variant>_<condition>`` ("headwind 6 m/s" → "headwind_6_ms")."""
        import re
        return f"nisus_{self.variant}_" + re.sub(r"[^A-Za-z0-9]+", "_", self.name.replace("m/s", "ms")).strip("_")

    @property
    def field_centre(self):
        a = math.radians(self.launch_heading_deg)
        return np.array([self.radius_m * math.cos(a), self.radius_m * math.sin(a)])

    def survey_waypoints(self):
        return nc.survey_pattern(self.field_centre, self.launch_heading_deg + 90.0, self.legs, self.leg_length, self.spacing, self.alt_m, self.plan.V_survey)


CONDITIONS = {
    "calm": dict(),
    "headwind 6 m/s": dict(wind=(-6.0, 0.0, 0.0), sigma=0.6),
    "gusty": dict(wind=(-4.0, 0.0, 0.0), sigma=1.5, gust=(5.0, 330.0, 4.0, (0.0, 0.0, 1.0))),
    "Jetson failure": dict(jetson_failure=(300.0, 330.0)),
}


def standard_scenarios() -> list:
    """OBS and Zero in calm air, both in the headwind and the gusts, and Zero with the Jetson failure."""
    out = [Scenario("OBS", "calm"), Scenario("Zero", "calm"),
           Scenario("OBS", "headwind 6 m/s", **CONDITIONS["headwind 6 m/s"]), Scenario("Zero", "headwind 6 m/s", **CONDITIONS["headwind 6 m/s"]),
           Scenario("Zero", "gusty", **CONDITIONS["gusty"]), Scenario("Zero", "Jetson failure", **CONDITIONS["Jetson failure"])]
    return out


# ------------------------------------------------------------------------------------------------- the scene
def _box(name, centre, half, rgba):
    return ch.Prop(ch.Link(name, pos=tuple(centre), geoms=[ch.Geom(f"{name}_g", "box", tuple(half), mass=None, role="link", rgba=rgba)]))


def scenery(scn: Scenario) -> list:
    """The launch pad, the survey field (a flat lighter rectangle), a farm house and a few trees near home, a tree
    line beside the field: props that only draw (and could be hit)."""
    props = []
    a = math.radians(scn.launch_heading_deg)
    u = np.array([math.cos(a), math.sin(a)]); n = np.array([-u[1], u[0]])
    props.append(_box("pad", (0.0, 0.0, 0.01), (1.5, 1.5, 0.01), (0.35, 0.35, 0.38, 1.0)))
    c = scn.field_centre
    half_l, half_w = scn.leg_length / 2 + 20, (scn.legs * scn.spacing) / 2 + 20
    quat = (math.cos(a / 2), 0, 0, math.sin(a / 2))
    props.append(ch.Prop(ch.Link("field", pos=(float(c[0]), float(c[1]), 0.005),
                                 geoms=[ch.Geom("field_g", "box", (half_w, half_l, 0.005), quat=quat, mass=None, role="link", rgba=(0.76, 0.72, 0.42, 1.0))])))
    house = -40 * u + 25 * n
    props.append(_box("house", (float(house[0]), float(house[1]), 2.0), (5.0, 4.0, 2.0), (0.75, 0.45, 0.35, 1.0)))
    props.append(_box("roof", (float(house[0]), float(house[1]), 4.6), (5.5, 4.5, 0.6), (0.45, 0.25, 0.2, 1.0)))
    rng = np.random.default_rng(1)
    for i in range(10):
        q = c + (rng.uniform(-half_l, half_l)) * u + (half_w + 15 + rng.uniform(0, 10)) * n
        h = rng.uniform(5, 9)
        props.append(ch.Prop(ch.Link(f"tree{i}", pos=(float(q[0]), float(q[1]), 0.0),
                                     geoms=[ch.Geom(f"tree{i}_t", "cylinder", (0.25, h / 2), pos=(0, 0, h / 2), mass=None, role="link", rgba=(0.4, 0.28, 0.15, 1.0)),
                                            ch.Geom(f"tree{i}_c", "sphere", (h * 0.35,), pos=(0, 0, h), mass=None, role="link", rgba=(0.2, 0.5, 0.2, 1.0))])))
    for i in range(4):
        q = -25 * u + (-30 + 20 * i) * n
        props.append(ch.Prop(ch.Link(f"htree{i}", pos=(float(q[0]), float(q[1]), 0.0),
                                     geoms=[ch.Geom(f"htree{i}_t", "cylinder", (0.2, 3.0), pos=(0, 0, 3.0), mass=None, role="link", rgba=(0.4, 0.28, 0.15, 1.0)),
                                            ch.Geom(f"htree{i}_c", "sphere", (2.2,), pos=(0, 0, 6.0), mass=None, role="link", rgba=(0.25, 0.55, 0.22, 1.0))])))
    return props


# ------------------------------------------------------------------------------------------------- the lab and the run
def make_lab(scn: Scenario, *, pm: ns.PropulsionMap | None = None, p=None, **kwargs):
    """The aircraft of the scenario in ChironLab with the ``Aero`` hook (``lab.aero``), the wind, the propulsion map
    and the coefficient table (``lab.coeff``)."""
    p = nisus.resolve(p)
    robot = nr.nisus_robot(scn.variant, scn.battery_key, p)
    pm = pm or ns.propulsion_map()
    ci = ns.cg_inertia(robot.mass_table, p)
    deriv = nf.derivatives(p, x_cg_m=ci["x_cg_m"], z_cg_m=ci["z_cg_m"])
    coeff = nf.coefficients(deriv)
    opts = dict(nr.LAB_OPTIONS); opts.update(kwargs)
    lab = ch.ChironLab(robot, ch.Flat(), props=scenery(scn), **opts)
    wind = nr.Wind(steady=tuple(scn.wind), sigma=scn.sigma, gust=scn.gust, seed=scn.seed)
    el = ns.phase_power(scn.variant)["electronics battery-side [W]"].to_dict()
    lab.aero = nr.Aero(robot, coeff, pm, ns.battery(scn.battery_key), wind=wind, electronics_w=el["prelaunch"], derating=scn.plan.derating)
    lab.add_hook(lab.aero)
    lab.coeff, lab.deriv, lab.pm, lab.ci, lab.el = coeff, deriv, pm, ci, el
    lab.scenario = scn
    return lab


def controller(lab, scn: Scenario) -> nc.FlightController:
    af = nf.airframe(scn.variant, scn.battery_key, lab.robot.params)
    em = nc.EnergyManager(lab.aero, lab.pm, af.mass_kg, af.cd0, af.aspect_ratio, af.oswald, af.wing_area_m2, scn.plan, lab.el["return"],
                          V_app=1.3 * af.stall_speed())
    wps = scn.survey_waypoints()
    jet_w = ns.JETSON_INSTALLATION_W + 2.0 / 0.9 if scn.variant == "Zero" else 0.0
    if scn.variant == "Zero":
        fc = nc.FlightController(lab.aero, scn.plan, launch_heading_deg=scn.launch_heading_deg, mode="auto", mission=nc.AutoMission(wps, repeats=40),
                                 survey_wps=wps, energy=em, electronics_phase_w=lab.el, jetson_w=jet_w, jetson_failure=scn.jetson_failure,
                                 wind_estimate=tuple(scn.wind[:2]), ground_s=scn.plan.ground_s, name=f"Nisus-Zero autopilot ({scn.name})")
    else:
        pilot = nc.ScriptedPilot(scn.field_centre, scn.launch_heading_deg + 90.0, lanes=scn.legs, lane_length=scn.leg_length,
                                 speed=scn.plan.V_survey, h=scn.alt_m)
        fc = nc.FlightController(lab.aero, scn.plan, launch_heading_deg=scn.launch_heading_deg, mode="pilot", pilot=pilot, survey_wps=wps[:1],
                                 energy=em, electronics_phase_w=lab.el, wind_estimate=tuple(scn.wind[:2]), ground_s=scn.plan.ground_s,
                                 name=f"Nisus-OBS scripted pilot ({scn.name})")
    fc.V_app = 1.3 * af.stall_speed()
    return fc


def run(lab, scn: Scenario, *, duration: float | None = None, launch_speed: float = 12.0, launch_pitch_deg: float = 10.0) -> ch.Episode:
    """The mission. The hand launch is a Chiron disturbance: an impulse of ``m x launch_speed`` over 0.30 s along the
    launch heading, ``launch_pitch_deg`` upwards, starting at the controller's launch time (the thrower's hand).
    ``ep.log`` gets ``aero`` (the hook's rows), ``mission`` (phase log), ``events``, ``energy``."""
    fc = controller(lab, scn)
    m = float(lab.model.body_subtreemass[lab._body_id("nisus")])
    a, e = math.radians(scn.launch_heading_deg), math.radians(launch_pitch_deg)
    lab._disturbances = [d for d in lab._disturbances if d.body != "nisus"]
    lab.add_disturbance(ch.Disturbance("nisus", t_start=fc.prelaunch_sim_s, duration=0.30, impulse=m * launch_speed,
                                       direction=(math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))))
    z0 = lab.nominal_base_height + 1.6                                # held at chest height by the thrower
    ep = lab.run(fc, duration=duration or scn.duration, rules=None, settle=0.0, seed=scn.seed, base_pos=(0.0, 0.0, z0), base_yaw=a,
                 info={"controller": fc.name, "treatment": f"{scn.variant}:{scn.name}"})
    aero = lab.aero
    ep.log["aero"] = np.asarray(aero.history, float).reshape(-1, len(nr.Aero.COLUMNS))
    ep.log["aero_columns"] = list(nr.Aero.COLUMNS)
    ep.log["mission"] = [list(x) for x in fc.log]
    ep.log["events"] = sorted(list(ep.log.get("events", [])) + [list(x) for x in fc.events], key=lambda r: r[0])
    ep.log["touchdown"] = fc.touchdown
    ep.log["return_reason"] = fc.return_reason
    ep.log["scenario"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(scn).items() if k != "plan"}
    ep.log["plan"] = asdict(scn.plan)
    ep.log["energy"] = {"E_nominal_wh": aero.battery.energy_wh, "E_available_wh": aero.E_available_Wh, "E_used_wh": aero.E_used_Wh,
                        "E_remaining_wh": aero.E_remaining_Wh, "reserve_wh": fc.energy.E_reserve if fc.energy else None,
                        "launch_impulse_Ns": m * launch_speed, "launch_speed": launch_speed}
    ep.log["mass_kg"] = m
    ep.log["mission_finished"] = fc.finished
    ep.outcome = outcome(ep)
    ep.controller = fc
    return ep


# ------------------------------------------------------------------------------------------------- the judge and the tables
def timeseries(ep) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(ep.log["aero"]), columns=list(ep.log["aero_columns"]))


def _phase_at(ep, t):
    starts = [(tt, name) for tt, name, note in ep.log["mission"] if note == "start"]
    return next((name for tt, name in reversed(starts) if tt <= t + 1e-9), "prelaunch")


def outcome(ep) -> dict:
    """Success: landed (phase reached) within 100 m of home (a belly landing in the home field), touchdown sink ≤ 2.5 m/s, no nose or wing strike while
    flying, the energy reserve kept. Otherwise the first failure found."""
    log = ep.log
    ts = timeseries(ep)
    t = np.asarray(log["t"])
    bodies = list(log["bodies"])
    belly = np.asarray(log["belly_contact"])[:, bodies.index("nisus")] if "belly_contact" in log else np.zeros(len(t), bool)
    E = log["energy"]
    reserve_ok = E["reserve_wh"] is None or E["E_remaining_wh"] >= E["reserve_wh"] - 1e-6
    phases = [name for _, name, note in log["mission"] if note == "start"]
    td = log.get("touchdown")
    # a belly contact (pod, wing, booms, tail touch the ground) while airborne, after the launch and before the touchdown
    t_launch = next((tt for tt, name, note in log["mission"] if name == "launch" and note == "start"), 0.0)
    t_td = td if td is not None else t[-1]
    flying = (t > t_launch + 1.0) & (t < t_td - 0.2)
    strike = bool(np.any(belly & flying))
    home_d = float(np.hypot(ts["x"].iloc[-1], ts["y"].iloc[-1]))
    sink = None
    if td is not None:
        i = int(np.searchsorted(ts["t"].to_numpy(), td))
        sink = float(-ts["vz"].iloc[max(i - 2, 0):i + 1].min())
    if strike:
        i = int(np.argmax(belly & flying))
        return {"success": False, "reason": "airframe strike", "t_end": float(t[i]), "detail": f"pod, wing or tail touched the ground at t = {t[i]:.1f} s in phase {_phase_at(ep, t[i])}"}
    if not reserve_ok:
        return {"success": False, "reason": "battery reserve", "t_end": float(t[-1]), "detail": f"remaining {E['E_remaining_wh']:.2f} Wh below the reserve {E['reserve_wh']:.2f} Wh"}
    if "landed" not in phases:
        return {"success": False, "reason": "not landed", "t_end": float(t[-1]), "detail": f"ended in phase {phases[-1]} after {t[-1]:.0f} s"}
    if home_d > 100.0:
        return {"success": False, "reason": "landed away from home", "t_end": float(td), "detail": f"{home_d:.0f} m from the launch point"}
    if sink is not None and sink > 2.5:
        return {"success": False, "reason": "hard landing", "t_end": float(td), "detail": f"sink {sink:.1f} m/s at touchdown"}
    return {"success": True, "reason": "landed", "t_end": float(td), "distance_m": home_d,
            "detail": f"touchdown {home_d:.0f} m from home, sink {sink:.1f} m/s, {E['E_remaining_wh']:.2f} Wh left of {E['E_available_wh']:.2f} available"}


def phase_table(ep) -> pd.DataFrame:
    """Start, duration, distance flown, mean airspeed, mean electrical power and energy of each phase (as flown)."""
    ts = timeseries(ep)
    starts = [(tt, name) for tt, name, note in ep.log["mission"] if note == "start"]
    rows = {}
    for i, (t0, name) in enumerate(starts):
        t1 = starts[i + 1][0] if i + 1 < len(starts) else ts["t"].iloc[-1]
        s = ts[(ts["t"] >= t0) & (ts["t"] < t1)]
        if len(s) < 2:
            continue
        dist = float(np.sum(np.hypot(np.diff(s["x"]), np.diff(s["y"]))))
        E = float(s["E_used_Wh"].iloc[-1] - s["E_used_Wh"].iloc[0])
        rows[name] = {"start [s]": t0, "duration [s]": t1 - t0, "distance [m]": dist, "mean airspeed [m/s]": float(s["V"].mean()),
                      "mean height [m]": float(s["h"].mean()), "mean power [W]": float(s["P_el_W"].mean()), "energy [Wh]": E,
                      "electronics [Wh]": float((s["P_electronics_W"] * np.gradient(s["t"])).sum() / 3600),
                      "max |bank| [deg]": float(s["roll_deg"].abs().max()), "max n_z": float(s["n_z"].max()), "stalled [s]": float(s["stalled"].sum() * np.median(np.diff(s["t"])) if len(s) > 1 else 0.0)}
    return pd.DataFrame(rows).T


def energy_table(ep) -> pd.DataFrame:
    """The energy as flown, phase by phase, in Wh, kWh and % of the nominal pack energy, with the electronics' share;
    the last rows: the prelaunch ground operation charged before the simulated time, totals, the reserve test."""
    E = ep.log["energy"]
    pt = phase_table(ep)
    nom = E["E_nominal_wh"]
    df = pt[["duration [s]", "energy [Wh]", "electronics [Wh]"]].copy()
    ground = next((float(d.split("charged: ")[1].split(" Wh")[0]) for _, s, d in ep.log["events"] if s == "energy" and "charged" in d), 0.0)
    df.loc["prelaunch", "energy [Wh]"] = df.loc["prelaunch", "energy [Wh]"] + ground if "prelaunch" in df.index else ground
    df.loc["prelaunch", "electronics [Wh]"] = df.loc["prelaunch", "energy [Wh]"]
    df["energy [kWh]"] = df["energy [Wh]"] / 1000
    df["% of nominal"] = 100 * df["energy [Wh]"] / nom
    df["electronics share"] = df["electronics [Wh]"] / df["energy [Wh]"].replace(0, np.nan)
    total = df["energy [Wh]"].sum()
    df.loc["TOTAL"] = [df["duration [s]"].sum(), total, df["electronics [Wh]"].sum(), total / 1000, 100 * total / nom, df["electronics [Wh]"].sum() / total]
    df.loc["remaining (of available)"] = [np.nan, E["E_remaining_wh"], np.nan, E["E_remaining_wh"] / 1000, 100 * E["E_remaining_wh"] / nom, np.nan]
    df.loc["reserve required"] = [np.nan, E["reserve_wh"], np.nan, (E["reserve_wh"] or 0) / 1000, 100 * (E["reserve_wh"] or 0) / nom, np.nan]
    return df


def summary(ep) -> dict:
    ts = timeseries(ep)
    E = ep.log["energy"]
    o = ep.outcome
    t_l = next((tt for tt, name, note in ep.log["mission"] if name == "climb" and note == "start"), 0.0)
    fly = ts[(ts["h"] > 3.0) & (ts["t"] > t_l)]
    return {"variant": ep.log["scenario"]["variant"], "condition": ep.log["scenario"]["name"], "success": o["success"], "reason": o["reason"],
            "airborne [s]": float(len(fly) * np.median(np.diff(ts["t"]))) if len(fly) > 1 else 0.0, "touchdown [s]": ep.log.get("touchdown"),
            "return reason": ep.log.get("return_reason"), "max height [m]": float(ts["h"].max()), "max airspeed [m/s]": float(ts["V"].max()),
            "min airspeed in flight [m/s]": float(fly["V"].min()) if len(fly) else np.nan, "max |bank| [deg]": float(fly["roll_deg"].abs().max()) if len(fly) else np.nan,
            "max n_z": float(fly["n_z"].max()) if len(fly) else np.nan, "stall time in flight [s]": float(fly["stalled"].sum() * np.median(np.diff(ts["t"]))),
            "energy used [Wh]": E["E_used_wh"], "remaining [Wh]": E["E_remaining_wh"], "remaining [% available]": 100 * E["E_remaining_wh"] / E["E_available_wh"],
            "reserve [Wh]": E["reserve_wh"], "detail": o["detail"]}


def compare(eps: list) -> pd.DataFrame:
    return pd.DataFrame([summary(e) for e in eps]).set_index(["variant", "condition"])


# ------------------------------------------------------------------------------------------------- the movie
def _cameras(ep, idx, scn: Scenario):
    """Per frame: the chase camera (behind and above the aircraft along its track), the synthetic onboard camera (at the
    nose, looking 15° down along the body axis) and, within 120 m of home after the return, a ground camera on a
    tripod by the pad."""
    log = ep.log
    ts = timeseries(ep)
    com = np.asarray(log["body_pos"])[:, 0]
    quat = np.asarray(log["body_quat"])[:, 0]
    vel = np.asarray(log["body_linvel"])[:, 0]
    t = np.asarray(log["t"])
    starts = [(tt, name) for tt, name, note in log["mission"] if note == "start"]
    chase, onboard, ground = [], [], []
    head = np.array([math.cos(math.radians(scn.launch_heading_deg)), math.sin(math.radians(scn.launch_heading_deg)), 0.0])
    tripod = np.array([-6.0, 8.0, 1.7])
    for i in idx:
        c, v = com[i], vel[i]
        if np.linalg.norm(v[:2]) > 2.0:
            d = v / np.linalg.norm(v)
            head = head + 0.2 * (d - head); head /= np.linalg.norm(head)
        pos = c - 7.0 * head + np.array([0, 0, 2.2])
        chase.append((tuple(pos), tuple(c + 2.0 * head), (0, 0, 1)))
        R = nr.quat_to_R(quat[i])
        fwd, up = R[:, 0], R[:, 2]
        look = math.cos(math.radians(15)) * fwd - math.sin(math.radians(15)) * up
        eye = c + 0.33 * fwd - 0.03 * up
        onboard.append((tuple(eye), tuple(eye + 10.0 * look), tuple(up)))
        phase = next((name for tt, name in reversed(starts) if tt <= t[i] + 1e-9), "prelaunch")
        near = np.linalg.norm(c[:2]) < 120.0 and phase in ("approach", "landed", "prelaunch", "launch")
        ground.append((tuple(tripod), tuple(c), (0, 0, 1)) if near else None)
    return chase, onboard, ground


def _overlay(img, lines, title, *, label_tr=None):
    import cv2
    h, w = img.shape[:2]
    k = min(w / 960.0, 1.0)                                      # the text scales with the frame (a 640 px movie stays readable)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar, dy = int(30 * k), int(22 * k)
    cv2.rectangle(img, (0, 0), (w, bar), (25, 30, 40), -1)
    cv2.putText(img, title, (int(12 * k), int(21 * k)), font, 0.58 * k, (240, 240, 240), 1, cv2.LINE_AA)
    width = min(max(cv2.getTextSize(text, font, 0.5 * k, 1)[0][0] for text, _ in lines) + int(24 * k), w)
    top = bar + int(6 * k)
    box = img[top:top + dy * len(lines) + int(10 * k), 0:width]
    box[:] = (0.45 * box + 0.55 * np.array([20, 24, 32])).astype(np.uint8)
    y = top + int(18 * k)
    for text, colour in lines:
        cv2.putText(img, text, (int(12 * k), y), font, 0.5 * k, colour, 1, cv2.LINE_AA)
        y += dy
    return img


def _map(ep, i_now, scn: Scenario, size=(300, 220)):
    """A top-down map: home, the field, the track so far, the aircraft, the wind arrow."""
    import cv2
    w, h = size
    img = np.full((h, w, 3), (60, 75, 55), np.uint8)
    ts = timeseries(ep)
    log = ep.log
    com = np.asarray(log["body_pos"])[:, 0]
    pts = np.vstack([com[:, :2], [[0, 0]], [scn.field_centre]])
    lo, hi = pts.min(0) - 80, pts.max(0) + 80
    span = max(hi - lo)
    cen = 0.5 * (lo + hi)

    def px(xy):
        return (int(w / 2 + (xy[0] - cen[0]) / span * (w - 20)), int(h / 2 - (xy[1] - cen[1]) / span * (w - 20)))

    a = math.radians(scn.launch_heading_deg)
    u = np.array([math.cos(a), math.sin(a)]); n = np.array([-u[1], u[0]])
    c = scn.field_centre
    half_l, half_w = scn.leg_length / 2 + 20, (scn.legs * scn.spacing) / 2 + 20
    corners = [c + sl * half_l * u + sw * half_w * n for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    cv2.fillPoly(img, [np.array([px(q) for q in corners], np.int32)], (175, 160, 95))
    track = com[:i_now + 1, :2]
    for j in range(1, len(track), 2):
        cv2.line(img, px(track[j - 1]), px(track[j]), (255, 230, 120), 1, cv2.LINE_AA)
    cv2.circle(img, px((0, 0)), 4, (240, 240, 240), -1)
    cv2.putText(img, "home", (px((0, 0))[0] + 6, px((0, 0))[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.circle(img, px(com[i_now, :2]), 4, (80, 200, 255), -1)
    wv = np.asarray(scn.wind[:2], float)
    if np.linalg.norm(wv) > 0.3:
        o = (w - 40, 30); e = (int(o[0] + 18 * wv[0] / np.linalg.norm(wv)), int(o[1] - 18 * wv[1] / np.linalg.norm(wv)))
        cv2.arrowedLine(img, o, e, (255, 255, 255), 2, tipLength=0.4)
        cv2.putText(img, f"wind {np.linalg.norm(wv):.0f} m/s", (w - 95, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"map  {span:.0f} m across", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def render_movie(ep, scn: Scenario, path, *, speed: float = 25.0, fps: int = 20, size=(960, 540), progress=False, onboard=True):
    """The mission as an MP4: Chiron's renderer with the chase camera (a ground camera near home for the launch and
    the landing), the synthetic onboard view (top right, labelled), the map (bottom right), the overlays and the end
    card; ``speed`` x real time."""
    import cv2
    from vegeta.aeromant._watermark import watermark
    from vegeta.chiron import viz
    from pekari_controller import end_card

    log = ep.log
    t = np.asarray(log["t"])
    dt = float(t[1] - t[0])
    every = max(1, int(round(speed / (fps * dt))))
    t_end = (log.get("touchdown") or t[-1]) + 6.0
    stop = min(int(np.searchsorted(t, t_end)) + 1, len(t))
    idx = list(range(0, stop, every))
    chase, onboard_cams, ground = _cameras(ep, idx, scn)
    main_cams = [g if g is not None else c for c, g in zip(chase, ground)]
    it_main = iter(main_cams)
    extent = max(scn.radius_m + scn.leg_length, 400.0) + 200
    common = dict(every=every, stop=idx[-1] + 1, show_time=False, ground=(-extent, extent, -extent, extent), scenery_range=5000.0,
                  ground_color="#6f9a52", background="#bcd7f0")
    imgs = viz.frames(ep, camera=lambda com: next(it_main), size=size, **common)
    if onboard:
        it_on = iter(onboard_cams)
        ins_size = (size[0] // 3 // 2 * 2, size[1] // 3 // 2 * 2)
        small = viz.frames(ep, camera=lambda com: next(it_on), size=ins_size, **common)
    ts = timeseries(ep)
    ta = ts["t"].to_numpy()
    starts = [(tt, name) for tt, name, note in log["mission"] if note == "start"]
    events = sorted(log.get("events", []), key=lambda e: e[0])
    E = log["energy"]
    title = f"{scn.label}   ({speed:g}x speed; MuJoCo/Chiron, forces from the coefficient table)"
    out = []
    for k, i in enumerate(idx[:len(imgs)]):
        ti = t[i]
        j = min(int(np.searchsorted(ta, ti)), len(ts) - 1)
        row = ts.iloc[j]
        phase = next((name for tt, name in reversed(starts) if tt <= ti + 1e-9), "prelaunch")
        img = np.ascontiguousarray(imgs[k])
        n_z = row["n_z"]
        remaining = E["E_available_wh"] - row["E_used_Wh"]
        margin = getattr(ep, "margin_series", None)
        lines = [(f"t = {ti:6.1f} s   phase: {phase}", (255, 225, 120)),
                 (f"airspeed {row['V']:5.1f} m/s   height {row['h']:5.1f} m   ground speed {math.hypot(row['vx'], row['vy']):4.1f} m/s", (210, 230, 255)),
                 (f"bank {row['roll_deg']:+6.1f} deg   pitch {row['pitch_deg']:+5.1f} deg   load factor {n_z:4.2f}   alpha {row['alpha_deg']:+5.1f} deg", (210, 230, 255)),
                 (f"throttle {100 * row['throttle']:3.0f} %   thrust {row['thrust_N']:4.1f} N   power {row['P_el_W']:5.0f} W (electronics {row['P_electronics_W']:4.1f} W)", (210, 230, 255)),
                 (f"battery used {row['E_used_Wh']:5.2f} Wh   remaining {remaining:5.2f} Wh of {E['E_available_wh']:.2f} available   reserve {E['reserve_wh'] or 0:.2f} Wh", (180, 255, 180))]
        if phase in ("climb", "outbound", "survey") and np.isfinite(row.get("margin_wh", np.nan)):
            lines.append((f"return margin {row['margin_wh']:+5.2f} Wh (energy left above return + allowance + reserve)", (180, 255, 180)))
        elif phase in ("return", "approach", "landed"):
            lines.append((f"returning ({log.get('return_reason')}): {remaining:5.2f} Wh left, reserve {E['reserve_wh'] or 0:.2f} Wh", (180, 255, 180)))
        if row["stalled"] > 0.5:
            lines.append(("STALL", (255, 110, 110)))
        recent = [e for e in events if 0 <= ti - e[0] < 6.0 and e[1] != "energy" or (e[1] == "energy" and "return" in e[2] and 0 <= ti - e[0] < 8.0)]
        if recent:
            lines.append((f"> {recent[-1][1]}: {recent[-1][2][:95]}", (255, 200, 90)))
        img = _overlay(img, lines, title)
        if onboard:
            ins = np.ascontiguousarray(small[k])
            cv2.rectangle(ins, (0, 0), (ins.shape[1] - 1, 20), (25, 30, 40), -1)
            cv2.putText(ins, "onboard camera (SYNTHETIC render)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)
            top = int(36 * min(size[0] / 960.0, 1.0))
            img[top:top + ins.shape[0], size[0] - ins.shape[1] - 10:size[0] - 10] = ins
        mp = _map(ep, i, scn, size=(int(size[0] * 0.3), int(size[1] * 0.4)))
        img[size[1] - mp.shape[0] - 10:size[1] - 10, size[0] - mp.shape[1] - 10:size[0] - 10] = mp
        out.append(img)
    out = end_card(out, ep, fps=fps, hold_s=2.5)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = out[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    try:
        for img in out:
            vw.write(watermark(cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)))
    finally:
        vw.release()
    return path
