"""FALCO as a Chiron/MuJoCo robot with its aerodynamic hook.

NISUS+'s robot module gives the mountains (``Massif``), the wind (``MountainWind``), the servo and ESC lags
(``NisusPlusActuators``) and the force hook (``NisusPlusAero``: the derivative table with crow, the drive at the
pack's loaded voltage, the propeller brake with regeneration); this one builds FALCO's body (the tractor fuselage,
the wing, one tail tube, the stabiliser and the single fin, the keel skid and the tail bumper as feet, the masses from
``falco_systems.mass_table``) and adds the **parked propeller** to the hook:

* the controller's command gets a seventh channel ``park`` (0/1). With ``park`` set and the throttle at idle the ESC's
  active brake stops the blades (the drive's brake region while it slows: regeneration as NISUS+'s; the model has no
  rotor inertia, so the stop and the parking nudge on the Hall sensor's mark together take ``park_time`` seconds —
  assumed) and ``parked`` becomes True: the propeller gives no thrust and the drag of a stopped blade pair
  (``locked_drag``), the ESC holds it with ``hold_w`` (assumed), the rpm reads 0. A throttle above idle with ``park``
  cleared un-parks it (the go-around: the motor spins up through the actuator's lag).
* the log gets the column ``parked``.

Units SI; the Chiron body frame is x forward, y left, z up.
"""
from __future__ import annotations

import math

import numpy as np

from vegeta.chiron import FootSpec, Geom, Link, PointMass, Robot

import falco
import falco_systems as fsy
import nisus_plus_robot as fr
from nisus_plus_robot import Massif, MountainWind, NisusPlusActuators, lab_options, to_body, quat_to_R, euler_from_R  # noqa: F401

G = 9.81
BODY = "falco"


def _group_x(table, prefix) -> float:
    w = table[table.index.str.startswith(prefix)]
    return float((w["mass [g]"] * w["x [mm]"]).sum() / w["mass [g]"].sum())


def falco_robot(battery_key: str = fsy.DEFAULT_PACK, p=None, *, table=None) -> Robot:
    """The aircraft as a Chiron ``Robot`` (root body ``falco``): the fuselage as an ellipsoid on its axis, the wing as a
    box, the tail tube as a capsule, the stabiliser and the fin as plates, the motor and the propeller disc for the
    eye, the keel skid and the tail bumper as feet; the mass table's other rows as point masses."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    t = fsy.mass_table(battery_key, p=p) if table is None else table
    pod_rgba, foam, carbon, dark = (0.86, 0.87, 0.9, 1.0), (0.95, 0.66, 0.2, 1.0), (0.12, 0.12, 0.14, 1.0), (0.1, 0.1, 0.1, 1.0)
    groups = {"wing": 0.0, "pod": 0.0, "tube": 0.0, "tail": 0.0}
    masses = []
    for item, row in t.iterrows():
        mi = float(row["mass [g]"]) / 1000.0
        if item.startswith("wing:"):
            groups["wing"] += mi
        elif item.startswith("fuselage:") and "shell" in item:
            groups["pod"] += mi
        elif item.startswith("tail: roll-wrapped"):
            groups["tube"] += mi
        elif item.startswith("tail: foam") or item.startswith("tail: covering") or item.startswith("tail: stabiliser spar") or item.startswith("tail: fin rod"):
            groups["tail"] += mi
        else:
            masses.append(PointMass(item[:60].replace(" ", "_").replace("/", "-"), mi, to_body(row["x [mm]"], row["y [mm]"], row["z [mm]"])))
    c0, b = p["root_chord"], p["span"]
    zt = p["boom_z"]
    x_pod = 0.5 * (L["x_nose"] + L["x_pod_end"])
    quat_y = (math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0)
    geoms = [
        Geom("fuselage", "ellipsoid", ((L["x_pod_end"] - L["x_nose"]) / 2000, p["pod_width"] / 2000, p["pod_height"] / 2000), pos=to_body(x_pod, 0, L["pod_axis_z"]),
             mass=groups["pod"], role="body", rgba=pod_rgba),
        Geom("wing", "box", (0.5 * (c0 + p["tip_chord"]) / 2000, b / 2000, 0.016), pos=to_body(_group_x(t, "wing:"), 0, 10.0), mass=groups["wing"], role="body", rgba=foam),
        Geom("tail_tube", "capsule", (p["tail_tube_od"] / 2000,), fromto=(*to_body(L["x_tube0"], 0, zt), *to_body(p["tail_x_end"], 0, zt)),
             mass=groups["tube"], role="body", rgba=carbon),
        Geom("stab", "box", (p["tail_chord"] / 2000, p["tail_span"] / 2000, 0.005), pos=to_body(L["tail_le"] + p["tail_chord"] / 2, 0, L["tail_z"]),
             mass=0.55 * groups["tail"], role="body", rgba=foam),
        Geom("fin", "box", (p["fin_chord"] / 2000, 0.005, p["fin_height"] / 2000), pos=to_body(L["fin_le"] + p["fin_chord"] / 2, 0, L["tail_z"] + p["fin_height"] / 2),
             mass=0.45 * groups["tail"], role="body", rgba=foam),
        Geom("motor", "cylinder", (p["motor_diameter"] / 2000, p["motor_length"] / 2000), pos=to_body(0.5 * (L["x_motor0"] + L["x_motor1"]), 0, L["motor_z"]),
             quat=quat_y, mass=None, role="visual", rgba=dark),
        Geom("spinner", "sphere", (p["hub_diameter"] / 2000,), pos=to_body(L["x_nose"] + p["spinner_length"] * 0.6, 0, L["motor_z"]), mass=None, role="visual", rgba=pod_rgba),
        Geom("prop_disc", "cylinder", (L["prop_R"] / 1000, 0.0008), pos=to_body(L["prop_x"], 0, L["motor_z"]), quat=quat_y, mass=None, role="visual",
             rgba=(0.6, 0.75, 0.95, 0.30)),
        Geom("skid", "box", ((p["skid_x1"] - p["skid_x0"]) / 2000, p["skid_width"] / 2000, p["skid_depth"] / 2000),
             pos=to_body(0.5 * (p["skid_x0"] + p["skid_x1"]), 0, -p["pod_height"] - p["skid_depth"] / 2), mass=None, role="foot",
             friction=(0.5, 0.005, 0.0001), rgba=(0.9, 0.4, 0.1, 1.0)),
        Geom("tail_bumper", "capsule", (0.006,), fromto=(*to_body(p["tail_x_end"] - 45.0, 0, L["tail_bumper_z"] + 6.0), *to_body(p["tail_x_end"] - 5.0, 0, L["tail_bumper_z"] + 6.0)),
             mass=None, role="foot", friction=(0.5, 0.005, 0.0001), rgba=dark),
        Geom("nose_tip", "sphere", (0.025,), pos=to_body(L["x_nose"] + 40, 0, L["pod_axis_z"]), mass=None, role="body", rgba=pod_rgba),
    ]
    feet = [FootSpec("skid", "skid", [], BODY), FootSpec("tail_bumper", "tail_bumper", [], BODY)]
    root = Link(BODY, geoms=geoms, masses=masses, log=True)
    robot = Robot("Falco-Zero", root=root, feet=feet,
                  notes=f"Falco-Zero on the {battery_key} pack: falco_systems.mass_table as geom and point masses; no joints (the control surfaces, the crow "
                        "flaps and the propeller's parking are states of the FalcoAero hook)",
                  sources={"geometry": "designs/falco.py", "masses": "designs/falco_systems.py: mass_table", "aerodynamics": "designs/falco_flight.py: derivatives",
                           "propulsion": "designs/falco_systems.py: drive (Boreas BEMT on the 20x15, the fitted AT5220-A KV220 on 8S3P)"})
    robot.validate()
    robot.params, robot.battery_key, robot.mass_table, robot.layout = p, battery_key, t, L
    return robot


def inertia_check(lab, table) -> dict:
    """MuJoCo's mass, COM and inertia against ``falco_systems.cg_inertia`` of the same table."""
    m = lab.model
    b = lab._body_id(BODY)
    mass = float(m.body_subtreemass[b])
    hand = fsy.cg_inertia(table)
    R = quat_to_R(m.body_iquat[b])
    I = R @ np.diag(m.body_inertia[b]) @ R.T
    com_body = m.body_ipos[b]
    return {"mujoco_mass_kg": mass, "table_mass_kg": hand["mass_kg"], "mass_error": mass / hand["mass_kg"] - 1,
            "mujoco_cg_x_mm": -com_body[0] * 1000, "table_cg_x_mm": hand["x_cg_m"] * 1000, "mujoco_cg_z_mm": com_body[2] * 1000, "table_cg_z_mm": hand["z_cg_m"] * 1000,
            "Ixx_ratio": float(I[0, 0]) / hand["Ixx"], "Iyy_ratio": float(I[1, 1]) / hand["Iyy"], "Izz_ratio": float(I[2, 2]) / hand["Izz"]}


class FalcoAero(fr.NisusPlusAero):
    """NISUS+'s hook with the parked propeller (see the module). ``command``: aileron, elevator, rudder [rad], throttle,
    crow, brake, park."""

    COLUMNS = fr.NisusPlusAero.COLUMNS + ["parked"]

    def __init__(self, robot: Robot, coeff: dict, drive, pack, massif, *, park_time: float = 1.0, hold_w: float = 2.0, **kw):
        p0 = robot.params
        robot.params = dict(p0, motor_z=robot.layout["motor_z"])           # NISUS's hook reads the thrust line's height from the parameters
        try:
            super().__init__(robot, coeff, drive, pack, massif, **kw)
        finally:
            robot.params = p0
        self.motor_pos = np.array(to_body(robot.layout["prop_x"], 0.0, robot.layout["motor_z"]))
        self.park_time, self.hold_w = park_time, hold_w          # the stop + the nudge [s]; the ESC's hold power [W] (assumed)

    def reset(self, lab):
        super().reset(lab)
        self.command = np.zeros(7)
        self.parked = False
        self.t_stopped = None
        self.last_rpm = 0.0
        self.park_events = []

    def _propulsor(self, u, thr, brake, rho):
        cmd = self.command
        park = float(cmd[6]) > 0.5 if len(cmd) > 6 else False
        V = max(u, 0.0)
        t = self.lab.time
        hold = {"thrust": 0.0, "electrical": self.hold_w, "rpm": 0.0, "current": 0.0, "voltage": 0.0, "shaft_power": 0.0, "mode": "parked"}
        if self.parked:
            if thr > self.idle and not park:
                self.parked, self.t_stopped = False, None
                self.park_events.append([t, "un-parked: the motor spins up"])
            else:
                return dict(hold, thrust=-self.drive.locked_drag(V, rho))
        if park and thr <= self.idle:
            if self.t_stopped is None:
                self.t_stopped = t
                self.park_events.append([t, f"the ESC's active brake stops the propeller at {V:.1f} m/s; the parking nudge follows"])
            if t - self.t_stopped >= self.park_time:
                self.parked = True
                self.park_events.append([t, "propeller parked horizontal (the Hall sensor's mark)"])
                return dict(hold, thrust=-self.drive.locked_drag(V, rho))
            r = self.drive.brake(V, 1.0, rho, self.v_batt, i_charge_max=self.pack.charge_limit_a(self.pack_T_C, self.soc))
            return dict(r, mode="parking")
        self.t_stopped = None
        return super()._propulsor(u, thr, brake, rho)

    def __call__(self, lab):
        n0 = len(self.history)
        super().__call__(lab)
        if len(self.history) > n0:
            self.history[-1].append(float(self.parked))
        self.last["parked"] = self.parked


__all__ = ["G", "BODY", "Massif", "MountainWind", "NisusPlusActuators", "lab_options", "falco_robot", "inertia_check", "FalcoAero", "to_body", "quat_to_R", "euler_from_R"]
