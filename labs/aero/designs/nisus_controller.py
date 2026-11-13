"""NISUS flight controller and missions in ChironLab (notebook 31).

**The flight controller** (``FlightController``) stands in for the Kakute F405-Wing Mini running ArduPilot Plane.
It is a simplified version of that stack, with the same structure:

* *attitude loops*: aileron = Kφ (φ_cmd − φ) − Kp p; elevator = −Kθ (θ_cmd − θ) + Kq q + trim; rudder = a yaw damper
  with the coordinated-turn feed-forward r = g sin φ cos θ / V plus an aileron–rudder mix (gains hand-tuned on the
  linear model; ``Gains``);
* *speed and height*: pitch holds the airspeed (θ_cmd from the speed error, with a slow integrator), throttle holds
  the height (and is saturated at full in the climb, idle in the final approach) — the classic scheme a small stable
  aeroplane flies well with, in place of ArduPilot's TECS;
* *navigation*: a line-following law on the leg to the active waypoint (course of the leg plus a cross-track
  correction, bank limited); the ground track comes from the velocity, so the aircraft crabs into the wind;
* *the energy manager*: every step it predicts the energy to come home from where it is (distance, the wind along
  the way, the cruise power, the electronics), adds the uncertainty allowance and the final reserve, and compares
  with what the battery has left; when the margin reaches zero in the survey it orders the return (``events``);
* *the phases* (``PHASES``): prelaunch on the ground (the electronics run; the ground time's energy before the
  simulated seconds is charged in one go), the hand launch (the impulse is a scheduled Chiron disturbance, the
  controller holds wings level and 8° of pitch at full throttle), the climb, the outbound leg, the survey, the
  return, the approach (an initial approach fix upwind of home, a 5° final, a flare), landed.

**Nisus-OBS** is flown by a **scripted pilot** (``ScriptedPilot``): a reproducible rule set — CRUISE (track hold) on
the straights, hand-flown 180° turns at a set bank at the field's edges as the OSD shows them — through the same
stabilisation loops (ArduPilot's FBWA/CRUISE manner); the flight controller still owns the launch, the energy
manager's return (its battery failsafe) and the landing.
**Nisus-Zero** flies an autonomous survey whose waypoints the mission computer streams (``AutoMission``); when the
Jetson fails (``jetson_failure``: its power branch drops, it reboots later) the waypoint stream stops and the flight
controller, which never depended on it, returns home on its own after a lost-companion timeout.

The controller knows only what the real one would: position, velocity, attitude and rates (ideal sensors here),
the airspeed (ideal pitot), the battery energy used (the current sensor on the main lead integrated), the wind only
through the ground speed it measures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta.chiron import Command

import nisus_robot as nr
import nisus_systems as ns

G = 9.81
PHASES = ("prelaunch", "launch", "climb", "outbound", "survey", "return", "approach", "landed")

__all__ = ["G", "PHASES", "Gains", "Waypoint", "AutoMission", "ScriptedPilot", "EnergyManager", "FlightController", "survey_pattern", "wrap"]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Gains:
    k_roll: float = 1.2           # aileron per rad of roll error
    k_p: float = 0.08             # aileron per rad/s of roll rate
    k_pitch: float = 1.8          # elevator per rad of pitch error
    k_q: float = 0.15             # elevator per rad/s of pitch rate
    k_yaw: float = 0.25           # rudder per rad/s of yaw-rate error
    k_ar: float = 0.15            # rudder per aileron (adverse-yaw mix)
    k_v: float = 0.035            # pitch [rad] per m/s of airspeed error (slow → nose down)
    k_vi: float = 0.004           # its integrator
    k_h: float = 0.035            # throttle per m of height error
    k_hd: float = 0.045           # throttle per m/s of climb rate
    k_track: float = 1.6          # roll [rad] per rad of course error
    L1: float = 35.0              # cross-track look-ahead [m]
    bank_max_deg: float = 40.0
    pitch_min_deg: float = -15.0
    pitch_max_deg: float = 25.0


@dataclass
class Waypoint:
    x: float
    y: float
    h: float
    speed: float = 15.0
    name: str = ""


def survey_pattern(centre, heading_deg, legs=3, leg_length=300.0, spacing=60.0, h=80.0, speed=15.0) -> list:
    """A lawnmower pattern of ``legs`` parallel legs of ``leg_length`` along ``heading_deg`` (from +x towards +y),
    ``spacing`` apart, centred on ``centre``: the waypoints in flying order."""
    c = np.asarray(centre, float)
    u = np.array([math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))])
    n = np.array([-u[1], u[0]])
    wps = []
    for i in range(legs):
        off = (i - (legs - 1) / 2) * spacing
        a, b = c + off * n - u * leg_length / 2, c + off * n + u * leg_length / 2
        pair = (a, b) if i % 2 == 0 else (b, a)
        wps += [Waypoint(*pair[0], h, speed, f"leg {i + 1} start"), Waypoint(*pair[1], h, speed, f"leg {i + 1} end")]
    return wps


class AutoMission:
    """The mission computer's waypoint stream: the survey waypoints in order, the pattern repeated ``repeats`` times
    (an open-ended survey: the flight controller's energy manager, not the waypoint count, ends it); ``alive`` False
    stops the stream (the Jetson failure case)."""

    def __init__(self, waypoints: list, repeats: int = 6):
        base = list(waypoints)
        self.waypoints = []
        for k in range(repeats):
            self.waypoints += base if k % 2 == 0 else base[::-1]
        self.i = 0
        self.alive = True

    def current(self):
        if not self.alive or self.i >= len(self.waypoints):
            return None
        return self.waypoints[self.i]

    def advance(self):
        self.i += 1

    @property
    def done(self):
        return self.i >= len(self.waypoints)


class ScriptedPilot:
    """OBS's pilot: a fixed, reproducible rule set flown through the flight controller's stabilised modes, as a
    pilot does it with the video and the on-screen display (OSD): on a straight the pilot selects CRUISE (the flight
    controller holds the ground track ``heading`` the pilot set, the height and the airspeed); at the field's edge
    (the OSD's position crosses the end of the lane) the pilot rolls into a hand-flown 180° turn at ``turn_bank_deg``
    (FBWA: the stick sets the bank) and rolls out on the reciprocal heading. The turns alternate, so the lanes step
    across the field by the turn's diameter; at the last lane the pilot turns the same way twice and works back. The
    survey ends when the flight controller's energy manager calls the return (ArduPilot's battery failsafe set from
    the predicted return energy) — the pilot does not decide it."""

    def __init__(self, centre, heading_deg, *, lanes=3, lane_length=300.0, turn_bank_deg=35.0, speed=15.0, h=80.0):
        self.centre = np.asarray(centre, float)
        self.heading = math.radians(heading_deg)
        self.u = np.array([math.cos(self.heading), math.sin(self.heading)])
        self.lanes, self.half, self.bank, self.speed, self.h = lanes, lane_length / 2, math.radians(turn_bank_deg), speed, h
        self.reset()

    def reset(self):
        self.state, self.dir_sign, self.lane, self.step, self.turn_sign, self.turns = "leg", 1, 0, 1, 1, 0
        self.log = []

    def command(self, t, pos, track):
        """(mode, value): ('track', heading [rad]) on a straight, ('bank', bank [rad]) in a turn."""
        along = float((pos[:2] - self.centre) @ self.u)
        leg_heading = self.heading if self.dir_sign > 0 else self.heading + math.pi
        if self.state == "leg":
            if self.dir_sign * along > self.half:
                self.state = "turn"
                self.target = wrap(leg_heading + math.pi)
                self.turns += 1
                self.log.append([t, f"pilot: field edge, {('right' if self.turn_sign > 0 else 'left')} turn to {math.degrees(self.target) % 360:.0f}°"])
                return "bank", self.turn_sign * self.bank
            return "track", leg_heading
        # in the turn: roll out within 12° of the reciprocal
        if abs(wrap(track - self.target)) < math.radians(12.0):
            self.state = "leg"
            self.dir_sign = -self.dir_sign
            self.lane += self.step
            if self.lane >= self.lanes - 1 or self.lane <= 0:
                self.step = -self.step                     # the last lane: the next turn goes the same way, back across
            else:
                self.turn_sign = -self.turn_sign
            return "track", self.target
        return "bank", self.turn_sign * self.bank


class EnergyManager:
    """The return decision from the route actually flown home: from where the aircraft is to the initial approach
    fix (``iaf``: downwind of home) at the cruise airspeed with the wind component along that leg, then the final
    from the fix to the touchdown point into the wind at the approach speed (60 % of its level power: the descent
    from the survey height pays the rest), one go-around circuit per ``plan.go_arounds``, the electronics all along;
    times (1 + uncertainty), plus the final reserve
    (``reserve_frac`` of the available energy). ``margin`` = what is left − needed."""

    def __init__(self, aero: nr.Aero, pm: ns.PropulsionMap, mass_kg, cd0, AR, oswald, S, plan: ns.MissionPlan, electronics_w: float,
                 *, V_app: float = 10.5):
        self.aero, self.pm, self.plan, self.el = aero, pm, plan, electronics_w
        self.P_cruise, _ = ns._level_power(pm, mass_kg, cd0, AR, oswald, S, plan.V_cruise)
        P_app, _ = ns._level_power(pm, mass_kg, cd0, AR, oswald, S, V_app)
        self.P_app, self.V_app = 0.6 * P_app, V_app
        self.E_reserve = plan.reserve_frac * aero.E_available_Wh

    def needed(self, pos_xy, wind_xy, iaf_xy, td_xy) -> dict:
        pos, w, iaf, td = (np.asarray(v, float) for v in (pos_xy, wind_xy, iaf_xy, td_xy))
        leg = iaf - pos
        d1 = float(np.linalg.norm(leg))
        u1 = leg / d1 if d1 > 1 else np.zeros(2)
        gs1 = max(self.plan.V_cruise + float(w @ u1), 3.0)          # wind along the leg (a tailwind > 0)
        final = td - iaf
        d2 = float(np.linalg.norm(final))
        u2 = final / d2 if d2 > 1 else np.zeros(2)
        gs2 = max(self.V_app + float(w @ u2), 2.0)
        t1, t2 = d1 / gs1, d2 / gs2 + 15.0                            # + 15 s for the turn onto the final and the flare
        E_back = (self.P_cruise + self.el) * t1 / 3600
        E_land = (self.P_app + self.el) * t2 / 3600
        E_ga = self.plan.go_arounds * (self.P_cruise + self.el) * self.plan.circuit_m / self.plan.V_cruise / 3600
        E = (E_back + E_land + E_ga) * (1 + self.plan.uncertainty_frac)
        return {"t_back_s": t1, "t_final_s": t2, "E_return_wh": E_back + E_land + E_ga, "E_go_around_wh": E_ga, "E_with_uncertainty_wh": E, "E_reserve_wh": self.E_reserve,
                "E_needed_wh": E + self.E_reserve, "E_remaining_wh": self.aero.E_remaining_Wh, "margin_wh": self.aero.E_remaining_Wh - E - self.E_reserve,
                "route_m": d1 + d2}


class FlightController:
    """The flight controller with its mission (see the module). ``mode``: 'auto' (Zero: the computer's waypoints) or
    'pilot' (OBS: the scripted pilot). ``jetson_failure``: (t_fail, t_reboot) — the computer's power branch drops at
    t_fail and its stream stops; power returns at t_reboot (the stream does not: the return is already ordered)."""

    def __init__(self, aero: nr.Aero, plan: ns.MissionPlan, *, home=(0.0, 0.0), launch_heading_deg=0.0, mode="auto", mission: AutoMission | None = None,
                 pilot: ScriptedPilot | None = None, survey_wps: list | None = None, energy: EnergyManager | None = None,
                 electronics_phase_w: dict | None = None, jetson_w: float = 0.0, jetson_failure: tuple | None = None,
                 wind_estimate=(0.0, 0.0), prelaunch_sim_s: float = 4.0, ground_s: float = 180.0, gains: Gains = Gains(),
                 companion_timeout_s: float = 5.0, airborne_cap_s: float = 1500.0, companion=None, name: str = "Nisus flight controller"):
        self.aero, self.plan, self.g = aero, plan, gains
        self.home = np.asarray(home, float)
        self.launch_heading = math.radians(launch_heading_deg)
        self.mode, self.mission, self.pilot, self.energy = mode, mission, pilot, energy
        self.companion = companion            # mode 'birds': the companion computer's mission (``command()`` → GuidanceCmd or None)
        self.survey_wps = survey_wps or []
        self.el_phase = electronics_phase_w or {}
        self.jetson_w, self.jetson_failure = jetson_w, jetson_failure
        self.wind_est = np.asarray(wind_estimate, float)
        self.prelaunch_sim_s, self.ground_s = prelaunch_sim_s, ground_s
        self.companion_timeout_s = companion_timeout_s
        self.airborne_cap_s = airborne_cap_s
        self.name = name
        self.V_app = 1.3 * 8.2

    # ---- lifecycle
    def reset(self, lab, seed=None):
        self.lab = lab
        self.phase = "prelaunch"
        self.t_phase = 0.0
        self.t_launch = self.prelaunch_sim_s
        self.log = []
        self.events = []
        self.vi = 0.0
        self.wp_i = 0
        self.return_ordered = False
        self.return_reason = ""
        self.jetson_alive = True
        self.companion_lost_since = None
        self.touchdown = None
        self.target = None
        self.margin = None
        self.last_cmd = np.zeros(4)
        self.pilot_t0 = None
        self.v_mean, self.v_var = 15.0, 0.0
        if self.pilot is not None:
            self.pilot.reset()
        self._set_phase("prelaunch", 0.0)
        # the ground operation's energy before the simulated seconds
        extra = max(self.ground_s - self.prelaunch_sim_s, 0.0) * self.el_phase.get("prelaunch", 0.0) / 3600
        self.aero.add_energy_wh(extra)
        self.events.append([0.0, "energy", f"ground operation {self.ground_s:.0f} s at {self.el_phase.get('prelaunch', 0.0):.1f} W charged: {extra:.3f} Wh"])

    def _set_phase(self, name, t):
        self.phase, self.t_phase = name, t
        self.log.append([t, name, "start"])
        self.aero.electronics_w = self._electronics(name)

    def _electronics(self, phase):
        key = {"prelaunch": "prelaunch", "launch": "launch+climb", "climb": "launch+climb", "outbound": "outbound", "survey": "survey",
               "return": "return", "approach": "approach+landing", "landed": "approach+landing"}[phase]
        w = self.el_phase.get(key, 0.0)
        if not self.jetson_alive:
            w -= self.jetson_w
        return w

    # ---- helpers
    def _state(self):
        a = self.aero.last
        pos, v = a["pos"], a["v"]
        return pos, v, a["V"], a["roll"], a["pitch"], a["yaw"], a["omega_b"]

    def _track_to(self, pos, v, A, B):
        """Bank command to fly the line A→B (world xy), with the cross-track correction, from the measured ground track."""
        d = B - A
        Ld = np.linalg.norm(d)
        u = d / Ld if Ld > 1e-6 else np.array([1.0, 0.0])
        rel = pos[:2] - A
        along = float(rel @ u)
        e = float(rel[0] * u[1] - rel[1] * u[0])                 # cross-track, positive to the right of the leg
        course_leg = math.atan2(u[1], u[0])
        corr = math.atan(e / self.g.L1)                            # right of the leg → steer left (a course further counter-clockwise)
        course_cmd = course_leg + corr * 0.9
        gs = np.linalg.norm(v[:2])
        track = math.atan2(v[1], v[0]) if gs > 2.0 else course_cmd
        err = wrap(course_cmd - track)
        bank = -max(-1, min(1, self.g.k_track * err)) * math.radians(self.g.bank_max_deg)     # a left turn is a negative (left-wing-down) bank
        return bank, along, Ld, e

    def _attitude(self, roll_cmd, pitch_cmd, roll, pitch, omega_b, V, throttle):
        g = self.g
        p_, q_, r_ = omega_b[0], -omega_b[1], -omega_b[2]
        pitch_cmd = max(math.radians(g.pitch_min_deg), min(math.radians(g.pitch_max_deg), pitch_cmd))
        da = g.k_roll * (roll_cmd - roll) - g.k_p * p_
        de = -(g.k_pitch * (pitch_cmd - pitch)) + g.k_q * q_ + math.radians(2.0)
        r_coord = G * math.sin(roll) * math.cos(pitch) / max(V, 6.0)
        dr = -g.k_yaw * (r_coord - r_) - g.k_ar * da
        self.aero.command = np.array([da, de, dr, throttle])
        self.last_cmd = self.aero.command.copy()

    def _speed_height(self, V_cmd, h_cmd, V, h, hdot, dt, *, throttle_override=None, climb_full=False):
        g = self.g
        ev = V_cmd - V
        self.vi = max(-0.15, min(0.15, self.vi + g.k_vi * ev * dt))
        theta_trim = math.radians(0.0) + 0.06 * (12.0 / max(V, 8.0) - 0.75)          # a little nose-up at low speed (trim curve, hand fit)
        pitch_cmd = theta_trim - g.k_v * ev - self.vi
        if throttle_override is not None:
            thr = throttle_override
        elif climb_full:
            thr = 1.0
        else:
            thr_ff = 0.42 + 0.03 * (V_cmd - 15.0)
            thr = thr_ff + g.k_h * (h_cmd - h) - g.k_hd * hdot
        return pitch_cmd, max(0.0, min(1.0, thr))

    # ---- the loop
    def __call__(self, obs):
        t = float(obs.t)
        if not self.aero.last:
            return Command(q_target={})
        pos, v, V, roll, pitch, yaw, omega_b = self._state()
        h, hdot = pos[2], v[2]
        dt = self.lab.control_dt
        if self.phase in ("outbound", "survey", "return"):                    # the airspeed's fluctuation (a 20 s window): the gust additive
            k = dt / 20.0
            self.v_mean += k * (V - self.v_mean)
            self.v_var += k * ((V - self.v_mean) ** 2 - self.v_var)
        home_d = float(np.linalg.norm(pos[:2] - self.home))
        # the computer's failure
        if self.jetson_failure is not None:
            t_fail, t_reboot = self.jetson_failure
            if self.jetson_alive and t >= t_fail:
                self.jetson_alive = False
                if self.mission is not None:
                    self.mission.alive = False
                self.events.append([t, "jetson", "power branch lost: computer off, waypoint stream stopped"])
                self.aero.electronics_w = self._electronics(self.phase)
            if (not self.jetson_alive) and t >= t_reboot and not getattr(self, "_rebooted", False):
                self._rebooted = True
                self.jetson_alive = True
                self.events.append([t, "jetson", "computer rebooted (power back); the mission stays in return"])
                self.aero.electronics_w = self._electronics(self.phase)
        # the energy manager (in flight, before the return)
        if self.energy is not None and self.phase in ("climb", "outbound", "survey"):
            dist = float(np.linalg.norm(self.home - pos[:2]))
            need = self.energy.needed(pos[:2], self.wind_est, self._iaf(), self._touchdown_point())
            self.margin = need["margin_wh"]
            self.aero.margin_wh = self.margin
            if self.margin <= 0.0 and self.phase in ("outbound", "survey") and not self.return_ordered:
                self.return_ordered, self.return_reason = True, "energy"
                self.events.append([t, "energy", f"return ordered: remaining {need['E_remaining_wh']:.2f} Wh ≤ needed {need['E_needed_wh']:.2f} Wh "
                                                 f"(route {need['route_m']:.0f} m: {need['E_return_wh']:.2f} Wh + {100 * self.plan.uncertainty_frac:.0f} % + reserve "
                                                 f"{need['E_reserve_wh']:.2f}) at {dist:.0f} m from home"])
                self._set_phase("return", t)
        ph = self.phase
        if ph in ("outbound", "survey") and not self.return_ordered and t - self.t_launch > self.airborne_cap_s:
            self.return_ordered, self.return_reason = True, "time cap"
            self.events.append([t, "mission", f"airborne {self.airborne_cap_s:.0f} s: return"])
            self._set_phase("return", t)
            ph = self.phase
        if ph == "prelaunch":
            self.aero.command = np.array([0.0, 0.0, 0.0, 0.0])
            if t >= self.t_launch:
                self.aero.hold = False                                   # the thrower lets go: the impulse starts now
                self._set_phase("launch", t)
                self.events.append([t, "launch", "hand launch: the thrower lets go, the impulse starts"])
        elif ph == "launch":
            self._attitude(0.0, math.radians(8.0), roll, pitch, omega_b, V, 1.0)
            if (V > 11.0 and h > 3.0) or t - self.t_phase > 3.0:
                self._set_phase("climb", t)
        elif ph == "climb":
            tgt = self._first_target()
            bank, *_ = self._track_to(pos, v, self.home, np.array([tgt.x, tgt.y]))
            pc, thr = self._speed_height(self.plan.V_climb, self.plan.climb_alt_m, V, h, hdot, dt, climb_full=h < self.plan.climb_alt_m - 10)
            self._attitude(bank if h > 15 else 0.0, pc, roll, pitch, omega_b, V, thr)
            if h >= self.plan.climb_alt_m - 5:
                self._set_phase("outbound", t)
        elif ph == "outbound":
            tgt = self._first_target()
            bank, along, Ld, e = self._track_to(pos, v, self.home, np.array([tgt.x, tgt.y]))
            pc, thr = self._speed_height(self.plan.V_cruise, tgt.h, V, h, hdot, dt)
            self._attitude(bank, pc, roll, pitch, omega_b, V, thr)
            if along >= Ld - 25.0:
                self._set_phase("survey", t)
                self.pilot_t0 = t
                if self.mission is not None:
                    self.mission.advance()
        elif ph == "survey":
            if self.mode == "birds":
                self._birds(t, pos, v, V, h, hdot, roll, pitch, yaw, omega_b, dt)
            elif self.mode == "auto":
                wp = self.mission.current() if self.mission is not None else None
                if wp is None:
                    if self.mission is not None and not self.mission.alive:
                        if self.companion_lost_since is None:
                            self.companion_lost_since = t
                        if t - self.companion_lost_since >= self.companion_timeout_s:
                            self.events.append([t, "failsafe", f"no waypoint from the computer for {self.companion_timeout_s:.0f} s: return"])
                            self.return_ordered, self.return_reason = True, "companion lost"
                            self._set_phase("return", t)
                        else:
                            self._hold(pos, v, V, h, hdot, roll, pitch, omega_b, dt)
                    else:
                        self.events.append([t, "mission", "survey complete: return"])
                        self.return_ordered, self.return_reason = True, "survey complete"
                        self._set_phase("return", t)
                else:
                    prev = self.mission.waypoints[self.mission.i - 1] if self.mission.i > 0 else None
                    A = np.array([prev.x, prev.y]) if prev is not None else pos[:2]
                    bank, along, Ld, e = self._track_to(pos, v, A, np.array([wp.x, wp.y]))
                    pc, thr = self._speed_height(wp.speed, wp.h, V, h, hdot, dt)
                    self._attitude(bank, pc, roll, pitch, omega_b, V, thr)
                    if along >= Ld - 20.0:
                        self.mission.advance()
            else:
                track = math.atan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 2.0 else yaw
                n_log = len(self.pilot.log)
                mode, val = self.pilot.command(t, pos, track)
                for entry in self.pilot.log[n_log:]:
                    self.events.append([entry[0], "pilot", entry[1]])
                pc, thr = self._speed_height(self.pilot.speed, self.pilot.h, V, h, hdot, dt)
                if mode == "track":
                    err = wrap(val - track)
                    bank = -max(-1, min(1, self.g.k_track * err)) * math.radians(self.g.bank_max_deg)
                else:
                    bank = val
                self._attitude(bank, pc, roll, pitch, omega_b, V, thr)
        elif ph == "return":
            iaf = self._iaf()
            bank, along, Ld, e = self._track_to(pos, v, pos[:2] if self.target is None else self.target, iaf)
            if self.target is None:
                self.target = pos[:2].copy()
            pc, thr = self._speed_height(self.plan.V_cruise, 45.0, V, h, hdot, dt)
            self._attitude(bank, pc, roll, pitch, omega_b, V, thr)
            if float(np.linalg.norm(pos[:2] - iaf)) < 40.0:
                self._set_phase("approach", t)
        elif ph == "approach":
            iaf, td = self._iaf(), self._touchdown_point()
            bank, along, Ld, e = self._track_to(pos, v, iaf, td)
            dist_td = max(Ld - along, 0.0)
            h_slope = dist_td * math.tan(math.radians(4.5))
            if along > Ld - 15.0 and h > 10.0:                          # at the aim point and still high (an updraft, a gust): go around
                self.go_arounds = getattr(self, "go_arounds", 0) + 1
                self.events.append([t, "landing", f"go-around {self.go_arounds}: {h:.0f} m high over the aim point"])
                self.target = None
                self._set_phase("return", t)
                return Command(q_target={})
            # the gust additive: twice the airspeed fluctuation the flight controller has measured (at most 3 m/s); and well
            # above the slope at idle, trade the height for speed (descend faster), up to the cruise speed
            V_gust = min(2.0 * math.sqrt(max(self.v_var, 0.0)), 3.0)
            V_tgt = self.V_app + V_gust + float(np.clip(0.4 * (h - min(h_slope, 50.0) - 3.0), 0.0, self.plan.V_cruise - self.V_app))
            if h > 4.0:
                pc, thr = self._speed_height(V_tgt, min(h_slope, 50.0), V, h, hdot, dt)
                thr = min(thr, 0.7)                                     # a headwind final needs nearly level-flight power
                self._attitude(bank if h > 8.0 else 0.4 * bank, pc, roll, pitch, omega_b, V, thr)
            else:
                # the flare: hold a sink rate that shrinks with the height (0.25 h, at least 0.3 m/s) with the pitch, and
                # catch a gust's extra sink with a little power until the last half metre; wings level
                sink_tgt = max(0.25 * h, 0.3)
                err = (-hdot) - sink_tgt                                 # > 0: sinking too fast
                pc = math.radians(2.0) + float(np.clip(0.12 * err, -0.05, math.radians(3.0)))   # at most 5° nose up: no stall in the flare
                thr = float(np.clip(0.5 * err, 0.0, 0.6)) if h > 0.5 else 0.0             # the motor catches a gust's extra sink
                self._attitude(0.0, pc, roll, pitch, omega_b, V, thr)
            if self.touchdown is None and h < 0.25 and float(np.linalg.norm(v[:2])) < 9.5 and t - self.t_phase > 5.0:
                self.touchdown = t
                self.events.append([t, "landing", f"touchdown at {np.linalg.norm(v[:2]):.1f} m/s ground speed, sink {-v[2]:.2f} m/s, {home_d:.0f} m from home"])
            if self.touchdown is not None and float(np.linalg.norm(v)) < 1.0:
                self._set_phase("landed", t)
                self.aero.command = np.zeros(4)
        else:
            self.aero.command = np.zeros(4)
            if t - self.t_phase > 10.0 and self.aero.electronics_w > 0:                   # power down ten seconds after the landing
                self.aero.electronics_w = 0.0
                self.events.append([t, "landed", "mission complete: electronics switched off"])
        return Command(q_target={})

    def _birds(self, t, pos, v, V, h, hdot, roll, pitch, yaw, omega_b, dt):
        """Mode 'birds': fly the companion's guidance (course, height, airspeed: ``vegeta.mission.GuidanceCmd``). No fresh
        command (the computer is off, crashed or silent) for ``companion_timeout_s``: hold a gentle orbit, then
        return, as with the waypoint stream. A 'done' command ends the mission (return)."""
        cmd = self.companion.command() if (self.companion is not None and self.jetson_alive) else None
        fresh = cmd is not None and cmd.valid and t - cmd.t < 1.0
        if cmd is not None and cmd.mode == "done":
            self.events.append([t, "mission", f"companion: {cmd.note or 'bird mission complete'}: return"])
            self.return_ordered, self.return_reason = True, "mission complete"
            self._set_phase("return", t)
            return
        if not fresh:
            if self.companion_lost_since is None:
                self.companion_lost_since = t
            if t - self.companion_lost_since >= self.companion_timeout_s:
                self.events.append([t, "failsafe", f"no guidance from the computer for {self.companion_timeout_s:.0f} s: return"])
                self.return_ordered, self.return_reason = True, "companion lost"
                self._set_phase("return", t)
            else:
                self._hold(pos, v, V, h, hdot, roll, pitch, omega_b, dt)
            return
        self.companion_lost_since = None
        track = math.atan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 2.0 else yaw
        err = wrap(cmd.course - track)
        bank = -max(-1, min(1, self.g.k_track * err)) * math.radians(self.g.bank_max_deg)
        V_cmd = max(cmd.airspeed, self.V_app + 1.0)                      # the guidance never commands a speed near the stall
        pc, thr = self._speed_height(V_cmd, cmd.height, V, h, hdot, dt)
        self._attitude(bank, pc, roll, pitch, omega_b, V, thr)

    def _hold(self, pos, v, V, h, hdot, roll, pitch, omega_b, dt):
        pc, thr = self._speed_height(self.plan.V_survey, h, V, h, hdot, dt)
        self._attitude(math.radians(20.0), pc, roll, pitch, omega_b, V, thr)

    def _first_target(self):
        if self.survey_wps:
            return self.survey_wps[0]
        return Waypoint(self.home[0] + 300 * math.cos(self.launch_heading), self.home[1] + 300 * math.sin(self.launch_heading), self.plan.climb_alt_m)

    def _iaf(self):
        """The initial approach fix: 450 m from home, downwind of it, so the final runs into the wind."""
        w = self.wind_est
        into = -w / np.linalg.norm(w) if np.linalg.norm(w) > 0.5 else np.array([math.cos(self.launch_heading), math.sin(self.launch_heading)])
        return self.home - 450.0 * into

    def _touchdown_point(self):
        """The aim point short of home on the final: 50 m in calm air (the flare and the float carry the aircraft on
        towards home), less in a headwind (the float over the ground shrinks with the ground speed), at least 10 m."""
        w = self.wind_est
        into = -w / np.linalg.norm(w) if np.linalg.norm(w) > 0.5 else np.array([math.cos(self.launch_heading), math.sin(self.launch_heading)])
        return self.home - max(50.0 - 6.0 * float(np.linalg.norm(w)), 10.0) * into

    @property
    def finished(self):
        return self.phase == "landed"
