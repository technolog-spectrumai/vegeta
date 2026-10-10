"""FALCO flight controller and missions over the mountains (notebook 33).

NISUS's controller (``nisus_controller``: the attitude loops, the line following, the waypoint stream and its failure,
the companion seam of mode 'birds') reworked for altitude and terrain — ArduPilot Plane's TECS and terrain following
in a simplified form:

* **speed by pitch, energy by thrust and drag** — the pitch holds the commanded **EAS** (the pitot's indicated speed:
  the stall and V_NE stay put as the air thins); the total energy is held by three actuators in turn: the throttle
  (from the thrust the drag and the climb rate need at the local density — the drive's model — plus a climb-rate
  integrator), then **crow** (0..1; only below V_FE) and the **propeller brake** (0..1) when even the freewheeling
  propeller gives more energy than wanted (a fast descent; lift that would carry the aircraft into the cloud). The
  climb and sink rates are limited per phase;
* **terrain** — the height above the ground from the terrain database (the onboard DEM: here the exact ``Massif``),
  a floor ``agl_min`` along the track and ahead of it (``Massif.max_along``: the highest ground in the next
  ``lookahead_s`` of flight and to its sides), the survey legs at a constant height above the slope;
* **phases** (``PHASES``): prelaunch, launch (a light bungee, full power), climb (a spiral over the meadow at full power
  to ``climb_alt``), transit (to the survey area at the work height), survey (the Jetson's waypoints, or the bird
  mission's guidance in mode 'birds'), return (to above home), descent (a crow + brake spiral over the meadow to the
  approach height), approach (a steep crow final into the wind, the flare on the lidar height), landed;
* **the energy manager** — from where it is: home at the cruise EAS, the descent (only the electronics: the height is
  the energy), the approach, the go-arounds, the uncertainty, the reserve; a weather event or the companion's loss
  orders the return too.

The controller knows what the real one would: the attitude and rates, position and velocity (GNSS), the EAS (pitot) and
the density (baro + temperature), the battery's energy and voltage, the terrain database, the wind only as estimated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta.chiron import Command

import falco_flight as ff
import falco_robot as fr
import falco_systems as fs
from nisus_controller import Waypoint, AutoMission, survey_pattern, wrap

G = 9.81
PHASES = ("prelaunch", "launch", "climb", "transit", "survey", "return", "descent", "approach", "landed")

__all__ = ["G", "PHASES", "Gains", "Waypoint", "AutoMission", "slope_survey", "EnergyManager", "FalcoController", "wrap"]


@dataclass
class Gains:
    k_roll: float = 1.0
    k_p: float = 0.10
    k_pitch: float = 1.6
    k_q: float = 0.20
    k_yaw: float = 0.25
    k_ar: float = 0.12
    k_v: float = 0.030            # pitch [rad] per m/s of EAS error
    k_vi: float = 0.004
    k_h: float = 0.20             # demanded climb rate per m of height error [1/s]
    k_hd: float = 2.0             # thrust per (m/s of climb-rate error) [N per m/s]
    k_hdi: float = 0.8            # its integrator [N per m]
    k_track: float = 1.6
    L1: float = 60.0
    bank_max_deg: float = 40.0
    pitch_min_deg: float = -38.0
    pitch_max_deg: float = 30.0


def slope_survey(massif: fr.Massif, x0, x1, ys, agl, speed_eas) -> list:
    """Survey legs along the contour (east-west) at the stations ``ys`` across the slope, ``agl`` above the ground at
    each end (the controller follows the terrain between them): waypoints in flying order; ``h`` is the height above
    the ground (``Waypoint.name`` says 'agl')."""
    wps = []
    for i, y in enumerate(ys):
        a, b = (x0, x1) if i % 2 == 0 else (x1, x0)
        wps += [Waypoint(a, y, agl, speed_eas, f"leg {i + 1} start (agl)"), Waypoint(b, y, agl, speed_eas, f"leg {i + 1} end (agl)")]
    return wps


class EnergyManager:
    """The return decision (NISUS's, over the mountains): home at the cruise EAS from here (the wind along the way), the
    descent's electronics, the approach, the go-arounds — x (1 + uncertainty) + the reserve."""

    def __init__(self, aero: fr.FalcoAero, drive, airframe: dict, plan: fs.MissionPlan, electronics: dict, *, home_alt: float):
        self.aero, self.drive, self.af, self.plan, self.el, self.home_alt = aero, drive, airframe, plan, electronics, home_alt
        self.E_reserve = plan.reserve_frac * aero.E_available_Wh
        self._P = {}

    def P_cruise(self, h):
        k = int(h // 250)
        if k not in self._P:
            a = fs.atmosphere(k * 250.0 + 125.0, self.aero.dT)
            V = self.plan.V_cruise_eas / math.sqrt(a["sigma"])
            af = self.af
            P, _ = fs.level_power(self.drive, af["mass_kg"], af["cd0"], af["AR"], af["oswald"], af["S"], V, k * 250.0 + 125.0, 0.0, self.aero.dT)
            self._P[k] = (P, V)
        return self._P[k]

    def needed(self, pos, wind_xy, home_xy) -> dict:
        pos = np.asarray(pos, float)
        P_c, V = self.P_cruise(pos[2])
        leg = np.asarray(home_xy, float) - pos[:2]
        d = float(np.linalg.norm(leg))
        u = leg / d if d > 1 else np.zeros(2)
        gs = max(V + float(np.asarray(wind_xy, float) @ u), 3.0)
        t1 = d / gs
        E_back = (P_c + self.el["return"]) * t1 / 3600
        t_desc = max(pos[2] - self.home_alt - 250.0, 0.0) / self.plan.descent_rate
        E_desc = self.el["descent"] * t_desc / 3600
        P0, V0 = self.P_cruise(self.home_alt)
        E_land = (0.5 * P0 + self.el["approach+landing"]) * self.plan.approach_s / 3600
        E_ga = self.plan.go_arounds * (P0 + self.el["return"]) * self.plan.circuit_m / V0 / 3600
        E = (E_back + E_desc + E_land + E_ga) * (1 + self.plan.uncertainty_frac)
        rem = self.aero.E_remaining_Wh
        return {"E_return_wh": E, "E_reserve_wh": self.E_reserve, "E_needed_wh": E + self.E_reserve, "E_remaining_wh": rem,
                "margin_wh": rem - E - self.E_reserve, "route_m": d}


class FalcoController:
    """The flight controller with its mission (see the module). ``mode``: 'auto' (the Jetson's waypoints, ``mission``)
    or 'birds' (the companion's guidance). ``weather_at``: a time [s] at which a storm warning orders the return and
    the fastest descent. ``jetson_failure``: (t_fail, t_reboot)."""

    def __init__(self, aero: fr.FalcoAero, deriv_c: dict, airframe: dict, plan: fs.MissionPlan, massif: fr.Massif, *, home=(0.0, 0.0),
                 launch_heading_deg=0.0, climb_alt: float = 3500.0, mode="auto", mission: AutoMission | None = None, survey_wps=None,
                 energy: EnergyManager | None = None, electronics_phase_w: dict | None = None, jetson_w: float = 0.0, jetson_failure=None,
                 weather_at: float | None = None, wind_estimate=(0.0, 0.0), prelaunch_sim_s: float = 4.0, ground_s: float = 300.0, gains: Gains = Gains(),
                 companion_timeout_s: float = 5.0, airborne_cap_s: float = 2400.0, companion=None, transit_agl: float = 200.0, agl_min: float = 80.0,
                 lookahead_s: float = 25.0, climb_rate: float = 7.0, descent_rate: float = 12.0, V_climb_eas: float = 16.0, V_cruise_eas: float = 18.0,
                 V_descent_eas: float = 21.0, orbit_r: float = 280.0, approach_agl: float = 120.0, glide_slope_deg: float = 9.0, name: str = "Falco flight controller"):
        self.aero, self.c, self.af, self.plan, self.massif, self.g = aero, deriv_c, airframe, plan, massif, gains
        self.home = np.asarray(home, float)
        self.home_alt = float(massif.height(*self.home))
        self.launch_heading = math.radians(launch_heading_deg)
        self.climb_alt, self.mode, self.mission, self.energy = climb_alt, mode, mission, energy
        self.survey_wps = survey_wps or []
        self.el_phase = electronics_phase_w or {}
        self.jetson_w, self.jetson_failure, self.weather_at = jetson_w, jetson_failure, weather_at
        self.wind_est = np.asarray(wind_estimate, float)
        self.prelaunch_sim_s, self.ground_s, self.companion_timeout_s, self.airborne_cap_s = prelaunch_sim_s, ground_s, companion_timeout_s, airborne_cap_s
        self.companion, self.name = companion, name
        self.transit_agl, self.agl_min, self.lookahead_s = transit_agl, agl_min, lookahead_s
        self.climb_rate, self.descent_rate = climb_rate, descent_rate
        self.V_climb_eas, self.V_cruise_eas, self.V_descent_eas = V_climb_eas, V_cruise_eas, V_descent_eas
        self.orbit_r, self.approach_agl, self.glide_slope = orbit_r, approach_agl, math.radians(glide_slope_deg)
        Vs = math.sqrt(2 * airframe["mass_kg"] * G / (fs.RHO0 * airframe["S"] * (deriv_c["CL_max"] + deriv_c["dCLmax_crow"])))
        self.V_app_eas = 1.4 * Vs                                          # 1.4 V_s with crow: the flare needs the margin
        self.k_drag = 1 / (math.pi * airframe["AR"] * airframe["oswald"])

    # ---- lifecycle
    def reset(self, lab, seed=None):
        self.lab = lab
        self.log, self.events = [], []
        self.vi = self.hi = 0.0
        self.return_ordered, self.return_reason = False, ""
        self.jetson_alive, self._rebooted = True, False
        self.companion_lost_since = None
        self.touchdown = None
        self.target = None
        self.margin = None
        self.go_arounds = 0
        self.t_launch = self.prelaunch_sim_s
        self.storm = False
        self.v_mean, self.v_var = 18.0, 0.0
        self._set_phase("prelaunch", 0.0)
        extra = max(self.ground_s - self.prelaunch_sim_s, 0.0) * self.el_phase.get("prelaunch", 0.0) / 3600
        self.aero.add_energy_wh(extra)
        self.events.append([0.0, "energy", f"ground operation {self.ground_s:.0f} s at {self.el_phase.get('prelaunch', 0.0):.1f} W charged: {extra:.3f} Wh"])

    def _set_phase(self, name, t):
        self.phase, self.t_phase = name, t
        self.log.append([t, name, "start"])
        self.aero.electronics_w = self._electronics(name)

    def _electronics(self, phase):
        key = {"prelaunch": "prelaunch", "launch": "launch+climb", "climb": "launch+climb", "transit": "transit", "survey": "survey", "return": "return",
               "descent": "descent", "approach": "approach+landing", "landed": "approach+landing"}[phase]
        w = self.el_phase.get(key, 0.0)
        return w - (0.0 if self.jetson_alive else self.jetson_w)

    def _order_return(self, t, reason, note):
        if self.return_ordered:
            return
        self.return_ordered, self.return_reason = True, reason
        self.events.append([t, reason, note])
        self.target = None
        self._set_phase("return", t)

    # ---- inner loops
    def _attitude(self, roll_cmd, pitch_cmd, roll, pitch, omega_b, V, eas, throttle, crow, brake):
        g = self.g
        s = float(np.clip((18.0 / max(eas, 8.0)) ** 2, 0.5, 1.6))          # gains scheduled on the dynamic pressure
        p_, q_, r_ = omega_b[0], -omega_b[1], -omega_b[2]
        pitch_cmd = max(math.radians(g.pitch_min_deg), min(math.radians(g.pitch_max_deg), pitch_cmd))
        da = s * (g.k_roll * (roll_cmd - roll) - g.k_p * p_)
        de_ff = -crow * self.c["dCm_crow"] / self.c["Cmde"]                 # the crow's pitching moment trimmed out
        de = s * (-(g.k_pitch * (pitch_cmd - pitch)) + g.k_q * q_) + math.radians(1.6) + de_ff
        r_coord = G * math.sin(roll) * math.cos(pitch) / max(V, 6.0)
        dr = -s * g.k_yaw * (r_coord - r_) - g.k_ar * da
        self.aero.command = np.array([da, de, dr, throttle, crow, brake])

    def _drag(self, eas, sigma, crow=0.0, n=1.0):
        q = 0.5 * fs.RHO0 * eas ** 2
        S = self.af["S"]
        CL = n * self.af["mass_kg"] * G / (q * S)
        k = self.k_drag + crow * (self.c["k_crow"] - self.c["k"])
        return q * S * (self.af["cd0"] + crow * self.c["dCD_crow"] + k * CL * CL)

    def _energy(self, V_eas_cmd, h_cmd, a, dt, *, climb_max=3.0, sink_max=3.0, full=False, allow_crow=True, hd_ff=0.0):
        """Pitch for the EAS; thrust, then crow, then the brake for the climb rate (see the module)."""
        g = self.g
        eas, V, h, hdot, rho, sig = a["EAS"], a["V"], a["h"], a["v"][2], a["rho"], a["sigma"]
        ev = V_eas_cmd - eas
        self.vi = max(-0.2, min(0.2, self.vi + g.k_vi * ev * dt))
        hd_dem = float(np.clip(g.k_h * (h_cmd - h) + hd_ff, -sink_max, climb_max))
        theta_trim = math.atan2(hd_dem, max(V, 5.0)) * 0.9 + 0.05 * (14.0 / max(eas, 9.0) - 0.75)
        pitch_cmd = theta_trim - g.k_v * ev - self.vi
        W = self.af["mass_kg"] * G
        err = hd_dem - hdot
        self.hi = max(-25.0, min(25.0, self.hi + g.k_hdi * err * dt))
        T_need = self._drag(eas, sig) + W * hd_dem / max(V, 5.0) + g.k_hd * err + self.hi
        dr = self.aero.drive
        u = max(a["u"], 0.0)
        if full:
            return pitch_cmd, 1.0, 0.0, 0.0
        T_free = dr.freewheel(u, rho)["thrust"]
        if T_need >= T_free:
            thr = dr.throttle_for_thrust(u, T_need, rho, self.aero.v_batt) if T_need > 0 else 0.0
            return pitch_cmd, max(thr, 0.0), 0.0, 0.0
        extra = T_free - T_need                                             # drag still wanted [N]
        crow = 0.0
        if allow_crow and eas <= ff.V_FE_EAS:
            D_crow = self._drag(eas, sig, 1.0) - self._drag(eas, sig)
            crow = float(np.clip(extra / max(D_crow, 1e-3), 0.0, 1.0))
            extra -= crow * D_crow
        brake = 0.0
        if extra > 0:
            D_brake = T_free - dr.brake(u, 1.0, rho)["thrust"]
            brake = float(np.clip(extra / max(D_brake, 1e-3), 0.0, 1.0))
        return pitch_cmd, 0.0, crow, brake

    def _track_to(self, pos, v, A, B):
        d = B - A
        Ld = float(np.linalg.norm(d))
        u = d / Ld if Ld > 1e-6 else np.array([1.0, 0.0])
        rel = pos[:2] - A
        along = float(rel @ u)
        e = float(rel[0] * u[1] - rel[1] * u[0])
        course_cmd = math.atan2(u[1], u[0]) + math.atan(e / self.g.L1) * 0.9
        track = math.atan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 2.0 else course_cmd
        bank = -max(-1, min(1, self.g.k_track * wrap(course_cmd - track))) * math.radians(self.g.bank_max_deg)
        return bank, along, Ld, e

    def _course(self, v, course, yaw):
        track = math.atan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 2.0 else yaw
        return -max(-1, min(1, self.g.k_track * wrap(course - track))) * math.radians(self.g.bank_max_deg)

    def _orbit(self, pos, v, centre, r, direction=1):
        """Bank to circle ``centre`` at radius r (counter-clockwise for direction +1)."""
        rel = pos[:2] - centre
        d = float(np.linalg.norm(rel)) + 1e-6
        tang = direction * np.array([-rel[1], rel[0]]) / d
        course = math.atan2(tang[1], tang[0]) - direction * math.atan((d - r) / 60.0)
        return self._course(v, course, 0.0)

    def _floor(self, pos, v):
        """The terrain floor: the highest ground ahead (``lookahead_s`` of flight) plus ``agl_min``."""
        gs = max(float(np.linalg.norm(v[:2])), 8.0)
        return self.massif.max_along(pos[:2], v[:2], gs * self.lookahead_s) + self.agl_min

    # ---- the loop
    def __call__(self, obs):
        t = float(obs.t)
        a = self.aero.last
        if not a:
            return Command(q_target={})
        pos, v, V, eas = a["pos"], a["v"], a["V"], a["EAS"]
        roll, pitch, yaw, omega_b = a["roll"], a["pitch"], a["yaw"], a["omega_b"]
        h, agl = a["h"], a["agl"]
        dt = self.lab.control_dt
        home_d = float(np.linalg.norm(pos[:2] - self.home))
        if self.phase in ("transit", "survey", "return"):
            k = dt / 20.0
            self.v_mean += k * (eas - self.v_mean)
            self.v_var += k * ((eas - self.v_mean) ** 2 - self.v_var)
        if self.jetson_failure is not None:
            t_fail, t_reboot = self.jetson_failure
            if self.jetson_alive and t >= t_fail and not self._rebooted:
                self.jetson_alive = False
                if self.mission is not None:
                    self.mission.alive = False
                self.events.append([t, "jetson", "power branch lost: computer off, waypoint stream stopped"])
                self.aero.electronics_w = self._electronics(self.phase)
            if not self.jetson_alive and t >= t_reboot and not self._rebooted:
                self._rebooted, self.jetson_alive = True, True
                self.events.append([t, "jetson", "computer rebooted (power back); the mission stays in return"])
                self.aero.electronics_w = self._electronics(self.phase)
        if self.weather_at is not None and t >= self.weather_at and not self.storm and self.phase in ("climb", "transit", "survey", "return"):
            self.storm = True
            self.events.append([t, "weather", "storm warning from the ground station: return and the fastest descent"])
            if self.phase != "return":
                self._order_return(t, "weather", "return ordered: storm warning")
        if self.energy is not None and self.phase in ("climb", "transit", "survey"):
            need = self.energy.needed(pos, self.wind_est, self.home)
            self.margin = need["margin_wh"]
            self.aero.margin_wh = self.margin
            if self.margin <= 0.0 and self.phase in ("transit", "survey"):
                self._order_return(t, "energy", f"return ordered: remaining {need['E_remaining_wh']:.1f} Wh ≤ needed {need['E_needed_wh']:.1f} Wh "
                                                f"(home {need['route_m']:.0f} m: {need['E_return_wh']:.1f} Wh with the allowance, reserve {need['E_reserve_wh']:.1f})")
        if self.phase in ("transit", "survey") and t - self.t_launch > self.airborne_cap_s:
            self._order_return(t, "time cap", f"airborne {self.airborne_cap_s:.0f} s: return")
        ph = self.phase
        if ph == "prelaunch":
            self.aero.command = np.zeros(6)
            if t >= self.t_launch:
                self.aero.hold = False
                self._set_phase("launch", t)
                self.events.append([t, "launch", "bungee launch at full power"])
        elif ph == "launch":
            self._attitude(0.0, math.radians(12.0), roll, pitch, omega_b, V, eas, 1.0, 0.0, 0.0)
            if (eas > self.V_climb_eas - 1.0 and agl > 8.0) or t - self.t_phase > 5.0:
                self._set_phase("climb", t)
        elif ph == "climb":
            centre = self.home + self.orbit_r * np.array([math.cos(self.launch_heading), math.sin(self.launch_heading)])
            bank = self._orbit(pos, v, centre, self.orbit_r) if agl > 30.0 else 0.0
            pc, thr, crow, brake = self._energy(self.V_climb_eas, self.climb_alt, a, dt, climb_max=self.climb_rate)
            self._attitude(bank, pc, roll, pitch, omega_b, V, eas, thr, crow, brake)
            if h >= self.climb_alt - 15.0:
                self._set_phase("transit", t)
        elif ph == "transit":
            tgt = self._first_target()
            if self.target is None:
                self.target = pos[:2].copy()
            bank, along, Ld, _ = self._track_to(pos, v, self.target, np.array([tgt.x, tgt.y]))
            h_cmd = max(self.climb_alt, self._floor(pos, v) + self.transit_agl - self.agl_min)
            pc, thr, crow, brake = self._energy(self.V_cruise_eas, h_cmd, a, dt, climb_max=4.0, sink_max=6.0)
            self._attitude(bank, pc, roll, pitch, omega_b, V, eas, thr, crow, brake)
            if along >= Ld - 40.0:
                self.target = None
                self._set_phase("survey", t)
                if self.mission is not None:
                    self.mission.advance()
        elif ph == "survey":
            if self.mode == "birds":
                self._birds(t, pos, v, a, roll, pitch, yaw, omega_b, dt)
            else:
                wp = self.mission.current() if self.mission is not None else None
                if wp is None:
                    if self.mission is not None and not self.mission.alive:
                        if self.companion_lost_since is None:
                            self.companion_lost_since = t
                        if t - self.companion_lost_since >= self.companion_timeout_s:
                            self._order_return(t, "companion lost", f"no waypoint from the computer for {self.companion_timeout_s:.0f} s: return")
                        else:
                            self._hold(pos, v, a, roll, pitch, omega_b, dt)
                    else:
                        self._order_return(t, "survey complete", "survey complete: return")
                else:
                    prev = self.mission.waypoints[self.mission.i - 1] if self.mission.i > 0 else None
                    A = np.array([prev.x, prev.y]) if prev is not None else pos[:2]
                    B = np.array([wp.x, wp.y])
                    bank, along, Ld, _ = self._track_to(pos, v, A, B)
                    # terrain following: the commanded height above the ground under the aircraft and ahead, never under the floor
                    ahead = pos[:2] + v[:2] * 8.0
                    h_cmd = max(float(self.massif.height(*ahead)) + wp.h, float(self.massif.height(*pos[:2])) + wp.h, self._floor(pos, v))
                    pc, thr, crow, brake = self._energy(wp.speed, h_cmd, a, dt, climb_max=5.0, sink_max=5.0)
                    self._attitude(bank, pc, roll, pitch, omega_b, V, eas, thr, crow, brake)
                    if along >= Ld - 30.0:
                        self.mission.advance()
        elif ph == "return":
            if self.target is None:
                self.target = pos[:2].copy()
            fix = self.home
            bank, along, Ld, _ = self._track_to(pos, v, self.target, fix)
            h_cmd = max(self._floor(pos, v) + self.transit_agl - self.agl_min, min(h, self.climb_alt))
            sink = self.descent_rate if self.storm else 6.0
            pc, thr, crow, brake = self._energy(self.V_cruise_eas if not self.storm else self.V_descent_eas, h_cmd, a, dt, climb_max=4.0, sink_max=sink)
            self._attitude(bank, pc, roll, pitch, omega_b, V, eas, thr, crow, brake)
            if home_d < self.orbit_r + 60.0:
                self._set_phase("descent", t)
                self.events.append([t, "descent", f"over the meadow at {h:.0f} m ({agl:.0f} m above it): crow + brake spiral"])
        elif ph == "descent":
            centre = self.home
            bank = self._orbit(pos, v, centre, self.orbit_r)
            h_cmd = self.home_alt + self.approach_agl + 60.0
            pc, thr, crow, brake = self._energy(self.V_descent_eas, h_cmd, a, dt, climb_max=2.0, sink_max=self.descent_rate)
            self._attitude(bank, pc, roll, pitch, omega_b, V, eas, thr, crow, brake)
            if agl < self.approach_agl + 90.0:
                self.target = None
                self._set_phase("approach", t)
        elif ph == "approach":
            iaf, td = self._iaf(), self._touchdown_point()
            if self.target is None:
                self.target = "to iaf"
            if self.target == "to iaf":
                bank, along, Ld, _ = self._track_to(pos, v, self.home, iaf)
                pc, thr, crow, brake = self._energy(self.V_cruise_eas - 2.0, self.home_alt + self.approach_agl, a, dt, climb_max=2.0, sink_max=4.0)
                self._attitude(bank, pc, roll, pitch, omega_b, V, eas, thr, crow, brake)
                if float(np.linalg.norm(pos[:2] - iaf)) < 60.0:
                    self.target = "final"
                    self.events.append([t, "landing", f"final: {math.degrees(self.glide_slope):.0f}° crow approach"])
                return Command(q_target={})
            bank, along, Ld, e = self._track_to(pos, v, iaf, td)
            dist = max(Ld - along, 0.0)
            h_slope = self.home_alt + dist * math.tan(self.glide_slope)
            if along > Ld - 15.0 and agl > 12.0:
                self.go_arounds += 1
                self.events.append([t, "landing", f"go-around {self.go_arounds}: {agl:.0f} m high over the aim point"])
                self.target = "to iaf"
                return Command(q_target={})
            V_gust = min(2.0 * math.sqrt(max(self.v_var, 0.0)), 3.0)
            V_tgt = self.V_app_eas + V_gust
            if agl > 6.0:
                gs = float(np.linalg.norm(v[:2]))
                pc, thr, crow, brake = self._energy(V_tgt, h_slope, a, dt, climb_max=2.0, sink_max=6.0, hd_ff=-gs * math.tan(self.glide_slope))
                crow = max(crow, 0.3)
                self._attitude(bank if agl > 10.0 else 0.4 * bank, pc, roll, pitch, omega_b, V, eas, thr, crow, brake)
            else:
                # the flare: crow retracted (the ailerons back down: the lift and the stall margin return), the pitch for a sink
                # that shrinks with the height, held 2.5° below the stall's angle of attack (the α protection)
                hdot = v[2]
                sink_tgt = max(0.3 * agl, 0.4)
                err = (-hdot) - sink_tgt
                pc = math.radians(1.0) + float(np.clip(0.10 * err, -0.05, math.radians(6.0)))
                a_max = math.radians(self.c["alpha_stall_deg"] - 2.5)
                pc = min(pc, pitch + (a_max - a["alpha"]))
                thr = float(np.clip(0.4 * err, 0.0, 0.5)) if agl > 0.6 else 0.0
                self._attitude(0.0, pc, roll, pitch, omega_b, V, eas, thr, 0.0, 0.0)
            if self.touchdown is None and agl < 0.45 and t - self.t_phase > 5.0:
                self.touchdown = t
                self.events.append([t, "landing", f"touchdown at {np.linalg.norm(v[:2]):.1f} m/s ground speed, sink {-v[2]:.2f} m/s, {home_d:.0f} m from home"])
            if self.touchdown is not None and float(np.linalg.norm(v)) < 1.0:
                self._set_phase("landed", t)
                self.aero.command = np.zeros(6)
        else:
            self.aero.command = np.zeros(6)
            if t - self.t_phase > 10.0 and self.aero.electronics_w > 0:
                self.aero.electronics_w = 0.0
                self.events.append([t, "landed", "mission complete: electronics switched off"])
        return Command(q_target={})

    def _birds(self, t, pos, v, a, roll, pitch, yaw, omega_b, dt):
        """Mode 'birds' (NISUS's seam): fly the companion's guidance (course, height, EAS); its heights are above sea level;
        the terrain floor still holds."""
        cmd = self.companion.command() if (self.companion is not None and self.jetson_alive) else None
        fresh = cmd is not None and cmd.valid and t - cmd.t < 1.0
        if cmd is not None and cmd.mode == "done":
            self._order_return(t, "mission complete", f"companion: {cmd.note or 'bird mission complete'}: return")
            return
        if not fresh:
            if self.companion_lost_since is None:
                self.companion_lost_since = t
            if t - self.companion_lost_since >= self.companion_timeout_s:
                self._order_return(t, "companion lost", f"no guidance from the computer for {self.companion_timeout_s:.0f} s: return")
            else:
                self._hold(pos, v, a, roll, pitch, omega_b, dt)
            return
        self.companion_lost_since = None
        bank = self._course(v, cmd.course, yaw)
        V_cmd = max(cmd.airspeed * math.sqrt(a["sigma"]), self.V_app_eas + 1.5)
        h_cmd = max(cmd.height, self._floor(pos, v))
        pc, thr, crow, brake = self._energy(V_cmd, h_cmd, a, dt, climb_max=5.0, sink_max=6.0)
        self._attitude(bank, pc, roll, pitch, a["omega_b"], a["V"], a["EAS"], thr, crow, brake)

    def _hold(self, pos, v, a, roll, pitch, omega_b, dt):
        h_cmd = max(a["h"], self._floor(pos, v))
        pc, thr, crow, brake = self._energy(self.V_cruise_eas, h_cmd, a, dt)
        self._attitude(math.radians(20.0), pc, roll, pitch, omega_b, a["V"], a["EAS"], thr, crow, brake)

    def _first_target(self):
        if self.survey_wps:
            return self.survey_wps[0]
        return Waypoint(self.home[0] + 1500 * math.cos(self.launch_heading), self.home[1] + 1500 * math.sin(self.launch_heading), 200.0)

    def _into(self):
        """The landing direction: along the valley (the launch heading's axis) — in a narrow valley the final runs along
        it whatever the wind, against the wind's component along it (a crosswind landing otherwise)."""
        axis = np.array([math.cos(self.launch_heading), math.sin(self.launch_heading)])
        along = float(self.wind_est @ axis)
        return -axis if along > 0.5 else axis

    def _iaf(self):
        """The initial approach fix down the valley from home, against the landing direction: the final runs into the wind."""
        return self.home - (self.approach_agl / math.tan(self.glide_slope) + 60.0) * self._into()

    def _touchdown_point(self):
        return self.home - max(40.0 - 4.0 * float(np.linalg.norm(self.wind_est)), 10.0) * self._into()

    @property
    def finished(self):
        return self.phase == "landed"
