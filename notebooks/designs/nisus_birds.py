"""NISUS-Zero bird photography in MuJoCo: the mission software (``vegeta.mission``) flying the 6-DOF aircraft.

The flight software is ``vegeta.mission``: detector scheduling, tracking, target selection, photo-pass guidance and
the shutter, on its own message bus. It knows nothing of MuJoCo. This module is the simulator side of the
seam:

- ``BirdMission`` is a Chiron scene hook (stepped with the aircraft, every ``every`` control steps). It turns the
  ``Aero`` hook's state into a ``NavState`` (the autopilot's navigation solution) and steps the birds
  (``vegeta.mission.sim.BirdField``). While the aircraft is in its survey phase and the Jetson is powered, it runs
  the mission stack (``vegeta.mission.sim.SimLoop``: the simulated YOLOX from the truth, everything else as on the
  hardware). ``command()`` is what the flight controller's mode 'birds' flies (course, height, airspeed). After
  ``bird_time_s`` of hunting it answers 'done' and the autopilot returns.
- ``BirdScenario`` is a ``nisus_scenario.Scenario`` plus the birds and the mission configuration (Jetson, YOLOX
  model, lens). ``run`` flies it end to end: hand launch, climb, outbound, the hunt, the return, the landing.
- ``summary`` and ``photo_table`` give the numbers; ``render_movie`` (``nisus_birds_movie``) draws the video.

YOLOX is not run: detections come from the truth through the detector model (``vegeta.mission.sim.detector``) at
the latency and rates of the ASSUMED Jetson budget (``vegeta.mission.compute``).

    import nisus_birds as nb
    bs = nb.BirdScenario()                     # four gulls in a thermal, two crows, calm air, Orin Nano Super
    ep = nb.run(bs)
    nb.summary(ep), nb.photo_table(ep)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from vegeta.mission import IMX219, IMX219_6MM, DetectorConfig, MissionConfig, NavState, budget
from vegeta.mission.guidance import GuidanceConfig
from vegeta.mission.messages import GuidanceCmd
from vegeta.mission.photo import PhotoConfig
from vegeta.mission.sim import BirdField, BirdGroup, SimDetectorConfig, SimLoop
from vegeta.mission.targeting import SchedulerConfig

import nisus_controller as nc
import nisus_flight as nf
import nisus_scenario as nsc
import nisus_systems as ns

__all__ = ["BirdMission", "BirdScenario", "bird_scenarios", "mission_config", "run", "summary", "photo_table", "compare", "LENSES"]

LENSES = {"stock 3.04 mm": IMX219, "6 mm M12": IMX219_6MM}


class BirdMission:
    """The companion computer's mission as a Chiron hook (see the module)."""

    def __init__(self, make_birds, cfg: MissionConfig, *, bird_time_s: float = 360.0, every: int = 2, seed: int = 0,
                 sim_det: SimDetectorConfig = SimDetectorConfig()):
        self.make_birds, self.cfg = make_birds, cfg
        self.bird_time_s, self.every, self.seed, self.sim_det = bird_time_s, every, seed, sim_det
        self.fc = None
        self.reset(None)

    def reset(self, lab):
        self.loop = SimLoop(self.make_birds(), self.cfg, seed=self.seed, sim_det=self.sim_det)
        self.k = 0
        self.t_start = None
        self.t_stop = None
        self.done = False
        self.t = 0.0
        self.last_t = None

    def _nav(self, lab):
        a = lab.aero.last
        fc = self.fc
        margin = fc.margin if (fc is not None and fc.margin is not None) else float("inf")
        return NavState(lab.time, np.asarray(a["pos"], float).copy(), np.asarray(a["v"], float).copy(), np.array(a["R"], float).reshape(3, 3),
                        np.asarray(a["omega_b"], float).copy(), float(a["V"]), float(margin), fc.phase if fc is not None else "")

    def __call__(self, lab):
        self.k += 1
        if self.k % self.every or not lab.aero.last:
            return
        t = lab.time
        dt = t - self.last_t if self.last_t is not None else lab.control_dt * self.every
        self.last_t = t
        self.t = t
        fc = self.fc
        hunting = fc is not None and fc.phase == "survey" and fc.jetson_alive and not self.done
        if hunting and not self.loop.active:
            if self.t_start is None:
                self.t_start = t
            self.loop.activate(t)
        elif not hunting and self.loop.active:
            self.loop.active = False
            self.t_stop = t
        if self.t_start is not None and not self.done and t - self.t_start >= self.bird_time_s:
            self.done = True
        self.loop.step(t, self._nav(lab), dt)

    def command(self):
        if self.done:
            sch = self.loop.stack.scheduler
            return GuidanceCmd(self.t, "done", valid=True, note=f"{self.bird_time_s:.0f} s of hunting over, {len(sch.done)} targets done")
        return self.loop.stack.command() if self.loop.active else None


def mission_config(device="orin_nano_super", model="yolox-tiny", lens="6 mm M12", *, field_centre=(0.0, 300.0), search_height=70.0,
                   search_tiles=2, search_hz=5.0, roi_hz=15.0, good_px=150.0, area_radius=450.0) -> MissionConfig:
    """The mission software's configuration for a Jetson, a YOLOX model and a lens; a Nano B01 gets the rates its
    budget allows (``budget`` scales them down)."""
    cam = LENSES[lens]
    det = DetectorConfig(device=device, model=model, search_tiles=search_tiles, search_hz=search_hz, roi_hz=roi_hz)
    return MissionConfig(camera=cam, detector=det,
                         scheduler=SchedulerConfig(area_centre=tuple(field_centre), area_radius=area_radius),
                         guidance=GuidanceConfig(search_centre=tuple(field_centre), search_height=search_height),
                         photo=PhotoConfig(good_px=good_px, min_px=0.55 * good_px))


@dataclass
class BirdScenario:
    name: str = "gulls in a thermal"
    groups: list = field(default_factory=lambda: [("black-headed gull", 4, "thermal", (50.0, 85.0), 35.0, (0.0, 0.0)),
                                                  ("carrion crow", 2, "wander", (40.0, 70.0), 120.0, (120.0, 80.0))])
    device: str = "orin_nano_super"
    model: str = "yolox-tiny"
    lens: str = "6 mm M12"
    wind: tuple = (0.0, 0.0, 0.0)
    sigma: float = 0.0
    jetson_failure: tuple | None = None
    radius_m: float = 350.0                 # the bird area's centre from home
    bird_time_s: float = 360.0
    seed: int = 1
    battery_key: str = "gens-ace-3s-2200"

    @property
    def label(self):
        return f"Nisus-Zero birds: {self.name} ({self.device}, {self.model}, {self.lens})"

    @property
    def slug(self):
        return "".join(c if c.isalnum() else "_" for c in f"birds_{self.name}_{self.device}_{self.lens}").strip("_").replace("__", "_")

    def scenario(self) -> nsc.Scenario:
        return nsc.Scenario("Zero", name=f"birds: {self.name}", battery_key=self.battery_key, wind=self.wind, sigma=self.sigma,
                            jetson_failure=self.jetson_failure, radius_m=self.radius_m, legs=1, leg_length=60.0, alt_m=70.0,
                            duration=self.bird_time_s + 420.0, seed=self.seed)

    def centre(self):
        return np.asarray(self.scenario().field_centre, float)

    def make_birds(self):
        c = self.centre()
        groups = [BirdGroup(sp, n, tuple(c + np.asarray(off, float)), beh, h, r) for sp, n, beh, h, r, off in self.groups]
        return BirdField(groups, wind=self.wind, seed=self.seed)

    def config(self) -> MissionConfig:
        return mission_config(self.device, self.model, self.lens, field_centre=tuple(self.centre()))


def bird_scenarios() -> list:
    """The standard set: the reference (Orin Nano Super, YOLOX-Tiny, 6 mm lens), the stock lens, the Nano B01, a
    pigeon flock, a 5 m/s wind, and the Jetson failing during the hunt."""
    base = BirdScenario()
    return [base,
            replace(base, lens="stock 3.04 mm"),
            replace(base, device="nano_b01", model="yolox-nano"),
            replace(base, name="pigeon flock", groups=[("feral pigeon", 6, "wander", (40.0, 70.0), 150.0, (0.0, 0.0)),
                                                       ("herring gull", 2, "thermal", (60.0, 90.0), 40.0, (-100.0, 60.0))]),
            replace(base, name="wind 5 m/s", wind=(-5.0, 0.0, 0.0), sigma=0.6),
            replace(base, name="Jetson failure", jetson_failure=(200.0, 230.0))]


def run(bs: BirdScenario, *, duration=None):
    """The whole flight: launch, climb, outbound to the bird area, ``bird_time_s`` of hunting, return, landing."""
    scn = bs.scenario()
    lab = nsc.make_lab(scn)
    mission = BirdMission(bs.make_birds, bs.config(), bird_time_s=bs.bird_time_s, seed=bs.seed)
    lab.add_hook(mission)
    fc = nsc.controller(lab, scn)
    fc.mode, fc.companion, fc.mission = "birds", mission, None
    fc.name = f"Nisus-Zero autopilot + bird mission ({bs.name})"
    mission.fc = fc
    ep = nsc.run(lab, scn, duration=duration, fc=fc)
    ep.birds = mission
    ep.bird_scenario = bs
    ep.log["birds"] = mission.loop.rec
    ep.log["photos"] = mission.loop.photos
    ep.log["near"] = mission.loop.near
    return ep


def summary(ep) -> dict:
    bs, loop = ep.bird_scenario, ep.birds.loop
    sc = loop.scores()
    b = budget(bs.config().detector)
    o = ep.outcome
    t_hunt = (ep.birds.t_stop or loop.rec[-1]["t"]) - (ep.birds.t_start or 0.0) if ep.birds.t_start is not None else 0.0
    good = [p for p in loop.photos if p["good_true"]]
    return {"scenario": bs.name, "Jetson": b["device"], "model": b["model"], "lens": bs.lens, "GPU load (assumed)": round(b["GPU load"], 2),
            "search Hz": round(b["search [Hz] achieved"], 1), "crop Hz": round(b["roi [Hz] achieved"], 1),
            "hunting [s]": round(t_hunt, 0), "birds": sc["birds in the field"], "photos": sc["photos"], "good photos": sc["good photos (truth)"],
            "birds photographed well": sc["birds photographed well"], "best photo [px]": round(max((p["px_true"] for p in loop.photos), default=0.0), 0),
            "closest approach [m]": round(sc["closest approach to any bird [m]"], 1), "approaches < 10 m": sc["approaches under 10 m"],
            "MOTA": round(sc["MOTA"], 2), "IDF1": round(sc["IDF1"], 2), "recall": round(sc["recall"], 2), "identity switches": sc["identity switches"],
            "breakoffs": sc["breakoffs"], "return": ep.log.get("return_reason"), "landed": o.get("success"),
            "energy left [%]": round(100 * ep.log["energy"]["E_remaining_wh"] / ep.log["energy"]["E_available_wh"], 0)}


def compare(eps) -> pd.DataFrame:
    return pd.DataFrame([summary(e) for e in eps]).set_index("scenario")


def photo_table(ep) -> pd.DataFrame:
    df = pd.DataFrame(ep.birds.loop.photos)
    if df.empty:
        return df
    cols = ["t", "track", "bird", "species", "px_est", "px_true", "range_est", "range_true", "blur_true", "centring_true", "good_est", "good_true"]
    return df[[c for c in cols if c in df]].round(2)
