"""Nisus+ Zero chasing birds in the mountains: NISUS-Zero's bird-photography mission software (``vegeta.mission``,
notebook 32) on the bigger aircraft, over a mountain slope (notebook 33).

Nothing of the mission software is new: the detector, the multi-bird tracker, the target scheduler, the stern
photo-pass guidance with its separation breakoffs and the shutter are ``vegeta.mission``'s; the seam to the MuJoCo
aircraft is NISUS's ``nisus_birds.BirdMission`` hook (the navigation state out, the birds stepped, the guidance in for
the flight controller's mode 'birds'), which works on ``nisus_plus_robot.NisusPlusAero`` unchanged. What the mountains change:

- **the birds**: soaring birds of the Alps (``vegeta.mission.sim.SPECIES``: golden eagle, griffon vulture, alpine
  chough) beating along the south face in its slope lift (the 'wander' behaviour over a 250 m reach) or circling wide in
  a thermal (160 m), their height band above sea level (``BirdField`` works in absolute heights). A tight thermal
  circle (~90 m radius) is about NISUS+'s own turn radius at 22 m/s TAS: the bird leaves the fixed camera's 34° frame
  before the range closes, and the kinematic loop gets no good photo of it — a gimbal (todo 5n.5) would;
- **the tracker**: the range from the bird's size assumes the planned species' wingspan (NISUS's 0.9 m crow/gull
  prior halves a 2 m eagle's range and fires the shutter far too early);
- **the aircraft**: NISUS+ flies 18-24 m/s TAS at 3000 m and is twice NISUS's size, so the guidance's speeds, its
  minimum speed (above NISUS+'s 1.4 V_s with crow, in TAS at the altitude) and the separations are NISUS+'s: a
  25 m minimum separation (``min_sep``: eagles attack drones — a breakoff starts well before; 15 m for the choughs,
  which do not, and which span 150 px only within ~28 m), the photo range 35 m (the 6 mm lens resolves a 2 m eagle at
  ~300 px there);
- **the heights**: the guidance's height limits and the scheduler's band are above sea level over the slope; the flight
  controller's terrain floor stays in force under the bird mission (``NisusPlusController._birds``);
- **the wind**: a south wind gives the ridge lift the birds use; the controller's crow and propeller brake keep the
  height when the same lift carries the aircraft up.

    import nisus_plus_birds as fb
    bs = fb.BirdScenario()                     # three golden eagles beating along the face in the ridge lift, a south wind
    ep = fb.run(bs)
    fb.summary(ep), fb.photo_table(ep)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from vegeta.mission import IMX219_6MM, DetectorConfig, MissionConfig, budget
from vegeta.mission.guidance import GuidanceConfig
from vegeta.mission.photo import PhotoConfig
from vegeta.mission.sim import SPECIES, BirdField, BirdGroup
from vegeta.mission.tracking import TrackerConfig
from vegeta.mission.targeting import SchedulerConfig

import nisus_plus_robot as fr
import nisus_plus_scenario as fsc
import nisus_plus_systems as fs
import nisus_birds as nb

__all__ = ["BirdScenario", "bird_scenarios", "mission_config", "run", "summary", "photo_table", "compare"]


def mission_config(centre_xy, ground_m, *, device="orin_nano_super", model="yolox-tiny", band=(150.0, 450.0), sigma_ref=0.78,
                   size_prior_m: float = 0.9, min_sep: float = 25.0, search_agl: float | None = None) -> MissionConfig:
    """The mission software's configuration for NISUS+ over a slope whose ground is ``ground_m`` [m ASL] at the bird area:
    heights above sea level, NISUS+'s speeds (TAS at the area's density ``sigma_ref``), the separations for big raptors,
    the tracker's wingspan prior for the range (``size_prior_m``: the species the flight is planned for), the search
    leg's height above the ground (``search_agl``: the planned birds' height; default the band's middle)."""
    k = 1 / math.sqrt(sigma_ref)
    h0, h1 = ground_m + band[0], ground_m + band[1]
    det = DetectorConfig(device=device, model=model, search_tiles=2, search_hz=5.0, roi_hz=15.0)
    guid = GuidanceConfig(search_centre=tuple(centre_xy), search_radius=260.0,
                          search_height=ground_m + search_agl if search_agl is not None else 0.5 * (h0 + h1), cruise=18.0 * k, dash=22.0 * k,
                          v_min=16.5 * k, slow_range=220.0, behind_m=45.0, pass_range=110.0, photo_range=35.0, closure_max=8.0, closure_min=3.0,
                          min_sep=min_sep, react_s=3.0, breakoff_s=5.0, climb_m=25.0, reposition_range=220.0, reposition_max_s=18.0,
                          h_min=h0 - 60.0, h_max=h1 + 120.0)
    sch = SchedulerConfig(area_centre=tuple(centre_xy), area_radius=700.0, dash_speed=22.0 * k, h_band=(h0 - 80.0, h1 + 120.0), commit_range=220.0)
    return MissionConfig(camera=IMX219_6MM, detector=det, tracker=TrackerConfig(size_prior_m=size_prior_m), scheduler=sch, guidance=guid,
                         photo=PhotoConfig(good_px=150.0, min_px=80.0))


@dataclass
class BirdScenario:
    """The birds (``groups``: species, count, behaviour, height band above the ground [m], radius, offset from the
    area's centre [m]), the area's centre on the south face, the wind, the Jetson, the hunting time."""
    name: str = "golden eagles in the ridge lift"
    groups: list = field(default_factory=lambda: [("golden eagle", 3, "wander", (220.0, 380.0), 250.0, (0.0, 0.0))])
    centre: tuple = (2400.0, 2300.0)
    wind: tuple = (0.0, 6.0, 0.0)
    sigma: float = 0.6
    device: str = "orin_nano_super"
    model: str = "yolox-tiny"
    lens: str = "6 mm M12"
    jetson_failure: tuple | None = None
    bird_time_s: float = 600.0
    min_sep: float = 25.0                    # [m] the guidance's separation (25 m from raptors that may attack)
    seed: int = 1
    battery_key: str = fs.DEFAULT_PACK

    @property
    def label(self):
        return f"Nisus+ Zero birds: {self.name}"

    @property
    def slug(self):
        return "nisus_plus_" + "".join(c if c.isalnum() else "_" for c in f"birds_{self.name}").strip("_").replace("__", "_")

    def ground(self, massif: fr.Massif = fr.Massif()) -> float:
        return float(massif.height(*self.centre))

    def scenario(self) -> fsc.Scenario:
        g = self.ground()
        return fsc.Scenario(name=f"birds: {self.name}", battery_key=self.battery_key, wind=self.wind, sigma=self.sigma, jetson_failure=self.jetson_failure,
                            climb_alt=max(3300.0, g + 450.0), survey_x=(self.centre[0] - 200.0, self.centre[0]), survey_y=(self.centre[1],),
                            survey_agl=300.0, duration=self.bird_time_s + 1300.0, seed=self.seed)

    def make_birds(self):
        massif = fr.Massif()
        c = np.asarray(self.centre, float)
        groups = []
        for sp, n, beh, band, r, off in self.groups:
            xy = c + np.asarray(off, float)
            g = float(massif.height(*xy))
            groups.append(BirdGroup(sp, n, tuple(xy), beh, (g + band[0], g + band[1]), r, spread=40.0))
        # the birds soar in the slope's lift: they hold their place over the face (a thermal would drift downwind; a ridge's
        # lift does not move), so their circles do not drift with the wind
        return BirdField(groups, wind=(0.0, 0.0, 0.0), seed=self.seed)

    def size_prior(self) -> float:
        """The wingspan the tracker assumes for the range: the planned species' (count-weighted; NISUS's 0.9 m crow/gull
        prior halves the range of a 2 m eagle and fires the shutter far too early)."""
        n = sum(g[1] for g in self.groups)
        return sum(SPECIES[g[0]].wingspan_m * g[1] for g in self.groups) / n

    def band(self) -> tuple:
        """The planned birds' height band above the ground and their count-weighted middle (the search leg's height:
        a camera 15° down sees small birds 50-100 m below only at ranges too long to detect them)."""
        n = sum(g[1] for g in self.groups)
        return (min(g[3][0] for g in self.groups), max(g[3][1] for g in self.groups)), sum(0.5 * (g[3][0] + g[3][1]) * g[1] for g in self.groups) / n

    def config(self) -> MissionConfig:
        sig = fs.atmosphere(self.ground() + 300.0)["sigma"]
        band, mid = self.band()
        return mission_config(self.centre, self.ground(), device=self.device, model=self.model, band=band, sigma_ref=sig,
                              size_prior_m=self.size_prior(), min_sep=self.min_sep, search_agl=mid)


def bird_scenarios() -> list:
    """Golden eagles beating along the face in the ridge lift (south wind); griffon vultures circling wide in a thermal (calm air); a flock of alpine
    choughs wandering along the face (small, quick, many); the eagles with the Jetson failing during the hunt."""
    base = BirdScenario()
    return [base,
            replace(base, name="griffon vultures circling", groups=[("griffon vulture", 5, "thermal", (200.0, 420.0), 160.0, (0.0, 0.0))], wind=(0.0, 0.0, 0.0), sigma=0.0),
            replace(base, name="alpine chough flock", groups=[("alpine chough", 8, "wander", (150.0, 300.0), 220.0, (0.0, 0.0)),
                                                              ("golden eagle", 1, "wander", (300.0, 420.0), 200.0, (250.0, 150.0))],
            min_sep=15.0),          # a 0.8 m chough spans 150 px only within ~28 m (6 mm lens): inside its flight distance
            replace(base, name="eagles, Jetson failure", jetson_failure=(None, None))]


def run(bs: BirdScenario, *, duration=None, log_geoms: bool = True, sim=None):
    """The whole flight: the bungee launch, the climb over the meadow, the transit to the face, ``bird_time_s`` of
    hunting, the return, the crow descent, the landing. ``sim``: another aircraft's pieces (``nisus_plus_scenario.Sim``)."""
    scn = bs.scenario()
    lab = fsc.make_lab(scn, log_geoms=log_geoms, sim=sim)
    mission = nb.BirdMission(bs.make_birds, bs.config(), bird_time_s=bs.bird_time_s, seed=bs.seed)
    lab.add_hook(mission)
    fc = fsc.controller(lab, scn)
    fc.mode, fc.companion, fc.mission = "birds", mission, None
    fc.name = f"{lab.sim.craft} autopilot + bird mission ({bs.name})"
    if bs.jetson_failure == (None, None):                         # fail 90 s into the hunt (the hunt's start is not known in advance)
        fc.jetson_failure = None
        orig = mission.__call__

        def failing(lab_, _orig=orig):
            _orig(lab_)
            if mission.t_start is not None and fc.jetson_failure is None:
                fc.jetson_failure = (mission.t_start + 90.0, mission.t_start + 130.0)
        lab._hooks = [failing if h is mission else h for h in lab._hooks]
    mission.fc = fc
    ep = fsc.run(lab, scn, duration=duration, fc=fc)
    ep.birds = mission
    ep.bird_scenario = bs
    ep.log["birds"] = mission.loop.rec
    ep.log["photos"] = mission.loop.photos
    ep.log["near"] = mission.loop.near
    return ep


def summary(ep) -> dict:
    """NISUS's bird summary (photos, tracking scores, separations) plus NISUS+'s altitude and the landing."""
    out = nb.summary(ep)
    ts = fsc.timeseries(ep)
    hunt = ts[(ts["t"] >= (ep.birds.t_start or 0.0)) & (ts["t"] <= (ep.birds.t_stop or ts["t"].iloc[-1]))]
    out.update({"hunt altitude [m]": f"{hunt['z_asl'].min():.0f}-{hunt['z_asl'].max():.0f}" if len(hunt) else "-",
                "min agl in the hunt [m]": round(float(hunt["agl"].min()), 0) if len(hunt) else np.nan,
                "crow used in the hunt": round(float(hunt["crow"].mean()), 2) if len(hunt) else np.nan, "outcome": ep.outcome["reason"]})
    return out


def compare(eps) -> pd.DataFrame:
    return pd.DataFrame([summary(e) for e in eps]).set_index("scenario")


photo_table = nb.photo_table
