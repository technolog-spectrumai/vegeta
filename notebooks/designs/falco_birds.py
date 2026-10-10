"""FALCO chasing birds: NISUS+'s four hunts (``nisus_plus_birds``: golden eagles in the ridge lift, griffon vultures
circling, the alpine chough flock, the eagles with the Jetson failing) on FALCO's pieces (``falco_scenario.SIM``);
the camera in the chin looks past the propeller's hub (computer vision masks the blades: accepted). The landing parks
the propeller as in the missions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import falco_scenario as fscn
import falco_systems as fsy
import nisus_plus_birds as npb
import nisus_plus_birds_movie as npm
from nisus_plus_birds import mission_config, summary, compare, photo_table  # noqa: F401


@dataclass
class BirdScenario(npb.BirdScenario):
    battery_key: str = fsy.DEFAULT_PACK

    @property
    def label(self):
        return f"{fscn.CRAFT} birds: {self.name}"

    @property
    def slug(self):
        return fscn.SLUG + "_" + "".join(c if c.isalnum() else "_" for c in f"birds_{self.name}").strip("_").replace("__", "_")

    def scenario(self):
        s = super().scenario()
        return fscn.Scenario(**{**{k: v for k, v in asdict(s).items() if k != "plan"}, "craft": fscn.CRAFT, "slug_prefix": fscn.SLUG, "battery_key": self.battery_key})


def bird_scenarios() -> list:
    return [BirdScenario(**asdict(b)) for b in npb.bird_scenarios()]


def run(bs: BirdScenario, *, duration=None, log_geoms: bool = True, progress: bool = False):
    """The whole flight on FALCO (``nisus_plus_birds.run`` with FALCO's ``sim``), judged by FALCO's outcome (the parked
    propeller at touchdown); ``progress``: a tqdm bar over the simulation's time."""
    ep = npb.run(bs, duration=duration, log_geoms=log_geoms, sim=fscn.SIM, progress=progress)
    fc = ep.controller
    ep.log["parked_at_touchdown"] = fc.parked_at_touchdown
    ep.log["t_park_cmd"] = fc.t_park_cmd
    ep.log["t_parked"] = next((e[0] for e in ep.log["events"] if e[1] == "propeller" and "parked horizontal" in e[2]), None)
    ep.outcome = fscn.outcome(ep, bs.scenario().land_radius)
    return ep


render_movie = npm.render_movie

__all__ = ["BirdScenario", "bird_scenarios", "run", "summary", "compare", "photo_table", "render_movie", "mission_config"]
