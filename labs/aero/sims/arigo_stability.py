"""ARIGO — PHASE 2: flight analysis on the current design and its CFD (no CFD is run here). Run after the CFD
results have been put into the models (arigo_cfd.sh, pasted back, design revised). Rinse and repeat.

WHAT IT RUNS (minutes)
  1. performance cards  speeds (stall, max level), turns (sustained / instantaneous over speed, corner speed), climb,
                        roll / pitch / yaw response, chases under proportional navigation (zig-zag, urban: hard turns
                        at 20 m height, open field: a long 18 m/s chase) + run-down of a fleeing bandit, endurance,
                        range, mass budget and pack trade; thrust and power from the propulsor maps   ~8 s per config.
  KPI scores (0-100, relative to the set): general, chase, urban chase, open-field long chase
  2. stability          MuJoCo 6-DOF: modes, stall and recovery, gust, engine / fan out   ~5-10 s per configuration
  3. camera             ARIGO-0's propeller in the camera's view          seconds
  4. videos             only with --videos (needs OpenGL: MUJOCO_GL=egl or osmesa, and ffmpeg)   ~15 s each
  Aircraft: ARIGO-0 (nose tractor; + chin strut), ARIGO-1 (pusher), ARIGO-2 (wing tractors) on the goshawk, zero
  and hybrid wings; ARIGO-20 (ducted fans).

USAGE
  labs/aero/sims/arigo_stability.sh                    everything above except the videos, all aircraft
  labs/aero/sims/arigo_stability.sh ARIGO-1 ARIGO-20   only these aircraft
  labs/aero/sims/arigo_stability.sh --videos           also the manoeuvre videos
  labs/aero/sims/arigo_stability.sh --fresh            ignore saved results

RESTARTING
  Ctrl-C any time, run the same command again: it goes on where it stopped. Results from older code are set aside
  as *.stale.yaml and recomputed.

OUTPUT — paste this back
  ONE file per run:  labs/aero/sims/output/runs/arigo_stability_<YYYYmmdd-HHMMSS>_<host>.yaml
  (run settings + every result + KPI scores), and next to it the charts:
      ..._properties.png  bar charts of every property     ..._kpi.png  KPI scores
      ..._curves.png      turn rate, climb rate and battery power over speed
  Detail log: labs/aero/sims/output/arigo_stability.log
  Several runs/machines -> one file:  python labs/aero/sims/arigo_merge.py labs/aero/sims/output/runs/arigo_stability_*.yaml -o all.yaml
                                     python labs/aero/sims/arigo_charts.py all.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arigo_run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(arigo_run.main(sys.argv[1:], "stability", __doc__))
