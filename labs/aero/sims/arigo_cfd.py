"""ARIGO — PHASE 1: CFD only (OpenFOAM via Aeromant). Run this, paste the result file back, the design is revised,
then run the stability phase (arigo_stability.sh). Rinse and repeat.

WHAT IT RUNS
  Angle-of-attack sweeps (-4..24 deg at cruise speed) of the airframes: lift, drag, pitching moment about the CG.
    ARIGO-1 (pusher)                    3 wings x 11 angles                     33 cases
    ARIGO-0 (nose tractor; + chin strut) 2 layouts x 3 wings x 11               66 cases
    ARIGO-20 (ducted fans)              11 angles at 30 m/s                     11 cases
    ARIGO-2 (wing tractors)             3 wings x 11 propellers off + 5 on      48 cases
  Screening quality: ~1.5 min a case on 8-12 cores; --full-cfd (fine mesh, 600 iterations): ~5-8 min a case.
  Needs OpenFOAM (source its bashrc first).

USAGE
  labs/aero/sims/arigo_cfd.sh                          all four aircraft, 8 cores
  labs/aero/sims/arigo_cfd.sh -j 12                    12 cores (or --cpu 12)
  labs/aero/sims/arigo_cfd.sh ARIGO-1 ARIGO-0          only these aircraft
  labs/aero/sims/arigo_cfd.sh --full-cfd -j 12 ARIGO-1 converged sweeps into arigo*_fine_cfd.yaml (screening kept)
  labs/aero/sims/arigo_cfd.sh --fresh                  ignore saved results

RESTARTING
  Ctrl-C any time, run the same command again: it goes on where it stopped. A case already solved for the same
  geometry comes back from its OpenFOAM folder in seconds; a changed geometry gets new folders (tagged _g<hash>).

OUTPUT — paste this back
  ONE file per run:  labs/aero/sims/output/runs/arigo_cfd_<YYYYmmdd-HHMMSS>_<host>.yaml
  (run settings + every CFD result of the run). Detail log: labs/aero/sims/output/arigo_cfd.log
  Several machines -> one file:  python labs/aero/sims/arigo_merge.py labs/aero/sims/output/runs/arigo_cfd_*.yaml -o cfd_all.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arigo_run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(arigo_run.main(sys.argv[1:], "cfd", __doc__))
