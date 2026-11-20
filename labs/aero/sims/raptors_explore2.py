"""RAPTORS exploration, run 2: around run 1's winners, with the run-2 levers. Only run 2's own specs are computed
(its own cache and run files: nothing of run 1 is repeated, but the two references per line below are re-scored).

    labs/aero/sims/raptors_explore2.sh -j 12                  # all four lines (the -j cap still applies)
    labs/aero/sims/raptors_explore2.sh buteo astor -j 12      # some lines
    labs/aero/sims/raptors_explore2.sh --list                 # the specs, then exit
    labs/aero/sims/raptors_explore2.sh --rescore output/runs/raptors_explore_<time>_<host>.yaml
                                                              # run 1 scored again under the current KPIs (no simulation)

Per line (``designs/arigo_products.SPACE2``): the baseline (ARIGO-1H on the lattice) and run 1's best, re-scored under
the run-2 models (chases from five starts, the operating band split from calm air) as references; then

- BUTEO: span 1000-1100 x chord 300-320 x the perch spring 6 / 8 / 10 m/s, turning station;
- ARPELLA: span 1100 / 1250 x chord 260 / 280 x 3S3P / 3S4P, with and without a parachute (the heavy calm-air landing);
- ASTOR: span 700-800 x chord 240 / 260 x motor 1000 / 1200 / 1400 W x 3S2P / 3S3P, rail + turning station +
  parachute; and run 1's best ARIGO-20;
- SPARFELL: wing scale 0.9 / 1.0 x fuselage scale 0.75 / 0.80 / 0.85 (0.65 does not fit the tail) (the fuselage set the packed length), 3S1P,
  quick joiner + lite avionics, with and without the rugged skin.

KPI change for run 2: B5 (quasi-steady gust stall) is reported, not scored. Everything else as ``raptors_explore.sh``
(cards, launch success and operating wind, -j cap, one timestamped YAML, charts). ~90 specs, ~80 s each on one core.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raptors_explore as RX  # noqa: E402

if __name__ == "__main__":
    sys.exit(RX.main(gen="explore2", tag="raptors_explore2", doc=__doc__))
