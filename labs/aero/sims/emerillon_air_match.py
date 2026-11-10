"""Play an air match: two Émerillons, a hoop on a 2 m stick at each end, drifting balloons, MuJoCo (``labs/aero/sims/air_match/``).

    .venv/bin/python labs/aero/sims/emerillon_air_match.py                          # 60 s, ER2 against ER2, seed 1: a summary sheet, a replay video, a JSON
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --seed 3 --duration 120 --variants ER1 ER2
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --plain                  # no token: a point for whoever flies through the other hoop
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --series 10              # ten matches, the table only
    MUJOCO_GL=egl .venv/bin/python labs/aero/sims/emerillon_air_match.py --camera grandstand   # also a MuJoCo camera video (needs OpenGL)

Rules: a point for flying through the other side's hoop; with the token (the default) only its carrier scores and the
other takes it by flying within 2 m of the carrier for 0.35 s without touching. Touching the other aircraft or a
balloon is a foul; the ground, a hoop or leaving the field a crash; both send the aircraft back to its start for 3 s.
Writes ``labs/aero/sims/output/air_match_<seed>.png``, ``.mp4`` and ``.json``.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "designs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from air_match import match  # noqa: E402

OUT = HERE / "output"


def line(seed, r):
    c = r["counts"]
    return (f"seed {seed}: A {r['score']['A']} : {r['score']['B']} B  winner {r['winner'] or 'draw'}  tags {c['tag']}  fouls {c['foul']}  crashes {c['crash']}  " +
            "  ".join(f"{n} {p['energy_wh']:.1f} Wh" for n, p in r["players"].items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variants", nargs=2, default=["ER2", "ER2"], choices=["ER1", "ER2"])
    ap.add_argument("--balloons", type=int, default=8)
    ap.add_argument("--plain", action="store_true", help="no token: the race to the hoops")
    ap.add_argument("--over", action="store_true", help="score by flying over the hoop instead of through it")
    ap.add_argument("--series", type=int, default=0, help="play this many matches (seeds from --seed) and print the table")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--camera", default=None, help="also a MuJoCo camera video: overview, grandstand, a1_chase, b1_chase")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    kw = dict(variants=tuple(args.variants), score="over" if args.over else "through", n_balloons=args.balloons, possession=not args.plain)
    if args.quick or os.environ.get("VEGETA_DEBUG") == "1":
        print(line(args.seed, match.play(6.0, seed=args.seed, **kw)))
        return 0
    if args.series:
        wins = {"A": 0, "B": 0, None: 0}
        for s in range(args.seed, args.seed + args.series):
            r = match.play(args.duration, seed=s, **kw)
            wins[r["winner"]] += 1
            print(line(s, r))
        print(f"A wins {wins['A']}, B wins {wins['B']}, draws {wins[None]}")
        return 0
    t0 = time.time()
    r = match.play(args.duration, seed=args.seed, **kw)
    print(line(args.seed, r))
    for e in r["events"]:
        print("  ", {k: v for k, v in e.items() if k != "pos"})
    import matplotlib
    matplotlib.use("Agg")
    from air_match import replay
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"air_match_{args.seed}"
    replay.summary(r, stem.with_suffix(".png"))
    files = [stem.with_suffix(".png")]
    if not args.no_video:
        files.append(replay.video(r, stem.with_suffix(".mp4")))
    if args.camera:
        files.append(replay.mujoco_video(r, OUT / f"air_match_{args.seed}_{args.camera}.mp4", camera=args.camera))
    stem.with_suffix(".json").write_text(json.dumps({"score": r["score"], "faults": r["faults"], "winner": r["winner"], "counts": r["counts"], "duration": r["duration"],
                                                     "events": r["events"], "players": {n: {k: v for k, v in p.items() if k != "hist"} for n, p in r["players"].items()}}, indent=1, default=float))
    print("written:", ", ".join(str(f) for f in files if f), f"and {stem.with_suffix('.json')}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
