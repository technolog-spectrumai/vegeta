"""Play an air match: two Émerillons, a hoop on a 2 m stick at each end, drifting balloons, MuJoCo (``labs/aero/sims/air_match/``).

    .venv/bin/python labs/aero/sims/emerillon_air_match.py                          # 60 s, ER2 against ER2, seed 1: a summary sheet, a replay video, a JSON
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --seed 3 --duration 120 --variants ER1 ER2
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --plain                  # no token: a point for whoever flies through the other hoop
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --series 10              # ten matches, the table only
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --mode formation          # step 2: seven in formation chase a hare through both hoops
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --mode chase3             # step 3: three, every one for itself, chase the token
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --mode chase7             # step 4: seven of them
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --mode circus             # seven strategies race three laps round the circus, wind and balloons
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --mode circus --tournament 7   # seven races, the grid turned each time: which strategy wins
    .venv/bin/python labs/aero/sims/emerillon_air_match.py --mode circus --tournament 4 --wind 0   # the same in still air (or --wind 9)
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


def play(mode, duration, seed, kw):
    if mode == "teamcircus":
        from air_match import teamcircus
        return teamcircus.play(kw["size"], seed, max(duration, 90.0), kw["head_start"], blue_tactic=kw["blue_tactic"])
    if mode == "teamchase":
        from air_match import teamchase
        return teamchase.play(kw["red"], kw["blue"], seed, duration, kw["head_start"])
    if mode == "circus":
        from air_match import circus
        return circus.race(seed, kw.get("laps", 3), kw.get("field"), wind=kw.get("wind"), duration=max(duration, 150.0), n_balloons=kw.get("circus_balloons", 14),
                           slipstream=kw.get("slipstream", True))
    if mode == "formation":
        from air_match import formation
        return formation.fly(7, duration, seed)
    if mode in ("chase3", "chase7"):
        return match.chase(int(mode[-1]), duration, seed, kw["variants"][0], kw["n_balloons"])
    return match.play(duration, seed=seed, **kw)


def line(seed, r):
    c = r["counts"]
    if "teamchase" in r and "reds" in r["teamchase"]:
        x = r["teamchase"]
        return (f"seed {seed}: {x['n']}v{x['n']} round the circus: {x['winner']} wins — reds out {len(x['caught'])}/{x['n']}; red laps "
                + ", ".join(f"{q['red']} ({q['strategy']}) {q['laps']}" for q in x["reds"]) + f"; blue laps {x['blue_laps']}")
    if "teamchase" in r:
        x = r["teamchase"]
        return (f"seed {seed}: {x['red']} reds vs {x['blue']} blues, head start {x['head_start_s']:g} s: {x['winner']} wins — caught "
                + ", ".join(f"{n} at {tt:.1f} s" for n, tt in sorted(x["caught"].items(), key=lambda q: q[1])) + f"; red seconds {x['red_seconds']}")
    if "race" in r:
        return f"seed {seed}: " + "  ".join(f"{s['place']}. {s['strategy']} {(str(s['time_s']) + ' s') if s['finished'] else 'DNF'} ({s['faults']} faults)" for s in r["race"]["standings"])
    if "formation" in r:
        s = r["formation"]["summary"]
        return (f"seed {seed}: formation of {s['members']}  laps {s['laps']}  slot error wedge {s['slot_error_wedge_m']:.2f} m, column {s['slot_error_column_m']:.2f} m  "
                f"closest pair {s['closest_pair_m']:.2f} m  changes {s['changes']}  missed hoops {s['missed_hoops']}  contacts {s['contacts']}")
    if set(r["score"]) != {"A", "B"}:
        return (f"seed {seed}: " + "  ".join(f"{k} {v:g}" for k, v in r["score"].items()) + f"  winner {r['winner']}  tags {c['tag']}  hoops {c['hoop']}  fouls {c['foul']}  crashes {c['crash']}")
    return (f"seed {seed}: A {r['score']['A']} : {r['score']['B']} B  winner {r['winner'] or 'draw'}  tags {c['tag']}  fouls {c['foul']}  crashes {c['crash']}  " +
            "  ".join(f"{n} {p['energy_wh']:.1f} Wh" for n, p in r["players"].items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="match", choices=["match", "formation", "chase3", "chase7", "circus", "teamchase", "teamcircus"])
    ap.add_argument("--size", type=int, default=3, help="teamcircus: reds and blues per team (3, 5, 7)")
    ap.add_argument("--blue-tactic", default="cut", choices=["cut", "track"], help="teamcircus: blues fly straight at their red (over the spina), or round the track")
    ap.add_argument("--red", default="scatter", choices=["scatter", "flock", "jink"], help="teamchase: the reds' tactic")
    ap.add_argument("--blue", default="assigned", choices=["assigned", "nearest", "pincer"], help="teamchase: the blues' tactic")
    ap.add_argument("--head-start", type=float, default=3.0, help="teamchase: seconds the reds fly before the blues may go")
    ap.add_argument("--tournament", type=int, default=0, help="circus: this many races of the seven strategies, the grid turned each time")
    ap.add_argument("--wind", type=float, default=None, help="circus: the wind at 10 m [m/s] (default 5)")
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--field", default="mixed", choices=["mixed", "fixed"], help="circus: four fixed and three adaptive strategies, or the seven fixed ones")
    ap.add_argument("--no-slipstream", action="store_true", help="circus: no wake behind the aircraft")
    ap.add_argument("--gusts", action="store_true", help="circus: a wind that changes during the race (speed, direction, gust fronts)")
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
    kw.update(red=args.red, blue=args.blue, head_start=args.head_start, size=args.size, blue_tactic=args.blue_tactic)
    if args.mode == "teamcircus" and args.tournament and not (args.quick or os.environ.get("VEGETA_DEBUG") == "1"):
        import pandas as pd
        from air_match import teamcircus
        t = teamcircus.tournament((args.size,), args.tournament, args.seed, max(args.duration, 90.0), args.head_start,
                                  progress=lambda n, bt, s, x: print(f"{n}v{n} {bt:5s} seed {s}: {x['winner']} ({len(x['caught'])} out, red laps {x['red_laps']})", flush=True))
        pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 120)
        print(pd.DataFrame(t["table"]).to_string(index=False))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"air_teamcircus_tournament_{args.size}v{args.size}.json").write_text(json.dumps(t, indent=1, default=float))
        return 0
    if args.mode == "teamchase" and args.tournament and not (args.quick or os.environ.get("VEGETA_DEBUG") == "1"):
        import pandas as pd
        from air_match import teamchase
        t = teamchase.tournament(args.tournament, args.seed, args.duration, args.head_start,
                                 progress=lambda rt, bt, s, x: print(f"{rt:8s} vs {bt:9s} seed {s}: {x['winner']} ({len(x['caught'])} caught, all at {x['all_caught_at']})", flush=True))
        pd.set_option("display.width", 250)
        print(pd.DataFrame(t["table"]).to_string(index=False))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"air_teamchase_tournament_hs{args.head_start:g}.json").write_text(json.dumps(t, indent=1, default=float))
        return 0
    if args.mode == "circus":
        from air_match import circus
        kw["laps"] = args.laps
        kw["field"] = circus.FIELD if args.field == "mixed" else circus.STRATEGIES
        kw["slipstream"] = not args.no_slipstream
        kw["circus_balloons"] = args.balloons if args.balloons != 8 else 14            # --balloons: how many over the track (default 14)
        kw["wind"] = None if args.wind is None else dict(circus.WIND, speed=args.wind, gust=0.0 if args.wind == 0 else circus.WIND["gust"])
        if args.gusts:
            kw["wind"] = dict(circus.CHANGING, speed=circus.WIND["speed"] if args.wind is None else args.wind)
        if args.tournament and not (args.quick or os.environ.get("VEGETA_DEBUG") == "1"):
            import pandas as pd
            t = circus.tournament(args.tournament, args.seed, args.laps, kw["field"], wind=kw["wind"], n_balloons=kw["circus_balloons"], slipstream=kw["slipstream"],
                                  progress=lambda i, rc: print(f"race {i + 1} (seed {rc['seed']}): " + " > ".join(rc["order"]) + f"   {rc['time']} s, {rc['faults']} faults", flush=True))
            pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
            print(pd.DataFrame(t["table"]).to_string(index=False))
            OUT.mkdir(parents=True, exist_ok=True)
            tag = ("_gusts" if args.gusts else "") + ("_fixed" if args.field == "fixed" else "") + ("_nowake" if args.no_slipstream else "") + ("" if args.wind is None else f"_wind{args.wind:g}") + ("" if kw["circus_balloons"] == 14 else f"_balloons{kw['circus_balloons']}")
            (OUT / f"air_circus_tournament{tag}.json").write_text(json.dumps(t, indent=1, default=float))
            print("written:", OUT / f"air_circus_tournament{tag}.json")
            return 0
    if args.quick or os.environ.get("VEGETA_DEBUG") == "1":
        if args.mode == "circus":
            from air_match import circus
            r = circus.Race(circus.STRATEGIES[:3], args.seed, 1, 14.0).run()
            print("quick:", [(s["strategy"], s["laps"], s["progress"]) for s in r["race"]["standings"]])
            return 0
        print(line(args.seed, play(args.mode, 6.0, args.seed, kw)))
        return 0
    if args.series:
        wins = {"A": 0, "B": 0, None: 0}
        for s in range(args.seed, args.seed + args.series):
            r = play(args.mode, args.duration, s, kw)
            wins[r["winner"]] = wins.get(r["winner"], 0) + 1
            print(line(s, r))
        print("wins:", {k: v for k, v in wins.items() if v})
        return 0
    t0 = time.time()
    r = play(args.mode, args.duration, args.seed, kw)
    print(line(args.seed, r))
    for e in r["events"]:
        print("  ", {k: v for k, v in e.items() if k != "pos"})
    import matplotlib
    matplotlib.use("Agg")
    from air_match import replay
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / (f"air_teamcircus_{args.size}v{args.size}_{args.seed}" if args.mode == "teamcircus" else f"air_match_{args.seed}" if args.mode == "match" else f"air_{args.mode}{'_gusts' if getattr(args, 'gusts', False) else ''}_{args.seed}")
    from air_match import teamcircus as _tcirc
    (_tcirc.summary if "teamchase" in r and "reds" in r["teamchase"] else replay.race_summary if "race" in r else replay.formation_summary if "formation" in r else replay.summary)(r, stem.with_suffix(".png"))
    files = [stem.with_suffix(".png")]
    if not args.no_video:
        files.append(replay.video(r, stem.with_suffix(".mp4")))
    if args.camera:
        files.append(replay.mujoco_video(r, OUT / f"air_match_{args.seed}_{args.camera}.mp4", camera=args.camera))
    stem.with_suffix(".json").write_text(json.dumps({"score": r["score"], "faults": r["faults"], "winner": r["winner"], "counts": r["counts"], "duration": r["duration"],
                                                     "events": r["events"], "formation": (r["formation"]["summary"] | {"changes": r["formation"]["changes"]}) if "formation" in r else None,
                                                     "teamchase": r.get("teamchase"), "race": {"standings": r["race"]["standings"], "wind": r["race"]["wind"], "laps": r["race"]["laps"]} if "race" in r else None, "players": {n: {k: v for k, v in p.items() if k != "hist"} for n, p in r["players"].items()}}, indent=1, default=float))
    print("written:", ", ".join(str(f) for f in files if f), f"and {stem.with_suffix('.json')}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
