#!/usr/bin/env python3
"""Tune the MuJoCo autopilot of the reference quad per race course (todo R.2): real MuJoCo, no mocks, no solver.

    python tune_autopilot.py                  # all four courses: a particle swarm (PSO), then a pattern-search polish (races/converge.py)
    python tune_autopilot.py -j 8             # 8 worker processes flying laps side by side (default 1; it prints the CPU count)
    python tune_autopilot.py --courses hangar quarry
    python tune_autopilot.py --pso-iterations 10 --polish-laps 150 --max-laps 400   # the swarm's hard limit, the polish's reserve
    python tune_autopilot.py --method gps     # polish only, from the stored settings (--method pso: the swarm only)
    python tune_autopilot.py --fresh          # start over instead of resuming
    python tune_autopilot.py --round2 | --quick   # the earlier fixed grids

Resumable: every PSO and polishing iteration is checkpointed to races/runs/tune/<course>.ckpt.json (atomically); a
rerun with the same options continues where it stopped and skips finished courses; other options are refused (use
--fresh). The same seed and options give the same laps whatever -j is.

Every step prints a line; the whole output also goes to races/runs/tune/tune_<date>.log. Results:
races/runs/tune/<course>_converge.csv (every lap), best.json and races/autopilot.json, written after every course,
with the time spent (PSO, polishing, total, per lap)."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "races" / "runs" / "tune"
BACKENDS = ("egl", "osmesa", "glfw", "")              # "" = MUJOCO_GL unset: MuJoCo's own choice


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def say(msg):
    print(f"tune_autopilot: {msg}", flush=True)


def pick_backend() -> str | None:
    """The MUJOCO_GL with which ``import mujoco`` works here (the laps need no renderer, but the import loads one)."""
    if os.environ.get("MUJOCO_GL") is not None:
        say(f"MUJOCO_GL={os.environ['MUJOCO_GL']!r} (set by you)")
        return os.environ["MUJOCO_GL"]
    for b in BACKENDS:
        env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"} | ({"MUJOCO_GL": b} if b else {})
        p = subprocess.run([sys.executable, "-c", "import mujoco"], env=env, capture_output=True, text=True, timeout=120)
        if p.returncode == 0:
            say(f"MUJOCO_GL={b or '(unset)'}: mujoco imports")
            return b
        say(f"MUJOCO_GL={b or '(unset)'}: mujoco does not import ({(p.stderr.strip().splitlines() or ['?'])[-1][:120]})")
    return None


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"tune_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = log.open("w")
    sys.stdout = sys.stderr = Tee(sys.__stdout__, fh)
    say(f"repository {ROOT}")
    say(f"python {sys.executable} ({sys.version.split()[0]}); log {log}")
    for p in (ROOT, ROOT / "designs", ROOT / "vegeta-cli" / "src", ROOT / "vegeta-core" / "src",
              ROOT / "vendor" / "ozaru" / "porunga" / "src", ROOT / "vendor" / "ozaru" / "yanus" / "src"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    b = pick_backend()
    if b is None:
        say("no MUJOCO_GL works: is mujoco installed in this python?  pip install mujoco  (or ./install_local.sh)")
        return 1
    if b:
        os.environ["MUJOCO_GL"] = b
    else:
        os.environ.pop("MUJOCO_GL", None)
        os.environ["MUJOCO_GL"] = ""                   # stop races.sim's setdefault("egl")
    try:
        import mujoco
        say(f"mujoco {mujoco.__version__}")
        from races import tune
    except Exception as e:
        say(f"cannot import mujoco / races ({type(e).__name__}: {e})")
        return 1
    say(f"starting races.tune {' '.join(argv) or '(all courses, full grid)'}")
    t0 = time.time()
    rc = tune.main(argv)
    say(f"finished in {(time.time() - t0) / 60:.1f} min (exit {rc}); results: {OUT / 'best.json'} and {OUT}/<course>.csv")
    return rc


if __name__ == "__main__":
    sys.exit(main())
