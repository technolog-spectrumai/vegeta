#!/usr/bin/env python3
"""Tune the MuJoCo autopilot of the reference quad per race course (todo R.2): real MuJoCo, no mocks, no solver.

    python tune_autopilot.py                  # all four courses: generalized pattern search until the mesh is below Δ_min (races/converge.py)
    python tune_autopilot.py --delta-min 0.03 --max-laps 200   # a coarser convergence, a smaller safety budget
    python tune_autopilot.py --round2         # the fixed grids instead (--quick: 4 laps per course)
    python tune_autopilot.py --quick          # 4 laps per course, to see it work
    python tune_autopilot.py --courses hangar quarry

Every step prints a line; the whole output also goes to races/runs/tune/tune_<date>.log. Results:
races/runs/tune/<course>.csv (every lap) and races/runs/tune/best.json (the winning settings per course)."""
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
