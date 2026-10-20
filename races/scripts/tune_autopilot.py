#!/usr/bin/env python3
"""Tune the MuJoCo autopilot of the reference quad per race course (todo R.2): real MuJoCo, no mocks, no solver.

    python tune_autopilot.py                  # all four courses, CMA-ES (the default; the PyPI package cma)
    python tune_autopilot.py --pso            # the particle swarm, then the pattern-search polish
    python tune_autopilot.py -j 8             # 8 worker processes flying laps side by side (default 1; it prints the CPU count)
    python tune_autopilot.py --courses hangar quarry
    python tune_autopilot.py --max-evals 300 --generations 40      # CMA-ES: the evaluation budget, the hard maximum of generations
    python tune_autopilot.py --pso --pso-iterations 0               # PSO: polish the stored settings only (--polish-evals 0: the swarm only)
    python tune_autopilot.py --train-k 0      # the nominal lap alone (the earlier objective; default: 6 training scenarios)
    python tune_autopilot.py --fresh          # start over instead of resuming

Every candidate is scored over the nominal lap and K training scenarios (races/validate.py, seed 1, never the held-out
seed of ./test_autopilot.sh): the mean of its worst ⌈(K+1)/3⌉ laps. The strategies are classes in races/optimizers.py.

Resumable: every generation / iteration is checkpointed to races/runs/tune/<course>.ckpt.json (atomically); a rerun
with the same options continues where it stopped and skips finished courses; other options are refused (use
--fresh). The same seed and options give the same laps whatever -j is.

Every step prints a line; the whole output also goes to races/runs/tune/tune_<date>.log. Results:
races/runs/tune/<course>_converge.csv (every lap), best.json and races/autopilot.json, written after every course,
with the time spent per phase."""
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


IMPORT_PROBE = "import mujoco"
RENDER_PROBE = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\\'1\\'/></worldbody></mujoco>'); "
                "d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 32, 32); r.update_scene(d); r.render()")


def pick_backend(probe: str = IMPORT_PROBE) -> str | None:
    """The MUJOCO_GL with which ``probe`` works here, each tried in a subprocess: ``IMPORT_PROBE`` (the laps need no
    renderer, but the import loads one) or ``RENDER_PROBE`` (a GL context that really renders a frame)."""
    if os.environ.get("MUJOCO_GL") is not None:
        say(f"MUJOCO_GL={os.environ['MUJOCO_GL']!r} (set by you)")
        return os.environ["MUJOCO_GL"]
    for b in BACKENDS:
        env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"} | ({"MUJOCO_GL": b} if b else {})
        what = "renders" if probe == RENDER_PROBE else "imports"
        p = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120)
        if p.returncode == 0:
            say(f"MUJOCO_GL={b or '(unset)'}: mujoco {what}")
            return b
        say(f"MUJOCO_GL={b or '(unset)'}: mujoco does not {what[:-1]} ({(p.stderr.strip().splitlines() or ['?'])[-1][:120]})")
    return None


def bootstrap(out: Path, name: str, probes=(IMPORT_PROBE,)):
    """The log (tee), the source trees on sys.path, a MUJOCO_GL for which the first workable of ``probes`` works;
    returns (the log path, the index of that probe), or None."""
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = sys.stderr = Tee(sys.__stdout__, log.open("w"))
    say(f"repository {ROOT}")
    say(f"python {sys.executable} ({sys.version.split()[0]}); log {log}")
    for p in (ROOT, ROOT / "designs", ROOT / "vegeta-cli" / "src", ROOT / "vegeta-core" / "src",
              ROOT / "vendor" / "ozaru" / "porunga" / "src", ROOT / "vendor" / "ozaru" / "yanus" / "src"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    for k, probe in enumerate(probes):
        b = pick_backend(probe)
        if b is not None:
            break
        if probe == RENDER_PROBE:
            say("no MUJOCO_GL renders here (EGL, OSMesa or GLFW with a display are needed for MuJoCo frames)")
    if b is None:
        say("no MUJOCO_GL works: is mujoco installed in this python?  pip install mujoco  (or ./install_local.sh)")
        return None
    os.environ["MUJOCO_GL"] = b                       # "" stops races.sim's setdefault("egl")
    try:
        import mujoco
        say(f"mujoco {mujoco.__version__}")
        import races  # noqa: F401
    except Exception as e:
        say(f"cannot import mujoco / races ({type(e).__name__}: {e})")
        return None
    return log, k


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if bootstrap(OUT, "tune") is None:
        return 1
    if "--pso" not in argv:
        try:
            import cma
            say(f"cma {cma.__version__} (CMA-ES, the default strategy)")
        except ImportError:
            say("CMA-ES needs the PyPI package cma: pip install cma  (or ./install_local.sh), or use --pso")
            return 1
    from races import tune
    say(f"starting races.tune {' '.join(argv) or '(all courses, full grid)'}")
    t0 = time.time()
    rc = tune.main(argv)
    say(f"finished in {(time.time() - t0) / 60:.1f} min (exit {rc}); results: {OUT / 'best.json'} and {OUT}/<course>.csv")
    return rc


if __name__ == "__main__":
    sys.exit(main())
