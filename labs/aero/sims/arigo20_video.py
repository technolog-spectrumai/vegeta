"""ARIGO-20 manoeuvre videos in MuJoCo (ARIGO-1's six, entered 8 m/s faster). Needs OpenGL + ffmpeg; ``--no-render``
flies only. Restartable. Writes ``labs/aero/sims/output/arigo20_video/*.mp4`` and ``arigo20_video.yaml``."""
import argparse
import os
import sys
import time
from pathlib import Path

if "--no-render" in sys.argv:
    os.environ.pop("MUJOCO_GL", None)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
sys.path.append(str(Path(__file__).resolve().parent))
import arigo1_resume as rs  # noqa: E402
import arigo20_mujoco as m20j  # noqa: E402
from arigo1_video import Recorder  # noqa: E402

rs.GLOBS = ["arigo1*.py", "arigo20*.py"]
OUT = ROOT / "labs" / "aero" / "sims" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manoeuvres", nargs="*", default=list(m20j.MANOEUVRES), choices=m20j.MANOEUVRES)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    vdir = OUT / "arigo20_video"
    vdir.mkdir(parents=True, exist_ok=True)
    path = OUT / "arigo20_video.yaml"
    doc = rs.load(path, a.fresh) or {"aircraft": "ARIGO-20", "note": "arigo20_model; autopilot gains by hand", "runs": {}}

    def ok(key):
        r = doc["runs"].get(key)
        return bool(r) and "error" not in r and (a.no_render or ("video" in r and (ROOT / r["video"]).exists()))
    todo = [m for m in a.manoeuvres if not ok(m)]
    print(f"  {len(a.manoeuvres) - len(todo)} manoeuvres already done (kept), {len(todo)} to run")
    mesh = None if a.no_render or not todo else m20j.export_mesh(vdir / "mesh")
    t0 = time.time()
    for n, name in enumerate(todo, 1):
        sim = m20j.Sim20(mesh=mesh, bandit=(name == "chase"))
        part = vdir / f"{name}.part.mp4"
        rec = None if a.no_render else Recorder(sim, part, tuple(a.size))
        try:
            _, summ = m20j.manoeuvre(name, sim, rec)
        except Exception as exc:
            summ = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        if rec:
            rec.close()
            if "error" not in summ:
                part.replace(vdir / f"{name}.mp4")
                summ["video"] = str((vdir / f"{name}.mp4").relative_to(ROOT))
        doc["runs"][name] = summ
        print(f"  {rs.eta(n, len(todo), t0)} ARIGO-20 {name}: {summ}")
        rs.save(path, doc)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
