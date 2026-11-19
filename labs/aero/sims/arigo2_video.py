"""ARIGO-2 manoeuvre videos in MuJoCo, run by ``arigo2_all.sh``: ARIGO-1's six manoeuvres plus an engine failure
in a full-power climb. Needs OpenGL (MUJOCO_GL=egl or osmesa) and ffmpeg; ``--no-render`` flies them only.
Restartable: a finished video is kept, a half-written one is redone.

Writes ``labs/aero/sims/output/arigo2_video/*.mp4`` and ``arigo2_video.yaml`` — PASTE THE YAML BACK.
"""
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
import arigo2_mujoco as m2j  # noqa: E402
from arigo1_video import Recorder  # noqa: E402

rs.GLOBS = ["arigo1*.py", "arigo2*.py"]
OUT = ROOT / "labs" / "aero" / "sims" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wings", nargs="*", default=["goshawk", "zero", "hybrid"], choices=("goshawk", "zero", "hybrid"))
    ap.add_argument("--manoeuvres", nargs="*", default=list(m2j.MANOEUVRES), choices=m2j.MANOEUVRES)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    vdir = OUT / "arigo2_video"
    vdir.mkdir(parents=True, exist_ok=True)
    path = OUT / "arigo2_video.yaml"
    doc = rs.load(path, a.fresh) or {"aircraft": "ARIGO-2", "note": "arigo1_model (twin) + two motors; autopilot gains by hand", "runs": {}}

    def ok(key):
        r = doc["runs"].get(key)
        return bool(r) and "error" not in r and (a.no_render or ("video" in r and (ROOT / r["video"]).exists()))
    todo = [(w, m) for w in a.wings for m in a.manoeuvres if not ok(f"{w}_{m}")]
    print(f"  {len(a.wings) * len(a.manoeuvres) - len(todo)} manoeuvres already done (kept), {len(todo)} to run")
    t0, meshes = time.time(), {}
    for n, (w, name) in enumerate(todo, 1):
        if not a.no_render and w not in meshes:
            meshes[w] = m2j.export_mesh(vdir / "mesh", w)
        sim = m2j.Sim2(w, mesh=meshes.get(w), bandit=(name == "chase"))
        key = f"{w}_{name}"
        part = vdir / f"{key}.part.mp4"
        rec = None if a.no_render else Recorder(sim, part, tuple(a.size))
        t1 = time.time()
        try:
            _, summ = m2j.manoeuvre(name, sim, rec)
        except Exception as exc:
            summ = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        if rec:
            rec.close()
            if "error" not in summ:
                part.replace(vdir / f"{key}.mp4")
                summ["video"] = str((vdir / f"{key}.mp4").relative_to(ROOT))
        doc["runs"][key] = summ
        print(f"  {rs.eta(n, len(todo), t0)} {key}: {time.time() - t1:.1f} s  {summ}")
        rs.save(path, doc)
    print(f"-> {path}  videos in {vdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
