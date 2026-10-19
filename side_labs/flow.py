"""flow — the side labs' registry: the books and their chapters, what each needs on the machine and which chapters
it builds on. Nothing here knows who reads the results.

``library.py`` (``wf``, the GUI) and ``kestrel_campaigns.py`` read their chapter maps from here; nothing else holds a
copy. Chapter codes (``K1`` …) follow the order the chapters are declared in. Python, not a configuration file.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "side_labs") not in sys.path:
    sys.path.insert(0, str(ROOT / "side_labs"))
from registry import AnyOf, Env, Flow, Package, Run, Tool  # noqa: E402,F401

flow = Flow("vegeta", root=ROOT)


# ------------------------------------------------------------------------------------------------- the project's hooks
def _kc():
    p = str(ROOT / "designs")
    if p not in sys.path:
        sys.path.insert(0, p)
    import kestrel_campaigns
    return kestrel_campaigns


def _openfoam() -> str | None:
    from vegeta import aeromant
    e = aeromant.OpenFOAMEnvironment.detect()
    return f"{e.flavor()} {e.version()}"


BLOWN = ("blown", "cruise", "transitions", "all")               # campaigns with the rotor disks: the calibration matters


def _kestrel_compatible(expected, found, chapter=None) -> dict:
    return _kc().compare(expected, found, blown=bool(chapter is not None and chapter.kind.name == "cfd" and chapter.campaign in BLOWN))


def _kestrel_describe() -> dict:
    kc = _kc()
    return {**dict(kc.CFD_CAMPAIGNS), **{f"fea:{k}": v for k, v in kc.FEA_CAMPAIGNS.items()}}


# ================================================================================================= book: kestrel (K)
kestrel = flow.book("kestrel", "K", title="Kestrel-4Q — the zero-stagger biplane quadrotor tailsitter (variants 4Q-1, 4Q-2, 4Q-3)",
                    design="designs/kestrel.py and kestrel_campaigns.py",
                    variants=("4Q-1", "4Q-2", "4Q-3"), fingerprint=lambda v: _kc().fingerprint(v), compatible=_kestrel_compatible,
                    describe=_kestrel_describe)

kestrel.kind("cfd", title="CFD — the airframe with the propellers as rotor disks (Aeromant / OpenFOAM)", script="side_labs/kestrel/airframe_cfd.py",
             workflow="airframe_cfd", needs=[Tool.openfoam(probe=_openfoam), Package("vegeta.aeromant")],
             needs_text="OpenFOAM (openfoam.com or .org); 4 cores recommended", time="screening case 10–20 min, fine case ~1 h",
             options=[("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--quality", ("screening", "fine"), "screening"), ("--processors", None, "4")],
             screening=("--quality", "screening"))                                   # auto mode: coarse meshes, few iterations
kestrel.kind("fea", title="FEA — the carbon frame and the printed spar–post joint (Talos: Gmsh + CalculiX)", script="side_labs/kestrel/structure_fea.py",
             workflow="structure_fea", needs=[Tool("ccx", why="the FEA solver", fix="install CalculiX (docs/installation.md)", version_cmd=("ccx", "-v")),
                                              Package("gmsh"), Package("vegeta.talos")],
             needs_text="CalculiX ccx on the PATH, Gmsh (python module)", time="a frame case a few minutes; the whole frame under an hour on 4 threads",
             options=[("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--threads", None, "4")],
             screening=("--element-size", "4", "--threads", "2"))                    # auto mode: a coarse mesh that fits in memory

kestrel.chapter("airframe-sweep", kind="cfd", campaign="sweep")                                          # K1
kestrel.chapter("airframe-stall", kind="cfd", campaign="stall")                                          # K2
kestrel.chapter("airframe-cruise", kind="cfd", campaign="cruise", takes=[Run("airframe-sweep")])          # K3
kestrel.chapter("airframe-transitions", kind="cfd", campaign="transitions", takes=[Run("airframe-sweep")])  # K4
kestrel.chapter("airframe-blown", kind="cfd", campaign="blown", takes=[Run("airframe-sweep")])            # K5
kestrel.chapter("airframe-sweep-fine", kind="cfd", campaign="sweep-fine")                                # K6
kestrel.chapter("airframe-all", kind="cfd", campaign="all")                                              # K7
kestrel.chapter("sections", kind="cfd", campaign="sections")                                             # K8
kestrel.chapter("frame", kind="fea", campaign="frame")                                                   # K9
kestrel.chapter("joint", kind="fea", campaign="joint", takes=[Run("frame")])                             # K10
kestrel.chapter("frame-and-joint", kind="fea", campaign="all")                                           # K11
kestrel.chapter("wing-interference", kind="cfd", campaign="interference")                                 # K12

# ================================================================================================= book: kestrel-x (X)
def _xc():
    for p in (str(ROOT / "designs"), str(ROOT / "designs" / "exp")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import kestrel_xc_campaigns
    return kestrel_xc_campaigns


kestrel_x = flow.book("kestrel-x", "X", title="Kestrel-X-C — the shallow X-wing with the payload module (variants X-C, X-C45, X-CX)",
                      design="designs/exp/kestrel_xc.py", variants=("X-C", "X-C45", "X-CX"),
                      fingerprint=lambda v: _xc().fingerprint(v), compatible=lambda e, f, ch=None: _xc().compare(e, f),
                      describe=lambda: {**dict(_xc().CFD_CAMPAIGNS), **{f"fea:{k}": v for k, v in _xc().FEA_CAMPAIGNS.items()}})
kestrel_x.kind("cfd", title="CFD — the bare X-C (Aeromant / OpenFOAM)", script="side_labs/kestrel_x/airframe_cfd.py", workflow="airframe_cfd",
               needs=[Tool.openfoam(probe=_openfoam), Package("vegeta.aeromant")], needs_text="OpenFOAM; 4 cores recommended", time="screening case 10–20 min, fine case ~1 h",
               options=[("--variant", ("X-C", "X-C45", "X-CX"), "X-C"), ("--quality", ("screening", "fine"), "screening"), ("--processors", None, "4")], screening=("--quality", "screening"))
kestrel_x.kind("fea", title="FEA — one panel's carbon spar in its socket (Talos: Gmsh + CalculiX)", script="side_labs/kestrel_x/panel_fea.py", workflow="structure_fea",
               needs=[Tool("ccx", why="the FEA solver", fix="install CalculiX (docs/installation.md)", version_cmd=("ccx", "-v")), Package("gmsh"), Package("vegeta.talos")],
               needs_text="CalculiX ccx on the PATH, Gmsh (python module)", time="minutes", options=[("--variant", ("X-C", "X-C45", "X-CX"), "X-C"), ("--threads", None, "4")],
               screening=("--element-size", "3", "--threads", "2"))
kestrel_x.chapter("airframe-sweep", kind="cfd", campaign="sweep")        # X1
kestrel_x.chapter("airframe-stall", kind="cfd", campaign="stall")        # X2
kestrel_x.chapter("panel", kind="fea", campaign="panel")                 # X3

# ================================================================================================= book: races (R)
races = flow.book("races", "R", title="Races — the reference 5-inch quad round four courses in MuJoCo (hangar, industrial zone, farm, quarry)",
                  design="races/ (refquad, courses, rules)", variants=())
races.kind("race", title="Race — the MuJoCo lap and its videos (chase, onboard, overview)", script="side_labs/races/race.py", workflow="race",
           needs=[Package("mujoco")], needs_text="mujoco (pip); ffmpeg for the videos; an OpenGL context (MUJOCO_GL=egl) for camera views, else the replay",
           time="a lap in under a minute; the videos a few minutes", options=[("--quality", ("screening", "fine"), "screening"), ("--collision", ("reset", "count", "stop"), "reset"), ("--safety", None, "0.6"), ("--tilt-max", None, "60"), ("--lookahead", None, "0.25")],
           screening=("--quality", "screening"))
races.chapter("hangar", kind="race", campaign="hangar")               # R1
races.chapter("industrial", kind="race", campaign="industrial")       # R2
races.chapter("farm", kind="race", campaign="farm")                   # R3
races.chapter("quarry", kind="race", campaign="quarry")               # R4


flow.add_import("import_legacy", title="import the first CFD campaign's results file into kestrel / airframe-sweep (once)",
                script="side_labs/kestrel/import_legacy.py", marker="legacy_")


# The laser work (books svarog and fenix) moved to Alabastron on 2026-10-18:
# https://github.com/technolog-spectrumai/alabastron
