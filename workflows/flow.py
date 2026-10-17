"""flow — the one registry of Vegeta's units: the books and their chapters (the workflows), the notebooks, what each
needs on the machine, takes from the others and gives (read by Yanus, ``vegeta-yanus``; ``docs/yanus.md``).

``library.py`` (``wf``, the GUI) and ``kestrel_campaigns.py`` read their chapter maps from here; nothing else holds a
copy. Chapter codes (``K1`` …) follow the order the chapters are declared in. Python, not a configuration file.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from vegeta.yanus import Env, File, Flow, Package, Run, Tool  # noqa: F401
except ImportError:                                            # a source tree without the editable install
    sys.path.insert(0, str(ROOT / "vegeta-yanus" / "src"))
    from vegeta.yanus import Env, File, Flow, Package, Run, Tool  # noqa: F401,E402

flow = Flow("vegeta", root=ROOT)


# ------------------------------------------------------------------------------------------------- the project's hooks
def _kc():
    p = str(ROOT / "notebooks" / "designs")
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
                    design="notebooks/designs/kestrel.py and kestrel_campaigns.py; the notebook 43_kestrel_study reads the runs",
                    variants=("4Q-1", "4Q-2", "4Q-3"), fingerprint=lambda v: _kc().fingerprint(v), compatible=_kestrel_compatible,
                    describe=_kestrel_describe)

kestrel.kind("cfd", title="CFD — the airframe with the propellers as rotor disks (Aeromant / OpenFOAM)", script="workflows/kestrel/airframe_cfd.py",
             workflow="airframe_cfd", needs=[Tool.openfoam(probe=_openfoam), Package("vegeta.aeromant")],
             needs_text="OpenFOAM (openfoam.com or .org); 4 cores recommended", time="screening case 10–20 min, fine case ~1 h",
             options=[("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--quality", ("screening", "fine"), "screening"), ("--processors", None, "4")])
kestrel.kind("fea", title="FEA — the carbon frame and the printed spar–post joint (Talos: Gmsh + CalculiX)", script="workflows/kestrel/structure_fea.py",
             workflow="structure_fea", needs=[Tool("ccx", why="the FEA solver", fix="install CalculiX (docs/installation.md)", version_cmd=("ccx", "-v")),
                                              Package("gmsh"), Package("vegeta.talos")],
             needs_text="CalculiX ccx on the PATH, Gmsh (python module)", time="a frame case a few minutes; the whole frame under an hour on 4 threads",
             options=[("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--threads", None, "4")])

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

flow.add_import("import_legacy", title="import the first CFD campaign's results file into kestrel / airframe-sweep (once)",
                script="workflows/kestrel/import_legacy.py", marker="legacy_")


# ================================================================================================= notebooks
DATA = "notebooks/designs/data"

flow.notebook("38_prop_lab", title="the resolved 9x6 propeller: CFD against Boreas, the calibration Kestrel reads",
              needs=[Package("vegeta.dedalus"), Package("vegeta.boreas"), Package("vegeta.aeromant")],
              gives=[File(f"{DATA}/prop_lab.json", keys=("meta", "calibration")), File(f"{DATA}/reports/38_prop_lab.json")])

flow.notebook("43_kestrel_study", title="Kestrel — the design study (the one notebook to open for Kestrel)",
              needs=[Package("vegeta.dedalus"), Package("vegeta.boreas")],
              takes=[Run("kestrel/airframe-sweep", variant="4Q-1"),
                     Run("kestrel/airframe-sweep", variant="4Q-2", required=True),
                     Run("kestrel/airframe-sweep", variant="4Q-3"),
                     Run("kestrel/frame-and-joint", variant="4Q-2", required=True),
                     File(f"38_prop_lab:{DATA}/prop_lab.json", keys=("calibration",))],
              gives=[File(f"{DATA}/reports/43_kestrel_study.json")],
              screening_env="KESTREL_SCREENING")

flow.notebook("44_kestrel_fast", title="Kestrel fast: candidate designs compared in a minute (analytic, no runs)",
              needs=[Package("vegeta.zephyr"), Package("vegeta.boreas")])
