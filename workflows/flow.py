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
    from vegeta.yanus import AnyOf, Data, Env, File, Flow, Package, Run, Tool  # noqa: F401
except ImportError:                                            # a source tree without the editable install
    sys.path.insert(0, str(ROOT / "vegeta-yanus" / "src"))
    from vegeta.yanus import AnyOf, Data, Env, File, Flow, Package, Run, Tool  # noqa: F401,E402

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
             options=[("--variant", ("4Q-1", "4Q-2", "4Q-3"), "4Q-2"), ("--quality", ("screening", "fine"), "screening"), ("--processors", None, "4")],
             screening=("--quality", "screening"))                                   # auto mode: coarse meshes, few iterations
kestrel.kind("fea", title="FEA — the carbon frame and the printed spar–post joint (Talos: Gmsh + CalculiX)", script="workflows/kestrel/structure_fea.py",
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
    p = str(ROOT / "notebooks" / "designs")
    if p not in sys.path:
        sys.path.insert(0, p)
    import kestrel_xc_campaigns
    return kestrel_xc_campaigns


kestrel_x = flow.book("kestrel-x", "X", title="Kestrel-X-C — the shallow X-wing with the payload module (variants X-C, X-C45, X-CX)",
                      design="notebooks/designs/kestrel_xc.py; the notebook 48_kestrel_xc reads the runs", variants=("X-C", "X-C45", "X-CX"),
                      fingerprint=lambda v: _xc().fingerprint(v), compatible=lambda e, f, ch=None: _xc().compare(e, f),
                      describe=lambda: {**dict(_xc().CFD_CAMPAIGNS), **{f"fea:{k}": v for k, v in _xc().FEA_CAMPAIGNS.items()}})
kestrel_x.kind("cfd", title="CFD — the bare X-C (Aeromant / OpenFOAM)", script="workflows/kestrel_x/airframe_cfd.py", workflow="airframe_cfd",
               needs=[Tool.openfoam(probe=_openfoam), Package("vegeta.aeromant")], needs_text="OpenFOAM; 4 cores recommended", time="screening case 10–20 min, fine case ~1 h",
               options=[("--variant", ("X-C", "X-C45", "X-CX"), "X-C"), ("--quality", ("screening", "fine"), "screening"), ("--processors", None, "4")], screening=("--quality", "screening"))
kestrel_x.kind("fea", title="FEA — one panel's carbon spar in its socket (Talos: Gmsh + CalculiX)", script="workflows/kestrel_x/panel_fea.py", workflow="structure_fea",
               needs=[Tool("ccx", why="the FEA solver", fix="install CalculiX (docs/installation.md)", version_cmd=("ccx", "-v")), Package("gmsh"), Package("vegeta.talos")],
               needs_text="CalculiX ccx on the PATH, Gmsh (python module)", time="minutes", options=[("--variant", ("X-C", "X-C45", "X-CX"), "X-C"), ("--threads", None, "4")],
               screening=("--element-size", "3", "--threads", "2"))
kestrel_x.chapter("airframe-sweep", kind="cfd", campaign="sweep")        # X1
kestrel_x.chapter("airframe-stall", kind="cfd", campaign="stall")        # X2
kestrel_x.chapter("panel", kind="fea", campaign="panel")                 # X3

flow.add_import("import_legacy", title="import the first CFD campaign's results file into kestrel / airframe-sweep (once)",
                script="workflows/kestrel/import_legacy.py", marker="legacy_")


# ================================================================================================= notebooks
DATA = "notebooks/designs/data"
KESTREL_DRIVE = Data(f"{DATA}/kestrel_drive.json", cache=True, note="the Boreas drive tables, ~40 s")
MAPS = {"propellers": f"{DATA}/propeller_maps.json", "edf": f"{DATA}/edf_maps.json", "exotic": f"{DATA}/exotic_propeller_maps.json"}
SECTION_BACKEND = AnyOf("section backend", Package("neuralfoil"), Tool("xfoil"), why="Zephyr's 2-D section polars",
                        fix="pip install 'vegeta-cli[sections]' (neuralfoil), or put xfoil on the PATH; ./install_local.sh installs it")

flow.notebook("38_prop_lab", title="the resolved 9x6 propeller: CFD against Boreas, the calibration Kestrel reads",
              needs=[Package("vegeta.dedalus"), Package("vegeta.boreas"), Package("vegeta.aeromant")],
              gives=[File(f"{DATA}/prop_lab.json", keys=("meta", "calibration")), File(f"{DATA}/reports/38_prop_lab.json")],
              takes=[KESTREL_DRIVE])

flow.notebook("43_kestrel_study", title="Kestrel — the design study (the one notebook to open for Kestrel)",
              needs=[Package("vegeta.dedalus"), Package("vegeta.boreas")],
              takes=[KESTREL_DRIVE, Data(f"{DATA}/kestrel_sections.json", cache=True, note="NeuralFoil polars of the candidate sections"), Run("kestrel/airframe-sweep", variant="4Q-1"),
                     Run("kestrel/airframe-sweep", variant="4Q-2", required=True),
                     Run("kestrel/airframe-sweep", variant="4Q-3"),
                     Run("kestrel/frame-and-joint", variant="4Q-2", required=True),
                     Run("kestrel/wing-interference", variant="4Q-2"),
                     File(f"38_prop_lab:{DATA}/prop_lab.json", keys=("calibration",))],
              gives=[File(f"{DATA}/reports/43_kestrel_study.json")],
              screening_env="KESTREL_SCREENING", auto=True)                       # yanus_auto: runs K1 / K11 in screening quality when they are missing

flow.notebook("45_kestrel_x", title="Kestrel-X — the X-wing sibling: four cantilever panels, no bracing, against Kestrel-4Q",
              needs=[Package("vegeta.boreas"), Package("vegeta.dedalus")],
              takes=[File(f"43_kestrel_study:{DATA}/reports/43_kestrel_study.json"), KESTREL_DRIVE,
                     Data(f"{DATA}/kestrel_sections.json", cache=True, note="NeuralFoil polars of the candidate sections")],
              gives=[File(f"{DATA}/kestrel_x_design.json"), File(f"{DATA}/reports/45_kestrel_x.json")], auto=True)   # yanus_auto: runs notebook 43 for its report

flow.notebook("46_kestrel_tandem", title="Kestrel-Q4T — the tandem-wing Kestrel: Q4T1N-10 against Q4T1P-10, Q4T1N-12 and Kestrel-4Q-2 (notebook 43)",
              needs=[Package("vegeta.boreas"), Package("vegeta.dedalus")],
              takes=[File(f"43_kestrel_study:{DATA}/reports/43_kestrel_study.json"), KESTREL_DRIVE,
                     Data(f"{DATA}/kestrel_drive_10x7_3.3Ah.json", cache=True, note="the 10x7 drive's Boreas table, ~40 s")],
              gives=[File(f"{DATA}/kestrel_tandem_design.json"), File(f"{DATA}/reports/46_kestrel_tandem.json")], auto=True)   # yanus_auto: runs notebook 43 for its report

flow.notebook("48_kestrel_xc", title="Kestrel-X-C — the X-wing made competitive: design, CAD, the CFD and FEA chapters, against X-A, X-B and Kestrel-4Q-2",
              needs=[Package("vegeta.boreas"), Package("vegeta.dedalus")],
              takes=[Run("kestrel-x/airframe-sweep", variant="X-C"), Run("kestrel-x/airframe-stall", variant="X-C"), Run("kestrel-x/panel", variant="X-C"),
                     File(f"43_kestrel_study:{DATA}/reports/43_kestrel_study.json"), KESTREL_DRIVE],
              gives=[File(f"{DATA}/kestrel_xc_design.json"), File(f"{DATA}/reports/48_kestrel_xc.json")], auto=True)

flow.notebook("47_kestrel_benchmark", title="Kestrel benchmark — the family on one table, read from notebooks 43, 45 and 46 (no physics)",
              takes=[File(f"43_kestrel_study:{DATA}/reports/43_kestrel_study.json"), File(f"45_kestrel_x:{DATA}/kestrel_x_design.json"), File(f"46_kestrel_tandem:{DATA}/kestrel_tandem_design.json"),
                     File(f"48_kestrel_xc:{DATA}/kestrel_xc_design.json")],
              gives=[File(f"{DATA}/reports/47_kestrel_benchmark.json")], auto=True)
flow.notebook("49_mini_raptor", title="Mini raptors — Apus Mini, Alcotan, Gavilan on a 5-inch racing stack: first principles, conceptual CAD (no CFD, no FEA)",
              takes=[], gives=[File(f"{DATA}/mini_raptor_design.json"), File(f"{DATA}/reports/49_mini_raptor.json")], auto=True)

flow.notebook("44_kestrel_fast", title="Kestrel fast: candidate designs compared in a minute (analytic, no runs)",
              needs=[Package("vegeta.zephyr"), Package("vegeta.boreas"), SECTION_BACKEND],
              takes=[KESTREL_DRIVE])


# ================================================================================================= the other notebooks
# Their needs (the vegeta packages they import), the files they hand to each other under notebooks/designs/data/ and
# take from each other (all optional: each falls back to its own defaults when the file is missing). Their scratch
# output (_runs/, output/) is not a give. Long CFD/FEA cells stay in the notebooks until they become workflows (todo M.2).
flow.notebook("06_core_revisions", folder="demo", title='Vegeta Core — revisions, evaluations, NOT RUN',
              needs=[Package("vegeta.core"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("07_ai_design_copilot", folder="demo", title='AI design copilot — iterate on a CadQuery design with Claude',
              needs=[Package("vegeta.ai"), Package("vegeta.dedalus"), Package("vegeta.fidia")])
flow.notebook("08_quadcopter", title='Quadcopter — frame, propeller, noise, vibration and life',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("08b_quadcopter_potato", title='Drongo — a potato and a cup of cream for hungry people',
              needs=[Package("vegeta.dedalus")])
flow.notebook("09a_fixed_wing_design", title='Fixed-wing drone — design and performance',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("09b_fixed_wing_durability", title='Fixed-wing drone — durability',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("10_agentic_design", folder="demo", title='Agentic design — a controlled AI campaign',
              needs=[Package("vegeta.ai"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.fidia"), Package("vegeta.talos")])
flow.notebook("11_rover_mechanics", title='Unmanned ground vehicle — a small rover for rugged terrain, mechanically',
              needs=[Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("12_boat_at_sea", title='A boat on the sea — hull, propulsion, structure, life, and the propeller in detail',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("13_submarine", title='Submarine — a small autonomous underwater vehicle: buoyancy and trim, resistance (CFD), pr',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("14_submarine_propeller", title='Submarine propeller — flow in 3D with particles, weak points, vibration and noise spectra,',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("15_fidia_prompt_to_3d", folder="demo", title='Fidia — from a prompt to a 3D model',
              needs=[Package("vegeta.dedalus"), Package("vegeta.fidia")])
flow.notebook("16_robot_dog", title='Robot dog — a quadruped for patrols, stairs and building inspection',
              needs=[Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("17_myropod_persephone", title='Myropods: Persephone — a segmented crawler for chimneys and tight dangerous spaces',
              needs=[Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("18_myropod_cleopatra", title='Myropods: Cleopatra — the agile three-segment walker',
              needs=[Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("19_myropod_apheloria", title='Myropods: Apheloria — the modular walker that rolls into a ball',
              needs=[Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("20_onager_sentinel", title='Project Onager — land robots. Onager Sentinel SX-1, a wheel-leg reconnaissance unit',
              needs=[Package("vegeta.chiron"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("21_onager_atlas", title='Project Onager — Onager Atlas, the forklift',
              needs=[Package("vegeta.chiron"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("22_onager_manus", title='Project Onager — Onager Manus, two manipulator pincers',
              needs=[Package("vegeta.chiron"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("23_onager_sweeper", title='Project Onager — Onager Sweeper, the street cleaner',
              needs=[Package("vegeta.aeromant"), Package("vegeta.chiron"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("24_velutina", title='Velutina — medical aid to the mountains',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("25_air_propeller", title='Air propellers — tractor and pusher, two and three blades: efficiency, CFD, flow movies, b',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(MAPS["propellers"], optional=True), File(f"{DATA}/air_propeller_results.json")])
flow.notebook("25b_ducted_fan", title='The ducted fan (EDF) — efficiency, CFD, noise, vibration, stages',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              takes=[File(f"25_air_propeller:{DATA}/air_propeller_results.json")],
              gives=[File(MAPS["edf"], optional=True)])
flow.notebook("25c_exotic_propellers", title='Exotic propellers — six and twelve blades, tractor and pusher',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus")],
              takes=[File(f"25_air_propeller:{DATA}/air_propeller_results.json")],
              gives=[File(MAPS["exotic"], optional=True)])
flow.notebook("26_merlin", title='MERLIN — the design',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/merlin_design.json")])
flow.notebook("26b_merlin_mission", title='MERLIN — the mission: the race to the fire, the mountain, the smoke',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.core"), Package("vegeta.dedalus")],
              takes=[File(f"26_merlin:{DATA}/merlin_design.json")])
flow.notebook("27_merlin_race", title="MERLIN's propulsor race — can a ducted fan win on the same battery?",
              needs=[],
              takes=[File("25_air_propeller:" + MAPS["propellers"], required=True), File("25b_ducted_fan:" + MAPS["edf"]), File("25c_exotic_propellers:" + MAPS["exotic"]), File(f"26_merlin:{DATA}/merlin_design.json")])
flow.notebook("28_microjet", title='Microjet — a model turbojet from cycle to CFD and FEA',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/microjet.json")])
flow.notebook("28_peregrine", title='PEREGRINE — a falcon-inspired folding-wing farm drone',
              needs=[Package("vegeta.aeromant"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/peregrine_design.json")],
              takes=[File("25_air_propeller:" + MAPS["propellers"], required=True), File("25b_ducted_fan:" + MAPS["edf"]), File("25c_exotic_propellers:" + MAPS["exotic"])])
flow.notebook("29_aguya", title='AGUYA — a turbojet sampler that reaches the fire at 150 m/s',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              takes=[File("25_air_propeller:" + MAPS["propellers"], required=True), File("25b_ducted_fan:" + MAPS["edf"]), File("25c_exotic_propellers:" + MAPS["exotic"]), File(f"28_microjet:{DATA}/microjet.json"), File(f"26_merlin:{DATA}/merlin_design.json")],
              gives=[File(f"{DATA}/aguya_design.json")])
flow.notebook("30_pekari_rover", title='Pekari Rover — a small tracked rover: machine, ground, drive train and track',
              needs=[Package("vegeta.chiron"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("31_nisus", title='NISUS — the observation drone and its computer-vision sibling',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/nisus_design.json")],
              takes=[Data(f"{DATA}/nisus_propulsion.json", cache=True, note="the NISUS propulsion map (Boreas)"), Data(f"{DATA}/nisus_component_sources.md")])
flow.notebook("32_nisus_birds", title='NISUS-Zero — photographing birds up close: the mission software prototype',
              needs=[Package("vegeta.mission")],
              takes=[Data(f"{DATA}/nisus_propulsion.json", cache=True, note="the NISUS propulsion map (Boreas)")])
flow.notebook("33_nisus_plus", title='NISUS+ — the mountain survey drone',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/nisus_plus_design.json")],
              takes=[Data(f"{DATA}/nisus_plus_propulsion.json", cache=True, note="the NISUS+ drive (Boreas)")])
flow.notebook("34_frame_study", title='Tail frames for the mountain aircraft — twin booms against a single tail',
              needs=[Package("vegeta.talos")],
              gives=[File(f"{DATA}/frame_study.json", keys=("status", "verdict"))])
flow.notebook("35_falco", title='FALCO — the tractor mountain aircraft',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/falco_design.json")],
              takes=[Data(f"{DATA}/falco_propulsion.json", cache=True, note="the FALCO drive (Boreas)")])
flow.notebook("36_falco_balsa_wing", title='FALCO — the wing and the tail built up in balsa for the laser cutter',
              needs=[Package("vegeta.atropos"), Package("vegeta.dedalus")],
              gives=[File(f"{DATA}/falco_balsa_wing.json")],
              takes=[Data(f"{DATA}/falco_propulsion.json", cache=True, note="the FALCO drive (Boreas)")])
flow.notebook("37_perseus_rover", title='Perseus — a tracked rover that cuts tree branches',
              needs=[Package("vegeta.chiron"), Package("vegeta.dedalus"), Package("vegeta.talos")])
