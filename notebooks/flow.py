"""flow — the notebooks' declarations for the old Yanus gate (what each needs, takes and gives). Temporary: it goes
with the old Yanus framework (todo S.8–S.12); results will come through Porunga keys instead."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("vegeta_side_labs_flow", ROOT / "side_labs" / "flow.py")
_side = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_side)
from vegeta.yanus import AnyOf, Data, Env, File, Package, Run, Tool  # noqa: E402,F401

flow = _side.flow                                              # the chapters the notebooks still take runs from
kestrel = _side.kestrel


# ================================================================================================= notebooks
DATA = "designs/data"
KESTREL_DRIVE = Data(f"{DATA}/kestrel_drive.json", cache=True, note="the Boreas drive tables, ~40 s")
MAPS = {"propellers": f"{DATA}/propeller_maps.json", "edf": f"{DATA}/edf_maps.json", "exotic": f"{DATA}/exotic_propeller_maps.json"}
SECTION_BACKEND = AnyOf("section backend", Package("neuralfoil"), Tool("xfoil"), why="Zephyr's 2-D section polars",
                        fix="pip install 'vegeta-cli[sections]' (neuralfoil), or put xfoil on the PATH; ./install_local.sh installs it")

flow.notebook("38_prop_lab", folder="exp", title="the resolved 9x6 propeller: CFD against Boreas, the calibration Kestrel reads",
              needs=[Package("vegeta.dedalus"), Package("vegeta.boreas"), Package("vegeta.aeromant")],
              gives=[File(f"{DATA}/prop_lab.json", keys=("meta", "calibration")), File(f"{DATA}/reports/38_prop_lab.json")],
              takes=[KESTREL_DRIVE])

flow.notebook("43_kestrel_study", folder="exp", title="Kestrel — the design study (the one notebook to open for Kestrel)",
              needs=[Package("vegeta.dedalus"), Package("vegeta.boreas")],
              takes=[KESTREL_DRIVE, Data(f"{DATA}/kestrel_sections.json", cache=True, note="NeuralFoil polars of the candidate sections"),
                     File(f"38_prop_lab:{DATA}/prop_lab.json", keys=("calibration",))],
              gives=[File(f"{DATA}/reports/43_kestrel_study.json")],
              screening_env="KESTREL_SCREENING")                                # the side-lab results come from the cache (kestrel_study.inputs_from_cache)

flow.notebook("45_kestrel_x", folder="exp", title="Kestrel-X — the X-wing sibling: four cantilever panels, no bracing, against Kestrel-4Q",
              needs=[Package("vegeta.boreas"), Package("vegeta.dedalus")],
              takes=[File(f"43_kestrel_study:{DATA}/reports/43_kestrel_study.json"), KESTREL_DRIVE,
                     Data(f"{DATA}/kestrel_sections.json", cache=True, note="NeuralFoil polars of the candidate sections")],
              gives=[File(f"{DATA}/kestrel_x_design.json"), File(f"{DATA}/reports/45_kestrel_x.json")], auto=True)   # yanus_auto: runs notebook 43 for its report

flow.notebook("46_kestrel_tandem", folder="exp", title="Kestrel-Q4T — the tandem-wing Kestrel: Q4T1N-10 against Q4T1P-10, Q4T1N-12 and Kestrel-4Q-2 (notebook 43)",
              needs=[Package("vegeta.boreas"), Package("vegeta.dedalus")],
              takes=[File(f"43_kestrel_study:{DATA}/reports/43_kestrel_study.json"), KESTREL_DRIVE,
                     Data(f"{DATA}/kestrel_drive_10x7_3.3Ah.json", cache=True, note="the 10x7 drive's Boreas table, ~40 s")],
              gives=[File(f"{DATA}/kestrel_tandem_design.json"), File(f"{DATA}/reports/46_kestrel_tandem.json")], auto=True)   # yanus_auto: runs notebook 43 for its report

flow.notebook("48_kestrel_xc", folder="exp", title="Kestrel-X-C — the X-wing made competitive: design, CAD, the CFD and FEA chapters, against X-A, X-B and Kestrel-4Q-2",
              needs=[Package("vegeta.boreas"), Package("vegeta.dedalus")],
              takes=[Run("kestrel-x/airframe-sweep", variant="X-C"), Run("kestrel-x/airframe-stall", variant="X-C"), Run("kestrel-x/panel", variant="X-C"),
                     File(f"43_kestrel_study:{DATA}/reports/43_kestrel_study.json"), KESTREL_DRIVE],
              gives=[File(f"{DATA}/kestrel_xc_design.json"), File(f"{DATA}/reports/48_kestrel_xc.json")], auto=True)

flow.notebook("47_kestrel_benchmark", folder="exp", title="Kestrel benchmark — the family on one table, read from notebooks 43, 45 and 46 (no physics)",
              takes=[File(f"43_kestrel_study:{DATA}/reports/43_kestrel_study.json"), File(f"45_kestrel_x:{DATA}/kestrel_x_design.json"), File(f"46_kestrel_tandem:{DATA}/kestrel_tandem_design.json"),
                     File(f"48_kestrel_xc:{DATA}/kestrel_xc_design.json")],
              gives=[File(f"{DATA}/reports/47_kestrel_benchmark.json")], auto=True)
for _n, _slug in (("01_hangar", "hangar"), ("02_industrial", "industrial"), ("03_farm", "farm"), ("04_quarry", "quarry")):
    flow.notebook(_n, folder="races", title=f"Race: {_slug} — the reference 5-inch quad's lap (mock flyer; MuJoCo and videos with GENERATE_VIDEO), exported under races/runs/notebook/",
                  takes=[], gives=[], auto=False)
flow.notebook("race_design_factors", folder="races", title="Race design factors — per course, pooled and worst-case requirements from the exported race runs (no race rerun)",
              takes=[], gives=[File(f"{DATA}/races/race_design_factors.json")], auto=False)
flow.notebook("49_mini_raptor", folder="devices", title="Mini raptors — Apus Mini, Alcotan, Gavilan on a 5-inch racing stack: first principles, conceptual CAD (no CFD, no FEA)",
              takes=[], gives=[File(f"{DATA}/mini_raptor_design.json"), File(f"{DATA}/reports/49_mini_raptor.json")], auto=True)

flow.notebook("44_kestrel_fast", folder="exp", title="Kestrel fast: candidate designs compared in a minute (analytic, no runs)",
              needs=[Package("vegeta.zephyr"), Package("vegeta.boreas"), SECTION_BACKEND],
              takes=[KESTREL_DRIVE])


# ================================================================================================= the other notebooks
# Their needs (the vegeta packages they import), the files they hand to each other under designs/data/ and
# take from each other (all optional: each falls back to its own defaults when the file is missing). Their scratch
# output (_runs/, output/) is not a give. Long CFD/FEA cells stay in the notebooks until they become workflows (todo M.2).
flow.notebook("06_core_revisions", folder="demo", title='Vegeta Core — revisions, evaluations, NOT RUN',
              needs=[Package("vegeta.core"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("07_ai_design_copilot", folder="demo", title='AI design copilot — iterate on a CadQuery design with Claude',
              needs=[Package("vegeta.ai"), Package("vegeta.dedalus"), Package("vegeta.fidia")])
flow.notebook("08_quadcopter", folder="devices", title='Quadcopter — frame, propeller, noise, vibration and life',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("08b_quadcopter_potato", folder="scenarios", title='Drongo — a potato and a cup of cream for hungry people',
              needs=[Package("vegeta.dedalus")])
flow.notebook("09a_fixed_wing_design", folder="devices", title='Fixed-wing drone — design and performance',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("09b_fixed_wing_durability", folder="devices", title='Fixed-wing drone — durability',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("10_agentic_design", folder="demo", title='Agentic design — a controlled AI campaign',
              needs=[Package("vegeta.ai"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.fidia"), Package("vegeta.talos")])
flow.notebook("11_rover_mechanics", folder="devices", title='Unmanned ground vehicle — a small rover for rugged terrain, mechanically',
              needs=[Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("12_boat_at_sea", folder="devices", title='A boat on the sea — hull, propulsion, structure, life, and the propeller in detail',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("13_submarine", folder="devices", title='Submarine — a small autonomous underwater vehicle: buoyancy and trim, resistance (CFD), pr',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("14_submarine_propeller", folder="parts", title='Submarine propeller — flow in 3D with particles, weak points, vibration and noise spectra,',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("15_fidia_prompt_to_3d", folder="demo", title='Fidia — from a prompt to a 3D model',
              needs=[Package("vegeta.dedalus"), Package("vegeta.fidia")])
flow.notebook("16_robot_dog", folder="devices", title='Robot dog — a quadruped for patrols, stairs and building inspection',
              needs=[Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("17_myropod_persephone", folder="devices", title='Myropods: Persephone — a segmented crawler for chimneys and tight dangerous spaces',
              needs=[Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("18_myropod_cleopatra", folder="devices", title='Myropods: Cleopatra — the agile three-segment walker',
              needs=[Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("19_myropod_apheloria", folder="devices", title='Myropods: Apheloria — the modular walker that rolls into a ball',
              needs=[Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("20_onager_sentinel", folder="devices", title='Project Onager — land robots. Onager Sentinel SX-1, a wheel-leg reconnaissance unit',
              needs=[Package("vegeta.chiron"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("21_onager_atlas", folder="devices", title='Project Onager — Onager Atlas, the forklift',
              needs=[Package("vegeta.chiron"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("22_onager_manus", folder="devices", title='Project Onager — Onager Manus, two manipulator pincers',
              needs=[Package("vegeta.chiron"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("23_onager_sweeper", folder="devices", title='Project Onager — Onager Sweeper, the street cleaner',
              needs=[Package("vegeta.aeromant"), Package("vegeta.chiron"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("24_velutina", folder="devices", title='Velutina — medical aid to the mountains',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.chronos"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")])
flow.notebook("25_air_propeller", folder="parts", title='Air propellers — tractor and pusher, two and three blades: efficiency, CFD, flow movies, b',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(MAPS["propellers"], optional=True), File(f"{DATA}/air_propeller_results.json")])
flow.notebook("25b_ducted_fan", folder="parts", title='The ducted fan (EDF) — efficiency, CFD, noise, vibration, stages',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              takes=[File(f"25_air_propeller:{DATA}/air_propeller_results.json")],
              gives=[File(MAPS["edf"], optional=True)])
flow.notebook("25c_exotic_propellers", folder="parts", title='Exotic propellers — six and twelve blades, tractor and pusher',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus")],
              takes=[File(f"25_air_propeller:{DATA}/air_propeller_results.json")],
              gives=[File(MAPS["exotic"], optional=True)])
flow.notebook("26_merlin", folder="devices", title='MERLIN — the design',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/merlin_design.json")])
flow.notebook("26b_merlin_mission", folder="races", title='MERLIN — the mission: the race to the fire, the mountain, the smoke',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.core"), Package("vegeta.dedalus")],
              takes=[File(f"26_merlin:{DATA}/merlin_design.json")])
flow.notebook("27_merlin_race", folder="races", title="MERLIN's propulsor race — can a ducted fan win on the same battery?",
              needs=[],
              takes=[File("25_air_propeller:" + MAPS["propellers"], required=True), File("25b_ducted_fan:" + MAPS["edf"]), File("25c_exotic_propellers:" + MAPS["exotic"]), File(f"26_merlin:{DATA}/merlin_design.json")])
flow.notebook("28_microjet", folder="exp", title='Microjet — a model turbojet from cycle to CFD and FEA',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/microjet.json")])
flow.notebook("28_peregrine", folder="devices", title='PEREGRINE — a falcon-inspired folding-wing farm drone',
              needs=[Package("vegeta.aeromant"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.mellonia"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/peregrine_design.json")],
              takes=[File("25_air_propeller:" + MAPS["propellers"], required=True), File("25b_ducted_fan:" + MAPS["edf"]), File("25c_exotic_propellers:" + MAPS["exotic"])])
flow.notebook("29_aguya", folder="devices", title='AGUYA — a turbojet sampler that reaches the fire at 150 m/s',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.core"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              takes=[File("25_air_propeller:" + MAPS["propellers"], required=True), File("25b_ducted_fan:" + MAPS["edf"]), File("25c_exotic_propellers:" + MAPS["exotic"]), File(f"28_microjet:{DATA}/microjet.json"), File(f"26_merlin:{DATA}/merlin_design.json")],
              gives=[File(f"{DATA}/aguya_design.json")])
flow.notebook("30_pekari_rover", folder="devices", title='Pekari Rover — a small tracked rover: machine, ground, drive train and track',
              needs=[Package("vegeta.chiron"), Package("vegeta.chronos"), Package("vegeta.dedalus"), Package("vegeta.talos")])
flow.notebook("31_nisus", folder="devices", title='NISUS — the observation drone and its computer-vision sibling',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/nisus_design.json")],
              takes=[Data(f"{DATA}/nisus_propulsion.json", cache=True, note="the NISUS propulsion map (Boreas)"), Data(f"{DATA}/nisus_component_sources.md")])
flow.notebook("32_nisus_birds", folder="scenarios", title='NISUS-Zero — photographing birds up close: the mission software prototype',
              needs=[Package("vegeta.mission")],
              takes=[Data(f"{DATA}/nisus_propulsion.json", cache=True, note="the NISUS propulsion map (Boreas)")])
flow.notebook("33_nisus_plus", folder="devices", title='NISUS+ — the mountain survey drone',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/nisus_plus_design.json")],
              takes=[Data(f"{DATA}/nisus_plus_propulsion.json", cache=True, note="the NISUS+ drive (Boreas)")])
flow.notebook("34_frame_study", folder="parts", title='Tail frames for the mountain aircraft — twin booms against a single tail',
              needs=[Package("vegeta.talos")],
              gives=[File(f"{DATA}/frame_study.json", keys=("status", "verdict"))])
flow.notebook("35_falco", folder="devices", title='FALCO — the tractor mountain aircraft',
              needs=[Package("vegeta.aeromant"), Package("vegeta.boreas"), Package("vegeta.dedalus"), Package("vegeta.talos")],
              gives=[File(f"{DATA}/falco_design.json")],
              takes=[Data(f"{DATA}/falco_propulsion.json", cache=True, note="the FALCO drive (Boreas)")])
flow.notebook("36_falco_balsa_wing", folder="parts", title='FALCO — the wing and the tail built up in balsa for the laser cutter',
              needs=[Package("vegeta.atropos"), Package("vegeta.dedalus")],
              gives=[File(f"{DATA}/falco_balsa_wing.json")],
              takes=[Data(f"{DATA}/falco_propulsion.json", cache=True, note="the FALCO drive (Boreas)")])
flow.notebook("37_perseus_rover", folder="devices", title='Perseus — a tracked rover that cuts tree branches',
              needs=[Package("vegeta.chiron"), Package("vegeta.dedalus"), Package("vegeta.talos")])
