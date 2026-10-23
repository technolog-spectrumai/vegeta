"""Sonic (choked) blowdown through a 10 mm pipe — a speculative study in the aero lab's limbo (exp/).
Notebook ``50_sonic_choked_flow``; ``python labs/aero/exp/sonic_flow.py --help`` re-runs it from a shell.

A 1 L supply container holds air or CO2 at 50 bar and 20 °C. A valve 20 mm down a 10 mm bore, 100 mm long pipe opens at
t = 0 and stays open for the 10 ms the flow is simulated; the pipe blows into still air at 1 atm. The pressure ratio
(50 : 1) is far above the critical one (~1.9), so the pipe chokes: its exit runs at Mach 1 and the jet outside is
under-expanded (a barrel shock and a Mach disk). The container empties and cools while it does.

Three pieces, one geometry (``Rig``):
- **CAD** (Dedalus / CadQuery): ``SupplyRig`` the hardware (container, pipe, ball valve, flange), ``FluidVolume`` the gas
  the CFD solves in (container + bore), revolved from the same (x, r) profile the mesh is built from — its measured
  volume checks the mesh's container volume.
- **CFD** (OpenFOAM v2412 through Aeromant's ``OpenFOAMEnvironment``): ``write_case`` an axisymmetric 5° wedge,
  ``rhoCentralFoam`` (density based, Kurganov–Tadmor central fluxes: the solver OpenFOAM ships for shocks and
  under-expanded jets), k-omega SST with wall functions, the valve as a diaphragm removed at t = 0 (``setFields``).
  Function objects record the exit plane (mass flow, area- and mass-averaged velocity, p, T, Mach), the valve plane and
  the container (mean p, T, gas mass) every time step.
- **0-D model** (``blowdown``): the container emptying through a choked throat, ideal gas (the CFD's thermo) and real gas
  (CoolProp, Helmholtz-energy equations of state) — CO2 at 50 bar is far from ideal (Z ≈ 0.64) and its expansion to
  the throat runs into the saturation dome.

Values are explicit; the ones the brief did not give (container volume, pipe length, valve place, ambient) are in
``Rig`` and ``Conditions`` with their defaults visible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RU = 8314.462618                       # J/(kmol K)


# -- gases ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gas:
    """A calorically perfect gas for the CFD (OpenFOAM ``hConst`` + ``perfectGas`` + ``sutherland``) and its CoolProp name
    for the real-gas check."""
    name: str
    mol_weight: float      # kg/kmol
    cp: float              # J/(kg K), at ~300 K
    As: float              # Sutherland coefficient, kg/(m s sqrt(K))
    Ts: float              # Sutherland temperature, K
    Pr: float
    coolprop: str

    @property
    def R(self) -> float:
        return RU / self.mol_weight

    @property
    def gamma(self) -> float:
        return self.cp / (self.cp - self.R)


AIR = Gas("air", 28.96, 1005.0, 1.458e-6, 110.4, 0.71, "Air")
# CO2: cp of the ideal gas at 300 K (NIST: 846 J/kg K, gamma 1.29); Sutherland from mu = 1.37e-5 Pa s at 273 K, S = 222 K
CO2 = Gas("co2", 44.01, 846.0, 1.503e-6, 222.0, 0.77, "CO2")
GASES = {g.name: g for g in (AIR, CO2)}


# -- geometry ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rig:
    """The blowdown rig, mm. x along the pipe axis: the container ends at x = 0, the pipe runs to ``pipe_length``, the
    still air outside starts at the pipe exit."""
    pipe_bore: float = 10.0            # the brief: a 10 mm diameter pipe
    pipe_length: float = 100.0         # container wall to pipe exit
    pipe_wall: float = 2.0
    valve_x: float = 20.0              # the valve's seat, from the container
    container_volume_l: float = 1.0
    container_bore: float = 100.0
    container_wall: float = 5.0
    ambient_length: float = 200.0      # the still air the jet blows into (CFD domain beyond the exit)
    ambient_radius: float = 100.0

    @property
    def container_length(self) -> float:
        return self.container_volume_l * 1e6 / (math.pi / 4 * self.container_bore ** 2)

    @property
    def exit_area_m2(self) -> float:
        return math.pi / 4 * (self.pipe_bore * 1e-3) ** 2

    def profile(self) -> list[tuple[float, float]]:
        """The gas inside container + pipe as a closed (x, r) polygon, mm (the axis is r = 0)."""
        L, R, Rp = self.container_length, self.container_bore / 2, self.pipe_bore / 2
        return [(-L, 0.0), (-L, R), (0.0, R), (0.0, Rp), (self.pipe_length, Rp), (self.pipe_length, 0.0)]


@dataclass(frozen=True)
class Conditions:
    p0: float = 50e5                   # container, absolute
    T0: float = 293.15
    p_amb: float = 101325.0
    T_amb: float = 293.15
    end_time: float = 10e-3            # the brief: 10 ms of flow, valve open throughout
    write_interval: float = 0.5e-3     # fields for the snapshots
    cfl: float = 0.4                   # on the in-plane cell size (see control_dict)


# -- CAD (Dedalus) ------------------------------------------------------------------------------------------------------
def _designs():
    import cadquery as cq
    from vegeta.dedalus import Design, Parameter

    def _rig(p) -> Rig:
        return Rig(pipe_bore=p["pipe_bore"], pipe_length=p["pipe_length"], valve_x=p["valve_x"],
                   container_volume_l=p["container_volume_l"], container_bore=p["container_bore"])

    params = [
        Parameter("pipe_bore", Rig.pipe_bore, "mm", min=1, description="pipe inner diameter"),
        Parameter("pipe_length", Rig.pipe_length, "mm", min=10, description="container wall to pipe exit"),
        Parameter("valve_x", Rig.valve_x, "mm", min=5, description="valve seat from the container"),
        Parameter("container_volume_l", Rig.container_volume_l, "L", min=0.01, description="supply container volume"),
        Parameter("container_bore", Rig.container_bore, "mm", min=20, description="container inner diameter"),
    ]

    def along_x(wp):                     # CadQuery revolves about the workplane's X axis: our pipe axis
        return wp

    class FluidVolume(Design):
        """The gas the CFD solves for: container and pipe bore, revolved from ``Rig.profile`` (mm)."""
        parameters = params

        def build(self, p):
            pts = _rig(p).profile()
            return cq.Workplane("XY").polyline(pts).close().revolve(360, (0, 0, 0), (1, 0, 0))

    class SupplyRig(Design):
        """The hardware: a flat-ended steel container, the pipe, a ball valve (body, stem, lever) and an exit flange."""
        parameters = params

        def build(self, p):
            r = _rig(p)
            L, R, w = r.container_length, r.container_bore / 2, r.container_wall
            Rp, Ro = r.pipe_bore / 2, r.pipe_bore / 2 + r.pipe_wall
            ax = cq.Workplane("YZ")
            shell = ax.workplane(offset=-L - w).circle(R + w).extrude(L + 2 * w)
            shell = shell.cut(ax.workplane(offset=-L).circle(R).extrude(L))
            pipe = ax.workplane(offset=-w).circle(Ro).extrude(r.pipe_length + w)
            # ball valve: a square body around the seat, a stem on +z and a lever along +x
            vb = 2.6 * Ro
            body = cq.Workplane("XY").box(2.2 * vb, 2 * vb, 2 * vb).translate((r.valve_x, 0, 0))
            stem = cq.Workplane("XY").workplane(offset=vb).center(r.valve_x, 0).circle(0.35 * Ro).extrude(0.8 * vb)
            lever = cq.Workplane("XY").box(5 * vb, 0.5 * vb, 0.25 * vb).translate((r.valve_x + 2.2 * vb, 0, 1.8 * vb + 0.12 * vb))
            flange = ax.workplane(offset=r.pipe_length - 3).circle(2.2 * Ro).extrude(3)
            solid = shell.union(pipe).union(body).union(stem).union(lever).union(flange)
            bore = ax.workplane(offset=-w - 1).circle(Rp).extrude(r.pipe_length + w + 2)
            return solid.cut(bore)

    return FluidVolume, SupplyRig


def make_cad(out: Path, rig: Rig = Rig()) -> dict:
    """STEP + STL of the rig and of the fluid volume into ``out``; returns their measurements."""
    FluidVolume, SupplyRig = _designs()
    kw = dict(pipe_bore=rig.pipe_bore, pipe_length=rig.pipe_length, valve_x=rig.valve_x,
              container_volume_l=rig.container_volume_l, container_bore=rig.container_bore)
    out = Path(out)
    report = {}
    for name, d in (("supply_rig", SupplyRig()), ("fluid_volume", FluidVolume())):
        g = d.generate(**kw)
        g.export_step(out / f"{name}.step")
        g.export_stl(out / f"{name}.stl", tolerance=0.05)
        report[name] = g.measure()
    fluid_l = report["fluid_volume"]["volume"] / 1e6
    expected = rig.container_volume_l + math.pi / 4 * rig.pipe_bore ** 2 * rig.pipe_length / 1e6
    report["fluid_volume_l"] = fluid_l
    report["fluid_volume_expected_l"] = expected
    return report


# -- OpenFOAM case ----------------------------------------------------------------------------------------------------
HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  written by labs/aero/exp/sonic_flow.py (Vegeta) for OpenFOAM v2412 (openfoam.com)
\\*---------------------------------------------------------------------------*/
FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}
"""
WEDGE_DEG = 5.0


def _grading(length, d_start, d_end):
    """Cells and blockMesh expansion ratio (last / first cell) for a geometric spacing from d_start to d_end over
    ``length``."""
    if abs(d_end - d_start) < 1e-12:
        return max(1, round(length / d_start)), 1.0
    q = (length - d_start) / (length - d_end)        # sum of a geometric series: L = (d_end q - d_start) / (q - 1)
    n = max(1, round(1 + math.log(d_end / d_start) / math.log(q)))
    return n, d_end / d_start


@dataclass(frozen=True)
class Mesh:
    """Cell sizes, mm."""
    pipe: float = 0.5                  # bore: radial and axial (20 cells across the diameter)
    container_far: float = 5.0         # at the container's far end and outer wall
    ambient_far: float = 4.0

    def blocks(self, rig: Rig):
        L, Rr, Rp = rig.container_length, rig.container_bore / 2, rig.pipe_bore / 2
        nr_pipe = round(Rp / self.pipe)
        n_cont_x, e_cont_x = _grading(L, self.container_far, self.pipe)            # coarse at the far end -> fine at x=0
        n_cont_r, e_cont_r = _grading(Rr - Rp, self.pipe, self.container_far)
        n_up = round(rig.valve_x / self.pipe)
        n_dn = round((rig.pipe_length - rig.valve_x) / self.pipe)
        n_amb_x, e_amb_x = _grading(rig.ambient_length, self.pipe, self.ambient_far)
        n_amb_r, e_amb_r = _grading(rig.ambient_radius - Rp, self.pipe, self.ambient_far)
        x0, x1, x2, x3, x4 = -L, 0.0, rig.valve_x, rig.pipe_length, rig.pipe_length + rig.ambient_length
        return [  # (x0, x1, r0, r1, nx, nr, ex, er, zone)
            (x0, x1, 0.0, Rp, n_cont_x, nr_pipe, e_cont_x, 1.0, "container"),
            (x0, x1, Rp, Rr, n_cont_x, n_cont_r, e_cont_x, e_cont_r, "container"),
            (x1, x2, 0.0, Rp, n_up, nr_pipe, 1.0, 1.0, "pipeUpstream"),
            (x2, x3, 0.0, Rp, n_dn, nr_pipe, 1.0, 1.0, "pipeDownstream"),
            (x3, x4, 0.0, Rp, n_amb_x, nr_pipe, e_amb_x, 1.0, "ambient"),
            (x3, x4, Rp, rig.ambient_radius, n_amb_x, n_amb_r, e_amb_x, e_amb_r, "ambient"),
        ]


def block_mesh_dict(rig: Rig, mesh: Mesh) -> str:
    h = math.radians(WEDGE_DEG / 2)
    verts, index = [], {}

    def v(x, r, side):                 # side -1 back, +1 front; on the axis both sides are one point
        key = (round(x, 9), round(r, 9), 0 if r == 0 else side)
        if key not in index:
            index[key] = len(verts)
            verts.append((x, r * math.cos(h), side * r * math.sin(h) if r else 0.0))
        return index[key]

    blocks, patches = [], {k: [] for k in ("containerWall", "pipeWall", "exitPlate", "ambient", "outlet", "front", "back")}
    for x0, x1, r0, r1, nx, nr, ex, er, zone in mesh.blocks(rig):
        b = [v(x0, r0, -1), v(x1, r0, -1), v(x1, r1, -1), v(x0, r1, -1),
             v(x0, r0, 1), v(x1, r0, 1), v(x1, r1, 1), v(x0, r1, 1)]
        blocks.append(f"    hex ({' '.join(map(str, b))}) {zone} ({nx} {nr} 1) simpleGrading ({ex:.6g} {er:.6g} 1)")
        patches["back"].append((b[0], b[1], b[2], b[3]))
        patches["front"].append((b[4], b[7], b[6], b[5]))
        xmin_face, xmax_face, rmax_face = (b[0], b[4], b[7], b[3]), (b[1], b[2], b[6], b[5]), (b[3], b[7], b[6], b[2])
        if zone == "container":
            patches["containerWall"].append(xmin_face)
            if r0 > 0:
                patches["containerWall"] += [rmax_face, xmax_face]
        elif zone.startswith("pipe"):
            patches["pipeWall"].append(rmax_face)
        else:
            patches["outlet"].append(xmax_face)
            if r0 > 0:
                patches["exitPlate"].append(xmin_face)
                patches["ambient"].append(rmax_face)
    types = dict(containerWall="wall", pipeWall="wall", exitPlate="wall", ambient="patch", outlet="patch",
                 front="wedge", back="wedge")
    out = [HEADER.format(cls="dictionary", obj="blockMeshDict"), "mergeType points;   // wedge: collapsed axis edges\n",
           "scale 0.001;\n", "vertices\n("]
    out += [f"    ({x:.9g} {y:.9g} {z:.9g})" for x, y, z in verts]
    out += [");\n", "blocks\n("] + blocks + [");\n", "edges ();\n", "boundary\n("]
    for name, faces in patches.items():
        out.append(f"    {name}\n    {{\n        type {types[name]};\n        faces\n        (")
        out += [f"            ({' '.join(map(str, f))})" for f in faces]
        out.append("        );\n    }")
    out.append(");\n")
    return "\n".join(out)


WALLS = "containerWall|pipeWall|exitPlate"


def _field(obj, dims, internal, boundary, cls="volScalarField"):
    body = [HEADER.format(cls=cls, obj=obj), f"dimensions {dims};", f"internalField uniform {internal};", "boundaryField\n{"]
    for patch, entries in boundary.items():
        body.append(f'    "{patch}"\n    {{')
        body += [f"        {k} {v};" for k, v in entries.items()]
        body.append("    }")
    body.append('    "front|back" { type wedge; }\n}')
    return "\n".join(body) + "\n"


def initial_fields(gas: Gas, cond: Conditions) -> dict[str, str]:
    g = f"{gas.gamma:.6g}"
    k0, om0 = 1e-3, 100.0
    return {
        "p": _field("p", "[1 -1 -2 0 0 0 0]", cond.p_amb, {
            WALLS: dict(type="zeroGradient"),
            "ambient": dict(type="totalPressure", p0=f"uniform {cond.p_amb}", psi="thermo:psi", rho="none", gamma=g,
                            value=f"uniform {cond.p_amb}"),
            "outlet": dict(type="waveTransmissive", field="p", psi="thermo:psi", gamma=g, fieldInf=cond.p_amb,
                           lInf=0.2, value=f"uniform {cond.p_amb}")}),
        "T": _field("T", "[0 0 0 1 0 0 0]", cond.T_amb, {
            WALLS: dict(type="zeroGradient"),                 # 10 ms: the walls take no heat to speak of
            "ambient|outlet": dict(type="inletOutlet", inletValue=f"uniform {cond.T_amb}", value=f"uniform {cond.T_amb}")}),
        "U": _field("U", "[0 1 -1 0 0 0 0]", "(0 0 0)", {
            WALLS: dict(type="noSlip"),
            "ambient": dict(type="pressureInletOutletVelocity", value="uniform (0 0 0)"),
            "outlet": dict(type="inletOutlet", inletValue="uniform (0 0 0)", value="uniform (0 0 0)")}, "volVectorField"),
        "k": _field("k", "[0 2 -2 0 0 0 0]", k0, {
            WALLS: dict(type="kqRWallFunction", value=f"uniform {k0}"),
            "ambient|outlet": dict(type="inletOutlet", inletValue=f"uniform {k0}", value=f"uniform {k0}")}),
        "omega": _field("omega", "[0 0 -1 0 0 0 0]", om0, {
            WALLS: dict(type="omegaWallFunction", value=f"uniform {om0}"),
            "ambient|outlet": dict(type="inletOutlet", inletValue=f"uniform {om0}", value=f"uniform {om0}")}),
        "nut": _field("nut", "[0 2 -1 0 0 0 0]", 0, {
            WALLS: dict(type="nutkWallFunction", value="uniform 0"),
            "ambient|outlet": dict(type="calculated", value="uniform 0")}),
        "alphat": _field("alphat", "[1 -1 -1 0 0 0 0]", 0, {
            WALLS: dict(type="compressible::alphatWallFunction", Prt=0.85, value="uniform 0"),
            "ambient|outlet": dict(type="calculated", value="uniform 0")}),
    }


def thermophysical(gas: Gas) -> str:
    return HEADER.format(cls="dictionary", obj="thermophysicalProperties") + f"""
thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}
mixture
{{
    specie         {{ molWeight {gas.mol_weight}; }}
    thermodynamics {{ Cp {gas.cp}; Hf 0; }}
    transport      {{ As {gas.As}; Ts {gas.Ts}; Pr {gas.Pr}; }}
}}
"""


def time_step(gas: Gas, cond: Conditions, mesh: Mesh) -> float:
    """A fixed step from the in-plane CFL: the fastest wave is the gas expanded to nothing (sqrt(2 cp T0), the jet
    core's limit) plus the stagnation sound speed. rhoCentralFoam's own Courant number counts the 5° wedge faces, whose
    spacing at the axis is r * 0.087: it would cut the step tenfold for no accuracy."""
    wave = math.sqrt(2 * gas.cp * cond.T0) + math.sqrt(gas.gamma * gas.R * cond.T0)
    return cond.cfl * mesh.pipe * 1e-3 / wave


def control_dict(rig: Rig, cond: Conditions, dt: float) -> str:
    xe = rig.pipe_length * 1e-3
    probes = [(xe - 0.25e-3, 0, 0), (xe + 1e-3 * rig.pipe_bore, 0, 0), (rig.valve_x * 1e-3 + 2e-3, 0, 0),
              (-rig.container_length * 0.5e-3, 0, 0)]
    plane = lambda name: f"""
    {name}
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      faceZone;
        name            {name};
        operation       sum;
        fields          (phi);
        writeFields     false;
        log             false;
        writeControl    timeStep;
        writeInterval   1;
    }}
    {name}Area
    {{
        ${name};
        operation       areaAverage;
        fields          (p T U rho Ma);
    }}
    {name}Mass
    {{
        ${name};
        operation       weightedAverage;
        weightField     phi;
        fields          (U T Ma);
    }}"""
    return HEADER.format(cls="dictionary", obj="controlDict") + f"""
application     rhoCentralFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {cond.end_time:.9g};
deltaT          {dt:.6g};
writeControl    adjustable;
writeInterval   {cond.write_interval:.9g};
purgeWrite      0;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;
adjustTimeStep  no;

functions
{{
    Ma
    {{
        type            MachNo;
        libs            (fieldFunctionObjects);
        executeControl  timeStep;
        writeControl    writeTime;
    }}
{plane("exitPlane")}
{plane("valvePlane")}
    container
    {{
        type            volFieldValue;
        libs            (fieldFunctionObjects);
        regionType      cellZone;
        name            container;
        operation       volAverage;
        fields          (p T rho);
        writeFields     false;
        log             false;
        writeControl    timeStep;
        writeInterval   1;
    }}
    containerMass
    {{
        $container;
        operation       volIntegrate;
        fields          (rho);
    }}
    probes
    {{
        type            probes;
        libs            (sampling);
        fields          (p T U Ma rho);
        probeLocations  ({' '.join(f'({x:.6g} {y:g} {z:g})' for x, y, z in probes)});
        writeControl    timeStep;
        writeInterval   1;
    }}
}}
"""


SCHEMES = HEADER.format(cls="dictionary", obj="fvSchemes") + """
fluxScheme      Kurganov;
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(tauMC)      Gauss linear;
    div(phi,k)      Gauss limitedLinear 1;
    div(phi,omega)  Gauss limitedLinear 1;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes
{
    default         linear;
    reconstruct(rho) vanLeer;
    reconstruct(U)  vanLeerV;
    reconstruct(T)  vanLeer;
}
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""

SOLUTION = HEADER.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    "(rho|rhoU|rhoE)" { solver diagonal; }
    "(U|e|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-10;
        relTol          0;
    }
}
"""


def write_case(case: Path, gas: Gas, rig: Rig = Rig(), cond: Conditions = Conditions(), mesh: Mesh = Mesh()) -> Path:
    """A ready-to-run rhoCentralFoam case in ``case`` (replaced if it exists)."""
    case = Path(case)
    if case.exists():
        shutil.rmtree(case)
    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True)
    (case / "system/blockMeshDict").write_text(block_mesh_dict(rig, mesh))
    (case / "system/controlDict").write_text(control_dict(rig, cond, time_step(gas, cond, mesh)))
    (case / "system/fvSchemes").write_text(SCHEMES)
    (case / "system/fvSolution").write_text(SOLUTION)
    xv, xe = rig.valve_x * 1e-3, rig.pipe_length * 1e-3
    big = 10.0
    (case / "system/topoSetDict").write_text(HEADER.format(cls="dictionary", obj="topoSetDict") + f"""
actions
(
    {{ name exitPlane;  type faceZoneSet; action new; source planeToFaceZone; point ({xe:.9g} 0 0); normal (1 0 0); option all; }}
    {{ name valvePlane; type faceZoneSet; action new; source planeToFaceZone; point ({xv:.9g} 0 0); normal (1 0 0); option all; }}
);
""")
    # the valve: closed until t = 0, so everything up to its seat starts at the container's state
    (case / "system/setFieldsDict").write_text(HEADER.format(cls="dictionary", obj="setFieldsDict") + f"""
defaultFieldValues (volScalarFieldValue p {cond.p_amb} volScalarFieldValue T {cond.T_amb});
regions
(
    boxToCell
    {{
        box (-{big} -{big} -{big}) ({xv:.9g} {big} {big});
        fieldValues (volScalarFieldValue p {cond.p0} volScalarFieldValue T {cond.T0});
    }}
);
""")
    (case / "constant/thermophysicalProperties").write_text(thermophysical(gas))
    (case / "constant/turbulenceProperties").write_text(HEADER.format(cls="dictionary", obj="turbulenceProperties") +
        "simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs off; }\n")
    for name, text in initial_fields(gas, cond).items():
        (case / "0" / name).write_text(text)
    (case / "sonic_case.json").write_text(json.dumps(
        dict(gas=asdict(gas), rig=asdict(rig), conditions=asdict(cond), mesh=asdict(mesh), wedge_deg=WEDGE_DEG, delta_t=time_step(gas, cond, mesh),
             container_length_mm=rig.container_length, exit_area_m2=rig.exit_area_m2), indent=1))
    (case / "case.foam").touch()
    return case


STEPS = (("blockMesh",), ("checkMesh",), ("topoSet",), ("setFields",), ("rhoCentralFoam",))


def run_case(case: Path, env=None, timeout: float | None = None, steps=STEPS, processors: int = 1) -> list:
    """The pipeline in ``case`` through Aeromant's OpenFOAM environment; logs ``log.<step>``. With ``processors`` > 1
    the solver runs decomposed (scotch) and the written times are reconstructed. Stops at the first failure and
    returns the command records."""
    from vegeta import aeromant
    from vegeta.aeromant._process import run_command
    env = env or aeromant.OpenFOAMEnvironment.detect()
    if processors > 1:
        (Path(case) / "system/decomposeParDict").write_text(HEADER.format(cls="dictionary", obj="decomposeParDict") +
            f"numberOfSubdomains {processors};\nmethod scotch;\n")
        expanded = []
        for argv in steps:
            if argv[0] == "rhoCentralFoam":
                expanded += [("decomposePar", "-force"), ("mpirun", "-np", str(processors), "rhoCentralFoam", "-parallel"),
                             ("reconstructPar",)]
            else:
                expanded.append(argv)
        steps = expanded
    records = []
    for argv in steps:
        log = argv[3] if argv[0] == "mpirun" else argv[0]
        rec = run_command(env.command(list(argv)), case, env=env.env, timeout=timeout, log_path=Path(case) / f"log.{log}")
        records.append(rec)
        if rec.returncode != 0 or "FOAM FATAL" in (rec.stdout + rec.stderr):
            raise RuntimeError(f"{argv[0]} failed in {case}: see {rec.log_file}\n" + "\n".join((rec.stdout + rec.stderr).splitlines()[-25:]))
    return records


# -- reading the results ---------------------------------------------------------------------------------------------
def _dat(path: Path) -> tuple[list[str], np.ndarray]:
    cols, rows = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            if "Time" in line:
                cols = line.lstrip("#").split()
            continue
        if line.strip():
            rows.append([float(t) for t in line.replace("(", " ").replace(")", " ").split()])
    n = min(map(len, rows))
    return cols, np.array([r[:n] for r in rows])


def _fo(case: Path, name: str) -> Path:
    files = sorted((Path(case) / "postProcessing" / name).glob("*/*.dat"), key=lambda p: float(p.parent.name))
    if not files:
        raise FileNotFoundError(f"no postProcessing/{name} in {case}")
    return files[-1] if len(files) == 1 else files   # restarts write one folder each


def _series(case, name):
    f = _fo(case, name)
    parts = [_dat(p)[1] for p in (f if isinstance(f, list) else [f])]
    return np.vstack(parts)


def histories(case: Path) -> dict[str, np.ndarray]:
    """Time histories in SI units. Wedge sums are scaled to the full 360°; velocities are the axial component."""
    full = 360.0 / WEDGE_DEG
    out = {}
    series = {p: [_series(case, p + s) for s in ("", "Area", "Mass")] for p in ("exitPlane", "valvePlane")}
    n = min(len(x) for xs in series.values() for x in xs)     # a running case: files of slightly different lengths
    for plane, (s, a, m) in series.items():    # sum(phi) | Time p T Ux Uy Uz rho Ma | Time Ux Uy Uz T Ma
        key = plane.replace("Plane", "")
        out["t"] = s[:n, 0]
        out[f"{key}_mdot"] = s[:n, 1] * full
        out[f"{key}_p"], out[f"{key}_T"], out[f"{key}_U_area"] = a[:n, 1], a[:n, 2], a[:n, 3]
        out[f"{key}_rho"], out[f"{key}_Ma_area"] = a[:n, 6], a[:n, 7]
        out[f"{key}_U"], out[f"{key}_T_mass"], out[f"{key}_Ma"] = m[:n, 1], m[:n, 4], m[:n, 5]
    n = len(out["t"])
    c, cm = _series(case, "container"), _series(case, "containerMass")
    tc = c[:, 0]
    out["container_p"] = np.interp(out["t"], tc, c[:, 1])
    out["container_T"] = np.interp(out["t"], tc, c[:, 2])
    out["container_rho"] = np.interp(out["t"], tc, c[:, 3])
    out["container_mass"] = np.interp(out["t"], cm[:, 0], cm[:, 1]) * full
    t = out["t"]
    out["released_mass"] = np.concatenate([[0.0], np.cumsum(0.5 * (out["exit_mdot"][1:] + out["exit_mdot"][:-1]) * np.diff(t))])
    return out


def summary(h: dict[str, np.ndarray], gas: Gas, rig: Rig = Rig(), cond: Conditions = Conditions()) -> dict:
    t = h["t"]
    choked = h["exit_Ma"] > 0.95
    first = float(t[np.argmax(choked)]) if choked.any() else None
    late = t > 0.5 * t[-1]
    return dict(
        gas=gas.name, end_time_ms=float(t[-1] * 1e3),
        exit_velocity_peak_m_s=float(h["exit_U"].max()), exit_velocity_end_m_s=float(h["exit_U"][-1]),
        exit_velocity_mean_2nd_half_m_s=float(h["exit_U"][late].mean()),
        exit_mach_end=float(h["exit_Ma"][-1]), exit_pressure_end_bar=float(h["exit_p"][-1] / 1e5),
        exit_temperature_end_K=float(h["exit_T_mass"][-1]),
        mass_flow_peak_kg_s=float(h["exit_mdot"].max()), mass_flow_end_kg_s=float(h["exit_mdot"][-1]),
        released_mass_g=float(h["released_mass"][-1] * 1e3),
        container_pressure_end_bar=float(h["container_p"][-1] / 1e5),
        container_temperature_end_K=float(h["container_T"][-1]),
        container_mass_start_g=float(h["container_mass"][0] * 1e3), container_mass_end_g=float(h["container_mass"][-1] * 1e3),
        first_choked_ms=None if first is None else first * 1e3,
        choked_fraction=float(choked.mean()),
    )


# -- 0-D blowdown ------------------------------------------------------------------------------------------------------
def choked_ideal(gas: Gas, p0, T0):
    """Throat (Mach 1) state from stagnation p0, T0 for the perfect gas: (mass flux kg/m2s, velocity, p*, T*)."""
    g, R = gas.gamma, gas.R
    Ts = T0 * 2 / (g + 1)
    ps = p0 * (2 / (g + 1)) ** (g / (g - 1))
    a = math.sqrt(g * R * Ts)
    return ps / (R * Ts) * a, a, ps, Ts


def choked_real(fluid: str, p0, T0):
    """Throat state for the real fluid (CoolProp): the pressure on the isentrope from (p0, T0) that maximises
    rho * sqrt(2 (h0 - h)). Returns (mass flux, velocity, p*, T*, vapour quality at the throat or None)."""
    from CoolProp.CoolProp import PropsSI
    from scipy.optimize import minimize_scalar
    s0, h0 = PropsSI("S", "P", p0, "T", T0, fluid), PropsSI("H", "P", p0, "T", T0, fluid)

    def G(p):
        h, rho = PropsSI("H", "P", p, "S", s0, fluid), PropsSI("D", "P", p, "S", s0, fluid)
        return rho * math.sqrt(max(2 * (h0 - h), 0.0))
    r = minimize_scalar(lambda x: -G(x * p0), bounds=(0.3, 0.85), method="bounded", options=dict(xatol=1e-6))
    ps = r.x * p0
    h = PropsSI("H", "P", ps, "S", s0, fluid)
    q = PropsSI("Q", "P", ps, "S", s0, fluid)
    return G(ps), math.sqrt(2 * (h0 - h)), ps, PropsSI("T", "P", ps, "S", s0, fluid), (q if 0 <= q <= 1 else None)


def blowdown(gas: Gas, rig: Rig = Rig(), cond: Conditions = Conditions(), real: bool = False, cd: float = 1.0,
             n: int = 400) -> dict[str, np.ndarray]:
    """The container emptying isentropically through a choked throat of the pipe's area (no pipe friction, no
    start-up wave): t, container p, T, mass, mass flow, throat velocity, p*, T*. ``real`` uses CoolProp."""
    V = rig.container_volume_l * 1e-3 + rig.exit_area_m2 * rig.valve_x * 1e-3
    A = rig.exit_area_m2 * cd
    t = np.linspace(0.0, cond.end_time, n)
    if real:
        from CoolProp.CoolProp import PropsSI
        f = gas.coolprop
        s0 = PropsSI("S", "P", cond.p0, "T", cond.T0, f)
        rho = PropsSI("D", "P", cond.p0, "T", cond.T0, f)
        state = lambda rho: (PropsSI("P", "D", rho, "S", s0, f), PropsSI("T", "D", rho, "S", s0, f))
    else:
        rho = cond.p0 / (gas.R * cond.T0)
        state = lambda rho: (cond.p0 * (rho * gas.R * cond.T0 / cond.p0) ** gas.gamma,
                             cond.T0 * (rho * gas.R * cond.T0 / cond.p0) ** (gas.gamma - 1))
    cols = {k: np.zeros(n) for k in ("p", "T", "mass", "mdot", "U", "p_star", "T_star")}
    for i in range(n):
        p, T = state(rho)
        G, U, ps, Ts = (choked_real(gas.coolprop, p, T)[:4] if real else choked_ideal(gas, p, T))
        for k, val in zip(cols, (p, T, rho * V, G * A, U, ps, Ts)):
            cols[k][i] = val
        if i + 1 < n:
            dt = t[i + 1] - t[i]
            # midpoint step on d(rho)/dt = -G A / V
            p2, T2 = state(rho - 0.5 * dt * G * A / V)
            G2 = (choked_real(gas.coolprop, p2, T2)[0] if real else choked_ideal(gas, p2, T2)[0])
            rho -= dt * G2 * A / V
    cols["t"] = t
    cols["released_mass"] = cols["mass"][0] - cols["mass"]
    return cols


# -- shell entry -------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("cad", "case", "run", "report"), help="cad: STEP/STL; case: write the cases; "
                    "run: write and solve them; report: plots and summary from solved cases")
    ap.add_argument("--gas", nargs="+", default=list(GASES), choices=list(GASES))
    ap.add_argument("--out", default=str(HERE / "output" / "sonic_flow"), help="cases and CAD (git-ignored)")
    ap.add_argument("--results", default=str(HERE / "sonic_flow_results"), help="plots and summary (committed)")
    ap.add_argument("--end-time", type=float, default=Conditions.end_time)
    ap.add_argument("--processors", type=int, default=1, help="MPI ranks per case")
    a = ap.parse_args(argv)
    out, cond = Path(a.out), replace(Conditions(), end_time=a.end_time)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "cad":
        print(json.dumps(make_cad(Path(a.results) / "cad"), indent=1, default=str))
    if a.what in ("case", "run"):
        for g in a.gas:
            c = write_case(out / g, GASES[g], cond=cond)
            print("case", c)
            if a.what == "run":
                run_case(c, processors=a.processors)
                print(json.dumps(summary(histories(c), GASES[g]), indent=1))
    if a.what == "report":
        import sonic_flow_report
        sonic_flow_report.report(out, Path(a.results), a.gas, cond)


if __name__ == "__main__":
    sys.exit(main())
