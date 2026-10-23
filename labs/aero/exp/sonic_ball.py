"""A rubber ball in the blowdown pipe (notebook 51 ``51_sonic_ball``): the ``sonic_flow`` rig with a 10 mm rubber ball
sitting tightly in the bore just downstream of the valve. The valve opens at t = 0. The gas drives the ball down the pipe,
and the ball shapes the flow in turn: it is a moving seal (a piston), so nothing passes it. The gas behind it expands
as it accelerates, and the air ahead is compressed and pushed out of the exit by a shock running ahead of the ball.

- **CAD** (Dedalus): ``BallRig`` the rig with the ball in place.
- **CFD** (OpenFOAM v2412): ``rhoCentralFoam`` as in ``sonic_flow`` (density based, central fluxes: it takes the 50 : 1
  jump at the valve, where the pressure-based ``rhoPimpleFoam`` diverged on the first step), on a moving mesh: a
  ``dynamicMotionSolverFvMesh`` with ``displacementLayeredMotion``. It is 1-D: the bore cells behind the ball
  stretch linearly from the valve plane, the cells ahead squeeze linearly to the exit plane, and nothing else moves.
  The ball is one patch (rear and front face) with a ``sixDoFRigidBodyDisplacement`` boundary: a rigid body of the
  ball's mass, constrained to slide along the axis, moved each step by the pressure and viscous forces OpenFOAM
  integrates on it. The run goes in short chunks and stops when the ball's front is near the exit
  (``stop_front_mm``): past that the squeezed cells would collapse.
- **0-D gun model** (``gun``): the container feeds a lumped gas volume behind the ball through a choked throat, the ball
  obeys m a = A (p_behind - p_ahead) - friction, and the air ahead is a piston-driven shock. Friction (a tight rubber
  ball) lives here only. The CFD ball slides freely: a constant friction force cannot hold a body at rest in a
  ``sixDoF`` model, and at 50 bar it is a few percent of the pressure force (see ``gun(friction=...)``).

Modelling choice, stated: the ball seals the bore, so for the gas it is a piston of the bore's area. The CFD gives it
flat faces 10 mm apart (the ball's diameter). The axial pressure force on a sealing body depends on the bore area only,
not on the face shape.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

import sonic_flow as sf
from sonic_flow import HEADER, HERE, WEDGE_DEG, Conditions, Gas, Mesh, Rig, _field, _grading


@dataclass(frozen=True)
class Ball:
    """The rubber projectile sealing the bore: a ball (``shape="sphere"``) or a cylinder with a hemispherical or
    paraboloidal head (``"hemisphere"``, ``"paraboloid"``) of total ``length``.

    While it seals the bore the gas sees only its rear face and its mass, so the CFD gives every shape the same sealing
    slug one diameter long (``piston``) and the shape's own mass. The head matters for the contact length (friction)
    and for the flight after the muzzle. Heads: the hemisphere is D/2 long; the paraboloid is D long (min(D, L - D/2)
    for the shortest bodies, so at least half a diameter of cylinder seals)."""
    diameter: float = 10.0             # mm: the bore, a tight fit
    density: float = 1100.0            # kg/m3, solid rubber
    rear_x: float = 25.0               # mm from the container: 5 mm after the valve seat
    stop_front_mm: float = 96.0        # end the CFD when the slug front reaches this (pipe exit at 100)
    shape: str = "sphere"              # sphere | hemisphere | paraboloid
    length: float = 0.0                # mm, total length of a cylinder + head (not used for the sphere)
    mu: float = 0.7                    # rubber on steel, sliding
    contact_pressure: float = 0.15e6   # Pa, from the interference fit: gives the 10 mm ball ~10 N (3 mm band)
    sphere_band: float = 3.0           # mm, contact band of a squeezed ball

    @property
    def L(self) -> float:
        return self.diameter if self.shape == "sphere" else self.length

    @property
    def head_length(self) -> float:
        D = self.diameter
        return {"sphere": D / 2, "hemisphere": D / 2, "paraboloid": min(D, self.length - D / 2)}[self.shape]

    @property
    def contact_length(self) -> float:
        """mm of the body pressed on the bore wall: the cylinder (or the ball's band)."""
        return self.sphere_band if self.shape == "sphere" else self.length - self.head_length

    @property
    def volume(self) -> float:
        """m3"""
        R, D = self.diameter * 0.5e-3, self.diameter * 1e-3
        if self.shape == "sphere":
            return math.pi / 6 * D ** 3
        cyl = math.pi * R * R * self.contact_length * 1e-3
        head = 2 / 3 * math.pi * R ** 3 if self.shape == "hemisphere" else 0.5 * math.pi * R * R * self.head_length * 1e-3
        return cyl + head

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def friction(self) -> float:
        """N, sliding friction of the squeezed rubber: mu * contact pressure * contact area (0-D model and the
        CFD-plus-friction estimate; the CFD body itself slides freely)."""
        return self.mu * self.contact_pressure * math.pi * self.diameter * 1e-3 * self.contact_length * 1e-3

    def centre(self, rig: Rig) -> float:
        return self.rear_x + self.diameter / 2


def projectile(head: str, l_over_d: float, diameter: float = 10.0, **kw) -> Ball:
    return Ball(diameter=diameter, shape=head, length=l_over_d * diameter, **kw)


# -- CAD ---------------------------------------------------------------------------------------------------------------
def make_cad(out: Path, rig: Rig = Rig(), ball: Ball = Ball()) -> dict:
    """STEP + STL of the rig with the ball, and of the ball alone."""
    import cadquery as cq
    from vegeta.dedalus import Design, Parameter
    _, SupplyRig = sf._designs()
    xc = ball.centre(rig)

    class BallRig(Design):
        """The supply rig with the rubber ball sitting in the bore."""
        parameters = [Parameter("ball_diameter", ball.diameter, "mm", min=1), Parameter("ball_x", xc, "mm")]

        def build(self, p):
            r = SupplyRig().build(SupplyRig().resolve())
            return r.union(cq.Workplane("XY").sphere(p["ball_diameter"] / 2).translate((p["ball_x"], 0, 0)))

    class RubberBall(Design):
        parameters = [Parameter("ball_diameter", ball.diameter, "mm", min=1)]

        def build(self, p):
            return cq.Workplane("XY").sphere(p["ball_diameter"] / 2)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, d in (("ball_rig", BallRig()), ("rubber_ball", RubberBall())):
        g = d.generate()
        g.export_step(out / f"{name}.step")
        g.export_stl(out / f"{name}.stl", tolerance=0.05)
        report[name] = g.measure()
    report["ball_mass_g"] = ball.mass * 1e3
    report["ball_mass_from_cad_g"] = report["rubber_ball"]["volume"] * 1e-9 * ball.density * 1e3
    return report


# -- mesh --------------------------------------------------------------------------------------------------------------
def blocks(rig: Rig, mesh: Mesh, ball: Ball):
    """Like ``sonic_flow.Mesh.blocks``, with the bore cut at the ball: no cells between its rear and front faces."""
    L, Rr, Rp = rig.container_length, rig.container_bore / 2, rig.pipe_bore / 2
    d = mesh.pipe
    nr = round(Rp / d)
    n_cx, e_cx = _grading(L, mesh.container_far, d)
    n_cr, e_cr = _grading(Rr - Rp, d, mesh.container_far)
    n_ax, e_ax = _grading(rig.ambient_length, d, mesh.ambient_far)
    n_ar, e_ar = _grading(rig.ambient_radius - Rp, d, mesh.ambient_far)
    xr, xf = ball.rear_x, ball.rear_x + ball.diameter
    n = lambda a, b: max(2, round((b - a) / d))
    xa = rig.pipe_length + rig.ambient_length
    return [
        (-L, 0.0, 0.0, Rp, n_cx, nr, e_cx, 1.0, "container"),
        (-L, 0.0, Rp, Rr, n_cx, n_cr, e_cx, e_cr, "container"),
        # the whole bore behind the ball is one moving zone, from the fixed pipe inlet (x = 0): its cells stretch
        # about 4x by the time the ball reaches the exit (from the valve alone it would be 14x)
        (0.0, rig.valve_x, 0.0, Rp, n(0, rig.valve_x), nr, 1.0, 1.0, "pipeBehindBall"),
        (rig.valve_x, xr, 0.0, Rp, n(rig.valve_x, xr), nr, 1.0, 1.0, "pipeBehindBall"),
        (xf, rig.pipe_length, 0.0, Rp, n(xf, rig.pipe_length), nr, 1.0, 1.0, "pipeAheadOfBall"),
        (rig.pipe_length, xa, 0.0, Rp, n_ax, nr, e_ax, 1.0, "ambient"),
        (rig.pipe_length, xa, Rp, rig.ambient_radius, n_ax, n_ar, e_ax, e_ar, "ambient"),
    ]


def block_mesh_dict(rig: Rig, mesh: Mesh, ball: Ball) -> str:
    h = math.radians(WEDGE_DEG / 2)
    verts, index = [], {}

    def v(x, r, side):
        key = (round(x, 9), round(r, 9), 0 if r == 0 else side)
        if key not in index:
            index[key] = len(verts)
            verts.append((x, r * math.cos(h), side * r * math.sin(h) if r else 0.0))
        return index[key]

    names = ("containerWall", "pipeWall", "ball", "exitPlate", "ambient", "outlet", "front", "back")
    patches = {k: [] for k in names}
    out_blocks = []
    for x0, x1, r0, r1, nx, nr, ex, er, zone in blocks(rig, mesh, ball):
        b = [v(x0, r0, -1), v(x1, r0, -1), v(x1, r1, -1), v(x0, r1, -1),
             v(x0, r0, 1), v(x1, r0, 1), v(x1, r1, 1), v(x0, r1, 1)]
        out_blocks.append(f"    hex ({' '.join(map(str, b))}) {zone} ({nx} {nr} 1) simpleGrading ({ex:.6g} {er:.6g} 1)")
        patches["back"].append((b[0], b[1], b[2], b[3]))
        patches["front"].append((b[4], b[7], b[6], b[5]))
        xmin, xmax, rmax = (b[0], b[4], b[7], b[3]), (b[1], b[2], b[6], b[5]), (b[3], b[7], b[6], b[2])
        if zone == "container":
            patches["containerWall"].append(xmin)
            if r0 > 0:
                patches["containerWall"] += [rmax, xmax]
        elif zone.startswith("pipe"):
            patches["pipeWall"].append(rmax)
            if abs(x1 - ball.rear_x) < 1e-9:
                patches["ball"].append(xmax)
            if abs(x0 - (ball.rear_x + ball.diameter)) < 1e-9:
                patches["ball"].append(xmin)
        else:
            patches["outlet"].append(xmax)
            if r0 > 0:
                patches["exitPlate"].append(xmin)
                patches["ambient"].append(rmax)
    types = dict(containerWall="wall", pipeWall="wall", ball="wall", exitPlate="wall", ambient="patch", outlet="patch",
                 front="wedge", back="wedge")
    out = [HEADER.format(cls="dictionary", obj="blockMeshDict"), "mergeType points;\n", "scale 0.001;\n", "vertices\n("]
    out += [f"    ({x:.9g} {y:.9g} {z:.9g})" for x, y, z in verts]
    out += [");\n", "blocks\n("] + out_blocks + [");\n", "edges ();\n", "boundary\n("]
    for name in names:
        out.append(f"    {name}\n    {{\n        type {types[name]};\n        faces\n        (")
        out += [f"            ({' '.join(map(str, f))})" for f in patches[name]]
        out.append("        );\n    }")
    out.append(");\n")
    return "\n".join(out)


# -- case --------------------------------------------------------------------------------------------------------------
WALLS = "containerWall|pipeWall|exitPlate|ball"
FIXED_WALLS = "containerWall|exitPlate"


def initial_fields(gas: Gas, cond: Conditions, rig: Rig, ball: Ball) -> dict[str, str]:
    f = sf.initial_fields(gas, cond)
    # the ball is a moving wall: its velocity is the mesh's
    f["U"] = _field("U", "[0 1 -1 0 0 0 0]", "(0 0 0)", {
        "containerWall|pipeWall|exitPlate": dict(type="noSlip"),
        "ball": dict(type="movingWallVelocity", value="uniform (0 0 0)"),
        "ambient": dict(type="pressureInletOutletVelocity", value="uniform (0 0 0)"),
        "outlet": dict(type="inletOutlet", inletValue="uniform (0 0 0)", value="uniform (0 0 0)")}, "volVectorField")
    f = {k: v.replace('"containerWall|pipeWall|exitPlate"', f'"{WALLS}"') if k != "U" else v for k, v in f.items()}
    share = WEDGE_DEG / 360.0                       # the wedge carries this share of the ball (and of its forces)
    xc = ball.centre(rig) * 1e-3
    m = ball.mass * share
    inertia = 0.4 * ball.mass * (ball.diameter * 0.5e-3) ** 2 * share
    f["pointDisplacement"] = HEADER.format(cls="pointVectorField", obj="pointDisplacement") + f"""
dimensions      [0 1 0 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    "{FIXED_WALLS}|ambient|outlet" {{ type fixedValue; value uniform (0 0 0); }}
    pipeWall                       {{ type calculated; value uniform (0 0 0); }}   // set by the layered motion
    ball
    {{
        type                    sixDoFRigidBodyDisplacement;
        mass                    {m:.9g};        // the ball {ball.mass * 1e3:.4g} g x {WEDGE_DEG}/360
        centreOfMass            ({xc:.9g} 0 0);
        momentOfInertia         ({inertia:.6g} {inertia:.6g} {inertia:.6g});
        velocity                (0 0 0);
        rho                     rho;
        accelerationRelaxation  1;
        accelerationDamping     1;
        report                  on;
        constraints
        {{
            axis        {{ sixDoFRigidBodyMotionConstraint line; centreOfRotation ({xc:.9g} 0 0); direction (1 0 0); }}
            noRotation  {{ sixDoFRigidBodyMotionConstraint orientation; }}
        }}
        solver                  {{ type Newmark; }}
        value                   uniform (0 0 0);
    }}
    "front|back" {{ type wedge; }}
}}
"""
    return f


def control_dict(gas: Gas, cond: Conditions, mesh: Mesh, end_time: float, write_interval: float, rig: Rig) -> str:
    """``sonic_flow``'s controlDict (rhoCentralFoam, fixed step) restarted chunk by chunk, plus the ball's force and
    face pressures."""
    base = sf.control_dict(rig, cond, sf.time_step(gas, cond, mesh))
    base = base.replace("startFrom       startTime;", "startFrom       latestTime;")
    base = re.sub(r"endTime         [^;]+;", f"endTime         {end_time:.9g};", base)
    base = re.sub(r"writeInterval   [^;]+;", f"writeInterval   {write_interval:.9g};", base, count=1)
    base = base.replace("writeFormat     binary;", "writeFormat     ascii;")
    extra = """
    pipeInlet
    {
        $exitPlane;
        name            pipeInlet;
    }
    ballForces
    {
        type            forces;
        libs            (forces);
        patches         (ball);
        rho             rho;
        CofR            (0 0 0);
        writeControl    timeStep;
        writeInterval   1;
        log             false;
    }
    ballPressure
    {
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      patch;
        name            ball;
        operation       max;
        fields          (p);
        writeFields     false;
        log             false;
        writeControl    timeStep;
        writeInterval   1;
    }
    ballPressureMin
    {
        $ballPressure;
        operation       min;
    }
"""
    i = base.rindex("}")
    return base[:i] + extra + base[i:]


# the shock-capturing set of sonic_flow, plus the mesh-motion solve
SCHEMES = sf.SCHEMES
SOLUTION = sf.SOLUTION.rstrip().rstrip("}") + """    "cellDisplacement.*" { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0; }
}
"""


BALL_CONDITIONS = Conditions(cfl=0.15)    # headroom: the cells ahead of the ball shrink as it travels


def write_case(case: Path, gas: Gas, rig: Rig = Rig(), cond: Conditions = BALL_CONDITIONS, mesh: Mesh = Mesh(),
               ball: Ball = Ball(), chunk: float = 25e-6) -> Path:
    case = Path(case)
    if case.exists():
        shutil.rmtree(case)
    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True)
    (case / "system/blockMeshDict").write_text(block_mesh_dict(rig, mesh, ball))
    (case / "system/controlDict").write_text(control_dict(gas, cond, mesh, chunk, chunk, rig))
    (case / "system/fvSchemes").write_text(SCHEMES)
    (case / "system/fvSolution").write_text(SOLUTION)
    plain = sf.write_case(case / "_plain", gas, rig, cond, mesh)      # borrow topoSet, setFields, thermo, turbulence
    for f in ("system/topoSetDict", "system/setFieldsDict", "constant/thermophysicalProperties"):
        shutil.copy(plain / f, case / f)
    shutil.rmtree(plain)
    # laminar: v2412's rhoCentralFoam stops in kOmegaSST::correct on a moving mesh (an oriented-flux check). Over the
    # half millisecond the ball takes, the flow is pressure waves and inertia; wall turbulence is a small part of it.
    (case / "constant/turbulenceProperties").write_text(HEADER.format(cls="dictionary", obj="turbulenceProperties") +
                                                        "simulationType laminar;\n")
    # the ball's two faces as faceZones, next to the valve and exit planes sonic_flow's topoSet makes
    xr, xf, eps, big = ball.rear_x * 1e-3, (ball.rear_x + ball.diameter) * 1e-3, 1e-6, 1.0
    topo = (case / "system/topoSetDict").read_text().rstrip().rstrip(");").rstrip()
    topo += f"""
    {{ name pipeInlet; type faceZoneSet; action new; source planeToFaceZone; point (0 0 0); normal (1 0 0); option all; }}
    {{ name ballRear;  type faceZoneSet; action new; source boxToFace; box ({xr - eps:.9g} -{big} -{big}) ({xr + eps:.9g} {big} {big}); }}
    {{ name ballFront; type faceZoneSet; action new; source boxToFace; box ({xf - eps:.9g} -{big} -{big}) ({xf + eps:.9g} {big} {big}); }}
);
"""
    (case / "system/topoSetDict").write_text(topo)
    # 1-D layered motion: only the bore cells either side of the ball move, stretched or squeezed linearly between a
    # fixed plane (valve, exit) and the ball; the container, the bore up to the valve and the ambient stand still
    (case / "constant/dynamicMeshDict").write_text(HEADER.format(cls="dictionary", obj="dynamicMeshDict") + """
dynamicFvMesh       dynamicMotionSolverFvMesh;
motionSolverLibs    (fvMotionSolvers sixDoFRigidBodyMotion);
motionSolver        displacementLayeredMotion;
displacementLayeredMotionCoeffs
{
    regions
    {
        pipeBehindBall
        {
            interpolationScheme linear;
            boundaryField
            {
                pipeInlet  { type fixedValue; value uniform (0 0 0); }
                ballRear   { type uniformFollow; patch ball; }
            }
        }
        pipeAheadOfBall
        {
            interpolationScheme linear;
            boundaryField
            {
                ballFront  { type uniformFollow; patch ball; }
                exitPlane  { type fixedValue; value uniform (0 0 0); }
            }
        }
    }
}
""")
    for name, text in initial_fields(gas, cond, rig, ball).items():
        (case / "0" / name).write_text(text)
    (case / "sonic_ball.json").write_text(json.dumps(dict(gas=asdict(gas), rig=asdict(rig), conditions=asdict(cond),
                                                          mesh=asdict(mesh), ball=asdict(ball), ball_mass_kg=ball.mass,
                                                          chunk=chunk), indent=1))
    (case / "case.foam").touch()
    return case


BALL_RE = re.compile(r"Centre of mass: \(([-\d.eE+]+) [-\d.eE+]+ [-\d.eE+]+\)\s*\n\s*Orientation:.*?\n\s*Linear velocity: \(([-\d.eE+]+)", re.S)


def ball_track(case: Path) -> dict[str, np.ndarray]:
    """Ball centre [m] and velocity [m/s] against time, from the solver logs (``report on``)."""
    t, x, u = [], [], []
    for log in sorted(Path(case).glob("log.rhoCentralFoam*"), key=lambda p: p.stat().st_mtime):
        txt = log.read_text(errors="replace")
        for block in re.split(r"\n(?=Time = )", txt):
            m = re.match(r"Time = ([-\d.eE+]+)", block)
            if not m:
                continue
            states = BALL_RE.findall(block)
            if states:
                t.append(float(m.group(1)))
                x.append(float(states[-1][0]))
                u.append(float(states[-1][1]))
    t, x, u = map(np.array, (t, x, u))
    order = np.argsort(t)
    t, x, u = t[order], x[order], u[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return {"t": t[keep], "x": x[keep], "u": u[keep]}


def run_case(case: Path, ball: Ball = Ball(), rig: Rig = Rig(), env=None, chunk: float = 25e-6, max_time: float = 3e-3,
             on_chunk=None) -> dict:
    """blockMesh .. setFields, then ``rhoCentralFoam`` in chunks of ``chunk`` seconds until the ball's front reaches
    ``ball.stop_front_mm`` (or ``max_time``)."""
    from vegeta import aeromant
    from vegeta.aeromant._process import run_command
    env = env or aeromant.OpenFOAMEnvironment.detect()
    case = Path(case)

    def run(argv, log):
        rec = run_command(env.command(list(argv)), case, env=env.env, log_path=case / log)
        if rec.returncode != 0 or "FOAM FATAL" in rec.stdout + rec.stderr:
            raise RuntimeError(f"{argv[0]} failed in {case}: {case / log}\n" + "\n".join((rec.stdout + rec.stderr).splitlines()[-30:]))
        return rec

    for step in ("blockMesh", "checkMesh", "topoSet", "setFields"):
        run((step,), f"log.{step}")
    meta = json.loads((case / "sonic_ball.json").read_text())
    gas, cond, mesh = Gas(**meta["gas"]), Conditions(**meta["conditions"]), Mesh(**meta["mesh"])
    dt0 = sf.time_step(gas, cond, mesh)
    t_end, i = 0.0, 0
    half = ball.diameter / 2
    front = ball.rear_x + ball.diameter
    u = 0.0
    while t_end < max_time:
        # shorter chunks on the final approach, so the run ends close to stop_front_mm
        left = (ball.stop_front_mm - front) * 1e-3
        step = chunk if u <= 0 else min(chunk, max(5e-6, 0.9 * left / u))
        t_end += step
        i += 1
        # the cells ahead of the ball shrink as it closes on the exit: keep the CFL on the smallest of them
        squeeze = (rig.pipe_length - (ball.rear_x + ball.diameter)) / max(rig.pipe_length - front, 1e-3)
        dt = dt0 / max(1.0, squeeze)
        cd = case / "system/controlDict"
        text = re.sub(r"endTime         [^;]+;", f"endTime         {t_end:.9g};", cd.read_text())
        cd.write_text(re.sub(r"deltaT          [^;]+;", f"deltaT          {dt:.6g};", text))
        run(("rhoCentralFoam",), f"log.rhoCentralFoam.{i:03d}")
        tr = ball_track(case)
        front = (tr["x"][-1] * 1e3 + half) if len(tr["x"]) else ball.rear_x + ball.diameter
        u = tr["u"][-1] if len(tr["u"]) else 0.0
        if on_chunk:
            on_chunk(t_end, front, u)
        if front >= ball.stop_front_mm - 0.5:
            break
    (case / "ball_done.json").write_text(json.dumps(dict(end_time=t_end, front_mm=front, velocity=u, chunks=i)))
    return ball_track(case)


def solve(case: Path, gas: Gas, rig: Rig = Rig(), cond: Conditions = BALL_CONDITIONS, mesh: Mesh = Mesh(),
          ball: Ball = Ball(), run: bool = True, **kw) -> Path:
    """The case solved once: a finished case (``ball_done.json``) for the same inputs is reused, anything else is
    written again and run. ``run=False`` only reuses."""
    case = Path(case)
    want = dict(gas=asdict(gas), rig=asdict(rig), conditions=asdict(cond), mesh=asdict(mesh), ball=asdict(ball))
    meta = case / "sonic_ball.json"
    if (case / "ball_done.json").exists() and meta.exists():
        have = json.loads(meta.read_text())
        defaults = dict(gas={}, rig=asdict(Rig()), conditions=asdict(Conditions()), mesh=asdict(Mesh()), ball=asdict(Ball()))
        # fields added to the dataclasses later count as their defaults in older finished cases
        if all({**{k: d for k, d in defaults[sec].items() if k not in have.get(sec, {})}, **have.get(sec, {})}.get(k) == v
               for sec, vals in want.items() for k, v in vals.items()):
            return case
    if not run:
        raise FileNotFoundError(f"{case} is not solved for these inputs (run=True solves it)")
    write_case(case, gas, rig, cond, mesh, ball)
    run_case(case, ball, rig, **kw)
    return case


def histories(case: Path) -> dict[str, np.ndarray]:
    """``sonic_flow.histories`` plus the ball: track, force (full 360°), max and min pressure on its faces."""
    h = sf.histories(case)
    full = 360.0 / WEDGE_DEG
    inlet = sf._series(case, "pipeInlet")                     # fixed plane at the pipe entrance: what leaves the container
    h["inlet_mdot"] = np.interp(h["t"], inlet[:, 0], inlet[:, 1]) * full
    tr = ball_track(case)
    h["ball_t"], h["ball_x"], h["ball_u"] = tr["t"], tr["x"], tr["u"]
    files = sorted((Path(case) / "postProcessing/ballForces").glob("*/force.dat"), key=lambda p: float(p.parent.name))
    f = np.vstack([sf._dat(p)[1] for p in files])              # Time total(x y z) pressure(x y z) viscous(x y z)
    h["force_t"], h["ball_force"] = f[:, 0], f[:, 1] * full
    h["ball_force_viscous"] = f[:, 7] * full
    pmax, pmin = sf._series(case, "ballPressure"), sf._series(case, "ballPressureMin")
    n = min(len(pmax), len(pmin))
    h["pball_t"], h["p_rear"], h["p_front"] = pmax[:n, 0], pmax[:n, 1], pmin[:n, 1]
    return h


# -- 0-D gun model -----------------------------------------------------------------------------------------------------
def gun(gas: Gas, rig: Rig = Rig(), cond: Conditions = Conditions(), ball: Ball = Ball(), friction: float | None = None,
        cd: float = 1.0, dt: float = 2e-7, t_max: float = 3e-3) -> dict[str, np.ndarray]:
    """Ideal-gas lumped interior ballistics. The container (isentropic) feeds the volume behind the ball through a
    throat of area cd·A_bore. The flow is choked while p_behind < p0·(critical ratio), subsonic-isentropic after that.
    The behind volume is adiabatic, a well-mixed gas doing p dV work on the ball. Ahead, the air is compressed by the
    piston-driven shock: p_ahead = p_amb (1 + γ_a (γ_a+1)/4 (u/a)² + γ_a (u/a) sqrt(1 + ((γ_a+1)/4)² (u/a)²)). Runs
    until the ball's front reaches the pipe exit."""
    friction = ball.friction if friction is None else friction
    A = rig.exit_area_m2
    g, R, cv = gas.gamma, gas.R, gas.cp - gas.R
    ga, aa = sf.AIR.gamma, math.sqrt(sf.AIR.gamma * sf.AIR.R * cond.T_amb)
    V0 = rig.container_volume_l * 1e-3 + A * rig.valve_x * 1e-3
    m0 = cond.p0 * V0 / (R * cond.T0)
    x = 0.0                                                          # ball travel
    Vb0 = A * (ball.rear_x - rig.valve_x) * 1e-3
    mb, Eb = cond.p_amb * Vb0 / (R * cond.T_amb), cond.p_amb * Vb0 / (g - 1)  # behind: mass, internal energy
    m, u, t = m0, 0.0, 0.0
    travel = (rig.pipe_length - ball.rear_x - ball.diameter) * 1e-3
    rows = []
    while x < travel and t < t_max:
        rho0 = m / V0
        p0 = cond.p0 * (rho0 / (cond.p0 / (R * cond.T0))) ** g
        T0 = p0 / (rho0 * R)
        Vb = Vb0 + A * x
        pb = (g - 1) * Eb / Vb
        r = pb / p0
        crit = (2 / (g + 1)) ** (g / (g - 1))
        if r <= crit:
            mdot = cd * A * p0 / math.sqrt(R * T0) * math.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
        elif r < 1.0:
            mdot = cd * A * p0 / math.sqrt(R * T0) * math.sqrt(2 * g / (g - 1) * (r ** (2 / g) - r ** ((g + 1) / g)))
        else:                                  # the driver gas has caught up with the container: no inflow
            mdot = 0.0
        M = u / aa
        pf = cond.p_amb * (1 + ga * (ga + 1) / 4 * M * M + ga * M * math.sqrt(1 + ((ga + 1) / 4) ** 2 * M * M))
        F = A * (pb - pf)
        Fnet = F - (friction if (u > 0 or F > friction) else F)
        a = Fnet / ball.mass
        rows.append((t, x, u, a, p0, T0, pb, pf, F, mdot))
        # explicit step
        m -= mdot * dt
        mb += mdot * dt
        Eb += (mdot * gas.cp * T0 - pb * A * u) * dt
        u = max(0.0, u + a * dt)
        x += u * dt
        t += dt
    cols = np.array(rows).T
    keys = ("t", "travel", "u", "a", "container_p", "container_T", "p_behind", "p_ahead", "force", "mdot")
    out = dict(zip(keys, cols))
    out["ball_x"] = (ball.centre(rig) * 1e-3 + out["travel"])
    return out


# -- fields along the axis ---------------------------------------------------------------------------------------------
def written_times(case: Path) -> list[float]:
    out = []
    for d in Path(case).iterdir():
        try:
            t = float(d.name)
        except ValueError:
            continue
        if t > 0 and (d / "p").exists():
            out.append(t)
    return sorted(out)


def axis_profiles(case: Path, fields=("p", "Ma", "T"), x_range=(-0.02, 0.16)):
    """Each written time: the first ring of cells at the axis (moving mesh included), x [m] and the fields. Returns
    (times, [dict(x=..., p=..., ...)])."""
    import pyvista as pv
    r = pv.OpenFOAMReader(str(Path(case) / "case.foam"))
    r.enable_all_cell_arrays()
    times, rows = [], []
    for t in written_times(case):
        r.set_active_time_value(t)
        m = r.read()["internalMesh"]
        c = m.cell_centers().points
        rr = np.hypot(c[:, 1], c[:, 2])
        k = (rr < rr.min() + 1e-6) & (c[:, 0] > x_range[0]) & (c[:, 0] < x_range[1])
        o = np.argsort(c[k, 0])
        row = {"x": c[k, 0][o]}
        for f in fields:
            if f in m.cell_data:
                row[f] = np.asarray(m.cell_data[f])[k][o]
        times.append(t)
        rows.append(row)
    return np.array(times), rows


# -- bore sweep --------------------------------------------------------------------------------------------------------
SWEEP_DIAMETERS = (7.0, 10.0, 12.0, 15.0)


def bore_setup(d: float, cells_per_radius: int = 5):
    """The rig, ball and mesh for a bore of ``d`` mm: a ball of the same diameter (tight), its rear 5 mm after the
    valve; everything else as the 10 mm rig. The mesh keeps ``cells_per_radius`` cells across the bore radius."""
    rig = replace(Rig(), pipe_bore=d)
    ball = replace(Ball(), diameter=d)
    mesh = Mesh(pipe=d / 2 / cells_per_radius, container_far=10.0, ambient_far=10.0)
    return rig, ball, mesh


def muzzle(track: dict, ball: Ball, rig: Rig) -> dict:
    """The ball's speed when its front reaches the pipe exit: the last tracked state carried the last few mm with
    the last acceleration (v² = v_s² + 2 a Δx)."""
    t, x, u = track["t"], track["x"], track["u"]
    k = max(1, len(t) // 20)
    a = (u[-1] - u[-1 - k]) / (t[-1] - t[-1 - k])
    dx = rig.pipe_length * 1e-3 - (x[-1] + ball.diameter * 0.5e-3)
    v = math.sqrt(max(u[-1] ** 2 + 2 * a * max(dx, 0.0), 0.0))
    return dict(speed=v, time=t[-1] + (max(dx, 0.0) / max(0.5 * (u[-1] + v), 1e-9)), speed_at_stop=u[-1],
                front_at_stop_mm=(x[-1] * 1e3 + ball.diameter / 2), accel_end=a)


def sweep(gas: Gas, diameters=SWEEP_DIAMETERS, root: Path = HERE / "output" / "sonic_ball", run: bool = True,
          cells_per_radius: int = 5, on_chunk=None) -> list[dict]:
    rows = []
    for d in diameters:
        rig, ball, mesh = bore_setup(d, cells_per_radius)
        case = solve(Path(root) / f"sweep_{gas.name}_d{d:g}", gas, rig, BALL_CONDITIONS, mesh, ball, run=run,
                     chunk=50e-6, on_chunk=on_chunk)
        m = muzzle(ball_track(case), ball, rig)
        rows.append(dict(diameter_mm=d, ball_mass_g=ball.mass * 1e3, **m))
    return rows


# -- projectile shapes: cylinder + hemispherical or paraboloidal head ---------------------------------------------------
LD_VALUES = (1, 2, 3, 4, 5, 6, 8, 10)


def force_vs_travel(case: Path) -> tuple[np.ndarray, np.ndarray]:
    """The CFD's net gas force on the body [N] against its rear face's travel [m] (frictionless run)."""
    tr = ball_track(case)
    files = sorted((Path(case) / "postProcessing/ballForces").glob("*/force.dat"), key=lambda p: float(p.parent.name))
    f = np.vstack([sf._dat(p)[1] for p in files])
    F = np.interp(tr["t"], f[:, 0], f[:, 1]) * 360.0 / WEDGE_DEG
    x = tr["x"] - tr["x"][0]
    # at rest (x ~ 0) the force builds up in time: the body breaks away once it beats friction, so keep the peak
    rest = x < 1e-6
    moving = ~rest & np.concatenate([[True], np.diff(x) > 0])
    return np.concatenate([[0.0], x[moving]]), np.concatenate([[F[rest].max() if rest.any() else F[0]], F[moving]])


def exit_with_friction(x: np.ndarray, F: np.ndarray, mass: float, friction: float, travel: float) -> float:
    """Exit speed [m/s] from the gas force along the bore and a constant sliding friction:
    v² = (2/m) ∫ (F - friction) dx over the travel. 0 when the body stalls (or never breaks away)."""
    xs = np.linspace(0.0, travel, 400)
    Fx = np.interp(xs, x, F)                     # past the tracked end: the last force
    if Fx[0] <= friction:
        return 0.0
    work = np.concatenate([[0.0], np.cumsum(0.5 * (Fx[1:] + Fx[:-1] - 2 * friction) * np.diff(xs))])
    if work.min() < 0:
        return 0.0
    return math.sqrt(2 * work[-1] / mass)


def shape_sweep(gas: Gas = sf.AIR, ld_values=LD_VALUES, root: Path = HERE / "output" / "sonic_ball", run: bool = True,
                cells_per_radius: int = 5, on_chunk=None, cfd_head: str = "paraboloid") -> dict:
    """Coupled CFD (frictionless) for the ``cfd_head`` series of L/D; returns the cases, their masses, exit speeds and
    force-vs-travel curves. Rig: the 10 mm bore; every body's rear runs the same 65 mm to the muzzle."""
    rig, _, mesh = bore_setup(10.0, cells_per_radius)
    out = {"ld": [], "mass": [], "speed": [], "x": [], "F": [], "case": []}
    for ld in ld_values:
        b = projectile(cfd_head, ld)
        case = solve(Path(root) / f"shape_{gas.name}_{cfd_head}_ld{ld:g}", gas, rig, BALL_CONDITIONS, mesh, b, run=run,
                     chunk=50e-6, on_chunk=on_chunk)
        x, F = force_vs_travel(case)
        out["ld"].append(ld); out["mass"].append(b.mass); out["case"].append(case)
        out["speed"].append(muzzle(ball_track(case), b, rig)["speed"])
        out["x"].append(x); out["F"].append(F)
    return out


def force_for_mass(sweep_out: dict, mass: float):
    """Gas force vs travel for a body of ``mass``: interpolated (linearly in mass) between the two CFD runs that
    bracket it. The force depends on the mass only through how fast the body runs away from the gas behind it."""
    ms = np.array(sweep_out["mass"])
    i = int(np.clip(np.searchsorted(ms, mass), 1, len(ms) - 1))
    w = (mass - ms[i - 1]) / (ms[i] - ms[i - 1])
    x = np.linspace(0, max(sweep_out["x"][i - 1][-1], sweep_out["x"][i][-1]), 400)
    F = (1 - w) * np.interp(x, sweep_out["x"][i - 1], sweep_out["F"][i - 1]) + w * np.interp(x, sweep_out["x"][i], sweep_out["F"][i])
    return x, F


def profile(b: Ball, n: int = 40) -> list[tuple[float, float]]:
    """Half profile (x from the rear face, r), mm, closed along the axis: cylinder then head."""
    R, Lc, h = b.diameter / 2, b.contact_length, b.head_length
    if b.shape == "sphere":
        a = np.linspace(-np.pi / 2, np.pi / 2, n)
        return [(R + R * math.sin(t), R * math.cos(t)) for t in a][::-1] + [(0.0, 0.0)]
    pts = [(0.0, 0.0), (0.0, R), (Lc, R)]
    s = np.linspace(0, 1, n)[1:]
    if b.shape == "hemisphere":
        pts += [(Lc + R * math.sin(t * math.pi / 2), R * math.cos(t * math.pi / 2)) for t in s]
    else:                                           # paraboloid: r = R sqrt(1 - (x - Lc)/h)
        pts += [(Lc + h * t, R * math.sqrt(max(1 - t, 0.0))) for t in s]
    return pts


def make_projectile_cad(out: Path, bodies: dict[str, Ball]) -> dict:
    """STEP + STL of each body (revolved from ``profile``, axis +x, rear face at x = 0); measured volumes."""
    import cadquery as cq
    from vegeta.dedalus import Design, Parameter
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, b in bodies.items():
        pts = profile(b)

        class Body(Design):
            parameters = [Parameter("diameter", b.diameter, "mm", min=1)]

            def build(self, p, pts=pts):
                wp = cq.Workplane("XY").polyline([pts[0]] + [q for q in pts[1:] if q[1] > 1e-9] + [(pts[-1][0], 0.0)]).close()
                return wp.revolve(360, (0, 0, 0), (1, 0, 0))
        g = Body().generate()
        g.export_step(out / f"{name}.step")
        g.export_stl(out / f"{name}.stl", tolerance=0.02)
        m = g.measure()
        report[name] = dict(volume_mm3=m["volume"], mass_g=m["volume"] * 1e-9 * b.density * 1e3, mass_model_g=b.mass * 1e3)
    return report
