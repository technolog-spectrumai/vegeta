"""The firing sound of the recoilless 8 mm emitter (Pekari, ``bearing_recoilless``) from CFD: the air around the gun
solved with OpenFOAM's ``rhoCentralFoam`` while the muzzle and the rear venturi blow into it, the pressure recorded on
a ring of microphones 1 m out, then carried to the listener.

- **Sources** (``source_tables``): the 0-D internal ballistics (``bearing_sound.shot_history``, the same equations as
  ``golf_recoilless.shot``) give the gas state at the two exits over time; they drive the CFD's two inflow patches:
  - the muzzle: first the air column the ball pushes out of the tube (the ball's speed, the piston pressure ahead of
    it), then, from the moment the ball leaves, the tube's blowdown — sonic while it is choked, then subsonic;
  - the rear venturi exit (throat 6.91 mm, 2:1 cone): its isentropic supersonic exit state while the chamber is
    high enough for it (the under-expanded back jet), then subsonic.
- **CFD** (``write_case``): an axisymmetric 5° wedge around the gun's axis, the 16 mm tube body a wall from the rear
  exit to the muzzle, still air 1.2 m around it; laminar (the jets' shocks and mixing resolved by the grid, no RANS
  that would damp the sound); non-reflecting far boundary (``waveTransmissive``). Cells 0.5 mm at the exits, growing
  to 5 mm (sound resolved to ~7 kHz at 10 cells per wavelength; finer with ``--far``).
- **Microphones**: 37 probes on a 1.0 m ring (0°–180° from the muzzle direction, every 5°) around the gun's middle.
- **To the listener** (``listener_signal``): the probe nearest the listener's angle, spherical spreading and the travel
  time from 1 m to the listener, plus one reflection from flat ground (the image source); 96 kHz WAV.

A burst of N balls: each shot is its own CFD case (the plenum drops after every ball and the regulator refills it,
``bearing_sound.burst``, so each shot's sources differ); the recorded shots are added at their firing times.

    python bearing_sound_cfd.py --shots 1 --write            # write the case(s) only
    python bearing_sound_cfd.py --shots 1 --run --procs 8    # write, run OpenFOAM, make the WAV
    python bearing_sound_cfd.py --shots 5 --post             # WAV from finished cases
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import bearing_recoilless as br
import bearing_sound as bs
import golf_recoilless as g
import sonic_flow as sf

GAM, R, CP, PA, TA = g.GAM, g.R, g.CP, g.PA, g.TA
C_AIR = math.sqrt(GAM * R * TA)
RATE = bs.RATE


# -- geometry ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gun:
    """The emitter's outside, mm. x along the bore, the muzzle at x = 0 firing towards +x."""
    bore: float = 8.0
    body_od: float = 16.0
    length: float = 1061.0          # muzzle to the rear venturi's exit: 1.0 m tube + 50 mm chamber + 11 mm cone
    rear_exit: float = 9.77         # the 2:1 cone's exit (throat 6.91 mm)
    eps: float = br.EPS
    domain: float = 1200.0          # still air beyond each end and around the body
    mic_radius: float = 1000.0      # the probe ring, from the gun's middle

    @property
    def x_mid(self) -> float:
        return -self.length / 2

    def mics(self) -> list[tuple[float, float, float]]:
        """(angle from the muzzle direction deg, x m, r m)."""
        return [(a, (self.x_mid + self.mic_radius * math.cos(math.radians(a))) * 1e-3,
                 max(self.mic_radius * math.sin(math.radians(a)), 1.0) * 1e-3) for a in range(0, 181, 5)]


@dataclass(frozen=True)
class Grid:
    fine: float = 0.5               # at the exits and along the body
    far: float = 5.0


def blocks(gun: Gun, grid: Grid):
    L, D = gun.length, gun.domain
    rs = [0.0, gun.bore / 2, gun.rear_exit / 2, gun.body_od / 2, gun.body_od / 2 + D]
    xs = [-L - D, -L, gun.x_mid, 0.0, D]
    nx = [sf._grading(D, grid.far, grid.fine), sf._grading(L / 2, grid.fine, grid.far),
          sf._grading(L / 2, grid.far, grid.fine), sf._grading(D, grid.fine, grid.far)]
    nr = [(max(1, round((rs[i + 1] - rs[i]) / grid.fine)), 1.0) for i in range(3)] + [sf._grading(D, grid.fine, grid.far)]
    out = []
    for i in range(4):
        for j in range(4):
            if i in (1, 2) and j < 3:
                continue                                         # inside the gun
            out.append((xs[i], xs[i + 1], rs[j], rs[j + 1], nx[i][0], nr[j][0], nx[i][1], nr[j][1], i, j))
    return out


def block_mesh_dict(gun: Gun, grid: Grid) -> str:
    h = math.radians(sf.WEDGE_DEG / 2)
    verts, index = [], {}

    def v(x, r, side):
        key = (round(x, 9), round(r, 9), 0 if r == 0 else side)
        if key not in index:
            index[key] = len(verts)
            verts.append((x, r * math.cos(h), side * r * math.sin(h) if r else 0.0))
        return index[key]

    patches = {k: [] for k in ("muzzle", "rearExit", "gunWall", "farfield", "front", "back")}
    hexes = []
    for x0, x1, r0, r1, nx, nr, ex, er, i, j in blocks(gun, grid):
        b = [v(x0, r0, -1), v(x1, r0, -1), v(x1, r1, -1), v(x0, r1, -1),
             v(x0, r0, 1), v(x1, r0, 1), v(x1, r1, 1), v(x0, r1, 1)]
        hexes.append(f"    hex ({' '.join(map(str, b))}) ({nx} {nr} 1) simpleGrading ({ex:.6g} {er:.6g} 1)")
        patches["back"].append((b[0], b[1], b[2], b[3]))
        patches["front"].append((b[4], b[7], b[6], b[5]))
        xmin, xmax, rmin, rmax = (b[0], b[4], b[7], b[3]), (b[1], b[2], b[6], b[5]), (b[0], b[1], b[5], b[4]), (b[3], b[7], b[6], b[2])
        if i == 0:
            patches["farfield"].append(xmin)
            if j < 3:
                patches["rearExit" if j < 2 else "gunWall"].append(xmax)
        if i == 3:
            patches["farfield"].append(xmax)
            if j < 3:
                patches["muzzle" if j == 0 else "gunWall"].append(xmin)
        if j == 3:
            patches["farfield"].append(rmax)
            if i in (1, 2):
                patches["gunWall"].append(rmin)
    types = dict(muzzle="patch", rearExit="patch", gunWall="wall", farfield="patch", front="wedge", back="wedge")
    out = [sf.HEADER.format(cls="dictionary", obj="blockMeshDict"), "mergeType points;\n", "scale 0.001;\n", "vertices\n("]
    out += [f"    ({x:.9g} {y:.9g} {z:.9g})" for x, y, z in verts]
    out += [");\n", "blocks\n("] + hexes + [");\n", "edges ();\n", "boundary\n("]
    for name, faces in patches.items():
        out.append(f"    {name}\n    {{\n        type {types[name]};\n        faces\n        (")
        out += [f"            ({' '.join(map(str, f))})" for f in faces]
        out.append("        );\n    }")
    out.append(");\n")
    return "\n".join(out)


def cell_count(gun: Gun, grid: Grid) -> int:
    return sum(b[4] * b[5] for b in blocks(gun, grid))


# -- the sources: exit states from the 0-D shot -----------------------------------------------------------------------
def _exit_state(p0, T0, mdot, area, eps=1.0):
    """(p, T, U) at an exit of area ``area`` fed from stagnation (p0, T0) carrying ``mdot``: the isentropic supersonic
    state of a nozzle of area ratio eps (sonic for eps = 1) while it holds, else subsonic at ambient pressure."""
    if p0 <= PA * 1.0001 or mdot <= 0:
        return PA, TA, 0.0
    M = g._mach_from_area(eps) if eps > 1 else 1.0
    f = 1 + (GAM - 1) / 2 * M * M
    pe = p0 / f ** (GAM / (GAM - 1))
    if pe >= PA:
        Te = T0 / f
        return pe, Te, M * math.sqrt(GAM * R * Te)
    Te = T0 * (PA / p0) ** ((GAM - 1) / GAM)
    return PA, Te, min(mdot * R * Te / (PA * area), math.sqrt(GAM * R * Te))


def source_tables(h: dict, gun: Gun, dt: float = 2e-6) -> dict:
    """Time tables (t, p, T, Ux) of the muzzle and of the rear exit, every ``dt``, from a ``shot_history``."""
    t = np.arange(0.0, h["t"][-1], dt)
    idx = np.clip(np.searchsorted(h["t"], t), 0, len(h["t"]) - 1)
    a_m = math.pi / 4 * (gun.bore * 1e-3) ** 2
    a_r = math.pi / 4 * (gun.rear_exit * 1e-3) ** 2
    mz, rr = [], []
    for k in idx:
        pc, Tc = h["p_c"][k], h["T_c"][k]
        if h["u_ball"][k] > 0 or h["mdot_muzzle"][k] == 0:      # the air column ahead of the ball
            pf = max(h["p_front"][k], PA)
            mz.append((pf, TA * (pf / PA) ** ((GAM - 1) / GAM), h["u_ball"][k]))
        else:
            mz.append(_exit_state(pc, Tc, h["mdot_muzzle"][k], a_m))
        p, T, U = _exit_state(pc, Tc, h["mdot_back"][k], a_r, gun.eps)
        rr.append((p, T, -U))
    return {"t": t, "muzzle": np.array(mz), "rearExit": np.array(rr)}


# -- the OpenFOAM case ------------------------------------------------------------------------------------------------
def _table(path: Path, t, vals, vector=False):
    rows = [f"({ti:.9g} ({v:.6g} 0 0))" if vector else f"({ti:.9g} {v:.8g})" for ti, v in zip(t, vals)]
    path.write_text("(\n" + "\n".join(rows) + "\n)\n")


def _tbl(name):
    return dict(type="uniformFixedValue", uniformValue="tableFile",
                uniformValueCoeffs=f'{{ file "<constant>/{name}"; outOfBounds clamp; interpolationScheme linear; }}')


def fields() -> dict[str, str]:
    g_ = f"{GAM:.6g}"
    wall = "gunWall"
    out = {
        "p": sf._field("p", "[1 -1 -2 0 0 0 0]", PA, {
            wall: dict(type="zeroGradient"), "muzzle": _tbl("muzzle_p"), "rearExit": _tbl("rearExit_p"),
            "farfield": dict(type="waveTransmissive", field="p", psi="thermo:psi", gamma=g_, fieldInf=PA, lInf=1.0,
                             value=f"uniform {PA}")}),
        "T": sf._field("T", "[0 0 0 1 0 0 0]", TA, {
            wall: dict(type="zeroGradient"), "muzzle": _tbl("muzzle_T"), "rearExit": _tbl("rearExit_T"),
            "farfield": dict(type="inletOutlet", inletValue=f"uniform {TA}", value=f"uniform {TA}")}),
        "U": sf._field("U", "[0 1 -1 0 0 0 0]", "(0 0 0)", {
            wall: dict(type="noSlip"), "muzzle": _tbl("muzzle_U"), "rearExit": _tbl("rearExit_U"),
            "farfield": dict(type="waveTransmissive", field="U", psi="thermo:psi", gamma=g_, fieldInf="(0 0 0)",
                             lInf=1.0, value="uniform (0 0 0)")}, "volVectorField"),
    }
    return {k: v.replace("linear; };", "linear; }") for k, v in out.items()}   # no ';' after the coeffs dict


def time_step(grid: Grid, cfl: float = 0.4) -> float:
    wave = math.sqrt(2 * CP * TA) + C_AIR            # the fastest jet (expanded to nothing) + sound
    return cfl * grid.fine * 1e-3 / wave


def control_dict(gun: Gun, end: float, dt: float, sample: float = 4e-6) -> str:
    every = max(1, round(sample / dt))
    locs = " ".join(f"({x:.6g} {r:.6g} 0)" for _, x, r in gun.mics())
    return sf.HEADER.format(cls="dictionary", obj="controlDict") + f"""
application     rhoCentralFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end:.9g};
deltaT          {dt:.6g};
writeControl    adjustable;
writeInterval   {max(end / 10, 1e-3):.6g};
purgeWrite      2;
writeFormat     binary;
writePrecision  8;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;
adjustTimeStep  no;

functions
{{
    mics
    {{
        type            probes;
        libs            (sampling);
        fields          (p);
        probeLocations  ({locs});
        writeControl    timeStep;
        writeInterval   {every};
    }}
}}
"""


def write_case(case: Path, h: dict, gun: Gun = Gun(), grid: Grid = Grid(), tail: float = 6e-3) -> Path:
    """A ready-to-run case for one shot ``h`` in ``case`` (replaced if it exists)."""
    import shutil
    case = Path(case)
    if case.exists():
        shutil.rmtree(case)
    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True)
    src = source_tables(h, gun)
    end = float(src["t"][-1]) + (gun.mic_radius + gun.length / 2) * 1e-3 / C_AIR + tail
    dt = time_step(grid)
    for patch in ("muzzle", "rearExit"):
        a = src[patch]
        _table(case / f"constant/{patch}_p", src["t"], a[:, 0])
        _table(case / f"constant/{patch}_T", src["t"], a[:, 1])
        _table(case / f"constant/{patch}_U", src["t"], a[:, 2], vector=True)
    (case / "system/blockMeshDict").write_text(block_mesh_dict(gun, grid))
    (case / "system/controlDict").write_text(control_dict(gun, end, dt))
    (case / "system/fvSchemes").write_text(sf.SCHEMES.replace("    div(phi,k)      Gauss limitedLinear 1;\n", "")
                                           .replace("    div(phi,omega)  Gauss limitedLinear 1;\n", ""))
    (case / "system/fvSolution").write_text(sf.SOLUTION)
    (case / "constant/thermophysicalProperties").write_text(sf.thermophysical(sf.AIR))
    (case / "constant/turbulenceProperties").write_text(sf.HEADER.format(cls="dictionary", obj="turbulenceProperties") +
                                                        "simulationType laminar;\n")
    for name, text in fields().items():
        (case / "0" / name).write_text(text)
    (case / "shot.json").write_text(json.dumps(dict(
        gun=asdict(gun), grid=asdict(grid), cells=cell_count(gun, grid), delta_t=dt, end_time=end,
        steps=round(end / dt), mics_deg=[a for a, _, _ in gun.mics()], plenum_bar_g=h["p_gauge"], ball_m_s=h["v"],
        barrel_s=h["barrel_s"], air_g=h["air_kg"] * 1e3), indent=1))
    (case / "case.foam").touch()
    return case


STEPS = (("blockMesh",), ("checkMesh",), ("rhoCentralFoam",))


# -- microphones -> listener ------------------------------------------------------------------------------------------
def read_mics(case: Path) -> tuple[np.ndarray, np.ndarray]:
    """Time and p - p_ambient [Pa] (n_t x n_mics) from every ``postProcessing/mics/<start>/p`` (restarts joined)."""
    parts = []
    for f in sorted((Path(case) / "postProcessing/mics").glob("*/p"), key=lambda p: float(p.parent.name)):
        a = np.loadtxt(f, comments="#")
        parts.append(a[a[:, 0] > (parts[-1][-1, 0] if parts else -1)] if parts else a)
    if not parts:
        raise FileNotFoundError(f"no microphone data in {case}/postProcessing/mics: has the CFD run?")
    a = np.vstack(parts)
    return a[:, 0], a[:, 1:] - PA


def listener_signal(case: Path, gun: Gun, *, distance: float, angle: float, height: float, ear: float,
                    seconds: float) -> np.ndarray:
    """The recorded shot at the listener (``distance`` m from the gun's middle, ``angle`` from the muzzle direction):
    the nearest ring microphone carried out with 1/r and its delay, plus the ground's image source."""
    t, p = read_mics(case)
    angles = np.array([a for a, _, _ in gun.mics()])
    r0 = gun.mic_radius * 1e-3
    tt = np.arange(int(seconds * RATE)) / RATE
    out = np.zeros_like(tt)
    for path_len, ang in ((math.hypot(distance, ear - height), angle),
                          (math.hypot(distance, ear + height), angle)):          # direct, then off the ground
        k = int(np.argmin(np.abs(angles - ang)))
        sig = p[:, k] - p[: max(5, len(t) // 200), k].mean()                     # zero the start
        out += np.interp(tt - (path_len - r0) / C_AIR, t, sig, left=0.0, right=0.0) * r0 / path_len
    n = int(0.005 * RATE)                                                          # 5 ms fade-out where the CFD ends
    end = int(min(seconds, t[-1] + (distance - r0) / C_AIR) * RATE)
    if 0 < end <= len(out):
        out[max(0, end - n):end] *= np.linspace(1, 0, min(n, end))
        out[end:] = 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--interval", type=float, default=0.12, help="time between balls [s]")
    ap.add_argument("--refill", type=float, default=0.25, help="plenum refill time constant [s] (assumed)")
    ap.add_argument("--distance", type=float, default=10.0, help="listener distance [m]")
    ap.add_argument("--angle", type=float, default=90.0, help="listener angle from the muzzle direction [deg]")
    ap.add_argument("--height", type=float, default=1.0, help="gun height above ground [m]")
    ap.add_argument("--ear", type=float, default=1.6, help="listener ear height [m]")
    ap.add_argument("--fine", type=float, default=Grid.fine, help="cell size at the exits [mm]")
    ap.add_argument("--far", type=float, default=Grid.far, help="largest cell size, at the microphones [mm]")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("pekari_sounds"))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the case(s) only")
    mode.add_argument("--run", action="store_true", help="write, run OpenFOAM, make the WAV (default)")
    mode.add_argument("--post", action="store_true", help="make the WAV from finished cases")
    args = ap.parse_args()
    if not (args.write or args.post):
        args.run = True
    gun, grid = Gun(), Grid(args.fine, args.far)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.shots}shot{'s' if args.shots > 1 else ''}"
    cases = [args.out / f"cfd_{tag}" / f"shot{i + 1}" for i in range(args.shots)]
    if not args.post:
        d = br.design(verbose=False)
        print(f"Pekari: {d['p_bar_g']:.1f} bar g, throat {d['throat_mm']:.2f} mm, {d['v']:.0f} m/s, back jet {d['jet_speed']:.0f} m/s")
        print(f"mesh {cell_count(gun, grid):,} cells (wedge), dt {time_step(grid):.3g} s")
        for c, h in zip(cases, bs.burst(args.shots, args.interval, args.refill, d)):
            write_case(c, h, gun, grid)
            info = json.loads((c / "shot.json").read_text())
            print(f"  {c}: plenum {h['p_gauge']:.1f} bar g, ball {h['v']:.0f} m/s, {info['end_time'] * 1e3:.1f} ms = {info['steps']:,} steps")
        if args.write:
            return
        for c in cases:
            if list((c / "postProcessing/mics").glob("*/p")) and (c / "log.rhoCentralFoam").exists():
                print(f"  {c}: has results, skipped (delete it to rerun)")
                continue
            print(f"  running {c} on {args.procs} cores ...")
            sf.run_case(c, steps=STEPS, processors=args.procs)
    seconds = args.interval * (args.shots - 1) + 0.6
    total = np.zeros(int(seconds * RATE))
    rows = []
    for i, c in enumerate(cases):
        s = listener_signal(c, gun, distance=args.distance, angle=args.angle, height=args.height, ear=args.ear,
                            seconds=seconds - i * args.interval)
        k = int(round(i * args.interval * RATE))
        total[k:k + len(s)] += s[: len(total) - k]
        info = json.loads((c / "shot.json").read_text())
        peak = float(np.abs(s).max())
        rows.append(dict(shot=i + 1, time_s=round(i * args.interval, 3), plenum_bar_g=round(info["plenum_bar_g"], 2),
                         ball_m_s=round(info["ball_m_s"], 1), peak_Pa=round(peak, 2),
                         peak_dB=round(20 * math.log10(max(peak, 1e-9) / 20e-6), 1)))
        print(f"  shot {i + 1}: ball {info['ball_m_s']:.0f} m/s, peak {rows[-1]['peak_dB']} dB at {args.distance:g} m")
    full = 1.1 * max(float(np.abs(total).max()), 1e-6)
    stem = f"pekari_{tag}_{args.distance:g}m_{args.angle:g}deg"
    bs.write_wav(args.out / f"{stem}.wav", total, full)
    (args.out / f"{stem}.json").write_text(json.dumps(dict(
        source="rhoCentralFoam, ring microphones at 1 m", listener=dict(distance_m=args.distance, angle_deg=args.angle,
        gun_height_m=args.height, ear_m=args.ear), burst=dict(shots=args.shots, interval_s=args.interval,
        refill_s=args.refill), full_scale_Pa=full, shots=rows), indent=1))
    print("wrote", args.out / f"{stem}.wav")


if __name__ == "__main__":
    main()
