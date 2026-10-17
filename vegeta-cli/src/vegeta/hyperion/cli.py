"""Command line interface: ``hyperion materials|material|lens|beam|cavity|laser``. Lengths in metres."""
from __future__ import annotations

import argparse
import json
import math
import sys


def _print(d: dict, as_json: bool):
    if as_json:
        print(json.dumps(d, indent=2, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else str(o)))
        return
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:<28} {_fmt(vv)}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k:<30} {', '.join(_fmt(x) for x in v)}")
        else:
            print(f"  {k:<30} {_fmt(v)}")


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


def cmd_materials(args) -> int:
    from . import materials
    rows = {m.name: {"n_633": m.n(633e-9), "n_1064": m.n(1064e-9), "dn_dT": m.dn_dT, "kappa": m.kappa, "alpha_T": m.alpha_T}
            for m in materials.MATERIALS.values()}
    _print(rows, args.json)
    return 0


def cmd_material(args) -> int:
    from . import materials
    m = materials.get(args.name)
    _print({"material": m.name, "wavelength": args.wavelength, "n": m.n(args.wavelength), "group_index": m.group_index(args.wavelength),
            "dn_dT": m.dn_dT, "kappa": m.kappa, "alpha_T": m.alpha_T, "nu": m.nu, "note": m.note}, args.json)
    return 0


def cmd_lens(args) -> int:
    from .elements import ThickLens
    try:
        material = float(args.material)                          # a bare index
    except ValueError:
        material = args.material                                 # a named material
    lens = ThickLens(args.r1, args.r2, args.thickness, material)
    cp = lens.cardinal_points(args.wavelength)
    _print({"lens": f"R1 {args.r1:g}, R2 {args.r2:g}, t {args.thickness:g} m, {args.material} at {args.wavelength:g} m", **cp}, args.json)
    return 0


def cmd_beam(args) -> int:
    import numpy as np
    from .beam import GaussianBeam
    from .elements import FreeSpace, ThinLens
    b = GaussianBeam(args.wavelength, args.w0, 0.0, args.m2)
    out = {"w0": b.w0, "m2": b.m2, "rayleigh_range": b.rayleigh_range, "divergence_half_angle": b.divergence}
    if args.lens is not None:
        tr = b.through([FreeSpace(args.at, name="to lens"), ThinLens(args.lens, name="lens"), FreeSpace(args.after, name="after")])
        out["after_lens"] = {"w_on_lens": tr.spot_at("lens")["w"], "w_end": tr.out.w0 if tr.out and tr.out.z0 == 0 else float(tr.w[-1]),
                             "waists": tr.waists, "out_w0": tr.out.w0 if tr.out else math.nan, "out_z0_from_end": tr.out.z0 if tr.out else math.nan}
    if args.z:
        z0, z1, dz = args.z
        z = np.arange(z0, z1 + 1e-12, dz)
        out["profile"] = {f"{zz:.4g}": float(b.w(zz)) for zz in z}
    _print(out, args.json)
    return 0


def cmd_cavity(args) -> int:
    from .elements import FreeSpace, ThinLens
    from .resonator import Resonator, two_mirror
    r = two_mirror(args.r1, args.r2, args.length, args.wavelength, (args.reflectivity[0], args.reflectivity[1]), args.loss, name="cavity")
    if args.lens is not None:
        els = list(r.train.elements)
        els[1:2] = [FreeSpace(args.at, name="arm 1"), ThinLens(args.lens, name="lens"), FreeSpace(args.length - args.at, name="arm 2")]
        from .elements import Train
        r = Resonator(Train(els), args.wavelength, False, args.loss, "cavity")
    s = r.summary()
    _print(s, args.json)
    return 0 if s["stable_t"] else 1


def cmd_laser(args) -> int:
    from .gain import Laser, Rod, medium, rod_temperature_rise, thermal_lens
    rod = Rod(medium(args.medium), args.radius, args.length)
    laser = Laser(rod, args.wp, args.wl, args.t, args.loss)
    out = laser.table(args.pump)
    out["optimum_coupler"] = laser.optimum_coupler(args.pump)
    tl = thermal_lens(rod, args.pump, args.wp, args.bulge)
    out["thermal_lens_f"] = tl.f
    out["rod_temperature"] = rod_temperature_rise(rod, args.pump, args.wp)
    _print(out, args.json)
    return 0


def main(argv=None, prog: str = "hyperion") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Hyperion — optics: materials, ABCD elements, Gaussian beams, resonators, a four-level laser")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("materials", help="the named materials and their indices")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_materials)
    p = sub.add_parser("material", help="one material at a wavelength")
    p.add_argument("--name", required=True)
    p.add_argument("--wavelength", type=float, default=1064e-9, help="[m]")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_material)
    p = sub.add_parser("lens", help="a thick lens's cardinal points")
    p.add_argument("--r1", type=float, required=True, help="front radius [m] (inf: flat)")
    p.add_argument("--r2", type=float, required=True, help="back radius [m]")
    p.add_argument("--thickness", type=float, required=True, help="[m]")
    p.add_argument("--material", default="BK7")
    p.add_argument("--wavelength", type=float, default=633e-9)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_lens)
    p = sub.add_parser("beam", help="a Gaussian beam: Rayleigh range, divergence, w(z), through a lens")
    p.add_argument("--wavelength", type=float, required=True)
    p.add_argument("--w0", type=float, required=True, help="waist radius [m]")
    p.add_argument("--m2", type=float, default=1.0)
    p.add_argument("--z", type=float, nargs=3, metavar=("Z0", "Z1", "DZ"), help="tabulate w(z) from the waist")
    p.add_argument("--lens", type=float, help="a thin lens of this focal length [m] ...")
    p.add_argument("--at", type=float, default=0.0, help="... this far from the waist [m]")
    p.add_argument("--after", type=float, default=0.0, help="and this much free space after it [m]")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_beam)
    p = sub.add_parser("cavity", help="a two-mirror cavity (optionally with a lens inside): stability, mode, spectrum")
    p.add_argument("--r1", type=float, required=True, help="mirror 1 radius [m] (inf: flat)")
    p.add_argument("--r2", type=float, required=True)
    p.add_argument("--length", type=float, required=True, help="[m]")
    p.add_argument("--wavelength", type=float, default=1064e-9)
    p.add_argument("--reflectivity", type=float, nargs=2, default=(1.0, 1.0))
    p.add_argument("--loss", type=float, default=0.0, help="round-trip loss besides the mirrors")
    p.add_argument("--lens", type=float, help="a thin lens of this focal length inside [m] ...")
    p.add_argument("--at", type=float, default=0.0, help="... this far from mirror 1 [m]")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_cavity)
    p = sub.add_parser("laser", help="a four-level end-pumped laser: threshold, slope, output, coupler, thermal lens")
    p.add_argument("--medium", default="Nd:YAG")
    p.add_argument("--radius", type=float, default=1.5e-3, help="rod radius [m]")
    p.add_argument("--length", type=float, required=True, help="rod length [m]")
    p.add_argument("--pump", type=float, required=True, help="incident pump power [W]")
    p.add_argument("--wp", type=float, required=True, help="pump spot radius in the rod [m]")
    p.add_argument("--wl", type=float, required=True, help="mode radius in the rod [m]")
    p.add_argument("--t", type=float, required=True, help="output coupler transmission")
    p.add_argument("--loss", type=float, default=0.01, help="other round-trip loss")
    p.add_argument("--bulge", type=float, default=0.0, help="stress and end-bulging as a fraction of the dn/dT lens")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_laser)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
