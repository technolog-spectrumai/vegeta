"""Command line interface: ``hyperion materials|material|lens|beam|cavity|laser|fibre|fibre-laser|combine``. Lengths in metres."""
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


def _fibre(args):
    from .elements import INF  # noqa: F401  (keeps the import graph identical to the API's)
    from .fibre import Fibre
    return Fibre(args.core, args.na, args.clad, args.coating, alpha_signal=args.loss_db_per_km / 1e3 / 4.343, alpha_pump=args.loss_db_per_km / 1e3 / 4.343)


def cmd_fibre(args) -> int:
    f = _fibre(args)
    lam = args.wavelength
    out = {"V": f.V(lam), "single_mode": f.single_mode(lam), "cutoff_wavelength": f.cutoff_wavelength, "mode_radius": f.mode_radius(lam),
           "MFD": 2 * f.mode_radius(lam), "mode_area": f.mode_area(lam), "overlap_core": f.overlap_core(lam), "overlap_clad": f.overlap_clad,
           "divergence_half_angle": f.beam(lam).divergence}
    if args.bend:
        out["bend_loss_db_per_m"] = {f"{r:g}": float(f.bend_loss_db_per_m(r, lam)) for r in args.bend}
    _print(out, args.json)
    return 0


def cmd_fibre_laser(args) -> int:
    from .fibre import YB_SILICA, ActiveFibre, FibreAmplifier, FibreLaser, core_temperature, ions_for_absorption, srs_threshold
    f = _fibre(args)
    yb = YB_SILICA.with_(N=ions_for_absorption(f, args.absorption, args.pump_wavelength))
    act = ActiveFibre(f, yb, args.length)
    if args.seed is not None:
        s = FibreAmplifier(act, args.pump_wavelength, args.wavelength, args.direction).solve(args.seed, args.pump, n=args.steps)
        out = {"amplifier": f"{args.length:g} m, {args.direction}-pumped, seed {args.seed:g} W, pump {args.pump:g} W", **s.to_dict()}
    else:
        laser = FibreLaser(act, args.pump_wavelength, args.wavelength, args.r_hr, args.r_oc)
        s = laser.solve(args.pump, n=args.steps)
        out = {"oscillator": f"{args.length:g} m, R_hr {args.r_hr:g}, R_oc {args.r_oc:g}, pump {args.pump:g} W", **s.to_dict(), "threshold_W": laser.threshold(n=args.steps // 2)}
    out["absorption_db_per_m"] = act.absorption_db_per_m(args.pump_wavelength)
    out["srs"] = srs_threshold(f, args.length, args.wavelength)
    out["temperature"] = core_temperature(f, s.peak_heat_per_m, h=args.h)
    _print(out, args.json)
    return 0 if s.output_W > 0 else 1


def cmd_combine(args) -> int:
    from .combining import CoherentCombiner, IncoherentCombiner, SpectralCombiner, channels, equivalent_gaussian
    lams = [args.wavelength + (i - (args.n - 1) / 2) * args.spacing for i in range(args.n)] if args.method == "spectral" else None
    layout = "overlap" if args.method in ("coherent-filled", "spectral") else args.layout
    ch = channels(args.n, args.wavelength, args.power, args.w0, pitch=args.pitch, layout=layout, m2=args.m2, phases=args.phases, losses=args.losses, wavelengths=lams)
    comb = {"incoherent": IncoherentCombiner(args.throughput), "coherent-filled": CoherentCombiner(args.throughput, "filled"),
            "coherent-tiled": CoherentCombiner(args.throughput, "tiled"), "spectral": SpectralCombiner(max(args.spacing / 2, 1e-12), args.dispersion, args.throughput)}[args.method]
    r = comb.combine(ch, n=args.grid)
    one = equivalent_gaussian(r.power_out, args.wavelength, args.w0, r.grid, args.m2)
    out = {"channels": [c.to_dict() for c in ch], **r.to_dict(),
           "one_gaussian_of_the_same_power": {"brightness": one.brightness, "sigma_tx": one.sigma_tx, "I_far_axis": one.I_far_axis},
           "brightness_vs_one_gaussian": r.brightness / one.brightness if one.brightness > 0 else math.nan}
    _print(out, args.json)
    return 0 if r.power_out > 1e-9 * r.power_in else 1


def _add_fibre_args(p):
    p.add_argument("--core", type=float, required=True, help="core radius [m]")
    p.add_argument("--na", type=float, required=True, help="core NA")
    p.add_argument("--clad", type=float, default=62.5e-6, help="cladding (pump guide) radius [m]")
    p.add_argument("--coating", type=float, default=None, help="coating radius [m]")
    p.add_argument("--loss-db-per-km", type=float, default=10.0, help="background loss")
    p.add_argument("--wavelength", type=float, default=1070e-9, help="signal [m]")


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
    p = sub.add_parser("fibre", help="a step-index fibre: V, cutoff, mode radius, overlaps, bend loss")
    _add_fibre_args(p)
    p.add_argument("--bend", type=float, nargs="*", help="bend radii [m] to report the LP01 loss at")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_fibre)
    p = sub.add_parser("fibre-laser", help="a Yb fibre oscillator (or, with --seed, an amplifier): the power evolution, SRS, the core's temperature")
    _add_fibre_args(p)
    p.add_argument("--length", type=float, required=True, help="active fibre length [m]")
    p.add_argument("--absorption", type=float, required=True, help="cladding-pumped small-signal absorption at the pump wavelength [dB/m]")
    p.add_argument("--pump", type=float, required=True, help="launched pump [W]")
    p.add_argument("--pump-wavelength", type=float, default=976e-9)
    p.add_argument("--r-hr", type=float, default=0.99)
    p.add_argument("--r-oc", type=float, default=0.10)
    p.add_argument("--seed", type=float, default=None, help="an amplifier: the seed power [W]")
    p.add_argument("--direction", default="counter", choices=("co", "counter"))
    p.add_argument("--h", type=float, default=50.0, help="surface heat transfer [W/m²/K]")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_fibre_laser)
    p = sub.add_parser("combine", help="N fibre-laser outputs combined incoherently, coherently (filled or tiled aperture) or spectrally: power, far field, beam quality")
    p.add_argument("--n", type=int, default=4, help="channels")
    p.add_argument("--power", type=float, default=1.0, help="power per channel [W]")
    p.add_argument("--w0", type=float, default=25e-6, help="each channel's waist [m]")
    p.add_argument("--wavelength", type=float, default=1070e-9)
    p.add_argument("--m2", type=float, default=1.0, help="each channel's M² (incoherent and spectral only)")
    p.add_argument("--method", default="incoherent", choices=("incoherent", "coherent-filled", "coherent-tiled", "spectral"))
    p.add_argument("--layout", default="ring", choices=("ring", "line", "hex", "overlap"))
    p.add_argument("--pitch", type=float, default=None, help="neighbour spacing [m] (default 3.5 w0)")
    p.add_argument("--phases", type=float, nargs="*", default=None, help="per channel [rad]")
    p.add_argument("--losses", type=float, nargs="*", default=None, help="per channel, a fraction")
    p.add_argument("--throughput", type=float, default=1.0, help="the combiner's own transmission (spectral: per channel)")
    p.add_argument("--spacing", type=float, default=3e-9, help="spectral: the wavelength step between channels [m]")
    p.add_argument("--dispersion", type=float, default=0.0, help="spectral: the residual angular dispersion [rad/m]")
    p.add_argument("--grid", type=int, default=256)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_combine)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
