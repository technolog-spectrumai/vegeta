"""Command line interface: ``zephyr section|polar|flap|backends``."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _section(args):
    from .section import from_dat, naca
    if args.dat:
        return from_dat(args.dat)
    if not args.naca:
        raise ValueError("give --naca <digits> or --dat <file>")
    return naca(args.naca, reflex_k21=args.k21)


def _add_section_args(ap):
    ap.add_argument("--naca", help="NACA 4- or 5-digit designation (e.g. 2412, 23012, 23112 reflexed)")
    ap.add_argument("--k21", type=float, default=None, help="reflexed 5-digit: k2/k1 instead of the standard (more reflex)")
    ap.add_argument("--dat", help="a Selig or Lednicer .dat file instead")


def cmd_section(args) -> int:
    from .section import thin_airfoil
    s = _section(args)
    ta = thin_airfoil(s)
    out = {"section": s.to_dict() if args.json else s.name, "thin_airfoil": ta}
    if args.points:
        xy = s.coordinates(args.points)
        out["coordinates"] = xy.tolist()
    if args.out:
        s.write_dat(args.out, n=args.points or 100)
        out["dat"] = args.out
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(s.name)
        for k, v in ta.items():
            print(f"  {k:<20} {v:.5g}")
        if args.out:
            print(f"  written {args.out}")
    return 0


def _alphas(spec):
    a0, a1, da = spec
    return np.arange(a0, a1 + 1e-9, da)


def cmd_polar(args) -> int:
    from .polar import analyze_result
    s = _section(args)
    res = analyze_result(s, _alphas(args.alpha), args.re, cl_design=args.cl, out=args.out, method=args.method, mach=args.mach,
                         n_crit=args.ncrit)
    print(res.to_json() if args.json else res)
    return 0 if res.ok else 1


def cmd_flap(args) -> int:
    from .polar import flap_sweep
    s = _section(args)
    try:
        r = flap_sweep(s, args.cf, args.re, alpha_deg=args.alpha_deg, method=args.method, n_crit=args.ncrit)
    except Exception as exc:
        print(f"zephyr flap: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"{r['section']}: flap {args.cf:g} c at α {args.alpha_deg:g}°, Re {args.re:g} ({r['source']})")
        print(f"  tau {r['tau']:.3f} (thin-airfoil {r['tau_thin']:.3f}), dcl/ddelta {r['dcl_ddelta_per_rad']:.3f}/rad, "
              f"x_cp of the flap's lift {r['x_cp_flap']:.3f} c")
        print("  delta [deg]   cl      cd      cm      confidence")
        for d, cl, cd, cm, c in zip(r["deltas_deg"], r["cl"], r["cd"], r["cm"], r["confidence"]):
            print(f"  {d:+6.1f}      {cl:+.3f}  {cd:.4f}  {cm:+.4f}  {c:.2f}")
    return 0


def cmd_backends(args) -> int:
    from .polar import backends
    b = backends()
    if args.json:
        print(json.dumps(b, indent=2))
    else:
        print(f"neuralfoil: {b['neuralfoil'] or 'not installed (pip install neuralfoil)'}")
        print(f"xfoil:      {b['xfoil'] or 'not on the PATH (set XFOIL=/path/to/xfoil)'}")
    return 0


def main(argv=None, prog: str = "zephyr") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Zephyr — 2-D wing sections: geometry, thin-airfoil theory, polars (NeuralFoil / XFOIL), flaps")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("section", help="a section's thin-airfoil numbers; write its .dat")
    _add_section_args(p)
    p.add_argument("--points", type=int, default=0, help="include the coordinates (points per surface)")
    p.add_argument("-o", "--out", help="write a Selig .dat file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_section)
    p = sub.add_parser("polar", help="the polar at a Reynolds number")
    _add_section_args(p)
    p.add_argument("--re", type=float, required=True, help="Reynolds number on the chord")
    p.add_argument("--alpha", type=float, nargs=3, default=(-6.0, 20.0, 1.0), metavar=("A0", "A1", "DA"), help="angle of attack sweep [deg]")
    p.add_argument("--cl", type=float, default=None, help="a design lift coefficient to report α, cd, cm, L/D at")
    p.add_argument("--method", default="auto", choices=("auto", "neuralfoil", "xfoil"))
    p.add_argument("--mach", type=float, default=0.0)
    p.add_argument("--ncrit", type=float, default=9.0, help="e^N transition criterion (9 clean; 4-6 a rough built-up wing)")
    p.add_argument("-o", "--out", help="write the polar JSON")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_polar)
    p = sub.add_parser("flap", help="a plain flap's effectiveness, lift centre and drag against deflection")
    _add_section_args(p)
    p.add_argument("--cf", type=float, required=True, help="flap chord fraction")
    p.add_argument("--re", type=float, required=True)
    p.add_argument("--alpha-deg", type=float, default=2.0)
    p.add_argument("--method", default="auto", choices=("auto", "neuralfoil", "xfoil"))
    p.add_argument("--ncrit", type=float, default=9.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_flap)
    p = sub.add_parser("backends", help="which backends this machine has")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_backends)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
