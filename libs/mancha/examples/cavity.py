"""Lid-driven cavity at Re = 100 (two-dimensional: one periodic layer in z), compared with Ghia et al. (1982),
with profile plots.

    python examples/cavity.py [--n 64] [--backend numpy|triton]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mancha.io import plots  # noqa: E402
from mancha.validation import cavity, references  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--backend", default="numpy")
ap.add_argument("--out", default="results/cavity")
args = ap.parse_args()

r = cavity(n=args.n, backend=args.backend)
print(f"steady after {r['steps']} steps (relative change {r['residual']:.1e})")
print(f"max |error| against Ghia: u {r['ghia_u_max_abs_error']:.4f}, v {r['ghia_v_max_abs_error']:.4f}; "
      f"primary vortex at {r['vortex_centre']} (Ghia {references.GHIA_RE100_PRIMARY_VORTEX})")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for name, (s, u), ref, label in (("u", r["u_centreline"], references.GHIA_RE100_U, "y"),
                                 ("v", r["v_centreline"], references.GHIA_RE100_V, "x")):
    fig = plots.velocity_profile(s, u, None, xlabel=label, ylabel=f"{name} / U", title=f"Re = 100, n = {args.n}")
    fig.axes[0].plot(ref[:, 0], ref[:, 1], "s", mfc="none", label="Ghia et al. (1982)")
    fig.axes[0].legend()
    fig.savefig(out / f"cavity_{name}.png", dpi=150)
r["simulation"].save(out)
print(f"plots and fields in {out}/")
