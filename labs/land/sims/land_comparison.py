#!/usr/bin/env python3
"""The land lab's tracked machines compared — Pekari Rover, Pekari-2M, Pumba (grab / chainsaw), Perseus, Timon, Calydon
(``designs/land_comparison.py``): the table as Markdown and CSV, and the machines to scale from their CAD (side
views, arms folded / boom stowed, a 1.8 m person for scale).

    python3 labs/land/sims/land_comparison.py              # table + figure to sims/output/
    python3 labs/land/sims/land_comparison.py --docs       # also docs/figures/ and docs/land_comparison.md's table
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "land" / "designs"))

import land_comparison as lc  # noqa: E402

OUT = ROOT / "labs" / "land" / "sims" / "output"
DOCS = ROOT / "docs"


def figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection

    import calydon
    import pekari_2m
    import pekari_rover
    import perseus_robot as pbr
    import perseus_rover
    import pumba
    import timon

    st = pbr.PERSEUS.stow
    shapes = [("Pekari Rover", pekari_rover.PekariRover, {}, (0.55, 0.55, 0.52)),
              ("Pekari-2M", pekari_2m.Pekari2M, {}, (0.95, 0.45, 0.05)),
              ("Pumba", pumba.Pumba, {}, (0.98, 0.78, 0.05)),
              ("Pumba + chainsaw (cutting)", pumba.Pumba, {"tool": "chainsaw 2 kW", "arm_pose": "cut"}, (0.98, 0.78, 0.05)),
              ("Perseus (boom stowed)", perseus_rover.PerseusRover,
               {"boom_deg": st["boom"], "stick_deg": st["stick"], "head_deg": st["head"]}, (0.85, 0.65, 0.12)),
              ("Timon", timon.Timon, {}, (0.10, 0.45, 0.85)),
              ("Calydon", calydon.Calydon, {}, (0.85, 0.60, 0.10))]
    fig, ax = plt.subplots(figsize=(25, 3.9))
    x0 = 0.0
    for name, cls, kw, color in shapes:
        g = cls().generate(**kw)
        shp = g.shape.val() if hasattr(g.shape, "val") else g.shape
        verts, tris = shp.tessellate(2.0, 0.5)
        v = np.array([(p.x, p.y, p.z) for p in verts]) / 1e3
        tri = v[np.array(tris)]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        order = np.argsort(tri[:, :, 1].mean(axis=1))[::-1]
        light = 0.45 + 0.55 * np.clip(-nrm[:, 1] * 0.6 + nrm[:, 2] * 0.5 + nrm[:, 0] * 0.2, 0, 1)
        dx = x0 - v[:, 0].min()
        ax.add_collection(PolyCollection(tri[order][:, :, [0, 2]] + np.array([dx, 0.0]),
                                         facecolors=np.clip(np.outer(light[order], color), 0, 1), linewidths=0))
        w = v[:, 0].max() - v[:, 0].min()
        ax.text(x0 + w / 2, v[:, 2].max() + 0.07, name, ha="center", fontsize=9)
        x0 += w + 0.45
    ax.plot([x0 + 0.2, x0 + 0.2], [0, 1.8], color="0.25", lw=8, solid_capstyle="butt")
    ax.text(x0 + 0.2, 1.86, "1.8 m", ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(-0.2, x0 + 0.6)
    ax.set_ylim(-0.05, 2.05)
    ax.set_aspect("equal")
    ax.set_xlabel("m")
    ax.set_title("The land lab's tracked machines to scale (side views of their CAD)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", action="store_true")
    ap.add_argument("--no-cad", action="store_true", help="parametric envelopes, no figure (no CadQuery needed)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    df = lc.table(cad=not a.no_cad)
    md = lc.markdown(cad=not a.no_cad)
    (OUT / "land_comparison.md").write_text(md)
    df.to_csv(OUT / "land_comparison.csv")
    print(md)
    if not a.no_cad:
        figure(OUT / "land_comparison.png")
    if a.docs:
        (DOCS / "figures").mkdir(parents=True, exist_ok=True)
        if not a.no_cad:
            shutil.copy(OUT / "land_comparison.png", DOCS / "figures" / "land_comparison.png")
        doc = DOCS / "land_comparison.md"
        text = doc.read_text()
        a0, a1 = "<!-- table:begin -->", "<!-- table:end -->"
        doc.write_text(text[:text.index(a0) + len(a0)] + "\n" + md + text[text.index(a1):])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
