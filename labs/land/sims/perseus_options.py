#!/usr/bin/env python3
"""Perseus's chassis with other manipulators and a second hand — the investigation's table
(``designs/perseus_options.py``), printed, written to ``sims/output/perseus_options.md`` and (``--docs``) into
``docs/perseus_options.md`` between its table markers.

    python3 labs/land/sims/perseus_options.py --docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "land" / "designs"))

import perseus_options as po  # noqa: E402

OUT = ROOT / "labs" / "land" / "sims" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", action="store_true")
    a = ap.parse_args(argv)
    md = po.markdown()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "perseus_options.md").write_text(md)
    print(md)
    print("branch pieces a hand would hold (green spruce, 1.3 m):")
    for c in po.branch_cases():
        print(f"  d {c['d_m']:.2f} m: {c['piece_kg']:.1f} kg; a bottom cut {c['kerf_bottom_cut']}")
    for r in (1.4, 1.8, 2.1):
        a1, a2 = po.boom_lift(r), po.boom_lift(r, v=po.STRONG_WRIST)
        print(f"boom grab at {r} m: {a1['capacity_kg']:.0f} kg ({a1['limit']}); with the 400 N m wrist "
              f"{a2['capacity_kg']:.0f} kg ({a2['limit']})")
    if a.docs:
        doc = ROOT / "docs" / "perseus_options.md"
        text = doc.read_text()
        b, e = "<!-- table:begin -->", "<!-- table:end -->"
        doc.write_text(text[:text.index(b) + len(b)] + "\n" + md + text[text.index(e):])
    return 0


if __name__ == "__main__":
    sys.exit(main())
