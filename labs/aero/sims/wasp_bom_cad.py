"""WASP's bill of materials and CAD files — no solver; seconds.

Writes into ``labs/aero/designs/data/wasp/``:

- ``<build>_bom.md`` and ``<build>_bom.csv``: every bought part (quantity, mass, price, where it is listed, what in
  the row is assumed), the printed parts by filament, the totals;
- ``<build>.step``: the whole aircraft (solid, propellers not built);
- ``<build>_arm.step``: one arm with its pod and root pad (the FEA's part, and the printer's);
- ``<build>_nose.step``: the nose cone as a printed shell.

    python labs/aero/sims/wasp_bom_cad.py
    python labs/aero/sims/wasp_bom_cad.py --build WASP-1F
"""
import argparse
import csv

import wasp_common as wc
from wasp_common import fam, wasp

import emerillon_market as mk  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=wc.DEFAULT, choices=wc.BUILDS)
    ap.add_argument("--no-cad", action="store_true", help="the bill of materials only")
    a = ap.parse_args(argv)
    r = fam.card(a.build)
    out = wc.ROOT / "labs" / "aero" / "designs" / "data" / "wasp"
    out.mkdir(parents=True, exist_ok=True)
    stem = wc.slug(a.build)
    rows = [{"qty": b["qty"], "part": b["part"], "g each": b["g_each"], "g": round(b["qty"] * b["g_each"], 1), "USD each": b["usd_each"],
             "USD": round(b["qty"] * b["usd_each"], 2), "data": b["kind"], "listing": b["url"], "note": b["note"]} for b in r["bom"]]
    names = {"petg-hf": "printed nose, bay skin, tail cone, fins (PETG HF)", "pa6-cf": "printed arms, motor pods, body rings (PA6-CF)", "tpu": "printed tail bumpers (TPU 95A)"}
    for key, g in r["printed_g"].items():
        f = mk.part(key)
        usd = round(g / 1000 * f.price_usd * mk.PRINT_WASTE, 2)
        rows.append({"qty": 1, "part": names[key], "g each": g, "g": g, "USD each": usd, "USD": usd, "data": "model",
                     "listing": f.url, "note": f"flying grams from the printed-mass rule; filament bought x{mk.PRINT_WASTE} for supports and failed prints"})
    if r["payload_g"]:
        rows.append({"qty": 1, "part": "payload", "g each": r["payload_g"], "g": r["payload_g"], "USD each": 0.0, "USD": 0.0, "data": "requirement", "listing": "", "note": "what it carries"})
    wiring = round(r["mass_g"] - sum(x["g"] for x in rows), 1)
    rows.append({"qty": 1, "part": "power wiring (allowance)", "g each": wiring, "g": wiring, "USD each": 0.0, "USD": 0.0, "data": "assumed", "listing": "", "note": "silicone wire and solder: in the hardware line's price"})
    total_g, total_usd = round(sum(x["g"] for x in rows), 1), round(sum(x["USD"] for x in rows), 2)
    with open(out / f"{stem}_bom.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    md = [f"# {a.build} — bill of materials", "", f"`{r['label']}`", "",
          f"{total_g:.0f} g" + (f" with a {r['payload_g']:.0f} g payload" if r["payload_g"] else "") + f", about ${total_usd:.0f} in parts and filament. "
          f"Model results: {r['kmh']:.0f} km/h, static thrust {float(r['static_tw']):.1f} x weight, {r['length'] * 1e3:.0f} mm long, "
          f"{r['across_tips'] * 1e3:.0f} mm across the propeller tips" + (f", {r['range_km']:.1f} km at {r['cruise_kmh']:.0f} km/h" if r["range_km"] else "") + ".", "",
          f"Listings read through a web search on {wasp.PRICED_ON}: search snippets, not order confirmations. `data`: **sourced** — mass and "
          "numbers from the listing; **partly** — the product is listed, what the note names is assumed; **assumed** — an allowance; **model** — "
          "computed. Not in the price: the printer, the radio, goggles, a charger, shipping, taxes, spares.", "",
          "| qty | part | g each | g | USD each | USD | data | note |", "|---:|---|---:|---:|---:|---:|---|---|"]
    for x in rows:
        part = f"[{x['part']}]({x['listing']})" if x["listing"] else x["part"]
        md.append(f"| {x['qty']} | {part} | {x['g each']:g} | {x['g']:g} | {x['USD each']:.2f} | {x['USD']:.2f} | {x['data']} | {x['note']} |")
    md += [f"| | **total** | | **{total_g:g}** | | **{total_usd:.2f}** | | |", "",
           "The propellers are two of each rotation (clockwise and counter-clockwise): order both, plus a spare pair of each."]
    (out / f"{stem}_bom.md").write_text("\n".join(md) + "\n")
    print(f"{a.build}: {total_g:.0f} g, ${total_usd:.2f} -> {out / (stem + '_bom.md')}")
    if not a.no_cad:
        g = fam.cad(a.build)
        g.export_step(str(out / f"{stem}.step"))
        fam.cad(a.build, part="arm").export_step(str(out / f"{stem}_arm.step"))
        fam.cad(a.build, part="nose").export_step(str(out / f"{stem}_nose.step"))
        m = g.measure()
        print(f"  CAD: {g}, valid {m['valid']}, {m['n_solids']} solid -> {out / (stem + '.step')}, _arm.step, _nose.step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
