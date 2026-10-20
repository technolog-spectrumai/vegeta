# Ogma

The physics of standard machine parts, and the file format that carries a frozen design.

| Distribution | Import | Deps | What it is |
|---|---|---|---|
| [Yax](yax/README.md) | `yax` | standard library | the design package: a directory with an STL and `yax.json` (units, frame, mass properties, hashes) — [spec](yax/docs/spec.md) |
| [Ogma](ogma/README.md) | `import ogma as og` | numpy | what standard parts *are* (data) and *do* (behaviour): `og.mass`, `og.gears`, `og.motors`, `og.batteries`, `og.propulsion`, `og.rotors`, `og.aero`, `og.atmosphere`, `og.wind`, `og.soil`, `og.tracks`, `og.cutting` |

**Who uses it.** Vegeta (where machines are conceived and designed) fills Ogma models — its tools compute the numbers.
Piccolo (where hatched machines learn to fly) and Chiron (MuJoCo) run them. Nomios and Ketos read Yax packages.
Ogma never imports any of them: `yax ← ogma ← chiron ← vegeta, piccolo`.

**Stability.** The physical world anchors it — gear geometry, DC-motor lines, the ISA, Bekker soils — so it changes
slowly and both sides can depend on it. It is organised by physics, never by machine; a module enters when a machine
needs it. Catalogues (actuators, cells, soils, materials) are data files with their sources.

**Tests.** Characterisation first: the physics moved here from Vegeta reproduces, to rounding, the numbers Vegeta's
code gave on fixed inputs (that code is the source of truth); published references come on top where they exist.

## How the hosts carry it

Vegeta and Piccolo keep Ogma in `vendor/ogma` (a squashed `git subtree`), record the revision in
`vendor/OGMA_REVISION` and install it offline: `pip install --no-deps --no-build-isolation --no-index -e
vendor/ogma/yax -e vendor/ogma/ogma`. `scripts/update_ogma.sh REF` is the only command that contacts this repository.

## Tests

```bash
scripts/test.sh            # both distributions and the layer rules, from the source tree, offline
scripts/test.sh --smoke    # the same, timed
./user_tests.sh            # the full check to run by hand (the same suite, verbose, with a report)
```

History: Yax was written in [Ozaru](https://github.com/technolog-spectrumai/ozaru) and moved here with its history
(2026-10-20).
