# Engineering Export 2 — progress

Branches: `dev_export2` in Yax and Vegeta, both cut from `dev_tags` (Yax 0501221, Vegeta 5dd61a9). Vegeta vendors
Yax `dev_export2` (`vendor/YAX_REVISION`). Not merged; the stacked PRs below `dev_tags` are untouched.

## Status

| Stage | Yax | Vegeta |
|---|---|---|
| 0 conventions | `yax.schema` (units in names, frames, layered validation, `report_losses`), `yax validate`, `docs/engineering.md` | — |
| 1 motors, propulsion | motor, transmission, limits, torque–speed curve; `data["motors"]` catalogue; propulsion with CQ, validity, `geometry_ref` | servos → joint motor blocks, Ogma units → catalogue (stated gear ratio and efficiency only); Pekari, Drongo |
| 2 materials, strength, connections | material (isotropic, orthotropic, composite layup, optional calibrated failure), component strength, connections | `labkit.materials` (Talos, density tables, axles); Pekari materials and axle pins, Drongo frame material |
| 3 aerodynamics | machine and link aero: reference, frame, points, coefficients, derivatives, n-D tables | `labkit.aero`; Falco and Nisus+ derivatives from their design files |
| 4 detachable | `data["assembly"]`: components refer to links and connections; separation limits; `00detach` | `labkit.detach` (from the joint tree); Drongo and Pekari assemblies, none declared detachable |
| integration | — | `tools/tests/test_export2_integration.py`, `tools/consumers/perkun.json` |

## System tags

`00core 00geom 00mass 00inert 00joint 00motor 00prop 00mat 00str 00conn 00aero 00sensor 00prov 00fmt 00detach`
(`00matl`, `00strgth` from the unmerged tags branch renamed to `00mat`, `00str`). A property may carry several.

## Tests (mocks only, all executed)

- Yax `./test.sh -q`: 61 tests, ~5 s — `test_engineering_{motors,materials,aero,detach}.py` against hand-written
  fixtures in `tests/fixtures/export2/`, plus the existing suite.
- Vegeta `tools/tests/test_export2_{motors,materials,aero,detach,integration}.py` + `libs/labkit/tests`: 18 tests,
  ~1.5 s. Mock Chiron robots (2–6 bodies) through the real `tools/export.py`; MuJoCo only compiles them.

## Not executed (run by hand)

- `python -m pytest tools/tests/test_export.py` — the 14 real designs (CAD measurements, Boreas BEMT for Drongo), minutes.
- `./test.sh --smoke` — Vegeta's smoke suite incl. the MuJoCo races.
- Yax `tests/test_formats.py::test_jsbsim_*` run in the suite but need `jsbsim` installed (they skip otherwise).

## Known limitations

- Chiron's servo holds no-load speed in rad/s: joint motors leave in rad/s, the catalogue unit in rpm, each as stated.
- Servo `kt` and resistance are the model's stall-point estimates, marked in `source`.
- Gear ratios and efficiencies exist only where an Ogma entry states them (Pekari, Perseus); others have none.
- Drongo's propellers and motors are geoms of the frame link, not links: its assembly has 3 components, none detachable.
- No design states fasteners, separation limits, strength allowables, sensors or validated failure criteria; those
  blocks are absent in real exports. No Talos result is reused yet: the stored FEA JSON files hold per-case rows
  without a link mapping.
- JSBSim gets propellers but not the `aerodynamics` block (reported).
