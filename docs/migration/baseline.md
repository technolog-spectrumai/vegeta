# Pre-migration baseline (Vegeta 2edb9c5, Piccolo bc4dbe5 — 2026-10-20)

This records which tests passed before robotics moved out of Vegeta (to Chiron and Piccolo) and how long they took. The tests ran
on a fresh Linux container: Python 3.13.16, MuJoCo 3.16.0, CadQuery 2.8.0 and NumPy 2.5.4. CalculiX, OpenFOAM and PrusaSlicer were not
installed, so their tests skip. The golden numbers are in `tests/smoke/golden.json`. The `robotics.*` entries are
the reference that Chiron and Piccolo must reproduce after the move.

## Vegeta

| Suite (`-m "not slow"`) | Result | Time |
|---|---|---|
| vegeta-cli/tests/cli | 4 passed | 3 s |
| …/dedalus | 28 passed | 7 s |
| …/talos | 35 passed, 23 skipped (no CalculiX) | 1 s |
| …/aeromant | 87 passed, 4 deselected | 10 s |
| …/mellonia | 15 passed, 7 skipped (no PrusaSlicer) | 0.2 s |
| …/boreas | 56 passed | 13 s |
| …/chronos | 7 passed | 0.7 s |
| …/chiron | 172 passed | 22 s |
| …/atropos | 22 passed | 2 s |
| …/zephyr | 19 passed, 2 skipped | 1.4 s |
| …/mission | 14 passed | 3.5 s |
| …/mock, …/cache | 8 + 17 passed | 1.5 s |
| vegeta-core | 14 passed, 5 skipped | 3 s |
| vegeta-ai | 15 passed, 1 skipped | 2 s |
| vegeta-fidia | 71 passed, 4 skipped, 31 deselected | 58 s |
| vendor/ozaru | 32 passed | 0.6 s |
| components | 138 passed, 11 deselected | 103 s |
| assemblies | 16 passed | 23 s |
| designs/tests | 405 passed, 20 deselected | 9 min 24 s |
| races/tests | 42 passed | 47 s |
| side_labs/tests | 36 passed, 1 skipped, 1 deselected | 87 s |
| notebooks/tests | 4 passed | 4 s |

Problems that came from the environment, not from the code. They are fixed in the container and listed here because they will show up again on a fresh machine:

- **gmsh needs `libGLU.so.1`.** Install `libglu1-mesa`.
- **VTK off-screen renders segfault without a display.** This affects dedalus and aeromant viz. Run them under `xvfb-run`; `./test.sh --smoke` does this itself when `$DISPLAY` is unset.
- **MuJoCo EGL rendering needs Mesa EGL.** Install `libegl1`, `libegl-mesa0` and `libgl1-mesa-dri`. Without them, two of the race tests that render fail.

## Piccolo

| Suite | Result | Time |
|---|---|---|
| bigos | 31 passed | 3 s |
| clivia | 9 passed | 0.1 s |
| centaur | 182 passed | 14 s |
| silvan | 247 passed | 4 s |
| mission_planner | 140 passed | 0.7 s |
| drako (host tests) | 17 passed, 3 skipped | 8 s |
| iguana (host tests) | 19 passed, 2 skipped | 8 s |
| vendor/viridis/lab | 34 passed | 0.7 s |

Qt tests need `libegl1` and `libxkbcommon0` with `QT_QPA_PLATFORM=offscreen`.

## Smoke suites

| Repository | Command | Tests | Time |
|---|---|---|---|
| Vegeta (before the move) | `./test.sh --smoke` | 364 passed, 32 skipped | 26 s |
| Ozaru | `scripts/test.sh --smoke` | 43 passed | 0.9 s |
