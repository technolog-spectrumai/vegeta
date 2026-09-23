# Philosophy

Vegeta is a collection of small Unix/Python engineering tools, not one large application.

## Local, modular, file-oriented
- Everything runs locally from Python. No server, database or daemon is needed to run an analysis.
- Packages communicate through standard engineering files: STEP, STL, Gmsh `.msh`, CalculiX
  `.inp`/`.frd`/`.dat`, OpenFOAM case directories, G-code, and JSON summaries.
- Native tool output is never hidden or deleted. `summary.json` only summarises it and points at it.

## Explicit engineering authority
- Supports, loads, materials, strengths, CFD values, templates and print settings are given by the
  engineer. The packages never invent them. A missing value is an error, not a default.
- Nothing is chained automatically: generating CAD does not mesh, meshing does not solve,
  FEA does not trigger CFD. Each step is an explicit call.
- Derived quantities are reported only when they are reliable. For example a safety factor is
  computed only when a yield strength was given; otherwise it is `None` with a message.

## Configuration in Python
All configuration is done through the Python API (dataclasses, keyword arguments). There are no
input config files for our own settings. Files that an external tool itself requires (a PrusaSlicer
`.ini`, an OpenFOAM dictionary) are generated from the Python values and kept as artifacts.

## Interfaces
1. Library API: all logic lives here.
2. CLI: thin argparse layer per package, loads Python files that define the objects, prints a table
   or `--json`, exit code reflects `status`.
3. GUI (later): a thin layer over the same API. For that reason the libraries never print, never call
   `plt.show()`, report progress through an optional callback and accept a cancel event for long runs.

## Failures
- Invalid explicit configuration raises `ValueError` immediately, with a message saying what is missing.
- Execution failures (tool not installed, non-zero exit, unreadable output) return a result with
  `status="failed"` and readable messages. A notebook session is never destroyed by a solver failure.
- Every external command records command line, working directory, return code, duration, stdout and
  stderr, and writes a log file next to the artifacts.

## Independence
Dedalus, Talos, Aeromant and Mellonia ship together in the `vegeta-cli` distribution as
`vegeta.dedalus`, `vegeta.talos`, `vegeta.aeromant` and `vegeta.mellonia`, but they do not import each
other, their parent namespace or any other part of Vegeta (each has a test enforcing this). Anything
that organises work across them (revisions, comparison) will depend on them, never the reverse.
They share a result *shape* by convention (see `result-shape.md`), not a shared framework; the small
helper modules (`result.py`, `_process.py`) are deliberately copied into each tool.
