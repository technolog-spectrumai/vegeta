#!/usr/bin/env bash
# Check the Vegeta installation and print the result.
#   1. components: Python packages, external tools, the vegeta command, Jupyter kernel
#   2. (default) a short working run through every tool: CAD -> FEA -> slicing -> CFD mesh (-> core)
# Exit code 0 when everything needed works; tools that are not installed are reported as MISSING.
#
# Usage: ./test.sh [--quick] [--venv DIR] [--python EXE]
#   --quick        component checks only (a few seconds)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${VEGETA_VENV:-$ROOT/.venv}"
PY=""
QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$PY" ]; then
  if [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"
  else echo "No virtual environment at $VENV — run ./install_local.sh first (or pass --python)." >&2; exit 1; fi
fi
BIN="$(dirname "$PY")"
status=0

row() { printf '  %-8s %-13s %s\n' "$1" "$2" "$3"; }
check() {  # check <label> <command...>
  local label="$1"; shift
  local out
  if out="$("$@" 2>&1)"; then row ok "$label" "$(echo "$out" | head -1)"
  else row MISSING "$label" "$(echo "$out" | tail -1)"; status=1; fi
}
optional() {  # like check, but absence is not an error
  local label="$1"; shift
  local out
  if out="$("$@" 2>&1)"; then row ok "$label" "$(echo "$out" | head -1)"
  else row "-" "$label" "not installed (optional)"; fi
}

echo "Vegeta installation check"
echo "  python: $PY ($("$PY" --version 2>&1))"
echo
echo "Components"
check cadquery    "$PY" -c "import cadquery; print(cadquery.__version__)"
check gmsh        "$PY" -c "import gmsh; print(gmsh.__version__)"
check calculix    bash -c "ccx -v | grep -i version"
check prusaslicer bash -c "prusa-slicer --help | head -1"
check openfoam    "$PY" -c "
import os, subprocess
from vegeta import aeromant
e = aeromant.OpenFOAMEnvironment.detect()
r = subprocess.run(e.command(['simpleFoam', '-help']), env={**os.environ, **e.env}, capture_output=True, text=True)
assert r.returncode == 0, (r.stderr or r.stdout)[-300:]
print('via', e.bashrc or ' '.join(e.prefix) or 'PATH')"
for tool in dedalus talos aeromant mellonia; do
  check "$tool" "$PY" -c "from vegeta import $tool; print('vegeta.$tool', $tool.__version__)"
done
check core        "$PY" -c "from vegeta import core; print('vegeta.core', core.__version__)"
check vegeta      "$BIN/vegeta" --version
check jupyter     "$BIN/jupyter" lab --version
optional kernel   bash -c "'$BIN/jupyter' kernelspec list 2>/dev/null | grep -E '^\s*vegeta\s' | awk '{print \"Python (vegeta) ->\", \$2}' | grep ."

if [ "$QUICK" -eq 0 ]; then
  echo
  echo "Working run (temporary files, about 10 s)"
  WORK="$(mktemp -d)"
  "$PY" - "$WORK" <<'PYEOF' || status=1
import sys
from pathlib import Path

work = Path(sys.argv[1])
failed = False


def row(state, label, text):
    print(f"  {state:<8} {label:<13} {text}", flush=True)


def report(label, res, text):
    global failed
    if res is not None and res.ok:
        row("ok", label, text())
    else:
        failed = True
        row("FAILED", label, "; ".join(res.messages)[:200] if res is not None else "not run")


from vegeta import aeromant, mellonia, talos
from vegeta.dedalus.examples import CantileverBeam
from vegeta.mellonia.examples import GENERIC_PLA_0_2MM

beam = CantileverBeam().generate(length=100, width=10, height=10)
cad = beam.export(work / "cad")
report("CAD", cad, lambda: f"beam volume {beam.volume:.0f} mm^3 (expected 10000)")

fea = None
if cad.ok:
    model = talos.StructuralModel(
        cad.artifacts["step"], "mm-N-MPa", talos.Material("steel", 210000, 0.3),
        regions=[talos.SurfacesOnPlane("fixed", "x", 0), talos.SurfacesOnPlane("tip", "x", 100)],
        supports=[talos.FixedSupport("fixed")], loads=[talos.Force("tip", fz=-100)],
        mesh_settings=talos.MeshSettings(5))
    mesh = model.mesh(work / "fea")
    fea = model.solve(work / "fea") if mesh.ok else mesh
theory = 100 * 100 ** 3 / (3 * 210000 * 10 * 10 ** 3 / 12)
report("FEA", fea, lambda: f"tip deflection {-fea.metrics['displacement_min'][2]:.4f} mm (beam theory {theory:.4f})")

prn = mellonia.slice_stl(cad.artifacts["stl"], GENERIC_PLA_0_2MM, mellonia.Orientation(), work / "print") if cad.ok else None
report("print", prn, lambda: f"{prn.metrics['layer_count']} layers, {prn.metrics['estimated_time']}")

cfd = None
try:
    env = aeromant.OpenFOAMEnvironment.detect()
    case = aeromant.CFDCase(
        "laminar_external_simplefoam", cad.artifacts["stl"],
        dict(velocity=1.0, kinematic_viscosity=1e-3, density=1.0, reference_area=1e-4, reference_length=0.1,
             center_of_rotation=(0, 0, 0)),
        workdir=work / "cfd", geometry_units="mm", environment=env)
    prep = case.prepare()
    cfd = case.run(steps=["blockMesh", "checkMesh"]) if prep.ok else prep
except RuntimeError as exc:
    failed = True
    row("FAILED", "CFD", str(exc)[:200])
else:
    report("CFD", cfd, lambda: f"OpenFOAM mesh with {cfd.metrics.get('mesh_cells')} cells")

try:
    from vegeta import core
except ImportError:
    row("-", "core", "vegeta-core not installed (optional)")
else:
    ws = core.Workspace.create(work / "ws")
    r1 = ws.add_design("beam", "vegeta.dedalus.examples:CantileverBeam").new_revision(length=100.0)
    gen = r1.generate()
    r2 = r1.branch(height=12.0)
    report("core", gen, lambda: f"revisions r1, r2; r2 CAD is {ws.status().rows[1]['CAD']}")

sys.exit(1 if failed else 0)
PYEOF
  rm -rf "$WORK"
fi

echo
if [ $status -eq 0 ]; then
  echo "RESULT: OK — Vegeta is installed and working."
else
  echo "RESULT: PROBLEMS FOUND — see MISSING/FAILED above (./install_local.sh installs the missing parts)."
fi
exit $status
