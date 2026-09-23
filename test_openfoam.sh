#!/usr/bin/env bash
# Test Vegeta's CFD tool (Aeromant) against the OpenFOAM installations on THIS machine and print the result.
#   1. finds installations (a sourced environment, /usr/lib/openfoam*, /opt/openfoam*, /opt/OpenFOAM-*,
#      conda environments, the Ubuntu package) or uses the ones you pass; shows version, flavour
#      (openfoam.com / openfoam.org) and which executables each one provides
#   2. for every flavour found: prepares and runs the sphere case of each template through every
#      pipeline step (laminar Re = 100 checked against the Schiller-Naumann drag correlation; RANS k-omega
#      SST as a pipeline smoke test); logs are kept when something fails
#   3. --pytest: additionally runs vegeta-cli/tests/aeromant against those installations
# Exit code 0 when every flavour found passes. Nothing is installed; files go to a temporary work directory.
#
# Usage: ./test_openfoam.sh [options]
#   --bashrc FILE     an installation to test (its etc/bashrc), repeatable; e.g. /opt/openfoam14/etc/bashrc
#   --prefix "CMD"    a launcher for an installation, repeatable; e.g. "micromamba run -p /opt/foam"
#   --flavor com|org|all   test only openfoam.com or only openfoam.org installations (default: all found)
#   --quick           mesh only (blockMesh, features, snappyHexMesh, checkMesh); no solver run
#   --pytest          also run the Aeromant test suite (integration tests need OpenFOAM)
#   --timeout S       seconds allowed per pipeline step (default 1800)
#   --work DIR        work directory (default: a temporary one); --keep keeps it after a successful run
#   --venv DIR, --python EXE   which Python to use (default: ./.venv from install_local.sh)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${VEGETA_VENV:-$ROOT/.venv}"
PY=""
FLAVOR="all"
QUICK=0
KEEP=0
PYTEST=0
TIMEOUT=1800
WORK=""
BASHRCS=()
PREFIXES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --bashrc) BASHRCS+=("$2"); shift 2 ;;
    --prefix) PREFIXES+=("$2"); shift 2 ;;
    --flavor) FLAVOR="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --keep) KEEP=1; shift ;;
    --pytest) PYTEST=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
case "$FLAVOR" in com|org|all) ;; *) echo "--flavor must be com, org or all" >&2; exit 2 ;; esac
if [ -z "$PY" ]; then
  if [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"
  else echo "No virtual environment at $VENV — run ./install_local.sh first (or pass --python)." >&2; exit 1; fi
fi
if ! "$PY" -c "import vegeta.aeromant" 2>/dev/null; then
  echo "vegeta.aeromant is not importable with $PY — run ./install_local.sh (or pass --python)." >&2; exit 1
fi
if [ -z "$WORK" ]; then WORK="$(mktemp -d -t vegeta-openfoam.XXXXXX)"; TEMP_WORK=1; else mkdir -p "$WORK"; TEMP_WORK=0; fi
for f in "${BASHRCS[@]:-}"; do
  [ -n "$f" ] && [ ! -f "$f" ] && { echo "--bashrc $f: no such file" >&2; exit 2; }
done

echo "Vegeta OpenFOAM check"
echo "  python: $PY ($("$PY" --version 2>&1))"
echo "  work:   $WORK"
export VEGETA_OF_BASHRCS="$(printf '%s\n' "${BASHRCS[@]:-}")"
export VEGETA_OF_PREFIXES="$(printf '%s\n' "${PREFIXES[@]:-}")"
export VEGETA_OF_FLAVOR="$FLAVOR" VEGETA_OF_QUICK="$QUICK" VEGETA_OF_TIMEOUT="$TIMEOUT"
status=0
"$PY" - "$WORK" <<'PYEOF' || status=1
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from vegeta import aeromant
from vegeta.aeromant import CFDCase, OpenFOAMEnvironment
from vegeta.aeromant.stl import Surface, write_stl_ascii

work = Path(sys.argv[1])
want = {"com": "openfoam.com", "org": "openfoam.org", "all": None}[os.environ["VEGETA_OF_FLAVOR"]]
quick = os.environ["VEGETA_OF_QUICK"] == "1"
timeout = float(os.environ["VEGETA_OF_TIMEOUT"])
failed = False
EXES = ("blockMesh", "snappyHexMesh", "checkMesh", "surfaceFeatureExtract", "surfaceFeatures", "simpleFoam", "foamRun")
NEEDED = {"openfoam.com": ("blockMesh", "surfaceFeatureExtract", "snappyHexMesh", "checkMesh", "simpleFoam"),
          "openfoam.org": ("blockMesh", "surfaceFeatures", "snappyHexMesh", "checkMesh", "foamRun")}


def row(state, label, text=""):
    print(f"  {state:<8} {label:<34} {text}", flush=True)


def launch(env):
    if env.bashrc:
        return f"source {env.bashrc}"
    if env.prefix:
        return " ".join(env.prefix)
    if env.env.get("WM_PROJECT_DIR"):
        return f"PATH with WM_PROJECT_DIR={env.env['WM_PROJECT_DIR']}"
    return "already sourced environment (PATH)"


def executables(env):
    """Which OpenFOAM executables this installation itself provides."""
    script = "; ".join(
        f'if [ -n "$FOAM_APPBIN" ]; then test -x "$FOAM_APPBIN/{x}" && echo ok:{x} || echo no:{x}; '
        f'else command -v {x} >/dev/null 2>&1 && echo ok:{x} || echo no:{x}; fi' for x in EXES)
    try:
        r = subprocess.run(env.command(["bash", "-c", script]), env={**os.environ, **env.env},
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"could not run bash through this installation: {exc}"
    found = {line.split(":", 1)[1]: line.startswith("ok:") for line in r.stdout.splitlines() if ":" in line}
    return found, (r.stderr.strip().splitlines() or [""])[-1] if not found else ""


def icosphere(radius=0.5, level=4):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    V = np.array(verts) * radius
    return Surface(V[np.array(f)], "sphere")


def schiller_naumann(re):
    return 24.0 / re * (1 + 0.15 * re ** 0.687)


# ---- 1. installations ------------------------------------------------------------------------------------
print()
print("Installations")
envs = []
for line in os.environ.get("VEGETA_OF_BASHRCS", "").splitlines():
    if line.strip():
        envs.append(("given", OpenFOAMEnvironment(bashrc=line.strip())))
for line in os.environ.get("VEGETA_OF_PREFIXES", "").splitlines():
    if line.strip():
        envs.append(("given", OpenFOAMEnvironment(prefix=shlex.split(line.strip()))))
envs += [("found", e) for e in OpenFOAMEnvironment.candidates()]
seen, chosen = set(), {}
for origin, env in envs:
    key = launch(env)
    if key in seen:
        continue
    seen.add(key)
    version, flavor = env.version(), env.flavor()
    exes, err = executables(env)
    have = [x for x in EXES if exes.get(x)]
    label = f"{flavor or 'unknown flavour'} {version or ''}".strip()
    if err and not exes:
        row("FAILED", label, f"{key}: {err}")
        continue
    missing = [x for x in NEEDED.get(flavor, ()) if not exes.get(x)]
    if flavor is None:
        row("-", label, f"{key}: WM_PROJECT_VERSION not recognised; executables: {', '.join(have) or 'none'}")
    elif missing:
        row("-", label, f"{key}: incomplete, missing {', '.join(missing)}")
    elif want and flavor != want:
        row("-", label, f"{key}: skipped (--flavor)")
    elif flavor in chosen:
        row("-", label, f"{key}: not used (another {flavor} installation is tested)")
    else:
        chosen[flavor] = env
        row("ok", label, f"{key} ({origin}); {', '.join(have)}")
if not chosen:
    failed = True
    hint = ("no usable OpenFOAM installation found. Pass one with --bashrc /path/to/OpenFOAM/etc/bashrc or "
            "--prefix \"micromamba run -p /opt/foam\"; install_local.sh puts conda-forge OpenFOAM under /opt/foam.")
    row("FAILED", "openfoam", hint)

# ---- 2. the sphere cases, per flavour ----------------------------------------------------------------------
stl = write_stl_ascii(icosphere(0.5, 4), work / "sphere.stl")
common = dict(velocity=1.0, density=1.0, reference_area=math.pi / 4, reference_length=1.0, center_of_rotation=(0, 0, 0))
CASES = [
    ("laminar_external", dict(common, kinematic_viscosity=0.01),
     lambda m: (abs(m["Cd"] - schiller_naumann(100.0)) / schiller_naumann(100.0) <= 0.10,
                f"Cd {m['Cd']:.3f} vs Schiller-Naumann {schiller_naumann(100.0):.3f} (10 % allowed), Cl {m['Cl']:+.3f}")),
    # the RANS template is a pipeline smoke test here (a bluff sphere is not a steady-RANS validation case):
    # every step must run and the coefficients must come out finite; convergence is reported, not required
    ("rans_ksst_external", dict(common, kinematic_viscosity=1e-5, iterations=300, residual_target=1e-4),
     lambda m: (all(math.isfinite(m.get(k, float("nan"))) for k in ("Cd", "Cl")),
                f"pipeline and coefficients ok: Cd {m['Cd']:.3f}, Cl {m['Cl']:+.3f} (Re 1e5 sphere, 300 iterations, values not validated)")),
]


def tail(path, n=3):
    try:
        lines = [l.rstrip() for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
        return " | ".join(lines[-n:])[-300:]
    except OSError:
        return ""


summary = {}
for flavor, env in chosen.items():
    print()
    print(f"{flavor} {env.version() or ''}: {launch(env)}")
    ok_all = True
    for template, params, verdict in CASES:
        label = template
        case = CFDCase(template, stl, params, work / flavor.replace(".", "_") / template, geometry_units="m", environment=env)
        prep = case.prepare(overwrite=True)
        if not prep.ok:
            row("FAILED", f"{label} prepare", "; ".join(prep.messages)[:200])
            ok_all = False
            continue
        steps = case.template.step_names(case.flavor)
        if quick:
            steps = steps[:steps.index("checkMesh") + 1]
        t0 = time.monotonic()
        res = case.run(steps=steps, timeout=timeout)
        for name, rec in zip(steps, res.execution):        # one record per step, in pipeline order
            if rec.ok:
                row("ok", f"{label} {name}", f"{rec.duration_s:.0f} s")
            else:
                why = rec.error or f"exit {rec.returncode}"
                row("FAILED", f"{label} {name}", f"{why}; log {rec.log_file}: {tail(rec.log_file)}" if rec.log_file else why)
        m = res.metrics
        if not res.ok:
            row("FAILED", f"{label} result", "; ".join(res.messages)[:300])
            ok_all = False
            continue
        if quick:
            row("ok", f"{label} mesh", f"{m.get('mesh_cells')} cells, checkMesh {'ok' if m.get('mesh_ok') else 'reported failed checks'} "
                                       f"({time.monotonic() - t0:.0f} s)")
            continue
        good, text = verdict(m)
        conv = "converged" if m.get("converged") else f"not converged in {m.get('iterations')} iterations"
        row("ok" if good else "FAILED", f"{label} result", f"{text}; {m.get('mesh_cells')} cells, {conv} ({time.monotonic() - t0:.0f} s)")
        if not good:
            ok_all = False
        summary[f"{flavor} {template}"] = {k: m.get(k) for k in ("Cd", "Cl", "mesh_cells", "converged", "iterations")}
    if not ok_all:
        failed = True
(work / "summary.json").write_text(json.dumps({"chosen": {f: e.describe() for f, e in chosen.items()}, "results": summary,
                                              "quick": quick}, indent=2, default=str))
sys.exit(1 if failed else 0)
PYEOF

if [ "$PYTEST" -eq 1 ]; then
  echo
  echo "Aeromant test suite (vegeta-cli/tests/aeromant)"
  [ ${#BASHRCS[@]} -gt 0 ] && export AEROMANT_TEST_OPENFOAM_BASHRC="${BASHRCS[0]}"
  [ ${#PREFIXES[@]} -gt 0 ] && export AEROMANT_TEST_OPENFOAM_PREFIX="${PREFIXES[0]}"
  (cd "$ROOT/vegeta-cli" && "$PY" -m pytest -q tests/aeromant) || status=1
fi

echo
if [ $status -eq 0 ]; then
  echo "RESULT: OK — Aeromant works with the OpenFOAM installation(s) above."
  if [ "$TEMP_WORK" -eq 1 ] && [ "$KEEP" -eq 0 ]; then rm -rf "$WORK"; else echo "  files kept in $WORK"; fi
else
  echo "RESULT: PROBLEMS FOUND — see FAILED above. Case files and logs are kept in $WORK"
  echo "  (send that directory, or its log.* files, along with this output when reporting the problem)."
fi
exit $status
