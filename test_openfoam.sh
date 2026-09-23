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
#   --pytest          also run the Aeromant test suite; it picks one installation per flavour itself
#                     (from the first --bashrc / --prefix given, else the usual places); -m "not slow" with --quick
#   --timeout S       seconds allowed per pipeline step (default 1800)
#   --work DIR        work directory (default: a temporary one)
#   --keep            keep the work directory after a successful run (it is always kept on failure)
#   --venv DIR, --python EXE   which Python to use (default: ./.venv from install_local.sh)
# A full run takes about 5-10 minutes per installation; --quick about two.
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
need() { [ $# -ge 2 ] && [ "${2#--}" = "$2" ] || { echo "$1 needs a value (see --help)" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --bashrc) need "$@"; BASHRCS+=("$2"); shift 2 ;;
    --prefix) need "$@"; PREFIXES+=("$2"); shift 2 ;;
    --flavor) need "$@"; FLAVOR="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --keep) KEEP=1; shift ;;
    --pytest) PYTEST=1; shift ;;
    --timeout) need "$@"; TIMEOUT="$2"; shift 2 ;;
    --work) need "$@"; WORK="$2"; shift 2 ;;
    --venv) need "$@"; VENV="$2"; shift 2 ;;
    --python) need "$@"; PY="$2"; shift 2 ;;
    -h|--help) awk 'NR > 1 && /^set -/ {exit} NR > 1 {print}' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
case "$FLAVOR" in com|org|all) ;; *) echo "--flavor must be com, org or all" >&2; exit 2 ;; esac
case "$TIMEOUT" in ''|*[!0-9.]*|0|0.*) echo "--timeout must be a positive number of seconds" >&2; exit 2 ;; esac
if [ -z "$PY" ]; then
  if [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"
  else echo "No virtual environment at $VENV — run ./install_local.sh first (or pass --python)." >&2; exit 1; fi
fi
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")" ;; *) PY="$(command -v "$PY" || echo "$PY")" ;; esac
if ! msg="$("$PY" -c "import vegeta.aeromant" 2>&1)"; then
  echo "vegeta.aeromant is not importable with $PY (${msg##*$'\n'}) — run ./install_local.sh (or pass --python)." >&2; exit 1
fi
ABS=()
for f in "${BASHRCS[@]:-}"; do
  [ -z "$f" ] && continue
  [ -f "$f" ] || { echo "--bashrc $f: no such file" >&2; exit 2; }
  ABS+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")")
done
BASHRCS=("${ABS[@]:-}")
if [ -z "$WORK" ]; then WORK="$(mktemp -d -t vegeta-openfoam.XXXXXX)" || { echo "could not create a work directory" >&2; exit 1; }; TEMP_WORK=1
else mkdir -p "$WORK" || exit 1; TEMP_WORK=0; fi

echo "Vegeta OpenFOAM check"
echo "  python: $PY ($("$PY" --version 2>&1))"
echo "  work:   $WORK"
if [ "$QUICK" -eq 1 ]; then echo "  mode:   --quick (meshing only, about 2 minutes per installation)"
else echo "  mode:   full (about 5-10 minutes per installation)"; fi
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
# executables each flavour needs, taken from the templates' own pipelines
NEEDED = {fl: tuple(dict.fromkeys(s.argv[0] for t in aeromant.TEMPLATES.values() for s in t.flavor(fl).pipeline
                                  if s.kind == "openfoam")) for fl in ("openfoam.com", "openfoam.org")}
EXES = tuple(dict.fromkeys(x for v in NEEDED.values() for x in v))


def row(state, label, text=""):
    print(f"  {state:<8} {label:<42} {text}", flush=True)


def launch(env):
    if env.bashrc:
        return f"source {env.bashrc}"
    if env.prefix:
        return " ".join(env.prefix)
    if env.env.get("WM_PROJECT_DIR"):
        return f"PATH with WM_PROJECT_DIR={env.env['WM_PROJECT_DIR']}"
    if os.environ.get("WM_PROJECT_DIR"):
        return f"already sourced in this shell (WM_PROJECT_DIR={os.environ['WM_PROJECT_DIR']})"
    return "executables on PATH"


def identity(env):
    return (env.bashrc, tuple(env.prefix), tuple(sorted(env.env.items())))


def executables(env):
    """Which OpenFOAM executables this installation itself provides."""
    script = 'echo appbin:$FOAM_APPBIN; ' + "; ".join(
        f'if [ -n "$FOAM_APPBIN" ]; then test -x "$FOAM_APPBIN/{x}" && echo ok:{x} || echo no:{x}; '
        f'else command -v {x} >/dev/null 2>&1 && echo ok:{x} || echo no:{x}; fi' for x in EXES)
    try:
        r = subprocess.run(env.command(["bash", "-c", script]), env={**os.environ, **env.env},
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, "", f"could not run bash through this installation: {exc}"
    found, appbin = {}, ""
    for line in r.stdout.splitlines():
        if line.startswith("appbin:"):
            appbin = line[7:].strip()
        elif line.startswith(("ok:", "no:")):
            found[line[3:]] = line.startswith("ok:")
    err = "" if found else (r.stderr.strip().splitlines() or [r.stdout.strip() or "no output"])[-1]
    return found, appbin, err


def foam_error(log_path, fallback=""):
    """The FOAM FATAL message of a log (plus the next lines), else the last non-empty line."""
    try:
        lines = [l.rstrip() for l in Path(log_path).read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return fallback
    for i in range(len(lines) - 1, -1, -1):
        if "FOAM FATAL" in lines[i]:
            return " | ".join(lines[i:i + 3])[:240]
    body = [l for l in lines if not l.startswith(("=", "#"))]
    return (" | ".join(body[-2:]) or fallback)[-240:]


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
seen, chosen, skipped, given_bad = set(), {}, 0, 0
for origin, env in envs:
    key = launch(env)
    if identity(env) in seen:
        continue
    seen.add(identity(env))
    version, flavor = env.version(), env.flavor()
    exes, appbin, err = executables(env)
    have = [x for x in EXES if exes.get(x)]
    label = f"{flavor or 'unknown flavour'} {version or ''}".strip()
    where = f"{key} ({origin})"
    unusable = "FAILED" if origin == "given" else "-"
    if err and not exes:
        row(unusable, label, f"{where}: {err}")
        given_bad += origin == "given"
        continue
    missing = [x for x in NEEDED.get(flavor, ()) if not exes.get(x)]
    if flavor is None:
        row(unusable, label, f"{where}: WM_PROJECT_VERSION not recognised; executables: {', '.join(have) or 'none'}")
        given_bad += origin == "given"
    elif missing:
        loc = f" in {appbin}" if appbin else ""
        row(unusable, label, f"{where}: incomplete, missing {', '.join(missing)}{loc}" +
            (" (FOAM_APPBIN directory does not exist: not compiled?)" if appbin and not Path(appbin).is_dir() else ""))
        given_bad += origin == "given"
    elif want and flavor != want:
        row("-", label, f"{where}: skipped (--flavor)")
        skipped += 1
    elif flavor in chosen:
        row("-", label, f"{where}: not used (another {flavor} installation is tested first)")
    else:
        chosen[flavor] = env
        row("ok", label, f"{where}; {', '.join(have)}")
if given_bad:
    failed = True
if not chosen:
    failed = True
    if given_bad:
        hint = "the installation(s) you passed cannot be used (see above)"
    elif skipped:
        hint = f"no usable {want} installation; {skipped} installation(s) of the other flavour skipped by --flavor"
    else:
        hint = ("no usable OpenFOAM installation found; pass one with --bashrc /opt/openfoam14/etc/bashrc (openfoam.org) "
                "or --prefix \"micromamba run -p /opt/foam\" (openfoam.com); install_local.sh puts conda-forge OpenFOAM under /opt/foam")
    row("FAILED", "openfoam", hint)

# ---- 2. the sphere cases, per flavour ----------------------------------------------------------------------
stl = write_stl_ascii(icosphere(0.5, 4), work / "sphere.stl")
common = dict(velocity=1.0, density=1.0, reference_area=math.pi / 4, reference_length=1.0, center_of_rotation=(0, 0, 0))
CASES = [
    ("laminar_external", dict(common, kinematic_viscosity=0.01),
     lambda m: (abs(m["Cd"] - schiller_naumann(100.0)) / schiller_naumann(100.0) <= 0.10 and abs(m["Cl"]) < 0.05,
                f"Cd {m['Cd']:.3f} vs Schiller-Naumann {schiller_naumann(100.0):.3f} (10 % allowed), Cl {m['Cl']:+.3f} (|Cl| < 0.05)")),
    # the RANS template is a pipeline smoke test here (a bluff sphere is not a steady-RANS validation case):
    # every step must run and the coefficients must come out finite; convergence is reported, not required
    ("rans_ksst_external", dict(common, kinematic_viscosity=1e-5, iterations=300, residual_target=1e-4),
     lambda m: (all(math.isfinite(m.get(k, float("nan"))) for k in ("Cd", "Cl")),
                f"pipeline and coefficients ok: Cd {m.get('Cd_mean_last50', m['Cd']):.3f} +/- {m.get('Cd_std_last50', 0):.3f}, Cl {m['Cl']:+.3f} "
                f"(Re 1e5 sphere on the default mesh; the value itself is not validated)")),
]


def tail(path, n=3):
    try:
        lines = [l.rstrip() for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
        return " | ".join(lines[-n:])[-300:]
    except OSError:
        return ""


summary, verdicts = {}, {}
for flavor, env in chosen.items():
    print()
    print(f"{flavor} {env.version() or ''}: {launch(env)}")
    ok_all, notes = True, []
    for template, params, verdict in CASES:
        label = template
        try:
            case = CFDCase(template, stl, params, work / flavor.replace(".", "_") / template, geometry_units="m", environment=env)
            prep = case.prepare(overwrite=True)
            if not prep.ok:
                row("FAILED", f"{label} prepare", "; ".join(prep.messages)[:200])
                ok_all = False
                continue
            steps = case.template.step_names(case.flavor)
            if quick:
                steps = steps[:steps.index("checkMesh") + 1]
            exe = {st.name: (st.argv[0] if st.kind == "openfoam" else "copy 0.orig -> 0") for st in case.files.pipeline}
            t0 = time.monotonic()
            res = case.run(steps=steps, timeout=timeout)
            m = res.metrics
            crashed = False
            for name, rec in zip(steps, res.execution):        # one record per step, in pipeline order
                shown = f"{label} {name}" + (f" ({exe[name]})" if exe[name] != name else "")
                if name == "checkMesh" and rec.error is None and m.get("mesh_cells") is not None:
                    if m.get("mesh_ok"):
                        row("ok", shown, f"{m['mesh_cells']} cells, mesh OK ({rec.duration_s:.1f} s)")
                    else:
                        first = next((x for x in res.messages if "checkMesh" in x), "failed checks")
                        row("-", shown, f"{m['mesh_cells']} cells, {first.split(':', 1)[-1].strip()[:160]} ({rec.duration_s:.1f} s)")
                elif rec.ok:
                    row("ok", shown, f"{rec.duration_s:.1f} s")
                else:
                    crashed = True
                    why = rec.error or f"exit {rec.returncode}"
                    detail = foam_error(rec.log_file, (rec.stderr.strip() or rec.stdout.strip()).splitlines()[-1:] and
                                        (rec.stderr.strip() or rec.stdout.strip()).splitlines()[-1] or "") if rec.log_file else ""
                    row("FAILED", shown, f"{why}: {detail}" if detail else why)
                    if rec.log_file:
                        print(f"           log: {rec.log_file}")
            if crashed or not res.ok:
                if not res.ok:
                    row("FAILED", f"{label} result", "; ".join(x.splitlines()[0] for x in res.messages)[:300])
                ok_all = False
                notes.append(f"{template}: failed")
                continue
            if quick:
                summary[f"{flavor} {template}"] = {k: m.get(k) for k in ("mesh_cells", "mesh_ok", "mesh_failed_checks")}
                if m.get("mesh_cells") is None:
                    row("FAILED", f"{label} mesh", "no cell count from checkMesh")
                    ok_all = False
                    notes.append(f"{template}: no mesh")
                else:
                    row("ok", f"{label} mesh", f"{m['mesh_cells']} cells in {time.monotonic() - t0:.0f} s (solver not run: --quick)")
                    notes.append(f"{template}: {m['mesh_cells']} cells")
                continue
            if m.get("Cd") is None or m.get("Cl") is None:
                row("FAILED", f"{label} result", "no Cd/Cl in the force-coefficient output: " + "; ".join(res.messages)[:200])
                ok_all = False
                notes.append(f"{template}: no coefficients")
                continue
            good, text = verdict(m)
            conv = "converged" if m.get("converged") else f"not converged after {m.get('iterations') or params['iterations']} iterations"
            row("ok" if good else "FAILED", f"{label} result", f"{text}; {conv} ({time.monotonic() - t0:.0f} s)")
            if not good:
                ok_all = False
            notes.append(f"{template}: Cd {m['Cd']:.3f}" + ("" if good else " (FAILED)"))
            summary[f"{flavor} {template}"] = {k: m.get(k) for k in ("Cd", "Cl", "mesh_cells", "converged", "iterations")}
        except Exception as exc:                                    # keep going with the next template / flavour
            row("FAILED", f"{label} result", f"{type(exc).__name__}: {str(exc)[:200]}")
            ok_all = False
            notes.append(f"{template}: {type(exc).__name__}")
    verdicts[flavor] = (ok_all, notes)
    if not ok_all:
        failed = True

print()
print("Summary")
for flavor, (ok_all, notes) in verdicts.items():
    row("ok" if ok_all else "FAILED", f"{flavor} {chosen[flavor].version() or ''}".strip(), "; ".join(notes))
if want and want not in verdicts:
    row("-", want, "no installation tested")
(work / "summary.json").write_text(json.dumps({"chosen": {f: e.describe() for f, e in chosen.items()}, "results": summary,
                                              "quick": quick}, indent=2, default=str))
sys.exit(1 if failed else 0)
PYEOF

if [ "$PYTEST" -eq 1 ]; then
  echo
  echo "Aeromant test suite (vegeta-cli/tests/aeromant; it picks one installation per flavour itself)"
  if ! "$PY" -c "import pytest" 2>/dev/null; then
    echo "  pytest is not installed for $PY (pip install -e 'vegeta-cli[test]')"; status=1
  else
    [ -n "${BASHRCS[0]:-}" ] && export AEROMANT_TEST_OPENFOAM_BASHRC="${BASHRCS[0]}"
    [ -n "${PREFIXES[0]:-}" ] && export AEROMANT_TEST_OPENFOAM_PREFIX="${PREFIXES[0]}"
    MARK=(); [ "$QUICK" -eq 1 ] && MARK=(-m "not slow")
    (cd "$ROOT/vegeta-cli" && "$PY" -m pytest -q "${MARK[@]}" tests/aeromant) || status=1
  fi
fi

echo
if [ $status -eq 0 ]; then
  if [ "$QUICK" -eq 1 ]; then echo "RESULT: OK — meshing works with the installation(s) above (solver not run; rerun without --quick for the full check)."
  else echo "RESULT: OK — Aeromant works with the OpenFOAM installation(s) above."; fi
  if [ "$TEMP_WORK" -eq 1 ] && [ "$KEEP" -eq 0 ]; then rm -rf "$WORK"; else echo "  files kept in $WORK"; fi
else
  echo "RESULT: PROBLEMS FOUND — see FAILED above."
  if ls -d "$WORK"/*/ >/dev/null 2>&1; then
    echo "  Case files and logs are kept in $WORK (send that directory, or its log.* files, with this output)."
  else
    echo "  Nothing was run; fix the installation problem above and rerun."
  fi
fi
exit $status
