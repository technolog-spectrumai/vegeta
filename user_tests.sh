#!/usr/bin/env bash
# Check that everything works — with mocks only. Never runs CFD, FEA, slicing or the AI API for real:
# VEGETA_DEBUG=1 (vegeta.mock: fixed numbers in seconds, Chiron episodes cut short) and VEGETA_NO_SOLVERS=1 (CalculiX,
# OpenFOAM, PrusaSlicer and XFOIL refused if anything tries to start them); the Anthropic key is hidden.
#
# Usage: ./user_tests.sh [SECTION|LAB ...] [--no-sims] [--notebooks] [-v] [--python EXE]
#   (none)       env layout smoke libs aero land legged marine studio export      (~15-30 min)
#   env          versions, git commit, the imports
#   layout       tools/labs.py: the lab rules (libs never import labs, labs never each other, nothing imports exp/)
#                and every machine's lifecycle stage
#   smoke        ./test.sh --smoke (imports, CLIs, boundaries, goldens, Yax export)                         (~1 min)
#   libs         the libraries' own tests: vegeta-cli tools, core, ai, fidia, ogma, labkit, Ozaru, Yax
#   aero land legged marine studio
#                one lab: its tests (not the slow ones), then its sims/ scripts (mocked: movies of a few frames)
#   export       every registered machine through Yax to USD (tools/export.py), Drongo with its rotors
#   notebooks    every lab's studies/ notebooks headless (also with --notebooks)                      (long)
#   --no-sims    the labs without their sims/ scripts
#   -v           pytest's full output
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=""; VERBOSE=0; SIMS=1; NOTEBOOKS=0; SECTIONS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;; -v|--verbose) VERBOSE=1; shift ;; --no-sims) SIMS=0; shift ;;
    --notebooks) NOTEBOOKS=1; shift ;; -h|--help) sed -n '2,21p' "$0"; exit 0 ;; *) SECTIONS+=("$1"); shift ;;
  esac
done
[ ${#SECTIONS[@]} -eq 0 ] && SECTIONS=(env layout smoke libs aero land legged marine studio export)
[ "$NOTEBOOKS" -eq 1 ] && SECTIONS+=(notebooks)
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
LOG="$ROOT/user_tests_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"
export VEGETA_DEBUG=1 VEGETA_NO_SOLVERS=1 MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen PYVISTA_OFF_SCREEN=true PYTHONWARNINGS="ignore::UserWarning"
export PORUNGA_DIR="${PORUNGA_DIR:-$(mktemp -d)/porunga}"
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
MOCK="-p vegeta.mock.pytest_plugin"
status=0; FAILED=()

hdr() { printf '\n==================== %s ====================\n' "$1"; }
timed() { local t0=$SECONDS; "$@"; local rc=$?; printf '  (%s s)\n' $((SECONDS - t0)); return $rc; }
mark() { [ "$1" -eq 0 ] || { status=1; FAILED+=("$2"); }; }
pyt() {  # pyt <label> <pytest args...>
  local label="$1"; shift
  if [ "$VERBOSE" -eq 1 ]; then
    "$PY" -m pytest -v -rfE -p no:cacheprovider --import-mode=importlib $MOCK -m "not slow" "$@"; local rc=$?
  else
    local out; out="$("$PY" -m pytest -q -p no:cacheprovider --import-mode=importlib $MOCK -m "not slow" -o console_output_style=classic --tb=line "$@" 2>&1)"; local rc=$?
    echo "$out" | grep -E "^(FAILED|ERROR)|^/.*Error|passed|failed|no tests ran" | tail -25 | sed 's/^/  /'
  fi
  [ "$rc" -eq 5 ] && rc=0                                  # nothing collected
  mark "$rc" "$label"
}

sec_env() {
  hdr "env"
  echo "  python   $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"
  echo "  git      $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo ' (uncommitted changes)')"
  echo "  mocks    VEGETA_DEBUG=1, solvers blocked (VEGETA_NO_SOLVERS=1), no AI key"
  "$PY" - <<'EOF'
import importlib
for m in ("numpy", "scipy", "pandas", "matplotlib", "cadquery", "gmsh", "pyvista", "mujoco", "pxr", "networkx",
          "vegeta.dedalus", "vegeta.talos", "vegeta.aeromant", "vegeta.chiron", "vegeta.labkit", "ogma", "yax"):
    try:
        mod = importlib.import_module(m)
        print(f"  ok      {m:<16} {getattr(mod, '__version__', '')}  {getattr(mod, '__file__', '') or ''}"[:130])
    except Exception as e:
        print(f"  MISSING {m:<16} {type(e).__name__}: {e} (rerun ./install_local.sh)")
EOF
}

sec_layout() {
  hdr "layout: the lab rules and the lifecycle"
  "$PY" tools/labs.py check | sed 's/^/  /'; mark "${PIPESTATUS[0]}" layout
  "$PY" tools/labs.py stages | sed 's/^/  /'
}

sec_smoke() {
  hdr "smoke (./test.sh --smoke)"
  timed "$ROOT/test.sh" --smoke --python "$PY" 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed|races:|smoke:" | tail -20 | sed 's/^/  /'
  mark "${PIPESTATUS[0]}" smoke
}

sec_libs() {
  hdr "libs: the libraries' own tests (mocked)"
  timed pyt libs libs/vegeta-cli/tests libs/vegeta-core/tests libs/vegeta-ai/tests libs/vegeta-fidia/tests libs/ogma/tests libs/labkit/tests
  hdr "vendor: Ozaru and Yax"
  (cd vendor/ozaru && timed pyt ozaru .)
  (cd vendor/yax && timed pyt yax .)
}

sec_lab() {
  local lab="$1"
  hdr "lab $lab: tests"
  timed pyt "$lab tests" "labs/$lab"
  if [ "$SIMS" -eq 1 ] && ls "labs/$lab/sims/"*.py >/dev/null 2>&1; then
    hdr "lab $lab: sims/ scripts (mocked)"
    timed "$PY" tools/lab.py run "$lab" sims; mark $? "$lab sims"
  fi
}

sec_export() {
  hdr "export: every registered machine through Yax"
  timed pyt export tools/tests/test_export.py tests/labs
  local out; out="$(mktemp -d)"
  "$PY" tools/export.py drongo --lump --to "$out/drongo.usda" --also sdf 2>&1 | sed 's/^/  /'; mark "${PIPESTATUS[0]}" "export drongo"
}

sec_notebooks() {
  for lab in aero land legged marine studio; do
    hdr "lab $lab: studies/ notebooks (mocked)"
    timed "$PY" tools/lab.py run "$lab" studies; mark $? "$lab notebooks"
  done
}

for s in "${SECTIONS[@]}"; do
  case "$s" in
    env) sec_env ;; layout) sec_layout ;; smoke) sec_smoke ;; libs) sec_libs ;; export) sec_export ;;
    notebooks) sec_notebooks ;;
    aero|land|legged|marine|studio) sec_lab "$s" ;;
    *) echo "unknown section: $s (see ./user_tests.sh --help)"; status=1 ;;
  esac
done

hdr "summary"
if [ "$status" -eq 0 ]; then echo "  all passed (mocks only)"; else echo "  FAILED: ${FAILED[*]}"; fi
echo "  log: $LOG"
exit "$status"
