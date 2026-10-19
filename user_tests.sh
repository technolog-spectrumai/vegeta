#!/usr/bin/env bash
# Run the Chiron / Cleopatra / Persephone checks and benchmarks and print a compact report to paste back.
#
# Usage: ./user_tests.sh [SECTION ...] [--python EXE] [-j N] [-v|--verbose]
#   -v, --verbose   pytest's full output (tracebacks, prints); default: one line per test as it finishes
#   env               versions, git commit, imports                                         (seconds)
#   unit              chiron + cli + chronos test suites, designs tests          (~2-3 min)
#   physics           body-joint physics checks with their numbers: restoring torque,
#                     damping dissipation, c = 0 equivalence, model equivalence           (~1 min)
#   cleopatra-smoke   benchmark/cleopatra/full_benchmark.py --scale smoke                  (~5-10 min)
#   persephone-smoke  benchmark/persephone/full_benchmark.py --scale smoke                 (minutes; first run of Persephone)
#   cleopatra-pilot   ... --scale pilot (a few seeds of every experiment)                  (~1 h on 4 cores)
#   cleopatra         the FULL Cleopatra benchmark (6440 runs; several hours; resumable)
#   persephone        the FULL Persephone benchmark (1140 runs; long; resumable)
#   apheloria-movies  scenarios/apheloria_pack.py: Apheloria packs into its ball and unpacks (2 movies)
#   onager            Onager Sentinel: the ChironLab tests, then scenarios/onager_patrol.py (2 movies; ~5 min)
#   onager-atlas      Onager Atlas (forklift): tests, then scenarios/onager_atlas_pallet.py (1 movie; ~3 min)
#   onager-manus      Onager Manus (pincers): tests, then scenarios/onager_manus_tasks.py (1 movie; ~4 min)
#   onager-sweeper    Onager Sweeper (street cleaner): tests, then scenarios/onager_sweeper_street.py (1 movie; ~5 min)
#   velutina          Velutina (mountain medical courier): tests, then scenarios/velutina_mission.py (1 movie; ~2 min)
#   merlin            MERLIN (wildfire sampler): tests, then scenarios/merlin_mission.py (race table + 1 movie; ~3 min)
#   peregrine         PEREGRINE (folding-wing farm drone): the lattice, flight-model and CAD tests (~30 s)
#   drongo            Drongo (potato and cream delivery): tests, then scenarios/drongo_delivery.py (2 movies; ~4 min)
#   nisus             NISUS (OBS / Zero survey drone): tests, then scenarios/nisus_mission.py (6 missions + movies; ~25 min)
#   nisus-birds       NISUS-Zero bird photography: vegeta.mission tests, then scenarios/nisus_birds.py (6 missions + videos; ~30 min)
#   nisus-plus        NISUS+ (mountain survey drone): tests, then scenarios/nisus_plus_mission.py (6 mountain missions + movies; ~20 min)
#   nisus-plus-birds  Nisus+ Zero chasing birds in the mountains: scenarios/nisus_plus_birds.py (4 hunts + videos; ~25 min)
#   falco             FALCO (tractor mountain aircraft): tests, then scenarios/falco_mission.py (6 missions, the propeller parked; ~20 min)
#   falco-birds       Falco-Zero chasing birds in the mountains: scenarios/falco_birds.py (4 hunts + videos; ~25 min)
#   kestrel           Kestrel-4Q (biplane quadrotor tailsitter): tests, then scenarios/kestrel_transitions.py (the transition matrix; ~3 min; --cfd needs OpenFOAM)
#   jet               Microjet and AGUYA: cycle, compressible templates, thermal FEA deck, CAD and mission tests (~1 min)
#   geometry          components/ and assemblies/: geometry-only CAD (designs, arrangements; ~2 min, no solver)
#   notebooks-debug   every notebook end to end with FEA and CFD mocked (scripts/debug_notebooks.sh; minutes)
#   --- mock-only checks (no CFD, FEA or long simulations; the default) ---
#   ozaru             vendor/ozaru: Porunga (cache), Mimesis (mocks), Yanus (requirement checks)       (seconds)
#   designs           designs/tests, not the slow ones                                               (~10-20 min)
#   components        components/tests (geometry only)                                               (~2 min)
#   side-labs         side_labs/tests: run contract, campaigns, CLI, GUI, no notebook named here      (~3 min)
#   exchange          notebooks/tests: side lab -> Porunga key -> notebook, mock never read as real   (seconds)
#   cli-mock          vegeta-cli: the mocks, the cache and the CLI tests                               (~1 min)
#   notebooks-mock    every notebook but the slow ones, mocks on (scripts/debug_notebooks.sh)         (~30-60 min)
#   mock (default)    env ozaru designs components side-labs exchange cli-mock notebooks-mock
#   --- long runs (by name only) ---
#   all               env unit physics cleopatra-smoke persephone-smoke
# Results: benchmark/<robot>/results/<scale>/ (results.json, *_runs.csv, configs.csv, raw/, plots/, report.md).
# Everything is printed and also saved to user_tests_<date>.log.
set -uo pipefail
VERBOSE=0

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=""
JOBS="$(nproc 2>/dev/null || echo 4)"
SECTIONS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;; -v|--verbose) VERBOSE=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) SECTIONS+=("$1"); shift ;;
  esac
done
[ ${#SECTIONS[@]} -eq 0 ] && SECTIONS=(mock)
MOCK_SECTIONS=(env ozaru designs components side-labs exchange cli-mock notebooks-mock)
EXP=(); for s in "${SECTIONS[@]}"; do case "$s" in mock) EXP+=("${MOCK_SECTIONS[@]}") ;; all) EXP+=(env unit physics cleopatra-smoke persephone-smoke) ;; *) EXP+=("$s") ;; esac; done
SECTIONS=("${EXP[@]}")
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
LOG="$ROOT/user_tests_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export MPLBACKEND=Agg PYTHONWARNINGS="ignore::UserWarning" QT_QPA_PLATFORM=offscreen
# the source tree importable without an install (activate.sh does the same); the mocks' cache stays out of the repo
for _d in vegeta-cli/src vegeta-core/src vegeta-ai/src vegeta-fidia/src vegeta-yanus/src vendor/ozaru/porunga/src vendor/ozaru/mimesis/src vendor/ozaru/yanus/src side_labs designs; do
  case ":${PYTHONPATH:-}:" in *":$ROOT/$_d:"*) ;; *) PYTHONPATH="$ROOT/$_d${PYTHONPATH:+:$PYTHONPATH}" ;; esac
done
# the mock sections never start a solver or call the AI API: CalculiX, OpenFOAM, PrusaSlicer and XFOIL are refused at the
# point they would start (VEGETA_NO_SOLVERS=1), and the API key is hidden; the long sections (by name) put them back
KEY_SAVED="${ANTHROPIC_API_KEY:-}"; TOKEN_SAVED="${ANTHROPIC_AUTH_TOKEN:-}"
mock_env() {  # mock_env on|off — set before every section
  if [ "$1" = on ]; then export VEGETA_DEBUG=1 VEGETA_NO_SOLVERS=1; MOCK_PLUGIN="-p vegeta.mock.pytest_plugin"; unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
  else unset VEGETA_DEBUG VEGETA_NO_SOLVERS; MOCK_PLUGIN=""; [ -n "$KEY_SAVED" ] && export ANTHROPIC_API_KEY="$KEY_SAVED"; [ -n "$TOKEN_SAVED" ] && export ANTHROPIC_AUTH_TOKEN="$TOKEN_SAVED"; fi
}
export PYTHONPATH VEGETA_YANUS_FLOW="$ROOT/notebooks/flow.py" PORUNGA_DIR="${PORUNGA_DIR:-$(mktemp -d)/porunga}"
status=0

hdr() { printf '\n==================== %s ====================\n' "$1"; }
ok() { printf '  ok      %s\n' "$1"; }
bad() { printf '  FAILED  %s\n' "$1"; status=1; }
timed() { local t0=$SECONDS; "$@"; local rc=$?; printf '  (%s s)\n' $((SECONDS - t0)); return $rc; }

sec_env() {
  hdr "env"
  echo "  python   $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"
  echo "  git      $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) on $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)$( [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ] && echo ' (uncommitted changes)')"
  echo "  cpus     $JOBS"
  echo "  solvers  $( [ -n "${VEGETA_NO_SOLVERS:-}" ] && echo 'blocked (VEGETA_NO_SOLVERS=1: no CalculiX, OpenFOAM, PrusaSlicer, XFOIL)' || echo 'allowed')"
  echo "  AI API   $( [ -n "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}" ] && echo 'key set' || echo 'no key: scripted answers')"
  "$PY" - <<'EOF'
import importlib
for m in ("mujoco", "numpy", "scipy", "pandas", "matplotlib", "tqdm", "pyvista", "vegeta.chiron", "vegeta.dedalus"):
    try:
        mod = importlib.import_module(m)
        print(f"  ok      {m:<15} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  MISSING {m:<15} {type(e).__name__}: {e}")
EOF
}

pytest_summary() {  # pytest_summary <dir> <args...>
  # default: one line per test as it finishes (ok / FAILED / skip, the test, [n/N]), then the failures' short reasons;
  # -v / --verbose: pytest's full output (every test, the tracebacks, print output)
  local dir="$1"; shift
  local out rc; out="$(mktemp)"
  if [ "$VERBOSE" -eq 1 ]; then
    (cd "$dir" && "$PY" -m pytest -v -rfE -p no:cacheprovider ${MOCK_PLUGIN:-} "$@" 2>&1) | tee "$out" | sed 's/^/    /'
    rc=${PIPESTATUS[0]}
  else
    (cd "$dir" && "$PY" -m pytest -v -p no:cacheprovider ${MOCK_PLUGIN:-} -o console_output_style=count --tb=line "$@" 2>&1) | tee "$out" \
      | awk '/::/ && / (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)/ {
               st = ($0 ~ / PASSED/) ? "ok    " : ($0 ~ / (FAILED|ERROR)/) ? "FAILED" : "skip  ";
               n = ""; if (match($0, /\[ *[0-9]+\/[0-9]+\]/)) n = substr($0, RSTART, RLENGTH);
               printf "    %s %s %s\n", st, $1, n; fflush() }'
    rc=${PIPESTATUS[0]}
    grep -E "^(FAILED|ERROR) " "$out" | head -20 | sed 's/^/    /'
  fi
  echo "  $(tail -1 "$out")"
  rm -f "$out"
  return $rc
}

sec_unit() {
  hdr "unit tests"
  echo "  vegeta-cli: chiron, cli, chronos"
  timed pytest_summary "$ROOT/vegeta-cli" tests/chiron tests/cli tests/chronos || status=1
  echo "  designs (Cleopatra, the robot dog and the Onager series in ChironLab, plot helpers)"
  timed pytest_summary "$ROOT/designs" tests || status=1
}

sec_physics() {
  hdr "body-joint physics checks (numbers)"
  local out
  out="$(cd "$ROOT/designs" && "$PY" -m pytest -q -s -p no:cacheprovider tests/test_body_joint_physics.py 2>&1)"; local rc=$?
  echo "$out" | grep -vE "^\s*$|^\.+$|passed|warnings summary|^  /|DeprecationWarning" | head -60 | sed 's/^/  /'
  echo "  $(echo "$out" | tail -1)"
  [ $rc -eq 0 ] || status=1
}

bench() {  # bench <robot> <scale>
  local robot="$1" scale="$2"
  hdr "$robot benchmark, scale $scale ($JOBS processes)"
  local t0=$SECONDS
  "$PY" "$ROOT/benchmark/$robot/full_benchmark.py" --scale "$scale" -j "$JOBS" 2>&1 | tr '\r' '\n' | grep -vE "chiron trials:.*[0-9]+%\|.*\]$|^\s*$" | tail -15 | sed 's/^/  /'
  local rc=${PIPESTATUS[0]}
  printf '  (%s s, exit %s)\n' $((SECONDS - t0)) "$rc"
  local rep="$ROOT/benchmark/$robot/results/$scale/report.md"
  if [ -f "$rep" ]; then
    echo "  --- report.md (outcome tables and plot status) ---"
    grep -vE "^- !\[" "$rep" | head -120 | sed 's/^/  /'
    echo "  plots: $(ls "$ROOT/benchmark/$robot/results/$scale/plots/"*.png 2>/dev/null | wc -l) PNG in benchmark/$robot/results/$scale/plots/"
  else
    bad "no report.md"
  fi
  [ "$rc" -eq 0 ] || status=1
}

for s in "${SECTIONS[@]}"; do
  case " ${MOCK_SECTIONS[*]} " in *" $s "*) mock_env on ;; *) mock_env off ;; esac
  case "$s" in
    env) sec_env ;;
    ozaru)
      MOCK_PLUGIN="" ; hdr "Ozaru (vendor/ozaru): Porunga, Mimesis, Yanus"
      timed pytest_summary "$ROOT/vendor/ozaru" || status=1 ;;
    designs)
      hdr "designs/tests (not slow; mocks only, no solver)"
      timed pytest_summary "$ROOT" designs/tests -m "not slow" || status=1 ;;
    components)
      hdr "components/tests (geometry only)"
      timed pytest_summary "$ROOT" components/tests -m "not slow" || status=1 ;;
    side-labs)
      hdr "side_labs/tests: run contract, campaigns, CLI, GUI, separation from the notebooks (no solver run)"
      timed pytest_summary "$ROOT" side_labs/tests -m "not slow" || status=1 ;;
    exchange)
      hdr "notebooks/tests: the side-lab result found by its Porunga key; mock and real never mix"
      timed pytest_summary "$ROOT" notebooks/tests || status=1 ;;
    cli-mock)
      MOCK_PLUGIN=""                       # these tests install the mocks themselves; the solvers stay blocked
      hdr "vegeta-cli: the mocks, the result cache, the CLI"
      timed pytest_summary "$ROOT/vegeta-cli" tests/mock tests/cache tests/cli || status=1 ;;
    notebooks-mock)
      hdr "notebooks with the mocks on (every one but the slow ones; scripts/debug_notebooks.sh --all for those)"
      t0=$SECONDS; "$ROOT/scripts/debug_notebooks.sh" -j "$JOBS" 2>&1 | tail -70 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s)\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    unit) sec_unit ;;
    physics) sec_physics ;;
    cleopatra-smoke) bench cleopatra smoke ;;
    persephone-smoke) bench persephone smoke ;;
    cleopatra-pilot) bench cleopatra pilot ;;
    persephone-pilot) bench persephone pilot ;;
    cleopatra) bench cleopatra full ;;
    persephone) bench persephone full ;;
    all) sec_env; sec_unit; sec_physics; bench cleopatra smoke; bench persephone smoke ;;
    apheloria-movies)
      hdr "Apheloria pack / unpack (MuJoCo, 2 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/apheloria_pack.py" 2>&1 | grep -v "^\s*$" | tail -40 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    onager)
      hdr "Onager Sentinel: ChironLab tests"
      timed pytest_summary "$ROOT/designs" tests/test_onager_chiron.py || status=1
      hdr "Onager Sentinel on patrol: FL motor off, RR wheel seized — drag vs the three-wheel limp (MuJoCo, 2 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/onager_patrol.py" 2>&1 | grep -v "^\s*$" | tail -40 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    onager-atlas|onager-manus|onager-sweeper)
      name=${s#onager-}
      case "$name" in atlas) script=onager_atlas_pallet.py ;; manus) script=onager_manus_tasks.py ;; *) script=onager_sweeper_street.py ;; esac
      hdr "Onager ${name}: ChironLab tests"
      timed pytest_summary "$ROOT/designs" "tests/test_onager_${name}.py" || status=1
      hdr "Onager ${name} mission in MuJoCo (1 movie)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/$script" 2>&1 | grep -v "^\s*$" | tail -25 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movie in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    jet)
      hdr "Microjet and AGUYA: cycle, templates, thermal FEA, CAD, mission"
      timed pytest_summary "$ROOT/vegeta-cli" tests/boreas/test_microjet.py tests/aeromant/test_compressible.py tests/talos/test_thermal.py || status=1
      timed pytest_summary "$ROOT/designs" tests/test_turbojet.py tests/test_aguya.py || status=1 ;;
    geometry)
      hdr "Components and assemblies: geometry only"
      timed pytest_summary "$ROOT" components/tests assemblies/tests -m "not slow" || status=1 ;;
    notebooks-debug)
      hdr "Notebooks in DEBUG mode (vegeta.mock: FEA and CFD mocked)"
      t0=$SECONDS; "$ROOT/scripts/debug_notebooks.sh" -j "$JOBS" 2>&1 | tail -60 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s)\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    velutina)
      hdr "Velutina: design and flight-model tests"
      timed pytest_summary "$ROOT/designs" tests/test_velutina.py || status=1
      hdr "Velutina flies medical aid to the mountain rescue site (hand-over at the wall; 1 movie)"
      t0=$SECONDS; "$PY" "$ROOT/scenarios/velutina_mission.py" 2>&1 | grep -v "^\s*$" | tail -10 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movie in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    merlin)
      hdr "MERLIN: design, race and plume tests"
      timed pytest_summary "$ROOT/designs" tests/test_merlin.py tests/test_merlin_race.py || status=1
      hdr "MERLIN races to a fire 20 km out and samples its smoke (1 movie)"
      t0=$SECONDS; "$PY" "$ROOT/scenarios/merlin_mission.py" 2>&1 | grep -v "^\s*$\|movie frames" | tail -16 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movie in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    peregrine)
      hdr "PEREGRINE: vortex lattice, flight models, CAD and movie tests"
      timed pytest_summary "$ROOT/designs" tests/test_peregrine.py || status=1 ;;
    drongo)
      hdr "Drongo: design, flight and delivery tests"
      timed pytest_summary "$ROOT/designs" tests/test_drongo.py || status=1
      hdr "Drongo delivers a potato and a cream: drop into the net, place on the zone (2 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/drongo_delivery.py" 2>&1 | grep -v "^\s*$" | tail -10 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    nisus)
      hdr "NISUS: geometry, mass, energy, aerodynamics, structure and simulation tests"
      timed pytest_summary "$ROOT/designs" tests/test_nisus.py || status=1
      hdr "NISUS survey missions: OBS and Zero, calm / headwind / gusts / Jetson failure (MuJoCo, 6 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/nisus_mission.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -12 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies, telemetry and nisus_mission.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    nisus-birds)
      hdr "NISUS bird photography: the mission software (vegeta.mission) and the MuJoCo seam"
      timed pytest_summary "$ROOT/vegeta-cli" tests/mission || status=1
      timed pytest_summary "$ROOT/designs" tests/test_nisus_birds.py -m "not slow" || status=1
      hdr "NISUS-Zero bird missions: reference, stock lens, Nano B01, pigeons, wind, Jetson failure (MuJoCo, 6 videos)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/nisus_birds.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -14 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) videos, photo tables and nisus_birds.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    nisus-plus)
      hdr "NISUS+: geometry, atmosphere, mass, pack, drive and brake, flight, structure and simulation tests"
      timed pytest_summary "$ROOT/designs" tests/test_nisus_plus.py tests/test_frame_study.py -m "not slow" || status=1
      hdr "NISUS+ mountain missions: calm / ridge lift / lee downdraft / hot day / storm escape / Jetson failure (MuJoCo, 6 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/nisus_plus_mission.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -12 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies, telemetry and nisus_plus_mission.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    nisus-plus-birds)
      hdr "Nisus+ Zero bird hunts in the mountains: eagles in the ridge lift, vultures, choughs, Jetson failure (MuJoCo, 4 videos)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/nisus_plus_birds.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -16 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) videos, photo tables and nisus_plus_birds.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    falco)
      hdr "FALCO: the tractor's layout and parked propeller, CAD, mass, the sourced motor's fit, the drive, one fin, structure, the parking in MuJoCo"
      timed pytest_summary "$ROOT/designs" tests/test_falco.py -m "not slow" || status=1
      hdr "FALCO mountain missions: calm / ridge lift / lee downdraft / hot day / storm escape / Jetson failure, the propeller parked for the landing (MuJoCo, 6 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/falco_mission.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -12 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies, telemetry and falco_mission.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    falco-birds)
      hdr "Falco-Zero bird hunts in the mountains: eagles in the ridge lift, vultures, choughs, Jetson failure (MuJoCo, 4 videos)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/falco_birds.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -16 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) videos, photo tables and falco_birds.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    kestrel)
      hdr "Kestrel-4Q: zero-stagger layout, CAD, masses, the hold point (allocation rank), the biplane lattice, the transitions both ways, the CFD cases, the truss"
      timed pytest_summary "$ROOT/designs" tests/test_kestrel.py tests/test_prop_lab.py tests/test_kestrel_fea.py tests/test_kestrel_report.py tests/test_kestrel2.py -m "not slow" || status=1
      hdr "Kestrel workflows: the run-directory contract on a fake solver, the campaigns, missing prerequisites, the legacy import, no solver from the notebook"
      timed pytest_summary "$ROOT/side_labs" tests -m "not slow" || status=1   # includes the CLI
      hdr "Kestrel-4Q transitions through the stall: 3 vehicles x forward/back x 1.5/3/5 s x still air/6 m/s headwind (model polar)"
      t0=$SECONDS; "$PY" "$ROOT/scenarios/kestrel_transitions.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -20 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) tables, figures and kestrel_transitions.json in scenarios/output/kestrel/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    *) echo "unknown section: $s (see --help)"; status=2 ;;
  esac
done
hdr "done (exit $status) — log: $LOG"
exit $status
