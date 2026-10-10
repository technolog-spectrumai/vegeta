#!/usr/bin/env bash
# Run the Chiron / Cleopatra / Persephone checks and benchmarks and print a compact report to paste back.
#
# Usage: ./user_tests.sh [SECTION ...] [--python EXE] [-j N]
#   env               versions, git commit, imports                                         (seconds)
#   unit              chiron + cli + chronos test suites, notebooks/designs tests          (~2-3 min)
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
#   jet               Microjet and AGUYA: cycle, compressible templates, thermal FEA deck, CAD and mission tests (~1 min)
#   geometry          components/ and assemblies/: geometry-only CAD (designs, arrangements; ~2 min, no solver)
#   notebooks-debug   every notebook end to end with FEA and CFD mocked (scripts/debug_notebooks.sh; minutes)
#   all (default)     env unit physics cleopatra-smoke persephone-smoke
# Results: benchmark/<robot>/results/<scale>/ (results.json, *_runs.csv, configs.csv, raw/, plots/, report.md).
# Everything is printed and also saved to user_tests_<date>.log.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=""
JOBS="$(nproc 2>/dev/null || echo 4)"
SECTIONS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) SECTIONS+=("$1"); shift ;;
  esac
done
[ ${#SECTIONS[@]} -eq 0 ] && SECTIONS=(env unit physics cleopatra-smoke persephone-smoke)
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
LOG="$ROOT/user_tests_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export MPLBACKEND=Agg PYTHONWARNINGS="ignore::UserWarning"
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
  local dir="$1"; shift
  local out
  out="$(cd "$dir" && "$PY" -m pytest -q -p no:cacheprovider "$@" 2>&1)"; local rc=$?
  echo "$out" | grep -E "^(FAILED|ERROR)" | head -20 | sed 's/^/    /'
  echo "  $(echo "$out" | tail -1)"
  return $rc
}

sec_unit() {
  hdr "unit tests"
  echo "  vegeta-cli: chiron, cli, chronos"
  timed pytest_summary "$ROOT/vegeta-cli" tests/chiron tests/cli tests/chronos || status=1
  echo "  notebooks/designs (Cleopatra, the robot dog and the Onager series in ChironLab, plot helpers)"
  timed pytest_summary "$ROOT/notebooks/designs" tests || status=1
}

sec_physics() {
  hdr "body-joint physics checks (numbers)"
  local out
  out="$(cd "$ROOT/notebooks/designs" && "$PY" -m pytest -q -s -p no:cacheprovider tests/test_body_joint_physics.py 2>&1)"; local rc=$?
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
  case "$s" in
    env) sec_env ;;
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
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_onager_chiron.py || status=1
      hdr "Onager Sentinel on patrol: FL motor off, RR wheel seized — drag vs the three-wheel limp (MuJoCo, 2 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/onager_patrol.py" 2>&1 | grep -v "^\s*$" | tail -40 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    onager-atlas|onager-manus|onager-sweeper)
      name=${s#onager-}
      case "$name" in atlas) script=onager_atlas_pallet.py ;; manus) script=onager_manus_tasks.py ;; *) script=onager_sweeper_street.py ;; esac
      hdr "Onager ${name}: ChironLab tests"
      timed pytest_summary "$ROOT/notebooks/designs" "tests/test_onager_${name}.py" || status=1
      hdr "Onager ${name} mission in MuJoCo (1 movie)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/$script" 2>&1 | grep -v "^\s*$" | tail -25 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movie in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    jet)
      hdr "Microjet and AGUYA: cycle, templates, thermal FEA, CAD, mission"
      timed pytest_summary "$ROOT/vegeta-cli" tests/boreas/test_microjet.py tests/aeromant/test_compressible.py tests/talos/test_thermal.py || status=1
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_turbojet.py tests/test_aguya.py || status=1 ;;
    geometry)
      hdr "Components and assemblies: geometry only"
      timed pytest_summary "$ROOT" components/tests assemblies/tests -m "not slow" || status=1 ;;
    notebooks-debug)
      hdr "Notebooks in DEBUG mode (vegeta.mock: FEA and CFD mocked)"
      t0=$SECONDS; "$ROOT/scripts/debug_notebooks.sh" -j "$JOBS" 2>&1 | tail -60 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s)\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    velutina)
      hdr "Velutina: design and flight-model tests"
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_velutina.py || status=1
      hdr "Velutina flies medical aid to the mountain rescue site (hand-over at the wall; 1 movie)"
      t0=$SECONDS; "$PY" "$ROOT/scenarios/velutina_mission.py" 2>&1 | grep -v "^\s*$" | tail -10 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movie in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    merlin)
      hdr "MERLIN: design, race and plume tests"
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_merlin.py tests/test_merlin_race.py || status=1
      hdr "MERLIN races to a fire 20 km out and samples its smoke (1 movie)"
      t0=$SECONDS; "$PY" "$ROOT/scenarios/merlin_mission.py" 2>&1 | grep -v "^\s*$\|movie frames" | tail -16 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movie in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    peregrine)
      hdr "PEREGRINE: vortex lattice, flight models, CAD and movie tests"
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_peregrine.py || status=1 ;;
    drongo)
      hdr "Drongo: design, flight and delivery tests"
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_drongo.py || status=1
      hdr "Drongo delivers a potato and a cream: drop into the net, place on the zone (2 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/drongo_delivery.py" 2>&1 | grep -v "^\s*$" | tail -10 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    nisus)
      hdr "NISUS: geometry, mass, energy, aerodynamics, structure and simulation tests"
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_nisus.py || status=1
      hdr "NISUS survey missions: OBS and Zero, calm / headwind / gusts / Jetson failure (MuJoCo, 6 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/nisus_mission.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -12 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies, telemetry and nisus_mission.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    nisus-birds)
      hdr "NISUS bird photography: the mission software (vegeta.mission) and the MuJoCo seam"
      timed pytest_summary "$ROOT/vegeta-cli" tests/mission || status=1
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_nisus_birds.py -m "not slow" || status=1
      hdr "NISUS-Zero bird missions: reference, stock lens, Nano B01, pigeons, wind, Jetson failure (MuJoCo, 6 videos)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/nisus_birds.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -14 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) videos, photo tables and nisus_birds.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    nisus-plus)
      hdr "NISUS+: geometry, atmosphere, mass, pack, drive and brake, flight, structure and simulation tests"
      timed pytest_summary "$ROOT/notebooks/designs" tests/test_nisus_plus.py tests/test_frame_study.py -m "not slow" || status=1
      hdr "NISUS+ mountain missions: calm / ridge lift / lee downdraft / hot day / storm escape / Jetson failure (MuJoCo, 6 movies)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/nisus_plus_mission.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -12 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) movies, telemetry and nisus_plus_mission.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    nisus-plus-birds)
      hdr "Nisus+ Zero bird hunts in the mountains: eagles in the ridge lift, vultures, choughs, Jetson failure (MuJoCo, 4 videos)"
      XV=""; command -v xvfb-run >/dev/null && [ -z "${DISPLAY:-}" ] && XV="xvfb-run -a"
      t0=$SECONDS; $XV "$PY" "$ROOT/scenarios/nisus_plus_birds.py" 2>&1 | grep -v "^\s*$\|WARN" | tail -16 | sed 's/^/  /'
      rc=${PIPESTATUS[0]}; printf '  (%s s, exit %s) videos, photo tables and nisus_plus_birds.json in scenarios/output/\n' $((SECONDS - t0)) "$rc"; [ "$rc" -eq 0 ] || status=1 ;;
    *) echo "unknown section: $s (see --help)"; status=2 ;;
  esac
done
hdr "done (exit $status) — log: $LOG"
exit $status
