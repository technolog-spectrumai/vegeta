#!/usr/bin/env bash
# Run the assemblies' tests and workflows and print a compact report to paste back.
#
# Usage: ./assembly_tests.sh [SECTION ...] [--python EXE] [--fidelity smoke|quick|full] [-j N] [--jobs N] [--threads N]
#                            [-- EXTRA workflow arguments, e.g. -- --no-cfd]
#   env          versions, git commit, which solvers are found (ccx, OpenFOAM, MuJoCo)                 (seconds)
#   unit         pytest assemblies -m "not slow": solvers off or faked; also checks that notebooks/,
#                scenarios/ and benchmark/ are untouched                                                (~8 min)
#   slow         pytest assemblies -m slow: the real FEA at smoke mesh, MuJoCo scenes, propulsor library (~10 min)
#   smoke        every workflow once at --fidelity smoke with the solvers ON, into runs/assemblies_smoke/
#                (its own .vida files; assemblies/data is not touched); CFD only where OpenFOAM is found (~minutes
#                without CFD; with OpenFOAM tens of minutes)
#   --- the real runs: write assemblies/data/<name>.vida (+ the JSON exports); resumable — a solved case or FEA model
#       is read back, a node whose key did not change is reused; --fidelity (default full) applies to all ---
#   microjet     python -m assemblies.workflows.microjet       turbojet: cycle, CFD, wheels' FEA       (hours with CFD)
#   propulsors   python -m assemblies.workflows.propulsors     propeller/exotic/EDF maps (full = notebooks 25/25b/25c)
#   merlin       python -m assemblies.workflows.merlin         airframe, wing FEA, polar CFD, race, mission (needs propulsors)
#   aguya        python -m assemblies.workflows.aguya          AGUYA on the microjet, race against MERLIN (needs the three above)
#   quadcopter   python -m assemblies.workflows.quadcopter     frame FEA, rotor and canopy CFD, blade FEA
#   fixed-wing   python -m assemblies.workflows.fixed_wing     wing FEA, aircraft CFD, rotor CFD, installed CFD
#   boat         python -m assemblies.workflows.boat           hull, double-body CFD, rotor CFD, bracket and blade FEA
#   submarine    python -m assemblies.workflows.submarine      hull CFD, pressure hull FEA, rotor CFD, blade FEA
#   rover        python -m assemblies.workflows.rover          terrains, arm and chassis FEA
#   onager       python -m assemblies.workflows.onager         4 variants, MuJoCo scenes, Sentinel leg FEA
#   walkers      python -m assemblies.workflows.walkers        dog, Cleopatra, Persephone, Apheloria pack/unpack
#   air          microjet propulsors merlin aguya quadcopter fixed-wing
#   water        boat submarine
#   ground       rover onager walkers
#   everything   air water ground
#   show         print the tree of every assemblies/data/*.vida (what is computed, reused, NOT RUN)
#   all (default) env unit
# Examples:
#   ./assembly_tests.sh                       # env + the fast tests
#   ./assembly_tests.sh slow smoke            # the real solvers at smoke settings, nothing in assemblies/data changes
#   ./assembly_tests.sh boat -j 8             # the boat at full fidelity, CFD on 8 cores
#   ./assembly_tests.sh ground -- --no-sim    # rover, Onager, walkers without the MuJoCo scenes
#   (options after -- go to every workflow that has them; the others drop them with a note)
#   ./assembly_tests.sh quadcopter -- --redo rotor_cfd   # solve one node again
# Everything is printed and also saved to assembly_tests_<date>.log.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=""
FIDELITY="full"
PROCS=1
JOBS=1
THREADS="$(nproc 2>/dev/null || echo 4)"
SECTIONS=()
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --fidelity) FIDELITY="$2"; shift 2 ;;
    -j) PROCS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    -h|--help) sed -n '2,39p' "$0"; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) SECTIONS+=("$1"); shift ;;
  esac
done
[ ${#SECTIONS[@]} -eq 0 ] && SECTIONS=(env unit)
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
LOG="$ROOT/assembly_tests_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export MPLBACKEND=Agg PYTHONWARNINGS="ignore::UserWarning" PYVISTA_OFF_SCREEN=true
cd "$ROOT"
status=0
SMOKE="$ROOT/runs/assemblies_smoke"
DATA="$ROOT/assemblies/data"

hdr() { printf '\n==================== %s ====================\n' "$1"; }
ok() { printf '  ok      %s\n' "$1"; }
bad() { printf '  FAILED  %s\n' "$1"; status=1; }
timed() { local t0=$SECONDS; "$@"; local rc=$?; printf '  (%s s)\n' $((SECONDS - t0)); return $rc; }
# MuJoCo renders nothing here, but Chiron may want a display: use xvfb-run when there is none and it is installed
headless() { if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null; then xvfb-run -a "$@"; else "$@"; fi; }

sec_env() {
  hdr "env"
  echo "  python   $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"
  echo "  git      $(git rev-parse --short HEAD 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null)$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo ' (uncommitted changes)')"
  echo "  cores    -j $PROCS (CFD), --jobs $JOBS, --threads $THREADS (FEA); fidelity $FIDELITY"
  echo "  ccx      $(command -v ccx || echo 'not found: FEA nodes stay NOT RUN')"
  echo "  OpenFOAM $(command -v simpleFoam || command -v foamRun || echo 'not found: CFD nodes stay NOT RUN')"
  "$PY" - <<'EOF'
import importlib
for m in ("numpy", "pandas", "cadquery", "gmsh", "pyvista", "mujoco", "vegeta.aeromant", "vegeta.talos", "vegeta.boreas", "vegeta.chiron", "assemblies"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:16s} MISSING ({type(e).__name__}: {e})")
EOF
}

sec_unit() {
  hdr "unit (solvers off or faked)"
  if timed "$PY" -m pytest assemblies -q -m "not slow" -p no:cacheprovider; then ok "assemblies fast tests"; else bad "assemblies fast tests"; fi
  if [ -z "$(git diff --stat notebooks scenarios benchmark)" ]; then ok "notebooks/ scenarios/ benchmark/ untouched"
  else bad "notebooks/ scenarios/ benchmark/ have changes:"; git diff --stat notebooks scenarios benchmark; fi
}

sec_slow() {
  hdr "slow (real FEA at smoke mesh, MuJoCo, libraries)"
  if timed headless "$PY" -m pytest assemblies -q -m slow -p no:cacheprovider -rs; then ok "assemblies slow tests"; else bad "assemblies slow tests"; fi
}

# wf NAME LABEL ARGS... : one workflow run; prints its tree. Options the workflow does not have are dropped with a
# note (so "ground -- --no-sim" reaches onager and walkers and leaves the rover alone).
wf() {
  local name="$1" label="$2"; shift 2
  local help args=() skip=0 a
  help="$("$PY" -m "assemblies.workflows.$name" --help 2>/dev/null)"
  for a in "$@"; do
    if [ $skip -eq 1 ]; then skip=0; continue; fi
    if [[ "$a" == --* ]] && ! grep -qE -- "(^|[ ,])${a}([ ,=]|$)" <<<"$help"; then
      echo "  ($name has no $a: dropped)"
      [[ "$a" != --no-* ]] && skip=1          # an option with a value: drop the value too
      continue
    fi
    args+=("$a")
  done
  if timed headless "$PY" -m "assemblies.workflows.$name" "${args[@]}"; then ok "$label"; else bad "$label"; fi
}

sec_smoke() {
  hdr "smoke: every workflow, solvers on, into $SMOKE"
  mkdir -p "$SMOKE"
  local c=(--fidelity smoke -j "$PROCS" --jobs "$JOBS" --threads "$THREADS" --no-export "${EXTRA[@]}")
  wf microjet   "microjet (smoke)"   "${c[@]}" --out "$SMOKE/microjet"   --vida "$SMOKE/microjet.vida"
  wf propulsors "propulsors (smoke)" --fidelity smoke --no-export --out "$SMOKE/propulsors" --vida "$SMOKE/propulsors.vida"
  wf merlin     "merlin (smoke)"     "${c[@]}" --out "$SMOKE/merlin"     --vida "$SMOKE/merlin.vida" --propulsors "$SMOKE/propulsors.vida"
  wf aguya      "aguya (smoke)"      "${c[@]}" --out "$SMOKE/aguya"      --vida "$SMOKE/aguya.vida" --engine "$SMOKE/microjet.vida" \
                                     --merlin "$SMOKE/merlin.vida" --propulsors "$SMOKE/propulsors.vida"
  local w
  for w in quadcopter fixed_wing boat submarine; do
    wf "$w" "$w (smoke)" "${c[@]}" --out "$SMOKE/$w" --vida "$SMOKE/$w.vida"
  done
  local g=(--fidelity smoke --threads "$THREADS" --no-export "${EXTRA[@]}")
  wf rover   "rover (smoke)"   "${g[@]}" --out "$SMOKE/rover"   --vida "$SMOKE/rover.vida"
  wf onager  "onager (smoke)"  "${g[@]}" --out "$SMOKE/onager"  --vida "$SMOKE/onager.vida"
  wf walkers "walkers (smoke)" "${g[@]}" --out "$SMOKE/walkers" --vida "$SMOKE/walkers.vida"
}

# the real runs: default paths (assemblies/data/<name>.vida), resumable
full() { local name="$1"; shift; wf "$name" "$name ($FIDELITY)" --fidelity "$FIDELITY" "$@" "${EXTRA[@]}"; }
sec_microjet()   { hdr "microjet ($FIDELITY)";   full microjet -j "$PROCS" --jobs "$JOBS" --threads "$THREADS"; }
sec_propulsors() { hdr "propulsors ($FIDELITY)"; full propulsors; }
sec_merlin()     { hdr "merlin ($FIDELITY)";     full merlin -j "$PROCS" --jobs "$JOBS" --threads "$THREADS"; }
sec_aguya()      { hdr "aguya ($FIDELITY)";      full aguya -j "$PROCS" --jobs "$JOBS" --threads "$THREADS"; }
sec_quadcopter() { hdr "quadcopter ($FIDELITY)"; full quadcopter -j "$PROCS" --jobs "$JOBS" --threads "$THREADS"; }
sec_fixed_wing() { hdr "fixed wing ($FIDELITY)"; full fixed_wing -j "$PROCS" --jobs "$JOBS" --threads "$THREADS"; }
sec_boat()       { hdr "boat ($FIDELITY)";       full boat -j "$PROCS" --jobs "$JOBS" --threads "$THREADS"; }
sec_submarine()  { hdr "submarine ($FIDELITY)";  full submarine -j "$PROCS" --jobs "$JOBS" --threads "$THREADS"; }
sec_rover()      { hdr "rover ($FIDELITY)";      full rover --threads "$THREADS"; }
sec_onager()     { hdr "onager ($FIDELITY)";     full onager --threads "$THREADS"; }
sec_walkers()    { hdr "walkers";                full walkers; }

sec_show() {
  hdr "show: assemblies/data/*.vida"
  local f name
  for f in "$DATA"/*.vida; do
    name="$(basename "$f" .vida)"
    [ -f "$ROOT/assemblies/workflows/$name.py" ] || continue
    "$PY" -m "assemblies.workflows.$name" --show || bad "show $name"
  done
}

run_section() {
  case "$1" in
    env) sec_env ;;
    unit) sec_unit ;;
    slow) sec_slow ;;
    smoke) sec_smoke ;;
    microjet) sec_microjet ;;
    propulsors) sec_propulsors ;;
    merlin) sec_merlin ;;
    aguya) sec_aguya ;;
    quadcopter) sec_quadcopter ;;
    fixed-wing|fixed_wing) sec_fixed_wing ;;
    boat) sec_boat ;;
    submarine) sec_submarine ;;
    rover) sec_rover ;;
    onager) sec_onager ;;
    walkers) sec_walkers ;;
    air) for s in microjet propulsors merlin aguya quadcopter fixed-wing; do run_section "$s"; done ;;
    water) for s in boat submarine; do run_section "$s"; done ;;
    ground) for s in rover onager walkers; do run_section "$s"; done ;;
    everything) for s in air water ground; do run_section "$s"; done ;;
    show) sec_show ;;
    all) sec_env; sec_unit ;;
    *) bad "unknown section $1 (see --help)" ;;
  esac
}

for s in "${SECTIONS[@]}"; do run_section "$s"; done
hdr "done"
[ $status -eq 0 ] && echo "  all ok" || echo "  some steps FAILED (see above)"
echo "  log: $LOG"
exit $status
