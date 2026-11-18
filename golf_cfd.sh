#!/usr/bin/env bash
# The golf-ball air cannon's transient CFD (labs/aero/exp/golf_gun.py, on sonic_ball's moving-mesh rhoCentralFoam):
# the three cases (slow 25 m/s, subsonic fast 120 m/s, near sonic Mach 0.85), each fired through an ideal valve at
# the pressure that gives its speed, compared with the 0-D gun. Long: about 30 min (near sonic) to a few hours (slow)
# per case on 1 core with --mesh medium; cases run in parallel, one core each.
#
# Usage: ./golf_cfd.sh [--mesh smoke|medium|fine] [--case slow|subsonic_fast|near_sonic]... [--smoke] [--serial]
#   --smoke   0.3 ms of each case on the coarse mesh: checks OpenFOAM and the case set-up in a minute or two
# Writes labs/aero/exp/output/golf_gun/<mesh>/golf_cfd_results.json and prints it at the end: paste that back.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
MESH=medium; SMOKE=0; SERIAL=0; CASES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mesh) MESH="$2"; shift 2 ;;
    --case) CASES+=("$2"); shift 2 ;;
    --smoke) SMOKE=1; MESH=smoke; shift ;;
    --serial) SERIAL=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ ${#CASES[@]} -eq 0 ] && CASES=(slow subsonic_fast near_sonic)
source "${VEGETA_VENV:-$ROOT/.venv}/bin/activate"
# Aeromant finds OpenFOAM itself (a sourced environment, /opt/foam from install_local.sh, /usr/lib/openfoam*, ...)
python -c "from vegeta import aeromant; print('OpenFOAM:', aeromant.OpenFOAMEnvironment.detect())" \
  || { echo "no OpenFOAM found: install with ./install_local.sh or source your v2412 etc/bashrc" >&2; exit 1; }
cd "$ROOT/labs/aero/exp"
OUT="output/golf_gun/$MESH"; mkdir -p "$OUT"
run_one() {
  python - "$1" "$MESH" "$SMOKE" > "$OUT/$1.json" 2> "$OUT/$1.err" <<'PY'
import json, sys
import golf_gun as g
name, mesh, smoke = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
print(json.dumps(g.run_cfd([name], mesh=mesh, smoke=smoke), default=float))
PY
  echo "$1: done ($(tail -c 200 "$OUT/$1.err" | tail -1))"
}
for c in "${CASES[@]}"; do
  if [ $SERIAL -eq 1 ]; then run_one "$c"; else run_one "$c" & fi
done
wait
python - "$OUT" "${CASES[@]}" <<'PY'
import json, sys, pathlib
out, names = pathlib.Path(sys.argv[1]), sys.argv[2:]
merged = {"mesh": out.name, "cases": {}}
for n in names:
    try:
        merged["cases"].update(json.loads((out / f"{n}.json").read_text())["cases"])
    except Exception as e:
        merged["cases"][n] = {"error": repr(e), "log": (out / f"{n}.err").read_text()[-1500:]}
(out / "golf_cfd_results.json").write_text(json.dumps(merged, indent=1))
print("=== paste from here ===")
print(json.dumps(merged, separators=(",", ":")))
print("=== to here ===")
PY
