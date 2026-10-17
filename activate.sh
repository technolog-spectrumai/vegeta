#!/usr/bin/env bash
# Activate the ready Vegeta environment (nothing is installed):   source activate.sh
#   - activates the virtual environment (.venv, or $VEGETA_VENV)
#   - puts the repository root first on the path and exports VEGETA_ROOT
#   - defines `wf`  = the workflows tool (workflows/cli.py): wf list | wf info … | wf run … | wf runs | wf status … | wf cache
#     and   `wfrun` = a book's script directly (wfrun kestrel_airframe --chapter stall …)
# Nothing is run; sourcing only arranges the environment.
if [ -n "${BASH_SOURCE:-}" ]; then _VEGETA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; else _VEGETA_ROOT="$(pwd)"; fi
export VEGETA_ROOT="$_VEGETA_ROOT"
_VENV="${VEGETA_VENV:-$VEGETA_ROOT/.venv}"
if [ -f "$_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$_VENV/bin/activate"
else
  echo "activate.sh: no virtual environment at $_VENV — run ./install_local.sh once (or export VEGETA_VENV=...)" >&2
  return 1 2>/dev/null || exit 1
fi
if [ -n "${BASH_VERSION:-}" ] && [ "${0##*/}" = "activate.sh" ]; then
  echo "activate.sh: source it, do not run it:   source activate.sh" >&2
  exit 1
fi
wf() { python "$VEGETA_ROOT/workflows/cli.py" "$@"; }
wfrun() {
  local book="$1"; shift
  local script
  script="$(python - "$book" <<'PY'
import sys
sys.path.insert(0, sys.argv[0] and __import__("os").environ["VEGETA_ROOT"] + "/workflows")
import library
b = sys.argv[1]
print(library.BOOKS[b]["script"] if b in library.BOOKS else library.IMPORTS.get(b, {}).get("script", ""))
PY
)"
  if [ -z "$script" ]; then echo "wfrun: unknown book '$book' (wf list)" >&2; return 2; fi
  python "$VEGETA_ROOT/$script" "$@"
}
export -f wf wfrun 2>/dev/null
cd "$VEGETA_ROOT" || return
python -c "import vegeta.aeromant, vegeta.talos" 2>/dev/null || echo "activate.sh: vegeta-cli is not importable from $_VENV — run ./install_local.sh" >&2
_of="$(python -c "from vegeta import aeromant; e=aeromant.OpenFOAMEnvironment.detect(); print(e.flavor(), e.version())" 2>/dev/null || echo "not found")"
_ccx="$(command -v ccx >/dev/null && echo found || echo "not found")"
echo "Vegeta: python=$(command -v python)  root=$VEGETA_ROOT  OpenFOAM: $_of  CalculiX: $_ccx"
echo "  wf                 interactive: pick a book, a chapter, run / prepare / list"
echo "  wf list | wf runs | wf status <run> | wf tail <run> | wf cache | wf clean-cache"
echo "  wf run kestrel_airframe --detach --variant 4Q-2 --chapter stall --quality fine"
