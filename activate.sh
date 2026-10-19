#!/usr/bin/env bash
# Activate the ready Vegeta environment (nothing is installed):   source activate.sh
#   - activates the virtual environment (.venv, or $VEGETA_VENV)
#   - exports VEGETA_ROOT and puts the Vegeta source trees, workflows/ and designs/ on PYTHONPATH
#     (the libraries import even without a pip install of the distributions)
#   - defines `wf`  = the workflows tool (workflows/cli.py): wf list | wf info … | wf run … | wf runs | wf status … | wf cache
#     and   `wfrun` = a book's script directly (wfrun kestrel_airframe --chapter stall …)
#   - defines `yn`  = Yanus, the door keeper (vegeta-yanus): yn status | yn why 43 | yn doctor | yn pin … | yn new …
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
yn() { python -m vegeta.yanus "$@"; }
export VEGETA_YANUS_FLOW="$VEGETA_ROOT/workflows/flow.py"
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
export -f wf wfrun yn 2>/dev/null
cd "$VEGETA_ROOT" || return
# the Vegeta libraries and the workflow modules importable even without a pip install of the distributions
_PP=""
for _d in "$VEGETA_ROOT"/vegeta-cli/src "$VEGETA_ROOT"/vegeta-core/src "$VEGETA_ROOT"/vegeta-ai/src "$VEGETA_ROOT"/vegeta-fidia/src "$VEGETA_ROOT"/vegeta-yanus/src "$VEGETA_ROOT"/vendor/ozaru/porunga/src "$VEGETA_ROOT"/vendor/ozaru/mimesis/src "$VEGETA_ROOT"/vendor/ozaru/yanus/src "$VEGETA_ROOT"/workflows "$VEGETA_ROOT"/designs; do
  [ -d "$_d" ] && _PP="${_PP:+$_PP:}$_d"
done
case ":${PYTHONPATH:-}:" in *":$VEGETA_ROOT/workflows:"*) ;; *) export PYTHONPATH="${_PP}${PYTHONPATH:+:$PYTHONPATH}" ;; esac
python -c "import vegeta.aeromant, vegeta.talos, vegeta.boreas" 2>/dev/null || echo "activate.sh: the vegeta libraries do not import (missing dependencies?) — run ./install_local.sh or ./test.sh --quick" >&2
_of="$(python -c "from vegeta import aeromant; e=aeromant.OpenFOAMEnvironment.detect(); print(e.flavor(), e.version())" 2>/dev/null || echo "not found")"
_ccx="$(command -v ccx >/dev/null && echo found || echo "not found")"
echo "Vegeta: python=$(command -v python)  root=$VEGETA_ROOT  OpenFOAM: $_of  CalculiX: $_ccx"
echo "  wf                 interactive: the tree of books and chapters; pick one by code (K1, k2 …)"
echo "  wf list | wf runs | wf status <run> | wf tail <run> | wf cache | wf clean-cache"
echo "  wf K2 --detach --variant 4Q-2 --quality fine"
echo "  yn status | yn why 43 | yn doctor | yn pin 43 K1 --variant 4Q-2 <run>   (Yanus: what is ready, what blocks, what to run)"
