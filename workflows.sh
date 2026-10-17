#!/usr/bin/env bash
# Open the Vegeta workflows desktop interface (the PyQt6 window of workflows/gui):
#     ./workflows.sh              the window
#     ./workflows.sh cli [ARGS]   the command-line tool instead (workflows/cli.py …), e.g. ./workflows.sh cli list
#     ./workflows.sh help
# Uses the project's virtual environment (.venv, or $VEGETA_VENV). PyQt6 is installed into it on first use when
# missing (pip); the Vegeta libraries themselves are not installed here — run ./install_local.sh once for those.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

cmd="${1:-gui}"
[ $# -gt 0 ] && shift
case "$cmd" in
  gui | cli) ;;
  help | -h | --help) sed -n '2,7p' "$0"; exit 0 ;;
  *) echo "unknown command: $cmd (gui | cli | help)" >&2; exit 2 ;;
esac

VENV="${VEGETA_VENV:-$PWD/.venv}"
if [ -x "$VENV/bin/python" ]; then
  py="$VENV/bin/python"
else
  echo "no virtual environment at $VENV — run ./install_local.sh once (or export VEGETA_VENV=/path/to/venv)" >&2
  exit 1
fi
export VEGETA_ROOT="$PWD"
export PYTHONPATH="$PWD/vegeta-cli/src:$PWD/vegeta-core/src:$PWD/vegeta-ai/src:$PWD/vegeta-fidia/src:$PWD/workflows:$PWD/notebooks/designs${PYTHONPATH:+:$PYTHONPATH}"

if [ "$cmd" = cli ]; then
  exec "$py" workflows/cli.py "$@"
fi

qt_hint() {
  echo "Qt could not start. On Ubuntu install the libraries it needs:" >&2
  echo "  sudo apt install libegl1 libxkbcommon-x11-0 libxcb-cursor0" >&2
  echo "(or the distribution's Qt: sudo apt install python3-pyqt6, and set VEGETA_VENV to a venv made with --system-site-packages)" >&2
}

if ! "$py" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyQt6') else 1)"; then
  echo "Installing PyQt6 into $VENV (first use) ..."
  "$py" -m pip install --quiet --disable-pip-version-check -r workflows/requirements-gui.txt
fi
if ! err=$("$py" -c "from PyQt6 import QtWidgets" 2>&1); then
  echo "$err" >&2
  qt_hint
  exit 1
fi
status=0
"$py" workflows/gui "$@" || status=$?
[ "$status" -eq 134 ] && qt_hint        # abort: "could not load the Qt platform plugin"
exit "$status"
