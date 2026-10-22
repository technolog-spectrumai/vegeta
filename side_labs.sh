#!/usr/bin/env bash
# Start the Vegeta workflows desktop interface:   ./side_labs.sh
# (the PyQt6 window of side_labs/gui — the tree of books and chapters, Run / Stop, the live log, the runs)
#
# Uses the project's virtual environment (.venv, or $VEGETA_VENV). PyQt6 is installed into it on first use when
# missing; the Vegeta libraries themselves come from ./install_local.sh.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

VENV="${VEGETA_VENV:-$PWD/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "no virtual environment at $VENV — run ./install_local.sh once (or export VEGETA_VENV=/path/to/venv)" >&2
  exit 1
fi
py="$VENV/bin/python"
export VEGETA_ROOT="$PWD"
export PYTHONPATH="$PWD/libs/vegeta-cli/src:$PWD/libs/vegeta-core/src:$PWD/libs/vegeta-ai/src:$PWD/libs/vegeta-fidia/src:$PWD/libs/labkit/src:$PWD/libs/ogma/src:$PWD/vendor/ozaru/porunga/src:$PWD/vendor/ozaru/mimezia/src:$PWD/vendor/ozaru/yanus/src:$PWD/labs/aero/campaigns/side_labs:$PWD/labs/aero/designs:$PWD/labs/aero/exp${PYTHONPATH:+:$PYTHONPATH}"

if ! "$py" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyQt6') else 1)"; then
  echo "Installing PyQt6 into $VENV (first use) ..."
  "$py" -m pip install --quiet --disable-pip-version-check -r labs/aero/campaigns/side_labs/requirements-gui.txt
fi
qt_hint() {
  echo "Qt could not start. On Ubuntu install the libraries it needs:" >&2
  echo "  sudo apt install libegl1 libxkbcommon-x11-0 libxcb-cursor0" >&2
}
if ! err=$("$py" -c "from PyQt6 import QtWidgets" 2>&1); then
  echo "$err" >&2
  qt_hint
  exit 1
fi
status=0
"$py" side_labs/gui "$@" || status=$?
[ "$status" -eq 134 ] && qt_hint        # abort: "could not load the Qt platform plugin"
exit "$status"
