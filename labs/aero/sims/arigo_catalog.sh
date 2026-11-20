#!/usr/bin/env bash
# ARIGO product catalogue / technical report (LaTeX -> PDF). No CFD.
#
#   labs/aero/sims/arigo_catalog.sh                 # newest stability run -> figures -> sims/output/catalog/arigo_catalog.pdf
#   labs/aero/sims/arigo_catalog.sh --compute -j 8  # first run the stability phase (arigo_stability.sh -j 8), then build
#   labs/aero/sims/arigo_catalog.sh --run FILE      # build from a given run file
#   labs/aero/sims/arigo_catalog.sh --help          # all options
#
# Needs: the repo's .venv (cadquery, pyvista, mujoco, matplotlib), and latexmk or pdflatex with a TeX Live that has
# babel-english/polish, titlesec, eso-pic, xurl, subcaption, tabularx (e.g. texlive-latex-extra + texlive-lang-polish).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="${VEGETA_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
COMPUTE=0; CPU=(); ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --compute) COMPUTE=1 ;;
    -j|--cpu) CPU=(-j "$2"); shift ;;
    -h|--help)
      sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
      echo; echo "  --compute         run the stability phase first (arigo_stability.sh, no CFD; restartable)"
      echo "  -j N / --cpu N    processes for --compute"; echo
      "$PY" "$HERE/../reports/arigo_catalog/build.py" --help; exit 0 ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
if [ "$COMPUTE" = 1 ]; then
  "$HERE/arigo_stability.sh" "${CPU[@]}" || { echo "stability run failed"; exit 1; }
fi
exec "$PY" "$HERE/../reports/arigo_catalog/build.py" "${ARGS[@]}"
