#!/usr/bin/env bash
# Ogma's full check, to run by hand: every test, verbose, a summary and a log (user_tests_<date>.log).
#   ./user_tests.sh [--python EXE]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
[ "${1:-}" = "--python" ] && PY="$2"
LOG="$ROOT/user_tests_$(date +%Y%m%d_%H%M%S).log"
{
  echo "ogma $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) — $("$PY" --version 2>&1), numpy $("$PY" -c 'import numpy; print(numpy.__version__)' 2>&1)"
  cd "$ROOT" && "$PY" -m pytest -v -rfE -p no:cacheprovider
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "exit $status — log: $LOG"
exit "$status"
