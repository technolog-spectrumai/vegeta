#!/usr/bin/env bash
# Yax's full check, to run by hand: every test, verbose, a summary and a log (user_tests_<date>.log).
#   ./user_tests.sh [--python EXE]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
[ "${1:-}" = "--python" ] && PY="$2"
LOG="$ROOT/user_tests_$(date +%Y%m%d_%H%M%S).log"
{
  echo "yax $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) — $("$PY" --version 2>&1)"
  "$PY" - <<'PYEOF'
for m in ("pxr", "mujoco", "cadquery", "jsbsim"):
    try:
        mod = __import__(m)
        print(f"  {m}: {getattr(mod, '__version__', 'ok')}")
    except Exception as e:                                         # optional engines: their tests skip
        print(f"  {m}: not installed ({type(e).__name__})")
PYEOF
  cd "$ROOT" && PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pytest -v -rfEs -p no:cacheprovider
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "exit $status — log: $LOG"
exit "$status"
