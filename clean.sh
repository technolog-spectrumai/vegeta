#!/usr/bin/env bash
# Clean what runs leave behind. Deletes only files git does not track (each path is checked); never .venv or
# *.egg-info (the editable installs need them).
#
# Usage: ./clean.sh [-n] [--caches] [--runs] [--old-layout] [--all]
#   (default)      Python caches (__pycache__, *.pyc, .pytest_cache, .ipynb_checkpoints), old test logs
#                  (user_tests_*.log, assembly_tests_*.log, MUJOCO_LOG.TXT), executed notebook copies (*-run.ipynb),
#                  the anatomy dump out/ (python tools/dump_ast.py rebuilds it)
#   --caches       the simulation caches: *.cache/, vegeta.cache/, cache/porunga, the side labs' cache/ (results are
#                  recomputed, slowly, on the next real run)
#   --runs         run outputs: labs/*/_runs, labs/*/output, labs/*/sims/output, the side labs' runs/, the benchmarks'
#                  results/, runs/, fidia_runs/ — your results: kept unless asked
#   --old-layout   untracked leftovers in the folders of the layout before labs/ (designs/, notebooks/, scenarios/,
#                  races/, side_labs/, benchmark/, components/, assemblies/): old run outputs, NOT in git history —
#                  gone for good (move what you want to keep into labs/<lab>/ first)
#   --all          all of the above
#   -n, --dry-run  list what would go, with sizes, and delete nothing
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DRY=0; CACHES=0; RUNS=0; OLD=0
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY=1 ;; --caches) CACHES=1 ;; --runs) RUNS=1 ;; --old-layout) OLD=1 ;;
    --all) CACHES=1; RUNS=1; OLD=1 ;; -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (./clean.sh --help)" >&2; exit 2 ;;
  esac
  shift
done

CAND=()
add() { local p; for p in "$@"; do [ -e "$p" ] && CAND+=("${p#./}"); done; }
each() { while IFS= read -r -d '' p; do CAND+=("${p#./}"); done; }
PRUNE=(-path ./.venv -prune -o -path ./.git -prune -o)

echo "== Python caches, test logs, executed notebooks, out/"
each < <(find . "${PRUNE[@]}" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ipynb_checkpoints \) -print0 -prune)
each < <(find . "${PRUNE[@]}" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*-run.ipynb' \) -print0)
each < <(find . -maxdepth 1 -type f \( -name 'user_tests_*.log' -o -name 'assembly_tests_*.log' -o -name 'MUJOCO_LOG.TXT' \) -print0)
each < <(find . "${PRUNE[@]}" -type f -name 'MUJOCO_LOG.TXT' -print0)
add out

if [ "$CACHES" -eq 1 ]; then
  echo "== simulation caches"
  each < <(find . "${PRUNE[@]}" -type d \( -name '*.cache' -o -name 'vegeta.cache' \) -print0 -prune)
  add cache/porunga labs/aero/campaigns/side_labs/cache
fi

if [ "$RUNS" -eq 1 ]; then
  echo "== run outputs"
  for p in labs/*/_runs labs/*/exp/_runs labs/*/output labs/*/exp/output labs/*/sims/output \
           labs/aero/campaigns/side_labs/runs labs/aero/campaigns/side_labs/runs_*.out \
           labs/legged/campaigns/benchmark/*/results labs/aero/sims/races/runs runs fidia_runs; do
    add "$p"
  done
fi

if [ "$OLD" -eq 1 ]; then
  echo "== the layout before labs/ (untracked leftovers)"
  for d in designs notebooks scenarios races side_labs benchmark components assemblies; do
    [ -d "$d" ] || continue
    add "$d"                                                          # kept below if anything in it is tracked
  done
fi

[ ${#CAND[@]} -eq 0 ] && { echo "== nothing to clean"; exit 0; }
# one git call for the whole list: whatever git tracks stays
declare -A TRACKED=()
while IFS= read -r t; do TRACKED["$t"]=1; done < <(git ls-files -- "${CAND[@]}" 2>/dev/null)
GO=()
for p in "${CAND[@]}"; do
  case "$p" in .venv*|*.egg-info*|.git|.git/*) continue ;; esac
  keep=0
  if [ -d "$p" ]; then
    for t in "${!TRACKED[@]}"; do case "$t" in "$p"/*) keep=1; break ;; esac; done
  else
    [ -n "${TRACKED[$p]:-}" ] && keep=1
  fi
  if [ "$keep" -eq 1 ]; then echo "  keep    $p (tracked by git)"; else GO+=("$p"); fi
done
[ ${#GO[@]} -eq 0 ] && { echo "== nothing to clean"; exit 0; }
kb=$(du -sck -- "${GO[@]}" 2>/dev/null | tail -1 | cut -f1)
big=$(printf '%s\n' "${GO[@]}" | grep -v -e '__pycache__' -e '\.pyc$' | head -40)
[ -n "$big" ] && echo "$big" | sed 's/^/  /'
n_cache=$(printf '%s\n' "${GO[@]}" | grep -c -e '__pycache__' -e '\.pyc$')
[ "$n_cache" -gt 0 ] && echo "  + $n_cache __pycache__ folders / .pyc files"
if [ "$DRY" -eq 1 ]; then echo "== would remove ${#GO[@]} path(s), $((kb / 1024)) MB (dry run: nothing deleted)"
else rm -rf -- "${GO[@]}"; echo "== removed ${#GO[@]} path(s), $((kb / 1024)) MB"; fi
