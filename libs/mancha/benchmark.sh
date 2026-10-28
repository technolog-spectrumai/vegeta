#!/usr/bin/env bash
# Mancha GPU benchmark, one command:  ./benchmark.sh [--quick] [--no-setup] [--python EXE] [--out DIR]
#
#   1. environment: creates .venv-gpu (torch, triton, numpy, pytest, matplotlib) unless --no-setup / --python
#   2. GPU check: nvidia-smi, torch.cuda (stops if there is no CUDA GPU)
#   3. correctness: Triton kernels against the NumPy reference on the GPU (tests/test_triton*.py)
#   4. throughput: MLUPS, memory, bandwidth for 64³/128³/256³ (periodic, sphere, propeller physics) + block sweep
#   5. validation on the GPU: Taylor–Green, Poiseuille, cavity (benchmarks/validate.py)
#   6. rotor acoustics on the GPU: D = 16 (coarse) then D = 24, FW-H to 100 m (benchmarks/acoustics)
#   7. summary.md + charts in the output folder (default results/gpu_<date>_<gpu>)
# --quick: smaller sizes and the coarse rotor only (~10 min); full run ~1 h depending on the GPU.
set -uo pipefail
cd "$(dirname "$0")"
QUICK=0; SETUP=1; PY=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --no-setup) SETUP=0; shift ;;
    --python) PY="$2"; SETUP=0; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PY" ]; then
  if [ "$SETUP" = 1 ]; then
    [ -x .venv-gpu/bin/python ] || python3 -m venv .venv-gpu
    .venv-gpu/bin/pip install -q --upgrade pip
    .venv-gpu/bin/pip install -q -e ".[gpu,test,plot]" || { echo "pip install failed" >&2; exit 1; }
  fi
  PY="$(pwd)/.venv-gpu/bin/python"
  [ -x "$PY" ] || PY=python3
fi

GPU="$("$PY" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' 2>/dev/null)"
if [ -z "$GPU" ]; then
  echo "No CUDA GPU visible to PyTorch (nvidia-smi says:)"; nvidia-smi 2>&1 | head -5
  echo "Install the NVIDIA driver and a CUDA build of torch (pip install torch), then rerun."; exit 1
fi
TAG="$(date +%Y%m%d_%H%M)_$(echo "$GPU" | tr ' /' '__')"
OUT="${OUT:-results/gpu_$TAG}"
mkdir -p "$OUT"
LOG="$OUT/benchmark.log"
echo "Mancha benchmark on $GPU -> $OUT" | tee "$LOG"
nvidia-smi >> "$LOG" 2>&1
"$PY" -c 'import torch, triton, numpy; print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__, "numpy", numpy.__version__)' | tee -a "$LOG"
status=0
step() { echo; echo "== $1" | tee -a "$LOG"; shift; ( time "$@" ) >> "$LOG" 2>&1 || { echo "   FAILED (see $LOG)" | tee -a "$LOG"; status=1; }; tail -3 "$LOG" | sed 's/^/   /'; }

step "correctness: Triton vs NumPy on the GPU" "$PY" -m pytest -q -p no:cacheprovider tests/test_triton.py tests/test_triton_compile.py
if [ "$QUICK" = 1 ]; then SIZES="64 128"; else SIZES="64 128 256"; fi
for case in periodic sphere propeller; do
  step "throughput: $case" "$PY" benchmarks/performance.py --backend triton --case "$case" --sizes $SIZES --steps 200 --out "$OUT/perf_$case.json"
done
for b in 64 128 256; do
  step "block size $b (propeller, 128³)" "$PY" benchmarks/performance.py --backend triton --case propeller --sizes 128 --block $b --out "$OUT/perf_block$b.json"
done
QV=""; [ "$QUICK" = 1 ] && QV="--quick"
step "validation on the GPU" "$PY" benchmarks/validate.py taylor_green poiseuille cavity --backend triton $QV --out "$OUT/validation"
step "rotor D=16 (coarse) on the GPU" "$PY" benchmarks/acoustics/rotor_fwh.py --diameter 16 --backend triton --out "$OUT/acoustics/D16"
RUNS="$OUT/acoustics/D16/rotor_D16.npz"
if [ "$QUICK" = 0 ]; then
  step "rotor D=24 on the GPU" "$PY" benchmarks/acoustics/rotor_fwh.py --diameter 24 --backend triton --out "$OUT/acoustics/D24"
  RUNS="$RUNS $OUT/acoustics/D24/rotor_D24.npz"
fi
step "FW-H to 100 m" "$PY" benchmarks/acoustics/farfield_100m.py $RUNS --out "$OUT/acoustics/farfield"

"$PY" - "$OUT" "$GPU" <<'PYEOF'
import json, sys
from pathlib import Path
out, gpu = Path(sys.argv[1]), sys.argv[2]
lines = [f"# Mancha GPU benchmark — {gpu}", "", "| case | size | MLUPS | time [s] | eff. bandwidth [GB/s] | peak memory [MiB] |", "|---|---|---|---|---|---|"]
for f in sorted(out.glob("perf_*.json")):
    d = json.loads(f.read_text())
    for r in d["results"]:
        if "skipped" in r:
            lines.append(f"| {f.stem[5:]} | {r['n']}³ | skipped: {r['skipped'][:60]} | | | |")
        else:
            lines.append(f"| {f.stem[5:]} | {r['n']}³ | {r['mlups']:.0f} | {r['wall_time_s']:.3f} | {r['effective_bandwidth_gb_s']:.0f} | {r.get('peak_allocated_bytes', 0) / 2**20:.0f} |")
for f in sorted((out / "acoustics").glob("**/rotor_D*.npz")):
    import numpy as np
    m = json.loads(str(np.load(f)["meta"]))
    lines.append(f"\nRotor D={m['D']}: {m['steps']} steps in {m['wall_time_s']:.1f} s, T = {m['thrust']:.4g}, Q = {m['torque']:.4g}")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF
echo; echo "Done: $OUT (summary.md, benchmark.log, figures). Send that folder back for the comparison." | tee -a "$LOG"
exit $status
