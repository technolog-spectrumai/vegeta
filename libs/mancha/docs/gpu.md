# The Triton backend

`Simulation(..., backend="triton")` keeps the populations in PyTorch tensors on the GPU and advances them with
one kernel launch per step (`mancha/backends/triton_kernels.py`).

## Kernel design

* **Fused step.** `stream_collide_bgk` pulls the 19 populations of `BLOCK` consecutive nodes, applies the
  boundary rules, computes the moments, collides (BGK + Guo force) and stores — one read and one write per
  population, no intermediate arrays. The Python loop runs over time steps only, never over cells.
* **Specialisation.** The six face kinds, "has solids", "has a pressure face", "has a force" and "needs 64-bit
  indices" are compile-time constants: an all-periodic run compiles to a kernel without any boundary code.
  Each boundary configuration compiles once (Triton caches the binaries on disk).
* **Boundaries in the kernel.** For each direction the source coordinates are computed; a boundary "key"
  (`kind·4 + axis rank`) picks the winning face exactly as the NumPy reference does. Populations that are
  replaced are not loaded from the source (masked loads), and the opposite population at the node is loaded only
  where needed. The pressure outlet needs the node's own velocity: its 19 local populations are loaded only on
  pressure-face nodes (a masked load that whole warps skip elsewhere).
* **Solids.** A per-node 32-bit mask, bit `i` = "source of direction `i` is solid", bit 19 = "node is solid":
  one 4-byte load per node instead of 19 mask lookups.
* **Macroscopic fields** (`macroscopic` kernel) are computed on the GPU into a `[4, N]` buffer; only what is
  asked for is copied to the host. The health check (finite, max |u|) also runs on the GPU and synchronises
  once per `check_interval`.
* **Block size.** `block_size` (power of two ≥ 16, default 128) — one node per thread with Triton's default 4
  warps. Compiled for sm_80 the kernel uses 44 registers (periodic) to 80 (every boundary kind and solids), with
  no spills (`ptxas -v`; `tests/test_triton_compile.py` checks no spills for sm_80 and sm_90). At block 256
  (two nodes per thread) it needs 72–145 registers.
* **Precision.** FP32 only (`dtype=float64` is refused on this backend).
* **Large grids.** Above 2³¹ population indices (≈ 480³) the kernel switches to 64-bit indices.

## Memory

`2 · 19 · 4 · N` bytes for the two population buffers, `16 · N` for the macroscopic buffer and `4 · N` for the
solid mask (172 bytes per node in all): 0.04 GiB at 64³, 0.34 GiB at 128³, 2.7 GiB at 256³. The backend
estimates this before allocating
and raises `InsufficientMemoryError` (with the GPU's free and total memory) if it exceeds 90 % of the free
memory; an out-of-memory during allocation is re-raised as the same error.

## Running without a GPU

The kernels run under Triton's interpreter on CPU tensors when `TRITON_INTERPRET=1` is set before Triton is
imported (`./test.sh --interpret`). It is very slow (~1000 node updates per second) and only meant to test the
kernels: the agreement tests in `tests/test_triton.py` run this way in CI. `tests/test_triton_compile.py`
compiles the kernels for sm_80 and sm_90 without a GPU.

## Benchmarking and profiling

```bash
python benchmarks/performance.py --sizes 64 128 256 --steps 500             # periodic box
python benchmarks/performance.py --case sphere --sizes 128 256              # boundaries + solids
python benchmarks/performance.py --block 64   # block-size sweep: repeat with 64, 128, 256
```

It reports MLUPS (million node updates per second), wall time, solver memory, peak allocated memory and an
effective bandwidth `MLUPS · 152 B` (`156 B` with solids), and writes a JSON with the GPU name. Compare the
bandwidth with the card's peak: an LBM step is memory-bound, so that ratio is the figure of merit (published
hand-tuned D3Q19 codes commonly report well over half of peak; Mancha has no measured figure yet).

Profiling with Nsight (on a machine with a GPU):

```bash
nsys profile -o mancha --trace=cuda python benchmarks/performance.py --sizes 128 --steps 200
ncu --kernel-name stream_collide_bgk --launch-skip 30 --launch-count 1 --set full \
    -o mancha_kernel python benchmarks/performance.py --sizes 128 --steps 50
```

Look at DRAM throughput (target: near peak), achieved occupancy (register-limited at 145 registers with
`BLOCK=256`, hence the default 128) and L2 hit rate of the shifted loads.

**Measured results: none yet.** The backend was developed in a container without a GPU; no GPU timings exist
and none are quoted. Record them in [validation.md](validation.md#performance) with the GPU named when measured.
