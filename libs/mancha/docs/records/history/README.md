# Historic records

Snapshots kept as they were on the date in their name, for comparison with later measurements. They are not
updated; newer results get new files.

| Date | File | What it is | Status |
|---|---|---|---|
| 2026-10-28 | `2026-10-28_speed_chart_APPROX.png` | Compute speed: Mancha NumPy on CPU (measured, 4-core Xeon container), Mancha Triton on T4 / RTX 4090 / A100 / H100 (**estimated** from memory bandwidth: 50–80 % of peak for plain BGK, 30–60 % for the propeller physics; 152–180 bytes per node and step), and the FW-H post-processing (measured on the CPU), for the D = 24 rotor acoustics job (0.56 M nodes × ~8 000 steps, the step count itself an estimate). | **Approximate** — no GPU run existed when it was made; replace the estimates with `benchmarks/performance.py` on a GPU. Source data: `../numpy_*.json`, `../fwh_speed.json`; script: `benchmarks/speed_chart.py`. |
