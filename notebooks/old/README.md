# Historical Kestrel notebooks

The Kestrel study grew through five notebooks run in order — **38** `prop_lab`, **39** `kestrel_tests`, **40**
`kestrel`, **41** `xwing`, **42** `kestrel2`. Their place is now taken by one notebook, **`43_kestrel_study`** (the
recommended entry point: design, performance, structures, transitions, a final assessment), with the long-running
CFD and FEA moved into explicit workflows (`workflows/kestrel/`, see `workflows/README.md`).

This directory keeps **39–42 unchanged**, with everything they produced still in place (`designs/data/reports/`,
`designs/data/kestrel_design.json`, `xwing_design.json`, `kestrel_cfd_results.json`, `kestrel_fea_results.json`, the
figures), as the record of how the design got here. **38 `prop_lab` stays in `notebooks/`**: it is the live source of
the propeller calibration the new notebook and the workflows read.

They are not migrated or refactored (that is a separate item in `todo.md`), and they predate the workflow rule:
39 runs the whole CFD and FEA campaign from its cells, 40 prepares FEA cases itself. To run one anyway, start the
kernel in `notebooks/` (their paths are relative to it — `designs/`, `_runs/`, `output/`):

```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute old/40_kestrel.ipynb --output ../_runs/40_kestrel-run.ipynb
```

(or open it in JupyterLab started by `./jupyter.sh` and run it from there, with the working directory set to
`notebooks/`). The X-wing comparison (41) and the back-transition remedies (42) are summarised in the new notebook's
§ 9 and in `designs/kestrel3.md`; their full tables are in the dossier.
