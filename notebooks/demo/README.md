# Feature demos

Short notebooks that show one feature of Vegeta each. They are not design studies:

| notebook | shows |
|---|---|
| `06_core_revisions` | Vegeta Core: workspaces, revisions, evaluations, NOT RUN |
| `07_ai_design_copilot` | the AI design copilot: iterating a CadQuery design with Claude |
| `10_agentic_design` | a bounded AI campaign: FEA on every candidate, criteria, budget, approval |
| `15_fidia_prompt_to_3d` | Fidia: from a prompt to a checked 3D model (offline without a key) |

They read no workflow runs and nothing from `designs/`. Their scratch files go to `notebooks/demo/_runs/`, which is gitignored. Run one with:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/demo/15_fidia_prompt_to_3d.ipynb --output-dir /tmp
```

or `scripts/run_notebooks.sh demo/15_fidia_prompt_to_3d.ipynb`. The plain `scripts/run_notebooks.sh` runs
`notebooks/*.ipynb` only.
