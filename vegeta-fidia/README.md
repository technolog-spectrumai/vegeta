# vegeta-fidia

Fidia (Italian for Phidias, the sculptor) is Vegeta's AI modelling:

- **Prompt to 3D** — `Session(prompt, out_dir, agent=...)`: plan → CadQuery code (a Dedalus design) → build in a
  sandboxed subprocess → deterministic checks → five rendered views → vision review → revise; bounded by
  iterations, minutes and tokens; user feedback, cancellation, a `STOP` file and an approval hook; every revision
  on disk and the best valid one in `best/`; GLB, glTF + `.bin`, OBJ + `.mtl`, STL and STEP, re-imported with
  trimesh and VTK.
- **The design copilot** (`DesignSession`) and bounded **parameter campaigns** (`Campaign`, needs `vegeta-core`).

The connection to the AI provider is `vegeta-ai`. Without an API key, `demo_agent()` / `fidia run --offline` runs
the whole loop with scripted answers.

```bash
pip install -e "vegeta-fidia[viz,test]"          # viz: pyvista renders and preview; core: campaigns
fidia run "a small stool with three legs" --offline --out runs/stool
pytest vegeta-fidia/tests                        # offline, CPU; live tests need ANTHROPIC_API_KEY
```
Guide: `../docs/fidia.md`. Notebook: `../notebooks/15_fidia_prompt_to_3d.ipynb`.
