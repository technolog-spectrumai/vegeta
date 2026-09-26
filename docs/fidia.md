# Fidia — AI modelling for Vegeta

**Fidia** (`vegeta-fidia`, `vegeta.fidia`; Italian for Phidias, the sculptor of the Parthenon) is where Vegeta
uses AI to model. It has three parts:

1. **Prompt to 3D** — from a text prompt to a checked, exported 3D model through a bounded loop (this page).
2. **The design copilot** — iterate on one Dedalus design file, one proposal at a time ([below](#the-design-copilot)).
3. **Campaigns** — a bounded parameter search on `vegeta.core` revisions ([below](#campaigns)).

The connection to the AI provider is `vegeta.ai` ([docs/ai.md](ai.md)); Fidia holds the prompts, schemas, loops
and checks. It imports only `vegeta.dedalus`, `vegeta.ai` and (lazily, for campaigns) `vegeta.core`.

## Quick start
```bash
pip install -e "vegeta-fidia[viz]"                               # done by install_local.sh
fidia run "a small stool with three legs" --offline --out runs/stool   # scripted demo agent: no key, no cost
export ANTHROPIC_API_KEY=sk-ant-...
fidia run "a desk lamp with a round base and a conical shade" --out runs/lamp --max-iterations 4
fidia resume runs/lamp --feedback "make the shade red and 20 % larger"
fidia show runs/lamp
```
Notebook: `notebooks/15_fidia_prompt_to_3d.ipynb` (runs offline without a key).

```python
from vegeta.fidia import Limits, Session, agent_from_config, demo_agent

agent = agent_from_config("claude-opus-5", effort="high", review_effort="medium")   # or demo_agent()
s = Session("a desk lamp with a round base and a conical shade", "runs/lamp", agent=agent,
            limits=Limits(max_iterations=5, max_minutes=15, max_tokens=300_000))
s.run()                                  # one line per revision; stops when done or at a limit
s.table(); s.revisions[0].sheet(); s.best.preview()
s.best.files()["glb"]                    # runs/lamp/rev-00N/export/model.glb (a copy is in runs/lamp/best/)
s.feedback("make the shade red"); s.run()
```

## First principles
1. **The model writes programs, not meshes.** An LLM cannot emit a reliable mesh but writes short programs well.
   The program (a Dedalus `Design` in CadQuery) is the source of truth; the mesh is derived. CadQuery runs on a
   CPU and says clearly when a solid is broken.
2. **Generated code is untrusted.** It runs out of process, with a timeout, a memory cap and a static import
   allow-list — never in the notebook kernel.
3. **Two kinds of truth.** *Objective* (is it a valid, closed, sane solid of the planned size?) is computed by
   Vegeta; *subjective* (does it look like the request?) needs eyes: five rendered views judged by a
   vision-capable reviewer. Only the objective checks decide validity.
4. **A loop that converges or stops.** Each iteration carries forward what failed (errors, checks, review, your
   feedback) and stops on success, no progress, limits or your cancel. The best valid result is never lost.
5. **The engineer stays in charge.** Feedback between iterations, stop at any time, nothing silent: every prompt,
   plan, code, check, render, review and decision is on disk.
6. **Outputs must survive other tools.** glTF/GLB and OBJ with their sidecars, proven by re-importing with two
   independent readers.

## The loop
```
prompt ─► plan ─► generate code ─► static check ─► sandboxed build ─► mesh checks ─► renders + export/re-import
             ▲          ▲                                                                  │
             │          └──── errors, failed checks, review issues, your feedback ◄── review (vision, advice)
             └── re-plan (only when feedback asks for it)
```
- **Calls per iteration:** a plan at the start (and again with `feedback(..., replan=True)`), then one *generate*
  and one *review* per revision. The review is **skipped** when the build fails or a check fails — the errors go
  straight back to the modeller. The token budget is checked **before every call** (used + the call's
  `max_tokens` must fit).
- **Done** = the revision is valid (no failed check), its export round-trip is clean, the reviewer says
  `accept` with a score ≥ `accept_score` (default 8/10), every acceptance check of the plan passes, and no
  feedback is waiting.
- **Stops:** done; `max_iterations` or `max_minutes` (per `run()` call); `max_tokens` (for the whole run
  directory); `patience` (valid revisions without a new best); `session.cancel()` (from another thread; abandons a
  running model call or kills a running build); a file named `STOP` in the run directory; *Interrupt kernel* /
  Ctrl-C (the revision is sealed as cancelled and the state saved); a declined approval; a failed model call; the
  reviewer giving up.
- **Best:** only valid revisions are eligible. A revision that has seen more of your feedback wins over an older
  one; then the higher review score, fewer warnings, the earlier revision. `pin_best("rev-003")` overrides it.
  `best/` is a copy of the best revision, replaced whole when the best changes.
- **Your controls:** `feedback(text, replan=False)`; `approve="ask"` (or a function) to confirm each generated
  file before it runs; `on_iteration=fn(revision)` returning feedback text or `False` to stop; `step()` to make one
  revision at a time; `Session.open(dir, agent=...)` to resume (numbering and tokens continue).

## What the model writes
One Python file: a Dedalus `Design` whose `build(p)` returns a CadQuery `Workplane`/`Shape` (one part) or a
`cq.Assembly` of **named, coloured parts** (`assy.add(wp, name="seat", color=cq.Color(r, g, b))`), in millimetres,
Z up, resting on z = 0, with the main dimensions as `Parameter`s. The full contract and an example are in
`vegeta.fidia.contract` (`CONTRACT`, `EXAMPLE_SOURCE`). `best/design.py` is an ordinary Dedalus design: use it with
`dedalus`, `vegeta-core` or `load_design` like any other.

## The sandbox — what it does and does not do
| Wall | Default | Notes |
|---|---|---|
| Static screen (`contract.validate_source`) | imports: `cadquery`, `math`, `numpy`, `typing`, `vegeta.dedalus` (Design, Parameter) | no `open/eval/exec/compile/getattr/__import__/globals`…, no dunder attributes, no `exporters/importers/save/load/tofile/memmap/system/remove`…, one `Design` class. A speed bump, not a proof. |
| Separate process | `python -I -m vegeta.fidia.runner` in a fresh temp directory, stdin closed | never `exec` in your Python process |
| Clean environment | `PATH`, locale, private `HOME`/`TMPDIR` | no variable containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`: the API key never reaches generated code |
| Time | `Sandbox(timeout_s=120)` + `RLIMIT_CPU` | the whole process group is killed on timeout or cancel |
| Memory | `memory_mb=4096` (`RLIMIT_AS`) | OpenCASCADE needs ~1.2 GB of address space; reported as `resource_limit` |
| File size | `file_mb=256` (`RLIMIT_FSIZE`) | |
| Results | `result.json` + `parts.npz` (loaded with `allow_pickle=False`) + `model.step` | nothing else is read back |
| Network | **not blocked** by default | add `Sandbox(wrapper=("unshare", "-rn"))` (needs unprivileged user namespaces) or run inside a container / `bwrap` / `firejail` |

The static screen plus a subprocess with resource limits is **not a security boundary** against a determined
attacker: Python can reach the file system through the allowed libraries. It stops mistakes and casual misuse,
keeps your keys and your kernel out of reach, and bounds time and memory. Use a wrapper or a container for
untrusted prompts. Build statuses: `ok`, `rejected` (static screen), `build_error` (with the traceback),
`timeout`, `cancelled`, `resource_limit`.

## Checks (decide validity)
| Fail — the revision is invalid | Warn — reported to the modeller and reviewer |
|---|---|
| B-rep not valid (OpenCASCADE), no solid, no volume | degenerate (zero-area) triangles |
| mesh not watertight (after merging CadQuery's repeated seam vertices) | several bodies in one part |
| inconsistent or inside-out winding | planned parts missing |
| over the triangle budget (`max_triangles=200_000`) | overall size > 35 % off the plan (a 90° turn about Z is allowed) |
| no parts | not resting on z = 0 |
| an export that does not re-import correctly | floating parts: not connected to the group on the floor (exact B-rep distances) |
| | parts without a colour |

## Export and re-import
| File | Axes, units | Sidecars | Notes |
|---|---|---|---|
| `model.glb` | Y up, metres | — | one node + base-colour PBR material per part |
| `model.gltf` | Y up, metres | `model.bin` | `buffers[*].uri` checked: exists, size = `byteLength` |
| `model.obj` | Z up, mm | `model.mtl` | one `o` object per part, one `newmtl` (`Kd`) per colour; `mtllib` and every `usemtl` checked |
| `model.stl` | Z up, mm | — | all parts in one mesh (merged seams) |
| `model.step` | Z up, mm | — | the exact B-rep from the runner |

glTF's transform from the CAD frame is in `manifest.json` (`gltf_from_cad`: rotate −90° about X, scale 1/1000).
`reimport.json` reads every file back with **trimesh** and **pyvista/VTK** and compares part count and names,
triangles, bounds (back in CAD millimetres) and colours with what was written.

## Renders
Five views — front, right, top (orthographic), iso and rear iso — at 512 px with part colours and feature edges,
plus `sheet.png` (3 × 2 tiles with a legend: overall size and part colours; 1536 × 1024 px ≈ 2k image tokens).
The reviewer sees only the current sheet; history reaches the model as text. Off-screen pyvista is probed once in a
subprocess; without a working GL stack (EGL or OSMesa) Fidia falls back to matplotlib. `FIDIA_RENDER=matplotlib`
forces the fallback.

## Files
```
<run>/run.json          state (rewritten atomically): limits, usage, revisions, best, queued feedback
      events.jsonl      append-only log: plan, calls (with usage), builds, revisions, best, feedback, stop
      prompt.txt  plan.json  plans/plan-NNN.json
      rev-NNN/design.py generation.json execution.json runner.log result.json parts.npz model.step
              checks.json renders/{front,right,top,iso,rear_iso,sheet}.png review.json
              export/{model.glb, model.gltf, model.bin, model.obj, model.mtl, model.stl, model.step,
                      manifest.json, reimport.json}
              revision.json     the sealed record (written once, never rewritten)
      best/             a copy of the best valid revision (+ BEST with its name)
```

## Costs
With Claude Opus 5 ($5 / $25 per million input / output tokens) a revision is two calls: *generate* (≈ 5–8k tokens
in, 2–8k out with adaptive thinking) and *review* (≈ 3–4k in including the image, 1–2k out); the plan adds one
call at the start. That is roughly **$0.2–0.4 per revision**, about $1 for a four-revision run. `effort` and
`review_effort` (`low`/`medium`), a cheaper model, `max_tokens` per call and the run's `max_tokens` budget control
it; `s.report()` shows the tokens and the estimated cost.

## Headless Blender — assessed, not used in the MVP
| | CadQuery + trimesh + pyvista (chosen) | Blender (`bpy` / `blender --background`) |
|---|---|---|
| Already in Vegeta | yes (Dedalus) | no |
| Install | small | `bpy` wheel ≈ 300 MB, pinned to one Python minor (5.0 → Python 3.11); apt `blender` 4.x |
| CPU only | yes | modelling yes; EEVEE needs a GPU, Cycles on a CPU takes seconds to minutes per view |
| Validity signal | B-rep validity, closed solids, exact volume | meshes only; validity must be computed |
| LLM code | reliable for mechanical/product shapes | good for organic shapes, but a large, version-sensitive API with global state |
| glTF | via trimesh (materials) | the best exporter: materials, textures, animation |
| Licence | Apache / MIT | GPL (bpy linked in-process) |

**Choice:** CadQuery code in the Dedalus contract, run in the sandbox; trimesh for checks and export; pyvista for
renders and the preview. Blender is worth adding when organic shapes, textures or photoreal renders are needed —
as another backend with the same contract (source in → `result.json` + `parts.npz` out), run the same way:
`blender --background --factory-startup --python runner.py -- design.py outdir` inside the same walls.
Setup for that later: `pip install bpy==5.0.1` in a Python 3.11 environment, or `apt install blender`; renders
with Cycles on CPU (`bpy.context.scene.cycles.device = "CPU"`) or EEVEE with a GPU.

## Why this is allowed (the exception to `rejected.md`)
`rejected.md` rejects "executing LLM-written CadQuery unattended" and "letting the LLM judge results". Fidia's
loop is the narrow exception, on these conditions: the code never runs in your Python process (the sandbox above,
no keys, resource walls); the review is advice — only Vegeta's deterministic checks make a revision valid, and an
invalid revision can never become the output; the loop is bounded (iterations, time, tokens, patience) and every
step is recorded and resumable; an approval hook exists (`approve="ask"` / `--confirm`); the result is a *model*
for you to use, not an engineering verdict — loads, materials and analyses stay yours.

## The design copilot
Iterate on a Dedalus design file with a proposer; Vegeta builds and measures every proposal **in the sandbox**
and shows the diff; the file changes only when you accept.
```python
from vegeta.ai import ClaudeProvider, ProviderConfig
from vegeta.fidia import DesignSession, ProviderProposer

claude = ProviderProposer(ClaudeProvider(ProviderConfig(model="claude-opus-5", effort="high")),
                          extra_system="All parts are FDM-printed in PETG; wall thickness >= 1.6 mm.")
s = DesignSession("designs/bracket.py:Bracket", claude, parameters={"thickness": 6.0},
                  notes="clamped at x=-40, 200 N down at x=+40")
p1 = s.ask("add two ribs under the plate to increase bending stiffness, at most 15 % more mass")
p1                       # summary, rationale, expected effects, risks, validation (built + measured), diff
s.geometry(p1)           # the proposal built in the sandbox and read back from STEP — nothing written yet
s.accept(p1, note="ribs look right")      # writes the file (backup bracket.py.1.bak)
p2 = s.ask("the ribs are too tall, halve their height")
s.reject(p2, "not what I meant")
```
Validation: the proposed file passes the static screen and loads as a `Design`; parameters resolve within their
ranges; the geometry builds and is a valid solid; volume, area, dimensions and face/solid counts are compared
before/after; added and removed parameters are listed. A failed validation cannot be accepted. The design file
must follow the contract above (one `Design` class, the allowed imports). `Proposal.kind` is `source`,
`parameters` or `answer` (no change: an explanation or a question back to you). Files: `<design>.ai.jsonl` (every
proposal and decision with token usage) and `<design>.<n>.bak`.
```bash
fidia propose designs/bracket.py:Bracket "add two stiffening ribs" -p thickness=6 --save p1.json
fidia accept  designs/bracket.py:Bracket p1.json          # re-validates, then writes (backup kept)
```
Notebook: `07_ai_design_copilot`.

## Campaigns
`Campaign` lets the model drive a parameter search on `vegeta.core` revisions (install `vegeta-fidia[core]`). It
proposes values; Vegeta checks them, branches a revision, runs your analyses, checks your criteria and shows the
model the results table before the next proposal.
```python
from vegeta.ai import ClaudeProvider, ProviderConfig
from vegeta.fidia import Analysis, Budget, Campaign, Criterion, Objective, ProviderProposer

campaign = Campaign(
    ws, start_revision,
    analyses=[Analysis("fea", "cantilever", cantilever)],           # your factory: loads, material, mesh
    criteria=[Criterion("fea.cantilever.safety_factor_yield", ">=", 2.5),
              Criterion("fea.cantilever.max_displacement", "<=", 0.8)],
    objective=Objective("geometry.volume", "min"),
    proposer=ProviderProposer(ClaudeProvider(ProviderConfig(effort="medium"))),
    free=["thickness", "width"], bounds={"thickness": (3, 12)},
    approval="ask",                      # or "auto" (explicit opt-in) or a function(proposal) -> bool
    budget=Budget(max_iterations=8, max_tokens=200_000, max_minutes=20, patience=3),
    name="light_bracket")
campaign.run(); campaign.table(); campaign.plot(); campaign.best
```
The model may change only the `free` parameter values, within the design's ranges and `bounds`, never repeating a
tried set; source changes, unknown or fixed parameters, out-of-range values and repeats are refused and recorded. A
declined approval stops the campaign. Limits: `max_iterations`, `max_tokens`, `max_minutes`, `patience`, and a
`STOP` file in `campaigns/<name>/`. Record: `campaigns/<name>/campaign.json` and `events.jsonl`; every candidate is
an ordinary revision with its evaluations; `run()` again with the same `name` continues. A campaign never labels
revisions — picking the design is your decision. Notebook: `10_agentic_design`.
