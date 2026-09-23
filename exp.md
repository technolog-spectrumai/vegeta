# Exploration: Vegeta as an AI-assisted, closed-loop engineering platform

Status: exploration only, nothing implemented. Branch `exp`, based on `main` (`cb3ddaa`).
Companion files: `plan.md` (recommended course of action), `rejected.md` (ideas not to pursue and why).

Calibration used throughout: 1–3 engineers now, at most ~10 in two years; Anthropic and OpenAI API
access; everything runs on engineers' Ubuntu workstations today, with a possible move to the cloud
later. Two robot/drone projects at a time is the working assumption for "scale".

---

## 1. Current architecture assessment

### What exists on `main`

| Layer | Package | State |
|---|---|---|
| CAD | `vegeta.dedalus` (CadQuery) | parametric designs, STEP/STL, measurements, source-identity hash |
| FEA | `vegeta.talos` (Gmsh + CalculiX) | linear static, explicit regions/loads/materials, validated vs beam theory |
| CFD | `vegeta.aeromant` (OpenFOAM) | template cases, explicit values, Cd/Cl/Cm, validated on a sphere |
| Print | `vegeta.mellonia` (PrusaSlicer) | explicit settings + orientation, G-code metrics |
| Workbench | `vegeta.core` | workspaces, immutable revisions, branching, labels, recorded evaluations, NOT RUN |
| Interfaces | `vegeta <tool>`, `vegeta ws/rev`, Jupyter notebooks, `install_local.sh`, `test.sh` | |

Properties that matter for this exploration:

- **Everything is a file.** A workspace is a directory; `revision.json` and `evaluation.json` are
  write-once; annotations are append-only JSON lines. There is no database and no server.
- **Nothing runs by itself.** Generating CAD never runs FEA; running FEA never runs CFD. Missing
  analyses show as `NOT RUN`.
- **Every evaluation is already a full provenance record**: parameters, source SHA-256 and snapshot,
  tool configuration, the factory's source text, tool versions, every command line, timestamps,
  environment. This is exactly the "trajectory log" an AI loop needs, and it exists already.
- **Analysis factories are deterministic executors.** `run_fea("static", static)` takes a Python
  function `static(rev) -> StructuralModel`. A proposal that changes only parameters is re-evaluated by
  the same function on the new geometry. That is the seam where an LLM can propose without touching
  the physics.
- **Results carry an honest status** (`success`/`failed`/`cancelled`) plus messages, and the tools
  refuse to guess (no default materials, no auto-meshing, no auto-generated CFD cases).

### What already supports AI or closed-loop work

1. Immutable revisions with parents are a search tree. An optimiser or an LLM can only add nodes.
2. `ws.status()` is a machine-readable state of "what has been tried and what came out".
3. Factories separate *what to try* (parameters) from *how to judge it* (fixed loads, materials, mesh).
4. Labels (`preferred`/`rejected`/`reference`) are human verdicts that a loop can read but not write.

### Gaps that block a loop, independent of any AI

| Gap | Why it blocks a loop |
|---|---|
| No comparison across revisions (Stage 11) | a loop must compare candidate results; today that is `ws.status()` and eyeballs |
| No explicit parameter sweeps (Stage 11) | the most useful "loop" for 1–3 engineers is a deterministic sweep, not an agent |
| No machine-readable acceptance criteria | "SF ≥ 2 and mass ≤ 120 g" lives in the engineer's head; nothing can evaluate a verdict |
| No cost accounting per evaluation | CFD runs are minutes to hours; a loop needs to know what it spends |
| `vegeta.core` has no read-only summary export | anything external (LLM, MCP, UI) needs a compact view of a workspace |
| No field export for viewers (VTU) | results are numbers + PNGs; no 3D results inspection outside CalculiX/ParaView manual work |
| Workstation-only deployment | fine now; a cloud move needs the tool versions pinned and reproducible |

These gaps are cheaper and more valuable to close than any component below, and several of the
components below quietly depend on them.

---

## 2. Component-by-component analysis

Each component answers the same 13 questions. A summary table follows in §2.10.

### 2.1 Closed-loop AI iteration (LLM → CAD/FEA/CFD/print → feedback → next iteration)

1. **Problem it solves.** An engineer iterating a bracket runs the same edit-generate-solve-compare
   cycle many times; the tedious part is choosing the next variant and keeping track. A loop can
   propose the next variant and keep the bookkeeping straight.
2. **Concrete benefit.** Fewer manual iterations to reach a feasible design (e.g. "SF ≥ 2 at minimum
   mass"); overnight campaigns; every attempt recorded as a revision.
3. **Disadvantages and new failure modes.** The LLM does not know physics; it pattern-matches. Typical
   failures: proposing parameter changes outside validity (mesh no longer resolves a fillet; CFD domain
   assumptions break), optimising the wrong metric (the nodal von Mises peak at a support singularity),
   "converging" to a mesh artefact, and generating plausible CadQuery code that changes the design
   intent. Running FEA/CFD costs minutes to hours, so a bad loop burns compute silently. Any loop that
   *runs* analyses contradicts "nothing runs automatically" unless it is explicitly scoped and budgeted.
4. **Implementation complexity.** Medium if built as a *proposal-driven campaign* on `vegeta.core`:
   acceptance criteria object, a `Proposal` type, a `Campaign` loop with limits, an approval policy.
   High if built as an autonomous agent with tool access (see `rejected.md`).
5. **Infra/operational cost.** LLM tokens (a proposal step is a few thousand tokens; with Claude Opus 5
   at $5/$25 per MTok a 20-iteration campaign is under a dollar), plus the compute the analyses already
   cost. No new services.
6. **Maintenance burden.** Prompts and proposal schemas drift with models; the physics side does not.
   Keeping the loop *thin* keeps this small.
7. **Effect on existing architecture.** None if proposals only create revisions and request
   evaluations through the existing API. The tools are untouched. `vegeta.core` gains: criteria,
   proposals, campaign records (all files, all append-only).
8. **Dependencies and coupling.** New optional distribution (`vegeta-ai`) depending on `vegeta-core`
   and one LLM SDK per adapter. Core does not depend on it.
9. **Short-term ROI (1–6 months).** Moderate. A deterministic sweep (Stage 11) gets most of the value
   for parametric designs at zero LLM cost. The LLM adds value when the search space is not a grid:
   "which of these six parameters matters", "the fillet is the problem, not the thickness".
10. **Long-term ROI (1–3 years).** High if the company runs many similar parts through similar
    criteria: a library of criteria + factories makes each new part a campaign, not a project.
11. **Scale that makes it worthwhile.** Already at 1–3 engineers, provided analyses take minutes not
    hours and criteria are explicit. Below that (one-off parts) a sweep is enough.
12. **Simpler alternative.** Stage 11 sweeps + a `criteria.evaluate(revision)` verdict + `ws.status()`
    covers 70% of the value; the LLM then only has to explain results and suggest the next sweep.
13. **Recommendation: prototype** — as a proposal-driven campaign, human-approved by default, after
    Stage 11 and acceptance criteria exist. Not as an autonomous agent.

### 2.2 Interchangeable LLM adapters (OpenAI and Claude first)

1. **Problem.** Vendor lock-in and price/quality differences between models; the user already has both.
2. **Benefit.** Swap providers per task (cheap model for summaries, capable model for proposals);
   compare proposals from two providers on the same state (a cheap form of "independent review").
3. **Disadvantages.** Adapters invite a framework. Provider features differ (structured outputs,
   thinking/effort, caching, tool-use semantics); a lowest-common-denominator adapter loses the
   features that make proposals reliable (strict JSON schemas). Model deprecations break prompts.
4. **Complexity.** Low if the contract is narrow: `propose(state: dict, schema) -> Proposal` and
   `explain(evaluation) -> str`. Two adapters of ~100 lines each using the official SDKs
   (`anthropic`, `openai`). High if it tries to abstract tool use, streaming and agents.
5. **Infra cost.** API keys on workstations; tokens. Nothing to run.
6. **Maintenance.** Low for two adapters with structured-output contracts; model IDs and pricing are
   configuration, not code.
7. **Architecture effect.** None on tools/core. Lives in `vegeta-ai`.
8. **Coupling.** `vegeta-ai` → `anthropic`, `openai` (optional extras). Keys via environment variables
   (an external-service connection, the one place environment configuration is acceptable).
9. **Short-term ROI.** Enabler for 2.1; on its own it delivers "explain this evaluation in plain
   words" and "propose three variants", which are useful the day they exist.
10. **Long-term ROI.** Keeps the option to move to local/open models later (same contract).
11. **Scale.** Any.
12. **Simpler alternative.** One provider only. Cost of the second adapter is small; keep both but make
    the *contract* the product, not the adapters.
13. **Recommendation: implement now**, minimal: a `Proposal` schema, a `Proposer` protocol, two
    adapters, no third-party agent framework. This is Stage 12 of the original spec.

### 2.3 Controlled agent loops (budgets, iteration limits, validation, approvals, stopping conditions)

1. **Problem.** An LLM-driven loop without hard controls runs until money or patience ends, and its
   frameworks' "stop" is not always enforced (arXiv 2607.14166 measured this gap in 2026).
2. **Benefit.** Predictable cost and behaviour: a campaign is a bounded object that always ends in a
   recorded state (`done`, `budget_exhausted`, `no_progress`, `stopped_by_user`, `failed`).
3. **Disadvantages.** Limits must be set by the engineer (more explicit configuration). Too-tight
   no-progress detection stops legitimate exploration; too-loose burns budget. Approval gates make
   overnight runs impossible unless a policy pre-approves a class of proposals.
4. **Complexity.** Low–medium: a loop with counters is easy; the subtle parts are *validation* (what
   counts as progress: the criteria verdict, not the LLM's opinion) and *approval policy* (which
   proposals may run unattended).
5. **Infra cost.** None.
6. **Maintenance.** Low; it is deterministic Python.
7. **Architecture effect.** Adds a `Campaign` record (append-only log of proposals, decisions,
   evaluations, spend) to a workspace. Fits the file model.
8. **Coupling.** `vegeta-ai` only.
9. **Short-term ROI.** It is the difference between a prototype the team trusts and one it disables.
10. **Long-term ROI.** The same controls govern any future automation (sweeps, multi-agent, MCP callers).
11. **Scale.** Any; mandatory the moment anything runs an analysis on the engineer's behalf.
12. **Simpler alternative.** None simpler that is safe. The simplest *form* is: max iterations,
    max evaluations per kind, wall-clock budget, token/cost budget, "stop when criteria met",
    "stop after N proposals without improvement", approval required for anything that changes source
    code, materials, loads, boundary conditions, templates or print settings.
13. **Recommendation: implement now**, as part of the same prototype as 2.1 (they are one design).

Best-practice summary used (sources in §6): three hard stops — max iterations, no-progress detection,
token/cost budget; approval placed where errors are expensive or irreversible; graduated autonomy that
can be widened as a policy proves itself; log every intervention; "stop" must be enforced by the
harness, not requested from the model.

### 2.4 Multi-agent engineering runs

Four patterns compared against Vegeta's needs:

| Pattern | What it means here | Value at 1–10 engineers | Cost/risk |
|---|---|---|---|
| **A. One agent + deterministic workflow** | one LLM proposes; Vegeta's fixed workflow executes, validates, records | highest: all value of 2.1 with the fewest moving parts | low; the workflow is code you already have |
| **B. One orchestrator + specialist tools** | one LLM calls typed tools ("run FEA", "compare", "branch") | some: flexible ordering of steps | medium; the LLM now decides *what runs* → budgets and approvals must wrap every tool; tool-use loops cost more tokens and are harder to reproduce |
| **C. Multiple specialist agents** (CAD agent, FEA agent, CFD agent, manufacturing agent talking to each other) | agents negotiate | low now: the specialists' knowledge lives in the deterministic tools and templates, not in prompts; inter-agent chatter is expensive, non-reproducible and hard to audit | high; new failure modes (agents agreeing on nonsense), hardest to control |
| **D. Parallel independent reviews** | N independent LLM calls review the same evaluation/proposal; disagreement flags risk | moderate: cheap cross-check of a proposal before it spends CFD hours; works with two providers (2.2) | low; embarrassingly parallel, no shared state |

1. **Problem.** Complex trade-offs across disciplines; the hope that specialists "collaborate".
2. **Benefit.** Only D has a clear near-term benefit: a second opinion before expensive runs, and
   a signal (disagreement) that is useful precisely because it is not a decision.
3. **Disadvantages.** C multiplies cost, latency and non-determinism; consensus between LLMs is not
   evidence. B moves control from code to model. Both fight the philosophy that Vegeta controls.
4. **Complexity.** A low; D low; B medium; C high.
5. **Infra cost.** Tokens ×N for C and D; D can use cheaper models.
6. **Maintenance.** C is the hardest to maintain (prompts × agents × models).
7. **Architecture effect.** A and D: none. B: needs a tool layer over core (overlaps with MCP, §2.5).
   C: needs orchestration infrastructure the project does not have.
8. **Coupling.** C typically drags in an agent framework.
9. **Short-term ROI.** A > D > B > C.
10. **Long-term ROI.** B becomes relevant if Vegeta is exposed as tools (MCP) anyway; C remains
    unconvincing without evidence that specialist prompts beat specialist code.
11. **Scale.** C only if the company has many disciplines with real tacit knowledge not encodable as
    templates — not the case for four wrapped solvers.
12. **Simpler alternative.** A now; D as an optional review step; B via MCP later if needed.
13. **Recommendation:** A **implement now** (it is 2.1); D **prototype later**; B **postpone**;
    C **reject** for now.

### 2.5 Vegeta as an MCP server

1. **Problem.** Engineers use Claude Code / IDE agents / chat assistants; those could call Vegeta
   directly ("what is the safety factor of r7", "branch r7 with thickness 8 and run static").
2. **Benefit.** Vegeta's capabilities become available inside tools engineers already use; an
   external agent loop (Claude Code, Managed Agents) can drive campaigns without Vegeta hosting an
   agent itself. That is attractive: **Vegeta stays the deterministic executor; the agent lives elsewhere.**
3. **Disadvantages and failure modes.** An MCP server runs with the developer's own privileges and
   exposes real actions (CSI/NSA and CSA guidance 2026); prompt injection through tool descriptions and
   results is real; remote transports need OAuth 2.1 (spec 2025-11-25+), which is server work you do
   not have. Write-capable tools ("run CFD") from a chat window bypass Vegeta's approval discipline
   unless the server re-implements budgets and approvals. Tool schemas duplicate the Python API and drift.
4. **Complexity.** Low for a read-only local server (stdio transport; tools: list workspaces, status,
   revision, evaluation, artifacts by id). Medium with write tools + approval/budget enforcement.
5. **Infra cost.** None for local stdio; a remote server means hosting, auth, TLS.
6. **Maintenance.** Medium: MCP spec revisions are frequent (2025-11-25, 2026-07-28); SDK churn.
7. **Architecture effect.** None on tools; core needs the read-only summary export (a gap anyway).
8. **Coupling.** `vegeta-mcp` → `vegeta-core` + `mcp` SDK. One-directional.
9. **Short-term ROI.** Low until someone actually uses Claude Code/IDE agents on engineering work;
   then a read-only server is a day of work with immediate convenience.
10. **Long-term ROI.** Possibly high: it is the cleanest way to get "agentic" behaviour without owning
    an agent, and it matches the cloud option (a remote MCP server is a natural cloud front door).
11. **Scale.** Worth it as soon as two or more people use LLM agents for engineering tasks.
12. **Simpler alternative.** The `vegeta` CLI with `--json` is already callable from Claude Code's bash
    tool; an agent can use it today. An MCP server adds discoverability and typed schemas, not capability.
13. **Recommendation: postpone; prototype (read-only, stdio)** when an external agent workflow exists.
    Write tools only with the same budget/approval machinery as 2.3.

### 2.6 Persistent Neo4j engineering knowledge base

Schema sketch if it were built (PROV-style: entities, activities, agents):

```
(:Project)-[:HAS]->(:Requirement {text, metric, limit})
(:Design {name, source_sha})-[:REVISION]->(:Revision {id, params, parent})
(:Revision)-[:EVALUATED_BY]->(:Evaluation {kind, name, status, metrics})-[:PRODUCED]->(:Artifact {hash, path, kind})
(:Evaluation)-[:USED]->(:Configuration {material, loads, template, settings})
(:Decision {label, note, by, at})-[:ABOUT]->(:Revision)
(:Failure {mode, symptom})-[:OBSERVED_IN]->(:Evaluation)
(:Lesson {text})-[:DERIVED_FROM]->(:Failure|:Decision)   -[:APPLIES_TO]->(:Component|:Material|:Process)
(:Component {part_number})-[:USED_IN]->(:Project)
```
Large files (STEP, STL, meshes, FRD, cases, G-code) stay on disk or object storage, referenced by
hash and path — the graph holds identity and relations only.

1. **Problem.** Knowledge lives in people and scattered workspaces: which materials/print settings
   worked, why a bracket failed, what a previous drone's arm looked like, which templates are trusted.
2. **Benefit.** Cross-project queries ("all evaluations of PETG parts with SF < 1.5", "designs that used
   this motor mount"), reuse of lessons, graph-RAG for the LLM proposer (context from past failures).
3. **Disadvantages and failure modes.** A second source of truth next to the workspace files
   (sync bugs, stale graph); a database service to run, back up and secure; Cypher and modelling
   skills needed; ontology churn as the company learns what it wants; at 1–3 engineers the graph
   stays mostly empty and nobody maintains it. Provenance already exists in files — the graph would
   duplicate it. LLM-written "lessons" can pollute it with confident nonsense unless human-curated.
4. **Complexity.** Medium–high: importer from workspaces, schema, query API, curation UI or convention.
5. **Infra cost.** Neo4j Community in Docker on a workstation is free but is a service to keep alive;
   shared use needs a server (the cloud move) with auth and backups; AuraDB is a paid SaaS.
6. **Maintenance.** Ongoing: schema migrations, importer upkeep, curation of lessons.
7. **Architecture effect.** Adds the first stateful service and the first non-file store; breaks "no
   database needed" unless strictly optional and derived (rebuildable from files).
8. **Coupling.** `vegeta-kb` → `neo4j` driver + `vegeta-core`. Must stay one-directional and derived.
9. **Short-term ROI.** Negative at current scale: the effort exceeds the value with two projects.
10. **Long-term ROI.** Positive once there are many projects, part reuse, and people who were not
    there when the lessons were learned — roughly the ≥10-engineer, multi-team stage.
11. **Scale.** ≥ ~5 projects with reuse, ≥ ~10 engineers, or a compliance need for traceability.
12. **Simpler alternative (most of the value now).** Structured records *in the workspace files* that
    are graph-ready later: `decisions.jsonl` and `lessons.jsonl` with typed fields (what, why,
    evidence = evaluation ids, applies-to tags), plus a `vegeta ws export --json` and a cross-workspace
    `vegeta kb search` over those files (grep-grade). Every record keeps evaluation ids and artifact
    hashes, so a Neo4j importer later is mechanical. Result: the ontology is learned on real data before
    a database exists.
13. **Recommendation: postpone / only at scale.** Do the file-based records now; keep IDs and hashes
    stable; revisit when cross-project queries are a weekly need.

### 2.7 Docker image vs Docker Compose

1. **Problem.** Reproducible tool versions (OpenFOAM v2412 from conda-forge, CalculiX 2.21, PrusaSlicer
   2.7.2, CadQuery 2.8) across laptops and, later, cloud machines. `install_local.sh` works but takes
   minutes and depends on Ubuntu and network.
2. **Benefit (image).** One `docker run` gives the exact validated toolchain; the same image runs on a
   cloud VM or a batch runner; CI can run the integration tests. This is the cheapest insurance for the
   cloud option.
3. **Benefit (Compose).** Only if there are several services (API, database, viewer, worker). Today
   there are none.
4. **Disadvantages.** Image: a few GB; GUI-less (fine, everything is CLI); file permissions and
   mounting workspaces; Jupyter inside a container needs port/volume handling. Compose on laptops adds
   a service stack nobody asked for.
5. **Complexity.** Image low (Dockerfile ≈ `install_local.sh`). Compose low mechanically, but it
   presupposes services that do not exist.
6. **Infra cost.** Image: registry space (GHCR private repo). Compose: none extra until services exist.
7. **Maintenance.** Image: rebuild on tool upgrades; run `test.sh` inside it. Compose: per service.
8. **Architecture effect.** None; packaging only.
9. **Coupling.** None.
10. **Short-term ROI.** Image: moderate (onboarding, CI, reproducibility). Compose: none.
11. **Long-term ROI.** Image: high if the cloud move happens (the image *is* the migration).
12. **Simpler alternative.** Keep `install_local.sh` and pin versions in it (already done). The image
    is the next step, not a replacement.
13. **Recommendation:** image **implement now (small)**; Compose **only at scale**, when a server
    exists.

### 2.8 FastAPI control/API layer

1. **Problem.** Remote or multi-user access; a GUI backend; something for cloud deployment to expose.
2. **Benefit.** Needed the day Vegeta runs on a shared machine or in the cloud, or when Studio (GUI)
   needs a backend.
3. **Disadvantages and failure modes.** It is a server: auth, sessions, job queue for long solver
   runs, concurrency on file-based workspaces (write-once files help, but two users branching at once
   need locking), secrets, uptime. It duplicates the Python/CLI API and drifts unless generated from it.
   It directly contradicts "no server needed" if it becomes the primary path.
4. **Complexity.** Medium for read + trigger endpoints; high with background jobs, auth and multi-user.
5. **Infra cost.** A machine to run it, or cloud. Nothing today.
6. **Maintenance.** Ongoing (dependencies, security).
7. **Architecture effect.** Would become the integration point for Studio, MCP (remote) and cloud
   workers; must stay a thin wrapper over `vegeta.core`.
8. **Coupling.** `vegeta-api` → `vegeta-core`, FastAPI, a job runner.
9. **Short-term ROI.** None on laptops; the Python API and CLI are the API.
10. **Long-term ROI.** Necessary for cloud; but the *shape* is decided by Studio and MCP needs, so
    building it first is premature.
11. **Scale.** A shared server or cloud deployment, or ≥ 5 concurrent users.
12. **Simpler alternative.** Keep the Python API as the contract; make sure every operation is
    callable with plain data (already true); add a JSON summary export. A future API layer is then a
    mechanical wrapper.
13. **Recommendation: postpone.** Revisit with Studio or the cloud move; then decide between FastAPI
    (own UI) and a remote MCP server (agent front door) — possibly both over the same core.

### 2.9 Interactive 3D CAD/mesh/results viewer

1. **Problem.** Inspecting geometry, mesh quality and stress fields needs a viewer; today: CadQuery's
   notebook viewer for CAD, matplotlib scatter for FEA, PNGs for CFD, ParaView by hand.
2. **Benefit.** Faster inspection, fewer wrong region selections (clicking a face beats reading a
   surface table), trust in results (seeing a singularity at a support).
3. **Disadvantages.** Viewers are a large surface (formats, GPU, browser, ipywidgets); a custom one is a
   product of its own. The original spec explicitly says not to build a renderer unnecessarily.
4. **Complexity.** Low for exports (VTU/VTK from Talos results, STL already there) + existing viewers
   (ParaView, `pyvista`/`trame` in Jupyter, CadQuery's viewer). High for an integrated web viewer.
5. **Infra cost.** None for exports; a web viewer needs the API layer (2.8).
6. **Maintenance.** Exports: low. Integrated viewer: high.
7. **Architecture effect.** Exports: none. Integrated: part of Studio (Stage 10).
8. **Coupling.** `pyvista` optional extra at most.
9. **Short-term ROI.** VTU export + ParaView/pyvista: high value per hour of work. Face picking for
   region selection: the single most useful GUI feature, but it belongs to Studio.
10. **Long-term ROI.** Part of Studio; do it there.
11. **Scale.** Any; but the integrated form waits for Studio.
12. **Simpler alternative.** `talos.export_vtu(result)` + `pyvista.Plotter` in notebooks + ParaView.
13. **Recommendation:** exports **implement now (small)**; integrated viewer **later, with Studio**.

### 2.10 Summary

| Component | Recommendation | When | One-line reason |
|---|---|---|---|
| Closed-loop AI iteration | prototype | after Stage 11 + criteria | value is real, but as proposals into a deterministic workflow, human-approved |
| LLM adapters (Claude, OpenAI) | implement now | now | small, enabling, Stage 12 of the spec |
| Controlled loops | implement now | with the prototype | non-negotiable for anything that spends compute |
| Multi-agent | A now, D later, B postpone, C reject | — | specialists are code and templates, not prompts |
| MCP server | postpone → prototype read-only | when an external agent workflow exists | CLI `--json` already serves agents |
| Neo4j knowledge base | only at scale | ≥ ~10 engineers / many projects | file-based records now, importer later |
| Docker image | implement now (small) | now | reproducibility and the cloud option |
| Docker Compose | only at scale | with a server | no services to compose |
| FastAPI layer | postpone | with Studio / cloud | Python + CLI is the API today |
| 3D viewer | exports now, viewer with Studio | now / later | ParaView/pyvista cover inspection |

---

## 3. Contradictions and architectural risks

1. **"Closed loop" vs "nothing runs automatically".** These conflict unless the loop is an explicit,
   bounded object the engineer starts (`campaign.run()`), with a policy saying what may run without
   asking. Resolution: a campaign is one explicit action with a budget; inside it, only *parameter*
   proposals within engineer-set bounds may auto-run; everything else waits for approval. The
   philosophy holds at the level that matters: the engineer decides what can be tried.
2. **Agent autonomy vs engineering authority.** The loop must never be allowed to change loads,
   materials, boundary conditions, mesh settings, CFD templates or print settings on its own; those
   define *what is being tested*, and changing them changes the question. They are approval-only.
3. **MCP/FastAPI vs "no server".** Both are servers. Keep them optional adapters over the Python API;
   the workstation path must keep working without them. The cloud move will need one of them —
   decide which by the actual client (agent → MCP; GUI → FastAPI), not up front.
4. **Neo4j vs plain files.** A graph that is not derivable from the files is a second truth. If it ever
   exists, it must be an *index* rebuilt from workspaces, and lessons must be human-curated records.
5. **LLM-generated CadQuery is arbitrary code execution.** A proposal that patches `build()` runs
   Python on the engineer's machine. It must be shown as a diff, validated (build succeeds,
   measurements sane, parameters unchanged unless intended) and explicitly accepted — never executed
   from a chat or MCP call unattended.
6. **Validation by LLM.** Letting a model judge whether a result is good replaces physics with opinion.
   Verdicts come from criteria evaluated on tool metrics; the LLM may *explain*, never *decide*.
7. **Metric traps.** Peak nodal von Mises at a clamped edge is mesh-dependent; Cd on a coarse template
   is trend-level. A loop optimising these blindly converges on artefacts. Criteria need to be defined
   on trustworthy quantities (displacement, reaction balance, averaged stress away from supports,
   converged coefficients) — a human modelling decision, recorded with the campaign.
8. **Cost and time.** CFD in a loop is hours; without wall-clock and evaluation-count budgets one
   campaign can block a workstation for a day.
9. **Secrets on workstations.** API keys in environment variables per engineer; no shared secret
   store yet. Acceptable now, a task for the cloud move.
10. **Duplicated API surfaces.** Python, CLI, later MCP, later HTTP: four descriptions of the same
    operations. Mitigation: one Python API, thin generated wrappers, tests that exercise all of them.
11. **Branch drift.** `vegeta.core` exists only on `dev_core`/`main`; `dev_mvp` lacks it. Anything AI
    builds on core, so `main` (or `dev_core`) is the base for the prototype.

---

## 4. Recommended target architecture

```
                 engineer (Jupyter / CLI / later Studio)
                              │ starts, approves, labels
                              ▼
   ┌──────────────── vegeta-ai (optional) ─────────────────┐
   │  Criteria ──▶ Campaign(budget, limits, policy)         │
   │                 │ asks                 ▲ verdicts      │
   │            Proposer (Claude | OpenAI) │                │
   │                 │ typed Proposal      │                │
   │                 ▼                     │                │
   │        validate → approve/auto → apply│                │
   └───────────────────┬───────────────────┴────────────────┘
                       │ new revision / run evaluation (public API only)
                       ▼
   ┌──────────────── vegeta-core ──────────────────────────┐
   │ workspace files: revisions, evaluations, annotations, │
   │ decisions.jsonl, lessons.jsonl, campaigns/            │
   └───────┬──────────┬──────────┬───────────┬─────────────┘
           ▼          ▼          ▼           ▼
        dedalus     talos     aeromant    mellonia        (unchanged, independent)
           │          │          │           │
        STEP/STL   mesh/frd   case/logs   G-code        (native artifacts on disk)

   later, optional adapters over the same core:  MCP server (agents) · FastAPI (Studio/cloud)
   later, optional derived index:                Neo4j importer (from workspace files)
   packaging:                                    one Docker image = the validated toolchain
```

Principles:
- **LLM decides what to try; Vegeta executes, validates, records and controls.** Proposals are data
  (parameter set, source patch, analysis request, "stop"), never actions.
- **Campaign = explicit, bounded, recorded.** Budgets (iterations, evaluations per kind, wall-clock,
  tokens/cost), stop conditions (criteria met, no progress in N, budget), approval policy
  (auto for bounded parameter changes if the engineer opts in; approval for everything else).
- **Criteria are engineering statements on trustworthy metrics**, written by the engineer, versioned
  with the campaign.
- **Files remain the truth.** Campaign logs live in the workspace; anything else (graph, API) is derived.
- **Cloud-ready without a server today:** the Docker image, pure-data operations, and stable IDs/hashes
  make a later server or remote MCP a wrapper, not a rewrite.

---

## 5. Minimal useful next step

Implement **Stage 11 (compare + explicit sweeps) plus machine-readable acceptance criteria** in
`vegeta.core`, and a **`explain`/`propose` call through one adapter** that turns `ws.status()` and
criteria verdicts into (a) a plain-language explanation and (b) up to three *parameter* proposals the
engineer can apply with one call (`ws.apply(proposal)` → new revision, nothing run). No loop yet.

Why this first: it delivers value with zero autonomy, it produces the exact inputs a campaign needs
(comparison, verdicts, proposals), and it tests whether LLM proposals are any good on real parts
before any budget is spent on running them automatically.

---

## 6. Staged roadmap

**Now (next 1–3 months)**
- Stage 11: `ws.compare(...)`, `ws.sweep(design, grid, analyses=[...])` (explicit, budgeted, recorded).
- `Criteria` (constraints + objective on evaluation metrics) and verdicts recorded per revision.
- `decisions.jsonl` / `lessons.jsonl` conventions in workspaces; `vegeta ws export --json`.
- `vegeta-ai` v0: `Proposal` schema, `Proposer` protocol, Claude and OpenAI adapters, `explain`,
  `propose` (parameter proposals only), `apply` (creates revisions; runs nothing).
- Talos VTU export; pyvista/ParaView instructions.
- Dockerfile reproducing `install_local.sh`; `test.sh` inside the image; image in CI.

**Later (3–12 months, after the above is used on real parts)**
- `Campaign` loop with budgets, stop conditions, approval policy; auto-run of bounded parameter
  proposals as an opt-in; source-patch proposals shown as diffs, approval-only.
- Parallel independent review (pattern D) as an optional pre-run check using the second provider.
- Read-only MCP server (stdio) when an engineer uses Claude Code/IDE agents on engineering tasks.
- Studio with face picking and a results viewer; the API layer it needs (FastAPI or MCP-remote) decided
  then, and doubling as the cloud front door.

**Only at scale (≥ ~10 engineers, many projects, or cloud multi-user)**
- Neo4j (or another graph) as a derived index imported from workspace files; graph-RAG for proposals.
- Docker Compose / Kubernetes with an API, a job queue for solver runs, shared storage for artifacts,
  central secrets.
- Orchestrator-with-tools agents (pattern B) over the MCP/API layer, still under campaign controls.
- Specialist-agent teams (pattern C) only if evidence shows prompts beat templates for some discipline.

---

## 7. Sources consulted

- Anthropic agent-design guidance and current model/pricing table (bundled `claude-api` skill, cached
  2026-06): prefer single calls and workflows over agents; promote actions to dedicated tools when they
  must be gated or audited; effort levels; task budgets.
- Agent-loop control: [Human-in-the-loop AI agents, 2026 guide](https://dev.to/lusivision/human-in-the-loop-ai-agents-a-practical-2026-guide-1h54);
  [Galileo — human-in-the-loop oversight](https://galileo.ai/blog/human-in-the-loop-agent-oversight);
  [Strata — practicing the human in the loop](https://www.strata.io/blog/agentic-identity/practicing-the-human-in-the-loop/);
  [Stop Means Stop — enforcement gap in agent-framework control primitives (arXiv 2607.14166)](https://arxiv.org/pdf/2607.14166);
  [Externalization in LLM agents — harness engineering review (arXiv 2604.08224)](https://arxiv.org/pdf/2604.08224).
- MCP: [MCP security best practices (modelcontextprotocol.io, 2026-07-28)](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices);
  [The 2026-07-28 specification](https://blog.modelcontextprotocol.io/posts/2026-07-28/);
  [NSA/CISA CSI: MCP security design](https://media.defense.gov/2026/Jun/02/2003943289/-1/-1/0/CSI_MCP_SECURITY.PDF);
  [CSA agentic MCP security best practices](https://labs.cloudsecurityalliance.org/agentic/agentic-mcp-security-best-practices-v1/);
  [Corgea MCP security checklist](https://corgea.com/learn/mcp-security-best-practices).
- Neo4j / provenance: [Getting started with provenance and Neo4j](https://medium.com/neo4j/getting-started-with-provenance-and-neo4j-b50f666d8656);
  [Institutional knowledge graph for small organisations](https://www.smarason.is/en/blog/building-institutional-knowledge-graph-neo4j);
  [Neo4j knowledge graph use case](https://neo4j.com/use-cases/knowledge-graph/);
  [Multi-agent framework leveraging knowledge graphs for virtual commissioning (arXiv 2606.03255)](https://arxiv.org/pdf/2606.03255).
