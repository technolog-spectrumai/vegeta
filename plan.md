# Plan: from open-loop workbench to controlled, AI-assisted iteration

Derived from `exp.md`. Ordered by value per unit of complexity; each step is useful on its own and
none requires a server, a database or an agent framework. Effort is for one engineer familiar with the
code. Human-authority rules (last section) apply to every step.

## Step 1 — Comparison and explicit sweeps (Stage 11)  · effort ~1 week

- **Goal.** Deterministic building blocks any loop needs, valuable on their own.
- **Scope.** `ws.compare(rids, metrics=None) -> Table` over existing results only (never generates);
  `ws.sweep(design, grid: dict[str, list], base=rev, analyses=[("fea", "static", factory)], limits=...)`
  creating revisions explicitly and running the named analyses with a wall-clock / count limit;
  CLI `vegeta ws compare`, `vegeta ws sweep`; results recorded like any evaluation; a sweep record
  (`sweeps/<name>.json`) listing revisions and limits.
- **Done when.** Compare works on a workspace with NOT RUN cells; a 3×3 sweep on the bracket runs with a
  limit and records every revision; tests + notebook `07_compare_sweep.ipynb`.
- **Risks.** Temptation to add optimisation logic — do not; a grid is a grid.

## Step 2 — Acceptance criteria and verdicts  · effort ~3–4 days

- **Goal.** Make "is this design acceptable" a machine-readable, recorded statement.
- **Scope.** `Criteria(name, constraints=[Constraint("fea:static", "safety_factor_yield", ">=", 2.0), ...],
  objective=Objective("dedalus", "volume", "min"))`; `criteria.evaluate(revision) -> Verdict`
  (`pass`/`fail`/`incomplete` with per-constraint values, NOT RUN → incomplete); verdicts written to
  `annotations.jsonl`; `ws.status(criteria=...)` shows them. Only metrics that exist in evaluation
  records; no derived physics.
- **Done when.** Verdicts appear in status; a constraint on a NOT RUN analysis yields `incomplete`, never
  `fail`; tests.
- **Risks.** Engineers writing criteria on mesh-dependent peaks; mitigate with docs and a `note` field
  explaining the choice of metric (mandatory).

## Step 3 — `vegeta-ai` v0: proposals without a loop  · effort ~1–2 weeks

- **Goal.** Test whether LLM proposals are useful on real parts before automating anything.
- **Scope.** New distribution `vegeta-ai/` (`vegeta.ai`), depends on `vegeta-core`; optional extras
  `[anthropic]`, `[openai]`.
  - `Proposal` types (data only): `ParameterProposal(changes, rationale)`, `AnalysisRequest(kind, name)`,
    `SourcePatchProposal(diff, rationale)`, `Stop(reason)`.
  - `Proposer` protocol: `propose(context: dict, criteria, n=3) -> list[Proposal]`,
    `explain(context: dict) -> str`. Context = `ws.status()` rows + verdicts + parameter tables, as JSON.
  - Adapters: `ClaudeProposer` (official `anthropic` SDK, structured output against the proposal JSON
    schema, adaptive thinking, model default `claude-opus-5`), `OpenAIProposer` (official `openai` SDK,
    structured output). Model IDs and prices are configuration values.
  - `ws.apply(proposal, note=...)`: parameter proposal → `branch()` (runs nothing); source patch →
    written to `proposals/<id>.diff`, shown, applied only by an explicit `accept()`; every proposal and
    decision appended to `proposals.jsonl` with prompt/response hashes and token usage.
  - CLI: `vegeta ai explain -w ws`, `vegeta ai propose -w ws --criteria criteria.py:CR`, `vegeta ai apply <id>`.
- **Done when.** On the bracket workspace, `propose` returns bounded parameter proposals; `apply` creates
  revisions; nothing runs; tests use a `FakeProposer` (no network); one adapter test each behind an
  API-key skip.
- **Risks.** Proposals outside parameter bounds (validate with the design's `ParameterSet`, reject and
  record); provider API drift (pin SDK versions; keep the adapter ≤ 150 lines).

## Step 4 — Controlled campaign loop  · effort ~2 weeks, only after Step 3 has been used on real parts

- **Goal.** Bounded, recorded closed loop: propose → (approve) → apply → run named analyses → verdict →
  repeat, under hard limits.
- **Scope.** `Campaign(workspace, design, criteria, analyses, proposer, budget=Budget(max_iterations,
  max_evaluations={"fea": 20, "cfd": 4}, wall_clock_s, max_tokens|max_cost_usd), stop=StopRules(
  criteria_met=True, no_progress_after=5), policy=Policy(auto_apply="parameters_within_bounds" |
  "none", approval=callback))`; `campaign.run()` is the single explicit action; state persisted after
  every step to `campaigns/<name>/campaign.json` + `events.jsonl` (resumable, inspectable, `NOT RUN`
  semantics untouched); terminal states `done | budget_exhausted | no_progress | stopped | failed`.
  Default policy: approval required for everything; auto-apply of parameter proposals is opt-in and
  bounded by the design's parameter ranges plus optional tighter campaign bounds.
- **Done when.** A campaign on the bracket reaches SF ≥ 2 at minimal volume within limits using the
  `FakeProposer`; a token/cost budget stops it; interrupting (`cancel` event) stops the running solver
  through the tools' existing cancel support; every event is on disk.
- **Risks.** Runs blocking a workstation — wall-clock budget is mandatory; loops chasing artefacts —
  criteria notes and the metric guidance in `docs/`.

## Step 5 — Docker image  · effort ~2–3 days

- **Goal.** The validated toolchain as one image; the path to CI and to the cloud.
- **Scope.** `Dockerfile` mirroring `install_local.sh` (Ubuntu 24.04, apt tools, conda-forge OpenFOAM,
  venv with `vegeta-cli`, `vegeta-core`, later `vegeta-ai`); `docker run -v $PWD/runs:/runs vegeta test`
  runs `test.sh`; Jupyter exposed on request; image built in CI and pushed to a private registry.
- **Done when.** `test.sh` passes inside the image; the sphere CFD validation runs inside it.
- **Risks.** Image size (several GB) — acceptable; keep one image, no Compose.

## Step 6 — Results export and inspection  · effort ~2 days

- **Scope.** `talos.export_vtu(result)` (nodes, displacement, von Mises) + docs for ParaView and
  `pyvista` in Jupyter; Aeromant already leaves the OpenFOAM case for ParaView (`paraFoam`/`foamToVTK`
  noted in docs).
- **Done when.** A VTU from the cantilever opens in ParaView with the fields; notebook cell with pyvista.

## Step 7 — Evaluate on a real part  · continuous

- Run Steps 1–3 (then 4) on a real bracket/arm/mount from a current project. Record in
  `lessons.jsonl` what the proposals got right and wrong. Decide from evidence whether the campaign
  (Step 4) and independent reviews are worth extending.

## Later (entry criteria, not dates)

| Item | Start when |
|---|---|
| Parallel independent review (second provider critiques a proposal before expensive runs) | a campaign has spent ≥ 1 CFD-hour on a bad proposal |
| Read-only MCP server (stdio) | an engineer uses Claude Code / an IDE agent on engineering tasks weekly |
| Studio (face picking, results viewer) + its API layer | Stage 10 is scheduled; choose FastAPI vs remote MCP by the client |
| Cloud deployment (image on a VM, shared runs storage, secrets) | a second team, or workstations too slow for CFD |
| Neo4j derived index + graph-RAG | ≥ ~10 engineers or ≥ ~5 projects with part/lesson reuse |

## Human-authority rules for every step

1. The AI proposes data; Vegeta applies it through the public API; the engineer starts every campaign.
2. Loads, materials, boundary conditions, mesh settings, CFD templates/values and print settings are
   never changed by a proposal without explicit approval; they define the question, not the answer.
3. Source patches are diffs, validated (build succeeds, parameters intact, measurements sane) and
   accepted explicitly; never executed from a chat, MCP or API call unattended.
4. Verdicts come from criteria on tool metrics; the LLM explains, it does not judge.
5. Every proposal, decision, budget and stop is recorded in the workspace files.
