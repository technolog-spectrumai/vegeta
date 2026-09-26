# Rejected ideas (and why)

Ideas considered during the exploration in `exp.md` that should not be pursued now. "Rejected" means
not worth its complexity at 1–10 engineers on workstations; some may return under the entry criteria
in `plan.md`.

| Idea | Why not |
|---|---|
| **Fully autonomous optimisation loop** (LLM runs CAD/FEA/CFD until it is "happy") | Contradicts the founding rule that nothing runs automatically; LLMs do not know physics and will optimise mesh artefacts (support-edge stress peaks, coarse-mesh Cd); unbounded compute. Bounded campaigns with criteria and approvals (`plan.md` Step 4) keep the value. |
| **Multi-agent "engineering team"** (CAD agent, FEA agent, CFD agent, manufacturing agent negotiating) | The specialists' knowledge already lives in deterministic code, templates and explicit configuration; agent-to-agent chatter is expensive, non-reproducible and unauditable; consensus between models is not evidence. One proposer + deterministic workflow gives the same outcomes with a fraction of the moving parts. |
| **Orchestrator agent that calls Vegeta tools directly** (bash/MCP with write access deciding what runs) | Moves control from code to the model; every tool call then needs its own budget and approval wrapper; harder to reproduce than a recorded campaign. Possible later over an MCP/API layer, under the same campaign controls. |
| **Letting the LLM judge results** ("does this look acceptable?") | Replaces physics with opinion. Verdicts must come from criteria evaluated on tool metrics; the model may explain and suggest, never decide. *Exception: Fidia's review of rendered views (see below) scores appearance only; validity is decided by deterministic checks.* |
| **Neo4j now** | Two projects and 1–3 engineers would leave the graph empty and unmaintained; it duplicates provenance the workspace files already hold; a service to run, back up and learn (Cypher, ontology). File-based `decisions.jsonl` / `lessons.jsonl` with stable evaluation ids and artifact hashes give most of the value and make a later importer mechanical. |
| **Storing artifacts (STEP/STL/mesh/FRD/cases/G-code) in a database** | Large, binary, already reproducible from records; keep on disk or object storage and reference by hash + path. |
| **FastAPI + database server now** | No server exists in the deployment model; the Python API and CLI are the API; a server brings auth, queues, concurrency on file workspaces and uptime with no user for it yet. Build it when Studio or the cloud move needs it, and choose the shape (FastAPI vs remote MCP) by the client. |
| **Docker Compose on laptops** | There are no services to compose; a single image covers reproducibility and the cloud option. Compose (or Kubernetes) belongs with a server. |
| **Agent frameworks as a core dependency** (LangChain/LangGraph/CrewAI/AutoGen-style) | Their loops and "stop" primitives are not reliably enforced, their abstractions hide token spend and tool calls, and they churn faster than Vegeta. Two thin adapters over official SDKs and a hand-written, deterministic campaign loop are smaller and inspectable. |
| **A generic "LLM abstraction layer" covering tool use, streaming and agents for both providers** | Lowest-common-denominator adapters lose the features that make proposals reliable (strict structured outputs, thinking/effort). Keep the contract narrow: `propose` and `explain` with JSON schemas. |
| **Executing LLM-written CadQuery code unattended** | It is arbitrary code execution on the engineer's machine; also silently changes design intent. Patches are diffs, validated and explicitly accepted. *Exception: Fidia (below).* |
| **Write-capable MCP tools from chat clients without campaign controls** | Bypasses budgets and approvals; MCP servers run with the developer's privileges and are exposed to prompt injection through tool descriptions and results. Read-only first; writes only through the same policy machinery as campaigns. |
| **Custom 3D renderer / web viewer now** | A product of its own; the original spec forbids unnecessary renderers. VTU export + ParaView/pyvista and CadQuery's viewer cover inspection until Studio. |
| **Automatic chaining CAD → FEA → CFD → print** ("run everything on every revision") | Wastes hours of CFD on variants that fail FEA; hides the engineer's choice of what to test. Sweeps and campaigns name their analyses explicitly and are budgeted. |
| **Keeping a second, editable copy of results for the LLM** (summaries the model rewrites) | Creates drift between what was measured and what is reported. The model reads the recorded metrics; its explanations are stored separately and marked as such. |
| **Environment/config files for AI settings** (YAML for prompts, budgets, models) | Violates the Python-configuration rule; budgets, criteria and policies are Python objects recorded in the workspace. API keys stay in environment variables because they connect to an external service. |

## A written exception: Fidia prompt-to-3D
Fidia (`docs/fidia.md`) runs LLM-written CadQuery in a loop and lets a model review the rendered result. It is
allowed because: the code never runs in the engineer's Python process (a subprocess with an import allow-list, no
API keys in its environment, time, memory and file-size limits, the process group killed on timeout or cancel;
network isolation with a wrapper); the review is advice — only deterministic checks (B-rep validity, watertight,
winding, triangle budget, export round-trip) make a revision valid, and an invalid revision can never become the
output; the loop is bounded (iterations, minutes, tokens, patience), recorded and resumable; an approval hook
(`approve="ask"`, `fidia run --confirm`) runs nothing without a yes; and its product is a model to look at and use,
not an engineering verdict. The copilot's validation uses the same sandbox.
