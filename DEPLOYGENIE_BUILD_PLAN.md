# DeployGenie AI — Build Plan

This is the working plan for turning [`ReleaseForge_AI_Complete_Project_Design.md`](./ReleaseForge_AI_Complete_Project_Design.md) — the full 15-agent, MCP/RAG/knowledge-graph, cloud-native vision — into something actually built, incrementally, in a way that's meant to teach the concepts along the way rather than just produce a working system.

The design doc is the source of truth for **what** the finished platform covers. This document is the source of truth for **in what order** we build it, and **why** that order isn't the doc's own Stage 1–16 sequence.

---

## Why not build it in the doc's original order

The doc's roadmap builds MCP integrations (Stage 2) and the LangGraph workflow (Stage 3) before there's a single agent that does anything useful (Stage 4+). That's backwards for both engineering risk and learning: it sinks time into plumbing before validating that the core loop works at all.

Instead, this plan follows one rule: **build a thin vertical slice end-to-end first, then widen it.** Get one path — Planner → Coding → Test → PR — working with the crudest possible tools (direct filesystem + git calls, no MCP) before adding review, deployment, monitoring, memory, RAG, or a knowledge graph. Every later phase either widens that working core or retrofits proper infrastructure (like MCP) around tools that already exist and are already understood.

Every item in the design doc's Section 22 concept checklist still gets built — just in an order where each phase produces something real that the next phase actually depends on.

---

## Architecture

```
VS Code Extension (TypeScript, thin client)
        |  reads the active workspace folder as the target repo
        |  localhost HTTP (configurable via deploygenie.backendUrl)
        v
DeployGenie Core (Python: FastAPI + LangGraph — the agent brain)
        |
        v
LLM Provider  (Ollama now -> OpenAI / Anthropic later, same interface)
        |
        v
MCP servers / tools   (introduced in Phase 7, once we know what to wrap)
```

**Why a VS Code extension instead of a web app:** `vscode.workspace.workspaceFolders` gives the active repo path for free — no manual "point it at a repo" config. VS Code's native UI (diff viewer, Quick Pick, notifications, output channels) covers most of the design doc's human-in-the-loop needs (Section 13) without building a custom React frontend.

**Why the agent brain is a separate Python process, not TypeScript in the extension itself:** LangGraph, the official MCP SDK, LlamaIndex, and the Neo4j driver are all Python-first and meaningfully more mature than their JS equivalents. The extension stays a thin client — UI plus "here's the repo, here's the request" — and the FastAPI/LangGraph service does the actual orchestration.

**LLM provider strategy:** `backend/app/llm/base.py` defines one `LLMProvider` interface. The Ollama adapter (`ollama_provider.py`) is built on the `openai` Python SDK pointed at Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`) rather than a bespoke Ollama client — so adding real OpenAI later is a different `base_url` / `api_key` / model name, not a rewrite. An Anthropic adapter needs its own implementation (different message format) but slots into the same interface. Selection happens via `DEPLOYGENIE_LLM_PROVIDER` in `backend/app/llm/factory.py`.

---

## Environment notes

- **Ollama models available:** `qwen2.5:7b-instruct` (general reasoning/planning), `llama3.2:3b`, `llava:latest` (vision, unused so far). No code-specialized model pulled yet — `qwen2.5-coder:7b` is recommended once the Coding Agent needs real code edits.
- **Docker Desktop:** installed but not on PATH in PowerShell or Git Bash as of this writing — needs a look before Phase 3 (image builds).
- **7B-class local models are noticeably weaker** than Claude/GPT-4-class at reliable multi-step planning and structured tool-call output. This isn't a blocker, but expect the graph to need retry/validation logic specifically to compensate — a deliberate design point, not a workaround.

---

## Phase Plan

### Phase 0 — Scaffolding — **done**
`extension/` (TypeScript, compiles, one command that pings the backend) and `backend/` (FastAPI + the `LLMProvider` interface + Ollama adapter). Verified end-to-end: extension command → backend → Ollama → response.

**Pulled forward from Phase 10:** a real sidebar chat UI (VS Code webview view, themed to match the editor) so the extension is actually usable against any open repo instead of a single diagnostic command. Shows the active repo name, sends conversation history to a new `/chat` backend endpoint, and is explicit in its system prompt that it can't read or change files yet — that's still Phase 1. This is the interaction surface Phase 1's agent graph plugs into next; the UI shell existing first means each later phase adds capability behind an already-working interface instead of building UI and agent logic at the same time.

### Phase 1 — Vertical slice: Planner → Coding → Test → PR — **next**
One LangGraph graph, three-ish nodes, editing real files in a real target repo, running its test suite, opening a real PR via the GitHub CLI. No MCP, no review agent, no deployment. Proves the entire "agent writes code and it actually works" premise before anything else is built on top of it.

### Phase 2 — Independent Review Agent + reflection loop
A separate LLM context/prompt from the Coding Agent (same-agent review misses its own blind spots). Adds the CHANGES_REQUESTED → fix → re-review branch in the graph — the design doc's first reflection pattern (Section 12).

### Phase 3 — Release Readiness + local Docker build
A scoring agent (tests/review/security/build → GO/NO-GO) plus Dockerfile generation and a local image build. Introduces structured/numeric LLM output and a genuine human-in-the-loop approval gate (Section 13) before deployment complexity arrives.

### Phase 4 — Deployment Strategy + local Kubernetes + Validation/Monitoring/Recovery
Rolling/blue-green/canary strategy selection, a real deploy to `kind`/`minikube`, smoke tests, a monitoring poll loop, and rollback-on-failure. The heaviest infra phase — it only makes sense once there's a real image and a real release-readiness verdict to act on.

### Phase 5 — Memory layers (Postgres-backed)
Conversation / project / release / incident memory. Comes after Phase 4 so there's real release and incident history to store, not synthetic test data.

### Phase 6 — RAG + Knowledge Graph (Qdrant/pgvector + Neo4j)
Retrieval over the release reports and incident records Phase 5 generated, plus ingested docs. Real system history teaches RAG/GraphRAG concepts better than retrieval over generic documents.

### Phase 7 — MCP-ify the tool layer
Retrofit the ad-hoc git/docker/k8s calls from Phases 1–4 into proper MCP servers plus one MCP client. Deliberately late: by now it's clear exactly what each agent needs, so wrapping it in MCP teaches what the protocol actually buys (tool discovery, permissioning, multi-server orchestration) instead of building ceremony around tools that don't exist yet.

### Phase 8 — Observability + Evaluation
OpenTelemetry traces, Prometheus/Grafana, token/cost tracking, task success rate, review acceptance rate, deployment success rate — instrumenting a system that's now stable enough to be worth watching.

### Phase 9 — Security hardening
RBAC, per-agent tool scoping (per the design doc's Section 14 table), secrets management, audit logs — retrofit onto a stable agent/tool graph rather than guessing permission boundaries upfront.

### Phase 10 — Extension UI polish + Cloud + CI/CD
Richer in-editor UI (webview panel, inline diff review), deploy DeployGenie itself to AWS (ECR/ECS or EKS), GitHub Actions CI/CD.

---

## Concept coverage

Every phase maps back to the design doc's Section 22 checklist — nothing in the original vision is dropped, only reordered:

| Concept area | Covered in |
|---|---|
| Multi-agent systems, planning, routing, reflection | Phases 1–2 |
| Human-in-the-loop, structured output | Phases 1–3 |
| Docker, deployment strategies, Kubernetes | Phases 3–4 |
| Memory | Phase 5 |
| RAG, hybrid retrieval, reranking, Graph RAG | Phase 6 |
| MCP client/server, tools, resources, permissions | Phase 7 |
| Observability, evaluation, cost/latency tracking | Phase 8 |
| Security, RBAC, audit logs | Phase 9 |
| Cloud (AWS), CI/CD, agent evaluation at scale | Phase 10 |
