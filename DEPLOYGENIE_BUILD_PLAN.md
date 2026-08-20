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
- **GitHub CLI (`gh`):** not installed. `open_pull_request()` falls back to returning a GitHub compare URL (`.../compare/base...branch?expand=1`) instead of failing outright when `gh` is missing.
- **7B-class local models are noticeably weaker** than Claude/GPT-4-class at reliable multi-step planning and structured tool-call output. This isn't a blocker, but expect the graph to need retry/validation logic specifically to compensate — a deliberate design point, not a workaround.
- **Local inference latency is real:** a single trivial `qwen2.5:7b-instruct` call measured ~35s (mostly model load + prompt eval on this hardware). A task request costs 3 sequential calls (classify → plan → code) before the approval gate, so expect 60–120s+ per task turn. Chat and approval/decline replies are 1 call or 0 calls (approval resume is pure Python, no LLM) and stay fast.

---

## Phase Plan

### Phase 0 — Scaffolding — **done**
`extension/` (TypeScript, compiles, one command that pings the backend) and `backend/` (FastAPI + the `LLMProvider` interface + Ollama adapter). Verified end-to-end: extension command → backend → Ollama → response.

**Pulled forward from Phase 10:** a real sidebar chat UI (VS Code webview view, themed to match the editor) so the extension is actually usable against any open repo instead of a single diagnostic command. Shows the active repo name, sends conversation history to a new `/chat` backend endpoint, and is explicit in its system prompt that it can't read or change files yet — that's still Phase 1. This is the interaction surface Phase 1's agent graph plugs into next; the UI shell existing first means each later phase adds capability behind an already-working interface instead of building UI and agent logic at the same time.

### Phase 1 — Vertical slice: Planner → Coding → Test → PR — **done**
One LangGraph graph (`backend/app/agents/graph.py`), driven entirely through the Phase 0 chat panel — no new UI needed. No MCP, no review agent, no deployment: file edits and git calls are plain Python/subprocess, on purpose (Phase 7 wraps them in MCP later, once they're understood).

Flow: `classify_intent` (LLM: is this a repo task or just conversation?) → `plan` (LLM: pick one target file + describe the change, as JSON) → `code` (LLM: write the file's full new contents, written to disk immediately) → `test` (detects and runs pytest/`npm test`/Django, no LLM) → `await_approval` — a real LangGraph `interrupt()`, pausing the graph and returning plan + test result to the chat panel — → `create_pr` (only on resume; git checkout/add/commit/push, then `gh pr create` or a compare-URL fallback).

Verified against a disposable scratch repo (never the real project repo): chit-chat correctly stayed conversational; a task request produced a correct file edit that passed its real pytest suite; approving triggered a real branch/commit and a graceful, clearly-reported failure on `git push` (no remote on the scratch repo — the failure path itself was the thing being tested); declining left the edit on disk, uncommitted, with no git side effects at all.

Deliberately out of scope here (arrives in later phases): no re-planning if tests fail — Phase 1 just reports failure and still asks for approval, since the fix-and-retry loop is Phase 2's reflection pattern; no multi-file changes; no conversation memory beyond the single pending approval (Phase 5).

**Also pulled forward from Phase 10:** the extension is now installed as a persistent VS Code extension (`vsce package` + `code --install-extension`, wrapped in a one-line `npm run release`) instead of only running via F5's Extension Development Host, and it auto-starts the backend itself (`extension/src/backendManager.ts`) — checks `/health`, spawns `uvicorn` if it isn't up, prompts once (folder picker) to locate the backend if unconfigured, then remembers the location in *user* (not workspace) settings so an untrusted repo's `.vscode/settings.json` can't redirect which Python binary gets launched.

### Phase 2 — Independent Review Agent + reflection loop — **done**
A `review` node sits between `test` and `await_approval` (`backend/app/agents/graph.py`). It's a separate LLM call with its own skeptical-reviewer prompt — not the Coding Agent grading its own work — that judges the diff against the request, factors in the real test result, and returns `APPROVED` or `CHANGES_REQUESTED` with critical/warning/suggestion feedback and a risk level, mirroring the design doc's Section 3.4 review-report format.

On `CHANGES_REQUESTED`, the graph routes back to `code` with the reviewer's feedback appended to the prompt, then re-tests and re-reviews — the reflection loop from Section 12. This is capped at `MAX_REVIEW_ROUNDS = 2` so a stuck local model can't loop forever; after the cap, it still surfaces everything (failing tests, the review's critical feedback, risk level) in the approval prompt instead of quietly proceeding as if nothing were wrong. The review is explicitly instructed to request changes whenever the real test result is FAILED, rather than trusting the model's own read of the diff for that call.

Verified against a fresh scratch repo: a normal request produced `APPROVED (risk: Low)` on the first pass; a deliberately broken request (rewriting `add` to multiply) produced a failing test, a `CHANGES_REQUESTED` verdict, a real loop back to `code`, a second failed attempt, and a correct stop at the round cap with the failure fully visible in the approval prompt — the small local model could not self-correct even with explicit feedback, which is a real, documented limitation rather than a bug to paper over. Declining after that left the repo untouched, as expected.

**Real-world bug found and fixed, same day:** a live request against an actual project ("add a field email in login page located in public folder") landed the change in a `login.html` at the repo root instead of the real `public/login.html` — the Planner's JSON output dropped the folder. Two fixes, both verified against a scratch repo reproducing the exact scenario:

1. **A deterministic correction, not just a prompt tweak** — `_resolve_target_file()` in `graph.py`: if the model's `target_file` doesn't match any real file but exactly one file in the repo shares that basename, the real path is used instead of the guess. Ambiguous or genuinely-new files are left alone. The Planner prompt was also tightened ("copy the path EXACTLY as shown, don't shorten it"), but the deterministic check is the part that actually can't fail silently.
2. **Also pulled forward from Phase 10: a real diff view.** The chat panel never showed *what* changed — only a text summary — so a wrong-file mistake was invisible until someone went and checked by hand. `/chat` now returns the target file plus its before/after content whenever a task reaches an approval step (`backend/app/main.py`), and the extension opens VS Code's native diff editor automatically (`extension/src/diffProvider.ts`, a `TextDocumentContentProvider` serving the "before" version as a virtual read-only document, since only the "after" version exists as a real file on disk by that point). You now see the exact change, file path included, before ever being asked to approve anything.

**Redesign, same day: per-stage approval gates, PR created early.** The single end-of-pipeline approval gate from Phase 1 was replaced with one gate per stage in the original design doc's diagram, and the pull request now gets created right after tests pass — before review, not after — with review Agent findings pushed as follow-up commits to that same PR. This is a deliberate move to match the design doc's own lifecycle diagram (Section 2) exactly, and to close the very gap the diff-view fix above was patching: more checkpoints where a mistake can be caught before it goes further.

New flow: `plan` → **approve** → `code` (writes the file) → **approve** → `test` (runs it for real) → **approve** → `publish` (creates the branch + PR on approval #1, or pushes a follow-up commit to the *same* branch on any retry) → `review` (runs automatically — an AI reading a PR doesn't need permission) → if `CHANGES_REQUESTED` and under the round cap, back to `code` for another approve/approve/approve/publish cycle against the same PR; otherwise `finish`, reporting the PR and stating plainly whether review's concerns were ever resolved.

Each declined gate stops the pipeline where it stands, with a message specific to what already happened — nothing was touched (declined at plan), the change is on disk but nothing's committed (declined at code or test), or the existing PR still reflects the previous version (declined mid-retry, after a PR already exists). Every new task run also explicitly resets all of a repo's per-task state first (`classify_intent` in `graph.py`) — without this, LangGraph's checkpointer would let an unrelated later request on the same repo mistake itself for a retry of an old, already-merged PR and try to push a "fix" to it.

Verified against a scratch repo with a real local git remote (a bare repo standing in for GitHub, since testing this needed `git push` to actually succeed — Phase 1/2's "no remote" scratch repos couldn't exercise the post-publish review path at all): all four gates worked with correct per-gate wording; a fresh unrelated second task correctly showed "push a branch and open a pull request" rather than "push this fix" (confirming the per-task reset works); a deliberately broken request pushed a real PR, got `CHANGES_REQUESTED`, looped back, and — notably, unlike Phase 2's earlier test — the model actually fixed the underlying bug on retry; round 2 then correctly flagged that the *fix*, while passing tests, no longer matched the original (deliberately bad) request to make `add` multiply — a genuinely correct, nuanced catch, not a false positive. Both commits landed on the same branch and PR, confirmed via `git log` on both the local repo and the fake remote.

### Phase 3 — Release Readiness + local Docker build — **next**
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
