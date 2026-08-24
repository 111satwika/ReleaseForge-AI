import * as vscode from 'vscode';
import { authHeader, ensureAuthenticated, ensureBackendRunning } from './backendManager';

interface EvalSummary {
  records_evaluated: number;
  agent: {
    total_tasks: number;
    completed: number;
    declined_by_user: number;
    failed: number;
    rolled_back: number;
    completion_rate: number | null;
    outcome_breakdown: Record<string, number>;
  };
  coding: {
    tests_run: number;
    tests_passed: number;
    test_pass_rate: number | null;
    reviews_run: number;
    reviews_approved: number;
    review_acceptance_rate: number | null;
    average_readiness_score: number | null;
  };
  deployment: {
    deployments_attempted: number;
    deployments_succeeded: number;
    deployment_success_rate: number | null;
    rollbacks: number;
    rollback_rate: number | null;
    strategy_breakdown: Record<string, { attempted: number; succeeded: number; rolled_back: number }>;
  };
}

interface AuditEvent {
  id: number;
  role: string;
  tool_name: string;
  server_name: string | null;
  workspace_path: string | null;
  status: string;
  detail: string | null;
  created_at: string;
}

interface Incident {
  id: number;
  workspace_path: string;
  release_id: number | null;
  deployment_strategy: string | null;
  action: string;
  root_cause: string;
  created_at: string;
}

interface FaithfulnessResult {
  faithful: boolean;
  unsupported_claims: string[];
  explanation: string;
  passages_checked: number;
}

interface FaithfulnessCheckState {
  question: string;
  answer: string;
  checking: boolean;
  result?: FaithfulnessResult;
  error?: string;
}

/**
 * A read-only "richer in-editor UI" surface (Phase 10) for the eval metrics (Phase 8)
 * and audit log (Phase 9) — both existed as real backend data with zero visibility
 * from inside the editor until now. A stat-tile KPI row, not a chart: each metric here
 * is a single headline number, exactly the case the design system's own form guidance
 * calls "not a chart" territory.
 */
export class DashboardPanel {
  public static currentPanel: DashboardPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private disposables: vscode.Disposable[] = [];
  private faithfulnessState: FaithfulnessCheckState | undefined;

  public static show(extensionUri: vscode.Uri): void {
    if (DashboardPanel.currentPanel) {
      DashboardPanel.currentPanel.panel.reveal();
      void DashboardPanel.currentPanel.refresh();
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      'releaseforge.dashboard',
      'ReleaseForge: Dashboard',
      vscode.ViewColumn.One,
      { enableScripts: true, retainContextWhenHidden: true },
    );
    DashboardPanel.currentPanel = new DashboardPanel(panel);
  }

  private constructor(panel: vscode.WebviewPanel) {
    this.panel = panel;
    this.panel.webview.html = this.getLoadingHtml();
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage((message) => {
      if (message.type === 'refresh') {
        void this.refresh();
      } else if (message.type === 'checkFaithfulness') {
        void this.checkFaithfulness(message.question ?? '', message.answer ?? '');
      }
    }, null, this.disposables);
    void this.refresh();
  }

  private dispose(): void {
    DashboardPanel.currentPanel = undefined;
    this.disposables.forEach((d) => d.dispose());
  }

  private async refresh(): Promise<void> {
    const backendUrl = vscode.workspace.getConfiguration('releaseforge').get<string>('backendUrl');
    try {
      const backendUp = await ensureBackendRunning();
      if (!backendUp) {
        this.panel.webview.html = this.getErrorHtml(
          "The ReleaseForge backend isn't running and couldn't be started automatically.",
        );
        return;
      }

      const token = await ensureAuthenticated();
      if (!token) {
        this.panel.webview.html = this.getErrorHtml('GitHub sign-in was not completed.');
        return;
      }
      const auth = authHeader(token);

      const folders = vscode.workspace.workspaceFolders;
      const workspacePath = folders && folders.length > 0 ? folders[0].uri.fsPath : undefined;
      const workspaceQuery = workspacePath ? `?workspace_path=${encodeURIComponent(workspacePath)}` : '';
      const workspaceParam = workspacePath ? `&workspace_path=${encodeURIComponent(workspacePath)}` : '';

      const [evalRes, auditRes] = await Promise.all([
        fetch(`${backendUrl}/eval/summary${workspaceQuery}`, { headers: auth }),
        fetch(`${backendUrl}/audit/log?limit=25${workspaceParam}`, { headers: auth }),
      ]);
      if (!evalRes.ok || !auditRes.ok) {
        throw new Error(`Backend returned ${evalRes.status}/${auditRes.status}`);
      }
      const evalSummary = (await evalRes.json()) as EvalSummary;
      const audit = (await auditRes.json()) as { events: AuditEvent[] };

      // /incidents requires workspace_path (there's no "across every repo" mode the way
      // /eval/summary and /audit/log have) — nothing to fetch without an open folder.
      let incidents: Incident[] = [];
      if (workspacePath) {
        const incidentsRes = await fetch(`${backendUrl}/incidents?limit=10${workspaceParam}`, { headers: auth });
        if (!incidentsRes.ok) {
          throw new Error(`Backend returned ${incidentsRes.status} for /incidents`);
        }
        incidents = ((await incidentsRes.json()) as { incidents: Incident[] }).incidents;
      }

      this.panel.webview.html = this.getHtml(evalSummary, audit.events, incidents, workspacePath, this.faithfulnessState);
    } catch (err) {
      this.panel.webview.html = this.getErrorHtml(`Couldn't reach the ReleaseForge backend at ${backendUrl}. (${err})`);
    }
  }

  private async checkFaithfulness(question: string, answer: string): Promise<void> {
    const backendUrl = vscode.workspace.getConfiguration('releaseforge').get<string>('backendUrl');
    if (!question.trim() || !answer.trim()) {
      return;
    }
    this.faithfulnessState = { question, answer, checking: true };
    await this.refresh(); // re-render immediately so "Checking…" shows before the (potentially slow) call below

    try {
      const backendUp = await ensureBackendRunning();
      if (!backendUp) {
        throw new Error("the backend isn't running");
      }
      const token = await ensureAuthenticated();
      if (!token) {
        throw new Error('GitHub sign-in was not completed');
      }
      const res = await fetch(`${backendUrl}/eval/faithfulness`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader(token) },
        body: JSON.stringify({ question, answer }),
      });
      if (!res.ok) {
        throw new Error(`backend returned ${res.status}`);
      }
      const result = (await res.json()) as FaithfulnessResult;
      this.faithfulnessState = { question, answer, checking: false, result };
    } catch (err) {
      this.faithfulnessState = { question, answer, checking: false, error: String(err) };
    }
    await this.refresh();
  }

  private getErrorHtml(message: string): string {
    return `<!DOCTYPE html><html><body style="font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 24px;">
      <p>${escapeHtml(message)}</p>
    </body></html>`;
  }

  private getLoadingHtml(): string {
    return `<!DOCTYPE html><html><body style="font-family: var(--vscode-font-family); color: var(--vscode-descriptionForeground); padding: 24px;">
      <p>Loading…</p>
    </body></html>`;
  }

  private getHtml(
    evalSummary: EvalSummary,
    events: AuditEvent[],
    incidents: Incident[],
    workspacePath: string | undefined,
    faithfulness: FaithfulnessCheckState | undefined,
  ): string {
    const tiles = [
      { label: 'Task completion', value: evalSummary.agent.completion_rate, sub: `${evalSummary.agent.completed} of ${evalSummary.agent.total_tasks} tasks` },
      { label: 'Test pass rate', value: evalSummary.coding.test_pass_rate, sub: `${evalSummary.coding.tests_passed} of ${evalSummary.coding.tests_run} runs` },
      { label: 'Review acceptance', value: evalSummary.coding.review_acceptance_rate, sub: `${evalSummary.coding.reviews_approved} of ${evalSummary.coding.reviews_run} reviews` },
      { label: 'Deployment success', value: evalSummary.deployment.deployment_success_rate, sub: `${evalSummary.deployment.deployments_succeeded} of ${evalSummary.deployment.deployments_attempted} deploys` },
      { label: 'Rollback rate', value: evalSummary.deployment.rollback_rate, sub: `${evalSummary.deployment.rollbacks} rollback(s)`, inverse: true },
    ];

    const outcomeRows = Object.entries(evalSummary.agent.outcome_breakdown)
      .sort((a, b) => b[1] - a[1])
      .map(([outcome, count]) => `<tr><td>${escapeHtml(outcome)}</td><td class="num">${count}</td></tr>`)
      .join('');

    const strategyRows = Object.entries(evalSummary.deployment.strategy_breakdown)
      .map(([strategy, s]) => `<tr><td>${escapeHtml(strategy)}</td><td class="num">${s.attempted}</td><td class="num">${s.succeeded}</td><td class="num">${s.rolled_back}</td></tr>`)
      .join('');

    const auditRows = events
      .map((e) => {
        const time = new Date(e.created_at).toLocaleString();
        return `<tr>
          <td class="dim">${escapeHtml(time)}</td>
          <td>${escapeHtml(e.role)}</td>
          <td><code>${escapeHtml(e.tool_name)}</code></td>
          <td>${statusPill(e.status)}</td>
          <td class="dim">${escapeHtml((e.detail ?? '').slice(0, 80))}</td>
        </tr>`;
      })
      .join('');

    const incidentRows = incidents
      .map((i) => {
        const time = new Date(i.created_at).toLocaleString();
        return `<tr>
          <td class="dim">${escapeHtml(time)}</td>
          <td>${escapeHtml(i.deployment_strategy ?? '—')}</td>
          <td>${actionPill(i.action)}</td>
          <td>${escapeHtml(i.root_cause)}</td>
        </tr>`;
      })
      .join('');

    const faithfulnessResultHtml = renderFaithfulnessResult(faithfulness);

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: var(--vscode-font-family);
    color: var(--vscode-foreground);
    background: var(--vscode-editor-background);
    padding: 20px 28px 40px;
    max-width: 980px;
  }
  h1 { font-size: 1.3em; margin: 0 0 2px; }
  .subtitle { color: var(--vscode-descriptionForeground); font-size: 0.9em; margin: 0 0 20px; }
  .toolbar { display: flex; justify-content: space-between; align-items: baseline; }
  #refresh {
    background: var(--vscode-button-secondaryBackground, transparent);
    color: var(--vscode-button-secondaryForeground, var(--vscode-foreground));
    border: 1px solid var(--vscode-button-border, var(--vscode-panel-border));
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 0.85em;
    cursor: pointer;
    font-family: inherit;
  }
  #refresh:hover { background: var(--vscode-list-hoverBackground); }
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin: 18px 0 28px;
  }
  .tile {
    background: var(--vscode-sideBar-background);
    border: 1px solid var(--vscode-panel-border);
    border-radius: 8px;
    padding: 12px 14px;
  }
  .tile .value {
    font-size: 1.7em;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
  }
  .tile .value.good { color: var(--vscode-charts-green, #3fb950); }
  .tile .value.bad { color: var(--vscode-charts-red, #f14c4c); }
  .tile .value.muted { color: var(--vscode-descriptionForeground); font-size: 1.3em; }
  .tile .label { font-size: 0.8em; color: var(--vscode-descriptionForeground); margin-top: 4px; }
  .tile .sub { font-size: 0.75em; color: var(--vscode-descriptionForeground); margin-top: 2px; }
  section { margin-bottom: 26px; }
  h2 { font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.04em; color: var(--vscode-descriptionForeground); margin: 0 0 8px; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 0.88em; }
  th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--vscode-panel-border); }
  th { color: var(--vscode-descriptionForeground); font-weight: 500; font-size: 0.85em; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.dim { color: var(--vscode-descriptionForeground); }
  code { font-family: var(--vscode-editor-font-family, monospace); font-size: 0.92em; }
  .empty { color: var(--vscode-descriptionForeground); font-size: 0.88em; padding: 8px 0; }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.82em;
    padding: 1px 8px;
    border-radius: 999px;
  }
  .pill .dot { width: 6px; height: 6px; border-radius: 50%; }
  .pill.ok { color: var(--vscode-charts-green, #3fb950); }
  .pill.ok .dot { background: var(--vscode-charts-green, #3fb950); }
  .pill.error { color: var(--vscode-charts-red, #f14c4c); }
  .pill.error .dot { background: var(--vscode-charts-red, #f14c4c); }
  .pill.scope_denied { color: var(--vscode-charts-yellow, #cca700); }
  .pill.scope_denied .dot { background: var(--vscode-charts-yellow, #cca700); }
  .pill.rolled_back { color: var(--vscode-charts-red, #f14c4c); }
  .pill.rolled_back .dot { background: var(--vscode-charts-red, #f14c4c); }
  .pill.left_as_is { color: var(--vscode-charts-yellow, #cca700); }
  .pill.left_as_is .dot { background: var(--vscode-charts-yellow, #cca700); }
  .pill.faithful { color: var(--vscode-charts-green, #3fb950); }
  .pill.faithful .dot { background: var(--vscode-charts-green, #3fb950); }
  .pill.not-faithful { color: var(--vscode-charts-red, #f14c4c); }
  .pill.not-faithful .dot { background: var(--vscode-charts-red, #f14c4c); }
  .check-form { display: flex; flex-direction: column; gap: 8px; max-width: 640px; }
  .check-form label { font-size: 0.82em; color: var(--vscode-descriptionForeground); }
  .check-form textarea, .check-form input {
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent);
    border-radius: 4px;
    padding: 6px 8px;
    font-family: inherit;
    font-size: 0.9em;
    resize: vertical;
  }
  .check-form button {
    align-self: flex-start;
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    border: none;
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 0.85em;
    cursor: pointer;
    font-family: inherit;
  }
  .check-form button:hover { background: var(--vscode-button-hoverBackground); }
  .check-form button:disabled { opacity: 0.6; cursor: default; }
  .check-result { margin-top: 6px; font-size: 0.88em; }
  .check-result .claims { margin: 6px 0 0; padding-left: 18px; }
  .check-result .explanation { color: var(--vscode-descriptionForeground); margin-top: 4px; }
</style>
</head>
<body>
  <div class="toolbar">
    <div>
      <h1>Release Dashboard</h1>
      <p class="subtitle">${workspacePath ? escapeHtml(workspacePath) : 'No repo open'} · ${evalSummary.records_evaluated} record(s) evaluated</p>
    </div>
    <button id="refresh">Refresh</button>
  </div>

  <div class="kpi-row">
    ${tiles.map((t) => renderTile(t)).join('')}
  </div>

  <section>
    <h2>Outcomes</h2>
    ${outcomeRows ? `<table><tbody>${outcomeRows}</tbody></table>` : '<div class="empty">No tasks recorded yet.</div>'}
  </section>

  <section>
    <h2>Deployment strategies</h2>
    ${strategyRows
      ? `<table><thead><tr><th>Strategy</th><th class="num">Attempted</th><th class="num">Succeeded</th><th class="num">Rolled back</th></tr></thead><tbody>${strategyRows}</tbody></table>`
      : '<div class="empty">No deployments recorded yet.</div>'}
  </section>

  <section>
    <h2>Recent audit events</h2>
    ${auditRows
      ? `<table><thead><tr><th>Time</th><th>Role</th><th>Tool</th><th>Status</th><th>Detail</th></tr></thead><tbody>${auditRows}</tbody></table>`
      : '<div class="empty">No MCP tool calls recorded yet.</div>'}
  </section>

  <section>
    <h2>Recent incidents</h2>
    ${incidentRows
      ? `<table><thead><tr><th>Time</th><th>Strategy</th><th>Action</th><th>Root cause</th></tr></thead><tbody>${incidentRows}</tbody></table>`
      : `<div class="empty">${workspacePath ? 'No rollbacks recorded yet.' : 'Open a repo to see its incident history.'}</div>`}
  </section>

  <section>
    <h2>Faithfulness check</h2>
    <p class="subtitle" style="margin: 0 0 10px;">Ask a real question this chat could receive, paste in an answer, and check whether it's actually supported by what ReleaseForge would have retrieved for it — the same check <code>POST /eval/faithfulness</code> runs.</p>
    <form class="check-form" id="faithfulness-form">
      <label for="ff-question">Question</label>
      <input id="ff-question" type="text" value="${escapeHtml(faithfulness?.question ?? '')}" placeholder="e.g. what vector database does hybrid retrieval use?" />
      <label for="ff-answer">Answer to check</label>
      <textarea id="ff-answer" rows="3" placeholder="Paste the answer you want fact-checked against retrieved passages">${escapeHtml(faithfulness?.answer ?? '')}</textarea>
      <button type="submit" id="ff-submit" ${faithfulness?.checking ? 'disabled' : ''}>${faithfulness?.checking ? 'Checking…' : 'Check faithfulness'}</button>
    </form>
    ${faithfulnessResultHtml}
  </section>

<script>
  const vscode = acquireVsCodeApi();
  document.getElementById('refresh').addEventListener('click', () => {
    vscode.postMessage({ type: 'refresh' });
  });
  document.getElementById('faithfulness-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const question = document.getElementById('ff-question').value;
    const answer = document.getElementById('ff-answer').value;
    document.getElementById('ff-submit').disabled = true;
    vscode.postMessage({ type: 'checkFaithfulness', question, answer });
  });
</script>
</body>
</html>`;
  }
}

function renderTile(t: { label: string; value: number | null; sub: string; inverse?: boolean }): string {
  if (t.value === null) {
    return `<div class="tile"><div class="value muted">—</div><div class="label">${escapeHtml(t.label)}</div><div class="sub">No data yet</div></div>`;
  }
  const pct = Math.round(t.value * 100);
  // "inverse" tiles (rollback rate) read as good when low, not high — the color logic
  // follows what the number actually means, not just "bigger is greener."
  const good = t.inverse ? pct <= 20 : pct >= 70;
  const bad = t.inverse ? pct > 50 : pct < 40;
  const cls = good ? 'good' : bad ? 'bad' : '';
  return `<div class="tile"><div class="value ${cls}">${pct}%</div><div class="label">${escapeHtml(t.label)}</div><div class="sub">${escapeHtml(t.sub)}</div></div>`;
}

function statusPill(status: string): string {
  const labels: Record<string, string> = { ok: 'OK', error: 'Error', scope_denied: 'Scope denied' };
  const cls = ['ok', 'error', 'scope_denied'].includes(status) ? status : 'error';
  return `<span class="pill ${cls}"><span class="dot"></span>${escapeHtml(labels[status] ?? status)}</span>`;
}

function actionPill(action: string): string {
  const labels: Record<string, string> = { rolled_back: 'Rolled back', left_as_is: 'Left as-is' };
  const cls = action === 'rolled_back' ? 'rolled_back' : 'left_as_is';
  return `<span class="pill ${cls}"><span class="dot"></span>${escapeHtml(labels[action] ?? action)}</span>`;
}

function renderFaithfulnessResult(state: FaithfulnessCheckState | undefined): string {
  if (!state || state.checking) {
    return '';
  }
  if (state.error) {
    return `<div class="check-result">Couldn't run the check: ${escapeHtml(state.error)}</div>`;
  }
  if (!state.result) {
    return '';
  }
  const { faithful, unsupported_claims, explanation, passages_checked } = state.result;
  const pill = faithful
    ? '<span class="pill faithful"><span class="dot"></span>Faithful</span>'
    : '<span class="pill not-faithful"><span class="dot"></span>Not faithful</span>';
  const claims = unsupported_claims.length
    ? `<ul class="claims">${unsupported_claims.map((c) => `<li>${escapeHtml(c)}</li>`).join('')}</ul>`
    : '';
  return `<div class="check-result">
    ${pill} <span class="dim">(checked against ${passages_checked} retrieved passage${passages_checked === 1 ? '' : 's'})</span>
    ${claims}
    <div class="explanation">${escapeHtml(explanation)}</div>
  </div>`;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
