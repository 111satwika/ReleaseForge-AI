import * as vscode from 'vscode';
import { ensureBackendRunning } from './backendManager';

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

      const folders = vscode.workspace.workspaceFolders;
      const workspacePath = folders && folders.length > 0 ? folders[0].uri.fsPath : undefined;
      const workspaceQuery = workspacePath ? `?workspace_path=${encodeURIComponent(workspacePath)}` : '';

      const [evalRes, auditRes] = await Promise.all([
        fetch(`${backendUrl}/eval/summary${workspaceQuery}`),
        fetch(`${backendUrl}/audit/log?limit=25${workspacePath ? `&workspace_path=${encodeURIComponent(workspacePath)}` : ''}`),
      ]);
      if (!evalRes.ok || !auditRes.ok) {
        throw new Error(`Backend returned ${evalRes.status}/${auditRes.status}`);
      }
      const evalSummary = (await evalRes.json()) as EvalSummary;
      const audit = (await auditRes.json()) as { events: AuditEvent[] };
      this.panel.webview.html = this.getHtml(evalSummary, audit.events, workspacePath);
    } catch (err) {
      this.panel.webview.html = this.getErrorHtml(`Couldn't reach the ReleaseForge backend at ${backendUrl}. (${err})`);
    }
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

  private getHtml(evalSummary: EvalSummary, events: AuditEvent[], workspacePath: string | undefined): string {
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

<script>
  const vscode = acquireVsCodeApi();
  document.getElementById('refresh').addEventListener('click', () => {
    vscode.postMessage({ type: 'refresh' });
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

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
