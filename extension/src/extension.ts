import * as vscode from 'vscode';
import { authHeader, ensureAuthenticated, ensureBackendRunning, initAuth, stopBackend } from './backendManager';
import { ChatViewProvider } from './chatViewProvider';
import { DashboardPanel } from './dashboardPanel';
import { RELEASEFORGE_DIFF_SCHEME, OriginalContentProvider } from './diffProvider';

export function activate(context: vscode.ExtensionContext) {
  initAuth(context);
  const originalContentProvider = new OriginalContentProvider();
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(RELEASEFORGE_DIFF_SCHEME, originalContentProvider),
  );

  const chatProvider = new ChatViewProvider(context.extensionUri, originalContentProvider);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, chatProvider),
  );

  void ensureBackendRunning();
  context.subscriptions.push({ dispose: stopBackend });

  context.subscriptions.push(
    vscode.commands.registerCommand('releaseforge.restartBackend', async () => {
      stopBackend();
      await ensureBackendRunning();
    }),
  );

  const disposable = vscode.commands.registerCommand('releaseforge.ping', async () => {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
      vscode.window.showErrorMessage('ReleaseForge: open a folder/workspace first.');
      return;
    }
    const workspacePath = workspaceFolders[0].uri.fsPath;
    const backendUrl = vscode.workspace.getConfiguration('releaseforge').get<string>('backendUrl');

    try {
      const token = await ensureAuthenticated();
      if (!token) {
        vscode.window.showWarningMessage('ReleaseForge: sign-in was not completed.');
        return;
      }
      const res = await fetch(`${backendUrl}/ping`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader(token) },
        body: JSON.stringify({ workspace_path: workspacePath }),
      });
      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }
      const data = (await res.json()) as { llm_reply: string; model: string };
      vscode.window.showInformationMessage(`ReleaseForge (${data.model}): ${data.llm_reply}`);
    } catch (err) {
      vscode.window.showErrorMessage(
        `ReleaseForge: failed to reach backend at ${backendUrl}. Is it running? (${err})`,
      );
    }
  });

  context.subscriptions.push(disposable);

  context.subscriptions.push(
    vscode.commands.registerCommand('releaseforge.showDashboard', () => {
      DashboardPanel.show(context.extensionUri);
    }),
  );
}

export function deactivate() {
  stopBackend();
}
