import * as vscode from 'vscode';
import { ensureBackendRunning, stopBackend } from './backendManager';
import { ChatViewProvider } from './chatViewProvider';

export function activate(context: vscode.ExtensionContext) {
  const chatProvider = new ChatViewProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, chatProvider),
  );

  void ensureBackendRunning();
  context.subscriptions.push({ dispose: stopBackend });

  context.subscriptions.push(
    vscode.commands.registerCommand('deploygenie.restartBackend', async () => {
      stopBackend();
      await ensureBackendRunning();
    }),
  );

  const disposable = vscode.commands.registerCommand('deploygenie.ping', async () => {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
      vscode.window.showErrorMessage('DeployGenie: open a folder/workspace first.');
      return;
    }
    const workspacePath = workspaceFolders[0].uri.fsPath;
    const backendUrl = vscode.workspace.getConfiguration('deploygenie').get<string>('backendUrl');

    try {
      const res = await fetch(`${backendUrl}/ping`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace_path: workspacePath }),
      });
      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }
      const data = (await res.json()) as { llm_reply: string; model: string };
      vscode.window.showInformationMessage(`DeployGenie (${data.model}): ${data.llm_reply}`);
    } catch (err) {
      vscode.window.showErrorMessage(
        `DeployGenie: failed to reach backend at ${backendUrl}. Is it running? (${err})`,
      );
    }
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {
  stopBackend();
}
