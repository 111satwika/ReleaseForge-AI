import * as vscode from 'vscode';

export const DEPLOYGENIE_DIFF_SCHEME = 'deploygenie-original';

/**
 * Serves the "before" content of a proposed change as a read-only virtual document,
 * so VS Code's diff editor can show before/after without a real file for the "before" side —
 * the real file on disk already has the "after" content written to it.
 */
export class OriginalContentProvider implements vscode.TextDocumentContentProvider {
  private readonly content = new Map<string, string>();

  private readonly changeEmitter = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.changeEmitter.event;

  setContent(relativePath: string, content: string): vscode.Uri {
    const uri = vscode.Uri.parse(`${DEPLOYGENIE_DIFF_SCHEME}:${relativePath}`);
    this.content.set(uri.toString(), content);
    this.changeEmitter.fire(uri);
    return uri;
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.content.get(uri.toString()) ?? '';
  }
}
