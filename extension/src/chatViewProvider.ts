import * as vscode from 'vscode';
import { ensureBackendRunning } from './backendManager';
import { OriginalContentProvider } from './diffProvider';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ProposedDiff {
  target_file: string;
  original_content: string | null;
  new_content: string;
}

export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'deploygenie.chatView';

  private view?: vscode.WebviewView;
  private history: ChatMessage[] = [];

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly originalContentProvider: OriginalContentProvider,
  ) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this.getHtml(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(async (message) => {
      if (message.type === 'ready') {
        this.postRepoInfo();
      } else if (message.type === 'send') {
        await this.handleSend(message.text as string);
      }
    });
  }

  private postRepoInfo(): void {
    const folders = vscode.workspace.workspaceFolders;
    const repoName = folders && folders.length > 0 ? folders[0].name : undefined;
    this.view?.webview.postMessage({ type: 'repo', name: repoName });
  }

  private async handleSend(text: string): Promise<void> {
    if (!text || !text.trim()) {
      return;
    }

    this.history.push({ role: 'user', content: text });
    this.view?.webview.postMessage({ type: 'message', role: 'user', content: text });

    const folders = vscode.workspace.workspaceFolders;
    if (!folders || folders.length === 0) {
      this.view?.webview.postMessage({
        type: 'message',
        role: 'assistant',
        content: 'Open a folder or workspace first — DeployGenie needs a repo to work with.',
      });
      return;
    }

    const workspacePath = folders[0].uri.fsPath;
    const backendUrl = vscode.workspace.getConfiguration('deploygenie').get<string>('backendUrl');

    this.view?.webview.postMessage({ type: 'thinking', value: true });
    try {
      const backendUp = await ensureBackendRunning();
      if (!backendUp) {
        this.view?.webview.postMessage({
          type: 'message',
          role: 'assistant',
          content: "DeployGenie's backend isn't running and couldn't be started automatically. Run \"DeployGenie: Restart Backend\" from the Command Palette once it's set up.",
        });
        return;
      }

      const res = await fetch(`${backendUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace_path: workspacePath, messages: this.history }),
      });
      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }
      const data = (await res.json()) as { reply: string; model: string; diff?: ProposedDiff | null };
      this.history.push({ role: 'assistant', content: data.reply });
      this.view?.webview.postMessage({
        type: 'message',
        role: 'assistant',
        content: data.reply,
        model: data.model,
      });

      if (data.diff) {
        await this.showDiff(data.diff, workspacePath);
      }
    } catch (err) {
      this.view?.webview.postMessage({
        type: 'message',
        role: 'assistant',
        content: `Couldn't reach the DeployGenie backend at ${backendUrl}. (${err})`,
      });
    } finally {
      this.view?.webview.postMessage({ type: 'thinking', value: false });
    }
  }

  private async showDiff(diff: ProposedDiff, workspacePath: string): Promise<void> {
    const leftUri = this.originalContentProvider.setContent(
      diff.target_file,
      diff.original_content ?? '',
    );
    const rightUri = vscode.Uri.joinPath(vscode.Uri.file(workspacePath), diff.target_file);
    const title = diff.original_content === null
      ? `DeployGenie: new file ${diff.target_file}`
      : `DeployGenie: proposed change to ${diff.target_file}`;

    await vscode.commands.executeCommand('vscode.diff', leftUri, rightUri, title);
  }

  private getHtml(webview: vscode.Webview): string {
    const nonce = getNonce();
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
  body {
    font-family: var(--vscode-font-family);
    color: var(--vscode-foreground);
    background: var(--vscode-sideBar-background);
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    height: 100vh;
    box-sizing: border-box;
  }
  #repo {
    padding: 8px 12px;
    font-size: 0.85em;
    color: var(--vscode-descriptionForeground);
    border-bottom: 1px solid var(--vscode-panel-border);
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }
  #repo .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--vscode-charts-green, #3fb950);
    flex-shrink: 0;
  }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  #empty {
    color: var(--vscode-descriptionForeground);
    font-size: 0.9em;
  }
  .msg {
    font-size: 0.92em;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    align-self: flex-end;
    background: var(--vscode-badge-background);
    color: var(--vscode-badge-foreground);
    padding: 6px 10px;
    border-radius: 8px;
    max-width: 85%;
  }
  .msg.assistant {
    align-self: flex-start;
    max-width: 92%;
  }
  .msg.assistant .model-tag {
    display: block;
    font-size: 0.75em;
    color: var(--vscode-descriptionForeground);
    margin-top: 4px;
  }
  #thinking {
    display: none;
    padding: 0 12px 6px;
    font-size: 0.85em;
    color: var(--vscode-descriptionForeground);
    flex-shrink: 0;
  }
  #thinking.active { display: block; }
  #inputRow {
    display: flex;
    gap: 6px;
    padding: 8px;
    border-top: 1px solid var(--vscode-panel-border);
    flex-shrink: 0;
  }
  #input {
    flex: 1;
    resize: none;
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, transparent);
    border-radius: 4px;
    padding: 6px 8px;
    font-family: inherit;
    font-size: 0.9em;
    min-height: 32px;
    max-height: 120px;
  }
  #input:focus { outline: 1px solid var(--vscode-focusBorder); }
  #send {
    background: var(--vscode-button-background);
    color: var(--vscode-button-foreground);
    border: none;
    border-radius: 4px;
    padding: 0 12px;
    cursor: pointer;
  }
  #send:hover { background: var(--vscode-button-hoverBackground); }
</style>
</head>
<body>
  <div id="repo"><span class="dot"></span><span id="repoName">No repo open</span></div>
  <div id="messages"><div id="empty">Describe a change and DeployGenie will edit a file, run your tests, and show you a diff to review before anything is committed or pushed.</div></div>
  <div id="thinking">DeployGenie is thinking…</div>
  <div id="inputRow">
    <textarea id="input" placeholder="Message DeployGenie…" rows="1"></textarea>
    <button id="send">Send</button>
  </div>
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('input');
  const sendEl = document.getElementById('send');
  const thinkingEl = document.getElementById('thinking');
  const repoNameEl = document.getElementById('repoName');
  let emptyEl = document.getElementById('empty');

  function addMessage(role, content, model) {
    if (emptyEl) { emptyEl.remove(); emptyEl = null; }
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = content;
    if (role === 'assistant' && model) {
      const tag = document.createElement('span');
      tag.className = 'model-tag';
      tag.textContent = model;
      div.appendChild(tag);
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function send() {
    const text = inputEl.value;
    if (!text.trim()) { return; }
    vscode.postMessage({ type: 'send', text });
    inputEl.value = '';
    inputEl.style.height = 'auto';
  }

  sendEl.addEventListener('click', send);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  });

  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (msg.type === 'message') {
      addMessage(msg.role, msg.content, msg.model);
    } else if (msg.type === 'thinking') {
      thinkingEl.classList.toggle('active', msg.value);
    } else if (msg.type === 'repo') {
      repoNameEl.textContent = msg.name ? msg.name : 'No repo open';
    }
  });

  vscode.postMessage({ type: 'ready' });
</script>
</body>
</html>`;
  }
}

function getNonce(): string {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}
