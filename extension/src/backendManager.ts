import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

let backendProcess: cp.ChildProcess | undefined;
let outputChannel: vscode.OutputChannel | undefined;
let extensionContext: vscode.ExtensionContext | undefined;
let cachedToken: string | undefined;

const SECRET_KEY = 'releaseforge.authToken';

function getOutputChannel(): vscode.OutputChannel {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel('ReleaseForge Backend');
  }
  return outputChannel;
}

/** Called once from extension.ts's activate() — gives this module access to
 * SecretStorage (the real, VS-Code-managed secure store) for the auth token, rather
 * than a plain settings value anything in the workspace could read. */
export function initAuth(context: vscode.ExtensionContext): void {
  extensionContext = context;
}

/** Called wherever the backend process itself is stopped/restarted — the backend's own
 * authenticated session is deliberately in-memory only (see app/security/oauth.py's
 * module docstring: "a fresh process should not silently trust a stale session"), so a
 * cached token from before a restart is definitely no longer valid there. */
export function invalidateAuth(): void {
  cachedToken = undefined;
}

export function authHeader(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

async function tokenStillValid(backendUrl: string, token: string): Promise<boolean> {
  try {
    const res = await fetch(`${backendUrl}/eval/summary`, { headers: authHeader(token) });
    return res.ok;
  } catch {
    return false;
  }
}

async function runDeviceFlow(backendUrl: string): Promise<string | undefined> {
  const startRes = await fetch(`${backendUrl}/auth/device/start`, { method: 'POST' });
  if (!startRes.ok) {
    vscode.window.showErrorMessage(`ReleaseForge: couldn't start GitHub sign-in (backend returned ${startRes.status}).`);
    return undefined;
  }
  const start = (await startRes.json()) as {
    device_code: string;
    user_code: string;
    verification_uri: string;
    expires_in: number;
    interval: number;
  };

  const action = await vscode.window.showInformationMessage(
    `ReleaseForge needs you to sign in with GitHub. Enter this code: ${start.user_code}`,
    'Open GitHub to sign in',
  );
  if (action === 'Open GitHub to sign in') {
    void vscode.env.openExternal(vscode.Uri.parse(start.verification_uri));
  }

  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: 'Waiting for GitHub sign-in…' },
    async () => {
      const deadline = Date.now() + start.expires_in * 1000;
      while (Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, start.interval * 1000));
        const pollRes = await fetch(`${backendUrl}/auth/device/poll`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_code: start.device_code }),
        });
        if (!pollRes.ok) {
          continue; // a transient error on one poll attempt isn't fatal — keep trying until the deadline
        }
        const result = (await pollRes.json()) as { status: string; token?: string; login?: string; detail?: string };
        if (result.status === 'authenticated' && result.token) {
          vscode.window.showInformationMessage(`ReleaseForge: signed in as ${result.login}.`);
          return result.token;
        }
        if (result.status === 'error') {
          vscode.window.showErrorMessage(`ReleaseForge: GitHub sign-in failed (${result.detail}).`);
          return undefined;
        }
        // "pending" — keep polling.
      }
      vscode.window.showErrorMessage('ReleaseForge: GitHub sign-in timed out.');
      return undefined;
    },
  );
}

/** Returns a real, currently-valid bearer token, running the device flow (with the user
 * directly, live) only when actually needed — not on every call. Returns undefined if
 * the user never completes sign-in; callers must handle that (same "backend couldn't be
 * reached" category of failure the rest of this file already handles). */
export async function ensureAuthenticated(): Promise<string | undefined> {
  const backendUrl = vscode.workspace.getConfiguration('releaseforge').get<string>('backendUrl')!;

  if (cachedToken && (await tokenStillValid(backendUrl, cachedToken))) {
    return cachedToken;
  }

  const stored = await extensionContext?.secrets.get(SECRET_KEY);
  if (stored && (await tokenStillValid(backendUrl, stored))) {
    cachedToken = stored;
    return stored;
  }

  const token = await runDeviceFlow(backendUrl);
  if (!token) {
    return undefined;
  }
  cachedToken = token;
  await extensionContext?.secrets.store(SECRET_KEY, token);
  return token;
}

async function checkHealth(backendUrl: string): Promise<boolean> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1500);
  try {
    const res = await fetch(`${backendUrl}/health`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

function resolvePythonPath(backendDir: string, configuredPythonPath: string): string {
  if (configuredPythonPath) {
    return configuredPythonPath;
  }
  return process.platform === 'win32'
    ? path.join(backendDir, '.venv', 'Scripts', 'python.exe')
    : path.join(backendDir, '.venv', 'bin', 'python');
}

async function promptForBackendDir(): Promise<string | undefined> {
  const choice = await vscode.window.showInformationMessage(
    "ReleaseForge couldn't find its backend — locate the ReleaseForge 'backend' folder (the one containing app/main.py). You'll only need to do this once.",
    'Locate backend folder',
  );
  if (choice !== 'Locate backend folder') {
    return undefined;
  }
  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: false,
    canSelectFolders: true,
    canSelectMany: false,
    openLabel: 'Select ReleaseForge backend folder',
  });
  if (!picked || picked.length === 0) {
    return undefined;
  }
  const dir = picked[0].fsPath;
  await vscode.workspace
    .getConfiguration('releaseforge')
    .update('backendDir', dir, vscode.ConfigurationTarget.Global);
  return dir;
}

/** Starts the backend if it isn't already answering /health. Safe to call repeatedly. */
export async function ensureBackendRunning(): Promise<boolean> {
  const config = vscode.workspace.getConfiguration('releaseforge');
  const backendUrl = config.get<string>('backendUrl')!;

  if (await checkHealth(backendUrl)) {
    return true;
  }

  let backendDir = config.get<string>('backendDir') || '';
  if (!backendDir || !fs.existsSync(path.join(backendDir, 'app', 'main.py'))) {
    const located = await promptForBackendDir();
    if (!located) {
      vscode.window.showWarningMessage(
        'ReleaseForge backend is not running and no location was set. Run "ReleaseForge: Restart Backend" once you have.',
      );
      return false;
    }
    backendDir = located;
  }

  const pythonPath = resolvePythonPath(backendDir, config.get<string>('pythonPath') || '');
  if (!fs.existsSync(pythonPath)) {
    vscode.window.showErrorMessage(
      `ReleaseForge: no Python interpreter at ${pythonPath}. Set "releaseforge.pythonPath" in settings, or create a venv at ${backendDir}/.venv.`,
    );
    return false;
  }

  const port = new URL(backendUrl).port || '8420';
  const channel = getOutputChannel();
  channel.appendLine(`Starting backend: ${pythonPath} -m uvicorn app.main:app --port ${port}`);

  backendProcess = cp.spawn(pythonPath, ['-m', 'uvicorn', 'app.main:app', '--port', port], {
    cwd: backendDir,
    env: process.env,
  });
  backendProcess.stdout?.on('data', (chunk) => channel.append(chunk.toString()));
  backendProcess.stderr?.on('data', (chunk) => channel.append(chunk.toString()));
  backendProcess.on('exit', (code) => {
    channel.appendLine(`\nBackend process exited with code ${code}`);
    backendProcess = undefined;
  });

  return vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: 'Starting ReleaseForge backend…' },
    async () => {
      // 120s, not 30s — found live, not guessed: lifespan spawns four real MCP
      // subprocess servers (filesystem/docker/kubernetes/observability) and does a full
      // handshake with each before /health can ever answer, and on this machine that
      // alone took 45-90s+ during this project's own live testing, before counting doc
      // re-ingestion on top. A real backend that's genuinely still starting up used to
      // get reported as "couldn't be started automatically" well before it was actually
      // done — confirmed directly: the process was alive and /health answered fine only
      // moments after this timeout would have already given up on it.
      for (let attempt = 0; attempt < 120; attempt++) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (await checkHealth(backendUrl)) {
          return true;
        }
      }
      channel.show(true);
      vscode.window.showErrorMessage(
        'ReleaseForge backend did not start in time — see the "ReleaseForge Backend" output channel.',
      );
      return false;
    },
  );
}

export function stopBackend(): void {
  backendProcess?.kill();
  backendProcess = undefined;
  invalidateAuth();
}
