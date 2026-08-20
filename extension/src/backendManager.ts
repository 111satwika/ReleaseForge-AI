import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

let backendProcess: cp.ChildProcess | undefined;
let outputChannel: vscode.OutputChannel | undefined;

function getOutputChannel(): vscode.OutputChannel {
  if (!outputChannel) {
    outputChannel = vscode.window.createOutputChannel('DeployGenie Backend');
  }
  return outputChannel;
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
    "DeployGenie couldn't find its backend — locate the DeployGenie 'backend' folder (the one containing app/main.py). You'll only need to do this once.",
    'Locate backend folder',
  );
  if (choice !== 'Locate backend folder') {
    return undefined;
  }
  const picked = await vscode.window.showOpenDialog({
    canSelectFiles: false,
    canSelectFolders: true,
    canSelectMany: false,
    openLabel: 'Select DeployGenie backend folder',
  });
  if (!picked || picked.length === 0) {
    return undefined;
  }
  const dir = picked[0].fsPath;
  await vscode.workspace
    .getConfiguration('deploygenie')
    .update('backendDir', dir, vscode.ConfigurationTarget.Global);
  return dir;
}

/** Starts the backend if it isn't already answering /health. Safe to call repeatedly. */
export async function ensureBackendRunning(): Promise<boolean> {
  const config = vscode.workspace.getConfiguration('deploygenie');
  const backendUrl = config.get<string>('backendUrl')!;

  if (await checkHealth(backendUrl)) {
    return true;
  }

  let backendDir = config.get<string>('backendDir') || '';
  if (!backendDir || !fs.existsSync(path.join(backendDir, 'app', 'main.py'))) {
    const located = await promptForBackendDir();
    if (!located) {
      vscode.window.showWarningMessage(
        'DeployGenie backend is not running and no location was set. Run "DeployGenie: Restart Backend" once you have.',
      );
      return false;
    }
    backendDir = located;
  }

  const pythonPath = resolvePythonPath(backendDir, config.get<string>('pythonPath') || '');
  if (!fs.existsSync(pythonPath)) {
    vscode.window.showErrorMessage(
      `DeployGenie: no Python interpreter at ${pythonPath}. Set "deploygenie.pythonPath" in settings, or create a venv at ${backendDir}/.venv.`,
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
    { location: vscode.ProgressLocation.Notification, title: 'Starting DeployGenie backend…' },
    async () => {
      for (let attempt = 0; attempt < 30; attempt++) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (await checkHealth(backendUrl)) {
          return true;
        }
      }
      channel.show(true);
      vscode.window.showErrorMessage(
        'DeployGenie backend did not start in time — see the "DeployGenie Backend" output channel.',
      );
      return false;
    },
  );
}

export function stopBackend(): void {
  backendProcess?.kill();
  backendProcess = undefined;
}
