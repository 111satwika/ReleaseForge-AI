#!/usr/bin/env node
const { execSync } = require('child_process');
const path = require('path');
const pkg = require('../package.json');

const root = path.resolve(__dirname, '..');
const vsixName = `${pkg.name}-${pkg.version}.vsix`;

function run(command) {
  console.log(`$ ${command}`);
  execSync(command, { cwd: root, stdio: 'inherit' });
}

run('npx --yes @vscode/vsce package');
run(`code --install-extension "${vsixName}" --force`);

console.log(
  `\nInstalled ${vsixName}. Reload any open VS Code windows ` +
    '(Ctrl+Shift+P -> "Developer: Reload Window") to pick it up.',
);
