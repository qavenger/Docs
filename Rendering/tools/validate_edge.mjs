// Dependency-free Microsoft Edge file:// smoke validator for this GI course.
// Usage:
//   node tools/validate_edge.mjs
//   node tools/validate_edge.mjs gi_module_rcgi.html RenderGraph_GI_Lab/index.html

import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(scriptDirectory, '..');
const edgePath = process.env.EDGE_PATH
  || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const defaultPages = [
  'index.html',
  'Realtime_GI_Field_Guide.html',
  'gi_module_foundations.html',
  'gi_module_rsm.html',
  'gi_module_ssao_ssgi.html',
  'gi_module_sh.html',
  'gi_module_voxelization.html',
  'gi_module_cone_tracing.html',
  'gi_module_probes.html',
  'gi_module_prt_probes.html',
  'gi_module_rcgi.html',
  'gi_module_voxel_gi.html',
  'gi_module_ddgi.html',
  'DX12_GI_Lab/index.html',
  'RenderGraph_GI_Lab/index.html',
];
const pages = process.argv.slice(2).length
  ? process.argv.slice(2)
  : defaultPages;
const validationBase = path.join(docsRoot, '.edge-validation');
const profileRoot = path.join(
  validationBase,
  `profile-${process.pid}-${Date.now()}`,
);
const port = 9400 + Math.floor(Math.random() * 350);

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function pollJson(url, timeoutMilliseconds = 15000) {
  const deadline = Date.now() + timeoutMilliseconds;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
    } catch (error) {
      lastError = error;
    }
    await delay(100);
  }
  throw new Error(`CDP timeout for ${url}: ${lastError?.message ?? 'no response'}`);
}

class CDP {
  constructor(url) {
    this.socket = new WebSocket(url);
    this.sequence = 0;
    this.pending = new Map();
    this.listeners = new Map();
    this.opened = new Promise((resolve, reject) => {
      this.socket.addEventListener('open', resolve, { once: true });
      this.socket.addEventListener('error', reject, { once: true });
    });
    this.socket.addEventListener('message', (event) => {
      const message = JSON.parse(String(event.data));
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result);
        return;
      }
      const callbacks = this.listeners.get(message.method) ?? [];
      callbacks.forEach((callback) => callback(message.params));
    });
  }

  async call(method, params = {}) {
    await this.opened;
    const id = ++this.sequence;
    const response = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.socket.send(JSON.stringify({ id, method, params }));
    return response;
  }

  on(method, callback) {
    const callbacks = this.listeners.get(method) ?? [];
    callbacks.push(callback);
    this.listeners.set(method, callbacks);
  }

  close() {
    this.socket.close();
  }
}

async function waitForDocument(cdp, timeoutMilliseconds = 20000) {
  const deadline = Date.now() + timeoutMilliseconds;
  while (Date.now() < deadline) {
    const result = await cdp.call('Runtime.evaluate', {
      expression: 'document.readyState',
      returnByValue: true,
    });
    if (result.result.value === 'complete') {
      await delay(900);
      return;
    }
    await delay(100);
  }
  throw new Error('document.readyState did not reach complete');
}

const auditExpression = String.raw`
(() => {
  const canvases = [...document.querySelectorAll('canvas')].map((canvas) => {
    let dataLength = 0;
    let serializationError = '';
    try {
      dataLength = canvas.toDataURL('image/png').length;
    } catch (error) {
      serializationError = String(error);
    }
    return {
      id: canvas.id || '(anonymous)',
      width: canvas.width,
      height: canvas.height,
      dataLength,
      serializationError,
    };
  });
  const workbenchIssues = [...document.querySelectorAll('[data-gi-workbench]')]
    .map((workbench, index) => {
      const panels = [...workbench.querySelectorAll('[data-gi-panel]')];
      const tabs = [...workbench.querySelectorAll('[data-gi-tab]')];
      return {
        index,
        mounted: workbench.dataset.giMounted === 'true',
        visiblePanels: panels.filter((panel) => !panel.hidden).length,
        selectedTabs: tabs.filter((tab) => tab.getAttribute('aria-selected') === 'true').length,
      };
    })
    .filter((item) => !item.mounted || item.visiblePanels !== 1 || item.selectedTabs !== 1);
  const longSingleLineCode = [...document.querySelectorAll('code[data-gi-code]')]
    .filter((code) => {
      const rows = code.querySelectorAll('.gi-code-line');
      return rows.length <= 1 && code.textContent.trim().length > 120;
    })
    .map((code) => ({
      source: code.dataset.giSource || '(unspecified)',
      length: code.textContent.trim().length,
    }));
  const unpairedFormulaBlocks = [
    ...document.querySelectorAll('.gi-formula, .math, .equation, .formula-inline'),
  ].filter((formula) => !formula.closest('[data-gi-workbench]'))
    .map((formula) => ({
      className: formula.className,
      excerpt: formula.textContent.trim().slice(0, 100),
    }));
  const brokenImages = [...document.images]
    .filter((image) => image.complete && image.naturalWidth === 0)
    .map((image) => image.getAttribute('src'));
  const absoluteLocalReferences = [
    ...document.querySelectorAll('[href], [src]'),
  ].map((element) => element.getAttribute('href') ?? element.getAttribute('src'))
    .filter((value) => value && /^(?:file:|[A-Za-z]:[\\/]|\/(?!\/))/.test(value));
  return {
    title: document.title,
    readyState: document.readyState,
    bodyTextLength: document.body?.innerText.length ?? 0,
    canvases,
    workbenches: document.querySelectorAll('[data-gi-workbench]').length,
    readableCodeBlocks: document.querySelectorAll('.gi-code-shell').length,
    formulaBlocks: document.querySelectorAll('.gi-formula').length,
    lightCards: document.querySelectorAll('.gi-light-card').length,
    workbenchIssues,
    longSingleLineCode,
    unpairedFormulaBlocks,
    brokenImages,
    absoluteLocalReferences,
    failedChecks: [...document.querySelectorAll('.gi-check.fail')]
      .map((item) => item.textContent.trim()),
    warningChecks: [...document.querySelectorAll('.gi-check.warn')]
      .map((item) => item.textContent.trim()),
  };
})()
`;

await fs.mkdir(profileRoot, { recursive: true });
const edge = spawn(edgePath, [
  '--headless=new',
  '--no-first-run',
  '--disable-features=msEdgeFirstRunExperience',
  '--allow-file-access-from-files',
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profileRoot}`,
  'about:blank',
], {
  windowsHide: true,
  stdio: ['ignore', 'ignore', 'ignore'],
});

let browserCdp;
let pageCdp;
let failed = false;
try {
  const browserInfo = await pollJson(`http://127.0.0.1:${port}/json/version`);
  const targets = await pollJson(`http://127.0.0.1:${port}/json/list`);
  const pageTarget = targets.find((target) => target.type === 'page');
  if (!pageTarget) throw new Error('Edge did not expose a page target');
  browserCdp = new CDP(browserInfo.webSocketDebuggerUrl);
  pageCdp = new CDP(pageTarget.webSocketDebuggerUrl);
  await pageCdp.call('Page.enable');
  await pageCdp.call('Runtime.enable');
  await pageCdp.call('Log.enable');

  let runtimeExceptions = [];
  let errorLogs = [];
  pageCdp.on('Runtime.exceptionThrown', (params) => {
    runtimeExceptions.push(params.exceptionDetails?.text ?? 'runtime exception');
  });
  pageCdp.on('Log.entryAdded', ({ entry }) => {
    if (entry.level === 'error') errorLogs.push(entry.text);
  });

  const reports = [];
  for (const relativePage of pages) {
    const absolutePage = path.resolve(docsRoot, relativePage);
    if (!absolutePage.startsWith(docsRoot + path.sep)) {
      throw new Error(`page leaves docs root: ${relativePage}`);
    }
    await fs.access(absolutePage);
    runtimeExceptions = [];
    errorLogs = [];
    await pageCdp.call('Page.navigate', {
      url: pathToFileURL(absolutePage).href,
    });
    await waitForDocument(pageCdp);
    const evaluation = await pageCdp.call('Runtime.evaluate', {
      expression: auditExpression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (evaluation.exceptionDetails) {
      throw new Error(
        evaluation.exceptionDetails.exception?.description
        ?? evaluation.exceptionDetails.text,
      );
    }
    const audit = evaluation.result.value;
    const canvasFailures = audit.canvases.filter(
      (canvas) => canvas.width < 1
        || canvas.height < 1
        || canvas.dataLength < 200
        || canvas.serializationError,
    );
    const pageFailed = runtimeExceptions.length > 0
      || errorLogs.length > 0
      || audit.workbenchIssues.length > 0
      || audit.longSingleLineCode.length > 0
      || audit.unpairedFormulaBlocks.length > 0
      || audit.brokenImages.length > 0
      || audit.absoluteLocalReferences.length > 0
      || audit.failedChecks.length > 0
      || canvasFailures.length > 0;
    failed ||= pageFailed;
    reports.push({
      page: relativePage,
      status: pageFailed ? 'FAIL' : 'PASS',
      runtimeExceptions,
      errorLogs,
      canvasFailures,
      ...audit,
    });
    process.stdout.write(
      `${pageFailed ? 'FAIL' : 'PASS'} ${relativePage}`
      + ` | canvas=${audit.canvases.length}`
      + ` workbench=${audit.workbenches}`
      + ` code=${audit.readableCodeBlocks}`
      + ` formula=${audit.formulaBlocks}`
      + ` light=${audit.lightCards}\n`,
    );
    if (pageFailed) {
      process.stdout.write(`${JSON.stringify(reports.at(-1), null, 2)}\n`);
    }
  }
  const reportPath = path.join(docsRoot, 'edge_validation_report.json');
  await fs.writeFile(reportPath, `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    edgePath,
    reports,
  }, null, 2)}\n`, 'utf8');
  process.stdout.write(`Report: ${reportPath}\n`);
} finally {
  try {
    if (browserCdp) await browserCdp.call('Browser.close');
  } catch {
    edge.kill();
  }
  pageCdp?.close();
  browserCdp?.close();
  await delay(250);
  const resolvedBase = path.resolve(validationBase);
  const resolvedProfile = path.resolve(profileRoot);
  if (!resolvedProfile.startsWith(resolvedBase + path.sep)) {
    throw new Error(`unsafe profile cleanup target: ${resolvedProfile}`);
  }
  await fs.rm(resolvedProfile, { recursive: true, force: true });
}

process.exitCode = failed ? 1 : 0;
