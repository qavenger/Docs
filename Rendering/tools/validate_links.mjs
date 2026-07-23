// Portable-reference validator for the GI document tree.
// Checks HTML href/src, Markdown links, duplicate ids and local fragments.

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const toolsDirectory = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(toolsDirectory, '..');

async function walk(directory) {
  const result = [];
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    if (entry.name.startsWith('.edge-validation')) continue;
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) result.push(...await walk(absolute));
    else result.push(absolute);
  }
  return result;
}

function normalizeTarget(rawValue) {
  return rawValue
    .trim()
    .replace(/^['"]|['"]$/g, '')
    .replace(/&amp;/g, '&');
}

function isExternal(value) {
  return /^(?:https?:|mailto:|tel:|javascript:|data:|blob:|\/\/)/i.test(value);
}

function isAbsoluteLocal(value) {
  return /^(?:file:|[A-Za-z]:[\\/]|\/(?!\/))/i.test(value);
}

function extractHtmlReferences(text) {
  return [...text.matchAll(/\b(?:href|src)\s*=\s*(["'])(.*?)\1/gi)]
    .map((match) => normalizeTarget(match[2]));
}

function extractMarkdownReferences(text) {
  return [...text.matchAll(/!?\[[^\]]*]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/g)]
    .map((match) => normalizeTarget(match[1].replace(/^<|>$/g, '')));
}

function extractIds(text) {
  return [...text.matchAll(/\bid\s*=\s*(["'])(.*?)\1/gi)]
    .map((match) => match[2]);
}

const sourceFiles = (await walk(docsRoot))
  .filter((file) => /\.(?:html?|md)$/i.test(file));
const contentCache = new Map();
const issues = [];
let referenceCount = 0;

async function read(file) {
  if (!contentCache.has(file)) {
    contentCache.set(file, await fs.readFile(file, 'utf8'));
  }
  return contentCache.get(file);
}

for (const sourceFile of sourceFiles) {
  const sourceText = await read(sourceFile);
  const relativeSource = path.relative(docsRoot, sourceFile);
  if (/\.html?$/i.test(sourceFile)) {
    const ids = extractIds(sourceText);
    const duplicates = [...new Set(
      ids.filter((id, index) => ids.indexOf(id) !== index),
    )];
    duplicates.forEach((id) => issues.push({
      type: 'duplicate-id',
      source: relativeSource,
      value: id,
    }));
  }

  const references = /\.html?$/i.test(sourceFile)
    ? extractHtmlReferences(sourceText)
    : extractMarkdownReferences(sourceText);
  for (const reference of references) {
    // Template-literal attributes are checked after rendering by the Edge
    // validator; they are not literal filesystem targets at source level.
    if (!reference || reference.includes('${') || isExternal(reference)) continue;
    ++referenceCount;
    if (isAbsoluteLocal(reference)) {
      issues.push({
        type: 'absolute-local-reference',
        source: relativeSource,
        value: reference,
      });
      continue;
    }
    const hashIndex = reference.indexOf('#');
    const pathPart = (hashIndex >= 0 ? reference.slice(0, hashIndex) : reference)
      .split('?')[0];
    const fragment = hashIndex >= 0
      ? decodeURIComponent(reference.slice(hashIndex + 1))
      : '';
    const targetFile = pathPart
      ? path.resolve(path.dirname(sourceFile), decodeURIComponent(pathPart))
      : sourceFile;
    if (!targetFile.startsWith(docsRoot + path.sep) && targetFile !== docsRoot) {
      issues.push({
        type: 'reference-leaves-package',
        source: relativeSource,
        value: reference,
      });
      continue;
    }
    try {
      const stat = await fs.stat(targetFile);
      if (!stat.isFile() && pathPart) {
        issues.push({
          type: 'target-not-file',
          source: relativeSource,
          value: reference,
        });
        continue;
      }
    } catch {
      issues.push({
        type: 'missing-target',
        source: relativeSource,
        value: reference,
      });
      continue;
    }
    if (fragment && /\.html?$/i.test(targetFile)) {
      const targetIds = new Set(extractIds(await read(targetFile)));
      if (!targetIds.has(fragment)) {
        issues.push({
          type: 'missing-fragment',
          source: relativeSource,
          value: reference,
        });
      }
    }
  }
}

const report = {
  generatedAt: new Date().toISOString(),
  docsRoot,
  sourceFiles: sourceFiles.length,
  localReferences: referenceCount,
  issues,
};
const reportPath = path.join(docsRoot, 'link_validation_report.json');
await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
process.stdout.write(
  `files=${sourceFiles.length} localReferences=${referenceCount} issues=${issues.length}\n`,
);
if (issues.length) process.stdout.write(`${JSON.stringify(issues, null, 2)}\n`);
process.stdout.write(`Report: ${reportPath}\n`);
process.exitCode = issues.length ? 1 : 0;
