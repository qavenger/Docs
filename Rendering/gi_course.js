(() => {
  'use strict';

  const queryAll = (selector, root = document) => [...root.querySelectorAll(selector)];

  function activateWorkbench(workbench, mode, focus = false) {
    const buttons = queryAll('[data-gi-tab]', workbench);
    const panels = queryAll('[data-gi-panel]', workbench);
    if (!buttons.some((button) => button.dataset.giTab === mode)) {
      mode = buttons[0]?.dataset.giTab;
    }
    buttons.forEach((button) => {
      const active = button.dataset.giTab === mode;
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      if (active && focus) button.focus();
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.giPanel !== mode;
    });
    workbench.dataset.giActive = mode ?? '';
  }

  function mountWorkbenches(root = document) {
    queryAll('[data-gi-workbench]', root).forEach((workbench, workbenchIndex) => {
      if (workbench.dataset.giMounted === 'true') return;
      workbench.dataset.giMounted = 'true';
      const buttons = queryAll('[data-gi-tab]', workbench);
      const panels = queryAll('[data-gi-panel]', workbench);
      const tabList = workbench.querySelector('.gi-tabs');
      if (tabList) tabList.setAttribute('role', 'tablist');
      buttons.forEach((button, index) => {
        const mode = button.dataset.giTab;
        const panel = panels.find((item) => item.dataset.giPanel === mode);
        const tabId = `gi-tab-${workbenchIndex}-${index}`;
        const panelId = `gi-panel-${workbenchIndex}-${index}`;
        button.id ||= tabId;
        button.type = 'button';
        button.setAttribute('role', 'tab');
        if (panel) {
          panel.id ||= panelId;
          panel.setAttribute('role', 'tabpanel');
          panel.setAttribute('aria-labelledby', button.id);
          button.setAttribute('aria-controls', panel.id);
        }
        button.addEventListener('click', () => activateWorkbench(workbench, mode));
        button.addEventListener('keydown', (event) => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
          event.preventDefault();
          const current = buttons.indexOf(button);
          let next = current;
          if (event.key === 'ArrowLeft') next = (current - 1 + buttons.length) % buttons.length;
          if (event.key === 'ArrowRight') next = (current + 1) % buttons.length;
          if (event.key === 'Home') next = 0;
          if (event.key === 'End') next = buttons.length - 1;
          activateWorkbench(workbench, buttons[next].dataset.giTab, true);
        });
      });
      activateWorkbench(workbench, workbench.dataset.giDefault || buttons[0]?.dataset.giTab);
    });
  }

  function normalizeIndent(source) {
    const lines = source.replace(/\r\n?/g, '\n').replace(/^\n+|\n+$/g, '').split('\n');
    const populated = lines.filter((line) => line.trim());
    const indent = populated.length
      ? Math.min(...populated.map((line) => line.match(/^\s*/)[0].length))
      : 0;
    return lines.map((line) => line.slice(indent)).join('\n');
  }

  // Lightweight formatter for compact educational JS functions. It is not a
  // general parser: it only inserts layout whitespace and never changes tokens.
  function formatCompactSource(input) {
    const source = String(input ?? '').trim();
    if (!source || source.includes('\n')) return normalizeIndent(source);
    let output = '';
    let indent = 0;
    let quote = '';
    let escaped = false;
    let lineComment = false;
    let blockComment = false;
    let parenDepth = 0;
    let bracketDepth = 0;
    const unit = '  ';
    const newline = () => {
      output = output.replace(/[ \t]+$/g, '');
      if (!output.endsWith('\n')) output += '\n';
      output += unit.repeat(Math.max(0, indent));
    };
    for (let i = 0; i < source.length; i += 1) {
      const char = source[i];
      const next = source[i + 1] ?? '';
      if (lineComment) {
        output += char;
        if (char === '\n') {
          lineComment = false;
          output += unit.repeat(indent);
        }
        continue;
      }
      if (blockComment) {
        output += char;
        if (char === '*' && next === '/') {
          output += next;
          i += 1;
          blockComment = false;
        }
        continue;
      }
      if (quote) {
        output += char;
        if (escaped) escaped = false;
        else if (char === '\\') escaped = true;
        else if (char === quote) quote = '';
        continue;
      }
      if (char === '/' && next === '/') {
        output += '//';
        i += 1;
        lineComment = true;
        continue;
      }
      if (char === '/' && next === '*') {
        output += '/*';
        i += 1;
        blockComment = true;
        continue;
      }
      if (char === '"' || char === "'" || char === '`') {
        quote = char;
        output += char;
        continue;
      }
      if (char === '(') parenDepth += 1;
      if (char === ')') parenDepth = Math.max(0, parenDepth - 1);
      if (char === '[') bracketDepth += 1;
      if (char === ']') bracketDepth = Math.max(0, bracketDepth - 1);
      if (char === '{') {
        output = output.replace(/[ \t]+$/g, '');
        output += ' {';
        indent += 1;
        newline();
        continue;
      }
      if (char === '}') {
        indent = Math.max(0, indent - 1);
        newline();
        output += '}';
        const tail = source.slice(i + 1).match(/^\s*(else|catch|finally)\b/);
        if (tail) output += ' ';
        else if (!/^\s*[;,)\]]/.test(source.slice(i + 1))) newline();
        continue;
      }
      if (char === ';' && parenDepth === 0 && bracketDepth === 0) {
        output += ';';
        newline();
        continue;
      }
      if (char === ',' && parenDepth === 0 && bracketDepth === 0) {
        output += ',';
        newline();
        continue;
      }
      if (/\s/.test(char)) {
        if (!/[ \n]$/.test(output)) output += ' ';
        continue;
      }
      output += char;
    }
    return output
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n[ \t]+\n/g, '\n\n')
      .replace(/\{\s+\{/g, '{ {')
      .trim();
  }

  async function copyText(text) {
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch {
        // file:// or enterprise policies may expose Clipboard API but reject
        // writes. Fall through to the local selection-based path.
      }
    }
    const area = document.createElement('textarea');
    area.value = text;
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.append(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }

  function decorateCode(code) {
    if (code.dataset.giCodeMounted === 'true') return;
    code.dataset.giCodeMounted = 'true';
    let source = code.textContent;
    if (code.dataset.giPretty === 'compact') source = formatCompactSource(source);
    else source = normalizeIndent(source);
    code.textContent = '';
    const pre = code.closest('pre') ?? code;
    pre.classList.add('gi-readable-code');
    source.split('\n').forEach((line, index) => {
      const row = document.createElement('span');
      row.className = 'gi-code-line';
      row.dataset.line = String(index + 1);
      const text = document.createElement('span');
      text.className = 'gi-code-line-text';
      text.textContent = line || ' ';
      row.append(text);
      code.append(row);
    });

    const shell = document.createElement('div');
    shell.className = 'gi-code-shell';
    pre.parentNode.insertBefore(shell, pre);
    const toolbar = document.createElement('div');
    toolbar.className = 'gi-code-toolbar';
    const badge = document.createElement('span');
    const status = code.dataset.giStatus || 'actual';
    badge.className = `gi-code-badge ${status}`;
    badge.textContent = {
      actual: '当前 Demo 实际执行',
      pseudo: '等价伪代码',
      native: 'DX12 / HLSL 对应实现',
      excerpt: '节选 · 非完整实现',
    }[status] || status;
    const path = document.createElement('span');
    path.className = 'gi-code-path';
    path.textContent = code.dataset.giSource || '内联教学代码';
    const wrap = document.createElement('button');
    wrap.className = 'gi-code-action';
    wrap.type = 'button';
    wrap.textContent = '自动换行';
    wrap.addEventListener('click', () => {
      const enabled = pre.classList.toggle('is-wrapped');
      wrap.textContent = enabled ? '保持缩进' : '自动换行';
    });
    const copy = document.createElement('button');
    copy.className = 'gi-code-action';
    copy.type = 'button';
    copy.textContent = '复制';
    copy.addEventListener('click', async () => {
      await copyText(source);
      copy.textContent = '已复制';
      setTimeout(() => { copy.textContent = '复制'; }, 1200);
    });
    toolbar.append(badge, path, wrap, copy);
    shell.append(toolbar, pre);
  }

  function mountCodeBlocks(root = document) {
    queryAll('code[data-gi-code]', root).forEach(decorateCode);
  }

  function mountSymbolLinks(root = document) {
    queryAll('[data-gi-workbench]', root).forEach((workbench) => {
      queryAll('[data-gi-symbol]', workbench).forEach((symbol) => {
        if (symbol.dataset.giSymbolMounted === 'true') return;
        symbol.dataset.giSymbolMounted = 'true';
        symbol.addEventListener('mouseenter', () => {
          queryAll(`[data-gi-symbol="${CSS.escape(symbol.dataset.giSymbol)}"]`, workbench)
            .forEach((item) => item.classList.add('is-highlighted'));
        });
        symbol.addEventListener('mouseleave', () => {
          queryAll('.is-highlighted', workbench).forEach((item) => item.classList.remove('is-highlighted'));
        });
      });
    });
  }

  function setCheck(container, key, state, detail) {
    const row = container.querySelector(`[data-gi-check="${CSS.escape(key)}"]`);
    if (!row) return;
    row.classList.remove('pass', 'fail', 'warn');
    row.classList.add(state);
    const detailNode = row.querySelector('[data-gi-check-detail]');
    if (detailNode) detailNode.textContent = detail;
  }

  function mount(root = document) {
    mountWorkbenches(root);
    mountCodeBlocks(root);
    mountSymbolLinks(root);
  }

  window.GICourse = {
    mount,
    activateWorkbench,
    formatCompactSource,
    decorateCode,
    setCheck,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => mount(), { once: true });
  } else {
    mount();
  }
})();
