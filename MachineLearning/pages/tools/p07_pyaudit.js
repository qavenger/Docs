// 检查页面里所有 Python 代码块：被调用的函数要么是内置 / 方法 / 模块属性，要么在页面某处有 def / class 定义
const { SPEC } = require('./p07_load.js');
const blocks = [];
const add = (src) => { if (src) blocks.push(String(src)); };
SPEC.steps.forEach((st) => { [].concat(st.code || []).forEach((c) => add(c.src)); (st.blocks || []).forEach((b) => { if (b.type === 'code') add(b.src); }); });
SPEC.graph.common.blocks.forEach((b) => add(b.code));
[].concat(SPEC.library.blocks).forEach((b) => add(b.code));
SPEC.graph.nodes.forEach((n) => { if (n.detail && n.detail.code && n.detail.code.src) add(n.detail.code.src); });
const all = blocks.join('\n');
const defined = new Set([...all.matchAll(/^\s*(?:def|class)\s+([A-Za-z_]\w*)/gm)].map((m) => m[1]));
const builtins = new Set('print len range enumerate zip list dict set tuple int float str max min abs sum round isinstance super sorted map filter any all iter next type getattr setattr'.split(' '));
const calls = new Set();
for (const m of all.matchAll(/(^|[^.\w])([A-Za-z_]\w*)\s*\(/gm)) calls.add(m[2]);
const keywords = new Set('if for while return and or not in lambda with def class elif else'.split(' '));
const missing = [...calls].filter((c) => !defined.has(c) && !builtins.has(c) && !keywords.has(c));
console.log(`${blocks.length} Python blocks · ${defined.size} definitions · ${calls.size} distinct bare calls`);
console.log('bare calls without a definition in the page (should only be locals such as lambdas):', missing);
