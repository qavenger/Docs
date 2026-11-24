// check_cpp.js — 把一篇讲义里的全部 C++ 代码块拼成一个翻译单元，用 clang++ 做语法 / 类型检查（-fsyntax-only）。
// 用法（在 pages/ 下）：node tools/check_cpp.js 04_transformer.lab.js [--keep]
// 规则：
//   · 顺序 = graph.common（跳过 LM.CPP 库本身，改为 #include 全部 cpp/minml_*.h）→ library → 各步骤代码（按页面顺序）→ 节点详情代码。
//     所以「先定义后使用」：步骤代码里用到的辅助函数，要么在更早的代码块里，要么放进 graph.common / library。
//   · 完全相同的代码块只编译一次（节点详情常常复用步骤代码）。
//   · 不是 C++ 的块（lang 为 hlsl 等）跳过；故意写成片段 / 伪代码的 C++ 块，在第一行写 // @nocompile 跳过。
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const os = require('os');
const { execFileSync } = require('child_process');

const pages = path.join(__dirname, '..');
const file = process.argv[2];
if (!file) { console.error('usage: node tools/check_cpp.js <page.lab.js> [--keep]'); process.exit(2); }
const keep = process.argv.includes('--keep');
const CLANG = process.env.CLANGXX || 'C:/Program Files/Microsoft Visual Studio/18/Professional/VC/Tools/Llvm/x64/bin/clang++.exe';

// 在 Node 里执行页面源码，截获 PipelineLab.mount(spec)
let spec = null;
const stubDraw = new Proxy({}, { get: () => () => ({}) });
const PLstub = {
    mount(s) { spec = s; }, rng: null, gauss2: null, gauss: null, clamp: (x, a, b) => (x < a ? a : x > b ? b : x), lerp: (a, b, t) => a + (b - a) * t,
    smoothstep: (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); }, fract: (x) => x - Math.floor(x),
    buf: (o, n, len, T) => { T = T || Float32Array; if (!o[n] || o[n].length !== len) o[n] = new T(len); return o[n]; },
    v3: { add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s], dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], len: (a) => Math.hypot(a[0], a[1], a[2]), norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; } },
    fft: { fft1d() {}, rows() {}, cols() {}, fft2d() {}, freq: (i, n) => (i < n / 2 ? i : i - n), shift: (x, y, W, H) => ((y + (H >> 1)) % H) * W + ((x + (W >> 1)) % W) },
    cmap: new Proxy({}, { get: () => (t, out) => out }), draw: stubDraw, mesh: stubDraw, scene: stubDraw, hexToRgb: () => [0, 0, 0], esc: (s) => String(s),
};
const sources = ['llm_common.js'];
if (fs.existsSync(path.join(pages, 'llm_weights.js'))) sources.push('llm_weights.js');
const pageSrc = fs.readFileSync(path.join(pages, file), 'utf8');
const ctx = { console, Math, Float32Array, Float64Array, Int32Array, Int8Array, Uint8Array, Uint16Array, Uint32Array, Int16Array, Array, Object, Map, Set, JSON, Number, String, Date, Promise, Proxy, Symbol, isFinite, isNaN, parseInt, parseFloat, setTimeout, clearTimeout };
ctx.window = ctx; ctx.PipelineLab = PLstub; ctx.PL = PLstub;
vm.createContext(ctx);
const combined = sources.map((f) => fs.readFileSync(path.join(pages, f), 'utf8')).join('\n;\n') + '\n;\n' + pageSrc;
vm.runInContext(combined.replace(/\bconst (LM|LM_WEIGHTS) =/g, 'var $1 ='), ctx, { filename: file });
const LMx = ctx.LM;
PLstub.rng = LMx.rng; PLstub.gauss = (r) => LMx.randn(r); PLstub.gauss2 = (r) => [LMx.randn(r), LMx.randn(r)];
if (!spec) { console.error('page did not call PipelineLab.mount'); process.exit(2); }

// 收集代码块
const blocks = [];
const lib = new Set(Object.values(LMx.CPP || {}).filter((v) => typeof v === 'string'));
const push = (where, src, lang) => {
    if (src == null) return;
    lang = String(lang || spec.codeLang || 'hlsl').toLowerCase();
    if (!['cpp', 'c', 'c++'].includes(lang)) return;
    const text = String(src);
    if (lib.has(text)) return;                                   // 库本身：用 #include
    if (/^\s*\/\/\s*@nocompile/.test(text)) return;
    if (blocks.some((b) => b.text === text)) return;
    blocks.push({ where, text });
};
const codeOf = (c, where) => { if (!c) return; (Array.isArray(c) ? c : [c]).forEach((k, i) => push(`${where}${Array.isArray(c) ? '#' + i : ''} · ${k.title || ''}`, k.src, k.lang)); };
const common = spec.graph && spec.graph.common;
if (common) (common.blocks || []).forEach((b, i) => push(`common#${i} ${b.label || ''}`, b.code, b.lang));
const library = Array.isArray(spec.library) ? { blocks: spec.library } : spec.library;
if (library) (library.blocks || []).forEach((b, i) => push(`library#${i} ${b.title || ''}`, b.code || b.src, b.lang));
(spec.steps || []).forEach((st, si) => {
    codeOf(st.code, `step ${String(si + 1).padStart(2, '0')} ${st.id}`);
    (st.blocks || []).forEach((b, bi) => { if (b.type === 'code') push(`step ${String(si + 1).padStart(2, '0')} ${st.id} block#${bi} · ${b.title || ''}`, b.src, b.lang); });
});
((spec.graph && spec.graph.nodes) || []).forEach((n) => { if (n.detail && n.detail.code) codeOf(n.detail.code, `node ${n.id}`); });

// 拼成一个翻译单元：每块前面加 #line 注释，报错时能对回代码块
const dedent = (s) => { const ls = s.replace(/\t/g, '    ').replace(/^\s*\n/, '').replace(/\s+$/, '').split('\n'); const ind = Math.min(...ls.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length)); return isFinite(ind) ? ls.map((l) => l.slice(ind)).join('\n') : ls.join('\n'); };
let tu = '#include "minml_gpt.h"\n#include <array>\n#include <map>\n#include <set>\n#include <numeric>\n#include <cstring>\n#include <cassert>\n#include <climits>\n#include <cfloat>\n#include <utility>\n#include <tuple>\n#include <unordered_map>\n#include <queue>\n#include <deque>\n#include <random>\n#include <chrono>\n#include <complex>\n';
blocks.forEach((b, i) => { tu += `\n// ═══ [${i}] ${b.where} ═══\n#line 1 "block${i}"\n${dedent(b.text)}\n`; });
const tmp = path.join(os.tmpdir(), `check_cpp_${path.basename(file, '.lab.js')}.cpp`);
fs.writeFileSync(tmp, tu);
console.log(`${blocks.length} C++ blocks → ${tmp}`);
try {
    execFileSync(CLANG, ['-std=c++17', '-fsyntax-only', '-Wall', '-Wno-unused-variable', '-Wno-unused-function', '-Wno-unused-but-set-variable', '-ferror-limit=50', '-I', path.join(pages, 'cpp'), tmp], { stdio: 'pipe' });
    console.log('✔ all C++ blocks compile');
} catch (e) {
    const out = String(e.stderr || e.stdout || e.message);
    // 把 blockN:行号 翻译成代码块的位置描述
    console.log(out.replace(/block(\d+):(\d+):(\d+)/g, (_, i, l, c) => `[${blocks[+i] ? blocks[+i].where : '?'} ] line ${l}:${c}`).split('\n').slice(0, 120).join('\n'));
    console.log('✖ C++ compile errors');
    process.exitCode = 1;
}
if (!keep && process.exitCode !== 1) fs.rmSync(tmp, { force: true });
