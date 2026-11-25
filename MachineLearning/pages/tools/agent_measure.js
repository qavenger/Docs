// agent_measure.js — 在 Node 里执行第 16–18 篇页面头部（P16 / P17 / P18 命名空间），用页面自己的评测函数跑完全部 bench，打印结果。
// 讲义 notes 里引用的数字都来自这里（与浏览器里实时跑出的完全相同：同样的种子、同样的代码）。
// 用法（在 pages/ 下）：node tools/agent_measure.js 18 [benchName…]
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const pages = path.join(__dirname, '..');
const no = process.argv[2] || '18';
const only = process.argv.slice(3);
const file = fs.readdirSync(pages).find((f) => f.startsWith(no + '_') && f.endsWith('.lab.js'));
if (!file) throw new Error('page ' + no);
let spec = null;
const stub = new Proxy({}, { get: () => () => ({}) });
const PL = { mount(s) { spec = s; }, draw: stub, mesh: stub, scene: stub, esc: (s) => String(s) };
const ctx = { console, Math, Float32Array, Float64Array, Int32Array, Int8Array, Uint8Array, Uint16Array, Uint32Array, Int16Array, ArrayBuffer, DataView, Array, Object, Map, Set, JSON, Number, String, Date, Promise, Proxy, Symbol, Error, RegExp, isFinite, isNaN, parseInt, parseFloat, atob, btoa, Buffer, performance, TextDecoder, TextEncoder, setTimeout, clearTimeout };
ctx.window = ctx; ctx.PipelineLab = PL; ctx.PL = PL;
vm.createContext(ctx);
const srcs = ['llm_common.js', 'agent_common.js', 'agent_cpp.js', 'agent_weights.js', file].map((f) => fs.readFileSync(path.join(pages, f), 'utf8'));
const code = srcs.join('\n;\n').replace(/\bconst (LM|AGT|AGT_CPP|AGT_W|P16|P17|P18) =/g, 'var $1 =');
vm.runInContext(code, ctx, { filename: file });
const NS = ctx['P' + no];
const M = NS.loadModel();
const benches = Object.keys(NS).filter((k) => /Bench$/.test(k) && (!only.length || only.includes(k)));
const short = (v) => JSON.stringify(v, (k, x) => (typeof x === 'number' ? +x.toFixed(4) : x));
for (const k of benches) {
    const t0 = Date.now();
    const B = NS[k](M);
    while (!B.done) ctx.AGT.benchWork(B, 1e6);
    const sec = (Date.now() - t0) / 1000;
    const out = Object.assign({}, B);
    delete out.M; delete out.jobs; delete out.cur; delete out.done; delete out.k;
    console.log(`\n■ ${k}: ${B.jobs.length} jobs, ${B.forwards} forwards, ${sec.toFixed(1)} s (${(1000 * sec / Math.max(1, B.forwards)).toFixed(3)} ms / forward)`);
    console.log(short(out));
}
