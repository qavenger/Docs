// p02_proto.js — 第 02 篇的 Node 原型：直接加载 02_mlp_backprop.lab.js 里的 P02 命名空间（与页面同一份代码），
// 调数据难度、学习率，测每步耗时，检查准确率。用法：node pages/tools/p02_proto.js [what]
const fs = require('fs');
const path = require('path');
const LM = require('../llm_common.js');
const src = fs.readFileSync(path.join(__dirname, '../02_mlp_backprop.lab.js'), 'utf8');
const head = src.slice(0, src.indexOf('PipelineLab.mount('));
const P02 = new Function('LM', head + '\nreturn P02;')(LM);
module.exports = { P02, LM };
if (require.main !== module) return;

const what = process.argv[2] || 'basic';
const seed = 20251123;
const t0 = Date.now();
const rand = LM.rng(seed);
const tr = P02.makeSet(P02.NTRAIN, rand), te = P02.makeSet(P02.NTEST, rand);
console.log(`data: ${Date.now() - t0} ms`);
let nz = 0; for (let i = 0; i < tr.X.length; i++) if (tr.X[i] === 0) nz++;
console.log(`zero fraction ${(nz / tr.X.length).toFixed(3)}, mean ||x||² ${(() => { let s = 0; for (let i = 0; i < tr.X.length; i++) s += tr.X[i] * tr.X[i]; return (s / tr.n).toFixed(2); })()}`);

function ascii(X, i) {
    const ch = ' .:-=+*#%@';
    let s = '';
    for (let y = 0; y < 16; y++) { for (let x = 0; x < 16; x++) s += ch[Math.min(9, Math.floor(X[i * 256 + y * 16 + x] * 10))]; s += '\n'; }
    return s;
}
if (what === 'show') {
    for (let d = 0; d < 10; d++) {
        const idx = []; for (let i = 0; i < tr.n && idx.length < 4; i++) if (tr.y[i] === d) idx.push(i);
        const rows = idx.map((i) => ascii(tr.X, i).split('\n'));
        for (let r = 0; r < 16; r++) console.log(rows.map((rr) => rr[r]).join(' | '));
        console.log('-'.repeat(70));
    }
    process.exit(0);
}

const all = (n) => Int32Array.from({ length: n }, (_, i) => i);
function run(label, cfgExtra, steps, spf) {
    const cfg = Object.assign({
        sizes: [256, 64, 10], act: 'relu', init: 'he', seed: 7, opt: { kind: 'sgd', lr: 0.1 }, wd: 0, dropout: 0,
        X: tr.X, y: tr.y, idx: all(tr.n), B: 32, cap: steps,
        evalSets: [{ name: 'train', X: tr.X, y: tr.y, idx: all(tr.n) }, { name: 'test', X: te.X, y: te.y, idx: all(te.n), conf: true }],
    }, cfgExtra);
    const R = P02.makeRun(cfg);
    P02.evalNow(R);
    const ts = Date.now();
    let frames = 0;
    while (R.step < R.cap && !R.diverged) { P02.tick(R, spf || 2, 100); frames++; }
    const ms = (Date.now() - ts) / frames;
    const H = R.hist, n = H.step.length;
    const pick = [0, Math.floor(n / 8), Math.floor(n / 4), Math.floor(n / 2), n - 1];
    console.log(`${label.padEnd(34)} ${ms.toFixed(2)} ms/frame | ` + pick.map((k) => `s${H.step[k]}: tr ${(100 * H.trainA[k]).toFixed(1)} te ${(100 * H.testA[k]).toFixed(1)} (L ${H.testL[k].toFixed(2)})`).join(' · ') + (R.diverged ? ' DIVERGED' : ''));
    return R;
}
if (what === 'basic') {
    run('softmax lr0.1', { sizes: [256, 10], init: 'zero' }, 1500);
    run('softmax lr0.3', { sizes: [256, 10], init: 'zero', opt: { kind: 'sgd', lr: 0.3 } }, 1500);
    run('softmax lr1', { sizes: [256, 10], init: 'zero', opt: { kind: 'sgd', lr: 1 } }, 1500);
    run('mlp64 relu he lr0.1', {}, 3000);
    run('mlp64 relu he lr0.3', { opt: { kind: 'sgd', lr: 0.3 } }, 3000);
}
if (what === 'acts') {
    for (const act of ['sigmoid', 'tanh', 'relu', 'gelu']) for (const init of ['small', 'xavier', 'he', 'large']) run(`mlp64 ${act} ${init} lr0.1`, { act, init }, 2000);
}
if (what === 'hidden') {
    for (const H of [16, 32, 64, 128]) run(`mlp${H} relu he lr0.1`, { sizes: [256, H, 10] }, 3000);
}
if (what === 'lr') {
    for (const lr of [0.01, 0.03, 0.1, 0.3, 1]) { run(`softmax lr${lr}`, { sizes: [256, 10], init: 'zero', opt: { kind: 'sgd', lr } }, 1500); run(`mlp64 lr${lr}`, { opt: { kind: 'sgd', lr } }, 3000); }
}
