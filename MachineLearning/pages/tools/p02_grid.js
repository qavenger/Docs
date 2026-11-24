// p02_grid.js — N=4000 时各种激活 / 初始化 / 学习率的训练结果（定默认值用）
const { P02, LM } = require('./p02_proto.js');
const rand = LM.rng(20251123);
const t0 = Date.now();
const tr = P02.makeSet(4000, rand), te = P02.makeSet(500, rand);
console.log('gen ms', Date.now() - t0);
const all = (n) => Int32Array.from({ length: n }, (_, i) => i);
function run(label, extra, steps) {
    const cfg = Object.assign({ sizes: [256, 64, 10], act: 'relu', init: 'he', seed: 7, opt: { kind: 'sgd', lr: 0.1 }, X: tr.X, y: tr.y, idx: all(tr.n), B: 32, cap: steps,
        evalSets: [{ name: 'train', X: tr.X, y: tr.y, idx: all(500) }, { name: 'test', X: te.X, y: te.y, idx: all(te.n), conf: true }] }, extra);
    const R = P02.makeRun(cfg); P02.evalNow(R);
    const ts = Date.now(); let fr = 0;
    while (R.step < R.cap && !R.diverged) { P02.tick(R, 2, 100); fr++; }
    const H = R.hist, n = H.step.length, at = (s) => { let k = H.step.findIndex((v) => v >= s); if (k < 0) k = n - 1; return `${(100 * H.trainA[k]).toFixed(0)}/${(100 * H.testA[k]).toFixed(1)}`; };
    console.log(`${label.padEnd(28)} ${((Date.now() - ts) / fr).toFixed(2)}ms/fr  @300 ${at(300)} @1000 ${at(1000)} @3000 ${at(3000)}  testL ${H.testL[n - 1].toFixed(3)}${R.diverged ? ' DIVERGED' : ''}`);
}
const mode = process.argv[2] || 'lr';
if (mode === 'lr') for (const lr of [0.01, 0.03, 0.1, 0.3, 1]) { run(`softmax lr${lr}`, { sizes: [256, 10], init: 'zero', opt: { kind: 'sgd', lr } }, 3000); run(`mlp64 relu he lr${lr}`, { opt: { kind: 'sgd', lr } }, 3000); }
if (mode === 'act') for (const act of ['sigmoid', 'tanh', 'relu', 'gelu']) for (const init of ['small', 'xavier', 'he', 'large']) run(`${act} ${init}`, { act, init }, 3000);
if (mode === 'hid') for (const h of [16, 32, 64, 128]) run(`relu he H${h}`, { sizes: [256, h, 10] }, 3000);
