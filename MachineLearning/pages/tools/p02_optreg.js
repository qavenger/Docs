// p02_optreg.js — 第 09 步（优化器对比）与第 11 步（小数据上的正则化）的原型
const { P02, LM } = require('./p02_proto.js');
const rand = LM.rng(20251123);
const tr = P02.makeSet(4000, rand), te = P02.makeSet(500, rand);
const all = (n) => Int32Array.from({ length: n }, (_, i) => i);
const range = (a, b) => Int32Array.from({ length: b - a }, (_, i) => a + i);
const mode = process.argv[2] || 'opt';
function go(label, extra, steps, sets) {
    const cfg = Object.assign({ sizes: [256, 64, 10], act: 'relu', init: 'he', seed: 7, opt: { kind: 'sgd', lr: 0.1 }, X: tr.X, y: tr.y, idx: all(tr.n), B: 32, cap: steps, evalSets: sets }, extra);
    const R = P02.makeRun(cfg); P02.evalNow(R);
    while (R.step < R.cap && !R.diverged) P02.tick(R, 2, 100);
    return R;
}
if (mode === 'opt') {
    const sets = [{ name: 'train', X: tr.X, y: tr.y, idx: all(500) }, { name: 'test', X: te.X, y: te.y, idx: all(te.n) }];
    for (const [lab, opt] of [['sgd 0.1', { kind: 'sgd', lr: 0.1 }], ['mom 0.1 μ.9', { kind: 'momentum', lr: 0.1, mu: 0.9 }], ['mom 0.01 μ.9', { kind: 'momentum', lr: 0.01, mu: 0.9 }], ['adam 1e-3', { kind: 'adam', lr: 1e-3 }], ['adam 3e-3', { kind: 'adam', lr: 3e-3 }], ['sgd 0.03', { kind: 'sgd', lr: 0.03 }], ['mom 0.03', { kind: 'momentum', lr: 0.03 }]]) {
        const R = go(lab, { opt }, 1500, sets), H = R.hist, ema = LM.ema(R.batchLoss, 0.98);
        const at = (s) => { const k = Math.max(0, H.step.findIndex((v) => v >= s)); return `${(100 * H.testA[k]).toFixed(1)}`; };
        console.log(`${lab.padEnd(14)} ema loss @100 ${ema[99].toFixed(3)} @300 ${ema[299].toFixed(3)} @1000 ${ema[999].toFixed(3)} @1500 ${ema[1499].toFixed(3)} | test acc @100 ${at(100)} @300 ${at(300)} @1500 ${at(1500)}${R.diverged ? ' DIVERGED' : ''}`);
    }
}
if (mode === 'reg') {
    const N = +process.argv[3] || 200;
    const sets = [{ name: 'train', X: tr.X, y: tr.y, idx: range(0, N) }, { name: 'val', X: tr.X, y: tr.y, idx: range(N, N + 500) }, { name: 'test', X: te.X, y: te.y, idx: all(te.n) }];
    const cfgs = [['none', {}], ['wd 1e-2', { wd: 1e-2 }], ['wd 3e-2', { wd: 3e-2 }], ['wd 1e-1', { wd: 1e-1 }], ['drop .5', { dropout: 0.5 }], ['wd3e-2+d.5', { wd: 3e-2, dropout: 0.5 }]];
    for (const [lab, ex] of cfgs) {
        const R = go(lab, Object.assign({ idx: range(0, N) }, ex), 2500, sets), H = R.hist, n = H.step.length;
        let bk = 0; for (let k = 1; k < n; k++) if (H.valL[k] < H.valL[bk]) bk = k;
        let ba = 0; for (let k = 1; k < n; k++) if (H.valA[k] > H.valA[ba]) ba = k;
        console.log(`N=${N} ${lab.padEnd(9)} final tr ${(100 * H.trainA[n - 1]).toFixed(0)} val ${(100 * H.valA[n - 1]).toFixed(1)} (L ${H.valL[n - 1].toFixed(3)}) test ${(100 * H.testA[n - 1]).toFixed(1)} | min valL ${H.valL[bk].toFixed(3)} @${H.step[bk]} (val ${(100 * H.valA[bk]).toFixed(1)}, test ${(100 * H.testA[bk]).toFixed(1)}) | best valA ${(100 * H.valA[ba]).toFixed(1)} @${H.step[ba]}`);
    }
}
