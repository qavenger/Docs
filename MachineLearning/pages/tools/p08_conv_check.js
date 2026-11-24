// p08_conv_check.js — 第 08 篇自定义可微算子的校验：P08.conv 与共享 AG.conv2d 逐元素对照（前向、dX、dW、db，K = 1 / 3 / 5，稀疏梯度），
// 以及所有自定义算子的有限差分梯度校验。用法（在 pages/ 下）：node tools/p08_conv_check.js
const { P08, LM } = require('./p08_proto.js'), { AG } = LM;
const rand = LM.rng(3);
let fails = 0;
const md = (u, v) => { let m = 0, s = 0; for (let i = 0; i < u.length; i++) { m = Math.max(m, Math.abs(u[i] - v[i])); s = Math.max(s, Math.abs(u[i])); } return m / (s || 1); };
function compare(N, C, H, W, Co, Kk, sparse) {
    const X = AG.param('x', N, C * H * W, LM.init.normal(rand, 1)), Wt = AG.param('w', Co, C * Kk * Kk, LM.init.normal(rand, 0.5)), b = AG.param('b', 1, Co, LM.init.normal(rand, 0.3));
    const R = AG.tensor(N, Co * H * W); for (let i = 0; i < R.size; i++) R.data[i] = sparse && rand() < 0.75 ? 0 : rand() - 0.5;
    const run = (f) => { [X, Wt, b].forEach((t) => { t.grad = null; }); AG.clearTape(); const y = f(); AG.backward(AG.sum(AG.mul(y, R))); return [y.data, X.grad, Wt.grad, b.grad].map((a) => Float32Array.from(a)); };
    const A = run(() => AG.conv2d(X, Wt, b, { C, H, W, K: Kk, pad: (Kk - 1) >> 1 })), B = run(() => P08.conv(X, Wt, b, { C, H, W, K: Kk }));
    const e = A.map((a, k) => md(a, B[k])), worst = Math.max(...e), ok = worst < 2e-6; if (!ok) fails++;
    console.log(`${ok ? 'OK  ' : 'FAIL'} conv vs AG.conv2d  N${N} C${C} ${H}x${W} -> ${Co} K${Kk}${sparse ? ' sparse g' : ''}: rel max diff y ${e[0].toExponential(1)} dX ${e[1].toExponential(1)} dW ${e[2].toExponential(1)} db ${e[3].toExponential(1)}`);
}
compare(2, 1, 16, 16, 8, 3, false); compare(3, 8, 8, 8, 16, 3, true); compare(2, 5, 4, 6, 3, 3, false); compare(2, 3, 7, 5, 4, 5, true); compare(2, 6, 4, 4, 1, 1, false); compare(2, 4, 8, 8, 3, 1, true);
/** 有限差分：loss = Σ R ⊙ f(...)²（平方让梯度依赖前向值），float32 上 eps 取 1e-2 */
function fd(name, build, inputs) {
    const R = { t: null };
    const f = () => { const y = build(); if (!R.t) { R.t = AG.tensor(y.r, y.c); for (let i = 0; i < R.t.size; i++) R.t.data[i] = rand() - 0.5; } return AG.sum(AG.mul(AG.square(y), R.t)); };
    inputs.forEach((t) => { t.grad = null; }); AG.clearTape(); AG.backward(f());
    let worst = 0;
    inputs.forEach((t) => { for (let k = 0; k < t.size; k++) { const old = t.data[k], e = 1e-2; t.data[k] = old + e; const lp = AG.noGrad(f).item(); t.data[k] = old - e; const lm = AG.noGrad(f).item(); t.data[k] = old; const num = (lp - lm) / (2 * e), a = t.grad ? t.grad[k] : 0; worst = Math.max(worst, Math.abs(num - a) / Math.max(1e-3, Math.abs(num) + Math.abs(a))); } });
    const ok = worst < 0.05; if (!ok) fails++;
    console.log(`${ok ? 'OK  ' : 'FAIL'} finite differences ${name.padEnd(24)} worst rel err ${worst.toExponential(2)}`);
}
{ const X = AG.param('x', 2, 2 * 30, LM.init.normal(rand, 1)), Wt = AG.param('w', 3, 18, LM.init.normal(rand, 0.5)), b = AG.param('b', 1, 3, LM.init.normal(rand, 0.3)); fd('conv K3', () => P08.conv(X, Wt, b, { C: 2, H: 5, W: 6, K: 3 }), [X, Wt, b]); }
{ const X = AG.param('x', 2, 2 * 20, LM.init.normal(rand, 1)), Wt = AG.param('w', 3, 2, LM.init.normal(rand, 0.5)), b = AG.param('b', 1, 3, LM.init.normal(rand, 0.3)); fd('conv K1', () => P08.conv(X, Wt, b, { C: 2, H: 4, W: 5, K: 1 }), [X, Wt, b]); }
{ const X = AG.param('x', 3, 4 * 9, LM.init.normal(rand, 1)); fd('gap', () => P08.gap(X, 4, 9), [X]); }
{ const X = AG.param('x', 3, 2 * 12, LM.init.normal(rand, 1)), g = AG.param('g', 1, 2, (a) => a.fill(1.3)), bb = AG.param('b', 1, 2, (a) => a.fill(0.2)); fd('batchNorm (train)', () => P08.batchNorm(X, g, bb, { C: 2 }), [X, g, bb]); const st = { mean: Float32Array.from([0.1, -0.2]), var: Float32Array.from([0.8, 1.5]) }; fd('batchNorm (eval)', () => P08.batchNorm(X, g, bb, { C: 2, train: false, stats: st }), [X, g, bb]); }
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
