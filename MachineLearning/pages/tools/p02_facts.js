// p02_facts.js — 页面里引用的数值事实：GELU tanh 近似的误差；默认样本（Softmax 判错、MLP 判对）
const { P02, LM } = require('./p02_proto.js');
// erf via high-precision series / continued fraction (double)
function erf(x) { const s = Math.sign(x); x = Math.abs(x); if (x < 3) { let sum = x, term = x, k = 0; while (Math.abs(term) > 1e-17 * Math.abs(sum)) { k++; term *= -x * x / k; sum += term / (2 * k + 1); } return s * 2 / Math.sqrt(Math.PI) * sum; } let f = 0; for (let n = 60; n >= 1; n--) f = n / 2 / (x + f); return s * (1 - Math.exp(-x * x) / Math.sqrt(Math.PI) / (x + f)); }
let worst = 0, at = 0; for (let x = -8; x <= 8; x += 1e-3) { const e = 0.5 * x * (1 + erf(x / Math.SQRT2)), a = P02.act1('gelu', x); if (Math.abs(e - a) > worst) { worst = Math.abs(e - a); at = x; } }
console.log(`max |GELU_tanh − GELU_erf| = ${worst.toExponential(3)} at x = ${at.toFixed(3)}`);
// default sample: train both models at default settings (same seeds as the page will use) and list disagreements
const rand = LM.rng(20251123);
const tr = P02.makeSet(P02.NTRAIN, rand), te = P02.makeSet(P02.NTEST, rand);
const all = (n) => Int32Array.from({ length: n }, (_, i) => i);
function train(sizes, init, seed) {
    const R = P02.makeRun({ sizes, act: 'relu', init, seed, opt: { kind: 'sgd', lr: 0.1 }, X: tr.X, y: tr.y, idx: all(tr.n), B: 32, cap: 3000,
        evalSets: [{ name: 'train', X: tr.X, y: tr.y, idx: all(500) }, { name: 'test', X: te.X, y: te.y, idx: all(te.n), conf: true }] });
    P02.evalNow(R); while (R.step < R.cap) P02.tick(R, 4, 100); return R;
}
const S = train([256, 10], 'zero', 1), M = train([256, 64, 10], 'he', 2);
console.log('softmax test', S.hist.testA.at(-1), 'mlp test', M.hist.testA.at(-1));
const ps = S.pred[1], pm = M.pred[1], cand = [];
for (let i = 0; i < te.n; i++) if (ps[i] !== te.y[i] && pm[i] === te.y[i]) cand.push(`${i}:${te.y[i]}→${ps[i]}`);
console.log('softmax wrong, mlp right:', cand.slice(0, 40).join(' '));
console.log('first 20 test labels', Array.from(te.y.slice(0, 20)).join(','));
