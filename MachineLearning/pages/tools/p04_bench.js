// p04_bench.js — 第 04 篇的早期测速：数据生成、AG 版 MLP 自编码器 / conv 的每步耗时
const LM = require('../llm_common.js');
const { AG } = LM;
const t0 = Date.now();
const rand = LM.rng(20251123);
const tr = LM.digits.makeSet(4000, rand), te = LM.digits.makeSet(500, rand);
console.log('data', Date.now() - t0, 'ms');
let nz = 0; for (let i = 0; i < tr.X.length; i++) if (tr.X[i] > 0) nz++;
console.log('nonzero frac', (nz / tr.X.length).toFixed(3));

function bceLogits(Z, X) {   // mean over rows of Σ_j softplus(z) − x·z
    const n = Z.r, m = Z.c, z = Z.data, x = X.data, Y = AG.tensor(1, 1);
    let s = 0; for (let i = 0; i < z.length; i++) { const v = z[i]; s += (v > 0 ? v + Math.log1p(Math.exp(-v)) : Math.log1p(Math.exp(v))) - x[i] * v; }
    Y.data[0] = s / n;
    return AG.track(Y, [Z], (g) => { const gz = Z.g(), k = g[0] / n; for (let i = 0; i < z.length; i++) gz[i] += k * (1 / (1 + Math.exp(-z[i])) - x[i]); });
}
function mlpAE(sizes, seed) {
    const r = LM.rng(seed), P = [];
    for (let l = 0; l + 1 < sizes.length; l++) { P.push(AG.param('W' + l, sizes[l], sizes[l + 1], LM.init.he(r, sizes[l]))); P.push(AG.param('b' + l, 1, sizes[l + 1])); }
    return P;
}
function fwd(P, X, nl, bott) {
    let h = X;
    for (let l = 0; l < nl; l++) { h = AG.linear(h, P[2 * l], P[2 * l + 1]); if (l !== nl - 1 && l !== bott) h = AG.relu(h); }
    return h;
}
function bench(sizes, B, steps) {
    const P = mlpAE(sizes, 1), opt = new LM.AdamW(P, { lr: 1e-3, b2: 0.999, wd: 0 });
    const nl = sizes.length - 1, bott = (nl / 2 - 1) | 0;
    const Xb = AG.tensor(B, 256), r = LM.rng(5);
    let ts = 0, loss = 0;
    for (let s = 0; s < steps; s++) {
        for (let k = 0; k < B; k++) { const i = Math.floor(r() * tr.n); Xb.data.set(tr.X.subarray(i * 256, (i + 1) * 256), k * 256); }
        const t1 = process.hrtime.bigint();
        opt.zeroGrad();
        const Z = fwd(P, Xb, nl, bott);
        const L = bceLogits(Z, Xb);
        AG.backward(L); opt.step();
        ts += Number(process.hrtime.bigint() - t1) / 1e6;
        loss = 0.98 * loss + 0.02 * L.item();
    }
    console.log(sizes.join('-'), 'B', B, (ts / steps).toFixed(2), 'ms/step', 'loss~', loss.toFixed(1));
}
bench([256, 64, 2, 64, 256], 32, 300);
bench([256, 128, 2, 128, 256], 32, 300);
bench([256, 128, 32, 2, 32, 128, 256], 32, 300);
bench([256, 128, 32, 2, 32, 128, 256], 64, 300);
bench([256, 64, 16, 2, 16, 64, 256], 64, 300);
// conv speed
{
    const B = 32, X = AG.tensor(B, 256); X.data.set(tr.X.subarray(0, B * 256));
    const r = LM.rng(3);
    const W1 = AG.param('w', 16, 1 * 16, LM.init.he(r, 16)), b1 = AG.param('b', 1, 16);
    const W2 = AG.param('w2', 8, 16 * 9, LM.init.he(r, 144)), b2 = AG.param('b2', 1, 8);
    let t = 0;
    for (let s = 0; s < 50; s++) {
        const t1 = process.hrtime.bigint();
        const h = AG.relu(AG.conv2d(X, W1, b1, { C: 1, H: 16, W: 16, K: 4, stride: 2, pad: 1 }));
        const y = AG.conv2d(h, W2, b2, { C: 16, H: 8, W: 8, K: 3, pad: 1 });
        const L = AG.mean(AG.square(y)); AG.backward(L);
        t += Number(process.hrtime.bigint() - t1) / 1e6;
    }
    const macs = B * (64 * 16 * 16 + 64 * 8 * 144);
    console.log('conv enc test', (t / 50).toFixed(2), 'ms', 'fwd MACs', macs, '→', (3 * macs / (t / 50) / 1e6).toFixed(2), 'GMAC/s (fwd+bwd≈3x)');
}
