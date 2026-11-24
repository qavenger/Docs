// p13_difdbg.js — 第 13 篇扩散原型的调试：按时间步看去噪误差、x̂₀ 的质量
const { P13, LM } = require('./p13_proto.js');
const n = 1500, steps = +(process.argv[2] || 2000), B = 16, lr = +(process.argv[3] || 2e-3), hid = +(process.argv[4] || 192);
const tr = P13.difDataset(n, 31), D = P13.DIF.D;
const m = P13.difCreate(3, hid), opt = new LM.AdamW(m.params, { lr, b1: 0.9, b2: 0.999, wd: 0 }), rand = LM.rng(4);
for (let it = 1; it <= steps; it++) P13.difTrainStep(m, opt, tr.X, tr.n, B, rand, LM.cosineLR(it, { maxLR: lr, minLR: lr / 10, warmup: 100, total: steps }));
const r = LM.rng(9);
for (const t of [1, 5, 10, 20, 40, 60, 80, 95, 100]) {
    const nb = 64, Xt = new Float32Array(nb * D), eps = new Float32Array(nb * D), ts = new Float32Array(nb).fill(t), a = m.ab[t];
    for (let b = 0; b < nb; b++) for (let j = 0; j < D; j++) { const e = LM.randn(r); eps[b * D + j] = e; Xt[b * D + j] = Math.sqrt(a) * tr.X[b * D + j] + Math.sqrt(1 - a) * e; }
    const vh = LM.AG.noGrad(() => P13.difForward(m, Xt, ts, nb)).data, ep = vh.map((v, i) => Math.sqrt(1 - a) * Xt[i] + Math.sqrt(a) * v);
    let le = 0, lx = 0, base = 0;
    for (let i = 0; i < nb * D; i++) { le += (ep[i] - eps[i]) ** 2; const x0 = Math.sqrt(a) * Xt[i] - Math.sqrt(1 - a) * vh[i]; lx += (Math.max(-1, Math.min(1, x0)) - tr.X[i]) ** 2; base += tr.X[i] ** 2; }
    console.log(`t=${t} ᾱ=${a.toExponential(2)}: eps mse ${(le / nb / D).toFixed(3)}  x0 rmse ${Math.sqrt(lx / nb / D).toFixed(3)}`);
}
// dataset mean image & variance
let v = 0; const mu = new Float64Array(D); for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) mu[j] += tr.X[i * D + j] / n; for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) v += (tr.X[i * D + j] - mu[j]) ** 2; console.log('data var per dim', (v / n / D).toFixed(3), 'mean', (mu.reduce((a, b) => a + b, 0) / D).toFixed(3));
