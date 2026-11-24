// p11_latent.js — 潜空间扩散原型：数字自编码器（256 → H → Z）+ 潜空间里的类别条件 ε 模型 + CFG；kNN 分类器评估
const fs = require('fs'), path = require('path');
const { LM, P11 } = require('./p11_load.js');
const { AG } = LM;
const args = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const H = args.H || 64, Z = args.Z || 8, AESTEPS = args.ae || 4000, LDSTEPS = args.ld || 20000, LDW = args.W || 64;
const rand = LM.rng(2024), train = LM.digits.makeSet(5000, rand), test = LM.digits.makeSet(500, rand);
// ── 自编码器 ──
const r = LM.rng(7), params = [];
const P = (n, a, b) => { const t = AG.param(n, a, b, LM.init.uniform(r, 1 / Math.sqrt(a))); params.push(t); return t; };
const ae = { W1: P('e1', 256, H), b1: P('eb1', 1, H), W2: P('e2', H, Z), b2: P('eb2', 1, Z), W3: P('d1', Z, H), b3: P('db1', 1, H), W4: P('d2', H, 256), b4: P('db2', 1, 256) };
const enc = (X) => AG.linear(AG.silu(AG.linear(X, ae.W1, ae.b1)), ae.W2, ae.b2);
const dec = (z) => AG.sigmoid(AG.linear(AG.silu(AG.linear(z, ae.W3, ae.b3)), ae.W4, ae.b4));
const opt = new LM.AdamW(params, { lr: 2e-3, b2: 0.999, wd: 0 }), B = 64, Xb = AG.tensor(B, 256);
let t0 = performance.now();
for (let s = 0; s < AESTEPS; s++) {
    for (let b = 0; b < B; b++) { const j = Math.floor(r() * train.n); Xb.data.set(train.X.subarray(j * 256, (j + 1) * 256), b * 256); }
    opt.zeroGrad(); const z = enc(Xb); const loss = AG.add(AG.mse(dec(z), Xb), AG.scale(AG.mean(AG.square(z)), 1e-3)); AG.backward(loss);
    opt.step(LM.cosineLR(s, { maxLR: 2e-3, minLR: 1e-4, warmup: 100, total: AESTEPS }));
}
const encodeAll = (S) => AG.noGrad(() => enc(AG.tensor(S.n, 256, S.X)).data);
const decodeZ = (z, n) => AG.noGrad(() => dec(AG.tensor(n, Z, z)).data);
const zTr = encodeAll(train), zTe = encodeAll(test), recTe = decodeZ(zTe, test.n);
let mse = 0; for (let i = 0; i < recTe.length; i++) mse += (recTe[i] - test.X[i]) ** 2 / recTe.length;
console.log(`AE 256→${H}→${Z}: ${AESTEPS} steps ${((performance.now() - t0) / AESTEPS).toFixed(1)} ms/step, test MSE ${mse.toFixed(4)}, params ${params.reduce((a, p) => a + p.size, 0)}`);
// 标准化潜变量（每维减均值除以标准差，相当于 Stable Diffusion 的 scale factor）
const mu = new Float32Array(Z), sd = new Float32Array(Z);
for (let i = 0; i < train.n; i++) for (let d = 0; d < Z; d++) mu[d] += zTr[i * Z + d] / train.n;
for (let i = 0; i < train.n; i++) for (let d = 0; d < Z; d++) sd[d] += (zTr[i * Z + d] - mu[d]) ** 2 / train.n;
for (let d = 0; d < Z; d++) sd[d] = Math.sqrt(sd[d]);
const norm = (zz, n) => { const o = new Float32Array(n * Z); for (let i = 0; i < n; i++) for (let d = 0; d < Z; d++) o[i * Z + d] = (zz[i * Z + d] - mu[d]) / sd[d]; return o; };
const den = (zz, n) => { const o = new Float32Array(n * Z); for (let i = 0; i < n; i++) for (let d = 0; d < Z; d++) o[i * Z + d] = zz[i * Z + d] * sd[d] + mu[d]; return o; };
const zn = norm(zTr, train.n), zne = norm(zTe, test.n);
// kNN（k = 5）分类器：在标准化潜空间里
const knn = (q, n) => { const pred = new Int32Array(n); for (let i = 0; i < n; i++) { const best = []; for (let j = 0; j < train.n; j++) { let d = 0; for (let k = 0; k < Z; k++) d += (q[i * Z + k] - zn[j * Z + k]) ** 2; if (best.length < 5 || d < best[4][0]) { best.push([d, train.y[j]]); best.sort((a, b) => a[0] - b[0]); if (best.length > 5) best.pop(); } } const v = new Int32Array(10); best.forEach((b) => v[b[1]]++); pred[i] = LM.argmax(v); } return pred; };
const pTe = knn(zne, test.n); let acc = 0; for (let i = 0; i < test.n; i++) if (pTe[i] === test.y[i]) acc++;
console.log('kNN on real test latents:', (100 * acc / test.n).toFixed(1) + '%');
// ── 潜空间扩散 ──
const sch = P11.SCHED.cosine;
const R = P11.trainer({ kind: 'vp', sched: sch, data: { X: zn, lab: train.y, n: train.n }, din: Z, width: LDW, depth: 3, ncls: 10, pUncond: 0.2, B: 128, lr: 2e-3, steps: LDSTEPS, seed: 3, warmup: 200, ff: 0, ema: 0.999 });
t0 = performance.now(); for (let s = 0; s < LDSTEPS; s++) R.step();
console.log(`LDM W=${LDW}: ${LDSTEPS} steps ${((performance.now() - t0) / LDSTEPS).toFixed(1)} ms/step, params ${P11.numParams(R.net)}, final loss ${LM.mean(R.lossHist.slice(-500)).toFixed(4)}`);
const ascii = (img) => { const ch = ' .:-=+*#%@'; const rows = []; for (let y = 0; y < 16; y++) { let s = ''; for (let x = 0; x < 16; x++) s += ch[Math.min(9, Math.floor(img[y * 16 + x] * 10))]; rows.push(s); } return rows; };
for (const s of [0, 1, 2, 3, 5]) {
    let accS = 0, tot = 0; const show = [];
    let pd = 0, np = 0;
    for (let c = 0; c < 10; c++) {
        const n = 20, x = new Float32Array(n * Z), rr = LM.rng(100 + c); for (let i = 0; i < n * Z; i++) x[i] = LM.randn(rr);
        P11.sampleVP(P11.epsNet(R.ema, { cls: c, guide: s }), sch, x, n, { K: 50, eta: 0, rand: LM.rng(9), clip: 4, din: Z });
        const pr = knn(x, n); for (let i = 0; i < n; i++) { if (pr[i] === c) accS++; tot++; }
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let d = 0; for (let k = 0; k < Z; k++) d += (x[i * Z + k] - x[j * Z + k]) ** 2; pd += Math.sqrt(d); np++; }
        if (c < 5) show.push(ascii(decodeZ(den(x.subarray(0, Z), 1), 1)));
    }
    console.log(`s=${s}: kNN acc ${(100 * accS / tot).toFixed(1)}%  mean pairwise latent dist (same class) ${(pd / np).toFixed(3)}`);
    if (s === 1 || s === 3) for (let y = 0; y < 16; y++) console.log(show.map((a) => a[y]).join(' | '));
}
// real data pairwise distance per class
let pd = 0, np = 0; for (let c = 0; c < 10; c++) { const idx = []; for (let i = 0; i < train.n && idx.length < 20; i++) if (train.y[i] === c) idx.push(i); for (let a = 0; a < idx.length; a++) for (let b = a + 1; b < idx.length; b++) { let d = 0; for (let k = 0; k < Z; k++) d += (zn[idx[a] * Z + k] - zn[idx[b] * Z + k]) ** 2; pd += Math.sqrt(d); np++; } }
console.log('real same-class pairwise latent dist:', (pd / np).toFixed(3));
if (args.save) fs.writeFileSync(path.join(__dirname, 'p11_ckpt', 'latent.json'), JSON.stringify({ H, Z, ae: LM.gpt.exportWeights({ params }), mu: Array.from(mu), sd: Array.from(sd), ldm: P11.exportNet(R.ema), ldmCfg: R.net.cfg, knnAcc: acc / test.n, mse }));
