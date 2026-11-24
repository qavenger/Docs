// p13_dif.js — 第 13 篇第 07 步原型：log-mel → PCA 潜空间（16 维、白化）→ 潜空间 DDPM（v-prediction）→ DDIM 采样 → 解码 → Griffin-Lim
const { P13, LM } = require('./p13_proto.js');
const n = +(process.argv[2] || 1500), steps = +(process.argv[3] || 3000), B = +(process.argv[4] || 64), lr = +(process.argv[5] || 2e-3), hid = +(process.argv[6] || 128);
let t0 = P13.now();
const tr = P13.difDataset(n, 31), te = P13.difDataset(200, 777), D = P13.DIF.D, K = P13.DIF.K;
console.log(`dataset ${n}: ${(P13.now() - t0).toFixed(0)} ms`);
t0 = P13.now(); const pca = P13.difPCA(tr.X, n, D, K); console.log(`PCA ${(P13.now() - t0).toFixed(0)} ms, explained ${(100 * pca.explained).toFixed(1)}%, λ ${Array.from(pca.lam.slice(0, 6), (v) => v.toFixed(2)).join(' ')} …`);
const Z = P13.difEncode(pca, tr.X, n), recTE = P13.difDecode(pca, P13.difEncode(pca, te.X, 200), 200);
const rmse = (A, Bx, cnt) => { let s = 0; for (let i = 0; i < cnt * D; i++) s += (A[i] - Bx[i]) ** 2; return Math.sqrt(s / cnt / D); };
console.log('PCA recon rmse (test)', rmse(recTE, te.X, 200).toFixed(3));
const m = P13.difCreate(3, hid, K), opt = new LM.AdamW(m.params, { lr, b1: 0.9, b2: 0.999, wd: 0 }), rand = LM.rng(4);
t0 = P13.now(); let acc = 0;
for (let it = 1; it <= steps; it++) {
    acc += P13.difTrainStep(m, opt, Z, n, B, rand, LM.cosineLR(it, { maxLR: lr, minLR: lr / 10, warmup: 100, total: steps }));
    if (it % 500 === 0) { console.log(`it ${it}: loss ${(acc / 500).toFixed(4)} (${((P13.now() - t0) / it).toFixed(2)} ms/step)`); acc = 0; }
}
t0 = P13.now(); const smp = P13.difSample(m, 32, 25, LM.rng(8)); console.log(`sample 32 × 25: ${(P13.now() - t0).toFixed(1)} ms`);
const imgs = P13.difDecode(pca, smp.x, 32);
const nn = (x, o, set, cnt) => { let best = Infinity, bi = -1; for (let i = 0; i < cnt; i++) { let s = 0; for (let j = 0; j < D; j++) s += (x[o + j] - set[i * D + j]) ** 2; if (s < best) { best = s; bi = i; } } return { d: Math.sqrt(best / D), i: bi }; };
const recTR = P13.difDecode(pca, Z, n);
const dS = [], dT = []; for (let i = 0; i < 32; i++) dS.push(nn(imgs, i * D, recTR, n).d); for (let i = 0; i < 32; i++) dT.push(nn(recTE, i * D, recTR, n).d);
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
console.log(`NN dist (PCA space decoded): samples mean ${mean(dS).toFixed(3)} | real test (PCA recon) ${mean(dT).toFixed(3)}`);
// 白化潜变量的统计：样本 vs 数据
const zs = (Zx, cnt) => { let m1 = 0, m2 = 0; for (let i = 0; i < cnt * K; i++) { m1 += Zx[i]; m2 += Zx[i] ** 2; } return [(m1 / cnt / K).toFixed(3), Math.sqrt(m2 / cnt / K).toFixed(3)]; };
console.log('latent mean/rms data', zs(Z, n), 'samples', zs(smp.x, 32));
const ascii = (x, o) => { const ch = ' .:-=+*#%@'; let s = ''; for (let mm = P13.DIF.NM - 1; mm >= 0; mm--) { for (let tt = 0; tt < P13.DIF.TF; tt++) s += ch[Math.max(0, Math.min(9, Math.floor((x[o + mm * P13.DIF.TF + tt] + 1) * 5)))]; s += '\n'; } return s; };
for (let i = 0; i < 3; i++) { const a = ascii(imgs, i * D).split('\n'), b = ascii(recTR, nn(imgs, i * D, recTR, n).i * D).split('\n'); console.log(a.map((l, k) => l + '   |nn|   ' + b[k]).join('\n')); }
