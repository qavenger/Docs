// p09_proto.js — 第 09 篇的 Node 原型：加载 09_latent_vae.lab.js 里的 P09（与页面同一份代码），训练 AE / VAE / VQ-VAE，
// 打印指标与耗时，并把潜空间散点、网格解码、重建、采样写成 PNG（tools/shots09/proto/）。用法：node pages/tools/p09_proto.js [what] [key=value ...]
const fs = require('fs'), path = require('path');
const LM = require('../llm_common.js');
const { AG } = LM;
const src = fs.readFileSync(path.join(__dirname, '..', '09_latent_vae.lab.js'), 'utf8');
const P09 = new Function('LM', src.slice(0, src.indexOf('PipelineLab.mount(')) + '\nreturn P09;')(LM);
const PNG = require('./p09_png.js');
module.exports = { P09, LM, PNG };
if (require.main !== module) return;

const what = process.argv[2] || 'ae';
const kv = Object.fromEntries(process.argv.slice(3).map((a) => a.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const OUT = path.join(__dirname, 'shots09', 'proto'); fs.mkdirSync(OUT, { recursive: true });
const seed = 20251123;
let t0 = Date.now();
const data = P09.makeData(LM.rng(seed));
console.log(`data ${Date.now() - t0} ms`);
P09.initEvalEps(seed);

function train(kind, o, log) {
    const R = P09.makeRun(kind, Object.assign({ data, seed: seed + (kind === 'ae' ? 11 : kind === 'vae' ? 22 : 33), beta: 1, K: 32, restart: false, cap: 3000, B: 32, lr: 2e-3 }, o));
    P09.evalNow(R);
    let ts = 0, frames = 0, tsEval = 0;
    while (R.step < R.cap) {
        const a = process.hrtime.bigint();
        P09.trainStep(R);
        const b = process.hrtime.bigint();
        P09.evalWork(R, o.budget || 50);
        const c = process.hrtime.bigint();
        ts += Number(b - a) / 1e6; tsEval += Number(c - b) / 1e6; frames++;
        if (log && R.step % log === 0) {
            const r = R.res, T = R.thist;
            console.log(`${kind} step ${R.step}  train rec ${LM.mean(R.hist.rec.slice(-100)).toFixed(2)}` + (kind === 'vae' ? ` kl ${LM.mean(R.hist.kl.slice(-100)).toFixed(2)}` : '') + (kind === 'vq' ? ` cb ${LM.mean(R.hist.cb.slice(-100)).toFixed(3)} perp ${LM.mean(R.hist.perp.slice(-100)).toFixed(1)} dead ${R.deadNow} restarts ${R.restarts}` : '') +
                ` | test@${r.step}: rec ${r.rec.toFixed(2)} mse ${r.mse.toFixed(4)}` + (kind === 'vae' ? ` kl ${r.kl.toFixed(2)} [${r.klDim.map((v) => v.toFixed(2)).join(', ')}]` : '') + (kind === 'vq' ? ` used ${r.used}/${R.o.K} perp ${r.perp.toFixed(1)}` : ''));
        }
    }
    P09.evalNow(R);
    console.log(`${kind}: ${(ts / frames).toFixed(2)} ms/step, eval ${(tsEval / frames).toFixed(2)} ms/frame (budget ${o.budget || 50})`);
    return R;
}
function scatterPNG(file, R) { const z = R.res.z, pts = []; for (let i = 0; i < P09.NTEST; i++) pts.push([z[2 * i], z[2 * i + 1], data.te.y[i]]); return PNG.writeScatter(file, pts, 400, 2); }
function gridDecode(R, box, n) {
    const Z = AG.tensor(n * n, 2);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { Z.data[(r * n + c) * 2] = box.x0 + (c + 0.5) / n * (box.x1 - box.x0); Z.data[(r * n + c) * 2 + 1] = box.y1 - (r + 0.5) / n * (box.y1 - box.y0); }
    const p = AG.noGrad(() => P09.sigmoidInto(P09.decode(R.model, Z)));
    return Array.from({ length: n * n }, (_, k) => p.subarray(k * 256, (k + 1) * 256));
}
function reconPNG(file, R, idx) {
    const imgs = [];
    AG.noGrad(() => {
        const X = AG.tensor(idx.length, 256); idx.forEach((i, k) => X.data.set(data.te.X.subarray(i * 256, (i + 1) * 256), k * 256));
        let L;
        if (R.kind === 'vq') L = P09.vqForward(R.model, AG.tensor(idx.length * 16, 64, P09.patchesOf(X.data, idx.length)), idx.length).logits;
        else { const e = P09.encode(R.model, X); L = P09.decode(R.model, e.mu); }
        const p = P09.sigmoidInto(L);
        idx.forEach((i, k) => imgs.push(data.te.X.subarray(i * 256, (i + 1) * 256)));
        idx.forEach((i, k) => imgs.push(p.subarray(k * 256, (k + 1) * 256)));
    });
    PNG.writeGrid(file, imgs, 16, 16, idx.length, 4);
}
const pick = Array.from({ length: 10 }, (_, d) => data.byClass[d][0]).concat(Array.from({ length: 10 }, (_, d) => data.byClass[d][1]));

if (what === 'pca') {
    t0 = Date.now(); const P = P09.pca(data.tr, data.te); console.log(`pca ${Date.now() - t0} ms`);
    console.log('explained variance (cum) k=1..', [1, 2, 3, 5, 10, 20, 32, 50, 64, 100].map((k) => `${k}:${(100 * P.cum[k]).toFixed(1)}%`).join(' '));
    console.log('test mse k=', [0, 1, 2, 3, 5, 10, 20, 32, 50, 64, 100, 256].map((k) => `${k}:${P.mse[k].toFixed(4)}`).join(' '));
    // 正交性检查
    let mo = 0; for (let a = 0; a < 6; a++) for (let b = 0; b < 6; b++) { let s = 0; for (let j = 0; j < 256; j++) s += P.vecs[a * 256 + j] * P.vecs[b * 256 + j]; mo = Math.max(mo, Math.abs(s - (a === b ? 1 : 0))); }
    console.log('max |UᵀU − I| (6×6)', mo.toExponential(2), 'top eigenvalues', Array.from(P.vals.slice(0, 6), (v) => v.toFixed(3)).join(' '));
    const ev = Array.from({ length: 16 }, (_, k) => { const a = P.vecs.subarray(k * 256, (k + 1) * 256), m = Math.max(...a.map(Math.abs)); return a.map((v) => 0.5 + 0.5 * v / m); });
    PNG.writeGrid(path.join(OUT, 'pca_eigendigits.png'), [P.mean].concat(ev), 16, 16, 17, 4);
    const rows = []; for (const k of [1, 2, 5, 10, 20, 50]) pick.slice(0, 10).forEach((i) => rows.push(P09.pcaRecon(P, i, k)));
    PNG.writeGrid(path.join(OUT, 'pca_recon.png'), pick.slice(0, 10).map((i) => data.te.X.subarray(i * 256, (i + 1) * 256)).concat(rows), 16, 16, 10, 4);
    const pts = []; for (let i = 0; i < P09.NTEST; i++) pts.push([P.proj[i * 256], P.proj[i * 256 + 1], data.te.y[i]]);
    PNG.writeScatter(path.join(OUT, 'pca_scatter.png'), pts);
}
if (what === 'ae' || what === 'vae') {
    const R = train(what, kv, kv.log || 500);
    const box = scatterPNG(path.join(OUT, `${what}_scatter.png`), R);
    PNG.writeGrid(path.join(OUT, `${what}_grid.png`), gridDecode(R, box, 14), 16, 16, 14, 3, 0);
    if (what === 'vae') {
        const b2 = { x0: -2.5, x1: 2.5, y0: -2.5, y1: 2.5 }; PNG.writeGrid(path.join(OUT, `vae_gridprior.png`), gridDecode(R, b2, 14), 16, 16, 14, 3, 0);
        const r = LM.rng(3), Z = AG.tensor(40, 2); for (let i = 0; i < 80; i++) Z.data[i] = LM.randn(r);
        const p = AG.noGrad(() => P09.sigmoidInto(P09.decode(R.model, Z))); PNG.writeGrid(path.join(OUT, 'vae_samples.png'), Array.from({ length: 40 }, (_, k) => p.subarray(k * 256, (k + 1) * 256)), 16, 16, 10, 4);
        let s0 = 0, s1 = 0; for (let i = 0; i < P09.NTEST; i++) { s0 += Math.exp(0.5 * R.res.lv[2 * i]); s1 += Math.exp(0.5 * R.res.lv[2 * i + 1]); }
        console.log('mean σ per dim', (s0 / 500).toFixed(3), (s1 / 500).toFixed(3), 'mu range', box);
    }
    reconPNG(path.join(OUT, `${what}_recon.png`), R, pick);
    console.log('box', JSON.stringify(box));
}
if (what === 'vq') {
    const R = train('vq', Object.assign({ budget: 25 }, kv), kv.log || 500);
    reconPNG(path.join(OUT, `vq_recon_K${R.o.K}${R.o.restart ? '_rs' : ''}.png`), R, pick);
    console.log('test code counts', Array.from(R.res.cnt).map((v) => v | 0).join(' '));
}
