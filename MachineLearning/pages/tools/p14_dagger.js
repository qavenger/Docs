// p14_dagger.js — 用规划器自己收集数据（带 ε 随机动作）→ 并入数据集 → 重新训练 M → 再评估 MPC。
// 用法：node tools/p14_dagger.js nEp=128 steps=6000 eps=0.2
const { LM, P14 } = require('./p14_load.js');
const { AG } = LM;
const fs = require('fs');
const cfg = {}; for (const a of process.argv.slice(2)) { const [k, v] = a.split('='); cfg[k] = +v; }
const load = (f, mk) => { const j = JSON.parse(fs.readFileSync(__dirname + '/' + f, 'utf8')); return LM.gpt.loadWeights(mk(j.cfg), j.b64); };
const vae = load('p14_vae_a4.json', (c) => P14.makeVAE(7, c)), dyn0 = load('p14_dyn_ms4.json', (c) => P14.makeDyn(3, c));
const ds0 = P14.makeDataset(256, 32, LM.rng(20251123));
// 1) 规划器收集：从随机起点出发，32 步，ε 概率随机动作
const nEp = cfg.nEp || 128, T = 32, eps = cfg.eps ?? 0.2, r = LM.rng(555), tmp = {}, W = { vae, dyn: dyn0 };
const n1 = T + 1, S = new Float32Array(nEp * n1 * 4), A = new Int32Array(nEp * T), X = new Float32Array(nEp * n1 * 256), Rw = new Float32Array(nEp * n1), bumps = new Uint8Array(nEp * T);
let t0 = Date.now();
for (let e = 0; e < nEp; e++) {
    const p = P14.randomFree(r), ep = P14.mpcBegin(W, p);
    for (let t = 0; t <= T; t++) {
        const k = e * n1 + t; S.set(ep.s, k * 4); X.set(ep.frame, k * 256); Rw[k] = P14.reward(ep.s[0], ep.s[1]);
        if (t === T) break;
        const pl = P14.planCEM(W, ep.st, { N: 32, H: 12, iters: 2, elite: 6, gamma: 0.97 }, r, ep.warm, tmp); ep.warm = pl.acts;
        const a = r() < eps ? Math.floor(r() * 5) : pl.acts[0];
        A[e * T + t] = a; bumps[e * T + t] = P14.envStep(ep.s, a); P14.render(ep.s[0], ep.s[1], ep.frame); P14.wmObserve(W, ep.st, a, ep.frame, tmp);
    }
}
console.log(`collected ${nEp} planner episodes in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
const cat = (a, b) => { const c = new a.constructor(a.length + b.length); c.set(a); c.set(b, a.length); return c; };
const ds = { nEp: ds0.nEp + nEp, T, S: cat(ds0.S, S), A: cat(ds0.A, A), X: cat(ds0.X, X), R: cat(ds0.R, Rw), bumps: cat(ds0.bumps, bumps) };
const lat = P14.encodeDataset(vae, ds.X, ds.nEp * n1);
// 2) 从头训练 M（多步）
const dyn = P14.makeDyn(3, dyn0.cfg), opt = new LM.AdamW(dyn.params, { lr: 3e-3, wd: 0, b2: 0.99 }), rb = LM.rng(21);
t0 = Date.now();
for (let it = 1; it <= (cfg.steps || 6000); it++) { const bt = P14.seqBatch(ds, lat, 4, 16, 16, rb); const o = P14.dynLoss(dyn, bt, { mode: 'ms', warm: 4 }); opt.zeroGrad(); AG.backward(o.loss); LM.clipGradNorm(dyn.params, 10); opt.step(3e-3); }
console.log(`trained M in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
fs.writeFileSync(__dirname + '/p14_dyn_dagger.json', JSON.stringify({ cfg: dyn.cfg, b64: LM.gpt.exportWeights(dyn) }));
