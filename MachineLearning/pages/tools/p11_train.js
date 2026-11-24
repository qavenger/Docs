// p11_train.js — 第 11 篇的离线训练：用页面里同一份 P11 代码训练 Step 04–09 用到的模型，写到 tools/p11_ckpt/final_<name>.json
// 用法（在 pages/ 下）：node tools/p11_train.js <vp|cfg|fm|rf|latent>（rf 要等 fm 写出配对）；之后 node tools/p11_bake.js 把权重写进 11_diffusion.lab.js 的 P11_W
const fs = require('fs'), path = require('path');
const { LM, P11 } = require('./p11_load.js');
const { AG } = LM;
const name = process.argv[2], dir = path.join(__dirname, 'p11_ckpt'); fs.mkdirSync(dir, { recursive: true });
const RECIPE = { width: 48, depth: 3, ff: 4, B: 128, lr: 2e-3, steps: 20000, warmup: 200, ema: 0.999 };   // 与 Step 03 实时训练同一配方（实时版只训 4000 步）
const save = (obj) => fs.writeFileSync(path.join(dir, `final_${name}.json`), JSON.stringify(obj));
const t0 = performance.now(), log = (s) => console.log(`[${name}] ${s} (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
function run2d(o) {
    const R = P11.trainer(Object.assign({ sched: P11.SCHED.cosine, data: P11.trainSet() }, RECIPE, o));
    const curve = []; let acc = 0;
    for (let s = 1; s <= RECIPE.steps; s++) { acc += R.step().loss; if (s % 100 === 0) { curve.push(+(acc / 100).toFixed(4)); acc = 0; } if (s % 5000 === 0) log(`step ${s}`); }
    return { R, curve };
}
if (name === 'vp' || name === 'cfg' || name === 'fm') {
    const o = name === 'vp' ? { kind: 'vp', seed: 1 } : name === 'cfg' ? { kind: 'vp', seed: 2, ncls: 3, pUncond: 0.2 } : { kind: 'fm', seed: 3 };
    const { R, curve } = run2d(o);
    save({ cfg: R.net.cfg, recipe: Object.assign({}, RECIPE, o), b64: P11.exportNet(R.ema), curve });
    if (name === 'fm') {   // reflow 用的配对：8192 个噪声 z → 这个 FM 模型的 ODE 解（欧拉 100 步）
        const n = 8192, Z = new Float32Array(2 * n), r = LM.rng(4242); for (let i = 0; i < 2 * n; i++) Z[i] = LM.randn(r);
        const X0 = Z.slice(); P11.sampleFM(R.ema, X0, n, { K: 100 });
        fs.writeFileSync(path.join(dir, 'final_fm.pairs.json'), JSON.stringify({ n, Z: Array.from(Z), X0: Array.from(X0) }));
        log('pairs written');
    }
}
if (name === 'rf') {
    const j = JSON.parse(fs.readFileSync(path.join(dir, 'final_fm.pairs.json'), 'utf8'));
    const { R, curve } = run2d({ kind: 'fm', seed: 4, pairs: { n: j.n, Z: Float32Array.from(j.Z), X0: Float32Array.from(j.X0) } });
    save({ cfg: R.net.cfg, recipe: Object.assign({}, RECIPE, { kind: 'fm', seed: 4, reflow: '8192 对 (z, ODE_fm(z))，欧拉 100 步' }), b64: P11.exportNet(R.ema), curve });
}
if (name === 'latent') {
    const { train } = P11.digitSets(), ae = P11.createAE(7), opt = new LM.AdamW(ae.params, { lr: 2e-3, b1: 0.9, b2: 0.999, wd: 0 }), B = 64, Xb = AG.tensor(B, 256), r = LM.rng(77), AESTEPS = 4000;
    for (let s = 0; s < AESTEPS; s++) {   // 重建 MSE + 1e-3·mean(z²)（轻微约束潜变量的尺度）
        for (let b = 0; b < B; b++) { const k = Math.floor(r() * train.n); Xb.data.set(train.X.subarray(k * 256, (k + 1) * 256), b * 256); }
        opt.zeroGrad(); const z = P11.aeEnc(ae, Xb); const loss = AG.add(AG.mse(P11.aeDec(ae, z), Xb), AG.scale(AG.mean(AG.square(z)), 1e-3)); AG.backward(loss);
        opt.step(LM.cosineLR(s, { maxLR: 2e-3, minLR: 1e-4, warmup: 100, total: AESTEPS }));
    }
    const Z = P11.AE_Z, zs = AG.noGrad(() => P11.aeEnc(ae, AG.tensor(train.n, 256, train.X))).data, mu = new Float32Array(Z), sd = new Float32Array(Z);
    for (let i = 0; i < train.n; i++) for (let d = 0; d < Z; d++) mu[d] += zs[i * Z + d] / train.n;
    for (let i = 0; i < train.n; i++) for (let d = 0; d < Z; d++) sd[d] += (zs[i * Z + d] - mu[d]) ** 2 / train.n;
    for (let d = 0; d < Z; d++) sd[d] = Math.sqrt(sd[d]);
    log('AE done');
    const L = { ae, mu, sd }, zn = P11.encodeDigits(L, train.X, train.n);
    const R = P11.trainer({ kind: 'vp', sched: P11.SCHED.cosine, data: { X: zn, lab: train.y, n: train.n }, din: Z, width: 64, depth: 3, ncls: 10, pUncond: 0.2, B: 128, lr: 2e-3, steps: 20000, seed: 5, warmup: 200, ff: 0, ema: 0.999 });
    for (let s = 1; s <= 20000; s++) { R.step(); if (s % 5000 === 0) log(`ldm step ${s}`); }
    save({ ae: LM.gpt.exportWeights({ params: ae.params }), aeSteps: AESTEPS, mu: Array.from(mu), sd: Array.from(sd), cfg: R.net.cfg, b64: P11.exportNet(R.ema), ldmSteps: 20000 });
}
log('saved');
