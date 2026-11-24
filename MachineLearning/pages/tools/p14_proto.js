// p14_proto.js — 第 14 篇的 Node 原型实验（数据、VAE、动力学）。用法：node tools/p14_proto.js [what]
const { LM, P14 } = require('./p14_load.js');
const { AG } = LM;
const what = process.argv[2] || 'vae';
const seed = 20251123;

function ascii(X, off) {
    const ch = ' .:-=+*#%@'; let s = '';
    for (let y = 0; y < 16; y++) { for (let x = 0; x < 16; x++) s += ch[Math.min(9, Math.floor(X[off + y * 16 + x] * 9.99))]; s += '\n'; }
    return s;
}
const t0 = Date.now();
const rand = LM.rng(seed);
const ds = P14.makeDataset(256, 32, rand);
const val = P14.makeDataset(64, 32, LM.rng(seed + 1));
console.log(`dataset ${Date.now() - t0} ms, frames ${ds.X.length / 256}`);
if (what === 'show') {
    for (let t = 0; t < 6; t++) console.log(ascii(ds.X, t * 4 * 256), ds.S.slice(t * 4 * 4, t * 4 * 4 + 4));
    // 速度分布
    let vs = []; for (let i = 0; i < ds.S.length / 4; i++) vs.push(Math.hypot(ds.S[i * 4 + 2], ds.S[i * 4 + 3]));
    vs.sort((a, b) => a - b); console.log('speed quantiles', [0.1, 0.5, 0.9, 0.99].map((q) => vs[Math.floor(q * vs.length)].toFixed(2)));
    let nb = 0; for (const b of ds.bumps) nb += b; console.log('bump frac', (nb / ds.bumps.length).toFixed(3));
    let ng = 0; for (let i = 0; i < ds.S.length / 4; i++) if (P14.inGoal(ds.S[i * 4], ds.S[i * 4 + 1])) ng++; console.log('in-goal frac', (ng / (ds.S.length / 4)).toFixed(3));
    process.exit(0);
}

function batchX(dsx, nFrames, B, r) {
    const X = AG.tensor(B, 256);
    for (let b = 0; b < B; b++) { const k = Math.floor(r() * nFrames); X.data.set(dsx.subarray(k * 256, (k + 1) * 256), b * 256); }
    return X;
}
if (what === 'vae') {
    const cfg = {};
    for (const a of process.argv.slice(3)) { const [k, v] = a.split('='); cfg[k] = +v; }
    const steps = cfg.steps || 3000, B = cfg.B || 32, lr = cfg.lr || 2e-3;
    delete cfg.steps; delete cfg.B; delete cfg.lr;
    const m = P14.makeVAE(7, cfg);
    const opt = new LM.AdamW(m.params, { lr, wd: 0, b2: 0.99 });
    const r = LM.rng(99), nF = ds.X.length / 256, nFv = val.X.length / 256;
    let ts = Date.now(), acc = { l: 0, rec: 0, kl: 0, n: 0 };
    for (let it = 1; it <= steps; it++) {
        const X = batchX(ds.X, nF, B, r);
        const o = P14.vaeLoss(m, X, r);
        opt.zeroGrad(); AG.backward(o.loss); opt.step(lr);
        acc.l += o.loss.item(); acc.rec += o.rec.item() / m.cfg.recW; acc.kl += o.kl.item(); acc.n++;
        if (it % 250 === 0) {
            // 验证：用 μ 重建的平均 Σ(x̂−x)²、每一维 KL
            const Xv = batchX(val.X, nFv, 256, LM.rng(5));
            const ev = AG.noGrad(() => { const e = P14.vaeEncodeAG(m, Xv); return { xh: P14.vaeDecodeAG(m, e.mu), mu: e.mu, lv: e.logvar }; });
            let se = 0; for (let i = 0; i < Xv.data.length; i++) se += (ev.xh.data[i] - Xv.data[i]) ** 2;
            const kld = new Float64Array(m.cfg.Z);
            for (let b = 0; b < 256; b++) for (let j = 0; j < m.cfg.Z; j++) { const mu = ev.mu.data[b * m.cfg.Z + j], lv = ev.lv.data[b * m.cfg.Z + j]; kld[j] += 0.5 * (mu * mu + Math.exp(lv) - 1 - lv) / 256; }
            console.log(`it ${it} loss ${(acc.l / acc.n).toFixed(2)} sse ${(acc.rec / acc.n).toFixed(3)} kl ${(acc.kl / acc.n).toFixed(2)} | val sse(μ) ${(se / 256).toFixed(3)} kl/dim ${Array.from(kld, (v) => v.toFixed(2)).join(' ')} | ${((Date.now() - ts) / acc.n).toFixed(2)} ms/step`);
            acc = { l: 0, rec: 0, kl: 0, n: 0 }; ts = Date.now();
        }
    }
    const Xv = batchX(val.X, nFv, 4, LM.rng(6));
    const xh = AG.noGrad(() => P14.vaeDecodeAG(m, P14.vaeEncodeAG(m, Xv).mu));
    for (let b = 0; b < 2; b++) { const a = ascii(Xv.data, b * 256).split('\n'), c = ascii(xh.data, b * 256).split('\n'); console.log(a.map((l, i) => l + '   ' + c[i]).join('\n')); }
    require("fs").writeFileSync(__dirname + "/p14_vae_" + (process.env.TAG || "tmp") + ".json", JSON.stringify({ cfg: m.cfg, b64: LM.gpt.exportWeights(m) }));
}

// 位置误差：解码帧里小球的质心（减去背景后）与真实位置之差
function centroid(img, off) {
    let sx = 0, sy = 0, sw = 0;
    for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) { const w = Math.max(0, img[off + j * 16 + i] - P14.BG[j * 16 + i]); sx += w * (i + 0.5); sy += w * (j + 0.5); sw += w; }
    return sw > 1e-6 ? [sx / sw, sy / sw, sw] : [NaN, NaN, 0];
}
module.exports.centroid = centroid;
if (what === 'vaepos') {
    const j = JSON.parse(require('fs').readFileSync(__dirname + '/p14_vae_' + (process.env.TAG || 'tmp') + '.json', 'utf8'));
    const m = LM.gpt.loadWeights(P14.makeVAE(7, j.cfg), j.b64);
    const n = 512, X = AG.tensor(n, 256), pos = [];
    const r = LM.rng(3);
    for (let i = 0; i < n; i++) { const p = P14.randomFree(r); pos.push(p); P14.render(p[0], p[1], X.data, i * 256); }
    const xh = AG.noGrad(() => P14.vaeDecodeAG(m, P14.vaeEncodeAG(m, X).mu));
    let e = 0, et = 0, mx = 0;
    for (let i = 0; i < n; i++) { const c = centroid(xh.data, i * 256), ct = centroid(X.data, i * 256); const d = Math.hypot(c[0] - pos[i][0], c[1] - pos[i][1]); e += d; et += Math.hypot(ct[0] - pos[i][0], ct[1] - pos[i][1]); mx = Math.max(mx, d); }
    console.log(`decoded centroid error ${(e / n).toFixed(3)} px (max ${mx.toFixed(2)}), true-frame centroid error ${(et / n).toFixed(3)} px`);
}

// ── 动力学：teacher forcing vs 多步训练；GRU vs 无记忆 MLP；开环复合误差；速度探针 ──
function loadVAE(tag) { const j = JSON.parse(require('fs').readFileSync(__dirname + '/p14_vae_' + tag + '.json', 'utf8')); return LM.gpt.loadWeights(P14.makeVAE(7, j.cfg), j.b64); }
module.exports.loadVAE = loadVAE;
/** 开环误差：在 nSeq 段验证序列上，先用 ctx 帧真实观测同步，然后只喂动作，预测 K 步；返回每个预测步的平均潜变量 MSE 与解码后小球位置误差（像素） */
function openLoop(dyn, vae, ds, lat, o) {
    const Z = dyn.cfg.Z, H = dyn.cfg.H, K = o.K, ctx = o.ctx, nSeq = o.nSeq, r = LM.rng(o.seed || 11), isMLP = !dyn.Wx;
    const lz = new Float64Array(K), lp = new Float64Array(K), tmp = {}, img = new Float32Array(256);
    for (let s = 0; s < nSeq; s++) {
        const e = Math.floor(r() * ds.nEp), t0 = Math.floor(r() * (ds.T - ctx - K + 1)), base = e * (ds.T + 1);
        const h = new Float32Array(H), z = new Float32Array(Z);
        z.set(lat.subarray((base + t0) * Z, (base + t0 + 1) * Z));
        for (let t = 0; t < ctx + K; t++) {
            const a = ds.A[e * ds.T + t0 + t];
            if (isMLP) P14.mlpDynFast(dyn, z, a, 1, tmp); else { P14.gruFast(dyn, z, a, h, 1, tmp); P14.headFast(dyn, h, z, 1); }
            const k = base + t0 + t + 1;
            if (t < ctx - 1) { z.set(lat.subarray(k * Z, (k + 1) * Z)); continue; }   // 上下文：用真实观测替换预测（滤波）
            const j = t - (ctx - 1); if (j >= K) break;
            let d = 0; for (let q = 0; q < Z; q++) d += (z[q] - lat[k * Z + q]) ** 2; lz[j] += d / Z;
            P14.decodeFast(vae, z, 1, img, tmp); const c = module.exports.centroid(img, 0);
            lp[j] += isFinite(c[0]) ? Math.hypot(c[0] - ds.S[k * 4], c[1] - ds.S[k * 4 + 1]) : 8;
        }
    }
    return { lz: Array.from(lz, (v) => v / nSeq), lp: Array.from(lp, (v) => v / nSeq) };
}
module.exports.openLoop = openLoop;
if (what === 'dyn') {
    const cfg = {}; for (const a of process.argv.slice(3)) { const [k, v] = a.split('='); cfg[k] = isNaN(+v) ? v : +v; }
    const vae = loadVAE(cfg.vae || 'a'), Z = vae.cfg.Z;
    const lat = P14.encodeDataset(vae, ds.X, ds.X.length / 256), latV = P14.encodeDataset(vae, val.X, val.X.length / 256);
    const steps = cfg.steps || 3000, B = cfg.B || 16, L = cfg.L || 16, lr = cfg.lr || 3e-3, mode = cfg.mode || 'tf';
    const dyn = P14.makeDyn(cfg.seed || 3, { Z, H: cfg.H || 32 }), opt = new LM.AdamW(dyn.params, { lr, wd: 0, b2: 0.99 });
    const mlp = P14.makeMLPDyn(4, { Z }), optM = new LM.AdamW(mlp.params, { lr, wd: 0, b2: 0.99 });
    const r = LM.rng(21 + (cfg.seed || 0)); let ts = Date.now(), acc = 0, accM = 0, accR = 0, n = 0, tsM = 0;
    for (let it = 1; it <= steps; it++) {
        const bt = P14.seqBatch(ds, lat, Z, B, L, r);
        const o = P14.dynLoss(dyn, bt, { mode, warm: cfg.warm || 4 }); opt.zeroGrad(); AG.backward(o.loss); LM.clipGradNorm(dyn.params, 10); opt.step(lr);
        acc += o.dyn; accR += o.rew; n++;
        const t1 = Date.now(); const om = P14.mlpDynLoss(mlp, bt); optM.zeroGrad(); AG.backward(om.loss); optM.step(lr); accM += om.loss.item(); tsM += Date.now() - t1;
        if (it % 500 === 0) {
            const ol = openLoop(dyn, vae, val, latV, { K: 20, ctx: 4, nSeq: 200 }), olM = openLoop(mlp, vae, val, latV, { K: 20, ctx: 4, nSeq: 200 });
            console.log(`it ${it} dyn ${(acc / n).toExponential(2)} rew ${(accR / n).toExponential(2)} mlp ${(accM / n).toExponential(2)} | ${((Date.now() - ts - tsM) / n).toFixed(2)} ms/step (mlp ${(tsM / n).toFixed(2)})`);
            console.log(`   open-loop px err  GRU: ${[0, 1, 2, 4, 9, 14, 19].map((k) => ol.lp[k].toFixed(2)).join(' ')}   MLP: ${[0, 1, 2, 4, 9, 14, 19].map((k) => olM.lp[k].toFixed(2)).join(' ')}`);
            acc = accM = accR = n = tsM = 0; ts = Date.now();
        }
    }
    require('fs').writeFileSync(__dirname + '/p14_dyn_' + (process.env.TAG || mode) + '.json', JSON.stringify({ cfg: dyn.cfg, b64: LM.gpt.exportWeights(dyn), mlp: { cfg: mlp.cfg, b64: LM.gpt.exportWeights(mlp) } }));
}
