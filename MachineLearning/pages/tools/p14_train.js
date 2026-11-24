// p14_train.js — 第 14 篇的离线训练 + 测量（用页面里同一份 P14 代码、同一个配方 P14.RECIPE）。
// 训练：V（VAE）、M（多步训练 / teacher forcing / 3 个集成成员 / 加入规划器数据后重训）、对照组 MLP、actor-critic；
// 测量：正文引用的所有数字（写进 P14_W.facts）；最后把权重（float16 base64）写进 14_world_models.lab.js 的 <P14_W> 段。
// 用法（在 pages/ 下）：node tools/p14_train.js            约 6–8 分钟
const fs = require('fs');
const path = require('path');
const { LM, P14, file } = require('./p14_load.js');
const SEED = 20251123, R = P14.RECIPE, log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const facts = {}, W_ = {};
const ds = P14.makeDataset(R.data.nEp, R.data.T, LM.rng(SEED), { stick: R.data.stick });
const val = P14.makeDataset(R.data.nVal, R.data.T, LM.rng(SEED + 1), { stick: R.data.stick });
const nF = ds.nEp * (ds.T + 1), nFv = val.nEp * (val.T + 1);
{   // 数据统计
    const sp = []; for (let i = 0; i < nF; i++) sp.push(Math.hypot(ds.S[i * 4 + 2], ds.S[i * 4 + 3])); sp.sort((a, b) => a - b);
    let nb = 0; for (const b of ds.bumps) nb += b; let ng = 0; for (let i = 0; i < nF; i++) if (P14.inGoal(ds.S[i * 4], ds.S[i * 4 + 1])) ng++;
    facts.data = { frames: nF, bumpFrac: +(nb / ds.bumps.length).toFixed(3), goalFrac: +(ng / nF).toFixed(3), speedQ: [0.1, 0.5, 0.9].map((q) => +sp[Math.floor(q * nF)].toFixed(2)), vmax: +(P14.PHYS.thrust / (1 - P14.PHYS.damp)).toFixed(2) };
    log('data', JSON.stringify(facts.data));
}
// ── V ──
const vr = P14.vaeRun(ds), vCurve = []; let acc = [0, 0, 0, 0];
for (let it = 1; it <= R.vae.steps; it++) {
    const o = P14.vaeRunStep(vr); acc[0] += o.loss; acc[1] += o.sse; acc[2] += o.kl; acc[3]++;
    if (it % 100 === 0) { vCurve.push([it, +(acc[0] / acc[3]).toFixed(3), +(acc[1] / acc[3]).toFixed(4), +(acc[2] / acc[3]).toFixed(3)]); acc = [0, 0, 0, 0]; }
}
const vae = vr.m;
{
    const tmp = {}, mu = new Float32Array(nFv * 4), lv = new Float32Array(nFv * 4), xh = new Float32Array(nFv * 256);
    for (let i = 0; i < nFv; i += 256) { const n = Math.min(256, nFv - i); P14.encodeFast(vae, val.X.subarray(i * 256, (i + n) * 256), n, mu.subarray(i * 4, (i + n) * 4), tmp, lv.subarray(i * 4, (i + n) * 4)); }
    P14.decodeFast(vae, mu, nFv, xh, tmp);
    let se = 0, pe = 0, pm = 0; const kl = [0, 0, 0, 0];
    for (let i = 0; i < nFv; i++) {
        for (let j = 0; j < 256; j++) se += (xh[i * 256 + j] - val.X[i * 256 + j]) ** 2;
        const p = P14.ballPos(xh.subarray(i * 256, (i + 1) * 256)), d = Math.hypot(p[0] - val.S[i * 4], p[1] - val.S[i * 4 + 1]); pe += d; pm = Math.max(pm, d);
        for (let j = 0; j < 4; j++) kl[j] += 0.5 * (mu[i * 4 + j] ** 2 + Math.exp(lv[i * 4 + j]) - 1 - lv[i * 4 + j]) / nFv;
    }
    facts.vae = { steps: R.vae.steps, valSSE: +(se / nFv).toFixed(3), posErr: +(pe / nFv).toFixed(3), posMax: +pm.toFixed(2), kl: kl.map((v) => +v.toFixed(2)), params: vae.params.reduce((s, p) => s + p.size, 0) };
    log('vae', JSON.stringify(facts.vae));
}
W_.vae = { cfg: vae.cfg, b64: P14.exportModel(vae), curve: vCurve };
// ── M ──
const lat = P14.encodeDataset(vae, ds.X, nF), latV = P14.encodeDataset(vae, val.X, nFv);
function trainDyn(mode, seed, noMLP, data, latents) {
    const r = P14.dynRun(data || ds, latents || lat, { mode, seed, batchSeed: R.dyn.batchSeed + (seed - R.dyn.seed) }), curve = []; let a = [0, 0, 0];
    for (let it = 1; it <= R.dyn.steps; it++) {
        const o = P14.dynRunStep(r, noMLP); a[0] += o.dyn; a[1] += o.mlp; a[2]++;
        if (it % 100 === 0) { curve.push([it, +(a[0] / a[2]).toExponential(3), noMLP ? null : +(a[1] / a[2]).toExponential(3)]); a = [0, 0, 0]; }
    }
    log('dyn', mode, seed, 'final', curve[curve.length - 1]);
    return { r, curve };
}
const dMS = trainDyn('ms', R.dyn.seed, false), dTF = trainDyn('tf', R.dyn.seed, false);
const ens = [4, 5, 6].map((s) => trainDyn('ms', s, true).r.m);
const Wms = { vae, dyn: dMS.r.m }, Wtf = { vae, dyn: dTF.r.m };
W_.ms = { cfg: dMS.r.m.cfg, b64: P14.exportModel(dMS.r.m), curve: dMS.curve };
W_.tf = { cfg: dTF.r.m.cfg, b64: P14.exportModel(dTF.r.m), curve: dTF.curve };
W_.mlp = { cfg: dTF.r.mlp.cfg, b64: P14.exportModel(dTF.r.mlp) };
W_.ens = ens.map((m) => ({ cfg: m.cfg, b64: P14.exportModel(m) }));
{
    const ol = (m) => P14.openLoopErr(m, vae, val, latV, { K: 20, ctx: 4, nSeq: 100 });
    const a = ol(dMS.r.m), b = ol(dTF.r.m), c = ol(dTF.r.mlp);
    const f = (x) => x.lp.map((v) => +v.toFixed(2));
    facts.openLoop = { ms: f(a), tf: f(b), mlp: f(c), msZ: a.lz.map((v) => +v.toExponential(2)), tfZ: b.lz.map((v) => +v.toExponential(2)), mlpZ: c.lz.map((v) => +v.toExponential(2)) };
    log('openLoop', JSON.stringify(facts.openLoop));
    const vp = P14.velProbe(dMS.r.m, val, latV), vpt = P14.velProbe(dTF.r.m, val, latV);
    facts.velProbe = { n: vp.n, fromH: vp.fromH.r2.map((v) => +v.toFixed(3)), fromZ: vp.fromZ.r2.map((v) => +v.toFixed(3)), fromHtf: vpt.fromH.r2.map((v) => +v.toFixed(3)) };
    log('velProbe', JSON.stringify(facts.velProbe));
}
// ── 规划（MPC）与模型被利用 ──
const starts = P14.evalStarts(R.mpc.nStarts);
function evalMPC(W, o, withPen) {
    o = Object.assign({ elite: eliteOf(o.N || R.mpc.N) }, R.mpc, o);
    const tmp = {}, r = LM.rng(R.mpc.seed), img = new Float32Array(256); let succ = 0, ret = 0, dec = 0, pen = 0, steps = 0; const fails = [], t0 = Date.now();
    for (const st of starts) {
        const ep = P14.mpcBegin(W, st);
        while (!ep.done) {
            if (o.policy) P14.mpcStep(W, ep, o, r, tmp); else P14.mpcStep(W, ep, o, r, tmp);
            dec++;
            if (withPen && ep.plan) {
                const d = P14.dreamSeq(W, ep.plan.st, ep.plan.acts, tmp); let mx = 0;
                for (let t = 0; t < ep.plan.acts.length; t++) { P14.decodeFast(W.vae, d.zs.subarray(t * 4, t * 4 + 4), 1, img, tmp); const p = P14.ballPos(img); if (isFinite(p[0])) mx = Math.max(mx, P14.penetration(p[0], p[1])); }
                if (mx > 0.3) pen++;
            }
        }
        if (ep.success) { succ++; steps += ep.t; } else fails.push(starts.indexOf(st));
        ret += ep.ret;
    }
    return { succ: +(succ / starts.length).toFixed(3), ret: +(ret / starts.length).toFixed(2), steps: +(steps / Math.max(1, succ)).toFixed(1), pen: withPen ? +(pen / dec).toFixed(3) : null, dec, fails, ms: +((Date.now() - t0) / dec).toFixed(2) };
}
const eliteOf = P14.eliteOf;
facts.random = evalMPC(Wms, { policy: (ep, r) => Math.floor(r() * 5) });
log('random', JSON.stringify(facts.random));
facts.mpc = { ms: evalMPC(Wms, {}, true), tf: evalMPC(Wtf, {}, true) };
facts.mpc.H = [4, 8, 12, 16].map((H) => [H, evalMPC(Wms, { H }).succ]);
facts.mpc.iters = [1, 2, 3].map((it) => [it, evalMPC(Wms, { iters: it }).succ]);
facts.mpc.N = [8, 32, 128].map((N) => [N, evalMPC(Wms, { N }).succ]);
log('mpc', JSON.stringify(facts.mpc));
{
    const We = { vae, dyn: dMS.r.m, ens: [dMS.r.m].concat(ens) };
    facts.kappa = [0, 1, 3, 10].map((k) => { const e = evalMPC(We, { kappa: k }, true); return [k, e.succ, e.pen, e.ret]; });
    log('kappa', JSON.stringify(facts.kappa));
}
// 规划器自己收集数据（ε = 0.2 的随机动作）→ 并入数据集 → 从头重训 M
{
    const nEp = 128, T = R.data.T, n1 = T + 1, r = LM.rng(555), tmp = {};
    const S = new Float32Array(nEp * n1 * 4), A = new Int32Array(nEp * T), X = new Float32Array(nEp * n1 * 256), Rw = new Float32Array(nEp * n1), bumps = new Uint8Array(nEp * T);
    for (let e = 0; e < nEp; e++) {
        const ep = P14.mpcBegin(Wms, P14.randomFree(r));
        for (let t = 0; t <= T; t++) {
            const k = e * n1 + t; S.set(ep.s, k * 4); X.set(ep.frame, k * 256); Rw[k] = P14.reward(ep.s[0], ep.s[1]);
            if (t === T) break;
            const pl = P14.planCEM(Wms, ep.st, { N: R.mpc.N, H: R.mpc.H, iters: R.mpc.iters, elite: eliteOf(R.mpc.N), gamma: R.mpc.gamma }, r, ep.warm, tmp); ep.warm = pl.acts;
            const a = r() < 0.2 ? Math.floor(r() * 5) : pl.acts[0];
            A[e * T + t] = a; bumps[e * T + t] = P14.envStep(ep.s, a); P14.render(ep.s[0], ep.s[1], ep.frame); P14.wmObserve(Wms, ep.st, a, ep.frame, tmp);
        }
    }
    const cat = (a, b) => { const c = new a.constructor(a.length + b.length); c.set(a); c.set(b, a.length); return c; };
    const ds2 = { nEp: ds.nEp + nEp, T, S: cat(ds.S, S), A: cat(ds.A, A), X: cat(ds.X, X), R: cat(ds.R, Rw), bumps: cat(ds.bumps, bumps) };
    let nb = 0; for (const b of bumps) nb += b;
    const visit = (data) => { let near = 0, n = 0; for (let i = 0; i < data.S.length / 4; i++) { const x = data.S[i * 4], y = data.S[i * 4 + 1]; n++; if (Math.hypot(x - 6, y - 10.5) < 2.2 || Math.hypot(x - 10, y - 10.5) < 2.2) near++; } return near / n; };
    facts.dagger = { nEp, bumpFrac: +(nb / bumps.length).toFixed(3), cornerFracRandom: +visit(ds).toFixed(3), cornerFracPlanner: +visit({ S }).toFixed(3) };
    const dDG = trainDyn('ms', R.dyn.seed, true, ds2, P14.encodeDataset(vae, ds2.X, ds2.nEp * n1));
    W_.dagger = { cfg: dDG.r.m.cfg, b64: P14.exportModel(dDG.r.m), curve: dDG.curve };
    facts.mpc.dagger = evalMPC({ vae, dyn: dDG.r.m }, {}, true);
    log('dagger', JSON.stringify(facts.dagger), JSON.stringify(facts.mpc.dagger));
    // 优化压力：只规划一次（开环），候选数 N ↑ → 想象回报 vs 真实回报
    const pressure = (W) => [1, 4, 16, 64, 256, 1024].map((N) => {
        let im = 0, re = 0; const tmp2 = {}, sts = starts.slice(0, 16);
        sts.forEach((p, si) => {
            const s = new Float32Array([p[0], p[1], 0, 0]), st = P14.wmStart(W, P14.render(s[0], s[1]));
            const pl = P14.planCEM(W, st, { N, H: R.mpc.H, iters: N > 1 ? 3 : 1, elite: Math.max(1, Math.round(N / 8)), gamma: 1 }, LM.rng(1000 + si), null, tmp2);
            im += pl.ret; for (let t = 0; t < R.mpc.H; t++) { P14.envStep(s, pl.acts[t]); re += P14.reward(s[0], s[1]); }
        });
        return [N, +(im / 16).toFixed(2), +(re / 16).toFixed(2)];
    });
    facts.pressure = { ms: pressure(Wms), tf: pressure(Wtf), dagger: pressure({ vae, dyn: dDG.r.m }) };
    log('pressure', JSON.stringify(facts.pressure));
}
// 穿墙的典型例子：第一次规划的想象轨迹陷进墙里最深的起点
{
    const tmp = {}, img = new Float32Array(256); let best = null;
    const cands = P14.evalStarts(60, 99);
    for (const p of cands) {
        const st = P14.wmStart(Wms, P14.render(p[0], p[1]));
        const pl = P14.planCEM(Wms, st, { N: R.mpc.N, H: R.mpc.H, iters: R.mpc.iters, elite: eliteOf(R.mpc.N), gamma: R.mpc.gamma }, LM.rng(R.mpc.seed), null, tmp);
        const d = P14.dreamSeq(Wms, st, pl.acts, tmp); let mx = 0;
        for (let t = 0; t < R.mpc.H; t++) { P14.decodeFast(vae, d.zs.subarray(t * 4, t * 4 + 4), 1, img, tmp); const q = P14.ballPos(img); if (isFinite(q[0])) mx = Math.max(mx, P14.penetration(q[0], q[1])); }
        if (!best || mx > best.pen) best = { p: p.map((v) => +v.toFixed(2)), pen: +mx.toFixed(2) };
    }
    facts.trap = best;
    log('trap', JSON.stringify(best));
}
// ── actor-critic（在想象里学策略） ──
{
    const rep = { Hs: P14.replayStates(Wms, ds, lat), lat, n: nF };
    const A = P14.acBegin(Wms, rep, R.ac.seed), curve = []; let a = [0, 0];
    const evalAC = () => evalMPC(Wms, { policy: (ep) => P14.acAct(A.ac, ep.st, A.tmp, true) });
    for (let it = 1; it <= R.ac.steps; it++) {
        const o = P14.acUpdate(A); a[0] += o.imagRet; a[1]++;
        if (it % 250 === 0) { const e = evalAC(); curve.push([it, +(a[0] / a[1]).toFixed(2), e.succ, e.ret]); a = [0, 0]; log('ac', it, JSON.stringify(curve[curve.length - 1])); }
    }
    facts.ac = { curve, final: evalAC() };
    W_.ac = { cfg: A.ac.cfg, actor: P14.exportModel({ params: A.ac.aParams }), critic: P14.exportModel({ params: A.ac.cParams }) };
}
// ── 写进页面 ──
const out = { seed: SEED, recipe: R, facts, models: W_ };
fs.writeFileSync(path.join(__dirname, 'p14_weights.json'), JSON.stringify(out));
const src = fs.readFileSync(file, 'utf8'), a0 = src.indexOf('// <P14_W>'), a1 = src.indexOf('// </P14_W>');
const block = `// <P14_W> 离线训练好的权重与测量结果（tools/p14_train.js 生成，请勿手改）\nconst P14_W = ${JSON.stringify(out)};\n`;
const next = a0 >= 0 && a1 > a0 ? src.slice(0, a0) + block + src.slice(a1) : src.replace('const P14 = (() => {', block + '// </P14_W>\n\nconst P14 = (() => {');
fs.writeFileSync(file, next);
log('written', path.basename(file), (block.length / 1024).toFixed(0) + ' KB');
