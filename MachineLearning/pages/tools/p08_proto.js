// p08_proto.js — 第 08 篇的 Node 原型：直接加载 08_cnn_unet.lab.js 里 PipelineLab.mount( 之前的 P08 命名空间（与页面同一份代码）。
// 用法（在 pages/ 下）：node tools/p08_proto.js [what] [steps]
const fs = require('fs'), path = require('path');
const LM = require('../llm_common.js');
const src = fs.readFileSync(path.join(__dirname, '../08_cnn_unet.lab.js'), 'utf8');
const head = src.slice(0, src.indexOf('PipelineLab.mount('));
const P08 = new Function('LM', head + '\nreturn P08;')(LM);
module.exports = { P08, LM };
if (require.main !== module) return;

const what = process.argv[2] || 'cnn', STEPS = +(process.argv[3] || 1500);
const seed = 20251123;
let t0 = Date.now();
const data = P08.makeData(LM.rng(seed));
console.log(`data ${Date.now() - t0} ms`);
const pct = (v) => (100 * v).toFixed(1) + '%';

function run(label, model, lr, o) {
    o = o || {};
    const R = P08.makeRun({ model, lr, cosine: !!o.cosine, B: o.B || 32, cap: o.cap || STEPS, seed: o.seed || seed + 7, aug: o.aug || 0, data, evalSets: [] });
    const marks = o.marks || [250, 500, 1000, 1500, 2000, 3000].filter((s) => s <= STEPS);
    let tTrain = 0; const out = [];
    while (R.step < R.cap) {
        const t = process.hrtime.bigint(); P08.trainStep(R); tTrain += Number(process.hrtime.bigint() - t) / 1e6;
        if (marks.includes(R.step)) { const te = P08.evaluate(model, data.te.X, data.te.y, data.allTe), trn = P08.evaluate(model, data.tr.X, data.tr.y, data.evalTr); out.push(`s${R.step}: tr ${pct(trn.acc)} te ${pct(te.acc)} L ${te.loss.toFixed(3)}`); }
    }
    if (!o.quiet) console.log(`${label.padEnd(30)} ${P08.numParams(model.params).toString().padStart(6)} params  ${(tTrain / STEPS).toFixed(2)} ms/step | ${out.join(' · ')}`);
    return R;
}
if (what === 'cnn') {
    run('MLP 64 adam 1e-3', P08.makeMLP({ hidden: 64 }, LM.rng(5)), 1e-3);
    run('CNN 8/16 flat adam 1e-3', P08.makeCNN({ c1: 8, c2: 16 }, LM.rng(5)), 1e-3);
    run('CNN 8/16 flat adam 2e-3', P08.makeCNN({ c1: 8, c2: 16 }, LM.rng(5)), 2e-3);
    run('CNN 6/12 flat adam 2e-3', P08.makeCNN({ c1: 6, c2: 12 }, LM.rng(5)), 2e-3);
    run('CNN 8/16 gap adam 2e-3', P08.makeCNN({ c1: 8, c2: 16, head: 'gap' }, LM.rng(5)), 2e-3);
}
if (what === 'mlp') {
    run('MLP 64 adam 1e-3', P08.makeMLP({ hidden: 64 }, LM.rng(5)), 1e-3);
    run('MLP 64 adam 2e-3', P08.makeMLP({ hidden: 64 }, LM.rng(5)), 2e-3);
    run('MLP 64 adam 5e-4', P08.makeMLP({ hidden: 64 }, LM.rng(5)), 5e-4);
}

if (what === 'lr') {
    run('MLP adam 1e-3 cos', P08.makeMLP({ hidden: 64 }, LM.rng(5)), 1e-3, { cosine: true });
    run('MLP adam 2e-3 cos', P08.makeMLP({ hidden: 64 }, LM.rng(5)), 2e-3, { cosine: true });
    run('MLP adam 3e-3 cos', P08.makeMLP({ hidden: 64 }, LM.rng(5)), 3e-3, { cosine: true });
    run('CNN adam 2e-3 cos', P08.makeCNN({ c1: 8, c2: 16 }, LM.rng(5)), 2e-3, { cosine: true });
    run('CNN adam 3e-3 cos', P08.makeCNN({ c1: 8, c2: 16 }, LM.rng(5)), 3e-3, { cosine: true });
}
if (what === 'shift') {
    const lr = +(process.argv[4] || 2e-3);
    const models = [
        ['MLP', P08.makeMLP({ hidden: 64 }, LM.rng(5)), {}],
        ['MLP + aug±2', P08.makeMLP({ hidden: 64 }, LM.rng(5)), { aug: 2 }],
        ['MLP + aug±3', P08.makeMLP({ hidden: 64 }, LM.rng(5)), { aug: 3 }],
        ['CNN', P08.makeCNN({ c1: 8, c2: 16 }, LM.rng(5)), {}],
        ['CNN + aug±2', P08.makeCNN({ c1: 8, c2: 16 }, LM.rng(5)), { aug: 2 }],
        ['CNN gap', P08.makeCNN({ c1: 8, c2: 16, head: 'gap' }, LM.rng(5)), {}],
    ];
    const sh = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
    console.log('model'.padEnd(14), 'dx: ' + sh.map((v) => String(v).padStart(5)).join(''), ' | dy: ' + [-3, -2, -1, 0, 1, 2, 3].map((v) => String(v).padStart(5)).join(''));
    for (const [name, m, o] of models) {
        run(name, m, lr, Object.assign({ cosine: true, marks: [STEPS] }, o));
        const ax = sh.map((d) => P08.evaluate(m, data.te.X, data.te.y, data.allTe, [d, 0]).acc), ay = [-3, -2, -1, 0, 1, 2, 3].map((d) => P08.evaluate(m, data.te.X, data.te.y, data.allTe, [0, d]).acc);
        console.log(name.padEnd(14), '    ' + ax.map((a) => (100 * a).toFixed(0).padStart(5)).join(''), ' |     ' + ay.map((a) => (100 * a).toFixed(0).padStart(5)).join(''));
    }
}
if (what === 'unet') {
    const sigma = +(process.argv[4] || 0.5), B = +(process.argv[5] || 8), lr = +(process.argv[6] || 3e-3);
    const ns = P08.noisySet(data, 200, sigma, 99);
    console.log(`sigma ${sigma}: noisy input MSE ${P08.mseOf(ns.Xn, ns.X0, ns.Xn.length).toFixed(4)} (PSNR ${P08.psnr(P08.mseOf(ns.Xn, ns.X0, ns.Xn.length)).toFixed(2)} dB)`);
    // classical baseline: best Gaussian blur (3x3 binomial / 5x5) + clamp
    for (const [nm, k] of [['box3', [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9)], ['binom3', [1, 2, 1, 2, 4, 2, 1, 2, 1].map((v) => v / 16)]]) {
        const { AG } = LM; const W = AG.tensor(1, 9, Float32Array.from(k));
        const y = AG.noGrad(() => P08.conv(AG.tensor(ns.n, 256, ns.Xn), W, null, { C: 1, H: 16, W: 16, K: 3 })).data;
        console.log(`  blur ${nm}: MSE ${P08.mseOf(y, ns.X0, y.length).toFixed(4)} PSNR ${P08.psnr(P08.mseOf(y, ns.X0, y.length)).toFixed(2)} dB`);
    }
    for (const [c1, c2] of [[4, 8], [6, 12], [8, 16]]) for (const skip of [true, false]) {
        const m = P08.makeUNet({ c1, c2, skip }, LM.rng(5));
        const R = P08.makeDenoiseRun({ model: m, lr, cosine: true, B, cap: STEPS, seed: 77, sigma, data });
        let t = 0; const out = [];
        while (R.step < R.cap) { const t0 = process.hrtime.bigint(); P08.denoiseStep(R); t += Number(process.hrtime.bigint() - t0) / 1e6; if ([100, 300, 600, 1000, 1500, 2000, 3000].includes(R.step)) { const e = P08.denoiseEval(m, ns); out.push(`s${R.step} ${P08.psnr(e.mse).toFixed(2)}dB`); } }
        console.log(`  U-Net c${c1}/${c2} ${skip ? 'skip  ' : 'noskip'} ${P08.numParams(m.params)} params ${(t / R.cap).toFixed(2)} ms/step | ${out.join(' · ')}`);
    }
}
if (what === 'deep') {
    const c = +(process.argv[4] || 4), B = +(process.argv[5] || 16), lr = +(process.argv[6] || 3e-3), Ls = (process.argv[7] || '2,8,16').split(',').map(Number);
    const { AG } = LM;
    const cfgs = (process.argv[8] || 'plain,plain+bn,res,res+bn,res+zero,res+bn+zero').split(',');
    for (const L of Ls) for (const cs of cfgs) {
        const res = cs.includes('res'), bn = cs.includes('bn'), zero = cs.includes('zero');
        const m = P08.makeDeep({ L, c, res, bn, zero }, LM.rng(5));
        // gradient norms per block conv at init on one batch
        const X = AG.tensor(B, 256, data.tr.X.subarray(0, B * 256)), y = data.tr.y.subarray(0, B);
        const tr = {}; const loss0 = AG.crossEntropy(P08.deepForward(m, X, tr, true), y); AG.backward(loss0);
        const gn = m.blocks.map((b) => LM.gradNorm(b.conv.w)), an = tr.acts.map((a) => LM.rms(a.data));
        m.params.forEach((p) => { p.grad = null; });
        const R = P08.makeRun({ model: m, lr, cosine: true, B, cap: STEPS, seed: 11, data, evalSets: [] });
        let t = 0, ema = null; const pts = [];
        while (R.step < R.cap) { const t0 = process.hrtime.bigint(); const l = P08.trainStep(R); t += Number(process.hrtime.bigint() - t0) / 1e6; ema = ema == null ? l : 0.98 * ema + 0.02 * l; if ([50, 100, 200, 400, 800, 1200].includes(R.step)) pts.push(`s${R.step} ${ema.toFixed(3)}`); }
        const te = P08.evaluate(m, data.te.X, data.te.y, data.allTe);
        const f = (a) => a.map((v) => v.toExponential(1)).join(' ');
        console.log(`L${String(L).padStart(2)} ${cs.padEnd(12)} ${P08.numParams(m.params)}p ${(t / R.cap).toFixed(2)}ms | loss0 ${loss0.item().toFixed(2)} | ${pts.join(' · ')} | test ${(100 * te.acc).toFixed(1)}%`);
        console.log(`      act rms: ${f([an[0], an[Math.floor(L / 2)], an[L]])}  grad |W| first/mid/last: ${f([gn[0], gn[Math.floor((L - 1) / 2)], gn[L - 1]])}`);
    }
}
if (what === 'deep2') {
    const B = +(process.argv[4] || 8), Ls = (process.argv[5] || '4,8,16,24').split(','), cfgs = (process.argv[6] || 'plain,res+zero').split(',');
    const { AG } = LM;
    for (const Ls_ of Ls) for (const cs of cfgs) {
        const L = +Ls_, res = cs.includes('res'), bn = cs.includes('bn'), zero = cs.includes('zero');
        const m = P08.makeDeep({ L, c: 4, res, bn, zero }, LM.rng(5));
        const X = AG.tensor(32, 256, data.tr.X.subarray(0, 32 * 256)), y = data.tr.y.subarray(0, 32), tr = {};
        AG.backward(AG.crossEntropy(P08.deepForward(m, X, tr, true), y));
        const g = tr.acts.map((a) => LM.rms(a.grad)); m.params.forEach((p) => { p.grad = null; });
        const R = P08.makeRun({ model: m, lr: 3e-3, cosine: true, B, cap: STEPS, seed: 11, data, evalSets: [] });
        let t = 0, ema = null; const pts = [];
        while (R.step < R.cap) { const t0 = process.hrtime.bigint(); const l = P08.trainStep(R); t += Number(process.hrtime.bigint() - t0) / 1e6; ema = ema == null ? l : 0.98 * ema + 0.02 * l; if ([100, 200, 400, 800, 1200].includes(R.step)) pts.push(`s${R.step} ${ema.toFixed(3)}`); }
        const te = P08.evaluate(m, data.te.X, data.te.y, data.allTe);
        console.log(`L${String(L).padStart(2)} ${cs.padEnd(12)} ${(t / R.cap).toFixed(2)}ms | ${pts.join(' · ')} | test ${(100 * te.acc).toFixed(1)}% | dL/dx rms in/mid/out ${[g[0], g[L >> 1], g[L]].map((v) => v.toExponential(1)).join(' ')}`);
    }
}
if (what === 'dead') {   // 24 层普通网络：初始化时与训练后每层的激活 RMS 和「整层为 0」的比例
    const { AG } = LM, L = +(process.argv[4] || 24);
    for (const cs of ['plain', 'res+zero']) {
        const m = P08.makeDeep({ L, c: 4, res: cs.includes('res'), zero: true }, LM.rng(5));
        const probe = (tag) => { const tr = {}; P08.predict(m, data.te.X, 100, tr); const rms = tr.acts.map((a) => LM.rms(a.data)); const dead = tr.acts.map((a) => { let z = 0; for (let i = 0; i < a.data.length; i++) z += a.data[i] === 0; return z / a.data.length; }); console.log(`${cs} ${tag}: act rms ${rms.filter((_, i) => i % 4 === 0 || i === L).map((v) => v.toExponential(1)).join(' ')} | zero frac ${dead.filter((_, i) => i % 4 === 0 || i === L).map((v) => v.toFixed(2)).join(' ')}`); };
        probe('init ');
        const R = P08.makeRun({ model: m, lr: 3e-3, cosine: true, B: 8, cap: STEPS, seed: 11, data, evalSets: [] });
        while (R.step < R.cap) { P08.trainStep(R); if (R.step === 50 || R.step === 200) probe('s' + R.step + ' '); }
        probe('final');
    }
}
if (what === 'frame') {   // 模拟页面每帧的 stage 开销（默认参数）
    const S = { seed, data, sample: 43, dx: 2, dy: 1 };
    const ms = (f) => { const t = process.hrtime.bigint(); const r = f(); return [Number(process.hrtime.bigint() - t) / 1e6, r]; };
    let [t, M] = ms(() => P08.mainRuns(S)); S.main = M; console.log(`mainRuns init ${t.toFixed(1)} ms`);
    const T = { main: 0, aug: 0, unet: 0, deep: 0, equiv: 0 }, F = 200;
    [t] = ms(() => { S.A = P08.augRuns(S, 2); S.E = {}; P08.shiftEvalBegin(S.E, [M.mlp, M.cnn, S.A.mlp, S.A.cnn]); }); console.log(`augRuns init ${t.toFixed(1)} ms`);
    [t] = ms(() => { S.U = P08.unetRuns(S, 0.5); }); console.log(`unetRuns init ${t.toFixed(1)} ms  base noisy ${S.U.base.noisy.toFixed(2)} dB blur ${S.U.base.blur.toFixed(2)} dB`);
    [t] = ms(() => { S.G = P08.deepRuns(S, 16, false); }); console.log(`deepRuns init ${t.toFixed(1)} ms`);
    for (let f = 0; f < F; f++) {
        T.main += ms(() => { P08.tick(M.cnn, 1, P08.EVAL_BUDGET); P08.tick(M.mlp, 1, P08.EVAL_BUDGET); })[0];
        T.aug += ms(() => { P08.tick(S.A.cnn, 1, 0); P08.tick(S.A.mlp, 1, 0); if (!S.E.pending) P08.shiftEvalBegin(S.E, [M.mlp, M.cnn, S.A.mlp, S.A.cnn]); P08.shiftEvalWork(S.E, data, P08.EVAL_BUDGET); })[0];
        T.unet += ms(() => P08.unetTick(S.U, 1))[0];
        T.deep += ms(() => P08.deepTick(S.G, 1))[0];
        T.equiv += ms(() => P08.equivDemo(S))[0];
    }
    for (const k in T) console.log(`${k.padEnd(6)} ${(T[k] / F).toFixed(2)} ms/frame`);
    console.log(`after ${F} frames: CNN test ${P08.pct(P08.last(M.cnn.hist.testA))} (${M.cnn.hist.step.length} pts) MLP ${P08.pct(P08.last(M.mlp.hist.testA))}; U-Net psnr ${S.U.hist.psnr.map((a) => P08.last(a).toFixed(2)).join(' / ')}; deep ema ${S.G.runs.map((R) => R.ema.toFixed(3)).join(' / ')}; shift rounds ${S.E.res ? 'yes' : 'no'}`);
    [t] = ms(() => P08.equivCurve(S)); console.log(`equivCurve ${t.toFixed(1)} ms`);
    let er; [t, er] = ms(() => P08.erfMaps(seed)); console.log(`erfMaps ${t.toFixed(1)} ms  rRand ${er.rRand.map((v) => v.toFixed(2)).join(' ')} | rBox ${er.rBox.map((v) => v.toFixed(2)).join(' ')} | sqrt(4L/3) ${P08.ERF_LS.map((L) => Math.sqrt(4 * L / 3).toFixed(2)).join(' ')}`);
    let fl; [t, fl] = ms(() => P08.deepFlow(S, 16, false)); console.log(`deepFlow ${t.toFixed(1)} ms`); fl.forEach((r, k) => console.log(`  ${['plain', 'res He', 'res zero'][k]}: loss0 ${r.loss.toFixed(2)} act ${[0, 8, 16].map((l) => r.act[l].toExponential(1)).join(' ')} grad ${[0, 8, 16].map((l) => r.grad[l].toExponential(1)).join(' ')}`));
    [t] = ms(() => P08.cnnERF(M.cnn.m, data.te.X.subarray(43 * 256, 44 * 256))); console.log(`cnnERF ${t.toFixed(2)} ms`);
}
if (what === 'useed') {   // U-Net 对初始化种子的敏感性（页面用 seed + 91）
    const sigma = 0.5, ns = P08.noisySet(data, 48, sigma, seed + 17);
    for (const ms of [seed + 91, seed + 92, seed + 93, 5, 6]) {
        const out = [];
        for (const skip of [true, false]) {
            const m = P08.makeUNet({ c1: 4, c2: 8, skip }, LM.rng(ms));
            const R = P08.makeDenoiseRun({ model: m, lr: 3e-3, cosine: true, B: 8, cap: STEPS, seed: seed + 13, sigma, data });
            const pts = [];
            while (R.step < R.cap) { P08.denoiseStep(R); if ([50, 100, 200, 400, 800, 1200].includes(R.step)) pts.push(P08.psnr(P08.denoiseEval(m, ns).mse).toFixed(2)); }
            // dead channels in enc1 / dec1 after training
            const tr = {}; P08.unetForward(m, LM.AG.tensor(16, 256, ns.Xn.subarray(0, 16 * 256)), tr); LM.AG.clearTape();
            const dead = (t, C) => { let n = 0; const HW = t.c / C; for (let c = 0; c < C; c++) { let any = false; for (let i = 0; i < t.r && !any; i++) for (let k = 0; k < HW; k++) if (t.data[i * t.c + c * HW + k] > 0) { any = true; break; } if (!any) n++; } return n; };
            out.push(`${skip ? 'skip' : 'noskip'} ${pts.join(' ')} dead s1 ${dead(tr.s1, 4)}/4 h2 ${dead(tr.h2, 4)}/4`);
        }
        console.log(`seed ${ms}: ${out.join(' | ')}`);
    }
}
if (what === 'uwidth') {
    const sigma = +(process.argv[4] || 0.5), ns = P08.noisySet(data, 48, sigma, seed + 17);
    console.log(`sigma ${sigma}: noisy ${P08.psnr(P08.mseOf(ns.Xn, ns.X0, ns.Xn.length)).toFixed(2)} dB, best blur ${P08.psnr(P08.blurBaseline(ns)).toFixed(2)} dB`);
    for (const [c1, c2] of (process.argv[5] || '4,8;6,12;8,16').split(';').map((s) => s.split(',').map(Number))) for (const skip of [true, false]) {
        const m = P08.makeUNet({ c1, c2, skip }, LM.rng(seed + 91));
        const R = P08.makeDenoiseRun({ model: m, lr: 3e-3, cosine: true, B: 8, cap: STEPS, seed: seed + 13, sigma, data });
        let t = 0; const pts = [];
        while (R.step < R.cap) { const t0 = process.hrtime.bigint(); P08.denoiseStep(R); t += Number(process.hrtime.bigint() - t0) / 1e6; if ([100, 300, 600, 1200].includes(R.step)) pts.push(P08.psnr(P08.denoiseEval(m, ns).mse).toFixed(2)); }
        console.log(`c${c1}/${c2} ${skip ? 'skip  ' : 'noskip'} ${P08.numParams(m.params)}p ${(t / R.cap).toFixed(2)} ms/step | ${pts.join(' ')}`);
    }
}
