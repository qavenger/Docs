// p05_stats.js — 第 05 篇的离线核对：语料 n-gram 交叉熵、生成器的真实熵率、32 字上下文下的贝叶斯最优 loss（逐位置）。
// 页面里同样的计算在 resident / lazy stage 里实时进行；这里用来确认数值与耗时。用法：node pages/tools/p05_stats.js
const LM = require('../llm_common.js');
const V = LM.vocab, W = LM.world;

const t0 = Date.now();
const train = V.encode(W.corpus(LM.rng(2024), 20000));
const val = V.encode(W.corpus(LM.rng(7), 2000));
console.log('train tokens', train.length, 'val tokens', val.length, 'vocab', V.size, `${Date.now() - t0} ms`);

// ── n-gram cross-entropy on val (counts from train, add-k smoothing) ──
function ngramCE(n, k) {
    const Vn = V.size, cnt = new Map(), ctx = new Map();
    const key = (a, i, m) => { let s = ''; for (let j = 0; j < m; j++) s += a[i + j] + ','; return s; };
    for (let i = 0; i + n <= train.length; i++) { const c = key(train, i, n - 1), f = c + train[i + n - 1]; cnt.set(f, (cnt.get(f) || 0) + 1); ctx.set(c, (ctx.get(c) || 0) + 1); }
    let s = 0, m = 0;
    for (let i = 0; i + n <= val.length; i++) { const c = key(val, i, n - 1), f = c + val[i + n - 1]; const p = ((cnt.get(f) || 0) + k) / ((ctx.get(c) || 0) + k * Vn); s -= Math.log(p); m++; }
    return s / m;
}
for (const n of [1, 2, 3, 4]) console.log(`${n}-gram val CE`, ngramCE(n, 0.01).toFixed(4));
// plug-in entropies on train
{
    const c1 = new Float64Array(V.size); for (const t of train) c1[t]++;
    let H1 = 0; for (const c of c1) if (c) { const p = c / train.length; H1 -= p * Math.log(p); }
    console.log('unigram plug-in H1', H1.toFixed(4), 'ln V', Math.log(V.size).toFixed(4));
}

// ── generator: enumerate all distinct sentences with their probabilities ──
const A = W.animals, sents = [];
for (let ai = 0; ai < 8; ai++) for (let bi = 0; bi < 8; bi++) {
    // world.sentence: a = A[r1], b = A[r2]; if b === a → b = A[(a+1)%8]; k = r3
    const pb = bi === ai ? 0 : bi === (ai + 1) % 8 ? 2 / 8 : 1 / 8;
    for (let k = 0; k < 10; k++) {
        const usesB = k === 7 || k === 8;
        if (!usesB && bi !== (ai + 1) % 8) continue;   // b irrelevant: count once (use b = a+1 slot, prob 1)
        const p = (1 / 8) * (1 / 10) * (usesB ? pb : 1);
        if (p === 0) continue;
        sents.push({ s: W.templates[k](A[ai], A[bi]), p, k });
    }
}
const strMap = new Map(); sents.forEach((x) => strMap.set(x.s, (strMap.get(x.s) || 0) + x.p));
console.log('distinct sentences', strMap.size, 'Σp', [...strMap.values()].reduce((a, b) => a + b, 0).toFixed(6));
const S = [...strMap.entries()].map(([s, p]) => ({ s, p, ids: V.encode(s) }));
let Hs = 0, EL = 0; S.forEach((x) => { Hs -= x.p * Math.log(x.p); EL += x.p * x.ids.length; });
console.log(`H(sentence) = ${Hs.toFixed(4)} nats, E[len] = ${EL.toFixed(3)} tokens → entropy rate ${(Hs / EL).toFixed(4)} nats/token`);

// ── Bayes-optimal next-token loss given only the window x_0..x_t (forward algorithm over (sentence, offset) states) ──
// state (s, j): currently at char j of sentence s. Stationary start: P(s, j) = p_s / E[len].
const NS = S.length, off = new Int32Array(NS + 1); S.forEach((x, i) => { off[i + 1] = off[i] + x.ids.length; });
const NST = off[NS], sid = new Int32Array(NST), emit = new Int32Array(NST), last = new Uint8Array(NST), pS = new Float64Array(NS);
S.forEach((x, i) => { pS[i] = x.p; for (let j = 0; j < x.ids.length; j++) { sid[off[i] + j] = i; emit[off[i] + j] = x.ids[j]; last[off[i] + j] = j === x.ids.length - 1 ? 1 : 0; } });
// sentence-start distribution of first chars
const startP = new Float64Array(V.size); S.forEach((x, i) => { startP[x.ids[0]] += x.p; });
console.log('states', NST);
function bayesWindow(x, Tn, perPos) {
    // alpha over states after observing x_0..x_t (normalised)
    let alpha = new Float64Array(NST);
    for (let st = 0; st < NST; st++) if (emit[st] === x[0]) alpha[st] = pS[sid[st]];
    let z = 0; for (let st = 0; st < NST; st++) z += alpha[st]; for (let st = 0; st < NST; st++) alpha[st] /= z;
    const pred = new Float64Array(V.size);
    let tot = 0;
    for (let t = 0; t < Tn; t++) {
        // predictive distribution of x_{t+1}
        pred.fill(0); let endMass = 0;
        for (let st = 0; st < NST; st++) { const a = alpha[st]; if (!a) continue; if (last[st]) endMass += a; else pred[emit[st + 1]] += a; }
        for (let c = 0; c < V.size; c++) pred[c] += endMass * startP[c];
        const nx = x[t + 1], pn = pred[nx];
        const l = -Math.log(pn); tot += l; if (perPos) perPos[t] += l;
        // update alpha with observation nx
        const na = new Float64Array(NST);
        for (let st = 0; st < NST; st++) { const a = alpha[st]; if (!a || last[st]) continue; if (emit[st + 1] === nx) na[st + 1] += a; }
        if (endMass > 0) for (let i = 0; i < NS; i++) if (S[i].ids[0] === nx) na[off[i]] += endMass * pS[i];
        z = 0; for (let st = 0; st < NST; st++) z += na[st]; for (let st = 0; st < NST; st++) na[st] /= z;
        alpha = na;
    }
    return tot / Tn;
}
{
    const T = 32, nWin = 2000, r = LM.rng(5), per = new Float64Array(T);
    const t1 = Date.now(); let s = 0;
    for (let w = 0; w < nWin; w++) { const st = Math.floor(r() * (val.length - T - 1)); s += bayesWindow(val.subarray(st, st + T + 1), T, per); }
    console.log(`Bayes-optimal loss (T=32 context, ${nWin} val windows): ${(s / nWin).toFixed(4)}  (${Date.now() - t1} ms)`);
    console.log('per position:', Array.from(per, (v) => (v / nWin).toFixed(2)).join(' '));
}
// base model for comparison
try {
    const WT = require('../llm_weights.js');
    const base = LM.gpt.loadWeights(LM.gpt.create(WT.base.cfg, 1), WT.base.b64);
    const T = 32, nWin = 256, r = LM.rng(5), per = new Float64Array(T);
    let s = 0;
    for (let w = 0; w < nWin; w += 16) {
        const x = new Int32Array(16 * T), y = new Int32Array(16 * T);
        for (let b = 0; b < 16; b++) { const st = Math.floor(r() * (val.length - T - 1)); for (let t = 0; t < T; t++) { x[b * T + t] = val[st + t]; y[b * T + t] = val[st + t + 1]; } }
        const out = LM.AG.noGrad(() => LM.gpt.forward(base, x, 16, T, { targets: y }));
        s += out.loss.item() * 16; for (let i = 0; i < 16 * T; i++) per[i % T] += out.loss.rowLoss[i];
    }
    console.log(`base model val loss on the same windows: ${(s / nWin).toFixed(4)}`);
    console.log('per position:', Array.from(per, (v) => (v / nWin).toFixed(2)).join(' '));
} catch (e) { console.log('no weights', e.message); }
