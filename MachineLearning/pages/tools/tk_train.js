// tk_train.js — 离线训练第 19 篇的检索模型（请求 → 工具，对比学习，批内负例 + 同应用难负例）与动作类型分类器；量化后写进 pages/tools_weights.js。
// 与页面同一份代码：tools_catalog.js（工具目录与请求生成）+ tools_common.js（分词、哈希特征、编码器前向）。
// 用法（在 pages/ 下）：node tools/tk_train.js [--steps 4000] [--tag x]
const fs = require('fs');
const path = require('path');
const CAT = require('../tools_catalog.js');
global.TKCAT = CAT;
const TK = require('../tools_common.js');
const LM = require('../llm_common.js');
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const STEPS = +opt('steps', 4000), B = 256, TAU = 0.05, NB = 1 << 14, D = 64, D2 = 64, LR = +opt('lr', 3e-3);
const t0 = Date.now(), rand = LM.rng(19), randn = () => LM.randn(rand);
// ───────────── 数据 ─────────────
const tools = CAT.tools, isNew = tools.map((t) => CAT.isNewTool(t));
const trainTools = tools.map((t, i) => i).filter((i) => !isNew[i]);
const gen = (t, seed, n, held) => CAT.utterances(t, LM.rng(seed), n, held);
const trainU = [], evalSeen = [], evalHeld = [], evalNew = [];
tools.forEach((t, i) => {
    if (isNew[i]) { for (const u of gen(t, 7000 + i, 4, false)) evalNew.push({ text: u.text, y: i }); return; }
    for (const u of gen(t, 1000 + i, 16, false)) trainU.push({ text: u.text, y: i });
    for (const u of gen(t, 5000 + i, 2, false)) evalSeen.push({ text: u.text, y: i });
    for (const u of gen(t, 9000 + i, 2, true)) evalHeld.push({ text: u.text, y: i });
});
const byTool = new Map(); trainU.forEach((u, k) => { if (!byTool.has(u.y)) byTool.set(u.y, []); byTool.get(u.y).push(k); });
const docs = tools.map((t) => TK.toolDoc(t, false));
console.log(`tools ${tools.length} (train ${trainTools.length}, new ${tools.length - trainTools.length}) · train utterances ${trainU.length} · eval seen ${evalSeen.length} / held ${evalHeld.length} / new ${evalNew.length}`);
// 特征预计算（CSR）
function csr(texts) { const off = [0], bk = [], wt = [], nn = []; for (const s of texts) { const f = TK.features(s, NB); let n2 = 0; for (const [b, w] of f) { bk.push(b); wt.push(w); n2 += w * w; } off.push(bk.length); nn.push(1 / Math.sqrt(Math.max(n2, 1))); } return { off: Int32Array.from(off), bk: Int32Array.from(bk), wt: Float32Array.from(wt), inv: Float32Array.from(nn) }; }
const FU = csr(trainU.map((u) => u.text)), FD = csr(docs), FAD = csr(CAT.APPS.map(TK.appDoc)), toolApp = tools.map((t) => CAT.APPS.findIndex((a) => a.ns === t.ns)), NA = CAT.APPS.length, LAMBDA_APP = +opt('lapp', 0.5);
// ───────────── 模型 ─────────────
const E = new Float32Array(NB * D).map(() => randn() * 0.1), b0 = new Float32Array(D), W1 = new Float32Array(D2 * D).map(() => randn() / Math.sqrt(D)), b1 = new Float32Array(D2);
const mE = new Float32Array(NB * D), vE = new Float32Array(NB * D), mW = new Float32Array(D2 * D), vW = new Float32Array(D2 * D), mb0 = new Float32Array(D), vb0 = new Float32Array(D), mb1 = new Float32Array(D2), vb1 = new Float32Array(D2);
function fwd(F, i, cache) {
    const h = new Float32Array(D), a = F.off[i], e = F.off[i + 1], inv = F.inv[i];
    for (let k = a; k < e; k++) { const o = F.bk[k] * D, w = F.wt[k]; for (let j = 0; j < D; j++) h[j] += E[o + j] * w; }
    for (let j = 0; j < D; j++) h[j] = Math.tanh(h[j] * inv + b0[j]);
    const z = new Float32Array(D2); let n2 = 0;
    for (let k = 0; k < D2; k++) { let s = b1[k]; const o = k * D; for (let j = 0; j < D; j++) s += W1[o + j] * h[j]; z[k] = s; n2 += s * s; }
    const nz = Math.sqrt(n2) || 1, zn = z.map((v) => v / nz);
    if (cache) cache.push({ F, i, h, zn, nz });
    return zn;
}
function bwd(c, dzn, gE, gW, gb0, gb1) {
    let d = 0; for (let k = 0; k < D2; k++) d += c.zn[k] * dzn[k];
    const dz = new Float32Array(D2); for (let k = 0; k < D2; k++) dz[k] = (dzn[k] - c.zn[k] * d) / c.nz;
    const dh = new Float32Array(D);
    for (let k = 0; k < D2; k++) { const g = dz[k]; if (!g) continue; gb1[k] += g; const o = k * D; for (let j = 0; j < D; j++) { gW[o + j] += g * c.h[j]; dh[j] += W1[o + j] * g; } }
    const F = c.F, a = F.off[c.i], e = F.off[c.i + 1], inv = F.inv[c.i];
    for (let j = 0; j < D; j++) { dh[j] *= 1 - c.h[j] * c.h[j]; gb0[j] += dh[j]; }
    for (let k = a; k < e; k++) { const b = F.bk[k], w = F.wt[k] * inv; let g = gE.get(b); if (!g) gE.set(b, (g = new Float32Array(D))); for (let j = 0; j < D; j++) g[j] += dh[j] * w; }
}
// 特征 dropout：每个请求随机丢掉 DROP 比例的特征（逼模型别只盯一个词，对没见过的说法更稳）
const DROP = +opt('drop', 0.2);
function dropout(F, i) { const a = F.off[i], e = F.off[i + 1], bk = [], wt = []; let n2 = 0; for (let k = a; k < e; k++) if (rand() >= DROP || e - a <= 2) { bk.push(F.bk[k]); wt.push(F.wt[k]); n2 += F.wt[k] * F.wt[k]; } if (!bk.length) { bk.push(F.bk[a]); wt.push(F.wt[a]); n2 = F.wt[a] * F.wt[a]; } return { off: Int32Array.of(0, bk.length), bk: Int32Array.from(bk), wt: Float32Array.from(wt), inv: Float32Array.of(1 / Math.sqrt(Math.max(n2, 1))) }; }
function adam(p, m, v, g, o, n, lr, t) { const b1c = 1 - 0.9 ** t, b2c = 1 - 0.999 ** t; for (let j = 0; j < n; j++) { const gj = g[j]; m[o + j] = 0.9 * m[o + j] + 0.1 * gj; v[o + j] = 0.999 * v[o + j] + 0.001 * gj * gj; p[o + j] -= (lr * (m[o + j] / b1c)) / (Math.sqrt(v[o + j] / b2c) + 1e-8); } }
// ───────────── 评测：Recall@1/5/10（对全部 1590 个工具）─────────────
function evalSet(set, embDocs, embQ) {
    let r1 = 0, r5 = 0, r10 = 0, mrr = 0;
    for (const q of set) {
        const z = embQ(q.text), sc = new Float32Array(tools.length); for (let i = 0; i < tools.length; i++) { let s = 0; for (let k = 0; k < D2; k++) s += embDocs[i * D2 + k] * z[k]; sc[i] = s; }
        const ys = sc[q.y]; let rank = 1; for (let i = 0; i < tools.length; i++) if (sc[i] > ys) rank++;
        if (rank <= 1) r1++; if (rank <= 5) r5++; if (rank <= 10) r10++; mrr += 1 / rank;
    }
    const n = set.length; return { r1: r1 / n, r5: r5 / n, r10: r10 / n, mrr: mrr / n };
}
function evalBM(set, zh) {
    const I = TK.bm25Build(tools.map((t) => TK.toolDoc(t, zh)));
    let r1 = 0, r5 = 0, r10 = 0, mrr = 0;
    for (const q of set) { const { score } = TK.bm25Score(I, q.text, false), ys = score[q.y]; let rank = 1; for (let i = 0; i < tools.length; i++) if (score[i] > ys || (score[i] === ys && i < q.y && ys === 0)) rank++; if (ys === 0) rank = Math.max(rank, 50); if (rank <= 1) r1++; if (rank <= 5) r5++; if (rank <= 10) r10++; mrr += 1 / rank; }
    const n = set.length; return { r1: r1 / n, r5: r5 / n, r10: r10 / n, mrr: mrr / n };
}
const sub = (set, n, seed) => { const r = LM.rng(seed), a = set.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, n); };
const ES = sub(evalSeen, 1200, 1), EH = sub(evalHeld, 1200, 2), EN = sub(evalNew, 600, 3);
function evalAll() {
    const docsE = new Float32Array(tools.length * D2); for (let i = 0; i < tools.length; i++) docsE.set(fwd(FD, i), i * D2);
    const FQ = new Map(); const embQ = (s) => { let F = FQ.get(s); if (!F) { F = csr([s]); FQ.set(s, F); } return fwd(F, 0); };
    return { seen: evalSet(ES, docsE, embQ), held: evalSet(EH, docsE, embQ), new: evalSet(EN, docsE, embQ) };
}
const bm = { en: { seen: evalBM(ES, false), held: evalBM(EH, false), new: evalBM(EN, false) }, zh: { seen: evalBM(ES, true), held: evalBM(EH, true), new: evalBM(EN, true) } };
const fmt = (r) => `R@1 ${(100 * r.r1).toFixed(1)} R@5 ${(100 * r.r5).toFixed(1)} R@10 ${(100 * r.r10).toFixed(1)}`;
console.log('BM25 (en docs)   seen', fmt(bm.en.seen), '| held', fmt(bm.en.held), '| new', fmt(bm.en.new));
console.log('BM25 (+zh docs)  seen', fmt(bm.zh.seen), '| held', fmt(bm.zh.held), '| new', fmt(bm.zh.new));
// ───────────── 训练 ─────────────
const appsTools = CAT.APPS.map((a) => a.tools.map((t) => t.i).filter((i) => !isNew[i]));
const curve = [], evals = [];
for (let step = 1; step <= STEPS; step++) {
    // 一半的批次随机挑工具，一半按应用成组（同一应用的工具互为难负例）
    let pick = [];
    if (step % 2) { const s = new Set(); while (s.size < B) s.add(trainTools[Math.floor(rand() * trainTools.length)]); pick = [...s]; }
    else { const s = new Set(); while (s.size < B) { const g = appsTools[Math.floor(rand() * appsTools.length)]; for (let k = 0; k < 8 && s.size < B; k++) s.add(g[Math.floor(rand() * g.length)]); } pick = [...s]; }
    const cq = [], cd = [], Q = [], Dn = [];
    for (const i of pick) { const us = byTool.get(i); Q.push(fwd(dropout(FU, us[Math.floor(rand() * us.length)]), 0, cq)); Dn.push(fwd(FD, i, cd)); }
    // 对称 InfoNCE
    const S = new Float32Array(B * B); for (let a = 0; a < B; a++) for (let b = 0; b < B; b++) { let s = 0; const qa = Q[a], db = Dn[b]; for (let k = 0; k < D2; k++) s += qa[k] * db[k]; S[a * B + b] = s / TAU; }
    const G = new Float32Array(B * B); let loss = 0;
    for (let a = 0; a < B; a++) { let m = -Infinity; for (let b = 0; b < B; b++) m = Math.max(m, S[a * B + b]); let z = 0; for (let b = 0; b < B; b++) z += Math.exp(S[a * B + b] - m); loss -= S[a * B + a] - m - Math.log(z); for (let b = 0; b < B; b++) G[a * B + b] += (Math.exp(S[a * B + b] - m) / z - (a === b ? 1 : 0)) / (2 * B); }
    for (let b = 0; b < B; b++) { let m = -Infinity; for (let a = 0; a < B; a++) m = Math.max(m, S[a * B + b]); let z = 0; for (let a = 0; a < B; a++) z += Math.exp(S[a * B + b] - m); loss -= S[b * B + b] - m - Math.log(z); for (let a = 0; a < B; a++) G[a * B + b] += (Math.exp(S[a * B + b] - m) / z - (a === b ? 1 : 0)) / (2 * B); }
    loss /= 2 * B;
    const gE = new Map(), gW = new Float32Array(D2 * D), gb0 = new Float32Array(D), gb1 = new Float32Array(D2);
    // 应用级损失：请求 → 125 个应用描述（全 softmax），让两级路由的第一级也学会
    const ca = [], AE = []; for (let j = 0; j < NA; j++) AE.push(fwd(FAD, j, ca));
    const dAE = AE.map(() => new Float32Array(D2)), dQ = Q.map(() => new Float32Array(D2));
    let lossA = 0;
    for (let a = 0; a < B; a++) {
        const y = toolApp[pick[a]], sc = new Float32Array(NA); let m = -Infinity;
        for (let j = 0; j < NA; j++) { let v = 0; for (let k = 0; k < D2; k++) v += Q[a][k] * AE[j][k]; sc[j] = v / TAU; m = Math.max(m, sc[j]); }
        let z = 0; for (let j = 0; j < NA; j++) z += Math.exp(sc[j] - m); lossA -= sc[y] - m - Math.log(z);
        for (let j = 0; j < NA; j++) { const g = (LAMBDA_APP * (Math.exp(sc[j] - m) / z - (j === y ? 1 : 0))) / B / TAU; if (Math.abs(g) < 1e-9) continue; for (let k = 0; k < D2; k++) { dQ[a][k] += g * AE[j][k]; dAE[j][k] += g * Q[a][k]; } }
    }
    for (let a = 0; a < B; a++) { const dq = dQ[a]; for (let b = 0; b < B; b++) { const g = G[a * B + b] / TAU; if (!g) continue; const db = Dn[b]; for (let k = 0; k < D2; k++) dq[k] += g * db[k]; } bwd(cq[a], dq, gE, gW, gb0, gb1); }
    for (let b = 0; b < B; b++) { const dd = new Float32Array(D2); for (let a = 0; a < B; a++) { const g = G[a * B + b] / TAU; if (!g) continue; const qa = Q[a]; for (let k = 0; k < D2; k++) dd[k] += g * qa[k]; } bwd(cd[b], dd, gE, gW, gb0, gb1); }
    for (let j = 0; j < NA; j++) bwd(ca[j], dAE[j], gE, gW, gb0, gb1);
    const lr = LR * (step < 200 ? step / 200 : 0.5 * (1 + Math.cos((Math.PI * (step - 200)) / (STEPS - 200))));
    for (const [r, g] of gE) adam(E, mE, vE, g, r * D, D, lr, step);
    adam(W1, mW, vW, gW, 0, D2 * D, lr, step); adam(b0, mb0, vb0, gb0, 0, D, lr, step); adam(b1, mb1, vb1, gb1, 0, D2, lr, step);
    if (step % 50 === 0) curve.push([step, +loss.toFixed(4), +(lossA / B).toFixed(4)]);
    if (step % 500 === 0 || step === STEPS) { const ev = evalAll(); evals.push([step, ev]); console.log(`step ${step} loss ${loss.toFixed(3)} · seen ${fmt(ev.seen)} | held ${fmt(ev.held)} | new ${fmt(ev.new)} · ${((Date.now() - t0) / 1000).toFixed(0)} s`); }
}
const trainSec = (Date.now() - t0) / 1000;
// ───────────── 量化：嵌入表 int8（每行一个 f16 缩放），其余 f16 ─────────────
const f32b = new Float32Array(1), u32b = new Uint32Array(f32b.buffer);
function f16(v) {   // 与 llm_common.js 的 toHalf 相同（四舍五入到最近）
    f32b[0] = v; const x = u32b[0], s = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 0xff) - 127 + 15, mt = x & 0x7fffff;
    if (e <= 0) { if (e < -10) return s; mt = (mt | 0x800000) >> (1 - e); return s | ((mt + 0x1000) >> 13); }
    if (e >= 31) return s | 0x7c00;
    const h = s | (e << 10) | (mt >> 13); return (mt & 0x1000) ? h + 1 : h;
}
const f16b64 = (arr) => { const u = new Uint16Array(arr.length); for (let i = 0; i < arr.length; i++) u[i] = f16(arr[i]); return Buffer.from(u.buffer).toString('base64'); };
const q8 = new Int8Array(NB * D), sc = new Float32Array(NB);
for (let r = 0; r < NB; r++) { let m = 0; for (let j = 0; j < D; j++) m = Math.max(m, Math.abs(E[r * D + j])); const s = m / 127 || 1e-8; sc[r] = s; for (let j = 0; j < D; j++) q8[r * D + j] = Math.max(-127, Math.min(127, Math.round(E[r * D + j] / s))); }
const encW = { NB, D, D2, E: Buffer.from(q8.buffer).toString('base64'), Es: f16b64(sc), b0: f16b64(b0), W1: f16b64(W1), b1: f16b64(b1) };
// 量化后的检索质量（与页面完全一致：用 TK.loadEncoder / TK.embed）
const enc = TK.loadEncoder(encW), docsQ = TK.embedAll(enc, docs);
const evQ = { seen: evalSet(ES, docsQ, (s) => TK.embed(enc, s)), held: evalSet(EH, docsQ, (s) => TK.embed(enc, s)), new: evalSet(EN, docsQ, (s) => TK.embed(enc, s)) };
console.log('int8 encoder     seen', fmt(evQ.seen), '| held', fmt(evQ.held), '| new', fmt(evQ.new));
// ───────────── 路由与混合检索（直接用页面的 TK.buildRetriever / TK.search）─────────────
const RT = TK.buildRetriever({ enc: encW });
function evalSearch(set, cfg) {
    let r1 = 0, r5 = 0, r10 = 0, a1 = 0, a3 = 0;
    for (const q of set) {
        const r = RT.search(q.text, { k: 10, mApps: 3, docZh: cfg.docZh, wB: cfg.wB }), top = r.top[cfg.mode], rank = top.indexOf(q.y);
        if (rank === 0) r1++; if (rank >= 0 && rank < 5) r5++; if (rank >= 0 && rank < 10) r10++;
        const ya = toolApp[q.y], ar = r.appTop.indexOf(ya); if (ar === 0) a1++; if (ar >= 0 && ar < 3) a3++;
    }
    const n = set.length; return { r1: r1 / n, r5: r5 / n, r10: r10 / n, app1: a1 / n, app3: a3 / n };
}
const CFGS = { bm25: { mode: 'bm25' }, bm25zh: { mode: 'bm25', docZh: true }, dense: { mode: 'dense' }, hyb1: { mode: 'hybrid', wB: 1 }, hyb05: { mode: 'hybrid', wB: 0.5 }, hyb025: { mode: 'hybrid', wB: 0.25 }, hybzh: { mode: 'hybrid', docZh: true, wB: 1 }, router: { mode: 'router' } };
const searchEval = {};
for (const [k, cfg] of Object.entries(CFGS)) { searchEval[k] = { seen: evalSearch(ES, cfg), held: evalSearch(EH, cfg), new: evalSearch(EN, cfg) }; console.log(k.padEnd(7), 'seen', fmt(searchEval[k].seen), '| held', fmt(searchEval[k].held), '| new', fmt(searchEval[k].new), '| app@1 held', (100 * searchEval[k].held.app1).toFixed(1), 'app@3', (100 * searchEval[k].held.app3).toFixed(1)); }

// ───────────── 动作类型分类器：哈希特征上的多分类逻辑回归 ─────────────
const NBA = 1 << 13, C = CAT.ACTIONS.length;
const actTrain = CAT.actionSamples(LM.rng(31), 8000), actTest = CAT.actionSamples(LM.rng(32), 1500);
const WA = new Float32Array(NBA * C), bA = new Float32Array(C), mWA = new Float32Array(NBA * C), vWA = new Float32Array(NBA * C), mbA = new Float32Array(C), vbA = new Float32Array(C);
const featA = (s) => { const f = TK.features(s, NBA); let n2 = 0; for (const [, w] of f) n2 += w * w; const inv = 1 / Math.sqrt(Math.max(n2, 1)); return [...f].map(([b, w]) => [b, w * inv]); };
const FA = actTrain.map((x) => featA(x.text)), YA = actTrain.map((x) => CAT.ACTIONS.indexOf(x.action));
for (let ep = 0, t = 0; ep < 6; ep++) for (let k = 0; k < FA.length; k += 64) {
    t++; const gW = new Map(), gb = new Float32Array(C);
    for (let i = k; i < Math.min(FA.length, k + 64); i++) { const z = Float32Array.from(bA); for (const [b, w] of FA[i]) for (let c = 0; c < C; c++) z[c] += WA[b * C + c] * w; const p = TK.softmax(z, 1); p[YA[i]] -= 1; for (let c = 0; c < C; c++) gb[c] += p[c] / 64; for (const [b, w] of FA[i]) { let g = gW.get(b); if (!g) gW.set(b, (g = new Float32Array(C))); for (let c = 0; c < C; c++) g[c] += (p[c] * w) / 64; } }
    for (const [r, g] of gW) adam(WA, mWA, vWA, g, r * C, C, 0.05, t); adam(bA, mbA, vbA, gb, 0, C, 0.05, t);
}
const qA = new Int8Array(NBA * C), sA = new Float32Array(NBA);
for (let r = 0; r < NBA; r++) { let m = 0; for (let c = 0; c < C; c++) m = Math.max(m, Math.abs(WA[r * C + c])); const s = m / 127 || 1e-8; sA[r] = s; for (let c = 0; c < C; c++) qA[r * C + c] = Math.round(WA[r * C + c] / s); }
const actW = { NB: NBA, C, Wq: Buffer.from(qA.buffer).toString('base64'), Ws: f16b64(sA), b: f16b64(bA) };
const A = TK.loadAction(actW); let okA = 0; const conf = CAT.ACTIONS.map(() => new Array(C).fill(0));
for (const x of actTest) { const r = TK.classifyAction(A, x.text); if (r.best === x.action) okA++; conf[CAT.ACTIONS.indexOf(x.action)][CAT.ACTIONS.indexOf(r.best)]++; }
console.log(`action classifier test acc ${(100 * okA / actTest.length).toFixed(1)}%`, JSON.stringify(conf));
// ───────────── 工具地图：t-SNE（困惑度 30）把 1590 个工具嵌入画到平面上 ─────────────
function tsne(X, n, d, iters) {
    const P = new Float32Array(n * n), D2m = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let s = 0; for (let k = 0; k < d; k++) { const t = X[i * d + k] - X[j * d + k]; s += t * t; } D2m[i * n + j] = D2m[j * n + i] = s; }
    const target = Math.log(30);
    for (let i = 0; i < n; i++) { let lo = 1e-4, hi = 1e4, beta = 1; for (let it = 0; it < 50; it++) { let z = 0, h = 0; for (let j = 0; j < n; j++) if (j !== i) { const p = Math.exp(-D2m[i * n + j] * beta); z += p; h += beta * D2m[i * n + j] * p; } const H = Math.log(z) + h / z; if (Math.abs(H - target) < 1e-4) break; if (H > target) { lo = beta; beta = hi >= 1e4 ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; } } let z = 0; for (let j = 0; j < n; j++) if (j !== i) { P[i * n + j] = Math.exp(-D2m[i * n + j] * beta); z += P[i * n + j]; } for (let j = 0; j < n; j++) P[i * n + j] /= z; }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = (P[i * n + j] + P[j * n + i]) / (2 * n); P[i * n + j] = P[j * n + i] = Math.max(v, 1e-12); }
    const Y = new Float32Array(2 * n).map(() => randn() * 1e-4), g = new Float32Array(2 * n), u = new Float32Array(2 * n), gains = new Float32Array(2 * n).fill(1), Q = new Float32Array(n * n);
    for (let it = 0; it < iters; it++) {
        const ex = it < 250 ? 12 : 1, mom = it < 250 ? 0.5 : 0.8; let z = 0;
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], q = 1 / (1 + dx * dx + dy * dy); Q[i * n + j] = Q[j * n + i] = q; z += 2 * q; }
        g.fill(0);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) { const q = Q[i * n + j], m = 4 * (ex * P[i * n + j] - q / z) * q; g[2 * i] += m * (Y[2 * i] - Y[2 * j]); g[2 * i + 1] += m * (Y[2 * i + 1] - Y[2 * j + 1]); }
        for (let k = 0; k < 2 * n; k++) { gains[k] = Math.sign(g[k]) !== Math.sign(u[k]) ? gains[k] + 0.2 : Math.max(0.01, gains[k] * 0.8); u[k] = mom * u[k] - 200 * gains[k] * g[k]; Y[k] += u[k]; }
    }
    return Y;
}
const tT = Date.now(), Y = tsne(docsQ, tools.length, D2, 800);
let mx = 0; for (const v of Y) mx = Math.max(mx, Math.abs(v)); const mapQ = Int16Array.from(Y, (v) => Math.round((v / mx) * 32000));
console.log(`t-SNE ${((Date.now() - tT) / 1000).toFixed(0)} s`);
// ───────────── 写出 ─────────────
const bpe = JSON.parse(fs.readFileSync(path.join(__dirname, 'tk_bpe.json'), 'utf8'));
fs.writeFileSync(path.join(__dirname, '..', 'tk_bpe.js'), '// tk_bpe.js — 第 19–21 篇数 token 用的 16k 字节级 BPE（tools/tk_bpe.js 训练），请勿手改。
const TK_BPE = ' + JSON.stringify(bpe) + ';
if (typeof module !== 'undefined' && module.exports) module.exports = TK_BPE;
');
const r4 = (o) => JSON.parse(JSON.stringify(o, (k, v) => (typeof v === 'number' ? +v.toFixed(4) : v)));
const out = `// tools_weights.js — 由 tools/tk_train.js 离线训练生成（检索编码器 int8 量化、动作分类器、16k BPE 规则、工具地图坐标），请勿手改。
const TK_W = ${JSON.stringify({ enc: encW, act: actW, map: Buffer.from(mapQ.buffer).toString('base64'), train: r4({ steps: STEPS, batch: B, tau: TAU, sec: trainSec, NB, D, D2, nTrain: trainU.length, nTools: tools.length, nNew: tools.length - trainTools.length, curve, evals, bm25: bm, int8: evQ, search: searchEval, action: { acc: okA / actTest.length, conf, labels: CAT.ACTIONS } }) })};
if (typeof module !== 'undefined' && module.exports) module.exports = TK_W;
`;
fs.writeFileSync(path.join(__dirname, '..', 'tools_weights.js'), out);
console.log(`wrote tools_weights.js ${(out.length / 1024).toFixed(0)} KB · total ${((Date.now() - t0) / 1000).toFixed(0)} s`);
