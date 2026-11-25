// p08_decode.js — 原型：基座模型在几个提示上的下一个字分布、温度扫描（正确率 / 多样性）、各统计量耗时。
const LM = require('../llm_common.js');
const WT = require('../llm_weights.js');
const V = LM.vocab, W = LM.world;
const base = LM.gpt.loadWeights(LM.gpt.create(WT.base.cfg, 1), WT.base.b64);
for (const pr of ['。', '今天', '小猫喜欢吃', '熊猫和', '森林住着', '白色的', '小狗吃骨头，', '所以']) {
    const lg = LM.gpt.nextLogits(base, Array.from(V.encode(pr)));
    const p = LM.softmax(lg, new Float32Array(V.size));
    const top = LM.topk(p, 10).map((i) => `${V.show(i)}:${p[i].toFixed(3)}`).join(' ');
    let H = 0; for (const q of p) if (q > 0) H -= q * Math.log(q);
    console.log(pr.padEnd(8), 'H', H.toFixed(2), top);
}
// legit sentences
const A = W.animals, legit = new Set();
for (let ai = 0; ai < 8; ai++) for (let k = 0; k < 10; k++) for (let bi = 0; bi < 8; bi++) if (bi !== ai) legit.add(W.templates[k](A[ai], A[bi]));
console.log('legit', legit.size);
function score(texts) {
    let tot = 0, ok = 0; const distinct = new Set();
    texts.forEach((s) => { const parts = s.split('。'); parts.slice(0, -1).forEach((x, i) => { if (i === 0 && !s.startsWith('')) return; tot++; distinct.add(x); if (legit.has(x + '。')) ok++; }); });
    return { tot, ok: ok / Math.max(1, tot), div: distinct.size / Math.max(1, tot), distinct: distinct.size };
}
const t0 = Date.now(); let nTok = 0;
for (const temp of [0.1, 0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]) {
    const texts = [];
    for (let k = 0; k < 12; k++) { const g = LM.gpt.generate(base, V.encode('。'), 31, { temperature: temp, rand: LM.rng(777 + k) }); nTok += g.gen.length; texts.push(V.decode(g.gen)); }
    const sc = score(texts);
    console.log(`T=${temp} sentences ${sc.tot} correct ${sc.ok.toFixed(2)} distinct ${sc.distinct} div ${sc.div.toFixed(2)} | ${texts[0]}`);
}
console.log('gen', nTok, 'tokens in', Date.now() - t0, 'ms →', ((Date.now() - t0) / nTok * 1000).toFixed(0), 'µs/token');
// n-gram timing
const train = V.encode(W.corpus(LM.rng(2024), 20000)), val = V.encode(W.corpus(LM.rng(7), 2000));
function ngramCE(n, k) {
    const Vn = V.size, cnt = new Map(), ctx = new Map();
    const keyAt = (a, i, m) => { let s = 0; for (let j = 0; j < m; j++) s = s * Vn + a[i + j]; return s; };
    for (let i = 0; i + n <= train.length; i++) { const c = keyAt(train, i, n - 1), f = c * Vn + train[i + n - 1]; cnt.set(f, (cnt.get(f) || 0) + 1); ctx.set(c, (ctx.get(c) || 0) + 1); }
    let s = 0, m = 0;
    for (let i = 0; i + n <= val.length; i++) { const c = keyAt(val, i, n - 1), f = c * Vn + val[i + n - 1]; s -= Math.log(((cnt.get(f) || 0) + k) / ((ctx.get(c) || 0) + k * Vn)); m++; }
    return { ce: s / m, ctxN: ctx.size };
}
const t1 = Date.now();
for (const n of [1, 2, 3, 4, 5]) { const r = ngramCE(n, 0.01); console.log(`${n}-gram CE ${r.ce.toFixed(4)} contexts ${r.ctxN}`); }
console.log('ngram time', Date.now() - t1, 'ms');
// windowed n-gram (context truncated to the 32-token window) on 32 fixed val windows
const vb = LM.gpt.batch(val, 32, 32, LM.rng(5));
function ngramWin(n, k) {
    const Vn = V.size, cnt = [], ctx = [];
    for (let o = 1; o <= n; o++) { const c1 = new Map(), c0 = new Map(); for (let i = 0; i + o <= train.length; i++) { let c = 0; for (let j = 0; j < o - 1; j++) c = c * Vn + train[i + j]; const f = c * Vn + train[i + o - 1]; c1.set(f, (c1.get(f) || 0) + 1); c0.set(c, (c0.get(c) || 0) + 1); } cnt.push(c1); ctx.push(c0); }
    let s = 0, m = 0;
    for (let b = 0; b < 32; b++) for (let t = 0; t < 32; t++) {
        const o = Math.min(n, t + 2); // available context = t+1 tokens → order ≤ t+2
        let c = 0; for (let j = t + 2 - o; j <= t; j++) c = c * Vn + vb.x[b * 32 + j];
        const f = c * Vn + vb.y[b * 32 + t];
        s -= Math.log(((cnt[o - 1].get(f) || 0) + k) / ((ctx[o - 1].get(c) || 0) + k * Vn)); m++;
    }
    return s / m;
}
for (const n of [1, 2, 3, 4]) console.log(`${n}-gram on the 32 fixed val windows (window-truncated context): ${ngramWin(n, 0.01).toFixed(4)}`);
const vl = LM.AG.noGrad(() => LM.gpt.forward(base, vb.x, 32, 32, { targets: vb.y }).loss.item());
console.log('base val on the 32 fixed windows', vl.toFixed(4));
