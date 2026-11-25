// p06_w2v.js — 第 06 篇原型：玩具世界语料上的 word2vec skip-gram + 负采样（逐对 SGD，word2vec.c 风格）
// 用法：node pages/tools/p06_w2v.js [win=2] [neg=5] [lr=0.05] [d=16] [pairs=200000]
const LM = require('../llm_common.js');
const win = +(process.argv[2] || 2), K = +(process.argv[3] || 5), lr0 = +(process.argv[4] || 0.05), d = +(process.argv[5] || 16), NP = +(process.argv[6] || 200000);

const A = LM.world.animals;
const cats = {
    animal: A.map((a) => a.name),
    food: A.map((a) => a.food),
    home: [...new Set(A.map((a) => a.home))],
    color: [...new Set(A.map((a) => a.color))],
    func: ['喜欢', '吃', '住在', '是', '的', '最爱', '它', '所以', '每天', '都', '住着', '今天', '在', '了', '很多', '和', '好朋友'],
};
const words = [], cat = [];
Object.entries(cats).forEach(([c, ws]) => ws.forEach((w) => { words.push(w); cat.push(c); }));
const wid = new Map(words.map((w, i) => [w, i])), V = words.length, maxLen = Math.max(...words.map((w) => w.length));
function segment(s) {   // 正向最大匹配
    const out = []; let i = 0;
    while (i < s.length) {
        let hit = null;
        for (let L = Math.min(maxLen, s.length - i); L >= 1; L--) { const w = s.substr(i, L); if (wid.has(w)) { hit = w; break; } }
        if (hit) { out.push(wid.get(hit)); i += hit.length; } else { if (!'。，'.includes(s[i])) throw new Error('oov ' + s[i] + ' in ' + s); i++; }
    }
    return out;
}
const rand = LM.rng(+(process.argv[7] || 99));
const sents = [];
for (let i = 0; i < 3000; i++) sents.push(segment(LM.world.sentence(rand)));
const freq = new Float64Array(V); let N = 0; sents.forEach((s) => s.forEach((w) => { freq[w]++; N++; }));
console.log('V', V, 'tokens', N, 'top', [...freq].map((f, i) => [words[i], (f / N).toFixed(3)]).sort((a, b) => b[1] - a[1]).slice(0, 8).join(' '));
// all (center, context) pairs, shuffled
const pairs = [];
sents.forEach((s) => { for (let i = 0; i < s.length; i++) for (let j = Math.max(0, i - win); j <= Math.min(s.length - 1, i + win); j++) if (j !== i) pairs.push(s[i] * 1024 + s[j]); });
LM.shuffle(pairs, rand);
console.log('pairs per epoch', pairs.length);
// negative table ∝ f^0.75
const pn = Array.from(freq, (f) => Math.pow(f, 0.75)); const Z = pn.reduce((a, b) => a + b, 0);
const cdf = new Float64Array(V); { let c = 0; for (let i = 0; i < V; i++) { c += pn[i] / Z; cdf[i] = c; } }
const sampleNeg = (r) => { const u = r(); let lo = 0; while (lo < V - 1 && cdf[lo] < u) lo++; return lo; };
// params
const Win = new Float32Array(V * d), Wout = new Float32Array(V * d);
const ir = LM.rng(5); for (let i = 0; i < Win.length; i++) Win[i] = (ir() - 0.5) / d;
const sig = (x) => 1 / (1 + Math.exp(-x));
const gc = new Float32Array(d), nr = LM.rng(7);
let lossAcc = 0, cnt = 0;
function cosStats() {
    const nrm = (i) => { let s = 0; for (let j = 0; j < d; j++) s += Win[i * d + j] ** 2; return Math.sqrt(s) || 1; };
    const cos = (a, b) => { let s = 0; for (let j = 0; j < d; j++) s += Win[a * d + j] * Win[b * d + j]; return s / nrm(a) / nrm(b); };
    const within = {}, across = {};
    ['animal', 'food', 'home', 'color'].forEach((c) => {
        let sw = 0, nw = 0, sa = 0, na = 0;
        for (let a = 0; a < V; a++) if (cat[a] === c) for (let b = 0; b < V; b++) if (b !== a && cat[b] !== 'func') { if (cat[b] === c) { sw += cos(a, b); nw++; } else { sa += cos(a, b); na++; } }
        within[c] = (sw / nw).toFixed(2); across[c] = (sa / na).toFixed(2);
    });
    return { within, across, cos };
}
const t0 = Date.now();
for (let p = 0; p < NP; p++) {
    const key = pairs[p % pairs.length], c = key >> 10, o = key & 1023;
    const lr = lr0 * Math.max(1e-4, 1 - p / NP);
    gc.fill(0);
    const vc = c * d;
    // positive
    let s = 0; for (let j = 0; j < d; j++) s += Win[vc + j] * Wout[o * d + j];
    let g = sig(s) - 1; lossAcc += -Math.log(Math.max(1e-12, sig(s)));
    for (let j = 0; j < d; j++) { gc[j] += g * Wout[o * d + j]; Wout[o * d + j] -= lr * g * Win[vc + j]; }
    for (let k = 0; k < K; k++) {
        const n = sampleNeg(nr); if (n === o) continue;
        let s2 = 0; for (let j = 0; j < d; j++) s2 += Win[vc + j] * Wout[n * d + j];
        const g2 = sig(s2); lossAcc += -Math.log(Math.max(1e-12, 1 - g2));
        for (let j = 0; j < d; j++) { gc[j] += g2 * Wout[n * d + j]; Wout[n * d + j] -= lr * g2 * Win[vc + j]; }
    }
    for (let j = 0; j < d; j++) Win[vc + j] -= lr * gc[j];
    cnt++;
    if ((p + 1) % (NP / 10) === 0) { const st = cosStats(); console.log('pairs', p + 1, 'loss', (lossAcc / cnt).toFixed(3), 'within', JSON.stringify(st.within), 'across', JSON.stringify(st.across)); lossAcc = 0; cnt = 0; }
}
console.log('ms', Date.now() - t0);
const st = cosStats();
const nn = (w) => { const i = wid.get(w); return words.map((x, j) => [x, st.cos(i, j)]).filter((x) => x[0] !== w).sort((a, b) => b[1] - a[1]).slice(0, 5).map((x) => x[0] + x[1].toFixed(2)).join(' '); };
['小猫', '熊猫', '鱼', '竹子', '森林', '白色', '吃', '住在'].forEach((w) => console.log(w, '→', nn(w)));
// analogy: a : food(a) :: b : ?
function vec(i) { return Win.subarray(i * d, i * d + d); }
function analogy(a, b, c, pool) {   // b − a + c
    const q = new Float32Array(d); const va = vec(a), vb = vec(b), vcc = vec(c);
    const n = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    for (let j = 0; j < d; j++) q[j] = vb[j] / n(vb) - va[j] / n(va) + vcc[j] / n(vcc);
    let best = -1, bs = -Infinity;
    for (let i = 0; i < V; i++) { if (i === a || i === b || i === c) continue; if (pool && !pool.includes(i)) continue; let s = 0; for (let j = 0; j < d; j++) s += q[j] * Win[i * d + j]; s /= n(vec(i)); if (s > bs) { bs = s; best = i; } }
    return best;
}
['food', 'home', 'color'].forEach((rel) => {
    let ok = 0, tot = 0, okPool = 0;
    A.forEach((a) => A.forEach((b) => {
        if (a === b || a[rel] === b[rel]) return;
        const r = analogy(wid.get(a.name), wid.get(a[rel]), wid.get(b.name));
        const r2 = analogy(wid.get(a.name), wid.get(a[rel]), wid.get(b.name), words.map((w, i) => i).filter((i) => cat[i] === (rel === 'food' ? 'food' : rel)));
        tot++; if (words[r] === b[rel]) ok++; if (words[r2] === b[rel]) okPool++;
    }));
    console.log('analogy', rel, `${ok}/${tot}`, 'restricted to category', `${okPool}/${tot}`);
});
// topical association: cos(animal, own attribute) vs cos(animal, other animals' attribute of same kind)
['food', 'home', 'color'].forEach((rel) => {
    let own = 0, no = 0, oth = 0, nt = 0;
    A.forEach((a) => A.forEach((b) => { const c = st.cos(wid.get(a.name), wid.get(b[rel])); if (a[rel] === b[rel]) { own += c; no++; } else { oth += c; nt++; } }));
    console.log('assoc', rel, 'own', (own / no).toFixed(2), 'other', (oth / nt).toFixed(2));
});
