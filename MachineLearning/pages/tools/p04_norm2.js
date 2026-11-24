// p04_norm2.js — 第 04 篇：‖X2‖ 尖峰出现在哪些 token 上；不同上下文的大范数向量是否同向
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab, D = 32, NV = 76;
const data = V.encode(LM.world.corpus(LM.rng(4242), 2000)), rand = LM.rng(3);
const rec = [];
for (let w = 0; w < 200; w++) {
  const s = Math.floor(rand() * (data.length - 33)), ids = data.slice(s, s + 32);
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, 32, { trace: true }));
  const x2 = out.trace.layers[1].x2.data, lg = out.logits.data;
  for (let i = 0; i < 32; i++) {
    const v = Array.from(x2.subarray(i * D, (i + 1) * D)), n = Math.hypot(...v);
    const p = LM.softmax(lg, null, 1, i * NV, NV); let H = 0; for (const q of p) if (q > 0) H -= q * Math.log(q);
    rec.push({ tok: V.itos[ids[i]], prev: i ? V.itos[ids[i - 1]] : '', n, H, v });
  }
}
rec.sort((a, b) => b.n - a.n);
const top = rec.slice(0, 320), cnt = {};
top.forEach((r) => { cnt[r.tok] = (cnt[r.tok] || 0) + 1; });
console.log('tokens among top-5% |X2|:', Object.entries(cnt).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([t, c]) => t + ':' + c).join(' '));
// non-。 large norms
const nonDot = rec.filter((r) => r.tok !== '。').slice(0, 200), cnt2 = {};
nonDot.forEach((r) => { cnt2[r.prev + r.tok] = (cnt2[r.prev + r.tok] || 0) + 1; });
console.log('largest non-。 positions (prev+tok):', Object.entries(cnt2).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([t, c]) => t + ':' + c).join(' '), ' norms', nonDot[0].n.toFixed(1), nonDot[199].n.toFixed(1));
// correlation with entropy excluding 。
const rs = rec.filter((r) => r.tok !== '。');
const m = (a) => a.reduce((s, v) => s + v, 0) / a.length, xs = rs.map((r) => r.n), ys = rs.map((r) => r.H), mx = m(xs), my = m(ys);
let sxy = 0, sxx = 0, syy = 0; for (let k = 0; k < xs.length; k++) { sxy += (xs[k] - mx) * (ys[k] - my); sxx += (xs[k] - mx) ** 2; syy += (ys[k] - my) ** 2; }
console.log('corr(|X2|, H) without 。 positions:', (sxy / Math.sqrt(sxx * syy)).toFixed(3), 'n =', rs.length);
// default sentence: cos between X2 at 猫(1), 子(6), 。(18)
const ids = V.encode('熊猫喜欢吃竹子，所以熊猫每天都吃竹子。');
const x2 = AG.noGrad(() => LM.gpt.forward(base, ids, 1, ids.length, { trace: true })).trace.layers[1].x2.data;
const vec = (i) => Array.from(x2.subarray(i * D, (i + 1) * D));
const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let j = 0; j < D; j++) { s += a[j] * b[j]; na += a[j] ** 2; nb += b[j] ** 2; } return s / Math.sqrt(na * nb); };
console.log('cos X2: 猫1–子6', cos(vec(1), vec(6)).toFixed(3), ' 猫1–。18', cos(vec(1), vec(18)).toFixed(3), ' 子6–。18', cos(vec(6), vec(18)).toFixed(3), ' 吃15–竹16', cos(vec(15), vec(16)).toFixed(3));
// what does the dominant direction decode to? u = mean of top-50 normalized
const u = new Float32Array(D); top.slice(0, 50).forEach((r) => { for (let j = 0; j < D; j++) u[j] += r.v[j] / r.n / 50; });
const hf = AG.noGrad(() => AG.layernorm(AG.tensor(1, D, u), base.lnfg, base.lnfb)); const lg = AG.noGrad(() => AG.matmulBT(hf, base.wte)).data;
const p = LM.softmax(lg); console.log('dominant direction decodes to:', LM.topk(p, 8).map((i) => V.itos[i] + p[i].toFixed(2)).join(' '));
