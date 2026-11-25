// p07_norm.js — 第 07 篇：最后残差流 ‖X2‖ 的尖峰是否与「下一个字不确定」（预测分布的熵）相关
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab, D = 32, NV = 76;
const data = V.encode(LM.world.corpus(LM.rng(4242), 2000)), rand = LM.rng(3);
const xs = [], ys = [];
for (let w = 0; w < 200; w++) {
  const s = Math.floor(rand() * (data.length - 33)), ids = data.slice(s, s + 32);
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, 32, { trace: true }));
  const x2 = out.trace.layers[1].x2.data, lg = out.logits.data;
  for (let i = 0; i < 32; i++) {
    let n = 0; for (let j = 0; j < D; j++) n += x2[i * D + j] ** 2; n = Math.sqrt(n);
    const p = LM.softmax(lg, null, 1, i * NV, NV); let H = 0; for (const q of p) if (q > 0) H -= q * Math.log(q);
    xs.push(n); ys.push(H);
  }
}
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const mx = mean(xs), my = mean(ys); let sxy = 0, sxx = 0, syy = 0;
for (let k = 0; k < xs.length; k++) { sxy += (xs[k] - mx) * (ys[k] - my); sxx += (xs[k] - mx) ** 2; syy += (ys[k] - my) ** 2; }
console.log('corr(|X2|, entropy) =', (sxy / Math.sqrt(sxx * syy)).toFixed(3), ' n =', xs.length);
const idx = xs.map((v, k) => k).sort((a, b) => xs[b] - xs[a]);
const top = idx.slice(0, Math.floor(xs.length * 0.05)), rest = idx.slice(Math.floor(xs.length * 0.05));
console.log('mean entropy of top-5% norms', mean(top.map((k) => ys[k])).toFixed(3), ' vs others', mean(rest.map((k) => ys[k])).toFixed(3));
console.log('norm quantiles 50/90/99:', [0.5, 0.9, 0.99].map((q) => xs.slice().sort((a, b) => a - b)[Math.floor(q * xs.length)].toFixed(2)).join(' '));
// hypothesis: the large-norm X2 vectors (uncertain positions) point in a shared direction
{
  const vecs = [], norms = [];
  const rand2 = LM.rng(5);
  for (let w = 0; w < 200; w++) {
    const s = Math.floor(rand2() * (data.length - 33)), ids = data.slice(s, s + 32);
    const x2 = AG.noGrad(() => LM.gpt.forward(base, ids, 1, 32, { trace: true })).trace.layers[1].x2.data;
    for (let i = 0; i < 32; i++) { const v = Array.from(x2.subarray(i * D, (i + 1) * D)); vecs.push(v); norms.push(Math.hypot(...v)); }
  }
  const order = norms.map((v, k) => k).sort((a, b) => norms[b] - norms[a]);
  const big = order.slice(0, 300), small = order.slice(order.length - 3000);
  const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let j = 0; j < D; j++) { s += a[j] * b[j]; na += a[j] ** 2; nb += b[j] ** 2; } return s / Math.sqrt(na * nb); };
  const avgCos = (set) => { let s = 0, n = 0; for (let a = 0; a < set.length; a += 3) for (let b = a + 1; b < set.length; b += 7) { s += cos(vecs[set[a]], vecs[set[b]]); n++; } return s / n; };
  console.log('mean pairwise cos among top-300 norm X2:', avgCos(big).toFixed(3), ' among 3000 smallest:', avgCos(small).toFixed(3));
}
