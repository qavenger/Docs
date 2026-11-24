// p04_ffn.js — 第 04 篇：FFN 神经元分析——第 2 层 MLP 在「X喜欢吃」的「吃」处是否有「动物专属」神经元？它们的输出向量推高哪个 token？
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab, A = LM.world.animals, D = 32, FF = 128, VS = 76;
const acts = [];   // [animal][neuron] at the 吃 position, layer 2
for (const a of A) {
  const ids = V.encode(`${a.name}喜欢吃`), T = ids.length;
  const tr = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true })).trace;
  acts.push(Array.from(tr.layers[1].f1.data.subarray((T - 1) * FF, T * FF)));
}
// selectivity: max over animals minus mean of the others
const lay = base.layers[1], g = base.lnfg.data, E = base.wte.data, W2 = lay.w2.data;
const rows = [];
for (let n = 0; n < FF; n++) {
  const col = acts.map((r) => r[n]); const mx = Math.max(...col), am = col.indexOf(mx);
  const others = col.filter((_, i) => i !== am); const mo = Math.max(...others);
  // direct logit effect of this neuron's value vector (ignoring LN scaling): (W2[n] ⊙ γ) · wte[v]
  const eff = new Float32Array(VS); for (let v = 0; v < VS; v++) { let s = 0; for (let j = 0; j < D; j++) s += W2[n * D + j] * g[j] * E[v * D + j]; eff[v] = s; }
  const top = LM.topk(eff, 3).map((v) => V.itos[v] + eff[v].toFixed(1)).join(' ');
  rows.push({ n, am, mx, gap: mx - mo, top });
}
rows.sort((a, b) => b.gap - a.gap);
console.log('most animal-selective L2 neurons at 吃 (X喜欢吃):');
rows.slice(0, 14).forEach((r) => console.log(`#${r.n} fires for ${A[r.am].name} (${r.mx.toFixed(2)}, gap ${r.gap.toFixed(2)})  promotes ${r.top}  food=${A[r.am].food}`));
// sparsity stats on default sentence
const s = V.encode('熊猫喜欢吃竹子，所以熊猫每天都吃竹子。'), T = s.length;
const tr = AG.noGrad(() => LM.gpt.forward(base, s, 1, T, { trace: true })).trace;
for (let l = 0; l < 2; l++) {
  const f = tr.layers[l].f1.data; let neg = 0, small = 0, big = 0, mn = 1e9, mx = -1e9;
  for (let i = 0; i < f.length; i++) { if (f[i] < 0) neg++; if (Math.abs(f[i]) < 0.05) small++; if (f[i] > 0.5) big++; mn = Math.min(mn, f[i]); mx = Math.max(mx, f[i]); }
  console.log(`L${l + 1} f1: frac<0 ${(neg / f.length).toFixed(2)}  |f|<0.05 ${(small / f.length).toFixed(2)}  f>0.5 ${(big / f.length).toFixed(3)} (${(big / T).toFixed(1)} per token)  range [${mn.toFixed(2)}, ${mx.toFixed(2)}]`);
}
// per-neuron: at pos 15 (2nd 吃), top-5 L2 neurons and what they promote
const f15 = Array.from(tr.layers[1].f1.data.subarray(15 * FF, 16 * FF));
LM.topk(f15, 6).forEach((n) => { const r = rows.find((q) => q.n === n); console.log(`pos15 L2 #${n} act ${f15[n].toFixed(2)} promotes ${r.top}`); });
