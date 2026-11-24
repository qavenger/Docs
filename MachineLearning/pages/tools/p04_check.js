// p04_check.js — 第 04 篇：核对正文里引用的几个数字
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab, D = 32, FF = 128;
const std = (a, i) => { let m = 0; for (let j = 0; j < D; j++) m += a[i * D + j] / D; let v = 0; for (let j = 0; j < D; j++) v += (a[i * D + j] - m) ** 2 / D; return Math.sqrt(v); };
{
  const ids = V.encode('熊猫喜欢吃竹子，所以熊猫每天都吃竹子。'), T = ids.length;
  const tr = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true })).trace;
  const s0 = [], s1 = []; for (let i = 0; i < T; i++) { s0.push(std(tr.x0.data, i)); s1.push(std(tr.layers[1].xin.data, i)); }
  console.log('sigma X0 per token: min', Math.min(...s0).toFixed(3), 'max', Math.max(...s0).toFixed(3), 'mean', (s0.reduce((a, b) => a + b) / T).toFixed(3));
  console.log('sigma L2 input:     min', Math.min(...s1).toFixed(3), 'max', Math.max(...s1).toFixed(3), 'mean', (s1.reduce((a, b) => a + b) / T).toFixed(3));
  let mx = 0; for (let l = 0; l < 2; l++) for (let h = 0; h < 4; h++) mx = Math.max(mx, tr.layers[l].att[(h * T + 15) * T + 5], tr.layers[l].att[(h * T + 15) * T + 6]);
  console.log('max attention from pos 15 to 竹(5)/子(6) over all heads:', mx.toFixed(3));
}
{
  const ids = V.encode('小猫吃鱼，老虎吃肉。'), T = ids.length;
  const tr = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true })).trace;
  const lay = base.layers[1], g = base.lnfg.data, E = base.wte.data, W2 = lay.w2.data;
  const promo = (n) => { const eff = []; for (let v = 4; v < 76; v++) { let s = 0; for (let j = 0; j < D; j++) s += W2[n * D + j] * g[j] * E[v * D + j]; eff.push([V.itos[v], s]); } eff.sort((a, b) => b[1] - a[1]); return eff.slice(0, 2).map((e) => e[0] + e[1].toFixed(2)).join(' '); };
  for (const i of [2, 7]) {
    const f = Array.from(tr.layers[1].f1.data.subarray(i * FF, (i + 1) * FF));
    console.log(`pos ${i} (${V.itos[ids[i]]}) top L2 neurons:`, LM.topk(f, 6).map((n) => `#${n}(${f[n].toFixed(2)})→${promo(n)}`).join('  '));
  }
}
{ // random init: top prob at pos 15
  const rnd = LM.gpt.create(W.base.cfg, 1);
  const ids = V.encode('熊猫喜欢吃竹子，所以熊猫每天都吃竹子。'), T = ids.length;
  const lg = AG.noGrad(() => LM.gpt.forward(rnd, ids, 1, T)).logits.data;
  const p = LM.softmax(lg, null, 1, 15 * 76, 76); console.log('random init pos15 max p', Math.max(...p).toFixed(4));
}
