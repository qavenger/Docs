// p07_kv.js — 第 07 篇：KV cache 生成 vs 完整前向逐位比较；前向耗时；logit lens；随机初始化模型的行为
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const rnd = LM.gpt.create(W.base.cfg, 1);   // 训练起点（train_base.js 用的同一个 seed）
const V = LM.vocab, VS = 76;
const prompt = V.encode('熊猫喜欢吃竹子，所以');
const steps = [];
const g = LM.gpt.generate(base, prompt, 32, { greedy: true, onStep: (dist, id, lg) => steps.push(Float32Array.from(lg)) });
console.log('greedy:', V.decode(g.ids), 'gen', g.gen.length, 'steps', steps.length);
let maxd = 0;
for (let k = 0; k < steps.length; k++) {
  const T = prompt.length + k, ids = Int32Array.from(g.ids.slice(0, T));
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T));
  const full = out.logits.data.subarray((T - 1) * VS, T * VS);
  let d = 0; for (let j = 0; j < VS; j++) d = Math.max(d, Math.abs(full[j] - steps[k][j]));
  maxd = Math.max(maxd, d);
}
console.log('max |logit diff| kv vs full =', maxd);
// timing
const ids = V.encode('小猫住在屋里。猴子最爱吃香蕉，它住在森林。');
let t0 = performance.now();
for (let r = 0; r < 200; r++) AG.noGrad(() => LM.gpt.forward(base, ids, 1, ids.length, { trace: true }));
console.log('forward T=' + ids.length + ' ms', ((performance.now() - t0) / 200).toFixed(3));
const ids32 = V.encode('小猫喜欢吃鱼，所以小猫每天都吃鱼。小狗住在院子。山里住着熊猫。');
t0 = performance.now();
for (let r = 0; r < 200; r++) AG.noGrad(() => LM.gpt.forward(base, ids32, 1, ids32.length, { trace: true }));
console.log('forward T=' + ids32.length + ' ms', ((performance.now() - t0) / 200).toFixed(3));
t0 = performance.now();
for (let r = 0; r < 50; r++) LM.gpt.generate(base, prompt, 32, { greedy: true });
console.log('generate ms', ((performance.now() - t0) / 50).toFixed(3));
// random model: attention entropy & loss
{
  const s = V.encode('熊猫喜欢吃竹子，所以熊猫每天都吃竹子。'), T = s.length, tg = new Int32Array(T); for (let i = 0; i < T; i++) tg[i] = i + 1 < T ? s[i + 1] : -1;
  for (const [nm, m] of [['base', base], ['rand', rnd]]) {
    const out = AG.noGrad(() => LM.gpt.forward(m, s, 1, T, { trace: true, targets: tg }));
    let mx = 0; const a = out.trace.layers[1].att; for (let h = 0; h < 4; h++) for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) mx = Math.max(mx, a[(h * T + i) * T + j] * (i + 1));
    // norms
    const nrm = (t, i) => { let q = 0; for (let j = 0; j < 32; j++) q += t.data[i * 32 + j] ** 2; return Math.sqrt(q); };
    const tr = out.trace;
    console.log(nm, 'loss', out.loss.item().toFixed(3), 'max(att·(i+1)) L2', mx.toFixed(2), 'norms x0/x1/x2/x1/x2 at pos 15:', [tr.x0, tr.layers[0].x1, tr.layers[0].x2, tr.layers[1].x1, tr.layers[1].x2].map((t) => nrm(t, 15).toFixed(2)).join(' '));
    // logit lens at pos 15 (predict 竹)
    const lens = [tr.x0, tr.layers[0].x1, tr.layers[0].x2, tr.layers[1].x1, tr.layers[1].x2].map((t) => {
      const hf = AG.layernorm(AG.tensor(T, 32, t.data.slice()), m.lnfg, m.lnfb); const lg = AG.matmulBT(hf, m.wte);
      const r = []; for (const i of [9, 15]) { const p = LM.softmax(lg.data.subarray(i * VS, (i + 1) * VS)); const b = LM.argmax(p); r.push(`${V.itos[b]}:${p[b].toFixed(2)} p(tgt)=${p[s[i + 1]].toFixed(2)}`); }
      return r.join(' | ');
    });
    console.log('  logit lens pos9 | pos15:\n   ' + lens.join('\n   '));
  }
}
// raw score scale of the trained heads: std of q·k/√dh (causal pairs)
{
  const s = V.encode('熊猫喜欢吃竹子，所以熊猫每天都吃竹子。'), T = s.length;
  const out = AG.noGrad(() => LM.gpt.forward(base, s, 1, T, { trace: true })), tr = out.trace;
  for (let l = 0; l < 2; l++) { const q = tr.layers[l].q.data, k = tr.layers[l].k.data; const row = [];
    for (let h = 0; h < 4; h++) { let s1 = 0, s2 = 0, n = 0, mx = -1e9; for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) { let d = 0; for (let c = 0; c < 8; c++) d += q[i * 32 + h * 8 + c] * k[j * 32 + h * 8 + c]; d /= Math.sqrt(8); s1 += d; s2 += d * d; n++; mx = Math.max(mx, d); } row.push(`H${h+1} std ${Math.sqrt(s2 / n - (s1 / n) ** 2).toFixed(2)} max ${mx.toFixed(1)}`); }
    console.log('L' + (l + 1), row.join('  '));
  }
}
