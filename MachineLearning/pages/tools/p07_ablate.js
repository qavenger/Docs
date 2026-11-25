// p07_ablate.js — 第 07 篇：逐个消融注意力头（把 W_O 中该头对应的 dh 行置零）/ MLP，看关键预测的概率变化
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab, H = base.cfg.H, d = base.cfg.d, dh = d / H;
function ablated(spec) {   // spec: list of 'L1H0' / 'L0M'
  const m = LM.gpt.clone(base);
  for (const s of spec) {
    const l = +s[1];
    if (s[2] === 'H') { const h = +s[3]; const wo = m.layers[l].wo.data; for (let r = h * dh; r < (h + 1) * dh; r++) for (let c = 0; c < d; c++) wo[r * d + c] = 0; }
    else { m.layers[l].w2.data.fill(0); m.layers[l].b2.data.fill(0); }
  }
  return m;
}
function pnext(m, s, tok) {
  const ids = V.encode(s), T = ids.length;
  const out = AG.noGrad(() => LM.gpt.forward(m, ids, 1, T));
  const p = LM.softmax(out.logits.data.subarray((T - 1) * 76, T * 76));
  const top = LM.argmax(p);
  return { p: p[V.stoi.get(tok)], top: V.itos[top], pt: p[top] };
}
const cases = [
  ['熊猫喜欢吃竹子，所以熊猫每天都吃', '竹'],
  ['猴子喜欢吃香蕉，所以猴子每天都吃', '香'],
  ['小狗喜欢吃骨头，所以小狗每天都吃', '骨'],
  ['熊猫喜欢吃竹子，所以', '熊'],
  ['小狗喜欢吃骨头，所以小', '狗'],
  ['熊猫喜欢吃', '竹'],
  ['小羊喜欢吃', '青'],
  ['猴子最爱吃香蕉，它住在', '森'],
  ['小猫住在屋里。猴子最爱吃香蕉，它住在', '森'],
  ['小兔住在', '草'],
];
const comps = ['none', 'L0H0', 'L0H1', 'L0H2', 'L0H3', 'L1H0', 'L1H1', 'L1H2', 'L1H3', 'L0M', 'L1M', 'L1H0+L1H1'];
const models = comps.map((c) => (c === 'none' ? base : ablated(c.split('+'))));
console.log('p(correct) with each component ablated'.padEnd(40), comps.map((c) => c.padStart(9)).join(''));
for (const [s, t] of cases) {
  const row = models.map((m) => { const r = pnext(m, s, t); return (r.p.toFixed(2) + (r.p < 0.5 ? '(' + r.top + ')' : '')).padStart(9); });
  console.log((s + '→' + t).padEnd(40 - (s.length + 2)), row.join(''));
}
// "小猫喜欢吃骨头，所以小" — name or food-implied?
for (const s of ['小猫喜欢吃骨头，所以小', '小狗喜欢吃鱼，所以小', '老虎喜欢吃竹子，所以', '老虎喜欢吃竹子，所以老']) {
  const ids = V.encode(s), T = ids.length; const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T));
  const p = LM.softmax(out.logits.data.subarray((T - 1) * 76, T * 76));
  console.log(s, '→', LM.topk(p, 4).map((i) => V.itos[i] + ':' + p[i].toFixed(3)).join(' '));
}
