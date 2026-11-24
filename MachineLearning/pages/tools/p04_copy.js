// p04_copy.js — 第 04 篇：模型在「所以熊猫每天都吃__」处是在「抄」前文，还是凭名字「回忆」事实？
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab, H = base.cfg.H;
function next(s, k = 5) {
  const ids = V.encode(s), T = ids.length;
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true }));
  const p = LM.softmax(out.logits.data.subarray((T - 1) * 76, T * 76));
  return { top: LM.topk(p, k).map((i) => `${V.itos[i]}:${p[i].toFixed(3)}`).join(' '), out, ids, T, p };
}
const tests = [
  '熊猫喜欢吃竹子，所以熊猫每天都吃',
  '熊猫喜欢吃香蕉，所以熊猫每天都吃',   // counterfactual: copy (香) or recall (竹)?
  '老虎喜欢吃竹子，所以老虎每天都吃',
  '小猫喜欢吃骨头，所以小猫每天都吃',
  '熊猫喜欢吃竹子，所以',
  '老虎喜欢吃竹子，所以',               // copy the name?
  '小猫喜欢吃骨头，所以',
  '熊猫喜欢吃竹子，所以老虎每天都吃',   // name switched in second half
  '猴子最爱吃香蕉，它住在',
  '猴子最爱吃竹子，它住在',
  '熊猫最爱吃香蕉，它住在',
  '小狗和老虎是好朋友。',
  '小猫吃鱼，老虎吃',
  '小猫吃鱼，熊猫吃',
  '今天老虎在森林吃了很多',
  '今天老虎在山里吃了很多',
  '黑白的熊猫喜欢吃',
  '白色的熊猫喜欢吃',
];
for (const s of tests) console.log(s.padEnd(22, '　'), '→', next(s).top);

// which heads look where from the "以" position (predicting the name), averaged over the 8 animals
const A = LM.world.animals;
console.log('\nquery = 以 (predict name). Attention mass per head on [name chars, food chars, 喜欢吃, ，所, 以 self]');
const acc = {};
for (const a of A) {
  const s = `${a.name}喜欢吃${a.food}，所以`, ids = V.encode(s), T = ids.length;
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true })), tr = out.trace;
  const nf = a.food.length, i = T - 1;
  const groups = { name: [0, 1], verb: [2, 3, 4], food: Array.from({ length: nf }, (_, k) => 5 + k), comma: [5 + nf, 6 + nf], self: [i] };
  for (let l = 0; l < 2; l++) for (let h = 0; h < H; h++) {
    const key = `L${l}H${h}`; acc[key] = acc[key] || { name: 0, verb: 0, food: 0, comma: 0, self: 0 };
    for (const g in groups) for (const j of groups[g]) acc[key][g] += tr.layers[l].att[(h * T + i) * T + j] / A.length;
  }
}
for (const k in acc) console.log(k, Object.entries(acc[k]).map(([g, v]) => `${g} ${v.toFixed(2)}`).join('  '));

console.log('\nquery = 2nd 吃 (predict food). Attention mass per head on [1st name, 1st food, 2nd name, 所以, 每天都]');
const acc2 = {};
for (const a of A) {
  const s = `${a.name}喜欢吃${a.food}，所以${a.name}每天都吃`, ids = V.encode(s), T = ids.length;
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true })), tr = out.trace;
  const nf = a.food.length, i = T - 1, c = 5 + nf;   // comma
  const groups = { name1: [0, 1], food1: Array.from({ length: nf }, (_, k) => 5 + k), name2: [c + 3, c + 4], suoyi: [c + 1, c + 2], meitian: [c + 5, c + 6, c + 7], self: [i] };
  for (let l = 0; l < 2; l++) for (let h = 0; h < H; h++) {
    const key = `L${l}H${h}`; acc2[key] = acc2[key] || {};
    for (const g in groups) { acc2[key][g] = acc2[key][g] || 0; for (const j of groups[g]) acc2[key][g] += tr.layers[l].att[(h * T + i) * T + j] / A.length; }
  }
}
for (const k in acc2) console.log(k, Object.entries(acc2[k]).map(([g, v]) => `${g} ${v.toFixed(2)}`).join('  '));

console.log('\nquery = 1st 吃 in "X喜欢吃" (predict food). mass on name / 喜欢 / self');
const acc3 = {};
for (const a of A) {
  const s = `${a.name}喜欢吃`, ids = V.encode(s), T = ids.length;
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true })), tr = out.trace;
  const i = T - 1, groups = { name0: [0], name1: [1], xh: [2, 3], self: [4] };
  for (let l = 0; l < 2; l++) for (let h = 0; h < H; h++) {
    const key = `L${l}H${h}`; acc3[key] = acc3[key] || {};
    for (const g in groups) { acc3[key][g] = acc3[key][g] || 0; for (const j of groups[g]) acc3[key][g] += tr.layers[l].att[(h * T + i) * T + j] / A.length; }
  }
}
for (const k in acc3) console.log(k, Object.entries(acc3[k]).map(([g, v]) => `${g} ${v.toFixed(2)}`).join('  '));
