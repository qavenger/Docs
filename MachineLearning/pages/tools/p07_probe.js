// p07_probe.js — 第 07 篇：快速检查基座模型在候选句子上的逐位置损失与 greedy 续写
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab;
const sents = [
  '熊猫喜欢吃竹子，所以熊猫每天都吃竹子。',
  '小狗喜欢吃骨头，所以小狗每天都吃骨头。',
  '猴子最爱吃香蕉，它住在森林。',
  '今天小兔在草地吃了很多萝卜。',
  '小猫吃鱼，老虎吃肉。',
  '黑白的熊猫喜欢吃竹子。',
  '小鸟和小羊是好朋友。小鸟住在树上。',
  '山里住着熊猫。熊猫是黑白的。',
];
for (const s of sents) {
  const ids = V.encode(s), T = ids.length;
  const tg = new Int32Array(T); for (let i = 0; i < T; i++) tg[i] = i + 1 < T ? ids[i + 1] : -1;
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { targets: tg }));
  const rl = out.loss.rowLoss;
  let line = '';
  for (let i = 0; i < T - 1; i++) {
    const lg = out.logits.data.subarray(i * 76, (i + 1) * 76);
    const p = LM.softmax(lg); const top = LM.argmax(p);
    line += `${V.itos[ids[i]]}→${V.itos[ids[i+1]]}:${rl[i].toFixed(2)}${top !== ids[i+1] ? '[' + V.itos[top] + ' ' + p[top].toFixed(2) + ']' : ''} `;
  }
  console.log(s, 'mean', out.loss.item().toFixed(3));
  console.log('  ', line);
}
// greedy continuation of the prompts
for (const pr of ['熊猫喜欢吃竹子，所以', '熊猫喜欢吃竹子，所以熊猫每天都吃', '猴子最爱吃香蕉，它住在', '小羊']) {
  const g = LM.gpt.generate(base, V.encode(pr), 20, { greedy: true });
  console.log('greedy', pr, '=>', V.decode(g.gen));
}
