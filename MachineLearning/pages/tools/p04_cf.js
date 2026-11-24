// p04_cf.js — 第 04 篇：两句反事实句子的逐位置损失（核对 Step 10 的说法）
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab;
for (const s of ['熊猫喜欢吃香蕉，所以熊猫每天都吃竹子。', '老虎喜欢吃竹子，所以老虎每天都吃肉。']) {
  const ids = V.encode(s), T = ids.length, tg = new Int32Array(T); for (let i = 0; i < T; i++) tg[i] = i + 1 < T ? ids[i + 1] : -1;
  const out = LM.AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { targets: tg }));
  const rl = out.loss.rowLoss, arr = [];
  for (let i = 0; i < T - 1; i++) if (rl[i] > 0.5) arr.push(`${V.itos[ids[i]]}→${V.itos[ids[i + 1]]}:${rl[i].toFixed(1)}`);
  console.log(s, 'mean', out.loss.item().toFixed(3), '| >0.5:', arr.join(' '));
}
