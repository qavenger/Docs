// p04_sents.js — 第 04 篇：候选句子的默认查询位置——那里模型预测什么、哪个头看向哪里
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab;
const S = [
  ['熊猫喜欢吃竹子，所以熊猫每天都吃竹子。', [9, 15]],
  ['小狗喜欢吃骨头，所以小狗每天都吃骨头。', [9, 15]],
  ['熊猫喜欢吃香蕉，所以熊猫每天都吃竹子。', [4, 9, 15]],
  ['老虎喜欢吃竹子，所以老虎每天都吃肉。', [9, 15]],
  ['小猫住在屋里。猴子最爱吃香蕉，它住在森林。', [11, 17]],
  ['今天小兔在草地吃了很多萝卜。', [6, 10]],
  ['小猫吃鱼，老虎吃肉。', [2, 7]],
  ['黑白的熊猫喜欢吃竹子。', [7]],
  ['小鸟和小羊是好朋友。', [3]],
];
for (const [s, qs] of S) {
  const ids = V.encode(s), T = ids.length;
  const tg = new Int32Array(T); for (let i = 0; i < T; i++) tg[i] = i + 1 < T ? ids[i + 1] : -1;
  const out = AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true, targets: tg }));
  console.log(`${s} T=${T} loss=${out.loss.item().toFixed(3)}`);
  for (const q of qs) {
    const p = LM.softmax(out.logits.data.subarray(q * 76, (q + 1) * 76));
    const top = LM.topk(p, 3).map((i) => V.itos[i] + p[i].toFixed(2)).join(' ');
    const heads = [];
    for (let l = 0; l < 2; l++) for (let h = 0; h < 4; h++) {
      let bj = 0; const a = out.trace.layers[l].att; for (let j = 0; j <= q; j++) if (a[(h * T + q) * T + j] > a[(h * T + q) * T + bj]) bj = j;
      heads.push(`L${l + 1}H${h + 1}→${V.itos[ids[bj]]}${bj}(${(a[(h * T + q) * T + bj] * 100).toFixed(0)})`);
    }
    console.log(`  q=${q} '${V.itos[ids[q]]}' next='${q + 1 < T ? V.itos[ids[q + 1]] : '-'}' pred: ${top} | ${heads.join(' ')}`);
  }
}
