// p07_keys.js — 第 07 篇：同一个字在不同位置的 key 有多像（L1 vs L2，默认句子）
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64), V = LM.vocab, D = 32;
const ids = V.encode('熊猫喜欢吃竹子，所以熊猫每天都吃竹子。'), T = ids.length;
const tr = LM.AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true })).trace;
const cos = (a, i, j) => { let s = 0, na = 0, nb = 0; for (let c = 0; c < D; c++) { s += a[i * D + c] * a[j * D + c]; na += a[i * D + c] ** 2; nb += a[j * D + c] ** 2; } return s / Math.sqrt(na * nb); };
for (const [i, j] of [[0, 10], [1, 11], [4, 15], [5, 16]]) console.log(`${V.itos[ids[i]]}${i}-${V.itos[ids[j]]}${j}  K L1 cos ${cos(tr.layers[0].k.data, i, j).toFixed(3)}  K L2 cos ${cos(tr.layers[1].k.data, i, j).toFixed(3)}`);
let s = 0, n = 0; for (let i = 0; i < T; i++) for (let j = 0; j < i; j++) if (ids[i] !== ids[j]) { s += cos(tr.layers[0].k.data, i, j); n++; } console.log('L1 mean cos of different chars', (s / n).toFixed(3));
