// p04_specials.js — 第 04 篇：特殊 token 的 embedding 在基座模型里是否还是初始化值（权重共享让它们经由输出层被训练）
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64), init = LM.gpt.create(W.base.cfg, 1), V = LM.vocab, D = 32;
const nrm = (m, v) => Math.hypot(...m.wte.data.subarray(v * D, (v + 1) * D));
let maxDiff = 0; for (let v = 0; v < 4; v++) for (let j = 0; j < D; j++) maxDiff = Math.max(maxDiff, Math.abs(base.wte.data[v * D + j] - init.wte.data[v * D + j]));
console.log('special rows |wte| trained:', [0, 1, 2, 3].map((v) => nrm(base, v).toFixed(2)).join(' '), ' at init:', [0, 1, 2, 3].map((v) => nrm(init, v).toFixed(3)).join(' '), ' max |Δ| vs init:', maxDiff.toFixed(2));
const lg = LM.gpt.nextLogits(base, V.encode('熊猫喜欢吃竹子，所以'));
const p = LM.softmax(lg);
console.log('p(special) after a prompt:', [0, 1, 2, 3].map((v) => V.show(v) + ' ' + p[v].toExponential(1)).join('  '), ' logits:', [0, 1, 2, 3].map((v) => lg[v].toFixed(1)).join(' '), ' max logit', Math.max(...lg).toFixed(1));
const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let j = 0; j < D; j++) { const x = base.wte.data[a * D + j], y = base.wte.data[b * D + j]; s += x * y; na += x * x; nb += y * y; } return s / Math.sqrt(na * nb); };
console.log('cos between special rows: pad-eos', cos(0, 1).toFixed(4), ' user-bot', cos(2, 3).toFixed(4), ' eos-bot', cos(1, 3).toFixed(4));
