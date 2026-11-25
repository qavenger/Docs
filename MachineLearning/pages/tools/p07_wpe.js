// p07_wpe.js — 第 07 篇：训练后的位置 embedding 有没有结构（相邻位置余弦相似度）；token/位置 embedding 的范数
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const P = base.wpe.data, E = base.wte.data, D = 32;
const nrm = (a, o) => { let s = 0; for (let j = 0; j < D; j++) s += a[o + j] ** 2; return Math.sqrt(s); };
const cos = (a, i, b, j) => { let s = 0; for (let k = 0; k < D; k++) s += a[i * D + k] * b[j * D + k]; return s / (nrm(a, i * D) * nrm(b, j * D)); };
let line = 'cos(wpe_t, wpe_t+1): '; for (let t = 0; t < 31; t++) line += cos(P, t, P, t + 1).toFixed(2) + ' '; console.log(line);
line = 'cos(wpe_0, wpe_t): '; for (let t = 0; t < 32; t++) line += cos(P, 0, P, t).toFixed(2) + ' '; console.log(line);
line = 'cos(wpe_10, wpe_t): '; for (let t = 0; t < 32; t++) line += cos(P, 10, P, t).toFixed(2) + ' '; console.log(line);
line = '|wpe_t|: '; for (let t = 0; t < 32; t++) line += nrm(P, t * D).toFixed(2) + ' '; console.log(line);
const V = LM.vocab; let s = 0, n = 0; for (let v = 4; v < 76; v++) { s += nrm(E, v * D); n++; } console.log('mean |wte| (non-special)', (s / n).toFixed(2), ' specials', [0, 1, 2, 3].map((v) => nrm(E, v * D).toFixed(3)).join(' '));
