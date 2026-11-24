// cpp_crosscheck.js — 用页面的 JS 实现（llm_common.js）跑与 cpp/minml_test.cpp 完全相同的设置，对照 C++ 库的输出。
const LM = require('../llm_common.js');
const r = LM.rng(42);
console.log('rng(42) first uniforms:', [r(), r(), r()].map((v) => v.toFixed(9)).join(' '));
const m = LM.gpt.create({ V: 76, T: 32, d: 32, L: 2, H: 4, ff: 128 }, 1);
const B = 2, T = 32, x = new Int32Array(B * T), y = new Int32Array(B * T);
for (let i = 0; i < B * T; i++) { x[i] = (i * 7) % 76; y[i] = (i * 7 + 3) % 76; }
const opt = new LM.AdamW(m.params, { lr: 3e-3, wd: 0.1, b2: 0.95 });
console.log('params', LM.gpt.numParams(m));
for (let it = 0; it < 4; it++) console.log(`step ${it} loss ${LM.gpt.trainStep(m, opt, x, y, B, T, 3e-3).loss.toFixed(6)}`);
const g = LM.gpt.generate(m, [4, 5, 6], 8, { temperature: 1, rand: LM.rng(5) });
console.log('generate:', g.ids.join(' '));
