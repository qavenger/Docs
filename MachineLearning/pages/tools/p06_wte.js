// p06_wte.js — 第 06 篇原型：基座模型（字符级）wte / wpe 里有没有结构
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const m = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = m.cfg.V, d = m.cfg.d, E = m.wte.data, P = m.wpe.data;
const voc = LM.vocab;
const nrm = (X, i) => { let s = 0; for (let j = 0; j < d; j++) s += X[i * d + j] ** 2; return Math.sqrt(s); };
const cos = (X, a, b) => { let s = 0; for (let j = 0; j < d; j++) s += X[a * d + j] * X[b * d + j]; return s / (nrm(X, a) * nrm(X, b) + 1e-12); };
console.log('vocab', voc.itos.join(''));
console.log('norms', Array.from({ length: V }, (_, i) => voc.show(i) + nrm(E, i).toFixed(2)).join(' '));
const nn = (ch) => { const i = voc.stoi.get(ch); return Array.from({ length: V }, (_, j) => [voc.show(j), cos(E, i, j)]).filter((x, j) => j !== i).sort((a, b) => b[1] - a[1]).slice(0, 6).map((x) => x[0] + x[1].toFixed(2)).join(' '); };
['猫', '狗', '鱼', '肉', '骨', '竹', '森', '白', '色', '吃', '喜', '。', '小', '在', '熊', '虎'].forEach((c) => console.log(c, '→', nn(c)));
// categories of chars: second char of animal names, first char of foods, etc.
const A = LM.world.animals;
const groups = {
    animal: [...new Set(A.flatMap((a) => [...a.name]))],
    food: [...new Set(A.flatMap((a) => [...a.food]))],
    home: [...new Set(A.flatMap((a) => [...a.home]))],
    color: [...new Set(A.flatMap((a) => [...a.color]))],
};
console.log(groups);
Object.entries(groups).forEach(([g, chs]) => {
    let sw = 0, nw = 0, sa = 0, na = 0;
    const ids = chs.map((c) => voc.stoi.get(c));
    for (const a of ids) for (let b = 4; b < V; b++) { if (b === a) continue; if (ids.includes(b)) { sw += cos(E, a, b); nw++; } else { sa += cos(E, a, b); na++; } }
    console.log(g, 'within', (sw / nw).toFixed(3), 'others', (sa / na).toFixed(3));
});
// wpe: cos sim of neighbors
let s1 = 0, s5 = 0, s15 = 0;
for (let i = 0; i < 31; i++) s1 += cos(P, i, i + 1); for (let i = 0; i < 27; i++) s5 += cos(P, i, i + 5); for (let i = 0; i < 17; i++) s15 += cos(P, i, i + 15);
console.log('wpe cos: Δ1', (s1 / 31).toFixed(3), 'Δ5', (s5 / 27).toFixed(3), 'Δ15', (s15 / 17).toFixed(3));
console.log('wpe norms', Array.from({ length: 32 }, (_, i) => nrm(P, i).toFixed(2)).join(' '));
console.log('wte mean norm', (Array.from({ length: V - 4 }, (_, i) => nrm(E, i + 4)).reduce((a, b) => a + b) / (V - 4)).toFixed(3), 'special', [0, 1, 2, 3].map((i) => nrm(E, i).toFixed(3)).join(' '));
const row = (X, i) => Array.from({ length: 32 }, (_, j) => cos(X, i, j).toFixed(2)).join(' ');
console.log('wpe row 0', row(P, 0)); console.log('wpe row 10', row(P, 10)); console.log('wpe row 20', row(P, 20));
