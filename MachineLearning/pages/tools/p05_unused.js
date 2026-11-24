// p05_unused.js — 预训练语料里从没出现过的 token（4 个特殊 token + 只出现在问句里的字）的 embedding 怎样演化（用页面代码、默认设置）
const fs = require('fs'), path = require('path'), vm = require('vm');
global.LM = require('../llm_common.js'); global.LM_WEIGHTS = require('../llm_weights.js');
let SPEC = null; global.PipelineLab = { mount: (s) => { SPEC = s; } }; global.PL = { rng: LM.rng, clamp: LM.clamp, cmap: {}, draw: {} };
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', '05_pretraining.lab.js'), 'utf8') + '\n;globalThis.P05 = P05;');
const S = { seed: SPEC.seed, epoch: 1, frame: 0 };
SPEC.params.forEach((p) => { if (!p.action) S[p.id] = p.value; });
['data', 'stats', 'base'].forEach((id) => SPEC.stages.find((s) => s.id === id).run(S, PL));
const cnt = S.D.cnt, unused = []; for (let i = 0; i < LM.vocab.size; i++) if (!cnt[i]) unused.push(i);
console.log('unused tokens:', unused.map((i) => LM.vocab.show(i)).join(' '));
const d = 32, stat = (E) => {
    const rows = unused.map((i) => E.subarray(i * d, (i + 1) * d)), nrm = rows.map((r) => Math.hypot(...r));
    let cmin = 1; for (let a = 0; a < rows.length; a++) for (let b = a + 1; b < rows.length; b++) { let s = 0; for (let j = 0; j < d; j++) s += rows[a][j] * rows[b][j]; cmin = Math.min(cmin, s / (nrm[a] * nrm[b])); }
    let used = 0, nu = 0; for (let i = 0; i < LM.vocab.size; i++) if (cnt[i]) { used += Math.hypot(...E.subarray(i * d, (i + 1) * d)); nu++; }
    return `norm ${Math.min(...nrm).toFixed(2)}–${Math.max(...nrm).toFixed(2)} (used tokens mean ${(used / nu).toFixed(2)}) min pairwise cos ${cmin.toFixed(4)}`;
};
const train = SPEC.stages.find((s) => s.id === 'train');
const marks = [0, 50, 100, 200, 500, 1000, 2000, 5000];
train.run(S, PL);
for (const m of marks) { while (S.run.step < m) train.run(S, PL); console.log(`step ${String(S.run.step).padStart(4)}: ${stat(S.run.m.wte.data)}`); }
console.log('base model:', stat(S.base.wte.data));
// 这些行收到的梯度有多小：第 1000 步时 wte 各行梯度的范数（未出现 vs 出现过的 token）
{
    const R = S.run, d = 32, Rn = (g, i) => Math.hypot(...g.subarray(i * d, (i + 1) * d));
    const S2 = { seed: SPEC.seed, epoch: 2, frame: 0 }; Object.assign(S2, { lr: S.lr, B: S.B, warmup: S.warmup, total: S.total, spf: 1, D: S.D, G: S.G, probes: S.probes });
    const R2 = P05.newRun(S2); while (R2.step < 1000) P05.step(R2, S.D, S.probes, S.G.legit);
    const g = R2.m.wte.grad, un = unused.map((i) => Rn(g, i)), us = []; for (let i = 0; i < LM.vocab.size; i++) if (cnt[i]) us.push(Rn(g, i));
    us.sort((a, b) => a - b);
    console.log(`step 1000 grad row norms: unused ${Math.min(...un).toExponential(1)}–${Math.max(...un).toExponential(1)}, used median ${us[us.length >> 1].toExponential(1)} (min ${us[0].toExponential(1)})`);
}
