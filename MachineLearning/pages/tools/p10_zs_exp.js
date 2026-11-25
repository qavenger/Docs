// zero-shot 实验：提示模板敏感性 + 留出组合（训练时从没出现过的颜色×形状）
const fs = require('fs');
const { LM, P07 } = require('./p10_load.js');
const A = Object.fromEntries(process.argv.slice(2).map((s) => s.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const exclude = A.hold ? new Set(String(A.hold).split(',').map(Number)) : null;
const steps = A.steps ?? 2000;
const tr = P07.clipTrainer({ N: 8, tau: 0.05, lr: 3e-3, steps, seed: A.seed ?? 1, dataSeed: 1234, exclude, hard: !!A.hard });
const t0 = Date.now(); for (let i = 0; i < steps; i++) tr.step(); console.log('trained', steps, 'steps', Date.now() - t0, 'ms', exclude ? 'holdout ' + [...exclude].map(P07.clsName).join(',') : '');
const er = LM.rng(999), test = []; for (let k = 0; k < 12; k++) for (let i = 0; i < 10; i++) test.push(P07.sampleScene(er, { single: true, cls: k }));
const ie = P07.embedImages(tr.clip.img, test).emb;
const tmpls = { '一个{c}的{s}': (c, s) => `一个${c}的${s}`, '{c}的{s}': (c, s) => `${c}的${s}`, '{c}{s}': (c, s) => `${c}${s}`, '一个{s}': null };
for (const [name, f] of Object.entries(tmpls)) {
    if (!f) continue;
    const caps = Array.from({ length: 12 }, (_, k) => f(P07.COLORS[Math.floor(k / 3)].name, P07.SHAPES[k % 3]));
    const te = P07.embedTexts(tr.clip.txt, caps).emb;
    const per = new Array(12).fill(0);
    let ok = 0; test.forEach((s, i) => { let b = 0, bv = -9; for (let k = 0; k < 12; k++) { let v = 0; for (let j = 0; j < 32; j++) v += ie.data[i * 32 + j] * te.data[k * 32 + j]; if (v > bv) { bv = v; b = k; } } if (b === s.cls) { ok++; per[s.cls]++; } });
    console.log(name.padEnd(12), 'zero-shot', (ok / test.length * 100).toFixed(1) + '%', exclude ? ' held-out classes: ' + [...exclude].map((k) => P07.clsName(k) + ' ' + per[k] * 10 + '%').join(', ') : '');
}
