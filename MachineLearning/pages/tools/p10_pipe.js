// 整条流水线的稳健性测试：CLIP（页面默认配方）→ float16 往返 → LLaVA（页面默认配方），多个种子
const { LM, P07 } = require('./p10_load.js');
const A = Object.fromEntries(process.argv.slice(2).map((s) => s.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
if (A.sq) P07.DATA.sq = A.sq;
const s = A.seed ?? 1;
const ct = P07.clipTrainer(P07.clipRecipe(s));
for (let i = 0; i < 2000; i++) ct.step();
const vit = P07.createViT(P07.CLIP_CFG.img, 1); P07.loadParams(vit.params, P07.exportParams(ct.clip.img.params));
const zs = P07.makeZsTest(), ie = P07.embedImages(vit, zs).emb.data, te = P07.embedTexts(ct.clip.txt, P07.prompts(0)).emb.data;
let ok = 0; zs.forEach((sc, i) => { let b = 0, bv = -9; for (let k = 0; k < 12; k++) { let v = 0; for (let j = 0; j < 32; j++) v += ie[i * 32 + j] * te[k * 32 + j]; if (v > bv) { bv = v; b = k; } } if (b === sc.cls) ok++; });
const lt = P07.llavaTrainer(P07.llavaRecipe(s, { vit }));
const er = LM.rng(777), test = [];
for (let i = 0; i < 48; i++) test.push(P07.sampleScene(er, { single: true }));
for (let i = 0; i < 48; i++) test.push(P07.sampleScene(er, { single: false }));
const res = [];
for (let i = 1; i <= 3000; i++) { lt.step(); if (i % 1000 === 0) { const c = lt.caption(test); res.push((c.filter((g, k) => g.text === test[k].caption).length / 96 * 100).toFixed(1)); } }
console.log(`sq=${P07.DATA.sq} seed=${s}: CLIP zero-shot ${(ok / 48 * 100).toFixed(1)}%  LLaVA exact @1000/2000/3000: ${res.join(' / ')}%`);
