// p05_eval_timing.js — validate_doc --eval 用：在页面里测每个训练步 / 评估块的真实耗时
const S = PipelineLab.S, R = S.run;
const out = { step: R.step };
const t = [];
for (let k = 0; k < 40; k++) { const a = performance.now(); P05.step(R, S.D, S.probes, S.G.legit); t.push(performance.now() - a); }
out.stepMs = t.map((x) => +x.toFixed(1));
out.mean = +(t.reduce((s, x) => s + x, 0) / t.length).toFixed(2);
out.rMs = +R.ms.toFixed(2);
// raw trainStep without eval chunk
const m = LM.gpt.create(P05.CFG, 3), opt = new LM.AdamW(m.params, { lr: 1e-3 }), b = LM.gpt.batch(S.D.train, 4, 32, LM.rng(1));
const t2 = [];
for (let k = 0; k < 30; k++) { const a = performance.now(); LM.gpt.trainStep(m, opt, b.x, b.y, 4, 32, 1e-3, { clip: 1 }); t2.push(performance.now() - a); }
out.rawTrainStep = +(t2.slice(5).reduce((s, x) => s + x, 0) / 25).toFixed(2);
const t3 = [];
for (let k = 0; k < 30; k++) { const a = performance.now(); LM.AG.noGrad(() => LM.gpt.forward(m, S.D.vx.subarray(0, 64), 2, 32, { targets: S.D.vy.subarray(0, 64) })); t3.push(performance.now() - a); }
out.valChunk2 = +(t3.slice(5).reduce((s, x) => s + x, 0) / 25).toFixed(2);
out.cores = navigator.hardwareConcurrency;
return out;
