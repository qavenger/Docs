// p07_eval_timing.js — 第 07 篇：在页面里测一次参数变化的 stage 开销（validate_doc.mjs --eval 用，函数体）
const S = PL.S, out = {};
const time = (fn, n) => { fn(); const t0 = performance.now(); for (let k = 0; k < n; k++) fn(); return +((performance.now() - t0) / n).toFixed(2); };
out.T = S.T;
out.forwardStageMs = time(() => { S.F = P04.analyse(P04.model(S, S.weights, S.ablate), S.ids); S.FO = P04.analyse(P04.model(S, 'rand', 'none'), S.ids, true); }, 20);
out.ablScanMs = time(() => P04.ablScan(S), 10);
out.generateMs = time(() => P04.generate(S), 10);
out.settleParamMs = time(() => PL._dbg.settle(), 10);
out.pTarget15 = +P04.probsRow(S.F, 15)[S.ids[16]].toFixed(4);
const n = S.F.cpNorm[4]; out.normX2 = { 猫1: +n[1].toFixed(1), 子6: +n[6].toFixed(1), 句末: +n[S.T - 1].toFixed(1) };
out.genText = LM.vocab.decode(P04.generate(S).ids);
return out;
