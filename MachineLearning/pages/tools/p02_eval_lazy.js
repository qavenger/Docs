// p02_eval_lazy.js — validate_doc.mjs --eval 的函数体：让 Step 09 / 11 的 lazy 训练跑到上限，截下预览、读结果，并测各训练 stage 的每帧耗时
const S = PipelineLab.S, D = PipelineLab._dbg;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const grab = (id) => { const pv = D.previews.get(id); pv.render(pv.canvas, S, PipelineLab); return pv.canvas.toDataURL('image/png'); };
const out = { imgs: {} };
const t0 = performance.now();
while ((S.mlp.step < S.mlp.cap || S.soft.step < S.soft.cap) && performance.now() - t0 < 40000) await sleep(500);
out.main = { soft: [S.soft.step, S.soft.hist.trainA.at(-1), S.soft.hist.testA.at(-1)], mlp: [S.mlp.step, S.mlp.hist.trainA.at(-1), S.mlp.hist.testA.at(-1)] };
D.previews.get('optLoss').canvas.scrollIntoView({ block: 'center' });   // lazy：只有它真正在视口里才会训练
await sleep(800);
const f0 = S.frame, tA = performance.now();
while (S.optRuns && S.optRuns[0].step < S.optRuns[0].cap && performance.now() - tA < 60000) await sleep(500);
out.opt = { frames: S.frame - f0, secs: (performance.now() - tA) / 1000, runs: S.optRuns.map((R) => ({ step: R.step, ema: R.lossEMA, test: R.hist.testA.at(-1), test300: R.hist.testA[R.hist.step.findIndex((s) => s >= 300)], test600: R.hist.testA[R.hist.step.findIndex((s) => s >= 600)] })) };
for (const id of ['optLoss', 'optAcc', 'traj2d', 'f2curve']) out.imgs[id] = grab(id);
D.previews.get('regAcc').canvas.scrollIntoView({ block: 'center' });
await sleep(800);
const tB = performance.now();
while (S.regRuns && S.regRuns[0].step < S.regRuns[0].cap && performance.now() - tB < 80000) await sleep(500);
out.reg = S.regRuns.map((R) => { const H = R.hist; let b = 0; for (let i = 1; i < H.valL.length; i++) if (H.valL[i] < H.valL[b]) b = i; return { step: R.step, trainA: H.trainA.at(-1), valA: H.valA.at(-1), valL: H.valL.at(-1), best: { step: H.step[b], valL: H.valL[b], valA: H.valA[b], testA: H.testA[b] } }; });
for (const id of ['regAcc', 'regLoss', 'regTable']) out.imgs[id] = grab(id);
// 每帧耗时：在全新的 run 上各跑 60 帧（不影响页面状态）
const optFresh = P02.OPTS.map((o) => P02.makeRun(P02.mlpCfg(S, { opt: P02.optHp(o.kind, S), cap: 1e9, seed: 1, evalSets: [{ name: 'test', X: S.data.te.X, y: S.data.te.y, idx: S.data.allTe }] })));
optFresh.forEach((R) => P02.evalNow(R));
let tC = performance.now(); for (let k = 0; k < 60; k++) optFresh.forEach((R) => P02.tick(R, P02.SPF2, P02.EVAL_BUDGET)); out.optTickMs = (performance.now() - tC) / 60;
const regFresh = [{}, { wd: S.wd }, { dropout: S.dropout }].map((ex) => P02.makeRun(P02.regCfg(S, Object.assign({ cap: 1e9 }, ex))));
regFresh.forEach((R) => P02.evalNow(R));
tC = performance.now(); for (let k = 0; k < 60; k++) regFresh.forEach((R) => P02.tick(R, P02.SPF2, 60)); out.regTickMs = (performance.now() - tC) / 60;
const mainFresh = [P02.makeRun(P02.softCfg(S)), P02.makeRun(P02.mlpCfg(S))];
mainFresh.forEach((R) => { R.cap = 1e9; P02.evalNow(R); });
tC = performance.now(); for (let k = 0; k < 60; k++) mainFresh.forEach((R) => P02.tick(R, P02.SPF, P02.EVAL_BUDGET)); out.mainTickMs = (performance.now() - tC) / 60;
tC = performance.now(); for (let k = 0; k < 60; k++) P02.trace(S); out.traceMs = (performance.now() - tC) / 60;
return out;
