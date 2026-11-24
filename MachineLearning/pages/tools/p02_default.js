// p02_default.js — 用与页面完全相同的配置 / 种子跑两个主训练，挑默认样本（Softmax 判错、MLP 判对），并测每帧耗时
const { P02, LM } = require('./p02_proto.js');
const S = { seed: 20251123, lr: 0.1, hidden: 64, act: 'relu', init: 'he', nsub: 500, wd: 1e-2, dropout: 0.5 };
S.data = P02.makeData(LM.rng(S.seed));
const soft = P02.makeRun(P02.softCfg(S)), mlp = P02.makeRun(P02.mlpCfg(S));
P02.evalNow(soft); P02.evalNow(mlp);
let frames = 0, tS = 0, tM = 0;
while (soft.step < soft.cap || mlp.step < mlp.cap) {
    let t0 = process.hrtime.bigint(); P02.tick(soft, P02.SPF, P02.EVAL_BUDGET); tS += Number(process.hrtime.bigint() - t0) / 1e6;
    t0 = process.hrtime.bigint(); P02.tick(mlp, P02.SPF, P02.EVAL_BUDGET); tM += Number(process.hrtime.bigint() - t0) / 1e6;
    frames++;
}
console.log(`frames ${frames}, softmax ${(tS / frames).toFixed(2)} ms/frame, mlp ${(tM / frames).toFixed(2)} ms/frame`);
console.log('softmax final train/test', P02.last(soft.hist.trainA), P02.last(soft.hist.testA), 'mlp', P02.last(mlp.hist.trainA), P02.last(mlp.hist.testA));
const te = S.data.te, ps = soft.pred[1], pm = mlp.pred[1], cand = [];
for (let i = 0; i < te.n; i++) if (ps[i] !== te.y[i] && pm[i] === te.y[i]) cand.push(`${i}:${te.y[i]}→${ps[i]}`);
console.log('softmax wrong & mlp right:', cand.join(' '));
// also check at intermediate steps for stability: how many of these are also wrong for softmax early on
