// p02_poly.js — 检查：先用 Float64Array 跑过梯度校验后，同一套 forward/backward 的 float32 训练是否变慢（V8 多态）
const { P02, LM } = require('./p02_proto.js');
const S = { seed: 20251123, lr: 0.1, hidden: 64, act: 'relu', init: 'he' };
S.data = P02.makeData(LM.rng(S.seed));
function timeMain(label) {
    const runs = [P02.makeRun(P02.softCfg(S)), P02.makeRun(P02.mlpCfg(S))];
    runs.forEach((R) => { R.cap = 1e9; P02.evalNow(R); });
    for (let k = 0; k < 30; k++) runs.forEach((R) => P02.tick(R, P02.SPF, P02.EVAL_BUDGET));   // warm-up
    const t0 = process.hrtime.bigint();
    for (let k = 0; k < 200; k++) runs.forEach((R) => P02.tick(R, P02.SPF, P02.EVAL_BUDGET));
    console.log(label.padEnd(28), (Number(process.hrtime.bigint() - t0) / 1e6 / 200).toFixed(2), 'ms/frame');
}
const mode = process.argv[2];
if (mode === 'gc-first') { const net = P02.makeNet([256, 64, 10], 'relu', 'he', LM.rng(1)); P02.gradCheck(net, S.data.tr.X, S.data.tr.y, 16); }
timeMain(mode || 'plain');
