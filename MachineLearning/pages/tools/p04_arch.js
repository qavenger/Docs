// p04_arch.js — 2 维 AE 的结构 / 学习率 / 激活对比（同样的步数下测试集 MSE、BCE、每步耗时）
const { P09, LM } = require('./p04_proto.js');
const { AG } = LM;
const data = P09.makeData(LM.rng(20251123)); P09.initEvalEps(20251123);
const cfgs = process.argv.slice(2).map((a) => JSON.parse(a));
for (const c of cfgs) {
    const reluOrig = AG.relu; if (c.act) AG.relu = AG[c.act];
    const R = P09.makeRun(c.kind || 'ae', { data, seed: 20251134, beta: c.beta ?? 1, K: 32, cap: c.steps || 3000, B: c.B || 32, lr: c.lr || 2e-3, arch: c.arch });
    let ts = 0;
    const marks = [];
    while (R.step < R.cap) { const a = process.hrtime.bigint(); P09.trainStep(R); ts += Number(process.hrtime.bigint() - a) / 1e6; if (R.step % 1000 === 0 || R.step === R.cap) { P09.evalNow(R); marks.push(`${R.step}: mse ${R.res.mse.toFixed(4)} rec ${R.res.rec.toFixed(1)}` + (R.kind === 'vae' ? ` kl ${R.res.kl.toFixed(2)}` : '')); } }
    AG.relu = reluOrig;
    console.log(JSON.stringify(c).padEnd(60), (ts / R.cap).toFixed(2), 'ms/step |', marks.join(' · '));
}
