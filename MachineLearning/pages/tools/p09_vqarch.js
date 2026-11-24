// p09_vqarch.js — VQ-VAE 结构 / 码本大小 / 死码重启的对比：每步耗时、测试集 MSE、已用码字数、困惑度、重启次数
const { P09, LM } = require('./p09_proto.js');
const data = P09.makeData(LM.rng(20251123)); P09.initEvalEps(20251123);
for (const a of process.argv.slice(2)) {
    const c = JSON.parse(a);
    const R = P09.makeRun('vq', { data, seed: 20251156, K: c.K || 32, restart: !!c.restart, cap: c.steps || 3000, B: c.B || 32, lr: c.lr || 2e-3, vqArch: { C: c.C || 32, enc3: c.enc3 !== false } });
    let ts = 0; const marks = [];
    while (R.step < R.cap) { const t = process.hrtime.bigint(); P09.trainStep(R); ts += Number(process.hrtime.bigint() - t) / 1e6;
        if ([250, 500, 1000, 2000, 3000, 4000, 6000].includes(R.step)) { P09.evalNow(R); marks.push(`${R.step}: mse ${R.res.mse.toFixed(4)} used ${R.res.used} ppl ${R.res.perp.toFixed(1)} dead~${R.deadNow}`); } }
    console.log(a.padEnd(48), (ts / R.cap).toFixed(2), 'ms/step | restarts', R.restarts, '|', marks.join(' · '));
}
