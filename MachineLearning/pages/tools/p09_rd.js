// p09_rd.js — β 扫描（率-失真曲线）：用页面同一份 P09 代码、与页面默认完全相同的数据 / 种子 / batch / 学习率 / 步数（3000），
// 对每个 β 训练一个 2 维 VAE，记录测试集的 rate = E[KL]、distortion = E[−log p(x|z)]、每一维的 KL、逐像素 MSE。
// 结果打印成一行 JS，贴进 09_latent_vae.lab.js 的 P09.RD 常量。用法：node pages/tools/p09_rd.js
const { P09, LM } = require('./p09_proto.js');
const seed = 20251123, data = P09.makeData(LM.rng(seed)); P09.initEvalEps(seed);
const betas = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16], rows = [];
for (const beta of betas) {
    const R = P09.makeRun('vae', { data, seed: seed + 22, beta, cap: 3000, B: 32, lr: 5e-3 });
    const t = Date.now(); while (R.step < R.cap) P09.trainStep(R); P09.evalNow(R);
    const r = R.res;
    rows.push({ beta, rate: +r.kl.toFixed(3), dist: +r.rec.toFixed(2), mse: +r.mse.toFixed(5), klDim: r.klDim.map((v) => +v.toFixed(3)) });
    console.error(`β ${beta}: rate ${r.kl.toFixed(3)} dist ${r.rec.toFixed(2)} mse ${r.mse.toFixed(4)} klDim ${r.klDim.map((v) => v.toFixed(3))} (${((Date.now() - t) / 1000).toFixed(1)} s)`);
}
// AE（无 KL 项，β → 0 的极限）作为参照：同样的结构、数据、步数
{
    const R = P09.makeRun('ae', { data, seed: seed + 11, cap: 3000, B: 32, lr: 5e-3 });
    while (R.step < R.cap) P09.trainStep(R); P09.evalNow(R);
    rows.push({ beta: 0, rate: null, dist: +R.res.rec.toFixed(2), mse: +R.res.mse.toFixed(5), klDim: null });
    console.error(`AE: dist ${R.res.rec.toFixed(2)} mse ${R.res.mse.toFixed(4)}`);
}
console.log('    const RD = ' + JSON.stringify(rows) + ';   // tools/p09_rd.js 的输出（默认种子，每个 β 训练 3000 步）');
