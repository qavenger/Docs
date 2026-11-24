// p14_beta.js — VAE 的 β（KL 权重）对重建 / KL / 位置误差的影响（页面 Step 03 的参数）。用法：node tools/p14_beta.js 0.1,1,10 3000
const { LM, P14 } = require('./p14_load.js');
const betas = (process.argv[2] || '0.1,1,10').split(',').map(Number), steps = +(process.argv[3] || 3000);
const ds = P14.makeDataset(256, 32, LM.rng(20251123)), val = P14.makeDataset(64, 32, LM.rng(20251124)), nFv = val.nEp * 33;
for (const beta of betas) {
    const R = P14.vaeRun(ds, { cfg: { beta } });
    for (let it = 0; it < steps; it++) P14.vaeRunStep(R);
    const tmp = {}, mu = new Float32Array(nFv * 4), lv = new Float32Array(nFv * 4), xh = new Float32Array(nFv * 256);
    P14.encodeFast(R.m, val.X, nFv, mu, tmp, lv); P14.decodeFast(R.m, mu, nFv, xh, tmp);
    let pe = 0, se = 0, nok = 0; const kl = [0, 0, 0, 0];
    for (let i = 0; i < nFv; i++) {
        for (let j = 0; j < 256; j++) se += (xh[i * 256 + j] - val.X[i * 256 + j]) ** 2;
        const p = P14.ballPos(xh.subarray(i * 256, (i + 1) * 256)); if (isFinite(p[0])) { pe += Math.hypot(p[0] - val.S[i * 4], p[1] - val.S[i * 4 + 1]); nok++; }
        for (let j = 0; j < 4; j++) kl[j] += 0.5 * (mu[i * 4 + j] ** 2 + Math.exp(lv[i * 4 + j]) - 1 - lv[i * 4 + j]) / nFv;
    }
    console.log(`beta ${beta}: steps ${steps} val SSE ${(se / nFv).toFixed(3)} posErr ${(pe / Math.max(1, nok)).toFixed(3)} px (decodable ${(100 * nok / nFv).toFixed(0)}%) KL/dim ${kl.map((v) => v.toFixed(2)).join(' ')}`);
}
