// p12_diag.js — 从「部分加噪的真实帧」开始采样：从哪个噪声水平 t₀ 出发还能得到干净的单个球？
// 用法：node pages/tools/p12_diag.js <weights.b64 | ckpt.json> [cfgJSON]
const fs = require('fs');
const { LM, P12 } = require('./p12_load.js');
const arg = process.argv[2];
let cfg = Object.assign({ d: 32, L: 2, H: 4, mlp: 4, video: false }, JSON.parse(process.argv[3] || '{}')), b64;
if (arg.endsWith('.json')) { const ck = JSON.parse(fs.readFileSync(arg, 'utf8')); cfg = ck.cfg; b64 = ck.b64; } else b64 = fs.readFileSync(arg, 'utf8');
const m = P12.createModel(cfg, 1); P12.loadParams(m, b64);
const rand = LM.rng(5), n = 64, K = 16;
const X0 = new Float32Array(n * 256); for (let i = 0; i < n; i++) P12.renderVideo(P12.sampleBall(rand), X0, i * 256, 1, 0);
for (const t0 of [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]) {
    // x_{t0} = (1 − t0)·x0 + t0·ε，然后从 t0 积分到 0（步长同 K = 16 的网格）
    const s = P12.newSampler(m, { B: n, K, rand });
    const k0 = Math.round((1 - t0) * K); s.k = k0;
    const tt = 1 - k0 / K;
    for (let i = 0; i < s.x.length; i++) s.x[i] = (1 - tt) * X0[i] + tt * LM.randn(rand);
    while (!s.done) P12.samplerWork(s);
    let one = 0, dist = 0, nd = 0;
    for (let i = 0; i < n; i++) {
        const a = P12.frameStats(s.x, i * 256), b = P12.frameStats(X0, i * 256);
        if (a.comps === 1) one++;
        if (isFinite(a.cx)) { dist += Math.hypot(a.cx - b.cx, a.cy - b.cy); nd++; }
    }
    console.log(`start t0 = ${tt.toFixed(3)}: exactly-one-ball ${(one / n * 100).toFixed(0)}%   centroid shift from the original ${(dist / nd).toFixed(2)} px`);
}
// 网络在 t = 1 时的 x̂₀ = x₁ − v̂：最优解是「所有位置的平均图」
{
    const s = P12.newSampler(m, { B: 16, K, rand });
    const v = P12.evalV(s, null);
    let mean = 0, sd = 0; const x0h = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) { x0h[i] = s.x[i] - v[i]; mean += x0h[i]; }
    mean /= v.length; for (let i = 0; i < v.length; i++) sd += (x0h[i] - mean) ** 2; sd = Math.sqrt(sd / v.length);
    const real = new Float32Array(256); let rm = 0; const R = LM.rng(9);
    for (let k = 0; k < 4000; k++) { P12.renderVideo(P12.sampleBall(R), real, 0, 1, 0); for (let i = 0; i < 256; i++) rm += real[i] / 4000 / 256; }
    console.log(`x̂₀ at t = 1: mean ${mean.toFixed(3)} (data mean ${rm.toFixed(3)}), spatial sd ${sd.toFixed(3)}`);
}
