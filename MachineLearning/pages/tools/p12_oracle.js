// p12_oracle.js — 用「精确的最优去噪器」检验采样器，并逐噪声水平比较网络的 x̂₀ 与最优 x̂₀。
// 最优去噪器：E[x₀ | x_t] = Σ_k w_k·x₀ᵏ，w_k ∝ exp(−‖x_t − (1−t)·x₀ᵏ‖² / 2t²)，x₀ᵏ 跑遍 (cx, cy, r) 的细网格（先验 ≈ 数据分布）。
// 用法：node pages/tools/p12_oracle.js [weights.b64 | ckpt.json]
const fs = require('fs');
const { LM, P12 } = require('./p12_load.js');
// 候选图
const cands = [];
for (const r of [1.6, 1.85, 2.1, 2.35, 2.6]) for (let i = 0; i < 28; i++) for (let j = 0; j < 28; j++) {
    const cx = r + (16 - 2 * r) * (i + 0.5) / 28, cy = r + (16 - 2 * r) * (j + 0.5) / 28;
    cands.push(P12.renderFrame(cx, cy, r, new Float32Array(256), 0));
}
const NC = cands.length;
console.log(`${NC} candidate frames`);
function oracleX0(xt, off, t, out) {
    const lw = new Float64Array(NC); let mx = -Infinity;
    const a = 1 - t, inv = 1 / (2 * t * t);
    for (let k = 0; k < NC; k++) { const c = cands[k]; let e = 0; for (let i = 0; i < 256; i++) { const d = xt[off + i] - a * c[i]; e += d * d; } lw[k] = -e * inv; if (lw[k] > mx) mx = lw[k]; }
    let z = 0; for (let k = 0; k < NC; k++) { lw[k] = Math.exp(lw[k] - mx); z += lw[k]; }
    out.fill(0); for (let k = 0; k < NC; k++) { const w = lw[k] / z; if (w < 1e-9) continue; const c = cands[k]; for (let i = 0; i < 256; i++) out[i] += w * c[i]; }
    return out;
}
// 1) 用最优去噪器跑同一个 Euler 采样器（x̂₀ 参数化：v = (x − x̂₀)/t）
const K = 16, n = 24, rand = LM.rng(3);
{
    const x = new Float32Array(n * 256); for (let i = 0; i < x.length; i++) x[i] = LM.randn(rand);
    const x0 = new Float32Array(256);
    for (let k = 0; k < K; k++) {
        const t = 1 - k / K, dt = 1 / K;
        for (let b = 0; b < n; b++) { oracleX0(x, b * 256, t, x0); for (let i = 0; i < 256; i++) { const v = (x[b * 256 + i] - x0[i]) / t; x[b * 256 + i] -= dt * v; } }
    }
    let one = 0, mass = 0; for (let b = 0; b < n; b++) { const s = P12.frameStats(x, b * 256); if (s.comps === 1) one++; mass += s.m; }
    console.log(`oracle denoiser + Euler K=${K}: exactly-one-ball ${(one / n * 100).toFixed(0)}%, mean mass ${(mass / n).toFixed(1)}`);
}
// 2) 网络 vs 最优：在真实帧加噪后的 x_t 上比较 x̂₀ 的误差
const arg = process.argv[2];
if (arg) {
    let cfg, b64;
    if (arg.endsWith('.json')) { const ck = JSON.parse(fs.readFileSync(arg, 'utf8')); cfg = ck.cfg; b64 = ck.b64; } else { cfg = { d: 32, L: 2, H: 4, mlp: 4, video: false, pred: process.argv[3] || 'v' }; b64 = fs.readFileSync(arg, 'utf8'); }
    const m = P12.createModel(cfg, 1); P12.loadParams(m, b64);
    const N = 32, X0 = new Float32Array(N * 256), R = LM.rng(8);
    for (let i = 0; i < N; i++) P12.renderVideo(P12.sampleBall(R), X0, i * 256, 1, 0);
    for (const t of [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
        const xt = new Float32Array(N * 256); for (let i = 0; i < xt.length; i++) xt[i] = (1 - t) * X0[i] + t * LM.randn(R);
        const s = P12.newSampler(m, { B: N, K: 1000, rand: R }); s.x.set(xt); s.k = Math.round((1 - t) * 1000);
        const v = P12.evalV(s, null);
        let eNet = 0, eOr = 0, eGap = 0; const o = new Float32Array(256);
        for (let b = 0; b < N; b++) {
            oracleX0(xt, b * 256, t, o);
            for (let i = 0; i < 256; i++) { const net = xt[b * 256 + i] - t * v[b * 256 + i], tr = X0[b * 256 + i]; eNet += (net - tr) ** 2; eOr += (o[i] - tr) ** 2; eGap += (net - o[i]) ** 2; }
        }
        const nn = N * 256;
        console.log(`t = ${t.toFixed(1)}: x̂₀ MSE  network ${(eNet / nn).toFixed(4)}   optimal ${(eOr / nn).toFixed(4)}   ‖net − optimal‖² ${(eGap / nn).toFixed(4)}`);
    }
}
