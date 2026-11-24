// 诊断：训练后各 t 的超额损失（模型 − 解析下限）、x̂₀ 误差，以及样本的 ASCII 密度图
const { LM, P11 } = require('./p11_load.js');
const args = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const data = (() => { const r = P11.sampleGMM(4096, LM.rng(1234)); return { X: r.X, lab: r.lab, n: 4096 }; })();
const sch = P11.SCHED.cosine;
const R = P11.trainer({ kind: 'vp', sched: sch, data, width: args.W || 64, depth: 3, B: 128, lr: args.lr || 2e-3, steps: args.steps || 4000, seed: 1, warmup: 100, ff: args.ff ?? 4 });
for (let s = 0; s < (args.steps || 4000); s++) R.step();
const rand = LM.rng(3), n = 2000, X0 = P11.sampleGMM(n, rand).X, xt = new Float32Array(2 * n), eps = new Float32Array(2 * n), x0h = new Float32Array(2 * n), ep = new Float32Array(2 * n);
const rows = [];
for (const i of [1, 10, 30, 60, 100, 150, 200, 300, 400, 500, 700, 900]) {
    const a = sch.a[i], s = sch.s[i];
    for (let q = 0; q < 2 * n; q++) { eps[q] = LM.randn(rand); xt[q] = a * X0[q] + s * eps[q]; }
    P11.exact(a, s * s, xt, n, { x0: x0h });
    P11.epsNet(R.net)(xt, n, i, ep);
    let fl = 0, md = 0, gap = 0; for (let q = 0; q < 2 * n; q++) { const e0 = (xt[q] - a * x0h[q]) / s; fl += (eps[q] - e0) ** 2; md += (eps[q] - ep[q]) ** 2; gap += (ep[q] - e0) ** 2; }
    rows.push(`t=${i}: floor ${(fl / 2 / n).toFixed(3)} model ${(md / 2 / n).toFixed(3)} |ε̂−ε*|² ${(gap / 2 / n).toFixed(4)} x̂₀err ${(Math.sqrt(gap / 2 / n) * s / a).toFixed(3)}`);
}
console.log(rows.join('\n'));
function ascii(X, N) {
    const G = 48, H = 24, cnt = new Int32Array(G * H);
    for (let i = 0; i < N; i++) { const gx = Math.floor((X[2 * i] + 1.2) / 2.4 * G), gy = Math.floor((1.2 - X[2 * i + 1]) / 2.4 * H); if (gx >= 0 && gx < G && gy >= 0 && gy < H) cnt[gy * G + gx]++; }
    const ch = ' .:-=+*#%@'; let s = ''; for (let y = 0; y < H; y++) { for (let x = 0; x < G; x++) s += ch[Math.min(9, cnt[y * G + x])]; s += '\n'; } return s;
}
const N = 1000, x = new Float32Array(2 * N); const r2 = LM.rng(77); for (let i = 0; i < 2 * N; i++) x[i] = LM.randn(r2);
P11.sampleVP(P11.epsNet(R.net), sch, x, N, { K: 100, eta: 1, rand: LM.rng(9) });
const tr = P11.sampleGMM(N, LM.rng(5)).X;
const A = ascii(tr, N).split('\n'), Bm = ascii(x, N).split('\n'); console.log(A.map((l, k) => l + ' | ' + Bm[k]).join('\n'));
