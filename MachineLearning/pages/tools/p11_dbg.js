const { LM, P11 } = require('./p11_load.js');
const data = (() => { const r = P11.sampleGMM(4096, LM.rng(1234)); return { X: r.X, lab: r.lab, n: 4096 }; })();
const sch = P11.SCHED.cosine;
const R = P11.trainer({ kind: 'vp', sched: sch, data, width: 64, depth: 3, B: 128, lr: 2e-3, steps: 4000, seed: 1, warmup: 100, ff: 4 });
for (let s = 0; s < 4000; s++) R.step();
const N = 600, REF = 1500, ref = P11.sampleGMM(REF, LM.rng(101)).X, syy = P11.mmd(ref.subarray(0, 2), 1, ref, REF).syy;
const lpRef = Array.from(P11.logp(ref, REF)).sort((a, b) => a - b), thr = lpRef[Math.floor(0.01 * REF)];
const off = (X, n) => { const l = P11.logp(X, n); let c = 0; for (let i = 0; i < n; i++) if (l[i] < thr) c++; return 100 * c / n; };
// 自定义：从真实 x_t（t = t0）出发，K 步 DDPM 到 0
function fromT(t0, epsFn, K) {
    const rand = LM.rng(5), X0 = P11.sampleGMM(N, rand).X, x = new Float32Array(2 * N);
    for (let q = 0; q < 2 * N; q++) x[q] = sch.a[t0] * X0[q] + sch.s[t0] * LM.randn(rand);
    const eps = new Float32Array(2 * N), r2 = LM.rng(9);
    const tau = Array.from({ length: K + 1 }, (_, k) => Math.round(k * t0 / K));
    let errs = [];
    for (let k = K; k >= 1; k--) {
        const i = tau[k], j = tau[k - 1];
        epsFn(x, N, i, eps);
        // compare with exact at current x
        const ex = new Float32Array(2 * N); P11.epsExact(sch)(x, N, i, ex); let e = 0; for (let q = 0; q < 2 * N; q++) e += (eps[q] - ex[q]) ** 2; if (k % Math.max(1, K / 5) === 0) errs.push(`t${i}:${(e / 2 / N).toFixed(3)}`);
        const ai = sch.a[i], si = sch.s[i], abj = sch.ab[j], abi = sch.ab[i], sig = Math.sqrt((1 - abj) / (1 - abi) * (1 - abi / abj)), cd = Math.sqrt(Math.max(0, 1 - abj - sig * sig)), aj = Math.sqrt(abj);
        for (let q = 0; q < 2 * N; q++) { const x0 = (x[q] - si * eps[q]) / ai; x[q] = aj * x0 + cd * eps[q] + sig * LM.randn(r2); }
    }
    return { mmd: P11.mmd(x, N, ref, REF, syy).mmd, off: off(x, N), errs };
}
for (const t0 of [100, 300, 500, 700, 900, 1000]) {
    const a = fromT(t0, P11.epsNet(R.net), Math.max(10, t0 / 10)), b = fromT(t0, P11.epsExact(sch), Math.max(10, t0 / 10));
    console.log(`from t=${t0}: net MMD ${a.mmd.toFixed(2)} off ${a.off.toFixed(1)}% | exact MMD ${b.mmd.toFixed(2)} off ${b.off.toFixed(1)}%   net-err along traj: ${a.errs.join(' ')}`);
}
