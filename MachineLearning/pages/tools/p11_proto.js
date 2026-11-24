// p11_proto.js — 第 11 篇原型：数据统计、指标基线、解析 score 采样器（隔离离散化误差）
const { LM, P11 } = require('./p11_load.js');
const g = P11.GMM;
console.log('GMM components', g.M, 'per arm', Array.from(g.nPer));
const rand = LM.rng(5);
const N = 400, REF = 1500;
const ref = P11.sampleGMM(REF, LM.rng(101)).X;
const t0 = performance.now(); const base = P11.mmd(ref.subarray(0, 2), 1, ref, REF); const syy = base.syy; console.log('ref self kernel', syy.toFixed(4), (performance.now() - t0).toFixed(1), 'ms');
// data stats
let mx = 0, my = 0, vx = 0, vy = 0; for (let i = 0; i < REF; i++) { mx += ref[2 * i] / REF; my += ref[2 * i + 1] / REF; }
for (let i = 0; i < REF; i++) { vx += (ref[2 * i] - mx) ** 2 / REF; vy += (ref[2 * i + 1] - my) ** 2 / REF; }
console.log('mean', mx.toFixed(3), my.toFixed(3), 'var', vx.toFixed(3), vy.toFixed(3));
const lp = P11.logp(ref, REF); const srt = Array.from(lp).sort((a, b) => a - b); const thr = srt[Math.floor(0.01 * REF)];
console.log('log p quantiles: 1%', thr.toFixed(2), '50%', srt[REF >> 1].toFixed(2), 'max', srt[REF - 1].toFixed(2));
const off = (X, n) => { const l = P11.logp(X, n); let c = 0; for (let i = 0; i < n; i++) if (l[i] < thr) c++; return c / n; };
const rep = (name, X, n) => { const t1 = performance.now(); const r = P11.mmd(X, n, ref, REF, syy); console.log(name.padEnd(34), 'MMD²×1e3', r.mmd.toFixed(3), 'off', (100 * off(X, n)).toFixed(1) + '%', (performance.now() - t1).toFixed(1) + 'ms'); };
for (let s = 0; s < 3; s++) rep('true samples (seed ' + s + ')', P11.sampleGMM(N, LM.rng(200 + s)).X, N);
const G = new Float32Array(2 * N); for (let i = 0; i < N; i++) { G[2 * i] = mx + Math.sqrt(vx) * LM.randn(rand); G[2 * i + 1] = my + Math.sqrt(vy) * LM.randn(rand); }
rep('single Gaussian fit', G, N);
const Z = new Float32Array(2 * N).map(() => LM.randn(rand)); rep('N(0, I)', Z, N);
// blurred data (σ larger)
const Bl = P11.sampleGMM(N, LM.rng(300)).X.map((v) => v + 0.05 * LM.randn(rand)); rep('data + N(0, 0.05²) blur', Bl, N);
const Bl2 = P11.sampleGMM(N, LM.rng(301)).X.map((v) => v + 0.1 * LM.randn(rand)); rep('data + N(0, 0.1²) blur', Bl2, N);
// exact-score samplers
for (const kind of ['cosine', 'linear']) {
    const sch = P11.SCHED[kind];
    console.log(kind, 'abar_T', sch.ab[1000].toExponential(3), 'a_T', sch.a[1000].toExponential(3), 'abar(500)', sch.ab[500].toFixed(3));
    for (const eta of [0, 1]) for (const K of [2, 5, 10, 20, 50, 100, 250]) {
        const x = new Float32Array(2 * N); const r2 = LM.rng(77); for (let i = 0; i < 2 * N; i++) x[i] = LM.randn(r2);
        const t1 = performance.now();
        P11.sampleVP(P11.epsExact(sch), sch, x, N, { K, eta, rand: LM.rng(9) });
        const ms = performance.now() - t1;
        const r = P11.mmd(x, N, ref, REF, syy);
        console.log(`  exact ${kind} eta=${eta} K=${String(K).padStart(4)}: MMD ${r.mmd.toFixed(3)} off ${(100 * off(x, N)).toFixed(1)}%  (${ms.toFixed(0)} ms)`);
    }
}
