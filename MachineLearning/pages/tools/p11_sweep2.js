// 第一步 x̂₀ 截断、日程、arm 平衡与半径分布
const args = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
global.P11_DATA_OPT = { sig: args.sig, wind: args.wind };
const { LM, P11 } = require('./p11_load.js');
const data = (() => { const r = P11.sampleGMM(4096, LM.rng(1234)); return { X: r.X, lab: r.lab, n: 4096 }; })();
const sch = P11.SCHED[args.sched || 'cosine'], STEPS = args.steps || 4000;
const R = P11.trainer({ kind: 'vp', sched: sch, data, width: args.W || 64, depth: 3, B: args.B || 128, lr: args.lr || 2e-3, steps: STEPS, seed: 1, warmup: 100, ff: args.ff ?? 0 });
for (let s = 1; s <= STEPS; s++) R.step();
const N = 600, REF = 1500, ref = P11.sampleGMM(REF, LM.rng(101)).X, syy = P11.mmd(ref.subarray(0, 2), 1, ref, REF).syy;
const lpRef = Array.from(P11.logp(ref, REF)).sort((a, b) => a - b), thr = lpRef[Math.floor(0.01 * REF)];
const stats = (X, n) => {
    const l = P11.logp(X, n), pc = new Float32Array(3 * n); P11.exact(1, 0, X, n, { pc }); let off = 0; const arm = [0, 0, 0], rad = [0, 0, 0, 0];
    for (let i = 0; i < n; i++) { if (l[i] < thr) off++; let b = 0; for (let c = 1; c < 3; c++) if (pc[3 * i + c] > pc[3 * i + b]) b = c; arm[b]++; const r = Math.hypot(X[2 * i], X[2 * i + 1]); rad[Math.min(3, Math.floor(r / 0.3))]++; }
    return `MMD ${P11.mmd(X, n, ref, REF, syy).mmd.toFixed(2)} off ${(100 * off / n).toFixed(1)}% arms ${arm.map((v) => (v / n).toFixed(2)).join('/')} r-bins ${rad.map((v) => (v / n).toFixed(2)).join('/')}`;
};
console.log(`sig=${P11.SIG} sched=${args.sched || 'cosine'} W=${args.W || 64} steps=${STEPS}`);
console.log('  data      ', stats(P11.sampleGMM(N, LM.rng(5)).X, N));
for (const clip of [1.25, 2.5, 1e9]) for (const [K, eta] of [[20, 0], [50, 0], [100, 1]]) {
    const x = new Float32Array(2 * N); const r2 = LM.rng(77); for (let i = 0; i < 2 * N; i++) x[i] = LM.randn(r2);
    P11.sampleVP(P11.epsNet(R.net), sch, x, N, { K, eta, rand: LM.rng(9), clip });
    console.log(`  clip ${String(clip).padEnd(5)} K${K} η${eta}`, stats(x, N));
}
