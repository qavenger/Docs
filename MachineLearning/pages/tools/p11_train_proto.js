// p11_train_proto.js — 原型：小 MLP 去噪器在三臂螺旋上训练多少步、多宽才够；采样质量（MMD / 偏离率）
const { LM, P11 } = require('./p11_load.js');
const args = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const W = args.W || 64, D = args.D || 3, B = args.B || 128, LR = args.lr || 2e-3, STEPS = args.steps || 4000, KIND = args.kind || 'vp', SCH = args.sched || 'cosine';
const data = (() => { const r = P11.sampleGMM(4096, LM.rng(1234)); return { X: r.X, lab: r.lab, n: 4096 }; })();
const N = 400, REF = 1500, ref = P11.sampleGMM(REF, LM.rng(101)).X, syy = P11.mmd(ref.subarray(0, 2), 1, ref, REF).syy;
const lpRef = Array.from(P11.logp(ref, REF)).sort((a, b) => a - b), thr = lpRef[Math.floor(0.01 * REF)];
const off = (X, n) => { const l = P11.logp(X, n); let c = 0; for (let i = 0; i < n; i++) if (l[i] < thr) c++; return 100 * c / n; };
const R = P11.trainer({ kind: KIND, sched: P11.SCHED[SCH], data, width: W, depth: D, B, lr: LR, steps: STEPS, seed: args.seed || 1, warmup: 100, ff: args.ff || 0 });
console.log(`ff=${args.ff || 0} kind=${KIND} sched=${SCH} W=${W} D=${D} B=${B} lr=${LR} steps=${STEPS} params=${P11.numParams(R.net)}`);
const evalNow = () => {
    const out = [];
    for (const [K, eta] of (KIND === 'vp' ? [[20, 0], [50, 0], [100, 1]] : [[4, 0], [16, 0], [50, 0]])) {
        const x = new Float32Array(2 * N); const r2 = LM.rng(77); for (let i = 0; i < 2 * N; i++) x[i] = LM.randn(r2);
        if (KIND === 'vp') P11.sampleVP(P11.epsNet(R.net), P11.SCHED[SCH], x, N, { K, eta, rand: LM.rng(9) }); else P11.sampleFM(R.net, x, N, { K });
        out.push(`K${K}${KIND === 'vp' ? 'η' + eta : ''}: ${P11.mmd(x, N, ref, REF, syy).mmd.toFixed(2)}/${off(x, N).toFixed(1)}%`);
    }
    return out.join('  ');
};
let acc = 0, na = 0; const t0 = performance.now();
for (let s = 1; s <= STEPS; s++) {
    const r = R.step(); acc += r.loss; na++;
    if (s % (args.every || 1000) === 0 || s === STEPS) { const ms = (performance.now() - t0) / s; console.log(`step ${s} loss ${(acc / na).toFixed(4)} (${ms.toFixed(2)} ms/step)  ${evalNow()}`); acc = 0; na = 0; }
}
