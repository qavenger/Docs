// 评估 checkpoint：VP 采样器扫描、FM 步数扫描、CFG 引导扫描（真实指标）
const fs = require('fs'), path = require('path');
const { LM, P11 } = require('./p11_load.js');
const load = (n, w = 'ema') => { const j = JSON.parse(fs.readFileSync(path.join(__dirname, 'p11_ckpt', n + '.json'), 'utf8')); return P11.loadNet(j.cfg, j[w]); };
const what = process.argv[2] || 'vp';
const N = 600, REF = 1500, ref = P11.sampleGMM(REF, LM.rng(101)).X, syy = P11.mmd(ref.subarray(0, 2), 1, ref, REF).syy;
const lpRef = Array.from(P11.logp(ref, REF)).sort((a, b) => a - b), thr = lpRef[Math.floor(0.01 * REF)];
const stats = (X, n, R = ref, S = syy, cls) => { const l = P11.logp(X, n, null, cls); let off = 0, outer = 0; for (let i = 0; i < n; i++) { if (l[i] < thr) off++; if (Math.hypot(X[2 * i], X[2 * i + 1]) > 0.9) outer++; } return `${P11.mmd(X, n, R, R.length / 2, S).mmd.toFixed(2).padStart(7)} ${(100 * off / n).toFixed(1).padStart(5)}% r>.9 ${(outer / n).toFixed(2)}`; };
const noise = (seed) => { const x = new Float32Array(2 * N), r = LM.rng(seed); for (let i = 0; i < 2 * N; i++) x[i] = LM.randn(r); return x; };
console.log('data:', stats(P11.sampleGMM(N, LM.rng(5)).X, N));
if (what === 'vp') {
    for (const [name, sk] of [['vp_cos', 'cosine'], ['vp_lin', 'linear']]) {
        const net = load(name), sch = P11.SCHED[sk];
        for (const eta of [0, 1]) for (const K of [2, 5, 10, 20, 50, 100, 250]) {
            const a = noise(77); P11.sampleVP(P11.epsNet(net), sch, a, N, { K, eta, rand: LM.rng(9), clip: 1.25 });
            const b = noise(77); P11.sampleVP(P11.epsExact(sch), sch, b, N, { K, eta, rand: LM.rng(9) });
            console.log(`${name} η${eta} K${String(K).padStart(3)}  net ${stats(a, N)}  | exact ${stats(b, N)}`);
        }
    }
}
if (what === 'fm') {
    const net = load('fm');
    for (const K of [1, 2, 3, 4, 6, 8, 16, 32, 64, 128]) { const a = noise(77); P11.sampleFM(net, a, N, { K }); console.log(`fm K${String(K).padStart(3)} ${stats(a, N)}`); }
}
if (what === 'cfg') {
    const net = load('cfg'), sch = P11.SCHED.cosine, n = 400;
    for (let c = 0; c < 3; c++) {
        const refc = P11.sampleGMM(1200, LM.rng(400 + c), c).X, syyc = P11.mmd(refc.subarray(0, 2), 1, refc, 1200).syy;
        for (const s of [0, 1, 1.5, 2, 3, 5, 8]) {
            const x = new Float32Array(2 * n), r = LM.rng(77); for (let i = 0; i < 2 * n; i++) x[i] = LM.randn(r);
            P11.sampleVP(P11.epsNet(net, { cls: s === 0 ? -1 : c, guide: s }), sch, x, n, { K: 50, eta: 0, rand: LM.rng(9), clip: 1.25 });
            const pc = new Float32Array(3 * n); P11.exact(1, 0, x, n, { pc }); let acc = 0, conf = 0, rs = 0;
            for (let i = 0; i < n; i++) { let b = 0; for (let k = 1; k < 3; k++) if (pc[3 * i + k] > pc[3 * i + b]) b = k; if (b === c) acc++; conf += pc[3 * i + c]; rs += Math.hypot(x[2 * i], x[2 * i + 1]); }
            // within-class spread: std of radius
            let mr = rs / n, vr = 0; for (let i = 0; i < n; i++) vr += (Math.hypot(x[2 * i], x[2 * i + 1]) - mr) ** 2 / n;
            console.log(`class ${c} s=${String(s).padEnd(3)} acc ${(100 * acc / n).toFixed(1).padStart(5)}%  MMD(class) ${P11.mmd(x, n, refc, 1200, syyc).mmd.toFixed(2).padStart(7)}  r̄ ${mr.toFixed(3)} σr ${Math.sqrt(vr).toFixed(3)}`);
        }
        let mr = 0, vr = 0; for (let i = 0; i < 1200; i++) mr += Math.hypot(refc[2 * i], refc[2 * i + 1]) / 1200; for (let i = 0; i < 1200; i++) vr += (Math.hypot(refc[2 * i], refc[2 * i + 1]) - mr) ** 2 / 1200;
        console.log(`class ${c} data: r̄ ${mr.toFixed(3)} σr ${Math.sqrt(vr).toFixed(3)}`);
    }
}
