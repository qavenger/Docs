// p11_numbers.js — 用页面各个 stage 的「同一套设置」重算正文里引用的数字（node tools/p11_numbers.js [part]）
const { LM, P11 } = require('./p11_load.js');
const part = process.argv[2] || 'all', COS = P11.SCHED.cosine, M = P11.loadModels(), noise = P11.noiseBank(300, 77);
const f = (q) => `MMD ${q.mmd.toFixed(2)} off ${(100 * q.off).toFixed(1)}% r ${q.r.toFixed(3)}`;
const run = (jobs) => { const R = P11.jobRunner(jobs); while (!R.done) R.step(1e9); return R.res; };
if (part === 'all' || part === 'base') {
    const n = 400, r = LM.rng(3), q = (X) => P11.quality(X, n), sg = Math.sqrt(0.263);
    console.log('true', f(q(P11.sampleGMM(n, LM.rng(200)).X)));
    for (const b of [0.1, 0.15, 0.2]) { const rb = LM.rng(7), X = P11.sampleGMM(n, LM.rng(201)).X.map((v) => v + b * LM.randn(rb)); console.log('blur', b, f(q(X))); }
    console.log('shrink', f(q(P11.sampleGMM(n, LM.rng(202)).X.map((v) => 0.9 * v))));
    console.log('gauss', f(q(new Float32Array(2 * n).map(() => sg * LM.randn(r)))), '| N(0,I)', f(q(P11.noiseBank(n, 204))));
    let rr = 0; const D = P11.sampleGMM(3000, LM.rng(5)).X; for (let i = 0; i < 3000; i++) rr += Math.hypot(D[2 * i], D[2 * i + 1]) / 3000; console.log('data mean radius', rr.toFixed(3));
    for (const N of [256, 300]) { const mm = []; for (let s = 0; s < 12; s++) mm.push(P11.mmdTrue(P11.sampleGMM(N, LM.rng(800 + s)).X, N)); const mu = LM.mean(mm); console.log(`true N=${N} MMD mean`, mu.toFixed(2), 'sd', Math.sqrt(LM.mean(mm.map((v) => (v - mu) ** 2))).toFixed(2)); }
}
if (part === 'all' || part === 'offref') {
    for (let s = 0; s < 8; s++) { const x = P11.noiseBank(256, 9000 + s); P11.sampleVP(P11.epsNet(M.vp), COS, x, 256, { K: 20, eta: 0 }); console.log('offref seed', 9000 + s, f(P11.quality(x, 256))); }
}
if (part === 'all' || part === 'ddpm') {
    for (const K of [5, 10, 20, 50, 100, 250]) { const x = noise.slice(); P11.sampleVP(P11.epsNet(M.vp), COS, x, 300, { K, eta: 1, rand: LM.rng(21) }); const y = noise.slice(); P11.sampleVP(P11.epsNet(M.vp), COS, y, 300, { K, eta: 0, rand: LM.rng(21) }); console.log(`K=${K}: DDPM ${f(P11.quality(x, 300))} | DDIM ${f(P11.quality(y, 300))}`); }
}
if (part === 'all' || part === 'sw6') {
    const Ks = [2, 5, 10, 20, 50, 100], jobs = [];
    [{ id: 'ddim', f: P11.epsNet(M.vp), sch: COS, eta: 0 }, { id: 'ddpm', f: P11.epsNet(M.vp), sch: COS, eta: 1 }, { id: 'exc', f: P11.epsExact(COS), sch: COS, eta: 0 }, { id: 'exl', f: P11.epsExact(P11.SCHED.linear), sch: P11.SCHED.linear, eta: 0 }]
        .forEach((se) => Ks.forEach((K) => jobs.push({ kind: 'vp', epsFn: se.f, sch: se.sch, K, eta: se.eta, n: 300, seed: 77, sid: se.id, metric: (x, n) => P11.quality(x, n) })));
    run(jobs).forEach((r) => console.log(`sw6 ${r.job.sid} K=${r.job.K}: ${f(r)}`));
}
if (part === 'all' || part === 'sw7') {
    for (let c = 0; c < 3; c++) {
        const gs = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
        const res = run(gs.map((g) => ({ kind: 'vp', epsFn: P11.epsNet(M.cfg, { cls: c, guide: g }), sch: COS, K: 30, eta: 0, n: 200, seed: 77, g, metric: (x, n) => P11.cfgQuality(x, n, c) })));
        console.log(`sw7 c=${c}: ` + res.map((r) => `s${r.job.g}: acc ${(100 * r.acc).toFixed(0)} cov ${(100 * r.cov).toFixed(0)} mmd ${r.mmd.toFixed(1)} r ${r.r.toFixed(2)}`).join(' | '));
    }
    for (const g of [1, 2]) { const x = noise.slice(); P11.sampleVP(P11.epsNet(M.cfg, { cls: 0, guide: g }), COS, x, 300, { K: 50, eta: 0 }); console.log(`cfg main c0 s${g}`, JSON.stringify(P11.cfgQuality(x, 300, 0))); }
    let rr = [0, 0, 0], nn = [0, 0, 0]; const D = P11.sampleGMM(3000, LM.rng(5)); for (let i = 0; i < 3000; i++) { rr[D.lab[i]] += Math.hypot(D.X[2 * i], D.X[2 * i + 1]); nn[D.lab[i]]++; } console.log('per-arm data mean radius', rr.map((v, k) => (v / nn[k]).toFixed(3)).join(' '));
}
if (part === 'all' || part === 'sw8') {
    const Ks = [1, 2, 3, 4, 8, 16, 32], jobs = [];
    Ks.forEach((K) => jobs.push({ kind: 'vp', epsFn: P11.epsNet(M.vp), sch: COS, K, eta: 0, n: 300, seed: 77, sid: 'vp', metric: (x, n) => P11.quality(x, n) }));
    Ks.forEach((K) => jobs.push({ kind: 'fm', net: M.fm, K, n: 300, seed: 77, sid: 'fm', metric: (x, n) => P11.quality(x, n) }));
    Ks.forEach((K) => jobs.push({ kind: 'fm', net: M.rf, K, n: 300, seed: 77, sid: 'rf', metric: (x, n) => P11.quality(x, n) }));
    run(jobs).forEach((r) => console.log(`sw8 ${r.job.sid} K=${r.job.K}: ${f(r)}`));
    const K = 40, n = 300;
    const ratio = { vp: P11.pathRatio(P11.sampleVP(P11.epsNet(M.vp), COS, noise.slice(), n, { K, keep: true }).traj, n), fm: P11.pathRatio(P11.sampleFM(M.fm, noise.slice(), n, { K, keep: true }).traj, n), rf: P11.pathRatio(P11.sampleFM(M.rf, noise.slice(), n, { K, keep: true }).traj, n) };
    const xe = noise.slice(), trE = [xe.slice()], ve = new Float32Array(2 * n); for (let k = 100; k >= 1; k--) { P11.velExact(xe, n, k / 100, ve); for (let q = 0; q < 2 * n; q++) xe[q] -= ve[q] / 100; trE.push(xe.slice()); }
    ratio.fmExact = P11.pathRatio(trE, n); ratio.vpExact = P11.pathRatio(P11.sampleVP(P11.epsExact(COS), COS, noise.slice(), n, { K: 100, keep: true }).traj, n);
    console.log('path ratios', JSON.stringify(Object.fromEntries(Object.entries(ratio).map(([k, v]) => [k, +v.toFixed(3)]))));
}
if (part === 'all' || part === 'sw9') {
    const D = P11.digitSets(), L = M.lat, m = 2000, zr = P11.encodeDigits(L, D.train.X.subarray(0, m * 256), m), yr = D.train.y.slice(0, m), zt = P11.encodeDigits(L, D.test.X, D.test.n);
    const pt = P11.knn(zt, D.test.n, zr, yr, m, 8); let ok = 0; for (let i = 0; i < D.test.n; i++) if (pt[i] === D.test.y[i]) ok++;
    let mse = 0; const rt = P11.decodeLatents(L, zt, D.test.n); for (let q = 0; q < rt.length; q++) mse += (rt[q] - D.test.X[q]) ** 2 / rt.length;
    console.log('latent knn real', (100 * ok / D.test.n).toFixed(1) + '%', 'recon MSE', mse.toFixed(4));
    const gs = [0, 0.5, 1, 1.5, 2, 3, 4, 5], jobs = [];
    gs.forEach((g) => { for (let c = 0; c < 10; c++) jobs.push({ kind: 'vp', epsFn: P11.epsNet(M.ldm, { cls: c, guide: g }), sch: COS, K: 30, eta: 0, n: 12, din: 8, clip: 4, seed: 900 + c, g, c, metric: (x, n) => { const p = P11.knn(x, n, zr, yr, m, 8); let k = 0; for (let i = 0; i < n; i++) if (p[i] === c) k++; return { ok: k, n }; } }); });
    const res = run(jobs); console.log('sw9 ' + gs.map((g) => { const r = res.filter((q) => q.job.g === g); return `s${g}: ${(100 * r.reduce((a, q) => a + q.ok, 0) / r.reduce((a, q) => a + q.n, 0)).toFixed(1)}%`; }).join(' | '));
}
if (part === 'all' || part === 'live') {
    const R = P11.newLiveRun({ sched: COS, data: P11.trainSet(), lr: 2e-3, seed: 11 }), t0 = performance.now();
    while (!R.finished) P11.liveWork(R, 1);
    const ev = R.evals; console.log('live:', ((performance.now() - t0) / 1000).toFixed(1) + ' s', 'evals', ev.length, ev.filter((_, i) => i % 25 === 0).map((e) => `${e[0]}:${e[1].toFixed(1)}/${(100 * e[2]).toFixed(1)}%`).join(' '), '| final', ev[ev.length - 1].map((v) => v.toFixed(3)).join(' '), '| loss(last 200)', LM.mean(R.tr.lossHist.slice(-200)).toFixed(4));
    console.log('floor', P11.lossFloor(COS, 20).total.toFixed(4));
}
