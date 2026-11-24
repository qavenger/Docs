// p11_verify.js — 校验内嵌权重（P11_W）与关键结论；量 Step 03 每帧开销
const { LM, P11, P11_W } = require('./p11_load.js');
const ld = (k) => P11.loadNet(P11_W[k].cfg, P11_W[k].b64);
const vp = ld('vp'), fm = ld('fm'), rf = ld('rf'), cfg = ld('cfg');
const N = 400, nb = () => P11.noiseBank(N, 77);
const q = (x) => { const r = P11.quality(x, N); return `MMD ${r.mmd.toFixed(2)} off ${(100 * r.off).toFixed(1)}%`; };
console.log('params vp', P11.numParams(vp), 'cfg', P11.numParams(cfg));
for (const K of [10, 20, 50]) { const x = nb(); P11.sampleVP(P11.epsNet(vp), P11.SCHED.cosine, x, N, { K, eta: 0 }); console.log(`vp DDIM K${K}: ${q(x)}`); }
{ const x = nb(); P11.sampleVP(P11.epsNet(vp), P11.SCHED.cosine, x, N, { K: 100, eta: 1, rand: LM.rng(3) }); console.log(`vp DDPM K100: ${q(x)}`); }
for (const K of [1, 2, 4, 8]) { const a = nb(), b = nb(); P11.sampleFM(fm, a, N, { K }); P11.sampleFM(rf, b, N, { K }); console.log(`K${K}: FM ${q(a)} | 2-RF ${q(b)}`); }
// 路径弯曲度：解析 FM 速度场 vs 解析 VP（cosine）PF-ODE vs 网络
const n = 300, x0 = P11.noiseBank(n, 55);
const fmExact = () => { const x = x0.slice(), traj = [x.slice()], v = new Float32Array(2 * n); for (let k = 100; k >= 1; k--) { P11.velExact(x, n, k / 100, v); for (let qq = 0; qq < 2 * n; qq++) x[qq] -= v[qq] / 100; traj.push(x.slice()); } return P11.pathRatio(traj, n); };
const vpEx = (s) => P11.pathRatio(P11.sampleVP(P11.epsExact(P11.SCHED[s]), P11.SCHED[s], x0.slice(), n, { K: 100, keep: true }).traj, n);
console.log(`path/chord (K=100): VP-cos exact ${vpEx('cosine').toFixed(3)} | VP-lin exact ${vpEx('linear').toFixed(3)} | FM exact ${fmExact().toFixed(3)} | FM net ${P11.pathRatio(P11.sampleFM(fm, x0.slice(), n, { K: 100, keep: true }).traj, n).toFixed(3)} | 2-RF net ${P11.pathRatio(P11.sampleFM(rf, x0.slice(), n, { K: 100, keep: true }).traj, n).toFixed(3)} | VP-cos net ${P11.pathRatio(P11.sampleVP(P11.epsNet(vp), P11.SCHED.cosine, x0.slice(), n, { K: 100, keep: true }).traj, n).toFixed(3)}`);
// 实时训练每帧开销
const R = P11.newLiveRun({ sched: P11.SCHED.cosine, data: P11.trainSet(), lr: 2e-3, seed: 11 });
for (let f = 0; f < 30; f++) P11.liveWork(R, 1);
const t0 = performance.now(); for (let f = 0; f < 200; f++) P11.liveWork(R, 1); console.log('liveWork spf=1:', ((performance.now() - t0) / 200).toFixed(2), 'ms/frame, rolls', R.rolls, 'last', R.last && R.last.mmd.toFixed(1));
// latent
const L = { ae: P11.createAE(1), mu: Float32Array.from(P11_W.latent.mu), sd: Float32Array.from(P11_W.latent.sd) }; LM.gpt.loadWeights({ params: L.ae.params }, P11_W.latent.ae);
const ldm = P11.loadNet(P11_W.latent.cfg, P11_W.latent.b64), D = P11.digitSets();
const zr = P11.encodeDigits(L, D.train.X, 2000), zt = P11.encodeDigits(L, D.test.X, D.test.n), pt = P11.knn(zt, D.test.n, zr, D.train.y, 2000, 8);
let a = 0; for (let i = 0; i < D.test.n; i++) if (pt[i] === D.test.y[i]) a++; console.log('latent kNN (2000 ref) on real test:', (100 * a / D.test.n).toFixed(1) + '%');
for (const s of [0, 1, 2, 3, 4]) { let acc = 0; const t1 = performance.now(); for (let c = 0; c < 10; c++) { const x = P11.noiseBank(20, 300 + c, 8); P11.sampleVP(P11.epsNet(ldm, { cls: c, guide: s }), P11.SCHED.cosine, x, 20, { K: 50, eta: 0, clip: 4, din: 8 }); const p = P11.knn(x, 20, zr, D.train.y, 2000, 8); for (let i = 0; i < 20; i++) if (p[i] === c) acc++; } console.log(`latent CFG s=${s}: kNN acc ${(acc / 2).toFixed(1)}%  (${(performance.now() - t1).toFixed(0)} ms)`); }
