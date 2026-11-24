// 数据难度扫描：臂的粗细 σ、缠绕角，在固定训练预算下的采样质量
const args = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
global.P11_DATA_OPT = { sig: args.sig, wind: args.wind, gap: args.gap, r0: args.r0 };
const { LM, P11 } = require('./p11_load.js');
const data = (() => { const r = P11.sampleGMM(4096, LM.rng(1234)); return { X: r.X, lab: r.lab, n: 4096 }; })();
const sch = P11.SCHED[args.sched || 'cosine'];
const STEPS = args.steps || 4000;
const R = P11.trainer({ kind: args.kind || 'vp', sched: sch, data, width: args.W || 64, depth: args.D || 3, B: args.B || 128, lr: args.lr || 2e-3, steps: STEPS, seed: 1, warmup: 100, ff: args.ff ?? 0 });
const N = 600, REF = 1500, ref = P11.sampleGMM(REF, LM.rng(101)).X, syy = P11.mmd(ref.subarray(0, 2), 1, ref, REF).syy;
const lpRef = Array.from(P11.logp(ref, REF)).sort((a, b) => a - b), thr = lpRef[Math.floor(0.01 * REF)];
const off = (X, n) => { const l = P11.logp(X, n); let c = 0; for (let i = 0; i < n; i++) if (l[i] < thr) c++; return 100 * c / n; };
const ev = () => { const out = []; for (const [K, eta] of (args.kind === 'fm' ? [[4, 0], [16, 0], [64, 0]] : [[20, 0], [50, 0], [100, 1]])) { const x = new Float32Array(2 * N); const r2 = LM.rng(77); for (let i = 0; i < 2 * N; i++) x[i] = LM.randn(r2); if (args.kind === 'fm') P11.sampleFM(R.net, x, N, { K }); else P11.sampleVP(P11.epsNet(R.net), sch, x, N, { K, eta, rand: LM.rng(9) }); out.push(`K${K}${args.kind === 'fm' ? '' : 'η' + eta} ${P11.mmd(x, N, ref, REF, syy).mmd.toFixed(2)}/${off(x, N).toFixed(1)}%`); } return out.join(' '); };
const t0 = performance.now(); let line = `sig=${P11.SIG} wind=${args.wind ?? 1.2} M=${P11.GMM.M} ff=${args.ff ?? 0} W=${args.W || 64} B=${args.B || 128} kind=${args.kind || 'vp'}:`;
for (let s = 1; s <= STEPS; s++) { R.step(); if (s % (args.every || STEPS) === 0) line += `\n   @${s}: ${ev()}`; }
console.log(line + `  (${((performance.now() - t0) / STEPS).toFixed(1)} ms/step)`);
