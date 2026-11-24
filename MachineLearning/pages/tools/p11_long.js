// 长训练：离线模型需要多少步 / 多大；EMA 的作用
const args = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
global.P11_DATA_OPT = { sig: args.sig ?? 0.1, wind: args.wind ?? 1.0, gap: args.gap ?? 0.14 };
const { LM, P11 } = require('./p11_load.js');
const data = (() => { const r = P11.sampleGMM(4096, LM.rng(1234)); return { X: r.X, lab: r.lab, n: 4096 }; })();
const KIND = args.kind || 'vp', sch = P11.SCHED[args.sched || 'cosine'], STEPS = args.steps || 20000;
const R = P11.trainer({ kind: KIND, sched: sch, data, width: args.W || 64, depth: args.D || 3, B: args.B || 128, lr: args.lr || 2e-3, steps: STEPS, seed: 1, warmup: 200, ff: args.ff ?? 0, ema: args.ema ?? 0.999 });
const N = 600, REF = 1500, ref = P11.sampleGMM(REF, LM.rng(101)).X, syy = P11.mmd(ref.subarray(0, 2), 1, ref, REF).syy;
const lpRef = Array.from(P11.logp(ref, REF)).sort((a, b) => a - b), thr = lpRef[Math.floor(0.01 * REF)];
const stats = (X, n) => { const l = P11.logp(X, n); let off = 0, outer = 0; for (let i = 0; i < n; i++) { if (l[i] < thr) off++; if (Math.hypot(X[2 * i], X[2 * i + 1]) > 0.9) outer++; } return `${P11.mmd(X, n, ref, REF, syy).mmd.toFixed(2)}/${(100 * off / n).toFixed(1)}%/r>.9:${(outer / n).toFixed(2)}`; };
const ev = (net) => { const out = []; for (const [K, eta] of (KIND === 'fm' ? [[2, 0], [8, 0], [32, 0]] : [[10, 0], [20, 0], [50, 0], [100, 1]])) { const x = new Float32Array(2 * N); const r2 = LM.rng(77); for (let i = 0; i < 2 * N; i++) x[i] = LM.randn(r2); if (KIND === 'fm') P11.sampleFM(net, x, N, { K }); else P11.sampleVP(P11.epsNet(net), sch, x, N, { K, eta, rand: LM.rng(9), clip: 1.25 }); out.push(`K${K}${KIND === 'fm' ? '' : 'η' + eta} ${stats(x, N)}`); } return out.join('  '); };
console.log(`${KIND} sig=${P11.SIG} M=${P11.GMM.M} W=${args.W || 64} ff=${args.ff ?? 0} B=${args.B || 128} lr=${args.lr || 2e-3} steps=${STEPS}  data r>.9: ${stats(P11.sampleGMM(N, LM.rng(5)).X, N)}`);
const t0 = performance.now(); let acc = 0;
for (let s = 1; s <= STEPS; s++) { acc += R.step().loss; if ([2000, 4000, 8000, 12000, 20000, 30000, 40000].includes(s) || s === STEPS) { console.log(`@${s} loss ${(acc / 1000).toFixed(4)} ${((performance.now() - t0) / s).toFixed(1)}ms/step\n   raw: ${ev(R.net)}\n   ema: ${ev(R.ema)}`); } if (s % 1000 === 0) acc = 0; }
