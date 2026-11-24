const { LM, P11 } = require('./p11_load.js');
const N = +process.argv[2] || 600;
const vals = []; let t0 = performance.now();
for (let s = 0; s < 12; s++) vals.push(P11.mmdTrue(P11.sampleGMM(N, LM.rng(900 + s)).X, N));
const ms = (performance.now() - t0) / 12, mean = vals.reduce((a, b) => a + b) / vals.length, sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
console.log(`N=${N} true samples: mean ${mean.toFixed(3)} sd ${sd.toFixed(3)}  (${ms.toFixed(1)} ms each)  per-class:`, [0, 1, 2].map((c) => P11.mmdTrue(P11.sampleGMM(N, LM.rng(5), c).X, N, c).toFixed(3)).join(' '));
const rand = LM.rng(3);
for (const b of [0.03, 0.05, 0.1]) { const X = P11.sampleGMM(N, LM.rng(31)).X.map((v) => v + b * LM.randn(rand)); console.log(`  blur ${b}: ${P11.mmdTrue(X, N).toFixed(2)}`); }
const sh = P11.sampleGMM(N, LM.rng(32)).X.map((v) => v * 0.9); console.log('  shrink ×0.9:', P11.mmdTrue(sh, N).toFixed(2));
const G = new Float32Array(2 * N).map(() => 0.51 * LM.randn(rand)); console.log('  single Gaussian (σ=0.51):', P11.mmdTrue(G, N).toFixed(2));
const Z = new Float32Array(2 * N).map(() => LM.randn(rand)); console.log('  N(0,I):', P11.mmdTrue(Z, N).toFixed(2));
