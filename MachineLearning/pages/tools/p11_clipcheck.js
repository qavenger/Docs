// 最终内嵌模型上「不截断 x̂₀」的后果（Step 05 的 warn）
const { LM, P11 } = require('./p11_load.js');
const M = P11.loadModels(), COS = P11.SCHED.cosine;
for (const clip of [1.25, 1e9]) for (const K of [10, 20, 50]) { const x = P11.noiseBank(300, 77); P11.sampleVP(P11.epsNet(M.vp), COS, x, 300, { K, eta: 0, clip }); const q = P11.quality(x, 300); let far = 0; for (let i = 0; i < 300; i++) if (Math.hypot(x[2 * i], x[2 * i + 1]) > 1.6) far++; console.log(`clip ${clip} K${K}: MMD ${q.mmd.toFixed(1)} off ${(100 * q.off).toFixed(1)}% |x|>1.6: ${(100 * far / 300).toFixed(1)}%`); }
const x = P11.noiseBank(256 * 4, 9000); let acc = 0; for (let s = 0; s < 4; s++) { const y = P11.noiseBank(256, 9000 + s); P11.sampleVP(P11.epsNet(M.vp), COS, y, 256, { K: 20, eta: 0 }); acc += P11.quality(y, 256).mmd / 4; } console.log('offref mean of 4 seeds', acc.toFixed(2));
