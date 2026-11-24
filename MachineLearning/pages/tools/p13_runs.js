// p13_runs.js — 第 13 篇：用页面同一份代码跑完各个训练 run（第 05/06/07/09 步），统计每帧耗时（平均 / 最大）和最终结果
// 用法（在 pages/ 下）：node tools/p13_runs.js [codec|lm|dif|asr|all] [seed=1]
const { P13, LM } = require('./p13_proto.js');
const which = process.argv[2] || 'all', seed = +(process.argv[3] || 1);
const D = P13.codecData();
function drive(name, R, tick, done, report) {
    const ts = []; let t0 = P13.now(), fr = 0;
    while (!done(R)) { const a = P13.now(); tick(R); ts.push(P13.now() - a); fr++; if (fr > 100000) break; }
    ts.sort((a, b) => a - b);
    const mean = ts.reduce((a, b) => a + b, 0) / ts.length;
    console.log(`[${name}] ${fr} frames, total ${((P13.now() - t0) / 1000).toFixed(1)} s, per frame mean ${mean.toFixed(2)} ms, p95 ${ts[Math.floor(ts.length * 0.95)].toFixed(2)}, max ${ts[ts.length - 1].toFixed(1)} ms`);
    report(R);
}
if (which === 'codec' || which === 'all') {
    for (const K of [16, 32, 64, 128]) drive(`codec K=${K}`, P13.newCodecRun(K, seed), (R) => P13.codecTick(R, D), (R) => R.step >= P13.CODEC.STEPS && R.last && R.last.step === R.step, (R) => {
        console.log('   eval-clip SNR by layers', R.last.snr.map((v) => v.toFixed(1)).join(' '), '| all test melodies', P13.codecTestSNR(R.Q, D, 8).map((v) => v.toFixed(1)).join(' '));
        console.log('   codebook usage on eval clip (layer 1..8 distinct codes):', R.last.use.map((h) => h.filter((c) => c > 0).length).join(' '), `of ${K}; kbps per layer ${(P13.CODEC.FPS * Math.log2(K) / 1000).toFixed(3)}`);
    });
}
if (which === 'lm' || which === 'all') {
    drive('lm', P13.newLMRun(seed), (R) => P13.lmTick(R, D), (R) => R.phase === 2 && R.step >= P13.LMC.STEPS && R.evals[R.evals.length - 1].step === R.step, (R) => {
        console.log('   val CE per codebook over time:'); R.evals.filter((e, i) => i % 4 === 0 || i === R.evals.length - 1).forEach((e) => console.log(`     step ${e.step}: ${e.val.map((v) => v.toFixed(3)).join(' ')}`));
        console.log(`   train ms/step (EMA) ${R.ms.toFixed(2)}; distinct frames ${R.seen.size}`);
        for (const temp of [0.7, 1.0, 1.3]) {
            const res = [];
            for (let w = 0; w < P13.CODEC.NTEST; w++) {
                const t0 = P13.now(), g = P13.lmGenerate(R, w, temp, 100 + w), ms = P13.now() - t0;
                const beats = g.pitch.filter((p) => p.energy > 1e-3), clar = beats.reduce((a, p) => a + p.clarity, 0) / beats.length, val = g.valid.reduce((a, b) => a + b, 0) / g.F;
                res.push({ ms, clar, val, notes: g.pitch.map((p) => (p.energy > 1e-3 ? P13.noteName(p.midi) : '·')).join(' ') });
            }
            console.log(`   T=${temp}: clarity ${res.map((r) => r.clar.toFixed(2)).join(' ')} | valid ${res.map((r) => (100 * r.val).toFixed(0) + '%').join(' ')} | gen ms ${res.map((r) => r.ms.toFixed(0)).join(' ')}`);
            if (temp === 1.0) res.slice(0, 3).forEach((r, i) => console.log(`     sample ${i}: ${r.notes}`));
        }
        const real = P13.pitchReport(R.test[0].x, P13.CODEC.BEAT); console.log('   real test 0:', real.map((p) => P13.noteName(p.midi)).join(' '));
    });
}
if (which === 'dif' || which === 'all') {
    drive('dif', P13.newDifRun(seed), (R) => P13.difTick(R), (R) => R.phase === 2 && R.step >= P13.DIFR.STEPS && R.samples && R.samples.step === R.step, (R) => {
        const L = R.loss, avg = (a, b) => { let s = 0; for (let i = a; i < b; i++) s += L[i]; return s / (b - a); };
        console.log(`   PCA explained ${(100 * R.pca.explained).toFixed(1)}%; loss first 200 ${avg(0, 200).toFixed(3)} last 200 ${avg(L.length - 200, L.length).toFixed(3)}`);
        console.log('   samples NN dist', R.samples.near.map((q) => q.d.toFixed(2)).join(' '), '| NN f0 of nearest train', R.samples.near.map((q) => R.meta[q.i].f0.toFixed(0)).join(' '));
        // real held-out reference: PCA-recon of fresh notes, NN distance to train
        const te = P13.difDataset(16, 9090), rec = P13.difDecode(R.pca, P13.difEncode(R.pca, te.X, 16), 16), Dd = P13.DIF.D, dd = [];
        for (let i = 0; i < 16; i++) { let b = Infinity; for (let j = 0; j < P13.DIFR.N; j++) { let d = 0; for (let q = 0; q < Dd; q++) d += (rec[i * Dd + q] - R.X[j * Dd + q]) ** 2; b = Math.min(b, d); } dd.push(Math.sqrt(b / Dd)); }
        console.log('   held-out real (PCA recon) NN dist', dd.map((v) => v.toFixed(2)).join(' '));
        const t0 = P13.now(), w = P13.difToWave(R.samples.imgs.subarray(0, Dd)); console.log(`   vocode one sample ${(P13.now() - t0).toFixed(1)} ms; pitch of samples:`, Array.from({ length: 8 }, (_, i) => { const y = P13.difToWave(R.samples.imgs.subarray(i * Dd, (i + 1) * Dd)); return f0est(y).toFixed(0); }).join(' '));
    });
}
function f0est(y) {   // 自相关粗估基频（只用于报告）
    let best = 0, bl = 0; const n = y.length;
    for (let lag = 11; lag < 45; lag++) { let s = 0, e0 = 0, e1 = 0; for (let i = 0; i + lag < n; i++) { s += y[i] * y[i + lag]; e0 += y[i] * y[i]; e1 += y[i + lag] ** 2; } const r = s / Math.sqrt(e0 * e1 + 1e-20); if (r > best) { best = r; bl = lag; } }
    return bl ? P13.SR / bl : 0;
}
if (which === 'asr' || which === 'all') {
    drive('asr', P13.newAsrRun(seed), (R) => P13.asrTick(R), (R) => R.phase === 1 && R.step >= P13.ASRR.STEPS && !R.ev && R.evals.length && R.evals[R.evals.length - 1].step === R.step, (R) => {
        console.log('   test acc over time:', R.evals.filter((e, i) => i % 3 === 0 || i === R.evals.length - 1).map((e) => `${e.step}:${(100 * e.acc).toFixed(1)}`).join(' '));
        const e = R.evals[R.evals.length - 1]; console.log('   confusion (a e i o u):'); for (let r = 0; r < 5; r++) console.log('     ' + Array.from(e.conf.subarray(r * 5, r * 5 + 5)).join('\t'));
        for (const [text, voice] of [['啊诶衣喔乌', { f0: 120, vtl: 1 }], ['啊诶衣喔乌', { f0: 220, vtl: 1.17 }], ['我爱你', { f0: 120, vtl: 1 }], ['猫爱鱼', { f0: 220, vtl: 1.17 }]]) {
            const t = P13.tts(text, voice), sc = P13.asrScan(R.m, t.x);
            let s = ''; for (let w = 0; w < sc.nW; w++) s += sc.en[w] > 0.1 ? 'aeiou'[LM.argmax(sc.post, w * 5, 5)] : '·';
            console.log(`   TTS ${text} f0=${voice.f0}: ${s}`);
        }
    });
}
