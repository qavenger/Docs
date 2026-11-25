// p03_final.js — 第 03 篇：用页面同一份代码、同样的设置与种子，把页面里所有实时训练的网络一口气训练到终点，打印最终数字。
// 页面正文（Step 05 / 07 / 08 / 09 的说明）引用的就是这里的输出。用法（在 pages/ 下）：node tools/p03_final.js [seed] [parts = 05,07,08,09]
const { P08, LM } = require('./p03_proto.js');
const seed = +(process.argv[2] || 20251123), parts = (process.argv[3] || '05,07,08,09').split(',');
const S = { seed, data: P08.makeData(LM.rng(seed)) };
const pct = (v) => (100 * v).toFixed(1) + '%';
const t0 = Date.now();
const toCap = (R) => { while (R.step < R.cap) P08.trainStep(R); };

// ── Step 05：CNN vs MLP ──
S.main = P08.mainRuns(S);
if (parts.includes('05') || parts.includes('07')) { toCap(S.main.cnn); toCap(S.main.mlp); }
if (parts.includes('05')) [S.main.cnn, S.main.mlp].forEach((R, k) => {
    const te = P08.evaluate(R.m, S.data.te.X, S.data.te.y, S.data.allTe), tr = P08.evaluate(R.m, S.data.tr.X, S.data.tr.y, S.data.evalTr);
    console.log(`Step 05 ${k ? 'MLP' : 'CNN'}: ${P08.numParams(R.m.params)} params, ${R.step} steps → test ${pct(te.acc)} (${Math.round(500 * (1 - te.acc))} errors, loss ${te.loss.toFixed(3)}), train(500) ${pct(tr.acc)}`);
});

// ── Step 07：增强模型 + 平移测试（页面的测试集：前 100 张、SHIFTS），以及全部 500 张上的水平平移 ──
if (parts.includes('07')) for (const aug of [2, 3]) {
    const A = P08.augRuns(S, aug); toCap(A.cnn); toCap(A.mlp);
    const runs = [S.main.mlp, S.main.cnn, A.mlp, A.cnn], names = ['MLP', 'CNN', `MLP+aug${aug}`, `CNN+aug${aug}`], E = {};
    P08.shiftEvalBegin(E, runs); P08.shiftEvalWork(E, S.data, 1e9);
    console.log(`Step 07 (aug ±${aug}), first ${P08.NSHIFT} test images:`);
    runs.forEach((R, k) => {
        const cx = P08.shiftCurve(E.res.acc[k], 'x'), cy = P08.shiftCurve(E.res.acc[k], 'y');
        const full = [-4, -3, -2, -1, 0, 1, 2, 3, 4].map((d) => P08.evaluate(R.m, S.data.te.X, S.data.te.y, S.data.allTe, [d, 0]).acc);
        console.log(`  ${names[k].padEnd(10)} dx ${cx.y.map((v) => (100 * v).toFixed(0).padStart(4)).join('')} | dy ${cy.y.map((v) => (100 * v).toFixed(0).padStart(4)).join('')} | all 500, dx: ${full.map((v) => (100 * v).toFixed(0).padStart(4)).join('')}`);
    });
}

// ── Step 08：U-Net vs 去掉跳连（σ = 0.3 / 0.5 / 0.8） ──
if (parts.includes('08')) for (const sigma of [0.3, 0.5, 0.8]) {
    const U = P08.unetRuns(S, sigma);
    U.runs.forEach((R) => { while (R.step < R.cap) P08.denoiseStep(R); });
    P08.unetEvalBegin(U); P08.unetEvalWork(U, 1e9);
    console.log(`Step 08 σ = ${sigma}: noisy ${U.base.noisy.toFixed(2)} dB, best blur ${U.base.blur.toFixed(2)} dB, U-Net ${P08.last(U.hist.psnr[0]).toFixed(2)} dB (${P08.numParams(U.runs[0].m.params)} params), no-skip ${P08.last(U.hist.psnr[1]).toFixed(2)} dB (${P08.numParams(U.runs[1].m.params)} params)`);
}

// ── Step 09：普通 vs 残差（零初始化），无归一化 / BatchNorm ──
if (parts.includes('09')) for (const bn of [false, true]) for (const L of [4, 8, 16, 24]) {
    const G = P08.deepRuns(S, L, bn);
    while (G.runs.some((R) => R.step < R.cap)) P08.deepTick(G, 1);
    const dead = (R) => R.actStats.filter((a) => a.zero >= 0.999).length;
    console.log(`Step 09 L = ${String(L).padStart(2)} ${bn ? 'BN  ' : 'none'}: plain loss ${G.runs[0].ema.toFixed(3)} test ${pct(P08.last(G.runs[0].hist.testA))} (dead layers ${dead(G.runs[0])}) | residual loss ${G.runs[1].ema.toFixed(3)} test ${pct(P08.last(G.runs[1].hist.testA))}`);
}
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
