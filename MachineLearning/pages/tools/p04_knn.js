// p04_knn.js — 潜空间的类别结构：测试集 500 个 2 维编码上的留一 5-NN 准确率（PCA-2 / AE / VAE），以及 B = 16 / 32 的对比
const { P09, LM } = require('./p04_proto.js');
const data = P09.makeData(LM.rng(20251123)); P09.initEvalEps(20251123);
function knn(z, y, n, k = 5) {
    let ok = 0;
    for (let i = 0; i < n; i++) {
        const d = []; for (let j = 0; j < n; j++) if (j !== i) d.push([(z[2 * i] - z[2 * j]) ** 2 + (z[2 * i + 1] - z[2 * j + 1]) ** 2, y[j]]);
        d.sort((a, b) => a[0] - b[0]); const v = new Array(10).fill(0); for (let q = 0; q < k; q++) v[d[q][1]]++;
        if (LM.argmax(v) === y[i]) ok++;
    }
    return ok / n;
}

// 2 维编码与「墨水量」「倾斜」的相关性：AE 到底编码了什么？
function factorCorr (z, X, n) {
    const ink = new Float64Array(n), slant = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 0, sx = 0, sy = 0, sxy = 0, syy = 0; for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const v = X[i * 256 + y * 16 + x]; s += v; sx += v * x; sy += v * y; } const cx = sx / s, cy = sy / s; for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const v = X[i * 256 + y * 16 + x]; sxy += v * (x - cx) * (y - cy); syy += v * (y - cy) ** 2; } ink[i] = s; slant[i] = sxy / syy; }
    const corr = (a, b) => { const ma = LM.mean(a), mb = LM.mean(b); let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return sab / Math.sqrt(saa * sbb); };
    const z0 = Float64Array.from({ length: n }, (_, i) => z[2 * i]), z1 = Float64Array.from({ length: n }, (_, i) => z[2 * i + 1]);
    return { ink: [corr(z0, ink), corr(z1, ink)], slant: [corr(z0, slant), corr(z1, slant)] };
}
const P = P09.pca(data.tr, data.te), zp = new Float32Array(1000); for (let i = 0; i < 500; i++) { zp[2 * i] = P.proj[i * 256]; zp[2 * i + 1] = P.proj[i * 256 + 1]; }
console.log('PCA-2 5-NN acc', knn(zp, data.te.y, 500).toFixed(3));
for (const a of process.argv.slice(2)) {
    const c = JSON.parse(a);
    const R = P09.makeRun(c.kind, { data, seed: 20251145, beta: c.beta ?? 1, cap: c.steps || 3000, B: c.B || 32, lr: c.lr || 2e-3, arch: c.arch, sched: c.sched || 'cos' });
    const t = Date.now(); while (R.step < R.cap) P09.trainStep(R); const ms = (Date.now() - t) / R.cap;
    P09.evalNow(R);
    const fc = factorCorr(R.res.z, data.te.X, 500);
    console.log(a.padEnd(40), ms.toFixed(2), 'ms/step', `corr(z, ink) ${fc.ink.map((v) => v.toFixed(2))} corr(z, slant) ${fc.slant.map((v) => v.toFixed(2))}`, `mse ${R.res.mse.toFixed(4)} rec ${R.res.rec.toFixed(1)}` + (c.kind === 'vae' ? ` kl ${R.res.kl.toFixed(2)}` : ''), '5-NN acc', knn(R.res.z, data.te.y, 500).toFixed(3));
}
