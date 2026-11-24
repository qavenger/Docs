// p12_pca.js — Step 02 的实验：同样的压缩倍数下，逐帧 patch（pt=1）与时空 tubelet（pt=2、4）做 PCA 的重建误差
const { LM, P12 } = require('./p12_load.js');
const tr = P12.makeSet(300, LM.rng(21)), te = P12.makeSet(60, LM.rng(22));
const t0 = Date.now();
const pcas = [1, 2, 4].map((pt) => P12.fitPCA(tr.X, tr.n, pt));
console.log(`fit 3 PCAs in ${Date.now() - t0} ms`);
const out = new Float32Array(P12.VD);
for (const rho of [1, 2, 4, 8, 16, 32]) {
    const row = pcas.map((p) => {
        const k = Math.max(1, Math.round(p.D / rho)); let e = 0;
        if (p.D / rho < 1) return '   —   ';
        for (let v = 0; v < te.n; v++) { P12.pcaRecon(p, te.X, v * P12.VD, k, out, 0); for (let i = 0; i < P12.VD; i++) e += (out[i] - te.X[v * P12.VD + i]) ** 2; }
        return `${(e / te.n / P12.VD).toFixed(4)} (k=${k})`;
    });
    console.log(`ratio ${String(rho).padStart(2)}×: ` + row.join('   '));
}
pcas.forEach((p) => { let tot = 0; for (const v of p.vals) tot += v; let c = 0; const ks = []; for (let i = 0; i < p.D; i++) { c += p.vals[i]; ks.push((c / tot).toFixed(2)); } console.log(`pt=${p.pt} explained variance (first 8): ${ks.slice(0, 8).join(' ')}`); });
