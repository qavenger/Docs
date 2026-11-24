// 解析的「不可约损失」：E||ε − E[ε|x_t]||²/2（每维），按 t 分箱；以及 t 均匀分布时的总下限
const { LM, P11 } = require('./p11_load.js');
for (const kind of ['cosine', 'linear']) {
    const sch = P11.SCHED[kind], rand = LM.rng(3), n = 3000, X0 = P11.sampleGMM(n, rand).X, xt = new Float32Array(2 * n), eps = new Float32Array(2 * n), x0h = new Float32Array(2 * n);
    let tot = 0; const rows = [];
    for (const i of [1, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) {
        const a = sch.a[i], s = sch.s[i];
        for (let q = 0; q < 2 * n; q++) { eps[q] = LM.randn(rand); xt[q] = a * X0[q] + s * eps[q]; }
        P11.exact(a, s * s, xt, n, { x0: x0h });
        let e = 0; for (let q = 0; q < 2 * n; q++) { const epred = (xt[q] - a * x0h[q]) / s; e += (eps[q] - epred) ** 2; }
        rows.push(`t=${i}:${(e / (2 * n)).toFixed(3)}`);
    }
    // uniform-t average (MC over t)
    let acc = 0; const NT = 4000;
    for (let k = 0; k < NT; k++) { const i = 1 + Math.floor(rand() * 1000), a = sch.a[i], s = sch.s[i]; const j = Math.floor(rand() * n); const x = new Float32Array(2), e2 = [LM.randn(rand), LM.randn(rand)]; x[0] = a * X0[2 * j] + s * e2[0]; x[1] = a * X0[2 * j + 1] + s * e2[1]; P11.exact(a, s * s, x, 1, { x0: x0h }); for (let d = 0; d < 2; d++) acc += (e2[d] - (x[d] - a * x0h[d]) / s) ** 2 / 2; }
    console.log(kind, 'floor(uniform t) =', (acc / NT).toFixed(4)); console.log('  ' + rows.join(' '));
}
