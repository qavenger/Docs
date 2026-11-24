// p11_bench.js — 第 11 篇：测 AG 训练一步 / 手写推理 GEMM 的耗时，决定网络大小与采样预算
const LM = require('../llm_common.js');
const { AG } = LM;
function mk(din, dc, W, depth, rand) {
    const P = []; const p = (n, r, c, lim) => { const t = AG.param(n, r, c, LM.init.uniform(rand, lim)); P.push(t); return t; };
    const L = []; let fin = din;
    for (let l = 0; l < depth; l++) { L.push({ W: p('W' + l, fin, W, 1 / Math.sqrt(fin + dc)), U: p('U' + l, dc, W, 1 / Math.sqrt(fin + dc)), b: p('b' + l, 1, W, 1 / Math.sqrt(fin + dc)) }); fin = W; }
    const Wo = p('Wo', W, din, 1 / Math.sqrt(W)), bo = p('bo', 1, din, 1 / Math.sqrt(W));
    return { P, L, Wo, bo, din, dc, W };
}
function fwd(m, X, Cn) {
    let h = X;
    for (const l of m.L) h = AG.silu(AG.add(AG.linear(h, l.W, l.b), AG.matmul(Cn, l.U)));
    return AG.linear(h, m.Wo, m.bo);
}
function bench(W, depth, B, dc) {
    const rand = LM.rng(1), m = mk(2, dc, W, depth, rand), opt = new LM.AdamW(m.P, { lr: 1e-3, b2: 0.999, wd: 0 });
    const X = AG.tensor(B, 2), Y = AG.tensor(B, 2), C = AG.tensor(B, dc);
    for (let i = 0; i < X.data.length; i++) { X.data[i] = LM.randn(rand); Y.data[i] = LM.randn(rand); }
    for (let i = 0; i < C.data.length; i++) C.data[i] = Math.sin(i);
    const step = () => { opt.zeroGrad(); const out = fwd(m, X, C); const L = AG.mse(out, Y); AG.backward(L); opt.step(); };
    for (let k = 0; k < 30; k++) step();
    const n = 200, t0 = performance.now(); for (let k = 0; k < n; k++) step();
    const ms = (performance.now() - t0) / n;
    const np = m.P.reduce((s, p) => s + p.size, 0);
    console.log(`train W=${W} depth=${depth} B=${B} dc=${dc}: ${ms.toFixed(2)} ms/step, params ${np}`);
}
// 手写推理：所有行共享同一个条件向量（采样时同一时刻）
function infer(m, X, n, cvec, out, bufs) {
    let hin = X, fin = m.din;
    m.L.forEach((l, li) => {
        const Wd = l.W.data, Ud = l.U.data, bd = l.b.data, W = m.W, bias = bufs.bias, h = bufs.h[li & 1];
        for (let j = 0; j < W; j++) { let s = bd[j]; for (let q = 0; q < m.dc; q++) s += cvec[q] * Ud[q * W + j]; bias[j] = s; }
        for (let i = 0; i < n; i++) {
            const o = i * W, xi = i * fin;
            for (let j = 0; j < W; j++) h[o + j] = bias[j];
            for (let p = 0; p < fin; p++) { const a = hin[xi + p]; const wr = p * W; for (let j = 0; j < W; j++) h[o + j] += a * Wd[wr + j]; }
            for (let j = 0; j < W; j++) { const v = h[o + j]; h[o + j] = v / (1 + Math.exp(-v)); }
        }
        hin = h; fin = W;
    });
    const Wo = m.Wo.data, bo = m.bo.data, d = m.din;
    for (let i = 0; i < n; i++) { for (let j = 0; j < d; j++) { let s = bo[j]; for (let p = 0; p < fin; p++) s += hin[i * fin + p] * Wo[p * d + j]; out[i * d + j] = s; } }
    return out;
}
function benchInfer(W, depth, n, dc) {
    const rand = LM.rng(2), m = mk(2, dc, W, depth, rand);
    const X = new Float32Array(2 * n).map(() => LM.randn(rand)), out = new Float32Array(2 * n), cvec = new Float32Array(dc).map((_, i) => Math.cos(i));
    const bufs = { bias: new Float32Array(W), h: [new Float32Array(n * W), new Float32Array(n * W)] };
    for (let k = 0; k < 20; k++) infer(m, X, n, cvec, out, bufs);
    const reps = 200, t0 = performance.now(); for (let k = 0; k < reps; k++) infer(m, X, n, cvec, out, bufs);
    const ms = (performance.now() - t0) / reps;
    // check vs AG
    const C = AG.tensor(n, dc); for (let i = 0; i < n; i++) C.data.set(cvec, i * dc);
    const ref = AG.noGrad(() => fwd(m, AG.tensor(n, 2, X), C)).data; let md = 0; for (let i = 0; i < ref.length; i++) md = Math.max(md, Math.abs(ref[i] - out[i]));
    const flop = n * 2 * (2 * W + (depth - 1) * W * W + W * 2);
    console.log(`infer W=${W} depth=${depth} n=${n}: ${ms.toFixed(2)} ms/eval  (${(flop / ms / 1e6).toFixed(2)} GFLOP/s)  maxdiff vs AG ${md.toExponential(1)}`);
}
for (const [W, d, B] of [[48, 3, 128], [64, 3, 64], [64, 3, 128], [64, 3, 256], [96, 3, 128], [128, 3, 128], [64, 4, 128]]) bench(W, d, B, 20);
for (const [W, d] of [[48, 3], [64, 3], [96, 3], [128, 3]]) benchInfer(W, d, 400, 20);
