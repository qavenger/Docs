// p15_bench.js — 第 15 篇：测 LM.AG 上小 MLP 的前向 + 反向吞吐（决定 NeRF / Fourier-feature 网络的规模）
const LM = require('../llm_common.js');
const { AG } = LM;
function mlp(dims, rand) {
    const P = [];
    for (let l = 0; l + 1 < dims.length; l++) {
        P.push(AG.param('W' + l, dims[l], dims[l + 1], LM.init.he(rand, dims[l])));
        P.push(AG.param('b' + l, 1, dims[l + 1]));
    }
    return P;
}
function fwd(P, X) {
    let h = X;
    for (let l = 0; l < P.length; l += 2) { h = AG.linear(h, P[l], P[l + 1]); if (l + 2 < P.length) h = AG.relu(h); }
    return h;
}
const rand = LM.rng(1);
for (const [N, dims] of [[2048, [39, 64, 64, 64, 4]], [2048, [39, 48, 48, 4]], [2048, [39, 64, 64, 4]], [512, [26, 64, 64, 64, 3]], [512, [26, 32, 32, 3]], [256, [26, 64, 64, 3]], [512, [2, 64, 64, 3]]]) {
    const P = mlp(dims, rand), opt = new LM.AdamW(P, { lr: 1e-3, wd: 0 });
    const X = AG.tensor(N, dims[0]); for (let i = 0; i < X.data.length; i++) X.data[i] = rand() * 2 - 1;
    const Y = AG.tensor(N, dims[dims.length - 1]); for (let i = 0; i < Y.data.length; i++) Y.data[i] = rand();
    let macs = 0; for (let l = 0; l + 1 < dims.length; l++) macs += dims[l] * dims[l + 1];
    for (let it = 0; it < 5; it++) { LM.zeroGrad(P); const L = AG.mse(fwd(P, X), Y); AG.backward(L); opt.step(); }
    const t0 = performance.now(); const R = 20;
    for (let it = 0; it < R; it++) { LM.zeroGrad(P); const L = AG.mse(fwd(P, X), Y); AG.backward(L); opt.step(); }
    const ms = (performance.now() - t0) / R;
    const t1 = performance.now();
    for (let it = 0; it < R; it++) AG.noGrad(() => fwd(P, X));
    const msF = (performance.now() - t1) / R;
    console.log(`N=${N} dims=${dims.join('-')}  MACs/sample=${macs}  train ${ms.toFixed(2)} ms (${(3 * macs * N / ms / 1e6).toFixed(2)} GMAC/s)  fwd ${msF.toFixed(2)} ms (${(macs * N / msF / 1e6).toFixed(2)} GMAC/s)`);
}
