// p15_gs_check.js — 3DGS 前向 / 反向的有限差分梯度校验（L = Σ wt·img，wt 随机）。覆盖：μ、q、ŝ、ô、f 全部 14 个参数
const { P15, LM } = require('./p15_util.js');
const rand = LM.rng(5);
const cam = Object.assign(P15.lookAt([0.3, 0.4, 2.2], [0, 0, 0]), { W: 32, H: 32, cx: 16, cy: 16, f: 16 / Math.tan(25 * Math.PI / 180) });
const g = P15.gsAlloc(8); g.n = 8;
for (let i = 0; i < g.n; i++) {
    for (let a = 0; a < 3; a++) { g.mean[i * 3 + a] = (rand() - 0.5) * 0.9; g.lscale[i * 3 + a] = Math.log(0.08 + 0.15 * rand()); g.fdc[i * 3 + a] = 2 * LM.randn(rand); }
    for (let a = 0; a < 4; a++) g.quat[i * 4 + a] = LM.randn(rand);
    g.lopac[i] = LM.randn(rand);
}
g.mean[7 * 3] = 2.0;   // 一个在屏幕边缘外（测 J 的截断）
P15.setExact(true);
const wt = Float32Array.from({ length: 32 * 32 * 3 }, () => rand() - 0.5);
function loss() { const c = P15.gsRender(g, cam, {}); let s = 0; for (let i = 0; i < wt.length; i++) s += wt[i] * c.O.rgb[i]; return s; }
const ctx = P15.gsRender(g, cam, {});
const GP = P15.gsRasterBack(ctx.P, ctx.B, 32, 32, ctx.O, wt, {}), GR = P15.gsProjectBack(g, cam, ctx.P, GP, {});
console.log('visible', ctx.B.nVis, 'pairs', ctx.B.pairs, 'clampX', Array.from(ctx.P.clampX.subarray(0, 8)).join(''));
let worst = { rel: 0 };
P15.ATTR.forEach(([k, d]) => {
    let wk = { rel: 0 };
    for (let i = 0; i < g.n * d; i++) {
        const v = g[k][i], e = 1e-3; g[k][i] = v + e; const a = loss(); g[k][i] = v - e; const b = loss(); g[k][i] = v;
        const num = (a - b) / (2 * e), an = GR[k][i], rel = Math.abs(num - an) / (Math.abs(num) + Math.abs(an) + 1e-3);
        if (rel > wk.rel) wk = { rel, i, num, an };
    }
    console.log(`${k.padEnd(7)} worst rel err ${wk.rel.toExponential(2)}  (idx ${wk.i}: num ${wk.num?.toFixed(5)} vs analytic ${wk.an?.toFixed(5)})`);
    if (wk.rel > worst.rel) worst = Object.assign({ k }, wk);
});
console.log('overall worst', worst.k, worst.rel.toExponential(2));
// 聚合误差：‖num − analytic‖ / ‖num‖（每个属性），以及被截断的高斯 #7 单独看
P15.ATTR.forEach(([k, d]) => {
    let se = 0, sn = 0, se7 = 0, sn7 = 0;
    for (let i = 0; i < g.n * d; i++) {
        const v = g[k][i], e = 1e-3; g[k][i] = v + e; const a = loss(); g[k][i] = v - e; const b = loss(); g[k][i] = v;
        const num = (a - b) / (2 * e), an = GR[k][i]; se += (num - an) ** 2; sn += num * num;
        if (Math.floor(i / d) === 7) { se7 += (num - an) ** 2; sn7 += num * num; }
    }
    console.log(`${k.padEnd(7)} ‖Δ‖/‖num‖ = ${(Math.sqrt(se / sn)).toExponential(2)}   (clamped #7: ${sn7 > 0 ? Math.sqrt(se7 / sn7).toExponential(2) : 'n/a'}, ‖num7‖ = ${Math.sqrt(sn7).toExponential(2)})`);
});
