// 截断分支：中心在 1.3·tan(fov/2) 之外、但足迹还落在屏幕里的大高斯
const { P15, LM } = require('./p15_util.js');
P15.setExact(true);
const cam = Object.assign(P15.lookAt([0, 0, 2.2], [0, 0, 0]), { W: 32, H: 32, cx: 16, cy: 16, f: 16 / Math.tan(25 * Math.PI / 180) });
const g = P15.gsAlloc(2); g.n = 2;
const set = (i, m, s, q, o, f) => { g.mean.set(m, i * 3); g.lscale.set(s.map(Math.log), i * 3); g.quat.set(q, i * 4); g.lopac[i] = o; g.fdc.set(f, i * 3); };
set(0, [1.55, 0.2, 0], [0.5, 0.3, 0.4], [0.9, 0.2, 0.3, 0.1], 1.5, [1, -0.5, 0.3]);
set(1, [-0.2, -1.6, 0.1], [0.3, 0.55, 0.3], [0.7, -0.3, 0.2, 0.5], 0.8, [-1, 0.5, 1.2]);
const rand = LM.rng(9), wt = Float32Array.from({ length: 32 * 32 * 3 }, () => rand() - 0.5);
const loss = () => { const c = P15.gsRender(g, cam, {}); let s = 0; for (let i = 0; i < wt.length; i++) s += wt[i] * c.O.rgb[i]; return s; };
const ctx = P15.gsRender(g, cam, {});
console.log('radius', Array.from(ctx.P.radius.subarray(0, 2)), 'clampX', Array.from(ctx.P.clampX.subarray(0, 2)), 'clampY', Array.from(ctx.P.clampY.subarray(0, 2)), 'tx/tz', (ctx.P.t[0] / ctx.P.t[2]).toFixed(3), 'ty/tz', (ctx.P.t[4] / ctx.P.t[5]).toFixed(3));
const GR = P15.gsProjectBack(g, cam, ctx.P, P15.gsRasterBack(ctx.P, ctx.B, 32, 32, ctx.O, wt, {}), {});
P15.ATTR.forEach(([k, d]) => {
    let se = 0, sn = 0;
    for (let i = 0; i < g.n * d; i++) { const v = g[k][i], e = 1e-3; g[k][i] = v + e; const a = loss(); g[k][i] = v - e; const b = loss(); g[k][i] = v; const num = (a - b) / (2 * e); se += (num - GR[k][i]) ** 2; sn += num * num; }
    console.log(`${k.padEnd(7)} ‖Δ‖/‖num‖ = ${Math.sqrt(se / sn).toExponential(2)}  ‖num‖ = ${Math.sqrt(sn).toExponential(2)}`);
});
