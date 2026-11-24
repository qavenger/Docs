// p15_g2.js — 2D 高斯拟合：有限差分校验 g2Back + 不同个数的收敛 / 耗时
const { P15, LM, savePNG, mosaic, path } = require('./p15_util.js');
const cams = P15.makeCameras(), img = P15.renderRef(cams.test[0], 1).rgb, W = P15.IMG, H = P15.IMG;
{   // 梯度校验
    P15.setExact(true);
    const R = P15.g2RunCreate(img, W, H, 12, 3), rand = LM.rng(4);
    for (let i = 0; i < 12; i++) { R.G.f[i * 3] = LM.randn(rand); R.G.f[i * 3 + 1] = LM.randn(rand); R.G.ls[i * 2] = Math.log(3 + 4 * rand()); R.G.ls[i * 2 + 1] = Math.log(2 + 3 * rand()); R.G.lo[i] = LM.randn(rand); }
    const wt = Float32Array.from({ length: W * H * 3 }, () => rand() - 0.5);
    const loss = () => { const O = P15.g2Render(R); let s = 0; for (let i = 0; i < wt.length; i++) s += wt[i] * O.rgb[i]; return s; };
    const O = P15.g2Render(R); P15.gsRasterBack(R.P, R.B, W, H, O, wt, R.GP); P15.g2Back(R.G, R.P, R.GP, R.GR);
    const GR = {}; P15.G2_KEYS.forEach(([k]) => { GR[k] = R.GR[k].slice(); });
    P15.G2_KEYS.forEach(([k, d]) => { let se = 0, sn = 0; for (let i = 0; i < 12 * d; i++) { const v = R.G[k][i], e = 1e-3; R.G[k][i] = v + e; const a = loss(); R.G[k][i] = v - e; const b = loss(); R.G[k][i] = v; const num = (a - b) / (2 * e); se += (num - GR[k][i]) ** 2; sn += num * num; } console.log(`g2 ${k.padEnd(3)} ‖Δ‖/‖num‖ = ${Math.sqrt(se / sn).toExponential(2)}`); });
    P15.setExact(false);
}
const outs = [];
for (const n of [64, 128, 256, 512]) {
    const R = P15.g2RunCreate(img, W, H, n, 5), log = [];
    for (let k = 1; k <= 1500; k++) { const p = P15.g2Step(R); if (k === 100 || k === 300 || k === 1000 || k === 1500) log.push(`${k}: ${p.toFixed(1)}`); }
    console.log(`n=${n}: PSNR ${log.join(' · ')} dB · ${(R.ms / R.it).toFixed(2)} ms/step`);
    outs.push(P15.g2Render(R).rgb.slice());
}
const m = mosaic(outs.concat([img]), W, H, 5); savePNG(path.join(__dirname, 'shots15', 'proto', 'g2.png'), m.rgb, m.W, m.H, 4);
