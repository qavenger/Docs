// p15_nerf.js — 第 15 篇 NeRF 原型：有限差分校验体渲染算子；训练 + 计时 + 输出渲染。用法：node tools/p15_nerf.js check | train [steps] [L] [W] [depth] [B] [S] [lr]
const { P15, LM, savePNG, mosaic, path, fs } = require('./p15_util.js');
const { AG } = LM;
const what = process.argv[2] || 'check';
if (what === 'check') {
    const rand = LM.rng(3), R = 3, S = 7, raw = AG.param('raw', R * S, 4, (a) => { for (let i = 0; i < a.length; i++) a[i] = 2 * LM.randn(rand); });
    const delta = new Float32Array(R * S).map(() => 0.05 + 0.3 * rand()), tgt = AG.tensor(R, 3, Float32Array.from({ length: R * 3 }, () => rand()));
    const f = () => AG.noGrad(() => AG.mse(P15.volRender(raw, R, S, delta), tgt).item());
    raw.grad = null; const L = AG.mse(P15.volRender(raw, R, S, delta), tgt); AG.backward(L);
    let maxRel = 0;
    for (let i = 0; i < raw.size; i++) {
        const v = raw.data[i], e = 1e-2; raw.data[i] = v + e; const a = f(); raw.data[i] = v - e; const b = f(); raw.data[i] = v;
        const num = (a - b) / (2 * e), an = raw.grad[i]; maxRel = Math.max(maxRel, Math.abs(num - an) / (Math.abs(num) + Math.abs(an) + 1e-6));
    }
    console.log('volRender grad check: max rel err', maxRel.toExponential(2));
    process.exit(0);
}
const steps = +(process.argv[3] || 1000), cfg = { L: +(process.argv[4] ?? 6), W: +(process.argv[5] || 64), depth: +(process.argv[6] || 2), S: +(process.argv[8] || 32) }, B = +(process.argv[7] || 128), lr0 = +(process.argv[9] || 2e-3);
const cams = P15.makeCameras();
const gt = cams.train.map((c) => P15.renderRef(c, 1).rgb), gtTest = cams.test.map((c) => P15.renderRef(c, 1).rgb);
const data = P15.buildRays(cams.train, gt);
const net = P15.nerfCreate(cfg, 7), opt = new LM.AdamW(net.params, { lr: lr0, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0 });
const rand = LM.rng(11);
console.log('cfg', cfg, 'B', B, 'params', net.params.reduce((s, p) => s + p.size, 0));
let t0 = performance.now(), lossAcc = 0, nAcc = 0;
const evalTest = () => { let p = 0; cams.test.forEach((c, k) => { p += P15.psnr(P15.nerfRender(net, c).rgb, gtTest[k]); }); return p / cams.test.length; };
for (let it = 0; it < steps; it++) {
    const lr = lr0 * Math.pow(0.1, it / steps);   // 指数衰减到 1/10
    lossAcc += P15.nerfStep(net, opt, data, rand, B, lr); nAcc++;
    if ((it + 1) % Math.max(1, steps / 10) === 0) {
        const ms = (performance.now() - t0) / nAcc, te0 = performance.now(), tp = evalTest(), tms = (performance.now() - te0) / cams.test.length;
        console.log(`it ${it + 1}: train PSNR ${(-10 * Math.log10(lossAcc / nAcc)).toFixed(2)} dB, test PSNR ${tp.toFixed(2)} dB, ${ms.toFixed(1)} ms/step, render ${tms.toFixed(0)} ms/view`);
        lossAcc = 0; nAcc = 0; t0 = performance.now();
    }
}
const out = path.join(__dirname, 'shots15', 'proto'); fs.mkdirSync(out, { recursive: true });
const renders = cams.test.map((c) => P15.nerfRender(net, c).rgb);
const m = mosaic(renders.concat(gtTest), P15.IMG, P15.IMG, 4); savePNG(path.join(out, `nerf_L${cfg.L}_W${cfg.W}_d${cfg.depth}_${steps}.png`), m.rgb, m.W, m.H, 4);
