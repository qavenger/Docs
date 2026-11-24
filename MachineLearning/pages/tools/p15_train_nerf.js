// p15_train_nerf.js — 第 15 篇：离线训练小 NeRF（与页面同一份 P15 代码），结果写进 tools/p15_nerf_<tag>.json，页面嵌入。
// 用法（在 pages/ 下）：node tools/p15_train_nerf.js <tag> <L> [steps=8000] [B=128]
//   L = 位置编码频带数（0 = 不用位置编码，消融用）。网络 = γ(x) → 64 → 64 → 4，每条光线 32 个分层样本，Adam，lr 3e-3 指数衰减到 3e-4。
const { P15, LM, savePNG, mosaic, path, fs } = require('./p15_util.js');
const tag = process.argv[2] || 'pe', L = +(process.argv[3] ?? 6), steps = +(process.argv[4] || 8000), B = +(process.argv[5] || 128);
const cfg = { L, W: 64, depth: 2, S: 32 }, LR0 = 3e-3, LR1 = 3e-4, SEED_NET = 7, SEED_RAYS = 11;
const cams = P15.makeCameras();
const gt = cams.train.map((c) => P15.renderRef(c, 1).rgb), gtTest = cams.test.map((c) => P15.renderRef(c, 1).rgb);
const data = P15.buildRays(cams.train, gt);
const net = P15.nerfCreate(cfg, SEED_NET), opt = new LM.AdamW(net.params, { lr: LR0, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0 });
const rand = LM.rng(SEED_RAYS);
const FILM = [0, 50, 200, 500, 1000, 2000, 4000, 8000].filter((s) => s <= steps);
const toU8 = (rgb) => Buffer.from(Uint8Array.from(rgb, (v) => Math.max(0, Math.min(255, Math.round(v * 255))))).toString('base64');
const evalTest = () => cams.test.map((c, k) => P15.psnr(P15.nerfRender(net, c).rgb, gtTest[k]));
const curve = [], film = [], testCurve = [];
let lossAcc = 0, nAcc = 0, t0 = Date.now(), trainMs = 0;
console.log(`[${tag}] cfg ${JSON.stringify(cfg)} B ${B} steps ${steps} params ${net.params.reduce((s, p) => s + p.size, 0)}`);
for (let it = 0; it <= steps; it++) {
    if (FILM.includes(it)) film.push({ step: it, img: toU8(P15.nerfRender(net, cams.test[0]).rgb) });
    if (it % 500 === 0 || it === steps) { const p = evalTest(); testCurve.push([it, +(p.reduce((a, b) => a + b, 0) / p.length).toFixed(3)]); console.log(`[${tag}] it ${it} test PSNR ${testCurve[testCurve.length - 1][1]} (${p.map((v) => v.toFixed(1)).join(' / ')}) · ${(trainMs / Math.max(1, it)).toFixed(1)} ms/step · ${((Date.now() - t0) / 1000).toFixed(0)} s`); }
    if (it === steps) break;
    const lr = LR0 * Math.pow(LR1 / LR0, it / steps), ts = performance.now();
    lossAcc += P15.nerfStep(net, opt, data, rand, B, lr); nAcc++;
    trainMs += performance.now() - ts;
    if ((it + 1) % 50 === 0) { curve.push([it + 1, +(-10 * Math.log10(lossAcc / nAcc)).toFixed(3)]); lossAcc = 0; nAcc = 0; }
}
const res = { tag, cfg, steps, B, lr: [LR0, LR1], seeds: [SEED_NET, SEED_RAYS], msPerStep: +(trainMs / steps).toFixed(1), curve, testCurve, film, b64: LM.gpt.exportWeights(net) };
fs.writeFileSync(path.join(__dirname, `p15_nerf_${tag}.json`), JSON.stringify(res));
const out = path.join(__dirname, 'shots15', 'proto'); fs.mkdirSync(out, { recursive: true });
const m = mosaic(cams.test.map((c) => P15.nerfRender(net, c).rgb).concat(gtTest), P15.IMG, P15.IMG, 4); savePNG(path.join(out, `nerf_final_${tag}.png`), m.rgb, m.W, m.H, 4);
console.log(`[${tag}] done: ${res.msPerStep} ms/step, final test PSNR ${testCurve[testCurve.length - 1][1]}`);
