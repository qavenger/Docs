// p15_train_gs.js — 第 15 篇：离线跑一遍页面默认设置下的 3DGS 训练（与页面同一份 P15 代码、同一个 seed），
// 把最终的高斯（fp16 + base64）、训练曲线、致密化事件写进 tools/p15_gs_<tag>.json，供页面嵌入（Step 05–07、10 用）；并分阶段计时。
// 用法（在 pages/ 下）：node tools/p15_train_gs.js [tag=default] [key=value ...]
const { P15, LM, savePNG, mosaic, path, fs } = require('./p15_util.js');
const args = process.argv.slice(2), tag = args.find((a) => !a.includes('=')) || 'default', o = {};
args.filter((a) => a.includes('=')).forEach((kv) => { const [k, v] = kv.split('='); o[k] = +v; });
const cams = P15.makeCameras();
const gts = cams.train.map((c) => P15.renderRef(c, 1).rgb), gtTest = cams.test.map((c) => P15.renderRef(c, 1).rgb);
const R = P15.gsRunCreate(cams.train, gts, o);
const evalTest = (g) => cams.test.map((c, k) => P15.psnr(P15.gsRender(g, c, {}).O.rgb, gtTest[k]));
const curve = [], testCurve = [];
let acc = 0, na = 0;
while (R.it < R.o.iters) {
    const r = P15.gsRunStep(R); acc += r.psnr; na++;
    if (R.it % 25 === 0) { curve.push([R.it, +(acc / na).toFixed(3), R.g.n]); acc = 0; na = 0; }
    if (R.it % 250 === 0) { const te = evalTest(R.g); testCurve.push([R.it, +(te.reduce((a, b) => a + b, 0) / te.length).toFixed(3)]); console.log(`it ${R.it} test ${testCurve[testCurve.length - 1][1]} N ${R.g.n} ${(R.ms / R.it).toFixed(2)} ms/it`); }
}
// 分阶段计时（最终模型，一个测试视角，重复 50 次）
const g = R.g, cam = cams.test[0], ctx = {}, rep = 50, T = { project: 0, sort: 0, raster: 0, rasterBack: 0, projectBack: 0 };
const now = () => performance.now();
for (let k = 0; k < rep + 5; k++) {
    let t = now(); ctx.P = P15.gsProject(g, cam, ctx.P || {}); const a = now() - t;
    t = now(); ctx.B = P15.gsBin(ctx.P, cam.W, cam.H, ctx.B); const b = now() - t;
    t = now(); ctx.O = P15.gsRaster(ctx.P, ctx.B, cam.W, cam.H, ctx.O || {}); const c = now() - t;
    const dImg = new Float32Array(cam.W * cam.H * 3).fill(1e-4);
    t = now(); const GP = P15.gsRasterBack(ctx.P, ctx.B, cam.W, cam.H, ctx.O, dImg, {}); const d = now() - t;
    t = now(); P15.gsProjectBack(g, cam, ctx.P, GP, {}); const e = now() - t;
    if (k >= 5) { T.project += a / rep; T.sort += b / rep; T.raster += c / rep; T.rasterBack += d / rep; T.projectBack += e / rep; }
}
let touched = 0; for (let i = 0; i < ctx.O.touched.length; i++) touched += ctx.O.touched[i];
console.log('stage ms:', Object.entries(T).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(' · '), `· pairs ${ctx.B.pairs} · visible ${ctx.B.nVis} · avg blended/px ${(touched / ctx.O.touched.length).toFixed(1)}`);
const pack = (gg) => { const params = P15.ATTR.map(([k, d]) => ({ size: gg.n * d, data: gg[k].subarray(0, gg.n * d) })); return LM.gpt.exportWeights({ params }); };
const te = evalTest(g);
const res = { tag, o: R.o, n: g.n, ext: R.ext, msPerIt: +(R.ms / R.it).toFixed(2), stageMs: T, curve, testCurve, testPSNR: te.map((v) => +v.toFixed(2)), events: R.events, b64: pack(g) };
fs.writeFileSync(path.join(__dirname, `p15_gs_${tag}.json`), JSON.stringify(res));
const out = path.join(__dirname, 'shots15', 'proto'); fs.mkdirSync(out, { recursive: true });
const m = mosaic(cams.test.map((c) => P15.gsRender(g, c, {}).O.rgb).concat(gtTest), P15.IMG, P15.IMG, 4); savePNG(path.join(out, `gs_final_${tag}.png`), m.rgb, m.W, m.H, 4);
console.log(`[${tag}] N ${g.n}, test PSNR ${te.map((v) => v.toFixed(2)).join(' / ')}, ${res.msPerIt} ms/it, json ${(JSON.stringify(res).length / 1024).toFixed(0)} KB`);
