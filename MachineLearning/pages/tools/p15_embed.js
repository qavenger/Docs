// p15_embed.js — 把离线训练的结果（tools/p15_nerf_*.json、tools/p15_gs_*.json）压成一行 `const P15_DATA = {...};`，
// 写进 15_nerf_gaussian_splatting.lab.js 里 `// <P15_DATA>` 与 `// </P15_DATA>` 两行标记之间（重复运行会替换）。
// 用法（在 pages/ 下）：node tools/p15_embed.js
const fs = require('fs'), path = require('path');
const J = (f) => JSON.parse(fs.readFileSync(path.join(__dirname, f), 'utf8'));
const pe = J('p15_nerf_pe.json'), nope = J('p15_nerf_nope.json'), gs = J('p15_gs_default.json'), f300 = J('p15_gs_fixed300.json'), f1300 = J('p15_gs_fixed1300.json');
// 默认新视角（az = 30°, el = 25°）下两个离线 NeRF 的渲染（用解码后的 fp16 权重、页面同一份代码算）：页面打开时直接显示，换视角后再逐行重渲染
const { P15 } = require('./p15_util.js');
const NOVEL = { az: 30, el: 25 }, camN = P15.camAt(NOVEL.az, NOVEL.el);
const u8 = (rgb) => Buffer.from(Uint8Array.from(rgb, (v) => Math.max(0, Math.min(255, Math.round(v * 255))))).toString('base64');
const viewPE = P15.nerfRender(P15.nerfFromB64(pe.cfg, pe.b64), camN), viewNo = P15.nerfRender(P15.nerfFromB64(nope.cfg, nope.b64), camN);
const thin = (c, k) => c.filter((_, i) => i % k === k - 1 || i === c.length - 1);
const FILM = [0, 200, 500, 1000, 2000, 8000];
const data = {
    nerf: {
        cfg: pe.cfg, steps: pe.steps, B: pe.B, lr: pe.lr, msPerStep: pe.msPerStep, curve: thin(pe.curve, 2), testCurve: pe.testCurve,
        film: pe.film.filter((f) => FILM.includes(f.step)), b64: pe.b64,
    },
    nope: { cfg: nope.cfg, steps: nope.steps, msPerStep: nope.msPerStep, curve: thin(nope.curve, 2), testCurve: nope.testCurve, b64: nope.b64 },
    gs: { o: gs.o, n: gs.n, ext: gs.ext, msPerIt: gs.msPerIt, stageMs: gs.stageMs, curve: gs.curve, testCurve: gs.testCurve, testPSNR: gs.testPSNR, events: gs.events, b64: gs.b64 },
    nerfView: { az: NOVEL.az, el: NOVEL.el, pe: u8(viewPE.rgb), peAcc: u8(viewPE.acc), peDepth: Array.from(viewPE.depth, (v) => +v.toFixed(3)), nope: u8(viewNo.rgb) },
    fixed300: { n: f300.n, msPerIt: f300.msPerIt, curve: f300.curve, testCurve: f300.testCurve, testPSNR: f300.testPSNR },
    fixed1300: { n: f1300.n, msPerIt: f1300.msPerIt, curve: f1300.curve, testCurve: f1300.testCurve, testPSNR: f1300.testPSNR },
};
const file = path.join(__dirname, '..', '15_nerf_gaussian_splatting.lab.js');
const src = fs.readFileSync(file, 'utf8'), a = src.indexOf('// <P15_DATA>'), b = src.indexOf('// </P15_DATA>');
if (a < 0 || b < 0) throw new Error('markers not found');
const line = `// <P15_DATA>  由 tools/p15_embed.js 生成（离线训练脚本：tools/p15_train_nerf.js、tools/p15_train_gs.js），不要手改\nconst P15_DATA = ${JSON.stringify(data)};\n`;
fs.writeFileSync(file, src.slice(0, a) + line + src.slice(b));
console.log(`embedded ${(line.length / 1024).toFixed(1)} KB`);
