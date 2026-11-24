// p15_scene.js — 渲染第 15 篇场景的全部训练 / 测试视角（参考体渲染），量耗时并输出 PNG 供检查
const { P15, savePNG, mosaic, path } = require('./p15_util.js');
const cams = P15.makeCameras();
const t0 = performance.now();
const tr = cams.train.map((c) => P15.renderRef(c, 1));
const t1 = performance.now();
const te = cams.test.map((c) => P15.renderRef(c, 1));
console.log(`train ${cams.train.length} views: ${(t1 - t0).toFixed(0)} ms (${((t1 - t0) / cams.train.length).toFixed(1)} ms/view); test ${cams.test.length}: ${(performance.now() - t1).toFixed(0)} ms`);
const out = path.join(__dirname, 'shots15', 'proto'); require('fs').mkdirSync(out, { recursive: true });
const m = mosaic(tr.map((r) => r.rgb), P15.IMG, P15.IMG, 8); savePNG(path.join(out, 'train_views.png'), m.rgb, m.W, m.H, 3);
const m2 = mosaic(te.map((r) => r.rgb), P15.IMG, P15.IMG, 4); savePNG(path.join(out, 'test_views.png'), m2.rgb, m2.W, m2.H, 5);
let accMean = 0; tr.forEach((r) => { for (let i = 0; i < r.acc.length; i++) accMean += r.acc[i]; }); accMean /= tr.length * P15.IMG * P15.IMG;
console.log('mean opacity (foreground fraction):', accMean.toFixed(3));
for (const s of [0.05, 0.2]) { const t2 = performance.now(); const r = P15.renderRef(cams.test[0], s); console.log(`sigScale ${s}: ${(performance.now() - t2).toFixed(0)} ms`); savePNG(path.join(out, `translucent_${s}.png`), r.rgb, P15.IMG, P15.IMG, 5); }
