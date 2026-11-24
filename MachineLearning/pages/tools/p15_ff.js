// p15_ff.js — Fourier features 实验：不同 L、全部像素 / 棋盘格一半像素训练，测 PSNR 与耗时
const { P15, LM, savePNG, mosaic, path, fs } = require('./p15_util.js');
const steps = +(process.argv[2] || 2000), B = +(process.argv[3] || 256);
const cams = P15.makeCameras(), img = P15.renderRef(cams.test[0], 1).rgb, W = P15.IMG, H = P15.IMG;
const outs = [];
for (const sub of [false, true]) for (const L of [0, 2, 4, 6, 8, 10]) {
    const R = P15.ffRunCreate(img, W, H, L, { sub, B });
    const t0 = performance.now(); for (let k = 0; k < steps; k++) P15.ffStep(R); const ms = (performance.now() - t0) / steps;
    const t1 = performance.now(); while (!P15.ffRenderRows(R, 6)); const rms = (performance.now() - t1) / (H / 6);
    const e = P15.ffEval(R); outs.push(R.pred.slice());
    console.log(`sub=${sub} L=${L}: train ${e.train.toFixed(2)} dB, test ${isNaN(e.test) ? '—' : e.test.toFixed(2)} dB · ${ms.toFixed(2)} ms/step · render 6 rows ${rms.toFixed(2)} ms`);
}
const m = mosaic(outs.concat([img]), W, H, 6); savePNG(path.join(__dirname, 'shots15', 'proto', `ff_${steps}.png`), m.rgb, m.W, m.H, 3);
