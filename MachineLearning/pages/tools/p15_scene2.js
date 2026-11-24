const { P15 } = require('./p15_util.js');
const cams = P15.makeCameras();
for (const fs of [P15.BETA / 4, P15.BETA / 2, P15.BETA, P15.BETA * 1.5]) {
    const t0 = performance.now(); const imgs = cams.train.slice(0, 8).map((c) => P15.renderRef(c, 1, fs)); const ms = (performance.now() - t0) / 8;
    const ref = cams.train.slice(0, 8).map((c) => P15.renderRef(c, 1, P15.BETA / 8));
    let p = 0; imgs.forEach((im, k) => { p += P15.psnr(im.rgb, ref[k].rgb); }); p /= 8;
    console.log(`fine = ${(fs / P15.BETA).toFixed(2)}β: ${ms.toFixed(1)} ms/view, PSNR vs β/8 = ${p.toFixed(1)} dB`);
}
