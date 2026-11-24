// p15_nerf_live.js — 页面里实时训练 NeRF 的设置（小 batch）在 Node 里的效果：node tools/p15_nerf_live.js B lr steps
const { P15, LM } = require('./p15_util.js');
const B = +(process.argv[2] || 8), lr = +(process.argv[3] || 1e-3), steps = +(process.argv[4] || 6000);
const cams = P15.makeCameras(), gt = cams.train.map((c) => P15.renderRef(c, 1).rgb), gtT = P15.renderRef(cams.test[0], 1).rgb;
const data = P15.buildRays(cams.train, gt), net = P15.nerfCreate({ L: 6, W: 64, depth: 2, S: 32 }, 7), opt = new LM.AdamW(net.params, { lr, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0 }), rand = LM.rng(11);
let t = 0; const t0 = performance.now();
for (let it = 1; it <= steps; it++) { const ts = performance.now(); P15.nerfStep(net, opt, data, rand, B, lr); t += performance.now() - ts; if (it % 1000 === 0) console.log(`B=${B} lr=${lr} it ${it}: test0 PSNR ${P15.psnr(P15.nerfRender(net, cams.test[0]).rgb, gtT).toFixed(2)} dB · ${(t / it).toFixed(2)} ms/step`); }
