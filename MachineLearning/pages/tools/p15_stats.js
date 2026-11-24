// p15_stats.js — 嵌入的 3DGS 结果的统计（正文引用的数字）
const { P15 } = require('./p15_util.js');
const fs = require('fs'), path = require('path');
const src = fs.readFileSync(path.join(__dirname, '../15_nerf_gaussian_splatting.lab.js'), 'utf8');
const D = JSON.parse(src.slice(src.indexOf('const P15_DATA = ') + 17, src.indexOf(';\n// </P15_DATA>')));
const g = P15.gsFromB64(D.gs.b64, D.gs.n), n = g.n, an = [], sm = [], op = [], smin = [];
for (let i = 0; i < n; i++) { const s = [0, 1, 2].map((a) => Math.exp(g.lscale[i * 3 + a])).sort((a, b) => a - b); an.push(s[2] / s[0]); sm.push(s[2]); smin.push(s[0]); op.push(1 / (1 + Math.exp(-g.lopac[i]))); }
const med = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1], q = (a, p) => a.slice().sort((x, y) => x - y)[Math.floor(a.length * p)];
console.log(`N ${n}; anisotropy max/min median ${med(an).toFixed(2)} p90 ${q(an, 0.9).toFixed(1)}; max-axis median ${med(sm).toFixed(4)}; min-axis median ${med(smin).toFixed(4)}; opacity median ${med(op).toFixed(3)}, >0.25: ${op.filter((o) => o > 0.25).length}, >0.9: ${op.filter((o) => o > 0.9).length}`);
const cams = P15.makeCameras(), gt = cams.test.map((c) => P15.renderRef(c, 1).rgb);
console.log('fp16-decoded 3DGS test PSNR', cams.test.map((c, k) => P15.psnr(P15.gsRender(g, c, {}).O.rgb, gt[k]).toFixed(2)).join(' / '));
const net = P15.nerfFromB64(D.nerf.cfg, D.nerf.b64);
console.log('fp16-decoded NeRF test PSNR', cams.test.map((c, k) => P15.psnr(P15.nerfRender(net, c).rgb, gt[k]).toFixed(2)).join(' / '));
const cam = P15.camAt(30, 25), ctx = P15.gsRender(g, cam, {}); let tot = 0, term = 0; for (let i = 0; i < ctx.O.touched.length; i++) { tot += ctx.O.touched[i]; if (ctx.O.Tf[i] < 1e-4 * 1.0001 || ctx.O.Tf[i] < 0.01) term++; }
console.log(`novel view: visible ${ctx.B.nVis}, pairs ${ctx.B.pairs}, avg blended/px ${(tot / ctx.O.touched.length).toFixed(1)}, T<0.01 px ${term}`);
console.log('edits:', ['beak', 'head'].map((k) => k + ' ' + P15.gsEdit(g, k).edited).join(', '));
