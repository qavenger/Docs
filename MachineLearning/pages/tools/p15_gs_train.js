// p15_gs_train.js — 3DGS 多视角拟合实验：node tools/p15_gs_train.js [key=value ...]（覆盖 GS_DEFAULT，如 tau=0.001 iters=3000 lrScale=2）
const { P15, LM, savePNG, mosaic, path, fs } = require('./p15_util.js');
const o = {}; process.argv.slice(2).forEach((kv) => { const [k, v] = kv.split('='); o[k] = isNaN(+v) ? v : +v; });
const cams = P15.makeCameras();
const gts = cams.train.map((c) => P15.renderRef(c, 1).rgb), gtTest = cams.test.map((c) => P15.renderRef(c, 1).rgb);
const R = P15.gsRunCreate(cams.train, gts, o);
console.log('opts', JSON.stringify(R.o), 'extent', R.ext.toFixed(3));
const evalTest = () => cams.test.map((c, k) => P15.psnr(P15.gsRender(R.g, c, {}).O.rgb, gtTest[k]));
let acc = 0, na = 0; const t0 = Date.now();
for (let it = 0; it < R.o.iters; it++) {
    const r = P15.gsRunStep(R); acc += r.psnr; na++;
    if (R.it % 100 === 99 && R.it <= R.o.until + 100) {   // 梯度统计（致密化前一刻的平均 NDC 梯度分布）
        const gr = []; for (let i = 0; i < R.g.n; i++) if (R.denom[i] > 0) gr.push(R.accum[i] / R.denom[i]);
        gr.sort((a, b) => a - b);
        if (R.it % 500 === 499 || R.it === 399) console.log(`   grad stats @${R.it}: p50 ${(gr[gr.length >> 1] || 0).toExponential(2)} p90 ${(gr[Math.floor(gr.length * 0.9)] || 0).toExponential(2)} p99 ${(gr[Math.floor(gr.length * 0.99)] || 0).toExponential(2)}`);
    }
    if (R.it % 500 === 0 || R.it === R.o.iters) {
        const te = evalTest(), ev = R.events.filter((e) => e.it > R.it - 500 && !e.reset);
        console.log(`it ${R.it}: train ${(acc / na).toFixed(2)} dB, test ${(te.reduce((a, b) => a + b, 0) / te.length).toFixed(2)} dB, N=${R.g.n}, ${(R.ms / R.it).toFixed(2)} ms/it, events: +${ev.reduce((s, e) => s + e.cloned, 0)} clone +${ev.reduce((s, e) => s + e.split, 0)} split −${ev.reduce((s, e) => s + e.prunedO + e.prunedS, 0)} prune`);
        acc = 0; na = 0;
    }
}
console.log(`total ${((Date.now() - t0) / 1000).toFixed(1)} s`);
const out = path.join(__dirname, 'shots15', 'proto'); fs.mkdirSync(out, { recursive: true });
const m = mosaic(cams.test.map((c) => P15.gsRender(R.g, c, {}).O.rgb).concat(gtTest), P15.IMG, P15.IMG, 4); savePNG(path.join(out, `gs_${o.tag || 'run'}.png`), m.rgb, m.W, m.H, 4);
if (o.save) fs.writeFileSync(path.join(__dirname, `p15_gs_${o.tag}.json`), JSON.stringify({ o: R.o, n: R.g.n, events: R.events }));
