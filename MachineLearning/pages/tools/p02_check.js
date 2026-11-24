// p02_check.js — 三方梯度校验、XOR 小网络、二维优化器轨迹的 Node 检查
const { P02, LM } = require('./p02_proto.js');
const rand = LM.rng(20251123);
const tr = P02.makeSet(64, rand);
for (const act of ['relu', 'tanh', 'sigmoid', 'gelu']) for (const H of [64]) {
    const net = P02.makeNet([256, H, 10], act, 'he', LM.rng(3));
    const t0 = Date.now();
    const gc = P02.gradCheck(net, tr.X, tr.y, 16);
    console.log(`${act} H${H} ${Date.now() - t0}ms loss64 ${gc.loss64.toFixed(6)} lossAG ${gc.lossAG.toFixed(6)} | ` + gc.per.map((p) => `${p.name}: num ${p.relNum.toExponential(1)} ag ${p.relAg.toExponential(1)}`).join('  '));
}
// a trained-ish net: few steps of training first
{
    const net = P02.makeNet([256, 64, 10], 'relu', 'he', LM.rng(3)), opt = P02.makeOpt('sgd', net, { lr: 0.1 }), c = {}, g = {};
    for (let s = 0; s < 200; s++) { const i0 = (s * 16) % 48; P02.forward(net, tr.X.subarray(i0 * 256, (i0 + 16) * 256), 16, c); P02.backward(net, tr.y.subarray(i0, i0 + 16), c, g); P02.optStep(opt, net, g); }
    const gc = P02.gradCheck(net, tr.X, tr.y, 16);
    console.log(`trained relu loss64 ${gc.loss64.toFixed(6)} | ` + gc.per.map((p) => `${p.name}: num ${p.relNum.toExponential(1)} ag ${p.relAg.toExponential(1)}`).join('  '));
    gc.rows.slice(0, 6).forEach((r) => console.log(`  ${r.name}[${r.i}] hand ${r.hand.toExponential(6)} num ${r.num.toExponential(6)} ag ${r.ag.toExponential(6)}  rel ${r.relNum.toExponential(1)} / ${r.relAg.toExponential(1)}`));
}
// XOR
{
    const d = P02.xorData(LM.rng(11), 160);
    for (const act of ['linear', 'sigmoid', 'tanh', 'relu', 'gelu']) { const t0 = Date.now(); const r = P02.trainSmall([2, 16, 2], act, d, 300, 4); console.log(`xor ${act.padEnd(7)} acc ${(100 * r.acc).toFixed(1)} loss ${r.loss[299].toFixed(3)} ${Date.now() - t0}ms`); }
}
// opt2d
for (const k of ['sgd', 'momentum', 'adam']) {
    const r = P02.runOpt2d(k, -4.5, 1.6, 150);
    const at = (t) => (r.fs[t] ?? NaN).toExponential(2);
    console.log(`2d ${k.padEnd(8)} f@0 ${at(0)} @20 ${at(20)} @50 ${at(50)} @100 ${at(100)} @150 ${at(150)} end (${r.pts[r.pts.length - 1].map((v) => v.toFixed(2)).join(', ')})`);
}
