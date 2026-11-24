// p08_prof.js — 第 08 篇：CNN 一步训练的分项耗时（真实 batch、真实稀疏梯度）
const { P08, LM } = require('./p08_proto.js'), { AG } = LM;
const data = P08.makeData(LM.rng(20251123));
const m = P08.makeCNN({ c1: 8, c2: 16 }, LM.rng(5));
const R = P08.makeRun({ model: m, lr: 2e-3, B: 32, cap: 1e9, seed: 1, data, evalSets: [] });
for (let i = 0; i < 300; i++) P08.trainStep(R);   // 先训练一会儿，让梯度的稀疏度接近真实情况
const T = {}; const tic = () => process.hrtime.bigint(); const acc = (k, t) => { T[k] = (T[k] || 0) + Number(process.hrtime.bigint() - t) / 1e6; };
const reps = 200;
// monkey-patch timing wrappers around ops via manual forward
for (let r = 0; r < reps; r++) {
    P08.nextBatch(R); R.opt.zeroGrad();
    let t = tic(); const X = AG.tensor(32, 256, R.Xb);
    const z1 = P08.conv(X, m.conv1.w, m.conv1.b, { C: 1, H: 16, W: 16, K: 3 }); acc('conv1 fwd', t);
    t = tic(); const a1 = AG.relu(z1), p1 = AG.maxPool2(a1, { C: 8, H: 16, W: 16 }); acc('relu+pool1 fwd', t);
    t = tic(); const z2 = P08.conv(p1, m.conv2.w, m.conv2.b, { C: 8, H: 8, W: 8, K: 3 }); acc('conv2 fwd', t);
    t = tic(); const a2 = AG.relu(z2), p2 = AG.maxPool2(a2, { C: 16, H: 8, W: 8 }); acc('relu+pool2 fwd', t);
    t = tic(); const lg = AG.linear(p2, m.fc.w, m.fc.b), loss = AG.crossEntropy(lg, R.yb); acc('fc+ce fwd', t);
    // backward, timing each tape entry
    loss.g().fill(0); loss.grad[0] = 1;
    const names = ['conv1 bwd', 'relu1 bwd', 'pool1 bwd', 'conv2 bwd', 'relu2 bwd', 'pool2 bwd', 'matmul bwd', 'addRow bwd', 'ce bwd'];
    for (let i = AG.tape.length - 1; i >= 0; i--) { t = tic(); AG.tape[i](); acc(names[i] || 'tape' + i, t); }
    AG.tape.length = 0;
    t = tic(); R.opt.step(); acc('adam', t);
    if (r === 0) { let nz = 0; const g = z2.grad; for (let i = 0; i < g.length; i++) nz += g[i] !== 0; let nzx = 0; for (let i = 0; i < p1.data.length; i++) nzx += p1.data[i] !== 0; console.log(`conv2 out-grad nonzero ${(100 * nz / g.length).toFixed(1)}%, conv2 input nonzero ${(100 * nzx / p1.data.length).toFixed(1)}%, tape ${names.length}`); }
}
let tot = 0; for (const k in T) { T[k] /= reps; tot += T[k]; }
for (const k in T) console.log(k.padEnd(16), T[k].toFixed(3), 'ms');
console.log('total'.padEnd(16), tot.toFixed(3), 'ms');
