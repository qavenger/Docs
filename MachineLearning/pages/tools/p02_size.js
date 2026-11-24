// p02_size.js — 训练集大小对 softmax / MLP 测试准确率的影响（调数据规模用）
const { P02, LM } = require('./p02_proto.js');
const rand = LM.rng(20251123);
const NTR = +process.argv[2] || 2000, NTE = 500;
const tr = P02.makeSet(NTR, rand), te = P02.makeSet(NTE, rand);
const all = (n) => Int32Array.from({ length: n }, (_, i) => i);
function run(label, extra, steps) {
    const cfg = Object.assign({ sizes: [256, 64, 10], act: 'relu', init: 'he', seed: 7, opt: { kind: 'sgd', lr: 0.1 }, X: tr.X, y: tr.y, idx: all(tr.n), B: 32, cap: steps,
        evalSets: [{ name: 'train', X: tr.X, y: tr.y, idx: all(Math.min(500, tr.n)) }, { name: 'test', X: te.X, y: te.y, idx: all(te.n), conf: true }] }, extra);
    const R = P02.makeRun(cfg); P02.evalNow(R);
    while (R.step < R.cap && !R.diverged) P02.tick(R, 2, 100);
    const H = R.hist, n = H.step.length, best = Math.max(...H.testA);
    console.log(`N=${NTR} ${label.padEnd(26)} final tr ${(100 * H.trainA[n - 1]).toFixed(1)} te ${(100 * H.testA[n - 1]).toFixed(1)} best te ${(100 * best).toFixed(1)} testL ${H.testL[n - 1].toFixed(3)}`);
}
run('softmax lr0.3', { sizes: [256, 10], init: 'zero', opt: { kind: 'sgd', lr: 0.3 } }, 3000);
run('mlp64 lr0.1', {}, 4000);
run('mlp128 lr0.1', { sizes: [256, 128, 10] }, 4000);
