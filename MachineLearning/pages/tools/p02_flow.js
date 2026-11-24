// p02_flow.js — 第 10 步原型：深层 MLP 各层激活 / 权重梯度的标准差（不同初始化 × 激活）
const { P02, LM } = require('./p02_proto.js');
const rand = LM.rng(20251123);
const tr = P02.makeSet(400, rand);
const depth = +process.argv[2] || 8, B = 64, Wd = 64;
const std = (a) => { let m = 0; for (let i = 0; i < a.length; i++) m += a[i]; m /= a.length; let v = 0; for (let i = 0; i < a.length; i++) v += (a[i] - m) ** 2; return Math.sqrt(v / a.length); };
const X = tr.X.subarray(0, B * 256), y = tr.y.subarray(0, B);
for (const act of ['sigmoid', 'tanh', 'relu', 'gelu']) for (const init of ['small', 'xavier', 'he', 'large']) {
    const sizes = [256]; for (let l = 0; l < depth; l++) sizes.push(Wd); sizes.push(10);
    const net = P02.makeNet(sizes, act, init, LM.rng(5));
    const t0 = process.hrtime.bigint();
    const cache = P02.forward(net, X, B, {}), g = P02.backward(net, y, cache, {});
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    const aS = [], gS = [];
    for (let l = 1; l <= depth; l++) aS.push(std(cache.A[l]));
    for (let l = 0; l <= depth; l++) gS.push(std(g.dW[l]));
    console.log(`${act.padEnd(7)} ${init.padEnd(6)} ${ms.toFixed(1)}ms  act std: ${aS.map((v) => v.toExponential(1)).join(' ')}\n                 dW std: ${gS.map((v) => v.toExponential(1)).join(' ')}  loss ${P02.lossOf(net, cache, y).toFixed(2)}`);
}
