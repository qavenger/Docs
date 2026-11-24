// p14_gradcheck.js — 融合 GRU 算子 P14.gruCell 的校验：(1) 前向与 15 个小算子拼成的 gruCellAG 逐项相等；
// (2) 对输入 X、上一步隐状态 h、全部权重做有限差分梯度校验（float32，中心差分）。用法：node tools/p14_gradcheck.js
const { LM, P14 } = require('./p14_load.js');
const { AG } = LM;
const m = P14.makeDyn(5, { Z: 4, H: 6 }), B = 3, In = 4 + 5, r = LM.rng(9);
const X = AG.param('X', B, In, (a) => { for (let i = 0; i < a.length; i++) a[i] = LM.randn(r); });
const Hp = AG.param('h', B, 6, (a) => { for (let i = 0; i < a.length; i++) a[i] = 0.8 * Math.tanh(LM.randn(r)); });
m.bx.data.forEach((_, i) => { m.bx.data[i] = 0.3 * LM.randn(r); m.bh.data[i] = 0.3 * LM.randn(r); });
const Wout = AG.tensor(B, 6); for (let i = 0; i < Wout.data.length; i++) Wout.data[i] = LM.randn(r);
const lossOf = (cell) => AG.sum(AG.mul(cell(m, X, Hp), Wout));
let maxFwd = 0;
const a = AG.noGrad(() => P14.gruCell(m, X, Hp)), b = AG.noGrad(() => P14.gruCellAG(m, X, Hp));
for (let i = 0; i < a.data.length; i++) maxFwd = Math.max(maxFwd, Math.abs(a.data[i] - b.data[i]));
console.log('forward max |fused − composite| =', maxFwd.toExponential(2));
const tens = [X, Hp, m.Wx, m.bx, m.Wh, m.bh];
for (const cell of [P14.gruCell, P14.gruCellAG]) {
    tens.forEach((t) => { t.grad = null; });
    AG.backward(lossOf(cell));
    const ana = tens.map((t) => Float32Array.from(t.grad));
    let worst = 0;
    tens.forEach((t, k) => {
        for (let i = 0; i < t.data.length; i++) {
            const v = t.data[i], e = 1e-2;
            t.data[i] = v + e; const lp = AG.noGrad(() => lossOf(cell)).item();
            t.data[i] = v - e; const lm = AG.noGrad(() => lossOf(cell)).item();
            t.data[i] = v;
            const num = (lp - lm) / (2 * e), rel = Math.abs(num - ana[k][i]) / Math.max(1e-2, Math.abs(num) + Math.abs(ana[k][i]));
            worst = Math.max(worst, rel);
        }
    });
    console.log(cell === P14.gruCell ? 'fused gruCell' : 'composite gruCellAG', 'worst relative grad error =', worst.toExponential(2));
}
