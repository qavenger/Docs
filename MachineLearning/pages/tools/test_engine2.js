// 第二批算子的梯度校验 + 共享数字生成器与第 02 篇逐像素一致性检查。用法：node pages/tools/test_engine2.js
const fs = require('fs');
const path = require('path');
const LM = require('../llm_common.js');
const { AG } = LM;

let fails = 0;
function numGradCheck(name, build, inputs, eps = 1e-2) {
    AG.clearTape();
    inputs.forEach((t) => { t.grad = null; });
    const out = build();
    AG.backward(out);
    let worst = 0;
    inputs.forEach((t, ti) => {
        const ana = Float64Array.from(t.grad || new Float32Array(t.size));
        const nCheck = Math.min(t.size, 40);
        for (let i = 0; i < nCheck; i++) {
            const k = Math.floor((i * 7919) % t.size), old = t.data[k];
            t.data[k] = old + eps; const lp = AG.noGrad(build).item();
            t.data[k] = old - eps; const lm = AG.noGrad(build).item();
            t.data[k] = old;
            const num = (lp - lm) / (2 * eps), a = ana[k];
            const rel = Math.abs(num - a) / Math.max(1e-3, Math.abs(num) + Math.abs(a));
            if (rel > worst) worst = rel;
            if (rel > 0.05) console.log(`  ${name} input#${ti}[${k}] analytic=${a.toExponential(3)} numeric=${num.toExponential(3)} rel=${rel.toFixed(3)}`);
        }
    });
    const ok = worst < 0.05; if (!ok) fails++;
    console.log(`${ok ? 'OK  ' : 'FAIL'} ${name.padEnd(22)} worst rel err ${worst.toExponential(2)}`);
}
const rand = LM.rng(42);
const P = (r, c, s = 1, off = 0) => AG.param('p', r, c, (a) => { for (let i = 0; i < a.length; i++) a[i] = off + s * LM.randn(rand); });
const proj = (T) => { const W = AG.tensor(T.r, T.c); const rr = LM.rng(9); for (let i = 0; i < W.size; i++) W.data[i] = rr() - 0.5; return AG.sum(AG.mul(T, W)); };

{ const A = P(3, 4); numGradCheck('exp', () => proj(AG.exp(A)), [A]); }
{ const A = AG.param('p', 3, 4, (a) => { for (let i = 0; i < a.length; i++) a[i] = 0.5 + 2 * rand(); }); numGradCheck('log', () => proj(AG.log(A)), [A]); }
{ const A = P(3, 4, 3); numGradCheck('softplus', () => proj(AG.softplus(A)), [A]); }
{ const A = P(3, 4); numGradCheck('square', () => proj(AG.square(A)), [A]); }
{ const A = P(3, 4, 2); numGradCheck('silu', () => proj(AG.silu(A)), [A]); }
{ const A = P(3, 4); numGradCheck('addConst', () => proj(AG.addConst(A, 0.7)), [A]); }
{ const A = P(3, 4), B = P(3, 2); numGradCheck('concatCols', () => proj(AG.concatCols(A, B)), [A, B]); }
{ const A = P(3, 6); numGradCheck('sliceCols', () => proj(AG.sliceCols(A, 1, 4)), [A]); }
{ const N = 2, C = 2, H = 6, W = 5, Co = 3, K = 3;
  const X = P(N, C * H * W), Wt = P(Co, C * K * K, 0.5), b = P(1, Co);
  numGradCheck('conv2d k3 pad1', () => proj(AG.conv2d(X, Wt, b, { C, H, W, K, pad: 1 })), [X, Wt, b]);
  numGradCheck('conv2d k3 stride2', () => proj(AG.conv2d(X, Wt, b, { C, H, W, K, stride: 2, pad: 1 })), [X, Wt, b]);
  numGradCheck('conv2d no bias pad0', () => proj(AG.conv2d(X, Wt, null, { C, H, W, K })), [X, Wt]);
  const y = AG.noGrad(() => AG.conv2d(X, Wt, b, { C, H, W, K, stride: 2, pad: 1 }));
  console.log(`     conv2d stride2 shape ${JSON.stringify(y.shape)} (expect C3 H3 W3)`);
}
{ const N = 2, C = 3, L = 9, Co = 2, K = 5;   // 一维卷积：H = 1，KH = 1
  const X = P(N, C * L), Wt = P(Co, C * K, 0.5), b = P(1, Co);
  numGradCheck('conv1d (H=1)', () => proj(AG.conv2d(X, Wt, b, { C, H: 1, W: L, KH: 1, KW: K, padW: 2 })), [X, Wt, b]); }
{ const N = 2, C = 2, H = 4, W = 6; const X = P(N, C * H * W);
  numGradCheck('avgPool2', () => proj(AG.avgPool2(X, { C, H, W })), [X]);
  numGradCheck('maxPool2', () => proj(AG.maxPool2(X, { C, H, W })), [X], 1e-3);
  numGradCheck('upsample2', () => proj(AG.upsample2(X, { C, H, W })), [X]); }
{ const N = 2, C = 3, HW = 5; const X = P(N, C * HW), B = P(N, C), S2 = P(N, C);
  numGradCheck('addChannel', () => proj(AG.addChannel(X, B, { C })), [X, B]);
  numGradCheck('mulChannel', () => proj(AG.mulChannel(X, S2, { C })), [X, S2]); }
// 一个小 U-Net 风格的串联：conv → silu → pool → conv → upsample → concat skip → conv
{ const N = 2, C = 1, H = 8, W = 8, F = 3;
  const X = P(N, C * H * W), W1 = P(F, C * 9, 0.5), W2 = P(F, F * 9, 0.4), W3 = P(1, 2 * F * 9, 0.3), b1 = P(1, F), b3 = P(1, 1), T = P(N, F);
  numGradCheck('mini U-Net chain', () => {
      const h1 = AG.silu(AG.addChannel(AG.conv2d(X, W1, b1, { C, H, W, K: 3, pad: 1 }), T, { C: F }));
      const d = AG.avgPool2(h1, { C: F, H, W });
      const h2 = AG.silu(AG.conv2d(d, W2, null, { C: F, H: H / 2, W: W / 2, K: 3, pad: 1 }));
      const u = AG.upsample2(h2, { C: F, H: H / 2, W: W / 2 });
      // 通道拼接 = 按行把两张特征图首尾相接（每行是一张图的 C·H·W），所以用 concatCols
      const cat = AG.concatCols(h1, u);
      const out = AG.conv2d(cat, W3, b3, { C: 2 * F, H, W, K: 3, pad: 1 });
      return AG.mse(out, AG.tensor(N, H * W));
  }, [X, W1, W2, W3, b1, b3, T]); }

// 共享数字生成器 vs 第 02 篇自己的生成器：同一个随机种子必须逐像素相同
{
    const p02 = path.join(__dirname, '..', '02_mlp_backprop.lab.js');
    if (fs.existsSync(p02)) {
        const src = fs.readFileSync(p02, 'utf8');
        const P02 = new Function('LM', src.slice(0, src.indexOf('PipelineLab.mount(')) + '\nreturn P02;')(LM);
        const a = P02.makeSet(200, LM.rng(123)), b = LM.digits.makeSet(200, LM.rng(123));
        let md = 0; for (let i = 0; i < a.X.length; i++) md = Math.max(md, Math.abs(a.X[i] - b.X[i]));
        const sameY = a.y.every((v, i) => v === b.y[i]);
        const ok = md === 0 && sameY; if (!ok) fails++;
        console.log(`${ok ? 'OK  ' : 'FAIL'} LM.digits == P02 digits   max |Δpixel| = ${md}, labels equal = ${sameY}`);
    }
}
// 卷积速度（16×16 数字、8 通道 3×3、batch 32）
{
    const N = 32, X = AG.tensor(N, 256), Wt = P(8, 9, 0.3), b = P(1, 8);
    for (let i = 0; i < X.size; i++) X.data[i] = Math.random();
    const t0 = Date.now(); let reps = 0;
    while (Date.now() - t0 < 300) { AG.clearTape(); Wt.grad = null; const y = AG.conv2d(X, Wt, b, { C: 1, H: 16, W: 16, K: 3, pad: 1 }); AG.backward(AG.mean(y)); reps++; }
    console.log(`     conv2d 1→8 ch 16×16 ×32 fwd+bwd: ${((Date.now() - t0) / reps).toFixed(2)} ms`);
}
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
