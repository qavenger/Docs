// p02_opt2d.js — 二维峡谷函数上各优化器设置的对比（选第 09 步的默认超参数）
// 统计：150 步后的 f、y 方向符号翻转次数（锯齿程度）、轨迹离开视窗的最大 |θ|
const { P02 } = require('./p02_proto.js');
const cfgs = [['sgd', { lr: 0.22 }], ['sgd', { lr: 0.4 }], ['sgd', { lr: 0.45 }], ['momentum', { lr: 0.22, mu: 0.9 }], ['momentum', { lr: 0.4, mu: 0.9 }], ['momentum', { lr: 0.45, mu: 0.9 }], ['adam', { lr: 0.2 }]];
for (const [k, hp] of cfgs) {
    P02.OPT2D[k] = hp;
    for (const st of [[-4.5, 1.6], [3.5, 2]]) {
        const r = P02.runOpt2d(k, st[0], st[1], 150), at = (t) => (r.fs[t] ?? NaN).toExponential(1);
        let flips = 0, maxA = 0;
        for (let i = 2; i < r.pts.length; i++) { const d1 = r.pts[i][1] - r.pts[i - 1][1], d0 = r.pts[i - 1][1] - r.pts[i - 2][1]; if (i < 40 && d1 * d0 < 0) flips++; maxA = Math.max(maxA, Math.abs(r.pts[i][0]), Math.abs(r.pts[i][1])); }
        console.log(`${k.padEnd(8)} ${JSON.stringify(hp).padEnd(22)} start ${st}  f@20 ${at(20)} @50 ${at(50)} @150 ${at(150)}  y-flips(first 40) ${flips}  max|θ| ${maxA.toFixed(2)}  n=${r.pts.length}`);
    }
}
