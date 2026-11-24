// p12_bench.js — 第 12 篇原型检查：真实数据上的度量、模型梯度校验、前向 / 反向耗时。用法：node pages/tools/p12_bench.js
const { LM, P12 } = require('./p12_load.js');
const { AG } = LM;
const rand = LM.rng(7);

// 1) 真实视频上的度量（应当：one = 1、res ≈ 0、jump ≈ 平均速度）
{
    const D = P12.makeSet(400, rand);
    let one = 0, res = 0, jump = 0, size = 0, rerr = 0, sp = 0, bounce = 0;
    for (let i = 0; i < D.n; i++) {
        const m = P12.videoMetrics(D.X, i * P12.VD);
        one += m.one; res += m.res; jump += m.jump; size += m.size; rerr += Math.abs(m.rHat - D.balls[i].r); sp += D.balls[i].s;
        const b = D.balls[i]; let bc = 0; for (let t = 1; t < 8; t++) { const a = P12.ballAt(b, t - 1), c = P12.ballAt(b, t), e = P12.ballAt(b, t + 1); if (Math.abs((e[0] - c[0]) - (c[0] - a[0])) > 1e-6 || Math.abs((e[1] - c[1]) - (c[1] - a[1])) > 1e-6) bc++; } if (bc) bounce++;
    }
    const n = D.n;
    console.log(`real: one ${(one / n).toFixed(3)} res ${(res / n).toFixed(3)} px  jump ${(jump / n).toFixed(3)} (mean speed ${(sp / n).toFixed(3)})  size cv ${(size / n).toFixed(3)}  |r̂−r| ${(rerr / n).toFixed(3)}  videos with a bounce ${(bounce / n * 100).toFixed(0)}%`);
    // 打乱帧顺序后的度量（逐帧独立的上限参照）
    const Y = new Float32Array(P12.VD); let r2 = 0, j2 = 0;
    for (let i = 0; i < 200; i++) { for (let t = 0; t < 8; t++) { const k = Math.floor(rand() * n), f = Math.floor(rand() * 8); Y.set(D.X.subarray(k * P12.VD + f * 256, k * P12.VD + f * 256 + 256), t * 256); } const m = P12.videoMetrics(Y, 0); r2 += m.res; j2 += m.jump; }
    console.log(`random frames from different videos: res ${(r2 / 200).toFixed(3)} jump ${(j2 / 200).toFixed(3)}`);
}

// 2) 梯度校验（小模型）
function gradCheck(cfg) {
    const m = P12.createModel(cfg, 3), B = 2, nT = cfg.video ? 8 : 1;
    // 给零初始化的参数一点随机值，否则很多梯度恰好为 0
    const rr = LM.rng(11); m.params.forEach((p) => { for (let i = 0; i < p.size; i++) if (p.data[i] === 0) p.data[i] = 0.05 * LM.randn(rr); });
    const vids = new Float32Array(B * nT * 256); for (let b = 0; b < B; b++) { const bl = P12.sampleBall(rr); P12.renderVideo(bl, vids, b * nT * 256, nT); }
    const bt = P12.makeBatch(m, vids, B, { rand: LM.rng(5), known: cfg.video ? [0, 2] : null });
    const cls = cfg.video ? [1, 4] : null;
    const lossOf = () => P12.maskedMSE(P12.forward(m, bt.tok, B, bt.ts, cls), bt.tgt, bt.w);
    m.params.forEach((p) => { p.grad = null; });
    const L = lossOf(); AG.backward(L);
    let worst = 0, name = '';
    for (const p of m.params) {
        for (let q = 0; q < 3; q++) {
            const k = Math.floor((q * 7919 + 13) % p.size), old = p.data[k], eps = 1e-2;
            p.data[k] = old + eps; const lp = AG.noGrad(lossOf).item(); p.data[k] = old - eps; const lm = AG.noGrad(lossOf).item(); p.data[k] = old;
            const num = (lp - lm) / (2 * eps), a = p.grad ? p.grad[k] : 0, rel = Math.abs(num - a) / Math.max(1e-4, Math.abs(num) + Math.abs(a));
            if (rel > worst) { worst = rel; name = `${p.name}[${k}] ana ${a.toExponential(3)} num ${num.toExponential(3)}`; }
        }
    }
    console.log(`gradcheck ${cfg.video ? 'video' : 'image'}: worst rel ${worst.toExponential(2)} (${name}), loss ${L.item().toFixed(4)}`);
}
gradCheck({ d: 16, L: 2, H: 2, mlp: 2, video: false });
gradCheck({ d: 16, L: 2, H: 2, mlp: 2, video: true });

// 3) 耗时
function timeIt(cfg, B, reps) {
    const m = P12.createModel(cfg, 1), nT = cfg.video ? 8 : 1, vids = new Float32Array(B * nT * 256), rr = LM.rng(2);
    for (let b = 0; b < B; b++) P12.renderVideo(P12.sampleBall(rr), vids, b * nT * 256, nT);
    const bt = P12.makeBatch(m, vids, B, { rand: rr }), cls = cfg.video ? new Array(B).fill(1) : null;
    for (let i = 0; i < 3; i++) { m.params.forEach((p) => { if (p.grad) p.grad.fill(0); }); const L = P12.maskedMSE(P12.forward(m, bt.tok, B, bt.ts, cls), bt.tgt, bt.w); AG.backward(L); }
    let t0 = performance.now();
    for (let i = 0; i < reps; i++) { m.params.forEach((p) => { if (p.grad) p.grad.fill(0); }); const L = P12.maskedMSE(P12.forward(m, bt.tok, B, bt.ts, cls), bt.tgt, bt.w); AG.backward(L); }
    const tTrain = (performance.now() - t0) / reps;
    t0 = performance.now();
    for (let i = 0; i < reps; i++) AG.noGrad(() => P12.forward(m, bt.tok, B, bt.ts, cls));
    const tFwd = (performance.now() - t0) / reps;
    console.log(`${cfg.video ? 'video' : 'image'} d${cfg.d} L${cfg.L} H${cfg.H} mlp${cfg.mlp}: ${P12.numParams(m)} params · B=${B}: train step ${tTrain.toFixed(1)} ms, fwd ${tFwd.toFixed(1)} ms (${(tFwd / B).toFixed(2)} ms / sample)`);
}
timeIt({ d: 32, L: 2, H: 4, mlp: 4, video: true }, 8, 10);
timeIt({ d: 32, L: 2, H: 4, mlp: 2, video: true }, 8, 10);
timeIt({ d: 48, L: 2, H: 4, mlp: 2, video: true }, 8, 6);
timeIt({ d: 32, L: 2, H: 4, mlp: 4, video: false }, 64, 10);
timeIt({ d: 32, L: 2, H: 4, mlp: 4, video: true }, 1, 20);
