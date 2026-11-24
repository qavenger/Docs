// 评估 W=48 的 checkpoint（解析 MMD，N = 500）
const fs = require('fs'), path = require('path');
const { LM, P11 } = require('./p11_load.js');
const load = (n, w = 'ema') => { const j = JSON.parse(fs.readFileSync(path.join(__dirname, 'p11_ckpt', n + '.json'), 'utf8')); return P11.loadNet(j.cfg, j[w]); };
const N = 500;
const lpRef = Array.from(P11.logp(P11.sampleGMM(3000, LM.rng(101)).X, 3000)).sort((a, b) => a - b), thr = lpRef[Math.floor(0.01 * 3000)];
const st = (X, n, cls) => { const l = P11.logp(X, n, null, cls); let off = 0; for (let i = 0; i < n; i++) if (l[i] < thr) off++; return `${P11.mmdTrue(X, n, cls).toFixed(2).padStart(7)} ${(100 * off / n).toFixed(1).padStart(5)}%`; };
const noise = (seed) => { const x = new Float32Array(2 * N), r = LM.rng(seed); for (let i = 0; i < 2 * N; i++) x[i] = LM.randn(r); return x; };
const what = process.argv[2];
if (what === 'vp') for (const name of ['vp48', 'vp48s4k', 'vp_cos']) { const net = load(name), sch = P11.SCHED.cosine; let line = name.padEnd(8); for (const [K, eta] of [[5, 0], [10, 0], [20, 0], [50, 0], [100, 0], [20, 1], [50, 1], [100, 1], [250, 1]]) { const a = noise(77); P11.sampleVP(P11.epsNet(net), sch, a, N, { K, eta, rand: LM.rng(9), clip: 1.25 }); line += ` | K${K}η${eta} ${st(a, N)}`; } console.log(line); }
if (what === 'fm') for (const name of ['fm48', 'rf48', 'fm']) { const net = load(name); let line = name.padEnd(6); for (const K of [1, 2, 3, 4, 8, 16, 32, 64]) { const a = noise(77); P11.sampleFM(net, a, N, { K }); line += ` | K${K} ${st(a, N)}`; } console.log(line); }
if (what === 'straight') {
    // 轨迹「弯曲度」：路径长 / 弦长（K = 100 步），以及 Liu et al. 的直线度 S = Σ‖(x_end − x_start) − Δx_k·K‖²/K
    const n = 300;
    const measure = (traj) => { let ratio = 0, S = 0; const K = traj.length - 1; for (let i = 0; i < n; i++) { let L = 0; const cx = traj[K][2 * i] - traj[0][2 * i], cy = traj[K][2 * i + 1] - traj[0][2 * i + 1]; for (let k = 0; k < K; k++) { const dx = traj[k + 1][2 * i] - traj[k][2 * i], dy = traj[k + 1][2 * i + 1] - traj[k][2 * i + 1]; L += Math.hypot(dx, dy); S += ((cx - dx * K) ** 2 + (cy - dy * K) ** 2) / K; } ratio += L / Math.hypot(cx, cy); } return `len/chord ${(ratio / n).toFixed(3)}  S ${(S / n).toFixed(3)}`; };
    const x0 = new Float32Array(2 * n), r = LM.rng(55); for (let i = 0; i < 2 * n; i++) x0[i] = LM.randn(r);
    for (const sk of ['cosine', 'linear']) { const a = x0.slice(); const res = P11.sampleVP(P11.epsExact(P11.SCHED[sk]), P11.SCHED[sk], a, n, { K: 100, keep: true }); console.log(`VP-${sk} DDIM exact : ${measure(res.traj)}`); }
    { const a = x0.slice(); const res = P11.sampleVP(P11.epsNet(load('vp48')), P11.SCHED.cosine, a, n, { K: 100, keep: true, clip: 1.25 }); console.log(`VP-cosine DDIM net  : ${measure(res.traj)}`); }
    for (const name of ['fm48', 'rf48']) { const a = x0.slice(); const res = P11.sampleFM(load(name), a, n, { K: 100, keep: true }); console.log(`${name} Euler       : ${measure(res.traj)}`); }
}
if (what === 'cfg') {
    const net = load('cfg48'), sch = P11.SCHED.cosine, n = 400;
    for (let c = 0; c < 3; c++) {
        let line = `class ${c}`;
        for (const s of [0, 0.5, 1, 1.5, 2, 3, 4, 6]) {
            const x = new Float32Array(2 * n), r = LM.rng(77); for (let i = 0; i < 2 * n; i++) x[i] = LM.randn(r);
            P11.sampleVP(P11.epsNet(net, { cls: c, guide: s }), sch, x, n, { K: 50, eta: 0, rand: LM.rng(9), clip: 1.25 });
            const pc = new Float32Array(3 * n); P11.exact(1, 0, x, n, { pc }); let acc = 0, rr = 0; for (let i = 0; i < n; i++) { let b = 0; for (let k = 1; k < 3; k++) if (pc[3 * i + k] > pc[3 * i + b]) b = k; if (b === c) acc++; rr += Math.hypot(x[2 * i], x[2 * i + 1]) / n; }
            line += ` | s${s}: acc ${(100 * acc / n).toFixed(0)}% mmd ${P11.mmdTrue(x, n, c).toFixed(1)} r̄ ${rr.toFixed(2)}`;
        }
        console.log(line);
    }
}
