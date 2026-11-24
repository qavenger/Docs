// p12_measure.js — 第 12 篇 Step 03–09 引用的实测数字（与页面同一份代码 + 同一份离线权重 P12_W）
// 用法（在 pages/ 下）：node tools/p12_measure.js [time|frame|uncond|i2v|cfg|long|all]
const { LM, P12, P12_W } = require('./p12_load.js');
const what = process.argv[2] || 'all';
const now = () => Number(process.hrtime.bigint()) / 1e6;
function loadModel(w) { const m = P12.createModel(w.cfg, 1); P12.loadParams(m, w.b64); return m; }
const IMG = loadModel(P12_W.image), VID = loadModel(P12_W.video);
const K = 16;
const avg = (a) => a.reduce((s, v) => s + v, 0) / Math.max(1, a.length);
const fin = (a) => a.filter((v) => isFinite(v));
function summarize(X, n, nF) {
    const ms = []; for (let i = 0; i < n; i++) ms.push(P12.videoMetrics(X, i * (nF || P12.T) * P12.NPX, nF));
    return { one: +avg(ms.map((m) => m.one)).toFixed(3), jump: +avg(fin(ms.map((m) => m.jump))).toFixed(2), res: +avg(fin(ms.map((m) => m.res))).toFixed(2), resMed: +median(fin(ms.map((m) => m.res))).toFixed(2), size: +avg(fin(ms.map((m) => m.size))).toFixed(3), ms };
}
function median(a) { const b = a.slice().sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : NaN; }
/** 逐帧生成：图像模型一次生成 8 帧（B = 8），噪声相关系数 rho：ε_t = √ρ·ε_shared + √(1−ρ)·ε_t' */
function perFrame(n, rho, seed) {
    const rr = LM.rng(seed), X = new Float32Array(n * P12.VD);
    for (let v = 0; v < n; v++) {
        const noise = new Float32Array(P12.VD), sh = new Float32Array(P12.NPX);
        for (let i = 0; i < P12.NPX; i++) sh[i] = LM.randn(rr);
        for (let t = 0; t < P12.T; t++) for (let i = 0; i < P12.NPX; i++) noise[t * P12.NPX + i] = Math.sqrt(rho) * sh[i] + Math.sqrt(1 - rho) * LM.randn(rr);
        const x = P12.sampleAll(IMG, { B: P12.T, K, rand: rr, noise });
        X.set(x, v * P12.VD);
    }
    return X;
}
function videoGen(n, o, seed) {
    const rr = LM.rng(seed), X = new Float32Array(n * P12.VD), B = 8;
    for (let v0 = 0; v0 < n; v0 += B) {
        const b = Math.min(B, n - v0), cond = o.cond ? o.cond.subarray(v0 * P12.VD, (v0 + b) * P12.VD) : null;
        const cls = o.cls != null ? new Array(b).fill(o.cls) : null;
        const x = P12.sampleAll(VID, { B: b, K, rand: rr, cls, w: o.w ?? 1, known: o.known || 0, cond });
        X.set(x, v0 * P12.VD);
    }
    return X;
}
const out = {};
/** 拟合轨迹在 t = 0 的速度：tri 是偶函数，(a, v) 与 (−a, −v) 是同一条轨迹，所以方向要看 t = 0 处三角波的斜率 */
function vel0(fit) { const s = (f) => { const L = P12.W - 2 * fit.r, m = ((f.a % (2 * L)) + 2 * L) % (2 * L); return m < L ? f.v : -f.v; }; return [s(fit.fx), s(fit.fy)]; }
if (what === 'time' || what === 'all') {
    for (const [name, m, B] of [['video B=1', VID, 1], ['video B=4', VID, 4], ['video B=8', VID, 8], ['image B=8', IMG, 8]]) {
        const nT = m.cfg.video ? P12.T : 1, tok = new Float32Array(B * nT * P12.NS * m.inD), ts = new Float32Array(B).fill(0.5);
        for (let i = 0; i < tok.length; i++) tok[i] = Math.random() - 0.5;
        for (let i = 0; i < 3; i++) LM.AG.noGrad(() => P12.forward(m, tok, B, ts, m.cfg.video ? new Array(B).fill(P12.NCLS) : null));
        const t0 = now(), R = 20; for (let i = 0; i < R; i++) LM.AG.noGrad(() => P12.forward(m, tok, B, ts, m.cfg.video ? new Array(B).fill(P12.NCLS) : null));
        console.log(`forward ${name}: ${((now() - t0) / R).toFixed(2)} ms`);
    }
    console.log('params image', P12.numParams(IMG), 'video', P12.numParams(VID));
}
if (what === 'frame' || what === 'all') {
    const ref = P12.makeSet(300, LM.rng(777)), sr = summarize(ref.X, ref.n); delete sr.ms; out.real = sr; console.log('real (300)', JSON.stringify(sr));
    for (const rho of [0, 0.5, 0.8, 0.95, 1]) { const X = perFrame(32, rho, 5), s = summarize(X, 32); delete s.ms; out['frame_rho' + rho] = s; console.log(`per-frame rho=${rho}`, JSON.stringify(s)); }
}
if (what === 'uncond' || what === 'all') {
    const X = videoGen(32, {}, 11), s = summarize(X, 32); const worst = s.ms.map((m, i) => [m.res, i]).filter((p) => isFinite(p[0])).sort((a, b) => b[0] - a[0]).slice(0, 4); delete s.ms; out.uncond = s; console.log('video uncond (32)', JSON.stringify(s), 'worst', JSON.stringify(worst));
}
if (what === 'i2v' || what === 'all') {
    const n = 32, rr = LM.rng(4242), cond = new Float32Array(n * P12.VD), balls = [];
    for (let b = 0; b < n; b++) { const bl = P12.sampleBall(rr); balls.push(bl); P12.renderVideo(bl, cond, b * P12.VD); }
    for (const known of [1, 2, 4]) {
        const X = videoGen(n, { known, cond }, 13 + known), s = summarize(X, n);
        // 与真实后续帧的球心误差（只看 known 之后的帧）
        let err = 0, ne = 0; const perT = new Float64Array(P12.T), nT = new Float64Array(P12.T);
        for (let v = 0; v < n; v++) { const m = s.ms[v]; for (let t = known; t < P12.T; t++) if (m.ok[t]) { const c = P12.ballAt(balls[v], t), e = Math.hypot(m.cx[t] - c[0], m.cy[t] - c[1]); err += e; ne++; perT[t] += e; nT[t]++; } }
        delete s.ms; s.posErr = +(err / ne).toFixed(2); s.errByT = Array.from(perT, (e, t) => (nT[t] ? +(e / nT[t]).toFixed(2) : null));
        out['i2v_k' + known] = s; console.log(`i2v known=${known}`, JSON.stringify(s));
    }
}
if (what === 'cfg' || what === 'all') {
    for (const w of (process.argv[3] ? process.argv[3].split(",").map(Number) : [0, 1, 2, 4])) {
        let ok = 0, tot = 0; const agg = [], spd = [];
        for (let c = 0; c < P12.NCLS; c++) {
            const X = w === 0 ? videoGen(16, {}, 100 + c) : videoGen(16, { cls: c, w }, 100 + c), s = summarize(X, 16);
            for (const m of s.ms) { if (!m.fit) continue; tot++; const v0 = vel0(m.fit); if (Math.hypot(v0[0], v0[1]) > 0.3 && P12.dirClass(v0[0], v0[1]) === c) ok++; spd.push(Math.hypot(v0[0], v0[1])); }
            delete s.ms; agg.push(s);
        }
        const r = { speed: +avg(spd).toFixed(2), acc: +(ok / tot).toFixed(3), one: +avg(agg.map((a) => a.one)).toFixed(3), res: +avg(agg.map((a) => a.res)).toFixed(2) };
        out['cfg_w' + w] = r; console.log(`class cond w=${w}${w === 0 ? ' (= unconditional, chance 0.25)' : ''}`, JSON.stringify(r));
    }
}
if (what === 'long' || what === 'all') {
    const L = 32, n = 16;
    for (const ctx of [1, 2, 4]) {
        const rr = LM.rng(900 + ctx), X = new Float32Array(n * L * P12.NPX);
        for (let v = 0; v < n; v++) {
            const first = P12.sampleAll(VID, { B: 1, K, rand: rr });          // 第一块：无条件 8 帧
            X.set(first, v * L * P12.NPX); let have = P12.T;
            while (have < L) {
                const cond = new Float32Array(P12.VD); cond.set(X.subarray(v * L * P12.NPX + (have - ctx) * P12.NPX, v * L * P12.NPX + have * P12.NPX), 0);
                const x = P12.sampleAll(VID, { B: 1, K, rand: rr, known: ctx, cond }), add = Math.min(P12.T - ctx, L - have);
                X.set(x.subarray(ctx * P12.NPX, (ctx + add) * P12.NPX), v * L * P12.NPX + have * P12.NPX); have += add;
            }
        }
        const s = summarize(X, n, L);
        // 漂移：每 8 帧窗口的物理残差、球质量相对第一块的变化
        const win = []; for (let w0 = 0; w0 + 8 <= L; w0 += 8) { const r = [], mm = []; for (let v = 0; v < n; v++) { const m = P12.videoMetrics(X, v * L * P12.NPX + w0 * P12.NPX, 8); if (isFinite(m.res)) r.push(m.res); mm.push(avg(m.st.map((q) => q.m))); } win.push({ w0, res: +avg(r).toFixed(2), mass: +avg(mm).toFixed(1) }); }
        delete s.ms; s.windows = win; out['long_ctx' + ctx] = s; console.log(`long ${L} frames ctx=${ctx}`, JSON.stringify(s));
    }
}
require('fs').writeFileSync(require('path').join(__dirname, `p12_measure_${what}.json`), JSON.stringify(out, null, 1));
