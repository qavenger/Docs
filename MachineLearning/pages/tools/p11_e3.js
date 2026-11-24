// 一次性编辑：P11 的密度网格、解析损失下限、前向马尔可夫链、噪声空间划分、score 误差曲线、模型加载（node tools/p11_e3.js）
const fs = require('fs'), path = require('path');
const file = path.join(__dirname, '..', '11_diffusion.lab.js');
let s = fs.readFileSync(file, 'utf8');
const rep = (a, b) => { if (!s.includes(a)) { console.error('missing:', a.slice(0, 90)); process.exit(1); } s = s.replace(a, () => b); };
rep(`    /** 权重存取（float16 base64，沿用 LM.gpt 的序列化，按 params 顺序） */`, `    // ═══════════════ 给预览用的解析量 ═══════════════
    /** 解析密度网格：G×G 个格点（覆盖 [−v, v]²，第 0 行在上）的 log q_t；o.cls ≥ 0 → q_t(x | c)；o.pc → 同时存 p(c | x_t) */
    function densityGrid(a, s2, v, G, o) {
        o = o || {};
        const n = G * G, X = new Float32Array(2 * n);
        for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { X[2 * (j * G + i)] = -v + (i + 0.5) / G * 2 * v; X[2 * (j * G + i) + 1] = v - (j + 0.5) / G * 2 * v; }
        const r = exact(a, s2, X, n, { logq: new Float32Array(n), cls: o.cls, pc: o.pc ? new Float32Array(3 * n) : null });
        return { G, v, l: r.logq, pc: r.pc };
    }
    /** 每个 t 的「不可约损失」E‖ε − E[ε | x_t]‖²/2（每维），按 bins 个 t 区间平均；total = t 均匀分布时的下限 */
    function lossFloor(sch, bins) {
        const rand = LM.rng(31), per = 300, X0 = sampleGMM(per, rand).X, xt = new Float32Array(2 * per), e = new Float32Array(2 * per), x0h = new Float32Array(2 * per), out = new Float64Array(bins);
        for (let b = 0; b < bins; b++) {
            let acc = 0, cnt = 0;
            for (let r = 0; r < 5; r++) {
                const i = Math.max(1, Math.min(T, Math.round((b + (r + 0.5) / 5) / bins * T))), a = sch.a[i], sg = sch.s[i];
                for (let q = 0; q < 2 * per; q++) { e[q] = LM.randn(rand); xt[q] = a * X0[q] + sg * e[q]; }
                exact(a, sg * sg, xt, per, { x0: x0h });
                for (let q = 0; q < 2 * per; q++) { const d = e[q] - (xt[q] - a * x0h[q]) / sg; acc += d * d; } cnt += 2 * per;
            }
            out[b] = acc / cnt;
        }
        return { bins: out, total: LM.mean(out) };
    }
    /** 逐步加噪的马尔可夫链 x_t = √(1 − β_t)·x_{t−1} + √β_t·z（每 every 步记一次）——与一步到位的 q(x_t | x₀) 同分布 */
    function forwardChain(sch, X0, n, rand, every) {
        const x = X0.slice(), traj = [x.slice()];
        for (let t = 1; t <= T; t++) { const c = Math.sqrt(1 - sch.beta[t]), d = Math.sqrt(sch.beta[t]); for (let q = 0; q < 2 * n; q++) x[q] = c * x[q] + d * LM.randn(rand); if (t % every === 0) traj.push(x.slice()); }
        return traj;
    }
    /** 噪声空间的划分：G×G 个起点 x_T（覆盖 [−v, v]²）各自采样到 x₀，记录落在哪条臂（DDIM 确定性 → 一张清晰的「地图」） */
    function partition(epsFn, sch, G, v, K, eta, seed) {
        const n = G * G, x = new Float32Array(2 * n);
        for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { x[2 * (j * G + i)] = -v + (i + 0.5) / G * 2 * v; x[2 * (j * G + i) + 1] = v - (j + 0.5) / G * 2 * v; }
        sampleVP(epsFn, sch, x, n, { K, eta, rand: LM.rng(seed) });
        const lp = logp(x, n), thr = offThreshold(), arm = armOf(x, n);
        for (let i = 0; i < n; i++) if (lp[i] < thr) arm[i] = -1;
        return { G, v, arm };
    }
    /** 学到的 ε̂ 与解析 ε* 的差距随 t 的变化：excess = E‖ε̂ − ε*‖²/2、floor = E‖ε − ε*‖²/2、x̂₀ 误差（RMSE） */
    function scoreError(net, sch, ts) {
        const rand = LM.rng(41), n = 400, X0 = sampleGMM(n, rand).X, xt = new Float32Array(2 * n), e = new Float32Array(2 * n), x0h = new Float32Array(2 * n), eh = new Float32Array(2 * n), f = epsNet(net);
        return ts.map((i) => {
            const a = sch.a[i], sg = sch.s[i];
            for (let q = 0; q < 2 * n; q++) { e[q] = LM.randn(rand); xt[q] = a * X0[q] + sg * e[q]; }
            exact(a, sg * sg, xt, n, { x0: x0h }); f(xt, n, i, eh);
            let ex = 0, fl = 0, xr = 0;
            for (let q = 0; q < 2 * n; q++) { const es = (xt[q] - a * x0h[q]) / sg; ex += (eh[q] - es) ** 2; fl += (e[q] - es) ** 2; xr += ((xt[q] - sg * eh[q]) / a - x0h[q]) ** 2; }
            return { t: i, excess: ex / (2 * n), floor: fl / (2 * n), x0rmse: Math.sqrt(xr / (2 * n)) };
        });
    }
    /** 读入内嵌的离线模型（P11_W）：vp / cfg / fm / rf 四个二维模型 + 数字的自编码器与潜空间扩散模型 */
    function loadModels() {
        if (typeof P11_W === 'undefined') return null;
        const m = {}; ['vp', 'cfg', 'fm', 'rf'].forEach((k) => { m[k] = loadNet(P11_W[k].cfg, P11_W[k].b64); });
        const L = P11_W.latent, ae = createAE(1); LM.gpt.loadWeights({ params: ae.params }, L.ae);
        m.lat = { ae, mu: Float32Array.from(L.mu), sd: Float32Array.from(L.sd) }; m.ldm = loadNet(L.cfg, L.b64);
        return m;
    }

    /** 权重存取（float16 base64，沿用 LM.gpt 的序列化，按 params 顺序） */`);
// loadModels 用到 loadNet：把它放到 loadNet 定义之后才调用即可（函数声明会提升，但 const loadNet 不会——loadModels 只在运行时调用，没有问题）
rep(`        velExact, pathRatio, noiseBank, offThreshold, quality, armOf, coverage, cfgQuality, jobRunner, LIVE, newLiveRun, liveWork, knn,`, `        velExact, pathRatio, noiseBank, offThreshold, quality, armOf, coverage, cfgQuality, jobRunner, LIVE, newLiveRun, liveWork, knn,
        densityGrid, lossFloor, forwardChain, partition, scoreError, loadModels,`);
fs.writeFileSync(file, s); console.log('ok');
