// 一次性编辑：给 P11 加单步采样、扫描任务、实时训练、质量指标（node tools/p11_e2.js）
const fs = require('fs'), path = require('path');
const file = path.join(__dirname, '..', '11_diffusion.lab.js');
let s = fs.readFileSync(file, 'utf8');
const rep = (a, b) => { if (!s.includes(a)) { console.error('missing:', a.slice(0, 90)); process.exit(1); } s = s.replace(a, () => b); };

rep(`    function sampleVP(epsFn, sch, x, n, o) {
        const din = o.din || 2, D = din * n, K = o.K, tau = tauGrid(K), eta = o.eta || 0, rand = o.rand, eps = new Float32Array(D), clip = o.clip ?? 2.5;
        const traj = o.keep ? [x.slice()] : null, x0s = o.keep ? [] : null;
        for (let k = K; k >= 1; k--) {
            const i = tau[k], j = tau[k - 1], ai = sch.a[i], si = sch.s[i], abj = sch.ab[j], abi = sch.ab[i];
            epsFn(x, n, i, eps);
            const sig = eta * Math.sqrt((1 - abj) / (1 - abi) * (1 - abi / abj)), cdir = Math.sqrt(Math.max(0, 1 - abj - sig * sig)), aj = Math.sqrt(abj);
            const x0k = o.keep ? new Float32Array(D) : null;
            for (let q = 0; q < D; q++) {
                let x0 = (x[q] - si * eps[q]) / ai; x0 = x0 < -clip ? -clip : x0 > clip ? clip : x0;
                if (x0k) x0k[q] = x0;
                x[q] = aj * x0 + cdir * eps[q] + (sig > 0 ? sig * LM.randn(rand) : 0);
            }
            if (traj) { traj.push(x.slice()); x0s.push(x0k); }
        }
        return { x, traj, x0s, tau };
    }`,
`    function sampleVP(epsFn, sch, x, n, o) {
        const din = o.din || 2, D = din * n, K = o.K, tau = tauGrid(K), eta = o.eta || 0, rand = o.rand, eps = new Float32Array(D), clip = o.clip ?? CLIP;
        const traj = o.keep ? [x.slice()] : null, x0s = o.keep ? [] : null;
        for (let k = K; k >= 1; k--) {
            const i = tau[k], j = tau[k - 1], x0k = o.keep ? new Float32Array(D) : null;
            epsFn(x, n, i, eps);
            vpStep(sch, x, D, i, j, eps, eta, rand, clip, x0k);
            if (traj) { traj.push(x.slice()); x0s.push(x0k); }
        }
        return { x, traj, x0s, tau };
    }
    /**
     * DDIM-η 的一步：从 τ_k = i 到 τ_{k−1} = j（eps = ε̂(x_i, i)）。
     * x̂₀ = (x − √(1−ᾱ_i)·ε̂)/√ᾱ_i（截断在 ±clip：cosine 日程 √ᾱ_T ≈ 5×10⁻⁵，不截断时网络误差会被放大上万倍）
     * x_j = √ᾱ_j·x̂₀ + √(1 − ᾱ_j − σ²)·ε̂ + σ·z，σ = η·√((1−ᾱ_j)/(1−ᾱ_i))·√(1 − ᾱ_i/ᾱ_j)
     */
    const CLIP = 1.25;
    function vpStep(sch, x, D, i, j, eps, eta, rand, clip, x0out) {
        const ai = sch.a[i], si = sch.s[i], abj = sch.ab[j], abi = sch.ab[i];
        const sig = eta * Math.sqrt((1 - abj) / (1 - abi) * (1 - abi / abj)), cdir = Math.sqrt(Math.max(0, 1 - abj - sig * sig)), aj = Math.sqrt(abj);
        for (let q = 0; q < D; q++) {
            let x0 = (x[q] - si * eps[q]) / ai; x0 = x0 < -clip ? -clip : x0 > clip ? clip : x0;
            if (x0out) x0out[q] = x0;
            x[q] = aj * x0 + cdir * eps[q] + (sig > 0 ? sig * LM.randn(rand) : 0);
        }
    }`);

rep(`    // ═══════════════ 潜空间扩散（Step 09）`, `    /** 解析的流匹配速度场 v*(x, t) = E[ε − x₀ | x_t] = (x_t − E[x₀|x_t])/t（路径 x_t = (1 − t)·x₀ + t·ε） */
    function velExact(X, n, t, out) {
        const x0 = new Float32Array(2 * n); exact(1 - t, t * t, X, n, { x0 });
        for (let q = 0; q < 2 * n; q++) out[q] = (X[q] - x0[q]) / t;
        return out;
    }
    /** 轨迹的「弯曲度」：每个粒子的路径长 / 起点到终点的直线距离（= 1 表示笔直），取平均 */
    function pathRatio(traj, n) {
        let s = 0; const K = traj.length - 1;
        for (let i = 0; i < n; i++) {
            let L = 0; for (let k = 0; k < K; k++) L += Math.hypot(traj[k + 1][2 * i] - traj[k][2 * i], traj[k + 1][2 * i + 1] - traj[k][2 * i + 1]);
            s += L / Math.max(1e-9, Math.hypot(traj[K][2 * i] - traj[0][2 * i], traj[K][2 * i + 1] - traj[0][2 * i + 1]));
        }
        return s / n;
    }
    const noiseBank = (n, seed, din) => { const x = new Float32Array((din || 2) * n), r = LM.rng(seed); for (let q = 0; q < x.length; q++) x[q] = LM.randn(r); return x; };

    // ═══════════════ 质量指标：MMD（分布距离）+ 偏离率（落在真实密度最低 1% 区域的比例）+ 臂的分类 / 覆盖率 ═══════════════
    let _thr = null;
    /** 偏离阈值：3000 个真实样本的 log p 的 1% 分位数（真实样本自己的偏离率 ≈ 1%） */
    function offThreshold() { if (_thr == null) { const X = sampleGMM(3000, LM.rng(101)).X, l = Array.from(logp(X, 3000)).sort((a, b) => a - b); _thr = l[30]; } return _thr; }
    function quality(X, n) { const l = logp(X, n), thr = offThreshold(); let off = 0; for (let i = 0; i < n; i++) if (l[i] < thr) off++; return { mmd: mmdTrue(X, n), off: off / n }; }
    /** 用真实密度做「贝叶斯分类器」：每个点归到 p(c | x) 最大的臂 */
    function armOf(X, n) { const pc = new Float32Array(3 * n), out = new Int8Array(n); exact(1, 0, X, n, { pc }); for (let i = 0; i < n; i++) { let b = 0; for (let c = 1; c < 3; c++) if (pc[3 * i + c] > pc[3 * i + b]) b = c; out[i] = b; } return out; }
    /** 覆盖率：臂 c 的 16 个分量里，有多少个在 1σ 之内至少有一个生成样本 */
    function coverage(X, n, cls) {
        const g = GMM; let cov = 0, tot = 0;
        for (let k = 0; k < g.M; k++) { if (g.cls[k] !== cls) continue; tot++; for (let i = 0; i < n; i++) if (Math.hypot(X[2 * i] - g.mu[2 * k], X[2 * i + 1] - g.mu[2 * k + 1]) < SIG) { cov++; break; } }
        return cov / tot;
    }
    /** CFG 的指标：类别准确率（贝叶斯分类器）、到该类真实分布的 MMD、覆盖率、平均半径 */
    function cfgQuality(X, n, cls) { const a = armOf(X, n); let acc = 0, rr = 0; for (let i = 0; i < n; i++) { if (a[i] === cls) acc++; rr += Math.hypot(X[2 * i], X[2 * i + 1]); } return { acc: acc / n, mmd: mmdTrue(X, n, cls), cov: coverage(X, n, cls), r: rr / n }; }

    // ═══════════════ 逐帧推进的采样任务（Step 06–09 的扫描：每帧只做固定数量的网络前向，不卡顿） ═══════════════
    /** job: { kind: 'vp' | 'fm', epsFn（vp）| net（fm）, sch, K, eta, n, din, seed, clip, cost（每个粒子一次前向的相对代价）, metric(x, n) → 结果 } */
    function jobRunner(jobs) {
        const R = { jobs, cur: 0, res: [], done: !jobs.length, work: 0, total: jobs.reduce((s, j) => s + j.K * j.n * (j.cost ?? 1), 0) };
        R.step = function (budget) {
            while (budget > 0 && R.cur < jobs.length) {
                const jb = jobs[R.cur], D = (jb.din || 2) * jb.n;
                if (!jb.x) { jb.x = noiseBank(jb.n, jb.seed, jb.din); jb.k = jb.K; jb.e = new Float32Array(D); jb.rand = LM.rng(jb.seed ^ 0x5bd1); jb.tau = tauGrid(jb.K); if (jb.kind === 'fm') jb.cv = new Float32Array(jb.net.dc); }
                if (jb.kind === 'fm') { netEval(jb.net, jb.x, jb.n, condVec(jb.net, 1000 * jb.k / jb.K, -1, jb.cv), jb.e); for (let q = 0; q < D; q++) jb.x[q] -= jb.e[q] / jb.K; }
                else { const i = jb.tau[jb.k], j = jb.tau[jb.k - 1]; jb.epsFn(jb.x, jb.n, i, jb.e); vpStep(jb.sch, jb.x, D, i, j, jb.e, jb.eta || 0, jb.rand, jb.clip ?? CLIP); }
                jb.k--; const c = jb.n * (jb.cost ?? 1); budget -= c; R.work += c;
                if (jb.k === 0) { R.res.push(Object.assign({ job: jb }, jb.metric(jb.x, jb.n))); jb.x = jb.e = null; R.cur++; }
            }
            R.done = R.cur >= jobs.length;
            return R;
        };
        return R;
    }

    // ═══════════════ Step 03：页面里实时训练的去噪器（每帧 spf 步训练 + 1 步「滚动采样」评估当前模型） ═══════════════
    const LIVE = { steps: 4000, B: 128, width: 48, depth: 3, ff: 4, warmup: 100, ema: 0.995, rollN: 256, rollK: 20, bins: 20 };
    function newLiveRun(o) {
        const tr = trainer({ kind: 'vp', sched: o.sched, data: o.data, width: LIVE.width, depth: LIVE.depth, ff: LIVE.ff, B: LIVE.B, lr: o.lr, steps: LIVE.steps, warmup: LIVE.warmup, ema: LIVE.ema, seed: o.seed });
        return { o, tr, binL: new Float64Array(LIVE.bins), binN: new Float64Array(LIVE.bins), evals: [], roll: null, last: null, rolls: 0, finished: false };
    }
    function liveWork(R, spf) {
        const tr = R.tr, sch = R.o.sched;
        for (let s = 0; s < spf && tr.it < LIVE.steps; s++) {
            const r = tr.step();
            for (let b = 0; b < r.t.length; b++) { const k = Math.min(LIVE.bins - 1, Math.floor((r.t[b] - 1) / T * LIVE.bins)); R.binL[k] = R.binN[k] < 1 ? r.err[b] : 0.985 * R.binL[k] + 0.015 * r.err[b]; R.binN[k]++; }
        }
        if (R.finished) return;
        if (!R.roll) R.roll = { x: noiseBank(LIVE.rollN, 9000 + R.rolls, 2), k: LIVE.rollK, it: tr.it, tau: tauGrid(LIVE.rollK), e: new Float32Array(2 * LIVE.rollN), f: epsNet(tr.ema) };
        const ro = R.roll, i = ro.tau[ro.k], j = ro.tau[ro.k - 1];
        ro.f(ro.x, LIVE.rollN, i, ro.e); vpStep(sch, ro.x, 2 * LIVE.rollN, i, j, ro.e, 0, null, CLIP); ro.k--;
        if (ro.k === 0) {
            const q = quality(ro.x, LIVE.rollN);
            R.last = { x: ro.x, it: ro.it, mmd: q.mmd, off: q.off }; R.evals.push([ro.it, q.mmd, q.off]); R.rolls++; R.roll = null;
            if (ro.it >= LIVE.steps) R.finished = true;
        }
    }

    // ═══════════════ 潜空间扩散（Step 09）`);

rep(`    /** 推理用：标准化潜变量 z' = (z − μ)/s ↔ 像素 */`, `    /** kNN 分类（k = 5，欧氏距离，近的票重）：在标准化潜空间里给生成的数字打标签（参考集 = 训练集的潜变量） */
    function knn(Q, n, ref, refY, m, d) {
        const out = new Int8Array(n), bd = new Float64Array(5), bl = new Int8Array(5);
        for (let i = 0; i < n; i++) {
            bd.fill(Infinity);
            for (let j = 0; j < m; j++) {
                let s = 0; for (let k = 0; k < d; k++) { const t = Q[i * d + k] - ref[j * d + k]; s += t * t; }
                if (s < bd[4]) { let p = 4; while (p > 0 && bd[p - 1] > s) { bd[p] = bd[p - 1]; bl[p] = bl[p - 1]; p--; } bd[p] = s; bl[p] = refY[j]; }
            }
            const v = new Int32Array(10); for (let k = 0; k < 5; k++) v[bl[k]] += 6 - k; out[i] = LM.argmax(v);
        }
        return out;
    }
    /** 推理用：标准化潜变量 z' = (z − μ)/s ↔ 像素 */`);

rep(`        trainer, tauGrid, epsNet, epsExact, sampleVP, sampleFM, MMD_H, mmd, mmdTrue, logp,`, `        trainer, tauGrid, epsNet, epsExact, sampleVP, vpStep, CLIP, sampleFM, MMD_H, mmd, mmdTrue, logp,
        velExact, pathRatio, noiseBank, offThreshold, quality, armOf, coverage, cfgQuality, jobRunner, LIVE, newLiveRun, liveWork, knn,`);
fs.writeFileSync(file, s); console.log('ok');
