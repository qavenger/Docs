// 01_linear_regression.lab.js — 系列第 01 篇：机器学习的 Hello World（线性回归 + 梯度下降）
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js 01_linear_regression.lab.js -o 01_linear_regression.html
// 页面里的每张预览都由下面的 JS 参考实现实时算出：数据、损失地形、GD / SGD 轨迹、多项式拟合、自动微分对照。

const P01 = (() => {
    const W_TRUE = 2.0, B_TRUE = 1.0;          // 真实规律 y = 2x + 1（模型并不知道）
    const NMAX = 200, NTEST = 200, K = 80;     // 最多样本数、测试集大小、GD 迭代步数
    const VIEW = { w: [-2, 6], b: [-3, 5] };   // 损失地形的显示窗口
    const GRID = 96;
    const DEG_MAX = 12;
    const RATE = 12;                           // 动画：每秒播放的迭代数

    /** 充分统计量：MSE 是 (w, b) 的二次函数 L = sxx·w² + 2sx·wb + b² − 2sxy·w − 2sy·b + syy（都是均值） */
    function stats(x, y, n) {
        let sx = 0, sxx = 0, sy = 0, sxy = 0, syy = 0;
        for (let i = 0; i < n; i++) { sx += x[i]; sxx += x[i] * x[i]; sy += y[i]; sxy += x[i] * y[i]; syy += y[i] * y[i]; }
        return { sx: sx / n, sxx: sxx / n, sy: sy / n, sxy: sxy / n, syy: syy / n, n };
    }
    const lossQ = (s, w, b) => s.sxx * w * w + 2 * s.sx * w * b + b * b - 2 * s.sxy * w - 2 * s.sy * b + s.syy;
    const gradQ = (s, w, b) => [2 * (s.sxx * w + s.sx * b - s.sxy), 2 * (s.sx * w + b - s.sy)];   // = (2·mean(r·x), 2·mean(r))
    function optimum(s) { const det = s.sxx - s.sx * s.sx, w = (s.sxy - s.sx * s.sy) / det; return [w, s.sy - w * s.sx]; }
    /** A = ½·Hessian = [[sxx, sx], [sx, 1]] 的特征分解 → 等高线椭圆、临界学习率 1/a₁、条件数 a₁/a₂ */
    function eigA(s) {
        const a = s.sxx, c = s.sx, d = 1, tr = a + d, det = a * d - c * c, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const a1 = tr / 2 + disc, a2 = tr / 2 - disc;
        let v1 = Math.abs(c) > 1e-12 ? [a1 - d, c] : [1, 0]; const l = Math.hypot(v1[0], v1[1]); v1 = [v1[0] / l, v1[1] / l];
        return { a1, a2, v1, v2: [-v1[1], v1[0]], kappa: a1 / Math.max(a2, 1e-12) };
    }
    /** 全量梯度下降：返回 (w, b) 轨迹与每步损失；发散时后面填 NaN */
    function runGD(s, w, b, lr, steps) {
        const tr = new Float64Array(2 * (steps + 1)).fill(NaN), L = new Float64Array(steps + 1).fill(NaN);
        for (let k = 0; k <= steps; k++) {
            const l = lossQ(s, w, b);
            if (!isFinite(l) || l > 1e8) break;
            tr[2 * k] = w; tr[2 * k + 1] = b; L[k] = l;
            const [gw, gb] = gradQ(s, w, b);
            w -= lr * gw; b -= lr * gb;
        }
        return { tr, L };
    }
    /** mini-batch SGD：每步无放回抽 B 个样本估计梯度；记录的是全数据上的损失 */
    function runSGD(x, y, n, s, w, b, lr, steps, B, rand) {
        const tr = new Float64Array(2 * (steps + 1)).fill(NaN), L = new Float64Array(steps + 1).fill(NaN), idx = Array.from({ length: n }, (_, i) => i);
        B = B <= 0 || B > n ? n : B;
        for (let k = 0; k <= steps; k++) {
            const l = lossQ(s, w, b); if (!isFinite(l) || l > 1e8) break;
            tr[2 * k] = w; tr[2 * k + 1] = b; L[k] = l;
            for (let i = 0; i < B; i++) { const j = i + Math.floor(rand() * (n - i)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }   // 部分洗牌 = 无放回抽样
            let gw = 0, gb = 0;
            for (let i = 0; i < B; i++) { const q = idx[i], r = w * x[q] + b - y[q]; gw += 2 * r * x[q] / B; gb += 2 * r / B; }
            w -= lr * gw; b -= lr * gb;
        }
        return { tr, L };
    }
    /** (A + λI) β = r 的高斯消元（部分主元，双精度） */
    function solve(A, r, m) {
        const M = A.map((row, i) => row.concat([r[i]]));
        for (let c = 0; c < m; c++) {
            let p = c; for (let i = c + 1; i < m; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
            [M[c], M[p]] = [M[p], M[c]];
            const piv = M[c][c] || 1e-300;
            for (let i = c + 1; i < m; i++) { const f = M[i][c] / piv; for (let j = c; j <= m; j++) M[i][j] -= f * M[c][j]; }
        }
        const out = new Array(m).fill(0);
        for (let i = m - 1; i >= 0; i--) { let v = M[i][m]; for (let j = i + 1; j < m; j++) v -= M[i][j] * out[j]; out[i] = v / (M[i][i] || 1e-300); }
        return out;
    }
    /** 多项式岭回归：特征 φⱼ = (x − 1)ʲ（先居中到 [−1, 1]，条件数好得多） */
    function fitPoly(x, y, n, k, lam) {
        const m = k + 1, A = Array.from({ length: m }, () => new Array(m).fill(0)), r = new Array(m).fill(0), phi = new Array(m);
        for (let i = 0; i < n; i++) {
            const t = x[i] - 1; phi[0] = 1; for (let j = 1; j < m; j++) phi[j] = phi[j - 1] * t;
            for (let a = 0; a < m; a++) { r[a] += phi[a] * y[i]; for (let b = 0; b < m; b++) A[a][b] += phi[a] * phi[b]; }
        }
        for (let a = 0; a < m; a++) A[a][a] += lam;
        return solve(A, r, m);
    }
    const polyAt = (beta, x) => { const t = x - 1; let v = 0, p = 1; for (let j = 0; j < beta.length; j++) { v += beta[j] * p; p *= t; } return v; };
    function polyMSE(beta, x, y, n) { let s = 0; for (let i = 0; i < n; i++) s += (polyAt(beta, x[i]) - y[i]) ** 2; return s / n; }

    /** 损失地形的格点（行 0 = b 最大，便于直接当图像画） */
    function landGrid(s, view, out) {
        for (let j = 0; j < GRID; j++) {
            const b = view.b[1] - (j + 0.5) / GRID * (view.b[1] - view.b[0]);
            for (let i = 0; i < GRID; i++) out[j * GRID + i] = lossQ(s, view.w[0] + (i + 0.5) / GRID * (view.w[1] - view.w[0]), b);
        }
        return out;
    }
    /**
     * 画损失地形：log 色阶 + 解析等高线椭圆 + 轨迹 / 点 / 梯度箭头。
     * o: { s, grid, view, opt, paths: [{ tr, upto, color, width, faint }], points: [{ w, b, color, r, label }], arrows, axisLabels }
     */
    function drawLand(c, o) {
        const v = o.view || VIEW, s = o.s;
        PL.draw.field(c, o.grid, GRID, GRID, { map: 'viridis', stretch: (q) => Math.log10(q + 1e-3) });
        const { ctx, w, h } = PL.draw.ctx(c, { keep: true, smooth: true });
        const X = (ww) => (ww - v.w[0]) / (v.w[1] - v.w[0]) * w, Y = (bb) => (v.b[1] - bb) / (v.b[1] - v.b[0]) * h;
        // 等高线：(θ−θ*)ᵀA(θ−θ*) = c 是以特征向量为轴的椭圆
        const e = eigA(s), [ow, ob] = optimum(s), Lmin = lossQ(s, ow, ob);
        ctx.strokeStyle = 'rgba(255,255,255,0.22)'; ctx.lineWidth = 1;
        [0.03, 0.1, 0.3, 1, 3, 10, 30, 100].forEach((lev) => {
            ctx.beginPath();
            for (let k = 0; k <= 96; k++) {
                const t = k / 96 * 2 * Math.PI, r1 = Math.sqrt(lev / e.a1) * Math.cos(t), r2 = Math.sqrt(lev / e.a2) * Math.sin(t);
                const ww = ow + r1 * e.v1[0] + r2 * e.v2[0], bb = ob + r1 * e.v1[1] + r2 * e.v2[1];
                if (k) ctx.lineTo(X(ww), Y(bb)); else ctx.moveTo(X(ww), Y(bb));
            }
            ctx.stroke();
        });
        if (o.arrows) {   // −∇L 方向场（长度做了压缩：只看方向和相对大小）
            ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.fillStyle = 'rgba(255,255,255,0.55)';
            const nA = 11;
            for (let j = 0; j < nA; j++) for (let i = 0; i < nA; i++) {
                const ww = v.w[0] + (i + 0.5) / nA * (v.w[1] - v.w[0]), bb = v.b[0] + (j + 0.5) / nA * (v.b[1] - v.b[0]);
                const [gw, gb] = gradQ(s, ww, bb), gl = Math.hypot(gw, gb) || 1, len = 11 * Math.min(1, 0.35 + 0.65 * Math.log10(1 + gl) / 1.6);
                const dx = -gw / gl * len, dy = gb / gl * len, x0 = X(ww), y0 = Y(bb);   // 屏幕 y 向下：−(−gb)
                ctx.beginPath(); ctx.moveTo(x0 - dx / 2, y0 - dy / 2); ctx.lineTo(x0 + dx / 2, y0 + dy / 2); ctx.stroke();
                const a = Math.atan2(dy, dx); ctx.beginPath(); ctx.moveTo(x0 + dx / 2, y0 + dy / 2); ctx.lineTo(x0 + dx / 2 - 4 * Math.cos(a - 0.5), y0 + dy / 2 - 4 * Math.sin(a - 0.5)); ctx.lineTo(x0 + dx / 2 - 4 * Math.cos(a + 0.5), y0 + dy / 2 - 4 * Math.sin(a + 0.5)); ctx.fill();
            }
        }
        (o.paths || []).forEach((p) => {
            const n = p.upto == null ? p.tr.length / 2 - 1 : p.upto;
            ctx.strokeStyle = p.color || LM.C.yellow; ctx.lineWidth = p.width || 1.6; ctx.globalAlpha = p.faint ? 0.35 : 1; ctx.setLineDash(p.faint ? [3, 3] : []);
            ctx.beginPath(); let pen = false;
            for (let k = 0; k <= n; k++) {
                const ww = p.tr[2 * k], bb = p.tr[2 * k + 1]; if (!isFinite(ww)) break;
                const px = PL.clamp(X(ww), -50, w + 50), py = PL.clamp(Y(bb), -50, h + 50);
                if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
            }
            ctx.stroke(); ctx.setLineDash([]);
            if (p.dots && !p.faint) { ctx.fillStyle = p.color || LM.C.yellow; for (let k = 0; k <= n; k++) { const ww = p.tr[2 * k]; if (!isFinite(ww)) break; ctx.beginPath(); ctx.arc(X(ww), Y(p.tr[2 * k + 1]), 1.8, 0, 7); ctx.fill(); } }
            ctx.globalAlpha = 1;
        });
        // 最优解（正规方程）与真实参数
        ctx.strokeStyle = LM.C.green; ctx.lineWidth = 2; const sx = X(ow), sy = Y(ob);
        ctx.beginPath(); ctx.moveTo(sx - 6, sy); ctx.lineTo(sx + 6, sy); ctx.moveTo(sx, sy - 6); ctx.lineTo(sx, sy + 6); ctx.stroke();
        (o.points || []).forEach((p) => {
            ctx.fillStyle = p.color || LM.C.yellow; ctx.beginPath(); ctx.arc(X(p.w), Y(p.b), p.r || 4, 0, 7); ctx.fill();
            ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
            if (p.label) { ctx.font = LM.FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; const tw0 = ctx.measureText(p.label).width; ctx.fillStyle = 'rgba(5,7,10,0.62)'; ctx.fillRect(X(p.w) + 4, Y(p.b) - 17, tw0 + 4, 15); ctx.fillStyle = p.color || LM.C.yellow; ctx.fillText(p.label, X(p.w) + 6, Y(p.b) - 3); }
        });
        // 坐标标注（深色底，保证在亮色区域也看得清）
        ctx.font = LM.FONT(11);
        const tag = (text, x, y, align, base, color) => {
            ctx.textAlign = align; ctx.textBaseline = base; const tw0 = ctx.measureText(text).width;
            const x0 = align === 'right' ? x - tw0 : align === 'center' ? x - tw0 / 2 : x, y0 = base === 'bottom' ? y - 13 : base === 'middle' ? y - 7 : y;
            ctx.fillStyle = 'rgba(5,7,10,0.62)'; ctx.fillRect(x0 - 2, y0 - 1, tw0 + 4, 15);
            ctx.fillStyle = color || 'rgba(230,230,230,0.85)'; ctx.fillText(text, x, y);
        };
        const tw = (v.w[1] - v.w[0]) > 6 ? 2 : 1, tb = (v.b[1] - v.b[0]) > 10 ? 5 : (v.b[1] - v.b[0]) > 6 ? 2 : 1;
        for (let t = Math.ceil(v.w[0] / tw) * tw; t <= v.w[1]; t += tw) if (X(t) > 14 && X(t) < w - 40) tag(String(+t.toFixed(2)), X(t), h - 2, 'center', 'bottom');
        for (let t = Math.ceil(v.b[0] / tb) * tb; t <= v.b[1]; t += tb) if (Y(t) > 24 && Y(t) < h - 22) tag(String(+t.toFixed(2)), 3, Y(t), 'left', 'middle');
        tag(o.wLabel || 'w →', w - 4, h - 2, 'right', 'bottom', LM.C.text);
        tag(o.bLabel || '↑ b', 4, 3, 'left', 'top', LM.C.text);
        if (o.caption) tag(o.caption, w - 4, 3, 'right', 'top', LM.C.yellow);
        c._p01 = { X, Y, view: v };
        return { X, Y, Lmin };
    }
    /** 数据空间：散点 + 当前直线 + 残差 + 真实直线（虚线） */
    function drawFit(c, S, w, b, o) {
        o = o || {};
        const pts = [], lines = [];
        for (let i = 0; i < S.n; i++) {
            const yh = w * S.x[i] + b;
            if (o.residuals) lines.push({ a: [S.x[i], S.y[i]], b: [S.x[i], yh], color: 'rgba(224,108,117,0.7)', width: 1 });
            pts.push({ x: S.x[i], y: S.y[i], color: LM.C.blue, r: S.n > 100 ? 2 : 3 });
        }
        if (o.truth !== false) lines.push({ a: [-0.2, B_TRUE - 0.2 * W_TRUE], b: [2.2, B_TRUE + 2.2 * W_TRUE], color: LM.C.gray, dash: [4, 4], width: 1 });
        if (isFinite(w) && isFinite(b)) lines.push({ a: [-0.2, b - 0.2 * w], b: [2.2, b + 2.2 * w], color: o.color || LM.C.yellow, width: 2 });
        (o.extra || []).forEach((l) => lines.push(l));
        return LM.draw.scatter(c, pts, { x: [-0.1, 2.1], y: [-2.5, 7.5], lines, xLabel: o.xLabel || 'x（特征）', yLabel: 'y（目标）', paths: o.paths, title: o.title });
    }
    const playhead = (S) => { const cyc = K + 2 * RATE; return Math.min(K, Math.floor((S.t * RATE) % cyc)); };
    return { W_TRUE, B_TRUE, NMAX, NTEST, K, VIEW, GRID, DEG_MAX, stats, lossQ, gradQ, optimum, eigA, runGD, runSGD, fitPoly, polyAt, polyMSE, landGrid, drawLand, drawFit, playhead };
})();

PipelineLab.mount({
    paramSteps: { n: ['data', 'model', 'loss', 'grad', 'gd', 'lr', 'scale', 'sgd', 'scaleup'], noise: ['data', 'model', 'loss', 'grad', 'gd', 'lr', 'scale', 'sgd', 'generalize', 'scaleup'], w0: ['model', 'loss', 'grad', 'gd', 'lr', 'scale', 'sgd', 'scaleup'], b0: ['model', 'loss', 'grad', 'gd', 'lr', 'scale', 'sgd', 'scaleup'], lr: ['grad', 'gd', 'lr', 'sgd', 'scaleup'], batch: ['sgd'], offset: ['scale'], degree: ['generalize'], npoly: ['generalize'], lam: ['generalize'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '01 · Hello World：线性回归与梯度下降',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 20251123,
    css: LM.CSS,
    regen: { label: '重新采样数据', title: '重新抽取 x 和噪声 ε（只重算 RESIDENT）' },

    params: [
        { id: 'n', label: '样本数 n', min: 8, max: 200, step: 1, value: 40, desc: '训练样本个数。样本越多，噪声被平均得越干净：最优解 ✚ 越接近真实参数 (2, 1)；SGD 的「全量」批也随之变大。' },
        { id: 'noise', label: '噪声 σ', min: 0, max: 1.5, step: 0.05, value: 0.5, desc: '数据里不可预测的部分 ε ~ N(0, σ²)。它决定了损失的下限 ≈ σ²：再好的模型也压不到这以下（Step 03 的谷底高度、Step 09 的测试误差下限）。' },
        { id: 'w0', label: '初始 w₀', min: -2, max: 6, step: 0.1, value: -1, desc: '梯度下降的起点（斜率）。也可以直接在 Step 03 的损失地形上点击设置起点。' },
        { id: 'b0', label: '初始 b₀', min: -3, max: 5, step: 0.1, value: 4, desc: '梯度下降的起点（截距）。和 w₀ 一起决定轨迹从地形的哪里出发。' },
        { id: 'lr', label: '学习率 η', min: 0.01, max: 0.6, step: 0.01, value: 0.2, desc: '每一步沿负梯度走多远。太小 → 爬得慢；超过临界值 1/a₁（Step 06 的红线）→ 来回震荡直至发散。' },
        { id: 'batch', label: '批大小 B', options: [1, 4, 16, { value: 0, label: '全量' }], value: 4, desc: 'SGD 每步用多少个样本估计梯度。B 越小，每步越便宜、轨迹越抖（Step 08）；「全量」就是普通梯度下降。' },
        { id: 'offset', label: '特征偏移 c', min: 0, max: 6, step: 0.5, value: 3, desc: 'Step 07 里把特征换成 x + c（例如没减均值的原始读数）。c 越大，损失地形越像一条细长的峡谷，条件数 κ 越大，GD 越慢。' },
        { id: 'degree', label: '多项式次数', min: 0, max: 12, step: 1, value: 9, desc: 'Step 09 的模型容量：用 1, x, …, x^k 拟合。次数太低欠拟合，太高会把噪声也背下来（训练误差↓、测试误差↑）。' },
        { id: 'npoly', label: '拟合样本数', min: 6, max: 60, step: 1, value: 12, desc: 'Step 09 用来拟合多项式的样本数（取前若干个训练样本）。样本越多，高次多项式越难过拟合。' },
        { id: 'lam', label: 'log₁₀ λ', min: -9, max: 1, step: 0.5, value: -9, map: (v) => Math.pow(10, v), desc: 'Step 09 的岭回归（L2 正则）强度 λ。把它从 −9 拖到 −2 左右，高次多项式的曲线会立刻变平滑，测试误差下降。' },
    ],

    stages: [
        {
            id: 'sample', lifetime: 'resident', run(S) {
                const u = PL.buf(S, 'u', P01.NMAX, Float64Array), z = PL.buf(S, 'z', P01.NMAX, Float64Array);
                const tu = PL.buf(S, 'tu', P01.NTEST, Float64Array), tz = PL.buf(S, 'tz', P01.NTEST, Float64Array);
                for (let i = 0; i < P01.NMAX; i++) { u[i] = 2 * S.rand(); z[i] = PL.gauss(S.rand); }
                for (let i = 0; i < P01.NTEST; i++) { tu[i] = 2 * S.rand(); tz[i] = PL.gauss(S.rand); }
            },
        },
        {
            id: 'dataset', lifetime: 'param', run(S) {
                const n = S.n, x = PL.buf(S, 'x', n, Float64Array), y = PL.buf(S, 'y', n, Float64Array);
                for (let i = 0; i < n; i++) { x[i] = S.u[i]; y[i] = P01.W_TRUE * x[i] + P01.B_TRUE + S.noise * S.z[i]; }
                const xt = PL.buf(S, 'xt', P01.NTEST, Float64Array), yt = PL.buf(S, 'yt', P01.NTEST, Float64Array);
                for (let i = 0; i < P01.NTEST; i++) { xt[i] = S.tu[i]; yt[i] = P01.W_TRUE * xt[i] + P01.B_TRUE + S.noise * S.tz[i]; }
                S.st = P01.stats(x, y, n); S.opt = P01.optimum(S.st); S.eig = P01.eigA(S.st);
                S.Lmin = P01.lossQ(S.st, S.opt[0], S.opt[1]); S.lrCrit = 1 / S.eig.a1;
                S.land = P01.landGrid(S.st, P01.VIEW, PL.buf(S, 'land', P01.GRID * P01.GRID));
            },
        },
        {
            id: 'gd', lifetime: 'param', run(S) {
                S.gd = P01.runGD(S.st, S.w0, S.b0, S.lr, P01.K);
                // 学习率扫描：临界值 η_c = 1/a₁ 的若干倍
                S.lrSet = [0.08, 0.45, 0.95, 1.04].map((r) => r * S.lrCrit);
                S.lrRuns = S.lrSet.map((lr) => P01.runGD(S.st, S.w0, S.b0, lr, P01.K));
            },
        },
        {
            id: 'scaling', lifetime: 'param', run(S) {
                // 原始特征 x' = x + c：参数 (w, b' = b − w·c)；标准化特征 z = (x − μ)/σ：参数 (w·σ, b + w·μ)
                const n = S.n, c = S.offset, xr = PL.buf(S, 'xRaw', n, Float64Array), xz = PL.buf(S, 'xStd', n, Float64Array);
                let mu = 0; for (let i = 0; i < n; i++) mu += S.x[i] / n;
                let sd = 0; for (let i = 0; i < n; i++) sd += (S.x[i] - mu) ** 2 / n; sd = Math.sqrt(sd) || 1;
                for (let i = 0; i < n; i++) { xr[i] = S.x[i] + c; xz[i] = (S.x[i] - mu) / sd; }
                const sr = P01.stats(xr, S.y, n), sz = P01.stats(xz, S.y, n), er = P01.eigA(sr), ez = P01.eigA(sz);
                const optR = P01.optimum(sr), optZ = P01.optimum(sz);
                const vR = { w: [optR[0] - 4, optR[0] + 4], b: [optR[1] - 4 - 4 * c, optR[1] + 4 + 4 * c] }, vZ = { w: [optZ[0] - 3, optZ[0] + 3], b: [optZ[1] - 3, optZ[1] + 3] };
                const lrR = 1 / (er.a1 + er.a2), lrZ = 1 / (ez.a1 + ez.a2);   // 各自的最优固定步长 2/(λmax+λmin)
                S.sc = {
                    mu, sd, c, sr, sz, er, ez, vR, vZ, lrR, lrZ,
                    runR: P01.runGD(sr, S.w0, S.b0 - S.w0 * c, lrR, P01.K),
                    runZ: P01.runGD(sz, S.w0 * sd, S.b0 + S.w0 * mu, lrZ, P01.K),
                    landR: P01.landGrid(sr, vR, PL.buf(S, 'landR', P01.GRID * P01.GRID)),
                    landZ: P01.landGrid(sz, vZ, PL.buf(S, 'landZ', P01.GRID * P01.GRID)),
                };
            },
        },
        {
            id: 'sgd', lifetime: 'param', run(S) {
                const sizes = [1, 4, 16, 0];
                S.sgdRuns = sizes.map((B, k) => P01.runSGD(S.x, S.y, S.n, S.st, S.w0, S.b0, S.lr, P01.K, B, PL.rng((S.seed ^ 0x51ed) + 977 * k)));
                S.sgdSizes = sizes;
            },
        },
        {
            id: 'poly', lifetime: 'param', run(S) {
                const n = Math.min(S.npoly, P01.NMAX), xs = PL.buf(S, 'xp', n, Float64Array), ys = PL.buf(S, 'yp', n, Float64Array);
                for (let i = 0; i < n; i++) { xs[i] = S.u[i]; ys[i] = P01.W_TRUE * xs[i] + P01.B_TRUE + S.noise * S.z[i]; }
                S.polyN = n;
                S.polyTrain = []; S.polyTest = []; S.polyFits = [];
                for (let k = 0; k <= P01.DEG_MAX; k++) {
                    const beta = P01.fitPoly(xs, ys, n, k, S.lam);
                    S.polyFits.push(beta);
                    S.polyTrain.push(P01.polyMSE(beta, xs, ys, n));
                    S.polyTest.push(P01.polyMSE(beta, S.xt, S.yt, P01.NTEST));
                }
            },
        },
        {
            id: 'autograd', lifetime: 'param', run(S) {
                // 同一个梯度下降，换成 LM.AG 自动微分（float32）来算梯度 —— 应与手写梯度（float64）逐步吻合
                const { AG } = LM, n = S.n, K = P01.K;
                const X = AG.tensor(n, 1, Float32Array.from(S.x)), Y = AG.tensor(n, 1, Float32Array.from(S.y));
                const W = AG.param('w', 1, 1), Bp = AG.param('b', 1, 1); W.data[0] = S.w0; Bp.data[0] = S.b0;
                const tr = new Float64Array(2 * (K + 1)).fill(NaN), L = new Float64Array(K + 1).fill(NaN);
                for (let k = 0; k <= K; k++) {
                    W.grad = null; Bp.grad = null;
                    const loss = AG.mse(AG.addRow(AG.matmul(X, W), Bp), Y);   // ŷ = X·w + b
                    const l = loss.item(); if (!isFinite(l) || l > 1e8) { AG.clearTape(); break; }
                    tr[2 * k] = W.data[0]; tr[2 * k + 1] = Bp.data[0]; L[k] = l;
                    AG.backward(loss);                                         // 反向传播得到 ∂L/∂w、∂L/∂b
                    W.data[0] -= S.lr * W.grad[0]; Bp.data[0] -= S.lr * Bp.grad[0];
                }
                let md = 0; for (let k = 0; k <= K; k++) if (isFinite(tr[2 * k]) && isFinite(S.gd.tr[2 * k])) md = Math.max(md, Math.abs(tr[2 * k] - S.gd.tr[2 * k]), Math.abs(tr[2 * k + 1] - S.gd.tr[2 * k + 1]));
                S.ag = { tr, L, maxDiff: md };
            },
        },
        { id: 'play', lifetime: 'frame', run(S) { S.it = P01.playhead(S); } },
    ],

    intro: {
        title: '机器学习的 Hello World：让机器自己找到 y = w·x + b',
        flow: ['数据 (x, y)', '模型 ŷ = w·x + b', '损失 MSE', '梯度 ∇L', '更新 w ← w − η∇L', '重复 → 收敛', '泛化到新数据'],
        html: LM.navHtml('01') + `
            <p>这一篇只有两个参数 <code>w</code>、<code>b</code>，但它已经包含了训练大模型的全部骨架：<strong>一个带参数的函数</strong>（模型）、<strong>一个衡量错误的数</strong>（损失）、<strong>损失对参数的导数</strong>（梯度）、<strong>沿负梯度反复走小步</strong>（优化器）。
            第 05 篇训练 GPT 时，这四样东西一个不少，只是参数从 2 个变成了几万（真实模型是几十亿到上万亿）、损失从均方误差换成交叉熵、优化器从 GD 换成 AdamW。</p>
            <p>页面里所有预览都由内嵌的 JS 参考实现<strong>实时算出</strong>：改任何参数，数据、损失地形、梯度下降轨迹都会立即重算；带 <span style="color:#a45fa5">LIVE</span> 的图在播放梯度下降的过程（每秒 12 步，可在顶栏暂停 / 调速）。代码块是同一算法的 C++ 版本（C++17 + 标准库，不用第三方库）：前 9 步是手写的 <code>std::vector&lt;double&gt;</code> 循环，最后一步换成本系列自带的迷你自动微分库 <code>pages/cpp/minml_*.h</code>（源码收在页面上方「训练循环 · 数据流」一节的 Common 里）。悬停代码里的函数名可以看定义，点击跳过去。</p>`,
    },

    steps: [
        {
            id: 'data', title: '任务与数据：从例子里学规律', tag: 'DATA',
            desc: '监督学习的输入是一堆<strong>（特征 x，目标 y）样本对</strong>。这里用一个我们知道答案的玩具问题：真实规律是 <code>y = 2x + 1</code>，再叠加不可预测的噪声 ε。可以把 x 想成「房屋面积（百平米）」、y 想成「总价（百万）」。机器只看得到散点，看不到虚线——它要做的就是从散点里把这条线「猜」回来。',
            formula: ['x_i ~ U(0, 2)      ε_i ~ N(0, σ^2)', 'y_i = 2 · x_i + 1 + ε_i   // 真实参数 w* = 2, b* = 1（模型不知道）'],
            code: {
                title: 'C++ · makeData', src: `
                // 训练集：n 个 (x_i, y_i) 样本对
                struct Dataset { std::vector<double> x, y; };

                // x_i ~ U(0, 2)，y_i = 2·x_i + 1 + σ·ε_i，ε_i ~ N(0, 1)：真实参数 w* = 2、b* = 1，模型并不知道
                Dataset makeData(int n, double noise, uint32_t seed) {
                    Rng rng = makeRng(seed);                              // 可复现的伪随机数（cpp/minml_core.h）
                    Dataset d;
                    d.x = doubles(n);
                    d.y = doubles(n);
                    for (int i = 0; i < n; i++) {
                        d.x[i] = 2.0 * uniform(rng);                      // uniform(rng) ∈ [0, 1)
                        d.y[i] = 2.0 * d.x[i] + 1.0 + noise * randn(rng); // 真实规律 + 噪声
                    }
                    return d;
                }`,
            },
            previews: [{
                id: 'scatter', label: '训练数据', sub: '蓝点 = 样本 · 灰虚线 = 真实规律 y = 2x + 1', large: true, size: [360, 270],
                render: (c, S) => P01.drawFit(c, S, NaN, NaN, {}),
            }],
            notes: '拖 <b>噪声 σ</b> 到 0：散点精确落在虚线上；拖到 1.5：肉眼都很难看出规律。拖 <b>样本数 n</b> 可以看到「数据越多、规律越清楚」。点顶栏「重新采样数据」换一批随机样本。',
        },
        {
            id: 'model', title: '模型：一个带参数的函数 ŷ = w·x + b', tag: 'MODEL',
            desc: '「模型」就是一个<strong>形式固定、参数待定</strong>的函数。线性回归的模型是一条直线，参数只有斜率 w 和截距 b：给定参数，对每个 x 都能算出预测 ŷ。黄线是当前参数（w₀, b₀）画出的直线，红色竖线是每个样本的<strong>残差</strong> r = ŷ − y——学习的目标就是让这些红线整体变短。',
            formula: ['ŷ_i = w · x_i + b', 'r_i = ŷ_i − y_i   // 残差：预测减真实'],
            code: {
                title: 'C++ · predict', src: `
                // ŷ_i = w·x_i + b：一个循环算出所有样本的预测
                std::vector<double> predict(const std::vector<double>& x, double w, double b) {
                    std::vector<double> yhat = doubles((int)x.size());
                    for (size_t i = 0; i < x.size(); i++) yhat[i] = w * x[i] + b;
                    return yhat;
                }`,
            },
            previews: [{
                id: 'fit0', label: '初始模型与残差', sub: '黄 = ŷ = w₀x + b₀ · 红 = 残差', large: true, size: [360, 270],
                render: (c, S) => { S._tfFit0 = P01.drawFit(c, S, S.w0, S.b0, { residuals: true }); },
                probe: (x, y, S, PL, info) => { if (!S._tfFit0) return ''; const t = S._tfFit0.inv(info.dx, info.dy); return `x = ${t[0].toFixed(2)}  ŷ = w₀x + b₀ = ${(S.w0 * t[0] + S.b0).toFixed(2)}`; },
            }],
            notes: '拖 <b>初始 w₀ / b₀</b> 手动「拟合」：你会本能地让红线变短——下一步把「变短」写成一个数。',
        },
        {
            id: 'loss', title: '损失函数：均方误差 MSE 与损失地形', tag: 'LOSS',
            desc: '把所有残差平方后取平均，得到一个数 L(w, b)：<strong>越小越好</strong>。平方让正负误差不抵消、让大误差受到更重的惩罚，而且处处可导。把每一组 (w, b) 的损失画出来就是右边的<strong>损失地形</strong>：横轴 w、纵轴 b、颜色 = log L。线性回归的 MSE 是 (w, b) 的二次函数，所以地形是一个碗，等高线是椭圆；碗底 ✚ 就是最优参数，可以用<strong>正规方程</strong>直接解出来。',
            formula: [
                'L(w, b) = (1/n) Σ_i (w · x_i + b − y_i)^2',
                '∂L/∂w = 0, ∂L/∂b = 0  ⇒  (X^T X) θ* = X^T y   // 正规方程，X = [x, 1]',
            ],
            code: {
                title: 'C++ · mseLoss / leastSquares', src: `
                // L(w, b) = (1/n)·Σ (ŷ_i − y_i)²
                double mseLoss(const std::vector<double>& x, const std::vector<double>& y, double w, double b) {
                    std::vector<double> yhat = predict(x, w, b);
                    double sum = 0;
                    for (size_t i = 0; i < x.size(); i++) {
                        double r = yhat[i] - y[i];                        // 残差
                        sum += r * r;
                    }
                    return sum / x.size();
                }

                // 闭式解：碗底满足梯度 = 0，即正规方程 (XᵀX)·θ = Xᵀy，X 的第 i 行是 [x_i, 1]
                Line leastSquares(const std::vector<double>& x, const std::vector<double>& y) {
                    int n = (int)x.size();
                    double sx = 0, sxx = 0, sy = 0, sxy = 0;
                    for (int i = 0; i < n; i++) { sx += x[i]; sxx += x[i] * x[i]; sy += y[i]; sxy += x[i] * y[i]; }
                    sx /= n; sxx /= n; sy /= n; sxy /= n;                 // 方程两边同除以 n：全部写成均值
                    // 2×2 方程组 [[mean(x²), mean(x)], [mean(x), 1]]·[w, b]ᵀ = [mean(x·y), mean(y)]ᵀ，消元直接解出
                    double det = sxx - sx * sx;                           // = Var(x)；所有 x 都相同时为 0，直线不唯一
                    Line opt;
                    opt.w = (sxy - sx * sy) / det;                        // = Cov(x, y) / Var(x)
                    opt.b = sy - opt.w * sx;                              // 最优直线一定穿过 (mean(x), mean(y))
                    return opt;
                }`,
            },
            previews: [
                {
                    id: 'land', label: '损失地形 L(w, b)', sub: '颜色 = log₁₀L · 白线 = 等高线 · ✚ = 碗底 · 黄点 = 当前 (w₀, b₀) · <b>点击设置起点</b>', large: true,
                    render: (c, S) => P01.drawLand(c, { s: S.st, grid: S.land, points: [{ w: S.w0, b: S.b0, label: `L = ${P01.lossQ(S.st, S.w0, S.b0).toFixed(2)}` }, { w: P01.W_TRUE, b: P01.B_TRUE, color: LM.C.gray, r: 3 }] }),
                    probe: (x, y, S, PL, info) => { const v = P01.VIEW, w = v.w[0] + info.u * (v.w[1] - v.w[0]), b = v.b[1] - info.v * (v.b[1] - v.b[0]); return `w = ${w.toFixed(2)}, b = ${b.toFixed(2)}\nL = ${P01.lossQ(S.st, w, b).toFixed(3)}`; },
                    pick: (x, y, S, PL, info) => { const v = P01.VIEW; PL.setParam('w0', +(v.w[0] + info.u * (v.w[1] - v.w[0])).toFixed(1)); PL.setParam('b0', +(v.b[1] - info.v * (v.b[1] - v.b[0])).toFixed(1)); },
                },
                {
                    id: 'slice', label: '沿 w 的剖面 L(w, b₀)', sub: '固定 b = b₀ 时损失随 w 是一条抛物线 · 绿虚线 = 全局最小 L*', size: [360, 200],
                    render: (c, S) => {
                        const wm = (S.st.sxy - S.st.sx * S.b0) / S.st.sxx, top = Math.max(P01.lossQ(S.st, S.w0, S.b0), P01.lossQ(S.st, wm, S.b0)) * 2.2 + 0.5;
                        PL.draw.plot(c, {
                            x: P01.VIEW.w, y: [0, top], xLabel: 'w', series: [{ fn: (w) => P01.lossQ(S.st, w, S.b0), color: LM.C.blue, label: 'L(w, b₀)' }],
                            marks: [{ x: S.w0, label: 'w₀', color: LM.C.yellow }, { x: wm, label: '剖面最低点', color: LM.C.green }],
                            hmarks: [{ y: S.Lmin, color: LM.C.green }],
                        });
                    },
                },
            ],
            notes: '碗底的高度 L* ≈ σ²（噪声方差）：噪声是模型永远学不到的部分。✚ 与灰点（真实参数 2, 1）不重合——有限样本里的噪声会把最优解拉偏；增大 <b>样本数 n</b> 它们会靠拢。',
            warn: '正规方程只适用于这种「损失是二次函数」的特例。神经网络的损失地形不是碗，没有闭式解，只能用下一步的梯度方法一步步往下走。',
        },
        {
            id: 'grad', title: '梯度：地形上最陡的上坡方向', tag: 'GRAD',
            desc: '梯度 ∇L = (∂L/∂w, ∂L/∂b) 指向<strong>损失增长最快</strong>的方向，长度表示坡有多陡；它的反方向 −∇L 就是「最快的下坡路」。箭头场画的就是每个位置的 −∇L：所有箭头都垂直穿过等高线、指向碗底。每个样本对梯度的贡献是 <code>2·r_i·x_i</code>——残差越大的样本「拉」得越用力（右下图）。',
            formula: [
                '∂L/∂w = (2/n) Σ_i r_i · x_i      ∂L/∂b = (2/n) Σ_i r_i   // r_i = w·x_i + b − y_i',
                '数值校验：∂L/∂w ≈ [L(w+ε, b) − L(w−ε, b)] / 2ε   // 中心差分，误差 O(ε²)',
            ],
            code: {
                title: 'C++ · gradients / numericGrad', src: `
                // ∂L/∂w = (2/n)·Σ r_i·x_i，∂L/∂b = (2/n)·Σ r_i，其中 r_i = ŷ_i − y_i
                Grad gradients(const std::vector<double>& x, const std::vector<double>& y, double w, double b) {
                    std::vector<double> yhat = predict(x, w, b);
                    int n = (int)x.size();
                    Grad g;
                    for (int i = 0; i < n; i++) {
                        double r = yhat[i] - y[i];
                        g.dw += r * x[i];                                 // 链式法则：∂(r²)/∂w = 2r · ∂r/∂w = 2r · x
                        g.db += r;                                        //           ∂(r²)/∂b = 2r · ∂r/∂b = 2r
                    }
                    g.dw *= 2.0 / n;                                      // 2·mean(r·x)
                    g.db *= 2.0 / n;                                      // 2·mean(r)
                    return g;
                }

                // 写完解析梯度后一定要做的检查：和中心差分比一比（误差 O(ε²)）
                Grad numericGrad(const std::vector<double>& x, const std::vector<double>& y, double w, double b, double eps = 1e-4) {
                    Grad g;
                    g.dw = (mseLoss(x, y, w + eps, b) - mseLoss(x, y, w - eps, b)) / (2 * eps);
                    g.db = (mseLoss(x, y, w, b + eps) - mseLoss(x, y, w, b - eps)) / (2 * eps);
                    return g;
                }`,
            },
            previews: [
                {
                    id: 'gradfield', label: '−∇L 方向场', sub: '箭头 = 下坡方向（长度已压缩）· 垂直于等高线', large: true,
                    render: (c, S) => P01.drawLand(c, { s: S.st, grid: S.land, arrows: true, points: [{ w: S.w0, b: S.b0 }] }),
                    probe: (x, y, S, PL, info) => { const v = P01.VIEW, w = v.w[0] + info.u * (v.w[1] - v.w[0]), b = v.b[1] - info.v * (v.b[1] - v.b[0]), g = P01.gradQ(S.st, w, b); return `(w, b) = (${w.toFixed(2)}, ${b.toFixed(2)})\n∇L = (${g[0].toFixed(2)}, ${g[1].toFixed(2)})`; },
                    pick: (x, y, S, PL, info) => { const v = P01.VIEW; PL.setParam('w0', +(v.w[0] + info.u * (v.w[1] - v.w[0])).toFixed(1)); PL.setParam('b0', +(v.b[1] - info.v * (v.b[1] - v.b[0])).toFixed(1)); },
                },
                {
                    id: 'gradcheck', label: '解析梯度 vs 数值梯度', sub: '在 (w₀, b₀) 处', size: [360, 150],
                    render: (c, S) => {
                        const [gw, gb] = P01.gradQ(S.st, S.w0, S.b0), e = 1e-4;
                        const nw = (P01.lossQ(S.st, S.w0 + e, S.b0) - P01.lossQ(S.st, S.w0 - e, S.b0)) / (2 * e), nb = (P01.lossQ(S.st, S.w0, S.b0 + e) - P01.lossQ(S.st, S.w0, S.b0 - e)) / (2 * e);
                        LM.draw.text(c, [
                            [{ t: '解析  ', color: LM.C.dim }, { t: `∂L/∂w = ${gw.toFixed(6)}   ∂L/∂b = ${gb.toFixed(6)}`, color: LM.C.blue }],
                            [{ t: '数值  ', color: LM.C.dim }, { t: `∂L/∂w = ${nw.toFixed(6)}   ∂L/∂b = ${nb.toFixed(6)}`, color: LM.C.orange }],
                            [{ t: '相对误差 ', color: LM.C.dim }, { t: `${(Math.abs(gw - nw) / (Math.abs(gw) + 1e-12)).toExponential(1)} , ${(Math.abs(gb - nb) / (Math.abs(gb) + 1e-12)).toExponential(1)}`, color: LM.C.green }],
                            { text: `‖∇L‖ = ${Math.hypot(gw, gb).toFixed(3)}：一步 η = ${S.lr} 会让 (w, b) 移动 ${(S.lr * Math.hypot(gw, gb)).toFixed(3)}`, color: LM.C.gray },
                        ], { px: 12 });
                    },
                },
                {
                    id: 'pull', label: '每个样本对 ∂L/∂w 的贡献 2·rᵢ·xᵢ/n', sub: '按 x 排序 · 和 = ∂L/∂w', size: [360, 180],
                    render: (c, S) => {
                        const n = S.n, idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => S.x[a] - S.x[b]);
                        const v = idx.map((i) => 2 * (S.w0 * S.x[i] + S.b0 - S.y[i]) * S.x[i] / n);
                        LM.draw.bars(c, v, null, { colors: v.map((q) => (q > 0 ? LM.C.red : LM.C.blue)), digits: false });
                    },
                },
            ],
            notes: '点击方向场任意位置，起点移过去，左下角的数值校验会跟着重算——两种算法给出的梯度应当一致到 1e-8 量级。<b>梯度校验</b>是写反向传播时最重要的调试手段（第 02 篇会对一个 MLP 再做一次）。',
        },
        {
            id: 'gd', title: '梯度下降：沿 −∇L 反复走一小步', tag: 'OPTIM',
            desc: '从起点出发，每一步算一次梯度、沿反方向挪 η 倍：<code>θ ← θ − η∇L(θ)</code>。损失地形上的黄线就是这 80 步的真实轨迹，左下是损失随步数下降的曲线（对数纵轴），右下是对应的直线在数据上如何一步步「转」到位。这四行代码就是所有神经网络训练循环的原型。',
            formula: ['w_{t+1} = w_t − η · ∂L/∂w(w_t, b_t)', 'b_{t+1} = b_t − η · ∂L/∂b(w_t, b_t)   // η = 学习率（步长）'],
            code: {
                title: 'C++ · gradientDescent', src: `
                // 全量梯度下降：从 (w, b) 出发走 steps 步。history[t] = 第 t 步的 (w, b, L)，最终参数就是 history.back()
                std::vector<Snapshot> gradientDescent(const std::vector<double>& x, const std::vector<double>& y,
                                                      double w, double b, double lr, int steps) {
                    std::vector<Snapshot> history;
                    history.push_back({w, b, mseLoss(x, y, w, b)});
                    for (int t = 0; t < steps; t++) {
                        Grad g = gradients(x, y, w, b);                   // 1. 算梯度
                        w -= lr * g.dw;                                   // 2. 沿负梯度走一步
                        b -= lr * g.db;
                        double L = mseLoss(x, y, w, b);
                        if (!std::isfinite(L) || L > 1e8) break;          // 学习率太大、已经发散（Step 06）：不再往下记
                        history.push_back({w, b, L});
                    }
                    return history;
                }`,
            },
            previews: [
                {
                    id: 'gdtraj', live: true, label: 'GD 轨迹', sub: '虚线 = 完整 80 步 · 实线 = 播放到当前步 · 黄点 = 当前参数', large: true,
                    render: (c, S) => { const k = S.it; P01.drawLand(c, { s: S.st, grid: S.land, paths: [{ tr: S.gd.tr, faint: true }, { tr: S.gd.tr, upto: k, dots: true }], points: [{ w: S.gd.tr[2 * k], b: S.gd.tr[2 * k + 1], label: `t = ${k}` }], caption: `η = ${S.lr}` }); },
                },
                {
                    id: 'gdloss', live: true, label: '损失 L(t)', sub: '对数纵轴 · 绿线 = 碗底 L*', size: [360, 200],
                    render: (c, S) => {
                        const L = Array.from(S.gd.L), k = S.it;
                        PL.draw.plot(c, { x: [0, P01.K], logY: true, xLabel: '迭代步 t', series: [{ data: L, color: '#3a5060' }, { data: L.slice(0, k + 1), x: L.slice(0, k + 1).map((_, i) => i), color: LM.C.blue, width: 2, label: 'GD' }], marks: [{ x: k, label: `L = ${LM.fmt(L[k])}` }], hmarks: [{ y: S.Lmin, color: LM.C.green }] });
                    },
                },
                {
                    id: 'gdfit', live: true, label: '数据空间里的同一过程', sub: '直线 ŷ = w_t·x + b_t', size: [360, 230],
                    render: (c, S) => { const k = S.it; P01.drawFit(c, S, S.gd.tr[2 * k], S.gd.tr[2 * k + 1], { residuals: true }); },
                },
            ],
            notes: '轨迹先沿陡峭方向快速落入「谷底」，再沿平缓的谷底慢慢爬向 ✚——两个方向的坡度差得越多，后半程越慢（Step 07 的条件数）。随时可以暂停：任何时刻的黄点都在 80 步的真实轨迹上。',
        },
        {
            id: 'lr', title: '学习率：太小爬不动，太大会发散', tag: 'OPTIM',
            desc: '同一个起点、四个学习率。对二次碗，GD 沿每个特征方向的误差每步乘以 <code>(1 − 2η·a)</code>（a 是该方向的曲率）：只要 <strong>η &gt; 1/a₁</strong>（a₁ = 最陡方向的曲率），这个因子的绝对值就超过 1，误差每步被放大，轨迹在峡谷两壁之间越弹越远直到发散。η 在 1/(2a₁) 到 1/a₁ 之间时会出现收敛但来回「之」字形的震荡。',
            formula: [
                'θ_{t+1} − θ* = (I − 2η A) (θ_t − θ*)      A = [[mean(x^2), mean(x)], [mean(x), 1]]',
                '收敛 ⇔ |1 − 2η a_i| < 1 对所有特征值 a_i 成立 ⇔ η < η_c = 1 / a_1',
            ],
            code: {
                title: 'C++ · eigA / criticalLR', src: `
                // 2×2 对称矩阵的两个特征值，a1 ≥ a2
                struct Eig2 { double a1 = 0, a2 = 0; };

                // A = ½·Hessian = [[mean(x²), mean(x)], [mean(x), 1]] 的特征值 = 等高线椭圆两个主轴方向上的曲率
                Eig2 eigA(const std::vector<double>& x) {
                    int n = (int)x.size();
                    double sx = 0, sxx = 0;
                    for (int i = 0; i < n; i++) { sx += x[i]; sxx += x[i] * x[i]; }
                    sx /= n; sxx /= n;
                    // [[a, c], [c, d]] 的特征值有闭式解：tr/2 ± √(tr²/4 − det)
                    double a = sxx, c = sx, d = 1.0, tr = a + d, det = a * d - c * c;
                    double disc = std::sqrt(std::max(0.0, tr * tr / 4 - det));
                    Eig2 e;
                    e.a1 = tr / 2 + disc;
                    e.a2 = tr / 2 - disc;
                    return e;
                }

                // MSE 的 Hessian = 2A；GD 稳定的上限是 2/λmax(Hessian) = 1/λmax(A) = 1/a1
                double criticalLR(const std::vector<double>& x) {
                    return 1.0 / eigA(x).a1;
                }`,
            },
            previews: [
                {
                    id: 'lrcurves', label: '四个学习率的损失曲线', sub: '以临界值 η_c = 1/a₁ 为单位 · 黄 = 顶栏当前 η', large: true, size: [360, 240],
                    render: (c, S) => {
                        const cols = [LM.C.gray, LM.C.blue, LM.C.purple, LM.C.red];
                        const series = S.lrRuns.map((r, k) => ({ data: Array.from(r.L), color: cols[k], label: `η = ${S.lrSet[k].toFixed(3)} (${[0.08, 0.45, 0.95, 1.04][k]}η_c)` }));
                        series.push({ data: Array.from(S.gd.L), color: LM.C.yellow, width: 2.5, label: `当前 η = ${S.lr}` });
                        PL.draw.plot(c, { x: [0, P01.K], y: [Math.max(1e-4, S.Lmin * 0.5), 1e4], logY: true, xLabel: '迭代步 t', series, hmarks: [{ y: S.Lmin, color: LM.C.green, label: 'L*' }] });
                    },
                },
                {
                    id: 'lrtraj', label: '轨迹对比', sub: '同色 = 同一学习率；红色（1.04η_c）正在发散', large: true,
                    render: (c, S) => {
                        const cols = [LM.C.gray, LM.C.blue, LM.C.purple, LM.C.red];
                        P01.drawLand(c, { s: S.st, grid: S.land, paths: S.lrRuns.map((r, k) => ({ tr: r.tr, color: cols[k], width: 1.3, upto: k === 3 ? 30 : null })), points: [{ w: S.w0, b: S.b0 }], caption: `η_c = ${S.lrCrit.toFixed(3)}` });
                    },
                },
                {
                    id: 'lrscan', label: '最终损失 vs 学习率', sub: '80 步后的 L（扫描 60 个 η）· 红线 = η_c', size: [360, 200],
                    render: (c, S) => {
                        const xs = [], ys = [];
                        for (let i = 0; i < 60; i++) { const lr = 0.005 * Math.pow(1.3 * S.lrCrit / 0.005, i / 59); const r = P01.runGD(S.st, S.w0, S.b0, lr, P01.K); xs.push(lr); ys.push(isFinite(r.L[P01.K]) ? r.L[P01.K] : 1e4); }
                        PL.draw.plot(c, { logX: true, logY: true, y: [Math.max(1e-4, S.Lmin * 0.5), 1e4], xLabel: '学习率 η（对数）', series: [{ data: ys, x: xs, color: LM.C.blue }], marks: [{ x: S.lrCrit, label: 'η_c', color: LM.C.red }, { x: S.lr, label: '当前', color: LM.C.yellow }] });
                    },
                },
            ],
            notes: '最终损失随 η 先降后在 η_c 处陡然飙升——<b>最好的学习率往往就在发散边缘的下方</b>。这也是训练大模型时学习率要 warmup、要随训练衰减的原因之一（第 05 篇）。',
            warn: '这里的临界值只对二次碗精确成立。神经网络的曲率随位置变化，实践中靠扫描（LR range test）和经验来定学习率；但「曲率越大、能承受的步长越小」这一点完全通用。',
        },
        {
            id: 'scale', title: '特征尺度与条件数：为什么要做归一化', tag: 'OPTIM',
            desc: '把特征换成 <code>x + c</code>（没有减均值的原始读数），模型能力完全没变（b 可以吸收偏移），但损失地形被拉成一条<strong>细长斜峡谷</strong>：两个方向的曲率之比——<strong>条件数 κ = a₁/a₂</strong>——从十几暴涨到成百上千。GD 的步长受最陡方向限制，沿峡谷底部却几乎走不动。<strong>标准化</strong> z = (x − μ)/σ 让 A 变成单位阵，碗变圆，GD 用最优步长一步就到底。',
            formula: [
                'κ = a_max / a_min      每步误差缩减因子 ≥ (κ − 1)/(κ + 1)   // 最优固定步长 η = 1/(a_max + a_min) 时',
                'x → x + c:  det A = Var(x) 不变，a_max ≈ c^2  ⇒  κ ≈ a_max^2 / Var(x) ∝ c^4',
                'z_i = (x_i − μ) / σ  ⇒  A = [[1, 0], [0, 1]], κ = 1',
            ],
            code: {
                title: 'C++ · standardize / conditionNumber', src: `
                // 标准化的结果：z 以及用到的 μ、σ
                struct Standardized { std::vector<double> z; double mu = 0, sigma = 1; };

                // z_i = (x_i − μ)/σ，μ、σ 是训练集上的均值和标准差
                Standardized standardize(const std::vector<double>& x) {
                    int n = (int)x.size();
                    Standardized s;
                    for (int i = 0; i < n; i++) s.mu += x[i] / n;
                    double var = 0;
                    for (int i = 0; i < n; i++) var += (x[i] - s.mu) * (x[i] - s.mu) / n;
                    s.sigma = var > 0 ? std::sqrt(var) : 1.0;             // 所有 x 都一样时不缩放
                    s.z = doubles(n);
                    for (int i = 0; i < n; i++) s.z[i] = (x[i] - s.mu) / s.sigma;
                    return s;                                             // 推理时必须用同一组 μ、σ 处理新数据
                }

                // 条件数 κ = a1 / a2：两个方向的曲率之比（eigA 见 Step 06）
                double conditionNumber(const std::vector<double>& x) {
                    Eig2 e = eigA(x);
                    return e.a1 / std::max(e.a2, 1e-12);
                }`,
            },
            previews: [
                {
                    id: 'scaleRaw', label: '原始特征 x + c', sub: '最优步长下的 GD 80 步', large: true,
                    render: (c, S) => P01.drawLand(c, { s: S.sc.sr, grid: S.sc.landR, view: S.sc.vR, paths: [{ tr: S.sc.runR.tr, dots: true }], points: [{ w: S.sc.runR.tr[0], b: S.sc.runR.tr[1] }], caption: `κ = ${S.sc.er.kappa.toFixed(0)}`, bLabel: "↑ b'" }),
                },
                {
                    id: 'scaleStd', label: '标准化特征 z', sub: '同一个初始直线，换一套参数化', large: true,
                    render: (c, S) => P01.drawLand(c, { s: S.sc.sz, grid: S.sc.landZ, view: S.sc.vZ, paths: [{ tr: S.sc.runZ.tr, dots: true }], points: [{ w: S.sc.runZ.tr[0], b: S.sc.runZ.tr[1] }], caption: `κ = ${S.sc.ez.kappa.toFixed(2)}`, wLabel: 'w_z →', bLabel: '↑ b_z' }),
                },
                {
                    id: 'scaleLoss', label: '损失曲线对比', sub: '两者初始损失相同（同一条初始直线）', size: [360, 200],
                    render: (c, S) => PL.draw.plot(c, { x: [0, P01.K], logY: true, xLabel: '迭代步 t', series: [{ data: Array.from(S.sc.runR.L), color: LM.C.red, label: `x + c（κ = ${S.sc.er.kappa.toFixed(0)}）` }, { data: Array.from(S.sc.runZ.L), color: LM.C.green, label: '标准化（κ = 1）' }], hmarks: [{ y: S.Lmin, color: LM.C.gray }] }),
                },
            ],
            notes: '把 <b>特征偏移 c</b> 从 0 拖到 6：左图的峡谷越来越细、红色曲线越来越平。κ 涨得比直觉还快：det A = Var(x) 不随 c 变，而 a₁ ≈ c²，所以 κ = a₁²/Var(x) 大约按 <b>c⁴</b> 增长（c = 0 时约 14，c = 3 时约 900）。右图永远一步到位。神经网络里的 <b>输入归一化、BatchNorm / LayerNorm、Adam 的逐参数步长</b>（第 02 篇）本质上都在对付这个问题。',
        },
        {
            id: 'sgd', title: '随机梯度下降：用一小批样本估计梯度', tag: 'OPTIM',
            desc: '全量梯度要扫一遍所有样本——大模型的训练集有上万亿 token，一步都算不起。<strong>Mini-batch SGD</strong> 每步只随机抽 B 个样本算梯度：它是全量梯度的<strong>无偏估计</strong>，但带噪声，噪声方差 ∝ 1/B。于是轨迹变抖，最后也不会精确停在 ✚，而是在碗底附近一个「噪声球」里打转——球的大小 ∝ η/B。',
            formula: [
                'g_t = (1/B) Σ_{i∈batch_t} ∇ℓ_i(θ_t)      E[g_t] = ∇L(θ_t)      Var[g_t] ∝ 1/B',
                'θ_{t+1} = θ_t − η · g_t',
            ],
            code: {
                title: 'C++ · sampleBatch / sgd', src: `
                // 从 idx 里无放回地随机抽 B 个，放到 idx 的前 B 位（部分 Fisher–Yates 洗牌）
                void sampleBatch(std::vector<int>& idx, int B, Rng& rng) {
                    int n = (int)idx.size();
                    for (int i = 0; i < B; i++) {
                        int j = i + randint(rng, n - i);                  // 在还没被抽到的 [i, n) 里挑一个
                        std::swap(idx[i], idx[j]);
                    }
                }

                // mini-batch SGD：每步只用随机 B 个样本估计梯度（batch ≤ 0 表示全量，就是普通 GD）；history 记录全数据上的损失
                std::vector<Snapshot> sgd(const std::vector<double>& x, const std::vector<double>& y, double w, double b,
                                          double lr, int steps, int batch, uint32_t seed) {
                    int n = (int)x.size();
                    int B = (batch <= 0 || batch > n) ? n : batch;
                    Rng rng = makeRng(seed);
                    std::vector<int> idx = ints(n);
                    for (int i = 0; i < n; i++) idx[i] = i;
                    std::vector<double> xb = doubles(B), yb = doubles(B);
                    std::vector<Snapshot> history;
                    history.push_back({w, b, mseLoss(x, y, w, b)});
                    for (int t = 0; t < steps; t++) {
                        sampleBatch(idx, B, rng);                         // 随机一小批
                        for (int i = 0; i < B; i++) { xb[i] = x[idx[i]]; yb[i] = y[idx[i]]; }
                        Grad g = gradients(xb, yb, w, b);                 // 有噪声的梯度估计，期望 = 全量梯度
                        w -= lr * g.dw;
                        b -= lr * g.db;
                        double L = mseLoss(x, y, w, b);
                        if (!std::isfinite(L) || L > 1e8) break;
                        history.push_back({w, b, L});
                    }
                    return history;
                }`,
            },
            previews: [
                {
                    id: 'sgdtraj', live: true, label: 'SGD 轨迹', sub: '黄 = 顶栏选中的 B · 灰虚线 = 全量 GD', large: true,
                    render: (c, S) => {
                        const k = S.it, sel = Math.max(0, S.sgdSizes.indexOf(S.batch)), r = S.sgdRuns[sel], full = S.sgdRuns[3];
                        P01.drawLand(c, { s: S.st, grid: S.land, paths: [{ tr: full.tr, color: LM.C.gray, faint: true }, { tr: r.tr, faint: true }, { tr: r.tr, upto: k, dots: true }], points: [{ w: r.tr[2 * k], b: r.tr[2 * k + 1], label: `B = ${S.batch || S.n}` }] });
                    },
                },
                {
                    id: 'sgdloss', label: '不同批大小的损失曲线', sub: '全数据损失 · 对数纵轴', size: [360, 220],
                    render: (c, S) => {
                        const cols = [LM.C.red, LM.C.orange, LM.C.blue, LM.C.green];
                        PL.draw.plot(c, { x: [0, P01.K], logY: true, xLabel: '迭代步 t', series: S.sgdRuns.map((r, k) => ({ data: Array.from(r.L), color: cols[k], width: S.sgdSizes[k] === S.batch ? 2.5 : 1.2, label: `B = ${S.sgdSizes[k] || '全量 ' + S.n}` })), hmarks: [{ y: S.Lmin, color: LM.C.gray }] });
                    },
                },
            ],
            notes: '选 <b>批大小 B</b> = 1：轨迹像醉汉下山，但方向大体正确；= 全量：平滑的 GD。再调大 <b>学习率 η</b>：噪声球变大。实际训练中一个 batch 的梯度计算是并行的，所以 B 通常取硬件放得下的最大值，再配合调整学习率。',
        },
        {
            id: 'generalize', title: '泛化：训练误差低不等于学会了', tag: 'EVAL',
            desc: '机器学习真正关心的是<strong>没见过的数据</strong>上的误差。把模型换成 k 次多项式（容量更大），只用很少的样本拟合，再到 200 个新样本（测试集）上评估：次数太低 → <strong>欠拟合</strong>，训练、测试误差都高；次数太高 → <strong>过拟合</strong>，曲线穿过每个训练点、把噪声也背了下来，测试误差反而飙升。<strong>正则化</strong>（岭回归惩罚 λ‖β‖²）和<strong>更多数据</strong>都能把它拉回来。',
            formula: [
                'ŷ = Σ_{j=0}^{k} β_j · (x − 1)^j      β = (Φ^T Φ + λ I)^{−1} Φ^T y   // 岭回归闭式解',
                '训练误差 = mean_{train}(ŷ − y)^2      测试误差 = mean_{test}(ŷ − y)^2   // 只有后者衡量泛化',
            ],
            code: {
                title: 'C++ · polyFeatures / fitRidge / polyMSE', src: `
                // 一个样本的多项式特征 φ_j = (x − 1)^j，j = 0..k（先居中到 [−1, 1]，否则 x^12 数值很差）
                std::vector<double> polyFeatures(double x, int k) {
                    std::vector<double> phi = doubles(k + 1);
                    double t = x - 1.0, p = 1.0;
                    for (int j = 0; j <= k; j++) { phi[j] = p; p *= t; }
                    return phi;
                }

                // 解 m×m 线性方程组 A·β = r（A 按行主序存，传值进来、在副本上消元）：高斯消元 + 部分主元
                std::vector<double> solve(std::vector<double> A, std::vector<double> r, int m) {
                    for (int c = 0; c < m; c++) {
                        int p = c;                                        // 第 c 列里绝对值最大的一行当主元，数值更稳
                        for (int i = c + 1; i < m; i++) if (std::fabs(A[i * m + c]) > std::fabs(A[p * m + c])) p = i;
                        for (int j = 0; j < m; j++) std::swap(A[c * m + j], A[p * m + j]);
                        std::swap(r[c], r[p]);
                        double piv = A[c * m + c] != 0 ? A[c * m + c] : 1e-300;   // 奇异时兜底（λ > 0 时不会发生）
                        for (int i = c + 1; i < m; i++) {                 // 把下面各行的第 c 列消成 0
                            double f = A[i * m + c] / piv;
                            for (int j = c; j < m; j++) A[i * m + j] -= f * A[c * m + j];
                            r[i] -= f * r[c];
                        }
                    }
                    std::vector<double> beta = doubles(m);
                    for (int i = m - 1; i >= 0; i--) {                    // 回代：从最后一个未知数往前解
                        double v = r[i];
                        for (int j = i + 1; j < m; j++) v -= A[i * m + j] * beta[j];
                        beta[i] = v / (A[i * m + i] != 0 ? A[i * m + i] : 1e-300);
                    }
                    return beta;
                }

                // 岭回归闭式解 β = (ΦᵀΦ + λI)⁻¹·Φᵀy。不必存下整个 Φ：逐个样本把 φ·φᵀ 和 φ·y 累加起来
                std::vector<double> fitRidge(const std::vector<double>& x, const std::vector<double>& y, int k, double lam) {
                    int m = k + 1;
                    std::vector<double> A = doubles(m * m), r = doubles(m);
                    for (size_t i = 0; i < x.size(); i++) {
                        std::vector<double> phi = polyFeatures(x[i], k);
                        for (int a = 0; a < m; a++) {
                            r[a] += phi[a] * y[i];                                         // Φᵀy
                            for (int c = 0; c < m; c++) A[a * m + c] += phi[a] * phi[c];   // ΦᵀΦ
                        }
                    }
                    for (int a = 0; a < m; a++) A[a * m + a] += lam;                       // + λI（L2 正则）
                    return solve(A, r, m);
                }

                // ŷ = Σ_j β_j·(x − 1)^j
                double polyAt(const std::vector<double>& beta, double x) {
                    std::vector<double> phi = polyFeatures(x, (int)beta.size() - 1);
                    double yhat = 0;
                    for (size_t j = 0; j < beta.size(); j++) yhat += beta[j] * phi[j];
                    return yhat;
                }

                // 多项式模型在一组样本上的 MSE：训练集、测试集各算一次，只有后者衡量泛化
                double polyMSE(const std::vector<double>& beta, const std::vector<double>& x, const std::vector<double>& y) {
                    double sum = 0;
                    for (size_t i = 0; i < x.size(); i++) {
                        double r = polyAt(beta, x[i]) - y[i];
                        sum += r * r;
                    }
                    return sum / x.size();
                }`,
            },
            previews: [
                {
                    id: 'polyfit', label: '多项式拟合', sub: '蓝点 = 训练样本 · 灰点 = 测试样本 · 黄线 = 当前次数的拟合', large: true, size: [360, 270],
                    render: (c, S) => {
                        const beta = S.polyFits[S.degree], curve = [];
                        for (let i = 0; i <= 200; i++) { const x = -0.05 + 2.1 * i / 200; curve.push([x, P01.polyAt(beta, x)]); }
                        const pts = [];
                        for (let i = 0; i < P01.NTEST; i += 2) pts.push({ x: S.xt[i], y: S.yt[i], color: '#3d4650', r: 2 });
                        for (let i = 0; i < S.polyN; i++) pts.push({ x: S.xp[i], y: S.yp[i], color: LM.C.blue, r: 3.5 });
                        LM.draw.scatter(c, pts, { x: [-0.1, 2.1], y: [-2.5, 7.5], paths: [{ pts: curve, color: LM.C.yellow, width: 2 }], lines: [{ a: [-0.2, 0.6], b: [2.2, 5.4], color: LM.C.gray, dash: [4, 4] }], xLabel: `k = ${S.degree}，λ = ${S.lam.toExponential(0)}`, yLabel: 'y' });
                    },
                },
                {
                    id: 'polyerr', label: '训练 / 测试误差 vs 多项式次数', sub: '对数纵轴 · 灰线 = 噪声下限 σ²', size: [360, 230],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, P01.DEG_MAX], logY: true, y: [1e-4, 100], xLabel: '多项式次数 k（训练误差低于 10⁻⁴ 时贴底显示）',
                        series: [{ data: S.polyTrain.map((v) => Math.max(v, 1.2e-4)), color: LM.C.train, label: '训练误差' }, { data: S.polyTest.map((v) => Math.min(v, 99)), color: LM.C.val, label: '测试误差', width: 2 }],
                        marks: [{ x: S.degree, label: '当前 k' }], hmarks: [{ y: Math.max(1e-4, S.noise * S.noise), color: LM.C.gray }],
                    }),
                },
            ],
            notes: '默认 12 个样本、k = 9：黄线扭成蛇形，测试误差比 k = 1 高出几个数量级。三种修法：把 <b>log₁₀λ</b> 拖到 −2；把 <b>拟合样本数</b> 拖到 60；或者降低 <b>多项式次数</b>。大模型参数远多于这里，却能泛化——海量数据、正则化与「早停」都在起作用。',
            warn: '测试集只能用来<strong>最终评估</strong>。如果反复看测试误差来调超参数，测试集就悄悄变成了训练集的一部分；正确做法是再切出一份验证集（第 05 篇的 val loss）。',
        },
        {
            id: 'scaleup', title: '从 Hello World 到大模型：同一个配方', tag: 'NEXT',
            desc: '把手写的 <code>gradients()</code> 换成<strong>自动微分</strong>：框架记录前向计算的每一步，再按链式法则自动反推梯度（第 02 篇手写一遍，第 04 篇之后全部交给 autograd）。下面左图是同一个梯度下降，一条用手写梯度（float64），一条用本系列的 JS 自动微分引擎 <code>LM.AG</code>（float32）——两条轨迹逐步重合。这四件套原样放大，就是训练 GPT 的流程：',
            blocks: [
                { type: 'html', html: `<table class="lm-tbl"><tr><th>要素</th><th>本篇（线性回归）</th><th>大语言模型（第 05 篇起）</th></tr>
                    <tr><td>数据</td><td>n 个 (x, y) 数对</td><td>万亿级 token 的文本；样本 = 一段文字，目标 = 每个位置的下一个 token</td></tr>
                    <tr><td>模型</td><td>ŷ = w·x + b，2 个参数</td><td>Transformer，10⁹–10¹² 个参数（本系列的迷你 GPT 约 2.9 万）</td></tr>
                    <tr><td>损失</td><td>均方误差 MSE</td><td>交叉熵（下一个 token 的负对数概率）</td></tr>
                    <tr><td>梯度</td><td>手推公式</td><td>自动微分（反向传播），一次 backward 算出全部参数的梯度</td></tr>
                    <tr><td>优化器</td><td>GD / mini-batch SGD</td><td>AdamW + warmup + cosine 学习率 + 梯度裁剪</td></tr>
                    <tr><td>评估</td><td>测试集 MSE</td><td>验证集 loss、下游任务、人类偏好（第 06 篇）</td></tr></table>` },
                {
                    type: 'code', title: 'C++ · 自动微分版：同一个训练循环（cpp/minml_autograd.h）', src: `
                    // 同一个梯度下降，梯度改由自动微分来算（与页面 JS 的 LM.AG 一一对应）。真实框架（如 PyTorch）里
                    // 循环体的后三行就是 opt.zero_grad() / loss.backward() / opt.step()。张量是 float，所以和 Step 05 的
                    // gradientDescent（double）只差 float 的舍入误差；history[t] 同样是第 t 次更新前的 (w, b, L)
                    std::vector<Snapshot> trainAutograd(const std::vector<double>& x, const std::vector<double>& y,
                                                        double w0, double b0, float lr, int steps) {
                        int n = (int)x.size();
                        Tensor xt = zeros(n, 1), yt = zeros(n, 1);                     // 数据排成 [n × 1] 的列向量
                        for (int i = 0; i < n; i++) { xt.v[i] = (float)x[i]; yt.v[i] = (float)y[i]; }
                        Var X = constant(xt), Y = constant(yt);                        // 常量：不求梯度
                        Var w = param(full(1, 1, (float)w0));                          // 需要梯度的参数：[1 × 1] 的 w 和 b
                        Var b = param(full(1, 1, (float)b0));
                        std::vector<Var> params = {w, b};
                        SGD opt = makeSGD(params, lr);                                 // 不带动量：p ← p − lr·grad
                        std::vector<Snapshot> history;
                        for (int t = 0; t <= steps; t++) {
                            // 前向：ŷ = X·w + b，L = mean((ŷ − y)²)；每个算子都把自己记进 tape（计算图）
                            Var loss = mse(linear(X, w, b), Y);
                            float L = loss->val.v[0];
                            if (!std::isfinite(L) || L > 1e8f) { clearTape(); break; } // 已发散：丢掉这次前向记下的 tape
                            history.push_back({w->val.v[0], b->val.v[0], L});
                            zeroGrad(params);                                          // 清掉上一步留下的梯度
                            backward(loss);                                            // 反向：沿 tape 逆序做链式法则 → w->grad、b->grad
                            sgdStep(opt, lr);                                          // w ← w − lr·w->grad，b 同理
                        }
                        return history;
                    }`,
                },
                {
                    type: 'previews', items: [
                        {
                            id: 'agcurve', label: '手写梯度 vs 自动微分', sub: '两条损失曲线重合（橙色虚线压在蓝线上）', large: true, size: [360, 220],
                            render: (c, S) => PL.draw.plot(c, { x: [0, P01.K], logY: true, xLabel: '迭代步 t', series: [{ data: Array.from(S.gd.L), color: LM.C.blue, width: 3, label: '手写 gradients()（float64）' }, { data: Array.from(S.ag.L), color: LM.C.orange, dash: [5, 4], width: 2, label: 'LM.AG 自动微分（float32）' }], hmarks: [{ y: S.Lmin, color: LM.C.gray }] }),
                        },
                        {
                            id: 'agdiff', label: '两条轨迹的最大差异', sub: '80 步内 |Δw|、|Δb| 的最大值', size: [360, 140],
                            render: (c, S) => {
                                const k = P01.K, ok = isFinite(S.gd.tr[2 * k]);
                                LM.draw.text(c, [
                                    [{ t: '手写   ', color: LM.C.dim }, { t: ok ? `w = ${S.gd.tr[2 * k].toFixed(6)}  b = ${S.gd.tr[2 * k + 1].toFixed(6)}` : '已发散', color: LM.C.blue }],
                                    [{ t: '自动   ', color: LM.C.dim }, { t: isFinite(S.ag.tr[2 * k]) ? `w = ${S.ag.tr[2 * k].toFixed(6)}  b = ${S.ag.tr[2 * k + 1].toFixed(6)}` : '已发散', color: LM.C.orange }],
                                    [{ t: '最大差 ', color: LM.C.dim }, { t: S.ag.maxDiff.toExponential(2) + '  （≈ float32 的舍入误差）', color: LM.C.green }],
                                    [{ t: '正规方程 ', color: LM.C.dim }, { t: `w* = ${S.opt[0].toFixed(6)}  b* = ${S.opt[1].toFixed(6)}`, color: LM.C.gray }],
                                ], { px: 12 });
                            },
                        },
                    ],
                },
                { type: 'note', html: '下一篇（02）把模型换成多层神经网络、把任务换成手写数字分类：损失变成交叉熵，梯度要穿过多层非线性——这就是<b>反向传播</b>。' },
            ],
        },
    ],

    graph: {
        title: '训练循环 · 数据流',
        intro: '每个节点 = 训练循环里的一个计算环节（在 GPU 上就是一次或几次 kernel launch；这里在 CPU 上用 JS 跑）。pin 是张量名 · 形状。<code>Params</code> 是 <b>HISTORY</b> 资源：每一步读取上一步更新后的 (w, b)，虚线回边表示跨迭代的依赖。',
        types: {
            data: { color: '#6ab0e0', label: 'data · 样本张量 [n]' },
            scalar: { color: '#e8c25a', label: 'scalar · 标量 / 超参数' },
            param: { color: '#5ad07a', label: 'param · 可训练参数 (w, b)' },
            grad: { color: '#e06c75', label: 'grad · 梯度' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用函数（悬停代码里的函数名可查看）', open: false,
            blocks: LM.CPP.blocks(['autograd', 'nn']).concat([{ label: '━ 本页公用 ━', code: `
                // 本篇手写的部分一律用 double（与页面 JS 参考实现的 Float64Array 相同）：Step 04 的数值梯度校验需要这个精度
                // n 个 double 组成的数组：std::vector<double> x = doubles(n);（与库里的 floats(n) 同一个用法）
                std::vector<double> doubles(int n, double x = 0.0) { return std::vector<double>((size_t)n, x); }

                // 一条直线 ŷ = w·x + b 的两个参数
                struct Line { double w = 0, b = 0; };

                // 损失对两个参数的梯度 (∂L/∂w, ∂L/∂b)
                struct Grad { double dw = 0, db = 0; };

                // 训练过程中的一步：当时的参数，以及全数据上的损失
                struct Snapshot { double w = 0, b = 0, loss = 0; };

                // 前向与损失：后面每一步都会调用。这里只列原型，实现在 Step 02、Step 03 的代码里（悬停函数名可看）
                std::vector<double> predict(const std::vector<double>& x, double w, double b);                    // ŷ = w·x + b
                double mseLoss(const std::vector<double>& x, const std::vector<double>& y, double w, double b);   // L = mean((ŷ − y)²)` }]),
        },
        nodes: [
            {
                id: 'data', title: 'makeData', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'data', desc: 'x ~ U(0,2), y = 2x+1+σε',
                dispatch: { kind: 'CPU', name: 'makeData', count: (S) => `n = ${S.n}（另有 200 个测试样本）` },
                outputs: [{ name: 'X', format: '[n] f64', type: 'data' }, { name: 'Y', format: '[n] f64', type: 'data' }],
                detail: { preview: 'scatter' },
            },
            {
                id: 'hyper', title: 'HyperParams', tag: 'CB', category: 'user', desc: 'η · B · (w₀, b₀) · steps',
                outputs: [{ name: 'Hyper', format: 'scalars', type: 'scalar' }],
                detail: { desc: '用户控制的超参数：学习率 η、批大小 B、初始参数、迭代步数 K = 80。它们不被训练，只决定训练怎样进行。', preview: null },
            },
            {
                id: 'batch', title: 'sampleBatch', tag: 'CPU', lifetime: 'history', step: 'sgd', desc: '部分洗牌 · 无放回抽 B 个',
                dispatch: { kind: 'CPU', name: 'sampleBatch', count: (S) => `B = ${S.batch || S.n}` },
                inputs: [{ name: 'X', format: '[n] f64', type: 'data' }, { name: 'Y', format: '[n] f64', type: 'data' }, { name: 'Hyper', format: 'B', type: 'scalar' }],
                outputs: [{ name: 'Xb', format: '[B] f64', type: 'data' }, { name: 'Yb', format: '[B] f64', type: 'data' }],
                detail: { preview: 'sgdtraj' },
            },
            {
                id: 'fwd', title: 'predict', tag: 'CPU', lifetime: 'history', step: 'model', desc: 'ŷ = w·x + b',
                dispatch: { kind: 'CPU', name: 'predict', count: 'B 次乘加' },
                inputs: [{ name: 'Xb', format: '[B] f64', type: 'data' }, { name: 'Params', format: '(w, b)', type: 'param', history: true }],
                outputs: [{ name: 'Yhat', format: '[B] f64', type: 'data' }],
                detail: { preview: 'gdfit' },
            },
            {
                id: 'loss', title: 'mseLoss', tag: 'CPU', lifetime: 'history', step: 'loss', desc: 'mean((ŷ − y)²)',
                dispatch: { kind: 'CPU', name: 'mseLoss', count: 'B 次平方 + 1 次归约' },
                inputs: [{ name: 'Yhat', format: '[B] f64', type: 'data' }, { name: 'Yb', format: '[B] f64', type: 'data' }],
                outputs: [{ name: 'L', format: 'scalar', type: 'scalar' }],
                detail: { preview: 'gdloss' },
            },
            {
                id: 'bwd', title: 'gradients', tag: 'CPU', lifetime: 'history', step: 'grad', desc: '(2·mean(r·x), 2·mean(r))',
                dispatch: { kind: 'CPU', name: 'gradients', count: '2 次归约' },
                inputs: [{ name: 'Yhat', format: '[B] f64', type: 'data' }, { name: 'Yb', format: '[B] f64', type: 'data' }, { name: 'Xb', format: '[B] f64', type: 'data' }],
                outputs: [{ name: 'Grad', format: '(∂w, ∂b)', type: 'grad' }],
                detail: { preview: 'gradfield' },
            },
            {
                id: 'upd', title: 'sgdUpdate', tag: 'CPU', category: 'output', lifetime: 'history', step: 'gd', desc: 'θ ← θ − η·g',
                dispatch: { kind: 'CPU', name: 'update', count: '2 个参数' },
                inputs: [{ name: 'Grad', format: '(∂w, ∂b)', type: 'grad' }, { name: 'Params', format: '(w, b)', type: 'param', history: true }, { name: 'Hyper', format: 'η', type: 'scalar' }],
                outputs: [{ name: 'Params', format: '(w, b)', type: 'param' }],
                detail: { preview: 'gdtraj', code: 0 },
            },
        ],
    },

    references: [
        { title: 'Nouvelles méthodes pour la détermination des orbites des comètes（附录：最小二乘法）', author: 'A.-M. Legendre', year: 1805, note: '最小二乘 / 正规方程的最早发表' },
        { title: 'Méthode générale pour la résolution des systèmes d\'équations simultanées', author: 'A. Cauchy', venue: 'Comptes Rendus', year: 1847, note: '梯度下降（最速下降法）的起源' },
        { title: 'A Stochastic Approximation Method', author: 'H. Robbins, S. Monro', venue: 'Annals of Mathematical Statistics', year: 1951, note: '随机梯度方法的收敛理论基础' },
        { title: 'Ridge Regression: Biased Estimation for Nonorthogonal Problems', author: 'A. E. Hoerl, R. W. Kennard', venue: 'Technometrics', year: 1970, note: 'L2 正则（岭回归）' },
        { title: 'Why Momentum Really Works', author: 'Gabriel Goh', venue: 'Distill', year: 2017, url: 'https://distill.pub/2017/momentum/', note: '二次碗上 GD 的特征值分析、条件数与动量的交互式讲解' },
        { title: 'Deep Learning（第 4、5 章）', author: 'I. Goodfellow, Y. Bengio, A. Courville', venue: 'MIT Press', year: 2016, url: 'https://www.deeplearningbook.org/', note: '数值优化、过拟合与正则化' },
    ],
});

