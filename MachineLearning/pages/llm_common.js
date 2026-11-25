// llm_common.js — 「从 Hello World 到多模态大模型」系列的共享代码（每篇构建时作为第一个输入）
//
//   LM.C / LM.draw     调色板与绘图辅助（热图、柱状图、token 条、散点、文本面板、曲线）
//   LM.AG              微型自动微分：二维张量 + 录带（tape）反向传播
//   LM.AdamW / LM.SGD  优化器；LM.clipGradNorm 梯度裁剪；LM.cosineLR 学习率调度
//   LM.gpt             与 nanoGPT 同构的迷你 GPT（Pre-LN、可学习位置、权重共享），含 KV cache 采样
//   LM.world / LM.vocab 中文「玩具世界」语料生成器与字符级 tokenizer
//   LM.CPP             页面代码块共用的 C++ 库（pages/cpp/minml_*.h 的原文，由 tools/gen_cpp.js 生成，不要手改）
//
// 张量约定：一律二维行主序 [rows × cols]。一个 batch 的序列展平成 [B·T × d]：第 b 条序列第 t 个位置 = 第 b·T + t 行。
// 本文件不依赖 DOM（绘图函数只在被调用时才用到全局 PL），因此也能在 Node 里 require 用来离线训练权重。
const LM = (function () {
    'use strict';
    const LM = {};

    // ═════════════════════════════════ 调色板 / 小工具 ═════════════════════════════════
    LM.C = {
        blue: '#6ab0e0', orange: '#e0a060', green: '#5ad07a', purple: '#c678dd', red: '#e06c75', cyan: '#56b6c2',
        yellow: '#e8c25a', gray: '#8a8f98', text: '#c8ccd4', dim: '#6a6a6a', grid: '#1c1c1c', panel: '#0b0d10',
        train: '#6ab0e0', val: '#e0a060', target: '#5ad07a', wrong: '#e06c75',
    };
    LM.SERIES = ['#6ab0e0', '#e0a060', '#5ad07a', '#c678dd', '#e06c75', '#56b6c2', '#e8c25a', '#d19a66'];

    LM.rng = function (seed) {                      // mulberry32，与 PL.rng 相同（Node 里也能用）
        let a = (seed >>> 0) || 1;
        return function () {
            a = (a + 0x6D2B79F5) | 0;
            let t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    };
    LM.randn = function (rand) {
        const u1 = Math.max(rand(), 1e-12), u2 = rand();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    LM.shuffle = function (a, rand) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; } return a; };
    LM.clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
    /** softmax of src[off .. off+n) / temp → out (Float32Array or Array, length n). Returns out. */
    LM.softmax = function (src, out, temp, off, n) {
        temp = temp || 1; off = off || 0; n = n || src.length - off; out = out || new Float32Array(n);
        let m = -Infinity;
        for (let i = 0; i < n; i++) if (src[off + i] > m) m = src[off + i];
        let s = 0;
        for (let i = 0; i < n; i++) { const e = Math.exp((src[off + i] - m) / temp); out[i] = e; s += e; }
        for (let i = 0; i < n; i++) out[i] /= s;
        return out;
    };
    LM.argmax = function (a, off, n) { off = off || 0; n = n || a.length - off; let b = 0; for (let i = 1; i < n; i++) if (a[off + i] > a[off + b]) b = i; return b; };
    /** indices of the k largest values (descending) */
    LM.topk = function (a, k) { return Array.from(a, (v, i) => i).sort((i, j) => a[j] - a[i]).slice(0, k); };
    LM.mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; };
    /** exponential moving average of a series (for smoothing noisy loss curves) */
    LM.ema = function (a, beta) { const out = new Array(a.length); let m = 0, w = 0; for (let i = 0; i < a.length; i++) { m = beta * m + (1 - beta) * a[i]; w = beta * w + (1 - beta); out[i] = m / w; } return out; };
    LM.fmt = (v, d = 3) => (v == null || !isFinite(v) ? '—' : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d));

    /** 2D PCA of rows X[n × d] (power iteration on the covariance). Returns { pts: Float32Array(2n), axes, mean, varRatio } */
    LM.pca2 = function (X, n, d, seed) {
        const mean = new Float64Array(d);
        for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j] / n;
        const Cv = new Float64Array(d * d);
        for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) { const xa = X[i * d + a] - mean[a]; for (let b = 0; b < d; b++) Cv[a * d + b] += xa * (X[i * d + b] - mean[b]) / n; }
        let tr = 0; for (let a = 0; a < d; a++) tr += Cv[a * d + a];
        const rand = LM.rng(seed || 7), axes = [], lams = [];
        for (let k = 0; k < 2; k++) {
            let v = new Float64Array(d); for (let j = 0; j < d; j++) v[j] = rand() - 0.5;
            let lam = 0;
            for (let it = 0; it < 120; it++) {
                const w = new Float64Array(d);
                for (let a = 0; a < d; a++) { let s = 0; for (let b = 0; b < d; b++) s += Cv[a * d + b] * v[b]; w[a] = s; }
                axes.forEach((u) => { let p = 0; for (let j = 0; j < d; j++) p += w[j] * u[j]; for (let j = 0; j < d; j++) w[j] -= p * u[j]; });
                let nrm = 0; for (let j = 0; j < d; j++) nrm += w[j] * w[j]; nrm = Math.sqrt(nrm) || 1;
                lam = nrm; for (let j = 0; j < d; j++) v[j] = w[j] / nrm;
            }
            // deterministic sign: largest |component| positive
            let bi = 0; for (let j = 1; j < d; j++) if (Math.abs(v[j]) > Math.abs(v[bi])) bi = j;
            if (v[bi] < 0) for (let j = 0; j < d; j++) v[j] = -v[j];
            axes.push(v); lams.push(lam);
        }
        const pts = new Float32Array(2 * n);
        for (let i = 0; i < n; i++) for (let k = 0; k < 2; k++) { let s = 0; for (let j = 0; j < d; j++) s += (X[i * d + j] - mean[j]) * axes[k][j]; pts[2 * i + k] = s; }
        return { pts, axes, mean, varRatio: lams.map((l) => (tr > 0 ? l / tr : 0)) };
    };

    // ═════════════════════════════════ 微型自动微分 LM.AG ═════════════════════════════════
    // Tensor = 二维矩阵 + 可选梯度。每个算子把「如何把输出梯度传回输入」的闭包压进 tape；backward 逆序执行 tape。
    class Tensor {
        constructor(r, c, data) { this.r = r; this.c = c; this.data = data || new Float32Array(r * c); this.grad = null; this.req = false; this.name = ''; }
        get size() { return this.r * this.c; }
        g() { return this.grad || (this.grad = new Float32Array(this.r * this.c)); }
        item() { return this.data[0]; }
        row(i) { return this.data.subarray(i * this.c, (i + 1) * this.c); }
    }
    const AG = { Tensor, tape: [], enabled: true };
    LM.AG = AG;
    AG.tensor = (r, c, data) => new Tensor(r, c, data instanceof Float32Array ? data : data ? Float32Array.from(data) : null);
    /** parameter tensor: req = true, init(data) fills it */
    AG.param = function (name, r, c, init) { const t = new Tensor(r, c); t.req = true; t.name = name; if (init) init(t.data); return t; };
    /** register a backward closure if any input needs a gradient (extension point for page-local ops) */
    AG.track = function (out, inputs, back) {
        if (!AG.enabled || !inputs.some((t) => t && t.req)) return out;
        out.req = true;
        AG.tape.push(() => { if (out.grad) back(out.grad); });
        return out;
    };
    /** run fn with gradient tracking off (inference) */
    AG.noGrad = function (fn) { const was = AG.enabled; AG.enabled = false; try { return fn(); } finally { AG.enabled = was; } };
    AG.backward = function (loss) {
        loss.g().fill(0); loss.grad[0] = 1;
        for (let i = AG.tape.length - 1; i >= 0; i--) AG.tape[i]();
        AG.tape.length = 0;
    };
    AG.clearTape = () => { AG.tape.length = 0; };

    /** C = A·B   A[n×k] B[k×m] */
    AG.matmul = function (A, B) {
        const n = A.r, k = A.c, m = B.c, a = A.data, b = B.data, C = new Tensor(n, m), c = C.data;
        for (let i = 0; i < n; i++) {
            const ci = i * m, ai = i * k;
            for (let p = 0; p < k; p++) { const av = a[ai + p]; if (av === 0) continue; const bp = p * m; for (let j = 0; j < m; j++) c[ci + j] += av * b[bp + j]; }
        }
        return AG.track(C, [A, B], (gc) => {
            if (A.req) { const ga = A.g(); for (let i = 0; i < n; i++) { const ci = i * m, ai = i * k; for (let p = 0; p < k; p++) { const bp = p * m; let s = 0; for (let j = 0; j < m; j++) s += gc[ci + j] * b[bp + j]; ga[ai + p] += s; } } }
            if (B.req) { const gb = B.g(); for (let i = 0; i < n; i++) { const ci = i * m, ai = i * k; for (let p = 0; p < k; p++) { const av = a[ai + p]; if (av === 0) continue; const bp = p * m; for (let j = 0; j < m; j++) gb[bp + j] += av * gc[ci + j]; } } }
        });
    };
    /** C = A·Bᵀ   A[n×k] B[m×k] → C[n×m] （权重共享的 LM head、CLIP 相似度矩阵） */
    AG.matmulBT = function (A, B) {
        const n = A.r, k = A.c, m = B.r, a = A.data, b = B.data, C = new Tensor(n, m), c = C.data;
        for (let i = 0; i < n; i++) { const ai = i * k; for (let j = 0; j < m; j++) { const bj = j * k; let s = 0; for (let p = 0; p < k; p++) s += a[ai + p] * b[bj + p]; c[i * m + j] = s; } }
        return AG.track(C, [A, B], (gc) => {
            const ga = A.req ? A.g() : null, gb = B.req ? B.g() : null;
            for (let i = 0; i < n; i++) {
                const ai = i * k;
                for (let j = 0; j < m; j++) {
                    const g = gc[i * m + j]; if (g === 0) continue; const bj = j * k;
                    if (ga) for (let p = 0; p < k; p++) ga[ai + p] += g * b[bj + p];
                    if (gb) for (let p = 0; p < k; p++) gb[bj + p] += g * a[ai + p];
                }
            }
        });
    };
    /** Y = X·W + b   X[n×k] W[k×m] b[1×m]（可省略） */
    AG.linear = function (X, W, bias) {
        const Y = AG.matmul(X, W);
        if (!bias) return Y;
        return AG.addRow(Y, bias);
    };
    /** Y = X + b（b 为一行，按行广播） */
    AG.addRow = function (X, b) {
        const n = X.r, m = X.c, Y = new Tensor(n, m), x = X.data, y = Y.data, bb = b.data;
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) y[i * m + j] = x[i * m + j] + bb[j];
        return AG.track(Y, [X, b], (g) => {
            if (X.req) { const gx = X.g(); for (let i = 0; i < g.length; i++) gx[i] += g[i]; }
            if (b.req) { const gb = b.g(); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) gb[j] += g[i * m + j]; }
        });
    };
    AG.add = function (A, B) {
        const Y = new Tensor(A.r, A.c), a = A.data, b = B.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = a[i] + b[i];
        return AG.track(Y, [A, B], (g) => {
            if (A.req) { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i]; }
            if (B.req) { const gb = B.g(); for (let i = 0; i < g.length; i++) gb[i] += g[i]; }
        });
    };
    AG.sub = function (A, B) {
        const Y = new Tensor(A.r, A.c), a = A.data, b = B.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = a[i] - b[i];
        return AG.track(Y, [A, B], (g) => {
            if (A.req) { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i]; }
            if (B.req) { const gb = B.g(); for (let i = 0; i < g.length; i++) gb[i] -= g[i]; }
        });
    };
    /** elementwise product */
    AG.mul = function (A, B) {
        const Y = new Tensor(A.r, A.c), a = A.data, b = B.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = a[i] * b[i];
        return AG.track(Y, [A, B], (g) => {
            if (A.req) { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i] * b[i]; }
            if (B.req) { const gb = B.g(); for (let i = 0; i < g.length; i++) gb[i] += g[i] * a[i]; }
        });
    };
    AG.scale = function (A, s) {
        const Y = new Tensor(A.r, A.c), a = A.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = a[i] * s;
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i] * s; });
    };
    AG.relu = function (A) {
        const Y = new Tensor(A.r, A.c), a = A.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = a[i] > 0 ? a[i] : 0;
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < g.length; i++) if (a[i] > 0) ga[i] += g[i]; });
    };
    AG.tanh = function (A) {
        const Y = new Tensor(A.r, A.c), a = A.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = Math.tanh(a[i]);
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i] * (1 - y[i] * y[i]); });
    };
    const GK = Math.sqrt(2 / Math.PI);
    /** GELU（tanh 近似，与 GPT-2 / F.gelu(approximate="tanh") 一致） */
    AG.gelu = function (A) {
        const Y = new Tensor(A.r, A.c), a = A.data, y = Y.data, th = new Float32Array(a.length);
        for (let i = 0; i < y.length; i++) { const x = a[i], t = Math.tanh(GK * (x + 0.044715 * x * x * x)); th[i] = t; y[i] = 0.5 * x * (1 + t); }
        return AG.track(Y, [A], (g) => {
            const ga = A.g();
            for (let i = 0; i < g.length; i++) { const x = a[i], t = th[i]; ga[i] += g[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GK * (1 + 3 * 0.044715 * x * x)); }
        });
    };
    AG.sigmoid = function (A) {
        const Y = new Tensor(A.r, A.c), a = A.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = 1 / (1 + Math.exp(-a[i]));
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i] * y[i] * (1 - y[i]); });
    };
    /** log σ(x)，数值稳定（DPO / Bradley-Terry 损失） */
    AG.logsigmoid = function (A) {
        const Y = new Tensor(A.r, A.c), a = A.data, y = Y.data;
        for (let i = 0; i < y.length; i++) { const x = a[i]; y[i] = x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)); }
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i] * (1 / (1 + Math.exp(a[i]))); });
    };
    /** mean of all elements → 1×1 */
    AG.mean = function (A) {
        const Y = new Tensor(1, 1), a = A.data, n = a.length;
        let s = 0; for (let i = 0; i < n; i++) s += a[i]; Y.data[0] = s / n;
        return AG.track(Y, [A], (g) => { const ga = A.g(), v = g[0] / n; for (let i = 0; i < n; i++) ga[i] += v; });
    };
    AG.sum = function (A) {
        const Y = new Tensor(1, 1), a = A.data;
        let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; Y.data[0] = s;
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < a.length; i++) ga[i] += g[0]; });
    };
    AG.transpose = function (A) {
        const n = A.r, m = A.c, Y = new Tensor(m, n), a = A.data, y = Y.data;
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) y[j * n + i] = a[i * m + j];
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) ga[i * m + j] += g[j * n + i]; });
    };
    /** LayerNorm over each row: y = γ·(x − μ)/√(σ² + ε) + β */
    AG.layernorm = function (X, gamma, beta, eps) {
        eps = eps || 1e-5;
        const n = X.r, d = X.c, x = X.data, Y = new Tensor(n, d), y = Y.data, xh = new Float32Array(n * d), rstd = new Float32Array(n);
        const gm = gamma.data, bt = beta.data;
        for (let i = 0; i < n; i++) {
            const o = i * d; let mu = 0; for (let j = 0; j < d; j++) mu += x[o + j]; mu /= d;
            let v = 0; for (let j = 0; j < d; j++) { const t = x[o + j] - mu; v += t * t; } v /= d;
            const rs = 1 / Math.sqrt(v + eps); rstd[i] = rs;
            for (let j = 0; j < d; j++) { const h = (x[o + j] - mu) * rs; xh[o + j] = h; y[o + j] = h * gm[j] + bt[j]; }
        }
        Y.xhat = xh; Y.rstd = rstd;
        return AG.track(Y, [X, gamma, beta], (g) => {
            const gg = gamma.req ? gamma.g() : null, gb = beta.req ? beta.g() : null, gx = X.req ? X.g() : null;
            for (let i = 0; i < n; i++) {
                const o = i * d; let s1 = 0, s2 = 0;
                for (let j = 0; j < d; j++) { const dh = g[o + j] * gm[j]; s1 += dh; s2 += dh * xh[o + j]; if (gg) gg[j] += g[o + j] * xh[o + j]; if (gb) gb[j] += g[o + j]; }
                if (gx) { s1 /= d; s2 /= d; for (let j = 0; j < d; j++) gx[o + j] += rstd[i] * (g[o + j] * gm[j] - s1 - xh[o + j] * s2); }
            }
        });
    };
    /** Y[i] = X[idx[i]]（行收集；embedding 查表就是它的特例）。反向 = 按行 scatter-add */
    AG.gather = function (X, idx) {
        const n = idx.length, d = X.c, Y = new Tensor(n, d), x = X.data, y = Y.data;
        for (let i = 0; i < n; i++) { const s = idx[i] * d; for (let j = 0; j < d; j++) y[i * d + j] = x[s + j]; }
        return AG.track(Y, [X], (g) => { const gx = X.g(); for (let i = 0; i < n; i++) { const s = idx[i] * d; for (let j = 0; j < d; j++) gx[s + j] += g[i * d + j]; } });
    };
    AG.embed = (ids, E) => AG.gather(E, ids);
    /** row concat [A; B] */
    AG.concat = function (A, B) {
        const d = A.c, Y = new Tensor(A.r + B.r, d);
        Y.data.set(A.data, 0); Y.data.set(B.data, A.data.length);
        return AG.track(Y, [A, B], (g) => {
            if (A.req) { const ga = A.g(); for (let i = 0; i < ga.length; i++) ga[i] += g[i]; }
            if (B.req) { const gb = B.g(), o = A.data.length; for (let i = 0; i < gb.length; i++) gb[i] += g[o + i]; }
        });
    };
    /** mean over consecutive groups of `k` rows: X[(n·k) × d] → [n × d] */
    AG.meanRows = function (X, k) {
        const n = X.r / k, d = X.c, Y = new Tensor(n, d), x = X.data, y = Y.data;
        for (let i = 0; i < n; i++) for (let r = 0; r < k; r++) { const o = (i * k + r) * d; for (let j = 0; j < d; j++) y[i * d + j] += x[o + j] / k; }
        return AG.track(Y, [X], (g) => { const gx = X.g(); for (let i = 0; i < n; i++) for (let r = 0; r < k; r++) { const o = (i * k + r) * d; for (let j = 0; j < d; j++) gx[o + j] += g[i * d + j] / k; } });
    };
    /** row-wise L2 normalisation y = x/‖x‖（CLIP 嵌入） */
    AG.l2norm = function (X) {
        const n = X.r, d = X.c, x = X.data, Y = new Tensor(n, d), y = Y.data, inv = new Float32Array(n);
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < d; j++) s += x[i * d + j] ** 2; const r = 1 / Math.sqrt(s + 1e-12); inv[i] = r; for (let j = 0; j < d; j++) y[i * d + j] = x[i * d + j] * r; }
        return AG.track(Y, [X], (g) => {
            const gx = X.g();
            for (let i = 0; i < n; i++) { let dot = 0; for (let j = 0; j < d; j++) dot += g[i * d + j] * y[i * d + j]; for (let j = 0; j < d; j++) gx[i * d + j] += inv[i] * (g[i * d + j] - y[i * d + j] * dot); }
        });
    };
    /**
     * 多头缩放点积注意力（融合算子）。Q/K/V: [B·T × D]，H 个头，每头 dh = D/H。
     * causal = true 时位置 i 只能看 j ≤ i。输出 [B·T × D]；out.probs = Float32Array(B·H·T·T)，下标 ((b·H + h)·T + i)·T + j。
     */
    AG.attention = function (Q, K, V, B, T, H, causal) {
        const D = Q.c, dh = D / H, sc = 1 / Math.sqrt(dh), q = Q.data, k = K.data, v = V.data;
        const O = new Tensor(B * T, D), o = O.data, P = new Float32Array(B * H * T * T), row = new Float32Array(T);
        for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
            const hc = h * dh;
            for (let i = 0; i < T; i++) {
                const qi = (b * T + i) * D + hc, jmax = causal ? i : T - 1;
                let m = -Infinity;
                for (let j = 0; j <= jmax; j++) { const kj = (b * T + j) * D + hc; let s = 0; for (let c = 0; c < dh; c++) s += q[qi + c] * k[kj + c]; s *= sc; row[j] = s; if (s > m) m = s; }
                let z = 0; for (let j = 0; j <= jmax; j++) { row[j] = Math.exp(row[j] - m); z += row[j]; }
                const pb = ((b * H + h) * T + i) * T, oi = (b * T + i) * D + hc;
                for (let j = 0; j <= jmax; j++) { const p = row[j] / z; P[pb + j] = p; const vj = (b * T + j) * D + hc; for (let c = 0; c < dh; c++) o[oi + c] += p * v[vj + c]; }
            }
        }
        O.probs = P; O.B = B; O.T = T; O.H = H;
        return AG.track(O, [Q, K, V], (g) => {
            const gq = Q.req ? Q.g() : null, gk = K.req ? K.g() : null, gv = V.req ? V.g() : null, dP = new Float32Array(T);
            for (let b = 0; b < B; b++) for (let h = 0; h < H; h++) {
                const hc = h * dh;
                for (let i = 0; i < T; i++) {
                    const pb = ((b * H + h) * T + i) * T, oi = (b * T + i) * D + hc, qi = oi, jmax = causal ? i : T - 1;
                    let dot = 0;
                    for (let j = 0; j <= jmax; j++) {
                        const vj = (b * T + j) * D + hc, p = P[pb + j]; let s = 0;
                        for (let c = 0; c < dh; c++) { s += g[oi + c] * v[vj + c]; if (gv) gv[vj + c] += p * g[oi + c]; }
                        dP[j] = s; dot += p * s;
                    }
                    for (let j = 0; j <= jmax; j++) {
                        const ds = P[pb + j] * (dP[j] - dot) * sc; if (ds === 0) continue;
                        const kj = (b * T + j) * D + hc;
                        for (let c = 0; c < dh; c++) { if (gq) gq[qi + c] += ds * k[kj + c]; if (gk) gk[kj + c] += ds * q[qi + c]; }
                    }
                }
            }
        });
    };
    /**
     * 交叉熵（自带 softmax）。logits [n × V]，targets Int32Array(n)，−1 = 忽略该行。
     * o.weights（每行权重，可为负：策略梯度的 advantage / loss mask），o.denom（默认 = 有效行数或 Σ|w|）。
     * 返回 1×1 张量；附带 .rowLoss（每行 −log p，忽略行为 NaN）与 .probs（每行 softmax，按需 o.keepProbs）。
     */
    AG.crossEntropy = function (L, targets, o) {
        o = o || {};
        const n = L.r, V = L.c, z = L.data, w = o.weights || null, P = new Float32Array(n * V), rowLoss = new Float32Array(n);
        let tot = 0, denom = 0;
        for (let i = 0; i < n; i++) {
            const t = targets[i], off = i * V;
            let m = -Infinity; for (let j = 0; j < V; j++) if (z[off + j] > m) m = z[off + j];
            let s = 0; for (let j = 0; j < V; j++) { const e = Math.exp(z[off + j] - m); P[off + j] = e; s += e; }
            for (let j = 0; j < V; j++) P[off + j] /= s;
            if (t < 0) { rowLoss[i] = NaN; continue; }
            const li = -(z[off + t] - m - Math.log(s)); rowLoss[i] = li;
            const wi = w ? w[i] : 1; tot += wi * li; denom += w ? Math.abs(wi) : 1;
        }
        if (o.denom != null) denom = o.denom;
        denom = denom || 1;
        const Y = new Tensor(1, 1); Y.data[0] = tot / denom; Y.rowLoss = rowLoss; if (o.keepProbs) Y.probs = P;
        return AG.track(Y, [L], (g) => {
            const gl = L.g();
            for (let i = 0; i < n; i++) {
                const t = targets[i]; if (t < 0) continue;
                const k = g[0] * (w ? w[i] : 1) / denom, off = i * V; if (k === 0) continue;
                for (let j = 0; j < V; j++) gl[off + j] += k * P[off + j];
                gl[off + t] -= k;
            }
        });
    };
    /** 每条序列的 Σ log p(target)：rowSeq[i] = 行 i 属于第几条序列（−1 = 不计）。返回 [nSeq × 1]（DPO 需要它） */
    AG.seqLogProb = function (L, targets, rowSeq, nSeq) {
        const n = L.r, V = L.c, z = L.data, Y = new Tensor(nSeq, 1), P = new Float32Array(n * V);
        for (let i = 0; i < n; i++) {
            const s = rowSeq[i], t = targets[i]; if (s < 0 || t < 0) continue;
            const off = i * V; let m = -Infinity; for (let j = 0; j < V; j++) if (z[off + j] > m) m = z[off + j];
            let sum = 0; for (let j = 0; j < V; j++) { const e = Math.exp(z[off + j] - m); P[off + j] = e; sum += e; }
            for (let j = 0; j < V; j++) P[off + j] /= sum;
            Y.data[s] += z[off + t] - m - Math.log(sum);
        }
        return AG.track(Y, [L], (g) => {
            const gl = L.g();
            for (let i = 0; i < n; i++) {
                const s = rowSeq[i], t = targets[i]; if (s < 0 || t < 0) continue;
                const k = g[s], off = i * V; if (k === 0) continue;
                for (let j = 0; j < V; j++) gl[off + j] -= k * P[off + j];
                gl[off + t] += k;
            }
        });
    };
    /** 均方误差 mean((A − B)²)，B 可以是普通张量（不求梯度） */
    AG.mse = function (A, B) {
        const a = A.data, b = B.data, n = a.length, Y = new Tensor(1, 1);
        let s = 0; for (let i = 0; i < n; i++) s += (a[i] - b[i]) ** 2; Y.data[0] = s / n;
        return AG.track(Y, [A, B], (g) => {
            const k = 2 * g[0] / n;
            if (A.req) { const ga = A.g(); for (let i = 0; i < n; i++) ga[i] += k * (a[i] - b[i]); }
            if (B.req) { const gb = B.g(); for (let i = 0; i < n; i++) gb[i] -= k * (a[i] - b[i]); }
        });
    };

    // ── 第二批算子（第 03–05、11–15 篇：VAE、卷积 / U-Net、扩散、世界模型…）。全部做过有限差分梯度校验（tools/test_engine.js） ──
    const unary = (fwd, dfn) => function (A) {   // dfn(x, y) = dy/dx
        const Y = new Tensor(A.r, A.c), a = A.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = fwd(a[i]);
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i] * dfn(a[i], y[i]); });
    };
    /** e^x（VAE：σ = exp(½·logσ²)） */
    AG.exp = unary(Math.exp, (x, y) => y);
    /** ln x（输入须为正） */
    AG.log = unary(Math.log, (x) => 1 / x);
    /** softplus(x) = ln(1 + eˣ)，数值稳定 */
    AG.softplus = unary((x) => (x > 20 ? x : x < -20 ? Math.exp(x) : Math.log1p(Math.exp(x))), (x) => 1 / (1 + Math.exp(-x)));
    /** x² */
    AG.square = unary((x) => x * x, (x) => 2 * x);
    /** SiLU / Swish：x·σ(x)（扩散 U-Net、LLaMA 的 SwiGLU 常用） */
    AG.silu = unary((x) => x / (1 + Math.exp(-x)), (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 + x * (1 - s)); });
    /** Y = A + c（常数） */
    AG.addConst = function (A, c) {
        const Y = new Tensor(A.r, A.c), a = A.data, y = Y.data;
        for (let i = 0; i < y.length; i++) y[i] = a[i] + c;
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < g.length; i++) ga[i] += g[i]; });
    };
    /** 按列拼接 [A | B]（行数相同）：U-Net 的跳连、条件向量拼接 */
    AG.concatCols = function (A, B) {
        const n = A.r, ca = A.c, cb = B.c, Y = new Tensor(n, ca + cb), y = Y.data;
        for (let i = 0; i < n; i++) { y.set(A.data.subarray(i * ca, (i + 1) * ca), i * (ca + cb)); y.set(B.data.subarray(i * cb, (i + 1) * cb), i * (ca + cb) + ca); }
        return AG.track(Y, [A, B], (g) => {
            const w = ca + cb;
            if (A.req) { const ga = A.g(); for (let i = 0; i < n; i++) for (let j = 0; j < ca; j++) ga[i * ca + j] += g[i * w + j]; }
            if (B.req) { const gb = B.g(); for (let i = 0; i < n; i++) for (let j = 0; j < cb; j++) gb[i * cb + j] += g[i * w + ca + j]; }
        });
    };
    /** 取第 [c0, c1) 列：把一个线性层的输出拆成 μ / logσ²、GRU 的几个门… */
    AG.sliceCols = function (A, c0, c1) {
        const n = A.r, c = A.c, m = c1 - c0, Y = new Tensor(n, m), y = Y.data, a = A.data;
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) y[i * m + j] = a[i * c + c0 + j];
        return AG.track(Y, [A], (g) => { const ga = A.g(); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) ga[i * c + c0 + j] += g[i * m + j]; });
    };
    // 特征图约定：一张图 = 张量的一行，按通道优先展平 [C × H × W]（与 PyTorch 的 NCHW 一致）；N 张图 = N 行。
    // 形状不存进张量，由调用方用 { C, H, W } 传入；conv2d 等返回的张量带 .shape = { C, H, W } 方便串联。
    const shp = (o) => ({ C: o.C, H: o.H, W: o.W });
    /**
     * 二维卷积（互相关，与 PyTorch Conv2d 相同）。X: [N × C·H·W]，Wt: [Cout × C·KH·KW]，b: [1 × Cout]（可省略）。
     * o: { C, H, W, K | KH, KW, stride = 1, pad | padH, padW }。输出 [N × Cout·Ho·Wo]，Ho = ⌊(H + 2·padH − KH)/stride⌋ + 1。
     * 一维卷积（音频）：H = 1、KH = 1、padH = 0。
     */
    AG.conv2d = function (X, Wt, b, o) {
        const N = X.r, C = o.C, H = o.H, Wd = o.W, KH = o.KH || o.K, KW = o.KW || o.K, st = o.stride || 1;
        const pH = o.padH ?? o.pad ?? 0, pW = o.padW ?? o.pad ?? 0, Co = Wt.r;
        const Ho = Math.floor((H + 2 * pH - KH) / st) + 1, Wo = Math.floor((Wd + 2 * pW - KW) / st) + 1;
        const x = X.data, w = Wt.data, Y = new Tensor(N, Co * Ho * Wo), y = Y.data, inS = C * H * Wd, outS = Co * Ho * Wo, kS = C * KH * KW;
        for (let n = 0; n < N; n++) {
            const xo = n * inS, yo = n * outS;
            for (let co = 0; co < Co; co++) {
                const bv = b ? b.data[co] : 0, wo = co * kS;
                for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
                    let s = bv;
                    for (let ci = 0; ci < C; ci++) {
                        const xc = xo + ci * H * Wd, wc = wo + ci * KH * KW;
                        for (let ky = 0; ky < KH; ky++) {
                            const iy = oy * st - pH + ky; if (iy < 0 || iy >= H) continue;
                            const xr = xc + iy * Wd, wr = wc + ky * KW;
                            for (let kx = 0; kx < KW; kx++) { const ix = ox * st - pW + kx; if (ix >= 0 && ix < Wd) s += x[xr + ix] * w[wr + kx]; }
                        }
                    }
                    y[yo + (co * Ho + oy) * Wo + ox] = s;
                }
            }
        }
        Y.shape = { C: Co, H: Ho, W: Wo };
        return AG.track(Y, [X, Wt, b], (g) => {
            const gx = X.req ? X.g() : null, gw = Wt.req ? Wt.g() : null, gb = b && b.req ? b.g() : null;
            for (let n = 0; n < N; n++) {
                const xo = n * inS, yo = n * outS;
                for (let co = 0; co < Co; co++) {
                    const wo = co * kS;
                    for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
                        const gv = g[yo + (co * Ho + oy) * Wo + ox]; if (gv === 0) continue;
                        if (gb) gb[co] += gv;
                        for (let ci = 0; ci < C; ci++) {
                            const xc = xo + ci * H * Wd, wc = wo + ci * KH * KW;
                            for (let ky = 0; ky < KH; ky++) {
                                const iy = oy * st - pH + ky; if (iy < 0 || iy >= H) continue;
                                const xr = xc + iy * Wd, wr = wc + ky * KW;
                                for (let kx = 0; kx < KW; kx++) {
                                    const ix = ox * st - pW + kx; if (ix < 0 || ix >= Wd) continue;
                                    if (gw) gw[wr + kx] += gv * x[xr + ix];
                                    if (gx) gx[xr + ix] += gv * w[wr + kx];
                                }
                            }
                        }
                    }
                }
            }
        });
    };
    /** 2×2 平均池化（H、W 须为偶数）：[N × C·H·W] → [N × C·(H/2)·(W/2)] */
    AG.avgPool2 = function (X, o) {
        const N = X.r, C = o.C, H = o.H, W = o.W, h = H >> 1, w = W >> 1, Y = new Tensor(N, C * h * w), x = X.data, y = Y.data;
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
            const b = n * C * H * W + c * H * W + 2 * i * W + 2 * j;
            y[n * C * h * w + (c * h + i) * w + j] = 0.25 * (x[b] + x[b + 1] + x[b + W] + x[b + W + 1]);
        }
        Y.shape = { C, H: h, W: w };
        return AG.track(Y, [X], (g) => {
            const gx = X.g();
            for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
                const b = n * C * H * W + c * H * W + 2 * i * W + 2 * j, v = 0.25 * g[n * C * h * w + (c * h + i) * w + j];
                gx[b] += v; gx[b + 1] += v; gx[b + W] += v; gx[b + W + 1] += v;
            }
        });
    };
    /** 2×2 最大池化：梯度只回到每个窗口里最大的那个位置 */
    AG.maxPool2 = function (X, o) {
        const N = X.r, C = o.C, H = o.H, W = o.W, h = H >> 1, w = W >> 1, Y = new Tensor(N, C * h * w), x = X.data, y = Y.data, arg = new Int32Array(N * C * h * w);
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
            const b = n * C * H * W + c * H * W + 2 * i * W + 2 * j, cand = [b, b + 1, b + W, b + W + 1];
            let best = cand[0]; for (let k = 1; k < 4; k++) if (x[cand[k]] > x[best]) best = cand[k];
            const oi = n * C * h * w + (c * h + i) * w + j; y[oi] = x[best]; arg[oi] = best;
        }
        Y.shape = { C, H: h, W: w };
        return AG.track(Y, [X], (g) => { const gx = X.g(); for (let i = 0; i < g.length; i++) gx[arg[i]] += g[i]; });
    };
    /** 最近邻 2× 上采样：[N × C·H·W] → [N × C·2H·2W]（U-Net 解码器） */
    AG.upsample2 = function (X, o) {
        const N = X.r, C = o.C, H = o.H, W = o.W, H2 = 2 * H, W2 = 2 * W, Y = new Tensor(N, C * H2 * W2), x = X.data, y = Y.data;
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let i = 0; i < H2; i++) for (let j = 0; j < W2; j++)
            y[n * C * H2 * W2 + (c * H2 + i) * W2 + j] = x[n * C * H * W + (c * H + (i >> 1)) * W + (j >> 1)];
        Y.shape = { C, H: H2, W: W2 };
        return AG.track(Y, [X], (g) => {
            const gx = X.g();
            for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let i = 0; i < H2; i++) for (let j = 0; j < W2; j++)
                gx[n * C * H * W + (c * H + (i >> 1)) * W + (j >> 1)] += g[n * C * H2 * W2 + (c * H2 + i) * W2 + j];
        });
    };
    /** 按通道广播加：X [N × C·HW] + B [N × C]（每张图每个通道一个数：扩散 U-Net 里加时间步 embedding） */
    AG.addChannel = function (X, B, o) {
        const N = X.r, C = o.C, HW = X.c / C, Y = new Tensor(N, X.c), x = X.data, y = Y.data, bb = B.data;
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const v = bb[n * C + c], o0 = n * X.c + c * HW; for (let k = 0; k < HW; k++) y[o0 + k] = x[o0 + k] + v; }
        return AG.track(Y, [X, B], (g) => {
            if (X.req) { const gx = X.g(); for (let i = 0; i < g.length; i++) gx[i] += g[i]; }
            if (B.req) { const gb = B.g(); for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { let s = 0; const o0 = n * X.c + c * HW; for (let k = 0; k < HW; k++) s += g[o0 + k]; gb[n * C + c] += s; } }
        });
    };
    /** 按通道广播乘：X [N × C·HW] ⊙ S [N × C]（FiLM 式条件调制） */
    AG.mulChannel = function (X, Sc, o) {
        const N = X.r, C = o.C, HW = X.c / C, Y = new Tensor(N, X.c), x = X.data, y = Y.data, s = Sc.data;
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const v = s[n * C + c], o0 = n * X.c + c * HW; for (let k = 0; k < HW; k++) y[o0 + k] = x[o0 + k] * v; }
        return AG.track(Y, [X, Sc], (g) => {
            if (X.req) { const gx = X.g(); for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const v = s[n * C + c], o0 = n * X.c + c * HW; for (let k = 0; k < HW; k++) gx[o0 + k] += g[o0 + k] * v; } }
            if (Sc.req) { const gs = Sc.g(); for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { let t = 0; const o0 = n * X.c + c * HW; for (let k = 0; k < HW; k++) t += g[o0 + k] * x[o0 + k]; gs[n * C + c] += t; } }
        });
    };
    /** 常用初始化：He / Kaiming（ReLU 类）与 Xavier / Glorot。fanIn = 输入维度（卷积是 C·KH·KW） */
    LM.init = {
        he: (rand, fanIn) => (a) => { const s = Math.sqrt(2 / fanIn); for (let i = 0; i < a.length; i++) a[i] = s * LM.randn(rand); },
        xavier: (rand, fanIn, fanOut) => (a) => { const s = Math.sqrt(2 / (fanIn + fanOut)); for (let i = 0; i < a.length; i++) a[i] = s * LM.randn(rand); },
        normal: (rand, std) => (a) => { for (let i = 0; i < a.length; i++) a[i] = std * LM.randn(rand); },
        uniform: (rand, lim) => (a) => { for (let i = 0; i < a.length; i++) a[i] = (2 * rand() - 1) * lim; },
    };

    // ═════════════════════════════════ 优化器 ═════════════════════════════════
    LM.zeroGrad = (params) => params.forEach((p) => { if (p.grad) p.grad.fill(0); });
    /** global L2 norm of all gradients; scales them down to maxNorm if larger. Returns the norm before clipping. */
    LM.clipGradNorm = function (params, maxNorm) {
        let s = 0;
        params.forEach((p) => { if (p.grad) for (let i = 0; i < p.grad.length; i++) s += p.grad[i] * p.grad[i]; });
        const nrm = Math.sqrt(s);
        if (maxNorm && nrm > maxNorm) { const k = maxNorm / (nrm + 1e-6); params.forEach((p) => { if (p.grad) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= k; }); }
        return nrm;
    };
    LM.gradNorm = (p) => { if (!p.grad) return 0; let s = 0; for (let i = 0; i < p.grad.length; i++) s += p.grad[i] ** 2; return Math.sqrt(s); };
    LM.rms = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / Math.max(1, a.length)); };
    /** AdamW（解耦权重衰减，Loshchilov & Hutter）。p.decay === true 的参数才衰减。step() 后 this.updRatio[k] = ‖Δp‖/‖p‖ */
    class AdamW {
        constructor(params, o) {
            o = o || {};
            this.params = params; this.lr = o.lr ?? 1e-3; this.b1 = o.b1 ?? 0.9; this.b2 = o.b2 ?? 0.95; this.eps = o.eps ?? 1e-8; this.wd = o.wd ?? 0.1;
            this.m = params.map((p) => new Float32Array(p.size)); this.v = params.map((p) => new Float32Array(p.size)); this.t = 0;
            this.updRatio = new Float32Array(params.length);
        }
        step(lr) {
            lr = lr ?? this.lr; this.t++;
            const b1 = this.b1, b2 = this.b2, bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
            this.params.forEach((p, k) => {
                if (!p.grad) return;
                const g = p.grad, m = this.m[k], v = this.v[k], d = p.data, wd = p.decay ? this.wd : 0;
                let du = 0, dp = 0;
                for (let i = 0; i < d.length; i++) {
                    m[i] = b1 * m[i] + (1 - b1) * g[i];
                    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
                    const u = lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + this.eps) + wd * d[i]);
                    d[i] -= u; du += u * u; dp += d[i] * d[i];
                }
                this.updRatio[k] = Math.sqrt(du) / (Math.sqrt(dp) + 1e-12);
            });
        }
        zeroGrad() { LM.zeroGrad(this.params); }
    }
    LM.AdamW = AdamW;
    /** SGD（可选 momentum）：v ← μv + g；p ← p − lr·v */
    class SGD {
        constructor(params, o) { o = o || {}; this.params = params; this.lr = o.lr ?? 0.1; this.mom = o.momentum ?? 0; this.v = params.map((p) => new Float32Array(p.size)); }
        step(lr) {
            lr = lr ?? this.lr;
            this.params.forEach((p, k) => { if (!p.grad) return; const v = this.v[k], g = p.grad, d = p.data; for (let i = 0; i < d.length; i++) { v[i] = this.mom * v[i] + g[i]; d[i] -= lr * v[i]; } });
        }
        zeroGrad() { LM.zeroGrad(this.params); }
    }
    LM.SGD = SGD;
    /** warmup（线性）+ cosine 衰减到 minLR：与 nanoGPT 的 get_lr 相同 */
    LM.cosineLR = function (it, o) {
        const { maxLR, minLR, warmup, total } = o;
        if (it < warmup) return maxLR * (it + 1) / warmup;
        if (it >= total) return minLR;
        const r = (it - warmup) / Math.max(1, total - warmup);
        return minLR + 0.5 * (1 + Math.cos(Math.PI * r)) * (maxLR - minLR);
    };

    // ═════════════════════════════════ 迷你 GPT ═════════════════════════════════
    const gpt = {};
    LM.gpt = gpt;
    gpt.defaultConfig = { V: 0, T: 32, d: 32, L: 2, H: 4, ff: 128 };
    /** 参数初始化同 GPT-2：N(0, 0.02)，残差输出投影再除以 √(2L)；LayerNorm γ=1 β=0；偏置 0 */
    gpt.create = function (cfg, seed) {
        cfg = Object.assign({}, gpt.defaultConfig, cfg);
        const rand = LM.rng(seed || 1), params = [], std = cfg.std || 0.02, rstd = std / Math.sqrt(2 * cfg.L);
        const P = (name, r, c, init, decay) => { const t = AG.param(name, r, c, init); t.decay = !!decay; params.push(t); return t; };
        const normal = (s) => (a) => { for (let i = 0; i < a.length; i++) a[i] = s * LM.randn(rand); };
        const ones = (a) => a.fill(1), zeros = null;
        const { V, T, d, L, ff } = cfg;
        const m = { cfg, params, layers: [] };
        m.wte = P('wte', V, d, normal(std), true);
        m.wpe = P('wpe', T, d, normal(std), true);
        for (let l = 0; l < L; l++) {
            const n = `h${l}.`;
            m.layers.push({
                ln1g: P(n + 'ln1.γ', 1, d, ones), ln1b: P(n + 'ln1.β', 1, d, zeros),
                wq: P(n + 'attn.Wq', d, d, normal(std), true), bq: P(n + 'attn.bq', 1, d, zeros),
                wk: P(n + 'attn.Wk', d, d, normal(std), true), bk: P(n + 'attn.bk', 1, d, zeros),
                wv: P(n + 'attn.Wv', d, d, normal(std), true), bv: P(n + 'attn.bv', 1, d, zeros),
                wo: P(n + 'attn.Wo', d, d, normal(rstd), true), bo: P(n + 'attn.bo', 1, d, zeros),
                ln2g: P(n + 'ln2.γ', 1, d, ones), ln2b: P(n + 'ln2.β', 1, d, zeros),
                w1: P(n + 'mlp.W1', d, ff, normal(std), true), b1: P(n + 'mlp.b1', 1, ff, zeros),
                w2: P(n + 'mlp.W2', ff, d, normal(rstd), true), b2: P(n + 'mlp.b2', 1, d, zeros),
            });
        }
        m.lnfg = P('ln_f.γ', 1, d, ones); m.lnfb = P('ln_f.β', 1, d, zeros);
        return m;
    };
    gpt.numParams = (m) => m.params.reduce((s, p) => s + p.size, 0);
    const posCache = new Map();
    function posIdx(B, T) { const key = B * 4096 + T; let a = posCache.get(key); if (!a) { a = new Int32Array(B * T); for (let i = 0; i < B * T; i++) a[i] = i % T; posCache.set(key, a); } return a; }
    /**
     * 前向。ids: Int32Array(B·T)。o: { targets, weights, denom, trace, inputEmb（[B·T × d]，替代 token embedding，多模态前缀用）, keepProbs }
     * 返回 { logits [B·T × V], loss?, trace? }。trace = { x0, layers: [{ h1, q, k, v, att, a, ao, x1, h2, f1, f2, x2 }], hf }
     */
    gpt.forward = function (m, ids, B, T, o) {
        o = o || {};
        const c = m.cfg;
        const tok = o.inputEmb || AG.embed(ids, m.wte);
        let x = AG.add(tok, AG.gather(m.wpe, posIdx(B, T)));
        const tr = o.trace ? { tok, x0: x, layers: [] } : null;
        for (let l = 0; l < c.L; l++) {
            const p = m.layers[l];
            const h1 = AG.layernorm(x, p.ln1g, p.ln1b);
            const q = AG.linear(h1, p.wq, p.bq), k = AG.linear(h1, p.wk, p.bk), v = AG.linear(h1, p.wv, p.bv);
            const a = AG.attention(q, k, v, B, T, c.H, true);
            const ao = AG.linear(a, p.wo, p.bo);
            const x1 = AG.add(x, ao);
            const h2 = AG.layernorm(x1, p.ln2g, p.ln2b);
            const f1 = AG.gelu(AG.linear(h2, p.w1, p.b1));
            const f2 = AG.linear(f1, p.w2, p.b2);
            const x2 = AG.add(x1, f2);
            if (tr) tr.layers.push({ xin: x, h1, q, k, v, att: a.probs, a, ao, x1, h2, f1, f2, x2 });
            x = x2;
        }
        const hf = AG.layernorm(x, m.lnfg, m.lnfb);
        const logits = AG.matmulBT(hf, m.wte);
        const out = { logits, trace: tr };
        if (tr) tr.hf = hf;
        if (o.targets) out.loss = AG.crossEntropy(logits, o.targets, o);
        return out;
    };
    /** 一步训练：前向 + 反向 + 梯度裁剪 + AdamW。返回 { loss, gradNorm, lr } */
    gpt.trainStep = function (m, opt, x, y, B, T, lr, o) {
        o = o || {};
        opt.zeroGrad();
        const out = gpt.forward(m, x, B, T, Object.assign({ targets: y }, o));
        AG.backward(out.loss);
        const gn = LM.clipGradNorm(m.params, o.clip ?? 1.0);
        opt.step(lr);
        return { loss: out.loss.item(), gradNorm: gn, lr, out };
    };
    /** 随机取 B 个长度 T+1 的窗口：x = 窗口[0..T)，y = 窗口[1..T]（右移一位 = 下一个 token） */
    gpt.batch = function (data, B, T, rand, out) {
        out = out || { x: new Int32Array(B * T), y: new Int32Array(B * T), starts: new Int32Array(B) };
        for (let b = 0; b < B; b++) {
            const s = Math.floor(rand() * (data.length - T - 1)); out.starts[b] = s;
            for (let t = 0; t < T; t++) { out.x[b * T + t] = data[s + t]; out.y[b * T + t] = data[s + t + 1]; }
        }
        return out;
    };
    /** 在 ids 末尾的下一个 token 上的 logits（无梯度；超出上下文时只看最后 T 个） */
    gpt.nextLogits = function (m, ids) {
        const T = Math.min(ids.length, m.cfg.T), x = Int32Array.from(ids.slice(ids.length - T));
        return AG.noGrad(() => { const out = gpt.forward(m, x, 1, T); return out.logits.data.slice((T - 1) * m.cfg.V, T * m.cfg.V); });
    };
    /** 解码分布：temperature → top-k → top-p（nucleus），返回概率（未选中的为 0） */
    gpt.decodeDist = function (logits, o) {
        o = o || {};
        const V = logits.length, temp = Math.max(1e-4, o.temperature ?? 1), p = LM.softmax(logits, new Float32Array(V), temp);
        const order = Array.from(p, (_, i) => i).sort((a, b) => p[b] - p[a]);
        const keep = new Uint8Array(V);
        let cum = 0;
        for (let r = 0; r < V; r++) {
            const i = order[r];
            if (o.topK && r >= o.topK) break;
            keep[i] = 1; cum += p[i];
            if (o.topP && o.topP < 1 && cum >= o.topP) break;
        }
        let s = 0; for (let i = 0; i < V; i++) { if (!keep[i]) p[i] = 0; s += p[i]; }
        for (let i = 0; i < V; i++) p[i] /= s;
        return p;
    };
    LM.sampleFrom = function (p, rand) { let u = rand(), i = 0; for (; i < p.length - 1; i++) { u -= p[i]; if (u <= 0) break; } return i; };
    /**
     * 自回归生成（带 KV cache：每个新 token 只算自己的 q/k/v，复用之前所有位置的 k/v）。
     * o: { temperature, topK, topP, rand, greedy, stop: Set(ids), keepDists, onStep(dist, id, logits) }。超出上下文长度时停止。
     * 注意：传给 onStep 的 logits 是复用的缓冲区，要保留就先复制（Float32Array.from(logits)）。
     */
    gpt.generate = function (m, prompt, maxNew, o) {
        o = o || {};
        const c = m.cfg, d = c.d, H = c.H, dh = d / H, V = c.V, Tm = c.T, rand = o.rand || Math.random;
        const kc = m.layers.map(() => new Float32Array(Tm * d)), vc = m.layers.map(() => new Float32Array(Tm * d));
        const ids = Array.from(prompt), xbuf = new Float32Array(d), hbuf = new Float32Array(d), tmp = new Float32Array(Math.max(d, c.ff)), att = new Float32Array(Tm);
        const lnRow = (x, g, b, out) => { let mu = 0; for (let j = 0; j < d; j++) mu += x[j]; mu /= d; let v = 0; for (let j = 0; j < d; j++) v += (x[j] - mu) ** 2; v /= d; const r = 1 / Math.sqrt(v + 1e-5); for (let j = 0; j < d; j++) out[j] = (x[j] - mu) * r * g[j] + b[j]; };
        const lin = (x, W, b, n, mOut, out) => { for (let j = 0; j < mOut; j++) out[j] = b ? b[j] : 0; for (let p = 0; p < n; p++) { const xv = x[p]; if (xv === 0) continue; const o2 = p * mOut; for (let j = 0; j < mOut; j++) out[j] += xv * W[o2 + j]; } return out; };
        const qv = new Float32Array(d), kv = new Float32Array(d), vv = new Float32Array(d), av = new Float32Array(d), fv = new Float32Array(c.ff), logits = new Float32Array(V);
        const dists = [];
        const step = (tok, pos) => {
            const E = m.wte.data, P = m.wpe.data;
            for (let j = 0; j < d; j++) xbuf[j] = E[tok * d + j] + P[pos * d + j];
            m.layers.forEach((p, l) => {
                lnRow(xbuf, p.ln1g.data, p.ln1b.data, hbuf);
                lin(hbuf, p.wq.data, p.bq.data, d, d, qv); lin(hbuf, p.wk.data, p.bk.data, d, d, kv); lin(hbuf, p.wv.data, p.bv.data, d, d, vv);
                kc[l].set(kv, pos * d); vc[l].set(vv, pos * d);
                av.fill(0);
                for (let h = 0; h < H; h++) {
                    const hc = h * dh; let mx = -Infinity;
                    for (let j = 0; j <= pos; j++) { let s = 0; for (let e = 0; e < dh; e++) s += qv[hc + e] * kc[l][j * d + hc + e]; s /= Math.sqrt(dh); att[j] = s; if (s > mx) mx = s; }
                    let z = 0; for (let j = 0; j <= pos; j++) { att[j] = Math.exp(att[j] - mx); z += att[j]; }
                    for (let j = 0; j <= pos; j++) { const w = att[j] / z; for (let e = 0; e < dh; e++) av[hc + e] += w * vc[l][j * d + hc + e]; }
                }
                lin(av, p.wo.data, p.bo.data, d, d, tmp); for (let j = 0; j < d; j++) xbuf[j] += tmp[j];
                lnRow(xbuf, p.ln2g.data, p.ln2b.data, hbuf);
                lin(hbuf, p.w1.data, p.b1.data, d, c.ff, fv);
                for (let j = 0; j < c.ff; j++) { const x = fv[j]; fv[j] = 0.5 * x * (1 + Math.tanh(GK * (x + 0.044715 * x * x * x))); }
                lin(fv, p.w2.data, p.b2.data, c.ff, d, tmp); for (let j = 0; j < d; j++) xbuf[j] += tmp[j];
            });
            lnRow(xbuf, m.lnfg.data, m.lnfb.data, hbuf);
            const E2 = m.wte.data;
            for (let v = 0; v < V; v++) { let s = 0; for (let j = 0; j < d; j++) s += hbuf[j] * E2[v * d + j]; logits[v] = s; }
            return logits;
        };
        let lg = null;
        for (let t = 0; t < ids.length && t < Tm; t++) lg = step(ids[t], t);
        const gen = [];
        for (let k = 0; k < maxNew && ids.length < Tm; k++) {
            const dist = o.greedy ? null : gpt.decodeDist(lg, o);
            const id = o.greedy ? LM.argmax(lg) : LM.sampleFrom(dist, rand);
            if (o.onStep) o.onStep(dist || LM.softmax(lg), id, lg);
            if (o.keepDists) dists.push(dist || LM.softmax(lg));
            ids.push(id); gen.push(id);
            if (o.stop && o.stop.has(id)) break;
            if (ids.length < Tm) lg = step(id, ids.length - 1);
        }
        return { ids, gen, dists };
    };
    // ── 权重序列化：float16 → base64（离线训练好的权重内嵌进页面） ──
    const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
    function toHalf(v) {
        f32[0] = v; const x = u32[0], s = (x >>> 16) & 0x8000; let e = ((x >>> 23) & 0xff) - 127 + 15, mt = x & 0x7fffff;
        if (e <= 0) { if (e < -10) return s; mt = (mt | 0x800000) >> (1 - e); return s | ((mt + 0x1000) >> 13); }
        if (e >= 31) return s | 0x7c00;
        const h = s | (e << 10) | (mt >> 13); return (mt & 0x1000) ? h + 1 : h;
    }
    function fromHalf(h) {
        const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
        if (e === 0) return s * f * Math.pow(2, -24);
        if (e === 31) return f ? NaN : s * Infinity;
        return s * (1 + f / 1024) * Math.pow(2, e - 15);
    }
    const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    function bytesToB64(bytes) { let s = ''; for (let i = 0; i < bytes.length; i += 3) { const a = bytes[i], b = bytes[i + 1] ?? 0, c = bytes[i + 2] ?? 0, n = (a << 16) | (b << 8) | c; s += B64[(n >> 18) & 63] + B64[(n >> 12) & 63] + (i + 1 < bytes.length ? B64[(n >> 6) & 63] : '=') + (i + 2 < bytes.length ? B64[n & 63] : '='); } return s; }
    function b64ToBytes(s) { const map = new Int16Array(128).fill(-1); for (let i = 0; i < 64; i++) map[B64.charCodeAt(i)] = i; const clean = s.replace(/[^A-Za-z0-9+/]/g, ''); const out = new Uint8Array(Math.floor(clean.length * 3 / 4)); let o = 0; for (let i = 0; i < clean.length; i += 4) { const n = (map[clean.charCodeAt(i)] << 18) | (map[clean.charCodeAt(i + 1)] << 12) | ((map[clean.charCodeAt(i + 2)] & 63) << 6) | (map[clean.charCodeAt(i + 3)] & 63); out[o++] = (n >> 16) & 255; if (o < out.length) out[o++] = (n >> 8) & 255; if (o < out.length) out[o++] = n & 255; } return out; }
    gpt.exportWeights = function (m) {
        const n = gpt.numParams(m), bytes = new Uint8Array(n * 2); let o = 0;
        m.params.forEach((p) => { for (let i = 0; i < p.size; i++) { const h = toHalf(p.data[i]); bytes[o++] = h & 255; bytes[o++] = h >> 8; } });
        return bytesToB64(bytes);
    };
    gpt.loadWeights = function (m, b64) {
        const bytes = b64ToBytes(b64); let o = 0;
        m.params.forEach((p) => { for (let i = 0; i < p.size; i++) { p.data[i] = fromHalf(bytes[o] | (bytes[o + 1] << 8)); o += 2; } });
        if (o !== bytes.length) throw new Error(`weight size mismatch: model ${o} bytes, blob ${bytes.length}`);
        return m;
    };
    gpt.clone = function (m) { const c = gpt.create(m.cfg, 1); c.params.forEach((p, i) => p.data.set(m.params[i].data)); return c; };

    // ═════════════════════════════════ 玩具世界语料 + 字符级 tokenizer ═════════════════════════════════
    // 8 种动物 × 3 个属性（爱吃 / 住处 / 颜色）= 24 条「世界知识」。预训练语料用 10 个句式反复陈述这些事实；
    // 其中两个句式需要把前文的名字或食物「抄」过来（所以…每天都吃…；它住在…）——这正是注意力要学会的事。
    const world = {
        animals: [
            { name: '小猫', food: '鱼', home: '屋里', color: '白色' },
            { name: '小狗', food: '骨头', home: '院子', color: '黄色' },
            { name: '小兔', food: '萝卜', home: '草地', color: '灰色' },
            { name: '小鸟', food: '虫子', home: '树上', color: '蓝色' },
            { name: '熊猫', food: '竹子', home: '山里', color: '黑白' },
            { name: '猴子', food: '香蕉', home: '森林', color: '棕色' },
            { name: '小羊', food: '青草', home: '山坡', color: '白色' },
            { name: '老虎', food: '肉', home: '森林', color: '橙色' },
        ],
        templates: [
            (a) => `${a.name}喜欢吃${a.food}。`,
            (a) => `${a.name}住在${a.home}。`,
            (a) => `${a.name}是${a.color}的。`,
            (a) => `${a.name}最爱吃${a.food}，它住在${a.home}。`,
            (a) => `${a.name}喜欢吃${a.food}，所以${a.name}每天都吃${a.food}。`,
            (a) => `${a.home}住着${a.name}。`,
            (a) => `今天${a.name}在${a.home}吃了很多${a.food}。`,
            (a, b) => `${a.name}和${b.name}是好朋友。`,
            (a, b) => `${a.name}吃${a.food}，${b.name}吃${b.food}。`,
            (a) => `${a.color}的${a.name}喜欢吃${a.food}。`,
        ],
        /** 问答（SFT / 对齐用）：问题 → 标准答案 */
        questions: [
            { q: (a) => `${a.name}喜欢吃什么？`, a: (a) => `${a.name}喜欢吃${a.food}。` },
            { q: (a) => `${a.name}住在哪里？`, a: (a) => `${a.name}住在${a.home}。` },
            { q: (a) => `${a.name}是什么颜色的？`, a: (a) => `${a.name}是${a.color}的。` },
        ],
    };
    world.sentence = function (rand, tplIndex) {
        const A = world.animals, a = A[Math.floor(rand() * A.length)];
        let b = A[Math.floor(rand() * A.length)]; if (b === a) b = A[(A.indexOf(a) + 1) % A.length];
        const k = tplIndex ?? Math.floor(rand() * world.templates.length);
        return world.templates[k](a, b);
    };
    world.corpus = function (rand, nSentences) { let s = ''; for (let i = 0; i < nSentences; i++) s += world.sentence(rand); return s; };
    LM.world = world;

    /** 字符级词表：4 个特殊 token + 玩具世界用到的全部汉字 / 标点（按首次出现排序，确定性） */
    function buildVocab() {
        const specials = ['<pad>', '<eos>', '<user>', '<bot>'];
        const seen = new Set(), chars = [];
        const add = (s) => { for (const ch of s) if (!seen.has(ch)) { seen.add(ch); chars.push(ch); } };
        world.animals.forEach((a) => { add(a.name); add(a.food); add(a.home); add(a.color); });
        const A = world.animals;
        world.templates.forEach((t) => add(t(A[0], A[1])));
        world.questions.forEach((q) => { add(q.q(A[0])); add(q.a(A[0])); });
        const itos = specials.concat(chars), stoi = new Map(itos.map((s, i) => [s, i]));
        return {
            itos, stoi, size: itos.length, specials: { pad: 0, eos: 1, user: 2, bot: 3 },
            /** string → ids（特殊 token 写成 <eos> 等） */
            encode(s) {
                const out = []; let i = 0;
                while (i < s.length) {
                    if (s[i] === '<') { const j = s.indexOf('>', i); const tok = j > 0 ? s.slice(i, j + 1) : null; if (tok && stoi.has(tok)) { out.push(stoi.get(tok)); i = j + 1; continue; } }
                    const ch = s[i]; if (!stoi.has(ch)) throw new Error('char not in vocab: ' + ch); out.push(stoi.get(ch)); i++;
                }
                return Int32Array.from(out);
            },
            decode(ids) { let s = ''; for (const i of ids) s += itos[i]; return s; },
            /** 显示用：特殊 token 缩写成 ⟨eos⟩ */
            show(i) { const t = itos[i]; return t && t.length > 1 ? '⟨' + t.slice(1, -1) + '⟩' : t; },
        };
    }
    LM.vocab = buildVocab();

    /** 一个通用的字符级词表构造器（多模态页自己的 caption 词表等） */
    LM.makeVocab = function (specials, text) {
        const seen = new Set(specials), itos = specials.slice();
        for (const ch of text) if (!seen.has(ch)) { seen.add(ch); itos.push(ch); }
        const stoi = new Map(itos.map((s, i) => [s, i]));
        return { itos, stoi, size: itos.length, encode: (s) => Int32Array.from(Array.from(s, (ch) => stoi.get(ch))), decode: (ids) => Array.from(ids, (i) => itos[i]).join('') };
    };

    // ═════════════════════════════════ 程序化手写数字（与第 02 篇同一个生成器，第 03、04 篇复用） ═════════════════════════════════
    // 16×16 灰度（0..1），字形 = 笔画折线 + 随机仿射 / 粗细 / 手抖 / 噪声。MNIST（28×28、6 万张）的缩小替身。
    LM.digits = (() => {
        const IMG = 16, D = IMG * IMG, K = 10;
        const arc = (cx, cy, rx, ry, a0, a1) => {
            const n = Math.max(4, Math.ceil(Math.abs(a1 - a0) / 15)), pts = [];
            for (let i = 0; i <= n; i++) { const a = (a0 + (a1 - a0) * i / n) * Math.PI / 180; pts.push([cx + rx * Math.cos(a), cy + ry * Math.sin(a)]); }
            return pts;
        };
        const line = (x0, y0, x1, y1) => [[x0, y0], [x1, y1]];
        const quad = (x0, y0, x1, y1, x2, y2) => { const pts = []; for (let i = 0; i <= 8; i++) { const t = i / 8, s = 1 - t; pts.push([s * s * x0 + 2 * s * t * x1 + t * t * x2, s * s * y0 + 2 * s * t * y1 + t * t * y2]); } return pts; };
        function strokes(d, r) {
            const j = (x, a) => x + (2 * r() - 1) * a;
            switch (d) {
                case 0: { const a0 = j(-90, 40); return [arc(j(0.5, 0.03), j(0.5, 0.02), j(0.22, 0.07), j(0.36, 0.03), a0, a0 + j(360, 20))]; }
                case 1: {
                    const tx = j(0.53, 0.08), ty = j(0.12, 0.03), s = [line(tx, ty, j(0.49, 0.08), j(0.88, 0.03))];
                    if (r() < 0.6) s.push(line(j(0.34, 0.06), j(0.3, 0.06), tx, ty));
                    if (r() < 0.25) s.push(line(j(0.34, 0.04), 0.88, j(0.68, 0.04), 0.88));
                    return s;
                }
                case 2: {
                    const a = arc(j(0.5, 0.03), j(0.34, 0.03), j(0.24, 0.04), j(0.21, 0.04), j(195, 15), j(380, 25)), e = a[a.length - 1];
                    const bx = j(0.24, 0.05), by = j(0.86, 0.03), s = [a, line(e[0], e[1], bx, by), line(bx, by, j(0.8, 0.06), j(0.86, 0.04))];
                    if (r() < 0.3) s.push(arc(bx + 0.06, by - 0.06, 0.07, 0.06, 90, 450));
                    return s;
                }
                case 3: {
                    const low = arc(j(0.49, 0.03), j(0.67, 0.03), j(0.25, 0.04), j(0.19, 0.03), j(270, 10), j(515, 20));
                    if (r() < 0.25) { const m = low[0]; return [line(j(0.28, 0.04), j(0.14, 0.03), j(0.72, 0.04), 0.14), line(0.72, 0.14, m[0], m[1]), low]; }
                    return [arc(j(0.49, 0.03), j(0.31, 0.03), j(0.22, 0.04), j(0.18, 0.03), j(205, 20), j(450, 10)), low];
                }
                case 4: {
                    if (r() < 0.65) { const x = j(0.64, 0.05), lx = j(0.2, 0.05), ly = j(0.62, 0.05); return [line(x, 0.12, j(x, 0.04), 0.88), line(x, 0.12, lx, ly), line(lx, ly, j(0.84, 0.05), ly)]; }
                    const ly = j(0.58, 0.05); return [line(j(0.36, 0.05), 0.12, 0.26, ly), line(0.26, ly, j(0.82, 0.05), ly), line(j(0.66, 0.05), j(0.3, 0.06), j(0.64, 0.05), 0.88)];
                }
                case 5: {
                    const tx = j(0.34, 0.04), ty = j(0.14, 0.03), mx = j(0.3, 0.04), my = j(0.5, 0.04);
                    return [line(j(0.74, 0.05), j(ty, 0.03), tx, ty), line(tx, ty, mx, my), arc(j(0.5, 0.03), j(0.64, 0.03), j(0.25, 0.04), j(0.21, 0.03), j(215, 15), j(485, 25))];
                }
                case 6: {
                    const ex = j(0.27, 0.03), ey = j(0.64, 0.04);
                    return [quad(j(0.7, 0.06), j(0.13, 0.03), j(0.3, 0.06), j(0.22, 0.05), ex, ey), arc(j(0.5, 0.03), j(0.66, 0.03), j(0.23, 0.04), j(0.2, 0.04), 0, 360)];
                }
                case 7: {
                    const rx = j(0.78, 0.05), ry = j(0.14, 0.03), bx = j(0.42, 0.1);
                    const s = [line(j(0.22, 0.05), j(0.14, 0.03), rx, ry), r() < 0.7 ? line(rx, ry, bx, 0.88) : quad(rx, ry, j(0.5, 0.06), 0.45, bx, 0.88)];
                    if (r() < 0.2) s.push(line(j(0.4, 0.04), 0.5, j(0.72, 0.04), 0.5));
                    return s;
                }
                case 8: return [arc(j(0.5, 0.03), j(0.3, 0.02), j(0.19, 0.04), j(0.17, 0.03), 0, 360), arc(j(0.5, 0.03), j(0.68, 0.02), j(0.23, 0.04), j(0.19, 0.03), 0, 360)];
                default: {
                    const cx = j(0.5, 0.03), cy = j(0.33, 0.03), rx = j(0.22, 0.04), sx = cx + rx;
                    return [arc(cx, cy, rx, j(0.2, 0.03), 0, 360), r() < 0.6 ? line(sx, cy, j(0.64, 0.08), 0.88) : quad(sx, cy, j(0.74, 0.05), 0.75, j(0.42, 0.06), 0.86)];
                }
            }
        }
        /** 渲染数字 d 到 out[256]（0..1）；o.clean = true 时不加噪点与笔触纹理 */
        function render(d, r, out, o) {
            o = o || {}; out = out || new Float32Array(D);
            const pls = strokes(d, r);
            const rot = (2 * r() - 1) * 0.25, sc = 0.82 + 0.3 * r(), asp = 0.85 + 0.3 * r(), sh = (2 * r() - 1) * 0.35;
            const tx = (2 * r() - 1) * 0.08 + (o.dx || 0), ty = (2 * r() - 1) * 0.07 + (o.dy || 0);
            const wa = 0.015 + 0.035 * r(), f1 = 0.6 + 1.0 * r(), f2 = 0.6 + 1.0 * r(), p1 = 6.2832 * r(), p2 = 6.2832 * r();
            const c = Math.cos(rot), s = Math.sin(rot), seg = [];
            for (const pl of pls) {
                const ox = 0.02 * LM.randn(r), oy = 0.02 * LM.randn(r);
                let px = 0, py = 0;
                pl.forEach(([u, v], i) => {
                    let x = u + wa * Math.sin(6.2832 * f1 * v + p1) + ox - 0.5, y = v + wa * Math.sin(6.2832 * f2 * u + p2) + oy - 0.5;
                    x += sh * y; x *= sc * asp; y *= sc / asp;
                    const X = (0.5 + c * x - s * y + tx) * IMG, Y = (0.5 + s * x + c * y + ty) * IMG;
                    if (i) seg.push(px, py, X, Y);
                    px = X; py = Y;
                });
            }
            const rad = 0.55 + 0.9 * r(), peak = 0.75 + 0.25 * r(), tex = o.clean ? 0 : 0.12;
            for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
                const qx = x + 0.5, qy = y + 0.5;
                let best = 1e9;
                for (let k = 0; k < seg.length; k += 4) {
                    const ax = seg[k], ay = seg[k + 1], bx = seg[k + 2] - ax, by = seg[k + 3] - ay, l2 = bx * bx + by * by;
                    let t = l2 > 0 ? ((qx - ax) * bx + (qy - ay) * by) / l2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
                    const dx = qx - ax - t * bx, dy = qy - ay - t * by, d2 = dx * dx + dy * dy;
                    if (d2 < best) best = d2;
                }
                let v = peak * Math.min(1, Math.max(0, rad + 0.5 - Math.sqrt(best)));
                if (v > 0 && tex) v *= 1 + tex * LM.randn(r);
                if (!o.clean && r() < 0.012) v = Math.max(v, 0.15 + 0.4 * r());
                out[y * IMG + x] = v < 0 ? 0 : v > 1 ? 1 : v;
            }
            return out;
        }
        /** 数据集：每类样本数相同、顺序打乱。返回 { X: Float32Array(n·256), y: Int32Array(n), n } */
        function makeSet(n, r, o) {
            const y = new Int32Array(n), X = new Float32Array(n * D);
            for (let i = 0; i < n; i++) y[i] = i % K;
            LM.shuffle(y, r);
            for (let i = 0; i < n; i++) render(y[i], r, X.subarray(i * D, (i + 1) * D), o);
            return { X, y, n };
        }
        return { IMG, D, K, render, makeSet };
    })();

    // ═════════════════════════════════ 绘图辅助（深色主题，设计坐标） ═════════════════════════════════
    const draw = {};
    LM.draw = draw;
    const FONT = (px, bold) => `${bold ? 'bold ' : ''}${px}px Consolas, "Microsoft YaHei", "PingFang SC", monospace`;
    LM.FONT = FONT;
    const cm = (name) => (typeof name === 'function' ? name : (PL.cmap[name] || PL.cmap.viridis));
    const rgbStr = (c) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
    LM.rgbStr = rgbStr;
    /** colour for token id (stable hue per id) */
    LM.tokColor = function (id, alpha) { const h = (id * 137.508) % 360; return `hsla(${h.toFixed(0)}, 55%, ${alpha ? 40 : 34}%, ${alpha ?? 1})`; };
    function autoRange(data, o) {
        if (o.range) return o.range;
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < data.length; i++) { const v = data[i]; if (v === v && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
        if (!isFinite(lo)) { lo = 0; hi = 1; }
        if (o.symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; return [-m, m]; }
        if (hi - lo < 1e-12) hi = lo + 1e-12;
        return [lo, hi];
    }
    /**
     * 矩阵热图（行列可带文字标签，可在格子里写数值）。
     * o: { rowLabels, colLabels, map, range, symmetric, digits, hiRow, hiCol, hiCells:[[r,c]], title, titleX, legend, cellMax, padL, padT, colorFor, labelEvery }
     * 所有带 title 的绘图函数都支持 titleX：live 预览左上角有 LIVE 徽标，标题会被盖住时传 titleX: 46。
     * NaN 画成暗色斜纹（例如因果掩码遮住的位置）。返回布局（也存到 canvas._lmHeat，供 LM.draw.heatHit 用于 probe / pick）。
     */
    draw.heat = function (canvas, data, R, C, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(canvas, { bg: o.bg });
        const map = cm(o.map || (o.symmetric ? 'diverge' : 'viridis')), [lo, hi] = autoRange(data, o);
        ctx.font = FONT(o.labelPx || 11);
        const rl = o.rowLabels, cl = o.colLabels;
        const padL = o.padL ?? (rl ? Math.min(90, 8 + Math.max(...rl.map((s) => ctx.measureText(String(s)).width))) : 4);
        const padT = (o.title ? 16 : 2) + (o.padT ?? (cl ? 16 : 2));
        const padR = o.legend === false ? 4 : 46, padB = o.padB ?? 4;
        let cw = (w - padL - padR) / C, ch = (h - padT - padB) / R;
        if (o.square !== false) { const s = Math.min(cw, ch); cw = ch = s; }
        if (o.cellMax) { cw = Math.min(cw, o.cellMax); ch = Math.min(ch, o.cellMax); }
        const x0 = padL, y0 = padT, rgb = [0, 0, 0];
        if (o.title) { ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = FONT(11, true); ctx.fillText(o.title, o.titleX ?? 4, 2); ctx.font = FONT(o.labelPx || 11); }
        for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
            const v = data[r * C + c], X = x0 + c * cw, Y = y0 + r * ch;
            if (v !== v) { ctx.fillStyle = '#101216'; ctx.fillRect(X, Y, cw, ch); ctx.strokeStyle = '#1d2026'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X, Y + ch); ctx.lineTo(X + cw, Y); ctx.stroke(); continue; }
            const col = o.colorFor ? o.colorFor(v, r, c) : rgbStr(map((v - lo) / (hi - lo), rgb));
            ctx.fillStyle = col; ctx.fillRect(X, Y, Math.ceil(cw), Math.ceil(ch));
        }
        if (o.grid !== false && cw >= 6 && ch >= 6) { ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 1; for (let c = 0; c <= C; c++) { ctx.beginPath(); ctx.moveTo(x0 + c * cw, y0); ctx.lineTo(x0 + c * cw, y0 + R * ch); ctx.stroke(); } for (let r = 0; r <= R; r++) { ctx.beginPath(); ctx.moveTo(x0, y0 + r * ch); ctx.lineTo(x0 + C * cw, y0 + r * ch); ctx.stroke(); } }
        if (o.digits != null && cw >= 16 && ch >= 11) {
            ctx.font = FONT(Math.min(11, Math.max(8, ch * 0.5))); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
                const v = data[r * C + c]; if (v !== v) continue;
                const t = (v - lo) / (hi - lo); map(t, rgb); const lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
                ctx.fillStyle = lum > 140 ? '#000' : '#e8e8e8';
                const s = Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(o.digits);
                ctx.fillText(s.replace(/^(-?)0\./, '$1.'), x0 + (c + 0.5) * cw, y0 + (r + 0.5) * ch);
            }
        }
        ctx.font = FONT(o.labelPx || 11); ctx.fillStyle = LM.C.gray;
        if (rl) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; const step = o.labelEvery || Math.max(1, Math.ceil(11 / ch)); for (let r = 0; r < R; r += step) { ctx.fillStyle = r === o.hiRow ? LM.C.yellow : LM.C.gray; ctx.fillText(String(rl[r]), x0 - 4, y0 + (r + 0.5) * ch); } }
        if (cl) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; const step = o.labelEvery || Math.max(1, Math.ceil((ctx.measureText('国').width + 2) / cw)); for (let c = 0; c < C; c += step) { ctx.fillStyle = c === o.hiCol ? LM.C.yellow : LM.C.gray; ctx.fillText(String(cl[c]), x0 + (c + 0.5) * cw, y0 - 2); } }
        ctx.lineWidth = 2; ctx.strokeStyle = LM.C.yellow;
        if (o.hiRow != null && o.hiRow >= 0) ctx.strokeRect(x0 - 1, y0 + o.hiRow * ch, C * cw + 2, ch);
        if (o.hiCol != null && o.hiCol >= 0) ctx.strokeRect(x0 + o.hiCol * cw, y0 - 1, cw, R * ch + 2);
        (o.hiCells || []).forEach(([r, c, col]) => { ctx.strokeStyle = col || LM.C.green; ctx.strokeRect(x0 + c * cw + 1, y0 + r * ch + 1, cw - 2, ch - 2); });
        if (o.legend !== false) {
            const lx = x0 + C * cw + 8, ly = y0, lh = Math.max(30, R * ch);
            for (let i = 0; i < lh; i++) { ctx.fillStyle = rgbStr(map(1 - i / (lh - 1), rgb)); ctx.fillRect(lx, ly + i, 7, 1.5); }
            ctx.fillStyle = LM.C.dim; ctx.font = FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            const f = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2));
            ctx.fillText(f(hi), lx + 9, ly - 1); ctx.textBaseline = 'bottom'; ctx.fillText(f(lo), lx + 9, ly + lh + 1);
        }
        const lay = { x0, y0, cw, ch, R, C, lo, hi };
        canvas._lmHeat = lay;
        return lay;
    };
    /** probe/pick helper for heat(): design coords → { r, c } or null */
    draw.heatHit = function (canvas, info) {
        const L = canvas._lmHeat; if (!L || !info) return null;
        const c = Math.floor((info.dx - L.x0) / L.cw), r = Math.floor((info.dy - L.y0) / L.ch);
        return r >= 0 && r < L.R && c >= 0 && c < L.C ? { r, c } : null;
    };
    /**
     * 柱状图。vals: 数组，labels: 每根柱子的文字。o: { horizontal, colors, color, hi（高亮下标或 Set）, hiColor, range:[lo,hi], digits, title, labelPx, valueSuffix, maxBars }
     * 横向（horizontal）适合「Top-k 下一个 token」这种文字标签较长的排行。
     */
    draw.bars = function (canvas, vals, labels, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(canvas, { bg: o.bg });
        const n = vals.length, isHi = (i) => (o.hi instanceof Set ? o.hi.has(i) : o.hi === i);
        let lo = o.range ? o.range[0] : Math.min(0, ...vals), hi = o.range ? o.range[1] : Math.max(1e-9, ...vals);
        if (hi - lo < 1e-12) hi = lo + 1;
        const colorOf = (i) => (isHi(i) ? (o.hiColor || LM.C.green) : o.colors ? o.colors[i % o.colors.length] : (o.color || LM.C.blue));
        const lp = o.labelPx || 11, top = o.title ? 18 : 4;
        if (o.title) { ctx.fillStyle = LM.C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.titleX ?? 4, 3); }
        ctx.font = FONT(lp);
        const fmtV = (v) => (o.digits === false ? '' : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(o.digits ?? 2)) + (o.valueSuffix || ''));
        if (o.horizontal) {
            const padL = Math.min(w * 0.45, 8 + Math.max(10, ...labels.map((s) => ctx.measureText(String(s)).width))), padR = 44;
            const bh = (h - top - 4) / n, X = (v) => padL + (v - lo) / (hi - lo) * (w - padL - padR);
            for (let i = 0; i < n; i++) {
                const y = top + i * bh, x0 = X(Math.max(lo, 0)), x1 = X(vals[i]);
                ctx.fillStyle = colorOf(i); ctx.globalAlpha = isHi(i) || o.hi == null ? 1 : 0.75;
                ctx.fillRect(Math.min(x0, x1), y + bh * 0.15, Math.max(1, Math.abs(x1 - x0)), bh * 0.7); ctx.globalAlpha = 1;
                ctx.fillStyle = isHi(i) ? LM.C.text : LM.C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                ctx.fillText(String(labels[i]), padL - 5, y + bh / 2);
                ctx.textAlign = 'left'; ctx.fillStyle = LM.C.dim; ctx.fillText(fmtV(vals[i]), Math.max(x0, x1) + 4, y + bh / 2);
            }
            return;
        }
        const padL = o.axis === false ? 6 : 34, padB = o.padB ?? (labels ? 18 : 6), padR = 6;
        const bw = (w - padL - padR) / n, Y = (v) => h - padB - (v - lo) / (hi - lo) * (h - padB - top);
        if (o.axis !== false) {
            ctx.strokeStyle = LM.C.grid; ctx.fillStyle = LM.C.dim; ctx.font = FONT(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            [lo, (lo + hi) / 2, hi].forEach((v) => { ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke(); ctx.fillText(Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2), padL - 3, Y(v)); });
        }
        for (let i = 0; i < n; i++) {
            const x = padL + i * bw, y0 = Y(Math.max(lo, 0)), y1 = Y(vals[i]);
            ctx.fillStyle = colorOf(i); ctx.globalAlpha = isHi(i) || o.hi == null ? 1 : 0.75;
            ctx.fillRect(x + bw * 0.12, Math.min(y0, y1), Math.max(1, bw * 0.76), Math.max(1, Math.abs(y1 - y0))); ctx.globalAlpha = 1;
            if (labels && (bw >= 9 || i % Math.ceil(9 / bw) === 0)) { ctx.font = FONT(Math.min(lp, Math.max(9, bw * 0.8))); ctx.fillStyle = isHi(i) ? LM.C.text : LM.C.gray; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(labels[i]), x + bw / 2, h - padB + 3); }
            if (o.showValues && bw >= 26) { ctx.font = FONT(11); ctx.fillStyle = LM.C.dim; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(fmtV(vals[i]), x + bw / 2, Math.min(y0, y1) - 1); }
        }
    };
    /**
     * token 条：把一串 token 画成带颜色的小方块（流式换行）。toks: [{ text, id, sub, color, dim, hi, outline }] 或字符串数组。
     * o: { px（字号）, showIds, title, gap, colorBy: 'id' | 'none' }
     */
    draw.tokens = function (canvas, toks, o) {
        o = o || {};
        const { ctx, w } = PL.draw.ctx(canvas, { bg: o.bg });
        const px = o.px || 13, padX = 4, gap = o.gap ?? 3, rowH = px + (o.showIds ? 25 : 10) + gap;
        let x = 6, y = o.title ? 22 : 6;
        if (o.title) { ctx.fillStyle = LM.C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.titleX ?? 6, 4); }
        const boxes = [];
        toks.forEach((t, i) => {
            if (typeof t === 'string') t = { text: t };
            if (t.br) { x = 6; y += rowH; return; }
            ctx.font = FONT(px);
            const label = t.text === ' ' ? '␣' : t.text === '\n' ? '⏎' : t.text;
            const tw = ctx.measureText(label).width + padX * 2;
            ctx.font = FONT(11);
            const sw = o.showIds && t.sub != null ? ctx.measureText(String(t.sub)).width + 4 : 0, bw = Math.max(tw, sw, px * 0.9);
            if (x + bw > w - 4 && x > 6) { x = 6; y += rowH; }
            const bh = px + 8;
            ctx.globalAlpha = t.dim ? 0.3 : 1;
            ctx.fillStyle = t.color || (o.colorBy === 'none' ? '#1d232c' : LM.tokColor(t.id ?? i));
            ctx.fillRect(x, y, bw, bh);
            if (t.hi || t.outline) { ctx.strokeStyle = t.outline || LM.C.yellow; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, bw + 2, bh + 2); }
            ctx.fillStyle = t.textColor || '#f0f0f0'; ctx.font = FONT(px); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(label, x + bw / 2, y + bh / 2 + 0.5);
            if (o.showIds && t.sub != null) { ctx.font = FONT(11); ctx.fillStyle = LM.C.dim; ctx.textBaseline = 'top'; ctx.fillText(String(t.sub), x + bw / 2, y + bh + 2); }
            ctx.globalAlpha = 1;
            boxes.push({ i, x, y, w: bw, h: bh });
            x += bw + gap;
        });
        canvas._lmTokens = boxes;
        return boxes;
    };
    draw.tokenHit = function (canvas, info) { const b = canvas._lmTokens; if (!b || !info) return -1; const hit = b.find((r) => info.dx >= r.x && info.dx <= r.x + r.w && info.dy >= r.y && info.dy <= r.y + r.h + 12); return hit ? hit.i : -1; };
    /**
     * 文本面板：lines = [ string | { text, color, bold, px } | [ {t, color} 片段… ] ]，自动按宽度换行。o: { px, title, lineGap }
     */
    draw.text = function (canvas, lines, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(canvas, { bg: o.bg });
        const px = o.px || 12, lh = px + (o.lineGap ?? 5);
        let y = 6;
        if (o.title) { ctx.fillStyle = LM.C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.titleX ?? 6, 4); y = 22; }
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        lines.forEach((ln) => {
            const segs = Array.isArray(ln) ? ln : [typeof ln === 'string' ? { t: ln } : { t: ln.text, color: ln.color, bold: ln.bold, px: ln.px }];
            let x = 6;
            segs.forEach((s) => {
                ctx.font = FONT(s.px || px, s.bold); ctx.fillStyle = s.color || LM.C.text;
                // 排版单位：拉丁字母 / 数字组成的词整体换行；组合附加符（如 r̂ 的 ^）跟着前一个字符一起画
                const units = String(s.t).match(/\n|[A-Za-z0-9_.'’%+\-]+\p{M}*|\P{M}\p{M}*|\p{M}+/gu) || [];
                const xi = 6 + (o.indent || 0);
                const drawUnit = (u) => {
                    const cw = ctx.measureText(u).width;
                    if (x + cw > w - 6 && x > xi) { x = xi; y += lh; }
                    if (s.bg) { ctx.fillStyle = s.bg; ctx.fillRect(x, y - 1, cw, (s.px || px) + 3); ctx.fillStyle = s.color || LM.C.text; }
                    ctx.fillText(u, x, y); x += cw;
                };
                for (const u of units) {
                    if (u === '\n') { x = 6; y += lh; continue; }
                    if (u.length > 1 && ctx.measureText(u).width > w - 12) { for (const ch of u.match(/\P{M}\p{M}*|\p{M}+/gu)) drawUnit(ch); continue; }   // 比一整行还长的词：逐字符折行
                    drawUnit(u);
                }
            });
            y += lh;
        });
        return y;
    };
    /**
     * 散点 / 轨迹图。pts: [{ x, y, color, r, label, ring }]；o: { x:[a,b], y:[a,b], paths:[{ pts:[[x,y]…], color, width, dash, arrow }],
     *   bg: (x, y) → [r,g,b] 0..255（背景场，例如决策边界），bgRes, xLabel, yLabel, equal（等比例坐标）, title, axes }
     * 返回 { X, Y, inv }：数据 ↔ 设计坐标的变换（canvas._lmScatter 同步保存，供 pick 反算数据坐标）。
     */
    draw.scatter = function (canvas, pts, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(canvas, { bg: o.bg0 });
        const padL = o.axes === false ? 4 : 34, padB = o.axes === false ? 4 : (o.xLabel ? 30 : 18), padT = o.title ? 18 : 6, padR = 8;
        let xr = o.x, yr = o.y;
        if (!xr || !yr) {
            let ax = Infinity, bx = -Infinity, ay = Infinity, by = -Infinity;
            const acc = (x, y) => { if (isFinite(x) && isFinite(y)) { ax = Math.min(ax, x); bx = Math.max(bx, x); ay = Math.min(ay, y); by = Math.max(by, y); } };
            pts.forEach((p) => acc(p.x, p.y)); (o.paths || []).forEach((pa) => pa.pts.forEach((p) => acc(p[0], p[1])));
            if (!isFinite(ax)) { ax = 0; bx = 1; ay = 0; by = 1; }
            const mx = (bx - ax) * 0.08 || 1, my = (by - ay) * 0.08 || 1;
            xr = xr || [ax - mx, bx + mx]; yr = yr || [ay - my, by + my];
        }
        let sx = (w - padL - padR) / (xr[1] - xr[0]), sy = (h - padT - padB) / (yr[1] - yr[0]);
        let ox = padL, oy = h - padB;
        if (o.equal) { const s = Math.min(sx, sy); ox += ((w - padL - padR) - s * (xr[1] - xr[0])) / 2; oy -= ((h - padT - padB) - s * (yr[1] - yr[0])) / 2; sx = sy = s; }
        const X = (v) => ox + (v - xr[0]) * sx, Y = (v) => oy - (v - yr[0]) * sy;
        if (o.bg) {
            const res = o.bgRes || 64, cwp = (X(xr[1]) - X(xr[0])) / res, chp = (Y(yr[0]) - Y(yr[1])) / res;
            for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
                const c = o.bg(xr[0] + (i + 0.5) / res * (xr[1] - xr[0]), yr[1] - (j + 0.5) / res * (yr[1] - yr[0]));
                ctx.fillStyle = rgbStr(c); ctx.fillRect(X(xr[0]) + i * cwp, Y(yr[1]) + j * chp, Math.ceil(cwp) + 0.5, Math.ceil(chp) + 0.5);
            }
        }
        if (o.title) { ctx.fillStyle = LM.C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.titleX ?? 4, 3); }
        if (o.axes !== false) {
            ctx.font = FONT(10); ctx.strokeStyle = LM.C.grid; ctx.fillStyle = LM.C.dim; ctx.lineWidth = 1;
            const ticks = (a, b, n) => { const span = b - a, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), st = [1, 2, 2.5, 5, 10].map((q) => q * mag).find((s) => s >= raw); const out = []; for (let v = Math.ceil(a / st) * st; v <= b + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v); return out; };
            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ticks(xr[0], xr[1], Math.max(2, Math.floor((w - padL) / 60))).forEach((v) => { ctx.beginPath(); ctx.moveTo(X(v), Y(yr[1])); ctx.lineTo(X(v), Y(yr[0])); ctx.stroke(); ctx.fillText(String(+v.toPrecision(3)), X(v), Y(yr[0]) + 2); });
            ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            ticks(yr[0], yr[1], Math.max(2, Math.floor((h - padT - padB) / 34))).forEach((v) => { ctx.beginPath(); ctx.moveTo(X(xr[0]), Y(v)); ctx.lineTo(X(xr[1]), Y(v)); ctx.stroke(); ctx.fillText(String(+v.toPrecision(3)), X(xr[0]) - 3, Y(v)); });
            ctx.fillStyle = '#777';
            if (o.xLabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xLabel, (X(xr[0]) + X(xr[1])) / 2, h - 1); }
            if (o.yLabel) { ctx.save(); ctx.translate(9, (Y(yr[0]) + Y(yr[1])) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
        }
        ctx.save(); ctx.beginPath(); ctx.rect(X(xr[0]), Y(yr[1]), X(xr[1]) - X(xr[0]), Y(yr[0]) - Y(yr[1])); ctx.clip();
        (o.lines || []).forEach((ln) => { ctx.strokeStyle = ln.color || LM.C.gray; ctx.lineWidth = ln.width || 1; ctx.setLineDash(ln.dash || []); ctx.beginPath(); ctx.moveTo(X(ln.a[0]), Y(ln.a[1])); ctx.lineTo(X(ln.b[0]), Y(ln.b[1])); ctx.stroke(); });
        ctx.setLineDash([]);
        (o.paths || []).forEach((pa) => {
            ctx.strokeStyle = pa.color || LM.C.yellow; ctx.lineWidth = pa.width || 1.5; ctx.setLineDash(pa.dash || []); ctx.globalAlpha = pa.alpha ?? 1;
            ctx.beginPath(); pa.pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke();
            if (pa.dots) { ctx.fillStyle = pa.color || LM.C.yellow; pa.pts.forEach((p) => { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), pa.dots, 0, 7); ctx.fill(); }); }
            ctx.globalAlpha = 1; ctx.setLineDash([]);
        });
        pts.forEach((p) => {
            ctx.globalAlpha = p.alpha ?? 1;
            ctx.fillStyle = p.color || LM.C.blue; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), p.r || 2.5, 0, 7); ctx.fill();
            if (p.ring) { ctx.strokeStyle = p.ring; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), (p.r || 2.5) + 2, 0, 7); ctx.stroke(); }
            ctx.globalAlpha = 1;
        });
        ctx.restore();
        pts.forEach((p) => { if (p.label) { ctx.font = FONT(p.labelPx || 11); ctx.fillStyle = p.labelColor || p.color || LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(p.label, X(p.x) + (p.r || 2.5) + 3, Y(p.y)); } });
        const tf = { X, Y, xr, yr, inv: (dx, dy) => [xr[0] + (dx - ox) / sx, yr[0] + (oy - dy) / sy] };
        canvas._lmScatter = tf;
        return tf;
    };
    /** 训练曲线：hist = { x: [...], series: [{ data, label, color, dash, smooth }] }，o 透传给 PL.draw.plot（logY、y、marks…） */
    draw.curves = function (canvas, series, o) {
        o = o || {};
        PL.draw.plot(canvas, Object.assign({}, o, {
            series: series.filter((s) => s.data && s.data.length).map((s, k) => ({
                data: s.smooth ? LM.ema(s.data, s.smooth) : s.data, x: s.x, label: s.label, color: s.color || LM.SERIES[k % LM.SERIES.length], dash: s.dash, width: s.width, fill: s.fill, bar: s.bar,
            })),
        }));
    };
    /** 小图像网格：imgs = [Float32Array(W·H)]（0..1 灰度或带符号；o.rgb = true 时为交错 RGB，W·H·3），labels 可选。o: { cols, symmetric, map, range, labelColors, gap, rgb, outline } */
    draw.imageGrid = function (canvas, imgs, W, H, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(canvas, { bg: o.bg });
        const n = imgs.length, cols = o.cols || Math.ceil(Math.sqrt(n)), rows = Math.ceil(n / cols), lab = o.labels ? 13 : 0, gap = o.gap ?? 4;
        const cell = Math.min((w - gap) / cols - gap, (h - gap) / rows - gap - lab), map = cm(o.map || (o.symmetric ? 'diverge' : 'gray'));
        const off = canvas._lmOff || (canvas._lmOff = (typeof document !== 'undefined' ? document.createElement('canvas') : null));
        off.width = W; off.height = H;
        const octx = off.getContext('2d'), img = octx.createImageData(W, H), rgb = [0, 0, 0];
        imgs.forEach((im, k) => {
            if (o.rgb) {   // 交错 RGB（0..1），W·H·3 个数
                for (let i = 0; i < W * H; i++) for (let ch = 0; ch < 3; ch++) img.data[i * 4 + ch] = LM.clamp(im[i * 3 + ch], 0, 1) * 255;
                for (let i = 0; i < W * H; i++) img.data[i * 4 + 3] = 255;
            } else {
                const r = o.range || autoRange(im, o);
                for (let i = 0; i < W * H; i++) { map((im[i] - r[0]) / (r[1] - r[0]), rgb); img.data[i * 4] = rgb[0]; img.data[i * 4 + 1] = rgb[1]; img.data[i * 4 + 2] = rgb[2]; img.data[i * 4 + 3] = 255; }
            }
            octx.putImageData(img, 0, 0);
            const cx = gap + (k % cols) * (cell + gap), cy = gap + Math.floor(k / cols) * (cell + gap + lab);
            ctx.imageSmoothingEnabled = false; ctx.drawImage(off, cx, cy, cell, cell * H / W);
            if (o.outline && o.outline[k]) { ctx.strokeStyle = o.outline[k]; ctx.lineWidth = 2; ctx.strokeRect(cx - 1, cy - 1, cell + 2, cell * H / W + 2); }
            if (o.labels) { ctx.font = FONT(11); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = o.labelColors ? o.labelColors[k] : LM.C.gray; ctx.fillText(String(o.labels[k]), cx + cell / 2, cy + cell * H / W + 1); }
        });
        canvas._lmGrid = { cols, cell, gap, lab, n, H, W };
    };
    draw.gridHit = function (canvas, info) { const g = canvas._lmGrid; if (!g || !info) return -1; const c = Math.floor((info.dx - g.gap) / (g.cell + g.gap)), r = Math.floor((info.dy - g.gap) / (g.cell * g.H / g.W + g.gap + g.lab)); const k = r * g.cols + c; return c >= 0 && c < g.cols && k >= 0 && k < g.n ? k : -1; };

    // ═════════════════════════════════ 系列导航 ═════════════════════════════════
    // 篇号 = 分类顺序（同时也是一种合法的前置知识顺序：每篇的前置篇号都更小），part = 知识结构里的位置（入口页和导航条按 part 分组：从共享基础到分化领域）
    LM.PARTS = [
        { id: 'base', name: '共享基础', desc: '所有模型都要用的：优化、反向传播、卷积、潜空间、强化学习' },
        { id: 'arch', name: '通用架构', desc: '把任何数据变成 token 序列，再用 Transformer 处理' },
        { id: 'llm', name: '大语言模型', desc: '预训练学知识，后训练学说话' },
        { id: 'domain', name: '分化领域', desc: '在共享基础上长出来的各个方向：多模态理解、扩散生成、视频、音频、世界模型、3D' },
        { id: 'agent', name: '智能体', desc: '让语言模型调用工具、在循环里完成任务：Tool Calling、Agent、Harness' },
    ];
    /** 篇号按「分类」顺序排（同一类连续编号）；pre = 直接的前置篇（读这一篇之前最好先读的）。这个顺序本身也满足前置关系：每篇的前置都排在它前面 */
    LM.PAGES = [
        { file: '01_linear_regression.html', no: '01', part: 'base', pre: [], title: 'Hello World：线性回归与梯度下降', short: '线性回归 · 梯度下降' },
        { file: '02_mlp_backprop.html', no: '02', part: 'base', pre: ['01'], title: '手写数字：从 Softmax 回归到 MLP 与反向传播', short: 'MLP · 反向传播' },
        { file: '03_cnn_unet.html', no: '03', part: 'base', pre: ['02'], title: '卷积神经网络：从卷积核到 U-Net', short: 'CNN · U-Net' },
        { file: '04_latent_vae.html', no: '04', part: 'base', pre: ['03'], title: '潜空间：自编码器、VAE 与 VQ-VAE', short: 'AE · VAE · VQ' },
        { file: '05_reinforcement_learning.html', no: '05', part: 'base', pre: ['02'], title: '强化学习：从价值迭代到策略梯度', short: '强化学习' },
        { file: '06_tokenizer_embedding.html', no: '06', part: 'arch', pre: ['02'], title: '文本变向量：Tokenizer、Embedding 与位置编码', short: 'Tokenizer · Embedding' },
        { file: '07_transformer.html', no: '07', part: 'arch', pre: ['06'], title: 'Transformer：逐层拆解一次前向传播', short: 'Attention · Transformer' },
        { file: '08_pretraining.html', no: '08', part: 'llm', pre: ['07'], title: '预训练：在浏览器里从零训练一个迷你 GPT', short: '预训练 · 采样' },
        { file: '09_alignment.html', no: '09', part: 'llm', pre: ['08', '05'], title: '后训练与对齐：SFT、奖励模型、RLHF、DPO、GRPO', short: 'SFT · RLHF · DPO' },
        { file: '10_multimodal.html', no: '10', part: 'domain', pre: ['08'], title: '多模态：ViT、CLIP 对比学习与迷你 LLaVA', short: 'ViT · CLIP · LLaVA' },
        { file: '11_diffusion.html', no: '11', part: 'domain', pre: ['03', '04'], title: '扩散模型与流匹配：从噪声里生成数据', short: '扩散 · 流匹配' },
        { file: '12_video_generation.html', no: '12', part: 'domain', pre: ['11', '07'], title: '视频生成：时空扩散与时间一致性', short: '视频生成' },
        { file: '13_audio_generation.html', no: '13', part: 'domain', pre: ['04', '08', '11'], title: '音频与语音：频谱、音频 token 与生成', short: '音频 · 语音' },
        { file: '14_world_models.html', no: '14', part: 'domain', pre: ['05', '04'], title: '世界模型：学会模拟环境，在想象中规划', short: '世界模型' },
        { file: '15_nerf_gaussian_splatting.html', no: '15', part: 'domain', pre: ['02'], title: '3D 重建：NeRF 与 3D Gaussian Splatting', short: 'NeRF · 3DGS' },
        { file: '16_tool_calling.html', no: '16', part: 'agent', pre: ['09'], title: '工具调用：模型怎样「调用函数」', short: 'Tool Calling' },
        { file: '17_agent_loop.html', no: '17', part: 'agent', pre: ['16'], title: 'Agent：在循环里用工具完成任务', short: 'Agent 循环' },
        { file: '18_agent_harness.html', no: '18', part: 'agent', pre: ['17'], title: 'Harness：Agent 外面那一圈运行时', short: 'Harness' },
        { file: '19_tool_retrieval.html', no: '19', part: 'agent', pre: ['18', '06'], title: '成千上万个工具：检索、路由与按需加载', short: '工具检索' },
        { file: '20_coding_agent.html', no: '20', part: 'agent', pre: ['18', '19'], title: '编程 Agent：几个通用工具，做成千上万种事', short: 'Coding Agent' },
        { file: '21_engine_mcp.html', no: '21', part: 'agent', pre: ['19', '20'], title: '接进游戏引擎与商用软件：MCP、脚本接口、Computer Use', short: '引擎 · MCP' },
    ];
    /** 前置关系：直接前置、全部前置（传递闭包）、直接后续、深度（最长前置链的长度，01 = 0） */
    LM.prePages = (no) => ((LM.PAGES.find((p) => p.no === no) || {}).pre || []).map((n) => LM.PAGES.find((p) => p.no === n)).filter(Boolean);
    LM.postPages = (no) => LM.PAGES.filter((p) => (p.pre || []).includes(no));
    LM.allPre = (no) => { const out = new Set(), go = (n) => LM.prePages(n).forEach((p) => { if (!out.has(p.no)) { out.add(p.no); go(p.no); } }); go(no); return [...out].sort(); };
    LM.depth = (no) => { const ps = LM.prePages(no); return ps.length ? 1 + Math.max(...ps.map((p) => LM.depth(p.no))) : 0; };
    /** 系列导航条 HTML。cur = 当前篇序号（'01'…，入口页传 null）；fromIndex = 入口页（链接加 pages/ 前缀）。上一篇 / 下一篇按篇号顺序；篇内再列出前置与后续 */
    LM.navHtml = function (cur, fromIndex) {
        const pre = fromIndex ? 'pages/' : '', home = fromIndex ? 'index.html' : '../index.html';
        const i = LM.PAGES.findIndex((p) => p.no === cur), prev = LM.PAGES[i - 1], next = LM.PAGES[i + 1];
        const chip = (p) => (p.no === cur ? `<span class="lm-nav-cur">${p.no} ${p.short}</span>` : `<a href="${pre}${p.file}">${p.no} ${p.short}</a>`);
        const groups = LM.PARTS.map((pt) => `<span class="lm-nav-part">${pt.name}</span>${LM.PAGES.filter((p) => p.part === pt.id).map(chip).join('')}`).join('<span class="lm-nav-sep"></span>');
        const link = (p) => `<a href="${pre}${p.file}">${p.no} ${p.short}</a>`, pres = cur ? LM.prePages(cur) : [], posts = cur ? LM.postPages(cur) : [];
        const preLine = cur ? `<div class="lm-nav-pre"><span class="lm-nav-k">前置知识</span>${pres.length ? pres.map(link).join('') : '<span class="lm-nav-none">无（从这里开始）</span>'}<span class="lm-nav-k">读完可以接着读</span>${posts.length ? posts.map(link).join('') : '<span class="lm-nav-none">—</span>'}</div>` : '';
        return `<div class="lm-nav"><a class="lm-nav-home" href="${home}">⌂ 系列目录</a>${groups}</div>` + preLine +
            (cur ? `<div class="lm-nav-pn">${prev ? `<a href="${prev.file}">← ${prev.no} ${prev.title}</a>` : '<span></span>'}${next ? `<a href="${next.file}">${next.no} ${next.title} →</a>` : '<span></span>'}</div>` : '');
    };
    LM.CSS = `
        .lm-nav { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 0 0 10px; font-size: 0.8rem; }
        .lm-nav a, .lm-nav span { padding: 2px 9px; border: 1px solid #22303c; border-radius: 10px; color: #8fb4cc; background: #0d141a; }
        .lm-nav a:hover { background: #142230; text-decoration: none; }
        .lm-nav .lm-nav-cur { color: #0a0a0a; background: #6ab0e0; border-color: #6ab0e0; font-weight: 600; }
        .lm-nav .lm-nav-home { color: #c8ccd4; border-color: #3a3f46; }
        .lm-nav .lm-nav-part { border: none; background: none; color: #6f8496; padding: 2px 2px 2px 4px; font-size: 0.74rem; letter-spacing: 0.05em; }
        .lm-nav .lm-nav-sep { border: none; background: none; padding: 0; width: 6px; }
        .lm-nav-pre { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: -4px 0 10px; font-size: 0.78rem; }
        .lm-nav-pre a { padding: 1px 8px; border: 1px solid #2a3a2a; border-radius: 10px; color: #9fd0a0; background: #0d1610; }
        .lm-nav-pre a:hover { background: #142a18; text-decoration: none; }
        .lm-nav-pre .lm-nav-k { color: #6f8496; margin-left: 4px; }
        .lm-nav-pre .lm-nav-none { color: #55606b; }
        .lm-nav-pn { display: flex; justify-content: space-between; gap: 12px; font-size: 0.8rem; margin: -2px 0 10px; }
        .lm-tbl { border-collapse: collapse; margin: 6px 0; font-size: 0.86rem; }
        .lm-tbl th, .lm-tbl td { border: 1px solid #262b31; padding: 3px 9px; text-align: left; vertical-align: top; }
        .lm-tbl th { background: #121820; color: #b8d6e6; font-weight: 600; }
        .lm-tbl td code { white-space: nowrap; }
        .lm-kpi { display: inline-block; margin: 2px 10px 2px 0; padding: 1px 8px; border-radius: 8px; background: #121820; color: #b8d6e6; font-family: var(--pl-mono); font-size: 0.82rem; }
    `;

    // <CPP-BEGIN> 由 tools/gen_cpp.js 从 pages/cpp/*.h 生成，请勿手改
    // 本系列自带的极简 C++ 机器学习库（只用 C++17 标准库）。每篇把用到的部分放进 graph.common：blocks: LM.CPP.blocks(['core', 'autograd', ...])
    LM.CPP = {
        core: `
// minml_core.h — 本系列自带的极简 C++ 机器学习库 · 第 1 部分：随机数与二维张量
// 只用 C++17 标准库，单线程，不追求速度，只求每一行都看得懂。约定与页面里的 JS 参考实现（llm_common.js 的 LM.AG）完全相同：
//   · 张量一律是二维行主序 [rows × cols]；一个 batch 的序列展平成 [B·T × d]：第 b 条序列第 t 个位置 = 第 b·T + t 行
//   · 一张图 = 张量的一行，按通道优先展平 [C × H × W]（与 PyTorch 的 NCHW 相同）
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// ── 随机数：mulberry32，与页面里 JS 的 LM.rng / PL.rng 产生逐位相同的序列 ──
struct Rng { uint32_t a = 1; };

inline Rng makeRng(uint32_t seed) { Rng r; r.a = seed ? seed : 1; return r; }

// 返回 [0, 1) 上的均匀随机数
inline double uniform(Rng& r) {
    r.a += 0x6D2B79F5u;
    uint32_t t = (r.a ^ (r.a >> 15)) * (1u | r.a);
    t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
    return (double)(t ^ (t >> 14)) / 4294967296.0;
}

// 标准正态 N(0, 1)：Box-Muller（与 LM.randn 相同：消耗两个均匀数，只用 cos 那一半）
inline float randn(Rng& r) {
    double u1 = std::max(uniform(r), 1e-12), u2 = uniform(r);
    return (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
}

// [0, n) 上的随机整数
inline int randint(Rng& r, int n) { return (int)(uniform(r) * n); }

// ── 二维张量 ──
struct Tensor {
    int rows = 0, cols = 0;
    std::vector<float> v;                                        // rows × cols 个数，行主序
    float& at(int i, int j) { return v[(size_t)i * cols + j]; }
    float at(int i, int j) const { return v[(size_t)i * cols + j]; }
    int size() const { return rows * cols; }
};

inline Tensor zeros(int rows, int cols) {
    Tensor t; t.rows = rows; t.cols = cols; t.v.assign((size_t)rows * cols, 0.f);
    return t;
}

inline Tensor full(int rows, int cols, float x) {
    Tensor t = zeros(rows, cols);
    std::fill(t.v.begin(), t.v.end(), x);
    return t;
}

// 每个元素独立地 ~ N(0, std²)
inline Tensor randn(int rows, int cols, float std, Rng& r) {
    Tensor t = zeros(rows, cols);
    for (float& x : t.v) x = std * randn(r);
    return t;
}

inline Tensor fromVector(int rows, int cols, const std::vector<float>& data) {
    Tensor t = zeros(rows, cols);
    for (int i = 0; i < rows * cols; i++) t.v[i] = data[i];
    return t;
}

// n 个 float / int 组成的数组（统一用 std::vector<float> x = floats(n); 的写法，比 x(n) 更不容易和函数调用混淆）
inline std::vector<float> floats(int n, float x = 0.f) { return std::vector<float>((size_t)n, x); }
inline std::vector<int> ints(int n, int x = 0) { return std::vector<int>((size_t)n, x); }

// ── 不需要梯度的小工具（采样、评估用） ──
// x ← softmax(x / temperature)，原地；减去最大值防止 exp 溢出
inline void softmaxInPlace(std::vector<float>& x, float temperature) {
    float m = x[0];
    for (float a : x) m = std::max(m, a);
    double s = 0;
    for (float& a : x) { a = std::exp((a - m) / temperature); s += a; }
    for (float& a : x) a = (float)(a / s);
}

// 最大元素的下标
inline int argmaxIndex(const std::vector<float>& x) {
    int b = 0;
    for (int i = 1; i < (int)x.size(); i++) if (x[i] > x[b]) b = i;
    return b;
}

// 按概率 p 抽一个下标（p 之和为 1）
inline int sampleIndex(const std::vector<float>& p, Rng& r) {
    double u = uniform(r);
    for (int i = 0; i + 1 < (int)p.size(); i++) { u -= p[i]; if (u <= 0) return i; }
    return (int)p.size() - 1;
}

// 打乱 [0, n) 的一个排列（Fisher–Yates）
inline std::vector<int> permutation(int n, Rng& r) {
    std::vector<int> p = ints(n);
    for (int i = 0; i < n; i++) p[i] = i;
    for (int i = n - 1; i > 0; i--) { int j = (int)(uniform(r) * (i + 1)); std::swap(p[i], p[j]); }
    return p;
}
`,
        autograd: `
// minml_autograd.h — 极简 C++ 机器学习库 · 第 2 部分：自动微分（与页面 JS 的 LM.AG 一一对应）
// 思路：每个算子算出前向结果，同时把「怎样把输出的梯度传回输入」写成一个闭包，压进全局 tape；
// backward(loss) 从 loss 出发逆序执行这些闭包（链式法则），每个参数的 grad 里就累加出了 ∂loss/∂参数。
#pragma once
#include "minml_core.h"

// 计算图里的一个节点：前向值 + 损失对它的梯度（同形状）+ 反向传播闭包
struct Node {
    Tensor val, grad;
    bool requiresGrad = false;
    std::function<void(Node&)> backFn;                           // 由创建它的算子填写
};

// 节点的句柄（共享所有权）：Var x;  x->val 是数值，x->grad 是梯度
struct Var {
    std::shared_ptr<Node> n;
    Node* operator->() const { return n.get(); }
};

// 全局 tape：按前向顺序记录所有需要反向的节点
struct Tape { std::vector<Var> nodes; bool enabled = true; };

inline Tape& tape() { static Tape t; return t; }

inline Var makeVar(Tensor val) { Var x; x.n = std::make_shared<Node>(); x->val = std::move(val); return x; }

// 可训练参数（requiresGrad = true，梯度会累加进 ->grad）
inline Var param(Tensor init) { Var p = makeVar(std::move(init)); p->requiresGrad = true; return p; }

// 常量输入（数据、标签），不求梯度
inline Var constant(Tensor t) { return makeVar(std::move(t)); }

// 取梯度缓冲区（第一次用时分配成全 0）
inline Tensor& gradOf(const Var& x) {
    if ((int)x->grad.v.size() != x->val.size()) x->grad = zeros(x->val.rows, x->val.cols);
    return x->grad;
}

// 生成算子的输出节点：只要有一个输入需要梯度、且没在 NoGrad 里，就记进 tape
inline Var record(Tensor val, std::vector<Var> inputs, std::function<void(Node&)> back) {
    Var out = makeVar(std::move(val));
    bool need = false;
    for (const Var& x : inputs) need = need || x->requiresGrad;
    if (need && tape().enabled) { out->requiresGrad = true; out->backFn = back; tape().nodes.push_back(out); }
    return out;
}

// 反向传播：∂loss/∂loss = 1，然后逆序执行 tape（loss 必须是 1×1）
inline void backward(const Var& loss) {
    Tensor& g = gradOf(loss);
    std::fill(g.v.begin(), g.v.end(), 0.f);
    g.v[0] = 1.f;
    for (int i = (int)tape().nodes.size() - 1; i >= 0; --i) {
        Node& nd = *tape().nodes[i].n;
        if (!nd.grad.v.empty()) nd.backFn(nd);
    }
    tape().nodes.clear();
}

// 丢弃 tape（只做前向、不反向时调用，否则 tape 会一直变长）
inline void clearTape() { tape().nodes.clear(); }

// RAII：作用域内不记录 tape —— 推理 / 评估时用（相当于 PyTorch 的 torch.no_grad()）
struct NoGrad {
    bool prev = true;
    NoGrad() { prev = tape().enabled; tape().enabled = false; }
    ~NoGrad() { tape().enabled = prev; }
};

// ═══════════════════════════ 矩阵乘与线性层 ═══════════════════════════
// C = A·B    A[n×k]  B[k×m]  →  C[n×m]
inline Var matmul(const Var& A, const Var& B) {
    const Tensor &a = A->val, &b = B->val;
    int n = a.rows, k = a.cols, m = b.cols;
    Tensor c = zeros(n, m);
    for (int i = 0; i < n; i++)
        for (int p = 0; p < k; p++) {
            float av = a.v[i * k + p];
            if (av == 0) continue;
            for (int j = 0; j < m; j++) c.v[i * m + j] += av * b.v[p * m + j];
        }
    return record(std::move(c), {A, B}, [A, B](Node& o) {
        const Tensor &a = A->val, &b = B->val;
        int n = a.rows, k = a.cols, m = b.cols;
        if (A->requiresGrad) {                                   // dA = dC · Bᵀ
            Tensor& ga = gradOf(A);
            for (int i = 0; i < n; i++)
                for (int p = 0; p < k; p++) {
                    float s = 0;
                    for (int j = 0; j < m; j++) s += o.grad.v[i * m + j] * b.v[p * m + j];
                    ga.v[i * k + p] += s;
                }
        }
        if (B->requiresGrad) {                                   // dB = Aᵀ · dC
            Tensor& gb = gradOf(B);
            for (int i = 0; i < n; i++)
                for (int p = 0; p < k; p++) {
                    float av = a.v[i * k + p];
                    if (av == 0) continue;
                    for (int j = 0; j < m; j++) gb.v[p * m + j] += av * o.grad.v[i * m + j];
                }
        }
    });
}

// C = A·Bᵀ    A[n×k]  B[m×k]  →  C[n×m]（权重共享的 LM head、CLIP 的相似度矩阵）
inline Var matmulBT(const Var& A, const Var& B) {
    const Tensor &a = A->val, &b = B->val;
    int n = a.rows, k = a.cols, m = b.rows;
    Tensor c = zeros(n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += a.v[i * k + p] * b.v[j * k + p];
            c.v[i * m + j] = s;
        }
    return record(std::move(c), {A, B}, [A, B](Node& o) {
        const Tensor &a = A->val, &b = B->val;
        int n = a.rows, k = a.cols, m = b.rows;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) {
                float g = o.grad.v[i * m + j];
                if (g == 0) continue;
                if (A->requiresGrad) { Tensor& ga = gradOf(A); for (int p = 0; p < k; p++) ga.v[i * k + p] += g * b.v[j * k + p]; }
                if (B->requiresGrad) { Tensor& gb = gradOf(B); for (int p = 0; p < k; p++) gb.v[j * k + p] += g * a.v[i * k + p]; }
            }
    });
}

// Y = X + b：b 是一行 [1×m]，加到 X 的每一行上（偏置）
inline Var addRow(const Var& X, const Var& b) {
    Tensor y = X->val;
    int n = y.rows, m = y.cols;
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) y.v[i * m + j] += b->val.v[j];
    return record(std::move(y), {X, b}, [X, b](Node& o) {
        int n = o.val.rows, m = o.val.cols;
        if (X->requiresGrad) { Tensor& gx = gradOf(X); for (int i = 0; i < n * m; i++) gx.v[i] += o.grad.v[i]; }
        if (b->requiresGrad) { Tensor& gb = gradOf(b); for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) gb.v[j] += o.grad.v[i * m + j]; }
    });
}

// 线性层 Y = X·W + b    X[n×in]  W[in×out]  b[1×out]
inline Var linear(const Var& X, const Var& W, const Var& b) { return addRow(matmul(X, W), b); }

// ═══════════════════════════ 逐元素运算 ═══════════════════════════
// 逐元素加 / 减 / 乘（形状相同）
inline Var add(const Var& A, const Var& B) {
    Tensor y = A->val;
    for (int i = 0; i < y.size(); i++) y.v[i] += B->val.v[i];
    return record(std::move(y), {A, B}, [A, B](Node& o) {
        for (int i = 0; i < o.val.size(); i++) {
            if (A->requiresGrad) gradOf(A).v[i] += o.grad.v[i];
            if (B->requiresGrad) gradOf(B).v[i] += o.grad.v[i];
        }
    });
}

inline Var sub(const Var& A, const Var& B) {
    Tensor y = A->val;
    for (int i = 0; i < y.size(); i++) y.v[i] -= B->val.v[i];
    return record(std::move(y), {A, B}, [A, B](Node& o) {
        for (int i = 0; i < o.val.size(); i++) {
            if (A->requiresGrad) gradOf(A).v[i] += o.grad.v[i];
            if (B->requiresGrad) gradOf(B).v[i] -= o.grad.v[i];
        }
    });
}

inline Var mul(const Var& A, const Var& B) {
    Tensor y = A->val;
    for (int i = 0; i < y.size(); i++) y.v[i] *= B->val.v[i];
    return record(std::move(y), {A, B}, [A, B](Node& o) {
        for (int i = 0; i < o.val.size(); i++) {
            if (A->requiresGrad) gradOf(A).v[i] += o.grad.v[i] * B->val.v[i];   // ∂(ab)/∂a = b
            if (B->requiresGrad) gradOf(B).v[i] += o.grad.v[i] * A->val.v[i];
        }
    });
}

// 逐元素一元运算：激活函数等。actF = 前向 y = f(x)，actD = 导数 dy/dx（可以用 x，也可以用已算好的 y）
enum Act { ACT_RELU, ACT_TANH, ACT_SIGMOID, ACT_GELU, ACT_SILU, ACT_EXP, ACT_LOG, ACT_SOFTPLUS, ACT_LOGSIGMOID, ACT_SQUARE };

inline float actF(Act a, float x) {
    const float k = 0.7978845608f;                                // √(2/π)，GELU 的 tanh 近似（GPT-2 同款）
    switch (a) {
        case ACT_RELU: return x > 0 ? x : 0.f;
        case ACT_TANH: return std::tanh(x);
        case ACT_SIGMOID: return 1.f / (1.f + std::exp(-x));
        case ACT_GELU: return 0.5f * x * (1 + std::tanh(k * (x + 0.044715f * x * x * x)));
        case ACT_SILU: return x / (1.f + std::exp(-x));
        case ACT_EXP: return std::exp(x);
        case ACT_LOG: return std::log(x);
        case ACT_SOFTPLUS: return x > 20 ? x : x < -20 ? std::exp(x) : std::log1p(std::exp(x));
        case ACT_LOGSIGMOID: return x >= 0 ? -std::log1p(std::exp(-x)) : x - std::log1p(std::exp(x));   // log σ(x)，数值稳定
        case ACT_SQUARE: return x * x;
    }
    return x;
}

inline float actD(Act a, float x, float y) {
    const float k = 0.7978845608f;
    switch (a) {
        case ACT_RELU: return x > 0 ? 1.f : 0.f;
        case ACT_TANH: return 1 - y * y;
        case ACT_SIGMOID: return y * (1 - y);
        case ACT_GELU: { float t = std::tanh(k * (x + 0.044715f * x * x * x)); return 0.5f * (1 + t) + 0.5f * x * (1 - t * t) * k * (1 + 3 * 0.044715f * x * x); }
        case ACT_SILU: { float s = 1.f / (1.f + std::exp(-x)); return s * (1 + x * (1 - s)); }
        case ACT_EXP: return y;
        case ACT_LOG: return 1.f / x;
        case ACT_SOFTPLUS: return 1.f / (1.f + std::exp(-x));
        case ACT_LOGSIGMOID: return 1.f / (1.f + std::exp(x));
        case ACT_SQUARE: return 2 * x;
    }
    return 1.f;
}

inline Var activation(const Var& A, Act act) {
    Tensor y = A->val;
    for (float& x : y.v) x = actF(act, x);
    return record(std::move(y), {A}, [A, act](Node& o) {
        Tensor& ga = gradOf(A);
        for (int i = 0; i < o.val.size(); i++) ga.v[i] += o.grad.v[i] * actD(act, A->val.v[i], o.val.v[i]);
    });
}

inline Var relu(const Var& A) { return activation(A, ACT_RELU); }
inline Var tanhAct(const Var& A) { return activation(A, ACT_TANH); }
inline Var sigmoid(const Var& A) { return activation(A, ACT_SIGMOID); }
inline Var gelu(const Var& A) { return activation(A, ACT_GELU); }
inline Var silu(const Var& A) { return activation(A, ACT_SILU); }
inline Var expAct(const Var& A) { return activation(A, ACT_EXP); }
inline Var logAct(const Var& A) { return activation(A, ACT_LOG); }
inline Var softplus(const Var& A) { return activation(A, ACT_SOFTPLUS); }
inline Var logSigmoid(const Var& A) { return activation(A, ACT_LOGSIGMOID); }
inline Var square(const Var& A) { return activation(A, ACT_SQUARE); }

// Y = s·X（常数缩放）
inline Var scale(const Var& A, float s) {
    Tensor y = A->val;
    for (float& x : y.v) x *= s;
    return record(std::move(y), {A}, [A, s](Node& o) { Tensor& ga = gradOf(A); for (int i = 0; i < o.val.size(); i++) ga.v[i] += o.grad.v[i] * s; });
}

// Y = X + c（常数）
inline Var addConst(const Var& A, float c) {
    Tensor y = A->val;
    for (float& x : y.v) x += c;
    return record(std::move(y), {A}, [A](Node& o) { Tensor& ga = gradOf(A); for (int i = 0; i < o.val.size(); i++) ga.v[i] += o.grad.v[i]; });
}

// ═══════════════════════════ 归约与形状 ═══════════════════════════
// 所有元素的平均 → 1×1
inline Var meanAll(const Var& A) {
    Tensor y = zeros(1, 1);
    double s = 0;
    for (float x : A->val.v) s += x;
    y.v[0] = (float)(s / A->val.size());
    return record(std::move(y), {A}, [A](Node& o) { Tensor& ga = gradOf(A); float g = o.grad.v[0] / A->val.size(); for (float& x : ga.v) x += g; });
}

// 所有元素的和 → 1×1
inline Var sumAll(const Var& A) {
    Tensor y = zeros(1, 1);
    double s = 0;
    for (float x : A->val.v) s += x;
    y.v[0] = (float)s;
    return record(std::move(y), {A}, [A](Node& o) { Tensor& ga = gradOf(A); for (float& x : ga.v) x += o.grad.v[0]; });
}

// 转置 [n×m] → [m×n]
inline Var transpose(const Var& A) {
    int n = A->val.rows, m = A->val.cols;
    Tensor y = zeros(m, n);
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) y.v[j * n + i] = A->val.v[i * m + j];
    return record(std::move(y), {A}, [A, n, m](Node& o) { Tensor& ga = gradOf(A); for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) ga.v[i * m + j] += o.grad.v[j * n + i]; });
}

// 取行：Y[i] = X[idx[i]]（embedding 查表就是它）；反向 = 按行累加回去
inline Var gatherRows(const Var& X, const std::vector<int>& idx) {
    int n = (int)idx.size(), d = X->val.cols;
    Tensor y = zeros(n, d);
    for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) y.v[i * d + j] = X->val.v[idx[i] * d + j];
    return record(std::move(y), {X}, [X, idx, n, d](Node& o) { Tensor& gx = gradOf(X); for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) gx.v[idx[i] * d + j] += o.grad.v[i * d + j]; });
}

inline Var embedding(const std::vector<int>& ids, const Var& E) { return gatherRows(E, ids); }

// 按行拼接 [A; B]（列数相同）
inline Var concatRows(const Var& A, const Var& B) {
    Tensor y = zeros(A->val.rows + B->val.rows, A->val.cols);
    std::copy(A->val.v.begin(), A->val.v.end(), y.v.begin());
    std::copy(B->val.v.begin(), B->val.v.end(), y.v.begin() + A->val.size());
    return record(std::move(y), {A, B}, [A, B](Node& o) {
        int na = A->val.size();
        if (A->requiresGrad) { Tensor& ga = gradOf(A); for (int i = 0; i < na; i++) ga.v[i] += o.grad.v[i]; }
        if (B->requiresGrad) { Tensor& gb = gradOf(B); for (int i = 0; i < B->val.size(); i++) gb.v[i] += o.grad.v[na + i]; }
    });
}

// 按列拼接 [A | B]（行数相同）：U-Net 跳连 = 特征图的通道拼接，因为一张图就是一行
inline Var concatCols(const Var& A, const Var& B) {
    int n = A->val.rows, ca = A->val.cols, cb = B->val.cols;
    Tensor y = zeros(n, ca + cb);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < ca; j++) y.v[i * (ca + cb) + j] = A->val.v[i * ca + j];
        for (int j = 0; j < cb; j++) y.v[i * (ca + cb) + ca + j] = B->val.v[i * cb + j];
    }
    return record(std::move(y), {A, B}, [A, B, n, ca, cb](Node& o) {
        for (int i = 0; i < n; i++) {
            if (A->requiresGrad) for (int j = 0; j < ca; j++) gradOf(A).v[i * ca + j] += o.grad.v[i * (ca + cb) + j];
            if (B->requiresGrad) for (int j = 0; j < cb; j++) gradOf(B).v[i * cb + j] += o.grad.v[i * (ca + cb) + ca + j];
        }
    });
}

// 取第 [c0, c1) 列（把一个线性层的输出拆成 μ / logσ²、GRU 的几个门…）
inline Var sliceCols(const Var& A, int c0, int c1) {
    int n = A->val.rows, c = A->val.cols, m = c1 - c0;
    Tensor y = zeros(n, m);
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) y.v[i * m + j] = A->val.v[i * c + c0 + j];
    return record(std::move(y), {A}, [A, n, c, m, c0](Node& o) { Tensor& ga = gradOf(A); for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) ga.v[i * c + c0 + j] += o.grad.v[i * m + j]; });
}

// 每 k 行取平均：[(n·k) × d] → [n × d]（把一张图的所有 patch token 平均成一个向量）
inline Var meanRows(const Var& X, int k) {
    int n = X->val.rows / k, d = X->val.cols;
    Tensor y = zeros(n, d);
    for (int i = 0; i < n; i++) for (int r = 0; r < k; r++) for (int j = 0; j < d; j++) y.v[i * d + j] += X->val.v[(i * k + r) * d + j] / k;
    return record(std::move(y), {X}, [X, n, k, d](Node& o) { Tensor& gx = gradOf(X); for (int i = 0; i < n; i++) for (int r = 0; r < k; r++) for (int j = 0; j < d; j++) gx.v[(i * k + r) * d + j] += o.grad.v[i * d + j] / k; });
}

// ═══════════════════════════ 归一化 ═══════════════════════════
// LayerNorm（对每一行）：y = γ·(x − μ)/√(σ² + ε) + β
inline Var layerNorm(const Var& X, const Var& gamma, const Var& beta, float eps = 1e-5f) {
    int n = X->val.rows, d = X->val.cols;
    Tensor y = zeros(n, d), xhat = zeros(n, d);
    std::vector<float> rstd = floats(n);
    for (int i = 0; i < n; i++) {
        const float* x = &X->val.v[i * d];
        float mu = 0; for (int j = 0; j < d; j++) mu += x[j]; mu /= d;
        float var = 0; for (int j = 0; j < d; j++) var += (x[j] - mu) * (x[j] - mu); var /= d;
        rstd[i] = 1.f / std::sqrt(var + eps);
        for (int j = 0; j < d; j++) { xhat.v[i * d + j] = (x[j] - mu) * rstd[i]; y.v[i * d + j] = xhat.v[i * d + j] * gamma->val.v[j] + beta->val.v[j]; }
    }
    return record(std::move(y), {X, gamma, beta}, [X, gamma, beta, xhat, rstd, n, d](Node& o) {
        for (int i = 0; i < n; i++) {
            float s1 = 0, s2 = 0;
            for (int j = 0; j < d; j++) {
                float g = o.grad.v[i * d + j], dh = g * gamma->val.v[j];
                s1 += dh; s2 += dh * xhat.v[i * d + j];
                if (gamma->requiresGrad) gradOf(gamma).v[j] += g * xhat.v[i * d + j];
                if (beta->requiresGrad) gradOf(beta).v[j] += g;
            }
            if (X->requiresGrad) {
                s1 /= d; s2 /= d;
                Tensor& gx = gradOf(X);
                for (int j = 0; j < d; j++) gx.v[i * d + j] += rstd[i] * (o.grad.v[i * d + j] * gamma->val.v[j] - s1 - xhat.v[i * d + j] * s2);
            }
        }
    });
}

// 每一行除以自己的 L2 范数（CLIP 的嵌入）
inline Var l2Normalize(const Var& X) {
    int n = X->val.rows, d = X->val.cols;
    Tensor y = zeros(n, d);
    std::vector<float> inv = floats(n);
    for (int i = 0; i < n; i++) {
        double s = 0; for (int j = 0; j < d; j++) s += X->val.v[i * d + j] * X->val.v[i * d + j];
        inv[i] = (float)(1.0 / std::sqrt(s + 1e-12));
        for (int j = 0; j < d; j++) y.v[i * d + j] = X->val.v[i * d + j] * inv[i];
    }
    return record(std::move(y), {X}, [X, inv, n, d](Node& o) {
        Tensor& gx = gradOf(X);
        for (int i = 0; i < n; i++) {
            float dot = 0; for (int j = 0; j < d; j++) dot += o.grad.v[i * d + j] * o.val.v[i * d + j];
            for (int j = 0; j < d; j++) gx.v[i * d + j] += inv[i] * (o.grad.v[i * d + j] - o.val.v[i * d + j] * dot);
        }
    });
}

// ═══════════════════════════ 注意力 ═══════════════════════════
// 多头缩放点积注意力。Q/K/V: [B·T × D]，H 个头，每头 dh = D/H；causal = true 时位置 i 只看 j ≤ i。
// probsOut（可选）收集注意力权重：下标 ((b·H + h)·T + i)·T + j
inline Var attention(const Var& Q, const Var& K, const Var& Vv, int B, int T, int H, bool causal, std::vector<float>* probsOut = nullptr) {
    int D = Q->val.cols, dh = D / H;
    float sc = 1.f / std::sqrt((float)dh);
    Tensor out = zeros(B * T, D);
    std::vector<float> P = floats(B * H * T * T), row = floats(T);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < T; i++) {
                int jmax = causal ? i : T - 1;
                const float* q = &Q->val.v[(b * T + i) * D + h * dh];
                float m = -1e30f;
                for (int j = 0; j <= jmax; j++) {
                    const float* k = &K->val.v[(b * T + j) * D + h * dh];
                    float s = 0; for (int c = 0; c < dh; c++) s += q[c] * k[c];
                    row[j] = s * sc; m = std::max(m, row[j]);
                }
                float z = 0; for (int j = 0; j <= jmax; j++) { row[j] = std::exp(row[j] - m); z += row[j]; }
                float* o = &out.v[(b * T + i) * D + h * dh];
                for (int j = 0; j <= jmax; j++) {
                    float p = row[j] / z;
                    P[((size_t)(b * H + h) * T + i) * T + j] = p;
                    const float* v = &Vv->val.v[(b * T + j) * D + h * dh];
                    for (int c = 0; c < dh; c++) o[c] += p * v[c];
                }
            }
    if (probsOut) *probsOut = P;
    return record(std::move(out), {Q, K, Vv}, [Q, K, Vv, P, B, T, H, causal, dh, sc](Node& o) {
        int D = H * dh;
        std::vector<float> dP = floats(T);
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++)
                for (int i = 0; i < T; i++) {
                    int jmax = causal ? i : T - 1;
                    const float* g = &o.grad.v[(b * T + i) * D + h * dh];
                    const float* pr = &P[((size_t)(b * H + h) * T + i) * T];
                    float dot = 0;
                    for (int j = 0; j <= jmax; j++) {                // dV += pᵢⱼ · dO；dP = dO · V
                        const float* v = &Vv->val.v[(b * T + j) * D + h * dh];
                        float s = 0;
                        for (int c = 0; c < dh; c++) { s += g[c] * v[c]; if (Vv->requiresGrad) gradOf(Vv).v[(b * T + j) * D + h * dh + c] += pr[j] * g[c]; }
                        dP[j] = s; dot += pr[j] * s;
                    }
                    for (int j = 0; j <= jmax; j++) {                // softmax 反向：dS = P ⊙ (dP − Σ P·dP)
                        float ds = pr[j] * (dP[j] - dot) * sc;
                        if (ds == 0) continue;
                        for (int c = 0; c < dh; c++) {
                            if (Q->requiresGrad) gradOf(Q).v[(b * T + i) * D + h * dh + c] += ds * K->val.v[(b * T + j) * D + h * dh + c];
                            if (K->requiresGrad) gradOf(K).v[(b * T + j) * D + h * dh + c] += ds * Q->val.v[(b * T + i) * D + h * dh + c];
                        }
                    }
                }
    });
}

// ═══════════════════════════ 损失函数 ═══════════════════════════
// 交叉熵（自带 softmax）。logits[n×V]，targets[n]（−1 = 忽略该行）。weights（可空）= 每行权重，可为负（策略梯度的 advantage）；
// denom ≤ 0 时自动取有效行数（或 Σ|w|）。返回 1×1。rowLossOut（可选）收集每行的 −log p。
inline Var crossEntropy(const Var& L, const std::vector<int>& targets, const std::vector<float>* weights = nullptr, float denom = 0.f, std::vector<float>* rowLossOut = nullptr) {
    int n = L->val.rows, V = L->val.cols;
    Tensor P = zeros(n, V);
    double tot = 0, cnt = 0;
    if (rowLossOut) rowLossOut->assign(n, NAN);
    for (int i = 0; i < n; i++) {
        const float* z = &L->val.v[i * V];
        float m = z[0]; for (int j = 1; j < V; j++) m = std::max(m, z[j]);
        double s = 0; for (int j = 0; j < V; j++) { P.v[i * V + j] = std::exp(z[j] - m); s += P.v[i * V + j]; }
        for (int j = 0; j < V; j++) P.v[i * V + j] = (float)(P.v[i * V + j] / s);
        int t = targets[i];
        if (t < 0) continue;
        double li = -(z[t] - m - std::log(s)), w = weights ? (*weights)[i] : 1.0;
        if (rowLossOut) (*rowLossOut)[i] = (float)li;
        tot += w * li; cnt += weights ? std::fabs(w) : 1.0;
    }
    float den = denom > 0 ? denom : (float)std::max(cnt, 1.0);
    Tensor y = zeros(1, 1); y.v[0] = (float)(tot / den);
    std::vector<float> w = weights ? *weights : std::vector<float>();   // 复制一份：反向时调用方的数组可能已经不在了
    return record(std::move(y), {L}, [L, P, targets, w, den, n, V](Node& o) {
        Tensor& gl = gradOf(L);
        for (int i = 0; i < n; i++) {
            int t = targets[i];
            if (t < 0) continue;
            float k = o.grad.v[0] * (w.empty() ? 1.f : w[i]) / den;
            for (int j = 0; j < V; j++) gl.v[i * V + j] += k * P.v[i * V + j];   // ∂/∂z = (softmax − onehot) · w / denom
            gl.v[i * V + t] -= k;
        }
    });
}

// 均方误差 mean((A − B)²)
inline Var mse(const Var& A, const Var& B) {
    Tensor y = zeros(1, 1);
    double s = 0;
    for (int i = 0; i < A->val.size(); i++) s += (A->val.v[i] - B->val.v[i]) * (A->val.v[i] - B->val.v[i]);
    int n = A->val.size();
    y.v[0] = (float)(s / n);
    return record(std::move(y), {A, B}, [A, B, n](Node& o) {
        float k = 2 * o.grad.v[0] / n;
        for (int i = 0; i < n; i++) {
            float d = A->val.v[i] - B->val.v[i];
            if (A->requiresGrad) gradOf(A).v[i] += k * d;
            if (B->requiresGrad) gradOf(B).v[i] -= k * d;
        }
    });
}

// 每条序列的 Σ log p(目标)：rowSeq[i] = 第 i 行属于第几条序列（−1 = 不计）。返回 [nSeq × 1]（DPO 用）
inline Var seqLogProb(const Var& L, const std::vector<int>& targets, const std::vector<int>& rowSeq, int nSeq) {
    int n = L->val.rows, V = L->val.cols;
    Tensor P = zeros(n, V), y = zeros(nSeq, 1);
    for (int i = 0; i < n; i++) {
        int s = rowSeq[i], t = targets[i];
        if (s < 0 || t < 0) continue;
        const float* z = &L->val.v[i * V];
        float m = z[0]; for (int j = 1; j < V; j++) m = std::max(m, z[j]);
        double sum = 0; for (int j = 0; j < V; j++) { P.v[i * V + j] = std::exp(z[j] - m); sum += P.v[i * V + j]; }
        for (int j = 0; j < V; j++) P.v[i * V + j] = (float)(P.v[i * V + j] / sum);
        y.v[s] += (float)(z[t] - m - std::log(sum));
    }
    return record(std::move(y), {L}, [L, P, targets, rowSeq, n, V](Node& o) {
        Tensor& gl = gradOf(L);
        for (int i = 0; i < n; i++) {
            int s = rowSeq[i], t = targets[i];
            if (s < 0 || t < 0) continue;
            float k = o.grad.v[s];
            for (int j = 0; j < V; j++) gl.v[i * V + j] -= k * P.v[i * V + j];
            gl.v[i * V + t] += k;
        }
    });
}

// ═══════════════════════════ 卷积族（一张图 = 一行，通道优先 C×H×W） ═══════════════════════════
struct ConvSpec { int C = 1, H = 1, W = 1, KH = 3, KW = 3, stride = 1, padH = 0, padW = 0; };

// 方形卷积核的常用写法：conv(C, H, W, K, stride, pad)
inline ConvSpec conv(int C, int H, int W, int K, int stride = 1, int pad = 0) {
    ConvSpec s; s.C = C; s.H = H; s.W = W; s.KH = K; s.KW = K; s.stride = stride; s.padH = pad; s.padW = pad;
    return s;
}

inline int convOutH(const ConvSpec& s) { return (s.H + 2 * s.padH - s.KH) / s.stride + 1; }
inline int convOutW(const ConvSpec& s) { return (s.W + 2 * s.padW - s.KW) / s.stride + 1; }

// 二维卷积（互相关，与 PyTorch Conv2d 相同）。X[N × C·H·W]，Wt[Cout × C·KH·KW]，b[1 × Cout]
inline Var conv2d(const Var& X, const Var& Wt, const Var& b, const ConvSpec& s) {
    int N = X->val.rows, Co = Wt->val.rows, Ho = convOutH(s), Wo = convOutW(s);
    int inS = s.C * s.H * s.W, outS = Co * Ho * Wo, kS = s.C * s.KH * s.KW;
    Tensor y = zeros(N, outS);
    for (int n = 0; n < N; n++)
        for (int co = 0; co < Co; co++)
            for (int oy = 0; oy < Ho; oy++)
                for (int ox = 0; ox < Wo; ox++) {
                    float acc = b->val.v[co];
                    for (int ci = 0; ci < s.C; ci++)
                        for (int ky = 0; ky < s.KH; ky++) {
                            int iy = oy * s.stride - s.padH + ky;
                            if (iy < 0 || iy >= s.H) continue;
                            for (int kx = 0; kx < s.KW; kx++) {
                                int ix = ox * s.stride - s.padW + kx;
                                if (ix < 0 || ix >= s.W) continue;
                                acc += X->val.v[n * inS + (ci * s.H + iy) * s.W + ix] * Wt->val.v[co * kS + (ci * s.KH + ky) * s.KW + kx];
                            }
                        }
                    y.v[n * outS + (co * Ho + oy) * Wo + ox] = acc;
                }
    return record(std::move(y), {X, Wt, b}, [X, Wt, b, s, N, Co, Ho, Wo, inS, outS, kS](Node& o) {
        for (int n = 0; n < N; n++)
            for (int co = 0; co < Co; co++)
                for (int oy = 0; oy < Ho; oy++)
                    for (int ox = 0; ox < Wo; ox++) {
                        float g = o.grad.v[n * outS + (co * Ho + oy) * Wo + ox];
                        if (g == 0) continue;
                        if (b->requiresGrad) gradOf(b).v[co] += g;
                        for (int ci = 0; ci < s.C; ci++)
                            for (int ky = 0; ky < s.KH; ky++) {
                                int iy = oy * s.stride - s.padH + ky;
                                if (iy < 0 || iy >= s.H) continue;
                                for (int kx = 0; kx < s.KW; kx++) {
                                    int ix = ox * s.stride - s.padW + kx;
                                    if (ix < 0 || ix >= s.W) continue;
                                    int xi = n * inS + (ci * s.H + iy) * s.W + ix, wi = co * kS + (ci * s.KH + ky) * s.KW + kx;
                                    if (Wt->requiresGrad) gradOf(Wt).v[wi] += g * X->val.v[xi];
                                    if (X->requiresGrad) gradOf(X).v[xi] += g * Wt->val.v[wi];
                                }
                            }
                    }
    });
}

// 2×2 平均池化：[N × C·H·W] → [N × C·(H/2)·(W/2)]
inline Var avgPool2(const Var& X, int C, int H, int W) {
    int N = X->val.rows, h = H / 2, w = W / 2;
    Tensor y = zeros(N, C * h * w);
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
        int base = n * C * H * W + c * H * W + 2 * i * W + 2 * j;
        y.v[n * C * h * w + (c * h + i) * w + j] = 0.25f * (X->val.v[base] + X->val.v[base + 1] + X->val.v[base + W] + X->val.v[base + W + 1]);
    }
    return record(std::move(y), {X}, [X, N, C, H, W, h, w](Node& o) {
        Tensor& gx = gradOf(X);
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
            int base = n * C * H * W + c * H * W + 2 * i * W + 2 * j;
            float g = 0.25f * o.grad.v[n * C * h * w + (c * h + i) * w + j];
            gx.v[base] += g; gx.v[base + 1] += g; gx.v[base + W] += g; gx.v[base + W + 1] += g;
        }
    });
}

// 2×2 最大池化：梯度只回到每个窗口里最大的那个位置
inline Var maxPool2(const Var& X, int C, int H, int W) {
    int N = X->val.rows, h = H / 2, w = W / 2;
    Tensor y = zeros(N, C * h * w);
    std::vector<int> arg = ints(y.size());
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < h; i++) for (int j = 0; j < w; j++) {
        int base = n * C * H * W + c * H * W + 2 * i * W + 2 * j, cand[4] = {base, base + 1, base + W, base + W + 1}, best = cand[0];
        for (int k = 1; k < 4; k++) if (X->val.v[cand[k]] > X->val.v[best]) best = cand[k];
        int oi = n * C * h * w + (c * h + i) * w + j;
        y.v[oi] = X->val.v[best]; arg[oi] = best;
    }
    return record(std::move(y), {X}, [X, arg](Node& o) { Tensor& gx = gradOf(X); for (int i = 0; i < o.val.size(); i++) gx.v[arg[i]] += o.grad.v[i]; });
}

// 最近邻 2× 上采样：[N × C·H·W] → [N × C·2H·2W]（U-Net 解码器）
inline Var upsample2(const Var& X, int C, int H, int W) {
    int N = X->val.rows, H2 = 2 * H, W2 = 2 * W;
    Tensor y = zeros(N, C * H2 * W2);
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < H2; i++) for (int j = 0; j < W2; j++)
        y.v[n * C * H2 * W2 + (c * H2 + i) * W2 + j] = X->val.v[n * C * H * W + (c * H + i / 2) * W + j / 2];
    return record(std::move(y), {X}, [X, N, C, H, W, H2, W2](Node& o) {
        Tensor& gx = gradOf(X);
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int i = 0; i < H2; i++) for (int j = 0; j < W2; j++)
            gx.v[n * C * H * W + (c * H + i / 2) * W + j / 2] += o.grad.v[n * C * H2 * W2 + (c * H2 + i) * W2 + j];
    });
}

// 按通道广播加：X[N × C·HW] + Bc[N × C]（扩散 U-Net 里把时间步 embedding 加到每个通道上）
inline Var addChannel(const Var& X, const Var& Bc, int C) {
    int N = X->val.rows, HW = X->val.cols / C;
    Tensor y = X->val;
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int k = 0; k < HW; k++) y.v[n * C * HW + c * HW + k] += Bc->val.v[n * C + c];
    return record(std::move(y), {X, Bc}, [X, Bc, N, C, HW](Node& o) {
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) {
            float s = 0;
            for (int k = 0; k < HW; k++) { float g = o.grad.v[n * C * HW + c * HW + k]; s += g; if (X->requiresGrad) gradOf(X).v[n * C * HW + c * HW + k] += g; }
            if (Bc->requiresGrad) gradOf(Bc).v[n * C + c] += s;
        }
    });
}

// 按通道广播乘：X[N × C·HW] ⊙ Sc[N × C]（FiLM 式条件调制）
inline Var mulChannel(const Var& X, const Var& Sc, int C) {
    int N = X->val.rows, HW = X->val.cols / C;
    Tensor y = X->val;
    for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int k = 0; k < HW; k++) y.v[n * C * HW + c * HW + k] *= Sc->val.v[n * C + c];
    return record(std::move(y), {X, Sc}, [X, Sc, N, C, HW](Node& o) {
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) {
            float s = 0, sc = Sc->val.v[n * C + c];
            for (int k = 0; k < HW; k++) {
                int i = n * C * HW + c * HW + k;
                s += o.grad.v[i] * X->val.v[i];
                if (X->requiresGrad) gradOf(X).v[i] += o.grad.v[i] * sc;
            }
            if (Sc->requiresGrad) gradOf(Sc).v[n * C + c] += s;
        }
    });
}
`,
        nn: `
// minml_nn.h — 极简 C++ 机器学习库 · 第 3 部分：初始化、线性层、优化器、学习率调度（与页面 JS 的 LM.AdamW / LM.SGD / LM.cosineLR 相同）
#pragma once
#include "minml_autograd.h"

// ── 初始化 ──
// He / Kaiming：std = √(2 / fanIn)，适合 ReLU 一类激活（第 02 篇）
inline Tensor heInit(int rows, int cols, int fanIn, Rng& r) { return randn(rows, cols, std::sqrt(2.f / fanIn), r); }

// Xavier / Glorot：std = √(2 / (fanIn + fanOut))，适合 tanh / sigmoid
inline Tensor xavierInit(int rows, int cols, int fanIn, int fanOut, Rng& r) { return randn(rows, cols, std::sqrt(2.f / (fanIn + fanOut)), r); }

// ── 线性层：Y = X·W + b，W 存成 [in × out]（PyTorch 的 nn.Linear 存 [out × in]，算的是同一个东西） ──
struct Linear { Var W, b; };

inline Linear makeLinear(int in, int out, float std, Rng& r) { Linear l; l.W = param(randn(in, out, std, r)); l.b = param(zeros(1, out)); return l; }

inline Var forward(const Linear& l, const Var& x) { return linear(x, l.W, l.b); }

// ── 梯度工具 ──
inline void zeroGrad(std::vector<Var>& params) { for (Var& p : params) std::fill(p->grad.v.begin(), p->grad.v.end(), 0.f); }

// 所有梯度的全局 L2 范数；超过 maxNorm 就整体缩小到 maxNorm。返回裁剪前的范数
inline float clipGradNorm(std::vector<Var>& params, float maxNorm) {
    double s = 0;
    for (Var& p : params) for (float g : p->grad.v) s += (double)g * g;
    float norm = (float)std::sqrt(s);
    if (maxNorm > 0 && norm > maxNorm) {
        float k = maxNorm / (norm + 1e-6f);
        for (Var& p : params) for (float& g : p->grad.v) g *= k;
    }
    return norm;
}

// ── SGD（可带动量）：v ← μ·v + g；p ← p − lr·v ──
struct SGD { std::vector<Var> params; std::vector<Tensor> vel; float lr = 0.1f, momentum = 0.f; };

inline SGD makeSGD(std::vector<Var> params, float lr, float momentum = 0.f) {
    SGD o; o.params = params; o.lr = lr; o.momentum = momentum;
    for (Var& p : params) o.vel.push_back(zeros(p->val.rows, p->val.cols));
    return o;
}

inline void sgdStep(SGD& o, float lr) {
    for (size_t k = 0; k < o.params.size(); k++) {
        Var& p = o.params[k];
        if (p->grad.v.empty()) continue;
        for (int i = 0; i < p->val.size(); i++) { o.vel[k].v[i] = o.momentum * o.vel[k].v[i] + p->grad.v[i]; p->val.v[i] -= lr * o.vel[k].v[i]; }
    }
}

// ── AdamW（Loshchilov & Hutter：权重衰减与梯度的自适应缩放解耦）。decay[k] = 第 k 个参数要不要做权重衰减 ──
struct AdamW {
    std::vector<Var> params;
    std::vector<bool> decay;
    std::vector<Tensor> m, v;                                    // 一阶 / 二阶矩
    float lr = 1e-3f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.1f;
    int t = 0;
};

inline AdamW makeAdamW(std::vector<Var> params, std::vector<bool> decay, float lr, float wd, float b1 = 0.9f, float b2 = 0.95f, float eps = 1e-8f) {
    AdamW o; o.params = params; o.decay = decay; o.lr = lr; o.wd = wd; o.b1 = b1; o.b2 = b2; o.eps = eps;
    for (Var& p : params) { o.m.push_back(zeros(p->val.rows, p->val.cols)); o.v.push_back(zeros(p->val.rows, p->val.cols)); }
    return o;
}

// 一步更新：m ← β₁m + (1−β₁)g；v ← β₂v + (1−β₂)g²；p ← p − lr·( m̂/(√v̂ + ε) + wd·p )，m̂、v̂ 是偏差修正后的值
inline void adamwStep(AdamW& o, float lr) {
    o.t++;
    double bc1 = 1 - std::pow((double)o.b1, o.t), bc2 = 1 - std::pow((double)o.b2, o.t);
    for (size_t k = 0; k < o.params.size(); k++) {
        Var& p = o.params[k];
        if (p->grad.v.empty()) continue;
        float wd = o.decay[k] ? o.wd : 0.f;
        for (int i = 0; i < p->val.size(); i++) {
            float g = p->grad.v[i];
            o.m[k].v[i] = o.b1 * o.m[k].v[i] + (1 - o.b1) * g;
            o.v[k].v[i] = o.b2 * o.v[k].v[i] + (1 - o.b2) * g * g;
            double mh = o.m[k].v[i] / bc1, vh = o.v[k].v[i] / bc2;
            p->val.v[i] -= (float)(lr * (mh / (std::sqrt(vh) + o.eps) + wd * p->val.v[i]));
        }
    }
}

// ── 学习率：线性 warmup → cosine 衰减到 minLR（nanoGPT 的 get_lr 同款） ──
inline float cosineLR(int it, float maxLR, float minLR, int warmup, int total) {
    if (it < warmup) return maxLR * (it + 1) / warmup;
    if (it >= total) return minLR;
    double r = (double)(it - warmup) / std::max(1, total - warmup);
    return (float)(minLR + 0.5 * (1 + std::cos(3.141592653589793 * r)) * (maxLR - minLR));
}
`,
        gpt: `
// minml_gpt.h — 极简 C++ 机器学习库 · 第 4 部分：迷你 GPT（与页面 JS 的 LM.gpt 逐个参数、逐个算子对应）
// 结构同 nanoGPT / GPT-2：可学习位置 embedding、Pre-LN、多头因果注意力、GELU(tanh) MLP（4×）、权重共享的 LM head。
#pragma once
#include "minml_nn.h"

struct GPTConfig {
    int V = 76;        // 词表大小（4 个特殊 token + 72 个汉字 / 标点）
    int T = 32;        // 上下文长度
    int d = 32;        // 残差流宽度
    int L = 2;         // Block 层数
    int H = 4;         // 注意力头数（每头 d / H = 8 维）
    int ff = 128;      // MLP 隐藏层宽度（4d）
    float std = 0.02f; // 初始化标准差（GPT-2）
};

// 一个 Transformer Block 的全部参数（W 都存成 [in × out]）
struct Block { Var ln1g, ln1b, Wq, bq, Wk, bk, Wv, bv, Wo, bo, ln2g, ln2b, W1, b1, W2, b2; };

struct GPT {
    GPTConfig cfg;
    Var wte, wpe;                                                // token / 位置 embedding：[V × d]、[T × d]
    std::vector<Block> blocks;
    Var lnfg, lnfb;                                              // 最后的 LayerNorm
    std::vector<Var> params;                                     // 全部参数（顺序与 JS 版相同）
    std::vector<bool> decay;                                     // 哪些参数做权重衰减（二维矩阵 / embedding）
};

// 新建一个参数并登记（顺序决定了随机数的消耗顺序：同一个 seed 得到与 JS 版逐位相同的初始权重）
inline Var addParam(GPT& m, Tensor init, bool decay) { Var p = param(std::move(init)); m.params.push_back(p); m.decay.push_back(decay); return p; }

// GPT-2 初始化：N(0, 0.02)；残差分支的输出投影（Wo、W2）再除以 √(2L)；LayerNorm γ = 1、β = 0；偏置 0
inline GPT makeGPT(const GPTConfig& c, uint32_t seed) {
    GPT m; m.cfg = c;
    Rng r = makeRng(seed);
    float s = c.std, rs = c.std / std::sqrt(2.f * c.L);
    m.wte = addParam(m, randn(c.V, c.d, s, r), true);
    m.wpe = addParam(m, randn(c.T, c.d, s, r), true);
    for (int l = 0; l < c.L; l++) {
        Block b;
        b.ln1g = addParam(m, full(1, c.d, 1.f), false); b.ln1b = addParam(m, zeros(1, c.d), false);
        b.Wq = addParam(m, randn(c.d, c.d, s, r), true); b.bq = addParam(m, zeros(1, c.d), false);
        b.Wk = addParam(m, randn(c.d, c.d, s, r), true); b.bk = addParam(m, zeros(1, c.d), false);
        b.Wv = addParam(m, randn(c.d, c.d, s, r), true); b.bv = addParam(m, zeros(1, c.d), false);
        b.Wo = addParam(m, randn(c.d, c.d, rs, r), true); b.bo = addParam(m, zeros(1, c.d), false);
        b.ln2g = addParam(m, full(1, c.d, 1.f), false); b.ln2b = addParam(m, zeros(1, c.d), false);
        b.W1 = addParam(m, randn(c.d, c.ff, s, r), true); b.b1 = addParam(m, zeros(1, c.ff), false);
        b.W2 = addParam(m, randn(c.ff, c.d, rs, r), true); b.b2 = addParam(m, zeros(1, c.d), false);
        m.blocks.push_back(b);
    }
    m.lnfg = addParam(m, full(1, c.d, 1.f), false); m.lnfb = addParam(m, zeros(1, c.d), false);
    return m;
}

inline int numParams(const GPT& m) { int n = 0; for (const Var& p : m.params) n += p->val.size(); return n; }

// 一个 Pre-LN Block：x ← x + Attn(LN₁(x))；x ← x + MLP(LN₂(x))
inline Var blockForward(const Block& b, const Var& x, int B, int T, int H) {
    Var h1 = layerNorm(x, b.ln1g, b.ln1b);
    Var q = linear(h1, b.Wq, b.bq), k = linear(h1, b.Wk, b.bk), v = linear(h1, b.Wv, b.bv);
    Var x1 = add(x, linear(attention(q, k, v, B, T, H, true), b.Wo, b.bo));
    Var h2 = layerNorm(x1, b.ln2g, b.ln2b);
    return add(x1, linear(gelu(linear(h2, b.W1, b.b1)), b.W2, b.b2));
}

// 前向：ids = B 条长度 T 的序列（展平）→ logits [B·T × V]。inputEmb 非空时替代 token embedding（多模态前缀）
inline Var gptForward(GPT& m, const std::vector<int>& ids, int B, int T, const Var* inputEmb = nullptr) {
    std::vector<int> pos = ints(B * T);
    for (int i = 0; i < B * T; i++) pos[i] = i % T;
    Var x = add(inputEmb ? *inputEmb : embedding(ids, m.wte), gatherRows(m.wpe, pos));
    for (const Block& b : m.blocks) x = blockForward(b, x, B, T, m.cfg.H);
    return matmulBT(layerNorm(x, m.lnfg, m.lnfb), m.wte);   // 权重共享：LM head 就是 wteᵀ
}

// ── 训练数据：随机取 B 个长度 T+1 的窗口，y 是 x 右移一位（每个位置都在预测下一个 token） ──
struct Batch { std::vector<int> x, y; };

inline Batch getBatch(const std::vector<int>& data, int B, int T, Rng& r) {
    Batch b; b.x.resize(B * T); b.y.resize(B * T);
    for (int i = 0; i < B; i++) {
        int s = (int)(uniform(r) * (data.size() - T - 1));
        for (int t = 0; t < T; t++) { b.x[i * T + t] = data[s + t]; b.y[i * T + t] = data[s + t + 1]; }
    }
    return b;
}

// 一步训练：前向 → 交叉熵 → 反向 → 梯度裁剪到 1.0 → AdamW。返回这一步的 loss
inline float trainStep(GPT& m, AdamW& opt, const Batch& b, int B, int T, float lr) {
    zeroGrad(m.params);
    Var loss = crossEntropy(gptForward(m, b.x, B, T), b.y);
    backward(loss);
    clipGradNorm(m.params, 1.0f);
    adamwStep(opt, lr);
    return loss->val.v[0];
}

// ── 解码：温度 → top-k → top-p（nucleus），返回重新归一化后的分布（没选中的为 0） ──
inline std::vector<float> decodeDist(std::vector<float> logits, float temperature, int topK, float topP) {
    int V = (int)logits.size();
    softmaxInPlace(logits, std::max(temperature, 1e-4f));
    std::vector<int> order = ints(V);
    for (int i = 0; i < V; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&logits](int a, int b) { return logits[a] > logits[b]; });
    std::vector<float> p = floats(V);
    double cum = 0, s = 0;
    for (int r = 0; r < V; r++) {
        if (topK > 0 && r >= topK) break;
        int i = order[r];
        p[i] = logits[i]; cum += logits[i];
        if (topP > 0 && topP < 1 && cum >= topP) break;
    }
    for (float x : p) s += x;
    for (float& x : p) x = (float)(x / s);
    return p;
}

// ── 推理用的逐 token 前向（不建计算图）+ KV cache：每个新 token 只算自己的 q/k/v，复用之前所有位置的 k/v ──
struct KVCache { std::vector<std::vector<float>> k, v; };       // 每层一份 [T × d]

// 一行的 LayerNorm（推理用）
inline void layerNormRow(const float* x, const Var& g, const Var& b, int d, float* out) {
    float mu = 0, var = 0;
    for (int j = 0; j < d; j++) mu += x[j];
    mu /= d;
    for (int j = 0; j < d; j++) var += (x[j] - mu) * (x[j] - mu);
    float rs = 1.f / std::sqrt(var / d + 1e-5f);
    for (int j = 0; j < d; j++) out[j] = (x[j] - mu) * rs * g->val.v[j] + b->val.v[j];
}

// 一行的线性层：out = x·W + b（推理用）
inline void linearRow(const float* x, const Var& W, const Var& b, int in, int outN, float* out) {
    for (int j = 0; j < outN; j++) out[j] = b->val.v[j];
    for (int p = 0; p < in; p++) { float xv = x[p]; if (xv == 0) continue; for (int j = 0; j < outN; j++) out[j] += xv * W->val.v[p * outN + j]; }
}

// 把位置 pos 的 token 喂进模型，返回下一个 token 的 logits（并把这一位置的 k / v 写进 cache）
inline std::vector<float> gptStep(const GPT& m, KVCache& c, int tok, int pos) {
    const GPTConfig& g = m.cfg;
    int d = g.d, dh = d / g.H;
    std::vector<float> x = floats(d), h = floats(d), q = floats(d), k = floats(d), v = floats(d), a = floats(d);
    std::vector<float> t = floats(std::max(d, g.ff)), f = floats(g.ff), att = floats(g.T), logits = floats(g.V);
    for (int j = 0; j < d; j++) x[j] = m.wte->val.v[tok * d + j] + m.wpe->val.v[pos * d + j];
    for (int l = 0; l < g.L; l++) {
        const Block& b = m.blocks[l];
        layerNormRow(x.data(), b.ln1g, b.ln1b, d, h.data());
        linearRow(h.data(), b.Wq, b.bq, d, d, q.data()); linearRow(h.data(), b.Wk, b.bk, d, d, k.data()); linearRow(h.data(), b.Wv, b.bv, d, d, v.data());
        std::copy(k.begin(), k.end(), c.k[l].begin() + pos * d);
        std::copy(v.begin(), v.end(), c.v[l].begin() + pos * d);
        std::fill(a.begin(), a.end(), 0.f);
        for (int hh = 0; hh < g.H; hh++) {
            float mx = -1e30f, z = 0;
            for (int j = 0; j <= pos; j++) {
                float s = 0;
                for (int e = 0; e < dh; e++) s += q[hh * dh + e] * c.k[l][j * d + hh * dh + e];
                att[j] = s / std::sqrt((float)dh); mx = std::max(mx, att[j]);
            }
            for (int j = 0; j <= pos; j++) { att[j] = std::exp(att[j] - mx); z += att[j]; }
            for (int j = 0; j <= pos; j++) for (int e = 0; e < dh; e++) a[hh * dh + e] += att[j] / z * c.v[l][j * d + hh * dh + e];
        }
        linearRow(a.data(), b.Wo, b.bo, d, d, t.data());
        for (int j = 0; j < d; j++) x[j] += t[j];
        layerNormRow(x.data(), b.ln2g, b.ln2b, d, h.data());
        linearRow(h.data(), b.W1, b.b1, d, g.ff, f.data());
        for (float& z : f) z = actF(ACT_GELU, z);
        linearRow(f.data(), b.W2, b.b2, g.ff, d, t.data());
        for (int j = 0; j < d; j++) x[j] += t[j];
    }
    layerNormRow(x.data(), m.lnfg, m.lnfb, d, h.data());
    for (int vv = 0; vv < g.V; vv++) { float s = 0; for (int j = 0; j < d; j++) s += h[j] * m.wte->val.v[vv * d + j]; logits[vv] = s; }
    return logits;
}

// 自回归生成：先把 prompt 喂进去，然后每步从 decodeDist 采样一个 token（greedy = 取最大），遇到 stopId 或上下文满了就停
inline std::vector<int> generate(const GPT& m, std::vector<int> prompt, int maxNew, float temperature, int topK, float topP, Rng& r, bool greedy = false, int stopId = -1) {
    const GPTConfig& g = m.cfg;
    KVCache c;
    c.k.assign(g.L, std::vector<float>(g.T * g.d, 0.f));
    c.v.assign(g.L, std::vector<float>(g.T * g.d, 0.f));
    std::vector<float> logits;
    for (int t = 0; t < (int)prompt.size() && t < g.T; t++) logits = gptStep(m, c, prompt[t], t);
    std::vector<int> out = prompt;
    for (int n = 0; n < maxNew && (int)out.size() < g.T; n++) {
        int id = greedy ? argmaxIndex(logits) : sampleIndex(decodeDist(logits, temperature, topK, topP), r);
        out.push_back(id);
        if (id == stopId) break;
        if ((int)out.size() < g.T) logits = gptStep(m, c, id, (int)out.size() - 1);
    }
    return out;
}

// ── 字符级词表（UTF-8）：一个汉字 / 标点 = 一个 token；特殊 token 写成 <eos> 这样的整体 ──
struct Vocab { std::vector<std::string> itos; };

// 把 UTF-8 字符串切成一个个字符（汉字占 3 个字节）
inline std::vector<std::string> utf8Chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        out.push_back(s.substr(i, n)); i += n;
    }
    return out;
}

inline int tokenId(const Vocab& v, const std::string& tok) {
    for (int i = 0; i < (int)v.itos.size(); i++) if (v.itos[i] == tok) return i;
    return -1;
}

inline std::vector<int> encode(const Vocab& v, const std::string& text) {
    std::vector<int> ids;
    std::vector<std::string> chars = utf8Chars(text);
    for (size_t i = 0; i < chars.size(); i++) {
        if (chars[i] == "<") {                                   // 特殊 token：<eos>、<user>、<bot>…
            std::string tok = "<";
            size_t j = i + 1;
            while (j < chars.size() && chars[j] != ">") tok += chars[j++];
            int id = tokenId(v, tok + ">");
            if (id >= 0) { ids.push_back(id); i = j; continue; }
        }
        ids.push_back(tokenId(v, chars[i]));
    }
    return ids;
}

inline std::string decode(const Vocab& v, const std::vector<int>& ids) {
    std::string s;
    for (int id : ids) s += v.itos[id];
    return s;
}
`,
        files: {"core":"minml_core.h","autograd":"minml_autograd.h","nn":"minml_nn.h","gpt":"minml_gpt.h"},
    };
    /** graph.common 用的 blocks：LM.CPP.blocks(['core', 'autograd', 'nn', 'gpt'])（按依赖顺序，缺的依赖自动补上） */
    LM.CPP.blocks = function (mods) {
        const order = ['core', 'autograd', 'nn', 'gpt'], need = new Set(mods || order), deps = { autograd: ['core'], nn: ['autograd'], gpt: ['nn'] };
        const addDeps = (m) => (deps[m] || []).forEach((d) => { need.add(d); addDeps(d); });
        [...need].forEach(addDeps);
        return order.filter((m) => need.has(m)).map((m) => ({ label: `━ cpp/${LM.CPP.files[m]} ━`, code: LM.CPP[m], lang: 'cpp' }));
    };
    // <CPP-END>

    return LM;
})();
// 步骤同时写了 blocks 和 formula / code / previews / notes / warn 时，框架只渲染 blocks；这里把后者按默认顺序接到 blocks 后面（系列里的「教学版 → 生产版」步骤是「表格 + 预览 + 说明」）
(function () {
    if (typeof PipelineLab === 'undefined' || !PipelineLab.mount || PipelineLab.mount._lmBlocks) return;
    const mount0 = PipelineLab.mount, list = (v) => (v == null ? [] : Array.isArray(v) ? v : [v]);
    PipelineLab.mount = function (spec) {
        for (const st of (spec && spec.steps) || []) {
            if (!st.blocks) continue;
            const extra = [];
            list(st.formula).forEach((f) => extra.push(typeof f === 'object' ? Object.assign({ type: 'formula' }, f) : { type: 'formula', text: f }));
            list(st.code).forEach((c) => extra.push(Object.assign({ type: 'code' }, c)));
            if (st.previews) extra.push({ type: 'previews', items: st.previews });
            list(st.notes || st.note).forEach((n) => extra.push({ type: 'note', html: n }));
            list(st.warn || st.warns).forEach((n) => extra.push({ type: 'warn', html: n }));
            if (extra.length) { st.blocks = st.blocks.concat(extra); for (const k of ['formula', 'code', 'previews', 'notes', 'note', 'warn', 'warns']) delete st[k]; }
        }
        return mount0.apply(this, arguments);
    };
    PipelineLab.mount._lmBlocks = true;
})();
if (typeof module !== 'undefined' && module.exports) module.exports = LM;
