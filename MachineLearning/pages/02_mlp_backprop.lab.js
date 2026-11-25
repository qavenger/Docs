// 02_mlp_backprop.lab.js — 系列第 02 篇：手写数字，从 Softmax 回归到 MLP 与反向传播（C++ 手写反传）
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js 02_mlp_backprop.lab.js -o 02_mlp_backprop.html
// 页面里的每张预览都由下面的 JS 参考实现实时算出：程序化手写数字、Softmax 回归与 MLP 的真实训练、三方梯度校验、各层统计。

const P02 = (() => {
    'use strict';
    const IMG = 16, D = IMG * IMG, K = 10;
    const NTRAIN = 4000, NTEST = 500;
    const BATCH = 32;

    // ═══════════════════ 程序化手写数字：笔画字形 + 随机仿射 / 粗细 / 抖动 / 噪声 ═══════════════════
    // 字形坐标 (u, v) ∈ [0,1]²，v 向下；每个数字 = 若干条折线笔画
    const arc = (cx, cy, rx, ry, a0, a1) => {
        const n = Math.max(4, Math.ceil(Math.abs(a1 - a0) / 15)), pts = [];
        for (let i = 0; i <= n; i++) { const a = (a0 + (a1 - a0) * i / n) * Math.PI / 180; pts.push([cx + rx * Math.cos(a), cy + ry * Math.sin(a)]); }
        return pts;
    };
    const line = (x0, y0, x1, y1) => [[x0, y0], [x1, y1]];
    const quad = (x0, y0, x1, y1, x2, y2) => { const pts = []; for (let i = 0; i <= 8; i++) { const t = i / 8, s = 1 - t; pts.push([s * s * x0 + 2 * s * t * x1 + t * t * x2, s * s * y0 + 2 * s * t * y1 + t * t * y2]); } return pts; };
    /** 数字 d 的笔画。j(x, a) = x ± a 的随机抖动：每个样本的圆心、半径、起止角、端点都不同；r() 还在几种写法之间选择 */
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
                if (r() < 0.3) s.push(arc(bx + 0.06, by - 0.06, 0.07, 0.06, 90, 450));   // 底部带小圈的 2
                return s;
            }
            case 3: {
                const low = arc(j(0.49, 0.03), j(0.67, 0.03), j(0.25, 0.04), j(0.19, 0.03), j(270, 10), j(515, 20));
                if (r() < 0.25) { const m = low[0]; return [line(j(0.28, 0.04), j(0.14, 0.03), j(0.72, 0.04), 0.14), line(0.72, 0.14, m[0], m[1]), low]; }   // 平顶的 3
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
                if (r() < 0.2) s.push(line(j(0.4, 0.04), 0.5, j(0.72, 0.04), 0.5));   // 带横杠的 7
                return s;
            }
            case 8: return [arc(j(0.5, 0.03), j(0.3, 0.02), j(0.19, 0.04), j(0.17, 0.03), 0, 360), arc(j(0.5, 0.03), j(0.68, 0.02), j(0.23, 0.04), j(0.19, 0.03), 0, 360)];
            default: {
                const cx = j(0.5, 0.03), cy = j(0.33, 0.03), rx = j(0.22, 0.04), sx = cx + rx;
                return [arc(cx, cy, rx, j(0.2, 0.03), 0, 360), r() < 0.6 ? line(sx, cy, j(0.64, 0.08), 0.88) : quad(sx, cy, j(0.74, 0.05), 0.75, j(0.42, 0.06), 0.86)];
            }
        }
    }
    /** 渲染一个数字到 out[256]（0..1 灰度）：字形 → 抖动 → 仿射 → 距离场画笔 → 噪声 */
    function renderDigit(d, r, out) {
        const pls = strokes(d, r);
        const rot = (2 * r() - 1) * 0.25, sc = 0.82 + 0.3 * r(), asp = 0.85 + 0.3 * r(), sh = (2 * r() - 1) * 0.35;   // 旋转 ±14°、缩放、长宽比、错切
        const tx = (2 * r() - 1) * 0.08, ty = (2 * r() - 1) * 0.07;                                                  // 平移 ±1.3 px
        const wa = 0.015 + 0.035 * r(), f1 = 0.6 + 1.0 * r(), f2 = 0.6 + 1.0 * r(), p1 = 6.2832 * r(), p2 = 6.2832 * r(); // 低频「手抖」位移场
        const c = Math.cos(rot), s = Math.sin(rot), seg = [];
        for (const pl of pls) {
            const ox = 0.02 * LM.randn(r), oy = 0.02 * LM.randn(r);   // 每一笔各自的小偏移（笔画接不上 / 出头）
            let px = 0, py = 0;
            pl.forEach(([u, v], i) => {
                let x = u + wa * Math.sin(6.2832 * f1 * v + p1) + ox - 0.5, y = v + wa * Math.sin(6.2832 * f2 * u + p2) + oy - 0.5;
                x += sh * y; x *= sc * asp; y *= sc / asp;
                const X = (0.5 + c * x - s * y + tx) * IMG, Y = (0.5 + s * x + c * y + ty) * IMG;   // → 像素坐标
                if (i) seg.push(px, py, X, Y);
                px = X; py = Y;
            });
        }
        const rad = 0.55 + 0.9 * r(), peak = 0.75 + 0.25 * r(), tex = 0.12;   // 笔画半宽（像素）、墨色深浅、笔触纹理
        for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
            const qx = x + 0.5, qy = y + 0.5;
            let best = 1e9;
            for (let k = 0; k < seg.length; k += 4) {
                const ax = seg[k], ay = seg[k + 1], bx = seg[k + 2] - ax, by = seg[k + 3] - ay, l2 = bx * bx + by * by;
                let t = l2 > 0 ? ((qx - ax) * bx + (qy - ay) * by) / l2 : 0; t = t < 0 ? 0 : t > 1 ? 1 : t;
                const dx = qx - ax - t * bx, dy = qy - ay - t * by, d2 = dx * dx + dy * dy;
                if (d2 < best) best = d2;
            }
            let v = peak * Math.min(1, Math.max(0, rad + 0.5 - Math.sqrt(best)));   // 1 像素宽的抗锯齿边缘
            if (v > 0) v *= 1 + tex * LM.randn(r);
            if (r() < 0.012) v = Math.max(v, 0.15 + 0.4 * r());                     // 稀疏的背景噪点
            out[y * IMG + x] = v < 0 ? 0 : v > 1 ? 1 : v;
        }
        return out;
    }
    /** 数据集：每类样本数相同、顺序打乱。返回 { X: Float32Array(n·256), y: Int32Array(n) } */
    function makeSet(n, r) {
        const y = new Int32Array(n), X = new Float32Array(n * D);
        for (let i = 0; i < n; i++) y[i] = i % K;
        LM.shuffle(y, r);
        for (let i = 0; i < n; i++) renderDigit(y[i], r, X.subarray(i * D, (i + 1) * D));
        return { X, y, n };
    }

    // ═══════════════════ 通用 MLP：sizes = [256, H, …, 10]，隐藏层激活 act，输出 softmax ═══════════════════
    // 约定（与 C++ 代码一致）：行向量，Z = A·W + b，W 的形状是 [fan_in × fan_out]；loss = batch 平均交叉熵
    /** 初始化标准差：太小 0.01 / 太大 1 / Xavier √(2/(fan_in+fan_out)) / He √(2/fan_in) */
    const initStd = (init, fi, fo) => (init === 'zero' ? 0 : init === 'small' ? 0.01 : init === 'large' ? 1 : init === 'xavier' ? Math.sqrt(2 / (fi + fo)) : Math.sqrt(2 / fi));
    function makeNet(sizes, act, init, rand, Arr) {
        Arr = Arr || Float32Array;
        const L = sizes.length - 1, W = [], b = [];
        for (let l = 0; l < L; l++) {
            const fi = sizes[l], fo = sizes[l + 1], w = new Arr(fi * fo), sd = initStd(init, fi, fo);
            if (sd > 0) for (let i = 0; i < w.length; i++) w[i] = sd * LM.randn(rand);
            W.push(w); b.push(new Arr(fo));
        }
        return { sizes: sizes.slice(), L, W, b, act, Arr };
    }
    function cloneNet(net, Arr) {
        Arr = Arr || net.Arr;
        return { sizes: net.sizes.slice(), L: net.L, act: net.act, Arr, W: net.W.map((w) => Arr.from(w)), b: net.b.map((v) => Arr.from(v)) };
    }
    function copyNet(dst, src) { src.W.forEach((w, l) => dst.W[l].set(w)); src.b.forEach((v, l) => dst.b[l].set(v)); }
    // 内核：同一份源码编译两次——KR 给 float32 训练用，KR64（new Function 重新编译出的副本）给 float64 梯度校验用。
    // 两份代码完全相同，但各有各的 V8 内联缓存；共用一份的话，Float64Array 会让热循环变成多态、训练慢近一倍。
    function KERNELS() {
        const GK = Math.sqrt(2 / Math.PI);
        /** 标量激活与导数（画图、单样本追踪用） */
        function act1(name, x) {
            switch (name) {
                case 'sigmoid': return 1 / (1 + Math.exp(-x));
                case 'tanh': return Math.tanh(x);
                case 'relu': return x > 0 ? x : 0;
                case 'gelu': return 0.5 * x * (1 + Math.tanh(GK * (x + 0.044715 * x * x * x)));
                default: return x;
            }
        }
        function dact1(name, x) {
            switch (name) {
                case 'sigmoid': { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }
                case 'tanh': { const t = Math.tanh(x); return 1 - t * t; }
                case 'relu': return x > 0 ? 1 : 0;
                case 'gelu': { const t = Math.tanh(GK * (x + 0.044715 * x * x * x)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GK * (1 + 3 * 0.044715 * x * x); }
                default: return 1;
            }
        }
        function actF(name, Z, A, n) {
            switch (name) {
                case 'sigmoid': for (let i = 0; i < n; i++) A[i] = 1 / (1 + Math.exp(-Z[i])); break;
                case 'tanh': for (let i = 0; i < n; i++) A[i] = Math.tanh(Z[i]); break;
                case 'relu': for (let i = 0; i < n; i++) A[i] = Z[i] > 0 ? Z[i] : 0; break;
                case 'gelu': for (let i = 0; i < n; i++) { const x = Z[i]; A[i] = 0.5 * x * (1 + Math.tanh(GK * (x + 0.044715 * x * x * x))); } break;
                default: for (let i = 0; i < n; i++) A[i] = Z[i];
            }
        }
        /** G ← G ⊙ act'(Z)（导数只由 Z 算，dropout 掩码另外乘） */
        function actB(name, Z, G, n) {
            switch (name) {
                case 'sigmoid': for (let i = 0; i < n; i++) { const s = 1 / (1 + Math.exp(-Z[i])); G[i] *= s * (1 - s); } break;
                case 'tanh': for (let i = 0; i < n; i++) { const t = Math.tanh(Z[i]); G[i] *= 1 - t * t; } break;
                case 'relu': for (let i = 0; i < n; i++) if (Z[i] <= 0) G[i] = 0; break;
                case 'gelu': for (let i = 0; i < n; i++) G[i] *= dact1('gelu', Z[i]); break;
                default:
            }
        }
        const nbuf = (o, k, n, Arr) => (o[k] && o[k].length === n && o[k] instanceof Arr ? o[k] : (o[k] = new Arr(n)));
        /** Z[n×m] = A[n×k]·W[k×m] + b（跳过 A 里的 0：图像大部分像素是 0） */
        function gemm(A, W, b, Z, n, k, m) {
            for (let i = 0; i < n; i++) {
                const zi = i * m, ai = i * k;
                for (let j = 0; j < m; j++) Z[zi + j] = b[j];
                for (let p = 0; p < k; p++) { const av = A[ai + p]; if (av === 0) continue; const wp = p * m; for (let j = 0; j < m; j++) Z[zi + j] += av * W[wp + j]; }
            }
        }
        /** 前向。cache.A[0] = X，cache.Z[l] = 第 l 层线性输出，cache.A[l+1] = act(Z[l])，cache.P = softmax(Z[L−1])。o.dropout = p（训练时） */
        function forward(net, X, n, cache, o) {
            o = o || {};
            const L = net.L, Arr = net.Arr;
            cache.n = n; cache.A = cache.A || []; cache.Z = cache.Z || []; cache.M = cache.M || [];
            cache.A[0] = X; cache.drop = o.dropout || 0;
            for (let l = 0; l < L; l++) {
                const fi = net.sizes[l], fo = net.sizes[l + 1], Z = nbuf(cache.Z, l, n * fo, Arr);
                gemm(cache.A[l], net.W[l], net.b[l], Z, n, fi, fo);
                if (l < L - 1) {
                    const A = nbuf(cache.A, l + 1, n * fo, Arr);
                    actF(net.act, Z, A, n * fo);
                    if (cache.drop > 0) {   // inverted dropout：训练时随机置零、其余放大 1/(1−p)，推理时什么都不做
                        const M = nbuf(cache.M, l, n * fo, Arr), keep = 1 - cache.drop, sc = 1 / keep, r = o.rand;
                        for (let i = 0; i < n * fo; i++) { const m = r() < keep ? sc : 0; M[i] = m; A[i] *= m; }
                    }
                }
            }
            const C = net.sizes[L], Zo = cache.Z[L - 1], P = nbuf(cache, 'P', n * C, Arr);   // C = 类别数（数字 10，XOR 2）
            cache.C = C;
            for (let i = 0; i < n; i++) {
                let mx = -Infinity; for (let j = 0; j < C; j++) if (Zo[i * C + j] > mx) mx = Zo[i * C + j];
                let s = 0; for (let j = 0; j < C; j++) { const e = Math.exp(Zo[i * C + j] - mx); P[i * C + j] = e; s += e; }
                for (let j = 0; j < C; j++) P[i * C + j] /= s;
            }
            return cache;
        }
        /** 平均交叉熵 −(1/n) Σ log P[i, y_i]（+ λ/2 Σ‖W‖²） */
        function lossOf(net, cache, y, wd) {
            let s = 0; for (let i = 0; i < cache.n; i++) s -= Math.log(Math.max(1e-30, cache.P[i * cache.C + y[i]]));
            s /= cache.n;
            if (wd) { let r = 0; net.W.forEach((w) => { for (let i = 0; i < w.length; i++) r += w[i] * w[i]; }); s += 0.5 * wd * r; }
            return s;
        }
        /**
         * 反向传播（手写）：δ_L = (P − Y)/n；dW_l = A_lᵀ·δ_l；db_l = Σ_rows δ_l；δ_{l−1} = (δ_l·W_lᵀ) ⊙ act'(Z_{l−1})。
         * grads.dW[l] / grads.db[l] / grads.dZ[l]（每层的 δ）；o.wd = L2 系数（只加在权重上）
         */
        function backward(net, y, cache, grads, o) {
            o = o || {};
            const n = cache.n, L = net.L, Arr = net.Arr;
            grads.dW = grads.dW || []; grads.db = grads.db || []; grads.dZ = grads.dZ || [];
            const C = cache.C;
            let dZ = nbuf(grads.dZ, L - 1, n * C, Arr);
            for (let i = 0; i < n * C; i++) dZ[i] = cache.P[i] / n;          // δ_L = (P − Y) / n
            for (let i = 0; i < n; i++) dZ[i * C + y[i]] -= 1 / n;
            for (let l = L - 1; l >= 0; l--) {
                const fi = net.sizes[l], fo = net.sizes[l + 1], A = cache.A[l], W = net.W[l];
                const dW = nbuf(grads.dW, l, fi * fo, Arr), db = nbuf(grads.db, l, fo, Arr);
                dW.fill(0); db.fill(0);
                for (let i = 0; i < n; i++) {
                    const ai = i * fi, zi = i * fo;
                    for (let j = 0; j < fo; j++) db[j] += dZ[zi + j];
                    for (let p = 0; p < fi; p++) { const av = A[ai + p]; if (av === 0) continue; const wp = p * fo; for (let j = 0; j < fo; j++) dW[wp + j] += av * dZ[zi + j]; }
                }
                if (o.wd) for (let i = 0; i < dW.length; i++) dW[i] += o.wd * W[i];
                if (l === 0) break;
                const dA = nbuf(grads.dZ, l - 1, n * fi, Arr);   // 先放 dA，再原地乘 act'(Z)
                for (let i = 0; i < n; i++) {
                    const zi = i * fo, ai = i * fi;
                    for (let p = 0; p < fi; p++) { const wp = p * fo; let s = 0; for (let j = 0; j < fo; j++) s += dZ[zi + j] * W[wp + j]; dA[ai + p] = s; }
                }
                if (cache.drop > 0) { const M = cache.M[l - 1]; for (let i = 0; i < n * fi; i++) dA[i] *= M[i]; }
                actB(net.act, cache.Z[l - 1], dA, n * fi);
                dZ = dA;
            }
            return grads;
        }
        return { act1, dact1, actF, actB, gemm, forward, lossOf, backward };
    }
    const KR = KERNELS(), KR64 = new Function('return (' + KERNELS.toString() + ')();')();
    const { act1, dact1, actF, actB, gemm, forward, lossOf, backward } = KR;

    // ═══════════════════ 优化器（逐元素，作用在 net.W / net.b 上） ═══════════════════
    function makeOpt(kind, net, hp) {
        const ps = net.W.concat(net.b);
        return { kind, lr: hp.lr, mu: hp.mu ?? 0.9, b1: hp.b1 ?? 0.9, b2: hp.b2 ?? 0.999, eps: hp.eps ?? 1e-8, t: 0, m: ps.map((p) => new Float32Array(p.length)), v: kind === 'adam' ? ps.map((p) => new Float32Array(p.length)) : null };
    }
    function optStep(opt, net, grads) {
        const ps = net.W.concat(net.b), gs = grads.dW.concat(grads.db);
        opt.t++;
        const { lr, mu, b1, b2, eps } = opt, bc1 = 1 - Math.pow(b1, opt.t), bc2 = 1 - Math.pow(b2, opt.t);
        ps.forEach((p, k) => {
            const g = gs[k], m = opt.m[k];
            if (opt.kind === 'sgd') for (let i = 0; i < p.length; i++) p[i] -= lr * g[i];
            else if (opt.kind === 'momentum') for (let i = 0; i < p.length; i++) { m[i] = mu * m[i] + g[i]; p[i] -= lr * m[i]; }
            else { const v = opt.v[k]; for (let i = 0; i < p.length; i++) { m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]; p[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps); } }
        });
    }

    // ═══════════════════ 训练 run：mini-batch 采样（按 epoch 洗牌）+ 分帧评估 ═══════════════════
    /**
     * cfg: { sizes, act, init, seed, opt: {kind, lr, …}, wd, dropout, X, y, idx（可用的训练样本下标）, B, cap, evalSets: [{ name, X, y, idx, conf }], evalEvery }
     */
    function makeRun(cfg) {
        const rand = LM.rng(cfg.seed), net = makeNet(cfg.sizes, cfg.act, cfg.init, rand);
        const R = {
            cfg, net, rand, opt: makeOpt(cfg.opt.kind, net, cfg.opt), cache: {}, grads: {}, step: 0, epoch: 0, cap: cfg.cap, B: cfg.B || BATCH,
            perm: Int32Array.from(cfg.idx), pos: 0, Xb: new Float32Array((cfg.B || BATCH) * cfg.sizes[0]), yb: new Int32Array(cfg.B || BATCH),
            lossEMA: null, batchLoss: [], hist: { step: [] }, ev: null, diverged: false, dropRand: LM.rng(cfg.seed ^ 0x5bd1e995),
            W0: net.W.map((w) => Float32Array.from(w)),   // 初始权重的拷贝（Step 08 画 ΔW₁ = W₁(t) − W₁(0)）
        };
        LM.shuffle(R.perm, R.rand);
        cfg.evalSets.forEach((s) => { R.hist[s.name + 'L'] = []; R.hist[s.name + 'A'] = []; });
        return R;
    }
    function nextBatch(R) {
        const B = R.B, d = R.cfg.sizes[0], X = R.cfg.X, y = R.cfg.y;
        for (let k = 0; k < B; k++) {
            if (R.pos >= R.perm.length) { LM.shuffle(R.perm, R.rand); R.pos = 0; R.epoch++; }
            const i = R.perm[R.pos++];
            R.Xb.set(X.subarray(i * d, (i + 1) * d), k * d); R.yb[k] = y[i];
        }
    }
    /** 一步训练：取 batch → 前向 → 反向 → 优化器更新。返回 batch loss */
    function trainStep(R) {
        nextBatch(R);
        const cfg = R.cfg;
        forward(R.net, R.Xb, R.B, R.cache, { dropout: cfg.dropout, rand: R.dropRand });
        const loss = lossOf(R.net, R.cache, R.yb, 0);
        backward(R.net, R.yb, R.cache, R.grads, { wd: cfg.wd });
        optStep(R.opt, R.net, R.grads);
        R.step++;
        if (!isFinite(loss)) R.diverged = true;
        R.lossEMA = R.lossEMA == null ? loss : 0.95 * R.lossEMA + 0.05 * loss;
        R.batchLoss.push(loss);
        return loss;
    }
    /** 评估器：拷一份权重快照，按帧分块地在各评估集上算 loss / 准确率（混淆矩阵可选）；全部算完记一个点 */
    function evalBegin(R) {
        const ev = R.ev || (R.ev = { net: cloneNet(R.net), cache: {} });
        copyNet(ev.net, R.net);
        ev.step = R.step; ev.set = 0; ev.pos = 0;
        ev.acc = R.cfg.evalSets.map(() => ({ loss: 0, correct: 0, n: 0 }));
        ev.conf = R.cfg.evalSets.map((s) => (s.conf ? new Int32Array(K * K) : null));
        ev.pred = R.cfg.evalSets.map((s) => (s.conf ? new Int8Array(s.idx.length) : null));
        ev.lossv = R.cfg.evalSets.map((s) => (s.conf ? new Float32Array(s.idx.length) : null));
    }
    const EVAL_CHUNK = 50;
    function evalWork(R, budget) {
        const ev = R.ev, sets = R.cfg.evalSets, d = R.cfg.sizes[0], buf = ev.xbuf || (ev.xbuf = new Float32Array(EVAL_CHUNK * d)), yb = ev.ybuf || (ev.ybuf = new Int32Array(EVAL_CHUNK));
        while (budget > 0 && ev.set < sets.length) {
            const s = sets[ev.set], n = Math.min(EVAL_CHUNK, budget, s.idx.length - ev.pos);
            for (let k = 0; k < n; k++) { const i = s.idx[ev.pos + k]; buf.set(s.X.subarray(i * d, (i + 1) * d), k * d); yb[k] = s.y[i]; }
            forward(ev.net, buf, n, ev.cache);
            const a = ev.acc[ev.set], P = ev.cache.P;
            for (let k = 0; k < n; k++) {
                const t = yb[k], pk = LM.argmax(P, k * K, K), lk = -Math.log(Math.max(1e-30, P[k * K + t]));
                a.loss += lk; a.correct += pk === t ? 1 : 0; a.n++;
                if (ev.conf[ev.set]) { ev.conf[ev.set][t * K + pk]++; ev.pred[ev.set][ev.pos + k] = pk; ev.lossv[ev.set][ev.pos + k] = lk; }
            }
            ev.pos += n; budget -= n;
            if (ev.pos >= s.idx.length) { ev.set++; ev.pos = 0; }
        }
        if (ev.set < sets.length) return false;
        const H = R.hist; H.step.push(ev.step);
        sets.forEach((s, k) => { H[s.name + 'L'].push(ev.acc[k].loss / ev.acc[k].n); H[s.name + 'A'].push(ev.acc[k].correct / ev.acc[k].n); });
        R.conf = ev.conf; R.pred = ev.pred; R.lossv = ev.lossv; R.evalStep = ev.step;
        return true;
    }
    /** 立刻完整评估一次（第 0 步、到达上限时） */
    function evalNow(R) { evalBegin(R); evalWork(R, 1e9); R.ev.pending = false; }
    /** 每帧调用：训练 spf 步 + 推进评估（评估集总样本数 / evalFrames 个样本） */
    function tick(R, spf, evalBudget) {
        if (R.step >= R.cap || R.diverged) return;
        for (let k = 0; k < spf && R.step < R.cap && !R.diverged; k++) trainStep(R);
        if (R.step >= R.cap || R.diverged) { evalNow(R); return; }
        if (!R.ev || !R.ev.pending) { evalBegin(R); R.ev.pending = true; }
        if (evalWork(R, evalBudget)) R.ev.pending = false;
    }

    // ═══════════════════ 三方梯度校验：手写反传（float64）vs 有限差分（float64）vs LM.AG 自动微分（float32） ═══════════════════
    const relErr = (a, b) => Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-12);
    const AGACT = { sigmoid: 'sigmoid', tanh: 'tanh', relu: 'relu', gelu: 'gelu' };
    /** net 为两层 MLP（float32 权重）；X[n×256], y[n]。返回逐元素对照表与每个张量的误差 */
    function gradCheck(net, X, y, n, eps) {
        eps = eps || 1e-5;
        const n64 = cloneNet(net, Float64Array), X64 = Float64Array.from(X.subarray(0, n * net.sizes[0]));
        const c64 = KR64.forward(n64, X64, n, {}), loss64 = KR64.lossOf(n64, c64, y), g64 = KR64.backward(n64, y, c64, {});
        // LM.AG：同一个网络用自动微分再算一遍（float32）
        const { AG } = LM;
        AG.clearTape();
        const P = [], names = [];
        net.W.forEach((w, l) => { const t = AG.param('W' + (l + 1), net.sizes[l], net.sizes[l + 1]); t.data.set(w); P.push(t); names.push('W' + (l + 1)); });
        net.b.forEach((v, l) => { const t = AG.param('b' + (l + 1), 1, net.sizes[l + 1]); t.data.set(v); P.push(t); names.push('b' + (l + 1)); });
        const L = net.L, Wt = P.slice(0, L), bt = P.slice(L);
        let h = AG.tensor(n, net.sizes[0], Float32Array.from(X.subarray(0, n * net.sizes[0])));
        for (let l = 0; l < L; l++) { h = AG.linear(h, Wt[l], bt[l]); if (l < L - 1) h = AG[AGACT[net.act]](h); }
        const lossT = AG.crossEntropy(h, y), lossAG = lossT.item();
        AG.backward(lossT);
        // 每个张量挑 5 个元素：|梯度| 最大的 3 个 + 2 个随机的非零梯度
        const rnd = LM.rng(12345), rows = [], per = [];
        const tensors = [];
        for (let l = 0; l < L; l++) tensors.push({ name: 'W' + (l + 1), p: n64.W[l], g: g64.dW[l], ag: Wt[l].grad });
        for (let l = 0; l < L; l++) tensors.push({ name: 'b' + (l + 1), p: n64.b[l], g: g64.db[l], ag: bt[l].grad });
        tensors.forEach((T) => {
            const order = Array.from(T.g, (v, i) => i).filter((i) => Math.abs(T.g[i]) > 1e-9).sort((a, b) => Math.abs(T.g[b]) - Math.abs(T.g[a]));
            const pick = order.slice(0, 3);
            for (let tries = 0; pick.length < Math.min(5, order.length) && tries < 50; tries++) { const i = order[Math.floor(rnd() * order.length)]; if (!pick.includes(i)) pick.push(i); }
            let worstNum = 0, num2 = 0, den2 = 0;
            pick.forEach((i) => {
                const old = T.p[i];
                T.p[i] = old + eps; const lp = KR64.lossOf(n64, KR64.forward(n64, X64, n, {}), y);
                T.p[i] = old - eps; const lm = KR64.lossOf(n64, KR64.forward(n64, X64, n, {}), y);
                T.p[i] = old;
                const num = (lp - lm) / (2 * eps), r1 = relErr(T.g[i], num), r2 = relErr(T.g[i], T.ag[i]);
                worstNum = Math.max(worstNum, r1);
                rows.push({ name: T.name, i, hand: T.g[i], num, ag: T.ag[i], relNum: r1, relAg: r2 });
            });
            for (let i = 0; i < T.g.length; i++) { num2 += (T.g[i] - T.ag[i]) ** 2; den2 += T.g[i] ** 2; }
            per.push({ name: T.name, size: T.g.length, relNum: worstNum, relAg: Math.sqrt(num2 / Math.max(den2, 1e-300)) });
        });
        return { rows, per, loss64, lossAG, eps, n };
    }

    // ═══════════════════ 第 05 步：XOR（四团点）上的线性网络 vs 带激活的 MLP ═══════════════════
    function xorData(rand, n) {
        const X = new Float32Array(2 * n), y = new Int32Array(n);
        for (let i = 0; i < n; i++) { const q = i % 4, sx = q & 1 ? 1 : -1, sy = q & 2 ? 1 : -1; X[2 * i] = sx + 0.32 * LM.randn(rand); X[2 * i + 1] = sy + 0.32 * LM.randn(rand); y[i] = sx * sy > 0 ? 1 : 0; }
        return { X, y, n };
    }
    /** 二维小网络（2 → 16 → 2）：全量 batch 的 Adam，复用同一套 forward / backward */
    function trainSmall(sizes, act, data, steps, seed) {
        const net = makeNet(sizes, act, 'xavier', LM.rng(seed)), opt = makeOpt('adam', net, { lr: 0.03 }), cache = {}, grads = {}, loss = [];
        for (let s = 0; s < steps; s++) { forward(net, data.X, data.n, cache); loss.push(lossOf(net, cache, data.y)); backward(net, data.y, cache, grads); optStep(opt, net, grads); }
        forward(net, data.X, data.n, cache);
        let ok = 0; for (let i = 0; i < data.n; i++) ok += LM.argmax(cache.P, i * cache.C, cache.C) === data.y[i] ? 1 : 0;
        return { net, loss, acc: ok / data.n };
    }

    // ═══════════════════ 第 09 步：二维病态函数上的 SGD / Momentum / Adam 轨迹 ═══════════════════
    // f(x, y) = 0.02·x² + 2·(y − 0.4·sin(1.2x))²：一条弯曲的窄峡谷，谷底沿 x 方向很平、横穿方向很陡
    const f2 = (x, y) => { const r = y - 0.4 * Math.sin(1.2 * x); return 0.02 * x * x + 2 * r * r; };
    const g2 = (x, y) => { const r = y - 0.4 * Math.sin(1.2 * x); return [0.04 * x - 4 * r * 0.48 * Math.cos(1.2 * x), 4 * r]; };
    // SGD 与 Momentum 用同一个 η = 0.4：横穿峡谷的曲率是 4，SGD 的稳定上限 2/4 = 0.5，所以 SGD 会来回锯齿；Momentum 的上限是 2(1+μ)/4 = 0.95
    const OPT2D = { sgd: { lr: 0.4 }, momentum: { lr: 0.4, mu: 0.9 }, adam: { lr: 0.2 } };
    function runOpt2d(kind, x0, y0, steps) {
        const hp = OPT2D[kind], pts = [[x0, y0]], fs = [f2(x0, y0)];
        let x = x0, y = y0, mx = 0, my = 0, vx = 0, vy = 0;
        for (let t = 1; t <= steps; t++) {
            const [gx, gy] = g2(x, y);
            if (kind === 'sgd') { x -= hp.lr * gx; y -= hp.lr * gy; }
            else if (kind === 'momentum') { mx = hp.mu * mx + gx; my = hp.mu * my + gy; x -= hp.lr * mx; y -= hp.lr * my; }
            else {
                const b1 = 0.9, b2 = 0.999;
                mx = b1 * mx + (1 - b1) * gx; my = b1 * my + (1 - b1) * gy; vx = b2 * vx + (1 - b2) * gx * gx; vy = b2 * vy + (1 - b2) * gy * gy;
                const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
                x -= hp.lr * (mx / c1) / (Math.sqrt(vx / c2) + 1e-8); y -= hp.lr * (my / c1) / (Math.sqrt(vy / c2) + 1e-8);
            }
            if (!isFinite(x) || Math.abs(x) > 1e3) break;
            pts.push([x, y]); fs.push(f2(x, y));
        }
        return { pts, fs };
    }

    // ═══════════════════ 第 10 步：深层 MLP 的逐层统计 ═══════════════════
    const stdOf = (a) => { let m = 0; for (let i = 0; i < a.length; i++) m += a[i]; m /= a.length; let v = 0; for (let i = 0; i < a.length; i++) v += (a[i] - m) * (a[i] - m); return Math.sqrt(v / a.length); };
    const INITS = ['small', 'xavier', 'he', 'large'];
    /** depth 个宽 64 的隐藏层；在 batch 上做一次前向 + 反向，返回每层激活 std、权重梯度 std、激活直方图 */
    function flowStats(depth, act, init, X, y, B, seed) {
        const sizes = [D]; for (let l = 0; l < depth; l++) sizes.push(64); sizes.push(K);
        const net = makeNet(sizes, act, init, LM.rng(seed)), cache = forward(net, X, B, {}), g = backward(net, y, cache, {});
        const aStd = [], gStd = [], aMean = [];
        for (let l = 1; l <= depth; l++) { aStd.push(stdOf(cache.A[l])); aMean.push(LM.mean(cache.A[l])); }
        for (let l = 0; l <= depth; l++) gStd.push(stdOf(g.dW[l]));
        const lo = act === 'sigmoid' ? 0 : act === 'tanh' ? -1 : -0.5, hi = act === 'sigmoid' ? 1 : act === 'tanh' ? 1 : 2.5, NB = 40, hist = new Float32Array(depth * NB);
        for (let l = 1; l <= depth; l++) {
            const A = cache.A[l];
            for (let i = 0; i < A.length; i++) { let b = Math.floor((A[i] - lo) / (hi - lo) * NB); b = b < 0 ? 0 : b >= NB ? NB - 1 : b; hist[(l - 1) * NB + b]++; }
            let mx = 0; for (let b = 0; b < NB; b++) mx = Math.max(mx, hist[(l - 1) * NB + b]);
            for (let b = 0; b < NB; b++) hist[(l - 1) * NB + b] = Math.log1p(50 * hist[(l - 1) * NB + b] / (mx || 1)) / Math.log(51);   // 对数拉伸：稀少的取值也看得见
        }
        return { aStd, aMean, gStd, hist, NB, lo, hi, loss: lossOf(net, cache, y) };
    }

    // ═══════════════════ 页面用的数据集 / 训练配置 / 单样本追踪 ═══════════════════
    const CAP = 3000, SPF = 4, EVAL_BUDGET = 50;       // 主训练：上限 3000 步、每帧 4 步、每帧评估 50 个样本（分帧评估）
    const OPT_CAP = 1200, REG_CAP = 1800, SPF2 = 2;    // Step 09 / 11 的对比训练（lazy：只在可见时运行；三个 run × 每帧 2 步）
    const range = (a, b) => Int32Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
    function makeData(rand) {
        const tr = makeSet(NTRAIN, rand), te = makeSet(NTEST, rand);
        const byClass = Array.from({ length: K }, () => []);
        for (let i = 0; i < te.n; i++) byClass[te.y[i]].push(i);
        const mean = new Float32Array(K * D), cnt = new Float32Array(K);
        for (let i = 0; i < tr.n; i++) { const c = tr.y[i]; cnt[c]++; for (let p = 0; p < D; p++) mean[c * D + p] += tr.X[i * D + p]; }
        for (let c = 0; c < K; c++) for (let p = 0; p < D; p++) mean[c * D + p] /= cnt[c] || 1;
        let nz = 0; for (let i = 0; i < tr.X.length; i++) if (tr.X[i] > 0) nz++;
        return { tr, te, byClass, mean, nzFrac: nz / tr.X.length, allTr: range(0, NTRAIN), allTe: range(0, NTEST), evalTr: range(0, 500), valIdx: range(NTRAIN - 500, NTRAIN) };
    }
    const evalSetsMain = (d) => [{ name: 'train', X: d.tr.X, y: d.tr.y, idx: d.evalTr }, { name: 'test', X: d.te.X, y: d.te.y, idx: d.allTe, conf: true }];
    function softCfg(S) { const d = S.data; return { sizes: [D, K], act: 'linear', init: 'zero', seed: S.seed + 11, opt: { kind: 'sgd', lr: S.lr }, X: d.tr.X, y: d.tr.y, idx: d.allTr, B: BATCH, cap: CAP, evalSets: evalSetsMain(d) }; }
    function mlpCfg(S, extra) { const d = S.data; return Object.assign({ sizes: [D, S.hidden, K], act: S.act, init: S.init, seed: S.seed + 22, opt: { kind: 'sgd', lr: S.lr }, X: d.tr.X, y: d.tr.y, idx: d.allTr, B: BATCH, cap: CAP, evalSets: evalSetsMain(d) }, extra); }
    function regCfg(S, extra) {
        const d = S.data, sub = range(0, S.nsub), subEval = range(0, Math.min(S.nsub, 250));   // 训练准确率只在其中 ≤ 250 张上评估（省时间）
        return mlpCfg(S, Object.assign({ idx: sub, cap: REG_CAP, seed: S.seed + 66, evalSets: [{ name: 'train', X: d.tr.X, y: d.tr.y, idx: subEval }, { name: 'val', X: d.tr.X, y: d.tr.y, idx: d.valIdx }, { name: 'test', X: d.te.X, y: d.te.y, idx: d.allTe }] }, extra));
    }
    const OPTS = [{ kind: 'sgd', name: 'SGD' }, { kind: 'momentum', name: 'Momentum' }, { kind: 'adam', name: 'Adam' }];
    const optHp = (k, S) => (k === 'adam' ? { kind: 'adam', lr: 1e-3 } : { kind: k, lr: S.lr, mu: 0.9 });
    /** 选中的测试样本在两个（正在训练的）模型里的前向 / 反向中间量（每帧重算，单样本很便宜） */
    function trace(S) {
        const d = S.data, i = Math.max(0, Math.min(d.te.n - 1, S.sample | 0)), x = d.te.X.subarray(i * D, (i + 1) * D), y = d.te.y[i];
        const t = S._trc || (S._trc = { cs: {}, c: {}, g: {}, ys: new Int32Array(1) });
        t.i = i; t.x = x; t.y = y; t.ys[0] = y;
        forward(S.soft.net, x, 1, t.cs);                                   // Softmax 回归（Step 04）
        t.zs = t.cs.Z[0]; t.ps = t.cs.P; t.predS = LM.argmax(t.ps); t.lossS = -Math.log(Math.max(1e-30, t.ps[y]));
        const net = S.mlp.net, H = net.sizes[1];
        forward(net, x, 1, t.c); backward(net, t.ys, t.c, t.g);            // MLP（Step 08）：前向 + 单样本反向
        t.H = H; t.z1 = t.c.Z[0]; t.h1 = t.c.A[1]; t.z2 = t.c.Z[1]; t.p = t.c.P; t.pred = LM.argmax(t.p); t.loss = -Math.log(Math.max(1e-30, t.p[y]));
        t.d2 = t.g.dZ[1]; t.d1 = t.g.dZ[0]; t.dW1 = t.g.dW[0]; t.dW2 = t.g.dW[1];
        if (!t.dH || t.dH.length !== H) t.dH = new Float32Array(H);
        for (let j = 0; j < H; j++) { let s = 0; for (let k = 0; k < K; k++) s += t.d2[k] * net.W[1][j * K + k]; t.dH[j] = s; }   // ∂ℓ/∂h₁ = δ₂W₂ᵀ
        return t;
    }
    const norm = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };
    const pct = (v) => (v == null || !isFinite(v) ? '—' : (100 * v).toFixed(1) + '%');
    const last = (a) => (a && a.length ? a[a.length - 1] : NaN);
    const ACTNAME = { relu: 'ReLU', gelu: 'GELU', tanh: 'tanh', sigmoid: 'sigmoid', linear: '无激活' };
    const INITNAME = { small: '太小 σ=0.01', xavier: 'Xavier', he: 'He', large: '太大 σ=1' };
    function status(R, N) {
        const ep = (R.step * R.B / N).toFixed(1);
        return `第 ${R.step} / ${R.cap} 步 · epoch ${ep}` + (R.diverged ? ' · 已发散（loss = NaN）' : R.step >= R.cap ? ' · 已停止' : ' · 训练中');
    }
    const statusShort = (R) => `第 ${R.step} / ${R.cap} 步` + (R.diverged ? ' · 已发散' : R.step >= R.cap ? ' · 已停止' : '');

    // ═══════════════════ 绘图辅助（深色主题、设计坐标；画布文字 ≥ 11 px） ═══════════════════
    const COL = { opt: [LM.C.blue, LM.C.orange, LM.C.green], reg: [LM.C.red, LM.C.blue, LM.C.green], init: [LM.C.purple, LM.C.cyan, LM.C.green, LM.C.red], act: { sigmoid: LM.C.purple, tanh: LM.C.cyan, relu: LM.C.orange, gelu: LM.C.green } };
    let offC = null;
    /** 把 16×16 的浮点图画到 (x, y, w, h)；stride / offset 用来直接读权重矩阵的一列 */
    function drawImg(ctx, data, x, y, w, h, o) {
        o = o || {};
        if (!offC) { offC = document.createElement('canvas'); offC.width = IMG; offC.height = IMG; }
        const octx = offC.getContext('2d'), im = octx.createImageData(IMG, IMG), map = PL.cmap[o.map || 'gray'], rgb = [0, 0, 0];
        const lo = o.range ? o.range[0] : 0, hi = o.range ? o.range[1] : 1, st = o.stride || 1, of = o.offset || 0, sp = hi - lo || 1;
        for (let i = 0; i < D; i++) {
            let t = (data[of + i * st] - lo) / sp; t = t < 0 ? 0 : t > 1 ? 1 : t; map(t, rgb);
            im.data[4 * i] = rgb[0]; im.data[4 * i + 1] = rgb[1]; im.data[4 * i + 2] = rgb[2]; im.data[4 * i + 3] = 255;
        }
        octx.putImageData(im, 0, 0);
        ctx.imageSmoothingEnabled = false; ctx.drawImage(offC, x, y, w, h); ctx.imageSmoothingEnabled = true;
    }
    function absMax(data, st, of, n) { let m = 0; for (let i = 0; i < n; i++) { const v = Math.abs(data[(of || 0) + i * (st || 1)]); if (v > m) m = v; } return m || 1e-9; }
    function txt(ctx, s, x, y, o) {
        o = o || {};
        const px = o.px || 11;
        ctx.font = LM.FONT(px, o.bold); ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'top';
        if (o.bg) {
            const tw = ctx.measureText(s).width, x0 = o.align === 'right' ? x - tw : o.align === 'center' ? x - tw / 2 : x;
            const y0 = o.base === 'middle' ? y - px / 2 - 2 : o.base === 'bottom' ? y - px - 3 : y - 2;
            ctx.fillStyle = o.bg; ctx.fillRect(x0 - 3, y0, tw + 6, px + 5);
        }
        ctx.fillStyle = o.color || LM.C.text; ctx.fillText(s, x, y);
    }
    /** 小图网格：items = [{ img, stride, offset, range, label, labelColor, outline }]。o: { cols, map, range, labels, header, title, titleColor, tx（标题 x）, gap } */
    function digitGrid(c, items, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const n = items.length, cols = o.cols || 10, rows = Math.max(1, Math.ceil(n / cols)), gap = o.gap ?? 3;
        const top = (o.title ? 18 : 0) + (o.header ? 15 : 0), lab = o.labels ? 14 : 0;
        const cell = Math.max(4, Math.floor(Math.min((w - gap) / cols - gap, (h - top - gap) / rows - gap - lab)));
        const x0 = Math.floor((w - (cols * (cell + gap) - gap)) / 2), y0 = top + gap;
        if (o.title) txt(ctx, o.title, o.tx ?? 6, 3, { bold: true, color: o.titleColor });
        if (o.header) o.header.forEach((s, k) => txt(ctx, String(s), x0 + k * (cell + gap) + cell / 2, y0 - 2, { align: 'center', base: 'bottom', color: LM.C.gray }));
        items.forEach((it, k) => {
            const cx = x0 + (k % cols) * (cell + gap), cy = y0 + Math.floor(k / cols) * (cell + gap + lab);
            drawImg(ctx, it.img, cx, cy, cell, cell, { map: o.map, range: it.range || o.range, stride: it.stride, offset: it.offset });
            if (it.outline) { ctx.strokeStyle = it.outline; ctx.lineWidth = 2; ctx.strokeRect(cx - 1, cy - 1, cell + 2, cell + 2); }
            if (o.labels && it.label != null) txt(ctx, String(it.label), cx + cell / 2, cy + cell + 1, { align: 'center', color: it.labelColor || LM.C.gray });
        });
        return { x0, y0, cell, gap, lab, cols, n };
    }
    const gridHit = (g, info) => {
        if (!g || !info) return -1;
        const cx = Math.floor((info.dx - g.x0) / (g.cell + g.gap)), cy = Math.floor((info.dy - g.y0) / (g.cell + g.gap + g.lab)), k = cy * g.cols + cx;
        return cx >= 0 && cx < g.cols && cy >= 0 && k < g.n ? k : -1;
    };
    /**
     * 竖直柱状图（文字 11 px）。o: { colors, colorFor(v, i), range, symmetric, minRange（自动量程的最小半幅）, ticks, title, titleColor, tx,
     *   top, values（柱旁写数值）, valueMin（|v| 小于它不写）, digits, hi（高亮下标）, labelColor(i) }
     */
    function barChart(c, vals, labels, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const n = vals.length, top = o.top ?? (o.title ? 24 : 22), padB = labels ? 18 : 6, padL = 44, padR = 6, dg = o.digits ?? 2;
        let lo = o.range ? o.range[0] : Math.min(0, ...vals), hi = o.range ? o.range[1] : Math.max(1e-9, ...vals);
        if (o.symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi), o.minRange || 0) || 1; lo = -m; hi = m; }
        if (!o.range && o.values) { const sp = hi - lo; if (hi > 0) hi += 0.16 * sp; if (lo < 0) lo -= 0.16 * sp; }   // 给柱旁的数值留出空间
        if (hi - lo < 1e-12) hi = lo + 1;
        const ph = h - top - padB, bw = (w - padL - padR) / n, Y = (v) => top + (hi - Math.min(hi, Math.max(lo, v))) / (hi - lo) * ph;
        const fmtT = (v) => (Math.abs(v) < 1e-12 ? '0' : Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : String(+v.toPrecision(2)));
        if (o.title) txt(ctx, o.title, o.tx ?? 6, 4, { bold: true, color: o.titleColor });
        ctx.lineWidth = 1;
        (o.ticks || (lo < 0 && hi > 0 ? [lo, 0, hi] : [lo, (lo + hi) / 2, hi])).forEach((v) => {
            ctx.strokeStyle = v === 0 ? '#3a3f46' : '#1c1c1c'; ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
            txt(ctx, fmtT(v), padL - 4, Y(v), { align: 'right', base: 'middle', color: LM.C.dim });
        });
        const z = Y(lo > 0 ? lo : hi < 0 ? hi : 0);
        for (let i = 0; i < n; i++) {
            const v = vals[i], x = padL + i * bw, y1 = Y(v);
            if (o.hi === i) { ctx.fillStyle = 'rgba(232,194,90,0.10)'; ctx.fillRect(x, top, bw, ph); }
            ctx.fillStyle = o.colorFor ? o.colorFor(v, i) : o.colors ? o.colors[i] : LM.C.blue;
            ctx.fillRect(x + bw * 0.14, Math.min(z, y1), Math.max(1, bw * 0.72), Math.max(1, Math.abs(y1 - z)));
            if (labels && (bw >= 11 || i % Math.ceil(11 / bw) === 0)) txt(ctx, String(labels[i]), x + bw / 2, h - padB + 3, { align: 'center', color: o.labelColor ? o.labelColor(i) : LM.C.gray });
            if (o.values && bw >= 26 && Math.abs(v) >= (o.valueMin ?? 0)) {
                const sv = Math.abs(v) < 0.5 * Math.pow(10, -dg) ? (0).toFixed(dg) : v.toFixed(dg);   // 不写「−0.00」
                txt(ctx, sv, x + bw / 2, v >= 0 ? Math.min(z, y1) - 2 : Math.max(z, y1) + 2, { align: 'center', base: v >= 0 ? 'bottom' : 'top', color: LM.C.text });
            }
        }
        return { padL, bw, top, ph, lo, hi };
    }
    /** 多行文字面板：rows = [[{ t, color, bold }…] | { text, color }]，只在中文标点后换行；o: { px, x0（第一行起点，避开 LIVE 角标）, indent } */
    function lines(c, rows, o) {
        o = o || {};
        const { ctx, w } = PL.draw.ctx(c), px = o.px || 11, lh = px + 5;
        let y = 6, first = true;
        rows.forEach((row) => {
            const segs = Array.isArray(row) ? row : [{ t: row.text, color: row.color, bold: row.bold }];
            let x = first ? (o.x0 ?? 6) : 6;
            segs.forEach((sg) => {
                ctx.font = LM.FONT(px, sg.bold);
                String(sg.t).split(/(?<=[，、；·：）])/).forEach((piece) => {
                    const tw = ctx.measureText(piece).width;
                    if (x + tw > w - 6 && x > 6 + (o.indent || 0)) { x = 6 + (o.indent || 0); y += lh; }
                    txt(ctx, piece, x, y, { px, bold: sg.bold, color: sg.color || LM.C.text });
                    x += tw;
                });
            });
            y += lh; first = false;
        });
        return y;
    }
    /** 图例框：items = [{ label, color, dash, noLine }]；(x, y) = 右上角（align = 'right'）或左上角 */
    function legend(ctx, items, x, y, align) {
        ctx.font = LM.FONT(11);
        const lw = Math.max(...items.map((it) => ctx.measureText(it.label).width)) + 26, lh = items.length * 15 + 6, x0 = align === 'left' ? x : x - lw;
        ctx.fillStyle = 'rgba(8,8,10,0.86)'; ctx.fillRect(x0, y, lw, lh);
        items.forEach((it, k) => {
            const yy = y + 4 + k * 15;
            if (!it.noLine) { ctx.strokeStyle = it.color; ctx.lineWidth = 2; ctx.setLineDash(it.dash || []); ctx.beginPath(); ctx.moveTo(x0 + 5, yy + 7); ctx.lineTo(x0 + 19, yy + 7); ctx.stroke(); ctx.setLineDash([]); }
            txt(ctx, it.label, x0 + (it.noLine ? 6 : 23), yy, { color: it.color });
        });
    }
    /** 折线图 + 可放在任意位置的图例。series: [{ x, y, color, dash, width, label }]；o = PL.draw.plot 的选项 + legendAt: [u, v]（图例角点的相对位置）、legendAlign、extra（额外图例行） */
    function curves(c, series, o) {
        PL.draw.plot(c, Object.assign({}, o, { series: series.map((s) => ({ data: s.y, x: s.x, color: s.color, dash: s.dash, width: s.width })) }));
        const items = series.filter((s) => s.label).map((s) => ({ label: s.label, color: s.color, dash: s.dash })).concat(o.extra || []);
        if (!items.length) return;
        const { ctx, w, h } = PL.draw.ctx(c, { keep: true });
        const padL = 50, padR = 10, padT = 10, padB = o.xLabel ? 34 : 21, at = o.legendAt || [1, 0], al = o.legendAlign || 'right';
        legend(ctx, items, padL + at[0] * (w - padL - padR) + (al === 'right' ? -4 : 4), padT + at[1] * (h - padT - padB) + 3, al);
    }
    /** 16×16 像素值面板：灰度格子 + 非零像素的数值 + 行列号 */
    function pixelPanel(c, x, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const top = o.title ? 20 : 2, padL = 20, padT = top + 14;
        const cs = Math.floor(Math.min((w - padL - 4) / IMG, (h - padT - 4) / IMG)), x0 = padL, y0 = padT;
        if (o.title) txt(ctx, o.title, 6, 3, { bold: true });
        for (let r = 0; r < IMG; r++) for (let q = 0; q < IMG; q++) {
            const v = x[r * IMG + q], g = Math.round(255 * Math.min(1, Math.max(0, v)));
            ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(x0 + q * cs, y0 + r * cs, cs, cs);
            if (v >= 0.05 && cs >= 18) txt(ctx, v >= 0.95 ? '1' : v.toFixed(1).replace(/^0/, ''), x0 + (q + 0.5) * cs, y0 + (r + 0.5) * cs + 1, { align: 'center', base: 'middle', color: g > 130 ? '#000' : '#d8d8d8' });
        }
        ctx.strokeStyle = 'rgba(90,90,90,0.45)'; ctx.lineWidth = 1;
        for (let k = 0; k <= IMG; k++) { ctx.beginPath(); ctx.moveTo(x0 + k * cs, y0); ctx.lineTo(x0 + k * cs, y0 + IMG * cs); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x0, y0 + k * cs); ctx.lineTo(x0 + IMG * cs, y0 + k * cs); ctx.stroke(); }
        for (let k = 0; k < IMG; k++) { txt(ctx, String(k), x0 + (k + 0.5) * cs, y0 - 2, { align: 'center', base: 'bottom', color: LM.C.dim }); txt(ctx, String(k), x0 - 3, y0 + (k + 0.5) * cs, { align: 'right', base: 'middle', color: LM.C.dim }); }
        if (o.hiRow != null) { ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 2; ctx.strokeRect(x0 - 1, y0 + o.hiRow * cs - 1, IMG * cs + 2, cs + 2); }
        return { x0, y0, cs };
    }
    /** Step 06 的网络图：输入图像 → H 个隐藏单元（颜色 = 激活）→ 10 个输出概率；连线 = 对预测类别的贡献 h₁ⱼ·W₂[j, ŷ] */
    function netDiagram(c, S) {
        const { ctx, w, h } = PL.draw.ctx(c), t = S.tr, W2 = S.mlp.net.W[1], H = t.H;
        txt(ctx, `样本 #${t.i}（标签 ${t.y}）→ MLP 预测 ${t.pred}`, 40, 3, { bold: true, color: t.pred === t.y ? LM.C.green : LM.C.red });
        const top = 26, bot = h - 40;
        const iw = 96, ix = 6, iy = (top + bot) / 2 - iw / 2;
        drawImg(ctx, t.x, ix, iy, iw, iw);
        txt(ctx, 'x · 256 维', ix + iw / 2, iy + iw + 5, { align: 'center', color: LM.C.gray });
        const rows = Math.min(16, H), cols = H / rows, hx0 = 132, hw = Math.min(84, 21 * cols), cw = hw / cols, ch = (bot - top) / rows, sz = Math.max(3, Math.min(cw, ch) - 3), pos = [];
        for (let j = 0; j < H; j++) pos.push([hx0 + (j % cols + 0.5) * cw, top + (Math.floor(j / cols) + 0.5) * ch]);
        ctx.strokeStyle = 'rgba(140,150,160,0.13)'; ctx.lineWidth = 1;   // 输入 → 隐藏：全连接（只画几根示意）
        for (let j = 0; j < H; j += Math.max(1, Math.floor(H / 12))) { ctx.beginPath(); ctx.moveTo(ix + iw + 2, iy + iw / 2); ctx.lineTo(pos[j][0] - sz / 2, pos[j][1]); ctx.stroke(); }
        const ox = 270, ow = 80, oh = (bot - top) / K, oy = (k) => top + (k + 0.5) * oh;
        const contrib = Array.from({ length: H }, (_, j) => t.h1[j] * W2[j * K + t.pred]);
        const cm = Math.max(1e-9, ...contrib.map(Math.abs));
        contrib.map((v, j) => j).sort((a, b) => Math.abs(contrib[b]) - Math.abs(contrib[a])).slice(0, 24).forEach((j) => {
            const v = contrib[j], a = Math.abs(v) / cm;
            ctx.strokeStyle = v > 0 ? `rgba(255,176,90,${(0.2 + 0.75 * a).toFixed(3)})` : `rgba(90,176,255,${(0.2 + 0.75 * a).toFixed(3)})`;
            ctx.lineWidth = 0.6 + 3 * a; ctx.beginPath(); ctx.moveTo(pos[j][0] + sz / 2, pos[j][1]); ctx.lineTo(ox - 14, oy(t.pred)); ctx.stroke();
        });
        const hm = absMax(t.h1, 1, 0, H), rgb = [0, 0, 0];
        for (let j = 0; j < H; j++) {
            PL.cmap.diverge(0.5 + 0.5 * t.h1[j] / hm, rgb); ctx.fillStyle = LM.rgbStr(rgb);
            ctx.fillRect(pos[j][0] - sz / 2, pos[j][1] - sz / 2, sz, sz); ctx.strokeStyle = '#2d333b'; ctx.lineWidth = 1; ctx.strokeRect(pos[j][0] - sz / 2, pos[j][1] - sz / 2, sz, sz);
        }
        txt(ctx, `h₁ = ${ACTNAME[S.act]}(z₁) · ${H} 个`, hx0 + hw / 2, bot + 5, { align: 'center', color: LM.C.gray });
        for (let k = 0; k < K; k++) {
            const y = oy(k), col = k === t.y ? LM.C.green : k === t.pred ? LM.C.red : LM.C.blue;
            txt(ctx, String(k), ox - 4, y, { align: 'right', base: 'middle', color: k === t.y ? LM.C.green : LM.C.gray });
            ctx.fillStyle = '#1a1f26'; ctx.fillRect(ox, y - oh * 0.32, ow, oh * 0.64);
            ctx.fillStyle = col; ctx.fillRect(ox, y - oh * 0.32, Math.max(1, ow * t.p[k]), oh * 0.64);
        }
        txt(ctx, 'p = softmax(z₂)', ox + ow / 2 - 8, bot + 5, { align: 'center', color: LM.C.gray });
        txt(ctx, `z₂[${t.pred}] = Σ h₁ⱼ·W₂[j, ${t.pred}] + b₂ = ${t.z2[t.pred].toFixed(2)}，p = ${t.p[t.pred].toFixed(3)}`, 6, h - 16, { color: LM.C.dim });
    }
    function arrow(ctx, x0, y0, x1, y1, color) {
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        const a = Math.atan2(y1 - y0, x1 - x0); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 6 * Math.cos(a - 0.45), y1 - 6 * Math.sin(a - 0.45)); ctx.lineTo(x1 - 6 * Math.cos(a + 0.45), y1 - 6 * Math.sin(a + 0.45)); ctx.fill();
    }
    /** Step 07 的计算图：左 = 前向值，右 = 反向梯度（选中样本、当前权重，全是真实数值） */
    function compGraph(c, S) {
        const { ctx, w } = PL.draw.ctx(c), t = S.tr, net = S.mlp.net, H = t.H, f2s = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2));
        txt(ctx, '计算图 · 样本 #' + t.i + ' · ℓ = ' + t.loss.toFixed(3), 40, 3, { bold: true });
        txt(ctx, '前向（值）↓', 8, 24, { color: LM.C.blue }); txt(ctx, '↑ 反向（梯度）', w - 8, 24, { align: 'right', color: LM.C.red });
        const bx = 168, bw = 64, bh = 22, ys = [56, 120, 184, 248, 312], L0 = bx - bw / 2 - 8, R0 = bx + bw / 2 + 10, f = (a) => f2s(norm(a));
        const nodes = [
            { name: 'x', shape: '[256]', fwd: '‖x‖ = ' + f(t.x), bwd: ['（输入，不求梯度）'] },
            { name: 'z₁', shape: '[' + H + ']', fwd: '‖z₁‖ = ' + f(t.z1), bwd: ['δ₁ = ∂ℓ/∂z₁', '‖δ₁‖ = ' + f(t.d1)] },
            { name: 'h₁', shape: '[' + H + ']', fwd: '‖h₁‖ = ' + f(t.h1), bwd: ['∂ℓ/∂h₁ = δ₂·W₂ᵀ', '‖·‖ = ' + f(t.dH)] },
            { name: 'z₂', shape: '[10]', fwd: '‖z₂‖ = ' + f(t.z2), bwd: ['δ₂ = p − y', '‖δ₂‖ = ' + f(t.d2)] },
            { name: 'ℓ', shape: '', fwd: 'ℓ = ' + t.loss.toFixed(3), bwd: ['∂ℓ/∂ℓ = 1'] },
        ];
        let active = 0; for (let j = 0; j < H; j++) if (t.z1[j] > 0) active++;
        const ops = [
            { name: '·W₁ + b₁', left: '‖W₁‖ = ' + f(net.W[0]), right: ['dW₁ = xᵀ·δ₁', '‖dW₁‖ = ' + f(t.dW1)] },
            { name: ACTNAME[S.act], left: S.act === 'relu' ? 'z₁>0：' + active + '/' + H : '逐元素', right: ['⊙ act′(z₁)'] },
            { name: '·W₂ + b₂', left: '‖W₂‖ = ' + f(net.W[1]), right: ['dW₂ = h₁ᵀ·δ₂', '‖dW₂‖ = ' + f(t.dW2)] },
            { name: 'softmax+CE', left: 'p[y] = ' + t.p[t.y].toFixed(3), right: [] },
        ];
        nodes.forEach((nd, k) => {
            const y = ys[k];
            ctx.fillStyle = '#16202a'; ctx.strokeStyle = '#3d5a73'; ctx.lineWidth = 1.2; ctx.fillRect(bx - bw / 2, y - bh / 2, bw, bh); ctx.strokeRect(bx - bw / 2, y - bh / 2, bw, bh);
            txt(ctx, nd.name + (nd.shape ? ' ' + nd.shape : ''), bx, y + 1, { align: 'center', base: 'middle', bold: true });
            txt(ctx, nd.fwd, L0, y, { align: 'right', base: 'middle', color: LM.C.blue });
            nd.bwd.forEach((sv, q) => txt(ctx, sv, R0, y - (nd.bwd.length - 1) * 7 + q * 14, { base: 'middle', color: LM.C.red }));
        });
        ops.forEach((op, k) => {
            const y0 = ys[k] + bh / 2, y1 = ys[k + 1] - bh / 2, ym = (y0 + y1) / 2;
            arrow(ctx, bx - 12, y0 + 2, bx - 12, y1 - 2, LM.C.blue); arrow(ctx, bx + 12, y1 - 2, bx + 12, y0 + 2, LM.C.red);
            txt(ctx, op.name, bx - 18, ym - 7, { align: 'right', base: 'middle', color: LM.C.gray });
            txt(ctx, op.left, bx - 18, ym + 7, { align: 'right', base: 'middle', color: LM.C.dim });
            op.right.forEach((sv, q) => txt(ctx, sv, bx + 22, ym - (op.right.length - 1) * 7 + q * 14, { base: 'middle', color: q ? LM.C.dim : '#d98a8f' }));
        });
    }
    /** Step 07 的三方梯度校验面板：每个张量两根对数刻度的误差条 + 逐元素对照表 */
    function gcPanel(c, gc, S) {
        const { ctx, w, h } = PL.draw.ctx(c);
        txt(ctx, `梯度校验 · ${gc.n} 张训练图 · 初始权重 · H = ${S.hidden} · ${ACTNAME[S.act]}`, 6, 3, { bold: true });
        const x0 = 76, x1 = w - 66, X = (v) => x0 + (Math.log10(Math.max(1e-13, v)) + 12) / 10 * (x1 - x0), yA = 40;
        ctx.strokeStyle = '#1c1c1c'; ctx.lineWidth = 1;
        for (let e = -12; e <= -2; e += 2) { const x = X(Math.pow(10, e)); ctx.beginPath(); ctx.moveTo(x, yA - 4); ctx.lineTo(x, yA + 8 * 17); ctx.stroke(); txt(ctx, '10' + String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]), x, yA - 6, { align: 'center', base: 'bottom', color: LM.C.dim }); }
        ctx.setLineDash([3, 3]); ctx.strokeStyle = '#555'; const xt = X(1e-7); ctx.beginPath(); ctx.moveTo(xt, yA - 2); ctx.lineTo(xt, yA + 8 * 17); ctx.stroke(); ctx.setLineDash([]);
        let y = yA;
        gc.per.forEach((p) => {
            [['差分', p.relNum, LM.C.green], ['LM.AG', p.relAg, LM.C.orange]].forEach(([lab, v, col], q) => {
                if (!q) txt(ctx, p.name, 6, y + 1, { bold: true });
                txt(ctx, lab, x0 - 4, y + 7, { align: 'right', base: 'middle', color: col });
                ctx.fillStyle = col; ctx.globalAlpha = 0.85; ctx.fillRect(x0, y + 2, Math.max(1, X(v) - x0), 10); ctx.globalAlpha = 1;
                txt(ctx, v.toExponential(1), X(v) + 4, y + 7, { base: 'middle', color: LM.C.text });
                y += 16;
            });
            y += 2;
        });
        txt(ctx, '虚线 = 10⁻⁷：CS231n 的经验，低于它基本可以放心', x0, y + 2, { color: LM.C.dim });
        y += 22;
        const cols = [6, 104, 190, 276];
        ['元素', '手写 f64', '差分 f64', 'LM.AG f32'].forEach((s, k) => txt(ctx, s, cols[k], y, { color: [LM.C.gray, LM.C.blue, LM.C.green, LM.C.orange][k] }));
        y += 15;
        const show = [gc.rows.find((r) => r.name === 'W1'), gc.rows.find((r) => r.name === 'b1'), gc.rows.find((r) => r.name === 'W2'), gc.rows.find((r) => r.name === 'b2')].filter(Boolean);
        show.forEach((r) => {
            const H = S.hidden, nm = r.name === 'W1' ? `W1[${Math.floor(r.i / H)},${r.i % H}]` : r.name === 'W2' ? `W2[${Math.floor(r.i / K)},${r.i % K}]` : `${r.name}[${r.i}]`;
            [nm, r.hand.toExponential(5), r.num.toExponential(5), r.ag.toExponential(5)].forEach((s, k) => txt(ctx, s, cols[k], y, { color: k ? LM.C.text : LM.C.gray }));
            y += 15;
        });
        txt(ctx, `loss：f64 ${gc.loss64.toFixed(7)} · f32 ${gc.lossAG.toFixed(7)}`, 6, y + 3, { color: LM.C.dim });
    }
    /** 混淆矩阵：行 = 真实类别，列 = 预测类别；对角线绿（按行占比），非对角红（按最大非对角计数） */
    function confusion(c, conf, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const padL = 40, padT = 44, cs = Math.floor(Math.min((w - padL - 8) / K, (h - padT - 8) / K)), x0 = padL, y0 = padT;
        if (o.title) txt(ctx, o.title, 40, 3, { bold: true });
        txt(ctx, '预测 →', x0 + K * cs / 2, 18, { align: 'center', color: LM.C.gray });
        txt(ctx, '真实↓', 4, y0 - 14, { color: LM.C.gray });
        let mo = 1; for (let r = 0; r < K; r++) for (let q = 0; q < K; q++) if (r !== q) mo = Math.max(mo, conf[r * K + q]);
        for (let r = 0; r < K; r++) {
            let rs = 0; for (let q = 0; q < K; q++) rs += conf[r * K + q];
            for (let q = 0; q < K; q++) {
                const v = conf[r * K + q], X = x0 + q * cs, Y = y0 + r * cs;
                ctx.fillStyle = r === q ? `rgba(90,208,122,${(0.15 + 0.75 * v / Math.max(1, rs)).toFixed(3)})` : v ? `rgba(224,108,117,${(0.2 + 0.7 * v / mo).toFixed(3)})` : '#101317';
                ctx.fillRect(X + 1, Y + 1, cs - 2, cs - 2);
                if (v) txt(ctx, String(v), X + cs / 2, Y + cs / 2 + 1, { align: 'center', base: 'middle', color: '#f0f0f0' });
            }
            txt(ctx, String(r), x0 - 6, y0 + (r + 0.5) * cs, { align: 'right', base: 'middle', color: LM.C.gray });
            txt(ctx, String(r), x0 + (r + 0.5) * cs, y0 - 3, { align: 'center', base: 'bottom', color: LM.C.gray });
        }
        return { x0, y0, cs };
    }
    /** Step 05：XOR 平面上的决策区域（背景 = P(类 1)），MLP 再叠加每个隐藏单元的「分界线」w·x + b = 0 */
    function xorPlot(c, S, which) {
        const X = S.xor, r = X[which], d = X.data, G = r.grid, NG = X.NG, rng = X.R;
        const pts = []; for (let i = 0; i < d.n; i++) pts.push({ x: d.X[2 * i], y: d.X[2 * i + 1], color: d.y[i] ? LM.C.orange : LM.C.blue, r: 2.6, ring: r.pred[i] !== d.y[i] ? LM.C.red : null });
        const lines = [];
        if (which === 'mlp') {
            const W1 = r.net.W[0], b1 = r.net.b[0], Hh = r.net.sizes[1];
            for (let j = 0; j < Hh; j++) {
                const wx = W1[j], wy = W1[Hh + j], b = b1[j];
                if (Math.abs(wy) > Math.abs(wx)) lines.push({ a: [-9, -(b - 9 * wx) / wy], b: [9, -(b + 9 * wx) / wy], color: 'rgba(230,230,230,0.28)' });
                else if (Math.abs(wx) > 1e-9) lines.push({ a: [-(b - 9 * wy) / wx, -9], b: [-(b + 9 * wy) / wx, 9], color: 'rgba(230,230,230,0.28)' });
            }
        }
        const bg = (x, y) => {
            const i = Math.min(NG - 1, Math.max(0, Math.floor((x + rng) / (2 * rng) * NG))), j = Math.min(NG - 1, Math.max(0, Math.floor((rng - y) / (2 * rng) * NG)));
            const p = G[j * NG + i], a = Math.min(1, Math.abs(p - 0.5) * 2.2), base = p > 0.5 ? [120, 72, 28] : [26, 64, 110];
            return [14 + (base[0] - 14) * a, 17 + (base[1] - 17) * a, 22 + (base[2] - 22) * a];
        };
        const title = which === 'mlp' ? `${ACTNAME[S.act]} MLP（2 → 16 → 2）· 准确率 ${pct(r.acc)}` : `去掉激活（两层线性）· 准确率 ${pct(r.acc)}`;
        return LM.draw.scatter(c, pts, { x: [-rng, rng], y: [-rng, rng], bg, bgRes: NG, lines, equal: true, title, xLabel: 'x₁', yLabel: 'x₂' });
    }
    /** Step 09：峡谷函数的背景（log f）+ 三条优化轨迹 */
    function valleyPlot(c, S) {
        const rgb = [0, 0, 0];
        const bg = (x, y) => { const v = (Math.log10(f2(x, y) + 1e-3) + 3) / 4.4; PL.cmap.viridis(Math.min(1, Math.max(0, v)), rgb); return [rgb[0] * 0.62, rgb[1] * 0.62, rgb[2] * 0.62]; };
        const paths = S.o2.map((r, k) => ({ pts: r.pts, color: COL.opt[k], width: 1.5, dots: 1.5 }));
        const st = S.o2start;
        return LM.draw.scatter(c, [{ x: st[0], y: st[1], color: '#ffffff', r: 4.5, label: '起点' }, { x: 0, y: 0, color: LM.C.yellow, r: 3.5, label: '最小值' }], { x: [-5, 5], y: [-2.6, 2.6], bg, bgRes: 80, paths, xLabel: 'θ₁', yLabel: 'θ₂' });
    }

    return {
        IMG, D, K, NTRAIN, NTEST, BATCH, CAP, SPF, EVAL_BUDGET, OPT_CAP, REG_CAP, SPF2, OPTS, COL, ACTNAME, INITNAME,
        strokes, renderDigit, makeSet, makeData, softCfg, mlpCfg, regCfg, optHp, trace, norm, pct, last, status, statusShort, range,
        act1, dact1, actF, actB, initStd, makeNet, cloneNet, copyNet, gemm, forward, lossOf, backward, makeOpt, optStep,
        makeRun, trainStep, evalBegin, evalWork, evalNow, tick, relErr, gradCheck, xorData, trainSmall, f2, g2, OPT2D, runOpt2d, stdOf, INITS, flowStats,
        drawImg, absMax, txt, digitGrid, gridHit, barChart, lines, legend, curves, pixelPanel, netDiagram, compGraph, gcPanel, confusion, xorPlot, valleyPlot,
    };
})();

PipelineLab.mount({
    paramSteps: { sample: ['data', 'softmax', 'xent', 'forward', 'backprop'], lr: ['softmax', 'xent', 'trainSoft', 'forward', 'backprop', 'trainMLP', 'optim'], hidden: ['forward', 'backprop', 'trainMLP', 'optim', 'regularize', 'scaleup'], act: ['nonlin', 'forward', 'backprop', 'trainMLP', 'optim', 'init', 'regularize', 'scaleup'], init: ['forward', 'backprop', 'trainMLP', 'optim', 'init', 'regularize'], depth: ['init'], nsub: ['regularize'], wd: ['regularize'], dropout: ['regularize'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '02 · 手写数字：从 Softmax 回归到 MLP 与反向传播',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 20251123,
    css: LM.CSS,
    regen: { label: '重新生成数据', title: '用新的随机种子重新生成 4000 + 500 张数字（只重算 RESIDENT），所有训练从头开始' },
    state: { o2start: [-4.5, 1.6] },

    params: [
        { id: 'sample', label: '样本 #', min: 0, max: 499, step: 1, value: 43, affects: 'frame', desc: '选中的测试样本（共 500 张）。Step 02 / 03 用 Softmax 回归分析它，Step 06 / 07 用 MLP 分析它；也可以直接点 Step 01 的网格或 Step 08 的错例。' },
        { id: 'lr', label: '学习率 η', options: [0.01, 0.03, 0.1, 0.3, 1], value: 0.1, desc: 'SGD 的步长：Step 04 / 08 / 11 的训练，以及 Step 09 里 SGD 和 Momentum 的 η。0.01 学得很慢；1 更快但曲线更抖。改了会从头重训。' },
        { id: 'hidden', label: '隐藏层宽度 H', options: [16, 32, 64, 128], value: 64, desc: 'MLP 的隐藏单元个数（参数量 267H + 10）。H 越大容量越大：训练准确率更容易到 100%，测试准确率在 64 附近饱和（Step 06–09、11）。' },
        { id: 'act', label: '激活函数', options: [{ value: 'relu', label: 'ReLU' }, { value: 'gelu', label: 'GELU' }, { value: 'tanh', label: 'tanh' }, { value: 'sigmoid', label: 'sigmoid' }], value: 'relu', desc: '隐藏层的非线性（Step 05）。sigmoid 的导数 ≤ 1/4：训练最慢，深层时梯度消失（Step 10）；ReLU / GELU 是现代默认。' },
        { id: 'init', label: '初始化', options: [{ value: 'small', label: '太小 σ=0.01' }, { value: 'xavier', label: 'Xavier' }, { value: 'he', label: 'He' }, { value: 'large', label: '太大 σ=1' }], value: 'he', desc: '权重的初始标准差（偏置一律为 0）。Step 10 看它对深层网络逐层信号的影响；Step 08 的训练也用它——「太大」会让初始 loss 很高、最终准确率明显变差。' },
        { id: 'depth', label: '深度（Step 10）', min: 2, max: 16, step: 1, value: 8, affects: 'frame', desc: 'Step 10 里深层 MLP 的隐藏层数（每层 64 个单元）。越深，逐层的放大 / 缩小因子被连乘的次数越多：梯度消失或爆炸越明显。不影响其他步骤。' },
        { id: 'nsub', label: '小训练集 N', options: [100, 200, 500, 1000], value: 500, affects: 'frame', desc: 'Step 11 只用前 N 张训练图训练。N 越小越容易过拟合；从 100 拖到 1000，看「更多数据」这个最有效的正则化。' },
        { id: 'wd', label: '权重衰减 λ', options: [{ value: 0, label: '0' }, { value: 1e-4, label: '1e-4' }, { value: 1e-3, label: '1e-3' }, { value: 1e-2, label: '1e-2' }, { value: 3e-2, label: '3e-2' }, { value: 1e-1, label: '1e-1' }], value: 1e-2, affects: 'frame', desc: 'Step 11 蓝色那组的 L2 正则系数：loss 加 λ/2·‖W‖²。1e-2 左右能压住验证 loss 的回升；1e-1 太强，训练准确率都上不去（欠拟合）。' },
        { id: 'dropout', label: 'dropout p', options: [0.2, 0.35, 0.5, 0.65], value: 0.5, affects: 'frame', desc: 'Step 11 绿色那组：训练时每个隐藏单元被置零的概率（推理时不置零）。p 越大正则越强、训练越慢。' },
        { id: 'retrain', label: '重新训练', action: (S) => { S.retrain = (S.retrain || 0) + 1; }, affects: 'param', desc: '把所有训练（Step 04 / 08 / 09 / 11）从第 0 步重新开始，再看一遍学习过程。' },
    ],

    stages: [
        { id: 'data', lifetime: 'resident', run(S) { S.data = P02.makeData(S.rand); } },
        {
            id: 'gcheck', lifetime: 'param', run(S) {
                // 与 Step 08 的 MLP 用同一组初始权重（同一个种子），在 16 张训练图上做三方校验
                const key = [S.hidden, S.act, S.init, S.seed].join();
                if (S._gcKey === key) return;
                S._gcKey = key;
                const net = P02.makeNet([P02.D, S.hidden, P02.K], S.act, S.init, LM.rng(S.seed + 22));
                S.gc = P02.gradCheck(net, S.data.tr.X, S.data.tr.y, 16);
            },
        },
        {
            id: 'xor', lifetime: 'param', lazy: true, run(S) {
                const key = S.act + ':' + S.seed;
                if (S._xorKey === key) return;
                S._xorKey = key;
                const NG = 64, R = 2.5, data = P02.xorData(LM.rng(S.seed + 55), 160);
                const fit = (act) => {   // 训练 + 在 64×64 网格上算 P(类 1)（背景）+ 每个点的预测
                    const r = P02.trainSmall([2, 16, 2], act, data, 300, S.seed + 56), n = NG * NG, G = new Float32Array(2 * n);
                    for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) { G[2 * (j * NG + i)] = -R + (i + 0.5) / NG * 2 * R; G[2 * (j * NG + i) + 1] = R - (j + 0.5) / NG * 2 * R; }
                    const cg = P02.forward(r.net, G, n, {}), grid = new Float32Array(n);
                    for (let k = 0; k < n; k++) grid[k] = cg.P[2 * k + 1];
                    const cd = P02.forward(r.net, data.X, data.n, {}), pred = new Int8Array(data.n);
                    for (let i = 0; i < data.n; i++) pred[i] = cd.P[2 * i + 1] > cd.P[2 * i] ? 1 : 0;
                    return Object.assign(r, { grid, pred });
                };
                S.xor = { data, NG, R, lin: fit('linear'), mlp: fit(S.act) };
            },
        },
        {
            id: 'soft', lifetime: 'history', run(S) {   // Step 04：Softmax 回归，每帧 4 步 SGD，3000 步后停止
                if (S._softEp !== S.epoch) { S._softEp = S.epoch; S.soft = P02.makeRun(P02.softCfg(S)); P02.evalNow(S.soft); }
                P02.tick(S.soft, P02.SPF, P02.EVAL_BUDGET);
            },
        },
        {
            id: 'mlp', lifetime: 'history', run(S) {    // Step 08：MLP，同样每帧 4 步
                if (S._mlpEp !== S.epoch) { S._mlpEp = S.epoch; S.mlp = P02.makeRun(P02.mlpCfg(S)); P02.evalNow(S.mlp); }
                P02.tick(S.mlp, P02.SPF, P02.EVAL_BUDGET);
            },
        },
        {
            id: 'optRuns', lifetime: 'history', lazy: true, run(S) {   // Step 09：三个优化器，同一初始化、同一串 batch（只在可见时运行）
                if (S._optEp !== S.epoch) {
                    S._optEp = S.epoch;
                    S.optRuns = P02.OPTS.map((o) => P02.makeRun(P02.mlpCfg(S, { opt: P02.optHp(o.kind, S), cap: P02.OPT_CAP, seed: S.seed + 33, evalSets: [{ name: 'test', X: S.data.te.X, y: S.data.te.y, idx: S.data.allTe }] })));   // 只评估测试集（训练曲线用 batch loss）
                    S.optRuns.forEach((R) => P02.evalNow(R));
                }
                S.optRuns.forEach((R) => P02.tick(R, P02.SPF2, P02.EVAL_BUDGET));
            },
        },
        {
            id: 'regRuns', lifetime: 'history', lazy: true, run(S) {   // Step 11：小训练集上的 无正则 / 权重衰减 / dropout
                const key = [S.nsub, S.wd, S.dropout].join();
                if (S._regEp !== S.epoch || S._regKey !== key) {
                    S._regEp = S.epoch; S._regKey = key;
                    S.regRuns = [{}, { wd: S.wd }, { dropout: S.dropout }].map((ex) => P02.makeRun(P02.regCfg(S, ex)));
                    S.regRuns.forEach((R) => P02.evalNow(R));
                }
                S.regRuns.forEach((R) => P02.tick(R, P02.SPF2, 60));
            },
        },
        { id: 'trace', lifetime: 'frame', run(S) { S.tr = P02.trace(S); } },
        {
            id: 'opt2d', lifetime: 'frame', run(S) {
                const st = S.o2start, key = st.join();
                if (S._o2Key === key) return;
                S._o2Key = key;
                S.o2 = ['sgd', 'momentum', 'adam'].map((k) => P02.runOpt2d(k, st[0], st[1], 150));
            },
        },
        {
            id: 'flow', lifetime: 'frame', lazy: true, run(S) {
                const key = [S.depth, S.act, S.seed].join();
                if (S._flowKey === key) return;
                S._flowKey = key;
                const B = 64, X = S.data.tr.X.subarray(0, B * P02.D), y = S.data.tr.y.subarray(0, B);
                S.flow = P02.INITS.map((init) => P02.flowStats(S.depth, S.act, init, X, y, B, S.seed + 44));
            },
        },
    ],

    intro: {
        title: '让网络自己学会「看」：从一层线性打分到多层网络与反向传播',
        flow: ['16×16 图 → 256 维向量', 'Softmax 回归', '交叉熵', '隐藏层 + 非线性', '前向传播', '反向传播', '训练循环', '优化器 / 初始化 / 正则化'],
        html: LM.navHtml('02') + `
            <p>第 01 篇的四件套——<strong>模型、损失、梯度、优化器</strong>——这一篇原样保留，只是全部放大：输入从 1 个数变成一张 16×16 的灰度图（256 个数），输出从 1 个数变成 10 个类别的概率，模型从一条直线变成多层神经网络（MLP，Multi-Layer Perceptron），梯度不再手推公式，而是用<strong>反向传播</strong>（Backpropagation）一层一层算出来。这套机制原封不动地用在后面所有的大模型里：第 07 篇 Transformer 的每一层都有一个和本篇几乎一样的 MLP。</p>
            <p><strong>数据是 MNIST 的缩小替身。</strong>真实的 MNIST 是 28×28 的手写数字，6 万张训练 + 1 万张测试；这里在页面里程序化生成 4000 张训练 + 500 张测试的 16×16 数字（笔画字形 + 随机仿射 / 粗细 / 抖动 / 噪点），点顶栏「重新生成数据」换一批。</p>
            <p>页面里<strong>实时训练</strong>两个模型：Step 04 的 Softmax 回归和 Step 08 的 MLP（每帧 4 步 mini-batch SGD，训练到 3000 步停止；改训练参数或点「重新训练」从头开始）。Step 02 / 03 和 06 / 07 用这两个模型的<strong>当前权重</strong>分析选中的那张图，所以训练进行中它们也在变（<span style="color:#a45fa5">LIVE</span>）。代码块是同一算法的 C++ 版本（C++17 + 标准库；张量和随机数用本系列自带的小库 <code>pages/cpp/minml_*.h</code>，在节点图的 Common 面板里），所有反向传播都是手写的——只有 Step 07 的对照组和 Step 12 的生产版用了库里的自动微分。</p>`,
    },

    steps: [
        // ───────────────────────────── 01 数据 ─────────────────────────────
        {
            id: 'data', title: '数据集与向量化：一张图 = 256 个数', tag: 'DATA',
            desc: '每张图是 16×16 = 256 个像素，值在 0（黑）到 1（白）之间。<strong>向量化</strong>就是把它按行展开成一个 256 维向量 <code>x</code>：模型看到的只是这 256 个数，它并不知道哪两个像素相邻（让模型利用「相邻」这个先验，正是后来卷积网络 CNN 的出发点）。一个 batch 的 B 张图上下堆成矩阵 <code>X [B × 256]</code>，每行一个样本；标签 <code>y ∈ {0, …, 9}</code>。数据是程序生成的：每个数字由几条笔画（圆弧、直线、二次曲线）组成，每个样本都随机抖动笔画参数、随机换写法（1 有没有勾、4 开口还是闭口、7 有没有横杠…），再施加随机旋转 / 缩放 / 错切 / 平移、笔画粗细、低频的「手抖」位移和稀疏噪点。',
            formula: [
                'x = vec(I) ∈ R^{256}      x_{16r + c} = I[r, c]   // 按行展开（row-major），与 C++ 里 Tensor 的行主序存储相同',
                'X = [x_1; x_2; …; x_B] ∈ R^{B×256}      y_i ∈ {0, 1, …, 9}   // 4000 张训练（每类 400）+ 500 张测试（每类 50）',
            ],
            code: {
                title: 'C++ · makeData（笔画字形 → 随机变形 → 距离场画笔）', src: `
                // 字形坐标 (u, v) ∈ [0,1]²、v 向下；一条笔画 = 一条折线
                struct Pt { double x, y; };
                using Polyline = std::vector<Pt>;

                // 椭圆弧：圆心 (cx, cy)、半径 (rx, ry)，角度从 a0 到 a1（度），每 15° 一段
                Polyline arc(double cx, double cy, double rx, double ry, double a0, double a1) {
                    int n = std::max(4, (int)std::ceil(std::fabs(a1 - a0) / 15));
                    Polyline pts;
                    for (int i = 0; i <= n; i++) {
                        double a = (a0 + (a1 - a0) * i / n) * 3.141592653589793 / 180;
                        pts.push_back({cx + rx * std::cos(a), cy + ry * std::sin(a)});
                    }
                    return pts;
                }

                // 直线段
                Polyline line(double x0, double y0, double x1, double y1) { return {{x0, y0}, {x1, y1}}; }

                // 二次 Bézier 曲线（取 9 个点）
                Polyline quad(double x0, double y0, double x1, double y1, double x2, double y2) {
                    Polyline pts;
                    for (int i = 0; i <= 8; i++) {
                        double t = i / 8.0, s = 1 - t;
                        pts.push_back({s * s * x0 + 2 * s * t * x1 + t * t * x2, s * s * y0 + 2 * s * t * y1 + t * t * y2});
                    }
                    return pts;
                }

                // x ± a 范围内的均匀抖动
                double jit(Rng& r, double x, double a) { return x + (2 * uniform(r) - 1) * a; }

                // 以概率 p 返回 true：在同一个数字的几种写法之间选择
                bool coin(Rng& r, double p) { return uniform(r) < p; }

                // 数字 d 的笔画：每个样本的圆心、半径、起止角、端点都抖一下。
                // 随机量都先存进局部变量再用：C++ 里函数实参的求值顺序是不确定的，这样随机数的消耗顺序才固定（与页面 JS 相同）
                std::vector<Polyline> strokes(int d, Rng& r) {
                    switch (d) {
                        case 0: {
                            double a0 = jit(r, -90, 40), cx = jit(r, .5, .03), cy = jit(r, .5, .02);
                            double rx = jit(r, .22, .07), ry = jit(r, .36, .03), a1 = a0 + jit(r, 360, 20);
                            return {arc(cx, cy, rx, ry, a0, a1)};
                        }
                        case 1: {
                            double tx = jit(r, .53, .08), ty = jit(r, .12, .03), bx = jit(r, .49, .08), by = jit(r, .88, .03);
                            std::vector<Polyline> s = {line(tx, ty, bx, by)};
                            if (coin(r, .6)) {                                                           // 起笔的小勾
                                double x = jit(r, .34, .06), y = jit(r, .3, .06);
                                s.push_back(line(x, y, tx, ty));
                            }
                            if (coin(r, .25)) {                                                          // 底座
                                double x0 = jit(r, .34, .04), x1 = jit(r, .68, .04);
                                s.push_back(line(x0, .88, x1, .88));
                            }
                            return s;
                        }
                        case 2: {
                            double cx = jit(r, .5, .03), cy = jit(r, .34, .03), rx = jit(r, .24, .04), ry = jit(r, .21, .04);
                            double a0 = jit(r, 195, 15), a1 = jit(r, 380, 25);
                            Polyline a = arc(cx, cy, rx, ry, a0, a1);
                            Pt e = a.back();
                            double bx = jit(r, .24, .05), by = jit(r, .86, .03), ex = jit(r, .8, .06), ey = jit(r, .86, .04);
                            std::vector<Polyline> s = {a, line(e.x, e.y, bx, by), line(bx, by, ex, ey)};
                            if (coin(r, .3)) s.push_back(arc(bx + .06, by - .06, .07, .06, 90, 450));   // 底部带小圈
                            return s;
                        }
                        case 3: {
                            double cx = jit(r, .49, .03), cy = jit(r, .67, .03), rx = jit(r, .25, .04), ry = jit(r, .19, .03);
                            double a0 = jit(r, 270, 10), a1 = jit(r, 515, 20);
                            Polyline low = arc(cx, cy, rx, ry, a0, a1);                                  // 下半部分
                            if (coin(r, .25)) {                                                          // 平顶的 3
                                double x0 = jit(r, .28, .04), y0 = jit(r, .14, .03), x1 = jit(r, .72, .04);
                                return {line(x0, y0, x1, .14), line(.72, .14, low[0].x, low[0].y), low};
                            }
                            double ux = jit(r, .49, .03), uy = jit(r, .31, .03), urx = jit(r, .22, .04), ury = jit(r, .18, .03);
                            double u0 = jit(r, 205, 20), u1 = jit(r, 450, 10);
                            return {arc(ux, uy, urx, ury, u0, u1), low};
                        }
                        case 4: {
                            if (coin(r, .65)) {                                                          // 闭口的 4
                                double x = jit(r, .64, .05), lx = jit(r, .2, .05), ly = jit(r, .62, .05);
                                double bx = jit(r, x, .04), rx = jit(r, .84, .05);
                                return {line(x, .12, bx, .88), line(x, .12, lx, ly), line(lx, ly, rx, ly)};
                            }
                            double ly = jit(r, .58, .05), tx = jit(r, .36, .05), rx = jit(r, .82, .05);  // 开口的 4
                            double vx0 = jit(r, .66, .05), vy0 = jit(r, .3, .06), vx1 = jit(r, .64, .05);
                            return {line(tx, .12, .26, ly), line(.26, ly, rx, ly), line(vx0, vy0, vx1, .88)};
                        }
                        case 5: {
                            double tx = jit(r, .34, .04), ty = jit(r, .14, .03), mx = jit(r, .3, .04), my = jit(r, .5, .04);
                            double hx = jit(r, .74, .05), hy = jit(r, ty, .03);                          // 顶上的横
                            double cx = jit(r, .5, .03), cy = jit(r, .64, .03), rx = jit(r, .25, .04), ry = jit(r, .21, .03);
                            double a0 = jit(r, 215, 15), a1 = jit(r, 485, 25);
                            return {line(hx, hy, tx, ty), line(tx, ty, mx, my), arc(cx, cy, rx, ry, a0, a1)};
                        }
                        case 6: {
                            double ex = jit(r, .27, .03), ey = jit(r, .64, .04);
                            double x0 = jit(r, .7, .06), y0 = jit(r, .13, .03), x1 = jit(r, .3, .06), y1 = jit(r, .22, .05);
                            double cx = jit(r, .5, .03), cy = jit(r, .66, .03), rx = jit(r, .23, .04), ry = jit(r, .2, .04);
                            return {quad(x0, y0, x1, y1, ex, ey), arc(cx, cy, rx, ry, 0, 360)};
                        }
                        case 7: {
                            double rx = jit(r, .78, .05), ry = jit(r, .14, .03), bx = jit(r, .42, .1);
                            double lx = jit(r, .22, .05), ly = jit(r, .14, .03);
                            std::vector<Polyline> s = {line(lx, ly, rx, ry)};
                            if (coin(r, .7)) s.push_back(line(rx, ry, bx, .88));                         // 直的竖笔
                            else { double qx = jit(r, .5, .06); s.push_back(quad(rx, ry, qx, .45, bx, .88)); }   // 弯的竖笔
                            if (coin(r, .2)) {                                                           // 带横杠的 7
                                double x0 = jit(r, .4, .04), x1 = jit(r, .72, .04);
                                s.push_back(line(x0, .5, x1, .5));
                            }
                            return s;
                        }
                        case 8: {
                            double ux = jit(r, .5, .03), uy = jit(r, .3, .02), urx = jit(r, .19, .04), ury = jit(r, .17, .03);   // 上圈
                            double lx = jit(r, .5, .03), ly = jit(r, .68, .02), lrx = jit(r, .23, .04), lry = jit(r, .19, .03);  // 下圈
                            return {arc(ux, uy, urx, ury, 0, 360), arc(lx, ly, lrx, lry, 0, 360)};
                        }
                        default: {                                                                       // 9
                            double cx = jit(r, .5, .03), cy = jit(r, .33, .03), rx = jit(r, .22, .04), ry = jit(r, .2, .03), sx = cx + rx;
                            std::vector<Polyline> s = {arc(cx, cy, rx, ry, 0, 360)};
                            if (coin(r, .6)) { double bx = jit(r, .64, .08); s.push_back(line(sx, cy, bx, .88)); }
                            else { double qx = jit(r, .74, .05), ex = jit(r, .42, .06); s.push_back(quad(sx, cy, qx, .75, ex, .86)); }
                            return s;
                        }
                    }
                }

                // 一个笔画段（像素坐标）：从 (ax, ay) 到 (bx, by)
                struct Seg { double ax, ay, bx, by; };

                // 渲染一个数字 → 256 个灰度（0..1，按行展开）：
                // 随机仿射 + 低频「手抖」位移 → 每个像素到最近笔画的距离 → 抗锯齿灰度
                std::vector<float> renderDigit(int d, Rng& r) {
                    const int IMG = 16;
                    std::vector<Polyline> pls = strokes(d, r);
                    double rot = (2 * uniform(r) - 1) * .25, sc = .82 + .3 * uniform(r);              // 旋转 ±14°、缩放
                    double asp = .85 + .3 * uniform(r), sh = (2 * uniform(r) - 1) * .35;              // 长宽比、错切
                    double tx = (2 * uniform(r) - 1) * .08, ty = (2 * uniform(r) - 1) * .07;          // 平移约 ±1.3 像素
                    double wa = .015 + .035 * uniform(r), f1 = .6 + uniform(r), f2 = .6 + uniform(r); // 低频「手抖」位移场：振幅、频率
                    double p1 = 6.2832 * uniform(r), p2 = 6.2832 * uniform(r);                        //                        相位
                    double c = std::cos(rot), s = std::sin(rot);
                    std::vector<Seg> segs;
                    for (const Polyline& pl : pls) {
                        double ox = .02 * randn(r), oy = .02 * randn(r);                     // 每一笔各自的小偏移（笔画接不上 / 出头）
                        double px = 0, py = 0;
                        for (size_t i = 0; i < pl.size(); i++) {
                            double u = pl[i].x, v = pl[i].y;
                            double x = u + wa * std::sin(6.2832 * f1 * v + p1) + ox - .5, y = v + wa * std::sin(6.2832 * f2 * u + p2) + oy - .5;
                            x += sh * y;                                                      // 错切
                            x *= sc * asp;                                                    // 缩放、长宽比
                            y *= sc / asp;
                            double X = (.5 + c * x - s * y + tx) * IMG, Y = (.5 + s * x + c * y + ty) * IMG;   // 旋转、平移 → 像素坐标
                            if (i > 0) segs.push_back({px, py, X, Y});
                            px = X; py = Y;
                        }
                    }
                    double rad = .55 + .9 * uniform(r), peak = .75 + .25 * uniform(r);       // 笔画半宽（像素）、墨色深浅
                    std::vector<float> img = floats(IMG * IMG);
                    for (int y = 0; y < IMG; y++)
                        for (int x = 0; x < IMG; x++) {
                            double qx = x + .5, qy = y + .5, best = 1e9;                     // 像素中心到最近笔画段的距离²
                            for (const Seg& g : segs) {
                                double bx = g.bx - g.ax, by = g.by - g.ay, l2 = bx * bx + by * by;
                                double t = l2 > 0 ? ((qx - g.ax) * bx + (qy - g.ay) * by) / l2 : 0;
                                t = std::min(1.0, std::max(0.0, t));
                                double dx = qx - g.ax - t * bx, dy = qy - g.ay - t * by;
                                best = std::min(best, dx * dx + dy * dy);
                            }
                            double v = peak * std::min(1.0, std::max(0.0, rad + .5 - std::sqrt(best)));   // 1 像素宽的软边
                            if (v > 0) v *= 1 + .12 * randn(r);                               // 笔触纹理
                            if (uniform(r) < .012) v = std::max(v, .15 + .4 * uniform(r));    // 稀疏的背景噪点
                            img[y * IMG + x] = (float)std::min(1.0, std::max(0.0, v));
                        }
                    return img;
                }

                // 数据集：X [n × 256]（每行一张图），y [n]
                struct Dataset { Tensor X; std::vector<int> y; int n = 0; };

                // n 张图，每类一样多、顺序打乱
                Dataset makeSet(int n, Rng& r) {
                    Dataset ds;
                    ds.n = n;
                    ds.y = ints(n);
                    for (int i = 0; i < n; i++) ds.y[i] = i % 10;
                    shuffleInPlace(ds.y, r);
                    ds.X = zeros(n, 256);
                    for (int i = 0; i < n; i++) {
                        std::vector<float> img = renderDigit(ds.y[i], r);
                        std::copy(img.begin(), img.end(), ds.X.v.begin() + (size_t)i * 256);
                    }
                    return ds;
                }

                // 页面里的数据：同一串随机数先生成 4000 张训练图，再生成 500 张测试图。
                // makeData(20251123) 就是页面默认的那一批（「重新生成数据」= 换一个 seed）
                struct Data { Dataset tr, te; };

                Data makeData(uint32_t seed) {
                    Rng r = makeRng(seed);
                    Data d;
                    d.tr = makeSet(4000, r);
                    d.te = makeSet(500, r);
                    return d;
                }`,
            },
            previews: [
                {
                    id: 'grid', label: '测试集样本（每列一个数字）', sub: '每类前 8 张 · 黄框 = 当前选中 · <b>点击选择样本</b>', large: true, size: [360, 320],
                    render: (c, S) => {
                        const d = S.data, items = [], idx = [];
                        for (let r = 0; r < 8; r++) for (let k = 0; k < 10; k++) { const i = d.byClass[k][r]; idx.push(i); items.push({ img: d.te.X.subarray(i * 256, (i + 1) * 256), outline: i === S.tr.i ? LM.C.yellow : null }); }
                        S._g01 = P02.digitGrid(c, items, { cols: 10, header: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] }); S._g01idx = idx;
                    },
                    probe: (x, y, S, PL, info) => { const k = P02.gridHit(S._g01, info); if (k < 0) return ''; const i = S._g01idx[k]; return `测试样本 #${i} · 标签 ${S.data.te.y[i]}（点击选中）`; },
                    pick: (x, y, S, PL, info) => { const k = P02.gridHit(S._g01, info); if (k >= 0) PL.setParam('sample', S._g01idx[k]); },
                },
                {
                    id: 'pixels', label: '选中样本的像素值', sub: '16 × 16 灰度，0 = 黑、1 = 白（只标出 ≥ 0.05 的像素）· 黄框 = 第 8 行', size: [360, 376],
                    render: (c, S) => { S._pp = P02.pixelPanel(c, S.tr.x, { title: `样本 #${S.tr.i} · 标签 y = ${S.tr.y}`, hiRow: 8 }); },
                    probe: (x, y, S, PL, info) => { const g = S._pp; if (!g) return ''; const q = Math.floor((info.dx - g.x0) / g.cs), r = Math.floor((info.dy - g.y0) / g.cs); if (q < 0 || q > 15 || r < 0 || r > 15) return ''; return `I[${r}, ${q}] = ${S.tr.x[r * 16 + q].toFixed(3)}  →  x[${16 * r + q}]`; },
                },
                {
                    id: 'vector', label: '展平成 256 维向量 x', sub: '柱 = 像素值，按下标排列；蓝 / 青交替 = 原图的一行 · 黄线之间 = 第 8 行（x[128] … x[143]）', size: [360, 170],
                    render: (c, S) => {
                        const x = S.tr.x, a = [], b = [], xs = [];
                        for (let j = 0; j < 256; j++) { xs.push(j); const odd = (j >> 4) & 1; a.push(odd ? NaN : x[j]); b.push(odd ? x[j] : NaN); }
                        PL.draw.plot(c, { x: [-1, 256], y: [0, 1.05], xLabel: '下标 j = 16 × 行 + 列', series: [{ data: a, x: xs, bar: true, color: LM.C.blue }, { data: b, x: xs, bar: true, color: LM.C.cyan }], marks: [{ x: 127.5, color: LM.C.yellow }, { x: 143.5, label: '第 8 行', color: LM.C.yellow }] });
                    },
                },
                {
                    id: 'means', label: '每类的平均图像（训练集 4000 张）', sub: '同一个数字的写法差异越大，平均图越模糊', size: [360, 56],
                    render: (c, S) => P02.digitGrid(c, Array.from({ length: 10 }, (_, k) => ({ img: S.data.mean, offset: k * 256, range: [0, 0.75], label: k })), { cols: 10, labels: true }),
                },
            ],
            notes: '在网格里点任意一张图选中它（顶栏 <b>样本 #</b> 同步变化），Step 02 / 03 / 06 / 07 都会跟着分析这张图。像素面板里黄框的第 8 行，就是向量里第 128–143 个分量。每类的平均图很模糊：同一个数字的写法差别很大——这正是下一步「每个类别一张模板」的线性模型的天花板。',
            warn: '真实 MNIST 的灰度是 0–255（通常除以 255，PyTorch 的官方示例再按均值 0.1307、标准差 0.3081 标准化）；数字先按比例缩放进 20×20 的框，再按质心居中放进 28×28。这里的 16×16 程序化数字只是缩小替身：字形来自固定的笔画模板，准确率数字只能看趋势，不能和 MNIST 论文直接比较。',
        },
        // ───────────────────────────── 02 Softmax 回归 ─────────────────────────────
        {
            id: 'softmax', title: 'Softmax 回归：线性打分 → 概率', tag: 'MODEL',
            desc: '最简单的分类器：给每个类别 k 一个 256 维的权重向量 <code>w_k</code>（W 的第 k 列）和偏置 <code>b_k</code>，打分 <code>z_k = x·w_k + b_k</code>，叫做 <strong>logit</strong>——分数越高越像这一类。<strong>softmax</strong> 把 10 个分数变成 10 个和为 1 的正数：指数让大的更大，除以总和保证加起来等于 1。整个模型就是一次矩阵乘法 <code>Z = XW + b</code>（W 是 256 × 10，加上偏置共 2570 个参数），所以叫<strong>线性分类器</strong>：每个类别对 256 个像素做一次加权投票（第三张图：每个像素投的「票」= 像素值 × 权重）。下面是 Step 04 <strong>正在训练</strong>的模型对选中样本的真实输出。',
            formula: [
                'z = x · W + b      W ∈ R^{256×10}, b ∈ R^{10}   // zₖ = Σⱼ xⱼ·Wⱼₖ + bₖ：第 k 类的打分（logit）',
                'p_k = softmax(z)_k = exp(z_k − m) / Σ_j exp(z_j − m)      m = max_{j} z_j   // 减去最大值：结果不变、exp 不溢出',
                'ŷ = argmax_{k} p_k   // 预测 = 概率最大的类别',
            ],
            code: {
                title: 'C++ · gemm / softmax / predictProba', src: `
                // Z = A·W + b    A [n × k]，W [k × m]，b [1 × m]（加到每一行上）→ Z [n × m]。跳过 A 里的 0：图像大部分像素是 0
                Tensor gemm(const Tensor& A, const Tensor& W, const Tensor& b) {
                    int n = A.rows, k = A.cols, m = W.cols;
                    Tensor Z = zeros(n, m);
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < m; j++) Z.at(i, j) = b.v[j];
                        for (int p = 0; p < k; p++) {
                            float a = A.at(i, p);
                            if (a == 0) continue;
                            for (int j = 0; j < m; j++) Z.at(i, j) += a * W.at(p, j);
                        }
                    }
                    return Z;
                }

                // 逐行 softmax：先减去这一行的最大值（结果不变，exp 不会溢出），再除以总和
                Tensor softmax(Tensor Z) {
                    for (int i = 0; i < Z.rows; i++) {
                        float m = Z.at(i, 0);
                        for (int j = 1; j < Z.cols; j++) m = std::max(m, Z.at(i, j));
                        double s = 0;
                        for (int j = 0; j < Z.cols; j++) { Z.at(i, j) = std::exp(Z.at(i, j) - m); s += Z.at(i, j); }
                        for (int j = 0; j < Z.cols; j++) Z.at(i, j) = (float)(Z.at(i, j) / s);
                    }
                    return Z;
                }

                // X [B × 256]，W [256 × 10]，b [1 × 10] → P [B × 10]，每行和为 1
                Tensor predictProba(const Tensor& X, const Tensor& W, const Tensor& b) { return softmax(gemm(X, W, b)); }`,
            },
            previews: [
                {
                    id: 'probs', label: 'softmax 概率 p', sub: '绿 = 正确类别 · 红 = 判错时的预测类别 · 来自 Step 04 正在训练的模型', live: true, large: true, size: [360, 230],
                    render: (c, S) => {
                        const t = S.tr;
                        P02.barChart(c, Array.from(t.ps), [...'0123456789'], { range: [0, 1.15], ticks: [0, 0.5, 1], values: true, valueMin: 0.005, tx: 40, colorFor: (v, i) => (i === t.y ? LM.C.green : i === t.predS ? LM.C.red : LM.C.blue), labelColor: (i) => (i === t.y ? LM.C.green : LM.C.gray), title: `样本 #${t.i}（标签 ${t.y}）→ 预测 ${t.predS} · 第 ${S.soft.step} 步的权重`, titleColor: t.predS === t.y ? LM.C.green : LM.C.red });
                    },
                },
                {
                    id: 'logits', label: 'logits z = xW + b', sub: 'softmax 之前的打分，可正可负；概率只取决于它们之间的差', live: true, size: [360, 180],
                    render: (c, S) => { const t = S.tr; P02.barChart(c, Array.from(t.zs), [...'0123456789'], { symmetric: true, values: true, digits: 1, colorFor: (v, i) => (i === t.y ? LM.C.green : v >= 0 ? LM.C.orange : LM.C.blue) }); },
                },
                {
                    id: 'contrib', label: '每个像素投的票 x ⊙ w_k', sub: '左 = 正确类别，右 = 打分最高的其他类别 · 橙 = 赞成、蓝 = 反对 · 全部像素之和 + b_k = z_k', live: true, size: [360, 176],
                    render: (c, S) => {
                        const t = S.tr, W = S.soft.net.W[0], b = S.soft.net.b[0];
                        let k2 = -1; for (let k = 0; k < 10; k++) if (k !== t.y && (k2 < 0 || t.zs[k] > t.zs[k2])) k2 = k;
                        const ks = [t.y, k2], imgs = ks.map((k) => { const a = new Float32Array(256); for (let p = 0; p < 256; p++) a[p] = t.x[p] * W[p * 10 + k]; return a; });
                        const m = Math.max(P02.absMax(imgs[0], 1, 0, 256), P02.absMax(imgs[1], 1, 0, 256)), { ctx } = PL.draw.ctx(c), sz = 130;
                        ks.forEach((k, q) => {
                            const x0 = 28 + q * (sz + 44), y0 = 38;
                            P02.drawImg(ctx, imgs[q], x0, y0, sz, sz, { map: 'diverge', range: [-m, m] });
                            P02.txt(ctx, `${q ? '最强对手' : '正确类别'} k = ${k}`, x0 + sz / 2, 4, { align: 'center', bold: true, color: q ? LM.C.red : LM.C.green });
                            P02.txt(ctx, `z = ${t.zs[k].toFixed(2)}（b = ${b[k].toFixed(2)}）`, x0 + sz / 2, 20, { align: 'center', color: LM.C.gray });
                        });
                    },
                },
            ],
            notes: '点 Step 01 网格里不同的样本：多数样本的概率集中在正确类别（绿柱）上；写得歪的样本会把概率分给长得像的类别（4 ↔ 9、3 ↔ 5、7 ↔ 1…），第三张图能看出是哪些像素把票投给了对手。训练第 0 步 W = 0，所有类别的概率都是 0.1。',
            warn: '本页一律用<strong>行向量</strong>约定 <code>z = xW + b</code>，W 的形状是 [输入 × 输出]，和代码里的 <code>gemm(X, W, b)</code> 一致；PyTorch 的 <code>nn.Linear</code> 把权重存成 [输出 × 输入]、计算 <code>x Wᵀ + b</code>——同一件事，差一个转置。',
        },
        // ───────────────────────────── 03 交叉熵 ─────────────────────────────
        {
            id: 'xent', title: '交叉熵损失：−log p_y，以及为什么不用 MSE', tag: 'LOSS',
            desc: '训练需要一个衡量「错得多离谱」的数。分类的标准选择是<strong>交叉熵</strong>（Cross-Entropy）：只看正确类别的概率 p_y，取 −log。p_y = 1 时损失为 0；p_y → 0 时损失 → ∞，「自信地错」罚得最重。它就是负对数似然：最小化它 = 最大似然估计。更重要的是它的梯度：交叉熵 + softmax 对 logit 的梯度简洁得惊人，<code>∂ℓ/∂z = p − onehot(y)</code>——错多少就推多少，不会「饱和」。如果改用 MSE（把 p 和 one-hot 做均方误差），链式法则还要再穿过 softmax 的导数，多出一个 p_y(1 − p_y) 量级的因子：模型自信地错（p_y ≈ 0）时这个因子 ≈ 0，梯度消失，几乎学不动——第一张图就是这个对比。',
            formula: [
                'ℓ(x, y) = −log p_y = −z_y + log Σ_j exp(z_j)      L = (1/B) Σ_i ℓ(x_i, y_i)',
                '∂ℓ/∂z_{k} = p_k − [k = y]      ⇔      ∂ℓ/∂z = p − onehot(y)   // 没有 p(1 − p) 因子',
                'MSE：ℓ = Σ_k (p_k − [k = y])^2   ⇒   ∂ℓ/∂z_{y} = −(20/9) · p_y (1 − p_y)^2   // 其余 9 个 logit 相等时；正确类别的概率 → 0 时梯度也 → 0',
            ],
            code: {
                title: 'C++ · oneHot / crossEntropyLoss / crossEntropyGrad / mseGrad', src: `
                // 标签 → one-hot 矩阵 Y [n × k]：第 i 行只有第 y_i 列是 1
                Tensor oneHot(const std::vector<int>& y, int k = 10) {
                    Tensor Y = zeros((int)y.size(), k);
                    for (int i = 0; i < (int)y.size(); i++) Y.at(i, y[i]) = 1;
                    return Y;
                }

                // 平均交叉熵 L = (1/B)·Σ_i (log Σ_j exp z_ij − z_{i,y_i})：直接从 logits 算（log-sum-exp），不会出现 log(0)
                float crossEntropyLoss(const Tensor& Z, const std::vector<int>& y) {
                    double total = 0;
                    for (int i = 0; i < Z.rows; i++) {
                        float m = Z.at(i, 0);
                        for (int j = 1; j < Z.cols; j++) m = std::max(m, Z.at(i, j));
                        double s = 0;
                        for (int j = 0; j < Z.cols; j++) s += std::exp(Z.at(i, j) - m);
                        total += m + std::log(s) - Z.at(i, y[i]);
                    }
                    return (float)(total / Z.rows);
                }

                // ∂L/∂Z = (P − Y)/B：错多少就推多少
                Tensor crossEntropyGrad(const Tensor& Z, const std::vector<int>& y) {
                    Tensor G = softmax(Z), Y = oneHot(y, Z.cols);
                    for (int i = 0; i < G.size(); i++) G.v[i] = (G.v[i] - Y.v[i]) / Z.rows;
                    return G;
                }

                // 对照组：L = (1/B)·Σ_i Σ_k (p_ik − y_ik)²；链式法则要再穿过 softmax 的雅可比 diag(p) − ppᵀ
                Tensor mseGrad(const Tensor& Z, const std::vector<int>& y) {
                    Tensor P = softmax(Z), Y = oneHot(y, Z.cols), G = zeros(Z.rows, Z.cols);
                    for (int i = 0; i < Z.rows; i++) {
                        double gp = 0;                                                // Σ_k g_k·p_k
                        for (int k = 0; k < Z.cols; k++) gp += 2 * (P.at(i, k) - Y.at(i, k)) / Z.rows * P.at(i, k);
                        for (int k = 0; k < Z.cols; k++) {
                            float g = 2 * (P.at(i, k) - Y.at(i, k)) / Z.rows;           // ∂L/∂p_k
                            G.at(i, k) = P.at(i, k) * (g - (float)gp);                  // 逐行 (diag(p) − ppᵀ)·g
                        }
                    }
                    return G;
                }`,
            },
            previews: [
                {
                    id: 'satur', label: '对正确类别 logit 的梯度大小 |∂ℓ/∂z_y|', sub: '设其余 9 个 logit = 0，横轴 = z_y · 黄线 = 选中样本（Softmax 回归）对应的位置', live: true, large: true, size: [360, 240],
                    render: (c, S) => {
                        const xs = [], ce = [], ms = [];
                        for (let i = 0; i <= 240; i++) { const z = -10 + 18 * i / 240, p = Math.exp(z) / (Math.exp(z) + 9); xs.push(z); ce.push(1 - p); ms.push(2 * p * (1 - p) * (1 - p) * 10 / 9); }
                        const t = S.tr, py = Math.min(1 - 1e-9, Math.max(1e-9, t.ps[t.y])), zEq = Math.log(9 * py / (1 - py));
                        P02.curves(c, [{ x: xs, y: ce, color: LM.C.blue, width: 2, label: '交叉熵' }, { x: xs, y: ms, color: LM.C.orange, width: 2, label: 'MSE' }], { x: [-10, 8], y: [1e-5, 3], logY: true, xLabel: 'z_y（其余 logit = 0）', legendAt: [0, 0.14], legendAlign: 'left', marks: [{ x: zEq, label: `#${t.i}：p_y = ${py.toFixed(3)}`, color: LM.C.yellow }] });
                    },
                },
                {
                    id: 'lossvp', label: '损失 vs 正确类别的概率 p_y', sub: '交叉熵在 p_y → 0 时 → ∞（自信地错 = 重罚）；MSE 最多只有 10/9', live: true, size: [360, 200],
                    render: (c, S) => {
                        const xs = [], ce = [], ms = [];
                        for (let i = 0; i <= 200; i++) { const p = Math.pow(10, -3 + 3 * i / 200); xs.push(p); ce.push(-Math.log(p)); ms.push((1 - p) * (1 - p) * 10 / 9); }
                        const t = S.tr, py = Math.max(1e-3, t.ps[t.y]);
                        P02.curves(c, [{ x: xs, y: ce, color: LM.C.blue, width: 2, label: '−log p_y' }, { x: xs, y: ms, color: LM.C.orange, width: 2, label: 'Σ(p − y)²' }], { x: [1e-3, 1], y: [0, 7], logX: true, xLabel: 'p_y（对数轴）', marks: [{ x: py, label: `ℓ = ${t.lossS.toFixed(2)}`, color: LM.C.yellow }] });
                    },
                },
                {
                    id: 'sampleGrad', label: '选中样本的梯度 ∂ℓ/∂z = p − y', sub: '正确类别为负（要把它的分数推高），其余为正（压低），总和为 0', live: true, size: [360, 170],
                    render: (c, S) => { const t = S.tr, g = Array.from(t.ps, (p, k) => p - (k === t.y ? 1 : 0)); P02.barChart(c, g, [...'0123456789'], { range: [-1, 1], ticks: [-1, 0, 1], values: true, valueMin: 0.005, colorFor: (v) => (v < 0 ? LM.C.green : LM.C.red), labelColor: (i) => (i === t.y ? LM.C.green : LM.C.gray) }); },
                },
                {
                    id: 'lossHist', label: '测试集 500 张的 −log p_y 分布', sub: 'Softmax 回归最近一次评估 · 每格宽 0.2，最右一格 = ≥ 4.8 · 红 = p_y < 0.5（多半判错）', live: true, size: [360, 170],
                    render: (c, S) => {
                        const R = S.soft, lv = R.lossv && R.lossv[1], nb = 25, cnt = new Array(nb).fill(0);
                        let mean = 0;
                        if (lv) { for (const v of lv) { cnt[Math.min(nb - 1, Math.floor(v / 0.2))]++; mean += v / lv.length; } }
                        P02.barChart(c, cnt, cnt.map((_, k) => (k % 5 === 0 ? String(k / 5) : '')), { colorFor: (v, k) => ((k + 1) * 0.2 > Math.LN2 ? LM.C.red : LM.C.blue), title: `第 ${R.evalStep ?? 0} 步 · 平均 ${mean.toFixed(3)}（= 测试 loss）`, titleColor: LM.C.gray, tx: 40 });
                    },
                },
            ],
            notes: '用 Step 01 选一个判错的样本（或 Step 08 错例里的）：p_y 很小、−log p_y 很大，梯度 p − y 在正确类别上是一根很长的负柱——它会把 z_y 往上推，把抢走概率的类别往下压。第一张图的黄线是选中样本在这条曲线上对应的位置：交叉熵的梯度在 p_y 很小时接近 1（满力推），MSE 的梯度却几乎为 0。直方图里大多数样本的损失接近 0，平均值被少数「自信地错」的样本拉高。',
            warn: '实现时不要先算 softmax 再取 log：p 下溢成 0 就得到 −∞。用 log-sum-exp 的形式直接从 logits 算（代码里的 <code>crossEntropyLoss</code>）；本系列库里的 <code>crossEntropy</code>、PyTorch 的 <code>F.cross_entropy</code> 也都是吃 logits、内部做 log-softmax。',
        },
        // ───────────────────────────── 04 训练 Softmax 回归 ─────────────────────────────
        {
            id: 'trainSoft', title: '训练 Softmax 回归：W 的 10 列长成 10 张模板', tag: 'TRAIN',
            desc: '把 Step 03 的梯度代入第 01 篇的 mini-batch SGD：每步随机取 32 张图，算平均交叉熵对 W、b 的梯度 <code>dW = Xᵀ(P − Y)/B</code>，再 <code>W ← W − η·dW</code>。第一张图就是 W 的 10 列，每列 256 个数排回 16×16：<span style="color:#ffb05a">橙色</span>像素投赞成票，<span style="color:#5ab0ff">蓝色</span>投反对票。训练开始时它们全是 0，几百步后就出现数字的轮廓——这就是线性模型学到的「模板」（0 的模板中间是蓝的：中间有墨水就不像 0）。线性模型的极限也一目了然：每个类别只有一张模板，同一个数字的各种写法只能平均进去，所以连<strong>训练</strong>准确率都停在 90% 出头。',
            formula: [
                'dZ = (P − Y) / B      dW = X_b^T · dZ      db = Σ_i dZ_{i}   // Y = one-hot 标签矩阵 [B × 10]',
                'W ← W − η · dW      b ← b − η · db   // B = 32；每帧 4 步，3000 步（= 24 个 epoch）后停止',
            ],
            code: {
                title: 'C++ · trainSoftmax / accuracy', src: `
                // 第 i 行最大元素的列号 = 预测的类别
                int argmaxRow(const Tensor& Z, int i) {
                    int best = 0;
                    for (int j = 1; j < Z.cols; j++) if (Z.at(i, j) > Z.at(i, best)) best = j;
                    return best;
                }

                // Softmax 回归的参数：W [256 × 10]，b [1 × 10]
                struct SoftmaxReg { Tensor W, b; };

                SoftmaxReg trainSoftmax(const Dataset& tr, float lr = 0.1f, int steps = 3000, int B = 32, uint32_t seed = 0) {
                    Rng r = makeRng(seed);
                    Tensor W = zeros(256, 10), b = zeros(1, 10);                      // 线性模型可以从 0 开始
                    BatchStream bs = makeBatchStream(range(0, tr.n), r);
                    for (int t = 0; t < steps; t++) {
                        std::vector<int> idx = nextBatch(bs, B, r);
                        Tensor Xb = takeRows(tr.X, idx);
                        Tensor dZ = crossEntropyGrad(gemm(Xb, W, b), takeLabels(tr.y, idx));   // (P − Y)/B   [B × 10]
                        Tensor dW = zeros(256, 10), db = zeros(1, 10);
                        for (int i = 0; i < B; i++) {
                            for (int k = 0; k < 10; k++) db.v[k] += dZ.at(i, k);                 // db = Σ_i dZ_i
                            for (int p = 0; p < 256; p++)
                                for (int k = 0; k < 10; k++) dW.at(p, k) += Xb.at(i, p) * dZ.at(i, k);   // dW = Xᵀ·dZ   [256 × 10]
                        }
                        for (int q = 0; q < dW.size(); q++) W.v[q] -= lr * dW.v[q];             // W ← W − η·dW
                        for (int k = 0; k < 10; k++) b.v[k] -= lr * db.v[k];                    // b ← b − η·db
                    }
                    return {W, b};
                }

                // 准确率：argmax(XW + b) 与标签相同的比例
                float accuracy(const Tensor& X, const std::vector<int>& y, const Tensor& W, const Tensor& b) {
                    Tensor Z = gemm(X, W, b);
                    int correct = 0;
                    for (int i = 0; i < X.rows; i++) correct += argmaxRow(Z, i) == y[i];
                    return (float)correct / X.rows;
                }`,
            },
            previews: [
                {
                    id: 'templates', label: 'W 的 10 列 = 10 张模板', sub: '橙 = 正权重（这里有墨水 → 更像这一类），蓝 = 负权重 · 10 张图共用一个色阶', live: true, large: true, size: [360, 210],
                    render: (c, S) => {
                        const W = S.soft.net.W[0], m = P02.absMax(W, 1, 0, W.length);
                        S._gT = P02.digitGrid(c, Array.from({ length: 10 }, (_, k) => ({ img: W, stride: 10, offset: k, label: k })), { cols: 5, map: 'diverge', range: [-m, m], labels: true, title: P02.statusShort(S.soft) + ` · max|W| = ${m.toFixed(2)}`, titleColor: LM.C.yellow, tx: 40 });
                    },
                    probe: (x, y, S, PL, info) => { const g = S._gT, k = P02.gridHit(g, info); if (k < 0) return ''; const cx = g.x0 + (k % g.cols) * (g.cell + g.gap), cy = g.y0 + Math.floor(k / g.cols) * (g.cell + g.gap + g.lab), q = Math.floor((info.dx - cx) / g.cell * 16), r = Math.floor((info.dy - cy) / g.cell * 16); if (q < 0 || q > 15 || r < 0 || r > 15) return `模板 ${k}`; return `W[${16 * r + q}, ${k}] = ${S.soft.net.W[0][(16 * r + q) * 10 + k].toFixed(3)}（像素 ${r},${q} 对类别 ${k} 的权重）`; },
                },
                {
                    id: 'softAcc', label: '准确率', sub: '蓝 = 训练集（固定抽 500 张评估）· 橙 = 测试集 500 张', live: true, size: [360, 220],
                    render: (c, S) => {
                        const R = S.soft, H = R.hist;
                        P02.curves(c, [{ x: H.step, y: H.trainA.map((v) => 100 * v), color: LM.C.train, width: 2, label: `训练 ${P02.pct(P02.last(H.trainA))}` }, { x: H.step, y: H.testA.map((v) => 100 * v), color: LM.C.val, width: 2, label: `测试 ${P02.pct(P02.last(H.testA))}` }],
                            { x: [0, P02.CAP], y: [0, 100], xLabel: '训练步数', legendAt: [1, 0.4], extra: [{ label: P02.status(R, P02.NTRAIN), color: LM.C.yellow, noLine: true }] });
                    },
                },
                {
                    id: 'softLoss', label: '交叉熵损失', sub: '对数纵轴', live: true, size: [360, 200],
                    render: (c, S) => { const H = S.soft.hist; P02.curves(c, [{ x: H.step, y: H.trainL, color: LM.C.train, width: 2, label: '训练' }, { x: H.step, y: H.testL, color: LM.C.val, width: 2, label: '测试' }], { x: [0, P02.CAP], y: [0.1, 3], logY: true, xLabel: '训练步数' }); },
                },
            ],
            notes: '训练准确率和测试准确率几乎重合、都停在 90% 附近——这不是过拟合，而是<b>欠拟合</b>：模型容量不够（对比 Step 08：MLP 的训练准确率能到 ~100%）。把 <b>学习率 η</b> 调到 0.3 或 1 会更快到达同一个平台；调到 0.01 则 3000 步都还没收敛。模板随训练的变化：先学到每个类别的「平均形状」，再慢慢长出区分相似类别的细节。',
            warn: 'Softmax 回归的 W 可以初始化为 0（每一列收到的梯度不同，自然就分开了）；多层网络不行——如果隐藏层的权重全都相同，所有隐藏单元收到的梯度也完全相同，永远学成同一个特征（Step 10 讲初始化）。',
        },
        // ───────────────────────────── 05 隐藏层与非线性 ─────────────────────────────
        {
            id: 'nonlin', title: '隐藏层与非线性：为什么线性层叠起来还是线性', tag: 'MODEL',
            desc: '要突破线性的极限，就在输入和输出之间加一层<strong>隐藏层</strong>：<code>h = act(xW₁ + b₁)</code>，再 <code>z = hW₂ + b₂</code>。关键是激活函数 act：如果去掉它，两层线性变换的复合 <code>x(W₁W₂) + (b₁W₂ + b₂)</code> 仍然是一个线性变换——叠多少层都等价于一层。第一张图就是这样一个「两层线性网络」在 XOR 上的结果：只能画出一条直线边界，准确率约 50%（XOR 不是线性可分的——Minsky & Papert 1969 正是用它说明单层感知机的局限）。加上逐元素的非线性之后，每个隐藏单元变成一个「折痕」或「台阶」（第二张图的白线是 16 个单元各自的分界线），组合起来就能拼出弯曲的边界；足够宽的单隐藏层网络可以以任意精度逼近连续函数（万能近似定理，Cybenko 1989、Hornik et al. 1989）。常用的激活：sigmoid（早期神经网络）、tanh、ReLU（2010 年代以来的默认选择：计算便宜，正半轴梯度恒为 1）、GELU（BERT、GPT 系列 Transformer 的默认，第 07 篇的 FFN 用的就是它）。',
            formula: [
                'h = act(x · W_1 + b_1)      z = h · W_2 + b_2      p = softmax(z)',
                '去掉 act：z = x (W_1 W_2) + (b_1 W_2 + b_2) = x · W′ + b′   // 仍是线性；W′ = W₁W₂ 的秩 ≤ min(H, 10)',
                'σ(x) = 1/(1 + e^{−x})，σ′ = σ(1 − σ) ≤ 1/4      tanh′ = 1 − tanh^2      ReLU(x) = max(0, x)',
                'GELU(x) = x · Φ(x) ≈ ½x(1 + tanh[√(2/π)(x + 0.044715x^3)])   // Φ = 标准正态分布函数；本页用 tanh 近似',
            ],
            code: {
                title: 'C++ · actFn / actGrad', src: `
                // 隐藏层的激活函数（顶栏「激活函数」）；Identity = 不加激活（XOR 的对照组）。库里的 Act 没有这一项，所以本页自己定义
                enum class ActType { Identity, Sigmoid, Tanh, ReLU, GELU };

                // act(z)：逐元素作用在线性打分上
                float actFn(ActType act, float z) {
                    const float k = 0.7978845608f;                                    // √(2/π)
                    switch (act) {
                        case ActType::Sigmoid: return 1.f / (1.f + std::exp(-z));
                        case ActType::Tanh: return std::tanh(z);
                        case ActType::ReLU: return z > 0 ? z : 0.f;
                        case ActType::GELU: return 0.5f * z * (1 + std::tanh(k * (z + 0.044715f * z * z * z)));   // tanh 近似（GPT-2 同款）
                        default: return z;
                    }
                }

                // act′(z)：反向传播时与上游梯度逐元素相乘
                float actGrad(ActType act, float z) {
                    const float k = 0.7978845608f;
                    switch (act) {
                        case ActType::Sigmoid: { float s = actFn(ActType::Sigmoid, z); return s * (1 - s); }   // 最大 0.25（z = 0 处）
                        case ActType::Tanh: { float t = std::tanh(z); return 1 - t * t; }
                        case ActType::ReLU: return z > 0 ? 1.f : 0.f;                  // 约定 z = 0 处取 0
                        case ActType::GELU: {
                            float t = std::tanh(k * (z + 0.044715f * z * z * z));
                            return 0.5f * (1 + t) + 0.5f * z * (1 - t * t) * k * (1 + 3 * 0.044715f * z * z);
                        }
                        default: return 1.f;
                    }
                }`,
            },
            previews: [
                {
                    id: 'xorLin', label: '没有激活函数：两层线性网络', sub: 'XOR：对角的两团是同一类（蓝 / 橙）· 背景 = 网络的预测（越亮越确定）· 红圈 = 判错', large: true, needs: ['xor'],
                    render: (c, S) => P02.xorPlot(c, S, 'lin'),
                },
                {
                    id: 'xorMLP', label: '加上激活函数', sub: '白线 = 16 个隐藏单元各自的分界线 wⱼ·x + bⱼ = 0；决策边界由它们拼成 · 切换顶栏的 <b>激活函数</b>', large: true, needs: ['xor'],
                    render: (c, S) => P02.xorPlot(c, S, 'mlp'),
                },
                {
                    id: 'actfn', label: '激活函数 act(z)', sub: '粗线 = 顶栏选中的激活', size: [360, 220],
                    render: (c, S) => {
                        const xs = []; for (let i = 0; i <= 160; i++) xs.push(-4 + i / 20);
                        P02.curves(c, ['sigmoid', 'tanh', 'relu', 'gelu'].map((a) => ({ x: xs, y: xs.map((z) => P02.act1(a, z)), color: P02.COL.act[a], width: a === S.act ? 3 : 1.2, label: P02.ACTNAME[a] })), { x: [-4, 4], y: [-1.2, 4], xLabel: 'z', legendAt: [0, 0], legendAlign: 'left' });
                    },
                },
                {
                    id: 'actgrad', label: '导数 act′(z)', sub: 'sigmoid 的导数最大只有 0.25；ReLU 在正半轴恒为 1、负半轴恒为 0', size: [360, 200],
                    render: (c, S) => {
                        const xs = []; for (let i = 0; i <= 160; i++) xs.push(-4 + i / 20);
                        P02.curves(c, ['sigmoid', 'tanh', 'relu', 'gelu'].map((a) => ({ x: xs, y: xs.map((z) => P02.dact1(a, z)), color: P02.COL.act[a], width: a === S.act ? 3 : 1.2, label: P02.ACTNAME[a] })), { x: [-4, 4], y: [-0.2, 1.25], xLabel: 'z', hmarks: [{ y: 0.25, color: '#555' }], legendAt: [0, 0], legendAlign: 'left' });
                    },
                },
            ],
            notes: '切换 <b>激活函数</b> 看两件事：第二张图的 XOR 边界（ReLU 是折线拼成的多边形，tanh / sigmoid / GELU 更光滑）；导数图里 sigmoid 的导数最大只有 0.25——每穿过一层 sigmoid，梯度至少缩小 4 倍，这在 Step 10 的深层网络里会变成梯度消失。',
            warn: 'XOR 是二维玩具，只用来说明「线性叠加还是线性」：两个网络都是 2 → 16 → 2、Xavier 初始化、全量 batch 的 Adam 训练 300 步。本页的 GELU 用 tanh 近似（与 GPT-2、本系列共享库的 <code>gelu</code> 一致），PyTorch 的 <code>F.gelu</code> 默认是 erf 精确版，两者最大相差约 4.7 × 10⁻⁴。',
        },
        // ───────────────────────────── 06 前向传播 ─────────────────────────────
        {
            id: 'forward', title: '前向传播：一张图穿过网络', tag: 'FORWARD',
            desc: '前向传播就是按顺序算一遍：256 个像素 → 隐藏层 H 个单元的线性打分 <code>z₁</code> → 激活 <code>h₁</code> → 输出层 10 个 logit <code>z₂</code> → softmax 概率。网络图里每个方块是一个隐藏单元，颜色是它对这张图的激活值；连到输出的线是它对预测类别的贡献 <code>h₁ⱼ · W₂[j, ŷ]</code>，最终的打分就是这些贡献之和——第二层是在对「隐藏特征」做加权投票，而不再直接对像素投票。前向时要把每层的 z 和 h 都存下来（cache），反向传播要用。',
            formula: [
                'z_1 = x · W_1 + b_1 ∈ R^H      h_1 = act(z_1)      z_2 = h_1 · W_2 + b_2 ∈ R^{10}      p = softmax(z_2)',
                'batch：Z_1 = X W_1 + b_1 [B × H] → H_1 → Z_2 = H_1 W_2 + b_2 [B × 10] → P   // 每层一次 GEMM 算完整个 batch',
                '参数量 = 256·H + H + H·10 + 10 = 267H + 10   // H = 64 时 17 098 个，约是 Softmax 回归的 6.7 倍',
            ],
            code: {
                title: 'C++ · MLP / mlpForward（返回 cache）', src: `
                // MLP：sizes = {256, H, …, 10}；第 l 层 W[l] [sizes[l] × sizes[l+1]]、b[l] [1 × sizes[l+1]]；隐藏层激活 act，输出层接 softmax
                struct MLP {
                    std::vector<int> sizes;
                    ActType act = ActType::ReLU;
                    std::vector<Tensor> W, b;
                };

                // 前向时存下来、反向传播要用的中间量：A[l] = 第 l 层的输入（A[0] = X），Z[l] = 第 l 层的线性输出，M[l] = dropout 掩码，P = 概率
                struct Cache {
                    std::vector<Tensor> A, Z, M;
                    Tensor P;
                };

                // 前向传播（整个 batch）。drop > 0 时做 inverted dropout（训练时，Step 11），r = 掩码用的随机数
                Cache mlpForward(const MLP& net, const Tensor& X, float drop = 0.f, Rng* r = nullptr) {
                    Cache c;
                    c.A.push_back(X);
                    int L = (int)net.W.size();
                    for (int l = 0; l < L; l++) {
                        Tensor Z = gemm(c.A[l], net.W[l], net.b[l]);                  // Z = A·W + b：一次 GEMM 算完整个 batch
                        c.Z.push_back(Z);
                        if (l == L - 1) break;                                         // 输出层不加激活
                        Tensor A = zeros(Z.rows, Z.cols);
                        for (int i = 0; i < Z.size(); i++) A.v[i] = actFn(net.act, Z.v[i]);
                        if (drop > 0) {         // inverted dropout：以概率 p 置零、其余放大 1/(1 − p)，期望不变；推理时不做
                            Tensor M = zeros(Z.rows, Z.cols);
                            for (int i = 0; i < M.size(); i++) {
                                M.v[i] = uniform(*r) < 1 - drop ? 1 / (1 - drop) : 0.f;
                                A.v[i] *= M.v[i];
                            }
                            c.M.push_back(M);
                        }
                        c.A.push_back(A);
                    }
                    c.P = softmax(c.Z.back());
                    return c;
                }`,
            },
            previews: [
                {
                    id: 'netdiag', label: '一张图穿过 MLP', sub: '方块颜色 = 隐藏单元的激活 h₁ⱼ（橙 > 0、蓝 < 0、暗 ≈ 0）· 连线 = 对预测类别贡献最大的 24 个单元（橙 = 加分、蓝 = 减分，粗细 ∝ |h₁ⱼ·W₂[j, ŷ]|）', live: true, large: true,
                    render: (c, S) => P02.netDiagram(c, S),
                },
                {
                    id: 'hidden', label: '隐藏层：z₁ 与 h₁ = act(z₁)', sub: '灰 = 线性打分 z₁ · 紫 = 激活后的 h₁（ReLU 把负值全部置 0）· 横轴 = 隐藏单元编号', live: true, size: [360, 200],
                    render: (c, S) => {
                        const t = S.tr, xs = Array.from({ length: t.H }, (_, j) => j);
                        let z = 0; for (let j = 0; j < t.H; j++) if (t.h1[j] === 0) z++;
                        PL.draw.plot(c, { x: [-1, t.H], xLabel: `隐藏单元 j（h₁ 恰好为 0 的有 ${z} / ${t.H} 个）`, series: [{ data: Array.from(t.z1), x: xs, bar: true, color: '#59606a' }, { data: Array.from(t.h1), x: xs, bar: true, color: LM.C.purple }], hmarks: [{ y: 0, color: '#3a3f46' }] });
                    },
                },
                {
                    id: 'outputs', label: '输出层：p = softmax(z₂)', sub: '绿 = 正确类别 · 红 = 判错时的预测类别 · 对比 Step 02 的 Softmax 回归', live: true, size: [360, 180],
                    render: (c, S) => { const t = S.tr; P02.barChart(c, Array.from(t.p), [...'0123456789'], { range: [0, 1.15], ticks: [0, 0.5, 1], values: true, valueMin: 0.005, tx: 40, colorFor: (v, i) => (i === t.y ? LM.C.green : i === t.pred ? LM.C.red : LM.C.blue), labelColor: (i) => (i === t.y ? LM.C.green : LM.C.gray), title: `MLP 第 ${S.mlp.step} 步 · ℓ = −log p_y = ${t.loss.toFixed(3)}`, titleColor: t.pred === t.y ? LM.C.green : LM.C.red }); },
                },
            ],
            notes: '切到 <b>sigmoid</b>：几乎所有隐藏单元都「半亮」（σ(0) = 0.5），信息靠细小的差别传递；ReLU / GELU 下有相当一部分单元恰好为（或接近）0，是一种稀疏表示。改 <b>隐藏层宽度 H</b> 可以看到网络图的单元数变化（训练从头开始）。在 Step 08 的错例里点一张判错的图，看看是哪些单元把票投给了错误的类别。',
            warn: '这里每帧都用<strong>当前</strong>（正在训练的）权重重算选中样本：训练刚开始时输出接近均匀分布 0.1，训练结束后才是「学会了」的样子。',
        },
        // ───────────────────────────── 07 反向传播 ─────────────────────────────
        {
            id: 'backprop', title: '反向传播：链式法则从输出往回传，再做三方梯度校验', tag: 'BACKWARD',
            desc: '要用梯度下降训练 W₁、b₁、W₂、b₂，需要损失对每个参数的偏导。<strong>反向传播</strong>就是把链式法则按「从输出到输入」的顺序组织起来（Rumelhart, Hinton & Williams 1986 让它广为人知）：从输出端的 <code>δ₂ = ∂ℓ/∂z₂ = p − y</code> 出发，每经过一个线性层，先用该层的输入算出参数梯度（<code>dW = 输入ᵀ · δ</code>），再把 δ 乘以权重的转置传回上一层；经过激活函数时逐元素乘以它的导数。每个中间量只算一次，总代价大约是前向的 2 倍——这是神经网络能训练的根本原因（用有限差分对每个参数求导，要做 2 × 参数个数 次前向）。写完反传一定要做<strong>梯度校验</strong>。最后一张图是三方对照：手写反传（float64）、中心差分（float64）、本系列的自动微分库（float32，和 PyTorch autograd 同一类工具；代码里用 C++ 版 <code>minml_autograd.h</code>，页面里实际运行的是与它逐个算子对应的 JS 版 <code>LM.AG</code>）。',
            formula: [
                'δ_2 = ∂ℓ/∂z_{2} = p − y                          dW_{2} = h_1^T · δ_2      db_{2} = δ_2',
                'δ_1 = ∂ℓ/∂z_{1} = (δ_2 · W_2^T) ⊙ act′(z_1)      dW_{1} = x^T · δ_1        db_{1} = δ_1   // ⊙ = 逐元素乘',
                'batch：δ_2 = (P − Y)/B；dW = (上一层输出)^T · δ；db = δ 按行求和   // 与 Step 04 的 Softmax 回归是同一个公式',
                '梯度校验：∂L/∂θ_{i} ≈ [L(θ + ε e_i) − L(θ − ε e_i)] / 2ε      相对误差 = |a − b| / max(|a|, |b|)',
            ],
            code: {
                title: 'C++ · mlpBackward / gradCheck / autogradGrads', src: `
                // 每层的梯度：dW[l]、db[l] 与 W[l]、b[l] 同形状；dZ[l] = δ_l = ∂L/∂Z_l（本步图里画的 δ₁、δ₂）
                struct Grads { std::vector<Tensor> dW, db, dZ; };

                // 手写反向传播：δ_L = (P − Y)/B；每层 dW = Aᵀ·δ（+ λW），db = Σ_行 δ；再 δ ← (δ·Wᵀ) ⊙ M ⊙ act′(Z) 传回上一层
                Grads mlpBackward(const MLP& net, const Cache& c, const std::vector<int>& y, float wd = 0.f) {
                    int L = (int)net.W.size(), n = c.P.rows;
                    Grads g;
                    g.dW.resize(L); g.db.resize(L); g.dZ.resize(L);
                    Tensor delta = c.P;                                                // δ_L = ∂L/∂Z_L = (P − Y)/B
                    for (int i = 0; i < n; i++) {
                        for (int k = 0; k < delta.cols; k++) delta.at(i, k) /= n;
                        delta.at(i, y[i]) -= 1.f / n;
                    }
                    for (int l = L - 1; l >= 0; l--) {
                        const Tensor& A = c.A[l];
                        const Tensor& W = net.W[l];
                        int fi = W.rows, fo = W.cols;
                        Tensor dW = zeros(fi, fo), db = zeros(1, fo);
                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < fo; j++) db.v[j] += delta.at(i, j);   // db = Σ_i δ_i
                            for (int p = 0; p < fi; p++) {
                                float a = A.at(i, p);
                                if (a == 0) continue;                                  // 输入为 0（黑像素、没激活的 ReLU 单元）：这一行不贡献梯度
                                for (int j = 0; j < fo; j++) dW.at(p, j) += a * delta.at(i, j);   // dW = Aᵀ·δ：每个样本贡献一个外积
                            }
                        }
                        if (wd > 0) for (int q = 0; q < dW.size(); q++) dW.v[q] += wd * W.v[q];   // L2 正则：dW += λ·W（只加在权重上，Step 11）
                        g.dW[l] = dW; g.db[l] = db; g.dZ[l] = delta;
                        if (l == 0) break;                                             // 输入不需要梯度
                        Tensor dA = zeros(n, fi);                                      // 传回上一层的输出：∂L/∂A_l = δ·Wᵀ
                        for (int i = 0; i < n; i++)
                            for (int p = 0; p < fi; p++) {
                                float s = 0;
                                for (int j = 0; j < fo; j++) s += delta.at(i, j) * W.at(p, j);
                                dA.at(i, p) = s;
                            }
                        for (int q = 0; q < dA.size(); q++) {
                            if (!c.M.empty()) dA.v[q] *= c.M[l - 1].v[q];               // dropout 置零的单元没有梯度
                            dA.v[q] *= actGrad(net.act, c.Z[l - 1].v[q]);              // 穿过激活：⊙ act′(Z)
                        }
                        delta = dA;
                    }
                    return g;
                }

                // ── 梯度校验要在 float64 下做 ──
                // 中心差分的分子 L(θ + ε) − L(θ − ε) 是两个几乎相等的数相减：float32 只有约 7 位有效数字，ε = 10⁻⁵ 时剩下的
                // 几乎全是舍入噪声。所以把网络拷成 double 再算：前向 / 反向与 mlpForward / mlpBackward 逐行相同，只是 float 换成 double
                // （下面两个函数只给原型、实现略；页面 JS 也是把同一份内核编译出一个 float64 副本）
                struct MLP64 { std::vector<int> sizes; ActType act; std::vector<std::vector<double>> W, b; };
                struct Grads64 { std::vector<std::vector<double>> dW, db; };

                // double 版的 crossEntropyLoss(mlpForward(net, X).Z.back(), y)（实现略）
                double lossFn64(const MLP64& net, const Tensor& X, const std::vector<int>& y);

                // double 版的 mlpBackward(net, mlpForward(net, X), y)（实现略）
                Grads64 mlpBackward64(const MLP64& net, const Tensor& X, const std::vector<int>& y);

                // 拷贝权重：float → double
                MLP64 toFloat64(const MLP& net) {
                    MLP64 d;
                    d.sizes = net.sizes;
                    d.act = net.act;
                    for (const Tensor& W : net.W) d.W.push_back(std::vector<double>(W.v.begin(), W.v.end()));
                    for (const Tensor& b : net.b) d.b.push_back(std::vector<double>(b.v.begin(), b.v.end()));
                    return d;
                }

                // 相对误差 |a − b| / max(|a|, |b|)（分母设个下限，避免 0/0）
                double relErr(double a, double b) { return std::fabs(a - b) / std::max({std::fabs(a), std::fabs(b), 1e-12}); }

                // 整个张量的相对误差 ‖a − b‖ / ‖a‖：对照自动微分的结果时用（全部元素）
                double relL2(const std::vector<double>& a, const std::vector<float>& b) {
                    double num = 0, den = 0;
                    for (size_t i = 0; i < a.size(); i++) { num += (a[i] - b[i]) * (a[i] - b[i]); den += a[i] * a[i]; }
                    return std::sqrt(num / std::max(den, 1e-300));
                }

                // 第三方对照：用本系列的自动微分库（minml_autograd.h，float32；和 PyTorch 的 autograd 是同一类工具）算同一个网络的梯度
                Grads autogradGrads(const MLP& net, const Tensor& X, const std::vector<int>& y) {
                    int L = (int)net.W.size();
                    std::vector<Var> W, b;
                    for (int l = 0; l < L; l++) { W.push_back(param(net.W[l])); b.push_back(param(net.b[l])); }
                    Var h = constant(X);
                    for (int l = 0; l < L; l++) {
                        h = linear(h, W[l], b[l]);                                     // h·W + b，同时把这个算子的反向闭包记进 tape
                        if (l == L - 1) break;
                        if (net.act == ActType::ReLU) h = relu(h);
                        else if (net.act == ActType::GELU) h = gelu(h);                // 库里的 gelu 也是 tanh 近似
                        else if (net.act == ActType::Tanh) h = tanhAct(h);
                        else if (net.act == ActType::Sigmoid) h = sigmoid(h);
                    }
                    // 交叉熵内部先做 log-softmax；backward 沿 tape 逆序执行每个算子的反向闭包（链式法则）
                    backward(crossEntropy(h, y));
                    Grads g;
                    for (int l = 0; l < L; l++) { g.dW.push_back(gradOf(W[l])); g.db.push_back(gradOf(b[l])); }
                    return g;
                }

                // 一个参数张量的校验结果：relNum = 抽查元素里「手写 vs 差分」的最大相对误差，relAg = 「手写 vs 自动微分」全部元素的相对 L2 误差
                struct CheckResult { std::string name; double relNum, relAg; };

                // 三方梯度校验：手写反传（float64）vs 中心差分（float64）vs 自动微分（float32）。
                // 差分：每个张量抽查 |梯度| 最大的 3 个 + 随机 2 个非零元素；自动微分：比较全部元素
                std::vector<CheckResult> gradCheck(const MLP& net32, const Tensor& X, const std::vector<int>& y, double eps = 1e-5) {
                    MLP64 net = toFloat64(net32);
                    Grads64 g = mlpBackward64(net, X, y);                              // 手写反传
                    Grads ag = autogradGrads(net32, X, y);                             // 自动微分
                    Rng r = makeRng(12345);
                    int L = (int)net.W.size();
                    std::vector<CheckResult> out;
                    for (int k = 0; k < 2 * L; k++) {                                  // 依次是 W₁ … W_L、b₁ … b_L
                        bool isW = k < L;
                        int l = isW ? k : k - L;
                        std::vector<double>& P = isW ? net.W[l] : net.b[l];
                        const std::vector<double>& G = isW ? g.dW[l] : g.db[l];
                        std::vector<int> order;                                        // 非零梯度的下标，按 |梯度| 从大到小
                        for (int i = 0; i < (int)G.size(); i++) if (std::fabs(G[i]) > 1e-9) order.push_back(i);
                        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return std::fabs(G[a]) > std::fabs(G[b]); });
                        std::vector<int> pick;
                        for (int i = 0; i < std::min(3, (int)order.size()); i++) pick.push_back(order[i]);
                        for (int tries = 0; (int)pick.size() < std::min(5, (int)order.size()) && tries < 50; tries++) {
                            int i = order[(int)(uniform(r) * order.size())];
                            if (std::find(pick.begin(), pick.end(), i) == pick.end()) pick.push_back(i);
                        }
                        double worst = 0;
                        for (int i : pick) {
                            double old = P[i];
                            P[i] = old + eps; double lp = lossFn64(net, X, y);
                            P[i] = old - eps; double lm = lossFn64(net, X, y);
                            P[i] = old;
                            double num = (lp - lm) / (2 * eps);                         // 中心差分：截断误差 O(ε²)
                            worst = std::max(worst, relErr(G[i], num));
                        }
                        const Tensor& A = isW ? ag.dW[l] : ag.db[l];
                        out.push_back({std::string(isW ? "W" : "b") + std::to_string(l + 1), worst, relL2(G, A.v)});
                    }
                    return out;
                }`,
            },
            previews: [
                {
                    id: 'cgraph', label: '计算图上的前向值与反向梯度', sub: '蓝 = 前向（‖·‖ = 向量的 L2 范数）· 红 = 反向：从 ∂ℓ/∂ℓ = 1 出发，每过一个节点乘一次局部导数 · 选中样本、当前权重', live: true, large: true,
                    render: (c, S) => P02.compGraph(c, S),
                },
                {
                    id: 'delta2', label: 'δ₂ = p − y（输出层误差，10 维）', sub: '和 Step 03 的 Softmax 回归是同一个公式', live: true, size: [360, 160],
                    render: (c, S) => { const t = S.tr; P02.barChart(c, Array.from(t.d2), [...'0123456789'], { symmetric: true, minRange: 0.05, values: true, valueMin: 0.005, colorFor: (v) => (v < 0 ? LM.C.green : LM.C.red), labelColor: (i) => (i === t.y ? LM.C.green : LM.C.gray), title: `‖δ₂‖ = ${P02.norm(t.d2).toPrecision(2)}（纵轴随数值自动缩放）`, titleColor: LM.C.gray, tx: 40 }); },
                },
                {
                    id: 'delta1', label: 'δ₁ = (δ₂·W₂ᵀ) ⊙ act′(z₁)（隐藏层误差，H 维）', sub: '橙 > 0、蓝 < 0、灰 = 恰好为 0（ReLU 没激活的单元：这一步它们的入边权重不更新）', live: true, size: [360, 160],
                    render: (c, S) => { const t = S.tr; let z = 0; for (let j = 0; j < t.H; j++) if (t.d1[j] === 0) z++; P02.barChart(c, Array.from(t.d1), null, { symmetric: true, colorFor: (v) => (v === 0 ? '#4a5058' : v > 0 ? LM.C.orange : LM.C.blue), title: `${t.H} 个隐藏单元 · ${z} 个 δ₁ 恰好为 0`, titleColor: LM.C.gray, tx: 40 }); },
                },
                {
                    id: 'dW2', label: 'dW₂ = h₁ᵀ·δ₂（转置显示：10 行 × H 列）', sub: '单个样本的梯度是外积（秩 1）：每一行 ∝ h₁，每一列 ∝ δ₂', live: true, size: [360, 170],
                    render: (c, S) => {
                        const t = S.tr, H = t.H, T = new Float32Array(10 * H);
                        for (let j = 0; j < H; j++) for (let k = 0; k < 10; k++) T[k * H + j] = t.dW2[j * 10 + k];
                        LM.draw.heat(c, T, 10, H, { symmetric: true, map: 'diverge', rowLabels: [...'0123456789'], legend: false, square: false, padT: 18 });
                        const { ctx } = PL.draw.ctx(c, { keep: true });
                        P02.txt(ctx, `‖dW₂‖ = ${P02.norm(t.dW2).toPrecision(3)} · 行 = 类别 k，列 = 隐藏单元 j`, 40, 3, { bold: true });
                    },
                },
                {
                    id: 'gcheck', label: '三方梯度校验', sub: '绿 = 手写反传 vs 中心差分（都是 float64，ε = 10⁻⁵，每个张量抽 5 个元素的最大相对误差）· 橙 = 手写反传 vs LM.AG（float32，全部元素的相对 L2 误差）', size: [360, 300],
                    render: (c, S) => P02.gcPanel(c, S.gc, S),
                },
            ],
            notes: '点不同的样本：δ₂ 在正确类别上为负（要把它的分数推高），在抢走概率的类别上为正。δ₁ 里 ReLU 没激活的单元（z₁ ≤ 0）梯度恰好为 0。梯度校验对四种激活都成立，两种对照的相对误差都在 10⁻⁷ 上下或更小：差分那一列主要是中心差分本身的舍入误差（loss 差值 ÷ 2ε），LM.AG 那一列是 float32 的舍入误差。改 <b>隐藏层宽度 / 激活函数 / 初始化</b> 会重新生成网络并重新校验。',
            warn: '有限差分在 ReLU 的拐点 z = 0 附近会失准：如果 ±ε 的扰动让某个 z 跨过 0，数值梯度就不对了（CS231n 讲义里说的 kinks）。float32 下也不要做有限差分：舍入误差约为 10⁻⁷ / ε，ε 取小了噪声大、取大了截断误差大——所以校验一律用 float64。',
        },
        // ───────────────────────────── 08 训练循环 ─────────────────────────────
        {
            id: 'trainMLP', title: '训练循环：MLP 在浏览器里实时学习', tag: 'TRAIN',
            desc: '把前向、反向、更新三步放进循环，就是完整的训练：每步取 32 张图 → forward → loss → backward → SGD 更新全部 267H + 10 个参数。下面是页面里正在进行的真实训练（每帧 4 步，3000 步后停止）：准确率曲线（灰虚线 = Step 04 的 Softmax 回归）、第一层 H 个隐藏单元的权重图（每个单元一张 16×16 的「感受野」）、测试集上的混淆矩阵和判错的样本。和 Softmax 回归相比：MLP 的测试准确率高出约 5 个百分点，训练准确率能冲到接近 100%——容量够了；但训练和测试之间也拉开了差距，这是<strong>过拟合</strong>的开始（Step 11）。',
            formula: [
                'for t = 1 … T:   (X_b, y_b) ← 下一个 batch      cache ← mlpForward(X_b)      grads ← mlpBackward(cache, y_b)      θ ← θ − η · grads',
                '1 个 epoch = 遍历一次训练集 = 4000 / 32 = 125 步   // 3000 步 = 24 个 epoch，一共看过 96 000 张图',
            ],
            code: {
                title: 'C++ · initMLP / trainMLP / evaluate / confusionMatrix', src: `
                // 初始化方式（顶栏「初始化」）；对应的标准差 σ 由 initStd 给出（定义见 Step 10）
                enum class Init { Small, Xavier, He, Large };
                float initStd(Init init, int fanIn, int fanOut);

                // W ~ N(0, σ²)，b = 0；sizes = {256, H, 10}
                MLP initMLP(const std::vector<int>& sizes, ActType act, Init init, Rng& r) {
                    MLP net;
                    net.sizes = sizes;
                    net.act = act;
                    for (size_t l = 0; l + 1 < sizes.size(); l++) {
                        int fanIn = sizes[l], fanOut = sizes[l + 1];
                        net.W.push_back(randn(fanIn, fanOut, initStd(init, fanIn, fanOut), r));
                        net.b.push_back(zeros(1, fanOut));
                    }
                    return net;
                }

                // SGD 更新全部参数：θ ← θ − η·dθ
                void sgdUpdate(MLP& net, const Grads& g, float lr) {
                    for (size_t l = 0; l < net.W.size(); l++) {
                        for (int i = 0; i < net.W[l].size(); i++) net.W[l].v[i] -= lr * g.dW[l].v[i];
                        for (int i = 0; i < net.b[l].size(); i++) net.b[l].v[i] -= lr * g.db[l].v[i];
                    }
                }

                MLP trainMLP(const Dataset& tr, int H = 64, ActType act = ActType::ReLU, Init init = Init::He,
                             float lr = 0.1f, int steps = 3000, int B = 32, uint32_t seed = 0) {
                    Rng r = makeRng(seed);
                    MLP net = initMLP({256, H, 10}, act, init, r);
                    BatchStream bs = makeBatchStream(range(0, tr.n), r);
                    for (int t = 0; t < steps; t++) {
                        std::vector<int> idx = nextBatch(bs, B, r);
                        std::vector<int> yb = takeLabels(tr.y, idx);
                        Cache c = mlpForward(net, takeRows(tr.X, idx));               // 1. 前向，存下每层的 Z 和 A
                        Grads g = mlpBackward(net, c, yb);                            // 2. 反向，手写链式法则
                        sgdUpdate(net, g, lr);                                        // 3. SGD 更新全部参数
                    }
                    return net;
                }

                // 评估结果：平均交叉熵、准确率、每个样本的预测
                struct EvalResult { float loss, acc; std::vector<int> pred; };

                // 推理：不做 dropout
                EvalResult evaluate(const MLP& net, const Tensor& X, const std::vector<int>& y) {
                    Tensor P = mlpForward(net, X).P;
                    EvalResult e;
                    e.pred = ints(X.rows);
                    double loss = 0;
                    int correct = 0;
                    for (int i = 0; i < X.rows; i++) {
                        e.pred[i] = argmaxRow(P, i);
                        correct += e.pred[i] == y[i];
                        loss -= std::log(std::max(1e-30, (double)P.at(i, y[i])));    // −log p_y
                    }
                    e.loss = (float)(loss / X.rows);
                    e.acc = (float)correct / X.rows;
                    return e;
                }

                // 混淆矩阵 C[真实 · k + 预测]：行 = 真实类别，列 = 预测类别
                std::vector<int> confusionMatrix(const std::vector<int>& y, const std::vector<int>& pred, int k = 10) {
                    std::vector<int> C = ints(k * k);
                    for (size_t i = 0; i < y.size(); i++) C[y[i] * k + pred[i]]++;
                    return C;
                }`,
            },
            previews: [
                {
                    id: 'mlpAcc', label: '准确率：MLP vs Softmax 回归', sub: '蓝 = MLP 训练集（固定抽 500 张评估）· 橙 = MLP 测试集 500 张 · 灰虚线 = Softmax 回归的测试集（Step 04）', live: true, large: true, size: [360, 250],
                    render: (c, S) => {
                        const R = S.mlp, H = R.hist, Hs = S.soft.hist;
                        P02.curves(c, [
                            { x: Hs.step, y: Hs.testA.map((v) => 100 * v), color: LM.C.gray, dash: [5, 4], width: 1.5, label: `Softmax 测试 ${P02.pct(P02.last(Hs.testA))}` },
                            { x: H.step, y: H.trainA.map((v) => 100 * v), color: LM.C.train, width: 2, label: `MLP 训练 ${P02.pct(P02.last(H.trainA))}` },
                            { x: H.step, y: H.testA.map((v) => 100 * v), color: LM.C.val, width: 2.4, label: `MLP 测试 ${P02.pct(P02.last(H.testA))}` },
                        ], { x: [0, P02.CAP], y: [0, 100], xLabel: '训练步数', legendAt: [1, 0.38], extra: [{ label: P02.status(R, P02.NTRAIN), color: LM.C.yellow, noLine: true }] });
                    },
                },
                {
                    id: 'mlpLoss', label: '交叉熵损失', sub: '训练 loss 一路下降，测试 loss 降到一定程度就停住：两者之差就是泛化差距 · 对数纵轴', live: true, size: [360, 200],
                    render: (c, S) => { const H = S.mlp.hist; P02.curves(c, [{ x: H.step, y: H.trainL, color: LM.C.train, width: 2, label: `训练 ${LM.fmt(P02.last(H.trainL))}` }, { x: H.step, y: H.testL, color: LM.C.val, width: 2, label: `测试 ${LM.fmt(P02.last(H.testL))}` }], { x: [0, P02.CAP], y: [0.005, 5], logY: true, xLabel: '训练步数' }); },
                },
                {
                    id: 'w1img', label: '第一层权重 W₁ 的 H 列：每个隐藏单元的「感受野」', sub: '每张图单独归一化 · 橙 = 正、蓝 = 负 · 训练开始时是 He 初始化的随机噪声，训练只在它上面叠加了一层结构（见右图）', live: true, large: true,
                    render: (c, S) => {
                        const W1 = S.mlp.net.W[0], H = S.mlp.net.sizes[1];
                        S._gW = P02.digitGrid(c, Array.from({ length: H }, (_, j) => { const m = P02.absMax(W1, H, j, 256); return { img: W1, stride: H, offset: j, range: [-m, m] }; }), { cols: H > 64 ? 16 : 8, map: 'diverge', title: `${H} 个隐藏单元 · ${P02.statusShort(S.mlp)}`, titleColor: LM.C.yellow, tx: 40 });
                    },
                    probe: (x, y, S, PL, info) => { const k = P02.gridHit(S._gW, info); return k < 0 ? '' : `隐藏单元 j = ${k}：W₁[:, ${k}] 排回 16×16`; },
                },
                {
                    id: 'dw1img', label: 'ΔW₁ = W₁(t) − W₁(0)：训练到底改了什么', sub: '同一批单元，减去初始权重后只剩学到的部分 · 每张图单独归一化 · dW₁ = xᵀ·δ₁ 是训练图像的加权和，所以 ΔW₁ 也由笔画组成；边缘像素从来没有墨水，权重从未被更新', live: true, large: true,
                    render: (c, S) => {
                        const R = S.mlp, W1 = R.net.W[0], W0 = R.W0[0], H = R.net.sizes[1];
                        const dW = S._dW1 && S._dW1.length === W1.length ? S._dW1 : (S._dW1 = new Float32Array(W1.length));
                        let m = 0; for (let i = 0; i < W1.length; i++) { dW[i] = W1[i] - W0[i]; if (Math.abs(dW[i]) > m) m = Math.abs(dW[i]); }
                        S._gD = P02.digitGrid(c, Array.from({ length: H }, (_, j) => { const mj = P02.absMax(dW, H, j, 256); return { img: dW, stride: H, offset: j, range: [-mj, mj] }; }), { cols: H > 64 ? 16 : 8, map: 'diverge', title: `第 ${R.step} 步 · 最大 |ΔW₁| = ${m.toFixed(3)}（初始 σ = ${P02.initStd(S.init, 256, H).toFixed(3)}）`, titleColor: LM.C.yellow, tx: 40 });
                    },
                    probe: (x, y, S, PL, info) => { const k = P02.gridHit(S._gD, info); return k < 0 ? '' : `隐藏单元 j = ${k}：ΔW₁[:, ${k}] 排回 16×16`; },
                },
                {
                    id: 'confusion', label: '混淆矩阵（测试集 500 张）', sub: '行 = 真实类别、列 = 预测类别；对角线 = 判对，红格 = 混淆（颜色按最大的非对角格归一化）', live: true, size: [360, 330],
                    render: (c, S) => { const R = S.mlp; S._cf = P02.confusion(c, R.conf[1], { title: `MLP 第 ${R.evalStep} 步的评估 · 测试准确率 ${P02.pct(P02.last(R.hist.testA))}` }); },
                    probe: (x, y, S, PL, info) => { const g = S._cf; if (!g) return ''; const q = Math.floor((info.dx - g.x0) / g.cs), r = Math.floor((info.dy - g.y0) / g.cs); if (q < 0 || q > 9 || r < 0 || r > 9) return ''; return `真实 ${r} → 预测 ${q}：${S.mlp.conf[1][r * 10 + q]} 张`; },
                },
                {
                    id: 'errors', label: '判错的测试样本', sub: '标签 = 真实 → 预测 · <b>点击</b>选中它，到 Step 06 / 07 看它的激活和梯度', live: true, size: [360, 172],
                    render: (c, S) => {
                        const R = S.mlp, d = S.data, pred = R.pred[1], list = [];
                        let nErr = 0; for (let i = 0; i < d.te.n; i++) if (pred[i] !== d.te.y[i]) { nErr++; if (list.length < 30) list.push(i); }
                        S._gE = P02.digitGrid(c, list.map((i) => ({ img: d.te.X.subarray(i * 256, (i + 1) * 256), label: `${d.te.y[i]}→${pred[i]}`, labelColor: LM.C.red, outline: i === S.tr.i ? LM.C.yellow : null })), { cols: 10, labels: true, title: `第 ${R.evalStep} 步：判错 ${nErr} / ${d.te.n} 张（最多显示 30 张）`, titleColor: LM.C.gray, tx: 40 });
                        S._gEidx = list;
                    },
                    probe: (x, y, S, PL, info) => { const k = P02.gridHit(S._gE, info); return k < 0 ? '' : `测试样本 #${S._gEidx[k]}（点击选中）`; },
                    pick: (x, y, S, PL, info) => { const k = P02.gridHit(S._gE, info); if (k >= 0) PL.setParam('sample', S._gEidx[k]); },
                },
                {
                    id: 'batchImgs', label: '当前这一步的 batch（32 张）', sub: '按 epoch 洗牌、不放回地取下标；数字 = 标签', live: true, size: [360, 98],
                    render: (c, S) => { const R = S.mlp; P02.digitGrid(c, Array.from({ length: R.B }, (_, k) => ({ img: R.Xb.subarray(k * 256, (k + 1) * 256), label: R.yb[k] })), { cols: 16, labels: true, title: `第 ${R.step} 步取到的 ${R.B} 张`, titleColor: LM.C.gray, tx: 40 }); },
                },
            ],
            notes: '拖 <b>学习率 η</b>：0.01 很慢，1 更快但 loss 曲线更抖。换 <b>激活函数</b>：sigmoid 明显最慢（导数 ≤ 1/4）。<b>初始化</b> 选「太大 σ=1」：一开始 loss 就很高，最终测试准确率也明显更差。混淆矩阵里最常见的错误都发生在长得像的数字之间。点判错的样本，可以到 Step 06 / 07 看它的激活和梯度。',
            warn: '隐藏单元的权重图不像 Softmax 回归的模板那样「像数字」：它们更像笔画片段、边缘和局部形状的检测器，由第二层再组合成类别——这就是<strong>表示学习</strong>（representation learning）。图中的「训练准确率」是在固定的 500 张训练图上评估的（为了省时间），与全量训练集只差抽样误差；曲线上的每个点来自分帧评估：先拷一份权重快照，再每帧评估 50 张，评估完一轮（1000 张）记一个点。',
        },
        // ───────────────────────────── 09 优化器 ─────────────────────────────
        {
            id: 'optim', title: '优化器：SGD、Momentum 与 Adam', tag: 'OPTIM', params: ['lr', 'hidden', 'act', 'init'],
            desc: '同样的梯度，用法可以不同。<strong>SGD</strong> 直接沿 −g 走一步。<strong>Momentum</strong>（动量，Polyak 1964 的 heavy ball）把历史梯度累积成「速度」v：方向一致的维度上越走越快，来回振荡的维度上正负抵消。<strong>Adam</strong>（Kingma & Ba 2015）为每个参数维护梯度的一阶矩 m（动量）和二阶矩 v（梯度平方的滑动平均），用 <code>m / √v</code> 作为更新方向：每个参数的步长被自动归一化，梯度小的方向也能迈开步子；m、v 都从 0 开始，前几步偏小，所以要除以 <code>1 − βᵗ</code> 做<strong>偏差修正</strong>。第一张图是一个弯曲峡谷形二维函数上的真实轨迹（点击地形换起点）；后两张是三个优化器从同一个初始化、用同一串 batch 训练同一个 MLP 的真实曲线。',
            formula: [
                'SGD：θ ← θ − η g',
                'Momentum：v ← μ v + g，θ ← θ − η v   // μ = 0.9；PyTorch SGD(momentum=0.9) 的写法，稳态时等效步长 η/(1 − μ) = 10η',
                'Adam：m ← β_1 m + (1 − β_1) g，v ← β_2 v + (1 − β_2) g^2，m̂ = m/(1 − β_1^t)，v̂ = v/(1 − β_2^t)，θ ← θ − η · m̂/(√v̂ + ε)',
                '默认值：β_1 = 0.9，β_2 = 0.999，ε = 10^{−8}，η = 10^{−3}   // Kingma & Ba 2015 推荐的设置',
                '二维峡谷：f(θ) = 0.02 θ_1^2 + 2 (θ_2 − 0.4 sin(1.2 θ_1))^2   // 最小值在原点；横穿峡谷的曲率 ≈ 4，SGD 稳定要求 η < 2/4',
            ],
            code: {
                title: 'C++ · SGD / Momentum / Adam / run2d / trainWith', src: `
                // 三种优化器（Step 08 用的就是最简单的 SGD）
                enum class Opt { SGD, Momentum, Adam };

                // 超参数 + 状态。m、v：每个参数张量各一份，形状相同、从 0 开始——Momentum 的速度存在 v 里，Adam 的一阶 / 二阶矩存在 m / v 里
                struct Optimizer {
                    Opt kind = Opt::SGD;
                    float lr = 0.1f, mu = 0.9f;                                       // Momentum 的 μ
                    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;                        // Adam 的默认值（Kingma & Ba 2015）
                    int t = 0;                                                        // 已经走了几步：偏差修正要用
                    std::vector<Tensor> m, v;
                };

                // 对一个参数张量 p 走一步（g = 它的梯度，m / v = 它的状态；调用前 o.t 已经加 1）
                void updateTensor(const Optimizer& o, Tensor& p, const Tensor& g, Tensor& m, Tensor& v) {
                    float c1 = 1 - std::pow(o.b1, o.t), c2 = 1 - std::pow(o.b2, o.t);   // 偏差修正：m、v 从 0 开始，前几步偏小
                    for (int i = 0; i < p.size(); i++) {
                        float gi = g.v[i];
                        float& mi = m.v[i];
                        float& vi = v.v[i];
                        if (o.kind == Opt::SGD) p.v[i] -= o.lr * gi;                                     // θ ← θ − η·g
                        else if (o.kind == Opt::Momentum) { vi = o.mu * vi + gi; p.v[i] -= o.lr * vi; }  // v ← μ·v + g；θ ← θ − η·v
                        else {
                            mi = o.b1 * mi + (1 - o.b1) * gi;                                            // 一阶矩：梯度的滑动平均
                            vi = o.b2 * vi + (1 - o.b2) * gi * gi;                                       // 二阶矩：梯度平方的滑动平均
                            p.v[i] -= o.lr * (mi / c1) / (std::sqrt(vi / c2) + o.eps);                   // θ ← θ − η·m̂ / (√v̂ + ε)
                        }
                    }
                }

                // MLP 的优化器：状态按 W₁ … W_L、b₁ … b_L 的顺序排
                Optimizer makeOpt(Opt kind, float lr, const MLP& net) {
                    Optimizer o;
                    o.kind = kind;
                    o.lr = lr;
                    for (const Tensor& W : net.W) { o.m.push_back(zeros(W.rows, W.cols)); o.v.push_back(zeros(W.rows, W.cols)); }
                    for (const Tensor& b : net.b) { o.m.push_back(zeros(b.rows, b.cols)); o.v.push_back(zeros(b.rows, b.cols)); }
                    return o;
                }

                void optStep(Optimizer& o, MLP& net, const Grads& g) {
                    o.t++;
                    int L = (int)net.W.size();
                    for (int l = 0; l < L; l++) {
                        updateTensor(o, net.W[l], g.dW[l], o.m[l], o.v[l]);
                        updateTensor(o, net.b[l], g.db[l], o.m[L + l], o.v[L + l]);
                    }
                }

                // 二维峡谷 f(θ) = 0.02·θ₁² + 2·(θ₂ − 0.4·sin 1.2θ₁)²：沿谷底很平、横穿很陡，最小值在原点
                float valley(float x, float y) { float r = y - 0.4f * std::sin(1.2f * x); return 0.02f * x * x + 2 * r * r; }

                // ∇f(θ)，θ 和梯度都是 [1 × 2]
                Tensor valleyGrad(const Tensor& p) {
                    float x = p.v[0], y = p.v[1], r = y - 0.4f * std::sin(1.2f * x);
                    Tensor g = zeros(1, 2);
                    g.v[0] = 0.04f * x - 4 * r * 0.48f * std::cos(1.2f * x);
                    g.v[1] = 4 * r;
                    return g;
                }

                // 从 (x0, y0) 出发走 steps 步，返回轨迹。页面里：SGD η = 0.4、Momentum η = 0.4 μ = 0.9、Adam η = 0.2
                // （横穿峡谷的曲率 ≈ 4，SGD 的稳定上限是 η < 2/4 = 0.5）
                std::vector<Pt> run2d(Opt kind, float lr, float x0, float y0, int steps = 150) {
                    Optimizer o;
                    o.kind = kind;
                    o.lr = lr;
                    Tensor p = zeros(1, 2), m = zeros(1, 2), v = zeros(1, 2);           // θ 当成一个 1 × 2 的参数张量
                    p.v[0] = x0;
                    p.v[1] = y0;
                    std::vector<Pt> path = {{x0, y0}};
                    for (int t = 0; t < steps; t++) {
                        o.t++;
                        updateTensor(o, p, valleyGrad(p), m, v);
                        if (!std::isfinite(p.v[0]) || std::fabs(p.v[0]) > 1e3f) break;   // 发散了就停
                        path.push_back({p.v[0], p.v[1]});
                    }
                    return path;
                }

                // 三个优化器：同一个 seed → 同一个初始化、同一串 batch，只换更新规则（页面里 SGD 和 Momentum 用顶栏的 η，Adam 用 10⁻³）
                MLP trainWith(Opt kind, float lr, const Dataset& tr, int H = 64, ActType act = ActType::ReLU, Init init = Init::He,
                              int steps = 1200, int B = 32, uint32_t seed = 0) {
                    Rng r = makeRng(seed);
                    MLP net = initMLP({256, H, 10}, act, init, r);
                    Optimizer opt = makeOpt(kind, lr, net);
                    BatchStream bs = makeBatchStream(range(0, tr.n), r);
                    for (int t = 0; t < steps; t++) {
                        std::vector<int> idx = nextBatch(bs, B, r);
                        std::vector<int> yb = takeLabels(tr.y, idx);
                        Grads g = mlpBackward(net, mlpForward(net, takeRows(tr.X, idx)), yb);
                        optStep(opt, net, g);
                    }
                    return net;
                }`,
            },
            previews: [
                {
                    id: 'traj2d', label: '弯曲峡谷上的 150 步', sub: '<span style="color:#6ab0e0">━ SGD η = 0.4</span> · <span style="color:#e0a060">━ Momentum η = 0.4, μ = 0.9</span> · <span style="color:#5ad07a">━ Adam lr = 0.2</span> · 背景亮度 = log f · <b>点击设置起点</b>', large: true, size: [360, 280],
                    render: (c, S) => { S._o2tf = P02.valleyPlot(c, S); },
                    probe: (x, y, S, PL, info) => { if (!S._o2tf) return ''; const p = S._o2tf.inv(info.dx, info.dy); return `θ = (${p[0].toFixed(2)}, ${p[1].toFixed(2)})  f = ${P02.f2(p[0], p[1]).toFixed(4)}`; },
                    pick: (x, y, S, PL, info) => { if (!S._o2tf) return; const p = S._o2tf.inv(info.dx, info.dy); S.o2start = [Math.max(-4.9, Math.min(4.9, +p[0].toFixed(2))), Math.max(-2.5, Math.min(2.5, +p[1].toFixed(2)))]; return 'frame'; },
                },
                {
                    id: 'f2curve', label: 'f(θ_t) 随步数', sub: '对数纵轴 · 同一个起点', size: [360, 200],
                    render: (c, S) => { P02.curves(c, S.o2.map((r, k) => ({ x: r.fs.map((_, i) => i), y: r.fs, color: P02.COL.opt[k], width: 2, label: `${P02.OPTS[k].name}  ${LM.fmt(P02.last(r.fs))}` })), { x: [0, 150], y: [1e-8, 30], logY: true, xLabel: '步数 t', legendAt: [0, 0.6], legendAlign: 'left' }); },
                },
                {
                    id: 'optLoss', label: 'MLP 上的训练 loss（batch loss 的指数滑动平均）', sub: 'SGD 与 Momentum 用顶栏的 η，Adam 用默认 lr = 10⁻³ · 三个训练只在这两张图可见时运行（滚到这里才开始）', live: true, needs: ['optRuns'], size: [360, 230],
                    render: (c, S) => {
                        const runs = S.optRuns;
                        P02.curves(c, runs.map((R, k) => {
                            const e = LM.ema(R.batchLoss, 0.98), st = Math.max(1, Math.ceil(e.length / 300)), x = [], y = [];
                            for (let i = 0; i < e.length; i += st) { x.push(i + 1); y.push(e[i]); }
                            return { x, y, color: P02.COL.opt[k], width: 2, label: `${P02.OPTS[k].name}${k < 2 ? ` η=${S.lr}` : ' 1e-3'}  ${LM.fmt(P02.last(e))}` };
                        }), { x: [0, P02.OPT_CAP], y: [1e-3, 4], logY: true, xLabel: `训练步数（第 ${runs[0].step} / ${P02.OPT_CAP} 步）` });
                    },
                },
                {
                    id: 'optAcc', label: '测试准确率', sub: '同一个测试集 500 张', live: true, needs: ['optRuns'], size: [360, 200],
                    render: (c, S) => {
                        const runs = S.optRuns;
                        P02.curves(c, runs.map((R, k) => ({ x: R.hist.step, y: R.hist.testA.map((v) => 100 * v), color: P02.COL.opt[k], width: 2, label: `${P02.OPTS[k].name} ${P02.pct(P02.last(R.hist.testA))}` })), { x: [0, P02.OPT_CAP], y: [60, 100], xLabel: `训练步数（第 ${runs[0].step} / ${P02.OPT_CAP} 步）`, legendAt: [1, 0.45] });
                    },
                },
            ],
            notes: '点击地形把起点放到别处：SGD（蓝，η = 0.4，已接近这个峡谷的稳定上限 2/4 = 0.5）在陡的方向上来回锯齿、沿平缓的谷底慢慢爬；Momentum（橙，同样的 η）冲过头又拐回来、振荡很大，但沿谷底快得多；Adam（绿）每个坐标的步长被归一化，平稳地沿谷底前进。MLP 上，同样 η 的 Momentum 比 SGD 收敛快得多——它的稳态等效步长是 10η；如果把 Momentum 的 η 降到 η/10，它就和 SGD 几乎一样（Node 原型里验证过）。Adam 不调参、直接用默认的 10⁻³，loss 也比 η = 0.1 的 SGD 低一些——省掉调学习率的麻烦，是它流行的主要原因。',
            warn: 'Adam 的学习率含义和 SGD 不同：它近似「每个参数每步移动的幅度」，所以 MLP 上取 10⁻³、二维函数上取 0.2（参数的尺度不同）。训练大模型用的 AdamW（第 08 篇）把权重衰减从梯度里拆出来单独做（Loshchilov & Hutter 2019），β₂ 也常取 0.95。二维函数上「谁最快」取决于学习率和起点：这里的 η 只是合理的固定取值，不是公平的调参比赛。',
        },
        // ───────────────────────────── 10 初始化与梯度流 ─────────────────────────────
        {
            id: 'init', title: '初始化与梯度流：深层网络为什么难训', tag: 'INIT',
            desc: '把隐藏层叠到 8 层（顶栏 <b>深度</b>），每层 64 个单元，在 64 张训练图上做一次前向 + 反向，统计每层激活和权重梯度的标准差。前向时每过一层，激活的方差大约乘以 <code>fan_in · Var(W) · (激活函数的增益)</code>；反向时梯度的方差按类似的因子缩放。这个因子小于 1 → 逐层指数衰减（<strong>梯度消失</strong>）；大于 1 → 指数爆炸。<strong>Xavier 初始化</strong>（Glorot & Bengio 2010）让 tanh 这类在 0 附近近似线性的对称激活的因子 ≈ 1；ReLU 把一半的输入置零、二阶矩减半，所以 <strong>He 初始化</strong>（He et al. 2015）把权重方差加倍到 2/fan_in。sigmoid 的导数最大只有 1/4、输出均值是 0.5，无论怎样初始化，权重梯度都会从输出层往输入层逐层缩小——这是 2010 年前深层网络很难训练的重要原因之一。',
            formula: [
                'Var(z_l) ≈ fan_in · Var(W_l) · E[h_{l−1}^2]   // 每层一个乘法因子 → L 层之后是因子^L',
                'ReLU：E[h^2] = Var(z)/2   ⇒   He：Var(W) = 2/fan_in      tanh 在 0 附近 ≈ 恒等   ⇒   Xavier：Var(W) = 2/(fan_in + fan_out)',
                '反向：∂L/∂h_{l−1} = δ_l · W_l^T   ⇒   方差也乘以 fan_out · Var(W) · E[act′^2]',
                '本页四种初始化：太小 σ = 0.01      Xavier σ = √(2/(fan_in + fan_out))      He σ = √(2/fan_in)      太大 σ = 1',
            ],
            code: {
                title: 'C++ · initStd / layerStats', src: `
                // 初始化的标准差 σ：太小 0.01 / 太大 1 / Xavier √(2/(fan_in + fan_out)) / He √(2/fan_in)
                float initStd(Init init, int fanIn, int fanOut) {
                    switch (init) {
                        case Init::Small: return 0.01f;
                        case Init::Large: return 1.f;
                        case Init::Xavier: return std::sqrt(2.f / (fanIn + fanOut));  // Glorot & Bengio 2010
                        default: return std::sqrt(2.f / fanIn);                       // He et al. 2015（为 ReLU 设计）
                    }
                }

                // 所有元素的标准差（除以 n）
                float stdOf(const Tensor& A) {
                    double mean = 0, var = 0;
                    for (float x : A.v) mean += x;
                    mean /= A.size();
                    for (float x : A.v) var += (x - mean) * (x - mean);
                    return (float)std::sqrt(var / A.size());
                }

                // actStd[l] = 第 l + 1 个隐藏层输出的标准差（前向），gradStd[l] = 第 l + 1 层权重梯度的标准差（反向，最后一个是输出层）
                struct LayerStats { std::vector<float> actStd, gradStd; };

                // depth 个宽 width 的隐藏层，在一个 batch 上（页面里：64 张训练图）做一次前向 + 反向
                LayerStats layerStats(const Tensor& X, const std::vector<int>& y, int depth = 8, int width = 64,
                                      ActType act = ActType::ReLU, Init init = Init::He, uint32_t seed = 0) {
                    std::vector<int> sizes = {256};
                    for (int l = 0; l < depth; l++) sizes.push_back(width);
                    sizes.push_back(10);
                    Rng r = makeRng(seed);
                    MLP net = initMLP(sizes, act, init, r);
                    Cache c = mlpForward(net, X);
                    Grads g = mlpBackward(net, c, y);
                    LayerStats st;
                    for (int l = 1; l <= depth; l++) st.actStd.push_back(stdOf(c.A[l]));
                    for (const Tensor& dW : g.dW) st.gradStd.push_back(stdOf(dW));
                    return st;
                }`,
            },
            previews: [
                {
                    id: 'actStd', label: '各隐藏层激活的标准差（前向）', sub: '<span style="color:#c678dd">━ 太小</span> · <span style="color:#56b6c2">━ Xavier</span> · <span style="color:#5ad07a">━ He</span> · <span style="color:#e06c75">━ 太大</span> · 粗线 = 顶栏选中的初始化 · 对数纵轴', large: true, size: [360, 250], needs: ['flow'],
                    render: (c, S) => {
                        const L = S.depth, xs = Array.from({ length: L }, (_, l) => l + 1);
                        P02.curves(c, S.flow.map((f, k) => ({ x: xs, y: f.aStd, color: P02.COL.init[k], width: P02.INITS[k] === S.init ? 3.2 : 1.4 })), { x: [1, L], logY: true, xLabel: `隐藏层 l（${P02.ACTNAME[S.act]}，每层 64 个单元）` });
                    },
                },
                {
                    id: 'gradStd', label: '各层权重梯度 dW 的标准差（反向）', sub: '最右一点 = 输出层 · 颜色同上 · 同一个 batch、同一个前向', size: [360, 220], needs: ['flow'],
                    render: (c, S) => {
                        const L = S.depth, xs = Array.from({ length: L + 1 }, (_, l) => l + 1);
                        P02.curves(c, S.flow.map((f, k) => ({ x: xs, y: f.gStd, color: P02.COL.init[k], width: P02.INITS[k] === S.init ? 3.2 : 1.4 })), { x: [1, L + 1], logY: true, xLabel: '层 l（1 = 第一层 … L+1 = 输出层）' });
                    },
                },
                {
                    id: 'actHist', label: '每层激活值的直方图（选中的初始化）', sub: '行 = 层（上 → 下 = 浅 → 深），列 = 激活值，亮 = 多；每行单独归一化，超出范围的值堆在两端', size: [360, 250], needs: ['flow'],
                    render: (c, S) => {
                        const f = S.flow[P02.INITS.indexOf(S.init)], L = S.depth;
                        const lay = LM.draw.heat(c, f.hist, L, f.NB, { rowLabels: Array.from({ length: L }, (_, l) => 'L' + (l + 1)), map: 'inferno', range: [0, 1], legend: false, square: false, padB: 18, title: `${P02.INITNAME[S.init]} · ${P02.ACTNAME[S.act]} · 激活值 ∈ [${f.lo}, ${f.hi}]` });
                        const { ctx } = PL.draw.ctx(c, { keep: true }), ticks = S.act === 'sigmoid' ? [0, 0.5, 1] : S.act === 'tanh' ? [-1, -0.5, 0, 0.5, 1] : [-0.5, 0, 1, 2, 2.5];
                        ticks.forEach((v, q) => P02.txt(ctx, String(v), lay.x0 + (v - f.lo) / (f.hi - f.lo) * f.NB * lay.cw, lay.y0 + L * lay.ch + 3, { align: q === 0 ? 'left' : q === ticks.length - 1 ? 'right' : 'center', color: LM.C.dim }));
                    },
                },
            ],
            notes: '默认 ReLU + He：每层的标准差几乎不变（一条平线）。换成 <b>太小</b>：激活每层缩小十几倍，8 层后只剩 10⁻¹⁰ ~ 10⁻¹¹，信号根本传不到输出；<b>太大</b>：每层放大约 6 倍（≈ √(64/2)）。切到 <b>sigmoid</b>：权重梯度从输出层到第一层缩小好几个数量级（梯度消失）。把 <b>深度</b> 拖到 16，指数效应更夸张。Step 08 的训练也使用这里的 <b>初始化</b> 设置（两层网络受的影响小得多，但「太大」仍明显变差）。',
            warn: '这些统计只是<strong>初始化时</strong>一个 batch 上的快照，训练开始后分布会变。现代深层网络还依靠归一化层（BatchNorm、LayerNorm）和残差连接让信号保持在合理范围（第 07 篇的 Transformer 每个子层都有 LayerNorm + 残差）。',
        },
        // ───────────────────────────── 11 正则化 ─────────────────────────────
        {
            id: 'regularize', title: '过拟合与正则化：权重衰减、dropout、早停', tag: 'REG', params: ['nsub', 'wd', 'dropout', 'lr', 'hidden', 'act', 'init'],
            desc: '训练数据越少，模型越容易把训练集「背下来」：训练准确率 100%，没见过的数据上却错很多。这一步只用前 N 张训练图（顶栏 <b>小训练集 N</b>）训练同一个 MLP，另取 500 张没参与训练的训练图作<strong>验证集</strong>（validation set），同时跑三个版本：不加正则；加<strong>权重衰减</strong>（L2 正则：loss 里加 λ/2·‖W‖²，梯度多出 λW，每步把权重往 0 拉一点）；加 <strong>dropout</strong>（训练时每个隐藏单元以概率 p 被置零、其余放大 1/(1−p)，推理时不做——Srivastava et al. 2014）。验证 loss 的最低点就是<strong>早停</strong>（early stopping）的位置：在那里存一份权重，之后的训练只会让模型对训练集越来越「自信」、对新数据越来越差。',
            formula: [
                'L_reg = L + (λ/2) Σ_l ‖W_l‖^2   ⇒   dW_{l} += λ W_l   // 只衰减权重矩阵，不衰减偏置',
                'dropout（训练时）：h̃ = h ⊙ m / (1 − p)，m_j ~ Bernoulli(1 − p)      推理时：h̃ = h   // inverted dropout：期望不变',
                '早停：t* = argmin_{t} L_val(t)，返回 θ_{t*}；测试集只在 t* 处看一次',
            ],
            code: {
                title: 'C++ · trainRegularized（权重衰减 + dropout + 早停）', src: `
                // 早停：记住验证 loss 最低的那份权重
                struct EarlyStop { float valLoss = 1e30f; int step = 0; MLP net; };

                // tr = 小训练集（页面里是前 N 张训练图），val = 另外 500 张不参与训练的图；wd = λ（L2 正则），drop = p（dropout）
                EarlyStop trainRegularized(const Dataset& tr, const Dataset& val, float wd = 0.f, float drop = 0.f,
                                           int H = 64, ActType act = ActType::ReLU, Init init = Init::He,
                                           float lr = 0.1f, int steps = 1800, int B = 32, int evalEvery = 30, uint32_t seed = 0) {
                    Rng r = makeRng(seed), dropR = makeRng(seed ^ 0x5bd1e995u);       // dropout 掩码用另一串随机数
                    MLP net = initMLP({256, H, 10}, act, init, r);
                    BatchStream bs = makeBatchStream(range(0, tr.n), r);
                    EarlyStop best;
                    for (int t = 1; t <= steps; t++) {
                        std::vector<int> idx = nextBatch(bs, B, r);
                        std::vector<int> yb = takeLabels(tr.y, idx);
                        Cache c = mlpForward(net, takeRows(tr.X, idx), drop, &dropR);  // dropout 只在训练时
                        Grads g = mlpBackward(net, c, yb, wd);                        // L2：dW += λ·W
                        sgdUpdate(net, g, lr);
                        if (t % evalEvery == 0) {
                            EvalResult e = evaluate(net, val.X, val.y);                // 推理：不做 dropout
                            if (e.loss < best.valLoss) { best.valLoss = e.loss; best.step = t; best.net = net; }
                        }
                    }
                    return best;
                }`,
            },
            previews: [
                {
                    id: 'regAcc', label: '训练（虚线）vs 验证（实线）准确率', sub: '<span style="color:#e06c75">━ 无正则</span> · <span style="color:#6ab0e0">━ 权重衰减 λ</span> · <span style="color:#5ad07a">━ dropout p</span> · 训练准确率在其中最多 250 张上评估 · 三个训练只在本图可见时运行', live: true, large: true, size: [360, 250], needs: ['regRuns'],
                    render: (c, S) => {
                        const runs = S.regRuns, names = ['无正则', `λ = ${S.wd}`, `p = ${S.dropout}`], series = [];
                        runs.forEach((R, k) => {
                            const H = R.hist;
                            series.push({ x: H.step, y: H.trainA.map((v) => 100 * v), color: P02.COL.reg[k], dash: [4, 3], width: 1.3 });
                            series.push({ x: H.step, y: H.valA.map((v) => 100 * v), color: P02.COL.reg[k], width: 2.2, label: `${names[k]} 验证 ${P02.pct(P02.last(H.valA))}` });
                        });
                        P02.curves(c, series, { x: [0, P02.REG_CAP], y: [0, 100], xLabel: `训练步数（N = ${S.nsub} 张训练图）`, legendAt: [1, 0.42], extra: [{ label: P02.status(runs[0], S.nsub), color: LM.C.yellow, noLine: true }] });
                    },
                },
                {
                    id: 'regLoss', label: '验证集 loss', sub: '竖线 = 各自的早停点（验证 loss 最低处）· 只画最低点附近的纵轴范围', live: true, size: [360, 220], needs: ['regRuns'],
                    render: (c, S) => {
                        const runs = S.regRuns, names = ['无正则', 'λ', 'p'];
                        let mn = Infinity; runs.forEach((R) => R.hist.valL.forEach((v) => { if (v < mn) mn = v; }));
                        const best = runs.map((R) => { const v = R.hist.valL; let b = 0; for (let i = 1; i < v.length; i++) if (v[i] < v[b]) b = i; return b; });
                        P02.curves(c, runs.map((R, k) => ({ x: R.hist.step, y: R.hist.valL, color: P02.COL.reg[k], width: 2, label: `${names[k]} 最低 ${R.hist.valL[best[k]].toFixed(3)} @${R.hist.step[best[k]]}` })),
                            { x: [0, P02.REG_CAP], y: [Math.max(0, 0.8 * mn), Math.max(1, 2.2 * mn)], xLabel: `训练步数（第 ${runs[0].step} / ${P02.REG_CAP} 步）`, marks: runs.map((R, k) => ({ x: R.hist.step[best[k]], color: P02.COL.reg[k] })) });
                    },
                },
                {
                    id: 'regTable', label: '汇总', sub: '测试准确率取在各自的早停点（测试集只看这一次）', live: true, size: [360, 150], needs: ['regRuns'],
                    render: (c, S) => {
                        const runs = S.regRuns, names = ['无正则', `权重衰减 λ = ${S.wd}`, `dropout p = ${S.dropout}`], lines = [{ text: `N = ${S.nsub} 张训练图 · 验证 500 张 · ${P02.status(runs[0], S.nsub)}`, color: LM.C.gray }];
                        runs.forEach((R, k) => {
                            const H = R.hist, v = H.valL; let b = 0; for (let i = 1; i < v.length; i++) if (v[i] < v[b]) b = i;
                            lines.push([{ t: names[k] + '  ', color: P02.COL.reg[k], bold: true }, { t: `训练 ${P02.pct(P02.last(H.trainA))} · 验证 ${P02.pct(P02.last(H.valA))}` }]);
                            lines.push({ text: `    早停 @ 第 ${H.step[b]} 步：验证 loss ${v[b].toFixed(3)} → 测试 ${P02.pct(H.testA[b])}`, color: LM.C.dim });
                        });
                        P02.lines(c, lines, { x0: 40, indent: 28 });
                    },
                },
            ],
            notes: '默认 N = 500：三条训练准确率（虚线）都很快接近 100%；不加正则的验证 loss 在约 700 步后开始回升（过拟合：对训练集越来越自信），权重衰减和 dropout 把它压低。默认数据下实测，各自早停点的测试准确率是：无正则 82.4%、λ = 1e-2 84.0%、dropout 0.5 86.2%；而训练结束时三者的验证准确率几乎一样——正则化的好处主要体现在 loss（概率是否过度自信）和早停点上。λ = 1e-1 太强：训练准确率都上不去（欠拟合）。把 <b>小训练集 N</b> 从 100 拖到 1000：<b>更多数据</b>才是最有效的「正则化」，带来的提升远大于任何技巧（对比 Step 08 用 4000 张图的结果）。',
            warn: '早停点必须由验证集选，测试集只在最后报一次数（第 01 篇说过的原则）。这里的权重衰减是加在 loss 上的 L2 正则，对 SGD 等价于每步把权重乘以 (1 − ηλ)；对 Adam 就不等价了（梯度会被 √v 归一化）——这正是 AdamW 把两者分开的原因。',
        },
        // ───────────────────────────── 12 教学版 → 生产版 ─────────────────────────────
        {
            id: 'scaleup', title: '教学版 → 生产版：框架、GPU、CNN 与 Transformer 里的 MLP', tag: 'NEXT',
            desc: '这一页手写的每一样东西，实际项目里都由框架完成，但原理完全一样：',
            blocks: [
                {
                    type: 'html', html: `<table class="lm-tbl"><tr><th>环节</th><th>本页（教学版）</th><th>实际做法（生产版）</th></tr>
                    <tr><td>求导</td><td>手写 <code>mlpBackward()</code>，逐层链式法则</td><td>autograd：<code>backward(loss)</code> 沿前向时记录的计算图自动反传（PyTorch / JAX 的做法，PyTorch 里写作 <code>loss.backward()</code>；本系列的 <code>minml_autograd.h</code> 是同一机制的极简版，见下面的代码）；只有写自定义算子时才需要手推梯度</td></tr>
                    <tr><td>计算</td><td>JS / C++ 循环，CPU 单线程</td><td>GPU 上的批量矩阵乘（cuBLAS、Tensor Core），batch 内的样本并行计算；混合精度（fp16 / bf16）</td></tr>
                    <tr><td>数据</td><td>16×16 程序化数字，4000 张</td><td>MNIST 28×28、6 万张训练图；再往上是 ImageNet（约 128 万张训练图）。数据增强（平移、旋转、弹性形变——Simard et al. 2003 在 MNIST 上用的正是弹性形变）</td></tr>
                    <tr><td>模型</td><td>256 → H → 10 的 MLP</td><td>图像：卷积网络 CNN（LeNet-5 在 MNIST 上 0.95% 错误率）→ ResNet → ViT（第 10 篇）</td></tr>
                    <tr><td>训练技巧</td><td>SGD / Momentum / Adam、He 初始化、L2 / dropout / 早停</td><td>再加 BatchNorm / LayerNorm、残差连接、学习率 warmup + 衰减、AdamW（第 08 篇）</td></tr>
                    <tr><td>在大模型里</td><td>隐藏层 + 激活</td><td>Transformer 每一层的 FFN 就是一个 MLP：d → 4d → GELU → d，对每个 token 独立计算（第 07 篇）</td></tr></table>`,
                },
                {
                    type: 'code', title: 'C++ · 自动微分版：同一个 MLP 与训练循环（写法对应 PyTorch）', src: `
                    // 生产版的写法：模型由 Linear 层拼成，前向时每个算子自动记进 tape，backward(loss) 自动反传——不再需要手写 mlpBackward()。
                    // 这里用本系列的 minml 库（pages/cpp/*.h）；PyTorch 里对应的是 nn.Linear、F.relu、F.cross_entropy、
                    // opt.zero_grad()、loss.backward()、opt.step()
                    struct AutoMLP { Linear fc1, fc2; };

                    AutoMLP makeAutoMLP(int dIn, int hidden, int nCls, uint32_t seed) {
                        Rng r = makeRng(seed);
                        AutoMLP m;
                        m.fc1 = makeLinear(dIn, hidden, std::sqrt(2.f / dIn), r);         // W 存成 [in × out]；He 初始化、偏置 0（与页面一致）
                        m.fc2 = makeLinear(hidden, nCls, std::sqrt(2.f / hidden), r);
                        return m;
                    }

                    // 输出 logits；softmax 合并进 crossEntropy
                    Var autoForward(const AutoMLP& m, const Var& x) { return forward(m.fc2, relu(forward(m.fc1, x))); }

                    AutoMLP trainAutograd(const Dataset& tr, int epochs = 24, int B = 32, float lr = 0.1f, uint32_t seed = 0) {
                        AutoMLP m = makeAutoMLP(256, 64, 10, seed);
                        std::vector<Var> params = {m.fc1.W, m.fc1.b, m.fc2.W, m.fc2.b};
                        SGD opt = makeSGD(params, lr);
                        Rng r = makeRng(seed + 1);
                        for (int ep = 0; ep < epochs; ep++) {
                            std::vector<int> order = permutation(tr.n, r);                  // 每个 epoch 重新洗牌（DataLoader 的 shuffle=True）
                            for (int s = 0; s < tr.n; s += B) {
                                std::vector<int> idx;
                                for (int k = s; k < std::min(s + B, tr.n); k++) idx.push_back(order[k]);
                                Var x = constant(takeRows(tr.X, idx));                      // 数据是常量，不求梯度
                                Var loss = crossEntropy(autoForward(m, x), takeLabels(tr.y, idx));   // 前向（同时记录计算图）
                                zeroGrad(params);                                           // 清空上一步累加的梯度
                                backward(loss);                                             // 自动反传：替代整个手写的 mlpBackward()
                                sgdStep(opt, lr);                                           // θ ← θ − η·∇θ
                            }
                        }
                        return m;
                    }`,
                },
                {
                    type: 'previews', items: [{
                        id: 'final', label: '本页两个模型的结果', sub: '实时读取 Step 04 / 08 的训练；参数量按当前的 H 计算', live: true, size: [360, 150],
                        render: (c, S) => {
                            const s = S.soft, m = S.mlp, np = (267 * S.hidden + 10).toLocaleString('en-US').replace(/,/g, ' ');
                            P02.lines(c, [
                                [{ t: 'Softmax 回归  ', color: LM.C.blue, bold: true }, { t: `参数 2 570 · 测试 ${P02.pct(P02.last(s.hist.testA))}（第 ${s.step} 步）` }],
                                [{ t: `MLP（H = ${S.hidden}）  `, color: LM.C.orange, bold: true }, { t: `参数 ${np} · 测试 ${P02.pct(P02.last(m.hist.testA))}（第 ${m.step} 步）` }],
                                { text: '同类模型在真实 MNIST（28×28、6 万张训练图）上的测试准确率：', color: LM.C.gray },
                                { text: '· Softmax 回归：约 92%', color: LM.C.gray },
                                { text: '· 800 个隐藏单元的两层 MLP：98.4%（Simard et al. 2003）', color: LM.C.gray },
                                { text: '· LeNet-5 卷积网络：99.05%（LeCun et al. 1998）', color: LM.C.gray },
                            ], { x0: 40 });
                        },
                    }],
                },
                { type: 'note', html: '真实 MNIST 上更多的参考点（错误率）：两层 MLP 加弹性形变数据增强 0.7%（Simard et al. 2003）；在 GPU 上训练的 6 层大 MLP 加弹性形变 0.35%（Ciresan et al. 2010）——足够大的 MLP 加上足够多的（增强）数据，也能非常强。' },
                { type: 'note', html: '下一篇（03）还留在图像上：把全连接层换成在图上滑动的小卷积核，同样的手写数字只用几分之一的参数就能学好。文字怎样变成向量（Tokenizer、Embedding、位置编码）在第 06 篇，再交给第 07 篇的 Transformer——它的每一层里都有本篇的 MLP，训练它用的也是本篇的反向传播。' },
            ],
        },
    ],

    graph: {
        title: '训练一步 · MLP 的数据流（batch = 32）',
        intro: '每个节点 = 一次真实的计算（在 GPU 上是一次或几次 kernel launch；这里是 JS 循环）；pin = 张量名 · 形状。上半是前向，下半是反向：反向节点读取前向存下来的激活（cache）。<code>Params</code> 是 <b>HISTORY</b> 资源：每一步读取上一步更新后的权重（虚线回边）；<code>Params₀</code> 是初始化，只在第 0 步用。',
        types: {
            img: { color: '#6ab0e0', label: 'img · 图像 / 输入张量 f32' },
            label: { color: '#e8c25a', label: 'label · 类别标签 int32' },
            act: { color: '#c678dd', label: 'act · 前向激活 / logits / 概率' },
            param: { color: '#5ad07a', label: 'param · 可训练参数 W₁ b₁ W₂ b₂' },
            grad: { color: '#e06c75', label: 'grad · 梯度 / 误差 δ' },
            scalar: { color: '#8a8f98', label: 'scalar · 标量 / 超参数 / 指标' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用函数', open: false,
            blocks: LM.CPP.blocks(['nn']).concat([{
                label: '━ 本页公用：洗牌、下标列表、mini-batch 下标流、按下标取 batch（Step 01 / 04 / 08 / 09 / 11 / 12 都用到）━', code: `
                // 原地打乱（Fisher–Yates，与页面 JS 的 LM.shuffle 消耗随机数的顺序相同）
                void shuffleInPlace(std::vector<int>& a, Rng& r) {
                    for (int i = (int)a.size() - 1; i > 0; i--) {
                        int j = (int)(uniform(r) * (i + 1));
                        std::swap(a[i], a[j]);
                    }
                }

                // 下标列表 [a, b)：range(0, 4000) = 全部训练图
                std::vector<int> range(int a, int b) {
                    std::vector<int> idx = ints(std::max(0, b - a));
                    for (int i = 0; i < (int)idx.size(); i++) idx[i] = a + i;
                    return idx;
                }

                // 无限的 mini-batch 下标流：按 epoch 洗牌、不放回地取；一个 batch 可以跨两个 epoch（与页面里的 JS 一致）
                struct BatchStream {
                    std::vector<int> order;                                       // 当前 epoch 的样本顺序
                    int pos = 0, epoch = 0;
                };

                BatchStream makeBatchStream(const std::vector<int>& idx, Rng& r) {
                    BatchStream s;
                    s.order = idx;
                    shuffleInPlace(s.order, r);
                    return s;
                }

                // 下一个 batch 的 B 个样本下标；这个 epoch 取完了就重新洗牌
                std::vector<int> nextBatch(BatchStream& s, int B, Rng& r) {
                    std::vector<int> idx = ints(B);
                    for (int k = 0; k < B; k++) {
                        if (s.pos == (int)s.order.size()) { shuffleInPlace(s.order, r); s.pos = 0; s.epoch++; }
                        idx[k] = s.order[s.pos++];
                    }
                    return idx;
                }

                // X[idx]：按下标取出若干行（一个 batch 的图像）
                Tensor takeRows(const Tensor& X, const std::vector<int>& idx) {
                    Tensor out = zeros((int)idx.size(), X.cols);
                    for (int k = 0; k < (int)idx.size(); k++)
                        for (int j = 0; j < X.cols; j++) out.at(k, j) = X.at(idx[k], j);
                    return out;
                }

                // y[idx]：对应的标签
                std::vector<int> takeLabels(const std::vector<int>& y, const std::vector<int>& idx) {
                    std::vector<int> out = ints((int)idx.size());
                    for (int k = 0; k < (int)idx.size(); k++) out[k] = y[idx[k]];
                    return out;
                }`,
            }]),
        },
        nodes: [
            {
                id: 'gen', title: 'makeData', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'data', desc: '笔画字形 + 随机变形 → 16×16',
                dispatch: { kind: 'CPU', name: 'renderDigit', count: '4500 张 × 256 像素' },
                outputs: [{ name: 'X_train', format: '[4000, 256] f32', type: 'img' }, { name: 'y_train', format: '[4000] i32', type: 'label' }, { name: 'X_test', format: '[500, 256] f32', type: 'img' }, { name: 'y_test', format: '[500] i32', type: 'label' }],
                detail: { preview: 'grid' },
            },
            {
                id: 'hyper', title: 'HyperParams', tag: 'CB', category: 'user', desc: 'η · H · act · init · B = 32',
                outputs: [{ name: 'Hyper', format: 'scalars', type: 'scalar' }],
                detail: { desc: '用户控制的超参数：学习率 η、隐藏层宽度 H、激活函数、初始化方式；batch 大小固定为 32、上限 3000 步。它们不被训练，只决定训练怎样进行。', preview: null },
            },
            {
                id: 'initp', title: 'initMLP', tag: 'CPU', category: 'bake', lifetime: 'param', step: 'trainMLP', desc: 'W ~ N(0, σ²)，b = 0',
                dispatch: { kind: 'CPU', name: 'initMLP', count: (S) => `${(267 * S.hidden + 10).toLocaleString('en-US')} 个参数` },
                inputs: [{ name: 'Hyper', format: 'H · init', type: 'scalar' }],
                outputs: [{ name: 'Params₀', format: 'W₁ [256,H] · W₂ [H,10]', type: 'param' }],
                detail: { preview: 'w1img' },
            },
            {
                id: 'batch', title: 'nextBatch', tag: 'CPU', lifetime: 'history', step: 'trainMLP', desc: '按 epoch 洗牌取 32 个下标',
                dispatch: { kind: 'CPU', name: 'nextBatch', count: 'B = 32 行 × 256' },
                inputs: [{ name: 'X_train', format: '[4000, 256] f32', type: 'img' }, { name: 'y_train', format: '[4000] i32', type: 'label' }],
                outputs: [{ name: 'Xb', format: '[32, 256] f32', type: 'img' }, { name: 'yb', format: '[32] i32', type: 'label' }],
                detail: { preview: 'batchImgs' },
            },
            {
                id: 'fc1', title: 'linear_1 · GEMM', tag: 'CPU', lifetime: 'history', step: 'forward', desc: 'Z₁ = Xb·W₁ + b₁',
                dispatch: { kind: 'CPU', name: 'gemm', count: (S) => `32×256×${S.hidden} MAC（只算非零像素，约 ${Math.round(100 * S.data.nzFrac)}%）` },
                inputs: [{ name: 'Xb', format: '[32, 256]', type: 'img' }, { name: 'Params', format: 'W₁ b₁', type: 'param', history: true }],
                outputs: [{ name: 'Z₁', format: '[32, H] f32', type: 'act' }],
                detail: { preview: 'hidden' },
            },
            {
                id: 'actn', title: 'activation', tag: 'CPU', lifetime: 'history', step: 'nonlin', desc: 'H₁ = act(Z₁)',
                dispatch: { kind: 'CPU', name: 'actFn', count: (S) => `32 × ${S.hidden} 个元素（${P02.ACTNAME[S.act]}）` },
                inputs: [{ name: 'Z₁', format: '[32, H]', type: 'act' }],
                outputs: [{ name: 'H₁', format: '[32, H] f32', type: 'act' }],
                detail: { preview: 'actfn' },
            },
            {
                id: 'fc2', title: 'linear_2 · GEMM', tag: 'CPU', lifetime: 'history', step: 'forward', desc: 'Z₂ = H₁·W₂ + b₂',
                dispatch: { kind: 'CPU', name: 'gemm', count: (S) => `32 × ${S.hidden} × 10 次乘加` },
                inputs: [{ name: 'H₁', format: '[32, H]', type: 'act' }, { name: 'Params', format: 'W₂ b₂', type: 'param', history: true }],
                outputs: [{ name: 'Z₂', format: '[32, 10] f32', type: 'act' }],
                detail: { preview: 'outputs' },
            },
            {
                id: 'ce', title: 'softmax_xent', tag: 'CPU', lifetime: 'history', step: 'xent', desc: 'P = softmax(Z₂)，L = mean(−log P[y])',
                dispatch: { kind: 'CPU', name: 'crossEntropyLoss', count: '32 行 × 10（log-sum-exp）' },
                inputs: [{ name: 'Z₂', format: '[32, 10]', type: 'act' }, { name: 'yb', format: '[32]', type: 'label' }],
                outputs: [{ name: 'P', format: '[32, 10] f32', type: 'act' }, { name: 'L', format: 'scalar', type: 'scalar' }],
                detail: { preview: 'sampleGrad' },
            },
            {
                id: 'bw2', title: 'backward linear_2', tag: 'CPU', lifetime: 'history', step: 'backprop', desc: 'δ₂ = (P − Y)/B → dW₂, db₂, dH₁',
                dispatch: [{ kind: 'CPU', name: 'dW₂ = H₁ᵀ·δ₂', count: (S) => `${S.hidden} × 10 × 32 次乘加` }, { kind: 'CPU', name: 'dH₁ = δ₂·W₂ᵀ', count: (S) => `32 × 10 × ${S.hidden} 次乘加` }],
                inputs: [{ name: 'P', format: '[32, 10]', type: 'act' }, { name: 'yb', format: '[32]', type: 'label' }, { name: 'H₁', format: '[32, H]', type: 'act' }, { name: 'Params', format: 'W₂', type: 'param', history: true }],
                outputs: [{ name: 'dW₂', format: '[H, 10]', type: 'grad' }, { name: 'db₂', format: '[10]', type: 'grad' }, { name: 'dH₁', format: '[32, H]', type: 'grad' }],
                detail: { preview: 'delta2' },
            },
            {
                id: 'bwact', title: 'backward activation', tag: 'CPU', lifetime: 'history', step: 'backprop', desc: 'δ₁ = dH₁ ⊙ act′(Z₁)',
                dispatch: { kind: 'CPU', name: 'actGrad', count: (S) => `32 × ${S.hidden} 个元素` },
                inputs: [{ name: 'dH₁', format: '[32, H]', type: 'grad' }, { name: 'Z₁', format: '[32, H]', type: 'act' }],
                outputs: [{ name: 'δ₁', format: '[32, H]', type: 'grad' }],
                detail: { preview: 'delta1' },
            },
            {
                id: 'bw1', title: 'backward linear_1', tag: 'CPU', lifetime: 'history', step: 'backprop', desc: 'dW₁ = Xbᵀ·δ₁，db₁ = Σ δ₁',
                dispatch: { kind: 'CPU', name: 'dW₁ = Xbᵀ·δ₁', count: (S) => `256×${S.hidden}×32 MAC（不需要 dX）` },
                inputs: [{ name: 'δ₁', format: '[32, H]', type: 'grad' }, { name: 'Xb', format: '[32, 256]', type: 'img' }],
                outputs: [{ name: 'dW₁', format: '[256, H]', type: 'grad' }, { name: 'db₁', format: '[H]', type: 'grad' }],
                detail: { preview: 'cgraph' },
            },
            {
                id: 'upd', title: 'sgdUpdate', tag: 'CPU', category: 'output', lifetime: 'history', step: 'trainMLP', desc: 'θ ← θ − η·dθ',
                dispatch: { kind: 'CPU', name: 'sgdUpdate', count: (S) => `${(267 * S.hidden + 10).toLocaleString('en-US')} 个参数` },
                inputs: [{ name: 'dW₁', format: '[256, H]', type: 'grad' }, { name: 'db₁', format: '[H]', type: 'grad' }, { name: 'dW₂', format: '[H, 10]', type: 'grad' }, { name: 'db₂', format: '[10]', type: 'grad' }, { name: 'Params', format: 'θ（上一步）', type: 'param', history: true }, { name: 'Params₀', format: 't = 0 时', type: 'param' }, { name: 'Hyper', format: 'η', type: 'scalar' }],
                outputs: [{ name: 'Params', format: 'W₁ b₁ W₂ b₂', type: 'param' }],
                detail: { preview: 'mlpLoss' },
            },
            {
                id: 'evalm', title: 'evaluate', tag: 'CPU', category: 'output', lifetime: 'history', step: 'trainMLP', desc: '分帧评估：准确率 / 混淆矩阵',
                dispatch: { kind: 'CPU', name: 'evaluate', count: '权重快照 · 每帧 50 张' },
                inputs: [{ name: 'Params', format: '快照', type: 'param' }, { name: 'X_test', format: '[500, 256]', type: 'img' }, { name: 'y_test', format: '[500]', type: 'label' }],
                outputs: [{ name: 'Metrics', format: 'acc · loss · confusion', type: 'scalar' }],
                detail: { preview: 'confusion' },
            },
        ],
    },

    references: [
        { title: 'Gradient-Based Learning Applied to Document Recognition', author: 'Y. LeCun, L. Bottou, Y. Bengio, P. Haffner', venue: 'Proceedings of the IEEE 86(11)', year: 1998, note: 'MNIST 数据集与 LeNet-5（0.95% 错误率）' },
        { title: 'Learning representations by back-propagating errors', author: 'D. E. Rumelhart, G. E. Hinton, R. J. Williams', venue: 'Nature 323', year: 1986, note: '让反向传播广为人知的论文' },
        { title: 'Perceptrons', author: 'M. Minsky, S. Papert', venue: 'MIT Press', year: 1969, note: '单层感知机的局限（XOR 不可线性分割）' },
        { title: 'Approximation by Superpositions of a Sigmoidal Function', author: 'G. Cybenko', venue: 'Mathematics of Control, Signals and Systems', year: 1989, note: '万能近似定理' },
        { title: 'Understanding the difficulty of training deep feedforward neural networks', author: 'X. Glorot, Y. Bengio', venue: 'AISTATS', year: 2010, note: 'Xavier 初始化；sigmoid 深层网络的饱和与梯度消失' },
        { title: 'Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification', author: 'K. He, X. Zhang, S. Ren, J. Sun', venue: 'ICCV', year: 2015, note: 'He（Kaiming）初始化' },
        { title: 'Rectified Linear Units Improve Restricted Boltzmann Machines', author: 'V. Nair, G. E. Hinton', venue: 'ICML', year: 2010, note: 'ReLU' },
        { title: 'Gaussian Error Linear Units (GELUs)', author: 'D. Hendrycks, K. Gimpel', venue: 'arXiv:1606.08415', year: 2016, url: 'https://arxiv.org/abs/1606.08415', note: 'GELU 及其 tanh 近似' },
        { title: 'Some methods of speeding up the convergence of iteration methods', author: 'B. T. Polyak', venue: 'USSR Computational Mathematics and Mathematical Physics', year: 1964, note: '动量法（heavy ball）' },
        { title: 'Adam: A Method for Stochastic Optimization', author: 'D. P. Kingma, J. Ba', venue: 'ICLR', year: 2015, url: 'https://arxiv.org/abs/1412.6980', note: 'Adam 与偏差修正' },
        { title: 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting', author: 'N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov', venue: 'JMLR 15', year: 2014, note: 'dropout' },
        { title: 'A Simple Weight Decay Can Improve Generalization', author: 'A. Krogh, J. A. Hertz', venue: 'NIPS 4', year: 1992, note: '权重衰减（L2 正则）' },
        { title: 'Decoupled Weight Decay Regularization', author: 'I. Loshchilov, F. Hutter', venue: 'ICLR', year: 2019, note: 'AdamW：为什么 Adam 里的 L2 ≠ 权重衰减' },
        { title: 'Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis', author: 'P. Y. Simard, D. Steinkraus, J. C. Platt', venue: 'ICDAR', year: 2003, note: 'MNIST 上的 MLP / CNN 与弹性形变数据增强' },
        { title: 'Deep, Big, Simple Neural Nets Excel on Handwritten Digit Recognition', author: 'D. C. Ciresan, U. Meier, L. M. Gambardella, J. Schmidhuber', venue: 'Neural Computation 22(12)', year: 2010, note: 'GPU 上训练的大 MLP：MNIST 0.35% 错误率' },
        { title: 'CS231n: Neural Networks Part 3 — Learning and Evaluation（梯度校验）', author: 'Stanford CS231n', url: 'https://cs231n.github.io/neural-networks-3/', note: '梯度校验的实践细节（相对误差、kinks、用双精度）' },
        { title: 'Neural Networks and Deep Learning', author: 'M. Nielsen', year: 2015, url: 'http://neuralnetworksanddeeplearning.com/', note: '在线书：MNIST + 反向传播的经典入门' },
        { title: 'Deep Learning（第 6–8 章）', author: 'I. Goodfellow, Y. Bengio, A. Courville', venue: 'MIT Press', year: 2016, url: 'https://www.deeplearningbook.org/', note: '前馈网络、正则化、优化' },
    ],
});
