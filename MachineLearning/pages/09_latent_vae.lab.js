// 09_latent_vae.lab.js — 系列第 09 篇：潜空间——自编码器、VAE 与 VQ-VAE（代码块是 PyTorch；页面里的 JS 参考实现用 LM.AG 实时训练）
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js 09_latent_vae.lab.js -o 09_latent_vae.html
// 页面里的每张预览都由下面的 JS 参考实现实时算出：PCA（完整特征分解）、2 维自编码器 / VAE / VQ-VAE 的真实训练、潜空间网格解码、插值、码本使用率、先验采样。

const P09 = (() => {
    'use strict';
    const { AG } = LM;
    const IMG = 16, D = 256, NC = 10, NTRAIN = 4000, NTEST = 500;

    // ═══════════════════ 数据：与第 02 篇同一个生成器、同一个种子 → 同一批 4000 + 500 张数字 ═══════════════════
    function makeData(rand) {
        const tr = LM.digits.makeSet(NTRAIN, rand), te = LM.digits.makeSet(NTEST, rand);
        const byClass = Array.from({ length: NC }, () => []);
        for (let i = 0; i < te.n; i++) byClass[te.y[i]].push(i);
        return { tr, te, byClass, disp: Array.from({ length: NC }, (_, c) => byClass[c][0]) };
    }

    // ═══════════════════ PCA：协方差矩阵的完整特征分解（Householder 三对角化 + 隐式 QL，即 EISPACK / JAMA 的 tred2 + tql2） ═══════════════════
    function tred2(V, d, e, n) {
        for (let j = 0; j < n; j++) d[j] = V[n - 1][j];
        for (let i = n - 1; i > 0; i--) {
            let scale = 0, h = 0;
            for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
            if (scale === 0) {
                e[i] = d[i - 1];
                for (let j = 0; j < i; j++) { d[j] = V[i - 1][j]; V[i][j] = 0; V[j][i] = 0; }
            } else {
                for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
                let f = d[i - 1], g = Math.sqrt(h);
                if (f > 0) g = -g;
                e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
                for (let j = 0; j < i; j++) e[j] = 0;
                for (let j = 0; j < i; j++) {
                    f = d[j]; V[j][i] = f; g = e[j] + V[j][j] * f;
                    for (let k = j + 1; k <= i - 1; k++) { g += V[k][j] * d[k]; e[k] += V[k][j] * f; }
                    e[j] = g;
                }
                f = 0;
                for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
                const hh = f / (h + h);
                for (let j = 0; j < i; j++) e[j] -= hh * d[j];
                for (let j = 0; j < i; j++) {
                    f = d[j]; g = e[j];
                    for (let k = j; k <= i - 1; k++) V[k][j] -= f * e[k] + g * d[k];
                    d[j] = V[i - 1][j]; V[i][j] = 0;
                }
            }
            d[i] = h;
        }
        for (let i = 0; i < n - 1; i++) {
            V[n - 1][i] = V[i][i]; V[i][i] = 1;
            const h = d[i + 1];
            if (h !== 0) {
                for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
                for (let j = 0; j <= i; j++) {
                    let g = 0;
                    for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
                    for (let k = 0; k <= i; k++) V[k][j] -= g * d[k];
                }
            }
            for (let k = 0; k <= i; k++) V[k][i + 1] = 0;
        }
        for (let j = 0; j < n; j++) { d[j] = V[n - 1][j]; V[n - 1][j] = 0; }
        V[n - 1][n - 1] = 1; e[0] = 0;
    }
    function tql2(V, d, e, n) {
        for (let i = 1; i < n; i++) e[i - 1] = e[i];
        e[n - 1] = 0;
        let f = 0, tst1 = 0;
        const eps = Math.pow(2, -52);
        for (let l = 0; l < n; l++) {
            tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
            let m = l;
            while (m < n) { if (Math.abs(e[m]) <= eps * tst1) break; m++; }
            if (m > l) {
                let iter = 0;
                do {
                    iter++;
                    let g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = Math.hypot(p, 1);
                    if (p < 0) r = -r;
                    d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r);
                    const dl1 = d[l + 1];
                    let h = g - d[l];
                    for (let i = l + 2; i < n; i++) d[i] -= h;
                    f += h;
                    p = d[m];
                    let c = 1, c2 = c, c3 = c, s = 0, s2 = 0;
                    const el1 = e[l + 1];
                    for (let i = m - 1; i >= l; i--) {
                        c3 = c2; c2 = c; s2 = s;
                        g = c * e[i]; h = c * p; r = Math.hypot(p, e[i]);
                        e[i + 1] = s * r; s = e[i] / r; c = p / r;
                        p = c * d[i] - s * g;
                        d[i + 1] = h + s * (c * g + s * d[i]);
                        for (let k = 0; k < n; k++) { const vk = V[k]; h = vk[i + 1]; vk[i + 1] = s * vk[i] + c * h; vk[i] = c * vk[i] - s * h; }
                    }
                    p = -s * s2 * c3 * el1 * e[l] / dl1;
                    e[l] = s * p; d[l] = c * p;
                } while (Math.abs(e[l]) > eps * tst1 && iter < 60);
            }
            d[l] += f; e[l] = 0;
        }
    }
    /** 对称矩阵 A（n×n，行主序）的特征分解：返回 { vals（降序）, vecs: Float32Array(n·n)，第 k 行 = 第 k 个单位特征向量 } */
    function eigSym(A, n) {
        const V = []; for (let i = 0; i < n; i++) V.push(Float64Array.from(A.subarray(i * n, (i + 1) * n)));
        const d = new Float64Array(n), e = new Float64Array(n);
        tred2(V, d, e, n); tql2(V, d, e, n);
        const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => d[b] - d[a]);
        const vals = new Float64Array(n), vecs = new Float32Array(n * n);
        order.forEach((c, k) => {
            vals[k] = Math.max(0, d[c]);
            let bi = 0; for (let j = 1; j < n; j++) if (Math.abs(V[j][c]) > Math.abs(V[bi][c])) bi = j;
            const sg = V[bi][c] < 0 ? -1 : 1;   // 确定的符号：绝对值最大的分量为正
            for (let j = 0; j < n; j++) vecs[k * n + j] = sg * V[j][c];
        });
        return { vals, vecs };
    }
    /** PCA：均值、协方差的全部 256 个主成分；测试集上「保留前 k 个主成分」的逐像素 MSE（k = 0…256，正交投影所以一次算完） */
    function pca(tr, te) {
        const n = tr.n, X = tr.X, mean = new Float64Array(D), C = new Float64Array(D * D);
        for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) mean[j] += X[i * D + j] / n;
        const nzI = new Int32Array(D), nzV = new Float64Array(D);
        for (let i = 0; i < n; i++) {            // XᵀX 只累加非零像素（约 1/3），再减 n·μμᵀ
            let m = 0; for (let j = 0; j < D; j++) { const v = X[i * D + j]; if (v !== 0) { nzI[m] = j; nzV[m] = v; m++; } }
            for (let a = 0; a < m; a++) { const ra = nzI[a] * D, va = nzV[a]; for (let b = a; b < m; b++) C[ra + nzI[b]] += va * nzV[b]; }
        }
        for (let a = 0; a < D; a++) for (let b = a; b < D; b++) { const v = C[a * D + b] / n - mean[a] * mean[b]; C[a * D + b] = v; C[b * D + a] = v; }
        const { vals, vecs } = eigSym(C, D);
        let total = 0; for (let k = 0; k < D; k++) total += vals[k];
        const cum = new Float64Array(D + 1); for (let k = 0; k < D; k++) cum[k + 1] = cum[k] + vals[k] / total;
        // 测试集投影系数：p_ik = u_kᵀ(x_i − μ)；保留 k 个成分的残差 = ‖x − μ‖² − Σ_{c<k} p_ic²
        const nt = te.n, proj = new Float32Array(nt * D), mse = new Float64Array(D + 1), xc = new Float64Array(D);
        for (let i = 0; i < nt; i++) {
            let r2 = 0; for (let j = 0; j < D; j++) { xc[j] = te.X[i * D + j] - mean[j]; r2 += xc[j] * xc[j]; }
            let acc = r2; mse[0] += r2;
            for (let k = 0; k < D; k++) { let s = 0; const o = k * D; for (let j = 0; j < D; j++) s += vecs[o + j] * xc[j]; proj[i * D + k] = s; acc -= s * s; mse[k + 1] += Math.max(0, acc); }
        }
        for (let k = 0; k <= D; k++) mse[k] /= nt * D;
        return { mean: Float32Array.from(mean), vals, vecs, total, cum, proj, mse };
    }
    /** 用前 k 个主成分重建第 i 张测试图：x̂ = μ + Σ_{c<k} p_ic·u_c */
    function pcaRecon(P, i, k, out) {
        out = out || new Float32Array(D); out.set(P.mean);
        for (let c = 0; c < k; c++) { const p = P.proj[i * D + c], o = c * D; for (let j = 0; j < D; j++) out[j] += p * P.vecs[o + j]; }
        return out;
    }

    // ═══════════════════ 本页自定义的可微算子（AG.track 扩展；有限差分校验见 tools/p09_gradcheck.js） ═══════════════════
    const softplus = (v) => (v > 0 ? v + Math.log1p(Math.exp(-v)) : Math.log1p(Math.exp(v)));
    /** 带 logits 的二元交叉熵：每张图对 256 个像素求和、再对 batch 取平均。ℓ = Σ_j softplus(z_j) − x_j·z_j，∂ℓ/∂z = σ(z) − x。附 .rowLoss（每张图的 ℓ） */
    function bce(Z, X) {
        const n = Z.r, m = Z.c, z = Z.data, x = X.data, Y = AG.tensor(1, 1), row = new Float32Array(n);
        let s = 0;
        for (let i = 0; i < n; i++) { let r = 0; for (let j = i * m, e = j + m; j < e; j++) r += softplus(z[j]) - x[j] * z[j]; row[i] = r; s += r; }
        Y.data[0] = s / n; Y.rowLoss = row;
        return AG.track(Y, [Z], (g) => { const gz = Z.g(), k = g[0] / n; for (let j = 0; j < z.length; j++) gz[j] += k * (1 / (1 + Math.exp(-z[j])) - x[j]); });
    }
    /** VAE 的 KL(q(z|x) ‖ N(0, I)) 闭式解，对 batch 取平均：−½ Σ_d (1 + logσ² − μ² − σ²)。用现成算子拼出来（与 PyTorch 代码逐项对应） */
    function klNormal(mu, logv) {
        const t = AG.sub(AG.sub(AG.addConst(logv, 1), AG.square(mu)), AG.exp(logv));
        return AG.scale(AG.sum(t), -0.5 / mu.r);
    }
    /**
     * 向量量化（VQ-VAE 的瓶颈）。Ze: [M × Dc] 编码器输出（M = 图片数 B × 每张图 16 个位置），E: [K × Dc] 码本。
     * 前向：idx_i = argmin_k ‖ze_i − e_k‖²，zq_i = e_{idx_i}。返回
     *   st：直通估计（straight-through）——前向值 = zq，反向把 ∂L/∂zq 原样交给 ze（量化本身不可导，就当它是恒等）；
     *   cb = (1/B) Σ_i ‖sg(ze_i) − e_{idx_i}‖²（码本损失：只更新码本）；cm = 同一个数，但梯度只给编码器（commitment）。
     */
    function vq(Ze, E, B) {
        const M = Ze.r, Dc = Ze.c, K = E.r, z = Ze.data, e = E.data, idx = new Int32Array(M), dist = new Float32Array(M);
        const st = AG.tensor(M, Dc), q = st.data;
        let tot = 0;
        for (let i = 0; i < M; i++) {
            let best = Infinity, bk = 0; const zo = i * Dc;
            for (let k = 0; k < K; k++) { let s = 0; const eo = k * Dc; for (let c = 0; c < Dc; c++) { const t = z[zo + c] - e[eo + c]; s += t * t; } if (s < best) { best = s; bk = k; } }
            idx[i] = bk; dist[i] = best; tot += best;
            for (let c = 0; c < Dc; c++) q[zo + c] = e[bk * Dc + c];
        }
        AG.track(st, [Ze], (g) => { const gz = Ze.g(); for (let i = 0; i < g.length; i++) gz[i] += g[i]; });
        const cb = AG.tensor(1, 1), cm = AG.tensor(1, 1);
        cb.data[0] = cm.data[0] = tot / B;
        AG.track(cb, [E], (g) => { const ge = E.g(), k = 2 * g[0] / B; for (let i = 0; i < M; i++) { const eo = idx[i] * Dc, zo = i * Dc; for (let c = 0; c < Dc; c++) ge[eo + c] += k * (e[eo + c] - z[zo + c]); } });
        AG.track(cm, [Ze], (g) => { const gz = Ze.g(), k = 2 * g[0] / B; for (let i = 0; i < M; i++) { const eo = idx[i] * Dc, zo = i * Dc; for (let c = 0; c < Dc; c++) gz[zo + c] += k * (z[zo + c] - e[eo + c]); } });
        return { st, idx, cb, cm, dist, ze: Ze };
    }
    // ── VQ-VAE 的卷积：16×16 图 ↔ 4×4 潜网格。按 im2col / col2im 写成矩阵乘（AG.matmul 比通用 AG.conv2d 快得多），与 PyTorch 的 Conv2d / ConvTranspose2d 是同一个计算 ──
    const G = 4, NG = G * G, KP = 8, SP = 4, PP = 2;   // 潜网格 4×4；输入 / 输出卷积核 8×8、步长 4、padding 2
    /** 第 p 个潜位置对应的 8×8 窗口里每个格子 → 16×16 图里的像素下标（−1 = padding） */
    const WIN = (() => { const t = new Int32Array(NG * KP * KP); for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) for (let u = 0; u < KP; u++) for (let v = 0; v < KP; v++) { const y = SP * i - PP + u, x = SP * j - PP + v; t[(i * G + j) * 64 + u * KP + v] = y >= 0 && y < IMG && x >= 0 && x < IMG ? y * IMG + x : -1; } return t; })();
    /** 4×4 网格上 3×3 邻域：第 p 个位置的第 k 个邻居（−1 = padding） */
    const NB3 = (() => { const t = new Int32Array(NG * 9); for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) for (let dy = 0; dy < 3; dy++) for (let dx = 0; dx < 3; dx++) { const y = i + dy - 1, x = j + dx - 1; t[(i * G + j) * 9 + dy * 3 + dx] = y >= 0 && y < G && x >= 0 && x < G ? y * G + x : -1; } return t; })();
    /** 输入图的 im2col（常量，不需要梯度）：n 张图 → [n·16 × 64]，行 = (图, 潜位置)，列 = 8×8 窗口 */
    function patchesOf(X, n, out) {
        out = out || new Float32Array(n * NG * 64);
        for (let b = 0; b < n; b++) for (let p = 0; p < NG; p++) { const o = (b * NG + p) * 64, w = p * 64; for (let k = 0; k < 64; k++) { const s = WIN[w + k]; out[o + k] = s < 0 ? 0 : X[b * D + s]; } }
        return out;
    }
    /** 潜网格上的 3×3 im2col（零填充）：Z [n·16 × C] → [n·16 × 9C]，列顺序 (邻居 k, 通道 c)；反向 = 按同一张表 scatter-add */
    function grid3x3(Z) {
        const n = Z.r / NG, C = Z.c, Y = AG.tensor(Z.r, 9 * C), z = Z.data, y = Y.data;
        for (let b = 0; b < n; b++) for (let p = 0; p < NG; p++) {
            const yo = (b * NG + p) * 9 * C;
            for (let k = 0; k < 9; k++) { const q = NB3[p * 9 + k]; if (q < 0) continue; const zo = (b * NG + q) * C, yk = yo + k * C; for (let c = 0; c < C; c++) y[yk + c] = z[zo + c]; }
        }
        return AG.track(Y, [Z], (g) => {
            const gz = Z.g();
            for (let b = 0; b < n; b++) for (let p = 0; p < NG; p++) {
                const yo = (b * NG + p) * 9 * C;
                for (let k = 0; k < 9; k++) { const q = NB3[p * 9 + k]; if (q < 0) continue; const zo = (b * NG + q) * C, yk = yo + k * C; for (let c = 0; c < C; c++) gz[zo + c] += g[yk + c]; }
            }
        });
    }
    /** 转置卷积的叠加（col2im）：每个潜位置一张 8×8「印章」S [n·16 × 64]，按步长 4、padding 2 叠加回 16×16，再加标量偏置 → [n × 256] */
    function fold(S, bias) {
        const n = S.r / NG, Y = AG.tensor(n, D), s = S.data, y = Y.data;
        y.fill(bias.data[0]);
        for (let b = 0; b < n; b++) for (let p = 0; p < NG; p++) { const so = (b * NG + p) * 64, w = p * 64, yo = b * D; for (let k = 0; k < 64; k++) { const t = WIN[w + k]; if (t >= 0) y[yo + t] += s[so + k]; } }
        return AG.track(Y, [S, bias], (g) => {
            if (S.req) { const gs = S.g(); for (let b = 0; b < n; b++) for (let p = 0; p < NG; p++) { const so = (b * NG + p) * 64, w = p * 64, yo = b * D; for (let k = 0; k < 64; k++) { const t = WIN[w + k]; if (t >= 0) gs[so + k] += g[yo + t]; } } }
            if (bias.req) { let sum = 0; for (let i = 0; i < g.length; i++) sum += g[i]; bias.g()[0] += sum; }
        });
    }

    // ═══════════════════ 模型 ═══════════════════
    // 约定（与 PyTorch 代码一致）：行向量，y = x·W + b，W 的形状 [fan_in × fan_out]；He 初始化、偏置 0。
    const H1 = 64, H2 = 16, ZD = 2;          // MLP 自编码器：256 → 64 → 16 → 2 → 16 → 64 → 256
    const VQC = 16, VQD = 8;                 // VQ-VAE：卷积通道 16，码字维度 8
    function lin(P, name, fi, fo, rand, noBias) {
        const W = AG.param(name + '.W', fi, fo, LM.init.he(rand, fi)); P.push(W);
        let b = null; if (!noBias) { b = AG.param(name + '.b', 1, fo); P.push(b); }
        return { W, b };
    }
    function mlp(P, name, sizes, rand) { const L = []; for (let l = 0; l + 1 < sizes.length; l++) L.push(lin(P, name + l, sizes[l], sizes[l + 1], rand)); return L; }
    function runMLP(L, X) { let h = X; for (let k = 0; k < L.length; k++) { h = AG.linear(h, L[k].W, L[k].b); if (k < L.length - 1) h = AG.relu(h); } return h; }
    /** kind = 'ae' | 'vae'：编码器 256 → 64 → 16 → 2（VAE 输出 μ 和 logσ²，共 4 个数），解码器 2 → 16 → 64 → 256（logits） */
    function makeAE(kind, seed, arch) {
        const rand = LM.rng(seed), P = [], h = arch || [H1, H2];
        const enc = mlp(P, 'enc', [D].concat(h, [kind === 'vae' ? 2 * ZD : ZD]), rand), dec = mlp(P, 'dec', [ZD].concat(h.slice().reverse(), [D]), rand);
        return { kind, P, enc, dec };
    }
    /** VQ-VAE：Conv2d(1, 32, 8, s4, p2) → ReLU → Conv2d(32, 8, 3, p1) → 量化（K 个码字）→ Conv2d(8, 32, 3, p1) → ReLU → ConvTranspose2d(32, 1, 8, s4, p2) */
    function makeVQ(K, seed, arch) {
        const rand = LM.rng(seed), P = [], a = Object.assign({ C: VQC, enc3: true }, arch), C = a.C;
        const e1 = lin(P, 'enc.conv1', 64, C, rand), e2 = lin(P, 'enc.conv2', (a.enc3 ? 9 : 1) * C, VQD, rand);
        const cb = AG.param('codebook', K, VQD, LM.init.uniform(rand, 1 / K)); P.push(cb);
        const d1 = lin(P, 'dec.conv1', 9 * VQD, C, rand), d2 = lin(P, 'dec.convT', C, 64, rand, true);
        const d2b = AG.param('dec.convT.b', 1, 1); P.push(d2b);
        return { kind: 'vq', K, enc3: a.enc3, P, e1, e2, cb, d1, d2, d2b };
    }
    /** 浅拷贝结构 + 深拷贝数据（评估用的权重快照：训练可以继续，评估分几帧做完也看到同一份权重） */
    function snapshot(m, into) {
        if (!into) { into = Object.assign({}, m); into.P = m.P.map((p) => AG.tensor(p.r, p.c)); const map = new Map(m.P.map((p, i) => [p, into.P[i]])); const rep = (o) => (o instanceof AG.Tensor ? map.get(o) : Array.isArray(o) ? o.map(rep) : o && typeof o === 'object' ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, rep(v)])) : o); Object.keys(m).forEach((k) => { if (k !== 'P' && k !== 'kind' && k !== 'K') into[k] = rep(m[k]); }); }
        m.P.forEach((p, i) => into.P[i].data.set(p.data));
        return into;
    }
    // ── 前向 ──
    /** AE / VAE 编码：返回 { z, mu, logv }（AE 的 mu = z、logv = null） */
    function encode(m, X) {
        const h = runMLP(m.enc, X);
        if (m.kind !== 'vae') return { z: h, mu: h, logv: null };
        return { mu: AG.sliceCols(h, 0, ZD), logv: AG.sliceCols(h, ZD, 2 * ZD), h };
    }
    const decode = (m, Z) => runMLP(m.dec, Z);
    /** 重参数化：z = μ + σ ⊙ ε，σ = exp(½·logσ²)，ε ~ N(0, I) 由调用方给出（[n × 2] 的常量张量） */
    const reparam = (mu, logv, eps) => AG.add(mu, AG.mul(AG.exp(AG.scale(logv, 0.5)), eps));
    /** VQ-VAE 前向：Pt = 输入 patch（常量 [n·16 × 64]）。返回 { ze, q（vq 的结果）, logits } */
    function vqForward(m, Pt, n) {
        const h = AG.relu(AG.linear(Pt, m.e1.W, m.e1.b));
        const ze = AG.linear(m.enc3 ? grid3x3(h) : h, m.e2.W, m.e2.b);
        const q = vq(ze, m.cb, n);
        const g = AG.relu(AG.linear(grid3x3(q.st), m.d1.W, m.d1.b));
        const logits = fold(AG.matmul(g, m.d2.W), m.d2b);
        return { ze, q, logits };
    }
    /** 只用码字下标解码（采样 / 码字可视化）：idx [n·16] → logits [n × 256] */
    function vqDecodeIdx(m, idx) {
        const M = idx.length, Dc = VQD, Z = AG.tensor(M, Dc), e = m.cb.data;
        for (let i = 0; i < M; i++) for (let c = 0; c < Dc; c++) Z.data[i * Dc + c] = e[idx[i] * Dc + c];
        const g = AG.relu(AG.linear(grid3x3(Z), m.d1.W, m.d1.b));
        return fold(AG.matmul(g, m.d2.W), m.d2b);
    }
    const sigmoidInto = (L, out) => { const a = L.data; out = out || new Float32Array(a.length); for (let i = 0; i < a.length; i++) out[i] = 1 / (1 + Math.exp(-a[i])); return out; };

    // ═══════════════════ 训练 run：按 epoch 洗牌取 batch → 前向 → 反向 → Adam；评估在权重快照上分帧完成 ═══════════════════
    const BETA_C = 0.25;                 // VQ-VAE 的 commitment 系数（论文取 0.25）
    const RING = 2048;                   // 先验 / 码字画廊用的 token 环形缓冲（最近 2048 张训练图）
    const DEAD_EVERY = 25, DEAD_TH = 0.03; // 死码重启：每 25 步检查一次，码字使用率的滑动平均低于 0.03 次 / batch 视为死码
    /** o: { data, seed, beta, K, restart, cap, B, lr } */
    function makeRun(kind, o) {
        const model = kind === 'vq' ? makeVQ(o.K, o.seed, o.vqArch) : makeAE(kind, o.seed, o.arch), B = o.B;
        const R = {
            kind, o, model, B, cap: o.cap, step: 0, epoch: 0,
            opt: new LM.AdamW(model.P, { lr: o.lr, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0 }),
            rand: LM.rng(o.seed ^ 0x2545f491), noise: LM.rng(o.seed ^ 0x51ed270b),
            perm: Int32Array.from({ length: NTRAIN }, (_, i) => i), pos: 0, bi: new Int32Array(B),
            Xb: AG.tensor(B, D), Pb: kind === 'vq' ? AG.tensor(B * NG, 64) : null, eps: kind === 'vae' ? AG.tensor(B, ZD) : null,
            hist: { rec: [], kl: [], cb: [], perp: [] }, thist: { step: [], rec: [], kl: [], kd0: [], kd1: [], mse: [], elbo: [], used: [], perp: [] },
            usage: kind === 'vq' ? new Float32Array(o.K).fill(NG * B / o.K) : null, restarts: 0, deadNow: 0,
            ring: kind === 'vq' ? { tok: new Int16Array(RING * NG), img: new Int32Array(RING), n: 0, pos: 0 } : null,
            snap: null, ev: null, res: null,
        };
        LM.shuffle(R.perm, R.rand);
        return R;
    }
    function nextBatch(R) {
        const X = R.o.data.tr.X, xb = R.Xb.data;
        for (let k = 0; k < R.B; k++) {
            if (R.pos >= R.perm.length) { LM.shuffle(R.perm, R.rand); R.pos = 0; R.epoch++; }
            const i = R.perm[R.pos++]; R.bi[k] = i;
            xb.set(X.subarray(i * D, (i + 1) * D), k * D);
        }
        if (R.Pb) patchesOf(xb, R.B, R.Pb.data);
        if (R.eps) for (let i = 0; i < R.eps.data.length; i++) R.eps.data[i] = LM.randn(R.noise);
    }
    /** 学习率：前 100 步线性 warmup，之后 cosine 衰减到 lr/20（固定步数的训练，最后几百步用小学习率收尾） */
    const lrAt = (R) => (R.o.sched === 'const' ? R.o.lr : LM.cosineLR(R.step, { maxLR: R.o.lr, minLR: R.o.lr / 20, warmup: 100, total: R.cap }));
    /** 一步训练。返回本 batch 的各项损失（每张图的 nats） */
    function trainStep(R) {
        nextBatch(R);
        const m = R.model, opt = R.opt;
        opt.zeroGrad();
        let rec, kl = null, q = null, loss;
        if (R.kind === 'vq') {
            const o = vqForward(m, R.Pb, R.B); q = o.q;
            rec = bce(o.logits, R.Xb);
            loss = AG.add(rec, AG.add(q.cb, AG.scale(q.cm, BETA_C)));
        } else if (R.kind === 'vae') {
            const e = encode(m, R.Xb), z = reparam(e.mu, e.logv, R.eps);
            rec = bce(decode(m, z), R.Xb); kl = klNormal(e.mu, e.logv);
            loss = AG.add(rec, AG.scale(kl, R.o.beta));
        } else {
            rec = bce(decode(m, encode(m, R.Xb).z), R.Xb);
            loss = rec;
        }
        AG.backward(loss);
        opt.step(lrAt(R));
        R.step++;
        R.hist.rec.push(rec.item());
        if (kl) R.hist.kl.push(kl.item());
        if (q) vqBookkeeping(R, q);
        return loss.item();
    }
    /** 码字使用率（滑动平均，每 batch 的分配次数）、困惑度 exp(H)；可选「死码重启」：把长期没人用的码字挪到本 batch 随机一个编码器输出上 */
    function vqBookkeeping(R, q) {
        const K = R.o.K, cnt = new Float32Array(K), M = q.idx.length;
        for (let i = 0; i < M; i++) cnt[q.idx[i]]++;
        let H = 0; for (let k = 0; k < K; k++) { const p = cnt[k] / M; if (p > 0) H -= p * Math.log(p); R.usage[k] = 0.97 * R.usage[k] + 0.03 * cnt[k]; }
        R.hist.cb.push(q.cb.item()); R.hist.perp.push(Math.exp(H));
        const rg = R.ring;   // 最近 RING 张训练图的 token（Step 10 的先验、码字画廊用）
        for (let b = 0; b < R.B; b++) { rg.img[rg.pos] = R.bi[b]; for (let t = 0; t < NG; t++) rg.tok[rg.pos * NG + t] = q.idx[b * NG + t]; rg.pos = (rg.pos + 1) % RING; rg.n = Math.min(RING, rg.n + 1); }
        let dead = 0; for (let k = 0; k < K; k++) if (R.usage[k] < DEAD_TH) dead++;
        R.deadNow = dead;
        if (!R.o.restart || R.step % DEAD_EVERY !== 0 || !dead) return;
        const cb = R.model.cb, ki = R.model.P.indexOf(cb), mom = R.opt.m[ki], vel = R.opt.v[ki], ze = q.ze.data;
        for (let k = 0; k < K; k++) {
            if (R.usage[k] >= DEAD_TH) continue;
            const i = Math.floor(R.rand() * M);
            for (let c = 0; c < VQD; c++) { cb.data[k * VQD + c] = ze[i * VQD + c] + 0.01 * LM.randn(R.rand); mom[k * VQD + c] = 0; vel[k * VQD + c] = 0; }
            R.usage[k] = 1; R.restarts++;
        }
    }

    // ── 分帧评估：先拷一份权重快照，再每帧处理一小块测试图，处理完一轮（500 张）就发布一组新结果 ──
    let EVAL_EPS = null;   // 测试图的固定噪声 ε（VAE 的重建项用同一组 ε，曲线才不抖）
    function evalBegin(R) {
        R.snap = snapshot(R.model, R.snap);
        const n = NTEST, K = R.o.K;
        R.ev = { step: R.step, pos: 0, rec: 0, kl: 0, mse: 0, z: new Float32Array(n * ZD), lv: R.kind === 'vae' ? new Float32Array(n * ZD) : null, klDim: new Float64Array(ZD),
            tok: R.kind === 'vq' ? new Int16Array(n * NG) : null, cnt: R.kind === 'vq' ? new Float64Array(K) : null, rowRec: new Float32Array(n) };
    }
    function evalWork(R, budget) {
        if (!R.ev) evalBegin(R);
        const ev = R.ev, te = R.o.data.te, m = R.snap, n = Math.min(budget, NTEST - ev.pos);
        if (n > 0) AG.noGrad(() => {
            const X = AG.tensor(n, D, te.X.subarray(ev.pos * D, (ev.pos + n) * D));
            let logits;
            if (R.kind === 'vq') {
                const o = vqForward(m, AG.tensor(n * NG, 64, patchesOf(X.data, n)), n); logits = o.logits;
                for (let i = 0; i < n * NG; i++) { ev.tok[ev.pos * NG + i] = o.q.idx[i]; ev.cnt[o.q.idx[i]]++; }
            } else if (R.kind === 'vae') {
                const e = encode(m, X), eps = AG.tensor(n, ZD, EVAL_EPS.subarray(ev.pos * ZD, (ev.pos + n) * ZD));
                ev.z.set(e.mu.data, ev.pos * ZD); ev.lv.set(e.logv.data, ev.pos * ZD);
                for (let i = 0; i < n * ZD; i++) { const mu = e.mu.data[i], lv = e.logv.data[i], k = -0.5 * (1 + lv - mu * mu - Math.exp(lv)); ev.kl += k; ev.klDim[i % ZD] += k; }
                const L = decode(m, reparam(e.mu, e.logv, eps)), r = bce(L, X); ev.rec += r.item() * n; ev.rowRec.set(r.rowLoss, ev.pos);
                logits = decode(m, e.mu);   // MSE 用后验均值解码（确定性的重建）
                const p = sigmoidInto(logits); for (let i = 0; i < p.length; i++) ev.mse += (p[i] - X.data[i]) ** 2;
                logits = null;
            } else {
                const e = encode(m, X); ev.z.set(e.z.data, ev.pos * ZD); logits = decode(m, e.z);
            }
            if (logits) {
                const r = bce(logits, X); ev.rec += r.item() * n; ev.rowRec.set(r.rowLoss, ev.pos);
                const p = sigmoidInto(logits); for (let i = 0; i < p.length; i++) ev.mse += (p[i] - X.data[i]) ** 2;
            }
        });
        ev.pos += n;
        if (ev.pos < NTEST) return false;
        // 一轮评估完成：发布结果
        const T = R.thist, rec = ev.rec / NTEST, kl = ev.kl / NTEST;
        T.step.push(ev.step); T.rec.push(rec); T.mse.push(ev.mse / (NTEST * D));
        if (R.kind === 'vae') { T.kl.push(kl); T.elbo.push(-(rec + kl)); T.kd0.push(ev.klDim[0] / NTEST); T.kd1.push(ev.klDim[1] / NTEST); }
        let used = 0, H = 0;
        if (R.kind === 'vq') { const tot = NTEST * NG; for (let k = 0; k < R.o.K; k++) if (ev.cnt[k] > 0) { used++; const p = ev.cnt[k] / tot; H -= p * Math.log(p); } T.used.push(used); T.perp.push(Math.exp(H)); }
        R.res = { step: ev.step, z: ev.z, lv: ev.lv, klDim: Array.from(ev.klDim, (v) => v / NTEST), tok: ev.tok, cnt: ev.cnt, rowRec: ev.rowRec, rec, kl, mse: ev.mse / (NTEST * D), used, perp: Math.exp(H) };
        R.ev = null;
        return true;
    }
    function evalNow(R) { R.ev = null; evalBegin(R); evalWork(R, NTEST); }
    /** 每帧：训练 spf 步（到上限就停），再推进 budget 张测试图的评估；训练停止后补一次完整评估 */
    function tick(R, spf, budget) {
        if (R.step >= R.cap) { if (!R.final) { evalNow(R); R.final = true; } return; }
        for (let k = 0; k < spf && R.step < R.cap; k++) trainStep(R);
        evalWork(R, budget);
    }
    function initEvalEps(seed) { const r = LM.rng(seed ^ 0x6b43a9b5); EVAL_EPS = new Float32Array(NTEST * ZD); for (let i = 0; i < EVAL_EPS.length; i++) EVAL_EPS[i] = LM.randn(r); }

    // ═══════════════════ 页面里的训练配置（Node 原型 tools/p09_*.js 用的也是它） ═══════════════════
    const CAP = 3000;
    const CFG = { ae: { B: 32, lr: 5e-3, seed: 11 }, vae: { B: 32, lr: 5e-3, seed: 22 }, vq: { B: 16, lr: 2e-3, seed: 33 } };
    const EVAL = { ae: 25, vae: 25, vq: 10 };          // 每帧评估多少张测试图（一轮 500 张：AE / VAE 20 帧，VQ 50 帧）
    function cfgOf(S, kind) { const c = CFG[kind]; return { data: S.data, seed: S.seed + c.seed, beta: S.beta, K: S.K, restart: S.restart, cap: CAP, B: c.B, lr: c.lr }; }
    const status = (R) => `第 ${R.step} / ${R.cap} 步` + (R.step >= R.cap ? ' · 已停止' : '');
    /** 离线 β 扫描（tools/p09_rd.js：页面同一份代码、默认种子、每个 β 训练 3000 步；beta = 0 那一行是 AE） */
    const RD = [{ beta: 0.1, rate: 6.727, dist: 91.17, mse: 0.0628, klDim: [3.207, 3.52] }, { beta: 0.25, rate: 5.499, dist: 91.36, mse: 0.06257, klDim: [2.61, 2.889] }, { beta: 0.5, rate: 4.592, dist: 91.39, mse: 0.0623, klDim: [2.153, 2.439] }, { beta: 1, rate: 3.601, dist: 91.85, mse: 0.06216, klDim: [1.679, 1.922] }, { beta: 2, rate: 2.559, dist: 93.39, mse: 0.0632, klDim: [1.194, 1.365] }, { beta: 4, rate: 1.011, dist: 99.45, mse: 0.07, klDim: [0.719, 0.292] }, { beta: 8, rate: 0.006, dist: 105.64, mse: 0.08103, klDim: [0.003, 0.003] }, { beta: 16, rate: 0, dist: 105.7, mse: 0.08099, klDim: [0, 0] }, { beta: 0, rate: null, dist: 90.49, mse: 0.06169, klDim: null }];

    // ═══════════════════ 视图：每步预览需要的派生量（都在无梯度下算；只在预览可见时运行） ═══════════════════
    const imgOf = (X, i) => X.subarray(i * D, (i + 1) * D);
    function batchOf(X, idx) { const out = new Float32Array(idx.length * D); idx.forEach((i, k) => out.set(imgOf(X, i), k * D)); return out; }
    /** AE / VAE：一组图的编码（VAE 取 μ、logσ²）与重建（解码 μ） */
    function aeRecon(m, Xb, n) {
        return AG.noGrad(() => { const e = encode(m, AG.tensor(n, D, Xb)); return { mu: e.mu.data.slice(), lv: e.logv ? e.logv.data.slice() : null, img: sigmoidInto(decode(m, e.mu)) }; });
    }
    function decodeZ(m, Z, n) { return AG.noGrad(() => sigmoidInto(decode(m, AG.tensor(n, ZD, Z)))); }
    function vqRecon(m, Xb, n) {
        return AG.noGrad(() => { const o = vqForward(m, AG.tensor(n * NG, 64, patchesOf(Xb, n)), n); return { img: sigmoidInto(o.logits), tok: Int32Array.from(o.q.idx) }; });
    }
    /** 编码的范围：0.5%–99.5% 分位数再外扩 margin（离群点不至于把图压扁） */
    function codeBox(z, n, margin) {
        const xs = new Float32Array(n), ys = new Float32Array(n);
        for (let i = 0; i < n; i++) { xs[i] = z[2 * i]; ys[i] = z[2 * i + 1]; }
        xs.sort(); ys.sort();
        const q = (a, p) => a[Math.min(n - 1, Math.max(0, Math.round(p * (n - 1))))];
        let x0 = q(xs, 0.005), x1 = q(xs, 0.995), y0 = q(ys, 0.005), y1 = q(ys, 0.995);
        const mx = Math.max(1e-3, x1 - x0) * margin, my = Math.max(1e-3, y1 - y0) * margin;
        return { x0: x0 - mx, x1: x1 + mx, y0: y0 - my, y1: y1 + my };
    }
    /** 潜空间网格：n×n 个格子中心解码；同时数每个格子里落了几个测试编码（0 = 洞） */
    function latentGrid(m, box, n, codes, nc) {
        const Z = new Float32Array(n * n * 2);
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { Z[(r * n + c) * 2] = box.x0 + (c + 0.5) / n * (box.x1 - box.x0); Z[(r * n + c) * 2 + 1] = box.y1 - (r + 0.5) / n * (box.y1 - box.y0); }
        const img = decodeZ(m, Z, n * n), cnt = new Int32Array(n * n);
        for (let i = 0; i < nc; i++) {
            const c = Math.floor((codes[2 * i] - box.x0) / (box.x1 - box.x0) * n), r = Math.floor((box.y1 - codes[2 * i + 1]) / (box.y1 - box.y0) * n);
            if (c >= 0 && c < n && r >= 0 && r < n) cnt[r * n + c]++;
        }
        let empty = 0; for (let k = 0; k < n * n; k++) if (!cnt[k]) empty++;
        return { n, box, Z, img, cnt, empty };
    }
    /** 标准正态分布的分位数 Φ⁻¹(p)（Acklam 的有理逼近，相对误差 < 1.2·10⁻⁹） */
    function normInv(p) {
        const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
        const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
        const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
        const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
        const lo = 0.02425;
        if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
        if (p > 1 - lo) return -normInv(1 - p);
        const q = p - 0.5, r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    /** VAE 的「流形图」：在标准正态的分位点上取 n×n 个 z（Kingma & Welling 2014 的画法） */
    function quantileGrid(m, n) {
        const Z = new Float32Array(n * n * 2);
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { Z[(r * n + c) * 2] = normInv((c + 0.5) / n); Z[(r * n + c) * 2 + 1] = normInv(1 - (r + 0.5) / n); }
        return { n, Z, img: decodeZ(m, Z, n * n) };
    }
    /** 沿单一维度移动 z（其余维 = 0）：t ∈ [−2.5, 2.5] 取 n 个点；返回解码与相邻两帧的像素变化 */
    function traversal(m, dim, n) {
        const Z = new Float32Array(n * 2);
        for (let k = 0; k < n; k++) Z[k * 2 + dim] = -2.5 + 5 * k / (n - 1);
        const img = decodeZ(m, Z, n);
        let tv = 0; for (let k = 1; k < n; k++) { let s = 0; for (let j = 0; j < D; j++) s += (img[k * D + j] - img[(k - 1) * D + j]) ** 2; tv += Math.sqrt(s); }
        return { img, tv };
    }
    /** 直线插值：z_t = (1 − t) z_A + t z_B，t = 0, 1/T, …, 1；返回解码与每一步的像素变化 ‖x̂_{t+1} − x̂_t‖ */
    function lerpDecode(m, zA, zB, T) {
        const Z = new Float32Array((T + 1) * 2);
        for (let t = 0; t <= T; t++) for (let d = 0; d < 2; d++) Z[t * 2 + d] = (1 - t / T) * zA[d] + t / T * zB[d];
        const img = decodeZ(m, Z, T + 1), steps = new Float32Array(T);
        for (let t = 0; t < T; t++) { let s = 0; for (let j = 0; j < D; j++) s += (img[(t + 1) * D + j] - img[t * D + j]) ** 2; steps[t] = Math.sqrt(s); }
        return { Z, img, steps };
    }
    function pixelLerp(xA, xB, T) {
        const img = new Float32Array((T + 1) * D), steps = new Float32Array(T);
        for (let t = 0; t <= T; t++) for (let j = 0; j < D; j++) img[t * D + j] = (1 - t / T) * xA[j] + t / T * xB[j];
        for (let t = 0; t < T; t++) { let s = 0; for (let j = 0; j < D; j++) s += (img[(t + 1) * D + j] - img[t * D + j]) ** 2; steps[t] = Math.sqrt(s); }
        return { img, steps };
    }
    // ── 重参数化 vs score function：∂/∂μ₁ E_{q(z|x)}[log p(x|z)] 的单样本估计（每帧 GS 个，凑满 GN 个为一轮；解码器用一轮开始时的快照） ──
    const GN = 256, GS = 32;
    function gradWork(S) {
        const R = S.vae, i = S.sample, key = `${S.epoch}:${S._vaeKey}:${i}`;
        let g = S._grad;
        if (!g || g.key !== key || (g.n >= GN && R.step - g.step >= 100) || (g.n >= GN && R.step >= R.cap && g.step < R.cap)) {
            const m = snapshot(R.model, g && g.m), e = AG.noGrad(() => encode(m, AG.tensor(1, D, imgOf(S.data.te.X, i))));
            g = S._grad = { key, step: R.step, n: 0, m, x: AG.tensor(1, D, imgOf(S.data.te.X, i)), mu: Array.from(e.mu.data), sd: Array.from(e.logv.data, (v) => Math.exp(0.5 * v)),
                pw: new Float32Array(GN), sf: new Float32Array(GN), f: new Float32Array(GN), e0: new Float32Array(GN), rand: LM.rng(S.seed ^ 0x3c6ef372 ^ i) };
        }
        if (g.n >= GN) return g;
        for (let s = 0; s < GS && g.n < GN; s++, g.n++) {
            const e0 = LM.randn(g.rand), e1 = LM.randn(g.rand), Z = AG.param('z', 1, 2);
            Z.data[0] = g.mu[0] + g.sd[0] * e0; Z.data[1] = g.mu[1] + g.sd[1] * e1;
            AG.clearTape();
            const L = bce(decode(g.m, Z), g.x); AG.backward(L);
            const f = -L.item();
            g.pw[g.n] = -Z.grad[0];                         // 重参数化：∂f/∂z₁（∂z₁/∂μ₁ = 1）
            g.f[g.n] = f; g.e0[g.n] = e0;
            g.sf[g.n] = f * e0 / g.sd[0];                   // score function：f(z)·(z₁ − μ₁)/σ₁² = f·ε₁/σ₁
        }
        if (g.n >= GN) {   // 一轮完成：统计（基线 b = f 的均值，对这一轮是常数）
            const fb = LM.mean(g.f), sfb = new Float32Array(GN);
            for (let k = 0; k < GN; k++) sfb[k] = (g.f[k] - fb) * g.e0[k] / g.sd[0];
            const st = (a) => { const m = LM.mean(a); let v = 0; for (let k = 0; k < a.length; k++) v += (a[k] - m) ** 2; return { mean: m, std: Math.sqrt(v / a.length) }; };
            g.sfb = sfb; g.stats = { pw: st(g.pw), sf: st(g.sf), sfb: st(sfb), fbar: fb };
            g.done = { step: g.step };
        }
        return g;
    }
    // ── VQ：码字画廊 = 分配给每个码字的输入 8×8 窗口的平均（最近 RING 张训练图） ──
    function codeGallery(R) {
        const K = R.o.K, sum = new Float64Array(K * 64), cnt = new Float64Array(K), rg = R.ring, X = R.o.data.tr.X;
        for (let s = 0; s < rg.n; s++) {
            const im = rg.img[s] * D;
            for (let p = 0; p < NG; p++) { const k = rg.tok[s * NG + p], w = p * 64, o = k * 64; cnt[k]++; for (let j = 0; j < 64; j++) { const t = WIN[w + j]; if (t >= 0) sum[o + j] += X[im + t]; } }
        }
        const mean = new Float32Array(K * 64);
        for (let k = 0; k < K; k++) if (cnt[k]) for (let j = 0; j < 64; j++) mean[k * 64 + j] = sum[k * 64 + j] / cnt[k];
        return { mean, cnt, n: rg.n };
    }
    // ── token 先验：p(z_t | 上下文)，计数 + 回退平滑。上下文 = 无（每个位置独立）/ 左邻 / 左邻 + 上邻 ──
    const ALPHA = 1;   // 回退平滑的伪计数：没见过的上下文退回到更短的上下文
    function fitPrior(tok, n, K) {
        const uni = new Float64Array(NG * K), uniN = new Float64Array(NG), Lm = new Map(), LUm = new Map();
        const get = (m, key) => { let a = m.get(key); if (!a) { a = new Float64Array(K + 1); m.set(key, a); } return a; };
        for (let i = 0; i < n; i++) for (let t = 0; t < NG; t++) {
            const z = tok[i * NG + t], l = t % G ? tok[i * NG + t - 1] : K, u = t >= G ? tok[i * NG + t - G] : K;   // K = 「没有这个邻居」
            uni[t * K + z]++; uniN[t]++;
            const a = get(Lm, t * (K + 1) + l); a[z]++; a[K]++;
            const b = get(LUm, (t * (K + 1) + l) * (K + 1) + u); b[z]++; b[K]++;
        }
        return { K, n, uni, uniN, L: Lm, LU: LUm };
    }
    function priorDist(P, t, l, u, mode, out) {
        const K = P.K;
        for (let z = 0; z < K; z++) out[z] = (P.uni[t * K + z] + 0.05) / (P.uniN[t] + 0.05 * K);
        if (mode === 0) return out;
        const a = P.L.get(t * (K + 1) + l);
        if (a) for (let z = 0; z < K; z++) out[z] = (a[z] + ALPHA * out[z]) / (a[K] + ALPHA);
        if (mode === 1) return out;
        const b = P.LU.get((t * (K + 1) + l) * (K + 1) + u);
        if (b) for (let z = 0; z < K; z++) out[z] = (b[z] + ALPHA * out[z]) / (b[K] + ALPHA);
        return out;
    }
    /** 按光栅顺序逐个采样 16 个 token（mode 0 = 独立，1 = 左邻，2 = 左邻 + 上邻） */
    function samplePrior(P, mode, rand, n) {
        const K = P.K, tok = new Int32Array(n * NG), p = new Float64Array(K);
        for (let s = 0; s < n; s++) for (let t = 0; t < NG; t++) {
            const l = t % G ? tok[s * NG + t - 1] : K, u = t >= G ? tok[s * NG + t - G] : K;
            priorDist(P, t, l, u, mode, p);
            let r = rand(), z = 0; for (; z < K - 1; z++) { r -= p[z]; if (r <= 0) break; }
            tok[s * NG + t] = z;
        }
        return tok;
    }
    /** 留出的测试图 token 在先验下的平均 −log₂ p（bits / 图） */
    function priorBits(P, tok, n, mode) {
        const K = P.K, p = new Float64Array(K); let s = 0;
        for (let i = 0; i < n; i++) for (let t = 0; t < NG; t++) {
            const l = t % G ? tok[i * NG + t - 1] : K, u = t >= G ? tok[i * NG + t - G] : K;
            priorDist(P, t, l, u, mode, p); s -= Math.log2(Math.max(1e-300, p[tok[i * NG + t]]));
        }
        return s / n;
    }
    // ── AE（对编码拟合 2 维高斯）vs VAE（N(0, I)）的采样：样本到最近真实编码的距离 / 编码之间典型的最近邻距离 ──
    function gaussFit(z, n) {
        let mx = 0, my = 0; for (let i = 0; i < n; i++) { mx += z[2 * i] / n; my += z[2 * i + 1] / n; }
        let cxx = 0, cxy = 0, cyy = 0; for (let i = 0; i < n; i++) { const dx = z[2 * i] - mx, dy = z[2 * i + 1] - my; cxx += dx * dx / n; cxy += dx * dy / n; cyy += dy * dy / n; }
        const l11 = Math.sqrt(cxx), l21 = cxy / l11, l22 = Math.sqrt(Math.max(1e-12, cyy - l21 * l21));   // Cholesky：Σ = L Lᵀ
        return { mx, my, cxx, cxy, cyy, l11, l21, l22 };
    }
    function nnDist(x, y, z, n, skip) { let best = Infinity; for (let j = 0; j < n; j++) { if (j === skip) continue; const d = (x - z[2 * j]) ** 2 + (y - z[2 * j + 1]) ** 2; if (d < best) best = d; } return Math.sqrt(best); }
    function coverage(z, n, g, rand, m) {
        const sp = new Float32Array(n); for (let i = 0; i < n; i++) sp[i] = nnDist(z[2 * i], z[2 * i + 1], z, n, i);
        const sorted = Float32Array.from(sp).sort(), base = sorted[n >> 1] || 1;
        const pts = new Float32Array(m * 2), d = new Float32Array(m);
        for (let s = 0; s < m; s++) {
            const e1 = LM.randn(rand), e2 = LM.randn(rand);
            const x = g ? g.mx + g.l11 * e1 : e1, y = g ? g.my + g.l21 * e1 + g.l22 * e2 : e2;
            pts[2 * s] = x; pts[2 * s + 1] = y; d[s] = nnDist(x, y, z, n, -1) / base;
        }
        let far = 0; for (let s = 0; s < m; s++) if (d[s] > 3) far++;
        return { pts, d, far: far / m, base };
    }
    /** PCA 里与给定 MSE 相当的主成分个数（测试集 MSE 第一次 ≤ m 的 k） */
    function pcaEquiv(P, m) { for (let k = 0; k <= D; k++) if (P.mse[k] <= m) return k; return D; }

    // ═══════════════════ 绘图辅助（深色主题、设计坐标；画布文字 ≥ 11 px） ═══════════════════
    const CLS = ['#4e9ee8', '#f0a050', '#58c870', '#c678dd', '#e85c6a', '#48c4d0', '#f0d050', '#b07c50', '#b8b8c8', '#f07cc0'];   // 数字 0–9 的颜色
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
    const OFFS = new Map();
    /** 把 W×H 的浮点图画到 (x, y, w, h)。o: { map, range, offset, stride } */
    function blit(ctx, data, W, H, x, y, w, h, o) {
        o = o || {};
        const key = W + 'x' + H; let off = OFFS.get(key);
        if (!off) { off = document.createElement('canvas'); off.width = W; off.height = H; OFFS.set(key, off); }
        const octx = off.getContext('2d'), im = octx.createImageData(W, H), map = PL.cmap[o.map || 'gray'], rgb = [0, 0, 0];
        const lo = o.range ? o.range[0] : 0, hi = o.range ? o.range[1] : 1, sp = hi - lo || 1, of = o.offset || 0, st = o.stride || 1;
        for (let i = 0; i < W * H; i++) {
            let t = (data[of + i * st] - lo) / sp; t = t < 0 ? 0 : t > 1 ? 1 : t; map(t, rgb);
            im.data[4 * i] = rgb[0]; im.data[4 * i + 1] = rgb[1]; im.data[4 * i + 2] = rgb[2]; im.data[4 * i + 3] = 255;
        }
        octx.putImageData(im, 0, 0);
        ctx.imageSmoothingEnabled = false; ctx.drawImage(off, x, y, w, h); ctx.imageSmoothingEnabled = true;
    }
    function hatch(ctx, x, y, w, h, color) {
        ctx.fillStyle = '#101216'; ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = color || '#2a2f37'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, y + h); ctx.lineTo(x + w, y); ctx.moveTo(x, y + h / 2); ctx.lineTo(x + w / 2, y); ctx.moveTo(x + w / 2, y + h); ctx.lineTo(x + w, y + h / 2); ctx.stroke();
    }
    /**
     * 若干行小图：rows = [{ label, imgs: [Float32Array | null], outline: [color], subs: [text], subColors, labelColor }]。
     * o: { title, titleColor, tx, labW, gap, top }。返回布局（rowsHit 用）
     */
    function rowsPanel(c, rows, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const top = o.top ?? (o.title ? 22 : 4), labW = o.labW ?? 66, gap = o.gap ?? 3, cols = Math.max(...rows.map((r) => r.imgs.length));
        const subH = rows.map((r) => (r.subs ? 13 : 0));
        const cell = Math.max(4, Math.floor(Math.min((w - labW - 4 - gap * (cols - 1)) / cols, (h - top - 4 - rows.length * gap - subH.reduce((a, b) => a + b, 0)) / rows.length)));
        if (o.title) txt(ctx, o.title, o.tx ?? 6, 4, { bold: true, color: o.titleColor });
        const ys = [];
        let y = top;
        rows.forEach((r, ri) => {
            ys.push(y);
            String(r.label).split('\n').forEach((ln, q, arr) => txt(ctx, ln, labW - 6, y + cell / 2 + (q - (arr.length - 1) / 2) * 13, { align: 'right', base: 'middle', color: r.labelColor || LM.C.gray }));
            r.imgs.forEach((im, k) => {
                const x = labW + k * (cell + gap);
                if (im) blit(ctx, im, IMG, IMG, x, y, cell, cell, { map: r.map, range: r.range });
                else hatch(ctx, x, y, cell, cell);
                if (r.outline && r.outline[k]) { ctx.strokeStyle = r.outline[k]; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, cell + 2, cell + 2); }
                if (r.subs && r.subs[k] != null) txt(ctx, String(r.subs[k]), x + cell / 2, y + cell + 1, { align: 'center', color: (r.subColors && r.subColors[k]) || LM.C.dim });
            });
            y += cell + gap + subH[ri];
        });
        return { labW, ys, cell, gap, cols, n: rows.length, ctx, w, h, bottom: y };
    }
    function rowsHit(L, info) {
        if (!L || !info) return null;
        const k = Math.floor((info.dx - L.labW) / (L.cell + L.gap)); if (k < 0 || k >= L.cols) return null;
        for (let r = 0; r < L.n; r++) if (info.dy >= L.ys[r] && info.dy < L.ys[r] + L.cell) return { r, k };
        return null;
    }
    /** 小图网格：items = [{ img, W, H, map, range, offset, label, labelColor, outline }]（img = null 画斜纹）。o: { title, tx, gap } */
    function tiles(c, items, cols, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const top = o.title ? 20 : 2, lab = items.some((it) => it.label != null) ? 13 : 0, gap = o.gap ?? 3, n = items.length, rows = Math.ceil(n / cols);
        const cell = Math.max(3, Math.floor(Math.min((w - 8 - gap * (cols - 1)) / cols, (h - top - 4 - rows * (gap + lab)) / rows)));
        const x0 = Math.floor((w - (cols * (cell + gap) - gap)) / 2), y0 = top + 2;
        if (o.title) txt(ctx, o.title, o.tx ?? 6, 3, { bold: true, color: o.titleColor });
        items.forEach((it, k) => {
            const x = x0 + (k % cols) * (cell + gap), y = y0 + Math.floor(k / cols) * (cell + gap + lab);
            if (it.img) blit(ctx, it.img, it.W || IMG, it.H || IMG, x, y, cell, cell, { map: it.map, range: it.range, offset: it.offset });
            else hatch(ctx, x, y, cell, cell);
            if (it.outline) { ctx.strokeStyle = it.outline; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, cell + 2, cell + 2); }
            if (it.label != null) txt(ctx, String(it.label), x + cell / 2, y + cell + 1, { align: 'center', color: it.labelColor || LM.C.gray });
        });
        return { x0, y0, cell, gap, lab, cols, n, ctx };
    }
    const tilesHit = (L, info) => { if (!L || !info) return -1; const q = Math.floor((info.dx - L.x0) / (L.cell + L.gap)), r = Math.floor((info.dy - L.y0) / (L.cell + L.gap + L.lab)), k = r * L.cols + q; return q >= 0 && q < L.cols && r >= 0 && k < L.n ? k : -1; };
    /** 二维直方图（潜空间密度）：返回 bg(x, y) → [r, g, b]，给 LM.draw.scatter 的 bg 用 */
    function densityBg(z, n, box, nb, extra) {
        const H = new Float32Array(nb * nb);
        const addPt = (x, y) => { const c = Math.floor((x - box.x0) / (box.x1 - box.x0) * nb), r = Math.floor((box.y1 - y) / (box.y1 - box.y0) * nb); if (c >= 0 && c < nb && r >= 0 && r < nb) H[r * nb + c]++; };
        for (let i = 0; i < n; i++) addPt(z[2 * i], z[2 * i + 1]);
        if (extra) extra(addPt);
        let mx = 0; for (let k = 0; k < H.length; k++) mx = Math.max(mx, H[k]);
        const rgb = [0, 0, 0];
        const bg = (x, y) => {
            const c = Math.min(nb - 1, Math.max(0, Math.floor((x - box.x0) / (box.x1 - box.x0) * nb))), r = Math.min(nb - 1, Math.max(0, Math.floor((box.y1 - y) / (box.y1 - box.y0) * nb)));
            const v = H[r * nb + c]; if (!v) return [12, 14, 18];
            PL.cmap.viridis(0.15 + 0.85 * Math.log1p(v) / Math.log1p(mx), rgb); return [rgb[0] * 0.8, rgb[1] * 0.8, rgb[2] * 0.8];
        };
        return { bg, H, nb, max: mx };
    }
    /** 小号潜空间面板（插值路径用）：在 (x, y, w, h) 里画点 + 路径 */
    function miniLatent(ctx, x, y, w, h, z, ys, n, box, path, title) {
        ctx.fillStyle = '#07090b'; ctx.fillRect(x, y, w, h); ctx.strokeStyle = '#1e242b'; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        const sx = (w - 8) / (box.x1 - box.x0), sy = (h - 22) / (box.y1 - box.y0), s = Math.min(sx, sy), ox = x + (w - s * (box.x1 - box.x0)) / 2, oy = y + 18 + (h - 22 - s * (box.y1 - box.y0)) / 2;
        const X = (v) => ox + (v - box.x0) * s, Y = (v) => oy + (box.y1 - v) * s;
        for (let i = 0; i < n; i++) { ctx.globalAlpha = 0.55; ctx.fillStyle = CLS[ys[i]]; ctx.fillRect(X(z[2 * i]) - 1.2, Y(z[2 * i + 1]) - 1.2, 2.4, 2.4); }
        ctx.globalAlpha = 1;
        if (path) {
            ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.6; ctx.beginPath(); for (let t = 0; t < path.length / 2; t++) { const px = X(path[2 * t]), py = Y(path[2 * t + 1]); if (t) ctx.lineTo(px, py); else ctx.moveTo(px, py); } ctx.stroke();
            for (let t = 0; t < path.length / 2; t++) { ctx.fillStyle = t === 0 ? LM.C.yellow : t === path.length / 2 - 1 ? LM.C.cyan : '#ffffff'; ctx.beginPath(); ctx.arc(X(path[2 * t]), Y(path[2 * t + 1]), t === 0 || t === path.length / 2 - 1 ? 3.5 : 2, 0, 7); ctx.fill(); }
        }
        txt(ctx, title, x + 6, y + 3, { bold: true });
    }
    /** n×n 的解码图网格（潜空间网格 / 流形图）。o: { outline(k) → color, title, tx } */
    function gridPanel(c, img, n, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const top = o.title ? 20 : 2, side = Math.min(w - 4, h - top - 2), cell = side / n, x0 = (w - side) / 2, y0 = top;
        if (o.title) txt(ctx, o.title, o.tx ?? 6, 3, { bold: true, color: o.titleColor });
        for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) {
            const k = r * n + q, x = x0 + q * cell, y = y0 + r * cell;
            blit(ctx, img, IMG, IMG, x, y, cell, cell, { offset: k * D });
            const oc = o.outline && o.outline(k);
            if (oc) { ctx.strokeStyle = oc; ctx.lineWidth = 1.5; ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2); }
        }
        ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.lineWidth = 1;
        for (let k = 0; k <= n; k++) { ctx.beginPath(); ctx.moveTo(x0 + k * cell, y0); ctx.lineTo(x0 + k * cell, y0 + side); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x0, y0 + k * cell); ctx.lineTo(x0 + side, y0 + k * cell); ctx.stroke(); }
        return { x0, y0, cell, n, side, ctx };
    }
    const gridHitP = (L, info) => { if (!L || !info) return -1; const q = Math.floor((info.dx - L.x0) / L.cell), r = Math.floor((info.dy - L.y0) / L.cell); return q >= 0 && q < L.n && r >= 0 && r < L.n ? r * L.n + q : -1; };
    /** 椭圆（数据坐标）：中心 (cx, cy)，半轴 (sx, sy)（与坐标轴对齐，VAE 的对角高斯） */
    function ellipse(cx, cy, sx, sy, k) { const pts = []; for (let t = 0; t <= 40; t++) { const a = t / 40 * 2 * Math.PI; pts.push([cx + k * sx * Math.cos(a), cy + k * sy * Math.sin(a)]); } return pts; }
    /** 一般 2×2 协方差的 k-σ 椭圆（Cholesky 因子 L：x = μ + k·L·(cos, sin)） */
    function ellipseL(g, k) { const pts = []; for (let t = 0; t <= 48; t++) { const a = t / 48 * 2 * Math.PI, c = Math.cos(a), s = Math.sin(a); pts.push([g.mx + k * g.l11 * c, g.my + k * (g.l21 * c + g.l22 * s)]); } return pts; }
    /** 潜空间散点（按数字着色）。o: { z, n, y, box, rings: [{ i, color }], paths, extra（附加点）, title, legend } → 返回 LM.draw.scatter 的变换 */
    function latentPlot(c, o) {
        const pts = [], y = o.y, n = o.n;
        for (let i = 0; i < n; i++) pts.push({ x: o.z[2 * i], y: o.z[2 * i + 1], color: CLS[y[i]], r: o.r || 2.1, alpha: o.alpha ?? 0.85 });
        (o.rings || []).forEach((rg) => pts.push({ x: o.z[2 * rg.i], y: o.z[2 * rg.i + 1], color: CLS[y[rg.i]], r: 3.4, ring: rg.color, label: rg.label, labelColor: rg.color }));
        (o.extra || []).forEach((p) => pts.push(p));
        const tf = LM.draw.scatter(c, pts, { x: [o.box.x0, o.box.x1], y: [o.box.y0, o.box.y1], equal: o.equal !== false, paths: o.paths, lines: o.lines, xLabel: o.xLabel || 'z₁', yLabel: o.yLabel || 'z₂', title: o.title, titleX: o.titleX ?? 46 });
        if (o.legend !== false) {
            const { ctx, w } = PL.draw.ctx(c, { keep: true });
            const x1 = w - 10; let x = x1 - 10 * 13;
            ctx.fillStyle = 'rgba(8,8,10,0.75)'; ctx.fillRect(x - 4, 20, 10 * 13 + 6, 16);
            for (let d = 0; d < NC; d++, x += 13) txt(ctx, String(d), x + 6, 22, { align: 'center', bold: true, color: CLS[d] });
        }
        return tf;
    }
    /** 最近的测试点（probe / pick 用）：tf = latentPlot 的返回值 */
    function nearest(tf, info, z, n) {
        if (!tf || !info) return -1;
        const [x, y] = tf.inv(info.dx, info.dy); let best = Infinity, bi = -1;
        for (let i = 0; i < n; i++) { const d = (z[2 * i] - x) ** 2 + (z[2 * i + 1] - y) ** 2; if (d < best) { best = d; bi = i; } }
        const px = tf.X(z[2 * bi]) - info.dx, py = tf.Y(z[2 * bi + 1]) - info.dy;
        return px * px + py * py < 100 ? bi : -1;
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
    /** 折线图 + 图例。series: [{ x, y, color, dash, width, label }]；o = PL.draw.plot 的选项 + legendAt: [u, v]、legendAlign、extra（附加图例行） */
    function curves(c, series, o) {
        PL.draw.plot(c, Object.assign({}, o, { series: series.filter((s) => s.y && s.y.length).map((s) => ({ data: s.y, x: s.x, color: s.color, dash: s.dash, width: s.width })) }));
        const items = series.filter((s) => s.label).map((s) => ({ label: s.label, color: s.color, dash: s.dash })).concat(o.extra || []);
        if (!items.length) return;
        const { ctx, w, h } = PL.draw.ctx(c, { keep: true });
        const padL = 50, padR = 10, padT = 10, padB = o.xLabel ? 34 : 21, at = o.legendAt || [1, 0], al = o.legendAlign || 'right';
        legend(ctx, items, padL + at[0] * (w - padL - padR) + (al === 'right' ? -4 : 4), padT + at[1] * (h - padT - padB) + 3, al);
    }
    const ema = (a, b) => { const out = new Array(a.length); let m = 0, wsum = 0; for (let i = 0; i < a.length; i++) { m = b * m + (1 - b) * a[i]; wsum = b * wsum + (1 - b); out[i] = m / wsum; } return out; };
    const range = (n, a = 1) => Array.from({ length: n }, (_, i) => a + i);
    /** 4×4 token 网格：每格的颜色 = 码字 id 的色相；size 足够大时写出 id */
    function tokenMap(ctx, tok, off, x, y, size, o) {
        o = o || {};
        const cs = size / G;
        for (let p = 0; p < NG; p++) {
            const k = tok[off + p], X = x + (p % G) * cs, Y = y + Math.floor(p / G) * cs;
            ctx.fillStyle = LM.tokColor(k); ctx.fillRect(X, Y, cs - 1, cs - 1);
            if (cs >= 17) txt(ctx, String(k), X + cs / 2 - 0.5, Y + cs / 2, { align: 'center', base: 'middle', color: '#f0f0f0', px: cs >= 22 ? 12 : 11 });
        }
        if (o.outline) { ctx.strokeStyle = o.outline; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, size + 1, size + 1); }
    }

    return {
        IMG, D, NC, NTRAIN, NTEST, H1, H2, ZD, VQC, VQD, G, NG, BETA_C, DEAD_EVERY, DEAD_TH, RING, CAP, CFG, EVAL, RD, GN, GS, CLS, WIN,
        makeData, eigSym, pca, pcaRecon, bce, klNormal, vq, patchesOf, grid3x3, fold, softplus,
        makeAE, makeVQ, snapshot, encode, decode, reparam, vqForward, vqDecodeIdx, sigmoidInto, runMLP,
        makeRun, nextBatch, trainStep, evalBegin, evalWork, evalNow, tick, initEvalEps, cfgOf, status,
        imgOf, batchOf, aeRecon, decodeZ, vqRecon, codeBox, latentGrid, normInv, quantileGrid, traversal, lerpDecode, pixelLerp, gradWork, codeGallery,
        fitPrior, priorDist, samplePrior, priorBits, gaussFit, nnDist, coverage, pcaEquiv,
        txt, blit, hatch, rowsPanel, rowsHit, tiles, tilesHit, densityBg, miniLatent, gridPanel, gridHitP, ellipse, ellipseL, latentPlot, nearest, legend, curves, ema, range, tokenMap,
    };
})();

PipelineLab.mount({
    paramSteps: { sample: ['pca', 'ae', 'vae', 'reparam', 'interp', 'vq'], sampleB: ['ae', 'interp'], k: ['pca'], beta: ['vae', 'reparam', 'vaeLatent', 'beta', 'interp', 'gen', 'next'], K: ['pca', 'vq', 'gen', 'next'], restart: [], spf: ['ae', 'holes', 'vae', 'reparam', 'vaeLatent', 'beta', 'interp', 'vq', 'gen'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '09 · 潜空间：自编码器、VAE 与 VQ-VAE',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 20251123,
    css: LM.CSS,
    regen: { label: '重新生成数据', title: '用新的随机种子重新生成 4000 + 500 张数字（只重算 RESIDENT），三个模型从头训练' },

    params: [
        { id: 'sample', label: '样本 A #', min: 0, max: 499, step: 1, value: 43, affects: 'frame', desc: '选中的测试图（500 张，与第 02 篇同一个测试集；#43 是一个 4）。Step 01 / 02 / 04 / 05 / 08 / 09 都分析它；也可以直接在潜空间散点上点选。' },
        { id: 'sampleB', label: '样本 B #', min: 0, max: 499, step: 1, value: 6, affects: 'frame', desc: 'Step 08 插值的终点（#6 是一个 7）。换成另一类数字，看潜空间插值会「路过」哪些数字。' },
        { id: 'k', label: 'PCA 成分数 k', min: 1, max: 100, step: 1, value: 10, affects: 'frame', desc: 'Step 01：只保留前 k 个主成分重建。k = 2 只剩模糊的平均数字，k = 50 已经很难看出差别；曲线上的竖线 = 当前 k。' },
        { id: 'beta', label: 'β（KL 权重）', options: [0.1, 0.25, 0.5, 1, 2, 4, 8], value: 1, affects: 'frame', desc: 'VAE 的损失 = 重建 + β·KL（Step 04–08）。β = 1 就是 ELBO；拖到 4 看第 2 维坍缩，拖到 8 看整个后验坍缩成先验（解码器只会画平均数字）。改了只重训 VAE。' },
        { id: 'K', label: '码本大小 K', options: [16, 32, 64, 128, 256], value: 128, affects: 'frame', desc: 'Step 09 / 10：VQ-VAE 码本里有几个码字（每个 token 携带 log₂K bit）。K 越大越容易出现从没被选中过的「死码」。改了只重训 VQ-VAE。' },
        { id: 'restart', label: '死码重启', value: false, affects: 'frame', desc: 'Step 09：每 25 步把长期没人用的码字挪到当前 batch 里随机一个编码器输出上（Jukebox 的 random restart）。打开它，码本使用率从约 1/3 变成接近 100%。改了只重训 VQ-VAE。' },
        { id: 'spf', label: '每帧训练步数', options: [1, 2, 4], value: 1, affects: 'frame', desc: '每个模型每帧训练几步（AE / VAE 的 batch 32，VQ-VAE 的 batch 16，都训练到 3000 步停止）。机器快就调大，更早看到训练结果。' },
        { id: 'retrain', label: '重新训练', action: (S) => { S.retrain = (S.retrain || 0) + 1; }, affects: 'param', desc: '三个模型都从第 0 步重新训练。' },
    ],

    stages: [
        {
            id: 'data', lifetime: 'resident', run(S) {
                S.data = P09.makeData(S.rand);
                S.pca = P09.pca(S.data.tr, S.data.te);
                P09.initEvalEps(S.seed);
                const z = new Float32Array(2 * P09.NTEST); for (let i = 0; i < P09.NTEST; i++) { z[2 * i] = S.pca.proj[i * P09.D]; z[2 * i + 1] = S.pca.proj[i * P09.D + 1]; }
                S.pcaZ = z; S.pcaBox = P09.codeBox(z, P09.NTEST, 0.08);
            },
        },
        {
            id: 'ae', lifetime: 'history', run(S) {   // Step 02：2 维 AE，每帧 spf 步，3000 步后停
                if (S._aeEp !== S.epoch) { S._aeEp = S.epoch; S.ae = P09.makeRun('ae', P09.cfgOf(S, 'ae')); P09.evalNow(S.ae); }
                P09.tick(S.ae, S.spf, P09.EVAL.ae);
            },
        },
        {
            id: 'vae', lifetime: 'history', run(S) {  // Step 04：2 维 VAE；β 变了只重训它
                const key = String(S.beta);
                if (S._vaeEp !== S.epoch || S._vaeKey !== key) { S._vaeEp = S.epoch; S._vaeKey = key; S.vae = P09.makeRun('vae', P09.cfgOf(S, 'vae')); P09.evalNow(S.vae); }
                P09.tick(S.vae, S.spf, P09.EVAL.vae);
            },
        },
        {
            id: 'vq', lifetime: 'history', run(S) {   // Step 09：VQ-VAE；AE / VAE 还在训练时隔帧训练（控制每帧开销）
                const key = S.K + ':' + S.restart;
                if (S._vqEp !== S.epoch || S._vqKey !== key) { S._vqEp = S.epoch; S._vqKey = key; S.vq = P09.makeRun('vq', P09.cfgOf(S, 'vq')); P09.evalNow(S.vq); }
                const busy = S.ae.step < S.ae.cap || S.vae.step < S.vae.cap;
                P09.tick(S.vq, busy && S.frame % 2 ? 0 : S.spf, P09.EVAL.vq);
            },
        },
        {
            id: 'aeView', lifetime: 'frame', lazy: true, run(S) {   // 展示集 + A、B 的 AE 重建与编码；编码范围
                const d = S.data, idx = d.disp.concat([S.sample, S.sampleB]), r = P09.aeRecon(S.ae.model, P09.batchOf(d.te.X, idx), idx.length);
                S.aeV = { img: r.img, z: r.mu, nA: idx.length - 2 };
                if (S._aeBoxStep !== S.ae.res.step || S._aeBoxEp !== S.epoch) { S._aeBoxStep = S.ae.res.step; S._aeBoxEp = S.epoch; S.aeBox = P09.codeBox(S.ae.res.z, P09.NTEST, 0.15); }
            },
        },
        {
            id: 'aeGrid', lifetime: 'frame', lazy: true, needs: ['aeView'], run(S) {   // Step 03：潜空间网格解码（每 4 步重算一次）
                const R = S.ae, key = `${S.epoch}:${R.step >> 2}:${R.res.step}`;
                if (S._aeGridKey === key) return;
                S._aeGridKey = key;
                S.aeGrid = P09.latentGrid(R.model, S.aeBox, 12, R.res.z, P09.NTEST);
                const g = P09.gaussFit(R.res.z, P09.NTEST);
                S.aeGauss = g; S.aeCov = P09.coverage(R.res.z, P09.NTEST, g, LM.rng(S.seed ^ 0x1b873593), 300);
                S.aeDens = P09.densityBg(R.res.z, P09.NTEST, S.aeBox, 28);
            },
        },
        {
            id: 'vaeView', lifetime: 'frame', lazy: true, run(S) {   // Step 04：展示集重建、A 的后验与 8 个后验样本的解码
                const d = S.data, R = S.vae, idx = d.disp.concat([S.sample, S.sampleB]), r = P09.aeRecon(R.model, P09.batchOf(d.te.X, idx), idx.length), n = idx.length;
                const muA = [r.mu[2 * (n - 2)], r.mu[2 * (n - 2) + 1]], sdA = [Math.exp(0.5 * r.lv[2 * (n - 2)]), Math.exp(0.5 * r.lv[2 * (n - 2) + 1])];
                const rnd = LM.rng(S.seed ^ 0x27d4eb2f), Z = new Float32Array(16);
                for (let k = 0; k < 8; k++) { Z[2 * k] = muA[0] + sdA[0] * LM.randn(rnd); Z[2 * k + 1] = muA[1] + sdA[1] * LM.randn(rnd); }
                S.vaeV = { img: r.img, mu: r.mu, lv: r.lv, muA, sdA, Zs: Z, post: P09.decodeZ(R.model, Z, 8) };
                let mx = 3.2; const z = R.res.z; for (let i = 0; i < P09.NTEST; i++) mx = Math.max(mx, Math.abs(z[2 * i]) + 0.2, Math.abs(z[2 * i + 1]) + 0.2);
                S.vaeBox = { x0: -mx, x1: mx, y0: -mx, y1: mx };
            },
        },
        {
            id: 'vaeGrid', lifetime: 'frame', lazy: true, run(S) {   // Step 06：分位点网格、聚合后验、先验样本（每 4 步重算）
                const R = S.vae, key = `${S.epoch}:${S._vaeKey}:${R.step >> 2}:${R.res.step}`;
                if (S._vaeGridKey === key) return;
                S._vaeGridKey = key;
                S.vaeGrid = P09.quantileGrid(R.model, 12);
                const rnd = LM.rng(S.seed ^ 0x165667b1), Z = new Float32Array(40); for (let i = 0; i < 40; i++) Z[i] = LM.randn(rnd);
                S.vaePrior = P09.decodeZ(R.model, Z, 20);
                const z = R.res.z, lv = R.res.lv, r2 = LM.rng(S.seed ^ 0x85ebca6b), box = { x0: -3.6, x1: 3.6, y0: -3.6, y1: 3.6 };
                S.vaeAgg = P09.densityBg(z, 0, box, 36, (add) => { for (let i = 0; i < P09.NTEST; i++) for (let s = 0; s < 4; s++) add(z[2 * i] + Math.exp(0.5 * lv[2 * i]) * LM.randn(r2), z[2 * i + 1] + Math.exp(0.5 * lv[2 * i + 1]) * LM.randn(r2)); });
                let m0 = 0, m1 = 0, v0 = 0, v1 = 0; const N = P09.NTEST;
                for (let i = 0; i < N; i++) { m0 += z[2 * i] / N; m1 += z[2 * i + 1] / N; }
                for (let i = 0; i < N; i++) { v0 += ((z[2 * i] - m0) ** 2 + Math.exp(lv[2 * i])) / N; v1 += ((z[2 * i + 1] - m1) ** 2 + Math.exp(lv[2 * i + 1])) / N; }
                S.vaeAggStats = { m0, m1, v0, v1 };   // 聚合后验的均值与方差（= Var(μ) + E[σ²]）
            },
        },
        {
            id: 'vaeTrav', lifetime: 'frame', lazy: true, run(S) {   // Step 07：沿每一维单独移动 z
                S.trav = [0, 1].map((d) => P09.traversal(S.vae.model, d, 11));
            },
        },
        { id: 'grad', lifetime: 'frame', lazy: true, run(S) { S.grad = P09.gradWork(S); } },   // Step 05：每帧 32 个样本，凑满 256 个一轮
        {
            id: 'interp', lifetime: 'frame', lazy: true, needs: ['aeView', 'vaeView'], run(S) {   // Step 08：A → B 的像素 / AE / VAE 插值
                const d = S.data, n = S.aeV.nA, za = S.aeV.z, zv = S.vaeV.mu, T = 10;
                const ae = P09.lerpDecode(S.ae.model, [za[2 * n], za[2 * n + 1]], [za[2 * n + 2], za[2 * n + 3]], T);
                const vae = P09.lerpDecode(S.vae.model, [zv[2 * n], zv[2 * n + 1]], [zv[2 * n + 2], zv[2 * n + 3]], T);
                S.interp = { T, pix: P09.pixelLerp(P09.imgOf(d.te.X, S.sample), P09.imgOf(d.te.X, S.sampleB), T), ae, vae };
            },
        },
        {
            id: 'vqView', lifetime: 'frame', lazy: true, run(S) {    // Step 09：展示集 + A 的 VQ 重建与 token
                const d = S.data, idx = d.disp.concat([S.sample]);
                S.vqV = Object.assign(P09.vqRecon(S.vq.model, P09.batchOf(d.te.X, idx), idx.length), { n: idx.length });
            },
        },
        {
            id: 'gallery', lifetime: 'frame', lazy: true, run(S) {   // Step 09：每个码字对应的平均输入窗口（每 50 步重算）
                const R = S.vq, key = `${S.epoch}:${S._vqKey}:${Math.floor(R.step / 50)}`;
                if (S._galKey === key) return;
                S._galKey = key; S.gallery = P09.codeGallery(R);
            },
        },
        {
            id: 'prior', lifetime: 'frame', lazy: true, run(S) {     // Step 10：三种模型的「先验 → 解码」采样（每 25 步重算）
                const key = `${S.epoch}:${S._vaeKey}:${S._vqKey}:${Math.floor(S.vq.step / 25)}:${S.ae.res.step}:${S.vae.res.step}`;
                if (S._priorKey === key) return;
                S._priorKey = key;
                const R = S.vq, rg = R.ring, K = R.o.K, N = P09.NTEST, rnd = LM.rng(S.seed ^ 0x9e3779b1);
                const P = P09.fitPrior(rg.tok, rg.n, K);
                const tokI = P09.samplePrior(P, 0, rnd, 10), tokLU = P09.samplePrior(P, 2, rnd, 10);
                const dec = (tok) => LM.AG.noGrad(() => P09.sigmoidInto(P09.vqDecodeIdx(R.model, tok)));   // 码本 token → 解码成图（0..1）
                const g = P09.gaussFit(S.ae.res.z, N), cAE = P09.coverage(S.ae.res.z, N, g, LM.rng(S.seed ^ 0x1b873593), 300), cVAE = P09.coverage(S.vae.res.z, N, null, LM.rng(S.seed ^ 0x1b873593), 300);
                const tt = R.res.tok;
                S.gen = {
                    vqI: dec(tokI), vqLU: dec(tokLU), tokI, tokLU, n: rg.n,
                    ae: P09.decodeZ(S.ae.model, cAE.pts.slice(0, 20), 10), vae: P09.decodeZ(S.vae.model, cVAE.pts.slice(0, 20), 10), cAE, cVAE,
                    bits: [0, 1, 2].map((m) => P09.priorBits(P, tt, N, m)), uniform: P09.NG * Math.log2(K), K,
                };
            },
        },
    ],

    intro: {
        title: '潜空间：把一张图压成几个数，再从这几个数把它画回来',
        flow: ['16×16 数字 = 256 个数', 'PCA：线性压缩', 'AE：2 维瓶颈', 'AE 潜空间的洞', 'VAE：编码成分布 + KL', '重参数化', 'β：率-失真 / 坍缩', '插值', 'VQ-VAE：离散 token', '先验 + 解码器 = 生成'],
        html: LM.navHtml('09') + `
            <p>第 02 篇的 MLP 把 256 个像素压成 10 个类别分数——它只需要记住「是哪个数字」。这一篇反过来问：能不能把一张图压成<strong>很少的几个数</strong>（潜变量 latent，记作 z），再只凭这几个数把整张图<strong>画回来</strong>？能画回来，说明 z 留住了这张图的要点；如果还能从 z 的分布里<strong>随便抽一个</strong>就画出一张像样的新图，压缩器就变成了<strong>生成模型</strong>。</p>
            <p>后面的分化领域几乎都建立在这样的「潜空间」上：第 11 篇的潜空间扩散（Stable Diffusion）在 VAE 的潜空间里加噪去噪；第 12 篇的视频模型先用视频 VAE 把时空体压小；第 13 篇的神经音频 codec 把波形变成 RVQ token；第 07 篇提到的离散图像 token、第 14 篇世界模型的编码器 V，用的都是这一篇的三种模型——<strong>自编码器 AE、变分自编码器 VAE、向量量化自编码器 VQ-VAE</strong>。</p>
            <p><strong>数据</strong>与第 02 篇完全相同：同一个生成器、同一个种子生成的 4000 张训练 + 500 张测试的 16×16 程序化数字（默认选中的测试样本 #43 就是第 02 篇默认的那张 4）。页面里<strong>实时训练</strong>三个模型：2 维 AE、2 维 VAE（batch 32）和 VQ-VAE（batch 16），各训练 3000 步（每帧 1 步；前两个还在训练时 VQ-VAE 隔帧训练；顶栏「每帧训练步数」可以加速）。所有预览都来自这些正在训练的模型，带 <span style="color:#a45fa5">LIVE</span> 的图随训练变化；改 β 只重训 VAE，改码本只重训 VQ-VAE。代码块是同一算法的 C++ 版（本系列自带的 minml 库：二维张量 + 录带式自动微分，见页面底部的 Common 面板）。</p>`,
    },

    steps: [
        // ───────────────────────────── 01 PCA ─────────────────────────────
        {
            id: 'pca', title: '压缩的直觉：PCA 就是最好的线性自编码器', tag: 'LINEAR',
            desc: '一张图是 256 个数，但并不是 256 个数的任意组合都像数字：真实的数字只占这个空间里很薄的一层（所谓「流形」）。最简单的压缩是<strong>找 k 个最重要的方向</strong>：把图投影上去只记 k 个系数（编码），再用这 k 个方向线性组合回来（解码）——这就是<strong>主成分分析 PCA</strong>。这 k 个方向是协方差矩阵最大的 k 个特征向量，第二张图把它们画成 16×16 的「特征数字」。PCA 有一个漂亮的性质：在所有「线性编码 + 线性解码」里，它的均方重建误差最小（Eckart–Young 定理），所以它就是<strong>最优的线性自编码器</strong>；用 MSE 训练一个没有激活函数的自编码器，学到的也正是这 k 维主子空间（Baldi & Hornik 1989，只是基不一定正交）。问题在于：这批数字的变化（粗细、倾斜、写法）是高度非线性的——k = 2 只解释了 15.7% 的方差，重建几乎只剩一团均值；要到 k ≈ 50 才看起来像原图。',
            formula: [
                'Σ = (1/n) Σ_i (x_i − μ)(x_i − μ)^T = U Λ U^T      λ_1 ≥ λ_2 ≥ … ≥ λ_{256} ≥ 0   // 协方差的特征分解',
                'z = U_k^T (x − μ) ∈ R^k      x̂ = μ + U_k z   // 编码 = 投影到前 k 个主成分；解码 = 线性组合回去',
                '(1/n) Σ_i ‖x_i − x̂_i‖^2 = Σ_{j>k} λ_j   // 训练集上的重建误差 = 丢掉的特征值之和；线性方法里没有更小的（Eckart–Young）',
            ],
            code: {
                title: 'C++ · eigSym / fitPCA / pcaEncode / pcaDecode / pcaMSE', src: `
                // 对称矩阵的特征分解：vals = 特征值（从大到小），vecs = 对应的单位特征向量（第 k 行，n × n）
                struct EigenSystem { std::vector<double> vals, vecs; };
                EigenSystem eigSym(const std::vector<double>& A, int n);   // Householder 三对角化 + 隐式 QL（EISPACK 的 tred2 / tql2；实现略，页面 JS 里有完整实现）

                struct PCA { std::vector<float> mean; EigenSystem eig; };

                // μ = 平均图；Σ = (1/n)·Σᵢ (xᵢ − μ)(xᵢ − μ)ᵀ；主成分 = Σ 的特征向量
                PCA fitPCA(const Tensor& X) {
                    int n = X.rows;
                    PCA p;
                    p.mean = floats(D);
                    for (int i = 0; i < n; i++) for (int j = 0; j < D; j++) p.mean[j] += X.v[i * D + j] / n;
                    std::vector<double> C = doubles(D * D);
                    for (int i = 0; i < n; i++)
                        for (int a = 0; a < D; a++) {
                            double xa = X.v[i * D + a] - p.mean[a];
                            for (int b = a; b < D; b++) C[a * D + b] += xa * (X.v[i * D + b] - p.mean[b]) / n;
                        }
                    for (int a = 0; a < D; a++) for (int b = 0; b < a; b++) C[a * D + b] = C[b * D + a];   // 对称
                    p.eig = eigSym(C, D);
                    return p;
                }

                // 编码：z_c = u_cᵀ(x − μ)，c = 0 … k−1
                std::vector<float> pcaEncode(const PCA& p, const float* x, int k) {
                    std::vector<float> z = floats(k);
                    for (int c = 0; c < k; c++) {
                        double s = 0;
                        for (int j = 0; j < D; j++) s += p.eig.vecs[c * D + j] * (x[j] - p.mean[j]);
                        z[c] = (float)s;
                    }
                    return z;
                }

                // 解码：x̂ = μ + Σ_c z_c·u_c（就是一个线性层，权重 = 前 k 个特征向量）
                std::vector<float> pcaDecode(const PCA& p, const std::vector<float>& z) {
                    std::vector<float> x = p.mean;
                    for (int c = 0; c < (int)z.size(); c++)
                        for (int j = 0; j < D; j++) x[j] += z[c] * (float)p.eig.vecs[c * D + j];
                    return x;
                }

                // 测试集上只保留前 k 个主成分时的逐像素 MSE（Step 01 的曲线）
                double pcaMSE(const PCA& p, const Tensor& Xt, int k) {
                    double s = 0;
                    for (int i = 0; i < Xt.rows; i++) {
                        const float* x = &Xt.v[i * D];
                        std::vector<float> xh = pcaDecode(p, pcaEncode(p, x, k));
                        for (int j = 0; j < D; j++) s += (x[j] - xh[j]) * (x[j] - xh[j]);
                    }
                    return s / ((double)Xt.rows * D);
                }`,
            },
            previews: [
                {
                    id: 'pcaRecon', label: 'PCA 重建：每类第一张测试图', sub: '第 2 行 = 前 k 个主成分（顶栏 k）· 第 3 行 = 只用 2 个主成分（和 Step 02 的 2 维自编码器同样的「预算」）', large: true, size: [360, 220],
                    render: (c, S) => {
                        const P = S.pca, d = S.data, idx = d.disp, k = S.k;
                        S._lPca = P09.rowsPanel(c, [
                            { label: '原图', imgs: idx.map((i) => P09.imgOf(d.te.X, i)) },
                            { label: `k = ${k}`, imgs: idx.map((i) => P09.pcaRecon(P, i, k)), labelColor: LM.C.yellow },
                            { label: 'k = 2', imgs: idx.map((i) => P09.pcaRecon(P, i, 2)) },
                        ], { title: `k = ${k}：解释方差 ${(100 * P.cum[k]).toFixed(1)}%，测试 MSE ${P.mse[k].toFixed(4)}` });
                    },
                    probe: (x, y, S, PL, info) => { const h = P09.rowsHit(S._lPca, info); return h ? `测试图 #${S.data.disp[h.k]}（数字 ${h.k}）` : ''; },
                },
                {
                    id: 'pcaEig', label: '平均图 μ 与前 15 个主成分（「特征数字」）', sub: '橙 = 正、蓝 = 负，每张单独归一化 · 数字 = 这个方向解释的方差比例', size: [360, 132],
                    render: (c, S) => {
                        const P = S.pca, items = [{ img: P.mean, label: 'μ', range: [0, 0.6] }];
                        for (let k = 0; k < 15; k++) { const a = P.vecs.subarray(k * 256, (k + 1) * 256); let m = 0; for (let j = 0; j < 256; j++) m = Math.max(m, Math.abs(a[j])); items.push({ img: a, map: 'diverge', range: [-m, m], label: (100 * P.vals[k] / P.total).toFixed(1) }); }
                        P09.tiles(c, items, 8);
                    },
                },
                {
                    id: 'pcaCurve', label: '测试集 MSE vs 主成分个数 k', sub: '蓝 = PCA · 水平线 = 本页三个正在训练的模型（实时）：2 维 AE / VAE 只相当于 PCA 的 3 个左右主成分，VQ-VAE 相当于几十个', live: true, size: [360, 230],
                    render: (c, S) => {
                        const P = S.pca, xs = P09.range(101, 0), ys = xs.map((k) => P.mse[k]), ser = [{ x: xs, y: ys, color: LM.C.blue, width: 2, label: 'PCA（前 k 个主成分）' }];
                        [[S.ae, 'AE 2 维', LM.C.orange], [S.vae, 'VAE 2 维', LM.C.green], [S.vq, 'VQ-VAE 16 token', LM.C.purple]].forEach(([R, name, col]) => { if (R && R.res) ser.push({ x: [0, 100], y: [R.res.mse, R.res.mse], color: col, dash: [5, 4], width: 1.6, label: `${name} ${R.res.mse.toFixed(4)} ≈ k = ${P09.pcaEquiv(P, R.res.mse)}` }); });
                        P09.curves(c, ser, { x: [0, 100], y: [0, 0.085], xLabel: '主成分个数 k', marks: [{ x: S.k, label: `k = ${S.k}`, color: LM.C.yellow }], legendAt: [1, 0.02] });
                    },
                },
                {
                    id: 'pcaScatter', label: '前两个主成分上的投影（500 张测试图）', sub: '颜色 = 数字类别 · 黄圈 = 样本 A · 这就是「2 维 PCA 潜空间」：各类几乎全叠在一起，对比 Step 02 的 AE', size: [360, 280],
                    render: (c, S) => { S._tfPca = P09.latentPlot(c, { z: S.pcaZ, n: P09.NTEST, y: S.data.te.y, box: S.pcaBox, rings: [{ i: S.sample, color: LM.C.yellow }], xLabel: 'PC₁', yLabel: 'PC₂', titleX: 6 }); },
                    probe: (x, y, S, PL, info) => { const i = P09.nearest(S._tfPca, info, S.pcaZ, P09.NTEST); return i < 0 ? '' : `测试图 #${i}（数字 ${S.data.te.y[i]}）· 点击选为样本 A`; },
                    pick: (x, y, S, PL, info) => { const i = P09.nearest(S._tfPca, info, S.pcaZ, P09.NTEST); if (i >= 0) PL.setParam('sample', i); },
                },
                {
                    id: 'pcaProg', label: '样本 A 随 k 增加的重建', sub: 'k 每次翻倍；k = 256 就是原图（正交基，没有任何信息损失）', size: [360, 96],
                    render: (c, S) => { const ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]; P09.tiles(c, ks.map((k) => ({ img: P09.pcaRecon(S.pca, S.sample, k), label: 'k=' + k })).concat([{ img: P09.imgOf(S.data.te.X, S.sample), label: '原图', labelColor: LM.C.yellow }]), 10); },
                },
            ],
            notes: '拖 <b>PCA 成分数 k</b>：k = 1–5 时所有数字都像同一团模糊的「8」——前几个主成分记的是墨水多少、倾斜这种「大」变化，而不是类别；k = 20 能认出数字，k = 50 后只剩细节差异。曲线上的水平线是后面几步正在训练的模型：2 维的 AE / VAE 只比 2 个主成分好一点（大约等于 3 个），VQ-VAE（16 个 token）相当于几十个主成分。',
            warn: 'PCA 的解码是线性的：x̂ 永远是 μ 加 k 张特征图的加权和，k 小时只能得到「模糊的平均数字」。图里的 MSE 是 500 张测试图上的，所以和公式里训练集上的 Σ_{j>k} λ_j 略有差别。页面里 JS 的特征分解是完整的 Householder + QL（256 × 256 的协方差，约 0.1 秒）；C++ 版把这部分写成了原型。',
        },
        // ───────────────────────────── 02 AE ─────────────────────────────
        {
            id: 'ae', title: '自编码器：编码器 → 2 维瓶颈 → 解码器（实时训练）', tag: 'AE',
            desc: '把 PCA 的两个线性映射换成两个 MLP：<strong>编码器</strong> f_φ 把 256 个像素压成 z ∈ R²，<strong>解码器</strong> g_θ 从 z 画回 256 个像素的 logits，sigmoid 之后就是每个像素的灰度。训练目标只有一个：重建得越像越好——这里用逐像素的<strong>二元交叉熵 BCE</strong>（把像素值当作伯努利概率，MNIST 上的经典做法），每张图对 256 个像素求和，单位是 nats / 图。z 只有 2 维，所以每张测试图可以直接画成平面上的一个点（第一张图，颜色 = 数字类别）。2 维是一个极端的瓶颈，模型必须自己决定这 2 个数记什么。在 Node 里用同一份代码验证过：训练 3000 步后测试集逐像素 MSE ≈ 0.062，比 PCA 的 2 个主成分（0.069）好约 10%，大约相当于 3 个主成分；类别也比 PCA 分得开——1、2、7 各自成团——但圆形的 0 / 3 / 5 / 8 / 9 仍然挤在一起（用 2 维编码做 5 近邻分类只有约 38% 的准确率，PCA 是 35%）。原因是这批数字的粗细、倾斜、缩放变化很大，它们对像素误差的影响比「是哪个数字」还大：AE 的一个坐标和墨水总量的相关系数可达 ±0.4–0.5。自编码器只管重建，并不知道我们关心类别。',
            formula: [
                'z = f_φ(x) ∈ R^2      x̂ = σ(g_θ(z)) ∈ (0, 1)^{256}   // f：256 → 64 → 16 → 2，g：2 → 16 → 64 → 256，隐藏层 ReLU',
                'ℓ(x) = Σ_{j=1}^{256} [ softplus(s_j) − x_j s_j ]      s = g_θ(f_φ(x))   // = −Σ_j [x_j log x̂_j + (1 − x_j) log(1 − x̂_j)]，带 logits 的 BCE',
                'Adam（β₁ = 0.9, β₂ = 0.999），lr = 5·10^{−3}：100 步 warmup + cosine 衰减到 lr/20 · batch 32 · 3000 步 = 24 个 epoch',
            ],
            code: {
                title: 'C++ · AE / makeAE / aeTrainStep / trainAE / aeEncode', src: `
                // 自编码器：编码器 256 → 64 → 16 → 2，解码器 2 → 16 → 64 → 256（输出 logits）
                struct AE { MLP enc, dec; };

                AE makeAE(Rng& r) {
                    std::vector<int> e = {D, 64, 16, 2}, d = {2, 16, 64, D};
                    AE m;
                    m.enc = makeMLP(e, r);
                    m.dec = makeMLP(d, r);
                    return m;
                }

                std::vector<Var> aeParams(const AE& m) { std::vector<Var> p; appendParams(p, m.enc); appendParams(p, m.dec); return p; }

                // 一步训练：z = f(x)，s = g(z)，loss = BCE(s, x)（nats / 图）
                float aeTrainStep(AE& m, AdamW& opt, const Tensor& xb, int step, int total) {
                    zeroGrad(opt.params);
                    Var z = mlpForward(m.enc, constant(xb));        // [B × 2]
                    Var logits = mlpForward(m.dec, z);              // [B × 256]
                    Var loss = bceWithLogits(logits, xb);
                    backward(loss);
                    adamwStep(opt, lrAt(step, 5e-3f, total));
                    return loss->val.v[0];
                }

                // 完整训练：batch 32、3000 步。三个随机数流与页面 JS 相同：初始化 / 取 batch / （VAE 的）噪声
                AE trainAE(const DigitSet& train, int steps, uint32_t seed) {
                    Rng init = makeRng(seed), data = makeRng(seed ^ 0x2545f491u);
                    AE m = makeAE(init);
                    AdamW opt = makeAdam(aeParams(m), 5e-3f);
                    BatchStream bs = makeStream(train.X.rows, data);
                    std::vector<int> idx;
                    for (int t = 0; t < steps; t++) aeTrainStep(m, opt, nextBatch(bs, train.X, 32, data, idx), t, steps);
                    return m;
                }

                // 编码一组图（推理，不记录 tape）：返回 [n × 2]
                Tensor aeEncode(const AE& m, const Tensor& X) { NoGrad ng; return mlpForward(m.enc, constant(X))->val; }`,
            },
            previews: [
                {
                    id: 'aeScatter', label: 'AE 的 2 维潜空间：500 张测试图的编码 z = f(x)', sub: '颜色 = 数字类别（右上角图例）· 黄圈 = 样本 A、青圈 = 样本 B · <b>点击</b>选样本 A · 范围随训练自动调整', live: true, large: true, size: [360, 340], needs: ['aeView'],
                    render: (c, S) => { const R = S.ae; S._tfAE = P09.latentPlot(c, { z: R.res.z, n: P09.NTEST, y: S.data.te.y, box: S.aeBox, rings: [{ i: S.sample, color: LM.C.yellow }, { i: S.sampleB, color: LM.C.cyan }], title: `${P09.status(R)} · 编码来自第 ${R.res.step} 步的评估` }); },
                    probe: (x, y, S, PL, info) => { const z = S.ae.res.z, i = P09.nearest(S._tfAE, info, z, P09.NTEST); return i < 0 ? '' : `测试图 #${i}（数字 ${S.data.te.y[i]}）z = (${z[2 * i].toFixed(2)}, ${z[2 * i + 1].toFixed(2)}) · 点击选为样本 A`; },
                    pick: (x, y, S, PL, info) => { const i = P09.nearest(S._tfAE, info, S.ae.res.z, P09.NTEST); if (i >= 0) PL.setParam('sample', i); },
                },
                {
                    id: 'aeRecon', label: '重建：原图 vs AE 的 x̂ = σ(g(f(x)))', sub: '每类第一张测试图 + 样本 A（黄框）· 只靠 2 个数画回来：笔画位置、粗细大致对，细节全丢', live: true, size: [360, 136], needs: ['aeView'],
                    render: (c, S) => {
                        const d = S.data, idx = d.disp.concat([S.sample]), V = S.aeV, out = idx.map((_, k) => (k < 10 ? V.img.subarray(k * 256, (k + 1) * 256) : V.img.subarray(V.nA * 256, (V.nA + 1) * 256)));
                        const ol = idx.map((_, k) => (k === 10 ? LM.C.yellow : null));
                        P09.rowsPanel(c, [{ label: '原图', imgs: idx.map((i) => P09.imgOf(d.te.X, i)), outline: ol }, { label: 'AE 重建', imgs: out, outline: ol, labelColor: LM.C.orange }], { title: P09.status(S.ae), tx: 40, titleColor: LM.C.yellow, labW: 60 });
                    },
                },
                {
                    id: 'aeMSE', label: '测试集逐像素 MSE（AE）', sub: '橙 = AE · 灰虚线 = PCA 取 2 / 3 / 5 个主成分 · 只画到当前步', live: true, size: [360, 200],
                    render: (c, S) => {
                        const T = S.ae.thist, P = S.pca;
                        P09.curves(c, [{ x: T.step, y: T.mse, color: LM.C.orange, width: 2, label: `AE 2 维：${LM.fmt(T.mse[T.mse.length - 1], 4)}` }, { x: [0, P09.CAP], y: [P.mse[2], P.mse[2]], color: LM.C.gray, dash: [4, 4], label: `PCA k=2：${P.mse[2].toFixed(4)}` }, { x: [0, P09.CAP], y: [P.mse[3], P.mse[3]], color: '#6c737c', dash: [2, 3], label: `PCA k=3：${P.mse[3].toFixed(4)}` }, { x: [0, P09.CAP], y: [P.mse[5], P.mse[5]], color: '#4f555c', dash: [2, 3], label: `PCA k=5：${P.mse[5].toFixed(4)}` }],
                            { x: [0, P09.CAP], y: [0.05, 0.085], xLabel: '训练步数', legendAt: [1, 0.02] });
                    },
                },
                {
                    id: 'aeLoss', label: 'BCE 损失（nats / 图）', sub: '蓝 = 训练 batch（滑动平均）· 橙 = 测试集', live: true, size: [360, 170],
                    render: (c, S) => { const R = S.ae, tr = P09.ema(R.hist.rec, 0.97); P09.curves(c, [{ x: P09.range(tr.length), y: tr, color: LM.C.train, width: 1.5, label: '训练' }, { x: R.thist.step, y: R.thist.rec, color: LM.C.val, width: 2, label: `测试 ${LM.fmt(R.thist.rec[R.thist.rec.length - 1], 2)}` }], { x: [0, P09.CAP], y: [85, 125], xLabel: '训练步数' }); },
                },
            ],
            notes: '训练开始时所有点挤在原点附近，几百步后各类慢慢分开：先分出来的是「笔画最不一样」的 1、7、2（细长的竖线、横线），圆形的数字最后也没完全分开。在散点上点一个点，它就成为样本 A（Step 01 / 04 / 05 / 08 / 09 都会跟着换）。MSE 曲线大约在 1000 步越过 PCA 的 2 个主成分。',
            warn: 'BCE 把灰度值当成伯努利分布的概率，这对 0 到 1 之间的灰度并不是严格的似然（Loaiza-Ganem & Cunningham 2019 的 continuous Bernoulli 讨论了这一点），但它是 MNIST 类数据上最常用的重建损失；换成 MSE 等价于假设固定方差的高斯噪声。另外 BCE 有一个下限：灰度在 0 和 1 之间的像素（抗锯齿边缘、笔触纹理、噪点）即使预测完全正确也有 > 0 的损失，所以曲线不会降到 0。2 维只是为了能画出来：实际系统的潜变量有成百上千维。',
        },
        // ───────────────────────────── 03 洞 ─────────────────────────────
        {
            id: 'holes', title: 'AE 潜空间的「洞」：编码之间的地方没人管', tag: 'AE',
            desc: '自编码器只在训练图的编码 z 上被要求「解码得像」，编码之间、编码之外的地方解码器画什么，<strong>没有任何损失项约束</strong>。把潜空间铺成 12 × 12 的网格、在每个格子中心解码（第一张图）：数据密集的地方解出清楚的数字，格子之间是两类数字的混合；红框是<strong>没有任何测试编码落进去</strong>的格子——解码器在那里画的东西是外推出来的。编码的分布也很不规则（第二张图的背景 = 编码的密度）：一团密集的核心加几条稀疏的「臂」，和任何简单的分布都不像。所以 AE 是一个不错的压缩器，却不是生成器——你不知道该从哪里抽 z。最朴素的补救是事后给编码拟合一个二维高斯再从里面抽（第二张图的椭圆和点）：抽到的点有相当一部分落在洞里，离最近的真实编码超过 3 倍典型间距（红点）。Step 10 会和 VAE 定量比较。',
            formula: [
                'L_AE = Σ_i ℓ(x_i, g(f(x_i)))   // 只约束 g 在 {f(x_i)} 这些点上的取值；其余的 z 上 g(z) 是什么都不影响损失',
                '格子中心 z_{rc} = (x_0 + (c + ½)Δx, y_1 − (r + ½)Δy)   // 范围 = 测试编码 0.5%–99.5% 分位数再外扩 15%',
                '拟合高斯：m = mean(z_i)，Σ = cov(z_i) = L L^T，z = m + L ε   // 「距离 / 典型间距」= 到最近测试编码的距离 ÷ 编码之间最近邻距离的中位数',
            ],
            code: {
                title: 'C++ · decodeGrid / cellCounts', src: `
                // 在 [x0, x1] × [y0, y1] 上铺 n×n 个格子，解码每个格子的中心（第 r 行从上往下，所以 y 从大到小）
                Tensor decodeGrid(const MLP& dec, float x0, float x1, float y0, float y1, int n) {
                    NoGrad ng;
                    Tensor Z = zeros(n * n, 2);
                    for (int r = 0; r < n; r++)
                        for (int c = 0; c < n; c++) {
                            Z.at(r * n + c, 0) = x0 + (c + 0.5f) / n * (x1 - x0);
                            Z.at(r * n + c, 1) = y1 - (r + 0.5f) / n * (y1 - y0);
                        }
                    return sigmoid(mlpForward(dec, constant(Z)))->val;     // [n² × 256]：每行一张解码图
                }

                // 每个格子里落了几个测试编码（0 = 「洞」：解码器在那里从没被要求画对任何东西）
                std::vector<int> cellCounts(const Tensor& codes, float x0, float x1, float y0, float y1, int n) {
                    std::vector<int> cnt = ints(n * n);
                    for (int i = 0; i < codes.rows; i++) {
                        int c = (int)std::floor((codes.at(i, 0) - x0) / (x1 - x0) * n);
                        int r = (int)std::floor((y1 - codes.at(i, 1)) / (y1 - y0) * n);
                        if (c >= 0 && c < n && r >= 0 && r < n) cnt[r * n + c]++;
                    }
                    return cnt;
                }`,
            },
            previews: [
                {
                    id: 'aeGrid', label: '在 AE 潜空间的 12 × 12 网格上解码', sub: '范围与 Step 02 的散点相同 · 红框 = 格子里没有任何测试编码（「洞」）· 每 4 步重画一次', live: true, large: true, size: [360, 380], needs: ['aeGrid'],
                    render: (c, S) => { const G = S.aeGrid; S._lAeGrid = P09.gridPanel(c, G.img, G.n, { outline: (k) => (G.cnt[k] ? null : 'rgba(232,90,100,0.85)'), title: `第 ${S.ae.step} 步 · ${G.empty} / ${G.n * G.n} 个格子是空的`, tx: 40, titleColor: LM.C.yellow }); },
                    probe: (x, y, S, PL, info) => { const k = P09.gridHitP(S._lAeGrid, info), G = S.aeGrid; return k < 0 ? '' : `z = (${G.Z[2 * k].toFixed(2)}, ${G.Z[2 * k + 1].toFixed(2)}) · 格子里有 ${G.cnt[k]} 个测试编码`; },
                },
                {
                    id: 'aeDensity', label: '编码的密度 + 事后拟合的高斯', sub: '背景 = 测试编码的二维直方图（对数色阶，暗 = 没有编码）· 白线 = 拟合高斯的 1σ / 2σ 椭圆 · 点 = 从这个高斯抽的 60 个 z：红 = 离最近编码 > 3 倍典型间距', live: true, size: [360, 320], needs: ['aeGrid'],
                    render: (c, S) => {
                        const g = S.aeGauss, cv = S.aeCov, pts = [];
                        for (let s = 0; s < 60; s++) pts.push({ x: cv.pts[2 * s], y: cv.pts[2 * s + 1], color: cv.d[s] > 3 ? LM.C.red : '#e8e8e8', r: 2.4 });
                        LM.draw.scatter(c, pts, { x: [S.aeBox.x0, S.aeBox.x1], y: [S.aeBox.y0, S.aeBox.y1], equal: true, bg: S.aeDens.bg, bgRes: 56, paths: [{ pts: P09.ellipseL(g, 1), color: 'rgba(255,255,255,0.8)', width: 1.2 }, { pts: P09.ellipseL(g, 2), color: 'rgba(255,255,255,0.6)', width: 1.2, dash: [4, 3] }], xLabel: 'z₁', yLabel: 'z₂', title: `红点占 ${(100 * cv.far).toFixed(0)}%（300 个样本里）`, titleX: 46 });
                    },
                },
            ],
            notes: '训练早期编码挤在一小团里，几乎没有洞；随着训练，各类被推开，空格子越来越多。把网格里红框格子和相邻的非空格子比一比：空格子里多半是两类数字叠在一起的「四不像」，或者墨水明显过多 / 过少。第二张图里，高斯的椭圆把核心周围大片没有数据的区域也包了进去。',
            warn: '2 维时问题看起来还不严重：解码器是光滑函数，洞里也能「插值」出像样的图，而且 12 × 12 的格子很粗。维度越高，数据在潜空间里越稀疏，随机取一个 z 落进洞里的概率越大——这正是 VAE 要解决的问题。',
        },
        // ───────────────────────────── 04 VAE ─────────────────────────────
        {
            id: 'vae', title: 'VAE：编码器输出一个分布，损失 = 重建 + KL', tag: 'VAE',
            desc: '变分自编码器（VAE，Kingma & Welling 2014；Rezende et al. 2014 同时独立提出）做了两处改动：① 编码器不再输出一个点，而是一个高斯分布 q(z|x) = N(μ, diag σ²)——网络输出 μ 和 log σ²（输出对数方差，σ 自然为正、数值也稳定）；训练时从这个分布里<strong>抽一个 z</strong> 再解码，所以解码器必须对 z 附近一小团都能画回 x，潜空间里不能再有「只在一个点上成立」的编码。② 损失里加一项 <strong>KL(q(z|x) ‖ N(0, I))</strong>，把每个 q 往标准正态拉：μ 不能跑太远，σ 不能缩成 0。两项合起来正好是<strong>证据下界 ELBO</strong> 的相反数：log p(x) ≥ E_q[log p(x|z)] − KL，高斯之间的 KL 有闭式解，不需要采样。第一张图里每个点是一张测试图的 μ，淡色椭圆是部分点的 ±1σ；黄色是样本 A 的后验 q(z|x_A)（±1σ、±2σ）和从中抽出的 8 个 z，第二张图是它们的解码——同一张图在模型眼里「可能的样子」。β = 1 时两维的 KL 加起来只有约 3.6 nats（≈ 5.2 bit）：编码器只肯为每张图付这么多信息。',
            formula: [
                'q_φ(z|x) = N(μ_φ(x), diag σ_φ^2(x))      [μ, log σ^2] = f_φ(x) ∈ R^{2+2}      p(z) = N(0, I)',
                'log p_θ(x) = ELBO + KL(q_φ(z|x) ‖ p_θ(z|x)) ≥ ELBO = E_{q(z|x)}[log p_θ(x|z)] − KL(q_φ(z|x) ‖ p(z))',
                'KL(q ‖ N(0, I)) = −½ Σ_{d=1}^{2} (1 + log σ_d^2 − μ_d^2 − σ_d^2)   // 闭式解；μ = 0、σ = 1 时为 0',
                'L(x) = BCE(x, σ(g_θ(z))) + β · KL      z = μ + σ ⊙ ε，ε ~ N(0, I)   // β = 1 时就是 −ELBO（单样本估计）；β 见 Step 07',
            ],
            code: {
                title: 'C++ · VAE / reparameterize / klNormal / vaeForward / vaeTrainStep / trainVAE', src: `
                // VAE：编码器 256 → 64 → 16 → 4，输出 [μ₁, μ₂, log σ₁², log σ₂²]；解码器与 AE 相同
                struct VAE { MLP enc, dec; };

                VAE makeVAE(Rng& r) {
                    std::vector<int> e = {D, 64, 16, 4}, d = {2, 16, 64, D};
                    VAE m;
                    m.enc = makeMLP(e, r);
                    m.dec = makeMLP(d, r);
                    return m;
                }

                std::vector<Var> vaeParams(const VAE& m) { std::vector<Var> p; appendParams(p, m.enc); appendParams(p, m.dec); return p; }

                // 重参数化：z = μ + σ ⊙ ε，σ = exp(½·log σ²)。ε 由调用方抽好传进来——随机性与参数无关，梯度能穿过 μ 和 σ（Step 05）
                Var reparameterize(const Var& mu, const Var& logvar, const Tensor& eps) {
                    return add(mu, mul(expAct(scale(logvar, 0.5f)), constant(eps)));
                }

                // KL(N(μ, σ²) ‖ N(0, I)) 的闭式解，对 batch 取平均：−½ Σ_d (1 + log σ_d² − μ_d² − σ_d²)
                Var klNormal(const Var& mu, const Var& logvar) {
                    Var t = sub(sub(addConst(logvar, 1.f), square(mu)), expAct(logvar));
                    return scale(sumAll(t), -0.5f / mu->val.rows);
                }

                struct VAEOut { Var mu, logvar, z, logits, rec, kl, loss; };

                // 前向：编码 → 采样 → 解码 → 损失 = BCE + β·KL
                VAEOut vaeForward(const VAE& m, const Tensor& xb, const Tensor& eps, float beta) {
                    VAEOut o;
                    Var h = mlpForward(m.enc, constant(xb));        // [B × 4]
                    o.mu = sliceCols(h, 0, 2);
                    o.logvar = sliceCols(h, 2, 4);
                    o.z = reparameterize(o.mu, o.logvar, eps);
                    o.logits = mlpForward(m.dec, o.z);
                    o.rec = bceWithLogits(o.logits, xb);            // 重建项：−E_q[log p(x|z)] 的单样本估计
                    o.kl = klNormal(o.mu, o.logvar);
                    o.loss = add(o.rec, scale(o.kl, beta));
                    return o;
                }

                float vaeTrainStep(VAE& m, AdamW& opt, const Tensor& xb, float beta, Rng& noise, int step, int total) {
                    zeroGrad(opt.params);
                    Tensor eps = randn(xb.rows, 2, 1.f, noise);     // 每张图一个 ε ~ N(0, I)
                    VAEOut o = vaeForward(m, xb, eps, beta);
                    backward(o.loss);
                    adamwStep(opt, lrAt(step, 5e-3f, total));
                    return o.loss->val.v[0];
                }

                VAE trainVAE(const DigitSet& train, float beta, int steps, uint32_t seed) {
                    Rng init = makeRng(seed), data = makeRng(seed ^ 0x2545f491u), noise = makeRng(seed ^ 0x51ed270bu);
                    VAE m = makeVAE(init);
                    AdamW opt = makeAdam(vaeParams(m), 5e-3f);
                    BatchStream bs = makeStream(train.X.rows, data);
                    std::vector<int> idx;
                    for (int t = 0; t < steps; t++) vaeTrainStep(m, opt, nextBatch(bs, train.X, 32, data, idx), beta, noise, t, steps);
                    return m;
                }`,
            },
            previews: [
                {
                    id: 'vaeScatter', label: 'VAE 的潜空间：500 张测试图的后验均值 μ', sub: '淡色椭圆 = 每 5 张图画一个 ±1σ · 黄 = 样本 A 的 q(z|x_A)（±1σ、±2σ）和 8 个抽样 · 灰虚线 = 先验 N(0, I) 的 1σ、2σ 圆 · <b>点击</b>选样本 A', live: true, large: true, size: [360, 340], needs: ['vaeView'],
                    render: (c, S) => {
                        const R = S.vae, z = R.res.z, lv = R.res.lv, V = S.vaeV, paths = [];
                        for (let i = 0; i < P09.NTEST; i += 5) paths.push({ pts: P09.ellipse(z[2 * i], z[2 * i + 1], Math.exp(0.5 * lv[2 * i]), Math.exp(0.5 * lv[2 * i + 1]), 1), color: P09.CLS[S.data.te.y[i]], width: 1, alpha: 0.35 });
                        paths.push({ pts: P09.ellipse(0, 0, 1, 1, 1), color: '#777', dash: [3, 4], width: 1 }, { pts: P09.ellipse(0, 0, 1, 1, 2), color: '#555', dash: [3, 4], width: 1 });
                        paths.push({ pts: P09.ellipse(V.muA[0], V.muA[1], V.sdA[0], V.sdA[1], 1), color: LM.C.yellow, width: 2 }, { pts: P09.ellipse(V.muA[0], V.muA[1], V.sdA[0], V.sdA[1], 2), color: LM.C.yellow, width: 1.2, dash: [4, 3] });
                        const extra = []; for (let k = 0; k < 8; k++) extra.push({ x: V.Zs[2 * k], y: V.Zs[2 * k + 1], color: LM.C.yellow, r: 2.2 });
                        S._tfVAE = P09.latentPlot(c, { z, n: P09.NTEST, y: S.data.te.y, box: S.vaeBox, paths, extra, rings: [{ i: S.sample, color: LM.C.yellow }], title: `${P09.status(R)} · β = ${S.beta}` });
                    },
                    probe: (x, y, S, PL, info) => { const R = S.vae, z = R.res.z, i = P09.nearest(S._tfVAE, info, z, P09.NTEST); return i < 0 ? '' : `测试图 #${i}（数字 ${S.data.te.y[i]}）μ = (${z[2 * i].toFixed(2)}, ${z[2 * i + 1].toFixed(2)})  σ = (${Math.exp(0.5 * R.res.lv[2 * i]).toFixed(2)}, ${Math.exp(0.5 * R.res.lv[2 * i + 1]).toFixed(2)})`; },
                    pick: (x, y, S, PL, info) => { const i = P09.nearest(S._tfVAE, info, S.vae.res.z, P09.NTEST); if (i >= 0) PL.setParam('sample', i); },
                },
                {
                    id: 'vaePost', label: '样本 A 的后验：原图、解码 μ、以及 8 个 z ~ q(z|x_A) 的解码', sub: '同一张图在潜空间里是一小团，不是一个点：这一团里的 z 都要能画回「差不多的 x」', live: true, size: [360, 104], needs: ['vaeView'],
                    render: (c, S) => {
                        const V = S.vaeV, n = 10, imgs = [P09.imgOf(S.data.te.X, S.sample), V.img.subarray(n * 256, (n + 1) * 256)];
                        for (let k = 0; k < 8; k++) imgs.push(V.post.subarray(k * 256, (k + 1) * 256));
                        P09.rowsPanel(c, [{ label: '', imgs, subs: ['原图', 'μ'].concat(P09.range(8).map((k) => 'z' + k)), subColors: [LM.C.yellow, LM.C.green] }], { labW: 8, title: `σ = (${V.sdA[0].toFixed(3)}, ${V.sdA[1].toFixed(3)})`, tx: 40, titleColor: LM.C.yellow });
                    },
                },
                {
                    id: 'vaeRec', label: '重建项 −E_q[log p(x|z)]（nats / 图）', sub: '蓝 = 训练 batch（滑动平均）· 橙 = 测试集（每张图抽一个 z）· 灰虚线 = AE 的最终值（Step 10 的离线训练）', live: true, size: [360, 170],
                    render: (c, S) => { const R = S.vae, tr = P09.ema(R.hist.rec, 0.97), ae = P09.RD[P09.RD.length - 1].dist; P09.curves(c, [{ x: P09.range(tr.length), y: tr, color: LM.C.train, width: 1.5, label: '训练' }, { x: R.thist.step, y: R.thist.rec, color: LM.C.val, width: 2, label: `测试 ${LM.fmt(R.thist.rec[R.thist.rec.length - 1], 2)}` }, { x: [0, P09.CAP], y: [ae, ae], color: LM.C.gray, dash: [4, 4], label: `AE ${ae}` }], { x: [0, P09.CAP], y: [85, 125], xLabel: '训练步数' }); },
                },
                {
                    id: 'vaeKL', label: 'KL 项（nats / 图）与每一维的 KL', sub: '紫 = 训练 batch · 绿 = 测试集总 KL · 细线 = 测试集上第 1 / 2 维各自的 KL', live: true, size: [360, 170],
                    render: (c, S) => { const R = S.vae, tr = P09.ema(R.hist.kl, 0.97), T = R.thist; P09.curves(c, [{ x: P09.range(tr.length), y: tr, color: LM.C.purple, width: 1.5, label: '训练' }, { x: T.step, y: T.kl, color: LM.C.green, width: 2, label: `测试 ${LM.fmt(T.kl[T.kl.length - 1], 2)}` }, { x: T.step, y: T.kd0, color: LM.C.cyan, width: 1, label: `z₁ ${LM.fmt(T.kd0[T.kd0.length - 1], 2)}` }, { x: T.step, y: T.kd1, color: LM.C.yellow, width: 1, label: `z₂ ${LM.fmt(T.kd1[T.kd1.length - 1], 2)}` }], { x: [0, P09.CAP], y: [0, 8], xLabel: '训练步数', legendAt: [1, 0.02] }); },
                },
            ],
            notes: '刚开始 KL 很快上升：编码器发现「付一点信息量」能大幅降低重建误差；之后两项此消彼长。点不同的样本 A：写得规整的数字 σ 小（模型很确定它在哪），写歪的、模棱两可的数字 σ 更大，8 个抽样的解码之间差异也更大。把 β 调到 4 或 8 再回来看这一步（训练要从头再来，约 50 秒）。',
            warn: '一个常见的坑：网络直接输出 σ 而不是 log σ²，σ 可能变负或变成 0，KL 里的 log σ² 就会爆掉。另一个坑是损失的归一化：这里重建项对 256 个像素<strong>求和</strong>、KL 对 2 维求和（都是「每张图」的 nats），两者才在同一个尺度上；如果重建项用逐像素平均（如 F.mse_loss 的默认 reduction=\'mean\'），等于把 β 放大了 256 倍，很容易一开始就后验坍缩（Step 07）。',
        },
        // ───────────────────────────── 05 重参数化 ─────────────────────────────
        {
            id: 'reparam', title: '重参数化技巧：让梯度穿过「采样」', tag: 'GRAD',
            desc: '训练要对 φ（编码器）求 E_{q_φ(z|x)}[f(z)] 的梯度，但期望所在的分布本身依赖 φ，「抽一个样本」这个操作没有导数。两种办法：① <strong>score function</strong> 估计（也叫似然比 / REINFORCE，Williams 1992）：∇_μ E[f] = E[f(z)·∇_μ log q(z)]——只需要 f 的<strong>值</strong>，f 不可导也行；② <strong>重参数化</strong>（reparameterization trick）：把随机性挪到一个与参数无关的 ε ~ N(0, I) 上，z = μ + σε，于是 ∇_μ E[f(μ + σε)] = E[∂f/∂z]——梯度直接穿过解码器，自动微分就能算（上一步 C++ 代码里的 reparameterize）。两者都是无偏的，但方差差得很远。下面用<strong>正在训练的 VAE 解码器</strong>和样本 A 的真实后验实测：对 ∂/∂μ₁ E_q[log p(x_A|z)] 各抽 256 个单样本估计。score function 的标准差是重参数化的几十倍（方差约几千倍，Node 里在 3 张图上测到 2800–12500 倍）；减去一个基线（f 的平均值）能把方差降到只比重参数化大几倍——但在这里潜变量只有 2 维、σ 很小，维度一高差距会迅速拉大。第 10 篇的策略梯度（REINFORCE）用的就是 score function：动作是离散的、环境不可导，只能这么做，也因此特别依赖基线。',
            formula: [
                '∇_μ E_{z~N(μ, σ²)}[f(z)] = E[ f(z) · (z − μ)/σ^2 ]   // score function：∇_μ log q(z) = (z − μ)/σ²，只用 f 的值',
                '                        = E_{ε~N(0,1)}[ f′(μ + σ ε) ]   // 重参数化：z = μ + σε，梯度穿过 f',
                'E[ (f(z) − b) · (z − μ)/σ^2 ] = 同一个期望（因为 E[(z − μ)/σ²] = 0）   // 基线 b 不改变期望，只降低方差',
                'f(z) = log p(x|z) = −BCE(x, σ(g(z)))   // 这里求 ∂/∂μ₁；单样本估计：pathwise = ∂f/∂z₁，score = f·ε₁/σ₁',
            ],
            code: {
                title: 'C++ · gradSample / meanStd / compareEstimators', src: `
                // ∂/∂μ₁ E_{z~q(z|x)}[log p(x|z)] 的两种单样本估计。q = N(μ, diag σ²)，ε = (e0, e1) ~ N(0, I) 已经抽好
                struct GradSample { float f, pathwise, score; };

                GradSample gradSample(const VAE& m, const Tensor& x, const std::vector<float>& mu, const std::vector<float>& sd, float e0, float e1) {
                    Tensor zt = zeros(1, 2);
                    zt.v[0] = mu[0] + sd[0] * e0;
                    zt.v[1] = mu[1] + sd[1] * e1;
                    Var z = param(zt);                                                // 把 z 当成「参数」，读它的梯度
                    Var f = scale(bceWithLogits(mlpForward(m.dec, z), x), -1.f);      // f(z) = log p(x|z)
                    backward(f);                                                      // （解码器权重的 grad 也会累加，下一步训练前 zeroGrad 即可）
                    GradSample g;
                    g.f = f->val.v[0];
                    g.pathwise = z->grad.v[0];                                        // 重参数化：∂f/∂μ₁ = ∂f/∂z₁ · ∂z₁/∂μ₁ = ∂f/∂z₁
                    g.score = g.f * e0 / sd[0];                                       // score function：f·(z₁ − μ₁)/σ₁² = f·ε₁/σ₁
                    return g;
                }

                void meanStd(const std::vector<float>& a, double& mean, double& sd) {
                    mean = 0; sd = 0;
                    for (float v : a) mean += v / a.size();
                    for (float v : a) sd += (v - mean) * (v - mean) / a.size();
                    sd = std::sqrt(sd);
                }

                // 抽 n 个样本：三种估计（重参数化 / score / score − 基线）的均值与标准差
                struct EstStats { double pwMean, pwStd, sfMean, sfStd, sfbMean, sfbStd; };

                EstStats compareEstimators(const VAE& m, const Tensor& x, const std::vector<float>& mu, const std::vector<float>& sd, int n, Rng& r) {
                    std::vector<float> pw = floats(n), sf = floats(n), f = floats(n), e = floats(n), sfb = floats(n);
                    for (int s = 0; s < n; s++) {
                        float e0 = randn(r), e1 = randn(r);
                        GradSample g = gradSample(m, x, mu, sd, e0, e1);
                        pw[s] = g.pathwise; sf[s] = g.score; f[s] = g.f; e[s] = e0;
                    }
                    double fbar = 0;
                    for (int s = 0; s < n; s++) fbar += f[s] / n;
                    for (int s = 0; s < n; s++) sfb[s] = (float)((f[s] - fbar) * e[s] / sd[0]);   // 基线 b = f 的均值
                    EstStats st;
                    meanStd(pw, st.pwMean, st.pwStd);
                    meanStd(sf, st.sfMean, st.sfStd);
                    meanStd(sfb, st.sfbMean, st.sfbStd);
                    return st;
                }`,
            },
            previews: [
                {
                    id: 'gradRun', label: '三种估计的累计平均 vs 样本数', sub: '同一组 ε · 蓝 = 重参数化 · 红 = score function · 绿 = score − 基线 · 灰虚线 = 重参数化 256 个样本的均值（当作「真值」）· 超出纵轴的部分被截断', live: true, large: true, size: [360, 260], needs: ['grad'],
                    render: (c, S) => {
                        const g = S.grad, n = g.n, run = (a) => { const out = []; let s = 0; for (let k = 0; k < n; k++) { s += a[k]; out.push(s / (k + 1)); } return out; };
                        const pw = run(g.pw), ref = pw.length ? pw[pw.length - 1] : 0, sfb = g.sfb ? run(g.sfb) : [], xs = P09.range(n);
                        const fb = n ? LM.mean(g.f.subarray(0, n)) : 0, sfbL = g.sfb ? sfb : run(Array.from(g.f.subarray(0, n), (f, k) => (f - fb) * g.e0[k] / g.sd[0]));
                        P09.curves(c, [{ x: xs, y: run(g.sf), color: LM.C.red, width: 1.6, label: 'score function' }, { x: xs, y: sfbL, color: LM.C.green, width: 1.6, label: 'score − 基线' }, { x: xs, y: pw, color: LM.C.blue, width: 2.2, label: '重参数化' }, { x: [1, P09.GN], y: [ref, ref], color: LM.C.gray, dash: [4, 4] }],
                            { x: [1, P09.GN], y: [ref - 30, ref + 30], xLabel: `样本数（第 ${g.step} 步的解码器，样本 A = #${S.sample}）`, legendAt: [1, 0.02] });
                    },
                },
                {
                    id: 'gradHist', label: '单样本估计的分布', sub: '蓝 = 重参数化，绿 = score − 基线（同一横轴）· score function 本身的分布宽得画不下，见右下角的标准差', live: true, size: [360, 190], needs: ['grad'],
                    render: (c, S) => {
                        const g = S.grad, n = g.n, { ctx, w, h } = PL.draw.ctx(c), NB = 41;
                        const fb = n ? LM.mean(g.f.subarray(0, n)) : 0, sfb = Array.from({ length: n }, (_, k) => (g.f[k] - fb) * g.e0[k] / g.sd[0]);
                        let mx = 1; for (let k = 0; k < n; k++) mx = Math.max(mx, Math.abs(g.pw[k]), Math.abs(sfb[k]));
                        const lim = Math.min(mx, 4 * Math.sqrt(sfb.reduce((a, v) => a + v * v, 0) / Math.max(1, n)) + 1), hist = (a) => { const H = new Float32Array(NB); for (const v of a) { const b = Math.round((v / lim + 1) / 2 * (NB - 1)); if (b >= 0 && b < NB) H[b]++; } return H; };
                        const Hp = hist(Array.from(g.pw.subarray(0, n))), Hs = hist(sfb), top = Math.max(1, ...Hp, ...Hs), x0 = 36, x1 = w - 10, y0 = 24, y1 = h - 28, bw = (x1 - x0) / NB;
                        P09.txt(ctx, `${n} / ${P09.GN} 个样本`, 40, 4, { bold: true, color: LM.C.yellow });
                        [[Hs, LM.C.green], [Hp, LM.C.blue]].forEach(([H, col]) => { ctx.globalAlpha = 0.7; ctx.fillStyle = col; for (let b = 0; b < NB; b++) { const hh = H[b] / top * (y1 - y0); ctx.fillRect(x0 + b * bw + 1, y1 - hh, bw - 2, hh); } ctx.globalAlpha = 1; });
                        ctx.strokeStyle = '#2a2f37'; ctx.beginPath(); ctx.moveTo(x0, y1 + 0.5); ctx.lineTo(x1, y1 + 0.5); ctx.stroke();
                        [-lim, 0, lim].forEach((v) => P09.txt(ctx, v.toFixed(0), x0 + (v / lim + 1) / 2 * (x1 - x0), y1 + 3, { align: 'center', color: LM.C.dim }));
                        if (g.stats) P09.txt(ctx, `σ：重参数化 ${g.stats.pw.std.toFixed(1)} · score−基线 ${g.stats.sfb.std.toFixed(1)} · score ${g.stats.sf.std.toFixed(0)}`, w - 8, h - 13, { align: 'right', color: LM.C.text });
                    },
                },
                {
                    id: 'gradStats', label: '统计（最近完成的一轮 256 个样本）', sub: '方差比 = Var(score 类估计) / Var(重参数化)：要达到同样的精度，score function 需要多这么多倍的样本', live: true, size: [360, 128], needs: ['grad'],
                    render: (c, S) => {
                        const g = S.grad, st = g.stats;
                        if (!st) { LM.draw.text(c, [{ text: `正在采样：${g.n} / ${P09.GN}…`, color: LM.C.gray }], { px: 12 }); return; }
                        LM.draw.text(c, [
                            [{ t: '重参数化      ', color: LM.C.blue }, { t: `均值 ${st.pw.mean.toFixed(2)} ± ${(st.pw.std / 16).toFixed(2)}   σ = ${st.pw.std.toFixed(2)}` }],
                            [{ t: 'score         ', color: LM.C.red }, { t: `均值 ${st.sf.mean.toFixed(2)} ± ${(st.sf.std / 16).toFixed(2)}   σ = ${st.sf.std.toFixed(1)}   方差比 ${(st.sf.std ** 2 / st.pw.std ** 2).toFixed(0)}×` }],
                            [{ t: 'score − 基线  ', color: LM.C.green }, { t: `均值 ${st.sfb.mean.toFixed(2)} ± ${(st.sfb.std / 16).toFixed(2)}   σ = ${st.sfb.std.toFixed(2)}   方差比 ${(st.sfb.std ** 2 / st.pw.std ** 2).toFixed(1)}×` }],
                            { text: `f = log p(x|z) 的均值 ${st.fbar.toFixed(1)} nats——score function 的方差主要来自这个很大的常数偏移`, color: LM.C.gray },
                        ], { px: 12, lineGap: 6 });
                    },
                },
            ],
            notes: '红线（score function）在前几十个样本里大幅摆动，256 个样本后误差仍有十几（「± 」= 标准误差 σ/√256）；蓝线几个样本后就稳定在真值附近。换一个样本 A，或者等 VAE 训练更久再看：方差比一直是几千倍的量级。为什么 score function 这么差？f 本身约 −90 nats，乘上 ε/σ 之后被放大，正负抵消要靠大量样本——减去均值基线后就好得多，这也是强化学习里「优势函数」的由来。',
            warn: '重参数化要求 f 对 z 可导、而且 z 是连续的。离散的潜变量（下面的 VQ-VAE、语言模型的 token、强化学习的动作）不能这样做：要么用 score function + 基线，要么用连续松弛（Gumbel-softmax），要么像 VQ-VAE 一样用「直通估计」近似。图中的估计只针对 μ₁ 这一个分量；实际训练时 autograd 一次就给出所有参数的重参数化梯度。',
        },
        // ───────────────────────────── 06 VAE 潜空间 ─────────────────────────────
        {
            id: 'vaeLatent', title: 'VAE 的潜空间：连续、铺满先验、可以直接采样', tag: 'VAE',
            desc: 'KL 项让所有 q(z|x) 叠起来的「聚合后验」q(z) = (1/N) Σ_i q(z|x_i) 尽量贴近先验 N(0, I)：编码不能跑到很远，也不能在先验里留下大片空白；重建项又要求每个 q 都能画回自己的图，于是不同的图在潜空间里互相「挤开」。结果是一个铺满单位圆附近的连续潜空间。第一张图按 Kingma & Welling 论文里的画法，在标准正态的分位点上取 12 × 12 个 z 解码（每一行 / 列在概率意义上等距）：数字从一处连续地变到另一处，几乎没有乱码。第二张图是聚合后验（每张测试图从 q(z|x) 抽 4 个点）和先验的 1σ / 2σ / 3σ 圆；第三张是直接从先验 N(0, I) 抽 z 解码出来的 20 张「新数字」。2 维的限制也一目了然：β = 1 时 KL 只有约 3.6 nats，网格里能清楚认出的只有 1、2、4、6、7 和一团 3/8，0 / 5 / 9 没有自己的位置（Node 里把网络加宽到 256 → 256 → 64 → 2、训练 1 万步，KL 能到 5.1、类别分得更开，但浏览器里实时训练负担不起）。',
            formula: [
                'q(z) = (1/N) Σ_i q(z|x_i) ≈ p(z) = N(0, I)   // 聚合后验 ≈ 先验：从 p(z) 采样 ≈ 从「真实数据的编码」附近采样',
                'z_{rc} = (Φ^{−1}((c + ½)/12), Φ^{−1}(1 − (r + ½)/12))   // Φ = 标准正态分布函数，Φ⁻¹ 用 Acklam 的有理逼近',
                'E_x KL(q(z|x) ‖ p(z)) = I_q(x; z) + KL(q(z) ‖ p(z))   // 平均 KL = 互信息 + 聚合后验与先验的差距（Hoffman & Johnson 2016）',
            ],
            code: {
                title: 'C++ · normalQuantile / decodeQuantileGrid / samplePriorImages', src: `
                // 标准正态分布的分位数 Φ⁻¹(p)：Acklam 的有理逼近（相对误差 < 1.2·10⁻⁹）
                double normalQuantile(double p) {
                    static const double a[6] = {-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239};
                    static const double b[5] = {-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572};
                    static const double c[6] = {-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783};
                    static const double d[4] = {0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416};
                    if (p > 1 - 0.02425) return -normalQuantile(1 - p);
                    if (p < 0.02425) {
                        double q = std::sqrt(-2 * std::log(p));
                        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
                    }
                    double q = p - 0.5, r = q * q;
                    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
                }

                // Kingma & Welling 的「流形图」：在 N(0, I) 的分位点上取 n×n 个 z 解码（第 r 行从上往下）
                Tensor decodeQuantileGrid(const VAE& m, int n) {
                    NoGrad ng;
                    Tensor Z = zeros(n * n, 2);
                    for (int r = 0; r < n; r++)
                        for (int c = 0; c < n; c++) {
                            Z.at(r * n + c, 0) = (float)normalQuantile((c + 0.5) / n);
                            Z.at(r * n + c, 1) = (float)normalQuantile(1 - (r + 0.5) / n);
                        }
                    return sigmoid(mlpForward(m.dec, constant(Z)))->val;
                }

                // 生成：z ~ p(z) = N(0, I)，x = σ(g(z))——VAE 自带先验，不需要额外拟合
                Tensor samplePriorImages(const VAE& m, int n, Rng& r) {
                    NoGrad ng;
                    return sigmoid(mlpForward(m.dec, constant(randn(n, 2, 1.f, r))))->val;
                }`,
            },
            previews: [
                {
                    id: 'vaeGrid', label: '在 N(0, I) 的分位点上解码（12 × 12）', sub: '横轴 z₁、纵轴 z₂ 都从 Φ⁻¹(1/24) ≈ −1.7 到 +1.7 · 每 4 步重画一次', live: true, large: true, size: [360, 380], needs: ['vaeGrid'],
                    render: (c, S) => { S._lVaeGrid = P09.gridPanel(c, S.vaeGrid.img, 12, { title: `${P09.status(S.vae)} · β = ${S.beta}`, tx: 40, titleColor: LM.C.yellow }); },
                    probe: (x, y, S, PL, info) => { const k = P09.gridHitP(S._lVaeGrid, info), Z = S.vaeGrid.Z; return k < 0 ? '' : `z = (${Z[2 * k].toFixed(2)}, ${Z[2 * k + 1].toFixed(2)})`; },
                },
                {
                    id: 'vaeAgg', label: '聚合后验 q(z) 与先验 p(z) = N(0, I)', sub: '背景 = 2000 个 z ~ q(z|x)（每张测试图 4 个）的二维直方图 · 白线 = 先验的 1σ / 2σ / 3σ 圆', live: true, size: [360, 320], needs: ['vaeGrid'],
                    render: (c, S) => {
                        const a = S.vaeAggStats;
                        LM.draw.scatter(c, [], { x: [-3.6, 3.6], y: [-3.6, 3.6], equal: true, bg: S.vaeAgg.bg, bgRes: 72, paths: [1, 2, 3].map((k) => ({ pts: P09.ellipse(0, 0, 1, 1, k), color: `rgba(255,255,255,${(0.9 - 0.2 * k).toFixed(2)})`, width: 1.2, dash: k > 1 ? [4, 3] : [] })), xLabel: 'z₁', yLabel: 'z₂', title: `q(z) 的均值 (${a.m0.toFixed(2)}, ${a.m1.toFixed(2)})、方差 (${a.v0.toFixed(2)}, ${a.v1.toFixed(2)})`, titleX: 46 });
                    },
                },
                {
                    id: 'vaeSamples', label: '从先验采样：z ~ N(0, I) → 解码', sub: '20 个固定的 z（随训练、随 β 一起变）', live: true, size: [360, 104], needs: ['vaeGrid'],
                    render: (c, S) => P09.tiles(c, P09.range(20, 0).map((k) => ({ img: S.vaePrior.subarray(k * 256, (k + 1) * 256) })), 10, { gap: 3 }),
                },
            ],
            notes: '和 Step 03 的 AE 网格比：这里没有「红框」——先验下有概率的地方都解码得出数字，而且相邻格子只差一点点。聚合后验的方差应该接近 1：β = 1 时大约是 0.9–1.0。数字的种类在网格里并不均等：1 和 7 各占一大片（它们的写法变化大、和别的数字差异也大），圆形数字挤在中间。',
            warn: '「聚合后验 = 先验」只是近似：KL 项是对每个 q(z|x) 分别施加的，它也会把每个后验都往原点拉、让它们互相重叠（这正是 2 维时细节变糊的原因之一）。先验和聚合后验之间仍有空隙（prior hole）时，从先验采样会落到没有数据的地方——更大的模型常换成可学习的先验（VampPrior、流模型先验），或者像 Step 10 的 VQ-VAE 那样在潜空间上再训练一个先验模型。',
        },
        // ───────────────────────────── 07 β ─────────────────────────────
        {
            id: 'beta', title: 'β 的权衡：率-失真曲线与后验坍缩', tag: 'RATE',
            desc: '把损失写成 D + β·R：<strong>D（distortion，失真）</strong> = 重建的负对数似然，<strong>R（rate，率）</strong> = KL，它是编码一张图时 z 平均「多带了」多少 nats 的信息（Alemi et al. 2018 的视角；R 是 x 与 z 之间互信息的上界）。β 就是两者的汇率：β 小 → 信息便宜，重建好，但潜空间更像 AE；β 大 → 信息贵，z 里只留最重要的东西——Higgins et al. 2017 的 β-VAE 就是用 β > 1 换取更「解耦」的表示。β 再大，KL 项就赢了：编码器干脆让 q(z|x) = N(0, I)、一点信息都不传，解码器只能输出所有数字的<strong>平均图</strong>——这就是<strong>后验坍缩</strong>（posterior collapse）。曲线上的点是在 Node 里用同一份代码、同样的数据和步数，为每个 β 各训练一个 VAE 的结果（tools/p09_rd.js）；黄点和黄线是页面里正在训练的 VAE（当前 β）在评估时的轨迹，训练结束时应该正好落到对应的点上。β = 4 时第 2 维的 KL 只剩 0.29 nats（第三张图「沿 z₂ 移动」那一行几乎不变：解码器开始忽略这一维）；β = 8 时两维都坍缩（KL = 0.006 nats，MSE = 0.081，正好是用平均图当重建的 MSE）。',
            formula: [
                'L_β = D + β · R      D = E_x E_{q(z|x)}[−log p(x|z)]      R = E_x KL(q(z|x) ‖ p(z))   // β = 1 ⇔ −ELBO',
                'R = I_q(x; z) + KL(q(z) ‖ p(z)) ≥ I_q(x; z)   // 率是 x 与 z 互信息的上界：编码多少信息就要付多少 KL',
                '坍缩：q(z_d|x) = N(0, 1) ∀x  ⇒  KL_d = 0，解码器对 z_d 没有依赖   // 补救：KL 退火（Bowman et al. 2016）、free bits（Kingma et al. 2016）',
            ],
            code: {
                title: 'C++ · rateDistortion / sweepBeta', src: `
                // 测试集上的率与失真（nats / 图）：R = E[KL(q(z|x) ‖ p(z))]，D = E_q[−log p(x|z)]（每张图从 q 抽一个 z）
                struct RatePoint { double rate, dist; std::vector<double> klDim; };

                RatePoint rateDistortion(const VAE& m, const Tensor& Xt, Rng& r) {
                    NoGrad ng;
                    VAEOut o = vaeForward(m, Xt, randn(Xt.rows, 2, 1.f, r), 1.f);
                    RatePoint p;
                    p.dist = o.rec->val.v[0];
                    p.rate = o.kl->val.v[0];
                    p.klDim = doubles(2);
                    for (int i = 0; i < Xt.rows; i++)
                        for (int d = 0; d < 2; d++) {
                            float mu = o.mu->val.at(i, d), lv = o.logvar->val.at(i, d);
                            p.klDim[d] += -0.5 * (1 + lv - mu * mu - std::exp(lv)) / Xt.rows;   // 每一维各自的 KL：≈ 0 = 这一维坍缩了
                        }
                    return p;
                }

                // β 扫描：每个 β 从头训练一个 VAE（同样的数据、种子、步数），记录 (R, D)——离线跑一次，结果嵌进页面
                std::vector<RatePoint> sweepBeta(const DigitSet& train, const Tensor& Xt, const std::vector<float>& betas, int steps) {
                    std::vector<RatePoint> out;
                    for (float beta : betas) {
                        VAE m = trainVAE(train, beta, steps, 20251145u);
                        Rng r = makeRng(7);
                        out.push_back(rateDistortion(m, Xt, r));
                    }
                    return out;
                }`,
            },
            previews: [
                {
                    id: 'rdCurve', label: '率-失真曲线：每个 β 训练 3000 步后的 (R, D)', sub: '点 = 离线 β 扫描 · 灰虚线 = AE（没有 KL 项，相当于 β → 0）· 黄 = 页面里正在训练的 VAE（当前 β）从第 0 步到现在的评估轨迹', live: true, large: true, size: [360, 280],
                    render: (c, S) => {
                        const pts = [], pts2 = [], T = S.vae.thist, rd = P09.RD.filter((r) => r.rate != null), ae = P09.RD[P09.RD.length - 1];
                        rd.forEach((r) => pts.push({ x: r.rate, y: r.dist, color: r.beta === S.beta ? LM.C.yellow : LM.C.blue, r: r.beta === S.beta ? 4 : 3.2, label: 'β=' + r.beta, labelColor: r.beta === S.beta ? LM.C.yellow : LM.C.gray, ring: r.beta === S.beta ? '#000' : null }));
                        const path = T.step.map((_, k) => [T.kl[k], T.rec[k]]).filter((p) => p[1] < 112);
                        if (S.vae.res) pts2.push({ x: S.vae.res.kl, y: Math.min(111.5, S.vae.res.rec), color: LM.C.yellow, r: 4.5, ring: '#fff' });
                        LM.draw.scatter(c, pts.concat(pts2), { x: [-0.3, 7.6], y: [88, 112], paths: [{ pts: rd.map((r) => [r.rate, r.dist]), color: 'rgba(106,176,224,0.6)', width: 1.4 }, { pts: path, color: 'rgba(232,194,90,0.75)', width: 1.4 }], lines: [{ a: [-1, ae.dist], b: [9, ae.dist], color: LM.C.gray, dash: [4, 4] }], xLabel: 'R = KL（nats / 图）', yLabel: 'D = 重建 −log p（nats / 图）', title: `当前 β = ${S.beta}：R = ${S.vae.res.kl.toFixed(2)}，D = ${S.vae.res.rec.toFixed(2)}（${P09.status(S.vae)}）`, titleX: 46 });
                    },
                },
                {
                    id: 'klStack', label: '每一维的 KL（离线扫描 + 当前的实时 VAE）', sub: '每根柱子 = 一个 β 的总 KL，青 = z₁、黄 = z₂ · 最右 = 正在训练的 VAE · 某一维的 KL ≈ 0 = 这一维坍缩了', live: true, size: [360, 200],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c), rd = P09.RD.filter((r) => r.rate != null), bars = rd.map((r) => ({ lab: String(r.beta), kd: r.klDim, hi: r.beta === S.beta })).concat([{ lab: '实时', kd: S.vae.res.klDim, hi: true, live: true }]);
                        const x0 = 34, x1 = w - 8, y0 = 22, y1 = h - 34, top = 7.5, bw = (x1 - x0) / bars.length, Y = (v) => y1 - v / top * (y1 - y0);
                        P09.txt(ctx, `当前 β = ${S.beta}`, 40, 4, { bold: true, color: LM.C.yellow });
                        [0, 2.5, 5, 7.5].forEach((v) => { ctx.strokeStyle = '#1c1c1c'; ctx.beginPath(); ctx.moveTo(x0, Y(v)); ctx.lineTo(x1, Y(v)); ctx.stroke(); P09.txt(ctx, String(v), x0 - 4, Y(v), { align: 'right', base: 'middle', color: LM.C.dim }); });
                        bars.forEach((b, k) => {
                            const x = x0 + k * bw + bw * 0.18, ww = bw * 0.64;
                            ctx.fillStyle = LM.C.cyan; ctx.fillRect(x, Y(b.kd[0]), ww, y1 - Y(b.kd[0]));
                            ctx.fillStyle = LM.C.yellow; ctx.fillRect(x, Y(b.kd[0] + b.kd[1]), ww, Y(b.kd[0]) - Y(b.kd[0] + b.kd[1]));
                            if (b.hi) { ctx.strokeStyle = b.live ? '#ffffff' : LM.C.yellow; ctx.lineWidth = 1.5; ctx.strokeRect(x - 2, Y(b.kd[0] + b.kd[1]) - 2, ww + 4, y1 - Y(b.kd[0] + b.kd[1]) + 2); }
                            P09.txt(ctx, b.lab, x + ww / 2, y1 + 4, { align: 'center', color: b.hi ? LM.C.text : LM.C.gray });
                            P09.txt(ctx, (b.kd[0] + b.kd[1]).toFixed(b.kd[0] + b.kd[1] < 0.1 ? 3 : 1), x + ww / 2, Y(b.kd[0] + b.kd[1]) - 2, { align: 'center', base: 'bottom', color: LM.C.dim });
                        });
                        P09.txt(ctx, 'β →', x0, h - 14, { color: LM.C.gray });
                    },
                },
                {
                    id: 'traverse', label: '沿单一维度移动 z（另一维 = 0）：解码器还在用这一维吗？', sub: '每行 11 个点，t 从 −2.5 到 +2.5 · 右边数字 = 相邻两张图像素差的总和：≈ 0 说明解码器完全忽略了这一维', live: true, size: [360, 136], needs: ['vaeTrav'],
                    render: (c, S) => {
                        const L = P09.rowsPanel(c, S.trav.map((t, d) => ({ label: `沿 z${d ? '₂' : '₁'}`, imgs: P09.range(11, 0).map((k) => t.img.subarray(k * 256, (k + 1) * 256)), labelColor: d ? LM.C.yellow : LM.C.cyan })), { title: `β = ${S.beta} · ${P09.status(S.vae)}`, tx: 40, titleColor: LM.C.yellow, labW: 52 });
                        S.trav.forEach((t, d) => P09.txt(L.ctx, t.tv.toFixed(1), L.w - 4, L.ys[d] + L.cell / 2, { align: 'right', base: 'middle', color: t.tv < 1 ? LM.C.red : LM.C.text }));
                    },
                },
            ],
            notes: '把 <b>β</b> 从 1 拖到 4：VAE 从头训练（约 50 秒），黄点沿着轨迹落到 β = 4 的点上，KL 柱子变矮，「沿 z₂」那一行的变化量变小；拖到 8：两行都不再变化，所有解码都是同一张模糊的平均数字（Step 06 的网格也变成一样的图）。拖到 0.1：KL 变大，但重建只好了一点点——曲线左段很平，说明这个 2 维解码器已经用不了更多信息。',
            warn: '曲线上的点各自只训练了 3000 步，是「这个模型 + 这个训练预算」的率-失真曲线，不是数据本身的理论极限。大 β 下的坍缩和训练动态有关：训练初期解码器还很弱，KL 的梯度占上风，编码器先坍缩，之后就很难恢复（Bowman et al. 2016 在文本 VAE 上用 KL 退火——让 β 从 0 慢慢升到 1——和 word dropout 缓解这个问题）。',
        },
        // ───────────────────────────── 08 插值 ─────────────────────────────
        {
            id: 'interp', title: '插值：在潜空间里走直线', tag: 'INTERP',
            desc: '从样本 A 走到样本 B 有两种走法。<strong>像素空间</strong>里直接交叉淡化 x_t = (1 − t)x_A + t·x_B：中间的图只是两张图叠在一起的「重影」，没有一张像真实的数字。<strong>潜空间</strong>里先编码、在 z 上走直线、再解码：每一步都是解码器画出来的「像样的数字」，笔画连续地变形。AE 和 VAE 的区别在于潜空间里的「距离」是否可信：AE 的编码密度很不均匀，直线走过密集区时变化很快、走过空白区时又几乎不动，偶尔还会穿过洞；VAE 的潜空间被 KL 拉得比较均匀，同样的步长对应差不多的变化。在 Node 里用同一份代码对 262 对不同类别的随机测试图统计「最大一步的变化 / 平均每步的变化」：AE 1.69、VAE 1.45（完全均匀 = 1），AE 有 5% 的路径出现超过平均 2.5 倍的「跳变」，VAE 是 0%。注意 2 维潜空间里，任何两个数字之间的直线都会经过其他数字（比如 4 → 7 会路过 6 / 8 / 3）——这不是 bug，而是 2 维装不下 10 类数字的必然结果。',
            formula: [
                'x_t = (1 − t) x_A + t x_B   // 像素空间：重影',
                'z_t = (1 − t) z_A + t z_B，x̂_t = σ(g(z_t))   // 潜空间：AE 用 z = f(x)，VAE 用后验均值 μ(x)',
                '均匀度 = max_t ‖x̂_{t+1} − x̂_t‖ / mean_t ‖x̂_{t+1} − x̂_t‖   // T = 10 步；= 1 表示每一步变化一样大',
            ],
            code: {
                title: 'C++ · interpolate / pixelBlend / stepChanges', src: `
                // 潜空间里的直线插值：z_t = (1 − t)·z_A + t·z_B，t = 0, 1/T, …, 1，逐个解码（VAE 用 μ 当 z）
                Tensor interpolate(const MLP& dec, const std::vector<float>& zA, const std::vector<float>& zB, int T) {
                    NoGrad ng;
                    Tensor Z = zeros(T + 1, 2);
                    for (int t = 0; t <= T; t++)
                        for (int d = 0; d < 2; d++) Z.at(t, d) = (1 - (float)t / T) * zA[d] + (float)t / T * zB[d];
                    return sigmoid(mlpForward(dec, constant(Z)))->val;
                }

                // 对照组：像素空间的交叉淡化——只是两张图叠在一起
                Tensor pixelBlend(const Tensor& xA, const Tensor& xB, int T) {
                    Tensor out = zeros(T + 1, D);
                    for (int t = 0; t <= T; t++)
                        for (int j = 0; j < D; j++) out.at(t, j) = (1 - (float)t / T) * xA.v[j] + (float)t / T * xB.v[j];
                    return out;
                }

                // 每一步的像素变化 ‖x̂_{t+1} − x̂_t‖：越均匀，说明潜空间里的「距离」越可信
                std::vector<float> stepChanges(const Tensor& frames) {
                    std::vector<float> d = floats(frames.rows - 1);
                    for (int t = 0; t + 1 < frames.rows; t++) {
                        double s = 0;
                        for (int j = 0; j < frames.cols; j++) { double e = frames.at(t + 1, j) - frames.at(t, j); s += e * e; }
                        d[t] = (float)std::sqrt(s);
                    }
                    return d;
                }`,
            },
            previews: [
                {
                    id: 'interpRows', label: '从样本 A 到样本 B：像素 / AE / VAE', sub: '每行 11 张，t = 0, 0.1, …, 1 · 黄框 = A、青框 = B（顶栏换样本，也可以在 Step 02 / 04 的散点上点选 A）', live: true, large: true, size: [360, 178], needs: ['interp'],
                    render: (c, S) => {
                        const I = S.interp, row = (a) => P09.range(11, 0).map((k) => a.subarray(k * 256, (k + 1) * 256)), ol = P09.range(11, 0).map((k) => (k === 0 ? LM.C.yellow : k === 10 ? LM.C.cyan : null));
                        P09.rowsPanel(c, [{ label: '像素', imgs: row(I.pix.img), outline: ol }, { label: 'AE', imgs: row(I.ae.img), outline: ol, labelColor: LM.C.orange }, { label: 'VAE', imgs: row(I.vae.img), outline: ol, labelColor: LM.C.green }], { title: `#${S.sample}（${S.data.te.y[S.sample]}）→ #${S.sampleB}（${S.data.te.y[S.sampleB]}）`, tx: 40, titleColor: LM.C.yellow, labW: 44 });
                    },
                },
                {
                    id: 'interpPaths', label: '同一条直线在两个潜空间里的位置', sub: '点 = 500 张测试图的编码（颜色 = 类别）· 白线 = 插值路径（11 个点，黄 = A、青 = B）', live: true, size: [360, 200], needs: ['interp', 'aeView', 'vaeView'],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c), y = S.data.te.y;
                        P09.miniLatent(ctx, 2, 2, w / 2 - 4, h - 4, S.ae.res.z, y, P09.NTEST, S.aeBox, S.interp.ae.Z, 'AE');
                        P09.miniLatent(ctx, w / 2 + 2, 2, w / 2 - 4, h - 4, S.vae.res.z, y, P09.NTEST, S.vaeBox, S.interp.vae.Z, 'VAE');
                    },
                },
                {
                    id: 'interpSteps', label: '每一步的像素变化 ‖x̂_{t+1} − x̂_t‖', sub: '灰 = 像素交叉淡化（必然均匀）· 橙 = AE · 绿 = VAE · 括号里 = 最大一步 / 平均一步', live: true, size: [360, 160], needs: ['interp'],
                    render: (c, S) => {
                        const I = S.interp, { ctx, w, h } = PL.draw.ctx(c), groups = [['像素', I.pix.steps, LM.C.gray], ['AE', I.ae.steps, LM.C.orange], ['VAE', I.vae.steps, LM.C.green]];
                        let mx = 0.1; groups.forEach((g) => { for (const v of g[1]) mx = Math.max(mx, v); });
                        const x0 = 8, gw = (w - 16) / 3, y0 = 20, y1 = h - 20;
                        groups.forEach(([name, st, col], q) => {
                            const bw = (gw - 14) / st.length, gx = x0 + q * gw;
                            for (let t = 0; t < st.length; t++) { const hh = st[t] / mx * (y1 - y0); ctx.fillStyle = col; ctx.fillRect(gx + t * bw + 1, y1 - hh, bw - 2, hh); }
                            const r = Math.max(...st) / (LM.mean(Array.from(st)) || 1);
                            P09.txt(ctx, `${name}（${r.toFixed(2)}）`, gx + (gw - 14) / 2, y1 + 4, { align: 'center', color: col });
                        });
                        P09.txt(ctx, `最大一步 = ${mx.toFixed(2)}`, w - 8, 4, { align: 'right', color: LM.C.dim });
                    },
                },
            ],
            notes: '默认是 4 → 7：AE 那一行中途路过 6、8、3，VAE 路过 6 / 8；像素那一行是一张 4 慢慢淡出、一张 7 慢慢淡入。试试同一类的两张图（比如两张写法不同的 7）：潜空间插值会连续地改变倾斜和粗细，像素插值仍然是重影。',
            warn: '对高斯潜空间，直线插值的中点 ‖z‖ 会比两端小（两个独立高斯向量的平均更靠近原点），高维时这些点落在先验的「低概率壳层」里；很多图像生成工作因此改用球面插值 slerp。2 维时这个效应不明显，所以这里用直线。',
        },
        // ───────────────────────────── 09 VQ-VAE ─────────────────────────────
        {
            id: 'vq', title: 'VQ-VAE：把图变成 16 个离散 token（实时训练）', tag: 'VQ',
            desc: 'VQ-VAE（van den Oord et al. 2017）换了一条路：潜变量不是连续的高斯，而是从一本<strong>码本</strong>（codebook，K 个 8 维向量 e_1 … e_K）里挑出来的。卷积编码器把 16×16 的图变成 4×4 = 16 个 8 维向量 z_e；每个向量换成码本里离它<strong>最近</strong>的码字——这一步叫<strong>向量量化</strong>；卷积解码器（3×3 卷积 + 转置卷积）再从这 16 个码字画回整张图。于是一张图 = 16 个 0…K−1 的整数（<strong>token</strong>）：这就是第 07 篇的「图像 token」、第 13 篇音频 codec、DALL·E / VQGAN 的离散化方式。量化（argmin）没有梯度，于是用<strong>直通估计</strong>（straight-through，Bengio et al. 2013）：反向时假装量化是恒等映射，把 ∂L/∂z_q 原样交给 z_e。码本自己靠<strong>码本损失</strong> ‖sg(z_e) − e‖² 往编码器输出靠拢；<strong>commitment 损失</strong>（系数 0.25，论文的取值）反过来让编码器输出别离选中的码字太远。码本只有被选中的码字才会收到梯度——初始化时离编码器输出太远的码字永远不会被选中，成为<strong>死码</strong>：默认 K = 128 时，训练到最后只有约 40 个码字在用（第三张图的红柱）。打开顶栏的<strong>死码重启</strong>（每 25 步把没人用的码字挪到一个随机的编码器输出上，Jukebox 的做法），128 个码字几乎全部用上，重建也更好。',
            formula: [
                'z_e = E(x) ∈ R^{4×4×8}      k*_p = argmin_k ‖z_e(p) − e_k‖^2      z_q(p) = e_{k*_p}   // 16 个位置各选一个码字',
                'x̂ = σ(G(z_q))      直通：z_q ≔ z_e + sg(z_q − z_e)   ⇒   ∂L/∂z_e = ∂L/∂z_q   // sg = stop-gradient（PyTorch 的 detach）',
                'L = BCE(x, x̂) + ‖sg(z_e) − z_q‖^2 + β ‖z_e − sg(z_q)‖^2      β = 0.25   // 重建 + 码本损失（只更新码本）+ commitment（只更新编码器）',
                '一张图的信息量 ≤ 16 · log_2 K bit   // K = 128：112 bit；原图 256 像素 × 8 bit = 2048 bit',
            ],
            code: {
                title: 'C++ · VQVAE / convTranspose2d / vectorQuantize / vqForward / vqTrainStep / restartDeadCodes', src: `
                // VQ-VAE：16×16 → 4×4 个 8 维向量 → 量化成 4×4 个 token → 画回 16×16
                //   编码器 Conv(1→16, k8, s4, p2) → ReLU → Conv(16→8, k3, p1)；解码器 Conv(8→16, k3, p1) → ReLU → ConvT(16→1, k8, s4, p2)
                const int GRID = 4, NPOS = GRID * GRID, CODE_DIM = 8, CH = 16;

                struct VQVAE { Var W1, b1, W2, b2, codebook, W3, b3, W4, b4; int K = 128; };

                VQVAE makeVQVAE(int K, Rng& r) {
                    VQVAE m;
                    m.K = K;
                    m.W1 = param(heInit(CH, 64, 64, r));                      m.b1 = param(zeros(1, CH));
                    m.W2 = param(heInit(CODE_DIM, CH * 9, CH * 9, r));        m.b2 = param(zeros(1, CODE_DIM));
                    Tensor cb = zeros(K, CODE_DIM);
                    for (float& v : cb.v) v = (float)((2 * uniform(r) - 1) / K);   // 码本初始化 U(−1/K, 1/K)（常见的默认值）
                    m.codebook = param(cb);
                    m.W3 = param(heInit(CH, CODE_DIM * 9, CODE_DIM * 9, r));  m.b3 = param(zeros(1, CH));
                    m.W4 = param(heInit(CH, 64, CH, r));                      m.b4 = param(zeros(1, 1));
                    return m;
                }

                std::vector<Var> vqParams(const VQVAE& m) { std::vector<Var> p = {m.W1, m.b1, m.W2, m.b2, m.codebook, m.W3, m.b3, m.W4, m.b4}; return p; }

                // 转置卷积（库里没有，写成自定义算子）：X[N × C·Hi·Wi]，Wt[C × Co·k·k] → Y[N × Co·Ho·Wo]，Ho = (Hi − 1)·s − 2p + k
                //   前向 = 每个输入格子把一张 k×k 的「印章」乘上自己的值叠加到输出上；反向 = 同样的下标对应关系反过来累加
                Var convTranspose2d(const Var& X, const Var& Wt, const Var& b, int C, int Hi, int Wi, int Co, int k, int s, int p) {
                    int N = X->val.rows, Ho = (Hi - 1) * s - 2 * p + k, Wo = (Wi - 1) * s - 2 * p + k;
                    Tensor y = zeros(N, Co * Ho * Wo);
                    for (int n = 0; n < N; n++)
                        for (int co = 0; co < Co; co++)
                            for (int i = 0; i < Ho * Wo; i++) y.v[(n * Co + co) * Ho * Wo + i] = b->val.v[co];
                    for (int n = 0; n < N; n++)
                        for (int c = 0; c < C; c++)
                            for (int iy = 0; iy < Hi; iy++)
                                for (int ix = 0; ix < Wi; ix++) {
                                    float xv = X->val.v[((n * C + c) * Hi + iy) * Wi + ix];
                                    for (int co = 0; co < Co; co++)
                                        for (int ky = 0; ky < k; ky++)
                                            for (int kx = 0; kx < k; kx++) {
                                                int oy = iy * s - p + ky, ox = ix * s - p + kx;
                                                if (oy < 0 || oy >= Ho || ox < 0 || ox >= Wo) continue;
                                                y.v[((n * Co + co) * Ho + oy) * Wo + ox] += xv * Wt->val.v[(c * Co + co) * k * k + ky * k + kx];
                                            }
                                }
                    return record(std::move(y), {X, Wt, b}, [X, Wt, b, C, Hi, Wi, Co, k, s, p, N, Ho, Wo](Node& o) {
                        for (int n = 0; n < N; n++)
                            for (int c = 0; c < C; c++)
                                for (int iy = 0; iy < Hi; iy++)
                                    for (int ix = 0; ix < Wi; ix++) {
                                        int xi = ((n * C + c) * Hi + iy) * Wi + ix;
                                        for (int co = 0; co < Co; co++)
                                            for (int ky = 0; ky < k; ky++)
                                                for (int kx = 0; kx < k; kx++) {
                                                    int oy = iy * s - p + ky, ox = ix * s - p + kx;
                                                    if (oy < 0 || oy >= Ho || ox < 0 || ox >= Wo) continue;
                                                    float g = o.grad.v[((n * Co + co) * Ho + oy) * Wo + ox];
                                                    int wi = (c * Co + co) * k * k + ky * k + kx;
                                                    if (X->requiresGrad) gradOf(X).v[xi] += g * Wt->val.v[wi];
                                                    if (Wt->requiresGrad) gradOf(Wt).v[wi] += g * X->val.v[xi];
                                                }
                                    }
                        if (b->requiresGrad)
                            for (int n = 0; n < N; n++)
                                for (int co = 0; co < Co; co++)
                                    for (int i = 0; i < Ho * Wo; i++) gradOf(b).v[co] += o.grad.v[(n * Co + co) * Ho * Wo + i];
                    });
                }

                // 向量量化 + 直通估计。ze[N × Dc·HW]（通道优先：第 d 个通道、第 p 个位置 = 第 d·HW + p 列）
                //   idx[n·HW + p] = argmin_k ‖ze(n, :, p) − e_k‖²；zq 的前向值 = e_idx，反向把 ∂L/∂zq 原样交给 ze（直通）
                //   codebookLoss = (1/N) Σ ‖sg(ze) − e‖²（梯度只给码本）；commitLoss = 同一个数，但梯度只给编码器
                struct VQOut { Var zq, codebookLoss, commitLoss; std::vector<int> idx; };

                VQOut vectorQuantize(const Var& ze, const Var& E, int Dc, int HW) {
                    int N = ze->val.rows, K = E->val.rows;
                    VQOut out;
                    out.idx = ints(N * HW);
                    Tensor zq = zeros(N, Dc * HW);
                    double tot = 0;
                    for (int n = 0; n < N; n++)
                        for (int p = 0; p < HW; p++) {
                            int best = 0;
                            float bestD = 1e30f;
                            for (int k = 0; k < K; k++) {
                                float d2 = 0;
                                for (int d = 0; d < Dc; d++) { float t = ze->val.v[n * Dc * HW + d * HW + p] - E->val.v[k * Dc + d]; d2 += t * t; }
                                if (d2 < bestD) { bestD = d2; best = k; }
                            }
                            out.idx[n * HW + p] = best;
                            tot += bestD;
                            for (int d = 0; d < Dc; d++) zq.v[n * Dc * HW + d * HW + p] = E->val.v[best * Dc + d];
                        }
                    std::vector<int> idx = out.idx;
                    out.zq = record(std::move(zq), {ze}, [ze](Node& o) { Tensor& g = gradOf(ze); for (int i = 0; i < o.val.size(); i++) g.v[i] += o.grad.v[i]; });
                    Tensor l1 = zeros(1, 1);
                    l1.v[0] = (float)(tot / N);
                    Tensor l2 = l1;
                    out.codebookLoss = record(std::move(l1), {E}, [ze, E, idx, N, Dc, HW](Node& o) {
                        Tensor& g = gradOf(E);
                        float s = 2 * o.grad.v[0] / N;
                        for (int n = 0; n < N; n++)
                            for (int p = 0; p < HW; p++)
                                for (int d = 0; d < Dc; d++) g.v[idx[n * HW + p] * Dc + d] += s * (E->val.v[idx[n * HW + p] * Dc + d] - ze->val.v[n * Dc * HW + d * HW + p]);
                    });
                    out.commitLoss = record(std::move(l2), {ze}, [ze, E, idx, N, Dc, HW](Node& o) {
                        Tensor& g = gradOf(ze);
                        float s = 2 * o.grad.v[0] / N;
                        for (int n = 0; n < N; n++)
                            for (int p = 0; p < HW; p++)
                                for (int d = 0; d < Dc; d++) g.v[n * Dc * HW + d * HW + p] += s * (ze->val.v[n * Dc * HW + d * HW + p] - E->val.v[idx[n * HW + p] * Dc + d]);
                    });
                    return out;
                }

                struct VQForward { VQOut q; Var ze, logits; };

                VQForward vqForward(const VQVAE& m, const Tensor& xb) {
                    VQForward f;
                    Var h = relu(conv2d(constant(xb), m.W1, m.b1, conv(1, IMG, IMG, 8, 4, 2)));      // [B × 16·4·4]
                    f.ze = conv2d(h, m.W2, m.b2, conv(CH, GRID, GRID, 3, 1, 1));                     // [B × 8·4·4]
                    f.q = vectorQuantize(f.ze, m.codebook, CODE_DIM, NPOS);
                    Var g = relu(conv2d(f.q.zq, m.W3, m.b3, conv(CODE_DIM, GRID, GRID, 3, 1, 1)));    // [B × 16·4·4]
                    f.logits = convTranspose2d(g, m.W4, m.b4, CH, GRID, GRID, 1, 8, 4, 2);           // [B × 256]
                    return f;
                }

                // 码字使用率：每个码字在本 batch 被选中的次数，取滑动平均（0.97）
                void updateUsage(std::vector<float>& usage, const std::vector<int>& idx) {
                    std::vector<float> cnt = floats((int)usage.size());
                    for (int k : idx) cnt[k] += 1;
                    for (size_t k = 0; k < usage.size(); k++) usage[k] = 0.97f * usage[k] + 0.03f * cnt[k];
                }

                // 死码重启（Jukebox 的 random restart）：滑动平均使用率 < 0.03 的码字，挪到本 batch 一个随机的编码器输出上，Adam 的动量也清零
                void restartDeadCodes(VQVAE& m, AdamW& opt, std::vector<float>& usage, const VQForward& f, Rng& r) {
                    int ci = 0;
                    for (int i = 0; i < (int)opt.params.size(); i++) if (opt.params[i].n == m.codebook.n) ci = i;
                    int N = f.ze->val.rows;
                    for (int k = 0; k < m.K; k++) {
                        if (usage[k] >= 0.03f) continue;
                        int n = randint(r, N), p = randint(r, NPOS);
                        for (int d = 0; d < CODE_DIM; d++) {
                            m.codebook->val.v[k * CODE_DIM + d] = f.ze->val.v[n * CODE_DIM * NPOS + d * NPOS + p] + 0.01f * randn(r);
                            opt.m[ci].v[k * CODE_DIM + d] = 0;
                            opt.v[ci].v[k * CODE_DIM + d] = 0;
                        }
                        usage[k] = 1;
                    }
                }

                float vqTrainStep(VQVAE& m, AdamW& opt, const Tensor& xb, std::vector<float>& usage, bool restart, Rng& r, int step, int total) {
                    zeroGrad(opt.params);
                    VQForward f = vqForward(m, xb);
                    Var rec = bceWithLogits(f.logits, xb);
                    Var loss = add(rec, add(f.q.codebookLoss, scale(f.q.commitLoss, 0.25f)));
                    backward(loss);
                    adamwStep(opt, lrAt(step, 2e-3f, total));
                    updateUsage(usage, f.q.idx);
                    if (restart && (step + 1) % 25 == 0) restartDeadCodes(m, opt, usage, f, r);
                    return rec->val.v[0];
                }`,
            },
            previews: [
                {
                    id: 'vqRecon', label: '原图 → VQ-VAE 重建 → 每张图的 16 个 token', sub: '第 3 行 = 4×4 的 token 网格（颜色 = 码字编号）· 最右 = 样本 A（黄框）', live: true, large: true, size: [360, 230], needs: ['vqView'],
                    render: (c, S) => {
                        const d = S.data, V = S.vqV, idx = d.disp.concat([S.sample]), ol = idx.map((_, k) => (k === 10 ? LM.C.yellow : null));
                        const L = P09.rowsPanel(c, [{ label: '原图', imgs: idx.map((i) => P09.imgOf(d.te.X, i)), outline: ol }, { label: 'VQ 重建', imgs: idx.map((_, k) => V.img.subarray(k * 256, (k + 1) * 256)), outline: ol, labelColor: LM.C.purple }, { label: 'token', imgs: idx.map(() => null) }], { title: `${P09.status(S.vq)} · K = ${S.vq.o.K} · 测试 MSE ${S.vq.res.mse.toFixed(4)}`, tx: 40, titleColor: LM.C.yellow, labW: 60 });
                        idx.forEach((_, k) => P09.tokenMap(L.ctx, V.tok, k * P09.NG, L.labW + k * (L.cell + L.gap), L.ys[2], L.cell, { outline: k === 10 ? LM.C.yellow : null }));
                    },
                },
                {
                    id: 'vqTokensA', label: '样本 A：一张图 = 16 个整数', sub: '左 = 原图，中 = 重建，右 = 4×4 的 token（码字编号）· 下面 = 按光栅顺序排成的序列（Step 10 的先验就在这个序列上建模）', live: true, size: [360, 168], needs: ['vqView'],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c), V = S.vqV, n = V.n - 1, s = 96, K = S.vq.o.K;
                        P09.blit(ctx, P09.imgOf(S.data.te.X, S.sample), 16, 16, 8, 22, s, s); P09.blit(ctx, V.img, 16, 16, 8 + s + 10, 22, s, s, { offset: n * 256 });
                        P09.tokenMap(ctx, V.tok, n * P09.NG, 8 + 2 * (s + 10), 22, s);
                        P09.txt(ctx, `#${S.sample} 原图`, 8 + s / 2, 4, { align: 'center', color: LM.C.yellow }); P09.txt(ctx, '重建', 8 + s * 1.5 + 10, 4, { align: 'center', color: LM.C.purple }); P09.txt(ctx, 'token', 8 + s * 2.5 + 20, 4, { align: 'center', color: LM.C.gray });
                        P09.txt(ctx, '[' + Array.from(V.tok.subarray(n * P09.NG, (n + 1) * P09.NG)).join(' ') + ']', 8, 130, { color: LM.C.text });
                        P09.txt(ctx, `16 × log₂${K} = ${(16 * Math.log2(K)).toFixed(0)} bit（原图 256 × 8 = 2048 bit）`, 8, 148, { color: LM.C.dim });
                        void w;
                    },
                },
                {
                    id: 'vqUsage', label: '码本使用率：每个码字最近被选中的次数（滑动平均）', sub: '紫 = 在用 · 红 = 死码（使用率 < 0.03 次 / batch）· 顶栏「死码重启」可以救回它们', live: true, size: [360, 184],
                    render: (c, S) => {
                        const R = S.vq, K = R.o.K, u = R.usage, { ctx, w, h } = PL.draw.ctx(c), x0 = 34, x1 = w - 8, y0 = 24, y1 = h - 20;
                        let mx = 1; for (let k = 0; k < K; k++) mx = Math.max(mx, u[k]);
                        const bw = (x1 - x0) / K, Y = (v) => y1 - Math.sqrt(v / mx) * (y1 - y0);
                        P09.txt(ctx, `${P09.status(R)} · 在用 ${K - R.deadNow} / ${K} · 困惑度 ${LM.fmt(LM.mean(R.hist.perp.slice(-50)), 1)} · 重启 ${R.restarts} 次`, 40, 4, { bold: true, color: LM.C.yellow });
                        for (let k = 0; k < K; k++) { const dead = u[k] < P09.DEAD_TH; ctx.fillStyle = dead ? LM.C.red : LM.C.purple; const yy = dead ? y1 - 3 : Y(u[k]); ctx.fillRect(x0 + k * bw + (bw > 3 ? 0.5 : 0), yy, Math.max(1, bw - (bw > 3 ? 1 : 0)), y1 - yy); }
                        [0, mx / 4, mx].forEach((v) => P09.txt(ctx, v.toFixed(v < 10 ? 1 : 0), x0 - 4, Y(v), { align: 'right', base: 'middle', color: LM.C.dim }));
                        P09.txt(ctx, '码字编号 0 … ' + (K - 1) + '（纵轴为平方根刻度）', x0, y1 + 4, { color: LM.C.gray });
                    },
                },
                {
                    id: 'vqCurve', label: '测试集 MSE 与在用码字数', sub: '紫 = VQ-VAE 的测试 MSE（左轴刻度）· 灰虚线 = PCA 取 20 / 40 个主成分 · 标题 = 测试集上用到的码字数与困惑度 exp(H)', live: true, size: [360, 190],
                    render: (c, S) => {
                        const T = S.vq.thist, P = S.pca;
                        P09.curves(c, [{ x: T.step, y: T.mse, color: LM.C.purple, width: 2, label: `VQ-VAE ${LM.fmt(T.mse[T.mse.length - 1], 4)}（用到 ${T.used[T.used.length - 1]} / ${S.vq.o.K} 个码字，困惑度 ${LM.fmt(T.perp[T.perp.length - 1], 1)}）` }, { x: [0, P09.CAP], y: [P.mse[20], P.mse[20]], color: LM.C.gray, dash: [4, 4], label: `PCA k=20：${P.mse[20].toFixed(4)}` }, { x: [0, P09.CAP], y: [P.mse[40], P.mse[40]], color: '#5d636b', dash: [2, 3], label: `PCA k=40：${P.mse[40].toFixed(4)}` }], { x: [0, P09.CAP], y: [0, 0.07], xLabel: '训练步数', legendAt: [1, 0.02] });
                    },
                },
                {
                    id: 'vqGallery', label: '码字画廊：分配给每个码字的输入窗口的平均（8 × 8）', sub: '统计最近 2048 张训练图的 16 个位置 · 斜纹 = 没有任何位置选中它（死码）· 每 50 步更新', live: true, size: [360, 250], needs: ['gallery'],
                    render: (c, S) => {
                        const Gl = S.gallery, K = S.vq.o.K, items = [];
                        for (let k = 0; k < K; k++) items.push(Gl.cnt[k] ? { img: Gl.mean, W: 8, H: 8, offset: k * 64, range: [0, 0.8] } : { img: null });
                        S._lGal = P09.tiles(c, items, K <= 32 ? 8 : K <= 128 ? 16 : 32, { gap: K > 128 ? 1 : 2, title: `${items.filter((it) => it.img).length} / ${K} 个码字被用到`, tx: 40, titleColor: LM.C.yellow });
                    },
                    probe: (x, y, S, PL, info) => { const k = P09.tilesHit(S._lGal, info); return k < 0 ? '' : `码字 ${k}：被选中 ${S.gallery.cnt[k]} 次（最近 ${S.gallery.n} 张图 × 16 个位置）`; },
                },
            ],
            notes: '前几百步只有十几个码字在用，困惑度（有效码字数）很低；训练中还会有码字「复活」，但默认 K = 128 时大部分码字一直是死的。打开 <b>死码重启</b>：红柱几乎全部消失，测试 MSE 也更低（Node 里同样 3000 步：约 40 个码字、MSE ≈ 0.020 → 128 个码字、MSE ≈ 0.016）。把 <b>码本大小 K</b> 调到 16：所有码字都在用，但每个 token 只有 4 bit，重建变差。码字画廊里能看出码字的分工：大多数是笔画片段（横、竖、弧在窗口里的不同位置），有一个专门表示空白背景。',
            warn: '和 AE / VAE 不同，这里 16 个 token 加起来最多 112 bit，远多于 VAE 的 KL（约 5 bit），所以重建要清楚得多——比较两者时要记得「信息量预算」不同。码本更新的另一种常见做法是指数滑动平均（EMA）：码字直接设为分配给它的编码器输出的滑动平均，不用码本损失（VQ-VAE 论文附录、VQ-VAE-2 用的就是它）。页面里的 JS 把同样的卷积写成 im2col + 矩阵乘（比通用卷积快得多）；C++ 版直接调用库里的 conv2d，两者是同一个计算。',
        },
        // ───────────────────────────── 10 生成 ─────────────────────────────
        {
            id: 'gen', title: '潜空间里的生成：先验 + 解码器', tag: 'GEN',
            desc: '有了潜空间，生成就是两步：<strong>从先验 p(z) 抽一个 z，交给解码器</strong>。三个模型的先验完全不同：<strong>AE</strong> 没有先验，只能事后给编码拟合一个二维高斯（Step 03），抽到的 z 有相当一部分落在洞里；<strong>VAE</strong> 训练时就把聚合后验拉向 N(0, I)，直接从 N(0, I) 抽即可；<strong>VQ-VAE</strong> 的 16 个 token 是离散的，必须<strong>另外学一个先验</strong>来描述「哪些 token 组合像数字」——论文用 PixelCNN，后来的 VQGAN、DALL·E 用 Transformer（就是第 05 篇的 GPT：把 token 序列当成句子来续写）。这里用最简单的自回归先验：按光栅顺序逐个采样 token，每个位置的分布只由计数决定——① 每个位置独立；② 看左边和上边两个邻居（回退平滑）。第一张图是四种「先验 + 解码器」采样出来的新图；第二张图定量比较 AE 和 VAE 的采样点离真实编码有多远；第三张图是测试图的 token 在三种先验下的平均信息量（bit / 图）：越低说明先验越懂数字的结构。',
            formula: [
                'p(x) = Σ_z p(z) p(x|z)      生成：z ~ p(z)，x = Decoder(z)',
                'VQ 先验：p(z_1, …, z_{16}) = Π_t p(z_t | z_{<t})      这里：p(z_t | z_{<t}) ≈ p(z_t | 左邻, 上邻)（计数 + 回退平滑）',
                'p(z | t, l, u) = (n(t, l, u, z) + α · p(z | t, l)) / (n(t, l, u) + α)，α = 1   // 没见过的上下文退回到更短的上下文',
                '−(1/N) Σ_x log_2 p(z(x))   // 测试图的 token 在先验下的平均 bit 数；均匀先验 = 16 · log₂K',
            ],
            code: {
                title: 'C++ · TokenPrior / fitPrior / priorDist / samplePrior / vqDecodeTokens / sampleFittedGaussian', src: `
                #include <unordered_map>

                // token 先验 p(z_t | 左邻 l, 上邻 u)：在训练图的 token 上计数，回退平滑
                struct TokenPrior {
                    int K = 0;
                    std::vector<float> uni;                                           // [16 × K]：每个位置的计数
                    std::unordered_map<long long, std::vector<float>> left, leftUp;   // 上下文 → K 个计数 + 1 个总数
                };

                int64_t ctxKey(int t, int l, int u, int K) { return ((int64_t)t * (K + 1) + l) * (K + 1) + u; }

                // tok：n 张图 × 16 个 token（光栅顺序）；邻居不存在时记为 K
                TokenPrior fitPrior(const std::vector<int>& tok, int n, int K) {
                    TokenPrior P;
                    P.K = K;
                    P.uni = floats(NPOS * K);
                    for (int i = 0; i < n; i++)
                        for (int t = 0; t < NPOS; t++) {
                            int z = tok[i * NPOS + t];
                            int l = t % GRID ? tok[i * NPOS + t - 1] : K, u = t >= GRID ? tok[i * NPOS + t - GRID] : K;
                            P.uni[t * K + z] += 1;
                            std::vector<float>& a = P.left[ctxKey(t, l, K, K)];
                            if (a.empty()) a = floats(K + 1);
                            a[z] += 1; a[K] += 1;
                            std::vector<float>& b = P.leftUp[ctxKey(t, l, u, K)];
                            if (b.empty()) b = floats(K + 1);
                            b[z] += 1; b[K] += 1;
                        }
                    return P;
                }

                // 第 t 个位置的条件分布：mode 0 = 每个位置独立，1 = 再看左邻，2 = 左邻 + 上邻（伪计数 α = 1 的回退平滑）
                std::vector<float> priorDist(const TokenPrior& P, int t, int l, int u, int mode) {
                    int K = P.K;
                    double nt = 0;
                    for (int z = 0; z < K; z++) nt += P.uni[t * K + z];
                    std::vector<float> p = floats(K);
                    for (int z = 0; z < K; z++) p[z] = (float)((P.uni[t * K + z] + 0.05) / (nt + 0.05 * K));
                    if (mode >= 1) {
                        auto it = P.left.find(ctxKey(t, l, K, K));
                        if (it != P.left.end()) for (int z = 0; z < K; z++) p[z] = (it->second[z] + p[z]) / (it->second[K] + 1.f);
                    }
                    if (mode >= 2) {
                        auto it = P.leftUp.find(ctxKey(t, l, u, K));
                        if (it != P.leftUp.end()) for (int z = 0; z < K; z++) p[z] = (it->second[z] + p[z]) / (it->second[K] + 1.f);
                    }
                    return p;
                }

                // 按光栅顺序逐个采样 16 个 token（自回归：每一步都以已经采好的邻居为条件）
                std::vector<int> samplePrior(const TokenPrior& P, int mode, Rng& r) {
                    std::vector<int> tok = ints(NPOS);
                    for (int t = 0; t < NPOS; t++) {
                        int l = t % GRID ? tok[t - 1] : P.K, u = t >= GRID ? tok[t - GRID] : P.K;
                        tok[t] = sampleIndex(priorDist(P, t, l, u, mode), r);
                    }
                    return tok;
                }

                // 只用 token 解码：查码本 → 解码器
                Tensor vqDecodeTokens(const VQVAE& m, const std::vector<int>& tok, int n) {
                    NoGrad ng;
                    Tensor zq = zeros(n, CODE_DIM * NPOS);
                    for (int s = 0; s < n; s++)
                        for (int p = 0; p < NPOS; p++)
                            for (int d = 0; d < CODE_DIM; d++) zq.v[s * CODE_DIM * NPOS + d * NPOS + p] = m.codebook->val.v[tok[s * NPOS + p] * CODE_DIM + d];
                    Var g = relu(conv2d(constant(zq), m.W3, m.b3, conv(CODE_DIM, GRID, GRID, 3, 1, 1)));
                    return sigmoid(convTranspose2d(g, m.W4, m.b4, CH, GRID, GRID, 1, 8, 4, 2))->val;
                }

                // AE 没有先验：事后给编码拟合二维高斯 N(m, Σ)，用 Cholesky 因子 L（Σ = L·Lᵀ）采样 z = m + L·ε
                Tensor sampleFittedGaussian(const Tensor& codes, int n, Rng& r) {
                    int N = codes.rows;
                    double mx = 0, my = 0, cxx = 0, cxy = 0, cyy = 0;
                    for (int i = 0; i < N; i++) { mx += codes.at(i, 0) / N; my += codes.at(i, 1) / N; }
                    for (int i = 0; i < N; i++) { double dx = codes.at(i, 0) - mx, dy = codes.at(i, 1) - my; cxx += dx * dx / N; cxy += dx * dy / N; cyy += dy * dy / N; }
                    double l11 = std::sqrt(cxx), l21 = cxy / l11, l22 = std::sqrt(std::max(1e-12, cyy - l21 * l21));
                    Tensor Z = zeros(n, 2);
                    for (int s = 0; s < n; s++) {
                        float e1 = randn(r), e2 = randn(r);
                        Z.at(s, 0) = (float)(mx + l11 * e1);
                        Z.at(s, 1) = (float)(my + l21 * e1 + l22 * e2);
                    }
                    return Z;
                }`,
            },
            previews: [
                {
                    id: 'genSamples', label: '四种「先验 + 解码器」采样出来的新图', sub: '每行 10 张 · AE、VAE 的 z 与下图统计用的是同一批 · VQ 的先验用最近 2048 张训练图的 token 拟合，每 25 步重新拟合、重新采样', live: true, large: true, size: [360, 300], needs: ['prior'],
                    render: (c, S) => {
                        const G = S.gen, row = (a) => P09.range(10, 0).map((k) => a.subarray(k * 256, (k + 1) * 256));
                        P09.rowsPanel(c, [{ label: 'AE\n拟合高斯', imgs: row(G.ae), labelColor: LM.C.orange }, { label: 'VAE\nN(0, I)', imgs: row(G.vae), labelColor: LM.C.green }, { label: 'VQ\n独立', imgs: row(G.vqI), labelColor: LM.C.purple }, { label: 'VQ\n左 + 上邻', imgs: row(G.vqLU), labelColor: LM.C.purple }], { title: `AE ${P09.status(S.ae)} · VAE β = ${S.beta} · VQ ${P09.status(S.vq)}`, tx: 40, titleColor: LM.C.yellow, labW: 76 });
                    },
                },
                {
                    id: 'genCoverage', label: '采样点离真实编码有多远？', sub: '300 个样本到最近测试编码的距离 ÷ 编码之间最近邻距离的中位数 · 橙 = AE + 拟合高斯，绿 = VAE + N(0, I) · 竖线 = 3 倍', live: true, size: [360, 190], needs: ['prior'],
                    render: (c, S) => {
                        const G = S.gen, NB = 24, top = 6, hist = (d) => { const H = new Float32Array(NB); for (const v of d) H[Math.min(NB - 1, Math.floor(v / top * NB))]++; return H; };
                        const xs = P09.range(NB, 0).map((b) => (b + 0.5) * top / NB);
                        P09.curves(c, [{ x: xs, y: Array.from(hist(G.cAE.d)), color: LM.C.orange, width: 2, label: `AE：${(100 * G.cAE.far).toFixed(0)}% > 3 倍` }, { x: xs, y: Array.from(hist(G.cVAE.d)), color: LM.C.green, width: 2, label: `VAE：${(100 * G.cVAE.far).toFixed(0)}% > 3 倍` }], { x: [0, top], xLabel: '距离 / 典型间距（最右一格 = ≥ 6）', marks: [{ x: 3, color: LM.C.red }], legendAt: [1, 0.02] });
                    },
                },
                {
                    id: 'genBits', label: '测试图的 16 个 token 在先验下的平均信息量', sub: 'bit / 图，越低越好 · 先验用最近 2048 张训练图拟合，在 500 张测试图的 token 上评估（真正留出的数据）', live: true, size: [360, 150], needs: ['prior'],
                    render: (c, S) => {
                        const G = S.gen;
                        LM.draw.bars(c, [G.uniform, G.bits[0], G.bits[1], G.bits[2]], [`均匀 16·log₂${G.K}`, '每个位置独立', '+ 左邻', '+ 左邻 + 上邻'], { horizontal: true, colors: [LM.C.gray, '#8d6aa8', '#a77cc4', LM.C.purple], digits: 1, valueSuffix: ' bit' });
                    },
                },
            ],
            notes: '第一行（AE）里时常能看到墨水过多或残缺的图——它们的 z 落在了洞里；第二行（VAE）几乎每张都是「某个数字」，但都很模糊（2 维、3.6 nats）。第三行（独立先验）是一堆不相干的笔画片段：每个位置的 token 单独看都合理，拼在一起却不成字；第四行加上左、上两个邻居之后，相邻的笔画开始接得上，能认出不少数字。第三张图量化了这件事（默认设置训练到第 3000 步、在 500 张测试图上实测）：每个位置独立 58.6 bit → 加左邻 53.5 bit——看到上下文，下一个 token 更好猜；再加上邻反而回到 56.7 bit：两个邻居的组合有 129² 种，2048 张图的计数太稀疏（见下面的 warn）。训练早期 token 的含义还在变，这三个数会来回跳。把这里的计数模型换成第 05 篇的 GPT，就是 VQGAN / DALL·E 的第二阶段。',
            warn: 'VQ 的先验是在<strong>正在训练</strong>的 VQ-VAE 的 token 上拟合的：训练早期 token 的含义还在变，采样会很乱，训练停止后最准确。计数模型的上下文只有两个邻居，而且 K = 128 时一个位置有 129² 种（左, 上）组合，绝大多数从没出现过，只能靠回退——这正是要换成神经网络先验（PixelCNN / Transformer）的原因：它们能共享统计规律、看到整个前缀。',
        },
        // ───────────────────────────── 11 教学版 → 生产版 ─────────────────────────────
        {
            id: 'next', title: '教学版 → 生产版：Stable Diffusion 的 VAE、VQGAN、RVQ、FSQ', tag: 'NEXT',
            desc: '这一页的三个模型都缩到了极限（16×16 灰度、2 维或 16 个 token、几万个参数），但真实系统用的正是同样的结构，只是放大并换了损失：',
            blocks: [
                {
                    type: 'html', html: `<table class="lm-tbl"><tr><th>系统</th><th>潜空间</th><th>和本页的区别</th></tr>
                    <tr><td>本页 VAE</td><td>16×16 → 2 维连续向量</td><td>MLP；BCE 重建 + β·KL</td></tr>
                    <tr><td>本页 VQ-VAE</td><td>16×16 → 4×4 个 token（K = 128）</td><td>两层卷积；BCE + 码本 + commitment；计数先验</td></tr>
                    <tr><td>VQ-VAE（van den Oord et al. 2017）</td><td>ImageNet 128×128 → 32×32 个离散 token（K = 512）</td><td>残差卷积网络；先验 = PixelCNN</td></tr>
                    <tr><td>VQ-VAE-2（Razavi et al. 2019）</td><td>两级（top / bottom）潜网格</td><td>层次化；码本用 EMA 更新</td></tr>
                    <tr><td>DALL·E（Ramesh et al. 2021）</td><td>256×256 → 32×32 个 token，词表 8192</td><td>离散 VAE（Gumbel-softmax 松弛）；先验 = 120 亿参数的 Transformer，文本 token + 图像 token 一起建模</td></tr>
                    <tr><td>VQGAN（Esser et al. 2021）</td><td>常用 f = 16：256×256 → 16×16 个 token</td><td>重建损失换成感知损失（LPIPS）+ patch 判别器的对抗损失，画面锐利得多；先验 = Transformer</td></tr>
                    <tr><td>Stable Diffusion 的 VAE（Rombach et al. 2022，LDM）</td><td>f = 8、4 个通道：512×512×3 → 64×64×4 的连续潜变量</td><td>KL 正则（权重很小，几乎是 AE）+ 感知损失 + 对抗损失；扩散模型在这个潜空间里训练（第 11 篇）</td></tr>
                    <tr><td>SoundStream / EnCodec（2021 / 2022）</td><td>音频帧 → 残差向量量化 RVQ：每一级量化上一级的残差</td><td>EnCodec 24 kHz 模型每秒 75 帧、每个码本 1024 个码字（10 bit）：1.5 / 3 / 6 / 12 / 24 kbps = 2 / 4 / 8 / 16 / 32 个码本（第 13 篇）</td></tr>
                    <tr><td>FSQ（Mentzer et al. 2024）</td><td>把几维潜变量各自四舍五入到固定的几个等级（如 8·5·5·5 = 1000 个码）</td><td>没有码本、没有码本 / commitment 损失，也就没有死码</td></tr>
                    <tr><td>视频 tokenizer（如 MAGVIT-v2，Yu et al. 2024）</td><td>时间方向也压缩的因果 3D 卷积编码器</td><td>lookup-free 量化（LFQ）；第 12 篇的视频生成建立在这类时空潜空间上</td></tr></table>` },
                {
                    type: 'previews', items: [{
                        id: 'summary', label: '本页三个模型的「信息量预算」与重建', sub: '实时读取 Step 02 / 04 / 09 的训练', live: true, size: [360, 150],
                        render: (c, S) => {
                            const ae = S.ae.res, va = S.vae.res, vq = S.vq.res, K = S.vq.o.K;
                            LM.draw.text(c, [
                                [{ t: 'PCA k = 2     ', color: LM.C.blue, bold: true }, { t: `2 个实数 · 测试 MSE ${S.pca.mse[2].toFixed(4)}` }],
                                [{ t: 'AE 2 维       ', color: LM.C.orange, bold: true }, { t: `2 个实数（信息量不受约束）· MSE ${ae.mse.toFixed(4)}` }],
                                [{ t: `VAE 2 维 β=${S.beta}  `, color: LM.C.green, bold: true }, { t: `KL ${va.kl.toFixed(2)} nats = ${(va.kl / Math.LN2).toFixed(1)} bit · MSE ${va.mse.toFixed(4)}` }],
                                [{ t: `VQ-VAE K=${K}   `, color: LM.C.purple, bold: true }, { t: `16 × log₂K = ${(16 * Math.log2(K)).toFixed(0)} bit（实际用到 ${vq.used} 个码字）· MSE ${vq.mse.toFixed(4)}` }],
                                { text: '原图：256 像素 × 8 bit = 2048 bit', color: LM.C.gray },
                            ], { px: 12, lineGap: 6 });
                        },
                    }],
                },
                { type: 'note', html: '共同的经验：① 纯像素重建损失（MSE / BCE）会让结果发糊——它惩罚的是「平均意义上的偏差」，遇到不确定就取平均；感知损失和对抗损失（VQGAN、SD 的 VAE）专门对付这一点。② 压缩率和生成质量要分开调：潜空间太小，解码器画不清；太大，后面的生成模型（Transformer / 扩散）要学的东西太多。Stable Diffusion 的论文把下采样倍数 f 从 1 试到 32，认为 4–16 之间的折中最好。③ 离散码本要防死码：k-means 初始化、EMA 更新、随机重启、降低码字维度并做 L2 归一化，或者干脆换成 FSQ。' },
                { type: 'note', html: '下一篇（第 10 篇）讲强化学习：Step 05 的 score function 估计在那里叫 REINFORCE；第 11 篇的扩散模型会在一个类似 Step 06 的连续潜空间里，学一个比 N(0, I) 强大得多的「先验」。' },
            ],
        },
    ],

    graph: {
        title: '训练一步 · VAE 与 VQ-VAE 的数据流',
        intro: '每个节点 = 一次真实的计算（在 GPU 上是一次或几次 kernel launch；这里是 JS 循环），pin = 张量名 · 形状。上面一条是 VAE（AE 相同，只是去掉采样和 KL），下面一条是 VQ-VAE；<code>θ_vae</code>、<code>θ_vq</code> 是 <b>HISTORY</b> 资源：每一步读取上一步更新后的参数（虚线回边）。PCA 只依赖数据，属于 RESIDENT。',
        types: {
            img: { color: '#6ab0e0', label: 'img · 图像 [B, 256] f32' },
            lat: { color: '#c678dd', label: 'lat · 潜变量 / 分布参数' },
            tok: { color: '#e8c25a', label: 'tok · 离散 token int' },
            param: { color: '#5ad07a', label: 'param · 可训练参数' },
            noise: { color: '#e0a060', label: 'noise · 随机数 ε' },
            scalar: { color: '#8a8f98', label: 'scalar · 损失 / 超参数' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用函数', open: false,
            blocks: LM.CPP.blocks(['autograd', 'nn']).concat([{
                label: '━ 本页公用：数据约定、batch、带 logits 的 BCE、MLP、Adam ━', code: `
                // 一张 16×16 灰度图 = 一行 256 个数（0 = 黑，1 = 白）；n 张图 = Tensor[n × 256]
                const int IMG = 16;
                const int D = IMG * IMG;

                struct DigitSet { Tensor X; std::vector<int> y; };

                // 程序化手写数字：与第 02 篇同一个生成器（笔画字形 + 随机仿射 / 粗细 / 手抖 / 噪点）
                DigitSet makeDigits(int n, Rng& r);   // 生成 n 张 16×16 数字，每类一样多、顺序打乱（实现见第 02 篇的 make_dataset）

                // double 数组（协方差这类需要累加精度的量）
                std::vector<double> doubles(int n) { return std::vector<double>((size_t)n, 0.0); }

                // 原地洗牌（Fisher–Yates，与页面 JS 的 LM.shuffle 相同）
                void shuffleInPlace(std::vector<int>& a, Rng& r) {
                    for (int i = (int)a.size() - 1; i > 0; i--) { int j = (int)(uniform(r) * (i + 1)); std::swap(a[i], a[j]); }
                }

                // 无限的 mini-batch 下标流：按 epoch 洗牌、不放回地取
                struct BatchStream { std::vector<int> perm; int pos = 0; };

                BatchStream makeStream(int n, Rng& r) {
                    BatchStream s;
                    s.perm = ints(n);
                    for (int i = 0; i < n; i++) s.perm[i] = i;
                    shuffleInPlace(s.perm, r);
                    return s;
                }

                // 下一个 batch 的 B 张图：返回 [B × 256]，idxOut 记下是哪几张
                Tensor nextBatch(BatchStream& s, const Tensor& X, int B, Rng& r, std::vector<int>& idxOut) {
                    Tensor xb = zeros(B, X.cols);
                    idxOut = ints(B);
                    for (int k = 0; k < B; k++) {
                        if (s.pos >= (int)s.perm.size()) { shuffleInPlace(s.perm, r); s.pos = 0; }
                        int i = s.perm[s.pos++];
                        idxOut[k] = i;
                        for (int j = 0; j < X.cols; j++) xb.v[k * X.cols + j] = X.v[i * X.cols + j];
                    }
                    return xb;
                }

                // 带 logits 的二元交叉熵：每张图对 256 个像素求和，再对 batch 取平均
                //   ℓ = Σ_j softplus(s_j) − x_j·s_j（= −Σ x log σ(s) + (1 − x) log(1 − σ(s))），∂ℓ/∂s = σ(s) − x
                Var bceWithLogits(const Var& S, const Tensor& X) {
                    int n = S->val.rows;
                    double tot = 0;
                    for (int i = 0; i < S->val.size(); i++) {
                        float s = S->val.v[i];
                        tot += (s > 0 ? s + std::log1p(std::exp(-s)) : std::log1p(std::exp(s))) - X.v[i] * s;
                    }
                    Tensor y = zeros(1, 1);
                    y.v[0] = (float)(tot / n);
                    return record(std::move(y), {S}, [S, X, n](Node& o) {
                        Tensor& gs = gradOf(S);
                        float k = o.grad.v[0] / n;
                        for (int i = 0; i < S->val.size(); i++) gs.v[i] += k * (1.f / (1.f + std::exp(-S->val.v[i])) - X.v[i]);
                    });
                }

                // MLP：隐藏层 ReLU，最后一层线性；He 初始化、偏置 0
                struct MLP { std::vector<Linear> layers; };

                MLP makeMLP(const std::vector<int>& sizes, Rng& r) {
                    MLP m;
                    for (size_t l = 0; l + 1 < sizes.size(); l++) m.layers.push_back(makeLinear(sizes[l], sizes[l + 1], std::sqrt(2.f / sizes[l]), r));
                    return m;
                }

                Var mlpForward(const MLP& m, Var h) {
                    for (size_t l = 0; l < m.layers.size(); l++) {
                        h = forward(m.layers[l], h);
                        if (l + 1 < m.layers.size()) h = relu(h);
                    }
                    return h;
                }

                // 把一个 MLP 的参数追加到列表里（交给优化器）
                void appendParams(std::vector<Var>& out, const MLP& m) {
                    for (const Linear& l : m.layers) { out.push_back(l.W); out.push_back(l.b); }
                }

                // 学习率：前 100 步线性 warmup，之后 cosine 衰减到 lr/20（与页面 JS 相同）
                float lrAt(int step, float lr, int total) { return cosineLR(step, lr, lr / 20, 100, total); }

                // Adam（β₁ = 0.9，β₂ = 0.999，无权重衰减）——本页三个模型都用它
                AdamW makeAdam(std::vector<Var> params, float lr) {
                    std::vector<bool> decay = std::vector<bool>(params.size(), false);
                    return makeAdamW(params, decay, lr, 0.f, 0.9f, 0.999f);
                }`,
            }]),
        },
        nodes: [
            {
                id: 'data', title: 'make_digits', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'pca', desc: '4000 + 500 张 16×16 数字',
                dispatch: { kind: 'CPU', name: 'makeDigits', count: '4500 张 × 256 像素' },
                outputs: [{ name: 'X_train', format: '[4000, 256] f32', type: 'img' }, { name: 'X_test', format: '[500, 256] f32', type: 'img' }],
                detail: { preview: 'pcaRecon' },
            },
            {
                id: 'pca', title: 'fit_pca', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'pca', desc: 'Σ = U Λ Uᵀ（256 × 256）',
                dispatch: { kind: 'CPU', name: 'fitPCA', count: 'XᵀX + Householder / QL' },
                inputs: [{ name: 'X_train', format: '[4000, 256]', type: 'img' }],
                outputs: [{ name: 'U', format: '[256, 256] f32 + μ', type: 'param' }],
                detail: { preview: 'pcaEig' },
            },
            {
                id: 'hyper', title: 'HyperParams', tag: 'CB', category: 'user', desc: 'β · K · 重启 · lr · B',
                outputs: [{ name: 'Hyper', format: 'scalars', type: 'scalar' }],
                detail: { desc: '用户控制的超参数：VAE 的 β、VQ-VAE 的码本大小 K 和是否死码重启；学习率（AE / VAE 5·10⁻³，VQ 2·10⁻³，warmup + cosine）、batch（32 / 16）和 3000 步上限固定。', preview: null },
            },
            {
                id: 'batch', title: 'next_batch', tag: 'CPU', lifetime: 'history', step: 'ae', desc: '按 epoch 洗牌取 B 张',
                dispatch: { kind: 'CPU', name: 'nextBatch', count: 'VAE：32 张 · VQ：16 张' },
                inputs: [{ name: 'X_train', format: '[4000, 256]', type: 'img' }],
                outputs: [{ name: 'Xb', format: '[B, 256] f32', type: 'img' }],
                detail: { preview: 'aeRecon' },
            },
            {
                id: 'enc', title: 'vae_encoder · MLP', tag: 'CPU', lifetime: 'history', step: 'vae', desc: '256 → 64 → 16 → [μ, log σ²]',
                dispatch: { kind: 'CPU', name: 'mlpForward(enc)', count: '32 × (256·64 + 64·16 + 16·4) MAC' },
                inputs: [{ name: 'Xb', format: '[32, 256]', type: 'img' }, { name: 'θ_vae', format: 'enc W b', type: 'param', history: true }],
                outputs: [{ name: 'μ, logσ²', format: '[32, 2] × 2', type: 'lat' }],
                detail: { preview: 'vaeScatter' },
            },
            {
                id: 'eps', title: 'sample_eps', tag: 'CPU', lifetime: 'history', step: 'reparam', desc: 'ε ~ N(0, I)',
                dispatch: { kind: 'CPU', name: 'randn', count: '32 × 2' },
                outputs: [{ name: 'ε', format: '[32, 2] f32', type: 'noise' }],
                detail: { preview: 'gradRun' },
            },
            {
                id: 'rep', title: 'reparameterize', tag: 'CPU', lifetime: 'history', step: 'reparam', desc: 'z = μ + exp(½logσ²) ⊙ ε',
                dispatch: { kind: 'CPU', name: 'reparameterize', count: '32 × 2 个元素' },
                inputs: [{ name: 'μ, logσ²', format: '[32, 2] × 2', type: 'lat' }, { name: 'ε', format: '[32, 2]', type: 'noise' }],
                outputs: [{ name: 'z', format: '[32, 2] f32', type: 'lat' }],
                detail: { preview: 'vaePost', code: 0 },
            },
            {
                id: 'dec', title: 'vae_decoder · MLP', tag: 'CPU', lifetime: 'history', step: 'vae', desc: '2 → 16 → 64 → 256 logits',
                dispatch: { kind: 'CPU', name: 'mlpForward(dec)', count: '32 × (2·16 + 16·64 + 64·256) MAC' },
                inputs: [{ name: 'z', format: '[32, 2]', type: 'lat' }, { name: 'θ_vae', format: 'dec W b', type: 'param', history: true }],
                outputs: [{ name: 'logits', format: '[32, 256] f32', type: 'img' }],
                detail: { preview: 'vaeGrid' },
            },
            {
                id: 'elbo', title: 'bce + β · kl', tag: 'CPU', lifetime: 'history', step: 'vae', desc: 'L = BCE + β·KL（−ELBO）',
                dispatch: { kind: 'CPU', name: 'bceWithLogits + klNormal', count: '32 × 256 + 32 × 2' },
                inputs: [{ name: 'logits', format: '[32, 256]', type: 'img' }, { name: 'Xb', format: '[32, 256]', type: 'img' }, { name: 'μ, logσ²', format: '[32, 2] × 2', type: 'lat' }, { name: 'Hyper', format: 'β', type: 'scalar' }],
                outputs: [{ name: 'L_vae', format: '1×1', type: 'scalar' }],
                detail: { preview: 'rdCurve' },
            },
            {
                id: 'upd', title: 'backward + adam', tag: 'CPU', category: 'output', lifetime: 'history', step: 'vae', desc: 'θ ← Adam(θ, ∇L)',
                dispatch: { kind: 'CPU', name: 'backward · adamwStep', count: (S) => `${S.vae ? S.vae.model.P.reduce((s, p) => s + p.size, 0).toLocaleString('en-US') : ''} 个参数` },
                inputs: [{ name: 'L_vae', format: '1×1', type: 'scalar' }, { name: 'θ_vae', format: '上一步', type: 'param', history: true }],
                outputs: [{ name: 'θ_vae', format: 'enc + dec', type: 'param' }],
                detail: { preview: 'vaeKL', code: 0 },
            },
            {
                id: 'vqenc', title: 'vq_encoder · Conv', tag: 'CPU', lifetime: 'history', step: 'vq', desc: 'Conv k8 s4 → ReLU → Conv k3',
                dispatch: { kind: 'CPU', name: 'conv2d × 2', count: '16 张 × 16 个位置' },
                inputs: [{ name: 'Xb', format: '[16, 256]', type: 'img' }, { name: 'θ_vq', format: 'W1 b1 W2 b2', type: 'param', history: true }],
                outputs: [{ name: 'z_e', format: '[16, 8·4·4] f32', type: 'lat' }],
                detail: { preview: 'vqRecon' },
            },
            {
                id: 'quant', title: 'vector_quantize', tag: 'CPU', lifetime: 'history', step: 'vq', desc: 'argmin ‖z_e − e_k‖² + 直通',
                dispatch: { kind: 'CPU', name: 'vectorQuantize', count: (S) => `16 × 16 个向量 × K = ${S.K} 个码字` },
                inputs: [{ name: 'z_e', format: '[16, 8·4·4]', type: 'lat' }, { name: 'θ_vq', format: '码本 [K, 8]', type: 'param', history: true }],
                outputs: [{ name: 'z_q', format: '[16, 8·4·4] f32', type: 'lat' }, { name: 'tokens', format: '[16, 16] int', type: 'tok' }],
                detail: { preview: 'vqUsage' },
            },
            {
                id: 'vqdec', title: 'vq_decoder · Conv / ConvT', tag: 'CPU', lifetime: 'history', step: 'vq', desc: 'Conv k3 → ReLU → ConvT k8 s4',
                dispatch: { kind: 'CPU', name: 'conv2d + convTranspose2d', count: '16 张' },
                inputs: [{ name: 'z_q', format: '[16, 8·4·4]', type: 'lat' }, { name: 'θ_vq', format: 'W3 b3 W4 b4', type: 'param', history: true }],
                outputs: [{ name: 'logits_vq', format: '[16, 256] f32', type: 'img' }],
                detail: { preview: 'vqTokensA' },
            },
            {
                id: 'vqloss', title: 'bce + codebook + 0.25·commit', tag: 'CPU', lifetime: 'history', step: 'vq', desc: '重建 + 码本损失 + commitment',
                dispatch: { kind: 'CPU', name: 'bceWithLogits', count: '16 × 256' },
                inputs: [{ name: 'logits_vq', format: '[16, 256]', type: 'img' }, { name: 'Xb', format: '[16, 256]', type: 'img' }, { name: 'z_e', format: '[16, 8·4·4]', type: 'lat' }, { name: 'z_q', format: '[16, 8·4·4]', type: 'lat' }],
                outputs: [{ name: 'L_vq', format: '1×1', type: 'scalar' }],
                detail: { preview: 'vqCurve' },
            },
            {
                id: 'vqupd', title: 'backward + adam + restart', tag: 'CPU', category: 'output', lifetime: 'history', step: 'vq', desc: 'θ ← Adam；死码 → 随机 z_e',
                dispatch: { kind: 'CPU', name: 'backward · adamwStep · restartDeadCodes', count: '每 25 步检查一次死码' },
                inputs: [{ name: 'L_vq', format: '1×1', type: 'scalar' }, { name: 'tokens', format: '[16, 16]', type: 'tok' }, { name: 'Hyper', format: 'K · 重启', type: 'scalar' }, { name: 'θ_vq', format: '上一步', type: 'param', history: true }],
                outputs: [{ name: 'θ_vq', format: 'enc + 码本 + dec', type: 'param' }],
                detail: { preview: 'vqGallery' },
            },
            {
                id: 'prior', title: 'fit_token_prior', tag: 'CPU', lifetime: 'frame', step: 'gen', desc: '计数：p(z_t | 左邻, 上邻)',
                dispatch: { kind: 'CPU', name: 'fitPrior', count: '最近 2048 张 × 16 个 token' },
                inputs: [{ name: 'tokens', format: '环形缓冲 [2048, 16]', type: 'tok' }],
                outputs: [{ name: 'prior', format: '计数表', type: 'scalar' }],
                detail: { preview: 'genBits' },
            },
            {
                id: 'gen', title: 'sample + decode', tag: 'CPU', category: 'output', lifetime: 'frame', step: 'gen', desc: 'z ~ 先验 → 解码器',
                dispatch: { kind: 'CPU', name: 'samplePrior · vqDecodeTokens', count: '每种 10 张' },
                inputs: [{ name: 'prior', format: '计数表', type: 'scalar' }, { name: 'θ_vq', format: '码本 + dec', type: 'param' }],
                outputs: [{ name: 'samples', format: '[10, 256] f32', type: 'img' }],
                detail: { preview: 'genSamples' },
            },
        ],
    },

    references: [
        { title: 'On Lines and Planes of Closest Fit to Systems of Points in Space', author: 'K. Pearson', venue: 'Philosophical Magazine 2(11)', year: 1901, note: '主成分分析的起源' },
        { title: 'The approximation of one matrix by another of lower rank', author: 'C. Eckart, G. Young', venue: 'Psychometrika 1', year: 1936, note: '截断 SVD / PCA 是最优的低秩近似' },
        { title: 'Neural networks and principal component analysis: Learning from examples without local minima', author: 'P. Baldi, K. Hornik', venue: 'Neural Networks 2(1)', year: 1989, note: '线性自编码器（MSE）学到的就是主子空间' },
        { title: 'Reducing the Dimensionality of Data with Neural Networks', author: 'G. E. Hinton, R. R. Salakhutdinov', venue: 'Science 313', year: 2006, note: '深层自编码器 vs PCA；MNIST 的 2 维编码' },
        { title: 'Auto-Encoding Variational Bayes', author: 'D. P. Kingma, M. Welling', venue: 'ICLR', year: 2014, url: 'https://arxiv.org/abs/1312.6114', note: 'VAE、重参数化技巧、MNIST 的 2 维流形图' },
        { title: 'Stochastic Backpropagation and Approximate Inference in Deep Generative Models', author: 'D. J. Rezende, S. Mohamed, D. Wierstra', venue: 'ICML', year: 2014, note: '同时独立提出的 VAE' },
        { title: 'An Introduction to Variational Autoencoders', author: 'D. P. Kingma, M. Welling', venue: 'Foundations and Trends in Machine Learning 12(4)', year: 2019, note: 'VAE 的系统综述' },
        { title: 'Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning', author: 'R. J. Williams', venue: 'Machine Learning 8', year: 1992, note: 'REINFORCE / score function 估计' },
        { title: 'ELBO surgery: yet another way to carve up the variational evidence lower bound', author: 'M. D. Hoffman, M. J. Johnson', venue: 'NIPS Workshop on Advances in Approximate Bayesian Inference', year: 2016, note: '平均 KL = 互信息 + KL(聚合后验 ‖ 先验)' },
        { title: 'β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework', author: 'I. Higgins et al.', venue: 'ICLR', year: 2017, note: 'β > 1 与解耦表示' },
        { title: 'Fixing a Broken ELBO', author: 'A. A. Alemi, B. Poole, I. Fischer, J. V. Dillon, R. A. Saurous, K. Murphy', venue: 'ICML', year: 2018, url: 'https://arxiv.org/abs/1711.00464', note: '率-失真视角下的 VAE' },
        { title: 'Generating Sentences from a Continuous Space', author: 'S. R. Bowman, L. Vilnis, O. Vinyals, A. M. Dai, R. Jozefowicz, S. Bengio', venue: 'CoNLL', year: 2016, note: '后验坍缩；KL 退火与 word dropout' },
        { title: 'Improved Variational Inference with Inverse Autoregressive Flow', author: 'D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever, M. Welling', venue: 'NeurIPS', year: 2016, note: 'free bits' },
        { title: 'The continuous Bernoulli: fixing a pervasive error in variational autoencoders', author: 'G. Loaiza-Ganem, J. P. Cunningham', venue: 'NeurIPS', year: 2019, note: '对灰度图用 BCE 的问题' },
        { title: 'Neural Discrete Representation Learning', author: 'A. van den Oord, O. Vinyals, K. Kavukcuoglu', venue: 'NeurIPS', year: 2017, url: 'https://arxiv.org/abs/1711.00937', note: 'VQ-VAE：码本、直通估计、commitment 损失、PixelCNN 先验' },
        { title: 'Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation', author: 'Y. Bengio, N. Léonard, A. Courville', venue: 'arXiv:1308.3432', year: 2013, url: 'https://arxiv.org/abs/1308.3432', note: '直通估计（straight-through estimator）' },
        { title: 'Generating Diverse High-Fidelity Images with VQ-VAE-2', author: 'A. Razavi, A. van den Oord, O. Vinyals', venue: 'NeurIPS', year: 2019, note: '层次化 VQ-VAE、EMA 码本更新' },
        { title: 'Jukebox: A Generative Model for Music', author: 'P. Dhariwal, H. Jun, C. Payne, J. W. Kim, A. Radford, I. Sutskever', venue: 'arXiv:2005.00341', year: 2020, url: 'https://arxiv.org/abs/2005.00341', note: '码本的 random restart' },
        { title: 'Conditional Image Generation with PixelCNN Decoders', author: 'A. van den Oord, N. Kalchbrenner, O. Vinyals, L. Espeholt, A. Graves, K. Kavukcuoglu', venue: 'NeurIPS', year: 2016, note: 'VQ-VAE 用的自回归先验' },
        { title: 'Zero-Shot Text-to-Image Generation', author: 'A. Ramesh et al.', venue: 'ICML', year: 2021, url: 'https://arxiv.org/abs/2102.12092', note: 'DALL·E：dVAE（32×32 token、8192 词表）+ Transformer 先验' },
        { title: 'Taming Transformers for High-Resolution Image Synthesis', author: 'P. Esser, R. Rombach, B. Ommer', venue: 'CVPR', year: 2021, url: 'https://arxiv.org/abs/2012.09841', note: 'VQGAN：感知 + 对抗损失、Transformer 先验' },
        { title: 'High-Resolution Image Synthesis with Latent Diffusion Models', author: 'R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer', venue: 'CVPR', year: 2022, url: 'https://arxiv.org/abs/2112.10752', note: 'Stable Diffusion 的潜空间（f = 8、KL / VQ 正则）' },
        { title: 'SoundStream: An End-to-End Neural Audio Codec', author: 'N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, M. Tagliasacchi', venue: 'IEEE/ACM TASLP 30', year: 2022, url: 'https://arxiv.org/abs/2107.03312', note: '残差向量量化（RVQ）' },
        { title: 'High Fidelity Neural Audio Compression', author: 'A. Défossez, J. Copet, G. Synnaeve, Y. Adi', venue: 'TMLR', year: 2023, url: 'https://arxiv.org/abs/2210.13438', note: 'EnCodec：1.5–24 kbps 的 RVQ codec' },
        { title: 'Finite Scalar Quantization: VQ-VAE Made Simple', author: 'F. Mentzer, D. Minnen, E. Agustsson, M. Tschannen', venue: 'ICLR', year: 2024, url: 'https://arxiv.org/abs/2309.15505', note: 'FSQ：逐维四舍五入代替码本' },
        { title: 'Language Model Beats Diffusion — Tokenizer is Key to Visual Generation', author: 'L. Yu et al.', venue: 'ICLR', year: 2024, note: 'MAGVIT-v2：因果 3D tokenizer、lookup-free 量化' },
        { title: 'Tutorial on Variational Autoencoders', author: 'C. Doersch', venue: 'arXiv:1606.05908', year: 2016, url: 'https://arxiv.org/abs/1606.05908', note: '面向入门读者的 VAE 推导' },
    ],
});

