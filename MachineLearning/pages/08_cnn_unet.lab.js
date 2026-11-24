// 08_cnn_unet.lab.js — 系列第 08 篇：卷积神经网络，从卷积核到 U-Net（PyTorch）
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js 08_cnn_unet.lab.js -o 08_cnn_unet.html
// 页面里的每张预览都由下面的 JS 参考实现实时算出：同一批程序化数字（与第 02 篇逐像素相同）上的固定卷积核、
// 实时训练的 CNN / MLP / U-Net、用反向传播量出的有效感受野、平移后的测试集、深层网络的梯度流。

const P08 = (() => {
    'use strict';
    const { AG } = LM;
    const IMG = 16, D = 256, K = 10, NTRAIN = 4000, NTEST = 500;

    // ═══════════════════ 数据：与第 02 篇同一个生成器、同一个种子 → 同一批 4000 + 500 张数字 ═══════════════════
    const range = (a, b) => Int32Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
    function makeData(rand) {
        const tr = LM.digits.makeSet(NTRAIN, rand), te = LM.digits.makeSet(NTEST, rand);
        const byClass = Array.from({ length: K }, () => []);
        for (let i = 0; i < te.n; i++) byClass[te.y[i]].push(i);
        return { tr, te, byClass, allTr: range(0, NTRAIN), allTe: range(0, NTEST), evalTr: range(0, 500) };
    }
    /** 整数像素平移（空出来的地方补 0）：dx > 0 向右，dy > 0 向下。src / dst 都是 C 个 H×W 平面 */
    function shiftImg(src, dx, dy, dst, C, H, W) {
        C = C || 1; H = H || IMG; W = W || IMG;
        for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const sx = x - dx, sy = y - dy;
            dst[c * H * W + y * W + x] = sx >= 0 && sx < W && sy >= 0 && sy < H ? src[c * H * W + sy * W + sx] : 0;
        }
        return dst;
    }

    // ═══════════════════ 算子：快速「same」卷积（自定义可微算子，经过 AG.conv2d 对照与有限差分校验） ═══════════════════
    // 共享库的 AG.conv2d 是通用的朴素实现（任意 stride / pad）；训练用的 stride 1、pad = (K−1)/2 卷积在这里换成扁平 padded 布局：
    // 输入先补零成 (H+2p)×(W+2p)，输出按 padded 行布局算（每行多出 2p 个丢弃的位置），最内层循环就是一段连续内存上的 9 个乘加。
    // 结果与 AG.conv2d 逐元素一致（float32 舍入级别的差异，见 tools/p08_conv_check.js），在本页的训练里快约 4 倍。
    function conv(X, Wt, b, o) {
        const N = X.r, C = o.C, H = o.H, W = o.W, Kk = o.K, p = (Kk - 1) >> 1, Co = Wt.r;
        const Wp = W + 2 * p, Hp = H + 2 * p, P = Hp * Wp, L = (H - 1) * Wp + W, KK = Kk * Kk, HW = H * W;
        const x = X.data, w = Wt.data, xp = new Float32Array(N * C * P);
        for (let nc = 0; nc < N * C; nc++) { const so = nc * HW, po = nc * P + p * Wp + p; for (let y = 0; y < H; y++) xp.set(x.subarray(so + y * W, so + y * W + W), po + y * Wp); }
        const offs = new Int32Array(KK); for (let ky = 0; ky < Kk; ky++) for (let kx = 0; kx < Kk; kx++) offs[ky * Kk + kx] = ky * Wp + kx;
        const Y = new AG.Tensor(N, Co * HW), y = Y.data, acc = new Float32Array(L), acc2 = new Float32Array(L);
        for (let n = 0; n < N; n++) for (let co = 0; co < Co; co += 2) {   // 一次算两个输出通道：每读一次输入做 18 个乘加
            const two = co + 1 < Co;
            acc.fill(b ? b.data[co] : 0); acc2.fill(two && b ? b.data[co + 1] : 0);
            for (let ci = 0; ci < C; ci++) {
                const xo = (n * C + ci) * P, wo = (co * C + ci) * KK, vo = wo + C * KK;
                if (Kk === 3) {
                    const r1 = xo + Wp, r2 = xo + 2 * Wp;
                    const w0 = w[wo], w1 = w[wo + 1], w2 = w[wo + 2], w3 = w[wo + 3], w4 = w[wo + 4], w5 = w[wo + 5], w6 = w[wo + 6], w7 = w[wo + 7], w8 = w[wo + 8];
                    if (two) {
                        const v0 = w[vo], v1 = w[vo + 1], v2 = w[vo + 2], v3 = w[vo + 3], v4 = w[vo + 4], v5 = w[vo + 5], v6 = w[vo + 6], v7 = w[vo + 7], v8 = w[vo + 8];
                        for (let q = 0; q < L; q++) {
                            const a0 = xp[xo + q], a1 = xp[xo + q + 1], a2 = xp[xo + q + 2], a3 = xp[r1 + q], a4 = xp[r1 + q + 1], a5 = xp[r1 + q + 2], a6 = xp[r2 + q], a7 = xp[r2 + q + 1], a8 = xp[r2 + q + 2];
                            acc[q] += w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4 + w5 * a5 + w6 * a6 + w7 * a7 + w8 * a8;
                            acc2[q] += v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3 + v4 * a4 + v5 * a5 + v6 * a6 + v7 * a7 + v8 * a8;
                        }
                    } else {
                        for (let q = 0; q < L; q++) acc[q] += w0 * xp[xo + q] + w1 * xp[xo + q + 1] + w2 * xp[xo + q + 2] + w3 * xp[r1 + q] + w4 * xp[r1 + q + 1] + w5 * xp[r1 + q + 2] + w6 * xp[r2 + q] + w7 * xp[r2 + q + 1] + w8 * xp[r2 + q + 2];
                    }
                } else {
                    for (let t = 0; t < KK; t++) {
                        const s = xo + offs[t], wv = w[wo + t], vv = two ? w[vo + t] : 0;
                        for (let q = 0; q < L; q++) { const a = xp[s + q]; acc[q] += wv * a; acc2[q] += vv * a; }
                    }
                }
            }
            const yo = (n * Co + co) * HW;
            for (let oy = 0; oy < H; oy++) for (let ox = 0; ox < W; ox++) { y[yo + oy * W + ox] = acc[oy * Wp + ox]; if (two) y[yo + HW + oy * W + ox] = acc2[oy * Wp + ox]; }
        }
        Y.shape = { C: Co, H, W };
        return AG.track(Y, [X, Wt, b], (g) => {
            const gq = new Float32Array(N * Co * L);   // 输出梯度写成 padded 行布局（丢弃列 = 0）
            for (let nc = 0; nc < N * Co; nc++) { const go = nc * HW, qo = nc * L; for (let oy = 0; oy < H; oy++) gq.set(g.subarray(go + oy * W, go + oy * W + W), qo + oy * Wp); }
            if (b && b.req) { const gb = b.g(); for (let n = 0; n < N; n++) for (let co = 0; co < Co; co++) { let s = 0; const go = (n * Co + co) * HW; for (let i = 0; i < HW; i++) s += g[go + i]; gb[co] += s; } }
            if (Wt.req) {   // dW[co, ci, t] = Σ_n Σ_q g'[q] · xp[q + offs[t]]（ReLU / maxpool 之后大部分 g' 是 0，跳过）
                const gw = Wt.g();
                for (let n = 0; n < N; n++) for (let co = 0; co < Co; co++) {
                    const qo = (n * Co + co) * L;
                    for (let ci = 0; ci < C; ci++) {
                        const xo = (n * C + ci) * P, wo = (co * C + ci) * KK;
                        if (Kk === 3) {
                            const r1 = xo + Wp, r2 = xo + 2 * Wp;
                            let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0, s8 = 0;
                            for (let q = 0; q < L; q++) {
                                const gv = gq[qo + q]; if (gv === 0) continue;
                                s0 += gv * xp[xo + q]; s1 += gv * xp[xo + q + 1]; s2 += gv * xp[xo + q + 2];
                                s3 += gv * xp[r1 + q]; s4 += gv * xp[r1 + q + 1]; s5 += gv * xp[r1 + q + 2];
                                s6 += gv * xp[r2 + q]; s7 += gv * xp[r2 + q + 1]; s8 += gv * xp[r2 + q + 2];
                            }
                            gw[wo] += s0; gw[wo + 1] += s1; gw[wo + 2] += s2; gw[wo + 3] += s3; gw[wo + 4] += s4; gw[wo + 5] += s5; gw[wo + 6] += s6; gw[wo + 7] += s7; gw[wo + 8] += s8;
                        } else {
                            for (let t = 0; t < KK; t++) { const s = xo + offs[t]; let a = 0; for (let q = 0; q < L; q++) { const gv = gq[qo + q]; if (gv !== 0) a += gv * xp[s + q]; } gw[wo + t] += a; }
                        }
                    }
                }
            }
            if (X.req) {   // dX：每个非零的 g'[q] 把 w·g' 散射回它读过的 K×K 个输入位置（padded 坐标），最后裁掉边框
                const gx = X.g(), gxp = new Float32Array(N * C * P);
                for (let n = 0; n < N; n++) for (let co = 0; co < Co; co++) {
                    const qo = (n * Co + co) * L;
                    for (let q = 0; q < L; q++) {
                        const gv = gq[qo + q]; if (gv === 0) continue;
                        for (let ci = 0; ci < C; ci++) {
                            const wo = (co * C + ci) * KK, o0 = (n * C + ci) * P + q;
                            if (Kk === 3) {
                                const o1 = o0 + Wp, o2 = o0 + 2 * Wp;
                                gxp[o0] += w[wo] * gv; gxp[o0 + 1] += w[wo + 1] * gv; gxp[o0 + 2] += w[wo + 2] * gv;
                                gxp[o1] += w[wo + 3] * gv; gxp[o1 + 1] += w[wo + 4] * gv; gxp[o1 + 2] += w[wo + 5] * gv;
                                gxp[o2] += w[wo + 6] * gv; gxp[o2 + 1] += w[wo + 7] * gv; gxp[o2 + 2] += w[wo + 8] * gv;
                            } else {
                                for (let t = 0; t < KK; t++) gxp[o0 + offs[t]] += w[wo + t] * gv;
                            }
                        }
                    }
                }
                for (let nc = 0; nc < N * C; nc++) { const xo = nc * HW, po = nc * P + p * Wp + p; for (let yy = 0; yy < H; yy++) for (let xx = 0; xx < W; xx++) gx[xo + yy * W + xx] += gxp[po + yy * Wp + xx]; }
            }
        });
    }

    // ═══════════════════ 模型：参数表 + 前向（全部用 LM.AG 张量，反向自动） ═══════════════════
    /** He 初始化的卷积 / 全连接参数；PyTorch 的 nn.Linear 存 [out × in]，这里存 [in × out]（AG.linear 的约定），数值上完全等价 */
    function P(params, name, r, c, init) { const t = AG.param(name, r, c, init); params.push(t); return t; }
    const convP = (ps, name, cin, cout, k, rand) => ({ w: P(ps, name + '.weight', cout, cin * k * k, LM.init.he(rand, cin * k * k)), b: P(ps, name + '.bias', 1, cout), cin, cout, k });
    const linP = (ps, name, din, dout, rand) => ({ w: P(ps, name + '.weight', din, dout, LM.init.he(rand, din)), b: P(ps, name + '.bias', 1, dout), din, dout });
    const numParams = (ps) => ps.reduce((s, p) => s + p.size, 0);

    /** 小 CNN：conv(1→c1, 3×3) → ReLU → maxpool → conv(c1→c2, 3×3) → ReLU → maxpool → flatten(c2·4·4) → linear → 10；head = 'flat' | 'gap' */
    function makeCNN(cfg, rand) {
        const c1 = cfg.c1 || 8, c2 = cfg.c2 || 16, ps = [];
        const m = { kind: 'cnn', cfg: Object.assign({ c1, c2, head: 'flat' }, cfg), params: ps };
        m.conv1 = convP(ps, 'conv1', 1, c1, 3, rand);
        m.conv2 = convP(ps, 'conv2', c1, c2, 3, rand);
        m.fc = m.cfg.head === 'gap' ? linP(ps, 'fc', c2, K, rand) : linP(ps, 'fc', c2 * 16, K, rand);
        return m;
    }
    function cnnForward(m, X, tr) {
        const c1 = m.cfg.c1, c2 = m.cfg.c2;
        const z1 = conv(X, m.conv1.w, m.conv1.b, { C: 1, H: 16, W: 16, K: 3 }), a1 = AG.relu(z1), p1 = AG.maxPool2(a1, { C: c1, H: 16, W: 16 });
        const z2 = conv(p1, m.conv2.w, m.conv2.b, { C: c1, H: 8, W: 8, K: 3 }), a2 = AG.relu(z2), p2 = AG.maxPool2(a2, { C: c2, H: 8, W: 8 });
        const f = m.cfg.head === 'gap' ? gap(p2, c2, 16) : p2;
        const logits = AG.linear(f, m.fc.w, m.fc.b);
        if (tr) Object.assign(tr, { z1, a1, p1, z2, a2, p2, f, logits });
        return logits;
    }
    /** 全局平均池化 [N × C·HW] → [N × C] */
    function gap(X, C, HW) {
        const N = X.r, Y = new AG.Tensor(N, C), x = X.data, y = Y.data;
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { let s = 0; const o = (n * C + c) * HW; for (let k = 0; k < HW; k++) s += x[o + k]; y[n * C + c] = s / HW; }
        return AG.track(Y, [X], (g) => { const gx = X.g(); for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const v = g[n * C + c] / HW, o = (n * C + c) * HW; for (let k = 0; k < HW; k++) gx[o + k] += v; } });
    }
    /**
     * BatchNorm2d：每个通道在 (N, H, W) 上求均值 / 方差做标准化，再乘 γ 加 β。o: { C, train（默认 true：用 batch 统计量并更新 stats）, stats: { mean, var }（滑动平均，动量 0.1，与 PyTorch 相同） }
     * 推理（train = false）用滑动平均。返回张量带 .mu / .rstd（每个通道）。
     */
    function batchNorm(X, gamma, beta, o) {
        const N = X.r, C = o.C, HW = X.c / C, M = N * HW, eps = 1e-5, x = X.data, Y = new AG.Tensor(N, X.c), y = Y.data;
        const xh = new Float32Array(x.length), rstd = new Float32Array(C), mus = new Float32Array(C), train = o.train !== false, st = o.stats;
        for (let c = 0; c < C; c++) {
            let mu, v;
            if (train || !st) {
                let s = 0; for (let n = 0; n < N; n++) { const of = (n * C + c) * HW; for (let k = 0; k < HW; k++) s += x[of + k]; }
                mu = s / M; let s2 = 0;
                for (let n = 0; n < N; n++) { const of = (n * C + c) * HW; for (let k = 0; k < HW; k++) { const d = x[of + k] - mu; s2 += d * d; } }
                v = s2 / M;
                if (st && train) { st.mean[c] = 0.9 * st.mean[c] + 0.1 * mu; st.var[c] = 0.9 * st.var[c] + 0.1 * v * M / Math.max(1, M - 1); }   // 滑动方差用无偏估计（PyTorch 的约定）
            } else { mu = st.mean[c]; v = st.var[c]; }
            const rs = 1 / Math.sqrt(v + eps), g = gamma.data[c], b = beta.data[c];
            rstd[c] = rs; mus[c] = mu;
            for (let n = 0; n < N; n++) { const of = (n * C + c) * HW; for (let k = 0; k < HW; k++) { const h = (x[of + k] - mu) * rs; xh[of + k] = h; y[of + k] = g * h + b; } }
        }
        Y.mu = mus; Y.rstd = rstd;
        return AG.track(Y, [X, gamma, beta], (gy) => {
            const gg = gamma.req ? gamma.g() : null, gb = beta.req ? beta.g() : null, gx = X.req ? X.g() : null;
            for (let c = 0; c < C; c++) {
                let sg = 0, sgh = 0;
                for (let n = 0; n < N; n++) { const of = (n * C + c) * HW; for (let k = 0; k < HW; k++) { sg += gy[of + k]; sgh += gy[of + k] * xh[of + k]; } }
                if (gg) gg[c] += sgh;
                if (gb) gb[c] += sg;
                if (!gx) continue;
                const g = gamma.data[c], rs = rstd[c];
                if (train || !st) { const a = g * rs / M; for (let n = 0; n < N; n++) { const of = (n * C + c) * HW; for (let k = 0; k < HW; k++) gx[of + k] += a * (M * gy[of + k] - sg - xh[of + k] * sgh); } }
                else for (let n = 0; n < N; n++) { const of = (n * C + c) * HW; for (let k = 0; k < HW; k++) gx[of + k] += g * rs * gy[of + k]; }
            }
        });
    }

    // ═══════════════════ 深层 CNN：普通堆叠 vs 残差连接（可选 BatchNorm） ═══════════════════
    /**
     * stem：conv(1→c) + ReLU + maxpool → c×8×8；然后 L 个块，每块一个 3×3 卷积 f(x) = [BN](conv(x))：
     *   plain：x ← ReLU(f(x))        res：x ← ReLU(x + f(x))（He et al. 2016 的后激活残差块，简化成每块一个卷积）
     * 最后 maxpool → c×4×4 → flatten → linear → 10。
     */
    function makeDeep(cfg, rand) {
        const L = cfg.L, c = cfg.c || 4, ps = [];
        const m = { kind: 'deep', cfg: { L, c, res: !!cfg.res, bn: !!cfg.bn, zero: !!cfg.zero }, params: ps, blocks: [] };
        m.stem = convP(ps, 'stem', 1, c, 3, rand);
        for (let l = 0; l < L; l++) {
            const blk = { conv: convP(ps, `block${l + 1}.conv`, c, c, 3, rand) };
            // zero：残差分支初始化为 0（有 BN 时把分支 BN 的 γ 置 0——Goyal et al. 2017；没有 BN 时把卷积权重置 0——Fixup / SkipInit 的思路），每个块一开始都是恒等映射
            const zero = cfg.zero && cfg.res;
            if (zero && !cfg.bn) blk.conv.w.data.fill(0);
            if (cfg.bn) blk.bn = { g: P(ps, `block${l + 1}.bn.weight`, 1, c, (a) => a.fill(zero ? 0 : 1)), b: P(ps, `block${l + 1}.bn.bias`, 1, c), stats: { mean: new Float32Array(c), var: new Float32Array(c).fill(1) } };
            m.blocks.push(blk);
        }
        m.fc = linP(ps, 'fc', c * 16, K, rand);
        return m;
    }
    function deepForward(m, X, tr, train) {
        const { c, res } = m.cfg;
        let x = AG.maxPool2(AG.relu(conv(X, m.stem.w, m.stem.b, { C: 1, H: 16, W: 16, K: 3 })), { C: c, H: 16, W: 16 });
        const acts = tr ? [x] : null;
        m.blocks.forEach((blk) => {
            let f = conv(x, blk.conv.w, blk.conv.b, { C: c, H: 8, W: 8, K: 3 });
            if (blk.bn) f = batchNorm(f, blk.bn.g, blk.bn.b, { C: c, train: train !== false, stats: blk.bn.stats });
            x = AG.relu(res ? AG.add(x, f) : f);
            if (acts) acts.push(x);
        });
        const logits = AG.linear(AG.maxPool2(x, { C: c, H: 8, W: 8 }), m.fc.w, m.fc.b);
        if (tr) Object.assign(tr, { acts, logits });
        return logits;
    }
    /** MLP：256 → H → 10（与第 02 篇的默认结构相同：ReLU、He 初始化、偏置 0） */
    function makeMLP(cfg, rand) {
        const H = cfg.hidden || 64, ps = [];
        const m = { kind: 'mlp', cfg: Object.assign({ hidden: H }, cfg), params: ps };
        m.fc1 = linP(ps, 'fc1', D, H, rand); m.fc2 = linP(ps, 'fc2', H, K, rand);
        return m;
    }
    function mlpForward(m, X, tr) {
        const z1 = AG.linear(X, m.fc1.w, m.fc1.b), h1 = AG.relu(z1), logits = AG.linear(h1, m.fc2.w, m.fc2.b);
        if (tr) Object.assign(tr, { z1, h1, logits });
        return logits;
    }
    const forwardOf = (m) => (m.kind === 'cnn' ? cnnForward : m.kind === 'deep' ? deepForward : mlpForward);

    // ═══════════════════ U-Net（两级）与去掉跳连的编码器-解码器：去噪，直接预测干净图像 ═══════════════════
    /**
     * enc1(1→c1) ↓ enc2(c1→c2) ↓ mid(c2→c2) ↑ [⊕ enc2] dec(→c1) ↑ [⊕ enc1] out(→1)；skip = false 时去掉两条 ⊕。激活默认 SiLU：
     * 用 ReLU 时只有 4 个通道的 dec 那一级在训练中几乎全部「死掉」，置零它 PSNR 不变（tools/p08_unet_act.js）。
     * 输出层是一个线性的 3×3 卷积，直接读 [↑h₂ ; s₁]：跳连的特征一定能到达输出（最初的版本在输出前还有一层只有 4 个通道的
     * 3×3 卷积 + ReLU，某些种子下这 4 个通道在训练初期全部「死掉」，输出变成常数——见 tools/p08_proto.js useed）。
     */
    function makeUNet(cfg, rand) {
        const c1 = cfg.c1 || 4, c2 = cfg.c2 || 8, skip = cfg.skip !== false, act = cfg.act || 'silu', ps = [];
        const m = { kind: 'unet', cfg: { c1, c2, skip, act }, params: ps };
        m.e1 = convP(ps, 'enc1', 1, c1, 3, rand);
        m.e2 = convP(ps, 'enc2', c1, c2, 3, rand);
        m.mid = convP(ps, 'mid', c2, c2, 3, rand);
        m.d2 = convP(ps, 'dec', skip ? 2 * c2 : c2, c1, 3, rand);
        m.out = convP(ps, 'out', skip ? 2 * c1 : c1, 1, 3, rand);
        return m;
    }
    function unetForward(m, X, tr) {
        const { c1, c2, skip } = m.cfg, cv = (x, l, C, H) => conv(x, l.w, l.b, { C, H, W: H, K: l.k }), f = m.cfg.act === 'relu' ? AG.relu : AG.silu;
        const s1 = f(cv(X, m.e1, 1, 16));                                         // [c1 × 16 × 16]
        const s2 = f(cv(AG.maxPool2(s1, { C: c1, H: 16, W: 16 }), m.e2, c1, 8));   // [c2 × 8 × 8]
        const bt = f(cv(AG.maxPool2(s2, { C: c2, H: 8, W: 8 }), m.mid, c2, 4));    // [c2 × 4 × 4] 瓶颈
        let u2 = AG.upsample2(bt, { C: c2, H: 4, W: 4 });
        if (skip) u2 = AG.concatCols(u2, s2);                                             // 通道拼接：一张图 = 一行，所以按列拼
        const h2 = f(cv(u2, m.d2, skip ? 2 * c2 : c2, 8));                         // [c1 × 8 × 8]
        let u1 = AG.upsample2(h2, { C: c1, H: 8, W: 8 });
        if (skip) u1 = AG.concatCols(u1, s1);
        const y = cv(u1, m.out, skip ? 2 * c1 : c1, 16);                                 // [1 × 16 × 16]：预测的干净图像
        if (tr) Object.assign(tr, { s1, s2, bt, u2, h2, u1, y });
        return y;
    }
    /** 去噪训练 run：每步取 B 张训练图 x₀，加新的高斯噪声 x = x₀ + σ·ε，最小化 MSE(f(x), x₀) */
    function makeDenoiseRun(cfg) {
        const m = cfg.model, R = {
            cfg, m, step: 0, cap: cfg.cap, B: cfg.B, rand: LM.rng(cfg.seed), noiseRand: LM.rng(cfg.seed ^ 0x6c8e9cf5),
            opt: new LM.AdamW(m.params, { lr: cfg.lr, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0 }),
            perm: Int32Array.from(cfg.data.allTr), pos: 0, X0: new Float32Array(cfg.B * D), Xn: new Float32Array(cfg.B * D), batchLoss: [],
        };
        LM.shuffle(R.perm, R.rand);
        return R;
    }
    function denoiseStep(R) {
        const X = R.cfg.data.tr.X, sg = R.cfg.sigma;
        for (let k = 0; k < R.B; k++) {
            if (R.pos >= R.perm.length) { LM.shuffle(R.perm, R.rand); R.pos = 0; }
            const i = R.perm[R.pos++];
            for (let j = 0; j < D; j++) { const v = X[i * D + j]; R.X0[k * D + j] = v; R.Xn[k * D + j] = v + sg * LM.randn(R.noiseRand); }
        }
        R.opt.zeroGrad();
        const y = unetForward(R.m, AG.tensor(R.B, D, R.Xn)), loss = AG.mse(y, AG.tensor(R.B, D, R.X0));
        AG.backward(loss);
        R.lr = R.cfg.cosine ? LM.cosineLR(R.step, { maxLR: R.cfg.lr, minLR: 0.05 * R.cfg.lr, warmup: 0, total: R.cap }) : R.cfg.lr;
        R.opt.step(R.lr);
        R.step++;
        R.batchLoss.push(loss.item());
        return loss.item();
    }
    /** 固定的含噪测试集：n 张测试图 + 固定种子的噪声（每个 σ 一份） */
    function noisySet(data, n, sigma, seed) {
        const r = LM.rng(seed), X0 = new Float32Array(n * D), Xn = new Float32Array(n * D);
        for (let i = 0; i < n * D; i++) { X0[i] = data.te.X[i]; Xn[i] = X0[i] + sigma * LM.randn(r); }
        return { n, X0, Xn, sigma };
    }
    const mseOf = (a, b, n) => { let s = 0; for (let i = 0; i < n; i++) s += (a[i] - b[i]) ** 2; return s / n; };
    const psnr = (mse) => 10 * Math.log10(1 / Math.max(mse, 1e-12));
    function denoiseEval(m, ns, n) {
        n = Math.min(n || ns.n, ns.n);
        const y = AG.noGrad(() => unetForward(m, AG.tensor(n, D, ns.Xn.subarray(0, n * D)))).data;
        return { mse: mseOf(y, ns.X0, n * D), y };
    }

    // ═══════════════════ 训练 run：按 epoch 洗牌取 batch（可选随机平移增强）→ 前向 → 交叉熵 → 反向 → Adam ═══════════════════
    /** cfg: { model, lr, cosine（lr 余弦衰减到 5%）, B, cap, seed, aug（平移增强的最大像素数，0 = 不增强）, data, evalSets: [{ name, X, y, idx }] } */
    function makeRun(cfg) {
        const rand = LM.rng(cfg.seed), m = cfg.model;
        const R = {
            cfg, m, rand, fwd: forwardOf(m), step: 0, cap: cfg.cap, B: cfg.B,
            opt: new LM.AdamW(m.params, { lr: cfg.lr, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0 }),
            perm: Int32Array.from(cfg.data.allTr), pos: 0, Xb: new Float32Array(cfg.B * D), yb: new Int32Array(cfg.B), tmp: new Float32Array(D),
            batchLoss: [], hist: { step: [] }, ev: null, augRand: LM.rng(cfg.seed ^ 0x2545f491),
        };
        LM.shuffle(R.perm, R.rand);
        cfg.evalSets.forEach((s) => { R.hist[s.name + 'L'] = []; R.hist[s.name + 'A'] = []; });
        return R;
    }
    function nextBatch(R) {
        const X = R.cfg.data.tr.X, y = R.cfg.data.tr.y, a = R.cfg.aug || 0;
        for (let k = 0; k < R.B; k++) {
            if (R.pos >= R.perm.length) { LM.shuffle(R.perm, R.rand); R.pos = 0; }
            const i = R.perm[R.pos++], src = X.subarray(i * D, (i + 1) * D);
            if (a > 0) {
                const dx = Math.floor(R.augRand() * (2 * a + 1)) - a, dy = Math.floor(R.augRand() * (2 * a + 1)) - a;
                shiftImg(src, dx, dy, R.Xb.subarray(k * D, (k + 1) * D));
            } else R.Xb.set(src, k * D);
            R.yb[k] = y[i];
        }
    }
    function trainStep(R) {
        nextBatch(R);
        R.opt.zeroGrad();
        const logits = R.fwd(R.m, AG.tensor(R.B, D, R.Xb), R.trace);   // R.trace（可选）：记录本 batch 的中间激活
        const loss = AG.crossEntropy(logits, R.yb);
        AG.backward(loss);
        R.lr = R.cfg.cosine ? LM.cosineLR(R.step, { maxLR: R.cfg.lr, minLR: 0.05 * R.cfg.lr, warmup: 0, total: R.cap }) : R.cfg.lr;
        R.opt.step(R.lr);
        R.step++;
        R.batchLoss.push(loss.item());
        return loss.item();
    }
    /** 无梯度批量推理：返回 logits（Float32Array，n × 10） */
    function predict(m, X, n, tr) { return AG.noGrad(() => forwardOf(m)(m, AG.tensor(n, D, X.subarray(0, n * D)), tr, false).data); }
    /** 评估：分块推理 → loss / 准确率（同步版，Node 原型与终点评估用） */
    function evaluate(m, X, y, idx, shift) {
        const CH = 50, buf = new Float32Array(CH * D);
        let loss = 0, ok = 0;
        for (let s = 0; s < idx.length; s += CH) {
            const n = Math.min(CH, idx.length - s);
            for (let k = 0; k < n; k++) { const i = idx[s + k], src = X.subarray(i * D, (i + 1) * D); if (shift) shiftImg(src, shift[0], shift[1], buf.subarray(k * D, (k + 1) * D)); else buf.set(src, k * D); }
            const Z = predict(m, buf, n);
            for (let k = 0; k < n; k++) { const t = y[idx[s + k]], pk = LM.argmax(Z, k * K, K); ok += pk === t ? 1 : 0; let mx = -Infinity; for (let j = 0; j < K; j++) mx = Math.max(mx, Z[k * K + j]); let z = 0; for (let j = 0; j < K; j++) z += Math.exp(Z[k * K + j] - mx); loss += Math.log(z) + mx - Z[k * K + t]; }
        }
        return { loss: loss / idx.length, acc: ok / idx.length };
    }

    // ═══════════════════ 分帧评估（与第 02 篇同样的做法：先拷一份权重快照，再每帧评估若干张，评估完一轮记一个点） ═══════════════════
    function cloneModel(m) {
        const r = LM.rng(1), c = m.kind === 'cnn' ? makeCNN(m.cfg, r) : m.kind === 'mlp' ? makeMLP(m.cfg, r) : m.kind === 'deep' ? makeDeep(m.cfg, r) : makeUNet(m.cfg, r);
        copyModel(c, m);
        return c;
    }
    function copyModel(dst, src) {
        dst.params.forEach((p, i) => p.data.set(src.params[i].data));
        if (src.blocks) src.blocks.forEach((b, l) => { if (b.bn) { dst.blocks[l].bn.stats.mean.set(b.bn.stats.mean); dst.blocks[l].bn.stats.var.set(b.bn.stats.var); } });
    }
    const ceRow = (Z, k, t) => { let mx = -Infinity; for (let j = 0; j < K; j++) mx = Math.max(mx, Z[k * K + j]); let z = 0; for (let j = 0; j < K; j++) z += Math.exp(Z[k * K + j] - mx); return Math.log(z) + mx - Z[k * K + t]; };
    const EVAL_CHUNK = 20;
    function evalBegin(R) {
        const ev = R.ev || (R.ev = { m: cloneModel(R.m), buf: new Float32Array(EVAL_CHUNK * D) });
        copyModel(ev.m, R.m);
        ev.step = R.step; ev.set = 0; ev.pos = 0; ev.pending = true;
        ev.acc = R.cfg.evalSets.map(() => ({ loss: 0, ok: 0, n: 0 }));
        ev.conf = R.cfg.evalSets.map((s) => (s.conf ? new Int32Array(K * K) : null));
        ev.pred = R.cfg.evalSets.map((s) => (s.conf ? new Int8Array(s.idx.length) : null));
    }
    function evalWork(R, budget) {
        const ev = R.ev, sets = R.cfg.evalSets;
        while (budget > 0 && ev.set < sets.length) {
            const s = sets[ev.set], n = Math.min(EVAL_CHUNK, budget, s.idx.length - ev.pos);
            for (let k = 0; k < n; k++) { const i = s.idx[ev.pos + k]; ev.buf.set(s.X.subarray(i * D, (i + 1) * D), k * D); }
            const Z = predict(ev.m, ev.buf, n), a = ev.acc[ev.set];
            for (let k = 0; k < n; k++) {
                const t = s.y[s.idx[ev.pos + k]], pk = LM.argmax(Z, k * K, K);
                a.loss += ceRow(Z, k, t); a.ok += pk === t ? 1 : 0; a.n++;
                if (ev.conf[ev.set]) { ev.conf[ev.set][t * K + pk]++; ev.pred[ev.set][ev.pos + k] = pk; }
            }
            ev.pos += n; budget -= n;
            if (ev.pos >= s.idx.length) { ev.set++; ev.pos = 0; }
        }
        if (ev.set < sets.length) return false;
        const H = R.hist; H.step.push(ev.step);
        sets.forEach((s, k) => { H[s.name + 'L'].push(ev.acc[k].loss / ev.acc[k].n); H[s.name + 'A'].push(ev.acc[k].ok / ev.acc[k].n); });
        R.conf = ev.conf; R.pred = ev.pred; R.evalStep = ev.step; ev.pending = false;
        return true;
    }
    function evalNow(R) { evalBegin(R); evalWork(R, 1e9); }
    /** 每帧：训练 spf 步 + 推进分帧评估（每帧 budget 张）；到达上限时立刻做一次完整评估，之后什么都不做 */
    function tick(R, spf, budget) {
        if (R.step >= R.cap) return;
        for (let k = 0; k < spf && R.step < R.cap; k++) trainStep(R);
        if (!R.cfg.evalSets.length) return;                 // 没有评估集的 run（Step 07 的增强版本由平移测试统一评估）
        if (R.step >= R.cap) { evalNow(R); return; }
        if (!R.ev || !R.ev.pending) evalBegin(R);
        evalWork(R, budget);
    }

    // ═══════════════════ 页面用的训练配置 ═══════════════════
    const CAP = 1500, LR = 3e-3, BATCH = 32, EVAL_BUDGET = 20;
    const evalSetsMain = (d) => [{ name: 'train', X: d.tr.X, y: d.tr.y, idx: d.evalTr }, { name: 'test', X: d.te.X, y: d.te.y, idx: d.allTe, conf: true }];
    /** Step 05 的主训练：CNN 与 MLP 用同一串 batch（同一个洗牌种子）、同一个优化器与学习率日程 */
    function mainRuns(S) {
        const d = S.data;
        const cnn = makeRun({ model: makeCNN({ c1: 8, c2: 16 }, LM.rng(S.seed + 81)), lr: LR, cosine: true, B: BATCH, cap: CAP, seed: S.seed + 7, data: d, evalSets: evalSetsMain(d) });
        const mlp = makeRun({ model: makeMLP({ hidden: 64 }, LM.rng(S.seed + 82)), lr: LR, cosine: true, B: BATCH, cap: CAP, seed: S.seed + 7, data: d, evalSets: evalSetsMain(d) });
        evalNow(cnn); evalNow(mlp);
        return { cnn, mlp };
    }
    /** Step 07：同样的两个模型、同样的初始权重和 batch 顺序，只是训练时每张图随机平移 ±aug 像素 */
    function augRuns(S, aug) {
        const d = S.data, mk = (model) => makeRun({ model, lr: LR, cosine: true, B: BATCH, cap: CAP, seed: S.seed + 7, aug, data: d, evalSets: [] });
        return { aug, cnn: mk(makeCNN({ c1: 8, c2: 16 }, LM.rng(S.seed + 81))), mlp: mk(makeMLP({ hidden: 64 }, LM.rng(S.seed + 82))) };
    }
    // 平移测试：dx = −4…4（dy = 0）与 dy = −3…3（dx = 0）；训练中每轮用前 40 张测试图，四个模型都训练完以后最后用前 100 张评估一轮
    const SHIFTS = []; for (let v = -4; v <= 4; v++) SHIFTS.push([v, 0]); for (let v = -3; v <= 3; v++) if (v) SHIFTS.push([0, v]);
    const NSHIFT = 100;
    function shiftEvalBegin(E, runs, nImg) {
        E.snap = runs.map((R, k) => { const m = E.snap && E.snap[k] && E.snap[k].kind === R.m.kind ? E.snap[k] : cloneModel(R.m); copyModel(m, R.m); return m; });
        E.steps = runs.map((R) => R.step); E.pos = 0; E.nImg = nImg || NSHIFT; E.ok = runs.map(() => new Int32Array(SHIFTS.length)); E.pending = true;
    }
    /** 交错顺序：第 p 对 = 第 ⌊p / 15⌋ 张图 × 第 p mod 15 个平移量——评估到一半时，每个平移量都已经有同样多的样本（第一轮就能画出完整但更嘈杂的曲线） */
    function shiftEvalWork(E, data, budget) {
        const nS = SHIFTS.length, total = nS * E.nImg, buf = E.buf || (E.buf = new Float32Array(EVAL_CHUNK * D)), sh = E.sh || (E.sh = new Int32Array(EVAL_CHUNK)), im = E.im || (E.im = new Int32Array(EVAL_CHUNK));
        while (budget > 0 && E.pos < total) {
            const n = Math.min(EVAL_CHUNK, budget, total - E.pos);
            for (let k = 0; k < n; k++) { const p = E.pos + k, s = p % nS, i = Math.floor(p / nS); shiftImg(data.te.X.subarray(i * D, (i + 1) * D), SHIFTS[s][0], SHIFTS[s][1], buf.subarray(k * D, (k + 1) * D)); sh[k] = s; im[k] = i; }
            E.snap.forEach((m, j) => { const Z = predict(m, buf, n); for (let k = 0; k < n; k++) if (LM.argmax(Z, k * K, K) === data.te.y[im[k]]) E.ok[j][sh[k]]++; });
            E.pos += n; budget -= n;
        }
        if (E.pos < total) return false;
        E.res = { steps: E.steps.slice(), acc: E.ok.map((o) => Array.from(o, (v) => v / E.nImg)), n: E.nImg, partial: false };
        E.pending = false;
        return true;
    }
    /** 要画的结果：最近一轮完整的评估；第一轮还没评估完时，用这一轮已经评估过的图（每个平移量 n 张）的准确率 */
    function shiftView(E) {
        if (E.res) return E.res;
        const nS = SHIFTS.length, full = Math.floor(E.pos / nS), cnt = SHIFTS.map((_, s) => full + (s < E.pos % nS ? 1 : 0));
        if (!cnt[0]) return null;
        return { steps: E.steps.slice(), acc: E.ok.map((o) => Array.from(o, (v, s) => v / Math.max(1, cnt[s]))), n: full, partial: true };
    }
    /** 从 SHIFTS 的结果里取出水平 / 竖直两条曲线 */
    function shiftCurve(acc, axis) {
        const xs = [], ys = [];
        SHIFTS.forEach(([dx, dy], s) => { if (axis === 'x' ? dy === 0 : dx === 0) { xs.push(axis === 'x' ? dx : dy); ys.push(acc[s]); } });
        const o = xs.map((v, i) => i).sort((a, b) => xs[a] - xs[b]);
        return { x: o.map((i) => xs[i]), y: o.map((i) => ys[i]) };
    }

    // ═══════════════════ Step 08：U-Net 去噪的训练与分帧评估 ═══════════════════
    const UCAP = 1200, UB = 8, UN_EVAL = 48, UN_SHOW = 8, UN_CHUNK = 8;
    function blurBaseline(ns) {   // 最好的固定高斯模糊（3×3 或 5×5 二项式核）——经典去噪的对照组
        const b3 = [1, 2, 1], b5 = [1, 4, 6, 4, 1];
        let best = Infinity;
        [[b3, 3, 16], [b5, 5, 256]].forEach(([b, k, s]) => {
            const w = new Float32Array(k * k); for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) w[i * k + j] = b[i] * b[j] / s;
            const y = AG.noGrad(() => conv(AG.tensor(ns.n, D, ns.Xn), AG.tensor(1, k * k, w), null, { C: 1, H: IMG, W: IMG, K: k })).data;
            best = Math.min(best, mseOf(y, ns.X0, y.length));
        });
        return best;
    }
    function unetRuns(S, sigma) {
        const d = S.data, mk = (skip) => makeDenoiseRun({ model: makeUNet({ c1: 4, c2: 8, skip }, LM.rng(S.seed + 91)), lr: LR, cosine: true, B: UB, cap: UCAP, seed: S.seed + 13, sigma, data: d });
        const U = { sigma, runs: [mk(true), mk(false)], ns: noisySet(d, UN_EVAL, sigma, S.seed + 17), hist: { step: [], psnr: [[], []] }, ev: null, show: null };
        U.base = { noisy: psnr(mseOf(U.ns.Xn, U.ns.X0, U.ns.Xn.length)), blur: psnr(blurBaseline(U.ns)) };
        unetEvalBegin(U); unetEvalWork(U, 1e9);
        return U;
    }
    function unetEvalBegin(U) {
        const ev = U.ev || (U.ev = { m: U.runs.map((R) => cloneModel(R.m)), y: U.runs.map(() => new Float32Array(UN_EVAL * D)) });
        ev.m.forEach((m, k) => copyModel(m, U.runs[k].m));
        ev.step = U.runs[0].step; ev.pos = 0; ev.pending = true;
    }
    function unetEvalWork(U, budget) {
        const ev = U.ev, ns = U.ns;
        while (budget > 0 && ev.pos < UN_EVAL) {
            const n = Math.min(UN_CHUNK, budget, UN_EVAL - ev.pos), X = AG.tensor(n, D, ns.Xn.subarray(ev.pos * D, (ev.pos + n) * D));
            ev.m.forEach((m, k) => ev.y[k].set(AG.noGrad(() => unetForward(m, X)).data, ev.pos * D));
            ev.pos += n; budget -= n;
        }
        if (ev.pos < UN_EVAL) return false;
        U.hist.step.push(ev.step);
        ev.y.forEach((y, k) => U.hist.psnr[k].push(psnr(mseOf(y, ns.X0, UN_EVAL * D))));
        U.show = { step: ev.step, y: ev.y.map((y) => Float32Array.from(y.subarray(0, UN_SHOW * D))) };
        ev.pending = false;
        return true;
    }
    function unetTick(U, spf) {
        const done = U.runs.every((R) => R.step >= R.cap);
        if (done && U.show && U.show.step >= U.runs[0].cap) return;
        for (let k = 0; k < spf; k++) U.runs.forEach((R) => { if (R.step < R.cap) denoiseStep(R); });
        if (!U.ev.pending) unetEvalBegin(U);
        unetEvalWork(U, U.runs[0].step >= U.runs[0].cap ? 1e9 : UN_CHUNK);
    }

    // ═══════════════════ Step 09：深层普通网络 vs 残差网络 ═══════════════════
    const DCAP = 800, DB = 8, DEEP_C = 4;
    function deepRuns(S, L, bn) {
        const d = S.data, mk = (res) => {
            const R = makeRun({ model: makeDeep({ L, c: DEEP_C, res, bn, zero: res }, LM.rng(S.seed + 101)), lr: LR, cosine: true, B: DB, cap: DCAP, seed: S.seed + 19, data: d, evalSets: [{ name: 'test', X: d.te.X, y: d.te.y, idx: d.allTe }] });
            R.trace = {}; R.lossEMA = []; R.accEMA = []; R.ema = null; R.aema = null;
            return R;
        };
        return { L, bn, runs: [mk(false), mk(true)] };
    }
    /** 深层网络的一步：训练 + 记录 loss / batch 准确率的滑动平均 + 本 batch 每层激活的 RMS 与 0 的比例 */
    function deepTick(G, spf) {
        G.runs.forEach((R) => {
            for (let k = 0; k < spf && R.step < R.cap; k++) {
                const l = trainStep(R), Z = R.trace.logits.data;
                let ok = 0; for (let i = 0; i < R.B; i++) ok += LM.argmax(Z, i * K, K) === R.yb[i] ? 1 : 0;
                R.ema = R.ema == null ? l : 0.97 * R.ema + 0.03 * l; R.aema = R.aema == null ? ok / R.B : 0.97 * R.aema + 0.03 * ok / R.B;
                R.lossEMA.push(R.ema); R.accEMA.push(R.aema);
                R.actStats = R.trace.acts.map((a) => { let z = 0; for (let i = 0; i < a.data.length; i++) z += a.data[i] === 0 ? 1 : 0; return { rms: LM.rms(a.data), zero: z / a.data.length }; });
                if (R.step >= R.cap) evalNow(R);
            }
        });
    }
    /** 初始化时（还没训练）的信号传播：每层激活的 RMS 与 ∂L/∂x_l 的 RMS（一个 batch 的一次前向 + 反向） */
    function deepFlow(S, L, bn) {
        const X = AG.tensor(32, D, S.data.tr.X.subarray(0, 32 * D)), y = S.data.tr.y.subarray(0, 32);
        return [{ res: false, zero: false }, { res: true, zero: false }, { res: true, zero: true }].map((v) => {
            const m = makeDeep({ L, c: DEEP_C, res: v.res, bn, zero: v.zero }, LM.rng(S.seed + 101)), tr = {};
            AG.clearTape();
            const loss = AG.crossEntropy(deepForward(m, X, tr, true), y);
            AG.backward(loss);
            return { act: tr.acts.map((a) => LM.rms(a.data)), grad: tr.acts.map((a) => LM.rms(a.grad || new Float32Array(1))), loss: loss.item() };
        });
    }

    // ═══════════════════ Step 01：固定卷积核 ═══════════════════
    const KERNELS = [
        { id: 'sobelx', name: 'Sobel x', note: '左右亮度差 → 竖直笔画的边缘', w: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
        { id: 'sobely', name: 'Sobel y', note: '上下亮度差 → 水平笔画的边缘', w: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
        { id: 'diag', name: '45° 边缘', note: '沿对角方向的亮度差', w: [0, 1, 2, -1, 0, 1, -2, -1, 0] },
        { id: 'laplace', name: '拉普拉斯', note: '中心减四邻：二阶导，笔画中线为负、两侧为正', w: [0, 1, 0, 1, -4, 1, 0, 1, 0] },
        { id: 'blur', name: '高斯模糊', note: '加权平均：去掉细节和噪点', w: [1, 2, 1, 2, 4, 2, 1, 2, 1].map((v) => v / 16) },
        { id: 'sharpen', name: '锐化', note: '原图 + 拉普拉斯的负值：边缘更陡', w: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
    ];
    const kernelById = (id) => KERNELS.find((k) => k.id === id) || KERNELS[0];
    /** 一次 conv 调用把 6 个固定核都作用在同一张图上（C_out = 6） */
    function applyKernels(img) {
        const W = AG.tensor(KERNELS.length, 9, Float32Array.from(KERNELS.flatMap((k) => k.w)));
        return AG.noGrad(() => conv(AG.tensor(1, D, img), W, null, { C: 1, H: IMG, W: IMG, K: 3 })).data;
    }
    /** Step 03：任意核大小 K（奇数）的 Sobel-x 型边缘核 = 二项式平滑（竖直）⊗ 二项式差分（水平），K = 3 时就是 Sobel x */
    function sobelK(k) {
        const binom = (n) => { let r = [1]; for (let i = 0; i < n - 1; i++) { const t = [1]; for (let j = 1; j < r.length; j++) t.push(r[j - 1] + r[j]); t.push(1); r = t; } return r; };
        const s = binom(k), b = binom(k - 1), dv = s.map((_, i) => (b[i - 1] || 0) - (b[i] || 0)), w = new Float32Array(k * k);
        let a = 0; for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { w[i * k + j] = s[i] * dv[j]; a += Math.abs(w[i * k + j]); }
        for (let i = 0; i < w.length; i++) w[i] /= a / 2;
        return w;
    }
    const CONV_CFGS = [
        { id: 'k3s1p1', K: 3, s: 1, p: 1, label: '3×3 · s=1 · p=1（same）' },
        { id: 'k3s1p0', K: 3, s: 1, p: 0, label: '3×3 · s=1 · p=0（valid）' },
        { id: 'k3s2p1', K: 3, s: 2, p: 1, label: '3×3 · s=2 · p=1' },
        { id: 'k5s1p2', K: 5, s: 1, p: 2, label: '5×5 · s=1 · p=2' },
        { id: 'k5s2p2', K: 5, s: 2, p: 2, label: '5×5 · s=2 · p=2' },
        { id: 'k7s2p3', K: 7, s: 2, p: 3, label: '7×7 · s=2 · p=3（ResNet 第一层）' },
    ];
    const outSize = (H, K, s, p) => Math.floor((H + 2 * p - K) / s) + 1;

    // ═══════════════════ Step 02：平移等变（卷积）vs 没有结构（MLP 隐藏层） ═══════════════════
    const conv1Maps = (m, img) => AG.noGrad(() => AG.relu(conv(AG.tensor(1, D, img), m.conv1.w, m.conv1.b, { C: 1, H: IMG, W: IMG, K: 3 }))).data;
    const mlpHidden = (m, img) => AG.noGrad(() => AG.relu(AG.linear(AG.tensor(1, D, img), m.fc1.w, m.fc1.b))).data;
    const relDiff = (a, b, mask) => { let s = 0, n = 0; for (let i = 0; i < a.length; i++) { if (mask && !mask[i]) continue; s += (a[i] - b[i]) ** 2; n += b[i] * b[i]; } return Math.sqrt(s / Math.max(n, 1e-12)); };
    const cosSim = (a, b) => { let ab = 0, aa = 0, bb = 0; for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; } return ab / Math.sqrt(Math.max(aa * bb, 1e-24)); };
    /**
     * 比较 F(T x) 与 T F(x) 的位置：两边都没有用到画面外的补零（3×3 卷积读 ±1 像素），也不在移进来的那条边上。
     * 在这些位置上两者逐位相同；差异只出现在边框：那里一边读到补的 0，另一边读到被移出画面的笔画（数字常常贴着边）。
     */
    function validMask(dx, dy, C) {
        const m = new Uint8Array(C * D), ok = (v) => v >= 1 && v <= IMG - 2;
        for (let c = 0; c < C; c++) for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) m[c * D + y * IMG + x] = ok(x) && ok(y) && ok(x - dx) && ok(y - dy) ? 1 : 0;
        return m;
    }
    function equivDemo(S) {
        const cnn = S.main.cnn.m, mlp = S.main.mlp.m, i = S.sample, x = S.data.te.X.subarray(i * D, (i + 1) * D), dx = S.dx, dy = S.dy, C = cnn.cfg.c1;
        const Tx = shiftImg(x, dx, dy, new Float32Array(D)), F = conv1Maps(cnn, x), FT = conv1Maps(cnn, Tx), TF = shiftImg(F, dx, dy, new Float32Array(C * D), C);
        const diff = new Float32Array(C * D), mask = validMask(dx, dy, C);
        let dIn = 0, dAll = 0; for (let k = 0; k < C * D; k++) { diff[k] = FT[k] - TF[k]; dAll = Math.max(dAll, Math.abs(diff[k])); if (mask[k]) dIn = Math.max(dIn, Math.abs(diff[k])); }
        let ch = 0, best = -1; for (let c = 0; c < C; c++) { let s = 0; for (let k = 0; k < D; k++) s += F[c * D + k]; if (s > best) { best = s; ch = c; } }
        const h = mlpHidden(mlp, x), hT = mlpHidden(mlp, Tx);
        return { i, x, Tx, F, FT, TF, diff, ch, C, dIn, dAll, h, hT, hCos: cosSim(h, hT), hRel: relDiff(hT, h), fRel: relDiff(FT, F), fRelAligned: relDiff(FT, TF, mask), step: S.main.cnn.step };
    }
    /** 平移 k = 0…4 像素（水平）时表示变了多少：40 张测试图的平均相对变化 */
    function equivCurve(S) {
        const cnn = S.main.cnn.m, mlp = S.main.mlp.m, C = cnn.cfg.c1, ks = [0, 1, 2, 3, 4], out = { k: ks, mlp: [], cnn: [], cnnAligned: [], step: S.main.cnn.step };
        const n = 40, Tx = new Float32Array(D), TF = new Float32Array(C * D);
        ks.forEach((k) => {
            const mask = validMask(k, 0, C); let a = 0, b = 0, c = 0;
            for (let i = 0; i < n; i++) {
                const x = S.data.te.X.subarray(i * D, (i + 1) * D); shiftImg(x, k, 0, Tx);
                a += relDiff(mlpHidden(mlp, Tx), mlpHidden(mlp, x));
                const F = conv1Maps(cnn, x), FT = conv1Maps(cnn, Tx); shiftImg(F, k, 0, TF, C);
                b += relDiff(FT, F); c += relDiff(FT, TF, mask);
            }
            out.mlp.push(a / n); out.cnn.push(b / n); out.cnnAligned.push(c / n);
        });
        return out;
    }

    // ═══════════════════ Step 04：感受野——用反向传播量出一个输出单元「看」多大一块输入 ═══════════════════
    const ERF_LS = [1, 2, 3, 4, 6, 8, 10], ERF_N = 32, ERF_C = 4;
    /** 随机 He 初始化的 ReLU 卷积堆（ERF_C 通道）与 1/9 均匀核线性堆，在 32×32 的高斯噪声输入上求 |∂y_中心 / ∂x| */
    function erfMaps(seed) {
        const rand = LM.rng(seed), HW = ERF_N * ERF_N, nImg = 8, c0 = (ERF_N / 2) * ERF_N + ERF_N / 2;
        const layers = []; for (let l = 0; l < 10; l++) { const cin = l ? ERF_C : 1; layers.push(AG.param('erf' + l, ERF_C, cin * 9, LM.init.he(rand, cin * 9))); }
        const X0 = new Float32Array(nImg * HW); for (let i = 0; i < X0.length; i++) X0[i] = LM.randn(rand);
        const box = AG.tensor(1, 9, new Float32Array(9).fill(1 / 9)), res = { rand: [], box: [], rRand: [], rBox: [] };
        const radius = (g) => { let s = 0, s2 = 0; for (let y = 0; y < ERF_N; y++) for (let x = 0; x < ERF_N; x++) { const v = g[y * ERF_N + x], r2 = (y - ERF_N / 2) ** 2 + (x - ERF_N / 2) ** 2; s += v; s2 += v * r2; } return Math.sqrt(s2 / Math.max(s, 1e-30)); };
        ERF_LS.forEach((L) => {
            AG.clearTape();
            const X = AG.param('x', nImg, HW); X.data.set(X0);
            let h = X;
            for (let l = 0; l < L; l++) h = AG.relu(conv(h, layers[l], null, { C: l ? ERF_C : 1, H: ERF_N, W: ERF_N, K: 3 }));
            const sel = AG.tensor(nImg, ERF_C * HW); for (let n = 0; n < nImg; n++) for (let c = 0; c < ERF_C; c++) sel.data[n * ERF_C * HW + c * HW + c0] = 1;
            AG.backward(AG.sum(AG.mul(h, sel)));   // y = Σ_n Σ_c h[n, c, 中心]：每张图的梯度只来自它自己的中心单元
            const g = new Float32Array(HW); for (let n = 0; n < nImg; n++) for (let k = 0; k < HW; k++) g[k] += Math.abs(X.grad[n * HW + k]) / nImg;
            res.rand.push(g); res.rRand.push(radius(g));
            AG.clearTape();
            const Xb = AG.param('xb', 1, HW); let hb = Xb;
            for (let l = 0; l < L; l++) hb = conv(hb, box, null, { C: 1, H: ERF_N, W: ERF_N, K: 3 });
            const sb = AG.tensor(1, HW); sb.data[c0] = 1;
            AG.backward(AG.sum(AG.mul(hb, sb)));
            const gb = Float32Array.from(Xb.grad, Math.abs); res.box.push(gb); res.rBox.push(radius(gb));
        });
        return res;
    }
    /** 训练中的 CNN：conv2 在 8×8 特征图中心 (4, 4) 的 16 个通道之和，对选中样本的输入梯度（理论感受野 = 输入的 [5, 12]²） */
    function cnnERF(m, img) {
        AG.clearTape();
        const X = AG.param('x', 1, D); X.data.set(img);
        const tr = {}; cnnForward(m, X, tr);
        const sel = AG.tensor(1, tr.z2.c), c2 = m.cfg.c2; for (let c = 0; c < c2; c++) sel.data[c * 64 + 4 * 8 + 4] = 1;
        AG.backward(AG.sum(AG.mul(tr.z2, sel)));
        return Float32Array.from(X.grad);
    }

    // ═══════════════════ 绘图辅助（深色主题、设计坐标；画布文字 ≥ 11 px） ═══════════════════
    let offC = null;
    /** 把 W×H 的浮点图画到 (x, y, w, h)；stride / offset 用来直接读权重矩阵的一列或多通道特征图的一个通道 */
    function drawImg(ctx, data, x, y, w, h, o) {
        o = o || {};
        const W = o.W || IMG, H = o.H || IMG;
        if (!offC) offC = document.createElement('canvas');
        if (offC.width !== W || offC.height !== H) { offC.width = W; offC.height = H; }
        const octx = offC.getContext('2d'), im = octx.createImageData(W, H), map = PL.cmap[o.map || 'gray'], rgb = [0, 0, 0];
        const lo = o.range ? o.range[0] : 0, hi = o.range ? o.range[1] : 1, st = o.stride || 1, of = o.offset || 0, sp = hi - lo || 1;
        for (let i = 0; i < W * H; i++) {
            let t = (data[of + i * st] - lo) / sp; t = t < 0 ? 0 : t > 1 ? 1 : t === t ? t : 0; map(t, rgb);
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
    /** 小图网格：items = [{ img, W, H, stride, offset, range, map, label, labelColor, outline }]。o: { cols, map, range, labels, header, title, titleColor, tx, gap, top, cellMax } */
    function imgGrid(c, items, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const n = items.length, cols = o.cols || 10, rows = Math.max(1, Math.ceil(n / cols)), gap = o.gap ?? 3;
        const top = o.top ?? ((o.title ? 18 : 0) + (o.header ? 15 : 0)), lab = o.labels ? 14 : 0;
        const cell = Math.max(4, Math.floor(Math.min((w - gap) / cols - gap, (h - top - gap) / rows - gap - lab, o.cellMax || 1e9)));
        const x0 = Math.floor((w - (cols * (cell + gap) - gap)) / 2), y0 = top + gap;
        if (o.title) txt(ctx, o.title, o.tx ?? 6, 3, { bold: true, color: o.titleColor });
        if (o.header) o.header.forEach((s, k) => txt(ctx, String(s), x0 + k * (cell + gap) + cell / 2, y0 - 2, { align: 'center', base: 'bottom', color: LM.C.gray }));
        items.forEach((it, k) => {
            const cx = x0 + (k % cols) * (cell + gap), cy = y0 + Math.floor(k / cols) * (cell + gap + lab);
            drawImg(ctx, it.img, cx, cy, cell, cell, { W: it.W, H: it.H, map: it.map || o.map, range: it.range || o.range, stride: it.stride, offset: it.offset });
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
    /** 竖直柱状图。o: { colors, colorFor(v, i), range, symmetric, ticks, title, titleColor, tx, top, values, valueMin, digits, hi, labelColor(i) } */
    function barChart(c, vals, labels, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c);
        const n = vals.length, top = o.top ?? (o.title ? 24 : 22), padB = labels ? 18 : 6, padL = 44, padR = 6, dg = o.digits ?? 2;
        let lo = o.range ? o.range[0] : Math.min(0, ...vals), hi = o.range ? o.range[1] : Math.max(1e-9, ...vals);
        if (o.symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -m; hi = m; }
        if (!o.range && o.values) { const sp = hi - lo; if (hi > 0) hi += 0.16 * sp; if (lo < 0) lo -= 0.16 * sp; }
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
                const sv = Math.abs(v) < 0.5 * Math.pow(10, -dg) ? (0).toFixed(dg) : v.toFixed(dg);
                txt(ctx, sv, x + bw / 2, v >= 0 ? Math.min(z, y1) - 2 : Math.max(z, y1) + 2, { align: 'center', base: v >= 0 ? 'bottom' : 'top', color: LM.C.text });
            }
        }
        return { padL, bw, top, ph, lo, hi };
    }
    /** 多行文字面板：rows = [[{ t, color, bold }…] | { text, color }]，只在中文标点 / 空格后换行；o: { px, x0（第一行起点，避开 LIVE 角标）, indent } */
    function lines(c, rows, o) {
        o = o || {};
        const { ctx, w } = PL.draw.ctx(c), px = o.px || 11, lh = px + 5;
        let y = o.y0 ?? 6, first = true;
        rows.forEach((row) => {
            const segs = Array.isArray(row) ? row : [{ t: row.text, color: row.color, bold: row.bold }];
            let x = first ? (o.x0 ?? 6) : 6;
            segs.forEach((sg) => {
                ctx.font = LM.FONT(px, sg.bold);
                String(sg.t).split(/(?<=[，、；·：）])|(?<= )(?!\d)/).forEach((piece) => {   // 只在中文标点后、或后面不是数字的空格处换行
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
        const lw = Math.max(...items.map((it) => ctx.measureText(it.label).width)) + 26, lh = items.length * 15 + 6, x0 = align === 'left' ? x : align === 'center' ? x - lw / 2 : x - lw;
        ctx.fillStyle = 'rgba(8,8,10,0.86)'; ctx.fillRect(x0, y, lw, lh);
        items.forEach((it, k) => {
            const yy = y + 4 + k * 15;
            if (!it.noLine) { ctx.strokeStyle = it.color; ctx.lineWidth = 2; ctx.setLineDash(it.dash || []); ctx.beginPath(); ctx.moveTo(x0 + 5, yy + 7); ctx.lineTo(x0 + 19, yy + 7); ctx.stroke(); ctx.setLineDash([]); }
            txt(ctx, it.label, x0 + (it.noLine ? 6 : 23), yy, { color: it.color });
        });
    }
    /** 折线图 + 可放在任意位置的图例。series: [{ x, y, color, dash, width, label }]；o = PL.draw.plot 的选项 + legendAt: [u, v]、legendAlign、extra（额外图例行） */
    function curves(c, series, o) {
        PL.draw.plot(c, Object.assign({}, o, { series: series.map((s) => ({ data: s.y, x: s.x, color: s.color, dash: s.dash, width: s.width })) }));
        const items = series.filter((s) => s.label).map((s) => ({ label: s.label, color: s.color, dash: s.dash })).concat(o.extra || []);
        if (!items.length) return;
        const { ctx, w, h } = PL.draw.ctx(c, { keep: true });
        const padL = 50, padR = 10, padT = 10, padB = o.xLabel ? 34 : 21, at = o.legendAt || [1, 0], al = o.legendAlign || 'right';
        legend(ctx, items, padL + at[0] * (w - padL - padR) + (al === 'right' ? -4 : al === 'left' ? 4 : 0), padT + at[1] * (h - padT - padB) + 3, al);
    }
    const pct = (v) => (v == null || !isFinite(v) ? '—' : (100 * v).toFixed(1) + '%');
    const last = (a) => (a && a.length ? a[a.length - 1] : NaN);
    const fmtN = (n) => n.toLocaleString('en-US').replace(/,/g, '\u00a0');   // 千分位用不换行空格：画布上的数字不会被折成两行
    const status = (R) => `第 ${R.step} / ${R.cap} 步` + (R.step >= R.cap ? ' · 已停止' : ' · 训练中');
    const SHIFT_BUDGET = 10;   // 平移测试每帧推进的（平移, 图）对数（4 个模型各推理这么多张）

    // ═══════════════════ Step 03：步长 / 补零 / 池化 / 多通道（一张图，每帧重算） ═══════════════════
    /** 选中的配置（K、stride、pad）下的 Sobel-x 型核作用在选中样本上：用共享库的通用 AG.conv2d */
    function cfgConv(img, id) {
        const c = CONV_CFGS.find((q) => q.id === id) || CONV_CFGS[0], w = sobelK(c.K), Ho = outSize(IMG, c.K, c.s, c.p);
        const out = AG.noGrad(() => AG.conv2d(AG.tensor(1, D, img), AG.tensor(1, c.K * c.K, w), null, { C: 1, H: IMG, W: IMG, K: c.K, stride: c.s, pad: c.p })).data;
        return { c, w, Ho, out };
    }
    /** ReLU(Sobel x 响应) → 2×2 maxpool；记下每个窗口里最大值所在的位置（反向时梯度只回到那里） */
    function poolDemo(edge) {
        const a = Float32Array.from(edge, (v) => Math.max(0, v));
        const p = AG.noGrad(() => AG.maxPool2(AG.tensor(1, D, a), { C: 1, H: IMG, W: IMG })).data, arg = new Int32Array(64);
        for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { const b = 2 * i * IMG + 2 * j; let best = b; [b + 1, b + IMG, b + IMG + 1].forEach((q) => { if (a[q] > a[best]) best = q; }); arg[i * 8 + j] = best; }
        return { a, p, arg };
    }
    /** conv2 的一个输出通道 j = Σ_c（输入通道 c ⋆ 核 w[j, c]）+ b[j]：把 8 个输入通道各自的贡献拆开（j = 对选中样本激活最强的通道） */
    function multiChannel(m, tr) {
        const c1 = m.cfg.c1, c2 = m.cfg.c2; let j = 0, best = -1;
        for (let c = 0; c < c2; c++) { let s = 0; for (let k = 0; k < 64; k++) s += tr.a2[c * 64 + k]; if (s > best) { best = s; j = c; } }
        const Wd = m.conv2.w.data, kern = new Float32Array(c1 * 9), contrib = new Float32Array(c1 * 64), z = new Float32Array(64);
        for (let c = 0; c < c1; c++) {
            for (let t = 0; t < 9; t++) kern[c * 9 + t] = Wd[(j * c1 + c) * 9 + t];
            const y = AG.noGrad(() => conv(AG.tensor(1, 64, tr.p1.subarray(c * 64, (c + 1) * 64)), AG.tensor(1, 9, kern.subarray(c * 9, c * 9 + 9)), null, { C: 1, H: 8, W: 8, K: 3 })).data;
            contrib.set(y, c * 64); for (let k = 0; k < 64; k++) z[k] += y[k];
        }
        const b = m.conv2.b.data[j]; for (let k = 0; k < 64; k++) z[k] += b;
        let md = 0; for (let k = 0; k < 64; k++) md = Math.max(md, Math.abs(z[k] - tr.z2[j * 64 + k]));
        return { j, kern, contrib, z, a: Float32Array.from(z, (v) => Math.max(0, v)), bias: b, inputs: tr.p1, maxDiff: md };
    }

    // ═══════════════════ 每帧的单样本分析（便宜：一张图） ═══════════════════
    /** 选中的测试样本穿过正在训练的 CNN 与 MLP：每层的特征图、两个模型的概率 */
    function traceSample(S) {
        const d = S.data, i = Math.max(0, Math.min(d.te.n - 1, S.sample | 0)), x = d.te.X.subarray(i * D, (i + 1) * D), y = d.te.y[i];
        const tr = {}, logits = AG.noGrad(() => cnnForward(S.main.cnn.m, AG.tensor(1, D, x), tr)).data, ml = predict(S.main.mlp.m, x, 1);
        const p = LM.softmax(logits, new Float32Array(K)), pm = LM.softmax(ml, new Float32Array(K));
        return { i, x, y, z1: tr.z1.data, a1: tr.a1.data, p1: tr.p1.data, z2: tr.z2.data, a2: tr.a2.data, p2: tr.p2.data, logits, p, pred: LM.argmax(p), pm, predM: LM.argmax(pm), step: S.main.cnn.step };
    }
    /** Step 07：选中样本水平平移 −4…4 像素，四个模型（MLP、CNN、MLP + 增强、CNN + 增强）各自的预测 */
    function shiftDemo(S) {
        const x = S.tr.x, n = 9, buf = new Float32Array(n * D);
        for (let k = 0; k < n; k++) shiftImg(x, k - 4, 0, buf.subarray(k * D, (k + 1) * D));
        const models = [S.main.mlp.m, S.main.cnn.m, S.A.mlp.m, S.A.cnn.m];
        return { imgs: buf, pred: models.map((m) => { const Z = predict(m, buf, n); return Array.from({ length: n }, (_, k) => LM.argmax(Z, k * K, K)); }) };
    }
    /** Step 08 的结构图：第一张展示图穿过当前的 U-Net，每一级的激活（取通道平均画缩略图） */
    function unetTrace(U) {
        const X = AG.tensor(1, D, U.ns.Xn.subarray(0, D)), tr = {};
        AG.noGrad(() => unetForward(U.runs[0].m, X, tr));
        const chMean = (t, C) => { const HW = t.c / C, out = new Float32Array(HW); for (let c = 0; c < C; c++) for (let k = 0; k < HW; k++) out[k] += t.data[c * HW + k] / C; return out; };
        const { c1, c2 } = U.runs[0].m.cfg;
        return { x: U.ns.Xn.subarray(0, D), s1: chMean(tr.s1, c1), s2: chMean(tr.s2, c2), bt: chMean(tr.bt, c2), h2: chMean(tr.h2, c1), y: tr.y.data, x0: U.ns.X0.subarray(0, D) };
    }

    return {
        IMG, D, K, NTRAIN, NTEST, range, makeData, shiftImg, conv, gap, batchNorm, numParams,
        makeCNN, cnnForward, makeMLP, mlpForward, makeDeep, deepForward, makeUNet, unetForward, forwardOf, cloneModel, copyModel,
        makeRun, nextBatch, trainStep, predict, evaluate, evalBegin, evalWork, evalNow, tick, CAP, LR, BATCH, EVAL_BUDGET, mainRuns, augRuns,
        SHIFTS, NSHIFT, shiftEvalBegin, shiftEvalWork, shiftCurve, makeDenoiseRun, denoiseStep, noisySet, mseOf, psnr, denoiseEval,
        UCAP, UB, UN_EVAL, UN_SHOW, unetRuns, unetTick, unetEvalBegin, unetEvalWork, blurBaseline, DCAP, DB, DEEP_C, deepRuns, deepTick, deepFlow,
        KERNELS, kernelById, applyKernels, sobelK, CONV_CFGS, outSize, conv1Maps, mlpHidden, relDiff, cosSim, validMask, equivDemo, equivCurve,
        ERF_LS, ERF_N, ERF_C, erfMaps, cnnERF, drawImg, absMax, txt, imgGrid, gridHit, barChart, lines, legend, curves, pct, last, fmtN, status,
        SHIFT_BUDGET, traceSample, shiftDemo, unetTrace, cfgConv, poolDemo, multiChannel, shiftView,
    };
})();

PipelineLab.mount({
    title: '08 · 卷积神经网络：从卷积核到 U-Net',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 20251123,
    css: LM.CSS,
    regen: { label: '重新生成数据', title: '用新的随机种子重新生成 4000 + 500 张数字（只重算 RESIDENT），所有训练从头开始' },

    params: [
        { id: 'sample', label: '样本 #', min: 0, max: 499, step: 1, value: 43, affects: 'frame', desc: '选中的测试样本（500 张，与第 02 篇是同一批图、同一个编号）。Step 01–04 用它演示卷积，Step 06 看它的逐层特征图，Step 07 把它左右平移。也可以直接点 Step 01 的小图网格或 Step 05 的错例。' },
        { id: 'kernel', label: '卷积核', options: P08.KERNELS.map((k) => ({ value: k.id, label: k.name })), value: 'sobelx', affects: 'frame', desc: 'Step 01 的固定 3×3 卷积核。Sobel x / y 找竖直 / 水平笔画的边，拉普拉斯是二阶导（笔画中线为负），高斯模糊做平均。也可以点 Step 01 第三张图里的某一列。' },
        { id: 'dx', label: '平移 dx', min: -4, max: 4, step: 1, value: 2, unit: ' px', affects: 'frame', desc: 'Step 02 把输入向右平移 dx 个像素（负数向左）：卷积的特征图跟着平移（等变），MLP 的隐藏层则换了一副样子。' },
        { id: 'dy', label: '平移 dy', min: -3, max: 3, step: 1, value: 1, unit: ' px', affects: 'frame', desc: 'Step 02 的竖直平移（向下为正）。' },
        { id: 'cfg', label: '卷积配置', options: P08.CONV_CFGS.map((c) => ({ value: c.id, label: c.label })), value: 'k3s2p1', affects: 'frame', desc: 'Step 03：核大小 K、步长 s、补零 p 三个超参数决定输出尺寸 ⌊(16 + 2p − K)/s⌋ + 1。s = 2 时输出边长减半；p = 0（valid）时每层缩小 K − 1。' },
        { id: 'spf', label: '每帧步数', options: [1, 2, 4], value: 1, affects: 'frame', desc: '每一帧训练几步（所有正在训练的网络都用它）。默认 1：CNN 每步约 4 ms（Node）；机器快就调到 2 或 4，更快看到训练终点。不会重置训练。' },
        { id: 'aug', label: '增强幅度 ±a', options: [1, 2, 3], value: 2, affects: 'frame', desc: 'Step 07 的数据增强：训练时每张图随机平移 dx, dy ∈ {−a…a}。改它只会让 Step 07 的两个增强模型从头训练。a = 3 时 MLP 在没平移的测试图上反而变差（容量不够同时记住所有位置）。' },
        { id: 'sigma', label: '噪声 σ', options: [0.3, 0.5, 0.8], value: 0.5, affects: 'frame', desc: 'Step 08 去噪任务的高斯噪声标准差（像素值在 0–1 之间）。改它会让两个去噪网络从头训练。σ 越大，跳连带来的细节越难保住，与模糊滤波的差距也随之变化。' },
        { id: 'depth', label: '深度 L', options: [4, 8, 16, 24], value: 16, affects: 'frame', desc: 'Step 09 深层网络的块数（每块一个 3×3 卷积）。L = 4 时普通网络和残差网络差不多；16 时普通网络明显变差；24 时普通网络完全训不动，残差网络照样能训。' },
        { id: 'norm', label: '归一化', options: [{ value: 'none', label: '无' }, { value: 'bn', label: 'BatchNorm' }], value: 'none', affects: 'frame', desc: 'Step 09 的每个卷积后面要不要加 BatchNorm。加了以后普通网络每层的激活不再缩小，但梯度反而朝输入方向越来越大，深层时照样训不好；残差网络（分支的 γ 初始化为 0）两种设置都能训。' },
        { id: 'retrain', label: '重新训练', action: (S) => { S.retrain = (S.retrain || 0) + 1; }, affects: 'param', desc: '把所有训练（Step 05 / 07 / 08 / 09）从第 0 步重新开始。' },
    ],

    stages: [
        { id: 'data', lifetime: 'resident', run(S) { S.data = P08.makeData(S.rand); } },
        {
            id: 'train', lifetime: 'history', run(S) {   // Step 05：CNN 与 MLP 同时训练（同一串 batch），每帧 spf 步，1500 步后停止
                if (S._mainEp !== S.epoch) { S._mainEp = S.epoch; S.main = P08.mainRuns(S); }
                P08.tick(S.main.cnn, S.spf, P08.EVAL_BUDGET);
                P08.tick(S.main.mlp, S.spf, P08.EVAL_BUDGET);
            },
        },
        {
            id: 'trace', lifetime: 'frame', run(S) {     // 选中样本：固定卷积核、平移等变、步长 / 池化 / 多通道、训练中 CNN 的逐层特征与输入梯度
                S.tr = P08.traceSample(S);
                S.kmaps = P08.applyKernels(S.tr.x);
                const ki = Math.max(0, P08.KERNELS.findIndex((k) => k.id === S.kernel)), km = S.kmaps.subarray(ki * 256, (ki + 1) * 256), key = S.sample + ':' + S.kernel;
                if (S.pixUser && S.pixUser.key === key) S.pixAt = S.pixUser.p;
                else { let b = 0; for (let i = 1; i < 256; i++) if (Math.abs(km[i]) > Math.abs(km[b])) b = i; S.pixAt = [b % 16, b >> 4]; }   // 默认：响应最强的像素
                S.eq = P08.equivDemo(S);
                S.cfgRes = P08.cfgConv(S.tr.x, S.cfg);
                S.poolRes = P08.poolDemo(S.kmaps.subarray(0, 256));
                S.mc = P08.multiChannel(S.main.cnn.m, S.tr);
                S.cerf = P08.cnnERF(S.main.cnn.m, S.tr.x);
            },
        },
        {
            id: 'equivCurve', lifetime: 'frame', lazy: true, run(S) {   // 40 张测试图 × 5 个平移量（主训练每 100 步重算一次）
                const key = Math.floor(S.main.cnn.step / 100) + ':' + S.epoch;
                if (S._ecKey === key) return;
                S._ecKey = key; S.ec = P08.equivCurve(S);
            },
        },
        { id: 'erf', lifetime: 'param', lazy: true, run(S) { if (S._erfSeed === S.seed) return; S._erfSeed = S.seed; S.erf = P08.erfMaps(S.seed + 3); } },
        {
            id: 'augShift', lifetime: 'history', lazy: true, run(S) {   // Step 07：两个增强模型的训练 + 四个模型的平移测试（分帧）
                const key = S.aug + ':' + S.epoch;
                if (S._augKey !== key) { S._augKey = key; S.A = P08.augRuns(S, S.aug); S.E = {}; }
                P08.tick(S.A.cnn, S.spf, 0); P08.tick(S.A.mlp, S.spf, 0);
                const runs = [S.main.mlp, S.main.cnn, S.A.mlp, S.A.cnn];
                const capped = runs.every((R) => R.step >= R.cap), done = capped && S.E.res && S.E.res.n === P08.NSHIFT && runs.every((R, k) => S.E.res.steps[k] >= R.cap);
                if (done) return;
                if (!S.E.pending) P08.shiftEvalBegin(S.E, runs, capped ? P08.NSHIFT : 40);   // 训练中：每轮 40 张图（约 60 帧刷新一次）；全部训练完：最后用 100 张评估一轮
                P08.shiftEvalWork(S.E, S.data, P08.SHIFT_BUDGET);
            },
        },
        { id: 'shiftDemo', lifetime: 'frame', lazy: true, needs: ['augShift'], run(S) { S.sd = P08.shiftDemo(S); } },
        {
            id: 'unet', lifetime: 'history', lazy: true, run(S) {   // Step 08：U-Net 与去掉跳连的版本，同一串含噪 batch
                const key = S.sigma + ':' + S.epoch;
                if (S._unKey !== key) { S._unKey = key; S.U = P08.unetRuns(S, S.sigma); }
                P08.unetTick(S.U, S.spf);
            },
        },
        { id: 'unetTrace', lifetime: 'frame', lazy: true, needs: ['unet'], run(S) { S.ut = P08.unetTrace(S.U); } },
        {
            id: 'deep', lifetime: 'history', lazy: true, run(S) {   // Step 09：深度 L 的普通网络与残差网络
                const key = [S.depth, S.norm, S.epoch].join();
                if (S._dpKey !== key) { S._dpKey = key; S.G = P08.deepRuns(S, S.depth, S.norm === 'bn'); }
                P08.deepTick(S.G, S.spf);
            },
        },
        {
            id: 'deepFlow', lifetime: 'frame', lazy: true, run(S) {   // 初始化时的信号传播（一个 batch 的前向 + 反向）
                const key = [S.depth, S.norm, S.seed].join();
                if (S._dfKey === key) return;
                S._dfKey = key; S.flow = P08.deepFlow(S, S.depth, S.norm === 'bn');
            },
        },
    ],

    intro: {
        title: '让网络知道「相邻」：卷积、池化、感受野，一直到 U-Net',
        flow: ['16×16 数字', '卷积核滑动', '局部 · 共享 · 等变', '通道 / 步长 / 池化', '感受野', '训练 CNN', '逐层特征图', '平移与增强', 'U-Net 去噪', '残差 + 归一化'],
        html: LM.navHtml('08') + `
            <p>第 02 篇的 MLP 先把 16×16 的图展平成 256 个数——它并不知道哪两个像素挨着：把所有图的像素按同一个固定顺序打乱再训练，学到的东西本质上一样（第一层权重的行跟着换个顺序而已）。<strong>卷积神经网络</strong>（Convolutional Neural Network，CNN）把关于图像的几条先验直接写进结构：<strong>局部连接</strong>（一个输出只看一小块邻域）、<strong>权重共享</strong>（同一个小卷积核在整张图上滑动）、<strong>池化</strong>（逐级降低分辨率、扩大视野）。结果是参数少得多、准确率更高，而且对图案出现在哪里没那么敏感（但也不是完全不敏感，Step 07）。</p>
            <p>数据与第 02 篇<strong>完全相同</strong>：同一个程序化数字生成器、同一个种子，4000 张训练 + 500 张测试，逐像素一致，样本编号也一样——两篇的准确率可以直接对比。它仍然是 MNIST（28×28、6 万张）的 16×16 缩小替身。</p>
            <p>页面里<strong>实时训练</strong>的网络：Step 05 的 CNN 和对照的 MLP（同一串 batch、同一个优化器，打开页面就开始训练，1500 步后停止）；Step 07 的两个数据增强版本、Step 08 的 U-Net 去噪器、Step 09 的深层普通 / 残差网络在滚动到那一步时才开始训练。预览全部来自真实计算。代码块是同一算法的 <strong>C++ 版本</strong>，基于本系列自带的 minml 小库（二维张量 + 录带式自动微分 + 卷积 / 池化 / 上采样 + Adam，节点图上方的 Common 面板里有全文，悬停函数名可以看实现）。</p>`,
    },

    steps: [
        // ───────────────────────────── 01 卷积运算 ─────────────────────────────
        {
            id: 'conv', title: '卷积：一个在图上滑动的小点积', tag: 'CONV',
            desc: '卷积层的基本运算非常简单：拿一个 K×K 的小权重矩阵（<strong>卷积核</strong>，kernel / filter），放到图上某个位置，和底下的 K×K 个像素逐元素相乘再求和，得到输出图上对应位置的一个数；然后把核挪一个像素再算一次……直到扫完整张图。输出叫<strong>特征图</strong>（feature map）：哪里的局部图案和核长得像，哪里的响应就大。图像外面先补一圈 0（<strong>补零</strong>，padding），输出就和输入一样大。下面先用几个手工设计的经典核（图像处理里用了几十年）作用在选中的数字上：Sobel x 对左右亮度差敏感，在竖直笔画的左右两条边上给出一正一负的响应；拉普拉斯核是二阶导数，在笔画中线上为负。CNN 要做的，就是让这些核的数值<strong>从数据里学出来</strong>（Step 05）。',
            formula: [
                'y[i, j] = b + Σ_{u=0}^{K−1} Σ_{v=0}^{K−1} w[u, v] · x[i + u − p, j + v − p]   // K = 3、p = 1；画面外的 x 当作 0',
                '一张 16×16 的输出 = 256 个位置 × 9 次乘加 = 2 304 次乘加，但只用 9 个权重（+ 1 个偏置）',
                'Sobel_x = [[−1, 0, 1], [−2, 0, 2], [−1, 0, 1]]      Laplace = [[0, 1, 0], [1, −4, 1], [0, 1, 0]]',
            ],
            code: {
                title: 'C++ · filter2d 与 6 个固定卷积核', src: `
                // 单通道 K×K 卷积（互相关）+ 补零 p = K/2：输出和输入一样大。核 k 是固定的滤波器，不是学出来的
                std::vector<float> filter2d(const std::vector<float>& x, int H, int W, const std::vector<float>& k, int K) {
                    std::vector<float> y = floats(H * W);
                    int p = K / 2;
                    for (int i = 0; i < H; i++)
                        for (int j = 0; j < W; j++) {
                            float s = 0;                                         // 一个输出 = 窗口与核的点积（K·K 次乘加）
                            for (int u = 0; u < K; u++)
                                for (int v = 0; v < K; v++) {
                                    int yy = i + u - p, xx = j + v - p;
                                    if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += k[u * K + v] * x[yy * W + xx];   // 画面外 = 补的 0
                                }
                            y[i * W + j] = s;
                        }
                    return y;
                }

                // 本步的 6 个固定核（3×3，行主序）
                const std::vector<float> SOBEL_X = {-1, 0, 1, -2, 0, 2, -1, 0, 1};   // 右减左：竖直笔画的左右两条边
                const std::vector<float> SOBEL_Y = {-1, -2, -1, 0, 0, 0, 1, 2, 1};   // 下减上：水平笔画的上下两条边
                const std::vector<float> EDGE_45 = {0, 1, 2, -1, 0, 1, -2, -1, 0};   // 沿对角方向的亮度差
                const std::vector<float> LAPLACE = {0, 1, 0, 1, -4, 1, 0, 1, 0};     // 四邻之和 − 4·中心：二阶导
                const std::vector<float> GAUSS_3 = {1 / 16.f, 2 / 16.f, 1 / 16.f, 2 / 16.f, 4 / 16.f, 2 / 16.f, 1 / 16.f, 2 / 16.f, 1 / 16.f};
                const std::vector<float> SHARPEN = {0, -1, 0, -1, 5, -1, 0, -1, 0};  // 原图 − 拉普拉斯：边缘更陡`,
            },
            previews: [
                {
                    id: 'kconv', label: '卷积核在图上滑动', sub: '左 = 输入（虚线 = 补零的一圈）· 右 = 输出特征图（橙 = 正、蓝 = 负）· 黄框 = 选中的输出像素和它读的 3×3 窗口 · <b>点击任一张图选像素</b>', large: true, size: [360, 232],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), k = P08.kernelById(S.kernel), ki = P08.KERNELS.indexOf(k), x = S.tr.x, y = S.kmaps.subarray(ki * 256, (ki + 1) * 256), [px, py] = S.pixAt, cs = 9.5, oy = 34, ox1 = 14, ox2 = 196;
                        P08.txt(ctx, `${k.name}：${k.note}`, 6, 4, { bold: true });
                        ctx.strokeStyle = '#4a5058'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.strokeRect(ox1 - cs, oy - cs, 18 * cs, 18 * cs); ctx.setLineDash([]);
                        P08.drawImg(ctx, x, ox1, oy, 16 * cs, 16 * cs);
                        const m = P08.absMax(y, 1, 0, 256);
                        P08.drawImg(ctx, y, ox2, oy, 16 * cs, 16 * cs, { map: 'diverge', range: [-m, m] });
                        ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 2;
                        ctx.strokeRect(ox1 + (px - 1) * cs, oy + (py - 1) * cs, 3 * cs, 3 * cs);
                        ctx.strokeRect(ox2 + px * cs, oy + py * cs, cs, cs);
                        ctx.strokeStyle = 'rgba(232,194,90,0.5)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(ox1 + (px + 2) * cs, oy + (py + 0.5) * cs); ctx.lineTo(ox2 + px * cs, oy + (py + 0.5) * cs); ctx.stroke();
                        S._kc = { cs, oy, ox1, ox2 };
                        P08.txt(ctx, `输入 x · 样本 #${S.tr.i}（标签 ${S.tr.y}）`, ox1 + 8 * cs, oy + 17 * cs + 6, { align: 'center', color: LM.C.gray });
                        P08.txt(ctx, `输出 y = k ⋆ x · 范围 ±${m.toFixed(2)}`, ox2 + 8 * cs, oy + 17 * cs + 6, { align: 'center', color: LM.C.gray });
                    },
                    probe: (x, y, S, PL, info) => {
                        const g = S._kc; if (!g) return '';
                        const ki = P08.KERNELS.indexOf(P08.kernelById(S.kernel));
                        for (const ox of [g.ox1, g.ox2]) { const q = Math.floor((info.dx - ox) / g.cs), r = Math.floor((info.dy - g.oy) / g.cs); if (q >= 0 && q < 16 && r >= 0 && r < 16) return `(${r}, ${q})  x = ${S.tr.x[r * 16 + q].toFixed(3)}  y = ${S.kmaps[ki * 256 + r * 16 + q].toFixed(3)}（点击选中）`; }
                        return '';
                    },
                    pick: (x, y, S, PL, info) => {
                        const g = S._kc; if (!g) return;
                        for (const ox of [g.ox1, g.ox2]) { const q = Math.floor((info.dx - ox) / g.cs), r = Math.floor((info.dy - g.oy) / g.cs); if (q >= 0 && q < 16 && r >= 0 && r < 16) { S.pixUser = { key: S.sample + ':' + S.kernel, p: [q, r] }; return 'frame'; } }
                    },
                },
                {
                    id: 'karith', label: '选中像素的 9 次乘加', sub: '窗口（含补的 0）⊙ 卷积核 = 逐元素乘积，加起来就是右图里那个输出像素', size: [360, 150],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), k = P08.kernelById(S.kernel), ki = P08.KERNELS.indexOf(k), [px, py] = S.pixAt, x = S.tr.x, cs = 24, y0 = 26;
                        const win = [], prod = []; let sum = 0;
                        for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) { const yy = py + u - 1, xx = px + v - 1, a = yy >= 0 && yy < 16 && xx >= 0 && xx < 16 ? x[yy * 16 + xx] : 0; win.push(a); prod.push(a * k.w[u * 3 + v]); sum += a * k.w[u * 3 + v]; }
                        const grid = (vals, x0, kind) => {
                            const m = kind === 'img' ? 1 : Math.max(1e-6, ...vals.map(Math.abs));
                            vals.forEach((val, t) => {
                                const gx = x0 + (t % 3) * cs, gy = y0 + Math.floor(t / 3) * cs, rgb = [0, 0, 0];
                                if (kind === 'img') { const g8 = Math.round(255 * Math.min(1, Math.max(0, val))); ctx.fillStyle = `rgb(${g8},${g8},${g8})`; } else { PL.cmap.diverge(0.5 + 0.5 * val / m, rgb); ctx.fillStyle = LM.rgbStr(rgb); }
                                ctx.fillRect(gx, gy, cs - 2, cs - 2);
                                const lum = kind === 'img' ? val : 0.5; const s = Math.abs(val) < 0.005 ? '0' : Math.abs(val - Math.round(val)) < 0.005 ? String(Math.round(val)) : Math.abs(val) >= 1 ? val.toFixed(1) : val.toFixed(2).replace(/^(-?)0\./, '$1.');
                                P08.txt(ctx, s, gx + (cs - 2) / 2, gy + (cs - 2) / 2 + 1, { align: 'center', base: 'middle', color: kind === 'img' && lum > 0.55 ? '#000' : '#f0f0f0' });
                            });
                        };
                        P08.txt(ctx, `窗口 (${py}, ${px})`, 8 + 1.5 * cs, 6, { align: 'center', color: LM.C.gray });
                        grid(win, 8, 'img'); P08.txt(ctx, '⊙', 92, y0 + 1.5 * cs, { base: 'middle', px: 14, color: LM.C.text });
                        P08.txt(ctx, `核：${k.name}`, 112 + 1.5 * cs, 6, { align: 'center', color: LM.C.gray });
                        grid(k.w, 112, 'w'); P08.txt(ctx, '=', 196, y0 + 1.5 * cs, { base: 'middle', px: 14 });
                        P08.txt(ctx, '逐元素乘积', 214 + 1.5 * cs, 6, { align: 'center', color: LM.C.gray });
                        grid(prod, 214, 'w');
                        const out = S.kmaps[ki * 256 + py * 16 + px];
                        P08.txt(ctx, `Σ = ${sum.toFixed(4)}`, 296, y0 + 1.5 * cs - 8, { base: 'middle', bold: true, color: LM.C.yellow });
                        P08.txt(ctx, `y = ${out.toFixed(4)}`, 296, y0 + 1.5 * cs + 8, { base: 'middle', color: LM.C.gray });
                        P08.txt(ctx, `9 次乘加 = 右图 (${py}, ${px}) 的值；整张图 256 个位置共用这 9 个权重`, 8, y0 + 3 * cs + 10, { color: LM.C.dim });
                    },
                },
                {
                    id: 'kgallery', label: '6 个固定核作用在同一张图上', sub: '上 = 核（3×3），下 = 输出（各自对称着色）· 实际上是一次 C_out = 6 的卷积 · <b>点击某一列换核</b>', size: [360, 168],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c), n = P08.KERNELS.length, cw = (w - 8) / n, ks = 30, os = 50;
                        P08.KERNELS.forEach((k, j) => {
                            const x0 = 4 + j * cw, cx = x0 + cw / 2, sel = k.id === S.kernel, km = P08.absMax(k.w, 1, 0, 9), om = S.kmaps.subarray(j * 256, (j + 1) * 256), m = P08.absMax(om, 1, 0, 256);
                            if (sel) { ctx.fillStyle = 'rgba(232,194,90,0.10)'; ctx.fillRect(x0 + 1, 2, cw - 2, 164); }
                            P08.drawImg(ctx, k.w, cx - ks / 2, 8, ks, ks, { W: 3, H: 3, map: 'diverge', range: [-km, km] });
                            P08.drawImg(ctx, om, cx - os / 2, 46, os, os, { map: 'diverge', range: [-m, m] });
                            ctx.strokeStyle = sel ? LM.C.yellow : '#2a2f36'; ctx.lineWidth = sel ? 2 : 1; ctx.strokeRect(cx - os / 2 - 1, 45, os + 2, os + 2);
                            P08.txt(ctx, k.name, cx, 104, { align: 'center', color: sel ? LM.C.yellow : LM.C.gray });
                            P08.txt(ctx, `±${m.toFixed(1)}`, cx, 120, { align: 'center', color: LM.C.dim });
                        });
                        S._kg = { cw };
                        P08.txt(ctx, '同一张图、6 种「看法」：每个核把它关心的局部图案变成一张特征图', 6, 146, { color: LM.C.dim });
                    },
                    probe: (x, y, S, PL, info) => { const g = S._kg; if (!g) return ''; const j = Math.floor((info.dx - 4) / g.cw); return j >= 0 && j < P08.KERNELS.length ? `${P08.KERNELS[j].name}：${P08.KERNELS[j].note}（点击选用）` : ''; },
                    pick: (x, y, S, PL, info) => { const g = S._kg; if (!g) return; const j = Math.floor((info.dx - 4) / g.cw); if (j >= 0 && j < P08.KERNELS.length) PL.setParam('kernel', P08.KERNELS[j].id); },
                },
                {
                    id: 'pickgrid', label: '测试样本（每列一个数字）', sub: '每类前 3 张 · 黄框 = 当前选中 · <b>点击选择样本</b>（全页共用）', size: [360, 132],
                    render: (c, S) => {
                        const d = S.data, items = [], idx = [];
                        for (let r = 0; r < 3; r++) for (let k = 0; k < 10; k++) { const i = d.byClass[k][r]; idx.push(i); items.push({ img: d.te.X.subarray(i * 256, (i + 1) * 256), outline: i === S.tr.i ? LM.C.yellow : null }); }
                        S._pg = P08.imgGrid(c, items, { cols: 10, header: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] }); S._pgIdx = idx;
                    },
                    probe: (x, y, S, PL, info) => { const k = P08.gridHit(S._pg, info); return k < 0 ? '' : `测试样本 #${S._pgIdx[k]} · 标签 ${S.data.te.y[S._pgIdx[k]]}（点击选中）`; },
                    pick: (x, y, S, PL, info) => { const k = P08.gridHit(S._pg, info); if (k >= 0) PL.setParam('sample', S._pgIdx[k]); },
                },
            ],
            notes: '点<b>左边的输入图</b>或<b>右边的输出图</b>选一个像素（默认：响应绝对值最大的那个），第二张图给出这一个输出的 9 次乘加——它和右图里那个像素的值完全相同。换<b>卷积核</b>（或点第三张图的某一列）、换<b>样本 #</b>（或点最后一张图）。Sobel x 在竖直笔画的左边为正（暗 → 亮）、右边为负；拉普拉斯在笔画两侧为正、中线为负。',
            warn: '深度学习框架里的「卷积」其实是<strong>互相关</strong>（cross-correlation）：核不翻转，直接逐元素相乘；数学上的卷积要先把核旋转 180°。对学出来的核这没有区别（翻转一下就是另一组权重），但拿 Sobel 核和信号处理软件对结果时要注意符号。',
        },
        // ───────────────────────────── 02 为什么用卷积 ─────────────────────────────
        {
            id: 'why', title: '为什么用卷积：局部连接、权重共享、平移等变', tag: 'WHY',
            desc: 'MLP 的第一层里，每个隐藏单元连着全部 256 个像素：第 02 篇的 64 个单元就是 16 448 个参数；而且它在左上角学到的「横笔画检测器」到了右下角完全用不上，必须再学一遍。卷积层加了两个约束：<strong>局部连接</strong>——每个输出只连着 K×K 的邻域；<strong>权重共享</strong>——所有位置用同一个核。于是参数量只和核大小、通道数有关，和图像大小无关：8 个 3×3 核 = 80 个参数，却输出 8 张 16×16 的特征图（2 048 个数）；同样输出 2 048 个数的全连接层要 52 万个参数。权重共享还带来一个精确的性质——<strong>平移等变</strong>（translation equivariance）：输入平移多少，特征图就平移多少，内容不变。下面用 Step 05 <strong>正在训练</strong>的 CNN 的第一层（卷积 + ReLU）真实验证：「先平移再卷积」与「先卷积再平移」在内部逐位相同。MLP 的隐藏层没有「位置」可言：平移输入之后，64 个单元的激活换了一副样子，没有什么操作能把它「挪回去」。',
            formula: [
                'MLP 第一层：z = x · W + b，W ∈ R^{256×64}   // 16 448 个参数，每个隐藏单元连着全部像素',
                '卷积层 1 → 8 通道、3×3：8 · (9 + 1) = 80 个参数   // 输出 8 × 16 × 16 = 2 048 个数；同样输出的全连接层：256 · 2048 + 2048 = 526 336',
                '平移等变：F(T_δ x) = T_δ F(x)      (T_δ x)[i, j] = x[i − δ_y, j − δ_x]   // F = 卷积 + 逐元素运算（ReLU）；在不碰到边框的位置精确成立',
            ],
            code: {
                title: 'C++ · convParams / denseParams / equivarianceError / hiddenChange', src: `
                // 参数量：卷积层只和核大小、通道数有关，和图像多大无关；全连接层的每个输出连着全部输入
                int convParams(int cin, int cout, int K) { return cout * (cin * K * K + 1); }
                int denseParams(int in, int out) { return out * (in + 1); }

                // 平移等变的数值检验：F = 卷积 + ReLU，T = 平移 (dx, dy)。
                // 在两边都没读到补零的位置上比较 F(T x) 与 T F(x)，返回最大差（实测：0，逐位相同）
                float equivarianceError(const ConvLayer& L, const std::vector<float>& x, int dx, int dy) {
                    NoGrad ng;
                    int C = L.W->val.rows;
                    std::vector<float> Fx = relu(applyConv(L, constant(fromVector(1, 256, x))))->val.v;
                    std::vector<float> FTx = relu(applyConv(L, constant(fromVector(1, 256, shiftImage(x, dx, dy, 1, 16, 16)))))->val.v;
                    std::vector<float> TFx = shiftImage(Fx, dx, dy, C, 16, 16);
                    float worst = 0;
                    for (int c = 0; c < C; c++)
                        for (int i = 1; i < 15; i++)
                            for (int j = 1; j < 15; j++) {
                                if (i - dy < 1 || i - dy > 14 || j - dx < 1 || j - dx > 14) continue;   // 边框：一边读到补的 0
                                int k = (c * 16 + i) * 16 + j;
                                worst = std::max(worst, std::fabs(FTx[k] - TFx[k]));
                            }
                    return worst;
                }

                // MLP 的隐藏层没有「位置」：平移输入之后只能问它变了多少 ‖h(T x) − h(x)‖ / ‖h(x)‖
                float hiddenChange(const Linear& fc1, const std::vector<float>& x, int dx, int dy) {
                    NoGrad ng;
                    std::vector<float> h = relu(forward(fc1, constant(fromVector(1, 256, x))))->val.v;
                    std::vector<float> hT = relu(forward(fc1, constant(fromVector(1, 256, shiftImage(x, dx, dy, 1, 16, 16)))))->val.v;
                    double d = 0, n = 0;
                    for (size_t j = 0; j < h.size(); j++) { d += (hT[j] - h[j]) * (hT[j] - h[j]); n += h[j] * h[j]; }
                    return (float)std::sqrt(d / std::max(n, 1e-12));
                }`,
            },
            previews: [
                {
                    id: 'equiv', label: '先平移再卷积 = 先卷积再平移', sub: '训练中的 conv1 + ReLU，显示对这张图响应最强的通道 · 右下 = 两种顺序之差（只有一圈边框不为 0）', large: true, live: true, size: [360, 330],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), e = S.eq, sz = 96, xs = [14, 132, 250], ys = [44, 188], HW = 256, ch = e.ch;
                        let fm = 1e-6; for (let k = 0; k < HW; k++) fm = Math.max(fm, e.F[ch * HW + k], e.FT[ch * HW + k]);
                        P08.txt(ctx, `平移 (dx, dy) = (${S.dx}, ${S.dy}) · 通道 ${ch} · 第 ${e.step} 步的 conv1`, 40, 4, { bold: true });
                        const cell = (img, col, row, label, o) => { P08.drawImg(ctx, img, xs[col], ys[row], sz, sz, Object.assign({}, o)); ctx.strokeStyle = '#2a2f36'; ctx.lineWidth = 1; ctx.strokeRect(xs[col] - 0.5, ys[row] - 0.5, sz + 1, sz + 1); P08.txt(ctx, label, xs[col] + sz / 2, ys[row] - 15, { align: 'center', color: LM.C.gray }); };
                        cell(e.x, 0, 0, 'x'); cell(e.F, 1, 0, 'F(x)', { offset: ch * HW, map: 'inferno', range: [0, fm] }); cell(e.TF, 2, 0, 'T·F(x)：先卷积再平移', { offset: ch * HW, map: 'inferno', range: [0, fm] });
                        cell(e.Tx, 0, 1, 'T·x'); cell(e.FT, 1, 1, 'F(T·x)：先平移再卷积', { offset: ch * HW, map: 'inferno', range: [0, fm] }); const dm = Math.max(1e-6, P08.absMax(e.diff, 1, ch * HW, HW)); cell(e.diff, 2, 1, 'F(T·x) − T·F(x)', { offset: ch * HW, map: 'diverge', range: [-dm, dm] });
                        ctx.strokeStyle = LM.C.green; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
                        const inner = (v, d) => [Math.max(1, 1 + d), Math.min(14, 14 + d)];
                        const [x0, x1] = inner(0, S.dx), [y0, y1] = inner(0, S.dy);
                        ctx.strokeRect(xs[2] + x0 * sz / 16, ys[1] + y0 * sz / 16, (x1 - x0 + 1) * sz / 16, (y1 - y0 + 1) * sz / 16); ctx.setLineDash([]);
                        P08.txt(ctx, `绿虚线内（两边都没读到补零）：最大差 = ${e.dIn === 0 ? '0，逐位相同' : e.dIn.toExponential(2)}`, 14, 294, { color: LM.C.green });
                        P08.txt(ctx, `全图最大差 = ${e.dAll.toFixed(3)}：只出现在边框（补零 / 移出画面的笔画）`, 14, 311, { color: LM.C.dim });
                    },
                },
                {
                    id: 'mlpShift', label: 'MLP 的隐藏层：平移之后换了一副样子', sub: '训练中的 MLP，64 个隐藏单元排成 8×8（同一个色阶）· 相对变化 = ‖h(T·x) − h(x)‖ / ‖h(x)‖ · 没有任何「平移」能把左图变成右图', live: true, size: [360, 200],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), e = S.eq, sz = 112, m = Math.max(1e-6, ...e.h, ...e.hT);
                        P08.txt(ctx, `第 ${S.main.mlp.step} 步的 MLP · 平移 (${S.dx}, ${S.dy})`, 40, 4, { bold: true });
                        P08.drawImg(ctx, e.h, 24, 40, sz, sz, { W: 8, H: 8, map: 'inferno', range: [0, m] });
                        P08.drawImg(ctx, e.hT, 196, 40, sz, sz, { W: 8, H: 8, map: 'inferno', range: [0, m] });
                        P08.txt(ctx, 'h(x) = ReLU(x·W₁ + b₁)', 24 + sz / 2, 24, { align: 'center', color: LM.C.gray });
                        P08.txt(ctx, 'h(T·x)', 196 + sz / 2, 24, { align: 'center', color: LM.C.gray });
                        P08.txt(ctx, `cos(h(x), h(T·x)) = ${e.hCos.toFixed(3)} · 相对变化 = ${e.hRel.toFixed(2)}`, 8, 158, { color: LM.C.orange });
                        P08.txt(ctx, `对照：CNN 特征图的相对变化 ${e.fRel.toFixed(2)}，对齐（再平移回去）后 ${e.fRelAligned === 0 ? '0' : e.fRelAligned.toExponential(1)}`, 8, 176, { color: LM.C.blue });
                    },
                },
                {
                    id: 'equivCurve', label: '平移 k 个像素，表示变了多少', sub: '40 张测试图的平均相对变化 ‖f(T·x) − f(x)‖ / ‖f(x)‖（水平平移）· 主训练每 100 步重算', size: [360, 200], needs: ['equivCurve'],
                    render: (c, S) => {
                        const e = S.ec, top = Math.max(1.2, ...e.mlp, ...e.cnn) * 1.08;
                        P08.curves(c, [
                            { x: e.k, y: e.mlp, color: LM.C.orange, width: 2, label: 'MLP 隐藏层 h（没法对齐）' },
                            { x: e.k, y: e.cnn, color: LM.C.blue, width: 1.5, dash: [5, 4], label: 'CNN 特征图，直接比较' },
                            { x: e.k, y: e.cnnAligned, color: LM.C.green, width: 2.5, label: 'CNN 特征图，对齐后（内部）' },
                        ], { x: [0, 4], y: [0, top], xLabel: `平移像素数 k（第 ${e.step} 步的权重）`, legendAt: [1, 0.5] });
                    },
                },
                {
                    id: 'params', label: '参数量对比（本页的真实模型）', sub: '卷积层的参数与图像大小无关', size: [360, 150],
                    render: (c, S) => {
                        const cn = S.main.cnn.m, ml = S.main.mlp.m, f = P08.fmtN;
                        P08.lines(c, [
                            [{ t: 'MLP 第一层 256 → 64：', color: LM.C.orange, bold: true }, { t: `${f(ml.fc1.w.size + ml.fc1.b.size)} 个参数（每个隐藏单元 257 个）` }],
                            [{ t: 'CNN 第一层 1→8、3×3：', color: LM.C.blue, bold: true }, { t: `${f(cn.conv1.w.size + cn.conv1.b.size)} 个参数，输出 8×16×16 = ${f(2048)} 个数` }],
                            [{ t: '同样输出的全连接层：', color: LM.C.gray }, { t: `256 × ${f(2048)} + ${f(2048)} = ${f(256 * 2048 + 2048)} 个参数`, color: LM.C.red }],
                            [{ t: '整个模型：', color: LM.C.gray }, { t: `CNN ${f(P08.numParams(cn.params))} 个 · MLP ${f(P08.numParams(ml.params))} 个`, bold: true }],
                            { text: '图像变成 224 × 224 时，卷积层的参数一个都不用多；第一层全连接则要再大约 200 倍。', color: LM.C.dim },
                        ], { px: 11 });
                    },
                },
            ],
            notes: '拖<b>平移 dx / dy</b>：下排两张特征图（先平移再卷积 vs 先卷积再平移）始终一模一样，右下角的差只在一圈边框上不为 0。第三张图：平移 1 个像素时，CNN 特征图的原始变化（蓝虚线）其实比 MLP 还大——局部特征本来就跟着笔画走；但对齐以后（绿线）恰好为 0。MLP 的变化没有结构，也就没法利用。',
            warn: '「等变」不等于「不变」（invariance）：特征图会跟着动，而分类器最后要输出一个不随位置变的类别——这还得靠池化、全连接层或数据增强，Step 07 会看到它并不是自动成立的。ReLU 这类逐元素运算保持等变；步长 > 1 的卷积和池化只对步长整数倍的平移等变。',
        },
        // ───────────────────────────── 03 形状 ─────────────────────────────
        {
            id: 'shape', title: '多通道、步长、补零与池化：特征图的形状', tag: 'SHAPE',
            desc: '真正的卷积层同时有很多个核，每个核产生一个输出<strong>通道</strong>（channel）；输入如果有多个通道（RGB 图，或者上一层的 8 张特征图），一个核就是 C_in × K × K 的小立方体：在每个输入通道上各做一次二维卷积，再把 C_in 个结果<strong>加起来</strong>（第三张图把 CNN 第二层的一个输出通道真实地拆开）。<strong>步长</strong>（stride）s 控制核每次挪几个像素：s = 2 时输出边长减半、计算量减到 1/4；<strong>补零</strong> p 控制边界：p = (K − 1)/2 时输出与输入一样大（same），p = 0 时每层缩小 K − 1（valid）。<strong>池化</strong>（pooling）是不带参数的下采样：2×2 最大池化在每个窗口里只留最大值，让后面的层用同样大的核看到两倍大的范围，也让窗口内的小位移不影响结果。',
            formula: [
                'H_out = ⌊(H + 2p − K) / s⌋ + 1      // 例：H = 16、K = 3、s = 2、p = 1 → 8',
                'y[c_o, i, j] = b[c_o] + Σ_{c_i} Σ_{u, v} w[c_o, c_i, u, v] · x[c_i, s·i + u − p, s·j + v − p]   // 权重形状 [C_out, C_in, K, K]',
                'maxpool：y[c, i, j] = max_{0 ≤ u, v < 2} x[c, 2i + u, 2j + v]   // 没有参数；反向时梯度只回到窗口里最大的那个位置',
                '参数 = C_out · (C_in · K^2 + 1)      乘加 = C_out · H_out · W_out · C_in · K^2   // 卷积层：参数少、计算多',
            ],
            code: {
                title: 'C++ · outSize / convAny / convCost', src: `
                // 输出尺寸：H_out = ⌊(H + 2p − K) / s⌋ + 1
                int outSize(int H, int K, int s, int p) { return (H + 2 * p - K) / s + 1; }

                // 任意核大小 / 步长 / 补零的单通道卷积（不求梯度）：库里的 conv2d 是通用实现
                Tensor convAny(const std::vector<float>& x, const std::vector<float>& k, int K, int s, int p) {
                    NoGrad ng;
                    Var y = conv2d(constant(fromVector(1, 256, x)), constant(fromVector(1, K * K, k)), constant(zeros(1, 1)), conv(1, 16, 16, K, s, p));
                    return y->val;                                               // [1 × H_out·W_out]
                }

                // 一层的代价：参数个数、处理一张图要做的乘加次数
                struct LayerCost { long long params = 0, macs = 0; };

                LayerCost convCost(int cin, int cout, int K, int Hout, int Wout) {
                    LayerCost c;
                    c.params = (long long)cout * (cin * K * K + 1);             // 权重 + 偏置
                    c.macs = (long long)cout * Hout * Wout * cin * K * K;       // 每个输出位置、每个输出通道：C_in·K·K 次乘加
                    return c;
                }`,
            },
            previews: [
                {
                    id: 'cfgconv', label: '步长与补零决定输出尺寸', sub: '左 = 补零后的输入（暗格 = 补的 0），黄 / 青框 = 输出 (0, 0) 与 (0, 1) 读的窗口 · 右 = 输出，与输入同一像素尺度 · 核 = Sobel-x 型边缘核', large: true, size: [360, 262],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), R = S.cfgRes, { K, s, p } = R.c, n = 16 + 2 * p, cs = Math.min(9, Math.floor(168 / n)), ox = 8, oy = 42, Ho = R.Ho;
                        P08.txt(ctx, `${R.c.label}：H_out = ⌊(16 + 2·${p} − ${K}) / ${s}⌋ + 1 = ${Ho}`, 6, 4, { bold: true });
                        ctx.fillStyle = '#15181d'; ctx.fillRect(ox, oy, n * cs, n * cs);
                        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i < p || j < p || i >= 16 + p || j >= 16 + p) { ctx.strokeStyle = '#262a31'; ctx.strokeRect(ox + j * cs + 0.5, oy + i * cs + 0.5, cs - 1, cs - 1); }
                        P08.drawImg(ctx, S.tr.x, ox + p * cs, oy + p * cs, 16 * cs, 16 * cs);
                        ctx.lineWidth = 2; ctx.strokeStyle = LM.C.yellow; ctx.strokeRect(ox, oy, K * cs, K * cs);
                        if (Ho > 1) { ctx.strokeStyle = LM.C.cyan; ctx.setLineDash([4, 3]); ctx.strokeRect(ox + s * cs, oy, K * cs, K * cs); ctx.setLineDash([]); }
                        const m = P08.absMax(R.out, 1, 0, Ho * Ho), ox2 = 196;
                        P08.drawImg(ctx, R.out, ox2, oy, Ho * cs, Ho * cs, { W: Ho, H: Ho, map: 'diverge', range: [-m, m] });
                        ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 2; ctx.strokeRect(ox2, oy, cs, cs);
                        if (Ho > 1) { ctx.strokeStyle = LM.C.cyan; ctx.strokeRect(ox2 + cs, oy, cs, cs); }
                        P08.txt(ctx, `输入 16×16 + 补零 ${p} → ${n}×${n}`, ox + n * cs / 2, oy - 16, { align: 'center', color: LM.C.gray });
                        P08.txt(ctx, `输出 ${Ho}×${Ho}`, ox2 + Ho * cs / 2, oy - 16, { align: 'center', color: LM.C.gray });
                        const y0 = oy + Math.max(n, Ho) * cs + 14;
                        P08.txt(ctx, `相邻输出的窗口相隔 s = ${s} 像素；每个输出 ${K}×${K} = ${K * K} 次乘加`, 8, y0, { color: LM.C.dim });
                        P08.txt(ctx, `共 ${Ho * Ho} 个位置 × ${K * K} = ${P08.fmtN(Ho * Ho * K * K)} 次乘加（same、s = 1 时 ${P08.fmtN(256 * K * K)}）`, 8, y0 + 16, { color: LM.C.dim });
                    },
                },
                {
                    id: 'pool', label: '2×2 最大池化', sub: '左 = ReLU(Sobel x 响应)，网格 = 2×2 窗口，白点 = 每个窗口被保留的位置 · 右 = 池化结果 8×8（画成同样大小）', size: [360, 200],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), P = S.poolRes, cs = 9, ox1 = 10, ox2 = 196, oy = 22, m = Math.max(1e-6, ...P.a);
                        P08.drawImg(ctx, P.a, ox1, oy, 16 * cs, 16 * cs, { map: 'inferno', range: [0, m] });
                        P08.drawImg(ctx, P.p, ox2, oy, 16 * cs, 16 * cs, { W: 8, H: 8, map: 'inferno', range: [0, m] });
                        ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1;
                        for (let k = 0; k <= 8; k++) { ctx.beginPath(); ctx.moveTo(ox1 + 2 * k * cs, oy); ctx.lineTo(ox1 + 2 * k * cs, oy + 16 * cs); ctx.stroke(); ctx.beginPath(); ctx.moveTo(ox1, oy + 2 * k * cs); ctx.lineTo(ox1 + 16 * cs, oy + 2 * k * cs); ctx.stroke(); }
                        ctx.fillStyle = '#ffffff';
                        for (let i = 0; i < 64; i++) { if (P.p[i] <= 1e-6) continue; const q = P.arg[i]; ctx.beginPath(); ctx.arc(ox1 + (q % 16 + 0.5) * cs, oy + (Math.floor(q / 16) + 0.5) * cs, 1.8, 0, 7); ctx.fill(); }
                        P08.txt(ctx, '16×16（256 个数）', ox1 + 8 * cs, oy + 16 * cs + 6, { align: 'center', color: LM.C.gray });
                        P08.txt(ctx, '8×8（64 个数）：没有参数', ox2 + 8 * cs, oy + 16 * cs + 6, { align: 'center', color: LM.C.gray });
                    },
                },
                {
                    id: 'multich', label: '多通道卷积：一个输出通道 = 8 个二维卷积之和', sub: '训练中 CNN 的 conv2，挑对这张图响应最强的输出通道 j · 上 = 输入通道（pool₁ 的 8 张 8×8）· 中 = w[j, c]（3×3）· 下 = 各自的贡献（单独着色）', live: true, size: [360, 254],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), M = S.mc, n = 8, cw = 36, gap = 6, ox = 8, ys = [36, 92, 132];
                        let im = 1e-6; for (let k = 0; k < 8 * 64; k++) im = Math.max(im, M.inputs[k]);
                        const km = P08.absMax(M.kern, 1, 0, 72), cm = P08.absMax(M.contrib, 1, 0, 512), zm = P08.absMax(M.z, 1, 0, 64);
                        P08.txt(ctx, `第 ${S.tr.step} 步 · 输出通道 j = ${M.j} · b[j] = ${M.bias.toFixed(3)}`, 40, 4, { bold: true });
                        for (let k = 0; k < n; k++) {
                            const x = ox + k * (cw + gap);
                            P08.drawImg(ctx, M.inputs, x, ys[0], cw, cw, { W: 8, H: 8, offset: k * 64, map: 'inferno', range: [0, im] });
                            P08.drawImg(ctx, M.kern, x + 6, ys[1], cw - 12, cw - 12, { W: 3, H: 3, offset: k * 9, map: 'diverge', range: [-km, km] });
                            P08.drawImg(ctx, M.contrib, x, ys[2], cw, cw, { W: 8, H: 8, offset: k * 64, map: 'diverge', range: [-cm, cm] });
                            P08.txt(ctx, `c${k}`, x + cw / 2, ys[0] - 14, { align: 'center', color: LM.C.dim });
                        }
                        P08.txt(ctx, 'Σ_c ↓ + b[j]', ox, ys[2] + cw + 8, { color: LM.C.gray });
                        P08.drawImg(ctx, M.z, 110, ys[2] + cw + 6, 48, 48, { W: 8, H: 8, map: 'diverge', range: [-zm, zm] });
                        P08.txt(ctx, 'z₂[j]', 164, ys[2] + cw + 24, { color: LM.C.gray });
                        P08.txt(ctx, 'ReLU →', 206, ys[2] + cw + 24, { color: LM.C.gray });
                        P08.drawImg(ctx, M.a, 258, ys[2] + cw + 6, 48, 48, { W: 8, H: 8, map: 'inferno', range: [0, Math.max(1e-6, ...M.a)] });
                        P08.txt(ctx, `8 个贡献之和 + b 与前向算出的 z₂[j] 最大差 ${M.maxDiff.toExponential(1)}（舍入误差）`, 8, ys[2] + cw + 62, { color: LM.C.dim });
                    },
                },
                {
                    id: 'shapes', label: '本页 CNN 的结构表', sub: '一张图的形状变化、参数、乘加（MAC）', size: [360, 236],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), f = P08.fmtN, cols = [6, 118, 210, 280];
                        const rows = [['输入', '1 × 16 × 16', '', ''], ['conv 3×3 1→8', '8 × 16 × 16', 80, 18432], ['ReLU + maxpool', '8 × 8 × 8', 0, 0], ['conv 3×3 8→16', '16 × 8 × 8', 1168, 73728], ['ReLU + maxpool', '16 × 4 × 4', 0, 0], ['flatten → linear', '10', 2570, 2560]];
                        ['层', '输出形状', '参数', '乘加 / 图'].forEach((s, k) => P08.txt(ctx, s, cols[k], 6, { bold: true, color: LM.C.gray }));
                        rows.forEach((r, i) => { const y = 26 + i * 18; r.forEach((v, k) => P08.txt(ctx, k >= 2 ? (v === '' ? '' : f(v)) : String(v), cols[k], y, { color: k === 0 ? LM.C.text : k === 1 ? LM.C.cyan : LM.C.text })); });
                        const y = 26 + rows.length * 18 + 4;
                        ctx.strokeStyle = '#2a2f36'; ctx.beginPath(); ctx.moveTo(6, y - 2); ctx.lineTo(354, y - 2); ctx.stroke();
                        P08.txt(ctx, 'CNN 合计', cols[0], y + 4, { bold: true, color: LM.C.blue }); P08.txt(ctx, f(P08.numParams(S.main.cnn.m.params)), cols[2], y + 4, { bold: true }); P08.txt(ctx, f(94720), cols[3], y + 4, { bold: true });
                        P08.txt(ctx, 'MLP（第 02 篇）', cols[0], y + 22, { bold: true, color: LM.C.orange }); P08.txt(ctx, f(P08.numParams(S.main.mlp.m.params)), cols[2], y + 22, { bold: true }); P08.txt(ctx, f(17024), cols[3], y + 22, { bold: true });
                        P08.txt(ctx, '卷积层：1 248 个参数（33%），却占 97% 的乘加', 6, y + 44, { color: LM.C.dim });
                    },
                },
            ],
            notes: '切换<b>卷积配置</b>：s = 2 的三种配置输出都是 8×8（7×7、s = 2、p = 3 正是 ResNet 第一层的配置）；p = 0 时输出 14×14。第二张图的白点是每个窗口里被保留的位置，反向传播时只有它们拿到梯度。第三张图随 Step 05 的训练实时变化：同一个输出通道对 8 个输入通道各有一个 3×3 核，有的输入通道几乎被忽略（贡献图接近 0）。',
            warn: '这一步的 Sobel-x 型核对任意 K 都是「竖直方向二项式平滑 × 水平方向二项式差分」，K = 3 时就是 Sobel x 的 1/4（各配置都把 Σ|w| 归一化到 2，便于比较）。第四张表：CNN 的参数只有 MLP 的 1/4.5，乘加却是它的 5.6 倍——每个卷积权重在 256 或 64 个位置上被重复使用，这正是 GPU 擅长的计算。',
        },
        // ───────────────────────────── 04 感受野 ─────────────────────────────
        {
            id: 'rf', title: '感受野：一个输出单元到底「看」多大一块输入', tag: 'RF',
            desc: '第 l 层的一个输出只依赖输入上的一块区域，叫它的<strong>感受野</strong>（receptive field）。每叠一层 3×3 卷积，感受野每边多 1 个像素；每经过一次步长 2 的下采样，后面每一层的扩张都翻倍——所以要「看到」整个物体，要么堆很多层，要么不断下采样。理论感受野是所有<em>能</em>影响输出的像素，但它们的影响并不均匀。最直接的测量办法是<strong>用反向传播求输出单元对每个输入像素的梯度</strong>（Luo et al. 2016 称之为<strong>有效感受野</strong>，effective receptive field）。下面是真实计算：32×32 的噪声图上，随机初始化（He）的 4 通道 ReLU 卷积网络中心单元的 |∂y/∂x|（8 张图平均），以及所有权重都是 1/9、没有激活的线性版本——后者正好是方盒函数自卷积 L 次，梯度呈高斯形，半径只按 √L 增长，而理论感受野的半宽按 L 增长。',
            formula: [
                'RF_l = RF_{l−1} + (K_l − 1) · J_{l−1}      J_l = J_{l−1} · s_l   // J = 该层相邻两个输出在输入上相隔几个像素（累积步长）',
                '本页 CNN 的 conv2：1 + (3 − 1)·1 + (2 − 1)·1 + (3 − 1)·2 = 8   // conv(3) → maxpool(2, s = 2) → conv(3)',
                'ERF(p) = E_x[ |∂y_{中心} / ∂x_p| ]   // 一次反向传播就得到所有 p',
                '均匀核 1/9 叠 L 层：每个方向的方差 = L · 2/3  ⇒  r_rms = √(4L/3)   // 而理论感受野的半宽 = L',
            ],
            code: {
                title: 'C++ · receptiveField / effectiveRF / rmsRadius', src: `
                // 理论感受野：从输入往上一层层推。J = 这一层相邻两个输出在输入上隔几个像素（累积步长）
                int receptiveField(const std::vector<int>& K, const std::vector<int>& stride) {
                    int rf = 1, J = 1;
                    for (size_t l = 0; l < K.size(); l++) { rf += (K[l] - 1) * J; J *= stride[l]; }
                    return rf;                                                   // 本页 CNN 的 conv2：K = {3, 2, 3}，s = {1, 2, 1} → 8
                }

                // 有效感受野（Luo et al. 2016）：nImg 张 N×N 噪声图穿过 L 层「3×3 卷积 + ReLU」，
                // 对每张图正中心那个输出位置（所有通道之和）求 ∂y/∂x，再对 |·| 取平均
                std::vector<float> effectiveRF(const std::vector<ConvLayer>& layers, int L, int N, int nImg, Rng& r) {
                    Var X = param(randn(nImg, N * N, 1.f, r));                   // 输入也设成 param：backward 之后 X->grad = ∂y/∂x
                    Var h = X;
                    for (int l = 0; l < L; l++) h = relu(applyConv(layers[l], h));
                    int C = layers[L - 1].W->val.rows, HW = N * N, center = (N / 2) * N + N / 2;
                    Tensor sel = zeros(nImg, C * HW);
                    for (int n = 0; n < nImg; n++)
                        for (int c = 0; c < C; c++) sel.v[(size_t)n * C * HW + c * HW + center] = 1.f;
                    backward(sumAll(mul(h, constant(sel))));                     // y = Σ_n Σ_c h[n, c, 中心]
                    std::vector<float> erf = floats(HW);
                    for (int n = 0; n < nImg; n++)
                        for (int k = 0; k < HW; k++) erf[k] += std::fabs(X->grad.v[(size_t)n * HW + k]) / nImg;
                    return erf;
                }

                // ERF 的「半径」：以梯度大小为权重，到中心距离的均方根
                float rmsRadius(const std::vector<float>& g, int N) {
                    double s = 0, s2 = 0;
                    for (int y = 0; y < N; y++)
                        for (int x = 0; x < N; x++) {
                            double v = g[y * N + x], d2 = (y - N / 2) * (y - N / 2) + (x - N / 2) * (x - N / 2);
                            s += v; s2 += v * d2;
                        }
                    return (float)std::sqrt(s2 / std::max(s, 1e-30));
                }`,
            },
            previews: [
                {
                    id: 'erfGrid', label: '有效感受野 |∂y_中心 / ∂x| vs 理论感受野（方框）', sub: '32×32 噪声输入（画中心 24×24）· L = 层数 · 每张图单独归一化（√ 拉伸）· 黄框边长 = 2L + 1', large: true, size: [360, 250], needs: ['erf'],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), E = S.erf, Ls = P08.ERF_LS, n = Ls.length, cell = 46, gap = 4, ox = 6, N = P08.ERF_N;
                        const row = (maps, y, rad, title, col) => {
                            P08.txt(ctx, title, ox, y - 16, { color: col });
                            maps.forEach((g, k) => {
                                const x = ox + k * (cell + gap), m = Math.max(1e-30, ...g), Nc = 24, o0 = (N - Nc) / 2, img = new Float32Array(Nc * Nc);
                                for (let yy = 0; yy < Nc; yy++) for (let xx = 0; xx < Nc; xx++) img[yy * Nc + xx] = Math.sqrt(g[(yy + o0) * N + xx + o0] / m);
                                P08.drawImg(ctx, img, x, y, cell, cell, { W: Nc, H: Nc, map: 'inferno', range: [0, 1] });
                                const L = Ls[k], side = (2 * L + 1) * cell / Nc, c0 = x + (Nc / 2 + 0.5) * cell / Nc;
                                ctx.strokeStyle = 'rgba(232,194,90,0.85)'; ctx.lineWidth = 1; ctx.strokeRect(c0 - side / 2, y + (Nc / 2 + 0.5) * cell / Nc - side / 2, side, side);
                                P08.txt(ctx, `r ${rad[k].toFixed(2)}`, x + cell / 2, y + cell + 2, { align: 'center', color: LM.C.dim });
                            });
                        };
                        Ls.forEach((L, k) => P08.txt(ctx, `L = ${L}`, ox + k * (cell + gap) + cell / 2, 4, { align: 'center', color: LM.C.gray }));
                        row(E.rand, 40, E.rRand, '随机 He 初始化 + ReLU（4 通道，8 张噪声图平均）', LM.C.orange);
                        row(E.box, 136, E.rBox, '所有权重 = 1/9、没有激活（线性）', LM.C.blue);
                        P08.txt(ctx, '黄框 = 理论感受野（L = 10 时 21×21）；r = 均方根半径（像素）', ox, 214, { color: LM.C.dim });
                    },
                },
                {
                    id: 'erfCurve', label: '有效感受野的半径 vs 层数', sub: '实线 = 实测（反向传播算出的梯度的均方根半径）· 蓝虚线 = √(4L/3) · 灰线 = 理论感受野半宽 L', size: [360, 210], needs: ['erf'],
                    render: (c, S) => {
                        const E = S.erf, Ls = P08.ERF_LS, xs = []; for (let L = 1; L <= 10; L += 0.25) xs.push(L);
                        PL.draw.plot(c, {
                            x: [0.5, 10.5], y: [0, 10.5], xLabel: '层数 L（3×3 卷积）',
                            series: [
                                { data: xs.map((L) => L), x: xs, color: '#4a5058', width: 1.5 },
                                { data: xs.map((L) => Math.sqrt(4 * L / 3)), x: xs, color: LM.C.blue, dash: [5, 4], width: 1.2 },
                                { data: E.rBox, x: Ls, color: LM.C.blue, width: 2 },
                                { data: E.rRand, x: Ls, color: LM.C.orange, width: 2 },
                            ],
                        });
                        const { ctx } = PL.draw.ctx(c, { keep: true });
                        P08.legend(ctx, [{ label: '理论感受野半宽 L', color: '#6a7078' }, { label: '均匀核（线性）实测', color: LM.C.blue }, { label: '√(4L/3)', color: LM.C.blue, dash: [5, 4] }, { label: '随机 ReLU 网络实测', color: LM.C.orange }], 56, 16, 'left');
                    },
                },
                {
                    id: 'erfTrained', label: '训练中的 CNN：conv2 中心单元对这张图的梯度', sub: '左 = 输入 + 梯度（红）· 右 = |∂y/∂x| · 黄框 = 理论感受野 8×8（输入的第 5–12 行 / 列）', live: true, size: [360, 220],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), g = S.cerf, cs = 9, ox1 = 10, ox2 = 196, oy = 26, a = Float32Array.from(g, Math.abs), m = Math.max(1e-12, ...a);
                        let nz = 0; for (let k = 0; k < 256; k++) if (a[k] > 1e-9) nz++;
                        P08.txt(ctx, `第 ${S.tr.step} 步 · ∂(Σ_c z₂[c, 4, 4]) / ∂x · 非零 ${nz} / 64 个像素`, 40, 4, { bold: true });
                        P08.drawImg(ctx, S.tr.x, ox1, oy, 16 * cs, 16 * cs);
                        for (let k = 0; k < 256; k++) { if (a[k] <= 1e-9) continue; ctx.fillStyle = `rgba(255,70,60,${(0.25 + 0.75 * Math.sqrt(a[k] / m)).toFixed(3)})`; ctx.fillRect(ox1 + (k % 16) * cs, oy + Math.floor(k / 16) * cs, cs, cs); }
                        P08.drawImg(ctx, Float32Array.from(a, (v) => Math.sqrt(v / m)), ox2, oy, 16 * cs, 16 * cs, { map: 'inferno', range: [0, 1] });
                        ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 1.5; [ox1, ox2].forEach((ox) => ctx.strokeRect(ox + 5 * cs, oy + 5 * cs, 8 * cs, 8 * cs));
                        P08.txt(ctx, '梯度全部落在黄框（理论感受野）内，靠近中心的像素梯度更大', 8, oy + 16 * cs + 8, { color: LM.C.dim });
                    },
                },
            ],
            notes: '上排随机网络的梯度斑点比黄框（理论感受野）小得多、形状也不规则；下排均匀核的半径与 √(4L/3) 精确吻合（第二张图的蓝线落在虚线上）。随机 ReLU 网络增长得稍快，但同样远低于 L。第三张图是 Step 05 正在训练的 CNN，换<b>样本 #</b> 看它随输入变化：它依赖输入，因为 ReLU 和 maxpool 选择了让哪些路径通过。',
            warn: '「有效感受野按 √L 增长、远小于理论感受野」是 Luo et al. 2016 对线性 / 随机初始化网络的结论；训练会让它变大一些，下采样和空洞卷积（dilated convolution）可以更快地扩大感受野。任务需要全局信息（判断整张图的布局）时，要保证最后几层的感受野足够大，或者用全局池化 / 注意力。',
        },
        // ───────────────────────────── 05 训练 CNN ─────────────────────────────
        {
            id: 'train', title: '训练一个小 CNN：3 818 个参数 vs MLP 的 17 098 个', tag: 'TRAIN',
            desc: '把卷积、池化和一个全连接层串起来，就是 LeNet 式的小 CNN：<code>conv(1→8) → ReLU → maxpool → conv(8→16) → ReLU → maxpool → flatten → linear(256→10)</code>。训练和第 02 篇完全一样——交叉熵、反向传播、mini-batch——只是梯度要穿过卷积层：卷积的反向传播还是卷积（dW 是输入与输出梯度的互相关，dx 是输出梯度与翻转核的卷积），自动微分替我们做了。对照组是第 02 篇的 MLP（256 → 64 → 10）：两个模型用<strong>同一串 batch、同一个优化器</strong>（Adam，lr = 3·10⁻³ 余弦衰减，batch 32），在页面里同时实时训练 1500 步。CNN 的参数只有 MLP 的 1/4.5，测试准确率却明显更高、错误更少（第二张图用对数纵轴画测试错误率）。第一层学到的 8 个 3×3 核（第三张图）从随机数逐渐变成方向各异的边缘 / 笔画检测器。',
            formula: [
                'CNN：x[1×16×16] → conv 3×3 (8) → ReLU → maxpool → [8×8×8] → conv 3×3 (16) → ReLU → maxpool → [16×4×4] → linear → z[10]',
                'L = −(1/B) Σ_b log softmax(z_b)[y_b]   // 与第 02 篇相同的交叉熵',
                '∂L/∂w[c_o, c_i, u, v] = Σ_{b, i, j} δ[b, c_o, i, j] · x[b, c_i, i + u − p, j + v − p]   // 卷积的梯度也是一次互相关，δ = ∂L/∂y',
                'Adam(β_1 = 0.9, β_2 = 0.999)，η = 3·10^{−3} 余弦衰减到 5%，batch 32，1500 步 = 12 个 epoch   // 两个模型共用一串 batch',
            ],
            code: {
                title: 'C++ · CNN / MLP / trainClassifier / accuracy', src: `
                // 本页的 CNN（3 818 个参数）：conv(1→8) → ReLU → maxpool → conv(8→16) → ReLU → maxpool → flatten → linear(256→10)
                struct CNN { ConvLayer c1, c2; Linear fc; };

                CNN makeCNN(Rng& r) {
                    CNN m;
                    m.c1 = makeConv(1, 8, 3, 16, 16, r);                         // 8 个 1×3×3 核
                    m.c2 = makeConv(8, 16, 3, 8, 8, r);                          // 16 个 8×3×3 核
                    m.fc = makeLinear(16 * 4 * 4, 10, std::sqrt(2.f / 256), r);  // He 初始化
                    return m;
                }

                Var cnnForward(const CNN& m, const Var& x) {                     // x: [B × 256]
                    Var h = maxPool2(relu(applyConv(m.c1, x)), 8, 16, 16);        // [B × 8·8·8]
                    h = maxPool2(relu(applyConv(m.c2, h)), 16, 8, 8);             // [B × 16·4·4]：每一行就是 flatten 后的 256 维
                    return forward(m.fc, h);                                      // logits [B × 10]
                }

                // 第 02 篇的 MLP（17 098 个参数）：256 → 64 → ReLU → 10
                struct MLP { Linear fc1, fc2; };

                MLP makeMLP(Rng& r) {
                    MLP m;
                    m.fc1 = makeLinear(256, 64, std::sqrt(2.f / 256), r);
                    m.fc2 = makeLinear(64, 10, std::sqrt(2.f / 64), r);
                    return m;
                }

                Var mlpForward(const MLP& m, const Var& x) { return forward(m.fc2, relu(forward(m.fc1, x))); }

                // 两种模型共用一个训练循环：isCNN 决定走哪个前向
                struct Classifier { bool isCNN = true; CNN cnn; MLP mlp; };

                Var logitsOf(const Classifier& m, const Var& x) { return m.isCNN ? cnnForward(m.cnn, x) : mlpForward(m.mlp, x); }

                std::vector<Var> paramsOf(const Classifier& m) {
                    if (m.isCNN) return {m.cnn.c1.W, m.cnn.c1.b, m.cnn.c2.W, m.cnn.c2.b, m.cnn.fc.W, m.cnn.fc.b};
                    return {m.mlp.fc1.W, m.mlp.fc1.b, m.mlp.fc2.W, m.mlp.fc2.b};
                }

                // 训练：Adam（β₂ = 0.999，不做权重衰减），lr = 3e-3 余弦衰减到 5%，batch 32，按 epoch 洗牌。
                // aug > 0 时每张图随机平移 ±aug 像素（Step 07 的数据增强）
                void trainClassifier(Classifier& m, const Dataset& tr, int steps, int aug, Rng& r) {
                    std::vector<Var> ps = paramsOf(m);
                    std::vector<bool> noDecay = std::vector<bool>(ps.size(), false);
                    AdamW opt = makeAdamW(ps, noDecay, 3e-3f, 0.f, 0.9f, 0.999f);
                    std::vector<int> order = permutation(tr.n, r);
                    int pos = 0;
                    for (int t = 0; t < steps; t++) {
                        std::vector<int> idx = ints(32);
                        for (int b = 0; b < 32; b++) {
                            if (pos == tr.n) { order = permutation(tr.n, r); pos = 0; }
                            idx[b] = order[pos++];
                        }
                        Tensor Xb = rowsOf(tr.X, idx);
                        if (aug > 0) Xb = augmentBatch(Xb, aug, r);
                        zeroGrad(ps);
                        Var loss = crossEntropy(logitsOf(m, constant(Xb)), labelsOf(tr.y, idx));
                        backward(loss);                                           // 卷积的反向传播写在库里的 conv2d 中
                        adamwStep(opt, cosineLR(t, 3e-3f, 1.5e-4f, 0, steps));
                    }
                }

                // 测试准确率（不求梯度，50 张一批）；dx、dy ≠ 0 时先把每张测试图平移（Step 07）
                float accuracy(const Classifier& m, const Dataset& te, int dx, int dy) {
                    NoGrad ng;
                    int ok = 0;
                    for (int s = 0; s < te.n; s += 50) {
                        int n = std::min(50, te.n - s);
                        Tensor Xb = zeros(n, 256);
                        for (int k = 0; k < n; k++) {
                            std::vector<float> img = shiftImage(rowOf(te.X, s + k), dx, dy, 1, 16, 16);
                            for (int j = 0; j < 256; j++) Xb.v[(size_t)k * 256 + j] = img[j];
                        }
                        Tensor z = logitsOf(m, constant(Xb))->val;
                        for (int k = 0; k < n; k++) ok += argmaxIndex(rowOf(z, k)) == te.y[s + k];
                    }
                    return (float)ok / te.n;
                }`,
            },
            previews: [
                {
                    id: 'acc', label: '准确率：CNN vs MLP（实时训练）', sub: '实线 = CNN、虚线 = MLP · 蓝 = 训练集（固定抽 500 张评估）· 橙 = 测试集 500 张', live: true, large: true, size: [360, 250],
                    render: (c, S) => {
                        const C = S.main.cnn, M = S.main.mlp, hc = C.hist, hm = M.hist;
                        P08.curves(c, [
                            { x: hm.step, y: hm.trainA.map((v) => 100 * v), color: LM.C.train, dash: [5, 4], width: 1.5, label: `MLP 训练 ${P08.pct(P08.last(hm.trainA))}` },
                            { x: hm.step, y: hm.testA.map((v) => 100 * v), color: LM.C.val, dash: [5, 4], width: 1.8, label: `MLP 测试 ${P08.pct(P08.last(hm.testA))}` },
                            { x: hc.step, y: hc.trainA.map((v) => 100 * v), color: LM.C.train, width: 2, label: `CNN 训练 ${P08.pct(P08.last(hc.trainA))}` },
                            { x: hc.step, y: hc.testA.map((v) => 100 * v), color: LM.C.val, width: 2.5, label: `CNN 测试 ${P08.pct(P08.last(hc.testA))}` },
                        ], { x: [0, P08.CAP], y: [0, 100], xLabel: '训练步数', legendAt: [1, 0.42], extra: [{ label: P08.status(C), color: LM.C.yellow, noLine: true }] });
                    },
                },
                {
                    id: 'err', label: '测试错误率（对数纵轴）', sub: '同一组评估点 · 后期的差距在对数坐标下才看得清', live: true, size: [360, 200],
                    render: (c, S) => {
                        const hc = S.main.cnn.hist, hm = S.main.mlp.hist, e = (a) => a.map((v) => Math.max(0.2, 100 * (1 - v)));
                        P08.curves(c, [
                            { x: hm.step, y: e(hm.testA), color: LM.C.purple, width: 2, dash: [5, 4], label: `MLP ${(100 * (1 - P08.last(hm.testA))).toFixed(1)}%` },
                            { x: hc.step, y: e(hc.testA), color: LM.C.green, width: 2.5, label: `CNN ${(100 * (1 - P08.last(hc.testA))).toFixed(1)}%` },
                        ], { x: [0, P08.CAP], y: [0.5, 100], logY: true, xLabel: `训练步数（第 ${S.main.cnn.step} 步）`, legendAt: [1, 0] });
                    },
                },
                {
                    id: 'kernels', label: '第一层的 8 个卷积核（实时）', sub: '每个 3×3 核单独归一化 · 橙 = 正、蓝 = 负 · 下排 = 它对选中样本的响应（ReLU 前）', live: true, size: [360, 150],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), W = S.main.cnn.m.conv1.w.data, n = 8, cw = 40, gap = 4, ox = 8;
                        P08.txt(ctx, `conv1 · 第 ${S.main.cnn.step} 步`, 40, 4, { bold: true, color: LM.C.yellow });
                        let zm = 1e-6; for (let k = 0; k < 8 * 256; k++) zm = Math.max(zm, Math.abs(S.tr.z1[k]));
                        for (let k = 0; k < n; k++) {
                            const x = ox + k * (cw + gap), m = P08.absMax(W, 1, k * 9, 9);
                            P08.drawImg(ctx, W, x, 26, cw, cw, { W: 3, H: 3, offset: k * 9, map: 'diverge', range: [-m, m] });
                            P08.drawImg(ctx, S.tr.z1, x, 76, cw, cw, { offset: k * 256, map: 'diverge', range: [-zm, zm] });
                            P08.txt(ctx, `#${k}`, x + cw / 2, 120, { align: 'center', color: LM.C.dim });
                        }
                    },
                },
                {
                    id: 'compare', label: '两个模型的对照（实时）', sub: '参数量、一张图的乘加、当前的测试准确率与错误数', live: true, size: [360, 130],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), C = S.main.cnn, M = S.main.mlp, f = P08.fmtN, err = (R) => Math.round(500 * (1 - P08.last(R.hist.testA))), cols = [8, 58, 128, 214, 292];
                        P08.txt(ctx, `评估点：第 ${C.evalStep} 步的权重快照`, 40, 4, { bold: true });
                        ['', '参数', '乘加 / 图', '测试准确率', '错误'].forEach((t, k) => P08.txt(ctx, t, cols[k], 26, { color: LM.C.gray }));
                        [['CNN', LM.C.green, C, 94720], ['MLP', LM.C.purple, M, 17024]].forEach(([nm, col, R, mac], r) => {
                            const y = 46 + r * 20;
                            P08.txt(ctx, nm, cols[0], y, { bold: true, color: col });
                            P08.txt(ctx, f(P08.numParams(R.m.params)), cols[1], y);
                            P08.txt(ctx, f(mac), cols[2], y);
                            P08.txt(ctx, P08.pct(P08.last(R.hist.testA)), cols[3], y, { bold: true, color: col });
                            P08.txt(ctx, `${err(R)} / 500`, cols[4], y);
                        });
                        P08.txt(ctx, P08.status(C), 8, 92, { color: LM.C.yellow });
                        P08.txt(ctx, '第 02 篇的 MLP（SGD、3000 步）在同一测试集上 95.2%', 8, 110, { color: LM.C.dim });
                    },
                },
                {
                    id: 'cnnErrors', label: 'CNN 判错的测试样本', sub: '标签 = 真实 → 预测 · <b>点击</b>选中，到 Step 06 看它的逐层特征图', live: true, size: [360, 172],
                    render: (c, S) => {
                        const R = S.main.cnn, d = S.data, pred = R.pred[1], list = []; let nErr = 0;
                        for (let i = 0; i < d.te.n; i++) if (pred[i] !== d.te.y[i]) { nErr++; if (list.length < 30) list.push(i); }
                        S._gE = P08.imgGrid(c, list.map((i) => ({ img: d.te.X.subarray(i * 256, (i + 1) * 256), label: `${d.te.y[i]}→${pred[i]}`, labelColor: LM.C.red, outline: i === S.tr.i ? LM.C.yellow : null })), { cols: 10, labels: true, title: `第 ${R.evalStep} 步的评估：判错 ${nErr} / ${d.te.n} 张（最多显示 30 张）`, titleColor: LM.C.gray, tx: 40 });
                        S._gEidx = list;
                    },
                    probe: (x, y, S, PL, info) => { const k = P08.gridHit(S._gE, info); return k < 0 ? '' : `测试样本 #${S._gEidx[k]}（点击选中）`; },
                    pick: (x, y, S, PL, info) => { const k = P08.gridHit(S._gE, info); if (k >= 0) PL.setParam('sample', S._gEidx[k]); },
                },
            ],
            notes: '训练一开始，CNN 的准确率就爬得比 MLP 快。Node 里用页面同样的设置训练到 1500 步（tools/p08_final.js）：CNN 测试 97.6%（500 张里错 12 张）、MLP 95.6%（错 22 张）——错误率差不多减半，两者的训练准确率都在 99% 以上。第三张图的 8 个核从 He 初始化的随机数出发，逐渐变成不同方向的边缘 / 笔画检测器，下排是它们在选中样本上的响应。调<b>每帧步数</b>可以加快训练；<b>重新训练</b> 从第 0 步开始。',
            warn: '页面里的 JS 参考实现用了一个更快的卷积内核（补零后的扁平内存布局 + 3×3 展开，与库里的 <code>conv2d</code> / <code>LM.AG.conv2d</code> 逐元素一致，tools/p08_conv_check.js 校验过），CNN 每步约 4 ms（Node）；C++ 代码块用的是库里最直白的 <code>conv2d</code>，结果相同。准确率曲线上的点来自分帧评估：先拷一份权重快照，每帧评估 20 张，评估完 1000 张记一个点（所以曲线比训练步数晚一点）。',
        },
        // ───────────────────────────── 06 逐层特征图 ─────────────────────────────
        {
            id: 'features', title: '逐层特征图：一张图在 CNN 里变成了什么', tag: 'FEATURES',
            desc: '选中的测试样本穿过<strong>正在训练</strong>的 CNN，每一层的真实输出。第一层的 8 个通道各自对一种局部笔画方向敏感（对照 Step 05 的 8 个核）；maxpool 把分辨率减半；第二层的 16 个通道组合第一层的特征，开始对更大的局部形状（拐角、一段圆弧、交叉点）有响应；第二次池化后剩下 16 张 4×4 的小图——这 256 个数就是全连接层看到的「特征向量」，与第 02 篇 MLP 直接看 256 个像素形成对比。越往后，空间分辨率越低、通道数（特征种类）越多，图越「抽象」：几乎所有 CNN 都是这个形态。',
            formula: [
                'a_1 = ReLU(conv_1(x)) ∈ R^{8×16×16}  →  p_1 = maxpool(a_1) ∈ R^{8×8×8}',
                'a_2 = ReLU(conv_2(p_1)) ∈ R^{16×8×8}  →  p_2 = maxpool(a_2) ∈ R^{16×4×4}  →  z = vec(p_2) · W + b ∈ R^{10}',
            ],
            code: {
                title: 'C++ · cnnTrace / channelOf', src: `
                // 前向时把每一层的输出留下来（画特征图用）
                struct CnnTrace { Var a1, p1, a2, p2, logits; };

                CnnTrace cnnTrace(const CNN& m, const Var& x) {
                    CnnTrace t;
                    t.a1 = relu(applyConv(m.c1, x));                             // [1 × 8·16·16]
                    t.p1 = maxPool2(t.a1, 8, 16, 16);                            // [1 × 8·8·8]
                    t.a2 = relu(applyConv(m.c2, t.p1));                          // [1 × 16·8·8]
                    t.p2 = maxPool2(t.a2, 16, 8, 8);                             // [1 × 16·4·4]
                    t.logits = forward(m.fc, t.p2);
                    return t;
                }

                // 取出第 c 个通道（一个 H×W 平面）
                std::vector<float> channelOf(const Var& fmap, int c, int H, int W) {
                    std::vector<float> a = floats(H * W);
                    for (int k = 0; k < H * W; k++) a[k] = fmap->val.v[(size_t)c * H * W + k];
                    return a;
                }`,
            },
            previews: [
                {
                    id: 'fmaps', label: '逐层特征图（训练中的 CNN）', sub: '每一层在本层所有通道之间共用一个色阶（亮 = 激活大）· 一行一层，形状写在左边', live: true, large: true, size: [360, 296],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), t = S.tr, lab = 70, gap = 3;
                        P08.txt(ctx, `样本 #${t.i}（标签 ${t.y}）→ CNN 预测 ${t.pred}（p = ${t.p[t.pred].toFixed(3)}）· 第 ${t.step} 步`, 40, 4, { bold: true, color: t.pred === t.y ? LM.C.green : LM.C.red });
                        const layer = (data, C, Wd, y, cols, cell, name, shape) => {
                            let m = 1e-6; for (let k = 0; k < C * Wd * Wd; k++) m = Math.max(m, data[k]);
                            P08.txt(ctx, name, 6, y + 2, { color: LM.C.text }); P08.txt(ctx, shape, 6, y + 16, { color: LM.C.cyan });
                            for (let k = 0; k < C; k++) { const x = lab + (k % cols) * (cell + gap), yy = y + Math.floor(k / cols) * (cell + gap); P08.drawImg(ctx, data, x, yy, cell, cell, { W: Wd, H: Wd, offset: k * Wd * Wd, map: 'inferno', range: [0, m] }); }
                            return y + Math.ceil(C / cols) * (cell + gap) + 8;
                        };
                        P08.txt(ctx, '输入', 6, 26, { color: LM.C.text }); P08.txt(ctx, '1×16×16', 6, 40, { color: LM.C.cyan });
                        P08.drawImg(ctx, t.x, lab, 24, 32, 32);
                        let y = 64;
                        y = layer(t.a1, 8, 16, y, 8, 32, 'conv1+ReLU', '8×16×16');
                        y = layer(t.p1, 8, 8, y, 8, 32, 'maxpool', '8×8×8');
                        y = layer(t.a2, 16, 8, y, 8, 32, 'conv2+ReLU', '16×8×8');
                        layer(t.p2, 16, 4, y, 8, 22, 'maxpool', '16×4×4');
                    },
                },
                {
                    id: 'probs', label: '输出概率：CNN vs MLP', sub: '同一张图、两个正在训练的模型 · 绿框 = 正确类别', live: true, size: [360, 190],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c), t = S.tr, padL = 34, top = 26, ph = h - top - 26, bw = (w - padL - 8) / 10;
                        P08.txt(ctx, `CNN 预测 ${t.pred}（p = ${t.p[t.pred].toFixed(2)}）· MLP 预测 ${t.predM}（p = ${t.pm[t.predM].toFixed(2)}）`, 40, 4, { bold: true });
                        [0, 0.5, 1].forEach((v) => { const y = top + (1 - v) * ph; ctx.strokeStyle = '#1c1c1c'; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - 8, y); ctx.stroke(); P08.txt(ctx, v.toFixed(1), padL - 4, y, { align: 'right', base: 'middle', color: LM.C.dim }); });
                        for (let k = 0; k < 10; k++) {
                            const x = padL + k * bw;
                            ctx.fillStyle = LM.C.green; ctx.fillRect(x + bw * 0.12, top + (1 - t.p[k]) * ph, bw * 0.36, Math.max(1, t.p[k] * ph));
                            ctx.fillStyle = LM.C.purple; ctx.fillRect(x + bw * 0.52, top + (1 - t.pm[k]) * ph, bw * 0.36, Math.max(1, t.pm[k] * ph));
                            if (k === t.y) { ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 1.5; ctx.strokeRect(x + 1, top - 2, bw - 2, ph + 4); }
                            P08.txt(ctx, String(k), x + bw / 2, h - 22, { align: 'center', color: k === t.y ? LM.C.yellow : LM.C.gray });
                        }
                        P08.legend(ctx, [{ label: 'CNN', color: LM.C.green }, { label: 'MLP', color: LM.C.purple }], w - 10, top, 'right');
                    },
                },
            ],
            notes: '换<b>样本 #</b>（或在 Step 05 点一个错例）看不同的数字。每一层的色阶在本层所有通道之间共享，所以能比较通道的强弱：有些通道对这个数字几乎完全不响应（整块黑）——ReLU 之后的激活是稀疏的。训练刚开始时，特征图就是随机核的边缘响应；随着训练，各通道逐渐分工。',
        },
        // ───────────────────────────── 07 平移鲁棒性 ─────────────────────────────
        {
            id: 'shift', title: '平移鲁棒性与数据增强：CNN 并不自动「平移不变」', tag: 'ROBUST', params: ['aug', 'spf', 'sample'],
            desc: '把测试集整体平移 d 个像素（空出来的地方补 0）再测准确率。MLP 掉得最快：它的每个权重都绑定在固定的像素位置上。CNN 好得多——卷积层是等变的，两次 2×2 池化又吸收了一部分小位移——但它<strong>远没有做到不变</strong>：最后的全连接层读的仍是 4×4 网格上固定位置的特征，平移 3 个像素准确率就明显下降。真正有效的办法是<strong>数据增强</strong>（data augmentation）：训练时随机平移每张图，让模型见过各种位置。下面两个增强版本与 Step 05 的两个模型初始化相同、batch 顺序相同，只是训练图被随机平移 ±a 像素（滚动到这一步时开始训练）。四个模型在同一组平移后的测试图上评估（每帧推进一小块，评估完一轮刷新一次曲线）。',
            formula: [
                'acc(d) = (1/n) Σ_i [ argmax f(T_d x_i) = y_i ]   // T_d = 整张图平移 d 像素、空出来的补 0；本页训练中 n = 40，训练完 n = 100',
                '数据增强：每一步、每张图独立地 x ← T_{(dx, dy)} x，dx, dy ~ U{−a, …, a}',
            ],
            code: {
                title: 'C++ · shiftCurve / robustnessExperiment', src: `
                // 平移鲁棒性：把整个测试集水平平移 d = −maxShift…maxShift 个像素，每个 d 的准确率
                struct ShiftCurve { std::vector<int> d; std::vector<float> acc; };

                ShiftCurve shiftCurve(const Classifier& m, const Dataset& te, int maxShift) {
                    ShiftCurve c;
                    for (int d = -maxShift; d <= maxShift; d++) { c.d.push_back(d); c.acc.push_back(accuracy(m, te, d, 0)); }
                    return c;
                }

                // 本步的四个模型：同样的初始化、同样的 batch 顺序，区别只在结构（MLP / CNN）和训练时有没有随机平移
                void robustnessExperiment(const Dataset& tr, const Dataset& te, int aug) {
                    for (int k = 0; k < 4; k++) {
                        Rng r = makeRng(7), init = makeRng(81);
                        Classifier m;
                        m.isCNN = k % 2 == 1;
                        if (m.isCNN) m.cnn = makeCNN(init); else m.mlp = makeMLP(init);
                        trainClassifier(m, tr, 1500, k >= 2 ? aug : 0, r);
                        ShiftCurve c = shiftCurve(m, te, 4);
                        std::printf("%s%s:", m.isCNN ? "CNN" : "MLP", k >= 2 ? " + 增强" : "");
                        for (float a : c.acc) std::printf(" %.0f%%", 100 * a);
                        std::printf("\\n");
                    }
                }`,
            },
            previews: [
                {
                    id: 'shiftX', label: '水平平移后的测试准确率', sub: '测试图整体平移 dx 像素（训练中每轮 40 张，全部训练完后 100 张）· 竖虚线 = 增强范围 ±a · 实线 = CNN、虚线 = MLP', live: true, large: true, size: [360, 260], needs: ['augShift'],
                    render: (c, S) => {
                        const V = P08.shiftView(S.E), cols = [LM.C.purple, LM.C.green, LM.C.orange, LM.C.blue];
                        const names = ['MLP', 'CNN', `MLP + 增强 ±${S.A.aug}`, `CNN + 增强 ±${S.A.aug}`], dash = [[5, 4], null, [5, 4], null];
                        if (!V) { const { ctx } = PL.draw.ctx(c); P08.txt(ctx, '平移测试开始…', 40, 8, { color: LM.C.yellow }); return; }
                        const series = [0, 1, 2, 3].map((k) => { const cv = P08.shiftCurve(V.acc[k], 'x'); return { x: cv.x, y: cv.y.map((v) => 100 * v), color: cols[k], dash: dash[k], width: k % 2 ? 2.5 : 2, label: `${names[k]}（第 ${V.steps[k]} 步）` }; });
                        const xl = V.partial ? `dx（像素）· 第一轮评估中：每个平移量已评估 ${V.n} 张` : `dx（像素）· 每个平移量 ${V.n} 张 · 增强模型训练到第 ${S.A.cnn.step} 步`;
                        P08.curves(c, series, { x: [-4, 4], y: [0, 100], xLabel: xl, marks: [{ x: -S.A.aug, color: '#4a5058' }, { x: S.A.aug, color: '#4a5058' }], legendAt: [0.5, 0.62], legendAlign: 'center' });
                    },
                },
                {
                    id: 'shiftY', label: '竖直平移', sub: '同样的四个模型（颜色同上）· dy > 0 = 向下 · 数字上下的空白比左右少，竖直平移更快把笔画推出画面', live: true, size: [360, 200], needs: ['augShift'],
                    render: (c, S) => {
                        const V = P08.shiftView(S.E), cols = [LM.C.purple, LM.C.green, LM.C.orange, LM.C.blue], dash = [[5, 4], null, [5, 4], null];
                        if (!V) { const { ctx } = PL.draw.ctx(c); P08.txt(ctx, '平移测试开始…', 40, 8, { color: LM.C.yellow }); return; }
                        P08.curves(c, [0, 1, 2, 3].map((k) => { const cv = P08.shiftCurve(V.acc[k], 'y'); return { x: cv.x, y: cv.y.map((v) => 100 * v), color: cols[k], dash: dash[k], width: k % 2 ? 2.5 : 2 }; }),
                            { x: [-3, 3], y: [0, 100], xLabel: `dy（像素）· ${V.partial ? `第一轮：每个平移量 ${V.n} 张` : `增强模型第 ${S.A.cnn.step} 步`}`, marks: [{ x: -S.A.aug, color: '#4a5058' }, { x: S.A.aug, color: '#4a5058' }] });
                    },
                },
                {
                    id: 'shiftDemo', label: '选中的样本左右平移：四个模型的预测', sub: '绿 = 对、红 = 错 · 四个模型都是此刻的权重', live: true, size: [360, 160], needs: ['shiftDemo'],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), sd = S.sd, n = 9, cell = 28, gap = 2, ox = 84, oy = 24, y = S.tr.y;
                        const names = ['MLP', 'CNN', 'MLP+增强', 'CNN+增强'], cols = [LM.C.purple, LM.C.green, LM.C.orange, LM.C.blue];
                        P08.txt(ctx, `样本 #${S.tr.i}（标签 ${y}）· dx = −4 … +4`, 40, 4, { bold: true });
                        for (let k = 0; k < n; k++) {
                            const x = ox + k * (cell + gap);
                            P08.drawImg(ctx, sd.imgs, x, oy, cell, cell, { offset: k * 256 });
                            P08.txt(ctx, String(k - 4), x + cell / 2, oy + cell + 1, { align: 'center', color: LM.C.dim });
                        }
                        names.forEach((nm, j) => {
                            const yy = oy + cell + 20 + j * 18;
                            P08.txt(ctx, nm, 6, yy, { color: cols[j] });
                            for (let k = 0; k < n; k++) { const p = sd.pred[j][k]; P08.txt(ctx, String(p), ox + k * (cell + gap) + cell / 2, yy, { align: 'center', bold: true, color: p === y ? LM.C.green : LM.C.red }); }
                        });
                    },
                },
                {
                    id: 'augBatch', label: '数据增强：CNN + 增强 当前这一步的 batch', sub: '每张图被独立地随机平移（空出来的补 0）· 数字 = 标签', live: true, size: [360, 104], needs: ['augShift'],
                    render: (c, S) => { const R = S.A.cnn; P08.imgGrid(c, Array.from({ length: 16 }, (_, k) => ({ img: R.Xb.subarray(k * 256, (k + 1) * 256), label: R.yb[k] })), { cols: 16, labels: true, title: `第 ${R.step} 步 · 平移 ±${S.A.aug} 像素`, titleColor: LM.C.gray, tx: 40 }); },
                },
            ],
            notes: 'Node 里用页面同样的设置把四个模型都训练到 1500 步（增强 ±2，tools/p08_final.js），前 100 张测试图上：水平平移 ±3 像素的平均准确率 MLP 44%、CNN 74%、MLP + 增强 86%、CNN + 增强 91%；±4 像素时 MLP 20%、CNN 50%、CNN + 增强 71%（增强只覆盖到 ±2）。拖<b>增强幅度 ±a</b> 重新训练两个增强模型：a = 3 时 MLP + 增强在没有平移的测试图上反而降到 91–93%（容量不够同时记住所有位置），CNN + 增强几乎不受影响（98–99%）。第三张图把选中的样本左右平移，看四个模型从哪里开始认错。',
            warn: '池化只对小于窗口的位移近似不变，而且有混叠（aliasing）：步长 2 的下采样让「平移 1 个像素」的特征变化比「平移 2 个像素」还大（Zhang 2019 在下采样前加低通滤波来缓解）。换成全局平均池化（GAP）的输出头会更平移不变，但在这么小的网络上要付出准确率的代价：Node 里 GAP 头的同一个 CNN 训练 1500 步只有约 85%（它只知道「哪些特征出现过」，不知道它们之间的相对位置）。',
        },
        // ───────────────────────────── 08 U-Net ─────────────────────────────
        {
            id: 'unet', title: '编码器-解码器与 U-Net：用跳连把细节送回去', tag: 'UNET', params: ['sigma', 'spf'],
            desc: '分类只要输出一个标签，很多任务却要输出一张<strong>和输入一样大的图</strong>：分割（每个像素属于什么）、去噪、超分辨率，以及第 11 篇扩散模型里的去噪网络。<strong>编码器-解码器</strong>先像 CNN 一样逐级下采样（看得越来越广、越来越抽象），再用<strong>上采样</strong>（这里是最近邻：每个像素复制成 2×2）逐级恢复分辨率。问题是所有信息都要挤过最底下的<strong>瓶颈</strong>（这里只剩 8 × 4 × 4 = 128 个数），笔画的精确位置在下采样时就丢了。<strong>U-Net</strong>（Ronneberger et al. 2015）加了<strong>跳连</strong>（skip connection）：把编码器每一级的特征图直接<strong>按通道拼接</strong>到解码器同一尺度上，高分辨率的细节绕过瓶颈。下面在去噪任务上真实训练两个网络（激活用 SiLU）——U-Net，以及去掉两条跳连的同一个网络：输入是加了高斯噪声的数字 x̃ = x₀ + σε，目标是干净的 x₀，损失是逐像素 MSE（滚动到这一步时开始训练，batch 8，1200 步）。',
            formula: [
                'x̃ = x_0 + σ · ε,   ε ~ N(0, I)      L = mean( (f(x̃) − x_0)^2 )   // 每一步现抽新的噪声',
                's_1 = enc_1(x̃) ∈ R^{4×16×16}  →  s_2 = enc_2(↓s_1) ∈ R^{8×8×8}  →  b = mid(↓s_2) ∈ R^{8×4×4}   // ↓ = 2×2 maxpool，每个 enc = 3×3 卷积 + SiLU',
                'h_2 = dec([↑b ; s_2]) ∈ R^{4×8×8}  →  ŷ = out([↑h_2 ; s_1]) ∈ R^{1×16×16}   // ↑ = 最近邻上采样，[· ; ·] = 通道拼接（跳连）；去掉跳连 = 只用 ↑b、↑h₂',
                'PSNR = 10 · log_10(1 / MSE)   // 像素值在 [0, 1]；高 3 dB ≈ MSE 减半',
            ],
            code: {
                title: 'C++ · UNet / unetForward / trainDenoiser', src: `
                // 两级 U-Net（1 573 个参数）：编码器每级「3×3 卷积 + SiLU」后 maxpool 减半，解码器上采样回来，
                // 再和编码器同一尺度的特征按通道拼接（跳连）。skip = false：去掉两条跳连，变成普通的编码器-解码器（1 249 个参数）
                struct UNet { ConvLayer e1, e2, mid, dec, out; bool skip = true; };

                UNet makeUNet(bool skip, Rng& r) {
                    UNet m;
                    m.skip = skip;
                    m.e1 = makeConv(1, 4, 3, 16, 16, r);
                    m.e2 = makeConv(4, 8, 3, 8, 8, r);
                    m.mid = makeConv(8, 8, 3, 4, 4, r);
                    m.dec = makeConv(skip ? 16 : 8, 4, 3, 8, 8, r);
                    m.out = makeConv(skip ? 8 : 4, 1, 3, 16, 16, r);
                    return m;
                }

                Var unetForward(const UNet& m, const Var& x) {                 // x: [B × 256]（含噪图）
                    Var s1 = silu(applyConv(m.e1, x));                           // [B × 4·16·16]   ── 跳连 1（激活用 SiLU，见下面的说明）
                    Var s2 = silu(applyConv(m.e2, maxPool2(s1, 4, 16, 16)));      // [B × 8·8·8]     ── 跳连 2
                    Var b = silu(applyConv(m.mid, maxPool2(s2, 8, 8, 8)));        // [B × 8·4·4]     瓶颈：只剩 128 个数
                    Var u2 = upsample2(b, 8, 4, 4);                               // [B × 8·8·8]
                    if (m.skip) u2 = concatCols(u2, s2);                          // 通道拼接 → 16 通道（一张图 = 一行，所以按列拼）
                    Var h2 = silu(applyConv(m.dec, u2));                          // [B × 4·8·8]
                    Var u1 = upsample2(h2, 4, 8, 8);                              // [B × 4·16·16]
                    if (m.skip) u1 = concatCols(u1, s1);                          // → 8 通道
                    return applyConv(m.out, u1);                                  // [B × 256]：预测的干净图像（线性输出）
                }

                std::vector<Var> unetParams(const UNet& m) { return {m.e1.W, m.e1.b, m.e2.W, m.e2.b, m.mid.W, m.mid.b, m.dec.W, m.dec.b, m.out.W, m.out.b}; }

                float psnr(float mse) { return 10.f * std::log10(1.f / mse); }   // 像素值在 [0, 1]

                // 去噪训练：每步取 8 张训练图 x₀（按 epoch 洗牌），现加噪声 x̃ = x₀ + σ·ε，最小化 MSE(f(x̃), x₀)
                void trainDenoiser(UNet& m, const Dataset& tr, float sigma, int steps, Rng& r) {
                    std::vector<Var> ps = unetParams(m);
                    std::vector<bool> noDecay = std::vector<bool>(ps.size(), false);
                    AdamW opt = makeAdamW(ps, noDecay, 3e-3f, 0.f, 0.9f, 0.999f);
                    std::vector<int> order = permutation(tr.n, r);
                    int pos = 0;
                    for (int t = 0; t < steps; t++) {
                        std::vector<int> idx = ints(8);
                        for (int b = 0; b < 8; b++) {
                            if (pos == tr.n) { order = permutation(tr.n, r); pos = 0; }
                            idx[b] = order[pos++];
                        }
                        Tensor x0 = rowsOf(tr.X, idx), xn = x0;
                        for (float& v : xn.v) v += sigma * randn(r);
                        zeroGrad(ps);
                        Var loss = mse(unetForward(m, constant(xn)), constant(x0));
                        backward(loss);
                        adamwStep(opt, cosineLR(t, 3e-3f, 1.5e-4f, 0, steps));
                    }
                }`,
            },
            previews: [
                {
                    id: 'unetGrid', label: '去噪结果（测试集，实时训练）', sub: '每列一张测试图 · 行：含噪输入 / U-Net / 去掉跳连 / 干净目标 · 灰度范围 [0, 1]', live: true, large: true, size: [360, 250], needs: ['unet'],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), U = S.U, sh = U.show, n = P08.UN_SHOW, cell = 34, gap = 3, ox = 64, oy = 24;
                        const last = (k) => P08.last(U.hist.psnr[k]);
                        P08.txt(ctx, `σ = ${U.sigma} · 第 ${sh.step} 步的评估 · ${P08.status(U.runs[0])}`, 40, 4, { bold: true, color: LM.C.yellow });
                        const rows = [['含噪', U.ns.Xn, `${U.base.noisy.toFixed(1)} dB`, LM.C.gray], ['U-Net', sh.y[0], `${last(0).toFixed(1)} dB`, LM.C.green], ['无跳连', sh.y[1], `${last(1).toFixed(1)} dB`, LM.C.red], ['干净', U.ns.X0, '目标', LM.C.gray]];
                        rows.forEach(([nm, data, sub, col], r) => {
                            const y = oy + r * (cell + 18);
                            P08.txt(ctx, nm, 6, y + 4, { color: col, bold: true }); P08.txt(ctx, sub, 6, y + 19, { color: LM.C.dim });
                            for (let k = 0; k < n; k++) P08.drawImg(ctx, data, ox + k * (cell + gap), y, cell, cell, { offset: k * 256, range: [0, 1] });
                        });
                    },
                },
                {
                    id: 'unetPsnr', label: '测试 PSNR（48 张固定的含噪测试图）', sub: '越高越好 · 灰线 = 含噪输入本身、红虚线 = 最好的固定高斯模糊（3×3 / 5×5 二项式核）', live: true, size: [360, 210], needs: ['unet'],
                    render: (c, S) => {
                        const U = S.U, H = U.hist, top = Math.max(16, ...H.psnr[0], U.base.blur) + 1;
                        P08.curves(c, [
                            { x: H.step, y: H.psnr[0], color: LM.C.green, width: 2.5, label: `U-Net ${P08.last(H.psnr[0]).toFixed(2)} dB` },
                            { x: H.step, y: H.psnr[1], color: LM.C.red, width: 2, label: `去掉跳连 ${P08.last(H.psnr[1]).toFixed(2)} dB` },
                        ], { x: [0, P08.UCAP], y: [Math.floor(U.base.noisy) - 1, top], xLabel: `训练步数（${P08.status(U.runs[0])}）`, hmarks: [{ y: U.base.blur, color: LM.C.red, label: `模糊 ${U.base.blur.toFixed(1)}` }, { y: U.base.noisy, color: LM.C.gray, label: `含噪 ${U.base.noisy.toFixed(1)}` }], legendAt: [1, 0.55] });
                    },
                },
                {
                    id: 'unetArch', label: 'U-Net 的结构与第一张图在每一级的激活', sub: '缩略图 = 该级所有通道的平均（实时）· 灰箭头 = ↓ maxpool / ↑ 上采样 · 绿箭头 = 跳连（按通道拼接）', live: true, size: [360, 262], needs: ['unetTrace'],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), t = S.ut, th = 40;
                        const box = (x, y, img, Wd, name, shape, col, rng) => {
                            ctx.fillStyle = '#12161c'; ctx.fillRect(x - 4, y - 4, th + 8, th + 8); ctx.strokeStyle = col || '#3d5a73'; ctx.lineWidth = 1.2; ctx.strokeRect(x - 4, y - 4, th + 8, th + 8);
                            const lo = Math.min(...img), m = rng || [lo, Math.max(lo + 1e-6, ...img)]; P08.drawImg(ctx, img, x, y, th, th, { W: Wd, H: Wd, map: rng ? 'gray' : 'inferno', range: m });
                            ctx.font = LM.FONT(11); const w1 = ctx.measureText(name + ' ').width, w2 = ctx.measureText(shape).width, x0 = x + th / 2 - (w1 + w2) / 2;
                            P08.txt(ctx, name, x0, y + th + 7, { color: LM.C.text }); P08.txt(ctx, shape, x0 + w1, y + th + 7, { color: LM.C.cyan });
                        };
                        const arrow = (x0, y0, x1, y1, col) => {
                            ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
                            const a = Math.atan2(y1 - y0, x1 - x0); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - 6 * Math.cos(a - 0.45), y1 - 6 * Math.sin(a - 0.45)); ctx.lineTo(x1 - 6 * Math.cos(a + 0.45), y1 - 6 * Math.sin(a + 0.45)); ctx.fill();
                        };
                        box(20, 24, t.x, 16, 'x̃', '1×16×16', '#4a5058', [0, 1]);
                        box(96, 24, t.s1, 16, 's₁', '4×16×16');
                        box(132, 112, t.s2, 8, 's₂', '8×8×8');
                        box(184, 196, t.bt, 4, '瓶颈 b', '8×4×4', LM.C.orange);
                        box(236, 112, t.h2, 8, 'h₂', '4×8×8');
                        box(300, 24, t.y, 16, 'ŷ', '1×16×16', LM.C.green, [0, 1]);
                        const g = '#6a7078';
                        arrow(68, 44, 88, 44, g); arrow(130, 88, 146, 104, g); arrow(166, 174, 190, 190, g); arrow(222, 190, 246, 174, g); arrow(276, 106, 292, 66, g);
                        arrow(142, 34, 292, 34, LM.C.green); arrow(180, 128, 228, 128, LM.C.green);
                        P08.txt(ctx, '跳连 1：拼接 s₁', 170, 16, { color: LM.C.green });
                        P08.txt(ctx, '跳连 2', 186, 134, { color: LM.C.green });
                        P08.txt(ctx, `第 ${S.U.runs[0].step} 步的 U-Net`, 252, 246, { color: LM.C.gray });
                    },
                },
            ],
            notes: 'Node 里用页面同样的设置训练到 1200 步（tools/p08_final.js）：σ = 0.5 时 U-Net 14.88 dB、去掉跳连 13.12 dB、最好的固定模糊 13.87 dB（含噪输入本身 5.99 dB）；σ = 0.3 时三者是 17.25 / 14.13 / 15.98 dB，σ = 0.8 时 12.99 / 11.97 / 11.72 dB。去掉跳连的网络在 σ = 0.3 / 0.5 时比一个固定的模糊滤波还差——瓶颈里的 128 个数记不住笔画的精确位置，输出是一团模糊的「平均数字」；U-Net 的输出笔画清楚、噪点被抹掉。两条路都在出力：把训练好的 U-Net 的 h₂ 整个置零（只剩跳连 1 那条浅路），σ = 0.5 时 PSNR 从 14.88 掉到 13.45 dB（tools/p08_unet_act.js）。改<b>噪声 σ</b> 会让两个网络从头训练。',
            warn: '去噪网络是第 11 篇扩散模型的核心：DDPM 的去噪器就是一个（大得多的）U-Net，只是噪声强度每一步都不同，所以还要把「现在是第几步」作为输入（时间步 embedding，加到每个通道上，即库里的 <code>addChannel</code>）。本页的 U-Net 只在一个固定的 σ 上训练，规模也极小（1 573 个参数，每级 4 / 8 个通道）。激活函数用 SiLU（扩散模型的 U-Net 也常用它）而不是 ReLU，是实测出来的：用 ReLU 时，只有 4 个通道的解码器那一级在训练中几乎全部「死掉」（激活比例 < 1%），把它置零 PSNR 纹丝不动——U-Net 退化成只走跳连 1 的两层浅网络；更早的一个版本在输出前还有一层 4 通道的 3×3 卷积 + ReLU，某些初始化下它整层死掉，输出变成常数（PSNR 停在 9.7 dB，tools/p08_proto.js useed）。SiLU 在负半轴也有梯度，不会整个通道死掉。',
        },
        // ───────────────────────────── 09 残差与归一化 ─────────────────────────────
        {
            id: 'resnet', title: '残差连接与归一化：让很深的网络也能训练', tag: 'RESNET', params: ['depth', 'norm', 'spf'],
            desc: '网络越深，表达能力越强，但把卷积层直接往深里堆，训练反而更差：He et al. 2016 发现 56 层的普通网络连在训练集上的误差都比 20 层的高（<strong>退化问题</strong>，degradation）——不是过拟合，而是优化困难。这里用一个很窄的网络（每层 4 个通道）在浏览器里真实复现：深度 L 的<strong>普通网络</strong> x ← ReLU(F(x)) 与<strong>残差网络</strong> x ← ReLU(x + F(x))，同一串 batch、同时训练（滚动到这一步时开始，batch 8，800 步）。残差块的 x + F(x) 让恒等映射成为默认选项：只要 F 输出 0，多出来的块就什么都不做，所以加深不会变差；反向传播时 ∂x_{l+1}/∂x_l 里有一个恒等项，梯度可以原样穿过任意多层。第三、四张图是<strong>还没训练时</strong>的信号传播（一个 batch 的一次前向 + 反向）：普通网络的激活一层层缩小；残差块如果像普通层一样做 He 初始化，每个块都让方差约翻倍，16 层后放大上百倍——所以实际的残差网络要把分支初始化为 0（有 BN 时把 BN 的 γ 置 0，Goyal et al. 2017；没有 BN 时 Fixup / SkipInit 也是这个思路），让每个块一开始都是恒等映射。下面训练的残差网络都这样初始化。',
            formula: [
                '普通：x_{l+1} = ReLU(F_l(x_l))      残差：x_{l+1} = ReLU(x_l + F_l(x_l))      F_l(x) = [BN](conv_l(x))',
                '∂x_{l+1}/∂x_l = D_l · (I + ∂F_l/∂x_l)   // D_l = ReLU 的导数（对角的 0 / 1）；恒等项 I 让梯度不必穿过每一层的权重',
                'BN：x̂ = (x − μ_B) / √(σ_B^2 + ε)，y = γ · x̂ + β   // μ_B、σ_B² 对每个通道在 batch × H × W 上求；推理时换成滑动平均',
                '零初始化：分支卷积的权重（有 BN 时是 BN 的 γ）= 0  ⇒  初始时 x_{l+1} = ReLU(x_l) = x_l   // x_l 已经 ≥ 0',
            ],
            code: {
                title: 'C++ · batchNorm2d（自定义算子）/ DeepNet / deepForward', src: `
                // BatchNorm2d：每个通道在 batch × H × W 上求均值 μ、方差 σ²，x̂ = (x − μ)/√(σ² + ε)，y = γ·x̂ + β；
                // 训练时更新滑动平均（动量 0.1），推理（train = false）时用它代替 batch 统计量
                struct BatchNorm { Var gamma, beta; std::vector<float> runMean, runVar; int C = 0; };

                BatchNorm makeBatchNorm(int C, float gamma0) {
                    BatchNorm bn;
                    bn.C = C; bn.gamma = param(full(1, C, gamma0)); bn.beta = param(zeros(1, C));
                    bn.runMean = floats(C); bn.runVar = floats(C, 1.f);
                    return bn;
                }

                Var batchNorm2d(BatchNorm& bn, const Var& X, bool train) {
                    int N = X->val.rows, C = bn.C, HW = X->val.cols / C;
                    float M = (float)N * HW;
                    Tensor y = zeros(N, X->val.cols), xh = zeros(N, X->val.cols);
                    std::vector<float> rstd = floats(C);
                    for (int c = 0; c < C; c++) {
                        double mu = 0, var = 0;
                        if (train) {
                            for (int n = 0; n < N; n++) for (int k = 0; k < HW; k++) mu += X->val.v[((size_t)n * C + c) * HW + k];
                            mu /= M;
                            for (int n = 0; n < N; n++) for (int k = 0; k < HW; k++) { double d = X->val.v[((size_t)n * C + c) * HW + k] - mu; var += d * d; }
                            var /= M;
                            bn.runMean[c] = 0.9f * bn.runMean[c] + 0.1f * (float)mu;
                            bn.runVar[c] = 0.9f * bn.runVar[c] + 0.1f * (float)(var * M / std::max(M - 1, 1.f));   // 滑动方差用无偏估计（同 PyTorch）
                        } else { mu = bn.runMean[c]; var = bn.runVar[c]; }
                        rstd[c] = 1.f / std::sqrt((float)var + 1e-5f);
                        for (int n = 0; n < N; n++) for (int k = 0; k < HW; k++) {
                            size_t i = ((size_t)n * C + c) * HW + k;
                            xh.v[i] = (float)(X->val.v[i] - mu) * rstd[c];
                            y.v[i] = bn.gamma->val.v[c] * xh.v[i] + bn.beta->val.v[c];
                        }
                    }
                    Var g = bn.gamma, b = bn.beta;
                    return record(std::move(y), {X, g, b}, [X, g, b, xh, rstd, N, C, HW, M, train](Node& o) {
                        for (int c = 0; c < C; c++) {
                            double sg = 0, sgh = 0;                                  // Σ dy、Σ dy·x̂
                            for (int n = 0; n < N; n++) for (int k = 0; k < HW; k++) { size_t i = ((size_t)n * C + c) * HW + k; sg += o.grad.v[i]; sgh += o.grad.v[i] * xh.v[i]; }
                            if (g->requiresGrad) gradOf(g).v[c] += (float)sgh;
                            if (b->requiresGrad) gradOf(b).v[c] += (float)sg;
                            if (!X->requiresGrad) continue;
                            float a = g->val.v[c] * rstd[c];
                            for (int n = 0; n < N; n++) for (int k = 0; k < HW; k++) {
                                size_t i = ((size_t)n * C + c) * HW + k;
                                // 训练模式下 μ、σ² 也依赖 x：dx = γ·rstd/M · (M·dy − Σdy − x̂·Σ(dy·x̂))
                                gradOf(X).v[i] += train ? (float)(a / M * (M * o.grad.v[i] - sg - xh.v[i] * sgh)) : a * o.grad.v[i];
                            }
                        }
                    });
                }

                // 深层 CNN：stem（3×3 卷积 1→4 + ReLU + maxpool → 4×8×8）+ L 个块（每块一个 3×3 卷积 F(x) = [BN](conv(x))）+ maxpool → linear
                //   普通：x ← ReLU(F(x))        残差：x ← ReLU(x + F(x))
                struct DeepNet { ConvLayer stem; std::vector<ConvLayer> convs; std::vector<BatchNorm> bns; Linear fc; bool residual = false, useBN = false; };

                DeepNet makeDeepNet(int L, bool residual, bool useBN, bool zeroInit, Rng& r) {
                    DeepNet m;
                    m.residual = residual; m.useBN = useBN;
                    m.stem = makeConv(1, 4, 3, 16, 16, r);
                    for (int l = 0; l < L; l++) {
                        ConvLayer c = makeConv(4, 4, 3, 8, 8, r);
                        bool zero = residual && zeroInit;                            // 零初始化的残差分支：每个块一开始是恒等映射
                        if (zero && !useBN) c.W->val = zeros(4, 36);
                        m.convs.push_back(c);
                        if (useBN) m.bns.push_back(makeBatchNorm(4, zero ? 0.f : 1.f));
                    }
                    m.fc = makeLinear(4 * 4 * 4, 10, std::sqrt(2.f / 64), r);
                    return m;
                }

                Var deepForward(DeepNet& m, const Var& x, bool train) {
                    Var h = maxPool2(relu(applyConv(m.stem, x)), 4, 16, 16);            // [B × 4·8·8]
                    for (size_t l = 0; l < m.convs.size(); l++) {
                        Var f = applyConv(m.convs[l], h);
                        if (m.useBN) f = batchNorm2d(m.bns[l], f, train);
                        h = relu(m.residual ? add(h, f) : f);                           // 残差：梯度经过 add 时原样分给 h
                    }
                    return forward(m.fc, maxPool2(h, 4, 8, 8));
                }`,
            },
            previews: [
                {
                    id: 'deepLoss', label: '训练 loss：普通网络 vs 残差网络（实时）', sub: '同一串 batch（8 张）、同一个优化器 · 曲线 = batch loss 的滑动平均 · 灰线 = ln 10（随机猜）', live: true, large: true, size: [360, 250], needs: ['deep'],
                    render: (c, S) => {
                        const G = S.G, [P, Rr] = G.runs, xs = (R) => R.lossEMA.map((_, i) => i + 1);
                        const acc = (R) => (R.step >= R.cap && R.hist.testA.length ? ` · 测试 ${P08.pct(P08.last(R.hist.testA))}` : ` · batch 准确率 ${P08.pct(R.aema)}`);
                        P08.curves(c, [
                            { x: xs(P), y: P.lossEMA, color: LM.C.red, width: 2, label: `普通：loss ${LM.fmt(P.ema)}${acc(P)}` },
                            { x: xs(Rr), y: Rr.lossEMA, color: LM.C.green, width: 2.5, label: `残差：loss ${LM.fmt(Rr.ema)}${acc(Rr)}` },
                        ], { x: [0, P08.DCAP], y: [0.05, 4], logY: true, xLabel: `训练步数 · L = ${G.L} · ${G.bn ? 'BatchNorm' : '无归一化'} · ${P08.status(P)}`, hmarks: [{ y: Math.log(10), color: LM.C.gray }], legendAt: [0, 0.72], legendAlign: 'left' });
                    },
                },
                {
                    id: 'deepActs', label: '训练中每一层的「死」激活比例', sub: '当前 batch 上 ReLU 输出恰好为 0 的比例 · 第 0 层 = stem · 比例到 1 = 这一层整层死掉，后面什么都传不过去', live: true, size: [360, 190], needs: ['deep'],
                    render: (c, S) => {
                        const G = S.G, [P, Rr] = G.runs;
                        if (!P.actStats) { const { ctx } = PL.draw.ctx(c); P08.txt(ctx, '等待第一步训练…', 40, 8, { color: LM.C.gray }); return; }
                        const L = P.actStats.length, xs = Array.from({ length: L }, (_, i) => i);
                        P08.curves(c, [
                            { x: xs, y: P.actStats.map((a) => a.zero), color: LM.C.red, width: 2, label: '普通' },
                            { x: xs, y: Rr.actStats.map((a) => a.zero), color: LM.C.green, width: 2, label: '残差' },
                        ], { x: [0, L - 1], y: [0, 1.05], xLabel: `层（第 ${P.step} 步）`, legendAt: [0, 0], legendAlign: 'left' });
                    },
                },
                {
                    id: 'flowAct', label: '初始化时：每层激活的 RMS', sub: '一个 batch（32 张）的一次前向，还没训练 · 对数纵轴', size: [360, 190], needs: ['deepFlow'],
                    render: (c, S) => {
                        const F = S.flow, L = F[0].act.length - 1, xs = Array.from({ length: L + 1 }, (_, i) => i);
                        P08.curves(c, [
                            { x: xs, y: F[0].act, color: LM.C.red, width: 2, label: '普通' },
                            { x: xs, y: F[1].act, color: LM.C.orange, width: 2, dash: [5, 4], label: '残差，分支 He 初始化' },
                            { x: xs, y: F[2].act, color: LM.C.green, width: 2.5, label: '残差，分支零初始化' },
                        ], { x: [0, L], y: [1e-3, 1e3], logY: true, xLabel: `层 l（L = ${L}，${S.norm === 'bn' ? 'BatchNorm' : '无归一化'}）`, legendAt: [0, 0.64], legendAlign: 'left' });
                    },
                },
                {
                    id: 'flowGrad', label: '初始化时：∂L/∂x_l 的 RMS（反向传播回来的梯度）', sub: '同一个 batch 的一次反向 · 越往左越靠近输入', size: [360, 190], needs: ['deepFlow'],
                    render: (c, S) => {
                        const F = S.flow, L = F[0].grad.length - 1, xs = Array.from({ length: L + 1 }, (_, i) => i);
                        P08.curves(c, [
                            { x: xs, y: F[0].grad.map((v) => Math.max(v, 1e-7)), color: LM.C.red, width: 2, label: `普通（loss₀ ${F[0].loss.toFixed(2)}）` },
                            { x: xs, y: F[1].grad.map((v) => Math.max(v, 1e-7)), color: LM.C.orange, width: 2, dash: [5, 4], label: `残差 He（loss₀ ${F[1].loss.toFixed(1)}）` },
                            { x: xs, y: F[2].grad.map((v) => Math.max(v, 1e-7)), color: LM.C.green, width: 2.5, label: `残差 零初始化（loss₀ ${F[2].loss.toFixed(2)}）` },
                        ], { x: [0, L], y: [1e-5, 10], logY: true, xLabel: '层 l（0 = stem 的输出）', legendAt: [0, 0.64], legendAlign: 'left' });
                    },
                },
            ],
            notes: 'Node 里用页面同样的设置训练 800 步（batch 8，tools/p08_final.js）的测试准确率。无归一化：L = 4 时普通 79.8% / 残差 88.4%；L = 16 时 76.8% / 90.0%；L = 24 时普通网络完全训不动——loss 停在 ln 10 = 2.30、准确率 10%：训练早期最后一层的 ReLU 就全部变成 0（第二张图的红线顶到 1），残差网络 91.4%。把<b>归一化</b>切到 BatchNorm：普通网络每层的激活不再缩小（第三张图变平），但梯度朝输入方向放大十几倍（第四张图），L = 16 时只剩 37.8%、L = 24 时 28.4%，比不加 BN 还差；残差 + BN（γ 零初始化）90.8% / 95.6%。',
            warn: 'BatchNorm 训练时用 batch 统计量、推理时用滑动平均；batch 很小时（这里 8 张 × 64 个位置）统计量有噪声，还让同一 batch 的样本互相影响。Transformer 用 LayerNorm（每个样本自己归一化，第 04 篇），扩散模型的 U-Net 常用 GroupNorm。这里的网络只有 4 个通道，是为了让 16–24 层的训练能在浏览器里实时跑；数字说明的是趋势，不能和 ImageNet 上的结果直接比较。BN 让深层普通网络的梯度朝输入方向爆炸，是 Yang et al. 2019 用平均场理论分析过的现象。',
        },
        // ───────────────────────────── 10 教学版 → 生产版 ─────────────────────────────
        {
            id: 'prod', title: '教学版 → 生产版：从 LeNet 到 ResNet、U-Net 与 ViT', tag: 'NEXT',
            desc: '本页的每样东西在真实系统里都有对应，只是规模大了几个数量级：',
            blocks: [
                {
                    type: 'html', html: `<table class="lm-tbl"><tr><th>模型 / 年份</th><th>做了什么</th><th>对应本页</th></tr>
                    <tr><td>Neocognitron（Fukushima 1980）</td><td>局部感受野、逐层组合、下采样的层级结构（受 Hubel & Wiesel 对猫视觉皮层的研究启发），还没有反向传播</td><td>Step 03 / 04</td></tr>
                    <tr><td>LeNet（LeCun et al. 1989 / 1998）</td><td>用反向传播训练卷积网络识别手写数字；LeNet-5 在 MNIST 上 0.95% 错误率</td><td>Step 05 的 CNN 就是 LeNet 式</td></tr>
                    <tr><td>AlexNet（Krizhevsky et al. 2012）</td><td>5 个卷积层 + 3 个全连接层、约 6 000 万参数；ReLU、dropout、数据增强，两块 GPU 训练；ImageNet 2012 top-5 错误率 15.3%（第二名 26.2%）</td><td>Step 05 / 07</td></tr>
                    <tr><td>VGG（Simonyan & Zisserman 2014）</td><td>只用 3×3 卷积往深里堆（16 / 19 层）：两层 3×3 的感受野等于一层 5×5，参数却少（2·9C² 对 25C²）</td><td>Step 04</td></tr>
                    <tr><td>U-Net（Ronneberger et al. 2015）</td><td>编码器-解码器 + 跳连，做生物医学图像分割；FCN（Long et al. 2015）把分类网络改成逐像素输出</td><td>Step 08</td></tr>
                    <tr><td>BatchNorm（Ioffe & Szegedy 2015）· ResNet（He et al. 2016）</td><td>残差连接让 152 层的网络可以训练，赢得 ILSVRC 2015 分类；后来的 pre-activation 版本训练到了 1000 层以上</td><td>Step 09</td></tr>
                    <tr><td>YOLO（Redmon et al. 2016）</td><td>检测：一个 CNN 在 S×S 网格（v1 是 7×7）的每个格子里直接回归边框和类别概率</td><td>「每个位置一个分类器」= 卷积</td></tr>
                    <tr><td>ViT（Dosovitskiy et al. 2021）</td><td>把图切成 16×16 的 patch 当 token 交给 Transformer；在 JFT-300M 这类大数据上预训练后追平 / 超过 CNN，数据少时 CNN 的先验更划算</td><td>第 07 篇</td></tr>
                    <tr><td>ConvNeXt（Liu et al. 2022）</td><td>用 Transformer 时代的训练技巧与设计（大核、LayerNorm、更少的激活）现代化 ResNet，ImageNet 上与 Swin Transformer 相当</td><td>CNN 并没有过时</td></tr>
                    <tr><td>DDPM（Ho et al. 2020）· Stable Diffusion（Rombach et al. 2022）</td><td>去噪 U-Net（加时间步 embedding、自注意力、GroupNorm）是扩散模型的主干；DiT（Peebles & Xie 2023）把它换成 Transformer</td><td>Step 08 → 第 11 篇</td></tr>
                    <tr><td>SAM（Kirillov et al. 2023）</td><td>「分割一切」：ViT 图像编码器 + 提示编码器 + 轻量掩码解码器；SA-1B 数据集 1 100 万张图、11 亿个掩码</td><td>分割</td></tr>
                    <tr><td>卷积的实现</td><td>本页 JS：补零后的扁平内存布局 + 3×3 展开；C++：直白的循环。生产：im2col + GEMM（下面的代码）、Winograd（Lavin & Gray 2016：F(2×2, 3×3) 把每 2×2 个输出的乘法从 36 次降到 16 次）、大核用 FFT，GPU 上由 cuDNN 等库按形状自动选算法</td><td>Step 03 的乘加数</td></tr></table>`,
                },
                {
                    type: 'code', title: 'C++ · im2col + GEMM：把卷积变成一次矩阵乘', src: `
                    // 实际框架里的卷积：im2col 把每个输出位置要读的 C·K·K 个输入排成一列，卷积就变成一次大矩阵乘（GEMM，GPU 最擅长的运算）
                    Tensor im2col(const std::vector<float>& x, int C, int H, int W, int K, int s, int p) {
                        int Ho = outSize(H, K, s, p), Wo = outSize(W, K, s, p);
                        Tensor cols = zeros(C * K * K, Ho * Wo);
                        for (int c = 0; c < C; c++)
                            for (int u = 0; u < K; u++)
                                for (int v = 0; v < K; v++)
                                    for (int i = 0; i < Ho; i++)
                                        for (int j = 0; j < Wo; j++) {
                                            int yy = i * s + u - p, xx = j * s + v - p;
                                            if (yy >= 0 && yy < H && xx >= 0 && xx < W) cols.at((c * K + u) * K + v, i * Wo + j) = x[(c * H + yy) * W + xx];
                                        }
                        return cols;                                                 // [C·K·K × Ho·Wo]：内存变成 K² 倍，换来连续访问
                    }

                    // [C_out × C·K·K] · [C·K·K × Ho·Wo] = [C_out × Ho·Wo]：与 conv2d 的结果完全相同（再加偏置）
                    Tensor convAsGemm(const Tensor& Wt, const std::vector<float>& x, int C, int H, int W, int K, int s, int p) {
                        NoGrad ng;
                        return matmul(constant(Wt), constant(im2col(x, C, H, W, K, s, p)))->val;
                    }`,
                },
                {
                    type: 'previews', items: [{
                        id: 'final', label: '本页实时训练的结果', sub: '读取各步正在训练的网络（没训练过的步骤显示「未开始」）', live: true, size: [360, 150],
                        render: (c, S) => {
                            const C = S.main.cnn, M = S.main.mlp, f = P08.fmtN, rows = [
                                [{ t: 'CNN  ', color: LM.C.green, bold: true }, { t: `${f(P08.numParams(C.m.params))} 个参数 · 测试 ${P08.pct(P08.last(C.hist.testA))}（第 ${C.evalStep} 步）` }],
                                [{ t: 'MLP  ', color: LM.C.purple, bold: true }, { t: `${f(P08.numParams(M.m.params))} 个参数 · 测试 ${P08.pct(P08.last(M.hist.testA))}（第 ${M.evalStep} 步）` }],
                            ];
                            rows.push(S.U ? [{ t: 'U-Net  ', color: LM.C.green, bold: true }, { t: `去噪 ${P08.last(S.U.hist.psnr[0]).toFixed(2)} dB · 去掉跳连 ${P08.last(S.U.hist.psnr[1]).toFixed(2)} dB · 模糊 ${S.U.base.blur.toFixed(2)} dB（σ = ${S.U.sigma}，第 ${S.U.runs[0].step} 步）` }] : { text: 'U-Net：未开始（滚动到 Step 08）', color: LM.C.dim });
                            rows.push(S.G ? [{ t: `深度 ${S.G.L}  `, color: LM.C.orange, bold: true }, { t: `普通 loss ${LM.fmt(S.G.runs[0].ema)} · 残差 loss ${LM.fmt(S.G.runs[1].ema)}（第 ${S.G.runs[0].step} 步）` }] : { text: '深层网络：未开始（滚动到 Step 09）', color: LM.C.dim });
                            rows.push({ text: '真实 MNIST 上的参考：LeNet-5 0.95% 错误率（LeCun et al. 1998）；第 02 篇引用的 800 单元 MLP 1.6%。', color: LM.C.dim });
                            P08.lines(c, rows, { px: 11, x0: 40 });
                        },
                    }],
                },
                { type: 'note', html: 'Node 里用页面同一份代码、同样的设置把四个分类模型训练到终点的结果（tools/p08_final.js）写在 Step 05 / 07 的说明里；数字只说明趋势——16×16 的程序化数字比真实 MNIST 简单，网络也极小。' },
                { type: 'note', html: '下一篇（09）用同样的卷积编码器 / 解码器做潜空间：自编码器、VAE 与 VQ-VAE；第 11 篇的扩散模型直接用 Step 08 的 U-Net 做去噪器，再加上时间步 embedding。' },
            ],
        },
    ],

    graph: {
        title: '训练一步 · CNN 的数据流（batch = 32）',
        intro: '每个节点 = 一次真实的计算（在 GPU 上是一次或几次 kernel launch；这里是 JS / C++ 循环），pin = 张量名 · 形状（一张图 = 一行，按通道优先展平，与 PyTorch 的 NCHW 相同）。<code>Params</code> 是 <b>HISTORY</b> 资源：每一步读取上一步更新后的权重（虚线回边）；反向传播由自动微分沿前向的 tape 逆序完成，读取前向时存下的特征图，这里合成一个节点。',
        types: {
            img: { color: '#6ab0e0', label: 'img · 图像 / 输入 f32' },
            label: { color: '#e8c25a', label: 'label · 类别标签 int' },
            act: { color: '#c678dd', label: 'act · 特征图 / logits' },
            param: { color: '#5ad07a', label: 'param · 卷积核 / 权重' },
            grad: { color: '#e06c75', label: 'grad · 梯度' },
            scalar: { color: '#8a8f98', label: 'scalar · 标量 / 超参数 / 指标' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/minml_*.h，可直接编译）+ 本页公用函数', open: false,
            blocks: LM.CPP.blocks(['autograd', 'nn']).concat([{
                label: '━ 本页公用：数据、平移、卷积层 ━', code: `
                // n 张 16×16 灰度数字：X 的每一行是一张图的 256 个像素（0..1，按行展开），y = 标签 0…9
                struct Dataset { Tensor X; std::vector<int> y; int n = 0; };

                // 第 02 篇的程序化数字生成器（笔画字形 + 随机仿射 / 粗细 / 手抖 / 噪点；与页面里的 LM.digits 逐像素相同，完整实现见第 02 篇）
                Dataset makeDigits(int n, Rng& r);

                // 第 i 行（一张图）拷成一个数组
                std::vector<float> rowOf(const Tensor& X, int i) {
                    std::vector<float> a = floats(X.cols);
                    for (int j = 0; j < X.cols; j++) a[j] = X.v[(size_t)i * X.cols + j];
                    return a;
                }

                // 取出若干行组成一个 batch：第 b 行 = 第 idx[b] 张图
                Tensor rowsOf(const Tensor& X, const std::vector<int>& idx) {
                    Tensor B = zeros((int)idx.size(), X.cols);
                    for (int b = 0; b < (int)idx.size(); b++)
                        for (int j = 0; j < X.cols; j++) B.v[(size_t)b * X.cols + j] = X.v[(size_t)idx[b] * X.cols + j];
                    return B;
                }

                std::vector<int> labelsOf(const std::vector<int>& y, const std::vector<int>& idx) {
                    std::vector<int> t = ints((int)idx.size());
                    for (int b = 0; b < (int)idx.size(); b++) t[b] = y[idx[b]];
                    return t;
                }

                // 整数像素平移（空出来的地方补 0）：dx > 0 向右、dy > 0 向下。img = C 个 H×W 平面（通道优先）
                std::vector<float> shiftImage(const std::vector<float>& img, int dx, int dy, int C, int H, int W) {
                    std::vector<float> out = floats(C * H * W);
                    for (int c = 0; c < C; c++)
                        for (int y = 0; y < H; y++)
                            for (int x = 0; x < W; x++) {
                                int sx = x - dx, sy = y - dy;
                                if (sx >= 0 && sx < W && sy >= 0 && sy < H) out[(c * H + y) * W + x] = img[(c * H + sy) * W + sx];
                            }
                    return out;
                }

                // 数据增强：batch 里每张图独立地随机平移 dx, dy ∈ {−a, …, a}（Step 07）
                Tensor augmentBatch(const Tensor& Xb, int a, Rng& r) {
                    Tensor out = zeros(Xb.rows, Xb.cols);
                    for (int b = 0; b < Xb.rows; b++) {
                        int dx = randint(r, 2 * a + 1) - a, dy = randint(r, 2 * a + 1) - a;
                        std::vector<float> s = shiftImage(rowOf(Xb, b), dx, dy, 1, 16, 16);
                        for (int j = 0; j < Xb.cols; j++) out.v[(size_t)b * Xb.cols + j] = s[j];
                    }
                    return out;
                }

                // 卷积层 = 参数 + 形状。W: [C_out × C_in·K·K]（PyTorch 的 [C_out, C_in, K, K] 按行展平），b: [1 × C_out]
                struct ConvLayer { Var W, b; ConvSpec s; };

                // He 初始化（std = √(2 / (C_in·K·K))）、偏置 0；stride 1、pad = K/2：输出和输入一样大（「same」卷积）
                ConvLayer makeConv(int cin, int cout, int K, int H, int W, Rng& r) {
                    ConvLayer L;
                    L.W = param(heInit(cout, cin * K * K, cin * K * K, r));
                    L.b = param(zeros(1, cout));
                    L.s = conv(cin, H, W, K, 1, K / 2);
                    return L;
                }

                Var applyConv(const ConvLayer& L, const Var& x) { return conv2d(x, L.W, L.b, L.s); }`,
            }]),
        },
        nodes: [
            {
                id: 'gen', title: 'makeDigits', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'conv', desc: '第 02 篇同一个生成器 · 4000 + 500 张',
                dispatch: { kind: 'CPU', name: 'makeDigits', count: '4500 张 × 256 像素' },
                outputs: [{ name: 'X_train', format: '[4000, 256] f32', type: 'img' }, { name: 'y_train', format: '[4000] int', type: 'label' }, { name: 'X_test', format: '[500, 256] f32', type: 'img' }, { name: 'y_test', format: '[500] int', type: 'label' }],
                detail: { preview: 'pickgrid' },
            },
            {
                id: 'hyper', title: 'HyperParams', tag: 'CB', category: 'user', desc: 'Adam η = 3e-3 cos · B = 32 · 1500 步',
                outputs: [{ name: 'Hyper', format: 'scalars', type: 'scalar' }],
                detail: { desc: '训练超参数：Adam（β₁ = 0.9、β₂ = 0.999、不做权重衰减）、学习率 3·10⁻³ 余弦衰减到 5%、batch 32、上限 1500 步；CNN 与 MLP 用同一组。', preview: null },
            },
            {
                id: 'init', title: 'makeCNN', tag: 'CPU', category: 'bake', lifetime: 'param', step: 'train', desc: 'He 初始化，偏置 0',
                dispatch: { kind: 'CPU', name: 'makeCNN', count: '3 818 个参数' },
                inputs: [{ name: 'Hyper', format: 'seed', type: 'scalar' }],
                outputs: [{ name: 'Params₀', format: 'conv1 [8, 9] · conv2 [16, 72] · fc [256, 10]', type: 'param' }],
                detail: { preview: 'kernels' },
            },
            {
                id: 'batch', title: 'next_batch', tag: 'CPU', lifetime: 'history', step: 'train', desc: '按 epoch 洗牌取 32 张',
                dispatch: { kind: 'CPU', name: 'rowsOf', count: '32 行 × 256' },
                inputs: [{ name: 'X_train', format: '[4000, 256]', type: 'img' }, { name: 'y_train', format: '[4000]', type: 'label' }],
                outputs: [{ name: 'Xb', format: '[32, 1·16·16] f32', type: 'img' }, { name: 'yb', format: '[32] int', type: 'label' }],
                detail: { preview: 'acc' },
            },
            {
                id: 'conv1', title: 'conv2d · conv1', tag: 'CPU', lifetime: 'history', step: 'conv', desc: '1 → 8 通道，3×3，p = 1',
                dispatch: { kind: 'CPU', name: 'conv2d', count: '32 × 8 × 256 × 9 = 589 824 次乘加' },
                inputs: [{ name: 'Xb', format: '[32, 256]', type: 'img' }, { name: 'Params', format: 'conv1.W [8, 9] · b [8]', type: 'param', history: true }],
                outputs: [{ name: 'Z₁', format: '[32, 8·16·16] f32', type: 'act' }],
                detail: { preview: 'kconv' },
            },
            {
                id: 'pool1', title: 'relu · maxPool2', tag: 'CPU', lifetime: 'history', step: 'shape', desc: '8×16×16 → 8×8×8',
                dispatch: { kind: 'CPU', name: 'maxPool2', count: '32 × 8 × 64 个 2×2 窗口' },
                inputs: [{ name: 'Z₁', format: '[32, 8·16·16]', type: 'act' }],
                outputs: [{ name: 'P₁', format: '[32, 8·8·8] f32', type: 'act' }],
                detail: { preview: 'pool', formula: 2 },
            },
            {
                id: 'conv2', title: 'conv2d · conv2', tag: 'CPU', lifetime: 'history', step: 'shape', desc: '8 → 16 通道，3×3，p = 1',
                dispatch: { kind: 'CPU', name: 'conv2d', count: '32 × 16 × 64 × 72 = 2 359 296 次乘加' },
                inputs: [{ name: 'P₁', format: '[32, 8·8·8]', type: 'act' }, { name: 'Params', format: 'conv2.W [16, 72] · b [16]', type: 'param', history: true }],
                outputs: [{ name: 'Z₂', format: '[32, 16·8·8] f32', type: 'act' }],
                detail: { preview: 'multich', formula: 1 },
            },
            {
                id: 'pool2', title: 'relu · maxPool2', tag: 'CPU', lifetime: 'history', step: 'features', desc: '16×8×8 → 16×4×4 = 256',
                dispatch: { kind: 'CPU', name: 'maxPool2', count: '32 × 16 × 16 个 2×2 窗口' },
                inputs: [{ name: 'Z₂', format: '[32, 16·8·8]', type: 'act' }],
                outputs: [{ name: 'P₂', format: '[32, 256] f32', type: 'act' }],
                detail: { preview: 'fmaps', formula: 1 },
            },
            {
                id: 'fc', title: 'linear · fc', tag: 'CPU', lifetime: 'history', step: 'train', desc: 'flatten → 256 → 10',
                dispatch: { kind: 'CPU', name: 'linear', count: '32 × 256 × 10 = 81 920 次乘加' },
                inputs: [{ name: 'P₂', format: '[32, 256]', type: 'act' }, { name: 'Params', format: 'fc.W [256, 10] · b [10]', type: 'param', history: true }],
                outputs: [{ name: 'logits', format: '[32, 10] f32', type: 'act' }],
                detail: { preview: 'probs', formula: 0 },
            },
            {
                id: 'ce', title: 'crossEntropy', tag: 'CPU', lifetime: 'history', step: 'train', desc: 'softmax + mean(−log p[y])',
                dispatch: { kind: 'CPU', name: 'crossEntropy', count: '32 行 × 10' },
                inputs: [{ name: 'logits', format: '[32, 10]', type: 'act' }, { name: 'yb', format: '[32]', type: 'label' }],
                outputs: [{ name: 'L', format: 'scalar', type: 'scalar' }],
                detail: { preview: 'err', formula: 1 },
            },
            {
                id: 'bwd', title: 'backward（自动微分）', tag: 'CPU', lifetime: 'history', step: 'train', desc: '逆序执行 tape：fc → pool → conv2 → … → conv1',
                dispatch: { kind: 'CPU', name: 'backward', count: '约 2 × 前向的乘加（conv1 不需要 dX）' },
                inputs: [{ name: 'L', format: 'scalar', type: 'scalar' }, { name: 'logits', format: '[32, 10]', type: 'act' }, { name: 'P₂', format: '[32, 256]', type: 'act' }, { name: 'Z₂', format: '[32, 1024]', type: 'act' }, { name: 'P₁', format: '[32, 512]', type: 'act' }, { name: 'Z₁', format: '[32, 2048]', type: 'act' }, { name: 'Xb', format: '[32, 256]', type: 'img' }],
                outputs: [{ name: 'dParams', format: '与 Params 同形状', type: 'grad' }],
                detail: { preview: 'erfTrained', formula: 2 },
            },
            {
                id: 'adam', title: 'adamwStep', tag: 'CPU', category: 'output', lifetime: 'history', step: 'train', desc: 'Adam，lr 余弦衰减',
                dispatch: { kind: 'CPU', name: 'adamwStep', count: '3 818 个参数' },
                inputs: [{ name: 'dParams', format: '[3818]', type: 'grad' }, { name: 'Params', format: '上一步', type: 'param', history: true }, { name: 'Params₀', format: 't = 0 时', type: 'param' }, { name: 'Hyper', format: 'η(t)', type: 'scalar' }],
                outputs: [{ name: 'Params', format: 'conv1 · conv2 · fc', type: 'param' }],
                detail: { preview: 'kernels', formula: 3 },
            },
            {
                id: 'evalm', title: 'accuracy', tag: 'CPU', category: 'output', lifetime: 'history', step: 'train', desc: '分帧评估：权重快照 · 每帧 20 张',
                dispatch: { kind: 'CPU', name: 'accuracy', count: '500 + 500 张 / 轮' },
                inputs: [{ name: 'Params', format: '快照', type: 'param' }, { name: 'X_test', format: '[500, 256]', type: 'img' }, { name: 'y_test', format: '[500]', type: 'label' }],
                outputs: [{ name: 'Metrics', format: 'acc · loss · 混淆', type: 'scalar' }],
                detail: { preview: 'acc' },
            },
        ],
    },

    references: [
        { title: 'Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position', author: 'K. Fukushima', venue: 'Biological Cybernetics 36', year: 1980, note: '卷积 + 下采样的层级结构的前身' },
        { title: 'Receptive fields, binocular interaction and functional architecture in the cat\'s visual cortex', author: 'D. H. Hubel, T. N. Wiesel', venue: 'Journal of Physiology 160', year: 1962, note: '「感受野」概念的来源' },
        { title: 'Backpropagation Applied to Handwritten Zip Code Recognition', author: 'Y. LeCun et al.', venue: 'Neural Computation 1(4)', year: 1989, note: '用反向传播训练的卷积网络' },
        { title: 'Gradient-Based Learning Applied to Document Recognition', author: 'Y. LeCun, L. Bottou, Y. Bengio, P. Haffner', venue: 'Proceedings of the IEEE 86(11)', year: 1998, note: 'LeNet-5、MNIST（0.95% 错误率）' },
        { title: 'ImageNet Classification with Deep Convolutional Neural Networks', author: 'A. Krizhevsky, I. Sutskever, G. E. Hinton', venue: 'NIPS', year: 2012, note: 'AlexNet' },
        { title: 'Very Deep Convolutional Networks for Large-Scale Image Recognition', author: 'K. Simonyan, A. Zisserman', venue: 'ICLR', year: 2015, url: 'https://arxiv.org/abs/1409.1556', note: 'VGG：只用 3×3 卷积' },
        { title: 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift', author: 'S. Ioffe, C. Szegedy', venue: 'ICML', year: 2015, url: 'https://arxiv.org/abs/1502.03167', note: 'BatchNorm' },
        { title: 'Deep Residual Learning for Image Recognition', author: 'K. He, X. Zhang, S. Ren, J. Sun', venue: 'CVPR', year: 2016, url: 'https://arxiv.org/abs/1512.03385', note: 'ResNet、退化问题' },
        { title: 'Identity Mappings in Deep Residual Networks', author: 'K. He, X. Zhang, S. Ren, J. Sun', venue: 'ECCV', year: 2016, url: 'https://arxiv.org/abs/1603.05027', note: 'pre-activation 残差块，1001 层' },
        { title: 'U-Net: Convolutional Networks for Biomedical Image Segmentation', author: 'O. Ronneberger, P. Fischer, T. Brox', venue: 'MICCAI', year: 2015, url: 'https://arxiv.org/abs/1505.04597', note: 'U-Net 与跳连' },
        { title: 'Fully Convolutional Networks for Semantic Segmentation', author: 'J. Long, E. Shelhamer, T. Darrell', venue: 'CVPR', year: 2015, note: '全卷积网络做逐像素预测' },
        { title: 'Understanding the Effective Receptive Field in Deep Convolutional Neural Networks', author: 'W. Luo, Y. Li, R. Urtasun, R. Zemel', venue: 'NIPS', year: 2016, url: 'https://arxiv.org/abs/1701.04128', note: '有效感受野：用梯度测量，高斯形、按 √L 增长' },
        { title: 'A guide to convolution arithmetic for deep learning', author: 'V. Dumoulin, F. Visin', venue: 'arXiv:1603.07285', year: 2016, url: 'https://arxiv.org/abs/1603.07285', note: '步长、补零、转置卷积的尺寸关系' },
        { title: 'Making Convolutional Networks Shift-Invariant Again', author: 'R. Zhang', venue: 'ICML', year: 2019, url: 'https://arxiv.org/abs/1904.11486', note: '下采样的混叠与抗混叠池化' },
        { title: 'Why do deep convolutional networks generalize so poorly to small image transformations?', author: 'A. Azulay, Y. Weiss', venue: 'JMLR 20', year: 2019, note: 'CNN 对小平移并不鲁棒' },
        { title: 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour', author: 'P. Goyal et al.', venue: 'arXiv:1706.02677', year: 2017, url: 'https://arxiv.org/abs/1706.02677', note: '残差块最后一个 BN 的 γ 初始化为 0' },
        { title: 'Fixup Initialization: Residual Learning Without Normalization', author: 'H. Zhang, Y. N. Dauphin, T. Ma', venue: 'ICLR', year: 2019, url: 'https://arxiv.org/abs/1901.09321', note: '不用 BN 时的残差网络初始化' },
        { title: 'Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks', author: 'S. De, S. L. Smith', venue: 'NeurIPS', year: 2020, url: 'https://arxiv.org/abs/2002.10444', note: 'SkipInit' },
        { title: 'A Mean Field Theory of Batch Normalization', author: 'G. Yang, J. Pennington, V. Rao, J. Sohl-Dickstein, S. S. Schoenholz', venue: 'ICLR', year: 2019, url: 'https://arxiv.org/abs/1902.08129', note: '没有残差的深层 BN 网络在初始化时梯度爆炸' },
        { title: 'You Only Look Once: Unified, Real-Time Object Detection', author: 'J. Redmon, S. Divvala, R. Girshick, A. Farhadi', venue: 'CVPR', year: 2016, note: 'YOLO 检测' },
        { title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', author: 'A. Dosovitskiy et al.', venue: 'ICLR', year: 2021, url: 'https://arxiv.org/abs/2010.11929', note: 'ViT（第 07 篇）' },
        { title: 'A ConvNet for the 2020s', author: 'Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie', venue: 'CVPR', year: 2022, url: 'https://arxiv.org/abs/2201.03545', note: 'ConvNeXt' },
        { title: 'Denoising Diffusion Probabilistic Models', author: 'J. Ho, A. Jain, P. Abbeel', venue: 'NeurIPS', year: 2020, url: 'https://arxiv.org/abs/2006.11239', note: '去噪 U-Net（第 11 篇）' },
        { title: 'Segment Anything', author: 'A. Kirillov et al.', venue: 'ICCV', year: 2023, url: 'https://arxiv.org/abs/2304.02643', note: 'SAM 与 SA-1B' },
        { title: 'Fast Algorithms for Convolutional Neural Networks', author: 'A. Lavin, S. Gray', venue: 'CVPR', year: 2016, url: 'https://arxiv.org/abs/1509.09308', note: 'Winograd 卷积' },
        { title: 'CS231n: Convolutional Neural Networks for Visual Recognition', author: 'Stanford CS231n', url: 'https://cs231n.github.io/convolutional-networks/', note: '卷积层、池化、感受野的经典讲义' },
        { title: 'Deep Learning（第 9 章 卷积网络）', author: 'I. Goodfellow, Y. Bengio, A. Courville', venue: 'MIT Press', year: 2016, url: 'https://www.deeplearningbook.org/', note: '等变、池化与卷积的先验' },
    ],
});
