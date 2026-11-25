// 07_transformer.lab.js — 系列第 07 篇：Transformer，逐层拆解一次前向传播
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js llm_weights.js 07_transformer.lab.js -o 07_transformer.html
// 页面里的每张预览都由下面的 JS 实时算出：用 llm_weights.js 里训练好的迷你 GPT（或同结构的随机初始化）对所选句子做一次真实前向，
// 画出每一步的中间张量；消融、logit lens、KV cache 校验、online softmax 也都是真实计算。
// 正文里关于模型行为的说法都先在 Node 里测过（pages/tools/）：p07_heads.js / p07_sents.js（各头看向哪里）、p07_copy.js（抄写还是回忆）、
// p07_ablate.js（逐个消融）、p07_ffn.js（#38 号神经元）、p07_norm*.js（‖X2‖ 与预测熵）、p07_kv.js（KV cache 校验、logit lens）、p07_cf.js / p07_check.js / p07_keys.js（核对引用的数字）。

const P04 = (() => {
    const VOC = LM.vocab, CFG = LM_WEIGHTS.base.cfg;
    const D = CFG.d, H = CFG.H, DH = D / H, NL = CFG.L, FF = CFG.ff, NV = CFG.V, TMAX = CFG.T;
    const C = LM.C, FONT = LM.FONT;

    // ── 输入句子（≤ 21 字，保证 T×T 热图上每个字都写得下）；q = 切换到该句时默认的查询位置 ──
    const SENTS = [
        { s: '熊猫喜欢吃竹子，所以熊猫每天都吃竹子。', q: 15, label: '熊猫喜欢吃竹子，所以熊猫每天都吃竹子。' },
        { s: '小狗喜欢吃骨头，所以小狗每天都吃骨头。', q: 15, label: '小狗喜欢吃骨头，所以小狗每天都吃骨头。' },
        { s: '熊猫喜欢吃香蕉，所以熊猫每天都吃竹子。', q: 9, label: '（反事实）熊猫喜欢吃香蕉，所以熊猫…' },
        { s: '老虎喜欢吃竹子，所以老虎每天都吃肉。', q: 9, label: '（反事实）老虎喜欢吃竹子，所以老虎…' },
        { s: '小猫住在屋里。猴子最爱吃香蕉，它住在森林。', q: 17, label: '小猫住在屋里。猴子最爱吃香蕉，它住在森林。' },
        { s: '今天小兔在草地吃了很多萝卜。', q: 10, label: '今天小兔在草地吃了很多萝卜。' },
        { s: '小猫吃鱼，老虎吃肉。', q: 7, label: '小猫吃鱼，老虎吃肉。' },
        { s: '黑白的熊猫喜欢吃竹子。', q: 7, label: '黑白的熊猫喜欢吃竹子。' },
    ];
    // 消融选项：8 个头 + 2 个 MLP
    const COMPS = [{ value: 'none', label: '不消融', short: '不消融' }];
    for (let l = 0; l < NL; l++) for (let h = 0; h < H; h++) COMPS.push({ value: `h${l}.${h}`, label: `L${l + 1}·H${h + 1}`, short: `L${l + 1}·H${h + 1}` });
    for (let l = 0; l < NL; l++) COMPS.push({ value: `m${l}`, label: `L${l + 1} 的 MLP`, short: `L${l + 1}·MLP` });
    const HL = (l, h) => `L${l + 1}·H${h + 1}`;
    // 残差流的 5 个检查点（与节点图里的资源名一致）
    const CP_NAME = ['X0', 'X1a', 'X1', 'X2a', 'X2'];
    const CP_LABEL = ['X0 嵌入', 'X1a +L1注意力', 'X1 +L1 MLP', 'X2a +L2注意力', 'X2 +L2 MLP'];
    const WR_LABEL = ['L1 注意力', 'L1 MLP', 'L2 注意力', 'L2 MLP'];
    const show = (id) => VOC.show(id);
    const chars = (ids) => Array.from(ids, show);
    const posLabels = (ids) => Array.from(ids, (id, i) => `${i} ${show(id)}`);
    const clampQ = (S) => Math.min(S.query, S.T - 1);

    // ═════════════════════════════ 前向与分析 ═════════════════════════════
    function targetsOf(ids) { const T = ids.length, t = new Int32Array(T); for (let i = 0; i < T; i++) t[i] = i + 1 < T ? ids[i + 1] : -1; return t; }
    function rowStats(a, T, n) {
        const mu = new Float32Array(T), sd = new Float32Array(T);
        for (let i = 0; i < T; i++) {
            let m = 0; for (let j = 0; j < n; j++) m += a[i * n + j]; m /= n;
            let v = 0; for (let j = 0; j < n; j++) v += (a[i * n + j] - m) ** 2;
            mu[i] = m; sd[i] = Math.sqrt(v / n);
        }
        return { mu, sd };
    }
    function rowNorms(a, T, n) { const o = new Float32Array(T); for (let i = 0; i < T; i++) { let s = 0; for (let j = 0; j < n; j++) s += a[i * n + j] ** 2; o[i] = Math.sqrt(s); } return o; }

    /** 一次完整前向（LM.gpt.forward + trace），再从 trace 派生出各步要画的量。light = 只要 logits / loss */
    function analyse(m, ids, light) {
        const AG = LM.AG, T = ids.length, tg = targetsOf(ids);
        const out = AG.noGrad(() => LM.gpt.forward(m, ids, 1, T, { trace: !light, targets: tg, keepProbs: true }));
        const R = { T, ids, tg, logits: out.logits.data, probs: out.loss.probs, rowLoss: out.loss.rowLoss, loss: out.loss.item() };
        if (light) return R;
        const tr = out.trace, sc0 = 1 / Math.sqrt(DH);
        R.tok = tr.tok.data; R.x0 = tr.x0.data; R.pos = m.wpe.data.slice(0, T * D);
        R.layers = tr.layers.map((Ly, l) => {
            const p = m.layers[l], q = Ly.q.data, k = Ly.k.data;
            // 打分 S = QKᵀ/√dh（完整 T×T，含将被掩掉的上三角）与注意力权重（掩掉处 = NaN，画成斜纹）
            const sc = new Float32Array(H * T * T), att = new Float32Array(H * T * T);
            for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
                let s = 0; const qo = i * D + h * DH, ko = j * D + h * DH;
                for (let c = 0; c < DH; c++) s += q[qo + c] * k[ko + c];
                const o = (h * T + i) * T + j;
                sc[o] = s * sc0; att[o] = j <= i ? Ly.att[o] : NaN;
            }
            // 每个头写进残差流的量 ‖Y_h · W_O[8h : 8h+8, :]‖
            const y = Ly.a.data, wo = p.wo.data, hw = new Float32Array(T * H), z = new Float32Array(D);
            for (let i = 0; i < T; i++) for (let h = 0; h < H; h++) {
                z.fill(0);
                for (let e = 0; e < DH; e++) { const yv = y[i * D + h * DH + e], ro = (h * DH + e) * D; for (let c = 0; c < D; c++) z[c] += yv * wo[ro + c]; }
                let ss = 0; for (let c = 0; c < D; c++) ss += z[c] * z[c]; hw[i * H + h] = Math.sqrt(ss);
            }
            // FFN 预激活 u = h2·W1 + b1（trace 里的 f1 是 GELU 之后的）
            const h2 = Ly.h2.data, w1 = p.w1.data, b1 = p.b1.data, u = new Float32Array(T * FF);
            for (let i = 0; i < T; i++) for (let n = 0; n < FF; n++) { let s = b1[n]; for (let j = 0; j < D; j++) s += h2[i * D + j] * w1[j * FF + n]; u[i * FF + n] = s; }
            // 每个神经元的「值」向量 W2[n] 直接推高哪个 token：(W2[n] ⊙ γ_f) · W_te[v]（忽略 ln_f 的除以标准差）
            const w2 = p.w2.data, gf = m.lnfg.data, E = m.wte.data, promo = new Int32Array(FF), promo2 = new Int32Array(FF), promoV = new Float32Array(FF), wg = new Float32Array(D);
            for (let n = 0; n < FF; n++) {
                let any = 0; for (let j = 0; j < D; j++) { wg[j] = w2[n * D + j] * gf[j]; any += Math.abs(wg[j]); }
                let b1 = -1, v1 = -Infinity, b2 = -1, v2 = -Infinity;
                if (any > 0) for (let v = 4; v < NV; v++) {
                    let s = 0; for (let j = 0; j < D; j++) s += wg[j] * E[v * D + j];
                    if (s > v1) { b2 = b1; v2 = v1; b1 = v; v1 = s; } else if (s > v2) { b2 = v; v2 = s; }
                }
                promo[n] = b1; promo2[n] = b2; promoV[n] = b1 >= 0 ? v1 : 0;
            }
            return {
                xin: Ly.xin.data, h1: Ly.h1.data, q, k, v: Ly.v.data, sc, att, y, ao: Ly.ao.data, x1: Ly.x1.data, h2, u, f1: Ly.f1.data, f2: Ly.f2.data, x2: Ly.x2.data, hw, promo, promo2, promoV,
                sIn: rowStats(Ly.xin.data, T, D), sOut: rowStats(Ly.h1.data, T, D), g: p.ln1g.data, b: p.ln1b.data,
            };
        });
        // 残差流检查点与各子层写入的大小
        R.cps = [R.x0]; R.writes = [];
        R.layers.forEach((Ly) => { R.cps.push(Ly.x1, Ly.x2); R.writes.push(Ly.ao, Ly.f2); });
        R.cpNorm = R.cps.map((x) => rowNorms(x, T, D));
        R.wrNorm = R.writes.map((x) => rowNorms(x, T, D));
        // logit lens：把每个检查点直接接上 ln_f + W_teᵀ
        R.lens = R.cps.map((x) => {
            const hf = AG.noGrad(() => AG.layernorm(AG.tensor(T, D, x), m.lnfg, m.lnfb));
            const lg = AG.noGrad(() => AG.matmulBT(hf, m.wte)).data;
            const top = new Int32Array(T), ptop = new Float32Array(T), ptgt = new Float32Array(T), pr = new Float32Array(NV);
            for (let i = 0; i < T; i++) { LM.softmax(lg, pr, 1, i * NV, NV); const b = LM.argmax(pr); top[i] = b; ptop[i] = pr[b]; ptgt[i] = tg[i] >= 0 ? pr[tg[i]] : NaN; }
            return { top, ptop, ptgt };
        });
        return R;
    }
    const probsRow = (F, i) => F.probs.subarray(i * NV, (i + 1) * NV);

    /** 训练后 / 随机初始化模型，可选消融（克隆后把该头在 W_O 里对应的 8 行、或整个 MLP 的输出置零）；按需缓存 */
    function model(S, which, comp) {
        const m0 = which === 'rand' ? S.mRand : S.mBase;
        if (!comp || comp === 'none') return m0;
        const cache = S.ablCache[which];
        if (cache[comp]) return cache[comp];
        const c = LM.gpt.clone(m0);
        if (comp[0] === 'h') { const l = +comp[1], h = +comp[3]; c.layers[l].wo.data.fill(0, h * DH * D, (h + 1) * DH * D); }
        else { const l = +comp[1]; c.layers[l].w2.data.fill(0); c.layers[l].b2.data.fill(0); }
        return (cache[comp] = c);
    }
    /** 在查询位置 i 把 10 个部件逐一消融，记录「正确的下一个字」的概率 */
    function ablScan(S) {
        const ids = S.ids, T = S.T, i = S.qi;
        const tgt = i + 1 < T ? ids[i + 1] : LM.argmax(probsRow(S.F0, i));
        const res = COMPS.map((c) => {
            const lg = LM.AG.noGrad(() => LM.gpt.forward(model(S, S.weights, c.value), ids, 1, T).logits.data);
            const p = LM.softmax(lg, null, 1, i * NV, NV), top = LM.argmax(p);
            return { comp: c.value, label: c.short, p: p[tgt], top, ptop: p[top] };
        });
        return { tgt, res, last: i + 1 >= T };
    }

    // ── 生成：KV cache（LM.gpt.generate）+ 每一步与完整前向的逐位比较 + FLOP 计数 ──
    const layerFlops = (t) => 2 * (4 * D * D + 2 * D * FF) + 4 * D * (t + 1);   // 处理位置 t 的一个 token（看 t+1 个 key）
    const headFlops = 2 * D * NV;
    function generate(S) {
        const m = model(S, S.weights, S.ablate), prompt = Array.from(S.ids.slice(0, S.qi + 1)), P = prompt.length;
        const maxNew = TMAX - P, logs = [];
        const g = LM.gpt.generate(m, prompt, maxNew, {
            greedy: !(S.temp > 0), temperature: Math.max(S.temp, 1e-3), rand: PL.rng((S.seed ^ 0x51f15e) >>> 0),
            onStep: (dist, id, lg) => logs.push(Float32Array.from(lg)),
        });
        const diffs = logs.map((lg, k) => {
            const T = P + k, x = Int32Array.from(g.ids.slice(0, T));
            const full = LM.AG.noGrad(() => LM.gpt.forward(m, x, 1, T).logits.data);
            let d = 0; for (let v = 0; v < NV; v++) d = Math.max(d, Math.abs(full[(T - 1) * NV + v] - lg[v]));
            return d;
        });
        // 累计 FLOPs：生成第 k 个新 token 时，KV cache 只处理新位置；不用 cache 则把整段前缀重算一遍
        const kv = [0], full = [0];
        let accKV = 0, accFull = 0, pre = 0;
        for (let s = 0; s < P - 1; s++) pre += NL * layerFlops(s);   // 提示词（除最后一个字）
        accKV = pre;
        for (let k = 1; k <= g.gen.length; k++) {
            const t = P + k - 2;                        // 产生第 k 个新 token 需要位置 t 的输出
            accKV += NL * layerFlops(t) + headFlops;
            let f = 0; for (let s = 0; s <= t; s++) f += NL * layerFlops(s);
            accFull += f + headFlops;
            kv.push(accKV); full.push(accFull);
        }
        return { ids: g.ids, P, gen: g.gen, diffs, kv, full };
    }

    // ── √dₖ 演示：真实随机采样（resident，一次算好 4 种维度）──
    const DKS = [8, 64, 128, 512];
    function dotDemo(rand) {
        const N = 1000, out = {};
        DKS.forEach((dk) => {
            const raw = new Float32Array(N);
            for (let n = 0; n < N; n++) { let s = 0; for (let c = 0; c < dk; c++) s += LM.randn(rand) * LM.randn(rand); raw[n] = s; }
            const q = new Float32Array(dk); for (let c = 0; c < dk; c++) q[c] = LM.randn(rand);
            const sc = new Float32Array(16);
            for (let j = 0; j < 16; j++) { let s = 0; for (let c = 0; c < dk; c++) s += q[c] * LM.randn(rand); sc[j] = s; }
            let m = 0, v = 0; for (let n = 0; n < N; n++) m += raw[n] / N; for (let n = 0; n < N; n++) v += (raw[n] - m) ** 2 / N;
            out[dk] = { raw, sc, std: Math.sqrt(v) };
        });
        return out;
    }
    function hist(data, lo, hi, nb, scale) {
        const c = new Float32Array(nb), w = (hi - lo) / nb;
        for (let i = 0; i < data.length; i++) { const v = data[i] * (scale || 1), b = Math.floor((v - lo) / w); if (b >= 0 && b < nb) c[b]++; }
        for (let b = 0; b < nb; b++) c[b] /= data.length * w;   // 概率密度
        return { c, x: Array.from({ length: nb }, (_, b) => lo + (b + 0.5) * w) };
    }

    // ── online softmax（FlashAttention 的核心）：按 key 分块扫描，只保留运行最大值 m、归一化和 ℓ、未归一化输出 o ──
    function online(F, l, h, i, bc) {
        const Ly = F.layers[l], q = Ly.q, k = Ly.k, v = Ly.v, n = i + 1, s = 1 / Math.sqrt(DH);
        let m = -Infinity, ell = 0; const o = new Float64Array(DH), tiles = [];
        for (let t0 = 0; t0 < n; t0 += bc) {
            const t1 = Math.min(n, t0 + bc), sc = []; let mt = -Infinity;
            for (let j = t0; j < t1; j++) { let d = 0; for (let c = 0; c < DH; c++) d += q[i * D + h * DH + c] * k[j * D + h * DH + c]; d *= s; sc.push(d); if (d > mt) mt = d; }
            const mn = Math.max(m, mt), r = Math.exp(m - mn);         // 第一块时 m = −∞ → r = 0
            for (let c = 0; c < DH; c++) o[c] *= r;
            let ps = 0;
            for (let j = t0; j < t1; j++) { const p = Math.exp(sc[j - t0] - mn); ps += p; for (let c = 0; c < DH; c++) o[c] += p * v[j * D + h * DH + c]; }
            ell = ell * r + ps; m = mn;
            tiles.push({ t0, t1, mt, m, ell, r, o1: o[0] / ell });
        }
        const out = Array.from(o, (x) => x / ell), ref = Array.from(Ly.y.subarray(i * D + h * DH, i * D + h * DH + DH));
        let md = 0; for (let c = 0; c < DH; c++) md = Math.max(md, Math.abs(out[c] - ref[c]));
        return { tiles, out, ref, maxDiff: md, n, bc };
    }

    // 参数量统计：本页模型（真实张量大小）与按配置推算的 GPT-2 small
    function paramBreakdown(V, T, d, L, ff) {
        const emb = V * d + T * d, attn = L * (4 * d * d + 4 * d), mlp = L * (2 * d * ff + ff + d), ln = L * 4 * d + 2 * d;
        return { emb, attn, mlp, ln, total: emb + attn + mlp + ln };
    }
    // 真实模型的 KV cache（每个 token，fp16）：2 × 层数 × KV 头数 × 头宽 × 2 字节。配置取自各自的论文 / 官方配置
    const KV_REAL = [
        { name: 'GPT-2 small（MHA）', L: 12, kv: 12, dh: 64 },
        { name: 'LLaMA-2 7B（MHA）', L: 32, kv: 32, dh: 128 },
        { name: 'Mistral 7B（GQA 32→8）', L: 32, kv: 8, dh: 128 },
        { name: 'Llama-3 8B（GQA 32→8）', L: 32, kv: 8, dh: 128 },
        { name: 'Llama-2 70B（GQA 64→8）', L: 80, kv: 8, dh: 128 },
    ];

    // ═════════════════════════════ 绘图（深色主题，设计坐标；所有文字 ≥ 11 px） ═════════════════════════════
    const cm = (n) => (typeof n === 'function' ? n : PL.cmap[n] || PL.cmap.viridis);
    const rgbS = LM.rgbStr;
    function rangeOf(data, o) {
        if (o.range) return o.range;
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < data.length; i++) { const v = data[i]; if (v === v && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
        if (!isFinite(lo)) { lo = 0; hi = 1; }
        if (o.symmetric) { const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1; return [-m, m]; }
        if (hi - lo < 1e-9) hi = lo + 1e-9;
        return [lo, hi];
    }
    const fmtR = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
    const fmtF = (v) => (v >= 1e9 ? (v / 1e9).toFixed(2) + ' G' : v >= 1e6 ? (v / 1e6).toFixed(2) + ' M' : v >= 1e3 ? (v / 1e3).toFixed(1) + ' k' : String(Math.round(v)));
    const lum = (rgb) => 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];

    /** 在已有 ctx 的 (x0, y0) 画 R × Cn 个格子。o: { map, lo, hi, dim(r,c)→alpha, vsep, hiRow, hiCol, hiCols:[a,b], hiCells:[[r,c,color]] } */
    function cells(ctx, data, R, Cn, x0, y0, cw, ch, o) {
        const map = cm(o.map), rgb = [0, 0, 0], lo = o.lo, span = (o.hi - o.lo) || 1;
        for (let r = 0; r < R; r++) for (let c = 0; c < Cn; c++) {
            const v = data[r * Cn + c], X = x0 + c * cw, Y = y0 + r * ch;
            if (v !== v) {
                ctx.fillStyle = '#101216'; ctx.fillRect(X, Y, Math.ceil(cw), Math.ceil(ch));
                if (cw >= 5) { ctx.strokeStyle = '#20242b'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X, Y + ch); ctx.lineTo(X + cw, Y); ctx.stroke(); }
                continue;
            }
            map(Math.min(1, Math.max(0, (v - lo) / span)), rgb);
            ctx.globalAlpha = o.dim ? o.dim(r, c) : 1;
            ctx.fillStyle = rgbS(rgb); ctx.fillRect(X, Y, Math.ceil(cw), Math.ceil(ch));
            ctx.globalAlpha = 1;
        }
        if (cw >= 7 && ch >= 7) {
            ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 1; ctx.beginPath();
            for (let c = 0; c <= Cn; c++) { ctx.moveTo(x0 + c * cw, y0); ctx.lineTo(x0 + c * cw, y0 + R * ch); }
            for (let r = 0; r <= R; r++) { ctx.moveTo(x0, y0 + r * ch); ctx.lineTo(x0 + Cn * cw, y0 + r * ch); }
            ctx.stroke();
        }
        if (o.vsep) {
            ctx.strokeStyle = 'rgba(220,220,220,0.55)'; ctx.lineWidth = 1; ctx.beginPath();
            for (let c = o.vsep; c < Cn; c += o.vsep) { ctx.moveTo(x0 + c * cw, y0 - 2); ctx.lineTo(x0 + c * cw, y0 + R * ch + 2); }
            ctx.stroke();
        }
        ctx.lineWidth = 2;
        if (o.hiCols) { ctx.strokeStyle = C.yellow; ctx.strokeRect(x0 + o.hiCols[0] * cw, y0 - 1, (o.hiCols[1] - o.hiCols[0]) * cw, R * ch + 2); }
        if (o.hiCol != null && o.hiCol >= 0) { ctx.strokeStyle = C.yellow; ctx.strokeRect(x0 + o.hiCol * cw, y0 - 1, cw, R * ch + 2); }
        if (o.hiRow != null && o.hiRow >= 0 && o.hiRow < R) { ctx.strokeStyle = C.yellow; ctx.strokeRect(x0 - 1, y0 + o.hiRow * ch, Cn * cw + 2, ch); }
        (o.hiCells || []).forEach(([r, c, col]) => { ctx.strokeStyle = col || C.green; ctx.strokeRect(x0 + c * cw + 1, y0 + r * ch + 1, cw - 2, ch - 2); });
    }
    function legendBar(ctx, x, y, hgt, map, lo, hi) {
        const rgb = [0, 0, 0], f = cm(map);
        for (let i = 0; i < hgt; i++) { f(1 - i / (hgt - 1), rgb); ctx.fillStyle = rgbS(rgb); ctx.fillRect(x, y + i, 7, 1.5); }
        ctx.fillStyle = C.gray; ctx.font = FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(fmtR(hi), x + 10, y - 1); ctx.textBaseline = 'bottom'; ctx.fillText(fmtR(lo), x + 10, y + hgt + 1);
    }
    /** 单个矩阵热图（行列带文字标签）。o: { rows, cols, title, map, range, symmetric, square, legend, dim, vsep, hiRow, hiCol, hiCells } → 布局 */
    function heat(canvas, data, R, Cn, o) {
        const { ctx, w, h } = PL.draw.ctx(canvas);
        ctx.font = FONT(11);
        const rl = o.rows, cl = o.cols;
        const padL = rl ? 8 + Math.max(...rl.map((s) => ctx.measureText(String(s)).width)) : 4;
        const padT = (o.title ? 18 : 3) + (cl ? 15 : 0), padR = o.legend === false ? 4 : 54, padB = 4;
        let cw = (w - padL - padR) / Cn, ch = (h - padT - padB) / R;
        if (o.square !== false) cw = ch = Math.min(cw, ch);
        const [lo, hi] = rangeOf(data, o), map = o.map || (o.symmetric ? 'diverge' : 'viridis');
        if (o.title) { ctx.fillStyle = C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, 4, 3); }
        cells(ctx, data, R, Cn, padL, padT, cw, ch, { map, lo, hi, dim: o.dim, vsep: o.vsep, hiRow: o.hiRow, hiCol: o.hiCol, hiCols: o.hiCols, hiCells: o.hiCells });
        ctx.font = FONT(11);
        if (rl) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; const st = Math.max(1, Math.ceil(12 / ch)); for (let r = 0; r < R; r += st) { ctx.fillStyle = r === o.hiRow ? C.yellow : C.gray; ctx.fillText(String(rl[r]), padL - 4, padT + (r + 0.5) * ch); } }
        if (cl) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; const st = Math.max(1, Math.ceil((ctx.measureText(String(cl[0])).width + 3) / cw)); for (let c = 0; c < Cn; c += st) { ctx.fillStyle = c === o.hiCol ? C.yellow : C.gray; ctx.fillText(String(cl[c]), padL + (c + 0.5) * cw, padT - 2); } }
        if (o.legend !== false) legendBar(ctx, padL + Cn * cw + 8, padT, Math.max(30, R * ch), map, lo, hi);
        return { x0: padL, y0: padT, cw, ch, R, C: Cn, ctx };
    }
    /** 若干矩阵并排（共享行标签、同样大小的格子），panel 之间可写运算符（+ = →）。返回每个 panel 的布局 */
    function panels(canvas, o) {
        const { ctx, w, h } = PL.draw.ctx(canvas);
        const R = o.rows.length, P = o.panels, gap = o.gap ?? 18;
        ctx.font = FONT(11);
        const padL = 8 + Math.max(...o.rows.map((s) => ctx.measureText(String(s)).width)), padT = 19, padB = 4;
        const sumC = P.reduce((s, p) => s + p.C, 0);
        let cw = (w - padL - gap * (P.length - 1) - 4) / sumC, ch = (h - padT - padB) / R;
        if (o.square) cw = ch = Math.min(cw, ch);
        let x = padL; const lays = [];
        P.forEach((p, k) => {
            const [lo, hi] = rangeOf(p.data, p);
            if (k > 0 && p.op) { ctx.fillStyle = C.text; ctx.font = FONT(15, true); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(p.op, x - gap / 2, padT + R * ch / 2); }
            ctx.font = FONT(11, true); ctx.fillStyle = p.titleColor || C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            ctx.fillText(p.title + (p.noRange ? '' : ' ' + (p.symmetric ? '±' + fmtR(hi) : fmtR(lo) + '…' + fmtR(hi))), x, 3);
            cells(ctx, p.data, R, p.C, x, padT, cw, ch, { map: p.map || (p.symmetric ? 'diverge' : 'viridis'), lo, hi, vsep: p.vsep, hiRow: o.hiRow, hiCols: p.hiCols });
            lays.push({ x0: x, y0: padT, cw, ch, R, C: p.C });
            x += p.C * cw + gap;
        });
        ctx.font = FONT(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        const st = Math.max(1, Math.ceil(12 / ch));
        for (let r = 0; r < R; r += st) { ctx.fillStyle = r === o.hiRow ? C.yellow : C.gray; ctx.fillText(String(o.rows[r]), padL - 4, padT + (r + 0.5) * ch); }
        lays.ctx = ctx; lays.w = w;
        return lays;
    }
    /** 2×2 的注意力小图（一层的 4 个头）。返回每个小图的布局 */
    function grid4(canvas, mats, T, o) {
        const { ctx, w, h } = PL.draw.ctx(canvas);
        const gap = 12, top = 17, cols = 2, rows = Math.ceil(mats.length / cols);
        const cs = Math.min((w - gap * (cols + 1)) / cols, (h - rows * (top + gap)) / rows) / T, pw = cs * T;
        const ox = (w - cols * pw - (cols - 1) * gap) / 2, lays = [];
        mats.forEach((mat, k) => {
            const cx = ox + (k % cols) * (pw + gap), cy = Math.floor(k / cols) * (pw + top + gap) + top + 2;
            ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = k === o.sel ? C.yellow : C.text;
            ctx.fillText(o.titles[k], cx, cy - 3);
            if (o.notes && o.notes[k]) { ctx.font = FONT(11); ctx.textAlign = 'right'; ctx.fillStyle = C.gray; ctx.fillText(o.notes[k], cx + pw, cy - 3); }
            cells(ctx, mat, T, T, cx, cy, cs, cs, { map: 'viridis', lo: 0, hi: 1, hiRow: o.hiRow });
            if (k === o.sel) { ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.strokeRect(cx - 3, cy - 3, pw + 6, pw + 6); }
            lays.push({ x0: cx, y0: cy, cw: cs, ch: cs, R: T, C: T });
        });
        return lays;
    }
    /**
     * 字符格子表：rows = [{ label, cells: [{ t, v, color, tc }] }]。v ∈ [0,1] 按色表上色（无 v 时用 color 或暗色底），t 写在格子中间。
     * o: { colLabels, hiCol, map, cwMax, chMax, px, legend: [lo, hi] }
     */
    function charGrid(canvas, o) {
        const { ctx, w, h } = PL.draw.ctx(canvas);
        const R = o.rows.length, Cn = o.rows[0].cells.length;
        ctx.font = FONT(11);
        const padL = 8 + Math.max(...o.rows.map((r) => ctx.measureText(r.label).width)), padT = (o.title ? 18 : 3) + (o.colLabels ? 15 : 0), padR = o.legend ? 54 : 4;
        const cw = Math.min(o.cwMax || 26, (w - padL - padR) / Cn), ch = Math.min(o.chMax || 26, (h - padT - 4) / R);
        const map = cm(o.map || 'viridis'), rgb = [0, 0, 0];
        if (o.title) { ctx.fillStyle = C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, 4, 3); }
        const px = Math.max(11, Math.min(o.px || 14, ch - 7));
        for (let r = 0; r < R; r++) for (let c = 0; c < Cn; c++) {
            const cell = o.rows[r].cells[c], X = padL + c * cw, Y = padT + r * ch;
            if (!cell) continue;
            let bg = cell.color || '#161a20', tc = cell.tc;
            if (cell.v != null && cell.v === cell.v) { map(Math.min(1, Math.max(0, cell.v)), rgb); bg = rgbS(rgb); if (!tc) tc = lum(rgb) > 140 ? '#000' : '#f0f0f0'; }
            ctx.fillStyle = bg; ctx.fillRect(X + 0.5, Y + 0.5, cw - 1, ch - 1);
            if (cell.t != null) { ctx.fillStyle = tc || '#e8e8e8'; ctx.font = FONT(cell.px || px, cell.bold); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(cell.t), X + cw / 2, Y + ch / 2 + 0.5); }
        }
        ctx.font = FONT(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        o.rows.forEach((r, k) => { ctx.fillStyle = r.labelColor || C.gray; ctx.fillText(r.label, padL - 5, padT + (k + 0.5) * ch); });
        if (o.colLabels) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; const st = Math.max(1, Math.ceil(20 / cw)); for (let c = 0; c < Cn; c += st) { ctx.fillStyle = c === o.hiCol ? C.yellow : C.gray; ctx.fillText(String(o.colLabels[c]), padL + (c + 0.5) * cw, padT - 2); } }
        if (o.hiCol != null && o.hiCol >= 0) { ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.strokeRect(padL + o.hiCol * cw, padT - 1, cw, R * ch + 2); }
        if (o.legend) legendBar(ctx, padL + Cn * cw + 8, padT, Math.max(30, R * ch), o.map || 'viridis', o.legend[0], o.legend[1]);
        return { x0: padL, y0: padT, cw, ch, R, C: Cn };
    }
    function niceTicks(a, b, n) {
        const span = b - a, raw = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw))), st = [1, 2, 2.5, 5, 10].map((q) => q * mag).find((s) => s >= raw);
        const out = []; for (let v = Math.ceil(a / st - 1e-9) * st; v <= b + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : +v.toPrecision(6));
        return out;
    }
    /**
     * 类别横轴（每个 token 一格，标签可以是汉字）的柱 / 折线图。
     * o: { labels, labels2, series: [{ data, color, kind: 'bar'|'line'|'dot', label, width, dash, alpha }], y: [lo, hi], hlines: [{ y, color, label, dash }],
     *      hiIdx, yLabel, ctx, rect: {x, y, w, h}, title }
     */
    function catPlot(canvas, o) {
        let ctx, X0 = 0, Y0 = 0, w, h;
        if (o.ctx) { ctx = o.ctx; X0 = o.rect.x; Y0 = o.rect.y; w = o.rect.w; h = o.rect.h; } else { const c = PL.draw.ctx(canvas); ctx = c.ctx; w = c.w; h = c.h; }
        const n = o.labels.length, padL = 38, padR = 8, padT = o.title ? 20 : 8, padB = o.labels2 ? 32 : 18;
        const [lo, hi] = o.y, pw = w - padL - padR, ph = h - padT - padB, bw = pw / n;
        const X = (i) => X0 + padL + (i + 0.5) * bw, Y = (v) => Y0 + padT + (1 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * ph;
        if (o.title) { ctx.fillStyle = C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, X0 + 4, Y0 + 3); }
        ctx.font = FONT(11); ctx.lineWidth = 1;
        niceTicks(lo, hi, Math.max(2, Math.floor(ph / 32))).forEach((v) => {
            ctx.strokeStyle = LM.C.grid; ctx.beginPath(); ctx.moveTo(X0 + padL, Y(v)); ctx.lineTo(X0 + w - padR, Y(v)); ctx.stroke();
            ctx.fillStyle = C.dim; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(+v.toFixed(2)), X0 + padL - 4, Y(v));
        });
        if (o.yLabel) { ctx.save(); ctx.translate(X0 + 9, Y0 + padT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillStyle = '#888'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
        if (o.hiIdx != null && o.hiIdx >= 0) { ctx.fillStyle = 'rgba(232,194,90,0.13)'; ctx.fillRect(X(o.hiIdx) - bw / 2, Y0 + padT, bw, ph); }
        const bars = o.series.filter((s) => s.kind === 'bar'), nb = bars.length;
        bars.forEach((s, k) => {
            const gw = bw * 0.78 / nb;
            for (let i = 0; i < n; i++) {
                const v = s.data[i]; if (v == null || v !== v) continue;
                const x = X(i) - bw * 0.39 + k * gw, y0 = Y(Math.max(lo, 0)), y1 = Y(v);
                ctx.globalAlpha = s.alpha ?? 1; ctx.fillStyle = typeof s.color === 'function' ? s.color(v, i) : s.color;
                ctx.fillRect(x, Math.min(y0, y1), Math.max(1, gw - 1), Math.max(1, Math.abs(y1 - y0)));
                ctx.globalAlpha = 1;
            }
        });
        o.series.filter((s) => s.kind !== 'bar').forEach((s) => {
            ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash || []); ctx.globalAlpha = s.alpha ?? 1;
            if (s.kind === 'line') { ctx.beginPath(); let pen = false; for (let i = 0; i < n; i++) { const v = s.data[i]; if (v == null || v !== v) { pen = false; continue; } if (pen) ctx.lineTo(X(i), Y(v)); else { ctx.moveTo(X(i), Y(v)); pen = true; } } ctx.stroke(); }
            ctx.setLineDash([]);
            for (let i = 0; i < n; i++) { const v = s.data[i]; if (v == null || v !== v) continue; ctx.beginPath(); ctx.arc(X(i), Y(v), s.r || (s.kind === 'dot' ? 3.2 : 2.2), 0, 7); ctx.fill(); }
            ctx.globalAlpha = 1;
            // 超出纵轴上限的点：贴在顶边，标出真实数值
            if (o.clipLabels) for (let i = 0; i < n; i++) { const v = s.data[i]; if (v > hi) { const t = '↑' + v.toFixed(0); ctx.font = FONT(11); const tw = ctx.measureText(t).width, tx = Math.min(X(i) + 4, X0 + w - padR - tw - 2); ctx.fillStyle = 'rgba(8,10,12,0.85)'; ctx.fillRect(tx - 2, Y0 + padT + 2, tw + 4, 14); ctx.fillStyle = s.color; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(t, tx, Y0 + padT + 3); } }
        });
        (o.hlines || []).forEach((l) => {
            ctx.strokeStyle = l.color || C.gray; ctx.lineWidth = 1.2; ctx.setLineDash(l.dash || [5, 4]);
            ctx.beginPath(); ctx.moveTo(X0 + padL, Y(l.y)); ctx.lineTo(X0 + w - padR, Y(l.y)); ctx.stroke(); ctx.setLineDash([]);
            if (l.label) { ctx.font = FONT(11); ctx.fillStyle = l.color || C.gray; ctx.textAlign = l.align || 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(l.label, l.align === 'right' ? X0 + w - padR - 2 : X0 + padL + 3, Y(l.y) - 2); }
        });
        ctx.font = FONT(11); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        const st = Math.max(1, Math.ceil(13 / bw));
        for (let i = 0; i < n; i += st) {
            ctx.fillStyle = i === o.hiIdx ? C.yellow : C.text; ctx.fillText(String(o.labels[i]), X(i), Y0 + h - padB + 3);
            if (o.labels2) { ctx.fillStyle = i === o.hiIdx ? C.yellow : C.dim; ctx.fillText(String(o.labels2[i]), X(i), Y0 + h - padB + 17); }
        }
        const lg = o.series.filter((s) => s.label);
        if (lg.length) {
            ctx.font = FONT(11); ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
            let ly = Y0 + padT + 8; const lw = Math.max(...lg.map((s) => ctx.measureText(s.label).width)) + 22, lx = o.legendX != null ? X0 + padL + o.legendX * pw : X0 + w - padR - lw;
            ctx.fillStyle = 'rgba(8,10,12,0.8)'; ctx.fillRect(lx - 4, ly - 8, lw + 6, lg.length * 15 + 2);
            lg.forEach((s) => {
                const col = s.legendColor || (typeof s.color === 'function' ? C.blue : s.color);
                ctx.fillStyle = col; ctx.strokeStyle = col;
                if (s.kind === 'bar') ctx.fillRect(lx, ly - 4, 12, 8);
                else if (s.dash) { ctx.lineWidth = 1.5; ctx.setLineDash(s.dash); ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 12, ly); ctx.stroke(); ctx.setLineDash([]); }
                else if (s.kind === 'dot') { ctx.beginPath(); ctx.arc(lx + 6, ly, 3.2, 0, 7); ctx.fill(); }
                else ctx.fillRect(lx, ly - 1, 12, 2.5);
                ctx.fillStyle = C.text; ctx.fillText(s.label, lx + 17, ly); ly += 15;
            });
        }
        return { X, Y, bw, padL: X0 + padL, n, top: Y0 + padT, bottom: Y0 + padT + ph };
    }
    /** probe / pick：设计坐标 → 格子 { r, c } */
    const hitLay = (lay, info) => { if (!lay || !info) return null; const c = Math.floor((info.dx - lay.x0) / lay.cw), r = Math.floor((info.dy - lay.y0) / lay.ch); return r >= 0 && r < lay.R && c >= 0 && c < lay.C ? { r, c } : null; };
    const hitCat = (lay, info) => { if (!lay || !info) return -1; const i = Math.floor((info.dx - lay.padL) / lay.bw); return i >= 0 && i < lay.n ? i : -1; };

    /** 选中位置的 top-k 下一个 token（横向柱状图），正确答案标绿 */
    function topkBars(canvas, F, i, k, title) {
        const p = probsRow(F, i), idx = LM.topk(p, k), tgt = F.tg[i];
        LM.draw.bars(canvas, idx.map((v) => p[v]), idx.map((v) => show(v)), { horizontal: true, range: [0, 1], digits: 3, title, hi: new Set(idx.map((v, r) => (v === tgt ? r : -1)).filter((r) => r >= 0)), hiColor: C.green, color: C.blue });
    }

    return {
        D, H, DH, NL, FF, NV, TMAX, SENTS, COMPS, HL, CP_NAME, CP_LABEL, WR_LABEL, DKS, KV_REAL, show, chars, posLabels, clampQ,
        analyse, model, ablScan, generate, dotDemo, hist, online, paramBreakdown, probsRow, layerFlops, headFlops,
        heat, panels, grid4, charGrid, catPlot, cells, hitLay, hitCat, topkBars, fmtR, fmtF, rangeOf,
    };
})();

// ═════════════════════════════ C++ 代码里本页公用的部分（graph.common）：词表（由真实词表生成）、读权重、几个小工具 ═════════════════════════════
const P04_CPP_COMMON = `
    // 字符级词表（与页面 JS 的 LM.vocab 逐个相同）：4 个特殊 token + 72 个汉字 / 标点，id = 在 itos 里的下标
    Vocab makeVocab() {
        Vocab v;
        v.itos = {"<pad>", "<eos>", "<user>", "<bot>"};
        for (const std::string& ch : utf8Chars("${LM.vocab.itos.slice(4).join('')}")) v.itos.push_back(ch);   // 一个汉字 = 一个 token
        return v;
    }

    // 训练好的基座权重：按 m.params 的顺序依次填满每个参数（页面里内嵌的是同一份数据，fp16 存储）
    void loadWeights(GPT& m, const std::string& path);   // 读文件、fp16 → fp32（实现略）

    // 本页的两套权重：trained = 训练 4000 步后的基座模型；否则 = 训练的起点（同一个种子的随机初始化）
    GPT loadModel(bool trained) {
        GPTConfig cfg;                                   // V = 76, T = 32, d = 32, L = 2, H = 4, ff = 128
        GPT m = makeGPT(cfg, 1);                         // GPT-2 式初始化，与页面 JS 的 LM.gpt.create(cfg, 1) 逐位相同
        if (trained) loadWeights(m, "base.bin");
        return m;
    }

    // 深拷贝一个模型（消融时改的是副本）。GPT 里的 Var 是共享指针：直接写 GPT c = m; 会和原模型共用同一批参数
    GPT cloneGPT(const GPT& m) {
        GPT c = makeGPT(m.cfg, 1);
        for (size_t k = 0; k < m.params.size(); k++) c.params[k]->val = m.params[k]->val;
        return c;
    }

    // 每个位置的答案 = 下一个字；最后一个位置没有答案，记 −1（crossEntropy 会跳过它）
    std::vector<int> targetsOf(const std::vector<int>& ids) {
        std::vector<int> tg = ints((int)ids.size(), -1);
        for (int i = 0; i + 1 < (int)ids.size(); i++) tg[i] = ids[i + 1];
        return tg;
    }

    // 张量的第 i 行复制成数组（取某个位置的 logits / 激活）
    std::vector<float> rowOf(const Tensor& t, int i) {
        return std::vector<float>(t.v.begin() + (size_t)i * t.cols, t.v.begin() + (size_t)(i + 1) * t.cols);
    }

    // 读完前 i+1 个字之后「下一个字」的概率分布：完整前向一次（gptForward），取 logits 的第 i 行做 softmax
    std::vector<float> nextTokenProbs(GPT& m, const std::vector<int>& ids, int i) {
        NoGrad ng;                                       // 只做前向分析，不记录反向传播用的 tape
        std::vector<float> p = rowOf(gptForward(m, ids, 1, (int)ids.size())->val, i);
        softmaxInPlace(p, 1.f);
        return p;
    }`;

PipelineLab.mount({
    paramSteps: { weights: ['embed', 'ln', 'qkv', 'scores', 'attn', 'proj', 'resid', 'ffn', 'out', 'loss', 'gen', 'prod'], sent: ['embed', 'ln', 'qkv', 'scores', 'attn', 'proj', 'resid', 'ffn', 'out', 'loss', 'gen', 'prod'], query: ['embed', 'ln', 'qkv', 'scores', 'attn', 'proj', 'resid', 'ffn', 'out', 'loss', 'gen', 'prod'], layer: ['ln', 'qkv', 'scores', 'attn', 'proj', 'ffn', 'prod'], head: ['qkv', 'scores', 'attn', 'proj', 'prod'], ablate: ['ln', 'qkv', 'scores', 'attn', 'proj', 'resid', 'ffn', 'out', 'loss', 'gen', 'prod'], dk: ['scores'], temp: ['gen'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '07 · Transformer：逐层拆解一次前向传播',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 20251123,
    css: LM.CSS,
    regen: { label: '重新采样', title: '重新抽取 Step 04 里的随机向量、Step 11 采样用的随机数（模型权重不变）' },

    params: [
        {
            id: 'weights', label: '权重', value: 'base',
            options: [{ value: 'base', label: '训练后（基座模型）' }, { value: 'rand', label: '随机初始化（训练前）' }],
            desc: '同一个网络的两套权重：训练 4000 步后的基座模型，和训练开始时的随机初始化（同一个随机种子）。切到随机初始化：注意力变成均匀分布、所有预测都 ≈ 1/76——结构相同，本领全在权重里。',
        },
        {
            id: 'sent', label: '输入句子', value: 0,
            options: P04.SENTS.map((s, i) => ({ value: i, label: s.label })),
            onChange: (v, S, PL) => PL.setParam('query', P04.SENTS[v].q),
            desc: '送进模型的句子（字符级，一个字一个 token）。「反事实」两句故意把食物和动物配错，用来区分模型是在「抄」前文还是凭名字「回忆」。切换句子会把查询位置跳到该句最有看点的位置。',
        },
        {
            id: 'query', label: '查询位置 i', min: 0, max: 20, step: 1, value: 15,
            format: (raw, S) => { const s = Array.from(P04.SENTS[S.sent ?? 0].s), i = Math.min(+raw, s.length - 1); return `${i} ${s[i]}`; },
            desc: '要细看的 token 位置：注意力图上高亮的那一行、Step 09 预测的是它的下一个字、消融和 logit lens 也看它。大多数注意力 / 逐 token 的图上可以直接点击选择。',
        },
        {
            id: 'layer', label: '层', value: 1, options: [{ value: 0, label: 'L1（第 1 层）' }, { value: 1, label: 'L2（第 2 层）' }],
            desc: 'Step 02–06、08 显示哪一层 Transformer Block 的张量。L2 的「主语头」和 MLP 是这个模型完成事实回忆的地方。',
        },
        {
            id: 'head', label: '头', value: 0, options: [0, 1, 2, 3].map((h) => ({ value: h, label: 'H' + (h + 1) })),
            desc: 'Step 03–06 放大哪一个注意力头（每层 4 个，每个 8 维）。默认 L2·H1：在「吃」处盯住主语名字的第 2 个字。',
        },
        {
            id: 'ablate', label: '消融', value: 'none', options: P04.COMPS.map((c) => ({ value: c.value, label: c.label })),
            desc: '把一个注意力头（它在 W_O 里的 8 行）或一整个 MLP 的输出置零后重新前向，全页的图都换成消融后的模型。试试 L2 的 MLP（i = 15）或 L1·H4（i = 9）。',
        },
        {
            id: 'dk', label: '点积维度 dₖ', value: 64, options: P04.DKS,
            desc: 'Step 04 的随机采样演示：两个 dₖ 维标准正态向量的点积。dₖ 越大，不缩放的点积越大、softmax 越接近 one-hot；除以 √dₖ 后与维度无关。',
        },
        {
            id: 'temp', label: '温度', min: 0, max: 1.5, step: 0.05, value: 0,
            format: (raw) => (+raw === 0 ? '0 贪心' : (+raw).toFixed(2)),
            desc: 'Step 11 续写时的采样温度。0 = 贪心（每步取概率最大的字）；调高后按 softmax(logits / 温度) 随机采样，越高越「放飞」。',
        },
    ],

    stages: [
        {
            id: 'load', lifetime: 'resident', run(S) {
                const cfg = LM_WEIGHTS.base.cfg;
                S.mBase = LM.gpt.loadWeights(LM.gpt.create(cfg, 1), LM_WEIGHTS.base.b64);
                S.mRand = LM.gpt.create(cfg, 1);                   // 训练脚本的起点（同一个种子）
                S.ablCache = { base: {}, rand: {} };
                S.dots = P04.dotDemo(PL.rng((S.seed ^ 0x2545f491) >>> 0));
                S.pb = P04.paramBreakdown(cfg.V, cfg.T, cfg.d, cfg.L, cfg.ff);
                S.pbGPT2 = P04.paramBreakdown(50257, 1024, 768, 12, 3072);
                S._lay = {};
            },
        },
        {
            id: 'forward', lifetime: 'param', run(S) {
                const ids = LM.vocab.encode(P04.SENTS[S.sent].s);
                S.ids = ids; S.T = ids.length; S.qi = P04.clampQ(S);
                const other = S.weights === 'rand' ? 'base' : 'rand';
                S.F = P04.analyse(P04.model(S, S.weights, S.ablate), ids);              // 全页画的就是这一次前向
                S.F0 = S.ablate === 'none' ? S.F : P04.analyse(P04.model(S, S.weights, 'none'), ids, true);   // 消融前（对照）
                S.FO = P04.analyse(P04.model(S, other, 'none'), ids, true);             // 另一套权重（对照）
                S.onl = P04.online(S.F, S.layer, S.head, S.qi, 4);
            },
        },
        { id: 'ablscan', lifetime: 'param', lazy: true, run(S) { S.abl = P04.ablScan(S); } },
        { id: 'generate', lifetime: 'param', lazy: true, run(S) { S.gen = P04.generate(S); } },
    ],

    intro: {
        title: '一次前向传播：模型凭什么在「熊猫喜欢吃竹子，所以熊猫每天都吃」后面写出「竹」？',
        flow: ['字 → token id', 'W_te + W_pe', 'LayerNorm', 'Q · K · V', 'QKᵀ/√dₕ + 掩码 → softmax', 'A·V → W_O', '+ 残差', 'LayerNorm → FFN', '× 2 层', 'ln_f → W_teᵀ', 'softmax → 下一个字'],
        html: LM.navHtml('07') + `
            <p>第 06 篇把文字变成了向量；这一篇把这些向量送进一个<strong>真的训练过的</strong>迷你 GPT，逐层拆开一次前向传播（forward pass）：从 token id 出发，经过 2 个 Transformer Block（每个 = 多头注意力 + 前馈网络 FFN，外面包着 LayerNorm 和残差连接），最后得到「下一个字」的概率分布。每张预览都是页面里的 JS 实现对所选句子<strong>实时算出的真实张量</strong>；代码块是同一个模型的 C++ 写法（C++17 + 标准库，调用本系列自带的小库 <code>pages/cpp/minml_*.h</code>，源码在「数据流」一节的 Common 面板里；代码里的函数名都可以悬停查看、点击跳转），结构与 GPT-2 / nanoGPT 相同。</p>
            <p>模型：字符级词表 76、上下文 T = 32、残差流宽度 d = 32、L = 2 层、每层 H = 4 个头（每头 dₕ = 8）、FFN 宽 128，共 <b>28 928</b> 个参数，在「玩具世界」语料（8 种动物 × 爱吃 / 住处 / 颜色，10 种句式）上训练了 4000 步，验证集 loss 0.507。作为对比，GPT-2 small 是 12 层、d = 768、12 个头、上下文 1024，约 1.24 亿参数——结构一样，只是每个维度大几十倍。</p>
            <p>主线问题：读到「熊猫喜欢吃竹子，所以熊猫每天都吃」时，模型以 99.8% 的把握说下一个字是「竹」。它是把前文的「竹」<em>抄</em>过来，还是认出主语是熊猫、再从权重里<em>回忆</em>出熊猫吃竹子？Step 05（注意力图）、06（逐个消融）、07（logit lens）、08（FFN 神经元）一起给出答案。</p>
            <p>用法：顶栏选<b>输入句子</b>和<b>查询位置 i</b>（很多图上也能直接点击）、<b>层</b>和<b>头</b>；<b>权重</b>切到「随机初始化」看同一个网络训练前的样子；<b>消融</b>把一个头或一个 MLP 的输出置零，看预测怎么变。页面上的层号、头号从 1 数（L2·H1 = 第 2 层第 1 个头），C++ 代码里对应 <code>m.blocks[1]</code> 和头 0。</p>`,
    },

    steps: [
        // ───────────────────────────── 01 嵌入 ─────────────────────────────
        {
            id: 'embed', title: '输入：token id → W_te + W_pe → 残差流 X₀', tag: 'EMBED',
            desc: '模型只认识整数。字符级 tokenizer 把每个汉字 / 标点映射成 0–75 的 id（第 06 篇）。第一层是两张查找表：<b>token embedding</b> W_te [76 × 32] 的第 id 行回答「这个字是什么」，<b>位置 embedding</b> W_pe [32 × 32] 的第 t 行回答「它在第几个位置」。两者相加得到 X₀ [T × 32]。此后整个网络都在这 T 个 32 维向量上加工，这串向量叫<strong>残差流</strong>（residual stream）。注意力本身对顺序不敏感（打乱输入顺序，输出只是跟着打乱），位置信息全靠 W_pe 注入。',
            formula: ['X_0[t] = W_te[id_t] + W_pe[t]      t = 0 … T−1   // X_0: [T × 32]', 'W_te[id] = onehot(id) · W_te   // 查表 = one-hot 乘矩阵，只是不用真的乘'],
            code: {
                title: 'C++ · embed：token + 位置 embedding', src: `
                // X₀[t] = W_te[id_t] + W_pe[t]：两次查表、一次加法，没有乘法（库里 gptForward 的第一行做的就是这个）
                Var embed(const GPT& m, const std::vector<int>& ids) {
                    int T = (int)ids.size();
                    std::vector<int> pos = ints(T);
                    for (int t = 0; t < T; t++) pos[t] = t;
                    Var tok = embedding(ids, m.wte);                 // [T × 32]：W_te 的第 id 行 ——「这个字是什么」
                    Var wpe = gatherRows(m.wpe, pos);                // [T × 32]：W_pe 的第 t 行 ——「它在第几个位置」
                    return add(tok, wpe);                            // X₀ [T × 32]：残差流的起点
                }

                // 本页的输入：m = loadModel(true)（训练后）或 loadModel(false)（随机初始化）；一句话 → T 个 id（「字 → id」小图）→ X₀
                Var embedSentence(const GPT& m) {
                    std::vector<int> ids = encode(makeVocab(), "熊猫喜欢吃竹子，所以熊猫每天都吃竹子。");   // T = 19 个整数
                    NoGrad ng;                                       // 只做前向分析，不记录反向传播用的 tape
                    return embed(m, ids);
                }

                // 右图：32 个位置向量两两的余弦相似度 cos(W_pe[t], W_pe[s])
                Tensor positionCosine(const GPT& m) {
                    const Tensor& P = m.wpe->val;                    // [32 × 32]
                    int n = P.rows, d = P.cols;
                    std::vector<float> norm = floats(n);
                    for (int t = 0; t < n; t++) {
                        float s = 0;
                        for (int j = 0; j < d; j++) s += P.at(t, j) * P.at(t, j);
                        norm[t] = std::sqrt(s);
                    }
                    Tensor c = zeros(n, n);
                    for (int t = 0; t < n; t++)
                        for (int s = 0; s < n; s++) {
                            float z = 0;
                            for (int j = 0; j < d; j++) z += P.at(t, j) * P.at(s, j);
                            c.at(t, s) = z / (norm[t] * norm[s]);
                        }
                    return c;
                }`,
            },
            previews: [
                {
                    id: 'embsum', label: 'X₀ = 词向量 + 位置向量', sub: '每行一个 token、每列一维 · 色阶对称（蓝负橙正）· <b>点击行选查询位置</b>', large: true, size: [480, 300],
                    render: (c, S) => {
                        const T = S.T, F = S.F;
                        S._lay.embsum = P04.panels(c, {
                            rows: P04.posLabels(S.ids), hiRow: S.qi, panels: [
                                { data: F.tok, C: P04.D, title: 'W_te[id]', symmetric: true },
                                { data: F.pos, C: P04.D, title: 'W_pe[t]', symmetric: true, op: '+' },
                                { data: F.x0, C: P04.D, title: 'X₀', symmetric: true, op: '=' },
                            ],
                        });
                        void T;
                    },
                    probe: (x, y, S, PL, info) => { const L = S._lay.embsum; if (!L) return ''; for (let k = 0; k < 3; k++) { const h = P04.hitLay(L[k], info); if (h) { const a = [S.F.tok, S.F.pos, S.F.x0][k][h.r * P04.D + h.c]; return `${P04.show(S.ids[h.r])}（位置 ${h.r}）· 维度 ${h.c}\n${['W_te', 'W_pe', 'X₀'][k]} = ${a.toFixed(3)}`; } } return ''; },
                    pick: (x, y, S, PL, info) => { const L = S._lay.embsum; if (!L) return; for (let k = 0; k < 3; k++) { const h = P04.hitLay(L[k], info); if (h) { PL.setParam('query', h.r); return; } } },
                },
                {
                    id: 'tokstrip', label: '字 → id', sub: '上：位置 t · 中：token · 下：词表 id · <b>点击选查询位置</b>', size: [360, 120],
                    render: (c, S) => {
                        const ids = Array.from(S.ids);
                        S._lay.tokstrip = P04.charGrid(c, {
                            rows: [
                                { label: 't', cells: ids.map((_, i) => ({ t: i, tc: i === S.qi ? LM.C.yellow : LM.C.gray, px: 11 })) },
                                { label: 'token', cells: ids.map((id) => ({ t: P04.show(id), color: LM.tokColor(id) })) },
                                { label: 'id', cells: ids.map((id) => ({ t: id, tc: LM.C.text, px: 11 })) },
                            ], hiCol: S.qi, cwMax: 30, chMax: 30,
                        });
                    },
                    pick: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.tokstrip, info); if (h) PL.setParam('query', h.c); },
                },
                {
                    id: 'wpesim', label: '位置向量之间的余弦相似度', sub: 'cos(W_pe[t], W_pe[s])，t, s = 0…31 · 黄框 = 本句用到的前 T 个位置', size: [360, 320],
                    render: (c, S) => {
                        const m = S.weights === 'rand' ? S.mRand : S.mBase, P = m.wpe.data, n = P04.TMAX, d = P04.D, out = new Float32Array(n * n), nr = new Float32Array(n);
                        for (let t = 0; t < n; t++) { let s = 0; for (let j = 0; j < d; j++) s += P[t * d + j] ** 2; nr[t] = Math.sqrt(s); }
                        for (let t = 0; t < n; t++) for (let s = 0; s < n; s++) { let z = 0; for (let j = 0; j < d; j++) z += P[t * d + j] * P[s * d + j]; out[t * n + s] = z / (nr[t] * nr[s]); }
                        const lab = Array.from({ length: n }, (_, t) => (t % 4 === 0 ? String(t) : ''));
                        const L = P04.heat(c, out, n, n, { rows: lab, cols: lab, range: [-1, 1], map: 'diverge' });
                        L.ctx.strokeStyle = LM.C.yellow; L.ctx.lineWidth = 1.5; L.ctx.strokeRect(L.x0, L.y0, S.T * L.cw, S.T * L.ch);
                        S._lay.wpesim = L; S._wpesim = out;
                    },
                    probe: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.wpesim, info); return h ? `cos(W_pe[${h.r}], W_pe[${h.c}]) = ${S._wpesim[h.r * P04.TMAX + h.c].toFixed(2)}` : ''; },
                },
            ],
            notes: '训练后的 W_te 每个字的向量范数约 1.3，W_pe 只有 0.1–0.55：残差流开头主要由「是什么字」决定。右图里相邻位置的位置向量高度相似（对角线附近一片橙），位置 0 和位置 20 以后几乎反向——模型自己学出了一条平滑的「从前到后」坐标轴。把 <b>权重</b> 切到随机初始化：三张热图只剩幅度 0.1 量级的噪声，位置相似度也没有任何结构。',
            warn: 'W_te 还会在最后一步再用一次：输出层把隐状态乘以 W_teᵀ 得到每个字的 logit（权重共享，Step 09）。所以 W_te 的每一行既是「这个字的输入表示」，也是「输出这个字的模板」。',
        },
        // ───────────────────────────── 02 LayerNorm ─────────────────────────────
        {
            id: 'ln', title: 'LayerNorm（Pre-LN）：每个 token 自己归一化', tag: 'NORM',
            desc: '每个子层（注意力、FFN）读残差流之前先做 LayerNorm：<b>对每个 token 自己的 32 个数</b>减均值、除标准差，再逐维乘可学习的 γ、加 β。它不跨 token、也不跨 batch（这点和 BatchNorm 不同），所以推理时单个 token 也能算。作用是不管残差流在第几层、长到多大，送进子层的输入尺度都差不多，训练更稳。GPT-2 起的做法是 <strong>Pre-LN</strong>：LN 放在子层<em>入口</em>，残差流本身从不被归一化，最后一个 Block 之后再加一个 ln_f。原始 Transformer 是 Post-LN（先相加再归一化），层数一多就很难训。',
            formula: ['μ = (1/d) Σ_j x_j      σ^2 = (1/d) Σ_j (x_j − μ)^2   // 每个 token（每一行）单独算', 'LN(x) = γ ⊙ (x − μ) / √(σ^2 + ε) + β      ε = 10^{−5}'],
            code: {
                title: 'C++ · rowStats / layerNormTokens（与库里的 layerNorm 同一个公式）', src: `
                // 每个 token（每一行）自己的均值 μ 与标准差 σ（左图：橙柱 = 进 LN 前的 x，蓝点 = LN 的输出 h）
                struct RowStats { std::vector<float> mu, sd; };

                RowStats rowStats(const Tensor& x) {
                    int T = x.rows, d = x.cols;
                    RowStats s;
                    s.mu = floats(T);
                    s.sd = floats(T);
                    for (int t = 0; t < T; t++) {
                        float mu = 0, var = 0;
                        for (int j = 0; j < d; j++) mu += x.at(t, j);
                        mu /= d;
                        for (int j = 0; j < d; j++) var += (x.at(t, j) - mu) * (x.at(t, j) - mu);
                        s.mu[t] = mu;
                        s.sd[t] = std::sqrt(var / d);                // 有偏方差：除以 d，不是 d − 1
                    }
                    return s;
                }

                // LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β，ε = 1e-5：每一行（每个 token）单独做，行与行互不相干，所以单个 token 也能算。
                // 一行的计算就是库里的 layerNormRow；整块、带反向传播的版本是 layerNorm(x, γ, β)，两者结果相同
                Tensor layerNormTokens(const Tensor& x, const Var& gamma, const Var& beta) {
                    Tensor h = zeros(x.rows, x.cols);
                    for (int t = 0; t < x.rows; t++)
                        layerNormRow(&x.v[(size_t)t * x.cols], gamma, beta, x.cols, &h.v[(size_t)t * x.cols]);
                    return h;
                }

                // 第 1 层（页面上的 L1）的 LN₁：残差流 x = X₀ → h₁，真正送进注意力的输入。L2 是 m.blocks[1]
                Var ln1Of(const GPT& m, const Var& x0) {
                    const Block& b = m.blocks[0];
                    Var h1 = layerNorm(x0, b.ln1g, b.ln1b);          // γ = b.ln1g、β = b.ln1b（右下图），初始化为 1、0
                    RowStats before = rowStats(x0->val);             // σ ≈ 0.2
                    RowStats after = rowStats(h1->val);              // 标准化后 σ = 1、μ = 0，再被 γ、β 重新拉伸
                    return h1;
                }`,
            },
            previews: [
                {
                    id: 'lnstats', label: 'LN 前后每个 token 的标准差与均值', sub: '橙柱 = 进 LN 前的残差流 · 灰虚线 = 标准化后 (x − μ)/σ · 蓝点 = 乘 γ 加 β 之后（真正送进注意力的 h）', large: true, size: [480, 320],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer], T = S.T, lab = P04.chars(S.ids);
                        const { ctx, w, h } = PL.draw.ctx(c);
                        const sdMax = Math.max(1.2, ...Ly.sIn.sd, ...Ly.sOut.sd) * 1.8;
                        let mlo = Math.min(-0.1, ...Ly.sIn.mu, ...Ly.sOut.mu), mhi = Math.max(0.1, ...Ly.sIn.mu, ...Ly.sOut.mu); const pad = (mhi - mlo) * 0.12; mlo -= pad; mhi += pad;
                        S._lay.lnstats = P04.catPlot(c, { ctx, rect: { x: 0, y: 0, w, h: h * 0.55 }, title: `σ（L${S.layer + 1} 的 LN₁）`, labels: lab, y: [0, sdMax], hiIdx: S.qi, series: [{ data: Ly.sIn.sd, kind: 'bar', color: LM.C.orange, label: 'σ(x)：进 LN 前' }, { data: Ly.sOut.sd, kind: 'dot', color: LM.C.blue, label: 'σ(γ⊙x̂+β)：LN 输出' }, { data: [], kind: 'line', dash: [5, 4], color: LM.C.gray, label: 'x̂ = (x−μ)/σ：σ=1, μ=0' }], hlines: [{ y: 1, color: LM.C.gray }] });
                        P04.catPlot(c, { ctx, rect: { x: 0, y: h * 0.55, w, h: h * 0.45 }, title: 'μ', labels: lab, y: [mlo, mhi], hiIdx: S.qi, series: [{ data: Ly.sIn.mu, kind: 'bar', color: LM.C.orange }, { data: Ly.sOut.mu, kind: 'dot', color: LM.C.blue }], hlines: [{ y: 0, color: LM.C.gray }] });
                    },
                    probe: (x, y, S, PL, info) => { const i = P04.hitCat(S._lay.lnstats, info); if (i < 0) return ''; const Ly = S.F.layers[S.layer]; return `${P04.show(S.ids[i])}（位置 ${i}）\n进 LN 前：μ = ${Ly.sIn.mu[i].toFixed(3)}, σ = ${Ly.sIn.sd[i].toFixed(3)}\n输出 h：μ = ${Ly.sOut.mu[i].toFixed(3)}, σ = ${Ly.sOut.sd[i].toFixed(3)}`; },
                    pick: (x, y, S, PL, info) => { const i = P04.hitCat(S._lay.lnstats, info); if (i >= 0) PL.setParam('query', i); },
                },
                {
                    id: 'lnheat', label: 'LN₁ 的输入与输出', sub: '左 = 残差流 x · 右 = h = LN(x) · 色阶各自对称', size: [360, 280],
                    render: (c, S) => { const Ly = S.F.layers[S.layer]; P04.panels(c, { rows: P04.posLabels(S.ids), hiRow: S.qi, panels: [{ data: Ly.xin, C: P04.D, title: `x（L${S.layer + 1} 入口）`, symmetric: true }, { data: Ly.h1, C: P04.D, title: 'h = LN(x)', symmetric: true, op: '→' }] }); },
                },
                {
                    id: 'lngb', label: 'LN₁ 的可学习参数 γ、β', sub: '32 维 · 初始化 γ = 1、β = 0', size: [360, 190],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer], xs = Array.from({ length: P04.D }, (_, j) => j);
                        PL.draw.plot(c, { x: [0, P04.D - 1], xLabel: '维度 j', series: [{ data: Array.from(Ly.g), x: xs, color: LM.C.purple, label: 'γ', width: 2 }, { data: Array.from(Ly.b), x: xs, color: LM.C.cyan, label: 'β', width: 2 }], hmarks: [{ y: 1, color: '#555' }, { y: 0, color: '#555' }] });
                    },
                },
            ],
            notes: '拖 <b>层</b>：L1 的输入就是 X₀（σ ≈ 0.2），L2 的输入已经加过 L1 两个子层的输出，σ 更大、各 token 差别也更大；LN 之后又回到同一尺度。灰虚线说明标准化本身让每个 token 的 σ 恰好为 1、μ 恰好为 0；蓝点偏离它们，是因为训练后的 γ、β 偏离了初始值 1、0（见右下图）——模型确实在用这两组参数重新拉伸各维。',
            warn: '方差用的是<b>有偏</b>估计（除以 d 而不是 d − 1），与 PyTorch 的 <code>nn.LayerNorm</code> 等框架实现一致；用统计库里默认无偏的方差函数（如 <code>torch.var</code>）手写 LN 时要特别注意。LLaMA 等现代模型改用 RMSNorm：不减均值、没有 β，见 Step 12。',
        },
        // ───────────────────────────── 03 QKV ─────────────────────────────
        {
            id: 'qkv', title: 'Q / K / V 投影：每个位置提出问题、亮出标签、准备内容', tag: 'ATTN',
            desc: '注意力的直觉：每个位置向前文「提一个问题」，前文每个位置「亮出一个标签」，按匹配程度把它们的「内容」汇总过来。三个线性层把 LN 后的 h 变成三份向量：<b>Query</b>（我在找什么）、<b>Key</b>（我是什么、可以被谁找到）、<b>Value</b>（被找到时交出去的内容）。32 维被切成 H = 4 段，每段 dₕ = 8 维就是一个<strong>头</strong>（head）：4 个头各自独立地做一遍注意力，可以同时关注不同的东西——Step 05 会看到它们确实分了工。',
            formula: ['Q = h · W_Q + b_Q      K = h · W_K + b_K      V = h · W_V + b_V   // 各 [T × 32]', 'Q_h = Q[:, 8h : 8h + 8]   // 头 h 只用自己那 8 列：[T × 8]，K_h、V_h 同理'],
            code: {
                title: 'C++ · projectQKV / headCols：投影并拆成 4 个头', src: `
                // 三个线性层：Q = h·W_Q + b_Q，K = h·W_K + b_K，V = h·W_V + b_V，各 [T × 32]（W 存成 [in × out]，算 x·W）
                struct QKV { Var q, k, v; };

                QKV projectQKV(const Block& b, const Var& h) {
                    QKV r;
                    r.q = linear(h, b.Wq, b.bq);                     // Query：我在找什么
                    r.k = linear(h, b.Wk, b.bk);                     // Key：我是什么、可以被谁找到
                    r.v = linear(h, b.Wv, b.bv);                     // Value：被找到时交出去的内容
                    return r;
                }

                // 拆头：头 h 只用自己那 dh = 8 列 —— 行主序的 [T × 32] 里就是第 h·dh … h·dh+dh−1 列（右图白竖线隔开的 4 段）
                Tensor headCols(const Tensor& X, int h, int dh) {
                    Tensor out = zeros(X.rows, dh);
                    for (int t = 0; t < X.rows; t++)
                        for (int e = 0; e < dh; e++) out.at(t, e) = X.at(t, h * dh + e);
                    return out;                                      // [T × dh]：左图的 Q_h / K_h / V_h
                }

                // 页面上的 L2·H1 = m.blocks[1] 的头 0（h1 = L2 的 LN₁ 输出）
                Tensor queryL2H1(const GPT& m, const Var& h1) {
                    QKV qkv = projectQKV(m.blocks[1], h1);
                    int dh = m.cfg.d / m.cfg.H;                      // 32 / 4 = 8
                    return headCols(qkv.q->val, 0, dh);              // Q_h：[T × 8]；K_h、V_h 同理
                }`,
            },
            previews: [
                {
                    id: 'qkvhead', label: '选中头的 Q_h、K_h、V_h', sub: '[T × 8] ×3 · 黄框 = 查询位置 i', large: true, size: [360, 320],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer], T = S.T, h = S.head, D = P04.D, DH = P04.DH;
                        const slice = (a) => { const o = new Float32Array(T * DH); for (let i = 0; i < T; i++) for (let e = 0; e < DH; e++) o[i * DH + e] = a[i * D + h * DH + e]; return o; };
                        const t = P04.HL(S.layer, h);
                        S._lay.qkvhead = P04.panels(c, { rows: P04.posLabels(S.ids), hiRow: S.qi, panels: [{ data: slice(Ly.q), C: DH, title: 'Q', symmetric: true }, { data: slice(Ly.k), C: DH, title: 'K', symmetric: true }, { data: slice(Ly.v), C: DH, title: 'V', symmetric: true }] });
                        const ctx = S._lay.qkvhead.ctx; ctx.font = LM.FONT(11, true); ctx.fillStyle = LM.C.yellow; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(t, S._lay.qkvhead.w - 4, 3);
                    },
                    probe: (x, y, S, PL, info) => { const L = S._lay.qkvhead; if (!L) return ''; for (let k = 0; k < 3; k++) { const hh = P04.hitLay(L[k], info); if (hh) { const a = [S.F.layers[S.layer].q, S.F.layers[S.layer].k, S.F.layers[S.layer].v][k]; return `${'QKV'[k]}[${hh.r}, ${hh.c}] = ${a[hh.r * P04.D + S.head * P04.DH + hh.c].toFixed(3)}（${P04.show(S.ids[hh.r])}）`; } } return ''; },
                    pick: (x, y, S, PL, info) => { const L = S._lay.qkvhead; if (!L) return; for (let k = 0; k < 3; k++) { const hh = P04.hitLay(L[k], info); if (hh) { PL.setParam('query', hh.r); return; } } },
                },
                {
                    id: 'qkvall', label: '整层的 Q、K、V', sub: '[T × 32] ×3 · 白竖线 = 头的分界 · 黄框 = 选中的头', size: [480, 260],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer], a = S.head * P04.DH, hc = [a, a + P04.DH];
                        P04.panels(c, { rows: P04.posLabels(S.ids), hiRow: S.qi, panels: [{ data: Ly.q, C: P04.D, title: 'Q', symmetric: true, vsep: P04.DH, hiCols: hc }, { data: Ly.k, C: P04.D, title: 'K', symmetric: true, vsep: P04.DH, hiCols: hc }, { data: Ly.v, C: P04.D, title: 'V', symmetric: true, vsep: P04.DH, hiCols: hc }] });
                    },
                },
            ],
            notes: '三张图看起来像「噪声」很正常：单个向量的含义要等到 Q 和 K 做点积（下一步）才显现。可以留意同一个字在两个位置的那两行（两个「熊」、两个「吃」…）：它们的 key 明显更像（默认句子里余弦约 0.5–0.9，不同字之间平均只有 0.05），但并不相同——位置向量和（到了 L2）上下文也掺了进来。拖 <b>头</b> 切换：每个头用的是 Q、K、V 里不同的 8 列。',
            warn: '本页的 C++ 库和 JS 都把权重存成 [in × out]、算 x·W；PyTorch 的 <code>nn.Linear</code> 存 [out × in]、算 x·Wᵀ，数值相同——读别的框架的代码或权重文件时注意这个转置。nanoGPT 把三个投影合成一个 <code>c_attn</code>（[d × 3d]），一次 GEMM 算完再切开——本页为了清楚拆成三个，数学上完全等价。',
        },
        // ───────────────────────────── 04 打分 ─────────────────────────────
        {
            id: 'scores', title: '打分 QKᵀ/√dₕ：每个位置对每个位置有多「感兴趣」', tag: 'ATTN',
            desc: '在头 h 里，位置 i 的 query 和位置 j 的 key 做点积，得到打分 S_ij；一次矩阵乘 Q_h K_hᵀ 就算出全部 T × T 个。为什么要除以 √dₕ？如果 q、k 的各分量独立、均值 0、方差 1，点积 q·k 的方差就是 dₕ——维度越高，打分的绝对值越大，softmax 越接近 one-hot，几乎所有位置的梯度都趋于 0。除以 √dₕ 把方差拉回 1：不管头有多宽，softmax 的「锐度」都差不多（Vaswani et al. 2017 原文的解释）。另外两张图用真实的随机采样演示这一点。',
            formula: ['S = Q_h K_h^T / √d_h      S_ij = (q_i · k_j) / √d_h   // [T × T]，d_h = 8', 'q, k ~ N(0, I_{d_k}):  E[q · k] = 0,  Var[q · k] = d_k  ⇒  Var[q · k / √d_k] = 1'],
            code: {
                title: 'C++ · attnScores / dotStd', src: `
                // 头 h 的打分 S = Q_h K_hᵀ / √dh：[T × T]，S[i][j] = q_i · k_j / √dh。Q、K 是整层的 [T × 32]，头 h 用自己那 dh 列。
                // 上三角 j > i 也算了（主图里变暗的部分），下一步的因果掩码会把它们去掉
                Tensor attnScores(const Tensor& Q, const Tensor& K, int h, int dh) {
                    int T = Q.rows;
                    float s = 1.f / std::sqrt((float)dh);            // dh = 8：除以 √8
                    Tensor S = zeros(T, T);
                    for (int i = 0; i < T; i++)
                        for (int j = 0; j < T; j++) {
                            float dot = 0;
                            for (int c = 0; c < dh; c++) dot += Q.at(i, h * dh + c) * K.at(j, h * dh + c);
                            S.at(i, j) = dot * s;
                        }
                    return S;
                }

                // 为什么除以 √dₖ（右上图）：n 对 q、k ~ N(0, I)，点积的标准差 ≈ √dₖ；除以 √dₖ 后 ≈ 1，与维度无关
                struct DotStd { float raw, scaled; };

                DotStd dotStd(int dk, int n, Rng& r) {
                    std::vector<float> d = floats(n);
                    for (int k = 0; k < n; k++) {
                        float s = 0;
                        for (int c = 0; c < dk; c++) s += randn(r) * randn(r);   // q_c · k_c，两者独立 ~ N(0, 1)
                        d[k] = s;
                    }
                    double mean = 0, var = 0;
                    for (float x : d) mean += x / n;
                    for (float x : d) var += (x - mean) * (x - mean) / n;
                    DotStd out;
                    out.raw = (float)std::sqrt(var);                 // dₖ = 64 时 ≈ 8，dₖ = 512 时 ≈ 22.6
                    out.scaled = out.raw / std::sqrt((float)dk);     // ≈ 1
                    return out;
                }

                // 右下图：一个随机 query 对 16 个随机 key 的打分。softmaxInPlace(s, 1) = 不缩放；softmaxInPlace(s, √dₖ) = 除以 √dₖ
                std::vector<float> randomScores(int dk, int nKeys, Rng& r) {
                    std::vector<float> q = floats(dk), s = floats(nKeys);
                    for (float& x : q) x = randn(r);
                    for (int j = 0; j < nKeys; j++) {
                        float z = 0;
                        for (int c = 0; c < dk; c++) z += q[c] * randn(r);
                        s[j] = z;
                    }
                    return s;
                }`,
            },
            previews: [
                {
                    id: 'scores', label: '选中头的打分矩阵 S', sub: '行 = 查询 i · 列 = 键 j · 上三角（j > i）也算了，但下一步会被掩掉（变暗）· <b>点击选行</b>', large: true, size: [360, 360],
                    render: (c, S) => {
                        const T = S.T, Ly = S.F.layers[S.layer], m = Ly.sc.subarray(S.head * T * T, (S.head + 1) * T * T);
                        S._lay.scores = P04.heat(c, m, T, T, { rows: P04.posLabels(S.ids), cols: P04.chars(S.ids), symmetric: true, hiRow: S.qi, dim: (r, cc) => (cc > r ? 0.28 : 1), title: `${P04.HL(S.layer, S.head)} · S = QKᵀ/√8` });
                    },
                    probe: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.scores, info); if (!h) return ''; const T = S.T, v = S.F.layers[S.layer].sc[(S.head * T + h.r) * T + h.c]; return `S[${h.r}, ${h.c}] = ${v.toFixed(3)}\n查询 ${P04.show(S.ids[h.r])} · 键 ${P04.show(S.ids[h.c])}${h.c > h.r ? '（将被掩掉）' : ''}`; },
                    pick: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.scores, info); if (h) PL.setParam('query', h.r); },
                },
                {
                    id: 'dothist', label: '随机向量点积的分布', sub: '1000 对 q、k ~ N(0, I) 的真实采样 · 橙 = q·k · 蓝 = q·k/√dₖ', size: [360, 220],
                    render: (c, S) => {
                        const dk = S.dk, d = S.dots[dk], R = 4 * Math.sqrt(dk), nb = 60;
                        const a = P04.hist(d.raw, -R, R, nb), b = P04.hist(d.raw, -R, R, nb, 1 / Math.sqrt(dk));
                        const ymax = Math.max(...b.c) * 1.1;
                        PL.draw.plot(c, { x: [-R, R], y: [0, ymax], xLabel: `点积的值（dₖ = ${dk}）`, series: [{ data: Array.from(a.c), x: a.x, color: LM.C.orange, bar: true, label: `q·k：std ${d.std.toFixed(1)} ≈ √${dk} = ${Math.sqrt(dk).toFixed(1)}` }, { data: Array.from(b.c), x: b.x, color: LM.C.blue, bar: true, label: `q·k/√dₖ：std ${(d.std / Math.sqrt(dk)).toFixed(2)}` }] });
                    },
                },
                {
                    id: 'dotsoft', label: '同一行打分的 softmax：不缩放 vs 缩放', sub: '一个随机 query 对 16 个随机 key · 灰虚线 = 均匀分布 1/16 · 不缩放时 dₖ 越大越接近 one-hot', size: [360, 220],
                    render: (c, S) => {
                        const d = S.dots[S.dk], raw = d.sc, p1 = LM.softmax(raw), p2 = LM.softmax(raw, null, Math.sqrt(S.dk));
                        const lab = Array.from({ length: 16 }, (_, j) => 'k' + j);
                        const H1 = (p) => -Array.from(p).reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0);
                        P04.catPlot(c, { labels: lab.map((_, j) => j), y: [0, 1.32], series: [{ data: Array.from(p1), kind: 'bar', color: LM.C.orange, label: `softmax(q·k)：max ${Math.max(...p1).toFixed(2)}，熵 ${H1(p1).toFixed(2)}` }, { data: Array.from(p2), kind: 'bar', color: LM.C.blue, label: `softmax(q·k/√dₖ)：max ${Math.max(...p2).toFixed(2)}，熵 ${H1(p2).toFixed(2)}` }], hlines: [{ y: 1 / 16, color: LM.C.gray }] });
                    },
                },
            ],
            notes: '主图里最亮（最正）的格子就是这个头最想看的地方，下一步 softmax 会把它们放大成权重。拖 <b>点积维度 dₖ</b> 到 512：不缩放的点积标准差约 22.6，softmax 几乎把全部概率压在一个 key 上（熵 → 0）；缩放后无论 dₖ 多大都是标准差 1 的钟形。dₖ = 64、128 分别是 GPT-2 和 LLaMA 的头宽；本页模型是 8。',
            warn: '训练后的 q、k 并不是独立标准正态，打分的尺度是学出来的（本模型各头 S 的标准差约 0.8–2.8）。√dₖ 只保证<em>初始化时</em> softmax 不饱和；训练中模型可以自己把打分放大，让注意力变得很尖锐。',
        },
        // ───────────────────────────── 05 掩码 + softmax ─────────────────────────────
        {
            id: 'attn', title: '因果掩码 + softmax：注意力权重，以及训练后各个头的分工', tag: 'ATTN',
            desc: `语言模型是<strong>因果</strong>（causal）的：位置 i 预测第 i+1 个字时只能看 0…i，否则训练时就是在偷看答案。做法是把 j > i 的打分设成 −∞，softmax 后权重恰好为 0；每行剩下的权重非负、和为 1，表示「位置 i 把注意力怎样分给 0…i」。这张 T×T 的矩阵就是常说的<strong>注意力图</strong>。在 Node 里对 8 种动物逐一测过，训练后的模型里几个头分工很清楚：
                <ul>
                <li><b>L2·H1、L2·H2 是「主语头」</b>：在「吃」「住在」这类下一个字是属性的位置，L2·H1 盯住主语名字的<b>第 2 个字</b>（猫 / 狗 / 虎 / 子…，平均 86–98%），L2·H2 主要看<b>第 1 个字</b>（熊 / 小 / 老…）。名字的第 1 个字「小」区分不了 5 种动物，所以 H1 盯第 2 个字更有用。默认句子里，i = 15（第 2 个「吃」）处 L2·H1 把 86% 给了「猫」、L2·H2 把 75% 给了「熊」；投向前文「竹」的权重在所有头里都不超过 11%。</li>
                <li><b>L1·H4 看「食物」</b>：在「所以」的「以」（下一个字是名字）处，L1·H4 把约 86% 的注意力放在前文的食物上——在这个世界里食物唯一对应一种动物，于是模型靠食物推出名字。</li>
                <li>其余头的模式比较分散。这个 2 层 × 4 头、只见过 10 种句式的小模型里，<b>没有</b>干净的「前一个 token」头，也没有大模型里那种经典的「归纳头」（induction head，Olsson et al. 2022）。</li>
                </ul>`,
            formula: ['M_ij = 0  (j ≤ i),   −∞  (j > i)', 'A = softmax(S + M)   // 按行：A_ij ≥ 0，Σ_j A_ij = 1，掩掉处恰为 0'],
            code: {
                title: 'C++ · causalSoftmax：因果掩码 + softmax', src: `
                // 因果掩码 + 按行 softmax：A = softmax(S + M)，M[i][j] = −∞（j > i）。
                // 不必真的加 −∞：只在 j ≤ i 里算就行，exp(−∞) = 0 的那些格子保持 0（库里的 attention 也是这么做的）
                Tensor causalSoftmax(const Tensor& S) {
                    int T = S.rows;
                    Tensor A = zeros(T, T);
                    for (int i = 0; i < T; i++) {
                        float m = S.at(i, 0);
                        for (int j = 1; j <= i; j++) m = std::max(m, S.at(i, j));   // 先减最大值再 exp，防止溢出
                        float z = 0;
                        for (int j = 0; j <= i; j++) {
                            A.at(i, j) = std::exp(S.at(i, j) - m);
                            z += A.at(i, j);
                        }
                        for (int j = 0; j <= i; j++) A.at(i, j) /= z;               // A[i][0…i] ≥ 0，和为 1
                    }
                    return A;
                }

                // 绿框：每一行权重最大的位置 j ≤ i —— 这个头在位置 i「最想看」谁
                std::vector<int> attnArgmax(const Tensor& A) {
                    std::vector<int> best = ints(A.rows);
                    for (int i = 0; i < A.rows; i++)
                        for (int j = 1; j <= i; j++)
                            if (A.at(i, j) > A.at(i, best[i])) best[i] = j;
                    return best;
                }`,
            },
            previews: [
                {
                    id: 'attn', label: '选中头的注意力权重 A', sub: '斜纹 = 被因果掩码遮住 · 绿框 = 每行权重最大处 · 黄框 = 查询行 i · <b>点击选行</b>', large: true, size: [360, 360],
                    render: (c, S) => {
                        const T = S.T, m = S.F.layers[S.layer].att.subarray(S.head * T * T, (S.head + 1) * T * T), hc = [];
                        for (let i = 0; i < T; i++) { let b = 0; for (let j = 1; j <= i; j++) if (m[i * T + j] > m[i * T + b]) b = j; hc.push([i, b, i === S.qi ? LM.C.green : 'rgba(90,208,122,0.55)']); }
                        S._lay.attn = P04.heat(c, m, T, T, { rows: P04.posLabels(S.ids), cols: P04.chars(S.ids), range: [0, 1], hiRow: S.qi, hiCells: hc, title: P04.HL(S.layer, S.head) + ' · A = softmax(S + M)' });
                    },
                    probe: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.attn, info); if (!h) return ''; const T = S.T, v = S.F.layers[S.layer].att[(S.head * T + h.r) * T + h.c]; return h.c > h.r ? '被掩码遮住（j > i）' : `A[${h.r}, ${h.c}] = ${v.toFixed(3)}\n${P04.show(S.ids[h.r])}（i = ${h.r}）看 ${P04.show(S.ids[h.c])}（j = ${h.c}）`; },
                    pick: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.attn, info); if (h) PL.setParam('query', h.r); },
                },
                {
                    id: 'attn4', label: '同一层的 4 个头', sub: '黄框 = 选中的头 · 右上 = 查询行 i 的最大权重 · <b>点击切换头</b>', size: [360, 360],
                    render: (c, S) => {
                        const T = S.T, A = S.F.layers[S.layer].att, mats = [0, 1, 2, 3].map((h) => A.subarray(h * T * T, (h + 1) * T * T));
                        const notes = mats.map((m) => { let b = 0; for (let j = 1; j <= S.qi; j++) if (m[S.qi * T + j] > m[S.qi * T + b]) b = j; return `→${P04.show(S.ids[b])} ${(m[S.qi * T + b] * 100).toFixed(0)}%`; });
                        S._lay.attn4 = P04.grid4(c, mats, T, { titles: [0, 1, 2, 3].map((h) => P04.HL(S.layer, h)), notes, sel: S.head, hiRow: S.qi });
                    },
                    probe: (x, y, S, PL, info) => { const L = S._lay.attn4; if (!L) return ''; for (let k = 0; k < 4; k++) { const h = P04.hitLay(L[k], info); if (h) { const T = S.T, v = S.F.layers[S.layer].att[(k * T + h.r) * T + h.c]; return h.c > h.r ? P04.HL(S.layer, k) + '：被掩码遮住' : `${P04.HL(S.layer, k)}：A[${h.r}, ${h.c}] = ${v.toFixed(3)}\n${P04.show(S.ids[h.r])} 看 ${P04.show(S.ids[h.c])}`; } } return ''; },
                    pick: (x, y, S, PL, info) => { const L = S._lay.attn4; if (!L) return; for (let k = 0; k < 4; k++) { const h = P04.hitLay(L[k], info); if (h) { PL.setParam('head', k); PL.setParam('query', h.r); return; } } },
                },
                {
                    id: 'attnq', label: '查询位置 i：8 个头各看向哪里', sub: '行 = 2 层 × 4 头 · 列 = 被看的位置 j ≤ i · 黄框 = 选中的头', size: [480, 220],
                    render: (c, S) => {
                        const T = S.T, i = S.qi, data = new Float32Array(8 * T).fill(NaN), rows = [], hc = [];
                        for (let l = 0; l < P04.NL; l++) for (let h = 0; h < P04.H; h++) {
                            const r = l * P04.H + h, A = S.F.layers[l].att; rows.push(P04.HL(l, h)); let b = 0;
                            for (let j = 0; j <= i; j++) { data[r * T + j] = A[(h * T + i) * T + j]; if (data[r * T + j] > data[r * T + b]) b = j; }
                            hc.push([r, b]);
                        }
                        S._lay.attnq = P04.heat(c, data, 8, T, { rows, cols: P04.chars(S.ids), range: [0, 1], square: false, hiRow: S.layer * P04.H + S.head, hiCells: hc, hiCol: i });
                    },
                    probe: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.attnq, info); if (!h || h.c > S.qi) return ''; const l = Math.floor(h.r / P04.H), hh = h.r % P04.H, T = S.T; return `${P04.HL(l, hh)}：${P04.show(S.ids[S.qi])}（i = ${S.qi}）→ ${P04.show(S.ids[h.c])}（j = ${h.c}）= ${S.F.layers[l].att[(hh * T + S.qi) * T + h.c].toFixed(3)}`; },
                    pick: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.attnq, info); if (h) { PL.setParam('layer', Math.floor(h.r / P04.H)); PL.setParam('head', h.r % P04.H); } },
                },
                {
                    id: 'attnrow', label: '选中头在查询位置 i 的那一行', sub: 'A[i, 0…i] · 和为 1', size: [360, 190],
                    render: (c, S) => {
                        const T = S.T, i = S.qi, A = S.F.layers[S.layer].att, row = [];
                        for (let j = 0; j <= i; j++) row.push(A[(S.head * T + i) * T + j]);
                        S._lay.attnrow = P04.catPlot(c, { labels: P04.chars(S.ids.slice(0, i + 1)), y: [0, 1], series: [{ data: row, kind: 'bar', color: (v) => (v === Math.max(...row) ? LM.C.green : LM.C.blue) }], hlines: [{ y: 1 / (i + 1), color: LM.C.gray, label: `均匀 1/${i + 1}`, align: 'right' }], title: `${P04.HL(S.layer, S.head)}：${P04.show(S.ids[i])}（i = ${i}）看向…` });
                    },
                    probe: (x, y, S, PL, info) => { const j = P04.hitCat(S._lay.attnrow, info); if (j < 0) return ''; const T = S.T; return `A[${S.qi}, ${j}] = ${S.F.layers[S.layer].att[(S.head * T + S.qi) * T + j].toFixed(3)}（${P04.show(S.ids[j])}）`; },
                },
            ],
            notes: '点击任一张注意力图的某一行就是选查询位置；点 4 头小图切换头。把 <b>权重</b> 切到随机初始化：所有头都变成均匀的「下三角渐变」（第 i 行每格 ≈ 1/(i+1)，见「那一行」柱状图里的灰线）——这些模式完全是训练出来的。再试 <b>输入句子</b> 的两句反事实：「老虎喜欢吃竹子，所以…」在 i = 9（以）处 L1·H4 照样去看「竹」，于是模型说下一个字是「熊」（Step 09）。两句话的句子里（小猫住在屋里。猴子…，它住在…），i = 17 处主语头看的是本句的「猴子」，而不是前一句的「小猫」。',
            warn: '实现上不必真的写 −∞：本页的 C++（<code>causalSoftmax</code>、库里的 <code>attention</code>）和 JS 都直接跳过 j > i 的项（等价）；PyTorch 等框架常用 <code>masked_fill(−inf)</code> 再整行 softmax。如果某一行全被掩掉（左侧 padding 时可能出现），softmax 会得到 NaN，生产代码要专门处理。另外「注意力看向哪里」只是相关性证据——它说明信息<em>可能</em>从哪里来，下一步的消融才是因果证据。',
        },
        // ───────────────────────────── 06 加权求和 + W_O + 消融 ─────────────────────────────
        {
            id: 'proj', title: '加权求和、多头拼接、W_O——以及用消融验证每个头的作用', tag: 'ATTN',
            desc: '有了权重，每个头输出 value 的加权平均 y_i = Σ_j A_ij v_j：注意力把信息从「被看的位置」<b>搬运</b>到查询位置。4 个头的 8 维输出拼回 32 维，乘输出投影 W_O 混合各头，再加到残差流上。W_O 可以按行切成 4 块，所以多头注意力就是各头贡献之和：O = Σ_h Y_h W_O[h]。这给了一个干净的实验手段：把 W_O 里某个头对应的 8 行置零，这个头就从模型里「消失」了——这叫<strong>消融</strong>（ablation）。下面的「逐个消融」图在查询位置把 8 个头和 2 个 MLP 逐一消融，看正确答案的概率掉了多少：这是<b>因果</b>证据，比「注意力看起来指向那里」可靠得多。',
            formula: ['Y_h = A_h · V_h   // [T × 8]：value 的加权平均', 'Y = [Y_1, Y_2, Y_3, Y_4]   // 拼接 → [T × 32]', 'O = Y · W_O + b_O = Σ_h Y_h · W_O[8h : 8h+8, :] + b_O   // 消融头 h = 把 W_O 的这 8 行置零'],
            code: {
                title: 'C++ · 加权求和 / 拼接 / 输出投影 / ablateHead / ablationScan', src: `
                // 头 h 的输出 Y_h = A_h · V_h：[T × dh]。第 i 行 = value 的加权平均 Σ_j A[i][j]·v_j ——
                // 注意力把信息从「被看的位置」搬运到查询位置。V 是整层的 [T × 32]，头 h 用自己那 dh 列
                Tensor headOutput(const Tensor& A, const Tensor& V, int h, int dh) {
                    int T = A.rows;
                    Tensor Y = zeros(T, dh);
                    for (int i = 0; i < T; i++)
                        for (int j = 0; j <= i; j++)                 // j > i 处 A = 0，跳过
                            for (int e = 0; e < dh; e++) Y.at(i, e) += A.at(i, j) * V.at(j, h * dh + e);
                    return Y;
                }

                // 多头拼接：H 个 [T × dh] 并排成 [T × H·dh]，头 h 占第 h·dh … h·dh+dh−1 列（主图 Y 里白竖线隔开的 4 段）
                Tensor concatHeads(const std::vector<Tensor>& Ys) {
                    int T = Ys[0].rows, dh = Ys[0].cols, H = (int)Ys.size();
                    Tensor Y = zeros(T, H * dh);
                    for (int h = 0; h < H; h++)
                        for (int t = 0; t < T; t++)
                            for (int e = 0; e < dh; e++) Y.at(t, h * dh + e) = Ys[h].at(t, e);
                    return Y;
                }

                // 一层注意力，把 Step 03–06 连起来逐头手算：O = [Y_1 … Y_H]·W_O + b_O = 这一层要加进残差流的增量（主图右）。
                // 结果与库里一步到位的 linear(attention(q, k, v, 1, T, H, true), W_O, b_O) 相同：attention 把打分、掩码、softmax、加权求和融合在一起
                Var attentionByHand(const Block& b, const Var& h, int H) {
                    QKV qkv = projectQKV(b, h);
                    int dh = qkv.q->val.cols / H;
                    std::vector<Tensor> Ys;
                    for (int hd = 0; hd < H; hd++) {
                        Tensor A = causalSoftmax(attnScores(qkv.q->val, qkv.k->val, hd, dh));
                        Ys.push_back(headOutput(A, qkv.v->val, hd, dh));
                    }
                    return linear(constant(concatHeads(Ys)), b.Wo, b.bo);   // W_O：混合各头
                }

                // 每个头写进残差流的量 ‖Y_h[i] · W_O[h·dh : h·dh+dh, :]‖（右上图）。W_O 按行切成 H 块：O = Σ_h Y_h·W_O[h] + b_O
                Tensor headWriteNorms(const Tensor& Y, const Var& Wo, int H) {
                    int T = Y.rows, D = Y.cols, dh = D / H;
                    const Tensor& W = Wo->val;                       // [32 × 32]，[in × out]：第 r 行乘的是 Y 的第 r 列
                    Tensor hw = zeros(T, H);
                    std::vector<float> z = floats(D);
                    for (int i = 0; i < T; i++)
                        for (int h = 0; h < H; h++) {
                            std::fill(z.begin(), z.end(), 0.f);
                            for (int e = 0; e < dh; e++)
                                for (int c = 0; c < D; c++) z[c] += Y.at(i, h * dh + e) * W.at(h * dh + e, c);
                            float ss = 0;
                            for (float x : z) ss += x * x;
                            hw.at(i, h) = std::sqrt(ss);
                        }
                    return hw;
                }

                // 消融头 (layer, head)：把 W_O 里这个头对应的 dh = 8 行置零，它写进残差流的量就恒为 0
                void ablateHead(GPT& m, int layer, int head) {
                    Tensor& W = m.blocks[layer].Wo->val;
                    int dh = m.cfg.d / m.cfg.H;
                    std::fill(W.v.begin() + head * dh * W.cols, W.v.begin() + (head + 1) * dh * W.cols, 0.f);
                }

                // 消融一整个 MLP：W₂、b₂ 置零，这一层 FFN 的输出恒为 0
                void ablateMLP(GPT& m, int layer) {
                    Block& b = m.blocks[layer];
                    std::fill(b.W2->val.v.begin(), b.W2->val.v.end(), 0.f);
                    std::fill(b.b2->val.v.begin(), b.b2->val.v.end(), 0.f);
                }

                // 逐个消融（右下图）：在查询位置 i 每次只去掉一个部件、重新前向，记录「正确的下一个字」还剩多少概率。
                // k = −1 不消融；0…7 = L(k/4+1)·H(k%4+1)；8、9 = L1、L2 的 MLP。答案 = ids[i+1]，句末没有答案时用原模型自己的 top-1
                struct AblationRow { float p; int top; float ptop; };

                std::vector<AblationRow> ablationScan(GPT& m, const std::vector<int>& ids, int i) {
                    int LH = m.cfg.L * m.cfg.H, T = (int)ids.size();
                    int target = i + 1 < T ? ids[i + 1] : argmaxIndex(nextTokenProbs(m, ids, i));
                    std::vector<AblationRow> rows;
                    for (int k = -1; k < LH + m.cfg.L; k++) {
                        GPT c = cloneGPT(m);                         // 每次从原模型复制一份再改
                        if (k >= 0 && k < LH) ablateHead(c, k / m.cfg.H, k % m.cfg.H);
                        if (k >= LH) ablateMLP(c, k - LH);
                        std::vector<float> p = nextTokenProbs(c, ids, i);
                        AblationRow r;
                        r.p = p[target];
                        r.top = argmaxIndex(p);
                        r.ptop = p[r.top];
                        rows.push_back(r);
                    }
                    return rows;
                }`,
            },
            previews: [
                {
                    id: 'yo', label: '多头输出 Y 与输出投影 O = Y·W_O + b', sub: 'Y 的白竖线 = 4 个头的分界（黄框 = 选中的头）· O 是这一层注意力要加进残差流的增量', large: true, size: [480, 300],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer], a = S.head * P04.DH;
                        P04.panels(c, { rows: P04.posLabels(S.ids), hiRow: S.qi, panels: [{ data: Ly.y, C: P04.D, title: `Y（L${S.layer + 1}，4 头拼接）`, symmetric: true, vsep: P04.DH, hiCols: [a, a + P04.DH] }, { data: Ly.ao, C: P04.D, title: 'O = Y·W_O + b', symmetric: true, op: '→' }] });
                    },
                },
                {
                    id: 'headwrite', label: '每个头写进残差流的量', sub: '‖Y_h · W_O[h]‖，行 = token · 列 = 头 · <b>点击选行 / 头</b>', size: [360, 290],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer];
                        S._lay.headwrite = P04.heat(c, Ly.hw, S.T, P04.H, { rows: P04.posLabels(S.ids), cols: [0, 1, 2, 3].map((h) => 'H' + (h + 1)), square: false, hiRow: S.qi, hiCol: S.head, range: [0, Math.max(1e-6, ...Ly.hw)], map: 'inferno', title: `L${S.layer + 1} 各头输出的范数` });
                    },
                    probe: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.headwrite, info); return h ? `${P04.HL(S.layer, h.c)} 在 ${P04.show(S.ids[h.r])}（${h.r}）写入 ‖·‖ = ${S.F.layers[S.layer].hw[h.r * P04.H + h.c].toFixed(3)}` : ''; },
                    pick: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.headwrite, info); if (h) { PL.setParam('head', h.c); PL.setParam('query', h.r); } },
                },
                {
                    id: 'ablscan', needs: ['ablscan'], label: '逐个消融：查询位置的正确答案还剩多少概率', sub: '每次只置零一个部件、重新前向 · 黄 = 顶栏「消融」当前选中的部件', size: [360, 290],
                    render: (c, S) => {
                        const A = S.abl, t = P04.show(A.tgt), cur = P04.COMPS.findIndex((q) => q.value === S.ablate);
                        LM.draw.bars(c, A.res.map((r) => r.p), A.res.map((r) => r.label), { horizontal: true, range: [0, 1], digits: 2, title: `p(「${t}」| 前 ${S.qi + 1} 个字)${A.last ? '（句末：用模型自己的 top-1 作答案）' : ''}`, hi: cur, hiColor: LM.C.yellow, colors: A.res.map((r) => (r.p < 0.5 ? LM.C.red : LM.C.blue)) });
                    },
                    probe: (x, y, S, PL, info) => { if (!S.abl) return ''; const k = Math.floor((info.dy - 18) / ((290 - 22) / S.abl.res.length)); const r = S.abl.res[k]; return r ? `${r.label}：p(${P04.show(S.abl.tgt)}) = ${r.p.toFixed(3)}，此时 top-1 = ${P04.show(r.top)}（${r.ptop.toFixed(2)}）` : ''; },
                },
                {
                    id: 'abltext', label: '当前消融的效果', sub: '查询位置 i 的下一个字：消融前 vs 消融后（全页都在用消融后的模型）', size: [360, 110],
                    render: (c, S) => {
                        const i = S.qi, p0 = P04.probsRow(S.F0, i), p1 = P04.probsRow(S.F, i), t0 = LM.topk(p0, 3), t1 = LM.topk(p1, 3), lab = P04.COMPS.find((q) => q.value === S.ablate).label;
                        const fmt = (p, t) => t.map((v) => `${P04.show(v)} ${p[v].toFixed(3)}`).join('   ');
                        LM.draw.text(c, [
                            [{ t: '消融：', color: LM.C.dim }, { t: lab, color: S.ablate === 'none' ? LM.C.gray : LM.C.yellow, bold: true }, { t: `   位置 ${i}「${P04.show(S.ids[i])}」之后`, color: LM.C.dim }],
                            [{ t: '消融前  ', color: LM.C.dim }, { t: fmt(p0, t0), color: LM.C.blue }],
                            [{ t: '消融后  ', color: LM.C.dim }, { t: fmt(p1, t1), color: S.ablate === 'none' ? LM.C.blue : LM.C.orange }],
                            { text: S.ablate === 'none' ? '在顶栏「消融」里选一个头或 MLP 试试。' : `整句平均 loss：${S.F0.loss.toFixed(3)} → ${S.F.loss.toFixed(3)}`, color: LM.C.gray },
                        ], { px: 12 });
                    },
                },
            ],
            notes: '默认句子、i = 15（第 2 个「吃」）：逐个消融时 p(竹) 掉得最多的是 <b>L2 的 MLP</b>（约 0.16）和 L2·H2（约 0.48），其次是 L1 的 MLP（约 0.66），L2·H1 只小降到 0.91；在 Node 里把 L2·H1 和 L2·H2 同时消融，p(竹) 降到 0.29：两个主语头互为备份。换成「小狗…」句子，情况反过来：名字第 1 个字「小」没用，消融盯第 2 个字的 L2·H1 让 p(骨) 掉到 0.21。在 i = 9（以）处消融 <b>L1·H4</b>，p(熊) 从 1.00 直接掉到 0.00——「看食物」这个头是预测名字的唯一来源。',
            warn: '这里用的是<b>置零消融</b>（zero ablation）：被消融的头输出一个训练中从没出现过的值，有时会夸大或掩盖它的作用。可解释性研究里更常用均值消融或激活修补（activation patching）。把它当作粗略但真实的因果检验。',
        },
        // ───────────────────────────── 07 残差流 ─────────────────────────────
        {
            id: 'resid', title: '残差流：每个子层只「读 → 算 → 加回去」', tag: 'RESID',
            desc: 'Transformer 的主干不是一层层的「变换」，而是一条<strong>残差流</strong>：x 从 embedding 出发，每个子层经过 LN 读它、算出一个增量、再<b>加</b>回去。所以每个子层只需要学「要补充什么」，梯度也能沿加法直通到最底层（第 02 篇讲的深网络难训问题，残差连接是关键解法之一）。主图是每个 token 的 ‖x‖ 在 5 个检查点的变化；第二张图是 <strong>logit lens</strong>（nostalgebraist 2020）：把中间的残差流直接接上最后的 ln_f 和 LM head，看「如果现在就停下来，模型会说什么」。在这个模型里，X0、X1a 两行几乎每个位置读出来的都是<b>它自己这个字</b>（权重共享：残差流里 W_te[自己] 的成分最大）；模板里的固定搭配（熊→猫、所→以、猫→每）在 L1 的 MLP 或 L2 的注意力之后就读得出来；而需要事实回忆的「吃→竹」直到<b>最后一个 MLP 写入之后</b>才出现。',
            formula: ['x ← x + Attn(LN_1(x))      x ← x + FFN(LN_2(x))   // 每个 Block 两次「读 → 算 → 加回」', 'lens_c[t] = softmax(LN_f(x_c[t]) · W_te^T)   // 在检查点 c 提前读出'],
            code: {
                title: 'C++ · residualTrace / logitLens', src: `
                // 残差流的 5 个检查点 X0 → X1a → X1 → X2a → X2，以及 4 个子层各自写入的增量 Δ（右下图）
                struct ResidualTrace { std::vector<Tensor> cps, writes; };

                // 每个 Block 两次「读 → 算 → 加回」：x ← x + Attn(LN₁(x))；x ← x + FFN(LN₂(x))（库里 blockForward 的展开版，记下每个检查点）
                ResidualTrace residualTrace(const GPT& m, const std::vector<int>& ids) {
                    int T = (int)ids.size(), H = m.cfg.H;
                    NoGrad ng;
                    Var x = embed(m, ids);
                    ResidualTrace r;
                    r.cps.push_back(x->val);                                         // X0
                    for (const Block& b : m.blocks) {
                        QKV a = projectQKV(b, layerNorm(x, b.ln1g, b.ln1b));
                        Var o = linear(attention(a.q, a.k, a.v, 1, T, H, true), b.Wo, b.bo);
                        x = add(x, o);
                        r.writes.push_back(o->val);
                        r.cps.push_back(x->val);                                     // X1a / X2a：注意力写入之后
                        Var f = linear(gelu(linear(layerNorm(x, b.ln2g, b.ln2b), b.W1, b.b1)), b.W2, b.b2);
                        x = add(x, f);
                        r.writes.push_back(f->val);
                        r.cps.push_back(x->val);                                     // X1 / X2：MLP 写入之后
                    }
                    return r;
                }

                // 每个 token 的 ‖x‖（主图的 5 条折线、右下图的柱）
                std::vector<float> rowNorms(const Tensor& x) {
                    std::vector<float> n = floats(x.rows);
                    for (int i = 0; i < x.rows; i++) {
                        float s = 0;
                        for (int j = 0; j < x.cols; j++) s += x.at(i, j) * x.at(i, j);
                        n[i] = std::sqrt(s);
                    }
                    return n;
                }

                // logit lens：把任意检查点的残差流直接接上最后的 ln_f + 权重共享的 LM head，看「如果现在就停下来，模型会说什么」
                struct LensCell { int top; float ptop, ptarget; };   // 读出的 top-1、它的概率、给真正下一个字的概率（格子颜色）

                std::vector<LensCell> logitLens(const GPT& m, const Tensor& x, const std::vector<int>& targets) {
                    NoGrad ng;
                    Tensor logits = matmulBT(layerNorm(constant(x), m.lnfg, m.lnfb), m.wte)->val;   // [T × 76]
                    std::vector<LensCell> out;
                    for (int i = 0; i < x.rows; i++) {
                        std::vector<float> p = rowOf(logits, i);
                        softmaxInPlace(p, 1.f);
                        LensCell c;
                        c.top = argmaxIndex(p);
                        c.ptop = p[c.top];
                        c.ptarget = targets[i] >= 0 ? p[targets[i]] : NAN;            // 句末没有下一个字
                        out.push_back(c);
                    }
                    return out;
                }`,
            },
            previews: [
                {
                    id: 'resnorm', label: '残差流范数 ‖x‖ 在 5 个检查点', sub: '颜色由浅到深 = 越来越深的层 · 超出纵轴的点贴在顶边并标出数值 · <b>点击选查询位置</b>', large: true, size: [480, 260],
                    render: (c, S) => {
                        const cols = ['#7a8a99', '#56b6c2', '#6ab0e0', '#c678dd', '#e0a060'], N = S.F.cpNorm;
                        const all = Math.max(...N.map((a) => Math.max(...a))), early = Math.max(...N.slice(0, 4).map((a) => Math.max(...a)));
                        const last = Array.from(N[4]).sort((a, b) => a - b), q75 = last[Math.floor(0.75 * (last.length - 1))];
                        const top = Math.min(all * 1.12, Math.max(early * 1.6, q75 * 1.7, 1));
                        S._lay.resnorm = P04.catPlot(c, { labels: P04.chars(S.ids), y: [0, top], hiIdx: S.qi, clipLabels: true, legendX: 0.44, series: N.map((a, k) => ({ data: Array.from(a), kind: 'line', color: cols[k], label: P04.CP_LABEL[k], width: k === 4 ? 2.4 : 1.6 })), yLabel: '‖x‖' });
                    },
                    probe: (x, y, S, PL, info) => { const i = P04.hitCat(S._lay.resnorm, info); if (i < 0) return ''; return `${P04.show(S.ids[i])}（${i}）\n` + S.F.cpNorm.map((a, k) => `${P04.CP_NAME[k]}: ${a[i].toFixed(3)}`).join('\n'); },
                    pick: (x, y, S, PL, info) => { const i = P04.hitCat(S._lay.resnorm, info); if (i >= 0) PL.setParam('query', i); },
                },
                {
                    id: 'lens', label: 'logit lens：每个检查点「现在就读出」的 top-1', sub: '格子里 = 读出的字（加粗 = 已经等于答案）· 颜色 = 读出分布给「真正的下一个字」的概率 · 最下行 = 真正的下一个字', size: [480, 190],
                    render: (c, S) => {
                        const T = S.T, tg = S.F.tg;
                        const rows = S.F.lens.map((L, k) => ({ label: P04.CP_NAME[k], cells: Array.from({ length: T }, (_, i) => ({ t: P04.show(L.top[i]), v: L.ptgt[i], bold: L.top[i] === tg[i] })) }));
                        rows.push({ label: '答案', labelColor: LM.C.green, cells: Array.from({ length: T }, (_, i) => ({ t: tg[i] >= 0 ? P04.show(tg[i]) : '·', color: '#13261a', tc: LM.C.green })) });
                        S._lay.lens = P04.charGrid(c, { rows, colLabels: P04.chars(S.ids), hiCol: S.qi, legend: [0, 1] });
                    },
                    probe: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.lens, info); if (!h || h.r >= 5) return ''; const L = S.F.lens[h.r], i = h.c; return `${P04.CP_NAME[h.r]} · 位置 ${i}（${P04.show(S.ids[i])}）\n读出 top-1：${P04.show(L.top[i])}（${L.ptop[i].toFixed(3)}）\np(真正的下一个字${S.F.tg[i] >= 0 ? ' ' + P04.show(S.F.tg[i]) : ''}) = ${isNaN(L.ptgt[i]) ? '—' : L.ptgt[i].toFixed(3)}`; },
                    pick: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.lens, info); if (h) PL.setParam('query', h.c); },
                },
                {
                    id: 'writes', label: '查询位置 i：每个子层写入的增量大小', sub: '‖X₀‖ 与 4 个子层输出 ‖Δ‖ 的比较', size: [360, 200],
                    render: (c, S) => {
                        const i = S.qi, vals = [S.F.cpNorm[0][i]].concat(S.F.wrNorm.map((a) => a[i])), lab = ['X₀ 嵌入'].concat(P04.WR_LABEL);
                        LM.draw.bars(c, vals, lab, { horizontal: true, digits: 2, colors: ['#7a8a99', '#56b6c2', '#6ab0e0', '#c678dd', '#e0a060'], title: `${P04.show(S.ids[i])}（i = ${i}）` });
                    },
                },
            ],
            notes: '默认 i = 15：logit lens 在 X0 … X2a 读出的都是「吃」本身，p(竹) 直到 X2a（L2 注意力之后）才升到约 0.16，L2 的 MLP 写入后变成 1.00——和上一步的消融结论一致：L2 的注意力把「主语是熊猫」搬过来，L2 的 MLP 把它变成「竹」。X0 … X2a 的 ‖x‖ 都在 1–3 之间；L2 的 MLP 在少数位置写入特别大的增量——这句里是句末「。」（≈ 21）、「猫」（≈ 13）和「子」（≈ 11），恰好都是下一个字最不确定的地方（下一句以哪个字开头、名字后面接哪个句式、句子在「竹子」处结束还是接「，所以」）。在 Node 里对 200 个语料窗口的 6400 个位置测过：‖X2‖ 与预测分布的熵相关系数 0.90（去掉「。」也有 0.86），原因这里不展开。随机初始化时 5 条线几乎重合在 0.17 左右：各子层的输出被 GPT-2 式初始化压得很小。',
            warn: 'logit lens 只是观察工具：中间层的表示并没有被训练成能被 ln_f + W_te 直接读出，读不出来不代表信息不在（大模型里常用额外训练一个线性读出头的「tuned lens」）。',
        },
        // ───────────────────────────── 08 FFN ─────────────────────────────
        {
            id: 'ffn', title: 'FFN（MLP）：逐位置的 32 → 128 → 32，模型的「键值记忆」', tag: 'MLP',
            desc: '注意力负责在位置之间<b>搬运</b>信息，FFN 则<b>对每个位置独立</b>地加工：32 维 → 128 维 → GELU → 32 维。参数大头在这里：每层 FFN 有 8 352 个参数，注意力只有 4 224 个（大模型里 FFN 约占每层参数的 2/3）。一种很有用的看法（Geva et al. 2021）：W₁ 的 128 列是 128 把「钥匙」，GELU 后的激活是每把钥匙与当前位置的匹配程度，W₂ 的 128 行是对应的「值」——FFN 像一张键值记忆表。在这个模型里能直接看到这样的条目：L2 的 <b>#38 号神经元</b>只在主语是熊猫的「吃」后面亮起（≈ 2.4，其他 7 种动物都 ≤ 0.07），它的值向量最推高的 token 正是「竹」（其次是「黑」——熊猫是黑白的）。类似地，只对小猫亮的 #111 把「鱼」排在第 1，只对小鸟亮的 #54 把「虫」排在前 2。',
            formula: ['FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2      W_1: [32 × 128]，W_2: [128 × 32]', 'GELU(u) = u · Φ(u) ≈ 0.5 u (1 + tanh(√(2/π) (u + 0.044715 u^3)))   // Φ = 标准正态 CDF'],
            code: {
                title: 'C++ · ffn / promotedTokens / paramBreakdown', src: `
                // FFN 对每个位置独立：32 → 128 → GELU → 32。u = 预激活（右上直方图），f = GELU 之后的激活（主图），out = 加回残差流的增量
                struct FFNOut { Var u, f, out; };

                FFNOut ffn(const Block& b, const Var& x1) {
                    FFNOut r;
                    Var h2 = layerNorm(x1, b.ln2g, b.ln2b);         // x1 = 注意力写入之后的残差流（X1a / X2a）
                    r.u = linear(h2, b.W1, b.b1);                    // [T × 128]：W₁ 的 128 列 = 128 把「钥匙」
                    r.f = gelu(r.u);                                 // GELU 的 tanh 近似（GPT-2 同款，最小值约 −0.17）
                    r.out = linear(r.f, b.W2, b.b2);                 // [T × 32]：W₂ 的 128 行 = 128 个「值」，按激活加权求和
                    return r;                                        // 下一个检查点 = x1 + r.out
                }

                // 神经元 n 的「值」向量（W₂ 的第 n 行）直接推高哪个 token：score(v) = (W₂[n] ⊙ γ_f) · W_te[v]，忽略 ln_f 的除以标准差。
                // 返回前两名（跳过 4 个特殊 token），右下图的标签。L2 的 #38 → 竹、黑；#111 → 鱼
                struct Promoted { int first, second; float score; };

                Promoted promotedTokens(const GPT& m, int layer, int n) {
                    const Tensor& W2 = m.blocks[layer].W2->val;      // [128 × 32]，[in × out]：第 n 行 = 神经元 n 写进残差流的方向
                    const Tensor& E = m.wte->val;                    // [76 × 32]
                    int d = W2.cols;
                    std::vector<float> wg = floats(d);
                    float any = 0;
                    for (int j = 0; j < d; j++) {
                        wg[j] = W2.at(n, j) * m.lnfg->val.v[j];
                        any += std::fabs(wg[j]);
                    }
                    Promoted p;
                    p.first = -1;
                    p.second = -1;
                    p.score = 0;
                    if (any == 0) return p;                          // 这个 MLP 被消融了（W₂ 全 0）
                    float v1 = -INFINITY, v2 = -INFINITY;
                    for (int v = 4; v < E.rows; v++) {
                        float s = 0;
                        for (int j = 0; j < d; j++) s += wg[j] * E.at(v, j);
                        if (s > v1) { p.second = p.first; v2 = v1; p.first = v; v1 = s; }
                        else if (s > v2) { p.second = v; v2 = s; }
                    }
                    p.score = v1;
                    return p;
                }

                // 参数都花在哪里（右下表）：每层注意力 4·d² + 4·d，每层 FFN 2·d·ff + ff + d
                struct ParamCount { int64_t emb, attn, mlp, ln, total; };

                ParamCount paramBreakdown(int64_t V, int64_t T, int64_t d, int64_t L, int64_t ff) {
                    ParamCount c;
                    c.emb = V * d + T * d;                           // W_te + W_pe
                    c.attn = L * (4 * d * d + 4 * d);                // W_Q、W_K、W_V、W_O 及偏置
                    c.mlp = L * (2 * d * ff + ff + d);               // W₁、W₂ 及偏置
                    c.ln = L * 4 * d + 2 * d;                        // 每层 2 个 LN 的 γ、β，加 ln_f
                    c.total = c.emb + c.attn + c.mlp + c.ln;         // 本页 28 928；GPT-2 small（50257, 1024, 768, 12, 3072）≈ 1.24 亿
                    return c;
                }`,
            },
            previews: [
                {
                    id: 'ffnheat', label: 'GELU 之后的激活 f [T × 128]', sub: '亮 = 强激活，黑 = ≈ 0 或负值（GELU 最小 −0.17）· 色阶截在第 99 百分位 · <b>点击选查询位置</b>', large: true, size: [480, 300],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer], srt = Float32Array.from(Ly.f1).sort(), q99 = Math.max(0.5, srt[Math.floor(0.99 * (srt.length - 1))]);
                        S._lay.ffnheat = P04.heat(c, Ly.f1, S.T, P04.FF, { rows: P04.posLabels(S.ids), range: [0, q99], map: 'inferno', square: false, hiRow: S.qi, title: `L${S.layer + 1} 的 FFN 激活（128 个神经元，最大 ${srt[srt.length - 1].toFixed(1)}）` });
                    },
                    probe: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.ffnheat, info); if (!h) return ''; const Ly = S.F.layers[S.layer], n = h.c; return `#${n} 在 ${P04.show(S.ids[h.r])}（${h.r}）：u = ${Ly.u[h.r * P04.FF + n].toFixed(2)} → GELU = ${Ly.f1[h.r * P04.FF + n].toFixed(2)}\n值向量推高：${Ly.promo[n] >= 0 ? P04.show(Ly.promo[n]) + '（+' + Ly.promoV[n].toFixed(2) + '）、' + P04.show(Ly.promo2[n]) : '—'}`; },
                    pick: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.ffnheat, info); if (h) PL.setParam('query', h.r); },
                },
                {
                    id: 'ffnhist', label: '预激活 u 的分布与 GELU', sub: '柱 = 整句 T × 128 个预激活的直方图（左轴为相对高度）· 曲线 = GELU(u)', size: [360, 220],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer], hs = P04.hist(Ly.u, -6, 4, 40), mx = Math.max(...hs.c) || 1;
                        let neg = 0, small = 0, big = 0; const f = Ly.f1; for (let k = 0; k < f.length; k++) { if (f[k] < 0) neg++; if (Math.abs(f[k]) < 0.05) small++; if (f[k] > 0.5) big++; }
                        const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
                        PL.draw.plot(c, { x: [-6, 4], y: [-0.5, 6], xLabel: '预激活 u', series: [{ data: Array.from(hs.c, (v) => 4.5 * v / mx), x: hs.x, bar: true, color: '#3a5a78', label: 'u 的直方图' }, { fn: gelu, color: LM.C.orange, width: 2, label: 'GELU(u)' }] });
                        const { ctx } = PL.draw.ctx(c, { keep: true });
                        ctx.font = LM.FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        [`f < 0：${(100 * neg / f.length).toFixed(0)}%`, `|f| < 0.05：${(100 * small / f.length).toFixed(0)}%`, `f > 0.5：每个 token ${(big / S.T).toFixed(1)} 个`].forEach((t, k) => { ctx.fillStyle = 'rgba(8,10,12,0.8)'; ctx.fillRect(52, 10 + k * 15, ctx.measureText(t).width + 6, 14); ctx.fillStyle = LM.C.text; ctx.fillText(t, 55, 11 + k * 15); });
                    },
                },
                {
                    id: 'ffntop', label: '查询位置 i 最活跃的 8 个神经元', sub: '标签 = 神经元编号 → 它的值向量最推高的前 2 个 token（直接 logit 贡献）', size: [360, 240],
                    render: (c, S) => {
                        const Ly = S.F.layers[S.layer], i = S.qi, row = Ly.f1.subarray(i * P04.FF, (i + 1) * P04.FF), top = LM.topk(row, 8), tg = S.F.tg[i];
                        const lab = (n) => (Ly.promo[n] >= 0 ? `#${n} → ${P04.show(Ly.promo[n])} ${P04.show(Ly.promo2[n])}` : `#${n} → —`);
                        LM.draw.bars(c, top.map((n) => row[n]), top.map(lab), { horizontal: true, digits: 2, range: [0, Math.max(1, row[top[0]] * 1.05)], title: `L${S.layer + 1}，${P04.show(S.ids[i])}（i = ${i}）· 绿 = 前 2 名里有下一个字`, hi: new Set(top.map((n, r) => (Ly.promo[n] === tg || Ly.promo2[n] === tg ? r : -1)).filter((r) => r >= 0)), hiColor: LM.C.green, color: LM.C.orange });
                    },
                },
                {
                    id: 'params', label: '参数都花在哪里', sub: '本页模型（真实张量大小）vs 按配置推算的 GPT-2 small（V = 50257, T = 1024, d = 768, L = 12）', size: [360, 165],
                    render: (c, S) => {
                        const a = S.pb, b = S.pbGPT2, pc = (x, t) => { const v = 100 * x / t; return (v < 1 ? v.toFixed(2) : v.toFixed(1)) + '%'; }, M = (x) => (x >= 1e6 ? (x / 1e6).toFixed(1) + ' M' : (x / 1e3).toFixed(0) + ' k');
                        const cell = (t, tc) => ({ t, tc: tc || LM.C.text, color: '#161a20', px: 12 });
                        const row = (label, x, y, col) => ({ label, labelColor: col, cells: [cell(String(x), col), cell(pc(x, a.total)), cell(M(y), LM.C.gray), cell(pc(y, b.total))] });
                        P04.charGrid(c, { rows: [row('嵌入 W_te+W_pe', a.emb, b.emb, LM.C.blue), row('注意力', a.attn, b.attn, LM.C.purple), row('FFN', a.mlp, b.mlp, LM.C.orange), row('LayerNorm', a.ln, b.ln, LM.C.gray), row('合计', a.total, b.total, LM.C.text)], colLabels: ['本页', '占比', 'GPT-2 small', '占比'], cwMax: 74, chMax: 28 });
                    },
                },
            ],
            notes: '把 <b>层</b> 切到 L1 对比：L1 的激活更稀疏（每个 token 只有约 2 个神经元 > 0.5），L2 每个 token 有十几个。默认 i = 15 时，L2 最活跃的正是 #38（→ 竹）。在顶栏 <b>消融</b> 里选「L2 的 MLP」：这一行全部失效，Step 09 里「竹」的概率掉到约 0.16。换成「小猫吃鱼，老虎吃肉。」看 i = 2 和 i = 7：最活跃的分别是 #111（值向量推高「鱼」）和 #64（推高「猴」「肉」）——直接 logit 只是近似，单个神经元也常常同时服务好几个字，最终答案是许多神经元叠加的结果。',
            warn: '「直接 logit 贡献」= (W₂ 的第 n 行 ⊙ γ_f) · W_te[v]，忽略了 ln_f 的除以标准差，只能比较相对大小。GELU 允许小的负值（最小约 −0.17），不像 ReLU 有严格的 0，所以「稀疏度」只能按阈值统计（这里用 |f| < 0.05 和 f > 0.5）。真实大模型里大多数神经元是多义的（一个神经元参与很多概念），像 #38 这么「干净」的例子在小模型里更常见。',
        },
        // ───────────────────────────── 09 输出 ─────────────────────────────
        {
            id: 'out', title: '输出：ln_f → 权重共享的 LM head → softmax', tag: 'OUT',
            desc: '最后一个 Block 之后再做一次 LayerNorm（ln_f），然后乘 W_teᵀ——和输入层用的是<b>同一张</b> embedding 表（权重共享，weight tying，Press & Wolf 2017）：第 v 个 token 的 logit 就是最后的隐状态与 W_te[v] 的点积，「和哪个字的 embedding 越像，就越可能输出哪个字」。softmax 把 76 个 logit 变成概率。每个位置都输出一个分布：训练时全部用上（Step 10），生成时只用最后一个（Step 11）。',
            formula: ['logits = LN_f(x_L) · W_te^T   // [T × 76]，与输入层共享 W_te', 'p_t(v) = exp(logits_{t,v}) / Σ_u exp(logits_{t,u})'],
            code: {
                title: 'C++ · lmHead / forwardByParts / topIndices：logits 与 top-k', src: `
                // 输出层：ln_f 之后乘 W_teᵀ —— 和输入层同一张表（权重共享）：logits[t][v] = LN_f(x_t) · W_te[v]，[T × 76]
                Var lmHead(const GPT& m, const Var& x) {
                    return matmulBT(layerNorm(x, m.lnfg, m.lnfb), m.wte);
                }

                // 完整的一次前向 = 嵌入 → 2 个 Block → 输出层，和库里的 gptForward(m, ids, 1, T) 一样
                Var forwardByParts(const GPT& m, const std::vector<int>& ids) {
                    int T = (int)ids.size();
                    Var x = embed(m, ids);                                                   // Step 01
                    for (const Block& b : m.blocks) x = blockForward(b, x, 1, T, m.cfg.H);   // Step 02–08，每层一次
                    return lmHead(m, x);                                                     // Step 09
                }

                // 概率最大的 k 个下标，从大到小
                std::vector<int> topIndices(const std::vector<float>& p, int k) {
                    std::vector<int> idx = ints((int)p.size());
                    for (int i = 0; i < (int)p.size(); i++) idx[i] = i;
                    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&p](int a, int b) { return p[a] > p[b]; });
                    idx.resize(k);
                    return idx;
                }

                // 主图：第 2 个「吃」（i = 15）之后最可能的 10 个字
                void printTop10(const GPT& m) {
                    Vocab voc = makeVocab();
                    std::vector<int> ids = encode(voc, "熊猫喜欢吃竹子，所以熊猫每天都吃竹子。");
                    NoGrad ng;
                    std::vector<float> p = rowOf(forwardByParts(m, ids)->val, 15);
                    softmaxInPlace(p, 1.f);                                                  // 先减最大值再 exp：数值稳定
                    for (int v : topIndices(p, 10)) std::printf("%s %.3f\\n", voc.itos[v].c_str(), p[v]);   // 竹 0.998 …
                }`,
            },
            previews: [
                {
                    id: 'topk', label: '查询位置 i 的下一个字：top-10（当前权重）', sub: '绿 = 句子里真正的下一个字', large: true, size: [360, 300],
                    render: (c, S) => P04.topkBars(c, S.F, S.qi, 10, `${S.weights === 'rand' ? '随机初始化' : '训练后'}${S.ablate !== 'none' ? '（已消融）' : ''}：「${P04.show(S.ids[S.qi])}」之后`),
                },
                {
                    id: 'topkOther', label: '同一位置，另一套权重', sub: '训练后 ↔ 随机初始化（不消融）', size: [360, 300],
                    render: (c, S) => P04.topkBars(c, S.FO, S.qi, 10, `${S.weights === 'rand' ? '训练后' : '随机初始化'}：「${P04.show(S.ids[S.qi])}」之后`),
                },
                {
                    id: 'predgrid', label: '每个位置的预测', sub: '第 2 行 = 模型的 top-1 预测（绿 = 猜对，红 = 猜错）· 第 3 行 = 它给真正下一个字的概率（%）· <b>点击选位置</b>', size: [480, 125],
                    render: (c, S) => {
                        const F = S.F, ids = Array.from(S.ids);
                        const top = ids.map((_, i) => LM.argmax(P04.probsRow(F, i)));
                        S._lay.predgrid = P04.charGrid(c, {
                            rows: [
                                { label: '输入', cells: ids.map((id) => ({ t: P04.show(id), color: LM.tokColor(id) })) },
                                { label: 'top-1', cells: top.map((v, i) => ({ t: P04.show(v), color: F.tg[i] < 0 ? '#1a1d22' : v === F.tg[i] ? '#1d4a2a' : '#5a1f24' })) },
                                { label: 'p(答案)%', cells: ids.map((_, i) => (F.tg[i] >= 0 ? { t: Math.round(100 * P04.probsRow(F, i)[F.tg[i]]), v: P04.probsRow(F, i)[F.tg[i]], px: 11 } : { t: '—', px: 11 })) },
                            ], hiCol: S.qi, cwMax: 30, chMax: 34,
                        });
                    },
                    probe: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.predgrid, info); if (!h) return ''; const i = h.c, p = P04.probsRow(S.F, i), t = LM.topk(p, 3); return `位置 ${i}（${P04.show(S.ids[i])}）之后：` + t.map((v) => `${P04.show(v)} ${p[v].toFixed(3)}`).join('  '); },
                    pick: (x, y, S, PL, info) => { const h = P04.hitLay(S._lay.predgrid, info); if (h) PL.setParam('query', h.c); },
                },
            ],
            notes: '默认句子第 2 个「吃」之后：训练后 p(竹) ≈ 0.998；随机初始化时 76 个字一视同仁，最高的也只有约 0.02（≈ 1/76 = 0.013）。最有意思的是反事实句：「熊猫喜欢吃<b>香蕉</b>，所以…」在 i = 9 预测「猴」（0.99）——模型由香蕉推断主语是猴子；而 i = 15 仍然以 1.00 预测「竹」，完全无视前文的香蕉。它从来不是在<b>抄写</b>前文的食物，而是在<b>回忆</b>主语对应的事实：在训练语料里两种做法给出的答案永远相同，模型学会了更省事的那一种。',
            warn: '权重共享让参数少了 76 × 32 个（GPT-2 small 里是 50257 × 768 ≈ 3 860 万）。logits 没有归一化，给一行加同一个常数 softmax 不变；数值上要先减最大值再 exp（库里的 <code>softmaxInPlace</code>、<code>crossEntropy</code> 和页面的 JS 都这么做，PyTorch 的 <code>F.softmax</code> 也一样）。',
        },
        // ───────────────────────────── 10 逐位置损失 ─────────────────────────────
        {
            id: 'loss', title: '逐位置损失：哪里容易、哪里难', tag: 'LOSS',
            desc: '训练目标是每个位置「真正的下一个字」的负对数概率 −log p_t(x_{t+1})，对所有位置取平均就是交叉熵损失（第 08 篇用它训练）。逐个位置看，能分清哪些预测是<b>模板</b>（句式里固定的字，loss ≈ 0）、哪些是<b>真正的不确定性</b>：例如「小狗…」句子开头，「小」后面可能是小猫、小狗、小兔、小鸟、小羊 5 种动物中的任何一个，模型给每种约 20% 的概率，loss ≈ ln 5 = 1.61——在均匀采样动物的语料上这已经是最优，再训练也降不下去。名字之后用哪个句式（喜欢吃 / 最爱吃 / 住在 / 是…）同样是真正的不确定。随机初始化的模型在每个位置都 ≈ ln 76 = 4.33。',
            formula: ['ℓ_t = −log p_t(x_{t+1})      L = (1 / (T−1)) Σ_t ℓ_t', '随机猜：ℓ = ln 76 ≈ 4.33      困惑度 PPL = e^L   // 「相当于在几个字里平均地猜」'],
            code: {
                title: 'C++ · perPositionLoss：逐位置交叉熵', src: `
                // 逐位置交叉熵 ℓ_t = −log p_t(x_{t+1})（主图的每根柱子，写进 rowLoss）；返回整句平均 L = Σ_t ℓ_t / (T − 1)
                float perPositionLoss(GPT& m, const std::vector<int>& ids, std::vector<float>& rowLoss) {
                    NoGrad ng;
                    Var logits = gptForward(m, ids, 1, (int)ids.size());
                    Var loss = crossEntropy(logits, targetsOf(ids), nullptr, 0.f, &rowLoss);   // 最后一个位置的答案是 −1：不计入
                    return loss->val.v[0];                           // 训练时最小化的就是它（第 08 篇）
                }

                // 整句汇总（右图）：平均 loss、困惑度 PPL = e^L（「相当于在几个字里平均地猜」，瞎猜 = 76）、最难的 3 个位置
                struct LossSummary { float loss, ppl; std::vector<int> hardest; };

                LossSummary lossSummary(GPT& m, const std::vector<int>& ids) {
                    std::vector<float> rowLoss;
                    LossSummary s;
                    s.loss = perPositionLoss(m, ids, rowLoss);
                    s.ppl = std::exp(s.loss);
                    rowLoss.pop_back();                              // 最后一个位置没有答案（NaN）
                    s.hardest = topIndices(rowLoss, 3);
                    return s;
                }`,
            },
            previews: [
                {
                    id: 'loss', label: '每个位置的 −log p(下一个字)', sub: '上排 = 要预测的下一个字 · 下排 = 当前字 · 蓝柱 = 当前权重 · 灰点 = 另一套权重 · <b>点击选位置</b>', large: true, size: [480, 280],
                    render: (c, S) => {
                        const T = S.T, n = T - 1, cur = Array.from(S.F.rowLoss.subarray(0, n)), oth = Array.from(S.FO.rowLoss.subarray(0, n));
                        const lab = Array.from(S.ids.slice(1), P04.show), lab2 = Array.from(S.ids.slice(0, n), P04.show);
                        const top = Math.max(5, ...cur, ...oth) * 1.25;
                        S._lay.loss = P04.catPlot(c, {
                            labels: lab, labels2: lab2, y: [0, top], hiIdx: S.qi < n ? S.qi : -1, yLabel: '−log p',
                            series: [{ data: cur, kind: 'bar', color: (v) => (v > 1 ? LM.C.orange : LM.C.blue), legendColor: LM.C.blue, label: (S.weights === 'rand' ? '随机初始化' : '训练后') + '（> 1 标橙）' }, { data: oth, kind: 'dot', color: LM.C.gray, label: S.weights === 'rand' ? '训练后' : '随机初始化' }],
                            hlines: [{ y: Math.log(76), color: LM.C.red, label: 'ln 76 = 4.33（瞎猜）' }, { y: S.F.loss, color: LM.C.green, label: `平均 L = ${S.F.loss.toFixed(3)}`, align: 'right' }],
                        });
                    },
                    probe: (x, y, S, PL, info) => { const i = P04.hitCat(S._lay.loss, info); if (i < 0) return ''; const p = P04.probsRow(S.F, i), t = LM.topk(p, 3); return `位置 ${i}：${P04.show(S.ids[i])} → ${P04.show(S.ids[i + 1])}\n−log p = ${S.F.rowLoss[i].toFixed(3)}（p = ${p[S.ids[i + 1]].toFixed(3)}）\n模型的前 3：` + t.map((v) => `${P04.show(v)} ${p[v].toFixed(2)}`).join('  '); },
                    pick: (x, y, S, PL, info) => { const i = P04.hitCat(S._lay.loss, info); if (i >= 0) PL.setParam('query', i); },
                },
                {
                    id: 'losstext', label: '整句汇总', sub: '最难的 3 个位置和模型当时的猜测', size: [360, 130],
                    render: (c, S) => {
                        const F = S.F, n = S.T - 1, order = Array.from({ length: n }, (_, i) => i).sort((a, b) => F.rowLoss[b] - F.rowLoss[a]).slice(0, 3);
                        const lines = [
                            [{ t: '平均 loss  ', color: LM.C.dim }, { t: F.loss.toFixed(3), color: LM.C.green, bold: true }, { t: `   困惑度 e^L = ${Math.exp(F.loss).toFixed(2)}`, color: LM.C.gray }],
                            [{ t: '另一套权重 ', color: LM.C.dim }, { t: S.FO.loss.toFixed(3), color: LM.C.gray }, { t: `   ln 76 = ${Math.log(76).toFixed(3)}`, color: LM.C.dim }],
                        ];
                        order.forEach((i) => { const p = P04.probsRow(F, i), t = LM.topk(p, 3); lines.push([{ t: `${P04.show(S.ids[i])}→${P04.show(S.ids[i + 1])}  `, color: LM.C.orange }, { t: F.rowLoss[i].toFixed(2) + '  ', color: LM.C.text }, { t: '猜 ' + t.map((v) => `${P04.show(v)}${p[v].toFixed(2)}`).join(' '), color: LM.C.gray }]); });
                        LM.draw.text(c, lines, { px: 12 });
                    },
                },
            ],
            notes: '默认句子里只有两处真的难：「猫→喜」（名字之后用哪个句式）和「子→，」（句子在「竹子」处结束还是接「，所以…」，模型更倾向于结束）；其余位置都是模板或可以由前文推出，loss ≈ 0。两句反事实里，loss 尖峰正好落在「不合常理」的字上：模型被「香蕉」和「竹子」吓了一跳，也在「以→熊 / 老」处猜错了名字。',
        },
        // ───────────────────────────── 11 生成 + KV cache ─────────────────────────────
        {
            id: 'gen', title: '自回归生成与 KV cache', tag: 'GEN',
            desc: '生成就是反复「前向 → 取最后一个位置的分布 → 选一个字 → 接到末尾」。朴素做法每生成一个字都把整段重算一遍，但因果掩码保证了前面位置的 K、V 永远不会再变（它们只依赖更早的字）——把它们缓存起来，这就是 <strong>KV cache</strong>：新 token 只需算自己的 q、k、v，再和缓存里所有的 k、v 做注意力，每步的计算量从 O(t) 个 token 降到 1 个。代价是显存：每个 token 每层要存一份 k 和 v。本步用「句子的前 i+1 个字」做提示词（prompt）续写到上下文上限 32，并逐步核对 KV cache 路径与完整前向的 logits。',
            formula: ['第 t 步：只算 q_t, k_t, v_t；K = [K_cache; k_t]，V = [V_cache; v_t]，a_t = softmax(q_t K^T / √d_h) V', 'KV cache 字节数 = 2 × L × T × n_kv × d_h × 每个数的字节   // 2 = K 和 V'],
            code: {
                title: 'C++ · generateKV（逐 token，带 KV cache）/ layerFlops / kvBytesPerToken', src: `
                // 自回归生成 + 逐步核对。提示词 = 句子的前 i+1 个字，续写到上下文上限 32；temperature = 0 时贪心（取概率最大的字）。
                // 每个新 token 只算自己的 q / k / v：库里的 gptStep 把它的 k、v 写进 cache 的第 pos 行，再和 cache 里 0…pos 的所有 k、v 做注意力。
                // 去掉核对那几行，就是库里的 generate(m, prompt, maxNew, temperature, 0, 1, r, greedy)
                struct GenResult { std::vector<int> ids; std::vector<float> diffs; };

                GenResult generateKV(GPT& m, const std::vector<int>& prompt, float temperature, Rng& r) {
                    const GPTConfig& g = m.cfg;
                    KVCache c;                                       // 每层一份 K、V：[T × d]，位置 t 存在第 t 行
                    c.k.assign(g.L, floats(g.T * g.d));
                    c.v.assign(g.L, floats(g.T * g.d));
                    std::vector<float> logits;
                    for (int t = 0; t < (int)prompt.size(); t++) logits = gptStep(m, c, prompt[t], t);   // 读提示词
                    GenResult out;
                    out.ids = prompt;
                    while ((int)out.ids.size() < g.T) {
                        int id = temperature > 0 ? sampleIndex(decodeDist(logits, temperature, 0, 1.f), r) : argmaxIndex(logits);
                        {   // 核对（右上图）：把整段重新完整前向一遍，最后一个位置的 logits 应与 KV cache 路径相同
                            NoGrad ng;
                            int T = (int)out.ids.size();
                            std::vector<float> ref = rowOf(gptForward(m, out.ids, 1, T)->val, T - 1);
                            float d = 0;
                            for (int v = 0; v < g.V; v++) d = std::max(d, std::fabs(ref[v] - logits[v]));
                            out.diffs.push_back(d);                  // ≈ 10⁻⁶：只差浮点运算的顺序
                        }
                        out.ids.push_back(id);
                        if ((int)out.ids.size() < g.T) logits = gptStep(m, c, id, (int)out.ids.size() - 1);   // 新 token：只算 1 个位置
                    }
                    return out;
                }

                // 处理位置 t 的一个 token、一层的 FLOPs（一次乘加算 2）：4 个 d×d 投影 + 2 个 FFN 矩阵，注意力看 t+1 个 key（打分 + 加权求和）
                double layerFlops(int d, int ff, int t) { return 2.0 * (4 * d * d + 2 * d * ff) + 4.0 * d * (t + 1); }

                // 累计 FLOPs（右下图）：KV cache 每步只处理新位置；不用 cache 则每生成一个字都把整段前缀重算一遍（二次增长）
                struct FlopCurve { std::vector<double> kv, noCache; };

                FlopCurve generationFlops(const GPTConfig& g, int P, int nGen) {
                    FlopCurve c;
                    double head = 2.0 * g.d * g.V, accKV = 0, accNo = 0;               // LM head：每个新字 2·d·V
                    for (int s = 0; s < P - 1; s++) accKV += g.L * layerFlops(g.d, g.ff, s);   // 先读提示词（除最后一个字）
                    c.kv.push_back(0);
                    c.noCache.push_back(0);
                    for (int k = 1; k <= nGen; k++) {
                        int t = P + k - 2;                           // 生成第 k 个新字需要位置 t 的输出
                        accKV += g.L * layerFlops(g.d, g.ff, t) + head;
                        double f = 0;
                        for (int s = 0; s <= t; s++) f += g.L * layerFlops(g.d, g.ff, s);
                        accNo += f + head;
                        c.kv.push_back(accKV);
                        c.noCache.push_back(accNo);
                    }
                    return c;
                }

                // KV cache 每个 token 的字节数 = 2（K 和 V）× 层数 × KV 头数 × 头宽 × 每个数的字节（右下柱状图）
                // 本页 kvBytesPerToken(2, 4, 8, 4) = 512 B（fp32）；LLaMA-2 7B kvBytesPerToken(32, 32, 128, 2) = 512 KB（fp16）
                int64_t kvBytesPerToken(int L, int nKV, int dh, int bytes) { return (int64_t)2 * L * nKV * dh * bytes; }`,
            },
            previews: [
                {
                    id: 'gen', needs: ['generate'], label: '续写结果', sub: '蓝 = 提示词（句子前 i+1 个字）· 绿 = 模型生成 · 拖<b>查询位置 i</b>换提示词、拖<b>温度</b>换解码方式', large: true, size: [480, 105],
                    render: (c, S) => {
                        const g = S.gen;
                        LM.draw.tokens(c, g.ids.map((id, k) => ({ text: P04.show(id), id, color: k < g.P ? '#1d3a52' : '#1d4a2a' })), { px: 15, title: `${S.temp > 0 ? '采样，温度 ' + S.temp.toFixed(2) : '贪心解码'} · 提示词 ${g.P} 字 + 生成 ${g.gen.length} 字（上下文上限 32）` });
                    },
                },
                {
                    id: 'kvcheck', needs: ['generate'], label: 'KV cache vs 完整前向：逐步的最大 |Δlogit|', sub: '每生成一个字，都用完整前向重算一遍同一位置来核对 · 灰虚线 ≈ float32 舍入量级（ε × |logit| ≈ 1.2·10⁻⁷ × 30）', size: [360, 210],
                    render: (c, S) => {
                        const d = S.gen.diffs.map((v) => Math.max(v, 1e-9)), xs = d.map((_, k) => k + 1);
                        PL.draw.plot(c, { x: [0.5, Math.max(2, d.length) + 0.5], y: [1e-9, 1e-2], logY: true, xLabel: '生成的第 k 个字', series: [{ data: d, x: xs, color: LM.C.green, width: 2, label: `最大 ${Math.max(...S.gen.diffs).toExponential(1)}` }], hmarks: [{ y: 1.2e-7 * 30, color: '#555' }] });
                    },
                },
                {
                    id: 'kvflops', needs: ['generate'], label: '累计计算量（FLOPs）', sub: '按本模型各层形状精确计数（GEMM + 注意力；不含 LN / softmax）', size: [360, 210],
                    render: (c, S) => {
                        const g = S.gen, xs = g.kv.map((_, k) => k), last = g.kv.length - 1;
                        PL.draw.plot(c, { x: [0, Math.max(1, last)], y: [0, g.full[last] * 1.08], xLabel: `已生成的字数（最后相差 ${(g.full[last] / Math.max(1, g.kv[last])).toFixed(1)} 倍）`, series: [{ data: g.full, x: xs, color: LM.C.red, width: 2, label: `无 cache ${P04.fmtF(g.full[last])}FLOP` }, { data: g.kv, x: xs, color: LM.C.green, width: 2, label: `KV cache ${P04.fmtF(g.kv[last])}FLOP` }] });
                    },
                },
                {
                    id: 'kvmem', label: 'KV cache 有多大：每个 token 要存多少字节', sub: '2 × 层数 × KV 头数 × 头宽 × 2 字节（fp16）· 本页模型用 fp32', size: [360, 230],
                    render: (c, S) => {
                        const rows = P04.KV_REAL.map((r) => ({ name: r.name, kb: 2 * r.L * r.kv * r.dh * 2 / 1024 }));
                        const mine = 2 * P04.NL * P04.D * 4 / 1024;
                        LM.draw.bars(c, rows.map((r) => r.kb), rows.map((r) => r.name), { horizontal: true, digits: 0, valueSuffix: ' KB', range: [0, 560], title: `本页模型：${(mine * 1024).toFixed(0)} B / token，满 32 个 = ${(mine * 32).toFixed(0)} KB`, colors: [LM.C.gray, LM.C.red, LM.C.green, LM.C.green, LM.C.purple] });
                    },
                },
            ],
            notes: '默认提示词「熊猫喜欢吃竹子，所以熊猫每天都吃」，贪心续写出「竹子。」之后会接着编新的句子（小狗喜欢吃骨头…）：预训练模型只会「接着写语料」，不会在该停的时候停——第 09 篇的 SFT 解决这个问题。核对曲线始终在 10⁻⁶ 附近：两条路径数学上完全等价，差别只是浮点运算的顺序不同。FLOPs 图里不用 cache 的曲线是二次增长；把 <b>查询位置 i</b> 拖小（提示词变短、生成变长），差距更大。',
            warn: '真实大模型推理的瓶颈往往不是 FLOPs，而是每步把整个 KV cache 和权重从显存读一遍的<b>带宽</b>。例如 LLaMA-2 7B（32 层、4096 维、fp16）每个 token 的 KV cache 是 512 KB，4096 个 token 就是 2 GB——这就是 MQA / GQA（Step 12）要压缩 K、V 头数的原因。',
        },
        // ───────────────────────────── 12 教学版 → 生产版 ─────────────────────────────
        {
            id: 'prod', title: '教学版 → 生产版：现代 LLM 改了什么', tag: 'NEXT',
            desc: '本页的迷你 GPT 与 GPT-2 结构相同（可学习位置、Pre-LN、GELU、MHA、权重共享）。2023 年以后的主流开源模型（LLaMA 系列、Mistral 等）在同一骨架上换了几个部件，目的几乎都是<b>更长的上下文、更省的推理、同样计算量下更好的效果</b>：',
            blocks: [
                {
                    type: 'html', html: `<table class="lm-tbl"><tr><th>部件</th><th>本页（≈ GPT-2）</th><th>现代 LLM（例）</th><th>为什么</th></tr>
                    <tr><td>位置</td><td>可学习的绝对位置 W_pe（最多 32 个）</td><td><b>RoPE</b>：在每个头里按位置旋转 q、k（LLaMA、Mistral…）</td><td>打分只依赖相对位置；配合插值 / NTK / YaRN 等方法可以扩展上下文</td></tr>
                    <tr><td>归一化</td><td>LayerNorm（Pre-LN）</td><td><b>RMSNorm</b>（仍是 pre-norm）</td><td>不减均值、没有 β，少一次归约，效果相当</td></tr>
                    <tr><td>FFN</td><td>GELU，隐层 4d</td><td><b>SwiGLU</b>，隐层约 8/3·d（LLaMA-7B：4096 → 11008）</td><td>门控 FFN 在同样参数量下效果更好；三个矩阵，所以隐层缩到 8/3·d</td></tr>
                    <tr><td>K / V 头</td><td>MHA：每个头自己的 K、V</td><td><b>GQA</b>：多个查询头共享一组 K、V（Llama-2 70B、Llama-3、Mistral 都是 8 组）；MQA = 只有 1 组</td><td>KV cache 小几倍，解码更快（Step 11 的 KV cache 柱状图）</td></tr>
                    <tr><td>注意力计算</td><td>显式算出 T × T 矩阵</td><td><b>FlashAttention</b>：按块读 K、V，用 online softmax 累加，T × T 矩阵不落显存</td><td>结果完全相同（下面的实验），显存 O(T) 而非 O(T²)，并且更快</td></tr>
                    <tr><td>稠密 / 稀疏</td><td>每个 token 用全部参数</td><td><b>MoE</b>：FFN 换成多个专家 + 路由（如 Mixtral 8×7B：8 个专家每次选 2 个，约 47B 总参数、每 token 约 13B 激活）</td><td>总参数（容量）大，单 token 计算量小</td></tr>
                    <tr><td>上下文</td><td>32</td><td>Llama-2 4k → Llama-3 8k → Llama-3.1 128k</td><td>长上下文的瓶颈是 KV cache 与注意力的 O(T²)</td></tr>
                    <tr><td>精度</td><td>fp32（权重存成 fp16）</td><td>bf16 训练；推理常用 fp16 / int8 / int4 量化</td><td>显存与带宽</td></tr></table>`,
                },
                {
                    type: 'code', title: 'C++ · rmsNorm / swiglu / applyRope / repeatKV / onlineSoftmax', src: `
                    // RMSNorm（LLaMA、Mistral…）：不减均值、没有 β —— y = γ ⊙ x / √(mean(x²) + ε)，少一次归约
                    Tensor rmsNorm(const Tensor& x, const std::vector<float>& gamma, float eps = 1e-6f) {
                        Tensor y = zeros(x.rows, x.cols);
                        for (int t = 0; t < x.rows; t++) {
                            float ms = 0;
                            for (int j = 0; j < x.cols; j++) ms += x.at(t, j) * x.at(t, j);
                            float r = 1.f / std::sqrt(ms / x.cols + eps);
                            for (int j = 0; j < x.cols; j++) y.at(t, j) = gamma[j] * x.at(t, j) * r;
                        }
                        return y;
                    }

                    // SwiGLU FFN：W₁ 是门、W₃ 是值，三个矩阵都没有偏置 —— FFN(x) = (SiLU(x·W₁) ⊙ (x·W₃)) · W₂
                    // 用库里现成的算子拼起来，反向传播自动就有
                    Var swiglu(const Var& x, const Var& W1, const Var& W3, const Var& W2) {
                        return matmul(mul(silu(matmul(x, W1)), matmul(x, W3)), W2);
                    }

                    // RoPE：每个头里相邻两维 (2i, 2i+1) 看作一个平面向量，按 pos·θᵢ 旋转，θᵢ = base^(−2i/dh)；q、k 都要转，v 不转。
                    // 转完之后 q_m · k_n 只依赖相对位置 m − n。x: [T × H·dh]，第 t 行在位置 pos[t]。
                    // （这是 RoFormer / Meta 原版 LLaMA 的配对方式；HuggingFace 的 rotate_half 把前一半和后一半配对，只差一个固定的维度置换）
                    void applyRope(Tensor& x, const std::vector<int>& pos, int dh, float base = 10000.f) {
                        for (int t = 0; t < x.rows; t++)
                            for (int h = 0; h < x.cols / dh; h++)
                                for (int i = 0; i < dh / 2; i++) {
                                    float a = pos[t] * std::pow(base, -2.f * i / dh);
                                    float x1 = x.at(t, h * dh + 2 * i), x2 = x.at(t, h * dh + 2 * i + 1);
                                    x.at(t, h * dh + 2 * i) = x1 * std::cos(a) - x2 * std::sin(a);
                                    x.at(t, h * dh + 2 * i + 1) = x1 * std::sin(a) + x2 * std::cos(a);
                                }
                    }

                    // GQA：nKV 组 K / V 被 nRep 个查询头共享（Llama-3 8B、Mistral 7B：32 个查询头、8 组 K / V）。
                    // [T × nKV·dh] → [T × nKV·nRep·dh]：查询头 h 用第 h / nRep 组。KV cache 只需要存 nKV 组
                    Tensor repeatKV(const Tensor& K, int nKV, int nRep, int dh) {
                        Tensor out = zeros(K.rows, nKV * nRep * dh);
                        for (int t = 0; t < K.rows; t++)
                            for (int h = 0; h < nKV * nRep; h++)
                                for (int e = 0; e < dh; e++) out.at(t, h * dh + e) = K.at(t, (h / nRep) * dh + e);
                        return out;
                    }

                    // online softmax（FlashAttention 的核心）：按 key 分块扫描，只保留运行最大值 m、归一化和 ℓ、未归一化的输出 o。
                    // 遇到更大的分数时，已累加的 ℓ 和 o 整体乘 r = e^(m旧 − m新) 就「改正」了。tiles = 下面表格的每一列（处理完一块之后的状态）
                    struct OnlineTile { int t0, t1; double blockMax, m, ell, r; };
                    struct OnlineResult { std::vector<OnlineTile> tiles; std::vector<float> out; };

                    // 查询位置 i 在一个头里的注意力输出（key 0…i，每块 block 个）。Qh / Kh / Vh = headCols(q, h, 8) 等，[T × dh]
                    OnlineResult onlineSoftmax(const Tensor& Qh, const Tensor& Kh, const Tensor& Vh, int i, int block) {
                        int n = i + 1, dh = Qh.cols;
                        double s = 1.0 / std::sqrt((double)dh), m = -INFINITY, ell = 0;
                        std::vector<double> o;
                        o.assign(dh, 0.0);
                        OnlineResult res;
                        for (int t0 = 0; t0 < n; t0 += block) {
                            int t1 = std::min(n, t0 + block);
                            std::vector<double> sc;
                            double mt = -INFINITY;
                            for (int j = t0; j < t1; j++) {
                                double dot = 0;
                                for (int c = 0; c < dh; c++) dot += (double)Qh.at(i, c) * Kh.at(j, c);
                                sc.push_back(dot * s);
                                mt = std::max(mt, dot * s);                      // 这一块的最大分 m̃
                            }
                            double mNew = std::max(m, mt), r = std::exp(m - mNew);   // 第一块 m = −∞ → r = 0
                            for (double& x : o) x *= r;                          // 旧的累加量按新的最大值重新缩放
                            double ps = 0;
                            for (int j = t0; j < t1; j++) {
                                double p = std::exp(sc[j - t0] - mNew);
                                ps += p;
                                for (int c = 0; c < dh; c++) o[c] += p * Vh.at(j, c);
                            }
                            ell = ell * r + ps;
                            m = mNew;
                            OnlineTile tile;
                            tile.t0 = t0; tile.t1 = t1; tile.blockMax = mt; tile.m = m; tile.ell = ell; tile.r = r;
                            res.tiles.push_back(tile);
                        }
                        for (double x : o) res.out.push_back((float)(x / ell));  // 与一次性 softmax 的 Y_h[i]（Step 06）相同，差 ~10⁻⁷
                        return res;
                    }

                    // 生产实现（FlashAttention 的 CUDA 内核；PyTorch 2.x 的 scaled_dot_product_attention 会自动派发到这类内核）做的就是
                    // 上面这个循环：每块 K、V 读进片上 SRAM 算完就扔，T × T 的注意力矩阵从不写回显存`,
                },
                {
                    type: 'previews', items: [
                        {
                            id: 'online', label: 'online softmax：按 4 个 key 一块扫描', sub: '选中层 / 头 / 查询位置的真实 q、K、V · 每列 = 处理完一块之后的状态', large: true, size: [480, 175],
                            render: (c, S) => {
                                const o = S.onl, ids = S.ids, tiles = o.tiles;
                                P04.charGrid(c, {
                                    rows: [
                                        { label: '这一块的 key', cells: tiles.map((t) => ({ t: Array.from(ids.slice(t.t0, t.t1), P04.show).join(''), color: '#1d2a3a', px: 13 })) },
                                        { label: '块内最大分 m̃', cells: tiles.map((t) => ({ t: t.mt.toFixed(2), color: '#161a20', tc: LM.C.text, px: 12 })) },
                                        { label: '运行最大值 m', cells: tiles.map((t, k) => ({ t: t.m.toFixed(2), color: '#161a20', tc: k > 0 && t.m > tiles[k - 1].m ? LM.C.yellow : LM.C.text, px: 12 })) },
                                        { label: '旧量缩放 r', cells: tiles.map((t, k) => ({ t: k === 0 ? '—' : t.r.toFixed(3), color: '#161a20', tc: t.r < 1 && k > 0 ? LM.C.yellow : LM.C.gray, px: 12 })) },
                                        { label: '归一化和 ℓ', cells: tiles.map((t) => ({ t: t.ell.toFixed(3), color: '#161a20', tc: LM.C.text, px: 12 })) },
                                    ], cwMax: 66, chMax: 30, title: `${P04.HL(S.layer, S.head)}，查询 ${P04.show(ids[S.qi])}（i = ${S.qi}）：${o.n} 个 key，${tiles.length} 块`,
                                });
                            },
                        },
                        {
                            id: 'onlineout', label: '分块结果 vs 一次性 softmax', sub: '输出向量的 8 维：橙 = online softmax · 蓝 = Step 06 里的 Y_h[i]', size: [360, 200],
                            render: (c, S) => {
                                const o = S.onl;
                                P04.catPlot(c, { labels: o.out.map((_, k) => k), y: [Math.min(0, ...o.out, ...o.ref) * 1.15 - 0.05, Math.max(0, ...o.out, ...o.ref) * 1.15 + 0.05], series: [{ data: o.ref, kind: 'bar', color: LM.C.blue, label: '一次性 softmax' }, { data: o.out, kind: 'bar', color: LM.C.orange, label: `online（最大差 ${o.maxDiff.toExponential(1)}）` }], hlines: [{ y: 0, color: '#444', dash: [] }] });
                            },
                        },
                    ],
                },
                { type: 'note', html: 'online softmax 的关键：已经累加的 ℓ 和 o 都带着旧的最大值 m，遇到更大的分数时整体乘 r = e<sup>m旧 − m新</sup> 即可「改正」，所以 key 可以一块一块流式地读，GPU 上每块放在片上 SRAM 里算完就扔，T × T 的矩阵从不写回显存。黄色数字 = 这一块刷新了最大值、旧量被缩小。最大差在 10⁻⁷ 量级：分块只改变了浮点运算的顺序。' },
                { type: 'warn', html: '表里的模型数字只列了我们有把握的公开配置；不同版本（如 Llama-3 与 3.1、Mistral 的各个版本）在上下文长度、词表等细节上有差别，以官方文档为准。下一篇（08）在浏览器里从随机初始化开始<b>训练</b>这个迷你 GPT——本页用的 llm_weights.js 就是同一份代码离线训练 4000 步的结果。' },
            ],
        },
    ],

    graph: {
        title: '一次前向传播 · 数据流（2 个 Block）',
        intro: '每个节点 = 一次真实的计算（GPU 上通常是一个 kernel：GEMM、LayerNorm、融合的注意力…；这里在 CPU 上用 JS 跑），pin = 张量名 · 形状，同名资源自动连线。残差流的 5 个检查点 <code>X0 → X1a → X1 → X2a → X2</code> 与 Step 07 一致：每个 GEMM 节点多出的那个 X 输入就是残差连接（加法融合在 GEMM 的尾部）。<code>wte</code> 同时连到 Embed 和 LM head——权重共享。各层的权重矩阵画成外部输入（不连线）。图按 3 行「蛇形」排布，换行处从右往左走的连线会画成虚线——那只是走线方向，这张图里没有跨帧反馈。',
        types: {
            tok: { color: '#e8c25a', label: 'tok · token id（int32）' },
            act: { color: '#6ab0e0', label: 'act · 激活 / 残差流 fp32' },
            param: { color: '#5ad07a', label: 'param · 权重' },
            att: { color: '#c678dd', label: 'att · 注意力权重 [H, T, T]' },
            out: { color: '#e06c75', label: 'out · logits / 概率 / loss' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用函数', open: false,
            blocks: LM.CPP.blocks(['gpt']).concat([
                { label: '━ 本页公用：词表（与页面相同）、读权重、深拷贝、取行、下一个字的分布 ━', code: P04_CPP_COMMON },
            ]),
        },
        nodes: (() => {
            const ext = (name, format) => ({ name, format, type: 'param', external: true });
            const X = 262, Y = 330;
            // 3 行 × 7 列的「蛇形」排布；col 同时告诉自动布局只用 7 列来算画布边界
            const at = (col, row, dy) => ({ x: 30 + col * X, y: 30 + row * Y + (dy || 0), col });
            const gemm = (cols, k) => ({ kind: 'CPU', name: 'gemm', count: (S) => S.T + '×' + cols + ' 个输出', idRange: 'i ∈ [0,T) · j ∈ [0,' + cols + ')', note: '每个输出 = ' + k + ' 项点积；FLOPs = 2·T·' + k + '·' + cols + '。GPU 上是一次 GEMM kernel，这里在 CPU 上用 JS 算' });
            const rowsOp = (name, note) => ({ kind: 'CPU', name, count: (S) => S.T + ' 行', idRange: 'i ∈ [0,T)', note });
            const LNNOTE = '每行 32 个数：求均值、方差 → 归一化 → ×γ + β';
            const nodes = [
                Object.assign({ id: 'tokens', title: 'encode', tag: 'CPU', category: 'bake', lifetime: 'param', step: 'embed', desc: '字 → id（字符级）', dispatch: { kind: 'CPU', name: 'encode', count: (S) => S.T + ' 个字', note: '字符级：一个汉字 / 标点 = 一个 token；targets = ids 左移一位' }, outputs: [{ name: 'ids', format: '[T] int32', type: 'tok' }, { name: 'targets', format: '[T] int32', type: 'tok' }], detail: { preview: 'tokstrip' } }, at(0, 0)),
                Object.assign({ id: 'weights', title: 'load_weights', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'embed', desc: 'LM_WEIGHTS.base（fp16 → fp32）', dispatch: { kind: 'CPU', name: 'loadWeights', count: '28 928 个参数', note: '页面加载时解码一次；「随机初始化」是同结构、同种子的 GPT-2 式初始化' }, outputs: [{ name: 'wte', format: '[76, 32]', type: 'param' }, { name: 'wpe', format: '[32, 32]', type: 'param' }], detail: { preview: 'wpesim' } }, at(0, 0, 175)),
                Object.assign({ id: 'embed', title: 'Embed', tag: 'GATHER', step: 'embed', lifetime: 'param', desc: 'wte[ids] + wpe[0:T]', dispatch: rowsOp('gather + add', 'X0[i] = wte[ids[i]] + wpe[i]：两次查表、一次加法，没有乘法'), inputs: [{ name: 'ids', format: '[T] int32', type: 'tok' }, { name: 'wte', format: '[76, 32]', type: 'param' }, { name: 'wpe', format: '[32, 32]', type: 'param' }], outputs: [{ name: 'X0', format: '[T, 32]', type: 'act' }], detail: { preview: 'embsum' } }, at(1, 0)),
            ];
            const block = (l, c0) => {
                const n = l + 1, xin = l === 0 ? 'X0' : 'X1', xa = 'X' + n + 'a', xo = 'X' + n, L = '·L' + n;
                const pos = (k) => { const c = c0 + k; return at(c % 7, Math.floor(c / 7)); };
                return [
                    Object.assign({ id: 'ln1_' + n, title: 'LN₁ · L' + n, tag: 'NORM', step: 'ln', lifetime: 'param', desc: '(x − μ)/σ · γ + β', dispatch: rowsOp('layernorm', LNNOTE), inputs: [{ name: xin, format: '[T, 32]', type: 'act' }], outputs: [{ name: 'h1' + L, format: '[T, 32]', type: 'act' }], detail: { preview: 'lnstats' } }, pos(0)),
                    Object.assign({ id: 'qkv_' + n, title: 'QKV GEMM · L' + n, tag: 'GEMM', step: 'qkv', lifetime: 'param', desc: 'h · [W_Q W_K W_V] + b', dispatch: gemm(96, 32), inputs: [{ name: 'h1' + L, format: '[T, 32]', type: 'act' }, ext('W_qkv' + L, '[32, 96]')], outputs: [{ name: 'qkv' + L, format: '[T, 3×32]', type: 'act' }], detail: { preview: 'qkvhead' } }, pos(1)),
                    Object.assign({ id: 'attn_' + n, title: 'Attention · L' + n, tag: 'ATTN', step: 'attn', lifetime: 'param', desc: 'softmax(QKᵀ/√8 + M) · V', dispatch: { kind: 'CPU', name: 'causal attention', count: (S) => '4 头 × ' + S.T + ' 行', idRange: 'h ∈ [0,4) · i ∈ [0,T) · j ≤ i', note: '一个融合的注意力 kernel：打分、因果掩码、softmax、加权求和；FLOPs ≈ 4·d·T(T+1)/2。生产实现见 Step 12 的 FlashAttention' }, inputs: [{ name: 'qkv' + L, format: '[T, 3×32]', type: 'act' }], outputs: [{ name: 'y' + L, format: '[T, 32]', type: 'act' }, { name: 'A' + L, format: '[4, T, T]', type: 'att' }], detail: { preview: 'attn', formula: 0, code: 0 } }, pos(2)),
                    Object.assign({ id: 'o_' + n, title: 'O GEMM + 残差 · L' + n, tag: 'GEMM', step: 'proj', lifetime: 'param', desc: xin + ' + y · W_O + b', dispatch: gemm(32, 32), inputs: [{ name: 'y' + L, format: '[T, 32]', type: 'act' }, { name: xin, format: '[T, 32]', type: 'act' }, ext('W_O' + L, '[32, 32]')], outputs: [{ name: xa, format: '[T, 32]', type: 'act' }], detail: { preview: 'yo' } }, pos(3)),
                    Object.assign({ id: 'ln2_' + n, title: 'LN₂ · L' + n, tag: 'NORM', step: 'ln', lifetime: 'param', desc: '(x − μ)/σ · γ + β', dispatch: rowsOp('layernorm', LNNOTE), inputs: [{ name: xa, format: '[T, 32]', type: 'act' }], outputs: [{ name: 'h2' + L, format: '[T, 32]', type: 'act' }], detail: { preview: 'lnheat' } }, pos(4)),
                    Object.assign({ id: 'fc1_' + n, title: 'FC1 GEMM + GELU · L' + n, tag: 'GEMM', step: 'ffn', lifetime: 'param', desc: 'GELU(h · W₁ + b₁)', dispatch: gemm(128, 32), inputs: [{ name: 'h2' + L, format: '[T, 32]', type: 'act' }, ext('W₁' + L, '[32, 128]')], outputs: [{ name: 'f' + L, format: '[T, 128]', type: 'act' }], detail: { preview: 'ffnheat' } }, pos(5)),
                    Object.assign({ id: 'fc2_' + n, title: 'FC2 GEMM + 残差 · L' + n, tag: 'GEMM', step: 'ffn', lifetime: 'param', desc: xa + ' + f · W₂ + b₂', dispatch: gemm(32, 128), inputs: [{ name: 'f' + L, format: '[T, 128]', type: 'act' }, { name: xa, format: '[T, 32]', type: 'act' }, ext('W₂' + L, '[128, 32]')], outputs: [{ name: xo, format: '[T, 32]', type: 'act' }], detail: { preview: 'resnorm', formula: 0, code: 0 } }, pos(6)),
                ];
            };
            nodes.push(...block(0, 2), ...block(1, 9));
            nodes.push(
                Object.assign({ id: 'lnf', title: 'ln_f', tag: 'NORM', step: 'out', lifetime: 'param', desc: '最后一次 LayerNorm', dispatch: rowsOp('layernorm', LNNOTE), inputs: [{ name: 'X2', format: '[T, 32]', type: 'act' }], outputs: [{ name: 'hf', format: '[T, 32]', type: 'act' }], detail: { preview: 'lens' } }, at(2, 2)),
                Object.assign({ id: 'head', title: 'LM head GEMM', tag: 'GEMM', step: 'out', lifetime: 'param', desc: 'hf · wteᵀ（权重共享）', dispatch: { kind: 'CPU', name: 'gemm', count: (S) => S.T + '×76 个输出', idRange: 'i ∈ [0,T) · v ∈ [0,76)', note: '每个输出 = hf[i] · wte[v]（32 项点积）；FLOPs = 2·T·32·76。用的是和 Embed 同一张 wte' }, inputs: [{ name: 'hf', format: '[T, 32]', type: 'act' }, { name: 'wte', format: '[76, 32]', type: 'param' }], outputs: [{ name: 'logits', format: '[T, 76]', type: 'out' }], detail: { preview: 'topk' } }, at(3, 2)),
                Object.assign({ id: 'ce', title: 'softmax / 交叉熵', tag: 'SOFTMAX', category: 'output', step: 'loss', lifetime: 'param', desc: 'p = softmax(logits) · −log p(下一个字)', dispatch: rowsOp('softmax + nll', '每行减最大值 → exp → 归一化；loss = 有下一个字的 T−1 行的 −log p 的平均'), inputs: [{ name: 'logits', format: '[T, 76]', type: 'out' }, { name: 'targets', format: '[T] int32', type: 'tok' }], outputs: [{ name: 'probs', format: '[T, 76]', type: 'out' }, { name: 'loss', format: 'scalar', type: 'out' }], detail: { preview: 'loss' } }, at(4, 2)),
            );
            return nodes;
        })(),
    },

    references: [
        { title: 'Attention Is All You Need', author: 'A. Vaswani, N. Shazeer, N. Parmar, et al.', venue: 'NeurIPS', year: 2017, url: 'https://arxiv.org/abs/1706.03762', note: 'Transformer 原论文：缩放点积注意力、多头、√dₖ 的解释、残差 + LayerNorm（Post-LN）' },
        { title: 'Language Models are Unsupervised Multitask Learners', author: 'A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever', venue: 'OpenAI', year: 2019, note: 'GPT-2：Pre-LN、可学习位置、残差投影按 1/√N 初始化——本页模型的结构' },
        { title: 'nanoGPT', author: 'A. Karpathy', venue: 'GitHub', year: 2022, url: 'https://github.com/karpathy/nanoGPT', note: '本系列迷你 GPT 的结构与命名参照它' },
        { title: 'Layer Normalization', author: 'J. L. Ba, J. R. Kiros, G. E. Hinton', venue: 'arXiv', year: 2016, url: 'https://arxiv.org/abs/1607.06450' },
        { title: 'On Layer Normalization in the Transformer Architecture', author: 'R. Xiong, Y. Yang, D. He, et al.', venue: 'ICML', year: 2020, url: 'https://arxiv.org/abs/2002.04745', note: 'Pre-LN 与 Post-LN 的对比与分析' },
        { title: 'Gaussian Error Linear Units (GELUs)', author: 'D. Hendrycks, K. Gimpel', venue: 'arXiv', year: 2016, url: 'https://arxiv.org/abs/1606.08415' },
        { title: 'Using the Output Embedding to Improve Language Models', author: 'O. Press, L. Wolf', venue: 'EACL', year: 2017, url: 'https://arxiv.org/abs/1608.05859', note: '输入 / 输出 embedding 权重共享' },
        { title: 'A Mathematical Framework for Transformer Circuits', author: 'N. Elhage, N. Nanda, C. Olsson, et al.', venue: 'Transformer Circuits Thread', year: 2021, url: 'https://transformer-circuits.pub/2021/framework/index.html', note: '「残差流」视角、注意力头的读写分解' },
        { title: 'In-context Learning and Induction Heads', author: 'C. Olsson, N. Elhage, N. Nanda, et al.', venue: 'Transformer Circuits Thread', year: 2022, url: 'https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html', note: '归纳头（本页的小模型里没有出现）' },
        { title: 'Transformer Feed-Forward Layers Are Key-Value Memories', author: 'M. Geva, R. Schuster, J. Berant, O. Levy', venue: 'EMNLP', year: 2021, url: 'https://arxiv.org/abs/2012.14913', note: 'FFN 的键值记忆视角（Step 08 的 #38 号神经元）' },
        { title: 'Locating and Editing Factual Associations in GPT', author: 'K. Meng, D. Bau, A. Andonian, Y. Belinkov', venue: 'NeurIPS', year: 2022, url: 'https://arxiv.org/abs/2202.05262', note: '大模型里事实回忆也集中在主语位置的中层 MLP' },
        { title: 'interpreting GPT: the logit lens', author: 'nostalgebraist', venue: 'LessWrong', year: 2020, note: 'logit lens（Step 07）' },
        { title: 'Online normalizer calculation for softmax', author: 'M. Milakov, N. Gimelshein', venue: 'arXiv', year: 2018, url: 'https://arxiv.org/abs/1805.02867' },
        { title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness', author: 'T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré', venue: 'NeurIPS', year: 2022, url: 'https://arxiv.org/abs/2205.14135' },
        { title: 'Fast Transformer Decoding: One Write-Head is All You Need', author: 'N. Shazeer', venue: 'arXiv', year: 2019, url: 'https://arxiv.org/abs/1911.02150', note: 'MQA' },
        { title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints', author: 'J. Ainslie, J. Lee-Thorp, M. de Jong, et al.', venue: 'EMNLP', year: 2023, url: 'https://arxiv.org/abs/2305.13245' },
        { title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding', author: 'J. Su, Y. Lu, S. Pan, et al.', venue: 'arXiv', year: 2021, url: 'https://arxiv.org/abs/2104.09864', note: 'RoPE' },
        { title: 'Root Mean Square Layer Normalization', author: 'B. Zhang, R. Sennrich', venue: 'NeurIPS', year: 2019, url: 'https://arxiv.org/abs/1910.07467' },
        { title: 'GLU Variants Improve Transformer', author: 'N. Shazeer', venue: 'arXiv', year: 2020, url: 'https://arxiv.org/abs/2002.05202', note: 'SwiGLU' },
        { title: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer', author: 'N. Shazeer, A. Mirhoseini, K. Maziarz, et al.', venue: 'ICLR', year: 2017, url: 'https://arxiv.org/abs/1701.06538' },
        { title: 'LLaMA: Open and Efficient Foundation Language Models', author: 'H. Touvron, T. Lavril, G. Izacard, et al.', venue: 'arXiv', year: 2023, url: 'https://arxiv.org/abs/2302.13971' },
        { title: 'Efficiently Scaling Transformer Inference', author: 'R. Pope, S. Douglas, A. Chowdhery, et al.', venue: 'MLSys', year: 2023, url: 'https://arxiv.org/abs/2211.05102', note: 'KV cache、带宽与 MQA 的推理代价分析' },
    ],
});
