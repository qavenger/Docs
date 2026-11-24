// index.lab.js — 系列入口：从机器学习 Hello World 到大模型与 AI 的各个分支（15 篇，按「共享基础 → 通用架构 → 大语言模型 → 分化领域」分组）
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js llm_weights.js index.lab.js -o ../index.html
// 每个步骤卡片是一篇讲义的缩影，预览同样由真实计算得到：GD、实时训练的小 MLP、基座模型的 embedding / 注意力 / 训练曲线 / 生成。

const PIDX = (() => {
    const base = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), LM_WEIGHTS.base.b64);
    const V = LM.vocab;
    const PROMPTS = ['熊猫', '小猫', '猴子', '今天', '小兔喜欢吃'];
    /** 自动挑出「找主语」的头：第二个「吃」处（要预测吃的是什么），哪个头把最多注意力放在主语「熊猫」的各个字上 */
    function attnDemo() {
        const s = '熊猫喜欢吃竹子，所以熊猫每天都吃竹子。', ids = V.encode(s), T = ids.length;
        const out = LM.AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true }));
        const H = base.cfg.H, qi = s.lastIndexOf('吃'), subj = [], best = { score: -1 };
        for (let j = 0; j <= qi; j++) if (s[j] === '熊' || s[j] === '猫') subj.push(j);
        out.trace.layers.forEach((ly, l) => {
            for (let h = 0; h < H; h++) {
                let p = 0; subj.forEach((j) => { p += ly.att[(h * T + qi) * T + j]; });
                if (p > best.score) { let kj = subj[0]; subj.forEach((j) => { if (ly.att[(h * T + qi) * T + j] > ly.att[(h * T + qi) * T + kj]) kj = j; }); Object.assign(best, { score: p, l, h, qi, kj }); }
            }
        });
        let food = 0; out.trace.layers.forEach((ly) => { for (let h = 0; h < H; h++) food = Math.max(food, ly.att[(h * T + qi) * T + s.indexOf('竹')]); });
        best.foodMax = food;
        const ly = out.trace.layers[best.l], A = new Float32Array(T * T);
        for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) A[i * T + j] = j > i ? NaN : ly.att[(best.h * T + i) * T + j];
        const lg = out.logits.data, Vn = base.cfg.V, last = best.qi;
        const pr = LM.softmax(lg, new Float32Array(Vn), 1, last * Vn, Vn);
        return { s, T, A, best, labels: Array.from(s), next: LM.topk(pr, 5).map((i) => ({ tok: V.show(i), p: pr[i] })) };
    }
    /** 基座模型 wte 中若干内容字的 2D PCA */
    function embDemo() {
        const groups = [
            { name: '动物', color: LM.C.blue, chars: '猫狗兔鸟熊猴羊虎' },
            { name: '食物', color: LM.C.orange, chars: '鱼骨萝虫竹蕉草肉' },
            { name: '颜色', color: LM.C.green, chars: '白黄灰蓝黑棕橙' },
            { name: '虚词 / 标点', color: LM.C.purple, chars: '的了在和是，。' },
        ];
        const items = [];
        groups.forEach((g) => { for (const ch of g.chars) items.push({ ch, color: g.color, group: g.name }); });
        const d = base.cfg.d, X = new Float32Array(items.length * d);
        items.forEach((it, i) => X.set(base.wte.data.subarray(V.stoi.get(it.ch) * d, (V.stoi.get(it.ch) + 1) * d), i * d));
        const pc = LM.pca2(X, items.length, d, 3);
        return { items, pts: pc.pts, var: pc.varRatio, groups };
    }
    /** 聊天模板 + 基座模型在 <bot> 之后的下一个 token 分布（它从没见过这些特殊 token） */
    function chatDemo() {
        const prompt = '<user>熊猫喜欢吃什么？<bot>', answer = '熊猫喜欢吃竹子。<eos>';
        const pIds = V.encode(prompt), aIds = V.encode(answer);
        const lg = LM.gpt.nextLogits(base, Array.from(pIds)), pr = LM.softmax(lg, new Float32Array(lg.length));
        const gen = LM.gpt.generate(base, pIds, 12, { greedy: true });
        return { pIds, aIds, top: LM.topk(pr, 6).map((i) => ({ tok: V.show(i), p: pr[i] })), cont: V.decode(gen.gen) };
    }
    /** 合成一张 24×24 的小图并切成 6×6 的 patch（第 07 篇的输入形式） */
    function imageDemo() {
        const W = 24, img = new Float32Array(W * W * 3);
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
            let c = [0.08, 0.09, 0.11];
            if ((x - 7) ** 2 + (y - 12) ** 2 < 25) c = [0.9, 0.25, 0.25];                       // 红色圆
            if (x >= 14 && x <= 21 && y >= 8 && y <= 15) c = [0.25, 0.45, 0.95];                // 蓝色方块
            img.set(c, (y * W + x) * 3);
        }
        return { W, P: 6, img };
    }
    /** 首页的小螺旋数据集（三类），用来实时训练一个 2→32→32→3 的 MLP */
    function spiral(rand, nPer) {
        const pts = [];
        for (let c = 0; c < 3; c++) for (let i = 0; i < nPer; i++) {
            const r = i / nPer * 1.0 + 0.05, t = c * 2.1 + r * 4.2 + 0.25 * LM.randn(rand);
            pts.push({ x: r * Math.cos(t), y: r * Math.sin(t), c });
        }
        return pts;
    }
    const foodMax = attnDemo().best.foodMax;   // 文中引用的数字也是现场算的
    return { base, V, PROMPTS, attnDemo, embDemo, chatDemo, imageDemo, spiral, foodMax };
})();

// 目录卡片上每篇的一句话简介
const PIDX_TOC = {
    '01': '模型、损失、梯度、学习率、SGD、泛化——训练任何模型的四件套',
    '02': '交叉熵、隐藏层、手推反向传播 + 梯度校验、优化器、初始化与梯度流',
    '03': '字节级 BPE、embedding 查表、word2vec 实时训练、正弦位置编码与 RoPE',
    '04': '用训练好的迷你 GPT 做一次前向：LayerNorm、QKV、注意力、残差、FFN、KV cache',
    '05': '在浏览器里从零预训练：batch、loss、反传、AdamW、学习率调度、采样、scaling law',
    '06': '对话模板与 loss mask、SFT、Bradley-Terry 奖励模型、PPO、DPO、GRPO',
    '07': 'patch 化与 ViT、CLIP 对比学习、zero-shot、视觉 token 接入语言模型（LLaVA）',
    '08': '卷积与权重共享、感受野、实时训练 CNN、平移鲁棒性、U-Net 去噪',
    '09': 'PCA、自编码器、VAE 与 ELBO、潜空间插值、VQ-VAE 码本与离散 token',
    '10': 'MDP、价值迭代、Q-learning 与 SARSA、探索、策略梯度、Actor-Critic 到 PPO',
    '11': '前向加噪、去噪网络、DDPM / DDIM 采样、Classifier-Free Guidance、潜空间扩散、流匹配',
    '12': '逐帧生成为何闪烁、从图像模型膨胀、时空分解注意力、物理一致性度量、图生视频与 CFG、长视频漂移',
    '13': '采样与混叠、STFT / Mel、Griffin-Lim、RVQ 编解码与音频 token 语言模型、mel 扩散、TTS 与 ASR',
    '14': '编码器 + 动力学模型、想象 rollout 与误差累积、潜空间规划、在模型里玩',
    '15': '体渲染、位置编码与谱偏差、训练小 NeRF、3D 高斯的投影 / 排序 / 混合与拟合',
};

// ── 第 08–15 篇的首页缩影：全部是真实计算（卷积、PCA、价值迭代、精确的扩散反向过程、视频、频谱、表格世界模型、体渲染） ──
const PIDX2 = (() => {
    const { AG } = LM;
    // 08：一张数字 × 4 个固定卷积核（共享算子 AG.conv2d）
    const KERNELS = [
        { name: 'Sobel x · 竖直边缘', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
        { name: 'Sobel y · 水平边缘', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
        { name: '拉普拉斯 · 轮廓', k: [0, 1, 0, 1, -4, 1, 0, 1, 0] },
        { name: '均值 · 模糊', k: [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9) },
    ];
    function convDemo(digit) {
        const img = LM.digits.render(digit, LM.rng(300 + digit));
        const X = AG.tensor(1, 256, img), W = AG.tensor(4, 9, Float32Array.from(KERNELS.flatMap((q) => q.k)));
        const Y = AG.noGrad(() => AG.conv2d(X, W, null, { C: 1, H: 16, W: 16, K: 3, pad: 1 }));
        return { img, maps: KERNELS.map((_, k) => Y.data.slice(k * 256, (k + 1) * 256)), names: KERNELS.map((q) => q.name) };
    }
    // 09：600 张数字的 PCA = 线性自编码器；在 (PC1, PC2) 平面上网格解码
    function pcaDemo() {
        const n = 600, set = LM.digits.makeSet(n, LM.rng(901)), pc = LM.pca2(set.X, n, 256, 5);
        const sd = [0, 1].map((k) => { let s = 0; for (let i = 0; i < n; i++) s += pc.pts[2 * i + k] ** 2; return Math.sqrt(s / n); });
        const grid = [], G = 5;
        for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
            const a = (c / (G - 1) * 2 - 1) * 2 * sd[0], b = (1 - r / (G - 1) * 2) * 2 * sd[1], im = new Float32Array(256);
            for (let j = 0; j < 256; j++) im[j] = pc.mean[j] + a * pc.axes[0][j] + b * pc.axes[1][j];
            grid.push(im);
        }
        return { pts: pc.pts, y: set.y, n, grid, G, var: pc.varRatio };
    }
    // 10 / 14：6×6 网格世界（# 墙，G 终点 +1，P 陷阱 −1，S 起点；每步 −0.04；以 slip 概率滑向两侧）
    const MAP = ['....#G', '.##.#.', '.#....', '.#.##.', '...#P.', 'S.....'];
    const GW = (() => {
        const H = MAP.length, W = MAP[0].length, cells = [];
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) cells.push(MAP[y][x]);
        const A = [[0, -1], [1, 0], [0, 1], [-1, 0]];   // 上 右 下 左
        const term = (s) => cells[s] === 'G' || cells[s] === 'P';
        const move = (s, a) => { const x = s % W, y = Math.floor(s / W), nx = x + A[a][0], ny = y + A[a][1]; if (nx < 0 || ny < 0 || nx >= W || ny >= H || cells[ny * W + nx] === '#') return s; return ny * W + nx; };
        const reward = (s2) => (cells[s2] === 'G' ? 1 : cells[s2] === 'P' ? -1 : -0.04);
        /** 真实转移：P(s'|s,a)，0.8 照做、各 0.1 滑向左右两侧（slip 可调） */
        const trans = (s, a, slip) => { const out = new Map(); [[a, 1 - slip], [(a + 1) % 4, slip / 2], [(a + 3) % 4, slip / 2]].forEach(([b, p]) => { const s2 = move(s, b); out.set(s2, (out.get(s2) || 0) + p); }); return out; };
        return { H, W, N: H * W, cells, A, term, move, reward, trans, start: cells.indexOf('S') };
    })();
    /** 价值迭代：V_{k+1}(s) = max_a Σ_s' P(s'|s,a)[r(s') + γ V_k(s')]，P 由 model(s, a) 给出（Map s' → p） */
    function valueIteration(model, gamma, K) {
        const N = GW.N, hist = [new Float64Array(N)];
        let V = hist[0];
        for (let k = 0; k < K; k++) {
            const V2 = new Float64Array(N);
            for (let s = 0; s < N; s++) {
                if (GW.cells[s] === '#' || GW.term(s)) continue;
                let best = -Infinity;
                for (let a = 0; a < 4; a++) { let q = 0; model(s, a).forEach((p, s2) => { q += p * (GW.reward(s2) + (GW.term(s2) ? 0 : gamma * V[s2])); }); best = Math.max(best, q); }
                V2[s] = best;
            }
            hist.push(V2); V = V2;
        }
        const pi = new Int8Array(N).fill(-1);
        for (let s = 0; s < N; s++) {
            if (GW.cells[s] === '#' || GW.term(s)) continue;
            let best = -Infinity, worst = Infinity;
            for (let a = 0; a < 4; a++) { let q = 0; model(s, a).forEach((p, s2) => { q += p * (GW.reward(s2) + (GW.term(s2) ? 0 : gamma * V[s2])); }); worst = Math.min(worst, q); if (q > best + 1e-12) { best = q; pi[s] = a; } }
            if (best - worst < 1e-9) pi[s] = -1;   // 四个动作在模型看来一样好（通常是「哪儿都去不了」）：没有可用的信息，不画箭头
        }
        return { hist, V, pi };
    }
    /** 在模型里（或真实环境里）按策略走一条轨迹 */
    function rollout(model, pi, rand, maxT) {
        let s = GW.start; const path = [s];
        for (let t = 0; t < maxT && !GW.term(s); t++) {
            const a = pi[s] < 0 ? Math.floor(rand() * 4) : pi[s];   // 模型没有偏好的地方：随便走
            let u = rand(), s2 = s; model(s, a).forEach((p, q) => { if (u >= 0 && (u -= p) < 0) s2 = q; });
            s = s2; path.push(s);
        }
        return path;
    }
    // 14：从 n 条随机转移里数出表格模型 P̂(s'|s,a)，在 P̂ 里规划（想象），再到真实环境里检验
    function worldDemo(nData, slip) {
        const rand = LM.rng(4242), N = GW.N, cnt = Array.from({ length: N * 4 }, () => new Map());
        const free = []; for (let s = 0; s < N; s++) if (GW.cells[s] !== '#' && !GW.term(s)) free.push(s);
        const truth = (s, a) => GW.trans(s, a, slip);
        for (let i = 0; i < nData; i++) {
            const s = free[Math.floor(rand() * free.length)], a = Math.floor(rand() * 4);
            let u = rand(), s2 = s; truth(s, a).forEach((p, q) => { if (u >= 0 && (u -= p) < 0) s2 = q; });
            const m = cnt[s * 4 + a]; m.set(s2, (m.get(s2) || 0) + 1);
        }
        let seen = 0; for (const s of free) for (let a = 0; a < 4; a++) if (cnt[s * 4 + a].size) seen++;
        const learned = (s, a) => { const m = cnt[s * 4 + a]; let tot = 0; m.forEach((c) => { tot += c; }); if (!tot) return new Map([[s, 1]]); const out = new Map(); m.forEach((c, q) => out.set(q, c / tot)); return out; };   // 没见过的 (s, a)：模型以为原地不动
        const planHat = valueIteration(learned, 0.95, 40), planTrue = valueIteration(truth, 0.95, 40);
        const success = (pi) => { const r = LM.rng(77); let ok = 0; for (let e = 0; e < 300; e++) { const p = rollout(truth, pi, r, 40); if (GW.cells[p[p.length - 1]] === 'G') ok++; } return ok / 300; };
        return { nData, coverage: seen / (free.length * 4), Vhat: planHat.V, piHat: planHat.pi, Vtrue: planTrue.V, piTrue: planTrue.pi,
            imagined: rollout(learned, planHat.pi, LM.rng(5), 30), real: rollout(truth, planHat.pi, LM.rng(5), 30), succHat: success(planHat.pi), succTrue: success(planTrue.pi) };
    }
    /** 网格世界绘制：V 热图 + 策略箭头 + 墙 / 终点 / 陷阱 + 可选路径 */
    function drawGrid(c, V, pi, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c), top = o.title ? 20 : 4, cs = Math.min((w - 8) / GW.W, (h - top - 4) / GW.H), x0 = (w - cs * GW.W) / 2, rgb = [0, 0, 0];
        if (o.title) { ctx.font = LM.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.titleX ?? 4, 3); }
        for (let s = 0; s < GW.N; s++) {
            const x = x0 + (s % GW.W) * cs, y = top + Math.floor(s / GW.W) * cs, ch = GW.cells[s];
            if (ch === '#') { ctx.fillStyle = '#2a2d33'; ctx.fillRect(x, y, cs, cs); continue; }
            if (ch === 'G') { ctx.fillStyle = '#1f6b3a'; } else if (ch === 'P') { ctx.fillStyle = '#7a2730'; } else { PL.cmap.diverge(0.5 + 0.5 * LM.clamp(V[s], -1, 1), rgb); ctx.fillStyle = LM.rgbStr(rgb); }
            ctx.fillRect(x, y, cs, cs); ctx.strokeStyle = '#050505'; ctx.strokeRect(x, y, cs, cs);
            ctx.font = LM.FONT(11); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            if (ch === 'G' || ch === 'P') { ctx.fillStyle = '#fff'; ctx.fillText(ch === 'G' ? '+1' : '−1', x + cs / 2, y + cs / 2); continue; }
            if (o.values) { ctx.fillStyle = 'rgba(255,255,255,0.75)'; ctx.fillText(V[s].toFixed(2), x + cs / 2, y + cs - 8); }
            if (pi && pi[s] >= 0) {
                const [dx, dy] = GW.A[pi[s]], cx = x + cs / 2, cy = y + cs / 2 - (o.values ? 4 : 0), L = cs * 0.28;
                ctx.strokeStyle = '#f0f0f0'; ctx.fillStyle = '#f0f0f0'; ctx.lineWidth = 1.6;
                ctx.beginPath(); ctx.moveTo(cx - dx * L, cy - dy * L); ctx.lineTo(cx + dx * L, cy + dy * L); ctx.stroke();
                const ax = cx + dx * L, ay = cy + dy * L; ctx.beginPath(); ctx.moveTo(ax + dx * 4, ay + dy * 4); ctx.lineTo(ax - dy * 4, ay + dx * 4); ctx.lineTo(ax + dy * 4, ay - dx * 4); ctx.fill();
            }
            if (s === GW.start) { ctx.fillStyle = LM.C.yellow; ctx.fillText('S', x + 8, y + 9); }
        }
        (o.paths || []).forEach((pa) => {
            ctx.strokeStyle = pa.color; ctx.lineWidth = 3; ctx.globalAlpha = 0.85; ctx.setLineDash(pa.dash || []); ctx.beginPath();
            pa.path.forEach((s, i) => { const px = x0 + (s % GW.W + 0.5) * cs + (pa.off || 0), py = top + (Math.floor(s / GW.W) + 0.5) * cs + (pa.off || 0); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
            ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        });
    }
    // 11：数据分布已知（二维高斯混合）时，扩散的去噪函数 x̂₀(x_t, t) 有闭式解 → 精确的 DDIM（概率流 ODE）反向轨迹
    const GMM = (() => { const mus = []; for (let k = 0; k < 7; k++) { const a = Math.PI / 2 + k * 2 * Math.PI / 7; mus.push([1.6 * Math.cos(a), 1.6 * Math.sin(a)]); } mus.push([0, 0]); return { mus, s: 0.13 }; })();
    const abar = (t) => { const f = (u) => Math.cos((u + 0.008) / 1.008 * Math.PI / 2) ** 2; return Math.max(1e-4, f(t) / f(0)); };   // cosine 日程，t ∈ [0, 1]
    function x0hat(x, y, ab) {
        const a = Math.sqrt(ab), b2 = 1 - ab, s2 = GMM.s * GMM.s, v = a * a * s2 + b2, k = a * s2 / v;
        let wsum = 0, ex = 0, ey = 0;
        const lw = GMM.mus.map(([mx, my]) => -((x - a * mx) ** 2 + (y - a * my) ** 2) / (2 * v)), m = Math.max(...lw);
        GMM.mus.forEach(([mx, my], i) => { const wv = Math.exp(lw[i] - m); wsum += wv; ex += wv * (mx + k * (x - a * mx)); ey += wv * (my + k * (y - a * my)); });
        return [ex / wsum, ey / wsum];
    }
    function diffusionDemo(nP, steps) {
        const rand = LM.rng(1111), traj = new Float32Array(nP * (steps + 1) * 2), data = [];
        for (let i = 0; i < nP; i++) { traj[(i * (steps + 1)) * 2] = LM.randn(rand); traj[(i * (steps + 1)) * 2 + 1] = LM.randn(rand); }
        for (let k = 0; k < steps; k++) {
            const t = 1 - k / steps, t2 = 1 - (k + 1) / steps, ab = abar(t), ab2 = t2 <= 0 ? 1 : abar(t2);
            for (let i = 0; i < nP; i++) {
                const o = (i * (steps + 1) + k) * 2, x = traj[o], y = traj[o + 1], [hx, hy] = x0hat(x, y, ab);
                const ex = (x - Math.sqrt(ab) * hx) / Math.sqrt(1 - ab), ey = (y - Math.sqrt(ab) * hy) / Math.sqrt(1 - ab);
                traj[o + 2] = Math.sqrt(ab2) * hx + Math.sqrt(1 - ab2) * ex; traj[o + 3] = Math.sqrt(ab2) * hy + Math.sqrt(1 - ab2) * ey;
            }
        }
        for (let i = 0; i < 500; i++) { const m = GMM.mus[Math.floor(rand() * GMM.mus.length)]; data.push([m[0] + GMM.s * LM.randn(rand), m[1] + GMM.s * LM.randn(rand)]); }
        return { nP, steps, traj, data };
    }
    // 12：一段 16 帧 × 16×16 的弹球视频（时空张量）
    function videoDemo() {
        const T = 16, S = 16, frames = [];
        let x = 3.2, y = 4.5, vx = 1.35, vy = 0.95; const r = 2.1;
        for (let t = 0; t < T; t++) {
            const f = new Float32Array(S * S);
            for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) f[j * S + i] = LM.clamp(r + 0.5 - Math.hypot(i + 0.5 - x, j + 0.5 - y), 0, 1);
            frames.push(f);
            x += vx; y += vy;
            if (x < r || x > S - r) { vx = -vx; x += 2 * vx; } if (y < r || y > S - r) { vy = -vy; y += 2 * vy; }
        }
        const xt = new Float32Array(T * S);   // x–t 切片：每帧按列取最大值 → 小球横坐标随时间的轨迹
        for (let t = 0; t < T; t++) for (let i = 0; i < S; i++) { let m = 0; for (let j = 0; j < S; j++) m = Math.max(m, frames[t][j * S + i]); xt[t * S + i] = m; }
        return { T, S, frames, xt };
    }
    // 13：合成 1.2 秒音频（8 kHz）：四个带泛音的音符 + 一段啁啾，STFT（Hann 窗 256、跳长 64）
    function audioDemo() {
        const sr = 8000, n = Math.floor(1.2 * sr), x = new Float32Array(n), notes = [523.25, 659.25, 783.99, 1046.5];
        for (let i = 0; i < n; i++) {
            const t = i / sr;
            if (t < 0.96) { const k = Math.floor(t / 0.24), f = notes[k], lt = t - k * 0.24, env = Math.min(1, lt / 0.01) * Math.exp(-lt * 6); x[i] = env * (0.6 * Math.sin(2 * Math.PI * f * t) + 0.25 * Math.sin(4 * Math.PI * f * t) + 0.12 * Math.sin(6 * Math.PI * f * t)); }
            else { const lt = t - 0.96, f0 = 300, f1 = 3200, d = 0.24, ph = 2 * Math.PI * (f0 * lt + (f1 - f0) * lt * lt / (2 * d)); x[i] = 0.5 * Math.sin(ph) * Math.sin(Math.PI * lt / d); }
        }
        const N = 256, hop = 64, nF = Math.floor((n - N) / hop) + 1, nB = N / 2 + 1, spec = new Float32Array(nF * nB), re = new Float64Array(N), im = new Float64Array(N);
        for (let f = 0; f < nF; f++) {
            for (let i = 0; i < N; i++) { re[i] = x[f * hop + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)); im[i] = 0; }
            PL.fft.fft1d(re, im, false);
            for (let b = 0; b < nB; b++) spec[b * nF + f] = Math.hypot(re[b], im[b]);   // 行 = 频率 bin，列 = 帧
        }
        return { sr, x, N, hop, nF, nB, spec };
    }
    // 15：解析体积（3 个彩色高斯雾团）的光线步进体渲染：C = Σ Tᵢ(1 − e^{−σᵢδ})cᵢ
    const BLOBS = [{ c: [0.0, 0.05, 0.0], s: 0.34, d: 10, rgb: [1.0, 0.55, 0.2] }, { c: [0.85, 0.3, 0.25], s: 0.22, d: 16, rgb: [0.3, 0.75, 1.0] }, { c: [-0.7, -0.4, 0.5], s: 0.25, d: 13, rgb: [0.5, 1.0, 0.45] }];
    function density(p) {
        let sig = 0; const col = [0, 0, 0];
        BLOBS.forEach((b) => { const d2 = (p[0] - b.c[0]) ** 2 + (p[1] - b.c[1]) ** 2 + (p[2] - b.c[2]) ** 2, v = b.d * Math.exp(-d2 / (2 * b.s * b.s)); sig += v; col[0] += v * b.rgb[0]; col[1] += v * b.rgb[1]; col[2] += v * b.rgb[2]; });
        if (sig > 1e-9) { col[0] /= sig; col[1] /= sig; col[2] /= sig; }
        return { sig, col };
    }
    function camera(yaw, pitch) { const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), eye = [5.2 * sy * cp, 5.2 * sp, 5.2 * cy * cp], f = PL.v3.norm(PL.v3.mul(eye, -1)), r = PL.v3.norm(PL.v3.cross(f, [0, 1, 0])), u = PL.v3.cross(r, f); return { eye, f, r, u }; }
    function renderVolume(yaw, pitch, R, nS) {
        const cam = camera(yaw, pitch), img = new Float32Array(R * R * 3), tan = Math.tan(19 * Math.PI / 180), t0 = 3.4, t1 = 7.0, dt = (t1 - t0) / nS;
        let center = null;
        for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) {
            const u = ((i + 0.5) / R * 2 - 1) * tan, v = (1 - (j + 0.5) / R * 2) * tan, d = PL.v3.norm([cam.f[0] + u * cam.r[0] + v * cam.u[0], cam.f[1] + u * cam.r[1] + v * cam.u[1], cam.f[2] + u * cam.r[2] + v * cam.u[2]]);
            let T = 1; const C = [0, 0, 0], rec = (i === R >> 1 && j === R >> 1) ? { t: [], T: [], sig: [] } : null;
            for (let k = 0; k < nS; k++) {
                const t = t0 + (k + 0.5) * dt, q = density([cam.eye[0] + t * d[0], cam.eye[1] + t * d[1], cam.eye[2] + t * d[2]]), a = 1 - Math.exp(-q.sig * dt);
                C[0] += T * a * q.col[0]; C[1] += T * a * q.col[1]; C[2] += T * a * q.col[2];
                if (rec) { rec.t.push(t); rec.T.push(T); rec.sig.push(q.sig); }
                T *= 1 - a; if (T < 1e-3 && !rec) break;
            }
            img.set(C, (j * R + i) * 3);
            if (rec) center = rec;
        }
        return { R, img, center };
    }
    return { convDemo, pcaDemo, GW, valueIteration, worldDemo, drawGrid, rollout, GMM, abar, diffusionDemo, videoDemo, audioDemo, renderVolume };
})();

PipelineLab.mount({
    paramSteps: { prompt: [], temp: [], lr: ['nn'], digit: ['cnn'], nData: ['world'], yaw: ['nerf'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '从 Hello World 到大模型与生成式 AI · 系列目录',
    tagColors: { '共享基础': 'green', '通用架构': 'blue', '大语言模型': 'magenta', '分化领域': 'rust' },
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 7,
    css: LM.CSS + `
        .lm-toc { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; margin: 10px 0 4px; }
        .lm-toc a { display: block; padding: 10px 12px; border: 1px solid #22303c; border-radius: 8px; background: #0d141a; color: #b8d6e6; }
        .lm-toc a:hover { background: #142230; text-decoration: none; }
        .lm-toc b { color: #6ab0e0; font-family: var(--pl-mono); margin-right: 6px; }
        .lm-toc span { display: block; color: #7f97a8; font-size: 0.82rem; margin-top: 3px; }
        .lm-part { margin-top: 14px; }
        .lm-part h3 { font-size: 0.95rem; color: #c8ccd4; margin: 0 0 2px; }
        .lm-part h3 span { font-weight: normal; font-size: 0.8rem; color: #6f8496; margin-left: 10px; }
        .lm-typer { display: grid; grid-template-columns: 1.4fr 1fr; gap: 12px; }
        @media (max-width: 760px) { .lm-typer { grid-template-columns: 1fr; } }`,
    regen: false,

    params: [
        { id: 'prompt', label: '开头', options: PIDX.PROMPTS, value: '熊猫', affects: 'frame', desc: '顶部「现场续写」的提示词：迷你 GPT 从这几个字开始一个字一个字往下写。' },
        { id: 'temp', label: '温度', min: 0.2, max: 2, step: 0.1, value: 0.8, affects: 'frame', desc: '采样温度。低于 1 → 更保守、几乎总写对；高于 1.5 → 分布被拉平，开始出现错别字和胡话（第 05 篇详细讲）。' },
        { id: 'lr', label: '学习率', min: 0.002, max: 0.05, step: 0.002, value: 0.02, desc: 'Step 02 里实时训练的小 MLP 的 Adam 学习率。调太大会看到决策边界乱跳。' },
        { id: 'digit', label: '数字', options: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], value: 3, desc: 'Step 08 里拿去做卷积的手写数字（与第 02、08、09 篇同一个生成器）。' },
        { id: 'nData', label: '经验条数', options: [20, 60, 150, 300, 1000, 5000], value: 150, desc: 'Step 14 的世界模型从多少条随机转移里「数」出环境规律。太少 → 模型没见过的地方以为原地不动，在想象里规划出的路线到真实环境就失败。' },
        { id: 'yaw', label: '相机方位', min: 0, max: 360, step: 5, value: 35, unit: '°', map: (d) => d * Math.PI / 180, desc: 'Step 15 体渲染的相机绕场景转的角度（俯仰固定 20°）：每个像素一条光线、沿光线累积密度和颜色。' },
    ],

    stages: [
        {
            id: 'demos', lifetime: 'resident', run(S) {
                S.attn = PIDX.attnDemo(); S.emb = PIDX.embDemo(); S.chat = PIDX.chatDemo(); S.img = PIDX.imageDemo();
                S.spiral = PIDX.spiral(PL.rng(5), 70);
            },
        },
        {
            id: 'linreg', lifetime: 'param', run(S) {
                // 第 01 篇的缩影：40 个点、80 步全量梯度下降
                const r = PL.rng(11), n = 40, x = [], y = [];
                for (let i = 0; i < n; i++) { const xi = 2 * r(); x.push(xi); y.push(2 * xi + 1 + 0.5 * PL.gauss(r)); }
                let w = -1, b = 4; const tr = [], L = [];
                for (let k = 0; k <= 80; k++) {
                    let l = 0, gw = 0, gb = 0;
                    for (let i = 0; i < n; i++) { const e = w * x[i] + b - y[i]; l += e * e / n; gw += 2 * e * x[i] / n; gb += 2 * e / n; }
                    tr.push([w, b]); L.push(l); w -= 0.25 * gw; b -= 0.25 * gb;
                }
                S.lin = { x, y, tr, L };
            },
        },
        {
            id: 'mlp', lifetime: 'history', run(S) {
                // 第 02 篇的缩影：2→32→32→3 MLP，Adam，在螺旋数据上实时训练（每帧 3 步，最多 900 步）
                const { AG } = LM;
                if (S._mlpEpoch !== S.epoch) {
                    const rand = PL.rng(99), init = (s) => (a) => { for (let i = 0; i < a.length; i++) a[i] = s * LM.randn(rand); };
                    const P = { W1: AG.param('W1', 2, 32, init(1.0)), b1: AG.param('b1', 1, 32), W2: AG.param('W2', 32, 32, init(0.25)), b2: AG.param('b2', 1, 32), W3: AG.param('W3', 32, 3, init(0.25)), b3: AG.param('b3', 1, 3) };
                    const n = S.spiral.length, X = AG.tensor(n, 2), Y = new Int32Array(n);
                    S.spiral.forEach((p, i) => { X.data[2 * i] = p.x * 2; X.data[2 * i + 1] = p.y * 2; Y[i] = p.c; });
                    S.net = { P, list: Object.values(P), X, Y, opt: new LM.AdamW(Object.values(P), { lr: S.lr, wd: 0, b2: 0.999 }), step: 0, hist: [], acc: 0 };
                    S._mlpEpoch = S.epoch;
                }
                const N = S.net;
                N.fwd = (X) => { const h1 = AG.tanh(AG.linear(X, N.P.W1, N.P.b1)); const h2 = AG.tanh(AG.linear(h1, N.P.W2, N.P.b2)); return AG.linear(h2, N.P.W3, N.P.b3); };
                for (let k = 0; k < 3 && N.step < 900; k++, N.step++) {
                    N.opt.zeroGrad();
                    const loss = AG.crossEntropy(N.fwd(N.X), N.Y, { keepProbs: true });
                    AG.backward(loss);
                    N.opt.step(S.lr);
                    let ok = 0; const P = loss.probs; for (let i = 0; i < N.Y.length; i++) { let b = 0; for (let c = 1; c < 3; c++) if (P[i * 3 + c] > P[i * 3 + b]) b = c; if (b === N.Y[i]) ok++; }
                    N.acc = ok / N.Y.length; N.hist.push(loss.item());
                }
            },
        },
        {
            id: 'typer', lifetime: 'frame', run(S) {
                // 顶部「现场续写」：每 4 秒写一条新样本，按时间逐字显示
                const slot = Math.floor(S.t / 4), key = `${S.prompt}|${S.temp}|${slot}`;
                if (S._typeKey !== key) {
                    const g = LM.gpt.generate(PIDX.base, PIDX.V.encode(S.prompt), 26, { temperature: S.temp, rand: PL.rng(1000 + slot), keepDists: true });
                    S.typed = { prompt: S.prompt, gen: g.gen, dists: g.dists };
                    S._typeKey = key;
                }
                S.typePos = Math.min(S.typed.gen.length - 1, Math.floor((S.t % 4) * 9));
            },
        },
        {
            id: 'demos2', lifetime: 'resident', run(S) {
                // 第 08–15 篇的缩影里不随参数变的部分
                S.pca = PIDX2.pcaDemo();
                S.vi = PIDX2.valueIteration((s, a) => PIDX2.GW.trans(s, a, 0.2), 0.95, 30);
                S.diff = PIDX2.diffusionDemo(360, 40);
                S.video = PIDX2.videoDemo();
                S.audio = PIDX2.audioDemo();
            },
        },
        { id: 'conv', lifetime: 'param', run(S) { S.conv = PIDX2.convDemo(S.digit); } },
        { id: 'world', lifetime: 'param', run(S) { S.world = PIDX2.worldDemo(S.nData, 0.2); } },
        { id: 'volume', lifetime: 'param', run(S) { S.vol = PIDX2.renderVolume(S.yaw, 20 * Math.PI / 180, 72, 56); } },
    ],

    intro: {
        title: '从机器学习的 Hello World，到大模型与 AI 的各个分支',
        flow: ['共享基础：优化 · 反向传播 · 卷积 · 潜空间 · 强化学习', '通用架构：Token · Transformer', '大语言模型：预训练 · 对齐', '分化领域：多模态 · 扩散 · 视频 · 音频 · 世界模型 · 3D'],
        html: LM.navHtml(null, true) + `
            <p>这是一套<strong>能跑的讲义</strong>：15 篇文章，每篇都是一个离线 HTML，页面里内嵌了算法的 JS 参考实现——损失地形、反向传播的梯度、卷积特征图、VAE 的潜空间、Q 表、BPE 的合并、注意力矩阵、训练曲线、奖励模型、扩散的反向轨迹、生成的视频和音频、世界模型的「想象」、NeRF 和高斯点云，全部是<strong>当场算出来</strong>的，改参数就重算。代码块是对应的 <strong>C++</strong> 实现：只用 C++17 标准库，外加本系列自带的一个小型 C++ 机器学习库（张量、自动微分、优化器、迷你 GPT，源码在 <code>pages/cpp/</code>，可以直接编译，并且和页面里的 JS 实现逐位对得上）。</p>
            <p>15 篇按知识结构分成四层：<strong>共享基础</strong>是所有模型都要用的工具（梯度下降、反向传播、卷积、潜空间、强化学习）；<strong>通用架构</strong>把任何数据变成 token 序列交给 Transformer；<strong>大语言模型</strong>是这套架构规模化后的第一个大成功；<strong>分化领域</strong>是从同一套基础上长出来的各个方向。篇号是写作顺序，可以按下面的分组、也可以按页面底部的「知识依赖图」挑着读。</p>
            <p>贯穿 LLM 部分的玩具是一个 <strong>2.9 万参数的迷你 GPT</strong>，在一个「玩具世界」语料（8 种动物 × 爱吃 / 住处 / 颜色）上训练：它小到能在浏览器里几十秒训完，又大到能展示真实大模型的关键机制——注意力学会回头去找主语、权重里记住世界知识、SFT 教会它按格式回答问题、强化学习（GRPO）让它在没训练过的问题上也答得更准。顶部就是它在现场续写。</p>
            <p class="pl-dim">每篇都是单个离线 HTML，直接用浏览器（推荐 Chrome / Edge）打开即可；带训练的页面打开后训练会自动推进，可以随时在顶栏暂停、调速。</p>
            ${LM.PARTS.map((pt) => `<div class="lm-part"><h3>${pt.name}<span>${pt.desc}</span></h3><div class="lm-toc">${LM.PAGES.filter((p) => p.part === pt.id).map((p) => `<a href="pages/${p.file}"><b>${p.no}</b>${p.title}<span>${PIDX_TOC[p.no] || ''}</span></a>`).join('')}</div></div>`).join('')}`,
    },

    sections: [{
        id: 'typer', title: 'Live Demo · 迷你 GPT 现场续写', position: 'top',
        sub: '左：基座模型从顶栏的「开头」逐字采样（每 4 秒一条新样本）。右：它在<strong>当前这个位置</strong>给下一个字的概率分布——黄色是实际抽中的字。这就是大语言模型推理时每一步在做的事。',
        html: '<div class="lm-typer"><canvas style="width:100%;height:190px"></canvas><canvas style="width:100%;height:190px"></canvas></div>',
        frame(el, S) {
            const [c1, c2] = el.querySelectorAll('canvas'), ty = S.typed, k = S.typePos, V = PIDX.V;
            const segs = [{ t: ty.prompt, color: LM.C.gray }];
            ty.gen.forEach((id, i) => { if (i <= k) segs.push({ t: V.show(id), color: i === k ? '#0a0a0a' : LM.C.text, bg: i === k ? LM.C.yellow : null }); });
            LM.draw.text(c1, [segs, { text: '' }, { text: `温度 ${S.temp} · 第 ${k + 1} 个生成的字 · 模型：${LM.gpt.numParams(PIDX.base).toLocaleString()} 参数，预训练 ${LM_WEIGHTS.base.steps} 步`, color: LM.C.dim, px: 11 }], { px: 20, lineGap: 10 });
            const dist = ty.dists[k], top = LM.topk(dist, 8);
            LM.draw.bars(c2, top.map((i) => dist[i]), top.map((i) => V.show(i)), { horizontal: true, hi: top.indexOf(ty.gen[k]), range: [0, 1], labelPx: 14, title: '下一个字的概率（Top-8）' });
        },
    }],

    steps: [
        {
            id: 'learn', title: '学习 = 用梯度下降最小化损失', tag: '共享基础',
            desc: '所有机器学习模型的训练都是同一个配方：一个带参数的函数（模型）、一个衡量预测有多错的数（损失）、损失对参数的导数（梯度）、沿负梯度一步步走（优化器）。第 01 篇只用两个参数 w、b 把这个配方的每个零件拆开看：损失地形、梯度场、学习率的临界值、条件数、SGD 的噪声、过拟合。<br><a href="pages/01_linear_regression.html">→ 打开第 01 篇：Hello World · 线性回归与梯度下降</a>',
            formula: ['L(w, b) = (1/n) Σ_i (w · x_i + b − y_i)^2', 'θ ← θ − η · ∇_θ L(θ)   // 本系列从头到尾都在重复这一行'],
            code: {
                title: 'C++ · 训练循环的原型', src: `
                // 线性回归的梯度下降：x、y 是 n 个样本，w、b 原地更新
                void train(const std::vector<float>& x, const std::vector<float>& y, float& w, float& b, float lr, int steps) {
                    int n = (int)x.size();
                    for (int s = 0; s < steps; s++) {
                        float gw = 0, gb = 0;
                        for (int i = 0; i < n; i++) {
                            float r = w * x[i] + b - y[i];          // 前向 + 残差
                            gw += 2 * r * x[i] / n;                 // 反向（手推梯度）
                            gb += 2 * r / n;
                        }
                        w -= lr * gw;                               // 更新
                        b -= lr * gb;
                    }
                }`,
            },
            previews: [
                {
                    id: 'linFit', live: true, label: '梯度下降拟合直线', sub: '黄线 = 第 t 步的模型（循环播放 80 步）', size: [360, 220],
                    render: (c, S) => {
                        const k = Math.min(80, Math.floor((S.t * 14) % 110)), [w, b] = S.lin.tr[k];
                        LM.draw.scatter(c, S.lin.x.map((x, i) => ({ x, y: S.lin.y[i], color: LM.C.blue, r: 3 })), { x: [-0.1, 2.1], y: [-1, 7], lines: [{ a: [-0.2, b - 0.2 * w], b: [2.2, b + 2.2 * w], color: LM.C.yellow, width: 2 }], xLabel: `t = ${k}   L = ${S.lin.L[k].toFixed(3)}` });
                    },
                },
                {
                    id: 'linLoss', label: '损失曲线', sub: '对数纵轴', size: [360, 220],
                    render: (c, S) => PL.draw.plot(c, { x: [0, 80], logY: true, xLabel: '迭代步', series: [{ data: S.lin.L, color: LM.C.blue, label: 'MSE' }] }),
                },
            ],
        },
        {
            id: 'nn', title: '神经网络：层层非线性 + 反向传播', tag: '共享基础',
            desc: '线性模型只能画直线。把「线性变换 + 非线性激活」叠几层就得到多层感知机（MLP），它能弯出任意形状的决策边界。代价是梯度要穿过很多层——<strong>反向传播</strong>用链式法则从输出往回逐层计算，一次反传就得到全部参数的梯度。下图是一个 2→32→32→3 的 MLP 正在本页<strong>实时训练</strong>（Adam，用本系列的 JS 自动微分引擎），背景颜色是它当前的决策区域。<br><a href="pages/02_mlp_backprop.html">→ 打开第 02 篇：手写数字 · MLP 与反向传播</a>',
            formula: ['h_1 = tanh(x W_1 + b_1)    h_2 = tanh(h_1 W_2 + b_2)    z = h_2 W_3 + b_3', 'L = −log softmax(z)_y      ∂L/∂z = softmax(z) − onehot(y)   // 反向传播的起点'],
            code: {
                title: 'C++ · 用本系列 C++ 库写的同一个 MLP', src: `
                // 2→32→32→3 的 MLP（tanh 激活）；Linear / AdamW / crossEntropy / backward 见 Common 里的 C++ 库
                struct MLP { Linear l1, l2, l3; };

                MLP makeMLP(Rng& r) {
                    MLP m;
                    m.l1 = makeLinear(2, 32, 1.0f, r);
                    m.l2 = makeLinear(32, 32, 0.25f, r);
                    m.l3 = makeLinear(32, 3, 0.25f, r);
                    return m;
                }

                Var mlpForward(const MLP& m, const Var& X) {
                    Var h1 = tanhAct(forward(m.l1, X));
                    Var h2 = tanhAct(forward(m.l2, h1));
                    return forward(m.l3, h2);                       // logits [n × 3]
                }

                void trainMLP(MLP& m, const Var& X, const std::vector<int>& Y, float lr, int steps) {
                    std::vector<Var> ps = {m.l1.W, m.l1.b, m.l2.W, m.l2.b, m.l3.W, m.l3.b};
                    std::vector<bool> noDecay = std::vector<bool>(ps.size(), false);
                    AdamW opt = makeAdamW(ps, noDecay, lr, 0.f, 0.9f, 0.999f);
                    for (int s = 0; s < steps; s++) {
                        zeroGrad(ps);
                        Var loss = crossEntropy(mlpForward(m, X), Y);   // 前向
                        backward(loss);                                 // 反向传播：一次算出所有参数的梯度
                        adamwStep(opt, lr);                             // 更新
                    }
                }`,
            },
            previews: [
                {
                    id: 'mlpBoundary', live: true, every: 3, label: '决策区域（实时训练中）', sub: '三种颜色 = 三个类别 · 背景 = 模型对每个位置的预测', large: true,
                    render: (c, S) => {
                        const N = S.net, cols = [[106, 176, 224], [224, 160, 96], [90, 208, 122]], R = 40;
                        const grid = LM.AG.tensor(R * R, 2);
                        for (let j = 0; j < R; j++) for (let i = 0; i < R; i++) { grid.data[2 * (j * R + i)] = (-1.2 + 2.4 * (i + 0.5) / R) * 2; grid.data[2 * (j * R + i) + 1] = (1.2 - 2.4 * (j + 0.5) / R) * 2; }
                        const Z = LM.AG.noGrad(() => N.fwd(grid)).data, probs = new Float32Array(3);
                        LM.draw.scatter(c, S.spiral.map((p) => ({ x: p.x, y: p.y, color: LM.rgbStr(cols[p.c]), r: 3, ring: '#000' })), {
                            x: [-1.2, 1.2], y: [-1.2, 1.2], equal: true, bgRes: R, axes: false,
                            bg: (x, y) => { const i = Math.min(R - 1, Math.max(0, Math.floor((x + 1.2) / 2.4 * R))), j = Math.min(R - 1, Math.max(0, Math.floor((1.2 - y) / 2.4 * R))); LM.softmax(Z, probs, 1, (j * R + i) * 3, 3); const k = LM.argmax(probs), conf = probs[k]; return cols[k].map((v) => v * (0.18 + 0.32 * conf)); },
                        });
                        const { ctx, w, h } = PL.draw.ctx(c, { keep: true, smooth: true }), msg = `第 ${N.step} 步 · 训练准确率 ${(N.acc * 100).toFixed(1)}%`;
                        ctx.font = LM.FONT(12, true); const tw = ctx.measureText(msg).width;
                        ctx.fillStyle = 'rgba(5,7,10,0.75)'; ctx.fillRect(w - tw - 14, h - 24, tw + 10, 18); ctx.fillStyle = LM.C.text; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(msg, w - 9, h - 15);
                    },
                },
                {
                    id: 'mlpLoss', live: true, label: '训练损失', sub: '交叉熵（对数纵轴）', size: [360, 200],
                    render: (c, S) => PL.draw.plot(c, { x: [0, 900], logY: true, xLabel: '训练步', series: [{ data: S.net.hist, x: S.net.hist.map((_, i) => i), color: LM.C.orange, label: 'loss' }] }),
                },
            ],
            notes: '拖 <b>学习率</b> 会从头重新训练：0.02 大约两三百步就能把三条螺旋臂分开；拖到 0.05 能看到边界在训练初期剧烈抖动。',
        },
        {
            id: 'tok', title: '文本变成向量：Tokenizer + Embedding', tag: '通用架构',
            desc: '神经网络只吃数字。<strong>Tokenizer</strong> 把文本切成 token 并映射成整数 id（真实模型用字节级 BPE，词表 3 万–25 万；本系列的迷你 GPT 用字符级，词表 76）；<strong>Embedding</strong> 把每个 id 查表变成一个 d 维向量。这些向量是训练出来的——下图是基座模型学到的 embedding 投影到 2D：同一类的字（动物、食物、颜色、虚词）自己聚到了一起，没人告诉过它这些类别。<br><a href="pages/03_tokenizer_embedding.html">→ 打开第 03 篇：Tokenizer、Embedding 与位置编码</a>',
            formula: ['ids = tokenize("熊猫喜欢吃竹子") = [i_1, …, i_T]', 'X = W_te[ids] + W_pe[0..T−1]   // [T × d]：token embedding + 位置 embedding'],
            code: {
                title: 'C++ · 字符级 tokenizer 与 embedding 查表', src: `
                // 字符级 tokenizer：一个汉字 / 标点 = 一个 token（encode / decode 见 Common · minml_gpt.h）
                // embedding 查表 + 位置：X = W_te[ids] + W_pe[0..T−1]
                Var embedTokens(const std::vector<int>& ids, const Var& wte, const Var& wpe) {
                    std::vector<int> pos = ints((int)ids.size());
                    for (int t = 0; t < (int)ids.size(); t++) pos[t] = t;
                    return add(embedding(ids, wte), gatherRows(wpe, pos));
                }`,
            },
            previews: [
                {
                    id: 'tokChips', label: 'Tokenize：字符 → id', sub: '每个方块 = 一个 token，下面是它的 id', size: [360, 120],
                    render: (c, S) => { const s = '熊猫喜欢吃竹子，它住在山里。'; LM.draw.tokens(c, Array.from(PIDX.V.encode(s), (id) => ({ text: PIDX.V.itos[id], id, sub: id })), { showIds: true, px: 15 }); },
                },
                {
                    id: 'embPca', label: '基座模型的 token embedding（PCA 2D）', sub: '颜色 = 类别（只用于上色，训练时模型并不知道）', large: true,
                    render: (c, S) => {
                        const E = S.emb;
                        LM.draw.scatter(c, E.items.map((it, i) => ({ x: E.pts[2 * i], y: E.pts[2 * i + 1], color: it.color, r: 4, label: it.ch, labelPx: 14 })), { equal: true, xLabel: `PC1（${(E.var[0] * 100).toFixed(0)}% 方差）`, yLabel: `PC2（${(E.var[1] * 100).toFixed(0)}%）` });
                        const { ctx } = PL.draw.ctx(c, { keep: true, smooth: true });
                        ctx.font = LM.FONT(11); E.groups.forEach((g, k) => { ctx.fillStyle = g.color; ctx.fillText('● ' + g.name, 40 + k * 78, 14); });
                    },
                },
            ],
        },
        {
            id: 'attn', title: 'Transformer：注意力让每个位置去「看」前文', tag: '通用架构',
            desc: 'Transformer 的核心是<strong>自注意力</strong>：每个位置算出 query，与前面所有位置的 key 做点积打分，softmax 后按权重汇总它们的 value。下图是训练好的迷你 GPT 在句子「熊猫喜欢吃竹子，所以熊猫每天都吃竹子。」上真实的注意力矩阵（程序自动挑出的一个头）：读到第二个「吃」、要预测吃的是什么时，这个头几乎把全部注意力投向了<strong>主语「熊猫」</strong>——它并不是去抄前文那个「竹」（所有头对它的注意力都不超过 ' + PIDX.foodMax.toFixed(2) + '），而是先回头确认「谁在吃」，再从预训练记住的知识（熊猫 → 竹子）里调出答案。注意力负责「去哪里找信息」，记忆在权重里。<br><a href="pages/04_transformer.html">→ 打开第 04 篇：Transformer 逐层拆解</a>',
            formula: ['Attention(Q, K, V) = softmax(Q K^T / √d_h + M) · V   // M = 因果掩码（未来位置 −∞）', 'x ← x + Attn(LN(x))      x ← x + MLP(LN(x))   // 一个 Pre-LN Block'],
            code: {
                title: 'C++ · 单头因果注意力', src: `
                // 单头因果注意力：q、k、v 都是 [T × dh]，返回 [T × dh]
                Tensor causalAttention(const Tensor& q, const Tensor& k, const Tensor& v) {
                    int T = q.rows, dh = q.cols;
                    Tensor out = zeros(T, dh);
                    for (int i = 0; i < T; i++) {
                        std::vector<float> s = floats(i + 1);           // 只看 j ≤ i：因果掩码
                        for (int j = 0; j <= i; j++) {
                            float dot = 0;
                            for (int c = 0; c < dh; c++) dot += q.at(i, c) * k.at(j, c);
                            s[j] = dot / std::sqrt((float)dh);          // 打分
                        }
                        softmaxInPlace(s, 1.0f);                        // 每行和为 1
                        for (int j = 0; j <= i; j++)
                            for (int c = 0; c < dh; c++) out.at(i, c) += s[j] * v.at(j, c);   // 加权求和
                    }
                    return out;
                }`,
            },
            previews: [
                {
                    id: 'attnMap', label: '注意力权重（训练后的迷你 GPT）', sub: '行 = 查询位置，列 = 被看的位置；斜纹 = 因果掩码遮住的未来', large: true, size: [380, 380],
                    render: (c, S) => { const A = S.attn; S._attnLay = LM.draw.heat(c, A.A, A.T, A.T, { rowLabels: A.labels, colLabels: A.labels, map: 'inferno', range: [0, 1], hiRow: A.best.qi, hiCells: [[A.best.qi, A.best.kj]], title: `第 ${A.best.l + 1} 层 · 头 ${A.best.h + 1} · 对主语的注意力 ${(A.best.score * 100).toFixed(0)}%` }); },
                    probe: (x, y, S, PL, info) => {
                        const L = S._attnLay, A = S.attn; if (!L) return '';
                        const cc = Math.floor((info.dx - L.x0) / L.cw), r = Math.floor((info.dy - L.y0) / L.ch);
                        if (r < 0 || cc < 0 || r >= L.R || cc >= L.C || cc > r) return '';
                        return `查询「${A.labels[r]}」(位置 ${r}) → 看「${A.labels[cc]}」(位置 ${cc})
权重 ${A.A[r * L.C + cc].toFixed(3)}`;
                    },
                },
                {
                    id: 'attnNext', label: '于是下一个字……', sub: '黄框那一行（第二个「吃」）处，整个模型输出的下一个 token 分布', size: [360, 170],
                    render: (c, S) => { const n = S.attn.next; LM.draw.bars(c, n.map((q) => q.p), n.map((q) => q.tok), { horizontal: true, hi: 0, range: [0, 1], labelPx: 14 }); },
                },
            ],
        },
        {
            id: 'pretrain', title: '预训练：在海量文本上预测下一个 token', tag: '大语言模型',
            desc: '预训练的目标简单到只有一句话：<strong>预测下一个 token</strong>（自回归语言建模）。随机截取一段文本，输入 x，目标 y 是 x 右移一位；损失是每个位置交叉熵的平均；AdamW + warmup + cosine 学习率跑上几千步（真实模型是上万亿 token）。下图是本系列基座模型真实的离线训练记录：loss 从 ln 76 ≈ 4.33 一路降到 0.51，中间先学会单字频率、再学会词和句式、最后记住每种动物的事实。第 05 篇会在浏览器里从零把它重新训练一遍。<br><a href="pages/05_pretraining.html">→ 打开第 05 篇：预训练一个迷你 GPT</a>',
            formula: ['L = −(1/BT) Σ_{b,t} log p_θ(x_{t+1} | x_{≤t})', 'C ≈ 6 · N · D   // 训练 FLOPs ≈ 6 × 参数量 × token 数'],
            code: {
                title: 'C++ · 预训练主循环（详见第 05 篇）', src: `
                // 预训练主循环：随机窗口 → 前向 → 交叉熵 → 反向 → 梯度裁剪 → AdamW（trainStep 等见 Common · minml_gpt.h）
                void pretrain(GPT& m, const std::vector<int>& data, int steps, int B, int T, float maxLR, Rng& r) {
                    AdamW opt = makeAdamW(m.params, m.decay, maxLR, 0.1f);
                    for (int it = 0; it < steps; it++) {
                        float lr = cosineLR(it, maxLR, maxLR / 10, 100, steps);   // warmup + cosine
                        Batch b = getBatch(data, B, T, r);
                        trainStep(m, opt, b, B, T, lr);
                    }
                }`,
            },
            previews: [
                {
                    id: 'baseCurve', label: '基座模型的训练曲线（离线训练日志）', sub: `batch ${LM_WEIGHTS.base.batch} × T 32 · ${LM_WEIGHTS.base.steps} 步 · 每 50 步记录一次`, large: true, size: [360, 240],
                    render: (c, S) => {
                        const cv = LM_WEIGHTS.base.curve;
                        PL.draw.plot(c, { logY: true, xLabel: '训练步', x: [0, LM_WEIGHTS.base.steps], series: [{ data: cv.map((r) => r[1]), x: cv.map((r) => r[0]), color: LM.C.train, label: 'train loss' }, { data: cv.map((r) => r[2]), x: cv.map((r) => r[0]), color: LM.C.val, label: 'val loss', width: 2 }], hmarks: [{ y: Math.log(76), label: 'ln 76（随机猜）', color: LM.C.gray }] });
                    },
                },
                {
                    id: 'baseSamples', label: '训练完成后的生成样本', sub: '温度 0.7，从四种动物名开头（训练脚本当时生成并记录）', size: [360, 240],
                    render: (c, S) => LM.draw.text(c, LM_WEIGHTS.base.samples.map((s) => ({ text: s, color: LM.C.text })), { px: 13, lineGap: 7 }),
                },
            ],
        },
        {
            id: 'align', title: '后训练：从「续写机器」到「对话助手」', tag: '大语言模型',
            desc: '基座模型只会续写。把问题包进<strong>对话模板</strong> <code>&lt;user&gt;…&lt;bot&gt;</code> 喂给它，它不认识这些特殊 token（预训练里从没出现过），只会把整段当普通文本接着写——下图是它在 <code>&lt;bot&gt;</code> 之后真实的预测：续写出「吃竹子。小兔喜欢吃萝卜。…」，事实没错，但既不是一句完整的回答，也不会停下来。<strong>SFT</strong> 用「问题 → 标准回答」数据微调（只在回答部分计算损失），教会它格式；再用<strong>人类偏好</strong>训练奖励模型，用 PPO / DPO / GRPO 把回答往人更喜欢的方向推，同时用 KL 约束防止它跑偏。<br><a href="pages/06_alignment.html">→ 打开第 06 篇：SFT、RLHF、DPO、GRPO</a>',
            formula: ['L_SFT = −Σ_{t ∈ 回答} log π_θ(y_t | x, y_{<t})   // 问题部分不计损失', 'max_π E_{y~π}[r(x, y)] − β · KL(π ‖ π_ref)      ⇒  π* ∝ π_ref · exp(r / β)'],
            code: {
                title: 'C++ · SFT 的 loss mask', src: `
                // SFT 样本：输入 = 问题 + 回答；目标 = 右移一位，问题部分设为 −1（crossEntropy 会跳过这些位置）
                struct SftSample { std::vector<int> ids, targets; };

                SftSample sftTargets(const std::vector<int>& promptIds, const std::vector<int>& answerIds) {
                    SftSample s;
                    s.ids = promptIds;
                    s.ids.insert(s.ids.end(), answerIds.begin(), answerIds.end());
                    s.targets = ints((int)s.ids.size(), -1);
                    for (int t = (int)promptIds.size() - 1; t + 1 < (int)s.ids.size(); t++) s.targets[t] = s.ids[t + 1];   // 只在回答上算损失
                    return s;
                }`,
            },
            previews: [
                {
                    id: 'chatChips', label: '一条 SFT 样本', sub: '灰色 = 问题（不计损失）· 彩色 = 回答（计算损失）', size: [360, 120],
                    render: (c, S) => {
                        const V = PIDX.V, ch = S.chat, toks = [];
                        ch.pIds.forEach((id) => toks.push({ text: V.show(id), id, dim: true }));
                        ch.aIds.forEach((id) => toks.push({ text: V.show(id), id }));
                        LM.draw.tokens(c, toks, { px: 14 });
                    },
                },
                {
                    id: 'chatBase', label: '基座模型在 ⟨bot⟩ 之后的预测', sub: `贪心续写：「${PIDX.chatDemo().cont.replace(/</g, '⟨').replace(/>/g, '⟩')}」`, size: [360, 190],
                    render: (c, S) => { const t = S.chat.top; LM.draw.bars(c, t.map((q) => q.p), t.map((q) => q.tok), { horizontal: true, range: [0, 1], labelPx: 14, color: LM.C.red }); },
                },
            ],
        },
        {
            id: 'mm', title: '多模态：图像也切成 token', tag: '分化领域',
            desc: '视觉 Transformer（ViT）把图像切成固定大小的 patch，每个 patch 展平后线性投影成一个向量——图像就变成了一串「视觉 token」，可以和文字 token 用同一套 Transformer 处理。<strong>CLIP</strong> 用对比学习把图像和文字映射到同一个向量空间；<strong>LLaVA</strong> 再用一个投影层把视觉 token 接到语言模型前面，让它能「看图说话」。下图把一张 24×24 的小图切成 16 个 6×6 的 patch。<br><a href="pages/07_multimodal.html">→ 打开第 07 篇：ViT、CLIP 与迷你 LLaVA</a>',
            formula: ['patches = reshape(img, [(H/P)·(W/P), P·P·3])      z = patches · W_E + E_pos', 'L_CLIP = ½ [CE(s·I T^T, diag) + CE(s·T I^T, diag)]   // I、T 为 L2 归一化后的图 / 文嵌入'],
            code: {
                title: 'C++ · patchify', src: `
                // 把 H×W×C 的图（行优先、通道交错）切成 P×P 的 patch，每个 patch 展平成一行：[(H/P)·(W/P) × P·P·C]
                Tensor patchify(const std::vector<float>& img, int H, int W, int C, int P) {
                    int nx = W / P, ny = H / P;
                    Tensor out = zeros(nx * ny, P * P * C);
                    for (int py = 0; py < ny; py++)
                        for (int px = 0; px < nx; px++)
                            for (int y = 0; y < P; y++)
                                for (int x = 0; x < P; x++)
                                    for (int c = 0; c < C; c++)
                                        out.at(py * nx + px, (y * P + x) * C + c) = img[((py * P + y) * W + (px * P + x)) * C + c];
                    return out;                                     // 每行就是一个视觉 token
                }`,
            },
            previews: [{
                id: 'patches', label: '一张图 = 16 个视觉 token', sub: '左：原图 · 右：按 patch 拆开（每块就是一个 token，按行优先编号）', size: [360, 180],
                render: (c, S) => {
                    const { ctx, w } = PL.draw.ctx(c), I = S.img, W = I.W, P = I.P, n = W / P, cell = 6;
                    const px = (x, y) => { const o = (y * W + x) * 3; return `rgb(${I.img[o] * 255 | 0},${I.img[o + 1] * 255 | 0},${I.img[o + 2] * 255 | 0})`; };
                    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) { ctx.fillStyle = px(x, y); ctx.fillRect(10 + x * cell, 14 + y * cell, cell, cell); }
                    const ox = 184, gap = 9, pc = 4.4;
                    for (let py = 0; py < n; py++) for (let pxi = 0; pxi < n; pxi++) {
                        const X0 = ox + pxi * (P * pc + gap), Y0 = 14 + py * (P * pc + gap);
                        for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) { ctx.fillStyle = px(pxi * P + x, py * P + y); ctx.fillRect(X0 + x * pc, Y0 + y * pc, pc + 0.3, pc + 0.3); }
                        ctx.strokeStyle = '#2a3038'; ctx.strokeRect(X0 - 0.5, Y0 - 0.5, P * pc + 1, P * pc + 1);
                        ctx.fillStyle = LM.C.gray; ctx.font = LM.FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(String(py * n + pxi), X0 + 1, Y0 + 1);
                    }
                    ctx.strokeStyle = LM.C.gray; for (let k = 1; k < n; k++) { ctx.beginPath(); ctx.moveTo(10 + k * P * cell, 14); ctx.lineTo(10 + k * P * cell, 14 + W * cell); ctx.stroke(); ctx.beginPath(); ctx.moveTo(10, 14 + k * P * cell); ctx.lineTo(10 + W * cell, 14 + k * P * cell); ctx.stroke(); }
                    void w;
                },
            }],
        },
        {
            id: 'cnn', title: '卷积：同一个小卷积核扫过整张图', tag: '共享基础',
            desc: '全连接层让每个输出连到所有像素；<strong>卷积</strong>只看一个小邻域（3×3），并且在所有位置<strong>共享同一组权重</strong>——参数从几万个降到 9 个，图像平移时输出也跟着平移。下面是同一张手写数字分别过 4 个经典卷积核（用本系列共享的 <code>AG.conv2d</code> 真实计算）：每个核都是一个「特征探测器」。CNN 的第一层学到的核和它们很像；把卷积、池化、上采样和跳连堆起来就是 U-Net——扩散模型里的去噪网络。<br><a href="pages/08_cnn_unet.html">→ 打开第 08 篇：卷积神经网络与 U-Net</a>',
            formula: ['y[c_o, i, j] = b[c_o] + Σ_{c_i} Σ_{u,v ∈ 3×3} W[c_o, c_i, u, v] · x[c_i, i + u − 1, j + v − 1]   // 互相关，padding = 1', '参数量：3×3×C_in×C_out（与图像大小无关）   vs   全连接 (H·W·C_in) × (H·W·C_out)'],
            code: {
                title: 'C++ · 一层卷积', src: `
                // 4 个 3×3 卷积核一次作用在一张 16×16 的图上：img [1 × 256]，kernels [4 × 9] → 4 张特征图 [1 × 4·256]
                Var edgeMaps(const Var& img, const Var& kernels) {
                    Var noBias = constant(zeros(1, 4));
                    return conv2d(img, kernels, noBias, conv(1, 16, 16, 3, 1, 1));   // padding = 1：输出与输入同尺寸
                }`,
            },
            previews: [
                { id: 'convIn', label: '输入：一张手写数字', sub: '16×16 灰度（顶栏可换数字）', size: [180, 180], render: (c, S) => PL.draw.field(c, S.conv.img, 16, 16, { range: [0, 1] }) },
                {
                    id: 'convMaps', label: '4 个卷积核的输出（特征图）', sub: '橙 = 正响应 · 蓝 = 负响应 · 每张图同一个 3×3 核扫遍所有位置', large: true, size: [360, 200],
                    render: (c, S) => LM.draw.imageGrid(c, S.conv.maps, 16, 16, { cols: 4, symmetric: true, labels: S.conv.names }),
                },
            ],
        },
        {
            id: 'latent', title: '潜空间：把 256 个像素压成 2 个数', tag: '共享基础',
            desc: '一张 16×16 的数字有 256 个像素，但「所有可能的手写数字」只占这个 256 维空间里很薄的一层。<strong>自编码器</strong>学一个编码器把图压成几个数（潜变量 z）、再学一个解码器从 z 还原。最简单的线性自编码器就是 PCA：下图左边是 600 张数字投影到前两个主成分上（按真实标签上色，PCA 本身没用到标签）：只靠两个<strong>线性</strong>方向，0 和 1、2 和 6 这类差别大的数字分得开一些，大多数数字仍然大面积重叠；右边沿这两个方向在潜空间里取网格、解码回图像。第 09 篇会换成非线性的 AE、能采样的 VAE 和输出离散 token 的 VQ-VAE——它们是潜空间扩散、视频压缩、音频编解码和世界模型的共同基础。<br><a href="pages/09_latent_vae.html">→ 打开第 09 篇：AE、VAE 与 VQ-VAE</a>',
            formula: ['z = W^T (x − μ)      x̂ = μ + W z   // PCA：W = 协方差矩阵的前 k 个特征向量（线性自编码器的最优解）', 'VAE：max E_q[log p(x|z)] − KL(q(z|x) ‖ N(0, I))   // 第 09 篇'],
            code: {
                title: 'C++ · PCA 作为线性自编码器', src: `
                // PCA = 线性自编码器：用幂迭代求协方差矩阵的前 k 个特征向量（第 09 篇换成非线性的 AE / VAE）
                struct PcaCodec { std::vector<float> mean; Tensor W; };   // W: [k × D]，每行一个主方向

                PcaCodec fitPca(const Tensor& X, int k, int iters, Rng& r) {
                    int n = X.rows, D = X.cols;
                    PcaCodec p;
                    p.mean = floats(D);
                    p.W = zeros(k, D);
                    for (int i = 0; i < n; i++) for (int j = 0; j < D; j++) p.mean[j] += X.at(i, j) / n;
                    for (int c = 0; c < k; c++) {
                        std::vector<float> v = floats(D);
                        for (float& a : v) a = randn(r);
                        for (int it = 0; it < iters; it++) {
                            std::vector<float> w = floats(D);           // w = Cov · v = Xcᵀ (Xc · v) / n
                            for (int i = 0; i < n; i++) {
                                float s = 0;
                                for (int j = 0; j < D; j++) s += (X.at(i, j) - p.mean[j]) * v[j];
                                for (int j = 0; j < D; j++) w[j] += (X.at(i, j) - p.mean[j]) * s / n;
                            }
                            for (int q = 0; q < c; q++) {               // 去掉已经找到的方向（正交化）
                                float d = 0;
                                for (int j = 0; j < D; j++) d += w[j] * p.W.at(q, j);
                                for (int j = 0; j < D; j++) w[j] -= d * p.W.at(q, j);
                            }
                            float nrm = 1e-12f;
                            for (float a : w) nrm += a * a;
                            for (int j = 0; j < D; j++) v[j] = w[j] / std::sqrt(nrm);
                        }
                        for (int j = 0; j < D; j++) p.W.at(c, j) = v[j];
                    }
                    return p;
                }

                std::vector<float> encodePca(const PcaCodec& p, const std::vector<float>& x) {   // 256 → k
                    std::vector<float> z = floats(p.W.rows);
                    for (int c = 0; c < p.W.rows; c++) for (int j = 0; j < p.W.cols; j++) z[c] += (x[j] - p.mean[j]) * p.W.at(c, j);
                    return z;
                }

                std::vector<float> decodePca(const PcaCodec& p, const std::vector<float>& z) {   // k → 256
                    std::vector<float> x = p.mean;
                    for (int c = 0; c < p.W.rows; c++) for (int j = 0; j < p.W.cols; j++) x[j] += z[c] * p.W.at(c, j);
                    return x;
                }`,
            },
            previews: [
                {
                    id: 'pcaScatter', label: '600 张数字的潜空间（PC1, PC2）', sub: '颜色 = 真实数字（只用于上色，PCA 本身没用到标签）· 坐标轴上是各方向解释的方差比例', large: true,
                    render: (c, S) => {
                        const P = S.pca, cols = ['#e06c75', '#e0a060', '#e8c25a', '#98c379', '#56b6c2', '#6ab0e0', '#c678dd', '#d19a66', '#abb2bf', '#5ad07a'];
                        const pts = []; for (let i = 0; i < P.n; i++) pts.push({ x: P.pts[2 * i], y: P.pts[2 * i + 1], color: cols[P.y[i]], r: 2.2, alpha: 0.8 });
                        const lab = []; for (let d = 0; d < 10; d++) { let sx = 0, sy = 0, m = 0; for (let i = 0; i < P.n; i++) if (P.y[i] === d) { sx += P.pts[2 * i]; sy += P.pts[2 * i + 1]; m++; } lab.push({ x: sx / m, y: sy / m, color: '#fff', r: 0.1, label: String(d), labelPx: 15, labelColor: cols[d] }); }
                        LM.draw.scatter(c, pts.concat(lab), { equal: true, xLabel: `PC1（${(P.var[0] * 100).toFixed(1)}%）`, yLabel: `PC2（${(P.var[1] * 100).toFixed(1)}%）` });
                    },
                },
                { id: 'pcaGrid', label: '在潜空间网格上解码', sub: '横 = PC1、纵 = PC2，各取 ±2σ：线性解码器只能做像素的加权叠加', size: [220, 220], render: (c, S) => LM.draw.imageGrid(c, S.pca.grid, 16, 16, { cols: S.pca.G, range: [0, 1], gap: 2 }) },
            ],
            notes: '两个主成分一共只解释了约 16% 的方差，网格里的图都是「几个模板的叠加」，所以模糊。同样压到 2 维，非线性自编码器能把数字分得清楚得多、还原得也更像，但它的潜空间里有「洞」——第 09 篇用 VAE 把洞补上。',
        },
        {
            id: 'rl', title: '强化学习：没有标准答案，只有奖励', tag: '共享基础',
            desc: '监督学习有「正确答案」；<strong>强化学习</strong>只有环境给的奖励，而且往往要走很多步之后才知道好坏。核心对象是马尔可夫决策过程（MDP）：状态、动作、转移概率、奖励、折扣 γ。<strong>价值函数</strong> V(s) = 从 s 出发能拿到的折扣回报；贝尔曼方程把它写成递推，反复套用就是<strong>价值迭代</strong>。下面的 6×6 网格世界（每步 −0.04，+1 终点，−1 陷阱，20% 概率滑向两侧）正在真实地做价值迭代：价值从终点一圈圈往外传，箭头是当前的贪心策略。它也是第 06 篇 RLHF（PPO / GRPO）和第 14 篇世界模型的基础。<br><a href="pages/10_reinforcement_learning.html">→ 打开第 10 篇：从价值迭代到策略梯度</a>',
            formula: ['V_{k+1}(s) = max_a Σ_{s′} P(s′ | s, a) · [ r(s′) + γ · V_k(s′) ]   // 贝尔曼最优方程，γ = 0.95', 'π(s) = argmax_a Σ_{s′} P(s′ | s, a) · [ r(s′) + γ · V(s′) ]'],
            code: {
                title: 'C++ · 价值迭代', src: `
                // 有限 MDP：P 展平成 [S·A·S]（P[(s·A + a)·S + s'] = 从 s 做 a 到达 s' 的概率），R[s'] = 到达 s' 的奖励
                struct Mdp { int S = 0, A = 0; std::vector<float> P, R; std::vector<bool> terminal; };

                float qValue(const Mdp& m, const std::vector<float>& V, int s, int a, float gamma) {
                    float q = 0;
                    for (int s2 = 0; s2 < m.S; s2++) {
                        float p = m.P[(s * m.A + a) * m.S + s2];
                        if (p > 0) q += p * (m.R[s2] + (m.terminal[s2] ? 0.f : gamma * V[s2]));
                    }
                    return q;
                }

                std::vector<float> valueIteration(const Mdp& m, float gamma, int iters) {
                    std::vector<float> V = floats(m.S);
                    for (int k = 0; k < iters; k++) {
                        std::vector<float> V2 = floats(m.S);
                        for (int s = 0; s < m.S; s++) {
                            if (m.terminal[s]) continue;
                            float best = -1e30f;
                            for (int a = 0; a < m.A; a++) best = std::max(best, qValue(m, V, s, a, gamma));
                            V2[s] = best;                                // 贝尔曼最优方程
                        }
                        V = V2;
                    }
                    return V;
                }

                // 贪心策略；四个动作看起来一样好时返回 −1（模型没有可用的信息）
                std::vector<int> greedyPolicy(const Mdp& m, const std::vector<float>& V, float gamma) {
                    std::vector<int> pi = ints(m.S, -1);
                    for (int s = 0; s < m.S; s++) {
                        if (m.terminal[s]) continue;
                        float best = -1e30f, worst = 1e30f;
                        for (int a = 0; a < m.A; a++) {
                            float q = qValue(m, V, s, a, gamma);
                            worst = std::min(worst, q);
                            if (q > best + 1e-12f) { best = q; pi[s] = a; }
                        }
                        if (best - worst < 1e-9f) pi[s] = -1;
                    }
                    return pi;
                }`,
            },
            previews: [{
                id: 'viGrid', live: true, label: '价值迭代（实时播放）', sub: '颜色 = V(s)（橙 > 0、蓝 < 0）· 箭头 = 当前贪心策略 · 数字 = V 值', large: true,
                render: (c, S) => {
                    const K = S.vi.hist.length - 1, k = Math.min(K, Math.floor((S.t * 3) % (K + 9))), Vk = S.vi.hist[k];
                    const pi = new Int8Array(PIDX2.GW.N).fill(-1), G = PIDX2.GW;
                    for (let s = 0; s < G.N; s++) { if (G.cells[s] === '#' || G.term(s)) continue; let best = -Infinity; for (let a = 0; a < 4; a++) { let q = 0; G.trans(s, a, 0.2).forEach((p, s2) => { q += p * (G.reward(s2) + (G.term(s2) ? 0 : 0.95 * Vk[s2])); }); if (q > best + 1e-12) { best = q; pi[s] = a; } } }
                    PIDX2.drawGrid(c, Vk, k ? pi : null, { values: true, title: `第 ${k} / ${K} 次迭代`, titleX: 46 });
                },
            }],
        },
        {
            id: 'diff', title: '扩散模型：学会一步步去掉噪声', tag: '分化领域',
            desc: '扩散模型的训练只有一句话：<strong>把数据加上不同强度的高斯噪声，训练网络猜出加了什么噪声</strong>。生成时反过来：从纯噪声出发，每一步用网络估计「干净的数据大概在哪」，往那边挪一点。下面的数据分布是一个已知的二维高斯混合（8 团），所以「干净数据的最佳猜测」 x̂₀(x_t, t) 可以<strong>精确算出来</strong>——这正是神经网络需要学会的那个函数。360 个粒子从标准高斯出发，按 DDIM（确定性的概率流 ODE）走 40 步，汇聚成数据分布。第 11 篇会训练真的去噪网络，再讲 CFG 引导和流匹配；第 12、13 篇把同一套方法用到视频和音频上。<br><a href="pages/11_diffusion.html">→ 打开第 11 篇：扩散模型与流匹配</a>',
            formula: ['x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε      ε ~ N(0, I)   // 前向加噪（cosine 日程）', 'x̂_0 = E[x_0 | x_t]      ε̂ = (x_t − √ᾱ_t x̂_0) / √(1 − ᾱ_t)      x_{t′} = √ᾱ_{t′} x̂_0 + √(1 − ᾱ_{t′}) ε̂   // DDIM 一步'],
            code: {
                title: 'C++ · DDIM 采样（去噪网络预测 ε）', src: `
                // 训练好的去噪网络：输入带噪数据 x_t（每行一个二维点）和噪声水平 t，输出它猜的噪声 ε（实现见第 11 篇）
                Tensor predictNoise(const Tensor& x, float t);

                // cosine 噪声日程 ᾱ(t)，t ∈ [0, 1]：t = 0 是干净数据，t = 1 几乎是纯噪声
                float alphaBar(float t) {
                    float f0 = std::cos(0.008f / 1.008f * 1.5707963f), ft = std::cos((t + 0.008f) / 1.008f * 1.5707963f);
                    return std::max(1e-4f, (ft * ft) / (f0 * f0));
                }

                Tensor ddimSample(int n, int steps, Rng& r) {
                    Tensor x = randn(n, 2, 1.0f, r);                    // 从纯噪声出发
                    for (int k = 0; k < steps; k++) {
                        float t = 1.0f - (float)k / steps, t2 = 1.0f - (float)(k + 1) / steps;
                        float a = alphaBar(t), a2 = t2 <= 0 ? 1.0f : alphaBar(t2);
                        Tensor eps = predictNoise(x, t);                // 网络猜出的噪声
                        for (int i = 0; i < x.size(); i++) {
                            float x0 = (x.v[i] - std::sqrt(1 - a) * eps.v[i]) / std::sqrt(a);   // 推出的干净数据
                            x.v[i] = std::sqrt(a2) * x0 + std::sqrt(1 - a2) * eps.v[i];         // 走到下一个噪声水平
                        }
                    }
                    return x;
                }`,
            },
            previews: [{
                id: 'diffFlow', live: true, label: '从噪声到数据：精确的 DDIM 反向轨迹', sub: '灰 = 真实数据样本 · 彩色点 = 粒子当前位置 · 细线 = 部分粒子走过的路径', large: true,
                render: (c, S) => {
                    const D = S.diff, k = Math.min(D.steps, Math.floor((S.t * 10) % (D.steps + 16))), pts = [], paths = [], st = D.steps + 1;
                    D.data.forEach(([x, y]) => pts.push({ x, y, color: '#3a3f46', r: 1.6 }));
                    for (let i = 0; i < D.nP; i++) {
                        const o = (i * st + k) * 2, hue = (i * 47) % 360;
                        pts.push({ x: D.traj[o], y: D.traj[o + 1], color: `hsl(${hue},70%,62%)`, r: 2.2 });
                        if (i % 12 === 0) { const pp = []; for (let q = 0; q <= k; q++) pp.push([D.traj[(i * st + q) * 2], D.traj[(i * st + q) * 2 + 1]]); paths.push({ pts: pp, color: `hsla(${hue},70%,62%,0.55)`, width: 1 }); }
                    }
                    LM.draw.scatter(c, pts, { x: [-3.2, 3.2], y: [-3.2, 3.2], equal: true, paths, axes: false, title: `第 ${k} / ${D.steps} 步 · 噪声水平 1 − ᾱ = ${(1 - (k >= D.steps ? 1 : PIDX2.abar(1 - k / D.steps))).toFixed(3)}`, titleX: 46 });
                },
            }],
            warn: '这里的 x̂₀ 是利用「数据分布已知」写出的闭式解，不是训练出来的；真实数据（图片、视频、声音）的分布没人知道，只能让神经网络从样本里学这个函数——那正是第 11 篇的内容。',
        },
        {
            id: 'video', title: '视频生成：时间也是一个维度', tag: '分化领域',
            desc: '视频是一个时空张量 [T × H × W × C]。逐帧独立生成会闪烁、物体会跳——模型必须同时看<strong>时间方向</strong>：时空 patch（tubelet）、时间注意力、在压缩过的潜空间里去噪。下面是一段 16 帧的弹球视频：左边逐帧展开，右边把每帧在竖直方向取最大值、按时间往下摞起来（x–t 切片）——匀速运动变成斜线，弹墙变成折线。视频模型生成的东西，在这张图上必须同样「连得上」。<br><a href="pages/12_video_generation.html">→ 打开第 12 篇：时空扩散与时间一致性</a>',
            formula: ['X ∈ R^{T × H × W × C}      tubelet = 一块 t_p × p × p 的时空小方块 → 1 个 token', 'token 数 = (T/t_p) · (H/p) · (W/p)      注意力代价 ∝ token 数^2'],
            code: {
                title: 'C++ · 把视频切成时空 patch', src: `
                // 视频 [T × H × W × C]（交错存储）切成 tp×p×p 的时空小块，每块展平成一个 token：[token 数 × tp·p·p·C]
                Tensor tubelets(const std::vector<float>& video, int T, int H, int W, int C, int tp, int p) {
                    int nt = T / tp, ny = H / p, nx = W / p;
                    Tensor out = zeros(nt * ny * nx, tp * p * p * C);
                    for (int it = 0; it < nt; it++)
                        for (int iy = 0; iy < ny; iy++)
                            for (int ix = 0; ix < nx; ix++) {
                                int row = (it * ny + iy) * nx + ix, col = 0;
                                for (int t = 0; t < tp; t++)
                                    for (int y = 0; y < p; y++)
                                        for (int x = 0; x < p; x++)
                                            for (int c = 0; c < C; c++)
                                                out.at(row, col++) = video[(((it * tp + t) * H + iy * p + y) * W + ix * p + x) * C + c];
                            }
                    return out;
                }`,
            },
            previews: [
                {
                    id: 'vidFrames', live: true, label: '16 帧弹球视频', sub: '黄框 = 当前帧（循环播放）', large: true, size: [360, 200],
                    render: (c, S) => { const V = S.video, k = Math.floor(S.t * 8) % V.T; LM.draw.imageGrid(c, V.frames, V.S, V.S, { cols: 8, range: [0, 1], gap: 3, outline: V.frames.map((_, i) => (i === k ? LM.C.yellow : null)) }); },
                },
                { id: 'vidXT', label: 'x–t 切片：运动轨迹', sub: '行 = 时间（向下）· 列 = 横坐标', size: [180, 180], render: (c, S) => PL.draw.field(c, S.video.xt, S.video.S, S.video.T, { range: [0, 1], map: 'inferno' }) },
            ],
        },
        {
            id: 'audio', title: '音频：从波形到频谱，再到 token', tag: '分化领域',
            desc: '声音是一条一维波形（这里每秒 8000 个采样点）。直接建模波形太长，所以先做<strong>短时傅里叶变换</strong>：每 64 个采样点取一个 256 点的窗、做一次 FFT，得到「哪个时刻有哪些频率」的频谱图。下面是页面里合成的 1.2 秒声音的真实频谱：四个音符各有基频和两条泛音（水平线），最后一段啁啾是一条往上扫的斜线。生成音频的模型要么在频谱上做扩散，要么先用神经编解码器把音频压成离散 token、再用语言模型逐个生成（第 13 篇）。<br><a href="pages/13_audio_generation.html">→ 打开第 13 篇：频谱、音频 token 与生成</a>',
            formula: ['X[m, k] = Σ_n x[n + m·hop] · w[n] · e^{−2πi·k·n/N}   // STFT：N = 256，hop = 64，Hann 窗', '频率分辨率 = sr / N = 31.25 Hz      时间分辨率 = hop / sr = 8 ms'],
            code: {
                title: 'C++ · 频谱图（含一个手写的 FFT）', src: `
                // 原地基 2 FFT（Cooley–Tukey），a.size() 必须是 2 的幂
                void fft(std::vector<std::complex<float>>& a) {
                    int n = (int)a.size();
                    for (int i = 1, j = 0; i < n; i++) {                // 位反转重排
                        int bit = n >> 1;
                        for (; j & bit; bit >>= 1) j ^= bit;
                        j ^= bit;
                        if (i < j) std::swap(a[i], a[j]);
                    }
                    for (int len = 2; len <= n; len <<= 1) {            // log₂n 级蝶形
                        std::complex<float> wl = std::polar(1.0f, -6.2831853f / len);
                        for (int i = 0; i < n; i += len) {
                            std::complex<float> w = 1.0f;
                            for (int j = 0; j < len / 2; j++) {
                                std::complex<float> u = a[i + j], v = a[i + j + len / 2] * w;
                                a[i + j] = u + v;
                                a[i + j + len / 2] = u - v;
                                w *= wl;
                            }
                        }
                    }
                }

                // 幅度谱图：每 hop 个采样点取一个 nFft 点的 Hann 窗做 FFT → [nFft/2+1 个频率 × 帧数]
                Tensor spectrogram(const std::vector<float>& x, int nFft, int hop) {
                    int frames = ((int)x.size() - nFft) / hop + 1, bins = nFft / 2 + 1;
                    Tensor S = zeros(bins, frames);
                    std::vector<std::complex<float>> buf;
                    buf.resize(nFft);
                    for (int f = 0; f < frames; f++) {
                        for (int i = 0; i < nFft; i++) buf[i] = x[f * hop + i] * (0.5f - 0.5f * std::cos(6.2831853f * i / nFft));   // Hann 窗
                        fft(buf);
                        for (int b = 0; b < bins; b++) S.at(b, f) = std::abs(buf[b]);
                    }
                    return S;
                }`,
            },
            previews: [
                { id: 'audWave', label: '波形', sub: '1.2 秒、8 kHz 采样', size: [360, 150], render: (c, S) => { const A = S.audio, n = A.x.length, d = [], xs = []; for (let i = 0; i < n; i += 4) { d.push(A.x[i]); xs.push(i / A.sr); } PL.draw.plot(c, { x: [0, n / A.sr], y: [-1, 1], xLabel: '时间 (s)', series: [{ data: d, x: xs, color: LM.C.cyan, width: 1 }] }); } },
                {
                    id: 'audSpec', label: '频谱图（dB）', sub: '横 = 时间 · 纵 = 频率（0–4 kHz，下低上高）· 60 dB 动态范围', large: true, size: [360, 220],
                    render: (c, S) => PL.draw.field(c, S.audio.spec, S.audio.nF, S.audio.nB, { stretch: 'db', dbScale: 20, dbRange: 60, map: 'inferno', flipY: true }),
                    probe: (x, y, S) => { const A = S.audio; return `t = ${(x * A.hop / A.sr).toFixed(3)} s\nf ≈ ${((A.nB - 1 - y) * A.sr / A.N).toFixed(0)} Hz`; },
                },
            ],
        },
        {
            id: 'world', title: '世界模型：先学会环境，再在想象里规划', tag: '分化领域',
            desc: '无模型的强化学习要在真实环境里试错成千上万次。<strong>世界模型</strong>先从经验里学「做了 a 之后世界会变成什么样」，再在这个模型里预演、规划——就像人在脑子里先想一遍。下面用最简单的世界模型演示：在 Step 10 的网格世界里随机收集 N 条转移，<strong>数出</strong>转移概率 P̂(s′|s, a)，然后完全在 P̂ 里做价值迭代（想象），最后把规划出的策略拿到真实环境里检验。经验太少时，模型没见过的地方以为「原地不动」，想象中的路线和真实走出的路线就会分叉。第 14 篇换成从像素学习的神经网络世界模型，还能让你在它的「想象」里亲手玩。<br><a href="pages/14_world_models.html">→ 打开第 14 篇：学会模拟环境，在想象中规划</a>',
            formula: ['P̂(s′ | s, a) = N(s, a, s′) / N(s, a)   // 从经验里计数（没见过的 (s, a)：假设原地不动）', 'π̂ = 价值迭代(P̂)      成功率 = 在真实环境 P 里执行 π̂ 到达 +1 的比例'],
            code: {
                title: 'C++ · 表格世界模型 + 想象中规划', src: `
                // 一条经验：在状态 s 做动作 a，到了 s2
                struct Transition { int s, a, s2; };

                // 从经验里数出转移概率 P̂；没见过的 (s, a) 假设原地不动。known 提供状态数、动作数、奖励和终止状态（假设已知）
                Mdp learnModel(const std::vector<Transition>& data, const Mdp& known) {
                    Mdp m = known;
                    std::vector<float> cnt = floats(m.S * m.A * m.S), tot = floats(m.S * m.A);
                    for (const Transition& t : data) { cnt[(t.s * m.A + t.a) * m.S + t.s2] += 1; tot[t.s * m.A + t.a] += 1; }
                    for (int s = 0; s < m.S; s++)
                        for (int a = 0; a < m.A; a++)
                            for (int s2 = 0; s2 < m.S; s2++) {
                                float n = tot[s * m.A + a];
                                m.P[(s * m.A + a) * m.S + s2] = n > 0 ? cnt[(s * m.A + a) * m.S + s2] / n : (s2 == s ? 1.f : 0.f);
                            }
                    return m;
                }

                // 在想象里规划：在学到的模型上做价值迭代（见 Step 10），得到的策略再拿到真实环境里检验
                std::vector<int> planInImagination(const std::vector<Transition>& data, const Mdp& known, float gamma) {
                    Mdp model = learnModel(data, known);
                    std::vector<float> V = valueIteration(model, gamma, 40);
                    return greedyPolicy(model, V, gamma);
                }`,
            },
            previews: [
                {
                    id: 'wmGrid', label: '在学到的模型里规划', sub: '颜色 = 模型以为的价值 V̂ · 箭头 = 规划出的策略 · 橙线 = 模型里想象的轨迹 · 青线 = 同一策略在真实环境里走出的轨迹', large: true,
                    render: (c, S) => { const W = S.world; PIDX2.drawGrid(c, W.Vhat, W.piHat, { paths: [{ path: W.imagined, color: LM.C.orange, off: -3 }, { path: W.real, color: LM.C.cyan, off: 3, dash: [5, 3] }], title: `经验 ${W.nData} 条 · 覆盖 ${(W.coverage * 100).toFixed(0)}% 的 (s, a)` }); },
                },
                {
                    id: 'wmStats', label: '想象 vs 现实', sub: '300 局真实环境测试（每局最多 40 步）', size: [360, 150],
                    render: (c, S) => { const W = S.world; LM.draw.bars(c, [W.succHat, W.succTrue], ['想象中规划的策略', '已知真实模型的最优策略'], { horizontal: true, range: [0, 1], colors: [LM.C.orange, LM.C.green], labelPx: 12 }); },
                },
            ],
            notes: '把顶栏 <b>经验条数</b> 从 20 往上拖：默认 150 条时覆盖了 85% 的 (s, a)，但通往终点的瓶颈处恰好有转移没见过，模型以为左半边是死路（没有箭头的格子 = 模型看来四个动作一样，只能乱走），真实成功率只有 35%；300 条时补上了缺口，成功率 97%。<b>1000 条时反而掉到 79%</b>：估计出的滑动概率有误差，规划器以为沿底边紧贴陷阱的捷径更划算，300 局里有 63 局掉进陷阱——规划器会钻模型误差的空子，这正是第 14 篇要讲的「模型被利用」。5000 条时与最优策略完全一致。',
        },
        {
            id: 'nerf', title: '3D 重建：沿光线积分出一张图', tag: '分化领域',
            desc: '<strong>NeRF</strong> 把场景表示成一个函数：空间里每个点的密度 σ 和颜色 c；渲染一个像素就是沿相机光线采样、按透射率把颜色累加起来（体渲染）。训练时让渲染出的图和照片一致，梯度沿着这条积分一路传回场景表示。<strong>3D Gaussian Splatting</strong> 把场景换成几十万个三维高斯，直接投影到屏幕上排序混合，快到可以实时。下面是对一个已知体积（三团彩色高斯雾）做的真实光线步进渲染（72×72 条光线、每条 56 个采样点），右边是画面中心那条光线上的密度和透射率。第 15 篇会从多视角照片里把这样的场景反推出来。<br><a href="pages/15_nerf_gaussian_splatting.html">→ 打开第 15 篇：NeRF 与 3D Gaussian Splatting</a>',
            formula: ['C(r) = Σ_i T_i · (1 − e^{−σ_i δ_i}) · c_i      T_i = exp(−Σ_{j<i} σ_j δ_j)   // 离散体渲染（emission–absorption）', 'r(t) = o + t · d   // 每个像素一条光线：o = 相机位置，d = 视线方向'],
            code: {
                title: 'C++ · 沿一条光线做体渲染', src: `
                // 场景表示：给一个空间点，返回密度 σ 和颜色（NeRF 里它是一个 MLP；实现见第 15 篇）
                struct Radiance { float sigma; float rgb[3]; };
                Radiance queryField(const float p[3]);

                // 沿光线 o + t·d 做体渲染：C = Σ Tᵢ (1 − e^{−σᵢδ}) cᵢ
                void renderRay(const float o[3], const float d[3], float t0, float t1, int n, float rgbOut[3]) {
                    float delta = (t1 - t0) / n, T = 1.0f;              // T = 透射率：光线还剩多少没被「吃掉」
                    rgbOut[0] = rgbOut[1] = rgbOut[2] = 0;
                    for (int i = 0; i < n; i++) {
                        float t = t0 + (i + 0.5f) * delta, p[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
                        Radiance q = queryField(p);
                        float alpha = 1 - std::exp(-q.sigma * delta);
                        for (int c = 0; c < 3; c++) rgbOut[c] += T * alpha * q.rgb[c];
                        T *= 1 - alpha;
                    }
                }`,
            },
            previews: [
                { id: 'volImg', label: '光线步进体渲染', sub: '顶栏「相机方位」绕场景旋转', large: true, render: (c, S) => { const V = S.vol; PL.draw.rgb(c, V.img, V.R, V.R, { stride: 3, encode: 'srgb' }); } },
                {
                    id: 'volRay', label: '中心光线上的 σ(t) 与透射率 T(t)', sub: 'T 从 1 开始下降：光线被雾「吃掉」的比例', size: [360, 190],
                    render: (c, S) => { const R = S.vol.center; const m = Math.max(...R.sig) || 1; PL.draw.plot(c, { x: [R.t[0], R.t[R.t.length - 1]], y: [0, 1.05], xLabel: '沿光线的距离 t', series: [{ data: R.sig.map((v) => v / m), x: R.t, color: LM.C.orange, label: `σ(t) / ${m.toFixed(1)}`, fill: true }, { data: R.T, x: R.t, color: LM.C.cyan, label: 'T(t)', width: 2 }] }); },
                },
            ],
        },
    ],

    graph: {
        title: '知识依赖图 · 从共享基础到分化领域',
        intro: '节点 = 一篇讲义，标题颜色 = 所在的层（<span style="color:#5ad07a">共享基础</span> · <span style="color:#6ab0e0">通用架构</span> · <span style="color:#c678dd">大语言模型</span> · <span style="color:#e0a060">分化领域</span>）；右侧的 pin = 这一篇<b>交出去</b>的核心概念，左侧的 pin = 它<b>用到</b>的前置概念，同名概念自动连线。想读某个分化领域时，顺着连线往左找它依赖的基础就行。点击节点看对应的缩影。',
        categories: {
            base: { color: '#2f7a48', label: '共享基础 · 所有模型都要用的工具' },
            arch: { color: '#2f5f8f', label: '通用架构 · 一切皆 token + Transformer' },
            llm: { color: '#7a3f8f', label: '大语言模型 · 预训练与对齐' },
            domain: { color: '#9a6428', label: '分化领域 · 在基础之上长出的方向' },
        },
        types: {
            base: { color: '#5ad07a', label: '共享基础交出的概念' },
            arch: { color: '#6ab0e0', label: '通用架构交出的概念' },
            llm: { color: '#c678dd', label: '大语言模型交出的概念' },
            domain: { color: '#e0a060', label: '分化领域交出的概念' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）', open: false,
            blocks: LM.CPP.blocks(['gpt']),
        },
        nodes: [
            { id: 'c01', title: '01 线性回归', tag: '01', category: 'base', step: 'learn', desc: '损失 · 梯度 · 学习率 · SGD · 泛化', outputs: [{ name: '梯度下降', format: 'θ ← θ − η∇L', type: 'base' }], detail: { preview: 'linFit' } },
            { id: 'c02', title: '02 MLP · 反向传播', tag: '02', category: 'base', step: 'nn', desc: '交叉熵 · 链式法则 · Adam · 初始化', inputs: [{ name: '梯度下降', format: '优化', type: 'base' }], outputs: [{ name: '反向传播', format: '自动微分', type: 'base' }, { name: 'MLP', format: '非线性层', type: 'base' }], detail: { preview: 'mlpBoundary' } },
            { id: 'c08', title: '08 CNN · U-Net', tag: '08', category: 'base', step: 'cnn', desc: '卷积 · 感受野 · 跳连', inputs: [{ name: '反向传播', format: '训练', type: 'base' }], outputs: [{ name: '卷积 · U-Net', format: '图像特征 / 去噪网络', type: 'base' }], detail: { preview: 'convMaps' } },
            { id: 'c09', title: '09 AE · VAE · VQ', tag: '09', category: 'base', step: 'latent', desc: '编码 / 解码 · ELBO · 码本', inputs: [{ name: 'MLP', format: '编码器 / 解码器', type: 'base' }], outputs: [{ name: '潜空间', format: 'z ∈ R^k', type: 'base' }, { name: '离散 token（VQ）', format: '码本索引', type: 'base' }], detail: { preview: 'pcaScatter' } },
            { id: 'c10', title: '10 强化学习', tag: '10', category: 'base', step: 'rl', desc: 'MDP · 价值迭代 · Q-learning · 策略梯度', inputs: [{ name: '梯度下降', format: '策略优化', type: 'base' }], outputs: [{ name: '策略梯度 · PPO', format: '∇ log π · A', type: 'base' }, { name: '价值与规划', format: 'V(s) · Q(s, a)', type: 'base' }], detail: { preview: 'viGrid' } },
            { id: 'c03', title: '03 Tokenizer · Embedding', tag: '03', category: 'arch', step: 'tok', desc: 'BPE · 词向量 · 位置编码', inputs: [{ name: '反向传播', format: '训练 embedding', type: 'base' }], outputs: [{ name: 'Token · Embedding', format: 'ids → 向量', type: 'arch' }, { name: '位置编码', format: '正弦 / RoPE', type: 'arch' }], detail: { preview: 'tokChips' } },
            { id: 'c04', title: '04 Transformer', tag: '04', category: 'arch', step: 'attn', desc: '注意力 · 残差流 · FFN · KV cache', inputs: [{ name: 'Token · Embedding', format: '输入序列', type: 'arch' }, { name: '位置编码', format: '顺序信息', type: 'arch' }, { name: 'MLP', format: 'FFN', type: 'base' }], outputs: [{ name: 'Transformer', format: '序列 → 序列', type: 'arch' }], detail: { preview: 'attnMap' } },
            { id: 'c05', title: '05 预训练', tag: '05', category: 'llm', step: 'pretrain', desc: '下一个 token · AdamW · 采样 · scaling law', inputs: [{ name: 'Transformer', format: '架构', type: 'arch' }], outputs: [{ name: '基座模型', format: '世界知识', type: 'llm' }], detail: { preview: 'baseCurve' } },
            { id: 'c06', title: '06 SFT · RLHF · DPO', tag: '06', category: 'llm', step: 'align', desc: '对话模板 · 奖励模型 · PPO / DPO / GRPO', inputs: [{ name: '基座模型', format: '起点', type: 'llm' }, { name: '策略梯度 · PPO', format: 'RLHF / GRPO', type: 'base' }], outputs: [{ name: '对话模型', format: '会按格式回答', type: 'llm' }], detail: { preview: 'chatBase' } },
            { id: 'c07', title: '07 ViT · CLIP · LLaVA', tag: '07', category: 'domain', step: 'mm', desc: 'patch token · 对比学习 · 视觉接入 LLM', inputs: [{ name: 'Transformer', format: '视觉编码器', type: 'arch' }, { name: '对话模型', format: '语言端', type: 'llm' }], outputs: [{ name: '图像 token', format: 'patch → 向量', type: 'domain' }], detail: { preview: 'patches' } },
            { id: 'c11', title: '11 扩散 · 流匹配', tag: '11', category: 'domain', step: 'diff', desc: '加噪 · 去噪 · DDIM · CFG · 流匹配', inputs: [{ name: '卷积 · U-Net', format: '去噪网络', type: 'base' }, { name: '潜空间', format: '潜空间扩散', type: 'base' }], outputs: [{ name: '扩散生成', format: '噪声 → 数据', type: 'domain' }], detail: { preview: 'diffFlow' } },
            { id: 'c12', title: '12 视频生成', tag: '12', category: 'domain', step: 'video', desc: '时空 patch · 时间一致性 · 图生视频', inputs: [{ name: '扩散生成', format: '生成框架', type: 'domain' }, { name: '图像 token', format: '时空 patch', type: 'domain' }, { name: 'Transformer', format: '时空注意力', type: 'arch' }], outputs: [{ name: '视频生成', format: '[T, H, W, C]', type: 'domain' }], detail: { preview: 'vidFrames' } },
            { id: 'c13', title: '13 音频 · 语音', tag: '13', category: 'domain', step: 'audio', desc: '频谱 · RVQ codec · 音频 token LM · TTS / ASR', inputs: [{ name: '离散 token（VQ）', format: 'RVQ', type: 'base' }, { name: 'Transformer', format: 'token 语言模型', type: 'arch' }, { name: '扩散生成', format: '频谱扩散', type: 'domain' }], outputs: [{ name: '音频生成', format: '波形 / token', type: 'domain' }], detail: { preview: 'audSpec' } },
            { id: 'c14', title: '14 世界模型', tag: '14', category: 'domain', step: 'world', desc: '编码器 · 动力学 · 想象 · 规划', inputs: [{ name: '价值与规划', format: '规划 / 策略', type: 'base' }, { name: '潜空间', format: '观测编码', type: 'base' }, { name: '视频生成', format: '动作条件视频', type: 'domain' }], outputs: [{ name: '世界模型', format: 'p(o′ | o, a)', type: 'domain' }], detail: { preview: 'wmGrid' } },
            { id: 'c15', title: '15 NeRF · 3DGS', tag: '15', category: 'domain', step: 'nerf', desc: '体渲染 · 位置编码 · 高斯泼溅', inputs: [{ name: '位置编码', format: 'Fourier features', type: 'arch' }, { name: 'MLP', format: '辐射场', type: 'base' }], outputs: [{ name: '3D 表示', format: '辐射场 / 高斯', type: 'domain' }], detail: { preview: 'volImg' } },
        ],
    },

    references: [
        { title: 'Attention Is All You Need', author: 'A. Vaswani et al.', venue: 'NeurIPS', year: 2017, note: 'Transformer（第 04 篇）' },
        { title: 'Language Models are Unsupervised Multitask Learners', author: 'A. Radford et al.（OpenAI）', year: 2019, note: 'GPT-2：本系列迷你 GPT 的结构原型（第 04、05 篇）' },
        { title: 'Language Models are Few-Shot Learners', author: 'T. Brown et al.', venue: 'NeurIPS', year: 2020, note: 'GPT-3：规模化与上下文学习' },
        { title: 'Training Compute-Optimal Large Language Models', author: 'J. Hoffmann et al.', venue: 'NeurIPS', year: 2022, note: 'Chinchilla scaling law（第 05 篇）' },
        { title: 'Training language models to follow instructions with human feedback', author: 'L. Ouyang et al.', venue: 'NeurIPS', year: 2022, note: 'InstructGPT：SFT → RM → PPO（第 06 篇）' },
        { title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model', author: 'R. Rafailov et al.', venue: 'NeurIPS', year: 2023, note: 'DPO（第 06 篇）' },
        { title: 'Learning Transferable Visual Models From Natural Language Supervision', author: 'A. Radford et al.', venue: 'ICML', year: 2021, note: 'CLIP（第 07 篇）' },
        { title: 'Visual Instruction Tuning', author: 'H. Liu, C. Li, Q. Wu, Y. J. Lee', venue: 'NeurIPS', year: 2023, note: 'LLaVA（第 07 篇）' },
        { title: 'nanoGPT', author: 'A. Karpathy', venue: 'GitHub', year: 2023, url: 'https://github.com/karpathy/nanoGPT', note: '本系列迷你 GPT 的 PyTorch 代码风格参考' },
        { title: 'Gradient-Based Learning Applied to Document Recognition', author: 'Y. LeCun, L. Bottou, Y. Bengio, P. Haffner', venue: 'Proceedings of the IEEE', year: 1998, note: 'LeNet：卷积网络（第 08 篇）' },
        { title: 'U-Net: Convolutional Networks for Biomedical Image Segmentation', author: 'O. Ronneberger, P. Fischer, T. Brox', venue: 'MICCAI', year: 2015, note: 'U-Net（第 08、11 篇）' },
        { title: 'Auto-Encoding Variational Bayes', author: 'D. P. Kingma, M. Welling', venue: 'ICLR', year: 2014, note: 'VAE（第 09 篇）' },
        { title: 'Neural Discrete Representation Learning', author: 'A. van den Oord, O. Vinyals, K. Kavukcuoglu', venue: 'NeurIPS', year: 2017, note: 'VQ-VAE（第 09、13 篇）' },
        { title: 'Reinforcement Learning: An Introduction（第 2 版）', author: 'R. S. Sutton, A. G. Barto', venue: 'MIT Press', year: 2018, note: '强化学习教材（第 10 篇）' },
        { title: 'Denoising Diffusion Probabilistic Models', author: 'J. Ho, A. Jain, P. Abbeel', venue: 'NeurIPS', year: 2020, note: 'DDPM（第 11 篇）' },
        { title: 'Flow Matching for Generative Modeling', author: 'Y. Lipman et al.', venue: 'ICLR', year: 2023, note: '流匹配（第 11 篇）' },
        { title: 'High-Resolution Image Synthesis with Latent Diffusion Models', author: 'R. Rombach et al.', venue: 'CVPR', year: 2022, note: '潜空间扩散 / Stable Diffusion（第 11、12 篇）' },
        { title: 'World Models', author: 'D. Ha, J. Schmidhuber', venue: 'arXiv / NeurIPS 2018', year: 2018, note: 'V + M + C 世界模型（第 14 篇）' },
        { title: 'NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis', author: 'B. Mildenhall et al.', venue: 'ECCV', year: 2020, note: 'NeRF（第 15 篇）' },
        { title: '3D Gaussian Splatting for Real-Time Radiance Field Rendering', author: 'B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis', venue: 'ACM TOG (SIGGRAPH)', year: 2023, note: '3DGS（第 15 篇）' },
    ],
});
