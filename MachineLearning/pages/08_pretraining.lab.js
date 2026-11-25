// 08_pretraining.lab.js — 系列第 08 篇：预训练（在浏览器里从零训练一个迷你 GPT）
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js llm_weights.js 08_pretraining.lab.js -o 08_pretraining.html
// history stage 从随机初始化开始实时训练 LM.gpt（与第 07 篇的基座模型同构）。每张预览都画这次训练真实算出的数据：
// batch、逐位置 loss、梯度范数、AdamW 更新比例、学习率、loss 曲线、n-gram / 贝叶斯最优参照、生成样本、事实探针。
// 离线核对（同一份代码、同一组种子，在 Node 里跑完整训练）：node tools/p08_check.js [lr=0.005 B=2 warmup=100 total=5000 seed=1]
// 其他工具：tools/p08_stats.js（n-gram 交叉熵、熵率、贝叶斯最优）· tools/p08_unused.js（未出现 token 的 embedding）·
//          tools/p08_final_shots.js（把页面训练到最后一步再截图，验证器只模拟约 20 帧）· tools/p08_eval_timing.js（validate_doc --eval：页面内每步耗时）
// 默认 B = 2：无头浏览器里 LM.gpt.trainStep 比 Node 慢约 1.7 倍（B = 4 ≈ 20 ms/步），B = 2 时每帧 stage ≈ 12 ms。

const P05 = (() => {
    const V = LM.vocab, W = LM.world, AG = LM.AG, C = LM.C, F = LM.FONT;
    const CFG = { V: V.size, T: 32, d: 32, L: 2, H: 4, ff: 128 };
    const T = CFG.T;
    const SEED = { corpus: 2024, val: 7, valWin: 5, sample: 4242, decode: 9001, sweep: 777 };   // corpus / val 与 tools/train_base.js 相同
    const VAL_B = 32, VAL_CHUNK = 1, NCHUNK = VAL_B / VAL_CHUNK, NPROBE = 6, PROBE_CHUNK = 4;   // 32 个固定验证窗口；评估拆成 32 + 6 + 1 = 39 小块，每个训练步顺带做一块（约 1–2 ms）
    const EVAL_LEN = NCHUNK + NPROBE + 1;
    const SAMPLE = { prompt: '。', len: 30, temp: 1.0 };
    const CKPTS = [0, 200, 400, 600, 1000, 1500, 2000, 3000, 5000, 8000];
    const COVER_BINS = 240;
    const NSHOW = 4;                                                 // batch 类预览最多画 4 条序列
    const LIVE_X = 50;                                               // LIVE 角标占住画布左上角：live 预览的标题从这里开始写
    /** 评估时刻：前 400 步每 40 步，之后每 50 步（间隔 ≥ 评估流水线的 39 步） */
    const evalAt = (s) => (s <= 400 ? s % 40 === 0 : s % 50 === 0);
    /** live 曲线的横轴：随训练分几档放大，最后等于总步数（早期不至于挤在最左边，又不会每帧都变） */
    const xSpan = (R) => Math.min(R.total, [100, 200, 500, 1000, 2000, 3000, 5000, 8000].find((v) => v >= R.step * 1.04) || R.total);
    const ATTRS = [
        { key: 'food', label: '爱吃', prompt: (a) => '。' + a.name + '喜欢吃', color: C.orange },
        { key: 'home', label: '住处', prompt: (a) => '。' + a.name + '住在', color: C.cyan },
        { key: 'color', label: '颜色', prompt: (a) => '。' + a.name + '是', color: C.purple },
    ];
    const perfNow = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

    // ═════════════════════════════ 数据与真实统计（resident） ═════════════════════════════
    function makeData() {
        const text = W.corpus(LM.rng(SEED.corpus), 20000);
        const train = V.encode(text), val = V.encode(W.corpus(LM.rng(SEED.val), 2000));
        const vb = LM.gpt.batch(val, VAL_B, T, LM.rng(SEED.valWin));          // 与离线训练基座模型时的验证 batch 相同
        const cnt = new Float64Array(V.size); for (let i = 0; i < train.length; i++) cnt[train[i]]++;
        let H1 = 0; for (const c of cnt) if (c) { const p = c / train.length; H1 -= p * Math.log(p); }
        return { text, train, val, vx: vb.x, vy: vb.y, vstarts: vb.starts, cnt, H1 };
    }
    /** 生成器能产生的全部句子及其概率（world.sentence：a、b 独立均匀，b 撞上 a 时取下一只；句式均匀） */
    function enumerate() {
        const A = W.animals, m = new Map();
        const add = (s, p, k) => { const e = m.get(s); if (e) e.p += p; else m.set(s, { s, p, k }); };
        for (let ai = 0; ai < 8; ai++) for (let k = 0; k < 10; k++) {
            if (k === 7 || k === 8) { for (let bi = 0; bi < 8; bi++) if (bi !== ai) add(W.templates[k](A[ai], A[bi]), (1 / 80) * (bi === (ai + 1) % 8 ? 2 / 8 : 1 / 8), k); }
            else add(W.templates[k](A[ai], A[(ai + 1) % 8]), 1 / 80, k);
        }
        const list = [...m.values()];
        list.forEach((e) => { e.ids = V.encode(e.s); });
        let Hs = 0, EL = 0; list.forEach((e) => { Hs -= e.p * Math.log(e.p); EL += e.p * e.ids.length; });
        return { list, legit: new Set(list.map((e) => e.s)), Hs, EL, rate: Hs / EL };
    }
    /** n-gram 交叉熵：训练集上计数，在固定验证窗口上评估；位置 t 只用窗口内已有的 min(n−1, t+1) 个字作上下文（与模型同等信息） */
    function ngramWin(train, vx, vy, nMax, k) {
        const Vn = V.size, cnt = [], ctx = [];
        for (let o = 1; o <= nMax; o++) {
            const c1 = new Map(), c0 = new Map();
            for (let i = 0; i + o <= train.length; i++) {
                let c = 0; for (let j = 0; j < o - 1; j++) c = c * Vn + train[i + j];
                const f = c * Vn + train[i + o - 1];
                c1.set(f, (c1.get(f) || 0) + 1); c0.set(c, (c0.get(c) || 0) + 1);
            }
            cnt.push(c1); ctx.push(c0);
        }
        const out = [];
        for (let n = 1; n <= nMax; n++) {
            let s = 0;
            for (let b = 0; b < VAL_B; b++) for (let t = 0; t < T; t++) {
                const o = Math.min(n, t + 2);
                let c = 0; for (let j = t + 2 - o; j <= t; j++) c = c * Vn + vx[b * T + j];
                const f = c * Vn + vy[b * T + t];
                s -= Math.log(((cnt[o - 1].get(f) || 0) + k) / ((ctx[o - 1].get(c) || 0) + k * Vn));
            }
            out.push(s / (VAL_B * T));
        }
        return out;
    }
    /**
     * 贝叶斯最优预测：知道生成规则（176 种句子及概率），只看窗口里的 x₀…x_t，精确算 P(x_{t+1} | x₀…x_t)。
     * 隐状态 = （当前句子，句内位置）；窗口起点平稳分布 ∝ P(句子)；句末以 P(句子) 转入下一句（HMM 前向算法）。
     */
    function bayes(G, vx, vy) {
        const S = G.list, NS = S.length, off = new Int32Array(NS + 1);
        S.forEach((e, i) => { off[i + 1] = off[i] + e.ids.length; });
        const N = off[NS], sid = new Int32Array(N), emit = new Int32Array(N), last = new Uint8Array(N), first = Array.from({ length: V.size }, () => []);
        S.forEach((e, i) => { first[e.ids[0]].push(i); for (let j = 0; j < e.ids.length; j++) { sid[off[i] + j] = i; emit[off[i] + j] = e.ids[j]; last[off[i] + j] = j === e.ids.length - 1 ? 1 : 0; } });
        let a = new Float64Array(N), na = new Float64Array(N);
        const pos = new Float64Array(T); let tot = 0;
        for (let b = 0; b < VAL_B; b++) {
            const x0 = vx[b * T]; let z = 0;
            a.fill(0); for (let s = 0; s < N; s++) if (emit[s] === x0) { a[s] = S[sid[s]].p; z += a[s]; }
            for (let s = 0; s < N; s++) a[s] /= z;
            for (let t = 0; t < T; t++) {
                const nx = vy[b * T + t]; let end = 0, pn = 0;
                na.fill(0);
                for (let s = 0; s < N; s++) { const v = a[s]; if (v === 0) continue; if (last[s]) end += v; else if (emit[s + 1] === nx) { na[s + 1] += v; pn += v; } }
                if (end > 0) first[nx].forEach((i) => { const wv = end * S[i].p; na[off[i]] += wv; pn += wv; });
                const l = -Math.log(pn); pos[t] += l / VAL_B; tot += l;
                for (let s = 0; s < N; s++) na[s] /= pn;
                const tmp = a; a = na; na = tmp;
            }
        }
        return { mean: tot / (VAL_B * T), pos };
    }
    /** 24 个事实探针：「。X喜欢吃」→ 食物，「。X住在」→ 住处，「。X是」→ 颜色；下标 = 动物 × 3 + 属性 */
    function makeProbes() {
        const list = [];
        W.animals.forEach((a, ai) => ATTRS.forEach((at, j) => { const pr = at.prompt(a), ans = a[at.key]; list.push({ ai, j, prompt: pr, ans, np: pr.length, ids: V.encode(pr + ans), aIds: Array.from(V.encode(ans)) }); }));
        const PT = Math.max(...list.map((q) => q.ids.length)), x = new Int32Array(list.length * PT);
        list.forEach((q, i) => { for (let t = 0; t < PT; t++) x[i * PT + t] = q.ids[Math.min(t, q.ids.length - 1)]; });   // 右侧补齐：因果掩码保证补齐的字不影响前面的位置
        return { list, PT, x };
    }
    /** 探针的第 part 块（PROBE_CHUNK 条事实，一次批量前向）写进 out */
    function runProbes(m, P, part, out) {
        const nb = PROBE_CHUNK, i0 = part * nb, Vn = V.size, p = new Float32Array(Vn);
        const lg = AG.noGrad(() => LM.gpt.forward(m, P.x.subarray(i0 * P.PT, (i0 + nb) * P.PT), nb, P.PT).logits.data);
        for (let r = 0; r < nb; r++) { const q = P.list[i0 + r]; let lp = 0; q.aIds.forEach((c, k) => { LM.softmax(lg, p, 1, (r * P.PT + q.np - 1 + k) * Vn, Vn); lp += Math.log(Math.max(p[c], 1e-30)); }); out[i0 + r] = Math.exp(lp); }
        return out;
    }
    /** 生成的 token → 句子片段；以「。」结束的完整句子查 176 句的表判对错，最后半句不判 */
    function segments(ids, legit) {
        const segs = []; let cur = [];
        ids.forEach((id) => { cur.push(id); if (V.itos[id] === '。') { segs.push({ text: cur.map((i) => V.show(i)).join(''), ok: legit.has(V.decode(cur)) }); cur = []; } });
        if (cur.length) segs.push({ text: cur.map((i) => V.show(i)).join(''), ok: null });
        return segs;
    }
    /** 语料开头的若干句：名字 / 事实 / 句式文字分段（第 01 步的着色） */
    function corpusHead(text, nSent) {
        const names = W.animals.map((a) => a.name), facts = [];
        W.animals.forEach((a) => { facts.push(a.food, a.home, a.color); });
        const keys = names.map((s) => [s, 'name']).concat(facts.map((s) => [s, 'fact'])).sort((p, q) => q[0].length - p[0].length);
        const sents = text.split('。').slice(0, nSent).map((s) => s + '。');
        return sents.map((s) => {
            const out = []; let i = 0;
            while (i < s.length) {
                const hit = keys.find(([k]) => s.startsWith(k, i));
                if (hit) { out.push({ t: hit[0], kind: hit[1] }); i += hit[0].length; } else { const last = out[out.length - 1]; if (last && last.kind === 'tpl') last.t += s[i]; else out.push({ t: s[i], kind: 'tpl' }); i++; }
            }
            return out;
        });
    }

    // ═════════════════════════════ 实时训练（history） ═════════════════════════════
    const TRACE = { k: 0, row: '吃', col: 0 };   // 第 05 步跟踪的一个标量：wte[吃][0] 的 Adam 状态
    function newRun(S) {
        const seed = (S.seed >>> 0) || 1;
        const R = { B: S.B, maxLR: S.lr, minLR: S.lr / 10, warmup: S.warmup, total: S.total, clip: 1.0, wd: 0.1, step: 0 };
        R.m = LM.gpt.create(CFG, seed);
        R.opt = new LM.AdamW(R.m.params, { lr: R.maxLR, b1: 0.9, b2: 0.95, eps: 1e-8, wd: R.wd });
        R.rand = LM.rng((seed * 7919 + 11) >>> 0);
        R.nP = R.m.params.length; R.nParams = LM.gpt.numParams(R.m);
        R.loss = new Float32Array(R.total).fill(NaN); R.ema = new Float32Array(R.total).fill(NaN); R.gn = new Float32Array(R.total).fill(NaN); R.lrs = new Float32Array(R.total).fill(NaN);
        R.tg = new Float32Array(R.total * R.nP); R.tu = new Float32Array(R.total * R.nP);
        R.trG = new Float32Array(R.total).fill(NaN); R.trM = new Float32Array(R.total).fill(NaN); R.medU = new Float32Array(R.total).fill(NaN);
        R.G = tensorGroups(R.m);
        R.traceEl = V.stoi.get(TRACE.row) * CFG.d + TRACE.col;
        R.batch = { x: new Int32Array(R.B * T), y: new Int32Array(R.B * T), starts: new Int32Array(R.B) };
        R.cover = new Float32Array(COVER_BINS); R.nClip = 0; R.ms = 0; R.emaAcc = 0; R.emaW = 0;
        R.evals = []; R.pend = null; R.evalModel = LM.gpt.create(CFG, 1);
        R.bx = null; R.by = null; R.bstarts = null; R.rowLoss = null; R.probs = null; R.lastLoss = NaN;
        startEval(R);
        return R;
    }
    function startEval(R) {
        R.evalModel.params.forEach((p, i) => p.data.set(R.m.params[i].data));     // 快照：评估分摊到后面 39 个训练步里做，结果仍属于这一步
        R.pend = { step: R.step, chunk: 0, sum: 0, pos: new Float64Array(T), probe: new Float32Array(24) };
    }
    function evalChunk(R, D, P, legit) {
        const E = R.pend; if (!E) return;
        if (E.chunk < NCHUNK) {
            const o = E.chunk * VAL_CHUNK * T, n = VAL_CHUNK * T;
            const out = AG.noGrad(() => LM.gpt.forward(R.evalModel, D.vx.subarray(o, o + n), VAL_CHUNK, T, { targets: D.vy.subarray(o, o + n) }));
            const rl = out.loss.rowLoss; for (let i = 0; i < n; i++) { E.sum += rl[i]; E.pos[i % T] += rl[i] / VAL_B; }
        } else if (E.chunk < NCHUNK + NPROBE) runProbes(R.evalModel, P, E.chunk - NCHUNK, E.probe);
        else {
            const g = LM.gpt.generate(R.evalModel, V.encode(SAMPLE.prompt), SAMPLE.len, { temperature: SAMPLE.temp, rand: LM.rng(SEED.sample) }).gen;
            R.evals.push({ step: E.step, tokens: E.step * R.B * T, val: E.sum / (VAL_B * T), pos: Float32Array.from(E.pos), probe: E.probe, sample: g, segs: segments(g, legit) });
            R.pend = null;
            return;
        }
        E.chunk++;
    }
    function flushEval(R, D, P, legit) { while (R.pend) evalChunk(R, D, P, legit); }
    /** 一个训练步：取 batch → 前向 + 反向 + 裁剪 + AdamW（LM.gpt.trainStep）→ 记录诊断量 → 推进一块评估 */
    function step(R, D, P, legit) {
        const it = R.step, B = R.B;
        const lr = LM.cosineLR(it, { maxLR: R.maxLR, minLR: R.minLR, warmup: R.warmup, total: R.total });
        LM.gpt.batch(D.train, B, T, R.rand, R.batch);
        const t0 = perfNow();
        const r = LM.gpt.trainStep(R.m, R.opt, R.batch.x, R.batch.y, B, T, lr, { clip: R.clip, keepProbs: true });
        const dt = perfNow() - t0; R.ms = R.ms ? 0.9 * R.ms + 0.1 * dt : dt;
        R.loss[it] = r.loss; R.gn[it] = r.gradNorm; R.lrs[it] = lr; R.lastLoss = r.loss;
        R.emaAcc = 0.98 * R.emaAcc + 0.02 * r.loss; R.emaW = 0.98 * R.emaW + 0.02; R.ema[it] = R.emaAcc / R.emaW;   // 带偏差修正的滑动平均
        const k = r.gradNorm > R.clip ? R.clip / (r.gradNorm + 1e-6) : 1;                // 裁剪系数：p.grad 已被乘以它
        if (k < 1) R.nClip++;
        for (let j = 0; j < R.nP; j++) { R.tg[it * R.nP + j] = LM.gradNorm(R.m.params[j]) / k; R.tu[it * R.nP + j] = R.opt.updRatio[j]; }
        R.medU[it] = median(R.G.mats.map((j) => R.opt.updRatio[j]));
        const e = R.traceEl, t1 = R.opt.t, vh = Math.sqrt(R.opt.v[TRACE.k][e] / (1 - Math.pow(R.opt.b2, t1))) + R.opt.eps;
        R.trG[it] = R.m.params[TRACE.k].grad[e] / vh; R.trM[it] = (R.opt.m[TRACE.k][e] / (1 - Math.pow(R.opt.b1, t1))) / vh;
        R.bx = Int32Array.from(R.batch.x); R.by = Int32Array.from(R.batch.y); R.bstarts = Int32Array.from(R.batch.starts);
        R.rowLoss = r.out.loss.rowLoss; R.probs = r.out.loss.probs;
        const bw = D.train.length / COVER_BINS;
        for (let b = 0; b < B; b++) { const s = R.batch.starts[b], i0 = Math.floor(s / bw), i1 = Math.min(COVER_BINS - 1, Math.floor((s + T) / bw)); if (i0 === i1) R.cover[i0] += (T + 1) / bw; else { const cut = (i0 + 1) * bw; R.cover[i0] += (cut - s) / bw; R.cover[i1] += (s + T + 1 - cut) / bw; } }
        R.step = it + 1;
        evalChunk(R, D, P, legit);
        if (R.step >= R.total) { flushEval(R, D, P, legit); startEval(R); flushEval(R, D, P, legit); }
        else if (evalAt(R.step)) { flushEval(R, D, P, legit); startEval(R); }
    }
    const lastEval = (R) => (R && R.evals.length ? R.evals[R.evals.length - 1] : null);
    /**
     * 居中滑动平均（只用已经训练过的 [0, n) 步）：第 i 点 = 第 i−h…i+h 步的平均，h = min(half, i)。
     * 最后 half 步凑不满对称窗口就不画——平滑线不会像指数滑动平均那样「滞后」于真实 loss。
     */
    function smoothC(a, n, half) {
        const out = new Float32Array(a.length).fill(NaN), cs = new Float64Array(n + 1);
        for (let i = 0; i < n; i++) cs[i + 1] = cs[i] + a[i];
        for (let i = 0; i < n - half; i++) { const h = Math.min(half, i); out[i] = (cs[i + h + 1] - cs[i - h]) / (2 * h + 1); }
        return out;
    }
    /** 最近 k 步训练 loss 的平均（控制台的读数） */
    function recentLoss(R, k) { const n = Math.min(k, R.step); let s = 0; for (let i = R.step - n; i < R.step; i++) s += R.loss[i]; return n ? s / n : NaN; }
    /** 矩阵参数（14 个）+ 其余 22 个向量参数（LN γ/β、偏置）合并成一项 */
    function tensorGroups(m) {
        const mats = [], vecs = [];
        m.params.forEach((p, j) => { if (p.decay) mats.push(j); else vecs.push(j); });
        const short = (n) => n.replace('attn.', '').replace('mlp.', '');
        return { mats, vecs, labels: mats.map((j) => short(m.params[j].name)).concat([`其余 ${vecs.length} 个向量`]) };
    }
    function perTensor(R, arr, it, G, withVec) {
        const o = it * R.nP, v = G.mats.map((j) => arr[o + j]);
        if (withVec) { let s = 0; G.vecs.forEach((j) => { s += arr[o + j] ** 2; }); v.push(Math.sqrt(s)); }
        return v;
    }
    function median(a) { const s = Array.from(a).filter((x) => isFinite(x)).sort((p, q) => p - q); return s.length ? s[s.length >> 1] : NaN; }

    // ═════════════════════════════ 解码（第 09 步，基座模型） ═════════════════════════════
    function decodeState(base, S) {
        const ids = Array.from(V.encode(S.prompt)), lg = LM.gpt.nextLogits(base, ids);
        const o = { temperature: S.temp, topK: S.topk, topP: S.topp };
        const raw = LM.softmax(lg, new Float32Array(V.size)), out = LM.gpt.decodeDist(lg, o);
        const samples = [0, 1, 2, 3, 4].map((k) => LM.gpt.generate(base, ids, T - ids.length, Object.assign({ rand: LM.rng(SEED.decode + k) }, o)).gen);
        const H = (p) => { let h = 0; for (const q of p) if (q > 0) h -= q * Math.log(q); return h; };
        let kept = 0; for (const q of out) if (q > 0) kept++;
        return { ids, raw, out, samples, kept, Hraw: H(raw), Hout: H(out) };
    }
    /** 纯温度采样扫描：每个温度 12 条、每条 31 个字，统计完整句子的正确率与「不同句子 / 句子数」 */
    function tempSweep(base, legit) {
        const temps = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0], ok = [], div = [];
        temps.forEach((tp) => {
            let tot = 0, good = 0; const seen = new Set();
            for (let k = 0; k < 12; k++) {
                const g = LM.gpt.generate(base, V.encode('。'), T - 1, { temperature: tp, rand: LM.rng(SEED.sweep + k) }).gen;
                segments(g, legit).forEach((s) => { if (s.ok === null) return; tot++; seen.add(s.text); if (s.ok) good++; });
            }
            ok.push(good / Math.max(1, tot)); div.push(seen.size / Math.max(1, tot));
        });
        return { temps, ok, div };
    }

    // ═════════════════════════════ 规模化（第 10 / 11 步） ═════════════════════════════
    const CH = { E: 1.69, A: 406.4, B: 410.7, a: 0.34, b: 0.28 };    // Hoffmann et al. 2022（方法 3 的拟合）
    const chLoss = (N, D) => CH.E + CH.A / Math.pow(N, CH.a) + CH.B / Math.pow(D, CH.b);
    const chOpt = (Cf) => { const G = Math.pow(CH.a * CH.A / (CH.b * CH.B), 1 / (CH.a + CH.b)), N = G * Math.pow(Cf / 6, CH.b / (CH.a + CH.b)); return { N, D: Cf / (6 * N) }; };
    const MODELS = [   // 公开的参数量 N 与训练 token 数 D；C = 6ND 是估算
        { name: 'GPT-3', N: 175e9, D: 300e9 }, { name: 'Gopher', N: 280e9, D: 300e9 },
        { name: 'Chinchilla', N: 70e9, D: 1.4e12 }, { name: 'LLaMA-3 8B', N: 8e9, D: 15e12 },
    ];
    function zeroGB(psi, nd, stage, K) { let w = 2 * psi, g = 2 * psi, os = K * psi; if (stage >= 1) os /= nd; if (stage >= 2) g /= nd; if (stage >= 3) w /= nd; return (w + g + os) / 1e9; }

    // ═════════════════════════════ 绘图 ═════════════════════════════
    const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const sup = (n) => String(n).split('').map((c) => SUP[c] || c).join('');
    function tickFmt(v) {
        const a = Math.abs(v);
        if (a === 0) return '0';
        if (a >= 1e4 || a < 1e-2) { const e = Math.floor(Math.log10(a) + 1e-9), mt = v / Math.pow(10, e); return (Math.abs(mt - 1) < 1e-6 ? '' : +mt.toFixed(1) + '×') + '10' + sup(e); }
        return String(+v.toPrecision(3));
    }
    const kfmt = (v) => (v >= 1e9 ? (v / 1e9).toFixed(v >= 1e10 ? 0 : 1) + 'B' : v >= 1e6 ? (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + 'k' : String(Math.round(v)));
    const efmt = (v, d = 1) => (v === 0 ? '0' : v.toExponential(d).replace('e+', 'e'));
    function niceTicks(lo, hi, n) {
        const span = hi - lo, raw = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const st = [1, 2, 2.5, 5, 10].map((q) => q * mag).find((s) => s >= raw * 0.999);
        const out = []; for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : +v.toPrecision(12));
        return out;
    }
    function logTicks(lo, hi, maxN) {
        const out = [], dec = [];
        for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++) for (const mt of [1, 2, 3, 5]) { const v = mt * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) { out.push(v); if (mt === 1) dec.push(v); } }
        if (out.length <= maxN) return out;
        if (dec.length >= 2) { const st = Math.ceil(dec.length / Math.max(2, maxN)); return dec.filter((_, i) => i % st === 0); }
        return out.filter((_, i) => i % Math.ceil(out.length / maxN) === 0);
    }
    /** 右侧边距里的标签：按 y 排序、互相推开、用短引线连回原位置 */
    function placeLabels(ctx, labs, top, bottom, x0) {
        labs.sort((p, q) => p.y - q.y);
        const gap = 12.5;
        labs.forEach((l, i) => { l.ly = Math.max(l.y, top + 5, i ? labs[i - 1].ly + gap : -1e9); });
        for (let i = labs.length - 1; i >= 0; i--) { const lim = (i === labs.length - 1 ? bottom - 5 : labs[i + 1].ly - gap); if (labs[i].ly > lim) labs[i].ly = lim; }
        ctx.font = F(10); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        labs.forEach((l) => {
            ctx.strokeStyle = l.color; ctx.globalAlpha = 0.6; ctx.lineWidth = 1; ctx.setLineDash(l.x != null ? [2, 2] : []); ctx.beginPath(); ctx.moveTo(l.x != null ? l.x : x0, l.y); ctx.lineTo(x0, l.y); ctx.lineTo(x0 + 5, l.ly); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
            ctx.fillStyle = l.color; ctx.fillText(l.text, x0 + 7, l.ly);
        });
    }
    /**
     * 曲线图（坐标轴自绘，便于放参照线标签）。o: { x:[a,b], y:[a,b], logX, logY, series:[{ x?, y, color, width, dash, alpha, dots, label }],
     *   refs:[{ y, label, color, dash }], marks:[{ x, label, color }], bands:[{ x0, x1, color }], points:[{ x, y, color, label, r }], legend:[{ label, color, dash }], endLabels, title, xLabel, yLabel, xFmt, yFmt }
     */
    function chart(cv, o) {
        const { ctx, w, h } = PL.draw.ctx(cv);
        const rightLab = (o.refs || []).some((r) => r.label) || o.endLabels;
        const padL = o.padL ?? 46, padR = rightLab ? (o.padR ?? 96) : (o.padR ?? 12), padT = o.title ? 24 : 10, padB = o.xLabel ? 34 : 20;
        const lx = !!o.logX, ly = !!o.logY;
        const fx = (v) => (lx ? Math.log10(Math.max(v, 1e-300)) : v), fy = (v) => (ly ? Math.log10(Math.max(v, 1e-300)) : v);
        const ax = fx(o.x[0]), bx = fx(o.x[1]), ay = fy(o.y[0]), by = fy(o.y[1]);
        const pw = w - padL - padR, ph = h - padT - padB;
        const X = (v) => padL + (fx(v) - ax) / ((bx - ax) || 1) * pw, Y = (v) => padT + ph - (fy(v) - ay) / ((by - ay) || 1) * ph;
        if (o.title) { ctx.font = F(11, true); ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.tx ?? LIVE_X, 5); }
        (o.bands || []).forEach((b) => { const x0 = X(Math.max(b.x0, o.x[0])), x1 = X(Math.min(b.x1, o.x[1])); if (x1 > x0) { ctx.fillStyle = b.color; ctx.fillRect(x0, padT, x1 - x0, ph); } });
        ctx.font = F(10); ctx.lineWidth = 1;
        const xt = o.xTicks || (lx ? logTicks(o.x[0], o.x[1], Math.max(2, Math.floor(pw / 48))) : niceTicks(o.x[0], o.x[1], Math.max(2, Math.floor(pw / 70))));
        const yt = o.yTicks || (ly ? logTicks(o.y[0], o.y[1], Math.max(2, Math.floor(ph / 17))) : niceTicks(o.y[0], o.y[1], Math.max(2, Math.floor(ph / 30))));
        ctx.strokeStyle = '#1b1f25'; ctx.fillStyle = '#737983';
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        xt.forEach((v) => { const x = X(v); if (x < padL - 1 || x > padL + pw + 1) return; ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke(); ctx.fillText((o.xFmt || tickFmt)(v), x, padT + ph + 3); });
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        yt.forEach((v) => { const y = Y(v); if (y < padT - 1 || y > padT + ph + 1) return; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke(); ctx.fillText((o.yFmt || tickFmt)(v), padL - 4, y); });
        ctx.strokeStyle = '#3a3f46'; ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + ph); ctx.lineTo(padL + pw, padT + ph); ctx.stroke();
        ctx.fillStyle = '#868c96';
        if (o.xLabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xLabel, padL + pw / 2, h - 2); }
        if (o.yLabel) { ctx.save(); ctx.translate(10, padT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
        ctx.save(); ctx.beginPath(); ctx.rect(padL, padT, pw, ph); ctx.clip();
        (o.refs || []).forEach((r) => { const y = Y(r.y); ctx.strokeStyle = r.color || C.gray; ctx.lineWidth = r.width || 1; ctx.setLineDash(r.dash || [4, 3]); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke(); });
        ctx.setLineDash([]);
        (o.series || []).forEach((s) => {
            const n = s.y.length; ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = s.width || 1.5; ctx.globalAlpha = s.alpha ?? 1; ctx.setLineDash(s.dash || []);
            ctx.beginPath(); let pen = false;
            for (let i = 0; i < n; i++) {
                const xv = s.x ? s.x[i] : i, yv = s.y[i];
                if (!isFinite(yv) || !isFinite(xv) || (ly && yv <= 0) || (lx && xv <= 0)) { pen = false; continue; }
                const px = X(xv), py = Y(yv); if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
            }
            if (s.line !== false) ctx.stroke();
            ctx.setLineDash([]);
            if (s.dots) for (let i = 0; i < n; i++) { const xv = s.x ? s.x[i] : i, yv = s.y[i]; if (!isFinite(yv) || (ly && yv <= 0) || (lx && xv <= 0)) continue; ctx.beginPath(); ctx.arc(X(xv), Y(yv), s.dots, 0, 7); ctx.fill(); }
            ctx.globalAlpha = 1;
        });
        (o.marks || []).forEach((m) => { const x = X(m.x); ctx.strokeStyle = m.color || C.yellow; ctx.lineWidth = 1; ctx.setLineDash(m.dash || [3, 3]); ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke(); ctx.setLineDash([]); });
        (o.points || []).forEach((p) => { ctx.fillStyle = p.color || C.yellow; ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), p.r || 3.5, 0, 7); ctx.fill(); ctx.stroke(); });
        ctx.restore();
        ctx.font = F(10); ctx.textBaseline = 'top';
        (o.points || []).forEach((p) => { if (!p.label) return; const x = X(p.x), y = Y(p.y); ctx.textAlign = p.align || 'left'; const tw = ctx.measureText(p.label).width, lx0 = p.align === 'right' ? x - 6 - tw : x + 6; ctx.fillStyle = 'rgba(5,7,10,0.7)'; ctx.fillRect(lx0 - 2, y + (p.dy ?? -14) - 1, tw + 4, 13); ctx.fillStyle = p.color || C.yellow; ctx.textAlign = 'left'; ctx.fillText(p.label, lx0, y + (p.dy ?? -14)); });
        (o.marks || []).forEach((m) => { if (!m.label) return; const x = X(m.x); ctx.textAlign = x > padL + pw * 0.7 ? 'right' : 'left'; const tw = ctx.measureText(m.label).width, lx0 = ctx.textAlign === 'right' ? x - 4 - tw : x + 4; ctx.fillStyle = 'rgba(5,7,10,0.75)'; ctx.fillRect(lx0 - 2, padT + 2, tw + 4, 13); ctx.fillStyle = m.color || C.yellow; ctx.fillText(m.label, x + (ctx.textAlign === 'right' ? -4 : 4), padT + 3); });
        const labs = [];
        (o.refs || []).forEach((r) => { if (r.label) labs.push({ y: Y(r.y), text: r.label, color: r.color || C.gray }); });
        if (o.endLabels) (o.series || []).forEach((s) => { if (!s.label) return; for (let i = s.y.length - 1; i >= 0; i--) if (isFinite(s.y[i]) && !(ly && s.y[i] <= 0)) { labs.push({ y: Y(s.y[i]), x: X(s.x ? s.x[i] : i), text: s.label, color: s.color }); break; } });
        if (labs.length) placeLabels(ctx, labs, padT, padT + ph, padL + pw + 2);
        if (o.legend && o.legend.length) {
            ctx.font = F(10); const lw = Math.max(...o.legend.map((l) => ctx.measureText(l.label).width)) + 26, lh = o.legend.length * 13 + 6;
            const lx0 = o.legendAt === 'left' || o.legendAt === 'bl' ? padL + 4 : padL + pw - lw - 4, ly0 = o.legendAt === 'bottom' || o.legendAt === 'bl' ? padT + ph - lh - 4 : padT + 4;
            ctx.fillStyle = 'rgba(8,9,12,0.82)'; ctx.fillRect(lx0, ly0, lw, lh);
            o.legend.forEach((l, i) => {
                const y = ly0 + 9 + i * 13;
                if (l.dot) { ctx.fillStyle = l.color; ctx.strokeStyle = '#000'; ctx.beginPath(); ctx.arc(lx0 + 12, y, 3.5, 0, 7); ctx.fill(); ctx.stroke(); }
                else { ctx.strokeStyle = l.color; ctx.lineWidth = 2; ctx.setLineDash(l.dash || []); ctx.beginPath(); ctx.moveTo(lx0 + 5, y); ctx.lineTo(lx0 + 19, y); ctx.stroke(); ctx.setLineDash([]); }
                ctx.fillStyle = l.color; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(l.label, lx0 + 23, y);
            });
        }
        return { X, Y, padL, padT, pw, ph };
    }
    /** 横向对数柱状图（梯度范数、更新比例这类跨几个数量级的量） */
    function logBars(cv, vals, labels, o) {
        const { ctx, w, h } = PL.draw.ctx(cv);
        const n = vals.length, padL = o.padL ?? 92, padR = 50, padT = o.title ? 24 : 8, padB = 22;
        const lo = Math.log10(o.range[0]), hi = Math.log10(o.range[1]), pw = w - padL - padR, bh = (h - padT - padB) / n;
        const X = (v) => padL + PL.clamp((Math.log10(Math.max(v, 1e-30)) - lo) / (hi - lo), 0, 1) * pw;
        if (o.title) { ctx.font = F(11, true); ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.tx ?? LIVE_X, 5); }
        ctx.font = F(10); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.lineWidth = 1;
        for (let e = Math.ceil(lo - 1e-9); e <= Math.floor(hi + 1e-9); e++) { const x = X(Math.pow(10, e)); ctx.strokeStyle = '#1b1f25'; ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, h - padB); ctx.stroke(); ctx.fillStyle = '#737983'; ctx.fillText('10' + sup(e), x, h - padB + 4); }
        (o.refs || []).forEach((r) => { const x = X(r.v); ctx.strokeStyle = r.color || C.yellow; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, h - padB); ctx.stroke(); ctx.setLineDash([]); });
        for (let i = 0; i < n; i++) {
            const y = padT + i * bh, v = vals[i], x1 = X(v);
            ctx.fillStyle = o.colors ? o.colors[i] : C.blue; ctx.globalAlpha = 0.9;
            ctx.fillRect(padL, y + bh * 0.16, Math.max(1, x1 - padL), bh * 0.68); ctx.globalAlpha = 1;
            if (o.ghost && isFinite(o.ghost[i])) { ctx.strokeStyle = 'rgba(255,255,255,0.55)'; const xg = X(o.ghost[i]); ctx.beginPath(); ctx.moveTo(xg, y + bh * 0.1); ctx.lineTo(xg, y + bh * 0.9); ctx.stroke(); }
            ctx.font = F(Math.min(11, Math.max(9, bh * 0.75))); ctx.fillStyle = o.labelColors ? o.labelColors[i] : C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(labels[i], padL - 5, y + bh / 2);
            ctx.fillStyle = C.dim; ctx.textAlign = 'left'; ctx.fillText(isFinite(v) ? efmt(v) : '—', Math.min(Math.max(x1, o.ghost && isFinite(o.ghost[i]) ? X(o.ghost[i]) : 0) + 4, w - padR + 4), y + bh / 2);
        }
    }
    /** 定宽 token 网格：rows = [{ label, labelColor, cells: [{ id, bg, fg, ring }], gap }]，每格一个字 */
    function tokenGrid(cv, rows, o) {
        const { ctx, w } = PL.draw.ctx(cv);
        const ncol = o.ncol || T, padL = o.padL ?? 62, padT = o.title ? 24 : 6, cw = (w - padL - 4) / ncol, ch = o.cellH || 21;
        if (o.title) { ctx.font = F(11, true); ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.tx ?? LIVE_X, 5); }
        if (o.colTicks) { ctx.font = F(9); ctx.fillStyle = C.dim; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; for (let t = 0; t < ncol; t += o.colTicks) ctx.fillText(String(t), padL + (t + 0.5) * cw, padT + 10); }
        let y = padT + (o.colTicks ? 12 : 0);
        const geom = [];
        rows.forEach((r) => {
            ctx.font = F(10); ctx.fillStyle = r.labelColor || C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(r.label, padL - 5, y + ch / 2);
            r.cells.forEach((c, t) => {
                const x = padL + t * cw;
                ctx.fillStyle = c.bg; ctx.fillRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
                ctx.font = F(Math.min(13, ch * 0.66, cw * 0.82)); ctx.fillStyle = c.fg || '#eef0f3'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(c.text != null ? c.text : V.show(c.id), x + cw / 2, y + ch / 2 + 0.5);
                if (c.ring) { ctx.strokeStyle = c.ring; ctx.lineWidth = 2; ctx.strokeRect(x + 1, y + 1, cw - 2, ch - 2); }
            });
            geom.push({ y, h: ch, key: r.key });
            y += ch + (r.gap ?? 3);
        });
        return { ctx, y, cw, padL, grid: { padL, cw, geom, ncol } };
    }
    /** probe / pick 用：设计坐标 → { key（行）, c（列）}；grid 是 tokenGrid 返回并存进 S 的布局（probe 拿不到 canvas） */
    function gridHit(g, info) {
        if (!g || !info) return null;
        const c = Math.floor((info.dx - g.padL) / g.cw); if (c < 0 || c >= g.ncol) return null;
        const r = g.geom.find((q) => info.dy >= q.y && info.dy < q.y + q.h);
        return r ? { key: r.key, c } : null;
    }
    /** 第 03 步热图的默认选中格：当前 batch 前 4 条里 loss 最大的位置（由代码自动挑选，任何时刻都有意义） */
    function autoSel(R) {
        const nb = Math.min(R.B, NSHOW); let best = 0, bv = -1;
        for (let i = 0; i < nb * T; i++) if (R.rowLoss[i] > bv) { bv = R.rowLoss[i]; best = i; }
        return { b: Math.floor(best / T), t: best % T };
    }
    const rgb = [0, 0, 0];
    function heatColor(t, map) { (PL.cmap[map || 'inferno'])(PL.clamp(t, 0, 1), rgb); return { bg: LM.rgbStr(rgb), fg: 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 145 ? '#0b0b0b' : '#f2f2f2' }; }
    const segColor = (s) => (s.ok === true ? C.green : s.ok === false ? C.red : C.gray);
    function segLine(segs) { return segs.map((s) => ({ t: s.text, color: segColor(s) })); }

    return {
        CFG, T, SEED, VAL_B, CKPTS, ATTRS, NSHOW, SAMPLE, CH, MODELS, LIVE_X, EVAL_LEN, xSpan,
        makeData, enumerate, ngramWin, bayes, makeProbes, runProbes, segments, corpusHead,
        newRun, step, flushEval, lastEval, smoothC, recentLoss, tensorGroups, perTensor, median, decodeState, tempSweep,
        chLoss, chOpt, zeroGB, chart, logBars, tokenGrid, gridHit, autoSel, heatColor, segLine, segColor, tickFmt, kfmt, efmt, evalAt,
    };
})();

PipelineLab.mount({
    paramSteps: { lr: ['loss', 'grad', 'adamw', 'sched', 'curve'], B: ['batch', 'loss', 'grad', 'adamw', 'sched', 'curve', 'scale'], warmup: ['loss', 'grad', 'adamw', 'sched', 'curve'], total: ['sched', 'curve'], spf: ['batch', 'loss', 'grad', 'adamw', 'sched', 'curve', 'probe', 'scale'], temp: ['decode'], topk: ['decode'], topp: ['decode'], prompt: ['decode'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '08 · 预训练：在浏览器里从零训练一个迷你 GPT',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 1,
    css: LM.CSS + `
        .p05-con { display: flex; flex-direction: column; gap: 8px; }
        .p05-prog { position: relative; height: 20px; background: #0d141a; border: 1px solid #22303c; border-radius: 10px; overflow: hidden; }
        .p05-fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0; background: linear-gradient(90deg, #1d4766, #5aa6d8); }
        .p05-fill.done { background: linear-gradient(90deg, #1f5a35, #4fbf72); }
        .p05-ptxt { position: absolute; left: 12px; top: 0; line-height: 20px; font-size: 0.8rem; color: #eef3f7; font-family: var(--pl-mono); white-space: nowrap; }
        .p05-grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 12px; }
        .p05-mini { width: 100%; height: 170px; display: block; }
        .p05-side { font-size: 0.88rem; line-height: 1.65; color: #c8ccd4; }
        .p05-side b { color: #b8d6e6; font-weight: 600; }
        .p05-ok { color: #5ad07a; } .p05-bad { color: #e06c75; } .p05-part { color: #8a8f98; }
        .p05-dim { color: #7d838c; font-size: 0.8rem; }
        @media (max-width: 860px) { .p05-grid { grid-template-columns: 1fr; } }
    `,
    regen: { label: '换随机种子', title: '换一组随机初始化与 batch 顺序，从头重新训练（语料不变）' },

    params: [
        { id: 'lr', label: '峰值学习率 η', options: [{ value: 3e-4, label: '3e-4' }, { value: 1e-3, label: '1e-3' }, { value: 2e-3, label: '2e-3' }, { value: 5e-3, label: '5e-3' }, { value: 1e-2, label: '1e-2' }, { value: 3e-2, label: '3e-2' }], value: 5e-3, steps: ['probe', 'scale'], desc: 'warmup 结束时的学习率 η<sub>max</sub>（之后余弦衰减到 η<sub>max</sub>/10）。3e-4 学得很慢：句式会了、事实记不住；3e-2 中段长期卡住。改动会从头重新训练。' },
        { id: 'B', label: 'batch B', options: [1, 2, 4, 8, 16], value: 2, steps: ['probe'], desc: '每步取几个长度 32 的窗口（每步 B×32 个预测）。B 越大梯度越准、loss 曲线越平滑，但每步越慢（B = 16 约是 B = 2 的 7 倍耗时）。改动会重新训练。' },
        { id: 'warmup', label: 'warmup 步数', min: 0, max: 500, step: 10, value: 100, steps: ['probe', 'scale'], desc: '学习率从 0 线性升到 η<sub>max</sub> 用的步数（Step 06 的绿色区域）。设为 0 时第一步就用满学习率：Adam 的相对更新一开始就很大（Step 05 / 06 的比值曲线）。' },
        { id: 'total', label: '总步数', options: [1000, 2000, 3000, 5000, 8000], value: 5000, steps: ['loss', 'grad', 'adamw', 'probe', 'scale'], desc: '训练的步数上限，也是余弦调度的长度：到了就停在最终状态。步数越多看的 token 越多（Step 10 的 D），最终 loss 越低。' },
        { id: 'spf', label: '每帧步数', options: [1, 2, 4, 8, 16], value: 1, affects: 'frame', desc: '每一帧跑几个训练步：只改变训练速度，不改变结果（同样的种子、同样的 batch 顺序）。机器快可以调大，控制台里有实测的每步耗时。顶栏的「速度」滑杆只影响动画时钟，不影响训练。' },
        { id: 'temp', label: '温度 τ', min: 0.1, max: 2, step: 0.05, value: 0.8, affects: 'frame', desc: 'Step 09 的采样温度：logits 除以 τ 再 softmax。τ < 1 分布变尖、更保守；τ > 1 变平、更多样也更多错（本页用 τ，避免和上下文长度 T 混淆）。' },
        { id: 'topk', label: 'top-k', options: [{ value: 0, label: '关' }, 1, 3, 5, 10, 20], value: 0, affects: 'frame', desc: 'Step 09：只保留概率最大的 k 个 token 再归一化。k = 1 就是贪心解码。' },
        { id: 'topp', label: 'top-p', min: 0.1, max: 1, step: 0.05, value: 0.9, affects: 'frame', desc: 'Step 09 的 nucleus 采样：按概率从大到小累加到 p 为止，其余截掉。p = 1 不截断。' },
        { id: 'prompt', label: '提示', options: ['。', '今天', '熊猫和', '森林住着', '小猫喜欢吃'], value: '。', affects: 'frame', desc: 'Step 09 用来续写的开头。「。」= 新句子的第一个字（选择最多），「小猫喜欢吃」几乎只有一个答案。' },
        { id: 'restart', label: '↻ 重新训练', action: () => {}, affects: 'param', desc: '用当前超参数和随机种子从第 0 步重新训练（结果与上一次完全相同；想看另一次随机初始化请点「换随机种子」）。' },
    ],

    stages: [
        {
            id: 'data', lifetime: 'resident', run(S) {
                S.D = P05.makeData();
                S.G = P05.enumerate();
                S.probes = P05.makeProbes();
                S.head = P05.corpusHead(S.D.text, 24);
                S.unused = []; for (let i = 0; i < LM.vocab.size; i++) if (!S.D.cnt[i]) S.unused.push(i);   // 预训练语料里从没出现过的 token
            },
        },
        {
            id: 'stats', lifetime: 'resident', run(S) {
                S.ngram = P05.ngramWin(S.D.train, S.D.vx, S.D.vy, 5, 0.01);
                S.bayes = P05.bayes(S.G, S.D.vx, S.D.vy);
                S.refs = [
                    { y: Math.log(LM.vocab.size), label: `ln V ${Math.log(LM.vocab.size).toFixed(2)}`, color: '#8a8f98' },
                    ...S.ngram.map((v, i) => ({ y: v, label: `${i + 1}-gram ${v.toFixed(2)}`, color: ['#9a7fb0', '#c678dd', '#56b6c2', '#4f9da6', '#3f8a8f'][i] })),
                    { y: S.bayes.mean, label: `贝叶斯 ${S.bayes.mean.toFixed(3)}`, color: LM.C.green, dash: [6, 3], width: 1.5 },
                ];
            },
        },
        { id: 'base', lifetime: 'resident', run(S) { S.base = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), LM_WEIGHTS.base.b64); } },
        { id: 'sweep', lifetime: 'resident', lazy: true, run(S) { S.sweep = P05.tempSweep(S.base, S.G.legit); } },
        {
            id: 'train', lifetime: 'history', run(S) {
                if (!S.run || S._runEpoch !== S.epoch) { S.run = P05.newRun(S); S._runEpoch = S.epoch; S.sel = null; }
                const R = S.run;
                for (let k = 0; k < S.spf && R.step < R.total; k++) P05.step(R, S.D, S.probes, S.G.legit);
            },
        },
        {
            id: 'decode', lifetime: 'frame', lazy: true, run(S) {
                const key = [S.temp, S.topk, S.topp, S.prompt].join('|');
                if (!S.dec || S.dec.key !== key) { S.dec = P05.decodeState(S.base, S); S.dec.key = key; }
            },
        },
    ],

    sections: [{
        id: 'console', title: '训练控制台 · 迷你 GPT 正在你的浏览器里训练', position: 'top',
        sub: '2 层 · 4 头 · d = 32 · 28 928 个参数 · 字符级词表 76 · 上下文 32。改训练超参数（学习率、batch、warmup、总步数）或点 ↻ 会从头训练；「每帧步数」只改速度。',
        html: `<div class="p05-con"><div class="p05-prog"><div class="p05-fill"></div><div class="p05-ptxt"></div></div><div class="p05-kpis"></div>
            <div class="p05-grid"><canvas class="p05-mini"></canvas><div class="p05-side"></div></div></div>`,
        frame(el, S) {
            const R = S.run; if (!R) return;
            const q = (s) => el.querySelector(s), done = R.step >= R.total, ev = P05.lastEval(R), tok = R.step * R.B * P05.T;
            const fill = q('.p05-fill'); fill.style.width = (100 * R.step / R.total).toFixed(2) + '%'; fill.classList.toggle('done', done);
            q('.p05-ptxt').textContent = done ? `训练完成 · ${R.total} 步（停在最终状态；改超参数或点 ↻ 重新训练）` : `训练中 · 第 ${R.step} / ${R.total} 步`;
            const gf = R.ms > 0 ? 6 * R.nParams * R.B * P05.T / (R.ms / 1000) / 1e9 : 0;
            q('.p05-kpis').innerHTML = [
                `train loss ${LM.fmt(P05.recentLoss(R, 50), 3)}（最近 50 步平均）`, ev ? `val loss ${ev.val.toFixed(3)} @ ${ev.step}` : 'val —', `η = ${P05.efmt(LM.cosineLR(Math.min(R.step, R.total - 1), { maxLR: R.maxLR, minLR: R.minLR, warmup: R.warmup, total: R.total }))}`,
                `已见 ${P05.kfmt(tok)} token（${(tok / S.D.train.length).toFixed(2)} epoch）`, `6ND = ${P05.efmt(6 * R.nParams * tok)} FLOP`, `实测 ${R.ms.toFixed(1)} ms/步 ≈ ${gf.toFixed(2)} GFLOP/s`,
            ].map((s) => `<span class="lm-kpi">${s}</span>`).join('');
            const xs = [], ys = []; R.evals.forEach((e) => { xs.push(e.step); ys.push(e.val); });
            P05.chart(q('canvas'), {
                x: [0, P05.xSpan(R)], y: [0.4, 5], logY: true, padR: 92, yTicks: [0.5, 1, 2, 4], tx: 8,
                series: [{ y: R.loss, color: LM.C.train, width: 1, alpha: 0.22 }, { y: P05.smoothC(R.loss, R.step, 25), color: LM.C.train, width: 1.8 }, { x: xs, y: ys, color: LM.C.val, width: 1.6, dots: 2 }],
                refs: [S.refs[2], S.refs[S.refs.length - 1]], marks: [{ x: R.step, color: LM.C.yellow }],
                legend: [{ label: 'train（±25 步平均）', color: LM.C.train }, { label: 'val', color: LM.C.val }],
            });
            const segHtml = (segs) => segs.map((s) => `<span class="${s.ok === true ? 'p05-ok' : s.ok === false ? 'p05-bad' : 'p05-part'}">${s.text.replace(/</g, '&lt;')}</span>`).join('');
            const facts = ev && ev.probe ? Array.from(ev.probe).filter((p) => p > 0.5).length : 0;
            q('.p05-side').innerHTML = ev ? `<b>第 ${ev.step} 步的样本</b>（从「。」开始续写 30 字，温度 1）：<br>${segHtml(ev.segs)}<br>
                <span class="p05-dim">绿 = 176 种合法句子之一 · 红 = 不存在的句子 · 灰 = 未写完</span><br>
                <b>事实探针</b>：24 条事实中 P(正确答案) &gt; 0.5 的有 <b>${facts}</b> 条（Step 08）<br>
                <span class="p05-dim">参照：离线训练的基座模型（batch 16 × 4000 步）val loss = ${LM_WEIGHTS.base.valLoss.toFixed(3)}；这个语料在 32 字上下文下的理论最优 = ${S.bayes.mean.toFixed(3)}</span>` : '';
        },
    }],

    intro: {
        title: '预训练：让一个随机初始化的 Transformer 学会「接着说」',
        flow: ['语料 → token id', '随机窗口 x / y', '前向 → logits', '交叉熵 loss', '反向 → 梯度', '裁剪 + AdamW', '学习率调度', '评估 · 采样'],
        html: LM.navHtml('08') + `
            <p>第 07 篇逐层拆开了一次前向传播，用的是一份<strong>已经训练好</strong>的权重。这一篇从<strong>随机初始化</strong>开始，把同一个迷你 GPT（2 层、4 头、d = 32，共 28 928 个参数）在你的浏览器里真正训练一遍：每一帧取一批随机文本窗口 → 前向算交叉熵 → 反向传播 → 梯度裁剪 → AdamW 更新。第 01 篇的四件套（模型、损失、梯度、优化器）一个不少，只是参数从 2 个变成了近 3 万个。</p>
            <p>训练数据是一个程序生成的「玩具世界」：8 种动物 × 3 个属性 = 24 条事实，用 10 个句式反复陈述。它简单到可以<strong>精确算出</strong>语料的熵和「理论上最好能做到的 loss」——所以本页能告诉你模型离极限还差多少，也能直接检查它有没有记住事实。</p>
            <p>最上面的「训练控制台」实时显示进度，默认 5000 步（batch 2）后停在最终状态，在普通电脑上要一两分钟。改训练超参数会从头重新训练；带 <span style="color:#a45fa5">LIVE</span> 的预览画的都是当前这一步真实算出的张量。代码块是同一件事的 C++ 写法（C++17 + 标准库，可以直接编译）：迷你 GPT（与第 07 篇是同一个模型）、自动微分、AdamW 等在本系列自带的小库 <code>cpp/minml_*.h</code> 里，见节点图上方的 Common；它的随机数和页面逐位相同，同一个种子得到同样的语料、验证窗口和初始权重。悬停代码里的函数名可以查看定义。</p>`,
    },

    steps: [
        // ─────────────────────────────── 01 语料
        {
            id: 'corpus', title: '语料：一个能算出熵的「玩具世界」', tag: 'DATA',
            desc: '预训练的数据就是大量<strong>无标注的文本</strong>：没有人告诉模型这句话在讲什么，它唯一的任务是预测下一个字（自监督）。真实 LLM 的语料是网页、书籍和代码，规模在万亿 token 量级；这里用一个程序生成的玩具世界代替：2 万个句子 ≈ 21.4 万个字，字符级词表 76 个 token（4 个特殊 token 预训练用不到，留给第 09 篇的对话模板）。生成器只能写出 <strong>176 种不同的句子</strong>，每种的概率都已知——这让我们能在第 07 步算出这份语料「理论上最低的 loss」。',
            formula: [
                'corpus = s_1 s_2 s_3 …      s_i ~ P(句式) · P(动物 a) · P(动物 b | a)   // 句子独立同分布，直接首尾相接',
                'ids = encode(corpus) ∈ {0, …, 75}^N      N = 214 195（训练）, 21 592（验证，另一个随机种子）',
            ],
            code: {
                title: 'C++ · makeCorpus / buildVocab / encode', src: `
                // 一种动物和它的 3 个属性
                struct Animal { std::string name, food, home, color; };      // 名字、爱吃、住处、颜色

                // 8 种动物 × 3 个属性 = 24 条事实（与页面 JS 的 LM.world.animals 相同，顺序也相同）
                const Animal ANIMALS[8] = {
                    {"小猫", "鱼", "屋里", "白色"}, {"小狗", "骨头", "院子", "黄色"},
                    {"小兔", "萝卜", "草地", "灰色"}, {"小鸟", "虫子", "树上", "蓝色"},
                    {"熊猫", "竹子", "山里", "黑白"}, {"猴子", "香蕉", "森林", "棕色"},
                    {"小羊", "青草", "山坡", "白色"}, {"老虎", "肉", "森林", "橙色"},
                };
                // 句式的个数（fillTemplate 的 k = 0 … 9）
                const int NUM_TEMPLATES = 10;

                // 第 k 个句式；a、b 是两只不同的动物（只有句式 7、8 用到 b）
                std::string fillTemplate(int k, const Animal& a, const Animal& b) {
                    switch (k) {
                        case 0: return a.name + "喜欢吃" + a.food + "。";
                        case 1: return a.name + "住在" + a.home + "。";
                        case 2: return a.name + "是" + a.color + "的。";
                        case 3: return a.name + "最爱吃" + a.food + "，它住在" + a.home + "。";
                        case 4: return a.name + "喜欢吃" + a.food + "，所以" + a.name + "每天都吃" + a.food + "。";   // 后半句要「抄」前文
                        case 5: return a.home + "住着" + a.name + "。";
                        case 6: return "今天" + a.name + "在" + a.home + "吃了很多" + a.food + "。";
                        case 7: return a.name + "和" + b.name + "是好朋友。";
                        case 8: return a.name + "吃" + a.food + "，" + b.name + "吃" + b.food + "。";
                        default: return a.color + "的" + a.name + "喜欢吃" + a.food + "。";
                    }
                }

                // 随机写一句：两只动物独立均匀地抽（第二只撞上第一只时取下一只），句式也均匀地抽
                std::string sentence(Rng& r) {
                    int i = randint(r, 8), j = randint(r, 8);       // 消耗随机数的顺序与页面 JS 的 LM.world.sentence 相同
                    if (j == i) j = (i + 1) % 8;
                    int k = randint(r, NUM_TEMPLATES);
                    return fillTemplate(k, ANIMALS[i], ANIMALS[j]);
                }

                // n 个句子首尾相接成一条长文本（不加分隔符：模型看到的是不分句的一长串字）
                std::string makeCorpus(int nSentences, uint32_t seed) {
                    Rng r = makeRng(seed);                          // 与页面 JS 的 LM.rng 逐位相同 → 同一个种子得到逐字相同的语料
                    std::string text;
                    for (int i = 0; i < nSentences; i++) text += sentence(r);
                    return text;
                }

                // 字符级词表：4 个特殊 token + 玩具世界用到的全部汉字 / 标点，按首次出现的顺序编号 → 与页面的 id 完全一致
                Vocab buildVocab() {
                    Vocab v;
                    v.itos = {"<pad>", "<eos>", "<user>", "<bot>"};                    // 预训练语料里不出现，留给第 09 篇的对话模板
                    std::vector<std::string> texts;
                    for (const Animal& a : ANIMALS) { texts.push_back(a.name); texts.push_back(a.food); texts.push_back(a.home); texts.push_back(a.color); }
                    for (int k = 0; k < NUM_TEMPLATES; k++) texts.push_back(fillTemplate(k, ANIMALS[0], ANIMALS[1]));
                    texts.push_back("小猫喜欢吃什么？"); texts.push_back("小猫住在哪里？"); texts.push_back("小猫是什么颜色的？");   // 第 09 篇的问句用到的字
                    for (const std::string& s : texts)
                        for (const std::string& ch : utf8Chars(s))                    // 一个汉字 / 标点 = 一个 token
                            if (tokenId(v, ch) < 0) v.itos.push_back(ch);
                    return v;                                                         // 共 76 个 token
                }

                // 训练 / 验证数据：两个随机种子各生成一份语料，编码成 id 序列（encode / decode 见 Common 的 minml_gpt.h）
                struct CorpusData { Vocab vocab; std::vector<int> train, val; };

                CorpusData loadCorpus() {
                    CorpusData D;
                    D.vocab = buildVocab();
                    D.train = encode(D.vocab, makeCorpus(20000, SEED.corpus));     // 214 195 个 id
                    D.val = encode(D.vocab, makeCorpus(2000, SEED.val));           // 21 592 个 id
                    return D;
                }`,
            },
            previews: [
                {
                    id: 'corpusText', label: '语料开头的 24 句', sub: '蓝 = 动物名 · 绿 = 事实（食物 / 住处 / 颜色）· 灰 = 句式文字 · 交替底色 = 句子边界（模型看到的是不分句的一长串字）', large: true, size: [720, 214],
                    render: (c, S) => {
                        const line = [];
                        S.head.forEach((segs, i) => segs.forEach((g) => line.push({ t: g.t, color: g.kind === 'name' ? LM.C.blue : g.kind === 'fact' ? LM.C.green : '#9aa0a8', bg: i % 2 ? '#16202a' : '#101418' })));
                        LM.draw.text(c, [line, { text: '', color: LM.C.dim }, { text: `训练语料共 ${S.D.train.length.toLocaleString()} 个字（2 万句）· 验证语料 ${S.D.val.length.toLocaleString()} 个字 · 词表 ${LM.vocab.size} · 不同句子 ${S.G.list.length} 种`, color: LM.C.gray, px: 11 }], { px: 14, lineGap: 7 });
                    },
                },
                {
                    id: 'facts', label: '玩具世界的 24 条事实', sub: '8 种动物 × 3 个属性 · 第 08 步的事实探针逐条检查模型记住了没有', size: [360, 230],
                    render: (c) => {
                        const { ctx, w } = PL.draw.ctx(c), padL = 56, padT = 26, cw = (w - padL - 6) / 3, ch = 24;
                        ctx.font = LM.FONT(11, true); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                        P05.ATTRS.forEach((a, j) => { ctx.fillStyle = a.color; ctx.fillText(a.label, padL + (j + 0.5) * cw, padT - 12); });
                        LM.world.animals.forEach((a, i) => {
                            const y = padT + i * ch;
                            ctx.fillStyle = i % 2 ? '#12171d' : '#0e1216'; ctx.fillRect(4, y, w - 8, ch);
                            ctx.font = LM.FONT(13); ctx.fillStyle = LM.C.blue; ctx.textAlign = 'right'; ctx.fillText(a.name, padL - 8, y + ch / 2);
                            P05.ATTRS.forEach((at, j) => { ctx.fillStyle = LM.C.green; ctx.textAlign = 'center'; ctx.fillText(a[at.key], padL + (j + 0.5) * cw, y + ch / 2); });
                        });
                    },
                },
                {
                    id: 'unigram', label: '字频（训练语料）', sub: '出现最多的 20 个字 · 柱高 = 占全部 token 的比例', size: [360, 230],
                    render: (c, S) => {
                        const idx = LM.topk(S.D.cnt, 20), n = S.D.train.length;
                        LM.draw.bars(c, idx.map((i) => S.D.cnt[i] / n), idx.map((i) => LM.vocab.show(i)), { color: LM.C.blue, digits: 2, title: `H₁ = −Σ p·ln p = ${S.D.H1.toFixed(3)} nats（ln 76 = 4.331）`, labelPx: 11 });
                    },
                },
            ],
            notes: '语料固定（种子 2024，与离线训练基座模型时相同），改超参数不会改变它。「。」「小」「吃」最常见：只按字频猜，平均每个字要付出 H₁ ≈ 3.8 nats 的 loss——这是模型要越过的第一道台阶（第 07 步）。',
            warn: '真实语料里绝大多数知识只出现寥寥几次，token 分布是重尾的；这里每条事实出现上千次。所以迷你 GPT 能在几千步里把 24 条事实全记住，大模型的事实记忆要难得多。',
        },
        // ─────────────────────────────── 02 batch
        {
            id: 'batch', title: '训练样本：随机窗口，y = x 右移一位', tag: 'BATCH',
            desc: '把整份语料当成一条 21 万个 id 的长序列。每一步随机取 B 个起点 s，切出 B 个长度 T + 1 = 33 的片段：前 32 个是输入 x，后 32 个（右移一位）是目标 y。<strong>一个窗口同时提供 32 个训练样本</strong>：位置 t 只能看 x₀…x<sub>t</sub>（因果掩码，第 07 篇），去预测 y<sub>t</sub> = x<sub>t+1</sub>。窗口不必从句首开始，句子被截断也没关系——模型必须学会在任何位置「接着说」。',
            formula: [
                's_b ~ U{0, …, N − T − 2}      x_b = ids[s_b : s_b + T]      y_b = ids[s_b + 1 : s_b + T + 1]',
                'y_{b,t} = x_{b,t+1}   // 每个位置都预测「下一个 token」；一步 = B · T 个预测（默认 2 × 32 = 64）',
            ],
            code: {
                title: 'C++ · 取一个 batch', src: `
                #include <cassert>

                // 取一个 batch 看看（getBatch 见 Common 的 minml_gpt.h）：x 与 y 是同一段文字，y 整体左移了一个字。
                // 页面训练时 r = makeRng(seed · 7919 + 11)：每一步都从这个随机数流里取 B 个窗口起点
                void showBatch(const CorpusData& D, Rng& r) {
                    const int B = 2, T = 32;
                    Batch b = getBatch(D.train, B, T, r);           // b.x、b.y 各 B·T 个 id：第 i 条窗口的位置 t 在下标 i·T + t
                    std::vector<int> x0 = ints(T), y0 = ints(T);    // 第 0 条窗口
                    for (int t = 0; t < T; t++) { x0[t] = b.x[t]; y0[t] = b.y[t]; }
                    std::printf("x = %s\\ny = %s\\n", decode(D.vocab, x0).c_str(), decode(D.vocab, y0).c_str());
                    for (int i = 0; i < B; i++)
                        for (int t = 0; t + 1 < T; t++) assert(b.y[i * T + t] == b.x[i * T + t + 1]);   // y_t == x_{t+1}
                }`,
            },
            previews: [
                {
                    id: 'batchGrid', live: true, label: '当前 batch 的 x 与 y', sub: '每条窗口两行：x（输入）和 y（目标）· 黄框：位置 10 的输入 x₁₀ 与它要预测的 y₁₀ = x₁₁ · 颜色 = token id', large: true, size: [720, 212],
                    render: (c, S) => {
                        const R = S.run; if (!R || !R.bx) return;
                        const nb = Math.min(R.B, 3), rows = [], tq = 10, ch = Math.min(26, Math.floor((212 - 60 - nb * 10) / (2 * nb)));   // 最多画 3 条：再多字号就小于 11 px
                        for (let b = 0; b < nb; b++) {
                            rows.push({ label: `x${'₀₁₂₃'[b]} s=${R.bstarts[b]}`, labelColor: LM.C.text, cells: Array.from(R.bx.subarray(b * 32, b * 32 + 32), (id, t) => ({ id, bg: LM.tokColor(id), ring: t === tq || t === tq + 1 ? (t === tq ? LM.C.yellow : 'rgba(232,194,90,0.45)') : null })), gap: 1 });
                            rows.push({ label: `y${'₀₁₂₃'[b]}`, labelColor: LM.C.green, cells: Array.from(R.by.subarray(b * 32, b * 32 + 32), (id, t) => ({ id, bg: LM.tokColor(id, 0.85), ring: t === tq ? LM.C.green : null })), gap: 9 });
                        }
                        const g = P05.tokenGrid(c, rows, { title: `第 ${R.step} 步的 batch（B = ${R.B}${R.B > 3 ? '，只画前 3 条' : ''}）· 位置编号 0…31`, colTicks: 5, padL: 82, cellH: ch });
                        const sh = (i) => LM.vocab.show(i), ctx = g.ctx;
                        ctx.font = LM.FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = LM.C.gray;
                        ctx.fillText(`例（第 0 条）：位置 10 看得到「…${Array.from(R.bx.subarray(5, 11), sh).join('')}」，要预测 y₁₀ = x₁₁ =「${sh(R.by[10])}」；每条窗口给出 32 个这样的样本`, 8, g.y + 4);
                    },
                },
                {
                    id: 'coverage', live: true, label: '窗口落在语料的哪里', sub: '横轴 = 整份训练语料（21.4 万字）· 色带 = 每段已被采样的平均次数 · 黄线 = 本步的 B 个窗口', size: [720, 96],
                    render: (c, S) => {
                        const R = S.run; if (!R) return;
                        const { ctx, w, h } = PL.draw.ctx(c), N = S.D.train.length, nb = R.cover.length, x0 = 8, x1 = w - 8, y0 = 24, y1 = 58, ep = R.step * R.B * 33 / N;
                        let mx = 0; for (const v of R.cover) mx = Math.max(mx, v);
                        for (let i = 0; i < nb; i++) { const hc = P05.heatColor(mx > 0 ? R.cover[i] / Math.max(mx, 2 * ep) : 0, 'viridis'); ctx.fillStyle = hc.bg; ctx.fillRect(x0 + (x1 - x0) * i / nb, y0, (x1 - x0) / nb + 0.6, y1 - y0); }
                        ctx.strokeStyle = '#3a3f46'; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                        if (R.bstarts) { ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 2; R.bstarts.forEach((s) => { const x = x0 + (x1 - x0) * s / N; ctx.beginPath(); ctx.moveTo(x, y0 - 5); ctx.lineTo(x, y1 + 5); ctx.stroke(); }); }
                        ctx.font = LM.FONT(11); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`第 ${R.step} 步 · 已采样 ${(R.step * R.B).toLocaleString()} 个窗口 = ${P05.kfmt(R.step * R.B * 32)} 个预测 · 平均每个字被看到 ${ep.toFixed(2)} 次（≈ epoch 数）`, P05.LIVE_X, 6);
                        ctx.fillStyle = LM.C.dim; ctx.textBaseline = 'top'; ctx.fillText('0', x0, y1 + 6); ctx.textAlign = 'right'; ctx.fillText(N.toLocaleString(), x1, y1 + 6);
                    },
                },
            ],
            notes: '拖 <b>batch B</b>：每步的窗口数随之改变（这里最多画前 3 条）。随机窗口会均匀地扫过整份语料：默认设置 5000 步一共 32 万个预测，每个字平均被看到约 1.5 次（≈ 1.5 个 epoch）。',
            warn: '这种「随机取窗口」的采样来自 nanoGPT 一类的小规模实现（窗口可能重叠、重复）。大规模预训练一般把文档拼接后顺序切块，每个 token 大约只看一遍；文档之间插入一个结束符（GPT-2 的 &lt;|endoftext|&gt;），所以窗口里会出现跨文档的拼接。',
        },
        // ─────────────────────────────── 03 前向与损失
        {
            id: 'loss', title: '前向与损失：每个位置一个交叉熵', tag: 'LOSS',
            desc: '前向传播（第 07 篇逐层拆过）给每个位置输出 76 个 logits，softmax 之后就是「下一个字」的概率分布；损失是正确答案的负对数概率 −log p(y<sub>t</sub>)，对 B·T 个位置取平均。<strong>刚初始化时 loss ≈ ln 76 = 4.33</strong>：权重是 N(0, 0.02) 的小随机数，logits 几乎都是 0，softmax 接近均匀分布，每个字的概率 ≈ 1/76——开训前先看这个数，是检查初始化有没有问题的第一道关。热图是当前 batch 每个位置的 −log p：越亮表示模型越「意外」。',
            formula: [
                'p_t = softmax(z_t)      z_t = h_t · W_te^T ∈ R^{76}   // 权重共享：LM head 就是 token embedding 矩阵',
                'ℓ_{b,t} = −log p_{b,t}[y_{b,t}]      L = (1 / BT) Σ_{b,t} ℓ_{b,t}   // 交叉熵 = 负对数似然',
                '初始化：z ≈ 0  ⇒  p ≈ 1/V  ⇒  L_0 ≈ ln V = ln 76 = 4.33',
            ],
            code: {
                title: 'C++ · 前向、逐位置损失', src: `
                // 一次前向 + 交叉熵。rowLoss = 每个位置的 −log p(y_t)（上面的热图），probs = 每个位置的 softmax（右上图）
                struct LossOut { Var loss; std::vector<float> rowLoss, probs; };

                LossOut forwardLoss(GPT& m, const Batch& b, int B, int T) {
                    LossOut o;
                    Var logits = gptForward(m, b.x, B, T);                          // [B·T × V]：每个位置 76 个 logits
                    o.loss = crossEntropy(logits, b.y, nullptr, 0.f, &o.rowLoss);   // 1×1 的 L = 平均 −log p(y)，记进 tape 供反向
                    // 等价的显式写法：每一行做 log-softmax，取正确答案那一项的 −log p，再对 B·T 个位置平均
                    int n = B * T, V = m.cfg.V;
                    o.probs = floats(n * V);
                    double sum = 0;
                    for (int i = 0; i < n; i++) {
                        const float* z = &logits->val.v[i * V];                     // 位置 i 的 76 个 logits
                        float mx = z[0];
                        for (int j = 1; j < V; j++) mx = std::max(mx, z[j]);
                        double s = 0;                                               // Σ exp(z − max)：先减去最大值，exp 不会溢出
                        for (int j = 0; j < V; j++) s += std::exp(z[j] - mx);
                        for (int j = 0; j < V; j++) o.probs[i * V + j] = (float)(std::exp(z[j] - mx) / s);
                        sum += -(z[b.y[i]] - mx - std::log(s));                     // log p(y) = z_y − max − log Σ，不经过 softmax 再取 log
                    }
                    assert(std::fabs(sum / n - o.loss->val.v[0]) < 1e-4);
                    return o;
                }

                // 刚初始化的模型：logits ≈ 0 → 每个字的概率 ≈ 1/76 → loss ≈ ln 76 = 4.33
                void checkInitLoss(const CorpusData& D, Rng& r) {
                    GPTConfig cfg;                                    // V = 76、T = 32、d = 32、L = 2、H = 4（与页面相同）
                    GPT m = makeGPT(cfg, 1);                          // N(0, 0.02)，残差输出投影再除以 √(2L)；种子 1 = 页面的初始权重
                    Batch b = getBatch(D.train, 2, cfg.T, r);
                    LossOut o = forwardLoss(m, b, 2, cfg.T);
                    std::printf("loss = %.3f, ln V = %.3f\\n", o.loss->val.v[0], std::log((float)cfg.V));   // 初始化时都 ≈ 4.33
                    clearTape();                                      // 只看 loss、不反向：丢掉这次前向记下的 tape
                }`,
            },
            previews: [
                {
                    id: 'lossHeat', live: true, label: '当前 batch 的逐位置损失 −log p(y<sub>t</sub>)', sub: '格子里写的是目标字 y<sub>t</sub> · 颜色 = 损失（0 → 6 nats，inferno）· B ≤ 2 时每条下面多一行「模型的第一猜测」（绿 = 猜对，红 = 猜错）· <b>点击格子</b>看模型在那里的完整预测', large: true, size: [720, 198],
                    render: (c, S) => {
                        const R = S.run; if (!R || !R.rowLoss) return;
                        const nb = Math.min(R.B, P05.NSHOW), sel = S.sel || P05.autoSel(R), rows = [], guess = nb <= 2 && R.probs, Vn = LM.vocab.size;
                        const ch = guess ? 26 : Math.min(30, Math.floor((198 - 64 - nb * 4) / nb));
                        for (let b = 0; b < nb; b++) {
                            rows.push({ key: b, label: `b = ${b}`, cells: Array.from(R.by.subarray(b * 32, b * 32 + 32), (id, t) => { const hc = P05.heatColor(R.rowLoss[b * 32 + t] / 6, 'inferno'); return { id, bg: hc.bg, fg: hc.fg, ring: sel.b === b && sel.t === t ? LM.C.cyan : null }; }), gap: guess ? 1 : 4 });
                            if (guess) rows.push({ key: null, label: '猜', labelColor: LM.C.dim, cells: Array.from(R.by.subarray(b * 32, b * 32 + 32), (id, t) => { const k = LM.argmax(R.probs, (b * 32 + t) * Vn, Vn); return { id: k, bg: k === id ? 'rgba(90,208,122,0.22)' : 'rgba(224,108,117,0.20)', fg: k === id ? LM.C.green : LM.C.red }; }), gap: 8 });
                        }
                        const g = P05.tokenGrid(c, rows, { title: `第 ${R.step - 1} 步（更新前）· batch 平均 loss = ${R.lastLoss.toFixed(3)}${R.step <= 1 ? '（ln 76 = 4.331）' : ''}${S.sel ? '' : ' · 青框 = 自动选中的最大 loss 格'}`, colTicks: 5, padL: 46, cellH: ch });
                        S._lossGeom = g.grid;
                        const ctx = g.ctx, x0 = g.padL, wbar = 160, yb = g.y + 6;   // 色标
                        for (let i = 0; i < wbar; i++) { ctx.fillStyle = P05.heatColor(i / (wbar - 1), 'inferno').bg; ctx.fillRect(x0 + i, yb, 1.2, 8); }
                        ctx.font = LM.FONT(10); ctx.fillStyle = LM.C.dim; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText('0', x0 - 4, yb + 4); ctx.textAlign = 'left'; ctx.fillText('6 nats', x0 + wbar + 4, yb + 4);
                        if (guess) { let ok = 0; for (let i = 0; i < nb * 32; i++) if (LM.argmax(R.probs, i * Vn, Vn) === R.by[i]) ok++; ctx.fillStyle = LM.C.gray; ctx.fillText(`第一猜测命中 ${ok} / ${nb * 32}`, x0 + wbar + 60, yb + 4); }
                    },
                    probe: (x, y, S, PL, info) => {
                        const R = S.run, hit = P05.gridHit(S._lossGeom, info); if (!R || !hit || hit.key == null || !R.rowLoss) return '';
                        const i = hit.key * 32 + hit.c, Vn = LM.vocab.size, tgt = R.by[i], p = R.probs ? R.probs[i * Vn + tgt] : NaN;
                        return `b = ${hit.key}, t = ${hit.c}：目标「${LM.vocab.show(tgt)}」\np = ${p.toFixed(4)}，−log p = ${R.rowLoss[i].toFixed(3)}`;
                    },
                    pick: (x, y, S, PL, info) => { const hit = P05.gridHit(S._lossGeom, info); if (hit && hit.key != null) S.sel = { b: hit.key, t: hit.c }; },
                },
                {
                    id: 'lossTopk', live: true, label: '选中位置：模型给出的下一个字', sub: '前 8 个候选 · 绿 = 正确答案 y<sub>t</sub>', size: [360, 228],
                    render: (c, S) => {
                        const R = S.run; if (!R || !R.probs) return;
                        const sel = S.sel || P05.autoSel(R), Vn = LM.vocab.size, i = sel.b * 32 + sel.t, off = i * Vn, tgt = R.by[i];
                        const p = R.probs.subarray(off, off + Vn), idx = LM.topk(p, 8);
                        if (!idx.includes(tgt)) idx.push(tgt);
                        const ctxStr = Array.from(R.bx.subarray(sel.b * 32 + Math.max(0, sel.t - 9), sel.b * 32 + sel.t + 1), (id) => LM.vocab.show(id)).join('');
                        LM.draw.bars(c, idx.map((k) => p[k]), idx.map((k) => LM.vocab.show(k) + (k === tgt ? ' ✓' : '')), { horizontal: true, hi: idx.indexOf(tgt), digits: 3, range: [0, 1], title: `        b=${sel.b} t=${sel.t}：…${ctxStr} → ?  −log p = ${R.rowLoss[i].toFixed(2)}`, labelPx: 12 });
                    },
                },
                {
                    id: 'lossPos', live: true, label: '损失随窗口内位置的变化', sub: '32 个固定验证窗口上的平均 · 橙 = 当前模型（最近一次评估）· 绿虚线 = 贝叶斯最优（同样的窗口、同样的上下文）', size: [360, 228],
                    render: (c, S) => {
                        const R = S.run, ev = P05.lastEval(R); if (!ev) return;
                        const pos = Array.from({ length: 32 }, (_, i) => i), top = Math.min(5, Math.max(1.6, ...ev.pos) * 1.08);
                        P05.chart(c, {
                            x: [0, 31], y: [0, top], xLabel: '位置 t（能看到 t + 1 个字）', title: `第 ${ev.step} 步的评估 · val loss = ${ev.val.toFixed(3)}（当前第 ${R.step} 步）`,
                            series: [{ x: pos, y: ev.pos, color: LM.C.val, width: 2, dots: 1.8 }, { x: pos, y: S.bayes.pos, color: LM.C.green, width: 1.5, dash: [5, 3] }],
                            legend: [{ label: '模型', color: LM.C.val }, { label: '贝叶斯最优', color: LM.C.green, dash: [5, 3] }],
                        });
                    },
                },
            ],
            notes: '亮格子大多落在<b>句子的第一个字</b>（下一句讲哪只动物、用哪个句式，本来就猜不到）和<b>还没学会的事实</b>上；「喜欢」后面的「吃」这种固定搭配很快就接近 0。右下：窗口开头的一两个位置上下文太少，连贝叶斯最优预测器的 loss 也接近或超过 1 nat——这部分 loss 永远降不下去；训练到最后，橙线几乎压在绿线上。',
            warn: '<code>crossEntropy</code> 直接吃 logits：内部先减去每行的最大值再算 log-softmax，数值稳定（PyTorch 的 F.cross_entropy 也是这样做的）；不要先 softmax 再取 log。targets 里的 −1 会被忽略（PyTorch 里对应 ignore_index = −1；第 09 篇 SFT 用它屏蔽问题部分）。热图画的是<strong>更新之前</strong>的预测，所以它和这一步的 loss 对应。',
        },
        // ─────────────────────────────── 04 反向
        {
            id: 'grad', title: '反向传播：36 个张量的梯度与全局裁剪', tag: 'GRAD',
            desc: '<code>backward(loss)</code> 沿计算图反向走一遍，得到全部 36 个参数张量（28 928 个数）的梯度。左图是每个权重矩阵的梯度范数（对数轴；22 个 LayerNorm / 偏置向量合并成一项）。刚初始化时 W<sub>q</sub>、W<sub>k</sub> 的梯度比其他矩阵小两三个数量级：注意力打分全接近 0、每个位置均匀地看所有前文，Q、K 稍微变一点几乎不影响输出；wte 最大，因为它同时是输入 embedding 和输出 LM head（权重共享）。所有梯度拼成一个长向量后，若全局范数 ‖g‖ 超过 1.0，就整体等比例缩到 1.0——<strong>梯度裁剪</strong>，防止个别「坏 batch」一步把参数推飞。',
            formula: [
                'g = ∇_θ L      ‖g‖ = √(Σ_k ‖g_k‖^2)   // 36 个张量拼成一个向量的 L2 范数',
                'g ← g · min(1, c / ‖g‖)      c = 1.0   // 只改长度，不改方向',
            ],
            code: {
                title: 'C++ · backward / 逐张量梯度范数 / 裁剪', src: `
                // 一个参数张量的梯度 L2 范数 ‖g_k‖（页面 JS 的 LM.gradNorm）
                float gradNorm(const Var& p) {
                    double s = 0;
                    for (float g : p->grad.v) s += (double)g * g;
                    return (float)std::sqrt(s);
                }

                // 反向 → 记下 36 个张量各自的梯度范数（裁剪前，左图）→ 全局裁剪
                struct GradReport { std::vector<float> norms; float total = 0; bool clipped = false; };

                GradReport backwardAndClip(GPT& m, const Var& loss, float maxNorm) {
                    GradReport r;
                    zeroGrad(m.params);                        // 梯度是累加的：反向之前先清零
                    backward(loss);                            // 逆序执行 tape：36 个参数的 ->grad 都填好了
                    for (const Var& p : m.params) r.norms.push_back(gradNorm(p));
                    r.total = clipGradNorm(m.params, maxNorm); // 返回裁剪「前」的全局范数 √(Σ_k ‖g_k‖²)
                    r.clipped = r.total > maxNorm;             // 若超过 maxNorm，所有 ->grad 已被原地乘以 maxNorm / ‖g‖
                    return r;
                }`,
            },
            previews: [
                {
                    id: 'gradBars', live: true, label: '各参数张量的梯度范数（裁剪前）', sub: '对数轴 · 白色刻线 = 第 0 步时的值', large: true, size: [360, 330],
                    render: (c, S) => {
                        const R = S.run; if (!R || R.step < 1) return;
                        const G = R.G, it = R.step - 1;
                        const v = P05.perTensor(R, R.tg, it, G, true), g0 = P05.perTensor(R, R.tg, 0, G, true);
                        const cols = G.labels.map((l) => (l.startsWith('wte') ? LM.C.orange : l.startsWith('wpe') ? LM.C.yellow : /W[qk]$/.test(l) ? LM.C.purple : l.startsWith('其余') ? LM.C.gray : LM.C.blue));
                        P05.logBars(c, v, G.labels, { range: [1e-5, 3], title: `第 ${it} 步 · 全局 ‖g‖ = ${R.gn[it].toFixed(3)}${R.gn[it] > 1 ? ' → 裁剪到 1.0' : ''}`, colors: cols, ghost: g0 });
                    },
                },
                {
                    id: 'gradNorm', live: true, label: '全局梯度范数 ‖g‖ 随训练', sub: '对数纵轴 · 细线 = 每步 · 粗线 = 滑动平均 · 红虚线 = 裁剪阈值 1.0', size: [360, 330],
                    render: (c, S) => {
                        const R = S.run; if (!R) return;
                        const sm = LM.ema(Array.from(R.gn.subarray(0, R.step)), 0.95);
                        P05.chart(c, {
                            x: [0, P05.xSpan(R)], y: [0.2, 8], logY: true, xLabel: '训练步', title: `已裁剪 ${R.nClip} / ${R.step} 步（${(100 * R.nClip / Math.max(1, R.step)).toFixed(0)}%）`,
                            series: [{ y: R.gn, color: LM.C.blue, width: 1, alpha: 0.35 }, { y: sm, color: LM.C.blue, width: 2 }],
                            refs: [{ y: 1, label: '裁剪 c = 1', color: LM.C.red }], marks: [{ x: R.step, color: LM.C.yellow }],
                        });
                    },
                },
            ],
            notes: '默认设置（B = 2）下前 1000 步里约 97% 的步被裁剪，全程约 65%：batch 小、梯度噪声大，范数常在 1 以上。把 <b>batch</b> 调到 4，全程只有约 23% 被裁剪；调到 16 只剩约 1.5%（Node 里实测）。把 <b>峰值学习率</b> 调到 3e-4，每一步都被裁剪——模型一直停在陡峭的高 loss 区域。',
            warn: '裁剪看的是<strong>全局</strong>范数（clipGradNorm；PyTorch 里是 clip_grad_norm_），不是逐张量。和 Adam 搭配时它的作用比在 SGD 里小：Adam 用 √v 把梯度尺度除掉了，整体缩放梯度几乎不改变单步更新；它主要防止一个异常大的梯度一下子把 m、v 带偏（之后十几步都受影响）。',
        },
        // ─────────────────────────────── 05 AdamW
        {
            id: 'adamw', title: 'AdamW：逐参数自适应步长 + 解耦权重衰减', tag: 'OPTIM',
            desc: 'SGD 对所有参数用同一个步长，而上一步看到不同张量的梯度能差好几个数量级。Adam 给每个参数维护两个滑动平均：梯度的 m（一阶矩，平滑掉噪声的方向）和梯度平方的 v（二阶矩，尺度），按 m̂/√v̂ 更新——<strong>每个参数的步长都被归一化到 ≈ η</strong>，不管它的梯度原本是大是小。m、v 从 0 开始，前几步偏小，用 1 − βᵗ 做偏差修正。AdamW 的权重衰减直接作用在参数上（η·λ·θ），不混进梯度，衰减强度因此不会被 √v 缩放。左图是这一步每个矩阵的相对更新 ‖Δθ‖/‖θ‖。',
            formula: [
                'm_t = β_1 m_{t−1} + (1 − β_1) g_t          v_t = β_2 v_{t−1} + (1 − β_2) g_t^2',
                'm̂_t = m_t / (1 − β_1^t)      v̂_t = v_t / (1 − β_2^t)   // 偏差修正',
                'θ_t = θ_{t−1} − η_t · ( m̂_t / (√v̂_t + ε) + λ · θ_{t−1} )   // λ 只作用于矩阵和 embedding',
                'β_1 = 0.9,  β_2 = 0.95,  ε = 10^{−8},  λ = 0.1',
            ],
            code: {
                title: 'C++ · AdamW 的一步（与 minml_nn.h 的 adamwStep 相同，另外记下 ‖Δθ‖/‖θ‖）', src: `
                // AdamW 的一步：与 Common 里 cpp/minml_nn.h 的 adamwStep 逐行相同，只是多返回每个张量的相对更新 ‖Δθ‖/‖θ‖
                // （左图；页面 JS 的 AdamW 在 step() 里同样算了 updRatio）。opt = makeAdamW(m.params, m.decay, η, 0.1f, 0.9f, 0.95f, 1e-8f)
                std::vector<float> adamwStepRatios(AdamW& o, float lr) {
                    std::vector<float> ratios = floats((int)o.params.size());
                    o.t++;                                                              // t 从 1 开始计数
                    double bc1 = 1 - std::pow((double)o.b1, o.t), bc2 = 1 - std::pow((double)o.b2, o.t);   // 偏差修正：1 − β^t
                    for (size_t k = 0; k < o.params.size(); k++) {
                        Var& p = o.params[k];
                        if (p->grad.v.empty()) continue;                                // 没收到梯度的参数不动
                        float wd = o.decay[k] ? o.wd : 0.f;                             // 只衰减矩阵和 embedding（makeGPT 登记的 m.decay）
                        double du = 0, dp = 0;
                        for (int i = 0; i < p->val.size(); i++) {
                            float g = p->grad.v[i];
                            o.m[k].v[i] = o.b1 * o.m[k].v[i] + (1 - o.b1) * g;          // 一阶矩：梯度的滑动平均
                            o.v[k].v[i] = o.b2 * o.v[k].v[i] + (1 - o.b2) * g * g;      // 二阶矩：梯度平方的滑动平均
                            double mh = o.m[k].v[i] / bc1, vh = o.v[k].v[i] / bc2;      // m̂、v̂
                            float u = (float)(lr * (mh / (std::sqrt(vh) + o.eps) + wd * p->val.v[i]));   // Δθ = η·(m̂/(√v̂ + ε) + λθ)
                            p->val.v[i] -= u;
                            du += (double)u * u; dp += (double)p->val.v[i] * p->val.v[i];
                        }
                        ratios[k] = (float)(std::sqrt(du) / (std::sqrt(dp) + 1e-12));    // ‖Δθ‖/‖θ‖（‖θ‖ 用更新后的值，与页面相同）
                    }
                    return ratios;
                }

                // 一个标量的 Adam 状态（右图跟踪的是 wte[吃][0]：k = 0 即 wte，e = tokenId(vocab, "吃") · d）。在 adamwStep 之后调用
                struct AdamTrace { float g, m; };     // g/√v̂（原始梯度，已按尺度归一）与 m̂/√v̂（实际的更新方向，更新量 = −η × 它）

                AdamTrace adamTrace(const AdamW& o, int k, int e) {
                    double vh = std::sqrt(o.v[k].v[e] / (1 - std::pow((double)o.b2, o.t))) + o.eps;
                    AdamTrace tr;
                    tr.g = (float)(o.params[k]->grad.v[e] / vh);
                    tr.m = (float)(o.m[k].v[e] / (1 - std::pow((double)o.b1, o.t)) / vh);
                    return tr;
                }`,
            },
            previews: [
                {
                    id: 'updBars', live: true, label: '各矩阵的相对更新 ‖Δθ‖ / ‖θ‖', sub: '对数轴 · 黄虚线 = 10⁻³（常用的经验参考值）· 白色刻线 = 第 0 步', large: true, size: [360, 330],
                    render: (c, S) => {
                        const R = S.run; if (!R || R.step < 1) return;
                        const G = R.G, it = R.step - 1;
                        const v = P05.perTensor(R, R.tu, it, G, false), g0 = P05.perTensor(R, R.tu, 0, G, false);
                        P05.logBars(c, v, G.labels.slice(0, -1), { range: [1e-5, 0.3], title: `第 ${it} 步 · η = ${P05.efmt(R.lrs[it])} · 中位数 ${P05.efmt(P05.median(v))}`, refs: [{ v: 1e-3, label: '10⁻³', color: LM.C.yellow }], ghost: g0, colors: G.labels.map((l) => (/W[o2]$/.test(l) ? LM.C.cyan : LM.C.blue)) });
                    },
                },
                {
                    id: 'adamTrace', live: true, label: '一个标量的 Adam 状态：wte[吃][0]', sub: '最近 200 步 · 灰 = g/√v̂（原始梯度，已按尺度归一）· 橙 = m̂/√v̂（实际更新方向）', size: [360, 330],
                    render: (c, S) => {
                        const R = S.run; if (!R || R.step < 1) return;
                        const it = R.step, a = Math.max(0, it - 200), xs = [], g = [], m = [];
                        for (let i = a; i < it; i++) { xs.push(i); g.push(R.trG[i]); m.push(R.trM[i]); }
                        P05.chart(c, {
                            x: [a, Math.max(a + 200, it)], y: [-4, 4], xLabel: '训练步', title: `第 ${it - 1} 步：m̂/√v̂ = ${R.trM[it - 1].toFixed(2)}，更新 = −η · 这个数`,
                            series: [{ x: xs, y: g, color: '#9aa0a8', width: 1, alpha: 0.8 }, { x: xs, y: m, color: LM.C.orange, width: 2 }],
                            refs: [{ y: 0, color: '#3a3f46', dash: [] }, { y: 1, color: '#2c3440' }, { y: -1, color: '#2c3440' }],
                            legend: [{ label: 'g / √v̂', color: '#9aa0a8' }, { label: 'm̂ / √v̂', color: LM.C.orange }], legendAt: 'left',
                        });
                    },
                },
                {
                    id: 'unusedEmb', live: true, label: '语料里从没出现过的 9 个 token：embedding 挤成同一个向量', sub: '4 个特殊 token 与只出现在问句里的「什么？哪颜」· 热图 = 它们的 wte 行两两之间的余弦相似度（−1 … 1）', size: [360, 300],
                    render: (c, S) => {
                        const R = S.run; if (!R) return;
                        const U = S.unused, d = P05.CFG.d, E = R.m.wte.data, n = U.length, cos = new Float32Array(n * n);
                        const rows = U.map((i) => E.subarray(i * d, (i + 1) * d)), nr = rows.map((r) => Math.hypot(...r));
                        for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { let s = 0; for (let j = 0; j < d; j++) s += rows[a][j] * rows[b][j]; cos[a * n + b] = s / (nr[a] * nr[b]); }
                        let used = 0, nu = 0; for (let i = 0; i < LM.vocab.size; i++) if (S.D.cnt[i]) { used += Math.hypot(...E.subarray(i * d, (i + 1) * d)); nu++; }
                        const lab = U.map((i) => (LM.vocab.itos[i].length > 1 ? LM.vocab.itos[i].slice(1, 4).replace('use', 'usr') : LM.vocab.itos[i]));
                        LM.draw.heat(c, cos, n, n, { rowLabels: lab, colLabels: lab, range: [-1, 1], symmetric: true, digits: 2, title: `        第 ${R.step} 步 · 范数 ${Math.min(...nr).toFixed(2)}–${Math.max(...nr).toFixed(2)}（常见 token ${(used / nu).toFixed(2)}）` });
                    },
                },
            ],
            notes: 'Adam 的第一步 m̂/√v̂ = ±1，每个元素恰好移动 η，所以相对更新 = η / 0.02（初始权重的标准差）——对照白色刻线。之后比值大体跟着学习率走（下一步）。右图：梯度本身在 ±2 之间乱跳，m 把它平均成一条平滑得多的曲线；方向一致时 m̂/√v̂ 接近 ±1，来回摇摆时接近 0——<strong>噪声大的参数自动走得慢</strong>。下图是 Adam 归一化的一个副作用：语料里没有的 9 个 token 只从权重共享的 LM head 收到梯度，而且永远是同一种「把我的概率压低」（方向 ≈ 隐状态的平均），梯度微乎其微（第 1000 步时约为普通 token 的 1/1500，用 SGD 几乎不会动），Adam 却照样让每个元素以 ≈ η 的速度朝同一方向走——随机初始化的差别很快被淹没：第 100 步余弦相似度已过 0.95，最后全是 1.00，范数是正常 token 的近 3 倍（离线训练的基座模型里也一样；第 09 篇第一次用到其中的特殊 token）。',
            warn: 'PyTorch 的 <code>torch.optim.Adam(weight_decay=…)</code> 是把 λθ 加进梯度（L2 正则），会被 √v 除掉一部分；AdamW 才是解耦的。偏置和 LayerNorm 参数通常不衰减（本系列在 makeGPT 登记参数时就用 m.decay 标好了：矩阵和 embedding 衰减，向量不衰减）。β₂ = 0.95 是 GPT-3 等大模型的选择，PyTorch 默认是 0.999。「10⁻³」只是经验值：这个小模型在峰值学习率附近到了 10⁻²–3×10⁻²，仍然训得很好。',
        },
        // ─────────────────────────────── 06 学习率调度
        {
            id: 'sched', title: '学习率调度：warmup + 余弦衰减', tag: 'SCHED',
            desc: '学习率不是常数。开头的 warmup 从 0 线性升到 η<sub>max</sub>：刚初始化时 Adam 的 v 还没估准、loss 地形又陡，一上来就用大步长容易把模型推进坏区域（第 01 篇：步长受最陡方向的曲率限制）。之后按余弦曲线降到 η<sub>min</sub> = η<sub>max</sub> / 10：后期用小步长在谷底「精修」，loss 还能再降一截——第 07 步的曲线在最后几百步往往会加速下降。',
            formula: [
                'η(t) = η_max · (t + 1) / t_warm                                          t < t_warm',
                'η(t) = η_min + ½ · (1 + cos(π · (t − t_warm) / (t_total − t_warm))) · (η_max − η_min)      t ≥ t_warm',
            ],
            code: {
                title: 'C++ · 每步的学习率', src: `
                // 整个训练的学习率表。cosineLR（Common 的 minml_nn.h）= 线性 warmup → 余弦衰减到 minLR。
                // 学习率不存在优化器里：第 it 步直接把 lrs[it] 传给 adamwStep(opt, lr)，所以每一步都可以不同
                std::vector<float> lrSchedule(float maxLR, int warmup, int total) {
                    float minLR = maxLR / 10;                                     // 页面：η_min = η_max / 10
                    std::vector<float> lrs = floats(total);
                    for (int it = 0; it < total; it++) lrs[it] = cosineLR(it, maxLR, minLR, warmup, total);
                    return lrs;   // 默认 5e-3、100、5000：lrs[0] = 5e-5，lrs[99] = 5e-3（峰值），lrs[4999] ≈ 5e-4
                }
                // …… 每一步：取 batch、前向、反向、裁剪、adamwStep(opt, lrs[it])（见 Step 07 的 trainLoop）`,
            },
            previews: [
                {
                    id: 'lrCurve', live: true, label: '学习率 η(t)', sub: '绿色区域 = warmup · 粗线 = 已经走过的部分 · 黄线 = 当前步', large: true, size: [360, 260],
                    render: (c, S) => {
                        const R = S.run; if (!R) return;
                        const n = 400, xs = [], ys = [], o = { maxLR: R.maxLR, minLR: R.minLR, warmup: R.warmup, total: R.total };
                        for (let i = 0; i <= n; i++) { const t = Math.round(i / n * (R.total - 1)); xs.push(t); ys.push(LM.cosineLR(t, o)); }
                        const cur = LM.cosineLR(Math.min(R.step, R.total - 1), o);
                        P05.chart(c, {
                            x: [0, R.total], y: [0, R.maxLR * 1.12], xLabel: '训练步', yFmt: (v) => (v === 0 ? '0' : (+v.toPrecision(2)).toExponential().replace('e+', 'e')), padL: 54,
                            bands: [{ x0: 0, x1: R.warmup, color: 'rgba(90,208,122,0.10)' }],
                            series: [{ x: xs, y: ys, color: '#3b5566', width: 1.5 }, { y: R.lrs, color: LM.C.yellow, width: 2.5 }],
                            marks: [{ x: R.step, label: `第 ${R.step} 步：η = ${P05.efmt(cur)}` }],
                            refs: [{ y: R.minLR, color: '#4a4f56' }],
                        });
                    },
                },
                {
                    id: 'ratioSched', live: true, label: '更新幅度跟着学习率走', sub: '蓝 = 14 个矩阵 ‖Δθ‖/‖θ‖ 的中位数 · 黄虚线 = η(t) / 0.02（「每个元素都恰好移动 η」时的值）', size: [360, 260],
                    render: (c, S) => {
                        const R = S.run; if (!R) return;
                        const ref = new Float32Array(R.total);
                        for (let i = 0; i < R.total; i++) ref[i] = LM.cosineLR(i, { maxLR: R.maxLR, minLR: R.minLR, warmup: R.warmup, total: R.total }) / 0.02;
                        P05.chart(c, {
                            x: [0, P05.xSpan(R)], y: [1e-4, 3], logY: true, xLabel: '训练步', title: `第 ${R.step - 1} 步：中位数 ${P05.efmt(R.medU[R.step - 1])}`,
                            series: [{ y: ref, color: LM.C.yellow, width: 1.2, dash: [5, 3] }, { y: LM.ema(Array.from(R.medU.subarray(0, R.step)), 0.9), color: LM.C.blue, width: 2 }],
                            refs: [{ y: 1e-3, color: '#5a5040' }], marks: [{ x: R.step, color: LM.C.yellow }],
                        });
                    },
                },
            ],
            notes: '把 <b>warmup 步数</b> 拖到 0：右图一开始就站在高处（第一步相对更新 = η<sub>max</sub> / 0.02 = 25%）。默认设置下 warmup = 0 最终几乎一样（Node 里实测 val 0.521 vs 0.520），只是前 1000 步事实学得慢一些（P &gt; 0.5 的事实 12 条 vs 19 条）——这个小模型配上裁剪和 Adam 相当稳；模型越大、学习率越高，跳过 warmup 越容易在开头发散。蓝线先低于黄线后差距拉大：m̂ 被噪声平均掉一部分，同时 ‖θ‖ 在训练中长大。',
            warn: 'η<sub>min</sub> = η<sub>max</sub> / 10 是 Chinchilla 论文建议的做法（衰减到峰值的 1/10）；也有实现衰减到 0，或用 WSD（warmup-stable-decay）调度：长时间常数学习率，最后一小段快速衰减。余弦调度的总长度应该和实际训练步数一致——调度提前结束或拖得太长都会吃亏。',
        },
        // ─────────────────────────────── 07 训练曲线
        {
            id: 'curve', title: '训练曲线：loss 依次越过 n-gram 的「台阶」', tag: 'TRAIN',
            desc: '蓝 = 训练 loss（细线是每个 batch，很抖；粗线是前后 ±25 步的平均，所以比当前步晚 25 步结束），橙 = 验证 loss（固定的 32 个验证窗口）。右边的参照线都是从这份语料<strong>真实统计</strong>出来的，并且和验证 loss 用同样的 32 个窗口、同样的上下文：只知道字频的 1-gram、知道前 1 个字的 2-gram……5-gram；绿色虚线是<strong>贝叶斯最优</strong>——知道生成规则的全部细节（176 种句子及其概率），只看同样的上下文，最好能做到的 loss。模型从 ln 76 出发，先学会字频，再学会字与字的搭配，然后必须用注意力去看更远的前文（抄写名字、把动物和事实对上）才能越过更高阶的台阶。',
            formula: [
                'H_n = −(1/BT) Σ_{b,t} log P̂(y_{b,t} | 前 min(n−1, t+1) 个字)   // n-gram：训练集上计数（加 0.01 平滑），验证窗口上评估',
                'L* = −(1/BT) Σ_{b,t} log P(y_{b,t} | x_{b,0} … x_{b,t})   // 贝叶斯最优：用真实生成规则做精确推断（隐马尔可夫前向算法）',
                'h = H(句子) / E[句长] = 4.763 / 10.70 = 0.445 nats/token   // 熵率：上下文无限长时的极限',
            ],
            code: {
                title: 'C++ · trainLoop / estimateLoss / ngramCrossEntropy', src: `
                #include <unordered_map>

                // 评估时刻：前 400 步每 40 步，之后每 50 步（与页面相同）
                bool shouldEval(int step) { return step <= 400 ? step % 40 == 0 : step % 50 == 0; }

                // 32 个固定验证窗口：用固定种子从验证语料里取一次，之后每次评估都用这一批（与页面的验证窗口逐个相同）
                Batch valWindows(const CorpusData& D, int T) {
                    Rng r = makeRng(SEED.valWin);
                    return getBatch(D.val, VAL_B, T, r);
                }

                // 验证 loss：固定窗口上全部 nWin·T 个位置的平均 −log p（只做前向）。
                // posLoss 非空时顺便收集每个位置 t 在各窗口上的平均（第 03 步的右下图）
                float estimateLoss(GPT& m, const Batch& val, int nWin, int T, std::vector<float>* posLoss = nullptr) {
                    NoGrad ng;                                                    // 不记录 tape
                    std::vector<float> rowLoss;
                    Var loss = crossEntropy(gptForward(m, val.x, nWin, T), val.y, nullptr, 0.f, &rowLoss);
                    if (posLoss) {
                        posLoss->assign(T, 0.f);
                        for (int i = 0; i < nWin * T; i++) (*posLoss)[i % T] += rowLoss[i] / nWin;
                    }
                    return loss->val.v[0];
                }

                // 一次评估：第几步、val loss
                struct EvalPoint { int step = 0; float val = 0; };

                // 完整的训练循环（页面每一帧只跑其中的「每帧步数」步；评估先给权重拍快照，再摊到之后的 39 步里做，结果相同）
                std::vector<EvalPoint> trainLoop(GPT& m, const std::vector<int>& trainData, const Batch& val,
                                                 float maxLR, int B, int T, int warmup, int total, uint32_t seed) {
                    AdamW opt = makeAdamW(m.params, m.decay, maxLR, 0.1f, 0.9f, 0.95f, 1e-8f);   // λ = 0.1，β₁ = 0.9，β₂ = 0.95
                    Rng r = makeRng(seed * 7919 + 11);                            // batch 顺序（与页面相同）
                    std::vector<EvalPoint> evals;
                    evals.push_back({0, estimateLoss(m, val, VAL_B, T)});
                    for (int it = 0; it < total; it++) {
                        float lr = cosineLR(it, maxLR, maxLR / 10, warmup, total);   // 第 06 步
                        Batch b = getBatch(trainData, B, T, r);                       // 第 02 步
                        zeroGrad(m.params);
                        Var loss = crossEntropy(gptForward(m, b.x, B, T), b.y);       // 第 03 步
                        backward(loss);                                               // 第 04 步
                        clipGradNorm(m.params, 1.0f);
                        adamwStep(opt, lr);                                           // 第 05 步（这 5 行就是 minml_gpt.h 的 trainStep）
                        if (shouldEval(it + 1) || it + 1 == total) evals.push_back({it + 1, estimateLoss(m, val, VAL_B, T)});
                    }
                    return evals;
                }
                // 页面的默认设置：GPT m = makeGPT(GPTConfig(), 1);  trainLoop(m, D.train, valWindows(D, 32), 5e-3f, 2, 32, 100, 5000, 1);

                // o 阶 n-gram 的计数：full[上下文·V + 下一个字]、ctx[上下文]；上下文的 o − 1 个 id 按 V 进制编码成一个整数
                struct NGramCounts { std::unordered_map<long long, int> full, ctx; };

                // n-gram 交叉熵 H_1 … H_nMax：训练集上计数（加 k 平滑），在固定验证窗口上评估；
                // 位置 t 只用窗口内已有的 min(n − 1, t + 1) 个字作上下文（与模型能看到的信息相同）
                std::vector<float> ngramCrossEntropy(const std::vector<int>& trainIds, const Batch& val, int nWin, int T,
                                                     int nMax, float k, int V) {
                    std::vector<NGramCounts> counts;                              // counts[o − 1] = o 阶的计数
                    for (int o = 1; o <= nMax; o++) {
                        NGramCounts cnt;
                        for (size_t i = 0; i + o <= trainIds.size(); i++) {
                            long long c = 0;
                            for (int j = 0; j < o - 1; j++) c = c * V + trainIds[i + j];
                            cnt.full[c * V + trainIds[i + o - 1]]++;
                            cnt.ctx[c]++;
                        }
                        counts.push_back(cnt);
                    }
                    std::vector<float> H;
                    for (int n = 1; n <= nMax; n++) {
                        double s = 0;
                        for (int b = 0; b < nWin; b++)
                            for (int t = 0; t < T; t++) {
                                int o = std::min(n, t + 2);                       // 窗口开头上下文不够长：退化成低阶
                                long long c = 0;
                                for (int j = t + 2 - o; j <= t; j++) c = c * V + val.x[b * T + j];
                                const NGramCounts& cnt = counts[o - 1];
                                long long f = c * V + val.y[b * T + t];
                                double nf = cnt.full.count(f) ? cnt.full.at(f) : 0, nc = cnt.ctx.count(c) ? cnt.ctx.at(c) : 0;
                                s -= std::log((nf + k) / (nc + k * V));
                            }
                        H.push_back((float)(s / (nWin * T)));
                    }
                    return H;   // 页面：ngramCrossEntropy(D.train, valWindows(D, 32), VAL_B, 32, 5, 0.01f, 76) = 右侧的 5 条参照线
                }`,
            },
            previews: [
                {
                    id: 'lossCurve', live: true, label: 'train / val loss 与语料的真实参照线', sub: '对数纵轴 · 右侧：ln V（均匀猜）、1–5-gram、贝叶斯最优（绿）· 都在同样 32 个验证窗口上计算', large: true, size: [720, 330],
                    render: (c, S) => {
                        const R = S.run; if (!R) return;
                        const xs = [], ys = []; R.evals.forEach((e) => { xs.push(e.step); ys.push(e.val); });
                        const ev = P05.lastEval(R);
                        P05.chart(c, {
                            x: [0, P05.xSpan(R)], y: [0.4, 5], logY: true, yTicks: [0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5], xLabel: `训练步（横轴随训练放大，最终 = 总步数 ${R.total}）`, padR: 104,
                            title: `第 ${R.step} 步 · train（最近 50 步平均）${LM.fmt(P05.recentLoss(R, 50), 3)} · val ${ev ? ev.val.toFixed(3) + ' @' + ev.step : '—'} · 离贝叶斯最优还差 ${ev ? (ev.val - S.bayes.mean).toFixed(3) : '—'}`,
                            series: [{ y: R.loss, color: LM.C.train, width: 1, alpha: 0.28 }, { y: P05.smoothC(R.loss, R.step, 25), color: LM.C.train, width: 2.2 }, { x: xs, y: ys, color: LM.C.val, width: 1.8, dots: 2.2 }],
                            refs: S.refs, marks: [{ x: R.step, color: LM.C.yellow }],
                            legend: [{ label: 'train（每步）', color: 'rgba(106,176,224,0.5)' }, { label: 'train ±25 步平均', color: LM.C.train }, { label: 'val（32 个固定窗口）', color: LM.C.val }],
                        });
                    },
                },
                {
                    id: 'stages', live: true, label: '越过每道台阶的时刻', sub: '验证 loss 第一次低于参照线的评估步', size: [360, 196],
                    render: (c, S) => {
                        const R = S.run; if (!R) return;
                        const lines = [[{ t: `        第 ${R.step} 步 · B = ${R.B} · η_max = ${P05.efmt(R.maxLR, 0)}`, color: LM.C.text }]];
                        S.refs.slice(1, -1).forEach((r) => {
                            const e = R.evals.find((q) => q.val < r.y);
                            lines.push([{ t: r.label.padEnd(13, ' '), color: r.color }, e ? { t: `第 ${e.step} 步越过（已见 ${P05.kfmt(e.tokens)} token）`, color: LM.C.text } : { t: '还没越过', color: LM.C.dim }]);
                        });
                        const ev = P05.lastEval(R), bz = S.refs[S.refs.length - 1];
                        lines.push([{ t: bz.label.padEnd(11, ' '), color: bz.color }, { t: ev ? `下限：当前 val 还高 ${(ev.val - S.bayes.mean).toFixed(3)} nats` : '下限', color: LM.C.green }]);
                        lines.push({ text: `熵率（无限长上下文）h = ${S.G.rate.toFixed(3)}；离线训练的基座模型 val = ${LM_WEIGHTS.base.valLoss.toFixed(3)}`, color: LM.C.dim, px: 11 });
                        LM.draw.text(c, lines, { px: 12, lineGap: 6 });
                    },
                },
            ],
            notes: '<b>峰值学习率</b>扫描（默认设置跑满 5000 步，Node 里用同一份代码实测的最终 val loss）：3e-4 → 0.81，1e-3 → 0.56，2e-3 → 0.54，<b>5e-3 → 0.52</b>，1e-2 → 0.53，3e-2 → 0.59——典型的 U 形，最好的学习率离「训不稳」不远（第 01 篇）。3e-4 时到第 1600 步才越过 2-gram，最后停在 0.81；3e-2 时第 1000–2000 步一直卡在 0.86–1.05，学习率衰减下来才追上。<b>batch</b> 2 / 4 / 8 / 16 的最终 val 为 0.520 / 0.512 / 0.513 / 0.508，越过 5-gram 分别在第 1700 / 1050 / 650 / 320 步——按步数大 batch 快得多，按 token 算（10.9 万 / 13.4 万 / 16.6 万 / 16.4 万）小 batch 反而更省，这就是「临界 batch size」的权衡。验证 loss 与训练 loss 几乎重合：没有过拟合。',
            warn: '这个世界只有 176 种句子，验证集里的句子训练集里全都有，所以这里的 val loss 衡量的是「学没学到分布」，不衡量对新内容的泛化。贝叶斯线是在这 32 个验证窗口上算的（换成 2000 个随机窗口是 0.494）：验证 loss 不会系统性地低于它，而训练曲线来自随机 batch，后期可以略低于这条线。真实预训练每个 token 基本只见一次，train / val 也很接近；过拟合主要出现在小数据多 epoch 的时候。',
        },
        // ─────────────────────────────── 08 探针
        {
            id: 'probe', title: '模型在学什么：采样快照与事实探针', tag: 'PROBE',
            desc: 'loss 是一个平均数，看不出模型到底会了什么。两种更直观的看法：（1）定期让当前模型从「。」开始自己写 30 个字（温度 1、每次同一个随机种子，所以样本的变化只来自权重的变化）：第 0 步是乱码，一两百步后有了句式，事实还常常张冠李戴（「小兔吃竹子」），再往后才全部说对——完整句子是 176 种合法句子之一就标绿，否则标红。（2）事实探针：直接读出 P(骨头 | 。小狗喜欢吃) 这样的概率，24 条事实一起测。',
            formula: [
                'P(答案 | 提示) = Π_i p(a_i | 提示, a_{<i})   // 例：P(骨头 | 。小狗喜欢吃) = p(骨 | …吃) · p(头 | …吃骨)',
            ],
            code: {
                title: 'C++ · answerProb / checkSentences', src: `
                #include <set>

                // P(答案 | 提示)：一次前向（teacher forcing）读出答案每个字的条件概率，连乘
                float answerProb(GPT& m, const Vocab& v, const std::string& prompt, const std::string& answer) {
                    NoGrad ng;
                    std::vector<int> ids = encode(v, prompt + answer), ans = encode(v, answer);
                    int n = (int)ids.size() - (int)ans.size(), V = m.cfg.V;      // n = 提示的 token 数
                    Var logits = gptForward(m, ids, 1, (int)ids.size());
                    double lp = 0;
                    for (int i = 0; i < (int)ans.size(); i++) {
                        std::vector<float> p = floats(V);
                        for (int j = 0; j < V; j++) p[j] = logits->val.at(n - 1 + i, j);   // 位置 n − 1 + i 预测答案的第 i 个字
                        softmaxInPlace(p, 1.f);
                        lp += std::log(std::max(p[ans[i]], 1e-30f));
                    }
                    return (float)std::exp(lp);
                }

                // 24 条事实探针，下标 = 动物 × 3 + 属性：「。X喜欢吃」→ 食物，「。X住在」→ 住处，「。X是」→ 颜色
                std::vector<float> runProbes(GPT& m, const Vocab& v) {
                    std::vector<float> P;
                    for (const Animal& a : ANIMALS) {
                        P.push_back(answerProb(m, v, "。" + a.name + "喜欢吃", a.food));
                        P.push_back(answerProb(m, v, "。" + a.name + "住在", a.home));
                        P.push_back(answerProb(m, v, "。" + a.name + "是", a.color));
                    }
                    return P;
                }

                // 176 种合法句子：句式 7、8 里有两只不同的动物（8 × 7 种组合），其余 8 个句式只和第一只动物有关
                std::set<std::string> legitSentences() {
                    std::set<std::string> S;
                    for (int i = 0; i < 8; i++)
                        for (int k = 0; k < NUM_TEMPLATES; k++)
                            for (int j = 0; j < 8; j++)
                                if (j != i) S.insert(fillTemplate(k, ANIMALS[i], ANIMALS[j]));
                    return S;                                                    // 8 × 8 + 2 × 8 × 7 = 176 句
                }

                // 生成的 token 按「。」切成句子：完整句子在合法句子表里 → ok = 1，否则 0；最后没写完的半句 ok = −1（不判）
                struct Segment { std::string text; int ok = -1; };

                std::vector<Segment> checkSentences(const Vocab& v, const std::vector<int>& ids, const std::set<std::string>& legit) {
                    std::vector<Segment> segs;
                    Segment cur;
                    for (int id : ids) {
                        cur.text += v.itos[id];
                        if (v.itos[id] == "。") { cur.ok = legit.count(cur.text) ? 1 : 0; segs.push_back(cur); cur = Segment(); }
                    }
                    if (!cur.text.empty()) segs.push_back(cur);
                    return segs;
                }

                // 评估时的采样快照：从「。」开始续写 30 个字、温度 1，每次都用同一个种子 → 样本的变化只来自权重的变化
                std::vector<Segment> sampleSnapshot(const GPT& m, const Vocab& v, const std::set<std::string>& legit) {
                    Rng r = makeRng(SEED.sample);
                    std::vector<int> prompt = encode(v, "。");
                    std::vector<int> out = generate(m, prompt, 30, 1.0f, 0, 1.0f, r);    // 不截断：top-k 关、top-p = 1
                    std::vector<int> gen = out;
                    gen.erase(gen.begin(), gen.begin() + prompt.size());                   // 只判续写的部分
                    return checkSentences(v, gen, legit);
                }`,
            },
            previews: [
                {
                    id: 'snapshots', live: true, label: '各检查点的采样（同一个随机种子）', sub: '检查点：第 0 / 200 / 400 / 600 / 1000 / 1500 / 2000 / 3000 / 5000 / 8000 步 + 最近一次评估（黄）· 从「。」开始续写 30 个字 · 绿 = 合法句子 · 红 = 不存在的句子（语法或事实错误）· 灰 = 未写完', large: true, size: [720, 300],
                    render: (c, S) => {
                        const R = S.run; if (!R) return;
                        const shown = R.evals.filter((e) => P05.CKPTS.includes(e.step)), ev = P05.lastEval(R);
                        if (ev && !shown.includes(ev)) shown.push(ev);
                        const lines = [[{ t: `       第 ${R.step} 步 · 温度 1 · 每次都用同一个随机种子 ${P05.SEED.sample}`, color: LM.C.text }]];
                        shown.forEach((e) => lines.push([{ t: `${String(e.step).padStart(5, ' ')} 步 │ `, color: e === ev ? LM.C.yellow : LM.C.dim }].concat(P05.segLine(e.segs))));
                        LM.draw.text(c, lines, { px: 13, lineGap: 7 });
                    },
                },
                {
                    id: 'factHeat', live: true, label: '24 条事实的探针 P(答案 | 提示)', sub: '最近一次评估 · 格子里是正确答案及其概率', large: true, size: [360, 300],
                    render: (c, S) => {
                        const R = S.run, ev = P05.lastEval(R); if (!ev || !ev.probe) return;
                        const { ctx, w } = PL.draw.ctx(c), padL = 50, padT = 46, cw = (w - padL - 8) / 3, ch = 29;
                        ctx.font = LM.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`第 ${ev.step} 步的评估（当前第 ${R.step} 步）· P > 0.5：${Array.from(ev.probe).filter((p) => p > 0.5).length} / 24`, P05.LIVE_X, 5);
                        ctx.font = LM.FONT(11); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                        P05.ATTRS.forEach((a, j) => { ctx.fillStyle = a.color; ctx.fillText(`${a.label}（「${a.prompt({ name: 'X' }).slice(2)}」）`, padL + (j + 0.5) * cw, padT - 4); });
                        LM.world.animals.forEach((a, i) => {
                            ctx.fillStyle = LM.C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.font = LM.FONT(12); ctx.fillText(a.name, padL - 6, padT + (i + 0.5) * ch);
                            P05.ATTRS.forEach((at, j) => {
                                const p = ev.probe[i * 3 + j], hc = P05.heatColor(p, 'viridis'), x = padL + j * cw, y = padT + i * ch;
                                ctx.fillStyle = hc.bg; ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
                                ctx.fillStyle = hc.fg; ctx.textAlign = 'center'; ctx.font = LM.FONT(12); ctx.fillText(`${a[at.key]}  ${p < 0.995 ? p.toFixed(2).replace(/^0/, '') : '1.0'}`, x + cw / 2, y + ch / 2 + 0.5);
                            });
                        });
                    },
                },
                {
                    id: 'factCurve', live: true, label: '探针概率随训练', sub: '细线 = 每条事实 · 粗线 = 同一属性 8 种动物的平均', size: [360, 300],
                    render: (c, S) => {
                        const R = S.run; if (!R || !R.evals.length) return;
                        const xs = R.evals.map((e) => e.step), series = [];
                        for (let i = 0; i < 24; i++) series.push({ x: xs, y: R.evals.map((e) => e.probe[i]), color: P05.ATTRS[i % 3].color, width: 1, alpha: 0.3 });
                        P05.ATTRS.forEach((a, j) => series.push({ x: xs, y: R.evals.map((e) => { let s = 0; for (let i = 0; i < 8; i++) s += e.probe[i * 3 + j]; return s / 8; }), color: a.color, width: 2.4, label: a.label }));
                        P05.chart(c, { x: [0, P05.xSpan(R)], y: [0, 1], xLabel: '训练步', series, endLabels: true, padR: 44, marks: [{ x: R.step, color: LM.C.yellow }], title: `第 ${R.step} 步` });
                    },
                },
            ],
            notes: '探针曲线常常是「台阶」形：某一条事实在几十步之内从接近 0 跳到接近 1。默认设置下食物最先学会（第 600–850 步越过 0.9），住处其次（800–1700 步），颜色最晚（1050–2450 步）：颜色只出现在 2 个句式里，食物出现在 6 个——<strong>出现得少的知识学得晚</strong>。不同随机种子的顺序和时间会有出入（点「换随机种子」）。把 <b>峰值学习率</b> 调到 3e-4：样本里句式完全正确，却是「老虎吃萝卜」——<strong>先学语法，后学知识</strong>。',
            warn: '探针用「。」开头把动物放在句首，避免歧义（「小狗是」也可能来自「小猫和小狗是好朋友」）。白色（小猫、小羊）、森林（猴子、老虎）、山里 / 山坡共享首字，所以探针算的是<strong>整个答案</strong>的概率，而不是只看首字。',
        },
        // ─────────────────────────────── 09 解码
        {
            id: 'decode', title: '采样与解码：temperature、top-k、top-p', tag: 'DECODE',
            desc: '模型只给出下一个字的概率分布；从中怎么选字，是推理时的决定，不改变模型本身。<strong>温度</strong> τ 先除 logits：τ &lt; 1 让分布更尖（更保守、更重复），τ &gt; 1 让它更平（更多样、也更多错）。<strong>top-k</strong> 只保留概率最大的 k 个；<strong>top-p</strong>（nucleus）按概率从大到小累加，到 p 为止，砍掉长尾。截断后重新归一化再抽样。这一步用离线训练好的基座模型（第 07 篇同一份权重），保证效果稳定；上面实时训练的模型训完后表现相近。',
            formula: [
                'p_i = exp(z_i / τ) / Σ_j exp(z_j / τ)   // τ = 温度（很多资料也写作 T）',
                'top-k：保留 rank(i) ≤ k      top-p：按 p 降序保留，直到累计概率首次 ≥ p      再归一化：p_i ← p_i / Σ_{保留} p_j',
            ],
            code: {
                title: 'C++ · decodeState / tempSweep（decodeDist、generate 在 Common 的 minml_gpt.h）', src: `
                // decodeDist(logits, τ, k, p) 与 generate(...) 在 Common 的 cpp/minml_gpt.h 里（悬停可查看），与页面 JS 逐行相同。decodeDist 的顺序：
                //   ① p = softmax(z / τ)    ② 按 p 从大到小排序
                //   ③ 依次保留：到第 k 个为止（k = 0 不限），或累计概率首次 ≥ top-p 为止（跨过 top-p 的那一个也保留；累计按温度之后的完整分布算）
                //   ④ 其余置 0，保留下来的重新归一化。generate 逐 token 前向 + KV cache，每步从 decodeDist 的分布里抽一个字

                // 离线训练好的基座模型（第 07 篇同一份权重）：按 m.params 的顺序读入 float16 权重（实现略）
                GPT loadBaseModel(const std::string& path);

                // ids 之后下一个字的 logits（只做前向；超出上下文时只看最后 T 个字）
                std::vector<float> nextLogits(GPT& m, const std::vector<int>& ids) {
                    NoGrad ng;
                    int T = std::min((int)ids.size(), m.cfg.T), V = m.cfg.V;
                    std::vector<int> x = ints(T);
                    for (int t = 0; t < T; t++) x[t] = ids[ids.size() - T + t];
                    Var logits = gptForward(m, x, 1, T);
                    std::vector<float> z = floats(V);
                    for (int j = 0; j < V; j++) z[j] = logits->val.at(T - 1, j);   // 最后一个位置那一行
                    return z;
                }

                // 分布的熵 H = −Σ p·ln p（nats）
                float entropy(const std::vector<float>& p) {
                    double h = 0;
                    for (float q : p) if (q > 0) h -= q * std::log(q);
                    return (float)h;
                }

                // 上面三张图的数据：解码前 / 后的分布、保留了几个 token、两者的熵，以及用 5 个固定种子续写到上下文写满
                struct DecodeState { std::vector<float> raw, out; int kept = 0; float Hraw = 0, Hout = 0; std::vector<std::string> samples; };

                DecodeState decodeState(GPT& base, const Vocab& v, const std::string& prompt, float temperature, int topK, float topP) {
                    DecodeState d;
                    std::vector<int> ids = encode(v, prompt);
                    std::vector<float> z = nextLogits(base, ids);
                    d.raw = z;
                    softmaxInPlace(d.raw, 1.f);                                   // 原始分布：τ = 1、不截断（灰柱）
                    d.out = decodeDist(z, temperature, topK, topP);               // 温度 → top-k → top-p → 重新归一化（蓝柱）
                    for (float q : d.out) if (q > 0) d.kept++;
                    d.Hraw = entropy(d.raw);
                    d.Hout = entropy(d.out);
                    for (int k = 0; k < 5; k++) {
                        Rng r = makeRng(SEED.decoding + k);
                        std::vector<int> g = generate(base, ids, base.cfg.T - (int)ids.size(), temperature, topK, topP, r);
                        d.samples.push_back(decode(v, g));                        // 提示 + 续写，共 32 个字
                    }
                    return d;
                }

                // 纯温度采样扫描（不加 top-k / top-p）：每个温度 12 条、每条续写 31 个字，
                // 统计完整句子里合法的比例（正确率）与「不同句子数 / 句子数」（多样性）
                struct TempSweep { std::vector<float> temps, ok, div; };

                TempSweep tempSweep(const GPT& base, const Vocab& v, const std::set<std::string>& legit) {
                    TempSweep sw;
                    sw.temps = {0.1f, 0.25f, 0.4f, 0.55f, 0.7f, 0.85f, 1.0f, 1.2f, 1.4f, 1.7f, 2.0f};
                    for (float tp : sw.temps) {
                        int tot = 0, good = 0;
                        std::set<std::string> seen;
                        for (int k = 0; k < 12; k++) {
                            Rng r = makeRng(SEED.sweep + k);
                            std::vector<int> g = generate(base, encode(v, "。"), base.cfg.T - 1, tp, 0, 1.0f, r);
                            g.erase(g.begin());                                   // 去掉提示「。」
                            for (const Segment& s : checkSentences(v, g, legit)) {
                                if (s.ok < 0) continue;                           // 没写完的半句不算
                                tot++; good += s.ok; seen.insert(s.text);
                            }
                        }
                        sw.ok.push_back((float)good / std::max(1, tot));
                        sw.div.push_back((float)seen.size() / std::max(1, tot));
                    }
                    return sw;
                }`,
            },
            previews: [
                {
                    id: 'decDist', needs: ['decode'], label: '下一个字的分布：解码前 vs 解码后', sub: '按原始概率排序的前 18 个候选 · 灰 = 原始 softmax（τ = 1）· 蓝 = 温度 + top-k + top-p 之后 · 红 × = 被截掉', large: true, size: [720, 250],
                    render: (c, S) => {
                        const d = S.dec; if (!d) return;
                        const { ctx, w, h } = PL.draw.ctx(c), idx = LM.topk(d.raw, 18), n = idx.length, padL = 44, padR = 10, padT = 30, padB = 40;
                        const mx = Math.max(...idx.map((i) => Math.max(d.raw[i], d.out[i]))) * 1.08, bw = (w - padL - padR) / n;
                        const Y = (v) => h - padB - v / mx * (h - padT - padB);
                        ctx.font = LM.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`「${S.prompt}」→ ?   τ = ${S.temp.toFixed(2)} · top-k = ${S.topk || '关'} · top-p = ${S.topp.toFixed(2)}   保留 ${d.kept} / 76 个 token · 熵 ${d.Hraw.toFixed(2)} → ${d.Hout.toFixed(2)} nats`, 6, 6);
                        ctx.font = LM.FONT(10); ctx.strokeStyle = '#1b1f25'; ctx.fillStyle = '#737983'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                        [0, 0.25, 0.5, 0.75, 1].map((q) => q * mx).forEach((v) => { ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(2), padL - 4, Y(v)); });
                        idx.forEach((i, k) => {
                            const x = padL + k * bw;
                            ctx.fillStyle = '#5c636d'; ctx.fillRect(x + bw * 0.12, Y(d.raw[i]), bw * 0.36, h - padB - Y(d.raw[i]));
                            if (d.out[i] > 0) { ctx.fillStyle = LM.C.blue; ctx.fillRect(x + bw * 0.5, Y(d.out[i]), bw * 0.36, h - padB - Y(d.out[i])); }
                            else { ctx.strokeStyle = LM.C.red; ctx.lineWidth = 1.5; const cx = x + bw * 0.68, cy = h - padB - 6; ctx.beginPath(); ctx.moveTo(cx - 4, cy - 4); ctx.lineTo(cx + 4, cy + 4); ctx.moveTo(cx + 4, cy - 4); ctx.lineTo(cx - 4, cy + 4); ctx.stroke(); }
                            ctx.font = LM.FONT(13); ctx.fillStyle = d.out[i] > 0 ? LM.C.text : LM.C.dim; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(LM.vocab.show(i), x + bw / 2, h - padB + 4);
                            ctx.font = LM.FONT(9); ctx.fillStyle = LM.C.dim; ctx.fillText(d.raw[i] >= 0.01 ? d.raw[i].toFixed(2).replace(/^0/, '') : '<.01', x + bw / 2, h - padB + 21);
                        });
                    },
                },
                {
                    id: 'decSamples', needs: ['decode'], label: '用当前设置续写 5 次', sub: '5 个固定随机种子 · 绿 = 合法句子 · 红 = 不存在的句子 · 灰 = 未写完', size: [360, 250],
                    render: (c, S) => {
                        const d = S.dec; if (!d) return;
                        const lines = d.samples.map((g, k) => [{ t: `${k + 1}  `, color: LM.C.dim }, { t: S.prompt, color: LM.C.yellow }].concat(P05.segLine(P05.segments(g, S.G.legit))));
                        LM.draw.text(c, lines, { px: 13, lineGap: 9, indent: 18 });
                    },
                },
                {
                    id: 'decSweep', needs: ['sweep'], label: '温度扫描：正确率 vs 多样性', sub: '纯温度采样（不加 top-k / top-p）· 每个温度 12 条 × 31 字 · 绿 = 完整句子里合法的比例 · 紫 = 不同句子数 / 句子数 · 黄线 = 当前 τ', size: [360, 250],
                    render: (c, S) => {
                        const sw = S.sweep; if (!sw) return;
                        P05.chart(c, {
                            x: [0, 2.1], y: [0, 1.05], xLabel: '温度 τ', yTicks: [0, 0.25, 0.5, 0.75, 1],
                            series: [{ x: sw.temps, y: sw.ok, color: LM.C.green, width: 2, dots: 2.5 }, { x: sw.temps, y: sw.div, color: LM.C.purple, width: 2, dots: 2.5 }],
                            marks: [{ x: S.temp, label: `τ = ${S.temp.toFixed(2)}` }],
                            legend: [{ label: '正确率', color: LM.C.green }, { label: '多样性', color: LM.C.purple }], legendAt: 'bottom',
                        });
                    },
                },
            ],
            notes: '提示选「。」：原始分布有十几个像样的候选。把 <b>温度 τ</b> 拖到 0.3：top-p 之后只剩「小」一个候选，5 次续写都以「小」开头、多半是「小 X 喜欢吃 Y」，开始雷同。拖到 1.8：分布变平，保留的候选从 6 个变成 17 个，但 <b>top-p</b> = 0.9 仍然砍掉了最离谱的长尾，5 次续写全部合法；再把 top-p 拖到 1.0（不截断）——同样的种子，立刻出现 5 个错句。top-p 在分布平坦时保留更多候选、尖锐时保留更少，这是它比固定的 top-k 更常用的原因。换成「小猫喜欢吃」：无论怎么调都几乎只有「鱼」。',
            warn: 'τ → 0（或 top-k = 1）就是贪心解码：在这个玩具世界里几乎总是对的，但会一遍遍重复同一句话；真实 LLM 的贪心解码也容易陷入重复循环（Holtzman 等 2020）。各框架的 top-k / top-p 组合顺序不完全相同，这里的顺序见代码注释。',
        },
        // ─────────────────────────────── 10 规模化
        {
            id: 'scale', title: '规模化：6ND 算力账与 Chinchilla 定律', tag: 'SCALE',
            desc: '训练算力 ≈ 6·N·D 次浮点运算（N = 参数量，D = 训练 token 数）：每个 token 前向约 2N（每个参数一次乘加），反向约 4N。Hoffmann 等（2022）训练了 400 多个模型（7000 万到 160 多亿参数），把 loss 拟合成 L(N, D) = E + A/N^α + B/D^β：E 是文本本身不可约的熵，后两项分别是「模型太小」和「数据太少」造成的额外损失——和本页的设定一一对应：这里的 E 就是第 07 步的熵率 / 贝叶斯最优。给定算力 C，参数和数据之间存在最优的分配：他们的结论是 N 与 D 应<strong>同比例</strong>增长；Chinchilla（700 亿参数、1.4 万亿 token，≈ 20 token/参数）用与 Gopher（2800 亿参数、3000 亿 token）相近的算力取得了更低的 loss。',
            blocks: [
                {
                    type: 'formula', text: `C ≈ 6 · N · D      L(N, D) = E + A / N^α + B / D^β
                    E = 1.69,  A = 406.4,  B = 410.7,  α = 0.34,  β = 0.28   // Hoffmann et al. 2022，方法 3 的拟合值
                    N_opt(C) = G · (C/6)^{β/(α+β)}      D_opt(C) = G^{−1} · (C/6)^{α/(α+β)}      G = (αA / βB)^{1/(α+β)}`,
                },
                {
                    type: 'code', title: 'C++ · chinchillaLoss / computeOptimal / trainFlops', src: `
                    // Chinchilla 拟合式 L(N, D) = E + A/N^α + B/D^β（Hoffmann et al. 2022，方法 3 的拟合值）
                    double chinchillaLoss(double N, double D, double E = 1.69, double A = 406.4, double B = 410.7,
                                          double alpha = 0.34, double beta = 0.28) {
                        return E + A / std::pow(N, alpha) + B / std::pow(D, beta);
                    }

                    // 一种分配：参数量 N、训练 token 数 D
                    struct NDPair { double N = 0, D = 0; };

                    // 给定算力 C，在 C = 6·N·D 的约束下最小化 L：把 D = C / 6N 代入，对 N 求导令其为 0
                    NDPair computeOptimal(double C, double A = 406.4, double B = 410.7, double alpha = 0.34, double beta = 0.28) {
                        double G = std::pow(alpha * A / (beta * B), 1 / (alpha + beta));
                        NDPair o;
                        o.N = G * std::pow(C / 6, beta / (alpha + beta));
                        o.D = C / (6 * o.N);
                        return o;
                    }

                    // 训练算力 ≈ 6ND 次浮点运算：每个 token 前向约 2N（每个参数一次乘加），反向约 4N
                    double trainFlops(double nParams, double nTokens) { return 6 * nParams * nTokens; }`,
                },
                {
                    type: 'previews', items: [
                        {
                            id: 'isoflop', label: 'Chinchilla 拟合式的等算力曲线', sub: '每条曲线 = 固定算力 C（FLOPs）下 L 随 N 的变化（D = C / 6N）· 圆点 = 这组拟合值下的最优 N · 白虚线 = 20 token/参数 · 彩色点 = 真实模型的 N、D 代入公式', large: true, size: [360, 320],
                            render: (c) => {
                                const Cs = [1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24], cols = ['#3f6f8f', '#4a86a8', '#56a0c4', '#6ab0e0', '#8fc3e8', '#b4d7f0', '#d8ebf8'];
                                const series = [], pts = [], path = [];
                                Cs.forEach((Cf, k) => {
                                    const xs = [], ys = [];
                                    for (let i = 0; i <= 120; i++) { const N = Math.pow(10, 6 + 6 * i / 120); xs.push(N); ys.push(P05.chLoss(N, Cf / (6 * N))); }
                                    series.push({ x: xs, y: ys, color: cols[k], width: 1.5 });
                                    const o = P05.chOpt(Cf); pts.push({ x: o.N, y: P05.chLoss(o.N, o.D), color: cols[k], r: 3.5, label: `10${['¹⁸', '¹⁹', '²⁰', '²¹', '²²', '²³', '²⁴'][k]}`, dy: -15 });
                                    const N20 = Math.sqrt(Cf / 120); path.push([N20, P05.chLoss(N20, 20 * N20)]);
                                });
                                series.push({ x: path.map((p) => p[0]), y: path.map((p) => p[1]), color: 'rgba(255,255,255,0.7)', width: 1.2, dash: [4, 3] });
                                const mcol = [LM.C.orange, LM.C.red, LM.C.green, LM.C.purple];
                                P05.MODELS.forEach((m, i) => pts.push({ x: m.N, y: P05.chLoss(m.N, m.D), color: mcol[i], r: 4.5 }));
                                P05.chart(c, {
                                    x: [1e6, 1e12], y: [1.8, 4.2], logX: true, xLabel: 'N（参数量）', yLabel: 'L（nats/token）', series, points: pts, padL: 48,
                                    legend: P05.MODELS.map((m, i) => ({ label: `${m.name}（D/N ≈ ${Math.round(m.D / m.N)}）`, color: mcol[i], dot: true })).concat([{ label: '20 token/参数', color: 'rgba(255,255,255,0.7)', dash: [4, 3] }]), legendAt: 'bl',
                                });
                            },
                        },
                        {
                            id: 'ourScale', live: true, label: '本页这次训练：可约部分随 token 数下降', sub: '双对数 · 纵轴 = val loss − 贝叶斯最优（还能学到的部分）· 横轴 = 已训练的 token 数 D', size: [360, 320],
                            render: (c, S) => {
                                const R = S.run; if (!R) return;
                                const ev = R.evals.filter((e) => e.step > 0), xs = ev.map((e) => e.tokens), ys = ev.map((e) => Math.max(1e-3, e.val - S.bayes.mean));
                                const tok = R.step * R.B * 32, Cf = 6 * R.nParams * tok, xMax = Math.pow(10, Math.ceil(Math.log10(Math.max(1e6, R.total * R.B * 32))));
                                P05.chart(c, {
                                    x: [1e3, xMax], y: [0.01, 5], logX: true, logY: true, xLabel: '已训练 token 数 D',
                                    title: `D = ${P05.kfmt(tok)} · D/N = ${(tok / R.nParams).toFixed(1)} · C = 6ND = ${P05.efmt(Cf)} FLOPs`,
                                    series: [{ x: xs, y: ys, color: LM.C.val, width: 1.8, dots: 2.2 }], marks: [{ x: Math.max(1e3, tok), color: LM.C.yellow }],
                                });
                            },
                        },
                    ],
                },
                {
                    type: 'html', html: `<table class="lm-tbl"><tr><th>模型</th><th>参数 N</th><th>训练 token D</th><th>D / N</th><th>C ≈ 6ND（估算）</th></tr>
                        <tr><td>GPT-3</td><td>1750 亿</td><td>3000 亿</td><td>1.7</td><td>3.2 × 10²³</td></tr>
                        <tr><td>Gopher</td><td>2800 亿</td><td>3000 亿</td><td>1.1</td><td>5.0 × 10²³</td></tr>
                        <tr><td>Chinchilla</td><td>700 亿</td><td>1.4 万亿</td><td>20</td><td>5.9 × 10²³</td></tr>
                        <tr><td>LLaMA-3 8B</td><td>80 亿</td><td>15 万亿</td><td>≈ 1900</td><td>7.2 × 10²³</td></tr>
                        <tr><td>本页（默认 5000 步）</td><td>28 928</td><td>32 万</td><td>11</td><td>5.6 × 10¹⁰</td></tr></table>` },
                { type: 'note', html: '左图里圆点（拟合式的最优）和白虚线（20 token/参数）在小算力处接近，算力越大分得越开：直接最小化这组方法 3 的拟合值，10²⁴ FLOPs 时的最优配比约 100 token/参数。Hoffmann 等用三种方法估计最优配比，常说的「约 20」来自方法 1、2 和 Chinchilla 模型本身；Besiroglu 等（2024）的复现指出方法 3 公布的拟合值与前两种方法不一致。右图：本页模型的可约 loss 在双对数坐标下大致沿直线下降，后段变平——这份语料里能学的东西快学完了。拖 <b>batch</b>、<b>总步数</b> 改变 D。' },
                { type: 'warn', html: '<strong>不要把本页的 loss 代入 Chinchilla 公式比较</strong>：公式是在 MassiveText（网页、书籍、新闻、代码等）上、用 7000 万参数以上的模型拟合的，E = 1.69 是那份数据在那个分词下的熵；本页的语料只有 176 种句子，熵率 0.445。缩放定律的形状（幂律 + 不可约项）是普遍现象，常数取决于数据、分词和模型族。另外 Chinchilla 最优只最小化<strong>训练</strong>算力；要部署的模型往往远超 20 token/参数（LLaMA-3 8B 用了 15 万亿 token），多花训练算力换更小、更便宜的推理。' },
            ],
        },
        // ─────────────────────────────── 11 生产
        {
            id: 'prod', title: '教学版 → 生产版：真实预训练多了什么', tag: 'PROD',
            desc: '本页的训练循环和真实大模型预训练在算法上是同一个：随机初始化 → 取 batch → 前向 → 交叉熵 → 反向 → 裁剪 → AdamW → 调度学习率 → 评估。区别在规模带来的工程问题：数据要清洗、去重、配比；一张卡放不下模型和优化器状态，要切分到成百上千张卡；要用低精度换速度；还要在几个月的训练里扛住 loss 尖峰和硬件故障。',
            blocks: [
                { type: 'html', html: `<table class="lm-tbl"><tr><th>环节</th><th>本页（教学版）</th><th>生产版</th></tr>
                    <tr><td>数据</td><td>程序生成的 21 万字，176 种句子</td><td>万亿 token 级：网页抓取 → 语言识别 → 质量过滤（启发式规则 + 分类器）→ 精确与近似（MinHash）去重 → 去除隐私与有害内容；按来源配比（网页、代码、书籍、论文、多语言），训练后期常提高高质量数据的比例</td></tr>
                    <tr><td>分词</td><td>字符级，76 个 token</td><td>字节级 BPE，词表数万到十几万（第 06 篇）</td></tr>
                    <tr><td>数值精度</td><td>fp32（JS）</td><td>bf16 混合精度：矩阵乘在 bf16 下做，主权重和 Adam 状态保持 fp32；更新的硬件上还有 FP8 训练</td></tr>
                    <tr><td>并行</td><td>单线程</td><td>数据并行（DDP）、分片数据并行（ZeRO / FSDP）、张量并行（Megatron：切开每个矩阵乘）、流水线并行（按层切到不同卡）、序列并行，组合成多维并行</td></tr>
                    <tr><td>显存</td><td>28 928 × 16 字节 ≈ 0.46 MB</td><td>模型状态按 ZeRO 切分（下图）；激活值用 activation checkpointing（反向时重算，用算力换显存）</td></tr>
                    <tr><td>稳定性</td><td>梯度裁剪</td><td>监控 loss 尖峰：从尖峰前的 checkpoint 重启并跳过那段数据（PaLM 的做法）；z-loss、QK-norm 等结构 / 损失上的稳定化</td></tr>
                    <tr><td>超参数</td><td>手动挑一个学习率</td><td>先在小模型上扫描，用 scaling law 外推；μP 让小模型上调好的学习率能直接迁移到大模型</td></tr>
                    <tr><td>效率</td><td>约 1 GFLOP/s（B = 2；控制台里有实测值）</td><td>MFU（模型 FLOPs 利用率 = 6ND / 时间 / 硬件峰值）：PaLM 报告 46.2%；FlashAttention、kernel 融合、torch.compile、通信与计算重叠</td></tr></table>` },
                {
                    type: 'previews', items: [{
                        id: 'zero', label: '每张 GPU 上的模型状态显存（ZeRO 论文的例子）', sub: 'Ψ = 75 亿参数、64 张 GPU、混合精度 Adam：每个参数 2 字节 bf16 权重 + 2 字节梯度 + 12 字节 fp32 主权重 / m / v', large: true, size: [720, 170],
                        render: (c) => {
                            const v = [0, 1, 2, 3].map((s) => P05.zeroGB(7.5e9, 64, s, 12));
                            LM.draw.bars(c, v, ['数据并行（每卡一整份）', 'ZeRO-1：切分优化器状态', 'ZeRO-2：再切分梯度', 'ZeRO-3 / FSDP：连权重也切分'], { horizontal: true, digits: 1, valueSuffix: ' GB', colors: [LM.C.red, LM.C.orange, LM.C.yellow, LM.C.green], labelPx: 12 });
                        },
                    }],
                },
                {
                    type: 'code', title: 'C++ · zeroMemoryGB / 生产版训练循环的骨架（分布式部分只写原型）', src: `
                    // 每张 GPU 上的模型状态显存（GB）。混合精度 Adam：bf16 权重 2 B + bf16 梯度 2 B + fp32 主权重、m、v 共 K = 12 B（ZeRO 论文的记法）
                    double zeroMemoryGB(double psi, int nGpus, int stage, double K = 12) {
                        double weights = 2 * psi, grads = 2 * psi, optStates = K * psi;
                        if (stage >= 1) optStates /= nGpus;     // ZeRO-1：切分优化器状态
                        if (stage >= 2) grads /= nGpus;         // ZeRO-2：再切分梯度
                        if (stage >= 3) weights /= nGpus;       // ZeRO-3：连权重也切分（PyTorch 里对应 FSDP）
                        return (weights + grads + optStates) / 1e9;
                    }

                    // ── 生产版训练循环的骨架。通信、低精度 kernel 由 NCCL、cuBLAS 一类的库提供，这里只写原型（实现略） ──
                    // 本进程的编号 / 进程总数（每张 GPU 一个进程）
                    struct DistInfo { int rank = 0, worldSize = 1; };

                    // 建立进程组：启动器为每张 GPU 拉起一个进程，各自拿到自己的 rank（实现略）
                    DistInfo initProcessGroup();
                    // 把 rank 0 的初始权重复制给所有进程，大家从同一份权重出发（实现略）
                    void broadcastParams(std::vector<Var>& params);
                    // 多卡梯度平均：所有进程的 p->grad 求和再除以进程数（NCCL 等通信库提供，实现略）
                    void allReduceMean(std::vector<Var>& params);
                    // ZeRO-2：梯度求平均之后，每个进程只留下自己负责的那 1/worldSize（实现略）
                    void reduceScatterMean(std::vector<Var>& params, const DistInfo& dist);
                    // ZeRO-3 / FSDP：权重平时只存一片，用到某一层之前把切片收集齐，用完就释放（实现略）
                    void allGatherParams(std::vector<Var>& params, const DistInfo& dist);
                    // bf16 混合精度前向：矩阵乘在 bf16 下做，主权重和 Adam 状态保持 fp32（实现略）
                    Var gptForwardBF16(GPT& m, const std::vector<int>& ids, int B, int T);
                    // 保存权重 + 优化器状态：遇到 loss 尖峰或硬件故障时从这里重启（实现略）
                    void saveCheckpoint(const GPT& m, const AdamW& opt, int step);

                    // 数据并行（DDP）：每个进程有一整份模型、各自取不同的随机窗口；反向后把梯度求平均，再做同样的更新 → 各进程的权重始终一致。
                    // 换成 ZeRO-2 / FSDP：allReduceMean 换成 reduceScatterMean，adamwStep 只更新自己负责的那一片，再 allGatherParams 拼回完整权重。
                    // 真实系统还会做 kernel 融合（FlashAttention、融合的 AdamW 等）与图编译，这里略
                    void trainDistributed(const std::vector<int>& trainData, float maxLR, int B, int T, int warmup, int total, int gradAccum) {
                        DistInfo dist = initProcessGroup();
                        GPTConfig cfg;
                        GPT m = makeGPT(cfg, 1);
                        broadcastParams(m.params);
                        AdamW opt = makeAdamW(m.params, m.decay, maxLR, 0.1f, 0.9f, 0.95f, 1e-8f);
                        Rng r = makeRng(1000 + dist.rank);                // 每个进程用不同的种子 → 取到不同的随机窗口
                        for (int it = 0; it < total; it++) {
                            float lr = cosineLR(it, maxLR, maxLR / 10, warmup, total);
                            zeroGrad(m.params);
                            for (int micro = 0; micro < gradAccum; micro++) {   // 梯度累积：几个 micro-batch 拼成一个大 batch（梯度在 p->grad 里累加）
                                Batch b = getBatch(trainData, B, T, r);
                                Var loss = crossEntropy(gptForwardBF16(m, b.x, B, T), b.y);
                                backward(scale(loss, 1.f / gradAccum));
                            }
                            allReduceMean(m.params);                      // 真实实现按层分桶、边反向边通信，把通信藏在计算后面
                            clipGradNorm(m.params, 1.0f);
                            adamwStep(opt, lr);
                            if (dist.rank == 0 && (it + 1) % 1000 == 0) saveCheckpoint(m, opt, it + 1);
                        }
                    }`,
                },
                { type: 'note', html: '下一篇（09）从这里训练好的基座模型出发：它会「接着说」，但不会回答问题——<code>&lt;user&gt;熊猫喜欢吃什么？&lt;bot&gt;</code> 这几个特殊 token 它在预训练里一次都没见过。SFT、奖励模型、RLHF、DPO 把它变成一个助手。' },
                { type: 'warn', html: 'ZeRO 的数字只算了模型状态（权重、梯度、优化器状态），不含激活值、临时缓冲和显存碎片；激活值随 batch × 序列长度 × 层数增长，长序列训练时常常比模型状态还大。' },
            ],
        },
    ],

    graph: {
        title: '预训练循环 · 一个训练步的数据流',
        intro: '每个节点是一个训练步里的一次真实计算（在 GPU 上就是一个或几个 kernel；这里在 CPU 上用 JS 跑）。pin 是张量名 · 形状。<code>Params</code> 与 <code>OptState</code> 是 <b>HISTORY</b> 资源：每一步读取上一步更新后的权重和 Adam 的 m、v（虚线回边）。评估节点按 Step 07 的时间表每 40–50 步才跑一次（先给权重拍一个快照），并且分摊到之后的 39 个训练步里做（每步一小块），避免某一帧突然卡顿。',
        types: {
            ids: { color: '#56b6c2', label: 'ids · token 序列 int64' },
            act: { color: '#6ab0e0', label: 'act · 激活 fp32' },
            scalar: { color: '#e8c25a', label: 'scalar · 标量 / 超参数' },
            param: { color: '#5ad07a', label: 'param · 参数与优化器状态' },
            grad: { color: '#e06c75', label: 'grad · 梯度' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用的设置（悬停代码里的函数名可查看）', open: false,
            blocks: LM.CPP.blocks(['gpt']).concat([{
                label: '━ 本页公用 ━', code: `
                // 本页的随机种子（与页面 JS 的 P05.SEED 相同）。C++ 的 Rng 与 JS 的 LM.rng 逐位相同，
                // 所以同一个种子得到与页面完全一样的语料、验证窗口和采样结果
                struct Seeds {
                    uint32_t corpus = 2024;    // 训练语料（2 万句）
                    uint32_t val = 7;          // 验证语料（2000 句）
                    uint32_t valWin = 5;       // 从验证语料里取 32 个固定窗口
                    uint32_t sample = 4242;    // 每次评估时的采样快照（第 08 步）
                    uint32_t decoding = 9001;  // 第 09 步的 5 次续写：种子 9001 … 9005
                    uint32_t sweep = 777;      // 第 09 步的温度扫描：种子 777 … 788
                };
                const Seeds SEED = {};

                // 固定验证窗口的个数（每个窗口长 T = 32）：每次评估都用同一批 32 × 32 个预测
                const int VAL_B = 32;`,
            }]),
        },
        nodes: [
            {
                id: 'corpus', title: 'makeCorpus · encode', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'corpus', desc: '20 000 句 → 214 195 个 id',
                dispatch: { kind: 'CPU', name: 'makeCorpus', count: '214 195 + 32×33 token' },
                outputs: [{ name: 'Data', format: '[214195] int64', type: 'ids' }, { name: 'ValData', format: '[32, 32] int64', type: 'ids' }],
                detail: { preview: 'corpusText' },
            },
            {
                id: 'hyper', title: 'HyperParams', tag: 'CB', category: 'user', desc: 'η<sub>max</sub> · B · warmup · total',
                outputs: [{ name: 'Hyper', format: 'scalars', type: 'scalar' }],
                detail: { desc: '训练超参数：峰值学习率、batch、warmup 步数、总步数；另有固定的 β₁ = 0.9、β₂ = 0.95、λ = 0.1、裁剪阈值 1.0。改任何一个都会从头训练。', preview: null },
            },
            {
                id: 'lr', title: 'cosineLR', tag: 'CPU', lifetime: 'history', step: 'sched', desc: 'warmup + cosine',
                dispatch: { kind: 'CPU', name: 'cosineLR', count: '1 个标量' },
                inputs: [{ name: 'Hyper', format: 'η_max, warmup, total', type: 'scalar' }],
                outputs: [{ name: 'LR', format: 'scalar', type: 'scalar' }],
                detail: { preview: 'lrCurve' },
            },
            {
                id: 'batch', title: 'getBatch', tag: 'CPU', lifetime: 'history', step: 'batch', desc: 'B 个随机窗口，y = x 右移一位',
                dispatch: { kind: 'CPU', name: 'getBatch', count: (S) => `B = ${S.B} 个窗口 × 33 个 id` },
                inputs: [{ name: 'Data', format: '[214195] int64', type: 'ids' }, { name: 'Hyper', format: 'B', type: 'scalar' }],
                outputs: [{ name: 'X', format: '[B, T] int64', type: 'ids' }, { name: 'Y', format: '[B, T] int64', type: 'ids' }],
                detail: { preview: 'batchGrid' },
            },
            {
                id: 'embed', title: 'Embed', tag: 'CS', lifetime: 'history', step: 'loss', desc: 'wte[X] + wpe[0..T)',
                dispatch: { kind: 'CS', name: 'gather + add', count: (S) => `${S.B * 32} 行 × d = 32` },
                inputs: [{ name: 'X', format: '[B, T] int64', type: 'ids' }, { name: 'Params', format: '28 928 fp32', type: 'param', history: true }],
                outputs: [{ name: 'X0', format: '[B·T, d] fp32', type: 'act' }],
                detail: { preview: 'batchGrid' },
            },
            {
                id: 'blocks', title: 'Block × L', tag: 'CS', lifetime: 'history', step: 'loss', desc: 'x += Attn(LN(x)); x += MLP(LN(x))',
                dispatch: { kind: 'CS', name: 'Attn + MLP', count: (S) => `2 层 · ${S.B * 32} token` },
                inputs: [{ name: 'X0', format: '[B·T, d] fp32', type: 'act' }, { name: 'Params', format: '28 928 fp32', type: 'param', history: true }],
                outputs: [{ name: 'XL', format: '[B·T, d] fp32', type: 'act' }],
                detail: { desc: '两个 Pre-LN Transformer Block（第 07 篇逐层拆解过）：LN → QKV GEMM → 因果注意力 → O GEMM → 残差 → LN → FC → GELU → FC → 残差。训练时还要为反向保存每一层的激活。', preview: 'lossHeat' },
            },
            {
                id: 'head', title: 'ln_f · LM head', tag: 'CS', lifetime: 'history', step: 'loss', desc: 'LN(x) · W_teᵀ（权重共享）',
                dispatch: { kind: 'CS', name: 'layernorm + GEMM', count: (S) => `${S.B * 32} × 76 个 logits` },
                inputs: [{ name: 'XL', format: '[B·T, d] fp32', type: 'act' }, { name: 'Params', format: '28 928 fp32', type: 'param', history: true }],
                outputs: [{ name: 'Logits', format: '[B·T, V] fp32', type: 'act' }],
                detail: { preview: 'lossTopk' },
            },
            {
                id: 'ce', title: 'crossEntropy', tag: 'CS', lifetime: 'history', step: 'loss', desc: 'mean(−log softmax(z)[y])',
                dispatch: { kind: 'CS', name: 'fused softmax + NLL', count: (S) => `${S.B * 32} 行` },
                inputs: [{ name: 'Logits', format: '[B·T, V] fp32', type: 'act' }, { name: 'Y', format: '[B, T] int64', type: 'ids' }],
                outputs: [{ name: 'Loss', format: 'scalar', type: 'scalar' }],
                detail: { preview: 'lossHeat', formula: 1 },
            },
            {
                id: 'bwd', title: 'backward', tag: 'CS', lifetime: 'history', step: 'grad', desc: '逆序执行 tape，链式法则',
                dispatch: { kind: 'CS', name: 'autograd', count: '约 2× 前向的计算量（4ND）' },
                inputs: [{ name: 'Loss', format: 'scalar', type: 'scalar' }],
                outputs: [{ name: 'Grads', format: '36 个张量 fp32', type: 'grad' }],
                detail: { preview: 'gradBars' },
            },
            {
                id: 'clip', title: 'clipGradNorm', tag: 'CS', lifetime: 'history', step: 'grad', desc: 'g ← g · min(1, 1/‖g‖)',
                dispatch: { kind: 'CS', name: 'global L2 norm + scale', count: '28 928 个数' },
                inputs: [{ name: 'Grads', format: '36 个张量 fp32', type: 'grad' }],
                outputs: [{ name: 'GradsClipped', format: '36 个张量 fp32', type: 'grad' }],
                detail: { preview: 'gradNorm', formula: 1 },
            },
            {
                id: 'adamw', title: 'adamwStep', tag: 'CS', category: 'output', lifetime: 'history', step: 'adamw', desc: 'θ −= η(m̂/√v̂ + λθ)',
                dispatch: { kind: 'CS', name: 'fused AdamW', count: '28 928 个参数 × (m, v)' },
                inputs: [{ name: 'GradsClipped', format: '36 个张量 fp32', type: 'grad' }, { name: 'LR', format: 'scalar', type: 'scalar' }, { name: 'Params', format: '28 928 fp32', type: 'param', history: true }, { name: 'OptState', format: 'm, v fp32', type: 'param', history: true }],
                outputs: [{ name: 'Params', format: '28 928 fp32', type: 'param' }, { name: 'OptState', format: 'm, v fp32', type: 'param' }],
                detail: { preview: 'updBars', formula: 2 },
            },
            {
                id: 'eval', title: 'evaluate', tag: 'CPU', category: 'output', lifetime: 'history', step: 'curve', desc: 'val loss · 事实探针 · 采样',
                dispatch: { kind: 'CPU', name: 'estimateLoss · runProbes · generate', count: '每 40–50 步 · 摊到 39 步' },
                inputs: [{ name: 'Params', format: '28 928 fp32', type: 'param' }, { name: 'ValData', format: '[32, 32] int64', type: 'ids' }],
                outputs: [{ name: 'Metrics', format: 'val loss · 24 个探针 · 样本', type: 'scalar' }],
                detail: { preview: 'lossCurve' },
            },
        ],
    },

    references: [
        { title: 'Language Models are Unsupervised Multitask Learners（GPT-2）', author: 'A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever', venue: 'OpenAI', year: 2019, note: '本系列迷你 GPT 的结构与初始化（残差投影按 1/√(2L) 缩小）' },
        { title: 'Language Models are Few-Shot Learners（GPT-3）', author: 'T. Brown et al.', venue: 'NeurIPS', year: 2020, note: 'Adam β₂ = 0.95、warmup + 余弦衰减、梯度裁剪 1.0；所有模型训练 3000 亿 token' },
        { title: 'Training Compute-Optimal Large Language Models（Chinchilla）', author: 'J. Hoffmann et al.', venue: 'NeurIPS', year: 2022, note: 'L(N, D) 拟合、约 20 token/参数' },
        { title: 'Chinchilla Scaling: A replication attempt', author: 'T. Besiroglu, E. Erdil, M. Barnett, J. You', venue: 'arXiv', year: 2024, note: '指出方法 3 公布的拟合值与方法 1、2 不一致' },
        { title: 'Scaling Laws for Neural Language Models', author: 'J. Kaplan et al.', venue: 'arXiv', year: 2020, note: 'loss 随 N、D、C 的幂律；C ≈ 6ND' },
        { title: 'Adam: A Method for Stochastic Optimization', author: 'D. P. Kingma, J. Ba', venue: 'ICLR', year: 2015, note: '一阶 / 二阶矩估计与偏差修正' },
        { title: 'Decoupled Weight Decay Regularization（AdamW）', author: 'I. Loshchilov, F. Hutter', venue: 'ICLR', year: 2019, note: '权重衰减与 L2 正则在 Adam 下不等价' },
        { title: 'SGDR: Stochastic Gradient Descent with Warm Restarts', author: 'I. Loshchilov, F. Hutter', venue: 'ICLR', year: 2017, note: '余弦学习率调度' },
        { title: 'On the difficulty of training recurrent neural networks', author: 'R. Pascanu, T. Mikolov, Y. Bengio', venue: 'ICML', year: 2013, note: '按范数裁剪梯度' },
        { title: 'The Curious Case of Neural Text Degeneration', author: 'A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi', venue: 'ICLR', year: 2020, note: 'nucleus（top-p）采样；贪心 / beam 解码的重复问题' },
        { title: 'Hierarchical Neural Story Generation', author: 'A. Fan, M. Lewis, Y. Dauphin', venue: 'ACL', year: 2018, note: 'top-k 采样' },
        { title: 'Prediction and Entropy of Printed English', author: 'C. E. Shannon', venue: 'Bell System Technical Journal', year: 1951, note: '用 n-gram 与人类预测估计英文的熵——第 07 步参照线的思路' },
        { title: 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models', author: 'S. Rajbhandari, J. Rasley, O. Ruwase, Y. He', venue: 'SC', year: 2020, note: '优化器状态 / 梯度 / 参数分片；第 11 步的显存例子' },
        { title: 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism', author: 'M. Shoeybi et al.', venue: 'arXiv', year: 2019, note: '张量并行' },
        { title: 'Mixed Precision Training', author: 'P. Micikevicius et al.', venue: 'ICLR', year: 2018, note: '低精度计算 + fp32 主权重' },
        { title: 'PaLM: Scaling Language Modeling with Pathways', author: 'A. Chowdhery et al.', venue: 'JMLR', year: 2023, note: 'MFU 的定义；loss 尖峰时回滚并跳过数据；z-loss' },
        { title: 'Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer（μP）', author: 'G. Yang, E. J. Hu et al.', venue: 'NeurIPS', year: 2021, note: '小模型上的超参数迁移到大模型' },
        { title: 'nanoGPT', author: 'A. Karpathy', venue: 'GitHub', url: 'https://github.com/karpathy/nanoGPT', note: '本系列训练循环的参照实现（cosineLR、getBatch、权重衰减分组对应它的 get_lr、get_batch、configure_optimizers）' },
    ],
});

