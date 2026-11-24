// 03_tokenizer_embedding.lab.js — 系列第 03 篇：文本变向量（Tokenizer、Embedding 与位置编码）
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js llm_weights.js 03_tokenizer_embedding.lab.js -o 03_tokenizer_embedding.html
// 页面里的每张预览都由下面的 JS 参考实现实时算出：字节级 BPE 训练 / 编码 / 解码、基座模型的 wte / wpe、
// 实时训练的 word2vec（skip-gram + 负采样）、正弦位置编码、RoPE、注意力的置换等变性。

const P03 = (() => {
    const FONT = LM.FONT;
    const MAXM = 400;            // 合并次数上限（纯中文语料约 412 次后每个预切分块都只剩 1 个 token）
    const SNAP = 20;             // 训练时每 20 次合并存一份各块的切分状态（拖动 k 时从最近的快照接着合并）
    const NSENT = 2000, NHELD = 300;
    const MIXES = [0, 0.1, 0.3, 0.6, 0.9];
    const W2V = { d: 16, lr: 0.05, perFrame: 500, total: 120000, wins: [1, 2, 3, 4, 5, 6], nSent: 3000, table: 100000 };
    const SENTS = [
        '小猫喜欢吃鱼，所以小猫每天都吃鱼。',
        '熊猫和老虎是好朋友。',
        '大象喜欢吃苹果。',
        'The panda likes bamboo. 熊猫喜欢吃竹子。',
        'naïve café，熊猫🐼',
    ];

    // ── 英文镜像世界：同样 8 种动物、同样的事实（BPE 语料里按「英文比例」混入）
    const EN = [
        { name: 'cat', food: 'fish', home: 'house', color: 'white' },
        { name: 'dog', food: 'bones', home: 'yard', color: 'yellow' },
        { name: 'rabbit', food: 'carrots', home: 'meadow', color: 'gray' },
        { name: 'bird', food: 'bugs', home: 'tree', color: 'blue' },
        { name: 'panda', food: 'bamboo', home: 'mountains', color: 'black and white' },
        { name: 'monkey', food: 'bananas', home: 'forest', color: 'brown' },
        { name: 'sheep', food: 'grass', home: 'hillside', color: 'white' },
        { name: 'tiger', food: 'meat', home: 'forest', color: 'orange' },
    ];
    const ENT = [
        (a) => `The ${a.name} likes to eat ${a.food}.`,
        (a) => `The ${a.name} lives in the ${a.home}.`,
        (a) => `The ${a.name} is ${a.color}.`,
        (a, b) => `The ${a.name} and the ${b.name} are good friends.`,
        (a) => `The ${a.name} likes ${a.food}, so it eats ${a.food} every day.`,
    ];
    function enSentence(rand) {
        const a = EN[Math.floor(rand() * EN.length)];
        let b = EN[Math.floor(rand() * EN.length)]; if (b === a) b = EN[(EN.indexOf(a) + 1) % EN.length];
        return ENT[Math.floor(rand() * ENT.length)](a, b);
    }

    // ── UTF-8（手写，与 TextEncoder 结果相同）与显示
    function utf8(cp) {
        if (cp < 0x80) return [cp];
        if (cp < 0x800) return [0xC0 | (cp >> 6), 0x80 | (cp & 63)];
        if (cp < 0x10000) return [0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63)];
        return [0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63)];
    }
    function strBytes(s) { const out = []; for (const ch of s) { const b = utf8(ch.codePointAt(0)); for (let i = 0; i < b.length; i++) out.push(b[i]); } return out; }
    const hex2 = (b) => b.toString(16).toUpperCase().padStart(2, '0');
    const cpStr = (cp) => 'U+' + cp.toString(16).toUpperCase().padStart(4, '0');
    /** 把一串字节切成「完整字符」和「游离字节」两种片段（游离字节 = 半个字，用十六进制显示） */
    function segsOf(bytes) {
        const out = []; let i = 0;
        while (i < bytes.length) {
            const b = bytes[i], n = b < 0x80 ? 1 : (b >> 5) === 6 ? 2 : (b >> 4) === 14 ? 3 : (b >> 3) === 30 ? 4 : 0;
            let ok = n > 0 && i + n <= bytes.length;
            for (let j = 1; ok && j < n; j++) if ((bytes[i + j] >> 6) !== 2) ok = false;
            if (ok && n === 1 && (b < 0x20 || b === 0x7F)) ok = false;
            if (ok) {
                let cp = n === 1 ? b : b & (0xFF >> (n + 1));
                for (let j = 1; j < n; j++) cp = (cp << 6) | (bytes[i + j] & 63);
                const ch = String.fromCodePoint(cp);
                out.push({ t: ch === ' ' ? '␣' : ch });
                i += n;
            } else {
                const last = out[out.length - 1];
                if (last && last.hex) last.t += ' ' + hex2(b); else out.push({ t: hex2(b), hex: true });
                i++;
            }
        }
        return out;
    }
    /** token 类别：byte = 单个字节，part = 含半个字，char = 恰好一个字符，multi = 多个完整字符 */
    function kindOf(bytes, segs) {
        segs = segs || segsOf(bytes);
        if (bytes.length === 1) return 'byte';
        if (segs.some((s) => s.hex)) return 'part';
        return [...segs.map((s) => s.t).join('')].length === 1 ? 'char' : 'multi';
    }
    const KCOL = { byte: '#8a8f98', part: '#e06c75', char: '#6ab0e0', multi: '#5ad07a' };
    const KNAME = { byte: '单字节', part: '含半个字', char: '单个字符', multi: '多个字符' };
    const segText = (segs) => segs.map((s) => (s.hex ? '‹' + s.t + '›' : s.t)).join('');

    // ── GPT-2 预切分正则（与 Python 的 regex 库写法相同）
    const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    const pretok = (s) => s.match(PAT) || [];
    /** 预切分 + 去重计数（Map 保持首次出现的顺序 = Python collections.Counter） */
    function chunkTable(text) {
        const m = new Map();
        for (const w of pretok(text)) m.set(w, (m.get(w) || 0) + 1);
        const words = [], counts = [], ids = [];
        for (const [w, c] of m) { words.push(w); counts.push(c); ids.push(Int32Array.from(strBytes(w))); }
        return { words, counts: Int32Array.from(counts), ids };
    }
    /** 从左到右、不重叠地把 (a, b) 换成 id；没有出现时原样返回同一个数组 */
    function mergePair(c, a, b, id) {
        let n = 0;
        for (let j = 0; j + 1 < c.length; j++) if (c[j] === a && c[j + 1] === b) { n++; j++; }
        if (!n) return c;
        const out = new Int32Array(c.length - n); let o = 0;
        for (let j = 0; j < c.length; j++) { if (j + 1 < c.length && c[j] === a && c[j + 1] === b) { out[o++] = id; j++; } else out[o++] = c[j]; }
        return out;
    }
    const totalTok = (cs, ns) => { let s = 0; for (let i = 0; i < cs.length; i++) s += cs[i].length * ns[i]; return s; };
    const nChars = (s) => { let n = 0; for (const _ of s) n++; return n; };
    /**
     * 字节级 BPE 训练（与 Step 02 的 C++ 代码 trainBPE 逐条一致）。held = 留出文本：每学到一条规则就同步应用到它们上，
     * 于是得到「前 k 条规则」下的留出 token 数（按规则顺序依次合并 = encode 的结果，见 Step 04）。
     */
    function trainBPE(text, held, K) {
        const T = chunkTable(text);
        let chunks = T.ids;
        const counts = T.counts;
        const H = held.map((h) => { const t = chunkTable(h); return { chunks: t.ids, counts: t.counts }; });
        const vocab = []; for (let i = 0; i < 256; i++) vocab.push([i]);
        const merges = [], ntok = [totalTok(chunks, counts)], hTok = H.map((h) => [totalTok(h.chunks, h.counts)]);
        const snaps = [chunks];
        const cnt = new Float64Array(1 << 20), touched = new Int32Array(1 << 16);
        for (let k = 0; k < K; k++) {
            let nt = 0;
            for (let i = 0; i < chunks.length; i++) {
                const c = chunks[i], n = counts[i];
                for (let j = 0; j + 1 < c.length; j++) { const key = (c[j] << 10) | c[j + 1]; if (cnt[key] === 0) touched[nt++] = key; cnt[key] += n; }
            }
            let best = -1, bc = 0;
            for (let t = 0; t < nt; t++) { const key = touched[t]; if (cnt[key] > bc) { bc = cnt[key]; best = key; } }   // 严格大于：平手取最先出现的
            for (let t = 0; t < nt; t++) cnt[touched[t]] = 0;
            if (best < 0) break;
            const a = best >> 10, b = best & 1023, id = 256 + k;
            chunks = chunks.map((c) => mergePair(c, a, b, id));
            H.forEach((h) => { h.chunks = h.chunks.map((c) => mergePair(c, a, b, id)); });
            const bytes = vocab[a].concat(vocab[b]);
            vocab.push(bytes);
            merges.push({ a, b, id, count: bc, kind: kindOf(bytes) });
            ntok.push(totalTok(chunks, counts));
            H.forEach((h, q) => hTok[q].push(totalTok(h.chunks, h.counts)));
            if ((k + 1) % SNAP === 0) snaps.push(chunks);
        }
        const rank = new Map(merges.map((m, r) => [(m.a << 10) | m.b, r]));
        return { merges, vocab, ntok, hTok, snaps, rank, words: T.words, counts, nChars: nChars(text), heldChars: held.map(nChars), heldSent: held.map((h) => (h.match(/[。.]/g) || []).length) };
    }
    /** 训练语料各块在「前 k 条规则」下的切分 */
    function stateAt(R, k) {
        const s = Math.min(Math.floor(k / SNAP), R.snaps.length - 1);
        let ch = R.snaps[s];
        for (let r = s * SNAP; r < k && r < R.merges.length; r++) { const m = R.merges[r]; ch = ch.map((c) => mergePair(c, m.a, m.b, m.id)); }
        return ch;
    }
    /** 相邻对计数，按次数降序（稳定排序：平手保持首次出现的顺序） */
    function pairStats(chunks, counts) {
        const m = new Map();
        for (let i = 0; i < chunks.length; i++) { const c = chunks[i]; for (let j = 0; j + 1 < c.length; j++) { const key = (c[j] << 10) | c[j + 1]; m.set(key, (m.get(key) || 0) + counts[i]); } }
        return Array.from(m, ([key, n]) => ({ a: key >> 10, b: key & 1023, n })).sort((x, y) => y.n - x.n);
    }
    /** 编码一个预切分块：反复合并「排名最靠前」的相邻对（minbpe 写法）；trace 记录每一步 */
    function encodeChunk(bytes, R, k, trace) {
        let ids = Int32Array.from(bytes);
        while (ids.length >= 2) {
            let best = Infinity;
            for (let j = 0; j + 1 < ids.length; j++) { const r = R.rank.get((ids[j] << 10) | ids[j + 1]); if (r !== undefined && r < k && r < best) best = r; }
            if (best === Infinity) break;
            const m = R.merges[best];
            ids = mergePair(ids, m.a, m.b, m.id);
            if (trace) trace.push({ r: best, ids: Array.from(ids) });
        }
        return Array.from(ids);
    }
    function encode(text, R, k) {
        const pieces = [], ids = [];
        for (const w of pretok(text)) { const b = strBytes(w), trace = [{ r: -1, ids: b }]; const c = encodeChunk(b, R, k, trace); pieces.push({ w, ids: c, trace }); ids.push(...c); }
        return { ids, pieces };
    }
    const td = typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8') : null;
    function decode(ids, R) { const bytes = []; ids.forEach((i) => { const v = R.vocab[i]; for (let j = 0; j < v.length; j++) bytes.push(v[j]); }); return td.decode(Uint8Array.from(bytes)); }
    /** BPE 训练语料：第 i 句以「英文比例」的概率换成英文镜像句（同一组随机数，比例变大时只会多换几句） */
    function corpusText(src, mix) { let s = ''; for (let i = 0; i < src.zh.length; i++) s += src.u[i] < mix ? src.en[i] + ' ' : src.zh[i]; return s; }
    /** 按英文比例缓存训练结果（训练只依赖语料；拖动合并次数 k 只截取前 k 条规则） */
    function getBPE(S, mix) {
        let R = S._bpeCache.get(mix);
        if (!R) { const text = corpusText(S.bpeSrc, mix); R = trainBPE(text, S.bpeSrc.held, MAXM); R.text = text; S._bpeCache.set(mix, R); }
        return R;
    }

    // ── word2vec：47 个词的词典（正向最大匹配分词）
    const LEX = (() => {
        const A = LM.world.animals;
        const groups = [
            ['animal', A.map((a) => a.name)],
            ['food', A.map((a) => a.food)],
            ['home', [...new Set(A.map((a) => a.home))]],
            ['color', [...new Set(A.map((a) => a.color))]],
            ['func', ['喜欢', '吃', '住在', '是', '的', '最爱', '它', '所以', '每天', '都', '住着', '今天', '在', '了', '很多', '和', '好朋友']],
        ];
        const words = [], cat = [];
        groups.forEach(([c, ws]) => ws.forEach((w) => { words.push(w); cat.push(c); }));
        const content = words.map((_, i) => i).filter((i) => cat[i] !== 'func');
        return { words, cat, content, id: new Map(words.map((w, i) => [w, i])), V: words.length, maxLen: Math.max(...words.map((w) => w.length)) };
    })();
    const CATCOL = { animal: '#6ab0e0', food: '#5ad07a', home: '#e0a060', color: '#c678dd', func: '#6a7078' };
    const CATNAME = { animal: '动物', food: '食物', home: '住处', color: '颜色', func: '功能词' };
    function segmentWords(s) {
        const out = []; let i = 0;
        while (i < s.length) {
            let hit = 0;
            for (let L = Math.min(LEX.maxLen, s.length - i); L >= 1; L--) { const w = s.substr(i, L); if (LEX.id.has(w)) { out.push(LEX.id.get(w)); hit = L; break; } }
            i += hit || 1;   // 标点等词典外的字符：跳过
        }
        return out;
    }
    /** 所有 (中心词, 上下文词) 对，打包成 句号 << 8 | 中心位置 << 4 | 上下文位置，然后打乱 */
    function buildPairs(sents, win, rand) {
        const out = [];
        sents.forEach((s, si) => { for (let i = 0; i < s.length; i++) for (let j = Math.max(0, i - win); j <= Math.min(s.length - 1, i + win); j++) if (j !== i) out.push((si << 8) | (i << 4) | j); });
        return LM.shuffle(Int32Array.from(out), rand);
    }
    /** 负采样表：词 w 占 ∝ f(w)^0.75 个格子（word2vec.c 的 unigram table） */
    function unigramTable(sents, V, size) {
        const f = new Float64Array(V); sents.forEach((s) => s.forEach((w) => f[w]++));
        const p = Array.from(f, (x) => Math.pow(x, 0.75)), Z = p.reduce((a, b) => a + b, 0), out = [];
        p.forEach((pi, w) => { const n = Math.round(size * pi / Z); for (let k = 0; k < n; k++) out.push(w); });
        return { table: Int32Array.from(out), freq: f };
    }
    function w2vInit(seed, d) {
        const rand = LM.rng(seed), V = LEX.V, Win = new Float32Array(V * d);
        for (let i = 0; i < Win.length; i++) Win[i] = (rand() - 0.5) / d;    // 均匀分布 ±0.5/d
        return { Win, Wout: new Float32Array(V * d), done: 0, rand, loss: [], sep: [], acc: null, last: null };
    }
    const sig = (x) => 1 / (1 + Math.exp(-x));
    /** 逐对 SGD：先在当前参数上算出全部梯度，再一起更新（= PyTorch batch = 1 的 SGD） */
    function w2vTrain(M, sents, pairs, tab, n, K, d) {
        const Win = M.Win, Wout = M.Wout, g = new Float32Array(d), ids = new Int32Array(K + 1), sc = new Float32Array(K + 1);
        let L = 0;
        for (let p = 0; p < n; p++) {
            const key = pairs[(M.done + p) % pairs.length], s = sents[key >> 8], c = s[(key >> 4) & 15], o = s[key & 15];
            const lr = W2V.lr * Math.max(1e-4, 1 - (M.done + p) / W2V.total);
            let m = 0; ids[m++] = o;
            for (let k = 0; k < K; k++) { const ng = tab[Math.floor(M.rand() * tab.length)]; if (ng !== o) ids[m++] = ng; }
            const vc = c * d;
            for (let q = 0; q < m; q++) { const uo = ids[q] * d; let z = 0; for (let j = 0; j < d; j++) z += Win[vc + j] * Wout[uo + j]; sc[q] = z; }
            g.fill(0);
            for (let q = 0; q < m; q++) {
                const pr = sig(sc[q]), gq = pr - (q === 0 ? 1 : 0), uo = ids[q] * d;
                L += q === 0 ? -Math.log(Math.max(pr, 1e-12)) : -Math.log(Math.max(1 - pr, 1e-12));
                for (let j = 0; j < d; j++) g[j] += gq * Wout[uo + j];
            }
            for (let q = 0; q < m; q++) { const gq = sig(sc[q]) - (q === 0 ? 1 : 0), uo = ids[q] * d; for (let j = 0; j < d; j++) Wout[uo + j] -= lr * gq * Win[vc + j]; }
            for (let j = 0; j < d; j++) Win[vc + j] -= lr * g[j];
            if (p === n - 1) M.last = { key, negs: Array.from(ids.subarray(1, m)), lr };
        }
        M.done += n;
        return L / Math.max(1, n);
    }
    /** 行归一化（余弦相似度 = 归一化后的点积） */
    function normRows(W, V, d) {
        const out = new Float32Array(V * d);
        for (let i = 0; i < V; i++) { let s = 0; for (let j = 0; j < d; j++) s += W[i * d + j] ** 2; const r = 1 / (Math.sqrt(s) || 1); for (let j = 0; j < d; j++) out[i * d + j] = W[i * d + j] * r; }
        return out;
    }
    const dotRow = (A, i, B, j, d) => { let s = 0; for (let k = 0; k < d; k++) s += A[i * d + k] * B[j * d + k]; return s; };
    /** 簇分离度：实词的「同类平均余弦 − 异类平均余弦」 */
    function separation(Wn, d) {
        const C = LEX.content; let sw = 0, nw = 0, sa = 0, na = 0;
        for (const a of C) for (const b of C) { if (a >= b) continue; const c = dotRow(Wn, a, Wn, b, d); if (LEX.cat[a] === LEX.cat[b]) { sw += c; nw++; } else { sa += c; na++; } }
        return sw / nw - sa / na;
    }
    /** 类比题：a : rel(a) = b : ?（三种关系，答案相同的题跳过） */
    const QUESTIONS = (() => {
        const A = LM.world.animals, out = [];
        ['food', 'home', 'color'].forEach((rel) => A.forEach((a) => A.forEach((b) => { if (a !== b && a[rel] !== b[rel]) out.push({ rel, a: LEX.id.get(a.name), b: LEX.id.get(a[rel]), c: LEX.id.get(b.name), ans: LEX.id.get(b[rel]) }); })));
        return out;
    })();
    function analogy(Wn, d, a, b, c) {
        let best = -1, bs = -Infinity;
        for (let x = 0; x < LEX.V; x++) {
            if (x === a || x === b || x === c) continue;
            let s = 0; for (let j = 0; j < d; j++) s += Wn[x * d + j] * (Wn[b * d + j] - Wn[a * d + j] + Wn[c * d + j]);
            if (s > bs) { bs = s; best = x; }
        }
        return best;
    }
    function analogyAcc(Wn, d) {
        const acc = { food: [0, 0], home: [0, 0], color: [0, 0] };
        QUESTIONS.forEach((q) => { const r = analogy(Wn, d, q.a, q.b, q.c); acc[q.rel][1]++; if (r === q.ans) acc[q.rel][0]++; });
        const all = Object.values(acc).reduce((s, v) => [s[0] + v[0], s[1] + v[1]], [0, 0]);
        return { food: acc.food[0] / acc.food[1], home: acc.home[0] / acc.home[1], color: acc.color[0] / acc.color[1], all: all[0] / all[1], n: all[1], ok: all[0] };
    }
    /** 2D 投影的「对齐」：在 8 种翻转 / 交换里选和上一帧最接近的，避免 PCA 轴来回翻转造成的跳动 */
    function align2(prev, cur, n) {
        if (!prev || prev.length !== cur.length) return cur;
        let best = null, be = Infinity;
        for (let sw = 0; sw < 2; sw++) for (let sx = -1; sx <= 1; sx += 2) for (let sy = -1; sy <= 1; sy += 2) {
            let e = 0;
            for (let i = 0; i < n; i++) { const x = (sw ? cur[2 * i + 1] : cur[2 * i]) * sx, y = (sw ? cur[2 * i] : cur[2 * i + 1]) * sy; e += (x - prev[2 * i]) ** 2 + (y - prev[2 * i + 1]) ** 2; }
            if (e < be) { be = e; best = [sw, sx, sy]; }
        }
        const out = new Float32Array(cur.length), [sw, sx, sy] = best;
        for (let i = 0; i < n; i++) { out[2 * i] = (sw ? cur[2 * i + 1] : cur[2 * i]) * sx; out[2 * i + 1] = (sw ? cur[2 * i] : cur[2 * i + 1]) * sy; }
        return out;
    }

    // ── 位置编码
    /** 正弦位置编码：PE[pos, 2i] = sin(pos·ω_i)，PE[pos, 2i+1] = cos(pos·ω_i)，ω_i = base^(−2i/d) */
    function sinPEAt(pos, d, base, out, off) { for (let i = 0; i < d / 2; i++) { const a = pos * Math.pow(base, -2 * i / d); out[off + 2 * i] = Math.sin(a); out[off + 2 * i + 1] = Math.cos(a); } return out; }
    function sinPE(T, d, base) { const pe = new Float64Array(T * d); for (let p = 0; p < T; p++) sinPEAt(p, d, base, pe, p * d); return pe; }
    /** RoPE（相邻维度成对）：第 i 对按 pos·θ_i 旋转 */
    function rope(x, pos, base) {
        const d = x.length, out = new Float64Array(d);
        for (let i = 0; i < d / 2; i++) { const a = pos * Math.pow(base, -2 * i / d), c = Math.cos(a), s = Math.sin(a); out[2 * i] = x[2 * i] * c - x[2 * i + 1] * s; out[2 * i + 1] = x[2 * i] * s + x[2 * i + 1] * c; }
        return out;
    }
    const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
    function cosMatrix(X, R, d) {
        const out = new Float32Array(R * R), n = new Float64Array(R);
        for (let i = 0; i < R; i++) { let s = 0; for (let j = 0; j < d; j++) s += X[i * d + j] ** 2; n[i] = Math.sqrt(s) || 1; }
        for (let i = 0; i < R; i++) for (let j = 0; j < R; j++) { let s = 0; for (let k = 0; k < d; k++) s += X[i * d + k] * X[j * d + k]; out[i * R + j] = s / (n[i] * n[j]); }
        return out;
    }

    // ── 基座模型：字符在玩具世界里的「角色」（只用来给 PCA 图上色）
    const ROLE = (() => {
        const r = new Map(), set = (s, k) => { for (const ch of s) if (!r.has(ch)) r.set(ch, k); };
        set('小熊猴老', 'name1'); set('猫狗兔鸟羊虎', 'name2'); set('鱼骨头萝卜虫竹香蕉青肉', 'food');
        set('屋里院地树上山森林坡', 'home'); set('白色黄灰蓝黑棕橙', 'color');
        return r;
    })();
    const ROLECOL = { name1: '#56b6c2', name2: '#6ab0e0', food: '#5ad07a', home: '#e0a060', color: '#c678dd', other: '#7a808a', unseen: '#e06c75' };
    const ROLENAME = { name1: '动物名首字', name2: '动物名次字', food: '食物', home: '住处', color: '颜色', other: '其他', unseen: '未见' };

    // ═════════ 绘图 ═════════
    function tokObj(R, id, extra) {
        const bytes = R ? R.vocab[id] : [id], segs = segsOf(bytes), kind = kindOf(bytes, segs);
        return Object.assign({ segs, id, kind, part: kind === 'part' || (kind === 'byte' && id >= 0x80), color: id < 256 ? '#2a3038' : LM.tokColor(id) }, extra);
    }
    function tokWidth(ctx, t, px) {
        let w = 0;
        t.segs.forEach((s) => { ctx.font = s.hex ? FONT(11) : FONT(px); w += ctx.measureText(s.t).width + (s.hex ? 3 : 0); });
        let sw = 0; if (t.sub != null) { ctx.font = FONT(11); sw = ctx.measureText(String(t.sub)).width; }
        return Math.max(w + 8, sw + 4, px + 2);
    }
    function tokBox(ctx, x, y, t, px, bw) {
        const bh = px + 8;
        ctx.globalAlpha = t.dim ? 0.35 : 1;
        ctx.fillStyle = t.color || '#2a3038'; ctx.fillRect(x, y, bw, bh);
        if (t.part) { ctx.setLineDash([3, 2]); ctx.strokeStyle = LM.C.red; ctx.lineWidth = 1.2; ctx.strokeRect(x + 0.6, y + 0.6, bw - 1.2, bh - 1.2); ctx.setLineDash([]); }
        if (t.outline) { ctx.strokeStyle = t.outline; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, bw + 2, bh + 2); }
        let tw = 0; t.segs.forEach((s) => { ctx.font = s.hex ? FONT(11) : FONT(px); tw += ctx.measureText(s.t).width + (s.hex ? 3 : 0); });
        let cx = x + (bw - tw) / 2; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        t.segs.forEach((s) => {
            ctx.font = s.hex ? FONT(11) : FONT(px); ctx.fillStyle = s.hex ? '#f2d58a' : (t.textColor || '#f0f0f0');
            ctx.fillText(s.t, cx + (s.hex ? 1.5 : 0), y + bh / 2 + 0.5); cx += ctx.measureText(s.t).width + (s.hex ? 3 : 0);
        });
        if (t.sub != null) { ctx.font = FONT(11); ctx.fillStyle = t.subColor || LM.C.dim; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(t.sub), x + bw / 2, y + bh + 2); }
        ctx.globalAlpha = 1;
    }
    /** token 条：items = token 对象 | { br } | { label, color } | { gap }；自动换行。返回 boxes（供 tokHit） */
    function drawToks(canvas, items, o) {
        o = o || {};
        const { ctx, w } = PL.draw.ctx(canvas, {});
        const px = o.px || 14, gap = o.gap ?? 3, hasSub = items.some((t) => t && t.sub != null);
        const rowH = px + 8 + (hasSub ? 15 : 0) + (o.rowGap ?? 6), x0 = 6;
        let x = x0, y = o.title ? 22 : 6;
        if (o.title) { ctx.fillStyle = LM.C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, 6, 5); }
        const boxes = [];
        items.forEach((t, i) => {
            if (t.br) { x = x0; y += rowH; return; }
            if (t.gap) { x += t.gap; return; }
            if (t.label != null) { ctx.font = FONT(11); ctx.fillStyle = t.color || LM.C.gray; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; const lw = ctx.measureText(t.label).width; if (x + lw > w - 4 && x > x0) { x = x0; y += rowH; } ctx.fillText(t.label, x, y + (px + 8) / 2); x += lw + 6; return; }
            const bw = tokWidth(ctx, t, px);
            if (x + bw > w - 4 && x > x0) { x = o.indent ?? x0; y += rowH; }
            tokBox(ctx, x, y, t, px, bw);
            boxes.push({ i, x, y, w: bw, h: px + 8 + (hasSub ? 13 : 0), t });
            x += bw + gap;
        });
        canvas._p03Toks = boxes;
        return { ctx, w, y: y + rowH, boxes };
    }
    function tokHit(canvas, info) { const b = canvas._p03Toks; if (!b || !info) return null; return b.find((r) => info.dx >= r.x - 1 && info.dx <= r.x + r.w + 1 && info.dy >= r.y && info.dy <= r.y + r.h) || null; }
    /** 在一块矩形里画热图（多面板预览用）。o: { range:[lo,hi], map, rowLabels, title, hiRow, labelEvery } */
    function heatAt(ctx, x0, y0, cw, ch, data, R, C, o) {
        o = o || {};
        const map = PL.cmap[o.map || 'diverge'], [lo, hi] = o.range || [-1, 1], rgb = [0, 0, 0];
        for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
            const v = data[r * C + c], t = (v - lo) / (hi - lo);
            map(t < 0 ? 0 : t > 1 ? 1 : t, rgb); ctx.fillStyle = LM.rgbStr(rgb); ctx.fillRect(x0 + c * cw, y0 + r * ch, Math.ceil(cw), Math.ceil(ch));
        }
        if (o.title) { ctx.fillStyle = LM.C.text; ctx.font = FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(o.title, x0, y0 - 3); }
        if (o.rowLabels) {
            ctx.font = FONT(o.labelPx || 11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            const every = o.labelEvery || Math.max(1, Math.ceil(12 / ch));
            for (let r = 0; r < R; r += every) { ctx.fillStyle = r === o.hiRow ? LM.C.yellow : LM.C.gray; ctx.fillText(String(o.rowLabels[r]), x0 - 4, y0 + (r + 0.5) * ch); }
        }
        if (o.hiRow != null && o.hiRow >= 0) { ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 1.5; ctx.strokeRect(x0 - 1, y0 + o.hiRow * ch - 0.5, C * cw + 2, ch + 1); }
        ctx.strokeStyle = '#2a2f36'; ctx.lineWidth = 1; ctx.strokeRect(x0 - 0.5, y0 - 0.5, C * cw + 1, R * ch + 1);
    }
    function colorBar(ctx, x, y, h, range, map) {
        const rgb = [0, 0, 0], m = PL.cmap[map || 'diverge'];
        for (let i = 0; i < h; i++) { m(1 - i / (h - 1), rgb); ctx.fillStyle = LM.rgbStr(rgb); ctx.fillRect(x, y + i, 7, 1.5); }
        ctx.font = FONT(11); ctx.fillStyle = LM.C.dim; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(String(range[1]), x + 10, y - 2); ctx.textBaseline = 'bottom'; ctx.fillText(String(range[0]), x + 10, y + h + 2);
    }
    const legend = (ctx, items, x, y, gapX) => {
        ctx.font = FONT(11); ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        items.forEach(([label, col]) => { ctx.fillStyle = col; ctx.fillRect(x, y - 4, 9, 9); ctx.fillStyle = LM.C.gray; ctx.fillText(label, x + 13, y); x += 13 + ctx.measureText(label).width + (gapX || 12); });
    };
    /**
     * 散点标签的贪心布局：每个标签在「离开所在簇中心」的方向附近、由近到远找一个不与已放标签和其他点重叠的位置，
     * 离点较远时画一条引线。items: [{ x, y（设计坐标）, r, text, color, g（簇）, maxR }]，按数组顺序放（重要的放前面）。
     */
    function placeLabels(ctx, items, W, H, X0, Y0) {
        X0 = X0 || 0; Y0 = Y0 || 0;
        const boxes = items.map((it) => ({ x0: it.x - it.r - 1, y0: it.y - it.r - 1, x1: it.x + it.r + 1, y1: it.y + it.r + 1, pt: it }));
        const ov = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
        const cent = {};
        items.forEach((it) => { const g = cent[it.g] || (cent[it.g] = { x: 0, y: 0, n: 0 }); g.x += it.x; g.y += it.y; g.n++; });
        Object.values(cent).forEach((g) => { g.x /= g.n; g.y /= g.n; });
        const ANG = [0, 180, -35, 35, -145, 145, -70, 70, -110, 110, -90, 90].map((a) => a * Math.PI / 180), out = [];
        ctx.font = FONT(11);
        items.forEach((it) => {
            const tw = ctx.measureText(it.text).width, th = 12, g = cent[it.g];
            let bx = it.x - g.x, by = it.y - g.y;
            if (Math.hypot(bx, by) < 2) { bx = g.x - W / 2; by = g.y - H / 2; }
            if (Math.hypot(bx, by) < 1) bx = 1;
            const base = Math.atan2(by, bx);
            let best = null;
            for (const rad of [it.r + 3, 11, 19, 29, 41, 55, 72]) {
                if (rad > (it.maxR ?? 72)) break;
                for (const da of ANG) {
                    const a = base + da, ax = it.x + rad * Math.cos(a), ay = it.y + rad * Math.sin(a), left = Math.cos(a) >= -0.3;
                    const b = { x0: left ? ax : ax - tw, x1: left ? ax + tw : ax, y0: ay - th / 2, y1: ay + th / 2 };
                    if (b.x0 < X0 + 2 || b.x1 > W - 2 || b.y0 < Y0 + 2 || b.y1 > H - 2) continue;
                    if (boxes.some((o) => o.pt !== it && ov(o, b))) continue;
                    best = { b, ax, ay, rad, left }; break;
                }
                if (best) break;
            }
            if (best) { boxes.push(best.b); out.push(Object.assign({ it }, best)); }
        });
        out.forEach(({ it, b, ax, ay, rad, left }) => {
            if (rad > it.r + 6) { ctx.strokeStyle = 'rgba(150,160,170,0.45)'; ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(it.x, it.y); ctx.lineTo(left ? b.x0 - 1 : b.x1 + 1, ay); ctx.stroke(); }
            ctx.font = FONT(11); ctx.fillStyle = it.color; ctx.textAlign = left ? 'left' : 'right'; ctx.textBaseline = 'middle'; ctx.fillText(it.text, ax, ay);
        });
        return out.length;
    }

    return {
        MAXM, NSENT, NHELD, MIXES, W2V, SENTS, EN, enSentence, utf8, strBytes, hex2, cpStr, segsOf, kindOf, KCOL, KNAME, segText,
        pretok, chunkTable, mergePair, trainBPE, stateAt, pairStats, encodeChunk, encode, decode, corpusText, getBPE,
        LEX, CATCOL, CATNAME, segmentWords, buildPairs, unigramTable, w2vInit, w2vTrain, normRows, dotRow, separation, QUESTIONS, analogy, analogyAcc, align2,
        sinPEAt, sinPE, rope, dot, cosMatrix, ROLE, ROLECOL, ROLENAME,
        tokObj, tokWidth, tokBox, drawToks, tokHit, heatAt, colorBar, legend, placeLabels, FONT,
    };
})();

// ═════════ 预览里共用的小工具 ═════════
const P03UI = (() => {
    const FONT = P03.FONT, LEX = P03.LEX;
    const wordTok = (i, extra) => Object.assign({ segs: [{ t: LEX.words[i] }], color: '#1a2029', textColor: P03.CATCOL[LEX.cat[i]] }, extra);
    const EX = [['小猫', '鱼', '小狗', '骨头'], ['熊猫', '竹子', '猴子', '香蕉'], ['小兔', '草地', '小鸟', '树上'], ['小狗', '院子', '熊猫', '山里'], ['小狗', '黄色', '小鸟', '蓝色'], ['老虎', '橙色', '猴子', '棕色']]
        .map(([a, b, c, ans]) => ({ a: LEX.id.get(a), b: LEX.id.get(b), c: LEX.id.get(c), ans: LEX.id.get(ans) }));
    /** scatter 预览里离鼠标最近的点（设计坐标，maxPx 以内） */
    function nearest(tf, pts, info, maxPx) {
        if (!tf || !info) return -1;
        let best = -1, bd = (maxPx || 14) ** 2;
        pts.forEach((p, i) => { const dx = tf.X(p.x) - info.dx, dy = tf.Y(p.y) - info.dy, d2 = dx * dx + dy * dy; if (d2 < bd) { bd = d2; best = i; } });
        return best;
    }
    const tokStr = (R, id) => P03.segText(P03.segsOf(R.vocab[id]));
    const bytesHex = (bytes) => bytes.map(P03.hex2).join(' ');
    const fmtN = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
    const hitBox = (boxes, info) => (boxes || []).find((b) => info && info.dx >= b.x - 1 && info.dx <= b.x + b.w + 1 && info.dy >= b.y && info.dy <= b.y + b.h) || null;
    /** 无位置 / 有位置两种情况下：原句与打乱句的注意力输出并排，相同 token 连线，右侧是该 token 两次输出之差的范数 */
    function drawPerm(c, P, res, title) {
        const { ctx, w } = PL.draw.ctx(c, {});
        const T = P.T, d = P.d, cw = 3.2, ch = 22, y0 = 44, xa = 26, xb = xa + d * cw + 58;
        let mx = 0; [res.yo, res.yp].forEach((a) => a.forEach((v) => { mx = Math.max(mx, Math.abs(v)); }));
        ctx.font = FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(title, 6, 5);
        P03.heatAt(ctx, xa, y0, cw, ch, res.yo, T, d, { range: [-mx, mx], rowLabels: P.toks, title: `原句「${P.src}」` });
        P03.heatAt(ctx, xb, y0, cw, ch, res.yp, T, d, { range: [-mx, mx], rowLabels: P.permToks, title: `打乱「${P.permToks.join('')}」` });
        let worst = 0;
        for (let i = 0; i < T; i++) {
            const t = P.perm.indexOf(i), same = res.diff[i] < 1e-4 * (res.rms + 1e-9);
            ctx.strokeStyle = same ? 'rgba(90,208,122,0.8)' : 'rgba(224,108,117,0.8)'; ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.moveTo(xa + d * cw + 3, y0 + (i + 0.5) * ch); ctx.lineTo(xb - 16, y0 + (t + 0.5) * ch); ctx.stroke();
            ctx.font = FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillStyle = same ? LM.C.green : LM.C.red;
            ctx.fillText(`‖Δ‖ ${res.diff[i] < 1e-12 ? '0' : res.diff[i].toExponential(1)}`, xb + d * cw + 6, y0 + (t + 0.5) * ch);
            worst = Math.max(worst, res.diff[i]);
        }
        ctx.font = FONT(11); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`每行 = 一个 token 的输出 [1 × ${d}] · 输出的典型范数 ≈ ${res.rms.toFixed(2)}`, 6, y0 + T * ch + 8);
        void w; void worst;
    }
    return { wordTok, EX, nearest, tokStr, bytesHex, fmtN, hitBox, drawPerm };
})();

// ═════════ 代码块（C++17 + 标准库；Tensor、Var、GPT、Vocab、utf8Chars… 来自本系列自带的 pages/cpp/minml_*.h） ═════════
const P03CPP = {
    // graph.common · 本页公用 ①：UTF-8 解码与 GPT-2 预切分（与页面 JS 的 pretok 相同，正则改成手写的分支）
    utf8: String.raw`
        // UTF-8 字节串 → Unicode 码位序列（输入假定是合法的 UTF-8）。
        // 第 i 个字符占 s 的字节 [start[i], start[i+1])：start 末尾多存一个 s.size()
        struct Utf8Text { std::vector<int> cp, start; };

        Utf8Text utf8Decode(const std::string& s) {
            Utf8Text t;
            for (size_t i = 0; i < s.size();) {
                unsigned char b = (unsigned char)s[i];
                int n = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : 4;     // 首字节开头 1 的个数 = 这个字符的字节数
                int cp = n == 1 ? b : b & (0xFF >> (n + 1));                           // 去掉首字节的前缀位
                for (int j = 1; j < n; j++) cp = (cp << 6) | ((unsigned char)s[i + j] & 0x3F);   // 每个后续字节贡献 6 位
                t.cp.push_back(cp);
                t.start.push_back((int)i);
                i += n;
            }
            t.start.push_back((int)s.size());
            return t;
        }

        // 字符串的 UTF-8 字节，每个是 0..255 的整数 = 字节级 token id
        // （std::string 里存的本来就是字节；源文件按 UTF-8 保存时 "猫" 就是 E7 8C AB 三个字节，MSVC 要加 /utf-8）
        std::vector<int> bytesOf(const std::string& s) {
            std::vector<int> ids;
            for (unsigned char b : s) ids.push_back(b);
            return ids;
        }

        // 字符类别 = GPT-2 正则里的 \p{L} 字母、\p{N} 数字、\s 空白，其余都算「其他」（标点、符号、emoji）。
        // 简化版：按 Unicode 区段粗分，够本页的中英文语料用；tiktoken 等真实实现查完整的 Unicode 属性表
        enum class CharKind { Letter, Number, Space, Other };

        CharKind charKind(int c) {
            if (c == ' ' || (c >= 0x09 && c <= 0x0D) || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
                c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000 || c == 0xFEFF) return CharKind::Space;
            if ((c >= '0' && c <= '9') || (c >= 0xFF10 && c <= 0xFF19)) return CharKind::Number;
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                (c >= 0xC0 && c <= 0x24F && c != 0xD7 && c != 0xF7) ||             // 带附加符号的拉丁字母：é ï ñ …
                (c >= 0x370 && c <= 0x4FF) ||                                      // 希腊、西里尔字母
                (c >= 0x3041 && c <= 0x30FF) ||                                    // 日文假名
                (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x20000 && c <= 0x2FFFF) ||   // 汉字
                (c >= 0xAC00 && c <= 0xD7A3) ||                                    // 韩文
                (c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A)) return CharKind::Letter;   // 全角拉丁字母
            return CharKind::Other;                                                // 「，」「。」「.」「🐼」…
        }

        // GPT-2 的预切分正则（tiktoken 的 "gpt2" 编码用的也是它）：
        //     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        // 下面按分支的先后顺序手写：从第 i 个字符开始找一个匹配，返回匹配结束的位置（不含）
        int matchPretoken(const std::vector<int>& c, int i) {
            int n = (int)c.size();
            if (c[i] == '\'') {                                                    // ① 英文缩写 's 't 're 've 'm 'll 'd
                const std::string SUFFIX[] = {"s", "t", "re", "ve", "m", "ll", "d"};
                for (const std::string& suf : SUFFIX) {
                    int L = (int)suf.size(), k = 0;
                    while (k < L && i + 1 + k < n && c[i + 1 + k] == suf[k]) k++;
                    if (k == L) return i + 1 + L;
                }
            }
            int j = (c[i] == ' ' && i + 1 < n) ? i + 1 : i;                        // ② ␣?字母+ | ␣?数字+ | ␣?其他+：
            CharKind k = charKind(c[j]);                                           //    前面的一个空格归到后面这一块
            if (k != CharKind::Space) {
                while (j < n && charKind(c[j]) == k) j++;
                return j;
            }
            int e = i;                                                             // ③ \s+(?!\S)：一段空白后面紧跟非空白时，
            while (e < n && charKind(c[e]) == CharKind::Space) e++;                //    最后一个空白留给下一块当前导空格
            if (e < n && e - i >= 2) return e - 1;
            return e;                                                              // ④ \s+
        }

        // 预切分：单词 / 数字 / 标点 / 空白各成一块（中文没有空格：一段连续的汉字就是一块）
        std::vector<std::string> pretokenize(const std::string& text) {
            Utf8Text t = utf8Decode(text);
            std::vector<std::string> out;
            for (int i = 0; i < (int)t.cp.size();) {
                int j = matchPretoken(t.cp, i);
                out.push_back(text.substr(t.start[i], t.start[j] - t.start[i]));
                i = j;
            }
            return out;
        }`,
    // graph.common · 本页公用 ②：BPE 的基础函数（与页面 JS 的 chunkTable / 相邻对计数 / mergePair 一一对应）
    bpeCore: `
        // 预切分 + 去重计数：每个不同的块、它当前的 token 序列、它在语料里出现的次数（按首次出现的顺序排列）
        struct ChunkTable {
            std::vector<std::string> words;                  // 不同的预切分块
            std::vector<std::vector<int>> chunks;            // 每块当前的 token 序列（一开始 = 它的 UTF-8 字节）
            std::vector<int> counts;                         // 每块在语料里出现的次数
        };

        ChunkTable chunkTable(const std::string& text) {
            ChunkTable T;
            std::map<std::string, int> slot;                 // 块 → 它在 words 里的下标
            for (const std::string& w : pretokenize(text)) {
                auto it = slot.find(w);
                if (it != slot.end()) { T.counts[it->second]++; continue; }
                slot[w] = (int)T.words.size();
                T.words.push_back(w);
                T.chunks.push_back(bytesOf(w));
                T.counts.push_back(1);
            }
            return T;
        }

        // 一种相邻 token 对 (a, b) 和它的出现次数 n
        struct PairCount { int a, b, n; };

        // 相邻对计数：每块按它在语料里出现的次数加权。结果按「首次出现」的顺序排列，所以找最大值时平手取最先出现的一对
        std::vector<PairCount> pairStats(const std::vector<std::vector<int>>& chunks, const std::vector<int>& counts) {
            std::vector<PairCount> stats;
            std::map<std::pair<int, int>, int> slot;         // (a, b) → 它在 stats 里的下标
            for (size_t i = 0; i < chunks.size(); i++) {
                const std::vector<int>& c = chunks[i];
                for (size_t j = 0; j + 1 < c.size(); j++) {
                    auto it = slot.find({c[j], c[j + 1]});
                    if (it != slot.end()) { stats[it->second].n += counts[i]; continue; }
                    slot[{c[j], c[j + 1]}] = (int)stats.size();
                    stats.push_back({c[j], c[j + 1], counts[i]});
                }
            }
            return stats;
        }

        // 从左到右、不重叠地把每个相邻的 (a, b) 换成 newId
        std::vector<int> mergePair(const std::vector<int>& ids, int a, int b, int newId) {
            std::vector<int> out;
            for (size_t i = 0; i < ids.size();) {
                if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) { out.push_back(newId); i += 2; }
                else { out.push_back(ids[i]); i += 1; }
            }
            return out;
        }`,
    // graph.common · 本页公用 ③：word2vec 的数据准备（与页面 JS 的 segmentWords / buildPairs / unigramTable 一一对应）
    w2vCore: `
        // 47 个词的小词典（空格分隔）：动物 8、食物 8、住处 7、颜色 7、功能词 17
        const char WORDS[] = "${P03.LEX.words.join(' ')}";

        struct Lexicon {
            std::vector<std::string> words;                  // id → 词
            std::map<std::string, int> id;                   // 词 → id
            int maxLen = 0;                                  // 最长的词有几个字
        };

        Lexicon makeLexicon(const std::string& spaceSeparated) {
            Lexicon lex;
            std::string w;
            for (char ch : spaceSeparated + " ") {
                if (ch != ' ') { w += ch; continue; }
                if (w.empty()) continue;
                lex.id[w] = (int)lex.words.size();
                lex.words.push_back(w);
                lex.maxLen = std::max(lex.maxLen, (int)utf8Chars(w).size());
                w.clear();
            }
            return lex;
        }

        // 正向最大匹配分词：每次取词典里最长的词；标点等词典外的字符直接跳过
        std::vector<int> segmentWords(const std::string& sentence, const Lexicon& lex) {
            std::vector<std::string> ch = utf8Chars(sentence);   // 按字符切开（一个汉字 = 3 个字节）
            std::vector<int> out;
            for (int i = 0; i < (int)ch.size();) {
                int hit = 0;
                for (int L = std::min(lex.maxLen, (int)ch.size() - i); L >= 1 && !hit; L--) {
                    std::string w;
                    for (int k = 0; k < L; k++) w += ch[i + k];
                    auto it = lex.id.find(w);
                    if (it != lex.id.end()) { out.push_back(it->second); hit = L; }
                }
                i += hit ? hit : 1;
            }
            return out;
        }

        // 一个训练样本：中心词 c、上下文词 o
        struct SkipPair { int c, o; };

        // 所有 (中心词, 上下文词) 对：左右各 window 个词，不跨句；然后用 r 打乱（Fisher–Yates）
        std::vector<SkipPair> skipgramPairs(const std::vector<std::vector<int>>& sents, int window, Rng& r) {
            std::vector<SkipPair> pairs;
            for (const std::vector<int>& s : sents)
                for (int i = 0; i < (int)s.size(); i++)
                    for (int j = std::max(0, i - window); j <= std::min((int)s.size() - 1, i + window); j++)
                        if (j != i) pairs.push_back({s[i], s[j]});
            for (int i = (int)pairs.size() - 1; i > 0; i--) std::swap(pairs[i], pairs[randint(r, i + 1)]);
            return pairs;
        }

        // 负采样表：词 w 占 round(size · f(w)^0.75 / Σ) 格，均匀抽一格 = 按 f^0.75 抽一个词（word2vec.c 的 unigram table）
        std::vector<int> unigramTable(const std::vector<std::vector<int>>& sents, int V, int size) {
            std::vector<int> f = ints(V);
            for (const std::vector<int>& s : sents) for (int w : s) f[w]++;          // 词频 f(w)
            double Z = 0;
            for (int w = 0; w < V; w++) Z += std::pow((double)f[w], 0.75);
            std::vector<int> table;
            for (int w = 0; w < V; w++) {
                int n = (int)std::round(size * std::pow((double)f[w], 0.75) / Z);
                for (int k = 0; k < n; k++) table.push_back(w);
            }
            return table;
        }`,
    // graph.common · 本页公用 ④：基座模型的字符级词表 / 权重，和几个向量小工具
    misc: `
        // 字符级词表（第 04–06 篇的迷你 GPT）：4 个特殊 token + 玩具世界的 72 个汉字 / 标点，共 76 个
        Vocab baseVocab() {
            Vocab v;
            v.itos = {"<pad>", "<eos>", "<user>", "<bot>"};
            for (const std::string& ch : utf8Chars("${LM.vocab.itos.slice(4).join('')}")) v.itos.push_back(ch);
            return v;
        }

        // 读入基座模型的 28 928 个参数（与页面里 llm_weights.js 相同的权重；读文件的实现略）
        void loadWeights(GPT& m, const std::string& path);

        // 每行除以自己的 L2 范数（全 0 的行不变）：之后两行的点积就是余弦相似度
        Tensor normRows(const Tensor& W) {
            Tensor out = W;
            for (int i = 0; i < W.rows; i++) {
                double s = 0;
                for (int j = 0; j < W.cols; j++) s += W.at(i, j) * W.at(i, j);
                float r = s > 0 ? (float)(1 / std::sqrt(s)) : 1.f;
                for (int j = 0; j < W.cols; j++) out.at(i, j) *= r;
            }
            return out;
        }

        // A 的第 i 行 · B 的第 j 行
        float dotRow(const Tensor& A, int i, const Tensor& B, int j) {
            double s = 0;
            for (int k = 0; k < A.cols; k++) s += A.at(i, k) * B.at(j, k);
            return (float)s;
        }`,
};

PipelineLab.mount({
    paramSteps: { sent: ['bytes', 'encode', 'input'], enMix: ['bytes', 'bpe', 'compress', 'encode', 'prod'], merges: ['bytes', 'bpe', 'compress', 'encode', 'prod'], tok: ['lookup'], win: ['w2v', 'sim'], neg: ['w2v', 'sim'], query: ['w2v', 'sim'], shuf: ['perm'], freqBase: ['sinpe', 'rope'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '03 · 文本变向量：Tokenizer、Embedding 与位置编码',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 20251123,
    css: LM.CSS,
    regen: { label: '重新生成语料', title: '重新抽取 BPE 与 word2vec 的训练语料（只重算 RESIDENT）' },
    state: { charSel: 1, chunkSel: 0 },

    params: [
        {
            id: 'sent', label: '示例句子', value: 0,
            options: [{ value: 0, label: '小猫喜欢吃鱼，所以…' }, { value: 1, label: '熊猫和老虎是好朋友' }, { value: 2, label: '大象喜欢吃苹果（新字）' }, { value: 3, label: '中英混排' }, { value: 4, label: 'naïve café 🐼' }],
            desc: 'Step 01、04、11 分析的句子。「大象」「苹果」不在玩具世界里：字符级词表编码不了（OOV），字节级照样可以。后两句演示 1 / 2 / 4 字节的 UTF-8 与中英混排。',
        },
        {
            id: 'enMix', label: '英文比例', value: 0.1, options: P03.MIXES.map((m) => ({ value: m, label: Math.round(m * 100) + '%' })),
            desc: 'BPE 训练语料（2000 句）里换成英文镜像句的比例。0% 时英文文本永远是 1 个字节 1 个 token；调大后英文的压缩率上升、中文的下降（Step 03、12）。',
        },
        {
            id: 'merges', label: '合并次数 k', min: 0, max: P03.MAXM, step: 1, value: 120,
            desc: 'BPE 学到的合并规则条数，词表大小 V = 256 + k。从 0 慢慢拖到 60：字节先拼成字、再拼成词；拖到 300 以上，玩具语料里的整句短语也成了一个 token（Step 02–04）。',
        },
        {
            id: 'tok', label: '查看的 token', min: 0, max: LM.vocab.size - 1, step: 1, value: LM.vocab.stoi.get('猫'), affects: 'frame',
            format: (v) => `${LM.vocab.show(v)} #${v}`,
            desc: 'Step 05 看基座模型 E 的哪一行（字符级词表，76 个 token）。也可以在 PCA 图上点一个字。试试「猫」「鱼」「白」，再试从没出现过的「⟨pad⟩」「什」。',
        },
        {
            id: 'win', label: 'word2vec 窗口', options: P03.W2V.wins, value: 2, affects: 'frame',
            desc: '上下文窗口（左右各几个词）。页面同时训练 6 个只差窗口的模型，这里选看哪一个：1–2 时同类词（动物 / 食物 / 住处 / 颜色）各聚一团；5–6 时每只动物和它自己的食物、住处、颜色靠在一起，类比题约九成答对（Step 06、07）。',
        },
        {
            id: 'neg', label: '负样本 K', min: 1, max: 15, step: 1, value: 5,
            desc: '每个真实的（中心词, 上下文词）对配几个随机负样本（按 f^0.75 抽）。改动后 6 个 word2vec 模型从头重新训练（Step 06、07）。',
        },
        {
            id: 'query', label: '查询词', min: 0, max: P03.LEX.V - 1, step: 1, value: 0, affects: 'frame', format: (v) => P03.LEX.words[v],
            desc: 'Step 07 看哪个词的最近邻；也可以直接在 Step 06 的 PCA 图上点一个词。对比窗口 2 和 6 时「小猫」的邻居。',
        },
        {
            id: 'shuf', label: '打乱方式', value: 0, options: [{ value: 0, label: '鱼喜欢吃小猫' }, { value: 1, label: '随机排列 A' }, { value: 2, label: '随机排列 B' }],
            desc: 'Step 08 怎样打乱「小猫喜欢吃鱼」：「鱼喜欢吃小猫」意思完全相反，token 却一个不差；也可以换成随机排列。',
        },
        {
            id: 'freqBase', label: '频率底数 base', value: 10000, options: [{ value: 100, label: '100' }, { value: 10000, label: '10 000' }, { value: 500000, label: '500 000' }],
            desc: '正弦位置编码和 RoPE 共用的频率 ω_i = base^(−2i/d)。原论文和 LLaMA-2 用 10 000，LLaMA-3 用 500 000；改成 100 让高维度也转起来（Step 09、10）。',
        },
    ],

    stages: [
        {
            id: 'corpus', lifetime: 'resident', run(S) {
                // BPE 语料：2000 句中文 + 对应的英文镜像句（按英文比例替换）；留出文本各 300 句
                const zh = [], en = [], u = [], zhH = [], enH = [];
                for (let i = 0; i < P03.NSENT; i++) { zh.push(LM.world.sentence(S.rand)); en.push(P03.enSentence(S.rand)); u.push(S.rand()); }
                for (let i = 0; i < P03.NHELD; i++) { zhH.push(LM.world.sentence(S.rand)); enH.push(P03.enSentence(S.rand)); }
                S.bpeSrc = { zh, en, u, held: [zhH.join(''), enH.join(' ') + ' '] };
                S._bpeCache = new Map();
                // word2vec 语料：3000 句，按词切分；6 种窗口的训练对 + 负采样表
                const ws = []; for (let i = 0; i < P03.W2V.nSent; i++) ws.push(P03.segmentWords(LM.world.sentence(S.rand)));
                S.w2vSents = ws;
                S.w2vPairs = P03.W2V.wins.map((w) => P03.buildPairs(ws, w, S.rand));
                S.w2vTab = P03.unigramTable(ws, P03.LEX.V, P03.W2V.table);
            },
        },
        {
            id: 'model', lifetime: 'resident', run(S) {
                // 第 04–06 篇的基座模型（字符级，V = 76，d = 32）：这里用它的 wte、wpe 和第 0 层注意力
                const m = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), LM_WEIGHTS.base.b64);
                const V = m.cfg.V, d = m.cfg.d, E = m.wte.data;
                const seenCh = new Set(LM.world.corpus(LM.rng(LM_WEIGHTS.base.seeds.corpus), 4000));   // 预训练语料里出现过的字
                const unseen = [], seen = [];
                for (let i = 0; i < V; i++) (seenCh.has(LM.vocab.itos[i]) ? seen : unseen).push(i);
                const norms = Array.from({ length: V }, (_, i) => Math.hypot(...E.subarray(i * d, (i + 1) * d)));
                const cos = P03.cosMatrix(E, V, d);
                let minCos = 1; unseen.forEach((a) => unseen.forEach((b) => { minCos = Math.min(minCos, cos[a * V + b]); }));
                const Xs = new Float32Array(seen.length * d);
                seen.forEach((t, r) => { for (let j = 0; j < d; j++) Xs[r * d + j] = E[t * d + j] / norms[t]; });
                const pc = LM.pca2(Xs, seen.length, d, 7);
                const init = LM.gpt.create(LM_WEIGHTS.base.cfg, 1);
                let initNorm = 0; for (let i = 0; i < V; i++) initNorm += Math.hypot(...init.wte.data.subarray(i * d, (i + 1) * d)) / V;
                S.lm = m;
                S.wte = { V, d, norms, cos, unseen, seen, minCos, pts: pc.pts, varRatio: pc.varRatio, initNorm };
                S.wpeSim = P03.cosMatrix(m.wpe.data, m.cfg.T, d);
            },
        },
        { id: 'bpe', lifetime: 'param', run(S) { S.R = P03.getBPE(S, S.enMix); } },
        {
            id: 'bpeK', lifetime: 'param', run(S) {
                const R = S.R, k = Math.min(S.merges, R.merges.length);
                const top = P03.pairStats(P03.stateAt(R, k), R.counts).slice(0, 10);
                const kinds = { part: 0, char: 0, multi: 0 }; for (let r = 0; r < k; r++) kinds[R.merges[r].kind]++;
                let ex = '', n = 0; for (const ch of R.text) { ex += ch; if ((ch === '。' || ch === '.') && ++n >= 4) break; }
                S.bpeK = { k, top, kinds, excerpt: P03.encode(ex, R, k), exText: ex };
            },
        },
        {
            id: 'enc', lifetime: 'param', run(S) {
                const R = S.R, k = S.bpeK.k, s = P03.SENTS[S.sent], e = P03.encode(s, R, k), dec = P03.decode(e.ids, R);
                const cids = [...s].map((ch) => (LM.vocab.stoi.has(ch) ? LM.vocab.stoi.get(ch) : -1));
                S.enc = Object.assign(e, { s, dec, ok: dec === s, solo: e.ids.map((i) => P03.decode([i], R)), nBytes: P03.strBytes(s).length, nChars: [...s].length, cids, oov: cids.filter((i) => i < 0).length });
                if (S.chunkSel >= e.pieces.length) S.chunkSel = 0;
            },
        },
        {
            id: 'text', lifetime: 'param', run(S) {
                S.chars = [...P03.SENTS[S.sent]].map((ch) => { const cp = ch.codePointAt(0); return { ch, cp, bytes: P03.utf8(cp), cid: LM.vocab.stoi.has(ch) ? LM.vocab.stoi.get(ch) : -1 }; });
                if (S.charSel >= S.chars.length) S.charSel = 0;
            },
        },
        {
            id: 'perm', lifetime: 'param', run(S) {
                // 基座模型第 0 层：LayerNorm → Q/K/V → 双向注意力（不加因果掩码）→ W_O。比较原句与打乱句里同一个 token 的输出
                const m = S.lm, L0 = m.layers[0], AG = LM.AG, d = m.cfg.d, H = m.cfg.H;
                const src = '小猫喜欢吃鱼', ids = LM.vocab.encode(src), T = ids.length, id0 = Array.from({ length: T }, (_, i) => i);
                let perm = [5, 2, 3, 4, 0, 1];                                   // perm[t] = 打乱后第 t 个位置放原句的第几个 token →「鱼喜欢吃小猫」
                if (S.shuf > 0) { perm = LM.shuffle(id0.slice(), LM.rng(9000 + S.shuf)); if (perm.every((v, i) => v === i)) perm.reverse(); }
                const pos = Int32Array.from(id0);
                const out = (order, withPos) => AG.noGrad(() => {
                    let x = AG.embed(Int32Array.from(order.map((i) => ids[i])), m.wte);
                    if (withPos) x = AG.add(x, AG.gather(m.wpe, pos));
                    const h = AG.layernorm(x, L0.ln1g, L0.ln1b);
                    const q = AG.linear(h, L0.wq, L0.bq), k = AG.linear(h, L0.wk, L0.bk), v = AG.linear(h, L0.wv, L0.bv);
                    return AG.linear(AG.attention(q, k, v, 1, T, H, false), L0.wo, L0.bo).data;
                });
                const res = [false, true].map((wp) => {
                    const yo = out(id0, wp), yp = out(perm, wp);
                    const diff = id0.map((i) => { const t = perm.indexOf(i); let s = 0; for (let j = 0; j < d; j++) s += (yo[i * d + j] - yp[t * d + j]) ** 2; return Math.sqrt(s); });
                    let nrm = 0; for (let i = 0; i < yo.length; i++) nrm += yo[i] ** 2;
                    return { yo, yp, diff, rms: Math.sqrt(nrm / T) };
                });
                S.perm = { src, toks: [...src], perm, permToks: perm.map((i) => src[i]), T, d, noPE: res[0], withPE: res[1] };
            },
        },
        {
            id: 'pe', lifetime: 'param', run(S) {
                const d = 32, T = 64, B = S.freqBase, D = 40, starts = [40, 100, 400], a = new Float64Array(d), b = new Float64Array(d);
                const dots = starts.map((i0) => { const out = []; for (let dd = -D; dd <= D; dd++) { P03.sinPEAt(i0, d, B, a, 0); P03.sinPEAt(i0 + dd, d, B, b, 0); out.push(P03.dot(a, b)); } return out; });
                let md = 0; dots.forEach((cv) => cv.forEach((v, i) => { md = Math.max(md, Math.abs(v - dots[0][i])); }));
                S.pe = { d, T, B, D, starts, pe: P03.sinPE(T, d, B), dots, maxDiff: md, simSin: P03.cosMatrix(P03.sinPE(32, d, B), 32, d) };
            },
        },
        {
            id: 'rope', lifetime: 'param', run(S) {
                const d = 32, B = S.freqBase, D = 40, ms = [40, 200, 1000], rr = LM.rng(4242);
                const q = Float64Array.from({ length: d }, () => LM.randn(rr)), k = Float64Array.from({ length: d }, () => LM.randn(rr));
                const pa = new Float64Array(d), pb = new Float64Array(d), qa = new Float64Array(d), kb = new Float64Array(d);
                const rope = ms.map((m) => { const out = []; for (let dd = -D; dd <= D; dd++) out.push(P03.dot(P03.rope(q, m, B), P03.rope(k, m + dd, B))); return out; });
                const abs = ms.map((m) => {
                    const out = [];
                    for (let dd = -D; dd <= D; dd++) { P03.sinPEAt(m, d, B, pa, 0); P03.sinPEAt(m + dd, d, B, pb, 0); for (let j = 0; j < d; j++) { qa[j] = q[j] + pa[j]; kb[j] = k[j] + pb[j]; } out.push(P03.dot(qa, kb)); }
                    return out;
                });
                const md = (cs) => { let v = 0; cs.forEach((cv) => cv.forEach((x, i) => { v = Math.max(v, Math.abs(x - cs[0][i])); })); return v; };
                const wave = {}; [100, 10000, 500000].forEach((bb) => { wave[bb] = Array.from({ length: d / 2 }, (_, i) => 2 * Math.PI / Math.pow(bb, -2 * i / d)); });
                let lo = Infinity, hi = -Infinity; rope.concat(abs).forEach((cv) => cv.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
                S.ropeD = { d, B, D, ms, q, k, rope, abs, maxRope: md(rope), maxAbs: md(abs), wave, yr: [lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)] };
            },
        },
        {
            id: 'inp', lifetime: 'param', run(S) {
                const m = S.lm, d = m.cfg.d, s0 = P03.SENTS[S.sent], ok = [...s0].every((ch) => LM.vocab.stoi.has(ch));
                const s = ok ? s0 : P03.SENTS[0], ids = LM.vocab.encode(s), T = ids.length;
                const E = new Float32Array(T * d), P = new Float32Array(T * d), X = new Float32Array(T * d), nE = [], nP = [], nX = [];
                for (let t = 0; t < T; t++) {
                    let a = 0, b = 0, c = 0;
                    for (let j = 0; j < d; j++) { const e = m.wte.data[ids[t] * d + j], p = m.wpe.data[t * d + j]; E[t * d + j] = e; P[t * d + j] = p; X[t * d + j] = e + p; a += e * e; b += p * p; c += (e + p) ** 2; }
                    nE.push(Math.sqrt(a)); nP.push(Math.sqrt(b)); nX.push(Math.sqrt(c));
                }
                S.inp = { s, ok, ids, T, d, E, P, X, nE, nP, nX };
            },
        },
        {
            id: 'mixSweep', lifetime: 'param', lazy: true, run(S) {
                // 同样 k 条规则，不同英文比例训练出来的 tokenizer 在中 / 英留出文本上的压缩率（训练结果按比例缓存）
                S.mixSweep = P03.MIXES.map((mix) => {
                    const R = P03.getBPE(S, mix), k = Math.min(S.merges, R.merges.length);
                    return { mix, zh: R.heldChars[0] / R.hTok[0][k], en: R.heldChars[1] / R.hTok[1][k], zhT: R.hTok[0][k] / R.heldSent[0], enT: R.hTok[1][k] / R.heldSent[1] };
                });
            },
        },
        {
            id: 'w2v', lifetime: 'history', run(S) {
                // 6 个只差窗口大小的 skip-gram 模型同时训练；每帧每个模型训练 perFrame 对（与帧率挂钩、与 S.dt 无关 → 结果可复现），到 total 对为止
                const W = P03.W2V, d = W.d;
                if (S._w2vEpoch !== S.epoch || !S.w2v) {
                    S._w2vEpoch = S.epoch;
                    S.w2v = W.wins.map((w, q) => {
                        const M = P03.w2vInit(((S.seed ^ 0x2545F491) >>> 0) + 977 * q, d), Wn = P03.normRows(M.Win, P03.LEX.V, d);
                        M.sep.push([0, P03.separation(Wn, d)]); M.acc = P03.analogyAcc(Wn, d);
                        return M;
                    });
                    S.w2vTick = 0;
                }
                const n = W.perFrame;
                S.w2vTick++;
                S.w2v.forEach((M, q) => {
                    const todo = Math.min(n, W.total - M.done);
                    if (todo <= 0) return;
                    M.loss.push([M.done + todo, P03.w2vTrain(M, S.w2vSents, S.w2vPairs[q], S.w2vTab.table, todo, S.neg, d)]);
                    if (S.w2vTick % 3 === 0 || M.done >= W.total) { const Wn = P03.normRows(M.Win, P03.LEX.V, d); M.sep.push([M.done, P03.separation(Wn, d)]); M.acc = P03.analogyAcc(Wn, d); }
                });
            },
        },
        {
            id: 'w2vView', lifetime: 'frame', run(S) {
                // 所选窗口的模型：归一化后的 PCA（与上一帧对齐，防止轴翻转）、实词相似度矩阵、查询词的最近邻、类比示例
                const q = P03.W2V.wins.indexOf(S.win), M = S.w2v[q], d = P03.W2V.d, V = P03.LEX.V;
                const key = `${S.epoch}|${q}|${M.done}|${S.query}`;
                if (S._w2vKey === key) return;
                const vk = `${S.epoch}|${q}`, Wn = P03.normRows(M.Win, V, d), pc = LM.pca2(Wn, V, d, 7);
                const same = S.w2vView && S.w2vView.vk === vk;
                const pts = P03.align2(same ? S.w2vView.pts : null, pc.pts, V);
                // 显示范围：每条边「向外扩立刻跟上，向内收慢慢收缩」，动画里坐标轴不会来回「呼吸」
                let bx0 = Infinity, bx1 = -Infinity, by0 = Infinity, by1 = -Infinity;
                for (let i = 0; i < V; i++) { bx0 = Math.min(bx0, pts[2 * i]); bx1 = Math.max(bx1, pts[2 * i]); by0 = Math.min(by0, pts[2 * i + 1]); by1 = Math.max(by1, pts[2 * i + 1]); }
                const pad = 0.2 * Math.max(bx1 - bx0, by1 - by0, 1e-6);
                let box = [bx0 - pad, bx1 + pad, by0 - pad, by1 + pad];
                if (same) { const b = S.w2vView.box, lo = (c, p) => (c < p ? c : p + 0.06 * (c - p)), hi = (c, p) => (c > p ? c : p + 0.06 * (c - p)); box = [lo(box[0], b[0]), hi(box[1], b[1]), lo(box[2], b[2]), hi(box[3], b[3])]; }
                const C = P03.LEX.content, n = C.length, sim = new Float32Array(n * n);
                for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) sim[a * n + b] = P03.dotRow(Wn, C[a], Wn, C[b], d);
                const nn = []; for (let j = 0; j < V; j++) if (j !== S.query) nn.push([j, P03.dotRow(Wn, S.query, Wn, j, d)]);
                nn.sort((x, y) => y[1] - x[1]);
                S.w2vView = { vk, q, M, pts, box, sim, nn: nn.slice(0, 8), ex: P03UI.EX.map((e) => Object.assign({ got: P03.analogy(Wn, d, e.a, e.b, e.c) }, e)), varRatio: pc.varRatio };
                S._w2vKey = key;
            },
        },
        { id: 'ropeAnim', lifetime: 'frame', run(S) { S.ropeM = Math.floor(S.t * 3) % 64; } },
    ],

    intro: {
        title: '从字符串到向量：模型真正「看见」的是什么',
        flow: ['文本', 'UTF-8 字节', 'BPE 合并 → token', 'token id', '查表 E[id]', '+ 位置信息', 'X [T × d] → 第 04 篇'],
        html: LM.navHtml('03') + `
            <p>神经网络只会做矩阵运算，不认识字。要把「小猫喜欢吃鱼」交给 Transformer，得先回答三个问题：<strong>切成什么单位</strong>（tokenizer），<strong>每个单位变成什么向量</strong>（embedding），<strong>怎样告诉模型谁在前、谁在后</strong>（位置编码）。这一篇沿着数据流把三件事走一遍，最后得到第 04 篇 Transformer 的输入 X。</p>
            <p>页面里的数据都是实时算出来的：在玩具世界语料上<strong>真实训练</strong>一个 GPT-2 式的字节级 BPE；在同一个世界的语料上<strong>实时训练</strong> word2vec（带 <span style="color:#a45fa5">LIVE</span> 的图，几秒收敛，可在顶栏暂停 / 调速）；位置编码、RoPE 和注意力的置换对称性都是真实计算。注意：第 04–06 篇的迷你 GPT 用的是<strong>字符级</strong>词表（76 个 token）——玩具世界一共只有 72 个不同的字符，字符级就够了；这里训练 BPE 是为了看清真实大模型的 tokenizer 在做什么。代码块是同一算法的 C++（C++17 + 标准库，不依赖第三方库）：BPE、word2vec 是手写的循环，模型部分用本系列自带的小型 C++ 库 <code>pages/cpp/minml_*.h</code>（在数据流图的 Common 里，悬停函数名可看定义）。</p>`,
    },

    steps: [
        // ─────────────────────────────── 01 字节 ───────────────────────────────
        {
            id: 'bytes', title: '字符与字节：Unicode 码位与 UTF-8', tag: 'TEXT',
            desc: '计算机里的文本是一串 <strong>Unicode 码位</strong>（code point，如「猫」= U+732B），存储和传输时用 <strong>UTF-8</strong> 编成 1–4 个字节：ASCII 1 字节，带重音的拉丁字母 2 字节，汉字和全角标点 3 字节，emoji 4 字节。tokenizer 的第一个决定是<strong>从哪一级开始切</strong>。按字符切，词表要覆盖所有可能出现的字（Unicode 收录了约 15 万个字符），没见过的字只能变成 &lt;unk&gt;（OOV，out-of-vocabulary）；按<strong>字节</strong>切，词表只有 256 个，任何字符串都能无损表示，代价是序列变长——一个汉字要 3 个 token。GPT-2 以来最常见的做法是两者结合：从字节出发，再用 BPE 把常见的字节组合合并成大 token（下一步）；LLaMA-1/2 用的 SentencePiece 则从字符出发，只把罕见字符退回字节（Step 12）。',
            formula: [
                `
                 U+0000  – U+007F   → 0xxxxxxx                              // 1 字节：ASCII
                 U+0080  – U+07FF   → 110xxxxx 10xxxxxx                     // 2 字节：é、ï …
                 U+0800  – U+FFFF   → 1110xxxx 10xxxxxx 10xxxxxx            // 3 字节：汉字、全角标点
                 U+10000 – U+10FFFF → 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx   // 4 字节：emoji`,
                '首字节开头 1 的个数 = 这个字符的字节数；后续字节一律以 10 开头   // 从任意位置都能找到字符边界',
            ],
            code: {
                title: 'C++ · utf8Encode / showBytes', src: `
                // 码位 → 1~4 个字节（与 "猫" 存进 std::string 后的 UTF-8 字节相同）
                std::vector<int> utf8Encode(int cp) {
                    if (cp < 0x80) return {cp};
                    if (cp < 0x800) return {0xC0 | (cp >> 6), 0x80 | (cp & 0x3F)};
                    if (cp < 0x10000) return {0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)};
                    return {0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3F), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)};
                }

                // 逐字符打印：字符、码位、UTF-8 字节（utf8Decode、bytesOf 见 Common）
                void showBytes(const std::string& s) {
                    Utf8Text t = utf8Decode(s);
                    for (size_t i = 0; i < t.cp.size(); i++) {
                        std::string ch = s.substr(t.start[i], t.start[i + 1] - t.start[i]);
                        std::vector<int> b = utf8Encode(t.cp[i]);
                        assert(b == bytesOf(ch));                    // 手算的字节 = 字符串里实际存的字节
                        std::printf("%s U+%04X", ch.c_str(), t.cp[i]);
                        for (int x : b) std::printf(" %02X", x);
                        std::printf("\\n");
                    }
                }

                void bytesDemo() {
                    showBytes("小猫 cat 🐼");
                    std::vector<int> ids = bytesOf("小猫");          // 字节级 token id 就是字节值 0..255：{229, 176, 143, 231, 140, 171}
                }`,
            },
            previews: [
                {
                    id: 'bytesView', label: '逐字符：码位 → UTF-8 字节', sub: '绿 = ASCII 字节 · 橙 = 多字节字符的首字节 · 蓝 = 后续字节（10xxxxxx）· <b>点击一个字符</b>看它的位布局', large: true, size: [720, 236],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, {});
                        const cols = [], rowH = 64, BW = 21;
                        let x = 8, y = 8, nb = 0;
                        S.chars.forEach((it, i) => {
                            const n = it.bytes.length, cw = Math.max(n * BW, 44) + 6;
                            if (x + cw > w - 6) { x = 8; y += rowH; }
                            cols.push({ i, x, y, w: cw, h: rowH - 6 });
                            if (i === S.charSel) { ctx.fillStyle = 'rgba(232,194,90,0.10)'; ctx.fillRect(x - 1, y - 3, cw - 2, rowH - 6); ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 1.5; ctx.strokeRect(x - 1, y - 3, cw - 2, rowH - 6); }
                            const cx = x + (cw - 2) / 2 - 1;
                            ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.font = P03.FONT(17); ctx.fillStyle = '#f0f0f0';
                            ctx.fillText(it.ch === ' ' ? '␣' : it.ch, cx, y);
                            ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.gray; ctx.fillText(P03.cpStr(it.cp), cx, y + 22);
                            it.bytes.forEach((b, j) => {
                                const bx = cx - n * BW / 2 + j * BW, by = y + 37, kind = b < 0x80 ? 0 : b >= 0xC0 ? 1 : 2;
                                ctx.fillStyle = ['#1d4a2c', '#5a3a18', '#1b3756'][kind]; ctx.fillRect(bx + 1, by, BW - 2, 17);
                                ctx.fillStyle = ['#a6e8b8', '#f2c48a', '#a8cdf2'][kind]; ctx.textBaseline = 'middle'; ctx.fillText(P03.hex2(b), bx + BW / 2, by + 9); ctx.textBaseline = 'top';
                            });
                            nb += n; x += cw;
                        });
                        // 剩下的空间：字节级模型实际收到的 id 序列（就是这些字节的十进制值）
                        const ids = [].concat(...S.chars.map((it) => it.bytes));
                        let yy = y + rowH + 2, xx = 8;
                        if (yy + 14 < h - 22) {
                            ctx.font = P03.FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = LM.C.text;
                            const head = '字节级 token id 序列（= 字节的十进制值）：'; ctx.fillText(head, xx, yy); xx += ctx.measureText(head).width;
                            ctx.fillStyle = LM.C.gray;
                            for (let i = 0; i < ids.length; i++) {
                                const t = String(ids[i]) + (i < ids.length - 1 ? ', ' : ''), tw = ctx.measureText(t).width;
                                if (xx + tw > w - 8) { if (yy + 29 > h - 22) { ctx.fillText('…', xx, yy); break; } xx = 8; yy += 15; }
                                ctx.fillText(t, xx, yy); xx += tw;
                            }
                        }
                        P03.legend(ctx, [['ASCII 字节 0xxxxxxx', '#3f9a5c'], ['首字节 11xxxxxx', '#c08040'], ['后续字节 10xxxxxx', '#4a80c0']], 8, h - 11);
                        ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = LM.C.text; ctx.font = P03.FONT(11);
                        ctx.fillText(`${S.chars.length} 个字符 → ${nb} 个字节`, w - 8, h - 11);
                        S._p03Cols = cols;
                    },
                    probe: (x, y, S, PL, info) => { const q = P03UI.hitBox(S._p03Cols, info); if (!q) return ''; const it = S.chars[q.i]; return `「${it.ch}」 ${P03.cpStr(it.cp)} = ${it.cp} → ${P03UI.bytesHex(it.bytes)}（${it.bytes.length} 字节）`; },
                    pick: (x, y, S, PL, info) => { const q = P03UI.hitBox(S._p03Cols, info); if (q) S.charSel = q.i; },
                },
                {
                    id: 'bitsView', label: 'UTF-8 的位布局', sub: '选中字符的码位二进制被拆开，依次填进模板里的 x', size: [360, 200],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c, {});
                        const it = S.chars[S.charSel] || S.chars[0], n = it.bytes.length, payload = [7, 11, 16, 21][n - 1], first = [7, 5, 4, 3][n - 1];
                        const bits = it.cp.toString(2).padStart(payload, '0'), groups = [bits.slice(0, first)];
                        for (let j = 0; j < n - 1; j++) groups.push(bits.slice(first + 6 * j, first + 6 * j + 6));
                        const prefix = n === 1 ? ['0'] : [['110', '1110', '11110'][n - 2]].concat(Array(n - 1).fill('10'));
                        const GC = ['#e0a060', '#6ab0e0', '#5ad07a', '#c678dd'], X0 = 92;
                        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                        ctx.font = P03.FONT(14, true); ctx.fillStyle = '#f0f0f0'; ctx.fillText(`「${it.ch === ' ' ? '␣' : it.ch}」 ${P03.cpStr(it.cp)}`, 8, 16);
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.gray; ctx.fillText(`= ${it.cp}（十进制）· ${n} 个字节 · ${payload} 个有效位`, 8, 36);
                        const label = (t, y) => { ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.dim; ctx.fillText(t, 8, y); };
                        const runs = (x, y, parts) => { ctx.font = P03.FONT(13); parts.forEach(([t, col]) => { ctx.fillStyle = col; ctx.fillText(t, x, y); x += ctx.measureText(t).width; }); return x; };
                        label('码位的位', 62); runs(X0, 62, groups.map((g, j) => [g + ' ', GC[j]]));
                        label('UTF-8 模板', 88); runs(X0, 88, prefix.map((p) => [p + 'x'.repeat(8 - p.length) + ' ', LM.C.dim]));
                        label('填入后', 114); runs(X0, 114, [].concat(...prefix.map((p, j) => [[p, '#8a8f98'], [groups[j] + ' ', GC[j]]])));
                        ctx.font = P03.FONT(13); const colW = ctx.measureText('xxxxxxxx ').width;
                        label('字节', 140); it.bytes.forEach((b, j) => { ctx.font = P03.FONT(13, true); ctx.fillStyle = GC[j]; ctx.fillText(P03.hex2(b), X0 + j * colW + colW * 0.3, 140); });
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.gray; ctx.fillText('灰色前缀告诉解码器：这是第几种字节、整个字符有几个字节', 8, 172);
                    },
                },
                {
                    id: 'oovView', label: '字符级词表 vs 字节', sub: '字符级 = 第 04–06 篇迷你 GPT 的 76 个 token（红框 = 词表里没有，只能当 &lt;unk&gt;）', size: [360, 200],
                    render: (c, S) => {
                        const items = [{ label: '字符级：', color: LM.C.text }];
                        S.chars.forEach((it) => items.push({ segs: [{ t: it.ch === ' ' ? '␣' : it.ch }], color: it.cid < 0 ? '#4a1c22' : '#1d2a38', sub: it.cid < 0 ? 'unk' : it.cid, subColor: it.cid < 0 ? LM.C.red : LM.C.dim, outline: it.cid < 0 ? LM.C.red : null }));
                        const r = P03.drawToks(c, items, { px: 13, gap: 2 });
                        const nb = S.chars.reduce((s, it) => s + it.bytes.length, 0), oov = S.chars.filter((it) => it.cid < 0).length, ctx = r.ctx;
                        ctx.font = P03.FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        const lines = [
                            [`字符级（76 个 token）：${S.chars.length} 个 token，`, LM.C.gray, oov ? `其中 ${oov} 个 OOV` : '没有 OOV', oov ? LM.C.red : LM.C.green],
                            [`字节级（256 个 token）：${nb} 个 token，`, LM.C.gray, '永远没有 OOV', LM.C.green],
                            [`字节级 BPE（k = ${S.bpeK.k}）：${S.enc.ids.length} 个 token`, LM.C.gray, '（Step 04）', LM.C.dim],
                        ];
                        lines.forEach(([a, ca, b, cb], i) => { const y = r.y + 4 + i * 17; ctx.fillStyle = ca; ctx.fillText(a, 6, y); ctx.fillStyle = cb; ctx.fillText(b, 6 + ctx.measureText(a).width, y); });
                    },
                },
            ],
            notes: '切换 <b>示例句子</b>：「大象喜欢吃苹果」里的「大象」「苹果」在玩具世界的字符级词表（76 个）里没有，只能变成 ⟨unk⟩，模型连「这是什么字」都无从知道；字节级照样能表示。「naïve café 🐼」里有 2 字节的 ï、é 和 4 字节的 emoji。',
            warn: 'C++ 的 <code>std::string("猫").size()</code> 是 3（字节数：std::string 存的就是 UTF-8 字节），码位数要先解码：<code>utf8Decode("猫").cp.size()</code> 是 1（UTF-32 的 <code>U"猫".size()</code> 也是 1）；JavaScript 的 <code>"🐼".length</code> 是 2（UTF-16 码元）。统计长度、截断文本时一定要分清这三种「长度」——在字节中间截断会得到非法的 UTF-8。',
        },
        // ─────────────────────────────── 02 BPE 训练 ───────────────────────────────
        {
            id: 'bpe', title: 'BPE 训练：反复合并最频繁的相邻对', tag: 'BPE',
            desc: '<strong>BPE</strong>（Byte Pair Encoding，最早是 Gage 1994 年的压缩算法，Sennrich 等 2016 年把它用于神经机器翻译的分词）的训练只有一个循环：统计语料里所有<strong>相邻 token 对</strong>的出现次数 → 把最频繁的一对合并成一个新 token、加进词表 → 重复 k 次，词表大小 = 256 + k。GPT-2 先用一个正则把文本<strong>预切分</strong>成单词 / 标点 / 空白块，合并只在块内发生，所以「。」永远不会和下一句的字粘在一起。下面是在玩具世界语料（2000 句）上<strong>真实训练</strong>的结果：最先合并的是最高频汉字的 3 个字节（「。」「小」「吃」「子」…）；接着出现<strong>跨字符的半个字</strong>——「吃」后面最常跟以 E9 开头的字（鱼 E9 B1 BC、骨 E9 AA A8、香、青），于是「吃 + E9」被合成一个 token；然后拼出「喜欢」「住在」「猴子」「小猫」这样的词。k 再大，玩具语料太重复，整句短语也会变成一个 token。',
            formula: [
                'count(a, b) = Σ_{块 w} freq(w) · #{ w 中相邻的 (a, b) }',
                '(a*, b*) = argmax_{(a, b)} count(a, b)      // 平手时取最先出现的一对',
                'id_new = 256 + k      bytes(id_new) = bytes(a*) ‖ bytes(b*)      // 新 token = 两半的字节拼接',
            ],
            code: {
                title: 'C++ · trainBPE（pretokenize / chunkTable / pairStats / mergePair 见 Common）', src: `
                // 一条合并规则：(a, b) → id；count = 学到它时这一对在语料里出现的次数
                struct Merge { int a, b, id, count; };

                // 字节级 BPE tokenizer：merges 按学到的先后排列（下标 = rank），vocab[id] = 这个 token 代表的字节串
                struct BPE {
                    std::vector<Merge> merges;
                    std::vector<std::string> vocab;                  // std::string 当字节串用：0..255 是单个字节，之后每条规则一个
                    std::map<std::pair<int, int>, int> rank;         // (a, b) → 规则序号（编码时查，Step 04）
                };

                BPE trainBPE(const std::string& text, int numMerges) {
                    ChunkTable T = chunkTable(text);                 // 预切分 + 去重：{块: 出现次数}，每块从 UTF-8 字节开始
                    BPE tok;
                    for (int i = 0; i < 256; i++) tok.vocab.push_back(std::string(1, (char)i));   // 初始词表 = 256 个字节
                    for (int k = 0; k < numMerges; k++) {
                        std::vector<PairCount> stats = pairStats(T.chunks, T.counts);
                        if (stats.empty()) break;                    // 每块都只剩 1 个 token 了
                        PairCount best = stats[0];
                        for (const PairCount& p : stats) if (p.n > best.n) best = p;   // 最频繁的一对（严格大于：平手取最先出现的）
                        int newId = 256 + k;
                        for (std::vector<int>& c : T.chunks) c = mergePair(c, best.a, best.b, newId);
                        tok.merges.push_back({best.a, best.b, newId, best.n});
                        tok.rank[{best.a, best.b}] = k;
                        tok.vocab.push_back(tok.vocab[best.a] + tok.vocab[best.b]);   // 新 token 的字节 = 两半拼接
                    }
                    return tok;
                }`,
            },
            previews: [
                {
                    id: 'mergeTable', label: '合并规则表', sub: '截至第 k 条的最近 13 条规则（黄 = 最新一条）· 红虚框 = 不是完整 UTF-8 的「半个字」', large: true, size: [360, 360],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c, {});
                        const R = S.R, k = S.bpeK.k, N = 13, from = Math.max(0, k - N);
                        ctx.font = P03.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`词表 V = 256 + ${k} = ${256 + k}`, 6, 5);
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.dim; ctx.textAlign = 'right'; ctx.fillText('合并时的次数', w - 6, 5);
                        if (k === 0) { ctx.textAlign = 'left'; ctx.fillStyle = LM.C.gray; ctx.fillText('k = 0：还没有任何合并，词表就是 256 个字节。', 6, 40); return; }
                        for (let r = from; r < k; r++) {
                            const m = R.merges[r], y = 26 + (r - from) * 25;
                            if (r === k - 1) { ctx.fillStyle = 'rgba(232,194,90,0.12)'; ctx.fillRect(2, y - 2, w - 4, 24); }
                            ctx.font = P03.FONT(11); ctx.fillStyle = r === k - 1 ? LM.C.yellow : LM.C.dim; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText('#' + r, 36, y + 10);
                            const tl = P03.tokObj(R, m.a), tr = P03.tokObj(R, m.b), tn = P03.tokObj(R, m.id);
                            let px = 12, wl = P03.tokWidth(ctx, tl, px), wr = P03.tokWidth(ctx, tr, px), wn = P03.tokWidth(ctx, tn, px);
                            if (42 + wl + wr + wn + 36 + 46 > w) { px = 11; wl = P03.tokWidth(ctx, tl, px); wr = P03.tokWidth(ctx, tr, px); wn = P03.tokWidth(ctx, tn, px); }
                            ctx.save(); ctx.beginPath(); ctx.rect(0, y - 2, w - 46, 24); ctx.clip();
                            let x = 42;
                            P03.tokBox(ctx, x, y, tl, px, wl); x += wl;
                            ctx.font = P03.FONT(12); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('+', x + 7, y + 10); x += 14;
                            P03.tokBox(ctx, x, y, tr, px, wr); x += wr;
                            ctx.font = P03.FONT(12); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('→', x + 11, y + 10); x += 22;
                            P03.tokBox(ctx, x, y, Object.assign(tn, { outline: r === k - 1 ? LM.C.yellow : null }), px, wn);
                            ctx.restore();
                            ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(String(m.count), w - 6, y + 10);
                        }
                    },
                },
                {
                    id: 'pairTop', label: '下一条规则的候选：相邻对频率 Top-10', sub: '在「前 k 条规则」切分后的语料上重新统计 · 绿 = 计数最高 = 第 k 条规则', size: [360, 230],
                    render: (c, S) => {
                        const R = S.R, top = S.bpeK.top;
                        if (!top.length) { LM.draw.text(c, ['没有可以合并的相邻对了：每个预切分块都只剩 1 个 token。'], { px: 12 }); return; }
                        const lab = (t) => { const s = `${P03UI.tokStr(R, t.a)} + ${P03UI.tokStr(R, t.b)}`; return s.length > 20 ? s.slice(0, 19) + '…' : s; };
                        LM.draw.bars(c, top.map((t) => t.n), top.map(lab), { horizontal: true, hi: 0, color: '#3d6f96', digits: 0, title: `第 ${S.bpeK.k} 条规则将合并：${lab(top[0])}` });
                    },
                },
                {
                    id: 'mergeKinds', label: '每条规则产生的新 token', sub: '横轴 = 规则序号，纵轴 = 新 token 的字节数 · 亮点 = 前 k 条（已采用）· 悬停看是哪个 token', size: [360, 230],
                    render: (c, S) => {
                        const R = S.R, k = S.bpeK.k, ks = S.bpeK.kinds;
                        const pts = R.merges.map((m, r) => ({ x: r, y: R.vocab[m.id].length, color: P03.KCOL[m.kind], r: 2.2, alpha: r < k ? 0.95 : 0.25 }));
                        const ymax = Math.max(6, ...pts.map((p) => p.y)) + 2;
                        S._kindTf = LM.draw.scatter(c, pts, { x: [-6, P03.MAXM + 6], y: [0, ymax], xLabel: '规则序号 r', yLabel: '字节数', lines: [{ a: [k, 0], b: [k, ymax], color: LM.C.yellow, dash: [3, 3] }], title: `前 ${k} 条：含半个字 ${ks.part} · 单个字符 ${ks.char} · 多个字符 ${ks.multi}` });
                        S._kindPts = pts;
                        const { ctx } = PL.draw.ctx(c, { clear: false });
                        P03.legend(ctx, [['含半个字', P03.KCOL.part], ['单个字符', P03.KCOL.char], ['多个字符', P03.KCOL.multi]], 44, 30, 10);
                    },
                    probe: (x, y, S, PL, info) => {
                        const i = P03UI.nearest(S._kindTf, S._kindPts || [], info, 8); if (i < 0) return '';
                        const m = S.R.merges[i]; return `#${i}「${P03UI.tokStr(S.R, m.id)}」${S.R.vocab[m.id].length} 字节 · ${P03.KNAME[m.kind]} · 合并时出现 ${m.count} 次`;
                    },
                },
                {
                    id: 'corpusToks', label: '训练语料的开头（前 k 条规则切分后）', sub: '每个色块 = 一个 token（颜色按 id）· 灰底 = 单个原始字节 · 黄字 = 十六进制显示的游离字节', size: [720, 150],
                    render: (c, S) => {
                        const R = S.R, e = S.bpeK.excerpt, items = [];
                        e.pieces.forEach((p, pi) => { if (pi) items.push({ gap: 4 }); p.ids.forEach((id) => items.push(P03.tokObj(R, id))); });
                        P03.drawToks(c, items, { px: 13, gap: 2, title: `${[...S.bpeK.exText].length} 个字符 → ${e.ids.length} 个 token（k = ${S.bpeK.k}）` });
                    },
                },
            ],
            notes: '把 <b>合并次数 k</b> 从 0 慢慢拖到 60：先是字节拼成字（红虚框的「半个字」夹在中间），再拼成「喜欢」「小猫」这样的词；拖到 300 以上，「熊猫住在山里」这种整句块也成了一个 token——这是玩具语料只有 10 个句式造成的，真实语料多样得多，token 大多停在词和词根一级。<b>英文比例</b>不为 0 时，「he」「The」「 the」「 likes」这样带前导空格的英文 token 会穿插出现（GPT-2 的正则把空格归到后面的单词上）。',
            warn: '<b>k 只是截取前缀</b>：BPE 是贪心的，训练 400 次得到的前 k 条规则与只训练 k 次完全相同，所以页面只训练一次（每个英文比例一次，约 15 ms），拖动 k 只截取规则表的前 k 条。平手时取最先出现的一对（C++ 版的 <code>pairStats</code> 按首次出现的顺序存放，<code>trainBPE</code> 用严格大于找最大值），JS 与 C++ 逐条相同。',
        },
        // ─────────────────────────────── 03 压缩率 ───────────────────────────────
        {
            id: 'compress', title: '压缩率与词表大小的代价', tag: 'BPE',
            desc: '每多一条合并规则，语料就少若干个 token。<strong>压缩率</strong>（平均每个 token 覆盖几个字符）随词表增大而上升，但收益递减：最常见的组合最先被合并，后面的规则每条只能省下很少的 token。词表也不是越大越好：embedding 矩阵是 V × d，输出层在每个位置都要对 V 个 token 算 logit 和 softmax；词表越大，每个 token 在语料里出现得越少，它的向量也越难学好。这里同时画了<strong>训练语料</strong>和两份<strong>留出文本</strong>（没参与训练的中文、英文句子）：tokenizer 只会压缩它见过的东西——英文比例为 0 时，英文文本始终是 1 个字节 1 个 token。',
            blocks: [
                { type: 'formula', text: '压缩率 = 字符数 / token 数      // k = 0 时一个汉字 = 3 个字节 token，压缩率 1/3' },
                { type: 'formula', text: 'Embedding 参数 = V · d      输出层每个位置 ≈ 2 · V · d 次浮点运算      注意力 ∝ T^2，FFN ∝ T      // T = 序列的 token 数' },
                {
                    type: 'code', title: 'C++ · compressionCurve', src: `
                    // 全部块的 token 总数（每块按它的出现次数加权）
                    int totalTokens(const ChunkTable& T) {
                        int n = 0;
                        for (size_t i = 0; i < T.chunks.size(); i++) n += (int)T.chunks[i].size() * T.counts[i];
                        return n;
                    }

                    // 一份文本在「前 k 条规则」下的 字符数 / token 数，k = 0, 1, …, K。
                    // 按规则顺序依次合并，与 bpeEncode（Step 04）的结果相同：第 r 条规则产生的新 token 只会出现在更晚的规则里，
                    // 所以编码时规则也是按 rank 从小到大用上的
                    std::vector<float> compressionCurve(const std::string& text, const BPE& tok) {
                        ChunkTable T = chunkTable(text);
                        float nChars = (float)utf8Decode(text).cp.size();
                        std::vector<float> curve = {nChars / totalTokens(T)};    // k = 0：一个字节一个 token
                        for (const Merge& m : tok.merges) {
                            for (std::vector<int>& c : T.chunks) c = mergePair(c, m.a, m.b, m.id);
                            curve.push_back(nChars / totalTokens(T));
                        }
                        return curve;
                    }
                    // 三条曲线：compressionCurve(训练语料, tok)、compressionCurve(留出中文, tok)、compressionCurve(留出英文, tok)`,
                },
                {
                    type: 'previews', items: [
                        {
                            id: 'compressCurve', label: '压缩率 vs 词表大小', sub: '字符 / token · 黄线 = 当前 k', large: true, size: [360, 250],
                            render: (c, S) => {
                                const R = S.R, K = R.merges.length, k = S.bpeK.k, xs = Array.from({ length: K + 1 }, (_, i) => 256 + i);
                                const tr = R.ntok.map((n) => R.nChars / n), zh = R.hTok[0].map((n) => R.heldChars[0] / n), en = R.hTok[1].map((n) => R.heldChars[1] / n);
                                PL.draw.plot(c, {
                                    x: [256, 256 + P03.MAXM], y: [0, 1.4 * Math.max(...tr, ...zh, ...en)], xLabel: '词表大小 V = 256 + k', yLabel: '字符 / token',
                                    series: [{ data: tr, x: xs, color: LM.C.blue, label: '训练语料', width: 2.5 }, { data: zh, x: xs, color: LM.C.green, label: '留出中文', dash: [5, 3], width: 1.5 }, { data: en, x: xs, color: LM.C.orange, label: '留出英文', width: 1.5 }],
                                    marks: [{ x: 256 + k, label: `中文 ${zh[Math.min(k, K)].toFixed(2)} · 英文 ${en[Math.min(k, K)].toFixed(2)} 字符/token` }],
                                });
                            },
                        },
                        {
                            id: 'costs', label: '序列变短 vs 词表变大', sub: '以 k = 0 为 1（对数纵轴）· 中文留出文本', size: [360, 250],
                            render: (c, S) => {
                                const R = S.R, K = R.merges.length, k = S.bpeK.k, xs = Array.from({ length: K + 1 }, (_, i) => 256 + i), T0 = R.hTok[0][0];
                                const T = R.hTok[0].map((n) => n / T0);
                                PL.draw.plot(c, {
                                    x: [256, 256 + P03.MAXM], logY: true, y: [3e-3, 6], xLabel: '词表大小 V',
                                    series: [{ data: xs.map((v) => v / 256), x: xs, color: LM.C.purple, label: 'V（∝ embedding、softmax）', width: 2 }, { data: T, x: xs, color: LM.C.green, label: '每句 token 数 T', width: 2 }, { data: T.map((v) => v * v), x: xs, color: LM.C.red, label: 'T²（∝ 注意力）', dash: [5, 3] }],
                                    marks: [{ x: 256 + k, label: `T = ${(R.hTok[0][Math.min(k, K)] / R.heldSent[0]).toFixed(1)} token/句，E = ${256 + k}×d` }], hmarks: [{ y: 1, color: '#444' }],
                                });
                            },
                        },
                    ],
                },
                {
                    type: 'html', html: `<table class="lm-tbl"><tr><th>模型</th><th>词表 V</th><th>d_model</th><th>Embedding V·d</th><th>说明</th></tr>
                        <tr><td>GPT-2 small</td><td>50 257</td><td>768</td><td>约 3860 万</td><td>输入与输出层共享，约占 1.24 亿参数的 31%</td></tr>
                        <tr><td>LLaMA-2 7B</td><td>32 000</td><td>4096</td><td>约 1.31 亿</td><td>输入、输出各一份（不共享）</td></tr>
                        <tr><td>LLaMA-3 8B</td><td>128 256</td><td>4096</td><td>约 5.25 亿</td><td>输入、输出各一份；词表变大 4 倍，同样文本的 token 更少</td></tr>
                        <tr><td>本系列迷你 GPT</td><td>76（字符级）</td><td>32</td><td>2 432</td><td>输入与输出层共享，占 28 928 个参数的 8.4%</td></tr></table>`,
                },
                { type: 'note', html: '拖 <b>合并次数 k</b>：蓝线（训练语料）一路上升，因为规则就是从它学来的；绿线（留出中文）几乎贴着它——玩具世界的新句子和训练句子来自同样的 10 个句式。把 <b>英文比例</b> 从 10% 调到 0%：橙线变成一条水平线，没有一条规则对英文有用。' },
                { type: 'warn', html: '评估 tokenizer 要看<strong>留出文本</strong>。这里的压缩率高得不真实（k = 400 时一个 token 平均 3–4 个汉字），因为玩具语料极度重复；真实语料多样得多，平均每个 token 覆盖的汉字要少得多。真实 tokenizer 的训练语料要大得多，还会刻意调配各种语言和代码的比例（Step 12）。' },
            ],
        },
        // ─────────────────────────────── 04 编码 / 解码 ───────────────────────────────
        {
            id: 'encode', title: '编码与解码：用学到的规则切新句子', tag: 'BPE',
            desc: '编码时<strong>不再统计频率</strong>，只按训练时学到的先后顺序应用规则：先预切分，每块转成字节，然后反复找「排名最靠前（最早学到）的规则」能合并的相邻对，合并它，直到没有规则可用。解码更简单：把每个 token 的字节拼起来，整体按 UTF-8 解码——即使某个 token 本身是「半个字」，拼起来也是完整的，所以 BPE 是<strong>无损</strong>的：对任何字符串都有 decode(encode(s)) = s。没见过的字（「大象」「苹果」）不会出错，只是退回到字节、多花几个 token。',
            formula: [
                'encode：ids ← bytes(块)；重复 { (a, b) ← argmin_{相邻对} rank(a, b)；若无规则可用则停；ids ← merge(ids, (a, b)) }',
                'decode：s = UTF8_decode( bytes(id_1) ‖ bytes(id_2) ‖ … ‖ bytes(id_T) )      // 先拼字节，再整体解码',
            ],
            code: {
                title: 'C++ · encodeChunk / bpeEncode / bpeDecode', src: String.raw`
                // 编码一个预切分块：反复找「rank 最小（最早学到）」的规则能合并的相邻对，合并它，直到没有规则可用；只用前 k 条规则
                std::vector<int> encodeChunk(const std::string& chunk, const BPE& tok, int k) {
                    std::vector<int> ids = bytesOf(chunk);
                    while (ids.size() >= 2) {
                        int best = k;                                // 最早学到的规则优先（best == k 表示一条都用不上）
                        for (size_t j = 0; j + 1 < ids.size(); j++) {
                            auto it = tok.rank.find({ids[j], ids[j + 1]});
                            if (it != tok.rank.end() && it->second < best) best = it->second;
                        }
                        if (best == k) break;                        // 没有规则可用了
                        const Merge& m = tok.merges[best];
                        ids = mergePair(ids, m.a, m.b, m.id);
                    }
                    return ids;
                }

                std::vector<int> bpeEncode(const std::string& text, const BPE& tok, int k) {
                    std::vector<int> ids;
                    for (const std::string& chunk : pretokenize(text)) {   // 先预切分：合并只在块内发生
                        std::vector<int> c = encodeChunk(chunk, tok, k);
                        ids.insert(ids.end(), c.begin(), c.end());
                    }
                    return ids;
                }

                // 解码：先把所有 token 的字节拼起来，再整体当 UTF-8 文本（std::string 存的就是字节，拼完就是原文）
                std::string bpeDecode(const std::vector<int>& ids, const BPE& tok) {
                    std::string bytes;
                    for (int id : ids) bytes += tok.vocab[id];
                    return bytes;
                }

                // 把字节串当 UTF-8 显示：不完整的字符（单独解码一个「半个字」token）换成 U+FFFD「�」，与浏览器的 TextDecoder 相同
                std::string utf8Replace(const std::string& s) {
                    std::string out;
                    for (size_t i = 0; i < s.size();) {
                        unsigned char b = (unsigned char)s[i];
                        size_t n = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;   // 0：10xxxxxx 不能开头
                        size_t k = 1;
                        while (n > 1 && k < n && i + k < s.size() && ((unsigned char)s[i + k] >> 6) == 2) k++;      // 数后续字节
                        if (n > 0 && k == n) out += s.substr(i, n);  // 完整的字符：原样保留
                        else out += "\xEF\xBF\xBD";                  // 缺字节的半个字 / 游离的后续字节 → �
                        i += k;
                    }
                    return out;
                }`,
            },
            previews: [
                {
                    id: 'encToks', label: '编码结果', sub: '色块 = token，下方数字 = id（&lt; 256 是原始字节，≥ 256 是第 id − 256 条规则）· 块之间留空 · <b>点击 token</b> 看它所在块的合并过程（黄线）', large: true, size: [720, 132],
                    render: (c, S) => {
                        const R = S.R, e = S.enc, items = [];
                        e.pieces.forEach((p, pi) => { if (pi) items.push({ gap: 6 }); p.ids.forEach((id) => items.push(P03.tokObj(R, id, { sub: id, piece: pi }))); });
                        const r = P03.drawToks(c, items, { px: 15, gap: 2, title: `${e.nChars} 个字符 · ${e.nBytes} 个字节 → ${e.ids.length} 个 token（k = ${S.bpeK.k}）` });
                        S._encBoxes = r.boxes;
                        r.ctx.strokeStyle = LM.C.yellow; r.ctx.lineWidth = 2;
                        r.boxes.filter((b) => b.t.piece === S.chunkSel).forEach((b) => { r.ctx.beginPath(); r.ctx.moveTo(b.x, b.y + b.h + 3); r.ctx.lineTo(b.x + b.w, b.y + b.h + 3); r.ctx.stroke(); });
                    },
                    probe: (x, y, S, PL, info) => { const b = P03UI.hitBox(S._encBoxes, info); if (!b) return ''; const id = b.t.id; return `id ${id}${id >= 256 ? `（第 ${id - 256} 条规则）` : '（原始字节）'} · 字节 ${P03UI.bytesHex(S.R.vocab[id])} · ${P03.KNAME[b.t.kind]}`; },
                    pick: (x, y, S, PL, info) => { const b = P03UI.hitBox(S._encBoxes, info); if (b) S.chunkSel = b.t.piece; },
                },
                {
                    id: 'encTrace', label: '一个块的编码过程', sub: '每行 = 应用一条规则后的切分（# = 规则序号）· 太长时省略中间几步', size: [720, 290],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c, {});
                        const R = S.R, p = S.enc.pieces[S.chunkSel] || S.enc.pieces[0];
                        ctx.font = P03.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`块「${p.w.replace(/ /g, '␣')}」：${p.trace[0].ids.length} 个字节 → ${p.ids.length} 个 token，用了 ${p.trace.length - 1} 条规则`, 6, 5);
                        let lines = p.trace.map((t, i) => Object.assign({ i }, t));
                        if (lines.length > 11) lines = lines.slice(0, 5).concat([{ gap: true }], lines.slice(-5));
                        lines.forEach((ln, j) => {
                            const y = 26 + j * 24;
                            ctx.font = P03.FONT(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                            if (ln.gap) { ctx.fillStyle = LM.C.dim; ctx.textAlign = 'left'; ctx.fillText(`… 省略 ${p.trace.length - 10} 步 …`, 60, y + 10); return; }
                            ctx.fillStyle = ln.r < 0 ? LM.C.gray : LM.C.yellow; ctx.fillText(ln.r < 0 ? '字节' : '#' + ln.r, 50, y + 10);
                            let x = 58;
                            ln.ids.forEach((id) => { const t = P03.tokObj(R, id), bw = P03.tokWidth(ctx, t, 12); if (x + bw < w - 4) P03.tokBox(ctx, x, y, t, 12, bw); x += bw + 2; });
                        });
                    },
                },
                {
                    id: 'roundTrip', label: '解码：拼回字节再整体解码', sub: '红色 = 单独解码时不是完整 UTF-8，显示为 �（U+FFFD）', size: [720, 118],
                    render: (c, S) => {
                        const e = S.enc, solo = [];
                        e.solo.forEach((s, i) => { solo.push({ t: s.replace(/ /g, '␣'), color: s.includes('�') ? LM.C.red : LM.C.text }); if (i < e.solo.length - 1) solo.push({ t: ' │ ', color: LM.C.dim }); });
                        LM.draw.text(c, [
                            [{ t: '原文     ', color: LM.C.dim }, { t: e.s, color: LM.C.text }],
                            [{ t: 'decode   ', color: LM.C.dim }, { t: e.dec, color: e.ok ? LM.C.green : LM.C.red }, { t: e.ok ? '    ✓ 与原文逐字节相同' : '    ✗ 不一致', color: e.ok ? LM.C.green : LM.C.red }],
                            [{ t: '逐个 token 单独解码  ', color: LM.C.dim }].concat(solo),
                            [{ t: `字符 ${e.nChars} · UTF-8 字节 ${e.nBytes} · 字符级 token ${e.nChars}（OOV ${e.oov}）· BPE token ${e.ids.length}（${(e.nBytes / e.ids.length).toFixed(2)} 字节/token）`, color: LM.C.gray }],
                        ], { px: 12, lineGap: 7 });
                    },
                },
            ],
            notes: '默认 k = 120 时「鱼」被拆成「…吃 E9」+「B1」+「BC」三个 token（红虚框）：字节级 BPE 只看频率，不知道字的边界；把 <b>合并次数 k</b> 拖到 200，「吃鱼」「小猫喜欢吃鱼」才成为干净的 token。选「大象喜欢吃苹果」：「喜欢吃 + 苹的首字节 E8」照样被合成一个大 token，「大象」「苹果」的其余部分留在字节级。选「中英混排」并把 <b>英文比例</b> 调到 0%：英文全部退回单个字节。',
            warn: '编码必须按规则的<strong>学习顺序</strong>应用，不能「贪心地找最长的 token」——同一个词表、不同的应用顺序会给出不同的切分。另外，流式输出（逐 token 显示生成结果）时，单独解码一个「半个字」token 会得到 �：要把字节攒到构成完整的 UTF-8 字符再显示。',
        },
        // ─────────────────────────────── 05 Embedding 查表 ───────────────────────────────
        {
            id: 'lookup', title: 'Token id → 向量：Embedding 就是查表', tag: 'EMBED',
            desc: 'token id 只是一个编号，数值大小没有意义（「猫」= 5、「鱼」= 6 不代表它们相近）。把 id 变成向量的标准做法是准备一个可训练的矩阵 <strong>E ∈ ℝ<sup>V×d</sup></strong>，第 i 行就是 token i 的向量。数学上它等于 one-hot 向量乘以 E；实现上就是<strong>按下标取一行</strong>（gather），不必真的做矩阵乘法，反向传播时梯度也只落在被取到的那几行。下面是第 04–06 篇基座模型（字符级，V = 76，d = 32）<strong>训练后</strong>的 E：在句子里扮演相同角色的字互相靠近——「小 / 熊 / 猴 / 老」（动物名的第一个字）一簇，「猫 / 狗 / 兔 / 鸟 / 羊」一簇，食物用字一簇，颜色用字一簇。',
            formula: [
                'E ∈ ℝ^{V × d}      onehot(i) · E = E[i, :]      // V = 76 个 token，d = 32 维：查表 = 取第 i 行',
                '∂L/∂E[j, :] = Σ_{t: id_t = j} ∂L/∂x_t      // 输入侧：只有出现过的 token 收到梯度',
                'cos(E_i, E_j) = E_i · E_j / (‖E_i‖ ‖E_j‖)',
            ],
            code: {
                title: 'C++ · one-hot × E = 查表', src: `
                // 第 i 个 token 的 one-hot 行向量 [1 × V]
                Tensor oneHot(int i, int V) {
                    Tensor t = zeros(1, V);
                    t.at(0, i) = 1.f;
                    return t;
                }

                void lookupDemo() {
                    Vocab vocab = baseVocab();                       // 字符级词表：76 个 token（见 Common）
                    GPTConfig cfg;
                    GPT model = makeGPT(cfg, 1);                     // 第 04 篇逐层拆解的迷你 GPT
                    loadWeights(model, "base.bin");                  // 与页面里 llm_weights.js 相同的基座权重
                    NoGrad ng;                                       // 只做前向，不记录计算图
                    int V = cfg.V, i = tokenId(vocab, "猫");
                    Tensor a = matmul(constant(oneHot(i, V)), model.wte)->val;   // onehot(i) · E：[1 × 76] · [76 × 32]
                    Tensor b = embedding({i}, model.wte)->val;       // E[i]：直接取第 i 行（gather）
                    assert(a.v == b.v);                              // 乘 one-hot = 取第 i 行
                    Tensor En = normRows(model.wte->val);            // 每行归一化后，点积 = 余弦相似度
                    std::vector<int> order;
                    for (int j = 0; j < V; j++) if (j != i) order.push_back(j);
                    std::sort(order.begin(), order.end(), [&](int x, int y) { return dotRow(En, i, En, x) > dotRow(En, i, En, y); });
                    for (int r = 0; r < 10; r++) std::printf("%s ", vocab.itos[order[r]].c_str());   // 与「猫」最像的 10 个 token
                }`,
            },
            previews: [
                {
                    id: 'lookupView', label: 'onehot(i) · E = E[i]', sub: '中间 = 基座模型训练后的 E（76 行 × 32 列，蓝负橙正）· 黄框 = 被取出的那一行', large: true, size: [720, 280],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, {});
                        const m = S.lm, V = m.cfg.V, d = m.cfg.d, E = m.wte.data, id = S.tok, rng = [-0.8, 0.8], top = 30, ch = (h - top - 14) / V;
                        ctx.font = P03.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                        ctx.fillText('onehot', 68, top - 4);
                        const ox = 62; for (let i = 0; i < V; i++) { ctx.fillStyle = i === id ? LM.C.yellow : '#1a1d22'; ctx.fillRect(ox, top + i * ch, 12, Math.max(1, ch - 0.5)); }
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.yellow; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(`${LM.vocab.show(id)} ${id}`, ox - 4, top + (id + 0.5) * ch);
                        ctx.font = P03.FONT(18); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'center'; ctx.fillText('×', 90, top + V * ch / 2);
                        const ex = 104, cw = 6;
                        P03.heatAt(ctx, ex, top, cw, ch, E, V, d, { range: rng, hiRow: id, title: `E  [${V} × ${d}]` });
                        ctx.font = P03.FONT(18); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('=', ex + d * cw + 18, top + V * ch / 2);
                        const rx = ex + d * cw + 38, rw = (w - rx - 44) / d, row = E.subarray(id * d, (id + 1) * d);
                        P03.heatAt(ctx, rx, top + 4, rw, 22, row, 1, d, { range: rng, title: `E[${id}] =「${LM.vocab.show(id)}」的向量 [1 × ${d}]` });
                        const by0 = top + 48, bh = h - by0 - 44, mid = by0 + bh / 2;
                        ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(rx, mid); ctx.lineTo(rx + d * rw, mid); ctx.stroke();
                        for (let j = 0; j < d; j++) { const v = Math.max(-1, Math.min(1, row[j] / rng[1])); ctx.fillStyle = v >= 0 ? '#e0a060' : '#5ab0ff'; ctx.fillRect(rx + j * rw + 1, v >= 0 ? mid - v * bh / 2 : mid, rw - 2, Math.abs(v) * bh / 2); }
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.dim; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText('+0.8', rx - 4, by0); ctx.fillText('−0.8', rx - 4, by0 + bh); ctx.fillText('0', rx - 4, mid);
                        const W = S.wte, unseen = W.unseen.includes(id);
                        ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = unseen ? LM.C.red : LM.C.gray;
                        ctx.fillText(`‖E[${id}]‖ = ${W.norms[id].toFixed(2)}（初始化时所有行平均 ≈ ${W.initNorm.toFixed(2)}）`, rx, h - 34);
                        if (unseen) ctx.fillText(`预训练里从没出现：与另外 ${W.unseen.length - 1} 个未见 token 坍缩成同一个向量`, rx, h - 18);
                        P03.colorBar(ctx, w - 40, top, V * ch * 0.5, rng);
                    },
                },
                {
                    id: 'wtePca', label: 'E 的 2D PCA（出现过的 67 个字）', sub: '每行先归一化（只看方向）· <b>点击</b>选 token · ' + ['name1', 'name2', 'food', 'home', 'color', 'other'].map((k) => `<span style="color:${P03.ROLECOL[k]}">■ ${P03.ROLENAME[k]}</span>`).join(' '), size: [360, 320],
                    render: (c, S) => {
                        const W = S.wte;
                        const pts = W.seen.map((t, r) => { const ch = LM.vocab.itos[t], role = P03.ROLE.get(ch) || 'other'; return { x: W.pts[2 * r], y: W.pts[2 * r + 1], color: P03.ROLECOL[role], r: t === S.tok ? 4.5 : 3, ring: t === S.tok ? LM.C.yellow : null, tok: t, role }; });
                        const tf = LM.draw.scatter(c, pts, { equal: true, xLabel: `PC1（${(W.varRatio[0] * 100).toFixed(0)}% 方差）`, yLabel: `PC2（${(W.varRatio[1] * 100).toFixed(0)}%）` });
                        S._wteTf = tf; S._wtePts = pts;
                        const { ctx, w, h } = PL.draw.ctx(c, { clear: false });
                        P03.placeLabels(ctx, pts.map((p) => ({ x: tf.X(p.x), y: tf.Y(p.y), r: p.r, text: LM.vocab.itos[p.tok], color: p.tok === S.tok ? LM.C.yellow : p.color, g: p.role, maxR: 29 })), w - 6, h - 30, 36, 4);
                    },
                    probe: (x, y, S, PL, info) => { const i = P03UI.nearest(S._wteTf, S._wtePts || [], info); if (i < 0) return ''; const t = S._wtePts[i].tok; return `「${LM.vocab.show(t)}」 id ${t} · ‖E‖ = ${S.wte.norms[t].toFixed(2)}`; },
                    pick: (x, y, S, PL, info) => { const i = P03UI.nearest(S._wteTf, S._wtePts || [], info); if (i >= 0) PL.setParam('tok', S._wtePts[i].tok); },
                },
                {
                    id: 'wteNN', label: '最近邻（余弦相似度）', sub: '在全部 76 个 token 里找 · 红 = 预训练里从没出现过的 token', size: [360, 320],
                    render: (c, S) => {
                        const W = S.wte, V = W.V, id = S.tok, nn = [];
                        for (let j = 0; j < V; j++) if (j !== id) nn.push([j, W.cos[id * V + j]]);
                        nn.sort((a, b) => b[1] - a[1]);
                        const top = nn.slice(0, 10), role = (t) => (W.unseen.includes(t) ? 'unseen' : (P03.ROLE.get(LM.vocab.itos[t]) || 'other'));
                        LM.draw.bars(c, top.map((t) => t[1]), top.map((t) => `${LM.vocab.show(t[0])} #${t[0]}`), { horizontal: true, range: [Math.min(0, ...top.map((t) => t[1])), 1], colors: top.map((t) => P03.ROLECOL[role(t[0])]), digits: 3, title: `与「${LM.vocab.show(id)}」最像的 10 个 token` });
                    },
                },
            ],
            notes: '拖 <b>查看的 token</b>，或在 PCA 图上点一个字：「猫」的近邻是「羊 鸟 兔 狗 虎」，「鱼」的近邻是「竹 肉 头 卜 虫」，「小」的近邻是「熊 猴 老」。再试「⟨pad⟩」或「什」：最近邻全是相似度 1.000 的其他未见 token。',
            warn: [
                '本系列模型的输入 embedding 与输出层<strong>共享</strong> E（weight tying，第 04 篇）。输出层的 softmax 会给 E 的<strong>每一行</strong>梯度：9 个在预训练语料里从没出现过的 token（4 个特殊 token，以及只在问句里出现的「什 么 ？ 哪 颜」）只受到「别预测我」这一种推力，训练结束时已坍缩成<strong>同一个向量</strong>（两两余弦相似度 1.000，范数约 3.3；初始化时约 0.11）。第 06 篇 SFT 时这些 token 才第一次真正被用到。',
                '字符级向量的结构比较弱（同类字之间的平均余弦只有 0.1–0.3）：一个字的意思要靠上下文组合，这部分工作由后面的注意力层完成。',
            ],
        },
        // ─────────────────────────────── 06 word2vec ───────────────────────────────
        {
            id: 'w2v', title: 'Embedding 是学出来的：word2vec（skip-gram + 负采样）', tag: 'EMBED',
            desc: 'E 不是人设计的，是训练出来的。最经典、也最容易看清的例子是 <strong>word2vec</strong>（Mikolov 等 2013）：<strong>skip-gram</strong> 用中心词的向量 v 去预测窗口里的上下文词；为了不必对整个词表做 softmax，<strong>负采样</strong>（negative sampling）把它改成二分类——真实的（中心词, 上下文词）对要打高分，随机抽来的 K 个「负样本」词要打低分。于是出现在相似上下文里的词得到相似的向量（分布假说：一个词的意思由它的邻居决定）。这里先把语料按词切开（正向最大匹配 + 47 个词的小词典），然后<strong>同时实时训练 6 个</strong>只有窗口大小不同的模型（每个 2 × 47 × 16 个参数）；大图是所选窗口模型的词向量（归一化后）的 2D PCA 投影。',
            formula: [
                'ℓ(c, o) = −log σ(u_o · v_c) − Σ_{k=1}^{K} log σ(−u_{n_k} · v_c)      n_k ~ P_n(w) ∝ f(w)^{0.75}',
                '∂ℓ/∂v_c = (σ(u_o · v_c) − 1) · u_o + Σ_k σ(u_{n_k} · v_c) · u_{n_k}      ∂ℓ/∂u_o = (σ(u_o · v_c) − 1) · v_c',
                'θ ← θ − η · ∂ℓ/∂θ      η = η_0 · (1 − t / t_max)      // 逐对 SGD，学习率线性衰减（η_0 = 0.05）',
            ],
            code: [
                {
                    title: 'C++ · sgnsUpdate / w2vTrain（手写 SGD，与页面逐步相同）', src: `
                    // 一个 (中心词 c, 上下文词 o) 对 + 负样本的 SGD 更新，返回这一对的损失。
                    // 所有梯度都在更新前的参数上算好，最后一起更新（= batch = 1 的 SGD）
                    float sgnsUpdate(Tensor& Win, Tensor& Wout, int c, int o, const std::vector<int>& negs, float lr) {
                        int d = Win.cols;
                        std::vector<int> ids = {o};                  // 第 0 个是正样本（标签 1），其余是负样本（标签 0）
                        for (int n : negs) if (n != o) ids.push_back(n);
                        std::vector<float> v = floats(d), g = floats(d), gq = floats((int)ids.size());
                        for (int j = 0; j < d; j++) v[j] = Win.at(c, j);   // 先拷贝 v = W_in[c]
                        float loss = 0;
                        for (size_t q = 0; q < ids.size(); q++) {
                            double z = 0;
                            for (int j = 0; j < d; j++) z += Wout.at(ids[q], j) * v[j];        // u_q · v
                            float p = (float)(1 / (1 + std::exp(-z)));                         // σ(u·v)：判为「真上下文」的概率
                            gq[q] = p - (q == 0 ? 1.f : 0.f);                                  // ∂loss/∂(u·v) = σ − 标签
                            loss -= std::log(std::max(q == 0 ? p : 1 - p, 1e-12f));
                            for (int j = 0; j < d; j++) g[j] += gq[q] * Wout.at(ids[q], j);    // ∂loss/∂v = Σ_q gq · u_q
                        }
                        for (size_t q = 0; q < ids.size(); q++)
                            for (int j = 0; j < d; j++) Wout.at(ids[q], j) -= lr * gq[q] * v[j];   // ∂loss/∂u_q = gq · v（重复的负样本各更新一次）
                        for (int j = 0; j < d; j++) Win.at(c, j) -= lr * g[j];
                        return loss;
                    }

                    // 一个 word2vec 模型：输入向量 W_in（训练完留下来用的词向量）、输出向量 W_out，各 [V × d]；rng 用来抽负样本
                    struct Word2Vec { Tensor Win, Wout; Rng rng; int done = 0; };

                    Word2Vec w2vInit(uint32_t seed, int V, int d) {
                        Word2Vec M;
                        M.rng = makeRng(seed);
                        M.Win = zeros(V, d);
                        for (float& x : M.Win.v) x = (float)((uniform(M.rng) - 0.5) / d);   // word2vec.c 的初始化：v ~ U(−0.5/d, 0.5/d)，u = 0
                        M.Wout = zeros(V, d);
                        return M;
                    }

                    // 接着训练 n 对（第 t 对 = pairs[t mod 总数]）：每对抽 K 个负样本；学习率从 lr0 线性降到 0（下限 1e-4·lr0）。返回这 n 对的平均损失
                    float w2vTrain(Word2Vec& M, const std::vector<SkipPair>& pairs, const std::vector<int>& table, int n, int K, float lr0, int total) {
                        double L = 0;
                        std::vector<int> negs = ints(K);
                        for (int p = 0; p < n; p++, M.done++) {
                            const SkipPair& s = pairs[M.done % pairs.size()];
                            float lr = lr0 * std::max(1e-4f, 1.f - (float)M.done / total);
                            for (int k = 0; k < K; k++) negs[k] = table[randint(M.rng, (int)table.size())];   // 按 f^0.75 抽负样本
                            L += sgnsUpdate(M.Win, M.Wout, s.c, s.o, negs, lr);
                        }
                        return (float)(L / std::max(1, n));
                    }

                    // 页面里的流程：3000 句语料 → 分词 → 训练对 + 负采样表 → 每帧训练 500 对，共 120 000 对（6 种窗口各一个模型）
                    Word2Vec trainWord2vec(const std::vector<std::string>& corpus, int window, int K, uint32_t seed, Rng& r) {
                        Lexicon lex = makeLexicon(WORDS);
                        int V = (int)lex.words.size(), d = 16;
                        std::vector<std::vector<int>> sents;
                        for (const std::string& s : corpus) sents.push_back(segmentWords(s, lex));   // 「小猫喜欢吃鱼。」→ 小猫 喜欢 吃 鱼
                        std::vector<SkipPair> pairs = skipgramPairs(sents, window, r);
                        std::vector<int> table = unigramTable(sents, V, 100000);
                        Word2Vec M = w2vInit(seed, V, d);
                        while (M.done < 120000) w2vTrain(M, pairs, table, 500, K, 0.05f, 120000);
                        return M;                                    // 词向量 = M.Win 的第 id 行
                    }`,
                },
                {
                    title: 'C++ · 同一个模型的批量 + 自动微分写法（两张 embedding 表）', src: `
                    // 用本系列的自动微分（Common · minml_autograd.h）来写：两张表都是参数，只写出损失，梯度由 backward 求
                    struct SkipGramNS { Var v, u; };                 // v = 中心词向量（W_in），u = 上下文向量（W_out），各 [V × d]

                    SkipGramNS makeSkipGramNS(int V, int d, Rng& r) {
                        Tensor init = zeros(V, d);
                        for (float& x : init.v) x = (float)((uniform(r) - 0.5) / d);
                        SkipGramNS m;
                        m.v = param(init);
                        m.u = param(zeros(V, d));
                        return m;
                    }

                    // 每行求和 [n × d] → [n × 1]（乘一个全 1 的列向量）
                    Var rowSum(const Var& X) { return matmul(X, constant(full(X->val.cols, 1, 1.f))); }

                    // B 个 (c, o) 对一起算；negs 是 B·K 个负样本（第 b 对的负样本在 negs[b·K … b·K+K)）
                    Var sgnsLoss(const SkipGramNS& m, const std::vector<int>& c, const std::vector<int>& o, const std::vector<int>& negs, int K) {
                        std::vector<int> cRep = ints((int)negs.size());
                        for (size_t i = 0; i < negs.size(); i++) cRep[i] = c[i / K];             // 每个负样本配上它的中心词
                        Var pos = rowSum(mul(embedding(o, m.u), embedding(c, m.v)));             // [B × 1]：u_o · v_c
                        Var neg = rowSum(mul(embedding(negs, m.u), embedding(cRep, m.v)));       // [B·K × 1]：u_n · v_c
                        Var ll = add(sumAll(logSigmoid(pos)), sumAll(logSigmoid(scale(neg, -1.f))));   // Σ log σ(u_o·v) + Σ log σ(−u_n·v)
                        return scale(ll, -1.f);                      // 用 sum：B = 1 且负样本不与 o 重复时，SGD 一步与上面的 sgnsUpdate 完全相同
                    }

                    // 一步 SGD；opt = makeSGD({m.v, m.u}, lr0)
                    float sgnsStep(SkipGramNS& m, SGD& opt, const std::vector<int>& c, const std::vector<int>& o, const std::vector<int>& negs, int K, float lr) {
                        zeroGrad(opt.params);
                        Var loss = sgnsLoss(m, c, o, negs, K);
                        backward(loss);
                        sgdStep(opt, lr);
                        return loss->val.v[0];
                    }`,
                },
            ],
            previews: [
                {
                    id: 'w2vPca', live: true, label: '词向量的 2D PCA（所选窗口）', sub: ['animal', 'food', 'home', 'color', 'func'].map((k) => `<span style="color:${P03.CATCOL[k]}">■ ${P03.CATNAME[k]}</span>`).join(' ') + ' · 坐标随训练实时变化（与上一帧对齐）· <b>点击</b>一个词设为 Step 07 的查询词', large: true, size: [360, 360],
                    render: (c, S) => {
                        const v = S.w2vView, M = v.M, L = P03.LEX;
                        const pts = L.words.map((wd, i) => ({ x: v.pts[2 * i], y: v.pts[2 * i + 1], color: P03.CATCOL[L.cat[i]], r: i === S.query ? 4.5 : L.cat[i] === 'func' ? 2.2 : 3.2, ring: i === S.query ? LM.C.yellow : null, alpha: L.cat[i] === 'func' ? 0.7 : 1 }));
                        const tf = LM.draw.scatter(c, pts, { x: [v.box[0], v.box[1]], y: [v.box[2], v.box[3]], equal: true, axes: false });
                        S._w2vTf = tf; S._w2vPts = pts;
                        const { ctx, w, h } = PL.draw.ctx(c, { clear: false });
                        const order = L.content.concat(L.words.map((_, i) => i).filter((i) => L.cat[i] === 'func'));
                        P03.placeLabels(ctx, order.map((i) => ({ x: tf.X(pts[i].x), y: tf.Y(pts[i].y), r: pts[i].r, text: L.words[i], color: i === S.query ? LM.C.yellow : L.cat[i] === 'func' ? '#6a7078' : P03.CATCOL[L.cat[i]], g: L.cat[i], maxR: L.cat[i] === 'func' ? 19 : 72 })), w, h);
                        ctx.font = P03.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`窗口 ${S.win} · 已训练 ${P03UI.fmtN(M.done)} / ${P03UI.fmtN(P03.W2V.total)} 对${M.done >= P03.W2V.total ? '（完成）' : ''}`, 44, 5);
                    },
                    probe: (x, y, S, PL, info) => { const i = P03UI.nearest(S._w2vTf, S._w2vPts || [], info); return i < 0 ? '' : `${P03.LEX.words[i]}（${P03.CATNAME[P03.LEX.cat[i]]}）`; },
                    pick: (x, y, S, PL, info) => { const i = P03UI.nearest(S._w2vTf, S._w2vPts || [], info); if (i >= 0) PL.setParam('query', i); },
                },
                {
                    id: 'w2vLoss', live: true, label: '训练损失（6 个窗口）', sub: '每帧的平均损失（EMA 平滑）· 黄 = 所选窗口 · 窗口越大，上下文越杂、损失越高', size: [360, 175],
                    render: (c, S) => {
                        const q = S.w2vView.q, K = S.neg, series = [];
                        const add = (M, i) => { if (!M.loss.length) return; series.push({ data: LM.ema(M.loss.map((p) => p[1]), 0.8), x: M.loss.map((p) => p[0]), color: i === q ? LM.C.yellow : LM.SERIES[i], width: i === q ? 2.5 : 1, label: i === q ? `窗口 ${P03.W2V.wins[i]}` : undefined }); };
                        S.w2v.forEach((M, i) => { if (i !== q) add(M, i); }); add(S.w2v[q], q);
                        const L0 = Math.log(2) * (K + 1);
                        PL.draw.plot(c, { x: [0, P03.W2V.total], y: [0, L0 + 0.4], xLabel: '已训练的 (中心词, 上下文) 对数', series, hmarks: [{ y: L0, label: `起点 (K+1)·ln2 = ${L0.toFixed(2)}`, color: '#555' }] });
                    },
                },
                {
                    id: 'w2vSep', live: true, label: '簇分离度（6 个窗口）', sub: '实词的「同类平均余弦 − 异类平均余弦」· 越大 = 动物 / 食物 / 住处 / 颜色分得越开', size: [360, 175],
                    render: (c, S) => {
                        const q = S.w2vView.q, series = [];
                        const add = (M, i) => { if (!M.sep.length) return; series.push({ data: M.sep.map((p) => p[1]), x: M.sep.map((p) => p[0]), color: i === q ? LM.C.yellow : LM.SERIES[i], width: i === q ? 2.5 : 1, label: i === q ? `窗口 ${P03.W2V.wins[i]}` : undefined }); };
                        S.w2v.forEach((M, i) => { if (i !== q) add(M, i); }); add(S.w2v[q], q);
                        PL.draw.plot(c, { x: [0, P03.W2V.total], y: [-0.1, 0.8], xLabel: '已训练的对数', series, hmarks: [{ y: 0, color: '#444' }] });
                    },
                },
                {
                    id: 'w2vPair', live: true, label: '这一帧最后一个训练样本（所选窗口）', sub: '黄框 = 中心词 c · 绿框 = 上下文词 o · 暗 = 窗口外 · 红框 = 随机抽到的负样本', size: [720, 86],
                    render: (c, S) => {
                        const M = S.w2vView.M, last = M.last;
                        if (!last) { LM.draw.text(c, [{ text: '（开始训练后显示：按顶栏的 ▶ 播放）', color: LM.C.dim }], { px: 12 }); return; }
                        const s = S.w2vSents[last.key >> 8], ci = (last.key >> 4) & 15, oi = last.key & 15;
                        const items = [{ gap: 38 }, { label: '句子', color: LM.C.gray }];
                        s.forEach((wid, j) => items.push(P03UI.wordTok(wid, { outline: j === ci ? LM.C.yellow : j === oi ? LM.C.green : null, dim: Math.abs(j - ci) > S.win })));
                        items.push({ gap: 18 }, { label: `负样本 K = ${last.negs.length}`, color: LM.C.gray });
                        last.negs.forEach((n) => items.push(P03UI.wordTok(n, { outline: LM.C.red })));
                        const r = P03.drawToks(c, items, { px: 13, gap: 3 });
                        r.ctx.font = P03.FONT(11); r.ctx.fillStyle = LM.C.dim; r.ctx.textAlign = 'left'; r.ctx.textBaseline = 'top';
                        r.ctx.fillText(`目标：σ(u_${P03.LEX.words[s[oi]]} · v_${P03.LEX.words[s[ci]]}) → 1，σ(u_负样本 · v_${P03.LEX.words[s[ci]]}) → 0 · 当前学习率 ${last.lr.toFixed(4)}`, 6, r.y + 2);
                    },
                },
            ],
            notes: '默认窗口 2：几秒内 4 类词各聚成一团（动物、食物、住处、颜色）——同类词在句子里的「位置角色」相同。把 <b>word2vec 窗口</b> 切到 6（6 个模型一直在同时训练，不用等）：簇按<strong>主题</strong>重组，每只动物和它自己的食物、住处、颜色靠在一起（小猫 ↔ 鱼、屋里、白色）。改 <b>负样本 K</b> 会让 6 个模型从头训练。',
            warn: '与 word2vec.c 的差异：窗口大小固定（原实现对每个中心词随机缩小窗口，相当于近处的词权重更大）；没有做高频词下采样（玩具语料只有 47 个词，几乎全是「高频词」）；负样本与正样本相同时直接跳过。负采样分布 f<sup>0.75</sup>、初始化（v 在 ±0.5/d 内均匀分布，u = 0）、线性衰减的学习率与原实现相同。最终使用的是输入向量 v（W_in）。',
        },
        // ─────────────────────────────── 07 相似度与类比 ───────────────────────────────
        {
            id: 'sim', title: '相似度与类比：向量空间里的「意思」', tag: 'EMBED',
            desc: '向量学好以后，「意思相近」就可以用<strong>余弦相似度</strong>衡量。大图是所选窗口模型里 30 个实词两两之间的余弦相似度（按词类排序）：窗口小时是沿对角线的 4 个亮块（同类词相似）；窗口大时，亮的变成「同一只动物的名字和属性」。著名的<strong>类比</strong>测试 v(king) − v(man) + v(woman) ≈ v(queen)，在这里对应「小猫 : 鱼 = 小狗 : ？」——用 v(鱼) − v(小猫) + v(小狗) 找最近的词（排除题目里的 3 个词）。右下是 6 个窗口在 164 道题上的真实准确率：窗口 ≤ 4 时大多答错，窗口 5–6 时约九成答对。原因是只有窗口够大，「小猫」和「鱼」才出现在彼此的上下文里，向量里才会有一个「属于小猫」的分量——减去小猫、加上小狗，正好把它换成「属于小狗」的。',
            formula: [
                'cos(a, b) = a · b / (‖a‖ ‖b‖)',
                'a : b = c : ?   →   argmax_{x ∉ {a, b, c}} x̂ · (b̂ − â + ĉ)      // x̂ = x / ‖x‖（3CosAdd，Mikolov 等 2013）',
            ],
            code: {
                title: 'C++ · nearest / analogy（normRows、dotRow 见 Common）', src: `
                // 一个近邻：词 id 和余弦相似度
                struct Neighbor { int id; float sim; };

                // 与 query 最像的 k 个词（不含它自己）
                std::vector<Neighbor> nearest(const Tensor& W, int query, int k) {
                    Tensor Wn = normRows(W);                         // 每行归一化：点积 = 余弦相似度
                    std::vector<Neighbor> all;
                    for (int j = 0; j < W.rows; j++) if (j != query) all.push_back({j, dotRow(Wn, query, Wn, j)});
                    std::stable_sort(all.begin(), all.end(), [](const Neighbor& a, const Neighbor& b) { return a.sim > b.sim; });
                    all.resize(std::min(k, (int)all.size()));
                    return all;
                }

                // a : b = c : ?   →   argmax_x  x̂ · (b̂ − â + ĉ)，排除 a、b、c 本身（3CosAdd）
                int analogy(const Tensor& W, int a, int b, int c) {
                    Tensor Wn = normRows(W);
                    int best = -1;
                    double bestScore = -1e30;
                    for (int x = 0; x < W.rows; x++) {
                        if (x == a || x == b || x == c) continue;
                        double s = 0;
                        for (int j = 0; j < W.cols; j++) s += Wn.at(x, j) * (Wn.at(b, j) - Wn.at(a, j) + Wn.at(c, j));
                        if (s > bestScore) { bestScore = s; best = x; }
                    }
                    return best;
                }

                void analogyDemo(const Word2Vec& M) {
                    Lexicon lex = makeLexicon(WORDS);
                    int x = analogy(M.Win, lex.id.at("小猫"), lex.id.at("鱼"), lex.id.at("小狗"));
                    std::printf("%s\\n", lex.words[x].c_str());     // 窗口 6：骨头
                    for (const Neighbor& nb : nearest(M.Win, lex.id.at("小猫"), 8)) std::printf("%s %.3f\\n", lex.words[nb.id].c_str(), nb.sim);
                }`,
            },
            previews: [
                {
                    id: 'simHeat', live: true, label: '实词的余弦相似度矩阵（所选窗口）', sub: '行列顺序相同：动物 8 · 食物 8 · 住处 7 · 颜色 7 · 黄框 = 查询词', large: true, size: [360, 360],
                    render: (c, S) => {
                        const v = S.w2vView, C = P03.LEX.content, n = C.length, labels = C.map((i) => P03.LEX.words[i]), qi = C.indexOf(S.query);
                        const lay = LM.draw.heat(c, v.sim, n, n, { rowLabels: labels, range: [-1, 1], map: 'diverge', hiRow: qi >= 0 ? qi : null, hiCol: qi >= 0 ? qi : null, legend: false, cellMax: 9.8, title: `        窗口 ${S.win} · 余弦相似度` });
                        S._simLay = lay;
                        const { ctx } = PL.draw.ctx(c, { clear: false });
                        ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1;
                        [8, 16, 23].forEach((b) => { ctx.beginPath(); ctx.moveTo(lay.x0, lay.y0 + b * lay.ch); ctx.lineTo(lay.x0 + n * lay.cw, lay.y0 + b * lay.ch); ctx.moveTo(lay.x0 + b * lay.cw, lay.y0); ctx.lineTo(lay.x0 + b * lay.cw, lay.y0 + n * lay.ch); ctx.stroke(); });
                        P03.colorBar(ctx, lay.x0 + n * lay.cw + 7, lay.y0, n * lay.ch, [-1, 1]);
                    },
                    probe: (x, y, S, PL, info) => {
                        const L = S._simLay; if (!L) return ''; const cc = Math.floor((info.dx - L.x0) / L.cw), r = Math.floor((info.dy - L.y0) / L.ch);
                        if (r < 0 || cc < 0 || r >= L.R || cc >= L.C) return ''; const C = P03.LEX.content; return `${P03.LEX.words[C[r]]} · ${P03.LEX.words[C[cc]]}：cos = ${S.w2vView.sim[r * L.C + cc].toFixed(3)}`;
                    },
                },
                {
                    id: 'nnBars', live: true, label: '查询词的最近邻', sub: '在全部 47 个词里找 · 颜色 = 词类', size: [360, 230],
                    render: (c, S) => {
                        const v = S.w2vView, W = P03.LEX.words;
                        LM.draw.bars(c, v.nn.map((t) => t[1]), v.nn.map((t) => W[t[0]]), { horizontal: true, range: [Math.min(0, ...v.nn.map((t) => t[1])), 1], colors: v.nn.map((t) => P03.CATCOL[P03.LEX.cat[t[0]]]), digits: 3, title: `        「${W[S.query]}」的最近邻（窗口 ${S.win}）` });
                    },
                },
                {
                    id: 'anaAcc', live: true, label: '类比准确率 vs 窗口大小', sub: '164 道题：a : rel(a) = b : ？（rel = 食物 / 住处 / 颜色；答案相同的题已排除）· 6 个模型实时评估', size: [360, 230],
                    render: (c, S) => {
                        const xs = P03.W2V.wins, accs = S.w2v.map((M) => M.acc), cur = accs[P03.W2V.wins.indexOf(S.win)];
                        PL.draw.plot(c, {
                            x: [0.7, 6.3], y: [0, 1.55], xLabel: '窗口大小（左右各几个词）',
                            series: [{ data: accs.map((a) => a.food), x: xs, color: LM.C.green, label: '动物 → 食物' }, { data: accs.map((a) => a.home), x: xs, color: LM.C.orange, label: '动物 → 住处' }, { data: accs.map((a) => a.color), x: xs, color: LM.C.purple, label: '动物 → 颜色' }, { data: accs.map((a) => a.all), x: xs, color: LM.C.text, width: 2.5, label: '全部' }],
                            marks: [{ x: S.win, label: `窗口 ${S.win}：${cur.ok}/${cur.n} = ${(cur.all * 100).toFixed(0)}%` }],
                        });
                    },
                },
                {
                    id: 'anaEx', live: true, label: '类比示例（所选窗口）', sub: '答案 = v(b) − v(a) + v(c) 的最近词（排除 a、b、c）', size: [360, 230],
                    render: (c, S) => {
                        const v = S.w2vView, W = P03.LEX.words;
                        LM.draw.text(c, [{ text: `        窗口 ${S.win}：v(b) − v(a) + v(c) 的最近词`, color: LM.C.gray }].concat(v.ex.map((e) => {
                            const ok = e.got === e.ans;
                            return [{ t: `${W[e.a]} : ${W[e.b]} = ${W[e.c]} : ？ → `, color: LM.C.text }, { t: W[e.got], color: ok ? LM.C.green : LM.C.red, bold: true }, { t: ok ? '  ✓' : `  ✗（应为 ${W[e.ans]}）`, color: ok ? LM.C.green : LM.C.dim }];
                        })), { px: 13, lineGap: 9 });
                    },
                },
            ],
            notes: '拖 <b>查询词</b>，或在 Step 06 的 PCA 图上点一个词。对比 <b>word2vec 窗口</b> = 2 和 6 时「小猫」的最近邻：前者是别的动物（同一种角色），后者是鱼、屋里、白色（同一个主题）。窗口 1 的准确率并不是 0：只看紧邻的词时，向量几乎只编码「词类」，偶尔也能蒙对。',
            warn: '「类比能成立」常被当作词向量有语义结构的证据，但它对数据和超参数很敏感（这里只改了一个窗口大小），在真实词向量上，标准类比测试也远不是全对，结果还很依赖评估方式（Levy & Goldberg 2014）。另外，现代 LLM 的 token embedding 只是第一层：一个词在句子里的意思主要由后面的注意力层根据上下文计算出来。',
        },
        // ─────────────────────────────── 08 置换等变 ───────────────────────────────
        {
            id: 'perm', title: '为什么需要位置信息：注意力对顺序不敏感', tag: 'POS',
            desc: 'embedding 只编码了「是哪个 token」，没有编码「在第几个」。而自注意力（第 04 篇）对每个位置做的事情完全一样：用 q·k 打分、softmax、加权求和——<strong>把输入打乱，输出只是跟着打乱</strong>（置换等变，permutation equivariance），每个 token 拿到的输出向量一模一样。下面用基座模型第 0 层<strong>真实的</strong>注意力权重（LayerNorm → Q/K/V → 注意力 → W_O，去掉因果掩码，即编码器式的双向注意力）计算「小猫喜欢吃鱼」和「鱼喜欢吃小猫」：没有位置信息时，两句话里「鱼」的输出<strong>逐位相同</strong>，模型根本分不清谁吃谁；加上位置向量后才不同。',
            formula: [
                'Attn(X) = softmax( (X W_Q)(X W_K)^T / √d_h ) · X W_V      // 每一行的计算方式完全相同',
                'Attn(Π X) = Π · Attn(X)      // Π 是置换矩阵：先打乱再算 = 先算再打乱',
                'X = E[ids] + P[0..T)      // 加上位置向量后，同一个 token 在不同位置的输入不同，输出也不同',
            ],
            code: {
                title: 'C++ · bidirAttention 与置换实验', src: `
                // 第 0 层的注意力（LayerNorm → Q/K/V → softmax → 加权和 → W_O），但不加因果掩码：每个位置都能看到所有位置
                Var bidirAttention(const Var& x, const Block& blk, int H) {
                    int T = x->val.rows;
                    Var h = layerNorm(x, blk.ln1g, blk.ln1b);
                    Var q = linear(h, blk.Wq, blk.bq), k = linear(h, blk.Wk, blk.bk), v = linear(h, blk.Wv, blk.bv);
                    return linear(attention(q, k, v, 1, T, H, false), blk.Wo, blk.bo);   // causal = false
                }

                // 按行重排：out[t] = X[perm[t]]
                Tensor permuteRows(const Tensor& X, const std::vector<int>& perm) {
                    Tensor out = zeros((int)perm.size(), X.cols);
                    for (int t = 0; t < (int)perm.size(); t++)
                        for (int j = 0; j < X.cols; j++) out.at(t, j) = X.at(perm[t], j);
                    return out;
                }

                // 逐元素 |a − b| ≤ atol
                bool allClose(const Tensor& a, const Tensor& b, float atol) {
                    for (int i = 0; i < a.size(); i++) if (std::fabs(a.v[i] - b.v[i]) > atol) return false;
                    return true;
                }

                void permutationDemo(const GPT& model, const Vocab& vocab) {
                    NoGrad ng;
                    const Block& blk = model.blocks[0];
                    int H = model.cfg.H;
                    std::vector<int> ids = encode(vocab, "小猫喜欢吃鱼");   // 字符级编码（Common · minml_gpt.h）
                    std::vector<int> perm = {5, 2, 3, 4, 0, 1};      // 打乱后第 t 个位置放原句的第 perm[t] 个字 →「鱼喜欢吃小猫」
                    std::vector<int> idsP = ints(6), pos = ints(6);
                    for (int t = 0; t < 6; t++) { idsP[t] = ids[perm[t]]; pos[t] = t; }
                    Var y = bidirAttention(embedding(ids, model.wte), blk, H);
                    Var yP = bidirAttention(embedding(idsP, model.wte), blk, H);
                    std::printf("%d\\n", allClose(permuteRows(y->val, perm), yP->val, 1e-5f));   // 1：输出只是跟着换了位置
                    Var P = gatherRows(model.wpe, pos);              // 位置向量 P[0..T)
                    Var z = bidirAttention(add(embedding(ids, model.wte), P), blk, H);
                    Var zP = bidirAttention(add(embedding(idsP, model.wte), P), blk, H);
                    std::printf("%d\\n", allClose(permuteRows(z->val, perm), zP->val, 1e-5f));   // 0：位置让顺序有了意义
                }`,
            },
            previews: [
                {
                    id: 'permNoPE', label: '只有 token embedding（无位置）', sub: '左 = 原句每个 token 的输出，右 = 打乱句 · 连线连接同一个 token · 绿 = 两次输出相同', large: true, size: [360, 220],
                    render: (c, S) => P03UI.drawPerm(c, S.perm, S.perm.noPE, 'X = E[ids]：顺序信息完全丢失'),
                },
                {
                    id: 'permPE', label: '加上位置向量 wpe', sub: '同样的 token、同样的权重，只多了 + P[位置]', size: [360, 220],
                    render: (c, S) => P03UI.drawPerm(c, S.perm, S.perm.withPE, 'X = E[ids] + P：同一个 token 的输出变了'),
                },
            ],
            notes: '<b>打乱方式</b> 换成随机排列，结论一样：没有位置信息时，每个 token 的输出差异都在 10⁻⁸ 量级——那只是 float32 求和顺序不同带来的舍入误差；有位置信息时差异在 0.1 左右，比舍入误差大六七个数量级——即使某个 token 自己的位置没变（如随机排列 A 里的「鱼」），别的 token 换了位置也会改变它的输出。',
            warn: '这个对称性只对<strong>双向</strong>注意力严格成立。GPT 的因果掩码（每个位置只能看它前面的 token）本身就打破了对称性，所以完全不加位置编码的 decoder 也能学到一些位置信息（Haviv 等 2022）；但显式的位置编码仍是标配。',
        },
        // ─────────────────────────────── 09 正弦位置编码 ───────────────────────────────
        {
            id: 'sinpe', title: '正弦位置编码：每个位置一组不同频率的波', tag: 'POS',
            desc: '原始 Transformer（Vaswani 等 2017）给位置 pos 生成一个 d 维向量，第 2i、2i+1 维是频率 ω<sub>i</sub> = base<sup>−2i/d</sup> 的正弦和余弦，直接<strong>加</strong>到 token embedding 上。低维度变化快（第 0 对的波长约 6 个位置），高维度变化极慢（最后一对的波长有几万个位置）——像一组转速不同的钟表指针，合起来能唯一确定位置。它有两个好性质：不用学习、任意位置都能算；两个位置的点积 PE(i)·PE(j) = Σ cos((i−j)ω<sub>k</sub>) <strong>只取决于距离 i − j</strong>（右上：三个相距很远的起点，曲线完全重合），并且大体随距离增大而下降。GPT-2 和本系列的迷你 GPT 改用<strong>可学习</strong>的位置向量（wpe，又一张 [T × d] 的查表矩阵）；下面比较两者的位置相似度矩阵：学出来的 wpe 也有一条「相邻位置最像」的对角线，此外还分成「前半句 / 后半句」两大块。',
            formula: [
                'PE(pos, 2i) = sin(pos · ω_i)      PE(pos, 2i+1) = cos(pos · ω_i)      ω_i = base^{−2i/d}',
                'PE(i) · PE(j) = Σ_{k=0}^{d/2−1} cos((i − j) · ω_k)      // sin a·sin b + cos a·cos b = cos(a − b)',
                'PE(pos + Δ) = R(Δ) · PE(pos)      // R(Δ) 由 2×2 旋转块组成，与 pos 无关：相对位移是一个线性变换',
            ],
            code: {
                title: 'C++ · sinusoidalPE', src: `
                // 正弦位置编码 [T × d]：PE[pos, 2i] = sin(pos·ω_i)，PE[pos, 2i+1] = cos(pos·ω_i)，ω_i = base^(−2i/d)
                Tensor sinusoidalPE(int T, int d, float base) {
                    Tensor pe = zeros(T, d);
                    for (int pos = 0; pos < T; pos++)
                        for (int i = 0; i < d / 2; i++) {
                            double a = pos * std::pow((double)base, -2.0 * i / d);   // pos · ω_i
                            pe.at(pos, 2 * i) = (float)std::sin(a);
                            pe.at(pos, 2 * i + 1) = (float)std::cos(a);
                        }
                    return pe;
                }

                void peDemo() {
                    Tensor pe = sinusoidalPE(512, 32, 10000.f);
                    std::printf("%f %f\\n", dotRow(pe, 40, pe, 45), dotRow(pe, 400, pe, 405));   // 相等：点积只依赖距离 5
                }`,
            },
            previews: [
                {
                    id: 'peHeat', label: '正弦位置编码 PE [64 × 32]', sub: '行 = 位置 pos，列 = 维度（偶数列 sin、奇数列 cos）· 左边转得快，右边几乎不动', large: true, size: [360, 300],
                    render: (c, S) => {
                        const P = S.pe;
                        const lay = LM.draw.heat(c, P.pe, P.T, P.d, { range: [-1, 1], map: 'diverge', square: false, legend: false, cellMax: 9.4, rowLabels: Array.from({ length: P.T }, (_, i) => (i % 12 === 0 ? String(i) : '')), colLabels: Array.from({ length: P.d }, (_, j) => (j % 8 === 0 ? String(j) : '')) });
                        S._peLay = lay;
                        const { ctx } = PL.draw.ctx(c, { clear: false });
                        P03.colorBar(ctx, lay.x0 + lay.C * lay.cw + 7, lay.y0, lay.R * lay.ch * 0.6, [-1, 1]);
                    },
                    probe: (x, y, S, PL, info) => {
                        const L = S._peLay; if (!L) return ''; const cc = Math.floor((info.dx - L.x0) / L.cw), r = Math.floor((info.dy - L.y0) / L.ch);
                        if (r < 0 || cc < 0 || r >= L.R || cc >= L.C) return ''; const i = cc >> 1, om = Math.pow(S.pe.B, -2 * i / S.pe.d);
                        return `pos ${r}，第 ${cc} 维（${cc % 2 ? 'cos' : 'sin'}，ω = ${om.toPrecision(3)}，波长 ${(2 * Math.PI / om).toPrecision(3)}）= ${S.pe.pe[r * L.C + cc].toFixed(3)}`;
                    },
                },
                {
                    id: 'peDot', label: 'PE(i) · PE(i + Δ)', sub: '三个起点 i 的曲线互相重合（粗、虚、点线叠在一起）· 只依赖相对距离 Δ', size: [360, 300],
                    render: (c, S) => {
                        const P = S.pe, xs = Array.from({ length: 2 * P.D + 1 }, (_, i) => i - P.D);
                        PL.draw.plot(c, {
                            x: [-P.D, P.D], xLabel: '相对距离 Δ', series: P.starts.map((i0, q) => ({ data: P.dots[q], x: xs, color: [LM.C.blue, LM.C.orange, LM.C.green][q], width: [5, 2.5, 1.5][q], dash: [null, [7, 4], [2, 3]][q], label: `i = ${i0}` })),
                            hmarks: [{ y: P.d / 2, color: '#555' }],
                        });
                        const { ctx, h } = PL.draw.ctx(c, { clear: false });
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.green; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; void h;
                        ctx.fillText(`三条曲线的最大差 ${P.maxDiff.toExponential(0)} · 灰线 = d/2 = ${P.d / 2}`, 56, 22);
                    },
                },
                {
                    id: 'peSim', label: '位置之间的余弦相似度：正弦 vs 学出来的 wpe', sub: '左 = 正弦编码（前 32 个位置），右 = 基座模型训练后的 wpe [32 × 32] · 两者都有「相邻位置最像」的对角线', size: [720, 270],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c, {});
                        const n = 32, cell = 6.6, y0 = 30, xa = 40, xb = xa + n * cell + 110, labs = Array.from({ length: n }, (_, i) => i);
                        P03.heatAt(ctx, xa, y0, cell, cell, S.pe.simSin, n, n, { range: [-1, 1], title: `正弦编码（base ${S.pe.B}，d = 32）`, rowLabels: labs, labelEvery: 8 });
                        P03.heatAt(ctx, xb, y0, cell, cell, S.wpeSim, n, n, { range: [-1, 1], title: '基座模型学到的 wpe（d = 32）', rowLabels: labs, labelEvery: 8 });
                        P03.colorBar(ctx, xb + n * cell + 16, y0, n * cell, [-1, 1]);
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.dim; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                        [xa, xb].forEach((x0) => ctx.fillText('位置 j →（行 = 位置 i）', x0 + n * cell / 2, y0 + n * cell + 6));
                    },
                },
            ],
            notes: '<b>频率底数 base</b> 改成 100：高维度也开始转，热图右半边不再是平的，点积曲线也随距离更快地振荡。学到的 wpe 有两层结构：一条细的对角线（相邻位置最像），加上前半句 / 后半句两大块、彼此负相关——模型用一个方向粗略编码「靠前 / 靠后」（Step 11 里 P 的热图上下两端颜色相反）。正弦编码（base 10 000）的前 32 个位置彼此都很像：大多数维度在这么短的距离里几乎不转。',
            warn: '正弦编码是<strong>加</strong>在输入上的：经过 W_Q、W_K 的线性变换之后，q·k 里混进了「内容 × 位置」的交叉项，「只依赖相对位置」不再精确成立——这正是下一步 RoPE 要解决的。本页 d = 32（与迷你 GPT 一致）；原论文 d_model = 512、base = 10 000。',
        },
        // ─────────────────────────────── 10 RoPE ───────────────────────────────
        {
            id: 'rope', title: 'RoPE：把位置变成旋转', tag: 'POS',
            desc: '<strong>RoPE</strong>（Rotary Position Embedding，Su 等 2021）不把位置加到输入上，而是在每个注意力层里，把 q 和 k 的第 (2i, 2i+1) 维当作一个二维向量，按位置<strong>旋转</strong> pos·θ<sub>i</sub>（θ<sub>i</sub> 与正弦编码的频率相同）。旋转不改变长度，两个旋转后的向量的点积只取决于<strong>转角之差</strong>：(R<sub>m</sub>q)·(R<sub>n</sub>k) = q·R<sub>n−m</sub>k——注意力分数天然只依赖相对位置 n − m，与绝对位置无关。左上：同一对 q、k 放在绝对位置 40、200、1000，分数随相对距离的曲线<strong>完全重合</strong>；右上对照「把正弦编码加到 q、k 上」，三条曲线各不相同。LLaMA、Mistral、Qwen 等主流开源模型都用 RoPE。',
            formula: [
                '[q′_{2i}, q′_{2i+1}] = [[cos mθ_i, −sin mθ_i], [sin mθ_i, cos mθ_i]] · [q_{2i}, q_{2i+1}]      θ_i = base^{−2i/d}',
                '(R_m q) · (R_n k) = q^T R_m^T R_n k = q^T R_{n−m} k      // 旋转矩阵 R_m^T R_n = R_{n−m}',
                'score(m, n) = Σ_i Re[ q̃_i · conj(k̃_i) · e^{i(m−n)θ_i} ]      // 复数写法：q̃_i = q_{2i} + i·q_{2i+1}',
            ],
            code: {
                title: 'C++ · applyRope / ropeScore', src: `
                // 把向量 x（一个头的 q 或 k，长度 d）旋转到位置 m：第 (2i, 2i+1) 维当作一个二维向量，转 m·θ_i，θ_i = base^(−2i/d)
                std::vector<float> applyRope(const std::vector<float>& x, float m, float base) {
                    int d = (int)x.size();
                    std::vector<float> out = floats(d);
                    for (int i = 0; i < d / 2; i++) {
                        double a = m * std::pow((double)base, -2.0 * i / d), c = std::cos(a), s = std::sin(a);
                        out[2 * i] = (float)(x[2 * i] * c - x[2 * i + 1] * s);
                        out[2 * i + 1] = (float)(x[2 * i] * s + x[2 * i + 1] * c);
                    }
                    return out;
                }

                // 位置 m 的 q 与位置 n 的 k 的注意力分数：各自旋转后再做点积
                float ropeScore(const std::vector<float>& q, const std::vector<float>& k, int m, int n, float base) {
                    std::vector<float> qm = applyRope(q, (float)m, base), kn = applyRope(k, (float)n, base);
                    double s = 0;
                    for (size_t i = 0; i < qm.size(); i++) s += qm[i] * kn[i];
                    return (float)s;
                }

                // 注意力层里：Q、K [T × D]（H 个头并排，每头 dh = D/H 维）的第 t 行转到位置 t，每个头各转各的；V 不变
                void ropeQK(Tensor& Q, Tensor& K, int H, float base) {
                    int dh = Q.cols / H;
                    for (int t = 0; t < Q.rows; t++)
                        for (int h = 0; h < H; h++) {
                            std::vector<float> q = floats(dh), k = floats(dh);
                            for (int e = 0; e < dh; e++) { q[e] = Q.at(t, h * dh + e); k[e] = K.at(t, h * dh + e); }
                            q = applyRope(q, (float)t, base);
                            k = applyRope(k, (float)t, base);
                            for (int e = 0; e < dh; e++) { Q.at(t, h * dh + e) = q[e]; K.at(t, h * dh + e) = k[e]; }
                        }
                }

                void ropeDemo() {
                    Rng r = makeRng(4242);                           // 与页面相同的一对随机 q、k（d = 32）
                    std::vector<float> q = floats(32), k = floats(32);
                    for (float& x : q) x = randn(r);
                    for (float& x : k) x = randn(r);
                    std::printf("%f %f\\n", ropeScore(q, k, 40, 45, 10000.f), ropeScore(q, k, 1000, 1005, 10000.f));   // 相等：只依赖 n − m = 5
                }`,
            },
            previews: [
                {
                    id: 'ropeScore', label: 'RoPE：(R_m q)·(R_{m+Δ} k)', sub: '同一对随机 q、k（d = 32）放在三个绝对位置 m · 三条曲线完全重合', large: true, size: [360, 240],
                    render: (c, S) => {
                        const D = S.ropeD, xs = Array.from({ length: 2 * D.D + 1 }, (_, i) => i - D.D);
                        PL.draw.plot(c, { x: [-D.D, D.D], y: D.yr, xLabel: '相对位置 Δ = n − m', series: D.ms.map((m, q) => ({ data: D.rope[q], x: xs, color: [LM.C.blue, LM.C.orange, LM.C.green][q], width: [5, 2.5, 1.5][q], dash: [null, [7, 4], [2, 3]][q], label: `m = ${m}` })), hmarks: [{ y: 0, color: '#333' }] });
                        const { ctx, h } = PL.draw.ctx(c, { clear: false });
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.green; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; void h; ctx.fillText(`三条曲线的最大差 ${D.maxRope.toExponential(1)}`, 56, 14);
                    },
                },
                {
                    id: 'absScore', label: '对照：(q + PE(m))·(k + PE(m+Δ))', sub: '把正弦编码加到 q、k 上 · 同样的三个绝对位置，曲线各不相同', size: [360, 240],
                    render: (c, S) => {
                        const D = S.ropeD, xs = Array.from({ length: 2 * D.D + 1 }, (_, i) => i - D.D);
                        PL.draw.plot(c, { x: [-D.D, D.D], y: D.yr, xLabel: '相对位置 Δ = n − m', series: D.ms.map((m, q) => ({ data: D.abs[q], x: xs, color: [LM.C.blue, LM.C.orange, LM.C.green][q], width: [3, 2, 1.5][q], dash: [null, [7, 4], [2, 3]][q], label: `m = ${m}` })), hmarks: [{ y: 0, color: '#333' }] });
                        const { ctx, h } = PL.draw.ctx(c, { clear: false });
                        ctx.font = P03.FONT(11); ctx.fillStyle = LM.C.red; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; void h; ctx.fillText(`三条曲线的最大差 ${D.maxAbs.toFixed(2)}`, 56, 14);
                    },
                },
                {
                    id: 'ropeRot', live: true, label: '第 i 对维度的旋转（q 的分量）', sub: '灰 = 旋转前的 (q_{2i}, q_{2i+1}) 方向，彩色 = 位置 m 处旋转后 · 位置 m 随时间递增', size: [360, 200],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c, {});
                        const D = S.ropeD, m = S.ropeM, d = D.d, pairs = [0, 3, 7, 15], cols = [LM.C.red, LM.C.orange, LM.C.blue, LM.C.green], R = 33, cy = 84;
                        ctx.font = P03.FONT(12, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`位置 m = ${m}（每秒 +3）`, 46, 6);
                        pairs.forEach((i, k) => {
                            const cx = 46 + k * 89, th = Math.pow(D.B, -2 * i / d), a0 = Math.atan2(D.q[2 * i + 1], D.q[2 * i]), ang = m * th;
                            ctx.strokeStyle = '#2a2f36'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(cx - R - 4, cy); ctx.lineTo(cx + R + 4, cy); ctx.moveTo(cx, cy - R - 4); ctx.lineTo(cx, cy + R + 4); ctx.stroke();
                            const turn = ang % (2 * Math.PI);
                            ctx.strokeStyle = cols[k]; ctx.globalAlpha = 0.5; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, R * 0.45, -a0, -(a0 + turn), true); ctx.stroke(); ctx.globalAlpha = 1;
                            ctx.strokeStyle = '#6a7078'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(a0), cy - R * Math.sin(a0)); ctx.stroke();
                            ctx.strokeStyle = cols[k]; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(a0 + ang), cy - R * Math.sin(a0 + ang)); ctx.stroke();
                            ctx.fillStyle = cols[k]; ctx.beginPath(); ctx.arc(cx + R * Math.cos(a0 + ang), cy - R * Math.sin(a0 + ang), 3, 0, 7); ctx.fill();
                            ctx.font = P03.FONT(11); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = cols[k];
                            ctx.fillText(`i = ${i}`, cx, cy + R + 8);
                            ctx.fillStyle = LM.C.gray; ctx.fillText(`θ = ${th < 0.01 ? th.toExponential(1) : th.toFixed(3)}`, cx, cy + R + 23);
                            ctx.fillText(`转 ${(ang / (2 * Math.PI)).toFixed(ang < 2 * Math.PI ? 3 : 1)} 圈`, cx, cy + R + 38);
                        });
                    },
                },
                {
                    id: 'ropeWave', label: '每对维度转一圈要多少个位置', sub: '波长 2π/θ_i（对数纵轴）· 粗线 = 顶栏选中的 base · 灰线 = 几个模型的上下文长度', size: [360, 200],
                    render: (c, S) => {
                        const D = S.ropeD, xs = Array.from({ length: D.d / 2 }, (_, i) => i);
                        PL.draw.plot(c, {
                            x: [0, D.d / 2 - 1], logY: true, y: [1, 1e7], xLabel: '维度对 i（d = 32）',
                            series: [100, 10000, 500000].map((b, q) => ({ data: D.wave[b], x: xs, color: [LM.C.red, LM.C.blue, LM.C.green][q], width: b === D.B ? 2.8 : 1, dash: b === D.B ? null : [4, 3], label: `base ${P03UI.fmtN(b)}` })),
                            hmarks: [{ y: 32, label: '32（迷你 GPT 的上下文）', color: '#666' }, { y: 4096, label: '4096（LLaMA-2）', color: '#666' }, { y: 131072, label: '128K（LLaMA-3.1）', color: '#666' }],
                        });
                    },
                },
            ],
            notes: '左下的动画里位置 m 在递增：第 0 对维度每个位置转 1 弧度（约 6 个位置一圈），第 15 对几乎不动。<b>频率底数 base</b>：LLaMA-2 用 10 000，LLaMA-3 把它增大到 500 000——低频维度转得更慢，几十万个位置才转一圈，长上下文里远处的位置仍然可以区分。',
            warn: [
                '两种配对方式：原论文和本页按相邻维度 (2i, 2i+1) 配对；HuggingFace 的 LLaMA 实现用 <code>rotate_half</code>，把第 i 维和第 i + d/2 维配对。两者只差一个固定的维度重排，但<strong>权重和实现不能混用</strong>。',
                'RoPE 只作用在每个头的 q、k 上，不作用在 v 上，也不加到残差流里；所以模型的输入层就只剩 token embedding（没有 wpe）。',
            ],
        },
        // ─────────────────────────────── 11 输入层 ───────────────────────────────
        {
            id: 'input', title: '输入层：X = E[ids] + P', tag: 'INPUT',
            desc: '把这一篇的东西拼起来，就是 GPT 类模型的输入层：文本 → tokenizer → ids → 查 token embedding → 加上位置向量 → <strong>X ∈ ℝ<sup>T×d</sup></strong>，交给第一个 Transformer Block。下面是基座模型对所选句子的真实输入，每行一个位置：token 向量（左）决定「是什么」，位置向量（中）决定「在哪里」，两者逐元素相加（右）。之后的每一层都在这个「残差流」上读写（第 04 篇）。',
            formula: [
                'x_t = E[id_t] + P[t]      t = 0, 1, …, T − 1      // E：[76 × 32]，P：[32 × 32]（可学习的 wpe）',
                'X = [x_0; x_1; …; x_{T−1}] ∈ ℝ^{T × d}      // T ≤ 32：迷你 GPT 的上下文长度',
            ],
            code: {
                title: 'C++ · 输入层（gptForward 的第一行）', src: `
                // 迷你 GPT 的输入层：X = E[ids] + P[0..T)（Common · minml_gpt.h 的 gptForward 里就是这一行）
                Var inputLayer(const GPT& model, const std::vector<int>& ids) {
                    int T = (int)ids.size();                         // T ≤ 32
                    std::vector<int> pos = ints(T);
                    for (int t = 0; t < T; t++) pos[t] = t;          // 0, 1, …, T−1
                    Var tokEmb = embedding(ids, model.wte);          // [T × 32]：查表
                    Var posEmb = gatherRows(model.wpe, pos);         // [T × 32]：第 t 行 = 位置 t 的向量
                    return add(tokEmb, posEmb);                      // 逐元素相加 → [T × 32]，交给 blockForward(model.blocks[0], …)
                }
                // 用法：Var X = inputLayer(model, encode(vocab, sentence));   // encode = 字符级编码（minml_gpt.h）`,
            },
            previews: [
                {
                    id: 'inputX', label: '基座模型的输入 X（所选句子）', sub: '蓝负橙正 · 行 = 位置 / token · 列 = 32 个维度 · P 的数值小得多，单独用 ±0.3 的色标才看得清（相加时它只是一个小偏移）', large: true, size: [720, 300],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, {});
                        const I = S.inp, T = I.T, d = I.d, cw = 5.6, ch = Math.min(14, (h - 62) / T), y0 = 30, rng = [-1, 1], gap = 52, x0 = 36;
                        const xs = [x0, x0 + d * cw + gap, x0 + 2 * (d * cw + gap)];
                        P03.heatAt(ctx, xs[0], y0, cw, ch, I.E, T, d, { range: rng, rowLabels: [...I.s], title: 'E[ids] · token 向量（色标 ±1）' });
                        P03.heatAt(ctx, xs[1], y0, cw, ch, I.P, T, d, { range: [-0.3, 0.3], rowLabels: Array.from({ length: T }, (_, t) => t), title: 'P[t] · 位置向量（色标 ±0.3）', labelEvery: T > 16 ? 2 : 1 });
                        P03.heatAt(ctx, xs[2], y0, cw, ch, I.X, T, d, { range: rng, title: 'X = E[ids] + P（色标 ±1）' });
                        P03.colorBar(ctx, xs[2] + d * cw + 10, y0, Math.max(40, T * ch * 0.6), rng);
                        ctx.font = P03.FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = I.ok ? LM.C.gray : LM.C.orange;
                        ctx.fillText(I.ok ? `「${I.s}」：T = ${T} 个 token → X 的形状 [${T} × ${d}]，送进第 04 篇的第一个 Block` : `所选句子里有字符级词表之外的字（迷你 GPT 无法编码），这里改用第一句「${I.s}」`, 6, y0 + T * ch + 10);
                    },
                },
                {
                    id: 'inputNorm', label: '每个位置的向量长度', sub: '‖E[id_t]‖、‖P[t]‖、‖x_t‖ · 位置向量比 token 向量小得多', size: [360, 210],
                    render: (c, S) => {
                        const I = S.inp, xs = Array.from({ length: I.T }, (_, t) => t);
                        PL.draw.plot(c, { x: [0, Math.max(1, I.T - 1)], y: [0, Math.max(...I.nX, ...I.nE) * 1.6], xLabel: '位置 t', series: [{ data: I.nE, x: xs, color: LM.C.blue, label: '‖E[id_t]‖ token' }, { data: I.nP, x: xs, color: LM.C.orange, label: '‖P[t]‖ 位置' }, { data: I.nX, x: xs, color: LM.C.green, label: '‖x_t‖ 相加后', dash: [5, 3], width: 2 }] });
                    },
                },
            ],
            notes: '切换 <b>示例句子</b>（含词表外字的句子会改用第一句）。位置向量的范数只有 0.1–0.55（32 个位置里两端大、中间小），token 向量约 0.7–1.9：「是什么」占主导，位置只是一个较小的偏移。P 的热图上下两端颜色相反——学到的 wpe 用一个大致单调变化的方向编码「靠前 / 靠后」（与 Step 09 相似度矩阵里的两大块一致）。',
            warn: '迷你 GPT 用可学习的 wpe，最多只能处理 T = 32 个位置——第 33 个位置根本没有向量，这是 learned 位置编码的硬上限。正弦编码和 RoPE 没有这张表，任何位置都能算，但训练长度之外的位置同样会退化，需要专门的外推方法（下一步）。',
        },
        // ─────────────────────────────── 12 生产版 ───────────────────────────────
        {
            id: 'prod', title: '教学版 → 生产版：真实的 tokenizer 与位置编码', tag: 'NEXT',
            desc: '本页的 BPE 与 GPT-2 的算法相同（字节级 + 同一个预切分正则 + 按学习顺序合并），只是语料和词表小几个数量级；word2vec 展示的「从上下文学向量」在 LLM 里由整个网络端到端完成，E 只是第一层。真实系统还有下面这些要点：',
            blocks: [
                {
                    type: 'html', html: `<table class="lm-tbl"><tr><th>模型 / 工具</th><th>分词算法</th><th>词表大小</th><th>说明</th></tr>
                        <tr><td>GPT-2（2019）</td><td>字节级 BPE</td><td>50 257</td><td>= 256 个字节 + 50 000 次合并 + 1 个 &lt;|endoftext|&gt;</td></tr>
                        <tr><td>GPT-3.5 / GPT-4（tiktoken cl100k_base）</td><td>字节级 BPE</td><td>约 10 万</td><td>预切分正则也改了，例如把数字按最多 3 位一组切开</td></tr>
                        <tr><td>BERT</td><td>WordPiece</td><td>30 522</td><td>[CLS] [SEP] [MASK] 等特殊 token；learned 位置编码，最长 512</td></tr>
                        <tr><td>LLaMA / LLaMA-2</td><td>SentencePiece BPE + byte fallback</td><td>32 000</td><td>按 Unicode 字符起步，词表外的罕见字符退回 256 个字节 token</td></tr>
                        <tr><td>LLaMA-3</td><td>字节级 BPE（tiktoken）</td><td>128 256</td><td>词表变大 → 同样的文本更少 token；RoPE base = 500 000</td></tr></table>`,
                },
                {
                    type: 'html', html: `<ul>
                        <li><b>训练与实现</b>：真实 tokenizer 在 GB 级、多语言 + 代码的语料上训练；常用 HuggingFace <code>tokenizers</code>（Rust）或 SentencePiece。推理时 tiktoken 等实现按 rank 合并，切分结果与本页的 <code>bpeEncode</code> 相同，只是快几个数量级。</li>
                        <li><b>语言比例决定 token 效率</b>：tokenizer 训练语料里某种语言越少，它在这种语言上的 token 就越碎——同一句话要更多 token，意味着更贵、上下文更短。下图是本页玩具实验的真实结果。</li>
                        <li><b>特殊 token 与对话模板</b>：<code>&lt;|endoftext|&gt;</code>（GPT-2 的 id 50256）、ChatML 的 <code>&lt;|im_start|&gt;</code> / <code>&lt;|im_end|&gt;</code>、LLaMA-3 的 <code>&lt;|begin_of_text|&gt;</code> <code>&lt;|start_header_id|&gt;</code> <code>&lt;|eot_id|&gt;</code>。它们不参与 BPE 合并，直接映射到固定 id；用户输入里出现同样的字符串时必须当普通文本编码，否则就是一种注入（第 06 篇的 SFT 模板）。</li>
                        <li><b>权重共享</b>：GPT-2 和本系列让输入 embedding 与输出层共用 E（Press &amp; Wolf 2017）；LLaMA 系列的大模型输入、输出各一份。</li>
                        <li><b>位置编码的演进</b>：正弦（2017）→ learned（GPT-2、BERT）→ 相对位置偏置（T5）、ALiBi（BLOOM）→ RoPE（LLaMA、Mistral、Qwen…）。</li>
                        <li><b>长上下文外推</b>：训练长度之外，RoPE 的低频维度会转到训练时没见过的角度。位置插值（PI，把位置按比例压缩）、NTK-aware 缩放、YaRN（按频率分段缩放）只需少量微调就能把上下文扩展数倍；LLaMA-3.1 支持 128K 上下文。</li>
                    </ul>`,
                },
                {
                    type: 'code', title: 'C++ · 载入 GPT-2 的真实规则 与 RoPE 位置插值', src: `
                    // 读入 GPT-2 公开的 50 000 条合并规则（merges.txt，按 rank 排列），得到与本页相同的 BPE 结构（解析文件的实现略）。
                    // 之后的编码就是 GPT-2 官方 tokenizer 的算法；只是 GPT-2 给 256 个单字节 token 另排了 id，编号和这里不一样
                    BPE loadGPT2Merges(const std::string& path);

                    void gpt2Demo() {
                        BPE gpt2 = loadGPT2Merges("gpt2/merges.txt");    // 词表 = 256 个字节 + 50 000 条规则（+ 1 个 <|endoftext|>）
                        std::string s = "The cat likes fish. 小猫喜欢吃鱼。";
                        std::vector<int> ids = bpeEncode(s, gpt2, (int)gpt2.merges.size());   // 同一个预切分正则、同样按 rank 合并（Step 02–04）
                        for (int id : ids) std::printf("[%s] ", utf8Replace(gpt2.vocab[id]).c_str());   // 汉字常被切成「半个字」的字节 token（显示成 �）
                        assert(bpeDecode(ids, gpt2) == s);               // 无损
                    }

                    // 位置插值（PI）：位置压缩 scale 倍（m → m / scale），让 scale 倍长的序列落回训练时见过的角度范围
                    std::vector<float> applyRopePI(const std::vector<float>& x, int m, float base, float scale) {
                        return applyRope(x, m / scale, base);
                    }`,
                },
                {
                    type: 'previews', items: [
                        {
                            id: 'mixCurve', needs: ['mixSweep'], label: 'tokenizer 训练语料的语言比例 → 压缩率', sub: '5 个 tokenizer 都用同样多的规则（顶栏的 k）· 各自在中、英留出文本上的字符 / token', large: true, size: [360, 240],
                            render: (c, S) => {
                                const M = S.mixSweep, xs = M.map((r) => r.mix * 100);
                                PL.draw.plot(c, {
                                    x: [0, 90], y: [0, 1.4 * Math.max(...M.map((r) => Math.max(r.zh, r.en)))], xLabel: 'BPE 训练语料里的英文比例（%）', yLabel: '字符 / token',
                                    series: [{ data: M.map((r) => r.zh), x: xs, color: LM.C.green, label: '留出中文', width: 2.5 }, { data: M.map((r) => r.en), x: xs, color: LM.C.orange, label: '留出英文', width: 2.5 }],
                                    marks: [{ x: S.enMix * 100, label: `当前 ${Math.round(S.enMix * 100)}%（k = ${S.bpeK.k}）` }],
                                });
                            },
                        },
                        {
                            id: 'mixTable', needs: ['mixSweep'], label: '同一个玩具世界的一句话要几个 token', sub: '每句平均 token 数（中文句与对应的英文镜像句表达同样的事实）', size: [360, 240],
                            render: (c, S) => {
                                const M = S.mixSweep;
                                LM.draw.text(c, [{ text: `k = ${S.bpeK.k} 条规则时，每句平均 token 数`, color: LM.C.gray }].concat(M.map((r) => [
                                    { t: `英文 ${String(Math.round(r.mix * 100)).padStart(2, ' ')}%   `, color: r.mix === S.enMix ? LM.C.yellow : LM.C.dim },
                                    { t: `中文 ${r.zhT.toFixed(1).padStart(5, ' ')}`, color: LM.C.green }, { t: '    ', color: LM.C.dim },
                                    { t: `英文 ${r.enT.toFixed(1).padStart(5, ' ')}`, color: LM.C.orange },
                                ])), { px: 13, lineGap: 10 });
                            },
                        },
                    ],
                },
                { type: 'note', html: '英文比例从 0% 调到 90%：英文留出文本的压缩率大幅上升，中文的下降——两种语言在<strong>争夺同一份规则预算</strong>。真实的多语言 tokenizer 也面临同样的取舍；词表越大，越能同时照顾多种语言（这也是 LLaMA-3 把词表从 32 000 扩到 128 256 的原因之一）。下一篇（04）把 Step 11 的 X 送进 Transformer，逐层看一次前向传播。' },
            ],
        },
    ],

    graph: {
        title: '文本 → 向量 · 数据流',
        intro: '每个节点 = 一次真实的计算（这里都在 CPU 上用 JS 跑），pin 是张量 / 数据名 · 形状。三条数据流：tokenizer（pretokenize → trainBPE → bpeEncode → bpeDecode）；迷你 GPT 的输入层（char_encode → embedding_gather → add_position，以及置换实验和位置编码）；word2vec 的训练循环（segmentWords → skipgramPairs → sgnsUpdate）——<code>W2V</code> 是 <b>HISTORY</b> 资源，每一帧读取上一帧更新后的词向量（虚线回边）。',
        types: {
            text: { color: '#e0a060', label: 'text · 字符串 / 预切分块' },
            ids: { color: '#e8c25a', label: 'ids · token id 序列' },
            table: { color: '#c678dd', label: 'table · 合并规则 / 词表' },
            param: { color: '#5ad07a', label: 'param · 可训练参数' },
            act: { color: '#6ab0e0', label: 'act · 向量 / 激活' },
            scalar: { color: '#8a8f98', label: 'scalar · 超参数' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用函数', open: false,
            blocks: LM.CPP.blocks(['gpt']).concat([
                { label: '━ 本页公用 · UTF-8 解码与 GPT-2 预切分 ━', code: P03CPP.utf8 },
                { label: '━ 本页公用 · BPE 基础：去重计数、相邻对计数、合并 ━', code: P03CPP.bpeCore },
                { label: '━ 本页公用 · word2vec：词典分词、训练对、负采样表 ━', code: P03CPP.w2vCore },
                { label: '━ 本页公用 · 基座模型（本篇只用 wte / wpe / 第 0 层注意力）与向量小工具 ━', code: P03CPP.misc },
            ]),
        },
        nodes: [
            {
                id: 'corpus', title: 'world.corpus', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'bpe', desc: '中文 2000 句（+ 英文镜像）· word2vec 3000 句',
                dispatch: { kind: 'CPU', name: 'LM.world.sentence', count: (S) => `BPE ${P03.NSENT} 句（英文 ${Math.round(S.enMix * 100)}%）· 留出 ${P03.NHELD}+${P03.NHELD} 句 · word2vec ${P03.W2V.nSent} 句` },
                outputs: [{ name: 'Corpus', format: 'str · 中 + 英', type: 'text' }, { name: 'CorpusZh', format: 'str · 中文', type: 'text' }],
                detail: { desc: '8 种动物 × 3 个属性的玩具世界，10 个中文句式（与第 05 篇预训练语料同一个生成器）；英文镜像句表达同样的事实。', preview: 'corpusToks' },
            },
            {
                id: 'hyper', title: 'HyperParams', tag: 'CB', category: 'user', desc: 'k · 英文比例 · 窗口 · K · base',
                outputs: [{ name: 'Hyper', format: 'scalars', type: 'scalar' }],
                detail: { desc: '用户控制的超参数：合并次数 k、BPE 语料的英文比例、word2vec 的窗口与负样本数、位置编码的频率底数。', preview: null },
            },
            {
                id: 'sentence', title: 'Sentence', tag: 'CB', category: 'user', desc: '示例句子',
                outputs: [{ name: 'Sentence', format: 'str', type: 'text' }],
                detail: { desc: '顶栏选择的示例句子（Step 01、04、11）。', preview: 'bytesView' },
            },
            {
                id: 'pretok', title: 'pretokenize', tag: 'CPU', lifetime: 'param', step: 'bpe', desc: 'GPT-2 正则 + 去重计数',
                dispatch: { kind: 'CPU', name: 'pretokenize · chunkTable', count: (S) => `${S.R ? S.R.words.length : '—'} 个不同的块` },
                inputs: [{ name: 'Corpus', format: 'str', type: 'text' }],
                outputs: [{ name: 'Chunks', format: '{块: 次数}', type: 'text' }],
            },
            {
                id: 'train', title: 'trainBPE', tag: 'CPU', lifetime: 'param', step: 'bpe', desc: 'argmax count(a,b) → merge × k',
                dispatch: { kind: 'CPU', name: 'trainBPE', count: (S) => `${S.R ? S.R.merges.length : '—'} 次合并（取前 k = ${S.merges} 条）` },
                inputs: [{ name: 'Chunks', format: '{块: 次数}', type: 'text' }, { name: 'Hyper', format: 'k', type: 'scalar' }],
                outputs: [{ name: 'Merges', format: '[k] (a,b)→id', type: 'table' }, { name: 'Vocab', format: '[256+k] bytes', type: 'table' }],
                detail: { preview: 'mergeTable' },
            },
            {
                id: 'encode', title: 'bpeEncode', tag: 'CPU', lifetime: 'param', step: 'encode', desc: '按 rank 反复合并',
                dispatch: { kind: 'CPU', name: 'bpeEncode', count: (S) => `${S.enc ? S.enc.ids.length : '—'} 个 token` },
                inputs: [{ name: 'Sentence', format: 'str', type: 'text' }, { name: 'Merges', format: '[k]', type: 'table' }],
                outputs: [{ name: 'Ids', format: '[T] int32', type: 'ids' }],
                detail: { preview: 'encToks' },
            },
            {
                id: 'decode', title: 'bpeDecode', tag: 'CPU', category: 'output', lifetime: 'param', step: 'encode', desc: 'bytes 拼接 → UTF-8',
                dispatch: { kind: 'CPU', name: 'bpeDecode', count: '无损：decode(encode(s)) = s' },
                inputs: [{ name: 'Ids', format: '[T] int32', type: 'ids' }, { name: 'Vocab', format: '[256+k] bytes', type: 'table' }],
                outputs: [{ name: 'Text', format: 'str', type: 'text' }],
                detail: { preview: 'roundTrip' },
            },
            {
                id: 'weights', title: 'base_weights', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'lookup', desc: 'LM_WEIGHTS.base → wte, wpe',
                dispatch: { kind: 'CPU', name: 'loadWeights', count: '28 928 个 fp16 参数' },
                outputs: [{ name: 'E', format: '[76 × 32] fp32', type: 'param' }, { name: 'P', format: '[32 × 32] fp32', type: 'param' }],
                detail: { preview: 'wtePca' },
            },
            {
                id: 'charIds', title: 'char_encode', tag: 'CPU', lifetime: 'param', step: 'input', desc: '字符级：stoi[ch]',
                dispatch: { kind: 'CPU', name: 'LM.vocab.encode', count: (S) => `T = ${S.inp ? S.inp.T : '—'}` },
                inputs: [{ name: 'Sentence', format: 'str', type: 'text' }],
                outputs: [{ name: 'CharIds', format: '[T] int32', type: 'ids' }],
                detail: { preview: 'oovView' },
            },
            {
                id: 'embed', title: 'embedding_gather', tag: 'CPU', lifetime: 'param', step: 'lookup', desc: 'E[ids]（= onehot · E）',
                dispatch: { kind: 'CPU', name: 'embedding', count: (S) => `${S.inp ? S.inp.T : '—'} 行 × 32` },
                inputs: [{ name: 'CharIds', format: '[T] int32', type: 'ids' }, { name: 'E', format: '[76 × 32]', type: 'param' }],
                outputs: [{ name: 'TokEmb', format: '[T × 32] fp32', type: 'act' }],
                detail: { preview: 'lookupView' },
            },
            {
                id: 'addpos', title: 'add_position', tag: 'CPU', category: 'output', lifetime: 'param', step: 'input', desc: 'X = E[ids] + P[0..T)',
                dispatch: { kind: 'CPU', name: 'add', count: '→ 第 04 篇 Block 0' },
                inputs: [{ name: 'TokEmb', format: '[T × 32]', type: 'act' }, { name: 'P', format: '[32 × 32]', type: 'param' }],
                outputs: [{ name: 'X', format: '[T × 32] fp32', type: 'act' }],
                detail: { preview: 'inputX' },
            },
            {
                id: 'perm', title: 'bidirAttention', tag: 'CPU', lifetime: 'param', step: 'perm', desc: 'Attn(ΠX) = Π·Attn(X)',
                dispatch: { kind: 'CPU', name: 'LN → QKV → attention(causal=false) → W_O', count: '6 个 token × 2 种顺序 × 有 / 无位置' },
                inputs: [{ name: 'E', format: '[76 × 32]', type: 'param' }, { name: 'P', format: '[32 × 32]', type: 'param' }],
                outputs: [{ name: 'PermOut', format: '[6 × 32] × 4', type: 'act' }],
                detail: { preview: 'permNoPE' },
            },
            {
                id: 'posenc', title: 'sinusoidalPE / applyRope', tag: 'CPU', lifetime: 'param', step: 'rope', desc: 'ω_i = base^(−2i/d)',
                dispatch: { kind: 'CPU', name: 'sinusoidalPE · applyRope', count: '64 × 32 + 3 × 81 次打分' },
                inputs: [{ name: 'Hyper', format: 'base', type: 'scalar' }],
                outputs: [{ name: 'PE', format: '[64 × 32] fp64', type: 'act' }, { name: 'Scores', format: '[3 × 81]', type: 'act' }],
                detail: { preview: 'ropeScore' },
            },
            {
                id: 'segment', title: 'segmentWords', tag: 'CPU', lifetime: 'resident', step: 'w2v', desc: '正向最大匹配（47 词）',
                dispatch: { kind: 'CPU', name: 'segmentWords', count: (S) => `${P03.W2V.nSent} 句 → ${S.w2vSents ? S.w2vSents.reduce((a, s) => a + s.length, 0) : '—'} 个词` },
                inputs: [{ name: 'CorpusZh', format: 'str', type: 'text' }],
                outputs: [{ name: 'Words', format: '[句][词 id]', type: 'ids' }],
            },
            {
                id: 'pairs', title: 'skipgramPairs', tag: 'CPU', lifetime: 'resident', step: 'w2v', desc: '窗口 1–6 · f^0.75 负采样表',
                dispatch: { kind: 'CPU', name: 'skipgramPairs · unigramTable', count: (S) => (S.w2vPairs ? S.w2vPairs.map((p) => p.length).join(' / ') + ' 对' : '—') },
                inputs: [{ name: 'Words', format: '[句][词 id]', type: 'ids' }],
                outputs: [{ name: 'Pairs', format: '6 × [M] (c, o)', type: 'ids' }, { name: 'NegTable', format: '[1e5] int32', type: 'ids' }],
            },
            {
                id: 'sgns', title: 'sgnsUpdate', tag: 'CPU', lifetime: 'history', step: 'w2v', desc: 'σ(u·v) 二分类 · 逐对 SGD',
                dispatch: { kind: 'CPU', name: 'sgnsUpdate', count: (S) => `每帧 ${P03.W2V.perFrame} 对 × 6 个模型，每个模型共 ${P03UI.fmtN(P03.W2V.total)} 对（60 fps 时约 4 秒）` },
                inputs: [{ name: 'Pairs', format: '6 × [M]', type: 'ids' }, { name: 'NegTable', format: '[1e5]', type: 'ids' }, { name: 'Hyper', format: 'K', type: 'scalar' }, { name: 'W2V', format: '6 × 2 × [47 × 16]', type: 'param', history: true }],
                outputs: [{ name: 'W2V', format: '6 × 2 × [47 × 16] fp32', type: 'param' }],
                detail: { preview: 'w2vLoss', code: 0 },
            },
            {
                id: 'w2vEval', title: 'pca · cosine · analogy', tag: 'CPU', category: 'output', lifetime: 'frame', step: 'sim', desc: 'PCA2 · 30×30 cos · 164 道类比',
                dispatch: { kind: 'CPU', name: 'pca2 · analogy', count: '所选窗口的模型（类比：6 个模型都评估）' },
                inputs: [{ name: 'W2V', format: '6 × 2 × [47 × 16]', type: 'param' }, { name: 'Hyper', format: '窗口', type: 'scalar' }],
                outputs: [{ name: 'View', format: '[47 × 2] + [30 × 30]', type: 'act' }],
                detail: { preview: 'w2vPca' },
            },
        ],
    },

    references: [
        { title: 'A New Algorithm for Data Compression', author: 'P. Gage', venue: 'C Users Journal', year: 1994, note: 'BPE 最初是一种压缩算法' },
        { title: 'Neural Machine Translation of Rare Words with Subword Units', author: 'R. Sennrich, B. Haddow, A. Birch', venue: 'ACL', year: 2016, url: 'https://arxiv.org/abs/1508.07909', note: '把 BPE 用于神经网络的子词切分' },
        { title: 'Language Models are Unsupervised Multitask Learners', author: 'A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever', venue: 'OpenAI', year: 2019, note: 'GPT-2：字节级 BPE、预切分正则、50 257 词表' },
        { title: 'SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing', author: 'T. Kudo, J. Richardson', venue: 'EMNLP（系统演示）', year: 2018, url: 'https://arxiv.org/abs/1808.06226' },
        { title: 'Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates', author: 'T. Kudo', venue: 'ACL', year: 2018, url: 'https://arxiv.org/abs/1804.10959', note: 'Unigram 语言模型分词' },
        { title: 'minbpe', author: 'A. Karpathy', venue: 'GitHub', year: 2024, url: 'https://github.com/karpathy/minbpe', note: '本页 BPE 的写法（相邻对计数、合并、按 rank 编码）与它（Python）一致：C++ 版的 pairStats / mergePair 对应它的 get_stats / merge' },
        { title: 'tiktoken', author: 'OpenAI', venue: 'GitHub', url: 'https://github.com/openai/tiktoken', note: 'GPT 系列的 BPE tokenizer 实现' },
        { title: 'Efficient Estimation of Word Representations in Vector Space', author: 'T. Mikolov, K. Chen, G. Corrado, J. Dean', venue: 'arXiv', year: 2013, url: 'https://arxiv.org/abs/1301.3781', note: 'word2vec：skip-gram / CBOW，类比测试' },
        { title: 'Distributed Representations of Words and Phrases and their Compositionality', author: 'T. Mikolov, I. Sutskever, K. Chen, G. Corrado, J. Dean', venue: 'NeurIPS', year: 2013, url: 'https://arxiv.org/abs/1310.4546', note: '负采样、f^0.75 噪声分布、高频词下采样' },
        { title: 'Neural Word Embedding as Implicit Matrix Factorization', author: 'O. Levy, Y. Goldberg', venue: 'NeurIPS', year: 2014, note: 'skip-gram + 负采样 ≈ 分解平移后的 PMI 矩阵' },
        { title: 'Linguistic Regularities in Sparse and Explicit Word Representations', author: 'O. Levy, Y. Goldberg', venue: 'CoNLL', year: 2014, note: '类比测试的分析（3CosAdd / 3CosMul）' },
        { title: 'Using the Output Embedding to Improve Language Models', author: 'O. Press, L. Wolf', venue: 'EACL', year: 2017, url: 'https://arxiv.org/abs/1608.05859', note: '输入 / 输出 embedding 权重共享' },
        { title: 'Attention Is All You Need', author: 'A. Vaswani et al.', venue: 'NeurIPS', year: 2017, url: 'https://arxiv.org/abs/1706.03762', note: '正弦位置编码' },
        { title: 'RoFormer: Enhanced Transformer with Rotary Position Embedding', author: 'J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, Y. Liu', venue: 'arXiv', year: 2021, url: 'https://arxiv.org/abs/2104.09864', note: 'RoPE' },
        { title: 'Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation', author: 'O. Press, N. A. Smith, M. Lewis', venue: 'ICLR', year: 2022, url: 'https://arxiv.org/abs/2108.12409', note: 'ALiBi' },
        { title: 'Transformer Language Models without Positional Encodings Still Learn Positional Information', author: 'A. Haviv, O. Ram, O. Press, P. Izsak, O. Levy', venue: 'Findings of EMNLP', year: 2022, url: 'https://arxiv.org/abs/2203.16634', note: '因果掩码本身泄露位置信息' },
        { title: 'Extending Context Window of Large Language Models via Positional Interpolation', author: 'S. Chen, S. Wong, L. Chen, Y. Tian', venue: 'arXiv', year: 2023, url: 'https://arxiv.org/abs/2306.15595', note: 'RoPE 位置插值（PI）' },
        { title: 'YaRN: Efficient Context Window Extension of Large Language Models', author: 'B. Peng, J. Quesnelle, H. Fan, E. Shippole', venue: 'ICLR', year: 2024, url: 'https://arxiv.org/abs/2309.00071' },
        { title: 'The Llama 3 Herd of Models', author: 'Llama Team, AI @ Meta', venue: 'arXiv', year: 2024, url: 'https://arxiv.org/abs/2407.21783', note: '128K 词表、RoPE base 500 000、128K 上下文' },
    ],
});
