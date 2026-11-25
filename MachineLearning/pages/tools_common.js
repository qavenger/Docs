// tools_common.js — 第 19–21 篇共用的算法：检索（BM25 / 向量 / 混合 / 两级路由）、动作类型分类、token 计数（BPE）、
// 工具定义的几种格式（MCP tools/list、Anthropic、OpenAI）、JSON Schema 校验、参数抽取、不同「工具加载策略」下的 prompt 组装与 token 账。
// 依赖：TKCAT（tools_catalog.js）、TK_W（tools_weights.js：检索模型、动作分类器、BPE 规则、工具地图坐标）。浏览器与 Node 通用。
const TK = (() => {
    const CAT = typeof TKCAT !== 'undefined' ? TKCAT : require('./tools_catalog.js');
    // ═════════════════════════ 0. 小工具 ═════════════════════════
    function fnv(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
    const CJK = /[㐀-鿿豈-﫿]/;
    function b64ToBytes(b64) { if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64')); const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }
    function f16ToF32(h) { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff; if (e === 0) return s * f * 2 ** -24; if (e === 31) return f ? NaN : s * Infinity; return s * (1 + f / 1024) * 2 ** (e - 15); }
    function f16Array(b64) { const u = b64ToBytes(b64), n = u.length >> 1, out = new Float32Array(n); for (let i = 0; i < n; i++) out[i] = f16ToF32(u[2 * i] | (u[2 * i + 1] << 8)); return out; }
    function topk(score, k) { const idx = []; for (let i = 0; i < score.length; i++) if (score[i] > -Infinity) idx.push(i); idx.sort((a, b) => score[b] - score[a]); return idx.slice(0, k); }
    function normalize(v) { let n = 0; for (let i = 0; i < v.length; i++) n += v[i] * v[i]; n = 1 / Math.sqrt(n || 1); for (let i = 0; i < v.length; i++) v[i] *= n; return v; }
    const dot = (a, b, ao, n) => { let s = 0; for (let j = 0; j < n; j++) s += a[ao + j] * b[j]; return s; };

    // ═════════════════════════ 1. 分词：英文词（拆 snake / camel、去复数）+ 中文单字与相邻两字 ═════════════════════════
    const stem = (w) => (w.length > 4 && w.endsWith('ies') ? w.slice(0, -3) + 'y' : w.length > 3 && /(sh|ch|x|ss)es$/.test(w) ? w.slice(0, -2) : w.length > 3 && w.endsWith('s') && !w.endsWith('ss') && !w.endsWith('us') ? w.slice(0, -1) : w);
    const STOP = new Set('the a an to in of for and or on with by me my it this that please can you i need could go ahead is are be from at into as some do does will would want should let us we our your its'.split(' '));
    const ZH_STOP = new Set('的了一下我要请帮在里把给能不吗个这那就也都和与对用想让是有你他她它们么呢吧啊'.split(''));
    function splitWords(text) {
        const s = String(text).replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase();
        const ascii = (s.match(/[a-z0-9]+/g) || []).map(stem), cjk = [];
        let run = '';
        for (const ch of s) { if (CJK.test(ch)) run += ch; else if (run) { cjk.push(run); run = ''; } }
        if (run) cjk.push(run);
        return { ascii, cjk };
    }
    /** BM25 用的词项：英文词（去停用词）、中文单字（去虚词）+ 相邻两字 */
    function terms(text) {
        const { ascii, cjk } = splitWords(text), out = [];
        for (const w of ascii) if (!STOP.has(w)) out.push(w);
        for (const run of cjk) { const cs = [...run]; for (let i = 0; i < cs.length; i++) { if (!ZH_STOP.has(cs[i])) out.push(cs[i]); if (i + 1 < cs.length) out.push(cs[i] + cs[i + 1]); } }
        return out;
    }
    /** 向量模型的输入：哈希特征（英文词、词内三字母、中文字、两字）→ 桶号 → 权重 */
    function features(text, NB) {
        const { ascii, cjk } = splitWords(text), f = new Map(), add = (s, w) => { const b = fnv(s) % NB; f.set(b, (f.get(b) || 0) + w); };
        for (const w of ascii) { add('w:' + w, 1); const p = '<' + w + '>'; for (let i = 0; i + 3 <= p.length; i++) add('t:' + p.slice(i, i + 3), 0.4); }
        for (const run of cjk) { const cs = [...run]; for (let i = 0; i < cs.length; i++) { add('c:' + cs[i], 0.7); if (i + 1 < cs.length) add('b:' + cs[i] + cs[i + 1], 1); } }
        return f;
    }

    // ═════════════════════════ 2. 工具与应用的「文档」：检索时被匹配的文字 ═════════════════════════
    const appOf = new Map(CAT.APPS.map((a) => [a.ns, a]));
    /** 工具文档：英文名 + 英文描述（真实 MCP 服务器的描述几乎都是英文）；withZh = 额外写上中文说法 */
    function toolDoc(t, withZh) {
        let d = `${t.name.replace(/_/g, ' ')}. ${t.description}`;
        if (withZh) { const zh = new Set(); for (const f of t.zh) { for (const v of [].concat(f.v)) zh.add(v); for (const o of f.o || []) if (o) zh.add(o); } d += ' ' + [...zh].join(' ') + ' ' + appOf.get(t.ns).zh.join(' '); }
        return d;
    }
    /** 应用（MCP 服务器）文档：名字 + 一句话描述 + 对象列表 —— 两级路由的第一级只看这个 */
    function appDoc(a) { const objs = Object.values(a.objects).map((o) => o[1].split(',')[0]); return `${a.name}. ${a.about} ${objs.length ? 'Objects: ' + objs.join(', ') + '.' : 'Tools: ' + a.tools.map((t) => t.name.replace(/_/g, ' ')).join(', ') + '.'}`; }

    // ═════════════════════════ 3. BM25（Okapi，k1 = 1.2、b = 0.75） ═════════════════════════
    function bm25Build(docs) {
        const N = docs.length, df = new Map(), post = new Map(), len = new Float32Array(N);
        docs.forEach((d, i) => { const ts = terms(d), tf = new Map(); for (const t of ts) tf.set(t, (tf.get(t) || 0) + 1); len[i] = ts.length; for (const [t, c] of tf) { df.set(t, (df.get(t) || 0) + 1); let p = post.get(t); if (!p) post.set(t, (p = [])); p.push(i, c); } });
        let avg = 0; for (let i = 0; i < N; i++) avg += len[i]; avg /= N;
        return { N, df, post, len, avg, k1: 1.2, b: 0.75 };
    }
    /** 返回 { score: Float32Array(N), hits: Map(i → [[词, 得分]…]) } */
    function bm25Score(I, query, wantHits) {
        const qt = [...new Set(terms(query))], score = new Float32Array(I.N), hits = wantHits ? new Map() : null;
        for (const t of qt) {
            const p = I.post.get(t); if (!p) continue;
            const n = p.length / 2, idf = Math.log(1 + (I.N - n + 0.5) / (n + 0.5));
            for (let j = 0; j < p.length; j += 2) {
                const i = p[j], c = p[j + 1], s = (idf * c * (I.k1 + 1)) / (c + I.k1 * (1 - I.b + (I.b * I.len[i]) / I.avg));
                score[i] += s;
                if (hits) { let h = hits.get(i); if (!h) hits.set(i, (h = [])); h.push([t, s]); }
            }
        }
        return { score, hits, qt };
    }

    // ═════════════════════════ 4. 向量检索：哈希特征 → 嵌入求和 → tanh → 线性 → 单位向量（离线训练，见 tools/tk_train.js） ═════════════════════════
    function loadEncoder(W) {
        const q = b64ToBytes(W.E), scale = f16Array(W.Es), E = new Float32Array(q.length);
        for (let r = 0; r < W.NB; r++) { const s = scale[r]; for (let j = 0; j < W.D; j++) { const v = q[r * W.D + j]; E[r * W.D + j] = (v > 127 ? v - 256 : v) * s; } }
        return { NB: W.NB, D: W.D, D2: W.D2, E, b0: f16Array(W.b0), W1: f16Array(W.W1), b1: f16Array(W.b1) };
    }
    function embed(enc, text, out) {
        const f = features(text, enc.NB), h = new Float32Array(enc.D);
        let n2 = 0;
        for (const [b, w] of f) { const o = b * enc.D; for (let j = 0; j < enc.D; j++) h[j] += enc.E[o + j] * w; n2 += w * w; }
        const inv = 1 / Math.sqrt(Math.max(n2, 1));
        for (let j = 0; j < enc.D; j++) h[j] = Math.tanh(h[j] * inv + enc.b0[j]);
        const z = out || new Float32Array(enc.D2);
        for (let k = 0; k < enc.D2; k++) z[k] = enc.b1[k] + dot(enc.W1, h, k * enc.D, enc.D);
        return normalize(z);
    }
    function embedAll(enc, texts) { const M = new Float32Array(texts.length * enc.D2), z = new Float32Array(enc.D2); texts.forEach((t, i) => { embed(enc, t, z); M.set(z, i * enc.D2); }); return M; }
    function cosScores(M, q, n, D) { const s = new Float32Array(n); for (let i = 0; i < n; i++) s[i] = dot(M, q, i * D, D); return s; }
    /** 倒数排名融合（RRF，k = 60）：把几路检索的名次合起来，不需要对分数做归一化 */
    function rrf(lists, n, k, w) { const s = new Float32Array(n).fill(0); lists.forEach((L, l) => { const wl = w ? w[l] : 1; L.forEach((i, r) => { s[i] += wl / ((k || 60) + r + 1); }); }); return s; }

    // ═════════════════════════ 5. 整套检索器：建索引一次，之后每个请求 < 几毫秒 ═════════════════════════
    function buildRetriever(W, o) {
        o = o || {};
        const tools = CAT.tools, apps = CAT.APPS, enc = loadEncoder(W.enc);
        const docsEn = tools.map((t) => toolDoc(t, false)), docsZh = tools.map((t) => toolDoc(t, true));
        const R = { tools, apps, enc, docsEn, docsZh, bmEn: bm25Build(docsEn), bmZh: bm25Build(docsZh), T: embedAll(enc, docsEn), A: embedAll(enc, apps.map(appDoc)), D: enc.D2, toolApp: Int32Array.from(tools.map((t) => apps.findIndex((a) => a.ns === t.ns))), wB: (W.train && W.train.wB) || { en: 0.25, zh: 1 } };
        R.search = (query, opt) => search(R, query, opt || {});
        return R;
    }
    /**
     * 一次检索：返回各路的排名与分数。opt: { k, docZh（BM25 是否也索引中文说法）, mode: 'bm25'|'dense'|'hybrid'|'router', mApps（两级路由保留几个应用）, only（只在这些应用里找：「已连接的服务器」）}
     */
    function search(R, query, opt) {
        const n = R.tools.length, k = opt.k || 10, allow = opt.only && opt.only.length ? new Set(opt.only) : null;
        const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
        const bm = bm25Score(opt.docZh ? R.bmZh : R.bmEn, query, true), q = embed(R.enc, query), dense = cosScores(R.T, q, n, R.D);
        const appS = cosScores(R.A, q, R.apps.length, R.D), appP = softmax(appS, 25);
        if (allow) for (let i = 0; i < n; i++) if (!allow.has(R.tools[i].ns)) { bm.score[i] = -Infinity; dense[i] = -Infinity; }
        const bmTop = topk(bm.score, 200).filter((i) => bm.score[i] > 0), deTop = topk(dense, 200);
        const hyb = rrf([bmTop, deTop], n, 60, [opt.wB != null ? opt.wB : opt.docZh ? R.wB.zh : R.wB.en, 1]);   // 加权 RRF：BM25 一路的权重 wB（训练后在留出集上挑的）
        if (allow) for (let i = 0; i < n; i++) if (!allow.has(R.tools[i].ns)) hyb[i] = -Infinity;
        const appTop = topk(appS, R.apps.length), keepApps = new Set(appTop.slice(0, opt.mApps || 3));
        const routed = new Float32Array(n).fill(-Infinity); for (let i = 0; i < n; i++) if (keepApps.has(R.toolApp[i]) && (!allow || allow.has(R.tools[i].ns))) routed[i] = dense[i];
        const ms = (typeof performance !== 'undefined' ? performance : Date).now() - t0;
        return { query, q, bm, dense, hyb, routed, appS, appP, appTop, ms, top: { bm25: bmTop.slice(0, k), dense: deTop.slice(0, k), hybrid: topk(hyb, k), router: topk(routed, k) } };
    }
    function softmax(s, scale) { let m = -Infinity; for (const v of s) if (v > m) m = v; const p = new Float32Array(s.length); let z = 0; for (let i = 0; i < s.length; i++) { p[i] = Math.exp((s[i] - m) * scale); z += p[i]; } for (let i = 0; i < s.length; i++) p[i] /= z; return p; }

    // ═════════════════════════ 6. 动作类型：直接回答 / 调一个工具 / 写代码 / 操作界面 / 先问清楚（哈希特征上的线性分类器） ═════════════════════════
    function loadAction(W) { const q = b64ToBytes(W.Wq), sc = f16Array(W.Ws), M = new Float32Array(q.length); const C = W.C; for (let r = 0; r < W.NB; r++) for (let c = 0; c < C; c++) { const v = q[r * C + c]; M[r * C + c] = (v > 127 ? v - 256 : v) * sc[r]; } return { NB: W.NB, C, M, b: f16Array(W.b) }; }
    function classifyAction(A, text) {
        const f = features(text, A.NB), z = Float32Array.from(A.b), contrib = [];
        let n2 = 0; for (const [, w] of f) n2 += w * w; const inv = 1 / Math.sqrt(Math.max(n2, 1));
        for (const [b, w] of f) for (let c = 0; c < A.C; c++) z[c] += A.M[b * A.C + c] * w * inv;
        const p = softmax(z, 1);
        return { p: Array.from(p), best: CAT.ACTIONS[topk(p, 1)[0]] };
    }

    // ═════════════════════════ 7. token 计数：16k 词表的字节级 BPE（与第 06 篇同一算法，在本系列讲义文本上训练） ═════════════════════════
    function loadBPE(B) {
        const m = new Uint16Array(b64ToBytes(B.merges).buffer), rank = new Map();
        for (let r = 0; r < B.K; r++) rank.set(m[2 * r] * 65536 + m[2 * r + 1], r);
        return { K: B.K, m, rank, re: new RegExp(B.pre, 'gu'), cache: new Map() };
    }
    const enc8 = typeof TextEncoder !== 'undefined' ? new TextEncoder() : null;
    function chunkTokens(B, w) {
        let n = B.cache.get(w); if (n !== undefined) return n;
        let ids = Array.from(enc8 ? enc8.encode(w) : Buffer.from(w, 'utf8'));
        while (ids.length > 1) {
            let best = Infinity, at = -1;
            for (let j = 0; j + 1 < ids.length; j++) { const r = B.rank.get(ids[j] * 65536 + ids[j + 1]); if (r !== undefined && r < best) { best = r; at = j; } }
            if (at < 0) break;
            const a = B.m[2 * best], b = B.m[2 * best + 1], out = [];
            for (let j = 0; j < ids.length; j++) { if (j + 1 < ids.length && ids[j] === a && ids[j + 1] === b) { out.push(256 + best); j++; } else out.push(ids[j]); }
            ids = out;
        }
        n = ids.length; if (B.cache.size < 200000) B.cache.set(w, n);
        return n;
    }
    function countTokens(B, text) { let n = 0; B.re.lastIndex = 0; for (const m of String(text).matchAll(B.re)) n += chunkTokens(B, m[0]); return n; }

    // ═════════════════════════ 8. 工具定义的几种格式 ═════════════════════════
    /** MCP（tools/list 返回的一项）*/
    const mcpTool = (t) => ({ name: t.name, title: t.title, description: t.description, inputSchema: t.inputSchema, annotations: t.annotations });
    /** Anthropic Messages API 的 tools[]（名字只能用字母数字 _ -：MCP 客户端把服务器名拼进去）*/
    const anthropicTool = (t) => ({ name: t.mcpName, description: t.description, input_schema: t.inputSchema });
    /** OpenAI Responses API 的 function tool */
    const openaiTool = (t) => ({ type: 'function', name: t.mcpName, description: t.description, parameters: t.inputSchema });
    const toolJSON = (t, fmt) => JSON.stringify(fmt === 'openai' ? openaiTool(t) : fmt === 'mcp' ? mcpTool(t) : anthropicTool(t));
    /** 工具搜索工具（按需加载用）：模型只看到它，需要时自己搜 */
    const TOOL_SEARCH = { name: 'tool_search', description: 'Search the catalog of available tools by what you want to do. Returns the full definitions of the best matching tools, which you can then call. Use this before calling any tool that is not already loaded.', input_schema: { type: 'object', properties: { query: { type: 'string', description: 'What you want to do, in a few words (any language).' }, limit: { type: 'integer', description: 'Max tools to return (default 5).' } }, required: ['query'] } };

    // ═════════════════════════ 9. JSON Schema 校验（type / enum / required / items / min/maxItems 子集） ═════════════════════════
    function validate(schema, value, at, errs) {
        at = at || '$'; errs = errs || [];
        if (!schema || !Object.keys(schema).length) return errs;
        const ty = schema.type, isInt = (v) => typeof v === 'number' && Number.isInteger(v);
        const okType = !ty || (ty === 'integer' ? isInt(value) : ty === 'number' ? typeof value === 'number' && isFinite(value) : ty === 'array' ? Array.isArray(value) : ty === 'object' ? value && typeof value === 'object' && !Array.isArray(value) : typeof value === ty);
        if (!okType) { errs.push(`${at}: expected ${ty}, got ${Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value}`); return errs; }
        if (schema.enum && !schema.enum.includes(value)) errs.push(`${at}: must be one of ${schema.enum.map((v) => JSON.stringify(v)).join(', ')}`);
        if (ty === 'object') { for (const r of schema.required || []) if (!(r in value)) errs.push(`${at}: missing required property "${r}"`); for (const [k, v] of Object.entries(value)) if (schema.properties && schema.properties[k]) validate(schema.properties[k], v, `${at}.${k}`, errs); else if (schema.properties) errs.push(`${at}: unknown property "${k}"`); }
        if (ty === 'array') { if (schema.minItems != null && value.length < schema.minItems) errs.push(`${at}: needs at least ${schema.minItems} items`); if (schema.maxItems != null && value.length > schema.maxItems) errs.push(`${at}: at most ${schema.maxItems} items`); if (schema.items) value.forEach((v, i) => validate(schema.items, v, `${at}[${i}]`, errs)); }
        return errs;
    }

    // ═════════════════════════ 10. 参数抽取（规则版：路径、名字、数字、枚举、布尔、三维向量、引号里的文字） ═════════════════════════
    // 真实系统里这一步由模型按 schema 生成（并可用约束解码保证 JSON 合法）；这里用规则演示「哪些能从请求里直接拿到、哪些缺了要追问」
    const ENUM_ZH = { point: '点光源', spot: '聚光灯', directional: '平行光', rect: '面光源', png: 'png', jpg: 'jpg', webp: 'webp', h264: 'h264', hevc: 'h265', av1: 'av1', vp9: 'vp9', mp3: 'mp3', wav: 'wav', aac: 'aac', production: '最高质量', preview: '预览', high: '高质量', medium: '中等', squash: '压缩合并', rebase: 'rebase', merge: '普通合并', open: '打开的', closed: '关闭的', Windows: 'windows', Android: '安卓', IOS: 'ios', candela: '坎德拉', lux: '勒克斯' };
    function extractArgs(t, text) {
        const P = t.inputSchema.properties, R = new Set(t.inputSchema.required), args = {}, how = {}, s = String(text);
        const quoted = [...s.matchAll(/[“"「『]([^”"」』]+)[”"」』]/g)].map((m) => m[1]);
        const pathM = s.match(/([A-Za-z]:[\\/][^\s"“”，,]+|\/[\w./-]+[\w-]|[\w-]+\.(?:fbx|gltf|glb|obj|usd|png|jpg|jpeg|webp|mp4|mov|wav|mp3|csv|xlsx|json|uasset|umap|blend|psd|txt|md))/i);
        const nameM = s.match(/(?:叫|名为|名字是|命名为|named|called)\s*[“"]?([\w\-一-龥]+)/i);
        const nums = [...s.matchAll(/(?<![\w.])(-?\d+(?:\.\d+)?)\s*(%|％)?/g)].map((m) => (m[2] ? +m[1] / 100 : +m[1]));
        const identM = s.match(/\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b/);   // PointLight_3、SM_Rock_01 这类引擎里的对象名
        const vecM = s.match(/[(（\[]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[)）\]]/);
        let numUsed = 0;
        for (const [k, p] of Object.entries(P)) {
            let v;
            if (p.enum) { const hit = p.enum.find((e) => new RegExp(`\\b${e}\\b`, 'i').test(s) || (ENUM_ZH[e] && s.toLowerCase().includes(ENUM_ZH[e].toLowerCase()))); if (hit) { v = hit; how[k] = '枚举值匹配'; } }
            else if (p.type === 'array' && p.minItems === 3) { if (vecM) { v = [+vecM[1], +vecM[2], +vecM[3]]; how[k] = '括号里的三个数'; } }
            else if (p.type === 'number' || p.type === 'integer') { if (numUsed < nums.length && !(vecM)) { v = p.type === 'integer' ? Math.round(nums[numUsed]) : nums[numUsed]; numUsed++; how[k] = '请求里的数字'; } }
            else if (p.type === 'boolean') { if (/(开启|打开|启用|enable|turn on|\bon\b|true)/i.test(s)) { v = true; how[k] = '「开启」'; } else if (/(关闭|禁用|关掉|disable|turn off|\boff\b|false)/i.test(s)) { v = false; how[k] = '「关闭」'; } }
            else if (/^(path|output|input|destination|zip_path|file)$/.test(k)) { if (pathM) { v = pathM[1]; how[k] = '路径'; } }
            else if (/^(name|new_name|title|branch_name)$/.test(k)) { if (nameM) { v = nameM[1]; how[k] = '「叫 …」'; } else if (quoted.length) { v = quoted[0]; how[k] = '引号里的文字'; } }
            else if (/^(content|body|text|message|prompt|query|q|sql|command|code|description)$/.test(k)) { if (quoted.length) { v = quoted[quoted.length - 1]; how[k] = '引号里的文字'; } }
            else if (p.type === 'string' && /^(actor|id|camera|layer|object|mesh|material|blueprint|sheet|channel_id|repo|owner|key|table|texture)$/.test(k)) { if (quoted.length) { v = quoted[0]; how[k] = '引号里的名字'; } else if (nameM) { v = nameM[1]; how[k] = '「叫 …」'; } else if (identM) { v = identM[1]; how[k] = '对象名（带下划线的标识符）'; } }
            if (v !== undefined) args[k] = v;
        }
        const missing = [...R].filter((k) => !(k in args));
        return { args, how, missing, errors: validate(t.inputSchema, args) };
    }

    // ═════════════════════════ 11. 不同「工具加载策略」下，第一轮请求的 prompt 由哪些部分组成 ═════════════════════════
    const SYSTEM = 'You are an assistant that can use tools. Call a tool when it helps; otherwise answer directly. Ask the user when required information is missing.';
    /**
     * strategy: 'all'（全部工具定义）| 'topk'（检索出的 k 个）| 'router'（先给应用清单，选中应用后再给它的工具）| 'search'（工具搜索：只给 tool_search + 少量常用工具）| 'code'（工具当成代码文件：只给文件树，模型自己读需要的接口）
     * 返回 [{ part, text, tokens }]
     */
    function promptParts(B, strategy, ctx) {
        const parts = [], add = (part, text) => parts.push({ part, text, tokens: countTokens(B, text) });
        add('系统提示', SYSTEM);
        if (strategy === 'all') add(`全部 ${CAT.tools.length} 个工具定义`, CAT.tools.map((t) => toolJSON(t)).join('\n'));
        else if (strategy === 'topk') add(`检索出的 ${ctx.picked.length} 个工具定义`, ctx.picked.map((t) => toolJSON(t)).join('\n'));
        else if (strategy === 'router') { add(`应用清单（${CAT.APPS.length} 个，每个一句话）`, CAT.APPS.map((a) => `- ${a.ns}: ${a.about}`).join('\n')); add(`第 2 轮：选中的应用「${ctx.app.ns}」的 ${ctx.app.tools.length} 个工具`, ctx.app.tools.map((t) => toolJSON(t)).join('\n')); }
        else if (strategy === 'search') { add('tool_search 工具', JSON.stringify(TOOL_SEARCH)); add(`常驻工具 ${ctx.always.length} 个`, ctx.always.map((t) => toolJSON(t)).join('\n')); add(`第 2 轮：tool_search 返回的 ${ctx.picked.length} 个定义`, ctx.picked.map((t) => toolJSON(t)).join('\n')); }
        else if (strategy === 'code') { add('服务器文件树（每个工具一个 .ts 文件）', CAT.APPS.map((a) => `servers/${a.ns}/ (${a.tools.length} files)`).join('\n')); add(`第 2 轮：读入的 ${ctx.picked.length} 个接口签名`, ctx.picked.map((t) => `export function ${t.name}(input: ${JSON.stringify(t.inputSchema.properties)}): Promise<any> // ${t.description}`).join('\n')); }
        return parts;
    }

    // ═════════════════════════ 12. MCP 消息（JSON-RPC 2.0） ═════════════════════════
    let rpcId = 0;
    const rpc = (method, params) => ({ jsonrpc: '2.0', id: ++rpcId, method, params });
    const mcp = {
        initialize: (client) => rpc('initialize', { protocolVersion: '2025-06-18', capabilities: { roots: { listChanged: true }, elicitation: {} }, clientInfo: { name: client || 'demo-host', version: '1.0.0' } }),
        toolsList: (cursor) => rpc('tools/list', cursor ? { cursor } : {}),
        toolsListResult: (id, tools, nextCursor) => ({ jsonrpc: '2.0', id, result: Object.assign({ tools: tools.map(mcpTool) }, nextCursor ? { nextCursor } : {}) }),
        call: (name, args) => rpc('tools/call', { name, arguments: args }),
        callResult: (id, text, structured, isError) => ({ jsonrpc: '2.0', id, result: Object.assign({ content: [{ type: 'text', text }] }, structured ? { structuredContent: structured } : {}, isError ? { isError: true } : {}) }),
        listChanged: () => ({ jsonrpc: '2.0', method: 'notifications/tools/list_changed' }),
        resetIds: () => { rpcId = 0; },
    };

    return { CAT, fnv, b64ToBytes, f16Array, topk, normalize, splitWords, terms, features, toolDoc, appDoc, bm25Build, bm25Score, loadEncoder, embed, embedAll, cosScores, rrf, softmax, buildRetriever, search, loadAction, classifyAction, loadBPE, countTokens, mcpTool, anthropicTool, openaiTool, toolJSON, TOOL_SEARCH, validate, extractArgs, SYSTEM, promptParts, mcp };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = TK;
