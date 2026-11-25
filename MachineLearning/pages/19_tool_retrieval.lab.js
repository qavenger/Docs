// 19_tool_retrieval.lab.js — 系列第 19 篇：成千上万个工具时，模型怎么知道该调哪一个（工具检索、两级路由、按需加载、参数校验、动作类型）
// 构建（在 pages/ 下）：node <skill>/scripts/build_doc.mjs llm_common.js tools_catalog.js tools_common.js tools_weights.js tools_cpp.js 19_tool_retrieval.lab.js -o 19_tool_retrieval.html
// 数据：tools_catalog.js（125 个应用、1590 个工具，JSON Schema 齐全）；模型：tools_weights.js（tools/tk_train.js 离线训练的检索编码器 + 动作分类器 + 16k BPE）。
const P19 = (() => {
    const CAT = TKCAT;
    const DOM = { game: ['游戏引擎与开发', '#e0a060'], gameops: ['游戏运营服务', '#f2c879'], dcc: ['DCC / 设计', '#c678dd'], media: ['音视频与图片', '#d19a66'], office: ['办公协作', '#6ab0e0'], dev: ['开发与运维', '#5ad07a'], data: ['数据分析', '#56b6c2'], crm: ['CRM', '#e06c75'], support: ['客服', '#be5046'], pm: ['项目管理', '#e8c25a'], commerce: ['电商与支付', '#98c379'], marketing: ['营销', '#ff79c6'], finance: ['财务 / HR', '#8a8f98'], gui: ['操作界面', '#ffffff'] };
    const CONN = { all: null, game: ['unreal', 'unity', 'perforce', 'feishu', 'gpuprof'], dcc: ['blender', 'maya', 'photoshop', 'substance', 'figma'], dev: ['github', 'git', 'jira', 'slack', 'docker', 'k8s', 'shell'], ops: ['tapd', 'steam', 'analytics', 'support', 'feishu'] };
    const ALWAYS = ['filesystem/read_file', 'web/web_search', 'shell/run_command'];   // 常驻工具：几乎每个任务都可能用到
    function init() {
        const R = TK.buildRetriever(TK_W), A = TK.loadAction(TK_W.act), B = TK.loadBPE(TK_BPE);
        const m = new Int16Array(TK.b64ToBytes(TK_W.map).buffer), map = Float32Array.from(m, (v) => v / 32000);
        const dom = R.tools.map((t) => R.apps[R.toolApp[t.i]].domain);
        return { R, A, B, map, dom };
    }
    /** 一次请求的全部结果：各路检索、路由、选中的工具、参数抽取、动作类型、在地图上的位置 */
    function analyze(C, S) {
        const R = C.R, res = R.search(S.query || ' ', { k: 20, docZh: S.docZh, mApps: S.mApps, only: CONN[S.connected] });
        const pickI = res.top.hybrid.length ? res.top.hybrid[0] : res.top.dense[0], pick = R.tools[pickI];
        const nb = res.top.dense.slice(0, 8); let wx = 0, wy = 0, ws = 0;
        for (const i of nb) { const w = Math.exp(res.dense[i] / 0.05); wx += w * C.map[2 * i]; wy += w * C.map[2 * i + 1]; ws += w; }
        return { res, pickI, pick, ex: TK.extractArgs(pick, S.query || ''), act: TK.classifyAction(C.A, S.query || ''), qx: wx / ws, qy: wy / ws };
    }
    /** 不同加载策略下前两轮请求的输入 token（用 16k BPE 实数） */
    function strategies(C, S, Q) {
        const B = C.B, k = S.k, picked = Q.res.top.hybrid.slice(0, k).map((i) => C.R.tools[i]);
        const app = C.R.apps[Q.res.appTop[0]], always = ALWAYS.map((id) => CAT.byId.get(id));
        const user = TK.countTokens(B, S.query || '');
        const out = {};
        for (const st of ['topk', 'router', 'search', 'code']) {   // 「全部放进去」依赖 lazy 的 tok stage，在预览里用 allRow() 现算
            const parts = TK.promptParts(B, st, { picked, app, always });
            out[st] = { parts, user, total: parts.reduce((s, p) => s + p.tokens, 0) + user };
        }
        return out;
    }
    /** 「全部放进去」那一行（需要 tok stage 数好的 1590 个定义的 token）*/
    function allRow(S) { if (!S.tok) return null; const user = S.ST.topk.user; return { parts: [{ part: '系统提示', tokens: S.tok.sys }, { part: `全部 ${CAT.tools.length} 个工具定义`, tokens: S.tok.all }], user, total: S.tok.sys + S.tok.all + user }; }
    /** 页面里实时评测：用与训练时相同的种子生成三组留出请求（各 300 条），每条都算出各路检索里正确工具的名次 */
    function newBench(C) {
        const tools = C.R.tools, isNew = tools.map((t) => CAT.isNewTool(t)), sets = { seen: [], held: [], new: [] };
        tools.forEach((t, i) => {
            if (isNew[i]) { const u = CAT.utterances(t, LM.rng(7000 + i), 4, false); sets.new.push({ text: u[0].text, y: i }, { text: u[1].text, y: i }); return; }
            sets.seen.push({ text: CAT.utterances(t, LM.rng(5000 + i), 2, false)[0].text, y: i });
            sets.held.push({ text: CAT.utterances(t, LM.rng(9000 + i), 2, true)[0].text, y: i });
        });
        const pick = (a, n, seed) => { const r = LM.rng(seed), b = a.slice(); for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b.slice(0, n); };
        const jobs = []; for (const [k, a] of Object.entries(sets)) for (const q of pick(a, 300, k.length * 31)) jobs.push(Object.assign({ set: k }, q));
        return { jobs, k: 0, res: { seen: {}, held: {}, new: {} } };
    }
    const MODES = ['bm25en', 'bm25zh', 'dense', 'hybEn', 'hybZh', 'router'];
    const MNAME = { bm25en: 'BM25（只有英文描述）', bm25zh: 'BM25（描述里加中文说法）', dense: '向量检索', hybEn: '混合（英文描述，BM25 权重 0.25）', hybZh: '混合（加中文说法，权重 1）', router: '两级路由（前 3 个应用）' };
    const MCOL = { bm25en: '#8a8f98', bm25zh: '#e8c25a', dense: '#6ab0e0', hybEn: '#c678dd', hybZh: '#5ad07a', router: '#e0a060' };
    function rankOf(score, y) { const sy = score[y]; if (!(sy > 0) && sy !== 0 && !isFinite(sy)) return Infinity; let r = 1; for (let i = 0; i < score.length; i++) if (score[i] > sy) r++; return r; }
    function benchWork(C, Bn, n) {
        const R = C.R, N = R.tools.length;
        for (let c = 0; c < n && Bn.k < Bn.jobs.length; c++, Bn.k++) {
            const q = Bn.jobs[Bn.k], bmE = TK.bm25Score(R.bmEn, q.text, false).score, bmZ = TK.bm25Score(R.bmZh, q.text, false).score, z = TK.embed(R.enc, q.text), de = TK.cosScores(R.T, z, N, R.D), as = TK.cosScores(R.A, z, R.apps.length, R.D);
            const deTop = TK.topk(de, 200), bmETop = TK.topk(bmE, 200).filter((i) => bmE[i] > 0), bmZTop = TK.topk(bmZ, 200).filter((i) => bmZ[i] > 0);
            const hE = TK.rrf([bmETop, deTop], N, 60, [R.wB.en, 1]), hZ = TK.rrf([bmZTop, deTop], N, 60, [R.wB.zh, 1]);
            const keep = new Set(TK.topk(as, 3)), ro = new Float32Array(N).fill(-Infinity); for (let i = 0; i < N; i++) if (keep.has(R.toolApp[i])) ro[i] = de[i];
            const rk = { bm25en: bmE[q.y] > 0 ? rankOf(bmE, q.y) : Infinity, bm25zh: bmZ[q.y] > 0 ? rankOf(bmZ, q.y) : Infinity, dense: rankOf(de, q.y), hybEn: hE[q.y] > 0 ? rankOf(hE, q.y) : Infinity, hybZh: hZ[q.y] > 0 ? rankOf(hZ, q.y) : Infinity, router: ro[q.y] > -Infinity ? rankOf(ro, q.y) : Infinity };
            const T = Bn.res[q.set];
            for (const m of MODES) { const a = (T[m] = T[m] || { n: 0, r1: 0, r5: 0, r10: 0 }); a.n++; if (rk[m] <= 1) a.r1++; if (rk[m] <= 5) a.r5++; if (rk[m] <= 10) a.r10++; }
            const ap = (T.app = T.app || { n: 0, a1: 0, a3: 0 }), ar = TK.topk(as, 3); ap.n++; if (ar[0] === R.toolApp[q.y]) ap.a1++; if (ar.includes(R.toolApp[q.y])) ap.a3++;
        }
    }
    // —— 画图的小工具 ——
    const mono = (px) => `${px}px ui-monospace, "Cascadia Mono", Consolas, monospace`;
    function fit(ctx, s, w) { s = String(s); if (ctx.measureText(s).width <= w) return s; while (s.length > 1 && ctx.measureText(s + '…').width > w) s = s.slice(0, -1); return s + '…'; }
    /** JSON 逐行上色画出来（键 = 青、字符串 = 绿、数字 / 布尔 = 橙） */
    function drawJSON(ctx, obj, x, y, w, maxLines, px) {
        const lines = JSON.stringify(obj, null, 2).replace(/\[\s*((?:[^\[\]{}]|\n)*?)\s*\]/g, (m, inner) => (/[{[]/.test(inner) ? m : '[' + inner.split(/,\s*/).map((x) => x.trim()).join(', ') + ']')).split('\n'), lh = px + 4;
        ctx.font = mono(px); ctx.textBaseline = 'top';
        lines.slice(0, maxLines).forEach((ln, i) => {
            let cx = x; const parts = ln.match(/("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?|true|false|null)|([^"\d\-tfn]+|.)/g) || [];
            for (const p of parts) {
                const isKey = /^"/.test(p) && /:\s*$/.test(p), isStr = /^"/.test(p), isNum = /^(-?\d|true|false|null)/.test(p);
                ctx.fillStyle = isKey ? '#56b6c2' : isStr ? '#98c379' : isNum ? '#e0a060' : '#c8ccd4';
                const t = fit(ctx, p, Math.max(10, x + w - cx)); ctx.fillText(t, cx, y + i * lh); cx += ctx.measureText(t).width; if (cx > x + w) break;
            }
        });
        if (lines.length > maxLines) { ctx.fillStyle = '#6a6a6a'; ctx.fillText(`…（还有 ${lines.length - maxLines} 行）`, x, y + maxLines * lh); }
        return lines.length;
    }
    const wait = (c, B, what) => LM.draw.text(c, [{ text: `        ${what}${B ? `：${B.k} / ${B.jobs.length}` : '（滚动到本步时开始计算）'}`, color: LM.C.gray }], { px: 12 });
    return { CAT, DOM, CONN, ALWAYS, init, analyze, strategies, allRow, newBench, benchWork, MODES, MNAME, MCOL, mono, fit, drawJSON, wait };
})();

PipelineLab.mount({
    paramSteps: { query: ['scale', 'def', 'bm25', 'dense', 'route', 'defer', 'args', 'action', 'prod'], docZh: ['scale', 'bm25', 'defer'], k: ['scale', 'defer'], mApps: ['route'], connected: ['scale', 'bm25', 'dense', 'route', 'defer'], fmt: ['def'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '19 · 成千上万个工具：检索、路由与按需加载',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 19,
    css: LM.CSS + `.pl-ctrl[data-param="query"] input[type=text]{width:26em}.pl-mini[data-param="query"] input[type=text]{width:30em;max-width:100%}`,
    tagColors: { '规模': 'rust', '定义': 'gray', '检索': 'blue', '评测': 'olive', '路由': 'teal', '加载': 'green', '参数': 'magenta', '决策': 'indigo', '生产版': 'gray' },

    params: [
        { id: 'query', label: '你的请求', type: 'text', value: '在虚幻引擎里把 PointLight_3 的强度改成 8000', alt: 'export the selected model to glb in blender', affects: 'param', desc: '随便输入一个要 Agent 做的事（中文、英文都行），例如「飞书群里通知大家今晚 8 点停服维护」「把 TAPD 上这个 bug 指派给小王」「compress all the png textures」。下面每一步都对它实时计算。' },
        { id: 'docZh', label: '描述里也写中文说法', value: true, affects: 'param', desc: '关掉：工具描述只有英文（真实 MCP 服务器大多如此），中文请求的词法检索会失效；打开：每个工具的检索文档里多了中文叫法（相当于工具作者写了中文例句）。' },
        { id: 'k', label: '放进上下文的工具数', options: [3, 5, 10, 20], value: 5, affects: 'param', desc: '检索后把前 k 个工具的完整定义交给模型。k 越大越不容易漏，但 token 越多、模型越容易选错。' },
        { id: 'mApps', label: '两级路由保留的应用数', options: [1, 2, 3, 5], value: 3, affects: 'param', desc: '两级路由先挑 m 个应用（MCP 服务器），再只在它们的工具里找。m 小 = 省事，但第一级挑错了就救不回来。' },
        { id: 'connected', label: '已连接的应用', options: [{ value: 'all', label: '全部 125 个' }, { value: 'game', label: 'Unreal + Unity + P4 + 飞书' }, { value: 'dcc', label: 'Blender + Maya + PS + Substance' }, { value: 'dev', label: 'GitHub + Git + Jira + Slack + Docker + K8s' }, { value: 'ops', label: 'TAPD + Steam + 数据 + 客服' }], value: 'all', affects: 'param', desc: '真实的 Agent 只能用「用户连上的服务器」里的工具——这是最便宜也最有效的过滤。切到某一组看同一个请求的结果怎么变。' },
        { id: 'fmt', label: '工具定义格式', options: [{ value: 'anthropic', label: 'Anthropic' }, { value: 'openai', label: 'OpenAI' }, { value: 'mcp', label: 'MCP tools/list' }], value: 'anthropic', affects: 'param', desc: '同一个工具在三种协议里的写法（字段名不同，内容相同）。' },
    ],

    stages: [
        { id: 'cat', lifetime: 'resident', run(S) { S.C = P19.init(); } },
        { id: 'tok', lifetime: 'resident', lazy: true, run(S) { const B = S.C.B, per = Int32Array.from(S.C.R.tools, (t) => TK.countTokens(B, TK.toolJSON(t))); let all = 0; for (const v of per) all += v; S.tok = { per, all, sys: TK.countTokens(B, TK.SYSTEM) }; } },
        { id: 'q', lifetime: 'param', run(S) { S.Q = P19.analyze(S.C, S); S.ST = P19.strategies(S.C, S, S.Q); } },
        { id: 'bench', lifetime: 'history', lazy: true, run(S) { if (!S.bn) S.bn = P19.newBench(S.C); P19.benchWork(S.C, S.bn, 30); } },
    ],

    intro: {
        title: '工具多到放不进一次请求时，Harness 要先替模型「找工具」',
        flow: ['规模：两千个工具的定义有多少 token', '一个工具定义长什么样', '词法检索 BM25', '向量检索', '评测召回率', '两级路由：先选应用', '按需加载：工具搜索 / Skills / 代码即工具', '参数抽取与校验', '答 / 调工具 / 写代码 / 点界面 / 先问'],
        html: LM.navHtml('19') + `
            <p>第 16–18 篇的模型只有两三个工具，全部写在系统提示里（「可用:看去拿放」）——这是教学版。真实的 Agent 面对的是<strong>工具目录</strong>：一个编程 Agent 接了 Git、GitHub、Jira、Slack、Docker、K8s；一个游戏团队的 Agent 还要接 Unreal、Unity、Perforce、Blender、Photoshop、飞书、Steam 后台、运营数据平台……每个都是一个 MCP 服务器，每个服务器几十到上百个工具。Anthropic 的工程博客举过例子：只接 5 个 MCP 服务器，工具定义就占了约 5.5 万 token；Zapier 一类平台宣称可以接入 8000 多个应用、3 万多个动作。</p>
            <p>全部塞进上下文既贵又不准：RAG-MCP 的实验里，把所有工具描述都给模型时选对工具的比例只有 13.62%，先检索再给只有相关的几个时是 43.13%；Anthropic 在自己的 MCP 评测上开启「工具搜索」后，Opus 4 从 49% 提到 74%，Opus 4.5 从 79.5% 提到 88.1%。所以工具一多，Harness 就要先做一次<strong>检索</strong>，只把最相关的几个定义交给模型，必要时让模型自己去搜。</p>
            <p>本篇用一个 <strong>125 个应用、1590 个工具</strong>的目录（按真实软件的 API 结构整理，JSON Schema 齐全）把这条流水线完整跑一遍：词法检索、离线训练的向量检索、混合排序、两级路由、按需加载的 token 账、参数抽取与校验、以及「这件事到底该不该调工具」。<strong>在顶栏输入任何请求</strong>，下面每一步都对它实时计算。最后一步之前的所有选择都发生在模型之外——这就是 Harness 的工作。</p>`,
    },

    steps: [
        {
            id: 'scale', title: '规模：两千个工具的定义放不进一次请求', tag: '规模',
            desc: '每个工具交给模型时都是一段 JSON：名字、描述、参数的 JSON Schema。单个不长，但乘以数量就很可观。下面用一个 16k 词表的字节级 BPE（第 06 篇的算法，在本系列讲义文本上训练）实数本页目录里每个工具定义的 token 数。真实的工具描述通常比本页的长得多（要写清楚什么时候用、参数怎么填、给例子），每个几百 token 很常见；API 还会额外加一段工具使用说明。',
            formula: ['tokens(全部) = Σ_i tokens(def_i) ≈ N × 每个定义的平均 token      // 每次请求都要付，模型每一步都要「读」一遍'],
            code: { title: 'C++ · ToolDef / toAnthropicJson / countTokens（BPE 计数）', src: TK_CPP.tooldef + '\n' + TK_CPP.bpecount },
            previews: [
                {
                    id: 'scaleBars', needs: ['tok'], label: '全部工具定义 vs 常见的上下文窗口', sub: '本页目录的实际 token 数（BPE 实数）· 「检索后」= 只放顶栏 k 个检索出的工具', large: true, size: [720, 190],
                    render: (c, S) => { if (!S.tok) return P19.wait(c, null, '正在数 token'); const topk = S.ST.topk.total; LM.draw.bars(c, [S.tok.all / 1000, 1000, 200, 128, topk / 1000], [`全部 ${P19.CAT.tools.length} 个工具定义`, '1M 上下文', '200K 上下文', '128K 上下文', `检索后只放 ${S.k} 个`], { horizontal: true, colors: [LM.C.red, '#3a3f46', '#3a3f46', '#3a3f46', LM.C.green], digits: 1, valueSuffix: 'K token' }); },
                },
                {
                    id: 'perTool', needs: ['tok'], label: '每个工具定义的 token 数', sub: '直方图 · 本页的描述偏短；真实描述常是这里的 2–5 倍', size: [360, 200],
                    render: (c, S) => { if (!S.tok) return P19.wait(c, null, '正在数 token'); const bins = new Array(12).fill(0); for (const v of S.tok.per) bins[Math.min(11, Math.floor(v / 25))]++; LM.draw.bars(c, bins, bins.map((_, i) => (i === 11 ? '275+' : `${i * 25}`)), { colors: [LM.C.blue], digits: 0, title: `平均 ${(S.tok.all / S.tok.per.length).toFixed(0)} token / 个` }); },
                },
                {
                    id: 'perDomain', label: '目录里的工具按领域分', sub: '125 个应用；云平台和大型创作软件（UE、Unity、Blender）的工具最多', size: [360, 200],
                    render: (c, S) => { const cnt = {}; for (const d of S.C.dom) cnt[d] = (cnt[d] || 0) + 1; const ks = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a]); LM.draw.bars(c, ks.map((k) => cnt[k]), ks.map((k) => P19.DOM[k][0]), { horizontal: true, colors: ks.map((k) => P19.DOM[k][1]), digits: 0, labelPx: 11 }); },
                },
            ],
            notes: '本页目录全部定义加起来约 17 万 token——已经接近一个 200K 上下文的上限，而且每一轮对话都要重新付一次（能命中前缀缓存的部分会便宜很多，但仍然占着上下文）。把 <b>放进上下文的工具数</b> 改成 20，看「检索后」那一条怎么变：即使放 20 个，也只是全部的 1–2%。',
            warn: '工具定义多了，问题不只是钱：模型要在一大堆相似的工具里挑（Unity 的 create_prefab 和 Unreal 的 create_blueprint、TAPD 的 create_bug 和 Jira 的 create_issue），越多越容易混淆——这正是 RAG-MCP 与 Anthropic 的实验里准确率掉下来的原因。',
        },
        {
            id: 'def', title: '一个工具定义：模型看到的就是这段 JSON', tag: '定义',
            desc: '下面是对你的请求检索出的第一名工具，按顶栏选的协议写成 JSON。三种协议的内容一样：<code>name</code>（Anthropic / OpenAI 要求只含字母数字和 <code>_-</code>，所以 MCP 客户端把服务器名拼进去：<code>mcp__unreal__set_light_intensity</code>）、<code>description</code>（模型选工具时几乎只看它）、参数的 JSON Schema。MCP 还多了 <code>annotations</code>：<code>readOnlyHint</code> / <code>destructiveHint</code> / <code>idempotentHint</code> / <code>openWorldHint</code>——给 Harness 看的提示（例如只读工具可以不弹确认框），但<strong>不是</strong>安全边界。',
            formula: ['Anthropic: { name, description, input_schema }      OpenAI: { type: "function", name, description, parameters }      MCP: { name, title, description, inputSchema, annotations }'],
            code: { title: 'C++ · ToolDef → JSON（见上一步）', src: '// @nocompile\n// 见上一步的 ToolDef / toAnthropicJson：三种格式只是字段名不同\n// MCP 的 tools/list 结果：{ "tools": [ { "name": …, "description": …, "inputSchema": …, "annotations": { "readOnlyHint": … } } ], "nextCursor": "…" }' },
            previews: [
                {
                    id: 'defJson', label: '检索第一名的工具定义（随你的请求变化）', sub: '键 = 青 · 字符串 = 绿 · 数字 / 布尔 = 橙', large: true, size: [720, 330],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c), t = S.Q.pick, obj = S.fmt === 'openai' ? TK.openaiTool(t) : S.fmt === 'mcp' ? TK.mcpTool(t) : TK.anthropicTool(t), tok = TK.countTokens(S.C.B, JSON.stringify(obj));
                        ctx.font = LM.FONT(12, true); ctx.fillStyle = LM.C.text; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
                        ctx.fillText(P19.fit(ctx, `${t.mcpName} · ${tok} token（紧凑 JSON）· 应用：${S.C.R.apps[S.C.R.toolApp[t.i]].name}`, w - 16), 8, 6);
                        const an = t.annotations; ctx.font = LM.FONT(11); ctx.fillStyle = an.readOnlyHint ? LM.C.green : an.destructiveHint ? LM.C.red : LM.C.orange; ctx.fillText(an.readOnlyHint ? '只读（readOnlyHint）：Harness 可以不弹确认' : an.destructiveHint ? '有破坏性（destructiveHint）：执行前要确认' : '会修改状态：通常需要确认或在沙箱里执行', 8, 24);
                        P19.drawJSON(ctx, obj, 8, 44, w - 16, 17, 11.5);
                    },
                },
            ],
            notes: '换几个请求看看不同工具的参数：「把 TAPD 上这个 bug 指派给小王」→ <code>assign_bug</code>（必填 id、assignee）；「重启一下线上的 api 服务」→ 多半是 K8s 的 <code>restart_deployment</code>（必填 name、namespace）。必填参数决定了第 08 步模型要从请求里拿到什么、拿不到就得问。',
            warn: '描述就是给模型读的文档：「Create a new pull request.」比「PR tool」好得多；写清楚「什么时候用、什么时候不该用」能直接减少选错。另外，描述来自第三方服务器，是<strong>不可信输入</strong>——曾有人演示过在工具描述里藏指令（tool poisoning），让模型泄露数据。',
        },
        {
            id: 'bm25', title: '词法检索 BM25：按字面找', tag: '检索',
            desc: '最朴素的办法：把每个工具的名字和描述当成一篇「文档」，用搜索引擎的老方法 <strong>BM25</strong> 给请求打分——请求里的词在文档里出现得越多、这个词在整个目录里越稀有，分越高；文档越长分数越被压低。分词规则：英文词拆开 snake_case / camelCase 并去掉复数，中文取单字和相邻两字。它的弱点一眼可见：请求是中文、描述是英文时，几乎没有词能对上；「调暗」和「intensity」字面毫无关系。打开顶栏的 <b>描述里也写中文说法</b>，相当于工具作者把常见中文叫法写进了描述。',
            formula: [
                'score(d, q) = Σ_{t ∈ q} idf(t) · tf(t, d) · (k1 + 1) / (tf(t, d) + k1 · (1 − b + b · |d| / avg|d|))      k1 = 1.2, b = 0.75',
                'idf(t) = ln(1 + (N − n_t + 0.5) / (n_t + 0.5))      // n_t = 含有词 t 的文档数',
            ],
            code: { title: 'C++ · buildBM25 / scoreBM25', src: TK_CPP.bm25 },
            previews: [
                {
                    id: 'bmTop', label: 'BM25 的前 10 名（随你的请求实时变化）', sub: '每行：名次 · 工具 · 分数条 · 命中的词（括号里是这个词贡献的分数）· 绿 = 第一名', large: true, size: [720, 300],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c), r = S.Q.res, I = S.docZh ? S.C.R.bmZh : S.C.R.bmEn, C = LM.C;
                        ctx.font = LM.FONT(12); ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.fillStyle = C.gray;
                        ctx.fillText(P19.fit(ctx, `请求的词项：${r.bm.qt.map((t) => (I.post.has(t) ? t : `${t}✗`)).join('  ')}   （✗ = 目录里没有这个词）`, w - 16), 8, 6);
                        const top = r.top.bm25.slice(0, 10);
                        if (!top.length) { ctx.fillStyle = C.red; ctx.fillText('没有任何工具与这个请求有共同的词——BM25 完全找不到（试试打开「描述里也写中文说法」）', 8, 40); return; }
                        const mx = r.bm.score[top[0]] || 1;
                        top.forEach((i, k) => {
                            const y = 30 + k * 26, t = S.C.R.tools[i], sc = r.bm.score[i];
                            ctx.fillStyle = k === 0 ? C.green : C.text; ctx.font = LM.FONT(12); ctx.fillText(`${k + 1}.`, 8, y + 3); ctx.fillText(P19.fit(ctx, t.mcpName.replace('mcp__', ''), 250), 30, y + 3);
                            ctx.fillStyle = '#1b2733'; ctx.fillRect(290, y + 2, 140, 16); ctx.fillStyle = k === 0 ? C.green : C.blue; ctx.fillRect(290, y + 2, (140 * sc) / mx, 16);
                            ctx.fillStyle = C.text; ctx.fillText(sc.toFixed(1), 436, y + 3);
                            const hits = (r.bm.hits.get(i) || []).sort((a, b) => b[1] - a[1]).map(([tt, s]) => `${tt}(${s.toFixed(1)})`).join(' ');
                            ctx.fillStyle = C.gray; ctx.fillText(P19.fit(ctx, hits, w - 486), 478, y + 3);
                        });
                    },
                },
            ],
            notes: '默认请求里「PointLight」「8000」这些词在目录里不常见，BM25 主要靠「强度」「虚幻」这样的中文叫法（打开了中文说法时）。关掉 <b>描述里也写中文说法</b> 再看：几乎只剩英文词能匹配，中文请求的结果基本是噪声。离线评测（第 05 步）里，只有英文描述时 BM25 的 Recall@10 不到 60%，加上中文说法后到 95%。',
            warn: 'BM25 的长处是可解释、不用训练、对专有名词（PointLight_3、/Game/Characters、错误码）很准，所以生产系统几乎都把它和向量检索一起用（第 05 步的混合检索）。',
        },
        {
            id: 'dense', title: '向量检索：按意思找', tag: '检索',
            desc: '把请求和每个工具的描述都编码成一个 64 维的单位向量，用余弦相似度排序——意思相近的说法即使字面不同也会靠近。本页的编码器很小（约 100 万参数）：把文本切成哈希特征（英文词、词内三字母、中文字、相邻两字，2<sup>14</sup> 个桶），查嵌入表求和 → tanh → 线性层 → 归一化。它在离线用<strong>对比学习</strong>训练：一批 256 个（请求，工具）对，第 i 个请求要和第 i 个工具最像、和同批其他 255 个都不像（批内负例）；一半的批次按应用成组，让同一个应用里的工具互为「难负例」；另外加了一个「请求 → 应用描述」的目标（第 06 步用）。训练数据是由模板生成的约 2.3 万条中英文请求；有 151 个工具<strong>从没出现在训练里</strong>，用来检验「新加一个工具，不重新训练也能被找到」。',
            formula: [
                'z = normalize(W₁ · tanh(Σ_f w_f · E[hash(f)] / ‖w‖ + b₀) + b₁)      sim(q, d) = z_q · z_d',
                'L = −(1/B) Σ_i log softmax_j(z_{q_i} · z_{d_j} / τ)_i      τ = 0.05，B = 256（批内负例 InfoNCE）',
            ],
            code: { title: 'C++ · hashFeatures / embedText / infoNCE', src: TK_CPP.encoder },
            previews: [
                {
                    id: 'denseMap', label: '1590 个工具的「地图」（t-SNE）与你的请求', sub: '每个点 = 一个工具，颜色 = 领域 · 白色星 = 你的请求（位置 = 最近 8 个工具坐标按相似度加权的平均）· 连线 = 前 10 名', large: true, size: [720, 400],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c), M = S.C.map, R = S.C.R, top = S.Q.res.top.dense.slice(0, 10), pad = 20, X = (v) => pad + ((v + 1) / 2) * (w - 190 - 2 * pad), Y = (v) => pad + ((1 - v) / 2) * (h - 2 * pad);
                        for (let i = 0; i < R.tools.length; i++) { ctx.fillStyle = P19.DOM[S.C.dom[i]][1]; ctx.globalAlpha = 0.8; ctx.fillRect(X(M[2 * i]) - 2, Y(M[2 * i + 1]) - 2, 4, 4); }
                        ctx.globalAlpha = 1; const qx = X(S.Q.qx), qy = Y(S.Q.qy);
                        ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 1;
                        top.forEach((i) => { ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(X(M[2 * i]), Y(M[2 * i + 1])); ctx.stroke(); ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X(M[2 * i]), Y(M[2 * i + 1]), 3.5, 0, 7); ctx.fill(); });
                        ctx.fillStyle = '#fff'; ctx.font = '18px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', qx, qy);
                        ctx.font = LM.FONT(11); ctx.textAlign = 'left'; let ly = 12;
                        for (const [k, [nm, col]] of Object.entries(P19.DOM)) { ctx.fillStyle = col; ctx.fillRect(w - 180, ly - 4, 10, 10); ctx.fillStyle = LM.C.text; ctx.fillText(nm, w - 164, ly + 1); ly += 17; if (k === 'gui') break; }
                        ctx.fillStyle = LM.C.gray; ctx.fillText('同一个应用的工具挤在一起；', w - 180, ly + 12); ctx.fillText('不同应用的同类操作（建任务、', w - 180, ly + 28); ctx.fillText('发消息、导出文件）也会靠近', w - 180, ly + 44);
                    },
                },
                {
                    id: 'denseTop', label: '向量检索的前 10 名', sub: '余弦相似度 · 绿 = 第一名', size: [360, 280],
                    render: (c, S) => { const r = S.Q.res, top = r.top.dense.slice(0, 10); LM.draw.bars(c, top.map((i) => r.dense[i]), top.map((i) => { const t = S.C.R.tools[i], s = `${t.name} · ${t.ns}`; return s.length > 30 ? s.slice(0, 29) + '…' : s; }), { horizontal: true, colors: top.map((_, k) => (k === 0 ? LM.C.green : LM.C.blue)), digits: 2, range: [0, 1] }); },
                },
                {
                    id: 'trainCurve', label: '离线训练曲线（tools/tk_train.js）', sub: '蓝 = 请求 → 工具的对比损失 · 橙 = 请求 → 应用的损失 · 对数纵轴', size: [360, 280],
                    render: (c, S) => { const cv = TK_W.train.curve; PL.draw.plot(c, { x: [0, TK_W.train.steps], y: [0.005, 6], logY: true, xLabel: '训练步', series: [{ data: cv.map((p) => p[1]), x: cv.map((p) => p[0]), color: LM.C.blue, width: 1.5, label: '工具' }, { data: cv.map((p) => p[2]), x: cv.map((p) => p[0]), color: LM.C.orange, width: 1.5, label: '应用' }] }); },
                },
            ],
            notes: `默认请求的第一名是 <code>set_light_intensity</code>，前几名都是 Unreal 的灯光工具（颜色、衰减半径、阴影）——在地图上它们挤在一起。试试「把 Hero.fbx 导入到 /Game/Characters」：Blender、Unreal、Unity 的导入工具都会出现在前几名，因为这句话本身就有歧义（/Game/ 路径其实暗示 Unreal）——第 06 步的「已连接的应用」能消掉这种歧义。训练用了 ${TK_W.train.steps} 步、${(TK_W.train.sec / 60).toFixed(0)} 分钟（单线程 JS），嵌入表量化成 int8 后检索质量几乎不变。`,
            warn: '这个编码器只认识训练数据里出现过的词：「给主角加一个刚体」能找到 Unity 的 add_component，是因为我们在描述里补了「加刚体」；换成「加个碰撞盒」它就可能找不到。真实系统用的是在海量多语言文本上预训练的嵌入模型（几亿参数），这类同义词它天生就懂——但「描述写清楚、给例句」依然是最有效的改进。',
        },
        {
            id: 'eval', title: '评测：漏没漏比排第几更重要', tag: '评测',
            desc: '检索只负责缩小范围：把前 k 个交给模型，由模型读完定义再做最后的选择。所以最关键的指标是 <strong>Recall@k</strong>——正确的工具有没有进前 k 名。下面在你的浏览器里实时评测三组留出请求（每组 300 条）：① 训练时见过的说法模板、但没见过这句话；② 训练时<strong>没见过的模板</strong>（「麻烦…」「…那边…」）；③ <strong>从没训练过的 151 个工具</strong>（只有描述）。对比六种做法：两种 BM25、向量、两种混合（加权的倒数排名融合 RRF）、两级路由。',
            formula: [
                'Recall@k = P(正确工具 ∈ 前 k 名)      RRF(d) = Σ_r w_r / (60 + rank_r(d))      // 只用名次融合，不用管两路分数的量纲',
            ],
            code: { title: 'C++ · rrfFuse / topK', src: TK_CPP.hybrid },
            previews: [
                {
                    id: 'evalBars', needs: ['bench'], live: true, label: '没见过的说法模板：Recall@1 / @5 / @10（页面里实时评测）', sub: '每组三根柱：浅 = @1、中 = @5、深 = @10', large: true, size: [720, 260],
                    render: (c, S) => {
                        const Bn = S.bn; if (!Bn || !Bn.res.held.dense) return P19.wait(c, Bn, '正在评测');
                        const { ctx, w, h } = PL.draw.ctx(c), T = Bn.res.held, x0 = 40, gw = (w - x0 - 10) / P19.MODES.length, Y = (v) => h - 50 - v * (h - 80);
                        ctx.strokeStyle = '#2a3038'; ctx.font = LM.FONT(11); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                        for (const v of [0, 0.25, 0.5, 0.75, 1]) { ctx.beginPath(); ctx.moveTo(x0, Y(v)); ctx.lineTo(w - 10, Y(v)); ctx.stroke(); ctx.fillText(`${v * 100}%`, x0 - 4, Y(v)); }
                        P19.MODES.forEach((m, j) => {
                            const a = T[m]; if (!a) return; const xs = x0 + j * gw + gw * 0.12, bw = gw * 0.24;
                            [['r1', 0.45], ['r5', 0.7], ['r10', 1]].forEach(([k, al], q) => { const v = a[k] / a.n; ctx.globalAlpha = al; ctx.fillStyle = P19.MCOL[m]; ctx.fillRect(xs + q * bw, Y(v), bw - 2, Y(0) - Y(v)); ctx.globalAlpha = 1; ctx.fillStyle = LM.C.text; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.font = LM.FONT(10); ctx.fillText((100 * v).toFixed(0), xs + q * bw + bw / 2 - 1, Y(v) - 1); });
                            ctx.font = LM.FONT(11); ctx.fillStyle = P19.MCOL[m]; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            const nm = P19.MNAME[m], cut = nm.indexOf('（'); ctx.fillText(cut > 0 ? nm.slice(0, cut) : nm, x0 + j * gw + gw / 2, h - 44); if (cut > 0) { ctx.fillStyle = LM.C.gray; ctx.fillText(P19.fit(ctx, nm.slice(cut), gw - 4), x0 + j * gw + gw / 2, h - 28); }
                        });
                        ctx.fillStyle = LM.C.gray; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`${Bn.k} / ${Bn.jobs.length} 条请求`, x0 + 4, 4);
                    },
                },
                {
                    id: 'evalSets', needs: ['bench'], live: true, label: '三组请求上的 Recall@10', sub: '见过的模板 / 没见过的模板 / 从没训练过的工具', size: [720, 200],
                    render: (c, S) => {
                        const Bn = S.bn; if (!Bn || !Bn.res.new.dense) return P19.wait(c, Bn, '正在评测');
                        const rows = [], labels = [], cols = [];
                        for (const m of ['bm25en', 'dense', 'hybZh', 'router']) for (const [k, nm] of [['seen', '见过的模板'], ['held', '没见过的模板'], ['new', '新工具']]) { const a = Bn.res[k][m]; rows.push(a ? a.r10 / a.n : 0); labels.push(`${P19.MNAME[m].split('（')[0]} · ${nm}`); cols.push(P19.MCOL[m]); }
                        LM.draw.bars(c, rows, labels, { horizontal: true, colors: cols, range: [0, 1], digits: 2, labelPx: 10 });
                    },
                },
            ],
            notes: '这组数字值得细看：① 只有英文描述时，BM25 的 Recall@10 不到六成；② 向量检索的 Recall@10 在没见过的模板上约 99%，在从没训练过的工具上约 94%——加新工具只要把描述编码进索引，不用重新训练；③ <b>混合不一定更好</b>：英文描述 + BM25 时，BM25 那一路给中文请求的全是噪声，硬融合进来反而把第一名挤掉（R@1 明显低于纯向量）；描述里写了中文说法后，混合在新工具上的 Recall@10 最高（约 98%）。离线全量评测（每组 600–1200 条）的结果与这里一致，见 tools_weights.js 的 train.search。',
            warn: '这里的评测请求是由模板生成的，比真实用户的说法规整得多——真实系统要用线上日志里的真实请求（人工标注应该调哪个工具）来评测，而且要分「检索漏了」和「检索到了但模型选错了」两种错误。',
        },
        {
            id: 'route', title: '两级路由：先选应用，再选工具', tag: '路由',
            desc: '另一种缩小范围的办法：先只看 125 个应用（MCP 服务器）各自的一句话描述，挑出最相关的 m 个，再在它们的工具里找。这对应真实系统里的几种做法：只加载用户连上的服务器、按服务器分组的工具清单、Skills 先只看名字和一句话描述。好处是第一级只有 125 个候选、描述短、可解释；坏处是<strong>第一级挑错了，后面就救不回来</strong>。另一种「路由」根本不需要模型：顶栏的 <b>已连接的应用</b>——用户只连了这几个服务器，其他的工具一开始就不存在。',
            formula: ['P(app | q) = softmax_a(25 · z_q · z_{app})      top-m 应用 → 只在它们的工具里按 z_q · z_tool 排序'],
            code: { title: 'C++ · routeTwoStage', src: TK_CPP.router },
            previews: [
                {
                    id: 'appProbs', label: '第一级：125 个应用里，你的请求最可能属于哪几个', sub: '绿 = 被保留的前 m 个（顶栏「两级路由保留的应用数」）', large: true, size: [360, 280],
                    render: (c, S) => { const r = S.Q.res, top = r.appTop.slice(0, 10); LM.draw.bars(c, top.map((i) => r.appP[i]), top.map((i) => S.C.R.apps[i].name), { horizontal: true, colors: top.map((_, k) => (k < S.mApps ? LM.C.green : '#3a3f46')), range: [0, 1], digits: 2 }); },
                },
                {
                    id: 'routeCompare', label: '同一个请求：一级检索 vs 两级路由', sub: '左 = 在全部工具里排序 · 右 = 只在前 m 个应用里排序', size: [360, 280],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c), r = S.Q.res, R = S.C.R, C = LM.C, col = (w - 24) / 2;
                        ctx.font = LM.FONT(12, true); ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.fillStyle = C.blue; ctx.fillText('一级（全部工具）', 8, 6); ctx.fillStyle = C.orange; ctx.fillText(`两级（前 ${S.mApps} 个应用）`, 16 + col, 6);
                        ctx.font = LM.FONT(11);
                        r.top.dense.slice(0, 10).forEach((i, k) => { ctx.fillStyle = i === r.top.router[0] ? C.green : C.text; ctx.fillText(P19.fit(ctx, `${k + 1}. ${R.tools[i].mcpName.replace('mcp__', '')}`, col), 8, 28 + k * 22); });
                        if (!r.top.router.length) { ctx.fillStyle = C.red; ctx.fillText('（选中的应用里没有工具）', 16 + col, 28); }
                        r.top.router.slice(0, 10).forEach((i, k) => { ctx.fillStyle = i === r.top.dense[0] ? C.green : C.text; ctx.fillText(P19.fit(ctx, `${k + 1}. ${R.tools[i].mcpName.replace('mcp__', '')}`, col), 16 + col, 28 + k * 22); });
                        ctx.fillStyle = C.gray; ctx.fillText('绿 = 两边的第一名一致', 8, 256);
                    },
                },
                {
                    id: 'routeAbl', label: '第一级要不要专门训练：离线消融（没见过的说法模板）', sub: '同样的编码器结构；左 = 只训练「请求 → 工具」，右 = 再加「请求 → 应用描述」的目标', size: [720, 150],
                    render: (c, S) => { const a = TK_W.train.ablation.noAppLoss.router.held, b = TK_W.train.search.router.held, d = TK_W.train.search.dense.held; LM.draw.bars(c, [a.app1, b.app1, a.app3, b.app3, a.r10, b.r10, d.r10], ['不训练应用 · 应用 top-1', '训练应用 · 应用 top-1', '不训练应用 · 应用 top-3', '训练应用 · 应用 top-3', '不训练应用 · 两级 Recall@10', '训练应用 · 两级 Recall@10', '对照：一级向量检索 Recall@10'], { horizontal: true, colors: ['#8a8f98', LM.C.green, '#8a8f98', LM.C.green, '#8a8f98', LM.C.orange, LM.C.blue], range: [0, 1], digits: 2, labelPx: 10 }); },
                },
            ],
            notes: '试试「重启一下线上的 api 服务」：一级检索的第一名是 K8s 的 <code>restart_deployment</code>，但应用路由把它判给了不相干的应用——两级路由就彻底找不到了。这就是误差传播。把 <b>已连接的应用</b> 切到「GitHub + Git + Jira + Slack + Docker + K8s」，范围一下子缩小到 7 个应用，结果立刻变得干净。离线消融：第一级不专门训练时，应用 top-3 只有约 65%；加上「请求 → 应用」的目标后约 95%，两级路由的 Recall@10 也从约 65% 提到约 95%（仍略低于一级检索）。',
            warn: '应用（服务器）描述同样是给模型和检索器看的：「HubSpot API」这种空泛的描述会让所有 SaaS 应用彼此相像——我们训练时就踩过这个坑（描述里统一写了 “API”，结果请求里一出现 api，路由就乱指）。描述里要写这个服务器管哪些对象、能做什么。',
        },
        {
            id: 'defer', title: '按需加载：工具搜索、Skills、代码即工具', tag: '加载',
            desc: '检索出来的工具怎么交给模型？几种做法的 token 账很不一样（下面是对你的请求实算的前两轮输入）：<strong>全部放进去</strong>；<strong>检索 top-k</strong>（Harness 在调用模型之前就检索好）；<strong>两级</strong>（先给应用清单，模型选了应用再给它的工具）；<strong>工具搜索</strong>（Anthropic 的 tool search / Claude Code 的 MCP 工具搜索：工具标成 defer_loading，模型一开始只看到一个 <code>tool_search</code> 和少数常驻工具，需要时自己搜，搜到的完整定义再追加进上下文）；<strong>代码即工具</strong>（Anthropic「Code execution with MCP」：每个工具是沙箱里的一个代码文件，模型先看文件树，只读需要的接口，然后写代码调用）。Skills 是同一个思路用在「操作说明」上：平时只放名字和一句话描述（每个约 100 token），用到时才读完整的 SKILL.md。',
            formula: [
                '输入 token（第 1 轮）= 系统提示 + 工具相关的部分 + 请求      第 2 轮 = 第 1 轮 + 模型的调用 + 新加载的定义',
                'Anthropic 报告：工具搜索使 token 减少约 85%（77.2K → 8.7K）；代码执行 + MCP 的一个例子从 15 万 token 降到 2 千（98.7%）',
            ],
            code: { title: 'C++ · hybridSearch / handleToolSearch', src: TK_CPP.toolsearch },
            previews: [
                {
                    id: 'stratTokens', needs: ['tok'], label: '五种加载策略：前两轮的输入 token（对你的请求实算）', sub: '每一段 = prompt 的一部分 · 灰 = 系统提示 + 请求 · 红 = 工具定义 · 蓝 = 清单 / 文件树 · 绿 = 第 2 轮新加载的定义', large: true, size: [720, 250],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c), ST = Object.assign({ all: P19.allRow(S) }, S.ST), C = LM.C, names = { all: '全部放进去', topk: `检索 top-${S.k}`, router: '两级（先选应用）', search: '工具搜索（按需加载）', code: '代码即工具' };
                        const rows = Object.keys(names).map((k) => [k, ST[k]]).filter(([, v]) => v), mx = Math.max(...rows.map(([, v]) => v.total)), lw = 150, W = w - lw - 130, logX = (v) => (Math.log10(1 + v) / Math.log10(1 + mx)) * W;
                        rows.forEach(([k, v], j) => {
                            const y = 18 + j * 44; ctx.font = LM.FONT(12); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillStyle = C.text; ctx.fillText(names[k], lw - 8, y + 9);
                            let acc = 0; const segs = [{ part: '请求', tokens: v.user }].concat(v.parts);
                            for (const p of segs) { const x0 = lw + logX(acc), x1 = lw + logX(acc + p.tokens); ctx.fillStyle = /第 2 轮/.test(p.part) ? C.green : /定义/.test(p.part) ? C.red : /清单|文件树|tool_search|常驻/.test(p.part) ? C.blue : '#56606b'; ctx.fillRect(x0, y, Math.max(1, x1 - x0 - 1), 18); acc += p.tokens; }
                            ctx.textAlign = 'left'; ctx.fillStyle = C.text; ctx.fillText(`${v.total.toLocaleString()} token`, lw + logX(acc) + 6, y + 9);
                            ctx.font = LM.FONT(10); ctx.fillStyle = C.gray; ctx.fillText(P19.fit(ctx, v.parts.map((p) => `${p.part} ${p.tokens}`).join(' · '), w - lw - 10), lw, y + 30);
                        });
                        if (!ST.all) { ctx.fillStyle = C.gray; ctx.font = LM.FONT(11); ctx.fillText('（「全部放进去」正在数 token…）', lw, 230); }
                        ctx.font = LM.FONT(10); ctx.fillStyle = C.dim; ctx.textAlign = 'right'; ctx.fillText('横轴为对数刻度', w - 8, 244);
                    },
                },
                {
                    id: 'searchTrace', label: '工具搜索的一次往返（消息是真实的结构，token 为实数）', sub: '① 请求（只带 tool_search + 3 个常驻工具）② 模型调用 tool_search ③ Harness 检索并返回完整定义 ④ 模型调用选中的工具', size: [720, 300],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c), B = S.C.B, C = LM.C, R = S.C.R, picked = S.Q.res.top.hybrid.slice(0, S.k).map((i) => R.tools[i]);
                        const always = P19.ALWAYS.map((id) => P19.CAT.byId.get(id));
                        const m1 = { system: TK.SYSTEM, tools: [TK.TOOL_SEARCH].concat(always.map(TK.anthropicTool)), messages: [{ role: 'user', content: S.query }] };
                        const m2 = { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_01', name: 'tool_search', input: { query: S.query, limit: S.k } }] };
                        const m3 = { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_01', content: picked.map(TK.anthropicTool) }] };
                        const m4 = { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_02', name: S.Q.pick.mcpName, input: S.Q.ex.args }] };
                        const rows = [['① 请求', m1, `tools = [tool_search, ${always.map((t) => t.name).join(', ')}] · 用户：${S.query}`], ['② 模型', m2, `tool_use tool_search({ query: "${S.query}", limit: ${S.k} })`], ['③ Harness', m3, `tool_result：${picked.map((t) => t.mcpName.replace('mcp__', '')).join(', ')} 的完整定义`], ['④ 模型', m4, `tool_use ${S.Q.pick.mcpName}(${JSON.stringify(S.Q.ex.args)})`]];
                        let total = 0;
                        rows.forEach(([who, msg, line], j) => {
                            const y = 10 + j * 64, tok = TK.countTokens(B, JSON.stringify(msg)); total += tok;
                            ctx.fillStyle = '#141a21'; ctx.fillRect(6, y, w - 12, 54); ctx.strokeStyle = j % 2 ? LM.C.yellow : LM.C.blue; ctx.strokeRect(6.5, y + 0.5, w - 13, 53);
                            ctx.font = LM.FONT(12, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = j % 2 ? LM.C.yellow : LM.C.blue; ctx.fillText(who, 14, y + 8);
                            ctx.font = LM.FONT(12); ctx.fillStyle = C.text; ctx.fillText(P19.fit(ctx, line, w - 200), 100, y + 8);
                            ctx.fillStyle = C.gray; ctx.fillText(`${tok.toLocaleString()} token · 累计 ${total.toLocaleString()}`, 100, y + 30);
                        });
                    },
                },
            ],
            notes: '对默认请求，「全部放进去」约 17 万 token，「工具搜索」两轮合计只有一两千——差两个数量级。代价是多一次往返（模型要先搜一次），以及检索漏了时模型可能根本不知道有那个工具；所以 Anthropic 建议把三五个最常用的工具常驻、其余延迟加载。「代码即工具」在一次要调用很多工具、或者结果很大（要在沙箱里先过滤）时最划算——第 21 篇会实测「逐个调用 vs 一段脚本」。',
            warn: '前缀缓存：系统提示和常驻的工具定义放在 prompt 最前面、保持不变，就能命中缓存（读缓存的价格大约是正常输入的十分之一）；按需加载的定义追加在后面，不会破坏前面的缓存。反过来，如果每次都按检索结果重排「系统提示里的工具列表」，缓存每次都失效。',
        },
        {
            id: 'args', title: '选中之后：参数从哪来、错了怎么办', tag: '参数',
            desc: '选中工具后，模型要按它的 JSON Schema 生成参数。真实系统里这一步由模型完成（可以用约束解码保证输出一定是合法 JSON，例如 OpenAI 的 strict 模式、Anthropic 的结构化输出），本页用一个<strong>规则版的抽取器</strong>演示：路径、名字、数字、枚举值、开关、括号里的三个数、引号里的文字、带下划线的对象名（PointLight_3）。然后 Harness 用 Schema <strong>校验</strong>参数：缺了必填参数、类型不对、不在枚举里——都不执行，而是把一条清楚的错误信息作为 tool_result 返回给模型（MCP 里是 <code>isError: true</code>），让它自己补上：再查一次（例如先 list 出所有灯）、或者问用户。',
            formula: ['validate(schema, args) = [] ⇒ 执行      否则 ⇒ tool_result { is_error: true, content: 「缺少必填参数 actor：先调用 list_light 找到名字，或问用户」}'],
            code: { title: 'C++ · Json / Schema / validate', src: TK_CPP.jsonschema },
            previews: [
                {
                    id: 'argTable', label: '选中工具的参数：从你的请求里能拿到什么', sub: '必填 = ★ · 绿 = 拿到了（后面是依据）· 红 = 缺的必填参数 · 灰 = 可选、没提到', large: true, size: [720, 300],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c), t = S.Q.pick, ex = S.Q.ex, P = t.inputSchema.properties, req = new Set(t.inputSchema.required), C = LM.C;
                        ctx.font = LM.FONT(12, true); ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.fillStyle = C.text; ctx.fillText(P19.fit(ctx, `${t.mcpName} — ${t.description}`, w - 16), 8, 6);
                        const keys = Object.keys(P).slice(0, 9);
                        keys.forEach((k, j) => {
                            const y = 32 + j * 22, p = P[k], ty = p.enum ? `enum(${p.enum.join('|')})` : p.type === 'array' ? `array<${(p.items && p.items.type) || 'any'}>` : p.type || 'any', has = k in ex.args;
                            ctx.font = P19.mono(12); ctx.fillStyle = has ? C.green : req.has(k) ? C.red : C.gray; ctx.fillText(`${req.has(k) ? '★' : ' '} ${k}`, 8, y);
                            ctx.fillStyle = C.gray; ctx.fillText(P19.fit(ctx, ty, 200), 170, y);
                            ctx.fillStyle = has ? C.green : req.has(k) ? C.red : C.dim; ctx.fillText(P19.fit(ctx, has ? `${JSON.stringify(ex.args[k])}   ← ${ex.how[k]}` : req.has(k) ? '缺（请求里没有）' : '—', w - 390), 380, y);
                        });
                        const y0 = 40 + keys.length * 22;
                        ctx.font = LM.FONT(12, true); ctx.fillStyle = ex.errors.length ? C.red : C.green; ctx.fillText(ex.errors.length ? `校验失败（${ex.errors.length} 处）→ 不执行，Harness 回给模型：` : '校验通过 → Harness 执行这次调用：', 8, y0);
                        ctx.font = P19.mono(11.5); ctx.fillStyle = C.text;
                        const msg = ex.errors.length ? JSON.stringify({ type: 'tool_result', is_error: true, content: `Invalid arguments: ${ex.errors.join('; ')}.${ex.missing.length ? ` Ask the user, or look up ${ex.missing.join(', ')} with a list/search tool first.` : ''}` }) : JSON.stringify({ jsonrpc: '2.0', id: 7, method: 'tools/call', params: { name: t.name, arguments: ex.args } });
                        const lines = []; for (let i = 0; i < msg.length; i += 92) lines.push(msg.slice(i, i + 92));
                        lines.slice(0, 5).forEach((ln, i) => ctx.fillText(ln, 8, y0 + 22 + i * 16));
                    },
                },
            ],
            notes: '默认请求里 <code>actor</code>（PointLight_3）和 <code>intensity</code>（8000）都拿到了，校验通过，Harness 发出 MCP 的 <code>tools/call</code>。改成「把灯光调暗一点」：没有具体的灯，也没有数值——校验失败，错误信息告诉模型「先 list 再改，或者问用户」。好的错误信息本身就是 Harness 设计的一部分：只说「invalid」的话，模型往往原样重试。',
            warn: '「调暗一点」「减半」这类相对的说法，规则抽取不了（需要先读当前值再算），这正是该交给模型（或者写一段代码，第 09 步）的部分。另外：参数是模型生成的，Harness 必须当作不可信输入处理——路径要限制在允许的目录里、SQL 要只读或参数化、命令行要转义。',
        },
        {
            id: 'action', title: '什么时候调工具、什么时候写代码、什么时候先问', tag: '决策',
            desc: '「选哪个工具」之前还有一个更大的问题：这件事该用什么<strong>动作</strong>做？① <strong>直接回答</strong>（「什么是 Draw Call」——不需要任何工具）；② <strong>调一个工具</strong>（「把 PointLight_3 的强度改成 8000」）；③ <strong>写一段代码</strong>（「把所有点光源的强度减半」——要遍历、要先读后算，逐个调用工具要几十上百次往返，一段脚本一次搞定）；④ <strong>操作界面</strong>（软件没有接口，只能截图 + 点鼠标，又慢又脆，最后的手段）；⑤ <strong>先问清楚</strong>（「帮我弄一下那个」）。下面是一个在模板数据上训练的小分类器对你的请求给出的概率——真实系统里这个判断由大模型自己做，Harness 负责把这几种动作都提供给它，并给每种动作设好成本和权限。',
            formula: ['逐个调用：往返次数 ≈ 1 + N（先列出，再对 N 个对象各调一次）      代码：1 次（循环在沙箱里跑）      界面：每一步 ≈ 一张截图（1456×819 约 1560 个图像 token）+ 一次点击'],
            code: { title: 'C++ · actionProbs', src: TK_CPP.action },
            previews: [
                {
                    id: 'actProbs', label: '你的请求属于哪种动作（小分类器的概率）', sub: '在约 8000 条模板请求上训练；模板上的准确率很高，但对真实说法只能当参考', large: true, size: [360, 230],
                    render: (c, S) => { const p = S.Q.act.p, A = P19.CAT.ACTIONS; LM.draw.bars(c, p, A.map((a) => ({ answer: '直接回答', tool: '调一个工具', code: '写一段代码', gui: '操作界面', ask: '先问清楚' })[a]), { horizontal: true, colors: A.map((a) => (a === S.Q.act.best ? LM.C.green : '#56606b')), range: [0, 1], digits: 2, labelPx: 11 }); },
                },
                {
                    id: 'actCost', label: '同一件事，不同动作的代价（按假设估算）', sub: '「对 N 个对象」：逐个调用 1 + N 次往返 vs 一段脚本 1 次；界面操作按每个对象 3 步、每步一张截图', size: [360, 230],
                    render: (c, S) => {
                        const N = 120, def = TK.countTokens(S.C.B, TK.toolJSON(S.Q.pick)), call = 40, result = 30;
                        const tool1 = def + call + result, toolN = def + (N + 1) * (call + result), code = def + 220 + 60, gui = N * 3 * (1560 + 60);
                        LM.draw.bars(c, [tool1, toolN, code, gui].map((v) => v / 1000), ['一个对象：调一次工具', `${N} 个对象：逐个调用`, `${N} 个对象：一段脚本`, `${N} 个对象：操作界面`], { horizontal: true, colors: [LM.C.green, LM.C.red, LM.C.blue, LM.C.purple], digits: 1, valueSuffix: 'K', title: '模型要读 / 写的 token（估算，千）' });
                    },
                },
            ],
            notes: '把请求改成「把关卡里所有点光源的强度减半」，分类器会判为「写一段代码」：120 盏灯逐个调用要 121 次往返、上下文里堆满结果；一段脚本一次跑完，只把汇总结果交回来——第 21 篇在一个模拟的游戏引擎里实测这件事。再试「什么是 Draw Call」（直接回答）、「这个老工具没有接口，帮我点一下导出按钮」（操作界面）、「帮我弄一下那个」（先问清楚）。',
            warn: '右图是按固定假设估算的（N = 120、每次调用约 40 token、结果约 30 token、界面每个对象 3 步）——它说明的是数量级差别，不是精确成本。真实的大模型会在一个回答里交替使用这几种动作：先问一句、再查一下、最后写段脚本批量改。',
        },
        {
            id: 'prod', title: '教学版 → 生产版', tag: '生产版',
            desc: '本页每个环节在真实系统里都有对应，只是规模和模型都大得多：',
            blocks: [
                { type: 'html', html: `<table class="lm-tbl">
                    <tr><th>环节</th><th>本页</th><th>真实系统</th></tr>
                    <tr><td>工具目录</td><td>125 个应用、1590 个工具（按真实 API 结构整理）</td><td>一个公司常见几十个 MCP 服务器、上千个工具；集成平台宣称数万个动作；工具会动态增减（MCP 的 notifications/tools/list_changed）</td></tr>
                    <tr><td>工具描述</td><td>一句英文 + 可选的中文说法</td><td>详细描述：做什么、什么时候用 / 不该用、参数含义；附例子（Anthropic 的 input_examples：在复杂参数上准确率从 72% 提到 90%）</td></tr>
                    <tr><td>检索模型</td><td>约 100 万参数的哈希特征编码器（int8），模板数据训练</td><td>在海量多语言数据上预训练的嵌入模型 + 交叉编码器重排；用线上日志持续标注、评测</td></tr>
                    <tr><td>加载方式</td><td>top-k / 两级路由 / 工具搜索 / 代码即工具的 token 账</td><td>defer_loading + 工具搜索（Anthropic）；Claude Code 在 MCP 工具过多时自动切到工具搜索；Skills 的三级渐进披露；把 MCP 服务器暴露成代码 API</td></tr>
                    <tr><td>过滤</td><td>「已连接的应用」</td><td>按用户 / 项目启用的服务器、权限范围、当前所在的应用（IDE 里只给代码相关工具）、组织的白名单</td></tr>
                    <tr><td>缓存</td><td>无</td><td>系统提示与常驻工具放在前缀命中 prompt cache；按需加载的定义追加在后面</td></tr>
                    <tr><td>决策</td><td>小分类器</td><td>大模型自己决定：回答 / 调用（可并行多个）/ 写代码 / 操作界面 / 追问；Harness 管权限、预算与确认</td></tr>
                    <tr><td>安全</td><td>无</td><td>工具描述和工具结果都是不可信输入（tool poisoning、提示注入）；服务器要审核、固定版本，描述变化要告警</td></tr>
                </table>` },
            ],
            formula: ['选对工具 = 检索不漏（Recall@k）× 模型从 k 个里选对 × 参数填对      // 三者都要评测'],
            previews: [
                {
                    id: 'summary', needs: ['tok'], label: '你的请求：一路走下来的结果', sub: '顶栏改请求，这里跟着变', size: [720, 170],
                    render: (c, S) => {
                        const C = LM.C, Q = S.Q, R = S.C.R, st = Object.assign({ all: P19.allRow(S) }, S.ST);
                        LM.draw.text(c, [[{ t: '请求：', color: C.gray }, { t: S.query, color: C.text }], [{ t: '动作：', color: C.gray }, { t: `${P19.CAT.ACT_ZH[Q.act.best]}（${(100 * Math.max(...Q.act.p)).toFixed(0)}%）`, color: C.yellow }], [{ t: '检索第一名：', color: C.gray }, { t: `${Q.pick.mcpName}（应用路由第一名：${R.apps[Q.res.appTop[0]].name} ${(100 * Q.res.appP[Q.res.appTop[0]]).toFixed(0)}%）`, color: C.green }], [{ t: '参数：', color: C.gray }, { t: `${JSON.stringify(Q.ex.args)}${Q.ex.missing.length ? `，缺 ${Q.ex.missing.join(', ')}` : '，校验通过'}`, color: Q.ex.missing.length ? C.red : C.green }], [{ t: '上下文：', color: C.gray }, { t: `检索 top-${S.k} ${st.topk.total.toLocaleString()} token · 工具搜索 ${st.search.total.toLocaleString()} token${st.all ? ` · 全部放进去 ${st.all.total.toLocaleString()} token` : ''} · 检索耗时 ${Q.res.ms.toFixed(1)} ms`, color: C.text }]], { px: 12, lineGap: 9 });
                    },
                },
            ],
            notes: '下一篇（20）换一个角度回答「成千上万种能力」：编程 Agent 只有六七个通用工具（读、写、改、搜文件、跑命令），因为 shell 本身就能调用成千上万个命令行程序；第 21 篇讲怎样把游戏引擎和商用软件接进来（MCP、脚本 API、Computer Use）。',
            warn: '本页的数字（Recall、token）只对这个目录和这些模板请求成立。迁移到你自己的工具时：先收集真实请求并标注该用哪个工具，再比较「只靠描述」「检索」「工具搜索」几种方案——这一步省不掉。',
        },
    ],

    graph: {
        title: '工具检索 · 一个请求在 Harness 里的数据流',
        intro: '节点 = 页面里真实执行的计算；左边两路检索（词法 / 向量）在中间融合，上面是应用路由，右边把选中的定义交给模型，再校验模型生成的参数。外部节点（模型、MCP 服务器）在页面里不运行。',
        common: { title: 'Common · 本系列自带的 C++ 库', open: false, blocks: LM.CPP.blocks(['gpt']) },
        types: { txt: { color: '#56b6c2', label: 'txt · 文本' }, vec: { color: '#c678dd', label: 'vec · 向量 / 分数' }, def: { color: '#e8c25a', label: 'def · 工具定义' }, call: { color: '#e0a060', label: 'call · 调用' } },
        nodes: [
            { id: 'bm25', title: 'BM25', tag: 'CPU', lifetime: 'frame', step: 'bm25', desc: '倒排索引打分', dispatch: { kind: 'CPU', name: 'scoreBM25', count: '每个请求一次' }, inputs: [{ name: 'Query', format: '请求文本', type: 'txt', external: true }], outputs: [{ name: 'LexScores', format: '[1590] float', type: 'vec' }], detail: { preview: 'bmTop' } },
            { id: 'encode', title: 'embedText', tag: 'CPU', lifetime: 'frame', step: 'dense', desc: '哈希特征 → 64 维单位向量', dispatch: { kind: 'CPU', name: 'hashFeatures · embedText', count: '每个请求一次' }, inputs: [{ name: 'Query', format: '请求文本', type: 'txt', external: true }, { name: 'ToolIndex', format: '[1590 × 64]（建索引时算好）', type: 'vec', external: true }], outputs: [{ name: 'DenseScores', format: '[1590] cos', type: 'vec' }, { name: 'QueryVec', format: '[64]', type: 'vec' }], detail: { preview: 'denseMap' } },
            { id: 'route', title: 'routeTwoStage', tag: 'CPU', lifetime: 'frame', step: 'route', desc: '125 个应用 → 前 m 个', dispatch: { kind: 'CPU', name: 'routeTwoStage', count: '每个请求一次' }, inputs: [{ name: 'QueryVec', format: '[64]', type: 'vec' }], outputs: [{ name: 'Routed', format: '前 m 个应用里的排序', type: 'vec' }], detail: { preview: 'appProbs' } },
            { id: 'fuse', title: 'rrfFuse · topK', tag: 'CPU', lifetime: 'frame', step: 'eval', desc: '两路名次加权融合', dispatch: { kind: 'CPU', name: 'rrfFuse', count: '每个请求一次' }, inputs: [{ name: 'LexScores', format: '分数', type: 'vec' }, { name: 'DenseScores', format: '分数', type: 'vec' }, { name: 'Routed', format: '（可选）', type: 'vec' }], outputs: [{ name: 'TopK', format: 'k 个工具', type: 'def' }], detail: { preview: 'evalBars' } },
            { id: 'load', title: 'handleToolSearch', tag: 'CPU', lifetime: 'frame', step: 'defer', desc: '把 k 个完整定义放进上下文', dispatch: { kind: 'CPU', name: 'handleToolSearch', count: '每次 tool_search' }, inputs: [{ name: 'TopK', format: 'k 个工具', type: 'def' }], outputs: [{ name: 'Defs', format: 'JSON 定义', type: 'def' }], detail: { preview: 'stratTokens' } },
            { id: 'llm', title: '模型（外部）', tag: 'LLM', lifetime: 'frame', step: 'action', desc: '选动作、选工具、生成参数', dispatch: { kind: 'CPU', name: '（页面里用规则 / 小分类器代替）', count: '每轮' }, inputs: [{ name: 'Defs', format: 'JSON 定义', type: 'def' }], outputs: [{ name: 'Call', format: 'tool_use { name, input }', type: 'call' }], detail: { preview: 'actProbs' } },
            { id: 'validate', title: 'validate', tag: 'CPU', category: 'output', lifetime: 'frame', step: 'args', desc: 'JSON Schema 校验 → 执行或回报错误', dispatch: { kind: 'CPU', name: 'validate', count: '每次调用' }, inputs: [{ name: 'Call', format: '参数', type: 'call' }], outputs: [{ name: 'MCPCall', format: 'tools/call 或 is_error', type: 'call' }], detail: { preview: 'argTable' } },
        ],
    },

    references: [
        { title: 'RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation', author: 'T. Gan, Q. Sun', venue: 'arXiv 2505.03275', year: 2025, note: '全部工具描述 13.62% vs 先检索 43.13%（引言）' },
        { title: 'Introducing advanced tool use on the Claude Developer Platform', author: 'Anthropic', venue: 'Engineering blog', year: 2025, note: '工具搜索（defer_loading）、程序化工具调用、input_examples（Step 01、07、10）' },
        { title: 'Code execution with MCP: Building more efficient agents', author: 'A. Jones, C. Kelly（Anthropic）', venue: 'Engineering blog', year: 2025, note: '工具当成代码文件：15 万 → 2 千 token（Step 07）' },
        { title: 'Equipping agents for the real world with Agent Skills', author: 'Anthropic', venue: 'Engineering blog', year: 2025, note: 'Skills 的渐进披露（Step 07）' },
        { title: 'Model Context Protocol Specification (2025-06-18)', author: 'MCP 项目', venue: 'modelcontextprotocol.io', year: 2025, note: 'tools/list、tools/call、annotations、outputSchema（Step 02、08）' },
        { title: 'ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs', author: 'Y. Qin et al.', venue: 'ICLR', year: 2024, note: '大规模 API 目录与 API 检索器' },
        { title: 'Gorilla: Large Language Model Connected with Massive APIs', author: 'S. G. Patil et al.', venue: 'arXiv 2305.15334', year: 2023, note: '检索增强的 API 调用' },
        { title: 'The Probabilistic Relevance Framework: BM25 and Beyond', author: 'S. Robertson, H. Zaragoza', venue: 'Foundations and Trends in IR', year: 2009, note: 'BM25（Step 03）' },
        { title: 'Dense Passage Retrieval for Open-Domain Question Answering', author: 'V. Karpukhin et al.', venue: 'EMNLP', year: 2020, note: '双塔向量检索与批内负例（Step 04）' },
        { title: 'Feature Hashing for Large Scale Multitask Learning', author: 'K. Weinberger et al.', venue: 'ICML', year: 2009, note: '哈希特征（Step 04）' },
        { title: 'Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods', author: 'G. V. Cormack, C. L. A. Clarke, S. Büttcher', venue: 'SIGIR', year: 2009, note: 'RRF（Step 05）' },
        { title: 'Visualizing Data using t-SNE', author: 'L. van der Maaten, G. Hinton', venue: 'JMLR', year: 2008, note: '工具地图（Step 04）' },
        { title: 'MCP Security Notification: Tool Poisoning Attacks', author: 'Invariant Labs', venue: 'Blog', year: 2025, note: '工具描述里藏指令（Step 02、10）' },
    ],
});
