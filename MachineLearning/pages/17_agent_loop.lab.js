// 17_agent_loop.lab.js — 系列第 17 篇：Agent——在「行动 → 观察」的循环里用工具完成多步任务
// 构建（在 pages/ 下）：node <skill>/scripts/build_doc.mjs llm_common.js agent_common.js agent_cpp.js agent_weights.js 17_agent_loop.lab.js -o 17_agent_loop.html
// 模型：与第 16 篇同一个离线训练的迷你 GPT（agent_weights.js）；房子环境、专家、Harness、评测都是 agent_common.js 的同一份代码，在浏览器里实时运行。
const P17 = (() => {
    const KINDS = ['find', 'move', 'list'], KNAME = { find: '找东西（X 在哪？）', move: '搬东西（把 X 拿到 R）', list: '看房间（R 里有什么？）' };
    const RXY = [[0.5, 0.42], [0.82, 0.42], [0.18, 0.42], [0.5, 0.8], [0.82, 0.8]];   // 默认布局里 5 个房间在地图上的位置（相对坐标）
    const RXY2 = [[0.5, 0.42], [0.82, 0.42], [0.18, 0.42], [0.18, 0.8], [0.82, 0.8]];
    function loadModel() { const w = AGT_W.main, m = LM.gpt.create(w.cfg, 1); LM.gpt.loadWeights(m, w.b64); return m; }
    /** 第 ep 局：任务 + 房子的初始状态（固定种子） */
    function makeEp(kind, ep, adj) { const rand = LM.rng(9000 + ep * 17 + KINDS.indexOf(kind) * 7919); const task = AGT.houseTask(rand, kind), H = AGT.newHouse(rand, { adj }); return { task, H }; }
    const clone = (H) => JSON.parse(JSON.stringify(H));
    /** 跑一局并记下每次工具调用之后房子的状态（地图动画用） */
    function rollout(M, E0, o) {
        const H = clone(E0.H), states = [{ room: H.room, hand: H.hand, at: Object.assign({}, H.at), call: null }];
        const r = AGT.runHouse(M, E0.task, H, Object.assign({}, o, { onStep: (h, tool, arg, res) => states.push({ room: h.room, hand: h.hand, at: Object.assign({}, h.at), call: `${tool}(${arg})`, res: res.res, ok: res.ok }) }));
        r.states = states;
        return r;
    }
    /** 失败分类（和 C++ 的 classify 相同） */
    function classify(r) {
        if (r.success) return 'ok';
        if (r.reason === 'tokens' || r.reason === 'context') return 'budget';
        if (r.reason === 'bad-token' || r.events.some((e) => e.type === 'parse-error')) return 'format';
        if (r.events.some((e) => e.type === 'call' && /去不了|这里没有|手里没有|手里有了|没有这个工具/.test(e.res))) return 'invalid';
        return 'wrong';
    }
    const FNAME = { ok: '成功', budget: '预算 / 上下文用完', format: '格式写坏', invalid: '做了不合法的动作', wrong: '说完了但没做到 / 答错' };
    /** Step 05：每种任务 40 局（贪心）——成功率、调用次数（vs 专家）、失败类型 */
    function succBench(M) {
        const res = {}, jobs = [];
        KINDS.forEach((k) => { for (let i = 0; i < 40; i++) { const E0 = makeEp(k, 100 + i); const ex = AGT.houseExpert(E0.task, clone(E0.H), LM.rng(1), {}); const nExp = ex.turns.filter((t) => t.role === 'bot' && t.text.startsWith('<call>')).length; jobs.push({ kind: 'episode', start: () => AGT.newHouseEpisode(M, E0.task, clone(E0.H), { greedy: true, maxCalls: 20 }), done: (r) => { const a = (res[k] = res[k] || { n: 0, ok: 0, calls: [], exp: [], fail: {} }); a.n++; if (r.success) a.ok++; a.calls.push(r.calls); a.exp.push(nExp); const f = classify(r); a.fail[f] = (a.fail[f] || 0) + 1; } }); } });
        return Object.assign(AGT.newBench(M, jobs), { res });
    }
    /** Step 06：30 局搬东西，每局在温度 0.8 下采 8 条轨迹：pass@k（用环境当验证器）；20 局找东西：多数投票 */
    function searchBench(M) {
        const res = { pass: [], greedy: 0, vote: { ok: 0, n: 0, single: 0 } }, jobs = [];
        for (let i = 0; i < 30; i++) {
            const E0 = makeEp('move', 300 + i), succ = [];
            jobs.push({ kind: 'episode', start: () => AGT.newHouseEpisode(M, E0.task, clone(E0.H), { greedy: true, maxCalls: 20 }), done: (r) => { if (r.success) res.greedy++; } });
            for (let k = 0; k < 8; k++) jobs.push({ kind: 'episode', start: () => AGT.newHouseEpisode(M, E0.task, clone(E0.H), { temp: 0.8, rand: LM.rng(77 + i * 131 + k), maxCalls: 20 }), done: (r) => { succ.push(r.success); if (succ.length === 8) res.pass.push(succ.slice()); } });
        }
        for (let i = 0; i < 20; i++) {
            const E0 = makeEp('find', 400 + i), ans = [];
            for (let k = 0; k < 5; k++) jobs.push({ kind: 'episode', start: () => AGT.newHouseEpisode(M, E0.task, clone(E0.H), { temp: 0.8, rand: LM.rng(91 + i * 37 + k), maxCalls: 20 }), done: (r) => { ans.push({ a: r.answer, ok: r.success }); if (ans.length === 5) { const votes = {}; ans.forEach((x) => { votes[x.a] = (votes[x.a] || 0) + 1; }); const best = Object.keys(votes).sort((a, b) => votes[b] - votes[a])[0]; const want = `${E0.task.item}在${AGT.ROOMS[E0.H.at[E0.task.item]]}。`; res.vote.n++; if (best === want) res.vote.ok++; if (ans[0].ok) res.vote.single++; } } });
        }
        return Object.assign(AGT.newBench(M, jobs), { res });
    }
    /** Step 07：工具超时概率 p ∈ {0, 0.1, 0.2, 0.3}，各 40 局搬东西（贪心）——模型自己重试 */
    const FAILS = [0, 0.1, 0.2, 0.3];
    function faultBench(M) {
        const res = {}, jobs = [];
        FAILS.forEach((p, j) => { for (let i = 0; i < 40; i++) { const E0 = makeEp('move', 500 + i); jobs.push({ kind: 'episode', start: () => AGT.newHouseEpisode(M, E0.task, clone(E0.H), { greedy: true, maxCalls: 24, failP: p, rand: LM.rng(31 + i * 7 + j) }), done: (r) => { const a = (res[j] = res[j] || { n: 0, ok: 0, timeouts: 0, retried: 0 }); a.n++; if (r.success) a.ok++; const ev = r.events.filter((e) => e.type === 'call'); ev.forEach((e, q) => { if (e.res === '超时') { a.timeouts++; const nx = ev[q + 1]; if (nx && nx.tool === e.tool && nx.arg === e.arg) a.retried++; } }); } }); } });
        return Object.assign(AGT.newBench(M, jobs), { res });
    }
    /** Step 08：默认布局 vs 没见过的布局，各 40 局（三种任务轮流） */
    function layoutBench(M) {
        const res = { train: { n: 0, ok: 0, invalid: 0 }, new: { n: 0, ok: 0, invalid: 0 } }, jobs = [];
        for (let i = 0; i < 40; i++) for (const [key, adj] of [['train', null], ['new', AGT.ADJ2]]) {
            const E0 = makeEp(KINDS[i % 3], 600 + i, adj);
            jobs.push({ kind: 'episode', start: () => AGT.newHouseEpisode(M, E0.task, clone(E0.H), { greedy: true, maxCalls: 20 }), done: (r) => { const a = res[key]; a.n++; if (r.success) a.ok++; if (r.events.some((e) => e.type === 'call' && e.res === '去不了')) a.invalid++; } });
        }
        return Object.assign(AGT.newBench(M, jobs), { res });
    }
    /** 最后一层注意力：上下文里第 pos 个 token（作为 query）看前面每个 token 的权重（4 个头平均） */
    function attnRow(M, ids, pos) {
        const x = Int32Array.from(ids.slice(0, pos + 1)), T = pos + 1;
        const out = LM.AG.noGrad(() => LM.gpt.forward(M, x, 1, T, { trace: true }));
        const L = out.trace.layers, H = M.cfg.H, rows = [];
        for (let l = 0; l < L.length; l++) { const P = L[l].att, w = new Float32Array(T); for (let h = 0; h < H; h++) for (let j = 0; j < T; j++) w[j] += P[(h * T + pos) * T + j] / H; rows.push(w); }
        return rows;
    }
    /** 自动挑一个「做决定」的位置：找东西 / 看房间 = 回答里第一个房间名或物品名的字；搬东西 = 「拿(」后面的第一个字 */
    function decisionPos(r, task) {
        const ids = r.ids, eos = ids.lastIndexOf(AGT.SP.eres);
        if (task.kind === 'move') { for (let i = 0; i < ids.length - 2; i++) if (ids[i] === AGT.SP.call && AGT.itos[ids[i + 1]] === '拿') return i + 3; }
        const names = task.kind === 'find' ? AGT.ROOMS : AGT.ITEMS;
        const from = task.kind === 'find' ? eos + task.item.length + 2 : eos + 5;   // 跳过回答开头的物品名（找东西）/「R里有」（看房间）
        for (let i = from; i < ids.length; i++) if (names.some((n) => n[0] === AGT.itos[ids[i]])) return i;
        return ids.length - 2;
    }
    const wait = (c, B, what) => LM.draw.text(c, [{ text: `        ${what}：${B ? `${B.k} / ${B.jobs.length}` : '等待开始（滚动到本步时开始计算）'}`, color: LM.C.gray }], { px: 12 });
    /** 画房子地图：房间、连线、物品、智能体、手里的东西；o.path = 走过的房间序列 */
    function drawMap(ctx, X, Y, W, H, st, o) {
        o = o || {}; const adj = o.adj || AGT.ADJ, xy = o.adj ? RXY2 : RXY, C = LM.C, P = (r) => [X + xy[r][0] * W, Y + xy[r][1] * H], bw = Math.min(110, W * 0.26), bh = Math.min(54, H * 0.26);
        ctx.strokeStyle = '#3a4450'; ctx.lineWidth = 3;
        adj.forEach((nb, a) => nb.forEach((b) => { if (b < a) return; const [x1, y1] = P(a), [x2, y2] = P(b); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }));
        if (o.path && o.path.length > 1) { ctx.strokeStyle = 'rgba(232,194,90,0.55)'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]); ctx.beginPath(); o.path.forEach((r, i) => { const [x, y] = P(r); const jx = (i % 3 - 1) * 4; if (i) ctx.lineTo(x + jx, y + jx); else ctx.moveTo(x, y); }); ctx.stroke(); ctx.setLineDash([]); }
        AGT.ROOMS.forEach((name, r) => {
            const [x, y] = P(r), here = st && st.room === r, target = o.target === r;
            ctx.fillStyle = here ? '#1d3346' : '#11181f'; ctx.fillRect(x - bw / 2, y - bh / 2, bw, bh);
            ctx.strokeStyle = target ? C.green : here ? C.yellow : '#3a4450'; ctx.lineWidth = here || target ? 2.5 : 1.2; ctx.strokeRect(x - bw / 2, y - bh / 2, bw, bh);
            ctx.font = LM.FONT(12, true); ctx.fillStyle = C.text; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(name, x, y - bh / 2 + 3);
            if (st) { const its = AGT.ITEMS.filter((it) => st.at[it] === r && st.hand !== it); ctx.font = LM.FONT(11); ctx.fillStyle = C.orange; ctx.fillText(its.join(' ') || '·', x, y + 2); }
            if (here) { ctx.fillStyle = C.yellow; ctx.beginPath(); ctx.arc(x + bw / 2 - 8, y - bh / 2 + 8, 5, 0, 7); ctx.fill(); }
        });
        if (st) { ctx.font = LM.FONT(11); ctx.fillStyle = C.gray; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`手里：${st.hand || '空'}${st.call ? `   上一步：${st.call} → ${st.res}` : ''}`, X + 4, Y + H - 14); }
    }
    return { KINDS, KNAME, FNAME, FAILS, loadModel, makeEp, clone, rollout, classify, succBench, searchBench, faultBench, layoutBench, attnRow, decisionPos, wait, drawMap };
})();

PipelineLab.mount({
    paramSteps: { task: ['env', 'expert', 'rollout', 'attn'], ep: ['env', 'expert', 'rollout', 'attn'], temp: ['env', 'rollout', 'attn'], failP: ['env', 'rollout', 'attn'], layout: ['env', 'expert', 'rollout', 'attn'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '17 · Agent：在循环里用工具完成任务',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 17,
    css: LM.CSS,
    tagColors: { '环境': 'green', '训练': 'magenta', 'Agent': 'teal', '评测': 'blue', '鲁棒性': 'rust', '生产版': 'gray' },

    params: [
        { id: 'task', label: '任务类型', options: [{ value: 'find', label: '找东西（X 在哪？）' }, { value: 'move', label: '搬东西（把 X 拿到 R）' }, { value: 'list', label: '看房间（R 里有什么？）' }], value: 'move', affects: 'param', desc: '三种任务：找东西要一路探索直到看见它；搬东西要先找到、拿起、再走到目标房间放下；看房间只要走过去。' },
        { id: 'ep', label: '第几局', min: 0, max: 29, step: 1, value: 4, affects: 'param', desc: '换一局：物品的位置、出发的房间、任务里的物品 / 房间都随之改变（固定种子，可复现）。' },
        { id: 'temp', label: '采样温度', min: 0, max: 1.6, step: 0.2, value: 0, affects: 'param', desc: '0 = 贪心（每步取最可能的动作）；调高后同一局每次走法不同——第 06 步用它采多条轨迹。' },
        { id: 'failP', label: '工具超时概率', options: [0, 0.1, 0.2, 0.3], value: 0, affects: 'param', desc: '每次工具调用以这个概率返回「超时」、什么都不做。训练数据里专家遇到超时会原样重试，看模型学会了没有（第 07 步）。' },
        { id: 'layout', label: '房子布局', options: [{ value: 'train', label: '训练时的布局' }, { value: 'new', label: '没见过的布局' }], value: 'train', affects: 'param', desc: '换一种房间连法：模型在训练里只见过一种布局，它记住的是「从客厅去书房」这样的路，而不是「看出口再决定」（第 08 步）。' },
    ],

    stages: [
        { id: 'model', lifetime: 'resident', run(S) { S.M = P17.loadModel(); } },
        {
            id: 'ep', lifetime: 'param', run(S) {   // 这一局：专家怎么走、模型怎么走（一次跑完，几十毫秒）
                const adj = S.layout === 'new' ? AGT.ADJ2 : null;
                S.E0 = P17.makeEp(S.task, S.ep, adj);
                const ex = AGT.houseExpert(S.E0.task, P17.clone(S.E0.H), LM.rng(1), {});
                S.expert = AGT.serialize(AGT.sysText(['看', '去', '拿', '放']), S.E0.task.q, ex.turns);
                S.run = P17.rollout(S.M, S.E0, { greedy: !S.temp, temp: S.temp, rand: LM.rng(3 + S.ep), maxCalls: 24, failP: S.failP, retry: 0 });
                S.runRoles = AGT.roles(S.run.ids);
                S.cls = P17.classify(S.run);
                const pos = P17.decisionPos(S.run, S.E0.task); S.apos = pos; S.attn = P17.attnRow(S.M, S.run.ids, pos);
            },
        },
        {
            id: 'anim', lifetime: 'frame', run(S) {   // 动画：每 0.7 秒执行一次工具调用；放完停 2 秒再重来
                const n = S.run.states.length, dur = n * 0.7 + 2, t = S.t % dur;
                S.si = Math.min(n - 1, Math.floor(t / 0.7));
                // 与地图同步：轨迹上放到第 si 次工具结果为止
                let k = 0, cnt = 0; const ids = S.run.ids; for (; k < ids.length; k++) { if (ids[k] === AGT.SP.eres) { cnt++; if (cnt > S.si) break; } }
                S.reveal = S.si >= n - 1 ? ids.length : Math.min(ids.length, k + 1);
            },
        },
        { id: 'bSucc', lifetime: 'history', lazy: true, run(S) { if (!S.bs) S.bs = P17.succBench(S.M); AGT.benchWork(S.bs, 90); } },
        { id: 'bSearch', lifetime: 'history', lazy: true, run(S) { if (!S.bse) S.bse = P17.searchBench(S.M); AGT.benchWork(S.bse, 90); } },
        { id: 'bFault', lifetime: 'history', lazy: true, run(S) { if (!S.bf) S.bf = P17.faultBench(S.M); AGT.benchWork(S.bf, 90); } },
        { id: 'bLayout', lifetime: 'history', lazy: true, run(S) { if (!S.bl) S.bl = P17.layoutBench(S.M); AGT.benchWork(S.bl, 90); } },
    ],

    intro: {
        title: 'Agent = 工具调用 + 循环：每一步看结果再决定下一步',
        flow: ['房子环境 + 4 个工具', '专家轨迹 → 模仿学习', '模型自己走（实时）', '注意力：它在看哪里', '成功率与失败类型', '多试几次 + 验证器', '工具出错：重试', '换个布局就不会走了'],
        html: LM.navHtml('17') + `
            <p>第 16 篇的模型一次只调用一个工具。把那个循环一直转下去，让模型<strong>根据上一个工具的结果决定下一步做什么</strong>，就得到了 Agent：它看到的「环境」只是一条条工具结果，它做的「动作」只是一条条工具调用，任务做完时自己写出 <code>&lt;eos&gt;</code>。没有新的魔法——还是同一个模型、同一个 Harness 主循环。</p>
            <p>本页的环境是一个有 5 个房间、5 件物品的小房子，工具有 4 个：<strong>看</strong>（当前房间里有什么）、<strong>去</strong>（走到相邻的房间，同时看到它）、<strong>拿</strong>、<strong>放</strong>。任务有三种：找东西、搬东西、看房间。每局物品随机摆放，模型事先不知道东西在哪，只能一边走一边看。模型和第 16 篇是同一个（17 万参数，离线训练 8000 步），它是靠<strong>模仿</strong>一个写死的「专家」学会这些的；页面里每一局都在你的浏览器里真实运行：模型生成调用，Harness 在模拟的房子里执行，再把观察喂回给模型。</p>`,
    },

    steps: [
        {
            id: 'env', title: '环境、工具与任务', tag: '环境',
            desc: 'Agent 的「世界」对模型来说只有文字：调用 <code>看()</code> 返回 <code>客厅:书,球</code>；调用 <code>去(书房)</code>，如果书房和当前房间相邻，就走过去并返回书房里的东西，否则返回 <code>去不了</code>；<code>拿(苹果)</code> 只有苹果就在当前房间、手又空着时才成功；<code>放(苹果)</code> 把手里的苹果放在当前房间。成功与否由<strong>环境的状态</strong>判定：搬东西要看苹果最后是不是真的在卧室，而不是看模型嘴上说了什么；找东西和看房间要核对回答。右边的地图就是当前这一局的真实状态（黄点 = 智能体，橙色 = 物品，绿框 = 目标房间）。',
            formula: [
                'o_t = Tool(a_t, s_t)      s_{t+1} = Step(s_t, a_t)      a_{t+1} ~ π(· | 任务, a_1, o_1, …, a_t, o_t)      // π = 语言模型',
                'success = 1[环境的最终状态满足任务]      // 搬东西：物品在目标房间且不在手里',
            ],
            code: { title: 'C++ · House / observe / houseStep / houseTools', src: AGT_CPP.house },
            previews: [
                {
                    id: 'mapNow', live: true, label: '这一局的房子（随动画一步步更新）', sub: '黄点 = 智能体所在的房间 · 橙色 = 物品 · 绿框 = 任务的目标房间 · 虚线 = 模型走过的路', large: true, size: [360, 300],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c), st = S.run.states[S.si], path = S.run.states.slice(0, S.si + 1).map((x) => x.room).filter((r, i, a) => i === 0 || r !== a[i - 1]);
                        ctx.font = LM.FONT(12, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`任务：${S.E0.task.q}`, 50, 6);
                        P17.drawMap(ctx, 6, 26, w - 12, h - 30, st, { adj: S.layout === 'new' ? AGT.ADJ2 : null, path, target: S.E0.task.kind === 'find' ? null : S.E0.task.room });
                    },
                },
                {
                    id: 'toolTbl', label: '4 个工具与它们的返回值', sub: '括号里是参数 · 失败的调用不改变环境，只返回一句话', size: [360, 300],
                    render: (c, S) => { const C = LM.C; LM.draw.text(c, [[{ t: '看()', color: C.yellow, bold: true }, { t: '  → 「客厅:书,球」（当前房间里的东西）', color: C.text }], [{ t: '去(房间)', color: C.yellow, bold: true }, { t: '  → 走过去并返回那个房间的东西；不相邻 → 「去不了」', color: C.text }], [{ t: '拿(物品)', color: C.yellow, bold: true }, { t: '  → 「拿到了」；不在这里 → 「这里没有」；手里有东西 → 「手里有了」', color: C.text }], [{ t: '放(物品)', color: C.yellow, bold: true }, { t: '  → 「放好了」；手里没有 → 「手里没有」', color: C.text }], { text: '' }, [{ t: '三种任务（每局随机）：', color: C.dim }], [{ t: '  钥匙在哪？ · 把苹果拿到卧室 · 书房里有什么？', color: C.text }], { text: '' }, [{ t: `布局：${S.layout === 'new' ? '没见过的布局（客厅—厨房—院子，客厅—卧室—书房）' : '训练时的布局（客厅连着厨房、卧室、书房；书房连着院子）'}`, color: C.gray }]], { px: 12, lineGap: 8 }); },
                },
            ],
            notes: '拖 <b>第几局</b> 换物品摆放和出发点，换 <b>任务类型</b> 看三种任务。注意模型看不到地图——它只看得到工具返回的文字。',
            warn: '这是一个完全可观测、确定性、只有 5 个房间的玩具环境。真实 Agent 面对的环境（网页、代码仓库、操作系统）观察很长、动作空间巨大、很多动作不可逆（删文件、下单）——所以第 18 篇的权限与沙箱是必需品而不是锦上添花。',
        },
        {
            id: 'expert', title: '专家轨迹：模仿学习（behavior cloning）', tag: '训练',
            desc: '模型是怎么学会在房子里走的？我们写了一个<strong>专家</strong>：先「看」，然后按固定顺序<strong>深度优先</strong>地把房间走一遍（每走进一个房间就看到里面的东西），一旦看见目标就停下；搬东西时再沿最短路径走过去拿、再走到目标房间放下。专家在随机生成的房子里跑出成千上万条轨迹，每条轨迹序列化成一段对话（第 16 篇的格式），在模型写的 token 上算交叉熵——这就是<strong>模仿学习</strong>（behavior cloning）。训练数据里还混了两种「Harness 事件」：12% 的局里某次调用会「超时」（专家原样重试），30% 的局里中途被 Harness 压缩成一条记忆（第 18 篇）。左图是当前这一局专家的完整轨迹（模型训练时学的就是这种文字），右上是训练中的成功率。',
            formula: [
                'L = − Σ_t m_t log π(a_t | 任务, a_{<t}, o_{<t})      // 只在专家的动作（模型写的 token）上算损失，观察 o 由环境给',
                '专家：看 → DFS 探索直到看见目标 → 最短路径 → 拿 → 最短路径 → 放',
            ],
            code: { title: 'C++ · shortestPath / Expert / expertAct / exploreFor / expertMove', src: AGT_CPP.expert },
            previews: [
                {
                    id: 'expChips', label: '这一局专家的轨迹（训练数据就长这样）', sub: '实线 = 模型要学会写出的 token（动作）· 虚线 = 不计损失（任务、环境的观察）', large: true, size: [720, 170],
                    render: (c, S) => { const { ctx } = PL.draw.ctx(c), ex = S.expert, mask = ex.role.map((r) => r === 'bot'); const r = AGT.drawChips(ctx, ex.ids, { y0: 8, w: 704, px: 12, mask }); ctx.font = LM.FONT(12); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`专家：${ex.segs.filter((s) => s.role === 'bot' && s.text.startsWith('<call>')).length} 次工具调用 · ${ex.ids.length} 个 token`, 8, r.y + 8); },
                },
                {
                    id: 'houseEval', label: '训练中的成功率（贪心，每 1000 步评测一次，每种任务 60 局，最后一次 200 局）', sub: '蓝 = 找东西 · 橙 = 搬东西 · 绿 = 看房间', size: [360, 200],
                    render: (c, S) => { const E = AGT_W.main.evals, xs = E.map((e) => e[0]), g = (k) => E.map((e) => e[1][k].succ); PL.draw.plot(c, { x: [0, AGT_W.main.steps], y: [0, 1.25], xLabel: '训练步', yLabel: '成功率', series: [{ data: g('find'), x: xs, color: LM.C.blue, width: 2, label: '找' }, { data: g('move'), x: xs, color: LM.C.orange, width: 2, label: '搬' }, { data: g('list'), x: xs, color: LM.C.green, width: 2, label: '看' }], hmarks: [{ y: 1, label: '', color: '#3a3f46' }] }); },
                },
                {
                    id: 'houseLoss', label: '多步任务的训练损失（每 50 步平均）', sub: '对数纵轴 · 只在专家动作的 token 上平均', size: [360, 200],
                    render: (c, S) => { const cv = AGT_W.main.curve; PL.draw.plot(c, { x: [0, AGT_W.main.steps], y: [0.003, 5], logY: true, xLabel: '训练步', series: [{ data: cv.map((p) => p[2]), x: cv.map((p) => p[0]), color: LM.C.orange, width: 1.5, label: '多步任务' }] }); },
                },
            ],
            notes: '离线评测（训练结束时每种任务 200 局，贪心解码）见右上图的最后一个点。专家本身在每一局都成功（它知道地图）；模型学到的是专家在每种「看到的情况」下会做什么。',
            warn: '模仿学习的老问题：模型只见过专家走过的状态。一旦它犯了一个专家从来不会犯的错（走进死胡同、拿错东西），后面的局面就是训练里没有的，错误会越滚越大（compounding error，DAgger 一类方法就是为此提出的）。真实的 Agent 训练因此在模仿之后还要用强化学习，在模型自己走出的轨迹上按结果奖励（第 05、09 篇）。',
        },
        {
            id: 'rollout', title: '让模型自己走：一个回合', tag: 'Agent',
            desc: '现在把模型放进房子里：Harness 拼好提示 <code>&lt;sys&gt;可用:看去拿放&lt;user&gt;把苹果拿到卧室&lt;bot&gt;</code>，模型写出第一个调用；Harness 在模拟的房子里执行，把观察写回去；模型读到观察，决定下一个调用……直到它写出最后一句话和 <code>&lt;eos&gt;</code>。每一步都是第 16 篇那个主循环，只是这次循环要转七八圈。下面的动画按真实的运行结果重放：左边地图上黄点移动，右边是同步的对话（颜色同第 16 篇）。',
            formula: [
                'repeat:  a_t = Generate(ctx) until </call>;  o_t = Tool(a_t);  ctx += a_t + <res> o_t </res>      until <eos>',
            ],
            code: { title: 'C++ · HouseRun / runMoveTask（Harness 主循环见第 16 篇的 runEpisode）', src: AGT_CPP.agent },
            previews: [
                {
                    id: 'agentTrace', live: true, label: '模型的这一局（与左上的地图同步）', sub: '每 0.7 秒一次工具调用 · 颜色：灰 系统提示 · 蓝 用户 · 橙 Harness 写入（观察）· 绿 / 黄 模型生成', large: true, size: [720, 230],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), r = S.run;
                        const p = AGT.drawChips(ctx, r.ids, { roles: S.runRoles, y0: 8, w: 704, px: 12, upto: S.reveal });
                        const done = S.reveal >= r.ids.length;
                        ctx.font = LM.FONT(12); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillStyle = !done ? LM.C.gray : r.success ? LM.C.green : LM.C.red;
                        ctx.fillText(done ? `${r.success ? '✓ 任务完成' : `✗ ${P17.FNAME[S.cls]}`} · 工具调用 ${r.calls} 次 · ${r.tokens} 个 token · 回答「${r.answer}」` : `第 ${S.si} 次工具调用之后……`, 8, p.y + 8);
                    },
                },
            ],
            notes: '换几局看看：大多数局里模型的走法和专家一模一样（深度优先、看见就停）。把 <b>采样温度</b> 调到 1.2 以上，偶尔能看到它走错房间、再自己走回来，或者在找到之前就「猜」一个答案——这些都是第 05 步统计里的失败类型。',
            warn: '这里每一步都由模型重新决定，没有任何写死的规划。也正因为如此，Agent 的行为只能统计地评估（第 05 步）：同一个模型在两局看起来很像的任务上，可能一局走得很漂亮、另一局原地打转。',
        },
        {
            id: 'attn', title: '模型在看哪里：注意力把观察和决定连起来', tag: 'Agent',
            desc: '模型怎么知道苹果在哪个房间？靠注意力回头去「看」之前的观察。下图取模型做一个关键决定的那个位置：找东西时是回答里房间名的第一个字，搬东西时是「拿(」后面要写物品名的地方，看房间时是回答里第一个物品名的字。对这个位置，把最后两层注意力（4 个头平均）对前面每个 token 的权重画成颜色：越亮表示这一步越依赖那个 token。通常最亮的是任务里的物品名、以及<strong>看到目标的那条观察</strong>——这就是第 07 篇讲的注意力，在 Agent 里扮演「工作记忆」。',
            formula: [
                'α_j = softmax_j(q_pos · k_j / √d_h)      // 对上下文里第 j 个 token 的注意力（最后一层，4 个头平均）',
            ],
            previews: [
                {
                    id: 'attnStrip', label: '决定那一刻，模型最后两层的注意力分布在哪些 token 上', sub: '每块 = 上下文里的一个 token · 底色越亮 = 注意力越大 · 黄框 = 正在做决定的位置（query）', large: true, size: [720, 260],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), ids = S.run.ids.slice(0, S.apos + 1), C = LM.C, c3 = [0, 0, 0];
                        let y = 6;
                        S.attn.slice(-2).forEach((w, li) => {
                            ctx.font = LM.FONT(11, true); ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`第 ${S.attn.length - 1 + li} 层（共 ${S.attn.length} 层）`, 8, y); y += 16;
                            let mx = 0; for (let j = 0; j < w.length - 1; j++) mx = Math.max(mx, w[j]);
                            let x = 8; const px = 12, hh = px + 8;
                            ctx.font = LM.FONT(px); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                            ids.forEach((id, j) => {
                                const t = AGT.itos[id], cw = Math.max(ctx.measureText(t).width + 6, px + 2);
                                if (x + cw > 712) { x = 8; y += hh + 3; }
                                PL.cmap.inferno(Math.min(1, w[j] / (mx || 1)), c3); ctx.fillStyle = LM.rgbStr(c3); ctx.fillRect(x, y, cw, hh);
                                ctx.fillStyle = w[j] / (mx || 1) > 0.6 ? '#0a0a0a' : '#e6edf3'; ctx.fillText(t, x + cw / 2, y + hh / 2 + 0.5);
                                if (j === ids.length - 1) { ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y - 1, cw + 2, hh + 2); }
                                x += cw + 2;
                            });
                            y += hh + 12;
                        });
                        ctx.font = LM.FONT(11); ctx.fillStyle = C.gray; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`query = 第 ${S.apos} 个 token「${AGT.itos[S.run.ids[S.apos]]}」（模型实际写下的字）`, 8, y);
                    },
                },
            ],
            notes: '换 <b>任务类型</b>：找东西时，回答里房间名的那个字通常最强烈地看向「看到钥匙的那条观察」里的房间名；搬东西时，「拿(」之后的物品名看向任务里的物品名。换几局看看规律是否稳定。',
            warn: '注意力权重只是一种线索，不是严格的「解释」：信息也可能通过残差流在前面的层里搬运过来。这里只画了最后两层，前面的层往往在做更局部的事（例如把「书」和「房」拼成一个词）。',
        },
        {
            id: 'eval', title: '成功率与失败类型：Agent 只能统计地评估', tag: '评测',
            desc: '一局的成败说明不了什么。下面在你的浏览器里对每种任务跑 40 局（固定种子、贪心解码），统计成功率（带 95% 置信区间）、平均工具调用次数（和专家比），以及失败的类型：<strong>预算用完</strong>（转圈或上下文满了）、<strong>格式写坏</strong>、<strong>不合法的动作</strong>（去不相邻的房间、拿不在这儿的东西）、<strong>说完了但没做到</strong>（搬东西任务里东西没到位、找东西答错）。这四类失败在真实 Agent 里也最常见，各自的对策不同：预算 → 更好的停止条件和规划；格式 → 约束解码；不合法动作 → 更好的观察 / 在 Harness 里校验；没做到 → 更强的模型或者验证器（下一步）。',
            formula: [
                'p̂ = k / n,   95% CI = Wilson(k, n)      // n = 40 局；区间宽说明要更多局才能比较两个版本',
            ],
            code: { title: 'C++ · Failure / classify / evaluateMove', src: AGT_CPP.evalAgent },
            previews: [
                {
                    id: 'succBars', needs: ['bSucc'], live: true, label: '每种任务 40 局的成功率（95% 置信区间）', sub: '页面里实时跑完 · 右边 = 平均工具调用次数（模型 / 专家）', large: true, size: [360, 220],
                    render: (c, S) => {
                        const B = S.bs; if (!B || !Object.keys(B.res).length) return P17.wait(c, B, '正在评测');
                        const { ctx, w, h } = PL.draw.ctx(c), C = LM.C, x0 = 120, x1 = w - 90;
                        ctx.font = LM.FONT(11, true); ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`${B.k} / ${B.jobs.length} 局`, 50, 6);
                        P17.KINDS.forEach((k, i) => {
                            const a = B.res[k], y = 40 + i * 56; if (!a) return;
                            const p = a.ok / a.n, ci = AGT.wilson(a.ok, a.n), X = (v) => x0 + v * (x1 - x0);
                            ctx.fillStyle = C.text; ctx.font = LM.FONT(12); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(P17.KNAME[k].slice(0, 3), x0 - 8, y);
                            ctx.fillStyle = '#1b2733'; ctx.fillRect(x0, y - 9, x1 - x0, 18); ctx.fillStyle = [C.blue, C.orange, C.green][i]; ctx.fillRect(x0, y - 9, X(p) - x0, 18);
                            ctx.strokeStyle = C.text; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(ci[0]), y); ctx.lineTo(X(ci[1]), y); ctx.moveTo(X(ci[0]), y - 5); ctx.lineTo(X(ci[0]), y + 5); ctx.moveTo(X(ci[1]), y - 5); ctx.lineTo(X(ci[1]), y + 5); ctx.stroke();
                            ctx.textAlign = 'left'; ctx.fillStyle = C.text; ctx.fillText(`${(100 * p).toFixed(0)}%（${a.ok}/${a.n}）`, x1 + 6, y - 7);
                            const mc = a.calls.reduce((s, v) => s + v, 0) / a.n, me = a.exp.reduce((s, v) => s + v, 0) / a.n; ctx.fillStyle = C.gray; ctx.fillText(`调用 ${mc.toFixed(1)} / ${me.toFixed(1)}`, x1 + 6, y + 8);
                        });
                    },
                },
                {
                    id: 'failBars', needs: ['bSucc'], live: true, label: '失败的局都是怎么失败的', sub: '三种任务合计 · 每类失败的局数', size: [360, 220],
                    render: (c, S) => {
                        const B = S.bs; if (!B || !Object.keys(B.res).length) return P17.wait(c, B, '正在评测');
                        const keys = ['budget', 'format', 'invalid', 'wrong'], vals = keys.map((f) => P17.KINDS.reduce((s, k) => s + ((B.res[k] && B.res[k].fail[f]) || 0), 0));
                        LM.draw.bars(c, vals, keys.map((f) => P17.FNAME[f]), { horizontal: true, colors: [LM.C.purple, LM.C.red, LM.C.orange, LM.C.yellow], digits: 0, valueSuffix: ' 局', title: '' });
                    },
                },
            ],
            notes: '置信区间是这里的重点：搬东西 40 局成功 39 局，95% 区间仍有 87%–99.6% 这么宽——两个版本的 Agent 如果只差几个百分点，40 局根本分不出来（第 18 篇用配对比较来解决）。唯一失败的那一局走对了路线：拿着球回到了厨房，却在放下之前用完了模型 192 个 token 的上下文（专家走这一局要 13 次调用）。',
            warn: '这里的「专家调用次数」是专家在同一局里的调用数，模型如果完全模仿专家就应该相等；模型的平均值更大，说明它偶尔多走了冤枉路。真实 Agent 评测（SWE-bench、WebArena、τ-bench 等）同样报告成功率，并越来越重视成本（token、调用次数、时间）。',
        },
        {
            id: 'search', title: '多试几次：采样 + 验证器（pass@k）与多数投票', tag: '评测',
            desc: '贪心解码每局只有一种走法，失败就是失败。如果环境能告诉我们「任务完成了没有」（搬东西任务可以直接检查苹果在哪），就可以在温度 0.8 下采样 N 条轨迹，只要其中一条成功就算成功——<strong>pass@k</strong>，验证器就是环境本身。代价是 N 倍的计算。找东西任务没有可检查的环境状态（答案对不对要看真实位置，现实中往往不知道），就只能<strong>多数投票</strong>（self-consistency）：采 5 次，取出现最多的答案。下面两张图都在浏览器里实时计算：30 局搬东西 × 8 条采样轨迹，20 局找东西 × 5 次采样。',
            formula: [
                'pass@k = 1 − E[ C(n − c, k) / C(n, k) ]      // n 条样本里有 c 条成功；这里 n = 8',
                'majority = argmax_a #{i : answer_i = a}',
            ],
            code: { title: 'C++ · bestOfN / majorityAnswer', src: AGT_CPP.search },
            previews: [
                {
                    id: 'passK', needs: ['bSearch'], live: true, label: '搬东西：pass@k（30 局 × 8 条轨迹，温度 0.8）', sub: '蓝 = pass@k · 灰线 = 贪心解码的成功率 · 用的是无偏估计（Chen 等 2021）', large: true, size: [360, 220],
                    render: (c, S) => {
                        const B = S.bse; if (!B || !B.res.pass.length) return P17.wait(c, B, '正在采样');
                        const comb = (n, k) => { let r = 1; for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; };
                        const ks = [1, 2, 3, 4, 5, 6, 7, 8], pk = ks.map((k) => B.res.pass.reduce((s, a) => { const n = a.length, cc = a.filter(Boolean).length; return s + (n - cc < k ? 1 : 1 - comb(n - cc, k) / comb(n, k)); }, 0) / B.res.pass.length);
                        const nG = Math.min(B.res.pass.length, 30);
                        PL.draw.plot(c, { x: [1, 8], y: [0, 1.05], xLabel: 'k（采样条数）', yLabel: '成功率', series: [{ data: pk, x: ks, color: LM.C.blue, width: 2.5, label: `pass@k（${B.res.pass.length} 局）` }], hmarks: [{ y: B.res.greedy / Math.max(1, nG), label: `贪心 ${(100 * B.res.greedy / Math.max(1, nG)).toFixed(0)}%`, color: LM.C.gray }] });
                    },
                },
                {
                    id: 'voteBars', needs: ['bSearch'], live: true, label: '找东西：一次采样 vs 5 次多数投票', sub: '温度 0.8 · 20 局 · 答对率', size: [360, 160],
                    render: (c, S) => { const B = S.bse; if (!B || !B.res.vote.n) return P17.wait(c, B, '正在采样'); const v = B.res.vote; LM.draw.bars(c, [v.single / v.n, v.ok / v.n], ['一次采样', '5 次多数投票'], { horizontal: true, colors: [LM.C.gray, LM.C.green], range: [0, 1], digits: 2, title: `${v.n} 局` }); },
                },
            ],
            notes: '结果可能和你预想的不同：温度 0.8 时单条轨迹的成功率（92%）略低于贪心（28/30），而 8 条里至少一条成功的也还是 28/30——多采样<b>没有带来任何提升</b>。原因有两个：一是这个模型模仿的是一个确定性的专家，每一步的分布都很尖，8 条轨迹往往一模一样（温度升到 1.6，8 条里至少一条成功的仍是 28/30）；二是失败的 2 局是结构性的——它们需要 12、13 次调用，加上回答超过了 192 个 token 的上下文，每次都在写最后一句话时被截断。找东西本来就 20/20，多数投票也没有可提升的空间。pass@k 有用的前提是：失败是随机的，而且采样真的能产生不同的尝试——真实的大模型在难题上多样性高得多（Codex 在 HumanEval 上 pass@1 约 29%、pass@100 约 72%）。',
            warn: 'pass@k 的前提是验证器既准确又便宜。代码任务可以跑测试；但很多真实任务的「成功」没法自动判断，或者动作有副作用不能重来（发出去的邮件收不回）——那时只能在模型内部做规划（想好再动手），或者先在沙箱 / 模拟环境里试。',
        },
        {
            id: 'faults', title: '工具会出错：超时与模型自己的重试', tag: '鲁棒性',
            desc: '真实的工具会失败：网络超时、服务限流、偶发的异常。训练数据里 25% 的局开启了故障注入（每次调用 12% 的概率「超时」），专家遇到超时会把同一个调用原样再发一次——于是模型学会了「看到 <code>&lt;res&gt;超时&lt;/res&gt;</code> 就重试」。下面把每次调用的超时概率设成 0、0.1、0.2、0.3，各跑 40 局搬东西，统计成功率，以及模型遇到超时后马上重试同一个调用的比例。把顶栏的 <b>工具超时概率</b> 调高，上面第 03 步的动画里也能看到超时和重试。',
            formula: [
                'P(一次调用成功) = 1 − p      // 专家 / 模型的策略：超时 → 原样重试，直到成功或预算用完',
            ],
            code: { title: 'C++ · flakyTools', src: AGT_CPP.flaky },
            previews: [
                {
                    id: 'faultPlot', needs: ['bFault'], live: true, label: '成功率 vs 工具超时概率（每点 40 局搬东西）', sub: '蓝 = 成功率 · 橙 = 遇到超时后马上原样重试的比例', large: true, size: [360, 210],
                    render: (c, S) => {
                        const B = S.bf; if (!B || !B.res[0]) return P17.wait(c, B, '正在评测');
                        const xs = [], ys = [], rs = []; P17.FAILS.forEach((p, j) => { const a = B.res[j]; if (!a) return; xs.push(p); ys.push(a.ok / a.n); rs.push(a.timeouts ? a.retried / a.timeouts : NaN); });
                        PL.draw.plot(c, { x: [0, 0.3], y: [0, 1.05], xLabel: '每次调用超时的概率', yLabel: '比例', series: [{ data: ys, x: xs, color: LM.C.blue, width: 2.5, label: '成功率' }, { data: rs, x: xs, color: LM.C.orange, width: 2, dash: [4, 3], label: '超时后重试' }] });
                    },
                },
            ],
            notes: '超时概率 0.1 时（训练里见过 0.12），33 次超时里模型原样重试了 31 次，40 局完成 36 局；失败的局几乎都是「上下文放不下了」——每次超时多一个往返，长一点的局就撞上 192 的上限。0.3 时完成 33 局，7 局失败里 6 局是上下文满了。下一篇会把重试挪进 Harness：模型根本看不到超时，上下文也不会被浪费。',
            warn: '只有「幂等」的工具才能放心重试：查询、读取可以；「拿」「放」「转账」这种有副作用的动作如果其实已经执行成功、只是响应丢了，重试就会执行两次。真实系统用幂等键（idempotency key）或者先查询状态再决定是否重试。',
        },
        {
            id: 'limits', title: '规划的局限：换一个布局就不会走了', tag: '鲁棒性',
            desc: '模型真的学会了「探索」吗？换一个它从没见过的房子布局试试（顶栏 <b>房子布局</b> = 没见过的布局：客厅—厨房—院子，客厅—卧室—书房）。专家在新布局里照样每局成功（它会读地图），模型却频繁碰到 <code>去不了</code>：观察里并不告诉它哪些房间相邻，训练时只有一种布局，它记住的是一串固定的路线，而不是「试一下，不行就换一个」的策略。下面各跑 40 局对比。这暴露了模仿学习的本质：模型学到的是训练分布里的捷径。',
            formula: [
                'π(a | 历史) 拟合的是训练分布 p_train(历史)；布局变了 → 历史落在分布之外 → 动作不再合理',
            ],
            previews: [
                {
                    id: 'layoutBars', needs: ['bLayout'], live: true, label: '训练时的布局 vs 没见过的布局（各 40 局，三种任务轮流）', sub: '蓝 = 成功率 · 红 = 至少碰到一次「去不了」的局的比例', large: true, size: [360, 200],
                    render: (c, S) => { const B = S.bl; if (!B || !B.res.new.n) return P17.wait(c, B, '正在评测'); const a = B.res.train, b = B.res.new; LM.draw.bars(c, [a.ok / a.n, a.invalid / a.n, b.ok / b.n, b.invalid / b.n], ['训练布局 · 成功', '训练布局 · 去不了', '新布局 · 成功', '新布局 · 去不了'], { horizontal: true, colors: [LM.C.blue, LM.C.red, LM.C.blue, LM.C.red], range: [0, 1], digits: 2, title: `${B.k} / ${B.jobs.length} 局` }); },
                },
            ],
            notes: '把 <b>房子布局</b> 切到「没见过的布局」再看第 01、03 步：地图变了，模型的动作序列却和训练时几乎一样。修法有几种：观察里写出出口（让任务真正需要读观察）、训练时随机化布局（领域随机化）、或者让模型在动手前先探索 / 规划。',
            warn: '大模型 Agent 的泛化比这个小模型好得多，但同样的现象依然存在：在训练里常见的软件 / 网站上表现很好，换一个界面布局、一个没见过的命令行工具，成功率就明显下降。评测时务必包含训练分布之外的任务。',
        },
        {
            id: 'prod', title: '教学版 → 生产版', tag: '生产版',
            desc: '本页的 Agent 与真实系统在结构上是同一个东西：模型 + 工具 + 循环。差别在于：',
            blocks: [
                { type: 'html', html: `<table class="lm-tbl">
                    <tr><th>环节</th><th>本页</th><th>真实系统</th></tr>
                    <tr><td>动作与观察</td><td>4 个工具、一行文字的观察</td><td>shell、文件读写、浏览器、搜索、代码执行、图形界面操作（截图 + 点击坐标）；观察可以有几千 token</td></tr>
                    <tr><td>思考</td><td>没有显式思考，直接写动作</td><td>ReAct（Yao 等 2023）：动作之间写出推理；推理模型在动作前先长时间「想」；Reflexion 让模型在失败后写反思再重试</td></tr>
                    <tr><td>训练</td><td>模仿一个写死的专家</td><td>人工 / 合成轨迹上的 SFT，再在可验证的任务上做强化学习（结果奖励），让模型在自己犯的错误上学会恢复</td></tr>
                    <tr><td>搜索</td><td>pass@k、多数投票</td><td>多条轨迹 + 测试 / 打分模型挑选；树搜索（Tree of Thoughts、LATS）；但有副作用的环境里很难重来</td></tr>
                    <tr><td>评测</td><td>每种任务 40 局</td><td>SWE-bench（修 GitHub issue）、WebArena（网页任务）、OSWorld（桌面任务）、τ-bench（工具 + 用户对话）等，报告成功率与成本</td></tr>
                </table>` },
            ],
            formula: ['Agent = 模型 + 工具 + 循环 + （第 18 篇的）Harness'],
            previews: [
                {
                    id: 'finalEval', label: '离线评测（训练结束，每种任务 200 局，贪心）', sub: 'tools/agent_train.js 的最后一次评测：成功率与平均工具调用次数', size: [360, 150],
                    render: (c, S) => { const E = AGT_W.main.evals[AGT_W.main.evals.length - 1][1]; LM.draw.bars(c, ['find', 'move', 'list'].map((k) => E[k].succ), ['找东西', '搬东西', '看房间'].map((n, i) => `${n}（调用 ${E[['find', 'move', 'list'][i]].calls} 次）`), { horizontal: true, colors: [LM.C.blue, LM.C.orange, LM.C.green], range: [0, 1], digits: 3 }); },
                },
            ],
            notes: '下一篇（18）讲包在这个循环外面的程序：上下文快满了怎么办、怎样复用 KV cache、哪些动作需要权限、工具出错时谁来重试、怎样可靠地比较两个版本——同一个模型，Harness 写得好坏能让成功率差出一大截。',
            warn: '本页所有数字都来自一个 5 个房间的玩具环境和 17 万参数的模型，用来说明机制和现象（模仿学习的捷径、采样与验证、故障与重试），不代表真实 Agent 的水平。',
        },
    ],

    graph: {
        title: 'Agent · 一局里的数据流',
        intro: '节点 = 页面里真实执行的计算。<b>HISTORY</b> 节点在一局里循环多次：模型写一个调用 → Harness 在房子里执行 → 观察写回上下文 → 模型写下一个调用。',
        common: { title: 'Common · 本系列自带的 C++ 库 + 第 16 篇的 C++（对话格式、会话、约束解码、Harness 主循环）', open: false, blocks: LM.CPP.blocks(['gpt']).concat([{ label: '━ 第 16 篇：对话格式 ━', code: AGT_CPP.chat }, { label: '━ 第 16 篇：loss mask ━', code: AGT_CPP.train }, { label: '━ 第 16 篇：KV cache 会话 ━', code: AGT_CPP.session }, { label: '━ 第 16 篇：Harness 主循环 ━', code: AGT_CPP.loop }, { label: '━ 第 16 篇：约束解码 ━', code: AGT_CPP.grammar }]) },
        types: { tok: { color: '#56b6c2', label: 'tok · token 序列' }, act: { color: '#e8c25a', label: 'act · 动作（工具调用）' }, obs: { color: '#e0a060', label: 'obs · 观察（工具结果）' }, st: { color: '#5ad07a', label: 'st · 环境状态' } },
        nodes: [
            { id: 'task', title: 'houseTask · newHouse', tag: 'CPU', category: 'bake', lifetime: 'param', step: 'env', desc: '任务 + 随机摆放的物品', dispatch: { kind: 'CPU', name: 'houseTask · newHouse', count: '每局一次' }, outputs: [{ name: 'State', format: '房间 · 物品 · 手', type: 'st' }, { name: 'Prompt', format: '<sys>可用:看去拿放<user>…<bot>', type: 'tok' }], detail: { preview: 'mapNow' } },
            { id: 'policy', title: '模型（π）', tag: 'GEMM', lifetime: 'history', step: 'rollout', desc: '读历史，写下一个调用', dispatch: { kind: 'CPU', name: 'feed · maskLogits · sample', count: '每个调用约 6 个 token' }, inputs: [{ name: 'Prompt', format: '第一步', type: 'tok' }, { name: 'History', format: '之后：调用 + 观察', type: 'tok', history: true }], outputs: [{ name: 'Action', format: '<call>去(书房)</call>', type: 'act' }], detail: { preview: 'agentTrace' } },
            { id: 'env', title: 'houseStep', tag: 'CPU', lifetime: 'history', step: 'env', desc: '执行动作，返回观察', dispatch: { kind: 'CPU', name: 'houseTools.run', count: '每个调用一次' }, inputs: [{ name: 'Action', format: '工具 + 参数', type: 'act' }, { name: 'State', format: '当前状态', type: 'st' }], outputs: [{ name: 'Obs', format: '<res>书房:苹果</res>', type: 'obs' }, { name: 'State′', format: '新状态', type: 'st' }] },
            { id: 'append', title: 'append', tag: 'CPU', lifetime: 'history', step: 'rollout', desc: '动作 + 观察写回上下文', dispatch: { kind: 'CPU', name: 'encode · push', count: '每个调用' }, inputs: [{ name: 'Obs', format: '观察', type: 'obs' }], outputs: [{ name: 'History', format: '[n] token', type: 'tok' }] },
            { id: 'judge', title: 'houseSuccess · classify', tag: 'CPU', category: 'output', lifetime: 'frame', step: 'eval', desc: '按环境状态判定成败', dispatch: { kind: 'CPU', name: 'houseSuccess · classify', count: '每局一次' }, inputs: [{ name: 'State′', format: '最终状态', type: 'st' }], outputs: [{ name: 'Result', format: '成功 / 失败类型', type: 'st' }], detail: { preview: 'succBars' } },
            { id: 'expert', title: 'houseExpert', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'expert', desc: 'DFS + 最短路径（离线造数据）', dispatch: { kind: 'CPU', name: 'expertMove', count: '训练时每步 16 局' }, outputs: [{ name: 'Demos', format: '轨迹（模仿学习）', type: 'tok' }], detail: { preview: 'expChips' } },
        ],
    },
    references: [
        { title: 'ReAct: Synergizing Reasoning and Acting in Language Models', author: 'S. Yao et al.', venue: 'ICLR', year: 2023, note: '「思考 → 行动 → 观察」交替的 Agent 循环' },
        { title: 'Reflexion: Language Agents with Verbal Reinforcement Learning', author: 'N. Shinn et al.', venue: 'NeurIPS', year: 2023, note: '失败后写出反思，下一次尝试时作为上下文' },
        { title: 'Tree of Thoughts: Deliberate Problem Solving with Large Language Models', author: 'S. Yao et al.', venue: 'NeurIPS', year: 2023, note: '在思考步骤上做树搜索' },
        { title: 'A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning', author: 'S. Ross, G. Gordon, D. Bagnell', venue: 'AISTATS', year: 2011, note: 'DAgger：模仿学习的误差累积与对策' },
        { title: 'Self-Consistency Improves Chain of Thought Reasoning in Language Models', author: 'X. Wang et al.', venue: 'ICLR', year: 2023, note: '多次采样 + 多数投票' },
        { title: 'Evaluating Large Language Models Trained on Code', author: 'M. Chen et al.', venue: 'arXiv', year: 2021, note: 'pass@k 的无偏估计' },
        { title: 'SWE-bench: Can Language Models Resolve Real-World GitHub Issues?', author: 'C. E. Jimenez et al.', venue: 'ICLR', year: 2024, note: '真实代码仓库上的 Agent 评测' },
        { title: 'WebArena: A Realistic Web Environment for Building Autonomous Agents', author: 'S. Zhou et al.', venue: 'ICLR', year: 2024, note: '网页任务上的 Agent 评测' },
    ],
});
