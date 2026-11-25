// 16_tool_calling.lab.js — 系列第 16 篇：工具调用（Tool Calling）的实现原理
// 构建（在 pages/ 下）：node <skill>/scripts/build_doc.mjs llm_common.js agent_common.js agent_cpp.js agent_weights.js 16_tool_calling.lab.js -o 16_tool_calling.html
// 模型：tools/agent_train.js 离线训练的「会用工具的迷你 GPT」（agent_weights.js）；页面里的解码、解析、执行、评测全部用 agent_common.js 的同一份代码实时运行。
const P16 = (() => {
    const QTYPES = ['add', 'sub', 'mul', 'fact', 'where'], QNAME = { add: '加法', sub: '减法', mul: '乘法', fact: '常识', where: '现在在哪' };
    const TOOLSETS = [['算', '查'], ['算'], ['查'], []], TSNAME = ['算 + 查', '只有算', '只有查', '无工具'];
    function loadModel() { const w = AGT_W.main, m = LM.gpt.create(w.cfg, 1); LM.gpt.loadWeights(m, w.b64); return m; }
    /** 第 qi 道 qtype 类型的题（固定种子）与这一局的「动态世界」 */
    function makeQ(qtype, qi) { return { task: AGT.qaTask(LM.rng(1009 + qi * 31 + QTYPES.indexOf(qtype) * 7919), qtype), world: AGT.newWorld(LM.rng(7001 + qi)) }; }
    const expected = (q, tools) => AGT.qaExpert(q.task, tools, q.world).filter((t) => t.role === 'bot').pop().text.replace('<eos>', '');
    const episode = (M, q, tools, o) => AGT.newEpisode(M, AGT.sysText(tools), q.task.q, AGT.qaEnv(tools, q.world), o);
    const isArith = (k) => k === 'add' || k === 'sub' || k === 'mul';
    /** Step 03：P(<call>) 表——每种题 × 每组可用工具，各 8 道题，只看 <bot> 之后第一个 token 的分布 */
    function decideBench(M) {
        const res = {}, jobs = [];
        QTYPES.forEach((k) => TOOLSETS.forEach((ts, j) => { for (let i = 0; i < 8; i++) { const q = makeQ(k, 100 + i); jobs.push({ kind: 'prefix', prefix: `<sys>${AGT.sysText(ts)}<user>${q.task.q}<bot>`, done: (p) => { const key = k + '|' + j; (res[key] = res[key] || []).push(p[AGT.SP.call]); } }); } }));
        return Object.assign(AGT.newBench(M, jobs), { res });
    }
    /** Step 05：答对率——有工具（算 + 查）/ 没有工具；Step 04：不执行工具、让模型自己写结果 */
    function gainBench(M) {
        const res = {}, jobs = [], add = (key, ok) => { const r = (res[key] = res[key] || { ok: 0, n: 0, fails: [] }); r.n++; if (ok === true) r.ok++; else if (ok && ok.fail) r.fails.push(ok.fail); };
        QTYPES.forEach((k) => [['算', '查'], []].forEach((ts) => { for (let i = 0; i < 20; i++) { const q = makeQ(k, 200 + i), want = expected(q, ts), key = `${k}|${ts.length ? 'tool' : 'none'}`; jobs.push({ kind: 'episode', start: () => episode(M, q, ts, { greedy: true, maxCalls: 3 }), done: (r) => add(key, r.ok && r.answer === want ? true : { fail: `${q.task.q} → ${r.answer || '（' + r.reason + '）'}` }) }); } }));
        ['add', 'sub', 'mul'].forEach((k) => { for (let i = 0; i < 20; i++) { const q = makeQ(k, 200 + i), want = expected(q, ['算']); jobs.push({ kind: 'episode', start: () => episode(M, q, ['算'], { greedy: true, maxCalls: 3, noExec: true }), done: (r) => { const fake = (r.events.find((e) => e.noExec) || {}).res; add(`${k}|fake`, r.ok && r.answer === want ? true : { fail: `${q.task.q} → 自己写的结果「${fake ?? ''}」→ ${r.answer}` }); add(`${k}|fakeRes`, fake === q.task.ans ? true : { fail: '' }); } }); } });
        return Object.assign(AGT.newBench(M, jobs), { res });
    }
    /** Step 06：调用解析失败率 vs 温度——不约束 / 约束解码；每格 24 道算术题 */
    const TEMPS = [0.6, 1.0, 1.4, 1.8, 2.4];
    function grammarBench(M) {
        const res = {}, jobs = [];
        TEMPS.forEach((t, ti) => [false, true].forEach((con) => { for (let i = 0; i < 24; i++) { const k = ['add', 'sub', 'mul'][i % 3], q = makeQ(k, 300 + i), key = `${ti}|${con ? 1 : 0}`; jobs.push({ kind: 'episode', start: () => episode(M, q, ['算', '查'], { temp: t, rand: LM.rng(5000 + i * 13 + ti), constrain: con, maxCalls: 3 }), done: (r) => { const rr = (res[key] = res[key] || { bad: 0, n: 0, ok: 0 }); rr.n++; if (r.events.some((e) => e.type === 'parse-error' || (e.type === 'call' && e.res === '没有这个工具') || (e.type === 'call' && e.res === '错误'))) rr.bad++; if (r.ok && r.answer === expected(q, ['算', '查'])) rr.ok++; } }); } }));
        return Object.assign(AGT.newBench(M, jobs), { res });
    }
    const mean = (a) => (a && a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
    /** 进度提示（训练 / 评测类预览在数据还没算完时画一行字） */
    const wait = (c, B, what) => LM.draw.text(c, [{ text: `        ${what}：${B ? `${B.k} / ${B.jobs.length}` : '等待开始（滚动到本步时开始计算）'}`, color: LM.C.gray }], { px: 12 });
    /** 一组 JSON schema 风格的工具定义（真实 API 的写法），用来和本页的「可用:算查」比 token 数 */
    const JSON_TOOLS = '[{"name":"calculator","description":"Evaluate an arithmetic expression with two integers","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"e.g. 37+48"}},"required":["expression"]}},{"name":"where_is","description":"Return where an animal is right now","parameters":{"type":"object","properties":{"animal":{"type":"string"}},"required":["animal"]}}]';
    return { QTYPES, QNAME, TOOLSETS, TSNAME, TEMPS, loadModel, makeQ, expected, episode, isArith, decideBench, gainBench, grammarBench, mean, wait, JSON_TOOLS };
})();

PipelineLab.mount({
    paramSteps: { qtype: ['template', 'train', 'decide', 'loop', 'grammar', 'choice'], qi: ['template', 'train', 'decide', 'loop', 'grammar', 'choice'], tools: ['template', 'train', 'decide', 'loop', 'grammar', 'choice'], temp: ['template', 'decide', 'loop', 'grammar', 'choice'], constrain: ['loop', 'grammar'], choice: ['template', 'decide', 'loop', 'grammar', 'choice'], exec: ['template', 'loop', 'grammar', 'choice'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '16 · 工具调用：模型怎样「调用函数」',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 16,
    css: LM.CSS,
    tagColors: { '格式': 'blue', '训练': 'magenta', '解码': 'teal', 'Harness': 'rust', '评测': 'green', '生产版': 'gray' },

    params: [
        { id: 'qtype', label: '问题类型', options: [{ value: 'add', label: '加法' }, { value: 'sub', label: '减法' }, { value: 'mul', label: '乘法' }, { value: 'fact', label: '常识（背得下来）' }, { value: 'where', label: '现在在哪（每局不同）' }], value: 'add', affects: 'param', desc: '要问模型的题。算术和「现在在哪」需要工具：前者模型算不准，后者每局都变、根本背不下来；常识题模型自己就知道。' },
        { id: 'qi', label: '题目编号', min: 0, max: 19, step: 1, value: 3, affects: 'param', desc: '同一类题里的第几道（换数字 / 换动物；「现在在哪」的答案也跟着这一局的随机世界变）。' },
        { id: 'tools', label: '可用工具', options: [{ value: '算查', label: '算 + 查' }, { value: '算', label: '只有算' }, { value: '查', label: '只有查' }, { value: '', label: '无' }], value: '算查', affects: 'param', desc: '写进系统提示的工具列表（「可用:算查」）。模型从训练数据里学会了：列表里有这个工具才调用，没有就自己回答或说查不到。' },
        { id: 'temp', label: '采样温度', min: 0, max: 2.4, step: 0.2, value: 0, affects: 'param', desc: '0 = 每步取概率最大的 token（贪心）；越高越随机，工具调用越容易写错格式——第 06 步的约束解码就是为它准备的。' },
        { id: 'constrain', label: '约束解码', value: false, affects: 'param', desc: '在 &lt;call&gt; 里面只允许合乎语法的下一个 token（工具名必须在列表里、参数必须是数字 / 动物名…），其余 logits 置为 −∞。' },
        { id: 'choice', label: 'tool_choice', options: [{ value: 'auto', label: 'auto（模型决定）' }, { value: 'none', label: 'none（禁止调用）' }, { value: 'required', label: 'required（必须调用）' }], value: 'auto', affects: 'param', desc: 'Harness 对第一个 token 的强制：none 把 &lt;call&gt; 的 logit 设成 −∞；required 把除 &lt;call&gt; 以外的都设成 −∞；auto 不干预。' },
        { id: 'exec', label: '执行工具', value: true, affects: 'param', desc: '关掉：Harness 看到调用后不去执行，只写一个 &lt;res&gt;，让模型自己把「结果」写下去——它从来没被训练过写结果，于是开始胡编。' },
    ],

    stages: [
        { id: 'model', lifetime: 'resident', run(S) { S.M = P16.loadModel(); } },
        {
            id: 'q', lifetime: 'param', run(S) {   // 当前这道题、这一局的世界、本轮可用的工具
                S.q = P16.makeQ(S.qtype, S.qi); S.ts = S.tools.split('');
                S.expert = AGT.serialize(AGT.sysText(S.ts), S.q.task.q, AGT.qaExpert(S.q.task, S.ts, S.q.world));
            },
        },
        {
            id: 'ep', lifetime: 'param', run(S) {   // 用当前设置把这道题完整跑一遍（几十个 token，一次跑完）
                const o = { greedy: !S.temp, temp: S.temp, rand: LM.rng(42 + S.qi), constrain: S.constrain, toolChoice: S.choice, noExec: !S.exec, keepTop: 6, maxCalls: 3 };
                S.ep = AGT.runEpisode(S.M, AGT.sysText(S.ts), S.q.task.q, AGT.qaEnv(S.ts, S.q.world), o);
                S.want = P16.expected(S.q, S.ts);
                S.epRoles = AGT.roles(S.ep.ids);
                S.firstTop = S.ep.tops[0];
                // 同一道题在三种 tool_choice 下各跑一遍（Step 07）
                S.choices = ['auto', 'none', 'required'].map((tc) => AGT.runEpisode(S.M, AGT.sysText(S.ts), S.q.task.q, AGT.qaEnv(S.ts, S.q.world), Object.assign({}, o, { toolChoice: tc, keepTop: 0 })));
            },
        },
        {
            id: 'anim', lifetime: 'frame', run(S) {   // 动画：提示一次性出现，之后每秒 8 个 token；放完停 1.5 秒再重来
                const n = S.ep.ids.length, p0 = S.ep.ids.indexOf(AGT.SP.bot) + 1, dur = (n - p0) / 8 + 1.5, t = S.t % dur;
                S.reveal = Math.min(n, p0 + Math.floor(t * 8));
            },
        },
        { id: 'bDecide', lifetime: 'history', lazy: true, run(S) { if (!S.bd) S.bd = P16.decideBench(S.M); AGT.benchWork(S.bd, 90); } },
        { id: 'bGain', lifetime: 'history', lazy: true, run(S) { if (!S.bg) S.bg = P16.gainBench(S.M); AGT.benchWork(S.bg, 90); } },
        { id: 'bGram', lifetime: 'history', lazy: true, run(S) { if (!S.bgr) S.bgr = P16.grammarBench(S.M); AGT.benchWork(S.bgr, 90); } },
    ],

    intro: {
        title: '工具调用：模型只会输出 token，「调用函数」是 Harness 和模型合演的一场戏',
        flow: ['工具写进系统提示', '<call> 特殊 token', '只在回答上算损失', '下一个 token = 要不要调用', 'Harness：解析 → 执行 → 写回', '约束解码', 'tool_choice'],
        html: LM.navHtml('16') + `
            <p>前面几篇的语言模型只会做一件事：给定前文，给下一个 token 打分。它不能上网、不会真的做加法、也不知道「现在」是几点。所谓<strong>工具调用</strong>（function calling），并没有给模型接上什么新器官：模型只是学会了在合适的时候<strong>写出一段特殊格式的文字</strong>——<code>&lt;call&gt;算(37+48)&lt;/call&gt;</code>；包在模型外面的程序（<strong>Harness</strong>）一看到这段文字就停下生成，解析出工具名和参数，真的去执行，再把结果 <code>&lt;res&gt;85&lt;/res&gt;</code> 写回上下文，然后让模型接着往下写。</p>
            <p>本页用一个 17 万参数的迷你 GPT 把这件事从头到尾做一遍。它在 Node 里用同一份代码离线训练了 8000 步（<code>tools/agent_train.js</code>，权重 <code>agent_weights.js</code>），训练数据是合成的对话：两位数的加减乘（工具「算」）、某只动物<strong>此刻</strong>在哪（工具「查」，每局随机，背不下来）、玩具世界的常识（不需要工具），以及第 17、18 篇要用的多步任务。页面里的分词、带 KV cache 的解码、流式解析、执行工具和所有评测都在你的浏览器里实时运行。</p>`,
    },

    steps: [
        {
            id: 'template', title: '对话模板：工具写进系统提示，调用写成特殊 token', tag: '格式',
            desc: '模型看到的永远是一串 token。一次带工具的对话被拼成：<code>&lt;sys&gt;可用:算查&lt;user&gt;问题&lt;bot&gt;</code>，模型从 <code>&lt;bot&gt;</code> 之后开始写。工具调用用两个<strong>特殊 token</strong> <code>&lt;call&gt;</code>、<code>&lt;/call&gt;</code> 包起来，工具结果用 <code>&lt;res&gt;</code>、<code>&lt;/res&gt;</code> 包起来——它们各占一个 id，普通文字里永远不会出现，所以 Harness 能可靠地识别「这里开始调用了」。真实系统做法相同，只是格式更啰嗦：工具定义是一段 JSON schema（名字、说明、参数类型），由对话模板渲染进系统提示；调用通常是一段 JSON，夹在 <code>&lt;tool_call&gt;</code> 之类的特殊 token 之间。下图的颜色就是这段文字<strong>由谁写</strong>：灰 = 系统提示，蓝 = 用户，橙 = Harness 写入（包括 <code>&lt;bot&gt;</code> 和工具结果），绿 = 模型生成，黄 = 模型生成的工具调用。',
            formula: [
                '<sys>可用:算查<user>37+48等于多少？<bot>  →  <call>算(37+48)</call>  <res>85</res>  37+48等于85。<eos>',
                '|词表| = 9 个特殊 token + 138 个字符 = 147      // <pad> <eos> <sys> <user> <bot> <call> </call> <res> </res>',
            ],
            code: { title: 'C++ · systemPrompt / serialize / calcExample', src: AGT_CPP.chat },
            previews: [
                {
                    id: 'tokChips', label: '当前这道题完整的一次对话（token 级）', sub: '每块 = 一个 token · 颜色 = 谁写的 · 模型用当前的设置真实生成，工具由 Harness 真实执行', large: true, size: [720, 140],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), ep = S.ep, n = ep.ids.length;
                        AGT.drawRoleLegend(ctx, 8, 12);
                        const r = AGT.drawChips(ctx, ep.ids, { roles: S.epRoles, y0: 28, w: 704, px: 13 });
                        const ok = ep.ok && ep.answer === S.want;
                        ctx.font = LM.FONT(12); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = ok ? LM.C.green : LM.C.red;
                        ctx.fillText(`${n} 个 token · 模型生成 ${ep.generated} 个 · 工具调用 ${ep.calls} 次 · 结束原因 ${ep.reason} · ${ok ? '答对 ✓' : `答错 ✗（标准答案：${S.want}）`}`, 8, r.y + 10);
                    },
                },
                {
                    id: 'vocabTbl', label: '9 个特殊 token', sub: '它们在普通文字里永远不会出现：Harness 靠它们切分「谁在说话」和「哪里是调用」', size: [360, 250],
                    render: (c, S) => {
                        const C = LM.C, rows = [['<pad>', '补齐 batch（不计损失）'], ['<eos>', '模型说完了'], ['<sys>', '系统提示开始（Harness 写）'], ['<user>', '用户说话开始'], ['<bot>', '轮到模型（Harness 写）'], ['<call>', '工具调用开始（模型写）'], ['</call>', '调用结束 → Harness 停下来执行'], ['<res>', '工具结果开始（Harness 写）'], ['</res>', '结果结束 → 模型接着写']];
                        LM.draw.text(c, rows.map(([t, d], i) => [{ t: `${String(i).padStart(2)}  `, color: C.dim }, { t: t.padEnd(8), color: C.yellow, bold: true }, { t: d, color: C.text }]).concat([[{ t: `其余 ${AGT.V - 9} 个 id 是单个汉字、数字和标点（字符级分词，第 06 篇）`, color: C.gray }]]), { px: 12, lineGap: 6 });
                    },
                },
            ],
            notes: '换 <b>问题类型</b> 和 <b>可用工具</b> 看模型写出的东西怎么变：算术题有「算」时先调用再回答；「现在在哪」有「查」时调用、没有时老实说「我查不到」；常识题无论有没有工具都直接回答。',
            warn: '这里的工具「定义」只是一个字的名字：模型能用哪些工具、参数长什么样，全靠训练数据教会。真实模型读的是 JSON schema 里的自然语言说明，所以能用训练时没见过的新工具——这个能力来自大规模的指令微调，2 层的迷你模型做不到（第 08 步）。',
        },
        {
            id: 'train', title: '训练：只在模型该写的 token 上算损失', tag: '训练',
            desc: '训练数据是一条条<strong>专家示范</strong>：问题 + 正确的调用 + 工具返回的真实结果 + 正确的回答。和第 09 篇的 SFT 一样，交叉熵只算在模型自己要写的位置上（绿色和黄色的 token）：系统提示、用户的话、尤其是<strong>工具结果</strong>都不计损失。为什么工具结果不能学？因为它本来就不该由模型来写——如果让模型学着预测 <code>&lt;res&gt;</code> 里的内容，它学到的是「编一个看起来像的结果」，这正是我们想用工具消除的幻觉。打开上方的 <b>执行工具</b> 开关试试：Harness 不执行、只写一个 <code>&lt;res&gt;</code>，模型就只能自己往下编（第 04 步的统计）。训练配方：8000 步 AdamW（峰值学习率 2e-3，余弦衰减），每步 32 条单轮问答（长度 48）+ 16 条多步任务（长度 192，第 17 篇），8 个线程数据并行，约 45 分钟。',
            formula: [
                'L = − Σ_t m_t · log p(x_{t+1} | x_{≤t}) / Σ_t m_t      m_t = 1 ⇔ x_{t+1} 是模型写的',
                'y_t = x_{t+1}（m_t = 1）或 −1（不计损失）      // C++ 版的 crossEntropy 遇到 −1 直接跳过',
            ],
            code: { title: 'C++ · maskedTargets / sftStep', src: AGT_CPP.train },
            previews: [
                {
                    id: 'maskChips', label: '一条训练样本与它的 loss mask（当前题目 × 当前可用工具）', sub: '实线块 = 计入损失（模型要学会写出它）· 虚线块 = 不计损失（系统提示、用户、Harness 写的工具结果）', large: true, size: [720, 120],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), ex = S.expert, mask = ex.role.map((r) => r === 'bot');
                        const n = mask.filter(Boolean).length;
                        const r = AGT.drawChips(ctx, ex.ids, { y0: 10, w: 704, px: 13, mask });
                        ctx.font = LM.FONT(12); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`${ex.ids.length} 个 token，其中 ${n} 个计入损失（${(100 * n / ex.ids.length).toFixed(0)}%）：模型要学的只是「什么时候调用、怎么写调用、怎么用结果回答」`, 8, r.y + 10);
                    },
                },
                {
                    id: 'trainCurve', label: '离线训练的损失（每 50 步平均）', sub: '对数纵轴 · 蓝 = 单轮问答（本篇）· 橙 = 多步任务（第 17 篇）· 只在 mask = 1 的位置上平均', size: [360, 210],
                    render: (c, S) => { const cv = AGT_W.main.curve; PL.draw.plot(c, { x: [0, AGT_W.main.steps], y: [0.003, 5], logY: true, xLabel: '训练步', series: [{ data: cv.map((p) => p[1]), x: cv.map((p) => p[0]), color: LM.C.blue, width: 1.5, label: '单轮问答' }, { data: cv.map((p) => p[2]), x: cv.map((p) => p[0]), color: LM.C.orange, width: 1.5, label: '多步任务' }] }); },
                },
                {
                    id: 'trainEval', label: '训练中的评测：答对率（贪心解码，每格 60 题，最后一次 200 题）', sub: '实线 = 有工具 · 虚线 = 没有工具（模型自己算）· 蓝 加法 · 绿 减法 · 橙 乘法', size: [360, 210],
                    render: (c, S) => {
                        const E = AGT_W.main.evals, xs = E.map((e) => e[0]), g = (k) => E.map((e) => e[1][k].acc), cols = { add: LM.C.blue, sub: LM.C.green, mul: LM.C.orange };
                        const series = []; for (const k of ['add', 'sub', 'mul']) series.push({ data: g(k + '+tool'), x: xs, color: cols[k], width: 2, label: P16.QNAME[k] }, { data: g(k + '-tool'), x: xs, color: cols[k], dash: [4, 3] });
                        PL.draw.plot(c, { x: [0, AGT_W.main.steps], y: [0, 1.4], xLabel: '训练步', yLabel: '答对率', series, hmarks: [{ y: 1, label: '', color: '#3a3f46' }] });
                    },
                },
            ],
            notes: '换 <b>可用工具</b>：同一道算术题，有「算」时样本里多了一次调用和一段（不计损失的）结果；没有工具时样本就是直接回答。离线评测（训练结束时每格 200 题）里，有工具时三种算术都是 100%；没有工具时加法 88.5%、减法 32%、乘法 25%——这个 17 万参数的模型背下了不少加法，减法和乘法却基本靠猜，而「把算式抄进 <code>算(…)</code>、再把结果抄进回答」它学得又快又准。',
            warn: '真实模型的心算比这个小模型强得多，但多位数乘法、最新的事实、私有数据这类事情同样必须交给工具。另一个常见的坑：如果训练数据里的工具结果有错（例如抓取的网页过时），模型会学到「结果和回答可以不一致」——工具结果虽然不计损失，但回答是照着它写的。',
        },
        {
            id: 'decide', title: '要不要调用工具：只是下一个 token 的概率', tag: '解码',
            desc: '模型并没有一个单独的「决策模块」。<code>&lt;bot&gt;</code> 之后它给第一个 token 打分：<code>&lt;call&gt;</code> 的概率高，就是「我要调用工具」；回答的第一个字概率高，就是「我直接回答」。这个判断取决于两样东西：<strong>问题</strong>（算术、此刻的位置需要工具；常识不需要）和<strong>系统提示里列出的工具</strong>（列表里没有「算」，就不该调用「算」）。下面的热图对每种题、每组可用工具各取 8 道题，只做一次前向（只读提示，不生成），把 P(<code>&lt;call&gt;</code>) 取平均——全部在你的浏览器里实时算出。',
            formula: [
                'p(<call> | <sys>可用:T <user>q <bot>)      // 第一个生成 token 的分布：调用 vs 直接回答',
                'KV cache：每个 token 只前向一次，后面的 token 读前面存下的 K、V      // 生成 n 个 token 只要 n 次单 token 前向',
            ],
            code: { title: 'C++ · Session / newSession / feed / callProbability', src: AGT_CPP.session },
            previews: [
                {
                    id: 'firstTok', label: '&lt;bot&gt; 之后第一个 token 的分布（当前题目 × 当前可用工具）', sub: '黄 = 实际选中的 token · 这一步决定了「调用」还是「直接回答」', large: true, size: [360, 230],
                    render: (c, S) => { const T = S.firstTop; if (!T) return; const show = (id) => (AGT.isSpecial(id) ? AGT.itos[id] : `「${AGT.itos[id]}」`); LM.draw.bars(c, T.top.map((x) => x[1]), T.top.map((x) => show(x[0])), { horizontal: true, hi: T.top.findIndex((x) => x[0] === T.id), range: [0, 1], labelPx: 13, title: `${S.q.task.q}（可用：${S.tools || '无'}）` }); },
                },
                {
                    id: 'callHeat', needs: ['bDecide'], live: true, label: 'P(&lt;call&gt;)：每种题 × 每组可用工具（各 8 道题的平均）', sub: '行 = 题型 · 列 = 系统提示里的工具 · 亮 = 倾向调用 · 页面里实时计算（每帧约 90 次单 token 前向）', size: [360, 230],
                    render: (c, S) => {
                        const B = S.bd; if (!B || !Object.keys(B.res).length) return P16.wait(c, B, '正在计算');
                        const data = []; P16.QTYPES.forEach((k) => P16.TOOLSETS.forEach((ts, j) => data.push(P16.mean(B.res[k + '|' + j]))));
                        LM.draw.heat(c, data.map((v) => (isFinite(v) ? v : 0)), 5, 4, { rowLabels: P16.QTYPES.map((k) => P16.QNAME[k]), colLabels: P16.TSNAME, range: [0, 1], map: 'viridis', digits: 2, title: `${B.k} / ${B.jobs.length} 道`, titleX: 46 });
                    },
                },
            ],
            notes: '把 <b>可用工具</b> 改成「无」：算术题的 P(<code>&lt;call&gt;</code>) 掉到几乎 0，模型改为直接写答案（加法多半对，减法、乘法多半错，见第 05 步）；改成「只有查」：算术题也不会去调用「查」——它学到的是「需要的那个工具在列表里才调用」。常识题那一行无论哪一列都接近 0。',
            warn: '真实模型的「该不该调用」也是这样一个概率，只是受系统提示里的大段说明影响。所以工具描述写得含糊、或者列出几十个相似的工具时，模型会调用错工具或者该调用时不调用——这是工具调用最常见的线上问题之一，靠改描述和补训练数据解决。',
        },
        {
            id: 'loop', title: 'Harness 主循环：生成 → 解析 → 执行 → 写回 → 继续', tag: 'Harness',
            desc: '<strong>Harness</strong> 是包在模型外面的那段程序。它的主循环很短：每次让模型生成一个 token；如果是 <code>&lt;call&gt;</code>，开始累积调用文本；一看到 <code>&lt;/call&gt;</code> 就<strong>停止生成</strong>，解析出工具名和参数（格式不对就写回「格式错误」），检查工具在不在列表里，真的去执行，把 <code>&lt;res&gt;结果&lt;/res&gt;</code> 追加进上下文，然后让模型继续——模型此时看到的上下文里已经有了正确的结果，只要把它抄进回答。遇到 <code>&lt;eos&gt;</code> 就结束。因为用了 KV cache，前面的 token 不用重算：工具结果的几个 token 喂进去、再生成回答，只需要几次单 token 前向。下面的动画按时间重放当前这道题的一次真实运行。',
            formula: [
                'while not <eos>:  t = sample(p(· | ctx));  ctx += t;  if t == </call>: (f, a) = parse(call); r = f(a); ctx += <res> r </res>',
            ],
            code: { title: 'C++ · ToolBox / calc / qaTools / parseCall / runEpisode', src: AGT_CPP.loop },
            previews: [
                {
                    id: 'trace', live: true, label: '重放：Harness 和模型轮流写这条对话', sub: '每秒 8 个 token · 黄框 = 正在写的 token · 下方事件 = Harness 做了什么', large: true, size: [720, 210],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), ep = S.ep, k = S.reveal;
                        AGT.drawRoleLegend(ctx, 50, 12);
                        const r = AGT.drawChips(ctx, ep.ids, { roles: S.epRoles, y0: 30, w: 704, px: 13, upto: k, hi: k - 1 });
                        // 事件：根据已经放出的 token 推出 Harness 到了哪一步
                        const shown = ep.ids.slice(0, k), lines = [];
                        let ci = 0; for (let i = 0; i < shown.length; i++) { if (shown[i] === AGT.SP.ecall) { const e = ep.events.filter((x) => x.type === 'call' || x.type === 'parse-error' || x.type === 'denied')[ci++]; if (e) lines.push(e.type === 'call' ? (e.noExec ? `看到 </call> → 解析出 ${e.tool}(${e.arg}) → 不执行，只写 <res>，让模型自己写结果` : `看到 </call> → 解析出 ${e.tool}(${e.arg}) → 执行 → 「${e.res}」→ 写回 <res>${e.res}</res>`) : `看到 </call> → 解析失败「${e.text}」→ 写回 <res>格式错误</res>`); } }
                        if (k >= ep.ids.length) lines.push(ep.reason === 'eos' ? `模型写出 <eos> → 结束。回答：${ep.answer}` : `结束（${ep.reason}）`);
                        ctx.font = LM.FONT(12); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        lines.forEach((ln, i) => { ctx.fillStyle = i === lines.length - 1 ? LM.C.yellow : LM.C.gray; ctx.fillText('▸ ' + ln, 8, r.y + 10 + i * 18); });
                    },
                },
                {
                    id: 'fakeBars', needs: ['bGain'], live: true, label: '不执行工具会怎样：让模型自己写 &lt;res&gt; 里的结果', sub: '每种算术 20 道题 · 蓝 = Harness 执行工具时的答对率 · 红 = 模型自己写结果时「结果」写对的比例', size: [360, 210],
                    render: (c, S) => {
                        const B = S.bg; if (!B || !B.res['add|fake']) return P16.wait(c, B, '正在评测');
                        const f = (key) => (B.res[key] ? B.res[key].ok / B.res[key].n : 0), vals = [], labels = [], cols = [];
                        for (const k of ['add', 'sub', 'mul']) { vals.push(f(k + '|tool'), f(k + '|fakeRes')); labels.push(P16.QNAME[k] + ' · 执行', P16.QNAME[k] + ' · 自己编'); cols.push(LM.C.blue, LM.C.red); }
                        LM.draw.bars(c, vals, labels, { horizontal: true, colors: cols, range: [0, 1], digits: 2, title: `${B.k} / ${B.jobs.length} 个回合` });
                    },
                },
            ],
            notes: '关掉 <b>执行工具</b> 再看重放：模型写完 <code>算(…)</code> 之后，Harness 只写了一个 <code>&lt;res&gt;</code>，模型就自己往下写「结果」——它从来没在这个位置上被训练过，写出来的通常是一串不相干的数字甚至句子。右图是 20 道题的统计。这就是「结果必须由 Harness 写」的原因：模型负责决定调用什么，计算本身交给真实的程序。',
            warn: '这个循环里每一步都可能出错：调用写到一半上下文就满了、参数不合法、工具抛异常、工具很慢、工具返回一大段文字把上下文撑爆、模型在结果之后又发起新的调用……第 18 篇把这些交给 Harness 逐一处理。',
        },
        {
            id: 'gain', title: '工具带来了什么：同一个模型，有工具 vs 没有工具', tag: '评测',
            desc: '用同一个模型、同一批题，分别在系统提示里给出「算 + 查」和「无」，贪心解码，看最终回答是否和标准答案一字不差。算术题：有工具时模型只要把算式抄对、再把结果抄对；没有工具时它只能「心算」，而 17 万参数的模型几乎学不会两位数加减乘。「现在在哪」：没有工具时正确的行为是说「我查不到」，有工具时才能答出这一局的真实位置。常识题：模型自己就记得，有没有工具都一样。下面的统计在你的浏览器里实时跑完（每格 20 道题）。',
            formula: [
                'acc = #{回答 == 标准答案} / n      // 贪心解码；「现在在哪」无工具时的标准答案 = 「我查不到…」',
            ],
            code: { title: 'C++ · runEpisode（同上）＋ 评测见第 18 篇', src: '// @nocompile\n// 评测只是把 runEpisode 跑 n 遍、比较 e.answer 与标准答案；完整的评测 Harness（固定种子、置信区间、配对比较）在第 18 篇\nfor (int i = 0; i < n; i++) { Episode e = runEpisode(m, v, systemPrompt(tools), questions[i], box, opts, rng); ok += e.reason == "eos" && e.answer == answers[i]; }' },
            previews: [
                {
                    id: 'accBars', needs: ['bGain'], live: true, label: '答对率：有工具（算 + 查）vs 没有工具', sub: '每格 20 道题 · 蓝 = 有工具 · 灰 = 没有工具 · 「现在在哪」无工具时的「答对」= 老实说查不到', large: true, size: [360, 250],
                    render: (c, S) => {
                        const B = S.bg; if (!B || !B.res['add|tool']) return P16.wait(c, B, '正在评测');
                        const f = (key) => (B.res[key] ? B.res[key].ok / B.res[key].n : NaN), vals = [], labels = [], cols = [];
                        for (const k of P16.QTYPES) { vals.push(f(k + '|tool'), f(k + '|none')); labels.push(P16.QNAME[k] + ' · 有工具', P16.QNAME[k] + ' · 无工具'); cols.push(LM.C.blue, '#6b7480'); }
                        LM.draw.bars(c, vals.map((v) => (isFinite(v) ? v : 0)), labels, { horizontal: true, colors: cols, range: [0, 1], digits: 2, title: `${B.k} / ${B.jobs.length} 个回合` });
                    },
                },
                {
                    id: 'failList', needs: ['bGain'], live: true, label: '没有工具时的错误回答（举例）', sub: '模型的「心算」：格式完全正确，数字却是编的——这就是幻觉', size: [360, 250],
                    render: (c, S) => {
                        const B = S.bg; if (!B || !B.res['add|none']) return P16.wait(c, B, '正在评测');
                        const lines = []; for (const k of ['add', 'sub', 'mul']) { const r = B.res[k + '|none']; if (r) r.fails.slice(0, 3).forEach((f) => lines.push([{ t: '✗ ', color: LM.C.red }, { t: f, color: LM.C.text }])); }
                        LM.draw.text(c, lines.length ? lines : [{ text: '（还没有错误）', color: LM.C.gray }], { px: 12, lineGap: 7 });
                    },
                },
            ],
            notes: '每格 20 道题的实时结果见左图：不用工具时加法 16/20、减法 8/20、乘法 6/20，用工具时全部 20/20。离线评测（训练结束时每格 200 题）给出同样的结论：有工具时算术与「现在在哪」都是 100%；没有工具时加法 88.5%、减法 32%、乘法 25%，「现在在哪」变成诚实的「查不到」，常识题两边相同。',
            warn: '这里的「答对」要求一字不差。真实系统评测工具调用时通常分开看：选对工具的比例、参数正确的比例、最终答案正确的比例（例如 Berkeley Function-Calling Leaderboard 用 AST 比对调用是否正确），因为三者出错的原因不同。',
        },
        {
            id: 'grammar', title: '约束解码：让调用一定能被解析', tag: '解码',
            desc: '采样温度一高，模型偶尔会写出 <code>算(37+)</code>、<code>查(小猫小猫)</code>、甚至一个不存在的工具名——Harness 解析失败，只能返回「格式错误」。<strong>约束解码</strong>（structured output / grammar-constrained decoding）从根上杜绝这件事：在 <code>&lt;call&gt;</code> 里面，每一步先算出「按语法，下一个 token 只能是哪些」，把其余 token 的 logit 置为 −∞ 再采样。本页的语法很小：工具名必须在可用列表里；「算」的参数是「1–3 位数字 运算符 1–3 位数字」；「查」的参数是 8 个动物名之一（逐字匹配前缀）。真实系统把 JSON schema 编译成有限状态机或下推自动机，每步给出合法 token 的集合。',
            formula: [
                'p̃(x_t) ∝ p(x_t) · 𝟙[x_t ∈ Allowed(prefix)]      // 语法之外的 token：logit = −∞',
                'Allowed(算(37) = {0–9 , + , − , ×}      Allowed(算(37+4) = {0–9 , )}      Allowed(查(小) = {猫, 狗, 兔, 鸟, 羊}',
            ],
            code: { title: 'C++ · argChoices / allowedNext / maskLogits', src: AGT_CPP.grammar },
            previews: [
                {
                    id: 'gramRate', needs: ['bGram'], live: true, label: '调用写坏的比例 vs 采样温度（每点 24 道算术题）', sub: '红 = 不约束 · 绿 = 约束解码 · 「写坏」= 解析失败、工具名不存在、或计算器收到非法算式', large: true, size: [360, 230],
                    render: (c, S) => {
                        const B = S.bgr; if (!B || !B.res['0|0']) return P16.wait(c, B, '正在评测');
                        const g = (con) => P16.TEMPS.map((t, i) => { const r = B.res[`${i}|${con}`]; return r ? r.bad / r.n : NaN; }), a = (con) => P16.TEMPS.map((t, i) => { const r = B.res[`${i}|${con}`]; return r ? r.ok / r.n : NaN; });
                        PL.draw.plot(c, { x: [0.5, 2.5], y: [0, 1], xLabel: '采样温度', yLabel: '比例', series: [{ data: g(0), x: P16.TEMPS, color: LM.C.red, width: 2, label: '写坏（不约束）' }, { data: g(1), x: P16.TEMPS, color: LM.C.green, width: 2, label: '写坏（约束）' }, { data: a(0), x: P16.TEMPS, color: LM.C.red, dash: [4, 3], label: '答对（不约束）' }, { data: a(1), x: P16.TEMPS, color: LM.C.green, dash: [4, 3], label: '答对（约束）' }] });
                    },
                },
                {
                    id: 'gramSteps', label: '当前这次运行：调用里每一步的候选（前 6 名）', sub: '划掉 = 被语法掩掉（约束解码打开时）· 黄 = 实际选中 · 打开「约束解码」并把温度调高试试', size: [360, 230],
                    render: (c, S) => {
                        const ep = S.ep, ids = ep.ids, lines = []; let inCall = false, prefix = '';
                        for (const T of ep.tops) {
                            const id = ids[T.pos];
                            if (id === AGT.SP.call) { inCall = true; prefix = ''; continue; }
                            if (!inCall) continue;
                            const al = AGT.allowedNext(prefix, S.ts);
                            lines.push([{ t: `${(prefix || '∅').padEnd(8, ' ')} → `, color: LM.C.dim }].concat(T.top.map(([i, p]) => ({ t: `${AGT.itos[i]} ${(100 * p).toFixed(0)}%  `, color: i === id ? LM.C.yellow : S.constrain && !al.has(i) ? '#55606b' : LM.C.text }))));
                            if (id === AGT.SP.ecall) { inCall = false; } else prefix += AGT.itos[id];
                            if (lines.length >= 12) break;
                        }
                        LM.draw.text(c, lines.length ? lines : [{ text: '这次运行没有工具调用（换一道算术题，或把可用工具设成「算 + 查」）', color: LM.C.gray }], { px: 11, lineGap: 5 });
                    },
                },
            ],
            notes: '温度 1.8 以下，两条红绿线都贴着 0：训练充分的格式很少写坏。温度升到 2.4，不约束时 24 次调用里有 8 次写坏；约束解码让「写坏」始终为 0——但它只保证<strong>能解析</strong>，不保证<strong>内容对</strong>：参数里的数字照样可能抄错（虚线 = 最终答对率：温度 1.8 时两边都是 19/24，2.4 时不约束 3/24、约束 7/24）。',
            warn: '约束解码改变的是采样分布，强行掐掉高概率的「错误」token 会把概率质量推给语法允许、但模型并不想写的 token；语法写得太严（例如只允许整数）还会让模型无法表达正确的东西。另一个代价是速度：复杂的 JSON schema 每一步都要算合法集合，生产系统（XGrammar、Outlines 等）为此做了预编译和缓存。',
        },
        {
            id: 'choice', title: 'tool_choice：auto / none / required 只是几个掩码', tag: '解码',
            desc: 'API 里常见的 <code>tool_choice</code> 参数，在 Harness 里的实现非常朴素：<strong>none</strong>——把 <code>&lt;call&gt;</code> 的 logit 设成 −∞，模型想调用也调用不了；<strong>required</strong>——在回答的第一个 token 上，把除 <code>&lt;call&gt;</code> 以外的 logit 都设成 −∞，模型必须先调用一个工具（再配合约束解码就能指定调用哪一个）；<strong>auto</strong>——不干预，让第 03 步的那个概率自己决定。下面把当前这道题在三种设置下各跑一遍。',
            formula: [
                'none: logit(<call>) = −∞      required: logit(x) = −∞ ∀x ≠ <call>（仅第一个 token）      auto: 不改',
            ],
            code: { title: 'C++ · maskLogits（见上一步）', src: '// @nocompile\n// tool_choice 的三种模式都在 maskLogits 里（第 06 步的代码块）：\n//   o.toolChoice == 1（none）     → logits[CALL] = −∞\n//   o.toolChoice == 2（required） → 第一个 token 上，除 CALL 以外全部 = −∞\n// 调用方只要改 EpisodeOptions：\nEpisodeOptions o; o.toolChoice = 2; Episode e = runEpisode(m, v, systemPrompt({ "算", "查" }), "37+48等于多少？", box, o, rng);' },
            previews: [
                {
                    id: 'choiceCmp', label: '同一道题，三种 tool_choice', sub: '模型、题目、采样设置都相同，只改 Harness 的掩码 · 右边 = 是否答对', large: true, size: [720, 220],
                    render: (c, S) => {
                        const { ctx } = PL.draw.ctx(c), names = ['auto', 'none', 'required'];
                        let y = 8;
                        S.choices.forEach((ep, i) => {
                            ctx.font = LM.FONT(12, true); ctx.fillStyle = S.choice === names[i] ? LM.C.yellow : LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(names[i], 8, y + 4);
                            const p0 = ep.ids.indexOf(AGT.SP.bot), r = AGT.drawChips(ctx, ep.ids.slice(p0), { x0: 80, y0: y, w: 560, px: 13 });
                            const ok = ep.ok && ep.answer === S.want; ctx.font = LM.FONT(12); ctx.fillStyle = ok ? LM.C.green : LM.C.red; ctx.textAlign = 'left'; ctx.fillText(ok ? '✓ 答对' : '✗', 650, y + 4);
                            y = r.y + 12;
                        });
                        ctx.font = LM.FONT(11); ctx.fillStyle = LM.C.gray; ctx.fillText(`标准答案：${S.want}`, 8, y + 2);
                    },
                },
                {
                    id: 'schemaCost', label: '工具定义要花多少 token', sub: '同样两个工具：本页的「可用:算查」vs JSON schema 写法（按本页的字符级分词数）· 每一轮对话都要重复带上', size: [360, 150],
                    render: (c, S) => { const a = AGT.sysText(['算', '查']).length + 1, b = P16.JSON_TOOLS.length; LM.draw.bars(c, [a, b], ['本页：可用:算查', 'JSON schema（2 个工具）'], { horizontal: true, colors: [LM.C.green, LM.C.orange], digits: 0, valueSuffix: ' token' }); },
                },
            ],
            notes: '默认是一道算术题：auto 和 required 都会调用「算」并答对。none 被禁止调用——可系统提示里明明列着计算器，模型已经「准备好」要调用，<code>&lt;call&gt;</code> 被屏蔽后它写出的是乱码（例如「到(399+87。」），有时干脆写出一个没有 <code>&lt;call&gt;</code> 的调用、被 Harness 判为非法 token。把 <b>可用工具</b> 改成「无」再看：模型会认真心算（加法 88.5% 都对）。「不许用」和「没有」对这个模型是两回事；真实的大模型在训练里见过「有工具但不许用」，不会这样。换成常识题：auto 和 none 直接回答；required 被迫先调用一个工具——模型从没在常识题上调用过，常常调用一个无关的工具，然后照样回答（或者答偏）：强制调用要慎用。',
            warn: '真实 API 的 tool_choice 还能指定「必须调用某一个工具」（OpenAI 的 {type: function, name}，Anthropic 的 {type: tool, name}），实现上同样是掩码：第一个 token 只允许 <code>&lt;call&gt;</code>，工具名位置只允许指定的那个名字。',
        },
        {
            id: 'prod', title: '教学版 → 生产版', tag: '生产版',
            desc: '本页的每个环节在真实系统里都有对应，主要差别在规模和工程细节：',
            blocks: [
                { type: 'html', html: `<table class="lm-tbl">
                    <tr><th>环节</th><th>本页</th><th>真实系统</th></tr>
                    <tr><td>工具定义</td><td>系统提示里一个字一个工具（「可用:算查」）</td><td>JSON schema：名字、自然语言说明、参数类型与必填项；由对话模板渲染进系统提示（OpenAI function calling 2023、Anthropic tool use 等）；MCP（Model Context Protocol，2024）把「工具服务器」标准化，任何客户端都能接</td></tr>
                    <tr><td>调用格式</td><td><code>&lt;call&gt;算(37+48)&lt;/call&gt;</code></td><td>夹在特殊 token 里的 JSON（工具名 + 参数对象）；一次回复里可以有多个调用（并行调用），每个带一个 id，结果按 id 回填</td></tr>
                    <tr><td>训练</td><td>合成数据上的 SFT，只在模型写的 token 上算损失</td><td>大量人工 / 合成的工具使用轨迹做 SFT，再用强化学习（结果可验证的任务）继续训练；Toolformer（2023）用「调用之后困惑度是否下降」自动筛选训练数据</td></tr>
                    <tr><td>解析</td><td>一个正则</td><td>流式解析 JSON；参数先按 schema 校验，失败就把错误信息作为工具结果返回，让模型自己改</td></tr>
                    <tr><td>约束解码</td><td>手写的小语法</td><td>JSON schema → 有限状态机 / 下推自动机（Outlines、XGrammar 等），也用于「结构化输出」</td></tr>
                    <tr><td>安全</td><td>无</td><td>工具结果可能包含恶意指令（<strong>提示注入</strong>：网页里写着「忽略之前的指令…」）；有副作用的工具（发邮件、删文件）需要权限检查与用户确认——第 18 篇</td></tr>
                </table>` },
            ],
            formula: ['工具调用 = 训练出的格式 + Harness 的执行与写回      // 模型从不直接执行任何东西'],
            previews: [
                {
                    id: 'evalTable', label: '离线评测（训练结束，每格 200 题，贪心解码）', sub: 'tools/agent_train.js 的最后一次评测：「+」= 系统提示里有「算 + 查」，「−」= 没有工具', size: [720, 140],
                    render: (c, S) => {
                        const E = AGT_W.main.evals[AGT_W.main.evals.length - 1][1], C = LM.C, f = (k) => `${(100 * E[k].acc).toFixed(1)}%`;
                        LM.draw.text(c, [
                            [{ t: '加法  ', color: C.dim }, { t: `+ ${f('add+tool')}`, color: C.green }, { t: `   − ${f('add-tool')}`, color: C.red }, { t: '      减法  ', color: C.dim }, { t: `+ ${f('sub+tool')}`, color: C.green }, { t: `   − ${f('sub-tool')}`, color: C.red }, { t: '      乘法  ', color: C.dim }, { t: `+ ${f('mul+tool')}`, color: C.green }, { t: `   − ${f('mul-tool')}`, color: C.red }],
                            [{ t: '常识（有工具也不调用）  ', color: C.dim }, { t: f('fact+tool'), color: C.green }, { t: `   平均调用 ${E['fact+tool'].calls} 次`, color: C.gray }],
                            [{ t: '现在在哪  ', color: C.dim }, { t: `+ ${f('where+tool')}`, color: C.green }, { t: `（平均调用 ${E['where+tool'].calls} 次）   − ${f('where-tool')}（无工具时的「答对」= 说查不到）`, color: C.gray }],
                            [{ t: `模型：${LM.gpt.numParams(S.M).toLocaleString()} 参数（d = ${AGT_W.main.cfg.d}，${AGT_W.main.cfg.L} 层，上下文 ${AGT_W.main.cfg.T}），离线训练 ${AGT_W.main.steps} 步、${Math.round(AGT_W.main.sec / 60)} 分钟（${AGT_W.main.workers} 个线程）`, color: C.dim }],
                        ], { px: 12, lineGap: 9 });
                    },
                },
            ],
            notes: '下一篇（17）把这个循环跑很多轮：模型在一个虚拟的房子里「看、走、拿、放」，靠工具返回的观察一步步完成任务——这就是 Agent。',
            warn: '本页的模型只认识 6 个固定的工具，也不会读工具说明；它说明的是机制，不是能力。真实模型能用陌生工具，靠的是大规模预训练 + 指令微调带来的语言理解，以及工具说明写得足够清楚。工具多到几百上千个、放不进提示时，Harness 还要先替模型「找工具」——见第 19 篇。',
        },
    ],

    graph: {
        title: '工具调用 · 一个回合里的数据流',
        intro: '节点 = 页面里真实执行的一段计算（JS 参考实现；代码块是对应的 C++）。上半部分是推理时 Harness 的循环（<b>HISTORY</b> = 每生成一个 token 走一圈），下半部分是离线训练。',
        common: { title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）', open: false, blocks: LM.CPP.blocks(['gpt']) },
        types: { tok: { color: '#56b6c2', label: 'tok · token 序列' }, logit: { color: '#e0a060', label: 'logit · 下一个 token 的打分' }, call: { color: '#e8c25a', label: 'call · 工具调用 / 结果' }, w: { color: '#c678dd', label: 'w · 模型权重' } },
        nodes: [
            { id: 'prompt', title: 'systemPrompt · serialize', tag: 'CPU', category: 'bake', lifetime: 'param', step: 'template', desc: '&lt;sys&gt;可用:… &lt;user&gt;… &lt;bot&gt;', dispatch: { kind: 'CPU', name: 'serialize · encode', count: '一次' }, outputs: [{ name: 'Prompt', format: '[n] token id', type: 'tok' }], detail: { preview: 'tokChips' } },
            { id: 'session', title: 'feed（KV cache）', tag: 'GEMM', lifetime: 'history', step: 'decide', desc: '每个新 token 前向一次', dispatch: { kind: 'CPU', name: 'gptStep', count: '每个 token 一次（3 层 × d = 64）' }, inputs: [{ name: 'Prompt', format: '第一次：整段提示', type: 'tok' }, { name: 'Context', format: '之后：只有新 token', type: 'tok', history: true }, { name: 'Weights', format: '17 万参数', type: 'w' }], outputs: [{ name: 'Logits', format: '[147]', type: 'logit' }], detail: { preview: 'firstTok' } },
            { id: 'mask', title: 'maskLogits', tag: 'CPU', lifetime: 'frame', step: 'grammar', desc: 'tool_choice · 约束解码', dispatch: { kind: 'CPU', name: 'allowedNext · maskLogits', count: '每个 token' }, inputs: [{ name: 'Logits', format: '[147]', type: 'logit' }], outputs: [{ name: 'Masked', format: '[147]（非法 = −∞）', type: 'logit' }], detail: { preview: 'gramSteps' } },
            { id: 'sample', title: 'sample', tag: 'CPU', lifetime: 'frame', step: 'decide', desc: '贪心 / 温度采样', dispatch: { kind: 'CPU', name: 'argmax · decodeDist', count: '每个 token' }, inputs: [{ name: 'Masked', format: '[147]', type: 'logit' }], outputs: [{ name: 'Token', format: '1 个 id', type: 'tok' }] },
            { id: 'parse', title: 'parseCall（流式）', tag: 'CPU', lifetime: 'frame', step: 'loop', desc: '看到 &lt;/call&gt; 就停下', dispatch: { kind: 'CPU', name: 'parseCall', count: '每次调用' }, inputs: [{ name: 'Token', format: '<call> … </call>', type: 'tok' }], outputs: [{ name: 'Call', format: '工具名 + 参数', type: 'call' }], detail: { preview: 'trace' } },
            { id: 'exec', title: 'ToolBox.run', tag: 'CPU', category: 'output', lifetime: 'frame', step: 'loop', desc: '真实执行：计算器 / 查询', dispatch: { kind: 'CPU', name: 'calc · lookup', count: '每次调用' }, inputs: [{ name: 'Call', format: '工具名 + 参数', type: 'call' }], outputs: [{ name: 'Result', format: '<res>…</res>', type: 'call' }] },
            { id: 'append', title: 'append <res>', tag: 'CPU', lifetime: 'history', step: 'loop', desc: '结果写回上下文 → 继续生成', dispatch: { kind: 'CPU', name: 'encode + push', count: '每次调用' }, inputs: [{ name: 'Result', format: '文本', type: 'call' }, { name: 'Token', format: '模型写的 token', type: 'tok' }], outputs: [{ name: 'Context', format: '[n + k]', type: 'tok', history: true }] },
            { id: 'data', title: 'calcExample · serialize', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'train', desc: '专家示范对话（离线）', dispatch: { kind: 'CPU', name: 'qaExpert · houseExpert', count: '每步 48 条' }, outputs: [{ name: 'Samples', format: '[B × T] + role', type: 'tok' }], detail: { preview: 'maskChips' } },
            { id: 'sft', title: 'maskedTargets · sftStep', tag: 'GEMM', category: 'output', lifetime: 'history', step: 'train', desc: '只在模型写的 token 上算交叉熵', dispatch: { kind: 'CPU', name: 'gptForward · crossEntropy · AdamW', count: '8000 步（离线）' }, inputs: [{ name: 'Samples', format: '[B × T]', type: 'tok' }], outputs: [{ name: 'Weights', format: '17 万参数（EMA）', type: 'w' }], detail: { preview: 'trainCurve' } },
        ],
    },
    references: [
        { title: 'Toolformer: Language Models Can Teach Themselves to Use Tools', author: 'T. Schick et al.', venue: 'NeurIPS', year: 2023, note: '用「调用之后预测是否变好」自动生成工具使用的训练数据' },
        { title: 'TALM: Tool Augmented Language Models', author: 'A. Parisi, Y. Zhao, N. Fiedel', venue: 'arXiv', year: 2022, note: '文本形式的工具调用 + 自举训练' },
        { title: 'ReAct: Synergizing Reasoning and Acting in Language Models', author: 'S. Yao et al.', venue: 'ICLR', year: 2023, note: '「思考 → 行动 → 观察」循环（第 17 篇）' },
        { title: 'Gorilla: Large Language Model Connected with Massive APIs', author: 'S. G. Patil, T. Zhang, X. Wang, J. E. Gonzalez', venue: 'arXiv', year: 2023, note: '大量真实 API 的调用；后来的 Berkeley Function-Calling Leaderboard' },
        { title: 'Function calling and other API updates', author: 'OpenAI', venue: 'Blog', year: 2023, note: 'JSON schema 工具定义与 function calling API' },
        { title: 'Introducing the Model Context Protocol', author: 'Anthropic', venue: 'Blog', year: 2024, url: 'https://modelcontextprotocol.io', note: 'MCP：工具 / 数据源服务器的开放协议' },
        { title: 'Efficient Guided Generation for Large Language Models', author: 'B. T. Willard, R. Louf', venue: 'arXiv', year: 2023, note: 'Outlines：把正则 / JSON schema 编译成有限状态机做约束解码' },
        { title: 'XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models', author: 'Y. Dong et al.', venue: 'arXiv', year: 2024, note: '下推自动机 + 预计算的高效约束解码' },
    ],
});
