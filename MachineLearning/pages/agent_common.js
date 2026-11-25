// agent_common.js — 第 16–18 篇（工具调用、Agent、Harness）共用的引擎：词表与对话模板、工具与环境、任务与专家轨迹、
// 训练数据（loss mask）、带 KV cache 的增量解码会话、约束解码、Harness 主循环。浏览器与 Node（tools/agent_train.js）共用同一份代码。
// 构建：node <skill>/scripts/build_doc.mjs llm_common.js agent_common.js agent_weights.js 16_tool_calling.lab.js -o 16_tool_calling.html
const AGT = (() => {
    'use strict';
    // ═════════════════════════════ 1. 词表：特殊 token + 字符 ═════════════════════════════
    // 特殊 token 各占一个 id；其余按字符切（中文一个字一个 token）
    const SPECIALS = ['<pad>', '<eos>', '<sys>', '<user>', '<bot>', '<call>', '</call>', '<res>', '</res>'];
    const SP = { pad: 0, eos: 1, sys: 2, user: 3, bot: 4, call: 5, ecall: 6, res: 7, eres: 8 };
    const ANIMALS = [
        { name: '小猫', food: '鱼', home: '屋里', color: '白色' }, { name: '小狗', food: '骨头', home: '院子', color: '黄色' },
        { name: '小兔', food: '萝卜', home: '草地', color: '灰色' }, { name: '小鸟', food: '虫子', home: '树上', color: '蓝色' },
        { name: '熊猫', food: '竹子', home: '山里', color: '黑白' }, { name: '猴子', food: '香蕉', home: '森林', color: '棕色' },
        { name: '小羊', food: '青草', home: '山坡', color: '白色' }, { name: '老虎', food: '肉', home: '森林', color: '橙色' },
    ];
    const PLACES = ['屋里', '院子', '草地', '树上', '山里', '森林', '山坡', '河边'];
    const ROOMS = ['客厅', '厨房', '卧室', '书房', '院子'];
    // 房子的布局（固定）：客厅连着厨房、卧室、书房；书房连着院子
    const ADJ = [[1, 2, 3], [0], [0], [0, 4], [3]];
    // 另一种布局（第 17 篇：模型没见过）：客厅连着厨房、卧室；厨房连着院子；卧室连着书房
    const ADJ2 = [[1, 2], [0, 4], [0, 3], [2], [1]];
    const ITEMS = ['钥匙', '苹果', '书', '球', '杯子'];
    const TOOLS = { calc: '算', lookup: '查', look: '看', go: '去', take: '拿', put: '放' };
    const CHARSET = '0123456789+-×=():,;。？，、' + '可用无' + '算查看去拿放' + ROOMS.join('') + ITEMS.join('') +
        ANIMALS.map((a) => a.name + a.food + a.home + a.color).join('') + PLACES.join('') +
        '喜欢吃什么住在哪里是颜色的现在等于多少算一下加减乘我查不到把拿到了放好已经里有和也没这空去不手超时记见过在个工具格式错误权限';
    const itos = SPECIALS.slice(), stoi = new Map();
    for (const ch of CHARSET) if (!itos.includes(ch)) itos.push(ch);
    itos.forEach((s, i) => stoi.set(s, i));
    const V = itos.length;
    /** 字符串 → token id（特殊 token 写成 <call> 这样的整体） */
    function encode(s) {
        const out = [];
        for (let i = 0; i < s.length;) {
            if (s[i] === '<') { const j = s.indexOf('>', i); const t = s.slice(i, j + 1); if (stoi.has(t)) { out.push(stoi.get(t)); i = j + 1; continue; } }
            const id = stoi.get(s[i]); if (id == null) throw new Error(`字符不在词表里：${s[i]}（${s}）`);
            out.push(id); i++;
        }
        return out;
    }
    const decode = (ids) => ids.map((i) => itos[i]).join('');
    const isSpecial = (id) => id < SPECIALS.length;

    // ═════════════════════════════ 2. 工具与环境 ═════════════════════════════
    /** 计算器：只认「两个整数 + 一个运算符」，否则返回「错误」 */
    function calc(expr) {
        const m = /^(\d{1,3})([+\-×])(\d{1,3})$/.exec(expr); if (!m) return '错误';
        const a = +m[1], b = +m[3];
        return String(m[2] === '+' ? a + b : m[2] === '-' ? a - b : a * b);
    }
    /** 一局里的「动态世界」：每只动物此刻在哪里（每局随机，模型无法背下来） */
    function newWorld(rand) { const where = {}; ANIMALS.forEach((a) => { where[a.name] = PLACES[Math.floor(rand() * PLACES.length)]; }); return { where }; }
    function lookup(world, arg) { return world.where[arg] || '没有'; }
    /** 房子：物品随机放进房间，智能体从某个房间出发 */
    function newHouse(rand, o) {
        o = o || {};
        const at = {}; ITEMS.forEach((it) => { at[it] = Math.floor(rand() * ROOMS.length); });
        return { at, room: o.start ?? Math.floor(rand() * ROOMS.length), hand: null, steps: 0, log: [], adj: o.adj || null };
    }
    const itemsIn = (H, r) => ITEMS.filter((it) => H.at[it] === r && H.hand !== it);
    const obs = (H, r) => `${ROOMS[r]}:${itemsIn(H, r).join(',') || '空'}`;
    /** 执行一次房子里的工具调用：返回 { res, ok }（不改变状态的失败 ok = false） */
    function houseStep(H, tool, arg) {
        H.steps++;
        if (tool === TOOLS.look) return { res: obs(H, H.room), ok: true };
        if (tool === TOOLS.go) { const r = ROOMS.indexOf(arg); if (r < 0 || !(H.adj || ADJ)[H.room].includes(r)) return { res: '去不了', ok: false }; H.room = r; return { res: obs(H, r), ok: true }; }
        if (tool === TOOLS.take) { if (!ITEMS.includes(arg) || H.at[arg] !== H.room || H.hand === arg) return { res: '这里没有', ok: false }; if (H.hand) return { res: '手里有了', ok: false }; H.hand = arg; return { res: '拿到了', ok: true }; }
        if (tool === TOOLS.put) { if (H.hand !== arg) return { res: '手里没有', ok: false }; H.at[arg] = H.room; H.hand = null; return { res: '放好了', ok: true }; }
        return { res: '没有这个工具', ok: false };
    }
    /** 最短路径（房间序号列表，不含起点） */
    function path(a, b, adj) {
        adj = adj || ADJ;
        const prev = new Array(ROOMS.length).fill(-1), q = [a], seen = new Set([a]);
        while (q.length) { const u = q.shift(); if (u === b) break; for (const v of adj[u]) if (!seen.has(v)) { seen.add(v); prev[v] = u; q.push(v); } }
        const out = []; for (let v = b; v !== a; v = prev[v]) out.unshift(v);
        return out;
    }

    // ═════════════════════════════ 3. 任务与「专家」轨迹 ═════════════════════════════
    // 一条对话 = [{ role: 'sys' | 'user' | 'bot' | 'tool', text }]；bot 的内容就是模型要学会输出的部分
    const sysText = (tools) => `可用:${tools.length ? tools.join('') : '无'}`;
    const pick = (rand, arr) => arr[Math.floor(rand() * arr.length)];
    /** 第 16 篇的单轮任务：kind ∈ add | sub | mul | fact | where */
    function qaTask(rand, kind) {
        if (kind === 'add' || kind === 'sub') { let a = 10 + Math.floor(rand() * 90), b = 10 + Math.floor(rand() * 90); if (kind === 'sub' && b > a) [a, b] = [b, a]; const op = kind === 'add' ? '+' : '-'; return { kind, q: `${a}${op}${b}等于多少？`, expr: `${a}${op}${b}`, ans: calc(`${a}${op}${b}`) }; }
        if (kind === 'mul') { const a = 10 + Math.floor(rand() * 90), b = 2 + Math.floor(rand() * 8); return { kind, q: `${a}×${b}等于多少？`, expr: `${a}×${b}`, ans: calc(`${a}×${b}`) }; }
        const an = pick(rand, ANIMALS);
        if (kind === 'fact') { const k = Math.floor(rand() * 3), q = [`${an.name}喜欢吃什么？`, `${an.name}住在哪里？`, `${an.name}是什么颜色的？`][k], ans = [`${an.name}喜欢吃${an.food}。`, `${an.name}住在${an.home}。`, `${an.name}是${an.color}的。`][k]; return { kind, q, ans }; }
        return { kind: 'where', q: `${an.name}现在在哪？`, animal: an.name };
    }
    /** 单轮任务的专家回答（tools = 本轮可用的工具名） */
    function qaExpert(task, tools, world) {
        const can = (t) => tools.includes(t), turns = [];
        if (task.kind === 'add' || task.kind === 'sub' || task.kind === 'mul') {
            if (can(TOOLS.calc)) turns.push({ role: 'bot', text: `<call>${TOOLS.calc}(${task.expr})</call>` }, { role: 'tool', text: `<res>${task.ans}</res>` });
            turns.push({ role: 'bot', text: `${task.expr}等于${task.ans}。<eos>` });
        } else if (task.kind === 'fact') turns.push({ role: 'bot', text: `${task.ans}<eos>` });
        else if (can(TOOLS.lookup)) { const r = lookup(world, task.animal); turns.push({ role: 'bot', text: `<call>${TOOLS.lookup}(${task.animal})</call>` }, { role: 'tool', text: `<res>${r}</res>` }, { role: 'bot', text: `${task.animal}现在在${r}。<eos>` }); }
        else turns.push({ role: 'bot', text: `我查不到${task.animal}现在在哪。<eos>` });
        return turns;
    }
    /** 房子任务：find（X 在哪）、move（把 X 拿到 R）、list（R 里有什么） */
    function houseTask(rand, kind) {
        kind = kind || pick(rand, ['find', 'move', 'list']);
        const item = pick(rand, ITEMS), room = Math.floor(rand() * ROOMS.length);
        if (kind === 'find') return { kind, item, q: `${item}在哪？` };
        if (kind === 'move') return { kind, item, room, q: `把${item}拿到${ROOMS[room]}` };
        return { kind, room, q: `${ROOMS[room]}里有什么？` };
    }
    /** 任务是否完成（Harness 用环境状态判定，不看模型说了什么——除了 find / list 要核对答案） */
    function houseSuccess(task, H, answer) {
        if (task.kind === 'move') return H.at[task.item] === task.room && H.hand !== task.item;
        if (task.kind === 'find') return answer === `${task.item}在${ROOMS[H.at[task.item]]}。`;
        const its = itemsIn(H, task.room);
        return answer === (its.length ? `${ROOMS[task.room]}里有${its.join('、')}。` : `${ROOMS[task.room]}里什么也没有。`);
    }
    /**
     * 专家策略（生成训练轨迹）：先「看」，然后按固定顺序深度优先探索（去 = 走过去并看到那个房间）；找到目标就停。
     * o.fail = 某一步工具调用「超时」的概率（专家会原样重试）；o.memo = 第几次调用之后让 Harness 压缩历史（-1 = 不压缩）
     */
    function houseExpert(task, H0, rand, o) {
        o = o || {};
        const H = JSON.parse(JSON.stringify(H0)), turns = [], seen = new Map();
        let calls = 0;
        const act = (tool, arg) => {
            const call = `<call>${tool}(${arg || ''})</call>`;
            if (o.fail && rand() < o.fail) { turns.push({ role: 'bot', text: call }, { role: 'tool', text: '<res>超时</res>' }); }   // 超时：什么都没发生，专家重试一次
            const r = houseStep(H, tool, arg);
            turns.push({ role: 'bot', text: call }, { role: 'tool', text: `<res>${r.res}</res>` });
            if (r.ok && (tool === TOOLS.look || tool === TOOLS.go)) seen.set(H.room, itemsIn(H, H.room));
            calls++;
            if (o.memo === calls) turns.push({ role: 'memo', text: memoText(H, seen) });
            return r;
        };
        const known = (it) => { for (const [r, its] of seen) if (its.includes(it)) return r; return -1; };
        const A = H.adj || ADJ, goTo = (r) => { for (const v of path(H.room, r, A)) act(TOOLS.go, ROOMS[v]); };
        // 深度优先探索，直到 stop() 为真
        const explore = (stop) => {
            const visited = new Set([H.room]);
            const dfs = (u) => { if (stop()) return true; for (const v of A[u]) { if (visited.has(v)) continue; visited.add(v); act(TOOLS.go, ROOMS[v]); if (dfs(v)) return true; act(TOOLS.go, ROOMS[u]); } return stop(); };
            dfs(H.room);
        };
        act(TOOLS.look);
        if (task.kind === 'find') {
            explore(() => known(task.item) >= 0);
            turns.push({ role: 'bot', text: `${task.item}在${ROOMS[known(task.item)]}。<eos>` });
        } else if (task.kind === 'list') {
            if (H.room !== task.room) goTo(task.room);
            const its = itemsIn(H, task.room);
            turns.push({ role: 'bot', text: (its.length ? `${ROOMS[task.room]}里有${its.join('、')}。` : `${ROOMS[task.room]}里什么也没有。`) + '<eos>' });
        } else {
            explore(() => known(task.item) >= 0);
            const r = known(task.item);
            if (r === task.room) turns.push({ role: 'bot', text: `${task.item}已经在${ROOMS[task.room]}了。<eos>` });
            else { goTo(r); act(TOOLS.take, task.item); goTo(task.room); act(TOOLS.put, task.item); turns.push({ role: 'bot', text: `${task.item}拿到${ROOMS[task.room]}了。<eos>` }); }
        }
        return { turns, H };
    }
    /** Harness 写的「记忆」：当前位置、看到过的房间里的东西、手里拿着什么（压缩历史时用来替换旧的工具往返） */
    function memoText(H, seen) {
        const parts = [`在${ROOMS[H.room]}`];
        for (const [r, its] of seen) parts.push(`${ROOMS[r]}:${its.join(',') || '空'}`);
        if (H.hand) parts.push(`手:${H.hand}`);
        return `<res>记:${parts.join(';')}</res>`;
    }

    // ═════════════════════════════ 4. 对话 → token + loss mask ═════════════════════════════
    /**
     * 序列化：<sys>可用:…<user>问题<bot>（bot 与 tool 交替）……<eos>。返回 { ids, role }：role[i] = 第 i 个 token 是谁写的
     * memo（历史压缩）：把它之前的工具往返全部删掉，换成一条记忆（Harness 的上下文压缩，第 18 篇）
     */
    function serialize(sys, user, turns) {
        let segs = [{ role: 'sys', text: `<sys>${sys}` }, { role: 'user', text: `<user>${user}` }, { role: 'tool', text: '<bot>' }];
        for (const t of turns) {
            if (t.role === 'memo') { segs = segs.slice(0, 3); segs.push({ role: 'tool', text: t.text }); continue; }
            segs.push(t);
        }
        const ids = [], role = [];
        for (const s of segs) { const e = encode(s.text); for (const id of e) { ids.push(id); role.push(s.role); } }
        return { ids, role, segs };
    }
    /** 一个训练样本：x = ids[0..n−1)，y = ids[1..n)；只有「下一个 token 是 bot 写的」位置计入损失（其余 = −1） */
    function sample(rand, mix) {
        mix = mix || {};
        if (rand() < (mix.qa ?? 0.3)) {
            const kind = pick(rand, ['add', 'add', 'sub', 'mul', 'fact', 'where', 'where']), task = qaTask(rand, kind);
            const r = rand(), tools = r < 0.15 ? [] : r < 0.3 ? [TOOLS.calc] : r < 0.45 ? [TOOLS.lookup] : [TOOLS.calc, TOOLS.lookup];
            return Object.assign(serialize(sysText(tools), task.q, qaExpert(task, tools, newWorld(rand))), { kind: 'qa' });
        }
        const task = houseTask(rand), H = newHouse(rand), fail = rand() < 0.25 ? 0.12 : 0;
        const memo = rand() < 0.3 ? 2 + Math.floor(rand() * 4) : -1;
        const ex = houseExpert(task, H, rand, { fail, memo });
        return Object.assign(serialize(sysText([TOOLS.look, TOOLS.go, TOOLS.take, TOOLS.put]), task.q, ex.turns), { kind: 'house' });
    }
    /** 把若干样本拼成 [B × T] 的 batch（右侧补 <pad>，目标 −1 = 不计损失）；超长的样本截断 */
    function batch(samples, T) {
        const B = samples.length, x = new Int32Array(B * T), y = new Int32Array(B * T).fill(-1);
        samples.forEach((s, b) => {
            const n = Math.min(s.ids.length, T + 1);
            for (let t = 0; t < T; t++) x[b * T + t] = t < n ? s.ids[t] : SP.pad;
            for (let t = 0; t + 1 < n; t++) if (s.role[t + 1] === 'bot') y[b * T + t] = s.ids[t + 1];
        });
        return { x, y };
    }

    // ═════════════════════════════ 5. 带 KV cache 的增量解码会话 ═════════════════════════════
    /** 会话：feed(id) 把一个 token 推进模型（算出它的 K、V 存起来），返回下一个 token 的 logits；stats 记录做了多少次「一个 token 的前向」 */
    function newSession(m) {
        const c = m.cfg, d = c.d, H = c.H, dh = d / H, Tm = c.T;
        const S = { m, ids: [], kc: m.layers.map(() => new Float32Array(Tm * d)), vc: m.layers.map(() => new Float32Array(Tm * d)), logits: new Float32Array(c.V), forwards: 0, ms: 0 };
        const x = new Float32Array(d), h = new Float32Array(d), q = new Float32Array(d), k = new Float32Array(d), v = new Float32Array(d), a = new Float32Array(d), f = new Float32Array(c.ff), tmp = new Float32Array(Math.max(d, c.ff)), att = new Float32Array(Tm);
        const GK = Math.sqrt(2 / Math.PI);
        const ln = (src, g, b, out) => { let mu = 0; for (let j = 0; j < d; j++) mu += src[j]; mu /= d; let va = 0; for (let j = 0; j < d; j++) va += (src[j] - mu) ** 2; va /= d; const r = 1 / Math.sqrt(va + 1e-5); for (let j = 0; j < d; j++) out[j] = (src[j] - mu) * r * g[j] + b[j]; };
        const lin = (src, W, b, n, mo, out) => { for (let j = 0; j < mo; j++) out[j] = b ? b[j] : 0; for (let p = 0; p < n; p++) { const xv = src[p]; if (xv === 0) continue; const o2 = p * mo; for (let j = 0; j < mo; j++) out[j] += xv * W[o2 + j]; } };
        S.feed = function (tok) {
            const pos = S.ids.length; if (pos >= Tm) throw new Error('上下文满了');
            const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
            const E = m.wte.data, P = m.wpe.data;
            for (let j = 0; j < d; j++) x[j] = E[tok * d + j] + P[pos * d + j];
            m.layers.forEach((p, l) => {
                ln(x, p.ln1g.data, p.ln1b.data, h);
                lin(h, p.wq.data, p.bq.data, d, d, q); lin(h, p.wk.data, p.bk.data, d, d, k); lin(h, p.wv.data, p.bv.data, d, d, v);
                S.kc[l].set(k, pos * d); S.vc[l].set(v, pos * d);
                a.fill(0);
                for (let hh = 0; hh < H; hh++) {
                    const hc = hh * dh; let mx = -Infinity;
                    for (let j = 0; j <= pos; j++) { let s = 0; for (let e = 0; e < dh; e++) s += q[hc + e] * S.kc[l][j * d + hc + e]; s /= Math.sqrt(dh); att[j] = s; if (s > mx) mx = s; }
                    let z = 0; for (let j = 0; j <= pos; j++) { att[j] = Math.exp(att[j] - mx); z += att[j]; }
                    for (let j = 0; j <= pos; j++) { const w = att[j] / z; for (let e = 0; e < dh; e++) a[hc + e] += w * S.vc[l][j * d + hc + e]; }
                }
                lin(a, p.wo.data, p.bo.data, d, d, tmp); for (let j = 0; j < d; j++) x[j] += tmp[j];
                ln(x, p.ln2g.data, p.ln2b.data, h);
                lin(h, p.w1.data, p.b1.data, d, c.ff, f);
                for (let j = 0; j < c.ff; j++) { const u = f[j]; f[j] = 0.5 * u * (1 + Math.tanh(GK * (u + 0.044715 * u * u * u))); }
                lin(f, p.w2.data, p.b2.data, c.ff, d, tmp); for (let j = 0; j < d; j++) x[j] += tmp[j];
            });
            ln(x, m.lnfg.data, m.lnfb.data, h);
            for (let vv = 0; vv < c.V; vv++) { let s = 0; for (let j = 0; j < d; j++) s += h[j] * E[vv * d + j]; S.logits[vv] = s; }
            S.ids.push(tok); S.forwards++; S.ms += (typeof performance !== 'undefined' ? performance : Date).now() - t0;
            return S.logits;
        };
        S.feedAll = (ids) => { let lg = null; for (const id of ids) lg = S.feed(id); return lg; };
        /** 回到前 n 个 token（KV cache 里多出来的部分作废即可，不用重算） */
        S.truncate = (n) => { S.ids.length = n; };
        return S;
    }

    // ═════════════════════════════ 6. 约束解码：工具调用的语法 ═════════════════════════════
    /**
     * 工具调用的文法（<call> 之后）：工具名 ( 参数 ) </call>。参数的形状由工具决定：
     *   算：数字{1,3} 运算符 数字{1,3}；查：动物名；去：房间名；拿 / 放：物品名；看：空
     * allowed(prefix) 返回「在已经生成的 prefix（<call> 之后的文字）后面，下一个 token 可以是哪些」（id 集合）；null = 不约束
     */
    const ARGS = { 算: null, 查: ANIMALS.map((a) => a.name), 看: [''], 去: ROOMS, 拿: ITEMS, 放: ITEMS };
    function allowedNext(prefix, tools) {
        const ids = new Set(), add = (ch) => ids.add(stoi.get(ch));
        if (prefix.length === 0) { tools.forEach(add); return ids; }
        const tool = prefix[0];
        if (prefix.length === 1) { add('('); return ids; }
        const inner = prefix.slice(2);
        if (inner.endsWith(')')) { ids.add(SP.ecall); return ids; }
        if (tool === '算') {
            const m = /^(\d*)([+\-×]?)(\d*)$/.exec(inner); if (!m) return ids;
            if (!m[2]) { if (m[1].length < 3) for (const ch of '0123456789') add(ch); if (m[1].length) for (const ch of '+-×') add(ch); }
            else { if (m[3].length < 3) for (const ch of '0123456789') add(ch); if (m[3].length) add(')'); }
            return ids;
        }
        const opts = ARGS[tool] || [];
        for (const w of opts) { if (w.startsWith(inner) && w.length > inner.length) add(w[inner.length]); if (w === inner) add(')'); }
        return ids;
    }

    // ═════════════════════════════ 7. Harness：跑一个回合（可以分帧推进） ═════════════════════════════
    /**
     * 让模型在一个任务上跑完：生成 → 遇到 </call> 就解析并执行工具 → 把 <res>结果</res> 喂回去 → 继续，直到 <eos> 或预算用完。
     * env: { tools: [工具名…], exec(tool, arg) → { res, ok } }
     * o: { greedy, temp, rand, constrain（约束解码）, toolChoice: 'auto' | 'none' | 'required', maxCalls, maxTokens,
     *      window（上下文 token 预算）, policy: 'none' | 'truncate' | 'keep-task' | 'compact', memo()（compact 时由 Harness 生成记忆）,
     *      permit(tool, arg) → true | '理由'（权限检查）, retry（工具超时时 Harness 自动重试的次数）, cache: false（不用 KV cache）,
     *      noExec: true（不执行工具：让模型自己往下写 <res>…</res>，第 16 篇用来说明「为什么结果必须由 Harness 写」）, keepTop: k（记下每个生成 token 的前 k 个候选） }
     * newEpisode 返回一个回合对象 E：E.advance(n) 最多再做 n 次「一个 token 的前向」（分帧推进，动画与实时评测用），完成后 E.done = true、E.result 是结果。
     */
    function newEpisode(m, sys, user, env, o) {
        o = o || {};
        const rand = o.rand || Math.random, T = m.cfg.T, budget = Math.min(T, o.window || T), maxCalls = o.maxCalls ?? 16, maxTokens = o.maxTokens ?? 400, cache = o.cache !== false;
        const S = newSession(m), ev = [], tops = [];
        let ctx = [...encode(`<sys>${sys}<user>${user}<bot>`)];
        const prefixLen = ctx.length;
        let fed = 0, generated = 0, calls = 0, answer = '', reason = '', lg = null, inCall = false, callText = '', inFake = false, fakeText = '', wantFit = 0;
        const E = { S, ev, tops, fwd: [], full: [], done: false, result: null, ctx: () => ctx, phase: 'gen', calls: () => calls, prefixLen };
        let lastCall = '', repeat = 0;
        const rebuild = (ids) => {   // 上下文被 Harness 改写：与旧上下文相同的前缀直接复用 KV cache，其余重新喂
            let k = 0; if (cache) while (k < S.ids.length && k < ids.length && S.ids[k] === ids[k]) k++;
            S.truncate(k); ctx = ids; fed = k;
        };
        const fit = (extra) => {     // 放不下时按策略处理（extra = 接下来至少还要写的 token 数）
            if (ctx.length + extra <= budget) return true;
            const pol = o.policy || 'none';
            if (pol === 'none') return false;
            if (pol === 'truncate') { rebuild(ctx.slice(Math.max(0, ctx.length + extra - Math.floor(budget / 2)))); ev.push({ type: 'ctx', policy: pol, len: ctx.length }); return true; }   // 一次截掉最旧的一半窗口（逐 token 滑动的话每一步都要整段重算）
            // cur = 正在写的这一段（最后一个 </res> 之后：半句回答或半个调用）——改写上下文时原样保留
            const cut = Math.max(prefixLen, ctx.lastIndexOf(SP.eres) + 1), head = ctx.slice(0, prefixLen), cur = ctx.slice(cut);
            if (pol === 'keep-task') {   // 保留 sys + user + <bot>，从最旧的工具往返开始删
                let mid = ctx.slice(prefixLen, cut);
                while (head.length + mid.length + cur.length + extra > budget && mid.length) { const j = mid.indexOf(SP.eres); mid = j >= 0 ? mid.slice(j + 1) : []; }
                rebuild(head.concat(mid, cur)); ev.push({ type: 'ctx', policy: pol, len: ctx.length }); return ctx.length + extra <= budget;
            }
            if (pol === 'compact' && o.memo) { const memo = encode(o.memo()); rebuild(head.concat(memo, cur)); ev.push({ type: 'ctx', policy: pol, len: ctx.length, memo: decode(memo) }); return ctx.length + extra <= budget; }
            return false;
        };
        const finish = (why) => {
            reason = why; E.done = true;
            E.result = { ok: reason === 'eos', reason, answer, text: decode(ctx), ids: ctx.slice(), events: ev, calls, tokens: ctx.length, generated, forwards: S.forwards, ms: S.ms, tops, fwd: E.fwd, fwdFull: E.full };
        };
        const appendRes = (text) => { const r = encode(`<res>${text}</res>`); if (!fit(r.length + 2)) { finish('context'); return; } for (const t of r) ctx.push(t); };
        const handleCall = () => {
            calls++;
            const mm = /^(.)\((.*)\)$/.exec(callText);
            if (!mm) { ev.push({ type: 'parse-error', text: callText }); appendRes('格式错误'); return; }
            const [, tool, arg] = mm;
            if (o.loopStop) { const key = `${tool}(${arg})`; repeat = key === lastCall ? repeat + 1 : 1; lastCall = key; if (repeat >= 3) { ev.push({ type: 'loop', tool, arg }); finish('loop'); return; } }
            if (o.noExec) { ev.push({ type: 'call', tool, arg, res: null, ok: null, noExec: true }); ctx.push(SP.res); inFake = true; fakeText = ''; return; }
            if (!env.tools.includes(tool)) { ev.push({ type: 'call', tool, arg, res: '没有这个工具', ok: false }); appendRes('没有这个工具'); return; }
            const perm = o.permit ? o.permit(tool, arg) : true;
            if (perm !== true) { ev.push({ type: 'denied', tool, arg, why: perm }); appendRes('没有权限'); return; }
            let r = env.exec(tool, arg), tries = 1;
            while (r.res === '超时' && tries <= (o.retry || 0)) { ev.push({ type: 'retry', tool, arg }); r = env.exec(tool, arg); tries++; }
            ev.push({ type: 'call', tool, arg, res: r.res, ok: r.ok, tries });
            appendRes(r.res);
        };
        /** 生成一个 token（必要时先把上下文里还没喂给模型的 token 喂进去） */
        const stepOne = () => {
            if (generated >= maxTokens) return finish('tokens');
            if (!fit(inCall ? 2 : 4)) return finish('context');
            if (cache) { while (fed < ctx.length) { lg = S.feed(ctx[fed]); fed++; } }
            else { S.truncate(0); lg = S.feedAll(ctx); fed = ctx.length; }   // 不用 KV cache：每生成一个 token 都把整个上下文重算一遍
            const logits = Float32Array.from(lg);
            // tool_choice 与约束解码都是对 logits 的掩码
            const last = ctx[ctx.length - 1], atStart = last === SP.bot || last === SP.eres;
            if (o.toolChoice === 'none' || calls >= maxCalls) logits[SP.call] = -Infinity;
            if (o.toolChoice === 'required' && atStart && calls === 0) for (let i = 0; i < logits.length; i++) if (i !== SP.call) logits[i] = -Infinity;
            if (inCall && o.constrain) { const al = allowedNext(callText, env.tools); if (al.size) for (let i = 0; i < logits.length; i++) if (!al.has(i)) logits[i] = -Infinity; }
            logits[SP.pad] = -Infinity;
            let id, p = null;
            if (o.greedy || !o.temp) id = LM.argmax(logits);
            else { p = LM.softmax(logits, new Float32Array(logits.length), o.temp); id = LM.sampleFrom(p, rand); }
            if (o.keepTop) { const q = p || LM.softmax(logits, new Float32Array(logits.length), 1), top = LM.topk(q, o.keepTop); tops.push({ pos: ctx.length, id, top: top.map((i) => [i, q[i]]), masked: logits.reduce((s, v) => s + (v === -Infinity ? 1 : 0), 0) }); }
            const L = ctx.length;
            ctx.push(id); generated++;
            if (o.trackFwd) { E.fwd.push(S.forwards); E.full.push((E.full.length ? E.full[E.full.length - 1] : 0) + L); }   // full：不用 KV cache 时，生成这个 token 要把当时的 L 个 token 全部重算
            if (inFake) {   // noExec：模型自己写的「工具结果」，直到它写出 </res>（写出 <eos> 等特殊 token 也算结束）
                if (id === SP.eres) { inFake = false; ev[ev.length - 1].res = fakeText; return; }
                if (isSpecial(id)) { ev[ev.length - 1].res = fakeText; return finish(id === SP.eos ? 'eos' : 'bad-token'); }
                fakeText += itos[id]; return;
            }
            if (id === SP.call) { inCall = true; callText = ''; return; }
            if (inCall && id !== SP.ecall) { callText += itos[id]; if (callText.length > 12) { inCall = false; ev.push({ type: 'parse-error', text: callText }); } return; }
            if (id === SP.ecall && inCall) { inCall = false; handleCall(); return; }
            if (id === SP.eos) return finish('eos');
            if (isSpecial(id)) { ev.push({ type: 'bad-token', tok: itos[id] }); return finish('bad-token'); }   // 只有 Harness 才能写的 token，或者没有 <call> 就写出的 </call>
            answer += itos[id];
        };
        /** 再推进最多 n 次前向（n = Infinity：跑完为止）；返回 E.done */
        E.advance = (n) => {
            const f0 = S.forwards;
            while (!E.done && S.forwards - f0 < (n ?? Infinity)) stepOne();
            return E.done;
        };
        E.status = () => ({ generated, calls, tokens: ctx.length, forwards: S.forwards, inCall, answer, text: decode(ctx) });
        return E;
    }
    function runEpisode(m, sys, user, env, o) { const E = newEpisode(m, sys, user, env, o); E.advance(Infinity); return E.result; }
    /** 房子环境的 exec（带可选的故障注入：每次调用以概率 failP 返回「超时」） */
    function houseEnv(H, o) {
        o = o || {}; const rand = o.rand || Math.random;
        return { tools: [TOOLS.look, TOOLS.go, TOOLS.take, TOOLS.put], exec: (tool, arg) => { if (o.failP && rand() < o.failP) { H.steps++; const t = { res: '超时', ok: false }; if (o.onStep) o.onStep(H, tool, arg, t); return t; } const r = houseStep(H, tool, arg); if (o.onStep) o.onStep(H, tool, arg, r); return r; } };
    }
    /** 单轮问答的 exec */
    function qaEnv(tools, world) { return { tools, exec: (tool, arg) => (tool === TOOLS.calc ? { res: calc(arg), ok: true } : tool === TOOLS.lookup ? { res: lookup(world, arg), ok: true } : { res: '没有这个工具', ok: false }) }; }
    /** 房子任务的一个回合（可分帧推进）：Harness 自己记下看到过什么（compact 策略用它写记忆）；完成后 E.result.success = 是否完成任务 */
    function newHouseEpisode(m, task, H, o) {
        o = o || {};
        const seen = new Map(), track = (h, tool, arg, r) => { if (r.ok && (tool === TOOLS.look || tool === TOOLS.go)) seen.set(h.room, itemsIn(h, h.room)); if (o.onStep) o.onStep(h, tool, arg, r); };
        const env = houseEnv(H, Object.assign({}, o, { onStep: track }));
        const E = newEpisode(m, sysText(env.tools), task.q, env, Object.assign({}, o, { memo: () => memoText(H, seen) }));
        const adv = E.advance;
        E.advance = (n) => { const d = adv(n); if (d && E.result && E.result.success == null) { E.result.success = E.result.ok && houseSuccess(task, H, E.result.answer); E.result.H = H; } return d; };
        E.house = H; E.task = task;
        return E;
    }
    function runHouse(m, task, H, o) { const E = newHouseEpisode(m, task, H, o); E.advance(Infinity); return E.result; }
    // ═════════════════════════════ 8. 页面用的小工具：token 的「作者」、token 条、分帧评测 ═════════════════════════════
    /** 每个 token 是谁写的：sys / user / harness（<bot>、<res>…</res>）/ bot（模型），以及是否在 <call>…</call> 里 */
    function roles(ids) {
        const out = []; let state = 'sys', inCall = false, inRes = false;
        for (const id of ids) {
            if (id === SP.sys) state = 'sys'; else if (id === SP.user) state = 'user'; else if (id === SP.bot) { out.push({ role: 'harness', call: false }); state = 'bot'; continue; }
            if (id === SP.res) inRes = true;
            if (id === SP.call) inCall = true;
            out.push({ role: inRes ? 'harness' : state, call: inCall });
            if (id === SP.eres) inRes = false;
            if (id === SP.ecall) inCall = false;
        }
        return out;
    }
    const ROLE_COL = { sys: '#7d8590', user: '#5aa6d8', harness: '#e0a060', bot: '#5ad07a', call: '#e8c25a' };
    const ROLE_NAME = { sys: '系统提示', user: '用户', harness: 'Harness 写入', bot: '模型生成', call: '模型生成的工具调用' };
    /**
     * 把一串 token 画成一行行的小块：颜色 = 谁写的；o.mask（布尔数组）为 false 的块画成虚框（例如「不计损失」）；
     * o.upto = 只画前 n 个（动画）；o.hi = 高亮第几个；o.x0 / o.y0 / o.w 指定区域。返回 { rows, y }（画到哪儿了）
     */
    function drawChips(ctx, ids, o) {
        o = o || {};
        const R = o.roles || roles(ids), x0 = o.x0 ?? 8, w = o.w ?? 700, px = o.px ?? 12, h = px + 8, gap = 2;
        let x = x0, y = o.y0 ?? 8, rows = 1;
        const n = Math.min(ids.length, o.upto ?? ids.length);
        ctx.font = LM.FONT(px); ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
        for (let i = 0; i < n; i++) {
            const t = itos[ids[i]], cw = Math.max(ctx.measureText(t).width + 6, px + 2);
            if (x + cw > x0 + w) { x = x0; y += h + 3; rows++; }
            const r = R[i], col = r.call && r.role === 'bot' ? ROLE_COL.call : ROLE_COL[r.role];
            const on = !o.mask || o.mask[i];
            ctx.globalAlpha = on ? 1 : 0.55;
            ctx.fillStyle = on ? col + '33' : 'transparent'; ctx.fillRect(x, y, cw, h);
            ctx.strokeStyle = col; ctx.lineWidth = i === o.hi ? 2.5 : 1; ctx.setLineDash(on ? [] : [3, 2]); ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, h - 1); ctx.setLineDash([]);
            ctx.fillStyle = on ? '#e6edf3' : '#8b949e'; ctx.fillText(t, x + cw / 2, y + h / 2 + 0.5);
            ctx.globalAlpha = 1;
            x += cw + gap;
        }
        return { rows, y: y + h };
    }
    /** 图例：几种颜色各代表谁写的 */
    function drawRoleLegend(ctx, x, y, keys) {
        ctx.font = LM.FONT(11); ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        for (const k of keys || ['sys', 'user', 'harness', 'bot', 'call']) { ctx.fillStyle = ROLE_COL[k] + '55'; ctx.fillRect(x, y - 5, 12, 10); ctx.strokeStyle = ROLE_COL[k]; ctx.strokeRect(x + 0.5, y - 4.5, 11, 9); ctx.fillStyle = '#c9d1d9'; ctx.fillText(ROLE_NAME[k], x + 16, y); x += 16 + ctx.measureText(ROLE_NAME[k]).width + 14; }
        return x;
    }
    /**
     * 分帧评测：jobs = [{ kind: 'prefix', prefix, done(p) } | { kind: 'episode', start() → 回合对象, done(result) }]；
     * work(B, budget) 每帧最多做 budget 次 token 前向（约 0.1–0.2 ms 一次），做完的 job 立刻回调。B.k = 已完成的 job 数
     */
    function newBench(M, jobs) { return { M, jobs, k: 0, cur: null, done: !jobs.length, forwards: 0 }; }
    function benchWork(B, budget) {
        let used = 0;
        while (!B.done && used < budget) {
            const j = B.jobs[B.k];
            if (j.kind === 'prefix') { const S = newSession(B.M), lg = S.feedAll(encode(j.prefix)); j.done(LM.softmax(lg, new Float32Array(lg.length), 1)); used += S.forwards; }
            else {
                if (!B.cur) B.cur = j.start();
                const f0 = B.cur.S.forwards; B.cur.advance(budget - used); used += B.cur.S.forwards - f0;
                if (!B.cur.done) break;
                j.done(B.cur.result); B.cur = null;
            }
            B.k++; if (B.k >= B.jobs.length) B.done = true;
        }
        B.forwards += used;
        return used;
    }
    /** 二项比例的 95% Wilson 置信区间 */
    function wilson(k, n) { if (!n) return [0, 1]; const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [Math.max(0, c - h), Math.min(1, c + h)]; }
    return {
        SPECIALS, SP, V, itos, stoi, encode, decode, isSpecial, ANIMALS, PLACES, ROOMS, ADJ, ADJ2, ITEMS, TOOLS,
        calc, newWorld, lookup, newHouse, itemsIn, obs, houseStep, path, sysText, qaTask, qaExpert, houseTask, houseSuccess, houseExpert, memoText,
        serialize, sample, batch, newSession, allowedNext, newEpisode, runEpisode, houseEnv, qaEnv, newHouseEpisode, runHouse,
        roles, ROLE_COL, ROLE_NAME, drawChips, drawRoleLegend, newBench, benchWork, wilson,
    };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = AGT;
