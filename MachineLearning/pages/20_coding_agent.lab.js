// 20_coding_agent.lab.js — 系列第 20 篇：编程 Agent——几个通用工具（读 / 搜 / 改 / 跑命令），怎样做成千上万种事
// 构建（在 pages/ 下）：node <skill>/scripts/build_doc.mjs llm_common.js tk_bpe.js tools_catalog.js tools_common.js tools_cpp.js coding_common.js coding_cpp.js 20_coding_agent.lab.js -o 20_coding_agent.html
// 数据：coding_common.js 里的小项目 tiny-rpg-core（20 个文件，带一个真实的 bug 和 2 个隐藏测试）；工具、Shell、测试运行器、仓库地图都在页面里真实执行。
const P20 = (() => {
    const C = LM.C;
    const mono = (px) => `${px}px ui-monospace, "Cascadia Mono", Consolas, "Microsoft YaHei", monospace`;
    function fit(ctx, s, w) {
        s = String(s);
        if (w <= 4) return '';
        if (ctx.measureText(s).width <= w) return s;
        let lo = 0, hi = s.length;
        while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ctx.measureText(s.slice(0, m) + '…').width <= w) lo = m; else hi = m - 1; }
        return s.slice(0, lo) + '…';
    }
    /** 按宽度折行：英文单词不拆开，中文逐字 */
    function wrap(ctx, s, w) {
        const out = [];
        for (const para of String(s).split('\n')) {
            const toks = para.match(/[\x21-\x7e]+|\s+|[^\x00-\x7f]/g) || [''];
            let cur = '';
            for (const t of toks) {
                if (cur && ctx.measureText(cur + t).width > w) { out.push(cur.replace(/\s+$/, '')); cur = /^\s+$/.test(t) ? '' : t; }
                else cur += t;
            }
            out.push(cur);
        }
        return out;
    }
    function lineColor(l) {
        if (/^\$ /.test(l)) return C.yellow;
        if (/^\s*✓/.test(l)) return C.green;
        if (/^\s*✗/.test(l)) return C.red;
        if (/^(diff --git|--- a\/|\+\+\+ b\/)/.test(l)) return '#9aa0a8';
        if (/^@@/.test(l)) return C.cyan;
        if (/^\+/.test(l)) return C.green;
        if (/^-/.test(l)) return C.red;
        if (/\b0 failed\b/.test(l)) return C.green;
        if (/\d+ failed\b/.test(l)) return C.red;
        if (/^> |^\s+at /.test(l)) return C.dim;
        if (/(Error|error:|拒绝|超时|not found|No such|Invalid|unsupported)/.test(l)) return C.red;
        return C.text;
    }
    /** 一行彩色片段 [{ t, color, bold }]，超出宽度截断 */
    function segs(ctx, x, y, parts, w, px) {
        let cx = x;
        for (const p of parts) {
            ctx.font = (p.bold ? 'bold ' : '') + mono(p.px || px);
            ctx.fillStyle = p.color || C.text;
            const t = fit(ctx, p.t, x + w - cx);
            if (!t) break;
            ctx.fillText(t, cx, y);
            cx += ctx.measureText(t).width;
        }
        return cx;
    }
    /** 一行终端输出：cat -n 的行号变暗、grep 的 路径:行号: 前缀变青，其余按内容上色 */
    function termLine(ctx, l, x, y, w, o) {
        let m = /^(\s*\d+)\t(.*)$/.exec(l);
        if (m) {
            ctx.fillStyle = C.dim; ctx.fillText(m[1], x, y);
            const nx = x + ctx.measureText(m[1] + '  ').width;
            ctx.fillStyle = o.hl && o.hl.test(m[2]) ? C.yellow : C.text;
            ctx.fillText(fit(ctx, m[2], w - (nx - x)), nx, y);
            return;
        }
        m = /^([\w./-]+\.\w+:\d+:)(.*)$/.exec(l);
        if (m && !(o.color && o.color(l))) {
            ctx.fillStyle = C.cyan; ctx.fillText(m[1], x, y);
            const nx = x + ctx.measureText(m[1]).width;
            ctx.fillStyle = o.hl && o.hl.test(m[2]) ? C.yellow : C.text;
            ctx.fillText(fit(ctx, m[2], w - (nx - x)), nx, y);
            return;
        }
        ctx.fillStyle = (o.color && o.color(l)) || lineColor(l);
        ctx.fillText(fit(ctx, l.replace(/\t/g, '    '), w), x, y);
    }
    /** 终端风格的输出；o.head = 顶部的彩色片段行（数组的数组），o.hl = 要高亮的代码行（正则），o.color = 自定义上色 */
    function term(c, text, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c), px = o.px || 11.5, lh = Math.round(px + 4.5);
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        let y = 8;
        for (const row of o.head || []) { segs(ctx, 8, y, row, w - 16, px); y += lh + 2; }
        if (o.head && o.head.length) { ctx.strokeStyle = '#2a2f36'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(8, y + 0.5); ctx.lineTo(w - 8, y + 0.5); ctx.stroke(); y += 6; }
        ctx.font = mono(px);
        const lines = String(text == null ? '' : text).replace(/\n+$/, '').split('\n');
        const maxL = Math.max(1, Math.floor((h - y - 4) / lh)), show = lines.length > maxL ? lines.slice(0, maxL - 1) : lines;
        show.forEach((l, i) => termLine(ctx, l, 8, y + i * lh, w - 16, o));
        if (lines.length > maxL) { ctx.fillStyle = C.dim; ctx.fillText(`…（还有 ${lines.length - show.length} 行）`, 8, y + show.length * lh); }
    }
    /** JSON 逐行上色（键 = 青、字符串 = 绿、数字 / 布尔 = 橙） */
    function drawJSON(ctx, obj, x, y, w, maxLines, px) {
        const lines = JSON.stringify(obj, null, 2).split('\n'), lh = px + 4;
        ctx.font = mono(px); ctx.textBaseline = 'top';
        lines.slice(0, maxLines).forEach((ln, i) => {
            let cx = x; const parts = ln.match(/("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?|true|false|null)|([^"\d\-tfn]+|.)/g) || [];
            for (const p of parts) {
                const isKey = /^"/.test(p) && /:\s*$/.test(p), isStr = /^"/.test(p), isNum = /^(-?\d|true|false|null)/.test(p);
                ctx.fillStyle = isKey ? C.cyan : isStr ? '#98c379' : isNum ? C.orange : C.text;
                const t = fit(ctx, p, Math.max(10, x + w - cx)); ctx.fillText(t, cx, y + i * lh); cx += ctx.measureText(t).width; if (cx > x + w - 12) break;
            }
        });
        if (lines.length > maxLines) { ctx.fillStyle = C.dim; ctx.fillText(`…（还有 ${lines.length - maxLines} 行）`, x, y + maxLines * lh); }
        return Math.min(lines.length, maxLines + 1);
    }
    const niceStep = (x) => { const p = Math.pow(10, Math.floor(Math.log10(Math.max(x, 1e-9)))); for (const m of [1, 2, 5, 10]) if (m * p >= x) return m * p; return 10 * p; };
    const CLS_ZH = { read: '只读', exec: '会执行 / 修改', deny: '拒绝名单' }, CLS_COL = { read: C.green, exec: C.yellow, deny: C.red };
    const DEC_ZH = { allow: '允许', ask: '问用户', deny: '拒绝' }, DEC_COL = { allow: C.green, ask: C.yellow, deny: C.red };
    const MODE_ZH = { default: '默认', acceptEdits: '自动接受改动', plan: '只做计划', bypass: '全部放行' };
    /** 一次调用的简短说明（表格里用） */
    function brief(call) {
        const i = call.input || {};
        if (call.name === 'Bash') return i.command;
        if (call.name === 'Grep') return `/${i.pattern}/ in ${i.path || '.'}`;
        if (call.name === 'Read') return `${i.file_path}${i.offset ? ` ${i.offset}–${i.offset + (i.limit || 2000) - 1}` : ''}`;
        if (call.name === 'Edit') return `${i.file_path}: ${String(i.old_string).trim()} → ${String(i.new_string).trim()}`;
        return JSON.stringify(i);
    }

    // —— 第 01 步：命令行等价物 ——
    function drawCli(c, S) {
        const { ctx, w, h } = PL.draw.ctx(c), K = S.cli, rows = K.rows, lh = Math.floor((h - 52) / rows.length);
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        segs(ctx, 8, 6, [{ t: '应用', color: C.gray, bold: true }, { t: '  ·  做成 MCP 专用工具：个数 / 定义的 token', color: C.red }, { t: '  ·  在 Bash 里：一条命令（下一行）', color: C.green }], w - 16, 11);
        rows.forEach((r, i) => {
            const y = 26 + i * lh;
            if (i % 2 === 0) { ctx.fillStyle = 'rgba(255,255,255,0.035)'; ctx.fillRect(4, y - 3, w - 8, lh); }
            segs(ctx, 8, y, [{ t: r.name, color: C.text, bold: true }], 160, 11);
            segs(ctx, 170, y, [{ t: `${r.n} 个工具 · ${r.tok.toLocaleString()} token`, color: C.red }], 210, 11);
            segs(ctx, 388, y, [{ t: `一条命令 ${r.ctok} token`, color: C.gray }, { t: '  ·  ', color: C.dim }, { t: CLS_ZH[r.cls.perm], color: CLS_COL[r.cls.perm] }], w - 396, 11);
            segs(ctx, 20, y + 14, [{ t: '$ ' + r.cmd, color: C.green }, { t: `   ${r.what}`, color: C.dim }], w - 28, 11);
        });
        const y = 26 + rows.length * lh + 4;
        segs(ctx, 8, y, [{ t: `合计 ${K.n} 个专用工具 = ${K.tok.toLocaleString()} token`, color: C.red, bold: true }, { t: '   vs   ', color: C.gray }, { t: `Bash 一个定义 ${K.bash} token（编程 Agent 的 8 个工具一共 ${K.general} token）`, color: C.green, bold: true }], w - 16, 11);
    }
    // —— 第 02 步：仓库的文件树 + 每个文件的 token ——
    function treeRows(base) {
        const rows = [], F = base.F0;
        const walk = (dir, depth) => {
            for (const e of CA.listDir(F, dir)) {
                const path = (dir ? dir + '/' : '') + e.replace(/\/$/, '');
                if (e.endsWith('/')) { let tok = 0; for (const g of base.files) if (g.startsWith(path + '/')) tok += base.tok.get(g); rows.push({ name: e, depth, dir: true, tok, path }); walk(path, depth + 1); }
                else rows.push({ name: e, depth, dir: false, tok: base.tok.get(path), path, lines: F.files.get(path).split('\n').length - 1 });
            }
        };
        walk('', 0);
        return rows;
    }
    function drawTree(c, S) {
        const { ctx, w, h } = PL.draw.ctx(c), B = S.base, rows = treeRows(B), max = Math.max(...rows.filter((r) => !r.dir).map((r) => r.tok));
        const lh = Math.min(13.5, (h - 30) / rows.length), x0 = 270, bw = w - x0 - 120;
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        segs(ctx, 8, 4, [{ t: `tiny-rpg-core · ${B.files.length} 个文件 · 共 ${B.total.toLocaleString()} token`, color: C.text, bold: true }, { t: '（16k BPE 实数；蓝 = 源码，橙 = 测试，红 = 数据）', color: C.gray }], w - 16, 11);
        rows.forEach((r, i) => {
            const y = 24 + i * lh;
            ctx.font = mono(11);
            ctx.fillStyle = r.dir ? C.cyan : r.path === 'src/inventory.js' ? C.yellow : C.text;
            ctx.fillText(fit(ctx, '  '.repeat(r.depth) + r.name + (r.dir ? '' : `  (${r.lines} 行)`), x0 - 14), 8, y);
            if (r.dir) { ctx.fillStyle = C.dim; ctx.fillText(`目录合计 ${r.tok.toLocaleString()}`, x0, y); return; }
            const bwi = Math.max(1.5, (bw * r.tok) / max);
            ctx.fillStyle = r.path.startsWith('data/') ? C.red : r.path.startsWith('tests/') ? C.orange : C.blue;
            ctx.globalAlpha = 0.85; ctx.fillRect(x0, y + 2, bwi, lh - 4); ctx.globalAlpha = 1;
            ctx.fillStyle = C.text; ctx.fillText(`${r.tok.toLocaleString()}${r.tok / B.total > 0.2 ? `（${((100 * r.tok) / B.total).toFixed(0)}%）` : ''}`, x0 + bwi + 6, y);
        });
    }
    // —— 第 04 步：仓库地图 ——
    function drawMap(c, M, budget) {
        const { ctx, w, h } = PL.draw.ctx(c), px = 11, lh = 15;
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        const nLines = (M.map.match(/^│/gm) || []).length;
        segs(ctx, 8, 6, [{ t: `仓库地图 ${M.tokens} / ${budget} token`, color: C.text, bold: true }, { t: ` · ${nLines} 行定义（全仓库 ${M.nDefs} 个）`, color: C.gray }, { t: ` · 需求关键词：${M.terms.join(', ') || '（无）'}`, color: C.yellow }], w - 16, 11);
        const lines = M.map.replace(/\n$/, '').split('\n'), top = 28, rows = Math.floor((h - top - 6) / lh), cols = lines.length > rows ? 2 : 1, cw = (w - 16) / cols;
        for (let i = 0; i < Math.min(lines.length, rows * cols); i++) {
            const col = Math.floor(i / rows), r = i % rows, l = lines[i], x = 8 + col * cw, y = top + r * lh;
            ctx.font = /:$/.test(l) ? 'bold ' + mono(px) : mono(px);
            ctx.fillStyle = /:$/.test(l) ? C.cyan : l === '⋮' ? C.dim : C.text;
            ctx.fillText(fit(ctx, l, cw - 12), x, y);
        }
        if (lines.length > rows * cols) { ctx.font = mono(px); ctx.fillStyle = C.dim; ctx.fillText(`…（还有 ${lines.length - rows * cols} 行）`, w - 170, h - lh); }
    }
    function drawGraph(c, M) {
        const { ctx, w, h } = PL.draw.ctx(c), G = M.graph, n = G.files.length, cx = w / 2, cy = h / 2 + 4, R = Math.min(w * 0.30, h / 2 - 30);
        const maxR = Math.max(...G.rank), maxW = Math.max(1e-9, ...G.edges.map((e) => e[2]));
        const ang = (i) => -Math.PI / 2 + (2 * Math.PI * i) / n, pos = G.files.map((_, i) => [cx + R * Math.cos(ang(i)), cy + R * Math.sin(ang(i))]);
        ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.font = mono(10.5); ctx.fillStyle = C.gray; ctx.fillText('边 = 引用（粗细 ∝ 权重），圆 = PageRank', 6, 4);
        for (const [u, v, wt] of G.edges) {
            const [x1, y1] = pos[u], [x2, y2] = pos[v], t = Math.log1p(wt) / Math.log1p(maxW);
            ctx.strokeStyle = `rgba(106,176,224,${0.15 + 0.6 * t})`; ctx.lineWidth = 0.6 + 2.6 * t;
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
            const a = Math.atan2(y2 - y1, x2 - x1), ex = x2 - 9 * Math.cos(a), ey = y2 - 9 * Math.sin(a);
            ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - 6 * Math.cos(a - 0.4), ey - 6 * Math.sin(a - 0.4)); ctx.lineTo(ex - 6 * Math.cos(a + 0.4), ey - 6 * Math.sin(a + 0.4)); ctx.fill();
        }
        const order = G.rank.map((r, i) => [r, i]).sort((a, b) => b[0] - a[0]), top3 = new Set(order.slice(0, 3).map((x) => x[1]));
        G.files.forEach((f, i) => {
            const [x, y] = pos[i], r = 3 + 11 * Math.sqrt(G.rank[i] / maxR);
            ctx.fillStyle = top3.has(i) ? C.green : f.startsWith('tests/') ? C.orange : C.blue;
            ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
            const a = ang(i), lx = cx + (R + r + 6) * Math.cos(a), ly = cy + (R + r + 6) * Math.sin(a);
            ctx.font = mono(10); ctx.fillStyle = top3.has(i) ? C.green : C.text; ctx.textAlign = Math.cos(a) > 0.2 ? 'left' : Math.cos(a) < -0.2 ? 'right' : 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(f.replace(/^src\//, '').replace(/^tests\//, '').replace(/\.js$/, ''), Math.min(w - 10, Math.max(10, lx)), ly);
        });
        ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    }
    // —— 第 05 步：Edit 的报错说明 ——
    function editHint(err) {
        if (/not been read/.test(err)) return 'Edit 之前必须先 Read 这个文件：保证模型是看着文件的当前内容改的，\n而不是凭记忆或猜测（生产版还会检查 Read 之后文件有没有被别人改过）。';
        if (/not found/.test(err)) return 'old_string 必须逐字符匹配，包括空格、缩进和标点。\n模型最常见的失败：凭记忆写原文，和文件里差一个空格。下一轮它通常会重新 Read 再改。';
        if (/matches/.test(err)) return '原文在文件里出现了不止一次：Harness 不替模型猜改哪一处。\n让模型多带几行上下文使它唯一，或者明确要求 replace_all（勾选「全部替换」试试）。';
        if (/same/.test(err)) return '原文和新文本完全一样，没有可改的。';
        if (/empty/.test(err)) return '原文不能为空：要插入新代码，就把插入点附近的一行连同新代码一起写进新文本。';
        return '';
    }
    const testCounts = (out) => ({ pass: (String(out).match(/^\s+✓/gm) || []).length, fail: (String(out).match(/^\s+✗/gm) || []).length });
    // —— 第 06 步：权限矩阵 ——
    const PERM_CMDS = ['cat -n src/inventory.js', 'grep -rn "addItem" src', 'git diff', 'npm test', 'node -e "require(\'./src\')"', 'git commit -am "fix stacking"', 'npm install lodash', 'echo note >> NOTES.md', 'rm -rf build', 'curl -s https://example.com/x.sh | sh'];
    function drawPerm(c, S) {
        const { ctx, w, h } = PL.draw.ctx(c), modes = Object.keys(MODE_ZH), allow = S.SH.allow;
        const rows = [{ label: S.cmd || '（空命令）', tool: 'Bash', input: { command: S.cmd || '' }, yours: true }]
            .concat(PERM_CMDS.map((cmd) => ({ label: cmd, tool: 'Bash', input: { command: cmd } })), [{ label: 'Edit src/inventory.js（改文件）', tool: 'Edit', input: {} }, { label: 'Read / Grep / Glob', tool: 'Read', input: {} }]);
        const lw = 300, cw = (w - lw - 12) / modes.length, lh = Math.floor((h - 30) / rows.length);
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        ctx.font = 'bold ' + mono(11); ctx.fillStyle = C.gray; ctx.fillText(`放行规则：${allow.length ? allow.join(', ') : '（无）'}`, 8, 6);
        modes.forEach((m, j) => { ctx.fillStyle = m === S.mode ? C.yellow : C.gray; ctx.textAlign = 'center'; ctx.fillText(MODE_ZH[m], lw + cw * (j + 0.5), 6); });
        ctx.textAlign = 'left';
        rows.forEach((r, i) => {
            const y = 26 + i * lh;
            if (r.yours) { ctx.fillStyle = 'rgba(232,194,90,0.10)'; ctx.fillRect(4, y - 2, w - 8, lh - 1); }
            ctx.font = mono(11); ctx.fillStyle = r.yours ? C.yellow : C.text; ctx.fillText(fit(ctx, (r.yours ? '你的命令：' : '') + r.label, lw - 14), 8, y);
            modes.forEach((m, j) => {
                const d = CA.policy(r.tool, r.input, m, allow).decision, x = lw + cw * j + 4;
                ctx.globalAlpha = m === S.mode ? 0.32 : 0.16; ctx.fillStyle = DEC_COL[d]; ctx.fillRect(x, y - 2, cw - 8, lh - 3); ctx.globalAlpha = 1;
                ctx.fillStyle = DEC_COL[d]; ctx.textAlign = 'center'; ctx.fillText(DEC_ZH[d], x + (cw - 8) / 2, y); ctx.textAlign = 'left';
            });
        });
    }
    // —— 第 07 步：轨迹 ——
    function drawTurn(c, S) {
        const T = S.traj[S.plan], k = Math.min(S.turn, T.steps.length) - 1, st = T.steps[k];
        const { ctx, w, h } = PL.draw.ctx(c);
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        let y = 6;
        segs(ctx, 8, y, [{ t: `${S.plan === 'smart' ? '聪明' : '笨'}的轨迹 · 第 ${k + 1} / ${T.steps.length} 步`, color: C.text, bold: true }, { t: `   调用 ${st.tCall} + 结果 ${st.tRes} token → 上下文累计 ${st.ctx.toLocaleString()} token`, color: C.gray }], w - 16, 11.5);
        y += 22;
        ctx.font = mono(11.5); ctx.fillStyle = C.yellow;
        for (const l of wrap(ctx, '模型：' + st.think, w - 16).slice(0, 2)) { ctx.fillText(l, 8, y); y += 16; }
        y += 4;
        ctx.font = mono(11); ctx.fillStyle = C.gray; ctx.fillText(`tool_use${st.perm ? `（命令分级：${CLS_ZH[st.perm.perm]}）` : ''}`, 8, y); y += 15;
        const nj = drawJSON(ctx, { name: st.call.name, input: st.call.input }, 14, y, w - 22, 9, 11);
        y += nj * 15 + 6;
        ctx.font = mono(11); ctx.fillStyle = st.error ? C.red : C.gray; ctx.fillText(`tool_result${st.error ? '（is_error）' : ''} · ${st.tRes} token`, 8, y); y += 15;
        const lines = st.result.replace(/\n+$/, '').split('\n'), lh = 15, maxL = Math.max(1, Math.floor((h - y - 4) / lh)), show = lines.length > maxL ? lines.slice(0, maxL - 1) : lines;
        ctx.font = mono(11);
        show.forEach((l, i) => termLine(ctx, l, 14, y + i * lh, w - 22, { hl: /const room/ }));
        if (lines.length > maxL) { ctx.fillStyle = C.dim; ctx.fillText(`…（还有 ${lines.length - show.length} 行，全部进了上下文）`, 14, y + show.length * lh); }
    }
    const sumCtx = (T) => [T.fixed].concat(T.steps.map((s) => s.ctx)).reduce((a, b) => a + b, 0);
    function drawGrowth(c, S) {
        const { ctx, w, h } = PL.draw.ctx(c), sm = S.traj.smart, nv = S.traj.naive;
        const ser = [{ k: 'smart', col: C.green, pts: [sm.fixed].concat(sm.steps.map((s) => s.ctx)) }, { k: 'naive', col: C.red, pts: [nv.fixed].concat(nv.steps.map((s) => s.ctx)) }];
        const L = 44, R = w - 14, T = 40, Bm = h - 24, nX = Math.max(...ser.map((s) => s.pts.length)) - 1, maxY = Math.max(...ser.flatMap((s) => s.pts)) * 1.08;
        const X = (i) => L + ((R - L) * i) / nX, Y = (v) => Bm - ((Bm - T) * v) / maxY;
        ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.font = mono(10.5);
        ctx.fillStyle = C.green; ctx.fillText(fit(ctx, `聪明：${sm.steps.length} 次调用 · 上下文 ${Math.round(sm.fixed / 100) / 10}K → ${Math.round(sm.steps[sm.steps.length - 1].ctx / 100) / 10}K · 累计输入 ${sumCtx(sm).toLocaleString()} token`, w - 12), 6, 4);
        ctx.fillStyle = C.red; ctx.fillText(fit(ctx, `笨：${nv.steps.length} 次调用 · 第一步就到 ${Math.round(nv.steps[0].ctx / 100) / 10}K · 累计输入 ${sumCtx(nv).toLocaleString()} token`, w - 12), 6, 19);
        const step = niceStep(maxY / 4);
        ctx.strokeStyle = '#23272e'; ctx.lineWidth = 1; ctx.fillStyle = C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (let v = 0; v <= maxY; v += step) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(R, Y(v)); ctx.stroke(); ctx.fillText(v >= 1000 ? `${v / 1000}K` : String(v), L - 4, Y(v)); }
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        for (let i = 0; i <= nX; i++) ctx.fillText(i === 0 ? '开始' : String(i), X(i), Bm + 5);
        for (const s of ser) {
            ctx.strokeStyle = s.col; ctx.lineWidth = s.k === S.plan ? 2.4 : 1.4; ctx.beginPath(); s.pts.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
            ctx.fillStyle = s.col; s.pts.forEach((v, i) => { ctx.beginPath(); ctx.arc(X(i), Y(v), 2.5, 0, 2 * Math.PI); ctx.fill(); });
        }
        const cur = ser.find((s) => s.k === S.plan), k = Math.min(S.turn, cur.pts.length - 1);
        ctx.strokeStyle = C.yellow; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(k), Y(cur.pts[k]), 6, 0, 2 * Math.PI); ctx.stroke();
        ctx.textAlign = 'left';
    }
    function drawTurnTable(c, S) {
        const { ctx, w, h } = PL.draw.ctx(c), T = S.traj[S.plan], k = Math.min(S.turn, T.steps.length) - 1, lh = Math.min(24, (h - 44) / (T.steps.length + 1));
        ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.font = 'bold ' + mono(10.5); ctx.fillStyle = C.gray;
        ctx.fillText('#', 6, 6); ctx.fillText('工具', 22, 6); ctx.fillText('参数', 70, 6); ctx.textAlign = 'right'; ctx.fillText('调用', w - 58, 6); ctx.fillText('结果', w - 8, 6); ctx.textAlign = 'left';
        T.steps.forEach((s, i) => {
            const y = 26 + i * lh;
            if (i === k) { ctx.fillStyle = 'rgba(232,194,90,0.14)'; ctx.fillRect(2, y - 3, w - 4, lh - 2); }
            ctx.font = mono(10.5); ctx.fillStyle = C.gray; ctx.fillText(String(i + 1), 6, y);
            ctx.fillStyle = C.cyan; ctx.fillText(s.call.name, 22, y);
            ctx.fillStyle = C.text; ctx.fillText(fit(ctx, brief(s.call), w - 70 - 108), 70, y);
            ctx.textAlign = 'right'; ctx.fillText(String(s.tCall), w - 58, y); ctx.fillStyle = s.tRes > 2000 ? C.red : C.text; ctx.fillText(s.tRes.toLocaleString(), w - 8, y); ctx.textAlign = 'left';
        });
        const y = 26 + T.steps.length * lh + 4;
        ctx.font = mono(10.5); ctx.fillStyle = C.gray; ctx.fillText(fit(ctx, `固定部分（系统提示 + 8 个工具定义 + 任务）：${T.fixed} token`, w - 12), 6, y);
    }
    // —— 第 08 步：子 Agent ——
    function drawSub(c, S) {
        const { ctx, w, h } = PL.draw.ctx(c), A = S.subA, lh = 14.5;
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        let y = 6;
        segs(ctx, 8, y, [{ t: 'Task（子 Agent）收到的任务：', color: C.gray }, { t: A.prompt, color: C.text }], w - 16, 11); y += 20;
        for (const l of A.log) {
            if (y > h - 120) break;
            segs(ctx, 8, y, [{ t: `▸ ${l.tool} `, color: C.cyan, bold: true }, { t: brief({ name: l.tool, input: l.input }), color: C.text }, { t: `   · ${l.t} token（只进子 Agent 的上下文）`, color: C.gray }], w - 16, 11); y += lh + 1;
            ctx.font = mono(10.5);
            const out = l.out.replace(/\n+$/, '').split('\n'), keep = l.tool === 'Grep' ? 6 : 3;
            for (const ln of out.slice(0, keep)) { termLine(ctx, ln, 24, y, w - 32, { hl: /room|gold|Math\.min\(q/ }); y += lh - 1; }
            if (out.length > keep) { ctx.fillStyle = C.dim; ctx.fillText(`   …（共 ${out.length} 行）`, 24, y); y += lh - 1; }
            y += 3;
        }
        y += 4;
        segs(ctx, 8, y, [{ t: `报告（主 Agent 只收到这一段，${A.tReport} token）：`, color: C.green, bold: true }], w - 16, 11); y += lh + 2;
        ctx.font = mono(11); ctx.fillStyle = C.text;
        for (const l of wrap(ctx, A.report, w - 24)) { if (y > h - 14) break; ctx.fillText(l, 16, y); y += lh + 1; }
    }
    // —— 第 09 步：评测矩阵 ——
    function drawVerify(c, S) {
        const { ctx, w, h } = PL.draw.ctx(c), FX = S.fixed;
        const cols = [{ name: '原始代码', ev: FX.orig }, { name: '你的改动', ev: S.E.ev, yours: true }, { name: '正确修复', ev: FX.good }, { name: '截断', ev: FX.clamp }, { name: 'room − 1', ev: FX.minus1 }];
        const ok = (ev, n) => !!(ev && ev.res[n] && ev.res[n].ok), names = Object.keys(FX.orig.res), lw = 348, cw = (w - lw - 10) / cols.length;
        // 按套件分组：inventory / loot / damage / save / inventory (hidden)，组标题一行，下面只写用例名
        const rows = [];
        let last = null;
        for (const n of names) { const [suite, ...rest] = n.split(' > '); if (suite !== last) { rows.push({ suite }); last = suite; } rows.push({ n, name: rest.join(' > '), f2p: !ok(FX.orig, n), hidden: /\(hidden\)/.test(n) }); }
        const lh = Math.min(17, (h - 44 - rows.length * 0) / (rows.length + 4)), top = 24;
        ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        ctx.font = 'bold ' + mono(11);
        cols.forEach((col, j) => { ctx.fillStyle = col.yours ? C.yellow : C.gray; ctx.textAlign = 'center'; ctx.fillText(col.name, lw + cw * (j + 0.5), 6); });
        ctx.textAlign = 'left'; ctx.fillStyle = C.gray; ctx.fillText('测试（F2P = 修之前失败、P2P = 修之前就通过）', 8, 6);
        rows.forEach((r, i) => {
            const y = top + i * lh;
            if (r.suite) {
                const hidden = /\(hidden\)/.test(r.suite);
                if (hidden) { ctx.fillStyle = 'rgba(198,120,221,0.07)'; ctx.fillRect(4, y - 1, w - 8, lh * 3); }
                ctx.font = 'bold ' + mono(10.5); ctx.fillStyle = hidden ? C.purple : C.cyan; ctx.fillText(r.suite.replace(' (hidden)', ' · 隐藏测试'), 8, y + 2);
                return;
            }
            ctx.font = mono(10.5);
            ctx.fillStyle = r.f2p ? C.orange : C.dim; ctx.fillText(r.f2p ? 'F2P' : 'P2P', 14, y);
            ctx.fillStyle = C.text; ctx.fillText(fit(ctx, r.name, lw - 60), 46, y);
            cols.forEach((col, j) => { const p = ok(col.ev, r.n), x = lw + cw * (j + 0.5); ctx.fillStyle = p ? C.green : C.red; ctx.textAlign = 'center'; ctx.fillText(p ? '✓' : '✗', x, y); ctx.textAlign = 'left'; });
        });
        const vis = names.filter((n) => !/\(hidden\)/.test(n)), f2p = names.filter((n) => !ok(FX.orig, n)), p2p = names.filter((n) => ok(FX.orig, n));
        const y1 = top + rows.length * lh + 8, y2 = y1 + lh + 2;
        ctx.strokeStyle = '#2a2f36'; ctx.beginPath(); ctx.moveTo(6, y1 - 5); ctx.lineTo(w - 6, y1 - 5); ctx.stroke();
        ctx.font = 'bold ' + mono(10.5); ctx.fillStyle = C.text; ctx.fillText('Agent 能看到的：可见测试全过？', 8, y1); ctx.fillText('SWE-bench 判定：resolved？', 8, y2);
        cols.forEach((col, j) => {
            const x = lw + cw * (j + 0.5), a = vis.every((n) => ok(col.ev, n)), r = f2p.every((n) => ok(col.ev, n)) && p2p.every((n) => ok(col.ev, n));
            ctx.textAlign = 'center'; ctx.fillStyle = a ? C.green : C.red; ctx.fillText(a ? '是' : '否', x, y1); ctx.fillStyle = r ? C.green : C.red; ctx.fillText(r ? '修好了' : '没修好', x, y2); ctx.textAlign = 'left';
        });
        if (S.E.ev.error) { ctx.font = mono(10.5); ctx.fillStyle = C.red; ctx.fillText(fit(ctx, `你的改动加载失败：${S.E.ev.error}`, w - 16), 8, h - 14); }
    }
    // —— 第 10 步：项目说明文件 ——
    const MEMO = `# CLAUDE.md — tiny-rpg-core
## Commands
- Test everything: npm test   (run it after every change; never say "fixed" before it passes)
- One file: node tests/run.js
## Rules
- Plain CommonJS, no new dependencies, 4-space indent.
- Inventory invariants: no slot above maxStack; adding items never loses any.
- data/items.json is generated from the design spreadsheet: never edit it by hand.
- Bump VERSION in src/save.js only together with a migration and a test.

# AGENTS.md — ProjectX (Unreal Engine 5)
- Build: Engine/Build/BatchFiles/Build.bat ProjectXEditor Win64 Development -Project="%CD%/ProjectX.uproject" -WaitMutex
- Tests: UnrealEditor-Cmd.exe ProjectX.uproject -ExecCmds="Automation RunTests ProjectX; Quit" -unattended -nullrhi
- .uasset / .umap are binary: never edit them as text. Change assets with the editor Python API
  (Tools/Scripts/*.py) via UnrealEditor-Cmd ProjectX.uproject -run=pythonscript -script=<file>.
- Perforce: files are read-only until checked out. Run p4 edit <file> before Edit, p4 add for new files.
- Never touch Content/Maps/Release/ without asking.`;

    return { C, mono, fit, wrap, term, termLine, segs, drawJSON, brief, drawCli, drawTree, drawMap, drawGraph, editHint, testCounts, drawPerm, drawTurn, drawGrowth, drawTurnTable, drawSub, drawVerify, MEMO, CLS_ZH, CLS_COL, DEC_ZH, DEC_COL, MODE_ZH };
})();

PipelineLab.mount({
    paramSteps: { task: ['map'], budget: ['map'], cmd: ['shell'], mode: ['shell'], allow: ['shell'], editFile: ['edit', 'shell', 'verify'], editOld: ['edit', 'shell', 'verify'], editNew: ['edit', 'shell', 'verify'], replaceAll: [], readFirst: ['edit', 'shell', 'verify'], plan: ['loop'], turn: ['loop'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '20 · 编程 Agent：几个通用工具，做成千上万种事',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 20,
    css: LM.CSS + `.pl-ctrl[data-param="task"] input[type=text],.pl-ctrl[data-param="cmd"] input[type=text]{width:20em}.pl-ctrl[data-param="editOld"] input[type=text],.pl-ctrl[data-param="editNew"] input[type=text]{width:17em}.pl-mini[data-param="task"] input[type=text],.pl-mini[data-param="cmd"] input[type=text],.pl-mini[data-param="editOld"] input[type=text],.pl-mini[data-param="editNew"] input[type=text]{width:28em;max-width:100%}.pl-ctrl input[type=text],.pl-mini input[type=text]{font-family:ui-monospace,Consolas,monospace}`,
    tagColors: { '工具': 'rust', '项目': 'gray', '搜索': 'blue', '地图': 'teal', '修改': 'magenta', '执行': 'olive', '循环': 'indigo', '子 Agent': 'green', '评测': 'rust', '生产版': 'gray' },

    params: [
        { id: 'task', label: '需求（仓库地图）', type: 'text', value: CA.TASK, alt: 'crit damage ignores armor', affects: 'param', desc: '交给 Agent 的需求，仓库地图按它个性化排序（中文会按一张小词表翻成代码里常见的英文词）。试试「存档迁移 v2 的金币」「crit damage ignores armor」。' },
        { id: 'budget', label: '地图预算', options: [{ value: 60, label: '60 token' }, { value: 120, label: '120 token' }, { value: 250, label: '250 token' }, { value: 500, label: '500 token' }], value: 250, affects: 'param', desc: '仓库地图最多占多少 token。预算越小，只剩排名最高的几个定义；Aider 默认给 1k token。' },
        { id: 'cmd', label: '命令', type: 'text', value: 'git diff && npm test', alt: 'grep -rn "count" src | wc -l', affects: 'param', desc: '在（改过的）虚拟仓库里执行的命令：ls、cat、head、tail、grep / rg、find、wc、git status / diff / log、npm test、node，支持 <code>|</code>、<code>&&</code>、<code>;</code>、<code>></code> 和通配符。' },
        { id: 'mode', label: '权限模式', options: [{ value: 'default', label: '默认（改动和执行要确认）' }, { value: 'acceptEdits', label: '自动接受改动' }, { value: 'plan', label: '只做计划（只读）' }, { value: 'bypass', label: '全部放行' }], value: 'default', affects: 'param', desc: 'Harness 怎么处理「会执行 / 修改」的调用：问用户、自动放行、还是拒绝。拒绝名单（rm、curl…）在任何模式下都生效。' },
        { id: 'allow', label: '放行规则', type: 'text', value: 'npm test', alt: 'npm test, git commit', affects: 'param', desc: '逗号分隔的命令前缀：匹配的命令不再问用户（相当于在设置里写 Bash(npm test)）。' },
        { id: 'editFile', label: 'Edit 的文件', options: ['src/inventory.js', 'src/save.js', 'src/damage.js', 'src/loot.js', 'src/quests.js'], value: 'src/inventory.js', affects: 'param', desc: 'Edit 工具要改的文件。改完以后，第 06 步的 Shell 和第 09 步的评测都运行在这个改过的仓库上。' },
        { id: 'editOld', label: '原文 old_string', type: 'text', value: 'const room = max - slot.count + 1;', alt: 'n -= k;', affects: 'param', desc: '要被替换的原文：必须在文件里逐字符出现、并且只出现一次。试试 <code>n -= k;</code>（3 处 → 报错）。' },
        { id: 'editNew', label: '新文本 new_string', type: 'text', value: 'const room = max - slot.count;', alt: 'const room = max - slot.count - 1;', affects: 'param', desc: '替换成什么。试试 <code>const room = max - slot.count - 1;</code>：可见测试全过，但第 09 步的隐藏测试会失败。' },
        { id: 'replaceAll', label: '全部替换', value: false, affects: 'param', desc: 'replace_all：原文出现多次时全部替换（不勾选时，多处匹配会报错）。' },
        { id: 'readFirst', label: '已经 Read 过这个文件', value: true, affects: 'param', desc: '取消勾选 = 模型没读过文件就想改：Edit 会拒绝，并告诉模型先 Read。' },
        { id: 'plan', label: '轨迹', options: [{ value: 'smart', label: '聪明：先复现、再搜索' }, { value: 'naive', label: '笨：先 cat 全部' }], value: 'smart', affects: 'param', desc: '两种修 bug 的做法：先跑测试、搜关键词、只读一段（聪明），或者先把整个项目读进上下文（笨）。' },
        { id: 'turn', label: '看第几步', min: 1, max: 6, step: 1, value: 3, affects: 'param', desc: '展开轨迹里的第几次工具调用：模型的想法、tool_use、真实执行出来的 tool_result。' },
    ],

    stages: [
        {
            id: 'base', lifetime: 'resident', run(S) {
                const B = TK.loadBPE(TK_BPE), F0 = CA.newFS(), files = [...F0.files.keys()].sort(), T = (s) => TK.countTokens(B, s);
                const tok = new Map(files.map((f) => [f, T(F0.files.get(f))]));
                let total = 0; for (const v of tok.values()) total += v;
                const Fs = CA.cloneFS(F0), grep = CA.toolGrep(Fs, 'maxStack', { path: 'src', output_mode: 'content', n: true }), grepAll = CA.toolGrep(Fs, 'maxStack', { output_mode: 'content', n: true }), read = CA.toolRead(Fs, 'src/inventory.js', 10, 21);
                const cost = [
                    { label: `cat 全部 ${files.length} 个文件`, tok: T(CA.runShell(CA.cloneFS(F0), 'cat ' + files.join(' ')).out), col: LM.C.red },
                    { label: 'cat src/*.js src/utils/*.js', tok: T(CA.runShell(CA.cloneFS(F0), 'cat src/*.js src/utils/*.js').out), col: LM.C.orange },
                    { label: 'Read src/inventory.js（整个）', tok: T(CA.toolRead(CA.cloneFS(F0), 'src/inventory.js').out), col: LM.C.yellow },
                    { label: 'Read inventory.js 第 10–30 行', tok: T(read.out), col: LM.C.green },
                    { label: 'Grep "maxStack" src（内容 + 行号）', tok: T(grep.out), col: LM.C.green },
                    { label: 'Grep "addItem"（只列文件名）', tok: T(CA.toolGrep(Fs, 'addItem', {}).out), col: LM.C.cyan },
                    { label: 'Glob "src/**/*.js"', tok: T(CA.toolGlob(Fs, 'src/**/*.js').out), col: LM.C.cyan },
                ];
                S.base = { B, F0, files, tok, total, test0: CA.runShell(CA.cloneFS(F0), 'npm test', B), cost, grep, grepAll, read };
            },
        },
        {
            id: 'cli', lifetime: 'resident', lazy: true, run(S) {
                const B = S.base.B, rows = CA.CLI.map((c) => {
                    const app = TKCAT.APPS.find((a) => a.ns === c.ns), tools = TKCAT.tools.filter((t) => t.ns === c.ns);
                    return { ns: c.ns, name: c.ns === 'sql' ? 'PostgreSQL' : app ? app.name : c.ns, n: tools.length, tok: tools.reduce((s, t) => s + TK.countTokens(B, TK.toolJSON(t)), 0), cmd: c.cmd, what: c.what, ctok: TK.countTokens(B, JSON.stringify({ command: c.cmd })), cls: CA.classify(c.cmd) };
                });
                S.cli = { rows, n: rows.reduce((s, r) => s + r.n, 0), tok: rows.reduce((s, r) => s + r.tok, 0), general: TK.countTokens(B, JSON.stringify(CA.TOOLS)), bash: TK.countTokens(B, JSON.stringify(CA.TOOLS.find((t) => t.name === 'Bash'))) };
            },
        },
        { id: 'traj', lifetime: 'resident', run(S) { const B = S.base.B; S.traj = { smart: CA.runPlan('smart', B), naive: CA.runPlan('naive', B) }; S.subA = CA.runSubagent(B); } },
        {
            id: 'fixed', lifetime: 'resident', run(S) {
                const out = { orig: CA.evaluate(S.base.F0) };
                for (const k of ['good', 'clamp', 'minus1']) { const F = CA.cloneFS(S.base.F0), fx = CA.FIXES[k]; F.read.add(fx.file); CA.toolEdit(F, fx.file, fx.old, fx.new); out[k] = CA.evaluate(F); }
                S.fixed = out;
            },
        },
        { id: 'map', lifetime: 'param', run(S) { const key = `${S.task}\u0001${S.budget}`; if (S._mapKey === key && S._mapBase === S.base) return; S._mapKey = key; S._mapBase = S.base; S.M = CA.repoMap(S.base.F0, S.task || '', +S.budget, S.base.B); } },
        {
            id: 'edit', lifetime: 'param', run(S) {
                const key = [S.editFile, S.editOld, S.editNew, S.replaceAll, S.readFirst].join('\u0001');
                if (S._edKey === key && S._edBase === S.base) return;
                S._edKey = key; S._edBase = S.base;
                const F = CA.cloneFS(S.base.F0);
                if (S.readFirst) F.read.add(S.editFile);
                const r = CA.toolEdit(F, S.editFile, S.editOld || '', S.editNew || '', S.replaceAll);
                const diff = r.error ? '' : CA.unifiedDiff(S.editFile, S.base.F0.files.get(S.editFile), F.files.get(S.editFile));
                S.E = { key, F, r, diff, t: CA.runTests(F, 'tests/run.js'), ev: CA.evaluate(F) };
            },
        },
        {
            id: 'sh', lifetime: 'param', run(S) {
                const key = [S.cmd, S.E.key, S.mode, S.allow].join('\u0001');
                if (S._shKey === key && S._shBase === S.base) return;
                S._shKey = key; S._shBase = S.base;
                const allow = String(S.allow || '').split(',').map((s) => s.trim()).filter(Boolean), pol = CA.policy('Bash', { command: S.cmd || '' }, S.mode, allow);
                S.SH = { allow, pol, r: pol.decision === 'deny' ? null : CA.runShell(CA.cloneFS(S.E.F), S.cmd || '', S.base.B) };
            },
        },
    ],

    intro: {
        title: '编程 Agent 没有一千个工具，只有一个 shell',
        flow: ['8 个通用工具 vs 440 个专用工具', '项目、任务与失败的测试', '先搜再读：Grep / Glob / Read', '仓库地图：PageRank 挑重点', 'Edit：精确替换的三条规则', 'Bash：任意命令 + 权限分级', '一次完整的修 bug 轨迹', '子 Agent：把脏活放进别的上下文', '测试全过 ≠ 修对了', '教学版 → 生产版'],
        html: LM.navHtml('20') + `
            <p>第 19 篇的回答是「工具太多就检索」。编程 Agent 给出了另一个回答：<strong>不需要那么多工具</strong>。Claude Code、OpenAI Codex CLI、Cursor 的 Agent 模式这类产品，核心只有十个左右的通用工具：读文件、搜文件、精确修改、写文件、跑 shell 命令，再加上子 Agent 和任务清单。Anthropic 在 SWE-bench Verified 上公布的参考实现更极端，只有一个 Bash 工具和一个文件编辑工具。</p>
            <p>原因是 <strong>shell 本身就是一个有几万个「工具」的目录</strong>：git、npm、cmake、docker、kubectl、ffmpeg、Unity 和虚幻引擎的命令行模式、<code>blender -b</code>……模型在预训练里读过它们的文档、脚本和问答，用法本来就会；不熟悉的命令就先跑 <code>--help</code>，或者读项目里的说明文件。于是 Harness 的工作从「在一千个定义里挑一个」变成了<strong>给模型一个安全、可观察、反馈快的环境</strong>：能高效地找到代码、能精确地改、能马上跑测试看结果；危险操作要经过确认，做完了能验证、能回滚。</p>
            <p>本篇在页面里放了一个真实的小项目：一个游戏的背包、掉落、伤害、存档逻辑（JavaScript，20 个文件），里面有一个玩家报告的 bug。工具都按真实产品的语义实现——Read 带行号，Edit 是精确替换并且要求先读后改，Grep 是正则搜索，Bash 是一个支持管道、<code>git diff</code> 和 <code>npm test</code> 的迷你 shell。<strong>测试真的在页面里运行</strong>：一个和 Node 一样的 CommonJS 加载器，加上死循环保护。你可以在顶栏输入自己的修改和命令，看测试、diff 和权限判定怎么变。</p>`,
    },

    steps: [
        {
            id: 'general', title: '8 个通用工具，抵得上 440 个专用工具', tag: '工具',
            desc: '第 19 篇的目录里，有 13 个应用其实都有命令行：git、GitHub（<code>gh</code>）、Docker、Kubernetes（<code>kubectl</code>）、AWS CLI、ffmpeg、ImageMagick、虚幻引擎（<code>UnrealEditor-Cmd</code>）、Unity（<code>-batchmode</code>）、Blender（<code>-b</code>）、Godot（<code>--headless</code>）、<code>psql</code>、<code>redis-cli</code>。做成 MCP 工具，它们一共是 440 个定义；在编程 Agent 里，它们全是 <strong>Bash 这一个工具</strong>的不同参数。下表每一行是该应用的一条真实命令，以及本页的命令分级（第 06 步）对它的判定。',
            formula: ['专用工具：上下文成本 = Σ_i tokens(def_i)          // 每种能力一个定义，每次请求都要带上', '通用工具：上下文成本 = tokens(Bash) + tokens(这条命令)   // 能力在模型的权重里，定义只有一个'],
            code: { title: 'C++ · Tool / ToolBox / dispatch（工具注册与分发）', src: CA_CPP.agenttools },
            previews: [
                {
                    id: 'genBars', needs: ['cli'], label: '把这些能力交给模型，要占多少上下文', sub: '本页实数（16k BPE）：第 19 篇目录里这 13 个应用的全部工具定义 vs 编程 Agent 的工具集', large: true, size: [720, 150],
                    render: (c, S) => { if (!S.cli) return; const K = S.cli; LM.draw.bars(c, [K.tok / 1000, K.general / 1000, K.bash / 1000], [`13 个应用的 ${K.n} 个专用工具`, '编程 Agent 的 8 个通用工具', '其中的 Bash（13 个应用都走它）'], { horizontal: true, range: [0, (K.tok / 1000) * 1.35], colors: [LM.C.red, LM.C.green, LM.C.blue], digits: 2, valueSuffix: 'K token' }); },
                },
                { id: 'cliTable', needs: ['cli'], label: '同一种能力：专用工具 vs 一条命令', sub: '分级 = 本页的命令分级：只读命令自动放行，其他的要确认（不认识的命令一律当成会改东西）', large: true, size: [720, 460], render: (c, S) => { if (S.cli) P20.drawCli(c, S); } },
            ],
            notes: [
                '<strong>什么时候还是该用专用工具</strong>：① 读、搜、改文件——专用的 Read / Grep / Edit 有行号、有截断、有「先读后改」的检查，权限也好管（Read 永远安全，Bash 要逐条判断）；② 需要登录态的远程服务（飞书、Jira、Figma、Steam 后台）——没有命令行或者鉴权麻烦，用 MCP；③ 正在运行的程序里的状态（编辑器里没保存的关卡、选中的物体）——命令行碰不到，要接编辑器的脚本接口（第 21 篇）；④ 只有图形界面的软件——最后才用 Computer Use。',
                '模型怎么知道这些命令？大部分来自预训练（文档、脚本、问答）；项目特有的命令写在项目说明文件里（CLAUDE.md / AGENTS.md，第 10 步），会话开始时整段放进上下文；还不知道就先跑 <code>xxx --help</code>——这本身也只是一次普通的 Bash 调用。',
            ],
            warn: 'Bash 的代价在安全上：一个工具能做任何事，也就能做任何坏事（删文件、把代码传出去、装恶意包）。所以通用工具一定要配命令分级、确认和沙箱（第 06、10 步）；专用工具天然就是一份白名单。',
        },
        {
            id: 'repo', title: '项目、任务与失败的测试', tag: '项目',
            desc: '项目叫 tiny-rpg-core：背包按物品叠放，每格最多 maxStack 个；掉落表带权重和种子；还有伤害公式、任务进度和带版本迁移的存档。玩家报告：<em>往背包里加药水时，一格里出现了 100 瓶（上限是 99）</em>。先看两件事：仓库有多大（每个文件的 BPE token 数），以及现在测试的结果。',
            code: { title: 'C++ · Workspace / readFileText / repoTokens（BPE 计数见第 06 篇）', src: TK_CPP.bpecount + '\n' + CA_CPP.repotokens },
            previews: [
                { id: 'tree', label: '仓库：每个文件有多少 token', sub: '黄色 = bug 所在的文件；数据文件一个就占了大半', large: true, size: [720, 370], render: (c, S) => P20.drawTree(c, S) },
                { id: 'failing', label: '$ npm test（原始代码，在页面里真实运行）', sub: '失败信息里的行号是真的：加载器和 Node 一样把每个文件包进一个函数，并保留文件名和行号', size: [720, 400], render: (c, S) => P20.term(c, S.base.test0.out, { head: [[{ t: '$ npm test', color: LM.C.yellow, bold: true }, { t: `   退出码 ${S.base.test0.code} · 输出 ${S.base.test0.tokens} token`, color: LM.C.gray }]] }) },
            ],
            notes: 'data/items.json（物品数据库）一个文件就占了全仓库 token 的大约 70%。真实项目里这种文件更多：生成的代码、锁文件、本地化表、关卡数据、日志。「先把整个仓库读进上下文」在任何像样的项目上都行不通，下一步看 Agent 怎样只读需要的部分。',
            warn: '测试失败的那一行 <code>tests/inventory.test.js:22</code> 往往是 Agent 最重要的线索：它说明了「哪个性质被破坏」（某一格超过了 99），比玩家的描述精确得多。所以好的 Agent 第一步几乎总是<strong>先复现</strong>。',
        },
        {
            id: 'search', title: '先搜再读：Grep / Glob / Read', tag: '搜索',
            desc: '人在陌生的代码库里找 bug，会先搜关键词，再打开命中的文件，只看相关的几十行。编程 Agent 也这样做，这叫<strong>按需检索</strong>（just-in-time context）：Glob 按文件名找，Grep 按内容找，找到以后用 Read 只读一个行号窗口。下图是几种「找到 bug 所在位置」的做法分别往上下文里放了多少 token（本页实算）。',
            formula: ['cost(cat 全部) ≈ tokens(仓库)                          // 和项目的大小成正比', 'cost(Grep → Read 一个窗口) ≈ tokens(命中行) + tokens(窗口)   // 和问题的大小成正比'],
            code: { title: 'C++ · toolRead / toolGrep', src: CA_CPP.readtool + '\n' + CA_CPP.greptool },
            previews: [
                { id: 'searchCost', label: '找到 bug 要放进上下文的 token', sub: '同一个仓库、同一个目标；绿色 = 聪明的 Agent 实际会用的两步', large: true, size: [720, 210], render: (c, S) => { const K = S.base.cost, mx = Math.max(...K.map((x) => x.tok)) / 1000; LM.draw.bars(c, K.map((x) => x.tok / 1000), K.map((x) => x.label), { horizontal: true, range: [0, mx * 1.25], colors: K.map((x) => x.col), digits: 1, valueSuffix: 'K token' }); } },
                { id: 'grepOut', label: 'Grep "maxStack" -n path=src（output_mode: content）', sub: '三行：哪个文件、哪一行。不限定 path 的话，前 137 行命中全在 data/items.json 里（那是物品数据库的字段名）', size: [720, 110], render: (c, S) => P20.term(c, S.base.grep.out, { head: [[{ t: 'path=src：3 行', color: LM.C.green }, { t: '   ·   path=.（全仓库）：', color: LM.C.gray }, { t: `${S.base.grepAll.matches} 行命中，前几行都是 data/items.json`, color: LM.C.red }]] }) },
                { id: 'readOut', label: 'Read src/inventory.js（offset 10，limit 21）', sub: 'cat -n 格式：行号 + 制表符 + 原文；高亮的是 bug 所在的行', size: [720, 370], render: (c, S) => P20.term(c, S.base.read.out, { hl: /const room/ }) },
            ],
            notes: [
                'Read 的输出带行号有两个用处：模型能说清楚「第 16 行」；Edit 失败时，模型能对着行号重新读一小段。Read 默认最多读 2000 行、超长的行截断——这类上限是 Harness 替模型装的保险丝。',
                'Claude Code 的开发者公开讲过：早期版本试过给代码库建向量索引（RAG），后来改成让模型自己用 Glob / Grep 搜，效果更好，也省掉了索引的同步、过期和代码外传问题。Cursor 这类 IDE 则同时维护语义索引。两条路线在生产里都有，取舍在于建索引的成本和检索的精度。',
            ],
            warn: '搜索词来自模型对问题的理解：报错信息里的标识符（maxStack、addItem）最可靠，自然语言的说法（「堆叠」）在代码里往往搜不到。所以先复现、拿到准确的报错再搜，比直接拿用户的描述去搜可靠得多。',
        },
        {
            id: 'map', title: '仓库地图：在固定预算里放最重要的定义', tag: '地图',
            desc: '搜索的前提是知道搜什么。面对完全陌生的仓库，Aider 先给模型一张<strong>仓库地图</strong>：每个文件里定义了哪些函数和类（只有签名那一行），按重要性排序，截到 token 预算以内。重要性用 PageRank 算：文件 A 引用了文件 B 里定义的符号，就连一条 A → B 的边，被很多文件引用的定义排名就高；需求里提到的词会给对应的文件和符号加权（中文先按一张小词表翻成代码里的英文词，真实系统里这一步由模型完成）。在顶栏改需求和预算，看地图怎么变。',
            formula: ['w(u→v) = Σ_{符号 s} mul(s) · √refs_u(s)                  // u 里引用 v 所定义的 s 的次数，开方', 'mul(s) = 10^{[长的驼峰或下划线名]} · 10^{[需求里提到]} · 0.1^{[私有]} · 0.1^{[≥ 5 个文件都定义]}', 'r = (1 − d)·p + d · Σ_u r_u · w(u→v) / Σ_{v′} w(u→v′)        // d = 0.85，p = 按需求个性化', 'rank(v, s) = Σ_u r_u · w_s(u→v) / Σ w(u→·)                // 文件的分数再分给具体的符号', 'm* = max{ m : tokens(render(排名前 m 的定义)) ≤ 预算 }        // 二分查找'],
            code: { title: 'C++ · buildGraph / pageRank / renderMap / fitToBudget', src: CA_CPP.repomap },
            previews: [
                { id: 'mapText', label: '交给模型的仓库地图', sub: '文件名（青）+ 定义所在的那一行；⋮ = 中间省略的代码', large: true, size: [720, 380], render: (c, S) => P20.drawMap(c, S.M, +S.budget) },
                { id: 'fileRank', label: '文件的 PageRank', sub: '个性化向量 p 来自需求关键词', size: [360, 250], render: (c, S) => { const f = S.M.fileRank.slice(0, 9); LM.draw.bars(c, f.map((x) => x.rank), f.map((x) => x.path), { horizontal: true, colors: f.map((x) => (x.path.startsWith('tests/') ? LM.C.orange : LM.C.blue)), digits: 3, labelPx: 11 }); } },
                { id: 'repoGraph', label: '引用图', sub: '边 = 一个文件引用了另一个文件里的定义（粗 = 权重高）；圆的大小 = PageRank；绿 = 排名前三', large: true, size: [720, 300], render: (c, S) => P20.drawGraph(c, S.M) },
            ],
            notes: [
                '预算 60 时只剩 Inventory 类和 getItem；250 时 addItem、count 出现；500 时几乎包含全部定义。Aider 默认给地图 1k token：在大仓库里这只是全部代码的很小一部分，却足够让模型知道该去哪个文件找。',
                '把需求改成 <code>crit damage ignores armor</code> 或「存档迁移 v2 的金币」，damage.js 或 save.js 会排到前面。',
            ],
            warn: '真实的实现用 tree-sitter 按语言解析出定义和引用（每种语言一份查询）；本页用正则近似 JavaScript 的 function、class、方法和箭头函数，会漏掉一些写法。PageRank 只看引用结构、不懂语义，被大量引用的工具函数（getItem、clamp）总会排得很高。',
        },
        {
            id: 'edit', title: 'Edit：精确替换，以及三条规则', tag: '修改',
            desc: '让模型改代码有几种格式：重写整个文件（简单，但贵，还容易丢内容）；统一 diff 或补丁（省 token，但模型常常算错行号）；<strong>查找替换</strong>（给出要替换的原文和新文本）。Claude Code 的 Edit、Anthropic API 的文本编辑工具（str_replace）、Aider 的 SEARCH/REPLACE 块都属于查找替换。本页的 Edit 按 Claude Code 的规则实现：<strong>① 改之前必须读过这个文件；② old_string 必须在文件里恰好出现一次（否则要明确 replace_all）；③ 新旧文本不能相同</strong>。在顶栏改「原文 / 新文本」，下面实时显示 Edit 的结果、git diff 和测试。',
            code: { title: 'C++ · toolEdit / lineDiff（LCS）', src: CA_CPP.edittool + '\n' + CA_CPP.diff },
            previews: [
                {
                    id: 'editResult', label: 'Edit 的结果（成功时显示 git diff）', sub: '失败时的报错原样作为 tool_result 交回模型；下面一段是这条报错想让模型做什么', large: true, size: [720, 250],
                    render: (c, S) => {
                        const E = S.E, ok = !E.r.error, head = [[{ t: 'Edit ', color: LM.C.yellow, bold: true }, { t: S.editFile, color: LM.C.text }, { t: ok ? `   ✓ 已更新（替换 ${E.r.count} 处）` : '   ✗ 失败', color: ok ? LM.C.green : LM.C.red, bold: true }]];
                        if (!ok) return P20.term(c, `<tool_use_error>${E.r.error}</tool_use_error>\n\n${P20.editHint(E.r.error)}`, { head, color: (l) => (/^</.test(l) ? LM.C.red : LM.C.text) });
                        P20.term(c, E.diff || '（没有变化）', { head });
                    },
                },
                {
                    id: 'editTests', label: '$ npm test（在改过的仓库上运行）', sub: '只有这 12 个可见测试是 Agent 能看到的；第 09 步再跑隐藏测试', size: [720, 275],
                    render: (c, S) => { const t = S.E.t, n = P20.testCounts(t.stdout); P20.term(c, t.stdout + (t.error ? t.error + '\n' : ''), { head: [[{ t: '$ npm test', color: LM.C.yellow, bold: true }, { t: `   ${n.pass} 个通过 · ${n.fail} 个失败 · 退出码 ${t.exitCode}`, color: n.fail || t.error ? LM.C.red : LM.C.green }]] }); },
                },
            ],
            notes: [
                '试试：原文写 <code>n -= k;</code>（文件里有 3 处 → 报错，要多带上下文）；原文写 <code>const room = max - slot.count+1;</code>（少了空格 → 找不到）；取消「已经 Read 过」（→ 先读后改的报错）；新文本写 <code>const room = max - slot.count - 1;</code>（可见测试全过，第 09 步会揭穿它）；新文本写 <code>const room = 0;</code>（死循环：测试运行器在 200 万次循环后终止它）。',
                '这些报错是写给模型看的：要说清楚错在哪、下一步怎么做（「多带一些上下文」「先 Read」）。好的报错能让模型下一轮自己修正；只写一个 failed 的报错，会让它反复重试同一个错误。Anthropic 的工程博客说，他们给 SWE-bench 做 Agent 时，花在优化工具上的时间比优化整体提示词还多。',
            ],
            warn: '为什么要求唯一？因为 Harness 不能替模型猜它想改哪一处。为什么要求先读？防止模型凭记忆或想象改文件，尤其是文件在会话中途被用户或别的工具改过的时候（生产版还会检查读过之后文件有没有变）。',
        },
        {
            id: 'shell', title: 'Bash：在改过的仓库里跑任意命令', tag: '执行',
            desc: '改完要跑测试、看 diff。这里是一个迷你 shell，运行在<strong>第 05 步改过的仓库</strong>上（每次都从干净的仓库重新应用你的修改）。它支持 ls、cat、head、tail、grep / rg、find、wc、git status / diff / log、npm test 和 node，以及管道 <code>|</code>、<code>&&</code>、<code>;</code>、重定向 <code>></code> 和通配符。每条命令先做<strong>命令分级</strong>（只读 / 会执行或修改 / 拒绝名单），再由当前的<strong>权限模式</strong>和放行规则决定是直接执行、先问用户，还是拒绝。',
            formula: ['class(cmd) = max_{每段管道的第一个词} level(词)       // read < exec < deny；不认识的词算 exec', 'decision = f(工具类型, class(cmd), 模式, 放行规则)     // allow / ask / deny'],
            code: { title: 'C++ · classifyCommand / decide / toolBash', src: CA_CPP.bashtool },
            previews: [
                {
                    id: 'shellOut', label: '你的命令', sub: '分级与决定在执行之前完成；「问用户」在页面里当作用户点了允许', large: true, size: [720, 520],
                    render: (c, S) => {
                        const H = S.SH, cls = CA.classify(S.cmd || ''), C = LM.C;
                        const head = [[{ t: '$ ' + (S.cmd || ''), color: C.yellow, bold: true }], [{ t: '分级：', color: C.gray }, { t: P20.CLS_ZH[cls.perm], color: P20.CLS_COL[cls.perm] }, { t: `（${cls.why}）   ${P20.MODE_ZH[S.mode]}模式 → `, color: C.gray }, { t: P20.DEC_ZH[H.pol.decision], color: P20.DEC_COL[H.pol.decision], bold: true }, { t: H.r ? `   · 退出码 ${H.r.code} · 输出 ${H.r.tokens} token` : '', color: C.gray }]];
                        P20.term(c, H.r ? (H.pol.decision === 'ask' ? '（Harness 会先问用户；这里当作用户点了「允许」）\n' : '') + (H.r.out || '（没有输出）') : `（没有执行：${H.pol.why}）`, { head, color: (l) => (/^（Harness 会先问/.test(l) ? C.gray : null) });
                    },
                },
                { id: 'permMatrix', label: '同一条命令在四种模式下的决定', sub: '第一行是你的命令；黄色列 = 当前模式；放行规则在顶栏改', size: [720, 330], render: (c, S) => P20.drawPerm(c, S) },
            ],
            notes: [
                '默认命令 <code>git diff && npm test</code> 先显示你的改动，再跑测试。再试试：<code>grep -rn "count" src | wc -l</code>、<code>cat -n src/save.js | head -n 20</code>、<code>npm test 2>&1 | tail -n 3</code>、<code>rm -rf src</code>（拒绝）、<code>npm install lodash</code>（要确认：会联网执行安装脚本）。',
                '放行规则是<strong>按整条命令的前缀</strong>匹配的：默认命令 <code>git diff && npm test</code> 不以 <code>npm test</code> 开头，所以默认模式下仍然要问；把命令改成 <code>npm test</code> 就会直接放行。',
                '输出会计 token，超过 30000 字符就截断（Claude Code 的工具说明里写的也是这个上限）。一条 <code>cat</code> 大文件或者刷屏的构建日志可能一下子吃掉几万 token，截断就是给上下文装的保险丝。',
            ],
            warn: '命令分级只是第一道闸：<code>node -e</code>、<code>python -c</code>，甚至 <code>npm test</code>（测试脚本里什么都能写），本质上都是执行任意代码。真正可靠的边界是<strong>沙箱</strong>（第 10 步）：限制能写哪些目录、能连哪些网络，这样即使放行了，也伤不到工作区以外的东西。',
        },
        {
            id: 'loop', title: '一次完整的修 bug 轨迹', tag: '循环',
            desc: '把前面的工具串进第 17 篇的 Agent 循环：模型每一轮决定调用什么工具，Harness 检查权限、执行，再把结果追加进上下文。下面「模型的决定」是写好的脚本（页面里跑不动一个会写代码的大模型），但<strong>每一次工具调用都真的在虚拟仓库上执行</strong>，输出和 token 都是实算的。这里对照两种策略：先复现、再搜索、只读一个窗口（聪明），以及先把整个项目 cat 进来（笨）。',
            formula: ['context_t = 系统提示 + 工具定义 + 任务 + Σ_{i<t} (调用_i + 结果_i)     // 每一轮都要把这些重新读一遍', '总输入 token = Σ_t context_t      // 早早塞进来的东西，之后每一轮都要再付一次'],
            code: { title: 'C++ · runCodingAgent（模型 → 权限 → 工具 → 上下文）', src: CA_CPP.agentloop },
            previews: [
                { id: 'turnView', label: '展开一次工具调用', sub: '顶栏「看第几步」切换；tool_result 是工具在虚拟仓库上真实执行的输出', large: true, size: [720, 430], render: (c, S) => P20.drawTurn(c, S) },
                { id: 'ctxGrowth', label: '上下文随步数增长', sub: '黄圈 = 当前展开的那一步', size: [360, 250], render: (c, S) => P20.drawGrowth(c, S) },
                { id: 'turnTable', label: '每一步的 token', sub: '调用 = 模型写出的想法和 tool_use；结果 = tool_result', size: [360, 250], render: (c, S) => P20.drawTurnTable(c, S) },
            ],
            notes: '笨办法的第一步就放进了 14.5K token，之后每一轮都要带着它们（累计输入 45,086；真实项目里是几十万 token，直接超出上下文）。聪明的轨迹每一步的上下文都不到 3K（累计输入 13,986）：<strong>先复现</strong>（npm test 拿到准确的报错和行号）→ <strong>定位</strong>（Grep 关键词）→ <strong>读一个窗口</strong>（Read 第 1–40 行）→ <strong>最小修改</strong>（Edit 一行）→ <strong>验证</strong>（再跑测试）→ <strong>自查</strong>（git diff）。这也是编程 Agent 的系统提示里反复强调的工作流。',
            warn: '真实的 Agent 不会每次都这么顺：Edit 的原文写错（第 05 步的报错会让它重读再改）、测试里还有别的失败、一个修改引出新问题。真实轨迹通常有十几到几十轮，第 17 篇的上下文压缩、第 18 篇的步数和花费预算在这里都会用到。',
        },
        {
            id: 'sub', title: '子 Agent：把搜索的脏活放进另一个上下文', tag: '子 Agent',
            desc: '有些子任务要读很多东西、结论却很短，例如「项目里还有没有别的地方会让数量超过上限」。主 Agent 调用 Task 工具启动一个<strong>子 Agent</strong>：它有一个全新的上下文和更少的工具（只读），在里面 Grep、Read，最后只把结论交回来。下面是一次真实执行，子 Agent 的 4 次工具调用都在虚拟仓库上运行，它还找出了第二个同类问题：存档迁移时把 gold 整个塞进了一个 arrow 格子。',
            formula: ['主上下文的增量 = tokens(Task 调用) + tokens(报告)      // 与子 Agent 读了多少无关', '子 Agent 的上下文：用完即弃'],
            code: { title: 'C++ · runSubagent / makeTaskTool', src: CA_CPP.subagent },
            previews: [
                { id: 'subBars', label: '谁的上下文付了这些 token', sub: '同样的 4 次调用：放进子 Agent，还是主 Agent 自己做', large: true, size: [720, 140], render: (c, S) => { const A = S.subA, own = A.log.reduce((s, l) => s + l.t, 0); LM.draw.bars(c, [A.sub / 1000, (A.tCall + A.tReport) / 1000, own / 1000], ['子 Agent 自己的上下文（用完即弃）', '主 Agent 多出来的：Task 调用 + 报告', '如果主 Agent 自己做这 4 次调用'], { horizontal: true, range: [0, (Math.max(A.sub, own) / 1000) * 1.35], colors: [LM.C.gray, LM.C.green, LM.C.red], digits: 2, valueSuffix: 'K token' }); } },
                { id: 'subLog', label: '子 Agent 的执行记录与报告', sub: '高亮的行 = 报告里引用的证据', size: [720, 440], render: (c, S) => P20.drawSub(c, S) },
            ],
            notes: [
                'Anthropic 讲上下文工程的文章提到，子 Agent 可能花几万 token 探索，却只返回 1000–2000 token 的精炼总结。本例规模小，比例是一样的。子 Agent 还可以并行：同时派几个去查不同的目录。',
                '可以去第 05 步把文件切到 <code>src/save.js</code>，自己动手修子 Agent 报告的第二个问题。',
            ],
            warn: '子 Agent 的报告是主 Agent 唯一能看到的东西：报告写错或漏写，主 Agent 无从核对。所以调查类子 Agent 一般只给只读工具，并要求报告里带上文件和行号，主 Agent 需要时自己再 Read 一眼。',
        },
        {
            id: 'verify', title: '测试全过 ≠ 修对了', tag: '评测',
            desc: 'Agent 说「测试通过了」，只证明它没破坏能看到的测试。SWE-bench 的评测方法是：补丁打上去以后，跑一组 Agent 看不到的测试——<strong>FAIL_TO_PASS</strong>（修之前失败、修之后必须通过，证明修好了这个 bug）和 <strong>PASS_TO_PASS</strong>（原来就通过的，一个都不能坏）。下表对五个版本真的跑了 12 个可见测试和 2 个隐藏测试：原始代码、你在第 05 步的改动、正确的一行修复、一个「只治症状」的修复（把数量截到 maxStack，多出来的物品直接丢掉），以及一个改过头的修复（room = max − count − 1）。',
            formula: ['resolved ⇔ 全部 FAIL_TO_PASS 通过 ∧ 全部 PASS_TO_PASS 通过'],
            code: { title: 'C++ · evaluatePatch（FAIL_TO_PASS / PASS_TO_PASS）', src: CA_CPP.verify },
            previews: [
                { id: 'verifyMatrix', label: '可见测试 + 隐藏测试', sub: '紫底 = 隐藏测试；「你的改动」一列跟着第 05 步变', large: true, size: [720, 430], render: (c, S) => P20.drawVerify(c, S) },
            ],
            notes: [
                '截断修复让可见测试全部通过：如果评测只看 Agent 自己跑的测试，它就算修好了；隐藏测试「加物品永远不会丢」把它揪了出来。room − 1 也能通过全部可见测试，但隐藏测试「先填满已有的格子」会失败。',
                'SWE-bench 的任务来自 12 个 Python 开源仓库的真实 issue 和修复 PR（2294 个）；SWE-bench Verified 是人工逐个确认过「题目说清楚了、测试公平」的 500 个子集。',
            ],
            warn: '隐藏测试本身也不完美：它只能检查写测试的人想到的性质。所以生产环境的验证还包括：让 Agent 先写一个复现测试（修之前失败、修之后通过）、人工 review diff、CI 跑全量测试和静态检查。',
        },
        {
            id: 'prod', title: '教学版 → 生产版', tag: '生产版',
            desc: '本页每个环节在真实产品里都有对应：',
            blocks: [
                {
                    type: 'html', html: `<table class="lm-tbl">
                    <tr><th>环节</th><th>本页</th><th>真实系统</th></tr>
                    <tr><td>工具集</td><td>8 个：Read / Write / Edit / Glob / Grep / Bash / Task / TodoWrite</td><td>Claude Code 十几个（另有 WebFetch、WebSearch、NotebookEdit 等）；Codex CLI 以 shell、apply_patch（自己的补丁格式）、update_plan 为主；Anthropic 的 SWE-bench 参考实现只有 Bash 和编辑工具；SWE-agent 专门设计了翻页查看器、搜索和带语法检查的编辑命令（ACI）</td></tr>
                    <tr><td>找代码</td><td>正则 Grep、Glob、Aider 式仓库地图</td><td>Claude Code：模型自己 Glob / Grep，不建索引；Aider：tree-sitter + PageRank 地图（默认 1k token）；Cursor：语义索引 + grep</td></tr>
                    <tr><td>改代码</td><td>精确替换 + 三条规则</td><td>str_replace（Claude）、apply_patch（Codex）、SEARCH/REPLACE 块或 udiff（Aider）、用快速的「应用模型」把草稿合进文件（Cursor）；改完自动跑格式化 / lint（hooks）</td></tr>
                    <tr><td>执行与权限</td><td>迷你 shell、命令分级、4 种模式、前缀放行规则</td><td>真实 shell + 超时 + 后台进程；Claude Code 的权限模式 default / acceptEdits / plan / bypassPermissions 和按工具、按命令前缀的 allow / deny 规则；Codex 的审批策略</td></tr>
                    <tr><td>隔离</td><td>每次从干净的虚拟仓库开始</td><td>沙箱：只能写工作区、网络走白名单（Claude Code 在 macOS 用 Seatbelt、Linux 用 bubblewrap；Codex 有 read-only / workspace-write / danger-full-access 三档）；或者整个跑在容器 / 云端虚拟机里</td></tr>
                    <tr><td>回滚</td><td>检查点（每个文件第一次改之前的内容）</td><td>编辑前自动快照、可以回到任意一轮（Claude Code 的 checkpoint / rewind）；git 本身：每步提交、分支、worktree</td></tr>
                    <tr><td>项目记忆</td><td>下图两个例子</td><td>CLAUDE.md / AGENTS.md / .cursor/rules：构建和测试命令、约定、禁区，会话开始时整段放进上下文；可以分层（用户级、项目级、子目录级）</td></tr>
                    <tr><td>验证</td><td>可见 + 隐藏测试</td><td>先写复现测试；CI 全量测试与静态检查；人工 review；SWE-bench Verified 一类的离线评测</td></tr>
                    <tr><td>游戏项目</td><td>纯文本的 JavaScript</td><td>.uasset / .umap、二进制 .prefab 不能当文本改：用引擎的命令行和脚本接口（<code>UnrealEditor-Cmd -run=pythonscript</code>、<code>Unity -batchmode -executeMethod</code>），或第 21 篇的编辑器 MCP；Perforce 的文件在 <code>p4 edit</code> 之前是只读的</td></tr>
                </table>`,
                },
            ],
            code: { title: 'C++ · rewind / projectMemory（检查点与项目说明文件）', src: CA_CPP.checkpoint },
            previews: [
                { id: 'memo', label: '项目说明文件：Agent 每次会话开始时读到的第一段话', sub: '上：本页的小项目；下：一个虚幻引擎项目（命令是 UE 自带的构建与自动化测试命令）', size: [720, 320], render: (c, S) => P20.term(c, P20.MEMO, { px: 11, head: [[{ t: `CLAUDE.md + AGENTS.md · ${TK.countTokens(S.base.B, P20.MEMO)} token（每次会话都付，所以要短）`, color: LM.C.gray }]], color: (l) => (/^#/.test(l) ? LM.C.cyan : l.trim() ? LM.C.text : null) }) },
            ],
            notes: '本篇回答了「写代码」这一类能力：少数通用工具 + 按需检索 + 快速反馈 + 权限与沙箱。第 21 篇接着回答「各种游戏引擎、各种商用软件」：没有命令行、或者状态在正在运行的编辑器里的时候，怎样用 MCP、脚本接口和 Computer Use 把它们接进来。',
            warn: '本页的 token 数只对这个小仓库成立，但比例关系是普遍的：仓库越大，「先搜再读」相对「全部读进来」的优势越大；项目说明文件、工具定义这些每轮都要付的固定开销，越短越好。',
        },
    ],

    graph: {
        title: '编程 Agent · 一次工具调用在 Harness 里的数据流',
        intro: '节点 = 页面里真实执行的计算（模型除外）。模型每一轮输出 tool_use；Harness 先按权限策略决定是否放行，再交给工具执行；结果追加进上下文（HISTORY），成为下一轮的输入。仓库地图在会话开始时算一次；任务结束后再跑一遍评测（第 09 步），不在这个循环里。',
        common: { title: 'Common · 本系列自带的 C++ 库', open: false, blocks: LM.CPP.blocks(['gpt']) },
        types: { txt: { color: '#56b6c2', label: 'txt · 文件 / 代码' }, call: { color: '#e0a060', label: 'call · 工具调用' }, res: { color: '#5ad07a', label: 'res · 工具结果' }, ctx: { color: '#c678dd', label: 'ctx · 上下文' } },
        nodes: [
            { id: 'repomap', title: 'repoMap', tag: 'CPU', category: 'bake', lifetime: 'param', step: 'map', desc: '定义 / 引用图 + PageRank', dispatch: { kind: 'CPU', name: 'buildGraph · pageRank · fitToBudget', count: '会话开始 / 文件变化时' }, inputs: [{ name: 'Repo', format: '工作区文件', type: 'txt', external: true }], outputs: [{ name: 'RepoMap', format: '≤ 预算 token', type: 'ctx' }], detail: { preview: 'mapText' } },
            { id: 'model', title: '模型（外部）', tag: 'LLM', lifetime: 'history', step: 'loop', desc: '读上下文，输出 tool_use 或最终回答', dispatch: { kind: 'CPU', name: '（页面里用写好的脚本代替）', count: '每轮一次' }, inputs: [{ name: 'RepoMap', format: '地图', type: 'ctx' }, { name: 'Context', format: '系统提示 + 工具定义 + 之前的调用与结果', type: 'ctx', history: true }], outputs: [{ name: 'ToolUse', format: '{ name, input }', type: 'call' }], detail: { preview: 'turnView' } },
            { id: 'gate', title: 'decide（权限）', tag: 'CPU', lifetime: 'frame', step: 'shell', desc: '命令分级 × 模式 × 放行规则', dispatch: { kind: 'CPU', name: 'classifyCommand · decide', count: '每次调用' }, inputs: [{ name: 'ToolUse', format: '调用', type: 'call' }], outputs: [{ name: 'Approved', format: 'allow / ask（用户确认）/ deny', type: 'call' }], detail: { preview: 'permMatrix' } },
            { id: 'tools', title: '工具：Read / Edit / Grep / Bash / Task', tag: 'CPU', lifetime: 'frame', step: 'search', desc: '在虚拟仓库上真实执行', dispatch: { kind: 'CPU', name: 'toolRead · toolEdit · toolGrep · runShell · runSubagent', count: '每次调用' }, inputs: [{ name: 'Approved', format: '调用', type: 'call' }], outputs: [{ name: 'Snippet', format: '带行号的文本', type: 'res' }, { name: 'Diff', format: '统一 diff', type: 'res' }, { name: 'Stdout', format: '≤ 30000 字符', type: 'res' }, { name: 'Report', format: '子 Agent 的结论', type: 'res' }], detail: { preview: 'searchCost' } },
            { id: 'append', title: 'append', tag: 'CPU', lifetime: 'history', step: 'loop', desc: '结果追加进上下文', dispatch: { kind: 'CPU', name: 'ctx.push_back', count: '每次调用' }, inputs: [{ name: 'Snippet', format: '结果', type: 'res' }, { name: 'Diff', format: '结果', type: 'res' }, { name: 'Stdout', format: '结果', type: 'res' }, { name: 'Report', format: '结果', type: 'res' }], outputs: [{ name: 'Context', format: '下一轮模型的输入', type: 'ctx' }], detail: { preview: 'ctxGrowth' } },
        ],
    },

    references: [
        { title: 'SWE-bench: Can Language Models Resolve Real-World GitHub Issues?', author: 'C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, K. Narasimhan', venue: 'ICLR', year: 2024, note: 'FAIL_TO_PASS / PASS_TO_PASS 的评测方法（Step 09）' },
        { title: 'Introducing SWE-bench Verified', author: 'OpenAI', venue: 'Blog', year: 2024, note: '人工确认过的 500 个任务（Step 09）' },
        { title: 'SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering', author: 'J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, O. Press', venue: 'NeurIPS', year: 2024, note: '为 Agent 设计的查看、搜索、编辑命令（Step 03、05、10）' },
        { title: 'Building effective agents', author: 'E. Schluntz, B. Zhang（Anthropic）', venue: 'Engineering blog', year: 2024, note: '附录「为工具做提示工程」：做 SWE-bench Agent 时花在工具上的时间比提示词多（Step 05）' },
        { title: 'Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet', author: 'Anthropic', venue: 'Engineering blog', year: 2025, note: '只有 Bash 和编辑工具的参考实现（引言、Step 10）' },
        { title: 'Effective context engineering for AI agents', author: 'Anthropic', venue: 'Engineering blog', year: 2025, note: '按需检索、项目说明文件、子 Agent 只交回精炼总结（Step 03、08、10）' },
        { title: 'Claude Code: Best practices for agentic coding', author: 'Anthropic', venue: 'Engineering blog', year: 2025, note: 'CLAUDE.md、先探索再计划再写代码、测试驱动（Step 07、10）' },
        { title: 'Claude Code documentation: permissions, sandboxing, checkpointing', author: 'Anthropic', venue: 'docs.anthropic.com', year: 2025, note: '权限模式、allow / deny 规则、沙箱、回滚（Step 06、10）' },
        { title: 'Codex CLI documentation: sandbox and approvals, AGENTS.md', author: 'OpenAI', venue: 'github.com/openai/codex', year: 2025, note: 'apply_patch、三档沙箱、审批策略（Step 10）' },
        { title: 'Building a better repository map with tree sitter', author: 'P. Gauthier', venue: 'aider.chat', year: 2023, note: '仓库地图：tree-sitter 标签 + PageRank + token 预算（Step 04）' },
        { title: 'Executable Code Actions Elicit Better LLM Agents', author: 'X. Wang et al.', venue: 'ICML', year: 2024, note: 'CodeAct：用代码而不是 JSON 作为动作（Step 01）' },
        { title: 'OpenHands: An Open Platform for AI Software Developers as Generalist Agents', author: 'X. Wang et al.', venue: 'ICLR', year: 2025, note: '开源的编程 Agent 平台（Step 10）' },
        { title: 'The PageRank Citation Ranking: Bringing Order to the Web', author: 'L. Page, S. Brin, R. Motwani, T. Winograd', venue: 'Stanford InfoLab Technical Report', year: 1999, note: 'PageRank（Step 04）' },
        { title: 'An O(ND) Difference Algorithm and Its Variations', author: 'E. W. Myers', venue: 'Algorithmica', year: 1986, note: 'git diff 默认的算法；本页用更直白的 LCS 动态规划（Step 05）' },
    ],
});
