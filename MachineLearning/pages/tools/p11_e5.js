// 一次性编辑：截图审查后的修正（node tools/p11_e5.js）
const fs = require('fs'), path = require('path');
const file = path.join(__dirname, '..', '11_diffusion.lab.js');
let s = fs.readFileSync(file, 'utf8');
const rep = (a, b, all) => { if (!s.includes(a)) { console.error('missing:', a.slice(0, 100)); process.exit(1); } s = all ? s.split(a).join(b) : s.replace(a, () => b); };

// ── P11V：扫描进度条 ──
rep(`    return { V0, VN, ARM, bgImage, tag, view, panel, multi, armColor, fmt, digitGrid };`, `    /** 逐帧扫描的进度条（画在已经画好的图上）：R = jobRunner */
    function progress(c, R) {
        if (!R || R.done) return;
        const { ctx, w } = PL.draw.ctx(c, { keep: true }), p = R.work / Math.max(1, R.total);
        ctx.fillStyle = 'rgba(232,194,90,0.18)'; ctx.fillRect(0, 0, w, 6); ctx.fillStyle = '#e8c25a'; ctx.fillRect(0, 0, w * p, 6);
        tag(ctx, \`扫描中 \${Math.round(100 * p)}%（\${R.res.length}/\${R.jobs.length}）\`, w - 6, 9, { align: 'right', color: C.yellow });
    }
    return { V0, VN, ARM, bgImage, tag, view, panel, multi, armColor, fmt, digitGrid, progress };`);

// ── 扫描：按步数交错排列任务（每帧都有新结果），前几帧预算大一些 ──
rep(`                    const Ks = [2, 5, 10, 20, 50, 100], jobs = [];
                    [{ id: 'ddim', f: P11.epsNet(S.M.vp), sch: P11S.COS, eta: 0, cost: 1 }, { id: 'ddpm', f: P11.epsNet(S.M.vp), sch: P11S.COS, eta: 1, cost: 1 },
                        { id: 'exc', f: P11.epsExact(P11S.COS), sch: P11S.COS, eta: 0, cost: 0.15 }, { id: 'exl', f: P11.epsExact(P11.SCHED.linear), sch: P11.SCHED.linear, eta: 0, cost: 0.15 }]
                        .forEach((se) => Ks.forEach((K) => jobs.push({ kind: 'vp', epsFn: se.f, sch: se.sch, K, eta: se.eta, n: 300, seed: 77, cost: se.cost, sid: se.id, metric: (x, n) => P11.quality(x, n) })));
                    S.sw6 = { epoch: S.epoch, R: P11.jobRunner(jobs), frame: null, Ks };
                }
                if (S.sw6.frame === S.frame) return;
                S.sw6.frame = S.frame; S.sw6.R.step(900);`, `                    const Ks = [2, 5, 10, 20, 50, 100], jobs = [], ser = [{ id: 'ddim', f: P11.epsNet(S.M.vp), sch: P11S.COS, eta: 0, cost: 1 }, { id: 'ddpm', f: P11.epsNet(S.M.vp), sch: P11S.COS, eta: 1, cost: 1 },
                        { id: 'exc', f: P11.epsExact(P11S.COS), sch: P11S.COS, eta: 0, cost: 0.15 }, { id: 'exl', f: P11.epsExact(P11.SCHED.linear), sch: P11.SCHED.linear, eta: 0, cost: 0.15 }];
                    Ks.forEach((K) => ser.forEach((se) => jobs.push({ kind: 'vp', epsFn: se.f, sch: se.sch, K, eta: se.eta, n: 300, seed: 77, cost: se.cost, sid: se.id, metric: (x, n) => P11.quality(x, n) })));
                    S.sw6 = { epoch: S.epoch, R: P11.jobRunner(jobs), frame: null, Ks, runs: 0 };
                }
                if (S.sw6.frame === S.frame) return;
                S.sw6.frame = S.frame; S.sw6.R.step(S.sw6.runs++ < 4 ? 3000 : 900);`);
rep(`                    S.sw7 = { epoch: S.epoch, cls: c, gs, R: P11.jobRunner(jobs), frame: null };
                }
                if (S.sw7.frame === S.frame) return;
                S.sw7.frame = S.frame; S.sw7.R.step(900);`, `                    S.sw7 = { epoch: S.epoch, cls: c, gs, R: P11.jobRunner(jobs), frame: null, runs: 0 };
                }
                if (S.sw7.frame === S.frame) return;
                S.sw7.frame = S.frame; S.sw7.R.step(S.sw7.runs++ < 4 ? 13000 : 900);   // 前几帧每帧做完一个 s，之后每帧 900`);
rep(`                    const Ks = [1, 2, 3, 4, 8, 16, 32], jobs = [], M = S.M;
                    Ks.forEach((K) => jobs.push({ kind: 'vp', epsFn: P11.epsNet(M.vp), sch: P11S.COS, K, eta: 0, n: 300, seed: 77, sid: 'vp', metric: (x, n) => P11.quality(x, n) }));
                    Ks.forEach((K) => jobs.push({ kind: 'fm', net: M.fm, K, n: 300, seed: 77, sid: 'fm', metric: (x, n) => P11.quality(x, n) }));
                    Ks.forEach((K) => jobs.push({ kind: 'fm', net: M.rf, K, n: 300, seed: 77, sid: 'rf', metric: (x, n) => P11.quality(x, n) }));
                    S.sw8 = { epoch: S.epoch, R: P11.jobRunner(jobs), frame: null, Ks };
                }
                if (S.sw8.frame === S.frame) return;
                S.sw8.frame = S.frame; S.sw8.R.step(900);`, `                    const Ks = [1, 2, 3, 4, 8, 16, 32], jobs = [], M = S.M;
                    Ks.forEach((K) => {
                        jobs.push({ kind: 'vp', epsFn: P11.epsNet(M.vp), sch: P11S.COS, K, eta: 0, n: 300, seed: 77, sid: 'vp', metric: (x, n) => P11.quality(x, n) });
                        jobs.push({ kind: 'fm', net: M.fm, K, n: 300, seed: 77, sid: 'fm', metric: (x, n) => P11.quality(x, n) });
                        jobs.push({ kind: 'fm', net: M.rf, K, n: 300, seed: 77, sid: 'rf', metric: (x, n) => P11.quality(x, n) });
                    });
                    S.sw8 = { epoch: S.epoch, R: P11.jobRunner(jobs), frame: null, Ks, runs: 0 };
                }
                if (S.sw8.frame === S.frame) return;
                S.sw8.frame = S.frame; S.sw8.R.step(S.sw8.runs++ < 4 ? 4000 : 900);`);
rep(`                    S.sw9 = { epoch: S.epoch, R: P11.jobRunner(jobs), frame: null, gs };
                }
                if (S.sw9.frame === S.frame) return;
                S.sw9.frame = S.frame; S.sw9.R.step(900);`, `                    S.sw9 = { epoch: S.epoch, R: P11.jobRunner(jobs), frame: null, gs, runs: 0 };
                }
                if (S.sw9.frame === S.frame) return;
                S.sw9.frame = S.frame; S.sw9.R.step(S.sw9.runs++ < 4 ? 13000 : 900);   // 前几帧每帧做完一个 s（10 个数字）`);
// 进度标签 → 进度条
rep(`                        if (!W.R.done) { const { ctx, w } = PL.draw.ctx(c, { keep: true }); P11V.tag(ctx, \`扫描中 \${Math.round(100 * W.R.work / W.R.total)}%\`, w - 6, 4, { align: 'right', color: LM.C.yellow }); }`, `                        P11V.progress(c, W.R);`, true);
rep(`PL.draw.plot(c, { x: [0, 6], y: [0.5, 5000], logY: true, xLabel: '引导强度 s', series: [{ data: r.map((q) => Math.max(0.5, q.mmd)), x: r.map((q) => q.job.g), color: LM.C.orange, width: 2, label: 'MMD（到该臂）' }], marks: [{ x: S.guide, color: LM.C.yellow }], hmarks: [{ y: 2.5, color: LM.C.gray, label: '估计噪声' }] }); if (!W.R.done) { const { ctx, w } = PL.draw.ctx(c, { keep: true }); P11V.tag(ctx, \`扫描中 \${Math.round(100 * W.R.work / W.R.total)}%\`, w - 6, 4, { align: 'right', color: LM.C.yellow }); } },`,
    `PL.draw.plot(c, { x: [0, 6], y: [0.5, 5000], logY: true, xLabel: '引导强度 s', series: [{ data: r.map((q) => Math.max(0.5, q.mmd)), x: r.map((q) => q.job.g), color: LM.C.orange, width: 2, label: 'MMD（到该臂）' }], marks: [{ x: S.guide, color: LM.C.yellow }], hmarks: [{ y: 2.5, color: LM.C.gray, label: '估计噪声' }] }); P11V.progress(c, W.R); },`);
rep(`render: (c, S) => { const r = S.sw7.R.res, xs = r.map((q) => q.job.g); PL.draw.plot(c, { x: [0, 6], y: [0, 1.05], xLabel: '引导强度 s', series: [{ data: r.map((q) => q.acc), x: xs, color: LM.C.green, width: 2, label: '准确率' }, { data: r.map((q) => q.cov), x: xs, color: LM.C.blue, width: 2, label: '覆盖率' }], marks: [{ x: S.guide, label: \`s = \${S.guide}\`, color: LM.C.yellow }, { x: 1, label: '条件模型', color: LM.C.gray }] }); },`,
    `render: (c, S) => { const r = S.sw7.R.res, xs = r.map((q) => q.job.g); PL.draw.plot(c, { x: [0, 6], y: [0, 1.05], xLabel: '引导强度 s', series: [{ data: r.map((q) => q.acc), x: xs, color: LM.C.green, width: 2, label: '准确率' }, { data: r.map((q) => q.cov), x: xs, color: LM.C.blue, width: 2, label: '覆盖率' }], marks: [{ x: S.guide, label: \`s = \${S.guide}\`, color: LM.C.yellow }, { x: 1, label: '条件模型', color: LM.C.gray }] }); P11V.progress(c, S.sw7.R); },`);

// ── Step 01：高斯图的标签、刻度表 ──
rep(`P11V.panel(c, { bg: S.bg0, pts: S.baseG, n: 400, color: '#e8c25a', r: 1.3, label: 'N(μ̂, Σ̂) 的 400 个样本', corner: \`MMD \${P11V.fmt(q.mmd)} · 偏离 \${(100 * q.off).toFixed(0)}%\` });`,
    `P11V.panel(c, { bg: S.bg0, pts: S.baseG, n: 400, color: '#e8c25a', r: 1.3, label: 'N(μ̂, Σ̂)', corner: \`MMD \${P11V.fmt(q.mmd)} · 偏离 \${(100 * q.off).toFixed(0)}%\` });`);
rep(`                    render: (c, S) => LM.draw.text(c, [[{ t: '点集'.padEnd(17, '　'), color: LM.C.dim }, { t: 'MMD²×1000   偏离率', color: LM.C.dim }]].concat(S.base.map(([name, q], k) => [
                        { t: name.padEnd(17, '　'), color: k === 0 ? LM.C.green : LM.C.text }, { t: \`\${P11V.fmt(q.mmd).padStart(8)}   \${(100 * q.off).toFixed(1).padStart(5)}%\`, color: k === 0 ? LM.C.green : k < 3 ? LM.C.yellow : LM.C.red },
                    ])), { px: 12, lineGap: 9 }),`, `                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c, { bg: '#06080b' }), x1 = w - 128, x2 = w - 14, row = 29;
                        ctx.font = LM.FONT(11); ctx.textBaseline = 'middle'; ctx.fillStyle = LM.C.dim; ctx.textAlign = 'left'; ctx.fillText('点集（各 400 个点）', 10, 14); ctx.textAlign = 'right'; ctx.fillText('MMD²×1000', x1, 14); ctx.fillText('偏离率', x2, 14);
                        S.base.forEach(([name, q], k) => {
                            const y = 14 + row * (k + 1), col = k === 0 ? LM.C.green : LM.C.text; ctx.font = LM.FONT(12); ctx.fillStyle = col; ctx.textAlign = 'left'; ctx.fillText(name, 10, y);
                            ctx.textAlign = 'right'; ctx.fillStyle = q.mmd < 3 ? LM.C.green : q.mmd < 30 ? LM.C.yellow : LM.C.red; ctx.fillText(P11V.fmt(q.mmd), x1, y);
                            ctx.fillStyle = q.off < 0.03 ? LM.C.green : q.off < 0.2 ? LM.C.yellow : LM.C.red; ctx.fillText(\`\${(100 * q.off).toFixed(1)}%\`, x2, y);
                            ctx.strokeStyle = '#161b21'; ctx.beginPath(); ctx.moveTo(8, y + row / 2); ctx.lineTo(w - 8, y + row / 2); ctx.stroke();
                        });
                    },`);
rep(`size: [360, 190], needs: ['base'],`, `size: [360, 180], needs: ['base'],`);

// ── Step 02：随机游走改成「坐标 vs t」的曲线 + 解析的均值 / ±2σ ──
rep(`walk: P11.forwardChain(S.sch, S.D.X.slice(0, 16), 8, LM.rng(8), 10),`, `walk: P11.forwardChain(S.sch, S.D.X.slice(0, 10), 5, LM.rng(8), 5),`);
rep(`                {
                    id: 'walk', label: '逐步加噪 = 随机游走 + 向原点收缩', sub: '8 个点按 q(x_t | x_{t−1}) 真的走 1000 步（每 10 步画一个点）', size: [240, 240], needs: ['fwd'],
                    render: (c, S) => { const tr = S.film.walk; P11V.panel(c, { v: P11V.VN, bg: S.bgN0, tint: [40, 62, 80], paths: [{ traj: tr, idx: [0, 1, 2, 3, 4, 5, 6, 7], color: (i) => P11V.ARM[S.D.lab[i]], width: 1, alpha: 0.85 }], pts: tr[0], n: 8, color: '#ffffff', r: 2.5, label: '白点 = x₀' }); },
                },`, `                {
                    id: 'walk', label: '逐步加噪 = 向原点收缩 + 越来越大的随机游走', sub: '5 个点按 q(x_t | x_{t−1}) 真的走 1000 步，画第一个坐标 · 虚线 = 解析均值 √ᾱ_t·x₀ · 灰 = 第一个点的 ±2√(1−ᾱ_t)', size: [360, 210], needs: ['fwd'],
                    render: (c, S) => {
                        const tr = S.film.walk, xs = tr.map((_, k) => 5 * k), sch = S.sch, series = [];
                        for (let i = 0; i < 5; i++) { const col = P11V.ARM[S.D.lab[i]], x0 = tr[0][2 * i]; series.push({ data: tr.map((p) => p[2 * i]), x: xs, color: col, width: 1.2 }); series.push({ data: xs.map((t) => sch.a[t] * x0), x: xs, color: col, dash: [4, 3], width: 1 }); }
                        const m0 = tr[0][0]; series.push({ data: xs.map((t) => sch.a[t] * m0 + 2 * sch.s[t]), x: xs, color: LM.C.gray, width: 1 }, { data: xs.map((t) => sch.a[t] * m0 - 2 * sch.s[t]), x: xs, color: LM.C.gray, width: 1 });
                        PL.draw.plot(c, { x: [0, 1000], y: [-2.6, 2.6], xLabel: \`时刻 t（\${S.sched}）\`, series, marks: [{ x: S.tv, color: LM.C.yellow }] });
                    },
                },`);

// ── Step 03：损失图的参考线标签挪到文字里 ──
rep(`PL.draw.plot(c, { x: [0, P11.LIVE.steps], y: [0.28, 0.5], xLabel: '训练步', series: [{ data: sm, x: xs, color: LM.C.blue, width: 2, label: '实时训练' }], hmarks: [{ y: S.floor.total, color: LM.C.gray, label: \`下限 \${S.floor.total.toFixed(3)}\` }, { y: offL, color: LM.C.green, label: \`离线 \${offL.toFixed(3)}\` }] });
                        const { ctx } = PL.draw.ctx(c, { keep: true }); P11V.tag(ctx, \`第 \${R.tr.it} / \${P11.LIVE.steps} 步 · loss \${sm.length ? sm[sm.length - 1].toFixed(3) : '—'}\`, 46, 4, { color: LM.C.yellow });`,
    `PL.draw.plot(c, { x: [0, P11.LIVE.steps], y: [0.28, 0.5], xLabel: '训练步', series: [{ data: sm, x: xs, color: LM.C.blue, width: 2, label: '实时训练' }], hmarks: [{ y: S.floor.total, color: LM.C.gray }, { y: offL, color: LM.C.green }] });
                        const { ctx } = PL.draw.ctx(c, { keep: true }); P11V.tag(ctx, \`第 \${R.tr.it} / \${P11.LIVE.steps} 步 · loss \${sm.length ? sm[sm.length - 1].toFixed(3) : '—'}\`, 46, 4, { color: LM.C.yellow });
                        P11V.tag(ctx, \`灰 = 下限 \${S.floor.total.toFixed(3)} · 绿 = 离线 \${offL.toFixed(3)}\`, 46, 22, { color: LM.C.gray });`);

// ── 默认查看时刻 ──
rep(`{ id: 'tv', label: '查看时刻 t', min: 0, max: 1000, step: 10, value: 300,`, `{ id: 'tv', label: '查看时刻 t', min: 0, max: 1000, step: 10, value: 120,`);

// ── Step 04：网络的箭头在下、解析的细白线在上 ──
rep(`arrows: [{ P: s.P, V: neg(s.eX), n: s.n, color: 'rgba(255,255,255,0.75)', scale: 0.09, width: 2.2, maxLen: 18, head: false }, { P: s.P, V: neg(s.eN), n: s.n, color: '#e8c25a', scale: 0.09, width: 1.2, maxLen: 18 }]`,
    `arrows: [{ P: s.P, V: neg(s.eN), n: s.n, color: '#e8c25a', scale: 0.09, width: 2.6, maxLen: 18 }, { P: s.P, V: neg(s.eX), n: s.n, color: '#ffffff', scale: 0.09, width: 1, maxLen: 18, head: false }]`);
rep(`sub: '黄 = 网络 · 白 = 解析答案 · 背景 = q_t · 拖 <b>查看时刻 t</b>（离线模型用 cosine 日程）'`, `sub: '黄箭头 = 网络 · 白细线 = 解析答案（重合时白线落在黄箭头中间）· 背景 = q_t · 拖 <b>查看时刻 t</b>（离线模型用 cosine 日程）'`);

// ── Step 05：尾迹只画 80 个粒子；双胞胎改成 3 个起点 × 4 次；小图让开 LIVE 徽标 ──
rep(`paths: [{ traj: d.traj, idx: Array.from({ length: d.n }, (_, i) => i), from: k - 4, upto: k, color: (i) => P11V.ARM[d.arm[i]], alpha: 0.4 }], pts: d.traj[k], n: d.n, colorOf: (i) => P11V.ARM[d.arm[i]], r: 1.5, label: \`DDPM · K = \${d.K}\``,
    `paths: [{ traj: d.traj, idx: Array.from({ length: 80 }, (_, i) => i), from: k - 3, upto: k, color: (i) => P11V.ARM[d.arm[i]], alpha: 0.3 }], pts: d.traj[k], n: d.n, colorOf: (i) => P11V.ARM[d.arm[i]], r: 1.5, label: \`DDPM · K = \${d.K}\``);
rep(`r: 1.4, alpha: x0 ? 0.85 : 0.15, label: x0 ? \`x̂₀（t = \${d.tau[d.K - k + 1]}）\` : 'x̂₀（尚未开始）' });`, `r: 1.4, alpha: x0 ? 0.85 : 0.15, labelX: 46, label: x0 ? \`x̂₀（t = \${d.tau[d.K - k + 1]}）\` : 'x̂₀（尚未开始）' });`);
rep(`                    const tw = new Float32Array(36); for (let r = 0; r < 3; r++) for (let i = 0; i < 6; i++) { tw[2 * (r * 6 + i)] = S.noise[2 * (40 + i)]; tw[2 * (r * 6 + i) + 1] = S.noise[2 * (40 + i) + 1]; }
                    const twin = P11.sampleVP(P11.epsNet(S.M.vp), P11S.COS, tw, 18, { K: S.K, eta: 1, rand: LM.rng(99), keep: true });
                    return { K: S.K, n, traj: res.traj, x0s: res.x0s, tau: res.tau, arm: P11.armOf(res.x, n), q: P11.quality(res.x, n), twin: twin.traj, twArm: P11.armOf(twin.x, 18) };`,
    `                    const tw = new Float32Array(24); for (let r = 0; r < 4; r++) for (let i = 0; i < 3; i++) { tw[2 * (r * 3 + i)] = S.noise[2 * (40 + i)]; tw[2 * (r * 3 + i) + 1] = S.noise[2 * (40 + i) + 1]; }
                    const twin = P11.sampleVP(P11.epsNet(S.M.vp), P11S.COS, tw, 12, { K: S.K, eta: 1, rand: LM.rng(99), keep: true });
                    return { K: S.K, n, traj: res.traj, x0s: res.x0s, tau: res.tau, arm: P11.armOf(res.x, n), q: P11.quality(res.x, n), twin: twin.traj, twArm: P11.armOf(twin.x, 12) };`);
rep(`                    id: 'ddpmTwin', label: '同一个 x_T，跑 3 次', sub: '6 个起点（白圈），每个起点只换注入的噪声 z 跑 3 次 · 同色 = 同一起点', size: [240, 240], needs: ['ddpm'],
                    render: (c, S) => { const d = S.dp, cols = ['#e06c75', '#e8c25a', '#56b6c2', '#c678dd', '#5ad07a', '#d19a66']; P11V.panel(c, { v: P11V.VN, bg: S.bgN0, tint: [40, 62, 80], paths: [{ traj: d.twin, idx: Array.from({ length: 18 }, (_, i) => i), color: (i) => cols[i % 6], alpha: 0.75, width: 1 }], pts: d.twin[d.twin.length - 1], n: 18, colorOf: (i) => cols[i % 6], r: 3, extra: (ctx, X, Y) => { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; for (let i = 0; i < 6; i++) { ctx.beginPath(); ctx.arc(X(d.twin[0][2 * i]), Y(d.twin[0][2 * i + 1]), 4, 0, 7); ctx.stroke(); } } }); },`,
    `                    id: 'ddpmTwin', label: '同一个 x_T，跑 4 次', sub: '3 个起点（白圈），每个起点只换注入的噪声 z 再跑 4 次 · 同色 = 同一起点 · 大点 = 终点', size: [240, 240], needs: ['ddpm'],
                    render: (c, S) => { const d = S.dp, cols = ['#e06c75', '#e8c25a', '#56b6c2']; P11V.panel(c, { v: P11V.VN, bg: S.bgN0, tint: [40, 62, 80], paths: [{ traj: d.twin, idx: Array.from({ length: 12 }, (_, i) => i), color: (i) => cols[i % 3], alpha: 0.45, width: 1 }], pts: d.twin[d.twin.length - 1], n: 12, colorOf: (i) => cols[i % 3], r: 3.5, extra: (ctx, X, Y) => { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.arc(X(d.twin[0][2 * i]), Y(d.twin[0][2 * i + 1]), 5, 0, 7); ctx.stroke(); } } }); },`);

// ── Step 06：偏离率扫描（少步数时并不偏离，而是缩成一团）换成平均半径 ──
rep(`                    id: 'sweep6off', label: '偏离率 vs 采样步数', sub: '同一个扫描 · 偏离率（落在真实密度最低 1% 区域的比例）', size: [360, 180], live: true, needs: ['sw6'],
                    render: (c, S) => {
                        const res = S.sw6.R.res, sid = { ddim: [LM.C.blue, null], ddpm: [LM.C.orange, null], exc: ['#ffffff', [5, 4]], exl: [LM.C.purple, [5, 4]] };
                        const series = Object.keys(sid).map((k) => { const r = res.filter((q) => q.job.sid === k); return { data: r.map((q) => 100 * q.off), x: r.map((q) => q.job.K), color: sid[k][0], dash: sid[k][1], width: 2 }; }).filter((s) => s.data.length);
                        PL.draw.plot(c, { x: [2, 100], logX: true, y: [0, 60], xLabel: '采样步数 K', series, hmarks: [{ y: 1, color: LM.C.gray, label: '真实样本 1%' }] });
                    },`, `                    id: 'sweep6off', label: '少步数时样本缩向中心：平均半径 vs K', sub: '同一个扫描 · 样本到原点的平均距离 · 灰 = 真实数据', size: [360, 180], live: true, needs: ['sw6'],
                    render: (c, S) => {
                        const res = S.sw6.R.res, sid = { ddim: [LM.C.blue, null], ddpm: [LM.C.orange, null], exc: ['#ffffff', [5, 4]], exl: [LM.C.purple, [5, 4]] };
                        const series = Object.keys(sid).map((k) => { const r = res.filter((q) => q.job.sid === k); return { data: r.map((q) => q.r), x: r.map((q) => q.job.K), color: sid[k][0], dash: sid[k][1], width: 2 }; }).filter((s) => s.data.length);
                        PL.draw.plot(c, { x: [2, 100], logX: true, y: [0, 0.9], xLabel: '采样步数 K', series, hmarks: [{ y: P11.DATA_R, color: LM.C.gray, label: \`数据 \${P11.DATA_R.toFixed(2)}\` }] });
                        P11V.progress(c, S.sw6.R);
                    },`);

// ── Step 07：引导方向在 t = 250；小图高一点 ──
rep(`const t = 400, sch = P11S.COS, a = sch.a[t], s2 = 1 - sch.ab[t], G = 15, v = 1.6, n = G * G, net = S.M.cfg, P = new Float32Array(2 * n);`, `const t = 250, sch = P11S.COS, a = sch.a[t], s2 = 1 - sch.ab[t], G = 15, v = 1.6, n = G * G, net = S.M.cfg, P = new Float32Array(2 * n);`);
rep(`sub: 't = 400 · 背景 = 解析的 p(c | x_t)（越亮越确定是这一类）· 箭头 ∝ ∇log p(c | x_t)'`, `sub: 't = 250 · 背景 = 解析的 p(c | x_t)（越亮越确定是这一类）· 箭头 ∝ ∇log p(c | x_t)'`);
rep(`sub: 's = 0（无条件）· 1（条件）· 2 · 4（冲出数据范围）', size: [360, 110], needs: ['cfgS'],`, `sub: 's = 0（无条件）· 1（条件）· 2 · 4（冲出数据范围）', size: [360, 120], needs: ['cfgS'],`);

// ── Step 09 / 10：尺寸与小数 ──
rep(`sub: '上 = 测试集原图 · 下 = 解码结果（重建模糊一些：8 个数要装下整个数字）', size: [360, 110], needs: ['lat0'],`, `sub: '上 = 测试集原图 · 下 = 解码结果（重建模糊一些：8 个数要装下整个数字）', size: [360, 92], needs: ['lat0'],`);
rep(`{ t: \`ᾱ_T = \${sch.ab[1000].toExponential(2)}   √ᾱ_T = \${Math.sqrt(sch.ab[1000]).toFixed(4)}\`, color: LM.C.text }`, `{ t: \`ᾱ_T = \${sch.ab[1000].toExponential(2)}   √ᾱ_T = \${Math.sqrt(sch.ab[1000]) < 1e-3 ? Math.sqrt(sch.ab[1000]).toExponential(2) : Math.sqrt(sch.ab[1000]).toFixed(4)}\`, color: LM.C.text }`);
rep(`id: 'termSNR', label: '终点（t = T）的信噪比', sub: '理想的终点应该是纯噪声（ᾱ_T = 0）', size: [360, 150],`, `id: 'termSNR', label: '终点（t = T）的信噪比', sub: '理想的终点应该是纯噪声（ᾱ_T = 0）', size: [360, 126],`);

// ── 图的说明 ──
rep(`intro: '上半部分是 Step 03 的一个训练步（实时运行，<b>HISTORY</b>：参数跨帧持久），下半部分是采样器（DDPM / DDIM / CFG 都是同一个循环，只是每步的公式不同）。节点 = 一次真实的计算，pin = 张量名 · 形状；同名资源自动连线。',`,
    `intro: '左边六个节点是 Step 03 的一个训练步（实时运行，<b>HISTORY</b>：参数跨帧持久，虚线回边），右边是采样器与评估（DDPM / DDIM / CFG 是同一个循环，只是每步的公式不同）。节点 = 一次真实的计算，pin = 张量名 · 形状；同名资源自动连线。',`);

// ── P11：数据的平均半径（Step 06 的参考线） ──
rep(`    const NTRAIN = 4096, DATA_SEED = 1234;`, `    const NTRAIN = 4096, DATA_SEED = 1234;
    const DATA_R = (() => { const X = sampleGMMRaw(3000, LM.rng(5)); let s = 0; for (let i = 0; i < 3000; i++) s += Math.hypot(X[2 * i], X[2 * i + 1]); return s / 3000; })();   // 真实数据到原点的平均距离`);
rep(`    /** 从混合分布采样 n 个点（cls ≥ 0 时只从该臂采样）。X: Float32Array(2n)，lab: Int32Array(n) 记录所属的臂 */`, `    function sampleGMMRaw(n, rand) { return sampleGMM(n, rand).X; }
    /** 从混合分布采样 n 个点（cls ≥ 0 时只从该臂采样）。X: Float32Array(2n)，lab: Int32Array(n) 记录所属的臂 */`);
rep(`T, ARMS, SIG, GMM, NTRAIN, trainSet, digitSets, sampleGMM,`, `T, ARMS, SIG, GMM, NTRAIN, DATA_R, trainSet, digitSets, sampleGMM,`);
fs.writeFileSync(file, s); console.log('ok');
