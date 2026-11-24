// p10_fix1.js — 一次性的文本替换（预览标签 / 量程 / 图例 / 注释），每条替换都必须命中
const fs = require('fs'), path = require('path');
const f = path.join(__dirname, '..', '10_reinforcement_learning.lab.js');
let s = fs.readFileSync(f, 'utf8'), n = 0;
const rep = (a, b) => { if (!s.includes(a)) throw new Error('missing: ' + a.slice(0, 140)); s = s.replace(a, b); n++; };

// ── gridCore：自定义格子颜色
rep("                ctx.fillStyle = o.V ? LM.rgbStr(valColor(o.V[s], R)) : '#161c23';", "                ctx.fillStyle = o.colorOf ? o.colorOf(s) : o.V ? LM.rgbStr(valColor(o.V[s], R)) : '#161c23';");
// ── J 曲线上方留出图例的位置
rep(`        const span = Math.max(0.2, hi - lo);
        return [lo - 0.08 * span, hi + 0.12 * span];`, `        const span = Math.max(0.2, hi - lo);
        return [lo - 0.08 * span, hi + 0.42 * span];   // 上方留出图例的位置`);
// ── 在画好的 plot 上补一行动态说明
rep('    const histo = (vals, lo, hi, nb) =>', `    /** 在画好的 plot 上补一行说明文字（左上角，深色底）：动态数字放这里，不挤 xLabel */
    function note(c, text, o) {
        o = o || {};
        const { ctx } = PL.draw.ctx(c, { keep: true });
        ctx.font = F(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        const x = o.x ?? 56, y = o.y ?? 13, tw = ctx.measureText(text).width;
        ctx.fillStyle = 'rgba(8,8,10,0.85)'; ctx.fillRect(x - 3, y - 2, tw + 6, 15);
        ctx.fillStyle = o.color || C.text; ctx.fillText(text, x, y);
    }
    const histo = (vals, lo, hi, nb) =>`);
rep('        jGreedy, jEps, qGap, jRange, jMarks, histo, seedOf,', '        jGreedy, jEps, qGap, jRange, jMarks, histo, note, seedOf,');
// ── Step 02
rep("                'V^π = (I − γ·P_π)^{−1} · r_π      V_{k+1} = r_π + γ·P_π·V_k   // 直接解 / 迭代：‖V_k − V^π‖∞ ≤ γ^k·‖V_0 − V^π‖∞',",
    `                'V^π = (I − γ·P_π)^{−1} · r_π   // 直接解（高斯消元）',
                'V_{k+1} = r_π + γ·P_π·V_k      ‖V_k − V^π‖∞ ≤ γ^k · ‖V_0 − V^π‖∞   // 迭代：γ-收缩',`);
rep("label: '迭代评估的收敛', sub: '对数纵轴 · 蓝 = 离真值的误差 ‖V_k − V^π‖∞ · 灰虚线 = 收缩上界 γ^k·‖V_0 − V^π‖∞',",
    "label: '迭代评估的收敛', sub: '对数纵轴 · 蓝 = 离真值的误差 ‖V<sub>k</sub> − V<sup>π</sup>‖<sub>∞</sub> · 灰虚线 = 收缩上界 γ<sup>k</sup>·‖V<sub>0</sub> − V<sup>π</sup>‖<sub>∞</sub>',");
rep("label: `γ^k·误差₀（γ = ${S.gamma}）` }, { data: S.peErr.map((v) => Math.max(v, 1e-4)), x: xs, color: LM.C.blue, width: 2, label: '‖V_k − V^π‖∞' }]",
    "label: `γᵏ·误差₀（γ = ${S.gamma}）` }, { data: S.peErr.map((v) => Math.max(v, 1e-4)), x: xs, color: LM.C.blue, width: 2, label: '‖Vₖ − V^π‖∞' }]");
rep("PL.draw.plot(c, { x: [0, K], y: [1e-4, Math.max(2, e0 * 1.5)], logY: true, xLabel: '扫描轮数 k'", "PL.draw.plot(c, { x: [0, K], y: [1e-4, Math.max(30, e0 * 30)], logY: true, xLabel: '扫描轮数 k'");
// ── Step 03
rep("label: '价值迭代的收敛', sub: '对数纵轴 · 蓝 = ‖V_k − V*‖∞ · 灰虚线 = γ^k 上界 · 绿线 = 贪心策略第一次全部最优的那一轮',",
    "label: '价值迭代的收敛', sub: '对数纵轴 · 蓝 = ‖V<sub>k</sub> − V*‖<sub>∞</sub> · 灰虚线 = γ<sup>k</sup> 上界 · 绿线 = 贪心策略第一次全部最优的那一轮',");
rep("y: [1e-6, Math.max(2, e0 * 1.5)], logY: true, xLabel: `扫描轮数 k（收敛到 10⁻¹⁰ 共 ${S.vi.iters} 轮）`, series: [{ data: xs.map((k) => e0 * Math.pow(S.gamma, k)), x: xs, color: LM.C.gray, dash: [4, 4], label: 'γ^k·误差₀' }, { data: S.viErr.map((v) => Math.max(v, 1e-6)), x: xs, color: LM.C.blue, width: 2, label: '‖V_k − V*‖∞' }], marks });",
    "y: [1e-6, Math.max(100, e0 * 100)], logY: true, xLabel: `扫描轮数 k（收敛到 10⁻¹⁰ 共 ${S.vi.iters} 轮）`, series: [{ data: xs.map((k) => e0 * Math.pow(S.gamma, k)), x: xs, color: LM.C.gray, dash: [4, 4], label: 'γᵏ·误差₀' }, { data: S.viErr.map((v) => Math.max(v, 1e-6)), x: xs, color: LM.C.blue, width: 2, label: '‖Vₖ − V*‖∞' }], marks });");
rep("id: 'regimes', needs: ['regimes'], size: [360, 330],", "id: 'regimes', needs: ['regimes'], size: [360, 430],");
// ── Step 04
rep("label: '策略迭代的每一轮', sub: '颜色 / 数字 = 精确的 V^{π_k} · 箭头 = π_k · 橙框 = 改进这一步要改的格子',",
    "label: '策略迭代的每一轮', sub: '颜色 / 数字 = 精确的 V<sup>π<sub>k</sub></sup> · 箭头 = π<sub>k</sub> · 橙框 = 改进这一步要改的格子',");
rep("title: `策略迭代 · π_${k}${k === S.piSeq.length - 1 ? '（= π*，不再变化）' : ''}`", "title: `策略迭代 · π${'₀₁₂₃₄₅₆₇₈₉'[k] || '_' + k}${k === S.piSeq.length - 1 ? '（= π*，不再变化）' : ''}`");
rep(`                        const lo = Math.min(...pi, ...vi.slice(0, n), S.vi.V[S.env.start]) - 0.05, hi = S.vi.V[S.env.start] + 0.1;
                        PL.draw.plot(c, { x: [0, n - 1], y: [lo, hi], xLabel: \`轮数（策略迭代 \${S.piSeq.length - 1} 次改进收敛；价值迭代到 10⁻¹⁰ 要 \${S.vi.iters} 轮）\`, series: [{ data: vi.slice(0, n), x: vi.slice(0, n).map((_, i) => i), color: LM.C.blue, width: 2, label: '价值迭代 V_k(S)' }, { data: pi, x: pi.map((_, i) => i), color: LM.C.orange, width: 2.5, label: '策略迭代 V^{π_k}(S)' }], hmarks: [{ y: S.vi.V[S.env.start], color: LM.C.green }] });`,
    `                        const lo = Math.min(...pi, ...vi.slice(0, n), S.vi.V[S.env.start]) - 0.05, top = Math.max(0, ...vi.slice(0, n), S.vi.V[S.env.start]), hi = top + 0.45 * (top - lo);
                        PL.draw.plot(c, { x: [0, n - 1], y: [lo, hi], xLabel: \`轮数（策略迭代 \${S.piSeq.length - 1} 次改进；价值迭代收敛要 \${S.vi.iters} 轮）\`, series: [{ data: vi.slice(0, n), x: vi.slice(0, n).map((_, i) => i), color: LM.C.blue, width: 2, label: '价值迭代 Vₖ(S)' }, { data: pi, x: pi.map((_, i) => i), color: LM.C.orange, width: 2.5, label: '策略迭代 V^πₖ(S)' }], hmarks: [{ y: S.vi.V[S.env.start], color: LM.C.green, label: \`V*(S) = \${P10D.sgn(S.vi.V[S.env.start])}\` }] });`);
// ── Step 05
rep("PL.draw.plot(c, { x: [0, P10D.CAP.ql], y: [0, 1.05], xLabel: '回合', series: [{ data: ev.map((e) => Math.min(1, e.rms)), x: xs, color: LM.C.purple, width: 2, label: 'RMS 误差' }, { data: ev.map((e) => e.opt), x: xs, color: LM.C.green, width: 2, label: '最优动作占比' }] });",
    "PL.draw.plot(c, { x: [0, P10D.CAP.ql], y: [0, 1.5], xLabel: `回合（第 ${S.qlSt.ep} 回合）`, series: [{ data: ev.map((e) => Math.min(1, e.rms)), x: xs, color: LM.C.purple, width: 2, label: 'RMS 误差' }, { data: ev.map((e) => e.opt), x: xs, color: LM.C.green, width: 2, label: '最优动作占比' }], hmarks: [{ y: 1, color: '#3a3f46' }] });");
rep(`                        const v = S.qlSt.visits, lv = Float64Array.from(v, (x) => Math.log10(1 + x)), mx = Math.max(1, ...lv);
                        P10D.drawGrid(c, S.env, { V: Float64Array.from(lv, (x) => 2 * x / mx - 1), R: 1, title:`,
    `                        const v = S.qlSt.visits, lv = Float64Array.from(v, (x) => Math.log10(1 + x)), mx = Math.max(1, ...lv), rgb = [0, 0, 0];
                        P10D.drawGrid(c, S.env, { colorOf: (s) => LM.rgbStr(PL.cmap.viridis(0.08 + 0.92 * lv[s] / mx, rgb)), title:`);
rep("label: '每个格子被访问了多少次', sub: '对数色标 · 探索得少的格子，Q 也学得差（对照上图三角形的颜色）',", "label: '每个格子被访问了多少次', sub: '对数色标（黄 = 多、紫 = 少）· 探索得少的格子，Q 也学得差（对照上图三角形的颜色）',");
rep("series: [{ data: ev.map((e) => e.Je), x: xs, color: LM.C.orange, width: 1.6, label: 'ε-greedy' }, { data: ev.map((e) => e.J), x: xs, color: LM.C.blue, width: 2, label: '贪心' }]",
    "series: [{ data: ev.map((e) => e.Je), x: xs, color: LM.C.orange, width: 1.6, label: 'ε-greedy 策略' }, { data: ev.map((e) => e.J), x: xs, color: LM.C.blue, width: 2, label: '贪心策略' }]");
// ── Step 06
rep('PL.draw.plot(c, { x: [1, P10D.CAP.cliff], y: [-110, 0], xLabel:', 'PL.draw.plot(c, { x: [1, P10D.CAP.cliff], y: [-110, 12], xLabel:');
// ── Step 07：图例放到 sub，动态数字放 note
rep("label: '平均奖励 vs 步数', sub: '对 n 次独立实验求平均（每 5 步一个点）· 每次实验是一组新的 q*',",
    "label: '平均奖励 vs 步数', sub: '对 n 次独立实验求平均（每 5 步一个点，每次实验一组新的 q*）· <span style=\"color:#8a8f98\">灰 = 贪心</span> · <span style=\"color:#6ab0e0\">蓝 = ε-greedy（顶栏 ε；选 0 时用 0.1）</span> · <span style=\"color:#5ad07a\">绿 = UCB（顶栏 c）</span> · <span style=\"color:#e0a060\">橙 = 乐观初值 Q₁ = 5 的贪心</span>',");
rep(`series.push({ data: d, x: xs, color: m.color, width: 1.8, label: m.name }); });
                        PL.draw.plot(c, { x: [0, T], y: [-0.2, 1.75], xLabel: \`步数 t（已平均 \${st.runs} / \${P10D.CAP.bandit} 次实验）\`, series, hmarks: [{ y: 1.54, color: '#3a3f46', label: 'E[max q*] ≈ 1.54' }] });`,
    `series.push({ data: d, x: xs, color: m.color, width: 1.8 }); });
                        PL.draw.plot(c, { x: [0, T], y: [-0.2, 1.75], xLabel: \`步数 t（已平均 \${st.runs} / \${P10D.CAP.bandit} 次实验）\`, series, hmarks: [{ y: 1.54, color: '#3a3f46' }] });
                        P10D.note(c, \`最后 100 步平均：\${st.ms.map((m, i) => { let z = 0; for (let t = T - 100; t < T; t++) z += st.acc.rew[i][t]; return (z / 100 / n).toFixed(2); }).join(' / ')} · 虚线 = E[max q*] ≈ 1.54\`, { color: LM.C.gray });`);
rep("label: '选中最优臂的比例', sub: '同一批实验',", "label: '选中最优臂的比例', sub: '同一批实验，颜色同上',");
rep("return { data: d, x: xs, color: m.color, width: 1.8, label: m.name }; }) });",
    "return { data: d, x: xs, color: m.color, width: 1.8 }; }) });\n                        P10D.note(c, `最后 100 步选对：${st.ms.map((m, i) => { let z = 0; for (let t = T - 100; t < T; t++) z += st.acc.opt[i][t]; return (z / n).toFixed(0) + '%'; }).join(' / ')}`, { color: LM.C.gray });");
// ── Step 08
rep("PL.draw.plot(c, { x: [0, P10D.CAP.dqn], y: P10D.jRange(S), xLabel: '环境步数（= 梯度更新次数）', series: [{ data: ev.map((e) => e.J), x: xs, color: LM.C.blue, width: 2, label: 'DQN 贪心策略' }], hmarks: P10D.jMarks(S), marks: [{ x: S.dqnSt.t, color: '#5a5f66' }] });",
    "PL.draw.plot(c, { x: [0, P10D.CAP.dqn], y: P10D.jRange(S), xLabel: `环境步数（第 ${S.dqnSt.t.toLocaleString()} 步 · 每步一次梯度更新）`, series: [{ data: ev.map((e) => e.J), x: xs, color: LM.C.blue, width: 2, label: 'DQN 的贪心策略' }], hmarks: P10D.jMarks(S), marks: [{ x: S.dqnSt.t, color: '#5a5f66' }] });");
rep("PL.draw.plot(c, { x: [0, P10D.CAP.dqn], y: [1e-4, 2], logY: true, xLabel: '环境步数',", "PL.draw.plot(c, { x: [0, P10D.CAP.dqn], y: [1e-4, 30], logY: true, xLabel: `环境步数（第 ${st.t.toLocaleString()} 步）`,");
rep("label: 'TD 损失与价值误差', sub: '对数纵轴 · 橙 = Huber TD 损失（滑动平均）· 紫 = RMS(V_θ − V*)',", "label: 'TD 损失与价值误差', sub: '对数纵轴 · 橙 = Huber TD 损失（滑动平均）· 紫 = RMS(V<sub>θ</sub> − V*)',");
rep("label: '网络学到的 V_θ(s) = max_a Q_θ(s, a) 与贪心策略', sub: '色标同 Step 03 · 黄线 = 当前这个回合走过的路',",
    "label: '网络学到的 V<sub>θ</sub>(s) = max<sub>a</sub> Q<sub>θ</sub>(s, a) 与贪心策略', sub: '色标同 Step 03 · 黄线 = 当前这个回合走过的路 · 悬停看 4 个 Q 值与 Q*',");
rep("label: '泛化：只在一个格子上更新，别的格子也变了', sub: '把当前网络拷一份，只把 Q(s₀, a₀) 往上推一步梯度（黄框 = s₀，<b>点击换格子</b>），看所有格子的 Q(·, a₀) 改变了多少 · 表格方法只会改黄框那一格',",
    "label: '泛化：只在一个格子上更新，别的格子也变了', sub: '把当前网络拷一份，只对 Q(s₀, a₀) 做一步梯度上推（黄框 = s₀，<b>点击换格子</b>）· 数字 = ΔQ(s, a₀) / ΔQ(s₀, a₀)：别的格子跟着动了多少 · 表格方法里除黄框外全是 0',");
rep(`                        let R = 1e-6; for (let s = 0; s < S.env.N; s++) R = Math.max(R, Math.abs(g.dQ[s]));
                        S._lGen = P10D.drawGrid(c, S.env, { V: g.dQ, R, numbers: false, hi: g.s0, title: \`a₀ = \${P10.ARROW[g.a0]}（s₀ 的贪心动作）\`, titleX: 46, sub: \`s₀ 自己 +\${g.dQ[g.s0].toFixed(3)} · 最远处也有 ±\${(R * 0.5).toFixed(3)} 量级\` });`,
    `                        const self = g.dQ[g.s0] || 1e-12, ratio = Float64Array.from(g.dQ, (v) => v / self);
                        S._lGen = P10D.drawGrid(c, S.env, { V: ratio, R: 1.5, numbers: true, hi: g.s0, title: \`第 \${S.dqnSt.t.toLocaleString()} 步的网络 · a₀ = \${P10.ARROW[g.a0]}\`, titleX: 46, sub: \`ΔQ(s₀, a₀) = \${self.toExponential(1)}\` });`);
// ── Step 09
rep("label: 'REINFORCE 的学习曲线：J(θ) = V^{π_θ}(S)（精确）', sub: '两次运行用同一个随机种子、同样的学习率 1 · 每回合更新一次 · 每 20 回合评估',",
    "label: 'REINFORCE 的学习曲线：J(θ) = V<sup>π<sub>θ</sub></sup>(S)（精确）', sub: '两次运行用同一个随机种子、同样的学习率 1 · 每回合更新一次 · 每 20 回合评估 · 绿 = 基线 b(s) = V(s)，V 向回报 G 做蒙特卡洛回归',");
rep("{ data: b.ev.map((e) => e.J), x: xs(b), color: LM.C.green, width: 2, label: '基线 b(s) = V(s)（蒙特卡洛学出来的）' }", "{ data: b.ev.map((e) => e.J), x: xs(b), color: LM.C.green, width: 2, label: '基线 V(s)' }");
rep("PL.draw.plot(c, { x: [-1, 1], y: [0, mx * 1.15], xLabel: `cos(ĝ, ∇J) · 相对均方误差 E‖ĝ − ∇J‖²/‖∇J‖²：不用基线 ${v.none.relMSE.toFixed(0)}，基线 V^π ${v.Vpi.relMSE.toFixed(1)}`, series: [{ data: hA, color: LM.C.red, bar: true, label: '不用基线' }, { data: hB, color: LM.C.green, bar: true, label: '基线 V^π(s)' }] });",
    "PL.draw.plot(c, { x: [-1, 1], y: [0, mx * 1.6], xLabel: 'cos(ĝ, ∇J)', series: [{ data: hA, color: LM.C.red, bar: true, label: '不用基线' }, { data: hB, color: LM.C.green, bar: true, label: '基线 V^π(s)' }] });\n                        P10D.note(c, `E‖ĝ − ∇J‖² / ‖∇J‖²：不用基线 ${v.none.relMSE.toFixed(0)} · 基线 V^π ${v.Vpi.relMSE.toFixed(1)}`, { color: LM.C.yellow });");
rep("label: '带基线那次运行的当前策略 π_θ(a | s)', sub:", "label: '带基线那次运行的当前策略 π<sub>θ</sub>(a | s)', sub:");
// ── Step 10
rep("series: [{ data: a.ev.map((e) => e.J), x: xs(a), color: LM.C.blue, width: 2, label: 'REINFORCE + 基线（回合结束才更新）' }, { data: b.ev.map((e) => e.J), x: xs(b), color: LM.C.green, width: 2, label: '单步 actor-critic（每步更新）' }]",
    "series: [{ data: a.ev.map((e) => e.J), x: xs(a), color: LM.C.blue, width: 2, label: 'REINFORCE + 基线' }, { data: b.ev.map((e) => e.J), x: xs(b), color: LM.C.green, width: 2, label: '单步 actor-critic' }]");
rep("label: '单步 actor-critic vs REINFORCE + 基线', sub: '同样的学习率 1、同样的 critic 学习率 0.1 · J 精确评估',", "label: '单步 actor-critic vs REINFORCE + 基线', sub: '同样的学习率 1、同样的 critic / 基线学习率 0.1 · 蓝 = 回合结束才更新 · 绿 = 每走一步就更新 · J 精确评估',");
rep("PL.draw.plot(c, { x: [0, 1], y: [0.01, 2000], logY: true, xLabel: `λ（快照：J = ${P10D.sgn(g.J)}，critic RMS 误差 ${g.criticRMS.toFixed(2)}，每个 λ ${g.K} 个回合）`,", "PL.draw.plot(c, { x: [0, 1], y: [0.01, 1e5], logY: true, xLabel: 'λ',");
rep("{ data: g.rows.map((r) => Math.max(0.01, r.bias2)), x: xs, color: LM.C.purple, width: 2, label: '偏差²' }] });",
    "{ data: g.rows.map((r) => Math.max(0.01, r.bias2)), x: xs, color: LM.C.purple, width: 2, label: '偏差²' }] });\n                        P10D.note(c, `快照 J = ${P10D.sgn(g.J)} · critic RMS 误差 ${g.criticRMS.toFixed(2)} · 每个 λ 采 ${g.K} 回合`, { color: LM.C.gray });");
// ── Step 11
rep("PL.draw.plot(c, { x: [lo, hi], y: [0, Math.max(1, ...hc, ...hn) * 1.15], xLabel: `ρ · 裁剪比例 ${l2 ? (100 * l2.clipFrac).toFixed(1) : '—'}% · 近似 KL：PPO ${l2 ? l2.kl.toFixed(3) : '—'}，不裁剪 ${l1 ? l1.kl.toFixed(3) : '—'}`,",
    "PL.draw.plot(c, { x: [lo, hi], y: [0, Math.max(1, ...hc, ...hn) * 1.6], xLabel: `ρ（第 ${runs[2].ev.length - 1} 批）`,");
rep("marks: e ? [{ x: 1 - e, color: '#8a8f98' }, { x: 1 + e, color: '#8a8f98' }] : [] });",
    "marks: e ? [{ x: 1 - e, color: '#8a8f98' }, { x: 1 + e, color: '#8a8f98' }] : [] });\n                        P10D.note(c, `PPO：裁剪 ${l2 ? (100 * l2.clipFrac).toFixed(1) : '—'}% · KL ${l2 ? l2.kl.toFixed(3) : '—'}　不裁剪：KL ${l1 ? l1.kl.toFixed(3) : '—'}`, { color: LM.C.yellow });");
rep("PL.draw.plot(c, { x: [0, P10D.CAP.ppo], y: [1e-4, 20], logY: true, xLabel: '第几批', series: runs.map((r) => ({ data: r.ev.slice(1).map((e) => PL.clamp(e.kl, 1e-4, 20)), x: r.ev.slice(1).map((_, i) => i + 1), color: r.cfg.color, width: 1.5, label: r.cfg.name })) });",
    "PL.draw.plot(c, { x: [0, P10D.CAP.ppo], y: [1e-4, 20], logY: true, xLabel: `第几批（第 ${runs[2].ev.length - 1} 批）`, series: runs.map((r) => ({ data: r.ev.slice(1).map((e) => PL.clamp(e.kl, 1e-4, 20)), x: r.ev.slice(1).map((_, i) => i + 1), color: r.cfg.color, width: 1.5 })) });");
rep("label: '每批数据让策略变了多少：近似 KL(π_old ‖ π_new)', sub: '对数纵轴 · 不裁剪时 KL 动辄 > 1，策略在一批数据上就塌成了确定性的',",
    "label: '每批数据让策略变了多少：近似 KL(π<sub>old</sub> ‖ π<sub>new</sub>)', sub: '对数纵轴 · 颜色同上 · 不裁剪时 KL 动辄 > 1：一批数据就把策略推成了（近乎）确定性的',");
fs.writeFileSync(f, s);
console.log('replacements', n);
