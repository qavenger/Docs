// p05_fix2.js — 策略梯度三步改成「多个随机种子 + 平均」：stage 与预览的文本替换（每条必须命中）
const fs = require('fs'), path = require('path');
const f = path.join(__dirname, '..', '05_reinforcement_learning.lab.js');
let s = fs.readFileSync(f, 'utf8'), n = 0;
const rep = (a, b) => { if (!s.includes(a)) throw new Error('missing: ' + a.slice(0, 160)); s = s.replace(a, b); n++; };

// ── stages
rep(`            id: 'pg', lifetime: 'history', lazy: true, run(S) {
                p10Live(S, 'pgSt', (S2) => ({ runs: [P10D.pgInitS(S2, 50, { baseline: false }), P10D.pgInitS(S2, 50, { baseline: true })] }), (S2, st, n) => {
                    for (let k = 0; k < n; k++) {
                        let any = false;
                        st.runs.forEach((r) => { if (r.episodes >= P10D.CAP.pg) return; any = true; r.rets.push(P10.reinforceStep(r)); if (r.episodes % P10D.EVAL.pg === 0) P10D.pgEval(S2, r); });
                        if (!any) break;
                    }
                }, P10D.RATE.pg);
            },`, `            id: 'pg', lifetime: 'history', lazy: true, run(S) {
                p10Live(S, 'pgSt', (S2) => ({ groups: [P10D.pgGroup(S2, '不用基线', LM.C.red, P10D.NSEED.pg, { baseline: false }, P10.reinforceStep), P10D.pgGroup(S2, '基线 V(s)', LM.C.green, P10D.NSEED.pg, { baseline: true }, P10.reinforceStep)] }),
                    (S2, st, n) => P10D.groupsAdvance(S2, st.groups, n, P10D.CAP.pg), P10D.RATE.pg);
            },`);
rep(`            id: 'ac', lifetime: 'history', lazy: true, run(S) {
                p10Live(S, 'acSt', (S2) => ({ runs: [P10D.pgInitS(S2, 55, { baseline: true }), P10D.pgInitS(S2, 55, {})] }), (S2, st, n) => {
                    for (let k = 0; k < n; k++) {
                        let any = false;
                        st.runs.forEach((r, i) => { if (r.episodes >= P10D.CAP.ac) return; any = true; r.rets.push(i ? P10.acEpisode(r) : P10.reinforceStep(r)); if (r.episodes % P10D.EVAL.ac === 0) P10D.pgEval(S2, r); });
                        if (!any) break;
                    }
                }, P10D.RATE.ac);
            },`, `            id: 'ac', lifetime: 'history', lazy: true, run(S) {
                p10Live(S, 'acSt', (S2) => ({ groups: [P10D.pgGroup(S2, 'REINFORCE + 基线', LM.C.blue, P10D.NSEED.ac, { baseline: true }, P10.reinforceStep), P10D.pgGroup(S2, '单步 actor-critic', LM.C.green, P10D.NSEED.ac, {}, P10.acEpisode)] }),
                    (S2, st, n) => P10D.groupsAdvance(S2, st.groups, n, P10D.CAP.ac), P10D.RATE.ac);
            },`);
rep("{ id: 'ppo', lifetime: 'history', lazy: true, run(S) { p10Live(S, 'ppoSt', (S2) => ({ runs: P10D.ppoInitS(S2) }), (S2, st, n) => P10D.ppoAdvance(S2, st.runs, Math.min(n, P10D.CAP.ppo + 1)), P10D.RATE.ppo); } },",
    "{ id: 'ppo', lifetime: 'history', lazy: true, run(S) { p10Live(S, 'ppoSt', (S2) => ({ groups: P10D.ppoInitS(S2) }), (S2, st, n) => P10D.ppoAdvance(S2, st.groups, Math.min(n, P10D.CAP.ppo + 1)), P10D.RATE.ppo); } },");

// ── Step 09 previews
rep(`                        const [a, b] = S.pgSt.runs, xs = (r) => r.ev.map((e) => e.ep);
                        PL.draw.plot(c, { x: [0, P10D.CAP.pg], y: P10D.jRange(S), xLabel: \`回合（第 \${b.episodes} / \${P10D.CAP.pg}）\`, series: [{ data: a.ev.map((e) => e.J), x: xs(a), color: LM.C.red, width: 2, label: '不用基线' }, { data: b.ev.map((e) => e.J), x: xs(b), color: LM.C.green, width: 2, label: '基线 V(s)' }], hmarks: P10D.jMarks(S) });`,
    `                        const [a, b] = S.pgSt.groups;
                        PL.draw.plot(c, { x: [0, P10D.CAP.pg], y: P10D.jRange(S), xLabel: \`回合（第 \${b.runs[0].episodes} / \${P10D.CAP.pg}）\`, series: [...P10D.groupSeries(a, \`不用基线（\${P10D.groupOk(a)}/\${a.runs.length} 成功）\`), ...P10D.groupSeries(b, \`基线 V(s)（\${P10D.groupOk(b)}/\${b.runs.length} 成功）\`)], hmarks: P10D.jMarks(S) });`);
rep("label: 'REINFORCE 的学习曲线：J(θ) = V<sup>π<sub>θ</sub></sup>(S)（精确）', sub: '两次运行用同一个随机种子、同样的学习率 1 · 每回合更新一次 · 每 20 回合评估 · 绿 = 基线 b(s) = V(s)，V 向回报 G 做蒙特卡洛回归',",
    "label: 'REINFORCE 的学习曲线：J(θ) = V<sup>π<sub>θ</sub></sup>(S)（精确）', sub: '每种 8 个随机种子（两种用同一组种子）：细线 = 各种子，粗线 = 平均 · 学习率 1、每回合更新一次 · 绿 = 基线 b(s) = V(s)，V 向回报 G 做蒙特卡洛回归 · 「成功」= J > −0.4',");
rep("label: '带基线那次运行的当前策略 π<sub>θ</sub>(a | s)', sub: '每格 4 根箭头，长度 ∝ 概率 · 颜色 = 学到的基线 V(s)',",
    "label: '带基线的第 1 个种子：当前策略 π<sub>θ</sub>(a | s)', sub: '每格 4 根箭头，长度 ∝ 概率 · 颜色 = 学到的基线 V(s)',");
rep("render: (c, S) => { const b = S.pgSt.runs[1]; P10D.drawGrid(c, S.env, { pi: b.pi, V: b.Vc, R: S.Rv, title: `第 ${b.episodes} 回合`, titleX: 46, sub: `J = ${P10D.sgn(b.ev[b.ev.length - 1].J)}` }); },",
    "render: (c, S) => { const b = S.pgSt.groups[1].runs[0]; P10D.drawGrid(c, S.env, { pi: b.pi, V: b.Vc, R: S.Rv, title: `第 ${b.episodes} 回合`, titleX: 46, sub: `J = ${P10D.sgn(b.ev[b.ev.length - 1].J)}` }); },");
// ── Step 10 previews
rep(`                        const [a, b] = S.acSt.runs, xs = (r) => r.ev.map((e) => e.ep);
                        PL.draw.plot(c, { x: [0, P10D.CAP.ac], y: P10D.jRange(S), xLabel: \`回合（第 \${b.episodes} / \${P10D.CAP.ac}）\`, series: [{ data: a.ev.map((e) => e.J), x: xs(a), color: LM.C.blue, width: 2, label: 'REINFORCE + 基线' }, { data: b.ev.map((e) => e.J), x: xs(b), color: LM.C.green, width: 2, label: '单步 actor-critic' }], hmarks: P10D.jMarks(S) });`,
    `                        const [a, b] = S.acSt.groups;
                        PL.draw.plot(c, { x: [0, P10D.CAP.ac], y: P10D.jRange(S), xLabel: \`回合（第 \${b.runs[0].episodes} / \${P10D.CAP.ac}）\`, series: [...P10D.groupSeries(a, \`REINFORCE + 基线（\${P10D.groupOk(a)}/\${a.runs.length}）\`), ...P10D.groupSeries(b, \`单步 actor-critic（\${P10D.groupOk(b)}/\${b.runs.length}）\`)], hmarks: P10D.jMarks(S) });`);
rep("label: '单步 actor-critic vs REINFORCE + 基线', sub: '同样的学习率 1、同样的 critic / 基线学习率 0.1 · 蓝 = 回合结束才更新 · 绿 = 每走一步就更新 · J 精确评估',",
    "label: '单步 actor-critic vs REINFORCE + 基线', sub: '各 8 个随机种子（细线）与平均（粗线）· 同样的学习率 1、同样的 critic / 基线学习率 0.1 · 蓝 = 回合结束才更新 · 绿 = 每走一步就更新 · 括号 = 成功（J > −0.4）的种子数',");
rep("const r = S.acSt.runs[1], Vt = P10.evalExact(S.env, r.pi, S.gamma), d = P10.discVisits(S.env, r.pi, S.gamma);", "const r = S.acSt.groups[1].runs[0], Vt = P10.evalExact(S.env, r.pi, S.gamma), d = P10.discVisits(S.env, r.pi, S.gamma);");
rep("label: 'critic 学到的 V(s) vs 当前 actor 的真实 V^π(s)', sub: '左 = critic · 右 = 精确策略评估 · 色标相同',", "label: 'critic 学到的 V(s) vs 当前 actor 的真实 V<sup>π</sup>(s)', sub: 'actor-critic 的第 1 个种子 · 左 = critic · 右 = 精确策略评估 · 色标相同',");
// ── Step 11 previews
rep(`                        const runs = S.ppoSt.runs, it = runs[2].ev.length - 1;
                        PL.draw.plot(c, { x: [0, P10D.CAP.ppo * P10D.PPO.M], y: P10D.jRange(S), xLabel: \`回合（第 \${it} / \${P10D.CAP.ppo} 批，每批 \${P10D.PPO.M} 回合）· 学习率 \${S.ppoLR}\`, series: runs.map((r) => ({ data: r.ev.map((e) => e.J), x: r.ev.map((e) => e.ep), color: r.cfg.color, width: r.cfg.color === LM.C.green ? 2.4 : 1.8, label: r.cfg.name })), hmarks: P10D.jMarks(S) });`,
    `                        const G = S.ppoSt.groups, it = G[2].runs[0].ev.length - 1, ser = (g) => { const m = g.runs.map((r) => r.ev), n = Math.min(...m.map((e) => e.length)); const out = g.runs.map((r) => ({ data: r.ev.map((e) => e.J), x: r.ev.map((e) => e.ep), color: g.color + '40', width: 1 })); out.push({ data: Array.from({ length: n }, (_, k) => LM.mean(m.map((e) => e[k].J))), x: m[0].slice(0, n).map((e) => e.ep), color: g.color, width: 2.4, label: g.name }); return out; };
                        PL.draw.plot(c, { x: [0, P10D.CAP.ppo * P10D.PPO.M], y: P10D.jRange(S), xLabel: \`回合（第 \${it} / \${P10D.CAP.ppo} 批，每批 \${P10D.PPO.M} 回合）· 学习率 \${S.ppoLR}\`, series: [...ser(G[0]), ...ser(G[1]), ...ser(G[2])], hmarks: P10D.jMarks(S) });`);
rep("label: '同样多的数据，三种用法', sub: '三条曲线用同一个随机种子、同一个学习率（顶栏）· J 精确评估',", "label: '同样多的数据，三种用法', sub: '每种 4 个随机种子（三种用同一组种子）：细线 = 各种子，粗线 = 平均 · 同一个学习率（顶栏）· J 精确评估',");
rep("const runs = S.ppoSt.runs, nb = 45, lo = 0, hi = 3, hc = P10D.histo((runs[2].ratios || []).map((r) => Math.min(r, hi - 1e-9)), lo, hi, nb), hn = P10D.histo((runs[1].ratios || []).map((r) => Math.min(r, hi - 1e-9)), lo, hi, nb);",
    "const G = S.ppoSt.groups, runs = [G[0].runs[0], G[1].runs[0], G[2].runs[0]], nb = 45, lo = 0, hi = 3, hc = P10D.histo((runs[2].ratios || []).map((r) => Math.min(r, hi - 1e-9)), lo, hi, nb), hn = P10D.histo((runs[1].ratios || []).map((r) => Math.min(r, hi - 1e-9)), lo, hi, nb);");
rep("label: '最近一批样本，更新完以后的概率比 ρ', sub: '绿 = PPO（裁剪）· 红 = 不裁剪 · 竖线 = 1 ± ε · ρ > 3 的堆在最右边',", "label: '最近一批样本，更新完以后的概率比 ρ', sub: '第 1 个种子 · 绿 = PPO（裁剪）· 红 = 不裁剪 · 竖线 = 1 ± ε · ρ > 3 的堆在最右边',");
rep("xLabel: `ρ（第 ${runs[2].ev.length - 1} 批）`,", "xLabel: `ρ（第 ${runs[2].ev.length - 1} 批）`,");
rep(`                        const runs = S.ppoSt.runs;
                        PL.draw.plot(c, { x: [0, P10D.CAP.ppo], y: [1e-4, 20], logY: true, xLabel: \`第几批（第 \${runs[2].ev.length - 1} 批）\`, series: runs.map((r) => ({ data: r.ev.slice(1).map((e) => PL.clamp(e.kl, 1e-4, 20)), x: r.ev.slice(1).map((_, i) => i + 1), color: r.cfg.color, width: 1.5 })) });`,
    `                        const G = S.ppoSt.groups, it = G[2].runs[0].ev.length - 1;
                        PL.draw.plot(c, { x: [0, P10D.CAP.ppo], y: [1e-4, 20], logY: true, xLabel: \`第几批（第 \${it} 批）\`, series: G.map((g) => { const m = g.runs.map((r) => r.ev), n = Math.min(...m.map((e) => e.length)); return { data: Array.from({ length: n - 1 }, (_, k) => PL.clamp(LM.mean(m.map((e) => e[k + 1].kl)), 1e-4, 20)), x: Array.from({ length: n - 1 }, (_, k) => k + 1), color: g.color, width: 1.6 }; }) });`);
rep("label: '每批数据让策略变了多少：近似 KL(π<sub>old</sub> ‖ π<sub>new</sub>)', sub: '对数纵轴 · 颜色同上 · 不裁剪时 KL 动辄 > 1：一批数据就把策略推成了（近乎）确定性的',",
    "label: '每批数据让策略变了多少：近似 KL(π<sub>old</sub> ‖ π<sub>new</sub>)', sub: '4 个种子的平均 · 对数纵轴 · 颜色同上 · 不裁剪时 KL 动辄 > 1：一批数据就把策略推成了（近乎）确定性的',");
// ── Step 12 summary
rep("[S.pgSt ? `REINFORCE+基线 · ${S.pgSt.runs[1].episodes} 回合` : 'REINFORCE（未运行）', S.pgSt ? last(S.pgSt.runs[1].ev) : NaN, C.purple],",
    "[S.pgSt ? `REINFORCE+基线 · ${S.pgSt.groups[1].runs[0].episodes} 回合` : 'REINFORCE（未运行）', S.pgSt ? P10D.groupLast(S.pgSt.groups[1]) : NaN, C.purple],");
rep("[S.acSt ? `actor-critic · ${S.acSt.runs[1].episodes} 回合` : 'actor-critic（未运行）', S.acSt ? last(S.acSt.runs[1].ev) : NaN, C.yellow],",
    "[S.acSt ? `actor-critic · ${S.acSt.groups[1].runs[0].episodes} 回合` : 'actor-critic（未运行）', S.acSt ? P10D.groupLast(S.acSt.groups[1]) : NaN, C.yellow],");
rep("[S.ppoSt ? `PPO · ${S.ppoSt.runs[2].episodes} 回合` : 'PPO（未运行）', S.ppoSt ? last(S.ppoSt.runs[2].ev) : NaN, C.orange],",
    "[S.ppoSt ? `PPO · ${S.ppoSt.groups[2].runs[0].episodes} 回合` : 'PPO（未运行）', S.ppoSt ? LM.mean(S.ppoSt.groups[2].runs.map((r) => r.ev[r.ev.length - 1].J)) : NaN, C.orange],");
rep("label: '本页各方法此刻达到的 J = V<sup>π</sup>(S)（精确）', sub: '读的是各实时训练当前的状态（没滚动到对应步骤 = 还没开始；「⏩ 训练到底」后再看）· 括号里是用掉的经验 · 绿竖线 = V*(S)',",
    "label: '本页各方法此刻达到的 J = V<sup>π</sup>(S)（精确）', sub: '读的是各实时训练当前的状态（没滚动到对应步骤 = 还没开始；「⏩ 训练到底」后再看）· 后面是用掉的经验 · 策略梯度三种取多个种子的平均 · 绿竖线 = V*(S)',");
// ε 也影响 DQN（ε 在 8000 步之后才降到顶栏的值，--deps 的 8 帧扫描看不到）
rep("{ id: 'eps', label: '探索率 ε', options: [0, 0.05, 0.1, 0.2, 0.3], value: 0.1,", "{ id: 'eps', label: '探索率 ε', options: [0, 0.05, 0.1, 0.2, 0.3], value: 0.1, steps: ['dqn'],");
fs.writeFileSync(f, s);
console.log('replacements', n);
