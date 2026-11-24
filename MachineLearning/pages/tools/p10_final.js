// p10_final.js — 用页面自己的代码和种子（S.seed = 20251123、默认参数）把每个实时训练跑到上限，打印正文里引用的数字。
// 用法（在 pages/ 下）：node tools/p10_final.js [key=value ...]   例如 node tools/p10_final.js eps=0.2 alpha=0.5
const { P10, P10D, LM } = require('./p10_load.js');
const S = { seed: 20251123, gamma: 0.95, slip: 0.2, rstep: -0.04, alpha: 0.1, eps: 0.1, ucbC: 2, dqnVar: 'full', ppoLR: 8, ppoClip: 0.2, spf: 1 };
process.argv.slice(2).forEach((kv) => { const [k, v] = kv.split('='); S[k] = isNaN(+v) ? v : +v; });
// 与页面 world / cliffEnv stage 相同
S.env = P10.makeEnv(P10.MAPS.main, { slip: S.slip, rstep: S.rstep });
S.vi = P10.valueIteration(S.env, S.gamma, { keep: 80 });
S.uni = P10.uniformPolicy(S.env); S.Vrand = P10.evalExact(S.env, S.uni, S.gamma); S.Jrand = S.Vrand[S.env.start];
S.cenv = P10.makeEnv(P10.MAPS.cliff, { slip: 0, rstep: -1, rGoal: -1, rTrap: -1, rCliff: -100 });
const t0 = Date.now(), log = (...a) => console.log(...a);
log(`V*(S) = ${S.vi.V[S.env.start].toFixed(4)}  J(random) = ${S.Jrand.toFixed(4)}  route ${P10D.routeOf(S.env, S.vi.pi).len} steps → ${P10D.routeOf(S.env, S.vi.pi).end}`);
const at = (ev, key, x) => { let best = ev[0]; for (const e of ev) if (e[key] <= x) best = e; return best; };
// Q-learning
{ const st = P10D.qlInit(S); P10D.qlEval(S, st); P10D.qlAdvance(S, st, Infinity);
  const ev = st.ev; const first = ev.find((e) => e.J > -0.5);
  log(`QL: first J > −0.5 at ep ${first ? first.ep : '—'}; ` + [100, 200, 500, 1000, 2000, 3000].map((x) => { const e = at(ev, 'ep', x); return `${x}: J ${e.J.toFixed(3)} Jε ${e.Je.toFixed(3)} rms ${e.rms.toFixed(3)} opt ${(100 * e.opt).toFixed(0)}%`; }).join(' | '));
  const tail = ev.slice(-50); log(`    last 50 evals: J mean ${LM.mean(tail.map((e) => e.J)).toFixed(3)} min ${Math.min(...tail.map((e) => e.J)).toFixed(3)} max ${Math.max(...tail.map((e) => e.J)).toFixed(3)}; Jε mean ${LM.mean(tail.map((e) => e.Je)).toFixed(3)}`); }
// cliff
{ const st = P10D.cliffInit(S); P10D.cliffAdvance(S, st, Infinity);
  const last = (m) => LM.mean(st.sum[m].slice(-100)), falls = (m) => LM.mean(st.falls[m].slice(-100));
  log(`cliff (α ${S.alpha}, ε ${S.eps}): last-100 online return SARSA ${last(0).toFixed(1)} QL ${last(1).toFixed(1)}; falls/ep SARSA ${falls(0).toFixed(3)} QL ${falls(1).toFixed(3)}`);
  [0, 1].forEach((m) => { const rows = new Map(); st.runs[m].forEach((r) => { const p = P10.greedyPath(S.cenv, r.Q, 60), ys = p.map((s) => S.cenv.xy(s)[1]); const minY = Math.min(...ys.slice(1, -1)); rows.set(minY, (rows.get(minY) || 0) + 1); }); log(`    ${m ? 'QL' : 'SARSA'} greedy route highest row used (over 50 runs): ${[...rows].sort().map(([y, c]) => `row ${y}: ${c}`).join(', ')}`); }); }
// bandit
{ const st = P10D.banditInit(S); P10D.banditAdvance(S, st, Infinity);
  st.ms.forEach((m, i) => { let r = 0, o = 0; for (let t = 900; t < 1000; t++) { r += st.acc.rew[i][t] / st.runs; o += st.acc.opt[i][t] / st.runs; } log(`bandit ${m.name}: last-100 reward ${(r / 100).toFixed(3)} optimal ${(o).toFixed(1)}%`); }); }
// DQN
{ const st = P10D.dqnInitS(S); P10D.dqnAdvance(S, st, Infinity);
  log(`DQN (${S.dqnVar}): ` + [2500, 5000, 7500, 10000, 15000, 20000, 25000, 30000].map((x) => { const e = at(st.ev, 't', x); return `${x}: J ${e.J.toFixed(3)} rms ${e.rms.toFixed(3)}`; }).join(' | '));
  const first = st.ev.find((e) => e.J > -0.3); log(`    first J > −0.3 at step ${first ? first.t : '—'}; episodes ${st.ep}`); }
// REINFORCE（页面：两组各 NSEED.pg 个种子）
{ const groups = [P10D.pgGroup(S, 'none', '#000000', P10D.NSEED.pg, { baseline: false }, P10.reinforceStep), P10D.pgGroup(S, 'base', '#000000', P10D.NSEED.pg, { baseline: true }, P10.reinforceStep)];
  P10D.groupsAdvance(S, groups, Infinity, P10D.CAP.pg);
  groups.forEach((g) => { const m = P10D.groupMean(g); log(`REINFORCE ${g.name}: ok ${P10D.groupOk(g)}/${g.runs.length} · ` + [500, 1000, 2000, 3000, 4000].map((x) => `${x}: ${at(m, 'ep', x).v.toFixed(3)}`).join(' | ') + `  finals [${g.runs.map((r) => r.ev[r.ev.length - 1].J.toFixed(2)).join(' ')}]`); }); }
// AC（页面：两组各 NSEED.ac 个种子）
{ const groups = [P10D.pgGroup(S, 'REINFORCE+b', '#000000', P10D.NSEED.ac, { baseline: true }, P10.reinforceStep), P10D.pgGroup(S, 'one-step AC', '#000000', P10D.NSEED.ac, {}, P10.acEpisode)];
  P10D.groupsAdvance(S, groups, Infinity, P10D.CAP.ac);
  groups.forEach((g) => { const m = P10D.groupMean(g); log(`AC step ${g.name}: ok ${P10D.groupOk(g)}/${g.runs.length} · ` + [200, 500, 1000, 2000, 4000].map((x) => `${x}: ${at(m, 'ep', x).v.toFixed(3)}`).join(' | ') + `  finals [${g.runs.map((r) => r.ev[r.ev.length - 1].J.toFixed(2)).join(' ')}]`); }); }
// PPO（页面：3 组各 NSEED.ppo 个种子）
{ const groups = P10D.ppoInitS(S); P10D.ppoAdvance(S, groups, P10D.CAP.ppo + 1);
  groups.forEach((g) => { const m = P10D.groupMean(g), kl = P10D.groupMean(g, 'kl').slice(1).map((e) => e.v), cl = P10D.groupMean(g, 'clip').slice(1).map((e) => e.v);
    log(`PPO ${g.name} (lr ${S.ppoLR}): ok ${P10D.groupOk(g)}/${g.runs.length} · ` + [80, 160, 400, 800, 2000].map((x) => `${x}ep: ${at(m, 'ep', x).v.toFixed(3)}`).join(' | ') + `  KL median ${kl.slice().sort((p, q) => p - q)[kl.length >> 1].toFixed(3)} max ${Math.max(...kl).toFixed(2)} meanClip ${(100 * LM.mean(cl)).toFixed(1)}%`); }); }
// 静态分析
{ const v = P10D.pgVariance(S, 600); log(`pgVariance relMSE: none ${v.out.none.relMSE.toFixed(1)} constJ ${v.out.constJ.relMSE.toFixed(1)} Vpi ${v.out.Vpi.relMSE.toFixed(1)}; err@600: none ${v.out.none.err[599].toFixed(2)} Vpi ${v.out.Vpi.err[599].toFixed(3)}`); }
{ const g = P10D.gaeScan(S, 200, 400); log(`GAE snapshot J ${g.J.toFixed(3)} critic ${g.criticRMS.toFixed(3)}: ` + g.rows.map((r) => `λ${r.lam}: b² ${r.bias2.toFixed(2)} var ${r.var.toFixed(1)}`).join(' | ')); }
{ const seq = P10.policyIteration(S.env, S.gamma, Int8Array.from({ length: S.env.N }, (_, s) => (S.env.term[s] ? -1 : 0))); log(`PI: ${seq.length - 1} improvements, V(S) per round ${seq.map((q) => q.V[S.env.start].toFixed(3)).join(' ')}, changed ${seq.map((q) => q.changed).join(',')}`); }
log(`total ${((Date.now() - t0) / 1000).toFixed(1)} s`);
