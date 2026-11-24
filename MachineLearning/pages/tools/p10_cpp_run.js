// p10_cpp_run.js — 把第 10 篇的全部 C++ 代码块（P10CPP，与页面里显示的完全相同）编译成可执行文件，
// 用与页面 JS 相同的种子跑同一组实验，逐项对数。用法（在 pages/ 下）：node tools/p10_cpp_run.js
// 需要 VS 自带的 clang++（与 tools/check_cpp.js 相同）。表格方法应当逐位一致（同一个 mulberry32 随机数流、同样的运算顺序）；
// DQN 的网络在 C++ 里用 float 运算、JS 里用 double 运算再存成 float32，只能近似一致。
const fs = require('fs'), os = require('os'), path = require('path'), { execFileSync } = require('child_process');
const { P10, P10CPP, LM } = require('./p10_load.js');
const CLANG = process.env.CLANGXX || 'C:/Program Files/Microsoft Visual Studio/18/Professional/VC/Tools/Llvm/x64/bin/clang++.exe';
const pages = path.join(__dirname, '..');
const dedent = (s) => { const ls = s.replace(/^\s*\n/, '').replace(/\s+$/, '').split('\n'); const ind = Math.min(...ls.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length)); return ls.map((l) => l.slice(ind)).join('\n'); };
const order = ['common', 'mdp', 'evalPolicy', 'vi', 'pi', 'ql', 'sarsa', 'bandit', 'dqn', 'pg', 'ac', 'ppo', 'puct'];
let tu = '#include "minml_nn.h"\n#include <string>\n#include <cstdio>\n';
order.forEach((k) => { tu += `\n// ═══ ${k} ═══\n${dedent(P10CPP[k])}\n`; });
tu += String.raw`
// ═══ 实验（与 JS 端完全相同的设置与种子） ═══
double jOf(const GridWorld& env, const std::vector<double>& pi, double g) { return evaluatePolicy(env, pi, g)[env.start]; }
std::vector<double> greedyPi(const GridWorld& env, const std::vector<double>& Q) { return onehotPolicy(env, greedyPolicy(env, Q)); }
int main() {
    GridWorld env = makeGridWorld(mainMap(), 0.2, -0.04, 1.0, -1.0, -100.0);
    VIResult vi = valueIteration(env, 0.95, 1e-10);
    std::printf("vi %.12f %d\n", vi.V[env.start], vi.iters);
    std::vector<double> uni = uniformPolicy(env);
    std::printf("vrand %.12f %.12f\n", evaluatePolicy(env, uni, 0.95)[env.start], policyEvaluation(env, uni, 0.95, 60)[env.start]);
    int rounds = 0;
    std::vector<int> act = policyIteration(env, 0.95, ints(env.N, 0), &rounds);
    int diff = 0; for (int s = 0; s < env.N; s++) if (!isTerminal(env, s) && act[s] != vi.pi[s]) diff++;
    std::printf("pi %d %d\n", rounds, diff);
    { Rng r = makeRng(7); std::printf("mc %.12f\n", monteCarloValue(env, uni, env.start, 0.95, 1000, 400, r)); }
    { Rng r = makeRng(11); std::vector<double> Q = doubles(env.N * NA); double sum = 0;
      for (int e = 0; e < 2000; e++) sum += qLearningEpisode(env, Q, 0.1, 0.95, 0.1, 200, r).ret;
      std::printf("ql %.12f %.12f %.12f\n", sum, Q[env.start * NA + 0], jOf(env, greedyPi(env, Q), 0.95)); }
    { GridWorld c = cliffWorld(); double sq = 0, ss = 0;
      Rng r1 = makeRng(101), r2 = makeRng(102); std::vector<double> Q1 = doubles(c.N * NA), Q2 = doubles(c.N * NA);
      for (int e = 0; e < 500; e++) { double a = qLearningEpisode(c, Q1, 0.5, 1.0, 0.1, 5000, r1).ret, b = sarsaEpisode(c, Q2, 0.5, 1.0, 0.1, 5000, r2).ret; if (e >= 400) { sq += a / 100; ss += b / 100; } }
      std::printf("cliff %.12f %.12f\n", sq, ss); }
    { std::vector<BanditMethod> ms(4); ms[1].eps = 0.1; ms[2].ucb = true; ms[2].c = 2; ms[3].q0 = 5; ms[3].alpha = 0.1;
      std::vector<BanditCurve> cv(4); for (auto& x : cv) { x.reward = doubles(1000); x.optimal = doubles(1000); }
      for (int run = 1; run <= 100; run++) banditRun(run, 1000, ms, cv);
      std::printf("bandit"); for (auto& x : cv) std::printf(" %.12f %.0f", x.reward[999], x.optimal[999]); std::printf("\n"); }
    { double J = 0; std::vector<double> th = doubles(env.N * NA); std::vector<double> g = policyGradientExact(env, th, 0.95, &J);
      double n2 = 0; for (double v : g) n2 += v * v; std::printf("pgexact %.12f %.12f\n", J, std::sqrt(n2)); }
    { Rng r = makeRng(13); std::vector<double> th = doubles(env.N * NA), V = doubles(env.N); double sum = 0;
      for (int e = 0; e < 500; e++) sum += reinforceUpdate(env, th, V, true, 1.0, 0.1, 0.95, 400, r);
      std::printf("reinforce %.12f %.12f\n", sum, jOf(env, softmaxPolicy(th, env.N), 0.95)); }
    { Rng r = makeRng(29); std::vector<double> th = doubles(env.N * NA), V = doubles(env.N); double sum = 0;
      for (int e = 0; e < 500; e++) sum += actorCriticEpisode(env, th, V, 1.0, 0.1, 0.95, 400, r);
      std::printf("ac %.12f %.12f\n", sum, jOf(env, softmaxPolicy(th, env.N), 0.95)); }
    { Rng r = makeRng(17); std::vector<double> th = doubles(env.N * NA), V = doubles(env.N); double cf = 0;
      for (int it = 0; it < 20; it++) { RolloutBatch b = collectBatch(env, softmaxPolicy(th, env.N), V, 8, 0.95, 0.95, 400, r); cf += ppoUpdate(th, V, b, 10, 4, 0.2, 8.0, 0.1, r).clipFrac; }
      std::printf("ppo %.12f %.12f\n", cf, jOf(env, softmaxPolicy(th, env.N), 0.95)); }
    { Rng r = makeRng(1); DQNConfig cfg; DQN d = makeDQN(env, cfg, r);
      for (int k = 0; k < 12000; k++) dqnStep(env, d, r);
      std::vector<double> Q = doubles(env.N * NA);
      { NoGrad ng; for (int s = 0; s < env.N; s++) { Var q = qnetForward(d.net, constant(fromVector(1, 2 + env.W + env.H, stateFeatures(env, s)))); for (int a = 0; a < NA; a++) Q[s * NA + a] = q->val.v[a]; } }
      double rms = 0; int n = 0; for (int s = 0; s < env.N; s++) if (!isTerminal(env, s)) { double e = maxQ(Q, s) - vi.V[s]; rms += e * e; n++; }
      std::printf("dqn %.12f %.12f\n", jOf(env, greedyPi(env, Q), 0.95), std::sqrt(rms / n)); }
    { TreeNode t; t.prior = {0.5, 0.3, 0.2}; t.valueSum = {1.0, 0.0, 0.0}; t.visits = {4, 0, 0}; std::printf("puct %d\n", puctSelect(t, 1.5)); }
    return 0;
}
`;
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'p10cpp-'));
const src = path.join(dir, 'p10.cpp'), exe = path.join(dir, 'p10.exe');
fs.writeFileSync(src, tu);
const t0 = Date.now();
try { execFileSync(CLANG, ['-std=c++17', '-O2', '-Wall', '-Wno-unused-variable', '-Wno-unused-function', '-I', path.join(pages, 'cpp'), src, '-o', exe], { stdio: 'pipe' }); }
catch (e) { console.log(String(e.stderr || e.message).split('\n').slice(0, 60).join('\n')); process.exit(1); }
console.log(`compiled in ${((Date.now() - t0) / 1000).toFixed(1)} s → ${exe}`);
const out = String(execFileSync(exe)).trim().split(/\r?\n/).map((l) => l.trim().split(/\s+/));
const C = Object.fromEntries(out.map((r) => [r[0], r.slice(1).map(Number)]));

// ── JS 端同样的实验 ──
const env = P10.makeEnv(P10.MAPS.main, { slip: 0.2, rstep: -0.04 }), vi = P10.valueIteration(env, 0.95);
const uni = P10.uniformPolicy(env), J = {};
const jOf = (pi) => P10.evalExact(env, pi, 0.95)[env.start];
const greedyPi = (Q) => P10.onehotPolicy(env, P10.greedy(env, Q));
J.vi = [vi.V[env.start], vi.iters];
J.vrand = [P10.evalExact(env, uni, 0.95)[env.start], P10.evalSweeps(env, uni, 0.95, 60).Vs[60][env.start]];
{ const seq = P10.policyIteration(env, 0.95, new Int8Array(env.N).fill(0)), last = seq[seq.length - 1].act; let diff = 0; for (let s = 0; s < env.N; s++) if (!env.term[s] && last[s] !== vi.pi[s]) diff++; J.pi = [seq.length, diff]; }
{ const r = LM.rng(7), ep = { S: [], A: [], R: [], D: [], T: 0 }; let sum = 0; for (let i = 0; i < 1000; i++) { P10.sampleEpisode(env, uni, r, 400, ep, env.start); sum += P10.returnsOf(ep, 0.95)[0]; } J.mc = [sum / 1000]; }
{ const r = LM.rng(11), Q = new Float64Array(env.N * 4); let sum = 0; for (let e = 0; e < 2000; e++) sum += P10.qlEpisode(env, Q, { alpha: 0.1, gamma: 0.95, eps: 0.1, maxT: 200 }, r).ret; J.ql = [sum, Q[env.start * 4], jOf(greedyPi(Q))]; }
{ const c = P10.makeEnv(P10.MAPS.cliff, { slip: 0, rstep: -1, rGoal: -1, rTrap: -1, rCliff: -100 }), r1 = LM.rng(101), r2 = LM.rng(102), Q1 = new Float64Array(c.N * 4), Q2 = new Float64Array(c.N * 4); let sq = 0, ss = 0;
  const o = { alpha: 0.5, gamma: 1, eps: 0.1, maxT: 5000 };
  for (let e = 0; e < 500; e++) { const a = P10.qlEpisode(c, Q1, o, r1).ret, b = P10.sarsaEpisode(c, Q2, o, r2).ret; if (e >= 400) { sq += a / 100; ss += b / 100; } } J.cliff = [sq, ss]; }
{ const ms = [{ kind: 'eps', eps: 0 }, { kind: 'eps', eps: 0.1 }, { kind: 'ucb', c: 2 }, { kind: 'eps', eps: 0, q0: 5, alpha: 0.1 }], acc = { rew: ms.map(() => new Float64Array(1000)), opt: ms.map(() => new Float64Array(1000)) };
  for (let run = 1; run <= 100; run++) P10.banditRun(run, 1000, ms, acc); J.bandit = ms.flatMap((_, i) => [acc.rew[i][999], acc.opt[i][999]]); }
{ const ex = P10.pgExact(env, new Float64Array(env.N * 4), 0.95); J.pgexact = [ex.J, Math.hypot(...ex.grad)]; }
{ const st = P10.pgInit(env, { gamma: 0.95, maxT: 400, lr: 1, lrV: 0.1, baseline: true }, 13); let sum = 0; for (let e = 0; e < 500; e++) sum += P10.reinforceStep(st); J.reinforce = [sum, jOf(st.pi)]; }
{ const st = P10.pgInit(env, { gamma: 0.95, maxT: 400, lr: 1, lrV: 0.1 }, 29); let sum = 0; for (let e = 0; e < 500; e++) sum += P10.acEpisode(st); J.ac = [sum, jOf(st.pi)]; }
{ const st = P10.pgInit(env, { gamma: 0.95, lam: 0.95, lr: 8, lrV: 0.1, M: 8, maxT: 400 }, 17); let cf = 0; for (let it = 0; it < 20; it++) { P10.collectGAE(st); cf += P10.ppoUpdate(st, 10, 4, 0.2).clipFrac; } J.ppo = [cf, jOf(st.pi)]; }
{ const st = P10.dqnInit(env, { hidden: 32, lr: 5e-4, buffer: 5000, batch: 32, warmup: 100, sync: 200, decay: 8000, epsEnd: 0.1, gamma: 0.95, maxT: 100, replay: true, target: true, outStd: 0.01 }, 1);
  for (let k = 0; k < 12000; k++) P10.dqnStep(st);
  const Q = P10.dqnTable(st); let rms = 0, n = 0; for (let s = 0; s < env.N; s++) if (!env.term[s]) { rms += (P10.maxQ(Q, s) - vi.V[s]) ** 2; n++; } J.dqn = [jOf(greedyPi(Q)), Math.sqrt(rms / n)]; }
J.puct = [0];   // Q(0) = 0.25 + 1.5·0.5·2/5 = 0.55 vs 0 + 1.5·0.3·2/1 = 0.9 → 选 1
J.puct = [1];
let bad = 0;
for (const k of Object.keys(C)) {
    const c = C[k], j = J[k] || [];
    const md = Math.max(...c.map((v, i) => Math.abs(v - j[i]) / Math.max(1e-12, Math.abs(j[i]))));
    const exact = c.every((v, i) => v === j[i] || Math.abs(v - j[i]) <= 1e-9 * Math.max(1, Math.abs(j[i])));
    if (!exact && k !== 'dqn') bad++;
    console.log(`${k.padEnd(9)} C++ [${c.map((v) => +v.toPrecision(10)).join(', ')}]\n          JS  [${j.map((v) => +(+v).toPrecision(10)).join(', ')}]  ${exact ? '✔ 一致' : `相对差 ${md.toExponential(2)}`}`);
}
console.log(bad ? `✘ ${bad} 项不一致` : '✔ 表格方法全部一致（DQN 只要求近似）');
