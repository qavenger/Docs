// p04_heads.js — 第 04 篇：分析基座模型的 8 个注意力头（2 层 × 4 头）各自在做什么
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG } = LM;
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const V = LM.vocab, cfg = base.cfg, H = cfg.H, L = cfg.L;
function run(ids) { const T = ids.length; return AG.noGrad(() => LM.gpt.forward(base, ids, 1, T, { trace: true })); }
const att = (tr, l, h, T, i, j) => tr.layers[l].att[(h * T + i) * T + j];

function showSentence(s) {
  const ids = V.encode(s), T = ids.length, out = run(ids), tr = out.trace;
  console.log('\n=== ' + s + ' (T=' + T + ')');
  for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
    let line = `L${l}H${h}: `;
    for (let i = 0; i < T; i++) {
      let bj = 0; for (let j = 0; j <= i; j++) if (att(tr, l, h, T, i, j) > att(tr, l, h, T, i, bj)) bj = j;
      const p = att(tr, l, h, T, i, bj);
      line += `${V.itos[ids[i]]}${i}→${V.itos[ids[bj]]}${bj}(${(p*100).toFixed(0)}) `;
    }
    console.log(line);
  }
  return { ids, tr, T };
}
const a = showSentence('熊猫喜欢吃竹子，所以熊猫每天都吃竹子。');
const b = showSentence('小猫住在屋里。熊猫喜欢吃竹子，所以熊猫每天都吃竹子。');
const c = showSentence('猴子最爱吃香蕉，它住在森林。');

// ── aggregate statistics over random corpus windows ──
const data = V.encode(LM.world.corpus(LM.rng(777), 3000));
const rand = LM.rng(99), T = 32, N = 300;
const stats = [];
for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) stats.push({ l, h, self: 0, prev: 0, first: 0, dot: 0, ind: 0, indN: 0, dup: 0, dupN: 0, ent: 0, n: 0, sentStart: 0, prev2: 0 });
const DOT = V.stoi.get('。');
for (let w = 0; w < N; w++) {
  const s = Math.floor(rand() * (data.length - T - 1));
  const ids = data.slice(s, s + T), out = run(ids), tr = out.trace;
  for (const st of stats) {
    for (let i = 1; i < T; i++) {
      st.self += att(tr, st.l, st.h, T, i, i);
      st.prev += att(tr, st.l, st.h, T, i, i - 1);
      if (i >= 2) st.prev2 += att(tr, st.l, st.h, T, i, i - 2);
      st.first += att(tr, st.l, st.h, T, i, 0);
      let dsum = 0; for (let j = 0; j <= i; j++) if (ids[j] === DOT) dsum += att(tr, st.l, st.h, T, i, j);
      st.dot += dsum;
      // sentence start: the token right after the last 。 before i (or 0)
      let ss = 0; for (let j = i - 1; j >= 0; j--) if (ids[j] === DOT) { ss = j + 1; break; }
      if (ss < i) st.sentStart += att(tr, st.l, st.h, T, i, ss);
      // induction: earlier occurrences j<i with ids[j]==ids[i]: attention to j+1 (j+1 < i)
      let ind = 0, has = false, dup = 0;
      for (let j = 0; j < i - 1; j++) if (ids[j] === ids[i]) { has = true; ind += att(tr, st.l, st.h, T, i, j + 1); dup += att(tr, st.l, st.h, T, i, j); }
      if (has) { st.ind += ind; st.indN++; st.dup += dup; st.dupN++; }
      let e = 0; for (let j = 0; j <= i; j++) { const p = att(tr, st.l, st.h, T, i, j); if (p > 0) e -= p * Math.log(p); }
      st.ent += e;
      st.n++;
    }
  }
}
console.log('\nAggregate over', N, 'windows (T=32). values = mean attention mass');
console.log('head   self  prev  prev2 first  。    sentStart induct dupTok entropy');
for (const st of stats) {
  const f = (v) => (v).toFixed(2).padStart(5);
  console.log(`L${st.l}H${st.h}  ${f(st.self / st.n)} ${f(st.prev / st.n)} ${f(st.prev2 / st.n)} ${f(st.first / st.n)} ${f(st.dot / st.n)} ${f(st.sentStart / st.n)}     ${f(st.ind / Math.max(1, st.indN))}  ${f(st.dup / Math.max(1, st.dupN))}  ${f(st.ent / st.n)}`);
}
