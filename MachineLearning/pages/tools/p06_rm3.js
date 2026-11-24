// p06_rm3.js — RM 变体：mean-pool 回答 token 的隐状态 + 头部用更大学习率；对比冻结 trunk 与整网微调
// 标注噪声：BTS=2（默认）与页面相同（r* 乘 2）。
// 用法：node pages/tools/p06_rm3.js <steps> <pairsPerStep> <trunkLR> <headLR> <nLabels> <pool:mean|last>
const fs = require('fs');
const LM = require('../llm_common.js'); const W = require('../llm_weights.js');
const { AG, vocab: V, world } = LM;
const [STEPS, PB, LR, HLR, NL] = process.argv.slice(2, 7).map(Number), POOL = process.argv[7] || 'mean';
const A = world.animals, Q = world.questions, fact = (a, q) => Q[q].a(A[a]);
const cfg = W.base.cfg, TM = 28;
global.LM = LM; global.LM_WEIGHTS = W; global.PipelineLab = global.PL = { mount() {} };
const { P06, P06_DATA } = require('../06_alignment.lab.js');   // 用页面内嵌的 SFT checkpoint
const sft = LM.gpt.loadWeights(LM.gpt.create(cfg, 1), P06_DATA.sft);
const base = LM.gpt.loadWeights(LM.gpt.create(cfg, 1), W.base.b64);
const attr = (a, q) => [A[a].food, A[a].home, A[a].color][q];
const TYPES = ['正确', '张冠李戴', '事实错误', '答非所问', '啰嗦', '停不下来'], RTRUE = [1.0, -1.0, -1.0, -0.6, 0.4, -1.5];
function seqLP(mm, prompt, resp) { const ids = [...prompt, ...resp], n = ids.length - 1, x = Int32Array.from(ids.slice(0, n)), y = new Int32Array(n).fill(-1); for (let t = prompt.length - 1; t < n; t++) y[t] = ids[t + 1]; return AG.noGrad(() => AG.seqLogProb(LM.gpt.forward(mm, x, 1, n).logits, y, new Int32Array(n).fill(0), 1).data[0]); }
const prompts = [];
A.forEach((_, a) => Q.forEach((_, q) => {
    const P = V.encode('<user>' + Q[q].q(A[a]) + '<bot>'), lp = (s) => seqLP(sft, P, V.encode(s));
    const argmaxBy = (list) => list.reduce((b, s) => { const v = lp(s); return !b || v > b.v ? { s, v } : b; }, null).s;
    const c2 = argmaxBy(A.map((_, a2) => a2).filter((a2) => a2 !== a).map((a2) => fact(a2, q) + '<eos>'));
    const c3 = argmaxBy(A.map((_, a2) => a2).filter((a2) => attr(a2, q) !== attr(a, q)).map((a2) => Q[q].a(Object.assign({}, A[a], { food: A[a2].food, home: A[a2].home, color: A[a2].color })) + '<eos>'));
    const c4 = argmaxBy([0, 1, 2].filter((q2) => q2 !== q).map((q2) => fact(a, q2) + '<eos>'));
    const c5 = argmaxBy([0, 1, 2].filter((q2) => q2 !== q).map((q2) => fact(a, q) + fact(a, q2) + '<eos>'));
    const pre = [...P, ...V.encode(fact(a, q))], g = LM.gpt.generate(base, pre, TM - pre.length, { greedy: true });
    const c6 = fact(a, q) + V.decode(g.gen.filter((i) => i > 3));
    prompts.push({ a, q, P, cands: [fact(a, q) + '<eos>', c2, c3, c4, c5, c6].map((s) => V.encode(s)) });
}));
const rand = LM.rng(1), K = 6, all = [];
prompts.forEach((p, pi) => { for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) all.push([pi, i, j]); });
const labels = [];
for (let n = 0; n < NL; n++) { const [pi, i, j] = all[n % all.length]; const pw = 1 / (1 + Math.exp(-(+(process.env.BTS || 2)) * (RTRUE[i] - RTRUE[j]))); labels.push(rand() < pw ? [pi, i, j] : [pi, j, i]); }
const testP = new Set([1, 5, 9, 13, 17, 21]), trainL = labels.filter(([pi]) => !testP.has(pi));
const rm = LM.gpt.clone(sft), wr = AG.param('rm.w', cfg.d, 1, (a) => a.fill(0)), br = AG.param('rm.b', 1, 1);
const opt = new LM.AdamW(rm.params, { lr: LR, wd: 0, b2: 0.95 }), hopt = new LM.AdamW([wr, br], { lr: HLR, wd: 0, b2: 0.95 });
function scoreBatch(seqs) {
    const n = seqs.length, x = new Int32Array(n * TM), M = AG.tensor(n, n * TM);
    seqs.forEach((s, b) => { const ids = [...s.P, ...s.c]; for (let t = 0; t < TM; t++) x[b * TM + t] = t < ids.length ? ids[t] : 0;
        if (POOL === 'mean') { const t0 = s.P.length, t1 = ids.length - 1; for (let t = t0; t <= t1; t++) M.data[b * n * TM + b * TM + t] = 1 / (t1 - t0 + 1); }
        else M.data[b * n * TM + b * TM + ids.length - 1] = 1; });
    const out = LM.gpt.forward(rm, x, n, TM, { trace: true });
    return AG.addRow(AG.matmul(AG.matmul(M, out.trace.hf), wr), br);
}
function evalAcc(set) { let ok = 0, tot = 0; AG.noGrad(() => set.forEach((pi) => { const p = prompts[pi], r = scoreBatch(p.cands.map((c) => ({ P: p.P, c }))).data; for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) { if (RTRUE[i] === RTRUE[j]) continue; tot++; if ((r[i] - r[j]) * (RTRUE[i] - RTRUE[j]) > 0) ok++; } })); return ok / tot; }
const br2 = LM.rng(5); const t0 = Date.now();
for (let it = 1; it <= STEPS; it++) {
    const seqs = []; for (let k = 0; k < PB; k++) { const [pi, w, l] = trainL[Math.floor(br2() * trainL.length)], p = prompts[pi]; seqs.push({ P: p.P, c: p.cands[w] }, { P: p.P, c: p.cands[l] }); }
    opt.zeroGrad(); hopt.zeroGrad();
    const r = scoreBatch(seqs), D = AG.tensor(PB, 2 * PB); for (let k = 0; k < PB; k++) { D.data[k * 2 * PB + 2 * k] = 1; D.data[k * 2 * PB + 2 * k + 1] = -1; }
    const loss = AG.scale(AG.mean(AG.logsigmoid(AG.matmul(D, r))), -1);
    AG.backward(loss); LM.clipGradNorm(rm.params.concat([wr, br]), 1.0); if (LR > 0) opt.step(LR); hopt.step(HLR);
    if (it % 100 === 0) console.log(`it ${it} acc(train prompts) ${evalAcc([...Array(24).keys()].filter((k) => !testP.has(k))).toFixed(3)} acc(test prompts) ${evalAcc([...testP]).toFixed(3)} ${((Date.now() - t0) / it).toFixed(1)}ms/it`);
}
const sc = prompts.map((p) => AG.noGrad(() => Array.from(scoreBatch(p.cands.map((c) => ({ P: p.P, c }))).data)));
console.log('mean r̂ by type:', TYPES.map((t, k) => `${t} ${LM.mean(sc.map((r) => r[k])).toFixed(2)}`).join(' | '));
