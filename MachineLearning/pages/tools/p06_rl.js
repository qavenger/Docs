// p06_rl.js — 第 06 篇原型：在真实迷你 GPT 上跑 DPO（AG.seqLogProb）和 GRPO（KV cache 采样 + 加权交叉熵 + k3 KL）
// 用法：node pages/tools/p06_rl.js dpo [steps=300] [pairs=2] [beta=0.1] [lr=1e-4]
//       node pages/tools/p06_rl.js grpo [steps=300] [G=4] [beta=0.04] [lr=1e-4]
const fs = require('fs');
const LM = require('../llm_common.js'); const W = require('../llm_weights.js');
const { AG, vocab: V, world } = LM;
const MODE = process.argv[2], STEPS = +(process.argv[3] || 300), NB = +(process.argv[4] || (MODE === 'dpo' ? 2 : 4)), BETA = +(process.argv[5] || (MODE === 'dpo' ? 0.1 : 0.04)), LR = +(process.argv[6] || 1e-4);
const A = world.animals, Q = world.questions, fact = (a, q) => Q[q].a(A[a]);
const cfg = W.base.cfg, K = 6, TM = 28;
global.LM = LM; global.LM_WEIGHTS = W; global.PipelineLab = global.PL = { mount() {} };
const { P06, P06_DATA } = require('../06_alignment.lab.js');   // 用页面内嵌的 SFT checkpoint
const sft = LM.gpt.loadWeights(LM.gpt.create(cfg, 1), P06_DATA.sft);
global.P06_DATA = P06_DATA;
const C = P06.buildCands(sft).prompts.map((p) => ({ cands: p.strs, lps: p.lps }));
const RTRUE = P06.R_TRUE, TYPES = ['正确', '张冠李戴', '事实错误', '答非所问', '啰嗦', '停不下来'];
const HOLD = [[4, 2], [7, 1], [3, 0], [0, 2], [5, 1], [6, 0]];
const prompts = [];
A.forEach((_, a) => Q.forEach((_, q) => prompts.push({ a, q, held: HOLD.some(([x, y]) => x === a && y === q), P: Array.from(V.encode('<user>' + Q[q].q(A[a]) + '<bot>')) })));
prompts.forEach((p, pi) => { p.cands = C[pi].cands.map((s) => Array.from(V.encode(s))); p.refLp = C[pi].lps; });
/** batch of (prompt, response) → x, y (targets on response tokens), rowSeq, T */
function pack(seqs) {
    const T = Math.max(...seqs.map((s) => s.P.length + s.R.length)) - 1, n = seqs.length;
    const x = new Int32Array(n * T), y = new Int32Array(n * T).fill(-1), rs = new Int32Array(n * T).fill(-1);
    seqs.forEach((s, b) => { const ids = [...s.P, ...s.R]; for (let t = 0; t < T; t++) { x[b * T + t] = t < ids.length ? ids[t] : 0; if (t >= s.P.length - 1 && t < ids.length - 1) { y[b * T + t] = ids[t + 1]; rs[b * T + t] = b; } } });
    return { x, y, rs, T, n };
}
function seqLogProbs(m, seqs) { const pk = pack(seqs); return AG.noGrad(() => Array.from(AG.seqLogProb(LM.gpt.forward(m, pk.x, pk.n, pk.T).logits, pk.y, pk.rs, pk.n).data)); }
// verifier: response (without specials) equals the canonical answer and ends with <eos>
const verify = (p, gen) => (gen[gen.length - 1] === 1 && V.decode(gen.slice(0, -1)) === fact(p.a, p.q) ? 1 : 0);
function sampledAcc(m, which, n = 16) {
    let ok = 0, tot = 0;
    prompts.forEach((p, pi) => { if (which === 'held' ? !p.held : p.held) return; for (let s = 0; s < n; s++) { const g = LM.gpt.generate(m, p.P, TM - p.P.length, { temperature: 1, rand: LM.rng(31337 + pi * 101 + s), stop: new Set([1]) }); ok += verify(p, g.gen); tot++; } });
    return ok / tot;
}
function typeProbs(m) {   // mean over prompts of π(candidate type) renormalised over the candidate set, and raw mean logp
    const out = new Array(K).fill(0);
    prompts.forEach((p) => { const lp = seqLogProbs(m, p.cands.map((R) => ({ P: p.P, R }))); const mx = Math.max(...lp), z = lp.reduce((s, v) => s + Math.exp(v - mx), 0); lp.forEach((v, k) => { out[k] += Math.exp(v - mx) / z / prompts.length; }); });
    return out;
}
const pol = LM.gpt.clone(sft), opt = new LM.AdamW(pol.params, { lr: LR, wd: 0, b2: 0.95 }), rand = LM.rng(9);
console.log('start: held-out sampled acc', sampledAcc(pol, 'held').toFixed(3), ' train-prompt acc', sampledAcc(pol, 'train').toFixed(3), ' types', typeProbs(pol).map((v, k) => TYPES[k] + ' ' + v.toExponential(1)).join(' '));
const t0 = Date.now(); let tSamp = 0;
if (MODE === 'dpo') {
    // preference labels (same construction as p06_align.js, 1 repeat)
    const lr2 = LM.rng(11), labels = [];
    const KD = +(process.env.KD || K); prompts.forEach((p, pi) => { for (let i = 0; i < KD; i++) for (let j = i + 1; j < KD; j++) { const pw = process.env.CLEAN ? (RTRUE[i] > RTRUE[j] ? 1 : RTRUE[i] < RTRUE[j] ? 0 : -1) : 1 / (1 + Math.exp(-(RTRUE[i] - RTRUE[j]))); if (pw < 0) continue; labels.push(lr2() < pw ? [pi, i, j] : [pi, j, i]); } });
    for (let it = 1; it <= STEPS; it++) {
        const seqs = [], ref = [];
        for (let b = 0; b < NB; b++) { const [pi, w, l] = labels[Math.floor(rand() * labels.length)], p = prompts[pi]; seqs.push({ P: p.P, R: p.cands[w] }, { P: p.P, R: p.cands[l] }); ref.push(p.refLp[w], p.refLp[l]); }
        const pk = pack(seqs);
        opt.zeroGrad();
        const lp = AG.seqLogProb(LM.gpt.forward(pol, pk.x, pk.n, pk.T).logits, pk.y, pk.rs, pk.n);      // [2NB × 1]
        const D = AG.tensor(NB, 2 * NB); for (let b = 0; b < NB; b++) { D.data[b * 2 * NB + 2 * b] = BETA; D.data[b * 2 * NB + 2 * b + 1] = -BETA; }
        const refT = AG.tensor(2 * NB, 1, Float32Array.from(ref));
        const z = AG.matmul(D, AG.sub(lp, refT));                                                     // β[(lp_w − ref_w) − (lp_l − ref_l)]
        let loss = AG.scale(AG.mean(AG.logsigmoid(z)), -1);
        const NLL = +(process.env.NLL || 0); if (NLL > 0) { const Sel = AG.tensor(1, 2 * NB); for (let b = 0; b < NB; b++) Sel.data[2 * b] = -NLL / (NB * seqs[2 * b].R.length); loss = AG.add(loss, AG.matmul(Sel, lp)); }
        AG.backward(loss); LM.clipGradNorm(pol.params, 1.0); opt.step(LR);
        if (it % (+(process.env.EVERY || 100)) === 0) console.log(`DPO it ${it} loss ${loss.item().toFixed(3)} | held acc ${sampledAcc(pol, 'held').toFixed(3)} train acc ${sampledAcc(pol, 'train').toFixed(3)} | types ${typeProbs(pol).map((v, k) => TYPES[k] + ' ' + v.toExponential(1)).join(' ')} | ${((Date.now() - t0) / it).toFixed(1)} ms/it`);
    }
    prompts.filter((p) => p.held).slice(0, 3).forEach((p) => console.log('   sample:', [0, 1, 2].map((s) => Array.from(LM.gpt.generate(pol, p.P, TM - p.P.length, { temperature: 1, rand: LM.rng(s), stop: new Set([1]) }).gen, (i) => V.show(i)).join('')).join(' | ')));
} else {
    const ref = sft;
    let zeroGroups = 0;
    for (let it = 1; it <= STEPS; it++) {
        const pi = Math.floor(rand() * prompts.length), p = prompts[pi];
        const s0 = Date.now();
        const outs = [];
        for (let g = 0; g < NB; g++) { const o = LM.gpt.generate(pol, p.P, TM - p.P.length, { temperature: 1, rand, stop: new Set([1]) }); outs.push(o.gen); }
        tSamp += Date.now() - s0;
        const r = outs.map((gen) => verify(p, gen)), mu = LM.mean(r), sd = Math.sqrt(LM.mean(r.map((v) => (v - mu) ** 2)));
        if (sd < 1e-6) { zeroGroups++; continue; }
        const Adv = r.map((v) => (v - mu) / (sd + 1e-4));
        const seqs = outs.map((gen) => ({ P: p.P, R: gen })), pk = pack(seqs);
        const refLog = AG.noGrad(() => LM.gpt.forward(ref, pk.x, pk.n, pk.T).logits.data);
        opt.zeroGrad();
        const out = LM.gpt.forward(pol, pk.x, pk.n, pk.T), lg = out.logits.data, Vn = cfg.V;
        const wts = new Float32Array(pk.n * pk.T);
        for (let row = 0; row < pk.n * pk.T; row++) {
            const t = pk.y[row]; if (t < 0) continue;
            const b = pk.rs[row], len = seqs[b].R.length;
            const lpRow = (L) => { let m = -Infinity; for (let j = 0; j < Vn; j++) m = Math.max(m, L[row * Vn + j]); let s = 0; for (let j = 0; j < Vn; j++) s += Math.exp(L[row * Vn + j] - m); return L[row * Vn + t] - m - Math.log(s); };
            const rho = Math.exp(lpRow(refLog) - lpRow(lg));                // π_ref / π_θ for the sampled token
            wts[row] = (Adv[b] - BETA * (1 - rho)) / (NB * len);           // CE weight = −∂J/∂log π
        }
        const loss = AG.crossEntropy(out.logits, pk.y, { weights: wts, denom: 1 });
        AG.backward(loss); LM.clipGradNorm(pol.params, 1.0); opt.step(LR);
        if (it % 50 === 0) console.log(`GRPO it ${it} zero-var groups ${zeroGroups} | held acc ${sampledAcc(pol, 'held').toFixed(3)} train acc ${sampledAcc(pol, 'train').toFixed(3)} | ${((Date.now() - t0) / it).toFixed(1)} ms/it (sampling ${(tSamp / it).toFixed(1)})`);
    }
}
