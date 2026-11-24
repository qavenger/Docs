// p06_sft2.js — 第 06 篇原型：两种留出方式对比（整只动物 vs 个别问答对），多种子
// 用法：node pages/tools/p06_sft2.js <mode: animal|pair> <B> <lr> <steps> <spec> [mixPre=0]
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG, vocab: V, world } = LM;
const [mode, Bs, lrs, sts, spec, mixs] = process.argv.slice(2);
const B = +Bs, LR = +lrs, STEPS = +sts, MIX = +(mixs || 0);
const show = (ids) => Array.from(ids, (i) => V.show(i)).join('');
const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const ex = [];
world.animals.forEach((a, ai) => world.questions.forEach((q, qi) => {
    const p = V.encode('<user>' + q.q(a) + '<bot>'), r = V.encode(q.a(a) + '<eos>');
    ex.push({ ai, qi, prompt: p, resp: r, ids: Int32Array.from([...p, ...r]) });
}));
const T = Math.max(...ex.map((e) => e.ids.length)) - 1;
let isTest;
if (mode === 'animal') { const H = spec.split(',').map(Number); isTest = (e) => H.includes(e.ai); }
else { const P = spec.split(';').map((s) => s.split(',').map(Number)); isTest = (e) => P.some(([a, q]) => a === e.ai && q === e.qi); }
const train = ex.filter((e) => !isTest(e)), test = ex.filter(isTest);
const pre = V.encode(world.corpus(LM.rng(2024), 3000));
function makeBatch(list, rand, Bn) {
    const x = new Int32Array(Bn * T), y = new Int32Array(Bn * T).fill(-1);
    for (let b = 0; b < Bn; b++) {
        if (MIX && rand() < MIX) {   // a pretraining window of length T (all positions count)
            const s = Math.floor(rand() * (pre.length - T - 1));
            for (let t = 0; t < T; t++) { x[b * T + t] = pre[s + t]; y[b * T + t] = pre[s + t + 1]; }
            continue;
        }
        const e = list[Math.floor(rand() * list.length)], n = e.ids.length, np = e.prompt.length;
        for (let t = 0; t < T; t++) { x[b * T + t] = t < n - 1 ? e.ids[t] : 0; if (t < n - 1 && t + 1 >= np) y[b * T + t] = e.ids[t + 1]; }
    }
    return { x, y };
}
function evalSet(m, list) {
    let exact = 0; const outs = [];
    list.forEach((e) => { const g = LM.gpt.generate(m, e.prompt, 14, { greedy: true, stop: new Set([1]) }); const s = show(g.gen); if (s === show(e.resp)) exact++; outs.push(s); });
    return { exact: exact / list.length, outs };
}
const valData = V.encode(world.corpus(LM.rng(7), 400)), vb = LM.gpt.batch(valData, 16, 32, LM.rng(5));
const vl = (mm) => AG.noGrad(() => LM.gpt.forward(mm, vb.x, 16, 32, { targets: vb.y }).loss.item());
const res = [];
for (const seed of [42, 43, 44]) {
    const m = LM.gpt.clone(base), opt = new LM.AdamW(m.params, { lr: LR, wd: 0.0, b2: 0.95 }), rand = LM.rng(seed);
    const line = [];
    for (let it = 1; it <= STEPS; it++) {
        const bt = makeBatch(train, rand, B);
        LM.gpt.trainStep(m, opt, bt.x, bt.y, B, T, LR);
        if (it % 50 === 0) { const tr = evalSet(m, train), te = evalSet(m, test); line.push(`${it}:${tr.exact.toFixed(2)}/${te.exact.toFixed(2)}`); }
    }
    const te = evalSet(m, test);
    console.log(`seed ${seed} [train/test exact] ${line.join(' ')} | preVal ${vl(m).toFixed(3)} | ${te.outs.join(' ')}`);
}
console.log('test prompts:', test.map((e) => show(e.prompt).replace('⟨user⟩', '').replace('⟨bot⟩', '')).join(' '));
