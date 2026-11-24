// p06_base.js — 第 06 篇原型：基座模型拿到对话模板时会生成什么？特殊 token 的 embedding 是什么状态？
// 用法：node pages/tools/p06_base.js
const LM = require('../llm_common.js');
const W = require('../llm_weights.js');
const { AG, vocab: V, world } = LM;

const base = LM.gpt.loadWeights(LM.gpt.create(W.base.cfg, 1), W.base.b64);
const show = (ids) => Array.from(ids, (i) => V.show(i)).join('');

// 1) chat prompts → greedy + sampled continuations
for (const a of world.animals.slice(0, 8)) {
    const q = world.questions[0].q(a);
    const prompt = V.encode('<user>' + q + '<bot>');
    const g = LM.gpt.generate(base, prompt, 32, { greedy: true });
    console.log('greedy', show(prompt), '→', show(g.gen));
}
const a = world.animals[4];
const prompt = V.encode('<user>' + world.questions[0].q(a) + '<bot>');
for (let s = 0; s < 6; s++) {
    const g = LM.gpt.generate(base, prompt, 32, { temperature: 1, rand: LM.rng(100 + s) });
    console.log('T=1   ', show(g.gen));
}
// next-token distribution right after <bot>
{
    const lg = LM.gpt.nextLogits(base, Array.from(prompt));
    const p = LM.softmax(lg);
    const top = LM.topk(p, 8).map((i) => `${V.show(i)}:${p[i].toFixed(3)}`);
    console.log('after <bot> top8:', top.join(' '), ' P(<eos>)=', p[1].toExponential(2));
}
// P(<eos>) after a full answer
{
    const ids = V.encode('<user>熊猫喜欢吃什么？<bot>熊猫喜欢吃竹子。');
    const p = LM.softmax(LM.gpt.nextLogits(base, Array.from(ids)));
    console.log('after answer: top5', LM.topk(p, 5).map((i) => `${V.show(i)}:${p[i].toFixed(3)}`).join(' '), ' P(<eos>)=', p[1].toExponential(2));
}
// plain prompt: the knowledge is there
{
    const ids = V.encode('熊猫喜欢吃');
    const g = LM.gpt.generate(base, ids, 12, { greedy: true });
    console.log('plain greedy 熊猫喜欢吃 →', show(g.gen));
}
// 2) special-token embeddings
{
    const d = base.cfg.d, E = base.wte.data;
    const nrm = (i) => { let s = 0; for (let j = 0; j < d; j++) s += E[i * d + j] ** 2; return Math.sqrt(s); };
    const norms = Array.from({ length: V.size }, (_, i) => nrm(i));
    console.log('‖wte‖ specials:', norms.slice(0, 4).map((x) => x.toFixed(3)).join(' '), ' others mean', LM.mean(norms.slice(4)).toFixed(3), 'min', Math.min(...norms.slice(4)).toFixed(3), 'max', Math.max(...norms.slice(4)).toFixed(3));
    // expected norm of an untouched N(0, 0.02) row: 0.02*sqrt(32) = 0.113
    const fresh = LM.gpt.create(W.base.cfg, 1).wte.data;
    const nf = (i) => { let s = 0; for (let j = 0; j < d; j++) s += fresh[i * d + j] ** 2; return Math.sqrt(s); };
    console.log('init ‖wte‖ specials:', [0, 1, 2, 3].map((i) => nf(i).toFixed(3)).join(' '));
    // cosine between trained special row and its init
    for (let i = 0; i < 4; i++) {
        let dot = 0, n1 = 0, n2 = 0; for (let j = 0; j < d; j++) { dot += E[i * d + j] * fresh[i * d + j]; n1 += E[i * d + j] ** 2; n2 += fresh[i * d + j] ** 2; }
        console.log(`  ${V.itos[i]} cos(trained, init) = ${(dot / Math.sqrt(n1 * n2)).toFixed(3)}`);
    }
    // mean logit of special tokens over a corpus batch
    const data = V.encode(world.corpus(LM.rng(5), 200));
    const b = LM.gpt.batch(data, 8, 32, LM.rng(3));
    const out = AG.noGrad(() => LM.gpt.forward(base, b.x, 8, 32, { targets: b.y }));
    const Vn = base.cfg.V, lg = out.logits.data; let ps = [0, 0, 0, 0], n = 8 * 32;
    for (let r = 0; r < n; r++) { const p = LM.softmax(lg, null, 1, r * Vn, Vn); for (let i = 0; i < 4; i++) ps[i] += p[i] / n; }
    console.log('mean P(special) on corpus:', ps.map((x) => x.toExponential(2)).join(' '), ' loss', out.loss.item().toFixed(3));
}
