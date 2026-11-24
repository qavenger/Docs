// 离线测试：自动微分梯度校验 + 迷你 GPT 测速 + 试训练。用法：node pages/tools/test_engine.js
const LM = require('../llm_common.js');
const { AG } = LM;

function numGradCheck(name, build, inputs, eps = 1e-2) {
    // build() → scalar tensor; inputs = tensors with req = true
    AG.clearTape();
    inputs.forEach((t) => { t.grad = null; });
    const out = build();
    AG.backward(out);
    let worst = 0;
    inputs.forEach((t, ti) => {
        const ana = Float64Array.from(t.grad || new Float32Array(t.size));
        for (let i = 0; i < Math.min(t.size, 24); i++) {
            const k = Math.floor((i * 7919) % t.size), old = t.data[k];
            t.data[k] = old + eps; const lp = AG.noGrad(build).item();
            t.data[k] = old - eps; const lm = AG.noGrad(build).item();
            t.data[k] = old;
            const num = (lp - lm) / (2 * eps), a = ana[k];
            const rel = Math.abs(num - a) / Math.max(1e-3, Math.abs(num) + Math.abs(a));
            if (rel > worst) worst = rel;
            if (rel > 0.05) console.log(`  ${name} input#${ti}[${k}] analytic=${a.toExponential(3)} numeric=${num.toExponential(3)} rel=${rel.toFixed(3)}`);
        }
    });
    console.log(`${worst < 0.05 ? 'OK  ' : 'FAIL'} ${name.padEnd(16)} worst rel err ${worst.toExponential(2)}`);
}
const rand = LM.rng(42);
const P = (r, c, s = 1) => AG.param('p', r, c, (a) => { for (let i = 0; i < a.length; i++) a[i] = s * LM.randn(rand); });
// a fixed random projection to turn tensors into a scalar
const proj = (T) => { const W = AG.tensor(T.r, T.c); const rr = LM.rng(9); for (let i = 0; i < W.size; i++) W.data[i] = rr() - 0.5; return AG.sum(AG.mul(T, W)); };

{ const A = P(3, 4), B = P(4, 5); numGradCheck('matmul', () => proj(AG.matmul(A, B)), [A, B]); }
{ const A = P(3, 4), B = P(5, 4); numGradCheck('matmulBT', () => proj(AG.matmulBT(A, B)), [A, B]); }
{ const X = P(3, 4), W = P(4, 2), b = P(1, 2); numGradCheck('linear', () => proj(AG.linear(X, W, b)), [X, W, b]); }
{ const A = P(3, 4); numGradCheck('gelu', () => proj(AG.gelu(A)), [A]); }
{ const A = P(3, 4); numGradCheck('tanh', () => proj(AG.tanh(A)), [A]); }
{ const A = P(3, 4); numGradCheck('sigmoid', () => proj(AG.sigmoid(A)), [A]); }
{ const A = P(3, 4, 3); numGradCheck('logsigmoid', () => proj(AG.logsigmoid(A)), [A]); }
{ const X = P(3, 6), g = P(1, 6), b = P(1, 6); numGradCheck('layernorm', () => proj(AG.layernorm(X, g, b)), [X, g, b], 1e-2); }
{ const E = P(5, 3); const idx = Int32Array.from([1, 3, 1, 0]); numGradCheck('gather', () => proj(AG.gather(E, idx)), [E]); }
{ const A = P(2, 3), B = P(3, 3); numGradCheck('concat', () => proj(AG.concat(A, B)), [A, B]); }
{ const X = P(6, 3); numGradCheck('meanRows', () => proj(AG.meanRows(X, 3)), [X]); }
{ const X = P(3, 4); numGradCheck('l2norm', () => proj(AG.l2norm(X)), [X]); }
{ const A = P(3, 4); numGradCheck('transpose', () => proj(AG.transpose(A)), [A]); }
{ const B = 2, T = 4, D = 8, H = 2; const Q = P(B * T, D), K = P(B * T, D), V = P(B * T, D);
  numGradCheck('attention causal', () => proj(AG.attention(Q, K, V, B, T, H, true)), [Q, K, V]);
  numGradCheck('attention full', () => proj(AG.attention(Q, K, V, B, T, H, false)), [Q, K, V]); }
{ const L = P(4, 6); const t = Int32Array.from([1, -1, 5, 0]); const w = Float32Array.from([1, 1, -0.5, 2]);
  numGradCheck('crossEntropy', () => AG.crossEntropy(L, t), [L]);
  numGradCheck('CE weighted', () => AG.crossEntropy(L, t, { weights: w, denom: 3 }), [L]); }
{ const L = P(5, 6); const t = Int32Array.from([1, 2, 5, 0, 3]); const rs = Int32Array.from([0, 0, 1, -1, 1]);
  numGradCheck('seqLogProb', () => proj(AG.seqLogProb(L, t, rs, 2)), [L]); }
{ const A = P(3, 4), B = P(3, 4); numGradCheck('mse', () => AG.mse(A, B), [A, B]); }

// full GPT gradient check (tiny)
{
    const m = LM.gpt.create({ V: 11, T: 5, d: 8, L: 2, H: 2, ff: 16, std: 0.3 }, 3);
    const x = Int32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), y = Int32Array.from([2, 3, 4, 5, 6, 7, 8, 9, 10, 0]);
    numGradCheck('gpt full', () => LM.gpt.forward(m, x, 2, 5, { targets: y }).loss, m.params.filter((_, i) => i % 3 === 0), 1e-2);
}

// KV-cache generate must match full forward
{
    const V = LM.vocab;
    const m = LM.gpt.create({ V: V.size, T: 32, d: 32, L: 2, H: 4, ff: 128, std: 0.2 }, 5);
    const ids = Array.from(V.encode('小猫喜欢吃鱼，所以小猫'));
    const full = LM.gpt.nextLogits(m, ids);
    let kv = null;
    LM.gpt.generate(m, ids, 1, { greedy: true, onStep: (d, id, lg) => { kv = Float32Array.from(lg); } });
    let md = 0; for (let i = 0; i < full.length; i++) md = Math.max(md, Math.abs(full[i] - kv[i]));
    console.log(`${md < 1e-4 ? 'OK  ' : 'FAIL'} kv-cache vs full forward: max |Δlogit| = ${md.toExponential(2)}`);
}

console.log('vocab size', LM.vocab.size, LM.vocab.itos.join(''));
const corpusRand = LM.rng(1);
const text = LM.world.corpus(corpusRand, 3);
console.log('sample corpus:', text);

// speed + trial training
const cfgArg = process.argv[2] ? JSON.parse(process.argv[2]) : {};
const cfg = Object.assign({ V: LM.vocab.size, T: 32, d: 32, L: 2, H: 4, ff: 128 }, cfgArg.model || {});
const B = cfgArg.B || 8, steps = cfgArg.steps || 600, lr = cfgArg.lr || 3e-3;
const m = LM.gpt.create(cfg, 1);
console.log('params', LM.gpt.numParams(m), JSON.stringify(cfg), 'B', B);
const data = LM.vocab.encode(LM.world.corpus(LM.rng(2024), 20000));
const valData = LM.vocab.encode(LM.world.corpus(LM.rng(7), 2000));
console.log('corpus tokens', data.length);
const opt = new LM.AdamW(m.params, { lr, wd: 0.1, b2: 0.95 });
const br = LM.rng(11);
const batch = { x: new Int32Array(B * cfg.T), y: new Int32Array(B * cfg.T), starts: new Int32Array(B) };
const t0 = Date.now();
for (let it = 0; it <= steps; it++) {
    LM.gpt.batch(data, B, cfg.T, br, batch);
    const lrIt = LM.cosineLR(it, { maxLR: lr, minLR: lr / 10, warmup: 50, total: steps });
    const r = LM.gpt.trainStep(m, opt, batch.x, batch.y, B, cfg.T, lrIt);
    if (it % Math.max(1, Math.floor(steps / 12)) === 0 || it === steps) {
        const vb = LM.gpt.batch(valData, 16, cfg.T, LM.rng(5));
        const vl = AG.noGrad(() => LM.gpt.forward(m, vb.x, 16, cfg.T, { targets: vb.y }).loss.item());
        const g = LM.gpt.generate(m, LM.vocab.encode('熊猫'), 24, { temperature: 0.8, rand: LM.rng(it + 1) });
        console.log(`it ${String(it).padStart(5)} loss ${r.loss.toFixed(3)} val ${vl.toFixed(3)} |g| ${r.gradNorm.toFixed(2)} lr ${lrIt.toExponential(1)}  ${((Date.now() - t0) / (it + 1)).toFixed(1)} ms/step  → ${LM.vocab.decode(g.ids)}`);
    }
}
if (cfgArg.save) {
    const fs = require('fs');
    fs.writeFileSync(cfgArg.save, LM.gpt.exportWeights(m));
    console.log('saved', cfgArg.save);
}
