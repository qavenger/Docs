// 离线训练「基座模型」权重并写入 pages/llm_weights.js（第 04 篇前向拆解、第 06 篇 SFT 的起点）。
// 与页面里实时训练用的是同一份 JS 实现（llm_common.js），只是步数更多、batch 更大。
// 用法：node pages/tools/train_base.js [steps=4000] [B=16]
const fs = require('fs');
const path = require('path');
const LM = require('../llm_common.js');
const { AG } = LM;

const steps = +(process.argv[2] || 4000), B = +(process.argv[3] || 16);
const cfg = { V: LM.vocab.size, T: 32, d: 32, L: 2, H: 4, ff: 128 };
const SEED = { init: 1, corpus: 2024, val: 7, batch: 11 };
const m = LM.gpt.create(cfg, SEED.init);
const data = LM.vocab.encode(LM.world.corpus(LM.rng(SEED.corpus), 20000));
const valData = LM.vocab.encode(LM.world.corpus(LM.rng(SEED.val), 2000));
const maxLR = 3e-3, minLR = 3e-4, warmup = 100;
const opt = new LM.AdamW(m.params, { lr: maxLR, wd: 0.1, b2: 0.95 });
const br = LM.rng(SEED.batch), vb = LM.gpt.batch(valData, 32, cfg.T, LM.rng(5));
const curve = [];
const t0 = Date.now();
let ema = null;
for (let it = 0; it <= steps; it++) {
    const b = LM.gpt.batch(data, B, cfg.T, br);
    const lr = LM.cosineLR(it, { maxLR, minLR, warmup, total: steps });
    const r = LM.gpt.trainStep(m, opt, b.x, b.y, B, cfg.T, lr);
    ema = ema == null ? r.loss : 0.95 * ema + 0.05 * r.loss;
    if (it % 50 === 0 || it === steps) {
        const vl = AG.noGrad(() => LM.gpt.forward(m, vb.x, 32, cfg.T, { targets: vb.y }).loss.item());
        curve.push([it, +r.loss.toFixed(4), +vl.toFixed(4)]);
        if (it % 500 === 0) console.log(`it ${it} loss ${ema.toFixed(3)} val ${vl.toFixed(3)} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
}
const b64 = LM.gpt.exportWeights(m);
const valLoss = curve[curve.length - 1][2];
// check that the float16 round trip keeps the loss
const m2 = LM.gpt.loadWeights(LM.gpt.create(cfg, 1), b64);
const vl2 = AG.noGrad(() => LM.gpt.forward(m2, vb.x, 32, cfg.T, { targets: vb.y }).loss.item());
console.log(`final val ${valLoss}, after f16 round trip ${vl2.toFixed(4)}`);
const samples = [0, 1, 2, 3].map((k) => LM.vocab.decode(LM.gpt.generate(m2, LM.vocab.encode(LM.world.animals[k * 2].name), 26, { temperature: 0.7, rand: LM.rng(100 + k) }).ids));
console.log(samples.join('\n'));
const out = `// llm_weights.js — 由 tools/train_base.js 用 llm_common.js 离线训练生成，请勿手改。
// 迷你 GPT「基座模型」：玩具世界语料上预训练 ${steps} 步（batch ${B} × T ${cfg.T}，AdamW，warmup ${warmup} + cosine ${maxLR}→${minLR}）。
// 权重 = 全部参数按 model.params 顺序的 float16 小端序，base64 编码。
const LM_WEIGHTS = {
    base: {
        cfg: ${JSON.stringify(cfg)},
        steps: ${steps}, batch: ${B}, seeds: ${JSON.stringify(SEED)}, maxLR: ${maxLR}, minLR: ${minLR}, warmup: ${warmup},
        valLoss: ${valLoss},
        curve: ${JSON.stringify(curve)},   // [step, train loss, val loss]
        samples: ${JSON.stringify(samples)},
        b64: '${b64}',
    },
};
if (typeof module !== 'undefined' && module.exports) module.exports = LM_WEIGHTS;
`;
fs.writeFileSync(path.join(__dirname, '..', 'llm_weights.js'), out);
console.log('wrote llm_weights.js', (out.length / 1024).toFixed(0), 'KB', ((Date.now() - t0) / 1000).toFixed(0), 's');
