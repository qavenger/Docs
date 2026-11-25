// 离线训练第 10 篇的 CLIP 图像塔，并把权重（float16 base64）写进 10_multimodal.lab.js 的 <P07_W> 段。
// 与页面里实时训练用的是同一份代码、同一组默认超参数与种子（P07.CLIP_DEFAULT），所以它等于 Step 04 默认设置训练到终点的结果。
// 之后用页面同样的配方（P07.LLAVA_DEFAULT）在 Node 里跑一遍迷你 LLaVA，打印各检查点的 caption 准确率（写进页面 note 的数字来自这里）。
// 用法：node pages/tools/p10_train.js [--no-llava]
const fs = require('fs');
const { LM, P07, file } = require('./p10_load.js');

const t0 = Date.now();
const LLAVA_ONLY = process.argv.includes('--llava-only');
let vit;
if (!LLAVA_ONLY) {
const C = P07.clipRecipe(P07.PAGE_SEED), tr = P07.clipTrainer(C);
const curve = [];
let ema = null;
for (let i = 0; i < C.steps; i++) {
    const r = tr.step(); ema = ema == null ? r.loss : 0.98 * ema + 0.02 * r.loss;
    if ((i + 1) % 500 === 0) console.log(`clip step ${i + 1} loss(ema) ${ema.toFixed(3)} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    if ((i + 1) % 50 === 0) curve.push([i + 1, +ema.toFixed(4)]);
}
function zeroShot(vit, txt, test, t) {
    const ie = P07.embedImages(vit, test).emb.data, te = P07.embedTexts(txt, P07.prompts(t)).emb.data, d = 32;
    let ok = 0;
    test.forEach((s, i) => { let b = 0, bv = -9; for (let k = 0; k < P07.NCLS; k++) { let v = 0; for (let j = 0; j < d; j++) v += ie[i * d + j] * te[k * d + j]; if (v > bv) { bv = v; b = k; } } if (b === s.cls) ok++; });
    return ok / test.length;
}
const zs = P07.makeZsTest();
const zsAcc = P07.TEMPLATES.map((t) => zeroShot(tr.clip.img, tr.clip.txt, zs, t.id));
console.log('zero-shot (48 test images) per template:', zsAcc.map((a) => (a * 100).toFixed(1) + '%').join(' / '));

const b64 = P07.exportParams(tr.clip.img.params);
vit = P07.createViT(P07.CLIP_CFG.img, 1); P07.loadParams(vit.params, b64);
let md = 0; vit.params.forEach((p, i) => { for (let k = 0; k < p.size; k++) md = Math.max(md, Math.abs(p.data[k] - tr.clip.img.params[i].data[k])); });
const zs16 = zeroShot(vit, tr.clip.txt, zs, 0);
console.log(`image tower: ${vit.params.reduce((s, p) => s + p.size, 0)} params, f16 round trip max |Δw| = ${md.toExponential(2)}, zero-shot after round trip ${(zs16 * 100).toFixed(1)}%`);

const W = { vit: { cfg: P07.CLIP_CFG.img, steps: C.steps, recipe: C, zs: zsAcc.map((a) => +a.toFixed(4)), curve, b64 } };
const src = fs.readFileSync(file, 'utf8');
const block = `// <P07_W> 离线训练好的权重（tools/p10_train.js 生成，请勿手改）
const P07_W = {
    // CLIP 图像塔：P07.clipRecipe(PAGE_SEED) 配方训练 ${C.steps} 步（= Step 04 默认设置训练到终点），float16 小端序 base64
    vit: { cfg: ${JSON.stringify(W.vit.cfg)}, steps: ${C.steps}, zs: ${JSON.stringify(W.vit.zs)}, curve: ${JSON.stringify(curve)},
        b64: '${b64}' },
};
// </P07_W>`;
const out = src.replace(/\/\/ <P07_W>[\s\S]*?\/\/ <\/P07_W>/, block);
if (out === src) throw new Error('marker block not found');
fs.writeFileSync(file, out);
console.log('wrote P07_W into', file, (b64.length / 1024).toFixed(0), 'KB base64');
} else { vit = P07.createViT(P07.CLIP_CFG.img, 1); P07.loadParams(vit.params, P07_W.vit.b64); }

if (!process.argv.includes('--no-llava')) {
    const test = P07.makeCapTest(), checks = [0, 150, 300, 600, 900, 1200, 1800, 2400, 3000];
    for (const [name, enc] of [['CLIP 图像塔', vit], ['随机初始化图像塔', P07.createViT(P07.CLIP_CFG.img, 99)]]) {
        const L = P07.llavaRecipe(P07.PAGE_SEED, { vit: enc }), lt = P07.llavaTrainer(L), t1 = Date.now();
        for (let it = 0; it <= L.steps; it++) {
            if (checks.includes(it)) {
                const caps = lt.caption(test); let ex = 0, ex1 = 0, ex2 = 0;
                caps.forEach((g, i) => { if (g.text === test[i].caption) { ex++; if (test[i].single) ex1++; else ex2++; } });
                const n1 = test.filter((s) => s.single).length, n2 = test.length - n1;
                console.log(`${name} step ${it}: exact ${(ex / test.length * 100).toFixed(1)}% (单 ${(ex1 / n1 * 100).toFixed(0)}% 双 ${(ex2 / n2 * 100).toFixed(0)}%) | ${caps.slice(8, 11).map((g) => g.text).join(' | ')}`);
            }
            if (it < L.steps) lt.step();
        }
        console.log(`${name}: ${((Date.now() - t1) / L.steps).toFixed(1)} ms/step`);
    }
}
