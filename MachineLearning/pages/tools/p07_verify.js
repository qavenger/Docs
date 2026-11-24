// 核实页面 note / warn 里的说法（全部用当前代码、页面默认配方）。用法：node p07_verify.js <mode> [seed]
//   clip:tau=0.5 | clip:tau=0.01 | clip:N=4 | clip:N=16 | clip:hard=1 | hold <seed> | pipe <seed> | stage1 | probe
const { LM, P07 } = require('./p07_load.js');
const { AG } = LM;
const pct = (v) => (v * 100).toFixed(1) + '%';
const [mode, arg] = process.argv.slice(2);
if (mode === 'nosnap') P07.DATA.snap = 0;
const data = P07.evalData(), G = data.gallery, n = G.length;
function clipReport(R, tag) {
    const g = R.gal, z = R.zs, L = R.loss, tail = L.slice(-200).reduce((a, b) => a + b, 0) / 200;
    const sw = [[8, 9], [10, 11]].map(([a, b]) => `${(g.sim[a * n + a] - g.sim[a * n + b]).toFixed(2)}/${(g.sim[b * n + b] - g.sim[b * n + a]).toFixed(2)}`).join(' ');
    const zs1000 = R.zsHist.find((r) => r[0] === 1000);
    console.log(`${tag}: loss(last 200) ${tail.toFixed(3)} · gallery i2t ${pct(g.i2t)} · zero-shot@1000 ${pct(zs1000[1])} @2000 ${z.acc.map(pct).join('/')} · swap margins ${sw}`);
}
if (mode === 'nosnap') { const R = P07.newClipRun({ seed: P07.PAGE_SEED, N: 8, tau: 0.05, hard: false, hold: 'none' }, data); while (P07.clipWork(R)); clipReport(R, 'nosnap (随机位置训练 + 随机位置测试)'); }
else if (mode.startsWith('clip')) {
    const o = { seed: P07.PAGE_SEED, N: 8, tau: 0.05, hard: false, hold: 'none' };
    const [k, v] = mode.slice(5).split('='); if (k) o[k] = k === 'hard' ? v === '1' : +v;
    const R = P07.newClipRun(o, data); while (P07.clipWork(R));
    clipReport(R, mode);
} else if (mode === 'hold') {
    const seed = +(arg || P07.PAGE_SEED), R = P07.newClipRun({ seed, N: 8, tau: 0.05, hard: false, hold: 'gt_yc' }, data); while (P07.clipWork(R));
    const per = (k) => { let ok = 0, c = 0; data.zs.forEach((s, i) => { if (s.cls === k) { c++; if (R.zs.pred[0][i] === k) ok++; } }); return ok / c; };
    console.log(`hold seed ${seed}: zero-shot ${pct(R.zs.acc[0])} · 绿三角 ${pct(per(5))} · 黄圆 ${pct(per(9))}`);
} else if (mode === 'pipe') {
    const seed = +arg, cr = P07.newClipRun({ seed, N: 8, tau: 0.05, hard: false, hold: 'none' }, data); while (P07.clipWork(cr));
    const vit = P07.createViT(P07.CLIP_CFG.img, 1); P07.loadParams(vit.params, P07.exportParams(cr.tr.clip.img.params));
    const lr = P07.newLlavaRun({ seed, vit }, data); while (P07.llavaWork(lr));
    console.log(`pipe seed ${seed}: CLIP zero-shot ${pct(cr.zs.acc[0])} · LLaVA exact ${lr.accHist.filter((r) => [1200, 1800, 3000].includes(r.it)).map((r) => r.it + ':' + pct(r.exact)).join(' ')}`);
} else if (mode === 'stage1') {
    // 照搬 LLaVA 阶段 1：解码器先只在 caption 文本上预训练，再冻结，只训投影层
    const c = P07.LLAVA_CFG, lm = LM.gpt.create({ V: c.V, T: c.T, d: c.d, L: c.L, H: c.H, ff: c.ff }, 3 + P07.PAGE_SEED), rr = LM.rng(55), ids = [];
    for (let i = 0; i < 4000; i++) { const sc = P07.sampleScene(rr, { pSingle: 0.5 }); ids.push(P07.BOS, ...P07.vocab.encode(sc.caption), P07.EOS); }
    const stream = Int32Array.from(ids), opt = new LM.AdamW(lm.params, { lr: 3e-3, wd: 0.1, b2: 0.98 }); let tl = 0;
    for (let it = 0; it < 400; it++) { const b = LM.gpt.batch(stream, 8, c.T, rr); tl = LM.gpt.trainStep(lm, opt, b.x, b.y, 8, c.T, LM.cosineLR(it, { maxLR: 3e-3, minLR: 3e-4, warmup: 30, total: 400 })).loss; }
    const tr = P07.llavaTrainer(P07.llavaRecipe(P07.PAGE_SEED, { vit: P07.bakedViT(), lmInit: lm, stage1: 1000, steps: 1000 }));
    let first = 0, last = 0, gn = 0; for (let i = 0; i < 1000; i++) { const r = tr.step(); if (i < 50) first += r.loss / 50; if (i >= 950) { last += r.loss / 50; gn += r.gn / 50; } }
    const caps = tr.caption(data.capTest); const ex = caps.filter((g, i) => g.text === data.capTest[i].caption).length / caps.length;
    const common = {}; caps.forEach((g) => { common[g.text] = (common[g.text] || 0) + 1; });
    console.log(`stage1: text-only LM loss ${tl.toFixed(3)} · projector-only 1000 steps: loss first50 ${first.toFixed(3)} → last50 ${last.toFixed(3)} · proj grad norm ${gn.toFixed(3)} · exact ${pct(ex)} · most common output`, Object.entries(common).sort((a, b) => b[1] - a[1])[0]);
} else if (mode === 'probe') {
    // 监督探针：同一个 ViT（当前的 CLIP 初始化）+ 线性头，图形位置完全随机（不吸附网格）
    P07.DATA.snap = +(arg || 0);
    const vit = P07.createViT(P07.CLIP_CFG.img, 1), head = AG.param('head', 32, 12, (a) => { const r = LM.rng(3); for (let i = 0; i < a.length; i++) a[i] = 0.1 * LM.randn(r); });
    const opt = new LM.AdamW(vit.params.concat([head]), { lr: 3e-3 }), rand = LM.rng(5), er = LM.rng(99), test = [];
    for (let k = 0; k < 12; k++) for (let i = 0; i < 10; i++) test.push(P07.sampleScene(er, { single: true, cls: k }));
    const res = [];
    for (let it = 1; it <= 1800; it++) {
        const sc = []; for (let i = 0; i < 16; i++) sc.push(P07.sampleScene(rand, { single: true }));
        opt.zeroGrad(); const loss = AG.crossEntropy(AG.matmul(P07.vitFwd(vit, P07.imageBatch(sc).X, 16).z, head), Int32Array.from(sc.map((s) => s.cls))); AG.backward(loss); opt.step();
        if (it % 600 === 0) { const lg = AG.noGrad(() => AG.matmul(P07.vitFwd(vit, P07.imageBatch(test).X, test.length).z, head)); let col = 0, sh = 0; test.forEach((s, i) => { const p = LM.argmax(lg.data, i * 12, 12); if (Math.floor(p / 3) === Math.floor(s.cls / 3)) col++; if (p % 3 === s.cls % 3) sh++; }); res.push(`${it}: 颜色 ${pct(col / 120)} 形状 ${pct(sh / 120)}`); }
    }
    console.log(`probe snap=${P07.DATA.snap}: ${res.join(' · ')}`);
}
