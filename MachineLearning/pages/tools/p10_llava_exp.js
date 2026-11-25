// LLaVA 实验：node p10_llava_exp.js layer=0 B=4 lr=3e-3 steps=2000 [rand=1] [stage1=500]
const fs = require('fs');
const { LM, P07 } = require('./p10_load.js');
const { AG } = LM;
const A = Object.fromEntries(process.argv.slice(2).map((s) => s.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const CACHE = require('os').tmpdir() + '/p10_clip_cache.bin';
function saveParams(params, f) { const n = params.reduce((s, p) => s + p.size, 0), buf = new Float32Array(n); let o = 0; params.forEach((p) => { buf.set(p.data, o); o += p.size; }); fs.writeFileSync(f, Buffer.from(buf.buffer)); }
function loadParams(params, f) { const b = fs.readFileSync(f), buf = new Float32Array(b.buffer, b.byteOffset, b.length / 4); let o = 0; params.forEach((p) => { p.data.set(buf.subarray(o, o + p.size)); o += p.size; }); }
let clipTr;
if (!A.rand) {
    clipTr = P07.clipTrainer({ N: 8, tau: 0.05, lr: 3e-3, steps: 2000, seed: 1, dataSeed: 1234 });
    if (fs.existsSync(CACHE) && !A.retrain) loadParams(clipTr.clip.params, CACHE);
    else { const t0 = Date.now(); for (let i = 0; i < 2000; i++) clipTr.step(); saveParams(clipTr.clip.params, CACHE); console.log('trained CLIP', Date.now() - t0, 'ms'); }
} else clipTr = { clip: P07.createCLIP(P07.CLIP_CFG, 1) };
const vit = clipTr.clip.img;
// zero-shot check of the encoder
{
    const er = LM.rng(999), test = []; for (let k = 0; k < 12; k++) for (let i = 0; i < 4; i++) test.push(P07.sampleScene(er, { single: true, cls: k }));
    const ie = P07.embedImages(vit, test).emb, te = P07.embedTexts(clipTr.clip.txt, Array.from({ length: 12 }, (_, k) => P07.captionOf([{ c: Math.floor(k / 3), s: k % 3 }]))).emb;
    let ok = 0; test.forEach((s, i) => { let b = 0, bv = -9; for (let k = 0; k < 12; k++) { let v = 0; for (let j = 0; j < 32; j++) v += ie.data[i * 32 + j] * te.data[k * 32 + j]; if (v > bv) { bv = v; b = k; } } if (b === s.cls) ok++; });
    console.log('encoder zero-shot', (ok / test.length * 100).toFixed(1) + '%');
}
let lmInit = null;
if (A.pre) {   // 纯文本预训练：caption 首尾相接成文本流，随机窗口做 next-token prediction
    const c = P07.LLAVA_CFG, lm = LM.gpt.create({ V: c.V, T: c.T, d: c.d, L: c.L, H: c.H, ff: c.ff }, A.seed ?? 3), rr = LM.rng(55), ids = [];
    for (let i = 0; i < 4000; i++) { const sc = P07.sampleScene(rr, { pSingle: A.pS ?? 0.5 }); ids.push(P07.BOS, ...P07.vocab.encode(sc.caption), P07.EOS); }
    const data = Int32Array.from(ids), opt = new LM.AdamW(lm.params, { lr: 3e-3, wd: 0.1, b2: 0.98 });
    let l = 0; for (let it = 0; it < A.pre; it++) { const b = LM.gpt.batch(data, 8, c.T, rr); l = LM.gpt.trainStep(lm, opt, b.x, b.y, 8, c.T, LM.cosineLR(it, { maxLR: 3e-3, minLR: 3e-4, warmup: 30, total: A.pre })).loss; }
    console.log('text-only pretrain', A.pre, 'steps, loss', l.toFixed(3));
    lmInit = lm;
}
const tr = P07.llavaTrainer({ projStd: A.ps, lmInit, dv: A.cat ? 64 : 32, vit, layer: A.cat ? [0, 1] : (A.layer ?? 0), B: A.B ?? 4, lr: A.lr ?? 3e-3, steps: A.steps ?? 2000, stage1: A.stage1 || 0, pSingle: A.pS ?? 0.5, seed: A.seed ?? 3 });
const er = LM.rng(777), test = [];
for (let i = 0; i < 48; i++) test.push(P07.sampleScene(er, { single: true }));
for (let i = 0; i < 48; i++) test.push(P07.sampleScene(er, { single: false }));
function evaluate() {
    const caps = tr.caption(test);
    let ex = 0, exS = 0, exD = 0, attr = 0, nattr = 0;
    caps.forEach((g, i) => {
        const sc = test[i], ok = g.text === sc.caption; if (ok) { ex++; if (sc.single) exS++; else exD++; }
        // attribute check: parse by regex
        const m = sc.single ? g.text.match(/^一个(..)的(圆形|方形|三角形)$/) : g.text.match(/^(..)的(圆形|方形|三角形)在(..)的(圆形|方形|三角形)左边$/);
        sc.objs.forEach((ob, j) => { nattr += 2; if (m) { if (m[1 + 2 * j] === P07.COLORS[ob.c].name) attr++; if (m[2 + 2 * j] === P07.SHAPES[ob.s]) attr++; } });
    });
    return { ex: ex / test.length, exS: exS / 48, exD: exD / 48, attr: attr / nattr, sample: caps.slice(46, 50).map((g) => g.text) };
}
const t0 = Date.now(); let ema = null;
for (let it = 1; it <= (A.steps ?? 2000); it++) {
    const r = tr.step(); ema = ema == null ? r.loss : 0.98 * ema + 0.02 * r.loss;
    if (it % (A.every ?? 250) === 0) { const e = evaluate(); console.log(`it ${it} st${tr.stage} loss ${ema.toFixed(3)} exact ${(e.ex * 100).toFixed(1)}% (1obj ${(e.exS * 100).toFixed(0)} 2obj ${(e.exD * 100).toFixed(0)}) attr ${(e.attr * 100).toFixed(1)}%  ${((Date.now() - t0) / it).toFixed(1)} ms/it  | ${e.sample.join(' | ')}`); }
}
console.log('truth:', test.slice(46, 50).map((s) => s.caption).join(' | '));
