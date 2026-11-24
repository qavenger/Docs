// p12_quick.js — 快速单线程训练试验：node pages/tools/p12_quick.js image|video [steps] [B] [lr]
const { LM, P12 } = require('./p12_load.js');
const kind = process.argv[2] || 'image', steps = +(process.argv[3] || 600), B = +(process.argv[4] || 32), lr = +(process.argv[5] || 2e-3);
const cfg = Object.assign({ d: 32, L: 2, H: 4, mlp: 4, video: kind === 'video' }, JSON.parse(process.argv[6] || '{}'));
const tag = process.argv[7] || kind;
if (process.argv[8]) Object.assign(P12.RECIPE, JSON.parse(process.argv[8]));
const m = P12.createModel(cfg, 1), ema = P12.createModel(cfg, 1); P12.copyParams(ema, m);
const opt = new LM.AdamW(m.params, { lr, b1: 0.9, b2: 0.99, wd: 0 });
const rand = LM.rng(123);
const t0 = Date.now(); let acc = 0, na = 0;
for (let it = 0; it < steps; it++) {
    opt.zeroGrad();
    const bt = P12.trainBatch(m, B, rand);
    const l = P12.lossStep(m, bt);
    LM.clipGradNorm(m.params, 1.0);
    opt.step(LM.cosineLR(it, { maxLR: lr, minLR: lr * 0.1, warmup: 100, total: steps }));
    P12.emaUpdate(ema, m, Math.min(0.995, (1 + it) / (10 + it)));
    acc += l; na++;
    if ((it + 1) % 100 === 0) { console.log(`step ${it + 1} loss ${(acc / na).toFixed(4)}  ${((Date.now() - t0) / (it + 1)).toFixed(0)} ms/step`); acc = 0; na = 0; }
}
// 评估：图像模型 → 64 帧，看单帧质量；视频模型 → 16 段，看时间一致性
function evalFrames(model, n, K) {
    const x = P12.sampleAll(model, { B: n, K, rand: LM.rng(99) });
    let one = 0, mass = 0, peak = 0; const hist = [0, 0, 0, 0];
    for (let i = 0; i < n; i++) { const s = P12.frameStats(x, i * 256); if (s.comps === 1) one++; hist[Math.min(3, s.comps)]++; mass += s.m; peak += s.peak; }
    console.log(`  K=${K}: frames with exactly 1 ball ${(one / n * 100).toFixed(0)}%  comps hist ${hist.join('/')}  mean mass ${(mass / n).toFixed(1)} (real ≈ ${(Math.PI * 2.1 * 2.1).toFixed(1)})  peak ${(peak / n).toFixed(2)}`);
    return x;
}
function evalVideos(model, n, K, o) {
    const x = P12.sampleAll(model, Object.assign({ B: n, K, rand: LM.rng(99) }, o || {}));
    let one = 0, res = 0, jump = 0, size = 0, nr = 0;
    for (let i = 0; i < n; i++) { const mm = P12.videoMetrics(x, i * P12.VD); one += mm.one; if (isFinite(mm.res)) { res += mm.res; nr++; } jump += mm.jump || 0; size += mm.size || 0; }
    console.log(`  K=${K}: one-ball ${(one / n * 100).toFixed(0)}%  phys res ${(res / Math.max(1, nr)).toFixed(3)} px  jump ${(jump / n).toFixed(2)}  size cv ${(size / n).toFixed(3)}`);
    return x;
}
function ascii(x, off) { const ch = ' .:-=+*#%@'; let s = ''; for (let y = 0; y < 16; y++) { for (let xx = 0; xx < 16; xx++) s += ch[Math.max(0, Math.min(9, Math.floor((x[off + y * 16 + xx] + 1) * 5)))]; s += '\n'; } return s; }
console.log('EMA model:');
if (!cfg.video) {
    for (const K of [4, 8, 16, 32]) evalFrames(ema, 64, K);
    const x = evalFrames(ema, 8, 16);
    const rows = [0, 1, 2, 3, 4, 5].map((i) => ascii(x, i * 256).split('\n'));
    for (let r = 0; r < 16; r++) console.log(rows.map((rr) => rr[r]).join(' | '));
} else {
    for (const K of [8, 16]) evalVideos(ema, 16, K);
}
require('fs').writeFileSync(require('path').join(__dirname, `p12_quick_${tag}.b64`), P12.exportParams(ema));
