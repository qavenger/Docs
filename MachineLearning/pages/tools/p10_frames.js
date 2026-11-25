// 模拟页面的帧循环：每次 work 的耗时（均值 / 最坏），评估结果是否合理
const { LM, P07 } = require('./p10_load.js');
const data = P07.evalData();
function sim(name, R, work, frames) {
    const ts = []; let k = 0;
    for (; k < frames; k++) { const t = performance.now(); if (!work(R)) break; ts.push(performance.now() - t); }
    ts.sort((a, b) => a - b);
    console.log(`${name}: ${k} works, mean ${(ts.reduce((a, b) => a + b, 0) / ts.length).toFixed(2)} ms, p95 ${ts[Math.floor(ts.length * 0.95)].toFixed(2)}, max ${ts[ts.length - 1].toFixed(2)} ms`);
}
const cr = P07.newClipRun({ seed: P07.PAGE_SEED, N: 8, tau: 0.05, hard: false, hold: 'none' }, data);
sim('clip', cr, P07.clipWork, 100000);
console.log('clip final: it', cr.tr.it, 'gal i2t', cr.gal.i2t, 't2i', cr.gal.t2i, 'pos', cr.gal.pos.toFixed(3), 'neg', cr.gal.neg.toFixed(3), 'zs acc', cr.zs.acc.map((a) => (a * 100).toFixed(1)).join('/'));
console.log('zsHist', cr.zsHist.filter((_, i) => i % 4 === 0).map((r) => r[0] + ':' + (r[1] * 100).toFixed(0)).join(' '));
console.log('galHist', cr.galHist.filter((_, i) => i % 10 === 0).map((r) => r[0] + ':' + (r[1] * 100).toFixed(0)).join(' '));
const vit = P07.bakedViT();
const lr = P07.newLlavaRun({ seed: P07.PAGE_SEED, vit }, data);
sim('llava', lr, P07.llavaWork, 100000);
console.log('llava acc', lr.accHist.map((r) => r.it + ':' + (r.exact * 100).toFixed(0)).join(' '));
console.log('gallery captions at end:', lr.caps.slice(0, 16).map((c, i) => (c === data.gallery[i].caption ? '✓' : '✗' + c)).join(' '));
