// p12_shift.js — 采样时间网格的 shift 与步数 K 对视频样本物理残差的影响。用法：node pages/tools/p12_shift.js <ckpt.json> [n]
const fs = require('fs');
const { LM, P12 } = require('./p12_load.js');
const ck = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')), n = +(process.argv[3] || 32);
const m = P12.createModel(ck.cfg, 1); P12.loadParams(m, ck.b64);
for (const [K, shift] of [[12, 1], [12, 3], [24, 1], [24, 3], [12, 6]]) {
    const x = P12.sampleAll(m, { B: n, K, shift, rand: LM.rng(5) });
    const res = [], one = []; let lost = 0;
    for (let i = 0; i < n; i++) { const mm = P12.videoMetrics(x, i * P12.VD); if (isFinite(mm.res)) res.push(mm.res); one.push(mm.one); for (let t = 0; t < 8; t++) if (!mm.ok[t]) lost++; }
    res.sort((a, b) => a - b);
    console.log(`K=${K} shift=${shift}: residual median ${res[Math.floor(res.length / 2)].toFixed(2)} mean ${(res.reduce((a, b) => a + b, 0) / res.length).toFixed(2)} | one-ball ${(one.reduce((a, b) => a + b, 0) / n * 100).toFixed(0)}% | frames without ball ${lost}`);
}
