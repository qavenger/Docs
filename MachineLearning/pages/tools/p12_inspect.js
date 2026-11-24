// p12_inspect.js — 看视频模型的样本：每段的物理残差、跳变、x–t 切片（ASCII）。用法：node pages/tools/p12_inspect.js <ckpt.json> [n] [K] [known]
const fs = require('fs');
const { LM, P12 } = require('./p12_load.js');
const ck = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const n = +(process.argv[3] || 12), K = +(process.argv[4] || 16), known = +(process.argv[5] || 0);
const m = P12.createModel(ck.cfg, 1); P12.loadParams(m, ck.b64);
const rr = LM.rng(99), cond = new Float32Array(n * P12.VD), balls = [];
for (let b = 0; b < n; b++) { const bl = P12.sampleBall(rr); balls.push(bl); P12.renderVideo(bl, cond, b * P12.VD); }
const t0 = Date.now();
const x = P12.sampleAll(m, { B: n, K, rand: LM.rng(5), known, cond });
console.log(`sampled ${n} videos (K=${K}, known=${known}) in ${Date.now() - t0} ms`);
const ch = ' .:-=+*#%@';
const res = [];
for (let i = 0; i < n; i++) {
    const mm = P12.videoMetrics(x, i * P12.VD); res.push(mm.res);
    const kx = P12.kymo(x, i * P12.VD, 8, 'x');
    const line = []; for (let t = 0; t < 8; t++) { let s = ''; for (let j = 0; j < 16; j++) s += ch[Math.max(0, Math.min(9, Math.floor((kx[t * 16 + j] + 1) * 5)))]; line.push(s); }
    console.log(`#${i} res ${mm.res.toFixed(2)} jump ${mm.jump.toFixed(2)} one ${mm.one.toFixed(2)} size ${mm.size.toFixed(2)} | cx ${Array.from(mm.cx).map((v) => v.toFixed(1)).join(' ')}`);
    if (i < 6) console.log('   x–t: ' + line.join(' | '));
}
res.sort((a, b) => a - b);
console.log(`residual median ${res[Math.floor(n / 2)].toFixed(2)}, quartiles ${res[Math.floor(n / 4)].toFixed(2)} / ${res[Math.floor(3 * n / 4)].toFixed(2)}`);
