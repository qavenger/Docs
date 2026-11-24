// p14_pressure.js — 优化压力：从固定起点只规划一次（开环），候选数 N 越大，想象回报越高；真实执行同一动作序列的回报呢？
const { LM, P14 } = require('./p14_load.js');
const fs = require('fs');
const load = (f, mk) => { const j = JSON.parse(fs.readFileSync(__dirname + '/' + f, 'utf8')); return LM.gpt.loadWeights(mk(j.cfg), j.b64); };
const vae = load('p14_vae_a4.json', (c) => P14.makeVAE(7, c));
for (const tag of (process.argv[2] || 'ms4,tf4,dagger').split(',')) {
    const W = { vae, dyn: load('p14_dyn_' + tag + '.json', (c) => P14.makeDyn(3, c)) }, starts = P14.evalStarts(24).slice(0, 16), H = 12, tmp = {};
    const rows = [];
    for (const N of [4, 16, 64, 256, 1024]) {
        let im = 0, re = 0, pen = 0;
        for (let si = 0; si < starts.length; si++) {
            const s = new Float32Array([starts[si][0], starts[si][1], 0, 0]), st = P14.wmStart(W, P14.render(s[0], s[1]));
            const pl = P14.planCEM(W, st, { N, H, iters: 3, elite: Math.max(2, N >> 3), gamma: 1 }, LM.rng(1000 + si), null, tmp);
            im += pl.ret;
            for (let t = 0; t < H; t++) { P14.envStep(s, pl.acts[t]); re += P14.reward(s[0], s[1]); }
        }
        rows.push(`N=${N}: imagined ${(im / starts.length).toFixed(2)} real ${(re / starts.length).toFixed(2)} gap ${((im - re) / starts.length).toFixed(2)}`);
    }
    console.log(tag, '|', rows.join(' | '));
}
