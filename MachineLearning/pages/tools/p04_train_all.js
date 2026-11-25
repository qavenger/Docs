// p04_train_all.js — 用页面同一份代码把 AE / VAE / VQ-VAE 各训练到上限步数，权重存到 tools/shots04/proto/models.json（后续分析脚本复用）。
// 用法：node pages/tools/p04_train_all.js [K=128] [restart=0] [vqlr=2e-3]
const fs = require('fs'), path = require('path');
const { P09, LM } = require('./p04_proto.js');
const kv = Object.fromEntries(process.argv.slice(2).map((a) => a.split('=')).map(([k, v]) => [k, +v]));
const seed = 20251123, data = P09.makeData(LM.rng(seed)); P09.initEvalEps(seed);
const OUT = path.join(__dirname, 'shots04', 'proto'); fs.mkdirSync(OUT, { recursive: true });
const runs = {
    ae: P09.makeRun('ae', { data, seed: seed + 11, cap: 3000, B: 32, lr: 5e-3 }),
    vae: P09.makeRun('vae', { data, seed: seed + 22, beta: 1, cap: 3000, B: 32, lr: 5e-3 }),
    vq: P09.makeRun('vq', { data, seed: seed + 33, K: kv.K || 128, restart: !!kv.restart, cap: 3000, B: 16, lr: kv.vqlr || 2e-3, vqArch: { C: 16 } }),
};
const out = {};
for (const [k, R] of Object.entries(runs)) {
    const t = Date.now(); while (R.step < R.cap) P09.trainStep(R); P09.evalNow(R);
    console.log(k, `${((Date.now() - t) / R.cap).toFixed(2)} ms/step`, `rec ${R.res.rec.toFixed(2)} mse ${R.res.mse.toFixed(4)}` + (k === 'vae' ? ` kl ${R.res.kl.toFixed(2)}` : '') + (k === 'vq' ? ` used ${R.res.used} ppl ${R.res.perp.toFixed(1)} restarts ${R.restarts}` : ''));
    out[k] = { o: { K: R.o.K, restart: R.o.restart }, P: R.model.P.map((p) => Array.from(p.data)) };
}
fs.writeFileSync(path.join(OUT, 'models.json'), JSON.stringify(out));
console.log('saved', path.join(OUT, 'models.json'));
