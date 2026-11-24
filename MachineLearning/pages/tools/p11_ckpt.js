// 训练并保存实验用 checkpoint：node p11_ckpt.js name kind=vp steps=20000 ff=4 ...（写到 tools/p11_ckpt/<name>.json）
const fs = require('fs'), path = require('path');
const name = process.argv[2], args = Object.fromEntries(process.argv.slice(3).map((a) => a.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const { LM, P11 } = require('./p11_load.js');
const data = (() => { const r = P11.sampleGMM(4096, LM.rng(1234)); return { X: r.X, lab: r.lab, n: 4096 }; })();
let pairs = null;
if (args.pairs) { const j = JSON.parse(fs.readFileSync(path.join(__dirname, 'p11_ckpt', args.pairs + '.pairs.json'), 'utf8')); pairs = { X0: Float32Array.from(j.X0), Z: Float32Array.from(j.Z), n: j.n }; }
const o = { kind: args.kind || 'vp', sched: P11.SCHED[args.sched || 'cosine'], data, pairs, width: args.W || 64, depth: args.D || 3, B: args.B || 128, lr: args.lr || 2e-3, steps: args.steps || 20000, seed: args.seed || 1, warmup: args.warmup || 200, ff: args.ff ?? 4, ema: args.ema ?? 0.999, ncls: args.ncls || 0, pUncond: args.pu ?? 0.2 };
const R = P11.trainer(o), t0 = performance.now();
for (let s = 1; s <= o.steps; s++) { R.step(); if (s % 5000 === 0) console.log(name, s, ((performance.now() - t0) / s).toFixed(1) + ' ms/step'); }
fs.mkdirSync(path.join(__dirname, 'p11_ckpt'), { recursive: true });
fs.writeFileSync(path.join(__dirname, 'p11_ckpt', name + '.json'), JSON.stringify({ cfg: R.net.cfg, o: Object.assign({}, o, { sched: args.sched || 'cosine', data: null, pairs: args.pairs || null }), raw: P11.exportNet(R.net), ema: P11.exportNet(R.ema), loss: R.lossHist.filter((_, i) => i % 50 === 0) }));
console.log('saved', name);
