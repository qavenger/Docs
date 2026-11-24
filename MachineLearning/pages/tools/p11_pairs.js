// reflow 用的配对：z ~ N(0, I) → x₀ = FM 模型的 ODE 解（欧拉 K 步）。写到 tools/p11_ckpt/<name>.pairs.json
const fs = require('fs'), path = require('path');
const { LM, P11 } = require('./p11_load.js');
const [name, out, n = 8192, K = 100] = process.argv.slice(2);
const j = JSON.parse(fs.readFileSync(path.join(__dirname, 'p11_ckpt', name + '.json'), 'utf8')), net = P11.loadNet(j.cfg, j.ema);
const Z = new Float32Array(2 * n), r = LM.rng(4242); for (let i = 0; i < 2 * n; i++) Z[i] = LM.randn(r);
const X = Z.slice(); const t0 = performance.now(); P11.sampleFM(net, X, +n, { K: +K });
console.log('pairs', n, 'K', K, (performance.now() - t0).toFixed(0), 'ms  MMD', P11.mmdTrue(X.subarray(0, 1200), 600).toFixed(2));
fs.writeFileSync(path.join(__dirname, 'p11_ckpt', out + '.pairs.json'), JSON.stringify({ n: +n, Z: Array.from(Z), X0: Array.from(X) }));
