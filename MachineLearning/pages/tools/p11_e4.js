// 一次性编辑：quality() 加平均半径（node tools/p11_e4.js）
const fs = require('fs'), path = require('path');
const file = path.join(__dirname, '..', '11_diffusion.lab.js');
let s = fs.readFileSync(file, 'utf8');
const a = 'function quality(X, n) { const l = logp(X, n), thr = offThreshold(); let off = 0; for (let i = 0; i < n; i++) if (l[i] < thr) off++; return { mmd: mmdTrue(X, n), off: off / n }; }';
if (!s.includes(a)) { console.error('missing'); process.exit(1); }
s = s.replace(a, () => 'function quality(X, n) { const l = logp(X, n), thr = offThreshold(); let off = 0, rr = 0; for (let i = 0; i < n; i++) { if (l[i] < thr) off++; rr += Math.hypot(X[2 * i], X[2 * i + 1]); } return { mmd: mmdTrue(X, n), off: off / n, r: rr / n }; }');
fs.writeFileSync(file, s); console.log('ok');
