const { P15 } = require('./p15_util.js');
const cams = P15.makeCameras();
// count sdf calls by wrapping via a copy of marchRef logic
const N = 2e6; let t0 = performance.now(), s = 0;
for (let i = 0; i < N; i++) s += P15.sdf(Math.sin(i) * 0.9, Math.cos(i * 1.3) * 0.8, Math.sin(i * 0.7) * 0.9);
console.log(`sdf: ${((performance.now() - t0) / N * 1e6).toFixed(0)} ns/call`, s > 0);
const c = [0, 0, 0]; t0 = performance.now();
for (let i = 0; i < N / 10; i++) P15.colorAt(Math.sin(i) * 0.9, Math.cos(i * 1.3) * 0.8, Math.sin(i * 0.7) * 0.9, c);
console.log(`colorAt: ${((performance.now() - t0) / (N / 10) * 1e6).toFixed(0)} ns/call`);
