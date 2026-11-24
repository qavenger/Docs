const { P15, LM } = require('./p15_util.js');
const { AG } = LM;
const rand = LM.rng(3), R = 3, S = 7, raw = AG.param('raw', R * S, 4, (a) => { for (let i = 0; i < a.length; i++) a[i] = 2 * LM.randn(rand); });
const delta = new Float32Array(R * S).map(() => 0.05 + 0.3 * rand()), tgt = AG.tensor(R, 3, Float32Array.from({ length: R * 3 }, () => rand()));
// double-precision reference of the same function
function ref(rv) {
    let L = 0;
    for (let ray = 0; ray < R; ray++) { let T = 1; const C = [0, 0, 0]; for (let s = 0; s < S; s++) { const i = ray * S + s, x = rv[i * 4] - 1, sg = x > 20 ? x : Math.log1p(Math.exp(x)), a = 1 - Math.exp(-sg * delta[i]); for (let c = 0; c < 3; c++) C[c] += T * a / (1 + Math.exp(-rv[i * 4 + 1 + c])); T *= 1 - a; } for (let c = 0; c < 3; c++) L += (C[c] - tgt.data[ray * 3 + c]) ** 2; }
    return L / (R * 3);
}
raw.grad = null; AG.backward(AG.mse(P15.volRender(raw, R, S, delta), tgt));
const rv = Float64Array.from(raw.data); let worst = null;
for (let i = 0; i < raw.size; i++) {
    const v = rv[i], e = 1e-5; rv[i] = v + e; const a = ref(rv); rv[i] = v - e; const b = ref(rv); rv[i] = v;
    const num = (a - b) / (2 * e), an = raw.grad[i], rel = Math.abs(num - an) / (Math.abs(num) + Math.abs(an) + 1e-9);
    if (!worst || rel > worst.rel) worst = { i, num, an, rel };
}
console.log('worst', worst);
