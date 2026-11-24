// 监督探针：同一个 ViT + 线性头直接做 12 类（颜色×形状）分类，看形状能不能学出来
const { LM, P07 } = require('./p07_load.js');
const { AG } = LM;
const A = Object.fromEntries(process.argv.slice(2).map((s) => s.split('=')).map(([k, v]) => [k, isNaN(+v) ? v : +v]));
const d = A.d ?? 32, L = A.L ?? 1, B = A.B ?? 16, steps = A.steps ?? 1500, lr = A.lr ?? 3e-3;
if (A.snap != null) P07.DATA.snap = A.snap; if (A.jit != null) P07.DATA.jit = A.jit; if (A.r1a != null) P07.DATA.r1 = [A.r1a, A.r1b];
const vit = P07.createViT({ d, L, H: 4, ff: 4 * d, e: 32 }, 1);
const head = AG.param('head', 32, 12, (a) => { const r = LM.rng(3); for (let i = 0; i < a.length; i++) a[i] = 0.1 * LM.randn(r); });
const params = vit.params.concat([head]);
const opt = new LM.AdamW(params, { lr });
const rand = LM.rng(5), er = LM.rng(99), test = [];
for (let k = 0; k < 12; k++) for (let i = 0; i < 10; i++) test.push(P07.sampleScene(er, { single: true, cls: k }));
for (let it = 1; it <= steps; it++) {
    const sc = []; for (let i = 0; i < B; i++) sc.push(P07.sampleScene(rand, { single: true }));
    opt.zeroGrad();
    const z = P07.vitFwd(vit, P07.imageBatch(sc).X, B).z;
    const loss = AG.crossEntropy(AG.matmul(z, head), Int32Array.from(sc.map((s) => s.cls)));
    AG.backward(loss); opt.step();
    if (it % 300 === 0) {
        const lg = AG.noGrad(() => AG.matmul(P07.vitFwd(vit, P07.imageBatch(test).X, test.length).z, head));
        let acc = 0, sh = 0; test.forEach((s, i) => { const p = LM.argmax(lg.data, i * 12, 12); if (p === s.cls) acc++; if (p % 3 === s.cls % 3) sh++; });
        console.log(`it ${it} loss ${loss.item().toFixed(3)} acc ${(acc / 1.2).toFixed(1)}% shape ${(sh / 1.2).toFixed(1)}%`);
    }
}
