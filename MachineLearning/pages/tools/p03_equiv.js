// p03_equiv.js — 第 03 篇 Step 02：平移等变的实测数值（训练 0 / 300 / 1500 步时）
const { P08, LM } = require('./p03_proto.js');
const S = { seed: 20251123, data: P08.makeData(LM.rng(20251123)), sample: 43, dx: 2, dy: 1 };
S.main = P08.mainRuns(S);
for (const target of [0, 300, 1500]) {
    while (S.main.cnn.step < target) { P08.trainStep(S.main.cnn); P08.trainStep(S.main.mlp); }
    const e = P08.equivDemo(S), c = P08.equivCurve(S);
    console.log(`step ${target}: dIn ${e.dIn.toExponential(2)} dAll ${e.dAll.toFixed(3)} ch ${e.ch} | MLP cos ${e.hCos.toFixed(3)} rel ${e.hRel.toFixed(3)} | CNN rel ${e.fRel.toFixed(3)} aligned ${e.fRelAligned.toExponential(2)}`);
    console.log(`   curve k=${c.k.join(',')}: mlp ${c.mlp.map((v) => v.toFixed(3)).join(' ')} | cnn ${c.cnn.map((v) => v.toFixed(3)).join(' ')} | aligned ${c.cnnAligned.map((v) => v.toExponential(1)).join(' ')}`);
}
