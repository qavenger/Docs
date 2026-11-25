// p09_sweep.js — 奖励模型超参数与标注噪声（真实奖励的尺度）扫描
const path = require('path');
global.LM = require('../llm_common.js'); global.LM_WEIGHTS = require('../llm_weights.js'); global.PipelineLab = global.PL = { mount() {} };
const { P06, P06_DATA } = require(path.join(__dirname, '..', '09_alignment.lab.js')); global.P06_DATA = P06_DATA;
const ref = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), P06_DATA.sft), C = P06.buildCands(ref);
const R0 = P06.R_TRUE.slice(), K = P06.K, NP = P06.NP;
for (const scale of [1, 2]) {
    R0.forEach((v, k) => { P06.R_TRUE[k] = v * scale; });
    for (const [H, lr, it, wd] of [[16, 1e-2, 200, 0.1], [32, 3e-3, 300, 0.1], [16, 1e-2, 200, 1], [32, 1e-2, 200, 1], [8, 1e-2, 200, 0.1]]) {
        const row = [];
        for (const n of [90, 360, 1440]) {
            const L = P06.makeLabels(n), t = performance.now(), rm = P06.trainRM(C, L, { hidden: H, lr, iters: it, wd }), dt = performance.now() - t;
            const c = rm.curve[rm.curve.length - 1], bp = P06.betaPath(C, rm.rhat), pk = bp.reduce((b, x) => (x.et > b.et ? x : b), bp[0]);
            const at05 = bp.find((x) => Math.abs(x.lb - Math.log10(0.5)) < 0.03);
            row.push(`${n}: agree ${c.agree.toFixed(3)} val ${c.valAcc.toFixed(2)}/${L.ceil.toFixed(2)} peak ${(pk.et / scale).toFixed(3)}@β${pk.beta.toFixed(2)} β.5 ${(at05.et / scale).toFixed(3)} end ${(bp[bp.length - 1].et / scale).toFixed(3)} ${dt.toFixed(0)}ms`);
        }
        console.log(`scale ${scale} H${H} lr${lr} it${it} wd${wd} | ${row.join(' | ')}`);
    }
}
