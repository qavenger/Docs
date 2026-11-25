// p09_rmreg.js — 奖励模型的正则化：weight decay / 迭代数 / 学习率对验证损失与 r* 一致率的影响
const path = require('path');
global.LM = require('../llm_common.js'); global.LM_WEIGHTS = require('../llm_weights.js'); global.PipelineLab = global.PL = { mount() {} };
const { P06, P06_DATA } = require(path.join(__dirname, '..', '09_alignment.lab.js')); global.P06_DATA = P06_DATA;
const ref = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), P06_DATA.sft), C = P06.buildCands(ref);
for (const [lr, wd, it] of [[1e-2, 0.1, 200], [1e-2, 1, 200], [1e-2, 3, 200], [3e-3, 1, 200], [1e-2, 10, 200], [3e-2, 3, 200]]) {
    const row = [];
    for (const n of [90, 360, 1440]) {
        const L = P06.makeLabels(n), rm = P06.trainRM(C, L, { lr, wd, iters: it }), cv = rm.curve, c = cv[cv.length - 1];
        const minVal = cv.reduce((b, r) => (r.val < b.val ? r : b), cv[0]);
        const bp = P06.betaPath(C, rm.rhat), pk = bp.reduce((b, x) => (x.et > b.et ? x : b), bp[0]), at1 = bp.find((x) => Math.abs(x.lb) < 0.03);
        const spread = Math.max(...rm.rhat) - Math.min(...rm.rhat);
        row.push(`${n}: val ${cv[0].val.toFixed(2)}→min ${minVal.val.toFixed(2)}@${minVal.it}→${c.val.toFixed(2)} acc ${c.valAcc.toFixed(2)}/${L.ceil.toFixed(2)} agree ${c.agree.toFixed(3)} spread ${spread.toFixed(1)} | gold β1 ${(at1.et - bp[0].et).toFixed(3)} peak +${(pk.et - bp[0].et).toFixed(3)}@β${pk.beta.toFixed(2)} end ${(bp[bp.length - 1].et - bp[0].et).toFixed(2)}`);
    }
    console.log(`lr ${lr} wd ${wd} it ${it}\n   ${row.join('\n   ')}`);
}
