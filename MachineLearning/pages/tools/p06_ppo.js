// p06_ppo.js — 表格 PPO 的收敛速度扫描（G、步长、裁剪 ε、β）
const path = require('path');
global.LM = require('../llm_common.js'); global.LM_WEIGHTS = require('../llm_weights.js'); global.PipelineLab = global.PL = { mount() {} };
const { P06, P06_DATA } = require(path.join(__dirname, '..', '06_alignment.lab.js')); global.P06_DATA = P06_DATA;
const ref = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), P06_DATA.sft), C = P06.buildCands(ref);
const rm = P06.trainRM(C, P06.makeLabels(360));
console.log('RM agree', rm.curve[rm.curve.length - 1].agree.toFixed(3));
for (const beta of [0.5, 0.2, 1]) {
    const tgt = P06.piStar(C, rm.rhat, beta), ts = P06.polStats(C, tgt, rm.rhat);
    console.log(`β=${beta}: π* KL(π*‖π_ref) ${ts.klRef.toFixed(3)} E[r̂] ${ts.er.toFixed(3)} E[r*] ${ts.et.toFixed(3)}`);
    for (const [G, lr, eps] of [[8, 1, 0.2], [16, 1, 0.2], [16, 2, 0.2], [32, 2, 0.2], [16, 2, 1e9], [32, 4, 0.2]]) {
        const t = performance.now(), run = P06.runPPO(C, rm.rhat, { beta, eps, G, lr }), dt = performance.now() - t, h = run.hist;
        console.log(`   G${G} lr${lr} ε${eps > 10 ? '∞' : eps}: KL(π‖π*) ${h[0].klStar.toExponential(1)} → it40 ${h[40].klStar.toExponential(1)} → it80 ${h[80].klStar.toExponential(1)} → end ${h[h.length - 1].klStar.toExponential(1)} | KL(π‖π_ref) ${h[h.length - 1].klRef.toFixed(3)} | E[r*] ${h[h.length - 1].et.toFixed(3)} | clip it1 ${h[1].clip.toFixed(2)} | ${dt.toFixed(0)} ms`);
    }
}
