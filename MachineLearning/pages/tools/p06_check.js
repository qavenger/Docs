// p06_check.js — 用页面里的 P06 函数在 Node 里跑一遍候选集管线与真实 LM 上的 DPO / GRPO，打印关键数字与耗时
// 用法（在 pages/ 下）：node tools/p06_check.js [quick]
const path = require('path');
global.LM = require('../llm_common.js');
global.LM_WEIGHTS = require('../llm_weights.js');
global.PipelineLab = global.PL = { mount() {} };
const { P06, P06_DATA } = require(path.join(__dirname, '..', '06_alignment.lab.js'));
global.P06_DATA = P06_DATA;
const { K, NP, TYPES, R_TRUE } = P06;
const quick = process.argv[2] === 'quick';
const ms = (f) => { const t = performance.now(); const r = f(); return [r, (performance.now() - t).toFixed(1) + ' ms']; };
const ref = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), P06_DATA.sft);
const [C, tC] = ms(() => P06.buildCands(ref));
console.log('buildCands', tC);
TYPES.forEach((t, k) => console.log(`  ${t.padEnd(5)} mean π_ref train ${LM.mean(C.prompts.filter((p) => !p.e.held).map((p) => p.pref[k])).toExponential(2)}  held ${LM.mean(C.prompts.filter((p) => p.e.held).map((p) => p.pref[k])).toExponential(2)}`));
const RMs = {};
for (const n of [90, 360, 1440]) {
    const L = P06.makeLabels(n);
    const [rm, t] = ms(() => P06.trainRM(C, L));
    RMs[n] = rm;
    const c = rm.curve[rm.curve.length - 1];
    console.log(`RM labels ${n} (${L.trn.length} train / ${L.val.length} val, val ceiling ${L.ceil.toFixed(3)}): ${t} | trn acc ${c.trnAcc.toFixed(3)} val acc ${c.valAcc.toFixed(3)} agree ${c.agree.toFixed(3)} | mean r̂ by type ${TYPES.map((tt, k) => tt + ' ' + LM.mean(C.prompts.map((_, pi) => rm.rhat[pi * K + k])).toFixed(2)).join(' ')}`);
}
for (const n of [90, 360, 1440]) {
    const [bp, t] = ms(() => P06.betaPath(C, RMs[n].rhat));
    const peak = bp.reduce((b, x) => (x.et > b.et ? x : b), bp[0]);
    console.log(`β path (${n} labels, ${t}): start E[r*] ${bp[0].et.toFixed(3)} | peak ${peak.et.toFixed(3)} at β=${peak.beta.toFixed(3)} KL=${peak.kl.toFixed(2)} | end (β=${bp[bp.length - 1].beta.toFixed(4)}) E[r*] ${bp[bp.length - 1].et.toFixed(3)} KL ${bp[bp.length - 1].kl.toFixed(2)} E[r̂] ${bp[bp.length - 1].er.toFixed(2)}`);
}
{
    const gold = P06.betaPath(C, Float64Array.from({ length: NP * K }, (_, i) => R_TRUE[i % K]));
    console.log('gold frontier: β=0.5 E[r*]', gold.find((x) => Math.abs(x.lb - Math.log10(0.5)) < 0.03)?.et.toFixed(3), ' max', Math.max(...gold.map((x) => x.et)).toFixed(3));
}
for (const beta of [1, 0.5]) for (const eps of [0.2, 1e9]) {
    const [run, t] = ms(() => P06.runPPO(C, RMs[360].rhat, { beta, eps }));
    const h = run.hist;
    console.log(`PPO β=${beta} ε=${eps > 10 ? '∞' : eps} (${t}): KL(π‖π*) ${h[0].klStar.toExponential(2)} → ${h[40].klStar.toExponential(2)} → ${h[h.length - 1].klStar.toExponential(2)} | KL(π‖π_ref) ${h[h.length - 1].klRef.toFixed(3)} (π*: ${P06.polStats(C, run.target, RMs[360].rhat).klRef.toFixed(3)}) | clip frac it1 ${h[1].clip.toFixed(3)} it20 ${h[20].clip.toFixed(3)}`);
}
for (const beta of [1]) {
    const [soft, t1] = ms(() => P06.runDPOtab(C, P06.softPairs(), beta));
    const L = P06.makeLabels(360), [hard, t2] = ms(() => P06.runDPOtab(C, L.trn.map(([pi, w, l]) => [pi, w, l, 1]), beta));
    const s = soft.hist, h = hard.hist;
    console.log(`DPO tab β=${beta}: soft labels (${t1}) KL(π‖π*(r*)) ${s[0].klStar.toExponential(2)} → ${s[60].klStar.toExponential(2)} → ${s[s.length - 1].klStar.toExponential(2)} | hard labels (${t2}) KL ${h[60].klStar.toExponential(2)} → ${h[h.length - 1].klStar.toExponential(2)} KL(π‖π_ref) ${h[h.length - 1].klRef.toFixed(2)} E[r*] ${h[h.length - 1].et.toFixed(3)}`);
}
if (!quick) {
    // real-LM GRPO
    const g = P06.grpoInit(ref, { G: 4 });
    P06.grpoEval(g);
    const held = (ev) => LM.mean(ev.P.filter((_, i) => P06.EX[i].held)), train = (ev) => LM.mean(ev.P.filter((_, i) => !P06.EX[i].held));
    let t = performance.now();
    while (g.step < P06.GRPO_STEPS) { P06.grpoStep(g); if (g.step % 20 === 0) P06.grpoEval(g); if (g.step % 100 === 0) { const ev = g.evals[g.evals.length - 1]; console.log(`GRPO G=4 step ${g.step}: held P ${held(ev).toFixed(3)} train P ${train(ev).toFixed(3)} zero-var ${g.zero} | ${((performance.now() - t) / g.step).toFixed(1)} ms/step`); } }
    // real-LM DPO
    for (const lam of [0, 1]) {
        const L = P06.makeLabels(360), d = P06.dpoRealInit(ref, C, L, { beta: 1, lam });
        P06.dpoRealEval(d);
        t = performance.now();
        while (d.step < P06.DPO_REAL_STEPS) { P06.dpoRealStep(d); if (d.step % 20 === 0) P06.dpoRealEval(d); }
        const e0 = d.evals[0], e1 = d.evals[d.evals.length - 1];
        console.log(`DPO real λ=${lam}: held P ${e0.heldP.toFixed(3)} → ${e1.heldP.toFixed(3)} train P ${e0.trainP.toFixed(3)} → ${e1.trainP.toFixed(3)} | Δlog π(正确) ${e1.dCorrect.toFixed(2)} Δlog π(张冠李戴) ${e1.dWrong.toFixed(2)} | ${((performance.now() - t) / d.step).toFixed(1)} ms/step | sample: ${d.samples.map((s) => P06.show(s.gen)).join(' / ')}`);
    }
    // SFT live timing (whole-animal mode)
    const base = LM.gpt.loadWeights(LM.gpt.create(LM_WEIGHTS.base.cfg, 1), LM_WEIGHTS.base.b64);
    const s = P06.sftInit(base, { hold: 'animal', lr: 5e-4, mix: 0.25 });
    t = performance.now();
    while (s.step < 400) { P06.sftStep(s); if (s.step % 20 === 0) P06.sftEval(s); }
    const ev = s.evals[s.evals.length - 1];
    console.log(`SFT whole-animal: held exact ${ev.heldExact.toFixed(2)} heldP ${ev.heldP.toFixed(3)} trainExact ${ev.trainExact.toFixed(2)} | ${((performance.now() - t) / 400).toFixed(1)} ms/step incl. eval | answers: ${s.answers.map((x) => P06.show(x.gen)).join(' ')}`);
}
