// p14_plan.js — MPC（CEM / random shooting）在真实环境里的成功率扫描。用法：node tools/p14_plan.js dyn=ms N=48 H=4,8,12 iters=1,3
const { LM, P14 } = require('./p14_load.js');
const fs = require('fs');
const cfg = {}; for (const a of process.argv.slice(2)) { const [k, v] = a.split('='); cfg[k] = v; }
const jv = JSON.parse(fs.readFileSync(__dirname + '/p14_vae_' + (cfg.vae || 'a') + '.json', 'utf8')), jd = JSON.parse(fs.readFileSync(__dirname + '/p14_dyn_' + (cfg.dyn || 'ms') + '.json', 'utf8'));
const W = { vae: LM.gpt.loadWeights(P14.makeVAE(7, jv.cfg), jv.b64), dyn: LM.gpt.loadWeights(P14.makeDyn(3, jd.cfg), jd.b64) };
const starts = P14.evalStarts(+(cfg.n || 24));
function evalPlanner(o) {
    const tmp = {}, r = LM.rng(+(cfg.seed || 77)); let succ = 0, steps = 0, ret = 0, t0 = Date.now(), dec = 0; const fails = [];
    for (const st of starts) {
        const ep = P14.mpcBegin(W, st);
        while (!ep.done) { P14.mpcStep(W, ep, o, r, tmp); dec++; }
        if (ep.success) { succ++; steps += ep.t; } else fails.push(st.map((v) => v.toFixed(1)).join(','));
        ret += ep.ret;
    }
    return { succ: succ / starts.length, steps: steps / Math.max(1, succ), ret: ret / starts.length, ms: (Date.now() - t0) / dec, fails };
}
if (cfg.random) { const res = evalPlanner({ policy: (ep, r) => Math.floor(r() * 5), Tmax: 40 }); console.log(`random policy: success ${(100 * res.succ).toFixed(0)}% ret ${res.ret.toFixed(1)}`); }
for (const H of (cfg.H || '4,8,12').split(',').map(Number)) for (const it of (cfg.iters || '1,3').split(',').map(Number)) for (const N of (cfg.N || '48').split(',').map(Number)) {
    const res = evalPlanner({ N, H, iters: it, elite: +(cfg.elite || 8), gamma: +(cfg.g || 0.97), Tmax: 40 });
    console.log(`H=${H} N=${N} iters=${it}: success ${(100 * res.succ).toFixed(0)}% steps ${res.steps.toFixed(1)} ret ${res.ret.toFixed(1)} | ${res.ms.toFixed(2)} ms/dec | fails ${res.fails.join(' ')}`);
}
