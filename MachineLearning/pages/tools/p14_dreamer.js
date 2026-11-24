// p14_dreamer.js — 在世界模型的想象里训练 actor-critic，再到真实环境里评估（贪心策略）。用法：node tools/p14_dreamer.js steps=3000 [key=value…]
const { LM, P14 } = require('./p14_load.js');
const fs = require('fs');
const cfg = {}; for (const a of process.argv.slice(2)) { const [k, v] = a.split('='); cfg[k] = isNaN(+v) ? v : +v; }
const jv = JSON.parse(fs.readFileSync(__dirname + '/p14_vae_' + (cfg.vae || 'a') + '.json', 'utf8')), jd = JSON.parse(fs.readFileSync(__dirname + '/p14_dyn_' + (cfg.dyn || 'ms') + '.json', 'utf8'));
const W = { vae: LM.gpt.loadWeights(P14.makeVAE(7, jv.cfg), jv.b64), dyn: LM.gpt.loadWeights(P14.makeDyn(3, jd.cfg), jd.b64) };
const ds = P14.makeDataset(256, 32, LM.rng(20251123));
const lat = P14.encodeDataset(W.vae, ds.X, ds.X.length / 256);
const rep = { Hs: P14.replayStates(W, ds, lat), lat, n: ds.X.length / 256 };
const acCfg = {}; ['B', 'Hi', 'gamma', 'lambda', 'eta', 'lrA', 'lrC', 'tau', 'Hh'].forEach((k) => { if (cfg[k] != null) acCfg[k] = cfg[k]; });
const R = P14.acBegin(W, rep, cfg.seed || 5, acCfg);
const starts = P14.evalStarts(24);
function evalGreedy() {
    const tmp = {}; let succ = 0, ret = 0; const fails = [];
    for (const st of starts) {
        const ep = P14.mpcBegin(W, st);
        while (!ep.done) P14.mpcStep(W, ep, { policy: (e) => P14.acAct(R.ac, e.st, tmp, true), Tmax: 40 }, null, tmp);
        if (ep.success) succ++; else fails.push(st.map((v) => v.toFixed(1)).join(','));
        ret += ep.ret;
    }
    return { succ: succ / starts.length, ret: ret / starts.length, fails };
}
const steps = cfg.steps || 3000; let t0 = Date.now(), acc = { ir: 0, lc: 0, ent: 0, n: 0 };
for (let it = 1; it <= steps; it++) {
    const o = P14.acUpdate(R); acc.ir += o.imagRet; acc.lc += o.lossC; acc.ent += o.entropy; acc.n++;
    if (it % 250 === 0) {
        const ms = (Date.now() - t0) / acc.n, ev = evalGreedy();
        console.log(`it ${it} imagRet ${(acc.ir / acc.n).toFixed(2)} critic ${(acc.lc / acc.n).toFixed(3)} H ${(acc.ent / acc.n).toFixed(2)} | real success ${(100 * ev.succ).toFixed(0)}% ret ${ev.ret.toFixed(1)} | ${ms.toFixed(2)} ms/upd | fails ${ev.fails.join(' ')}`);
        acc = { ir: 0, lc: 0, ent: 0, n: 0 }; t0 = Date.now();
    }
}
