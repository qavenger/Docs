// p10_seeds.js — 策略梯度三兄弟在很多个随机种子上的成功率（页面的 pgInitS / ppoInitS 配置）
const { P10, P10D, LM } = require('./p10_load.js');
const S = { seed: 20251123, gamma: 0.95, slip: 0.2, rstep: -0.04, alpha: 0.1, eps: 0.1, ucbC: 2, dqnVar: 'full', ppoLR: +(process.argv[3] || 8), ppoClip: 0.2, spf: 1 };
S.env = P10.makeEnv(P10.MAPS.main, { slip: S.slip, rstep: S.rstep }); S.vi = P10.valueIteration(S.env, S.gamma);
S.Jrand = P10.evalExact(S.env, P10.uniformPolicy(S.env), S.gamma)[S.env.start];
const NS = +(process.argv[2] || 16), J = (r) => P10.evalExact(S.env, r.pi, S.gamma)[S.env.start];
const summarize = (name, finals, mids) => console.log(`${name.padEnd(28)} success(J>−0.4) ${finals.filter((v) => v > -0.4).length}/${finals.length}  mean@mid ${LM.mean(mids).toFixed(3)}  mean@end ${LM.mean(finals).toFixed(3)}  [${finals.map((v) => v.toFixed(2)).join(' ')}]`);
for (const lr of [0.3, 1, 3]) for (const base of [false, true]) {
    const fin = [], mid = [];
    for (let k = 0; k < NS; k++) { const r = P10.pgInit(S.env, { gamma: S.gamma, maxT: 400, lr, lrV: 0.1, baseline: base }, P10D.seedOf(S, 50 + k)); for (let e = 0; e < 4000; e++) { P10.reinforceStep(r); if (e === 999) mid.push(J(r)); } fin.push(J(r)); }
    summarize(`REINFORCE lr ${lr} ${base ? '+b' : 'no b'}`, fin, mid);
}
for (const lr of [0.3, 1, 3]) {
    const fin = [], mid = [];
    for (let k = 0; k < NS; k++) { const r = P10.pgInit(S.env, { gamma: S.gamma, maxT: 400, lr, lrV: 0.1 }, P10D.seedOf(S, 50 + k)); for (let e = 0; e < 4000; e++) { P10.acEpisode(r); if (e === 499) mid.push(J(r)); } fin.push(J(r)); }
    summarize(`one-step AC lr ${lr}`, fin, mid);
}
for (const [name, ep, nmb, clip] of [['A2C', 1, 1, 1e9], ['noclip', 10, 4, 1e9], ['PPO 0.2', 10, 4, 0.2]]) {
    const fin = [], mid = [];
    for (let k = 0; k < NS; k++) { const r = P10.pgInit(S.env, { gamma: S.gamma, lam: 0.95, lr: S.ppoLR, lrV: 0.1, M: 8, maxT: 400 }, P10D.seedOf(S, 50 + k)); for (let it = 0; it < 250; it++) { P10.collectGAE(r); P10.ppoUpdate(r, ep, nmb, clip); if (it === 49) mid.push(J(r)); } fin.push(J(r)); }
    summarize(`${name} lr ${S.ppoLR} (mid = 400 ep)`, fin, mid);
}
