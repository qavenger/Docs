// p14_trace.js — 打印一个 MPC 回合的真实轨迹与每步计划的想象轨迹（调试用）。用法：node tools/p14_trace.js x y [H] [N] [iters]
const { LM, P14 } = require('./p14_load.js');
const fs = require('fs');
const args = process.argv.slice(2).map(Number);
const jv = JSON.parse(fs.readFileSync(__dirname + '/p14_vae_a4.json', 'utf8')), jd = JSON.parse(fs.readFileSync(__dirname + '/p14_dyn_' + (process.env.DYN || 'ms4') + '.json', 'utf8'));
const W = { vae: LM.gpt.loadWeights(P14.makeVAE(7, jv.cfg), jv.b64), dyn: LM.gpt.loadWeights(P14.makeDyn(3, jd.cfg), jd.b64) };
const ep = P14.mpcBegin(W, [args[0], args[1]]), r = LM.rng(77), tmp = {}, img = new Float32Array(256);
const cen = (im) => { let sx = 0, sy = 0, sw = 0; for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) { const w = Math.max(0, im[j * 16 + i] - P14.BG[j * 16 + i]); sx += w * (i + 0.5); sy += w * (j + 0.5); sw += w; } return [sx / sw, sy / sw]; };
const o = { N: args[3] || 48, H: args[2] || 12, iters: args[4] || 3, elite: 8, gamma: 0.97, Tmax: 40 };
while (!ep.done) {
    P14.mpcStep(W, ep, o, r, tmp);
    const d = P14.dreamSeq(W, ep.plan.st, ep.plan.acts, tmp);
    const pts = []; for (let t = 0; t < ep.plan.acts.length; t += 3) { P14.decodeFast(W.vae, d.zs.subarray(t * 4, t * 4 + 4), 1, img, tmp); const c = cen(img); pts.push(`(${c[0].toFixed(1)},${c[1].toFixed(1)})`); }
    console.log(`t=${ep.t} a=${P14.ACT_NAME[ep.acts[ep.t - 1]]} real (${ep.s[0].toFixed(2)},${ep.s[1].toFixed(2)}) v(${ep.s[2].toFixed(2)},${ep.s[3].toFixed(2)}) planRet ${ep.plan.ret.toFixed(2)} plan ${Array.from(ep.plan.acts).map((a) => P14.ACT_NAME[a]).join('')} imag ${pts.join(' ')}`);
}
console.log('success', ep.success);
