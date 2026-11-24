// p14_ticks.js — 在 Node 里跑页面 stage 的逐帧逻辑（P14V.setup / *Tick），检查正确性并计时。用法：node tools/p14_ticks.js [frames]
const fs = require('fs'), path = require('path');
const LM = require('../llm_common.js');
const src = fs.readFileSync(path.join(__dirname, '..', '14_world_models.lab.js'), 'utf8');
const head = src.slice(0, src.indexOf('PipelineLab.mount('));
const { P14, P14V } = new Function('LM', head + '\nreturn { P14, P14V };')(LM);
const S = { seed: 20251123, frame: 0, epoch: 1, t: 0, spf: 1, beta: 1, mode: 'ms', ctx: 4, ep: 5, wm: 'ms', H: 12, N: 32, iters: 2, kappa: -1 };
let t0 = performance.now(); P14V.setup(S); console.log('setup', (performance.now() - t0).toFixed(0), 'ms');
const frames = +(process.argv[2] || 60), tm = {};
const time = (k, f) => { const a = performance.now(); f(); tm[k] = (tm[k] || 0) + performance.now() - a; };
time('speedErr', () => P14V.speedErr(S)); time('imagine', () => P14V.imagineEp(S)); time('trap', () => P14V.trapCalc(S));
for (let f = 0; f < frames; f++) {
    S.frame++; S.t += 1 / 60;
    time('replay', () => P14V.replayFrame(S)); time('vae', () => P14V.vaeTick(S)); time('dyn', () => P14V.dynTick(S)); time('ol', () => P14V.olTick(S));
    time('mpc', () => P14V.mpcTick(S)); time('ac', () => P14V.acTick(S)); time('pressure', () => P14V.pressureTick(S)); time('play', () => P14V.playTick(S));
}
Object.keys(tm).forEach((k) => console.log(k.padEnd(9), (tm[k] / (['speedErr', 'imagine', 'trap'].includes(k) ? 1 : frames)).toFixed(2), 'ms', ['speedErr', 'imagine', 'trap'].includes(k) ? '(once)' : '/frame'));
console.log('vae it', S.vr.it, 'sse', S.vr.h.sse.slice(-1)[0], 'dyn it', S.dr.it, 'mpc res', S.mp.res.length, S.mp.i, 'ac it', S.acr.it, 'evals', JSON.stringify(S.acr.evals), 'ol done', S.ol.done);
console.log('trap pen', S.tr.pen.toFixed(2), 'imag', S.tr.rImag.toFixed(2), 'real', S.tr.rReal.toFixed(2), 'acts', Array.from(S.tr.acts).map((a) => P14.ACT_NAME[a]).join(''));
console.log('spErr', JSON.stringify(S.spErr.rows, (k, v) => (typeof v === 'number' ? +v.toFixed(2) : v)));
console.log('ol ms', S.ol.res.ms.slice(0, 5).map((v) => v.toFixed(2)), 'stay', S.ol.res.stay.slice(0, 5).map((v) => v.toFixed(2)));
console.log('pca varRatio', S.pca.varRatio.map((v) => v.toFixed(3)), 'play steps', S.play.steps);
