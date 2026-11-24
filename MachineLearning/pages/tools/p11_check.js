const { LM, P11 } = require('./p11_load.js');
let t = performance.now(); const m = P11.loadModels(); console.log('loadModels', (performance.now() - t).toFixed(0), 'ms', Object.keys(m).join(','));
t = performance.now(); const fl = P11.lossFloor(P11.SCHED.cosine, 20); console.log('floor cosine', fl.total.toFixed(4), Array.from(fl.bins).map((v) => v.toFixed(2)).join(' '), (performance.now() - t).toFixed(0), 'ms');
t = performance.now(); const fl2 = P11.lossFloor(P11.SCHED.linear, 20); console.log('floor linear', fl2.total.toFixed(4), (performance.now() - t).toFixed(0), 'ms');
t = performance.now(); const dg = P11.densityGrid(1, 0, 1.35, 72); console.log('densityGrid 72²', (performance.now() - t).toFixed(1), 'ms');
t = performance.now(); const pa = P11.partition(P11.epsNet(m.vp), P11.SCHED.cosine, 40, 2.4, 20, 0, 5); let cnt = [0, 0, 0, 0]; pa.arm.forEach((a) => cnt[a + 1]++); console.log('partition', (performance.now() - t).toFixed(0), 'ms', cnt.join('/'));
t = performance.now(); const se = P11.scoreError(m.vp, P11.SCHED.cosine, [10, 50, 100, 200, 300, 500, 700, 900]); console.log('scoreError', (performance.now() - t).toFixed(0), 'ms', se.map((r) => `t${r.t}: ex ${r.excess.toFixed(3)} fl ${r.floor.toFixed(3)} x0 ${r.x0rmse.toFixed(3)}`).join(' | '));
t = performance.now(); { const x = P11.noiseBank(300, 7); P11.sampleVP(P11.epsNet(m.vp), P11.SCHED.cosine, x, 300, { K: 50, eta: 1, rand: LM.rng(3), keep: true }); } console.log('ddpm 300×50 keep', (performance.now() - t).toFixed(0), 'ms');
