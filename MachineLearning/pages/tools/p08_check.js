// p08_check.js — 用页面自己的代码（08_pretraining.lab.js 里的 P05 与 stages，原样执行）在 Node 里跑完整训练，
// 核对长时间行为：各步 loss、越过 n-gram 台阶的时刻、事实探针、采样快照、每帧耗时。
// 用法：node tools/p08_check.js [lr=0.003] [B=4] [warmup=100] [total=3000] [seed=20251123]
const fs = require('fs'), path = require('path'), vm = require('vm');
global.LM = require('../llm_common.js');
global.LM_WEIGHTS = require('../llm_weights.js');
let SPEC = null;
global.PipelineLab = { mount: (spec) => { SPEC = spec; } };
global.PL = { rng: LM.rng, clamp: LM.clamp, cmap: {}, draw: {} };
const src = fs.readFileSync(path.join(__dirname, '..', '08_pretraining.lab.js'), 'utf8');
vm.runInThisContext(src + '\n;globalThis.P05 = P05;', { filename: '08_pretraining.lab.js' });

const S = { seed: SPEC.seed, epoch: 1, frame: 0 };
SPEC.params.forEach((p) => { if (!p.action) S[p.id] = p.map ? p.map(p.value) : p.value; });
process.argv.slice(2).forEach((kv) => { const [k, v] = kv.split('='); S[k] = isNaN(+v) ? v : +v; });
const stage = (id) => SPEC.stages.find((s) => s.id === id);
['data', 'stats', 'base'].forEach((id) => { const t0 = Date.now(); stage(id).run(S, PL); console.log(`stage ${id}: ${Date.now() - t0} ms`); });
console.log('refs:', S.refs.map((r) => r.label).join(' | '), `| entropy rate ${S.G.rate.toFixed(4)} (${S.G.list.length} sentences)`);
console.log(`settings: lr ${S.lr} B ${S.B} warmup ${S.warmup} total ${S.total} spf ${S.spf} seed ${S.seed}`);

const train = stage('train'), ft = [];
const t0 = Date.now();
do { const a = performance.now(); train.run(S, PL); ft.push(performance.now() - a); S.frame++; } while (S.run.step < S.run.total);
const R = S.run;
const sorted = ft.slice(1).sort((a, b) => a - b), mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
console.log(`frames ${ft.length}: mean ${mean(ft.slice(1)).toFixed(2)} ms, median ${sorted[sorted.length >> 1].toFixed(2)}, p95 ${sorted[Math.floor(sorted.length * 0.95)].toFixed(2)}, max ${sorted[sorted.length - 1].toFixed(2)} ms; first 20 frames mean ${mean(ft.slice(1, 21)).toFixed(2)} ms; reset frame ${ft[0].toFixed(1)} ms; total ${((Date.now() - t0) / 1000).toFixed(1)} s`);
console.log(`clipped ${R.nClip} / ${R.total} steps; clipped in first 1000: ${Array.from(R.gn.subarray(0, 1000)).filter((g) => g > 1).length}`);
const show = [0, 10, 50, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 5000].filter((s) => s <= R.total);
show.forEach((s) => {
    const e = R.evals.find((q) => q.step === s); if (!e) return;
    const att = [0, 1, 2].map((j) => { let m = 0; for (let i = 0; i < 8; i++) m += e.probe[i * 3 + j]; return (m / 8).toFixed(2); });
    const n05 = Array.from(e.probe).filter((p) => p > 0.5).length;
    const ok = e.segs.filter((g) => g.ok === true).length, bad = e.segs.filter((g) => g.ok === false).length;
    console.log(`step ${String(s).padStart(4)} train-ema ${s ? R.ema[s - 1].toFixed(3) : '  —  '} val ${e.val.toFixed(3)} | P food/home/color ${att.join('/')} facts>0.5 ${String(n05).padStart(2)} | gn ${s ? R.gn[s - 1].toFixed(2) : '—'} medUpd ${s ? R.medU[s - 1].toExponential(1) : '—'} | ${ok}✓ ${bad}✗ ${e.segs.map((g) => g.text).join('')}`);
});
S.refs.slice(1).forEach((r) => { const e = R.evals.find((q) => q.val < r.y); console.log(`  crosses ${r.label.padEnd(14)} at step ${e ? e.step : '—'}`); });
// facts learned last
const last = R.evals[R.evals.length - 1], learnedAt = [];
P05.ATTRS.forEach((a, j) => LM.world.animals.forEach((an, i) => { const e = R.evals.find((q) => q.probe[i * 3 + j] > 0.9); learnedAt.push({ fact: `${an.name}${a.label}${an[a.key]}`, step: e ? e.step : Infinity, final: last.probe[i * 3 + j] }); }));
learnedAt.sort((a, b) => a.step - b.step);
console.log('first P>0.9:', learnedAt.map((f) => `${f.fact}@${f.step}`).join(' '));
console.log('final probes < 0.9:', learnedAt.filter((f) => f.final < 0.9).map((f) => `${f.fact}=${f.final.toFixed(2)}`).join(' ') || 'none');
console.log(`final: val ${last.val.toFixed(4)} (bayes ${S.bayes.mean.toFixed(4)}, base model ${LM_WEIGHTS.base.valLoss}) ms/step ${R.ms.toFixed(1)}`);
