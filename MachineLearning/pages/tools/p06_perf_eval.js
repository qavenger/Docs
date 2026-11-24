// p06_perf_eval.js — validate_doc --eval 用：在浏览器里测每个训练 stage 每帧的真实开销，以及一次性 stage 的耗时
const PL = window.PipelineLab, S = PL.S, stages = PL._dbg.stages;
const find = (id) => stages.find((s) => s.id === id);
const run = (id) => { const s = find(id); s.run(S, PL); s.valid = true; };
const time = (f) => { const t0 = performance.now(); f(); return +(performance.now() - t0).toFixed(1); };
const out = { oneTime: {}, perFrame: {} };
S.C = null; out.oneTime.cands = time(() => run('cands'));
out.oneTime.baseInfo = time(() => run('baseInfo'));
run('labels');
S.rmCache = {}; out.oneTime.rm_1440 = time(() => run('rm'));
S.paths = null; out.oneTime.rmAll_plus_paths = time(() => run('rmAll'));
out.oneTime.pistar = time(() => run('pistar'));
S.ppoKey = null; out.oneTime.ppo_two_runs = time(() => run('ppo'));
S.dpoKey = null; out.oneTime.dpoTab_two_runs = time(() => run('dpoTab'));
S.pvCache = new Map(); out.oneTime.promptView = time(() => run('promptView'));
for (const id of ['sft', 'dpoReal', 'grpo']) {
    const s = find(id);
    if (id === 'sft') S.sft = null; if (id === 'dpoReal') S.dpoR = null; if (id === 'grpo') S.grpo = null;
    const init = time(() => s.run(S, PL));
    const N = 60, t = [];
    for (let k = 0; k < N; k++) { const t0 = performance.now(); s.run(S, PL); t.push(performance.now() - t0); }
    t.sort((a, b) => a - b);
    out.perFrame[id] = { init_ms: init, mean_ms: +(t.reduce((a, b) => a + b, 0) / N).toFixed(2), median_ms: +t[N >> 1].toFixed(2), max_ms: +t[N - 1].toFixed(1) };
}
return out;
