// p05_eval_timing.js — 在浏览器里测每个实时训练 stage 每帧的耗时（validate_doc.mjs --eval 用；lazy stage 不计入验证器的 per-frame）
const S = PipelineLab.S, stages = PipelineLab._dbg.stages, out = {};
for (const id of ['ql', 'cliff', 'bandit', 'dqn', 'dqnGen', 'pg', 'ac', 'ppo', 'mc', 'play']) {
    const st = stages.find((x) => x.id === id);
    if (!st) { out[id] = 'missing'; continue; }
    for (let k = 0; k < 3; k++) st.run(S, PipelineLab);   // 初始化 + 预热
    const ts = [];
    for (let k = 0; k < 40; k++) { const t0 = performance.now(); st.run(S, PipelineLab); ts.push(performance.now() - t0); }
    ts.sort((a, b) => a - b);
    out[id] = { median: +ts[20].toFixed(2), p90: +ts[36].toFixed(2), max: +ts[39].toFixed(2) };
}
for (const id of ['regimes', 'pgvar', 'gae', 'world', 'walk']) {
    const st = stages.find((x) => x.id === id); const t0 = performance.now(); st.run(S, PipelineLab); out[id + ' (once)'] = +(performance.now() - t0).toFixed(1);
}
return out;
