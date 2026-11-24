// p14_patch_tmp.js — 一次性的批量修改脚本（改完即删）
const fs = require('fs'); const f = __dirname + '/../14_world_models.lab.js'; let s = fs.readFileSync(f, 'utf8'); let n = 0;
const rep = (a, b) => { if (!s.includes(a)) { console.error('MISSING:', a.slice(0, 120)); process.exitCode = 1; return; } s = s.split(a).join(b); n++; };
// 节点图：两行布局（上 = 训练，下 = 每一步决策的想象回路）
const pos = { collect: [0, 40], vae: [250, 0], encode: [500, 60], dyn: [750, 30], observe: [500, 330], plan: [760, 300], actor: [760, 520], env: [1010, 320], decode: [1010, 560] };
Object.entries(pos).forEach(([id, [x, y]]) => rep(`{ id: '${id}', title:`, `{ id: '${id}', x: ${x}, y: ${y}, title:`));
rep(`        title: '世界模型 · 数据流（训练 + 想象 + 规划）',`, `        title: '世界模型 · 数据流（训练 + 想象 + 规划）', height: 760,`);
// 速度分组误差：放进 resident（加载时算一次，约 0.1 s），避免滚动到 Step 04 时卡一下
rep(`        S.valSeq = { e: 3, t0: 4 };   // Step 03 / 04 固定展示的验证片段`, `        S.valSeq = { e: 3, t0: 4 };   // Step 03 / 04 固定展示的验证片段
        S.spErr = null; speedErr(S);`);
rep(`{ id: 'speedErr', lifetime: 'frame', lazy: true, run(S) { P14V.speedErr(S); } },\n`, ``);
rep(`id: 'dspeed', needs: ['speedErr'], label:`, `id: 'dspeed', label:`);
// Step 06：CEM 分布直接用规划器这一步算出的 P
rep(`if (pl) { ep.plan = { acts: pl.acts, ret: pl.ret, unc: pl.unc, st: cloneSt(ep.st) }; ep.warm = pl.acts; }`, `if (pl) { ep.plan = { acts: pl.acts, ret: pl.ret, unc: pl.unc, P: pl.P, st: cloneSt(ep.st) }; ep.warm = pl.acts; }`);
rep(`Mp.view = { pts, acts: ep.plan.acts, ret: ep.plan.ret, frame: ep.frame.slice(), traj: ep.traj.slice(), t: ep.t, i: Mp.i };`, `Mp.view = { pts, acts: ep.plan.acts, ret: ep.plan.ret, P: ep.plan.P, frame: ep.frame.slice(), traj: ep.traj.slice(), t: ep.t, i: Mp.i };`);
rep("render: (c, S) => { const Mp = S.mp, v = Mp && Mp.view; if (!v || !Mp.ep.plan) return; const pl = P14.planCEM(Mp.W, Mp.ep.plan.st, Mp.o, LM.rng(5 + v.t), null, {}); const Hz = Mp.o.H; LM.draw.heat(c, pl.P, Hz, 5, { rowLabels: Array.from({ length: Hz }, (_, j) => 'j=' + (j + 1)), colLabels: P14.ACT_NAME, map: 'inferno', range: [0, 1], digits: 2, square: false, title: '刚才那一步的规划（重算一遍）', titleX: 46 }); },", "render: (c, S) => { const Mp = S.mp, v = Mp && Mp.view; if (!v || !v.P) return; const Hz = Mp.o.H; LM.draw.heat(c, v.P, Hz, 5, { rowLabels: Array.from({ length: Hz }, (_, j) => 'j=' + (j + 1)), colLabels: P14.ACT_NAME, map: 'inferno', range: [0, 1], digits: 2, square: false, title: `起点 ${v.i + 1} · t = ${v.t} 这一步的规划`, titleX: 46 }); },");
fs.writeFileSync(f, s); console.log('replacements', n);
