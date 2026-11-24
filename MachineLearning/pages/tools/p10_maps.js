// p10_maps.js — 设计主网格世界：在不同 (γ, slip, 每步奖励) 下打印最优策略与从起点出发的路线
// 用法（在 pages/ 下）：node tools/p10_maps.js [mapJSON]
const { P10 } = require('./p10_load.js');
const rows = process.argv[2] ? JSON.parse(process.argv[2]) : P10.MAPS.main;

function show(env, res, title) {
    const lines = [];
    for (let y = 0; y < env.H; y++) {
        let s1 = '', s2 = '';
        for (let x = 0; x < env.W; x++) {
            const c = env.cell[y * env.W + x], s = env.sid[y * env.W + x];
            if (s < 0) { s1 += c === 'C' ? ' C ' : ' # '; s2 += c === 'C' ? '  C   ' : '  #   '; continue; }
            if (env.term[s]) { s1 += ' ' + c + ' '; s2 += (res.V[s] >= 0 ? ' ' : '') + res.V[s].toFixed(2) + ' '; continue; }
            s1 += ' ' + P10.ARROW[res.pi[s]] + ' '; s2 += (res.V[s] >= 0 ? ' ' : '') + res.V[s].toFixed(2) + ' ';
        }
        lines.push(s1.padEnd(env.W * 3 + 2) + ' | ' + s2);
    }
    // greedy path from start (deterministic moves = intended direction)
    let s = env.start; const path = [];
    for (let t = 0; t < 60 && !env.term[s]; t++) { const [x, y] = env.xy(s); path.push(`${x},${y}`); s = env.mv[s * 4 + res.pi[s]]; }
    const [gx, gy] = env.xy(s);
    console.log(`── ${title}: V*(start) = ${res.V[env.start].toFixed(4)}, iters ${res.iters}, path len ${path.length} → ${env.kind[s]} (${gx},${gy})`);
    console.log(lines.join('\n'));
}
const regimes = [
    { gamma: 0.95, slip: 0.2, rstep: -0.04 },
    { gamma: 0.95, slip: 0.0, rstep: -0.04 },
    { gamma: 0.8, slip: 0.2, rstep: -0.04 },
    { gamma: 0.99, slip: 0.2, rstep: -0.04 },
    { gamma: 0.95, slip: 0.2, rstep: -0.3 },
    { gamma: 0.95, slip: 0.2, rstep: -1.0 },
    { gamma: 0.95, slip: 0.2, rstep: 0.05 },
    { gamma: 0.95, slip: 0.4, rstep: -0.04 },
];
for (const g of regimes) {
    const env = P10.makeEnv(rows, { slip: g.slip, rstep: g.rstep });
    show(env, P10.valueIteration(env, g.gamma), JSON.stringify(g));
}
const env = P10.makeEnv(rows, { slip: 0.2, rstep: -0.04 });
console.log('N states', env.N);
