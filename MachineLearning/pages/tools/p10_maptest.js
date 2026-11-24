// p10_maptest.js — 对几张手工设计的地图检查：默认绕远、slip=0 / γ 小时抄近路、每步奖励很负时跳陷阱、为正时不结束
const { P10 } = require('./p10_load.js');
const maps = {
    A: ['........', '.##.#...', '...#..X.', '.#....XG', '.#.X#...', 'S.......'],
    B: ['........', '.###.#..', '.....X.G', '.#.#.X..', '...#....', 'S.....#.'],
    C: ['........', '.##..X..', '...#...G', '.#...X#.', '...#....', 'S.#.....'],
    D: ['........', '.#.##.#.', '.#....XG', '...#X...', '.#......', 'S..#.#..'],
    E: ['....#...', '.##...X.', '....#..G', '.#X...X.', '...#....', 'S.......'],
};
const pick = process.argv[2];
function route(rows, g) {
    const env = P10.makeEnv(rows, { slip: g.slip, rstep: g.rstep }), res = P10.valueIteration(env, g.gamma, { tol: 1e-9, kmax: 3000 });
    let s = env.start; const cells = [];
    for (let t = 0; t < 60 && !env.term[s]; t++) { cells.push(s); s = env.mv[s * 4 + res.pi[s]]; }
    return { env, res, end: env.term[s] ? env.kind[s] : '∞', len: cells.length, cells };
}
function ascii(r) {
    const { env, res } = r, on = new Set(r.cells), out = [];
    for (let y = 0; y < env.H; y++) {
        let s1 = '';
        for (let x = 0; x < env.W; x++) {
            const c = env.cell[y * env.W + x], s = env.sid[y * env.W + x];
            if (s < 0) s1 += ' # '; else if (env.term[s]) s1 += ' ' + c + ' '; else s1 += (on.has(s) ? '[' : ' ') + P10.ARROW[res.pi[s]] + (on.has(s) ? ']' : ' ');
        }
        out.push(s1);
    }
    return out;
}
const regimes = [['默认 γ.95 s.2 r-.04', { gamma: 0.95, slip: 0.2, rstep: -0.04 }], ['slip 0', { gamma: 0.95, slip: 0, rstep: -0.04 }], ['γ .8', { gamma: 0.8, slip: 0.2, rstep: -0.04 }], ['γ .9', { gamma: 0.9, slip: 0.2, rstep: -0.04 }], ['γ .99', { gamma: 0.99, slip: 0.2, rstep: -0.04 }], ['r -.4', { gamma: 0.95, slip: 0.2, rstep: -0.4 }], ['r +.05', { gamma: 0.95, slip: 0.2, rstep: 0.05 }]];
for (const [name, rows] of Object.entries(maps)) {
    if (pick && pick !== name) continue;
    const rs = regimes.map(([, g]) => route(rows, g));
    console.log(`=== ${name}  N=${rs[0].env.N}  ` + regimes.map(([n], i) => `${n}: ${rs[i].end}${rs[i].len}`).join(' | '));
    const cols = rs.map(ascii);
    for (let y = 0; y < rows.length; y++) console.log(cols.map((c) => c[y]).join('   '));
}
