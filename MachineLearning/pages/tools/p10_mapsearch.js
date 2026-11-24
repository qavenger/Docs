// p10_mapsearch.js — 随机搜索主地图：默认设置绕远路、slip=0 或 γ 小时抄近路、每步奖励很负时「跳陷阱」、为正时永不结束
const { P10 } = require('./p10_load.js');
const LM = require('../llm_common.js');
const W = 8, H = 6;
function routeOf(rows, g) {
    const env = P10.makeEnv(rows, { slip: g.slip, rstep: g.rstep }), res = P10.valueIteration(env, g.gamma, { tol: 1e-9, kmax: 3000 });
    let s = env.start; const cells = [];
    for (let t = 0; t < 60 && !env.term[s]; t++) { cells.push(env.cellOf[s]); s = env.mv[s * 4 + res.pi[s]]; }
    return { end: env.term[s] ? env.kind[s] : '.', len: cells.length, key: cells.join(','), V: res.V[env.start], env };
}
function reachable(rows) {
    const cell = rows.join(''), seen = new Set([cell.indexOf('S')]), q = [cell.indexOf('S')];
    while (q.length) { const i = q.pop(), x = i % W, y = (i / W) | 0; if (cell[i] === 'G' || cell[i] === 'X') continue; for (let d = 0; d < 4; d++) { const nx = x + P10.DX[d], ny = y + P10.DY[d]; if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue; const j = ny * W + nx; if (cell[j] === '#' || seen.has(j)) continue; seen.add(j); q.push(j); } }
    let free = 0; for (let i = 0; i < cell.length; i++) if (cell[i] !== '#') free++;
    return seen.size === free;
}
const rand = LM.rng(+(process.argv[2] || 1)), found = [];
for (let trial = 0; trial < +(process.argv[3] || 4000); trial++) {
    const g = Array(W * H).fill('.');
    g[5 * W + 0] = 'S';
    const gy = Math.floor(rand() * 3); g[gy * W + 7] = 'G';
    const nw = 4 + Math.floor(rand() * 4), nt = 3 + Math.floor(rand() * 3);
    const put = (ch, n) => { for (let k = 0; k < n; k++) { let i; do { i = Math.floor(rand() * W * H); } while (g[i] !== '.' || i === 4 * W + 0 || i === 5 * W + 1); g[i] = ch; } };
    put('#', nw); put('X', nt);
    const rows = Array.from({ length: H }, (_, y) => g.slice(y * W, y * W + W).join(''));
    if (!reachable(rows)) continue;
    const def = routeOf(rows, { gamma: 0.95, slip: 0.2, rstep: -0.04 });
    if (def.end !== 'G') continue;
    const s0 = routeOf(rows, { gamma: 0.95, slip: 0, rstep: -0.04 });
    const lo = routeOf(rows, { gamma: 0.8, slip: 0.2, rstep: -0.04 });
    const hi = routeOf(rows, { gamma: 0.99, slip: 0.2, rstep: -0.04 });
    const neg = routeOf(rows, { gamma: 0.95, slip: 0.2, rstep: -0.4 });
    const pos = routeOf(rows, { gamma: 0.95, slip: 0.2, rstep: 0.05 });
    let score = 0;
    if (s0.end === 'G' && s0.len < def.len) score += 2;
    if (lo.end === 'G' && lo.len < def.len) score += 3;
    if (hi.key === def.key) score += 1;
    if (neg.end === 'X') score += 2;
    if (pos.end === '.') score += 1;
    if (def.len >= 11) score += 1;
    if (score >= 9) found.push({ rows, score, d: def.len, s0: s0.len, lo: lo.len, hi: hi.len, neg: neg.end + neg.len, N: def.env.N });
}
found.sort((a, b) => b.score - a.score);
console.log(found.length, 'maps');
found.slice(0, 12).forEach((f) => console.log(JSON.stringify(f.rows), `score ${f.score} N ${f.N} default ${f.d} slip0 ${f.s0} γ.8 ${f.lo} γ.99 ${f.hi} neg ${f.neg}`));
