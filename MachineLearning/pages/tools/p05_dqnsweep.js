// p05_dqnsweep.js — DQN 超参数 / 消融扫描（多个随机种子），打印各检查点的 J(greedy) 与 RMS(V − V*)
const { P10, LM } = require('./p05_load.js');
const G = 0.95, env = P10.makeEnv(P10.MAPS.main, { slip: 0.2, rstep: -0.04 }), vi = P10.valueIteration(env, G);
const base = Object.assign({ feat: 'xy+rowcol', epsEnd: 0.1, gamma: G, replay: true, target: true }, require('./p05_load.js').P10D.DQN_CFG);   // 与页面相同的 DQN 配置
const configs = JSON.parse(process.argv[2] || '[{}]'), steps = +(process.argv[3] || 30000), seeds = +(process.argv[4] || 4), every = 5000;
for (const c of configs) {
    const o = Object.assign({}, base, c), rows = [];
    let ms = 0;
    for (let seed = 1; seed <= seeds; seed++) {
        const st = P10.dqnInit(env, o, seed), Qt = new Float32Array(env.N * 4), row = [];
        const t0 = Date.now();
        for (let k = 1; k <= steps; k++) {
            P10.dqnStep(st);
            if (k % every === 0) {
                P10.dqnTable(st, Qt);
                const pi = P10.greedy(env, Qt), J = P10.evalExact(env, P10.onehotPolicy(env, pi), G)[env.start];
                let err = 0, n = 0; for (let s = 0; s < env.N; s++) if (!env.term[s]) { err += (P10.maxQ(Qt, s) - vi.V[s]) ** 2; n++; }
                row.push([J, Math.sqrt(err / n)]);
            }
        }
        ms += Date.now() - t0; rows.push(row);
    }
    const K = rows[0].length;
    const line = Array.from({ length: K }, (_, k) => `${(k + 1) * every / 1000}k: J ${LM.mean(rows.map((r) => r[k][0])).toFixed(3)} [${rows.map((r) => r[k][0].toFixed(2)).join(' ')}] rms ${LM.mean(rows.map((r) => r[k][1])).toFixed(2)}`);
    console.log(JSON.stringify(c), `${(ms / seeds / steps).toFixed(3)} ms/step  (V* = ${vi.V[env.start].toFixed(3)})`);
    line.forEach((l) => console.log('   ' + l));
}
