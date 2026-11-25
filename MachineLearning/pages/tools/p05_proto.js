// p05_proto.js — 第 05 篇的 Node 原型：用页面里同一份 P10 代码做实验、测耗时。用法（在 pages/ 下）：node tools/p05_proto.js <what>
const { P10, LM } = require('./p05_load.js');
const what = process.argv[2] || 'grad';
const G = 0.95, SLIP = 0.2, RSTEP = -0.04;
const env = P10.makeEnv(P10.MAPS.main, { slip: SLIP, rstep: RSTEP });
const vi = P10.valueIteration(env, G);
const now = () => Number(process.hrtime.bigint()) / 1e6;

if (what === 'grad') {
    // 1) 有限差分校验 pgExact 的梯度；2) tdLoss 的有限差分校验
    const rand = LM.rng(3), theta = Float64Array.from({ length: env.N * 4 }, () => LM.randn(rand) * 0.5);
    const ex = P10.pgExact(env, theta, G);
    let md = 0, mg = 0;
    for (let k = 0; k < env.N * 4; k += 7) {
        const e = 1e-5, tp = theta.slice(), tm = theta.slice(); tp[k] += e; tm[k] -= e;
        const fd = (P10.pgExact(env, tp, G).J - P10.pgExact(env, tm, G).J) / (2 * e);
        md = Math.max(md, Math.abs(fd - ex.grad[k])); mg = Math.max(mg, Math.abs(ex.grad[k]));
    }
    console.log(`pgExact: J = ${ex.J.toFixed(5)}, max |fd − exact| = ${md.toExponential(2)} (max |g| ${mg.toExponential(2)})`);
    const AG = LM.AG, Qp = AG.param('q', 6, 4, LM.init.normal(rand, 2)), act = Int32Array.from([0, 1, 2, 3, 1, 2]), y = Float64Array.from([0.3, -2, 1.5, 0, 4, -0.2]);
    Qp.grad = null; const L = P10.tdLoss(Qp, act, y); AG.backward(L);
    let m2 = 0; for (let k = 0; k < 24; k++) { const e = 1e-3, o = Qp.data[k]; Qp.data[k] = o + e; const lp = AG.noGrad(() => P10.tdLoss(Qp, act, y)).data[0]; Qp.data[k] = o - e; const lm = AG.noGrad(() => P10.tdLoss(Qp, act, y)).data[0]; Qp.data[k] = o; m2 = Math.max(m2, Math.abs((lp - lm) / (2 * e) - Qp.grad[k])); }
    console.log(`tdLoss: max |fd − autograd| = ${m2.toExponential(2)}`);
    console.log(`VI: V*(start) = ${vi.V[env.start].toFixed(4)}, iters = ${vi.iters}, N = ${env.N}`);
}
if (what === 'ql') {
    for (const alpha of [0.05, 0.1, 0.2, 0.5]) for (const eps of [0.1, 0.2]) {
        const Q = new Float64Array(env.N * 4), rand = LM.rng(11), o = { alpha, gamma: G, eps, maxT: 200 };
        const t0 = now(); const marks = {};
        let steps = 0;
        for (let e = 1; e <= 5000; e++) {
            const r = P10.qlEpisode(env, Q, o, rand); steps += r.len;
            if ([100, 300, 1000, 2000, 5000].includes(e)) {
                const pi = P10.greedy(env, Q), J = P10.evalExact(env, P10.onehotPolicy(env, pi), G)[env.start];
                let err = 0, n = 0, agree = 0; for (let s = 0; s < env.N; s++) if (!env.term[s]) { const v = P10.maxQ(Q, s); err += (v - vi.V[s]) ** 2; n++; if (vi.Q[s * 4 + pi[s]] >= vi.V[s] - 1e-6) agree++; }
                marks[e] = `J=${J.toFixed(3)} rmsV=${Math.sqrt(err / n).toFixed(3)} opt=${agree}/${n}`;
            }
        }
        console.log(`α=${alpha} ε=${eps}: ${(now() - t0).toFixed(1)} ms for 5000 eps (${steps} steps)  V*=${vi.V[env.start].toFixed(3)}`);
        for (const [e, m] of Object.entries(marks)) console.log(`   ep ${e}: ${m}`);
    }
}
if (what === 'cliff') {
    const cenv = P10.makeEnv(P10.MAPS.cliff, { slip: 0, rstep: -1, rGoal: -1, rCliff: -100 });
    for (const alpha of [0.1, 0.5]) {
        const R = 50, E = 500, sums = { q: new Float64Array(E), s: new Float64Array(E) }, t0 = now();
        const paths = {};
        for (let run = 0; run < R; run++) {
            for (const m of ['q', 's']) {
                const Q = new Float64Array(cenv.N * 4), rand = LM.rng(1000 + run * 2 + (m === 'q' ? 0 : 1)), o = { alpha, gamma: 1, eps: 0.1, maxT: 5000 };
                for (let e = 0; e < E; e++) { const r = (m === 'q' ? P10.qlEpisode : P10.sarsaEpisode)(cenv, Q, o, rand); sums[m][e] += r.ret / R; }
                if (run === 0) paths[m] = P10.greedyPath(cenv, Q, 60).map((s) => cenv.xy(s).join(',')).join(' ');
            }
        }
        const last = (a) => LM.mean(Array.from(a.slice(E - 100)));
        console.log(`α=${alpha}: ${(now() - t0).toFixed(0)} ms, last-100 mean online return: Q-learning ${last(sums.q).toFixed(1)}  SARSA ${last(sums.s).toFixed(1)}; first-10 ${LM.mean(Array.from(sums.q.slice(0, 10))).toFixed(0)} / ${LM.mean(Array.from(sums.s.slice(0, 10))).toFixed(0)}`);
        console.log('  Q path:', paths.q); console.log('  S path:', paths.s);
    }
}
if (what === 'bandit') {
    const T = 1000, methods = [{ kind: 'eps', eps: 0 }, { kind: 'eps', eps: 0.01 }, { kind: 'eps', eps: 0.1 }, { kind: 'ucb', c: 2 }, { kind: 'eps', eps: 0, q0: 5, alpha: 0.1 }];
    const acc = { rew: methods.map(() => new Float64Array(T)), opt: methods.map(() => new Float64Array(T)) };
    const R = 2000, t0 = now();
    for (let r = 0; r < R; r++) P10.banditRun(r + 1, T, methods, acc);
    const ms = now() - t0;
    console.log(`${R} runs × ${methods.length} methods: ${ms.toFixed(0)} ms (${(ms / R).toFixed(3)} ms/run)`);
    methods.forEach((m, i) => { const rw = acc.rew[i].map((v) => v / R), op = acc.opt[i].map((v) => v / R); console.log(JSON.stringify(m), `avg reward @1000 ${LM.mean(Array.from(rw.slice(900))).toFixed(3)}  %opt ${(100 * LM.mean(Array.from(op.slice(900)))).toFixed(1)}  @100 ${LM.mean(Array.from(rw.slice(90, 110))).toFixed(3)}`); });
}
if (what === 'dqn') {
    const variants = (process.argv[3] || 'full').split(',');
    const hidden = +(process.argv[4] || 32), steps = +(process.argv[5] || 20000), lr = +(process.argv[6] || 1e-3);
    for (const v of variants) {
        for (const seed of [1, 2]) {
            const o = { feat: process.argv[7] || "xy", hidden, lr, buffer: 5000, batch: 32, warmup: 200, sync: 200, decay: +(process.argv[8] || 4000), epsEnd: +(process.argv[9] || 0.1), gamma: G, maxT: 100, replay: v === 'full' || v === 'noTarget', target: v === 'full' || v === 'noReplay' };
            const st = P10.dqnInit(env, o, seed), t0 = now(), log = [];
            const Qt = new Float32Array(env.N * 4);
            for (let k = 1; k <= steps; k++) {
                P10.dqnStep(st);
                if (k % 2000 === 0) {
                    P10.dqnTable(st, Qt);
                    const pi = P10.greedy(env, Qt), J = P10.evalExact(env, P10.onehotPolicy(env, pi), G)[env.start];
                    let err = 0, n = 0, agree = 0; for (let s = 0; s < env.N; s++) if (!env.term[s]) { const vv = P10.maxQ(Qt, s); err += (vv - vi.V[s]) ** 2; n++; if (vi.Q[s * 4 + pi[s]] >= vi.V[s] - 1e-6) agree++; }
                    log.push(`${k}: J=${J.toFixed(3)} rms=${Math.sqrt(err / n).toFixed(3)} opt=${agree} L=${st.lastLoss.toExponential(1)}`);
                }
            }
            const ms = now() - t0;
            console.log(`${v} h=${hidden} seed ${seed}: ${(ms / steps).toFixed(3)} ms/step, eps=${st.ep}`);
            console.log('   ' + log.join('\n   '));
        }
    }
}
if (what === 'pgvar') {
    // REINFORCE 单回合估计 vs 精确梯度：无偏性、方差、余弦相似度（θ = 0：均匀随机策略；θ = 训练到一半的策略）
    const thetas = { uniform: new Float64Array(env.N * 4) };
    { const th = new Float64Array(env.N * 4), rand = LM.rng(5), ep = { S: [], A: [], R: [], D: [], T: 0 }, g = new Float64Array(env.N * 4);
      for (let it = 0; it < 300; it++) { const pi = P10.softmaxRows(th, env.N); g.fill(0); P10.sampleEpisode(env, pi, rand, 400, ep); P10.reinforceGrad(ep, pi, G, P10.evalExact(env, pi, G), g, 1); for (let k = 0; k < g.length; k++) th[k] += 0.5 * g[k]; }
      thetas.mid = th; }
    for (const [name, th] of Object.entries(thetas)) {
        const ex = P10.pgExact(env, th, G), gn = Math.hypot(...ex.grad), K = 2000, rand = LM.rng(77), ep = { S: [], A: [], R: [], D: [], T: 0 };
        const bases = { none: null, Vpi: ex.V, constJ: new Float64Array(env.N).fill(ex.J) };
        let lens = 0;
        console.log(`θ = ${name}: J = ${ex.J.toFixed(4)}, ‖∇J‖ = ${gn.toExponential(3)}`);
        for (const [bn, b] of Object.entries(bases)) {
            const mean = new Float64Array(env.N * 4), g = new Float64Array(env.N * 4); let se = 0, cosS = 0; const cs = [];
            const r2 = LM.rng(77);
            for (let k = 0; k < K; k++) {
                g.fill(0); P10.sampleEpisode(env, ex.pi, r2, 400, ep); lens += ep.T; P10.reinforceGrad(ep, ex.pi, G, b, g, 1);
                let e = 0, dot = 0, nn = 0; for (let i = 0; i < g.length; i++) { mean[i] += g[i] / K; e += (g[i] - ex.grad[i]) ** 2; dot += g[i] * ex.grad[i]; nn += g[i] * g[i]; }
                se += e / K; const c = dot / (Math.sqrt(nn) * gn + 1e-30); cs.push(c); cosS += c / K;
            }
            let bias = 0; for (let i = 0; i < mean.length; i++) bias += (mean[i] - ex.grad[i]) ** 2;
            cs.sort((a, b2) => a - b2);
            console.log(`  b=${bn.padEnd(6)}: ‖mean−∇J‖/‖∇J‖ = ${(Math.sqrt(bias) / gn).toFixed(3)} (K=${K}), E‖ĝ−∇J‖²/‖∇J‖² = ${(se / gn / gn).toFixed(1)}, cos median ${cs[K >> 1].toFixed(3)} mean ${cosS.toFixed(3)}`);
        }
        console.log(`  mean episode length ${(lens / 3 / K).toFixed(1)}`);
    }
}
if (what === 'pgtrain') {
    // REINFORCE（无基线 / 学习的基线）的学习曲线：J(θ) 精确值 vs 回合数
    const lrs = (process.argv[3] || '0.1,0.3,1').split(',').map(Number), E = +(process.argv[4] || 3000), M = +(process.argv[5] || 1);
    for (const lr of lrs) for (const useB of [false, true]) {
        const res = [];
        for (let seed = 1; seed <= 4; seed++) {
            const th = new Float64Array(env.N * 4), Vb = new Float64Array(env.N), rand = LM.rng(seed * 13), ep = { S: [], A: [], R: [], D: [], T: 0 }, g = new Float64Array(env.N * 4);
            const row = [];
            for (let e = 1; e <= E; e++) {
                const pi = P10.softmaxRows(th, env.N);
                if ((e - 1) % M === 0) g.fill(0);
                P10.sampleEpisode(env, pi, rand, 400, ep);
                const Gs = P10.reinforceGrad(ep, pi, G, useB ? Vb : null, g, 1 / M);
                if (useB) for (let t = 0; t < ep.T; t++) Vb[ep.S[t]] += 0.1 * (Gs[t] - Vb[ep.S[t]]);
                if (e % M === 0) for (let k = 0; k < g.length; k++) th[k] += lr * g[k];
                if (e % (E / 6) === 0) row.push(P10.evalExact(env, P10.softmaxRows(th, env.N), G)[env.start]);
            }
            res.push(row);
        }
        console.log(`lr=${lr} M=${M} ${useB ? 'baseline' : 'none    '}: ` + res[0].map((_, k) => `${(k + 1) * E / 6}:${LM.mean(res.map((r) => r[k])).toFixed(3)}`).join('  ') + `   (V* ${vi.V[env.start].toFixed(3)})`);
    }
}
if (what === 'ppo') {
    // A2C（1 轮）vs PPO（K 轮 × 小批量，裁剪）vs PPO 不裁剪：同样多的回合，J(θ) 精确值
    const lrs = (process.argv[3] || '0.5,1,2').split(',').map(Number), iters = +(process.argv[4] || 300), M = +(process.argv[5] || 8), seeds = 4;
    const variants = [['A2C', 1, 1, 1e9], ['PPO', 4, 4, 0.2], ['PPO-noclip', 4, 4, 1e9]];
    for (const lr of lrs) for (const [name, K, nmb, clip] of variants) {
        const res = [], t0 = Date.now(); let cf = 0, klm = 0;
        for (let seed = 1; seed <= seeds; seed++) {
            const st = P10.pgInit(env, { gamma: G, lam: 0.95, lr, lrV: 0.1, M, maxT: 400 }, seed * 17), row = [];
            for (let it = 1; it <= iters; it++) {
                P10.collectGAE(st); const u = P10.ppoUpdate(st, K, nmb, clip); cf += u.clipFrac / iters / seeds; klm += u.kl / iters / seeds;
                if (it % (iters / 6) === 0) row.push(P10.evalExact(env, st.pi, G)[env.start]);
            }
            res.push(row);
        }
        console.log(`lr=${lr} ${name.padEnd(10)} (${((Date.now() - t0) / seeds / iters).toFixed(2)} ms/iter, clip ${(100 * cf).toFixed(1)}%, KL ${klm.toFixed(3)}): ` + res[0].map((_, k) => `${(k + 1) * iters / 6 * M}ep:${LM.mean(res.map((r) => r[k])).toFixed(3)}`).join('  '));
    }
}
if (what === 'ppo2') {
    const iters = +(process.argv[3] || 200), M = 8, seeds = 6, K = +(process.argv[4] || 10), nmb = 4;
    for (const lr of (process.argv[5] || '2,4,8,16').split(',').map(Number)) for (const clip of [0.2, 1e9]) {
        const res = [], kls = []; let cf = 0, worst = [];
        for (let seed = 1; seed <= seeds; seed++) {
            const st = P10.pgInit(env, { gamma: G, lam: 0.95, lr, lrV: 0.1, M, maxT: 400 }, seed * 17), row = []; let klmax = 0;
            for (let it = 1; it <= iters; it++) {
                P10.collectGAE(st); const u = P10.ppoUpdate(st, K, nmb, clip); cf += u.clipFrac / iters / seeds; klmax = Math.max(klmax, u.kl);
                if (it % (iters / 5) === 0) row.push(P10.evalExact(env, st.pi, G)[env.start]);
            }
            res.push(row); kls.push(klmax);
        }
        const fin = res.map((r) => r[r.length - 1]);
        console.log(`K=${K} lr=${lr} ${clip > 1 ? 'noclip' : 'clip .2'} (clip ${(100 * cf).toFixed(1)}%, max KL/iter ${LM.mean(kls).toFixed(3)}): ` + res[0].map((_, k) => `${(k + 1) * iters / 5 * M}ep:${LM.mean(res.map((r) => r[k])).toFixed(3)}`).join('  ') + `   final per seed [${fin.map((v) => v.toFixed(2)).join(' ')}]`);
    }
}
if (what === 'ac') {
    const E = +(process.argv[3] || 3000), seeds = 6;
    const runs = [['REINFORCE+b lr1', (st) => P10.reinforceStep(st), { lr: 1, lrV: 0.1, baseline: true }]];
    for (const lr of (process.argv[4] || '0.1,0.3,1').split(',').map(Number)) for (const lrV of [0.1, 0.3]) runs.push([`AC lr${lr} lrV${lrV}`, (st) => P10.acEpisode(st), { lr, lrV }]);
    for (const [name, fn, oo] of runs) {
        const res = []; const t0 = Date.now();
        for (let seed = 1; seed <= seeds; seed++) {
            const st = P10.pgInit(env, Object.assign({ gamma: G, maxT: 400 }, oo), seed * 29), row = [];
            for (let e = 1; e <= E; e++) { fn(st); if (e % (E / 6) === 0) row.push(P10.evalExact(env, st.pi, G)[env.start]); }
            res.push(row);
        }
        const fin = res.map((r) => r[r.length - 1]);
        console.log(`${name.padEnd(18)} (${((Date.now() - t0) / seeds / E).toFixed(3)} ms/ep): ` + res[0].map((_, k) => `${(k + 1) * E / 6}:${LM.mean(res.map((r) => r[k])).toFixed(3)}`).join('  ') + `  final [${fin.map((v) => v.toFixed(2)).join(' ')}]`);
    }
}
if (what === 'gae') {
    // 某个训练快照（单步 AC 训练 E0 个回合后）的策略与 critic：GAE(λ) 估计的策略梯度 vs 精确 ∇J 的偏差² / 方差
    for (const E0 of (process.argv[3] || '50,200,1000').split(',').map(Number)) {
        const st = P10.pgInit(env, { gamma: G, maxT: 400, lr: 1, lrV: 0.1 }, 29);
        for (let e = 0; e < E0; e++) P10.acEpisode(st);
        const ex = P10.pgExact(env, st.theta, G), gn2 = ex.grad.reduce((s, v) => s + v * v, 0), K = 1000, lams = [0, 0.3, 0.6, 0.8, 0.9, 0.95, 1];
        let vErr = 0, nV = 0; for (let s = 0; s < env.N; s++) if (!env.term[s] && ex.d[s] > 0.01) { vErr += (st.Vc[s] - ex.V[s]) ** 2; nV++; }
        const out = [];
        for (const lam of lams) {
            const rand = LM.rng(99), ep = { S: [], A: [], R: [], D: [], T: 0 }, mean = new Float64Array(env.N * 4), sq = []; const gs = [];
            for (let k = 0; k < K; k++) {
                P10.sampleEpisode(env, ex.pi, rand, 400, ep);
                const g = new Float64Array(env.N * 4); let A = 0;
                const adv = new Array(ep.T);
                for (let t = ep.T - 1; t >= 0; t--) { const s = ep.S[t], vn = ep.D[t] ? 0 : st.Vc[ep.S[t + 1]], d = ep.R[t] + G * vn - st.Vc[s]; A = d + G * lam * A; adv[t] = A; }
                let gt = 1; for (let t = 0; t < ep.T; t++) { const s = ep.S[t], a = ep.A[t], w = gt * adv[t]; for (let b = 0; b < 4; b++) g[s * 4 + b] -= w * ex.pi[s * 4 + b]; g[s * 4 + a] += w; gt *= G; }
                for (let i = 0; i < g.length; i++) mean[i] += g[i] / K; gs.push(g);
            }
            let bias2 = 0; for (let i = 0; i < mean.length; i++) bias2 += (mean[i] - ex.grad[i]) ** 2;
            let vr = 0; for (const g of gs) for (let i = 0; i < g.length; i++) vr += (g[i] - mean[i]) ** 2 / K;
            out.push(`λ${lam}: bias² ${(bias2 / gn2).toFixed(2)} var ${(vr / gn2).toFixed(1)}`);
        }
        console.log(`after ${E0} AC episodes: J = ${ex.J.toFixed(3)}, critic RMS err (visited) ${Math.sqrt(vErr / nV).toFixed(3)}\n   ` + out.join('\n   '));
    }
}
