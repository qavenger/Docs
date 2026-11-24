// 10_reinforcement_learning.lab.js — 系列第 10 篇：强化学习，从价值迭代到策略梯度
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js 10_reinforcement_learning.lab.js -o 10_reinforcement_learning.html
// 页面里的每张预览都由下面 P10 命名空间里的 JS 参考实现实时算出（Node 原型脚本 tools/p10_*.js 加载的是同一份代码）。

const P10 = (() => {
    'use strict';
    // ═════════════════════════ 网格世界 MDP ═════════════════════════
    // 坐标：x = 列（向右），y = 行（向下，与屏幕一致）。动作 0 ↑ 1 → 2 ↓ 3 ←。
    const DX = [0, 1, 0, -1], DY = [-1, 0, 1, 0], ARROW = ['↑', '→', '↓', '←'], ANAME = ['上', '右', '下', '左'];
    const MAPS = {
        main: [
            '........',
            '.##..X..',
            '...#...G',
            '.#...X#.',
            '...#....',
            'S.X.....',
        ],
        cliff: ['............', '............', '............', 'SCCCCCCCCCCG'],
    };
    /**
     * 网格世界。字符：. 空地 · S 起点 · # 墙 · G 目标（终止）· X 陷阱（终止）· C 悬崖（罚 rCliff、传送回起点，不终止）。
     * 奖励记在「进入的格子」上：普通格 rstep、G → rGoal、X → rTrap、C → rCliff；撞墙原地不动，也付 rstep。
     * 随机滑动：以 1 − slip 的概率按意图移动，各以 slip/2 的概率滑向两个垂直方向（Russell & Norvig 的约定）。
     */
    function makeEnv(rows, o) {
        o = Object.assign({ slip: 0, rstep: -0.04, rGoal: 1, rTrap: -1, rCliff: -100 }, o);
        const H = rows.length, W = rows[0].length, cell = rows.join('');
        const sid = new Int32Array(W * H).fill(-1), cellOf = [];
        for (let i = 0; i < W * H; i++) if (cell[i] !== '#' && cell[i] !== 'C') { sid[i] = cellOf.length; cellOf.push(i); }
        const N = cellOf.length, start = sid[cell.indexOf('S')];
        const term = new Uint8Array(N), kind = cellOf.map((i) => cell[i]);
        for (let s = 0; s < N; s++) term[s] = kind[s] === 'G' || kind[s] === 'X' ? 1 : 0;
        // 确定性的「朝某方向走一步」表：mv = 到达状态，mr = 奖励，md = 是否终止
        const mv = new Int32Array(N * 4), mr = new Float64Array(N * 4), md = new Uint8Array(N * 4);
        for (let s = 0; s < N; s++) {
            const x = cellOf[s] % W, y = (cellOf[s] / W) | 0;
            for (let d = 0; d < 4; d++) {
                const nx = x + DX[d], ny = y + DY[d], k = s * 4 + d;
                const c = nx < 0 || ny < 0 || nx >= W || ny >= H ? '#' : cell[ny * W + nx];
                if (c === '#') { mv[k] = s; mr[k] = o.rstep; md[k] = 0; }
                else if (c === 'C') { mv[k] = start; mr[k] = o.rCliff; md[k] = 0; }
                else { const s2 = sid[ny * W + nx]; mv[k] = s2; mr[k] = c === 'G' ? o.rGoal : c === 'X' ? o.rTrap : o.rstep; md[k] = c === 'G' || c === 'X' ? 1 : 0; }
            }
        }
        // 转移表 P(s', r | s, a)：每个 (s, a) 最多 3 个结果（意图方向 + 两个垂直方向，去重合并）
        const T = [];
        for (let s = 0; s < N; s++) for (let a = 0; a < 4; a++) {
            const out = [];
            if (!term[s]) {
                const add = (d, p) => { if (p <= 0) return; const k = s * 4 + d, e = out.find((q) => q[1] === mv[k] && q[2] === mr[k]); if (e) e[0] += p; else out.push([p, mv[k], mr[k], md[k]]); };
                add(a, 1 - o.slip); add((a + 1) % 4, o.slip / 2); add((a + 3) % 4, o.slip / 2);
            }
            T.push(out);
        }
        const xy = (s) => [cellOf[s] % W, (cellOf[s] / W) | 0];
        return { rows, W, H, N, cell, sid, cellOf, kind, term, start, mv, mr, md, T, xy, slip: o.slip, o };
    }
    /** 采样一步：返回 [s', r, done]（rand 是 [0,1) 均匀随机数发生器） */
    function step(env, s, a, rand) {
        let d = a;
        if (env.slip > 0) { const u = rand(); if (u >= 1 - env.slip) d = u < 1 - env.slip / 2 ? (a + 1) % 4 : (a + 3) % 4; }
        const k = s * 4 + d;
        return [env.mv[k], env.mr[k], env.md[k]];
    }

    // ═════════════════════════ 动态规划（已知模型） ═════════════════════════
    /** 一步前瞻：Q(s,a) = Σ_{s'} P(s'|s,a)·[r + γ·V(s')]（终止状态 V = 0） */
    function qFromV(env, V, gamma, Q) {
        Q = Q || new Float64Array(env.N * 4);
        for (let s = 0; s < env.N; s++) for (let a = 0; a < 4; a++) {
            let q = 0; for (const [p, s2, r, done] of env.T[s * 4 + a]) q += p * (r + (done ? 0 : gamma * V[s2]));
            Q[s * 4 + a] = q;
        }
        return Q;
    }
    /** 贪心策略（并列时取编号小的动作；终止状态记 −1） */
    function greedy(env, Q, out) {
        out = out || new Int8Array(env.N);
        for (let s = 0; s < env.N; s++) {
            if (env.term[s]) { out[s] = -1; continue; }
            let b = 0; for (let a = 1; a < 4; a++) if (Q[s * 4 + a] > Q[s * 4 + b] + 1e-12) b = a;
            out[s] = b;
        }
        return out;
    }
    /** 价值迭代（同步更新）。记录前 keep 轮的 V_k、每轮的 ‖V_{k+1} − V_k‖∞，收敛到 tol 为止 */
    function valueIteration(env, gamma, o) {
        o = Object.assign({ tol: 1e-10, kmax: 5000, keep: 80 }, o);
        const N = env.N, Q = new Float64Array(N * 4);
        let V = new Float64Array(N);
        const Vs = [V.slice()], deltas = [];
        for (let k = 0; k < o.kmax; k++) {
            qFromV(env, V, gamma, Q);
            const Vn = new Float64Array(N); let dmax = 0;
            for (let s = 0; s < N; s++) {
                if (env.term[s]) continue;
                let m = -Infinity; for (let a = 0; a < 4; a++) m = Math.max(m, Q[s * 4 + a]);
                Vn[s] = m; dmax = Math.max(dmax, Math.abs(m - V[s]));
            }
            V = Vn; deltas.push(dmax);
            if (Vs.length <= o.keep) Vs.push(V.slice());
            if (dmax < o.tol) break;
        }
        const Qs = qFromV(env, V, gamma);
        return { V, Q: Qs, pi: greedy(env, Qs), Vs, deltas, iters: deltas.length };
    }
    /** 稠密线性方程组 A x = b（部分主元高斯消元，A 为 n×n 行主序，会被改写） */
    function solve(A, b, n) {
        for (let c = 0; c < n; c++) {
            let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(A[i * n + c]) > Math.abs(A[p * n + c])) p = i;
            if (p !== c) { for (let j = 0; j < n; j++) { const t = A[c * n + j]; A[c * n + j] = A[p * n + j]; A[p * n + j] = t; } const t = b[c]; b[c] = b[p]; b[p] = t; }
            const piv = A[c * n + c] || 1e-300;
            for (let i = c + 1; i < n; i++) { const f = A[i * n + c] / piv; if (f === 0) continue; for (let j = c; j < n; j++) A[i * n + j] -= f * A[c * n + j]; b[i] -= f * b[c]; }
        }
        const x = new Float64Array(n);
        for (let i = n - 1; i >= 0; i--) { let v = b[i]; for (let j = i + 1; j < n; j++) v -= A[i * n + j] * x[j]; x[i] = v / (A[i * n + i] || 1e-300); }
        return x;
    }
    /** 策略 π(a|s)（N×4 概率表）下的 P_π 与 r_π */
    function policyMatrix(env, pi) {
        const N = env.N, P = new Float64Array(N * N), r = new Float64Array(N), cont = new Float64Array(N * N);
        for (let s = 0; s < N; s++) {
            if (env.term[s]) continue;
            for (let a = 0; a < 4; a++) {
                const pa = pi[s * 4 + a]; if (!pa) continue;
                for (const [p, s2, rr, done] of env.T[s * 4 + a]) { r[s] += pa * p * rr; P[s * N + s2] += pa * p; if (!done) cont[s * N + s2] += pa * p; }
            }
        }
        return { P, r, cont };
    }
    /** 精确策略评估：解 (I − γ·P_π) V = r_π（终止状态的后继不计价值） */
    function evalExact(env, pi, gamma) {
        const N = env.N, { r, cont } = policyMatrix(env, pi), A = new Float64Array(N * N);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) A[i * N + j] = (i === j ? 1 : 0) - gamma * cont[i * N + j];
        return solve(A, Float64Array.from(r), N);
    }
    /** 迭代策略评估：V_{k+1}(s) = Σ_a π(a|s) Σ P·[r + γ V_k(s')]，返回前 K 轮 */
    function evalSweeps(env, pi, gamma, K) {
        const N = env.N, Q = new Float64Array(N * 4), Vs = [new Float64Array(N)], deltas = [];
        let V = Vs[0];
        for (let k = 0; k < K; k++) {
            qFromV(env, V, gamma, Q);
            const Vn = new Float64Array(N); let dmax = 0;
            for (let s = 0; s < N; s++) { if (env.term[s]) continue; let v = 0; for (let a = 0; a < 4; a++) v += pi[s * 4 + a] * Q[s * 4 + a]; Vn[s] = v; dmax = Math.max(dmax, Math.abs(v - V[s])); }
            V = Vn; Vs.push(V); deltas.push(dmax);
        }
        return { Vs, deltas };
    }
    const onehotPolicy = (env, act) => { const pi = new Float64Array(env.N * 4); for (let s = 0; s < env.N; s++) if (act[s] >= 0) pi[s * 4 + act[s]] = 1; return pi; };
    const uniformPolicy = (env) => new Float64Array(env.N * 4).fill(0.25);
    /** 策略迭代：精确评估 → 贪心改进，直到策略不再变化 */
    function policyIteration(env, gamma, act0) {
        const act = Int8Array.from(act0), seq = [];
        for (let it = 0; it < 100; it++) {
            const V = evalExact(env, onehotPolicy(env, act), gamma), Q = qFromV(env, V, gamma), before = Int8Array.from(act);
            let changed = 0;
            for (let s = 0; s < env.N; s++) {   // 改进：从当前动作出发，只有严格更好（> 1e-9）才换
                if (env.term[s]) continue;
                let b = act[s]; for (let a = 0; a < 4; a++) if (Q[s * 4 + a] > Q[s * 4 + b] + 1e-9) b = a;
                if (b !== act[s]) { act[s] = b; changed++; }
            }
            seq.push({ act: before, V, Q, changed });
            if (!changed) break;
        }
        return seq;
    }
    /** 按确定性策略 act 从 s0 走到终止（或 maxT 步）；slip > 0 时结果随机 */
    function rollout(env, policy, s0, rand, maxT) {
        const S = [s0], A = [], R = [];
        let s = s0;
        for (let t = 0; t < (maxT || 400) && !env.term[s]; t++) {
            const a = policy(s, rand), [s2, r, done] = step(env, s, a, rand);
            A.push(a); R.push(r); S.push(s2); s = s2;
            if (done) break;
        }
        return { S, A, R };
    }

    // ═════════════════════════ 无模型：TD 控制（Q-learning / SARSA） ═════════════════════════
    /** 最大 Q 的动作，并列时随机挑（全 0 的初始 Q 表上很重要：否则永远先选「上」） */
    function argmaxR(Q, o, rand) {
        let m = Q[o], b = 0, n = 1;
        for (let a = 1; a < 4; a++) { const q = Q[o + a]; if (q > m + 1e-12) { m = q; b = a; n = 1; } else if (q >= m - 1e-12 && rand() * ++n < 1) b = a; }
        return b;
    }
    const egreedy = (Q, s, eps, rand) => (rand() < eps ? Math.floor(rand() * 4) : argmaxR(Q, s * 4, rand));
    const maxQ = (Q, s) => Math.max(Q[s * 4], Q[s * 4 + 1], Q[s * 4 + 2], Q[s * 4 + 3]);
    /** Q-learning 一个回合：行为策略 ε-greedy，更新目标用 max（离策略 off-policy）。o: { alpha, gamma, eps, maxT }；path 记录走过的状态 */
    function qlEpisode(env, Q, o, rand, path) {
        let s = env.start, ret = 0, t = 0, falls = 0;
        if (path) { path.length = 0; path.push(s); }
        for (; t < o.maxT && !env.term[s]; t++) {
            const a = egreedy(Q, s, o.eps, rand), [s2, r, done] = step(env, s, a, rand), k = s * 4 + a;
            Q[k] += o.alpha * (r + (done ? 0 : o.gamma * maxQ(Q, s2)) - Q[k]);
            ret += r; if (r <= env.o.rCliff) falls++;
            s = s2; if (path) path.push(s);
        }
        return { ret, len: t, end: env.term[s] ? env.kind[s] : 'T', falls };
    }
    /** SARSA 一个回合：目标用「实际要执行的下一个动作」a'（在策略 on-policy） */
    function sarsaEpisode(env, Q, o, rand, path) {
        let s = env.start, a = egreedy(Q, s, o.eps, rand), ret = 0, t = 0, falls = 0;
        if (path) { path.length = 0; path.push(s); }
        for (; t < o.maxT && !env.term[s]; t++) {
            const [s2, r, done] = step(env, s, a, rand), a2 = done ? 0 : egreedy(Q, s2, o.eps, rand), k = s * 4 + a;
            Q[k] += o.alpha * (r + (done ? 0 : o.gamma * Q[s2 * 4 + a2]) - Q[k]);
            ret += r; if (r <= env.o.rCliff) falls++;
            s = s2; a = a2; if (path) path.push(s);
        }
        return { ret, len: t, end: env.term[s] ? env.kind[s] : 'T', falls };
    }
    /** 按 Q 的贪心动作（确定性移动）从起点走，最多 maxT 步：画「学到的路线」 */
    function greedyPath(env, Q, maxT) {
        const path = [env.start], seen = new Set([env.start]);
        let s = env.start;
        for (let t = 0; t < (maxT || 60) && !env.term[s]; t++) {
            let b = 0; for (let a = 1; a < 4; a++) if (Q[s * 4 + a] > Q[s * 4 + b]) b = a;
            s = env.mv[s * 4 + b]; path.push(s);
            if (seen.has(s) && !env.term[s]) break; seen.add(s);
        }
        return path;
    }

    // ═════════════════════════ 多臂赌博机（Sutton & Barto §2.3 的 10-armed testbed） ═════════════════════════
    /**
     * 一次独立实验：q*(a) ~ N(0, 1)，拉臂 a 的奖励 ~ N(q*(a), 1)。所有方法用同一组 q*，结果累加进 acc.rew[m][t]、acc.opt[m][t]。
     * 方法 m: { kind: 'eps', eps, q0, alpha } | { kind: 'ucb', c }；alpha 缺省 = 样本平均 1/N(a)
     */
    function banditRun(seed, T, methods, acc) {
        const K = 10, rq = LM.rng(seed), qs = new Float64Array(K);
        for (let a = 0; a < K; a++) qs[a] = LM.randn(rq);
        let best = 0; for (let a = 1; a < K; a++) if (qs[a] > qs[best]) best = a;
        const Q = new Float64Array(K), N = new Float64Array(K);
        methods.forEach((m, mi) => {
            const rand = LM.rng((seed * 31 + mi * 7 + 1) >>> 0), rew = acc.rew[mi], opt = acc.opt[mi];
            Q.fill(m.q0 || 0); N.fill(0);
            for (let t = 0; t < T; t++) {
                let a = -1;
                if (m.kind === 'ucb') {
                    for (let i = 0; i < K; i++) if (N[i] === 0) { a = i; break; }   // 没拉过的臂视为上界无穷大
                    if (a < 0) { const lt = Math.log(t + 1); let bv = -Infinity; for (let i = 0; i < K; i++) { const u = Q[i] + m.c * Math.sqrt(lt / N[i]); if (u > bv) { bv = u; a = i; } } }
                } else if (m.eps > 0 && rand() < m.eps) a = Math.floor(rand() * K);
                else { a = 0; let bv = Q[0], n = 1; for (let i = 1; i < K; i++) { if (Q[i] > bv) { bv = Q[i]; a = i; n = 1; } else if (Q[i] === bv && rand() * ++n < 1) a = i; } }
                const r = qs[a] + LM.randn(rand);
                N[a]++; Q[a] += (m.alpha || 1 / N[a]) * (r - Q[a]);
                rew[t] += r; if (a === best) opt[t]++;
            }
        });
        return { qs, best };
    }

    // ═════════════════════════ DQN：小 MLP 当 Q 函数（LM.AG 自动微分） ═════════════════════════
    const AG = LM.AG;
    /**
     * 网络的输入特征（每个状态一行，F 维）。'xy'：坐标归一化到 [−1, 1]；'rowcol'：行、列各一个 one-hot（W + H 维）；'xy+rowcol'：两者拼接。
     * 网络看到的是「位置」而不是表格下标：相邻格子共享特征，一个格子上的更新会带动周围的 Q 值。
     */
    function featAll(env, kind) {
        const F = (kind === 'xy' ? 0 : env.W + env.H) + (kind === 'rowcol' ? 0 : 2), X = new Float32Array(env.N * F);
        for (let s = 0; s < env.N; s++) {
            const [x, y] = env.xy(s), o = s * F; let k = 0;
            if (kind !== 'rowcol') { X[o] = 2 * x / (env.W - 1) - 1; X[o + 1] = 2 * y / (env.H - 1) - 1; k = 2; }
            if (kind !== 'xy') { X[o + k + x] = 1; X[o + k + env.W + y] = 1; }
        }
        return { X, F };
    }
    /** MLP 的各层（与 C++ 库的 makeLinear 相同：W [in × out] ~ N(0, 2/in)（He 初始化），b = 0）。rand = null 时全 0（给目标网络，随后整体拷贝） */
    function mlp(sizes, rand, outStd) {
        const L = [];
        for (let i = 0; i + 1 < sizes.length; i++) {
            const std = i + 2 < sizes.length || outStd == null ? Math.sqrt(2 / sizes[i]) : outStd;
            L.push({ W: AG.param('W' + i, sizes[i], sizes[i + 1], rand ? LM.init.normal(rand, std) : null), b: AG.param('b' + i, 1, sizes[i + 1]) });
        }
        return L;
    }
    const mlpParams = (L) => L.flatMap((l) => [l.W, l.b]);
    function mlpForward(L, X) { let h = X; L.forEach((l, i) => { h = AG.linear(h, l.W, l.b); if (i + 1 < L.length) h = AG.relu(h); }); return h; }
    /** 不建计算图的逐行前向（选动作、算 TD 目标、评估用）：X[n×in] → out[n×4] */
    function mlpRows(L, X, n, out, tmp) {
        let src = X, inD = L[0].W.r;
        L.forEach((l, li) => {
            const o = l.W.c, W = l.W.data, b = l.b.data, dst = li + 1 < L.length ? tmp[li & 1] : out;
            for (let i = 0; i < n; i++) for (let j = 0; j < o; j++) {
                let s = b[j]; for (let k = 0; k < inD; k++) s += src[i * inD + k] * W[k * o + j];
                dst[i * o + j] = li + 1 < L.length && s < 0 ? 0 : s;
            }
            src = dst; inD = o;
        });
        return out;
    }
    /** TD 损失 mean Huber(Q(s, a) − y)（= F.smooth_l1_loss）：自定义算子，梯度只回到被选中的那一格 */
    function tdLoss(Qt, act, y) {
        const B = Qt.r, A = Qt.c, q = Qt.data, out = AG.tensor(1, 1), d = new Float64Array(B);
        let s = 0;
        for (let i = 0; i < B; i++) { const e = q[i * A + act[i]] - y[i]; d[i] = e; s += Math.abs(e) < 1 ? 0.5 * e * e : Math.abs(e) - 0.5; }
        out.data[0] = s / B;
        return AG.track(out, [Qt], (g) => { const gq = Qt.g(); for (let i = 0; i < B; i++) { const e = d[i]; gq[i * A + act[i]] += g[0] * (Math.abs(e) < 1 ? e : Math.sign(e)) / B; } });
    }
    const copyNet = (dst, src) => dst.forEach((l, i) => { l.W.data.set(src[i].W.data); l.b.data.set(src[i].b.data); });
    /** o: { hidden, lr, buffer, batch, warmup, sync, decay, epsEnd, gamma, maxT, replay, target } */
    function dqnInit(env, o, seed) {
        const fe = featAll(env, o.feat || 'xy+rowcol'), F = fe.F, rand = LM.rng(seed), net = mlp([F, o.hidden, o.hidden, 4], rand, o.outStd), tgt = mlp([F, o.hidden, o.hidden, 4], null);
        copyNet(tgt, net);
        const C = o.replay ? o.buffer : o.batch, H = o.hidden;
        return {
            env, o, rand, net, tgt, opt: new LM.AdamW(mlpParams(net), { lr: o.lr, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0 }),
            buf: { s: new Int32Array(C), a: new Int32Array(C), r: new Float32Array(C), s2: new Int32Array(C), d: new Uint8Array(C), n: 0, i: 0, C },
            X: fe.X, F, q1: new Float32Array(4), tmp: [new Float32Array(Math.max(env.N, o.batch) * H), new Float32Array(Math.max(env.N, o.batch) * H)],
            Xb: new Float32Array(o.batch * F), Xn: new Float32Array(o.batch * F), qn: new Float32Array(o.batch * 4), ab: new Int32Array(o.batch), yb: new Float64Array(o.batch),
            s: env.start, t: 0, upd: 0, epLen: 0, epRet: 0, ep: 0, rets: [], losses: [], lastLoss: NaN, path: [env.start],
        };
    }
    /** 一个环境步 + 一次梯度更新（DQN 的标准节奏：每走一步，从回放池里抽一个 batch 训练一次） */
    function dqnStep(st) {
        const { env, o, rand, buf } = st, eps = Math.max(o.epsEnd, 1 - st.t / o.decay * (1 - o.epsEnd));
        let a;
        if (rand() < eps) a = Math.floor(rand() * 4);
        else { mlpRows(st.net, st.X.subarray(st.F * st.s, st.F * st.s + st.F), 1, st.q1, st.tmp); a = argmaxR(st.q1, 0, rand); }
        const [s2, r, done] = step(env, st.s, a, rand);
        buf.s[buf.i] = st.s; buf.a[buf.i] = a; buf.r[buf.i] = r; buf.s2[buf.i] = s2; buf.d[buf.i] = done;
        buf.i = (buf.i + 1) % buf.C; buf.n = Math.min(buf.n + 1, buf.C);
        st.t++; st.epLen++; st.epRet += r; st.path.push(s2);
        if (done || st.epLen >= o.maxT) { st.rets.push(st.epRet); st.ep++; st.s = env.start; st.epLen = 0; st.epRet = 0; st.path = [env.start]; }   // 超时截断：存的仍是 done = 0（可以自举）
        else st.s = s2;
        if (buf.n < Math.min(o.warmup, buf.C)) return;
        // 抽 batch：有回放 = 从池里均匀抽；无回放 = 就用最近的 B 条（池容量 = B，样本高度相关）
        const B = o.batch, Xb = st.Xb, Xn = st.Xn, F = st.F;
        for (let i = 0; i < B; i++) {
            const j = o.replay ? Math.floor(rand() * buf.n) : (buf.i - 1 - i + buf.C * 2) % buf.C;
            st.ab[i] = buf.a[j];
            Xb.set(st.X.subarray(F * buf.s[j], F * buf.s[j] + F), F * i);
            Xn.set(st.X.subarray(F * buf.s2[j], F * buf.s2[j] + F), F * i);
            st.yb[i] = j;   // 先记下标，下面算目标
        }
        mlpRows(o.target ? st.tgt : st.net, Xn, B, st.qn, st.tmp);   // 目标网络（或在线网络本身）给出 max_a' Q(s', a')
        for (let i = 0; i < B; i++) { const j = st.yb[i]; let m = st.qn[4 * i]; for (let k = 1; k < 4; k++) m = Math.max(m, st.qn[4 * i + k]); st.yb[i] = buf.r[j] + (buf.d[j] ? 0 : o.gamma * m); }
        st.opt.zeroGrad();
        const loss = tdLoss(mlpForward(st.net, AG.tensor(B, F, Xb)), st.ab, st.yb);
        AG.backward(loss);
        st.opt.step(o.lr);
        st.upd++; st.lastLoss = loss.data[0];
        if (o.target && st.upd % o.sync === 0) copyNet(st.tgt, st.net);
    }
    /** 当前网络在所有格子上的 Q 表 [N×4] */
    const dqnTable = (st, out) => mlpRows(st.net, st.X, st.env.N, out || new Float32Array(st.env.N * 4), st.tmp);

    // ═════════════════════════ 策略梯度：表格 softmax 策略（= one-hot 状态上的一层 nn.Linear） ═════════════════════════
    function softmaxRows(theta, N, pi) {
        pi = pi || new Float64Array(N * 4);
        for (let s = 0; s < N; s++) {
            const o = s * 4, m = Math.max(theta[o], theta[o + 1], theta[o + 2], theta[o + 3]);
            let z = 0; for (let a = 0; a < 4; a++) { const e = Math.exp(theta[o + a] - m); pi[o + a] = e; z += e; }
            for (let a = 0; a < 4; a++) pi[o + a] /= z;
        }
        return pi;
    }
    function sampleAct(pi, s, rand) { const u = rand(), o = s * 4; let c = 0; for (let a = 0; a < 3; a++) { c += pi[o + a]; if (u < c) return a; } return 3; }
    /** 折扣访问频率 d_γ(s) = Σ_t γ^t·P(s_t = s)：解 (I − γ·P_πᵀ) d = e_start */
    function discVisits(env, pi, gamma) {
        const N = env.N, { cont } = policyMatrix(env, pi), A = new Float64Array(N * N), b = new Float64Array(N);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) A[i * N + j] = (i === j ? 1 : 0) - gamma * cont[j * N + i];
        b[env.start] = 1;
        return solve(A, b, N);
    }
    /** 精确的 J(θ) = V^π(start) 与 ∇J（策略梯度定理）：∂J/∂θ[s,a] = d_γ(s)·π(a|s)·A^π(s, a) */
    function pgExact(env, theta, gamma) {
        const N = env.N, pi = softmaxRows(theta, N), V = evalExact(env, pi, gamma), Q = qFromV(env, V, gamma), d = discVisits(env, pi, gamma), grad = new Float64Array(N * 4);
        for (let s = 0; s < N; s++) if (!env.term[s]) for (let a = 0; a < 4; a++) grad[s * 4 + a] = d[s] * pi[s * 4 + a] * (Q[s * 4 + a] - V[s]);
        return { J: V[env.start], V, Q, d, grad, pi };
    }
    /** 按策略 π 采样一个回合到 ep = { S, A, R, D, T }（数组复用）。S 比 A、R 多一个（最后到达的状态） */
    function sampleEpisode(env, pi, rand, maxT, ep, s0) {
        ep = ep || { S: [], A: [], R: [], D: [], T: 0 };
        ep.S.length = ep.A.length = ep.R.length = ep.D.length = 0;
        let s = s0 == null ? env.start : s0; ep.S.push(s);
        for (let t = 0; t < maxT && !env.term[s]; t++) {
            const a = sampleAct(pi, s, rand), [s2, r, done] = step(env, s, a, rand);
            ep.A.push(a); ep.R.push(r); ep.D.push(done); ep.S.push(s2); s = s2;
        }
        ep.T = ep.A.length;
        return ep;
    }
    /** 回合的折扣回报 G_t = r_{t+1} + γ·r_{t+2} + …（倒序累加） */
    function returnsOf(ep, gamma, G) { G = G || []; G.length = ep.T; let g = 0; for (let t = ep.T - 1; t >= 0; t--) { g = ep.R[t] + gamma * g; G[t] = g; } return G; }
    /**
     * REINFORCE 估计 ĝ += w · Σ_t γ^t·(G_t − b(s_t))·∇log π(a_t|s_t)。
     * 表格 softmax 的 ∇_θ[s,·] log π(a|s) = onehot(a) − π(·|s)，只有 s 那一行非零。base: 每个状态的基线（null = 不用基线）
     */
    function reinforceGrad(ep, pi, gamma, base, g, w) {
        const G = returnsOf(ep, gamma);
        let gt = 1;
        for (let t = 0; t < ep.T; t++) {
            const s = ep.S[t], a = ep.A[t], k = w * gt * (G[t] - (base ? base[s] : 0)), o = s * 4;
            for (let b = 0; b < 4; b++) g[o + b] -= k * pi[o + b];
            g[o + a] += k;
            gt *= gamma;
        }
        return G;
    }
    /** 策略梯度家族共用的训练状态：θ（N×4 logits，初始 0 = 均匀随机策略）、critic / 基线表 Vc、随机数、回合缓冲 */
    function pgInit(env, o, seed) {
        return { env, o, theta: new Float64Array(env.N * 4), Vc: new Float64Array(env.N), pi: new Float64Array(env.N * 4).fill(0.25), rand: LM.rng(seed), ep: { S: [], A: [], R: [], D: [], T: 0 }, g: new Float64Array(env.N * 4), episodes: 0, updates: 0, B: null };
    }
    /** REINFORCE 一次更新（每回合一次）：θ += lr·Σ_t γ^t (G_t − b(s_t)) ∇log π；o.baseline 时 b = Vc，Vc 向 G_t 做蒙特卡洛回归 */
    function reinforceStep(st) {
        const { env, o, theta, Vc, g } = st;
        sampleEpisode(env, st.pi, st.rand, o.maxT, st.ep);
        g.fill(0);
        const G = reinforceGrad(st.ep, st.pi, o.gamma, o.baseline ? Vc : null, g, 1);
        if (o.baseline) for (let t = 0; t < st.ep.T; t++) { const s = st.ep.S[t]; Vc[s] += o.lrV * (G[t] - Vc[s]); }
        for (let k = 0; k < theta.length; k++) theta[k] += o.lr * g[k];
        softmaxRows(theta, env.N, st.pi);
        st.episodes++; st.updates++;
        let R = 0; for (let t = 0; t < st.ep.T; t++) R += st.ep.R[t];
        return R;
    }
    /**
     * 用当前策略采 o.M 个回合，GAE(λ) 估计优势：δ_t = r_t + γ·V(s_{t+1}) − V(s_t)，Â_t = δ_t + γλ·Â_{t+1}；回报目标 R̂_t = Â_t + V(s_t)。
     * 结果放进 st.B = { s, a, adv, ret, pold, n }（pold = 采样时的 π_old(a|s)）
     */
    function collectGAE(st) {
        const { env, o, pi, Vc, rand, ep } = st;
        const B = st.B || (st.B = { s: [], a: [], adv: [], ret: [], pold: [], n: 0, epRet: [], idx: null });
        B.n = 0; B.epRet.length = 0;
        for (let m = 0; m < o.M; m++) {
            sampleEpisode(env, pi, rand, o.maxT, ep);
            let A = 0, R = 0;
            for (let t = ep.T - 1; t >= 0; t--) {
                const s = ep.S[t], vn = ep.D[t] ? 0 : Vc[ep.S[t + 1]], d = ep.R[t] + o.gamma * vn - Vc[s], i = B.n + t;
                A = d + o.gamma * o.lam * A;   // 只有最后一步可能 done；超时截断时 vn 自举
                B.s[i] = s; B.a[i] = ep.A[t]; B.adv[i] = A; B.ret[i] = A + Vc[s]; B.pold[i] = pi[s * 4 + ep.A[t]];
                R += ep.R[t];
            }
            B.n += ep.T; B.epRet.push(R); st.episodes++;
        }
        // 优势标准化（PPO 的常规做法；A2C 这里也做，便于只比较「多轮复用 + 裁剪」）
        let mu = 0, v = 0; for (let i = 0; i < B.n; i++) mu += B.adv[i] / B.n;
        for (let i = 0; i < B.n; i++) v += (B.adv[i] - mu) ** 2 / B.n;
        const sd = Math.sqrt(v) + 1e-8; B.advN = B.advN || []; for (let i = 0; i < B.n; i++) B.advN[i] = (B.adv[i] - mu) / sd;
        return B;
    }
    /**
     * PPO 更新：同一批样本做 epochs 轮、每轮 nmb 个小批量的梯度上升，目标 L^CLIP = mean min(ρ·Â, clip(ρ, 1 − ε, 1 + ε)·Â)。
     * 被裁剪（ρ 已越界且还想往外走）的样本梯度为 0。epochs = nmb = 1 时 ρ ≡ 1，就是普通的 A2C（优势 actor-critic）。
     * critic：每轮把表格 Vc 向 R̂ 拉 lrV。返回 { clipFrac, kl, ratios }
     */
    function ppoUpdate(st, epochs, nmb, clip) {
        const { env, o, theta, Vc, g } = st, B = st.B, n = B.n, p = [0, 0, 0, 0];
        if (!B.idx || B.idx.length < n) B.idx = new Int32Array(Math.max(n, 64));
        const idx = B.idx;
        let clipped = 0;
        const rowSoft = (s) => { const q = s * 4, m = Math.max(theta[q], theta[q + 1], theta[q + 2], theta[q + 3]); let z = 0; for (let a = 0; a < 4; a++) { p[a] = Math.exp(theta[q + a] - m); z += p[a]; } for (let a = 0; a < 4; a++) p[a] /= z; };
        for (let e = 0; e < epochs; e++) {
            for (let i = 0; i < n; i++) idx[i] = i;   // 每轮一个新的随机排列（Fisher–Yates，= C++ 的 permutation(n, rng)）
            for (let i = n - 1; i > 0; i--) { const j = Math.floor(st.rand() * (i + 1)), t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
            for (let mb = 0; mb < nmb; mb++) {
                const lo = Math.floor(mb * n / nmb), hi = Math.floor((mb + 1) * n / nmb);
                g.fill(0);
                for (let k = lo; k < hi; k++) {
                    const i = idx[k], s = B.s[i], a = B.a[i], A = B.advN[i];
                    rowSoft(s);
                    const rho = p[a] / B.pold[i];
                    if ((A > 0 && rho > 1 + clip) || (A < 0 && rho < 1 - clip)) { clipped++; continue; }
                    const w = A * rho / (hi - lo), q = s * 4;   // ∇(ρ·Â) = Â·ρ·∇log π(a|s) = Â·ρ·(onehot(a) − π(·|s))
                    for (let b = 0; b < 4; b++) g[q + b] -= w * p[b];
                    g[q + a] += w;
                }
                for (let k = 0; k < theta.length; k++) theta[k] += o.lr * g[k];
                st.updates++;
            }
            for (let i = 0; i < n; i++) { const s = B.s[i]; Vc[s] += o.lrV * (B.ret[i] - Vc[s]); }
        }
        softmaxRows(theta, env.N, st.pi);
        // 新旧策略在这批样本上的概率比与近似 KL（k1 = mean log(π_old/π)）
        const ratios = B.ratios || (B.ratios = []); ratios.length = n; let kl = 0;
        for (let i = 0; i < n; i++) { const r = st.pi[B.s[i] * 4 + B.a[i]] / B.pold[i]; ratios[i] = r; kl -= Math.log(r) / n; }
        return { clipFrac: clipped / Math.max(1, epochs * n), kl, ratios };
    }
    /**
     * 单步 actor-critic（Sutton & Barto §13.5，在线、每一步都更新）：δ = r + γ·V(s') − V(s) 同时是 critic 的 TD 误差和 actor 的优势估计。
     * critic：V(s) += lrV·δ；actor：θ[s] += lr·I·δ·∇log π(a|s)，I = γ^t。返回本回合的（不折扣）奖励和
     */
    function acEpisode(st) {
        const { env, o, theta, Vc, rand, pi } = st, p = [0, 0, 0, 0];
        let s = env.start, I = 1, R = 0;
        for (let t = 0; t < o.maxT && !env.term[s]; t++) {
            const q = s * 4, m = Math.max(theta[q], theta[q + 1], theta[q + 2], theta[q + 3]);
            let z = 0; for (let a = 0; a < 4; a++) { p[a] = Math.exp(theta[q + a] - m); z += p[a]; }
            for (let a = 0; a < 4; a++) p[a] /= z;
            let a = 3; { const u = rand(); let c = 0; for (let b = 0; b < 3; b++) { c += p[b]; if (u < c) { a = b; break; } } }
            const [s2, r, done] = step(env, s, a, rand), d = r + (done ? 0 : o.gamma * Vc[s2]) - Vc[s];
            Vc[s] += o.lrV * d;
            const k = o.lr * I * d;
            for (let b = 0; b < 4; b++) theta[q + b] -= k * p[b];
            theta[q + a] += k;
            I *= o.gamma; R += r; s = s2;
        }
        softmaxRows(theta, env.N, pi);
        st.episodes++;
        return R;
    }
    return {
        DX, DY, ARROW, ANAME, MAPS, makeEnv, step, qFromV, greedy, valueIteration, solve, policyMatrix, evalExact, evalSweeps, onehotPolicy, uniformPolicy, policyIteration, rollout,
        argmaxR, egreedy, maxQ, qlEpisode, sarsaEpisode, greedyPath, banditRun,
        featAll, mlp, mlpParams, mlpForward, mlpRows, tdLoss, copyNet, dqnInit, dqnStep, dqnTable,
        softmaxRows, sampleAct, discVisits, pgExact, sampleEpisode, returnsOf, reinforceGrad,
        pgInit, reinforceStep, collectGAE, ppoUpdate, acEpisode,
    };
})();

// ═════════════════════════ C++ 代码块（与上面的 JS 参考实现逐函数对应） ═════════════════════════
// tools/check_cpp.js 做编译检查；tools/p10_cpp_run.js 把它们编译成可执行文件，用同一个种子跑一遍，与 JS 的结果逐项对数。
const P10CPP = {
    common: `
        // rl_gridworld —— 网格世界 MDP 与采样（与页面 JS 的 P10.makeEnv / P10.step 一一对应）
        const int NA = 4;                                    // 动作数：0 ↑  1 →  2 ↓  3 ←（y 轴向下，与屏幕一致）
        const int DX[NA] = {0, 1, 0, -1};
        const int DY[NA] = {-1, 0, 1, 0};

        // n 个 double。表格方法统一用双精度一维数组：V[s]、Q[s·NA + a]、π[s·NA + a]、θ[s·NA + a]
        std::vector<double> doubles(int n, double x = 0.0) { return std::vector<double>((size_t)n, x); }

        // 转移模型 P(s', r | s, a) 的一个分支
        struct Outcome { double p; int next; double r; bool done; };

        struct GridWorld {
            int W = 0, H = 0, N = 0, start = 0;              // N = 状态数（墙、悬崖格不是状态）
            double slip = 0, rCliff = -100;
            std::vector<char> kind;                          // 每个状态的格子字符：. S G（目标）X（陷阱）
            std::vector<int> cellOf;                         // 状态 → 格子下标 y·W + x
            std::vector<int> moveTo, moveDone;               // [N·NA] 朝方向 d 走一步：到达的状态、是否进入终止格
            std::vector<double> moveR;                       // [N·NA] 这一步的奖励
            std::vector<std::vector<Outcome>> P;             // [N·NA] 转移模型（动态规划用；无模型方法只能调 envStep）
        };

        bool isTerminal(const GridWorld& env, int s) { return env.kind[s] == 'G' || env.kind[s] == 'X'; }

        // 由字符地图构造 MDP：. 空地  S 起点  # 墙  G 目标（终止）  X 陷阱（终止）  C 悬崖（罚 rCliff、传送回起点，不终止）
        // 奖励记在「进入的格子」上；撞墙原地不动，也付 rStep。滑动：1 − slip 按意图走，各 slip/2 滑向两个垂直方向
        GridWorld makeGridWorld(const std::vector<std::string>& rows, double slip, double rStep, double rGoal, double rTrap, double rCliff) {
            GridWorld env;
            env.H = (int)rows.size(); env.W = (int)rows[0].size(); env.slip = slip; env.rCliff = rCliff;
            std::vector<int> sid = ints(env.W * env.H, -1);
            for (int i = 0; i < env.W * env.H; i++) {
                char c = rows[i / env.W][i % env.W];
                if (c == '#' || c == 'C') continue;
                sid[i] = env.N++;
                env.cellOf.push_back(i);
                env.kind.push_back(c);
                if (c == 'S') env.start = sid[i];
            }
            env.moveTo = ints(env.N * NA); env.moveDone = ints(env.N * NA); env.moveR = doubles(env.N * NA);
            for (int s = 0; s < env.N; s++)
                for (int d = 0; d < NA; d++) {
                    int x = env.cellOf[s] % env.W + DX[d], y = env.cellOf[s] / env.W + DY[d], k = s * NA + d;
                    char c = (x < 0 || y < 0 || x >= env.W || y >= env.H) ? '#' : rows[y][x];
                    if (c == '#') { env.moveTo[k] = s; env.moveR[k] = rStep; }                  // 撞墙：原地不动
                    else if (c == 'C') { env.moveTo[k] = env.start; env.moveR[k] = rCliff; }     // 悬崖：罚分，回到起点
                    else {
                        env.moveTo[k] = sid[y * env.W + x];
                        env.moveR[k] = c == 'G' ? rGoal : c == 'X' ? rTrap : rStep;
                        env.moveDone[k] = (c == 'G' || c == 'X') ? 1 : 0;
                    }
                }
            env.P.resize(env.N * NA);
            for (int s = 0; s < env.N; s++)
                for (int a = 0; a < NA; a++) {
                    if (isTerminal(env, s)) continue;
                    int dirs[3] = {a, (a + 1) % NA, (a + 3) % NA};                    // 意图方向、两个垂直方向
                    double probs[3] = {1 - slip, slip / 2, slip / 2};
                    for (int j = 0; j < 3; j++) {
                        if (probs[j] <= 0) continue;
                        int k = s * NA + dirs[j];
                        bool merged = false;                                          // 结果相同的分支合并（例如两个方向都撞墙）
                        for (Outcome& o : env.P[s * NA + a])
                            if (o.next == env.moveTo[k] && o.r == env.moveR[k]) { o.p += probs[j]; merged = true; }
                        if (!merged) env.P[s * NA + a].push_back({probs[j], env.moveTo[k], env.moveR[k], env.moveDone[k] == 1});
                    }
                }
            return env;
        }

        // 走一步的结果
        struct StepResult { int next; double r; bool done; };

        // 在环境里走一步：有滑动时消耗一个均匀随机数（与 JS 的 P10.step 完全一样，所以同一个种子得到同一条轨迹）
        StepResult envStep(const GridWorld& env, int s, int a, Rng& rng) {
            int d = a;
            if (env.slip > 0) {
                double u = uniform(rng);
                if (u >= 1 - env.slip) d = u < 1 - env.slip / 2 ? (a + 1) % NA : (a + 3) % NA;
            }
            int k = s * NA + d;
            return {env.moveTo[k], env.moveR[k], env.moveDone[k] == 1};
        }

        // max_a Q(s, a)
        double maxQ(const std::vector<double>& Q, int s) {
            return std::max(std::max(Q[s * NA], Q[s * NA + 1]), std::max(Q[s * NA + 2], Q[s * NA + 3]));
        }`,

    mdp: `
        // 本页的主地图（8 × 6）：(5, 2) 那一格是夹在两个陷阱之间的「独木桥」
        std::vector<std::string> mainMap() {
            return {"........", ".##..X..", "...#...G", ".#...X#.", "...#....", "S.X....."};
        }

        // 一个回合（轨迹）：S 比 A、R 多一个元素（最后到达的状态）；D[t] = 第 t 步是否进入终止格
        struct Episode { std::vector<int> S, A, D; std::vector<double> R; int T = 0; };

        // 按随机策略 π(·|s) 抽一个动作（pi 是 [N·NA] 的概率表）
        int sampleAction(const std::vector<double>& pi, int s, Rng& rng) {
            double u = uniform(rng), c = 0;
            for (int a = 0; a < NA - 1; a++) { c += pi[s * NA + a]; if (u < c) return a; }
            return NA - 1;
        }

        // 智能体–环境循环：从 s0 出发，按 π 选动作、环境返回 (s', r)，直到进入终止格或走满 maxT 步
        Episode sampleEpisode(const GridWorld& env, const std::vector<double>& pi, int s0, int maxT, Rng& rng) {
            Episode ep;
            int s = s0;
            ep.S.push_back(s);
            for (int t = 0; t < maxT && !isTerminal(env, s); t++) {
                int a = sampleAction(pi, s, rng);
                StepResult st = envStep(env, s, a, rng);
                ep.A.push_back(a); ep.R.push_back(st.r); ep.D.push_back(st.done ? 1 : 0); ep.S.push_back(st.next);
                s = st.next;
            }
            ep.T = (int)ep.A.size();
            return ep;
        }

        // 均匀随机策略：每个动作 1/4
        std::vector<double> uniformPolicy(const GridWorld& env) { return doubles(env.N * NA, 1.0 / NA); }`,

    evalPolicy: `
        // 折扣回报 G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + …（从后往前累加）
        std::vector<double> discountedReturns(const Episode& ep, double gamma) {
            std::vector<double> G = doubles(ep.T);
            double g = 0;
            for (int t = ep.T - 1; t >= 0; t--) { g = ep.R[t] + gamma * g; G[t] = g; }
            return G;
        }

        // 一步前瞻：Q(s, a) = Σ_{s'} P(s'|s, a)·[r + γ·V(s')]（进入终止格后没有未来，V = 0）
        std::vector<double> qFromV(const GridWorld& env, const std::vector<double>& V, double gamma) {
            std::vector<double> Q = doubles(env.N * NA);
            for (int s = 0; s < env.N; s++)
                for (int a = 0; a < NA; a++)
                    for (const Outcome& o : env.P[s * NA + a]) Q[s * NA + a] += o.p * (o.r + (o.done ? 0.0 : gamma * V[o.next]));
            return Q;
        }

        // 迭代策略评估：把贝尔曼期望方程当成更新规则，同步扫描 sweeps 轮  V_{k+1}(s) = Σ_a π(a|s)·Q_k(s, a)
        std::vector<double> policyEvaluation(const GridWorld& env, const std::vector<double>& pi, double gamma, int sweeps) {
            std::vector<double> V = doubles(env.N);
            for (int k = 0; k < sweeps; k++) {
                std::vector<double> Q = qFromV(env, V, gamma), Vn = doubles(env.N);
                for (int s = 0; s < env.N; s++)
                    if (!isTerminal(env, s)) for (int a = 0; a < NA; a++) Vn[s] += pi[s * NA + a] * Q[s * NA + a];
                V = Vn;
            }
            return V;
        }

        // 稠密线性方程组 M·x = b（部分主元高斯消元；M 为 n×n 行主序，按值传入，可以随便改）
        std::vector<double> solveLinear(std::vector<double> M, std::vector<double> b, int n) {
            for (int c = 0; c < n; c++) {
                int p = c;
                for (int i = c + 1; i < n; i++) if (std::fabs(M[i * n + c]) > std::fabs(M[p * n + c])) p = i;
                for (int j = 0; j < n; j++) std::swap(M[c * n + j], M[p * n + j]);
                std::swap(b[c], b[p]);
                for (int i = c + 1; i < n; i++) {
                    double f = M[i * n + c] / M[c * n + c];
                    for (int j = c; j < n; j++) M[i * n + j] -= f * M[c * n + j];
                    b[i] -= f * b[c];
                }
            }
            std::vector<double> x = doubles(n);
            for (int i = n - 1; i >= 0; i--) {
                double v = b[i];
                for (int j = i + 1; j < n; j++) v -= M[i * n + j] * x[j];
                x[i] = v / M[i * n + i];
            }
            return x;
        }

        // 精确策略评估：V^π 是线性方程组 (I − γ·P_π)·V = r_π 的解（P_π 只计进入非终止格的转移）
        std::vector<double> evaluatePolicy(const GridWorld& env, const std::vector<double>& pi, double gamma) {
            int N = env.N;
            std::vector<double> M = doubles(N * N), r = doubles(N);
            for (int i = 0; i < N; i++) M[i * N + i] = 1;
            for (int s = 0; s < N; s++)
                for (int a = 0; a < NA; a++)
                    for (const Outcome& o : env.P[s * NA + a]) {
                        double p = pi[s * NA + a] * o.p;
                        r[s] += p * o.r;
                        if (!o.done) M[s * N + o.next] -= gamma * p;
                    }
            return solveLinear(M, r, N);
        }

        // 蒙特卡洛估计：从 s0 出发按 π 采样 n 个回合，V^π(s0) ≈ 这些回合折扣回报的平均
        double monteCarloValue(const GridWorld& env, const std::vector<double>& pi, int s0, double gamma, int n, int maxT, Rng& rng) {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += discountedReturns(sampleEpisode(env, pi, s0, maxT, rng), gamma)[0];
            return sum / n;
        }`,

    vi: `
        // 贪心策略：每个状态取 Q 最大的动作（并列取编号小的；终止格记 −1）
        std::vector<int> greedyPolicy(const GridWorld& env, const std::vector<double>& Q) {
            std::vector<int> pi = ints(env.N, -1);
            for (int s = 0; s < env.N; s++) {
                if (isTerminal(env, s)) continue;
                int b = 0;
                for (int a = 1; a < NA; a++) if (Q[s * NA + a] > Q[s * NA + b] + 1e-12) b = a;
                pi[s] = b;
            }
            return pi;
        }

        struct VIResult { std::vector<double> V, Q; std::vector<int> pi; int iters = 0; };

        // 价值迭代：反复做「贝尔曼最优备份」V_{k+1}(s) = max_a Σ P·[r + γ·V_k(s')]，直到 ‖V_{k+1} − V_k‖∞ < tol
        VIResult valueIteration(const GridWorld& env, double gamma, double tol) {
            VIResult res;
            res.V = doubles(env.N);
            for (int k = 0; k < 100000; k++) {
                std::vector<double> Q = qFromV(env, res.V, gamma), Vn = doubles(env.N);
                double delta = 0;
                for (int s = 0; s < env.N; s++) {
                    if (isTerminal(env, s)) continue;
                    Vn[s] = maxQ(Q, s);
                    delta = std::max(delta, std::fabs(Vn[s] - res.V[s]));
                }
                res.V = Vn; res.iters = k + 1;
                if (delta < tol) break;                     // 收缩映射：误差每轮至少缩小到 γ 倍
            }
            res.Q = qFromV(env, res.V, gamma);
            res.pi = greedyPolicy(env, res.Q);              // 对 V* 贪心就是最优策略
            return res;
        }`,

    pi: `
        // 确定性策略（每个状态一个动作）→ 概率表 π[s·NA + a]
        std::vector<double> onehotPolicy(const GridWorld& env, const std::vector<int>& act) {
            std::vector<double> pi = doubles(env.N * NA);
            for (int s = 0; s < env.N; s++) if (act[s] >= 0) pi[s * NA + act[s]] = 1;
            return pi;
        }

        // 策略迭代：精确评估 V^π → 对 Q^π 贪心改进，直到策略不再变化。*rounds 返回评估了几次
        std::vector<int> policyIteration(const GridWorld& env, double gamma, std::vector<int> act, int* rounds) {
            for (*rounds = 1; *rounds < 100; ++*rounds) {
                std::vector<double> V = evaluatePolicy(env, onehotPolicy(env, act), gamma);   // 评估（解线性方程组）
                std::vector<double> Q = qFromV(env, V, gamma);
                bool changed = false;
                for (int s = 0; s < env.N; s++) {                                              // 改进：严格更好才换
                    if (isTerminal(env, s)) continue;
                    int b = act[s];
                    for (int a = 0; a < NA; a++) if (Q[s * NA + a] > Q[s * NA + b] + 1e-9) b = a;
                    if (b != act[s]) { act[s] = b; changed = true; }
                }
                if (!changed) break;                        // 策略改进定理：不再变化 ⇔ 已满足贝尔曼最优方程
            }
            return act;
        }`,

    ql: `
        // 最大 Q 的动作，并列时随机挑一个（全 0 的初始 Q 表上很重要，否则永远先选「上」）
        int argmaxRandom(const std::vector<double>& Q, int s, Rng& rng) {
            int o = s * NA, b = 0, n = 1;
            double m = Q[o];
            for (int a = 1; a < NA; a++) {
                double q = Q[o + a];
                if (q > m + 1e-12) { m = q; b = a; n = 1; }
                else if (q >= m - 1e-12) { n++; if (uniform(rng) * n < 1) b = a; }
            }
            return b;
        }

        // ε-greedy：以概率 ε 随机探索，否则利用当前的 Q
        int epsilonGreedy(const std::vector<double>& Q, int s, double eps, Rng& rng) {
            if (uniform(rng) < eps) return (int)(uniform(rng) * NA);
            return argmaxRandom(Q, s, rng);
        }

        struct EpisodeStats { double ret = 0; int len = 0, falls = 0; };    // 奖励和（不折扣）、步数、掉下悬崖的次数

        // Q-learning 的一个回合：行为策略 ε-greedy，TD 目标用 max（学的是贪心策略的价值 → 离策略 off-policy）
        EpisodeStats qLearningEpisode(const GridWorld& env, std::vector<double>& Q, double alpha, double gamma, double eps, int maxT, Rng& rng) {
            EpisodeStats st;
            int s = env.start;
            for (; st.len < maxT && !isTerminal(env, s); st.len++) {
                int a = epsilonGreedy(Q, s, eps, rng);
                StepResult t = envStep(env, s, a, rng);                 // 只能和环境交互：拿不到 P，也拿不到 r 的公式
                double target = t.r + (t.done ? 0.0 : gamma * maxQ(Q, t.next));
                Q[s * NA + a] += alpha * (target - Q[s * NA + a]);      // 向 TD 目标挪一小步
                st.ret += t.r;
                if (t.r <= env.rCliff) st.falls++;
                s = t.next;
            }
            return st;
        }`,

    sarsa: `
        // SARSA 的一个回合：TD 目标用「真的会执行的下一个动作」a'（学的是 ε-greedy 策略本身的价值 → 在策略 on-policy）
        EpisodeStats sarsaEpisode(const GridWorld& env, std::vector<double>& Q, double alpha, double gamma, double eps, int maxT, Rng& rng) {
            EpisodeStats st;
            int s = env.start, a = epsilonGreedy(Q, s, eps, rng);
            for (; st.len < maxT && !isTerminal(env, s); st.len++) {
                StepResult t = envStep(env, s, a, rng);
                int a2 = t.done ? 0 : epsilonGreedy(Q, t.next, eps, rng);    // 先选好下一个动作……
                double target = t.r + (t.done ? 0.0 : gamma * Q[t.next * NA + a2]);
                Q[s * NA + a] += alpha * (target - Q[s * NA + a]);
                st.ret += t.r;
                if (t.r <= env.rCliff) st.falls++;
                s = t.next; a = a2;                                         // ……下一步就执行它（S, A, R, S', A'）
            }
            return st;
        }

        // 悬崖行走（Sutton & Barto 例 6.6）：4 × 12，每步 −1，掉下悬崖 −100 并回到起点，不打折（γ = 1）
        GridWorld cliffWorld() {
            return makeGridWorld({"............", "............", "............", "SCCCCCCCCCCG"}, 0.0, -1.0, -1.0, -1.0, -100.0);
        }`,

    bandit: `
        // 标准正态 N(0, 1)：Box–Muller，双精度（与 JS 的 LM.randn 相同：消耗两个均匀数）
        double gauss(Rng& r) {
            double u1 = std::max(uniform(r), 1e-12), u2 = uniform(r);
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
        }

        struct BanditMethod { bool ucb = false; double eps = 0, c = 2, q0 = 0, alpha = 0; };   // alpha = 0：样本平均 1/N(a)
        struct BanditCurve { std::vector<double> reward, optimal; };   // 每个时间步：奖励的累加、选中最优臂的次数（跨实验累加）

        // 选臂。UCB：argmax Q(a) + c·√(ln t / N(a))，没拉过的臂先拉；ε-greedy：以 ε 概率随机，否则贪心（并列随机）
        int banditSelect(const BanditMethod& m, const std::vector<double>& Q, const std::vector<double>& N, int t, Rng& rng) {
            int K = (int)Q.size(), a = 0;
            if (m.ucb) {
                for (int i = 0; i < K; i++) if (N[i] == 0) return i;
                double best = -1e300;
                for (int i = 0; i < K; i++) {
                    double u = Q[i] + m.c * std::sqrt(std::log(t + 1.0) / N[i]);
                    if (u > best) { best = u; a = i; }
                }
                return a;
            }
            if (m.eps > 0 && uniform(rng) < m.eps) return (int)(uniform(rng) * K);
            int n = 1;
            for (int i = 1; i < K; i++) {
                if (Q[i] > Q[a]) { a = i; n = 1; }
                else if (Q[i] == Q[a]) { n++; if (uniform(rng) * n < 1) a = i; }
            }
            return a;
        }

        // 一次独立实验（10 臂测试台，Sutton & Barto §2.3）：q*(a) ~ N(0, 1)，拉臂 a 的奖励 ~ N(q*(a), 1)
        // 所有方法面对同一组 q*，结果累加进 curves[m]（最后除以实验次数就是平均曲线）
        void banditRun(uint32_t seed, int T, const std::vector<BanditMethod>& methods, std::vector<BanditCurve>& curves) {
            const int K = 10;
            Rng rq = makeRng(seed);
            std::vector<double> qs = doubles(K);
            for (int a = 0; a < K; a++) qs[a] = gauss(rq);
            int best = 0;
            for (int a = 1; a < K; a++) if (qs[a] > qs[best]) best = a;
            for (int mi = 0; mi < (int)methods.size(); mi++) {
                const BanditMethod& m = methods[mi];
                Rng rng = makeRng(seed * 31 + mi * 7 + 1);
                std::vector<double> Q = doubles(K, m.q0), N = doubles(K);
                for (int t = 0; t < T; t++) {
                    int a = banditSelect(m, Q, N, t, rng);
                    double r = qs[a] + gauss(rng);
                    N[a] += 1;
                    Q[a] += (m.alpha > 0 ? m.alpha : 1.0 / N[a]) * (r - Q[a]);   // 增量式平均：Q ← Q + (r − Q)/N
                    curves[mi].reward[t] += r;
                    if (a == best) curves[mi].optimal[t] += 1;
                }
            }
        }`,

    dqn: `
        // 网络的输入：坐标 (x, y) 归一化到 [−1, 1]，再拼上「第几列」「第几行」两个 one-hot（2 + W + H = 16 维）
        std::vector<float> stateFeatures(const GridWorld& env, int s) {
            std::vector<float> f = floats(2 + env.W + env.H);
            int x = env.cellOf[s] % env.W, y = env.cellOf[s] / env.W;
            f[0] = (float)(2.0 * x / (env.W - 1) - 1);
            f[1] = (float)(2.0 * y / (env.H - 1) - 1);
            f[2 + x] = 1;
            f[2 + env.W + y] = 1;
            return f;
        }

        // Q 网络：16 → 32 → 32 → 4 的 MLP（ReLU），一次前向给出 4 个动作的 Q 值。输出层初始化得很小：一开始所有 Q ≈ 0
        struct QNet { Linear l1, l2, l3; };

        QNet makeQNet(int in, int hidden, Rng& r) {
            QNet q;
            q.l1 = makeLinear(in, hidden, std::sqrt(2.f / in), r);
            q.l2 = makeLinear(hidden, hidden, std::sqrt(2.f / hidden), r);
            q.l3 = makeLinear(hidden, NA, 0.01f, r);
            return q;
        }

        Var qnetForward(const QNet& q, const Var& x) { return forward(q.l3, relu(forward(q.l2, relu(forward(q.l1, x))))); }

        std::vector<Var> qnetParams(const QNet& q) { return {q.l1.W, q.l1.b, q.l2.W, q.l2.b, q.l3.W, q.l3.b}; }

        // 目标网络 θ⁻ ← θ：拷贝参数的值（注意 QNet tgt = net; 只复制了句柄，两个网络会共享同一组参数）
        void copyQNet(QNet& dst, const QNet& src) {
            std::vector<Var> d = qnetParams(dst), s = qnetParams(src);
            for (int i = 0; i < (int)d.size(); i++) d[i]->val = s[i]->val;
        }

        // 经验回放池：环形缓冲区，满了就覆盖最旧的
        struct Transition { int s, a; float r; int next; bool done; };
        struct ReplayBuffer { std::vector<Transition> data; int capacity = 0, pos = 0; };

        void replayPush(ReplayBuffer& buf, Transition t) {
            if ((int)buf.data.size() < buf.capacity) buf.data.push_back(t); else buf.data[buf.pos] = t;
            buf.pos = (buf.pos + 1) % buf.capacity;
        }

        // TD 损失：mean Huber(Q(s_i, a_i) − y_i)（= PyTorch 的 F.smooth_l1_loss）。自定义算子：每行只有被选中的那个动作有梯度
        Var huberTDLoss(const Var& Q, const std::vector<int>& act, const std::vector<float>& y) {
            int B = Q->val.rows;
            std::vector<float> d = floats(B);
            double s = 0;
            for (int i = 0; i < B; i++) {
                d[i] = Q->val.at(i, act[i]) - y[i];
                s += std::fabs(d[i]) < 1 ? 0.5 * d[i] * d[i] : std::fabs(d[i]) - 0.5;   // |d| < 1 平方，之外线性（梯度被截到 ±1）
            }
            Tensor out = zeros(1, 1);
            out.v[0] = (float)(s / B);
            return record(std::move(out), {Q}, [Q, act, d, B](Node& o) {
                Tensor& g = gradOf(Q);
                for (int i = 0; i < B; i++) g.at(i, act[i]) += o.grad.v[0] * (std::fabs(d[i]) < 1 ? d[i] : (d[i] > 0 ? 1.f : -1.f)) / B;
            });
        }

        struct DQNConfig { int hidden = 32, buffer = 5000, batch = 32, warmup = 100, sync = 200, decaySteps = 8000, maxT = 100;
                           double lr = 5e-4, epsEnd = 0.1, gamma = 0.95; bool replay = true, target = true; };
        struct DQN { DQNConfig cfg; QNet net, tgt; AdamW opt; ReplayBuffer buf; int s = 0, t = 0, updates = 0, epLen = 0; };

        DQN makeDQN(const GridWorld& env, DQNConfig cfg, Rng& rng) {
            DQN d;
            d.cfg = cfg;
            int F = 2 + env.W + env.H;
            d.net = makeQNet(F, cfg.hidden, rng);
            Rng unused = makeRng(1);
            d.tgt = makeQNet(F, cfg.hidden, unused);
            copyQNet(d.tgt, d.net);
            std::vector<Var> ps = qnetParams(d.net);
            d.opt = makeAdamW(ps, std::vector<bool>(ps.size(), false), (float)cfg.lr, 0.f, 0.9f, 0.999f, 1e-8f);
            d.buf.capacity = cfg.replay ? cfg.buffer : cfg.batch;   // 不用回放 = 池子只有一个 batch 大：只能用最近的、高度相关的样本
            d.s = env.start;
            return d;
        }

        // DQN 的一步：ε-greedy 走一步 → 存进回放池 → 均匀抽一个 batch → 用目标网络算 TD 目标 → Huber 损失 → Adam 更新
        void dqnStep(const GridWorld& env, DQN& d, Rng& rng) {
            const DQNConfig& c = d.cfg;
            int F = 2 + env.W + env.H, a = 0;
            double eps = std::max(c.epsEnd, 1 - (double)d.t / c.decaySteps * (1 - c.epsEnd));   // ε 从 1 线性降到 epsEnd
            if (uniform(rng) < eps) a = (int)(uniform(rng) * NA);
            else {
                NoGrad ng;
                Var q = qnetForward(d.net, constant(fromVector(1, F, stateFeatures(env, d.s))));
                std::vector<double> q4 = doubles(NA);
                for (int k = 0; k < NA; k++) q4[k] = q->val.v[k];
                a = argmaxRandom(q4, 0, rng);
            }
            StepResult st = envStep(env, d.s, a, rng);
            replayPush(d.buf, {d.s, a, (float)st.r, st.next, st.done});
            d.t++; d.epLen++;
            if (st.done || d.epLen >= c.maxT) { d.s = env.start; d.epLen = 0; }   // 超时截断不是终止：存的 done 仍为 false
            else d.s = st.next;
            int n = (int)d.buf.data.size();
            if (n < std::min(c.warmup, d.buf.capacity)) return;
            std::vector<float> xs, xn, y = floats(c.batch);
            std::vector<int> act = ints(c.batch);
            std::vector<Transition> mb;
            for (int i = 0; i < c.batch; i++) {
                int j = c.replay ? (int)(uniform(rng) * n) : (d.buf.pos - 1 - i + 2 * d.buf.capacity) % d.buf.capacity;
                mb.push_back(d.buf.data[j]);
                std::vector<float> f = stateFeatures(env, mb[i].s), g = stateFeatures(env, mb[i].next);
                xs.insert(xs.end(), f.begin(), f.end());
                xn.insert(xn.end(), g.begin(), g.end());
                act[i] = mb[i].a;
            }
            {
                NoGrad ng;                                   // TD 目标 y = r + γ·max_a' Q_θ⁻(s', a')，当常数用（不求梯度）
                Var qn = qnetForward(c.target ? d.tgt : d.net, constant(fromVector(c.batch, F, xn)));
                for (int i = 0; i < c.batch; i++) {
                    float m = qn->val.at(i, 0);
                    for (int k = 1; k < NA; k++) m = std::max(m, qn->val.at(i, k));
                    y[i] = (float)(mb[i].r + (mb[i].done ? 0.0 : c.gamma * m));
                }
            }
            zeroGrad(d.opt.params);
            Var loss = huberTDLoss(qnetForward(d.net, constant(fromVector(c.batch, F, xs))), act, y);
            backward(loss);
            adamwStep(d.opt, (float)c.lr);
            d.updates++;
            if (c.target && d.updates % c.sync == 0) copyQNet(d.tgt, d.net);   // 每 sync 次更新同步一次目标网络
        }`,

    pg: `
        // softmax 策略表：π(·|s) = softmax(θ[s, ·])。θ 是 [N·NA] 的 logits —— 等价于「one-hot 状态 → 一层线性层 → softmax」
        std::vector<double> softmaxPolicy(const std::vector<double>& theta, int N) {
            std::vector<double> pi = doubles(N * NA);
            for (int s = 0; s < N; s++) {
                int o = s * NA;
                double m = std::max(std::max(theta[o], theta[o + 1]), std::max(theta[o + 2], theta[o + 3])), z = 0;
                for (int a = 0; a < NA; a++) { pi[o + a] = std::exp(theta[o + a] - m); z += pi[o + a]; }
                for (int a = 0; a < NA; a++) pi[o + a] /= z;
            }
            return pi;
        }

        // REINFORCE：g += w · Σ_t γ^t·(G_t − b(s_t))·∇_θ log π(a_t|s_t)。
        // 表格 softmax 的 ∇_θ[s,·] log π(a|s) = onehot(a) − π(·|s)，只有 s 那一行非零。baseline 为空 = 不用基线
        void reinforceGradient(const Episode& ep, const std::vector<double>& pi, double gamma, const std::vector<double>& baseline, double w, std::vector<double>& g) {
            std::vector<double> G = discountedReturns(ep, gamma);
            double gt = 1;
            for (int t = 0; t < ep.T; t++) {
                int s = ep.S[t], a = ep.A[t], o = s * NA;
                double k = w * gt * (G[t] - (baseline.empty() ? 0.0 : baseline[s]));
                for (int b = 0; b < NA; b++) g[o + b] -= k * pi[o + b];
                g[o + a] += k;
                gt *= gamma;
            }
        }

        // 一次 REINFORCE 更新（每回合一次）：θ += lr·ĝ。用基线时，b(s) = V(s) 同时向回报 G_t 做蒙特卡洛回归
        double reinforceUpdate(const GridWorld& env, std::vector<double>& theta, std::vector<double>& V, bool useBaseline, double lr, double lrV, double gamma, int maxT, Rng& rng) {
            std::vector<double> pi = softmaxPolicy(theta, env.N), g = doubles(env.N * NA), none;
            Episode ep = sampleEpisode(env, pi, env.start, maxT, rng);
            reinforceGradient(ep, pi, gamma, useBaseline ? V : none, 1.0, g);
            if (useBaseline) {
                std::vector<double> G = discountedReturns(ep, gamma);
                for (int t = 0; t < ep.T; t++) V[ep.S[t]] += lrV * (G[t] - V[ep.S[t]]);
            }
            for (int k = 0; k < (int)theta.size(); k++) theta[k] += lr * g[k];      // 梯度上升：J 要变大
            double R = 0;
            for (double r : ep.R) R += r;
            return R;
        }

        // 折扣访问频率 d(s) = Σ_t γ^t·P(s_t = s | s_0 = start)：解 (I − γ·P_πᵀ)·d = e_start
        std::vector<double> discountedVisits(const GridWorld& env, const std::vector<double>& pi, double gamma) {
            int N = env.N;
            std::vector<double> M = doubles(N * N), b = doubles(N);
            for (int i = 0; i < N; i++) M[i * N + i] = 1;
            for (int s = 0; s < N; s++)
                for (int a = 0; a < NA; a++)
                    for (const Outcome& o : env.P[s * NA + a]) if (!o.done) M[o.next * N + s] -= gamma * pi[s * NA + a] * o.p;
            b[env.start] = 1;
            return solveLinear(M, b, N);
        }

        // 已知模型时的精确梯度（策略梯度定理）：∂J/∂θ[s, a] = d(s)·π(a|s)·A^π(s, a)，J = V^π(start)。用来检验采样估计
        std::vector<double> policyGradientExact(const GridWorld& env, const std::vector<double>& theta, double gamma, double* J) {
            std::vector<double> pi = softmaxPolicy(theta, env.N), V = evaluatePolicy(env, pi, gamma), Q = qFromV(env, V, gamma);
            std::vector<double> d = discountedVisits(env, pi, gamma), grad = doubles(env.N * NA);
            for (int s = 0; s < env.N; s++)
                if (!isTerminal(env, s)) for (int a = 0; a < NA; a++) grad[s * NA + a] = d[s] * pi[s * NA + a] * (Q[s * NA + a] - V[s]);
            *J = V[env.start];
            return grad;
        }`,

    ac: `
        // 单步 actor-critic（Sutton & Barto §13.5，在线：每走一步就更新一次）
        // δ = r + γ·V(s') − V(s) 既是 critic 的 TD 误差，也是 actor 的优势估计（V 准确时 E[δ | s, a] = A^π(s, a)）
        double actorCriticEpisode(const GridWorld& env, std::vector<double>& theta, std::vector<double>& V, double lr, double lrV, double gamma, int maxT, Rng& rng) {
            int s = env.start;
            double I = 1, R = 0;
            for (int t = 0; t < maxT && !isTerminal(env, s); t++) {
                int o = s * NA;
                double p[NA], m = std::max(std::max(theta[o], theta[o + 1]), std::max(theta[o + 2], theta[o + 3])), z = 0;
                for (int b = 0; b < NA; b++) { p[b] = std::exp(theta[o + b] - m); z += p[b]; }
                for (int b = 0; b < NA; b++) p[b] /= z;
                int a = NA - 1;
                double u = uniform(rng), c = 0;
                for (int b = 0; b < NA - 1; b++) { c += p[b]; if (u < c) { a = b; break; } }
                StepResult st = envStep(env, s, a, rng);
                double delta = st.r + (st.done ? 0.0 : gamma * V[st.next]) - V[s];
                V[s] += lrV * delta;                                    // critic：TD(0)
                double k = lr * I * delta;                              // actor：θ[s] += lr·γ^t·δ·(onehot(a) − π(·|s))
                for (int b = 0; b < NA; b++) theta[o + b] -= k * p[b];
                theta[o + a] += k;
                I *= gamma; R += st.r; s = st.next;
            }
            return R;
        }

        // GAE(λ)：δ_t = r_t + γ·V(s_{t+1}) − V(s_t)，Â_t = δ_t + γλ·Â_{t+1}；回报目标 R̂_t = Â_t + V(s_t)
        // λ = 0 → 单步 TD 误差（方差小，偏差取决于 V 准不准）；λ = 1 → 蒙特卡洛回报 − V（无偏，方差大）
        void gaeAdvantages(const Episode& ep, const std::vector<double>& V, double gamma, double lambda, std::vector<double>& adv, std::vector<double>& ret) {
            adv = doubles(ep.T); ret = doubles(ep.T);
            double A = 0;
            for (int t = ep.T - 1; t >= 0; t--) {
                double vNext = ep.D[t] ? 0.0 : V[ep.S[t + 1]];       // 超时截断的最后一步：用 V 自举
                double delta = ep.R[t] + gamma * vNext - V[ep.S[t]];
                A = delta + gamma * lambda * A;
                adv[t] = A; ret[t] = A + V[ep.S[t]];
            }
        }`,

    ppo: `
        // 一批 on-policy 数据：M 个回合展平成 n 个 (s, a, Â, R̂, π_old(a|s)) 样本
        struct RolloutBatch { std::vector<int> s, a; std::vector<double> adv, ret, pOld; int n = 0; };

        RolloutBatch collectBatch(const GridWorld& env, const std::vector<double>& pi, const std::vector<double>& V, int M, double gamma, double lambda, int maxT, Rng& rng) {
            RolloutBatch b;
            for (int m = 0; m < M; m++) {
                Episode ep = sampleEpisode(env, pi, env.start, maxT, rng);
                std::vector<double> adv, ret;
                gaeAdvantages(ep, V, gamma, lambda, adv, ret);
                for (int t = 0; t < ep.T; t++) {
                    b.s.push_back(ep.S[t]); b.a.push_back(ep.A[t]); b.adv.push_back(adv[t]); b.ret.push_back(ret[t]);
                    b.pOld.push_back(pi[ep.S[t] * NA + ep.A[t]]);
                }
                b.n += ep.T;
            }
            double mu = 0, var = 0;                                       // 优势标准化（PPO 的常规做法）
            for (double x : b.adv) mu += x / b.n;
            for (double x : b.adv) var += (x - mu) * (x - mu) / b.n;
            for (double& x : b.adv) x = (x - mu) / (std::sqrt(var) + 1e-8);
            return b;
        }

        struct PPOStats { double clipFrac = 0, kl = 0; };

        // PPO：同一批样本做 epochs 轮、每轮 minibatches 个小批量的梯度上升，目标 L^CLIP = mean min(ρ·Â, clip(ρ, 1 − ε, 1 + ε)·Â)
        // ρ = π_θ(a|s) / π_old(a|s)。已经越界、还想往外走的样本，梯度为 0。epochs = minibatches = 1 时 ρ ≡ 1，就是 A2C
        PPOStats ppoUpdate(std::vector<double>& theta, std::vector<double>& V, const RolloutBatch& b, int epochs, int minibatches, double clip, double lr, double lrV, Rng& rng) {
            PPOStats st;
            int clipped = 0, N = (int)V.size();
            std::vector<double> g = doubles((int)theta.size());
            for (int e = 0; e < epochs; e++) {
                std::vector<int> idx = permutation(b.n, rng);
                for (int mb = 0; mb < minibatches; mb++) {
                    int lo = mb * b.n / minibatches, hi = (mb + 1) * b.n / minibatches;
                    std::fill(g.begin(), g.end(), 0.0);
                    for (int k = lo; k < hi; k++) {
                        int i = idx[k], s = b.s[i], a = b.a[i], o = s * NA;
                        double p[NA], m = std::max(std::max(theta[o], theta[o + 1]), std::max(theta[o + 2], theta[o + 3])), z = 0;
                        for (int c = 0; c < NA; c++) { p[c] = std::exp(theta[o + c] - m); z += p[c]; }
                        for (int c = 0; c < NA; c++) p[c] /= z;
                        double rho = p[a] / b.pOld[i], A = b.adv[i];
                        if ((A > 0 && rho > 1 + clip) || (A < 0 && rho < 1 - clip)) { clipped++; continue; }
                        double w = A * rho / (hi - lo);                    // ∇(ρ·Â) = Â·ρ·∇log π(a|s) = Â·ρ·(onehot(a) − π(·|s))
                        for (int c = 0; c < NA; c++) g[o + c] -= w * p[c];
                        g[o + a] += w;
                    }
                    for (int k = 0; k < (int)theta.size(); k++) theta[k] += lr * g[k];
                }
                for (int i = 0; i < b.n; i++) V[b.s[i]] += lrV * (b.ret[i] - V[b.s[i]]);   // critic 向 R̂ 回归
            }
            std::vector<double> pi = softmaxPolicy(theta, N);
            for (int i = 0; i < b.n; i++) st.kl -= std::log(pi[b.s[i] * NA + b.a[i]] / b.pOld[i]) / b.n;   // 近似 KL(π_old ‖ π_θ)
            st.clipFrac = (double)clipped / (epochs * b.n);
            return st;
        }

        // 训练循环：每轮采 M = 8 个回合，做 10 轮 × 4 个小批量的 PPO 更新
        void trainPPO(const GridWorld& env, std::vector<double>& theta, std::vector<double>& V, int iters, double clip, double lr, Rng& rng) {
            for (int it = 0; it < iters; it++) {
                RolloutBatch b = collectBatch(env, softmaxPolicy(theta, env.N), V, 8, 0.95, 0.95, 400, rng);
                ppoUpdate(theta, V, b, 10, 4, clip, lr, 0.1, rng);
            }
        }`,

    puct: `
        // AlphaZero 的 MCTS 选择规则（PUCT）：在搜索树的一个节点上，选 Q(s,a) + c·P(a|s)·√N(s) / (1 + N(s,a)) 最大的动作
        // P = 策略网络给的先验，Q = 子树里价值网络 / 对局结果的平均，N = 访问次数：先验高、访问少的动作得到探索奖励
        struct TreeNode { std::vector<double> prior, valueSum; std::vector<int> visits; };

        int puctSelect(const TreeNode& n, double cPuct) {
            int total = 0, best = 0;
            for (int v : n.visits) total += v;
            double bestScore = -1e300;
            for (int a = 0; a < (int)n.prior.size(); a++) {
                double q = n.visits[a] ? n.valueSum[a] / n.visits[a] : 0.0;
                double u = cPuct * n.prior[a] * std::sqrt((double)total) / (1 + n.visits[a]);
                if (q + u > bestScore) { bestScore = q + u; best = a; }
            }
            return best;
        }`,
};

// ═════════════════════════ 页面胶水：绘图、实时训练的分帧推进、评估 ═════════════════════════
const P10D = (() => {
    'use strict';
    const C = LM.C, F = LM.FONT;
    // 每帧推进多少（乘以顶栏「训练速度」）与上限
    const RATE = { ql: 4, cliff: 1, bandit: 6, dqn: 16, pg: 4, ac: 4, ppo: 1 };
    const CAP = { ql: 3000, cliff: 500, bandit: 2000, dqn: 30000, pg: 4000, ac: 4000, ppo: 250 };
    const EVAL = { ql: 20, pg: 20, ac: 20, dqn: 250 };
    const CLIFF_RUNS = 50, BANDIT_T = 1000;
    const DQN_CFG = { hidden: 32, lr: 5e-4, buffer: 5000, batch: 32, warmup: 100, sync: 200, decay: 8000, maxT: 100, outStd: 0.01 };
    const PG_LR = 1, PG_LRV = 0.1, PPO = { M: 8, epochs: 10, nmb: 4, lam: 0.95, lrV: 0.1 };
    const rgb = [0, 0, 0];
    const lum = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    function valColor(v, R) { PL.cmap.diverge(PL.clamp(0.5 + 0.5 * v / R, 0, 1), rgb); return rgb; }
    const fmtV = (v, d) => { const s = Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(d ?? 2); return s.replace(/^-0\./, '−.').replace(/^0\./, '.').replace(/^-/, '−'); };
    const sgn = (v, d) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d ?? 3);
    function arrowAt(ctx, cx, cy, a, len, col, wd, from) {
        const dx = P10.DX[a], dy = P10.DY[a], x0 = from ? cx : cx - dx * len * 0.5, y0 = from ? cy : cy - dy * len * 0.5, x1 = x0 + dx * len, y1 = y0 + dy * len, hd = Math.min(6, Math.max(3, len * 0.35));
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = wd || 2; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - dx * hd * 0.6, y1 - dy * hd * 0.6); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - dx * hd - dy * hd * 0.65, y1 - dy * hd + dx * hd * 0.65); ctx.lineTo(x1 - dx * hd + dy * hd * 0.65, y1 - dy * hd - dx * hd * 0.65); ctx.closePath(); ctx.fill();
    }
    /**
     * 在 (x0, y0) 处以格子边长 cs 画网格世界。o: {
     *   V（每个状态的值 → 背景色）, R（对称色标 ±R）, Q（N·4 → 每格 4 个三角形）, numbers（写 V 的数值）, digits,
     *   act（Int8Array：贪心箭头）, pi（N·4 概率：4 个箭头，长度 ∝ 概率）, path（状态序列折线）, pathColor, path2, agent, hi（黄框）, marks（Set：橙框）
     * }
     */
    function gridCore(ctx, env, x0, y0, cs, o) {
        const R = o.R || 1, W = env.W, H = env.H, small = cs < 26;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        let cliffRun = null;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, ch = env.cell[i], s = env.sid[i], X = x0 + x * cs, Y = y0 + y * cs;
            if (ch === '#') {
                ctx.fillStyle = '#20242b'; ctx.fillRect(X + 0.5, Y + 0.5, cs - 1, cs - 1);
                ctx.strokeStyle = '#2c3139'; ctx.lineWidth = 1; ctx.beginPath();
                for (let k = 1; k < 4; k++) { const t = k * cs / 4; ctx.moveTo(X + t, Y + 1); ctx.lineTo(X + 1, Y + t); ctx.moveTo(X + cs - 1, Y + t); ctx.lineTo(X + t, Y + cs - 1); }
                ctx.stroke(); continue;
            }
            if (ch === 'C') { ctx.fillStyle = '#3d1216'; ctx.fillRect(X + 0.5, Y + 0.5, cs - 1, cs - 1); if (!cliffRun) cliffRun = [X, Y, X + cs]; else cliffRun[2] = X + cs; continue; }
            if (env.term[s]) {
                const goal = ch === 'G';
                ctx.fillStyle = goal ? '#163d24' : '#4a161b'; ctx.fillRect(X + 0.5, Y + 0.5, cs - 1, cs - 1);
                ctx.strokeStyle = goal ? C.green : C.red; ctx.lineWidth = 1.5; ctx.strokeRect(X + 1.5, Y + 1.5, cs - 3, cs - 3);
                const rv = goal ? env.o.rGoal : env.o.rTrap;
                ctx.fillStyle = goal ? C.green : C.red; ctx.font = F(small ? 11 : 13, true);
                ctx.fillText(env.o.rCliff > -50 || !goal ? (rv > 0 ? '+' : '−') + Math.abs(rv) : 'G', X + cs / 2, Y + cs / 2 + 0.5);
                continue;
            }
            if (o.Q) {
                const cx = X + cs / 2, cy = Y + cs / 2, P = [[X, Y], [X + cs, Y], [X + cs, Y + cs], [X, Y + cs]];
                for (let a = 0; a < 4; a++) {   // 三角形 a：中心 + 这条边的两个角（↑ 上边、→ 右边、↓ 下边、← 左边）
                    const p0 = P[a], p1 = P[(a + 1) % 4];
                    ctx.fillStyle = LM.rgbStr(valColor(o.Q[s * 4 + a], R));
                    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.closePath(); ctx.fill();
                }
                ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X + cs, Y + cs); ctx.moveTo(X + cs, Y); ctx.lineTo(X, Y + cs); ctx.stroke();
                ctx.strokeStyle = '#0b0d10'; ctx.strokeRect(X + 0.5, Y + 0.5, cs - 1, cs - 1);
            } else {
                ctx.fillStyle = o.colorOf ? o.colorOf(s) : o.V ? LM.rgbStr(valColor(o.V[s], R)) : '#161c23';
                ctx.fillRect(X + 0.5, Y + 0.5, cs - 1, cs - 1);
            }
            if (ch === 'S') { ctx.fillStyle = '#9aa3ad'; ctx.font = F(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('S', X + 3, Y + 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; }
        }
        if (cliffRun) { ctx.fillStyle = '#e06c75'; ctx.font = F(11, true); ctx.fillText('悬  崖   −100，回到起点', (cliffRun[0] + cliffRun[2]) / 2, cliffRun[1] + cs / 2 + 0.5); }
        // 数值、箭头
        const arrowsY = o.numbers && !small ? -cs * 0.12 : 0;
        for (let s = 0; s < env.N; s++) {
            if (env.term[s]) continue;
            const [x, y] = env.xy(s), cx = x0 + (x + 0.5) * cs, cy = y0 + (y + 0.5) * cs;
            if (o.numbers && !small) {
                const v = o.V ? o.V[s] : NaN; if (v !== v) continue;
                const col = o.Q ? [20, 20, 20] : valColor(v, R);
                ctx.fillStyle = lum(col) > 140 ? '#0b0d10' : '#e6e9ee'; ctx.font = F(11);
                ctx.fillText(fmtV(v, o.digits), cx, cy + cs * (o.act || o.pi ? 0.27 : 0));
            }
        }
        for (let s = 0; s < env.N; s++) {
            if (env.term[s]) continue;
            const [x, y] = env.xy(s), cx = x0 + (x + 0.5) * cs, cy = y0 + (y + 0.5) * cs + (o.numbers ? arrowsY : 0);
            if (o.pi) {
                for (let a = 0; a < 4; a++) { const p = o.pi[s * 4 + a], len = cs * 0.44 * p; if (len > 1.5) arrowAt(ctx, cx, cy, a, len, p > 0.5 ? '#f2f2f2' : '#b8bec6', small ? 1.2 : 1.6, true); }
            } else if (o.act && o.act[s] >= 0) {
                const col = o.Q || o.V ? '#f2f2f2' : '#d8dde3';
                ctx.save(); ctx.shadowColor = 'rgba(0,0,0,0.9)'; ctx.shadowBlur = 3;
                arrowAt(ctx, cx, cy, o.act[s], cs * (small ? 0.5 : 0.42), col, small ? 1.6 : 2.2);
                ctx.restore();
            }
        }
        const ctr = (s) => { const [x, y] = env.xy(s); return [x0 + (x + 0.5) * cs, y0 + (y + 0.5) * cs]; };
        const drawPath = (path, col, wd, dots, off) => {
            if (!path || path.length < 2) return;
            ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.lineJoin = 'round'; ctx.globalAlpha = 0.9; ctx.beginPath();
            path.forEach((s, i) => { const [px, py] = ctr(s); if (i) ctx.lineTo(px + off, py + off); else ctx.moveTo(px + off, py + off); });
            ctx.stroke(); ctx.globalAlpha = 1;
            if (dots) { ctx.fillStyle = col; path.forEach((s) => { const [px, py] = ctr(s); ctx.beginPath(); ctx.arc(px + off, py + off, 2, 0, 7); ctx.fill(); }); }
        };
        drawPath(o.path2, o.path2Color || 'rgba(86,182,194,0.85)', small ? 1.5 : 2, false, 0);
        drawPath(o.path, o.pathColor || C.yellow, small ? 1.8 : 2.4, !small && o.dots !== false, o.path2 ? 2 : 0);
        if (o.marks) { ctx.strokeStyle = C.orange; ctx.lineWidth = 2; o.marks.forEach((s) => { const [x, y] = env.xy(s); ctx.strokeRect(x0 + x * cs + 2, y0 + y * cs + 2, cs - 4, cs - 4); }); }
        if (o.hi != null && o.hi >= 0) { const [x, y] = env.xy(o.hi); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2.5; ctx.strokeRect(x0 + x * cs + 1, y0 + y * cs + 1, cs - 2, cs - 2); }
        if (o.agent != null && o.agent >= 0) { const [px, py] = ctr(o.agent); ctx.fillStyle = C.yellow; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, cs * 0.2, 0, 7); ctx.fill(); ctx.stroke(); }
    }
    /** 整张画布一个网格（带标题）。返回布局 { x0, y0, cs }（存进 S 供 pick / probe 反算格子） */
    function drawGrid(c, env, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0d10' });
        const top = o.title ? 20 : 4, bot = o.foot ? 18 : 4;
        const cs = Math.floor(Math.min((w - 8) / env.W, (h - top - bot) / env.H));
        const x0 = Math.round((w - cs * env.W) / 2), y0 = top + Math.round((h - top - bot - cs * env.H) / 2);
        gridCore(ctx, env, x0, y0, cs, o);
        ctx.textBaseline = 'top';
        if (o.title) { ctx.fillStyle = C.text; ctx.font = F(11, true); ctx.textAlign = 'left'; ctx.fillText(o.title, o.titleX ?? 4, 4); }
        if (o.sub) { ctx.fillStyle = o.subColor || C.yellow; ctx.font = F(11); ctx.textAlign = 'right'; ctx.fillText(o.sub, w - 4, 4); }
        if (o.foot) { ctx.fillStyle = C.gray; ctx.font = F(11); ctx.textAlign = 'left'; ctx.fillText(o.foot, 4, h - 15); }
        return { x0, y0, cs };
    }
    /** 多个小网格并排（small multiples）：panels = [{ env, o, label, labelColor }] */
    function gridPanels(c, panels, cols, o) {
        o = o || {};
        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0d10' });
        const rows = Math.ceil(panels.length / cols), top = o.title ? 18 : 2, lab = 15, gap = 6;
        const pw = (w - gap * (cols + 1)) / cols, ph = (h - top - gap * (rows + 1)) / rows;
        if (o.title) { ctx.fillStyle = C.text; ctx.font = F(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.titleX ?? 4, 3); }
        const lays = [];
        panels.forEach((p, k) => {
            const cx = gap + (k % cols) * (pw + gap), cy = top + gap + Math.floor(k / cols) * (ph + gap);
            const cs = Math.floor(Math.min(pw / p.env.W, (ph - lab) / p.env.H)), gx = cx + Math.round((pw - cs * p.env.W) / 2), gy = cy + lab;
            ctx.fillStyle = p.labelColor || C.gray; ctx.font = F(11, p.bold); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText(p.label, cx + pw / 2, cy + 1);
            gridCore(ctx, p.env, gx, gy, cs, p.o || {});
            if (p.frame) { ctx.strokeStyle = p.frame; ctx.lineWidth = 2; ctx.strokeRect(gx - 2, gy - 2, cs * p.env.W + 4, cs * p.env.H + 4); }
            lays.push({ x0: gx, y0: gy, cs });
        });
        return lays;
    }
    const hitCell = (L, env, info) => { if (!L || !info) return -1; const x = Math.floor((info.dx - L.x0) / L.cs), y = Math.floor((info.dy - L.y0) / L.cs); return x < 0 || y < 0 || x >= env.W || y >= env.H ? -1 : env.sid[y * env.W + x]; };
    const cellName = (env, s) => { const [x, y] = env.xy(s); return `(${x}, ${y})`; };
    /** 播放头：0 → n 匀速走完再停 hold 秒，循环 */
    const cycle = (S, n, rate, hold) => { const per = n + (hold ?? 2) * rate; return Math.min(n, Math.floor((S.t * rate) % per)); };
    /** 训练曲线上「已经算到第 k 个点」：完整曲线淡色 + 已到部分实色（第 06 篇同款） */
    function played(data, k, color, label, xs, o) {
        const n = Math.min(k + 1, data.length);
        const out = [{ data: data.slice(0, n), x: xs.slice(0, n), color, width: (o && o.width) || 2, label, dash: o && o.dash }];
        return out;
    }
    /** 贪心策略 / ε-greedy 策略在已知模型下的精确价值 J = V^π(start) */
    function jGreedy(env, Q, gamma) { return P10.evalExact(env, P10.onehotPolicy(env, P10.greedy(env, Q)), gamma)[env.start]; }
    function jEps(env, Q, eps, gamma) {
        const g = P10.greedy(env, Q), pi = new Float64Array(env.N * 4);
        for (let s = 0; s < env.N; s++) if (g[s] >= 0) for (let a = 0; a < 4; a++) pi[s * 4 + a] = eps / 4 + (a === g[s] ? 1 - eps : 0);
        return P10.evalExact(env, pi, gamma)[env.start];
    }
    /** Q 表与 Q* 的差距：非终止状态上 RMS(max_a Q − V*)、贪心动作是最优动作的比例 */
    function qGap(env, Q, vi) {
        let e = 0, n = 0, ok = 0;
        const g = P10.greedy(env, Q);
        for (let s = 0; s < env.N; s++) { if (env.term[s]) continue; e += (P10.maxQ(Q, s) - vi.V[s]) ** 2; n++; if (vi.Q[s * 4 + g[s]] >= vi.V[s] - 1e-6) ok++; }
        return { rms: Math.sqrt(e / n), opt: ok / n };
    }
    /** J 曲线的纵轴范围：覆盖「随机策略」「永远走不到终点（每步付 r_step）」与 V*(start) */
    function jRange(S) {
        const loop = S.rstep / (1 - S.gamma), lo = Math.min(S.Jrand, loop, S.vi.V[S.env.start]) , hi = S.vi.V[S.env.start];
        const span = Math.max(0.2, hi - lo);
        return [lo - 0.08 * span, hi + 0.42 * span];   // 上方留出图例的位置
    }
    function jMarks(S) {
        const out = [{ y: S.vi.V[S.env.start], color: C.green, label: `V*(S) = ${sgn(S.vi.V[S.env.start])}` }, { y: S.Jrand, color: '#6a7480', label: `随机策略 ${sgn(S.Jrand)}` }];
        const loop = S.rstep / (1 - S.gamma);
        if (loop < S.Jrand - 0.02) out.push({ y: loop, color: '#5a4a3a', label: '永不结束' });
        return out;
    }
    /** 在画好的 plot 上补一行说明文字（左上角，深色底）：动态数字放这里，不挤 xLabel */
    function note(c, text, o) {
        o = o || {};
        const { ctx } = PL.draw.ctx(c, { keep: true });
        ctx.font = F(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        const x = o.x ?? 56, y = o.y ?? 13, tw = ctx.measureText(text).width;
        ctx.fillStyle = 'rgba(8,8,10,0.85)'; ctx.fillRect(x - 3, y - 2, tw + 6, 15);
        ctx.fillStyle = o.color || C.text; ctx.fillText(text, x, y);
    }
    const histo = (vals, lo, hi, nb) => { const h = new Array(nb).fill(0); vals.forEach((v) => { const k = Math.floor((v - lo) / (hi - lo) * nb); h[PL.clamp(k, 0, nb - 1)]++; }); return h; };

    // ───────────── 实时训练：每个 history stage 的状态对象 + 分帧推进 ─────────────
    const seedOf = (S, k) => ((S.seed ^ (0x9e3779b9 * k)) >>> 0) || 1;
    function qlInit(S) {
        return { Q: new Float64Array(S.env.N * 4), rand: LM.rng(seedOf(S, 5)), o: { alpha: S.alpha, gamma: S.gamma, eps: S.eps, maxT: 200 }, ep: 0, rets: [], ev: [], path: [S.env.start], visits: new Float64Array(S.env.N) };
    }
    function qlEval(S, st) { const g = qGap(S.env, st.Q, S.vi); st.ev.push({ ep: st.ep, J: jGreedy(S.env, st.Q, S.gamma), Je: jEps(S.env, st.Q, st.o.eps, S.gamma), rms: g.rms, opt: g.opt }); }
    function qlAdvance(S, st, n) {
        for (let k = 0; k < n && st.ep < CAP.ql; k++) {
            const r = P10.qlEpisode(S.env, st.Q, st.o, st.rand, st.path);
            st.path.forEach((s) => { st.visits[s]++; });
            st.ep++; st.rets.push(r.ret);
            if (st.ep % EVAL.ql === 0) qlEval(S, st);
        }
    }
    function cliffInit(S) {
        const env = S.cenv, mk = (m) => Array.from({ length: CLIFF_RUNS }, (_, i) => ({ Q: new Float64Array(env.N * 4), rand: LM.rng(seedOf(S, 100 + 2 * i + m)) }));
        return { runs: [mk(0), mk(1)], ep: 0, sum: [[], []], falls: [[], []], path: [[env.start], [env.start]], o: { alpha: S.alpha, gamma: 1, eps: S.eps, maxT: 5000 } };
    }
    function cliffAdvance(S, st, n) {
        const env = S.cenv;
        for (let k = 0; k < n && st.ep < CAP.cliff; k++) {
            [P10.sarsaEpisode, P10.qlEpisode].forEach((fn, m) => {
                let tot = 0, falls = 0;
                st.runs[m].forEach((r, i) => { const res = fn(env, r.Q, st.o, r.rand, i === 0 ? st.path[m] : null); tot += res.ret; falls += res.falls; });
                st.sum[m].push(tot / CLIFF_RUNS); st.falls[m].push(falls / CLIFF_RUNS);
            });
            st.ep++;
        }
    }
    const BANDIT_METHODS = (S) => [{ kind: 'eps', eps: 0, name: '贪心（ε = 0）', color: C.gray }, { kind: 'eps', eps: S.eps > 0 ? S.eps : 0.1, name: `ε-greedy（ε = ${S.eps > 0 ? S.eps : 0.1}）`, color: C.blue }, { kind: 'ucb', c: S.ucbC, name: `UCB（c = ${S.ucbC}）`, color: C.green }, { kind: 'eps', eps: 0, q0: 5, alpha: 0.1, name: '乐观初值 Q₁ = 5（贪心）', color: C.orange }];
    function banditInit(S) { const ms = BANDIT_METHODS(S); return { ms, runs: 0, acc: { rew: ms.map(() => new Float64Array(BANDIT_T)), opt: ms.map(() => new Float64Array(BANDIT_T)) } }; }
    function banditAdvance(S, st, n) { for (let k = 0; k < n && st.runs < CAP.bandit; k++) { st.runs++; P10.banditRun(seedOf(S, 7) + st.runs, BANDIT_T, st.ms, st.acc); } }
    /** 一次 UCB / ε-greedy 实验的内部记录（Step 07 的「一次实验里发生了什么」） */
    function banditTrace(S, seed) {
        const K = 10, T = BANDIT_T, rq = LM.rng(seed), qs = Float64Array.from({ length: K }, () => LM.randn(rq));
        const m = { kind: 'ucb', c: S.ucbC }, rand = LM.rng(seed * 31 + 1), Q = new Float64Array(K), N = new Float64Array(K);
        const Qs = new Float32Array(T * K), Ns = new Float32Array(T * K), A = new Int8Array(T);
        for (let t = 0; t < T; t++) {
            let a = -1; for (let i = 0; i < K; i++) if (N[i] === 0) { a = i; break; }
            if (a < 0) { const lt = Math.log(t + 1); let bv = -Infinity; for (let i = 0; i < K; i++) { const u = Q[i] + m.c * Math.sqrt(lt / N[i]); if (u > bv) { bv = u; a = i; } } }
            const r = qs[a] + LM.randn(rand); N[a]++; Q[a] += (r - Q[a]) / N[a]; A[t] = a;
            Qs.set(Q, t * K); Ns.set(N, t * K);
        }
        return { qs, Qs, Ns, A, K, T, best: LM.argmax(qs) };
    }
    function dqnInitS(S) {
        const o = Object.assign({ gamma: S.gamma, epsEnd: S.eps, replay: S.dqnVar === 'full' || S.dqnVar === 'noTarget', target: S.dqnVar === 'full' || S.dqnVar === 'noReplay' }, DQN_CFG);
        const st = P10.dqnInit(S.env, o, seedOf(S, 11)); st.ev = []; st.Qt = new Float32Array(S.env.N * 4); st.lossEma = []; st.lm = 0; st.lw = 0;
        P10.dqnTable(st, st.Qt); dqnEval(S, st);
        return st;
    }
    function dqnEval(S, st) { const g = qGap(S.env, st.Qt, S.vi); st.ev.push({ t: st.t, J: jGreedy(S.env, st.Qt, S.gamma), rms: g.rms, opt: g.opt }); }
    function dqnAdvance(S, st, n) {
        for (let k = 0; k < n && st.t < CAP.dqn; k++) {
            const u0 = st.upd; P10.dqnStep(st);
            if (st.upd > u0) { st.lm = 0.98 * st.lm + 0.02 * st.lastLoss; st.lw = 0.98 * st.lw + 0.02; if (st.upd % 50 === 0) st.lossEma.push([st.t, st.lm / st.lw]); }
            if (st.t % EVAL.dqn === 0) { P10.dqnTable(st, st.Qt); dqnEval(S, st); }
        }
        P10.dqnTable(st, st.Qt);
    }
    const dqnEpsNow = (st) => Math.max(st.o.epsEnd, 1 - st.t / st.o.decay * (1 - st.o.epsEnd));
    function pgInitS(S, k, extra) { const st = P10.pgInit(S.env, Object.assign({ gamma: S.gamma, maxT: 400, lr: PG_LR, lrV: PG_LRV }, extra), seedOf(S, k)); st.ev = [{ ep: 0, J: S.Jrand }]; st.rets = []; return st; }
    function pgEval(S, st) { st.ev.push({ ep: st.episodes, J: P10.evalExact(S.env, st.pi, S.gamma)[S.env.start] }); }
    // ── 策略梯度的结果很依赖随机种子：每种方法同时跑几个种子（同一组种子），画每个种子的细线 + 平均的粗线 ──
    const NSEED = { pg: 8, ac: 8, ppo: 4 };
    /** 一组运行：name、color、step(run) = 推进一个回合 / 一批，runs = n 个种子 */
    function pgGroup(S, name, color, n, extra, step) { return { name, color, step, runs: Array.from({ length: n }, (_, i) => pgInitS(S, 50 + i, extra)) }; }
    function groupsAdvance(S, groups, n, cap) {
        for (let k = 0; k < n; k++) {
            let any = false;
            groups.forEach((g) => g.runs.forEach((r) => { if (r.episodes >= cap) return; any = true; g.step(r); if (r.episodes % EVAL.pg === 0) pgEval(S, r); }));
            if (!any) break;
        }
    }
    /** 一组的平均曲线（各种子在同样的回合数上评估，逐点平均） */
    function groupMean(g, key) {
        key = key || 'J';
        const n = Math.min(...g.runs.map((r) => r.ev.length)), out = [];
        for (let k = 0; k < n; k++) out.push({ ep: g.runs[0].ev[k].ep, v: LM.mean(g.runs.map((r) => r.ev[k][key])) });
        return out;
    }
    /** plot 的 series：每个种子一条细线（半透明）+ 平均一条粗线（带图例） */
    function groupSeries(g, label) {
        const out = g.runs.map((r) => ({ data: r.ev.map((e) => e.J), x: r.ev.map((e) => e.ep), color: g.color + '40', width: 1 }));
        const m = groupMean(g);
        out.push({ data: m.map((e) => e.v), x: m.map((e) => e.ep), color: g.color, width: 2.4, label: label || g.name });
        return out;
    }
    const groupLast = (g) => { const m = groupMean(g); return m.length ? m[m.length - 1].v : NaN; };
    const groupOk = (g) => g.runs.filter((r) => r.ev[r.ev.length - 1].J > -0.4).length;
    function ppoInitS(S) {
        const mk = (epochs, nmb, clip, name, color) => {
            const runs = Array.from({ length: NSEED.ppo }, (_, i) => { const st = P10.pgInit(S.env, { gamma: S.gamma, lam: PPO.lam, lr: S.ppoLR, lrV: PPO.lrV, M: PPO.M, maxT: 400 }, seedOf(S, 50 + i)); st.ev = [{ ep: 0, J: S.Jrand, kl: NaN, clip: 0 }]; return st; });
            return { name, color, cfg: { epochs, nmb, clip }, runs };
        };
        return [mk(1, 1, 1e9, 'A2C（每批 1 步）', C.gray), mk(PPO.epochs, PPO.nmb, 1e9, `不裁剪（每批 ${PPO.epochs * PPO.nmb} 步）`, C.red), mk(PPO.epochs, PPO.nmb, S.ppoClip, S.ppoClip > 10 ? '不裁剪（同上）' : `PPO ε = ${S.ppoClip}`, C.green)];
    }
    function ppoAdvance(S, groups, n) {
        for (let k = 0; k < n; k++) groups.forEach((g) => g.runs.forEach((st) => {
            if (st.ev.length > CAP.ppo) return;
            P10.collectGAE(st);
            const u = P10.ppoUpdate(st, g.cfg.epochs, g.cfg.nmb, g.cfg.clip);
            st.last = u; st.ratios = u.ratios.slice();
            st.ev.push({ ep: st.episodes, J: P10.evalExact(S.env, st.pi, S.gamma)[S.env.start], kl: u.kl, clip: u.clipFrac });
        }));
    }
    /** Step 09：在均匀随机策略 θ = 0 处，比较单回合 REINFORCE 估计与精确梯度 */
    function pgVariance(S, K) {
        const env = S.env, ex = P10.pgExact(env, new Float64Array(env.N * 4), S.gamma), gn = Math.hypot(...ex.grad), out = {};
        const bases = { none: null, constJ: new Float64Array(env.N).fill(ex.J), Vpi: ex.V };
        for (const [name, b] of Object.entries(bases)) {
            const rand = LM.rng(seedOf(S, 60)), ep = { S: [], A: [], R: [], D: [], T: 0 }, g = new Float64Array(env.N * 4), mean = new Float64Array(env.N * 4), cos = [], err = [];
            let se = 0;
            for (let k = 1; k <= K; k++) {
                g.fill(0); P10.sampleEpisode(env, ex.pi, rand, 400, ep); P10.reinforceGrad(ep, ex.pi, S.gamma, b, g, 1);
                let e = 0, dot = 0, nn = 0, em = 0;
                for (let i = 0; i < g.length; i++) { mean[i] += (g[i] - mean[i]) / k; e += (g[i] - ex.grad[i]) ** 2; dot += g[i] * ex.grad[i]; nn += g[i] * g[i]; em += (mean[i] - ex.grad[i]) ** 2; }
                se += e; cos.push(dot / (Math.sqrt(nn) * gn + 1e-30)); err.push(Math.sqrt(em) / gn);
            }
            out[name] = { relMSE: se / K / (gn * gn), cos, err };
        }
        return { J: ex.J, gn, out, K };
    }
    /** Step 10：单步 AC 训练 E0 个回合后的快照上，GAE(λ) 估计的梯度相对精确 ∇J 的偏差² / 方差（每回合，相对 ‖∇J‖²） */
    function gaeScan(S, E0, K) {
        const env = S.env, st = P10.pgInit(env, { gamma: S.gamma, maxT: 400, lr: PG_LR, lrV: PG_LRV }, seedOf(S, 70));
        for (let e = 0; e < E0; e++) P10.acEpisode(st);
        const ex = P10.pgExact(env, st.theta, S.gamma), gn2 = ex.grad.reduce((s, v) => s + v * v, 0), lams = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1], rows = [];
        let ce = 0, cn = 0; for (let s = 0; s < env.N; s++) if (!env.term[s] && ex.d[s] > 0.01) { ce += (st.Vc[s] - ex.V[s]) ** 2; cn++; }
        for (const lam of lams) {
            const rand = LM.rng(seedOf(S, 71)), ep = { S: [], A: [], R: [], D: [], T: 0 }, mean = new Float64Array(env.N * 4), g = new Float64Array(env.N * 4), adv = [];
            let sq = 0;
            for (let k = 0; k < K; k++) {
                P10.sampleEpisode(env, ex.pi, rand, 400, ep); g.fill(0);
                let A = 0; for (let t = ep.T - 1; t >= 0; t--) { const s = ep.S[t], vn = ep.D[t] ? 0 : st.Vc[ep.S[t + 1]]; A = ep.R[t] + S.gamma * vn - st.Vc[s] + S.gamma * lam * A; adv[t] = A; }
                let gt = 1; for (let t = 0; t < ep.T; t++) { const s = ep.S[t], a = ep.A[t], w = gt * adv[t], o = s * 4; for (let b = 0; b < 4; b++) g[o + b] -= w * ex.pi[o + b]; g[o + a] += w; gt *= S.gamma; }
                for (let i = 0; i < g.length; i++) { mean[i] += g[i] / K; sq += g[i] * g[i] / K; }
            }
            let m2 = 0, b2 = 0; for (let i = 0; i < mean.length; i++) { m2 += mean[i] * mean[i]; b2 += (mean[i] - ex.grad[i]) ** 2; }
            const varEp = sq - m2;                              // 单回合估计的总方差 tr Cov
            rows.push({ lam, bias2: Math.max(0, b2 - varEp / K) / gn2, var: varEp / gn2 });   // 偏差² 减去有限样本噪声 var/K
        }
        return { rows, J: ex.J, criticRMS: Math.sqrt(ce / cn), E0, K };
    }
    /** 按确定性策略 act（意图方向）从起点走：终止、重复访问或 maxT 步就停。返回 { path, end: 'G' | 'X' | '∞' } */
    function routeOf(env, act, maxT) {
        const path = [env.start], seen = new Set([env.start]);
        let s = env.start;
        for (let t = 0; t < (maxT || 40) && !env.term[s]; t++) { s = env.mv[s * 4 + act[s]]; path.push(s); if (seen.has(s)) break; seen.add(s); }
        return { path, end: env.term[s] ? env.kind[s] : '∞', len: path.length - 1 };
    }
    /** Step 08：在当前 DQN 网络的一份拷贝上，只对一个格子 s0 的 Q(s0, a0) 做一次梯度上推，看其他格子的 Q(·, a0) 跟着变了多少 */
    function dqnGeneralize(S, st, s0) {
        const env = S.env, net = P10.mlp([st.F, st.o.hidden, st.o.hidden, 4], null);
        P10.copyNet(net, st.net);
        const before = P10.mlpRows(net, st.X, env.N, new Float32Array(env.N * 4), st.tmp);
        const g = P10.greedy(env, before), a0 = g[s0] >= 0 ? g[s0] : 0;
        const Qt = P10.mlpForward(net, LM.AG.tensor(1, st.F, st.X.slice(st.F * s0, st.F * s0 + st.F)));
        LM.AG.backward(P10.tdLoss(Qt, Int32Array.of(a0), Float64Array.of(Qt.data[a0] + 0.999)));   // 目标 = 当前值 + 1：Huber 的梯度 ≈ −∇Q(s0, a0)
        P10.mlpParams(net).forEach((p) => { if (p.grad) for (let i = 0; i < p.data.length; i++) p.data[i] -= 0.05 * p.grad[i]; });
        const after = P10.mlpRows(net, st.X, env.N, new Float32Array(env.N * 4), st.tmp), dQ = new Float64Array(env.N);
        for (let s = 0; s < env.N; s++) dQ[s] = env.term[s] ? 0 : after[s * 4 + a0] - before[s * 4 + a0];
        return { s0, a0, dQ };
    }
    return {
        RATE, CAP, EVAL, CLIFF_RUNS, BANDIT_T, DQN_CFG, PG_LR, PG_LRV, PPO, valColor, fmtV, sgn, arrowAt, gridCore, drawGrid, gridPanels, hitCell, cellName, cycle, played,
        jGreedy, jEps, qGap, jRange, jMarks, histo, note, seedOf, qlInit, qlEval, qlAdvance, cliffInit, cliffAdvance, BANDIT_METHODS, banditInit, banditAdvance, banditTrace,
        dqnInitS, dqnEval, dqnAdvance, dqnEpsNow, pgInitS, pgEval, NSEED, pgGroup, groupsAdvance, groupMean, groupSeries, groupLast, groupOk, ppoInitS, ppoAdvance, pgVariance, gaeScan, routeOf, dqnGeneralize,
    };
})();

/** 实时训练 stage 的公共骨架：参数一变（S.epoch 自增）就重建状态；「⏩ 训练到底」时一次跑到上限 */
function p10Live(S, key, init, advance, rate) {
    let st = S[key];
    if (!st || st._epoch !== S.epoch) { st = S[key] = init(S); st._epoch = S.epoch; }
    const ff = S.ffwdEpoch === S.epoch && !st._ffwd;
    advance(S, st, ff ? Infinity : rate * S.spf);
    if (S.ffwdEpoch === S.epoch) st._ffwd = true;
    return st;
}

PipelineLab.mount({
    paramSteps: { gamma: ['mdp', 'eval', 'vi', 'piter', 'qlearn', 'dqn', 'reinforce', 'ac', 'ppo', 'prod'], slip: ['mdp', 'eval', 'vi', 'piter', 'qlearn', 'dqn', 'reinforce', 'ac', 'ppo', 'prod'], rstep: ['mdp', 'eval', 'vi', 'piter', 'qlearn', 'dqn', 'reinforce', 'ac', 'ppo', 'prod'], alpha: ['qlearn', 'cliff'], eps: ['qlearn', 'cliff', 'bandit'], ucbC: ['bandit'], dqnVar: ['dqn'], ppoLR: ['ppo', 'prod'], ppoClip: ['ppo', 'prod'], spf: ['qlearn', 'cliff', 'bandit', 'dqn', 'reinforce', 'ac', 'ppo', 'prod'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '10 · 强化学习：从价值迭代到策略梯度',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 20251123,
    css: LM.CSS,
    regen: false,
    state: { ffwdEpoch: -1, sel: -1, genS: -1 },

    params: [
        { id: 'gamma', label: '折扣 γ', min: 0.5, max: 0.99, step: 0.01, value: 0.95,
            desc: '未来奖励每晚一步打一次 γ 折，有效视野约 1/(1 − γ) 步。影响主地图上的所有方法。拖到 0.9 以下：最优路线改走中间的「独木桥」（Step 03）；拖到 0.5：远处的格子几乎感觉不到目标（Step 02 / 03 的价值图）。' },
        { id: 'slip', label: '滑动概率', min: 0, max: 0.5, step: 0.05, value: 0.2,
            desc: '地面有多滑：以 1 − slip 的概率按意图走，各以 slip/2 的概率滑向两侧。0 = 确定性环境，最优路线走「独木桥」；默认 0.2 时过桥那一步有 20% 掉进陷阱，最优路线绕远；0.4 以上宁可贴着墙等着「滑」进目标。' },
        { id: 'rstep', label: '每步奖励', min: -0.5, max: 0.1, step: 0.01, value: -0.04,
            desc: '每走一步（没进终止格）得到的奖励，即「生活成本」；−0.04 是 Russell & Norvig 的经典取值。−0.08 左右：着急，走独木桥；−0.15 以下：从起点直接跳进旁边的陷阱；大于 0：永远不去终点（Step 03 右下）。' },
        { id: 'alpha', label: 'TD 学习率 α', options: [0.05, 0.1, 0.2, 0.5], value: 0.1,
            desc: 'Q-learning / SARSA 每次把 Q(s, a) 向 TD 目标挪多少（Step 05、06）。小：稳但慢；大：快，但在随机滑动下 Q 抖得厉害。悬崖行走取 0.5 时，SARSA 会退到最上面一行。' },
        { id: 'eps', label: '探索率 ε', options: [0, 0.05, 0.1, 0.2, 0.3], value: 0.1, steps: ['dqn'],
            desc: '以概率 ε 随机选动作。用于 Q-learning、SARSA（Step 05、06）、DQN（ε 从 1 线性降到这个值，Step 08）和赌博机的 ε-greedy 曲线（Step 07；选 0 时那条曲线用 0.1）。ε 越大，悬崖上 SARSA 离崖边越远。' },
        { id: 'ucbC', label: 'UCB 系数 c', min: 0, max: 4, step: 0.25, value: 2,
            desc: 'Step 07 的置信上界 Q(a) + c·√(ln t / N(a))：c 越大越爱试「没怎么试过」的臂。0 = 纯贪心；2 是 Sutton & Barto 图 2.4 的取值；4：前期探索过度。' },
        { id: 'dqnVar', label: 'DQN 变体', options: [{ value: 'full', label: '回放 + 目标网络' }, { value: 'noTarget', label: '无目标网络' }, { value: 'noReplay', label: '无经验回放' }, { value: 'neither', label: '两者都不用' }], value: 'full',
            desc: 'Step 08 的 DQN 用不用经验回放池、目标网络。在这个小问题上去掉目标网络影响不大；去掉经验回放（每次只用最近、彼此高度相关的 32 条样本）就学不会。' },
        { id: 'ppoLR', label: 'PPO 学习率', options: [1, 2, 4, 8, 16], value: 4,
            desc: 'Step 11 三条曲线共用的步长（优势已标准化）。越大，同一批数据上每轮走得越远：4 以上不裁剪的版本基本都崩，PPO 的裁剪把每批数据带来的变化限制住；但到 16 时 PPO 也开始有种子失败，而 A2C（每批只走一步）反而最好。' },
        { id: 'ppoClip', label: 'PPO 裁剪 ε', options: [0.1, 0.2, 0.3, { value: 1e9, label: '不裁剪' }], value: 0.2,
            desc: 'Step 11 的裁剪范围：概率比 ρ = π_θ/π_old 已超出 [1 − ε, 1 + ε] 且还想往外走的样本，不再提供梯度。选「不裁剪」：绿线与红线重合。' },
        { id: 'spf', label: '训练速度', options: [{ value: 1, label: '1×' }, { value: 2, label: '2×' }, { value: 4, label: '4×' }, { value: 8, label: '8×' }], value: 1, affects: 'frame',
            desc: '实时训练每帧走多少步（基础速度的倍数）。只在对应步骤的预览可见时训练；机器快可以调到 4× 或 8×。' },
        { id: 'ffwd', label: '⏩ 训练到底', action: (S) => { S.ffwdEpoch = S.epoch + 1; }, affects: 'param', title: '重置并把所有实时训练直接跑到上限（各自下次可见时执行）',
            desc: '把 Q-learning、悬崖行走、赌博机、DQN、REINFORCE、actor-critic、PPO 的实时训练直接跑到上限（各自下次可见时执行；DQN 要几秒），用来看最终状态。' },
    ],

    stages: [
        {
            id: 'cliffEnv', lifetime: 'resident', run(S) {
                S.cenv = P10.makeEnv(P10.MAPS.cliff, { slip: 0, rstep: -1, rGoal: -1, rTrap: -1, rCliff: -100 });
                S.cvi = P10.valueIteration(S.cenv, 1, { tol: 1e-9 });
            },
        },
        {
            id: 'world', lifetime: 'param', run(S) {
                const env = S.env = P10.makeEnv(P10.MAPS.main, { slip: S.slip, rstep: S.rstep }), N = env.N;
                const vi = S.vi = P10.valueIteration(env, S.gamma, { keep: 80 });
                S.uni = P10.uniformPolicy(env);
                S.Vrand = P10.evalExact(env, S.uni, S.gamma); S.Jrand = S.Vrand[env.start];
                S.pe = P10.evalSweeps(env, S.uni, S.gamma, 60);
                S.peErr = S.pe.Vs.map((V) => { let e = 0; for (let s = 0; s < N; s++) e = Math.max(e, Math.abs(V[s] - S.Vrand[s])); return e; });
                let m1 = 0, m2 = 0; for (let s = 0; s < N; s++) { m1 = Math.max(m1, Math.abs(vi.V[s])); m2 = Math.max(m2, Math.abs(S.Vrand[s])); }
                S.Rv = Math.max(0.25, m1); S.Rr = Math.max(0.25, m2);
                S.viAct = vi.Vs.map((V) => P10.greedy(env, P10.qFromV(env, V, S.gamma)));
                S.viErr = vi.Vs.map((V) => { let e = 0; for (let s = 0; s < N; s++) e = Math.max(e, Math.abs(V[s] - vi.V[s])); return e; });
                S.viBad = S.viAct.map((act) => { const bad = new Set(); for (let s = 0; s < N; s++) if (!env.term[s] && vi.Q[s * 4 + act[s]] < vi.V[s] - 1e-9) bad.add(s); return bad; });
                S.viOptAt = S.viBad.findIndex((b) => b.size === 0);
                const a0 = new Int8Array(N); for (let s = 0; s < N; s++) a0[s] = env.term[s] ? -1 : 0;   // 策略迭代的起点：处处「向上」
                S.piSeq = P10.policyIteration(env, S.gamma, a0);
                S.route = P10D.routeOf(env, vi.pi);
                if (!(S.sel >= 0 && S.sel < N) || env.term[S.sel]) S.sel = env.start;
                if (!(S.genS >= 0 && S.genS < N) || env.term[S.genS]) S.genS = env.sid[2 * env.W + 4];
            },
        },
        {
            id: 'walk', lifetime: 'param', run(S) {
                const env = S.env, mk = () => ({ S: [], A: [], R: [], D: [], T: 0 }), rand = LM.rng(P10D.seedOf(S, 1)), eps = [];
                for (let k = 0; k < 16; k++) eps.push(P10.sampleEpisode(env, S.uni, rand, 60, mk()));
                const tl = []; eps.forEach((ep, i) => { for (let t = 0; t <= ep.T; t++) tl.push([i, t]); for (let p = 0; p < 5; p++) tl.push([i, ep.T]); });
                const r2 = LM.rng(P10D.seedOf(S, 2)), e = mk(), near = env.sid[5 * env.W + 2], st = { goal: 0, near: 0, other: 0, timeout: 0, n: 2000, len: 0, G: 0 };
                for (let k = 0; k < st.n; k++) {
                    P10.sampleEpisode(env, S.uni, r2, 200, e);
                    const last = e.S[e.T]; st.len += e.T / st.n; st.G += P10.returnsOf(e, S.gamma)[0] / st.n;
                    if (!env.term[last]) st.timeout++; else if (env.kind[last] === 'G') st.goal++; else if (last === near) st.near++; else st.other++;
                }
                S.walk = { eps, tl, stats: st };
            },
        },
        {
            id: 'regimes', lifetime: 'param', lazy: true, run(S) {
                const mk = (label, g, slip, rstep) => {
                    const env = P10.makeEnv(P10.MAPS.main, { slip, rstep }), vi = P10.valueIteration(env, g), r = P10D.routeOf(env, vi.pi);
                    let R = 0.25; for (let s = 0; s < env.N; s++) R = Math.max(R, Math.abs(vi.V[s]));
                    return { label, env, vi, r, R };
                };
                S.reg = [mk('当前设置', S.gamma, S.slip, S.rstep), mk('slip = 0', S.gamma, 0, S.rstep), mk('γ = 0.8', 0.8, S.slip, S.rstep), mk('每步 −0.2', S.gamma, S.slip, -0.2), mk('每步 +0.05', S.gamma, S.slip, 0.05), mk('slip = 0.4', S.gamma, 0.4, S.rstep)];
            },
        },
        {
            id: 'mc', lifetime: 'frame', lazy: true, run(S) {
                const key = S.epoch + '|' + S.sel;
                if (S.mcKey === key) return;
                const env = S.env, s0 = S.sel, rand = LM.rng(P10D.seedOf(S, 3) + s0), e = { S: [], A: [], R: [], D: [], T: 0 }, G = new Float64Array(600);
                for (let k = 0; k < 600; k++) { P10.sampleEpisode(env, S.uni, rand, 400, e, s0); G[k] = P10.returnsOf(e, S.gamma)[0]; }
                S.mc = { s0, G, exact: S.Vrand[s0] }; S.mcKey = key;
            },
        },
        { id: 'bTrace', lifetime: 'param', lazy: true, run(S) { S.btr = P10D.banditTrace(S, 8); } },
        { id: 'pgvar', lifetime: 'param', lazy: true, run(S) { S.pgv = P10D.pgVariance(S, 600); } },
        { id: 'gae', lifetime: 'param', lazy: true, run(S) { S.gae = P10D.gaeScan(S, 200, 400); } },
        {
            id: 'play', lifetime: 'frame', run(S) {
                const tl = S.walk.tl, [e, t] = tl[Math.floor(S.t * 5) % tl.length];
                S.wk = { e, t };
                S.peK = P10D.cycle(S, 60, 6, 2);
                S.viK = P10D.cycle(S, Math.min(60, S.vi.Vs.length - 1), 5, 2.5);
                S.piK = P10D.cycle(S, S.piSeq.length - 1, 0.8, 1.5);
                S.mcN = Math.min(600, 1 + Math.floor(S.t * 30) % 690);
                S.btT = P10D.cycle(S, P10D.BANDIT_T - 1, 50, 2);
            },
        },
        { id: 'ql', lifetime: 'history', lazy: true, run(S) { p10Live(S, 'qlSt', (S2) => { const st = P10D.qlInit(S2); P10D.qlEval(S2, st); return st; }, P10D.qlAdvance, P10D.RATE.ql); } },
        { id: 'cliff', lifetime: 'history', lazy: true, run(S) { p10Live(S, 'clSt', P10D.cliffInit, P10D.cliffAdvance, P10D.RATE.cliff); } },
        { id: 'bandit', lifetime: 'history', lazy: true, run(S) { p10Live(S, 'bdSt', P10D.banditInit, P10D.banditAdvance, P10D.RATE.bandit); } },
        { id: 'dqn', lifetime: 'history', lazy: true, run(S) { p10Live(S, 'dqnSt', P10D.dqnInitS, P10D.dqnAdvance, P10D.RATE.dqn); } },
        {
            id: 'dqnGen', lifetime: 'frame', lazy: true, needs: ['dqn'], run(S) {
                const st = S.dqnSt; if (!st) return;
                const key = [S.epoch, st.t, S.genS].join('|');
                if (S.genKey !== key) { S.gen = P10D.dqnGeneralize(S, st, S.genS); S.genKey = key; }
            },
        },
        {
            id: 'pg', lifetime: 'history', lazy: true, run(S) {
                p10Live(S, 'pgSt', (S2) => ({ groups: [P10D.pgGroup(S2, '不用基线', LM.C.red, P10D.NSEED.pg, { baseline: false }, P10.reinforceStep), P10D.pgGroup(S2, '基线 V(s)', LM.C.green, P10D.NSEED.pg, { baseline: true }, P10.reinforceStep)] }),
                    (S2, st, n) => P10D.groupsAdvance(S2, st.groups, n, P10D.CAP.pg), P10D.RATE.pg);
            },
        },
        {
            id: 'ac', lifetime: 'history', lazy: true, run(S) {
                p10Live(S, 'acSt', (S2) => ({ groups: [P10D.pgGroup(S2, 'REINFORCE + 基线', LM.C.blue, P10D.NSEED.ac, { baseline: true }, P10.reinforceStep), P10D.pgGroup(S2, '单步 actor-critic', LM.C.green, P10D.NSEED.ac, {}, P10.acEpisode)] }),
                    (S2, st, n) => P10D.groupsAdvance(S2, st.groups, n, P10D.CAP.ac), P10D.RATE.ac);
            },
        },
        { id: 'ppo', lifetime: 'history', lazy: true, run(S) { p10Live(S, 'ppoSt', (S2) => ({ groups: P10D.ppoInitS(S2) }), (S2, st, n) => P10D.ppoAdvance(S2, st.groups, Math.min(n, P10D.CAP.ppo + 1)), P10D.RATE.ppo); } },
    ],

    intro: {
        title: '强化学习：没有标准答案，只有奖励',
        flow: ['MDP：状态 · 动作 · 奖励 · 折扣', '价值 V、Q 与贝尔曼方程', '已知模型：价值迭代 / 策略迭代', '无模型：Q-learning / SARSA', '探索：ε-greedy · UCB', '函数逼近：DQN', '策略梯度：REINFORCE → Actor-Critic → PPO'],
        html: LM.navHtml('10') + `
            <p>前面几篇都是<strong>监督学习</strong>：每个输入都配着标准答案。<strong>强化学习（Reinforcement Learning, RL）</strong>里没有人告诉你「这一步该怎么走」，只有环境偶尔给出的<strong>奖励</strong>，而且常常来得很晚——走到终点才 +1，之前的每一步到底是功是过，要智能体自己从结果里倒推。游戏 AI、机器人控制、AlphaGo，以及第 06 篇让语言模型「对齐」的 RLHF / GRPO，都建立在这一页的几个概念上：<strong>价值函数、贝尔曼方程、时序差分、探索、策略梯度</strong>。</p>
            <p>全页用同一个 8 × 6 的<strong>网格世界</strong>（左下 S 出发，右边 G = +1，两个红色陷阱 = −1，地面会滑）。它小到可以用动态规划<strong>精确</strong>求出最优解 V*、Q*、π*，于是后面每个「学出来」的东西——Q-learning 的 Q 表、DQN 的网络、策略梯度的策略——都能和真值逐格比较，看清它们差在哪里、为什么。</p>
            <p><strong>哪些是真的</strong>：每张预览都由页面里的 JS 参考实现实时算出；Step 05–11 的 Q-learning、SARSA、赌博机、DQN（LM.AG 上的小 MLP）、REINFORCE、actor-critic、PPO 在浏览器里<strong>实时训练</strong>，只在对应预览可见时训练（顶栏「训练速度」可加速，「⏩ 训练到底」直接跑完）；评估用的 J = V^π(S) 是已知模型下<strong>精确</strong>算的，不是采样估计。代码块是同一算法的 <strong>C++ 版本</strong>（C++17 + 本系列自带的 minml 库），和页面 JS 用同一个随机数发生器（mulberry32）、同样的运算顺序：同一个种子下，表格方法的结果与 JS <strong>逐位一致</strong>（用 <code>tools/p10_cpp_run.js</code> 编译运行核对过；DQN 的网络因浮点精度不同只能近似一致）。</p>`,
    },

    steps: [
        {
            id: 'mdp', title: 'MDP：智能体、环境与奖励', tag: 'MDP',
            desc: '强化学习的设定是一个循环：智能体看到<strong>状态</strong> s，选一个<strong>动作</strong> a；环境返回<strong>奖励</strong> r 和下一个状态 s′；如此往复，直到回合结束。把这套规则写下来就是<strong>马尔可夫决策过程（Markov Decision Process, MDP）</strong>：状态集 S、动作集 A、转移模型 P(s′ | s, a)、奖励 R、折扣 γ。「马尔可夫」指下一步只取决于<strong>当前</strong>的状态和动作，与之前怎么走到这里无关。本页的网格世界有 42 个状态（其中 4 个终止格）、4 个动作；进目标 +1、进陷阱 −1，回合结束；其余每走一步得到一个小小的「生活成本」r_step；地面会滑：以 1 − slip 的概率按意图走，各以 slip/2 的概率滑向两侧（Russell & Norvig 的经典设定），撞墙原地不动。智能体的「大脑」叫<strong>策略</strong> π(a | s)：每个状态下选各个动作的概率。下面的黄点用的是最笨的策略——四个方向各 1/4 随机乱走。',
            formula: [
                'MDP = (S, A, P, R, γ)      π(a | s)：策略 = 在状态 s 选动作 a 的概率',
                'P(s′ = 意图方向 | s, a) = 1 − slip      P(s′ = 左侧) = P(s′ = 右侧) = slip / 2      // 撞墙 → s′ = s',
                'τ = (s_0, a_0, r_1, s_1, a_1, r_2, s_2, …)      // 一条轨迹；终止格之后没有未来',
            ],
            code: { title: 'C++ · mainMap / sampleEpisode（智能体–环境循环）', src: P10CPP.mdp },
            previews: [
                {
                    id: 'walk', live: true, large: true, size: [360, 300], label: '随机策略在网格世界里乱走', sub: '黄点 = 智能体 · 黄线 = 本回合走过的路 · <b>点击格子</b>选中它（右边看它的转移概率，Step 02 从它出发采样）',
                    render: (c, S) => {
                        const ep = S.walk.eps[S.wk.e], t = Math.min(S.wk.t, ep.T), path = ep.S.slice(0, t + 1), s = path[path.length - 1];
                        let R = 0; for (let k = 0; k < t; k++) R += ep.R[k];
                        const end = t >= ep.T ? (S.env.kind[s] === 'G' ? ' · 到达目标' : S.env.kind[s] === 'X' ? ' · 掉进陷阱' : ' · 60 步截断') : '';
                        S._lWalk = P10D.drawGrid(c, S.env, { path, agent: s, hi: S.sel, title: `回合 #${S.wk.e + 1} · 第 ${t} 步`, titleX: 46, sub: `奖励和 ${P10D.sgn(R, 2)}${end}` });
                    },
                    probe: (x, y, S, PL, info) => { const s = P10D.hitCell(S._lWalk, S.env, info); return s < 0 ? '墙' : `${P10D.cellName(S.env, s)} ${({ G: '目标 +1', X: '陷阱 −1', S: '起点' })[S.env.kind[s]] || ''}`; },
                    pick: (x, y, S, PL, info) => { const s = P10D.hitCell(S._lWalk, S.env, info); if (s >= 0 && !S.env.term[s]) { S.sel = s; return 'frame'; } },
                },
                {
                    id: 'trans', size: [360, 150], label: '转移概率 P(s′ | s, a)', sub: '选中的格子（中心）分别选 ↑ → ↓ ← 时，落到周围哪一格、概率多少 · 撞墙的概率记在中心',
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c, { bg: '#0b0d10' }), env = S.env, s = S.sel, [sx, sy] = env.xy(s), cs = 23, pw = 3 * cs, gap = (w - 4 * pw) / 5;
                        ctx.fillStyle = LM.C.text; ctx.font = LM.FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`格子 ${P10D.cellName(env, s)} · slip = ${S.slip}`, 4, 3);
                        for (let a = 0; a < 4; a++) {
                            const X0 = gap + a * (pw + gap), Y0 = 34, prob = new Map();
                            env.T[s * 4 + a].forEach(([p, s2]) => prob.set(s2, (prob.get(s2) || 0) + p));
                            let er = 0; env.T[s * 4 + a].forEach(([p, , r]) => { er += p * r; });
                            for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
                                const x = sx + dx, y = sy + dy, X = X0 + (dx + 1) * cs, Y = Y0 + (dy + 1) * cs, inside = x >= 0 && y >= 0 && x < env.W && y < env.H;
                                const ch = inside ? env.cell[y * env.W + x] : '#', s2 = inside ? env.sid[y * env.W + x] : -1;
                                ctx.fillStyle = ch === '#' ? '#20242b' : ch === 'G' ? '#163d24' : ch === 'X' ? '#4a161b' : '#161c23'; ctx.fillRect(X + 0.5, Y + 0.5, cs - 1, cs - 1);
                                const p = s2 >= 0 ? prob.get(s2) : 0;
                                if (p > 0) { ctx.fillStyle = dx === 0 && dy === 0 ? LM.C.orange : LM.C.yellow; ctx.font = LM.FONT(11, true); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(p >= 0.995 ? '1' : p.toFixed(2).replace(/^0/, ''), X + cs / 2, Y + cs / 2 + 0.5); }
                            }
                            ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 1.5; ctx.strokeRect(X0 + cs + 0.5, Y0 + cs + 0.5, cs - 1, cs - 1);
                            ctx.fillStyle = LM.C.text; ctx.font = LM.FONT(13, true); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(`a = ${P10.ARROW[a]}`, X0 + pw / 2, 17);
                            ctx.fillStyle = LM.C.gray; ctx.font = LM.FONT(11); ctx.fillText(`E[r] = ${P10D.sgn(er, 2)}`, X0 + pw / 2, Y0 + pw + 4);
                        }
                    },
                },
                {
                    id: 'walkStats', size: [360, 150], label: '随机策略的 2000 个回合都怎么结束', sub: '每回合最多 200 步 · 同时得到 V^π(S) 的蒙特卡洛估计（下一步会精确地算它）',
                    render: (c, S) => {
                        const st = S.walk.stats, n = st.n;
                        LM.draw.bars(c, [st.goal / n, st.near / n, st.other / n, st.timeout / n].map((v) => 100 * v), ['到达目标 +1', '起点旁的陷阱', '另外两个陷阱', '200 步没结束'], { horizontal: true, colors: [LM.C.green, LM.C.red, '#b0565e', LM.C.gray], digits: 1, valueSuffix: '%', range: [0, 100], title: `平均 ${st.len.toFixed(1)} 步 · 平均折扣回报 ${P10D.sgn(st.G)}` });
                    },
                },
            ],
            notes: '点击左图任意格子，中图显示在那一格选 ↑ → ↓ ← 各会落到哪里。把 <b>滑动概率</b> 拖到 0：每个动作都只有一个结果；拖到 0.5：意图方向只有一半机会。随机乱走的智能体大多数回合死在起点右边那个陷阱里——它离起点只有 2 步。',
            warn: '奖励只是环境给的信号，我们真正要最大化的是整个回合<b>累积</b>的（折扣）奖励，下一步定义。怎么设计奖励本身就是 RL 的大坑：Step 03 会看到，把每步奖励改得太负，最优策略会变成「直接跳进陷阱」。',
        },
        {
            id: 'eval', title: '回报与价值：评估一个策略', tag: 'EVAL',
            desc: '<strong>回报（return）</strong> G_t 是从 t 时刻往后所有奖励的折扣和。γ < 1 让无限长的回合也有有限的回报，也表达「越晚到手的奖励越不值钱」：有效视野约 1/(1 − γ) 步（γ = 0.95 → 约 20 步）。<strong>价值函数</strong> V^π(s) 是从 s 出发、按策略 π 走下去的期望回报。它满足<strong>贝尔曼期望方程</strong>：一个格子的价值 = 走一步的期望奖励 + γ × 下一格的价值（按 π 和 P 加权平均）。这是 N 个未知数的线性方程组，可以直接解（高斯消元），也可以把它当成更新规则从 V = 0 反复扫描——<strong>迭代策略评估</strong>：第 k 轮的 V_k 恰好是「只看 k 步」的价值，误差每轮缩小到 γ 倍左右。不知道 P 时，只能真的从 s 出发采样很多回合、对回报取平均（<strong>蒙特卡洛</strong>，Monte Carlo）：无偏，但回报的方差很大（右下：掉进陷阱 ≈ −1，走到目标是正数），收敛按 1/√n 走，很慢。',
            formula: [
                'G_t = r_{t+1} + γ·r_{t+2} + γ^2·r_{t+3} + … = Σ_{k≥0} γ^k · r_{t+k+1}',
                'V^π(s) = E_π[ G_t | s_t = s ]      Q^π(s, a) = E_π[ G_t | s_t = s, a_t = a ]',
                'V^π(s) = Σ_a π(a|s) · Σ_{s′} P(s′|s, a) · [ r + γ·V^π(s′) ]   // 贝尔曼期望方程',
                'V^π = (I − γ·P_π)^{−1} · r_π   // 直接解（高斯消元）',
                'V_{k+1} = r_π + γ·P_π·V_k      ‖V_k − V^π‖∞ ≤ γ^k · ‖V_0 − V^π‖∞   // 迭代：γ-收缩',
            ],
            code: { title: 'C++ · discountedReturns / qFromV / policyEvaluation / evaluatePolicy / monteCarloValue', src: P10CPP.evalPolicy },
            previews: [
                {
                    id: 'peV', live: true, large: true, size: [360, 300], label: '均匀随机策略的价值 V^π：迭代策略评估', sub: '第 k 轮 = 从 V = 0 开始做了 k 次贝尔曼备份 · 蓝 = 负，橙 = 正 · <b>点击格子</b>：右下从它出发做蒙特卡洛',
                    render: (c, S) => { const k = S.peK; S._lPe = P10D.drawGrid(c, S.env, { V: S.pe.Vs[k], R: S.Rr, numbers: true, hi: S.sel, title: `迭代策略评估 · 第 ${k} 轮`, titleX: 46, sub: `‖V_k − V^π‖∞ = ${S.peErr[k].toFixed(3)}` }); },
                    probe: (x, y, S, PL, info) => { const s = P10D.hitCell(S._lPe, S.env, info); return s < 0 || S.env.term[s] ? '' : `${P10D.cellName(S.env, s)}  V_k = ${S.pe.Vs[S.peK][s].toFixed(4)}  V^π = ${S.Vrand[s].toFixed(4)}`; },
                    pick: (x, y, S, PL, info) => { const s = P10D.hitCell(S._lPe, S.env, info); if (s >= 0 && !S.env.term[s]) { S.sel = s; return 'frame'; } },
                },
                {
                    id: 'peConv', live: true, size: [360, 190], label: '迭代评估的收敛', sub: '对数纵轴 · 蓝 = 离真值的误差 ‖V<sub>k</sub> − V<sup>π</sup>‖<sub>∞</sub> · 灰虚线 = 收缩上界 γ<sup>k</sup>·‖V<sub>0</sub> − V<sup>π</sup>‖<sub>∞</sub>',
                    render: (c, S) => {
                        const K = S.peErr.length - 1, xs = S.peErr.map((_, i) => i), e0 = S.peErr[0];
                        PL.draw.plot(c, { x: [0, K], y: [1e-4, Math.max(30, e0 * 30)], logY: true, xLabel: '扫描轮数 k', series: [{ data: xs.map((k) => e0 * Math.pow(S.gamma, k)), x: xs, color: LM.C.gray, dash: [4, 4], label: `γᵏ·误差₀（γ = ${S.gamma}）` }, { data: S.peErr.map((v) => Math.max(v, 1e-4)), x: xs, color: LM.C.blue, width: 2, label: '‖Vₖ − V^π‖∞' }], marks: [{ x: S.peK, label: `k = ${S.peK}` }] });
                    },
                },
                {
                    id: 'mc', live: true, needs: ['mc'], size: [360, 200], label: '蒙特卡洛估计：从选中格子出发的回报平均值', sub: '蓝 = 前 n 个回合的平均 · 浅蓝 = ±2 个标准误 · 绿 = 精确的 V^π（线性方程组）',
                    render: (c, S) => {
                        const m = S.mc, n = S.mcN, mean = [], lo = [], hi = [], xs = [];
                        let s1 = 0, s2 = 0;
                        for (let k = 0; k < n; k++) { const g = m.G[k]; s1 += g; s2 += g * g; const mu = s1 / (k + 1), sd = Math.sqrt(Math.max(0, s2 / (k + 1) - mu * mu)), se = 2 * sd / Math.sqrt(k + 1); mean.push(mu); lo.push(mu - se); hi.push(mu + se); xs.push(k + 1); }
                        const y0 = m.exact - 0.6, y1 = m.exact + 0.6;
                        PL.draw.plot(c, { x: [1, 600], y: [y0, y1], xLabel: `回合数 n（从 ${P10D.cellName(S.env, m.s0)} 出发）`, series: [{ data: lo.map((v) => PL.clamp(v, y0, y1)), x: xs, color: '#2f5b7a', width: 1 }, { data: hi.map((v) => PL.clamp(v, y0, y1)), x: xs, color: '#2f5b7a', width: 1 }, { data: mean.map((v) => PL.clamp(v, y0, y1)), x: xs, color: LM.C.blue, width: 2, label: `平均 ${P10D.sgn(mean[n - 1])}` }], hmarks: [{ y: m.exact, color: LM.C.green, label: `V^π = ${P10D.sgn(m.exact)}` }] });
                    },
                },
                {
                    id: 'mcHist', needs: ['mc'], size: [360, 150], label: '这 600 个回报的分布', sub: '一个回报 = 一次掷骰子：左边一堆 ≈ 很快掉进陷阱，右边是走到了目标',
                    render: (c, S) => {
                        const G = Array.from(S.mc.G), lo = Math.min(...G) - 0.02, hi = Math.max(...G) + 0.02, nb = 36, h = P10D.histo(G, lo, hi, nb);
                        PL.draw.plot(c, { x: [lo, hi], y: [0, Math.max(...h) * 1.15], xLabel: '折扣回报 G', series: [{ data: h, color: LM.C.blue, bar: true }], marks: [{ x: S.mc.exact, label: 'V^π', color: LM.C.green }] });
                    },
                },
            ],
            notes: '默认 γ = 0.95：随机策略从起点出发的价值只有 V^π(S) ≈ −0.89——它多半会掉进旁边的陷阱。看迭代评估怎样一圈一圈地把陷阱的「坏消息」传开：第 k 轮的值只考虑了 k 步之内的事。把 <b>折扣 γ</b> 拖到 0.5：收敛快得多（误差每轮减半），远处格子的价值只剩生活成本 r_step/(1 − γ)。点击左图不同的格子，看右下的蒙特卡洛估计：600 个回合后误差仍有 ±0.05 左右。',
            warn: 'γ 越接近 1，迭代评估越慢（误差每轮只缩小到 γ 倍），γ = 1 且策略可能永远不结束时 V^π 根本不存在（每步都在扣分），所以本页 γ ≤ 0.99。代码里的价值是「从这一格出发」的期望；终止格的价值定义为 0。',
        },
        {
            id: 'vi', title: '贝尔曼最优方程与价值迭代', tag: 'DP',
            desc: '我们要的不是评估一个固定策略，而是找<strong>最好的</strong>策略。最优价值 V*(s) 是所有策略里能拿到的最大期望回报，它满足<strong>贝尔曼最优方程</strong>（Bellman 1957）：把期望方程里「按 π 平均」换成「挑最好的动作」(max)。<strong>价值迭代</strong>就是反复做这个「最优备份」：每一轮对每个格子算出 4 个动作的一步前瞻 Q 值，取最大的当新的 V。这个算子是 γ-收缩的，所以从任何初值出发都收敛到唯一的 V*。有了 V*，对 Q* 贪心就是最优策略 π*。注意左图的橙框（贪心动作还不是最优的格子）：价值离收敛还远时，策略往往早就对了——这是下一步策略迭代更快的原因。右下：同一张地图，改一下环境或奖励，最优行为就截然不同——全是同一份价值迭代代码算出来的。',
            formula: [
                'V*(s) = max_a Σ_{s′} P(s′|s, a) · [ r + γ·V*(s′) ]   // 贝尔曼最优方程',
                'Q*(s, a) = Σ_{s′} P(s′|s, a) · [ r + γ·max_{a′} Q*(s′, a′) ]      π*(s) = argmax_a Q*(s, a)',
                'V_{k+1}(s) = max_a Σ_{s′} P · [ r + γ·V_k(s′) ]      ‖V_k − V*‖∞ ≤ γ^k · ‖V_0 − V*‖∞',
            ],
            code: { title: 'C++ · greedyPolicy / valueIteration', src: P10CPP.vi },
            previews: [
                {
                    id: 'viV', live: true, large: true, size: [360, 300], label: '价值迭代：V_k 与贪心策略', sub: '箭头 = 对 V_k 一步前瞻的贪心动作 · 橙框 = 这个动作还不是最优 · 黄线 = 从 S 出发按箭头走的路线',
                    render: (c, S) => {
                        const k = S.viK, act = S.viAct[k], r = P10D.routeOf(S.env, act), bad = S.viBad[k];
                        S._lVi = P10D.drawGrid(c, S.env, { V: S.vi.Vs[k], R: S.Rv, numbers: true, act, path: r.path, marks: bad, title: `价值迭代 · 第 ${k} 轮`, titleX: 46, sub: bad.size ? `${bad.size} 格非最优 · 误差 ${S.viErr[k].toFixed(3)}` : `策略已最优 · 误差 ${S.viErr[k].toExponential(1)}` });
                    },
                    probe: (x, y, S, PL, info) => { const s = P10D.hitCell(S._lVi, S.env, info); if (s < 0 || S.env.term[s]) return ''; const q = Array.from({ length: 4 }, (_, a) => S.vi.Q[s * 4 + a]); return `${P10D.cellName(S.env, s)}  V* = ${S.vi.V[s].toFixed(4)}\nQ*: ${q.map((v, a) => P10.ARROW[a] + ' ' + v.toFixed(3)).join('  ')}`; },
                },
                {
                    id: 'viConv', live: true, size: [360, 190], label: '价值迭代的收敛', sub: '对数纵轴 · 蓝 = ‖V<sub>k</sub> − V*‖<sub>∞</sub> · 灰虚线 = γ<sup>k</sup> 上界 · 绿线 = 贪心策略第一次全部最优的那一轮',
                    render: (c, S) => {
                        const K = S.viErr.length - 1, xs = S.viErr.map((_, i) => i), e0 = S.viErr[0], marks = [{ x: S.viK, label: `k = ${S.viK}` }];
                        if (S.viOptAt >= 0) marks.push({ x: S.viOptAt, label: '策略已最优', color: LM.C.green });
                        PL.draw.plot(c, { x: [0, K], y: [1e-6, Math.max(100, e0 * 100)], logY: true, xLabel: `扫描轮数 k（收敛到 10⁻¹⁰ 共 ${S.vi.iters} 轮）`, series: [{ data: xs.map((k) => e0 * Math.pow(S.gamma, k)), x: xs, color: LM.C.gray, dash: [4, 4], label: 'γᵏ·误差₀' }, { data: S.viErr.map((v) => Math.max(v, 1e-6)), x: xs, color: LM.C.blue, width: 2, label: '‖Vₖ − V*‖∞' }], marks });
                    },
                },
                {
                    id: 'regimes', needs: ['regimes'], size: [360, 430], label: '同一张地图，不同的环境 / 奖励 → 不同的最优策略', sub: '每格 = 一次完整的价值迭代 · 黄线 = 从 S 出发的路线 · 其余参数同顶栏',
                    render: (c, S) => {
                        const tag = (r) => (r.end === 'G' ? `走到目标 ${r.len} 步` : r.end === 'X' ? `跳进陷阱 ${r.len} 步` : '永不结束');
                        P10D.gridPanels(c, S.reg.map((g, i) => ({ env: g.env, label: `${g.label} · ${tag(g.r)}`, labelColor: i === 0 ? LM.C.yellow : g.r.end === 'X' ? LM.C.red : g.r.end === '∞' ? LM.C.orange : LM.C.text, o: { V: g.vi.V, R: g.R, act: g.vi.pi, path: g.r.path } })), 2);
                    },
                },
            ],
            notes: '默认（γ 0.95 · slip 0.2 · 每步 −0.04）：最优路线绕最上面一行（14 步），避开中间夹在两个陷阱之间的「独木桥」——过桥那一步有 20% 的概率滑进陷阱。把 <b>滑动概率</b> 拖到 0.15 或更低，或把 <b>折扣 γ</b> 拖到 0.9 或更低：改走独木桥（10 步）——前者是风险变小了，后者是「没耐心」：远期的 +1 被折得太狠，多走 4 步不再划算。<b>每步奖励</b> −0.08 左右：着急，也走桥；−0.15 及以下：从起点直接往右两步跳进陷阱——活着的每一步都在扣分，−1 一次了断反而划算；大于 0：永远不去终点。',
            warn: '奖励写错了，最优策略会一丝不苟地执行错误的目标——这是「奖励欺骗（reward hacking）」最简单的版本，第 06 篇会在 RLHF 里再见到它。价值迭代需要完整的 P 和 R，每轮代价 O(|S|·|A|·分支数)；状态一多（围棋约 10¹⁷⁰ 个局面）就不可能了。',
        },
        {
            id: 'piter', title: '策略迭代：评估 → 改进 → 再评估', tag: 'DP',
            desc: '<strong>策略迭代</strong>（Howard 1960）交替做两件事：先<strong>精确评估</strong>当前策略（Step 02 的线性方程组），再对 Q^π <strong>贪心改进</strong>——每个格子换成一步前瞻最好的动作。<strong>策略改进定理</strong>保证新策略在每个格子上都不比旧的差；确定性策略只有有限个，所以有限步内一定停下，停下时策略满足贝尔曼最优方程，就是 π*。从最笨的「处处向上」出发，几轮就到了。每一轮比价值迭代贵（要解一个 42 元方程组），但轮数少得多。「评估—改进」交替的思想叫<strong>广义策略迭代（GPI）</strong>，后面的方法全是它的变体：Q-learning 一边估 Q、一边按 Q 贪心；actor-critic 里 critic 评估、actor 改进。',
            formula: [
                '评估：V^{π_k} = (I − γ·P_{π_k})^{−1} · r_{π_k}      改进：π_{k+1}(s) = argmax_a Q^{π_k}(s, a)',
                'V^{π_{k+1}}(s) ≥ V^{π_k}(s)  对所有 s   // 策略改进定理',
                '停止 ⇔ π_k 对自己的 Q^{π_k} 已经是贪心的 ⇔ 满足贝尔曼最优方程 ⇔ π_k = π*',
            ],
            code: { title: 'C++ · onehotPolicy / policyIteration', src: P10CPP.pi },
            previews: [
                {
                    id: 'piSeq', live: true, large: true, size: [360, 300], label: '策略迭代的每一轮', sub: '颜色 / 数字 = 精确的 V<sup>π<sub>k</sub></sup> · 箭头 = π<sub>k</sub> · 橙框 = 改进这一步要改的格子',
                    render: (c, S) => {
                        const k = S.piK, q = S.piSeq[k], nxt = S.piSeq[Math.min(k + 1, S.piSeq.length - 1)].act, ch = new Set();
                        for (let s = 0; s < S.env.N; s++) if (!S.env.term[s] && nxt[s] !== q.act[s]) ch.add(s);
                        P10D.drawGrid(c, S.env, { V: q.V, R: S.Rv, numbers: true, act: q.act, marks: ch, path: P10D.routeOf(S.env, q.act).path, title: `策略迭代 · π${'₀₁₂₃₄₅₆₇₈₉'[k] || '_' + k}${k === S.piSeq.length - 1 ? '（= π*，不再变化）' : ''}`, titleX: 46, sub: `V(S) = ${P10D.sgn(q.V[S.env.start])} · 下一轮改 ${q.changed} 格` });
                    },
                },
                {
                    id: 'piCurve', size: [360, 200], label: '起点的价值：策略迭代 vs 价值迭代', sub: '横轴 = 轮数（策略迭代：评估 + 改进；价值迭代：一次扫描）· 绿 = V*(S)',
                    render: (c, S) => {
                        const pi = S.piSeq.map((q) => q.V[S.env.start]), vi = S.vi.Vs.map((V) => V[S.env.start]), n = Math.min(40, vi.length);
                        const lo = Math.min(...pi, ...vi.slice(0, n), S.vi.V[S.env.start]) - 0.05, top = Math.max(0, ...vi.slice(0, n), S.vi.V[S.env.start]), hi = top + 0.45 * (top - lo);
                        PL.draw.plot(c, { x: [0, n - 1], y: [lo, hi], xLabel: `轮数（策略迭代 ${S.piSeq.length - 1} 次改进；价值迭代收敛要 ${S.vi.iters} 轮）`, series: [{ data: vi.slice(0, n), x: vi.slice(0, n).map((_, i) => i), color: LM.C.blue, width: 2, label: '价值迭代 Vₖ(S)' }, { data: pi, x: pi.map((_, i) => i), color: LM.C.orange, width: 2.5, label: '策略迭代 V^πₖ(S)' }], hmarks: [{ y: S.vi.V[S.env.start], color: LM.C.green, label: `V*(S) = ${P10D.sgn(S.vi.V[S.env.start])}` }] });
                    },
                },
            ],
            notes: '默认设置下从「处处向上」出发：6 次改进就得到和价值迭代完全相同的 π*（逐格一致）。π₀、π₁、π₂ 的起点价值几乎不动（−0.80 ≈ 永远撞墙的 r_step/(1 − γ)）——改进是从目标附近开始、一轮轮往回传的；到 π₃ 路线接通，起点价值一下子跳到 −0.17，再两轮就是最优的 −0.147。改 <b>折扣 γ</b> / <b>滑动概率</b>：轮数基本不变，这是策略迭代的典型特点（实践中通常几轮到十几轮）。',
            warn: '这里的「精确评估」是 O(N³) 的高斯消元。状态多时改成只扫几轮迭代评估（modified policy iteration），价值迭代就是「每次只扫一轮」的极端情况。',
        },
        {
            id: 'qlearn', title: 'Q-learning：不知道模型，从经验里学', tag: 'TD',
            desc: '前两步都偷看了 P 和 R。真实问题里通常没有模型：机器人不知道摩擦系数，游戏 AI 拿不到游戏源码。<strong>无模型（model-free）</strong>方法只用和环境交互得到的样本 (s, a, r, s′)。<strong>Q-learning</strong>（Watkins 1989）把贝尔曼最优方程里的期望换成刚刚得到的那一个样本：<strong>TD 目标</strong> r + γ·max Q(s′, ·)，然后把 Q(s, a) 往目标挪一小步 α。这叫<strong>时序差分（Temporal Difference, TD）</strong>：用自己对下一步的估计来更新这一步（自举 bootstrapping），不用等回合结束，方差也比蒙特卡洛小得多。选动作用 <strong>ε-greedy</strong>：大多数时候按当前 Q 走，以概率 ε 随机探索。Q-learning 学的是<strong>贪心</strong>策略的价值，数据却来自 ε-greedy——学习的策略和收集数据的策略不同，这叫<strong>离策略（off-policy）</strong>。只要每个 (s, a) 被访问无限次、步长逐渐变小（Robbins–Monro 条件），Q 以概率 1 收敛到 Q*（Watkins & Dayan 1992）。',
            formula: [
                'Q(s, a) ← Q(s, a) + α · [ r + γ·max_{a′} Q(s′, a′) − Q(s, a) ]   // 方括号里 = TD 误差 δ',
                'a = 随机动作（概率 ε），否则 argmax_a Q(s, a)   // ε-greedy；并列时随机挑',
                'J(π) = V^π(S)   // 评估用：已知模型下精确算出「当前贪心策略」和「ε-greedy 策略」从起点出发的价值',
            ],
            code: { title: 'C++ · argmaxRandom / epsilonGreedy / qLearningEpisode', src: P10CPP.ql },
            previews: [
                {
                    id: 'qlQ', live: true, large: true, size: [360, 300], needs: ['ql'], label: 'Q 表：每格 4 个三角形 = 4 个动作的 Q(s, a)', sub: '色标与 Step 03 的 V* 相同 · 白箭头 = 贪心动作 · 黄线 = 最近一个回合（ε-greedy）实际走的路',
                    render: (c, S) => {
                        const st = S.qlSt;
                        S._lQl = P10D.drawGrid(c, S.env, { Q: st.Q, R: S.Rv, act: P10.greedy(S.env, st.Q), path: st.path, dots: false, title: `Q-learning · 第 ${st.ep} 回合${st.ep >= P10D.CAP.ql ? '（已停）' : ''}`, titleX: 46, sub: `α = ${S.alpha} · ε = ${S.eps}` });
                    },
                    probe: (x, y, S, PL, info) => { const s = P10D.hitCell(S._lQl, S.env, info); if (s < 0 || S.env.term[s] || !S.qlSt) return ''; const Q = S.qlSt.Q; return `${P10D.cellName(S.env, s)}  Q:  ${[0, 1, 2, 3].map((a) => P10.ARROW[a] + ' ' + Q[s * 4 + a].toFixed(3)).join('  ')}\nQ*: ${[0, 1, 2, 3].map((a) => P10.ARROW[a] + ' ' + S.vi.Q[s * 4 + a].toFixed(3)).join('  ')}`; },
                },
                {
                    id: 'qlCurve', live: true, size: [360, 210], needs: ['ql'], label: '学到的策略有多好：J = V^π(S)（精确）', sub: '每 20 回合评估一次 · 蓝 = 贪心策略 · 橙 = 实际在用的 ε-greedy 策略 · 两者之差 = 探索的代价',
                    render: (c, S) => {
                        const ev = S.qlSt.ev, xs = ev.map((e) => e.ep), yr = P10D.jRange(S);
                        PL.draw.plot(c, { x: [0, P10D.CAP.ql], y: yr, xLabel: `回合（第 ${S.qlSt.ep} / ${P10D.CAP.ql}）`, series: [{ data: ev.map((e) => e.Je), x: xs, color: LM.C.orange, width: 1.6, label: 'ε-greedy 策略' }, { data: ev.map((e) => e.J), x: xs, color: LM.C.blue, width: 2, label: '贪心策略' }], hmarks: P10D.jMarks(S), marks: [{ x: S.qlSt.ep, color: '#5a5f66' }] });
                    },
                },
                {
                    id: 'qlErr', live: true, size: [360, 170], needs: ['ql'], label: 'Q 表离 Q* 还有多远', sub: '紫 = RMS(max_a Q − V*)（非终止格）· 绿 = 贪心动作已是最优动作的格子比例',
                    render: (c, S) => {
                        const ev = S.qlSt.ev, xs = ev.map((e) => e.ep);
                        PL.draw.plot(c, { x: [0, P10D.CAP.ql], y: [0, 1.5], xLabel: `回合（第 ${S.qlSt.ep} 回合）`, series: [{ data: ev.map((e) => Math.min(1, e.rms)), x: xs, color: LM.C.purple, width: 2, label: 'RMS 误差' }, { data: ev.map((e) => e.opt), x: xs, color: LM.C.green, width: 2, label: '最优动作占比' }], hmarks: [{ y: 1, color: '#3a3f46' }] });
                    },
                },
                {
                    id: 'qlVisit', live: true, size: [360, 230], needs: ['ql'], label: '每个格子被访问了多少次', sub: '对数色标（黄 = 多、紫 = 少）· 探索得少的格子，Q 也学得差（对照上图三角形的颜色）',
                    render: (c, S) => {
                        const v = S.qlSt.visits, lv = Float64Array.from(v, (x) => Math.log10(1 + x)), mx = Math.max(1, ...lv), rgb = [0, 0, 0];
                        P10D.drawGrid(c, S.env, { colorOf: (s) => LM.rgbStr(PL.cmap.viridis(0.08 + 0.92 * lv[s] / mx, rgb)), title: `最多 ${Math.round(Math.max(...v)).toLocaleString()} 次 · 最少 ${Math.round(Math.min(...Array.from(v).filter((_, s) => !S.env.term[s])))} 次`, titleX: 46 });
                    },
                },
            ],
            notes: '默认 α = 0.1、ε = 0.1：大约 120 回合后贪心策略第一次能走到目标，几百回合后 J ≈ −0.17…−0.18（V* = −0.147），之后会在几条价值接近的路线之间来回换（蓝线偶尔的下凹）；橙线（实际在用的 ε-greedy 策略）一直低一截，约 −0.3——那是 10% 随机动作的代价，在会滑的地面上尤其贵。3000 回合时 Q 表的 RMS 误差约 0.27、82% 的格子贪心动作已最优。<b>探索率 ε</b> = 0.2：RMS 误差降到约 0.13（远处格子访问得多了），但 ε-greedy 本身更差（约 −0.38）。<b>TD 学习率 α</b> = 0.5：60 回合就找到目标，但随机滑动让 TD 目标噪声很大，Q 一直抖，贪心策略时好时坏（3000 回合时甚至暂时变回绕圈）。<b>ε = 0</b> 也能学：Q 初值 0 比真实价值高，没试过的动作看起来更好（乐观初值），会被依次试一遍——Step 07 专门讲。',
            warn: '常数 α 下 Q 永远在 Q* 附近抖动，理论上的收敛要求 Σα = ∞、Σα² < ∞（例如 α = 1/访问次数）。表格方法需要每个 (s, a) 都被访问很多次，状态空间一大就不现实——Step 08 用神经网络代替这张表。',
        },
        {
            id: 'cliff', title: 'SARSA vs Q-learning：悬崖行走', tag: 'TD',
            desc: '<strong>SARSA</strong>（Rummery & Niranjan 1994；名字来自更新用到的五元组 S, A, R, S′, A′）只改了 Q-learning 的一处：TD 目标里用<strong>真的会执行的下一个动作</strong> a′ 的 Q 值，而不是 max。于是它学的是 ε-greedy 策略<strong>本身</strong>的价值——<strong>在策略（on-policy）</strong>。差别在 Sutton & Barto 的<strong>悬崖行走</strong>（例 6.6）上最明显：4 × 12 的格子，每步 −1，最下面一排是悬崖，掉下去 −100 并被送回起点。Q-learning 学到的是贴着悬崖边走的最短路——这是<strong>贪心</strong>策略的最优解，可它执行时仍然有 ε 的概率随机走一步，于是时不时掉下去；SARSA 把「我自己会随机乱走」算进了价值，学会离悬崖远一点。看训练时每个回合<strong>实际</strong>拿到的奖励（在线表现），SARSA 反而更好。',
            formula: [
                'SARSA：      Q(s, a) ← Q(s, a) + α · [ r + γ·Q(s′, a′) − Q(s, a) ],   a′ ~ ε-greedy(Q, s′)，下一步就执行它',
                'Q-learning： Q(s, a) ← Q(s, a) + α · [ r + γ·max_{a′} Q(s′, a′) − Q(s, a) ]',
                'Expected SARSA：目标 = r + γ·Σ_{a′} π(a′|s′)·Q(s′, a′)   // 用期望代替一个样本，方差更小',
            ],
            code: { title: 'C++ · sarsaEpisode / cliffWorld', src: P10CPP.sarsa },
            previews: [
                {
                    id: 'cliffMap', live: true, large: true, size: [360, 330], needs: ['cliff'], label: '两种算法学到的路线（50 次独立运行中的第 1 次）', sub: '颜色 = max_a Q · 黄线 = 按 Q 贪心走的路线 · 青线 = 最近一个回合（ε-greedy）实际走的路',
                    render: (c, S) => {
                        const st = S.clSt, env = S.cenv, names = ['SARSA（在策略）', 'Q-learning（离策略）'];
                        P10D.gridPanels(c, [0, 1].map((m) => { const Q = st.runs[m][0].Q, V = Float64Array.from({ length: env.N }, (_, s) => P10.maxQ(Q, s)), gp = P10.greedyPath(env, Q, 40); return { env, label: `${names[m]} · 最近 10 回合平均 ${st.sum[m].length ? LM.mean(st.sum[m].slice(-10)).toFixed(1) : '—'}`, labelColor: m ? LM.C.red : LM.C.green, bold: true, o: { V, R: 25, path: gp, path2: st.path[m].length < 120 ? st.path[m] : null, dots: false } }; }), 1, { title: `第 ${st.ep} 回合 · α = ${S.alpha} · ε = ${S.eps}`, titleX: 46 });
                    },
                },
                {
                    id: 'cliffCurve', live: true, size: [360, 220], needs: ['cliff'], label: '在线表现：每个回合实际拿到的奖励和', sub: '50 次独立运行的平均，再对相邻 10 个回合做滑动平均 · 灰 = 两条确定性路线的奖励（贴边 −13 / 最上一行 −17）',
                    render: (c, S) => {
                        const st = S.clSt, sm = (a) => a.map((_, i) => LM.mean(a.slice(Math.max(0, i - 9), i + 1))), xs = st.sum[0].map((_, i) => i + 1);
                        PL.draw.plot(c, { x: [1, P10D.CAP.cliff], y: [-110, 12], xLabel: `回合（第 ${st.ep} / ${P10D.CAP.cliff}）`, series: [{ data: sm(st.sum[1]).map((v) => Math.max(v, -110)), x: xs, color: LM.C.red, width: 2, label: 'Q-learning' }, { data: sm(st.sum[0]).map((v) => Math.max(v, -110)), x: xs, color: LM.C.green, width: 2, label: 'SARSA' }], hmarks: [{ y: -13, color: '#4a4f57' }, { y: -17, color: '#4a4f57' }] });
                    },
                },
            ],
            notes: '默认 α = 0.1、ε = 0.1（页面上的 50 次运行，最后 100 回合）：Q-learning 的在线奖励约 −49，50 次运行学到的全是贴着悬崖边的路线，平均每回合掉下去 0.32 次；SARSA 约 −23，44 次运行走离悬崖一行的路线——和 Sutton & Barto 图 6.4 的结论一致。<b>TD 学习率 α</b> = 0.5：SARSA 退到最上面一行（50 次里 48 次），在线奖励约 −26。<b>探索率 ε</b> = 0.3：SARSA −45、Q-learning −175，差距更大；ε = 0：两者完全相同（都是纯贪心）。如果让 ε 随训练逐渐降到 0，两者都会收敛到贴边的最优路线。',
            warn: '在策略 / 离策略的区分在后面反复出现：DQN（Step 08）是离策略的，所以能用旧数据（经验回放）；REINFORCE、actor-critic、PPO（Step 09–11）是在策略的，旧数据要么扔掉，要么像 PPO 那样用重要性权重修正、并且不能走太远。第 06 篇的 RLHF 也是在策略的。',
        },
        {
            id: 'bandit', title: '探索与利用：多臂赌博机', tag: 'EXPLORE',
            desc: '只有一个状态的 MDP 叫<strong>多臂赌博机（multi-armed bandit）</strong>：10 台老虎机，每台的平均奖励 q*(a) 不同而且未知，每次拉一台、得到一个带噪声的奖励。它把 RL 的核心矛盾剥得干干净净：<strong>利用（exploit）</strong>当前看起来最好的，还是<strong>探索（explore）</strong>可能更好的？纯贪心会被开头几次运气骗住，永远守着一台次优的机器；ε-greedy 永远留 ε 的预算随机试——学得到最好的臂，但到最后仍在浪费；<strong>乐观初值</strong>把所有 Q 初始化得很高，每台机器都会「令人失望」一次，于是都被试过；<strong>UCB</strong>（Upper Confidence Bound，Auer 等 2002）给每台机器的估计加一个不确定性奖励 c·√(ln t / N(a))：拉得越少越值得一试，随时间对数增长、随次数衰减。这里是 Sutton & Barto §2.3 的 10 臂测试台：q*(a) ~ N(0, 1)，奖励 ~ N(q*(a), 1)，每次实验 1000 步，曲线是很多次独立实验的平均——它在你眼前一次次累加，越来越平滑。',
            formula: [
                'Q_{n+1}(a) = Q_n(a) + [ r − Q_n(a) ] / N(a)   // 增量式样本平均',
                'ε-greedy：以 ε 随机，否则 argmax_a Q(a)      UCB：A_t = argmax_a [ Q_t(a) + c·√( ln t / N_t(a) ) ]',
                '遗憾 regret(T) = T·max_a q*(a) − E[Σ_t r_t]   // ε 固定的 ε-greedy：线性增长；UCB：O(log T)',
            ],
            code: { title: 'C++ · gauss / banditSelect / banditRun', src: P10CPP.bandit },
            previews: [
                {
                    id: 'banditReward', live: true, large: true, size: [360, 250], needs: ['bandit'], label: '平均奖励 vs 步数', sub: '对 n 次独立实验求平均（每 5 步一个点，每次实验一组新的 q*）· <span style="color:#8a8f98">灰 = 贪心</span> · <span style="color:#6ab0e0">蓝 = ε-greedy（顶栏 ε；选 0 时用 0.1）</span> · <span style="color:#5ad07a">绿 = UCB（顶栏 c）</span> · <span style="color:#e0a060">橙 = 乐观初值 Q₁ = 5 的贪心</span>',
                    render: (c, S) => {
                        const st = S.bdSt, n = Math.max(1, st.runs), xs = [], T = P10D.BANDIT_T, series = [];
                        for (let t = 0; t < T; t += 5) xs.push(t + 3);
                        st.ms.forEach((m, i) => { const d = []; for (let t = 0; t < T; t += 5) { let s = 0; for (let k = 0; k < 5; k++) s += st.acc.rew[i][t + k]; d.push(s / 5 / n); } series.push({ data: d, x: xs, color: m.color, width: 1.8 }); });
                        PL.draw.plot(c, { x: [0, T], y: [-0.2, 1.75], xLabel: `步数 t（已平均 ${st.runs} / ${P10D.CAP.bandit} 次实验）`, series, hmarks: [{ y: 1.54, color: '#3a3f46' }] });
                        P10D.note(c, `后 100 步（灰 / 蓝 / 绿 / 橙）：${st.ms.map((m, i) => { let z = 0; for (let t = T - 100; t < T; t++) z += st.acc.rew[i][t]; return (z / 100 / n).toFixed(2); }).join(' / ')}`, { color: LM.C.gray });
                    },
                },
                {
                    id: 'banditOpt', live: true, size: [360, 200], needs: ['bandit'], label: '选中最优臂的比例', sub: '同一批实验，颜色同上',
                    render: (c, S) => {
                        const st = S.bdSt, n = Math.max(1, st.runs), xs = [], T = P10D.BANDIT_T;
                        for (let t = 0; t < T; t += 5) xs.push(t + 3);
                        PL.draw.plot(c, { x: [0, T], y: [0, 100], xLabel: '步数 t', yLabel: '%', series: st.ms.map((m, i) => { const d = []; for (let t = 0; t < T; t += 5) { let s = 0; for (let k = 0; k < 5; k++) s += st.acc.opt[i][t + k]; d.push(100 * s / 5 / n); } return { data: d, x: xs, color: m.color, width: 1.8 }; }) });
                        P10D.note(c, `后 100 步选对：${st.ms.map((m, i) => { let z = 0; for (let t = T - 100; t < T; t++) z += st.acc.opt[i][t]; return (z / n).toFixed(0) + '%'; }).join(' / ')}`, { color: LM.C.gray });
                    },
                },
                {
                    id: 'banditInside', live: true, size: [360, 220], needs: ['bTrace'], label: '一次 UCB 实验的内部（第 8 号测试台）', sub: '蓝柱 = 估计 Q_t(a) · 空心框 = 上界 Q + c·√(ln t/N) · 绿刻度 = 真实 q*(a) · 黄 = 这一步选中的臂 · 柱下 = 拉过的次数',
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0d10' }), tr = S.btr, t = S.btT, K = tr.K, top = 24, bot = 34, lo = -2.5, hi = 3.5;
                        const Y = (v) => top + (hi - PL.clamp(v, lo, hi)) / (hi - lo) * (h - top - bot), bw = (w - 20) / K;
                        ctx.fillStyle = LM.C.text; ctx.font = LM.FONT(11, true); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`t = ${t + 1}   c = ${S.ucbC}`, 46, 5);
                        ctx.strokeStyle = '#2a2f36'; ctx.beginPath(); ctx.moveTo(10, Y(0)); ctx.lineTo(w - 10, Y(0)); ctx.stroke();
                        for (let a = 0; a < K; a++) {
                            const q = tr.Qs[t * K + a], n = tr.Ns[t * K + a], x = 10 + a * bw, u = n > 0 ? q + S.ucbC * Math.sqrt(Math.log(t + 1) / n) : hi;
                            ctx.fillStyle = tr.A[t] === a ? LM.C.yellow : LM.C.blue; ctx.globalAlpha = n > 0 ? 0.9 : 0.25;
                            ctx.fillRect(x + bw * 0.2, Math.min(Y(0), Y(q)), bw * 0.6, Math.max(1, Math.abs(Y(q) - Y(0)))); ctx.globalAlpha = 1;
                            ctx.strokeStyle = '#8aa0b4'; ctx.lineWidth = 1; ctx.strokeRect(x + bw * 0.12, Y(u), bw * 0.76, Math.max(1, Y(q) - Y(u)));
                            ctx.strokeStyle = LM.C.green; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x + bw * 0.08, Y(tr.qs[a])); ctx.lineTo(x + bw * 0.92, Y(tr.qs[a])); ctx.stroke();
                            ctx.fillStyle = a === tr.best ? LM.C.green : LM.C.gray; ctx.font = LM.FONT(11, a === tr.best); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            ctx.fillText(String(Math.round(n)), x + bw / 2, h - bot + 4); ctx.fillText(a === tr.best ? '★' : String(a + 1), x + bw / 2, h - bot + 18);
                        }
                    },
                },
            ],
            notes: '默认 c = 2、ε = 0.1：几百次实验后曲线就和 Sutton & Barto 图 2.2 / 2.4 一样了。页面跑满 2000 次实验时，最后 100 步的平均奖励 / 选对最优臂的比例：贪心 1.02 / 35%（只有约 1/3 的实验找到了最好的臂），ε-greedy 1.36 / 78%，UCB 1.48 / 85%，乐观初值 1.50 / 85%（开头很差——每台都要试到失望——后面追上来）。拖 <b>UCB 系数 c</b> 到 0：UCB 退化成纯贪心；到 4：前期探索太多。下图里看 UCB 怎么选：上界（空心框顶）最高的那台被拉——要么估计高，要么拉得少。',
            warn: 'UCB 的 O(log T) 遗憾界要求奖励有界或次高斯、问题平稳。真实系统里臂会变（非平稳），ε 或 UCB 都要配合遗忘（常数 α）。在有很多状态的 MDP 里，「拉得少的臂」变成「没去过的状态」，计数没法直接用——深度 RL 里常用熵正则（Step 11 的 PPO 常带）、噪声网络或「好奇心」式的内在奖励来鼓励探索。',
        },
        {
            id: 'dqn', title: '函数逼近：DQN', tag: 'DQN',
            desc: 'Q 表有 |S|×|A| 个格子，Atari 的一帧画面就有天文数字般多的可能状态——表放不下，也学不完。办法是用一个<strong>神经网络</strong> Q_θ(s, a) 代替这张表：输入状态，一次输出所有动作的 Q 值；相似的状态共享参数，在一个格子上学到的东西会<strong>泛化</strong>到别的格子（右下）。但把 Q-learning 直接套在网络上很不稳定：连续的样本高度相关；TD 目标 r + γ·max Q_θ(s′) 本身随着 θ 在变（自己追自己的影子）；再加上离策略学习——Sutton & Barto 把「函数逼近 + 自举 + 离策略」称为<strong>致命三元组（deadly triad）</strong>。<strong>DQN</strong>（Mnih 等 2013 / 2015）的两个关键补丁：<strong>经验回放</strong>——把转移存进一个大池子，每次随机抽一个 batch 训练，打散相关性、重复利用数据（只有离策略方法能这么做）；<strong>目标网络</strong> θ⁻——TD 目标用一份定期同步的旧参数来算，目标在一段时间内固定不动。这里的网络是 16 → 32 → 32 → 4 的 MLP，输入是格子的坐标 + 行 / 列 one-hot；用 Huber 损失（对 TD 误差做截断）和 Adam。',
            formula: [
                'L(θ) = E_{(s,a,r,s′)~D} [ Huber( y − Q_θ(s, a) ) ]      y = r + γ·max_{a′} Q_{θ⁻}(s′, a′)   // D = 回放池，y 不求梯度',
                'θ⁻ ← θ   每 200 次更新      ε_t = max(ε, 1 − t/8000)   // 目标网络同步；ε 从 1 线性退火',
                'Huber(d) = d^2/2（|d| < 1），|d| − 1/2（其余）   // 梯度被截到 ±1：一次离谱的 TD 误差不会把网络带飞',
            ],
            code: { title: 'C++ · stateFeatures / QNet / ReplayBuffer / huberTDLoss / dqnStep（用 minml 自动微分）', src: P10CPP.dqn },
            previews: [
                {
                    id: 'dqnV', live: true, large: true, size: [360, 300], needs: ['dqn'], label: '网络学到的 V<sub>θ</sub>(s) = max<sub>a</sub> Q<sub>θ</sub>(s, a) 与贪心策略', sub: '色标同 Step 03 · 黄线 = 当前这个回合走过的路 · 悬停看 4 个 Q 值与 Q*',
                    render: (c, S) => {
                        const st = S.dqnSt, V = Float64Array.from({ length: S.env.N }, (_, s) => (S.env.term[s] ? 0 : P10.maxQ(st.Qt, s)));
                        S._lDqn = P10D.drawGrid(c, S.env, { V, R: S.Rv, numbers: true, act: P10.greedy(S.env, st.Qt), path: st.path.slice(-40), dots: false, title: `DQN · 第 ${st.t.toLocaleString()} 步 · ${st.ep} 回合${st.t >= P10D.CAP.dqn ? '（已停）' : ''}`, titleX: 46, sub: `ε = ${P10D.dqnEpsNow(st).toFixed(2)} · ${({ full: '回放 + 目标网络', noTarget: '无目标网络', noReplay: '无回放', neither: '都不用' })[S.dqnVar]}` });
                    },
                    probe: (x, y, S, PL, info) => { const s = P10D.hitCell(S._lDqn, S.env, info); if (s < 0 || S.env.term[s] || !S.dqnSt) return ''; const Q = S.dqnSt.Qt; return `${P10D.cellName(S.env, s)}  Q_θ: ${[0, 1, 2, 3].map((a) => P10.ARROW[a] + ' ' + Q[s * 4 + a].toFixed(3)).join('  ')}\nQ*:  ${[0, 1, 2, 3].map((a) => P10.ARROW[a] + ' ' + S.vi.Q[s * 4 + a].toFixed(3)).join('  ')}`; },
                },
                {
                    id: 'dqnCurve', live: true, size: [360, 200], needs: ['dqn'], label: '贪心策略的价值 J = V^π(S)（精确）', sub: '每 250 步评估一次网络的贪心策略',
                    render: (c, S) => {
                        const ev = S.dqnSt.ev, xs = ev.map((e) => e.t);
                        PL.draw.plot(c, { x: [0, P10D.CAP.dqn], y: P10D.jRange(S), xLabel: `环境步数（第 ${S.dqnSt.t.toLocaleString()} 步 · 每步一次梯度更新）`, series: [{ data: ev.map((e) => e.J), x: xs, color: LM.C.blue, width: 2, label: 'DQN 的贪心策略' }], hmarks: P10D.jMarks(S), marks: [{ x: S.dqnSt.t, color: '#5a5f66' }] });
                    },
                },
                {
                    id: 'dqnLoss', live: true, size: [360, 170], needs: ['dqn'], label: 'TD 损失与价值误差', sub: '对数纵轴 · 橙 = Huber TD 损失（滑动平均）· 紫 = RMS(V<sub>θ</sub> − V*)',
                    render: (c, S) => {
                        const st = S.dqnSt, le = st.lossEma;
                        PL.draw.plot(c, { x: [0, P10D.CAP.dqn], y: [1e-4, 30], logY: true, xLabel: `环境步数（第 ${st.t.toLocaleString()} 步）`, series: [{ data: le.map((p) => Math.max(1e-4, p[1])), x: le.map((p) => p[0]), color: LM.C.orange, width: 1.5, label: 'TD 损失' }, { data: st.ev.map((e) => Math.max(1e-4, e.rms)), x: st.ev.map((e) => e.t), color: LM.C.purple, width: 2, label: 'RMS(V_θ − V*)' }] });
                    },
                },
                {
                    id: 'dqnGen', live: true, size: [360, 300], needs: ['dqnGen'], label: '泛化：只在一个格子上更新，别的格子也变了', sub: '把当前网络拷一份，只对 Q(s₀, a₀) 做一步梯度上推（黄框 = s₀，<b>点击换格子</b>）· 数字 = ΔQ(s, a₀) / ΔQ(s₀, a₀)：别的格子跟着动了多少 · 表格方法里除黄框外全是 0',
                    render: (c, S) => {
                        const g = S.gen; if (!g) return;
                        const self = g.dQ[g.s0] || 1e-12, ratio = Float64Array.from(g.dQ, (v) => v / self);
                        S._lGen = P10D.drawGrid(c, S.env, { V: ratio, R: 1.5, numbers: true, hi: g.s0, title: `第 ${S.dqnSt.t.toLocaleString()} 步的网络 · a₀ = ${P10.ARROW[g.a0]}`, titleX: 46, sub: `ΔQ(s₀, a₀) = ${self.toExponential(1)}` });
                    },
                    pick: (x, y, S, PL, info) => { const s = P10D.hitCell(S._lGen, S.env, info); if (s >= 0 && !S.env.term[s]) { S.genS = s; return 'frame'; } },
                },
            ],
            notes: '默认「回放 + 目标网络」：前 8000 步 ε 从 1 降到 0.1，页面这次运行大约 8700 步时贪心策略第一次接近最优，之后 J 在 −0.18…−0.22 之间（V* = −0.147），偶尔变差又恢复——DQN 的典型不稳定。<b>DQN 变体</b> 选「无经验回放」：每次只用最近 32 条连续样本，网络被眼前这几步带着走。我们在 Node 里用与页面相同的配置、4 个种子各跑 3 万步：完整 DQN 的 J ≈ −0.21、RMS(V − V*) ≈ 0.07；无目标网络 J ≈ −0.23（前期反而更快——目标网络让价值传播慢了半拍），但价值误差一直大一倍（≈ 0.13–0.19）；无经验回放 J ≈ −0.57（4 个种子里只有 1 个学会）、RMS ≈ 0.3；两者都不用 J ≈ −0.58。右下：训练初期一步更新几乎把整张图一起抬高（输出层的偏置在起作用）；训练后出现结构——同一列、同一行的格子跟着动得多，有的甚至比 s₀ 自己动得还多。',
            warn: '在这个只有 42 个状态的玩具问题上，目标网络对<b>策略</b>几乎没有帮助（只让价值估计更准），经验回放才是关键；Mnih 等（2015）在 Atari 上的消融实验里两者都重要，去掉回放的损失更大。我们试过只给网络 (x, y) 两个坐标：一个平滑的小 MLP 表示不了墙两侧截然不同的价值，几万步也学不会，所以加了行 / 列 one-hot。输出层初始化得很小（Q ≈ 0）也很重要：用普通的 He 初始化，开头 Q 值很大很乱，学得明显更慢。右下那种「牵一发而动全身」正是 DQN 不稳定的根源之一：更新一个格子会改变别的格子的 TD 目标。',
        },
        {
            id: 'reinforce', title: '策略梯度：REINFORCE 与基线', tag: 'PG',
            desc: '另一条路线：不学价值，直接把<strong>策略</strong>参数化成 π_θ(a | s)，沿着让期望回报 J(θ) 变大的方向做梯度<strong>上升</strong>。难点是 J 依赖于「策略访问到哪些状态」，看起来没法求导；<strong>策略梯度定理</strong>（Sutton 等 2000）说梯度只需要 ∇log π：∇J = E[Σ_t γ^t · G_t · ∇log π(a_t | s_t)]。直观地说：一条轨迹回报高，就把它走过的每个动作的概率都调高一点；回报低就调低。<strong>REINFORCE</strong>（Williams 1992）就是用一个回合的样本估计这个期望。它是无偏的，但<strong>方差</strong>极大：同样一个好动作，这回合碰巧掉进陷阱，就被错怪。减去一个<strong>基线</strong> b(s) 不改变期望（因为 E[∇log π] = 0），却能大幅降低方差——最好的基线接近 V^π(s)，于是 G_t − V(s_t) 衡量的是「这次比平时好多少」。这里的策略是每个状态一行 4 个 logits 的 softmax 表（= one-hot 状态上的一层线性层）；因为模型已知，我们能<strong>精确</strong>算出 ∇J，直接看到估计有多不靠谱。',
            formula: [
                '∇_θ J(θ) = E_π[ Σ_t γ^t · ( G_t − b(s_t) ) · ∇_θ log π_θ(a_t | s_t) ]   // 对任意只依赖状态的 b(s) 都成立',
                'π_θ(a | s) = softmax(θ[s, ·])_a      ∇_{θ[s, ·]} log π_θ(a | s) = onehot(a) − π_θ(· | s)',
                '精确梯度：∂J/∂θ[s, a] = d_γ(s) · π(a|s) · A^π(s, a)      d_γ(s) = Σ_t γ^t P(s_t = s)，A^π = Q^π − V^π',
            ],
            code: { title: 'C++ · softmaxPolicy / reinforceGradient / reinforceUpdate / policyGradientExact', src: P10CPP.pg },
            previews: [
                {
                    id: 'pgCurve', live: true, large: true, size: [360, 240], needs: ['pg'], label: 'REINFORCE 的学习曲线：J(θ) = V<sup>π<sub>θ</sub></sup>(S)（精确）', sub: '每种 8 个随机种子（两种用同一组种子）：细线 = 各种子，粗线 = 平均 · 学习率 1、每回合更新一次 · 绿 = 基线 b(s) = V(s)，V 向回报 G 做蒙特卡洛回归 · 「成功」= J > −0.4',
                    render: (c, S) => {
                        const [a, b] = S.pgSt.groups;
                        PL.draw.plot(c, { x: [0, P10D.CAP.pg], y: P10D.jRange(S), xLabel: `回合（第 ${b.runs[0].episodes} / ${P10D.CAP.pg}）`, series: [...P10D.groupSeries(a, `不用基线（${P10D.groupOk(a)}/${a.runs.length} 成功）`), ...P10D.groupSeries(b, `基线 V(s)（${P10D.groupOk(b)}/${b.runs.length} 成功）`)], hmarks: P10D.jMarks(S) });
                    },
                },
                {
                    id: 'pgVar', needs: ['pgvar'], size: [360, 210], label: '单回合估计 ĝ 和真梯度 ∇J 的夹角（θ = 0，均匀随机策略）', sub: '600 个回合各给一个估计 · 横轴 = cos(ĝ, ∇J)：1 = 方向完全对，0 = 正交，负 = 反了',
                    render: (c, S) => {
                        const v = S.pgv.out, nb = 40, hA = P10D.histo(v.none.cos, -1, 1, nb), hB = P10D.histo(v.Vpi.cos, -1, 1, nb), mx = Math.max(...hA, ...hB);
                        PL.draw.plot(c, { x: [-1, 1], y: [0, mx * 1.6], xLabel: 'cos(ĝ, ∇J)', series: [{ data: hA, color: LM.C.red, bar: true, label: '不用基线' }, { data: hB, color: LM.C.green, bar: true, label: '基线 V^π(s)' }] });
                        P10D.note(c, `E‖ĝ − ∇J‖² / ‖∇J‖²：不用基线 ${v.none.relMSE.toFixed(0)} · 基线 V^π ${v.Vpi.relMSE.toFixed(1)}`, { color: LM.C.yellow });
                    },
                },
                {
                    id: 'pgAvg', needs: ['pgvar'], size: [360, 200], label: '平均 n 个回合的估计，离真梯度还有多远', sub: '‖mean_n(ĝ) − ∇J‖ / ‖∇J‖，对数坐标 · 灰虚线 = 1/√n 的斜率',
                    render: (c, S) => {
                        const v = S.pgv.out, K = S.pgv.K, xs = Array.from({ length: K }, (_, i) => i + 1), e0 = v.Vpi.err[K - 1] * Math.sqrt(K);
                        PL.draw.plot(c, { x: [1, K], y: [0.02, 200], logX: true, logY: true, xLabel: '回合数 n', series: [{ data: xs.map((n) => e0 / Math.sqrt(n)), x: xs, color: LM.C.gray, dash: [4, 4] }, { data: v.none.err, x: xs, color: LM.C.red, width: 2, label: '不用基线' }, { data: v.constJ.err, x: xs, color: LM.C.orange, width: 1.6, label: '常数基线 b = J' }, { data: v.Vpi.err, x: xs, color: LM.C.green, width: 2, label: '基线 V^π(s)' }], hmarks: [{ y: 1, color: '#4a4f57', label: '误差 = 梯度本身' }] });
                    },
                },
                {
                    id: 'pgPolicy', live: true, size: [360, 300], needs: ['pg'], label: '带基线的第 1 个种子：当前策略 π<sub>θ</sub>(a | s)', sub: '每格 4 根箭头，长度 ∝ 概率 · 颜色 = 学到的基线 V(s)',
                    render: (c, S) => { const b = S.pgSt.groups[1].runs[0]; P10D.drawGrid(c, S.env, { pi: b.pi, V: b.Vc, R: S.Rv, title: `第 ${b.episodes} 回合`, titleX: 46, sub: `J = ${P10D.sgn(b.ev[b.ev.length - 1].J)}` }); },
                },
            ],
            notes: '默认设置下，在均匀随机策略处，不用基线的单回合估计与真梯度的夹角几乎是随机的（cos 集中在 0 附近），相对均方误差约 3500；用 V^π 当基线降到约 18，<b>低了两个数量级</b>——平均 600 个回合后，不用基线的估计误差仍是梯度本身的 2 倍，用基线的只有 0.24。原因：这里的回报几乎全是负的（每步 −0.04、掉陷阱 −1），不用基线时每个走过的动作都被「压低」，真正的信号埋在噪声里。学习曲线（页面 8 个种子，4000 回合）：不用基线只有 1 个种子学会（平均 J ≈ −0.75），用基线 5 个学会（平均 −0.41）。失败的种子都停在 J ≈ −0.80：策略学会了不往陷阱走、贴着墙原地打转（−0.04/(1 − γ)），比随机乱走（−0.89）好，却再也走不到远处的目标——一个局部最优。',
            warn: '策略梯度只保证走向<b>局部</b>最优：策略一旦变得接近确定性，就不再探索，梯度也跟着消失。实践中常加<b>熵正则</b>（奖励策略保持随机），或者用方差更小的估计（下一步的 actor-critic）。公式里的 γ^t 保证估计对 J = V^π(S) 无偏（Sutton & Barto 的写法）；大多数深度 RL 实现（包括 PPO）把它省掉，梯度就不再是任何目标的精确梯度，但实践中更好用（相当于更重视后面的状态）。REINFORCE 是在策略的：每个回合用完就扔。',
        },
        {
            id: 'ac', title: 'Actor-Critic：用 TD 误差当优势', tag: 'AC',
            desc: 'REINFORCE 要等回合结束才能算 G_t，而且 G_t 带着整条轨迹的随机性。<strong>Actor-critic</strong> 把 Step 05 的时序差分请回来：一个 <strong>critic</strong> 学价值 V(s)，一个 <strong>actor</strong>（策略）用 critic 打的分来更新。关键量是<strong>优势（advantage）</strong> A^π(s, a) = Q^π(s, a) − V^π(s)：这个动作比这个状态的平均水平好多少。TD 误差 δ = r + γ·V(s′) − V(s) 只用一步就能算，而当 V = V^π 时 E[δ | s, a] = A^π(s, a)——它就是一个只看一步、方差小得多的优势估计。代价是<strong>偏差</strong>：critic 不准，δ 就不准。<strong>GAE(λ)</strong>（Schulman 等 2016）在两者之间连续插值：Â = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + …，λ = 0 是单步 TD，λ = 1 是蒙特卡洛回报减 V。下面左图是在线的单步 actor-critic（每走一步就更新 actor 和 critic），右下图在一个训练快照上实测了 GAE 的偏差—方差权衡。这正是 Step 04 广义策略迭代的「评估 + 改进」，只是两边都用采样、一小步一小步地做。',
            formula: [
                'δ_t = r_{t+1} + γ·V(s_{t+1}) − V(s_t)      V = V^π 时 E[ δ_t | s_t, a_t ] = A^π(s_t, a_t)',
                'critic：V(s_t) ← V(s_t) + α_V·δ_t      actor：θ ← θ + α·γ^t·δ_t·∇_θ log π_θ(a_t | s_t)',
                'Â_t^{GAE(λ)} = Σ_{k≥0} (γλ)^k · δ_{t+k}   // λ = 0：单步 TD（偏差大、方差小）；λ = 1：G_t − V(s_t)（无偏、方差大）',
            ],
            code: { title: 'C++ · actorCriticEpisode / gaeAdvantages', src: P10CPP.ac },
            previews: [
                {
                    id: 'acCurve', live: true, large: true, size: [360, 240], needs: ['ac'], label: '单步 actor-critic vs REINFORCE + 基线', sub: '各 8 个随机种子（细线）与平均（粗线）· 同样的学习率 1、同样的 critic / 基线学习率 0.1 · 蓝 = 回合结束才更新 · 绿 = 每走一步就更新 · 括号 = 成功（J > −0.4）的种子数',
                    render: (c, S) => {
                        const [a, b] = S.acSt.groups;
                        PL.draw.plot(c, { x: [0, P10D.CAP.ac], y: P10D.jRange(S), xLabel: `回合（第 ${b.runs[0].episodes} / ${P10D.CAP.ac}）`, series: [...P10D.groupSeries(a, `REINFORCE + 基线（${P10D.groupOk(a)}/${a.runs.length}）`), ...P10D.groupSeries(b, `单步 actor-critic（${P10D.groupOk(b)}/${b.runs.length}）`)], hmarks: P10D.jMarks(S) });
                    },
                },
                {
                    id: 'acCritic', live: true, size: [360, 180], needs: ['ac'], label: 'critic 学到的 V(s) vs 当前 actor 的真实 V<sup>π</sup>(s)', sub: 'actor-critic 的第 1 个种子 · 左 = critic · 右 = 精确策略评估 · 色标相同',
                    render: (c, S) => {
                        const r = S.acSt.groups[1].runs[0], Vt = P10.evalExact(S.env, r.pi, S.gamma), d = P10.discVisits(S.env, r.pi, S.gamma);
                        let e = 0, n = 0; for (let s = 0; s < S.env.N; s++) if (!S.env.term[s] && d[s] > 0.01) { e += (r.Vc[s] - Vt[s]) ** 2; n++; }
                        P10D.gridPanels(c, [{ env: S.env, label: `critic V（第 ${r.episodes} 回合）`, o: { V: r.Vc, R: S.Rv } }, { env: S.env, label: `真实 V^π · 常去的格子上 RMS 差 ${Math.sqrt(e / Math.max(1, n)).toFixed(3)}`, o: { V: Vt, R: S.Rv } }], 2);
                    },
                },
                {
                    id: 'gae', needs: ['gae'], size: [360, 210], label: 'GAE(λ) 的偏差—方差权衡（实测）', sub: '单步 AC 训练 200 回合后的快照：用它的 critic 算 GAE，策略梯度估计相对精确 ∇J 的偏差² 与方差（每回合，÷‖∇J‖²）',
                    render: (c, S) => {
                        const g = S.gae, xs = g.rows.map((r) => r.lam);
                        PL.draw.plot(c, { x: [0, 1], y: [0.01, 1e5], logY: true, xLabel: 'λ', series: [{ data: g.rows.map((r) => Math.max(0.01, r.var)), x: xs, color: LM.C.orange, width: 2, label: '方差' }, { data: g.rows.map((r) => Math.max(0.01, r.bias2)), x: xs, color: LM.C.purple, width: 2, label: '偏差²' }] });
                        P10D.note(c, `快照 J = ${P10D.sgn(g.J)} · critic RMS 误差 ${g.criticRMS.toFixed(2)} · 每个 λ 采 ${g.K} 回合`, { color: LM.C.gray });
                    },
                },
            ],
            notes: '页面上各 8 个种子（与 Step 09 是同一组种子）：单步 actor-critic 8 个全部学会，500 回合时平均 J ≈ −0.27，4000 回合时 ≈ −0.18；REINFORCE + 基线 500 回合时 ≈ −0.62，最后 5/8 学会（平均 −0.41）。在 Node 里用 16 个种子测过：actor-critic 在学习率 0.3、1、3 下都是 16/16。右下：从 λ = 0 到 1，方差涨了约 18 倍，偏差（来自不准的 critic）降到接近 0（λ = 1 的偏差理论上严格为 0，剩下的是有限样本误差）。batch 越大，方差被平均得越多，λ 就可以取得越大——PPO 常用 λ = 0.95。',
            warn: 'critic 和 actor 互相依赖：critic 估的是「当前」策略的价值，actor 一变它就过时了，所以两者的学习率要配合（critic 通常要快一些）。这里的 critic 是一张表；深度 RL 里它是另一个网络（或和策略共享主干的另一个输出头），第 06 篇的 RLHF 里它叫「价值模型」，和策略模型一样大。',
        },
        {
            id: 'ppo', title: 'PPO：重复使用一批数据，但别走太远', tag: 'PPO',
            desc: '在策略方法的数据很贵：每采一批回合只能用来更新一次，策略一变，这批数据就「过期」了。想多用几遍，就得修正分布的差异——<strong>重要性采样</strong>：用概率比 ρ = π_θ(a|s) / π_old(a|s) 给旧样本加权，目标 E[ρ·Â] 在 θ = θ_old 处的梯度正好是策略梯度。但 θ 离 θ_old 一远，这个替代目标就不可信了，优化器会沿着它冲到一个糟糕的策略上。TRPO（Schulman 等 2015）用 KL 约束把每步限制在「信任域」里，计算复杂；<strong>PPO</strong>（Schulman 等 2017）用一个简单的<strong>裁剪</strong>：ρ 已经超出 [1 − ε, 1 + ε]、而且还想沿着优势的方向继续往外走的样本，梯度直接置 0（右上图的平台）。于是同一批数据可以放心地做很多轮小批量梯度上升。这里每批 8 个回合，用 GAE(λ = 0.95) 算优势并标准化，做 10 轮 × 4 个小批量 = 40 步更新；对照组是「每批只走 1 步」的 A2C 和「同样 40 步但不裁剪」。<strong>第 06 篇的 RLHF 用的就是它</strong>：状态 = 问题 + 已经生成的 token，动作 = 下一个 token，奖励 = 奖励模型的分数 − β·KL；GRPO 再把 critic 去掉，用同一问题一组回答的平均奖励当基线——正是 Step 09 的基线思想。',
            formula: [
                'ρ_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)      L^PG(θ) = E[ ρ_t · Â_t ]   // θ = θ_old 处与策略梯度相同',
                'L^CLIP(θ) = E[ min( ρ_t·Â_t, clip(ρ_t, 1 − ε, 1 + ε)·Â_t ) ]   // ρ 越界且还想往外走 → 梯度为 0',
                '每批：M = 8 回合 → GAE(γ, λ = 0.95) → Â 标准化 → 10 轮 × 4 个小批量的梯度上升（critic 同时向 R̂ 回归）',
            ],
            code: { title: 'C++ · RolloutBatch / collectBatch / ppoUpdate / trainPPO', src: P10CPP.ppo },
            previews: [
                {
                    id: 'ppoCurve', live: true, large: true, size: [360, 240], needs: ['ppo'], label: '同样多的数据，三种用法', sub: '每种 4 个随机种子（三种用同一组种子）：细线 = 各种子，粗线 = 平均 · 同一个学习率（顶栏）· J 精确评估',
                    render: (c, S) => {
                        const G = S.ppoSt.groups, it = G[2].runs[0].ev.length - 1, ser = (g) => { const m = g.runs.map((r) => r.ev), n = Math.min(...m.map((e) => e.length)); const out = g.runs.map((r) => ({ data: r.ev.map((e) => e.J), x: r.ev.map((e) => e.ep), color: g.color + '40', width: 1 })); out.push({ data: Array.from({ length: n }, (_, k) => LM.mean(m.map((e) => e[k].J))), x: m[0].slice(0, n).map((e) => e.ep), color: g.color, width: 2.4, label: g.name }); return out; };
                        PL.draw.plot(c, { x: [0, P10D.CAP.ppo * P10D.PPO.M], y: P10D.jRange(S), xLabel: `回合（第 ${it} / ${P10D.CAP.ppo} 批，每批 ${P10D.PPO.M} 回合）· 学习率 ${S.ppoLR}`, series: [...ser(G[0]), ...ser(G[1]), ...ser(G[2])], hmarks: P10D.jMarks(S) });
                    },
                },
                {
                    id: 'ppoClipFn', size: [360, 180], label: '裁剪目标 L^CLIP 作为 ρ 的函数（单个样本）', sub: '实线 = min(ρÂ, clip(ρ)Â) · 虚线 = 不裁剪的 ρÂ · 平的部分梯度为 0',
                    render: (c, S) => {
                        const e = S.ppoClip > 10 ? 0 : S.ppoClip, f = (A) => (r) => Math.min(r * A, (e ? PL.clamp(r, 1 - e, 1 + e) : r) * A);
                        PL.draw.plot(c, { x: [0, 2], y: [-2, 2], xLabel: 'ρ = π_θ / π_old', series: [{ fn: f(1), color: LM.C.green, width: 2.5, label: 'Â = +1' }, { fn: f(-1), color: LM.C.red, width: 2.5, label: 'Â = −1' }, { fn: (r) => r, color: LM.C.green, dash: [4, 4], width: 1 }, { fn: (r) => -r, color: LM.C.red, dash: [4, 4], width: 1 }], marks: e ? [{ x: 1 - e, label: '1 − ε' }, { x: 1 + e, label: '1 + ε' }] : [{ x: 1, label: 'ρ = 1' }] });
                    },
                },
                {
                    id: 'ppoRatio', live: true, size: [360, 190], needs: ['ppo'], label: '最近一批样本，更新完以后的概率比 ρ', sub: '第 1 个种子 · 绿 = PPO（裁剪）· 红 = 不裁剪 · 竖线 = 1 ± ε · ρ > 3 的堆在最右边',
                    render: (c, S) => {
                        const G = S.ppoSt.groups, runs = [G[0].runs[0], G[1].runs[0], G[2].runs[0]], nb = 45, lo = 0, hi = 3, hc = P10D.histo((runs[2].ratios || []).map((r) => Math.min(r, hi - 1e-9)), lo, hi, nb), hn = P10D.histo((runs[1].ratios || []).map((r) => Math.min(r, hi - 1e-9)), lo, hi, nb);
                        const e = S.ppoClip > 10 ? 0 : S.ppoClip, l2 = runs[2].last, l1 = runs[1].last;
                        PL.draw.plot(c, { x: [lo, hi], y: [0, Math.max(1, ...hc, ...hn) * 1.6], xLabel: `ρ（第 ${runs[2].ev.length - 1} 批）`, series: [{ data: hn, color: LM.C.red, bar: true, label: '不裁剪' }, { data: hc, color: LM.C.green, bar: true, label: 'PPO' }], marks: e ? [{ x: 1 - e, color: '#8a8f98' }, { x: 1 + e, color: '#8a8f98' }] : [] });
                        P10D.note(c, `PPO：裁剪 ${l2 ? (100 * l2.clipFrac).toFixed(1) : '—'}% · KL ${l2 ? l2.kl.toFixed(3) : '—'}　不裁剪：KL ${l1 ? l1.kl.toFixed(3) : '—'}`, { color: LM.C.yellow });
                    },
                },
                {
                    id: 'ppoKL', live: true, size: [360, 160], needs: ['ppo'], label: '每批数据让策略变了多少：近似 KL(π<sub>old</sub> ‖ π<sub>new</sub>)', sub: '4 个种子的平均 · 对数纵轴 · 颜色同上 · 不裁剪时 KL 动辄 > 1：一批数据就把策略推成了（近乎）确定性的',
                    render: (c, S) => {
                        const G = S.ppoSt.groups, it = G[2].runs[0].ev.length - 1;
                        PL.draw.plot(c, { x: [0, P10D.CAP.ppo], y: [1e-4, 20], logY: true, xLabel: `第几批（第 ${it} 批）`, series: G.map((g) => { const m = g.runs.map((r) => r.ev), n = Math.min(...m.map((e) => e.length)); return { data: Array.from({ length: n - 1 }, (_, k) => PL.clamp(LM.mean(m.map((e) => e[k + 1].kl)), 1e-4, 20)), x: Array.from({ length: n - 1 }, (_, k) => k + 1), color: g.color, width: 1.6 }; }) });
                    },
                },
            ],
            notes: '默认学习率 4、ε = 0.2（页面 4 个种子）：PPO 400 回合时平均 J ≈ −0.30、2000 回合时 ≈ −0.19，4 个都学会；A2C（每批数据只走一步）400 回合时还在 −0.82，2000 回合才到 −0.27——同样多的数据，PPO 多用了 40 遍；不裁剪的版本同样走 40 步，4 个种子全部失败（某一批的 KL 冲到 10 以上，策略被推成了绕圈的确定性策略，平均停在 −0.6）。在 Node 里用 16 个种子测过 <b>PPO 学习率</b>：2 时不裁剪也大多能学（14/16），A2C 太慢；4 时 PPO 16/16、不裁剪 1/16；16 时 A2C 一步就走得够远、反而最好（16/16），PPO 也开始有种子失败（12/16），不裁剪 0/16。',
            warn: '真实的 PPO 在神经网络上用 Adam、带熵奖励和价值损失系数，常见设置是 ε = 0.1–0.2、每批 3–10 轮；它不是严格的信任域方法，实践中还常监控 KL、超了就提前停止这一批的更新。第 06 篇 RLHF 的「动作」是逐 token 的，还要同时放下策略、参考模型、奖励模型、价值模型 4 个网络。',
        },
        {
            id: 'prod', title: '教学版 → 生产版', tag: 'NEXT',
            desc: '这一页每一步都对应真实系统里的一块积木，只是规模小了很多个数量级、环境简单到能精确求解。真实世界里，状态是像素、关节角度或 token 序列，价值和策略都是大网络，模型通常未知，样本昂贵——下面是从教学版到生产版的主要变化。',
            blocks: [
                { type: 'html', html: `<table class="lm-tbl"><tr><th>主题</th><th>本页教学版</th><th>真实系统</th></tr>
                    <tr><td>值函数方法</td><td>Q 表 / 16-32-32-4 的 MLP，42 个状态</td><td>DQN（Mnih 等 2015）：84×84 灰度画面、4 帧堆叠，卷积网络输出每个动作的 Q 值，回放池 100 万条、目标网络每 1 万步同步，同一套网络和超参数玩 49 个 Atari 游戏；Rainbow（Hessel 等 2018）组合了 Double DQN、优先回放、Dueling、多步回报、分布式 RL、噪声网络六项改进</td></tr>
                    <tr><td>策略梯度</td><td>softmax 表，每批 8 回合</td><td>A3C / A2C（Mnih 等 2016）用大量并行环境采样；PPO（Schulman 等 2017）是最常用的通用算法，OpenAI Five（Dota 2）就用它在大规模并行自我对弈上训练；连续动作常用高斯策略</td></tr>
                    <tr><td>规划 + 学习</td><td>价值迭代（已知模型）</td><td>AlphaGo（Silver 等 2016）：策略网络（先模仿人类棋谱，再自我对弈强化）+ 价值网络 + 蒙特卡洛树搜索，4:1 击败李世石；AlphaGo Zero（2017）不用人类棋谱，一个网络两个头，用 MCTS 当「策略改进算子」做自我对弈（下面的 PUCT 选择规则），100:0 击败 AlphaGo；AlphaZero（2018）同一算法下国际象棋、将棋、围棋；MuZero（2020）连规则都不给，学一个模型来规划（第 14 篇）</td></tr>
                    <tr><td>语言模型</td><td>—</td><td>RLHF：PPO 优化「奖励模型分数 − β·KL」；GRPO：去掉 critic，用同一问题的一组回答做基线；可验证奖励（数学、代码）上的大规模 RL 让推理链自己变长（第 06 篇）</td></tr>
                    <tr><td>离线 RL</td><td>—</td><td>只有一个固定的数据集、不能再和环境交互（医疗、推荐、自动驾驶日志）：最大的问题是对数据里没出现过的动作过度乐观，CQL 等方法压低这些 Q 值；Decision Transformer 把 RL 写成序列建模</td></tr>
                    <tr><td>有模型 RL</td><td>动态规划假设模型已知</td><td>学一个环境模型，在「想象」里规划或训练策略：Dyna、World Models、Dreamer、MuZero——第 14 篇的主题</td></tr>
                    <tr><td>探索</td><td>ε-greedy、UCB、乐观初值</td><td>熵正则、参数噪声、内在奖励（好奇心、随机网络蒸馏）；稀疏奖励的难题（如 Montezuma’s Revenge）催生了大量这类方法</td></tr>
                    <tr><td>工程</td><td>单线程、几毫秒一帧</td><td>成千上万个并行环境、GPU 上的仿真（机器人）、仿真到现实的迁移（域随机化）、奖励设计与安全约束；结果对随机种子和实现细节很敏感，论文通常要报告多个种子</td></tr></table>` },
                {
                    type: 'previews', items: [
                        {
                            id: 'summary', live: true, size: [360, 250], label: '本页各方法此刻达到的 J = V<sup>π</sup>(S)（精确）', sub: '读的是各实时训练当前的状态（没滚动到对应步骤 = 还没开始；「⏩ 训练到底」后再看）· 后面是用掉的经验 · 策略梯度三种取多个种子的平均 · 绿竖线 = V*(S)',
                            render: (c, S) => {
                                const last = (ev) => (ev && ev.length ? ev[ev.length - 1].J : NaN), C = LM.C;
                                const rows = [
                                    ['价值迭代（已知模型）', S.vi.V[S.env.start], C.green],
                                    [S.qlSt ? `Q-learning · ${S.qlSt.ep} 回合` : 'Q-learning（未运行）', S.qlSt ? last(S.qlSt.ev) : NaN, C.blue],
                                    [S.dqnSt ? `DQN · ${S.dqnSt.t.toLocaleString()} 步` : 'DQN（未运行）', S.dqnSt ? last(S.dqnSt.ev) : NaN, C.cyan],
                                    [S.pgSt ? `REINFORCE+基线 · ${S.pgSt.groups[1].runs[0].episodes} 回合` : 'REINFORCE（未运行）', S.pgSt ? P10D.groupLast(S.pgSt.groups[1]) : NaN, C.purple],
                                    [S.acSt ? `actor-critic · ${S.acSt.groups[1].runs[0].episodes} 回合` : 'actor-critic（未运行）', S.acSt ? P10D.groupLast(S.acSt.groups[1]) : NaN, C.yellow],
                                    [S.ppoSt ? `PPO · ${S.ppoSt.groups[2].runs[0].episodes} 回合` : 'PPO（未运行）', S.ppoSt ? LM.mean(S.ppoSt.groups[2].runs.map((r) => r.ev[r.ev.length - 1].J)) : NaN, C.orange],
                                    ['均匀随机策略', S.Jrand, C.gray],
                                ];
                                const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0d10' }), padL = 168, padR = 52, top = 8, rh = (h - top - 22) / rows.length;
                                const lo = Math.min(S.Jrand, S.rstep / (1 - S.gamma), ...rows.map((r) => (r[1] === r[1] ? r[1] : Infinity))) - 0.05, hi = S.vi.V[S.env.start] + 0.03, X = (v) => padL + (v - lo) / (hi - lo) * (w - padL - padR);
                                ctx.font = LM.FONT(11); ctx.textBaseline = 'middle';
                                rows.forEach(([name, J, col], i) => {
                                    const y = top + i * rh + rh / 2;
                                    ctx.fillStyle = J === J ? C.text : C.dim; ctx.textAlign = 'right'; ctx.fillText(name, padL - 6, y);
                                    if (J !== J) return;
                                    ctx.fillStyle = col; ctx.fillRect(X(lo), y - rh * 0.3, Math.max(1, X(J) - X(lo)), rh * 0.6);
                                    ctx.fillStyle = C.gray; ctx.textAlign = 'left'; ctx.fillText(P10D.sgn(J), X(J) + 4, y);
                                });
                                ctx.strokeStyle = C.green; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(S.vi.V[S.env.start]), top); ctx.lineTo(X(S.vi.V[S.env.start]), h - 20); ctx.stroke(); ctx.setLineDash([]);
                                ctx.fillStyle = C.dim; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(`横轴：J，从 ${lo.toFixed(2)}（左端）到 V*(S) = ${P10D.sgn(S.vi.V[S.env.start])}`, padL, h - 3);
                            },
                        },
                    ],
                },
                { type: 'code', title: 'C++ · puctSelect（AlphaZero 的 MCTS 选择规则）', src: P10CPP.puct },
                { type: 'note', html: '想继续：第 06 篇把这一页的 PPO、基线思想用在语言模型上（RLHF、GRPO）；第 14 篇学一个环境模型，在模型的「想象」里规划和训练策略。经典教材是 Sutton & Barto 的《Reinforcement Learning: An Introduction》（第二版，作者网站可免费阅读），本页的网格世界、悬崖行走、10 臂测试台都来自那里。' },
            ],
        },
    ],

    graph: {
        title: '强化学习 · 数据流', height: 1420,
        intro: '左边一列是<b>已知模型</b>的动态规划（一次算完：PARAM）；右边是<b>无模型</b>的学习循环（HISTORY：每帧推进、跨帧保存 Q 表 / 网络 / 策略参数，虚线回边 = 读上一帧的结果）。所有学习结果最后都送进精确评估节点，和 V* 比较。',
        types: {
            model: { color: '#e0a060', label: 'model · 转移模型 P、奖励 R' },
            env: { color: '#56b6c2', label: 'env · 只能交互的环境 envStep()' },
            table: { color: '#6ab0e0', label: 'table · V / Q / π 表 [N·4]' },
            sample: { color: '#c678dd', label: 'sample · 转移 / 回合 / batch' },
            net: { color: '#5ad07a', label: 'net · 网络参数 θ / 策略参数' },
            scalar: { color: '#e8c25a', label: 'scalar · 超参数 / 评估结果' },
        },
        common: { title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用的网格世界', open: false, blocks: LM.CPP.blocks(['autograd', 'nn']).concat([{ label: '━ 本页公用：rl_gridworld ━', code: P10CPP.common }]) },
        nodes: [
            { id: 'world', x: 30, y: 420, title: 'makeGridWorld', tag: 'CPU', category: 'bake', lifetime: 'param', step: 'mdp', desc: '8×6 字符地图 → P(s′,r|s,a)',
                dispatch: { kind: 'CPU', name: 'makeGridWorld', count: (S) => `${S.env.N} 个状态 × 4 个动作，每个最多 3 个结果` },
                outputs: [{ name: 'Model', format: 'P, R [N·4×≤3]', type: 'model' }, { name: 'Env', format: 'envStep()', type: 'env' }],
                detail: { preview: 'trans' } },
            { id: 'hyper', x: 30, y: 700, title: 'HyperParams', tag: 'CB', category: 'user', desc: 'γ · slip · r_step · α · ε …',
                outputs: [{ name: 'Hyper', format: 'scalars', type: 'scalar' }],
                detail: { desc: '用户控制的超参数：折扣 γ、滑动概率、每步奖励（它们改变 MDP 本身），以及各学习算法的学习率、探索率、裁剪范围。', preview: null } },
            { id: 'vi', x: 360, y: 20, title: 'valueIteration', tag: 'CPU', lifetime: 'param', step: 'vi', desc: 'V ← max_a Σ P·[r + γV]',
                dispatch: { kind: 'CPU', name: 'valueIteration', count: (S) => `${S.vi.iters} 轮 × ${S.env.N} 状态 × 4 动作` },
                inputs: [{ name: 'Model', format: 'P, R', type: 'model' }, { name: 'Hyper', format: 'γ', type: 'scalar' }],
                outputs: [{ name: 'Vstar', format: 'V*, Q*, π* [N·4] f64', type: 'table' }], detail: { preview: 'viV' } },
            { id: 'pe', x: 360, y: 210, title: 'evaluatePolicy', tag: 'CPU', lifetime: 'param', step: 'eval', desc: '(I − γP_π)V = r_π',
                dispatch: { kind: 'CPU', name: 'solveLinear', count: (S) => `${S.env.N}×${S.env.N} 高斯消元` },
                inputs: [{ name: 'Model', format: 'P, R', type: 'model' }, { name: 'Hyper', format: 'γ', type: 'scalar' }],
                outputs: [{ name: 'Vpi', format: 'V^π [N] f64', type: 'table' }], detail: { preview: 'peV' } },
            { id: 'piter', x: 700, y: 210, title: 'policyIteration', tag: 'CPU', lifetime: 'param', step: 'piter', desc: '评估 → 贪心改进',
                dispatch: { kind: 'CPU', name: 'policyIteration', count: (S) => `${S.piSeq.length} 次精确评估` },
                inputs: [{ name: 'Vpi', format: 'evaluatePolicy()', type: 'table' }],
                outputs: [{ name: 'PiSeq', format: 'π₀ … π* [N]', type: 'table' }], detail: { preview: 'piSeq' } },
            { id: 'ql', x: 360, y: 400, title: 'qLearningEpisode', tag: 'CPU', lifetime: 'history', step: 'qlearn', desc: 'Q ← Q + α[r + γ max Q′ − Q]',
                dispatch: { kind: 'CPU', name: 'qLearningEpisode', count: () => `每帧 ${P10D.RATE.ql} 回合 × 训练速度` },
                inputs: [{ name: 'Env', format: 'envStep()', type: 'env' }, { name: 'Hyper', format: 'α, ε, γ', type: 'scalar' }, { name: 'Qtab', format: 'Q [N·4] f64', type: 'table', history: true }],
                outputs: [{ name: 'Qtab', format: 'Q [N·4] f64', type: 'table' }], detail: { preview: 'qlQ' } },
            { id: 'cliff', x: 360, y: 600, title: 'sarsaEpisode ×50', tag: 'CPU', lifetime: 'history', step: 'cliff', desc: 'r + γQ(s′, a′) vs max',
                dispatch: { kind: 'CPU', name: 'sarsaEpisode / qLearningEpisode', count: () => `${P10D.CLIFF_RUNS} 次独立运行 × 2 种算法` },
                inputs: [{ name: 'Hyper', format: 'α, ε', type: 'scalar' }, { name: 'Qcliff', format: 'Q [36·4] ×100', type: 'table', history: true }],
                outputs: [{ name: 'Qcliff', format: 'Q [36·4] ×100', type: 'table' }], detail: { preview: 'cliffCurve' } },
            { id: 'bandit', x: 360, y: 790, title: 'banditRun', tag: 'CPU', lifetime: 'history', step: 'bandit', desc: '10 臂 × 1000 步 × 4 种方法',
                dispatch: { kind: 'CPU', name: 'banditRun', count: () => `每帧 ${P10D.RATE.bandit} 次独立实验` },
                inputs: [{ name: 'Hyper', format: 'ε, c', type: 'scalar' }, { name: 'Curves', format: 'Σ r, Σ 最优 [4×1000]', type: 'sample', history: true }],
                outputs: [{ name: 'Curves', format: 'Σ r, Σ 最优 [4×1000]', type: 'sample' }], detail: { preview: 'banditReward' } },
            { id: 'dqnAct', x: 360, y: 980, title: 'dqnStep · act + store', tag: 'CPU', lifetime: 'history', step: 'dqn', desc: 'ε-greedy(Q_θ) → replayPush',
                dispatch: { kind: 'CPU', name: 'dqnStep', count: () => `每帧 ${P10D.RATE.dqn} 步 × 训练速度` },
                inputs: [{ name: 'Env', format: 'envStep()', type: 'env' }, { name: 'QNet', format: 'θ: 16·32·32·4', type: 'net', history: true }],
                outputs: [{ name: 'Replay', format: '≤ 5000 × (s, a, r, s′, done)', type: 'sample' }], detail: { preview: 'dqnV' } },
            { id: 'dqnTrain', x: 700, y: 980, title: 'huberTDLoss + Adam', tag: 'CPU', lifetime: 'history', step: 'dqn', desc: 'y = r + γ max Q_θ⁻(s′)',
                dispatch: { kind: 'CPU', name: 'backward + adamwStep', count: 'batch 32，每个环境步一次' },
                inputs: [{ name: 'Replay', format: 'batch 32', type: 'sample' }, { name: 'QNet', format: 'θ', type: 'net', history: true }, { name: 'Target', format: 'θ⁻', type: 'net', history: true }],
                outputs: [{ name: 'QNet', format: 'θ: 16·32·32·4', type: 'net' }], detail: { preview: 'dqnLoss' } },
            { id: 'dqnSync', x: 1030, y: 1000, title: 'copyQNet', tag: 'COPY', lifetime: 'history', step: 'dqn', desc: 'θ⁻ ← θ 每 200 次更新',
                inputs: [{ name: 'QNet', format: 'θ', type: 'net' }], outputs: [{ name: 'Target', format: 'θ⁻', type: 'net' }], detail: { preview: null } },
            { id: 'rollout', x: 360, y: 1190, title: 'collectBatch', tag: 'CPU', lifetime: 'history', step: 'ppo', desc: 'M 回合 → GAE(λ) → Â 标准化',
                dispatch: { kind: 'CPU', name: 'sampleEpisode + gaeAdvantages', count: () => `${P10D.PPO.M} 回合 / 批` },
                inputs: [{ name: 'Env', format: 'envStep()', type: 'env' }, { name: 'Theta', format: 'θ [N·4] f64', type: 'net', history: true }, { name: 'Critic', format: 'V [N] f64', type: 'table', history: true }],
                outputs: [{ name: 'Batch', format: '(s, a, Â, R̂, π_old) × n', type: 'sample' }], detail: { preview: 'ppoRatio' } },
            { id: 'ppo', x: 700, y: 1190, title: 'ppoUpdate', tag: 'CPU', lifetime: 'history', step: 'ppo', desc: 'min(ρÂ, clip(ρ)Â) · 10 轮 × 4 批',
                dispatch: { kind: 'CPU', name: 'ppoUpdate', count: '40 步梯度上升 / 批' },
                inputs: [{ name: 'Batch', format: 'n 个样本', type: 'sample' }, { name: 'Hyper', format: 'lr, ε_clip', type: 'scalar' }],
                outputs: [{ name: 'Theta', format: 'θ [N·4] f64', type: 'net' }, { name: 'Critic', format: 'V [N] f64', type: 'table' }], detail: { preview: 'ppoCurve' } },
            { id: 'evalJ', x: 1030, y: 420, title: 'evaluatePolicy · J', tag: 'CPU', category: 'output', lifetime: 'history', step: 'prod', desc: 'J = V^π(S) vs V*(S)',
                dispatch: { kind: 'CPU', name: 'evaluatePolicy', count: '每 20 回合 / 250 步一次' },
                inputs: [{ name: 'Model', format: 'P, R', type: 'model' }, { name: 'Vstar', format: 'V*', type: 'table' }, { name: 'Qtab', format: 'greedy(Q)', type: 'table' }, { name: 'QNet', format: 'greedy(Q_θ)', type: 'net' }, { name: 'Theta', format: 'softmax(θ)', type: 'net' }],
                outputs: [{ name: 'J', format: 'scalar', type: 'scalar' }], detail: { preview: 'summary' } },
        ],
    },

    references: [
        { title: 'Reinforcement Learning: An Introduction（第二版）', author: 'R. S. Sutton, A. G. Barto', venue: 'MIT Press', year: 2018, url: 'http://incompleteideas.net/book/the-book-2nd.html', note: '本页的主要参考：MDP、动态规划、TD、悬崖行走（例 6.6）、10 臂测试台（§2.3）、策略梯度与 actor-critic（第 13 章）' },
        { title: 'Artificial Intelligence: A Modern Approach（马尔可夫决策过程一章）', author: 'S. Russell, P. Norvig', venue: 'Pearson', note: '会滑的网格世界与「每步奖励改变最优策略」的经典例子（4×3 世界）' },
        { title: 'Dynamic Programming', author: 'R. Bellman', venue: 'Princeton University Press', year: 1957, note: '贝尔曼方程与动态规划' },
        { title: 'Dynamic Programming and Markov Processes', author: 'R. A. Howard', venue: 'MIT Press', year: 1960, note: '策略迭代' },
        { title: 'Q-learning', author: 'C. J. C. H. Watkins, P. Dayan', venue: 'Machine Learning 8', year: 1992, note: 'Q-learning 及其收敛性证明（算法本身出自 Watkins 1989 年的博士论文）' },
        { title: 'On-line Q-learning using connectionist systems', author: 'G. A. Rummery, M. Niranjan', venue: 'Cambridge University Engineering Department, Technical Report', year: 1994, note: 'SARSA（当时叫 modified connectionist Q-learning）' },
        { title: 'Finite-time Analysis of the Multiarmed Bandit Problem', author: 'P. Auer, N. Cesa-Bianchi, P. Fischer', venue: 'Machine Learning 47', year: 2002, note: 'UCB1 及其对数遗憾界' },
        { title: 'Playing Atari with Deep Reinforcement Learning', author: 'V. Mnih et al.', year: 2013, url: 'https://arxiv.org/abs/1312.5602', note: 'DQN 的第一篇（经验回放 + 卷积网络）' },
        { title: 'Human-level control through deep reinforcement learning', author: 'V. Mnih et al.', venue: 'Nature 518', year: 2015, note: 'DQN（加入目标网络），49 个 Atari 游戏' },
        { title: 'Rainbow: Combining Improvements in Deep Reinforcement Learning', author: 'M. Hessel et al.', venue: 'AAAI', year: 2018, url: 'https://arxiv.org/abs/1710.02298' },
        { title: 'Simple statistical gradient-following algorithms for connectionist reinforcement learning', author: 'R. J. Williams', venue: 'Machine Learning 8', year: 1992, note: 'REINFORCE' },
        { title: 'Policy Gradient Methods for Reinforcement Learning with Function Approximation', author: 'R. S. Sutton, D. McAllester, S. Singh, Y. Mansour', venue: 'NeurIPS (NIPS 12)', year: 2000, note: '策略梯度定理' },
        { title: 'Asynchronous Methods for Deep Reinforcement Learning', author: 'V. Mnih et al.', venue: 'ICML', year: 2016, url: 'https://arxiv.org/abs/1602.01783', note: 'A3C / A2C' },
        { title: 'High-Dimensional Continuous Control Using Generalized Advantage Estimation', author: 'J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel', venue: 'ICLR', year: 2016, url: 'https://arxiv.org/abs/1506.02438', note: 'GAE(λ)' },
        { title: 'Trust Region Policy Optimization', author: 'J. Schulman, S. Levine, P. Moritz, M. Jordan, P. Abbeel', venue: 'ICML', year: 2015, url: 'https://arxiv.org/abs/1502.05477' },
        { title: 'Proximal Policy Optimization Algorithms', author: 'J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov', year: 2017, url: 'https://arxiv.org/abs/1707.06347', note: 'PPO 的裁剪目标' },
        { title: 'Mastering the game of Go with deep neural networks and tree search', author: 'D. Silver et al.', venue: 'Nature 529', year: 2016, note: 'AlphaGo' },
        { title: 'Mastering the game of Go without human knowledge', author: 'D. Silver et al.', venue: 'Nature 550', year: 2017, note: 'AlphaGo Zero' },
        { title: 'A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play', author: 'D. Silver et al.', venue: 'Science 362', year: 2018, note: 'AlphaZero（PUCT 选择规则）' },
        { title: 'Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems', author: 'S. Levine, A. Kumar, G. Tucker, J. Fu', year: 2020, url: 'https://arxiv.org/abs/2005.01643' },
    ],
});

