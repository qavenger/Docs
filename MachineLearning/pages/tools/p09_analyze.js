// p09_analyze.js — 在 p09_train_all.js 存下的模型上做第 09 篇需要的分析（与页面同一份 P09 代码）：
//   interp：AE / VAE 潜空间直线插值的「均匀度」（逐步像素变化的 max/mean，随机 300 对）
//   prior ：AE（拟合高斯）vs VAE（N(0, I)）采样点到最近真实编码的距离
//   tokens：VQ token 先验（计数模型：独立 / 左邻 / 左 + 上邻）的采样，写 PNG
//   grad  ：重参数化（pathwise）vs score function（REINFORCE）梯度估计的方差
const fs = require('fs'), path = require('path');
const { P09, LM, PNG } = require('./p09_proto.js');
const { AG } = LM;
const seed = 20251123, data = P09.makeData(LM.rng(seed)); P09.initEvalEps(seed);
const OUT = path.join(__dirname, 'shots09', 'proto');
const M = JSON.parse(fs.readFileSync(path.join(OUT, 'models.json'), 'utf8'));
const ae = P09.makeAE('ae', 1), vae = P09.makeAE('vae', 1), vq = P09.makeVQ(M.vq.o.K, 1, { C: 16 });
[[ae, M.ae], [vae, M.vae], [vq, M.vq]].forEach(([m, s]) => m.P.forEach((p, i) => p.data.set(s.P[i])));
const what = process.argv[2] || 'all';
const N = P09.NTEST, te = data.te;
const enc2 = (m, i0, n) => AG.noGrad(() => { const X = AG.tensor(n, 256, te.X.subarray(i0 * 256, (i0 + n) * 256)); return P09.encode(m, X).mu.data.slice(); });
const dec = (m, Z, n) => AG.noGrad(() => P09.sigmoidInto(P09.decode(m, AG.tensor(n, 2, Z))));
const zAE = enc2(ae, 0, N), zVAE = enc2(vae, 0, N);

if (what === 'all' || what === 'interp') {
    const r = LM.rng(5), T = 10; let sAE = 0, sVAE = 0, sPix = 0, cnt = 0, jumpsAE = 0, jumpsVAE = 0;
    for (let q = 0; q < 300; q++) {
        const a = Math.floor(r() * N); let b = Math.floor(r() * N); if (te.y[a] === te.y[b]) continue;
        const stat = (z, m) => { const Z = new Float32Array(2 * (T + 1)); for (let t = 0; t <= T; t++) for (let d = 0; d < 2; d++) Z[2 * t + d] = (1 - t / T) * z[2 * a + d] + t / T * z[2 * b + d]; const X = dec(m, Z, T + 1); const st = []; for (let t = 0; t < T; t++) { let s = 0; for (let j = 0; j < 256; j++) s += (X[(t + 1) * 256 + j] - X[t * 256 + j]) ** 2; st.push(Math.sqrt(s)); } return Math.max(...st) / LM.mean(st); };
        const ra = stat(zAE, ae), rv = stat(zVAE, vae); sAE += ra; sVAE += rv; cnt++; if (ra > 2.5) jumpsAE++; if (rv > 2.5) jumpsVAE++;
    }
    console.log(`interp max/mean step change over ${cnt} pairs: AE ${(sAE / cnt).toFixed(2)} (pairs with a jump >2.5×: ${(100 * jumpsAE / cnt).toFixed(0)}%) · VAE ${(sVAE / cnt).toFixed(2)} (${(100 * jumpsVAE / cnt).toFixed(0)}%)  · uniform = 1.0`);
    // 一对示例写 PNG：像素交叉淡化 / AE / VAE
    const a = te.n > 0 ? data.byClass[4][0] : 0, b = data.byClass[7][0], imgs = [];
    for (let t = 0; t <= T; t++) { const x = new Float32Array(256); for (let j = 0; j < 256; j++) x[j] = (1 - t / T) * te.X[a * 256 + j] + t / T * te.X[b * 256 + j]; imgs.push(x); }
    for (const [z, m] of [[zAE, ae], [zVAE, vae]]) { const Z = new Float32Array(2 * (T + 1)); for (let t = 0; t <= T; t++) for (let d = 0; d < 2; d++) Z[2 * t + d] = (1 - t / T) * z[2 * a + d] + t / T * z[2 * b + d]; const X = dec(m, Z, T + 1); for (let t = 0; t <= T; t++) imgs.push(X.subarray(t * 256, (t + 1) * 256)); }
    PNG.writeGrid(path.join(OUT, 'interp.png'), imgs, 16, 16, T + 1, 4);
}
if (what === 'all' || what === 'prior') {
    // AE：对测试编码拟合 2 维高斯；VAE：标准正态先验。各采 500 个点，看离最近的真实编码有多远（单位 = 真实编码最近邻距离的中位数）
    const nn = (z, Zs, n, skip) => { let best = Infinity; for (let j = 0; j < n; j++) { if (j === skip) continue; const d = (z[0] - Zs[2 * j]) ** 2 + (z[1] - Zs[2 * j + 1]) ** 2; if (d < best) best = d; } return Math.sqrt(best); };
    const med = (a) => { const s = a.slice().sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
    for (const [name, Zs, sampler] of [['AE + 拟合高斯', zAE, null], ['VAE + N(0,I)', zVAE, 'prior']]) {
        const base = med(Array.from({ length: N }, (_, i) => nn([Zs[2 * i], Zs[2 * i + 1]], Zs, N, i)));
        let mx = 0, my = 0; for (let i = 0; i < N; i++) { mx += Zs[2 * i] / N; my += Zs[2 * i + 1] / N; }
        let cxx = 0, cxy = 0, cyy = 0; for (let i = 0; i < N; i++) { const dx = Zs[2 * i] - mx, dy = Zs[2 * i + 1] - my; cxx += dx * dx / N; cxy += dx * dy / N; cyy += dy * dy / N; }
        const l11 = Math.sqrt(cxx), l21 = cxy / l11, l22 = Math.sqrt(Math.max(1e-12, cyy - l21 * l21)), r = LM.rng(9), ds = [];
        const Zsamp = new Float32Array(2 * 40);
        for (let s = 0; s < 500; s++) { const e1 = LM.randn(r), e2 = LM.randn(r); const z = sampler === 'prior' ? [e1, e2] : [mx + l11 * e1, my + l21 * e1 + l22 * e2]; ds.push(nn(z, Zs, N, -1) / base); if (s < 40) { Zsamp[2 * s] = z[0]; Zsamp[2 * s + 1] = z[1]; } }
        const far = ds.filter((d) => d > 3).length / ds.length;
        console.log(`${name}: sample→nearest code distance / median code spacing: median ${med(ds).toFixed(2)}, mean ${LM.mean(ds).toFixed(2)}, >3× : ${(100 * far).toFixed(1)}%   (fitted mean ${mx.toFixed(2)},${my.toFixed(2)} std ${Math.sqrt(cxx).toFixed(2)},${Math.sqrt(cyy).toFixed(2)})`);
        const X = dec(name.startsWith('AE') ? ae : vae, Zsamp, 40); PNG.writeGrid(path.join(OUT, `samples_${name.startsWith('AE') ? 'ae' : 'vae'}.png`), Array.from({ length: 40 }, (_, k) => X.subarray(k * 256, (k + 1) * 256)), 16, 16, 10, 4);
    }
}
if (what === 'all' || what === 'tokens') {
    const K = vq.K, NT = data.tr.n, tok = new Int16Array(NT * 16);
    AG.noGrad(() => { for (let i0 = 0; i0 < NT; i0 += 200) { const n = Math.min(200, NT - i0); const o = P09.vqForward(vq, AG.tensor(n * 16, 64, P09.patchesOf(data.tr.X.subarray(i0 * 256, (i0 + n) * 256), n)), n); tok.set(o.q.idx, i0 * 16); } });
    const used = new Set(tok); console.log(`VQ K=${K}: train tokens use ${used.size} codes`);
    // 计数模型
    const uni = new Float64Array(16 * K), left = new Map(), lu = new Map();
    const key = (...a) => a.join(',');
    const inc = (m, k, z) => { let e = m.get(k); if (!e) { e = new Map(); m.set(k, e); } e.set(z, (e.get(z) || 0) + 1); };
    for (let i = 0; i < NT; i++) for (let t = 0; t < 16; t++) { const z = tok[i * 16 + t], l = t % 4 ? tok[i * 16 + t - 1] : -1, u = t >= 4 ? tok[i * 16 + t - 4] : -1; uni[t * K + z]++; inc(left, key(t, l), z); inc(lu, key(t, l, u), z); }
    const sampleFrom = (counts, r) => { let tot = 0; counts.forEach((c) => { tot += c; }); let u = r() * tot; for (const [z, c] of counts) { u -= c; if (u <= 0) return z; } return counts.keys().next().value; };
    const r = LM.rng(11), modes = ['independent', 'left', 'left+up'], imgs = [];
    // 训练集 token 的对数似然（每张图的 nats）：看三种先验各解释了多少结构
    for (const mode of modes) {
        const grids = new Int32Array(30 * 16);
        for (let s = 0; s < 30; s++) for (let t = 0; t < 16; t++) {
            const l = t % 4 ? grids[s * 16 + t - 1] : -1, u = t >= 4 ? grids[s * 16 + t - 4] : -1;
            let c = null;
            if (mode === 'left+up') c = lu.get(key(t, l, u));
            if ((!c || mode === 'left') && mode !== 'independent') c = left.get(key(t, l));
            if (!c) { c = new Map(); for (let z = 0; z < K; z++) if (uni[t * K + z]) c.set(z, uni[t * K + z]); }
            grids[s * 16 + t] = sampleFrom(c, r);
        }
        const X = AG.noGrad(() => P09.sigmoidInto(P09.vqDecodeIdx(vq, grids)));
        for (let s = 0; s < 30; s++) imgs.push(X.subarray(s * 256, (s + 1) * 256));
        // 训练集每张图的平均 −log p（无平滑：只对训练集自己评估，作为拟合程度的参考）
        let nll = 0;
        for (let i = 0; i < NT; i++) for (let t = 0; t < 16; t++) {
            const z = tok[i * 16 + t], l = t % 4 ? tok[i * 16 + t - 1] : -1, u = t >= 4 ? tok[i * 16 + t - 4] : -1;
            const c = mode === 'independent' ? null : mode === 'left' ? left.get(key(t, l)) : lu.get(key(t, l, u));
            if (!c) { nll -= Math.log(uni[t * K + z] / NT); continue; }
            let tot = 0; c.forEach((v) => { tot += v; }); nll -= Math.log(c.get(z) / tot);
        }
        console.log(`prior ${mode.padEnd(12)} train NLL ${(nll / NT).toFixed(2)} nats/img (${(nll / NT / Math.LN2).toFixed(1)} bits) · contexts ${mode === 'left' ? left.size : mode === 'left+up' ? lu.size : 16}`);
    }
    PNG.writeGrid(path.join(OUT, 'vq_prior_samples.png'), imgs, 16, 16, 15, 4);
    // 码字画廊：16 个位置全是码字 k
    const gal = new Int32Array(K * 16); for (let k = 0; k < K; k++) gal.fill(k, k * 16, (k + 1) * 16);
    const Xg = AG.noGrad(() => P09.sigmoidInto(P09.vqDecodeIdx(vq, gal)));
    PNG.writeGrid(path.join(OUT, 'vq_codes_uniform.png'), Array.from({ length: K }, (_, k) => Xg.subarray(k * 256, (k + 1) * 256)), 16, 16, 16, 3);
    const counts = new Int32Array(K); for (const z of tok) counts[z]++;
    const bg = LM.argmax(counts); console.log('most common code', bg, 'count', counts[bg], 'of', tok.length);
    const gal2 = new Int32Array(K * 16).fill(bg); for (let k = 0; k < K; k++) gal2[k * 16 + 5] = k;
    const Xg2 = AG.noGrad(() => P09.sigmoidInto(P09.vqDecodeIdx(vq, gal2)));
    PNG.writeGrid(path.join(OUT, 'vq_codes_single.png'), Array.from({ length: K }, (_, k) => Xg2.subarray(k * 256, (k + 1) * 256)), 16, 16, 16, 3);
}
if (what === 'all' || what === 'grad') {
    // 对测试图 i：f(z) = log p(x|z)（BCE 的负值），q = N(μ, σ²)。∂/∂μ₁ E_q[f] 的单样本估计：
    //   pathwise：∂f/∂z₁ 在 z = μ + σε 处（autograd 穿过解码器）；score：f(z)·(z₁ − μ₁)/σ₁²
    for (const i of [data.byClass[3][0], data.byClass[7][0], data.byClass[0][1]]) {
        const X = AG.tensor(1, 256, te.X.subarray(i * 256, (i + 1) * 256));
        const e = AG.noGrad(() => P09.encode(vae, X)), mu = Array.from(e.mu.data), sd = Array.from(e.logv.data, (v) => Math.exp(0.5 * v));
        const r = LM.rng(3), S = 2000, pw = [], sf = [], sfb = [];
        let fbar = 0; const fs_ = [];
        for (let s = 0; s < S; s++) {
            const eps = [LM.randn(r), LM.randn(r)], Z = AG.param('z', 1, 2); Z.data[0] = mu[0] + sd[0] * eps[0]; Z.data[1] = mu[1] + sd[1] * eps[1];
            AG.clearTape(); const L = P09.bce(P09.decode(vae, Z), X); AG.backward(L);
            const f = -L.item(); pw.push(-Z.grad[0]); fs_.push([f, eps[0]]); fbar += f / S;
        }
        fs_.forEach(([f, e0]) => { sf.push(f * e0 / sd[0]); sfb.push((f - fbar) * e0 / sd[0]); });
        const sdv = (a) => { const m = LM.mean(a); return Math.sqrt(LM.mean(a.map((v) => (v - m) ** 2))); };
        console.log(`img #${i} (y=${te.y[i]}) σ = ${sd.map((v) => v.toFixed(3))}: pathwise mean ${LM.mean(pw).toFixed(2)} std ${sdv(pw).toFixed(2)} · score mean ${LM.mean(sf).toFixed(2)} std ${sdv(sf).toFixed(1)} · score−baseline mean ${LM.mean(sfb).toFixed(2)} std ${sdv(sfb).toFixed(1)} · var ratio ${(sdv(sf) ** 2 / sdv(pw) ** 2).toFixed(0)}× / ${(sdv(sfb) ** 2 / sdv(pw) ** 2).toFixed(0)}×`);
    }
}
