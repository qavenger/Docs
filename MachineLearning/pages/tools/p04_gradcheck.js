// p04_gradcheck.js — 第 04 篇自定义可微算子的有限差分梯度校验（与 tools/test_engine2.js 同一个方法）。用法：node pages/tools/p04_gradcheck.js
// 直接加载 04_latent_vae.lab.js 里 PipelineLab.mount( 之前的部分（与页面同一份代码）。
// 单个算子：逐元素中心差分（小张量，float32 分辨率够用）；整个网络：随机方向的方向导数（信号是全部参数梯度之和，不受 float32 分辨率限制）。
const fs = require('fs'), path = require('path');
const LM = require('../llm_common.js');
const { AG } = LM;
const src = fs.readFileSync(path.join(__dirname, '..', '04_latent_vae.lab.js'), 'utf8');
const P09 = new Function('LM', src.slice(0, src.indexOf('PipelineLab.mount(')) + '\nreturn P09;')(LM);
module.exports = { P09, LM };
if (require.main !== module) return;

let fails = 0;
function numGradCheck(name, build, inputs, eps = 1e-2, tol = 0.02) {
    AG.clearTape();
    inputs.forEach((t) => { t.grad = null; });
    AG.backward(build());
    let worst = 0;
    inputs.forEach((t, ti) => {
        const ana = Float64Array.from(t.grad || new Float32Array(t.size));
        const nCheck = Math.min(t.size, 60);
        for (let i = 0; i < nCheck; i++) {
            const k = Math.floor((i * 7919) % t.size), old = t.data[k];
            t.data[k] = old + eps; const lp = AG.noGrad(build).item();
            t.data[k] = old - eps; const lm = AG.noGrad(build).item();
            t.data[k] = old;
            const num = (lp - lm) / (2 * eps), a = ana[k];
            const rel = Math.abs(num - a) / Math.max(1e-3, Math.abs(num) + Math.abs(a));
            if (rel > worst) worst = rel;
            if (rel > tol) console.log(`  ${name} input#${ti}[${k}] analytic=${a.toExponential(3)} numeric=${num.toExponential(3)} rel=${rel.toFixed(3)}`);
        }
    });
    const ok = worst < tol; if (!ok) fails++;
    console.log(`${ok ? 'OK  ' : 'FAIL'} ${name.padEnd(36)} worst rel err ${worst.toExponential(2)}`);
}
function dirCheck(name, build, inputs, eps = 1e-2, tol = 0.005, nDir = 4) {
    AG.clearTape(); inputs.forEach((t) => { t.grad = null; });
    AG.backward(build());
    const g = inputs.map((t) => Float64Array.from(t.grad || new Float32Array(t.size))), r = LM.rng(77);
    let worst = 0;
    for (let d = 0; d < nDir; d++) {
        const v = inputs.map((t) => Float64Array.from({ length: t.size }, () => LM.randn(r)));
        let ana = 0; inputs.forEach((t, k) => { for (let i = 0; i < t.size; i++) ana += g[k][i] * v[k][i]; });
        const old = inputs.map((t) => Float32Array.from(t.data));
        const at = (s) => { inputs.forEach((t, k) => { for (let i = 0; i < t.size; i++) t.data[i] = old[k][i] + s * v[k][i]; }); return AG.noGrad(build).item(); };
        const num = (at(eps) - at(-eps)) / (2 * eps);
        inputs.forEach((t, k) => t.data.set(old[k]));
        let gn = 0, vn = 0; inputs.forEach((t, k) => { for (let i = 0; i < t.size; i++) { gn += g[k][i] ** 2; vn += v[k][i] ** 2; } });
        const rel = Math.abs(num - ana) / Math.max(1e-9, Math.sqrt(gn * vn) / Math.sqrt(inputs.reduce((a, t) => a + t.size, 0))); worst = Math.max(worst, rel);   // 相对于 ‖g‖·‖v‖/√n（随机方向导数的典型大小）
        if (rel > tol) console.log(`  ${name} dir#${d} analytic=${ana.toExponential(4)} numeric=${num.toExponential(4)} rel=${rel.toFixed(4)}`);
    }
    const ok = worst < tol; if (!ok) fails++;
    console.log(`${ok ? 'OK  ' : 'FAIL'} ${name.padEnd(36)} worst rel err ${worst.toExponential(2)}（${nDir} 个随机方向）`);
}
const rand = LM.rng(42);
const P = (r, c, s = 1, off = 0) => AG.param('p', r, c, (a) => { for (let i = 0; i < a.length; i++) a[i] = off + s * LM.randn(rand); });
const proj = (T) => { const W = AG.tensor(T.r, T.c); const rr = LM.rng(9); for (let i = 0; i < W.size; i++) W.data[i] = rr() - 0.5; return AG.sum(AG.mul(T, W)); };

// 1. BCE with logits（目标是 0..1 的灰度）
{ const Z = P(3, 24, 2), X = AG.tensor(3, 24); for (let i = 0; i < X.size; i++) X.data[i] = rand() < 0.6 ? 0 : rand();
  numGradCheck('bce(logits, x)', () => P09.bce(Z, X), [Z]);
  const b = AG.noGrad(() => P09.bce(Z, X)).item(); let ref = 0; for (let i = 0; i < Z.size; i++) { const s = 1 / (1 + Math.exp(-Z.data[i])); ref -= X.data[i] * Math.log(s) + (1 - X.data[i]) * Math.log(1 - s); }
  console.log(`     bce value ${b.toFixed(5)} vs −Σ[x log σ + (1−x) log(1−σ)] = ${(ref / 3).toFixed(5)}`); }
// 2. KL(N(μ, σ²) ‖ N(0, 1)) 闭式解：梯度 + 与蒙特卡洛估计对照
{ const mu = P(4, 2), lv = P(4, 2, 0.7);
  numGradCheck('klNormal(mu, logvar)', () => P09.klNormal(mu, lv), [mu, lv]);
  const kl = AG.noGrad(() => P09.klNormal(mu, lv)).item(), r = LM.rng(5); let mc = 0; const N = 400000;
  for (let s = 0; s < N; s++) { const i = s % 4; for (let d = 0; d < 2; d++) { const m = mu.data[i * 2 + d], sd = Math.exp(0.5 * lv.data[i * 2 + d]), z = m + sd * LM.randn(r); mc += (-0.5 * ((z - m) / sd) ** 2 - Math.log(sd)) - (-0.5 * z * z); } }
  console.log(`     KL closed form ${kl.toFixed(4)} vs Monte Carlo ${(mc / (N / 4) / 4).toFixed(4)}`); }
// 3. 重参数化 z = μ + exp(½ logσ²)·ε 通过下游函数
{ const mu = P(3, 2), lv = P(3, 2, 0.5), eps = AG.tensor(3, 2); for (let i = 0; i < 6; i++) eps.data[i] = LM.randn(rand);
  numGradCheck('reparam → tanh → proj', () => proj(AG.tanh(P09.reparam(mu, lv, eps))), [mu, lv]); }
// 4. grid3x3 im2col 与 fold（col2im）
{ const Z = P(2 * 16, 3); numGradCheck('grid3x3 (4×4 grid, C=3)', () => proj(P09.grid3x3(Z)), [Z]); }
{ const S = P(2 * 16, 64), b = P(1, 1); numGradCheck('fold (8×8 stamps, s4 p2) + bias', () => proj(P09.fold(S, b)), [S, b]); }
{ const S = P(16, 64), b = AG.tensor(1, 1); const Y = AG.noGrad(() => P09.fold(S, b));
  const ref = new Float32Array(256); for (let p = 0; p < 16; p++) { const iy = p >> 2, ix = p & 3; for (let ky = 0; ky < 8; ky++) for (let kx = 0; kx < 8; kx++) { const oy = 4 * iy - 2 + ky, ox = 4 * ix - 2 + kx; if (oy >= 0 && oy < 16 && ox >= 0 && ox < 16) ref[oy * 16 + ox] += S.data[p * 64 + ky * 8 + kx]; } }
  let md = 0; for (let i = 0; i < 256; i++) md = Math.max(md, Math.abs(ref[i] - Y.data[i])); console.log(`     fold vs ConvTranspose2d(k=8, s=4, p=2) definition: max |diff| = ${md.toExponential(2)}`); }
{ const X = new Float32Array(256); for (let i = 0; i < 256; i++) X[i] = rand(); const Pt = P09.patchesOf(X, 1); let md = 0;
  for (let p = 0; p < 16; p++) { const oy = p >> 2, ox = p & 3; for (let ky = 0; ky < 8; ky++) for (let kx = 0; kx < 8; kx++) { const iy = 4 * oy - 2 + ky, ix = 4 * ox - 2 + kx, v = iy >= 0 && iy < 16 && ix >= 0 && ix < 16 ? X[iy * 16 + ix] : 0; md = Math.max(md, Math.abs(v - Pt[p * 64 + ky * 8 + kx])); } }
  console.log(`     patchesOf vs Conv2d(k=8, s=4, p=2) windows: max |diff| = ${md.toExponential(2)}`); }
// 5. VQ：码本损失对码本、commitment 对编码器输出（idx 在 ±eps 内不变时就是精确梯度）；直通估计
{ const Ze = P(2 * 16, 3, 0.5), E = P(12, 3, 0.5);
  numGradCheck('vq codebook loss → E', () => P09.vq(Ze, E, 2).cb, [E], 1e-3);
  numGradCheck('vq commitment loss → Ze', () => P09.vq(Ze, E, 2).cm, [Ze], 1e-3);
  AG.clearTape(); Ze.grad = null;
  const q = P09.vq(Ze, E, 2), L1 = proj(AG.tanh(q.st)); AG.backward(L1);
  const st = Float64Array.from(Ze.grad);
  const Zq = AG.param('zq', 32, 3); Zq.data.set(q.st.data); AG.clearTape(); AG.backward(proj(AG.tanh(Zq)));
  let md = 0; for (let i = 0; i < st.length; i++) md = Math.max(md, Math.abs(st[i] - Zq.grad[i]));
  console.log(`     straight-through: max |∂L/∂ze − ∂L/∂zq| = ${md.toExponential(2)}（定义如此：应为 0）`); }
// 整网校验时把 ReLU 临时换成光滑的 tanh、eps 用 1e-2：只检验本页算子与组合方式；ReLU 的折点和 float32 累加噪声会让小 eps 的差分不可靠
const reluOrig = AG.relu; AG.relu = AG.tanh;
// 6. 整个 VQ-VAE：解码器参数（码字固定时是精确梯度）；码本只由码本损失更新
{ const m = P09.makeVQ(16, 3), X = new Float32Array(2 * 256); for (let i = 0; i < X.length; i++) X[i] = rand() < 0.7 ? 0 : rand();
  const Pt = AG.tensor(2 * 16, 64, P09.patchesOf(X, 2)), Xt = AG.tensor(2, 256, X);
  const loss = () => { const o = P09.vqForward(m, Pt, 2); return AG.add(P09.bce(o.logits, Xt), AG.add(o.q.cb, AG.scale(o.q.cm, 0.25))); };
  dirCheck('VQ-VAE decoder params', loss, [m.d1.W, m.d1.b, m.d2.W, m.d2b]);
  dirCheck('VQ-VAE dec.conv1 bias', loss, [m.d1.b]);
  dirCheck('VQ-VAE convT bias', loss, [m.d2b]);
  // 编码器经由直通估计：替身前向 = 把量化换成「ze + 固定偏移 (zq₀ − ze₀)」、码字下标固定——它的真实梯度就是直通估计给的梯度
  const o0 = AG.noGrad(() => P09.vqForward(m, Pt, 2)), off = AG.tensor(o0.ze.r, o0.ze.c); for (let i = 0; i < off.size; i++) off.data[i] = o0.q.st.data[i] - o0.ze.data[i];
  const idx0 = o0.q.idx, encOnly = () => { const h = AG.relu(AG.linear(Pt, m.e1.W, m.e1.b)), ze = AG.linear(P09.grid3x3(h), m.e2.W, m.e2.b);
      if (AG.enabled) { const o = P09.vqForward(m, Pt, 2); let same = true; for (let i = 0; i < idx0.length; i++) if (o.q.idx[i] !== idx0[i]) same = false; if (!same) throw new Error('idx changed'); return AG.add(P09.bce(o.logits, Xt), AG.scale(o.q.cm, 0.25)); }
      const zin = AG.add(ze, off), g = AG.relu(AG.linear(P09.grid3x3(zin), m.d1.W, m.d1.b)), lg = P09.fold(AG.matmul(g, m.d2.W), m.d2b);
      let cm = 0; for (let i = 0; i < ze.r; i++) for (let c = 0; c < ze.c; c++) cm += (ze.data[i * ze.c + c] - m.cb.data[idx0[i] * ze.c + c]) ** 2;
      const L = P09.bce(lg, Xt); L.data[0] += 0.25 * cm / 2; return L; };
  dirCheck('VQ-VAE encoder via straight-through', encOnly, [m.e1.W, m.e1.b, m.e2.W, m.e2.b], 2e-3);
  AG.clearTape(); m.P.forEach((p) => { p.grad = null; }); AG.backward(loss());
  const full = Float64Array.from(m.cb.grad);
  AG.clearTape(); m.cb.grad = null; AG.backward(P09.vqForward(m, Pt, 2).q.cb);
  let md = 0; for (let i = 0; i < full.length; i++) md = Math.max(md, Math.abs(full[i] - m.cb.grad[i]));
  console.log(`     codebook grad (full loss) vs codebook loss alone: max |diff| = ${md.toExponential(2)}（应为 0）`); }
// 7. MLP 自编码器 / VAE 的完整损失（全部参数）
{ const m = P09.makeAE('vae', 4), X = AG.tensor(3, 256); for (let i = 0; i < X.size; i++) X.data[i] = rand() < 0.7 ? 0 : rand();
  const eps = AG.tensor(3, 2); for (let i = 0; i < 6; i++) eps.data[i] = LM.randn(rand);
  dirCheck('VAE loss = BCE + β·KL (all params)', () => { const e = P09.encode(m, X), z = P09.reparam(e.mu, e.logv, eps); return AG.add(P09.bce(P09.decode(m, z), X), AG.scale(P09.klNormal(e.mu, e.logv), 1.0)); }, m.P, 3e-4); }
{ const m = P09.makeAE('ae', 5), X = AG.tensor(3, 256); for (let i = 0; i < X.size; i++) X.data[i] = rand() < 0.7 ? 0 : rand();
  dirCheck('AE loss = BCE (all params)', () => P09.bce(P09.decode(m, P09.encode(m, X).z), X), m.P, 3e-4); }
AG.relu = reluOrig;
console.log(fails ? `${fails} FAILED` : 'all gradient checks passed');
