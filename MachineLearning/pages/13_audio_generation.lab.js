// 13_audio_generation.lab.js — 系列第 13 篇：音频与语音（频谱、音频 token 与生成）
// 构建（在 pages/ 目录下）：node <skill>/scripts/build_doc.mjs llm_common.js 13_audio_generation.lab.js -o 13_audio_generation.html
// 页面里的每个声音都由下面的 JS 合成（旋律、啁啾、共振峰元音、噪声），每张预览 / 每个 ▶ 按钮都来自 stage 真实算出的数据。
// Node 原型（与页面同一份代码：加载本文件 PipelineLab.mount 之前的部分）：node tools/p13_proto.js <what>

const P13 = (() => {
    'use strict';
    const SR = 8000;                       // 全篇统一采样率（电话音质）：浏览器里能实时算 STFT / Griffin-Lim / 训练
    const TAU = 2 * Math.PI;
    const AG = LM.AG;
    const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

    // ═════════════════════════════ FFT（基 2，原地；正变换 e^{−i}，逆变换 e^{+i} 带 1/n） ═════════════════════════════
    const plans = new Map();
    function fftPlan(n) {
        let p = plans.get(n); if (p) return p;
        const bits = Math.round(Math.log2(n)), rev = new Uint32Array(n), cs = new Float64Array(n >> 1), sn = new Float64Array(n >> 1);
        for (let i = 0; i < n; i++) { let r = 0, x = i; for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
        for (let i = 0; i < n >> 1; i++) { cs[i] = Math.cos(TAU * i / n); sn[i] = Math.sin(TAU * i / n); }
        p = { n, rev, cs, sn }; plans.set(n, p); return p;
    }
    /** 原地复数 FFT：re / im 为 Float64Array(n)，n = 2 的幂 */
    function fft(re, im, inverse) {
        const n = re.length, p = fftPlan(n), rev = p.rev, cs = p.cs, sn = p.sn, sg = inverse ? 1 : -1;
        for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
        for (let size = 2; size <= n; size <<= 1) {
            const half = size >> 1, step = n / size;
            for (let i = 0; i < n; i += size) {
                for (let j = 0, k = 0; j < half; j++, k += step) {
                    const wr = cs[k], wi = sg * sn[k], a = i + j, b = a + half;
                    const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
                }
            }
        }
        if (inverse) { const s = 1 / n; for (let i = 0; i < n; i++) { re[i] *= s; im[i] *= s; } }
    }
    const winCache = new Map();
    /** 周期 Hann 窗 w[n] = 0.5 − 0.5·cos(2πn/N)（与 torch.hann_window 默认一致） */
    function hann(N) { let w = winCache.get(N); if (!w) { w = new Float64Array(N); for (let n = 0; n < N; n++) w[n] = 0.5 - 0.5 * Math.cos(TAU * n / N); winCache.set(N, w); } return w; }

    // ═════════════════════════════ STFT / iSTFT（center = True，两端补 N/2 个零；两帧打包成一次复数 FFT） ═════════════════════════════
    /** x → { re, im, T, nb, N, H, L }：第 t 帧第 k 个 bin 在 [t·nb + k]；帧 t 覆盖样本 [t·H − N/2, t·H + N/2) */
    function stft(x, N, H, out) {
        const L = x.length, T = 1 + Math.floor(L / H), nb = (N >> 1) + 1, w = hann(N);
        const re = out && out.re && out.re.length === T * nb ? out.re : new Float64Array(T * nb);
        const im = out && out.im && out.im.length === T * nb ? out.im : new Float64Array(T * nb);
        const fr = new Float64Array(N), fi = new Float64Array(N), h2 = N >> 1;
        for (let t = 0; t < T; t += 2) {
            const s0 = t * H - h2, s1 = s0 + H, two = t + 1 < T;
            for (let n = 0; n < N; n++) {
                const i0 = s0 + n, i1 = s1 + n;
                fr[n] = i0 >= 0 && i0 < L ? x[i0] * w[n] : 0;
                fi[n] = two && i1 >= 0 && i1 < L ? x[i1] * w[n] : 0;
            }
            fft(fr, fi, false);
            const o0 = t * nb, o1 = o0 + nb;
            for (let k = 0; k < nb; k++) {
                const kr = (N - k) & (N - 1), zr = fr[k], zi = fi[k], vr = fr[kr], vi = fi[kr];
                re[o0 + k] = 0.5 * (zr + vr); im[o0 + k] = 0.5 * (zi - vi);
                if (two) { re[o1 + k] = 0.5 * (zi + vi); im[o1 + k] = -0.5 * (zr - vr); }
            }
        }
        return { re, im, T, nb, N, H, L };
    }
    const envCache = new Map();
    function wsumEnv(N, H, T, L) {   // Σ_t w²[n − t·H]（iSTFT 的窗平方和归一化）
        const key = N + ',' + H + ',' + T + ',' + L; let e = envCache.get(key); if (e) return e;
        const w = hann(N), h2 = N >> 1; e = new Float64Array(L);
        for (let t = 0; t < T; t++) for (let n = 0; n < N; n++) { const i = t * H - h2 + n; if (i >= 0 && i < L) e[i] += w[n] * w[n]; }
        for (let i = 0; i < L; i++) e[i] = e[i] > 1e-8 ? 1 / e[i] : 0;
        if (envCache.size > 24) envCache.clear();
        envCache.set(key, e); return e;
    }
    /** 逆 STFT：加窗重叠相加，再除以 Σw²（与 torch.istft 相同的最小二乘逆） */
    function istft(re, im, T, N, H, L, out) {
        const nb = (N >> 1) + 1, w = hann(N), h2 = N >> 1, x = out && out.length === L ? out : new Float64Array(L);
        x.fill(0);
        const fr = new Float64Array(N), fi = new Float64Array(N);
        for (let t = 0; t < T; t += 2) {
            const two = t + 1 < T, o0 = t * nb, o1 = o0 + nb;
            for (let k = 0; k < N; k++) {
                let ar, ai, br = 0, bi = 0;
                if (k < nb) { ar = re[o0 + k]; ai = im[o0 + k]; if (two) { br = re[o1 + k]; bi = im[o1 + k]; } }
                else { const q = N - k; ar = re[o0 + q]; ai = -im[o0 + q]; if (two) { br = re[o1 + q]; bi = -im[o1 + q]; } }
                if (k === 0 || k === h2) { ai = 0; bi = 0; }           // DC / Nyquist 只取实部（与 irfft 相同）
                fr[k] = ar - bi; fi[k] = ai + br;                        // Z = A + i·B
            }
            fft(fr, fi, true);
            const s0 = t * H - h2, s1 = s0 + H;
            for (let n = 0; n < N; n++) {
                const i0 = s0 + n; if (i0 >= 0 && i0 < L) x[i0] += fr[n] * w[n];
                if (two) { const i1 = s1 + n; if (i1 >= 0 && i1 < L) x[i1] += fi[n] * w[n]; }
            }
        }
        const e = wsumEnv(N, H, T, L);
        for (let i = 0; i < L; i++) x[i] *= e[i];
        return x;
    }
    /** |X|² → dB（10·log10，下限 −100 dB） */
    function powerDb(sp, out) {
        const n = sp.re.length, o = out && out.length === n ? out : new Float32Array(n);
        for (let i = 0; i < n; i++) { const p = sp.re[i] * sp.re[i] + sp.im[i] * sp.im[i]; o[i] = 10 * Math.log10(p + 1e-10); }
        return o;
    }
    function magnitude(sp, out) { const n = sp.re.length, o = out && out.length === n ? out : new Float64Array(n); for (let i = 0; i < n; i++) o[i] = Math.hypot(sp.re[i], sp.im[i]); return o; }

    // ═════════════════════════════ Mel 滤波器组（HTK 公式，三角形峰值 = 1，与 torchaudio 默认 mel_scale="htk", norm=None 相同） ═════════════════════════════
    const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
    const mel2hz = (m) => 700 * (Math.pow(10, m / 2595) - 1);
    /** n_mels 个三角滤波器：rows[m] = { k0, w: Float64Array }（稀疏：只存非零的 bin），fc = 各中心频率 */
    function melBank(N, nMels, fmin, fmax) {
        fmin = fmin || 0; fmax = fmax || SR / 2;
        const nb = (N >> 1) + 1, m0 = hz2mel(fmin), m1 = hz2mel(fmax), pts = [];
        for (let i = 0; i < nMels + 2; i++) pts.push(mel2hz(m0 + (m1 - m0) * i / (nMels + 1)));
        const rows = [], dense = new Float64Array(nMels * nb);
        let empty = 0;
        for (let m = 0; m < nMels; m++) {
            const fl = pts[m], fc = pts[m + 1], fr = pts[m + 2], ws = [];
            let k0 = -1;
            for (let k = 0; k < nb; k++) {
                const f = k * SR / N, v = f <= fl || f >= fr ? 0 : f <= fc ? (f - fl) / (fc - fl) : (fr - f) / (fr - fc);
                if (v > 0) { if (k0 < 0) k0 = k; ws.length = k - k0 + 1; ws[k - k0] = v; dense[m * nb + k] = v; }
            }
            if (k0 < 0) { empty++; k0 = 0; }
            const w = Float64Array.from(ws, (v) => v || 0);
            rows.push({ k0, w });
        }
        return { N, nb, nMels, rows, dense, fc: pts.slice(1, -1), edges: pts, empty };
    }
    /** 功率谱（|X|²）→ mel 能量 [T × nMels] */
    function melSpec(sp, bank, out) {
        const T = sp.T, nb = sp.nb, M = bank.nMels, o = out && out.length === T * M ? out : new Float64Array(T * M);
        for (let t = 0; t < T; t++) for (let m = 0; m < M; m++) {
            const r = bank.rows[m]; let s = 0;
            for (let j = 0; j < r.w.length; j++) { const i = t * nb + r.k0 + j; s += r.w[j] * (sp.re[i] * sp.re[i] + sp.im[i] * sp.im[i]); }
            o[t * M + m] = s;
        }
        return o;
    }
    /** mel 能量 → 线性功率谱的近似逆：非负最小二乘（乘性更新，Lee & Seung），初值 = 转置 + 归一化 */
    function melInverse(mel, T, bank, iters) {
        const nb = bank.nb, M = bank.nMels, P = new Float64Array(T * nb), num = new Float64Array(nb), mp = new Float64Array(M), den = new Float64Array(nb);
        const colSum = new Float64Array(M); bank.rows.forEach((r, m) => { for (let j = 0; j < r.w.length; j++) colSum[m] += r.w[j]; });
        for (let t = 0; t < T; t++) {
            const p = P.subarray(t * nb, (t + 1) * nb), y = mel.subarray(t * M, (t + 1) * M);
            num.fill(0);
            for (let m = 0; m < M; m++) { const r = bank.rows[m], v = colSum[m] > 0 ? y[m] / colSum[m] : 0; for (let j = 0; j < r.w.length; j++) num[r.k0 + j] += r.w[j] * v; }
            for (let k = 0; k < nb; k++) p[k] = num[k] + 1e-12;
            for (let it = 0; it < (iters || 0); it++) {   // P ← P ⊙ (Mᵀy) / (MᵀM P)
                for (let m = 0; m < M; m++) { const r = bank.rows[m]; let s = 0; for (let j = 0; j < r.w.length; j++) s += r.w[j] * p[r.k0 + j]; mp[m] = s; }
                num.fill(0); den.fill(0);
                for (let m = 0; m < M; m++) { const r = bank.rows[m]; for (let j = 0; j < r.w.length; j++) { num[r.k0 + j] += r.w[j] * y[m]; den[r.k0 + j] += r.w[j] * mp[m]; } }
                for (let k = 0; k < nb; k++) if (den[k] > 0) p[k] *= num[k] / (den[k] + 1e-20);
            }
        }
        return P;
    }

    // ═════════════════════════════ Griffin-Lim（只知道幅度，迭代估计相位） ═════════════════════════════
    /** 目标幅度 A [T × nb] → 迭代器；o.init = 'random' | 'zero'，o.momentum（快速 Griffin-Lim，Perraudin 2013；0 = 原始算法） */
    function glCreate(A, T, N, H, L, o) {
        o = o || {};
        const nb = (N >> 1) + 1, n = T * nb, re = new Float64Array(n), im = new Float64Array(n), rand = LM.rng(o.seed || 1234);
        for (let i = 0; i < n; i++) { const ph = o.init === 'zero' ? 0 : TAU * rand(); re[i] = A[i] * Math.cos(ph); im[i] = A[i] * Math.sin(ph); }
        let an = 0; for (let i = 0; i < n; i++) an += A[i] * A[i];
        const G = { A, T, N, H, L, nb, re, im, x: new Float64Array(L), it: 0, sc: [], aNorm: Math.sqrt(an), mom: o.momentum || 0, pre: o.momentum ? { re: new Float64Array(n), im: new Float64Array(n) } : null, sp: null };
        istft(re, im, T, N, H, L, G.x);
        return G;
    }
    /** 一次迭代：x = iSTFT(X)；Y = STFT(x)；记录谱收敛度 ‖|Y| − A‖/‖A‖；X ← A·Y/|Y| */
    function glStep(G) {
        const sp = (G.sp = stft(G.x, G.N, G.H, G.sp)), A = G.A, n = A.length, re = G.re, im = G.im;
        let err = 0;
        for (let i = 0; i < n; i++) {
            const yr = sp.re[i], yi = sp.im[i], m = Math.hypot(yr, yi); err += (m - A[i]) * (m - A[i]);
            let pr = m > 1e-12 ? A[i] * yr / m : A[i], pi = m > 1e-12 ? A[i] * yi / m : 0;
            if (G.pre) { const qr = pr, qi = pi; pr = qr + G.mom * (qr - G.pre.re[i]); pi = qi + G.mom * (qi - G.pre.im[i]); G.pre.re[i] = qr; G.pre.im[i] = qi; }
            re[i] = pr; im[i] = pi;
        }
        G.sc.push(Math.sqrt(err) / (G.aNorm || 1));
        istft(re, im, G.T, G.N, G.H, G.L, G.x);
        G.it++;
        return G.sc[G.sc.length - 1];
    }

    // ═════════════════════════════ 合成器：音符（加性合成）、旋律、啁啾、噪声 ═════════════════════════════
    const midiHz = (m) => 440 * Math.pow(2, (m - 69) / 12);
    const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const noteName = (m) => NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1);
    /** 两种音色：风琴（谐波 1/k，持续）与拨弦（高次谐波衰减更快）。env(i, n) = 第 i 个样本的包络，harm(k) = 第 k 次谐波的振幅 */
    const TIMBRES = [
        { name: '风琴', harm: (k) => 1 / k, decay: () => 0, att: 0.010, rel: 0.030, body: () => 1 },
        { name: '拨弦', harm: (k) => 1 / Math.pow(k, 0.7), decay: (k) => 2.5 + 2.2 * k, att: 0.003, rel: 0.020, body: (t) => Math.exp(-3.2 * t) },
    ];
    /** 在 out[off …off+n) 上叠加一个音符：谐波加性合成，起音时相位归零（retrigger），谐波限制在 3.8 kHz 以下（不越过奈奎斯特） */
    function addNote(out, off, n, f0, tb, amp) {
        const env = new Float64Array(n), na = Math.max(1, Math.round(tb.att * SR)), nr = Math.max(1, Math.round(tb.rel * SR));
        for (let i = 0; i < n; i++) env[i] = Math.min(1, i / na, (n - i) / nr) * tb.body(i / SR);
        for (let k = 1; k * f0 < 3800; k++) {   // 递推振荡器 s[i] = 2cos(w)·s[i−1] − s[i−2]（= sin(w·i)），衰减按每个样本乘一个常数
            const a = amp * tb.harm(k), w = TAU * k * f0 / SR, c2 = 2 * Math.cos(w), dec = Math.exp(-tb.decay(k) / SR);
            let y0 = 0, y1 = Math.sin(w), g = a;
            for (let i = 0; i < n && off + i < out.length; i++) { out[off + i] += g * env[i] * y0; const y2 = c2 * y1 - y0; y0 = y1; y1 = y2; g *= dec; }
        }
    }
    const SCALE = [60, 62, 64, 67, 69, 72, 74, 76];          // C 大调五声音阶：C4 D4 E4 G4 A4 C5 D5 E5
    /** 随机旋律：在音阶上随机游走（多为级进），以主音结束。返回 [{ deg, beats }] */
    function makeMelody(rand, nNotes) {
        const out = []; let d = [0, 2, 4, 5][Math.floor(rand() * 4)];
        for (let i = 0; i < nNotes; i++) {
            const last = i === nNotes - 1;
            if (last) d = d >= 4 ? 5 : 0;
            out.push({ deg: d, beats: last ? 2 : rand() < 0.22 ? 2 : 1 });
            const r = rand(), st = r < 0.32 ? 1 : r < 0.64 ? -1 : r < 0.76 ? 2 : r < 0.88 ? -2 : r < 0.94 ? 3 : 0;
            d += st; if (d < 0) d = -d; if (d > 7) d = 14 - d;
        }
        return out;
    }
    /** 旋律 → 波形。beat = 一拍的样本数（与编解码器的帧对齐）；返回 { x, onsets: [{ i, n, midi }] } */
    function renderMelody(mel, beat, tbIndex, amp, pad) {
        pad = pad || 0;
        const total = mel.reduce((s, n) => s + n.beats, 0) * beat + 2 * pad, x = new Float64Array(total), notes = [];
        let pos = pad;
        mel.forEach((nt) => { const n = nt.beats * beat, midi = SCALE[nt.deg]; addNote(x, pos, n, midiHz(midi), TIMBRES[tbIndex], amp); notes.push({ i: pos, n, midi, deg: nt.deg }); pos += n; });
        return { x, notes };
    }
    function chirpLin(f0, f1, n, amp) { const x = new Float64Array(n), T = n / SR; for (let i = 0; i < n; i++) { const t = i / SR; x[i] = amp * Math.sin(TAU * (f0 * t + 0.5 * (f1 - f0) * t * t / T)); } return x; }
    function fadeEdges(x, ms) { const m = Math.max(1, Math.round(ms * SR / 1000)); for (let i = 0; i < m && i < x.length; i++) { const g = i / m; x[i] *= g; x[x.length - 1 - i] *= g; } return x; }

    // ═════════════════════════════ 共振峰合成器（Klatt 1980 式级联谐振器 + Rosenberg 声门脉冲） ═════════════════════════════
    /** 二阶谐振器 y[n] = A·x[n] + B·y[n−1] + C·y[n−2]（Klatt 1980）：中心频率 F、带宽 BW（Hz），直流增益 1 */
    function resonatorCoef(F, BW) { const T = 1 / SR, C = -Math.exp(-TAU * BW * T), B = 2 * Math.exp(-Math.PI * BW * T) * Math.cos(TAU * F * T); return { A: 1 - B - C, B, C }; }
    /**
     * 帧级声学参数 → 波形。track: 每 5 ms 一帧 { f0, F: [F1..F4], BW: [B1..B4], av（浊音振幅）, ah（送气噪声振幅）}，帧间线性插值。
     * 声源：Rosenberg 脉冲（开相 40%、闭相 16%），一阶差分近似唇辐射；加 ±jitter 的基频抖动。
     */
    function formantSynth(track, rand, o) {
        o = o || {};
        const hopS = Math.round(0.005 * SR), n = (track.length - 1) * hopS + 1, out = new Float64Array(n), jit = o.jitter ?? 0.01;
        const y1 = [0, 0, 0, 0], y2 = [0, 0, 0, 0], coef = [null, null, null, null];
        let phase = 0, prevG = 0, curJ = 1;
        for (let i = 0; i < n; i++) {
            const fi = i / hopS, a = Math.min(track.length - 2, Math.floor(fi)), u = fi - a, p = track[a], q = track[a + 1];
            const f0 = p.f0 + (q.f0 - p.f0) * u, av = p.av + (q.av - p.av) * u, ah = (p.ah || 0) + ((q.ah || 0) - (p.ah || 0)) * u;
            // 声门脉冲：相位 0..1 走完一个基音周期（Rosenberg：开相 40% 余弦上升，闭相 16% 余弦下降）
            phase += f0 * curJ / SR;
            if (phase >= 1) { phase -= 1; curJ = 1 + jit * (2 * rand() - 1); }
            const tp = 0.40, tn = 0.16;
            const g = phase < tp ? 0.5 * (1 - Math.cos(Math.PI * phase / tp)) : phase < tp + tn ? Math.cos(0.5 * Math.PI * (phase - tp) / tn) : 0;
            let s = av * (g - prevG) * 6 + ah * (2 * rand() - 1) * 0.08;   // 浊音（差分 ≈ 唇辐射）+ 送气噪声
            prevG = g;
            if ((i & 7) === 0) for (let k = 0; k < 4; k++) coef[k] = resonatorCoef(p.F[k] + (q.F[k] - p.F[k]) * u, p.BW[k] + (q.BW[k] - p.BW[k]) * u);   // 每 1 ms 更新谐振器系数
            for (let k = 0; k < 4; k++) { const c = coef[k], y = c.A * s + c.B * y1[k] + c.C * y2[k]; y2[k] = y1[k]; y1[k] = y; s = y; }
            out[i] = s;
        }
        return out;
    }
    /** 五个元音（取自 Peterson & Barney 1952 成年男性平均共振峰；e、o 用 /ɛ/、/ɔ/）——第 08、09 步的 TTS 与 ASR 共用 */
    const VOWELS = [
        { key: 'a', ipa: 'ɑ', F: [730, 1090, 2440, 3300] },
        { key: 'e', ipa: 'ɛ', F: [530, 1840, 2480, 3300] },
        { key: 'i', ipa: 'i', F: [270, 2290, 3010, 3500] },
        { key: 'o', ipa: 'ɔ', F: [570, 840, 2410, 3300] },
        { key: 'u', ipa: 'u', F: [300, 870, 2240, 3300] },
    ];
    const VBW = [70, 90, 150, 250];
    /** 一个持续元音：说话人 = { f0, vtl（声道缩放：女声 / 童声的共振峰整体偏高）}，返回波形 */
    function vowelClip(v, spk, dur, rand, o) {
        o = o || {};
        const nf = Math.round(dur / 0.005) + 1, tr = [];
        for (let f = 0; f < nf; f++) {
            const t = f / (nf - 1), env = Math.min(1, t / 0.12, (1 - t) / 0.12);
            const F = v.F.map((x, k) => Math.min(3850, x * spk.vtl * (1 + (spk.fj ? spk.fj[k] : 0))));
            tr.push({ f0: spk.f0 * (1 + 0.04 * Math.sin(TAU * 4.5 * t * dur + (spk.vib || 0))) * (1 - 0.06 * t), F, BW: VBW.map((b, k) => b * (spk.bw || 1) * (k === 3 ? 1.5 : 1)), av: env, ah: o.breathy || 0.02 * env });
        }
        return formantSynth(tr, rand, { jitter: spk.jit ?? 0.01 });
    }

    // ═════════════════════════════ MDCT（正弦窗、50% 重叠；TDAC：重叠相加后完美重建，且是正交变换） ═════════════════════════════
    const mdctPlans = new Map();
    function mdctPlan(M) {
        let p = mdctPlans.get(M); if (p) return p;
        const N = 2 * M, n0 = 0.5 + M / 2, w = new Float64Array(N), pr = new Float64Array(N), pi = new Float64Array(N), qr = new Float64Array(M), qi = new Float64Array(M);
        for (let n = 0; n < N; n++) { w[n] = Math.sin(Math.PI * (n + 0.5) / N); pr[n] = Math.cos(-Math.PI * n / N); pi[n] = Math.sin(-Math.PI * n / N); }
        for (let k = 0; k < M; k++) { const a = -Math.PI * n0 * (2 * k + 1) / N; qr[k] = Math.cos(a); qi[k] = Math.sin(a); }
        const ir = new Float64Array(M), ii = new Float64Array(M), jr = new Float64Array(N), ji = new Float64Array(N);   // 逆变换的前 / 后旋转因子
        for (let k = 0; k < M; k++) { const a = Math.PI * n0 * 2 * k / N; ir[k] = Math.cos(a); ii[k] = Math.sin(a); }
        for (let n = 0; n < N; n++) { const a = Math.PI * (n + n0) / N; jr[n] = Math.cos(a); ji[n] = Math.sin(a); }
        p = { M, N, n0, w, pr, pi, qr, qi, ir, ii, jr, ji, sc: Math.sqrt(2 / M), br: new Float64Array(N), bi: new Float64Array(N) };
        mdctPlans.set(M, p); return p;
    }
    /** X[k] = √(2/M)·Σ_n w[n]·x[n]·cos(π/M·(n + ½ + M/2)(k + ½))：一帧 2M 个样本 → M 个系数（一次 2M 点 FFT） */
    function mdctFrame(seg, p, out) {
        const N = p.N, M = p.M, br = p.br, bi = p.bi;
        for (let n = 0; n < N; n++) { const v = seg[n] * p.w[n]; br[n] = v * p.pr[n]; bi[n] = v * p.pi[n]; }
        fft(br, bi, false);
        for (let k = 0; k < M; k++) out[k] = p.sc * (br[k] * p.qr[k] - bi[k] * p.qi[k]);
    }
    /** 逆变换：y[n] = √(2/M)·w[n]·Σ_k X[k]·cos(…)，n < 2M；相邻帧重叠相加即还原 */
    function imdctFrame(X, p, out) {
        const N = p.N, M = p.M, br = p.br, bi = p.bi;
        for (let k = 0; k < N; k++) { if (k < M) { br[k] = X[k] * p.ir[k]; bi[k] = X[k] * p.ii[k]; } else { br[k] = 0; bi[k] = 0; } }
        fft(br, bi, true);
        for (let n = 0; n < N; n++) out[n] = p.sc * p.w[n] * N * (br[n] * p.jr[n] - bi[n] * p.ji[n]);
    }
    /** 整段信号 → 帧 [F × M]：帧 t 覆盖样本 [(t−1)·M, (t+1)·M)，两端补零；F = ⌈L/M⌉ + 1 */
    function mdct(x, M) {
        const p = mdctPlan(M), L = x.length, F = Math.ceil(L / M) + 1, C = new Float64Array(F * M), seg = new Float64Array(2 * M);
        for (let t = 0; t < F; t++) { const s = (t - 1) * M; for (let n = 0; n < 2 * M; n++) { const i = s + n; seg[n] = i >= 0 && i < L ? x[i] : 0; } mdctFrame(seg, p, C.subarray(t * M, (t + 1) * M)); }
        return { C, F, M, L };
    }
    function imdct(C, F, M, L) {
        const p = mdctPlan(M), y = new Float64Array(L), buf = new Float64Array(2 * M);
        for (let t = 0; t < F; t++) { imdctFrame(C.subarray(t * M, (t + 1) * M), p, buf); const s = (t - 1) * M; for (let n = 0; n < 2 * M; n++) { const i = s + n; if (i >= 0 && i < L) y[i] += buf[n]; } }
        return y;
    }

    // ═════════════════════════════ 残差向量量化 RVQ（EMA k-means 码本，与 SoundStream / EnCodec 的做法相同） ═════════════════════════════
    /** NQ 层、每层 K 个 D 维码字 */
    function rvqCreate(D, K, NQ, o) {
        o = o || {};
        const Q = { D, K, NQ, gamma: o.gamma ?? 0.95, dead: o.dead ?? 0.02, C: [], cnt: [], sum: [], norm: [], inited: 0, step: 0, rand: LM.rng(o.seed || 77), hist: [] };
        for (let l = 0; l < NQ; l++) { Q.C.push(new Float64Array(K * D)); Q.cnt.push(new Float64Array(K)); Q.sum.push(new Float64Array(K * D)); Q.norm.push(new Float64Array(K)); }
        return Q;
    }
    function refreshNorms(Q, l) { const C = Q.C[l], nr = Q.norm[l], D = Q.D; for (let k = 0; k < Q.K; k++) { let s = 0; for (let j = 0; j < D; j++) s += C[k * D + j] * C[k * D + j]; nr[k] = s; } }
    /** 最近码字：argmin_k ‖r − c_k‖² = argmin_k (‖c_k‖² − 2·r·c_k) */
    function nearest(Q, l, r, ro) {
        const C = Q.C[l], nr = Q.norm[l], D = Q.D, K = Q.K; let best = 0, bv = Infinity;
        for (let k = 0; k < K; k++) { let dot = 0; const o = k * D; for (let j = 0; j < D; j++) dot += r[ro + j] * C[o + j]; const v = nr[k] - 2 * dot; if (v < bv) { bv = v; best = k; } }
        return best;
    }
    /** 编码 F 帧（X: [F × D]）→ codes [F × NQ]；每层减去选中的码字，把残差交给下一层 */
    function rvqEncode(Q, X, F, nq, codes) {
        nq = nq || Q.NQ; codes = codes || new Int32Array(F * Q.NQ);
        const D = Q.D, r = new Float64Array(D);
        for (let t = 0; t < F; t++) {
            for (let j = 0; j < D; j++) r[j] = X[t * D + j];
            for (let l = 0; l < nq; l++) { const k = nearest(Q, l, r, 0); codes[t * Q.NQ + l] = k; const c = Q.C[l]; for (let j = 0; j < D; j++) r[j] -= c[k * D + j]; }
        }
        return codes;
    }
    /** 解码：前 nq 层码字相加 */
    function rvqDecode(Q, codes, F, nq) {
        const D = Q.D, X = new Float64Array(F * D);
        for (let t = 0; t < F; t++) for (let l = 0; l < nq; l++) { const k = codes[t * Q.NQ + l], c = Q.C[l]; for (let j = 0; j < D; j++) X[t * D + j] += c[k * D + j]; }
        return X;
    }
    /**
     * 一步训练（mini-batch）：batch 的残差逐层通过「当前」码本；第 l 层用 EMA 更新
     *   N_k ← γ·N_k + (1−γ)·n_k      m_k ← γ·m_k + (1−γ)·Σ_{i→k} r_i      c_k = m_k / N_k
     * 第一次见到数据时，用 batch 里随机的残差向量初始化码本；EMA 计数太小的「死码」换成 batch 里的随机残差（SoundStream / EnCodec 的做法）。
     * 返回每层之后的相对残差能量 ‖r_{l+1}‖² / ‖x‖²
     */
    function rvqTrainStep(Q, X, nFrames, B, rand) {
        const D = Q.D, K = Q.K, NQ = Q.NQ, g = Q.gamma, R = new Float64Array(B * D), idx = new Int32Array(B), asg = new Int32Array(B);
        for (let b = 0; b < B; b++) { const t = Math.floor(rand() * nFrames); idx[b] = t; for (let j = 0; j < D; j++) R[b * D + j] = X[t * D + j]; }
        let e0 = 0; for (let i = 0; i < B * D; i++) e0 += R[i] * R[i];
        const ratios = [];
        for (let l = 0; l < NQ; l++) {
            const C = Q.C[l], cnt = Q.cnt[l], sum = Q.sum[l];
            if (Q.inited <= l) {   // 初始化：K 个不同的随机残差
                const perm = LM.shuffle(Array.from({ length: B }, (_, i) => i), rand);
                for (let k = 0; k < K; k++) { const b = perm[k % B]; for (let j = 0; j < D; j++) { C[k * D + j] = R[b * D + j] * (1 + 0.01 * (rand() - 0.5)); sum[k * D + j] = C[k * D + j]; } cnt[k] = 1; }
                refreshNorms(Q, l); Q.inited = l + 1;
            }
            const nk = new Float64Array(K), sk = new Float64Array(K * D);
            for (let b = 0; b < B; b++) { const k = nearest(Q, l, R, b * D); asg[b] = k; nk[k]++; for (let j = 0; j < D; j++) sk[k * D + j] += R[b * D + j]; }
            for (let k = 0; k < K; k++) {
                cnt[k] = g * cnt[k] + (1 - g) * nk[k];
                for (let j = 0; j < D; j++) sum[k * D + j] = g * sum[k * D + j] + (1 - g) * sk[k * D + j];
            }
            // 量化：残差减去选中码字（用更新前的码本，与编码一致）
            for (let b = 0; b < B; b++) { const k = asg[b]; for (let j = 0; j < D; j++) R[b * D + j] -= C[k * D + j]; }
            for (let k = 0; k < K; k++) {
                if (cnt[k] < Q.dead * B / K) {   // 死码：换成这个 batch 里的一个随机残差（加回量化前的值）
                    const b = Math.floor(rand() * B), kk = asg[b];
                    for (let j = 0; j < D; j++) { const v = R[b * D + j] + C[kk * D + j]; sum[k * D + j] = v; }
                    cnt[k] = 1;
                }
                for (let j = 0; j < D; j++) C[k * D + j] = sum[k * D + j] / cnt[k];
            }
            refreshNorms(Q, l);
            let e = 0; for (let i = 0; i < B * D; i++) e += R[i] * R[i];
            ratios.push(e / (e0 || 1));
        }
        Q.step++;
        return ratios;
    }

    // ═════════════════════════════ 多码本 token 语言模型（MusicGen 式 delay pattern） ═════════════════════════════
    // 主干 = LM.gpt 的 Transformer（与第 04 / 05 篇相同）；输入 = NQ 个码本各自的 embedding 相加；输出 = NQ 个线性头并行预测。
    // 延迟排布：位置 p 的第 k 行放 q_k(p − k)（第 k 个码本整体晚 k 步），p − k < 0 处放 PAD（id = K，只作输入、从不作为目标）。
    /** 原始码矩阵 codes[F × NQ] → 延迟排布 [(F + NQ − 1) × NQ] */
    function delayPattern(codes, F, NQ, K) {
        const P = F + NQ - 1, out = new Int32Array(P * NQ).fill(K);
        for (let p = 0; p < P; p++) for (let k = 0; k < NQ; k++) { const t = p - k; if (t >= 0 && t < F) out[p * NQ + k] = codes[t * NQ + k]; }
        return out;
    }
    /** 延迟排布 → 原始码矩阵（生成结束后还原；没生成完整的尾帧丢掉） */
    function undelay(D, P, NQ) {
        const F = P - NQ + 1, out = new Int32Array(Math.max(0, F) * NQ);
        for (let t = 0; t < F; t++) for (let k = 0; k < NQ; k++) out[t * NQ + k] = D[(t + k) * NQ + k];
        return out;
    }
    function tlmCreate(cfg, seed) {
        const g = LM.gpt.create({ V: 1, T: cfg.T, d: cfg.d, L: cfg.L, H: cfg.H, ff: cfg.ff }, seed);
        const rand = LM.rng((seed * 31 + 7) >>> 0), emb = [], Wh = [], bh = [];
        for (let k = 0; k < cfg.NQ; k++) {
            const E = AG.param(`emb${k}`, cfg.K + 1, cfg.d, LM.init.normal(rand, 0.02)); E.decay = true; emb.push(E);
            const W = AG.param(`head${k}.W`, cfg.d, cfg.K, LM.init.normal(rand, 0.02)); W.decay = true; Wh.push(W);
            bh.push(AG.param(`head${k}.b`, 1, cfg.K));
        }
        const params = g.params.filter((p) => p !== g.wte).concat(emb, Wh, bh);
        return { cfg, g, emb, Wh, bh, params, nParams: params.reduce((s, p) => s + p.size, 0) };
    }
    const dummyIds = new Map();
    /** X、Y：Int32Array(B·T·NQ)，第 i 个位置第 k 个码本在 [i·NQ + k]；Y 里 −1 = 不计损失 */
    function tlmForward(m, X, B, T, Y) {
        const NQ = m.cfg.NQ, n = B * T;
        let inp = null;
        for (let k = 0; k < NQ; k++) { const ids = new Int32Array(n); for (let i = 0; i < n; i++) ids[i] = X[i * NQ + k]; const e = AG.gather(m.emb[k], ids); inp = inp ? AG.add(inp, e) : e; }
        let di = dummyIds.get(n); if (!di) { di = new Int32Array(n); dummyIds.set(n, di); }
        const hf = LM.gpt.forward(m.g, di, B, T, { inputEmb: inp, trace: true }).trace.hf;
        const logits = [], losses = [];
        for (let k = 0; k < NQ; k++) {
            const lg = AG.linear(hf, m.Wh[k], m.bh[k]); logits.push(lg);
            if (Y) { const tg = new Int32Array(n); for (let i = 0; i < n; i++) tg[i] = Y[i * NQ + k]; losses.push(AG.crossEntropy(lg, tg)); }
        }
        let loss = null;
        if (Y) { loss = losses[0]; for (let k = 1; k < NQ; k++) loss = AG.add(loss, losses[k]); loss = AG.scale(loss, 1 / NQ); }
        return { logits, losses, loss };
    }
    /** 随机取 B 个窗口：x = 延迟流[s, s+T)，y = 延迟流[s+1, s+T+1)；目标里的 PAD 记为 −1 */
    function tlmBatch(stream, P, NQ, K, B, T, rand, out) {
        out = out || { x: new Int32Array(B * T * NQ), y: new Int32Array(B * T * NQ), starts: new Int32Array(B) };
        for (let b = 0; b < B; b++) {
            const s = Math.floor(rand() * (P - T - 1)); out.starts[b] = s;
            for (let t = 0; t < T; t++) for (let k = 0; k < NQ; k++) { const i = (b * T + t) * NQ + k; out.x[i] = stream[(s + t) * NQ + k]; const yv = stream[(s + t + 1) * NQ + k]; out.y[i] = yv === K ? -1 : yv; }
        }
        return out;
    }
    function tlmTrainStep(m, opt, X, Y, B, T, lr) {
        LM.zeroGrad(m.params);
        const out = tlmForward(m, X, B, T, Y);
        AG.backward(out.loss);
        const gn = LM.clipGradNorm(m.params, 1.0);
        opt.step(lr);
        return { loss: out.loss.item(), per: out.losses.map((l) => l.item()), gn };
    }
    /** 无梯度的整段评估：每个码本的平均交叉熵 */
    function tlmEval(m, X, Y, B, T) { return AG.noGrad(() => tlmForward(m, X, B, T, Y).losses.map((l) => l.item())); }
    /**
     * 自回归生成（KV cache，与 LM.gpt.generate 同一套逐 token 计算）：每一步把 NQ 个码本的 embedding 相加作为输入，NQ 个头同时采样。
     * 延迟排布保证「第 k 个码本的第 t 帧」在第 t + k 步才被采样——那时同一帧的前 k 个码本已经确定。
     * 上下文写满 T 后保留后半段重新编码（滑动窗口），所以可以生成比 T 更长的序列。
     */
    function tlmGenerate(m, prompt, P0, nNew, o) {
        o = o || {};
        const c = m.g.cfg, d = c.d, H = c.H, dh = d / H, Tm = c.T, NQ = m.cfg.NQ, K = m.cfg.K, rand = o.rand || Math.random, temp = o.temperature ?? 1;
        const L = m.g.layers, kc = L.map(() => new Float32Array(Tm * d)), vc = L.map(() => new Float32Array(Tm * d));
        const x = new Float32Array(d), h = new Float32Array(d), q = new Float32Array(d), kv = new Float32Array(d), vv = new Float32Array(d), av = new Float32Array(d), tmp = new Float32Array(d), fv = new Float32Array(c.ff), att = new Float32Array(Tm);
        const GK = Math.sqrt(2 / Math.PI), lnRow = (src, g, b, dst) => { let mu = 0; for (let j = 0; j < d; j++) mu += src[j]; mu /= d; let v = 0; for (let j = 0; j < d; j++) v += (src[j] - mu) ** 2; v /= d; const r = 1 / Math.sqrt(v + 1e-5); for (let j = 0; j < d; j++) dst[j] = (src[j] - mu) * r * g[j] + b[j]; };
        const lin = (src, W, b, nIn, nOut, dst) => { for (let j = 0; j < nOut; j++) dst[j] = b ? b[j] : 0; for (let p = 0; p < nIn; p++) { const s = src[p]; if (s === 0) continue; const o2 = p * nOut; for (let j = 0; j < nOut; j++) dst[j] += s * W[o2 + j]; } };
        const logits = m.cfg.NQ ? Array.from({ length: NQ }, () => new Float32Array(K)) : null;
        const step = (toks, toff, pos) => {   // 一个位置：返回 NQ 组 logits
            const P = m.g.wpe.data;
            for (let j = 0; j < d; j++) x[j] = P[pos * d + j];
            for (let k = 0; k < NQ; k++) { const E = m.emb[k].data, id = toks[toff + k]; for (let j = 0; j < d; j++) x[j] += E[id * d + j]; }
            L.forEach((p, l) => {
                lnRow(x, p.ln1g.data, p.ln1b.data, h);
                lin(h, p.wq.data, p.bq.data, d, d, q); lin(h, p.wk.data, p.bk.data, d, d, kv); lin(h, p.wv.data, p.bv.data, d, d, vv);
                kc[l].set(kv, pos * d); vc[l].set(vv, pos * d); av.fill(0);
                for (let hh = 0; hh < H; hh++) {
                    const hc = hh * dh; let mx = -Infinity;
                    for (let j = 0; j <= pos; j++) { let s = 0; for (let e = 0; e < dh; e++) s += q[hc + e] * kc[l][j * d + hc + e]; s /= Math.sqrt(dh); att[j] = s; if (s > mx) mx = s; }
                    let z = 0; for (let j = 0; j <= pos; j++) { att[j] = Math.exp(att[j] - mx); z += att[j]; }
                    for (let j = 0; j <= pos; j++) { const w = att[j] / z; for (let e = 0; e < dh; e++) av[hc + e] += w * vc[l][j * d + hc + e]; }
                }
                lin(av, p.wo.data, p.bo.data, d, d, tmp); for (let j = 0; j < d; j++) x[j] += tmp[j];
                lnRow(x, p.ln2g.data, p.ln2b.data, h);
                lin(h, p.w1.data, p.b1.data, d, c.ff, fv);
                for (let j = 0; j < c.ff; j++) { const u = fv[j]; fv[j] = 0.5 * u * (1 + Math.tanh(GK * (u + 0.044715 * u * u * u))); }
                lin(fv, p.w2.data, p.b2.data, c.ff, d, tmp); for (let j = 0; j < d; j++) x[j] += tmp[j];
            });
            lnRow(x, m.g.lnfg.data, m.g.lnfb.data, h);
            for (let k = 0; k < NQ; k++) lin(h, m.Wh[k].data, m.bh[k].data, d, K, logits[k]);
            return logits;
        };
        const total = P0 + nNew, seq = new Int32Array(total * NQ).fill(K);
        seq.set(prompt.subarray(0, P0 * NQ));
        let w0 = 0, lg = null;   // 当前窗口在 seq 里的起点
        const encode = (from, to) => { for (let p = from; p < to; p++) lg = step(seq, p * NQ, p - w0); };
        encode(0, P0);
        const pr = new Float32Array(K);
        for (let p = P0; p < total; p++) {
            if (p - w0 >= Tm) { w0 = p - (Tm >> 1); encode(w0, p); }   // 滑动窗口：保留后 T/2 个位置重新编码
            for (let k = 0; k < NQ; k++) {
                const t = p - k;                                         // 这一行对应第 t 帧
                if (t < 0) { seq[p * NQ + k] = K; continue; }
                if (o.force && t < o.forceF) { seq[p * NQ + k] = o.force[t * NQ + k]; continue; }   // 提示里已知的帧（延迟的尾巴）
                LM.softmax(lg[k], pr, temp);
                if (o.topK) { const idx = LM.topk(pr, o.topK), keep = new Set(idx); let s = 0; for (let i = 0; i < K; i++) { if (!keep.has(i)) pr[i] = 0; s += pr[i]; } for (let i = 0; i < K; i++) pr[i] /= s; }
                seq[p * NQ + k] = o.greedy ? LM.argmax(lg[k]) : LM.sampleFrom(pr, rand);
            }
            if (p + 1 < total) lg = step(seq, p * NQ, p - w0);
        }
        return seq;
    }

    // ═════════════════════════════ TTS：文本 → 拼音 → 音素 + 声调 → 时长 → F0 / 共振峰轨迹 → 共振峰声码器 ═════════════════════════════
    /** 音素的共振峰目标（成年男性；说话人的声道缩放 vtl 整体乘上去）。amp = 浊音振幅，asp = 送气 */
    const PHONES = {
        a: { F: [730, 1090, 2440, 3300], amp: 1 }, o: { F: [570, 840, 2410, 3300], amp: 1 }, e: { F: [530, 1840, 2480, 3300], amp: 1 },
        E: { F: [460, 1220, 2420, 3300], amp: 1 }, i: { F: [270, 2290, 3010, 3500], amp: 0.9 }, u: { F: [300, 870, 2240, 3300], amp: 0.9 },
        v: { F: [250, 1900, 2400, 3300], amp: 0.9 },
        m: { F: [280, 1000, 2200, 3300], amp: 0.22, bw: 2.2 }, n: { F: [280, 1450, 2500, 3300], amp: 0.22, bw: 2.2 }, l: { F: [360, 1300, 2700, 3300], amp: 0.55, bw: 1.4 },
        h: { asp: true },
    };
    const PHONE_IPA = { a: 'a', o: 'o', e: 'ɛ', E: 'ɤ', i: 'i', u: 'u', v: 'y', m: 'm', n: 'n', l: 'l', h: 'x' };
    /** 字 → 拼音（声母 + 韵母的元音目标序列 + 声调）。韵母写成目标音素串：ai = a→i，ao = a→u，iao = i→a→u */
    const LEXICON = {
        妈: ['m', 'a', 1], 麻: ['m', 'a', 2], 马: ['m', 'a', 3], 骂: ['m', 'a', 4],
        你: ['n', 'i', 3], 好: ['h', 'ao', 3], 我: ['', 'uo', 3], 爱: ['', 'ai', 4],
        啊: ['', 'a', 1], 诶: ['', 'e', 1], 衣: ['', 'i', 1], 喔: ['', 'o', 1], 乌: ['', 'u', 1],
        猫: ['m', 'ao', 1], 鱼: ['', 'v', 2], 来: ['l', 'ai', 2], 喵: ['m', 'iao', 1],
    };
    const PINYIN = { a: 'a', ao: 'ao', ai: 'ai', uo: 'uo', e: 'ê', i: 'i', o: 'o', u: 'u', v: 'ü', iao: 'iao' };
    const TONE_MARK = { a: ['ā', 'á', 'ǎ', 'à'], o: ['ō', 'ó', 'ǒ', 'ò'], e: ['ē', 'é', 'ě', 'è'], ê: ['ê̄', 'ế', 'ê̌', 'ề'], i: ['ī', 'í', 'ǐ', 'ì'], u: ['ū', 'ú', 'ǔ', 'ù'], ü: ['ǖ', 'ǘ', 'ǚ', 'ǜ'] };
    /** 声调的五度标记（赵元任）：1 = 阴平 55，2 = 阳平 35，3 = 上声 214，4 = 去声 51 */
    const TONES = { 1: [5, 5], 2: [3, 5], 3: [2, 1, 4], 4: [5, 1], 0: [3, 3] };
    function pinyinOf(syl) {
        const [ini, fin, tone] = syl, py = PINYIN[fin] || fin, mark = (s) => { const v = ['a', 'o', 'ê', 'e', 'i', 'u', 'ü'].find((c) => s.includes(c) && !(c === 'i' && s.includes('iu') && false)); const vv = s.includes('a') ? 'a' : s.includes('o') ? 'o' : v; return vv && tone >= 1 && tone <= 4 ? s.replace(vv, TONE_MARK[vv][tone - 1]) : s; };
        const base = ini === '' && fin === 'uo' ? 'wo' : ini === '' && fin === 'i' ? 'yi' : ini === '' && fin === 'u' ? 'wu' : ini === '' && fin === 'v' ? 'yu' : ini === '' && fin === 'ai' ? 'ai' : ini + py;
        return mark(base.replace('ü', ini === '' ? 'u' : 'ü'));
    }
    /**
     * 文本 → 音素时长表 + 帧级声学参数（每 5 ms 一帧）。o: { f0（基频中值 Hz）, rate（语速倍数）, vtl（声道缩放）}
     * 规则：上声连读变调（3 + 3 → 2 + 3）；声调 = 五度值线性插值；句末延长 15%；整句基频缓降 8%（declination）。
     */
    function ttsFrontend(text, o) {
        o = o || {};
        const base = o.f0 || 120, rate = o.rate || 1, vtl = o.vtl || 1;
        const syls = Array.from(text).map((ch) => (LEXICON[ch] ? { ch, s: LEXICON[ch].slice() } : null)).filter(Boolean);
        syls.forEach((q, i) => { q.tone0 = q.s[2]; if (q.s[2] === 3 && syls[i + 1] && syls[i + 1].s[2] === 3) q.s[2] = 2; });   // 上声变调
        const dt = 0.005, segs = [], frames = [];
        let t = 0.06;   // 开头留 60 ms 静音
        const lv = (l) => base * Math.pow(2, (l - 3) * 0.25);
        syls.forEach((q, si) => {
            const last = si === syls.length - 1, sc = (last ? 1.15 : 1) / rate, [ini, fin, tone] = q.s;
            const dIni = ini ? (ini === 'h' ? 0.085 : 0.075) * sc : 0, dFin = 0.26 * sc * (fin.length > 1 ? 1.15 : 1);
            if (ini) segs.push({ ph: ini, t0: t, t1: t + dIni, si });
            const tf = t + dIni;
            const tgs = fin.split('');
            tgs.forEach((ph, k) => { const a = tf + dFin * k / tgs.length, b = tf + dFin * (k + 1) / tgs.length; segs.push({ ph, t0: a, t1: b, si }); });
            q.t0 = t; q.t1 = tf + dFin; q.tf = tf;
            t = tf + dFin + 0.025 / rate;
        });
        const total = t + 0.06, nf = Math.ceil(total / dt) + 1;
        const phAt = (tt) => segs.find((s) => tt >= s.t0 && tt < s.t1);
        for (let f = 0; f < nf; f++) {
            const tt = f * dt, s = phAt(tt);
            const q = s ? syls[s.si] : syls.find((z) => tt >= z.t0 && tt < z.t1 + 0.025 / rate) || null;
            // F0：当前音节的声调曲线（在韵母上展开）× 整句缓降
            let f0 = base * 0.9;
            if (q) { const c = TONES[q.s[2]], u = PL_clamp((tt - q.tf) / Math.max(1e-3, q.t1 - q.tf), 0, 1), x = u * (c.length - 1), i0 = Math.min(c.length - 2, Math.floor(x)); f0 = lv(c[i0] + (c[i0 + 1] - c[i0]) * (x - i0)); }
            f0 *= 1 - 0.08 * tt / total;
            let F, bwk = 1, av = 0, ah = 0;
            if (!s) { const nx = segs.find((z) => z.t0 > tt) || segs[segs.length - 1]; F = PHONES[nx.ph].F || PHONES.a.F; }
            else if (PHONES[s.ph].asp) { const nx = segs.find((z) => z.si === s.si && z.t0 >= s.t1 - 1e-9) || s; F = PHONES[nx.ph].F; ah = 0.9; av = 0; }
            else {
                // 共振峰：音素中段保持目标值，边界 ±30 ms 内向相邻音素线性过渡（协同发音）
                const P = PHONES[s.ph], i = segs.indexOf(s), prev = segs[i - 1], next = segs[i + 1], tr = 0.03;
                F = P.F.slice(); bwk = P.bw || 1; av = P.amp;
                const blend = (other, w) => { if (!other || PHONES[other.ph].asp) return; const Q = PHONES[other.ph]; for (let k = 0; k < 4; k++) F[k] += (Q.F[k] - F[k]) * w; };
                if (next && next.t0 - tt < tr && next.si === s.si) blend(next, 0.5 * (1 - (next.t0 - tt) / tr));
                if (prev && tt - s.t0 < tr && prev.si === s.si) blend(prev, 0.5 * (1 - (tt - s.t0) / tr));
            }
            // 音节包络：起音 / 收音各 20 ms，音节之间的短停顿里浊音为 0
            const inSyl = q && tt >= q.t0 && tt < q.t1;
            if (!inSyl) { av = 0; ah = 0; } else { const e = Math.min(1, (tt - q.t0) / 0.02, (q.t1 - tt) / 0.03); av *= e; ah *= e; }
            frames.push({ f0, F: F.map((x, k) => Math.min(FMAX_TTS[k], x * vtl)), BW: VBW.map((b) => b * bwk), av, ah });
        }
        return { syls, segs, frames, dt, total: nf * dt };
    }
    const PL_clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
    // TTS 前端里每个共振峰各自的上限：童声（声道 ×1.3）的 /i/ 本来 F3 = 3913、F4 = 4550 Hz，都截到 3850 的话两个谐振器叠在同一个频率，/i/ 会比别的元音响 20 多 dB
    const FMAX_TTS = [1300, 3000, 3450, 3850];
    function tts(text, o, rand) { const fe = ttsFrontend(text, o); const x = formantSynth(fe.frames, rand || LM.rng(7), { jitter: 0.008 }); return Object.assign(fe, { x }); }

    // ═════════════════════════════ ASR 特征：log-mel（Whisper 的归一化：log10、截到最大值 −8、(x + 4)/4） ═════════════════════════════
    const ASR = { N: 256, H: 80, NM: 32, DUR: 0.32 };
    ASR.L = Math.round(ASR.DUR * SR); ASR.T = 1 + Math.floor(ASR.L / ASR.H);
    let asrBank = null;
    /** 一段波形 → [NM × T] 的 log-mel（通道优先：第 m 个 mel 通道是一行，正好是 1D 卷积的输入布局 C × W） */
    function asrFeatures(x, out, off) {
        if (!asrBank) asrBank = melBank(ASR.N, ASR.NM);
        const sp = stft(x, ASR.N, ASR.H), T = sp.T, ms = melSpec(sp, asrBank), NM = ASR.NM;
        let mx = -Infinity; const lg = new Float64Array(T * NM);
        for (let i = 0; i < T * NM; i++) { lg[i] = Math.log10(Math.max(ms[i], 1e-10)); if (lg[i] > mx) mx = lg[i]; }
        out = out || new Float32Array(NM * T); off = off || 0;
        for (let t = 0; t < T; t++) for (let m = 0; m < NM; m++) out[off + m * T + t] = (Math.max(lg[t * NM + m], mx - 8) + 4) / 4;
        return out;
    }
    /** 随机说话人：基频 85–280 Hz（对数均匀：男声 → 女声 → 童声），声道缩放 0.85–1.25，各共振峰再抖 ±4%，信噪比 15–35 dB */
    function randomSpeaker(rand) {
        return { f0: 85 * Math.pow(280 / 85, rand()), vtl: 0.85 + 0.4 * rand(), fj: [0, 1, 2, 3].map(() => 0.08 * (rand() - 0.5)), bw: 0.8 + 0.5 * rand(), vib: TAU * rand(), jit: 0.005 + 0.015 * rand(), snr: 15 + 20 * rand(), gain: 0.25 + 0.75 * rand() };
    }
    /** n 条元音样本（类别循环、打乱）：X [n × NM·T]、y、说话人 */
    function asrDataset(n, seed) {
        const rand = LM.rng(seed), D = ASR.NM * ASR.T, X = new Float32Array(n * D), y = new Int32Array(n), spk = [];
        for (let i = 0; i < n; i++) y[i] = i % 5;
        LM.shuffle(y, rand);
        for (let i = 0; i < n; i++) {
            const s = randomSpeaker(rand), x = vowelClip(VOWELS[y[i]], s, ASR.DUR, rand);
            let p = 0; for (const v of x) p += v * v; p /= x.length;
            const g = s.gain / Math.sqrt(p * 40 + 1e-12), ns = Math.sqrt(p * g * g / Math.pow(10, s.snr / 10));
            const z = new Float64Array(ASR.L); for (let j = 0; j < ASR.L; j++) z[j] = (x[j] || 0) * g + ns * LM.randn(rand);
            asrFeatures(z, X, i * D); spk.push(s);
        }
        return { X, y, n, D, spk };
    }
    /** Whisper 式前端（缩小版）：Conv1d(k3) → GELU → Conv1d(k3, stride 2) → GELU → 时间平均 → Linear → 5 类 */
    function asrCreate(seed, C1) {
        const rand = LM.rng(seed), NM = ASR.NM, T = ASR.T, T2 = Math.floor((T + 2 - 3) / 2) + 1;
        C1 = C1 || 32;
        const P = (name, r, c, init) => { const t = AG.param(name, r, c, init); return t; };
        const m = {
            NM, T, T2, C1,
            W1: P('conv1.W', C1, NM * 3, LM.init.he(rand, NM * 3)), b1: P('conv1.b', 1, C1),
            W2: P('conv2.W', C1, C1 * 3, LM.init.he(rand, C1 * 3)), b2: P('conv2.b', 1, C1),
            Wc: P('fc.W', C1, 5, LM.init.xavier(rand, C1, 5)), bc: P('fc.b', 1, 5),
        };
        const pool = new Float32Array(C1 * T2 * C1); for (let c = 0; c < C1; c++) for (let w = 0; w < T2; w++) pool[(c * T2 + w) * C1 + c] = 1 / T2;
        m.pool = AG.tensor(C1 * T2, C1, pool);
        m.params = [m.W1, m.b1, m.W2, m.b2, m.Wc, m.bc];
        return m;
    }
    function asrForward(m, X, n) {
        const x = AG.tensor(n, m.NM * m.T, X);
        const h1 = AG.gelu(AG.conv2d(x, m.W1, m.b1, { C: m.NM, H: 1, W: m.T, KH: 1, KW: 3, padH: 0, padW: 1 }));
        const h2 = AG.gelu(AG.conv2d(h1, m.W2, m.b2, { C: m.C1, H: 1, W: m.T, KH: 1, KW: 3, stride: 2, padH: 0, padW: 1 }));
        const pooled = AG.matmul(h2, m.pool);
        return { h1, h2, pooled, logits: AG.linear(pooled, m.Wc, m.bc) };
    }

    // ═════════════════════════════ mel 频谱上的扩散（AudioLDM 的缩小版：没有 VAE，直接在 16 × 16 的 log-mel 上做 DDPM） ═════════════════════════════
    const DIF = { N: 512, H: 256, NM: 16, TF: 16, TSTEPS: 100, DBR: 60, K: 16 };
    DIF.L = (DIF.TF - 1) * DIF.H; DIF.D = DIF.NM * DIF.TF;
    let difBank = null;
    /** 一个随机音符：音高连续（不限于音阶）、时长、起点、音色亮度都随机 */
    function difNote(rand) {
        const x = new Float64Array(DIF.L), f0 = 180 * Math.pow(720 / 180, rand()), dur = 0.12 + 0.3 * rand(), on = Math.floor(rand() * 0.12 * SR);
        const n = Math.min(DIF.L - on, Math.round(dur * SR)), tb = TIMBRES[rand() < 0.5 ? 0 : 1];
        addNote(x, on, n, f0, tb, 0.3);
        return { x, f0, dur, on, tb: tb.name };
    }
    /** 波形 → [NM × TF] 的 log-mel，映射到 [−1, 1]（相对固定参考电平：0 dB → +1，−60 dB → −1） */
    function difMel(x, out, off) {
        if (!difBank) difBank = melBank(DIF.N, DIF.NM);
        const sp = stft(x, DIF.N, DIF.H), ms = melSpec(sp, difBank), T = sp.T;
        out = out || new Float32Array(DIF.D); off = off || 0;
        for (let t = 0; t < DIF.TF; t++) for (let m = 0; m < DIF.NM; m++) { const v = t < T ? 10 * Math.log10(ms[t * DIF.NM + m] + 1e-12) : -200; out[off + m * DIF.TF + t] = PL_clamp((v - DIF.REF) / (DIF.DBR / 2) + 1, -1, 1); }
        return out;
    }
    DIF.REF = 18;   // 参考电平（dB）：训练集里最响的 mel 格大约在这里
    function difDataset(n, seed) { const r = LM.rng(seed), X = new Float32Array(n * DIF.D), meta = []; for (let i = 0; i < n; i++) { const nt = difNote(r); difMel(nt.x, X, i * DIF.D); meta.push(nt); } return { X, n, meta }; }
    /** [−1, 1] 的 log-mel 图 → mel 能量 → 线性功率谱（非负最小二乘）→ Griffin-Lim 32 次 → 波形（这一步就是「声码器」） */
    function difToWave(img, iters) {
        if (!difBank) difBank = melBank(DIF.N, DIF.NM);
        const T = DIF.TF, mel = new Float64Array(T * DIF.NM);
        for (let t = 0; t < T; t++) for (let m = 0; m < DIF.NM; m++) { const v = img[m * T + t]; mel[t * DIF.NM + m] = v <= -0.999 ? 0 : Math.pow(10, ((v - 1) * DIF.DBR / 2 + DIF.REF) / 10); }
        const P = melInverse(mel, T, difBank, 30), A = new Float64Array(P.length);
        for (let i = 0; i < P.length; i++) A[i] = Math.sqrt(P[i]);
        const G = glCreate(A, T, DIF.N, DIF.H, DIF.L, { seed: 5, momentum: 0.99 });
        for (let k = 0; k < (iters || 32); k++) glStep(G);
        return G.x;
    }
    /**
     * 潜空间编码器（AudioLDM 用 VAE；这里用 PCA = 线性自编码器，第 09 篇第 1 步）：前 k 个主成分，每一维再除以 √λ 白化成单位方差。
     * 子空间迭代（块幂法 + Gram-Schmidt）求协方差矩阵的前 k 个特征向量。
     */
    /** 分块的 PCA（每帧只做一小块，避免卡顿）：pcaBegin → pcaAccum（累加协方差）→ pcaIterate（子空间迭代）→ pcaFinish */
    function pcaBegin(X, n, D, k) { const mu = new Float64Array(D); for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) mu[j] += X[i * D + j] / n; const rand = LM.rng(11), U = new Float64Array(k * D); for (let i = 0; i < k * D; i++) U[i] = rand() - 0.5; return { X, n, D, k, mu, Cv: new Float64Array(D * D), pos: 0, it: 0, U, W: new Float64Array(k * D) }; }
    function pcaAccum(P, count) {
        const { X, n, D, mu, Cv } = P, xc = new Float64Array(D), i1 = Math.min(n, P.pos + count);
        for (let i = P.pos; i < i1; i++) { for (let j = 0; j < D; j++) xc[j] = X[i * D + j] - mu[j]; for (let a = 0; a < D; a++) { const v = xc[a] / n; if (v === 0) continue; for (let b = 0; b < D; b++) Cv[a * D + b] += v * xc[b]; } }
        P.pos = i1; return P.pos >= n;
    }
    function gramSchmidt(A, k, D) { for (let c = 0; c < k; c++) { for (let p = 0; p < c; p++) { let d = 0; for (let j = 0; j < D; j++) d += A[c * D + j] * A[p * D + j]; for (let j = 0; j < D; j++) A[c * D + j] -= d * A[p * D + j]; } let nr = 0; for (let j = 0; j < D; j++) nr += A[c * D + j] ** 2; nr = Math.sqrt(nr) || 1; for (let j = 0; j < D; j++) A[c * D + j] /= nr; } }
    function pcaIterate(P, count) {
        const { D, k, Cv, U, W } = P;
        if (P.it === 0) gramSchmidt(U, k, D);
        for (let r = 0; r < count; r++) { W.fill(0); for (let c = 0; c < k; c++) for (let a = 0; a < D; a++) { let sm = 0; for (let b = 0; b < D; b++) sm += Cv[a * D + b] * U[c * D + b]; W[c * D + a] = sm; } gramSchmidt(W, k, D); U.set(W); P.it++; }
    }
    function pcaFinish(P) {
        const { D, k, Cv, U, mu } = P, lam = new Float64Array(k); let tr = 0; for (let a = 0; a < D; a++) tr += Cv[a * D + a];
        for (let c = 0; c < k; c++) { let sm = 0; for (let a = 0; a < D; a++) { let r = 0; for (let b = 0; b < D; b++) r += Cv[a * D + b] * U[c * D + b]; sm += U[c * D + a] * r; } lam[c] = sm; }
        return { mu, U: Float64Array.from(U), lam, k, D, tr, explained: lam.reduce((a, b) => a + b, 0) / tr };
    }
    function difPCA(X, n, D, k, iters) { const P = pcaBegin(X, n, D, k); pcaAccum(P, n); pcaIterate(P, iters || 60); return pcaFinish(P); }
    function difEncode(pca, X, n) { const { k, D, mu, U, lam } = pca, Z = new Float32Array(n * k); for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) { let sm = 0; for (let j = 0; j < D; j++) sm += (X[i * D + j] - mu[j]) * U[c * D + j]; Z[i * k + c] = sm / Math.sqrt(lam[c]); } return Z; }
    function difDecode(pca, Z, n) { const { k, D, mu, U, lam } = pca, X = new Float32Array(n * D); for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) { let v = mu[j]; for (let c = 0; c < k; c++) v += Z[i * k + c] * Math.sqrt(lam[c]) * U[c * D + j]; X[i * D + j] = PL_clamp(v, -1, 1); } return X; }
    /** 余弦噪声日程（Nichol & Dhariwal 2021）：ᾱ_t = cos²((t/T + s)/(1 + s) · π/2) */
    function difSchedule(Tn) {
        const s = 0.008, f = (t) => Math.cos((t / Tn + s) / (1 + s) * Math.PI / 2) ** 2, ab = new Float64Array(Tn + 1);
        for (let t = 0; t <= Tn; t++) ab[t] = f(t) / f(0);
        ab[Tn] = Math.max(ab[Tn], 1e-5);
        return ab;
    }
    /** 去噪网络：[x_t, 时间步的正弦 embedding] → MLP（SiLU）→ 预测噪声 ε */
    function difCreate(seed, hid, D) {
        const rand = LM.rng(seed), E = 32; hid = hid || 128; D = D || DIF.K;
        const P = (name, r, c, init, decay) => { const t = AG.param(name, r, c, init); t.decay = !!decay; return t; };
        const m = { D, E, hid, W1: P('W1', D + E, hid, LM.init.he(rand, D + E), true), b1: P('b1', 1, hid), W2: P('W2', hid, hid, LM.init.he(rand, hid), true), b2: P('b2', 1, hid), W3: P('W3', hid, D, LM.init.normal(rand, 0.01), true), b3: P('b3', 1, D) };
        m.params = [m.W1, m.b1, m.W2, m.b2, m.W3, m.b3];
        m.ab = difSchedule(DIF.TSTEPS);
        return m;
    }
    function timeEmb(ts, E) { const n = ts.length, out = new Float32Array(n * E); for (let i = 0; i < n; i++) for (let j = 0; j < E / 2; j++) { const fr = Math.exp(-Math.log(1000) * j / (E / 2)), a = ts[i] * fr; out[i * E + j] = Math.sin(a); out[i * E + E / 2 + j] = Math.cos(a); } return out; }
    function difForward(m, Xt, ts, n) {
        const h0 = AG.concatCols(AG.tensor(n, m.D, Xt), AG.tensor(n, m.E, timeEmb(ts, m.E)));
        const h1 = AG.silu(AG.linear(h0, m.W1, m.b1)), h2 = AG.silu(AG.linear(h1, m.W2, m.b2));
        return AG.linear(h2, m.W3, m.b3);
    }
    /** 一步训练：随机 t、随机噪声 ε，x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε；网络预测 v = √ᾱ_t·ε − √(1−ᾱ_t)·x₀（v-prediction），损失 = ‖v̂ − v‖² 的均值 */
    function difTrainStep(m, opt, X0, nData, B, rand, lr) {
        const D = m.D, Xt = new Float32Array(B * D), V = new Float32Array(B * D), ts = new Float32Array(B);
        for (let b = 0; b < B; b++) {
            const i = Math.floor(rand() * nData), t = 1 + Math.floor(rand() * DIF.TSTEPS), a = Math.sqrt(m.ab[t]), s = Math.sqrt(1 - m.ab[t]);
            ts[b] = t;
            for (let j = 0; j < D; j++) { const e = LM.randn(rand), x0 = X0[i * D + j]; Xt[b * D + j] = a * x0 + s * e; V[b * D + j] = a * e - s * x0; }
        }
        LM.zeroGrad(m.params);
        const loss = AG.mse(difForward(m, Xt, ts, B), AG.tensor(B, D, V));
        AG.backward(loss);
        LM.clipGradNorm(m.params, 1.0);
        opt.step(lr);
        return loss.item();
    }
    /** DDIM（η = 0，确定性）从纯噪声走 nSteps 步回到 x₀；返回每个样本最后的 x₀（以及中间若干步的 x̂₀ 用于画图） */
    function difSample(m, n, nSteps, rand, keep) {
        const D = m.D, ab = m.ab, x = new Float32Array(n * D), ts = new Float32Array(n), snaps = [];
        for (let i = 0; i < n * D; i++) x[i] = LM.randn(rand);
        const seq = []; for (let k = 0; k < nSteps; k++) seq.push(Math.round(DIF.TSTEPS * (1 - k / nSteps)));
        AG.noGrad(() => {
            seq.forEach((t, k) => {
                const tp = k + 1 < nSteps ? seq[k + 1] : 0;
                ts.fill(t);
                const vh = difForward(m, x, ts, n).data, a = ab[t], ap = ab[tp], sa = Math.sqrt(a), s1 = Math.sqrt(1 - a);
                const x0 = new Float32Array(n * D);
                for (let i = 0; i < n * D; i++) {
                    const v0 = PL_clamp(sa * x[i] - s1 * vh[i], -4, 4), e = (x[i] - sa * v0) / s1;   // x̂₀ = √ᾱ·x_t − √(1−ᾱ)·v̂，ε̂ 由 x̂₀ 反推
                    x0[i] = v0; x[i] = Math.sqrt(ap) * v0 + Math.sqrt(1 - ap) * e;
                }
                if (keep && keep.includes(k)) snaps.push({ k, t, x: Float32Array.from(x), x0 });
            });
        });
        return { x, snaps };
    }

    // ═════════════════════════════ 第 01 步：采样、混叠、量化 ═════════════════════════════
    /** 采样后听到的频率：把 f 折回 [0, fs/2]（f_a = |f − fs·round(f/fs)|） */
    const aliasFreq = (f, fs) => { const r = ((f % fs) + fs) % fs; return r > fs / 2 ? fs - r : r; };
    /** 理想重建（Whittaker–Shannon）：x(t) = Σ_k x[k]·sinc(fs·t − k)，截断到 ±taps 个样本并加 Hann 窗。xs[i] 是第 k0 + i 个样本 */
    function sincAt(xs, k0, fs, t, taps) {
        const u = fs * t, c = Math.floor(u); let s = 0;
        for (let k = c - taps + 1; k <= c + taps; k++) { const i = k - k0; if (i < 0 || i >= xs.length) continue; const d = u - k, w = 0.5 + 0.5 * Math.cos(Math.PI * d / taps); s += xs[i] * (d === 0 ? 1 : Math.sin(Math.PI * d) / (Math.PI * d)) * w; }
        return s;
    }
    /** 任意采样率的信号 → 目标采样率（带限 sinc 插值，就是 DAC 的重建滤波器；降采样时先低通到新的奈奎斯特）。只用于播放 */
    function resample(x, fsIn, fsOut, taps) {
        taps = taps || 16; const n = Math.floor(x.length * fsOut / fsIn), y = new Float64Array(n), lp = Math.min(1, fsOut / fsIn);
        for (let j = 0; j < n; j++) {
            const u = j * fsIn / fsOut, c = Math.floor(u); let s = 0;
            for (let k = c - taps + 1; k <= c + taps; k++) { if (k < 0 || k >= x.length) continue; const e = u - k, d = e * lp, w = 0.5 + 0.5 * Math.cos(Math.PI * e / taps); s += x[k] * lp * (d === 0 ? 1 : Math.sin(Math.PI * d) / (Math.PI * d)) * w; }
            y[j] = s;
        }
        return y;
    }
    /** 量化：均匀 b bit（round 到 2^(b−1) 级），或 μ-law（μ = 255：先压扩 y = sgn(x)·ln(1 + μ|x|)/ln(1 + μ)，再均匀量化，再展开） */
    function quantize(x, bits, mu) {
        const q = Math.pow(2, bits - 1), y = new Float64Array(x.length), MU = 255, lmu = Math.log1p(MU);
        for (let i = 0; i < x.length; i++) {
            let v = PL_clamp(x[i], -1, 1);
            if (mu) v = Math.sign(v) * Math.log1p(MU * Math.abs(v)) / lmu;
            v = PL_clamp(Math.round(v * q), -q, q - 1) / q;
            if (mu) v = Math.sign(v) * (Math.pow(1 + MU, Math.abs(v)) - 1) / MU;
            y[i] = v;
        }
        return y;
    }
    function snrDb(x, y, i0, i1) { i0 = i0 || 0; i1 = i1 ?? x.length; let s = 0, e = 0; for (let i = i0; i < i1; i++) { s += x[i] * x[i]; e += (x[i] - y[i]) ** 2; } return 10 * Math.log10((s + 1e-30) / (e + 1e-30)); }
    /** 采样率 fs 下，对 0 … fmax 的一串纯音做 FFT，读出谱峰的位置 = 实测的输出频率 */
    function measureFold(fs, fmax, n) {
        const N = 1024, w = hann(N), out = [];
        for (let i = 0; i <= n; i++) {
            const f = fmax * i / n, re = new Float64Array(N), im = new Float64Array(N);
            for (let k = 0; k < N; k++) re[k] = Math.sin(TAU * f * k / fs + 0.3) * w[k];
            fft(re, im, false);
            let b = 0, bv = -1; for (let k = 0; k <= N / 2; k++) { const p = re[k] * re[k] + im[k] * im[k]; if (p > bv) { bv = p; b = k; } }
            let d = 0;   // 抛物线插值求峰的精确位置
            if (b > 0 && b < N / 2) { const a0 = Math.log(re[b - 1] ** 2 + im[b - 1] ** 2 + 1e-30), a1 = Math.log(bv + 1e-30), a2 = Math.log(re[b + 1] ** 2 + im[b + 1] ** 2 + 1e-30); d = 0.5 * (a0 - a2) / ((a0 - 2 * a1 + a2) || 1); }
            out.push({ f, fo: (b + d) * fs / N });
        }
        return out;
    }

    // ═════════════════════════════ 演示片段（第 02–04 步）：旋律 → 啁啾 → 元音 a、i → 噪声 → 三下咔嗒 ═════════════════════════════
    function makeClip() {
        const L = Math.round(2.65 * SR), x = new Float64Array(L), rand = LM.rng(2024), segs = [];
        const put = (y, t, gain) => { const at = Math.round(t * SR); for (let i = 0; i < y.length && at + i < L; i++) x[at + i] += y[i] * gain; };
        [60, 64, 67, 72].forEach((m, k) => addNote(x, Math.round(k * 0.25 * SR), Math.round(0.25 * SR), midiHz(m), TIMBRES[0], 0.16));
        segs.push({ t0: 0, t1: 1, label: '旋律 C4 E4 G4 C5' });
        put(fadeEdges(chirpLin(200, 3800, Math.round(0.5 * SR), 0.3), 5), 1.05, 1); segs.push({ t0: 1.05, t1: 1.55, label: '啁啾' });
        const rmsN = (y, r) => { let p = 0; for (const v of y) p += v * v; const g = r / Math.sqrt(p / y.length + 1e-20); for (let i = 0; i < y.length; i++) y[i] *= g; return y; };
        put(rmsN(vowelClip(VOWELS[0], { f0: 120, vtl: 1 }, 0.3, rand), 0.13), 1.62, 1); segs.push({ t0: 1.62, t1: 1.92, label: '元音 a' });
        put(rmsN(vowelClip(VOWELS[2], { f0: 112, vtl: 1 }, 0.3, rand), 0.13), 1.92, 1); segs.push({ t0: 1.92, t1: 2.22, label: 'i' });
        const nz = new Float64Array(Math.round(0.14 * SR)); for (let i = 0; i < nz.length; i++) nz[i] = (2 * rand() - 1) * Math.exp(-i / (0.035 * SR)) * 0.4;
        put(nz, 2.27, 1); segs.push({ t0: 2.27, t1: 2.41, label: '噪声' });
        [2.47, 2.52, 2.57].forEach((t) => { const i = Math.round(t * SR); x[i] += 0.7; x[i + 1] -= 0.35; });
        segs.push({ t0: 2.45, t1: 2.6, label: '咔嗒' });
        return { x, L, segs };
    }

    // ═════════════════════════════ 第 05 / 06 步的数据：48 段训练旋律 + 6 段测试旋律（两种音色交替）→ MDCT 帧 ═════════════════════════════
    const CODEC = { M: 256, NQ: 8, B: 64, GAMMA: 0.98, STEPS: 1500, EVAL_EVERY: 10, BEAT: 2048, NTRAIN: 48, NTEST: 6, EVALF: 64 };
    CODEC.FPS = SR / CODEC.M;
    function melodySet(n, seed) {
        const r = LM.rng(seed), out = [];
        for (let i = 0; i < n; i++) { const mel = makeMelody(r, 8 + Math.floor(r() * 5)), m = renderMelody(mel, CODEC.BEAT, i % 2, 0.22, 0); m.mel = mel; m.tb = i % 2; m.mdct = mdct(m.x, CODEC.M); out.push(m); }
        return out;
    }
    function codecData() {
        const train = melodySet(CODEC.NTRAIN, 11), test = melodySet(CODEC.NTEST, 999), F = train.reduce((a, m) => a + m.mdct.F, 0), X = new Float64Array(F * CODEC.M);
        let o = 0; train.forEach((m) => { X.set(m.mdct.C, o); o += m.mdct.C.length; });
        const ev = test[0].mdct, EF = Math.min(CODEC.EVALF, ev.F);
        return { train, test, X, F, sec: train.reduce((a, m) => a + m.x.length, 0) / SR, evC: ev.C.subarray(0, EF * CODEC.M), EF };
    }
    /** 系数域的 SNR（MDCT 是正交变换：系数的误差能量 = 波形的误差能量） */
    function coefSNR(X, Xh) { let s = 0, e = 0; for (let i = 0; i < X.length; i++) { s += X[i] * X[i]; e += (X[i] - Xh[i]) ** 2; } return 10 * Math.log10(s / (e + 1e-30)); }
    function newCodecRun(K, seed) { return { K, Q: rvqCreate(CODEC.M, K, CODEC.NQ, { gamma: CODEC.GAMMA, seed }), step: 0, curve: [], evals: [], rand: LM.rng((seed * 13 + 1) >>> 0), last: null }; }
    function codecEval(R, D) {
        const codes = rvqEncode(R.Q, D.evC, D.EF, CODEC.NQ), snr = [], use = [];
        for (let nq = 1; nq <= CODEC.NQ; nq++) snr.push(coefSNR(D.evC, rvqDecode(R.Q, codes, D.EF, nq)));
        for (let l = 0; l < CODEC.NQ; l++) { const h = new Int32Array(R.K); for (let t = 0; t < D.EF; t++) h[codes[t * CODEC.NQ + l]]++; use.push(h); }
        R.last = { step: R.step, codes, snr, use };
        R.evals.push({ step: R.step, snr });
    }
    /** 每帧：要么训练一步（mini-batch EMA k-means），要么做一次评估（每 10 步一次），不在同一帧里都做 */
    function codecTick(R, D) {
        const due = R.step % CODEC.EVAL_EVERY === 0 || R.step >= CODEC.STEPS;
        if (due && (!R.last || R.last.step !== R.step)) { codecEval(R, D); return; }
        if (R.step < CODEC.STEPS) { R.curve.push(rvqTrainStep(R.Q, D.X, D.F, CODEC.B, R.rand)); R.step++; }
    }
    /** 全部测试旋律上的 SNR（按层数），Node 里核对用 */
    function codecTestSNR(Q, D, nqMax) {
        const out = [];
        for (let nq = 1; nq <= nqMax; nq++) { let s = 0, e = 0; D.test.forEach((m) => { const c = rvqEncode(Q, m.mdct.C, m.mdct.F, Q.NQ), Xh = rvqDecode(Q, c, m.mdct.F, nq); for (let i = 0; i < Xh.length; i++) { s += m.mdct.C[i] ** 2; e += (m.mdct.C[i] - Xh[i]) ** 2; } }); out.push(10 * Math.log10(s / e)); }
        return out;
    }

    // ═════════════════════════════ 第 06 步的训练：阶段 1 训练 LM 自己的编解码器（4 层）→ 阶段 2 打 token → 阶段 3 训练延迟排布 LM ═════════════════════════════
    const LMC = { K: 64, NQ: 4, CODEC_STEPS: 500, CODEC_SPF: 2, TOK_CHUNK: 160, B: 2, T: 32, STEPS: 2000, LR: 3e-3, WARM: 100, EVAL_EVERY: 100, VALB: 8, PROMPT: 8, GEN: 88 };
    function newLMRun(seed) {
        const R = { phase: 0, seed, Q: rvqCreate(CODEC.M, LMC.K, LMC.NQ, { gamma: CODEC.GAMMA, seed: seed + 5 }), crand: LM.rng(seed * 7 + 3), cstep: 0, tokPos: 0, codes: null, step: 0, loss: [], evals: [], ms: 0, batch: null };
        R.m = tlmCreate({ NQ: LMC.NQ, K: LMC.K, T: LMC.T, d: 32, L: 2, H: 4, ff: 128 }, seed);
        R.opt = new LM.AdamW(R.m.params, { lr: LMC.LR, b1: 0.9, b2: 0.95, eps: 1e-8, wd: 0.1 });
        R.rand = LM.rng((seed * 7919 + 11) >>> 0);
        return R;
    }
    function lmTick(R, D) {
        if (R.phase === 0) {   // 阶段 1：编解码器（与第 05 步同一个 EMA k-means，但只要 4 层、K = 64）
            for (let k = 0; k < LMC.CODEC_SPF && R.cstep < LMC.CODEC_STEPS; k++) { rvqTrainStep(R.Q, D.X, D.F, CODEC.B, R.crand); R.cstep++; }
            if (R.cstep >= LMC.CODEC_STEPS) { R.phase = 1; R.codes = new Int32Array(D.F * LMC.NQ); }
            return;
        }
        if (R.phase === 1) {   // 阶段 2：冻结码本，给整个训练集打 token（分几帧做完）
            const t1 = Math.min(D.F, R.tokPos + LMC.TOK_CHUNK);
            R.codes.set(rvqEncode(R.Q, D.X.subarray(R.tokPos * CODEC.M, t1 * CODEC.M), t1 - R.tokPos, LMC.NQ), R.tokPos * LMC.NQ); R.tokPos = t1;
            if (R.tokPos >= D.F) {
                R.stream = delayPattern(R.codes, D.F, LMC.NQ, LMC.K); R.P = D.F + LMC.NQ - 1;
                R.seen = new Set(); for (let t = 0; t < D.F; t++) R.seen.add(R.codes.subarray(t * LMC.NQ, t * LMC.NQ + LMC.NQ).join(','));
                R.test = D.test.map((m) => ({ codes: rvqEncode(R.Q, m.mdct.C, m.mdct.F, LMC.NQ), F: m.mdct.F, x: m.x }));
                let F = 0; R.test.forEach((t) => { F += t.F; }); const all = new Int32Array(F * LMC.NQ); let o = 0; R.test.forEach((t) => { all.set(t.codes, o); o += t.codes.length; });
                R.vb = tlmBatch(delayPattern(all, F, LMC.NQ, LMC.K), F + LMC.NQ - 1, LMC.NQ, LMC.K, LMC.VALB, LMC.T, LM.rng(77));
                R.phase = 2;
            }
            return;
        }
        const evalDue = R.step % LMC.EVAL_EVERY === 0 || R.step >= LMC.STEPS;
        if (evalDue && (!R.evals.length || R.evals[R.evals.length - 1].step !== R.step)) { R.evals.push({ step: R.step, val: tlmEval(R.m, R.vb.x, R.vb.y, LMC.VALB, LMC.T) }); return; }
        if (R.step >= LMC.STEPS) return;
        R.batch = tlmBatch(R.stream, R.P, LMC.NQ, LMC.K, LMC.B, LMC.T, R.rand, R.batch);
        const t0 = now(), r = tlmTrainStep(R.m, R.opt, R.batch.x, R.batch.y, LMC.B, LMC.T, LM.cosineLR(R.step, { maxLR: LMC.LR, minLR: LMC.LR / 10, warmup: LMC.WARM, total: LMC.STEPS }));
        const dt = now() - t0; R.ms = R.ms ? 0.9 * R.ms + 0.1 * dt : dt;
        R.loss.push(r.loss); R.lastPer = r.per; R.step++;
    }
    /** 用当前模型续写：提示 = 测试旋律的前 8 帧（第一个音符），再生成 88 帧；解码 → 波形 → 每拍的音高分析 */
    function lmGenerate(R, which, temp, seed) {
        const t = R.test[which], F0 = LMC.PROMPT, prompt = delayPattern(t.codes.subarray(0, F0 * LMC.NQ), F0, LMC.NQ, LMC.K);
        const seq = tlmGenerate(R.m, prompt, F0, LMC.GEN + LMC.NQ - 1, { temperature: temp, rand: LM.rng(seed), force: t.codes, forceF: F0 });
        const codes = undelay(seq, F0 + LMC.GEN + LMC.NQ - 1, LMC.NQ), F = codes.length / LMC.NQ;
        const valid = new Uint8Array(F); for (let f = 0; f < F; f++) valid[f] = R.seen.has(codes.subarray(f * LMC.NQ, f * LMC.NQ + LMC.NQ).join(',')) ? 1 : 0;
        const x = imdct(rvqDecode(R.Q, codes, F, LMC.NQ), F, CODEC.M, (F - 1) * CODEC.M);
        const nReal = Math.min(t.x.length, x.length), real = t.x.subarray(0, nReal), recon = imdct(rvqDecode(R.Q, t.codes, t.F, LMC.NQ), t.F, CODEC.M, t.x.length);
        return { codes, F, valid, x, real, recon, prompt: x.subarray(0, F0 * CODEC.M), pitch: pitchReport(x, CODEC.BEAT), F0 };
    }
    /** 每拍（2048 个样本）中间 1024 点做 FFT：在音阶的 8 个音里找谐波能量（按 1/h 加权）最大的那个；清晰度 = 它的谐波能量占比 */
    function pitchReport(y, beat) {
        const out = [], N = 1024, w = hann(N);
        for (let s = 0; s + beat <= y.length; s += beat) {
            const re = new Float64Array(N), im = new Float64Array(N), o = s + (beat - N) / 2;
            for (let i = 0; i < N; i++) re[i] = (y[o + i] || 0) * w[i];
            fft(re, im, false);
            const P = new Float64Array(N / 2); let tot = 0; for (let k = 1; k < N / 2; k++) { P[k] = re[k] ** 2 + im[k] ** 2; tot += P[k]; }
            let best = -1, bs = 0, bw = -1;
            SCALE.forEach((m) => { const f0 = midiHz(m); let e = 0, ew = 0; for (let h = 1; h * f0 < 3800; h++) { const kc = Math.round(h * f0 * N / SR); let eh = 0; for (let k = kc - 2; k <= kc + 2; k++) if (k > 0 && k < N / 2) eh += P[k]; e += eh; ew += eh / h; } if (ew > bw) { bw = ew; bs = e; best = m; } });
            out.push({ midi: best, clarity: tot > 1e-9 ? bs / tot : 0, energy: tot });
        }
        return out;
    }

    // ═════════════════════════════ 第 07 步的训练：阶段 1 合成音符 → 阶段 2 PCA 潜空间 → 阶段 3 潜空间扩散 ═════════════════════════════
    const DIFR = { N: 1200, CHUNK: 40, STEPS: 4000, SPF: 2, B: 32, LR: 2e-3, SAMPLE_EVERY: 250, NS: 8, DDIM: 25, HID: 128 };
    function newDifRun(seed) { return { phase: 0, seed, X: new Float32Array(DIFR.N * DIF.D), meta: [], made: 0, drand: LM.rng(seed + 31), step: 0, loss: [], samples: null, rand: LM.rng(seed * 3 + 4) }; }
    function difTick(R) {
        if (R.phase === 0) { for (let k = 0; k < DIFR.CHUNK && R.made < DIFR.N; k++) { const nt = difNote(R.drand); difMel(nt.x, R.X, R.made * DIF.D); R.meta.push({ f0: nt.f0, dur: nt.dur, tb: nt.tb }); R.made++; } if (R.made >= DIFR.N) R.phase = 1; return; }
        if (R.phase === 1) {   // PCA 分块：先累加协方差（每帧 200 条），再做 60 次子空间迭代（每帧 10 次）
            if (!R.P) R.P = pcaBegin(R.X, DIFR.N, DIF.D, DIF.K);
            if (R.P.pos < DIFR.N) { pcaAccum(R.P, 200); return; }
            if (R.P.it < 60) { pcaIterate(R.P, 10); return; }
            R.pca = pcaFinish(R.P); R.P = null; R.Z = difEncode(R.pca, R.X, DIFR.N);
            R.m = difCreate(R.seed, DIFR.HID, DIF.K); R.opt = new LM.AdamW(R.m.params, { lr: DIFR.LR, b1: 0.9, b2: 0.999, wd: 0 });
            R.phase = 2; return;
        }
        const due = R.step % DIFR.SAMPLE_EVERY === 0 || R.step >= DIFR.STEPS;
        if (due && (!R.samples || R.samples.step !== R.step)) { difSampleNow(R); return; }
        if (R.step >= DIFR.STEPS) return;
        for (let k = 0; k < DIFR.SPF; k++) { R.loss.push(difTrainStep(R.m, R.opt, R.Z, DIFR.N, DIFR.B, R.rand, LM.cosineLR(R.step, { maxLR: DIFR.LR, minLR: DIFR.LR / 10, warmup: 100, total: DIFR.STEPS }))); R.step++; }
    }
    function difSampleNow(R) {
        const keep = [0, 3, 6, 9, 12, 15, 18, 21, 24], s = difSample(R.m, DIFR.NS, DIFR.DDIM, LM.rng(R.seed + 1000), keep);
        const imgs = difDecode(R.pca, s.x, DIFR.NS), traj = s.snaps.map((q) => difDecode(R.pca, q.x0.subarray(0, DIF.K), 1)), D = DIF.D, near = [];
        for (let i = 0; i < DIFR.NS; i++) { let best = Infinity, bi = 0; for (let j = 0; j < DIFR.N; j++) { let d = 0; for (let q = 0; q < D; q++) d += (imgs[i * D + q] - R.X[j * D + q]) ** 2; if (d < best) { best = d; bi = j; } } near.push({ i: bi, d: Math.sqrt(best / D) }); }
        R.samples = { step: R.step, z: s.x, imgs, traj, trajT: s.snaps.map((q) => q.t), near, waves: [] };
    }

    // ═════════════════════════════ 第 09 步的训练：阶段 1 合成元音数据（600 训练 + 250 测试，不同说话人）→ 阶段 2 训练卷积分类器 ═════════════════════════════
    const ASRR = { NTR: 600, NTE: 250, CHUNK: 30, B: 16, C1: 16, STEPS: 900, LR: 3e-3, EVAL_EVERY: 50, EVAL_CHUNK: 50 };
    function newAsrRun(seed) {
        const D = ASR.NM * ASR.T;
        return { phase: 0, seed, made: 0, step: 0, loss: [], evals: [], ev: null, rand: LM.rng(seed * 5 + 2), D,
            tr: { X: new Float32Array(ASRR.NTR * D), y: new Int32Array(ASRR.NTR), spk: [], n: ASRR.NTR, D }, te: { X: new Float32Array(ASRR.NTE * D), y: new Int32Array(ASRR.NTE), spk: [], n: ASRR.NTE, D },
            trR: LM.rng(seed + 21), teR: LM.rng(seed + 4242) };
    }
    function asrSample(ds, i, rand, cls) {
        const s = randomSpeaker(rand), x = vowelClip(VOWELS[cls], s, ASR.DUR, rand), D = ds.D;
        let p = 0; for (const v of x) p += v * v; p /= x.length;
        const g = s.gain / Math.sqrt(p * 40 + 1e-12), ns = Math.sqrt(p * g * g / Math.pow(10, s.snr / 10));
        const z = new Float64Array(ASR.L); for (let j = 0; j < ASR.L; j++) z[j] = (x[j] || 0) * g + ns * LM.randn(rand);
        asrFeatures(z, ds.X, i * D); ds.y[i] = cls; ds.spk[i] = s;
    }
    function asrTick(R) {
        if (R.phase === 0) {   // 数据分块合成（训练集、测试集用不同的随机种子 = 不同的说话人）
            for (let k = 0; k < ASRR.CHUNK && R.made < ASRR.NTR + ASRR.NTE; k++, R.made++) { const isTr = R.made < ASRR.NTR, i = isTr ? R.made : R.made - ASRR.NTR; asrSample(isTr ? R.tr : R.te, i, isTr ? R.trR : R.teR, i % 5); }
            if (R.made >= ASRR.NTR + ASRR.NTE) { R.m = asrCreate(R.seed, ASRR.C1); R.opt = new LM.AdamW(R.m.params, { lr: ASRR.LR, b1: 0.9, b2: 0.999, wd: 0 }); R.Xb = new Float32Array(ASRR.B * R.D); R.yb = new Int32Array(ASRR.B); R.phase = 1; }
            return;
        }
        if (R.ev) {   // 评估分块做：每帧 50 条测试样本
            const e = R.ev, i1 = Math.min(ASRR.NTE, e.pos + ASRR.EVAL_CHUNK), D = R.D;
            const L = AG.noGrad(() => asrForward(R.m, R.te.X.subarray(e.pos * D, i1 * D), i1 - e.pos).logits.data);
            for (let i = e.pos; i < i1; i++) { const p = LM.argmax(L, (i - e.pos) * 5, 5); e.pred[i] = p; e.conf[R.te.y[i] * 5 + p]++; if (p === R.te.y[i]) e.ok++; }
            e.pos = i1;
            if (e.pos >= ASRR.NTE) { R.evals.push({ step: e.step, acc: e.ok / ASRR.NTE, conf: e.conf, pred: e.pred }); R.ev = null; }
            return;
        }
        const due = R.step % ASRR.EVAL_EVERY === 0 || R.step >= ASRR.STEPS;
        if (due && (!R.evals.length || R.evals[R.evals.length - 1].step !== R.step)) { R.ev = { step: R.step, pos: 0, ok: 0, conf: new Int32Array(25), pred: new Int8Array(ASRR.NTE) }; return; }
        if (R.step >= ASRR.STEPS) return;
        const D = R.D;
        for (let b = 0; b < ASRR.B; b++) { const i = Math.floor(R.rand() * ASRR.NTR); R.Xb.set(R.tr.X.subarray(i * D, (i + 1) * D), b * D); R.yb[b] = R.tr.y[i]; }
        LM.zeroGrad(R.m.params);
        const loss = AG.crossEntropy(asrForward(R.m, R.Xb, ASRR.B).logits, R.yb);
        AG.backward(loss); R.opt.step(LM.cosineLR(R.step, { maxLR: ASRR.LR, minLR: ASRR.LR / 20, warmup: 30, total: ASRR.STEPS }));
        R.loss.push(loss.item()); R.step++;
    }
    const ASR_GATE = 0.02;
    /** 用识别器扫一遍任意语音：窗长 0.32 s、每 40 ms 一个窗 → 每个窗 5 个元音的后验概率 + 中段能量 */
    function asrScan(m, x) {
        const L = ASR.L, hop = 320, nW = Math.max(1, Math.floor((x.length - L) / hop) + 1), D = ASR.NM * ASR.T, X = new Float32Array(nW * D), en = new Float64Array(nW);
        let p = 0; for (const v of x) p += v * v; p /= x.length; const g = 1 / Math.sqrt(p * 40 + 1e-12);
        for (let w = 0; w < nW; w++) { const seg = new Float64Array(L); let e = 0; for (let j = 0; j < L; j++) { const v = (x[w * hop + j] || 0) * g; seg[j] = v; if (j > L / 4 && j < 3 * L / 4) e += v * v; } en[w] = e / (L / 2); asrFeatures(seg, X, w * D); }
        const lg = AG.noGrad(() => asrForward(m, X, nW).logits.data), post = new Float32Array(nW * 5);
        for (let w = 0; w < nW; w++) LM.softmax(lg, post.subarray(w * 5, w * 5 + 5), 1, w * 5, 5);
        let emax = 0; for (const e of en) emax = Math.max(emax, e); for (let w = 0; w < nW; w++) en[w] /= emax || 1;   // 相对能量：≤ ASR_GATE（−17 dB）视为静音 / 辅音，不判
        return { nW, hop, post, en };
    }

    // ═════════════════════════════ 绘图（深色主题、设计坐标） ═════════════════════════════
    const LIVE_X = 50;   // live 预览左上角有 LIVE 徽标：标题从这里开始写
    const C = LM.C, F = LM.FONT;
    const tickFmt = (v) => { const a = Math.abs(v); if (a === 0) return '0'; if (a >= 1e4 || a < 1e-2) { const e = Math.floor(Math.log10(a) + 1e-9), mt = v / Math.pow(10, e); return (Math.abs(mt - 1) < 1e-6 ? '' : +mt.toFixed(1) + '×') + '10' + String(e).split('').map((c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]).join(''); } return String(+v.toPrecision(3)); };
    function niceTicks(lo, hi, n) { const span = hi - lo, raw = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw))), st = [1, 2, 2.5, 5, 10].map((q) => q * mag).find((s2) => s2 >= raw * 0.999); const out = []; for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-6; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : +v.toPrecision(12)); return out; }
    function logTicks(lo, hi, maxN) { const out = [], dec = []; for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++) for (const mt of [1, 2, 5]) { const v = mt * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) { out.push(v); if (mt === 1) dec.push(v); } } if (out.length <= maxN) return out; return dec.length >= 2 ? dec : out.filter((_, i) => i % Math.ceil(out.length / maxN) === 0); }
    /**
     * 曲线图（坐标轴自绘）。o: { x:[a,b], y:[a,b], logX, logY, series:[{ x?, y, color, width, dash, alpha, dots, stems, label }],
     *   refs:[{ y, label, color, dash }], marks:[{ x, label, color }], bands:[{ x0, x1, color }], legend:[{ label, color, dash, dot }], legendAt, title, tx, xLabel, yLabel, xFmt, yFmt, padL, padR }
     */
    function chart(cv, o) {
        const { ctx, w, h } = PL.draw.ctx(cv);
        const padL = o.padL ?? 44, padR = o.padR ?? 12, padT = o.title ? 24 : 10, padB = o.xLabel ? 34 : 20;
        const lx = !!o.logX, ly = !!o.logY, fx = (v) => (lx ? Math.log10(Math.max(v, 1e-300)) : v), fy = (v) => (ly ? Math.log10(Math.max(v, 1e-300)) : v);
        const ax = fx(o.x[0]), bx = fx(o.x[1]), ay = fy(o.y[0]), by = fy(o.y[1]), pw = w - padL - padR, ph = h - padT - padB;
        const X = (v) => padL + (fx(v) - ax) / ((bx - ax) || 1) * pw, Y = (v) => padT + ph - (fy(v) - ay) / ((by - ay) || 1) * ph;
        if (o.title) { ctx.font = F(11, true); ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.tx ?? 6, 5); }
        (o.bands || []).forEach((b) => { const x0 = X(Math.max(b.x0, o.x[0])), x1 = X(Math.min(b.x1, o.x[1])); if (x1 > x0) { ctx.fillStyle = b.color; ctx.fillRect(x0, padT, x1 - x0, ph); } });
        ctx.font = F(11); ctx.lineWidth = 1;
        const xt = o.xTicks || (lx ? logTicks(o.x[0], o.x[1], Math.max(2, Math.floor(pw / 50))) : niceTicks(o.x[0], o.x[1], Math.max(2, Math.floor(pw / 70))));
        const yt = o.yTicks || (ly ? logTicks(o.y[0], o.y[1], Math.max(2, Math.floor(ph / 18))) : niceTicks(o.y[0], o.y[1], Math.max(2, Math.floor(ph / 30))));
        ctx.strokeStyle = '#1b1f25'; ctx.fillStyle = '#7a808a';
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        xt.forEach((v) => { const x = X(v); if (x < padL - 1 || x > padL + pw + 1) return; ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke(); ctx.fillText((o.xFmt || tickFmt)(v), x, padT + ph + 3); });
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        yt.forEach((v) => { const y = Y(v); if (y < padT - 1 || y > padT + ph + 1) return; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke(); ctx.fillText((o.yFmt || tickFmt)(v), padL - 4, y); });
        ctx.strokeStyle = '#3a3f46'; ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + ph); ctx.lineTo(padL + pw, padT + ph); ctx.stroke();
        ctx.fillStyle = '#868c96';
        if (o.xLabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xLabel, padL + pw / 2, h - 2); }
        if (o.yLabel) { ctx.save(); ctx.translate(10, padT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
        ctx.save(); ctx.beginPath(); ctx.rect(padL, padT, pw, ph); ctx.clip();
        (o.refs || []).forEach((r) => { const y = Y(r.y); ctx.strokeStyle = r.color || C.gray; ctx.lineWidth = r.width || 1; ctx.setLineDash(r.dash || [4, 3]); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke(); });
        ctx.setLineDash([]);
        (o.series || []).forEach((s2) => {
            const n = s2.y.length; ctx.strokeStyle = s2.color; ctx.fillStyle = s2.color; ctx.lineWidth = s2.width || 1.5; ctx.globalAlpha = s2.alpha ?? 1; ctx.setLineDash(s2.dash || []);
            if (s2.stems) { const y0 = Y(Math.max(o.y[0], Math.min(o.y[1], 0))); for (let i = 0; i < n; i++) { const xv = s2.x ? s2.x[i] : i; ctx.beginPath(); ctx.moveTo(X(xv), y0); ctx.lineTo(X(xv), Y(s2.y[i])); ctx.stroke(); } }
            else if (s2.line !== false) {
                ctx.beginPath(); let pen = false;
                for (let i = 0; i < n; i++) { const xv = s2.x ? s2.x[i] : i, yv = s2.y[i]; if (!isFinite(yv) || !isFinite(xv) || (ly && yv <= 0) || (lx && xv <= 0)) { pen = false; continue; } const px = X(xv), py = Y(yv); if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; } }
                ctx.stroke();
            }
            ctx.setLineDash([]);
            if (s2.dots) for (let i = 0; i < n; i++) { const xv = s2.x ? s2.x[i] : i, yv = s2.y[i]; if (!isFinite(yv) || (ly && yv <= 0)) continue; ctx.beginPath(); ctx.arc(X(xv), Y(yv), s2.dots, 0, 7); ctx.fill(); }
            ctx.globalAlpha = 1;
        });
        (o.marks || []).forEach((m) => { const x = X(m.x); ctx.strokeStyle = m.color || C.yellow; ctx.lineWidth = 1; ctx.setLineDash(m.dash || [3, 3]); ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke(); ctx.setLineDash([]); });
        ctx.restore();
        ctx.font = F(11); ctx.textBaseline = 'top';
        (o.marks || []).forEach((m) => { if (!m.label) return; const x = X(m.x); ctx.textAlign = x > padL + pw * 0.7 ? 'right' : 'left'; const tw = ctx.measureText(m.label).width, lx0 = ctx.textAlign === 'right' ? x - 4 - tw : x + 4; ctx.fillStyle = 'rgba(5,7,10,0.78)'; ctx.fillRect(lx0 - 2, padT + 2 + (m.row || 0) * 14, tw + 4, 14); ctx.fillStyle = m.color || C.yellow; ctx.fillText(m.label, x + (ctx.textAlign === 'right' ? -4 : 4), padT + 3 + (m.row || 0) * 14); });
        (o.refs || []).forEach((r) => { if (!r.label) return; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; const tw = ctx.measureText(r.label).width, y = Y(r.y); ctx.fillStyle = 'rgba(5,7,10,0.7)'; ctx.fillRect(padL + pw - tw - 6, y - 14, tw + 4, 13); ctx.fillStyle = r.color || C.gray; ctx.fillText(r.label, padL + pw - 4, y - 1); });
        if (o.legend && o.legend.length) {
            ctx.font = F(11); const lw = Math.max(...o.legend.map((l) => ctx.measureText(l.label).width)) + 28, lh = o.legend.length * 14 + 6;
            const lx0 = /left|bl/.test(o.legendAt || '') ? padL + 4 : padL + pw - lw - 4, ly0 = /bottom|bl|br/.test(o.legendAt || '') ? padT + ph - lh - 4 : padT + 4;
            ctx.fillStyle = 'rgba(8,9,12,0.84)'; ctx.fillRect(lx0, ly0, lw, lh);
            o.legend.forEach((l, i) => {
                const y = ly0 + 10 + i * 14;
                if (l.dot) { ctx.fillStyle = l.color; ctx.beginPath(); ctx.arc(lx0 + 12, y, 3.5, 0, 7); ctx.fill(); }
                else { ctx.strokeStyle = l.color; ctx.lineWidth = 2; ctx.setLineDash(l.dash || []); ctx.beginPath(); ctx.moveTo(lx0 + 5, y); ctx.lineTo(lx0 + 20, y); ctx.stroke(); ctx.setLineDash([]); }
                ctx.fillStyle = l.color; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(l.label, lx0 + 24, y);
            });
        }
        return { ctx, X, Y, padL, padT, pw, ph, w, h };
    }
    const cmapRGB = [0, 0, 0];
    /**
     * 时频图（或任意 [W × H] 网格，列 = 时间）：data[t·H + k]（t < W，k < H，k = 0 在下面）。
     * o: { range:[lo, hi], map, title, tx, xMax（横轴最大值，如秒）, yMax（纵轴最大值，如 kHz）, xLabel, yLabel, xFmt, yFmt, segs:[{ x0, x1, label }], marks:[{ x, color, label }], padL, padR, padB, legend }
     * 返回布局 { X(x), Y(y), inv(dx, dy) → [x, y] }（存进 S 供 probe / pick 反算）
     */
    const offPool = new Map();
    function gridView(cv, data, W, H, o) { const { ctx, w, h } = PL.draw.ctx(cv); return gridPanel(ctx, { x: 0, y: 0, w, h }, data, W, H, o); }
    /** 同上，但画在 ctx 的一个矩形区域里（一张画布放几块面板） */
    function gridPanel(ctx, rc, data, W, H, o) {
        const w = rc.w, h = rc.h;
        const padL = rc.x + (o.padL ?? 40), padR = o.legend === false ? 8 : (o.padR ?? 40), padT = rc.y + (o.title ? 22 : 6) + (o.segs ? 15 : 0) + (o.padTop || 0), padB = o.padB ?? (o.xLabel ? 32 : 18);
        const pw = rc.x + w - padL - padR, ph = rc.y + h - padT - padB, lo = o.range[0], hi = o.range[1], map = PL.cmap[o.map || 'inferno'] || PL.cmap.inferno;
        const okey = W + 'x' + H; let off = offPool.get(okey); if (!off) { off = document.createElement('canvas'); off.width = W; off.height = H; offPool.set(okey, off); }
        const oc = off.getContext('2d'), img = oc.createImageData(W, H), d = img.data;
        for (let t = 0; t < W; t++) for (let k = 0; k < H; k++) {
            const v = data[t * H + k], q = (4 * ((H - 1 - k) * W + t));
            if (v !== v) { d[q] = 16; d[q + 1] = 18; d[q + 2] = 22; d[q + 3] = 255; continue; }
            map(PL_clamp((v - lo) / (hi - lo), 0, 1), cmapRGB); d[q] = cmapRGB[0]; d[q + 1] = cmapRGB[1]; d[q + 2] = cmapRGB[2]; d[q + 3] = 255;
        }
        oc.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = !!o.smooth; ctx.drawImage(off, padL, padT, pw, ph);
        const xMax = o.xMax ?? W, yMax = o.yMax ?? H, X = (x) => padL + x / xMax * pw, Y = (y) => padT + ph - y / yMax * ph;
        if (o.title) { ctx.font = F(11, true); ctx.fillStyle = o.titleColor || C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, rc.x + (o.tx ?? 6), rc.y + 5); }
        ctx.strokeStyle = '#3a3f46'; ctx.lineWidth = 1; ctx.strokeRect(padL - 0.5, padT - 0.5, pw + 1, ph + 1);
        ctx.font = F(11); ctx.fillStyle = '#7a808a';
        if (o.xTicks !== false) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; niceTicks(0, xMax, Math.max(2, Math.floor(pw / 70))).forEach((v) => { const x = X(v); if (x > padL + pw + 1) return; ctx.fillRect(x, padT + ph, 1, 3); ctx.fillText((o.xFmt || tickFmt)(v), x, padT + ph + 4); }); }
        if (o.yTicks !== false) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; (o.yTickVals || niceTicks(0, yMax, Math.max(2, Math.floor(ph / 26)))).forEach((v) => { const y = Y(v); if (y < padT - 1) return; ctx.fillRect(padL - 3, y, 3, 1); ctx.fillText((o.yFmt || tickFmt)(v), padL - 5, y); }); }
        ctx.fillStyle = '#868c96';
        if (o.xLabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xLabel, padL + pw / 2, rc.y + h - 2); }
        if (o.yLabel) { ctx.save(); ctx.translate(rc.x + 10, padT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
        if (o.segs) { ctx.font = F(11); ctx.textBaseline = 'bottom'; o.segs.forEach((sg, i) => { const x0 = X(sg.x0), x1 = X(sg.x1); ctx.fillStyle = i % 2 ? '#3d5a6c' : '#5b7f95'; ctx.fillRect(x0, padT - 4, Math.max(1, x1 - x0), 2); ctx.fillStyle = '#a9c3d3'; ctx.textAlign = 'center'; ctx.fillText(sg.label, (x0 + x1) / 2, padT - 5); }); }
        (o.marks || []).forEach((m) => { const x = X(m.x); ctx.strokeStyle = m.color || C.cyan; ctx.lineWidth = m.width || 1.5; ctx.setLineDash(m.dash || []); ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke(); ctx.setLineDash([]); if (m.label) { ctx.font = F(11); ctx.textAlign = x > padL + pw * 0.75 ? 'right' : 'left'; ctx.textBaseline = 'top'; const tw = ctx.measureText(m.label).width, lx0 = ctx.textAlign === 'right' ? x - 4 - tw : x + 4; ctx.fillStyle = 'rgba(5,7,10,0.75)'; ctx.fillRect(lx0 - 2, padT + 3, tw + 4, 14); ctx.fillStyle = m.color || C.cyan; ctx.fillText(m.label, ctx.textAlign === 'right' ? x - 4 : x + 4, padT + 4); } });
        if (o.legend !== false) {
            const lx = padL + pw + 8, lh = ph; for (let i = 0; i < lh; i++) { map(1 - i / (lh - 1), cmapRGB); ctx.fillStyle = LM.rgbStr(cmapRGB); ctx.fillRect(lx, padT + i, 7, 1.5); }
            ctx.fillStyle = C.dim; ctx.font = F(11); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.hiLabel || String(Math.round(hi)), lx + 9, padT - 1); ctx.textBaseline = 'bottom'; ctx.fillText(o.loLabel || String(Math.round(lo)), lx + 9, padT + ph + 1);
        }
        if (o.overlay) o.overlay(ctx, X, Y);
        return { X, Y, padL, padT, pw, ph, inv: (dx, dy) => [(dx - padL) / pw * xMax, (padT + ph - dy) / ph * yMax], ctx };
    }
    /** 频谱图的 dB 范围：[max − dr, max] */
    function dbRange(db, dr) { let m = -Infinity; for (const v of db) if (v > m) m = v; return [m - (dr || 80), m]; }
    /** token 矩阵：rows 层 × cols 帧，格子颜色 = 码字 id；o: { title, hiRows（高亮前几行）, cellH, labels, mark（列）, invalid（Uint8Array：1 = 训练集里没出现过的帧） } */
    function tokenView(cv, codes, cols, rows, stride, o) {
        const { ctx, w, h } = PL.draw.ctx(cv);
        const padL = o.padL ?? 46, padT = o.title ? 24 : 6, cw = (w - padL - 6) / cols, ch = o.cellH || Math.min(22, (h - padT - 22) / rows);
        if (o.title) { ctx.font = F(11, true); ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.tx ?? 6, 5); }
        for (let r = 0; r < rows; r++) {
            const y = padT + r * ch, dim = o.hiRows != null && r >= o.hiRows;
            ctx.font = F(11); ctx.fillStyle = dim ? '#4a4f56' : (o.rowColors ? o.rowColors[r] : C.gray); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(o.labels ? o.labels[r] : `q${r + 1}`, padL - 5, y + ch / 2);
            for (let t = 0; t < cols; t++) {
                const id = codes[t * stride + r]; ctx.globalAlpha = dim ? 0.22 : 1;
                ctx.fillStyle = id < 0 || id === o.pad ? '#15181d' : LM.tokColor(id * 7 + r * 61);
                ctx.fillRect(padL + t * cw, y + 0.5, Math.max(1, cw - (cw > 4 ? 0.6 : 0)), ch - 1);
                if (o.text && cw >= 14 && id >= 0 && id !== o.pad) { ctx.globalAlpha = dim ? 0.4 : 1; ctx.fillStyle = '#e8eaed'; ctx.font = F(Math.min(11, ch * 0.6)); ctx.textAlign = 'center'; ctx.fillText(String(id), padL + (t + 0.5) * cw, y + ch / 2 + 0.5); }
            }
            ctx.globalAlpha = 1;
        }
        if (o.invalid) for (let t = 0; t < cols; t++) if (o.invalid[t]) { ctx.fillStyle = C.red; ctx.fillRect(padL + t * cw, padT + rows * ch + 2, Math.max(1, cw), 3); }
        (o.marks || []).forEach((m) => { const x = padL + m.t * cw; ctx.strokeStyle = m.color || C.yellow; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, padT - 2); ctx.lineTo(x, padT + rows * ch + 2); ctx.stroke(); if (m.label) { ctx.font = F(11); ctx.fillStyle = m.color || C.yellow; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(m.label, x + 3, padT + rows * ch + 6); } });
        return { padL, padT, cw, ch, ctx, y: padT + rows * ch };
    }
    function waveMinMax(x, i0, i1, nBins) { const mn = new Float64Array(nBins), mx = new Float64Array(nBins); for (let b = 0; b < nBins; b++) { const a = Math.floor(i0 + (i1 - i0) * b / nBins), c = Math.max(a + 1, Math.floor(i0 + (i1 - i0) * (b + 1) / nBins)); let lo = Infinity, hi = -Infinity; for (let i = a; i < c; i++) { const v = x[i] || 0; if (v < lo) lo = v; if (v > hi) hi = v; } mn[b] = lo; mx[b] = hi; } return { mn, mx }; }
    /** 长波形的包络图（每个像素列画 min–max 竖线） */
    function waveView(cv, x, sr, o) {
        const { ctx, w, h } = PL.draw.ctx(cv), padL = o.padL ?? 40, padR = 10, padT = o.title ? 22 : 6, padB = 18, pw = w - padL - padR, ph = h - padT - padB, n = Math.max(1, Math.floor(pw));
        const { mn, mx } = waveMinMax(x, 0, x.length, n); let pk = o.peak || 0; if (!pk) for (const v of x) pk = Math.max(pk, Math.abs(v)); pk = pk || 1;
        if (o.title) { ctx.font = F(11, true); ctx.fillStyle = C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, o.tx ?? 6, 5); }
        const Y = (v) => padT + ph / 2 - v / pk * ph / 2 * 0.95;
        ctx.strokeStyle = '#1b1f25'; ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(padL + pw, Y(0)); ctx.stroke();
        (o.bands || []).forEach((b) => { ctx.fillStyle = b.color; ctx.fillRect(padL + b.x0 / (x.length / sr) * pw, padT, (b.x1 - b.x0) / (x.length / sr) * pw, ph); });
        ctx.fillStyle = o.color || C.blue; for (let b = 0; b < n; b++) ctx.fillRect(padL + b, Y(mx[b]), 1, Math.max(1, Y(mn[b]) - Y(mx[b])));
        ctx.font = F(11); ctx.fillStyle = '#7a808a'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        niceTicks(0, x.length / sr, Math.max(2, Math.floor(pw / 70))).forEach((v) => { const px = padL + v / (x.length / sr) * pw; ctx.fillRect(px, padT + ph, 1, 3); ctx.fillText(tickFmt(v) + (v === 0 ? ' s' : ''), px, padT + ph + 4); });
        (o.marks || []).forEach((m) => { const px = padL + m.x / (x.length / sr) * pw; ctx.strokeStyle = m.color || C.yellow; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, padT + ph); ctx.stroke(); ctx.setLineDash([]); if (m.label) { ctx.fillStyle = m.color || C.yellow; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(m.label, px + 3, padT + 2); } });
        return { padL, padT, pw, ph };
    }

    // ═════════════════════════════ 播放（WebAudio；只在用户点击「▶」时创建 AudioContext，验证器不依赖播放） ═════════════════════════════
    const AUDIO = { ctx: null, src: null, btn: null, t0: 0, dur: 0, raf: 0, label: '' };
    function stopAudio() {
        if (AUDIO.src) { try { AUDIO.src.onended = null; AUDIO.src.stop(); } catch (e) { /* 已经停了 */ } AUDIO.src = null; }
        if (AUDIO.btn) { AUDIO.btn.classList.remove('on'); AUDIO.btn.style.setProperty('--p', '0'); const ic = AUDIO.btn.querySelector('.ic'); if (ic) ic.textContent = '▶'; AUDIO.btn = null; }
        if (AUDIO.raf && typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(AUDIO.raf);
    }
    /** 播放一段波形：峰值归一化到 0.8，两端 5 ms 淡入淡出；采样率低于 8 kHz 的先用 sinc 插值升到 16 kHz（AudioBuffer 至少要支持 8 kHz） */
    function playAudio(x, sr, btn) {
        stopAudio();
        const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
        if (!AUDIO.ctx) AUDIO.ctx = new AC();
        const ac = AUDIO.ctx; if (ac.state === 'suspended') ac.resume();
        let data = x, rate = sr;
        if (sr < 8000) { data = resample(x, sr, 16000); rate = 16000; }
        let pk = 0; for (let i = 0; i < data.length; i++) pk = Math.max(pk, Math.abs(data[i]));
        const g = pk > 1e-9 ? 0.8 / pk : 0, buf = ac.createBuffer(1, Math.max(1, data.length), rate), ch = buf.getChannelData(0), nf = Math.min(data.length >> 1, Math.round(0.005 * rate));
        for (let i = 0; i < data.length; i++) ch[i] = data[i] * g * Math.min(1, i / nf, (data.length - 1 - i) / nf);
        const src = ac.createBufferSource(); src.buffer = buf; src.connect(ac.destination); src.start();
        AUDIO.src = src; AUDIO.btn = btn; AUDIO.t0 = ac.currentTime; AUDIO.dur = buf.duration;
        btn.classList.add('on'); const ic = btn.querySelector('.ic'); if (ic) ic.textContent = '■';
        src.onended = () => { if (AUDIO.src === src) stopAudio(); };
        const tick = () => { if (AUDIO.src !== src) return; btn.style.setProperty('--p', Math.min(1, (ac.currentTime - AUDIO.t0) / AUDIO.dur).toFixed(3)); AUDIO.raf = requestAnimationFrame(tick); };
        tick();
    }
    /** 声音登记表：stage 把可播放的东西写进 S.snd[key] = { x, sr } 或返回它的函数（点击时才算，例如重采样、声码器） */
    function sound(key, S) { const e = S.snd && S.snd[key]; if (!e) return null; return typeof e === 'function' ? e(S) : e; }
    if (typeof document !== 'undefined') {
        document.addEventListener('click', (ev) => {
            const b = ev.target && ev.target.closest ? ev.target.closest('[data-snd]') : null; if (!b) return;
            ev.preventDefault();
            if (AUDIO.btn === b) { stopAudio(); return; }
            const S = typeof PipelineLab !== 'undefined' ? PipelineLab.S : null, snd = S ? sound(b.getAttribute('data-snd'), S) : null;
            if (!snd || !snd.x || !snd.x.length) { b.classList.add('na'); setTimeout(() => b.classList.remove('na'), 700); return; }
            playAudio(snd.x, snd.sr || SR, b);
        });
    }
    /** 一排播放按钮的 HTML：items = [[key, 文字], …] */
    const playBar = (items, title) => `<div class="p13-bar"><span class="p13-bar-t">${title || '🔊 听一听'}</span>${items.map(([k, t]) => `<button type="button" class="p13-play" data-snd="${k}"><span class="ic">▶</span> ${t}</button>`).join('')}</div>`;

    /** 步骤定义里的 play: [[key, 文字]…] → 用 blocks 排版：公式 → 代码 → 预览 → ▶ 播放条 → 说明 → 注意 */
    function mkStep(s) {
        if (!s.play) return s;
        const b = [], out = Object.assign({}, s);
        (Array.isArray(s.formula) ? s.formula : s.formula ? [s.formula] : []).forEach((t) => b.push(typeof t === 'string' ? { type: 'formula', text: t } : Object.assign({ type: 'formula' }, t)));
        (Array.isArray(s.code) ? s.code : s.code ? [s.code] : []).forEach((c) => b.push(Object.assign({ type: 'code' }, c)));
        if (s.previews) b.push({ type: 'previews', items: s.previews });
        b.push({ type: 'html', html: playBar(s.play, s.playTitle) });
        (Array.isArray(s.notes) ? s.notes : s.notes ? [s.notes] : []).forEach((h) => b.push({ type: 'note', html: h }));
        (Array.isArray(s.warn) ? s.warn : s.warn ? [s.warn] : []).forEach((h) => b.push({ type: 'warn', html: h }));
        ['formula', 'code', 'previews', 'notes', 'warn', 'play', 'playTitle'].forEach((k) => delete out[k]);
        out.blocks = b;
        return out;
    }
    /** 扩散的 16 × 16 mel 图：[通道 × 帧] → [帧 × 通道]（gridPanel 的布局） */
    function melT(img) { const T = DIF.TF, M = DIF.NM, o = new Float32Array(T * M); for (let m = 0; m < M; m++) for (let t = 0; t < T; t++) o[t * M + m] = img[m * T + t]; return o; }

    return {
        melT, ASR_GATE, SR, TAU, now, fft, fftPlan, hann, stft, istft, powerDb, magnitude, hz2mel, mel2hz, melBank, melSpec, melInverse, glCreate, glStep,
        midiHz, noteName, TIMBRES, addNote, SCALE, makeMelody, renderMelody, chirpLin, fadeEdges,
        resonatorCoef, formantSynth, VOWELS, VBW, vowelClip,
        mdctPlan, mdctFrame, imdctFrame, mdct, imdct, rvqCreate, rvqEncode, rvqDecode, rvqTrainStep, nearest, refreshNorms,
        delayPattern, undelay, tlmCreate, tlmForward, tlmBatch, tlmTrainStep, tlmEval, tlmGenerate,
        PHONES, PHONE_IPA, LEXICON, TONES, pinyinOf, ttsFrontend, tts,
        ASR, asrFeatures, randomSpeaker, asrDataset, asrCreate, asrForward,
        DIF, difNote, difMel, difDataset, difToWave, difPCA, difEncode, difDecode, difSchedule, difCreate, difForward, difTrainStep, difSample,
        aliasFreq, sincAt, resample, quantize, snrDb, measureFold, makeClip,
        CODEC, melodySet, codecData, coefSNR, newCodecRun, codecEval, codecTick, codecTestSNR,
        LMC, newLMRun, lmTick, lmGenerate, pitchReport, DIFR, newDifRun, difTick, difSampleNow, ASRR, newAsrRun, asrTick, asrSample, asrScan,
        pcaBegin, pcaAccum, pcaIterate, pcaFinish, LIVE_X, chart, gridView, gridPanel, dbRange, tokenView, waveView, niceTicks, tickFmt, AUDIO, playAudio, stopAudio, sound, playBar, mkStep,
    };
})();

// 训练类预览在模型还没准备好（合成数据 / PCA / 第一次评估）时画进度提示，不留空白画布
function P13wait(c, R, kind) {
    const D = P13.DIFR, A = P13.ASRR;
    let msg = '正在准备…';
    if (kind === 'dif') msg = !R ? '等待开始（滚动到本步时开始训练）' : R.phase === 0 ? `正在合成训练数据：${R.made} / ${D.N} 个音符` : R.phase === 1 ? (R.P && R.P.pos < D.N ? `正在做 PCA：累加协方差 ${R.P.pos} / ${D.N}` : `正在做 PCA：子空间迭代 ${R.P ? R.P.it : 0} / 60`) : `第 ${R.step} 步：等待第一次采样…`;
    else if (kind === 'asr') msg = !R ? '等待开始（滚动到本步时开始训练）' : R.phase === 0 ? `正在合成训练 / 测试数据：${R.made} / ${A.NTR + A.NTE} 条元音` : `第 ${R.step} 步：等待第一次测试集评估…`;
    else if (kind === 'codec') msg = !R ? '等待开始（滚动到本步时开始训练）' : `码本训练第 ${R.step} 步：等待第一次评估…`;
    else if (kind === 'lm') { const L = P13.LMC; msg = !R ? '等待开始（滚动到本步时开始训练）' : R.phase === 0 ? `训练阶段 1 / 3：训练本步自己的编解码器码本 ${R.cstep} / ${L.CODEC_STEPS} 步` : R.phase === 1 ? `训练阶段 2 / 3：冻结码本，给训练旋律打 token（${R.tokPos} 帧）` : `训练阶段 3 / 3：语言模型第 ${R.step} 步，等待第一次续写…`; }
    LM.draw.text(c, [{ text: '        ' + msg, color: LM.C.gray }], { px: 12 });
}

// ═══════════════ 本页公用的 C++（graph.common）：FFT / 窗函数 / STFT / ISTFT ═══════════════
const P13_CPP_COMMON = String.raw`
// 全篇统一的采样率（8 kHz，电话音质：浏览器里能实时算 STFT / Griffin-Lim / 训练）与 π
const int SR = 8000;
const double kPi = 3.14159265358979323846;

// 复数谱：第 t 帧第 k 个频率在 [t·nb + k]；N = 窗长，H = 跳长，nb = N/2 + 1 个频率（实信号只需要非负频率）
struct Spec { int T = 0, N = 0, H = 0, nb = 0; std::vector<float> re, im; };

// 原地基 2 复数 FFT（Cooley–Tukey）；inverse = true 时做逆变换并除以 n
void fft(std::vector<double>& re, std::vector<double>& im, bool inverse) {
    int n = (int)re.size();
    for (int i = 1, j = 0; i < n; i++) {                        // 位反转重排
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    double sg = inverse ? 1.0 : -1.0;
    for (int size = 2; size <= n; size <<= 1) {                 // log₂n 级蝶形
        int half = size / 2;
        for (int i = 0; i < n; i += size)
            for (int j = 0; j < half; j++) {
                double wr = std::cos(2 * kPi * j / size), wi = sg * std::sin(2 * kPi * j / size);
                int a = i + j, b = a + half;
                double xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
    }
    if (inverse) for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// 周期 Hann 窗 w[n] = 0.5 − 0.5·cos(2πn/N)
std::vector<double> hann(int N) {
    std::vector<double> w = std::vector<double>(N);
    for (int n = 0; n < N; n++) w[n] = 0.5 - 0.5 * std::cos(2 * kPi * n / N);
    return w;
}

// STFT（center = true：两端各补 N/2 个零）：第 t 帧覆盖样本 [t·H − N/2, t·H + N/2)，共 1 + ⌊L/H⌋ 帧
Spec stft(const std::vector<float>& x, int N, int H) {
    int L = (int)x.size(), h2 = N / 2;
    Spec S;
    S.N = N; S.H = H; S.T = 1 + L / H; S.nb = N / 2 + 1;
    S.re = floats(S.T * S.nb);
    S.im = floats(S.T * S.nb);
    std::vector<double> w = hann(N), fr = std::vector<double>(N), fi = std::vector<double>(N);
    for (int t = 0; t < S.T; t++) {
        for (int n = 0; n < N; n++) {
            int i = t * H - h2 + n;
            fr[n] = (i >= 0 && i < L) ? x[i] * w[n] : 0.0;      // 加窗
            fi[n] = 0.0;
        }
        fft(fr, fi, false);
        for (int k = 0; k < S.nb; k++) { S.re[t * S.nb + k] = (float)fr[k]; S.im[t * S.nb + k] = (float)fi[k]; }
    }
    return S;
}

// 逆 STFT：每帧补全共轭对称 → 逆 FFT → 加窗重叠相加 → 除以 Σw²（最小二乘意义下的逆，与 torch.istft 相同）
std::vector<float> istft(const Spec& X, int L) {
    int N = X.N, H = X.H, h2 = N / 2;
    std::vector<double> w = hann(N), acc = std::vector<double>(L, 0.0), wsum = std::vector<double>(L, 0.0);
    std::vector<double> fr = std::vector<double>(N), fi = std::vector<double>(N);
    for (int t = 0; t < X.T; t++) {
        for (int k = 0; k < N; k++) {
            int q = k < X.nb ? k : N - k;                       // 负频率 = 对应正频率的共轭
            fr[k] = X.re[t * X.nb + q];
            fi[k] = (k == 0 || k == h2) ? 0.0 : (k < X.nb ? X.im[t * X.nb + q] : -X.im[t * X.nb + q]);   // DC / Nyquist 只取实部
        }
        fft(fr, fi, true);
        for (int n = 0; n < N; n++) {
            int i = t * H - h2 + n;
            if (i >= 0 && i < L) { acc[i] += fr[n] * w[n]; wsum[i] += w[n] * w[n]; }
        }
    }
    std::vector<float> x = floats(L);
    for (int i = 0; i < L; i++) x[i] = wsum[i] > 1e-8 ? (float)(acc[i] / wsum[i]) : 0.f;
    return x;
}
`;

PipelineLab.mount({
    paramSteps: { fs: ['wave'], tone: ['wave'], bits: ['wave'], win: ['stft', 'mel', 'griffin'], hop: ['stft', 'mel', 'griffin'], nmels: ['mel', 'griffin'], K: ['codec'], nq: ['codec'], temp: [], phrase: ['tts'], voice: ['tts'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '13 · 音频与语音：频谱、音频 token 与生成',
    lang: 'zh-CN',
    codeLang: 'cpp',
    seed: 13,
    css: LM.CSS + `
        .p13-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 10px 0 6px; }
        .p13-bar-t { color: #8fb4cc; font-size: 0.8rem; margin-right: 2px; }
        .p13-play { --p: 0; font: inherit; font-size: 0.82rem; color: #d6e6f2; cursor: pointer; padding: 3px 12px; border-radius: 12px; border: 1px solid #2a3d4f;
            background: linear-gradient(90deg, rgba(90,166,216,0.38) calc(var(--p) * 100%), #0f1a24 calc(var(--p) * 100%)); }
        .p13-play:hover { border-color: #4f86ad; }
        .p13-play.on { border-color: #6ab0e0; color: #fff; }
        .p13-play.na { border-color: #e06c75; color: #e06c75; }
        .p13-play .ic { display: inline-block; width: 0.9em; }
        .p13-note { color: #7d838c; font-size: 0.8rem; }
    `,
    regen: { label: '换随机种子', title: '换一组随机初始化（Griffin-Lim 的初始相位、码本、语言模型、扩散模型、识别器），所有训练从头开始；合成的数据不变' },

    params: [
        { id: 'fs', label: '采样率 fs', min: 1000, max: 16000, step: 250, value: 8000, unit: ' Hz', affects: 'frame', desc: 'Step 01：对「连续」信号采样的频率。奈奎斯特频率 = fs/2：高于它的成分会被折回低频（混叠）。把 fs 降到 6000 以下，5 kHz 的音就变成了另一个音高。' },
        { id: 'tone', label: '输入频率 f', min: 50, max: 7950, step: 10, value: 5000, unit: ' Hz', affects: 'frame', desc: 'Step 01：被采样的纯音频率。f < fs/2 时采样点能还原它；f > fs/2 时，同一串采样点恰好也落在频率 |f − fs·round(f/fs)| 的正弦上——重建出来的是这个「假」频率。' },
        { id: 'bits', label: '量化位数', options: [16, 12, 8, 6, 4, 3, 2], value: 8, affects: 'frame', desc: 'Step 01：每个样本用几个 bit。均匀量化每少 1 bit，满幅正弦的信噪比少约 6 dB（6.02b + 1.76）；μ-law 让小信号也有足够的级数（WaveNet 用 8 bit μ-law，256 类）。' },
        { id: 'win', label: '窗长 N', options: [{ value: 64, label: '64（8 ms）' }, { value: 128, label: '128（16 ms）' }, { value: 256, label: '256（32 ms）' }, { value: 512, label: '512（64 ms）' }, { value: 1024, label: '1024（128 ms）' }], value: 256, affects: 'frame', desc: 'Step 02–04：STFT 每帧的样本数。窗越长频率分辨率越高（谐波分得开），时间分辨率越差（咔嗒声糊成一片）——这是时频不确定性。' },
        { id: 'hop', label: '跳长', options: [{ value: 2, label: 'N/2' }, { value: 4, label: 'N/4' }, { value: 8, label: 'N/8' }], value: 4, affects: 'frame', desc: 'Step 02–04：相邻两帧相隔多少样本。跳长只决定时间轴的采样密度，不改变每帧的分辨率；Griffin-Lim 需要足够的重叠（N/4 或更密）才能收敛到好的相位。' },
        { id: 'nmels', label: 'mel 通道数', options: [16, 24, 40, 64, 80], value: 40, affects: 'frame', desc: 'Step 03–04：mel 滤波器组的三角滤波器个数（Whisper 用 80）。通道少 → 频谱更粗、更省；窗很短时通道太多，低频的三角形里落不到任何 FFT bin（空滤波器）。' },
        { id: 'K', label: '码本大小 K', options: [16, 32, 64, 128], value: 64, affects: 'frame', desc: 'Step 05：RVQ 每层码本的码字数。每层每帧花 log₂K bit：K 越大每层越准、码率越高、训练越慢（K = 128 时每帧训练约 2 倍耗时）。改动会让第 05 步从头训练。' },
        { id: 'nq', label: '解码层数', min: 1, max: 8, step: 1, value: 4, affects: 'frame', desc: 'Step 05：解码时用前几层码本（码率 = 帧率 × 层数 × log₂K）。只用 1 层只剩轮廓，层数越多残差越小——SoundStream / EnCodec 用同一个模型支持多种码率就是这个原理。' },
        { id: 'temp', label: '采样温度', min: 0.3, max: 1.6, step: 0.05, value: 1.0, affects: 'frame', desc: 'Step 06：音频 token 语言模型续写时的温度。低温更保守（不容易出错，但容易重复）；高温更多样，也更容易生成训练集里没有的帧（听起来是杂音或不准的音）。' },
        { id: 'phrase', label: 'TTS 文本', options: ['妈麻马骂', '你好', '我爱你', '啊诶衣喔乌', '猫爱鱼'], value: '妈麻马骂', affects: 'frame', desc: 'Step 08–09：要合成的句子。「妈麻马骂」只差声调；「你好」会触发上声变调（nǐ hǎo → ní hǎo）；「啊诶衣喔乌」是第 09 步识别器认识的五个元音。' },
        { id: 'voice', label: '声音', options: [{ value: 'm', label: '男声' }, { value: 'f', label: '女声' }, { value: 'c', label: '童声' }], value: 'm', affects: 'frame', desc: 'Step 08–09：说话人 = 基频（120 / 220 / 300 Hz）+ 声道长度（共振峰整体 ×1 / ×1.17 / ×1.3）。换声音后第 09 步的识别器还认得出吗？' },
    ],

    stages: [
        { id: 'clip', lifetime: 'resident', run(S) { S.snd = S.snd || {}; S.clip = P13.makeClip(); S.snd.clip = { x: S.clip.x, sr: P13.SR }; } },
        {
            id: 'samp', lifetime: 'frame', run(S) {   // Step 01：只在 fs / f / 位数变化时重算
                const key = [S.fs, S.tone, S.bits].join('|'); if (S.sm && S.sm.key === key) return;
                const fs = S.fs, f = S.tone, fa = P13.aliasFreq(f, fs), k0 = -64, nS = Math.ceil(0.006 * fs) + 128, xs = new Float64Array(nS);
                for (let i = 0; i < nS; i++) xs[i] = Math.sin(P13.TAU * f * (k0 + i) / fs + 0.4);
                const tw = 0.004, NC = 600, cont = new Float64Array(NC), rec = new Float64Array(NC), ts = new Float64Array(NC);
                for (let j = 0; j < NC; j++) { const t = tw * j / (NC - 1); ts[j] = t * 1000; cont[j] = Math.sin(P13.TAU * f * t + 0.4); rec[j] = P13.sincAt(xs, k0, fs, t, 48); }
                const kx = [], ky = []; for (let k = 0; k <= Math.floor(tw * fs); k++) { kx.push(1000 * k / fs); ky.push(xs[k - k0]); }
                const fold = P13.measureFold(fs, 16000, 80);
                // 0 → 8 kHz 的线性啁啾，直接按 fs 采样（没有抗混叠滤波）
                const nC = Math.round(2 * fs), ch = new Float64Array(nC); for (let i = 0; i < nC; i++) { const t = i / fs; ch[i] = 0.5 * Math.sin(P13.TAU * (50 * t + 0.5 * (7950 / 2) * t * t)); }
                const Nf = fs >= 8000 ? 256 : 128, sp = P13.stft(ch, Nf, Nf / 4), cdb = P13.powerDb(sp);
                // 量化：演示片段里的元音 a（−20 dBFS）与满幅 / 不同电平的正弦
                const seg = S.clip.x.subarray(Math.round(1.70 * P13.SR), Math.round(1.70 * P13.SR) + 160), lev = 0.1 / Math.max(...Array.from(seg, Math.abs)), vx = Float64Array.from(seg, (v) => v * lev);
                const qu = P13.quantize(vx, S.bits, false), qm = P13.quantize(vx, S.bits, true);
                const levels = [], su = [], smu = [], sine = new Float64Array(4000);
                for (let L = -60; L <= 0; L += 5) { const a = Math.pow(10, L / 20); for (let i = 0; i < sine.length; i++) sine[i] = a * Math.sin(P13.TAU * 441 * i / 8000 + 0.1); levels.push(L); su.push(P13.snrDb(sine, P13.quantize(sine, S.bits, false))); smu.push(P13.snrDb(sine, P13.quantize(sine, S.bits, true))); }
                const clipQ = Float64Array.from(S.clip.x, (v) => v * 0.1), cqU = P13.quantize(clipQ, S.bits, false), cqM = P13.quantize(clipQ, S.bits, true);
                S.sm = { key, fs, f, fa, aliased: f > fs / 2, ts, cont, rec, kx, ky, fold, chirp: ch, cdb, cT: sp.T, cNb: sp.nb, vx, qu, qm, levels, su, smu, snrU: P13.snrDb(clipQ, cqU), snrM: P13.snrDb(clipQ, cqM) };
                const T = 1.2, hi = 32000, nT = Math.round(T * hi);
                S.snd.tone = () => { const y = new Float64Array(nT); for (let i = 0; i < nT; i++) y[i] = 0.5 * Math.sin(P13.TAU * f * i / hi); return { x: y, sr: hi }; };
                S.snd.toneS = () => { const n = Math.round(T * fs), y = new Float64Array(n); for (let i = 0; i < n; i++) y[i] = 0.5 * Math.sin(P13.TAU * f * i / fs); return { x: P13.resample(y, fs, hi, 24), sr: hi }; };
                S.snd.chirp = () => { const n = 2 * hi, y = new Float64Array(n); for (let i = 0; i < n; i++) { const t = i / hi; y[i] = 0.5 * Math.sin(P13.TAU * (50 * t + 0.5 * (7950 / 2) * t * t)); } return { x: y, sr: hi }; };
                S.snd.chirpS = () => ({ x: P13.resample(ch, fs, hi, 24), sr: hi });
                S.snd.qU = { x: cqU, sr: P13.SR }; S.snd.qM = { x: cqM, sr: P13.SR };
            },
        },
        {
            id: 'spec', lifetime: 'frame', run(S) {   // Step 02–04：演示片段的 STFT / mel，只在窗长 / 跳长 / mel 数变化时重算
                const key = [S.win, S.hop, S.nmels].join('|'); if (S.sp && S.sp.key === key) return;
                const N = S.win, H = N / S.hop, sp = P13.stft(S.clip.x, N, H), db = P13.powerDb(sp), bank = P13.melBank(N, S.nmels), mel = P13.melSpec(sp, bank);
                const mdb = Float32Array.from(mel, (v) => 10 * Math.log10(v + 1e-10));
                S.sp = { key, N, H, sp, db, T: sp.T, nb: sp.nb, bank, mel, mdb, range: P13.dbRange(db, 80), mrange: P13.dbRange(mdb, 80) };
                if (S.selT == null || S.selT >= sp.T * H / P13.SR) S.selT = 1.75;
            },
        },
        {
            id: 'trio', lifetime: 'resident', lazy: true, run(S) {   // Step 02：同一段（旋律结尾 + 咔嗒）在三种窗长下的频谱图
                S.trio = [64, 256, 1024].map((N) => { const H = N / 4, sp = P13.stft(S.clip.x, N, H), db = P13.powerDb(sp); return { N, H, db, T: sp.T, nb: sp.nb, range: P13.dbRange(db, 80) }; });
            },
        },
        {
            id: 'gl', lifetime: 'history', lazy: true, run(S) {   // Step 04：Griffin-Lim 与快速 Griffin-Lim 各迭代一次 / 帧，最多 100 次（只在本步可见时运行）
                const key = [S.epoch, S.win, S.hop, S.seed].join('|');
                if (!S.gl || S.gl.key !== key) {
                    const p = S.sp, A = P13.magnitude(p.sp), L = S.clip.L;
                    S.gl = { key, A, max: 100, g: P13.glCreate(A, p.T, p.N, p.H, L, { seed: S.seed + 1 }), fg: P13.glCreate(A, p.T, p.N, p.H, L, { seed: S.seed + 1, momentum: 0.99 }), x0: null };
                    S.gl.x0 = Float64Array.from(P13.glCreate(A, p.T, p.N, p.H, L, { init: 'zero' }).x);
                    S.snd.gl0 = { x: S.gl.x0, sr: P13.SR };
                }
                const G = S.gl; if (G.g.it < G.max) { P13.glStep(G.g); P13.glStep(G.fg); G.db = P13.powerDb(G.fg.sp); }
                S.snd.gl = { x: G.g.x, sr: P13.SR }; S.snd.fgl = { x: G.fg.x, sr: P13.SR };
            },
        },
        {
            id: 'glmel', lifetime: 'frame', lazy: true, run(S) {   // Step 04：只给 log-mel（再用非负最小二乘还原线性谱），快速 GL 32 次
                const key = S.sp.key; if (S.gm && S.gm.key === key) return;
                const p = S.sp, P = P13.melInverse(p.mel, p.T, p.bank, 30), A = Float64Array.from(P, (v) => Math.sqrt(v)), G = P13.glCreate(A, p.T, p.N, p.H, S.clip.L, { seed: 9, momentum: 0.99 });
                for (let k = 0; k < 32; k++) P13.glStep(G);
                const A0 = P13.magnitude(p.sp); let e = 0, a = 0; for (let i = 0; i < A0.length; i++) { e += (A[i] - A0[i]) ** 2; a += A0[i] * A0[i]; }
                S.gm = { key, x: G.x, sc: G.sc[G.sc.length - 1], invErr: Math.sqrt(e / a), Pdb: Float32Array.from(P, (v) => 10 * Math.log10(v + 1e-10)) };
                S.snd.glmel = { x: G.x, sr: P13.SR };
            },
        },
        { id: 'cdata', lifetime: 'resident', lazy: true, run(S) { S.cdata = P13.codecData(); const t = S.cdata.test[0]; S.snd.codecOrig = { x: t.x, sr: P13.SR }; } },
        {
            id: 'codec', lifetime: 'history', lazy: true, needs: ['cdata'], run(S) {   // Step 05：RVQ 码本的 EMA k-means 训练（只在本步可见时运行）
                const key = [S.epoch, S.K, S.seed].join('|');
                if (!S.codec || S.codec.key !== key) { S.codec = P13.newCodecRun(S.K, S.seed); S.codec.key = key; }
                P13.codecTick(S.codec, S.cdata);
                const R = S.codec, D = S.cdata, t = D.test[0];
                [1, 2, 4, 8].forEach((n) => { S.snd['codec' + n] = () => ({ x: P13.imdct(P13.rvqDecode(R.Q, P13.rvqEncode(R.Q, t.mdct.C, t.mdct.F, P13.CODEC.NQ), t.mdct.F, n), t.mdct.F, P13.CODEC.M, t.x.length), sr: P13.SR }); });
                S.snd.codecNq = () => S.snd['codec' + 1] && { x: P13.imdct(P13.rvqDecode(R.Q, P13.rvqEncode(R.Q, t.mdct.C, t.mdct.F, P13.CODEC.NQ), t.mdct.F, S.nq), t.mdct.F, P13.CODEC.M, t.x.length), sr: P13.SR };
            },
        },
        {
            id: 'codecView', lifetime: 'frame', lazy: true, needs: ['codec'], run(S) {   // Step 05：当前码本 + 当前层数下的解码结果（评估片段）
                const R = S.codec; if (!R || !R.last) return;
                const key = R.last.step + '|' + S.nq + '|' + R.key; if (S.cv && S.cv.key === key) return;
                const D = S.cdata, EF = D.EF, y = P13.imdct(P13.rvqDecode(R.Q, R.last.codes, EF, S.nq), EF, P13.CODEC.M, (EF - 1) * P13.CODEC.M), sp = P13.stft(y, 256, 64);
                if (!S.cvOrig) { const x0 = D.test[0].x.subarray(0, (EF - 1) * P13.CODEC.M), s0 = P13.stft(x0, 256, 64); S.cvOrig = { db: P13.powerDb(s0), T: s0.T, nb: s0.nb }; S.cvOrig.range = P13.dbRange(S.cvOrig.db, 80); }
                S.cv = { key, db: P13.powerDb(sp), T: sp.T, nb: sp.nb };
            },
        },
        {
            id: 'lm', lifetime: 'history', lazy: true, needs: ['cdata'], run(S) {   // Step 06：编解码器 → 打 token → 延迟排布 LM（只在本步可见时运行）
                const key = [S.epoch, S.seed].join('|');
                if (!S.lm || S.lm.key !== key) { S.lm = P13.newLMRun(S.seed); S.lm.key = key; }
                P13.lmTick(S.lm, S.cdata);
            },
        },
        {
            id: 'lmGen', lifetime: 'frame', lazy: true, needs: ['lm'], run(S) {   // Step 06：每 250 步（以及训练结束时）、或温度改变时重新续写
                const R = S.lm; if (!R || R.phase < 2) return;
                const bucket = R.step >= P13.LMC.STEPS ? P13.LMC.STEPS : Math.floor(R.step / 250) * 250, key = [R.key, bucket, S.temp].join('|');
                if (S.lg && S.lg.key === key) return;
                const g = [0, 1].map((w) => P13.lmGenerate(R, w, S.temp, 500 + w + 7 * S.seed));
                g.forEach((q) => { const sp = P13.stft(q.x, 256, 64); q.db = P13.powerDb(sp); q.T = sp.T; q.nb = sp.nb; });
                S.lg = { key, step: R.step, g };
                S.snd.lmGen = { x: g[0].x, sr: P13.SR }; S.snd.lmGen2 = { x: g[1].x, sr: P13.SR }; S.snd.lmPrompt = { x: g[0].prompt, sr: P13.SR }; S.snd.lmReal = { x: g[0].real, sr: P13.SR }; S.snd.lmRecon = { x: g[0].recon, sr: P13.SR };
            },
        },
        {
            id: 'dif', lifetime: 'history', lazy: true, run(S) {   // Step 07：合成音符 → PCA 潜空间 → 潜空间扩散（只在本步可见时运行）
                const key = [S.epoch, S.seed].join('|');
                if (!S.dif || S.dif.key !== key) { S.dif = P13.newDifRun(S.seed); S.dif.key = key; }
                P13.difTick(S.dif);
                const R = S.dif;
                for (let i = 0; i < 4; i++) S.snd['dif' + i] = () => (R.samples ? { x: P13.difToWave(R.samples.imgs.subarray(i * P13.DIF.D, (i + 1) * P13.DIF.D)), sr: P13.SR } : null);
                S.snd.difReal = () => (R.phase > 0 ? { x: P13.difToWave(R.X.subarray(0, P13.DIF.D)), sr: P13.SR } : null);
            },
        },
        {
            id: 'tts', lifetime: 'frame', run(S) {   // Step 08：文本 → 波形（规则前端 + 共振峰声码器），只在文本 / 声音变化时重算
                const key = S.phrase + '|' + S.voice; if (S.tt && S.tt.key === key) return;
                const V = { m: { f0: 120, vtl: 1 }, f: { f0: 220, vtl: 1.17 }, c: { f0: 300, vtl: 1.3 } }[S.voice], r = P13.tts(S.phrase, V, LM.rng(7)), sp = P13.stft(r.x, 256, 64);
                S.tt = Object.assign(r, { key, voice: V, db: P13.powerDb(sp), T: sp.T, nb: sp.nb });
                S.tt.range = P13.dbRange(S.tt.db, 70);
                S.snd.tts = { x: r.x, sr: P13.SR };
            },
        },
        {
            id: 'asr', lifetime: 'history', lazy: true, run(S) {   // Step 09：合成元音数据 → 训练卷积识别器（只在本步可见时运行）
                const key = [S.epoch, S.seed].join('|');
                if (!S.asr || S.asr.key !== key) { S.asr = P13.newAsrRun(S.seed); S.asr.key = key; }
                P13.asrTick(S.asr);
            },
        },
        {
            id: 'asrScan', lifetime: 'frame', lazy: true, needs: ['asr'], run(S) {   // Step 09：用当前识别器扫一遍第 08 步合成的句子
                const R = S.asr; if (!R || !R.m) return;
                const key = [R.key, Math.floor(R.step / 25), S.tt.key].join('|'); if (S.as && S.as.key === key) return;
                S.as = Object.assign(P13.asrScan(R.m, S.tt.x), { key, step: R.step });
            },
        },
    ],

    intro: {
        title: '声音也是一串数：从波形、频谱到音频 token，再到生成与识别',
        flow: ['波形（采样 / 量化）', 'STFT 频谱图', 'mel 频谱', '相位重建（声码器）', '编解码器：RVQ → token', 'token 语言模型', 'mel 潜空间扩散', 'TTS：文本 → 语音', 'ASR：语音 → 文本'],
        html: LM.navHtml('13') + `
            <p>图像是二维的像素网格，文本是一串 token，声音则是<strong>一维的压力随时间变化</strong>：麦克风每秒读几千到几万次数，得到一条很长的波形。1 秒 16 kHz 的语音就有 16 000 个数——直接把它当成 token 序列喂给第 04 篇的 Transformer，序列太长、每个数的含义又太「底层」。音频模型的历史，很大程度上就是在找<strong>更好的表示</strong>：频谱图（看得见音高和音色）、mel 频谱（贴近人耳）、神经编解码器的离散 token（能像文字一样做语言建模），然后在这些表示上做生成（TTS、音乐生成）和理解（ASR）。</p>
            <p>本页的声音全部在页面里<strong>合成</strong>：五声音阶的旋律（风琴 / 拨弦两种音色）、啁啾、用共振峰合成器（Klatt 1980 的级联谐振器）做的元音和汉语音节、噪声与咔嗒声。采样率统一为 <b>8 kHz</b>（电话音质），这样 STFT、Griffin-Lim 和五个训练都能在浏览器里实时跑。每个 <b>▶</b> 按钮播放的都是页面刚刚算出来的真实结果：原始声音、Griffin-Lim 的重建、不同码率下编解码器的输出、语言模型 / 扩散模型生成的声音、合成的语音。</p>
            <p>带 <span style="color:#a45fa5">LIVE</span> 的步骤在训练（第 04 步 Griffin-Lim 迭代、第 05 步码本、第 06 步 token 语言模型、第 07 步扩散模型、第 09 步识别器）。为了不让整页卡顿，<strong>每个训练只在它所在的步骤出现在屏幕上时才跑</strong>，滚走就暂停、回来接着训。代码块是同一算法的 C++ 版本（C++17 标准库 + 本系列自带的极简库 <code>cpp/minml_*.h</code>，见节点图上方的 Common）。会用到前面几篇的东西：第 09 篇的 VQ（向量量化）、第 04 / 05 篇的 Transformer 语言模型、第 11 篇的扩散，用到时都会先回顾一句。</p>
            <table class="lm-tbl"><tr><th>本页（教学版）</th><th>真实系统（量级）</th></tr>
                <tr><td>8 kHz 采样，几秒钟的合成声音</td><td>语音 16 kHz（Whisper），音乐 24–48 kHz（EnCodec 24 / 48 kHz，MusicGen 32 kHz）</td></tr>
                <tr><td>编解码器：固定的 MDCT 变换 + 8 层 RVQ（K = 64，码本用 k-means 实时训练），31.25 帧/秒，0.19–1.5 kbps</td><td>SoundStream / EnCodec：卷积编码器 / 解码器 + RVQ（码本 1024），75 帧/秒，1.5–24 kbps，对抗损失 + 多尺度频谱损失端到端训练</td></tr>
                <tr><td>token LM：4.3 万参数，4 个码本的 delay pattern，上下文 32 帧 ≈ 1 秒</td><td>MusicGen：3 亿–33 亿参数，4 个码本的 delay pattern，50 帧/秒</td></tr>
                <tr><td>识别：5 个元音分类，约 2 千参数的卷积网络</td><td>Whisper：68 万小时弱监督数据，编码器-解码器 Transformer，直接输出文字</td></tr></table>`,
    },

    steps: [
        // ─────────────────────────────── 01 波形
        {
            id: 'wave', title: '声音 = 波形：采样、奈奎斯特与混叠、量化', tag: 'SIGNAL',
            desc: '声音是空气压力随时间的变化 x(t)。数字化分两步：<strong>采样</strong>（每隔 1/fs 秒读一次数）和<strong>量化</strong>（把读数取整到有限个等级）。采样定理说：只要信号里没有高于 <strong>fs/2（奈奎斯特频率）</strong>的成分，这串离散的点就能<strong>完全</strong>还原原信号——还原的方法是 sinc 插值（DAC 里的重建滤波器）。反过来，高于 fs/2 的成分会和某个低频成分产生<strong>一模一样的采样点</strong>，还原时就变成了那个低频——这叫<strong>混叠</strong>（aliasing）。所以录音设备在采样前一定先低通滤波（抗混叠滤波）。下图的黄点是采样点，灰线是真实的 5 kHz 正弦，彩色线是用 sinc 插值从这些点真实重建出来的波形。',
            formula: [
                'x[k] = x(k / fs)      x(t) = Σ_k x[k] · sinc(fs·t − k)   // 重建：f < fs/2 时完全还原',
                'sin(2π f k/fs) = ±sin(2π f_a k/fs)      f_a = |f − fs · round(f / fs)|   // f > fs/2 时采样点和 f_a 的正弦相同',
                '均匀量化（b bit，满幅正弦）：SNR ≈ 6.02·b + 1.76 dB      μ-law：y = sgn(x)·ln(1 + μ|x|) / ln(1 + μ)，μ = 255',
            ],
            code: {
                title: 'C++ · sampleTone / aliasFrequency / sincReconstruct / quantizeUniform / quantizeMuLaw', src: `
                // 以采样率 fs 对 x(t) = sin(2πf·t + φ) 采样 n 个点：x[k] = x(k / fs)
                std::vector<float> sampleTone(float f, float fs, int n, float phi) {
                    std::vector<float> x = floats(n);
                    for (int k = 0; k < n; k++) x[k] = (float)std::sin(2 * kPi * f * k / fs + phi);
                    return x;
                }

                // 采样后「听到」的频率：把 f 折回 [0, fs/2]
                float aliasFrequency(float f, float fs) {
                    float r = std::fmod(f, fs);
                    return r > fs / 2 ? fs - r : r;
                }

                // 理想重建（Whittaker–Shannon）：x(t) = Σ_k x[k]·sinc(fs·t − k)，截断到 ±taps 个样本并乘 Hann 窗
                float sincReconstruct(const std::vector<float>& x, float fs, double t, int taps) {
                    double u = fs * t, s = 0;
                    int c = (int)std::floor(u);
                    for (int k = c - taps + 1; k <= c + taps; k++) {
                        if (k < 0 || k >= (int)x.size()) continue;
                        double d = u - k, sinc = d == 0 ? 1.0 : std::sin(kPi * d) / (kPi * d);
                        s += x[k] * sinc * (0.5 + 0.5 * std::cos(kPi * d / taps));
                    }
                    return (float)s;
                }

                // 均匀量化：每半边 2^(bits−1) 级，四舍五入
                float quantizeUniform(float x, int bits) {
                    float q = (float)(1 << (bits - 1));
                    float v = std::round(std::clamp(x, -1.f, 1.f) * q);
                    return std::clamp(v, -q, q - 1) / q;
                }

                // μ-law（μ = 255）：先压扩，小信号被放大 → 量化级数更多地给了小幅度；再均匀量化、展开回去
                float quantizeMuLaw(float x, int bits) {
                    const float mu = 255.f;
                    float y = std::log1p(mu * std::min(std::fabs(x), 1.f)) / std::log1p(mu);
                    float yq = quantizeUniform(x < 0 ? -y : y, bits);
                    float a = (std::pow(1 + mu, std::fabs(yq)) - 1) / mu;
                    return yq < 0 ? -a : a;
                }

                // 信噪比 SNR = 10·log10(Σx² / Σ(x − y)²)
                double snrDb(const std::vector<float>& x, const std::vector<float>& y) {
                    double s = 0, e = 0;
                    for (size_t i = 0; i < x.size(); i++) { s += x[i] * x[i]; e += (x[i] - y[i]) * (x[i] - y[i]); }
                    return 10 * std::log10((s + 1e-30) / (e + 1e-30));
                }`,
            },
            play: [['tone', '原始（f）'], ['toneS', '以 fs 采样后再重建'], ['chirp', '啁啾 0 → 8 kHz'], ['chirpS', '啁啾（按 fs 采样）'], ['qU', '演示片段 −20 dBFS · 均匀量化'], ['qM', '同上 · μ-law']],
            previews: [
                {
                    id: 'samp', label: '采样与重建（4 ms）', sub: '灰 = 真实的「连续」正弦 · 黄 = 采样点 · 绿 / 红 = 用 sinc 插值从采样点重建的波形（红 = 混叠：重建出来的是另一个频率）', large: true, size: [720, 250],
                    render: (c, S) => {
                        const m = S.sm, col = m.aliased ? LM.C.red : LM.C.green;
                        P13.chart(c, {
                            x: [0, 4], y: [-1.3, 1.3], xLabel: '时间 (ms)', yTicks: [-1, 0, 1],
                            title: `fs = ${m.fs} Hz → 奈奎斯特 ${m.fs / 2} Hz · 输入 f = ${m.f} Hz · ` + (m.aliased ? `采样点重建出的是 ${Math.round(m.fa)} Hz（混叠）` : '重建与原信号重合'),
                            series: [{ x: m.ts, y: m.cont, color: '#5b6068', width: 1.2 }, { x: m.ts, y: m.rec, color: col, width: 2.2 }, { x: m.kx, y: m.ky, color: LM.C.yellow, width: 1, stems: true }, { x: m.kx, y: m.ky, color: LM.C.yellow, line: false, dots: 3.2 }],
                            legend: [{ label: `原信号 ${m.f} Hz`, color: '#8a8f98' }, { label: `重建（${Math.round(m.fa)} Hz）`, color: col }, { label: `采样点（每 ${(1000 / m.fs).toFixed(3)} ms）`, color: LM.C.yellow, dot: true }],
                        });
                    },
                },
                {
                    id: 'fold', label: '输入频率 → 采样后的频率', sub: '点 = 对 81 个纯音分别采样再做 FFT，读出的谱峰位置（实测）· 灰虚线 = 折叠公式 · 黄 = 当前 f', size: [360, 240],
                    render: (c, S) => {
                        const m = S.sm, th = [], tx = []; for (let i = 0; i <= 400; i++) { const f = 16000 * i / 400; tx.push(f); th.push(P13.aliasFreq(f, m.fs)); }
                        P13.chart(c, {
                            x: [0, 16000], y: [0, 8200], xLabel: '输入频率 f (Hz)', yLabel: '输出 (Hz)', xFmt: (v) => (v ? v / 1000 + 'k' : '0'), yFmt: (v) => (v ? v / 1000 + 'k' : '0'), padL: 46,
                            series: [{ x: tx, y: th, color: '#6a6f78', width: 1.2, dash: [4, 3] }, { x: m.fold.map((q) => q.f), y: m.fold.map((q) => q.fo), color: LM.C.blue, line: false, dots: 2.4 }, { x: [m.f], y: [m.fa], color: LM.C.yellow, line: false, dots: 5 }],
                            marks: [{ x: m.fs / 2, label: 'fs/2', color: LM.C.green }, { x: m.fs, label: 'fs', color: LM.C.orange, row: 1 }],
                        });
                    },
                },
                {
                    id: 'chirpSpec', label: '0 → 8 kHz 啁啾，直接按 fs 采样（没有抗混叠滤波）', sub: '频谱图：越过 fs/2 之后频率「反弹」回来——听起来音调先升后降 · 白虚线 = 真实频率', size: [360, 240],
                    render: (c, S) => {
                        const m = S.sm;
                        P13.gridView(c, m.cdb, m.cT, m.cNb, {
                            range: P13.dbRange(m.cdb, 70), xMax: 2, yMax: m.fs / 2000, xLabel: '时间 (s)', yLabel: 'kHz', legend: false, smooth: true, padR: 10,
                            overlay: (ctx, X, Y) => { ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(0), Y(0.05)); for (let i = 1; i <= 50; i++) { const t = 2 * i / 50, f = (50 + 3975 * t) / 1000; if (f > m.fs / 2000) { ctx.lineTo(X(t - 2 / 50 * (f - m.fs / 2000) / (3.975 * 2 / 50)), Y(m.fs / 2000)); break; } ctx.lineTo(X(t), Y(f)); } ctx.stroke(); ctx.setLineDash([]); },
                        });
                    },
                },
                {
                    id: 'quantWave', label: '量化：元音 a 的 20 ms（−20 dBFS）', sub: '灰 = 原波形 · 橙 = 均匀量化 · 青 = μ-law（同样的位数）', size: [360, 240],
                    render: (c, S) => {
                        const m = S.sm, t = Array.from(m.vx, (_, i) => 1000 * i / P13.SR);
                        P13.chart(c, { x: [0, 20], y: [-0.13, 0.13], xLabel: '时间 (ms)', title: `${S.bits} bit：均匀 SNR ${P13.snrDb(m.vx, m.qu).toFixed(1)} dB · μ-law ${P13.snrDb(m.vx, m.qm).toFixed(1)} dB`, series: [{ x: t, y: m.vx, color: '#6a6f78', width: 2.5 }, { x: t, y: m.qu, color: LM.C.orange, width: 1.4 }, { x: t, y: m.qm, color: LM.C.cyan, width: 1.4, dash: [3, 2] }] });
                    },
                },
                {
                    id: 'quantSnr', label: '信噪比 vs 信号电平', sub: '正弦（441 Hz）在不同电平下量化后实测的 SNR · 灰虚线 = 均匀量化的理论值 6.02b + 1.76 + 电平', size: [360, 240],
                    render: (c, S) => {
                        const m = S.sm, th = m.levels.map((L) => 6.02 * S.bits + 1.76 + L);
                        P13.chart(c, { x: [-60, 0], y: [-10, 100], xLabel: '信号电平 (dBFS)', yLabel: 'SNR (dB)', series: [{ x: m.levels, y: th, color: '#6a6f78', dash: [4, 3] }, { x: m.levels, y: m.su, color: LM.C.orange, width: 2, dots: 2.5 }, { x: m.levels, y: m.smu, color: LM.C.cyan, width: 2, dots: 2.5 }], legend: [{ label: `均匀 ${S.bits} bit`, color: LM.C.orange }, { label: `μ-law ${S.bits} bit`, color: LM.C.cyan }], legendAt: 'left' });
                    },
                },
            ],
            notes: '默认 fs = 8000、f = 5000 Hz：采样点每 0.125 ms 一个，重建出来的红线是 3000 Hz——播放「采样后」就能听到音调明显变低。把 <b>输入频率 f</b> 拖过 fs/2 来回扫，右上图的黄点沿着三角形的折线走；把 <b>采样率 fs</b> 拖到 16000，红线变绿。啁啾的频谱图里，越过 fs/2 的部分像镜子一样反射回来。量化：<b>量化位数</b>每少 1 bit 均匀量化的 SNR 少 6 dB；信号小 20 dB，SNR 也跟着少 20 dB（左下图里的橙线），μ-law 则在很宽的电平范围里保持差不多的 SNR——这就是电话（G.711）和 WaveNet 用 8 bit μ-law 的原因。',
            warn: '这里的「连续」信号是解析式，播放时用 32 kHz 近似连续；「采样后」那一路用 sinc 插值重建回 32 kHz（模拟理想 DAC），所以听到的混叠是真实的混叠，而不是重建方法的瑕疵。真实 ADC 在采样前有模拟抗混叠低通滤波器，所以普通录音里不会出现这种「反弹」；但在合成器、降采样代码里忘了先低通，就会出现。',
        },
        // ─────────────────────────────── 02 STFT
        {
            id: 'stft', title: '短时傅里叶变换（STFT）与频谱图：时间和频率不能同时精确', tag: 'STFT',
            desc: '一整段声音做一次傅里叶变换，只知道「有哪些频率」，不知道「什么时候」。STFT 把信号切成一帧帧（窗长 N），每帧乘一个窗函数（Hann）再做 FFT，帧与帧相隔 H 个样本（跳长）。|X|² 按时间排起来就是<strong>频谱图</strong>（spectrogram）：横轴时间、纵轴频率、颜色 = 能量（dB）。从图上能直接读出旋律的谐波（一组等间距的横线）、啁啾（斜线）、元音的共振峰（几条亮带）、噪声（整片铺开）和咔嗒（竖线）。<strong>窗长的权衡</strong>：频率分辨率 ≈ fs/N，时间分辨率 ≈ N/fs，两者的乘积是常数——窗长 1024（128 ms）时谐波清清楚楚但三下咔嗒糊成一团；窗长 64（8 ms）时咔嗒分得开，谐波却连成了一片。',
            formula: [
                'X[t, k] = Σ_{n=0}^{N−1} x[t·H − N/2 + n] · w[n] · e^{−2πi·k·n/N}      w[n] = 0.5 − 0.5·cos(2πn/N)   // Hann 窗，center = true',
                'Δf = fs / N      Δt = N / fs      Δf · Δt = 1   // 窗长 256：Δf = 31.25 Hz，Δt = 32 ms',
                'S_dB[t, k] = 10 · log10(|X[t, k]|^2 + 10^{−10})',
            ],
            code: {
                title: 'C++ · powerDb / timeFrequencyCell / spectrogramDb（stft / istft / fft 见 Common）', src: `
                // 功率谱 → dB：10·log10(|X|² + 1e−10)
                std::vector<float> powerDb(const Spec& S) {
                    std::vector<float> db = floats(S.T * S.nb);
                    for (int i = 0; i < S.T * S.nb; i++) db[i] = 10.f * std::log10(S.re[i] * S.re[i] + S.im[i] * S.im[i] + 1e-10f);
                    return db;
                }

                // 一个时频格子的大小：Δt = N / fs 秒，Δf = fs / N 赫兹（乘积恒为 1）
                struct TFCell { float dt, df; };
                TFCell timeFrequencyCell(int N) { TFCell c; c.dt = (float)N / SR; c.df = (float)SR / N; return c; }

                // 例：演示片段的频谱图（窗长 N、跳长 H = N/4）
                std::vector<float> spectrogramDb(const std::vector<float>& x, int N) { return powerDb(stft(x, N, N / 4)); }`,
            },
            previews: [
                {
                    id: 'spectro', label: '演示片段的频谱图', sub: '颜色 = 能量（dB，显示最响的 80 dB）· 青线 = 选中的帧 · <b>点击图上任意位置</b>选一帧，看右下两图', large: true, size: [720, 280],
                    render: (c, S) => {
                        const p = S.sp, dur = S.clip.L / P13.SR;
                        S._spLay = P13.gridView(c, p.db, p.T, p.nb, { range: p.range, xMax: p.T * p.H / P13.SR, yMax: 4, xLabel: '时间 (s)', yLabel: '频率 (kHz)', segs: S.clip.segs.map((g) => ({ x0: g.t0, x1: g.t1, label: g.label })), marks: [{ x: S.selT, color: LM.C.cyan }], title: `N = ${p.N}（${(1000 * p.N / P13.SR).toFixed(0)} ms），H = ${p.H}（${(1000 * p.H / P13.SR).toFixed(1)} ms）→ ${p.T} 帧 × ${p.nb} 个频率 bin · Δf = ${(P13.SR / p.N).toFixed(1)} Hz · 时长 ${dur.toFixed(2)} s`, smooth: p.N >= 256 });
                    },
                    probe: (x, y, S, PL, info) => { const L = S._spLay; if (!L) return ''; const [t, f] = L.inv(info.dx, info.dy), p = S.sp, ti = Math.round(t * P13.SR / p.H), k = Math.round(f * 1000 * p.N / P13.SR); if (ti < 0 || ti >= p.T || k < 0 || k >= p.nb) return ''; return `t = ${t.toFixed(3)} s，f = ${(f * 1000).toFixed(0)} Hz\n${p.db[ti * p.nb + k].toFixed(1)} dB`; },
                    pick: (x, y, S, PL, info) => { const L = S._spLay; if (!L) return; const [t] = L.inv(info.dx, info.dy); S.selT = PL.clamp(t, 0, S.clip.L / P13.SR - 0.01); return 'frame'; },
                },
                {
                    id: 'frameWave', label: '选中的一帧：加窗前后', sub: '灰 = 原始样本 · 蓝 = 乘上 Hann 窗之后（两端压到 0，减少频谱泄漏）', size: [360, 230],
                    render: (c, S) => {
                        const p = S.sp, t0 = Math.round(S.selT * P13.SR / p.H) * p.H - p.N / 2, w = P13.hann(p.N), xs = [], a = [], b = [];
                        for (let n = 0; n < p.N; n++) { const i = t0 + n, v = i >= 0 && i < S.clip.L ? S.clip.x[i] : 0; xs.push(1000 * n / P13.SR); a.push(v); b.push(v * w[n]); }
                        const pk = Math.max(0.05, ...a.map(Math.abs)) * 1.1;
                        P13.chart(c, { x: [0, 1000 * p.N / P13.SR], y: [-pk, pk], xLabel: '帧内时间 (ms)', title: `t = ${S.selT.toFixed(3)} s 处的一帧（N = ${p.N}）`, series: [{ x: xs, y: a, color: '#5b6068', width: 1.2 }, { x: xs, y: b, color: LM.C.blue, width: 1.6 }] });
                    },
                },
                {
                    id: 'frameMag', label: '这一帧的频谱 |X[t, k]|（dB）', sub: '横轴 0–4 kHz（奈奎斯特）· 谐波 = 等间距的峰，元音的共振峰 = 峰的包络', size: [360, 230],
                    render: (c, S) => {
                        const p = S.sp, ti = Math.min(p.T - 1, Math.round(S.selT * P13.SR / p.H)), fx = [], y = [];
                        for (let k = 0; k < p.nb; k++) { fx.push(k * P13.SR / p.N); y.push(p.db[ti * p.nb + k]); }
                        P13.chart(c, { x: [0, 4000], y: [p.range[1] - 80, p.range[1] + 3], xLabel: '频率 (Hz)', yLabel: 'dB', xFmt: (v) => (v ? v / 1000 + 'k' : '0'), series: [{ x: fx, y, color: LM.C.orange, width: 1.5 }], title: `第 ${ti} 帧 · ${p.nb} 个 bin，间隔 ${(P13.SR / p.N).toFixed(1)} Hz` });
                    },
                },
                {
                    id: 'trio', needs: ['trio'], label: '同一段声音，三种窗长', sub: '固定显示 0.5–1.1 s（旋律 G4 C5 + 啁啾开头）和 2.2–2.65 s（噪声 + 三下咔嗒）· 窗短 → 时间准、频率糊；窗长 → 频率准、时间糊', size: [720, 300],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c), rows = S.trio, rh = h / 3;
                        rows.forEach((r, i) => {
                            const crop = (t0, t1) => { const a = Math.round(t0 * P13.SR / r.H), b = Math.min(r.T, Math.round(t1 * P13.SR / r.H)); return { d: r.db.subarray(a * r.nb, b * r.nb), n: b - a, dur: t1 - t0 }; };
                            const A = crop(0.5, 1.1), B = crop(2.2, 2.65);
                            P13.gridPanel(ctx, { x: 0, y: i * rh, w: w * 0.56, h: rh }, A.d, A.n, r.nb, { range: r.range, xMax: A.dur, yMax: 4, legend: false, padR: 6, padB: i === 2 ? 18 : 6, xTicks: i === 2, xFmt: (v) => (v + 0.5).toFixed(1), title: `N = ${r.N}（${1000 * r.N / P13.SR} ms）· Δf = ${(P13.SR / r.N).toFixed(1)} Hz`, smooth: true });
                            P13.gridPanel(ctx, { x: w * 0.56, y: i * rh, w: w * 0.44, h: rh }, B.d, B.n, r.nb, { range: r.range, xMax: B.dur, yMax: 4, legend: false, padL: 10, padR: 6, padB: i === 2 ? 18 : 6, xTicks: i === 2, yTicks: false, xFmt: (v) => (v + 2.2).toFixed(1), smooth: true, title: '噪声 + 咔嗒 ×3', titleColor: '#8a8f98' });
                        });
                    },
                },
            ],
            play: [['clip', '演示片段']],
            notes: '把 <b>窗长 N</b> 调到 1024：旋律的每条谐波都是一根细线，但咔嗒（最右边）变成一个看不出有三下的宽块、音符之间的边界也糊了；调到 64：三下咔嗒是三根竖线，谐波却连成了宽带。<b>跳长</b>只改变时间轴上采样得多密（帧数），不改变每一帧看得多清楚。点击频谱图选不同的帧：旋律帧的频谱是一排等间距的尖峰（基频的整数倍），元音 a 的峰被几个共振峰「托」起来，噪声帧则是平的。',
            warn: '帧 t 的中心在 t·H（center = true，两端补 N/2 个零，与 torch.stft / librosa 的默认一致，但它们默认用反射补齐，这里补零）。频谱图一般用功率（|X|²）的 dB；也有画幅度 |X| 的（20·log10），两者差 2 倍。窗函数决定了频谱泄漏：矩形窗的旁瓣只低 13 dB，Hann 窗低 31 dB。',
        },
        // ─────────────────────────────── 03 Mel
        {
            id: 'mel', title: 'Mel 频谱：按人耳的刻度合并频率，再取对数', tag: 'MEL',
            desc: '人耳对低频的变化敏感、对高频迟钝：100 Hz 和 200 Hz 听起来差很多，3100 Hz 和 3200 Hz 几乎一样。<strong>mel 刻度</strong>就是按感知把频率重新标号（低频近似线性、高频近似对数）。mel 滤波器组是一排在 mel 刻度上等间距、相邻重叠一半的<strong>三角形</strong>：每个三角形把它覆盖的 FFT bin 按权重加起来，得到一个 mel 通道的能量。于是每帧从 N/2 + 1 个线性频率 bin 压缩成几十个 mel 通道（Whisper 用 80 个，这里默认 40 个）。最后取<strong>对数</strong>：能量跨好几个数量级，取 log 后小能量的细节也能被网络看见，而且响度变化变成了加法偏移。log-mel 频谱就是今天几乎所有语音模型的输入（Whisper、TTS 的声学模型、AudioLDM）。',
            formula: [
                'mel(f) = 2595 · log10(1 + f / 700)   // HTK 公式（torchaudio 默认）；librosa / Whisper 用 Slaney 版：1 kHz 以下线性',
                'M[t, m] = Σ_k W[m, k] · |X[t, k]|^2      W[m, ·] = 第 m 个三角形（中心在 mel 刻度上等间距）',
                'logmel[t, m] = 10 · log10(M[t, m] + 10^{−10})   // 能量 → dB',
            ],
            code: {
                title: 'C++ · hzToMel / melToHz / melFilterbank / logMel', src: `
                float hzToMel(float f) { return 2595.f * std::log10(1.f + f / 700.f); }
                float melToHz(float m) { return 700.f * (std::pow(10.f, m / 2595.f) - 1.f); }

                // nMels 个三角滤波器：W[m·nb + k]，峰值 1，中心在 mel 刻度上等间距（0 … SR/2）
                struct MelBank { int nMels = 0, nb = 0; std::vector<float> W; };

                MelBank melFilterbank(int N, int nMels) {
                    MelBank B; B.nMels = nMels; B.nb = N / 2 + 1; B.W = floats(nMels * B.nb);
                    float m1 = hzToMel(SR / 2.f);
                    for (int m = 0; m < nMels; m++) {
                        float fl = melToHz(m1 * m / (nMels + 1)), fc = melToHz(m1 * (m + 1) / (nMels + 1)), fr = melToHz(m1 * (m + 2) / (nMels + 1));
                        for (int k = 0; k < B.nb; k++) {
                            float f = (float)k * SR / N, v = 0;
                            if (f > fl && f <= fc) v = (f - fl) / (fc - fl);       // 上升沿
                            else if (f > fc && f < fr) v = (fr - f) / (fr - fc);  // 下降沿
                            B.W[m * B.nb + k] = v;
                        }
                    }
                    return B;
                }

                // log-mel 频谱 [T × nMels]：每帧的功率谱乘滤波器组，再取 10·log10
                std::vector<float> logMel(const Spec& S, const MelBank& B) {
                    std::vector<float> out = floats(S.T * B.nMels);
                    for (int t = 0; t < S.T; t++)
                        for (int m = 0; m < B.nMels; m++) {
                            double e = 0;
                            for (int k = 0; k < S.nb; k++) {
                                int i = t * S.nb + k;
                                e += B.W[m * B.nb + k] * (S.re[i] * S.re[i] + S.im[i] * S.im[i]);
                            }
                            out[t * B.nMels + m] = 10.f * (float)std::log10(e + 1e-10);
                        }
                    return out;
                }`,
            },
            previews: [
                {
                    id: 'melbank', label: 'mel 滤波器组', sub: '每个三角形 = 一个 mel 通道 · 越往高频越宽 · 底部短线 = FFT bin 的位置（窗长 N 决定）', size: [720, 200],
                    render: (c, S) => {
                        const b = S.sp.bank, series = [];
                        b.rows.forEach((r, m) => { const xs = [], ys = []; if (r.k0 > 0) { xs.push((r.k0 - 1) * P13.SR / b.N); ys.push(0); } r.w.forEach((v, j) => { xs.push((r.k0 + j) * P13.SR / b.N); ys.push(v); }); xs.push((r.k0 + r.w.length) * P13.SR / b.N); ys.push(0); series.push({ x: xs, y: ys, color: LM.SERIES[m % 4], width: 1.3 }); });
                        const o = P13.chart(c, { x: [0, 4000], y: [0, 1.12], xLabel: '频率 (Hz)', xFmt: (v) => (v ? v / 1000 + 'k' : '0'), yTicks: [0, 0.5, 1], series, title: `${b.nMels} 个三角滤波器 · N = ${b.N}：${b.nb} 个 bin，间隔 ${(P13.SR / b.N).toFixed(1)} Hz` + (b.empty ? ` · ⚠ ${b.empty} 个滤波器里一个 bin 都没有（空滤波器）` : '') });
                        o.ctx.fillStyle = 'rgba(200,204,212,0.5)'; for (let k = 0; k < b.nb; k++) o.ctx.fillRect(o.X(k * P13.SR / b.N), o.padT + o.ph - 4, 1, 4);
                    },
                },
                {
                    id: 'melspec', label: 'log-mel 频谱', sub: '纵轴 = mel 通道（标注中心频率）· 低频通道窄、分得细，高频通道宽 · 同样的演示片段、同样的帧', large: true, size: [720, 250],
                    render: (c, S) => {
                        const p = S.sp, nm = p.bank.nMels, fc = p.bank.fc, ticks = [0, Math.floor(nm / 4), Math.floor(nm / 2), Math.floor(3 * nm / 4), nm - 1];
                        S._melLay = P13.gridView(c, p.mdb, p.T, nm, { range: p.mrange, xMax: p.T * p.H / P13.SR, yMax: nm, xLabel: '时间 (s)', yLabel: 'mel 通道', yTickVals: ticks.map((t) => t + 0.5), yFmt: (v) => `${Math.round(fc[Math.floor(v)])}Hz`, padL: 56, segs: S.clip.segs.map((g) => ({ x0: g.t0, x1: g.t1, label: g.label })), marks: [{ x: S.selT, color: LM.C.cyan }], title: `${p.T} 帧 × ${nm} 个 mel 通道（线性频谱是 ${p.nb} 个 bin：压缩 ${(p.nb / nm).toFixed(1)} 倍）`, smooth: false });
                    },
                    pick: (x, y, S, PL, info) => { const L = S._melLay; if (!L) return; const [t] = L.inv(info.dx, info.dy); S.selT = PL.clamp(t, 0, S.clip.L / P13.SR - 0.01); return 'frame'; },
                },
                {
                    id: 'melcurve', label: 'mel 刻度', sub: '蓝 = mel(f)（HTK）· 点 = 当前各滤波器的中心 · 灰虚线 = 线性刻度（对照）', size: [360, 220],
                    render: (c, S) => {
                        const xs = [], ys = [], lin = []; for (let i = 0; i <= 100; i++) { const f = 40 * i; xs.push(f); ys.push(P13.hz2mel(f)); lin.push(f * P13.hz2mel(4000) / 4000); }
                        const fc = S.sp.bank.fc;
                        P13.chart(c, { x: [0, 4000], y: [0, 2200], xLabel: '频率 (Hz)', yLabel: 'mel', xFmt: (v) => (v ? v / 1000 + 'k' : '0'), series: [{ x: xs, y: lin, color: '#5b6068', dash: [4, 3] }, { x: xs, y: ys, color: LM.C.blue, width: 2 }, { x: fc, y: fc.map(P13.hz2mel), color: LM.C.orange, line: false, dots: 2.5 }] });
                    },
                },
                {
                    id: 'melinv', needs: ['glmel'], label: '从 mel 还原线性频谱（选中的帧）', sub: '灰 = 真实频谱 · 蓝 = 只知道 mel 能量时的最佳非负估计（非负最小二乘）· 高频的谐波细节丢了', size: [360, 220],
                    render: (c, S) => {
                        const p = S.sp, ti = Math.min(p.T - 1, Math.round(S.selT * P13.SR / p.H)), fx = [], a = [], b = [];
                        for (let k = 0; k < p.nb; k++) { fx.push(k * P13.SR / p.N); a.push(p.db[ti * p.nb + k]); b.push(S.gm.Pdb[ti * p.nb + k]); }
                        P13.chart(c, { x: [0, 4000], y: [p.range[1] - 80, p.range[1] + 3], xLabel: '频率 (Hz)', yLabel: 'dB', xFmt: (v) => (v ? v / 1000 + 'k' : '0'), series: [{ x: fx, y: a, color: '#6a6f78', width: 1.3 }, { x: fx, y: b, color: LM.C.blue, width: 1.6 }], title: `整段的相对误差 ‖A − Â‖/‖A‖ = ${S.gm.invErr.toFixed(2)}` });
                    },
                },
                {
                    id: 'logcmp', label: '为什么取对数（选中的帧）', sub: '橙 = 线性能量 / 最大值 · 蓝 = dB 映射到 [0, 1]（最响的 80 dB）· 线性刻度下只有一两个通道看得见', size: [360, 220],
                    render: (c, S) => {
                        const p = S.sp, nm = p.bank.nMels, ti = Math.min(p.T - 1, Math.round(S.selT * P13.SR / p.H)), row = p.mel.subarray(ti * nm, (ti + 1) * nm), mx = Math.max(...row) || 1, xs = Array.from(row, (_, i) => i);
                        const top = p.mrange[1];
                        P13.chart(c, { x: [0, nm - 1], y: [0, 1.05], xLabel: 'mel 通道', yTicks: [0, 0.5, 1], series: [{ x: xs, y: Array.from(row, (v) => v / mx), color: LM.C.orange, width: 2, dots: 2 }, { x: xs, y: Array.from(row, (v) => Math.max(0, (10 * Math.log10(v + 1e-10) - top + 80) / 80)), color: LM.C.blue, width: 2, dots: 2 }], legend: [{ label: '线性', color: LM.C.orange }, { label: 'log（dB）', color: LM.C.blue }] });
                    },
                },
            ],
            notes: '<b>mel 通道数</b>从 16 调到 80：log-mel 图越来越接近线性频谱图，低频谐波也逐渐分得开。把 <b>窗长 N</b> 调到 64 再选 64 或 80 个通道：低频的三角形窄于 bin 间隔（125 Hz），出现空滤波器（标题会提示），对应的 mel 通道永远是 0——librosa 会对此给出警告。右下两图跟着频谱图上选中的帧（点击频谱图 / log-mel 图选帧）。',
            warn: '不同库的 mel 约定不一样：HTK 公式 vs Slaney 公式（librosa 默认、Whisper 用它）；三角形峰值为 1（这里、torchaudio 默认 norm=None）vs 按带宽归一化成等面积（Slaney norm）；功率谱 vs 幅度谱；log10 vs 自然对数、dB 的下限。模型和特征提取必须用完全相同的约定，否则训练好的模型拿到的输入分布就变了——这是部署时最常见的 bug 之一。',
        },
        // ─────────────────────────────── 04 Griffin-Lim
        {
            id: 'griffin', title: '从频谱回到波形：相位去哪了？Griffin-Lim 与神经声码器', tag: 'VOCODER',
            desc: '频谱图只保留了幅度 |X|，<strong>相位</strong>被扔掉了；而生成模型（TTS 的声学模型、AudioLDM）输出的正是（mel）频谱图。要变回声音，必须猜出一个相位，使得逆 STFT 出来的波形，再做 STFT 时幅度恰好等于目标——这个问题叫<strong>相位重建</strong>。STFT 帧之间有重叠，所以不是随便一组复数谱都对应某个真实波形。<strong>Griffin-Lim</strong>（1984）交替做两件事：① 逆 STFT 得到波形（投影到「真实存在的信号」），② 再 STFT 后把幅度换回目标、只保留相位（投影到「幅度正确」）。每次迭代谱收敛度都不会变差。快速 Griffin-Lim（Perraudin 等 2013）加了一个 0.99 的动量，收敛快得多（librosa / torchaudio 的默认）。今天的系统用<strong>神经声码器</strong>（WaveNet、HiFi-GAN…）直接从 mel 生成波形，质量比 Griffin-Lim 好得多，但 Griffin-Lim 不需要训练，是最好的基线。',
            formula: [
                'x_i = iSTFT(X_i)      Y_i = STFT(x_i)      X_{i+1} = A · Y_i / |Y_i|   // A = 目标幅度，只换相位',
                '谱收敛度 SC_i = ‖ |Y_i| − A ‖_F / ‖A‖_F   // Griffin & Lim 证明它单调不增',
                '快速 GL：T_i = P(X_i)，X_{i+1} = T_i + α·(T_i − T_{i−1})，α = 0.99   // Perraudin, Balazs & Søndergaard 2013',
            ],
            code: {
                title: 'C++ · griffinLim / spectralConvergence', src: `
                // 谱收敛度：‖|STFT(x)| − A‖ / ‖A‖
                float spectralConvergence(const Spec& Y, const std::vector<float>& A) {
                    double e = 0, a = 0;
                    for (size_t i = 0; i < A.size(); i++) { double m = std::hypot(Y.re[i], Y.im[i]); e += (m - A[i]) * (m - A[i]); a += A[i] * A[i]; }
                    return (float)std::sqrt(e / a);
                }

                // Griffin-Lim：已知幅度 A[T × nb]，从随机相位出发；momentum = 0 是原始算法，0.99 是快速 Griffin-Lim
                std::vector<float> griffinLim(const std::vector<float>& A, int T, int N, int H, int L, int iters, float momentum, Rng& r) {
                    Spec X; X.T = T; X.N = N; X.H = H; X.nb = N / 2 + 1; X.re = floats(T * X.nb); X.im = floats(T * X.nb);
                    for (int i = 0; i < T * X.nb; i++) { double ph = 2 * kPi * uniform(r); X.re[i] = A[i] * (float)std::cos(ph); X.im[i] = A[i] * (float)std::sin(ph); }
                    std::vector<float> prevRe = floats(T * X.nb), prevIm = floats(T * X.nb);
                    std::vector<float> x = istft(X, L);
                    for (int it = 0; it < iters; it++) {
                        Spec Y = stft(x, N, H);                                       // 当前波形的 STFT
                        for (int i = 0; i < T * X.nb; i++) {
                            float m = std::hypot(Y.re[i], Y.im[i]) + 1e-12f;
                            float pr = A[i] * Y.re[i] / m, pi = A[i] * Y.im[i] / m;   // 幅度换回 A，保留相位
                            X.re[i] = pr + momentum * (pr - prevRe[i]); X.im[i] = pi + momentum * (pi - prevIm[i]);
                            prevRe[i] = pr; prevIm[i] = pi;
                        }
                        x = istft(X, L);                                              // 投影回「真实存在的波形」
                    }
                    return x;
                }`,
            },
            previews: [
                {
                    id: 'glsc', needs: ['gl', 'glmel'], live: true, label: '谱收敛度随迭代', sub: '对数纵轴 · 蓝 = Griffin-Lim · 橙 = 快速 Griffin-Lim（动量 0.99）· 绿虚线 = 只给 log-mel 时（还原线性谱 + 快速 GL 32 次）离真实幅度的距离', large: true, size: [720, 230],
                    render: (c, S) => {
                        const G = S.gl; if (!G) return;
                        const n = G.g.sc.length, xs = Array.from({ length: n }, (_, i) => i + 1);
                        P13.chart(c, {
                            x: [0, G.max], y: [0.005, 1.5], logY: true, xLabel: '迭代次数', title: `第 ${G.g.it} / ${G.max} 次迭代 · SC：GL ${LM.fmt(G.g.sc[n - 1], 3)} · 快速 GL ${LM.fmt(G.fg.sc[n - 1], 3)}`, tx: P13.LIVE_X,
                            series: [{ x: xs, y: G.g.sc, color: LM.C.blue, width: 2 }, { x: xs, y: G.fg.sc, color: LM.C.orange, width: 2 }],
                            refs: S.gm ? [{ y: S.gm.invErr, label: `从 mel 出发：${S.gm.invErr.toFixed(2)}`, color: LM.C.green }] : [],
                            legend: [{ label: 'Griffin-Lim', color: LM.C.blue }, { label: '快速 GL（α = 0.99）', color: LM.C.orange }], legendAt: 'bl',
                        });
                    },
                },
                {
                    id: 'glspec', needs: ['gl'], live: true, label: '快速 GL 的重建：频谱图', sub: '与第 02 步的原图几乎一样（同样的 dB 范围）', size: [360, 230],
                    render: (c, S) => { const G = S.gl, p = S.sp; if (!G || !G.db) return; P13.gridView(c, G.db, p.T, p.nb, { range: p.range, xMax: p.T * p.H / P13.SR, yMax: 4, xLabel: '时间 (s)', yLabel: 'kHz', legend: false, padR: 10, title: `第 ${G.fg.it} 次迭代`, tx: P13.LIVE_X, smooth: p.N >= 256 }); },
                },
                {
                    id: 'glwave', needs: ['gl'], live: true, label: '波形对比：元音 a 的 25 ms', sub: '灰 = 原始 · 橙 = 快速 GL 的重建——幅度谱相同，波形却完全不同（相位不同）', size: [360, 230],
                    render: (c, S) => {
                        const G = S.gl; if (!G) return;
                        const i0 = Math.round(1.72 * P13.SR), n = 200, ts = [], a = [], b = []; for (let i = 0; i < n; i++) { ts.push(1000 * i / P13.SR); a.push(S.clip.x[i0 + i]); b.push(G.fg.x[i0 + i]); }
                        const pk = Math.max(...a.map(Math.abs), ...b.map(Math.abs)) * 1.1 || 1;
                        P13.chart(c, { x: [0, 25], y: [-pk, pk], xLabel: '时间 (ms)', title: `波形 SNR = ${P13.snrDb(a, b).toFixed(1)} dB（≈ 0：逐点对不上）`, tx: P13.LIVE_X, series: [{ x: ts, y: a, color: '#6a6f78', width: 2.2 }, { x: ts, y: b, color: LM.C.orange, width: 1.5 }] });
                    },
                },
            ],
            play: [['clip', '原始'], ['gl0', '零相位（第 0 次）'], ['gl', 'Griffin-Lim（当前）'], ['fgl', '快速 GL（当前）'], ['glmel', '只用 log-mel 重建']],
            notes: '默认窗长 256、跳长 N/4：原始 Griffin-Lim（蓝）几十次迭代后谱收敛度还在 0.1 以上，快速 GL（橙）同样的次数低得多。依次播放「零相位」→「Griffin-Lim」→「快速 GL」→「原始」：零相位（每个 bin 相位都取 0）是嗡嗡的金属声，几十次迭代后已经很接近原声，只剩一点「相位感」（phasiness）。把 <b>跳长</b> 调成 N/2：重叠太少，约束不够，收敛明显变差；调成 N/8 更好但慢一倍。「从 mel 出发」（绿线）差得多：mel 丢掉了高频的谐波细节，相位再准也补不回来——这正是神经声码器要学的东西：从 mel 里「脑补」出细节和相位。',
            warn: '谱收敛度只衡量幅度：GL 收敛后波形和原波形逐点对比几乎没有关系（右下图的 SNR ≈ 0 dB），但人耳对相位不敏感，所以听起来很像。初始相位是随机的（换随机种子会得到不同但同样好的重建）。真实 TTS 里 Griffin-Lim 的音质明显差于 HiFi-GAN 一类的神经声码器——早期 Tacotron（2017）就用 Griffin-Lim，Tacotron 2（2018）换成了 WaveNet 声码器。',
        },

        // ─────────────────────────────── 05 编解码器
        {
            id: 'codec', title: '神经音频编解码器：帧特征 + 残差向量量化（RVQ）= 音频 token', tag: 'CODEC',
            desc: '要让语言模型「说」音频，先得把波形变成<strong>离散 token</strong>，并且 token 还能解码回声音——这就是神经音频编解码器（SoundStream、EnCodec）。结构是三段：<strong>编码器</strong>把波形压成低帧率的向量序列（这里每 32 ms 一帧、256 维）；<strong>量化器</strong>把每帧向量变成几个整数；<strong>解码器</strong>把整数变回波形。量化用的是第 09 篇 VQ 的升级版——<strong>残差向量量化</strong>：第 1 层码本找最近的码字，剩下的误差（残差）交给第 2 层再量化，依此类推。每多一层，码率增加 log₂K bit/帧，误差继续变小；解码时只用前几层，就得到低码率版本——一个模型支持多种码率。码本用 <strong>EMA k-means</strong> 训练：每个码字移向分给它的那些残差的滑动平均，很久没被选中的「死码」换成随机样本（SoundStream / EnCodec 的做法）。下面的码本在你的浏览器里实时训练，所有 SNR 都是真实测出来的。',
            formula: [
                'r_0 = z_t      i_l = argmin_k ‖r_{l−1} − C_l[k]‖^2      r_l = r_{l−1} − C_l[i_l]   // l = 1 … NQ',
                'ẑ_t = Σ_{l ≤ n_q} C_l[i_l]      码率 = 帧率 × n_q × log₂K   // 31.25 帧/秒 × 4 层 × 6 bit = 750 bps',
                'N_k ← γ·N_k + (1−γ)·n_k      m_k ← γ·m_k + (1−γ)·Σ_{r→k} r      C[k] = m_k / N_k   // EMA k-means，γ = 0.98',
                'z_t = MDCT(x[(t−1)·M … (t+1)·M))，M = 256   // 本页的「编码器」是固定的正交变换，见下面的说明',
            ],
            code: {
                title: 'C++ · mdctFrame / imdctFrame / RVQ（rvqEncode / rvqDecode / rvqTrainStep）/ bitrateKbps', src: `
                // MDCT：一帧 2M 个样本（正弦窗）→ M 个系数；按定义直接算（页面里的 JS 用一次 2M 点 FFT 算同一个东西）
                std::vector<float> mdctFrame(const std::vector<float>& seg, int M) {
                    std::vector<float> X = floats(M);
                    for (int k = 0; k < M; k++) {
                        double s = 0;
                        for (int n = 0; n < 2 * M; n++) s += std::sin(kPi * (n + 0.5) / (2 * M)) * seg[n] * std::cos(kPi / M * (n + 0.5 + M / 2.0) * (k + 0.5));
                        X[k] = (float)(std::sqrt(2.0 / M) * s);
                    }
                    return X;
                }

                // 逆 MDCT：输出 2M 个样本，相邻帧（相隔 M）重叠相加就完美重建（时域混叠抵消 TDAC）
                std::vector<float> imdctFrame(const std::vector<float>& X, int M) {
                    std::vector<float> y = floats(2 * M);
                    for (int n = 0; n < 2 * M; n++) {
                        double s = 0;
                        for (int k = 0; k < M; k++) s += X[k] * std::cos(kPi / M * (n + 0.5 + M / 2.0) * (k + 0.5));
                        y[n] = (float)(std::sqrt(2.0 / M) * std::sin(kPi * (n + 0.5) / (2 * M)) * s);
                    }
                    return y;
                }

                // 残差向量量化：NQ 层、每层 K 个 D 维码字；C[l] = [K × D]，sum / cnt = EMA 统计量
                struct RVQ { int D = 0, K = 0, NQ = 0, inited = 0; float gamma = 0.98f; std::vector<Tensor> C, sum, cnt; };

                RVQ makeRVQ(int D, int K, int NQ) {
                    RVQ q; q.D = D; q.K = K; q.NQ = NQ;
                    for (int l = 0; l < NQ; l++) { q.C.push_back(zeros(K, D)); q.sum.push_back(zeros(K, D)); q.cnt.push_back(zeros(1, K)); }
                    return q;
                }

                // 最近的码字：argmin_k ‖r − c_k‖²
                int nearestCode(const Tensor& C, const std::vector<float>& r) {
                    int best = 0; double bv = 1e300;
                    for (int k = 0; k < C.rows; k++) {
                        double d = 0;
                        for (int j = 0; j < C.cols; j++) { double e = r[j] - C.at(k, j); d += e * e; }
                        if (d < bv) { bv = d; best = k; }
                    }
                    return best;
                }

                // 编码一帧：每层选最近的码字、减掉它，残差交给下一层 → NQ 个整数（= 这一帧的 NQ 个 token）
                std::vector<int> rvqEncode(const RVQ& q, std::vector<float> r) {
                    std::vector<int> codes = ints(q.NQ);
                    for (int l = 0; l < q.NQ; l++) {
                        codes[l] = nearestCode(q.C[l], r);
                        for (int j = 0; j < q.D; j++) r[j] -= q.C[l].at(codes[l], j);
                    }
                    return codes;
                }

                // 解码：前 nq 层的码字相加（nq 越小码率越低）
                std::vector<float> rvqDecode(const RVQ& q, const std::vector<int>& codes, int nq) {
                    std::vector<float> z = floats(q.D);
                    for (int l = 0; l < nq; l++) for (int j = 0; j < q.D; j++) z[j] += q.C[l].at(codes[l], j);
                    return z;
                }

                // 一步 mini-batch EMA k-means：B 帧的残差逐层穿过「当前」码本，每层更新统计量；死码换成 batch 里的随机残差
                void rvqTrainStep(RVQ& q, const Tensor& X, int B, Rng& r) {
                    Tensor R = zeros(B, q.D);
                    for (int b = 0; b < B; b++) { int t = randint(r, X.rows); for (int j = 0; j < q.D; j++) R.at(b, j) = X.at(t, j); }
                    for (int l = 0; l < q.NQ; l++) {
                        Tensor& C = q.C[l];
                        if (q.inited <= l) {                                          // 第一次：用 batch 里的随机残差初始化
                            for (int k = 0; k < q.K; k++) { int b = randint(r, B); for (int j = 0; j < q.D; j++) { C.at(k, j) = R.at(b, j); q.sum[l].at(k, j) = R.at(b, j); } q.cnt[l].v[k] = 1; }
                            q.inited = l + 1;
                        }
                        std::vector<int> asg = ints(B);
                        Tensor nk = zeros(1, q.K), sk = zeros(q.K, q.D);
                        for (int b = 0; b < B; b++) {
                            std::vector<float> rb = floats(q.D);
                            for (int j = 0; j < q.D; j++) rb[j] = R.at(b, j);
                            asg[b] = nearestCode(C, rb); nk.v[asg[b]] += 1;
                            for (int j = 0; j < q.D; j++) sk.at(asg[b], j) += rb[j];
                        }
                        for (int b = 0; b < B; b++) for (int j = 0; j < q.D; j++) R.at(b, j) -= C.at(asg[b], j);   // 残差给下一层
                        for (int k = 0; k < q.K; k++) {
                            q.cnt[l].v[k] = q.gamma * q.cnt[l].v[k] + (1 - q.gamma) * nk.v[k];
                            for (int j = 0; j < q.D; j++) q.sum[l].at(k, j) = q.gamma * q.sum[l].at(k, j) + (1 - q.gamma) * sk.at(k, j);
                            if (q.cnt[l].v[k] < 0.02f * B / q.K) {                    // 死码：换成一个随机样本（加回它被减掉的码字 = 量化前的值）
                                int b = randint(r, B);
                                for (int j = 0; j < q.D; j++) q.sum[l].at(k, j) = R.at(b, j) + C.at(asg[b], j);
                                q.cnt[l].v[k] = 1;
                            }
                        }
                        for (int k = 0; k < q.K; k++) for (int j = 0; j < q.D; j++) C.at(k, j) = q.sum[l].at(k, j) / q.cnt[l].v[k];
                    }
                }

                float bitrateKbps(float frameRate, int nq, int K) { return frameRate * nq * std::log2((float)K) / 1000.f; }`,
            },
            play: [['codecOrig', '原始（测试旋律 1）'], ['codec1', '1 层'], ['codec2', '2 层'], ['codec4', '4 层'], ['codec8', '8 层'], ['codecNq', '当前「解码层数」']],
            previews: [
                {
                    id: 'codecTok', needs: ['codec'], live: true, label: '音频 token：测试旋律的 8 层码', sub: '测试旋律 1 的前 2 秒（64 帧 × 8 层）· 每格 = 一个码字 id（颜色）· 亮的行 = 解码时用的前「解码层数」层', large: true, size: [720, 210],
                    render: (c, S) => {
                        const R = S.codec; if (!R || !R.last) return P13wait(c, R, 'codec');
                        const L = P13.tokenView(c, R.last.codes, S.cdata.EF, 8, 8, { title: `第 ${R.step} / ${P13.CODEC.STEPS} 步 · K = ${R.K}（每个码 ${Math.log2(R.K)} bit）· 用前 ${S.nq} 层 = 每帧 ${S.nq} 个 token = ${(P13.CODEC.FPS * S.nq * Math.log2(R.K) / 1000).toFixed(3)} kbps`, tx: P13.LIVE_X, hiRows: S.nq, cellH: 18 });
                        L.ctx.font = LM.FONT(11); L.ctx.fillStyle = LM.C.gray; L.ctx.textAlign = 'left'; L.ctx.textBaseline = 'top';
                        L.ctx.fillText(`帧率 ${P13.CODEC.FPS} 帧/秒（每帧 ${1000 / P13.CODEC.FPS} ms）· 原始 PCM 8 kHz × 16 bit = 128 kbps · 每 8 帧 = 一拍（一个音符）`, L.padL, L.y + 6);
                    },
                },
                {
                    id: 'codecRD', needs: ['codec'], live: true, label: '码率–失真：用几层 vs 信噪比', sub: '评估片段上实测的 SNR（MDCT 正交 ⇒ 系数域 SNR = 波形 SNR）· 细线 = 更早的检查点 · 黄线 = 当前「解码层数」', size: [360, 250],
                    render: (c, S) => {
                        const R = S.codec; if (!R || !R.last) return P13wait(c, R, 'codec');
                        const xs = [1, 2, 3, 4, 5, 6, 7, 8], series = [];
                        [50, 150, 400, 800].forEach((st, i) => { const e = R.evals.find((q) => q.step === st); if (e && e.step < R.step) series.push({ x: xs, y: e.snr, color: '#3e5566', width: 1, alpha: 0.5 + 0.12 * i }); });
                        series.push({ x: xs, y: R.last.snr, color: LM.C.blue, width: 2.2, dots: 3 });
                        const kb = (n) => (P13.CODEC.FPS * n * Math.log2(R.K) / 1000).toFixed(2);
                        P13.chart(c, { x: [0.5, 8.5], y: [0, 40], xTicks: xs, xFmt: (v) => `${v}`, xLabel: `层数 n_q（每层 +${(P13.CODEC.FPS * Math.log2(R.K)).toFixed(0)} bps）`, yLabel: 'SNR (dB)', series, marks: [{ x: S.nq, label: `${S.nq} 层 · ${kb(S.nq)} kbps · ${R.last.snr[S.nq - 1].toFixed(1)} dB` }], title: `第 ${R.step} 步`, tx: P13.LIVE_X });
                    },
                },
                {
                    id: 'codecSpec', needs: ['codecView'], live: true, label: '原始 vs 解码（前 2 秒）', sub: '上 = 原始 · 下 = 用前「解码层数」层码本解码 · 同一个 dB 范围', size: [360, 250],
                    render: (c, S) => {
                        const v = S.cv, o = S.cvOrig; if (!v || !o) return;
                        const { ctx, w, h } = PL.draw.ctx(c), dur = (S.cdata.EF - 1) * P13.CODEC.M / P13.SR;
                        P13.gridPanel(ctx, { x: 0, y: 0, w, h: h / 2 }, o.db, o.T, o.nb, { range: o.range, xMax: dur, yMax: 4, legend: false, padR: 6, padB: 4, xTicks: false, title: '原始', smooth: true });
                        P13.gridPanel(ctx, { x: 0, y: h / 2, w, h: h / 2 }, v.db, v.T, v.nb, { range: o.range, xMax: dur, yMax: 4, legend: false, padR: 6, padB: 16, title: `${S.nq} 层 · SNR ${S.codec.last.snr[S.nq - 1].toFixed(1)} dB`, smooth: true });
                    },
                },
                {
                    id: 'codecCurve', needs: ['codec'], live: true, label: '训练曲线：每层之后剩下的残差', sub: 'batch 上的 −10·log10(‖r_l‖² / ‖z‖²)，越高越好 · 8 条线 = 第 1 … 8 层（滑动平均）', size: [360, 230],
                    render: (c, S) => {
                        const R = S.codec; if (!R) return P13wait(c, R, 'codec');
                        const n = R.curve.length, series = [];
                        for (let l = 0; l < 8; l++) { const y = LM.ema(R.curve.map((q) => -10 * Math.log10(q[l] + 1e-12)), 0.95); series.push({ y, color: `hsl(${200 - l * 22}, 60%, ${45 + l * 3}%)`, width: 1.4 }); }
                        P13.chart(c, { x: [0, Math.max(100, n)], y: [0, 45], xLabel: '训练步', yLabel: 'dB', series, title: n ? `第 ${n} 步：1 层 ${(-10 * Math.log10(R.curve[n - 1][0])).toFixed(1)} dB · 8 层 ${(-10 * Math.log10(R.curve[n - 1][7])).toFixed(1)} dB` : '初始化', tx: P13.LIVE_X });
                    },
                },
                {
                    id: 'codecUse', needs: ['codec'], live: true, label: '码本使用情况', sub: '柱 = 第 1 层每个码字在评估片段里被选中的次数 · 标题 = 8 层里每层用到的不同码字数（最少–最多）', size: [360, 230],
                    render: (c, S) => {
                        const R = S.codec; if (!R || !R.last) return P13wait(c, R, 'codec');
                        const h0 = Array.from(R.last.use[0]), used = R.last.use.map((h) => h.filter((v) => v > 0).length);
                        LM.draw.bars(c, h0, null, { colors: h0.map((_, k) => LM.tokColor(k * 7)), title: `第 ${R.step} 步 · 每层用到 ${Math.min(...used)}–${Math.max(...used)} 个不同的码字（共 ${R.K}）`, titleX: P13.LIVE_X, digits: false });
                    },
                },
            ],
            notes: '刚开始（第 0 步）码本是随机挑的样本，1 层的 SNR 只有几 dB；几百步后 1 层约 8 dB、4 层约 20 dB、8 层约 26 dB（默认 K = 64，Node 里用同一份代码跑满 1500 步、在 6 段测试旋律上实测：7.6 / 12.4 / 16.9 / 20.3 / 22.6 / 24.1 / 25.2 / 26.0 dB）。依次播放 1、2、4、8 层：1 层只听得出音高轮廓、带着嗡嗡的噪声，8 层已经很接近原声，而码率只有 1.5 kbps（原始 PCM 的 1/85）。<b>码本大小 K</b> 调到 128：每层多 1 bit，8 层的 SNR 到 37.0 dB；调到 16：只有 11.9 dB。这份数据很「规整」（只有 8 个音高、2 种音色），所以码本学得这么快；真实音乐和语音要大得多的码本（1024）和更多的层。',
            warn: '本页的「编码器 / 解码器」是固定的 MDCT（Opus 的 CELT 模式、AAC 等传统编解码器用的变换），没有训练：真实的 SoundStream / EnCodec 用卷积网络学这个变换（编码器步长 2·4·5·8 = 320 倍下采样，24 kHz → 75 帧/秒），和 RVQ 一起端到端训练，损失 = 重建 + 多尺度频谱 + 对抗（判别器），量化用直通估计（第 09 篇）把梯度传过去。SoundStream 训练时随机丢掉后面几层（quantizer dropout），让一个模型支持多种码率；这里的 RVQ 逐层贪心地量化残差，天然就能截断。这份合成数据里同一个音符每次出现的波形完全相同（起音相位归零、和帧对齐），真实录音没有这么理想。',
        },
        // ─────────────────────────────── 06 token LM
        {
            id: 'tokenlm', title: '音频 token 语言模型：delay pattern 与续写', tag: 'LM',
            desc: '有了 token，生成音频就变成了第 05 篇的老问题：<strong>预测下一个 token</strong>。麻烦在于每帧有 NQ 个 token（NQ 个码本）。全部摊平成一条序列（帧 1 的 4 个、帧 2 的 4 个…）最直接，但序列长 4 倍、注意力贵 16 倍。MusicGen 的 <strong>delay pattern</strong>：把第 k 个码本整体往后错 k 步，于是每个位置放 NQ 个 token，模型在每一步<strong>并行</strong>预测 NQ 个——序列长度不变；而第 k 个码本的第 t 帧，要到第 t + k 步才预测，那时同一帧的前 k 个（更粗的）码本已经确定，粗到细的依赖关系保住了一大半。这里的模型就是第 04 / 05 篇的迷你 GPT（2 层、4 头、d = 32），只是把「一张 embedding 表 + 一个输出头」换成 NQ 张表相加 + NQ 个头。训练分三个阶段，全在你的浏览器里跑：① 训练它自己的编解码器（与第 05 步同一个算法，4 层、K = 64）；② 冻结码本，给 48 段训练旋律（约 165 秒）打 token；③ 训练语言模型。最后用测试旋律的第一个音符当提示，让它续写 88 帧（约 2.8 秒），解码成声音。',
            formula: [
                '延迟排布：D[p, k] = q_k(p − k)      p − k < 0 时为 PAD   // 第 k 个码本晚 k 步',
                'h_p = GPT( Σ_k E_k[D[p, k]] + P[p] )      logits_k = h_p · W_k + b_k   // NQ 张 embedding 相加，NQ 个头并行',
                'L = (1/NQ) Σ_k CE(logits_k[p], D[p + 1, k])   // 下一个位置的 NQ 个 token；PAD 不计',
            ],
            code: {
                title: 'C++ · delayPattern / undelay / TokenLM（makeTokenLM / gptHidden / tokenLMLoss / tokenLMTrainStep / tokenLMGenerate）', src: `
                // 延迟排布：codes[t·NQ + k]（F 帧）→ [(F + NQ − 1) × NQ]，第 k 个码本整体晚 k 步，空位填 PAD = K
                std::vector<int> delayPattern(const std::vector<int>& codes, int F, int NQ, int K) {
                    int P = F + NQ - 1;
                    std::vector<int> D = ints(P * NQ, K);
                    for (int p = 0; p < P; p++)
                        for (int k = 0; k < NQ; k++) { int t = p - k; if (t >= 0 && t < F) D[p * NQ + k] = codes[t * NQ + k]; }
                    return D;
                }

                // 还原：第 t 帧第 k 个码本在位置 t + k
                std::vector<int> undelay(const std::vector<int>& D, int P, int NQ) {
                    int F = P - NQ + 1;
                    std::vector<int> codes = ints(F * NQ);
                    for (int t = 0; t < F; t++) for (int k = 0; k < NQ; k++) codes[t * NQ + k] = D[(t + k) * NQ + k];
                    return codes;
                }

                // 迷你 GPT 主干 + 每个码本一张 embedding 表（多 1 行给 PAD）和一个输出头
                struct TokenLM { GPT g; std::vector<Var> emb, Wh, bh; int NQ = 0, K = 0; std::vector<Var> params; std::vector<bool> decay; };

                TokenLM makeTokenLM(int NQ, int K, int T, uint32_t seed) {
                    TokenLM m; m.NQ = NQ; m.K = K;
                    GPTConfig c; c.V = 1; c.T = T;                                  // 词表用不上：输入输出都换成了每个码本自己的
                    m.g = makeGPT(c, seed);
                    Rng r = makeRng(seed * 31 + 7);
                    for (int k = 0; k < NQ; k++) {
                        m.emb.push_back(param(randn(K + 1, c.d, 0.02f, r)));
                        m.Wh.push_back(param(randn(c.d, K, 0.02f, r)));
                        m.bh.push_back(param(zeros(1, K)));
                    }
                    for (size_t i = 1; i < m.g.params.size(); i++) { m.params.push_back(m.g.params[i]); m.decay.push_back(m.g.decay[i]); }   // 跳过 wte
                    for (int k = 0; k < NQ; k++) {
                        m.params.push_back(m.emb[k]); m.decay.push_back(true);
                        m.params.push_back(m.Wh[k]); m.decay.push_back(true);
                        m.params.push_back(m.bh[k]); m.decay.push_back(false);
                    }
                    return m;
                }

                // gptForward 去掉最后的 LM head：返回最后一个 LayerNorm 之后的隐状态 [B·T × d]
                Var gptHidden(GPT& m, const Var& inputEmb, int B, int T) {
                    std::vector<int> pos = ints(B * T);
                    for (int i = 0; i < B * T; i++) pos[i] = i % T;
                    Var x = add(inputEmb, gatherRows(m.wpe, pos));
                    for (const Block& b : m.blocks) x = blockForward(b, x, B, T, m.cfg.H);
                    return layerNorm(x, m.lnfg, m.lnfb);
                }

                // 一段延迟排布的位置 [p0, p0 + n) → 输入 embedding（NQ 张表查出来相加）
                Var tokenLMInput(TokenLM& m, const std::vector<int>& seq, int p0, int n) {
                    Var inp;
                    for (int k = 0; k < m.NQ; k++) {
                        std::vector<int> ids = ints(n);
                        for (int i = 0; i < n; i++) ids[i] = seq[(p0 + i) * m.NQ + k];
                        Var e = embedding(ids, m.emb[k]);
                        inp = k == 0 ? e : add(inp, e);
                    }
                    return inp;
                }

                // 一个 batch（B 段长 T 的窗口，展平成 B·T 个位置）的平均交叉熵；y 里 −1 = PAD，不计损失
                Var tokenLMLoss(TokenLM& m, const std::vector<int>& x, const std::vector<int>& y, int B, int T) {
                    Var h = gptHidden(m.g, tokenLMInput(m, x, 0, B * T), B, T), loss;
                    for (int k = 0; k < m.NQ; k++) {
                        std::vector<int> tg = ints(B * T);
                        for (int i = 0; i < B * T; i++) tg[i] = y[i * m.NQ + k];
                        Var lk = crossEntropy(linear(h, m.Wh[k], m.bh[k]), tg);    // 第 k 个头预测第 k 个码本
                        loss = k == 0 ? lk : add(loss, lk);
                    }
                    return scale(loss, 1.f / m.NQ);
                }

                float tokenLMTrainStep(TokenLM& m, AdamW& opt, const std::vector<int>& x, const std::vector<int>& y, int B, int T, float lr) {
                    zeroGrad(m.params);
                    Var loss = tokenLMLoss(m, x, y, B, T);
                    backward(loss);
                    clipGradNorm(m.params, 1.0f);
                    adamwStep(opt, lr);
                    return loss->val.v[0];
                }

                // 续写：seq = 延迟排布的前 P0 个位置；known = 提示的原始码（F0 帧）。每步对最后 ≤ T 个位置做一次前向，NQ 个头同时采样
                // （页面里的 JS 用 KV cache 逐 token 计算，结果相同；上下文满了以后保留后半段重新编码）
                std::vector<int> tokenLMGenerate(TokenLM& m, std::vector<int> seq, int P0, int nNew, const std::vector<int>& known, int F0, float temperature, Rng& r) {
                    NoGrad ng;
                    int T = m.g.cfg.T;
                    for (int p = P0; p < P0 + nNew; p++) {
                        int w0 = std::max(0, p - T), n = p - w0;
                        Var h = gptHidden(m.g, tokenLMInput(m, seq, w0, n), 1, n);
                        Tensor last = zeros(1, h->val.cols);
                        for (int j = 0; j < h->val.cols; j++) last.v[j] = h->val.at(n - 1, j);
                        for (int k = 0; k < m.NQ; k++) {
                            int t = p - k;
                            if (t < 0) { seq.push_back(m.K); continue; }                   // 延迟排布开头的空位
                            if (t < F0) { seq.push_back(known[t * m.NQ + k]); continue; }  // 提示里已知的帧（延迟的尾巴）
                            std::vector<float> logits = linear(constant(last), m.Wh[k], m.bh[k])->val.v;
                            softmaxInPlace(logits, temperature);
                            seq.push_back(sampleIndex(logits, r));
                        }
                    }
                    return seq;
                }`,
            },
            play: [['lmPrompt', '提示（第 1 个音符）'], ['lmGen', '续写 A'], ['lmGen2', '续写 B（另一段提示）'], ['lmReal', '真实的后续（参照）'], ['lmRecon', '真实旋律经 4 层编解码']],
            previews: [
                {
                    id: 'delayGrid', needs: ['lm'], live: true, label: 'delay pattern：同一段真实 token 的两种排法', sub: '左 = 原始排布（每列一帧，4 行 = 4 个码本）· 右 = 延迟排布（第 k 行整体右移 k 格）· 格子里的数字 = 帧号 t，颜色 = 真实的码字 id · 黄框 = 模型在同一步并行预测的 4 个 token', large: true, size: [720, 210],
                    render: (c, S) => {
                        const R = S.lm, { ctx, w, h } = PL.draw.ctx(c), NQ = 4, F = 10, P = F + NQ - 1, has = R && R.phase >= 2, codes = has ? R.test[0].codes : null, pSel = 7;
                        const cell = Math.min(28, (w * 0.42 - 40) / F), ch = 26, y0 = 44;
                        const grid = (x0, cols, at, title) => {
                            ctx.font = LM.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(title, x0, 8);
                            for (let k = 0; k < NQ; k++) {
                                ctx.font = LM.FONT(11); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(`码本 ${k + 1}`, x0 + 46, y0 + k * ch + ch / 2);
                                for (let j = 0; j < cols; j++) {
                                    const t = at(j, k), x = x0 + 52 + j * cell, y = y0 + k * ch;
                                    if (t < 0 || t >= F) { ctx.fillStyle = '#15181d'; ctx.fillRect(x, y + 1, cell - 2, ch - 2); ctx.fillStyle = '#4a4f56'; ctx.font = LM.FONT(10); ctx.textAlign = 'center'; ctx.fillText('PAD', x + cell / 2 - 1, y + ch / 2); continue; }
                                    ctx.fillStyle = has ? LM.tokColor(codes[t * NQ + k] * 7 + k * 61) : '#2a3440'; ctx.fillRect(x, y + 1, cell - 2, ch - 2);
                                    ctx.fillStyle = '#eef0f3'; ctx.font = LM.FONT(11); ctx.textAlign = 'center'; ctx.fillText(String(t), x + cell / 2 - 1, y + ch / 2 + 0.5);
                                }
                            }
                            ctx.font = LM.FONT(11); ctx.fillStyle = LM.C.dim; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                            for (let j = 0; j < cols; j += 2) ctx.fillText(String(j), x0 + 52 + j * cell + cell / 2 - 4, y0 + NQ * ch + 3);
                        };
                        const xL = 4, xR = w * 0.47;
                        grid(xL, F, (j) => j, '原始排布：第 t 列 = 第 t 帧的 4 个码');
                        grid(xR, P, (j, k) => j - k, '延迟排布：第 p 列第 k 行 = q_k(p − k)');
                        ctx.strokeStyle = LM.C.yellow; ctx.lineWidth = 2; ctx.strokeRect(xR + 52 + pSel * cell - 1, y0 - 1, cell, NQ * ch + 2);
                        for (let k = 0; k < NQ; k++) { ctx.strokeRect(xL + 52 + (pSel - k) * cell + 1, y0 + k * ch + 2, cell - 4, ch - 4); }
                        ctx.font = LM.FONT(11); ctx.fillStyle = LM.C.yellow; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
                        ctx.fillText(`位置 p = ${pSel}：一步同时预测帧 ${pSel}、${pSel - 1}、${pSel - 2}、${pSel - 3} 的第 1、2、3、4 个码——预测帧 t 的第 k 个码时，它的前 k − 1 个码已经在前面的位置出现过`, 6, y0 + NQ * ch + 22);
                        ctx.fillStyle = has ? LM.C.gray : LM.C.orange;
                        ctx.fillText(has ? `颜色 = 测试旋律 1 前 ${F} 帧的真实 token（第 06 步自己的编解码器：4 层、K = ${P13.LMC.K}）· 序列长度只多 NQ − 1 = 3 个位置（摊平要 ×4）` : `还没有 token（训练阶段 ${R ? R.phase + 1 : 1} / 3${R && R.phase === 0 ? ' · 码本 ' + R.cstep + ' / ' + P13.LMC.CODEC_STEPS + ' 步' : R && R.phase === 1 ? ' · 打 token ' + R.tokPos + ' 帧' : ''}，见下方）：先只画排布方式`, 6, y0 + NQ * ch + 40);
                    },
                },
                {
                    id: 'lmCurve', needs: ['lm'], live: true, label: '训练进度与损失', sub: '对数纵轴 · 蓝 = 训练 loss（4 个码本平均，滑动平均）· 彩色点线 = 验证集上每个码本的交叉熵（测试旋律）', size: [360, 250],
                    render: (c, S) => {
                        const R = S.lm; if (!R) return P13wait(c, R, 'lm');
                        if (R.phase < 2) {
                            const lines = [[{ t: `        训练阶段 ${R.phase + 1} / 3`, color: LM.C.text, bold: true }]];
                            lines.push([{ t: '① 编解码器（EMA k-means，4 层 × 64）：', color: LM.C.gray }, { t: `${R.cstep} / ${P13.LMC.CODEC_STEPS} 步`, color: R.phase > 0 ? LM.C.green : LM.C.yellow }]);
                            lines.push([{ t: '② 冻结码本，给 48 段训练旋律打 token：', color: LM.C.gray }, { t: `${R.tokPos} / ${S.cdata.F} 帧`, color: R.phase > 1 ? LM.C.green : R.phase === 1 ? LM.C.yellow : LM.C.dim }]);
                            lines.push([{ t: '③ 训练延迟排布 LM：', color: LM.C.gray }, { t: '等待', color: LM.C.dim }]);
                            lines.push({ text: '（只在本步出现在屏幕上时训练；真实系统也是先训好编解码器、冻结，再训练语言模型）', color: LM.C.dim, px: 11 });
                            LM.draw.text(c, lines, { px: 12, lineGap: 9 });
                            return;
                        }
                        const ev = R.evals, cols = [LM.C.orange, LM.C.green, LM.C.purple, LM.C.cyan], series = [{ y: LM.ema(R.loss, 0.97), color: LM.C.blue, width: 2 }];
                        for (let k = 0; k < 4; k++) series.push({ x: ev.map((e) => e.step), y: ev.map((e) => e.val[k]), color: cols[k], width: 1.2, dots: 2, dash: [3, 2] });
                        const last = ev[ev.length - 1];
                        P13.chart(c, { x: [0, P13.LMC.STEPS], y: [0.003, 6], logY: true, xLabel: '训练步', series, title: `第 ${R.step} / ${P13.LMC.STEPS} 步 · val ${last ? last.val.map((v) => v.toFixed(2)).join(' / ') : '—'}`, tx: P13.LIVE_X, legend: [{ label: 'train', color: LM.C.blue }].concat(cols.map((cc, k) => ({ label: `码本 ${k + 1}`, color: cc, dash: [3, 2] }))), legendAt: 'right' });
                    },
                },
                {
                    id: 'lmGen', needs: ['lmGen'], live: true, label: '续写的结果（解码成声音后的频谱图）', sub: '每块：左边青线之前 = 提示（真实的第一个音符，8 帧），之后 = 模型生成的 88 帧 · 上方 = 每拍检测到的音高（绿 = 清楚的单音，黄 / 红 = 含糊）', large: true, size: [720, 280],
                    render: (c, S) => {
                        const R = S.lm, G = S.lg; if (!R || R.phase < 2 || !G) return P13wait(c, R, 'lm');
                        const { ctx, w, h } = PL.draw.ctx(c);
                        G.g.forEach((q, i) => {
                            const dur = q.x.length / P13.SR, ok = q.pitch.filter((p) => p.energy > 1e-3), clar = ok.reduce((a, p) => a + p.clarity, 0) / Math.max(1, ok.length), val = q.valid.reduce((a, b) => a + b, 0) / q.F;
                            const L = P13.gridPanel(ctx, { x: 0, y: i * h / 2, w, h: h / 2 }, q.db, q.T, q.nb, { range: P13.dbRange(q.db, 75), xMax: dur, yMax: 4, legend: false, padR: 8, padB: i ? 16 : 4, xTicks: i === 1, padL: 40, title: `${i ? 'B' : 'A'}：第 ${G.step} 步的模型 · 温度 ${S.temp.toFixed(2)} · 平均清晰度 ${clar.toFixed(2)} · ${(100 * val).toFixed(0)}% 的帧在训练集里出现过`, tx: i ? 6 : P13.LIVE_X, smooth: true, segs: q.pitch.map((p, b) => ({ x0: b * 0.256, x1: (b + 1) * 0.256, label: p.energy > 1e-3 ? P13.noteName(p.midi) : '' })), marks: [{ x: q.F0 * P13.CODEC.M / P13.SR, color: LM.C.cyan, width: 2 }] });
                            q.pitch.forEach((p, b) => { if (p.energy <= 1e-3) return; ctx.fillStyle = p.clarity > 0.8 ? LM.C.green : p.clarity > 0.5 ? LM.C.yellow : LM.C.red; ctx.fillRect(L.X(b * 0.256) + 2, L.padT - 3, L.X(0.256) - L.X(0) - 4, 2); });
                        });
                    },
                },
                {
                    id: 'lmTok', needs: ['lmGen'], live: true, label: '续写 A 的 token（还原成原始排布）', sub: '4 行 × 96 帧 · 青线 = 提示结束 · 底部红条 = 这一帧的 4 个码组合在训练集里从没出现过', size: [360, 200],
                    render: (c, S) => {
                        const G = S.lg; if (!G) return P13wait(c, S.lm, 'lm');
                        const q = G.g[0], inv = Uint8Array.from(q.valid, (v) => 1 - v);
                        P13.tokenView(c, q.codes, q.F, 4, 4, { title: `${q.F} 帧 · ${inv.reduce((a, b) => a + b, 0)} 帧是新组合`, tx: P13.LIVE_X, cellH: 26, invalid: inv, marks: [{ t: q.F0, color: LM.C.cyan }], padL: 40 });
                    },
                },
            ],
            notes: '训练开始几百步，续写就有了「音符感」：每 8 帧换一个音、音高落在五声音阶上；到 2000 步，验证集上 4 个码本的交叉熵约为 0.24 / 0.05 / 0.007 / 0.02 nats（Node 里用同一份代码实测，和页面里一致）——第 1 个码本最难（下一个音符本来就是随机的），后面的码本几乎被前面的码本和上下文决定了。训练结束时，默认温度 1 下续写 A 的平均清晰度 0.86、80% 的帧在训练集里出现过，续写 B 是 0.96、86%（另外 6 段提示上实测：清晰度 0.80–0.97，有效帧 85–99%）。把 <b>采样温度</b> 调到 1.4：多出很多训练集里没有的帧组合，听起来是杂音和跑调；调到 0.5：几乎不出错，但旋律更容易原地打转。',
            warn: 'MusicGen 用 EnCodec（32 kHz，4 个码本 × 2048 个码字，50 帧/秒）+ 文本条件（T5 编码器的交叉注意力），在 2 万小时音乐上训练，模型 3 亿到 33 亿参数；它比较了摊平、delay、并行等几种排法，delay 在质量和速度之间折中得最好。AudioLM 则分层：先用自监督模型（w2v-BERT）的 k-means token 建模「内容」，再生成 SoundStream 的粗码本、细码本。这里的模型只有 4.3 万参数、上下文 32 帧（1 秒），训练数据是极其规整的合成旋律——它学到的是「这 8 个音高的 token 怎么接」，谈不上音乐性。',
        },
        // ─────────────────────────────── 07 扩散
        {
            id: 'diffusion', title: '扩散生成音频：在 mel 的潜空间里去噪（AudioLDM 的思路）', tag: 'DIFFUSION',
            desc: '另一条路线不走 token，而是把 mel 频谱图当成<strong>图像</strong>，用第 11 篇的扩散模型生成（回顾：前向过程逐步往数据里加高斯噪声，网络学习在每个噪声水平下去噪；生成时从纯噪声出发一步步去噪）。AudioLDM 不直接在 mel 上扩散，而是先用 VAE 把 mel 压进一个小得多的<strong>潜空间</strong>，在潜空间里扩散，再解码回 mel、用声码器变成波形（「潜空间扩散」，和 Stable Diffusion 同一个思路）。这里做一个完整的缩小版：数据是 1200 个随机音符（音高在 180–720 Hz 之间连续变化，时长、起点、音色随机）的 16 × 16 log-mel 图；潜空间编码器用 <strong>PCA</strong>（第 09 篇：PCA 就是线性自编码器），取 16 维并白化；去噪网络是一个两层 MLP，预测 v（v-prediction）；DDIM 走 25 步生成；声码器就是第 04 步的「mel → 线性谱 → 快速 Griffin-Lim」。',
            formula: [
                'z = diag(λ)^{−½} · U^T (m − μ)      m̂ = μ + U · diag(λ)^{½} · ẑ   // 编码 / 解码：PCA 前 16 个主成分，保留 94% 的方差',
                'z_t = √ᾱ_t · z_0 + √(1 − ᾱ_t) · ε      v = √ᾱ_t · ε − √(1 − ᾱ_t) · z_0      L = ‖v̂_θ(z_t, t) − v‖^2',
                'ẑ_0 = √ᾱ_t · z_t − √(1 − ᾱ_t) · v̂      z_{t′} = √ᾱ_{t′} · ẑ_0 + √(1 − ᾱ_{t′}) · ε̂   // DDIM（η = 0）',
            ],
            code: {
                title: 'C++ · PCA 潜空间（pcaFit / pcaEncode / pcaDecode）/ cosineSchedule / Denoiser / diffusionTrainStep / ddimSample', src: `
                // 潜空间：PCA 的前 k 个主成分（U 是 [k × D]），每一维除以 √λ 白化成单位方差
                struct PCA { int D = 0, k = 0; std::vector<float> mu, lam; Tensor U; };

                // 子空间迭代：反复 U ← orth(U·Cov)，收敛到协方差矩阵的前 k 个特征向量
                PCA pcaFit(const Tensor& X, int k, int iters, Rng& r) {
                    PCA p; p.D = X.cols; p.k = k; p.mu = floats(p.D); p.lam = floats(k);
                    for (int i = 0; i < X.rows; i++) for (int j = 0; j < p.D; j++) p.mu[j] += X.at(i, j) / X.rows;
                    Tensor Cv = zeros(p.D, p.D);
                    for (int i = 0; i < X.rows; i++) for (int a = 0; a < p.D; a++) for (int b = 0; b < p.D; b++) Cv.at(a, b) += (X.at(i, a) - p.mu[a]) * (X.at(i, b) - p.mu[b]) / X.rows;
                    p.U = randn(k, p.D, 1.f, r);
                    for (int it = 0; it <= iters; it++) {
                        Tensor W = zeros(k, p.D);
                        for (int c = 0; c < k; c++) for (int a = 0; a < p.D; a++) for (int b = 0; b < p.D; b++) W.at(c, a) += Cv.at(a, b) * p.U.at(c, b);
                        for (int c = 0; c < k; c++) {                                   // Gram-Schmidt 正交化
                            for (int q = 0; q < c; q++) { double d = 0; for (int j = 0; j < p.D; j++) d += W.at(c, j) * W.at(q, j); for (int j = 0; j < p.D; j++) W.at(c, j) -= (float)d * W.at(q, j); }
                            double n = 0; for (int j = 0; j < p.D; j++) n += W.at(c, j) * W.at(c, j);
                            if (it == iters) p.lam[c] = (float)std::sqrt(n);                // 最后一轮：‖Cov·u‖ = 特征值
                            for (int j = 0; j < p.D; j++) W.at(c, j) /= (float)std::sqrt(n);
                        }
                        if (it < iters) p.U = W;
                    }
                    return p;
                }

                Tensor pcaEncode(const PCA& p, const Tensor& X) {
                    Tensor Z = zeros(X.rows, p.k);
                    for (int i = 0; i < X.rows; i++) for (int c = 0; c < p.k; c++) { double s = 0; for (int j = 0; j < p.D; j++) s += (X.at(i, j) - p.mu[j]) * p.U.at(c, j); Z.at(i, c) = (float)(s / std::sqrt(p.lam[c])); }
                    return Z;
                }

                Tensor pcaDecode(const PCA& p, const Tensor& Z) {
                    Tensor X = zeros(Z.rows, p.D);
                    for (int i = 0; i < Z.rows; i++) for (int j = 0; j < p.D; j++) { double v = p.mu[j]; for (int c = 0; c < p.k; c++) v += Z.at(i, c) * std::sqrt(p.lam[c]) * p.U.at(c, j); X.at(i, j) = std::clamp((float)v, -1.f, 1.f); }
                    return X;
                }

                // 余弦噪声日程（Nichol & Dhariwal 2021）：ᾱ_t = cos²((t/T + s)/(1 + s)·π/2) / cos²(s/(1 + s)·π/2)
                std::vector<float> cosineSchedule(int T) {
                    std::vector<float> ab = floats(T + 1);
                    const double s = 0.008, f0 = std::pow(std::cos(s / (1 + s) * kPi / 2), 2);
                    for (int t = 0; t <= T; t++) ab[t] = (float)std::max(1e-5, std::pow(std::cos(((double)t / T + s) / (1 + s) * kPi / 2), 2) / f0);
                    return ab;
                }

                // 时间步 t 的正弦 embedding（E 维）
                Tensor timeEmbedding(const std::vector<float>& ts, int E) {
                    Tensor e = zeros((int)ts.size(), E);
                    for (int i = 0; i < (int)ts.size(); i++) for (int j = 0; j < E / 2; j++) { double fr = std::exp(-std::log(1000.0) * j / (E / 2)); e.at(i, j) = (float)std::sin(ts[i] * fr); e.at(i, E / 2 + j) = (float)std::cos(ts[i] * fr); }
                    return e;
                }

                // 去噪网络：[z_t, emb(t)] → 两层 SiLU MLP → v̂
                struct Denoiser { Linear l1, l2, l3; int E = 32; std::vector<Var> params; };

                Denoiser makeDenoiser(int D, int hid, Rng& r) {
                    Denoiser m;
                    m.l1.W = param(heInit(D + m.E, hid, D + m.E, r)); m.l1.b = param(zeros(1, hid));
                    m.l2.W = param(heInit(hid, hid, hid, r)); m.l2.b = param(zeros(1, hid));
                    m.l3.W = param(randn(hid, D, 0.01f, r)); m.l3.b = param(zeros(1, D));
                    m.params = { m.l1.W, m.l1.b, m.l2.W, m.l2.b, m.l3.W, m.l3.b };
                    return m;
                }

                Var denoise(const Denoiser& m, const Var& zt, const std::vector<float>& ts) {
                    Var h = concatCols(zt, constant(timeEmbedding(ts, m.E)));
                    h = silu(forward(m.l1, h));
                    h = silu(forward(m.l2, h));
                    return forward(m.l3, h);
                }

                // 一步训练：随机 t、随机噪声，目标 v = √ᾱ·ε − √(1−ᾱ)·z₀
                float diffusionTrainStep(Denoiser& m, AdamW& opt, const Tensor& Z, const std::vector<float>& abar, int B, Rng& r, float lr) {
                    int D = Z.cols, T = (int)abar.size() - 1;
                    Tensor zt = zeros(B, D), v = zeros(B, D);
                    std::vector<float> ts = floats(B);
                    for (int b = 0; b < B; b++) {
                        int i = randint(r, Z.rows), t = 1 + randint(r, T);
                        float a = std::sqrt(abar[t]), s = std::sqrt(1 - abar[t]);
                        ts[b] = (float)t;
                        for (int j = 0; j < D; j++) { float e = randn(r); zt.at(b, j) = a * Z.at(i, j) + s * e; v.at(b, j) = a * e - s * Z.at(i, j); }
                    }
                    zeroGrad(m.params);
                    Var loss = mse(denoise(m, constant(zt), ts), constant(v));
                    backward(loss);
                    clipGradNorm(m.params, 1.0f);
                    adamwStep(opt, lr);
                    return loss->val.v[0];
                }

                // DDIM（确定性）：从纯噪声出发，走 steps 步回到 ẑ₀
                Tensor ddimSample(const Denoiser& m, const std::vector<float>& abar, int n, int steps, Rng& r) {
                    NoGrad ng;
                    int T = (int)abar.size() - 1, D = m.l3.W->val.cols;
                    Tensor z = randn(n, D, 1.f, r);
                    for (int k = 0; k < steps; k++) {
                        int t = (int)std::lround(T * (1.0 - (double)k / steps)), tp = k + 1 < steps ? (int)std::lround(T * (1.0 - (double)(k + 1) / steps)) : 0;
                        Var vh = denoise(m, constant(z), floats(n, (float)t));
                        float a = std::sqrt(abar[t]), s = std::sqrt(1 - abar[t]), ap = tp ? abar[tp] : 1.f;
                        for (int i = 0; i < n * D; i++) {
                            float z0 = std::clamp(a * z.v[i] - s * vh->val.v[i], -4.f, 4.f), eps = (z.v[i] - a * z0) / s;
                            z.v[i] = std::sqrt(ap) * z0 + std::sqrt(1 - ap) * eps;
                        }
                    }
                    return z;
                }`,
            },
            play: [['dif0', '生成 1'], ['dif1', '生成 2'], ['dif2', '生成 3'], ['dif3', '生成 4'], ['difReal', '训练集里的一个真实音符']],
            playTitle: '🔊 听一听（mel → 线性谱 → 快速 GL，点击时现算）',
            previews: [
                {
                    id: 'difTraj', needs: ['dif'], live: true, label: '一次生成：DDIM 每隔 3 步的 x̂₀（解码回 mel）', sub: '从左到右：t = 100（纯噪声）→ t = 0 · 每格 = 16 个 mel 通道 × 16 帧（0.5 秒）· 低频在下', large: true, size: [720, 190],
                    render: (c, S) => {
                        const R = S.dif; if (!R || R.phase < 2 || !R.samples) return P13wait(c, R, 'dif');
                        const { ctx, w, h } = PL.draw.ctx(c), sm = R.samples, n = sm.traj.length, cw = (w - 8) / n;
                        sm.traj.forEach((img, i) => { P13.gridPanel(ctx, { x: 4 + i * cw, y: 18, w: cw, h: h - 22 }, P13.melT(img), 16, 16, { range: [-1, 1], map: 'inferno', legend: false, padL: 4, padR: 4, padB: 4, xTicks: false, yTicks: false, title: `t = ${sm.trajT[i]}`, tx: 4 }); });   // 面板整体下移 18 px：左上角留给 LIVE 徽标
                        ctx.font = LM.FONT(11); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(`第 ${sm.step} 步的模型 · 每格 16 帧 × 16 个 mel 通道`, w - 6, 3);
                    },
                },
                {
                    id: 'difGrid', needs: ['dif'], live: true, label: '生成的 8 个音符 vs 训练集里最近的样本', sub: '上排 = 生成 · 下排 = 训练集里离它最近的真实音符（看是不是在「背」训练集）· 数字 = 均方根距离', size: [360, 260],
                    render: (c, S) => {
                        const R = S.dif; if (!R || !R.samples) return P13wait(c, R, 'dif');
                        const { ctx, w, h } = PL.draw.ctx(c), sm = R.samples, D = P13.DIF.D, cw = (w - 8) / 4, rh = (h - 26) / 4;
                        for (let i = 0; i < 8; i++) {
                            const col = i % 4, row = Math.floor(i / 4) * 2;
                            P13.gridPanel(ctx, { x: 4 + col * cw, y: 22 + row * rh, w: cw, h: rh }, P13.melT(sm.imgs.subarray(i * D, (i + 1) * D)), 16, 16, { range: [-1, 1], map: 'inferno', legend: false, padL: 3, padR: 3, padB: 2, xTicks: false, yTicks: false });
                            P13.gridPanel(ctx, { x: 4 + col * cw, y: 22 + (row + 1) * rh, w: cw, h: rh }, P13.melT(R.X.subarray(sm.near[i].i * D, (sm.near[i].i + 1) * D)), 16, 16, { range: [-1, 1], map: 'viridis', legend: false, padL: 3, padR: 3, padB: 2, xTicks: false, yTicks: false, title: `${sm.near[i].d.toFixed(2)}`, titleColor: LM.C.gray, tx: 4 });
                        }
                        ctx.font = LM.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`第 ${sm.step} 步 · 上 = 生成 / 下 = 训练集里最近的样本`, P13.LIVE_X, 4);
                    },
                },
                {
                    id: 'difLoss', needs: ['dif'], live: true, label: '训练损失 ‖v̂ − v‖²', sub: '滑动平均 · 随机 t 的平均（小 t 时 v ≈ ε 本来就猜不准，所以不会趋于 0）', size: [360, 220],
                    render: (c, S) => { const R = S.dif; if (!R || R.phase < 2) return P13wait(c, R, 'dif'); P13.chart(c, { x: [0, P13.DIFR.STEPS], y: [0.5, 1.2], xLabel: '训练步', series: [{ y: R.loss, color: 'rgba(106,176,224,0.25)', width: 1 }, { y: LM.ema(R.loss, 0.98), color: LM.C.blue, width: 2 }], title: `第 ${R.step} / ${P13.DIFR.STEPS} 步 · PCA 保留 ${(100 * R.pca.explained).toFixed(1)}% 方差`, tx: P13.LIVE_X }); },
                },
                {
                    id: 'difLatent', needs: ['dif'], live: true, label: '潜空间的前两维', sub: '灰 = 训练集（前 400 个）· 橙 = 生成的 8 个样本 · 白化后每一维都是单位方差', size: [360, 220],
                    render: (c, S) => {
                        const R = S.dif; if (!R || !R.samples) return P13wait(c, R, 'dif');
                        const K = P13.DIF.K, pts = [];
                        for (let i = 0; i < 400; i++) pts.push({ x: R.Z[i * K], y: R.Z[i * K + 1], color: '#4a5260', r: 2 });
                        for (let i = 0; i < 8; i++) pts.push({ x: R.samples.z[i * K], y: R.samples.z[i * K + 1], color: LM.C.orange, r: 4 });
                        LM.draw.scatter(c, pts, { x: [-3.5, 3.5], y: [-3.5, 3.5], equal: true, xLabel: 'z₁', yLabel: 'z₂' });
                    },
                },
            ],
            notes: '训练分三阶段（只在本步可见时跑）：合成 1200 个音符的 log-mel → 分块做 PCA → 训练 4000 步（每帧 2 步）。几百步后生成的就是「一个音符」的样子：有起止、有一组随频率升高而变弱的谐波带。Node 里用同一份代码跑满 4000 步实测：生成样本离最近训练样本的均方根距离 0.19–0.34，和新合成的真实音符（同样经过 PCA）离训练集的距离 0.09–0.35 相当，说明它不是在背训练集；用自相关估计生成声音的基频，8 个样本落在 308–667 Hz，覆盖了训练范围（180–720 Hz）的大部分。点 ▶ 听：音色比原始音符粗糙一些——16 × 16 的 mel 本身丢了很多细节，再加 Griffin-Lim 的相位误差。',
            warn: '没有直接在 256 维的 mel 图上训练 MLP 去噪器，是因为实测不收敛：网络在高噪声时要输出 ε ≈ z_t（恒等映射），没有残差连接的窄 MLP 学不好；ε-prediction 换成 v-prediction（Salimans & Ho 2022）、再换到 16 维的潜空间之后才好起来——真实系统用 U-Net / DiT 这类带跳连的网络，也正是在 VAE 的潜空间里训练。AudioLDM 的潜空间来自 mel 上训练的 VAE、条件来自 CLAP 的文本-音频联合嵌入、声码器是 HiFi-GAN；这里没有文本条件（无条件生成）。',
        },

        // ─────────────────────────────── 08 TTS
        {
            id: 'tts', title: '语音合成（TTS）：文本 → 音素 → 时长 / 基频 / 共振峰 → 声码器', tag: 'TTS',
            desc: '经典 TTS 流水线有三段：<strong>文本前端</strong>（文字 → 拼音 / 音素，处理多音字、变调）、<strong>声学模型</strong>（每个音素多长、音高怎么走、频谱什么样）、<strong>声码器</strong>（声学参数 → 波形）。神经 TTS 把后两段换成网络：Tacotron 2 用注意力从字符直接预测 mel、FastSpeech 2 显式预测时长 / 音高 / 能量、HiFi-GAN 把 mel 变成波形；VITS 端到端；VALL-E 则把 TTS 当成「在 codec token 上做语言建模」（第 06 步）。这里用规则做一个能跑通的小版本：查字典得到拼音和声调 → 上声变调 → 每个音节分配时长（声母 75 ms、韵母 260 ms）→ <strong>声调 = 基频曲线</strong>（汉语的「妈麻马骂」只差 F0 的走向）→ 共振峰目标 + 音素边界附近的线性过渡（协同发音）→ 第 01 步之后就一直在用的共振峰合成器：声门脉冲经过 4 个级联谐振器（源-滤波器模型）。',
            formula: [
                'F0(u) = base · 2^{(L(u) − 3)/4}      L(u) = 五度标记的线性插值   // 阴平 55、阳平 35、上声 214、去声 51；u ∈ [0, 1] 是在韵母里的位置',
                '上声变调：T3 + T3 → T2 + T3   // 你好 nǐ hǎo → ní hǎo',
                's[n] = g′(φ_n) 或 噪声      y_k[n] = A_k·y_{k−1}[n] + B_k·y_k[n−1] + C_k·y_k[n−2]   // 声源 → 级联谐振器 F1…F4（Klatt 1980）',
            ],
            code: {
                title: 'C++ · g2p / toneSandhi / toneF0 / phoneTarget / ttsFrontend / Resonator / rosenberg / formantSynth', src: `
                struct Syllable { std::string initial, final; int tone = 1; };

                // 字 → 拼音（查一张十几个字的小字典，实现略：页面里的 JS 是 LEXICON 表，如 妈 → m + a + 1，好 → h + ao + 3）
                std::vector<Syllable> g2p(const std::string& text);

                // 上声变调：两个第三声相连，前一个读成第二声
                void toneSandhi(std::vector<Syllable>& s) { for (size_t i = 0; i + 1 < s.size(); i++) if (s[i].tone == 3 && s[i + 1].tone == 3) s[i].tone = 2; }

                // 五度标记 → 基频：相邻两度差 2^(1/4)，调域 1 个八度；u ∈ [0, 1] 是在韵母里的位置
                float toneF0(int tone, float u, float base) {
                    static const float contour[5][3] = { { 3, 3, 3 }, { 5, 5, 5 }, { 3, 4, 5 }, { 2, 1, 4 }, { 5, 3, 1 } };   // 轻声、阴平、阳平、上声、去声
                    float x = std::clamp(u, 0.f, 1.f) * 2;
                    int i = std::min(1, (int)x);
                    float level = contour[tone][i] + (contour[tone][i + 1] - contour[tone][i]) * (x - i);
                    return base * std::pow(2.f, (level - 3) / 4);
                }

                // 每 5 ms 一帧的声学参数 = 声学模型的输出 = 声码器的输入
                struct Frame { float f0 = 120, av = 0, ah = 0; float F[4] = { 500, 1500, 2500, 3300 }; float BW[4] = { 70, 90, 150, 250 }; };

                // 音素的共振峰目标（成年男性，Peterson & Barney 1952 等）：amp = 浊音强度，鼻音 m / n 很弱
                Frame phoneTarget(char ph) {
                    Frame f; f.av = 1;
                    float t[11][4] = { { 730, 1090, 2440, 3300 }, { 570, 840, 2410, 3300 }, { 530, 1840, 2480, 3300 }, { 270, 2290, 3010, 3500 }, { 300, 870, 2240, 3300 },
                                       { 250, 1900, 2400, 3300 }, { 460, 1220, 2420, 3300 }, { 280, 1000, 2200, 3300 }, { 280, 1450, 2500, 3300 }, { 360, 1300, 2700, 3300 }, { 730, 1090, 2440, 3300 } };
                    const std::string names = "aoeiuvEmnlh";
                    int k = (int)names.find(ph);
                    if (k < 0) k = 0;
                    for (int j = 0; j < 4; j++) f.F[j] = t[k][j];
                    if (ph == 'm' || ph == 'n') { f.av = 0.22f; for (float& b : f.BW) b *= 2.2f; }
                    if (ph == 'l') f.av = 0.55f;
                    if (ph == 'h') { f.av = 0; f.ah = 0.9f; }                          // 送气：噪声声源
                    return f;
                }

                // 规则声学模型：声母 75 ms + 韵母 260 ms（元音目标之间线性过渡），F0 = 声调曲线 × 整句缓降；vtl = 声道缩放
                std::vector<Frame> ttsFrontend(std::vector<Syllable> syl, float base, float vtl) {
                    toneSandhi(syl);
                    std::vector<Frame> out;
                    for (size_t s = 0; s < syl.size(); s++) {
                        int nIni = syl[s].initial.empty() ? 0 : 15, nFin = 52;          // 帧数（5 ms 一帧）
                        for (int i = 0; i < nIni + nFin; i++) {
                            bool ini = i < nIni;
                            float u = ini ? 0 : (float)(i - nIni) / nFin;
                            int nv = (int)syl[s].final.size(), v = std::min(nv - 1, (int)(u * nv));
                            Frame f = phoneTarget(ini ? syl[s].initial[0] : syl[s].final[v]);
                            if (!ini && v + 1 < nv) {                                   // 复韵母：向下一个元音目标过渡
                                Frame g = phoneTarget(syl[s].final[v + 1]);
                                float w = std::clamp((u * nv - v - 0.5f) * 2, 0.f, 1.f) * 0.5f;
                                for (int j = 0; j < 4; j++) f.F[j] += (g.F[j] - f.F[j]) * w;
                            }
                            if (ini && syl[s].initial == "h") { Frame g = phoneTarget(syl[s].final[0]); for (int j = 0; j < 4; j++) f.F[j] = g.F[j]; }
                            const float Fmax[4] = { 1300, 3000, 3450, 3850 };   // 每个共振峰各自的上限（8 kHz 采样，奈奎斯特 4 kHz）：别让 F3、F4 叠在同一个频率
                            for (int j = 0; j < 4; j++) f.F[j] = std::min(Fmax[j], f.F[j] * vtl);
                            f.f0 = toneF0(syl[s].tone, u, base) * (1 - 0.08f * s / std::max<size_t>(1, syl.size()));
                            float env = std::min({ 1.f, i / 4.f, (nIni + nFin - i) / 6.f });  // 音节的起音 / 收音
                            f.av *= env; f.ah *= env;
                            out.push_back(f);
                        }
                        for (int i = 0; i < 5; i++) { Frame gap = out.back(); gap.av = 0; gap.ah = 0; out.push_back(gap); }   // 音节之间 25 ms 停顿
                    }
                    return out;
                }

                // 二阶谐振器（Klatt 1980）：y[n] = A·x[n] + B·y[n−1] + C·y[n−2]，中心频率 F、带宽 BW，直流增益 1
                struct Resonator { float A = 0, B = 0, C = 0, y1 = 0, y2 = 0; };

                void setResonator(Resonator& r, float F, float BW) {
                    r.C = -(float)std::exp(-2 * kPi * BW / SR);
                    r.B = 2 * (float)(std::exp(-kPi * BW / SR) * std::cos(2 * kPi * F / SR));
                    r.A = 1 - r.B - r.C;
                }

                float resonate(Resonator& r, float x) { float y = r.A * x + r.B * r.y1 + r.C * r.y2; r.y2 = r.y1; r.y1 = y; return y; }

                // Rosenberg 声门脉冲：相位 φ ∈ [0, 1)，开相 40% 余弦上升、闭相 16% 余弦下降，其余时间声门关闭
                float rosenberg(float phase) {
                    const float tp = 0.40f, tn = 0.16f;
                    if (phase < tp) return 0.5f * (1 - (float)std::cos(kPi * phase / tp));
                    if (phase < tp + tn) return (float)std::cos(0.5 * kPi * (phase - tp) / tn);
                    return 0.f;
                }

                // 共振峰声码器：声门脉冲的差分（≈ 唇辐射）+ 送气噪声 → 4 个级联谐振器；帧参数在 5 ms 之间线性插值
                std::vector<float> formantSynth(const std::vector<Frame>& frames, Rng& r) {
                    int hop = SR / 200, n = ((int)frames.size() - 1) * hop;
                    std::vector<float> out = floats(n);
                    Resonator res[4];
                    float phase = 0, prev = 0;
                    for (int i = 0; i < n; i++) {
                        const Frame &p = frames[i / hop], &q = frames[i / hop + 1];
                        float u = (float)(i % hop) / hop;
                        float f0 = p.f0 + (q.f0 - p.f0) * u, av = p.av + (q.av - p.av) * u, ah = p.ah + (q.ah - p.ah) * u;
                        phase += f0 / SR;
                        if (phase >= 1) phase -= 1;
                        float g = rosenberg(phase), s = av * (g - prev) * 6 + ah * (2 * (float)uniform(r) - 1) * 0.08f;
                        prev = g;
                        if (i % 8 == 0) for (int k = 0; k < 4; k++) setResonator(res[k], p.F[k] + (q.F[k] - p.F[k]) * u, p.BW[k] + (q.BW[k] - p.BW[k]) * u);
                        for (int k = 0; k < 4; k++) s = resonate(res[k], s);
                        out[i] = s;
                    }
                    return out;
                }`,
            },
            play: [['tts', '合成语音（当前文本 / 声音）']],
            previews: [
                {
                    id: 'ttsPipe', label: '文本前端：字 → 拼音（声调）→ 音素与时长', sub: '色块 = 每个音素占的时间（声母 / 韵母里的元音目标）· 灰 = 音节之间的停顿', large: true, size: [720, 160],
                    render: (c, S) => {
                        const t = S.tt, { ctx, w } = PL.draw.ctx(c), padL = 10, pw = w - 20, X = (s) => padL + s / t.total * pw;
                        ctx.font = LM.FONT(13, true); ctx.textBaseline = 'middle';
                        t.syls.forEach((q) => { const x0 = X(q.t0), x1 = X(q.t1); ctx.fillStyle = LM.C.text; ctx.textAlign = 'center'; ctx.font = LM.FONT(15, true); ctx.fillText(q.ch, (x0 + x1) / 2, 18); ctx.font = LM.FONT(12); ctx.fillStyle = LM.C.yellow; ctx.fillText(P13.pinyinOf(q.s) + (q.s[2] !== q.tone0 ? `（原 ${q.tone0} 声）` : ''), (x0 + x1) / 2, 42); ctx.fillStyle = LM.C.gray; ctx.fillText(`第 ${q.s[2]} 声 ${P13.TONES[q.s[2]].join('')}`, (x0 + x1) / 2, 62); });
                        t.segs.forEach((g) => { const x0 = X(g.t0), x1 = X(g.t1), P = P13.PHONES[g.ph]; ctx.fillStyle = P.asp ? '#5b6068' : g.ph === 'm' || g.ph === 'n' || g.ph === 'l' ? '#3d6b8c' : LM.tokColor(g.ph.charCodeAt(0) * 3); ctx.fillRect(x0 + 0.5, 78, Math.max(1, x1 - x0 - 1), 30); ctx.fillStyle = '#f0f0f0'; ctx.font = LM.FONT(12); ctx.textAlign = 'center'; ctx.fillText('/' + P13.PHONE_IPA[g.ph] + '/', (x0 + x1) / 2, 93); ctx.fillStyle = LM.C.dim; ctx.font = LM.FONT(11); ctx.fillText(`${Math.round(1000 * (g.t1 - g.t0))}`, (x0 + x1) / 2, 118); });
                        ctx.fillStyle = LM.C.dim; ctx.textAlign = 'left'; ctx.font = LM.FONT(11); ctx.fillText(`ms · 总长 ${t.total.toFixed(2)} s · ${t.frames.length} 帧声学参数（每帧 5 ms）· ${S.voice === 'm' ? '男声' : S.voice === 'f' ? '女声' : '童声'}：基频中值 ${t.voice.f0} Hz，共振峰 ×${t.voice.vtl}`, padL, 140);
                    },
                },
                {
                    id: 'ttsTracks', label: '声学模型的输出：基频（声调）与共振峰轨迹', sub: '上 = F0（Hz，对数刻度）· 下 = F1 / F2 / F3（Hz）· 竖虚线 = 音节边界 · 只画浊音部分', large: true, size: [720, 260],
                    render: (c, S) => {
                        const t = S.tt, fr = t.frames, xs = fr.map((_, i) => i * t.dt), { ctx, w, h } = PL.draw.ctx(c), V = (i) => fr[i].av > 0.05;
                        const pad = { l: 46, r: 12 }, pw = w - pad.l - pad.r, X = (s) => pad.l + s / t.total * pw, top = { y: 22, h: 80 }, bot = { y: 128, h: 104 };
                        const f0lo = 60, f0hi = 480, Yf = (f) => top.y + top.h - Math.log(f / f0lo) / Math.log(f0hi / f0lo) * top.h, Yk = (f) => bot.y + bot.h - f / 4000 * bot.h;
                        ctx.font = LM.FONT(11, true); ctx.fillStyle = LM.C.text; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`F0（声调）`, 6, 4); ctx.fillText('共振峰', 6, bot.y - 16);
                        ctx.font = LM.FONT(11); ctx.fillStyle = '#7a808a'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                        [80, 120, 200, 300, 450].forEach((f) => { ctx.fillText(String(f), pad.l - 4, Yf(f)); ctx.fillStyle = '#1b1f25'; ctx.fillRect(pad.l, Yf(f), pw, 1); ctx.fillStyle = '#7a808a'; });
                        [0, 1000, 2000, 3000, 4000].forEach((f) => { ctx.fillText(f ? f / 1000 + 'k' : '0', pad.l - 4, Yk(f)); ctx.fillStyle = '#1b1f25'; ctx.fillRect(pad.l, Yk(f), pw, 1); ctx.fillStyle = '#7a808a'; });
                        t.syls.forEach((q) => { ctx.strokeStyle = '#3a3f46'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(q.t0), top.y); ctx.lineTo(X(q.t0), bot.y + bot.h); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = LM.C.gray; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(q.ch, (X(q.t0) + X(q.t1)) / 2, top.y + top.h + 3); });
                        const line = (Y, f, col, wd) => { ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.beginPath(); let pen = false; for (let i = 0; i < fr.length; i++) { if (!V(i)) { pen = false; continue; } const px = X(xs[i]), py = Y(f(fr[i])); if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; } } ctx.stroke(); };
                        line(Yf, (q) => q.f0, LM.C.yellow, 2.4);
                        [LM.C.orange, LM.C.green, LM.C.cyan].forEach((col, k) => line(Yk, (q) => q.F[k], col, 2));
                        ctx.font = LM.FONT(11); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; [LM.C.orange, LM.C.green, LM.C.cyan].forEach((col, k) => { ctx.fillStyle = col; ctx.fillText(`F${k + 1}`, w - 34, bot.y + 10 + k * 14); });
                    },
                },
                {
                    id: 'ttsSpec', label: '合成结果的频谱图', sub: '虚线 = 上图的 F1 / F2 / F3 目标轨迹 · 横向的细条纹 = 基频的谐波（声调就是它们整体上下移动）', size: [720, 220],
                    render: (c, S) => {
                        const t = S.tt;
                        P13.gridView(c, t.db, t.T, t.nb, { range: t.range, xMax: t.T * 64 / P13.SR, yMax: 4, xLabel: '时间 (s)', yLabel: 'kHz', smooth: true, overlay: (ctx, X, Y) => { [LM.C.orange, LM.C.green, LM.C.cyan].forEach((col, k) => { ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]); ctx.beginPath(); let pen = false; t.frames.forEach((q, i) => { if (q.av < 0.05) { pen = false; return; } const px = X(i * t.dt), py = Y(q.F[k] / 1000); if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; } }); ctx.stroke(); }); ctx.setLineDash([]); } });
                    },
                },
            ],
            notes: '选「妈麻马骂」依次听：四个音节的共振峰完全一样（都是 m + a），只有上图的黄线不同——平、升、降升、降，这就是汉语的声调。选「你好」：前一个「你」由第三声变成第二声（标注「原 3 声」），这是规则前端做的。<b>声音</b>换成女声 / 童声：基频升高、谐波变稀，共振峰整体上移（声道更短）。共振峰合成能听懂，但有明显的「机器味」：真实语音的频谱细节、声源的随机变化、辅音都比这几条规则复杂得多。',
            warn: '这是 1980 年代规则 TTS（如 Klatt 的 MITalk / DECtalk）的极简版。没有做的：多音字消歧、韵律预测（重音、停顿）、绝大多数声母（塞音、擦音、塞擦音需要噪声声源和并联谐振器）、鼻音的反共振峰。声调轮廓用的是孤立字的五度值，连续语流里的真实声调还会互相影响（协同发音）。',
        },
        // ─────────────────────────────── 09 ASR
        {
            id: 'asr', title: '语音识别（ASR）：log-mel → 卷积编码器 → 元音（Whisper 前端的缩小版）', tag: 'ASR',
            desc: '识别是合成的逆过程：输入 log-mel 频谱，输出文字。Whisper 的编码器先用两层一维卷积（核宽 3，第二层步长 2，GELU）处理 80 通道的 log-mel，再接 Transformer；解码器像第 05 篇的 GPT 一样逐个生成文字 token。这里只做最前面的一小段：同样的 Whisper 式 log-mel 归一化 → 两层一维卷积（把 mel 通道当成输入通道，沿时间卷积，用 <code>AG.conv2d</code> 的 H = 1）→ 时间上取平均 → 线性层 → 5 个元音。数据是共振峰合成器生成的 600 条训练元音 + 250 条测试元音，<strong>每条都来自一个随机说话人</strong>：基频 85–280 Hz（男声到童声）、声道长度 ×0.85–1.25、共振峰再抖 ±4%、随机噪声和音量。测试集的说话人和训练集完全不同——要泛化的正是「同一个元音，不同人说」。训练完再用它扫一遍第 08 步合成的句子。',
            formula: [
                'f = log10(max(mel, 10^{−10}))      f ← max(f, max(f) − 8)      f ← (f + 4) / 4   // Whisper 的 log_mel_spectrogram 归一化',
                'h₁ = GELU(Conv1d_{32→16, k=3}(f))      h₂ = GELU(Conv1d_{16→16, k=3, stride 2}(h₁))      p = softmax(W · mean_t(h₂) + b)',
                '输入 32 个 mel 通道 × 33 帧（0.32 s，窗 32 ms、跳 10 ms）   // 共 2 421 个参数',
            ],
            code: {
                title: 'C++ · whisperLogMel / conv1d / VowelNet（makeVowelNet / vowelNetForward / vowelTrainStep）', src: `
                // Whisper 式特征：log10 → 截到最大值 − 8 → (x + 4) / 4；输出一行 [NM × T]（通道优先 = 一维卷积的输入布局）
                Tensor whisperLogMel(const std::vector<float>& x, const MelBank& B, int N, int H) {
                    Spec S = stft(x, N, H);
                    std::vector<float> lg = floats(S.T * B.nMels);
                    float mx = -1e30f;
                    for (int t = 0; t < S.T; t++)
                        for (int m = 0; m < B.nMels; m++) {
                            double e = 0;
                            for (int k = 0; k < S.nb; k++) { int i = t * S.nb + k; e += B.W[m * B.nb + k] * (S.re[i] * S.re[i] + S.im[i] * S.im[i]); }
                            lg[t * B.nMels + m] = (float)std::log10(std::max(e, 1e-10));
                            mx = std::max(mx, lg[t * B.nMels + m]);
                        }
                    Tensor f = zeros(1, B.nMels * S.T);
                    for (int t = 0; t < S.T; t++) for (int m = 0; m < B.nMels; m++) f.v[m * S.T + t] = (std::max(lg[t * B.nMels + m], mx - 8) + 4) / 4;
                    return f;
                }

                // 一维卷积 = H = 1 的二维卷积：输入 [N × C·1·W]，卷积核 1 × K
                ConvSpec conv1d(int C, int W, int K, int stride, int pad) {
                    ConvSpec s; s.C = C; s.H = 1; s.W = W; s.KH = 1; s.KW = K; s.stride = stride; s.padH = 0; s.padW = pad;
                    return s;
                }

                struct VowelNet { int NM = 32, T = 33, C1 = 16, T2 = 17; Var W1, b1, W2, b2, pool; Linear fc; std::vector<Var> params; };

                VowelNet makeVowelNet(int NM, int T, int C1, Rng& r) {
                    VowelNet m; m.NM = NM; m.T = T; m.C1 = C1; m.T2 = (T + 2 - 3) / 2 + 1;
                    m.W1 = param(heInit(C1, NM * 3, NM * 3, r)); m.b1 = param(zeros(1, C1));
                    m.W2 = param(heInit(C1, C1 * 3, C1 * 3, r)); m.b2 = param(zeros(1, C1));
                    m.fc.W = param(xavierInit(C1, 5, C1, 5, r)); m.fc.b = param(zeros(1, 5));
                    Tensor P = zeros(C1 * m.T2, C1);                                   // 时间平均 = 乘一个固定矩阵
                    for (int c = 0; c < C1; c++) for (int t = 0; t < m.T2; t++) P.at(c * m.T2 + t, c) = 1.f / m.T2;
                    m.pool = constant(P);
                    m.params = { m.W1, m.b1, m.W2, m.b2, m.fc.W, m.fc.b };
                    return m;
                }

                // x：[N × NM·T] → logits [N × 5]
                Var vowelNetForward(const VowelNet& m, const Var& x) {
                    Var h1 = gelu(conv2d(x, m.W1, m.b1, conv1d(m.NM, m.T, 3, 1, 1)));
                    Var h2 = gelu(conv2d(h1, m.W2, m.b2, conv1d(m.C1, m.T, 3, 2, 1)));   // 步长 2：时间分辨率减半（Whisper 编码器同款）
                    return forward(m.fc, matmul(h2, m.pool));
                }

                float vowelTrainStep(VowelNet& m, AdamW& opt, const Tensor& X, const std::vector<int>& y, float lr) {
                    zeroGrad(m.params);
                    Var loss = crossEntropy(vowelNetForward(m, constant(X)), y);
                    backward(loss);
                    adamwStep(opt, lr);
                    return loss->val.v[0];
                }`,
            },
            play: [['tts', '第 08 步合成的句子（被识别的那一段）']],
            previews: [
                {
                    id: 'asrPost', needs: ['asrScan'], live: true, label: '识别第 08 步合成的句子：每 40 ms 一个窗的元音后验概率', sub: '行 = 5 个元音 · 颜色 = P(元音 | 这 0.32 s) · 灰色列 = 能量太低（停顿 / 鼻音 / 送气），不判 · 上方 = 每个窗的判决 · 最上 = 实际的音素（第 08 步的计划）', large: true, size: [720, 240],
                    render: (c, S) => {
                        const R = S.asr, A = S.as;
                        if (!R || R.phase < 1 || !A) return P13wait(c, R, 'asr');
                        const nW = A.nW, grid = new Float32Array(nW * 5);
                        for (let w2 = 0; w2 < nW; w2++) for (let k = 0; k < 5; k++) grid[w2 * 5 + k] = A.en[w2] > P13.ASR_GATE ? A.post[w2 * 5 + (4 - k)] : NaN;
                        const t0 = P13.ASR.L / 2 / P13.SR, dur = nW * A.hop / P13.SR;
                        const L = P13.gridView(c, grid, nW, 5, { range: [0, 1], map: 'viridis', xMax: dur, yMax: 5, xLabel: `窗中心时间 (s)`, xFmt: (v) => (v + t0).toFixed(1), yTickVals: [0.5, 1.5, 2.5, 3.5, 4.5], yFmt: (v) => 'uoiea'[Math.floor(v)], padL: 30, title: `第 ${A.step} 步的识别器 · 「${S.phrase}」（${S.voice === 'm' ? '男声' : S.voice === 'f' ? '女声' : '童声'}）`, tx: P13.LIVE_X, loLabel: '0', hiLabel: '1', segs: [], padTop: 16 });
                        const ctx = L.ctx, tt = S.tt; ctx.font = LM.FONT(11); ctx.textBaseline = 'bottom';
                        ctx.textAlign = 'left'; ctx.fillStyle = LM.C.gray; ctx.fillText('实际', 2, L.padT - 21); ctx.fillText('识别', 2, L.padT - 2); ctx.textAlign = 'center';
                        // 第一行：第 08 步合成计划里真实的音素（横轴对齐到窗中心：窗中心 = 起点 + 0.16 s）
                        if (tt && tt.segs) tt.segs.forEach((g, i) => { const a = Math.max(0, g.t0 - t0), b = Math.min(dur, g.t1 - t0); if (b <= a) return; const x0 = L.X(a), x1 = L.X(b); ctx.fillStyle = i % 2 ? '#3d5a6c' : '#5b7f95'; ctx.fillRect(x0, L.padT - 19, Math.max(1, x1 - x0 - 1), 2); ctx.fillStyle = '#a9c3d3'; ctx.fillText(g.ph, (x0 + x1) / 2, L.padT - 21); });
                        // 第二行：每个窗的判决（能量太低的窗不判）
                        for (let w2 = 0; w2 < nW; w2++) { if (A.en[w2] <= P13.ASR_GATE) continue; const k = LM.argmax(A.post, w2 * 5, 5); ctx.fillStyle = LM.C.yellow; ctx.fillText('aeiou'[k], L.X((w2 + 0.5) * A.hop / P13.SR), L.padT - 2); }
                    },
                },
                {
                    id: 'asrCurve', needs: ['asr'], live: true, label: '训练曲线', sub: '对数纵轴 · 蓝 = 训练 loss（滑动平均）· 橙 = 测试集（新说话人）错误率', size: [360, 230],
                    render: (c, S) => {
                        const R = S.asr; if (!R || R.phase < 1) return P13wait(c, R, 'asr');
                        const ev = R.evals, last = ev[ev.length - 1];
                        P13.chart(c, { x: [0, P13.ASRR.STEPS], y: [0.002, 3], logY: true, xLabel: '训练步', series: [{ y: LM.ema(R.loss, 0.95), color: LM.C.blue, width: 2 }, { x: ev.map((e) => e.step), y: ev.map((e) => Math.max(0.002, 1 - e.acc)), color: LM.C.orange, width: 2, dots: 2.5 }], title: `第 ${R.step} / ${P13.ASRR.STEPS} 步 · 测试准确率 ${last ? (100 * last.acc).toFixed(1) + '%' : '—'}`, tx: P13.LIVE_X, legend: [{ label: 'loss', color: LM.C.blue }, { label: '测试错误率', color: LM.C.orange }], legendAt: 'bl' });
                    },
                },
                {
                    id: 'asrConf', needs: ['asr'], live: true, label: '测试集混淆矩阵（250 条，新说话人）', sub: '行 = 真实元音 · 列 = 识别结果 · 对角线以外 = 认错', size: [360, 230],
                    render: (c, S) => {
                        const R = S.asr, e = R && R.evals[R.evals.length - 1]; if (!e) return P13wait(c, R, 'asr');
                        const lab = ['a /ɑ/', 'e /ɛ/', 'i', 'o /ɔ/', 'u'];
                        LM.draw.heat(c, Float32Array.from(e.conf), 5, 5, { rowLabels: lab, colLabels: ['a', 'e', 'i', 'o', 'u'], digits: 0, map: 'viridis', title: `        第 ${e.step} 步 · 准确率 ${(100 * e.acc).toFixed(1)}%` });
                    },
                },
                {
                    id: 'asrSpace', needs: ['asr'], live: true, label: '测试样本在 F1–F2 平面上的位置', sub: '每个点 = 一条测试样本（说话人缩放后的真实 F1、F2）· 颜色 = 真实元音 · 红圈 = 识别错了 · 声道越短（女声 / 童声）点越往右上', size: [360, 260],
                    render: (c, S) => {
                        const R = S.asr; if (!R || R.made < P13.ASRR.NTR + 20) return P13wait(c, R, 'asr');
                        const e = R.evals[R.evals.length - 1], cols = [LM.C.red, LM.C.orange, LM.C.green, LM.C.blue, LM.C.purple], pts = [], n = Math.min(P13.ASRR.NTE, R.made - P13.ASRR.NTR);
                        for (let i = 0; i < n; i++) { const s = R.te.spk[i], v = P13.VOWELS[R.te.y[i]], F1 = v.F[0] * s.vtl * (1 + s.fj[0]), F2 = v.F[1] * s.vtl * (1 + s.fj[1]), bad = e && e.pred[i] !== R.te.y[i]; pts.push({ x: F2, y: F1, color: cols[R.te.y[i]], r: bad ? 3.5 : 2.2, ring: bad ? '#ff4d4d' : null }); }
                        P13.VOWELS.forEach((v, k) => pts.push({ x: v.F[1], y: v.F[0], color: cols[k], r: 0.1, label: v.key, labelColor: cols[k], labelPx: 12 }));
                        LM.draw.scatter(c, pts, { x: [600, 3200], y: [180, 1000], xLabel: 'F2 (Hz)', yLabel: 'F1 (Hz)' });
                    },
                },
                {
                    id: 'asrFeat', needs: ['asr'], live: true, label: '识别器的输入：每个元音的一条测试样本', sub: '32 个 mel 通道 × 33 帧（低频在下）· 共振峰 = 亮带，谐波 = 细横纹 · 同一个元音不同说话人差别很大', size: [360, 200],
                    render: (c, S) => {
                        const R = S.asr; if (!R || R.made < P13.ASRR.NTR + 5) return P13wait(c, R, 'asr');
                        const { ctx, w, h } = PL.draw.ctx(c), D = R.D, cw = (w - 6) / 5;
                        for (let k = 0; k < 5; k++) {
                            let i = 0; while (i < P13.ASRR.NTE - 1 && R.te.y[i] !== k) i++;
                            const img = R.te.X.subarray(i * D, (i + 1) * D), T = P13.ASR.T, NM = P13.ASR.NM, g = new Float32Array(D);
                            for (let m = 0; m < NM; m++) for (let t = 0; t < T; t++) g[t * NM + m] = img[m * T + t];
                            P13.gridPanel(ctx, { x: 3 + k * cw, y: 18, w: cw, h: h - 18 }, g, T, NM, { range: [-1, 1.7], map: 'inferno', legend: false, padL: 3, padR: 3, padB: 16, xTicks: false, yTicks: false, title: `${P13.VOWELS[k].key}  ${Math.round(R.te.spk[i].f0)}Hz`, tx: 4 });
                        }
                    },
                },
            ],
            notes: '默认设置（Node 里用同一份代码实测，和页面里一致）：150 步测试准确率约 91%，600 步起稳定在 97.2%（250 条新说话人的样本里认错 7 条，全是 /ɑ/ 被认成 /ɔ/）——左下的 F1–F2 图解释了原因：/ɑ/ 和 /ɔ/ 在共振峰空间里本来就挨着，女声 / 童声把整片点推向右上，男声的 /ɛ/ 也会和童声的 /ɑ/ 重叠（Peterson & Barney 1952 的经典发现）。上面的后验图：把 <b>TTS 文本</b>选成「啊诶衣喔乌」，识别出 a e i o u 五段；换成「我爱你」「猫爱鱼」，单元音段认得准，复元音（ai、ao）的两个元音挤在 0.3 秒里、而识别窗长 0.32 秒，窗里混着两个元音，常被判成第三个（「爱」的后半段被认成 u）——这个识别器只见过稳定的单元音。把 <b>声音</b>换成童声：基频 300 Hz、声道 ×1.3 都超出了训练分布（≤ 280 Hz、≤ ×1.25），/ɛ/ 和 /ɔ/ 大多被认错（/ɔ/ 几乎全被认成 /ɑ/）——分布外的输入是 ASR 最常见的失败原因。',
            warn: '这是「孤立元音分类」，离真正的 ASR 还差很远：没有辅音、没有词、没有语言模型。真实系统要把一段语音对齐到一串文字：早期用 HMM + GMM，后来用 CTC（Graves 等 2006：允许空白 token、对所有对齐方式求和）或 RNN-T，Whisper 则用编码器-解码器 + 交叉注意力，直接自回归地输出文字（还带时间戳、语言、任务等特殊 token），训练数据是 68 万小时带字幕的网络音频（弱监督）。',
        },
        // ─────────────────────────────── 10 生产
        {
            id: 'prod', title: '教学版 → 生产版：真实的音频模型多了什么', tag: 'PROD',
            desc: '本页每一步都对应真实系统里的一个模块，只是规模小了几个数量级。下表把它们对上号；只列有把握的数字。',
            blocks: [
                { type: 'html', html: `<table class="lm-tbl"><tr><th>模块</th><th>本页（教学版）</th><th>真实系统</th></tr>
                    <tr><td>采样 / 量化</td><td>8 kHz、16 bit；μ-law 演示</td><td>语音 16 kHz（Whisper 的输入），音乐 32–48 kHz；电话 G.711 = 8 kHz × 8 bit μ-law / A-law = 64 kbps；WaveNet 用 8 bit μ-law（256 类 softmax）</td></tr>
                    <tr><td>特征</td><td>STFT（Hann）、HTK mel、10·log10</td><td>Whisper：16 kHz，25 ms 窗、10 ms 跳，80 个 mel 通道（large-v3 为 128），30 秒一段 → 3000 帧，编码器步长 2 → 1500 个位置</td></tr>
                    <tr><td>声码器</td><td>Griffin-Lim / 快速 GL</td><td>WaveNet（自回归、扩张因果卷积）→ WaveRNN / WaveGlow → HiFi-GAN（GAN，多周期 + 多尺度判别器）；扩散声码器 DiffWave、WaveGrad</td></tr>
                    <tr><td>编解码器</td><td>固定 MDCT + 8 层 RVQ（K = 64，k-means），31.25 帧/秒，0.19–1.5 kbps</td><td>SoundStream（2021）：全卷积编码器 / 解码器 + RVQ，24 kHz，3–18 kbps，quantizer dropout 支持可变码率；EnCodec（2022）：24 kHz 单声道 1.5 / 3 / 6 / 12 / 24 kbps（2–32 个码本 × 1024，75 帧/秒），多尺度 STFT 判别器；DAC（2023）等改进</td></tr>
                    <tr><td>token 语言模型</td><td>4.3 万参数，4 个码本 delay pattern，无条件续写</td><td>AudioLM（2022）：语义 token（w2v-BERT + k-means）→ 粗声学 token → 细声学 token 三级；MusicGen（2023）：单个 Transformer + delay pattern，EnCodec 32 kHz 的 4 个码本、50 帧/秒，文本 / 旋律条件，3 亿–33 亿参数</td></tr>
                    <tr><td>TTS</td><td>规则前端 + 共振峰合成</td><td>Tacotron 2（注意力 seq2seq → mel → WaveNet）、FastSpeech 2（非自回归，显式预测时长 / 音高 / 能量）、VITS（端到端 VAE + 流 + GAN）；VALL-E（2023）：EnCodec 6 kbps 的 8 个码本，第 1 个码本自回归、其余 7 个非自回归，3 秒录音即可模仿音色</td></tr>
                    <tr><td>扩散</td><td>PCA 潜空间（16 维）+ MLP 去噪 + DDIM 25 步</td><td>AudioLDM（2023）：mel 上训练的 VAE 潜空间 + U-Net 潜空间扩散，CLAP 文本条件，HiFi-GAN 声码器；之后的系统多用 DiT / 流匹配</td></tr>
                    <tr><td>ASR</td><td>5 个元音分类，2 421 个参数，合成数据</td><td>Whisper（2022）：68 万小时多语言弱监督数据，编码器-解码器 Transformer，39M（tiny）到 1550M（large）参数；自监督预训练 wav2vec 2.0 / HuBERT + CTC 微调</td></tr>
                    <tr><td>实时 / 流式</td><td>整段离线处理</td><td>编解码器、ASR、TTS 都要流式：因果卷积 / 有限前瞻、分块注意力，延迟 = 帧长 + 前瞻 + 计算；EnCodec 和 SoundStream 都有流式（因果）版本</td></tr></table>` },
                {
                    type: 'code', title: 'C++ · tokensFor / latencyMs（序列长度与延迟的估算）', src: `
                    // 一段音频变成多少个 token：帧率 × 时长 × 码本数
                    // EnCodec 24 kHz 6 kbps：75 × 8 = 600 个/秒；MusicGen：50 × 4 = 200 个/秒；本页：31.25 × 4 = 125 个/秒
                    int tokensFor(float seconds, float frameRate, int nq) { return (int)std::lround(seconds * frameRate * nq); }

                    // 流式处理的算法延迟：要攒够一帧（frameLen 个样本）再加上前瞻（lookahead 个样本）才能输出
                    float latencyMs(int frameLen, int lookahead, int sr) { return 1000.f * (frameLen + lookahead) / sr; }`,
                },
                {
                    type: 'previews', items: [
                        {
                            id: 'prodRate', label: '码率对比（对数横轴）', sub: '蓝 = 真实系统公开的码率 · 橙 = 本页的编解码器（Step 05 实测的 SNR 写在括号里，K = 64 时 Node 里跑满 1500 步的结果）', large: true, size: [720, 260],
                            render: (c) => {
                                const rows = [['CD：44.1 kHz × 16 bit × 2 声道', 1411], ['PCM 16 kHz × 16 bit（语音模型的输入）', 256], ['PCM 8 kHz × 16 bit（本页的原始音频）', 128], ['G.711：8 kHz × 8 bit μ-law', 64], ['EnCodec 24 kHz 最高档', 24], ['SoundStream 最高档', 18], ['EnCodec 24 kHz 6 kbps（VALL-E 用）', 6], ['SoundStream 最低档', 3], ['EnCodec 24 kHz 最低档', 1.5], ['本页 8 层 × 6 bit（≈ 26 dB）', 1.5], ['本页 4 层（≈ 20 dB）', 0.75], ['本页 1 层（≈ 8 dB）', 0.1875]];
                                const { ctx, w, h } = PL.draw.ctx(c), padL = 250, padR = 60, top = 6, bh = (h - top - 22) / rows.length, X = (v) => padL + Math.log10(v / 0.1) / Math.log10(2000 / 0.1) * (w - padL - padR);
                                ctx.font = LM.FONT(11); ctx.strokeStyle = '#1b1f25'; ctx.fillStyle = '#7a808a'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                                [0.1, 1, 10, 100, 1000].forEach((v) => { ctx.fillRect(X(v), top, 1, h - top - 22); ctx.fillText(v >= 1 ? `${v} kbps` : `${v}`, X(v), h - 18); });
                                rows.forEach(([lab, v], i) => { const y = top + i * bh, ours = lab.startsWith('本页'); ctx.fillStyle = ours ? LM.C.orange : LM.C.blue; ctx.globalAlpha = 0.85; ctx.fillRect(padL, y + bh * 0.18, Math.max(2, X(v) - padL), bh * 0.64); ctx.globalAlpha = 1; ctx.fillStyle = ours ? LM.C.orange : LM.C.gray; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(lab, padL - 6, y + bh / 2); ctx.fillStyle = LM.C.dim; ctx.textAlign = 'left'; ctx.fillText(v >= 1 ? `${v}` : v.toFixed(3), X(v) + 4, y + bh / 2); });
                            },
                        },
                        {
                            id: 'prodTok', label: '10 秒音频 = 多少个位置 / token', sub: '同样 10 秒：codec token 的序列比一句话的文字长得多 · 这是音频 LM 要用 delay pattern、分层建模的原因', size: [720, 190],
                            render: (c) => {
                                const rows = [['EnCodec 6 kbps 摊平（75 帧/秒 × 8 个码本）', 6000], ['MusicGen 摊平（50 × 4）', 2000], ['MusicGen delay pattern（50 个位置/秒）', 500], ['本页 摊平（31.25 × 4）', 1250], ['本页 delay pattern（31.25 个位置/秒）', 313], ['Whisper 编码器（10 ms × 步长 2 → 50 个位置/秒）', 500]];
                                LM.draw.bars(c, rows.map((r) => r[1]), rows.map((r) => r[0]), { horizontal: true, digits: 0, colors: rows.map((r) => (r[0].startsWith('本页') ? LM.C.orange : LM.C.blue)), labelPx: 11 });
                            },
                        },
                    ],
                },
                { type: 'note', html: '本页的几条路线在真实系统里是组合起来用的：MusicGen = 编解码器（第 05 步）+ token LM（第 06 步）；AudioLDM = mel（第 03 步）+ 潜空间扩散（第 07 步）+ 神经声码器（第 04 步）；VALL-E = 文本前端（第 08 步）+ 编解码器 + token LM；Whisper = log-mel（第 03 步）+ 卷积 / Transformer 编码器（第 09 步）+ 文本解码器（第 05 篇）。下一步可以读第 11 篇（扩散与流匹配的细节）和第 12 篇（视频生成：时间维度上同样的「压缩 → token / 潜空间 → 生成」套路）。' },
                { type: 'warn', html: '真实系统的难点大多不在模型结构，而在数据和评估：语音 / 音乐数据的版权与授权、说话人隐私（声音克隆的滥用，VALL-E 论文专门讨论了风险）、评估指标（主观 MOS、ViSQOL、FAD、WER）和人耳感知的差距。本页所有的「好坏」都是在极其规整的合成数据上测的，不能直接外推。' },
            ],
        },
    ].map(P13.mkStep),

    graph: {
        title: '音频流水线 · 从波形到 token 再回到波形',
        intro: '每个节点 = 一段真实的计算（页面里由 JS 参考实现执行，代码块是对应的 C++）。pin = 数据名 · 形状：波形 Wave 是 8 kHz 的单声道采样；STFT / mel 是 [帧 × 频率] 的谱；Tokens 是 RVQ 编解码器输出的 [帧 × 码本] 整数。同名数据自动连线，<b>HISTORY</b> 节点是页面里实时训练的模型（codec、token 语言模型、扩散、识别器）。',
        types: {
            wave: { color: '#56b6c2', label: 'wave · 波形（float32，8 kHz）' },
            spec: { color: '#e0a060', label: 'spec · 频谱 / mel（[帧 × 频率]）' },
            token: { color: '#c678dd', label: 'token · 离散音频 token（[帧 × 码本]）' },
            text: { color: '#8a8f98', label: 'text · 文字 / 类别' },
        },
        common: {
            title: 'Common · 本系列自带的 C++ 库（pages/cpp/*.h，可直接编译）+ 本页公用的信号处理函数', open: false,
            blocks: LM.CPP.blocks(['gpt']).concat([{ label: '━ 本页公用：FFT / 窗函数 / STFT / ISTFT ━', code: P13_CPP_COMMON }]),
        },
        nodes: [
            {
                id: 'synth', title: 'makeClip', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'wave', desc: '旋律 / 啁啾 / 共振峰元音',
                dispatch: { kind: 'CPU', name: 'makeClip', count: '每段 1–2 秒 × 8000 采样/秒' },
                outputs: [{ name: 'Wave', format: '[N] f32 · 8 kHz', type: 'wave' }],
            },
            {
                id: 'stft', title: 'stft', tag: 'CPU', lifetime: 'param', step: 'stft', desc: 'Hann 窗 + FFT，逐帧',
                dispatch: { kind: 'CPU', name: 'stft', count: '帧数 × N log N' },
                inputs: [{ name: 'Wave', format: '[N] f32', type: 'wave' }],
                outputs: [{ name: 'Spec', format: '[T × (N/2+1)] complex', type: 'spec' }],
            },
            {
                id: 'mel', title: 'logMel', tag: 'CPU', lifetime: 'param', step: 'mel', desc: '三角滤波器组 + log',
                dispatch: { kind: 'CPU', name: 'logMel', count: 'T × F × M' },
                inputs: [{ name: 'Spec', format: '|X|²', type: 'spec' }],
                outputs: [{ name: 'LogMel', format: '[T × M] f32', type: 'spec' }],
            },
            {
                id: 'gl', title: 'griffinLim', tag: 'CPU', lifetime: 'history', step: 'griffin', desc: '只给幅度，迭代猜相位',
                dispatch: { kind: 'CPU', name: 'griffinLim', count: '每次迭代一次 ISTFT + 一次 STFT' },
                inputs: [{ name: 'Spec', format: '|X|（丢掉相位）', type: 'spec' }],
                outputs: [{ name: 'WaveGL', format: '[N] f32', type: 'wave' }],
            },
            {
                id: 'enc', title: 'rvqEncode', tag: 'CPU', lifetime: 'history', step: 'codec', desc: 'MDCT 帧特征 → Q 层残差量化',
                dispatch: { kind: 'CPU', name: 'rvqEncode', count: '每帧 Q 次最近邻搜索' },
                inputs: [{ name: 'Wave', format: '[N] f32', type: 'wave' }],
                outputs: [{ name: 'Tokens', format: '[T × Q] int', type: 'token' }],
            },
            {
                id: 'dec', title: 'rvqDecode', tag: 'CPU', category: 'output', lifetime: 'history', step: 'codec', desc: '码字求和 → IMDCT 重叠相加',
                dispatch: { kind: 'CPU', name: 'rvqDecode', count: 'T × Q 次查表' },
                inputs: [{ name: 'Tokens', format: '[T × Q] int', type: 'token' }],
                outputs: [{ name: 'WaveRec', format: '[N] f32', type: 'wave' }],
            },
            {
                id: 'lm', title: 'tokenLMGenerate', tag: 'CPU', category: 'output', lifetime: 'history', step: 'tokenlm', desc: 'delay pattern + 自回归续写',
                dispatch: { kind: 'CPU', name: 'tokenLMGenerate', count: '每生成一帧 = 一次 Transformer 前向' },
                inputs: [{ name: 'Tokens', format: '训练语料', type: 'token' }],
                outputs: [{ name: 'TokensGen', format: '[T × Q] int', type: 'token' }],
            },
            {
                id: 'dif', title: 'ddimSample', tag: 'CPU', category: 'output', lifetime: 'history', step: 'diffusion', desc: 'mel 的 PCA 潜空间里去噪',
                dispatch: { kind: 'CPU', name: 'ddimSample', count: '每个样本 × 采样步数' },
                inputs: [{ name: 'LogMel', format: '训练数据', type: 'spec' }],
                outputs: [{ name: 'MelGen', format: '[T × M] f32', type: 'spec' }],
            },
            {
                id: 'tts', title: 'formantSynth', tag: 'CPU', category: 'output', lifetime: 'param', step: 'tts', desc: '文本 → 音素 → F0 / 共振峰 → 波形',
                dispatch: { kind: 'CPU', name: 'ttsFrontend · formantSynth', count: '每个采样点一次源 + 共振滤波' },
                inputs: [{ name: 'Text', format: '汉字串', type: 'text', external: true }],
                outputs: [{ name: 'WaveTTS', format: '[N] f32', type: 'wave' }],
            },
            {
                id: 'asr', title: 'vowelNetForward', tag: 'CPU', category: 'output', lifetime: 'history', step: 'asr', desc: 'log-mel → 1D 卷积 → 元音',
                dispatch: { kind: 'CPU', name: 'vowelNetForward', count: '每段音频一次前向' },
                inputs: [{ name: 'LogMel', format: '[T × M] f32', type: 'spec' }],
                outputs: [{ name: 'Label', format: '5 个元音的概率', type: 'text' }],
            },
        ],
    },

    references: [
        { title: 'Signal Estimation from Modified Short-Time Fourier Transform', author: 'D. Griffin, J. Lim', venue: 'IEEE Trans. ASSP', year: 1984, note: 'Griffin-Lim 相位重建（Step 04）' },
        { title: 'A Scale for the Measurement of the Psychological Magnitude Pitch', author: 'S. S. Stevens, J. Volkmann, E. B. Newman', venue: 'JASA', year: 1937, note: 'mel 刻度的来源（Step 03）' },
        { title: 'WaveNet: A Generative Model for Raw Audio', author: 'A. van den Oord et al.', venue: 'arXiv', year: 2016, note: '直接逐采样点生成波形' },
        { title: 'Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions', author: 'J. Shen et al.', venue: 'ICASSP', year: 2018, note: 'Tacotron 2：文本 → mel → 声码器（Step 08）' },
        { title: 'HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis', author: 'J. Kong, J. Kim, J. Bae', venue: 'NeurIPS', year: 2020, note: '神经声码器（Step 04 / 08）' },
        { title: 'SoundStream: An End-to-End Neural Audio Codec', author: 'N. Zeghidour et al.', venue: 'IEEE/ACM TASLP', year: 2021, note: '残差向量量化（RVQ）音频编解码器（Step 05）' },
        { title: 'High Fidelity Neural Audio Compression', author: 'A. Défossez, J. Copet, G. Synnaeve, Y. Adi', venue: 'TMLR', year: 2023, note: 'EnCodec（Step 05）' },
        { title: 'AudioLM: a Language Modeling Approach to Audio Generation', author: 'Z. Borsos et al.', venue: 'IEEE/ACM TASLP', year: 2023, note: '在音频 token 上做语言模型（Step 06）' },
        { title: 'Simple and Controllable Music Generation', author: 'J. Copet et al.', venue: 'NeurIPS', year: 2023, note: 'MusicGen：多码本的 delay pattern（Step 06）' },
        { title: 'Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers', author: 'C. Wang et al.', venue: 'arXiv', year: 2023, note: 'VALL-E：codec token 上的 TTS' },
        { title: 'AudioLDM: Text-to-Audio Generation with Latent Diffusion Models', author: 'H. Liu et al.', venue: 'ICML', year: 2023, note: 'mel 潜空间里的扩散（Step 07）' },
        { title: 'Progressive Distillation for Fast Sampling of Diffusion Models', author: 'T. Salimans, J. Ho', venue: 'ICLR', year: 2022, note: 'v-prediction（Step 07 提到）' },
        { title: 'Robust Speech Recognition via Large-Scale Weak Supervision', author: 'A. Radford et al.', venue: 'ICML', year: 2023, note: 'Whisper（Step 09）' },
    ],
});
