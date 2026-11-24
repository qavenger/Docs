// p13_proto.js — 第 13 篇的 Node 原型：加载 13_audio_generation.lab.js 里 PipelineLab.mount 之前的部分（= 页面同一份 P13 代码），
// 做正确性校验、计时与参数实验。用法（在 pages/ 下）：node tools/p13_proto.js <what>
const fs = require('fs');
const path = require('path');
const LM = require('../llm_common.js');
const src = fs.readFileSync(path.join(__dirname, '../13_audio_generation.lab.js'), 'utf8');
const head = src.slice(0, src.indexOf('PipelineLab.mount('));
const P13 = new Function('LM', head + '\nreturn P13;')(LM);
module.exports = { P13, LM };
if (require.main !== module) return;

const what = process.argv[2] || 'check';
const SR = P13.SR;
const t = () => Date.now();
const db = (x) => 10 * Math.log10(x);
function snr(x, y) { let s = 0, e = 0; for (let i = 0; i < x.length; i++) { s += x[i] * x[i]; e += (x[i] - y[i]) ** 2; } return db(s / (e || 1e-30)); }

if (what === 'check') {
    // 1. FFT vs naive DFT
    const n = 64, re = new Float64Array(n), im = new Float64Array(n), r = LM.rng(1);
    for (let i = 0; i < n; i++) { re[i] = r() - 0.5; im[i] = r() - 0.5; }
    const R0 = Float64Array.from(re), I0 = Float64Array.from(im);
    P13.fft(re, im, false);
    let md = 0;
    for (let k = 0; k < n; k++) { let sr = 0, si = 0; for (let j = 0; j < n; j++) { const a = -2 * Math.PI * j * k / n; sr += R0[j] * Math.cos(a) - I0[j] * Math.sin(a); si += R0[j] * Math.sin(a) + I0[j] * Math.cos(a); } md = Math.max(md, Math.abs(sr - re[k]), Math.abs(si - im[k])); }
    P13.fft(re, im, true);
    let mi = 0; for (let i = 0; i < n; i++) mi = Math.max(mi, Math.abs(re[i] - R0[i]), Math.abs(im[i] - I0[i]));
    console.log('fft vs dft max err', md.toExponential(2), ' roundtrip', mi.toExponential(2));
    // 2. STFT → iSTFT
    const L = 20000, x = new Float64Array(L); for (let i = 0; i < L; i++) x[i] = Math.sin(i * 0.05) * 0.3 + 0.2 * (r() - 0.5);
    for (const [N, H] of [[256, 64], [512, 128], [64, 16], [1024, 256], [256, 128], [256, 32]]) {
        const sp = P13.stft(x, N, H); const y = P13.istft(sp.re, sp.im, sp.T, N, H, L);
        // naive check of one frame
        const tq = 7, w = P13.hann(N); let e = 0;
        for (let k = 0; k < sp.nb; k++) { let sr = 0, si = 0; for (let j = 0; j < N; j++) { const ii = tq * H - N / 2 + j; const v = ii >= 0 && ii < L ? x[ii] * w[j] : 0; const a = -2 * Math.PI * j * k / N; sr += v * Math.cos(a); si += v * Math.sin(a); } e = Math.max(e, Math.abs(sr - sp.re[tq * sp.nb + k]), Math.abs(si - sp.im[tq * sp.nb + k])); }
        const t8 = sp.T - 1; let e2 = 0; for (let k = 0; k < sp.nb; k++) { let sr = 0, si = 0; for (let j = 0; j < N; j++) { const ii = t8 * H - N / 2 + j; const v = ii >= 0 && ii < L ? x[ii] * w[j] : 0; const a = -2 * Math.PI * j * k / N; sr += v * Math.cos(a); si += v * Math.sin(a); } e2 = Math.max(e2, Math.abs(sr - sp.re[t8 * sp.nb + k]), Math.abs(si - sp.im[t8 * sp.nb + k])); }
        console.log(`stft N=${N} H=${H}: T=${sp.T} frame err ${e.toExponential(1)} last ${e2.toExponential(1)}  istft SNR ${snr(x, y).toFixed(1)} dB`);
    }
    // 3. MDCT PR + Parseval
    for (const M of [64, 128, 256]) {
        const m = P13.mdct(x, M), y = P13.imdct(m.C, m.F, M, L);
        let ec = 0, ex = 0; for (const v of m.C) ec += v * v; for (const v of x) ex += v * v;
        // direct formula check on one frame
        const p = P13.mdctPlan(M), tq = 5, seg = new Float64Array(2 * M); for (let n2 = 0; n2 < 2 * M; n2++) { const i = (tq - 1) * M + n2; seg[n2] = i >= 0 && i < L ? x[i] : 0; }
        let e = 0; for (let k = 0; k < M; k++) { let s = 0; for (let n2 = 0; n2 < 2 * M; n2++) s += p.w[n2] * seg[n2] * Math.cos(Math.PI / M * (n2 + 0.5 + M / 2) * (k + 0.5)); s *= Math.sqrt(2 / M); e = Math.max(e, Math.abs(s - m.C[tq * M + k])); }
        console.log(`mdct M=${M}: F=${m.F} direct err ${e.toExponential(1)} PR SNR ${snr(x, y).toFixed(1)} dB  energy ratio ${(ec / ex).toFixed(6)}`);
    }
    // 4. mel bank
    for (const [N, nm] of [[256, 40], [64, 64], [512, 80], [1024, 40]]) { const b = P13.melBank(N, nm); console.log(`mel N=${N} nMels=${nm}: empty filters ${b.empty}`); }
}

if (what === 'timing') {
    const L = 20000, x = new Float64Array(L), r = LM.rng(2); for (let i = 0; i < L; i++) x[i] = Math.sin(i * 0.07) * 0.3 + 0.1 * (r() - 0.5);
    for (const [N, H] of [[256, 64], [512, 128], [1024, 256], [64, 16], [256, 32]]) {
        let sp = P13.stft(x, N, H); P13.istft(sp.re, sp.im, sp.T, N, H, L);
        const t0 = P13.now(); for (let k = 0; k < 20; k++) sp = P13.stft(x, N, H, sp); const t1 = P13.now();
        for (let k = 0; k < 20; k++) P13.istft(sp.re, sp.im, sp.T, N, H, L); const t2 = P13.now();
        const A = P13.magnitude(sp), G = P13.glCreate(A, sp.T, N, H, L, {}); P13.glStep(G);
        const t3 = P13.now(); for (let k = 0; k < 10; k++) P13.glStep(G); const t4 = P13.now();
        console.log(`N=${N} H=${H} T=${sp.T}: stft ${((t1 - t0) / 20).toFixed(2)} ms  istft ${((t2 - t1) / 20).toFixed(2)} ms  GL iter ${((t4 - t3) / 10).toFixed(2)} ms`);
    }
    for (const M of [128, 256]) { const t0 = P13.now(); for (let k = 0; k < 10; k++) { const m = P13.mdct(x, M); P13.imdct(m.C, m.F, M, L); } console.log(`mdct+imdct M=${M} 2.5 s: ${((P13.now() - t0) / 10).toFixed(2)} ms`); }
}

if (what === 'codec') {
    const M = +(process.argv[3] || 256), K = +(process.argv[4] || 64), NQ = 8, B = +(process.argv[5] || 128), gamma = +(process.argv[6] || 0.95), steps = +(process.argv[7] || 1500);
    const beat = 2048, rand = LM.rng(5);
    const mk = (n, seed) => { const r = LM.rng(seed), out = []; for (let i = 0; i < n; i++) { const mel = P13.makeMelody(r, 8 + Math.floor(r() * 5)); out.push(P13.renderMelody(mel, beat, i % 2, 0.22, 0).x); } return out; };
    const train = mk(48, 11), test = mk(6, 999);
    const frames = (xs) => { const all = []; let F = 0; xs.forEach((x) => { const m = P13.mdct(x, M); all.push(m); F += m.F; }); const X = new Float64Array(F * M); let o = 0; all.forEach((m) => { X.set(m.C, o); o += m.C.length; }); return { X, F, all }; };
    const TR = frames(train), TE = frames(test);
    console.log(`M=${M} K=${K} B=${B} γ=${gamma}: train frames ${TR.F} (${(train.reduce((s, x) => s + x.length, 0) / SR).toFixed(1)} s), test ${TE.F}`);
    const Q = P13.rvqCreate(M, K, NQ, { gamma, seed: 3 });
    const t0 = P13.now();
    const evalSNR = () => {
        const out = [];
        for (let nq = 1; nq <= NQ; nq++) {
            let s = 0, e = 0;
            TE.all.forEach((m, i) => { const codes = P13.rvqEncode(Q, m.C, m.F, NQ); const Xh = P13.rvqDecode(Q, codes, m.F, nq); const y = P13.imdct(Xh, m.F, M, test[i].length); for (let j = 0; j < y.length; j++) { s += test[i][j] ** 2; e += (test[i][j] - y[j]) ** 2; } });
            out.push(db(s / e));
        }
        return out;
    };
    for (let st = 1; st <= steps; st++) {
        const rr = P13.rvqTrainStep(Q, TR.X, TR.F, B, rand);
        if (st === 1 || st % 250 === 0) console.log(`step ${st}: train ratios(dB) ${rr.map((v) => (-db(v)).toFixed(1)).join(' ')} | test SNR ${evalSNR().map((v) => v.toFixed(1)).join(' ')}`);
    }
    const ms = (P13.now() - t0) / steps;
    console.log(`train step ≈ ${ms.toFixed(2)} ms (incl. evals)`);
    const t1 = P13.now(); for (let k = 0; k < 50; k++) P13.rvqTrainStep(Q, TR.X, TR.F, B, rand); console.log(`pure train step ${((P13.now() - t1) / 50).toFixed(2)} ms`);
    // usage
    const codes = P13.rvqEncode(Q, TR.X, TR.F, NQ);
    for (let l = 0; l < NQ; l++) { const h = new Float64Array(K); for (let t2 = 0; t2 < TR.F; t2++) h[codes[t2 * NQ + l]]++; let H = 0, used = 0; h.forEach((c) => { if (c) { used++; const p = c / TR.F; H -= p * Math.log2(p); } }); console.log(`layer ${l}: used ${used}/${K}, entropy ${H.toFixed(2)} bits (max ${Math.log2(K)})`); }
    const fps = SR / M; console.log(`bitrate per layer ${(fps * Math.log2(K)).toFixed(0)} bps; frames/s ${fps}`);
}

// ── 第 06 步原型：LM 自己的编解码器（4 层）→ 数据集打 token → 延迟排布 → 训练 → 生成 → 解码 → 音高分析 ──
function melodySet(n, seed, beat) { const r = LM.rng(seed), out = []; for (let i = 0; i < n; i++) { const mel = P13.makeMelody(r, 8 + Math.floor(r() * 5)); out.push(Object.assign(P13.renderMelody(mel, beat, i % 2, 0.22, 0), { mel, tb: i % 2 })); } return out; }
function pitchReport(y, beat) {
    // 每拍中间 1024 个样本做 FFT：对 8 个音阶音打分（谐波能量占比），返回 [{ midi, clarity }]
    const out = [], N = 1024, w = P13.hann(N);
    for (let s = 0; s + beat <= y.length; s += beat) {
        const re = new Float64Array(N), im = new Float64Array(N), o = s + (beat - N) / 2;
        for (let i = 0; i < N; i++) re[i] = (y[o + i] || 0) * w[i];
        P13.fft(re, im, false);
        const P = new Float64Array(N / 2); let tot = 0; for (let k = 1; k < N / 2; k++) { P[k] = re[k] ** 2 + im[k] ** 2; tot += P[k]; }
        let best = -1, bs = -1;
        let bw = -1; P13.SCALE.forEach((m) => { const f0 = P13.midiHz(m); let e = 0, ew = 0; for (let h = 1; h * f0 < 3800; h++) { const kc = Math.round(h * f0 * N / SR); let eh = 0; for (let k = kc - 2; k <= kc + 2; k++) if (k > 0 && k < N / 2) eh += P[k]; e += eh; ew += eh / h; } if (ew > bw) { bw = ew; bs = e; best = m; } });
        out.push({ midi: best, clarity: tot > 1e-9 ? bs / tot : 0, energy: tot });
    }
    return out;
}
if (what === 'lm') {
    const K = +(process.argv[3] || 64), NQ = 4, M = 256, beat = 2048, steps = +(process.argv[4] || 3000), B = +(process.argv[5] || 4), T = +(process.argv[6] || 32), lr = +(process.argv[7] || 3e-3);
    const train = melodySet(48, 11, beat), test = melodySet(6, 999, beat);
    const frames = (xs) => { const all = xs.map((s) => P13.mdct(s.x, M)); const F = all.reduce((a, m) => a + m.F, 0), X = new Float64Array(F * M); let o = 0; all.forEach((m) => { X.set(m.C, o); o += m.C.length; }); return { X, F, all }; };
    const TR = frames(train), TE = frames(test);
    let t0 = P13.now();
    const Q = P13.rvqCreate(M, K, NQ, { gamma: 0.98, seed: 3 }), rr = LM.rng(5);
    for (let s = 0; s < 600; s++) P13.rvqTrainStep(Q, TR.X, TR.F, 64, rr);
    console.log(`codec 600 steps: ${(P13.now() - t0).toFixed(0)} ms`);
    t0 = P13.now();
    const codesTR = P13.rvqEncode(Q, TR.X, TR.F, NQ), codesTE = P13.rvqEncode(Q, TE.X, TE.F, NQ);
    console.log(`tokenize ${TR.F + TE.F} frames: ${(P13.now() - t0).toFixed(0)} ms`);
    // test SNR with 4 layers
    { let s = 0, e = 0, o = 0; TE.all.forEach((m, i) => { const c = codesTE.subarray(o * NQ, (o + m.F) * NQ); o += m.F; const y = P13.imdct(P13.rvqDecode(Q, c, m.F, NQ), m.F, M, test[i].x.length); for (let j = 0; j < y.length; j++) { s += test[i].x[j] ** 2; e += (test[i].x[j] - y[j]) ** 2; } }); console.log(`LM codec test SNR (4 layers) ${db(s / e).toFixed(1)} dB`); }
    const seen = new Set(); for (let t2 = 0; t2 < TR.F; t2++) seen.add(Array.from(codesTR.subarray(t2 * NQ, t2 * NQ + NQ)).join(','));
    console.log(`distinct frame 4-tuples in train: ${seen.size}`);
    const sTR = P13.delayPattern(codesTR, TR.F, NQ, K), sTE = P13.delayPattern(codesTE, TE.F, NQ, K), PTR = TR.F + NQ - 1, PTE = TE.F + NQ - 1;
    const cfg = { NQ, K, T, d: 32, L: 2, H: 4, ff: 128 };
    const m = P13.tlmCreate(cfg, 1);
    console.log(`params ${m.nParams}`);
    const opt = new LM.AdamW(m.params, { lr, b1: 0.9, b2: 0.95, eps: 1e-8, wd: 0.1 }), rand = LM.rng(9);
    const vb = P13.tlmBatch(sTE, PTE, NQ, K, 16, T, LM.rng(77));
    t0 = P13.now(); let acc = 0, n = 0;
    for (let it = 0; it < steps; it++) {
        const b = P13.tlmBatch(sTR, PTR, NQ, K, B, T, rand);
        const r = P13.tlmTrainStep(m, opt, b.x, b.y, B, T, LM.cosineLR(it, { maxLR: lr, minLR: lr / 10, warmup: 100, total: steps }));
        acc += r.loss; n++;
        if ((it + 1) % 250 === 0) { const ve = P13.tlmEval(m, vb.x, vb.y, 16, T); console.log(`it ${it + 1}: train ${(acc / n).toFixed(3)} val ${ve.map((v) => v.toFixed(3)).join(' ')}  (${((P13.now() - t0) / (it + 1)).toFixed(2)} ms/step)`); acc = 0; n = 0; }
    }
    // generation: prompt = first 8 frames of each test melody, continue 88 positions
    let o = 0;
    const rep = [];
    TE.all.forEach((mm, i) => {
        const c = codesTE.subarray(o * NQ, (o + mm.F) * NQ); o += mm.F;
        const F0 = 8, prompt = P13.delayPattern(c.subarray(0, F0 * NQ), F0, NQ, K);
        const t1 = P13.now();
        const seq = P13.tlmGenerate(m, prompt, F0, 96, { temperature: 1.0, rand: LM.rng(100 + i), force: c, forceF: F0 });
        const gen = P13.undelay(seq, F0 + 96, NQ), Fg = gen.length / NQ;
        const ms = P13.now() - t1;
        let valid = 0; for (let t2 = 0; t2 < Fg; t2++) if (seen.has(Array.from(gen.subarray(t2 * NQ, t2 * NQ + NQ)).join(','))) valid++;
        const y = P13.imdct(P13.rvqDecode(Q, gen, Fg, NQ), Fg, M, (Fg - 1) * M);
        const pr = pitchReport(y, beat);
        const clar = pr.filter((p) => p.energy > 1e-3).map((p) => p.clarity);
        rep.push({ ms, valid: valid / Fg, clar: clar.reduce((a, b) => a + b, 0) / clar.length, notes: pr.map((p) => (p.energy > 1e-3 ? P13.noteName(p.midi) + ':' + p.clarity.toFixed(2) : '—')).join(' ') });
    });
    rep.forEach((r, i) => console.log(`gen ${i}: ${r.ms.toFixed(1)} ms, valid frames ${(100 * r.valid).toFixed(0)}%, mean clarity ${r.clar.toFixed(2)}\n   ${r.notes}`));
    // reference: clarity of real test melodies
    const pr0 = pitchReport(test[0].x, beat); console.log('real test0:', pr0.map((p) => P13.noteName(p.midi) + ':' + p.clarity.toFixed(2)).join(' '));
}

if (what === 'tts') {
    for (const text of ['妈麻马骂', '你好', '我爱你', '啊诶衣喔乌', '猫爱鱼', '喵']) {
        const t0 = P13.now(), r = P13.tts(text, { f0: 120, rate: 1, vtl: 1 }), ms = P13.now() - t0;
        let pk = 0, rms = 0; for (const v of r.x) { pk = Math.max(pk, Math.abs(v)); rms += v * v; } rms = Math.sqrt(rms / r.x.length);
        console.log(`${text}: ${r.syls.map((s) => P13.pinyinOf(s.s)).join(' ')} | ${(r.x.length / SR).toFixed(2)} s, peak ${pk.toFixed(3)} rms ${rms.toFixed(3)}, ${ms.toFixed(1)} ms | segs ${r.segs.map((s) => s.ph + '@' + s.t0.toFixed(2)).join(' ')}`);
    }
    // vowel spectra: formant peak check via LPC-free method: smoothed spectrum peaks
    for (const v of P13.VOWELS) {
        const x = P13.vowelClip(v, { f0: 120, vtl: 1 }, 0.4, LM.rng(1)), N = 1024, re = new Float64Array(N), im = new Float64Array(N), w = P13.hann(N);
        for (let i = 0; i < N; i++) re[i] = x[1000 + i] * w[i];
        P13.fft(re, im, false);
        const P = []; for (let k = 0; k < N / 2; k++) P.push(10 * Math.log10(re[k] ** 2 + im[k] ** 2 + 1e-12));
        // harmonic peaks (every 120 Hz) → find local maxima of harmonic envelope
        const harm = []; for (let h = 1; h * 120 < 3900; h++) { const k = Math.round(h * 120 * N / SR); harm.push({ f: h * 120, d: Math.max(P[k - 1], P[k], P[k + 1]) }); }
        const pk = harm.filter((q, i) => i > 0 && i < harm.length - 1 && q.d > harm[i - 1].d && q.d >= harm[i + 1].d).map((q) => q.f);
        console.log(`${v.key}: target F ${v.F.slice(0, 3).join('/')} → harmonic-envelope peaks ${pk.join(' ')}`);
    }
}

if (what === 'asr') {
    const nTr = +(process.argv[3] || 600), steps = +(process.argv[4] || 600), B = +(process.argv[5] || 16), lr = +(process.argv[6] || 3e-3);
    let t0 = P13.now();
    const tr = P13.asrDataset(nTr, 21), te = P13.asrDataset(250, 4242);
    console.log(`datasets: ${(P13.now() - t0).toFixed(0)} ms, D = ${tr.D} (NM ${P13.ASR.NM} × T ${P13.ASR.T})`);
    let mn = Infinity, mx = -Infinity, s = 0; for (const v of tr.X) { mn = Math.min(mn, v); mx = Math.max(mx, v); s += v; } console.log(`feature range ${mn.toFixed(2)} .. ${mx.toFixed(2)}, mean ${(s / tr.X.length).toFixed(2)}`);
    const m = P13.asrCreate(5, +(process.argv[7] || 32)), opt = new LM.AdamW(m.params, { lr, b1: 0.9, b2: 0.999, wd: 0 }), rand = LM.rng(3), D = tr.D;
    const evalSet = (ds) => LM.AG.noGrad(() => { const out = P13.asrForward(m, ds.X, ds.n), L = out.logits.data; let ok = 0; const conf = new Int32Array(25); for (let i = 0; i < ds.n; i++) { const p = LM.argmax(L, i * 5, 5); conf[ds.y[i] * 5 + p]++; if (p === ds.y[i]) ok++; } return { acc: ok / ds.n, conf }; });
    t0 = P13.now(); const Xb = new Float32Array(B * D), yb = new Int32Array(B);
    for (let it = 1; it <= steps; it++) {
        for (let b = 0; b < B; b++) { const i = Math.floor(rand() * tr.n); Xb.set(tr.X.subarray(i * D, (i + 1) * D), b * D); yb[b] = tr.y[i]; }
        LM.zeroGrad(m.params);
        const out = P13.asrForward(m, Xb, B), loss = LM.AG.crossEntropy(out.logits, yb);
        LM.AG.backward(loss); opt.step(lr);
        if (it % 100 === 0) { const e = evalSet(te), etr = evalSet({ X: tr.X.subarray(0, 200 * D), y: tr.y.subarray(0, 200), n: 200 }); console.log(`it ${it}: loss ${loss.item().toFixed(3)} train acc ${etr.acc.toFixed(3)} test acc ${e.acc.toFixed(3)} (${((P13.now() - t0) / it).toFixed(2)} ms/step)`); }
    }
    const e = evalSet(te); console.log('confusion (rows = truth a e i o u):'); for (let r = 0; r < 5; r++) console.log('  ' + Array.from(e.conf.subarray(r * 5, r * 5 + 5)).join('\t'));
    // TTS posteriorgram
    const r = P13.tts('啊诶衣喔乌', { f0: 120 }), L = P13.ASR.L, hopW = 320, nW = Math.floor((r.x.length - L) / hopW) + 1, Xw = new Float32Array(nW * D);
    let p2 = 0; for (const v of r.x) p2 += v * v; p2 /= r.x.length; const g = 1 / Math.sqrt(p2 * 40);
    for (let w = 0; w < nW; w++) { const seg = new Float64Array(L); for (let j = 0; j < L; j++) seg[j] = r.x[w * hopW + j] * g; P13.asrFeatures(seg, Xw, w * D); }
    const outW = LM.AG.noGrad(() => P13.asrForward(m, Xw, nW)).logits.data;
    console.log('TTS 啊诶衣喔乌 windows:', Array.from({ length: nW }, (_, w) => 'aeiou'[LM.argmax(outW, w * 5, 5)]).join(''));
}

if (what === 'dif') {
    const n = +(process.argv[3] || 1500), steps = +(process.argv[4] || 3000), B = +(process.argv[5] || 16), lr = +(process.argv[6] || 2e-3), hid = +(process.argv[7] || 192);
    let t0 = P13.now();
    const tr = P13.difDataset(n, 31), te = P13.difDataset(200, 777), D = P13.DIF.D;
    console.log(`dataset ${n}: ${(P13.now() - t0).toFixed(0)} ms`);
    let mx = -9, mn = 9; for (const v of tr.X) { mx = Math.max(mx, v); mn = Math.min(mn, v); } console.log('range', mn.toFixed(2), mx.toFixed(2));
    const m = P13.difCreate(3, hid), opt = new LM.AdamW(m.params, { lr, b1: 0.9, b2: 0.999, wd: 0 }), rand = LM.rng(4);
    t0 = P13.now(); let acc = 0;
    for (let it = 1; it <= steps; it++) {
        acc += P13.difTrainStep(m, opt, tr.X, tr.n, B, rand, LM.cosineLR(it, { maxLR: lr, minLR: lr / 10, warmup: 100, total: steps }));
        if (it % 500 === 0) { console.log(`it ${it}: loss ${(acc / 500).toFixed(4)} (${((P13.now() - t0) / it).toFixed(2)} ms/step)`); acc = 0; }
    }
    t0 = P13.now();
    const smp = P13.difSample(m, 16, 25, LM.rng(8));
    console.log(`sample 16 × 25 steps: ${(P13.now() - t0).toFixed(0)} ms`);
    const nn = (x, o, set) => { let best = Infinity; for (let i = 0; i < set.n; i++) { let s = 0; for (let j = 0; j < D; j++) s += (x[o + j] - set.X[i * D + j]) ** 2; best = Math.min(best, s); } return Math.sqrt(best / D); };
    const dS = [], dT = []; for (let i = 0; i < 16; i++) dS.push(nn(smp.x, i * D, tr)); for (let i = 0; i < 16; i++) dT.push(nn(te.X, i * D, tr));
    console.log('NN rms dist to train: samples', dS.map((v) => v.toFixed(2)).join(' '), '| real test', dT.map((v) => v.toFixed(2)).join(' '));
    // noise baseline
    const rn = new Float32Array(D); const r2 = LM.rng(1); for (let j = 0; j < D; j++) rn[j] = Math.max(-1, Math.min(1, LM.randn(r2))); console.log('random-noise NN dist', nn(rn, 0, tr).toFixed(2));
    const ascii = (x, o) => { const ch = ' .:-=+*#%@'; let s = ''; for (let mm = P13.DIF.NM - 1; mm >= 0; mm--) { for (let tt = 0; tt < P13.DIF.TF; tt++) s += ch[Math.max(0, Math.min(9, Math.floor((x[o + mm * P13.DIF.TF + tt] + 1) * 5)))]; s += '\n'; } return s; };
    for (let i = 0; i < 4; i++) { const a = ascii(smp.x, i * D).split('\n'), b = ascii(tr.X, i * D).split('\n'); console.log(a.map((l, k) => l + '   |   ' + b[k]).join('\n')); }
    t0 = P13.now(); const y = P13.difToWave(smp.x.subarray(0, D)); console.log(`vocoder ${(P13.now() - t0).toFixed(1)} ms, len ${y.length}`);
}
