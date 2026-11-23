// ════════════════════════════════════════════════════════════════════════════
// BxDF 05 · 次表面散射 BSSRDF：Dipole → Normalized Diffusion → 实时 SSS
// Build (shared file first):
//   node <skill>/scripts/build_doc.mjs bxdf_common.js bxdf_05_subsurface.lab.js -o bxdf_05_subsurface.html
// Units: lengths in mm, σ in 1/mm, R(r) in 1/mm². η = η_medium / η_outside (= η_t/η_i for light entering);
// Fresnel from inside to outside uses 1/η. Scene: 1 scene unit = 12 mm. Linear RGB; BX.display only for display.
// Module-local helpers live in SS (nothing in bxdf_common.js is redefined).
// ════════════════════════════════════════════════════════════════════════════
const SS = (() => {
    const PI = Math.PI;
    const MMU = 12;                      // 1 scene unit = 12 mm
    const W = 160, H = 120;              // G-buffer resolution
    const KMAX = 0.5;                    // LUT curvature range κ ∈ [0, 0.5] 1/mm  (r ≥ 2 mm)
    const LUT_W = 64, LUT_H = 48;
    const EL = 3.2, EXPO = 1.0, ROUGH = 0.45;   // light irradiance, display exposure, skin specular roughness
    const CH = ['#e06c75', '#5ad07a', '#6ab0e0'], CHN = ['R', 'G', 'B'];
    const clamp = BX.clamp;

    // Jensen et al. 2001 (Fig. 5) measured reduced scattering σs′ and absorption σa, mm⁻¹, RGB
    const MEDIA = {
        skin1: { label: 'Skin1', ss: [0.74, 0.88, 1.01], sa: [0.032, 0.17, 0.48] },
        skin2: { label: 'Skin2', ss: [1.09, 1.59, 1.79], sa: [0.013, 0.070, 0.145] },
        marble: { label: 'Marble', ss: [2.19, 2.62, 3.00], sa: [0.0021, 0.0041, 0.0071] },
        ketchup: { label: 'Ketchup', ss: [0.18, 0.07, 0.03], sa: [0.061, 0.97, 1.45] },
        apple: { label: 'Apple', ss: [2.29, 2.39, 1.97], sa: [0.0030, 0.0034, 0.046] },
        potato: { label: 'Potato', ss: [0.68, 0.70, 0.55], sa: [0.0024, 0.0090, 0.12] },
        wholemilk: { label: 'Wholemilk', ss: [2.55, 3.21, 3.77], sa: [0.0011, 0.0024, 0.014] },
    };

    // ── boundary: diffuse Fresnel reflectance seen from inside the medium ──
    const fdrPoly = (eta) => -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta;   // Jensen 2001 (Egan & Hilgeman fit)
    /** Fresnel moments of the INTERNAL reflectance: C_i = ∫0^1 F_r(μ; 1/η) μ^i dμ  (F_dr = 2·C1) */
    const moments = (eta, n = 512) => {
        let c1 = 0, c2 = 0;
        for (let k = 0; k < n; k++) { const mu = (k + 0.5) / n, F = BX.fresnelDielectric(mu, 1 / eta); c1 += F * mu; c2 += F * mu * mu; }
        return [c1 / n, c2 / n];
    };
    // Christensen & Burley 2015 shape fits s(A); the profile uses d = (distance) / s
    const sMfp = (A) => 1.85 - A + 7 * Math.abs(A - 0.8) ** 3;     // searchlight, ℓ = 1/σt′ mean free path
    const sDiffuse = (A) => 1.9 - A + 3.5 * (A - 0.8) ** 2;        // diffuse transmission, ℓ
    const sDmfp = (A) => 3.5 + 100 * (A - 0.33) ** 4;              // searchlight, ℓd = 1/σtr diffuse mean free path (used here)
    const totalRd = (alpha, Aeta) => { const q = Math.sqrt(3 * (1 - alpha)); return alpha / 2 * (1 + Math.exp(-4 / 3 * Aeta * q)) * Math.exp(-q); };

    function medium(matId, eta, k) {
        const m = MEDIA[matId] || MEDIA.skin1, Fdr = fdrPoly(eta), Aeta = (1 + Fdr) / (1 - Fdr), [C1, C2] = moments(eta);
        const ch = [0, 1, 2].map((c) => {
            const sa = m.sa[c] / k, ssp = m.ss[c] / k, st = sa + ssp, alpha = ssp / st, str = Math.sqrt(3 * sa * st);
            const ell = 1 / st, elld = 1 / Math.max(str, 1e-9), zr = ell, zv = ell * (1 + 4 * Aeta / 3), Rd = totalRd(alpha, Aeta);
            const A = Rd, s = sDmfp(A), d = elld / s;   // Burley: d = ℓd / s(A) (dmfp fit; robust to the η = 1.3 boundary)
            return { sa, ssp, st, alpha, str, ell, elld, D: ell / 3, zr, zv, Rd, A, s, d };
        });
        return { id: matId, label: m.label, eta, Fdr, Aeta, C1, C2, AetaMom: (1 + 2 * C1) / (1 - 2 * C1), AetaBetter: (1 + 3 * C2) / (1 - 2 * C1), ch };
    }

    // ── radial profiles R(r) (1/mm²) and their radial CDFs P(r) = ∫0^r 2πr′R dr′ / total ──
    const dipole = (r, c) => {
        const dr = Math.hypot(r, c.zr), dv = Math.hypot(r, c.zv);
        return c.alpha / (4 * PI) * (c.zr * (c.str * dr + 1) * Math.exp(-c.str * dr) / (dr * dr * dr) + c.zv * (c.str * dv + 1) * Math.exp(-c.str * dv) / (dv * dv * dv));
    };
    const cdfDipole = (r, c) => {
        const er = Math.exp(-c.str * c.zr), ev = Math.exp(-c.str * c.zv), dr = Math.hypot(r, c.zr), dv = Math.hypot(r, c.zv);
        return (er - c.zr * Math.exp(-c.str * dr) / dr + ev - c.zv * Math.exp(-c.str * dv) / dv) / (er + ev);
    };
    const burley = (r, c) => { r = Math.max(r, 1e-6); return c.A * (Math.exp(-r / c.d) + Math.exp(-r / (3 * c.d))) / (8 * PI * c.d * r); };
    const cdfBurley = (r, c) => 1 - 0.25 * Math.exp(-r / c.d) - 0.75 * Math.exp(-r / (3 * c.d));
    const gauss2 = (v, r) => Math.exp(-r * r / (2 * v)) / (2 * PI * v);
    const sog = (r, fit, k) => { let s = 0; for (let i = 0; i < fit.v.length; i++) s += fit.w[k][i] * gauss2(fit.v[i], r); return s; };
    const cdfSog = (r, fit, k) => { let s = 0, t = 0; for (let i = 0; i < fit.v.length; i++) { s += fit.w[k][i] * (1 - Math.exp(-r * r / (2 * fit.v[i]))); t += fit.w[k][i]; } return t > 0 ? s / t : 1; };
    // d'Eon & Luebke 2007 (GPU Gems 3, ch. 14) three-layer skin: variances in mm², RGB weights (Σ = 1 per channel)
    const DEON = { v: [0.0064, 0.0484, 0.187, 0.567, 1.99, 7.41], w: [[0.233, 0.100, 0.118, 0.113, 0.358, 0.078], [0.455, 0.336, 0.198, 0.007, 0.004, 0], [0.649, 0.344, 0, 0.007, 0, 0]] };

    /** channel k of the selected model → { R(r) absolute, P(r) radial CDF (normalized), total = ∫R dA } */
    function profileOf(model, med, fit, k) {
        const c = med.ch[k];
        if (model === 'dipole') return { R: (r) => dipole(r, c), P: (r) => cdfDipole(r, c), total: c.Rd };
        if (model === 'sog') { const tot = fit.w[k].reduce((a, b) => a + b, 0); return { R: (r) => sog(r, fit, k), P: (r) => cdfSog(r, fit, k), total: tot }; }
        return { R: (r) => burley(r, c), P: (r) => cdfBurley(r, c), total: c.A };
    }

    /** mass of the normalized 2D kernel inside the strip |x| < X:  F(X) = 2/π ∫0^∞ P(X cosh u) / cosh u du */
    function stripMass(P, X) {
        if (X <= 0) return 0;
        const U = 12, n = 160, du = U / n;
        let s = 0;
        for (let i = 0; i < n; i++) { const u = (i + 0.5) * du, c = Math.cosh(u); s += P(X * c) / c; }
        const tail = PI / 2 - 2 * Math.atan(Math.tanh(U / 2));    // ∫_U^∞ sech(u) du, P ≈ 1 there
        return Math.min(1, (2 / PI) * (s * du + tail));
    }
    const halfMass = (P, x) => (x === 0 ? 0 : Math.sign(x) * stripMass(P, Math.abs(x)) / 2);   // ∫_{−∞}^{x} K − 1/2
    const radialQ = (P, q) => { let hi = 0.01; while (P(hi) < q && hi < 1e5) hi *= 1.5; let lo = hi / 1.5; for (let i = 0; i < 30; i++) { const m = 0.5 * (lo + hi); if (P(m) < q) lo = m; else hi = m; } return hi; };
    function extent(P3, q = 0.99) {   // half-width R_K holding q of the 1D marginal, widest channel
        let RK = 0;
        for (const p of P3) {
            let hi = 0.01; while (stripMass(p.P, hi) < q && hi < 1e5) hi *= 1.5;
            let lo = hi / 1.5;
            for (let it = 0; it < 30; it++) { const m = 0.5 * (lo + hi); if (stripMass(p.P, m) < q) lo = m; else hi = m; }
            RK = Math.max(RK, hi);
        }
        return RK;
    }
    /** N taps, shared offsets (mm, quadratic spacing), per-channel weights = kernel mass in each tap's cell */
    function buildKernel(P3, RK, N) {
        const off = [], half = (N - 1) / 2;
        for (let i = 0; i < N; i++) { const t = (i - half) / half; off.push(RK * t * Math.abs(t)); }
        const w = [0, 1, 2].map((k) => {
            const a = new Float32Array(N);
            for (let i = 0; i < N; i++) {
                const lo = i === 0 ? -0.5 : halfMass(P3[k].P, 0.5 * (off[i - 1] + off[i]));
                const hi = i === N - 1 ? 0.5 : halfMass(P3[k].P, 0.5 * (off[i] + off[i + 1]));
                a[i] = hi - lo;
            }
            return a;
        });
        const cell = off.map((o, i) => (i === 0 ? off[1] - off[0] : i === N - 1 ? off[N - 1] - off[N - 2] : 0.5 * (off[i + 1] - off[i - 1])));
        return { N, RK, off, w, cell, c0: half };
    }
    /** true 2D cell masses vs separable K⊗K on a G×G grid spanning ±X2 mm */
    function kernel2D(P3, RK, G = 61) {
        const X2 = RK * 0.6, h = 2 * X2 / G, tru = [], sep = [];
        for (let k = 0; k < 3; k++) {
            const p = P3[k], m = new Float64Array(G), T = new Float32Array(G * G), Sp = new Float32Array(G * G), sub = 3;
            for (let i = 0; i < G; i++) m[i] = halfMass(p.P, -X2 + (i + 1) * h) - halfMass(p.P, -X2 + i * h);
            for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
                const x = -X2 + (i + 0.5) * h, y = -X2 + (j + 0.5) * h;
                let v = 0;
                if (i === (G - 1) / 2 && j === (G - 1) / 2) v = p.P(h / Math.sqrt(PI));       // centre cell ≈ equal-area disk
                else { for (let a = 0; a < sub; a++) for (let b = 0; b < sub; b++) v += p.R(Math.hypot(x + ((a + 0.5) / sub - 0.5) * h, y + ((b + 0.5) / sub - 0.5) * h)); v *= h * h / (sub * sub) / p.total; }
                T[j * G + i] = v; Sp[j * G + i] = m[i] * m[j];
            }
            tru.push(T); sep.push(Sp);
        }
        return { G, X2, h, tru, sep };
    }
    /** SoG fit to the dipole: fixed geometric variances shared by RGB, r-weighted non-negative least squares */
    function fitSoG(med, n = 6) {
        const ellMin = Math.min(...med.ch.map((c) => c.ell)), ldMax = Math.max(...med.ch.map((c) => c.elld));
        const sd0 = 0.25 * ellMin, sd1 = 1.2 * ldMax, v = [];
        for (let i = 0; i < n; i++) { const sd = sd0 * Math.pow(sd1 / sd0, i / (n - 1)); v.push(sd * sd); }
        const K = 320, rMax = 6 * ldMax, w = [];
        for (let c = 0; c < 3; c++) {
            const M = [...Array(n)].map(() => new Float64Array(n)), b = new Float64Array(n);
            for (let k = 0; k < K; k++) {
                const t = (k + 0.5) / K, r = rMax * t * t, wt = r * 2 * rMax * t / K, R = dipole(r, med.ch[c]), g = v.map((vi) => gauss2(vi, r));
                for (let i = 0; i < n; i++) { b[i] += wt * g[i] * R; for (let j = 0; j < n; j++) M[i][j] += wt * g[i] * g[j]; }
            }
            const x = new Float64Array(n);
            for (let it = 0; it < 1500; it++) for (let i = 0; i < n; i++) { let s = b[i]; for (let j = 0; j < n; j++) if (j !== i) s -= M[i][j] * x[j]; x[i] = Math.max(0, s / M[i][i]); }
            w.push(Array.from(x));
        }
        return { v, w };
    }

    // ── importance sampling Burley: analytic inverse CDF (Golubev) ──
    const sampleR = (u, d) => {
        const q = 1 - u, g = 1 + 4 * q * (2 * q + Math.sqrt(1 + 4 * q * q));
        const n = Math.pow(g, -1 / 3), p = g * n * n, c = 1 + p + n;
        return 3 * Math.log(c / (4 * q)) * d;
    };

    // ── pre-integrated LUT over the sphere surface ──
    /** ∫0^{2π} max(0, a + b cos φ) dφ, b ≥ 0 */
    const ringI = (a, b) => { if (a >= b) return 2 * PI * a; if (a <= -b) return 0; const t = Math.acos(-a / b); return 2 * (a * t + Math.sqrt(b * b - a * a)); };
    function bakeLUT(P3) {
        const lut = new Float32Array(LUT_W * LUT_H * 3), NPSI = 96;
        let Rcut = 0; P3.forEach((p) => { Rcut = Math.max(Rcut, radialQ(p.P, 0.999)); });
        const I = new Float64Array(LUT_W * NPSI), w = [0, 1, 2].map(() => new Float64Array(NPSI));
        for (let j = 0; j < LUT_H; j++) {
            const kap = KMAX * (j + 0.5) / LUT_H, rc = 1 / kap;
            const psiMax = Rcut >= 2 * rc ? PI : 2 * Math.asin(Rcut / (2 * rc)), dpsi = psiMax / NPSI, den = [0, 0, 0];
            for (let m = 0; m < NPSI; m++) {
                const psi = (m + 0.5) * dpsi, dist = 2 * rc * Math.sin(psi / 2), sp = Math.sin(psi), cp = Math.cos(psi);
                for (let k = 0; k < 3; k++) { const v = P3[k].R(dist) * sp; w[k][m] = v; den[k] += v; }
                for (let i = 0; i < LUT_W; i++) { const ct = -1 + 2 * (i + 0.5) / LUT_W, st = Math.sqrt(1 - ct * ct); I[i * NPSI + m] = ringI(ct * cp, st * sp); }
            }
            for (let i = 0; i < LUT_W; i++) for (let k = 0; k < 3; k++) {
                let num = 0;
                for (let m = 0; m < NPSI; m++) num += w[k][m] * I[i * NPSI + m];
                lut[(j * LUT_W + i) * 3 + k] = den[k] > 0 ? num / (2 * PI * den[k]) : Math.max(0, -1 + 2 * (i + 0.5) / LUT_W);
            }
        }
        return lut;
    }
    function lutSample(lut, NoL, kap, out) {
        const fx = clamp((NoL + 1) / 2 * LUT_W - 0.5, 0, LUT_W - 1), fy = clamp(kap / KMAX * LUT_H - 0.5, 0, LUT_H - 1);
        const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(x0 + 1, LUT_W - 1), y1 = Math.min(y0 + 1, LUT_H - 1), ax = fx - x0, ay = fy - y0;
        for (let k = 0; k < 3; k++) {
            const a = lut[(y0 * LUT_W + x0) * 3 + k], b = lut[(y0 * LUT_W + x1) * 3 + k], c = lut[(y1 * LUT_W + x0) * 3 + k], d = lut[(y1 * LUT_W + x1) * 3 + k];
            out[k] = (a + (b - a) * ax) * (1 - ay) + (c + (d - c) * ax) * ay;
        }
        return out;
    }

    // ── scene lighting ──
    const lightDir = (t) => {   // uniform-speed sweep of the azimuth between −125° and −40° (light from the left)
        const u = ((t * 0.1) % 2 + 2) % 2, tri = u < 1 ? u : 2 - u;
        const phi = (-125 + 85 * tri) * PI / 180, el = 34 * PI / 180;
        return [Math.sin(phi) * Math.cos(el), Math.sin(el), Math.cos(phi) * Math.cos(el)];
    };
    const ambient = (ny) => 0.2 * (0.6 + 0.4 * ny);
    /** shadow from OTHER objects (all scene objects are convex): skip hits on the shading object itself */
    function visOthers(sc, p, n, L, self) {
        let o = [p[0] + n[0] * 1e-3, p[1] + n[1] * 1e-3, p[2] + n[2] * 1e-3];
        for (let it = 0; it < 4; it++) {
            const h = PL.scene.intersect(sc, o, L, Infinity);
            if (!h) return 1;
            if (h.id !== self) return 0;
            o = [h.p[0] + L[0] * 1e-3, h.p[1] + L[1] * 1e-3, h.p[2] + L[2] * 1e-3];
        }
        return 1;
    }
    /** GGX specular × N·L (world-space dot products; mirrors D_GGX / V_SmithGGXCorrelated in the shared HLSL) */
    function ggxSpec(n, L, V, a, F0) {
        const NoL = BX.dot(n, L), NoV = BX.dot(n, V);
        if (NoL <= 0 || NoV <= 0) return 0;
        const h = BX.norm([L[0] + V[0], L[1] + V[1], L[2] + V[2]]), NoH = Math.max(0, BX.dot(n, h)), VoH = Math.max(0, BX.dot(V, h));
        const a2 = a * a, d = (NoH * a2 - NoH) * NoH + 1, D = a2 / (PI * d * d);
        const vis = 0.5 / (NoL * Math.sqrt(NoV * NoV * (1 - a2) + a2) + NoV * Math.sqrt(NoL * NoL * (1 - a2) + a2));
        return D * vis * BX.schlick(F0, VoH) * NoL;
    }
    /** one separable pass (Jimenez 2015 structure). src stride 1 (scalar E) or 3 (RGB); dst RGB */
    function blur(S, src, stride, dst, horiz) {
        const gb = S.gb, K = S.kern, N = K.N, c0 = K.c0, off = K.off, wR = K.w[0], wG = K.w[1], wB = K.w[2];
        const tanHalf = PL.scene.basis(S.sc, W / H).tanHalf, follow = S.follow, g1 = stride === 3 ? 1 : 0, g2 = stride === 3 ? 2 : 0;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = y * W + x, id = gb.id[i], si = i * stride, mR = src[si], mG = src[si + g1], mB = src[si + g2];
            if (id < 1) { dst[i * 3] = mR; dst[i * 3 + 1] = mG; dst[i * 3 + 2] = mB; continue; }   // stencil: not SSS
            const z = gb.depth[i], ppm = 0.5 * H / (z * tanHalf * MMU);                              // pixels per mm
            let aR = wR[c0] * mR, aG = wG[c0] * mG, aB = wB[c0] * mB;
            for (let t = 0; t < N; t++) {
                if (t === c0) continue;
                const o = off[t], f = (horiz ? x : y) + o * ppm, f0 = Math.floor(f), fr = f - f0, lim = horiz ? W : H;
                let sR = 0, sG = 0, sB = 0;
                for (let q = 0; q < 2; q++) {
                    let ci = f0 + q; ci = ci < 0 ? 0 : ci >= lim ? lim - 1 : ci;
                    const j = horiz ? y * W + ci : ci * W + x, sj = j * stride, wq = q ? fr : 1 - fr;
                    let cR = src[sj], cG = src[sj + g1], cB = src[sj + g2];
                    if (follow) {
                        const s = gb.id[j] !== id ? 1 : clamp(Math.abs(gb.depth[j] - z) * MMU / Math.abs(o) - 1, 0, 1);
                        cR += (mR - cR) * s; cG += (mG - cG) * s; cB += (mB - cB) * s;
                    }
                    sR += wq * cR; sG += wq * cG; sB += wq * cB;
                }
                aR += wR[t] * sR; aG += wG[t] * sG; aB += wB[t] * sB;
            }
            dst[i * 3] = aR; dst[i * 3 + 1] = aG; dst[i * 3 + 2] = aB;
        }
    }

    // ── drawing helpers ──
    const drawBuf = (canvas, buf, scale = 1, ids = null) => PL.draw.pixels(canvas, W, H, (x, y, out) => {
        const p = y * W + x, i = p * 3, k = ids && ids[p] < 1 ? scale * 0.15 : scale, d = BX.display([buf[i] * k, buf[i + 1] * k, buf[i + 2] * k], EXPO);
        out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
    });
    const fmt = (v, p = 3) => (!isFinite(v) ? '∞' : v !== 0 && (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3) ? v.toExponential(2) : v.toPrecision(p));
    function table(canvas, S) {
        const { ctx, w, h } = PL.draw.ctx(canvas), m = S.med;
        const rows = [['σa', 'sa', 'mm⁻¹'], ['σs′', 'ssp', 'mm⁻¹'], ['σt′ = σa+σs′', 'st', 'mm⁻¹'], ['α′ = σs′/σt′', 'alpha', ''], ['σtr = √(3σaσt′)', 'str', 'mm⁻¹'],
            ['ℓ = 1/σt′', 'ell', 'mm'], ['ℓd = 1/σtr', 'elld', 'mm'], ['A = ∫Rd dA', 'Rd', ''], ['s(A) · ℓd 拟合', 's', ''], ['d = ℓd/s', 'd', 'mm']];
        const lab = 168, cw = (w - lab - 6) / 3, rh = (h - 22) / (rows.length + 1);
        ctx.font = '11px Consolas, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        ctx.fillStyle = '#8a8a8a'; ctx.fillText(`${m.label} · η=${m.eta.toFixed(2)} · Aη=${m.Aeta.toFixed(3)}`, 6, 9);
        CHN.forEach((c, k) => { ctx.fillStyle = CH[k]; ctx.textAlign = 'right'; ctx.fillText(c, lab + cw * (k + 1), 22 + rh * 0.5); });
        rows.forEach((r, i) => {
            const y = 22 + rh * (i + 1.5);
            if (i % 2 === 0) { ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(0, y - rh / 2, w, rh); }
            ctx.textAlign = 'left'; ctx.fillStyle = '#aaa'; ctx.fillText(r[0] + (r[2] ? ` (${r[2]})` : ''), 6, y);
            m.ch.forEach((c, k) => { ctx.textAlign = 'right'; ctx.fillStyle = CH[k]; ctx.fillText(fmt(c[r[1]]), lab + cw * (k + 1), y); });
        });
        ctx.textAlign = 'left';
    }
    function bars(canvas, groups, o = {}) {
        const { ctx, w, h } = PL.draw.ctx(canvas), padL = 38, padR = 8, padT = 18, padB = 30;
        const vmax = o.yMax || Math.max(1e-9, ...groups.flatMap((g) => g.v));
        const gw = (w - padL - padR) / groups.length, bw = gw * 0.8 / 3;
        ctx.font = '10px Consolas, monospace'; ctx.strokeStyle = '#1c1c1c'; ctx.fillStyle = '#5a5a5a'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (const f of [0, 0.5, 1]) { const y = h - padB - f * (h - padT - padB); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); ctx.fillText((vmax * f).toFixed(2), padL - 4, y); }
        groups.forEach((g, i) => {
            g.v.forEach((v, c) => { const bh = Math.max(0, v) / vmax * (h - padT - padB); ctx.fillStyle = CH[c]; ctx.fillRect(padL + i * gw + gw * 0.1 + c * bw, h - padB - bh, bw - 1, bh); });
            ctx.fillStyle = '#777'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            String(g.label).split('\n').forEach((s, li) => ctx.fillText(s, padL + (i + 0.5) * gw, h - padB + 3 + li * 11));
        });
        if (o.title) { ctx.fillStyle = '#888'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.title, padL + 2, 2); }
    }
    function dipoleDiagram(canvas, S) {
        const { ctx, w, h } = PL.draw.ctx(canvas), m = S.med, c = m.ch[0], zb = 2 * m.Aeta * c.D;
        const top = 16, bot = h - 14, k = (bot - top) / (c.zr + c.zv), y0 = top + c.zv * k, cx = w * 0.3, r = c.zv * 0.9, xo = cx + r * k;
        ctx.fillStyle = 'rgba(224,108,117,0.07)'; ctx.fillRect(0, y0, w, h - y0);
        ctx.strokeStyle = '#666'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(8, y0); ctx.lineTo(w - 8, y0); ctx.stroke();
        ctx.setLineDash([4, 4]); ctx.strokeStyle = '#8a7a3a'; ctx.beginPath(); ctx.moveTo(8, y0 - zb * k); ctx.lineTo(w - 8, y0 - zb * k); ctx.stroke();
        ctx.strokeStyle = '#555'; ctx.beginPath(); ctx.moveTo(cx, y0 + c.zr * k); ctx.lineTo(xo, y0); ctx.moveTo(cx, y0 - c.zv * k); ctx.lineTo(xo, y0); ctx.stroke(); ctx.setLineDash([]);
        ctx.strokeStyle = '#e8c25a'; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(cx - 26, top - 4); ctx.lineTo(cx, y0); ctx.stroke();
        ctx.fillStyle = '#e06c75'; ctx.beginPath(); ctx.arc(cx, y0 + c.zr * k, 4.5, 0, 2 * PI); ctx.fill();
        ctx.strokeStyle = '#6ab0e0'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, y0 - c.zv * k, 4.5, 0, 2 * PI); ctx.stroke();
        ctx.fillStyle = '#ddd'; ctx.beginPath(); ctx.arc(xo, y0, 3, 0, 2 * PI); ctx.fill();
        ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        ctx.fillStyle = '#e06c75'; ctx.fillText(`+ real source  z = −zr = −${c.zr.toFixed(2)} mm`, cx + 9, y0 + c.zr * k);
        ctx.fillStyle = '#6ab0e0'; ctx.fillText(`− virtual (mirror)  z = +zv = ${c.zv.toFixed(2)} mm`, cx + 9, y0 - c.zv * k);
        ctx.fillStyle = '#b8a050'; ctx.fillText(`extrapolated boundary z = 2AηD = ${zb.toFixed(2)} mm (φ = 0)`, 10, y0 - zb * k - 8);
        ctx.fillStyle = '#999'; ctx.fillText('surface z = 0', w - 92, y0 + 9);
        ctx.fillStyle = '#ddd'; ctx.fillText(`xo  (r = ${r.toFixed(2)} mm)`, xo + 5, y0 - 8); ctx.fillStyle = '#e8c25a'; ctx.fillText('xi', cx - 30, top + 2);
                ctx.fillText('dr', (cx + xo) / 2 + 4, y0 + c.zr * k / 2 + 4); ctx.fillText('dv', (cx + xo) / 2 + 4, y0 - c.zv * k / 2);
    }
    function scatter(canvas, S) {
        const { ctx, w, h } = PL.draw.ctx(canvas, { bg: '#08080a' }), R = S.samp.ext, cx = w / 2, cy = h / 2, k = Math.min(w, h) / 2 / R;
        ctx.globalAlpha = 0.5;
        [2, 1, 0].forEach((c) => { ctx.fillStyle = CH[c]; const p = S.samp.pts[c]; for (let i = 0; i < p.length; i += 2) ctx.fillRect(cx + p[i] * k - 0.7, cy - p[i + 1] * k - 0.7, 1.4, 1.4); });
        ctx.globalAlpha = 1; ctx.lineWidth = 1;
        [0, 1, 2].forEach((c) => { ctx.strokeStyle = CH[c]; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(cx, cy, S.samp.r50[c] * k, 0, 2 * PI); ctx.stroke(); });
        ctx.setLineDash([]); ctx.fillStyle = '#888'; ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        ctx.fillText(`±${R.toFixed(1)} mm · 虚线 = 中位半径`, 5, 4);
    }
    /** profile plot helper: per-channel series for one or more models */
    function profPlot(canvas, S, models, o = {}) {
        const rP = o.r || S.rPlot, series = [];
        models.forEach((md) => [0, 1, 2].forEach((k) => series.push({ fn: (r) => md.f(r, k), color: CH[k], dash: md.dash, width: md.width || 1.5, label: md.label ? `${CHN[k]} ${md.label(k)}` : undefined, samples: 260 })));
        let ymax = 0;
        series.forEach((s) => { const v = s.fn(rP / 200); if (isFinite(v)) ymax = Math.max(ymax, v); });
        PL.draw.plot(canvas, { x: [0, rP], y: o.y || [ymax * (o.dec || 1e-6), ymax * 2], logY: true, xLabel: o.xLabel || 'r (mm)', series, marks: o.marks });
    }

    return {
        PI, MMU, W, H, KMAX, LUT_W, LUT_H, EL, EXPO, ROUGH, CH, CHN, MEDIA, fdrPoly, moments, sMfp, sDiffuse, sDmfp, totalRd, medium,
        dipole, burley, gauss2, sog, DEON, profileOf, stripMass, halfMass, radialQ, extent, buildKernel, kernel2D, fitSoG, sampleR,
        ringI, bakeLUT, lutSample, lightDir, ambient, visOthers, ggxSpec, blur, drawBuf, fmt, table, bars, dipoleDiagram, scatter, profPlot,
    };
})();

// ── HLSL shared by the Common panel and several steps ──
const SSS_HLSL_CB = `
    // 本模块单位：长度 mm，σ 为 1/mm；场景 1 unit = MMPerUnit mm
    cbuffer SSSParams : register(b0) {
        float3 SurfaceAlbedo;     // A = ∫R dA（每通道，multiple-scattering albedo）
        float3 DiffuseMFP;        // ℓd = 1/σtr（mm）—— 本页 Burley 的距离参数
        float  Eta;               // 介质相对 IOR（光射入时 η_t/η_i）
        float  MMPerUnit;         // 本页 12
        float  KappaMax;          // LUT 曲率上限（1/mm），本页 0.5
        float  FollowSurface;     // 1 = 深度感知
    };
    cbuffer LightCB : register(b1) { float3 LightDir; float LightIrradiance; float AmbientIrr; };
    cbuffer ViewCB  : register(b2) {                 // 引擎注入
        float2 ViewSize; float2 InvViewSize; float TanHalfFovY; float3 CameraPos;
        float3 CameraRight; float padV0; float3 CameraUp; float padV1; float3 CameraForward; float padV2;   // 世界空间相机基（右 / 上 / 前）
    };`;
const SSS_HLSL_PROFILE = `
    // Burley normalized diffusion：∫ R dA = A。d = ℓd / s(A)
    float3 BurleyShapeS(float3 A)                 // Christensen & Burley 2015 · searchlight · ℓd = 1/σtr
    {
        float3 x = A - 0.33;
        return 3.5 + 100.0 * x * x * x * x;       // ℓ = 1/σt′ 版本：1.85 − A + 7·|A − 0.8|³
    }
    float3 BurleyProfile(float r, float3 A, float3 d)   // r 与 d 都是 mm → 1/mm²
    {
        r = max(r, 1e-6);                              // 1/r 奇异但可积：按面积积分时没问题
        return A * (exp(-r / d) + exp(-r / (3.0 * d))) / (8.0 * PI * d * r);
    }
    float3 BurleyCDF(float r, float3 d)              // 径向 CDF（含 2πr）
    {
        return 1.0 - 0.25 * exp(-r / d) - 0.75 * exp(-r / (3.0 * d));
    }`;

PipelineLab.mount({
    paramSteps: { mat: ['bssrdf', 'medium', 'dipole', 'sog', 'burley', 'sample', 'kernel', 'ssss', 'lut', 'prod'], model: ['bssrdf', 'kernel', 'ssss', 'lut'], ellK: ['bssrdf', 'medium', 'dipole', 'sog', 'burley', 'sample', 'kernel', 'ssss', 'lut', 'prod'], eta: ['bssrdf', 'medium', 'dipole', 'sog', 'burley', 'sample', 'kernel', 'ssss', 'lut', 'prod'], taps: ['kernel', 'ssss'], follow: ['ssss'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: 'BxDF 05 · 次表面散射 BSSRDF：Dipole → Normalized Diffusion',
    lang: 'zh-CN',
    codeLang: 'hlsl',
    seed: 5,
    regen: false,

    params: [
        { id: 'mat', label: '介质', value: 'skin1', affects: 'resident', title: 'Jensen et al. 2001 实测 σs′ / σa（mm⁻¹）。换介质 = 换资产：重烘全部 RESIDENT（剖面、LUT、核尺寸）',
            options: Object.keys(SS.MEDIA).map((k) => ({ value: k, label: SS.MEDIA[k].label })),
            desc: '实测介质（Jensen et al. 2001 的 σ<sub>s</sub>′ / σ<sub>a</sub> 表，mm⁻¹）。换介质 = 换资产：剖面、核尺寸、LUT 全部重烘，除 Step 08 的 G-buffer 外每一步都跟着变。Skin1 / Skin2 看红光渗得最远；Marble、Wholemilk 几乎不吸收、整体发亮；Ketchup 的绿 / 蓝几乎不散射，只有红光在“发光”。' },
        { id: 'model', label: '剖面', value: 'burley', affects: 'resident', title: '屏幕空间核与 LUT 使用的径向剖面 R(r)',
            options: [{ value: 'burley', label: 'Burley' }, { value: 'dipole', label: 'Dipole' }, { value: 'sog', label: 'SoG fit' }],
            desc: '屏幕空间核、阴影边实验和预积分 LUT 用哪条径向剖面 R(r)：Burley normalized diffusion（默认）、经典偶极子、或对偶极子的 Sum-of-Gaussians 拟合。三者总能量基本一致（Burley 的 A 取偶极子的 R<sub>d</sub>，SoG 拟合偶极子），差在中心尖峰与尾巴——切换后对比 Step 01 的阴影边、Step 07 的核和 Step 09 / 10 的渲染。' },
        { id: 'ellK', label: 'ℓ ×', min: 0.25, max: 4, step: 0.05, value: 1, digits: 2, affects: 'resident', title: '所有自由程乘 k（σa、σs′ 同除以 k）：α′、A 不变，只把剖面在空间上拉伸',
            desc: '所有自由程乘 k（σ<sub>a</sub>、σ<sub>s</sub>′ 同除以 k）：α′ 与 A 不变，剖面只在空间上拉伸 k 倍——等价于把物体缩小到 1/k。拖到 3–4 看阴影边的红色渗透成比例变宽、物体像蜡；拖到 0.25 接近 Lambert。' },
        { id: 'eta', label: 'η', min: 1.0, max: 1.8, step: 0.01, value: 1.3, digits: 2, affects: 'resident', title: '介质相对折射率（Jensen 的表格用 1.3）',
            desc: '介质相对折射率（Jensen 的表格用 1.3）。它通过边界项 A<sub>η</sub> = (1 + F<sub>dr</sub>)/(1 − F<sub>dr</sub>) 进入偶极子：η 越大，体内的 Fresnel / 全内反射越多，虚源离表面越远（Step 03 的示意图），总反照率 A 越低；Step 11 的随机游走里它就是边界上的 Fresnel。η = 1 时边界不反射。' },
        { id: 'taps', label: 'taps', options: [9, 17, 25, 33], value: 17, affects: 'param', title: '可分离核每个方向的采样数',
            desc: '可分离核每个方向的采样数（Step 07 的 1D 核与 Step 09 的 Blur H / V）。tap 越少，每个 tap 代表的区间越宽：拖到 9 再拖到 33，对比 Step 07 的核权重与 Step 09 阴影边的平滑程度。只重建核，不重烘剖面。' },
        { id: 'follow', label: 'follow surface', value: true, affects: 'frame', title: '深度 / 模板感知：跨到别的表面的样本换回中心色',
            desc: '深度 / 模板感知（Jimenez 2015）：模糊样本落到别的物体上、或深度差相对 tap 距离太大时，换回中心像素的颜色。关掉它看 Step 09：地面的光被“吸”进物体、物体轮廓出现暗晕。只影响每帧的模糊 pass。' },
    ],

    stages: [
        {
            id: 'medium', lifetime: 'resident', run(S) {
                S.med = SS.medium(S.mat, S.eta, S.ellK);
                S.fit = SS.fitSoG(S.med);
                S.P3 = [0, 1, 2].map((k) => SS.profileOf(S.model, S.med, S.fit, k));
                S.RK = SS.extent(S.P3);
                S.rPlot = Math.min(60, 3.2 * Math.max(...S.med.ch.map((c) => c.elld)));
            },
        },
        {
            id: 'bleed', lifetime: 'resident', run(S) {   // 1D shadow edge: E = 1 for x < 0, 0 for x > 0
                const X = 0.45 * S.RK, n = 161, xs = new Float64Array(n), M = [0, 1, 2].map(() => new Float64Array(n));
                for (let i = 0; i < n; i++) { const x = -X + 2 * X * i / (n - 1); xs[i] = x; for (let k = 0; k < 3; k++) M[k][i] = S.P3[k].total * (0.5 - SS.halfMass(S.P3[k].P, x)); }
                S.bleed = { X, xs, M };
            },
        },
        { id: 'kernel2d', lifetime: 'resident', run(S) { S.k2 = SS.kernel2D(S.P3, S.RK, 61); } },
        {
            id: 'samples', lifetime: 'resident', run(S) {
                const rnd = S.rand, n = 1400, pts = [], r50 = [];
                for (let c = 0; c < 3; c++) {
                    const d = S.med.ch[c].d, p = new Float32Array(2 * n);
                    for (let i = 0; i < n; i++) { const r = SS.sampleR(rnd(), d), ph = 2 * Math.PI * rnd(); p[2 * i] = r * Math.cos(ph); p[2 * i + 1] = r * Math.sin(ph); }
                    pts.push(p); r50.push(SS.sampleR(0.5, d));
                }
                const d0 = S.med.ch[0].d, rH = SS.sampleR(0.985, d0), NB = 40, NS = 8000, hist = new Float64Array(NB);
                for (let i = 0; i < NS; i++) { const b = Math.floor(SS.sampleR(rnd(), d0) / rH * NB); if (b < NB) hist[b]++; }
                S.samp = { pts, r50, ext: SS.sampleR(0.9, Math.max(...S.med.ch.map((c) => c.d))), hist: Array.from(hist, (v) => v / (NS * rH / NB)), rH, NB };
            },
        },
        { id: 'lut', lifetime: 'resident', run(S) { S.lut = SS.bakeLUT(S.P3); } },
        {
            id: 'scene', lifetime: 'resident', run(S) {
                const A = S.med.ch.map((c) => c.A), W = SS.W, H = SS.H;
                S.sc = PL.scene.create({
                    camera: { pos: [0.3, 1.45, 3.7], target: [0.28, 0.4, 0], fov: 38 },
                    objects: [
                        { type: 'plane', n: [0, 1, 0], d: 0, albedo: [0.30, 0.30, 0.30], albedo2: [0.17, 0.17, 0.17], checker: 0.5 },
                        { type: 'sphere', c: [-0.55, 0.6, 0], r: 0.6, albedo: A },
                        { type: 'sphere', c: [0.38, 0.3, 0.3], r: 0.3, albedo: A },
                        { type: 'box', c: [1.25, 0.4, -0.6], size: [0.8, 0.8, 0.8], rotY: 0.55, albedo: A },
                    ],
                    sky: [0.02, 0.025, 0.03],
                });
                const gb = S.gb = PL.scene.gbuffer(S.sc, W, H, S.gb);
                const dv = PL.buf(S, 'depthVis', W * H), kap = PL.buf(S, 'kappa', W * H);
                for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
                    const i = y * W + x, id = gb.id[i];
                    dv[i] = id < 0 ? NaN : gb.depth[i];
                    kap[i] = 0;
                    if (id < 1) continue;
                    const nb = (dx, dy) => { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) return -1; const j = yy * W + xx; return gb.id[j] === id ? j : -1; };
                    let jx = nb(1, 0); if (jx < 0) jx = nb(-1, 0);
                    let jy = nb(0, 1); if (jy < 0) jy = nb(0, -1);
                    const fN = [0, 0, 0], fP = [0, 0, 0];   // fwidth = |ddx| + |ddy|
                    for (const j of [jx, jy]) if (j >= 0) for (let c = 0; c < 3; c++) { fN[c] += Math.abs(gb.nrm[3 * j + c] - gb.nrm[3 * i + c]); fP[c] += Math.abs(gb.pos[3 * j + c] - gb.pos[3 * i + c]); }
                    const lp = Math.hypot(fP[0], fP[1], fP[2]);
                    kap[i] = lp > 1e-9 ? Math.hypot(fN[0], fN[1], fN[2]) / lp / SS.MMU : 0;
                }
            },
        },
        {
            id: 'walk', lifetime: 'resident', lazy: true, run(S) {   // MC random walk in a semi-infinite medium (ground truth)
                const rng = PL.rng((S.seed ^ 0x51ed27) >>> 0), eta = S.med.eta, NB = 44, rP = S.rPlot, out = [];
                for (let k = 0; k < 3; k++) {
                    const c = S.med.ch[k], st = c.st, alpha = c.alpha, hist = new Float64Array(NB);
                    let esc = 0, nw = 0, steps = 0;
                    while (steps < 320000 && nw < 40000) {
                        nw++;
                        let x = 0, y = 0, z = 0, dx = 0, dy = 0, dz = -1;
                        for (let it = 0; it < 3000; it++) {
                            const t = -Math.log(1 - rng()) / st;
                            x += dx * t; y += dy * t; z += dz * t; steps++;
                            if (z > 0) {
                                const back = z / dz; x -= dx * back; y -= dy * back; z = 0;
                                if (rng() < BX.fresnelDielectric(dz, 1 / eta)) { dz = -dz; continue; }   // internal reflection / TIR
                                esc++; const b = Math.floor(Math.hypot(x, y) / rP * NB); if (b < NB) hist[b]++;
                                break;
                            }
                            if (rng() > alpha) break;   // absorbed
                            const u1 = rng(), u2 = rng(); dz = 1 - 2 * u1; const s = Math.sqrt(Math.max(0, 1 - dz * dz)), ph = 2 * Math.PI * u2; dx = s * Math.cos(ph); dy = s * Math.sin(ph);
                        }
                    }
                    const R = Array.from(hist, (v, b) => v / (nw * Math.PI * (((b + 1) * rP / NB) ** 2 - (b * rP / NB) ** 2)));
                    out.push({ R, x: R.map((_, b) => (b + 0.5) * rP / NB), A: esc / nw, nw });
                }
                S.walk = out;
            },
        },
        {
            id: 'walk3d', lifetime: 'resident', lazy: true, run(S) {   // a few random walks per channel, same rules as 'walk' (searchlight at the origin, medium z < 0) → 3D polylines + exit points
                const rng = PL.rng((S.seed ^ 0x3d3d77) >>> 0), eta = S.med.eta, NP = 4, NX = 160, ch = [];
                for (let k = 0; k < 3; k++) {
                    const c = S.med.ch[k], paths = [], exits = [];
                    for (let n = 0; n < 20000 && (paths.length < NP || exits.length < NX); n++) {
                        let x = 0, y = 0, z = 0, dx = 0, dy = 0, dz = -1, out = false;
                        const pts = [[0, 0, 0]];
                        for (let it = 0; it < 3000; it++) {
                            const t = -Math.log(1 - rng()) / c.st;
                            x += dx * t; y += dy * t; z += dz * t;
                            if (z > 0) {
                                const back = z / dz; x -= dx * back; y -= dy * back; z = 0;
                                pts.push([x, y, 0]);
                                if (rng() < BX.fresnelDielectric(dz, 1 / eta)) { dz = -dz; continue; }   // internal reflection / TIR
                                out = true; break;
                            }
                            pts.push([x, y, z]);
                            if (rng() > c.alpha) break;   // absorbed
                            const u1 = rng(), u2 = rng(); dz = 1 - 2 * u1; const s = Math.sqrt(Math.max(0, 1 - dz * dz)), ph = 2 * Math.PI * u2; dx = s * Math.cos(ph); dy = s * Math.sin(ph);
                        }
                        if (!out) continue;
                        if (exits.length < NX) exits.push([x, y]);
                        if (paths.length < NP) paths.push(pts);
                    }
                    const rs = exits.map((e) => Math.hypot(e[0], e[1])).sort((a, b) => a - b);
                    ch.push({ paths, exits, r50: rs[rs.length >> 1] || 0, r90: rs[Math.floor(rs.length * 0.9)] || 0 });
                }
                S.walk3d = { ch, X: Math.max(...ch.map((q) => q.r90), 1e-3) };
            },
        },
        { id: 'kernel', lifetime: 'param', run(S) { S.kern = SS.buildKernel(S.P3, S.RK, S.taps); } },
        {
            id: 'lighting', lifetime: 'frame', run(S) {
                const gb = S.gb, n = SS.W * SS.H, L = S.L = SS.lightDir(S.t), cam = S.sc.camera.pos, F0 = BX.f0FromIor(Math.max(S.eta, 1.0001));
                const E = PL.buf(S, 'E', n), vis = PL.buf(S, 'vis', n), spec = PL.buf(S, 'spec', n), amb = PL.buf(S, 'amb', n), NoLb = PL.buf(S, 'NoL', n);
                for (let i = 0; i < n; i++) {
                    const id = gb.id[i];
                    if (id < 0) { E[i] = vis[i] = spec[i] = amb[i] = NoLb[i] = 0; continue; }
                    const nn = [gb.nrm[3 * i], gb.nrm[3 * i + 1], gb.nrm[3 * i + 2]], p = [gb.pos[3 * i], gb.pos[3 * i + 1], gb.pos[3 * i + 2]];
                    const NoL = BX.dot(nn, L), v = SS.visOthers(S.sc, p, nn, L, id);
                    NoLb[i] = NoL; vis[i] = v; amb[i] = SS.ambient(nn[1]);
                    E[i] = SS.EL * Math.max(0, NoL) * v + amb[i];
                    spec[i] = id >= 1 && v > 0 ? SS.EL * SS.ggxSpec(nn, L, BX.norm([cam[0] - p[0], cam[1] - p[1], cam[2] - p[2]]), SS.ROUGH * SS.ROUGH, F0) : 0;
                }
            },
        },
        { id: 'blurH', lifetime: 'frame', run(S) { S.BH = PL.buf(S, 'BH', SS.W * SS.H * 3); SS.blur(S, S.E, 1, S.BH, true); } },
        { id: 'blurV', lifetime: 'frame', run(S) { S.BV = PL.buf(S, 'BV', SS.W * SS.H * 3); SS.blur(S, S.BH, 3, S.BV, false); } },
        {
            id: 'combine', lifetime: 'frame', run(S) {
                const gb = S.gb, n = SS.W * SS.H, A = S.med.ch.map((c) => c.A), ip = 1 / Math.PI, D = [0, 0, 0];
                const Lm = PL.buf(S, 'imgLambert', n * 3), Ls = PL.buf(S, 'imgSSS', n * 3), Lp = PL.buf(S, 'imgPre', n * 3), dif = PL.buf(S, 'imgDiff', n);
                for (let i = 0; i < n; i++) {
                    const id = gb.id[i];
                    if (id < 0) { for (let c = 0; c < 3; c++) Lm[3 * i + c] = Ls[3 * i + c] = Lp[3 * i + c] = S.sc.sky[c]; dif[i] = 0; continue; }
                    const sss = id >= 1, sp = S.spec[i];
                    if (sss) SS.lutSample(S.lut, S.NoL[i], S.kappa[i], D);
                    for (let c = 0; c < 3; c++) {
                        const a = sss ? A[c] : gb.alb[3 * i + c], lam = a * ip * S.E[i] + sp;
                        Lm[3 * i + c] = lam;
                        Ls[3 * i + c] = sss ? a * ip * S.BV[3 * i + c] + sp : lam;
                        Lp[3 * i + c] = sss ? a * ip * (SS.EL * D[c] * S.vis[i] + S.amb[i]) + sp : lam;
                    }
                    dif[i] = BX.lum([Ls[3 * i], Ls[3 * i + 1], Ls[3 * i + 2]]) - BX.lum([Lm[3 * i], Lm[3 * i + 1], Lm[3 * i + 2]]);
                }
            },
        },
    ],

    library: {
        title: '附录 · 辅助函数（HLSL）',
        intro: '步骤代码里调用、但不属于该步主题的辅助函数：Step 08 由线性深度重建世界坐标，Step 11 随机游走用到的随机数、通道选择、Fresnel 边界与 Henyey-Greenstein 采样。闭合网格求交 <code>TraceClosedMesh</code> 属于引擎 / 平台的光线追踪接口（DXR 里是 <code>RayQuery</code>），只作为外部函数声明。',
        blocks: [
            {
                title: 'G-buffer · 由线性深度重建世界坐标（Step 08）', code: `
                // 线性 view z → 世界坐标：像素中心 → NDC → z_view = 1 平面上的点，乘深度后用相机基转到世界空间
                float3 ReconstructWorldPos(uint2 px, float viewZ)
                {
                    float2 ndc    = (float2(px) + 0.5) * InvViewSize * float2(2.0, -2.0) + float2(-1.0, 1.0);   // y 向上
                    float  aspect = ViewSize.x * InvViewSize.y;
                    float2 xy     = ndc * float2(TanHalfFovY * aspect, TanHalfFovY);
                    return CameraPos + viewZ * (xy.x * CameraRight + xy.y * CameraUp + CameraForward);
                }`,
            },
            {
                title: '随机游走 · 随机数、通道选择、边界与相函数（Step 11）', code: `
                #define MAX_WALK 256                                       // 最多散射次数，超过按吸收处理
                cbuffer RandomWalkCB : register(b3) { float Anisotropy; };   // HG 的 g：皮肤 ≈ 0.8；Step 11 的平板 ground truth 用约化 σs′、g = 0

                // 每条路径一个 PCG 随机数状态
                struct RNG { uint state; };

                // PCG-RXS-M-XS 32 位：推进 LCG 状态并返回 [0, 1) 的均匀数（取高 24 位，保证严格 < 1）
                float Rand(inout RNG rng)
                {
                    rng.state = rng.state * 747796405u + 2891336453u;
                    uint w = ((rng.state >> ((rng.state >> 28u) + 4u)) ^ rng.state) * 277803737u;
                    w = (w >> 22u) ^ w;
                    return float(w >> 8) * (1.0 / 16777216.0);
                }
                // 两个独立的均匀数（相函数采样用）
                float2 Rand2(inout RNG rng) { float a = Rand(rng); return float2(a, Rand(rng)); }

                // TraceClosedMesh 的命中记录
                struct HitInfo {
                    float  t;      // 沿射线到边界的距离
                    float3 P;      // 交点
                    float3 N;      // 几何法线，指向体外，单位长
                };

                // 选通道：p_c = β_c / Σβ（吞吐量大的通道更常被选中）；ChannelAvg 必须用同一组概率
                int PickChannel(float3 beta, inout RNG rng)
                {
                    float3 pc = beta / max(beta.x + beta.y + beta.z, 1e-30);
                    float  u  = Rand(rng);
                    return u < pc.x ? 0 : (u < pc.x + pc.y ? 1 : 2);
                }

                // 按 PickChannel 的概率加权的通道平均 Σ p_c·x_c：距离采样的单样本 MIS 混合 pdf（或混合透过概率）
                float ChannelAvg(float3 x, float3 beta)
                {
                    float3 pc = beta / max(beta.x + beta.y + beta.z, 1e-30);
                    return dot(pc, x);
                }

                // 非偏振介电 Fresnel（与 BX.fresnelDielectric 相同）：cosI = 入射侧余弦，eta = η_t/η_i；cosI < 0 表示从另一侧入射；全内反射 → 1
                float FresnelDielectric(float cosI, float eta)
                {
                    if (cosI < 0.0) { eta = 1.0 / eta; cosI = -cosI; }
                    cosI = min(cosI, 1.0);
                    float sin2T = (1.0 - cosI * cosI) / (eta * eta);
                    if (sin2T >= 1.0) return 1.0;
                    float cosT  = sqrt(1.0 - sin2T);
                    float rPar  = (eta * cosI - cosT) / (eta * cosI + cosT);
                    float rPerp = (cosI - eta * cosT) / (cosI + eta * cosT);
                    return 0.5 * (rPar * rPar + rPerp * rPerp);
                }

                // 体内一侧撞到边界：以概率 F 反射、1 − F 折射（权重恰为 1，与页面 JS 随机游走相同）
                // 返回 true = 折射出去（dir 换成体外方向）；false = 反射回体内（dir 镜像）
                bool ExitBoundary(HitInfo h, inout float3 dir, inout RNG rng)
                {
                    float cosI = dot(dir, h.N);                              // dir 正往外走 → cosI > 0
                    if (Rand(rng) < FresnelDielectric(cosI, 1.0 / Eta)) {   // 体内 → 体外：η_t/η_i = 1/η；全内反射时 F = 1
                        dir = reflect(dir, h.N);
                        return false;
                    }
                    dir = refract(dir, -h.N, Eta);                           // HLSL refract：法线朝入射一侧，eta = η_i/η_t = η
                    return true;
                }

                // Duff et al. 2017 无分支正交基：(t, b, n) 右手系，n 为单位向量
                void BuildOrthonormalBasis(float3 n, out float3 t, out float3 b)
                {
                    float s = n.z >= 0.0 ? 1.0 : -1.0;
                    float a = -1.0 / (s + n.z);
                    float k = n.x * n.y * a;
                    t = float3(1.0 + s * n.x * n.x * a, s * k, -s * n.x);
                    b = float3(k, s + n.y * n.y * a, -n.y);
                }

                // Henyey-Greenstein 采样：cosθ（相对当前传播方向 dir）用解析反 CDF，φ 均匀；g > 0 前向，g = 0 为均匀球面
                float3 SampleHG(float3 dir, float g, float2 u)
                {
                    float cosT;
                    if (abs(g) < 1e-3) cosT = 1.0 - 2.0 * u.x;
                    else {
                        float s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
                        cosT = (1.0 + g * g - s * s) / (2.0 * g);
                    }
                    float  sinT = sqrt(max(0.0, 1.0 - cosT * cosT)), phi = 2.0 * PI * u.y;
                    float3 t, b;
                    BuildOrthonormalBasis(dir, t, b);
                    return (sinT * cos(phi)) * t + (sinT * sin(phi)) * b + cosT * dir;
                }`,
            },
        ],
    },
    externs: {
        TraceClosedMesh: '引擎 / 平台的光线求交：从体内点 p 沿 dir 在 (0, tMax] 内找同一闭合网格的最近交点，命中则填 HitInfo（t、P、外法线 N）并返回 true。DXR 1.1 里通常是 RayQuery::TraceRayInline + 只含该物体的实例掩码，法线按 primitive index 从顶点缓冲取。',
    },

    intro: {
        title: 'BSSRDF：从偶极子到 Normalized Diffusion，再到实时 SSS',
        flow: ['BSSRDF', '体参数', 'Dipole', 'SoG', 'Burley', 'CDF 采样', '可分离核', 'G-buffer · E', 'Blur H → V', '预积分 LUT', '生产版'],
        html: `<p>BRDF 假设光在哪儿进就在哪儿出；皮肤、大理石、蜡、牛奶里的光子会在体内随机游走几毫米才出来。本篇按数据流拆开次表面散射：<strong>物理模型</strong>（BSSRDF → 经典偶极子 → Sum-of-Gaussians / Burley 拟合剖面）、<strong>采样</strong>（按剖面 CDF 采半径）与<strong>实时近似</strong>（屏幕空间可分离模糊、预积分 LUT）。每张图都是页面内 JS 参考实现真实算出来的；顶栏切换实测介质（Jensen 2001 的 σ<sub>a</sub> / σ<sub>s</sub>′ 表）、剖面模型和自由程倍率，带 LIVE 的画布随光源扫动每帧重算。</p>
<p><strong>参数化约定</strong>：长度一律 mm，σ 为 mm⁻¹，R(r) 为 mm⁻²；η = η<sub>介质</sub>/η<sub>外</sub>（= 光射入时的 η<sub>t</sub>/η<sub>i</sub>，与系列一致），体内→体外的 Fresnel 用 1/η；A = ∫R dA 为表面反照率（multiple-scattering albedo）；Burley 用 <code>d = ℓ<sub>d</sub> / s(A)</code>、ℓ<sub>d</sub> = 1/σ<sub>tr</sub>、s 取 searchlight·ℓ<sub>d</sub> 拟合，A 取偶极子的总漫反射率 R<sub>d</sub> 以便两种模型能量一致（为什么不用 ℓ = 1/σ<sub>t</sub>′ 那条拟合，见 Step 05 / 11 的实测）。场景 1 unit = 12 mm；线性 RGB，显示统一走 <code>BX.display</code>（ACES + sRGB）。上一篇 <a href="bxdf_01_diffuse.html">01 漫反射</a> 的 Lambert 是这里的基线；下一篇 <a href="bxdf_06_layering.html">06 多层混合</a> 把 SSS 当作镜面涂层下的 base layer。</p>` + BX.nav('subsurface'),
    },

    sections: [{
        id: 'overview', title: '同一 G-buffer · 三种漫反射', position: 'top',
        sub: '左：Lambert（硬的 N·L 交界、硬阴影边）· 中：屏幕空间可分离 SSS（Step 09：Blur H → Blur V → Combine）· 右：预积分皮肤 LUT（Step 10）。光源方位角匀速来回扫动；顶栏切介质 / 剖面 / ℓ× 看差别。',
        html: `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">${['Lambert', 'Screen-space separable SSS', 'Pre-integrated LUT'].map((t) => `<div><canvas style="width:100%;aspect-ratio:4/3"></canvas><div style="font:11px Consolas,monospace;color:#888;margin-top:4px">${t}</div></div>`).join('')}</div>`,
        frame(el, S) {
            const cs = el.querySelectorAll('canvas');
            SS.drawBuf(cs[0], S.imgLambert); SS.drawBuf(cs[1], S.imgSSS); SS.drawBuf(cs[2], S.imgPre);
        },
    }],

    steps: [
        // ───────────────────────────────────────────── 01
        {
            id: 'bssrdf', title: '从 BRDF 到 BSSRDF：光从别处出来', tag: 'BSSRDF',
            desc: 'BRDF 是 BSSRDF 在“出射点 = 入射点”时的特例。半透明介质里，光在 x<sub>i</sub> 进入、在体内多次散射、从几毫米外的 x<sub>o</sub> 出来，所以 S(x<sub>i</sub>, ω<sub>i</sub>; x<sub>o</sub>, ω<sub>o</sub>) 多了两个空间自变量，渲染方程变成“表面 × 半球”的双重积分。多次散射后方向信息基本丢光，于是有<strong>可分离近似</strong>：方向只剩边界上的两次 Fresnel 透射，空间部分是一条径向剖面 R(r)。到了实时渲染，这又等价于<strong>“Lambert + 被 R 模糊过的辐照度”</strong>。Lambert 对皮肤、大理石、蜡失败的原因就在这里：它是 R = A·δ(r)，交界线是硬截断的 N·L，阴影边没有红色渗透，薄处不透光，看起来像石膏。',
            formula: [
                'L_o(x_o, ω_o) = ∫_A ∫_{H^2} S(x_i, ω_i; x_o, ω_o) · L_i(x_i, ω_i) · cos θ_i dω_{i} dA(x_i)',
                'BRDF 极限：S = f_r(ω_i, ω_o) · δ(x_i − x_o)              // 光只在入射点出射',
                '可分离：  S ≈ (1/π) · F_t(η, ω_i) · R(|x_o − x_i|) · F_t(η, ω_o)   // Jensen 2001',
                '实时版：  L_o ≈ (A/π) · (R̂ ⊛ E)(x_o),   R = A · R̂,   ∫ R̂ dA = 1   // Ft 吸收进 A',
                'Lambert： R̂ = δ(r)  ⇒  L_o = (A/π) · E(x_o)',
            ],
            code: {
                title: 'HLSL · 实时 SSS 与 Lambert 只差“E 有没有被模糊”', src: `
                // 可分离 BSSRDF：S ≈ C · Ft(η, wi) · R(|xo − xi|) · Ft(η, wo)
                // 实时版把 Ft 两项吸收进 A，剩下：漫反射 = A/π × (归一化剖面 ⊛ 辐照度)
                float3 ShadeSubsurface(float3 A, float3 blurredIrradiance)   // (R̂ ⊛ E)(xo)，来自 SSS pass
                {
                    return A * INV_PI * blurredIrradiance;
                }
                float3 ShadeLambert(float3 A, float3 irradiance)             // R̂ = δ(r)：不模糊
                {
                    return A * INV_PI * irradiance;
                }`,
            },
            previews: [
                {
                    id: 'bleed', label: '阴影边：M(x) = A·(K ⊛ step)', sub: 'x < 0 受光、x > 0 阴影 · 实线 = 当前剖面，虚线 = Lambert', size: [360, 180],
                    render: (c, S) => {
                        const b = S.bleed, series = [];
                        [0, 1, 2].forEach((k) => {
                            series.push({ data: Array.from(b.M[k]), x: Array.from(b.xs), color: SS.CH[k], label: `${SS.CHN[k]}  A=${S.P3[k].total.toFixed(3)}` });
                            series.push({ data: Array.from(b.xs, (x) => (x < 0 ? S.P3[k].total : 0)), x: Array.from(b.xs), color: SS.CH[k], dash: [3, 3], width: 1 });
                        });
                        PL.draw.plot(c, { x: [-b.X, b.X], y: [0, Math.max(...S.P3.map((p) => p.total)) * 1.08], xLabel: 'x (mm) · 阴影边在 x = 0', series, marks: [{ x: 0, label: 'shadow edge', color: '#777' }] });
                    },
                },
                {
                    id: 'sphLambert', label: 'Lambert 球', sub: 'f = A/π · 单个主光', render: (c, S) => {
                        const A = S.med.ch.map((q) => q.A);
                        BX.renderSphere(c, () => A.map((a) => a / Math.PI), { lights: BX.LIGHTS.key });
                    },
                },
                {
                    id: 'sphSSS', label: 'SSS 球（r = 3.6 mm）', sub: '同一 A，预积分 LUT（Step 10）', render: (c, S) => {
                        const A = S.med.ch.map((q) => q.A), D = [0, 0, 0], kap = 1 / 3.6;
                        BX.renderSphere(c, (wi) => { SS.lutSample(S.lut, wi[2], kap, D); const s = 1 / Math.max(Math.abs(wi[2]), 1e-4); return [A[0] / Math.PI * D[0] * s, A[1] / Math.PI * D[1] * s, A[2] / Math.PI * D[2] * s]; }, { lights: BX.LIGHTS.key, twoSided: true });
                    },
                },
                {
                    id: 'walk3d', needs: ['walk3d'], large: true, label: '3D：体内随机游走——光在 x<sub>i</sub> 进、在别处出',
                    sub: '半无限介质（约化 σ<sub>s</sub>′、各向同性散射、η 边界含 Fresnel / 全内反射），法向入射于 x<sub>i</sub>。每个通道画最先<strong>逃逸</strong>的 4 条路径与 160 个出射点（R / G / B 三色；落在盒外的不画），地面上的虚线圈 = 出射半径中位数，线框盒子 = 介质，半宽 = 红通道 90 % 出射半径。红光走得最远，蓝光几乎就在 x<sub>i</sub> 附近出来',
                    render: (c, S) => sssWalk3d(c, S),
                },
            ],
            notes: '左图是“阴影边”这个最简单的 1D 实验：辐照度是阶跃函数，出射亮度 = A × 剖面边缘积分与阶跃的卷积。红通道的 ℓ<sub>d</sub> 最长，所以红光渗进阴影最远——这正是皮肤阴影边发红的原因。拖 <code>ℓ ×</code> 看渗透宽度线性变化；切 <code>Ketchup</code> 看绿/蓝几乎不散射（α′ ≈ 0.07），只有红光在“发光”。3D 图把同一件事画成真实的光子路径（与 Step 11 的 ground truth 同一套规则）：同一个入射点，红光在体内走几毫米才出来，蓝光的出射点几乎都挤在 x<sub>i</sub> 周围零点几毫米内——出射点的面密度就是 R(r)。拖动旋转看路径的深度。',
            warn: '可分离近似丢掉了方向与距离的耦合：单次散射（强前向、依赖 ω<sub>i</sub>）、掠射入射时的方向性、以及“局部是半无限平面”之外的几何（薄片、凹角）。这些在 Step 11 的随机游走里才回来。',
        },
        // ───────────────────────────────────────────── 02
        {
            id: 'medium', title: '体参数：σ<sub>a</sub>、σ<sub>s</sub>′、σ<sub>t</sub>′、α′ 与平均自由程', tag: 'MEDIUM',
            desc: '扩散理论把体介质压缩成几个数。<strong>相似性关系</strong> σ<sub>s</sub>′ = σ<sub>s</sub>(1−g) 把强前向散射（皮肤 g ≈ 0.8）等效成各向同性散射——多次散射后两者分不出来。α′ 决定“能活着出来多少光”，σ<sub>tr</sub> 决定扩散光通量随距离的指数衰减。两个长度最常被混用：<strong>ℓ = 1/σ<sub>t</sub>′</strong> 是两次有效散射之间的平均距离（也是偶极子实源的深度），<strong>ℓ<sub>d</sub> = 1/σ<sub>tr</sub></strong> 是“模糊半径”的量级。美术参数（表面颜色 A + 散射距离）就是这些量的另一种坐标。',
            formula: [
                'σ_{s}′ = σ_s · (1 − g)        σ_{t}′ = σ_a + σ_{s}′        α′ = σ_{s}′ / σ_{t}′',
                'ℓ = 1/σ_{t}′  (输运平均自由程)        D = 1 / (3 σ_{t}′)  (扩散系数)',
                'σ_tr = √(3 σ_a σ_{t}′)        ℓ_d = 1/σ_tr  (漫射平均自由程)        φ(z) ∝ e^{−σ_tr z}',
            ],
            code: {
                title: 'HLSL · 由 σa / σs / g 求派生量（每通道）', src: `
                struct Medium  { float3 sigma_a; float3 sigma_s; float g; };   // 1/mm
                struct Derived { float3 sigma_tp, alphap, sigma_tr, ell, ell_d, D; };

                Derived DeriveMedium(Medium m)
                {
                    Derived o;
                    float3 sigma_sp = m.sigma_s * (1.0 - m.g);          // 约化散射 σs′（相似性关系）
                    o.sigma_tp = m.sigma_a + sigma_sp;                   // σt′
                    o.alphap   = sigma_sp / o.sigma_tp;                  // 约化反照率 α′
                    o.sigma_tr = sqrt(3.0 * m.sigma_a * o.sigma_tp);     // 有效衰减系数
                    o.ell      = 1.0 / o.sigma_tp;                       // ℓ  ：偶极子实源深度
                    o.ell_d    = 1.0 / o.sigma_tr;                       // ℓd ：扩散衰减长度
                    o.D        = o.ell / 3.0;                            // 扩散系数
                    return o;
                }`,
            },
            previews: [
                { id: 'medTable', label: '派生量（当前介质）', sub: 'σ 除以 ℓ× 后的值；A、s、d 见 Step 03/05', size: [360, 230], render: (c, S) => SS.table(c, S) },
                {
                    id: 'rdAlpha', label: 'A = ∫R<sub>d</sub> dA 随 1−α′', sub: '偶极子总漫反射率 · 竖线 = 三通道', size: [360, 180],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [1e-4, 1], y: [0, 1], logX: true, xLabel: '1 − α′（吸收占比，log）',
                        series: [{ fn: (x) => SS.totalRd(1 - x, S.med.Aeta), color: '#e0a060', label: `Aη = ${S.med.Aeta.toFixed(2)}` }, { fn: (x) => SS.totalRd(1 - x, 1), color: '#777', dash: [3, 3], label: 'Aη = 1 (η = 1)' }],
                        marks: S.med.ch.map((q, k) => ({ x: 1 - q.alpha, label: SS.CHN[k], color: SS.CH[k] })),
                    }),
                },
                {
                    id: 'penetr', label: '扩散通量 e^(−σtr·z)', sub: '红光走得最深 → 模糊最宽', size: [360, 180],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, S.rPlot], y: [1e-4, 1.5], logY: true, xLabel: 'z 或 r (mm)',
                        series: S.med.ch.map((q, k) => ({ fn: (z) => Math.exp(-q.str * z), color: SS.CH[k], label: `${SS.CHN[k]} ℓd = ${q.elld.toFixed(2)} mm` })),
                    }),
                },
            ],
            notes: '右上图：总反照率对 α′ 极其敏感——Skin1 红通道 α′ ≈ 0.96，蓝通道 0.68，前者 A ≈ 0.44、后者只剩 0.13，皮肤的颜色就是这么来的。改 <code>η</code> 看 A<sub>η</sub> 曲线移动：内部全反射越强，光被“关”在体内越久，A 越高。拖 <code>ℓ ×</code> 时表格里的长度线性变化，但 α′ 和 A 不变。',
            warn: '<strong>单位</strong>：Jensen 的表格是 mm⁻¹。场景单位不是 mm 时必须换算——本页场景 1 unit = 12 mm，屏幕空间核每像素还要按深度换算（Step 08）。<strong>ℓ 与 ℓ<sub>d</sub> 别混</strong>：Burley 的三个 s(A) 拟合分别对应不同的距离定义，用错了模糊半径会差 2–5 倍。',
        },
        // ───────────────────────────────────────────── 03
        {
            id: 'dipole', title: '经典偶极子扩散（Jensen 2001）：R<sub>d</sub>(r)', tag: 'DIPOLE',
            desc: '把入射光换成深度 z<sub>r</sub> = ℓ 处的一个各向同性点源，解半无限介质里的扩散方程。边界条件是“表面处向内的漫射通量 = 内部 Fresnel 反射回去的那部分”，用<strong>镜像法</strong>满足：在表面上方 z<sub>v</sub> 处放一个负的虚源，使通量在外推边界 z = 2A<sub>η</sub>D 处为零。A<sub>η</sub> 由体内一侧的漫射 Fresnel 反射率 F<sub>dr</sub>（= 2C<sub>1</sub>，内部反射的一阶矩，含全内反射）决定。出射率 R<sub>d</sub>(r) = 两个点源在 x<sub>o</sub> 处的通量梯度之和。它在 r → 0 有限（实源有深度），尾巴是 e<sup>−σ<sub>tr</sub>r</sup>/r²。<strong>Multipole</strong>（Donner & Jensen 2005）用一串镜像对处理有限厚度和多层（表皮 / 真皮），这里不展开。',
            formula: [
                'F_dr = −1.440/η^2 + 0.710/η + 0.668 + 0.0636 η     // Jensen 2001 拟合 ≈ 2C₁（体内一侧，含 TIR）',
                'A_η = (1 + F_dr) / (1 − F_dr)        // 改进边界（d\'Eon & Irving 2011）：Aη = (1 + 3C₂)/(1 − 2C₁)',
                'C_n = ∫_0^1 F_r(μ; 1/η) · μ^n dμ     // Fresnel 矩，体内→体外',
                'z_r = ℓ        z_v = z_r + 4 A_η D = ℓ · (1 + 4A_η/3)        d_r = √(r^2 + z_r^2)    d_v = √(r^2 + z_v^2)',
                'R_d(r) = α′/(4π) · [ z_r (σ_tr d_r + 1) e^{−σ_tr d_r} / d_r^3 + z_v (σ_tr d_v + 1) e^{−σ_tr d_v} / d_v^3 ]',
                '∫ R_d dA = (α′/2) · (1 + e^{−(4/3) A_η √(3(1−α′))}) · e^{−√(3(1−α′))}',
            ],
            code: {
                title: 'HLSL · DipoleRd(r)（Jensen 2001）', src: `
                float FdrJensen(float eta)                       // 体内一侧的漫射 Fresnel 反射率
                {
                    return -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta;
                }
                float3 DipoleRd(float r, Derived m, float eta)   // r: mm → 1/mm²
                {
                    float  Fdr = FdrJensen(eta);
                    float  A   = (1.0 + Fdr) / (1.0 - Fdr);
                    float3 zr  = m.ell;                          // 实源深度
                    float3 zv  = m.ell * (1.0 + 4.0 / 3.0 * A);  // 虚源高度 = zr + 4·A·D
                    float3 dr  = sqrt(r * r + zr * zr);
                    float3 dv  = sqrt(r * r + zv * zv);
                    float3 s   = m.sigma_tr;
                    float3 real = zr * (s * dr + 1.0) * exp(-s * dr) / (dr * dr * dr);
                    float3 virt = zv * (s * dv + 1.0) * exp(-s * dv) / (dv * dv * dv);
                    return m.alphap / (4.0 * PI) * (real + virt);
                }
                float3 DipoleTotalRd(float3 alphap, float A)     // ∫ 2πr·Rd dr 的闭式
                {
                    float3 q = sqrt(3.0 * (1.0 - alphap));
                    return 0.5 * alphap * (1.0 + exp(-4.0 / 3.0 * A * q)) * exp(-q);
                }`,
            },
            previews: [
                { id: 'dipGeo', label: '镜像源几何（R 通道，按比例）', sub: '实源 −z<sub>r</sub>、虚源 +z<sub>v</sub>、外推边界', size: [360, 200], render: (c, S) => SS.dipoleDiagram(c, S) },
                {
                    id: 'dipProf', label: 'R<sub>d</sub>(r) · log', sub: '图例里 ∫ = 数值积分 2πr·R<sub>d</sub>，应等于闭式 A', size: [360, 200],
                    render: (c, S) => SS.profPlot(c, S, [{ f: (r, k) => SS.dipole(r, S.med.ch[k]), label: (k) => { let s = 0; const n = 4000, rm = 40 * S.med.ch[k].elld; for (let i = 0; i < n; i++) { const r = (i + 0.5) / n * rm; s += 2 * Math.PI * r * SS.dipole(r, S.med.ch[k]) * rm / n; } return `∫=${s.toFixed(3)} A=${S.med.ch[k].Rd.toFixed(3)}`; } }]),
                },
                {
                    id: 'aEta', label: 'A<sub>η</sub>(η)', sub: '多项式 F<sub>dr</sub> vs 数值 Fresnel 矩', size: [360, 180],
                    render: (c, S) => {
                        const cache = {}, mom = (e) => (cache[e] = cache[e] || SS.moments(e, 256));
                        PL.draw.plot(c, {
                            x: [1, 2], xLabel: 'η（介质 / 外部）',
                            series: [
                                { fn: (e) => { const f = SS.fdrPoly(e); return (1 + f) / (1 - f); }, color: '#e0a060', label: 'Fdr poly', samples: 60 },
                                { fn: (e) => { const m = mom(e); return (1 + 2 * m[0]) / (1 - 2 * m[0]); }, color: '#6ab0e0', dash: [4, 3], label: '2C₁ numeric', samples: 60 },
                                { fn: (e) => { const m = mom(e); return (1 + 3 * m[1]) / (1 - 2 * m[0]); }, color: '#5ad07a', label: 'C₁,C₂ improved', samples: 60 },
                            ],
                            marks: [{ x: S.eta, label: `η=${S.eta.toFixed(2)}`, color: '#e8c25a' }],
                        });
                    },
                },
            ],
            notes: '右上图：三通道在 r ≲ ℓ 处量级相近，远处斜率由 σ<sub>tr</sub> 决定——红通道最平、走得最远；图例里数值积分 ∫2πr·R<sub>d</sub> 与闭式 A 一致。拖 <code>η</code> 看曲线整体移动（A<sub>η</sub> 变大 → 虚源更远 → 体内滞留更久、总能量下降）。下图：Jensen 的多项式与数值积分的 2C<sub>1</sub> 在 η ∈ [1, 2] 几乎重合，“改进边界”在 η = 1.3 时 A<sub>η</sub> 小约 10%。',
            warn: '<strong>F<sub>dr</sub> 是体内一侧的值</strong>（光从介质射向空气，η<sub>t</sub>/η<sub>i</sub> = 1/η，临界角外全反射）。用成体外的 F<sub>dr</sub>（η = 1.3 时只有约 0.07）A<sub>η</sub> 会从 2.6 掉到 1.15，剖面明显变窄变暗。偶极子没有单次散射项、扩散近似在 α′ 低或 r ≲ ℓ 时偏差大——Step 11 的 MC 对比里看得到。',
        },
        // ───────────────────────────────────────────── 04
        {
            id: 'sog', title: 'Sum-of-Gaussians 拟合（d\'Eon & Luebke 2007）', tag: 'SOG',
            desc: '偶极子 / 多极子剖面求值不贵，但拿它做 2D 卷积很贵。GPU Gems 3 的皮肤渲染观察到：径向剖面可以用少数几个 2D Gaussian 之和逼近，而 Gaussian 有两条 GPU 友好的性质：<strong>2D Gaussian 可分离</strong>（两次 1D 模糊），<strong>Gaussian ⊛ Gaussian 仍是 Gaussian</strong>（方差相加），所以纹理空间里每一层都可以在上一层模糊结果上“递增”再模糊。方差在 RGB 间共享（每层模糊只做一次），颜色差异全在权重里。这里对当前介质的偶极子做同样的拟合：6 个方差取几何级数，权重用 r 加权的非负最小二乘求出（Gauss-Seidel 投影迭代）。',
            formula: [
                'R(r) ≈ Σ_{i=1}^{6} w_i · G(v_i, r),        G(v, r) = e^{−r^2/(2v)} / (2π v)        // ∫ G dA = 1',
                'min_{w ≥ 0}  ∫ r · (R_d(r) − Σ w_i G(v_i, r))^2 dr                              // r 加权 = 按面元计误差',
                'G(v_1) ⊛ G(v_2) = G(v_1 + v_2)          // 纹理空间递增模糊：第 i 层只需再模糊 vᵢ − vᵢ₋₁',
            ],
            code: {
                title: 'HLSL · d\'Eon & Luebke 三层皮肤 SoG（方差 mm²）', src: `
                static const float  kVar[6] = { 0.0064, 0.0484, 0.187, 0.567, 1.99, 7.41 };
                static const float3 kW[6]   = { float3(0.233, 0.455, 0.649), float3(0.100, 0.336, 0.344),
                                                float3(0.118, 0.198, 0.000), float3(0.113, 0.007, 0.007),
                                                float3(0.358, 0.004, 0.000), float3(0.078, 0.000, 0.000) };
                float Gauss2D(float v, float r) { return exp(-r * r / (2.0 * v)) / (2.0 * PI * v); }

                float3 SkinProfileSoG(float r)                   // 每通道 Σw = 1：颜色交给 albedo 贴图
                {
                    float3 R = 0;
                    [unroll] for (int i = 0; i < 6; i++) R += kW[i] * Gauss2D(kVar[i], r);
                    return R;
                }
                // 纹理空间：irradiance[0] = E；irradiance[i] = Blur(irradiance[i−1], kVar[i] − kVar[i−1])
                // 最终 = Σ kW[i] · irradiance[i]（可分离 1D Gaussian，每层两次 pass）`,
            },
            previews: [
                {
                    id: 'sogFit', label: '偶极子（实线）vs SoG 拟合（虚线）', sub: '6 个共享方差 · r 加权 NNLS', size: [360, 200],
                    render: (c, S) => SS.profPlot(c, S, [
                        { f: (r, k) => SS.dipole(r, S.med.ch[k]) },
                        { f: (r, k) => SS.sog(r, S.fit, k), dash: [4, 3], label: (k) => `Σw=${S.fit.w[k].reduce((a, b) => a + b, 0).toFixed(3)} (A=${S.med.ch[k].Rd.toFixed(3)})` },
                    ]),
                },
                {
                    id: 'sogW', label: '拟合权重 w<sub>i</sub>', sub: '横轴 = 每个 Gaussian 的标准差 √v<sub>i</sub>', size: [360, 180],
                    render: (c, S) => SS.bars(c, S.fit.v.map((v, i) => ({ label: `√v=${Math.sqrt(v).toFixed(2)}\nmm`, v: [S.fit.w[0][i], S.fit.w[1][i], S.fit.w[2][i]] })), { title: '当前介质 · 每组 = R G B' }),
                },
                {
                    id: 'deon', label: 'GPU Gems 3 皮肤剖面（固定）', sub: '三层皮肤 multipole 的 6-Gaussian 拟合，每通道 ∫ = 1', size: [360, 180],
                    render: (c, S) => SS.profPlot(c, S, [{ f: (r, k) => SS.sog(r, SS.DEON, k) }], { r: 4, y: [1e-6, 30] }),
                },
            ],
            notes: '左图：中段贴合得很好，最远处的尾巴 Gaussian 总会欠估计（e<sup>−r²</sup> 比 e<sup>−σ<sub>tr</sub>r</sup> 衰减得快）。切 <code>Wholemilk</code> 看宽剖面需要的大方差。顶栏 <code>剖面 = SoG fit</code> 时，屏幕空间核和 LUT 都用这里拟合出的剖面。下图是 d\'Eon 发表的皮肤剖面：红通道 0.358 的权重落在 √v ≈ 1.4 mm 的宽 Gaussian 上；蓝通道 99% 的权重在 √v ≤ 0.22 mm 的两个窄 Gaussian 里，绿通道多一层 0.43 mm。',
            warn: 'GPU Gems 3 的权重每通道归一化为 1，颜色来自 albedo 贴图（再配 pre/post-scatter texturing 的 √albedo 拆分）；这里的拟合保留了绝对能量（Σw ≈ A）。若把两种约定混用，会把颜色乘两遍。',
        },
        // ───────────────────────────────────────────── 05
        {
            id: 'burley', title: 'Burley Normalized Diffusion（Christensen & Burley 2015）', tag: 'BURLEY',
            desc: '与其拟合某个物理近似，不如直接拟合 brute-force Monte Carlo 的参考曲线（含单次散射）。Burley 注意到参考曲线的形状可以用<strong>两个指数之和除以 r</strong> 表示；归一化让 ∫R dA = A 精确成立，而 A 就是美术给的表面颜色——不需要像偶极子那样数值反演 σ<sub>a</sub>/σ<sub>s</sub>′。剩下唯一的问题是“散射距离”对应多宽的形状：Christensen & Burley 对三种距离定义各给了一个 s(A) 拟合。本页用 searchlight · ℓ<sub>d</sub>（ℓ<sub>d</sub> = 1/σ<sub>tr</sub>）那一条，d = ℓ<sub>d</sub>/s(A)；A 取偶极子的 ∫R<sub>d</sub>，让两条曲线能量相同、只比形状。',
            formula: [
                'R(r) = A · s · (e^{−s r} + e^{−s r/3}) / (8π r) = A · (e^{−r/d} + e^{−r/(3d)}) / (8π d r),    d = 1/s',
                '∫_0^∞ R(r) · 2πr dr = (A/(4d)) · (d + 3d) = A',
                'd = ℓ / s(A):     s = 1.85 − A + 7|A − 0.8|^3        // searchlight · ℓ = 1/σt′',
                '                  s = 1.9 − A + 3.5 (A − 0.8)^2       // 漫射入射 · ℓ',
                'd = ℓ_d / s(A):   s = 3.5 + 100 (A − 0.33)^4          // searchlight · ℓd = 1/σtr（本页）',
            ],
            code: { title: 'HLSL · Burley 剖面与形状拟合（Common 里也有）', src: SSS_HLSL_PROFILE },
            previews: [
                {
                    id: 'burProf', label: 'Burley（实线）vs 偶极子（虚线）', sub: '同一 A · Burley 用 ℓ<sub>d</sub>，偶极子用 ℓ = 1/σ<sub>t</sub>′ · log', size: [360, 200],
                    render: (c, S) => SS.profPlot(c, S, [
                        { f: (r, k) => SS.burley(r, S.med.ch[k]), label: (k) => `d=${S.med.ch[k].d.toFixed(2)}mm s=${S.med.ch[k].s.toFixed(2)}` },
                        { f: (r, k) => SS.dipole(r, S.med.ch[k]), dash: [4, 3] },
                    ]),
                },
                {
                    id: 'sFit', label: 's(A) 三种拟合', sub: '竖线 = 当前三通道的 A', size: [360, 180],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, 1], y: [0, 14], xLabel: 'A（表面反照率）',
                        series: [{ fn: SS.sMfp, color: '#e0a060', label: 'searchlight · ℓ' }, { fn: SS.sDiffuse, color: '#c678dd', label: 'diffuse · ℓ', dash: [4, 3] }, { fn: SS.sDmfp, color: '#56b6c2', label: 'searchlight · ℓd（本页）' }],
                        marks: S.med.ch.map((q, k) => ({ x: q.A, label: SS.CHN[k], color: SS.CH[k] })),
                    }),
                },
            ],
            notes: '左图：两者能量相同（∫ 都是 A），差别在形状——Burley 在 r → 0 处是 1/r 尖峰（单次散射的贡献），偶极子是平顶；中远距离斜率接近。切 <code>Ketchup</code>（A 很低）看差别变大，这正是 Burley 拟合 MC 而偶极子失效的区域；切 <code>Marble</code> 看高 A 时 s(A) 变大、Burley 的远尾比偶极子短。右图：同一个“距离数字”在三种定义下对应的模糊宽度可以差好几倍。',
            warn: '<strong>参数化要和拟合配套</strong>：本页用 Step 11 的随机游走实测（能量加权误差）——η = 1 时 ℓ 拟合 + MC 的 A 误差约 3%（与原文 5.3% 相符）；但 η = 1.3 的 Fresnel 边界下，ℓ 拟合配偶极子的 A 误差升到 30–40%（尾巴短 2–3 倍），ℓ<sub>d</sub> 拟合只有 11–19%，所以选了后者。另外 Christensen 2015 的 SIGGRAPH talk 把第一条写作 <code>7(0.8 − A)³</code>（无绝对值），Pixar 技术备忘录 / UE 用 <code>7|A − 0.8|³</code>，只在 A > 0.8 时不同。R(r) 在 r → 0 处 1/r 奇异但可积：数值积分、核权重、LUT 烘焙都必须按<strong>面积</strong>积分，点采样中心像素会发散。',
        },
        // ───────────────────────────────────────────── 06
        {
            id: 'sample', title: '按 CDF 重要性采样剖面', tag: 'SAMPLE',
            desc: '离线路径追踪里（King et al. 2013 的 probe-ray 方法），BSSRDF 的空间积分靠重要性采样：在 x<sub>o</sub> 的切平面上按 pdf ∝ 2πr·R(r) 采半径、均匀采 φ，再沿投影轴打一条探测射线找到真实表面上的 x<sub>i</sub>。Burley 剖面的径向 CDF 有闭式；令 x = e<sup>−r/(3d)</sup> 后 CDF 是 x 的三次方程，Golubev 给出了 Cardano 形式的解析反函数——不需要 Newton 迭代，也不需要查表。',
            formula: [
                'pdf(r) = 2πr · R(r) / A = (e^{−r/d} + e^{−r/(3d)}) / (4d)',
                'CDF(r) = 1 − (1/4) e^{−r/d} − (3/4) e^{−r/(3d)}',
                'u′ = 1 − u        G = 1 + 4u′ (2u′ + √(1 + 4u′^2))',
                'r = 3d · ln( (1 + G^{−1/3} + G^{1/3}) / (4u′) )        // x = exp(−r/3d) 满足 x³ + 3x = 4u′',
            ],
            code: {
                title: 'HLSL · SampleBurleyRadius（Golubev 的解析反函数）', src: `
                float SampleBurleyRadius(float u, float d)       // u ∈ [0,1) → r ∈ [0, ∞)
                {
                    u = 1.0 - u;                                 // 用补 CDF：保证 u > 0
                    float g = 1.0 + (4.0 * u) * (2.0 * u + sqrt(1.0 + (4.0 * u) * u));
                    float n = pow(g, -1.0 / 3.0);                // g^(−1/3)
                    float p = g * n * n;                         // g^(+1/3)
                    float c = 1.0 + p + n;
                    return 3.0 * log(c / (4.0 * u)) * d;
                }
                float PdfBurleyRadius(float r, float d)          // 对 r 的密度（已含 2πr）
                {
                    return (exp(-r / d) + exp(-r / (3.0 * d))) / (4.0 * d);
                }
                // RGB 三个 d：按 1/3 选通道采 r，pdf 用三通道 pdf 的平均（one-sample MIS）
                // 探测轴：50% 沿 N、25% 沿 T、25% 沿 B，同样按三轴 pdf 做 MIS（King et al. 2013）`,
            },
            previews: [
                { id: 'sampDisk', label: '切平面上的样本', sub: '每通道 1400 个 (r, φ)', size: [200, 200], render: (c, S) => SS.scatter(c, S) },
                {
                    id: 'sampHist', label: 'R 通道：采样直方图 vs pdf', sub: '8000 个随机样本 · 柱 = 直方图密度', size: [360, 200],
                    render: (c, S) => {
                        const s = S.samp, d = S.med.ch[0].d, xs = s.hist.map((_, b) => (b + 0.5) * s.rH / s.NB);
                        PL.draw.plot(c, {
                            x: [0, s.rH], y: [0, 0.6 / d], xLabel: 'r (mm)',
                            series: [{ data: s.hist, x: xs, bar: true, color: 'rgba(224,108,117,0.55)', label: 'histogram' }, { fn: (r) => (Math.exp(-r / d) + Math.exp(-r / (3 * d))) / (4 * d), color: '#e8c25a', label: 'pdf(r)' }],
                        });
                    },
                },
            ],
            notes: '左图虚线圆是各通道的中位半径（u = 0.5）：红通道的样本散得最开。改 <code>ℓ ×</code> 只改尺度，直方图形状不变——形状完全由 d 决定，这也是 Burley 剖面能被一个 1D 函数表格化 / 解析采样的原因。',
            warn: 'pdf 必须按<strong>实际</strong>用到的采样策略算（通道选择 × 轴选择的 MIS 混合），否则 RGB 之间出现色偏。探测射线命中多个表面时（耳朵、手指）要随机选一个并乘命中数，否则薄处会偏暗。',
        },
        // ───────────────────────────────────────────── 07
        {
            id: 'kernel', title: '2D 核 → 可分离 1D 核（rank-1 近似）', tag: 'KERNEL',
            desc: '屏幕空间卷积的直接做法是 2D 盘采样（HDRP 就这样做，配蓝噪声与时域累积）。Jimenez 2015 改成先 H 后 V 两次 1D 模糊：每像素 2N 次采样代替 N² 次。但径向核里只有 Gaussian 真正可分离，所以 K(x)·K(y) 只是 rank-1 近似。本页的 1D 核取 2D 核的<strong>边缘积分</strong> K(x)，tap 权重 = 该 tap 单元内的质量：这样轴对齐的阴影边（1D 阶跃）结果与 2D 卷积严格一致，误差集中在对角方向。tap 偏移用 Jimenez 的二次分布（中心密、外围稀），以 mm 存储，每像素再按深度换算成像素。',
            formula: [
                'K(x) = ∫ R̂(√(x^2 + y^2)) dy                       // 1D 边缘（line spread function）',
                'F(X) = ∫_{|x|<X} K(x) dx = (2/π) ∫_0^∞ P(X·cosh u) / cosh u du     // P = 径向 CDF，分部积分 + 换元',
                'w_i = ∫_{a_i}^{b_i} K(x) dx         o_i = R_K · t_i |t_i|,   t_i ∈ [−1, 1]      // RK 由 F(RK) = 0.99 定',
                '(R̂ ⊛ E)(x, y) ≈ Σ_j w_j Σ_i w_i · E(x + o_i, y + o_j)      // rank-1：K ⊗ K ≠ R̂',
            ],
            code: {
                title: 'C++ · CPU 端构建核（param：taps 或剖面变化时）', lang: 'cpp', src: `
                // kernel[i].rgb = 三通道权重，kernel[i].w = 偏移（mm）；kernel[c0] 为中心 tap
                float StripMass(const Profile& P, float X)      // |x| < X 条带内的质量 F(X)
                {
                    const int n = 160; const float U = 12.0f, du = U / n;
                    float s = 0.0f;
                    for (int i = 0; i < n; i++) { float u = (i + 0.5f) * du, c = coshf(u); s += P.CDF(X * c) / c; }
                    float tail = 0.5f * PI - 2.0f * atanf(tanhf(0.5f * U));   // P ≈ 1 的尾部
                    return fminf(1.0f, 2.0f / PI * (s * du + tail));
                }
                float HalfMass(const Profile& P, float x) { return x == 0 ? 0 : copysignf(0.5f * StripMass(P, fabsf(x)), x); }

                void BuildKernel(const Profile P[3], int N, float RK, float4* kernel)
                {
                    for (int i = 0; i < N; i++) {
                        float t = 2.0f * i / (N - 1) - 1.0f;
                        kernel[i].w = RK * t * fabsf(t);                     // 二次分布（Jimenez: EXPONENT = 2）
                    }
                    for (int c = 0; c < 3; c++)
                        for (int i = 0; i < N; i++) {
                            float lo = (i == 0)     ? -0.5f : HalfMass(P[c], 0.5f * (kernel[i - 1].w + kernel[i].w));
                            float hi = (i == N - 1) ? +0.5f : HalfMass(P[c], 0.5f * (kernel[i].w + kernel[i + 1].w));
                            kernel[i][c] = hi - lo;                          // 单元质量：Σ = 1，中心奇异也没问题
                        }
                }`,
            },
            previews: [
                { id: 'k2true', label: '真 2D 核 R̂(r)', sub: '单元质量 · 每通道 log（4 decades）', render: (c, S) => k2draw(c, S, 'tru') },
                { id: 'k2sep', label: '可分离 K ⊗ K', sub: '同一显示映射', render: (c, S) => k2draw(c, S, 'sep') },
                {
                    id: 'k2err', label: 'log₁₀(K⊗K / R̂) · R 通道', sub: '橙 = 可分离偏大，蓝 = 偏小（±1 decade）',
                    render: (c, S) => { const k = S.k2, G = k.G, e = PL.buf(S, 'k2e', G * G); for (let i = 0; i < G * G; i++) e[i] = Math.log10(Math.max(k.sep[0][i], 1e-30) / Math.max(k.tru[0][i], 1e-30)); PL.draw.field(c, e, G, G, { range: [-1, 1], map: 'diverge' }); },
                    probe: (x, y, S) => { const k = S.k2, i = y * k.G + x; return `(${(-k.X2 + (x + 0.5) * k.h).toFixed(2)}, ${(-k.X2 + (y + 0.5) * k.h).toFixed(2)}) mm\nratio = ${(k.sep[0][i] / k.tru[0][i]).toFixed(3)}`; },
                },
                {
                    id: 'k1taps', label: '1D 核：tap 密度 w<sub>i</sub>/Δ<sub>i</sub>', sub: '点 = tap 偏移（mm）· log', size: [360, 180],
                    render: (c, S) => {
                        const K = S.kern;
                        PL.draw.plot(c, {
                            x: [-K.RK * 1.05, K.RK * 1.05], logY: true, xLabel: `offset (mm) · ${K.N} taps · R_K = ${K.RK.toFixed(2)} mm`,
                            series: [0, 1, 2].map((k) => ({ data: Array.from(K.w[k], (w, i) => w / K.cell[i]), x: K.off, color: SS.CH[k], label: `${SS.CHN[k]} centre w=${K.w[k][K.c0].toFixed(3)}` })),
                            marks: K.off.filter((_, i) => i % 2 === 0).map((o) => ({ x: o, color: '#2a2f35' })),
                        });
                    },
                },
            ],
            notes: '右上误差图里，沿坐标轴可分离核偏大、沿对角线偏小——“十字形”是 rank-1 近似的签名；Gaussian 剖面（试 <code>剖面 = SoG fit</code>）误差最小。<code>taps</code> 越少，外围 tap 越稀：宽剖面（Wholemilk、<code>ℓ ×</code> 大）时阴影边会出现台阶状重影。',
            warn: '核宽度必须以世界单位（mm）定义、每像素按深度换算；按像素定义的核会在相机拉近 / 拉远时改变材质外观。中心 tap 不能点采样 Burley 的 1/r 奇异——用单元质量（或 CDF 差）求权重。',
        },
        // ───────────────────────────────────────────── 08
        {
            id: 'gbuf', title: 'G-buffer 与漫反射辐照度 E', tag: 'GBUF',
            desc: '场景：大球（r = 7.2 mm）、小球（r = 3.6 mm）、立方体（边长 9.6 mm）三个 SSS 物体放在 Lambert 棋盘地面上（1 unit = 12 mm，像鼻尖 / 耳垂的尺度）。<code>PL.scene</code> 光线投射出 160×120 的 G-buffer：线性 view 深度、世界法线、SSS 模板（object id）。光照 = 方向光 + 天空环境，阴影用阴影射线（引擎里是 shadow map）。<strong>E 只含光照、不含 albedo</strong>——它是要被模糊的信号；albedo 在 Combine 里乘回（post-scatter texturing），纹理细节不会被模糊掉。',
            formula: [
                'E(x) = E_L · saturate(N·L) · V(x) + E_amb(N)        // 辐照度；SSS 只模糊它',
                'px/mm = (H/2) / (z_view · tan(fov/2) · mm_per_unit)    // 每像素换算核宽度',
            ],
            code: {
                title: 'HLSL · PS_DiffuseLighting（延迟光照：漫反射辐照度与镜面分开输出）', src: `
                Texture2D<float>  SceneDepth;      // 线性 view z（scene units）
                Texture2D<float4> GBufferNormal;   // world normal
                Texture2D<float>  ShadowMask;      // 本页：阴影射线；引擎：shadow map

                struct LightingOut { float4 diffuseIrr : SV_Target0; float4 specular : SV_Target1; };

                LightingOut PS_DiffuseLighting(float4 pos : SV_Position)
                {
                    uint2  px  = uint2(pos.xy);
                    float3 N   = GBufferNormal[px].xyz;
                    float3 P   = ReconstructWorldPos(px, SceneDepth[px]);
                    float3 V   = normalize(CameraPos - P);
                    float  NoL = dot(N, LightDir), vis = ShadowMask[px];
                    LightingOut o;
                    o.diffuseIrr = float4(LightIrradiance * saturate(NoL) * vis + AmbientIrr * (0.6 + 0.4 * N.y), 1);
                    float3 H = normalize(LightDir + V);
                    float  a = Roughness * Roughness, NoV = saturate(dot(N, V));
                    float3 spec = D_GGX(saturate(dot(N, H)), a) * V_SmithGGXCorrelated(NoV, saturate(NoL), a)
                                * F_Schlick(F0FromIor(Eta), dot(V, H));
                    o.specular = float4(spec * LightIrradiance * saturate(NoL) * vis, 1);   // 镜面不进 SSS 模糊
                    return o;
                }`,
            },
            previews: [
                {
                    id: 'gbDepth', label: 'SceneDepth', sub: '线性 view z', render: (c, S) => PL.draw.field(c, S.depthVis, SS.W, SS.H, { map: 'gray' }),
                    probe: (x, y, S) => { const i = y * SS.W + x; return `z = ${S.gb.depth[i].toFixed(3)} unit = ${(S.gb.depth[i] * SS.MMU).toFixed(1)} mm\nid = ${S.gb.id[i]}`; },
                },
                { id: 'gbNormal', label: 'GBufferNormal', sub: 'n·0.5 + 0.5', render: (c, S) => PL.draw.normal(c, S.gb.nrm, SS.W, SS.H) },
                {
                    id: 'gbMask', label: 'SSSMask（模板 / id）', sub: '0 = 地面（Lambert），1–3 = SSS 物体', render: (c, S) => {
                        const pal = [[0.25, 0.25, 0.25], [0.88, 0.42, 0.46], [0.9, 0.62, 0.38], [0.78, 0.47, 0.86]];
                        PL.draw.pixels(c, SS.W, SS.H, (x, y, out) => { const id = S.gb.id[y * SS.W + x]; const p = id < 0 ? [0.03, 0.03, 0.04] : pal[id] || pal[1]; out[0] = p[0]; out[1] = p[1]; out[2] = p[2]; return out; });
                    },
                },
                {
                    id: 'gbE', label: 'DiffuseIrradiance E', sub: '不含 albedo · 光源扫动', live: true,
                    render: (c, S) => PL.draw.field(c, S.E, SS.W, SS.H, { map: 'inferno', range: [0, SS.EL + 0.25], stretch: 'sqrt' }),
                    probe: (x, y, S) => { const i = y * SS.W + x; return `E = ${S.E[i].toFixed(3)}\nN·L = ${S.NoL[i].toFixed(3)} · V = ${S.vis[i]}`; },
                },
            ],
            notes: '大球的影子随光源扫过小球和立方体——这些硬阴影边就是下一步 SSS 最显眼的地方。悬停深度图读每像素的 mm 深度：核宽度 px/mm 就由它换算（远处物体同样的 mm 核占更少像素）。',
            warn: '<strong>E 里不要乘 albedo</strong>（或只乘 √albedo，d\'Eon 的 pre/post 拆分）：先乘 albedo 再模糊会把雀斑、毛孔等贴图细节一起糊掉。镜面必须单独输出、不进模糊，否则高光会晕开成“蜡像”。',
        },
        // ───────────────────────────────────────────── 09
        {
            id: 'ssss', title: '屏幕空间可分离 SSS：Blur H → Blur V → Combine', tag: 'SSSS',
            desc: '两次全屏 pass，每次沿一个轴用 Step 07 的 1D 核卷积 E；只在 SSS 模板像素上运行。两个关键细节来自 Jimenez 2015：① tap 偏移以 mm 存储，每像素按自身深度换成像素（远处核变窄）；② <strong>follow surface</strong>：样本若落到别的物体上（模板 id 不同）或深度差相对 tap 距离太大（坡度 > 45° 开始拒绝），就把它 lerp 回中心颜色，避免背景 / 地面的光被“吸”进物体、物体边缘出现暗晕。最后 Combine 乘回 albedo、加上未模糊的镜面。',
            formula: [
                'H:  B_x(p)  = Σ_i w_i · lerp(E(p + o_i·k_p·x̂), E(p), s_i)          // kₚ = px/mm（按深度）',
                'V:  B_xy(p) = Σ_i w_i · lerp(B_x(p + o_i·k_p·ŷ), B_x(p), s_i)',
                's_i = saturate(|z_i − z_p| / |o_i| − 1)        id(i) ≠ id(p) ⇒ s_i = 1       // follow surface',
                'L_o = (A/π) · B_xy + L_spec',
            ],
            code: [
                {
                    title: 'HLSL · PS_SSSBlur（同一 shader，BlurDir = (1,0) / (0,1) 各画一次）', src: `
                    StructuredBuffer<float4> SSSKernel;   // .rgb = 权重，.w = 偏移 (mm)；[0] = 中心 tap
                    Texture2D<float4> SrcTex;             // H: DiffuseIrradiance；V: SSSBlurH
                    Texture2D<uint>   SSSMask;            // 0 = 非 SSS；否则 = profile / object id

                    float4 PS_SSSBlur(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target0
                    {
                        float3 colorM = SrcTex.SampleLevel(PointClamp, uv, 0).rgb;
                        uint   idM    = SSSMask[uint2(pos.xy)];
                        if (idM == 0) return float4(colorM, 1);                  // 模板：非 SSS 像素直接通过
                        float  depthM  = SceneDepth.SampleLevel(PointClamp, uv, 0);
                        float  pxPerMM = 0.5 * ViewSize.y / (depthM * TanHalfFovY * MMPerUnit);
                        float2 stepUV  = BlurDir * pxPerMM * InvViewSize;

                        float3 acc = colorM * SSSKernel[0].rgb;
                        [unroll] for (int i = 1; i < N_SAMPLES; i++) {
                            float  o     = SSSKernel[i].w;                          // mm
                            float2 uvS   = uv + o * stepUV;
                            float3 color = SrcTex.SampleLevel(LinearClamp, uvS, 0).rgb;
                            if (FollowSurface > 0) {
                                float depth = SceneDepth.SampleLevel(PointClamp, uvS, 0);
                                uint  id    = SSSMask[uint2(uvS * ViewSize)];
                                float s = (id != idM) ? 1.0 : saturate(abs(depth - depthM) * MMPerUnit / abs(o) - 1.0);
                                color = lerp(color, colorM, s);                    // 跨到别的表面 → 换回中心色
                            }
                            acc += SSSKernel[i].rgb * color;
                        }
                        return float4(acc, 1);
                    }`,
                },
                {
                    title: 'HLSL · PS_SSSCombine', src: `
                    Texture2D<float4> SSSBlurHV, SpecularLighting, GBufferAlbedo, DiffuseIrradiance;
                    float4 PS_SSSCombine(float4 pos : SV_Position) : SV_Target0
                    {
                        uint2  px = uint2(pos.xy);
                        float3 A  = GBufferAlbedo[px].rgb;
                        float3 E  = SSSMask[px] ? SSSBlurHV[px].rgb : DiffuseIrradiance[px].rgb;
                        return float4(A * INV_PI * E + SpecularLighting[px].rgb, 1);   // post-scatter texturing
                    }`,
                },
            ],
            previews: [
                { id: 'ssH', label: 'SSSBlurH', sub: '只沿 x 模糊 · 白 albedo 显示，非 SSS 像素压暗', live: true, render: (c, S) => SS.drawBuf(c, S.BH, 1 / Math.PI, S.gb.id) },
                { id: 'ssV', label: 'SSSBlurHV', sub: '再沿 y 模糊', live: true, render: (c, S) => SS.drawBuf(c, S.BV, 1 / Math.PI, S.gb.id) },
                {
                    id: 'ssFinal', label: 'SceneColor（SSSS）', sub: 'A/π · B<sub>xy</sub> + 镜面', live: true, large: true, render: (c, S) => SS.drawBuf(c, S.imgSSS),
                    probe: (x, y, S) => { const i = (y * SS.W + x) * 3; return `L = (${S.imgSSS[i].toFixed(3)}, ${S.imgSSS[i + 1].toFixed(3)}, ${S.imgSSS[i + 2].toFixed(3)})\nLambert = (${S.imgLambert[i].toFixed(3)}, ${S.imgLambert[i + 1].toFixed(3)}, ${S.imgLambert[i + 2].toFixed(3)})`; },
                },
                { id: 'ssLambert', label: 'Lambert 对照', sub: '同一 E，不模糊', live: true, render: (c, S) => SS.drawBuf(c, S.imgLambert) },
                { id: 'ssDiff', label: 'Δ 亮度 = SSSS − Lambert', sub: '橙 = 变亮（光被搬进来），蓝 = 变暗', live: true, render: (c, S) => PL.draw.field(c, S.imgDiff, SS.W, SS.H, { symmetric: true, hold: true }) },
            ],
            notes: 'Δ 图里能量守恒一目了然：亮侧边缘变暗、阴影侧变亮，红通道搬得最远。关掉顶栏 <code>follow surface</code>：物体轮廓会吸进黑色天空和地面的光（边缘暗晕 / 发灰），小球与地面接触处尤其明显。<code>taps = 9</code> + <code>Wholemilk</code> 看欠采样的台阶重影。',
            warn: 'Jimenez 原版的 follow-surface 用固定常数 <code>saturate(300·dpw·sssWidth·|Δz|)</code>；这里改成按 tap 距离归一化的坡度阈值，物理含义更清楚但同样是启发式。可分离核在对角边上会出现十字形伪影（Step 07），生产中常把每帧的核旋转一个随机角度再配合 TAA 抹平。',
        },
        // ───────────────────────────────────────────── 10
        {
            id: 'lut', title: '预积分皮肤（Penner 2011）：LUT(N·L, κ)', tag: 'LUT',
            desc: '不做屏幕空间模糊，而是承认“交界线变软”主要来自<strong>曲率</strong>：在半径 r 的球面上，邻近点的法线彼此不同，按剖面加权平均它们的 saturate(N·L) 就得到一个只依赖 (N·L, r) 的函数，离线烘成 2D LUT，前向 / 延迟着色里一次查表。Penner 原文在一个圆环上积分；Burley 剖面在 r → 0 处 1/r 奇异，环积分会退化成 saturate(N·L)，所以本页改为<strong>在整个球面上</strong>积分（面元 sinψ dψ 抵消奇异性），φ 方向的积分有解析式。曲率在 shader 里用屏幕导数估计：κ = |fwidth(N)| / |fwidth(P)|。',
            formula: [
                'Penner：D(θ, r) = ∫_{−π}^{π} saturate(cos(θ + x)) · R(2r sin(x/2)) dx / ∫_{−π}^{π} R(2r sin(x/2)) dx',
                '本页：  D(θ, r) = ∫_0^{π} R(2r sin(ψ/2)) sinψ · I(θ, ψ) dψ / (2π ∫_0^{π} R(2r sin(ψ/2)) sinψ dψ)',
                'I(θ, ψ) = ∫_0^{2π} saturate(a + b cosφ) dφ = 2(a·acos(−a/b) + √(b^2 − a^2)),   a = cosθ cosψ,  b = sinθ sinψ',
                'κ = |fwidth(N)| / |fwidth(P)|        L_o = (A/π) · (E_L · D(N·L, 1/κ) · V + E_amb) + L_spec',
            ],
            code: {
                title: 'HLSL · CS_BakePreintegratedLUT + PS_PreintegratedSkin', src: `
                RWTexture2D<float4> PreintLUT;      // x: N·L ∈ [−1,1]，y: κ ∈ [0, KappaMax] (1/mm)

                float RingIntegral(float a, float b)  // ∫0^{2π} saturate(a + b·cosφ) dφ, b ≥ 0
                {
                    if (a >= b)  return 2.0 * PI * a;
                    if (a <= -b) return 0.0;
                    return 2.0 * (a * acos(-a / b) + sqrt(b * b - a * a));
                }
                [numthreads(8, 8, 1)]
                void CS_BakePreintegratedLUT(uint3 id : SV_DispatchThreadID)
                {
                    float NoL = (id.x + 0.5) / LUT_W * 2.0 - 1.0, sinT = sqrt(1.0 - NoL * NoL);
                    float rc  = 1.0 / ((id.y + 0.5) / LUT_H * KappaMax);   // 曲率半径 mm
                    float psiMax = (RCut >= 2.0 * rc) ? PI : 2.0 * asin(RCut / (2.0 * rc));
                    float3 num = 0, den = 0;
                    for (int m = 0; m < N_PSI; m++) {                       // ψ：以着色点为极点的球面极角
                        float  psi = (m + 0.5) * psiMax / N_PSI;
                        float3 w   = BurleyProfile(2.0 * rc * sin(0.5 * psi), SurfaceAlbedo, DiffuseMFP / BurleyShapeS(SurfaceAlbedo)) * sin(psi);
                        num += w * RingIntegral(NoL * cos(psi), sinT * sin(psi));
                        den += w * 2.0 * PI;
                    }
                    PreintLUT[id.xy] = float4(num / den, 1);
                }
                float3 PS_PreintegratedSkin(float3 N, float3 P, float3 A, float vis, float3 spec)
                {
                    float  kappa = length(fwidth(N)) / length(fwidth(P)) / MMPerUnit;   // 1/mm
                    float  NoL   = dot(N, LightDir);
                    float3 D     = PreintLUT.SampleLevel(LinearClamp, float2(NoL * 0.5 + 0.5, kappa / KappaMax), 0).rgb;
                    return A * INV_PI * (LightIrradiance * D * vis + AmbientIrr * (0.6 + 0.4 * N.y)) + spec;
                }`,
            },
            previews: [
                {
                    id: 'lutImg', label: 'PreintLUT', sub: `x: N·L −1→1 · y: κ 0→${SS.KMAX} mm⁻¹（向上）`, size: [360, 270], smooth: true,
                    render: (c, S) => PL.draw.pixels(c, SS.LUT_W, SS.LUT_H, (x, y, out) => { const j = SS.LUT_H - 1 - y, i = (j * SS.LUT_W + x) * 3; out[0] = BX.toSrgb(S.lut[i]); out[1] = BX.toSrgb(S.lut[i + 1]); out[2] = BX.toSrgb(S.lut[i + 2]); return out; }),
                    probe: (x, y, S) => { const j = SS.LUT_H - 1 - y, i = (j * SS.LUT_W + x) * 3; return `N·L = ${(-1 + 2 * (x + 0.5) / SS.LUT_W).toFixed(2)} · κ = ${(SS.KMAX * (j + 0.5) / SS.LUT_H).toFixed(3)} 1/mm\nD = (${S.lut[i].toFixed(3)}, ${S.lut[i + 1].toFixed(3)}, ${S.lut[i + 2].toFixed(3)})`; },
                },
                {
                    id: 'lutRows', label: 'LUT 行：D(N·L)', sub: '实线 κ = 1/3.6（小球），虚线 κ = 1/7.2（大球），灰 = saturate(N·L)', size: [360, 200],
                    render: (c, S) => {
                        const D = [0, 0, 0], ser = [{ fn: (x) => Math.max(0, x), color: '#666', label: 'Lambert' }];
                        [[1 / 3.6, []], [1 / 7.2, [4, 3]]].forEach(([kap, dash]) => [0, 1, 2].forEach((k) => ser.push({ fn: (x) => SS.lutSample(S.lut, x, kap, D)[k], color: SS.CH[k], dash, samples: 128, label: dash.length ? undefined : `${SS.CHN[k]}` })));
                        PL.draw.plot(c, { x: [-1, 1], y: [0, 1], xLabel: 'N·L', series: ser });
                    },
                },
                { id: 'curv', label: '曲率 κ（屏幕导数）', sub: `viridis 0 → ${SS.KMAX} mm⁻¹`, render: (c, S) => PL.draw.field(c, S.kappa, SS.W, SS.H, { map: 'viridis', range: [0, SS.KMAX] }), probe: (x, y, S) => `κ = ${S.kappa[y * SS.W + x].toFixed(3)} 1/mm → r = ${(1 / Math.max(S.kappa[y * SS.W + x], 1e-6)).toFixed(1)} mm` },
                { id: 'preRender', label: 'SceneColor（预积分）', sub: '每像素一次查表 · 阴影仍是硬的', live: true, render: (c, S) => SS.drawBuf(c, S.imgPre) },
                { id: 'sphL2', label: 'Lambert', sub: '参照', render: (c, S) => { const A = S.med.ch.map((q) => q.A); BX.renderSphere(c, () => A.map((a) => a / Math.PI), { lights: BX.LIGHTS.studio }); } },
                { id: 'sphR12', label: '预积分 · r = 12 mm', sub: 'studio 三灯', render: (c, S) => preSphere(c, S, 12) },
                { id: 'sphR3', label: '预积分 · r = 3 mm', sub: '曲率越大越“透”', render: (c, S) => preSphere(c, S, 3) },
            ],
            notes: 'LUT 底部（κ → 0，平面）退化成 saturate(N·L)；越往上，交界线越软，N·L < 0 的一侧还有红光“绕过来”。曲率图里立方体的面是 0（平面 = 没有 SSS），棱上 κ 爆表——预积分在这里亮出一条边，是屏幕导数估曲率的经典伪影。对比顶部总览：预积分<strong>不会</strong>软化投射阴影的边缘（Penner 另用一张 penumbra LUT 处理）。',
            warn: '这张 LUT 是“材质资产”的一部分：剖面或介质一变就要重烘（本页标为 RESIDENT，换介质 = 换资产）。曲率要用平滑的几何法线（不是法线贴图后的），否则毛孔级细节会让 κ 噪声很大；生产中常把 κ 预计算进顶点色 / 贴图。',
        },
        // ───────────────────────────────────────────── 11
        {
            id: 'prod', title: '教学版 → 生产版：随机游走、透射与常见坑', tag: 'PROD',
            desc: '所有“剖面 + 平面假设”的方法在薄处、凹角、小于 ℓ<sub>d</sub> 的几何上都会出错。离线渲染器（Arnold、RenderMan、Cycles、Hyperion）因此转向<strong>体内随机游走</strong>（Chiang et al. 2016）：把闭合网格当作均匀体介质，光线折射进入后做体积路径追踪，自然处理厚度、透射、多次进出表面。美术仍然只给表面颜色 A 和散射距离 d，用 albedo 反演把它们映射到 σ<sub>t</sub>、单次散射反照率 α。下图就是这种“ground truth”：在半无限平板里做真实的随机游走（各向同性相函数、η 边界含全内反射、法向 searchlight 入射），统计出射半径的直方图，与偶极子、Burley 对比。',
            blocks: [
                { type: 'formula', text: 'α = 1 − exp(−5.09406 A + 2.61188 A^2 − 4.31805 A^3)        // Chiang 2016：表面颜色 A → 单次散射反照率\nσ_t = 1 / (d · s(A)),     s(A) = 1.9 − A + 3.5 (A − 0.8)^2       σ_s = α σ_t\nR_MC(r_k) = N_k / (N_walk · π (r_{k+1}^2 − r_k^2))        // 出射半径直方图 → 面密度' },
                {
                type: 'code', title: 'HLSL（伪代码）· 体内随机游走 SSS', src: `
                float3 AlbedoInversion(float3 A)  { return 1.0 - exp(A * (-5.09406 + A * (2.61188 - A * 4.31805))); }
                float3 SigmaT(float3 A, float3 d) { return 1.0 / (d * (1.9 - A + 3.5 * (A - 0.8) * (A - 0.8))); }

                // p、dir：刚折射进表面的点与方向。返回出射时的吞吐量，dir 改成体外的出射方向（辅助函数见附录）
                float3 RandomWalkSSS(float3 p, inout float3 dir, float3 A, float3 d, inout RNG rng)
                {
                    float3 sigma_t = SigmaT(A, d), alpha = AlbedoInversion(A), beta = 1.0;
                    for (int i = 0; i < MAX_WALK; i++) {
                        int    c   = PickChannel(beta, rng);                        // 按吞吐量选一个通道采样距离（hero channel）
                        float  t   = -log(1.0 - Rand(rng)) / sigma_t[c];
                        HitInfo h;
                        if (TraceClosedMesh(p, dir, t, h)) {                         // 先撞到边界：只走到 h.t
                            float3 trB = exp(-sigma_t * h.t);
                            beta *= trB / ChannelAvg(trB, beta);                     // 透过率 / P(走过 h.t)，单样本 MIS
                            p = h.P;
                            if (ExitBoundary(h, dir, rng)) return beta;              // 折射出去：交给表面 BSDF / 相机路径
                            continue;                                                // Fresnel 反射或全内反射：dir 已翻回体内
                        }
                        float3 trT = exp(-sigma_t * t);
                        beta *= alpha * sigma_t * trT / ChannelAvg(sigma_t * trT, beta);   // σs·T / 混合 pdf（光谱 MIS）
                        p   += dir * t;
                        dir  = SampleHG(dir, Anisotropy, Rand2(rng));               // 相函数（可各向异性），权重 p/pdf = 1
                    }
                    return 0;
                }`,
                },
                { type: 'previews', items: [
                { id: 'mcR', label: 'R 通道：MC 随机游走 vs 模型', sub: '柱 = MC 直方图 · 实线 = 偶极子 · 虚线 = Burley', size: [360, 200], needs: ['walk'], render: (c, S) => mcPlot(c, S, 0) },
                { id: 'mcB', label: 'B 通道：MC 随机游走 vs 模型', sub: '低 α′ 时单次散射占比大', size: [360, 200], needs: ['walk'], render: (c, S) => mcPlot(c, S, 2) },
                ] },
                {
                    type: 'html', html: `<table style="font-size:0.78rem;border-collapse:collapse;width:100%;margin:6px 0">
                    <tr style="color:#8ab"><td style="padding:3px 8px">教学版（本页）</td><td style="padding:3px 8px">生产版</td></tr>
                    <tr><td style="padding:3px 8px">单方向光 + 硬阴影射线</td><td style="padding:3px 8px">所有光源 + GI 写进同一张 diffuse 缓冲；阴影边由 SSS 自然软化，shadow map 不需要额外模糊</td></tr>
                    <tr><td style="padding:3px 8px">全分辨率、固定 N taps 可分离</td><td style="padding:3px 8px">半分辨率 / checkerboard；UE 的 Burley 用重要性采样的 2D 盘 + 每帧旋转 + TAA；HDRP 用蓝噪声盘采样</td></tr>
                    <tr><td style="padding:3px 8px">每物体一个剖面</td><td style="padding:3px 8px">模板 / G-buffer 里存 profile id，查 profile 表（UE Subsurface Profile、HDRP Diffusion Profile）</td></tr>
                    <tr><td style="padding:3px 8px">无透射</td><td style="padding:3px 8px">薄处透射：从 shadow map 取厚度 t，按剖面积分 T(t) 加背光项（Jimenez 2010 / UE、HDRP 的 thickness map）</td></tr>
                    <tr><td style="padding:3px 8px">离线 ground truth：平板随机游走</td><td style="padding:3px 8px">闭合网格内的随机游走 + albedo 反演（Chiang 2016），或 probe-ray + Burley 剖面（King 2013）</td></tr></table>`,
                },
                { type: 'note', html: 'MC 曲线在 r → 0 处远高于偶极子：那是单次散射（以及前几次散射），扩散近似没有这一项；Burley 拟合的是 MC，中心尖峰对得上。蓝通道（α′ 低）偏差最大。图例里的 A<sub>MC</sub> 是逃逸光子占比，可与偶极子闭式 A 对比。用同一个随机游走离线实测的能量加权误差（η = 1.3）：Skin1 红通道偶极子 ≈ 28%、Burley（ℓ<sub>d</sub> 拟合）≈ 11%、Burley（ℓ 拟合 + 偶极子 A）≈ 31%；η = 1 时 ℓ 拟合降到 ≈ 3%。拖 <code>η</code> 看三者如何一起变化。' },
                { type: 'warn', html: '<strong>常见坑</strong>：① 单位：mm 剖面配 cm / m 场景，SSS 要么看不见要么糊成一片；② 先乘 albedo 再模糊 → 贴图细节丢失、颜色被乘两遍；③ 镜面进了模糊 → 蜡像感；④ 没有 follow surface / profile id → 轮廓暗晕、地面颜色渗进皮肤；⑤ Fresnel 用错一侧（F<sub>dr</sub> 是体内一侧）；⑥ 核不归一化或截断太早 → 能量增减，皮肤整体变色；⑦ 预积分的曲率用了法线贴图后的法线 → κ 满屏噪点；⑧ 随机游走没做 albedo 反演 → 美术给的颜色和渲染结果对不上（多次散射会让颜色更饱和、更暗）。' },
            ],
        },
    ],

    graph: {
        intro: '<strong>RDG 风格</strong>的实时 SSS 帧结构：每个节点 = 一次 Dispatch / Draw（或 CPU 端构建）；pin = 资源名 · 格式，同名资源自动连线。屏幕空间路径（Lighting → Blur H → Blur V → Combine）与预积分路径（LUT 烘焙一次、Preintegrated 每帧查表）共用同一份 G-buffer；两条路径在引擎里二选一（或按材质选）。本页场景与相机静止，所以 G-buffer 只光栅化一次（RESIDENT）；运动场景中它和光照一样是 PER-FRAME。',
        types: {
            scalar: { color: '#e8c25a', label: 'scalar · cbuffer 常量' },
            gbuf: { color: '#4a8aff', label: 'gbuffer · 深度 / 法线 / albedo / 模板' },
            hdr: { color: '#e06c75', label: 'hdr · RGBA16F 线性光照' },
            kernel: { color: '#a45fa5', label: 'kernel · StructuredBuffer<float4>' },
            lut: { color: '#5ae87a', label: 'lut · RGBA8 预积分表' },
        },
        common: {
            title: 'Common.hlsl · 系列共享 + SSS 模块（cbuffer / 剖面函数）',
            blocks: [
                { label: '━ 系列共享 ━', code: BX.HLSL_COMMON },
                { label: '━ SSS cbuffer ━（绿 = 用户参数 / 红 = 系统注入）', code: SSS_HLSL_CB },
                { label: '━ 剖面函数 ━（Burley · Christensen & Burley 2015）', code: SSS_HLSL_PROFILE },
            ],
        },
        nodes: [
            {
                id: 'params', title: 'SSSParams', tag: 'CB', category: 'user', desc: 'A · ℓ · η · taps · follow',
                outputs: [{ name: 'SSSParams', format: 'cbuffer b0', type: 'scalar' }],
                detail: { desc: '材质 / 剖面参数（Subsurface Profile 资产）。本身不运行，经 cbuffer 传给烘焙与模糊。', code: { src: SSS_HLSL_CB }, preview: 'medTable' },
            },
            {
                id: 'light', title: 'Frame.LightCB', tag: 'CB', category: 'system', desc: 'L(t) · E_L · ambient',
                outputs: [{ name: 'LightCB', format: 'cbuffer b1', type: 'scalar' }],
                detail: { desc: '引擎每帧注入的光源方向（本页按时间匀速扫动方位角）与强度。依赖它的 pass 都是 PER-FRAME。', preview: null },
            },
            {
                id: 'kernel', title: 'CPU_BuildSSSKernel', tag: 'CPU', category: 'bake', lifetime: 'param', step: 'kernel', desc: '剖面 → N 个 (rgb 权重, 偏移 mm)',
                dispatch: { kind: 'CPU', name: 'BuildKernel', count: (S) => `${S.taps} taps × 3 通道`, note: '剖面 / taps 变化时重建，上传 StructuredBuffer' },
                inputs: [{ name: 'SSSParams', format: 'cbuffer', type: 'scalar' }],
                outputs: [{ name: 'SSSKernel', format: 'float4 × N', type: 'kernel' }],
                detail: { preview: 'k1taps' },
            },
            {
                id: 'lutbake', title: 'CS_BakePreintegratedLUT', tag: 'CS', category: 'bake', lifetime: 'resident', step: 'lut', desc: '球面积分 D(N·L, κ)',
                dispatch: { name: 'CS_BakePreintegratedLUT', threads: [8, 8, 1], groups: [SS.LUT_W / 8, SS.LUT_H / 8, 1], idRange: `ID.xy ∈ [0,${SS.LUT_W})×[0,${SS.LUT_H})`, note: '每 thread 一个 texel，内循环 96 个 ψ 样本' },
                inputs: [{ name: 'SSSParams', format: 'cbuffer', type: 'scalar' }],
                outputs: [{ name: 'PreintLUT', format: `RGBA8 ${SS.LUT_W}×${SS.LUT_H}`, type: 'lut' }],
                detail: { preview: 'lutImg' },
            },
            {
                id: 'gbuffer', title: 'PS_GBuffer', tag: 'PS', lifetime: 'resident', step: 'gbuf', desc: '深度 · 法线 · albedo · SSS 模板',
                dispatch: { kind: 'PS', name: 'BasePass', target: `GBuffer ${SS.W}×${SS.H}`, idRange: 'SV_Position.xy ∈ viewport', note: '本页：光线投射；SSS 材质写模板 / profile id' },
                outputs: [
                    { name: 'SceneDepth', format: 'R32F', type: 'gbuf' }, { name: 'GBufferNormal', format: 'RGB10A2', type: 'gbuf' },
                    { name: 'GBufferAlbedo', format: 'RGBA8', type: 'gbuf' }, { name: 'SSSMask', format: 'R8_UINT', type: 'gbuf' },
                ],
                detail: { preview: 'gbMask', code: { title: 'G-buffer 布局', src: 'SceneDepth    R32F     // 线性 view z\nGBufferNormal RGB10A2  // world normal\nGBufferAlbedo RGBA8    // A（SSS 材质 = 表面反照率）\nSSSMask       R8_UINT  // 0 = 非 SSS，否则 profile / object id' } },
            },
            {
                id: 'lighting', title: 'PS_DiffuseLighting', tag: 'PS', lifetime: 'frame', step: 'gbuf', desc: 'E = E_L·saturate(N·L)·V + E_amb',
                dispatch: { kind: 'PS', name: 'PS_DiffuseLighting', target: `MRT ${SS.W}×${SS.H}`, idRange: 'SV_Position.xy ∈ viewport', note: '漫反射辐照度与镜面分开输出' },
                inputs: [{ name: 'SceneDepth', format: 'R32F', type: 'gbuf' }, { name: 'GBufferNormal', format: 'RGB10A2', type: 'gbuf' }, { name: 'LightCB', format: 'cbuffer', type: 'scalar' }],
                outputs: [{ name: 'DiffuseIrradiance', format: 'RGBA16F', type: 'hdr' }, { name: 'SpecularLighting', format: 'RGBA16F', type: 'hdr' }],
                detail: { preview: 'gbE', code: 0 },
            },
            {
                id: 'blurH', title: 'PS_SSSBlur · H', tag: 'PS', lifetime: 'frame', step: 'ssss', desc: 'BlurDir = (1,0)',
                dispatch: { kind: 'PS', name: 'PS_SSSBlur', target: `SSSBlurH ${SS.W}×${SS.H}`, idRange: 'SV_Position.xy ∈ viewport（模板 = SSS）', note: (S) => `${S.taps} taps，偏移 mm → px 按深度换算` },
                inputs: [{ name: 'DiffuseIrradiance', format: 'RGBA16F', type: 'hdr' }, { name: 'SceneDepth', format: 'R32F', type: 'gbuf' }, { name: 'SSSMask', format: 'R8_UINT', type: 'gbuf' }, { name: 'SSSKernel', format: 'float4 × N', type: 'kernel' }],
                outputs: [{ name: 'SSSBlurH', format: 'RGBA16F', type: 'hdr' }],
                detail: { preview: 'ssH', code: 0, formula: 0 },
            },
            {
                id: 'blurV', title: 'PS_SSSBlur · V', tag: 'PS', lifetime: 'frame', step: 'ssss', desc: 'BlurDir = (0,1)',
                dispatch: { kind: 'PS', name: 'PS_SSSBlur', target: `SSSBlurHV ${SS.W}×${SS.H}`, idRange: 'SV_Position.xy ∈ viewport（模板 = SSS）', note: '同一 shader，换方向' },
                inputs: [{ name: 'SSSBlurH', format: 'RGBA16F', type: 'hdr' }, { name: 'SceneDepth', format: 'R32F', type: 'gbuf' }, { name: 'SSSMask', format: 'R8_UINT', type: 'gbuf' }, { name: 'SSSKernel', format: 'float4 × N', type: 'kernel' }],
                outputs: [{ name: 'SSSBlurHV', format: 'RGBA16F', type: 'hdr' }],
                detail: { preview: 'ssV', code: 0, formula: 1 },
            },
            {
                id: 'combine', title: 'PS_SSSCombine', tag: 'PS', category: 'output', lifetime: 'frame', step: 'ssss', desc: 'A/π · B_xy + spec',
                dispatch: { kind: 'PS', name: 'PS_SSSCombine', target: 'SceneColor（viewport）', idRange: 'SV_Position.xy ∈ viewport' },
                inputs: [{ name: 'SSSBlurHV', format: 'RGBA16F', type: 'hdr' }, { name: 'SpecularLighting', format: 'RGBA16F', type: 'hdr' }, { name: 'GBufferAlbedo', format: 'RGBA8', type: 'gbuf' }],
                outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'hdr' }],
                detail: { preview: 'ssFinal', code: 1, formula: 3 },
            },
            {
                id: 'preint', title: 'PS_PreintegratedSkin', tag: 'PS', category: 'output', lifetime: 'frame', step: 'lut', desc: 'LUT(N·L, κ)，κ = |fwidth N|/|fwidth P|',
                dispatch: { kind: 'PS', name: 'PS_PreintegratedSkin', target: 'SceneColor（另一条路径）', idRange: 'SV_Position.xy ∈ viewport', note: '前向 / 延迟均可，每像素一次查表' },
                inputs: [{ name: 'GBufferNormal', format: 'RGB10A2', type: 'gbuf' }, { name: 'SceneDepth', format: 'R32F', type: 'gbuf' }, { name: 'GBufferAlbedo', format: 'RGBA8', type: 'gbuf' }, { name: 'PreintLUT', format: 'RGBA8', type: 'lut' }, { name: 'LightCB', format: 'cbuffer', type: 'scalar' }],
                outputs: [{ name: 'SceneColorPreint', format: 'RGBA16F', type: 'hdr' }],
                detail: { preview: 'preRender', formula: 3 },
            },
        ],
    },

    references: [
        { title: 'A Practical Model for Subsurface Light Transport', author: 'Henrik Wann Jensen, Stephen R. Marschner, Marc Levoy, Pat Hanrahan', venue: 'SIGGRAPH', year: 2001, note: 'BSSRDF 可分离形式、偶极子 R_d(r)、F_dr 多项式、实测 σs′/σa 表（本页介质预设）' },
        { title: 'A Rapid Hierarchical Rendering Technique for Translucent Materials', author: 'Henrik Wann Jensen, Juan Buhler', venue: 'SIGGRAPH', year: 2002, note: '以 R_d 与漫射平均自由程 ℓd 为美术参数，数值反演 α′' },
        { title: 'Light Diffusion in Multi-Layered Translucent Materials', author: 'Craig Donner, Henrik Wann Jensen', venue: 'SIGGRAPH', year: 2005, note: 'Multipole：有限厚度与多层（皮肤表皮 / 真皮）' },
        { title: 'Advanced Techniques for Realistic Real-Time Skin Rendering', author: 'Eugene d\'Eon, David Luebke', venue: 'GPU Gems 3, Ch. 14', year: 2007, note: 'Sum-of-Gaussians 皮肤剖面、纹理空间递增模糊、pre/post-scatter texturing' },
        { title: 'A Quantized-Diffusion Model for Rendering Translucent Materials', author: 'Eugene d\'Eon, Geoffrey Irving', venue: 'SIGGRAPH', year: 2011, note: '基于 Fresnel 矩的改进边界条件 A = (1+3C₂)/(1−2C₁)' },
        { title: 'Screen-Space Perceptual Rendering of Human Skin', author: 'Jorge Jimenez, Veronica Sundstedt, Diego Gutierrez', venue: 'ACM Transactions on Applied Perception', year: 2009, note: '屏幕空间 SSS 的起点' },
        { title: 'Real-Time Realistic Skin Translucency', author: 'Jorge Jimenez, David Whelan, Veronica Sundstedt, Diego Gutierrez', venue: 'IEEE Computer Graphics and Applications', year: 2010, note: '用 shadow map 厚度近似薄处透射' },
        { title: 'Pre-Integrated Skin Shading', author: 'Eric Penner, George Borshukov', venue: 'GPU Pro 2', year: 2011, note: '(N·L, 曲率) 预积分 LUT、屏幕导数曲率、penumbra 软阴影' },
        { title: 'BSSRDF Importance Sampling', author: 'Alan King, Christopher Kulla, Alejandro Conty, Marcos Fajardo', venue: 'SIGGRAPH Talks', year: 2013, note: 'probe-ray 采样 x_i、三轴 MIS' },
        { title: 'An Approximate Reflectance Profile for Efficient Subsurface Scattering', author: 'Per H. Christensen', venue: 'SIGGRAPH Talks', year: 2015, url: 'https://doi.org/10.1145/2775280.2792555', note: 'R(r) = A·s·(e^{−sr}+e^{−sr/3})/(8πr)、s(A) 拟合与 CDF' },
        { title: 'Approximate Reflectance Profiles for Efficient Subsurface Scattering', author: 'Per H. Christensen, Brent Burley', venue: 'Pixar Technical Memo 15-04', year: 2015, note: '三种 s(A) 拟合（ℓ / 漫射入射 / ℓd）的完整版本' },
        { title: 'Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering', author: 'Brent Burley', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2015, note: 'Normalized diffusion 剖面的来源' },
        { title: 'Separable Subsurface Scattering', author: 'Jorge Jimenez, Károly Zsolnai, Adrian Jarabo, Christian Freude, Thomas Auzinger, Xian-Chun Wu, Javier von der Pahlen, Michael Wimmer, Diego Gutierrez', venue: 'Computer Graphics Forum', year: 2015, note: '两次 1D pass 的 rank-1 可分离核、follow-surface' },
        { title: 'Practical and Controllable Subsurface Scattering for Production Path Tracing', author: 'Matt Jen-Yuan Chiang, Peter Kutz, Brent Burley', venue: 'SIGGRAPH Talks', year: 2016, url: 'https://doi.org/10.1145/2897839.2927433', note: '随机游走 SSS 与 A → α 反演' },
        { title: 'Sampling Burley\'s Normalized Diffusion Profiles', author: 'Evgenii Golubev', venue: 'Blog post', url: 'https://zero-radiance.github.io/post/sampling-diffusion/', note: 'CDF 的解析反函数（HDRP 采用）' },
        { title: 'Physically Based Rendering: From Theory to Implementation (3rd ed.), Ch. 15', author: 'Matt Pharr, Wenzel Jakob, Greg Humphreys', venue: 'Morgan Kaufmann', year: 2016, note: 'Photon beam diffusion、Fresnel 矩、分离 BSSRDF 的归一化' },
    ],
});

// ── preview helpers that need the mounted state (hoisted function declarations) ──
function k2draw(canvas, S, key) {
    const k = S.k2, G = k.G, mx = [0, 1, 2].map((c) => { let m = 0; for (const v of k.tru[c]) m = Math.max(m, v); return m || 1; });
    PL.draw.pixels(canvas, G, G, (x, y, out) => {
        const i = y * G + x;
        for (let c = 0; c < 3; c++) out[c] = BX.clamp(1 + Math.log10(Math.max(k[key][c][i], 1e-30) / mx[c]) / 4, 0, 1);
        return out;
    });
}
function preSphere(canvas, S, radiusMM) {
    const A = S.med.ch.map((q) => q.A), D = [0, 0, 0], kap = 1 / radiusMM;
    BX.renderSphere(canvas, (wi) => { SS.lutSample(S.lut, wi[2], kap, D); const s = 1 / Math.max(Math.abs(wi[2]), 1e-4); return [A[0] / Math.PI * D[0] * s, A[1] / Math.PI * D[1] * s, A[2] / Math.PI * D[2] * s]; }, { lights: BX.LIGHTS.studio, twoSided: true });
}
function mcPlot(canvas, S, k) {
    const w = S.walk[k], c = S.med.ch[k], rP = S.rPlot;
    const ymax = Math.max(SS.dipole(0, c), SS.burley(rP / 200, c), ...w.R);
    PL.draw.plot(canvas, {
        x: [0, rP], y: [ymax * 1e-5, ymax * 2], logY: true, xLabel: 'r (mm)',
        series: [
            { data: w.R, x: w.x, bar: true, color: k === 0 ? 'rgba(224,108,117,0.45)' : 'rgba(106,176,224,0.45)', label: `MC  A=${w.A.toFixed(3)} · ${w.nw} walks` },
            { fn: (r) => SS.dipole(r, c), color: '#e0a060', label: `dipole  A=${c.Rd.toFixed(3)}` },
            { fn: (r) => SS.burley(r, c), color: '#e8e8e8', dash: [4, 3], label: `Burley  A=${c.A.toFixed(3)}` },
        ],
    });
}
// ── 3D：随机游走路径（局部系：表面 z = 0，介质 z < 0；mesh3d 是 y-up：(x, y, z) → (x, z, −y)）──
function sssWalk3d(canvas, S) {
    const W3 = S.walk3d, B = 1.2, D = -1.2, k = B / W3.X, P = (p) => [p[0] * k, p[2] * k, -p[1] * k];   // mm → 显示单位：盒子半宽 = 红通道 90 % 出射半径
    const lines = [], points = [];
    const sq = (y) => [[-B, y, -B], [B, y, -B], [B, y, B], [-B, y, B], [-B, y, -B]];
    lines.push({ pts: sq(0), color: 'rgba(200,205,215,0.6)', width: 1.2 });
    lines.push({ pts: sq(D), color: 'rgba(200,205,215,0.22)', width: 1, dash: [3, 4] });
    for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) lines.push({ pts: [[sx * B, 0, sz * B], [sx * B, D, sz * B]], color: 'rgba(200,205,215,0.22)', width: 1, dash: [3, 4] });
    [2, 1, 0].forEach((ch) => {
        const w = W3.ch[ch], col = SS.CH[ch], r = w.r50 * k;
        for (const pts of w.paths) lines.push({ pts: pts.map(P), color: col, width: ch === 0 ? 1.2 : 1.9 });
        for (const e of w.exits) if (Math.max(Math.abs(e[0]), Math.abs(e[1])) <= W3.X) points.push({ p: P([e[0], e[1], 0]), color: col, r: 1.5 });   // 盒外的 ≈ 10 % 不画
        lines.push({ pts: Array.from({ length: 73 }, (_, i) => { const a = 2 * Math.PI * i / 72; return [r * Math.cos(a), 0, r * Math.sin(a)]; }), color: col, width: 1.2, dash: [4, 3] });
    });
    lines.push({ pts: [[0, 1.0, 0], [0, 0, 0]], color: '#e8c25a', width: 2.4, arrow: true, labelAt: 'start', label: '入射点 xᵢ' });
    const mm = (v) => `${v.toFixed(v < 10 ? 1 : 0)} mm`;
    PL.draw.mesh3d(canvas, {
        meshes: [PL.mesh.plane({ c: [0, 0, 0], size: 2 * B, surface: false, wire: true, wireColor: 'rgba(160,170,180,0.18)' })],
        lines, points,
        labels: [
            { p: [B, 0.05, B], text: `表面 z = 0（半宽 ${mm(W3.X)}）`, color: '#aab' },
            { p: [B, D, B], text: '介质', color: '#889' },
        ],
        orbit: { yaw: 0.5, pitch: 0.5, dist: 6.6 },
        xray: 0.5,
        overlay: (ctx) => {
            ctx.font = '11px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
            ctx.fillStyle = '#99a'; ctx.fillText('出射半径中位数（虚线圈）', 8, 30);
            [0, 1, 2].forEach((ch) => { ctx.fillStyle = SS.CH[ch]; ctx.fillText(`${SS.CHN[ch]}  ${mm(W3.ch[ch].r50)}`, 8, 45 + 13 * ch); });
        },
    });
}
