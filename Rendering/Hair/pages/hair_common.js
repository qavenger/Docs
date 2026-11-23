// ════════════════════════════════════════════════════════════════════════════
// Hair series · shared helpers — prepended to every page at build time (run inside pages/):
//   node <skill>/scripts/build_doc.mjs hair_common.js hair_0N_xxx.lab.js -o hair_0N_xxx.html
//   node <skill>/scripts/build_doc.mjs hair_common.js index.lab.js -o ../index.html     (entry page lives one level up)
// Layout (GitHub Pages friendly): Hair/index.html is the only top-level file; everything else lives in Hair/pages/.
//
// Conventions shared by every page (formulas, HLSL and JS must all follow these):
//   • World: y up, x right, z toward the default viewer. Lengths in cm; fiber widths in cm too (80 µm = 0.008).
//   • Fiber frame (pbrt-v3 / Chiang et al. 2016): local x = tangent t (root → tip), y = normal n, z = b = t × n.
//     sinθ = ω·t is the longitudinal angle measured from the normal plane (θ ∈ [−π/2, π/2]);
//     φ = atan2(ω·b, ω·n) is the azimuth in the normal plane. HAIR.toFiber(w, f) gives [sinθ, ω·n, ω·b].
//   • wi points toward the light, wo toward the viewer. h ∈ [−1, 1] = offset across the fiber width, γ = asin(h).
//   • Scattering function S(wi, wo; h) = Σ_p M_p · A_p · N_p  (pbrt-v3 "fsum"; p = 0 R, 1 TT, 2 TRT, 3 residual).
//     Normalization: ∫_{S²} S dωi = 1 for σa = 0 (white furnace). A directional light of irradiance E gives
//     L_o = S · E — the projected fiber width is already folded into S, so there is no extra cos factor.
//   • Roughness: β_m (longitudinal) and β_n (azimuthal) ∈ [0, 1] as in pbrt-v3; cuticle tilt α in degrees (≈ 2°), α > 0
//     = physical tilt: R peaks at θh ≈ −α (toward the root), TT ≈ +α/2, TRT ≈ +2α (toward the tip) — see HAIR.bsdf.
//     Absorption σa per unit fiber diameter (RGB), e.g. from melanin concentrations.
//   • Linear RGB everywhere; HAIR.display() = exposure → ACES fit → sRGB, for display only.
// ════════════════════════════════════════════════════════════════════════════
const HAIR = (() => {
    const PI = Math.PI;
    const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
    const safeSqrt = (x) => Math.sqrt(Math.max(0, x));
    const safeAsin = (x) => Math.asin(clamp(x, -1, 1));

    // ── fiber frame ──
    /** Frame for a fiber with tangent t; n is chosen ⟂ t and facing `facing` (usually wo) so that φo = 0. */
    const fiberFrame = (t, facing) => {
        t = norm(t);
        let n = [facing[0] - dot(facing, t) * t[0], facing[1] - dot(facing, t) * t[1], facing[2] - dot(facing, t) * t[2]];
        if (Math.hypot(n[0], n[1], n[2]) < 1e-6) n = Math.abs(t[1]) < 0.9 ? cross(t, [0, 1, 0]) : cross(t, [1, 0, 0]);
        n = norm(n);
        return { t, n, b: cross(t, n) };
    };
    const toFiber = (w, f) => [dot(w, f.t), dot(w, f.n), dot(w, f.b)];   // pbrt-v3 hair-local coordinates
    const angles = (wl) => ({ sinT: wl[0], cosT: safeSqrt(1 - wl[0] * wl[0]), phi: Math.atan2(wl[2], wl[1]) });

    // ── dielectric Fresnel (unpolarized; eta = η_t / η_i) ──
    const fresnel = (cosI, eta) => {
        cosI = clamp(cosI, -1, 1);
        if (cosI < 0) { eta = 1 / eta; cosI = -cosI; }
        const sin2T = (1 - cosI * cosI) / (eta * eta);
        if (sin2T >= 1) return 1;
        const cosT = Math.sqrt(1 - sin2T);
        const rs = (eta * cosI - cosT) / (eta * cosI + cosT), rp = (cosI - eta * cosT) / (cosI + eta * cosT);
        return (rs * rs + rp * rp) / 2;
    };

    // ── d'Eon et al. 2011 longitudinal lobe, pbrt-v3 implementation ──
    const I0 = (x) => { let val = 0, x2i = 1, ifact = 1, i4 = 1; for (let i = 0; i < 25; i++) {   // pbrt-v3 uses 10 terms; 25 keeps it accurate up to the x = 12 switch
         if (i > 1) ifact *= i; val += x2i / (i4 * ifact * ifact); x2i *= x * x; i4 *= 4; } return val; };
    // Large-x asymptotic I0(x) ≈ e^x / √(2πx) · (1 + 1/(8x) + 9/(128x²)). pbrt-v3's LogI0 writes the correction as
    // 0.5·(1/(8x)) = 1/(16x), which makes I0 ≈ 0.6 % too small at x = 12 (M_p then integrates to ≈ 0.996 for v ≈ 0.05).
    const logI0 = (x) => (x > 12 ? x - 0.5 * Math.log(2 * PI * x) + Math.log1p(1 / (8 * x) + 9 / (128 * x * x)) : Math.log(I0(x)));
    /** M_p(θi, θo; v): ∫ M_p cosθi dθi = 1 */
    const Mp = (cosI, cosO, sinI, sinO, v) => {
        const a = cosI * cosO / v, b = sinI * sinO / v;
        return v <= 0.1 ? Math.exp(logI0(a) - b - 1 / v + 0.6931 + Math.log(1 / (2 * v))) : Math.exp(-b) * I0(a) / (Math.sinh(1 / v) * 2 * v);
    };
    // ── azimuthal lobe: trimmed logistic around the Bravais-geometry exit angle Φ(p) ──
    const logistic = (x, s) => { x = Math.abs(x); const e = Math.exp(-x / s); return e / (s * (1 + e) * (1 + e)); };
    const logisticCDF = (x, s) => 1 / (1 + Math.exp(-x / s));
    const trimmedLogistic = (x, s, a, b) => logistic(x, s) / (logisticCDF(b, s) - logisticCDF(a, s));
    const Phi = (p, gammaO, gammaT) => 2 * p * gammaT - 2 * gammaO + p * PI;
    const Np = (phi, p, s, gammaO, gammaT) => {
        let d = phi - Phi(p, gammaO, gammaT);
        while (d > PI) d -= 2 * PI;
        while (d < -PI) d += 2 * PI;
        return trimmedLogistic(d, s, -PI, PI);
    };
    /** A_p attenuation (p = 0..3, RGB): Fresnel at entry + absorption T along each internal path */
    const Ap = (cosThetaO, eta, h, T) => {
        const cosGammaO = safeSqrt(1 - h * h), f = fresnel(cosThetaO * cosGammaO, eta);
        const a0 = [f, f, f], a1 = T.map((t) => (1 - f) * (1 - f) * t), a2 = a1.map((a, c) => a * T[c] * f);
        const a3 = a2.map((a, c) => a * f * T[c] / (1 - T[c] * f));
        return [a0, a1, a2, a3];
    };
    const sigmaAFromConcentration = (ce, cp) => [0.419, 0.697, 1.37].map((e, i) => ce * e + cp * [0.187, 0.4, 1.05][i]);
    const sigmaAFromReflectance = (c, betaN) => c.map((x) => {
        const d = 5.969 - 0.215 * betaN + 2.532 * betaN ** 2 - 10.73 * betaN ** 3 + 5.574 * betaN ** 4 + 0.245 * betaN ** 5;
        return (Math.log(Math.max(1e-4, x)) / d) ** 2;
    });

    /** Chiang et al. 2016 / pbrt-v3 hair BSDF. o: { eta = 1.55, sigmaA = [r,g,b], betaM = 0.3, betaN = 0.3, alpha = 2 (deg) } */
    const bsdf = (o = {}) => {
        const eta = o.eta || 1.55, sigmaA = o.sigmaA || sigmaAFromConcentration(1.3, 0), betaM = o.betaM == null ? 0.3 : o.betaM, betaN = o.betaN == null ? 0.3 : o.betaN;
        const v0 = (0.726 * betaM + 0.812 * betaM ** 2 + 3.7 * betaM ** 20) ** 2, v = [v0, 0.25 * v0, 4 * v0, 4 * v0];
        const s = 0.626657069 * (0.265 * betaN + 1.194 * betaN ** 2 + 5.372 * betaN ** 22);
        // Cuticle scales point toward the tip, so with t = root → tip the R lobe must shift toward the ROOT (θh ≈ −α) and
        // TT / TRT toward the tip (Marschner 2003: α_R < 0). pbrt-v3's code applies +α along its curve tangent, which
        // under our root → tip tangent would put R toward the tip — so the user-facing α is negated here.
        const sin2k = [Math.sin(-(o.alpha == null ? 2 : o.alpha) * PI / 180)], cos2k = [safeSqrt(1 - sin2k[0] ** 2)];
        for (let i = 1; i < 3; i++) { sin2k[i] = 2 * cos2k[i - 1] * sin2k[i - 1]; cos2k[i] = cos2k[i - 1] ** 2 - sin2k[i - 1] ** 2; }
        /** per-lobe contributions [R, TT, TRT, residual] (RGB each) for local directions wi, wo and offset h */
        const lobes = (wi, wo, h) => {
            const sinO = wo[0], cosO = safeSqrt(1 - sinO * sinO), phiO = Math.atan2(wo[2], wo[1]);
            const sinI = wi[0], cosI = safeSqrt(1 - sinI * sinI), phiI = Math.atan2(wi[2], wi[1]);
            const sinTt = sinO / eta, cosTt = safeSqrt(1 - sinTt * sinTt);
            const etap = Math.sqrt(Math.max(1e-12, eta * eta - sinO * sinO)) / Math.max(1e-6, cosO);
            const sinGt = h / etap, cosGt = safeSqrt(1 - sinGt * sinGt), gammaT = safeAsin(sinGt), gammaO = safeAsin(h);
            const T = sigmaA.map((sa) => Math.exp(-sa * (2 * cosGt / Math.max(1e-6, cosTt))));
            const ap = Ap(cosO, eta, h, T), phi = phiI - phiO, out = [];
            for (let p = 0; p < 3; p++) {
                let sOp, cOp;
                if (p === 0) { sOp = sinO * cos2k[1] - cosO * sin2k[1]; cOp = cosO * cos2k[1] + sinO * sin2k[1]; }
                else if (p === 1) { sOp = sinO * cos2k[0] + cosO * sin2k[0]; cOp = cosO * cos2k[0] - sinO * sin2k[0]; }
                else { sOp = sinO * cos2k[2] + cosO * sin2k[2]; cOp = cosO * cos2k[2] - sinO * sin2k[2]; }
                const k = Mp(cosI, Math.abs(cOp), sinI, sOp, v[p]) * Np(phi, p, s, gammaO, gammaT);
                out.push(ap[p].map((a) => a * k));
            }
            const kr = Mp(cosI, cosO, sinI, sinO, v[3]) / (2 * PI);
            out.push(ap[3].map((a) => a * kr));
            return out;
        };
        const S = (wi, wo, h) => { const l = lobes(wi, wo, h); return [0, 1, 2].map((c) => l[0][c] + l[1][c] + l[2][c] + l[3][c]); };
        /** far-field: average over the fiber width (midpoint rule in h) */
        /** far-field: average over the fiber width with an nh-point midpoint rule in h. Converges slowly for narrow
         *  azimuthal lobes (β_n = 0.3: worst-case ≈ 28 % error at nh = 64, 14 % at 96) — fine for plots with nh ≥ 256;
         *  for images prefer one stochastic h per pixel per frame (S(wi, wo, 2u − 1)) with temporal accumulation. */
        const Sbar = (wi, wo, nh = 64) => { const r = [0, 0, 0]; for (let k = 0; k < nh; k++) { const h = -1 + (2 * k + 1) / nh, s3 = S(wi, wo, h); r[0] += s3[0] / nh; r[1] += s3[1] / nh; r[2] += s3[2] / nh; } return r; };
        return { eta, sigmaA, betaM, betaN, v, s, sin2k, cos2k, lobes, S, Sbar };
    };

    // ── Kajiya-Kay (1989) in world space: diffuse sin(T,L), specular (sin(T,H))^p ──
    const kajiyaKay = (T, L, V, p) => {
        const tl = dot(T, L), H = norm([L[0] + V[0], L[1] + V[1], L[2] + V[2]]), th = dot(T, H);
        return { diffuse: safeSqrt(1 - tl * tl), specular: Math.pow(safeSqrt(1 - th * th), p) };
    };

    // ── procedural groom: guides on a root patch, strands interpolated with clumping, wave, curl, frizz ──
    /** o: { seed, width, depth (root patch, cm), length, lengthVar, guidesX, guidesZ, perGuide, segs, wave, waveFreq,
     *  curl, curlFreq, clump, frizz, drape, spread, rootWidth, tipWidth }. Returns { guides, strands }, each strand
     *  { p: Float32Array 3(segs+1), w0, w1 (root/tip width), id, guide, rootUV } — deterministic for a given seed. */
    const makeGroom = (o = {}) => {
        const O = Object.assign({ seed: 7, width: 8, depth: 1.6, length: 14, lengthVar: 0.25, guidesX: 8, guidesZ: 2, perGuide: 90, segs: 24, wave: 0.5, waveFreq: 1.6, curl: 0.0, curlFreq: 6, clump: 0.6, frizz: 0.06, drape: 2.5, spread: 0.35, rootWidth: 0.008, tipWidth: 0.003 }, o);
        const rnd = PL.rng(O.seed), N = O.segs;
        const guideRoot = (gx, gz) => [(-0.5 + (gx + 0.5) / O.guidesX) * O.width, 0, (-0.5 + (gz + 0.5) / O.guidesZ) * O.depth];
        const guides = [];
        for (let gz = 0; gz < O.guidesZ; gz++) for (let gx = 0; gx < O.guidesX; gx++) {
            const r = guideRoot(gx, gz), ph = rnd() * 2 * PI, L = O.length * (1 - 0.15 * rnd()), p = new Float32Array(3 * (N + 1));
            for (let k = 0; k <= N; k++) {
                const t = k / N;
                p[3 * k] = r[0] * (1 + O.spread * t) + O.wave * Math.sin(2 * PI * O.waveFreq * t + ph) * t;
                p[3 * k + 1] = -L * t;
                p[3 * k + 2] = r[2] - O.drape * t * t;
            }
            guides.push({ root: r, p, gx, gz });
        }
        const guideAt = (u, w) => {   // bilinear blend of the 4 surrounding guides at root coordinates (u, w) ∈ [0,1]²
            const fx = clamp(u * O.guidesX - 0.5, 0, O.guidesX - 1), fz = clamp(w * O.guidesZ - 0.5, 0, O.guidesZ - 1);
            const x0 = Math.floor(fx), z0 = Math.floor(fz), x1 = Math.min(O.guidesX - 1, x0 + 1), z1 = Math.min(O.guidesZ - 1, z0 + 1), tx = fx - x0, tz = fz - z0;
            return [[x0, z0, (1 - tx) * (1 - tz)], [x1, z0, tx * (1 - tz)], [x0, z1, (1 - tx) * tz], [x1, z1, tx * tz]].map(([x, z, wt]) => ({ g: guides[z * O.guidesX + x], wt }));
        };
        const strands = [];
        const total = O.guidesX * O.guidesZ * O.perGuide;
        for (let i = 0; i < total; i++) {
            const u = rnd(), w = rnd(), root = [(-0.5 + u) * O.width, 0, (-0.5 + w) * O.depth], ws = guideAt(u, w);
            const nearest = ws.reduce((a, b) => (b.wt > a.wt ? b : a)).g;
            const len = 1 - O.lengthVar * rnd(), ph = rnd() * 2 * PI, fr = [rnd() * 2 * PI, rnd() * 2 * PI];
            const p = new Float32Array(3 * (N + 1));
            const g0 = ws.reduce((acc, { g, wt }) => [acc[0] + g.root[0] * wt, acc[1], acc[2] + g.root[2] * wt], [0, 0, 0]);
            for (let k = 0; k <= N; k++) {
                const t = k / N, tk = Math.min(N, Math.round(t * len * N));
                let base = [0, 0, 0];
                ws.forEach(({ g, wt }) => { base[0] += g.p[3 * tk] * wt; base[1] += g.p[3 * tk + 1] * wt; base[2] += g.p[3 * tk + 2] * wt; });
                // clumping: pull toward the nearest guide along the length
                const c = O.clump * Math.pow(t, 0.7);
                const gp = [nearest.p[3 * tk], nearest.p[3 * tk + 1], nearest.p[3 * tk + 2]];
                base = [base[0] * (1 - c) + gp[0] * c, base[1], base[2] * (1 - c) + gp[2] * c];
                const off = [(root[0] - g0[0]) * (1 - c), 0, (root[2] - g0[2]) * (1 - c)];
                const cr = O.curl * Math.sqrt(t), ca = 2 * PI * O.curlFreq * t * len + ph;
                p[3 * k] = base[0] + off[0] + cr * Math.cos(ca) + O.frizz * t * Math.sin(17 * t + fr[0]);
                p[3 * k + 1] = base[1];
                p[3 * k + 2] = base[2] + off[2] + cr * Math.sin(ca) + O.frizz * t * Math.sin(13 * t + fr[1]);
            }
            strands.push({ p, w0: O.rootWidth, w1: O.tipWidth, id: i, guide: guides.indexOf(nearest), rootUV: [u, w] });
        }
        return { guides, strands, opts: O };
    };

    // ── orthographic camera + strand rasterizer with analytic sub-pixel coverage ──
    /** view along -forward; yaw around y, pitch around x (radians); size = world units across the image height */
    const ortho = (o = {}) => {
        const yaw = o.yaw || 0, pitch = o.pitch || 0, center = o.center || [0, -7, 0], size = o.size || 18;
        const fwd = norm([-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)]);
        const right = norm(cross(fwd, [0, 1, 0])), up = cross(right, fwd);
        return { center, size, fwd, right, up, toViewer: [-fwd[0], -fwd[1], -fwd[2]] };
    };
    const project = (cam, p, W, H) => {
        const q = [p[0] - cam.center[0], p[1] - cam.center[1], p[2] - cam.center[2]], k = H / cam.size;
        return [W / 2 + dot(q, cam.right) * k - 0.5, H / 2 - dot(q, cam.up) * k - 0.5, dot(q, cam.fwd)];
    };
    /** Rasterize strands. Returns per-pixel buffers: alpha (order-independent 1 − Π(1 − c)), cov (summed coverage),
     *  depth (nearest, Infinity = empty), id (nearest strand, −1), v (root→tip param), h (signed offset across the
     *  nearest strand ∈ [−1,1]), tan (nearest world tangent, xyz). o: { widthScale = 1, minWidthPx = 0, filter(strand) } */
    const raster = (strands, cam, W, H, o = {}) => {
        const n = W * H, ws = o.widthScale || 1, k = H / cam.size;
        const alpha = new Float32Array(n), cov = new Float32Array(n), depth = new Float32Array(n).fill(Infinity), id = new Int32Array(n).fill(-1);
        const vb = new Float32Array(n), hb = new Float32Array(n), tan = new Float32Array(3 * n);
        const last = new Int32Array(n).fill(-1), before = new Float32Array(n), cmax = new Float32Array(n);
        for (const s of strands) {
            if (o.filter && !o.filter(s)) continue;
            const P = s.p, segs = P.length / 3 - 1;
            let a = project(cam, [P[0], P[1], P[2]], W, H);
            for (let j = 0; j < segs; j++) {
                const b = project(cam, [P[3 * j + 3], P[3 * j + 4], P[3 * j + 5]], W, H);
                const tm = (j + 0.5) / segs, halfW = Math.max((o.minWidthPx || 0) / 2, (s.w0 + (s.w1 - s.w0) * tm) * ws * k / 2);
                const t3 = norm([P[3 * j + 3] - P[3 * j], P[3 * j + 4] - P[3 * j + 1], P[3 * j + 5] - P[3 * j + 2]]);
                const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-9, r = halfW + 1;
                const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0]) - r)), x1 = Math.min(W - 1, Math.ceil(Math.max(a[0], b[0]) + r));
                const y0 = Math.max(0, Math.floor(Math.min(a[1], b[1]) - r)), y1 = Math.min(H - 1, Math.ceil(Math.max(a[1], b[1]) + r));
                for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
                    const u = clamp(((x - a[0]) * dx + (y - a[1]) * dy) / L2, 0, 1), px = a[0] + u * dx - x, py = a[1] + u * dy - y;
                    const d = Math.hypot(px, py);
                    const c = halfW >= 0.5 ? clamp(halfW + 0.5 - d, 0, 1) : 2 * halfW * Math.max(0, 1 - d);
                    if (c <= 0) continue;
                    const i = y * W + x;
                    if (last[i] !== s.id) { last[i] = s.id; before[i] = alpha[i]; cmax[i] = 0; }
                    if (c > cmax[i]) { cov[i] += c - cmax[i]; cmax[i] = c; alpha[i] = 1 - (1 - before[i]) * (1 - c); }
                    const z = a[2] + u * (b[2] - a[2]);
                    // nearest-strand buffers: the strand passes within its half-width (or half a pixel, for sub-pixel
                    // strands) of the pixel centre — a fixed coverage threshold would drop every sub-pixel strand
                    if (d <= Math.max(halfW, 0.5) && z < depth[i]) {
                        depth[i] = z; id[i] = s.id; vb[i] = (j + u) / segs;
                        const side = Math.sign(px * dy - py * dx) || 1;
                        hb[i] = clamp(side * d / Math.max(halfW, 0.5), -1, 1);
                        tan[3 * i] = t3[0]; tan[3 * i + 1] = t3[1]; tan[3 * i + 2] = t3[2];
                    }
                }
                a = b;
            }
        }
        return { W, H, alpha, cov, depth, id, v: vb, h: hb, tan };
    };

    // ── display / color ──
    const aces = (x) => clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1);
    const toSrgb = (x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
    const display = (rgb, exposure = 1) => rgb.map((c) => toSrgb(aces(Math.max(0, c) * exposure)));
    const hex2lin = (hex) => [1, 3, 5].map((i) => { const c = parseInt(hex.slice(i, i + 2), 16) / 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); });
    const lum = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

    // ── series navigation (pages/ layout) ──
    const MODULES = [
        { id: 'shading', no: '01', file: 'hair_01_shading.html', title: '毛发着色：Kajiya-Kay → Marschner → d\'Eon / Chiang', short: '着色模型' },
        { id: 'groom', no: '02', file: 'hair_02_groom.html', title: 'Groom：guide → 插值 → 光栅化与覆盖 → LOD', short: 'Groom 发丝' },
        { id: 'lighting', no: '03', file: 'hair_03_lighting.html', title: '自阴影与多次散射：Deep Opacity → Dual Scattering', short: '自阴影与多次散射' },
        { id: 'cards', no: '04', file: 'hair_04_cards.html', title: '插片 Hair Cards：烘焙 → 着色 → 透明 → LOD', short: '插片' },
    ];
    const PAGES_DIR = 'pages/';
    const href = (m, fromRoot) => (fromRoot ? PAGES_DIR : '') + m.file;
    const nav = (id) => {
        const i = MODULES.findIndex((m) => m.id === id), p = MODULES[i - 1], q = MODULES[i + 1];
        return `<p style="margin-top:8px;font-size:0.74rem;color:#789">毛发渲染系列 · <a href="../index.html">索引</a>` +
            (p ? ` · ← <a href="${p.file}">${p.no} ${p.short}</a>` : '') + (q ? ` · <a href="${q.file}">${q.no} ${q.short}</a> →` : '') + '</p>';
    };

    const HLSL_COMMON = `
        // 毛发系列 · Common.hlsl —— T = 发丝切线（根 → 梢）；sinθ = dot(ω, T)，θ 从法平面量起
        static const float PI = 3.14159265;
        float Pow2(float x) { return x * x; }
        float Pow5(float x) { float x2 = x * x; return x2 * x2 * x; }
        float SafeSqrt(float x) { return sqrt(max(0.0, x)); }

        // ω 在发丝法平面内的投影夹角：cos(φi − φo)
        float CosPhi(float3 L, float3 V, float3 T) {
            float3 Lp = L - dot(L, T) * T, Vp = V - dot(V, T) * T;
            return dot(Lp, Vp) * rsqrt(max(dot(Lp, Lp) * dot(Vp, Vp), 1e-8));
        }
        // 纵向高斯（Marschner / UE 风格的 M_p）
        float Hair_G(float B, float Theta) { return exp(-0.5 * Theta * Theta / (B * B)) / (sqrt(2.0 * PI) * B); }
        // 发丝角质层的 Fresnel（η ≈ 1.55）
        float Hair_F(float CosTheta) { const float n = 1.55; float F0 = ((1 - n) / (1 + n)) * ((1 - n) / (1 + n)); return F0 + (1 - F0) * Pow5(1 - CosTheta); }`;

    // Full HLSL of the series' reference hair BSDF (Chiang 2016 / pbrt-v3, with this series' fixes) + sampling helpers.
    // Every hair page puts it into graph.common, so any call to HairBSDF_Chiang / Mp / Np / Ap / Hash … links here.
    const HLSL_BSDF = `
        // ── 随机数 ──
        uint PcgHash(uint v) {
            uint s = v * 747796405u + 2891336453u;
            uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
            return (w >> 22u) ^ w;
        }
        // 每像素、每帧一个 [0,1) 随机数
        float Hash(uint2 pixel, uint frame) { return PcgHash(pixel.x + PcgHash(pixel.y + PcgHash(frame))) * (1.0 / 4294967296.0); }
        // Jimenez 2014：屏幕空间交错梯度噪声，[0,1)，常配合 TAA 做逐帧抖动
        float InterleavedGradientNoise(float2 pixel) { return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715)))); }
        // 逐帧变化的版本（UE 的做法：每帧把像素坐标平移一个固定偏移，64 帧一循环）
        float InterleavedGradientNoise(float2 pixel, uint frame) { return InterleavedGradientNoise(pixel + 5.588238 * float(frame % 64)); }

        // ── 纤维坐标系：x = 切线 T（根 → 梢），y = 法平面内朝向观察者，z = T × y（φo = 0）──
        float3 ToFiberFrame(float3 w, float3 T, float3 V) {
            float3 n = V - dot(V, T) * T;
            n = dot(n, n) > 1e-12 ? normalize(n) : normalize(cross(T, abs(T.y) < 0.9 ? float3(0, 1, 0) : float3(1, 0, 0)));
            float3 b = cross(T, n);
            return float3(dot(w, T), dot(w, n), dot(w, b));
        }

        // ── 纵向 M_p（d'Eon 2011）：∫ M_p cosθi dθi = 1；log I0 的大参数修正项是 1/(8x) ──
        float LogI0(float x) {
            if (x > 12) return x - 0.5 * log(2 * PI * x) + log(1 + 1 / (8 * x));
            float val = 0, x2i = 1, ifact = 1, i4 = 1;
            [unroll] for (int i = 0; i < 16; ++i) { if (i > 1) ifact *= i; val += x2i / (i4 * ifact * ifact); x2i *= x * x; i4 *= 4; }
            return log(val);
        }
        float Mp(float cosI, float cosO, float sinI, float sinO, float v) {
            float a = cosI * cosO / v, b = sinI * sinO / v;
            return (v <= 0.1) ? exp(LogI0(a) - b - 1 / v + 0.6931 + log(1 / (2 * v)))
                              : exp(-b) * exp(LogI0(a)) / (sinh(1 / v) * 2 * v);
        }

        // ── 方位 N_p（Chiang 2016）：Φ(p, h) 周围的截断 logistic ──
        float Logistic(float x, float s) { x = abs(x); float e = exp(-x / s); return e / (s * (1 + e) * (1 + e)); }
        float LogisticCDF(float x, float s) { return 1 / (1 + exp(-x / s)); }
        float Np(float phi, int p, float s, float gammaO, float gammaT) {
            float dphi = phi - (2 * p * gammaT - 2 * gammaO + p * PI);
            dphi = dphi - 2 * PI * floor((dphi + PI) / (2 * PI));   // 折回 [−π, π)
            return Logistic(dphi, s) / (LogisticCDF(PI, s) - LogisticCDF(-PI, s));
        }

        // ── 衰减 A_p：入射 Fresnel × 内部透射率 T；p ≥ 3 用几何级数合并 ──
        float FrDielectric(float cosI, float eta) {   // 非偏振，eta = η_t / η_i
            cosI = clamp(cosI, -1.0, 1.0);
            if (cosI < 0) { eta = 1 / eta; cosI = -cosI; }
            float sin2T = (1 - cosI * cosI) / (eta * eta);
            if (sin2T >= 1) return 1;
            float cosT = sqrt(1 - sin2T);
            float rs = (eta * cosI - cosT) / (eta * cosI + cosT), rp = (cosI - eta * cosT) / (cosI + eta * cosT);
            return 0.5 * (rs * rs + rp * rp);
        }
        void Ap(float cosThetaO, float eta, float h, float3 T, out float3 ap[4]) {
            float f = FrDielectric(cosThetaO * SafeSqrt(1 - h * h), eta);
            ap[0] = f;
            ap[1] = Pow2(1 - f) * T;
            ap[2] = ap[1] * T * f;
            ap[3] = ap[2] * f * T / (1 - T * f);
        }

        // ── Chiang 2016 / pbrt-v3 毛发 BSDF（R、TT、TRT + 剩余项）──
        struct HairParams {
            float  eta;            // 1.55
            float3 sigmaA;         // 两种黑色素浓度 → σa
            float  v[4];           // 纵向方差 v0, v0/4, 4·v0, 4·v0（βm → v0）
            float  s;              // 方位 logistic 宽度（βn → s）
            float  sin2kAlpha[3];  // sin(2^k·α′)，本系列 α′ = −α（切线 根 → 梢）
            float  cos2kAlpha[3];
        };
        // 两种黑色素浓度 → σa（Chiang 2016，同 HAIR.sigmaAFromConcentration）
        float3 SigmaAFromConcentration(float eumelanin, float pheomelanin) { return eumelanin * float3(0.419, 0.697, 1.37) + pheomelanin * float3(0.187, 0.4, 1.05); }
        // 由目标发色反推吸收系数 σa（Chiang 2016 的拟合，同 HAIR.sigmaAFromReflectance）
        float3 SigmaAFromReflectance(float3 c, float betaN) {
            float b = betaN, d = 5.969 - 0.215 * b + 2.532 * b * b - 10.73 * pow(b, 3) + 5.574 * pow(b, 4) + 0.245 * pow(b, 5);
            float3 l = log(max(c, 1e-4)) / d;
            return l * l;
        }
        // 美术参数 → HairParams（同 HAIR.bsdf：v = v0·(1, ¼, 4, 4)，s = √(π/8)·(…)，α′ = −α）；每个材质在 CPU 上算一次
        HairParams MakeHairParams(float3 sigmaA, float betaM, float betaN, float alphaDeg) {
            HairParams P;
            P.eta = 1.55;
            P.sigmaA = sigmaA;
            float v0 = Pow2(0.726 * betaM + 0.812 * betaM * betaM + 3.7 * pow(betaM, 20));
            P.v[0] = v0; P.v[1] = 0.25 * v0; P.v[2] = 4 * v0; P.v[3] = 4 * v0;
            P.s = 0.626657069 * (0.265 * betaN + 1.194 * betaN * betaN + 5.372 * pow(betaN, 22));
            P.sin2kAlpha[0] = sin(-alphaDeg * PI / 180);          // 切线 根 → 梢：R 偏向发根
            P.cos2kAlpha[0] = SafeSqrt(1 - Pow2(P.sin2kAlpha[0]));
            [unroll] for (int k = 1; k < 3; ++k) {                 // 倍角递推：sin 2x = 2 sin x cos x，cos 2x = cos²x − sin²x
                P.sin2kAlpha[k] = 2 * P.cos2kAlpha[k - 1] * P.sin2kAlpha[k - 1];
                P.cos2kAlpha[k] = Pow2(P.cos2kAlpha[k - 1]) - Pow2(P.sin2kAlpha[k - 1]);
            }
            return P;
        }
        // wi、wo：纤维局部方向（ToFiberFrame）；h ∈ [−1, 1]：截面偏移（远场时每帧随机取）
        float3 HairBSDF_Chiang(float3 wi, float3 wo, float h, HairParams P) {
            float sinO = wo.x, cosO = SafeSqrt(1 - sinO * sinO), phiO = atan2(wo.z, wo.y);
            float sinI = wi.x, cosI = SafeSqrt(1 - sinI * sinI), phiI = atan2(wi.z, wi.y);
            float sinTt = sinO / P.eta, cosTt = SafeSqrt(1 - sinTt * sinTt);
            float etap = sqrt(max(1e-12, P.eta * P.eta - sinO * sinO)) / max(1e-6, cosO);   // Bravais η′
            float sinGt = h / etap, cosGt = SafeSqrt(1 - sinGt * sinGt);
            float gammaT = asin(clamp(sinGt, -1.0, 1.0)), gammaO = asin(clamp(h, -1.0, 1.0));
            float3 T = exp(-P.sigmaA * (2 * cosGt / max(1e-6, cosTt)));
            float3 ap[4];
            Ap(cosO, P.eta, h, T, ap);
            float phi = phiI - phiO;
            float3 f = 0;
            [unroll] for (int p = 0; p < 3; ++p) {
                float sinOp, cosOp;   // 角质层倾角：θo 平移 −2α′（R）、+α′（TT）、+4α′（TRT）
                if (p == 0)      { sinOp = sinO * P.cos2kAlpha[1] - cosO * P.sin2kAlpha[1]; cosOp = cosO * P.cos2kAlpha[1] + sinO * P.sin2kAlpha[1]; }
                else if (p == 1) { sinOp = sinO * P.cos2kAlpha[0] + cosO * P.sin2kAlpha[0]; cosOp = cosO * P.cos2kAlpha[0] - sinO * P.sin2kAlpha[0]; }
                else             { sinOp = sinO * P.cos2kAlpha[2] + cosO * P.sin2kAlpha[2]; cosOp = cosO * P.cos2kAlpha[2] - sinO * P.sin2kAlpha[2]; }
                f += Mp(cosI, abs(cosOp), sinI, sinOp, P.v[p]) * ap[p] * Np(phi, p, P.s, gammaO, gammaT);
            }
            f += Mp(cosI, cosO, sinI, sinO, P.v[3]) * ap[3] / (2 * PI);   // p ≥ 3：方位均匀
            return f;
        }
        // 世界空间入口：L、V = 指向光源 / 观察者的单位向量，T = 发丝切线（根 → 梢）
        float3 HairBSDF_World(float3 L, float3 V, float3 T, float h, HairParams P) { return HairBSDF_Chiang(ToFiberFrame(L, T, V), ToFiberFrame(V, T, V), h, P); }`;

    return {
        PI, clamp, dot, cross, norm, fiberFrame, toFiber, angles, fresnel, I0, logI0, Mp, logistic, logisticCDF, trimmedLogistic, Phi, Np, Ap,
        sigmaAFromConcentration, sigmaAFromReflectance, bsdf, kajiyaKay, makeGroom, ortho, project, raster,
        aces, toSrgb, display, hex2lin, lum, MODULES, PAGES_DIR, href, nav, HLSL_COMMON, HLSL_BSDF,
    };
})();
