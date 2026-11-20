// ════════════════════════════════════════════════════════════════════════════
// BxDF 02 · 微表面镜面：Cook-Torrance · GGX
// Build:    node <skill>/scripts/build_doc.mjs bxdf_common.js bxdf_02_microfacet.lab.js -o bxdf_02_microfacet.html
// Validate: node <skill>/scripts/validate_doc.mjs bxdf_02_microfacet.html
// Conventions: see bxdf_common.js header (n = +Z, wi → light, wo → viewer, α = r², linear RGB).
// Module-local helpers live in MF; nothing from BX is redefined.
// ════════════════════════════════════════════════════════════════════════════

const MF = (() => {
    const { PI, dot, norm, clamp } = BX;
    const lerp = (a, b, t) => a + (b - a) * t;

    // ── isotropic NDFs, c = cosθh ──
    const dGGX = (c, a) => { if (c <= 0) return 0; const a2 = a * a, d = c * c * (a2 - 1) + 1; return a2 / (PI * d * d); };
    const dBeck = (c, a) => { if (c <= 0) return 0; const c2 = c * c, t2 = (1 - c2) / c2; return Math.exp(-t2 / (a * a)) / (PI * a * a * c2 * c2); };
    const bpExp = (a) => Math.max(0, 2 / (a * a) - 2);                         // Walter 2007: Blinn-Phong n ↔ Beckmann α
    const dBP = (c, n) => (c <= 0 ? 0 : (n + 2) / (2 * PI) * Math.pow(c, n));
    // ── Smith Λ (c = cosθ of the direction) ──
    const lamGGX = (c, a) => { const c2 = c * c; if (c2 < 1e-14) return 1e7; return (Math.sqrt(1 + a * a * (1 - c2) / c2) - 1) / 2; };
    const lamBeck = (c, a) => {          // Walter 2007 rational fit of the erf form (pbrt uses the same), m = 1/(α tanθ)
        const c2 = c * c; if (c2 >= 1) return 0; if (c2 < 1e-14) return 1e7;
        const m = c / (a * Math.sqrt(1 - c2));
        return m >= 1.6 ? 0 : (1 - 1.259 * m + 0.396 * m * m) / (3.535 * m + 2.181 * m * m);
    };
    const anisoAlpha = (r, k) => { const asp = Math.sqrt(1 - 0.9 * k), a = r * r; return [clamp(a / asp, 1e-4, 1), clamp(a * asp, 1e-4, 1)]; };   // Burley 2012
    // ── empirical lobes (specular only, ks = 1): normalized Phong around r = reflect(wi), normalized Blinn-Phong ──
    const phong = (wi, wo, np) => (wo[2] > 0 && wi[2] > 0 ? (np + 2) / (2 * PI) * Math.pow(Math.max(0, -wi[0] * wo[0] - wi[1] * wo[1] + wi[2] * wo[2]), np) : 0);
    const blinn = (wi, wo, n) => (wo[2] > 0 && wi[2] > 0 ? (n + 8) / (8 * PI) * Math.pow(Math.max(0, BX.half(wi, wo)[2]), n) : 0);

    // ── Fresnel table: 257 RGB entries over cosθ ∈ [0,1] (exact conductor Fresnel is too slow per sample) ──
    const FT = 256;
    const fTable = (fn, out) => { for (let i = 0; i <= FT; i++) { const v = fn(i / FT); out[3 * i] = v[0]; out[3 * i + 1] = v[1]; out[3 * i + 2] = v[2]; } return out; };
    const fLook = (tab, c) => {
        const x = clamp(c, 0, 1) * FT, i = Math.min(FT - 1, Math.floor(x)), t = x - i, j = 3 * i;
        return [tab[j] + (tab[j + 3] - tab[j]) * t, tab[j + 1] + (tab[j + 4] - tab[j + 1]) * t, tab[j + 2] + (tab[j + 5] - tab[j + 2]) * t];
    };

    // ── Cook-Torrance: D·G2·F / (4 cosθi cosθo), local frame, returns RGB ──
    const ct = (wi, wo, ax, ay, tab) => {
        // exactly-horizontal directions hit BX.ggxLambda's cos² < 1e-14 clamp (Λ := 1e7, far below the true ∝ 1/cosθ) → bogus spike; treat as 0
        if (wi[2] <= 1e-6 || wo[2] <= 1e-6) return [0, 0, 0];
        const h = BX.half(wi, wo), k = BX.ggxD(h, ax, ay) * BX.smithG2(wo, wi, ax, ay) / (4 * wi[2] * wo[2]);
        const F = fLook(tab, dot(wi, h));
        return [F[0] * k, F[1] * k, F[2] * k];
    };

    // ── samplers: return { h, wi, oh = wo·h, w = f·cosθi/pdf with F = 1 } ──
    const refl = (wo, h) => { const d = 2 * dot(wo, h); return [d * h[0] - wo[0], d * h[1] - wo[1], d * h[2] - wo[2]]; };
    const sampNDF = (wo, a, u1, u2) => {          // h ~ D(h)cosθh
        const h = BX.sampleNDF(a, u1, u2), oh = dot(wo, h);
        if (oh <= 0) return { h, wi: null, oh, w: 0 };                       // back-facing microfacet: wasted
        const wi = refl(wo, h);
        if (wi[2] <= 0) return { h, wi, oh, w: 0 };                          // below the horizon: wasted
        return { h, wi, oh, w: BX.smithG2(wo, wi, a) * oh / (wo[2] * h[2]) };
    };
    const sampVNDF = (wo, a, u1, u2) => {         // h ~ G1(wo)⟨wo·h⟩D(h)/cosθo  (Heitz 2018)
        const h = BX.sampleVNDF(wo, a, a, u1, u2), oh = dot(wo, h), wi = refl(wo, h);
        if (wi[2] <= 0) return { h, wi, oh, w: 0 };
        return { h, wi, oh, w: BX.smithG2(wo, wi, a) / BX.smithG1(wo, a) };
    };
    const woOf = (mu) => [Math.sqrt(Math.max(0, 1 - mu * mu)), 0, mu];
    /** directional albedo E(μ) with F = 1 (tab = null) or RGB with a Fresnel table — VNDF + Hammersley through BX.albedoMC */
    const albedo = (mu, a, n, tab) => {
        const e = BX.albedoMC((wo, u1, u2) => {
            const s = sampVNDF(wo, a, u1, u2);
            if (!s.w) return null;
            if (!tab) return { wi: s.wi, weight: s.w };
            const F = fLook(tab, s.oh); return { wi: s.wi, weight: [F[0] * s.w, F[1] * s.w, F[2] * s.w] };
        }, woOf(mu), n);
        return tab && !Array.isArray(e) ? [e, e, e] : e;
    };

    // ── LUT lookups (texel centers, clamp to edge) ──
    const lut2 = (T, N, x, y) => {
        const fx = clamp(x * N - 0.5, 0, N - 1), fy = clamp(y * N - 0.5, 0, N - 1);
        const x0 = Math.min(N - 2, Math.floor(fx)), y0 = Math.min(N - 2, Math.floor(fy)), tx = fx - x0, ty = fy - y0, i = y0 * N + x0;
        return lerp(lerp(T[i], T[i + 1], tx), lerp(T[i + N], T[i + N + 1], tx), ty);
    };
    const lut1 = (T, N, x, stride = 1, c = 0) => {
        const f = clamp(x * N - 0.5, 0, N - 1), i = Math.min(N - 2, Math.floor(f)), t = f - i;
        return lerp(T[i * stride + c], T[(i + 1) * stride + c], t);
    };
    const lut1rgb = (T, N, x) => [lut1(T, N, x, 3, 0), lut1(T, N, x, 3, 1), lut1(T, N, x, 3, 2)];

    // ── orthonormal basis around n (Duff et al. 2017) → [t, b] ──
    const basis = (n) => {
        const s = n[2] >= 0 ? 1 : -1, a = -1 / (s + n[2]), b = n[0] * n[1] * a;
        return [[1 + s * n[0] * n[0] * a, s * b, -s * n[0]], [b, s + n[1] * n[1] * a, -n[1]]];
    };

    // ── procedural environment in view space (x right, y up, z toward the viewer): sky + two softboxes ──
    const KEY = norm([-0.45, 0.5, 0.74]), RIM = norm([0.8, 0.28, -0.53]);
    const envSky = (d) => {          // smooth sky / ground gradient only (Step 08 noise demo: isolates BRDF-sampling variance)
        const y = d[1];
        if (y >= 0) { const t = Math.sqrt(y); return [0.5 - 0.36 * t, 0.53 - 0.3 * t, 0.58 - 0.1 * t]; }
        const t = Math.min(1, -y * 3); return [0.09 - 0.05 * t, 0.08 - 0.045 * t, 0.07 - 0.04 * t];
    };
    const env = (d) => {             // + key softbox (≈ 25° radius) and rim softbox (≈ 17°); moderate contrast so 128 spp converges
        const c = envSky(d);
        const k = dot(d, KEY); if (k > 0.89) { const s = 4.5 * PL.smoothstep(0.89, 0.91, k); c[0] += s; c[1] += s * 0.95; c[2] += s * 0.88; }
        const q = dot(d, RIM); if (q > 0.95) { const s = 3 * PL.smoothstep(0.95, 0.96, q); c[0] += s * 0.8; c[1] += s * 0.9; c[2] += s; }
        return c;
    };
    // lat-long mapping used by the prefiltered levels (the HLSL uses a cube map; the JS uses lat-long for brevity)
    const envUV = (d) => [Math.atan2(d[0], -d[2]) / (2 * PI) + 0.5, Math.acos(clamp(d[1], -1, 1)) / PI];
    const envDir = (u, v) => { const ph = (u - 0.5) * 2 * PI, th = v * PI, s = Math.sin(th); return [s * Math.sin(ph), Math.cos(th), -s * Math.cos(ph)]; };
    const sampleLL = (L, d) => {        // bilinear: wrap in u, clamp in v
        const [u, v] = envUV(d), W = L.W, H = L.H, D = L.data;
        const fx = u * W - 0.5, fy = clamp(v * H - 0.5, 0, H - 1);
        const x0 = Math.floor(fx), tx = fx - x0, y0 = Math.min(H - 2, Math.floor(fy)), ty = fy - y0;
        const xa = ((x0 % W) + W) % W, xb = (xa + 1) % W, out = [0, 0, 0];
        for (let c = 0; c < 3; c++) {
            const a0 = D[(y0 * W + xa) * 3 + c], a1 = D[(y0 * W + xb) * 3 + c], b0 = D[((y0 + 1) * W + xa) * 3 + c], b1 = D[((y0 + 1) * W + xb) * 3 + c];
            out[c] = lerp(lerp(a0, a1, tx), lerp(b0, b1, tx), ty);
        }
        return out;
    };
    const samplePre = (levels, d, r) => {    // trilinear over roughness mips: mip = r · (count − 1)
        const m = clamp(r, 0, 1) * (levels.length - 1), i = Math.min(levels.length - 2, Math.floor(m)), t = m - i;
        const a = sampleLL(levels[i], d), b = sampleLL(levels[i + 1], d);
        return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
    };

    // ── sphere helpers (same pixel / tangent convention as BX.renderSphere, tangent axis = up) ──
    const spherePx = (x, y, res) => {
        const u = 2 * (x + 0.5) / res - 1, v = -(2 * (y + 0.5) / res - 1), r2 = u * u + v * v;
        if (r2 > 1) return null;
        const n = [u, v, Math.sqrt(1 - r2)];
        let t = BX.cross([0, 1, 0], n);
        if (Math.hypot(t[0], t[1], t[2]) < 1e-4) t = [1, 0, 0];
        t = norm(t);
        return { n, t, b: BX.cross(n, t) };
    };
    const bgPx = (y, res, out) => { const g = 0.035 + 0.03 * (1 - y / res); out[0] = g * 0.8; out[1] = g * 0.9; out[2] = g * 1.15; return out; };
    /** draw an RGB radiance buffer (NaN = background) through BX.display */
    const drawBuf = (canvas, buf, res, ex = 1) => PL.draw.pixels(canvas, res, res, (x, y, out) => {
        const i = (y * res + x) * 3;
        if (Number.isNaN(buf[i])) return bgPx(y, res, out);
        const d = BX.display([buf[i], buf[i + 1], buf[i + 2]], ex); out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
    });
    /** white furnace: the sphere sits in a uniform environment L = 1 (also the background); alb(μ) → rgb | number.
     *  Display = plain sRGB encode (no ACES): furnace radiance is ≤ 1 by construction and the tone-map shoulder would hide the loss. */
    const srgb3 = (e, out) => { const v = Array.isArray(e) ? e : [e, e, e]; for (let c = 0; c < 3; c++) out[c] = BX.toSrgb(clamp(v[c], 0, 1)); return out; };
    const furnace = (canvas, alb, res = 96) => {
        PL.draw.pixels(canvas, res, res, (x, y, out) => {
            const u = 2 * (x + 0.5) / res - 1, v = -(2 * (y + 0.5) / res - 1), r2 = u * u + v * v, ring = Math.abs(Math.sqrt(r2) - 1) * res / 2;
            if (ring < 0.5) { out[0] = out[1] = out[2] = 0.72; return out; }     // silhouette marker only (not data)
            if (r2 > 1) { out[0] = out[1] = out[2] = 1; return out; }
            return srgb3(alb(Math.sqrt(1 - r2)), out);
        });
    };
    /** grid of furnace spheres: rows × cols cells of `cell` px, fn(row, col, μ) → rgb | number */
    const furnaceGrid = (canvas, rows, cols, cell, fn) => PL.draw.pixels(canvas, cols * cell, rows * cell, (x, y, out) => {
        const col = Math.floor(x / cell), row = Math.floor(y / cell), cx = x % cell, cy = y % cell, m = 0.08 * cell, d = cell - 2 * m;
        const u = 2 * (cx + 0.5 - m) / d - 1, v = 2 * (cy + 0.5 - m) / d - 1, r2 = u * u + v * v;
        if (Math.abs(Math.sqrt(r2) - 1) * d / 2 < 0.5) { out[0] = out[1] = out[2] = 0.72; return out; }
        if (r2 > 1) { out[0] = out[1] = out[2] = 1; return out; }
        return srgb3(fn(row, col, Math.sqrt(1 - r2)), out);
    });
    /** per-pixel Monte-Carlo of ∫ L_env f cosθi dωi on the sphere with one sampler (F from the material table) */
    const iblMC = (S, res, spp, sampler, out, seed, strat, envFn = env) => {
        const a = S.a;
        for (let y = 0; y < res; y++) for (let x = 0; x < res; x++) {
            const i = (y * res + x) * 3, P = spherePx(x, y, res);
            if (!P) { out[i] = out[i + 1] = out[i + 2] = NaN; continue; }
            const { n, t, b } = P, wo = [t[2], b[2], n[2]];
            const rnd = PL.rng((seed ^ Math.imul(y * res + x + 1, 0x9e3779b1)) >>> 0), o1 = rnd(), o2 = rnd();
            let r = 0, g = 0, bl = 0;
            for (let s = 0; s < spp; s++) {
                let u1, u2;
                if (strat) { const hm = BX.hammersley(s, spp); u1 = (hm[0] + o1) % 1; u2 = (hm[1] + o2) % 1; } else { u1 = rnd(); u2 = rnd(); }
                const smp = sampler(wo, a, u1, u2);
                if (!smp.w) continue;
                const l = smp.wi, wW = [t[0] * l[0] + b[0] * l[1] + n[0] * l[2], t[1] * l[0] + b[1] * l[1] + n[1] * l[2], t[2] * l[0] + b[2] * l[1] + n[2] * l[2]];
                const L = envFn(wW), F = fLook(S.Ftab, smp.oh), w = smp.w / spp;
                r += L[0] * F[0] * w; g += L[1] * F[1] * w; bl += L[2] * F[2] * w;
            }
            out[i] = r; out[i + 1] = g; out[i + 2] = bl;
        }
    };
    /** BX.hemiImage + make sure a lobe narrower than one pixel still shows: the pixel that contains the analytic peak
     *  direction also evaluates fn exactly at the peak (max of the two) — otherwise r → 0 lobes fall between pixel centres */
    const hemiPeak = (fn, res, out, peak) => {
        out = BX.hemiImage(fn, res, out);
        if (peak && peak[2] > 0) {
            const [px, py] = BX.hemiPixel(peak, res), x = Math.min(res - 1, Math.max(0, Math.round(px))), y = Math.min(res - 1, Math.max(0, Math.round(py))), i = y * res + x;
            if (!Number.isNaN(out[i])) out[i] = Math.max(out[i], fn(peak));
        }
        return out;
    };
    /** log-scale heat map of a hemisphere image (NaN outside the disk draws black) */
    const logField = (c, data, res, dec = 4, map = 'inferno') => {
        let m = 0;
        for (let i = 0; i < data.length; i++) if (data[i] > m) m = data[i];
        const top = Math.log10(m || 1);
        return PL.draw.field(c, data, res, res, { map, stretch: (v) => Math.log10(Math.max(v, 1e-30)), range: [top - dec, top] });
    };
    const hemiProbe = (res, fmt) => (x, y, S) => {
        const w = BX.hemiDir(x, y, res);
        if (!w) return '';
        const th = Math.acos(w[2]) * 180 / PI, ph = Math.atan2(w[1], w[0]) * 180 / PI;
        return `θ = ${th.toFixed(1)}°  φ = ${ph.toFixed(0)}°\n` + fmt(w, S);
    };
    return {
        lerp, dGGX, dBeck, bpExp, dBP, lamGGX, lamBeck, anisoAlpha, phong, blinn, FT, fTable, fLook, ct, refl, sampNDF, sampVNDF, woOf, albedo,
        lut2, lut1, lut1rgb, basis, envSky, env, envUV, envDir, sampleLL, samplePre, spherePx, bgPx, drawBuf, srgb3, furnace, furnaceGrid, iblMC, hemiPeak, logField, hemiProbe,
    };
})();

const MF_RES = 96;                                   // hemisphere image resolution
const MF_M = 48;                                     // μ samples of the per-material albedo tables
const MF_RLIST = [0.125, 0.25, 0.5, 0.75, 1.0];      // fixed roughness set for the furnace curves
const MF_COLORS = ['#56b6c2', '#6ab0e0', '#c678dd', '#e0a060', '#e06c75'];
const MF_DFG_N = 32, MF_IBL_RES = 56;
const MF_PRE = [[48, 24], [32, 16], [32, 16], [16, 8], [16, 8], [16, 8]];   // prefiltered levels, r = k / 5
const MF_PLOT = [360, 180];
const MF_DEG = 180 / Math.PI;

// ── HLSL shared by this module's steps (shown in the graph Common panel) ──
const MF_HLSL_SMITH = `
    // 本模块 · Smith G1 / G2，各向异性 GGX（切线空间：x = T，y = B，z = N）
    float G1_Smith(float NoX, float a) { return 1.0 / (1.0 + Lambda_GGX(NoX, a)); }
    float G2_HeightCorrelated(float NoV, float NoL, float a) {
        return 1.0 / (1.0 + Lambda_GGX(NoV, a) + Lambda_GGX(NoL, a));
    }
    float D_GGXAniso(float3 H, float ax, float ay) {
        float3 s = float3(H.x / ax, H.y / ay, H.z);
        float  d = dot(s, s);
        return 1.0 / (PI * ax * ay * d * d);
    }
    float Lambda_GGXAniso(float3 W, float ax, float ay) {
        float c2 = max(W.z * W.z, 1e-7);
        return 0.5 * (sqrt(1.0 + (ax * ax * W.x * W.x + ay * ay * W.y * W.y) / c2) - 1.0);
    }`;
const MF_HLSL_D = `
    float D_GGX(float NoH, float a) {                    // 系列共享（Common）
        float a2 = a * a;
        float d  = (NoH * a2 - NoH) * NoH + 1.0;
        return a2 / (PI * d * d);
    }
    float D_GGXAniso(float3 H, float ax, float ay) {     // 本模块：H 在切线空间
        float3 s = float3(H.x / ax, H.y / ay, H.z);
        float  d = dot(s, s);
        return 1.0 / (PI * ax * ay * d * d);
    }`;
const MF_HLSL_SAMPLING = `
    float2 Hammersley(uint i, uint n) { return float2((i + 0.5) / n, reversebits(i) * 2.3283064365386963e-10); }

    // 经典 NDF 采样：pdf(H) = D(H)·NoH（Walter 2007）
    float3 SampleGGXNDF(float a, float2 u) {
        float tan2 = a * a * u.x / max(1.0 - u.x, 1e-7);
        float cosT = rsqrt(1.0 + tan2), sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
        float phi  = 2.0 * PI * u.y;
        return float3(sinT * cos(phi), sinT * sin(phi), cosT);
    }
    // 可见法线采样（Heitz 2018, JCGT 7(4)）：pdf(H) = G1(V)·max(0, V·H)·D(H) / NoV
    float3 SampleGGXVNDF(float3 V, float ax, float ay, float2 u) {
        float3 Vh = normalize(float3(ax * V.x, ay * V.y, V.z));              // 拉伸成 α = 1 的半球
        float  lensq = Vh.x * Vh.x + Vh.y * Vh.y;
        float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
        float3 T2 = cross(Vh, T1);
        float  r = sqrt(u.x), phi = 2.0 * PI * u.y;
        float  t1 = r * cos(phi), t2 = r * sin(phi);
        float  s = 0.5 * (1.0 + Vh.z);
        t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;                      // 投影圆盘：被遮挡的半边压缩掉
        float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
        return normalize(float3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));      // 反拉伸回椭球
    }`;
const MF_HLSL_FRESNEL = `
    // 精确电介质 Fresnel（非偏振），eta = η_t / η_i
    float F_Dielectric(float cosI, float eta) {
        float s2t = (1.0 - cosI * cosI) / (eta * eta);
        if (s2t >= 1.0) return 1.0;                                          // 全反射
        float cosT = sqrt(1.0 - s2t);
        float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
        float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
        return 0.5 * (rs * rs + rp * rp);
    }
    // 精确导体 Fresnel（复 IOR η + iκ，逐 RGB 通道，入射侧为空气）
    float3 F_Conductor(float cosI, float3 eta, float3 k) {
        float  c2 = cosI * cosI, s2 = 1.0 - c2;
        float3 e2 = eta * eta, k2 = k * k;
        float3 t0 = e2 - k2 - s2;
        float3 a2b2 = sqrt(t0 * t0 + 4.0 * e2 * k2);
        float3 a  = sqrt(max(0.0, 0.5 * (a2b2 + t0)));
        float3 t1 = a2b2 + c2, t2 = 2.0 * cosI * a;
        float3 Rs = (t1 - t2) / (t1 + t2);
        float3 t3 = c2 * a2b2 + s2 * s2, t4 = t2 * s2;
        float3 Rp = Rs * (t3 - t4) / (t3 + t4);
        return 0.5 * (Rp + Rs);
    }`;
const MF_HLSL_BRDF = `
    // Cook-Torrance · GGX · 高度相关 Smith · Schlick（世界空间只用与 N 的点积）
    float3 BRDF_CookTorrance(float3 N, float3 L, float3 V, float3 F0, float rough) {
        float  a   = max(rough * rough, 1e-4);
        float3 H   = normalize(L + V);
        float  NoL = dot(N, L), NoV = dot(N, V);
        if (NoL <= 0.0 || NoV <= 0.0) return 0.0;
        float  NoH = saturate(dot(N, H)), VoH = saturate(dot(V, H));
        float  D   = D_GGX(NoH, a);
        float  Vis = V_SmithGGXCorrelated(NoV, NoL, a);     // = G2 / (4·NoV·NoL)
        float3 F   = F_Schlick(F0, VoH);                     // 自变量是 V·H，不是 N·V
        return D * Vis * F;
    }
    // 方向光：Lo = BRDF · E · NoL`;
const MF_HLSL_FURNACE = `
    // 白炉：L_env ≡ 1、F ≡ 1 → 出射辐亮度 = E(μo)。VNDF 采样下单样本权重 = G2 / G1(V)
    float WhiteFurnace(float NoV, float a, uint N) {
        float3 V = float3(sqrt(1.0 - NoV * NoV), 0.0, NoV);   // 切线空间，φo = 0
        float  E = 0.0;
        for (uint i = 0; i < N; ++i) {
            float3 H = SampleGGXVNDF(V, a, a, Hammersley(i, N));
            float3 L = reflect(-V, H);
            if (L.z > 0.0) E += G2_HeightCorrelated(V.z, L.z, a) / G1_Smith(V.z, a);
        }
        return E / N;                                         // 1 − E = 单次散射丢掉的能量
    }`;
const MF_HLSL_ELUT = `
    RWTexture2D<float> EnergyLUT;     // E(μ, r)   LUT_N × LUT_N  R16F，x = μ = NoV，y = 感知粗糙度 r
    RWTexture1D<float> EavgLUT;       // E_avg(r)  LUT_N          R16F

    [numthreads(8, 8, 1)]
    void CS_BakeEnergyLUT(uint3 id : SV_DispatchThreadID) {
        float mu = (id.x + 0.5) / LUT_N, r = (id.y + 0.5) / LUT_N;   // texel 中心
        EnergyLUT[id.xy] = WhiteFurnace(mu, max(r * r, 1e-4), 128);  // Step 06 的同一个积分
    }
    [numthreads(LUT_N, 1, 1)]                                     // 第二个 pass：每行一个 thread
    void CS_BakeEavg(uint3 id : SV_DispatchThreadID) {
        float s = 0.0;
        for (uint i = 0; i < LUT_N; ++i) s += EnergyLUT[uint2(i, id.x)] * (i + 0.5) / LUT_N;
        EavgLUT[id.x] = 2.0 * s / LUT_N;                            // E_avg = 2∫ E(μ) μ dμ
    }`;
const MF_HLSL_KC = `
    // 运行时 · Kulla-Conty 2017：补全波瓣 f_ms（对 ωi、ωo 对称，形状近似漫反射）
    float3 KullaConty_fms(float NoV, float NoL, float r, float3 Favg) {
        float Eo = EnergyLUT.SampleLevel(samLinearClamp, float2(NoV, r), 0);
        float Ei = EnergyLUT.SampleLevel(samLinearClamp, float2(NoL, r), 0);
        float Ea = EavgLUT.SampleLevel(samLinearClamp, r, 0);
        float  fms = (1.0 - Eo) * (1.0 - Ei) / (PI * max(1.0 - Ea, 1e-4));
        float3 Fms = Favg * Favg * Ea / (1.0 - Favg * (1.0 - Ea));   // 每多弹一次多乘一次 F_avg
        return Fms * fms;                                           // f = f_ss + F_ms · f_ms
    }
    float3 Favg_Schlick(float3 F0) { return F0 + (1.0 - F0) / 21.0; } // 2∫(F0 + (1−F0)(1−μ)^5) μ dμ

    // Turquin 2019：不加新波瓣，只把单次散射波瓣整体放大
    float3 Turquin_Scale(float NoV, float r, float3 F0) {
        float Eo = EnergyLUT.SampleLevel(samLinearClamp, float2(NoV, r), 0);
        return 1.0 + F0 * (1.0 / Eo - 1.0);                          // f = f_ss · Turquin_Scale
    }`;
const MF_HLSL_DFG = `
    RWTexture2D<float2> DFGLut;     // (A, B)：∫ f cosθ dω = F0·A + B（Schlick），32×32 RG16F

    [numthreads(8, 8, 1)]
    void CS_BakeDFG(uint3 id : SV_DispatchThreadID) {
        float  NoV = (id.x + 0.5) / 32.0, r = (id.y + 0.5) / 32.0, a = max(r * r, 1e-4);
        float3 V = float3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
        float2 AB = 0.0;
        for (uint i = 0; i < 128; ++i) {
            float3 H = SampleGGXVNDF(V, a, a, Hammersley(i, 128));
            float3 L = reflect(-V, H);
            if (L.z <= 0.0) continue;
            float w  = G2_HeightCorrelated(NoV, L.z, a) / G1_Smith(NoV, a);
            float Fc = Pow5(1.0 - saturate(dot(V, H)));
            AB += float2((1.0 - Fc) * w, Fc * w);
        }
        DFGLut[id.xy] = AB / 128.0;      // A + B = E(μ, r)：同一张 LUT 也给出能量补偿所需的 E
    }`;
const MF_HLSL_PREFILTER = `
    TextureCube<float4>   EnvCube;
    RWTexture2DArray<float4> PrefilteredMip;   // 每个 mip 一次 dispatch，z = cube face

    // Karis 2013：假设 N = V = R，把 GGX 波瓣卷积进环境图；mip ↔ 粗糙度 r = mip / (MIP_COUNT − 1)
    [numthreads(8, 8, 1)]
    void CS_PrefilterEnv(uint3 id : SV_DispatchThreadID) {
        float3 R = CubeTexelDirection(id, MipSize);   // N = V = R
        float  r = MipLevel / (MIP_COUNT - 1.0), a = max(r * r, 1e-4);
        float3 T, B; BuildBasis(R, T, B);             // Duff et al. 2017
        float3 sum = 0.0; float wsum = 0.0;
        for (uint i = 0; i < 128; ++i) {
            float3 h = SampleGGXNDF(a, Hammersley(i, 128));
            float3 H = h.x * T + h.y * B + h.z * R;
            float3 L = 2.0 * dot(R, H) * H - R;
            float  NoL = dot(R, L);
            if (NoL > 0.0) { sum += EnvCube.SampleLevel(samLinear, L, 0).rgb * NoL; wsum += NoL; }
        }
        PrefilteredMip[id] = float4(sum / max(wsum, 1e-6), 1.0);
    }`;
const MF_HLSL_LIGHTING = `
    Texture2D<float>  EnergyLUT;  Texture1D<float> EavgLUT;  Texture2D<float2> DFGLut;
    TextureCube<float4> PrefilteredEnv;

    float SpecularAA(float rough, float3 N) {          // Tokuyoshi & Kaplanyan 2019（Filament 同款）
        float3 du = ddx(N), dv = ddy(N);
        float  variance = SPEC_AA_VARIANCE * (dot(du, du) + dot(dv, dv));   // σ²_screen ≈ 0.25
        float  kernel   = min(2.0 * variance, SPEC_AA_THRESHOLD);           // κ ≈ 0.18
        float  a2 = saturate(rough * rough * rough * rough + kernel);       // α'² = α² + kernel
        return sqrt(sqrt(a2));                                              // 回到感知粗糙度
    }

    float4 PS_DeferredLighting(float4 pos : SV_Position) : SV_Target {
        Surface s = LoadGBuffer(pos.xy);                 // N, F0, rough（GBuffer）
        float3 V   = normalize(ViewPos - s.P);
        float  NoV = abs(dot(s.N, V)) + 1e-5;            // 避免 0 和负值 → NaN
        float  r   = max(SpecularAA(s.rough, s.N), 0.045);
        float  a   = r * r;
        float2 AB  = DFGLut.SampleLevel(samLinearClamp, float2(NoV, r), 0);
        float  Ess = AB.x + AB.y;                         // = E(μo, r)
        float3 energyComp = 1.0 + s.F0 * (1.0 / Ess - 1.0);   // Turquin（Filament 做法）

        // ① 解析光源：Cook-Torrance × 能量补偿
        float3 Lo = 0.0;
        for (uint i = 0; i < LightCount; ++i) {
            float3 L = Lights[i].dir, H = normalize(L + V);
            float  NoL = saturate(dot(s.N, L));
            if (NoL <= 0.0) continue;
            float  NoH = saturate(dot(s.N, H)), VoH = saturate(dot(V, H));
            float3 f = D_GGX(NoH, a) * V_SmithGGXCorrelated(NoV, NoL, a) * F_Schlick(s.F0, VoH);
            Lo += f * energyComp * Lights[i].E * NoL;
        }
        // ② IBL：split-sum = 预滤波环境 × DFG
        float3 R   = reflect(-V, s.N);
        float3 pre = PrefilteredEnv.SampleLevel(samLinear, R, r * (MIP_COUNT - 1)).rgb;
        Lo += pre * (s.F0 * AB.x + AB.y) * energyComp;
        return float4(Lo, 1.0);
    }`;

// ── small drawing helpers for this module ──
function mfScatter(canvas, S, key) {
    const { ctx, w, h } = PL.draw.ctx(canvas);
    const R = Math.min(w, h - 26) / 2 - 6, cx = w / 2, cy = 26 + (h - 26) / 2, d = S[key], st = S[key + 'St'], n = d.length / 4;
    ctx.strokeStyle = '#1d2329';
    for (const c of [0.5, Math.cos(60 / MF_DEG)]) { ctx.beginPath(); ctx.arc(cx, cy, R * Math.sqrt(1 - c), 0, 2 * Math.PI); ctx.stroke(); }
    ctx.strokeStyle = '#3a4148'; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
    const proj = (x, y, z) => { const r = Math.sqrt(Math.max(0, 1 - Math.abs(z))), s = Math.hypot(x, y) || 1; return [cx + x / s * r * R, cy - y / s * r * R]; };
    for (let k = 0; k < n; k++) {
        const x = d[4 * k], y = d[4 * k + 1], z = d[4 * k + 2], wt = d[4 * k + 3];
        if (Number.isNaN(x)) continue;
        const [px, py] = proj(x, y, z);
        if (z <= 0) {          // below the horizon: mirrored to the upper hemisphere for display
            ctx.strokeStyle = '#e06c75'; ctx.lineWidth = 1; ctx.beginPath();
            ctx.moveTo(px - 2.5, py - 2.5); ctx.lineTo(px + 2.5, py + 2.5); ctx.moveTo(px + 2.5, py - 2.5); ctx.lineTo(px - 2.5, py + 2.5); ctx.stroke();
            continue;
        }
        const rel = wt / (st.mean || 1);
        ctx.fillStyle = rel > 2 ? '#e0a060' : '#56b6c2';
        ctx.beginPath(); ctx.arc(px, py, Math.min(6, 0.8 + 1.4 * Math.sqrt(rel)), 0, 2 * Math.PI); ctx.fill();
    }
    const wo = S.scWo, [ox, oy] = proj(wo[0], wo[1], wo[2]);
    ctx.fillStyle = '#e8c25a'; ctx.beginPath(); ctx.arc(ox, oy, 3.5, 0, 2 * Math.PI); ctx.fill();
    ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.fillStyle = '#9aa';
    ctx.fillText(`浪费 ${(st.bad * 100).toFixed(1)}%（背面 h ${(st.back * 100).toFixed(1)}%）`, 4, 2);
    ctx.fillText(`E ≈ ${st.mean.toFixed(3)}   σ/μ = ${st.rel.toFixed(2)}`, 4, 14);
}

PipelineLab.mount({
    title: 'BxDF 02 · 微表面镜面：Cook-Torrance · GGX',
    lang: 'zh-CN',
    codeLang: 'hlsl',
    seed: 2,
    regen: false,

    params: [
        { id: 'rough', label: 'r', min: 0.02, max: 1, step: 0.01, value: 0.5, digits: 2, title: '感知粗糙度 r；GGX α = r²（Burley 2012）' },
        { id: 'theta', label: 'θ', min: 0, max: 89, step: 1, value: 55, unit: '°', map: (d) => d * Math.PI / 180, title: '入射角：波瓣 / 热力图里是光源 θi；Step 08 采样里当作视线 θo（BRDF 互易）' },
        {
            id: 'mat', label: '材质', value: 'gold', title: 'Fresnel 数据：导体用 BX.METALS 的复 IOR；电介质用 η；自定义只有 F0（只能用 Schlick）',
            options: [{ value: 'gold', label: '金 Au' }, { value: 'copper', label: '铜 Cu' }, { value: 'aluminum', label: '铝 Al' }, { value: 'silver', label: '银 Ag' }, { value: 'dielectric', label: '电介质 η' }, { value: 'custom', label: '自定义 F0' }],
        },
        { id: 'fres', label: 'Fresnel', value: 'schlick', options: [{ value: 'schlick', label: 'Schlick' }, { value: 'exact', label: '精确' }], title: 'BRDF 里用哪种 Fresnel（Schlick 的 F0 取精确公式在 0° 的值）' },
        { id: 'ior', label: 'η', min: 1.0, max: 3.0, step: 0.01, value: 1.5, digits: 2, title: '电介质相对 IOR η = η_t/η_i（mat = 电介质时生效）' },
        { id: 'f0', label: 'F0', value: '#d8b26a', map: BX.hex2lin, title: '自定义 F0（sRGB 拾色 → 线性 RGB；mat = 自定义 F0 时生效）' },
        { id: 'aniso', label: 'aniso', min: 0, max: 0.95, step: 0.01, value: 0.8, digits: 2, title: '各向异性（只用于 Step 07）：aspect = √(1 − 0.9·aniso)' },
        { id: 'lutN', label: 'LUT', options: [8, 16, 32], value: 32, title: 'E(μ, r) LUT 分辨率（Step 09；只有它变化时才真正重烘）' },
    ],

    stages: [
        // ─────────── RESIDENT: material-independent bakes ───────────
        {
            id: 'furnaceFixed', lifetime: 'resident', run(S) {
                S.EssFixed = MF_RLIST.map((r) => {
                    const a = BX.alpha(r), out = new Float32Array(MF_M);
                    for (let k = 0; k < MF_M; k++) out[k] = MF.albedo((k + 0.5) / MF_M, a, 256, null);
                    return out;
                });
            },
        },
        {
            id: 'dfg', lifetime: 'resident', run(S) {
                const N = MF_DFG_N, A = PL.buf(S, 'dfgA', N * N), B = PL.buf(S, 'dfgB', N * N), n = 128;
                for (let j = 0; j < N; j++) {
                    const a = BX.alpha((j + 0.5) / N);
                    for (let i = 0; i < N; i++) {
                        const wo = MF.woOf((i + 0.5) / N);
                        let sa = 0, sb = 0;
                        for (let k = 0; k < n; k++) {
                            const [u1, u2] = BX.hammersley(k, n), s = MF.sampVNDF(wo, a, u1, u2);
                            if (!s.w) continue;
                            const fc = BX.pow5(1 - BX.clamp(s.oh, 0, 1));
                            sa += (1 - fc) * s.w; sb += fc * s.w;
                        }
                        A[j * N + i] = sa / n; B[j * N + i] = sb / n;
                    }
                }
            },
        },
        {
            id: 'prefilter', lifetime: 'resident', run(S) {
                const K = MF_PRE.length, n = 128;
                S.pre = MF_PRE.map(([W, H], k) => {
                    const data = new Float32Array(W * H * 3), r = k / (K - 1), a = BX.alpha(r);
                    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
                        const R = MF.envDir((x + 0.5) / W, (y + 0.5) / H), o = (y * W + x) * 3;
                        let c = [0, 0, 0];
                        if (k === 0) {        // mip 0 = the source: 4×4 box supersample of the procedural sky
                            for (let sy = 0; sy < 4; sy++) for (let sx = 0; sx < 4; sx++) { const e = MF.env(MF.envDir((x + (sx + 0.5) / 4) / W, (y + (sy + 0.5) / 4) / H)); c[0] += e[0] / 16; c[1] += e[1] / 16; c[2] += e[2] / 16; }
                        } else {              // Karis 2013: N = V = R, GGX-weighted by NoL
                            const [T, B] = MF.basis(R);
                            let ws = 0;
                            for (let i = 0; i < n; i++) {
                                const [u1, u2] = BX.hammersley(i, n), hl = BX.sampleNDF(a, u1, u2);
                                const Hw = [hl[0] * T[0] + hl[1] * B[0] + hl[2] * R[0], hl[0] * T[1] + hl[1] * B[1] + hl[2] * R[1], hl[0] * T[2] + hl[1] * B[2] + hl[2] * R[2]];
                                const d = 2 * BX.dot(R, Hw), L = [d * Hw[0] - R[0], d * Hw[1] - R[1], d * Hw[2] - R[2]], NoL = BX.dot(R, L);
                                if (NoL <= 0) continue;
                                const e = MF.env(L); c[0] += e[0] * NoL; c[1] += e[1] * NoL; c[2] += e[2] * NoL; ws += NoL;
                            }
                            c = c.map((v) => v / Math.max(ws, 1e-9));
                        }
                        data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2];
                    }
                    return { W, H, r, data };
                });
            },
        },
        // ─────────── PARAM: everything that follows the material / roughness / angle ───────────
        {
            id: 'material', lifetime: 'param', run(S) {
                S.a = BX.alpha(S.rough);
                const m = S.mat;
                let exact = null, F0;
                if (BX.METALS[m]) { const { eta, k } = BX.METALS[m]; exact = (c) => BX.fresnelConductor(c, eta, k); F0 = exact(1); }
                else if (m === 'dielectric') { exact = (c) => { const f = BX.fresnelDielectric(c, S.ior); return [f, f, f]; }; const f0 = BX.f0FromIor(S.ior); F0 = [f0, f0, f0]; }
                else F0 = S.f0.slice();
                S.F0 = F0; S.Fexact = exact; S.Fschlick = (c) => BX.schlick(F0, c);
                S.useExact = S.fres === 'exact' && !!exact;
                MF.fTable(S.useExact ? exact : S.Fschlick, PL.buf(S, 'Ftab', (MF.FT + 1) * 3));
                const Favg = [0, 0, 0];
                for (let i = 0; i < 256; i++) { const mu = (i + 0.5) / 256, f = MF.fLook(S.Ftab, mu); for (let c = 0; c < 3; c++) Favg[c] += 2 * f[c] * mu / 256; }
                S.Favg = Favg;
            },
        },
        {
            id: 'phong', lifetime: 'param', run(S) {
                const wi = BX.dir(S.theta, 0), n = MF.bpExp(S.a), np = n / 4;
                S.hPh = MF.hemiPeak((wo) => MF.phong(wi, wo, np), MF_RES, S.hPh, [-wi[0], -wi[1], wi[2]]);
                S.hBP = MF.hemiPeak((wo) => MF.blinn(wi, wo, n), MF_RES, S.hBP, [-wi[0], -wi[1], wi[2]]);
                let m = 1e-9;
                for (let i = 0; i < S.hPh.length; i++) { if (S.hPh[i] > m) m = S.hPh[i]; if (S.hBP[i] > m) m = S.hBP[i]; }
                S.phMax = m;
            },
        },
        {
            id: 'ndf', lifetime: 'param', run(S) {
                const a = S.a, n = MF.bpExp(a), K = 2048;
                const th = PL.buf(S, 'cdfTh', K), cg = PL.buf(S, 'cdfG', K), cb = PL.buf(S, 'cdfB', K), cp = PL.buf(S, 'cdfP', K);
                // numeric CDF ∫0^θ D cosθh 2π sinθh dθh on a grid dense near θh = 0 (θ ∝ u³) so narrow lobes still integrate
                let sg = 0, sb = 0, sp = 0, prev = 0;
                for (let k = 0; k < K; k++) {
                    const t1 = Math.PI / 2 * Math.pow((k + 1) / K, 3), tm = (prev + t1) / 2, dt = t1 - prev, c = Math.cos(tm), jac = 2 * Math.PI * Math.sin(tm) * c * dt;
                    sg += MF.dGGX(c, a) * jac; sb += MF.dBeck(c, a) * jac; sp += MF.dBP(c, n) * jac;
                    th[k] = t1 * MF_DEG; cg[k] = sg; cb[k] = sb; cp[k] = sp; prev = t1;
                }
                S.normD = [sg, sb, sp];
                S.hDG = MF.hemiPeak((h) => MF.dGGX(h[2], a) * h[2], MF_RES, S.hDG, [0, 0, 1]);
                S.hDB = MF.hemiPeak((h) => MF.dBeck(h[2], a) * h[2], MF_RES, S.hDB, [0, 0, 1]);
            },
        },
        {
            id: 'masking', lifetime: 'param', run(S) {
                const a = S.a, M = 30, n = 1024, X = PL.buf(S, 'mkX', M), Wg = PL.buf(S, 'mkG', M), Wn = PL.buf(S, 'mkN', M), Wa = PL.buf(S, 'mkA', M);
                for (let k = 0; k < M; k++) {
                    const t = (k + 0.5) / M * 88 / MF_DEG, wo = [Math.sin(t), 0, Math.cos(t)];
                    let s = 0;
                    for (let i = 0; i < n; i++) {           // h ~ D cosθh → estimator of ∫ ⟨wo·h⟩ D dωh is ⟨wo·h⟩ / cosθh
                        const [u1, u2] = BX.hammersley(i, n), h = BX.sampleNDF(a, u1, u2);
                        s += Math.max(0, BX.dot(wo, h)) / h[2];
                    }
                    X[k] = t * MF_DEG; Wn[k] = s / n / wo[2]; Wg[k] = Wn[k] * BX.smithG1(wo, a); Wa[k] = 1 + BX.ggxLambda(wo, a);
                }
            },
        },
        {
            id: 'brdf', lifetime: 'param', run(S) {
                const a = S.a, wi = BX.dir(S.theta, 0);
                S.hF = MF.hemiPeak((wo) => BX.lum(MF.ct(wi, wo, a, a, S.Ftab)), MF_RES, S.hF, [-wi[0], -wi[1], wi[2]]);
            },
        },
        {
            id: 'furnace', lifetime: 'param', run(S) {
                const a = S.a, n = 256, E = PL.buf(S, 'EssCur', MF_M), EF = PL.buf(S, 'EFcur', MF_M * 3);
                const lb = PL.buf(S, 'lossBelow', MF_M), ls = PL.buf(S, 'lossShadow', MF_M);
                for (let k = 0; k < MF_M; k++) {
                    const mu = (k + 0.5) / MF_M, wo = MF.woOf(mu);
                    let below = 0, shadow = 0;
                    for (let i = 0; i < n; i++) {
                        const [u1, u2] = BX.hammersley(i, n), s = MF.sampVNDF(wo, a, u1, u2);
                        if (s.wi[2] <= 0) below++; else shadow += 1 - s.w;
                    }
                    lb[k] = below / n; ls[k] = shadow / n;
                    E[k] = MF.albedo(mu, a, n, null);
                    const e = MF.albedo(mu, a, n, S.Ftab); EF[3 * k] = e[0]; EF[3 * k + 1] = e[1]; EF[3 * k + 2] = e[2];
                }
            },
        },
        {
            id: 'aniso', lifetime: 'param', run(S) {
                const [ax, ay] = MF.anisoAlpha(S.rough, S.aniso);
                S.ax = ax; S.ay = ay;
                S.hDA = MF.hemiPeak((h) => BX.ggxD(h, ax, ay) * h[2], MF_RES, S.hDA, [0, 0, 1]);
            },
        },
        {
            id: 'sampling', lifetime: 'param', run(S) {
                const a = S.a, th = Math.min(S.theta, 88.5 / MF_DEG), wo = [Math.sin(th), 0, Math.cos(th)], n = 256;
                S.scWo = wo;
                const stats = (fn, NS, wo2, keep) => {
                    let s1 = 0, s2 = 0, bad = 0, back = 0;
                    for (let i = 0; i < NS; i++) {
                        const [u1, u2] = BX.hammersley(i, NS), r = fn(wo2, a, u1, u2);
                        if (keep) {
                            if (!r.wi) keep.fill(NaN, 4 * i, 4 * i + 4);
                            else { keep[4 * i] = r.wi[0]; keep[4 * i + 1] = r.wi[1]; keep[4 * i + 2] = r.wi[2]; keep[4 * i + 3] = r.w; }
                        }
                        if (!r.wi) back++;
                        if (!r.w) bad++;
                        s1 += r.w; s2 += r.w * r.w;
                    }
                    const mean = s1 / NS, sd = Math.sqrt(Math.max(0, s2 / NS - mean * mean));
                    return { mean, rel: mean > 0 ? sd / mean : 0, bad: bad / NS, back: back / NS };
                };
                S.scNSt = stats(MF.sampNDF, n, wo, PL.buf(S, 'scN', n * 4));
                S.scVSt = stats(MF.sampVNDF, n, wo, PL.buf(S, 'scV', n * 4));
                // relative standard deviation of the single-sample estimator vs θo
                const M = 30, X = PL.buf(S, 'vrX', M), VN = PL.buf(S, 'vrN', M), VV = PL.buf(S, 'vrV', M);
                for (let k = 0; k < M; k++) {
                    const t = (k + 0.5) / M * 88.5 / MF_DEG, w2 = [Math.sin(t), 0, Math.cos(t)];
                    X[k] = t * MF_DEG; VN[k] = stats(MF.sampNDF, 512, w2).rel; VV[k] = stats(MF.sampVNDF, 512, w2).rel;
                }
                // weight histograms at θ (relative to the estimate E), 1024 samples
                const NB = 30, HN = PL.buf(S, 'hgN', NB), HV = PL.buf(S, 'hgV', NB), NS = 1024;
                HN.fill(0); HV.fill(0);
                for (const [fn, Hh, st] of [[MF.sampNDF, HN, S.scNSt], [MF.sampVNDF, HV, S.scVSt]]) {
                    for (let i = 0; i < NS; i++) {
                        const [u1, u2] = BX.hammersley(i, NS), r = fn(wo, a, u1, u2);
                        const rel = r.w / (st.mean || 1), b = Math.min(NB - 1, Math.floor(rel / 3 * NB));
                        Hh[b] += 1 / NS;
                    }
                }
            },
        },
        {
            id: 'iblRef', lifetime: 'param', lazy: true, run(S) {      // converged-ish reference: stratified VNDF, 128 spp
                const res = MF_IBL_RES;
                MF.iblMC(S, res, 128, MF.sampVNDF, PL.buf(S, 'iblRefBuf', res * res * 3), 99, true);
            },
        },
        {
            // sky-only environment, 4 spp, two independent renders per sampler: Var[X] = E[(X1 − X2)²] / 2 needs no reference
            id: 'noise', lifetime: 'param', lazy: true, run(S) {
                const res = MF_IBL_RES, NB = 10, n3 = res * res * 3;
                const A = PL.buf(S, 'nzN', n3), B = PL.buf(S, 'nzV', n3), A2 = PL.buf(S, 'nzN2', n3), B2 = PL.buf(S, 'nzV2', n3);
                MF.iblMC(S, res, 4, MF.sampNDF, A, 1234, false, MF.envSky);
                MF.iblMC(S, res, 4, MF.sampVNDF, B, 1234, false, MF.envSky);
                MF.iblMC(S, res, 4, MF.sampNDF, A2, 777, false, MF.envSky);
                MF.iblMC(S, res, 4, MF.sampVNDF, B2, 777, false, MF.envSky);
                const vN = new Float64Array(NB), vV = new Float64Array(NB), sm = new Float64Array(NB), cnt = new Float64Array(NB);
                const lum = (T, i) => 0.2126 * T[i] + 0.7152 * T[i + 1] + 0.0722 * T[i + 2];
                for (let y = 0; y < res; y++) for (let x = 0; x < res; x++) {
                    const P = MF.spherePx(x, y, res);
                    if (!P) continue;
                    const i = (y * res + x) * 3, b = Math.min(NB - 1, Math.floor(P.n[2] * NB));
                    vN[b] += (lum(A, i) - lum(A2, i)) ** 2 / 2; vV[b] += (lum(B, i) - lum(B2, i)) ** 2 / 2;
                    sm[b] += (lum(A, i) + lum(A2, i) + lum(B, i) + lum(B2, i)) / 4; cnt[b]++;
                }
                const X = PL.buf(S, 'nzX', NB), EN = PL.buf(S, 'nzEN', NB), EV = PL.buf(S, 'nzEV', NB);
                for (let b = 0; b < NB; b++) {
                    const m = cnt[b] ? sm[b] / cnt[b] : 1;
                    X[b] = (b + 0.5) / NB; EN[b] = cnt[b] ? Math.sqrt(vN[b] / cnt[b]) / m : NaN; EV[b] = cnt[b] ? Math.sqrt(vV[b] / cnt[b]) / m : NaN;
                }
            },
        },
        {
            // E(μ, r) is material-independent: the stage is PARAM-BAKE but only re-bakes when lutN changes
            id: 'energyLUT', lifetime: 'param', run(S) {
                const N = S.lutN;
                if (S._lutN === N && S.Elut && S.Elut.length === N * N) return;
                const E = PL.buf(S, 'Elut', N * N), Ea = PL.buf(S, 'Eavg', N);
                for (let j = 0; j < N; j++) {
                    const a = BX.alpha((j + 0.5) / N);
                    let s = 0;
                    for (let i = 0; i < N; i++) { const mu = (i + 0.5) / N, e = MF.albedo(mu, a, 128, null); E[j * N + i] = e; s += e * mu; }
                    Ea[j] = 2 * s / N;
                }
                S._lutN = N;
            },
        },
        {
            id: 'ms', lifetime: 'param', run(S) {
                const N = S.lutN, r = S.rough, M = MF_M;
                // E_ms(μo) = ∫ f_ms cosθi dωi for roughness rr, integrated numerically over μi with LUT lookups
                const emsOf = (rr) => {
                    const Ea = MF.lut1(S.Eavg, N, rr);
                    let I = 0;                                 // 2∫(1 − E(μi)) μi dμi (≈ 1 − E_avg up to LUT interpolation)
                    for (let i = 0; i < 64; i++) { const mu = (i + 0.5) / 64; I += (1 - MF.lut2(S.Elut, N, mu, rr)) * mu; }
                    I = 2 * I / 64;
                    const ems = new Float32Array(M);
                    for (let k = 0; k < M; k++) { const Eo = MF.lut2(S.Elut, N, (k + 0.5) / M, rr); ems[k] = 1 - Ea > 1e-5 ? (1 - Eo) / (1 - Ea) * I : 0; }
                    return { Ea, ems, Fms: S.Favg.map((f) => f * f * Ea / (1 - f * (1 - Ea))) };
                };
                const cur = emsOf(r), Ea = cur.Ea, Fms = cur.Fms;
                S.EavgCur = Ea; S.Fms = Fms;
                const ems = PL.buf(S, 'msE', M), kc = PL.buf(S, 'msKC', M), tq = PL.buf(S, 'msTQ', M);
                const kcC = PL.buf(S, 'msKCc', M * 3), tqC = PL.buf(S, 'msTQc', M * 3);
                for (let k = 0; k < M; k++) {
                    const mu = (k + 0.5) / M, Eo = MF.lut2(S.Elut, N, mu, r);
                    ems[k] = cur.ems[k];
                    kc[k] = S.EssCur[k] + ems[k];
                    tq[k] = S.EssCur[k] / Eo;                               // Turquin with F0 = 1
                    for (let c = 0; c < 3; c++) {
                        const ss = S.EFcur[3 * k + c];
                        kcC[3 * k + c] = ss + Fms[c] * ems[k];
                        tqC[3 * k + c] = ss * (1 + S.F0[c] * (1 / Eo - 1));
                    }
                }
                // fixed-roughness strip with the current material: single scattering vs + Kulla-Conty
                S.msStrip = [0.25, 0.5, 0.75, 1.0].map((rr) => {
                    const a = BX.alpha(rr), o = emsOf(rr), ss = new Float32Array(M * 3), kk = new Float32Array(M * 3);
                    for (let k = 0; k < M; k++) {
                        const e = MF.albedo((k + 0.5) / M, a, 128, S.Ftab);
                        for (let c = 0; c < 3; c++) { ss[3 * k + c] = e[c]; kk[3 * k + c] = e[c] + o.Fms[c] * o.ems[k]; }
                    }
                    return { ss, kc: kk };
                });
            },
        },
    ],

    intro: {
        title: '微表面镜面 BRDF：从 Phong 到生产级 GGX',
        flow: ['Phong / Blinn-Phong', 'D · NDF', 'G · Smith', 'F · Fresnel', 'Cook-Torrance', '白炉 E(μ)', '各向异性', 'VNDF 采样', 'Kulla-Conty', 'Split-sum IBL'],
        html: '<p>本篇只讲<strong>镜面反射</strong>（漫反射见 01）。主线是工业界默认组合：<strong>GGX NDF + 高度相关 Smith G2 + Schlick Fresnel</strong>，α = r²（Burley 2012）。Beckmann、可分离 G、精确 Fresnel、Turquin 补偿作为对照穿插在各步里。'
            + '所有预览都是页面里 JS 参考实现真实算出来的：球体渲染逐像素调用 BRDF，波瓣图在入射平面（x–z）内扫描 ω<sub>o</sub>，半球热力图用等面积投影（半径² = 1 − cosθ），白炉 / LUT 用 VNDF + Hammersley 数值积分。'
            + '约定与系列一致：n = +Z，ω<sub>i</sub> 指向光源，ω<sub>o</sub> 指向观察者，线性 RGB，显示时才过 ACES + sRGB（白炉测试与 Fresnel 色条例外：值域 ≤ 1，直接 sRGB 编码）。参数 <code>θ</code> 是入射角（Step 08 里当作视线角）。</p>'
            + BX.nav('microfacet'),
    },

    steps: [
        // ─────────────────────────── 01 ───────────────────────────
        {
            id: 'phong', title: '起点：Phong → 归一化 Blinn-Phong → 法线分布', tag: 'PHONG',
            desc: 'Phong（1975）把高光写成围绕镜面反射方向 r 的 <code>cos^n</code> 波瓣；Blinn（1977）改用半程向量 h：<code>(n·h)^n</code>。原始版本没有归一化——改指数 n，高光总能量跟着变，美术只能同时拧 k<sub>s</sub> 和 n。乘上归一化系数后，n 只控制"多尖"，不再改变"多亮"。更关键的是：<code>(n·h)^n</code> 本身就可以读成<strong>微表面法线分布</strong>——"有多少微小镜面的法线恰好朝向 h"。这正是 Cook-Torrance 的 D 项；Blinn-Phong 等价于"D = cos<sup>n</sup>θ<sub>h</sub>、G 被 cosθ<sub>i</sub>cosθ<sub>o</sub> 隐式抵消、F 是常数"的微表面模型。',
            formula: [
                'f_Phong = k_s · (n_p + 2)/(2π) · max(0, r·ω_o)^{n_p}     // r = reflect(ω_i)，Lafortune 1994 归一化',
                'f_Blinn = k_s · (n + 8)/(8π) · (n·h)^n                   // h = normalize(ω_i + ω_o)，RTR 近似归一化',
                'D_BP(h) = (n + 2)/(2π) · cos^{n}θ_h                        // 当 NDF 用：∫ D cosθh dω = 1 精确成立\nf ≈ F · D_BP · G_implicit / (4 cosθ_{i} cosθ_{o}),   G_implicit = cosθ_{i} cosθ_{o}',
                'n = 2/α^2 − 2      n_p ≈ n / 4                           // Walter 2007：n ↔ Beckmann α；Phong 指数约为 Blinn 的 1/4',
            ],
            code: {
                title: 'HLSL · 归一化 Phong / Blinn-Phong（仅镜面项）', src: `
                float3 Spec_PhongNormalized(float3 N, float3 L, float3 V, float3 ks, float np) {
                    float3 R = reflect(-L, N);                       // 光的镜面反射方向
                    return ks * (np + 2.0) / (2.0 * PI) * pow(saturate(dot(R, V)), np);
                }
                float3 Spec_BlinnPhongNormalized(float3 N, float3 L, float3 V, float3 ks, float n) {
                    float3 H = normalize(L + V);
                    return ks * (n + 8.0) / (8.0 * PI) * pow(saturate(dot(N, H)), n);
                }
                // 同一个 cos^n θh 当作 NDF：精确归一化
                float D_BlinnPhong(float NoH, float n)    { return (n + 2.0) / (2.0 * PI) * pow(NoH, n); }
                float BlinnExponentFromAlpha(float a)     { return 2.0 / (a * a) - 2.0; }   // Walter 2007`,
            },
            previews: [
                {
                    id: 'phongSphere', label: 'Phong（n<sub>p</sub> = n/4）', sub: '光源 = [0, sin2θ, cos2θ]：高光中心处 θi = θo = θ', smooth: true,
                    render: (c, S) => { const np = MF.bpExp(S.a) / 4; BX.renderSphere(c, (wi, wo) => { const k = (np + 2) / (2 * Math.PI) * Math.pow(Math.max(0, -wi[0] * wo[0] - wi[1] * wo[1] + wi[2] * wo[2]), np); return S.F0.map((f) => f * k); }, { lights: [{ dir: [0, Math.sin(2 * S.theta), Math.cos(2 * S.theta)], color: [3, 3, 3] }] }); },
                },
                {
                    id: 'blinnSphere', label: '归一化 Blinn-Phong', sub: 'n = 2/α² − 2 · 同一光源', smooth: true,
                    render: (c, S) => { const n = MF.bpExp(S.a); BX.renderSphere(c, (wi, wo) => { const k = (n + 8) / (8 * Math.PI) * Math.pow(Math.max(0, BX.half(wi, wo)[2]), n); return S.F0.map((f) => f * k); }, { lights: [{ dir: [0, Math.sin(2 * S.theta), Math.cos(2 * S.theta)], color: [3, 3, 3] }] }); },
                },
                {
                    id: 'phongLobe', label: '入射平面内的波瓣 f(ω<sub>i</sub>, ω<sub>o</sub>)', sub: '黄线 = ω<sub>i</sub>（θ 参数），虚线 = 镜面方向', size: MF_PLOT,
                    render: (c, S) => {
                        const wi = BX.dir(S.theta, 0), n = MF.bpExp(S.a), np = n / 4;
                        BX.lobe(c, {
                            wi, series: [
                                { fn: (wo) => (wo[2] > 0 ? (np + 2) / (2 * Math.PI) * Math.pow(Math.max(0, -wi[0] * wo[0] + wi[2] * wo[2]), np) : 0), color: '#6ab0e0', label: `Phong n_p=${np.toFixed(0)}` },
                                { fn: (wo) => (wo[2] > 0 ? (n + 8) / (8 * Math.PI) * Math.pow(Math.max(0, BX.half(wi, wo)[2]), n) : 0), color: '#e0a060', label: `Blinn-Phong n=${n.toFixed(0)}` },
                            ],
                        });
                    },
                },
                {
                    id: 'phongHemi', label: 'Phong 波瓣 over ω<sub>o</sub>', sub: '等面积投影 · ω<sub>i</sub> 在 +x · 两图同一色标', smooth: true,
                    render: (c, S) => PL.draw.field(c, S.hPh, MF_RES, MF_RES, { map: 'inferno', range: [0, S.phMax] }),
                    probe: MF.hemiProbe(MF_RES, (w, S) => `f_Phong = ${MF.phong(BX.dir(S.theta, 0), w, MF.bpExp(S.a) / 4).toFixed(3)} /sr`),
                },
                {
                    id: 'blinnHemi', label: 'Blinn-Phong 波瓣 over ω<sub>o</sub>', sub: '掠射时沿入射平面（x 方向）拉长', smooth: true,
                    render: (c, S) => PL.draw.field(c, S.hBP, MF_RES, MF_RES, { map: 'inferno', range: [0, S.phMax] }),
                    probe: MF.hemiProbe(MF_RES, (w, S) => `f_Blinn = ${MF.blinn(BX.dir(S.theta, 0), w, MF.bpExp(S.a)).toFixed(3)} /sr`),
                },
            ],
            notes: '两个球的光源随 θ 移动，保证高光中心处 θ<sub>i</sub> = θ<sub>o</sub> = θ：θ 小时两者几乎一样；θ 拖到 70° 以上，Phong 的高光沿球的边缘横向摊开，Blinn-Phong 仍是紧凑的一团。下面两张热力图是平面上看到的情形（固定 n、扫 ω<sub>o</sub>）：Phong 始终是绕镜面方向的圆斑，Blinn-Phong 在掠射时沿入射平面被拉长、被地平线压扁——湿路面上的路灯倒影被拉成竖条是后者。Phong 指数取 n/4 才能和 Blinn-Phong 的高光大小接近。',
            warn: '(n+8)/(8π) 只是 Blinn-Phong <em>BRDF</em> 的近似归一化（Real-Time Rendering 推广的写法）；严格归一化的是把 cos<sup>n</sup>θ<sub>h</sub> 当 NDF 时的 (n+2)/(2π)。映射 n = 2/α² − 2 是与 Beckmann 的对应（Walter 2007），拿来对应 GGX 只在低粗糙度下近似成立。',
        },
        // ─────────────────────────── 02 ───────────────────────────
        {
            id: 'ndf', title: 'D · 法线分布：Beckmann vs GGX（Trowbridge-Reitz）', tag: 'D',
            desc: 'D(h) 是微表面法线的密度（单位 1/sr）：h 方向单位立体角内，有多少微表面面积（以宏观面积为 1）。它必须满足<strong>投影面积归一化</strong> ∫D(h)cosθ<sub>h</sub>dω<sub>h</sub> = 1——所有微小镜面投影回宏观平面，正好铺满 1 单位面积。Beckmann 来自高斯斜率分布，尾部按 exp(−tan²θ/α²) 迅速衰减；GGX（Walter 2007，与 Trowbridge-Reitz 1975 同一个分布）尾部只按 ~tan<sup>−4</sup> 多项式衰减：高光核心外有一圈柔和的"光晕"，更接近测量数据，也成了业界默认。',
            formula: [
                'D_GGX(h)  = α^2 / (π cos^4θ_h (α^2 + tan^2θ_h)^2) = α^2 / (π ((α^2 − 1) cos^2θ_h + 1)^2)',
                'D_Beck(h) = exp(−tan^2θ_h / α^2) / (π α^2 cos^4θ_h)',
                '∫ D(h) cosθ_{h} dω_{h} = 1      // 投影面积归一化；同一 α 下三者峰值都是 D(0) = 1/(πα^2)',
                'CDF_{GGX}(θ) = tan^2θ / (α^2 + tan^2θ)      CDF_{Beck}(θ) = 1 − exp(−tan^2θ / α^2)   // 中图是数值积分，可与之对照',
            ],
            code: {
                title: 'HLSL · D_Beckmann（D_GGX 见 Common）', src: `
                float D_Beckmann(float NoH, float a) {
                    float c2 = NoH * NoH, t2 = (1.0 - c2) / c2;      // tan²θh
                    return exp(-t2 / (a * a)) / (PI * a * a * c2 * c2);
                }
                // D_GGX(NoH, a) = a² / (π ((a² − 1)·NoH² + 1)²)   —— Common.hlsl
                // fp16 / mediump：1 − NoH² 在 NoH ≈ 1 时被抵消，Filament 用 |N × H|² 代替：
                float D_GGX_Mediump(float3 N, float3 H, float NoH, float a) {
                    float3 NxH = cross(N, H);
                    float  k = a / (dot(NxH, NxH) + NoH * NoH * a * a);   // = a / ((1 − NoH²) + NoH²·a²)
                    return k * k * INV_PI;
                }`,
            },
            previews: [
                {
                    id: 'ndfPlot', label: 'D(θ<sub>h</sub>) · logY', sub: '三条曲线在 θh = 0 重合，差别全在尾部', size: MF_PLOT,
                    render: (c, S) => {
                        const a = S.a, n = MF.bpExp(a), pk = 1 / (Math.PI * a * a);
                        PL.draw.plot(c, {
                            x: [0, 89.9], y: [pk * 1e-8, pk * 3], logY: true, xLabel: 'θh（度）', samples: 600,
                            series: [
                                { fn: (t) => MF.dGGX(Math.cos(t / MF_DEG), a), color: '#e0a060', label: 'GGX' },
                                { fn: (t) => MF.dBeck(Math.cos(t / MF_DEG), a), color: '#6ab0e0', label: 'Beckmann' },
                                { fn: (t) => MF.dBP(Math.cos(t / MF_DEG), n), color: '#5ad07a', dash: [4, 3], label: 'Blinn-Phong' },
                            ],
                            marks: [{ x: Math.atan(a) * MF_DEG, label: 'θh = atan α', color: '#888' }],
                        });
                    },
                },
                {
                    id: 'ndfCdf', label: '累积投影面积 ∫<sub>0</sub><sup>θ</sup> D cosθ<sub>h</sub> dω', sub: '数值积分；右端 = 归一化检验（应为 1）', size: MF_PLOT,
                    render: (c, S) => {
                        const [g, b, p] = S.normD;
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, 1.08], xLabel: 'θh（度）',
                            series: [
                                { data: S.cdfG, x: S.cdfTh, color: '#e0a060', label: `GGX ∫=${g.toFixed(4)}` },
                                { data: S.cdfB, x: S.cdfTh, color: '#6ab0e0', label: `Beckmann ∫=${b.toFixed(4)}` },
                                { data: S.cdfP, x: S.cdfTh, color: '#5ad07a', dash: [4, 3], label: `Blinn-Phong ∫=${p.toFixed(4)}` },
                            ],
                            hmarks: [{ y: 1, color: '#555' }],
                        });
                    },
                },
                {
                    id: 'ndfHemiG', label: 'GGX · D(h)cosθ<sub>h</sub>', sub: '等面积投影，log 4 个数量级', smooth: true,
                    render: (c, S) => MF.logField(c, S.hDG, MF_RES), probe: MF.hemiProbe(MF_RES, (w, S) => `D·cos = ${(MF.dGGX(w[2], S.a) * w[2]).toExponential(2)}`),
                },
                {
                    id: 'ndfHemiB', label: 'Beckmann · D(h)cosθ<sub>h</sub>', sub: '同一色标范围外即黑：尾部更短', smooth: true,
                    render: (c, S) => {
                        let m = 0; for (let i = 0; i < S.hDG.length; i++) if (S.hDG[i] > m) m = S.hDG[i];
                        const top = Math.log10(m || 1);
                        PL.draw.field(c, S.hDB, MF_RES, MF_RES, { map: 'inferno', stretch: (v) => Math.log10(Math.max(v, 1e-30)), range: [top - 4, top] });
                    },
                    probe: MF.hemiProbe(MF_RES, (w, S) => `D·cos = ${(MF.dBeck(w[2], S.a) * w[2]).toExponential(2)}`),
                },
            ],
            notes: '左图 θh = 0 处三条曲线重合（同一 α 映射下峰值都是 1/(πα²)），差别全在尾部：r = 0.5、θh = 45° 处 GGX 比 Beckmann 高约四个数量级——这就是 GGX 高光外那圈"光晕"。中图：GGX 的累积曲线爬得最慢，说明有可观的面积分布在大 θh；三者右端都 ≈ 1。两张热力图共用同一色标（以 GGX 峰值为顶），Beckmann 的外圈很快变黑。',
            warn: 'D 在 α → 0 时退化为 δ 函数：数值上 1/(πα²) 爆掉，所以生产代码会 clamp α（例如 Filament 限制 r ≥ 0.045，移动端更大）。热力图画的是 D·cosθ<sub>h</sub>（采样用的密度），不是 D 本身；r 很小时波瓣比一个像素还窄，包含峰值方向的那个像素会额外在峰值处求值（取大者），所以仍能看到一个亮点。',
        },
        // ─────────────────────────── 03 ───────────────────────────
        {
            id: 'smith', title: 'G · Smith 遮蔽：Λ、G1 与高度相关 G2', tag: 'G',
            desc: 'D 只说"有多少微表面朝向 h"，没说"从 ω<sub>o</sub> 看过去它们会不会被别的微表面挡住"。Smith 假设（微表面高度与法线不相关）把遮蔽写成单个辅助函数 Λ(ω)：G1(ω) = 1/(1 + Λ(ω))。Λ 的含义见中图：从 ω<sub>o</sub> 看，所有朝向 ω<sub>o</sub> 的微表面投影面积之和是 (1 + Λ)·cosθ<sub>o</sub>，比宏观表面多出的 Λ 份恰好是被挡住的；乘上 G1 后回到 cosθ<sub>o</sub>（Heitz 2014 的"可见投影面积守恒"）。同时考虑入射、出射两个方向时，高度相关形式 G2 = 1/(1 + Λ<sub>i</sub> + Λ<sub>o</sub>) 比可分离的 G1·G1 更准：位置高的微表面对两个方向往往同时可见，可分离形式重复惩罚了它们。',
            formula: [
                'Λ_GGX(ω)  = (−1 + √(1 + α^2 tan^2θ)) / 2',
                'Λ_Beck(ω) = (erf(m) − 1)/2 + exp(−m^2)/(2m√π),   m = 1/(α tanθ)   // JS / HLSL 用 Walter 2007 有理近似',
                'G1(ω) = 1 / (1 + Λ(ω))          ∫ G1(ω_o) ⟨ω_o·h⟩ D(h) dω_{h} = cosθ_{o}     // 不乘 G1 时 = (1 + Λ) cosθo',
                'G_{2,sep} = G1(ω_i) · G1(ω_o)        G_{2,corr} = 1 / (1 + Λ(ω_i) + Λ(ω_o))     // Heitz 2014\nV = G2 / (4 cosθ_{i} cosθ_{o})                                                  // 引擎把分母并进"可见性项"',
            ],
            code: {
                title: 'HLSL · Λ / G1 / G2（V_SmithGGXCorrelated 见 Common）', src: `
                float Lambda_Beckmann(float NoX, float a) {
                    float m = NoX / (a * sqrt(1.0 - NoX * NoX));       // m = 1 / (a·tanθ)
                    return m < 1.6 ? (1.0 - 1.259 * m + 0.396 * m * m) / (3.535 * m + 2.181 * m * m) : 0.0;
                }
                float G2_Separable(float NoV, float NoL, float a) { return G1_Smith(NoV, a) * G1_Smith(NoL, a); }
                // G2_HeightCorrelated(NoV, NoL, a) = 1 / (1 + Λ(V) + Λ(L))       —— Common · 本模块
                // V_SmithGGXCorrelated(NoV, NoL, a) = G2 / (4·NoV·NoL)，已把 Λ 展开化简 —— Common · 系列共享
                //   0.5 / (NoL·√(NoV²(1−a²)+a²) + NoV·√(NoL²(1−a²)+a²))`,
            },
            previews: [
                {
                    id: 'g1Plot', label: 'G1(θ) = 1/(1+Λ)', sub: '实线 GGX · 虚线 Beckmann（同一 α）', size: MF_PLOT,
                    render: (c, S) => {
                        const rs = [S.rough, 0.2, 0.7, 1.0], cols = ['#e0a060', '#6ab0e0', '#c678dd', '#e06c75'];
                        const series = [];
                        rs.forEach((r, k) => {
                            const a = BX.alpha(r);
                            series.push({ fn: (t) => 1 / (1 + MF.lamGGX(Math.cos(t / MF_DEG), a)), color: cols[k], width: k ? 1.2 : 2.2, label: k ? `r=${r}` : `r=${r.toFixed(2)}（当前）` });
                            series.push({ fn: (t) => 1 / (1 + MF.lamBeck(Math.cos(t / MF_DEG), a)), color: cols[k], width: k ? 1 : 1.6, dash: [4, 3] });
                        });
                        PL.draw.plot(c, { x: [0, 90], y: [0, 1.05], xLabel: 'θ（度）', series, samples: 300 });
                    },
                },
                {
                    id: 'g1Check', label: '可见投影面积检验', sub: '数值积分 ∫⟨ω<sub>o</sub>·h⟩D dω / cosθ<sub>o</sub>，NDF 采样 1024 点', size: MF_PLOT,
                    render: (c, S) => {
                        let hi = 1; for (let k = 0; k < S.mkN.length; k++) hi = Math.max(hi, S.mkN[k], S.mkA[k]);
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0.9, hi * 1.08], xLabel: 'θo（度）',
                            series: [
                                { data: S.mkN, x: S.mkX, color: '#e06c75', width: 2, label: '不乘 G1（数值）' },
                                { data: S.mkA, x: S.mkX, color: '#e8c25a', dash: [4, 3], label: '1 + Λ（解析）' },
                                { data: S.mkG, x: S.mkX, color: '#5ad07a', width: 2, label: '× G1（数值）' },
                            ],
                            hmarks: [{ y: 1, color: '#555' }],
                        });
                    },
                },
                {
                    id: 'g2Plot', label: '入射平面内 G2(ω<sub>i</sub>, ω<sub>o</sub>)', sub: 'ω<sub>i</sub> 取 θ，ω<sub>o</sub> 在镜面一侧扫描', size: MF_PLOT,
                    render: (c, S) => {
                        const wi = BX.dir(S.theta, 0), wo = (t) => [-Math.sin(t / MF_DEG), 0, Math.cos(t / MF_DEG)], a1 = BX.alpha(1);
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, 1.05], xLabel: 'θo（度）',
                            series: [
                                { fn: (t) => BX.smithG2(wo(t), wi, S.a), color: '#5ad07a', width: 2, label: `高度相关 · r=${S.rough.toFixed(2)}` },
                                { fn: (t) => BX.smithG1(wo(t), S.a) * BX.smithG1(wi, S.a), color: '#e06c75', width: 2, dash: [4, 3], label: `可分离 · r=${S.rough.toFixed(2)}` },
                                { fn: (t) => BX.smithG2(wo(t), wi, a1), color: '#56b6c2', width: 1.2, label: '高度相关 · r=1' },
                                { fn: (t) => BX.smithG1(wo(t), a1) * BX.smithG1(wi, a1), color: '#e0a060', width: 1.2, dash: [4, 3], label: '可分离 · r=1' },
                            ],
                            marks: [{ x: S.theta * MF_DEG, label: 'θo = θi', color: '#e8c25a' }],
                        });
                    },
                },
            ],
            notes: '中图：红线（不乘 G1）随 θ<sub>o</sub> 上升——掠射时看到的微表面总投影面积远超宏观面积，这是物理上不可能的"看到的比存在的多"；乘 G1 后绿线恒为 1。把 r 拖到 1 差距最明显。右图：θ 越大、r 越大，可分离形式比高度相关形式暗得越多（重复计算了遮蔽）。',
            warn: 'Λ 里的 α 就是 r²，不要把感知粗糙度直接塞进去；UE4 早期的 Schlick-GGX（k = α/2）只是 G1 的近似。<code>V_SmithGGXCorrelated</code> 已经包含 1/(4·NoV·NoL)，外面再除一次是常见 bug。',
        },
        // ─────────────────────────── 04 ───────────────────────────
        {
            id: 'fresnel', title: 'F · Fresnel：Schlick 近似 vs 精确（电介质 / 导体）', tag: 'F',
            desc: 'F 决定照到微表面的光有多少被镜面反射，自变量是光线与<strong>微表面法线 h</strong> 的夹角（cosθ<sub>d</sub> = ω<sub>i</sub>·h），不是宏观 NoV。电介质（塑料、水，η ≈ 1.33–1.5）F0 只有 2–4%，掠射时升到 1；导体的复折射率 η + iκ 让 F0 本身就很高且带颜色——金、铜的颜色就来自 Fresnel。Schlick 用 F0 + (1 − F0)(1 − cosθ)<sup>5</sup> 只拟合"从 F0 单调升到 1"，丢掉了导体在 80° 附近的下凹和色调偏移（铝最明显）。',
            formula: [
                'F_Schlick(θ) = F_0 + (1 − F_0)(1 − cosθ)^5          // θ = ∠(ω_i, h)',
                'F_0 = ((η − 1)/(η + 1))^2                             // 电介质；导体取 F_0 = F_exact(0°)',
                'F = (r_s^2 + r_p^2)/2,  r_s = (cosθ − η cosθ_{t})/(cosθ + η cosθ_{t}),  r_p = (η cosθ − cosθ_{t})/(η cosθ + cosθ_{t})\n// 导体：η → η + iκ（复数），同一公式逐 RGB 通道 —— BX.fresnelConductor · BX.METALS',
                'F_avg = 2 ∫_0^1 F(μ) μ dμ        // Schlick：F_avg = F_0 + (1 − F_0)/21，Step 09 用到',
            ],
            code: { title: 'HLSL · 精确 Fresnel（F_Schlick 见 Common）', src: MF_HLSL_FRESNEL },
            previews: [
                {
                    id: 'fPlot', label: 'F(θ) · RGB', sub: '实线 = 精确，虚线 = Schlick（F0 相同）', size: MF_PLOT,
                    render: (c, S) => {
                        const cols = ['#e06c75', '#5ad07a', '#6ab0e0'], series = [];
                        for (let ch = 0; ch < 3; ch++) {
                            if (S.Fexact) series.push({ fn: (t) => S.Fexact(Math.cos(t / MF_DEG))[ch], color: cols[ch], width: 1.8, label: ['R', 'G', 'B'][ch] + ' 精确' });
                            series.push({ fn: (t) => S.Fschlick(Math.cos(t / MF_DEG))[ch], color: cols[ch], width: 1.2, dash: [4, 3], label: S.Fexact ? null : ['R', 'G', 'B'][ch] + ' Schlick' });
                        }
                        PL.draw.plot(c, { x: [0, 90], y: [0, 1.02], xLabel: 'θ = ∠(ωi, h)（度）', series, samples: 181 });
                    },
                },
                {
                    id: 'fSwatch', label: '反射颜色随角度', sub: '上 = 精确 · 下 = Schlick · 左 0° → 右 90° · 直接 sRGB 编码 F（无 tone map）', size: [360, 72],
                    render: (c, S) => PL.draw.pixels(c, 180, 36, (x, y, out) => {
                        const cs = Math.cos((x + 0.5) / 180 * Math.PI / 2), useEx = y < 17 && S.Fexact;
                        if (y >= 17 && y < 19) { out[0] = out[1] = out[2] = 0.05; return out; }
                        return MF.srgb3(useEx ? S.Fexact(cs) : S.Fschlick(cs), out);
                    }),
                },
                {
                    id: 'fErr', label: 'Schlick 误差 F<sub>Schlick</sub> − F<sub>exact</sub>', sub: '自定义 F0 没有 η、κ → 无精确值（曲线为 0）', size: MF_PLOT,
                    render: (c, S) => {
                        const cols = ['#e06c75', '#5ad07a', '#6ab0e0'];
                        PL.draw.plot(c, {
                            x: [0, 90], xLabel: 'θ（度）', samples: 181,
                            series: [0, 1, 2].map((ch) => ({ fn: (t) => { const cs = Math.cos(t / MF_DEG); return S.Fexact ? S.Fschlick(cs)[ch] - S.Fexact(cs)[ch] : 0; }, color: cols[ch] })),
                            hmarks: [{ y: 0, color: '#555' }],
                        });
                    },
                },
            ],
            notes: '材质切到"铝"看 80° 附近的下凹，切到"金"看 Schlick 在 60–85° 偏色；切到"电介质"拖 η：F0 很小，但所有曲线在掠射角都趋向 1（"万物皆有 Fresnel"）。顶部 <code>Fresnel</code> 选"精确"后，后面所有 BRDF 预览改用精确曲线（查 257 项的表，见 MF.fTable）。',
            warn: '这里的 η、κ 只是 3 个波长的 RGB 拟合，不是光谱渲染。生产管线通常只存 F0（metallic 工作流里 F0 = baseColor），需要边缘色调时用 Gulbrandsen 2014 的 (F0, edge tint) 参数化或 Hoffman 2019 的 F82-tint，而不是直接存 η、κ。',
        },
        // ─────────────────────────── 05 ───────────────────────────
        {
            id: 'brdf', title: 'Cook-Torrance 总装：f = D·G·F / (4 cosθ<sub>i</sub> cosθ<sub>o</sub>)', tag: 'BRDF',
            desc: '三项相乘再除以 4cosθ<sub>i</sub>cosθ<sub>o</sub>。分母的来历：D 是"按 h 计"的密度，换成"按 ω<sub>i</sub> 计"要乘反射映射的雅可比 dω<sub>h</sub>/dω<sub>i</sub> = 1/(4|ω<sub>i</sub>·h|)；再乘微表面的投影面积比 ⟨ω<sub>o</sub>·h⟩/cosθ<sub>o</sub> 和辐照度换算 1/cosθ<sub>i</sub>，两个 ⟨ω·h⟩ 相消后剩 1/(4cosθ<sub>i</sub>cosθ<sub>o</sub>)。去掉 G 时，掠射角 cosθ<sub>o</sub> → 0 让波瓣发散（中图虚线）；G2 正好把它压回来。',
            formula: [
                'f(ω_i, ω_o) = D(h) · G_2(ω_i, ω_o) · F(ω_i·h) / (4 cosθ_{i} cosθ_{o})        h = normalize(ω_i + ω_o)',
                '            = D(h) · V(ω_i, ω_o) · F(ω_i·h)                              // V = G_2 / (4 cosθi cosθo)',
                'dω_{h} / dω_{i} = 1 / (4 |ω_i·h|)       // 反射雅可比，Step 08 的 pdf 换算也用它\nL_o = f · E · cosθ_{i}                 // 方向光，E 为法向辐照度',
            ],
            code: { title: 'HLSL · BRDF_CookTorrance', src: MF_HLSL_BRDF },
            previews: [
                {
                    id: 'ctSphere', label: 'Cook-Torrance · GGX', sub: '三灯 studio（key / rim / fill），仅镜面项', smooth: true,
                    render: (c, S) => BX.renderSphere(c, (wi, wo) => MF.ct(wi, wo, S.a, S.a, S.Ftab)),
                },
                {
                    id: 'ctLobe', label: '入射平面波瓣（log 3 decades）', sub: '实线 = 完整 f · 虚线 = 去掉 G2', size: MF_PLOT,
                    render: (c, S) => {
                        const a = S.a, wi = BX.dir(S.theta, 0), full = (wo) => BX.lum(MF.ct(wi, wo, a, a, S.Ftab));
                        let m = 0; for (let k = 0; k <= 356; k++) { const t = (k / 356 - 0.5) * Math.PI * 178 / 180; m = Math.max(m, full([Math.sin(t), 0, Math.cos(t)])); }
                        const noG = (wo) => { if (wo[2] <= 1e-6 || wi[2] <= 1e-6) return 0; const h = BX.half(wi, wo); return Math.min(2 * m, BX.lum(MF.fLook(S.Ftab, BX.dot(wi, h))) * BX.ggxD(h, a) / (4 * wi[2] * wo[2])); };
                        BX.lobe(c, { wi, log: 3, scale: m, series: [{ fn: full, color: '#e0a060', width: 2, label: 'D·G2·F/(4cc)' }, { fn: noG, color: '#e06c75', dash: [4, 3], label: '无 G2' }] });
                    },
                },
                {
                    id: 'ctHemi', label: 'f(ω<sub>i</sub>, ω<sub>o</sub>) over ω<sub>o</sub>', sub: 'ω<sub>i</sub> 在 +x（右侧），镜面峰在左 · sqrt 拉伸', smooth: true,
                    render: (c, S) => PL.draw.field(c, S.hF, MF_RES, MF_RES, { map: 'inferno', stretch: 'sqrt' }),
                    probe: MF.hemiProbe(MF_RES, (w, S) => `f = ${BX.lum(MF.ct(BX.dir(S.theta, 0), w, S.a, S.a, S.Ftab)).toExponential(3)} /sr`),
                },
            ],
            notes: 'θ 推到 80°：中图主波瓣不再对准镜面方向，而是向地平线倾斜（off-specular peak），虚线（无 G2）在地平线附近冲出图外。右图热力图里高光被拉成"月牙"——湿路面上的路灯倒影为什么是竖条。把 r 拖到 0.05 看球上的高光收缩成小点、亮度飙升（D 峰值 ∝ 1/α²）。',
            warn: 'F 的自变量是 V·H（= L·H）；写成 NoV 是常见错误（那是 split-sum 环境光里的近似，见 Step 10）。插值法线 / 法线贴图会让 NoL、NoV ≤ 0：分母要 clamp（Filament：<code>NoV = abs(NoV) + 1e-5</code>），否则 NaN 会在 TAA 历史里扩散。',
        },
        // ─────────────────────────── 06 ───────────────────────────
        {
            id: 'energy', title: '白炉测试：单次散射丢掉了多少能量', tag: 'ENERGY',
            desc: '把环境设为处处 L = 1，并令 F ≡ 1。一个不吸收、能量守恒的材质应该"消失"在白色背景里：出射辐亮度 = 方向反照率 E(μ<sub>o</sub>) = ∫ f cosθ<sub>i</sub> dω<sub>i</sub> = 1。Cook-Torrance 只算了<strong>一次</strong>反射：被 G 挡住的光、以及反射到地平线以下的光，在真实表面上会继续弹射、大部分最终仍会出射——模型里却直接丢了。丢失随粗糙度迅速增加：r = 1 时余弦加权平均 E<sub>avg</sub> 只有约 0.43，正入射处更只剩 1 − ln2 ≈ 0.31（解析值）。和直觉相反，<strong>掠射时 E 反而回升</strong>：从掠射方向看得见的微表面大多朝向观察者，反射光多半向上逃逸（右图红色区域在 μ → 0 时收窄）。粗糙金属因此整体发灰、偏暗，中心比边缘更暗。',
            formula: [
                'E(μ_o) = ∫_{H^2} f(ω_i, ω_o)|_{F=1} cosθ_{i} dω_{i},      μ_o = cosθ_{o}',
                '       ≈ (1/N) Σ_k G_2(ω_o, ω_i^{(k)}) / G_1(ω_o)       // VNDF 重要性采样（Step 08），单样本权重 ≤ 1',
                '1 − E = P(ω_i 落到地平线下) + E[1 − G_2/G_1]               // 右图把两部分分开画\nE_avg = 2 ∫_0^1 E(μ) μ dμ                                // 余弦加权平均（Step 09）',
                'α = 1, μ_o = 1:  D ≡ 1/π,  G_2 = 2μ_i/(1 + μ_i)  ⇒  E = ∫_0^1 μ/(1 + μ) dμ = 1 − ln 2 ≈ 0.307   // 可用来校验积分器',
            ],
            code: { title: 'HLSL · 白炉积分（Step 09 的 LUT bake 复用它）', src: MF_HLSL_FURNACE },
            previews: [
                {
                    id: 'eCurves', label: 'E(μ<sub>o</sub>) · F ≡ 1', sub: '细线 = 固定 r（RESIDENT），粗白线 = 当前 r', size: MF_PLOT,
                    render: (c, S) => {
                        const x = Array.from({ length: MF_M }, (_, k) => (k + 0.5) / MF_M);
                        const series = MF_RLIST.map((r, k) => ({ data: S.EssFixed[k], x, color: MF_COLORS[k], width: 1.2, label: `r=${r}` }));
                        series.push({ data: S.EssCur, x, color: '#eee', width: 2.2, label: `当前 r=${S.rough.toFixed(2)}` });
                        PL.draw.plot(c, { x: [0, 1], y: [0.25, 1.05], xLabel: 'μo = cosθo', series, hmarks: [{ y: 1, color: '#555' }, { y: 1 - Math.LN2, label: '1 − ln2', color: '#e06c75' }] });
                    },
                },
                {
                    id: 'eStrip', label: '白炉球 · r = 0.125 / 0.25 / 0.5 / 0.75 / 1', sub: '背景 = 环境 L = 1；能量守恒就该看不见球', size: [360, 72],
                    render: (c, S) => MF.furnaceGrid(c, 1, 5, 40, (row, col, mu) => MF.lut1(S.EssFixed[col], MF_M, mu)),
                },
                {
                    id: 'eSphere', label: '白炉球 · 当前 r', sub: 'F ≡ 1 · sRGB 直接编码（无 tone map）', smooth: true,
                    render: (c, S) => MF.furnace(c, (mu) => MF.lut1(S.EssCur, MF_M, mu)),
                    probe: (x, y, S) => { const u = 2 * (x + 0.5) / 96 - 1, v = 2 * (y + 0.5) / 96 - 1, r2 = u * u + v * v; if (r2 > 1) return 'L = 1（环境）'; const mu = Math.sqrt(1 - r2); return `μo = ${mu.toFixed(3)}\nE = ${MF.lut1(S.EssCur, MF_M, mu).toFixed(4)}`; },
                },
                {
                    id: 'eLoss', label: '能量去哪了（当前 r）', sub: 'E + 被遮挡 + 落到地平线下 = 1', size: MF_PLOT,
                    render: (c, S) => {
                        const x = Array.from({ length: MF_M }, (_, k) => (k + 0.5) / MF_M), E = Array.from(S.EssCur);
                        const sh = E.map((e, k) => e + S.lossShadow[k]), bl = sh.map((v, k) => v + S.lossBelow[k]);
                        PL.draw.plot(c, {
                            x: [0, 1], y: [0, 1.08], xLabel: 'μo = cosθo',
                            series: [
                                { data: bl, x, color: '#e06c75', fill: true, label: '+ 反射到地平线下' },
                                { data: sh, x, color: '#e0a060', fill: true, label: '+ 被 G 遮挡' },
                                { data: E, x, color: '#5ad07a', fill: true, label: 'E（出射）' },
                            ],
                        });
                    },
                },
            ],
            notes: '白炉条带从左到右 r 增大：r ≤ 0.25 的球几乎看不见；r = 1 的球中心最暗（E ≈ 0.31），边缘反而接近背景。把 r 拖到 1 看右图：μ 越大，红色（反射进表面）越多——正入射时一半的微表面法线倾斜超过 45°，反射方向直接进了地面。中等粗糙度（r ≈ 0.5）的 E 在 μ ≈ 0.2–0.4 处最低，白炉球上表现为靠近轮廓的一圈暗环。',
            warn: '白炉测试必须关掉漫反射、令 F ≡ 1，并且背景与球走同一套显示变换——这里用直接 sRGB 编码而不是 ACES：tone map 的肩部会把 0.9 和 1.0 压到几乎一样，掩盖能量损失。这里的 E 只对各向同性 GGX；各向异性时 E 还依赖 φ<sub>o</sub>。',
        },
        // ─────────────────────────── 07 ───────────────────────────
        {
            id: 'aniso', title: '各向异性 GGX：α<sub>x</sub>、α<sub>y</sub> 与切线标架', tag: 'ANISO',
            desc: '拉丝金属、缎面、头发的高光会沿某个方向拉长：微表面沿切线 t 与副切线 b 的斜率方差不同。GGX 天然推广成椭圆分布——把 h 的 x、y 分量分别除以 α<sub>x</sub>、α<sub>y</sub>；Λ 同样按方向投影出等效 α。代价是需要可靠的切线标架（网格切线、flow map 或程序化方向），D 和 Λ 都要在 (t, b, n) 空间里算。α<sub>x</sub> &gt; α<sub>y</sub> 时法线沿 t 散得更开，高光沿 t 方向拉长。',
            formula: [
                'D(h) = 1 / (π α_x α_y ((h_x/α_x)^2 + (h_y/α_y)^2 + h_z^2)^2)        // h 在切线空间 (t, b, n)',
                'Λ(ω) = (−1 + √(1 + (α_x^2 ω_x^2 + α_y^2 ω_y^2) / ω_z^2)) / 2',
                'aspect = √(1 − 0.9 · aniso),   α_x = α / aspect,   α_y = α · aspect      // Burley 2012；本页再 clamp 到 [10^{−4}, 1]',
            ],
            code: {
                title: 'HLSL · 各向异性 Cook-Torrance（D_GGXAniso / Lambda_GGXAniso 见 Common）', src: `
                void AnisoAlpha(float r, float aniso, out float ax, out float ay) {     // Burley 2012
                    float aspect = sqrt(1.0 - 0.9 * aniso);
                    ax = clamp(r * r / aspect, 1e-4, 1.0);
                    ay = clamp(r * r * aspect, 1e-4, 1.0);
                }
                float3 BRDF_GGXAniso(float3 N, float3 T, float3 L, float3 V, float3 F0, float ax, float ay) {
                    float3 B = cross(N, T);                        // T 先对 N 做 Gram-Schmidt
                    float3x3 TBN = float3x3(T, B, N);
                    float3 l = mul(TBN, L), v = mul(TBN, V), h = normalize(l + v);   // 切线空间
                    if (l.z <= 0.0 || v.z <= 0.0) return 0.0;
                    float  G2 = 1.0 / (1.0 + Lambda_GGXAniso(l, ax, ay) + Lambda_GGXAniso(v, ax, ay));
                    return D_GGXAniso(h, ax, ay) * G2 * F_Schlick(F0, saturate(dot(v, h))) / (4.0 * l.z * v.z);
                }`,
            },
            previews: [
                {
                    id: 'anSphereUp', label: '切线 = cross(up, n)', sub: '沿纬线方向的拉丝', smooth: true,
                    render: (c, S) => BX.renderSphere(c, (wi, wo) => MF.ct(wi, wo, S.ax, S.ay, S.Ftab), { axis: [0, 1, 0] }),
                },
                {
                    id: 'anSphereView', label: '切线 = cross(view, n)', sub: '绕视线的同心圆（车削金属）', smooth: true,
                    render: (c, S) => BX.renderSphere(c, (wi, wo) => MF.ct(wi, wo, S.ax, S.ay, S.Ftab), { axis: [0, 0, 1] }),
                },
                {
                    id: 'anHemi', label: 'D(h)cosθ<sub>h</sub> · 各向异性', sub: '水平 = t（α<sub>x</sub>），竖直 = b（α<sub>y</sub>）· log 4', smooth: true,
                    render: (c, S) => MF.logField(c, S.hDA, MF_RES),
                    probe: MF.hemiProbe(MF_RES, (w, S) => `D·cos = ${(BX.ggxD(w, S.ax, S.ay) * w[2]).toExponential(2)}\nαx = ${S.ax.toFixed(3)}  αy = ${S.ay.toFixed(3)}`),
                },
            ],
            notes: 'aniso = 0 时两个球与 Step 05 完全一样；拉到 0.95，α<sub>x</sub>/α<sub>y</sub> ≈ 10，右图的 NDF 变成扁椭圆。左球的高光沿纬线拉长，右球的高光沿以视线为轴的圆周拉长——切线场决定了一切。',
            warn: '切线必须与法线贴图用同一个切线空间（MikkTSpace），否则拉丝方向会在 UV 接缝断开；mip 后的切线也要重新正交化。Burley 的映射在 r 较大时会让 α<sub>x</sub> &gt; 1，这里 clamp 到 1（GGX 仍可计算，但已不是有意义的"表面"）。白炉 / LUT 补偿对各向异性只能近似（常用 α = √(α<sub>x</sub>α<sub>y</sub>) 查表）。',
        },
        // ─────────────────────────── 08 ───────────────────────────
        {
            id: 'sample', title: '重要性采样：NDF 采样 vs 可见法线 VNDF（Heitz 2018）', tag: 'SAMPLE',
            desc: '离线渲染、LUT 预积分、环境预滤波都要对 f·cos 做 Monte-Carlo。经典做法按 D(h)cosθ<sub>h</sub> 抽 h、反射得 ω<sub>i</sub>——它完全不知道 ω<sub>o</sub>：掠射视角下会抽到背对视线的微表面（ω<sub>o</sub>·h &lt; 0）和反射到地平线以下的方向，这些样本全部浪费；单样本权重 G2·(ω<sub>o</sub>·h)/(cosθ<sub>o</sub>cosθ<sub>h</sub>) 还是无界的，偶尔抽到 cosθ<sub>h</sub> 很小的 h 就是一颗"萤火虫"。VNDF 按"从 ω<sub>o</sub> 看得见的法线" G1(ω<sub>o</sub>)⟨ω<sub>o</sub>·h⟩D(h)/cosθ<sub>o</sub> 抽样，权重化简为 G2/G1(ω<sub>o</sub>) ≤ 1。',
            formula: [
                'NDF : pdf(h) = D(h) cosθ_{h}                      pdf(ω_i) = D(h) cosθ_{h} / (4 |ω_o·h|)      w = F · G_2 · (ω_o·h) / (cosθ_{o} cosθ_{h})',
                'VNDF: pdf(h) = G_1(ω_o) ⟨ω_o·h⟩ D(h) / cosθ_{o}    pdf(ω_i) = G_1(ω_o) D(h) / (4 cosθ_{o})      w = F · G_2 / G_1(ω_o) ≤ 1',
                'w = f · cosθ_{i} / pdf(ω_i)        // 单样本权重；估计量 = (1/N) Σ L(ω_i) · w',
            ],
            code: {
                title: 'HLSL · BRDF 采样估计（SampleGGXNDF / SampleGGXVNDF 见 Common）', src: `
                float3 IntegrateSpecular(float3 V, float a, float3 F0, uint N, bool useVNDF) {   // 切线空间，V.z = NoV
                    float3 sum = 0.0;
                    for (uint i = 0; i < N; ++i) {
                        float2 u = Hammersley(i, N);
                        float3 H = useVNDF ? SampleGGXVNDF(V, a, a, u) : SampleGGXNDF(a, u);
                        float  VoH = dot(V, H);
                        if (VoH <= 0.0) continue;                         // NDF：背面微表面，样本作废
                        float3 L = 2.0 * VoH * H - V;
                        if (L.z <= 0.0) continue;                         // 反射到地平线下，样本作废
                        float  G2 = G2_HeightCorrelated(V.z, L.z, a);
                        float  w  = useVNDF ? G2 / G1_Smith(V.z, a)            // ≤ 1
                                            : G2 * VoH / (V.z * H.z);          // 无界
                        sum += SampleEnv(L) * F_Schlick(F0, VoH) * w;
                    }
                    return sum / N;
                }`,
            },
            previews: [
                {
                    id: 'scNDF', label: 'NDF 采样 · 256 个 ω<sub>i</sub>', sub: '红叉 = 地平线下（镜像显示）· 橙 = 权重 &gt; 2·均值 · 黄 = ω<sub>o</sub>', size: [200, 200],
                    render: (c, S) => mfScatter(c, S, 'scN'),
                },
                {
                    id: 'scVNDF', label: 'VNDF 采样 · 256 个 ω<sub>i</sub>', sub: '同一组 Hammersley 点', size: [200, 200],
                    render: (c, S) => mfScatter(c, S, 'scV'),
                },
                {
                    id: 'varPlot', label: '单样本相对标准差 σ/μ', sub: '每个 θo 512 个样本 · F ≡ 1 · logY', size: MF_PLOT,
                    render: (c, S) => {
                        let lo = Infinity, hi = 0;
                        for (let k = 0; k < S.vrN.length; k++) for (const v of [S.vrN[k], S.vrV[k]]) if (v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
                        PL.draw.plot(c, {
                            x: [0, 90], y: [Math.max(1e-3, lo * 0.7), hi * 1.5], logY: true, xLabel: 'θo（度）',
                            series: [{ data: S.vrN, x: S.vrX, color: '#e0a060', width: 2, label: 'NDF' }, { data: S.vrV, x: S.vrX, color: '#56b6c2', width: 2, label: 'VNDF' }],
                            marks: [{ x: S.theta * MF_DEG, label: 'θ', color: '#e8c25a' }],
                        });
                    },
                },
                {
                    id: 'histPlot', label: '权重分布 w / E（当前 θ）', sub: '最右一格 = 溢出（≥ 2.9）· 1024 样本', size: MF_PLOT,
                    render: (c, S) => {
                        const x = Array.from({ length: 30 }, (_, k) => (k + 0.5) / 10);
                        PL.draw.plot(c, { x: [0, 3], y: [0, 1], xLabel: 'w / E', series: [{ data: S.hgN, x, color: '#e0a060', fill: true, label: 'NDF' }, { data: S.hgV, x, color: '#56b6c2', fill: true, label: 'VNDF' }] });
                    },
                },
                {
                    id: 'nzNDF', needs: ['noise'], label: '天空光 · NDF 采样 4 spp', sub: '仅天空渐变（无 softbox）· 噪声只来自 BRDF 采样', smooth: true,
                    render: (c, S) => MF.drawBuf(c, S.nzN, MF_IBL_RES),
                },
                {
                    id: 'nzVNDF', needs: ['noise'], label: '天空光 · VNDF 采样 4 spp', sub: '同一组随机数', smooth: true,
                    render: (c, S) => MF.drawBuf(c, S.nzV, MF_IBL_RES),
                },
                {
                    id: 'nzErr', needs: ['noise'], label: '单像素相对标准差（按 μo 分桶）', sub: '两次独立 4 spp 渲染：Var = E[(X₁ − X₂)²]/2 · 左 = 轮廓，右 = 球心', size: MF_PLOT,
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, 1], xLabel: 'μo = n·v', y: [0, Math.max(...Array.from(S.nzEN).filter(isFinite), ...Array.from(S.nzEV).filter(isFinite), 0.05) * 1.15],
                        series: [{ data: S.nzEN, x: S.nzX, color: '#e0a060', width: 2, label: 'NDF 4 spp' }, { data: S.nzEV, x: S.nzX, color: '#56b6c2', width: 2, label: 'VNDF 4 spp' }],
                    }),
                },
            ],
            notes: '把 θ 拖到 80°：左图出现大量红叉与橙色大点，右图的点几乎都落在可见区且大小均匀；σ/μ 曲线在掠射角相差一个数量级以上。下排两个球只用天空渐变照明、每像素 4 spp、随机数相同，噪声只来自 BRDF 采样本身；最右的曲线按 μ<sub>o</sub> 分桶，NDF 的标准差越靠近轮廓（μ<sub>o</sub> → 0，掠射）越大，VNDF 基本持平。r 很小时两种方法几乎一样（波瓣窄、浪费少）。',
            warn: 'VNDF 的 pdf 依赖 ω<sub>o</sub>，不能预存一张"按 h 的 CDF"表；与光源采样做 MIS 时要用 pdf(ω<sub>i</sub>) = G1·D/(4cosθ<sub>o</sub>)，别混用 NDF 的 pdf。Heitz 2018 的实现要求 ω<sub>o</sub> 在上半球（透射时用 |cosθ|，见 03）。',
        },
        // ─────────────────────────── 09 ───────────────────────────
        {
            id: 'ms', title: '多次散射能量补偿：Kulla-Conty 2017 · E(μ, r) LUT', tag: 'MS',
            desc: 'Step 06 丢掉的 1 − E(μ<sub>o</sub>) 来自光在微表面之间的多次弹射。精确模拟（Heitz 2016 的随机游走）太贵；Kulla-Conty 加一个"补全波瓣" f<sub>ms</sub> ∝ (1 − E(μ<sub>o</sub>))(1 − E(μ<sub>i</sub>))：对 ω<sub>i</sub>、ω<sub>o</sub> 对称（保持互易），形状像漫反射，归一化后让白炉正好回到 1。E(μ, r) 与材质无关，烘成一张 2D LUT，E<sub>avg</sub>(r) 是 1D LUT（本页放在 PARAM-BAKE stage 里，只有 LUT 分辨率变化时才真正重烘）。有色 Fresnel 时每多弹一次就多乘一次 F，所以多次散射的颜色比单次更饱和：F<sub>ms</sub> = F<sub>avg</sub>²E<sub>avg</sub>/(1 − F<sub>avg</sub>(1 − E<sub>avg</sub>))。',
            formula: [
                'f_ms(μ_o, μ_i) = (1 − E(μ_o)) (1 − E(μ_i)) / (π (1 − E_avg))       // ∫ f_ms cosθi dωi = 1 − E(μ_o)',
                'F_ms = F_avg^2 E_avg / (1 − F_avg (1 − E_avg))                      // 几何级数：每次弹射 × F_avg、逃逸概率 E_avg',
                'f = f_ss + F_ms · f_ms                                               // Kulla-Conty 2017',
                'f ≈ f_ss · (1 + F_0 (1/E(μ_o) − 1))                                  // Turquin 2019：不加新波瓣，只放大单次波瓣',
            ],
            code: [
                { title: 'HLSL · CS_BakeEnergyLUT + CS_BakeEavg（PARAM-BAKE）', src: MF_HLSL_ELUT },
                { title: 'HLSL · 运行时补偿（Kulla-Conty / Turquin）', src: MF_HLSL_KC },
            ],
            previews: [
                {
                    id: 'elut', label: 'E(μ, r) LUT', sub: 'x = μ（0 → 1），y = r（下 0 → 上 1）· viridis 0–1',
                    render: (c, S) => PL.draw.field(c, S.Elut, S.lutN, S.lutN, { map: 'viridis', range: [0, 1], flipY: true }),
                    probe: (x, y, S) => { const N = S.lutN, j = N - 1 - y; return `μ = ${((x + 0.5) / N).toFixed(3)}  r = ${((j + 0.5) / N).toFixed(3)}\nE = ${S.Elut[j * N + x].toFixed(4)}`; },
                },
                {
                    id: 'eavgPlot', label: 'E<sub>avg</sub>(r) 与 1 − E<sub>avg</sub>', sub: '1D LUT（由 2D LUT 的每一行积分）', size: MF_PLOT,
                    render: (c, S) => {
                        const N = S.lutN, x = Array.from({ length: N }, (_, k) => (k + 0.5) / N), Ea = Array.from(S.Eavg);
                        PL.draw.plot(c, {
                            x: [0, 1], y: [0, 1.05], xLabel: 'r（感知粗糙度）',
                            series: [{ data: Ea, x, color: '#5ad07a', width: 2, label: 'E_avg' }, { data: Ea.map((v) => 1 - v), x, color: '#e06c75', label: '1 − E_avg（丢失）' }],
                            marks: [{ x: S.rough, label: `r=${S.rough.toFixed(2)} → ${S.EavgCur.toFixed(3)}`, color: '#e8c25a' }],
                        });
                    },
                },
                {
                    id: 'kcCheck', label: '白炉检验（F ≡ 1，当前 r）', sub: 'E<sub>ss</sub> 直接积分；补偿项用 LUT 插值 → 偏离 1 的部分 = LUT 误差', size: MF_PLOT,
                    render: (c, S) => {
                        const x = Array.from({ length: MF_M }, (_, k) => (k + 0.5) / MF_M);
                        PL.draw.plot(c, {
                            x: [0, 1], y: [0.4, 1.12], xLabel: 'μo = cosθo',
                            series: [
                                { data: S.EssCur, x, color: '#888', label: 'E_ss（单次）' },
                                { data: S.msKC, x, color: '#5ad07a', width: 2, label: 'E_ss + E_ms（Kulla-Conty）' },
                                { data: S.msTQ, x, color: '#6ab0e0', dash: [4, 3], width: 1.6, label: 'Turquin' },
                            ],
                            hmarks: [{ y: 1, color: '#555' }],
                        });
                    },
                },
                {
                    id: 'msSS', label: '白炉 · 仅单次散射', sub: '当前材质 Fresnel', smooth: true,
                    render: (c, S) => MF.furnace(c, (mu) => MF.lut1rgb(S.EFcur, MF_M, mu)),
                },
                {
                    id: 'msKC', label: '白炉 · + Kulla-Conty', sub: 'F<sub>ms</sub>·f<sub>ms</sub>：更亮、更饱和', smooth: true,
                    render: (c, S) => MF.furnace(c, (mu) => MF.lut1rgb(S.msKCc, MF_M, mu)),
                },
                {
                    id: 'msTQ', label: '白炉 · Turquin 缩放', sub: 'f<sub>ss</sub>·(1 + F0(1/E − 1))', smooth: true,
                    render: (c, S) => MF.furnace(c, (mu) => MF.lut1rgb(S.msTQc, MF_M, mu)),
                },
                {
                    id: 'msStrip', label: '固定 r = 0.25 / 0.5 / 0.75 / 1 · 当前材质', sub: '上排 = 仅单次散射 · 下排 = + Kulla-Conty', size: [360, 180],
                    render: (c, S) => MF.furnaceGrid(c, 2, 4, 45, (row, col, mu) => MF.lut1rgb(row ? S.msStrip[col].kc : S.msStrip[col].ss, MF_M, mu)),
                },
            ],
            notes: '材质选金、看右下条带：r 越大，下排（+KC）比上排亮得越多、颜色越"金"——多次弹射每次都乘一次 F，颜色更饱和。三个单球用当前 r：Turquin（右）的亮度与 KC 接近，但色调完全由 F0 决定、波瓣形状保持单次散射的样子。LUT 切到 8：白炉曲线在 μ → 0 处偏离 1——那是 8×8 双线性插值的误差，也是生产中 LUT 至少 32² 的原因。r &lt; 0.2 时三个球几乎一样。',
            warn: 'E LUT 必须用与运行时<strong>完全相同</strong>的 D / G（同一 α 映射、同一 G2 形式）烘焙，否则补偿过量或不足。LUT 坐标用感知粗糙度 r（分布更均匀），而积分内部用 α = r²。f<sub>ms</sub> 的"漫反射形状"在低粗糙度并不准确，但那时 1 − E 很小，可以忽略。',
        },
        // ─────────────────────────── 10 ───────────────────────────
        {
            id: 'prod', title: '教学版 → 生产版：Split-sum IBL、DFG LUT、镜面抗锯齿', tag: 'PROD',
            desc: '前面每一步都是"逐方向精确求值 / 数值积分"。实时渲染里面光源（环境光）不能每像素做几百次采样，标准做法是 Karis 2013 的 <strong>split-sum</strong>：把 ∫L·f·cos 拆成"按 GGX 波瓣预滤波的环境图"×"BRDF 自身的积分"。前者假设 n = v = r，按粗糙度烘进 mip 链；后者对 Schlick F 线性展开成 F0·A + B，A、B 只依赖 (NoV, r)，就是 DFG LUT——而 A + B = E(μ, r)，同一张 LUT 顺带给出 Step 09 的能量补偿。',
            blocks: [
                {
                    type: 'formula', label: '(A) Split-sum 近似（Karis 2013）', text: '∫ L(ω_i) f cosθ_{i} dω_{i} ≈ [ ∫ L(ω_i) D(h) cosθ_{i} dω_{i} / ∫ D(h) cosθ_{i} dω_{i} ]_{n=v=r} · ∫ f cosθ_{i} dω_{i}\n∫ f cosθ_{i} dω_{i} = F_0 · A(μ_o, r) + B(μ_o, r),   A = ∫ (1 − F_c) f_{F=1} cos,   B = ∫ F_c f_{F=1} cos,   F_c = (1 − ω_o·h)^5\nE(μ_o, r) = A + B            mip = r · (MIP_COUNT − 1)      // 预滤波 mip 与感知粗糙度线性对应',
                },
                {
                    type: 'previews', items: [
                        {
                            id: 'dfgLut', label: 'DFG LUT（R = A，G = B）', sub: 'x = NoV（0 → 1），y = r（下 0 → 上 1）· RESIDENT',
                            render: (c, S) => PL.draw.pixels(c, MF_DFG_N, MF_DFG_N, (x, y, out) => { const i = (MF_DFG_N - 1 - y) * MF_DFG_N + x; out[0] = S.dfgA[i]; out[1] = S.dfgB[i]; out[2] = 0; return out; }),
                            probe: (x, y, S) => { const N = MF_DFG_N, j = N - 1 - y, i = j * N + x; return `NoV = ${((x + 0.5) / N).toFixed(3)}  r = ${((j + 0.5) / N).toFixed(3)}\nA = ${S.dfgA[i].toFixed(4)}  B = ${S.dfgB[i].toFixed(4)}\nA + B = E = ${(S.dfgA[i] + S.dfgB[i]).toFixed(4)}`; },
                        },
                        {
                            id: 'preTiles', label: '预滤波环境 mip 链（lat-long）', sub: '左上 → 右下：r = 0, 0.2, 0.4 / 0.6, 0.8, 1.0', size: [360, 120], smooth: true,
                            render: (c, S) => PL.draw.pixels(c, 144, 48, (x, y, out) => {
                                const k = Math.floor(y / 24) * 3 + Math.floor(x / 48), u = ((x % 48) + 0.5) / 48, v = ((y % 24) + 0.5) / 24;
                                const d = BX.display(MF.sampleLL(S.pre[k], MF.envDir(u, v)));
                                out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
                            }),
                        },
                        {
                            id: 'splitSum', label: 'Split-sum：预滤波 × (F0·A + B)', sub: '每像素 2 次纹理查询', smooth: true,
                            render: (c, S) => PL.draw.pixels(c, MF_IBL_RES, MF_IBL_RES, (x, y, out) => {
                                const P = MF.spherePx(x, y, MF_IBL_RES);
                                if (!P) return MF.bgPx(y, MF_IBL_RES, out);
                                const n = P.n, NoV = n[2], R = [2 * NoV * n[0], 2 * NoV * n[1], 2 * NoV * n[2] - 1];
                                const pre = MF.samplePre(S.pre, R, S.rough), A = MF.lut2(S.dfgA, MF_DFG_N, NoV, S.rough), B = MF.lut2(S.dfgB, MF_DFG_N, NoV, S.rough);
                                const d = BX.display([0, 1, 2].map((ch) => pre[ch] * (S.F0[ch] * A + B)));
                                out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
                            }),
                        },
                        {
                            id: 'iblRef', needs: ['iblRef'], label: '参考：逐像素 VNDF 128 spp', sub: '分层采样 · 完整 D·G2·F，无 n = v = r 假设', smooth: true,
                            render: (c, S) => MF.drawBuf(c, S.iblRefBuf, MF_IBL_RES),
                        },
                    ],
                },
                { type: 'note', html: '对比右边两个球：粗糙度中等时 split-sum 与参考非常接近；差别集中在<strong>掠射边缘</strong>——参考里边缘的反射被拉长、带着正确的 Fresnel 变化，split-sum 因为 n = v = r 假设只能给出圆形波瓣。r 很小时 split-sum 在 mip 0 与 mip 1 之间插值，倒影会偏糊（mip 数越多越好）。' },
                { type: 'code', title: 'HLSL · CS_BakeDFG（RESIDENT）', src: MF_HLSL_DFG },
                { type: 'code', title: 'HLSL · CS_PrefilterEnv（RESIDENT，每个 mip 一次）', src: MF_HLSL_PREFILTER },
                {
                    type: 'formula', label: '(B) 镜面抗锯齿（Kaplanyan 2016 / Tokuyoshi & Kaplanyan 2019）', text: 'σ^2 = σ_s^2 · (|∂n/∂x|^2 + |∂n/∂y|^2)       // 像素足迹内的法线方差，σ_s² ≈ 0.25\nα\'^2 = saturate(α^2 + min(2σ^2, κ))           // κ ≈ 0.18：限幅，避免平滑曲面被过度模糊',
                },
                { type: 'code', title: 'HLSL · PS_DeferredLighting（生产版：解析光 + split-sum + 能量补偿 + 镜面 AA）', open: true, src: MF_HLSL_LIGHTING },
                {
                    type: 'warn', html: '<strong>常见坑</strong>：① α 与 r 混用——D、Λ、积分里用 α = r²，LUT 坐标和 mip 选择用 r；② <code>V_SmithGGXCorrelated</code> 已含 1/(4·NoV·NoL)，再除一次；③ 解析光的 F 用了 NoV（应为 VoH）；④ NoV ≤ 0 不 clamp → NaN 污染 TAA；⑤ fp16 下 1 − NoH² 抵消成 0（用 |N×H|²）；⑥ α → 0 不 clamp → D 爆成 inf；⑦ 能量补偿 LUT 与运行时 BRDF 用了不同的 G 形式；⑧ 预滤波采样数不足 → 高光斑点（按 pdf 选源 mip，Colbert & Křivánek 2007）；⑨ 法线贴图 mip 后方差丢失 → 远处闪烁（镜面 AA / Toksvig / LEAN）；⑩ DFG LUT 当 sRGB 纹理采样（必须是线性 RG16F）。',
                },
                { type: 'note', html: '演进方向：<strong>Filament</strong> 把 LUT 存成 (B, A + B) 并用 mix(dfg.x, dfg.y, F0) 取值，能量补偿直接用 dfg.y；<strong>UE</strong> 的 EnvBRDF 也是同一张 (A, B) 表；面光源用 LTC（Heitz 2016）替代逐方向采样；Fdez-Agüera 2019 把多次散射补偿并进 split-sum（同一张 DFG LUT 即可）；离线渲染器用 VNDF + MIS 直接积分，不需要 split-sum。透射版本（共用本篇 GGX / VNDF）见 03，E(μ) 思路在 06 分层材质里复用。' },
            ],
        },
    ],

    graph: {
        intro: '<strong>RDG 风格</strong>：BxDF 的各项（D、V、F、采样）只是被内联的函数（⊘ INLINE 节点），真正的 pass 是三个 LUT / 环境预滤波 bake 和每帧的光照 PS。E(μ, r) LUT 是 PARAM-BAKE（只随 LUT 分辨率变化），DFG 与预滤波环境是 RESIDENT；同名资源自动连线。',
        types: {
            fn: { color: '#9a7ad8', label: 'fn · 内联 HLSL 函数' },
            gbuf: { color: '#56b6c2', label: 'gbuffer · N / T / r / F0' },
            lut: { color: '#e0884a', label: 'lut · R16F / RG16F 查找表' },
            env: { color: '#4a8aff', label: 'env · RGBA16F 立方体贴图' },
            out: { color: '#5ae87a', label: 'output · SceneColor' },
        },
        common: {
            title: 'Common.hlsl · 系列共享（D_GGX / Lambda / V / F_Schlick）+ 本模块（Smith G1/G2、各向异性、采样、精确 Fresnel）',
            blocks: [
                { label: '━ 系列共享 ━', code: BX.HLSL_COMMON },
                { label: '━ 本模块 · Smith / 各向异性 GGX ━', code: MF_HLSL_SMITH },
                { label: '━ 本模块 · 采样（Hammersley / NDF / VNDF） ━', code: MF_HLSL_SAMPLING },
                { label: '━ 本模块 · 精确 Fresnel ━', code: MF_HLSL_FRESNEL },
            ],
        },
        nodes: [
            {
                id: 'gbuffer', title: 'GBuffer', tag: 'RT', category: 'system', lifetime: 'frame', desc: 'N · T · r · F0（metallic → F0）',
                outputs: [{ name: 'GBuffer', format: 'RGBA8 ×3 + D32', type: 'gbuf' }],
                detail: { desc: '引擎 base pass 写出的表面属性：法线、切线（各向异性用）、感知粗糙度 r、F0（metallic 工作流里由 baseColor / metallic 推出）。本页的参数栏相当于一个像素的 GBuffer。', preview: null },
            },
            {
                id: 'envcube', title: 'EnvironmentCube', tag: 'TEX', category: 'system', lifetime: 'resident', desc: '天空 + 两个 softbox（程序化）',
                outputs: [{ name: 'EnvCube', format: 'RGBA16F cube', type: 'env' }],
                detail: {
                    desc: '捕获的环境光（反射探针 / 天空）。本页用一个程序化环境：天空渐变 + 左上 key softbox + 右后 rim softbox，lat-long 显示。',
                    preview: { label: '源环境（lat-long）', size: [360, 180], smooth: true, render: (c, S) => PL.draw.pixels(c, 96, 48, (x, y, out) => { const d = BX.display(MF.env(MF.envDir((x + 0.5) / 96, (y + 0.5) / 48))); out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out; }) },
                },
            },
            {
                id: 'fnD', title: 'D_GGX / D_GGXAniso', tag: 'FN', step: 'ndf', desc: 'α² / (π((α²−1)NoH²+1)²)',
                dispatch: { kind: 'INLINE', name: 'D_GGX(NoH, a)', note: '被 PS_DeferredLighting 内联；各向异性版在切线空间' },
                outputs: [{ name: 'D_GGX', format: 'float(NoH, a)', type: 'fn' }],
                detail: { code: { title: 'D_GGX · D_GGXAniso', src: MF_HLSL_D } },
            },
            {
                id: 'fnV', title: 'V_SmithGGXCorrelated', tag: 'FN', step: 'smith', desc: 'G2 / (4·NoV·NoL)',
                dispatch: { kind: 'INLINE', name: 'V_SmithGGXCorrelated(NoV, NoL, a)', note: '光照 PS 与三个 bake 共用同一 G2 形式' },
                outputs: [{ name: 'V_Smith', format: 'float(NoV, NoL, a)', type: 'fn' }],
                detail: { preview: 'g2Plot' },
            },
            {
                id: 'fnF', title: 'F_Schlick / F_Conductor', tag: 'FN', step: 'fresnel', desc: 'F0 + (1−F0)(1−VoH)⁵',
                dispatch: { kind: 'INLINE', name: 'F_Schlick(F0, VoH)', note: '精确导体 Fresnel 只用于离线参考 / 拟合 F0' },
                outputs: [{ name: 'F_Fresnel', format: 'float3(F0, VoH)', type: 'fn' }],
            },
            {
                id: 'fnS', title: 'SampleGGXVNDF / NDF', tag: 'FN', step: 'sample', desc: 'h ~ G1·⟨V·H⟩·D / NoV',
                dispatch: { kind: 'INLINE', name: 'SampleGGXVNDF(V, ax, ay, u)', note: 'LUT bake 用 VNDF；预滤波沿用 Karis 的 NDF 采样（N = V）' },
                outputs: [{ name: 'SampleGGX', format: 'float3(V, a, u)', type: 'fn' }],
                detail: { code: { title: 'SampleGGXNDF / SampleGGXVNDF', src: MF_HLSL_SAMPLING }, preview: 'scVNDF' },
            },
            {
                id: 'bakeE', title: 'CS_BakeEnergyLUT', tag: 'CS', category: 'bake', lifetime: 'param', step: 'ms', desc: 'E(μ,r) = ∫ f|F=1 cos · E_avg(r)',
                dispatch: [
                    { name: 'CS_BakeEnergyLUT', threads: [8, 8, 1], groups: (S) => [Math.ceil(S.lutN / 8), Math.ceil(S.lutN / 8), 1], idRange: 'ID.xy ∈ [0, LUT_N)²', note: '每 texel 128 个 VNDF 样本（Hammersley）' },
                    { name: 'CS_BakeEavg', threads: (S) => [S.lutN, 1, 1], groups: [1, 1, 1], idRange: 'ID.x = 粗糙度行', note: 'E_avg = 2∫E μ dμ' },
                ],
                inputs: [{ name: 'SampleGGX', format: 'fn', type: 'fn' }, { name: 'V_Smith', format: 'fn', type: 'fn' }],
                outputs: [{ name: 'EnergyLUT', format: 'R16F LUT_N²', type: 'lut' }, { name: 'EavgLUT', format: 'R16F 1D', type: 'lut' }],
                detail: {
                    desc: '与材质无关，只随 LUT 分辨率变化；工程里通常离线烘好随包发布。',
                    formula: 'E(μ, r) = (1/N) Σ_k G_2(ω_o, ω_i^{(k)}) / G_1(ω_o),   h^{(k)} ~ VNDF,  α = r^2   // texel 中心 μ = (i + 0.5)/N\nE_avg(r) = 2 Σ_i E(μ_i, r) μ_i / N',
                    code: { title: 'CS_BakeEnergyLUT / CS_BakeEavg', src: MF_HLSL_ELUT }, preview: 'elut',
                },
            },
            {
                id: 'bakeDFG', title: 'CS_BakeDFG', tag: 'CS', category: 'bake', lifetime: 'resident', step: 'prod', desc: '(A, B)：F0·A + B',
                dispatch: { name: 'CS_BakeDFG', threads: [8, 8, 1], groups: [4, 4, 1], idRange: 'ID.xy ∈ [0,32)²', note: '每 texel 128 个 VNDF 样本' },
                inputs: [{ name: 'SampleGGX', format: 'fn', type: 'fn' }, { name: 'V_Smith', format: 'fn', type: 'fn' }],
                outputs: [{ name: 'DFGLut', format: 'RG16F 32²', type: 'lut' }],
                detail: { code: { title: 'CS_BakeDFG', src: MF_HLSL_DFG }, preview: 'dfgLut' },
            },
            {
                id: 'prefilter', title: 'CS_PrefilterEnv', tag: 'CS', category: 'bake', lifetime: 'resident', step: 'prod', desc: 'GGX 卷积 · N = V = R · 6 mips',
                dispatch: MF_PRE.map(([W, H], k) => ({ name: `CS_PrefilterEnv · mip ${k}（r = ${(k / 5).toFixed(1)}）`, threads: [8, 8, 1], groups: [Math.max(1, (128 >> k) / 8), Math.max(1, (128 >> k) / 8), 6], idRange: `ID.xy ∈ [0,${Math.max(4, 128 >> k)})² · z = face`, note: k ? '128 个 NDF 样本 · NoL 加权' : 'mip 0 = 源（拷贝 / 下采样）' })),
                inputs: [{ name: 'EnvCube', format: 'RGBA16F cube', type: 'env' }, { name: 'SampleGGX', format: 'fn', type: 'fn' }],
                outputs: [{ name: 'PrefilteredEnv', format: 'RGBA16F cube · 6 mips', type: 'env' }],
                detail: { desc: '生产里是 128² cube × 6 mip；本页 JS 用 48×24 → 16×8 的 lat-long 近似。', code: { title: 'CS_PrefilterEnv', src: MF_HLSL_PREFILTER }, preview: 'preTiles' },
            },
            {
                id: 'lighting', title: 'PS_DeferredLighting', tag: 'PS', category: 'output', lifetime: 'frame', step: 'prod', desc: 'CT × 补偿 + split-sum IBL',
                dispatch: { kind: 'PS', name: 'PS_DeferredLighting', target: 'SceneColor（全屏三角形）', idRange: 'SV_Position.xy ∈ viewport', note: '每像素：N 个解析光 + 2 次 IBL 查询 + 1 次 DFG 查询' },
                inputs: [
                    { name: 'GBuffer', format: 'RGBA8 ×3 + D32', type: 'gbuf' },
                    { name: 'D_GGX', format: 'fn', type: 'fn' }, { name: 'V_Smith', format: 'fn', type: 'fn' }, { name: 'F_Fresnel', format: 'fn', type: 'fn' },
                    { name: 'EnergyLUT', format: 'R16F', type: 'lut' }, { name: 'EavgLUT', format: 'R16F', type: 'lut' },
                    { name: 'DFGLut', format: 'RG16F', type: 'lut' }, { name: 'PrefilteredEnv', format: 'RGBA16F cube', type: 'env' },
                ],
                outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'out' }],
                detail: { code: { title: 'PS_DeferredLighting', src: MF_HLSL_LIGHTING }, preview: 'splitSum' },
            },
        ],
    },

    references: [
        { title: 'Illumination for Computer Generated Pictures', author: 'Bui Tuong Phong', venue: 'Communications of the ACM 18(6)', year: 1975, note: 'Phong 高光' },
        { title: 'Models of Light Reflection for Computer Synthesized Pictures', author: 'James F. Blinn', venue: 'SIGGRAPH', year: 1977, note: '半程向量；首次把 Torrance-Sparrow 微表面引入图形学' },
        { title: 'Using the Modified Phong Reflectance Model for Physically Based Rendering', author: 'Eric P. Lafortune, Yves D. Willems', venue: 'Tech. Report CW197, KU Leuven', year: 1994, note: '(n+2)/(2π) 归一化' },
        { title: 'Average Irregularity Representation of a Rough Surface for Ray Reflection', author: 'T. S. Trowbridge, K. P. Reitz', venue: 'JOSA 65(5)', year: 1975, note: '与 GGX 相同的分布' },
        { title: 'A Reflectance Model for Computer Graphics', author: 'Robert L. Cook, Kenneth E. Torrance', venue: 'ACM TOG 1(1)', year: 1982, note: 'D·G·F / (4 cos cos) 的 Cook-Torrance 形式' },
        { title: 'An Inexpensive BRDF Model for Physically-based Rendering', author: 'Christophe Schlick', venue: 'Computer Graphics Forum 13(3)', year: 1994, note: 'Schlick Fresnel 近似' },
        { title: 'Microfacet Models for Refraction through Rough Surfaces', author: 'Bruce Walter, Stephen R. Marschner, Hongsong Li, Kenneth E. Torrance', venue: 'EGSR', year: 2007, note: 'GGX、NDF 采样、Beckmann Λ 有理近似、Blinn-Phong ↔ Beckmann 映射' },
        { title: 'Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs', author: 'Eric Heitz', venue: 'JCGT 3(2)', year: 2014, url: 'https://jcgt.org/published/0003/02/03/', note: 'Smith Λ、可见投影面积守恒、高度相关 G2' },
        { title: 'Sampling the GGX Distribution of Visible Normals', author: 'Eric Heitz', venue: 'JCGT 7(4)', year: 2018, url: 'https://jcgt.org/published/0007/04/01/', note: 'VNDF 采样（BX.sampleVNDF）' },
        { title: 'Physically-Based Shading at Disney', author: 'Brent Burley', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2012, url: 'https://blog.selfshadow.com/publications/s2012-shading-course/', note: 'α = r²、各向异性 aspect 映射' },
        { title: 'Real Shading in Unreal Engine 4', author: 'Brian Karis', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2013, url: 'https://blog.selfshadow.com/publications/s2013-shading-course/', note: 'split-sum、预滤波环境、EnvBRDF LUT' },
        { title: 'Revisiting Physically Based Shading at Imageworks', author: 'Christopher Kulla, Alejandro Conty', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2017, url: 'https://blog.selfshadow.com/publications/s2017-shading-course/', note: 'E(μ) / E_avg 能量补偿、F_avg、F_ms' },
        { title: 'Practical Multiple Scattering Compensation for Microfacet Models', author: 'Emmanuel Turquin', venue: 'Industrial Light & Magic, Technical Report', year: 2019, note: '1 + F0(1/E − 1) 缩放' },
        { title: 'A Multiple-Scattering Microfacet Model for Real-Time Image-based Lighting', author: 'Carmelo J. Fdez-Agüera', venue: 'JCGT 8(1)', year: 2019, url: 'https://www.jcgt.org/published/0008/01/03/', note: '把多次散射补偿并进 split-sum IBL（复用同一张 DFG LUT）' },
        { title: 'Multiple-Scattering Microfacet BSDFs with the Smith Model', author: 'Eric Heitz, Johannes Hanika, Eugene d’Eon, Carsten Dachsbacher', venue: 'ACM TOG 35(4) (SIGGRAPH)', year: 2016, note: '多次散射随机游走参考' },
        { title: 'Improved Geometric Specular Antialiasing', author: 'Yusuke Tokuyoshi, Anton S. Kaplanyan', venue: 'I3D', year: 2019, note: 'α² + min(2σ², κ)' },
        { title: 'Artist Friendly Metallic Fresnel', author: 'Ole Gulbrandsen', venue: 'JCGT 3(4)', year: 2014, note: '(F0, edge tint) 参数化导体 Fresnel' },
        { title: 'Building an Orthonormal Basis, Revisited', author: 'Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani, Ryusuke Villemin', venue: 'JCGT 6(1)', year: 2017, note: '预滤波里的切线标架' },
        { title: 'Physically Based Rendering in Filament', author: 'Romain Guy, Mathias Agopian', venue: 'Google Filament documentation', url: 'https://google.github.io/filament/Filament.html', note: 'mediump D_GGX、DFG 存储布局、能量补偿、镜面 AA 的工程实现' },
    ],
});
