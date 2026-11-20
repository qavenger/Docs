// ════════════════════════════════════════════════════════════════════════════
// BxDF series · shared helpers — prepended to every module at build time:
//   node <skill>/scripts/build_doc.mjs bxdf_common.js bxdf_0N_xxx.lab.js -o bxdf_0N_xxx.html
//
// Conventions shared by every module (formulas, HLSL and JS must all follow these):
//   • Local shading frame: n = +Z, tangent t = +X, bitangent b = +Y; θ from +Z, φ from +X toward +Y.
//   • wi points toward the light, wo toward the viewer — both point AWAY from the surface.
//   • A BxDF is f(wi, wo) in 1/sr returning a number or linear RGB [r,g,b]. A directional light of
//     irradiance E (measured at normal incidence) reflects L = f · E · cosθi.
//   • Perceptual roughness r ∈ [0,1]; GGX α = r² (Burley 2012). Anisotropy uses αx (along t), αy (along b).
//   • Relative IOR η = η_t / η_i (transmitted side over incident side); F0 = ((η−1)/(η+1))².
//   • Linear RGB everywhere; BX.display() (exposure → ACES fit → sRGB) is applied only for display.
//   • View-space for sphere renders: x right, y up, z toward the viewer.
// ════════════════════════════════════════════════════════════════════════════
const BX = (() => {
    const PI = Math.PI, INV_PI = 1 / Math.PI;
    const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
    const isRGB = Array.isArray;

    // ── directions & local-frame trigonometry ──
    const dir = (theta, phi = 0) => [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
    const cosTheta = (w) => w[2];
    const sin2Theta = (w) => Math.max(0, 1 - w[2] * w[2]);
    const tan2Theta = (w) => sin2Theta(w) / Math.max(1e-12, w[2] * w[2]);
    const cosPhi = (w) => { const s = Math.sqrt(sin2Theta(w)); return s < 1e-9 ? 1 : clamp(w[0] / s, -1, 1); };
    const sinPhi = (w) => { const s = Math.sqrt(sin2Theta(w)); return s < 1e-9 ? 0 : clamp(w[1] / s, -1, 1); };
    const half = (wi, wo) => norm([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
    const reflect = (w, n = [0, 0, 1]) => { const k = 2 * dot(w, n); return [k * n[0] - w[0], k * n[1] - w[1], k * n[2] - w[2]]; };
    /** Refraction. wi away from the surface on the incident side, n on the same side as wi, eta = η_t/η_i.
     *  Returns wt (away from the surface, on the other side) or null on total internal reflection. */
    const refract = (wi, n, eta) => {
        const cosI = dot(n, wi), sin2T = Math.max(0, 1 - cosI * cosI) / (eta * eta);
        if (sin2T >= 1) return null;
        const cosT = Math.sqrt(1 - sin2T), k = cosI / eta - cosT;
        return [-wi[0] / eta + k * n[0], -wi[1] / eta + k * n[1], -wi[2] / eta + k * n[2]];
    };

    // ── Fresnel ──
    const f0FromIor = (eta) => ((eta - 1) / (eta + 1)) ** 2;
    const pow5 = (x) => { const x2 = x * x; return x2 * x2 * x; };
    /** Schlick: F0 number or RGB; c = cos of the angle to the (micro)normal */
    const schlick = (F0, c) => { const m = pow5(clamp(1 - c, 0, 1)); return isRGB(F0) ? F0.map((f) => f + (1 - f) * m) : F0 + (1 - F0) * m; };
    /** Exact unpolarized dielectric Fresnel. cosI < 0 means arriving from the other side (η is inverted). TIR → 1. */
    const fresnelDielectric = (cosI, eta) => {
        cosI = clamp(cosI, -1, 1);
        if (cosI < 0) { eta = 1 / eta; cosI = -cosI; }
        const sin2T = (1 - cosI * cosI) / (eta * eta);
        if (sin2T >= 1) return 1;
        const cosT = Math.sqrt(1 - sin2T);
        const rPar = (eta * cosI - cosT) / (eta * cosI + cosT), rPerp = (cosI - eta * cosT) / (cosI + eta * cosT);
        return (rPar * rPar + rPerp * rPerp) / 2;
    };
    // complex helpers for conductors
    const cAdd = (a, b) => [a[0] + b[0], a[1] + b[1]], cSub = (a, b) => [a[0] - b[0], a[1] - b[1]];
    const cMul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    const cDiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
    const cSqrt = (a) => { const r = Math.hypot(a[0], a[1]); const re = Math.sqrt(Math.max(0, (r + a[0]) / 2)), im = Math.sqrt(Math.max(0, (r - a[0]) / 2)); return [re, a[1] < 0 ? -im : im]; };
    const cNorm = (a) => a[0] * a[0] + a[1] * a[1];
    /** Exact conductor Fresnel per channel; eta, k are RGB arrays (complex IOR η + ik). Returns RGB. */
    const fresnelConductor = (cosI, eta, k) => [0, 1, 2].map((c) => {
        const ci = clamp(cosI, 0, 1), e = [eta[c], k[c]];
        const sin2T = cDiv([1 - ci * ci, 0], cMul(e, e)), cosT = cSqrt(cSub([1, 0], sin2T));
        const ec = [e[0] * ci, e[1] * ci];
        const rPar = cDiv(cSub(ec, cosT), cAdd(ec, cosT));
        const ecT = cMul(e, cosT), rPerp = cDiv(cSub([ci, 0], ecT), cAdd([ci, 0], ecT));
        return (cNorm(rPar) + cNorm(rPerp)) / 2;
    });
    /** Complex IOR presets at ≈ (650, 550, 450) nm — RGB fits commonly used with pbrt/Mitsuba scenes */
    const METALS = {
        gold: { eta: [0.143, 0.375, 1.442], k: [3.983, 2.386, 1.603] },
        copper: { eta: [0.200, 0.924, 1.102], k: [3.913, 2.453, 2.142] },
        aluminum: { eta: [1.657, 0.880, 0.521], k: [9.224, 6.270, 4.837] },
        silver: { eta: [0.155, 0.117, 0.138], k: [4.828, 3.122, 2.147] },
    };

    // ── GGX / Trowbridge-Reitz microfacets (local frame, h·n = h[2]) ──
    const alpha = (r) => Math.max(1e-4, r * r);
    const ggxD = (h, ax, ay = ax) => {
        if (h[2] <= 0) return 0;
        const t = (h[0] / ax) ** 2 + (h[1] / ay) ** 2 + h[2] * h[2];
        return 1 / (PI * ax * ay * t * t);
    };
    /** Smith Λ(w) for (anisotropic) GGX; uses |cosθ| so it also works below the surface */
    const ggxLambda = (w, ax, ay = ax) => {
        const c2 = w[2] * w[2];
        if (c2 <= 0) return Infinity;   // exact horizon: G → 0. Near it Λ ~ α·tanθ/2, so D·G2/(4 cos cos) stays finite
        return (Math.sqrt(1 + ((w[0] * ax) ** 2 + (w[1] * ay) ** 2) / c2) - 1) / 2;
    };
    const smithG1 = (w, ax, ay = ax) => 1 / (1 + ggxLambda(w, ax, ay));
    /** Height-correlated masking-shadowing G2 (Heitz 2014) */
    const smithG2 = (wo, wi, ax, ay = ax) => 1 / (1 + ggxLambda(wo, ax, ay) + ggxLambda(wi, ax, ay));
    /** Visible-normal sampling (Heitz 2018, JCGT 7(4)). wo in the upper hemisphere; returns microfacet normal h. */
    const sampleVNDF = (wo, ax, ay, u1, u2) => {
        const vh = norm([ax * wo[0], ay * wo[1], wo[2]]);
        const lensq = vh[0] * vh[0] + vh[1] * vh[1];
        const t1 = lensq > 0 ? [-vh[1] / Math.sqrt(lensq), vh[0] / Math.sqrt(lensq), 0] : [1, 0, 0];
        const t2 = cross(vh, t1);
        const r = Math.sqrt(u1), ph = 2 * PI * u2;
        const p1 = r * Math.cos(ph);
        let p2 = r * Math.sin(ph);
        const s = 0.5 * (1 + vh[2]);
        p2 = (1 - s) * Math.sqrt(Math.max(0, 1 - p1 * p1)) + s * p2;
        const pz = Math.sqrt(Math.max(0, 1 - p1 * p1 - p2 * p2));
        const nh = [p1 * t1[0] + p2 * t2[0] + pz * vh[0], p1 * t1[1] + p2 * t2[1] + pz * vh[1], p1 * t1[2] + p2 * t2[2] + pz * vh[2]];
        return norm([ax * nh[0], ay * nh[1], Math.max(1e-6, nh[2])]);
    };
    /** Density of h under VNDF sampling: G1(wo) · max(0, wo·h) · D(h) / cosθo */
    const pdfVNDF = (wo, h, ax, ay = ax) => smithG1(wo, ax, ay) * Math.max(0, dot(wo, h)) * ggxD(h, ax, ay) / Math.max(1e-9, wo[2]);
    /** Classic NDF sampling (isotropic): pdf(h) = D(h)·cosθh */
    const sampleNDF = (a, u1, u2) => { const t = Math.atan(a * Math.sqrt(u1 / Math.max(1e-12, 1 - u1))); return dir(t, 2 * PI * u2); };

    // ── hemisphere sampling & low-discrepancy points ──
    const sampleCosine = (u1, u2) => { const r = Math.sqrt(u1), p = 2 * PI * u2; return [r * Math.cos(p), r * Math.sin(p), Math.sqrt(Math.max(0, 1 - u1))]; };   // pdf = cosθ/π
    const sampleUniform = (u1, u2) => { const z = u1, r = Math.sqrt(Math.max(0, 1 - z * z)), p = 2 * PI * u2; return [r * Math.cos(p), r * Math.sin(p), z]; };   // pdf = 1/(2π)
    const radicalInverse2 = (i) => { let b = i >>> 0; b = ((b << 16) | (b >>> 16)) >>> 0; b = (((b & 0x55555555) << 1) | ((b & 0xAAAAAAAA) >>> 1)) >>> 0; b = (((b & 0x33333333) << 2) | ((b & 0xCCCCCCCC) >>> 2)) >>> 0; b = (((b & 0x0F0F0F0F) << 4) | ((b & 0xF0F0F0F0) >>> 4)) >>> 0; b = (((b & 0x00FF00FF) << 8) | ((b & 0xFF00FF00) >>> 8)) >>> 0; return b / 4294967296; };
    const hammersley = (i, n) => [(i + 0.5) / n, radicalInverse2(i)];

    // ── integration ──
    const acc = (s, v, w) => { if (isRGB(v)) { if (!isRGB(s)) s = [s, s, s]; return [s[0] + v[0] * w, s[1] + v[1] * w, s[2] + v[2] * w]; } return isRGB(s) ? [s[0] + v * w, s[1] + v * w, s[2] + v * w] : s + v * w; };
    /** Directional albedo E(wo) = ∫ f(wi, wo) cosθi dωi by midpoint quadrature in (cosθ, φ). Smooth lobes only. */
    const albedoQuad = (f, wo, n = 48) => {
        let s = 0;
        const dw = (1 / n) * (2 * PI / (2 * n));
        for (let i = 0; i < n; i++) {
            const c = (i + 0.5) / n, st = Math.sqrt(1 - c * c);
            for (let j = 0; j < 2 * n; j++) { const p = (j + 0.5) / (2 * n) * 2 * PI; s = acc(s, f([st * Math.cos(p), st * Math.sin(p), c], wo), c * dw); }
        }
        return s;
    };
    const radicalInverse3 = (i) => { let f = 1, r = 0; while (i > 0) { f /= 3; r += f * (i % 3); i = Math.floor(i / 3); } return r; };
    /** Monte Carlo albedo with your sampler: sample(wo, u1, u2, u3) → { wi, weight } where weight = f·cosθi/pdf (0 if invalid).
     *  (u1, u2) are Hammersley points; u3 is a base-3 radical inverse, e.g. for choosing reflection vs transmission. */
    const albedoMC = (sample, wo, n = 256) => {
        let s = 0;
        for (let i = 0; i < n; i++) { const [u1, u2] = hammersley(i, n), r = sample(wo, u1, u2, radicalInverse3(i)); if (r && r.weight) s = acc(s, r.weight, 1 / n); }
        return s;
    };

    // ── hemisphere images (equal-area projection: disk radius² = 1 − cosθ) ──
    const hemiDir = (x, y, res) => {
        const u = 2 * (x + 0.5) / res - 1, v = -(2 * (y + 0.5) / res - 1), r2 = u * u + v * v;
        if (r2 > 1) return null;
        const c = 1 - r2, s = Math.sqrt(Math.max(0, 1 - c * c)), r = Math.sqrt(r2);
        return r < 1e-9 ? [0, 0, 1] : [u / r * s, v / r * s, c];
    };
    const hemiPixel = (w, res) => {   // direction → [x, y] pixel (inverse of hemiDir)
        const r = Math.sqrt(Math.max(0, 1 - Math.abs(w[2]))), s = Math.hypot(w[0], w[1]) || 1;
        return [((w[0] / s) * r + 1) / 2 * res - 0.5, (1 - (w[1] / s) * r) / 2 * res - 0.5];
    };
    /** res×res Float32Array of fn(w) over the upper hemisphere; NaN outside the disk (draws black) */
    const hemiImage = (fn, res, out) => {
        out = out && out.length === res * res ? out : new Float32Array(res * res);
        for (let y = 0; y < res; y++) for (let x = 0; x < res; x++) { const w = hemiDir(x, y, res); out[y * res + x] = w ? fn(w) : NaN; }
        return out;
    };

    // ── display ──
    const aces = (x) => clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1);   // Narkowicz 2015 fit
    const toSrgb = (x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055);
    const display = (rgb, exposure = 1) => [toSrgb(aces(rgb[0] * exposure)), toSrgb(aces(rgb[1] * exposure)), toSrgb(aces(rgb[2] * exposure))];
    const hex2lin = (hex) => [1, 3, 5].map((i) => { const c = parseInt(hex.slice(i, i + 2), 16) / 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); });
    const lum = (c) => (isRGB(c) ? 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] : c);

    // ── lit sphere render (orthographic, view along −z) ──
    const LIGHTS = {
        studio: [
            { dir: norm([-0.55, 0.6, 0.58]), color: [2.8, 2.7, 2.55] },     // key, upper left front
            { dir: norm([0.85, 0.15, -0.5]), color: [1.2, 1.25, 1.4] },     // rim, right behind
            { dir: norm([0.6, -0.35, 0.72]), color: [0.25, 0.27, 0.32] },   // fill, lower right
        ],
        key: [{ dir: norm([-0.55, 0.6, 0.58]), color: [3, 3, 3] }],
    };
    /** f(wi, wo, info) with LOCAL directions; info = { n, t, b (view space), x, y }. o: { res, lights, axis (tangent
     *  direction for anisotropy, default up), exposure, ambient(wo, info) → rgb radiance, bg (linear RGB radiance or
     *  (x, y) → rgb, shown through the same display transform — e.g. [1,1,1] for a white-furnace backdrop) }. Returns max radiance. */
    const renderSphere = (canvas, f, o = {}) => {
        const res = o.res || 112, lights = o.lights || LIGHTS.studio, axis = o.axis || [0, 1, 0], ex = o.exposure || 1;
        let maxL = 0;
        const info = { n: null, t: null, b: null, x: 0, y: 0 };
        PL.draw.pixels(canvas, res, res, (x, y, out) => {
            const u = 2 * (x + 0.5) / res - 1, v = -(2 * (y + 0.5) / res - 1), r2 = u * u + v * v;
            if (r2 > 1) {
                if (o.bg) { const b = typeof o.bg === 'function' ? o.bg(x, y) : o.bg, d = display(b, ex); out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out; }
                const g = 0.035 + 0.03 * (1 - (y / res)); out[0] = g * 0.8; out[1] = g * 0.9; out[2] = g * 1.15; return out;
            }
            const n = [u, v, Math.sqrt(1 - r2)];
            let t = cross(axis, n);
            if (Math.hypot(t[0], t[1], t[2]) < 1e-4) t = [1, 0, 0];
            t = norm(t);
            const b = cross(n, t), toL = (w) => [dot(w, t), dot(w, b), dot(w, n)];
            const wo = toL([0, 0, 1]);
            info.n = n; info.t = t; info.b = b; info.x = x; info.y = y;
            let L = [0, 0, 0];
            for (const l of lights) {
                const wi = toL(l.dir);
                if (wi[2] <= 0 && !o.twoSided) continue;
                const fv = f(wi, wo, info), c = Math.abs(wi[2]);
                L = isRGB(fv) ? [L[0] + fv[0] * l.color[0] * c, L[1] + fv[1] * l.color[1] * c, L[2] + fv[2] * l.color[2] * c]
                    : [L[0] + fv * l.color[0] * c, L[1] + fv * l.color[1] * c, L[2] + fv * l.color[2] * c];
            }
            if (o.ambient) { const a = o.ambient(wo, info); L = [L[0] + a[0], L[1] + a[1], L[2] + a[2]]; }
            maxL = Math.max(maxL, lum(L));
            const d = display(L, ex);
            out[0] = d[0]; out[1] = d[1]; out[2] = d[2];
            return out;
        });
        return maxL;
    };

    // ── polar lobe plot in the plane of incidence (local x–z plane) ──
    /** o: { wi (in x–z plane), series:[{ fn(wo) → number, color, label, dash }], cos (multiply by cosθo), log (decades,
     *  e.g. 3), scale (fixed max), full (also plot below the surface, for BTDFs), title,
     *  marks: [{ dir: [x, 0, z], color, label, dash }] extra direction rays (e.g. the smooth refraction direction) } */
    const lobe = (canvas, o) => {
        const { ctx, w, h } = PL.draw.ctx(canvas);
        const full = !!o.full, cx = w / 2, cy = full ? h / 2 : h - 18, R = full ? Math.min(w, h) / 2 - 14 : Math.min(w / 2 - 12, h - 30);
        const N = full ? 361 : 181, t0 = full ? -PI : -PI / 2, t1 = full ? PI : PI / 2;
        const curves = (o.series || []).map((s) => {
            const pts = [];
            for (let k = 0; k < N; k++) {
                const t = t0 + (t1 - t0) * k / (N - 1), wo = [Math.sin(t), 0, Math.cos(t)];
                let v = s.fn(wo);
                v = isRGB(v) ? lum(v) : v;
                if (o.cos) v *= Math.abs(wo[2]);
                pts.push([t, isFinite(v) ? Math.max(0, v) : 0]);
            }
            return { s, pts };
        });
        let vmax = o.scale || 0;
        if (!vmax) curves.forEach((c) => c.pts.forEach((p) => { vmax = Math.max(vmax, p[1]); }));
        vmax = vmax || 1;
        const dec = o.log ? (o.log === true ? 3 : o.log) : 0;
        const rad = (v) => (dec ? Math.max(0, (Math.log10(Math.max(v, 1e-30) / vmax) + dec) / dec) : v / vmax);
        ctx.strokeStyle = '#1d2329'; ctx.lineWidth = 1;
        for (const q of [0.25, 0.5, 0.75, 1]) { ctx.beginPath(); ctx.arc(cx, cy, R * q, full ? 0 : PI, 2 * PI); ctx.stroke(); }
        for (const a of [30, 60, 120, 150, 210, 240, 300, 330]) {
            if (!full && (a < 180 || a > 360)) continue;
            const r = a * PI / 180; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(r), cy + R * Math.sin(r)); ctx.stroke();
        }
        ctx.strokeStyle = '#3a4148'; ctx.beginPath(); ctx.moveTo(cx - R - 6, cy); ctx.lineTo(cx + R + 6, cy); ctx.stroke();
        ctx.strokeStyle = '#4a5560'; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - R - 4); ctx.stroke(); ctx.setLineDash([]);
        if (o.wi) {
            const wi = o.wi, L = R * 1.0;
            ctx.strokeStyle = '#e8c25a'; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(cx + wi[0] * L, cy - wi[2] * L); ctx.lineTo(cx, cy); ctx.stroke();
            ctx.fillStyle = '#e8c25a'; ctx.beginPath(); ctx.arc(cx + wi[0] * L, cy - wi[2] * L, 3, 0, 2 * PI); ctx.fill();
            ctx.strokeStyle = 'rgba(232,194,90,0.45)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx - wi[0] * L, cy - wi[2] * L); ctx.stroke(); ctx.setLineDash([]);
        }
        (o.marks || []).forEach((m) => {
            const d = m.dir, L = R * 1.02;
            ctx.strokeStyle = m.color || '#9aa5b1'; ctx.lineWidth = 1; ctx.setLineDash(m.dash || [3, 3]);
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + d[0] * L, cy - d[2] * L); ctx.stroke(); ctx.setLineDash([]);
            if (m.label) { ctx.fillStyle = m.color || '#9aa5b1'; ctx.font = '10px Consolas, monospace'; ctx.textAlign = d[0] >= 0 ? 'left' : 'right'; ctx.textBaseline = 'middle'; ctx.fillText(m.label, cx + d[0] * L + (d[0] >= 0 ? 4 : -4), cy - d[2] * L); }
        });
        curves.forEach(({ s, pts }, k) => {
            ctx.strokeStyle = s.color || ['#6ab0e0', '#e0a060', '#5ad07a', '#c678dd', '#e06c75'][k % 5]; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash || []);
            ctx.beginPath();
            pts.forEach(([t, v], i) => { const r = R * rad(v), X = cx + r * Math.sin(t), Y = cy - r * Math.cos(t); if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); });
            ctx.stroke(); ctx.setLineDash([]);
        });
        ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
        let ly = 4;
        if (o.title) { ctx.fillStyle = '#888'; ctx.fillText(o.title, 6, ly); ly += 13; }
        curves.forEach(({ s }, k) => { if (!s.label) return; ctx.fillStyle = s.color || ['#6ab0e0', '#e0a060', '#5ad07a', '#c678dd', '#e06c75'][k % 5]; ctx.fillText('— ' + s.label, 6, ly); ly += 13; });
        ctx.fillStyle = '#555'; ctx.textAlign = 'right';
        ctx.fillText(dec ? `log, ${dec} decades` : `max ${vmax.toPrecision(3)}`, w - 6, 4);
        return vmax;
    };

    // ── series navigation ──
    const MODULES = [
        { id: 'diffuse', no: '01', file: 'bxdf_01_diffuse.html', title: '漫反射：Lambert → Oren-Nayar → Burley', short: '漫反射' },
        { id: 'microfacet', no: '02', file: 'bxdf_02_microfacet.html', title: '微表面镜面：Cook-Torrance · GGX', short: '微表面镜面' },
        { id: 'transmission', no: '03', file: 'bxdf_03_transmission.html', title: '粗糙透射 BTDF（Walter 2007）', short: '透射 BTDF' },
        { id: 'sheen', no: '04', file: 'bxdf_04_sheen.html', title: '织物 Sheen：Charlie / Ashikhmin', short: '织物 Sheen' },
        { id: 'subsurface', no: '05', file: 'bxdf_05_subsurface.html', title: '次表面散射 BSSRDF：Dipole → Normalized Diffusion', short: '次表面散射' },
        { id: 'layering', no: '06', file: 'bxdf_06_layering.html', title: '多层混合与叠加：Clearcoat → 分层 BSDF → 材质总装', short: '多层混合' },
    ];
    const nav = (id) => {
        const i = MODULES.findIndex((m) => m.id === id), p = MODULES[i - 1], q = MODULES[i + 1];
        return `<p style="margin-top:8px;font-size:0.74rem;color:#789">BxDF 系列 · <a href="index.html">索引</a>` +
            (p ? ` · ← <a href="${p.file}">${p.no} ${p.short}</a>` : '') + (q ? ` · <a href="${q.file}">${q.no} ${q.short}</a> →` : '') + '</p>';
    };

    // ── HLSL shared by every module's Common panel ──
    const HLSL_COMMON = `
        // BxDF 系列 · Common.hlsl —— 局部着色坐标系 n = +Z；世界空间代码只用与 N 的点积
        static const float PI     = 3.14159265;
        static const float INV_PI = 0.31830989;

        float  Pow5(float x)          { float x2 = x * x; return x2 * x2 * x; }
        float  F0FromIor(float eta)   { float r = (eta - 1.0) / (eta + 1.0); return r * r; }
        float3 F_Schlick(float3 F0, float VoH) { return F0 + (1.0 - F0) * Pow5(1.0 - saturate(VoH)); }

        // Trowbridge-Reitz / GGX，a = roughness²
        float D_GGX(float NoH, float a) {
            float a2 = a * a;
            float d  = (NoH * a2 - NoH) * NoH + 1.0;
            return a2 / (PI * d * d);
        }
        float Lambda_GGX(float NoX, float a) {
            float a2 = a * a, c2 = max(NoX * NoX, 1e-7);
            return 0.5 * (sqrt(1.0 + a2 * (1.0 - c2) / c2) - 1.0);
        }
        // 高度相关 Smith：V = G2 / (4·NoV·NoL)
        float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
            float a2 = a * a;
            float gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
            float gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
            return 0.5 / (gv + gl);
        }`;

    return {
        PI, INV_PI, clamp, dot, cross, norm, dir, cosTheta, sin2Theta, tan2Theta, cosPhi, sinPhi, half, reflect, refract,
        f0FromIor, pow5, schlick, fresnelDielectric, fresnelConductor, METALS,
        alpha, ggxD, ggxLambda, smithG1, smithG2, sampleVNDF, pdfVNDF, sampleNDF,
        sampleCosine, sampleUniform, hammersley, albedoQuad, albedoMC,
        hemiDir, hemiPixel, hemiImage, aces, toSrgb, display, hex2lin, lum,
        LIGHTS, renderSphere, lobe, MODULES, nav, HLSL_COMMON,
    };
})();
