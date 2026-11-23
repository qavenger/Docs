// ════════════════════════════════════════════════════════════════════════════
// BxDF 03 · 粗糙透射 BTDF（Walter 2007）
// build: node <skill>/scripts/build_doc.mjs bxdf_common.js bxdf_03_transmission.lab.js -o bxdf_03_transmission.html
//
// 本模块约定（在 bxdf_common.js 的全系列约定之上）：
//   • 局部坐标 n = +Z。上半球 (z > 0) = 外部空气 η_ext = 1；下半球 = 介质内部 η_int = ior。
//   • wi → 光源，wo → 相机，二者都背离表面；η_i / η_o = wi / wo 所在一侧的 IOR。
//   • 透射半向量 h_t = −(η_i·wi + η_o·wo) 归一化后再翻到 +n 侧（ior > 1 时 Walter 原式本来就在 +n 侧）。
//   • f_t 采用 Walter Eq.21 的 radiance 模式（分子 η_o²）；importance 模式 = 乘 (η_i/η_o)²。
//   • G2 用全系列共享的高度相关 Smith 1/(1+Λi+Λo)（反射、透射同一式，与 pbrt-v4 一致）。
// ════════════════════════════════════════════════════════════════════════════

const TR = (() => {
    const PI = Math.PI;
    const { dot, norm, clamp } = BX;
    const neg = (v) => [-v[0], -v[1], -v[2]];
    const etaOf = (w, ior) => (w[2] >= 0 ? 1 : ior);

    /** Walter Eq.16 generalized half vector, face-forwarded to +n. null when degenerate. */
    const halfT = (wi, wo, ior) => {
        const ei = etaOf(wi, ior), eo = etaOf(wo, ior);
        const x = -(ei * wi[0] + eo * wo[0]), y = -(ei * wi[1] + eo * wo[1]), z = -(ei * wi[2] + eo * wo[2]);
        const l = Math.hypot(x, y, z);
        if (l < 1e-9) return null;
        const s = (z < 0 ? -1 : 1) / l;
        return [x * s, y * s, z * s];
    };
    const halfR = (wi, wo) => {
        const x = wi[0] + wo[0], y = wi[1] + wo[1], z = wi[2] + wo[2], l = Math.hypot(x, y, z);
        if (l < 1e-9) return null;
        const s = (z < 0 ? -1 : 1) / l;
        return [x * s, y * s, z * s];
    };
    /** dω_h / dω_o for transmission (Walter Eq.17) — pure geometry, no validity test */
    const jacobT = (wi, wo, ior) => {
        const h = halfT(wi, wo, ior);
        if (!h) return NaN;
        const ei = etaOf(wi, ior), eo = etaOf(wo, ior), d = ei * dot(wi, h) + eo * dot(wo, h);
        return eo * eo * Math.abs(dot(wo, h)) / (d * d);
    };
    /** Every factor of the Walter BTDF for (wi, wo) on opposite sides; null if invalid */
    const termsT = (wi, wo, a, ior) => {
        const ci = wi[2], co = wo[2];
        if (ci * co >= 0) return null;
        const h = halfT(wi, wo, ior);
        if (!h) return null;
        const ih = dot(wi, h), oh = dot(wo, h);
        if (ih * ci <= 0 || oh * co <= 0) return null;          // χ⁺: microfacet back-facing to one of them
        const ei = etaOf(wi, ior), eo = etaOf(wo, ior), denom = ei * ih + eo * oh;
        const F = BX.fresnelDielectric(ih, ior);
        const D = BX.ggxD(h, a), G = BX.smithG2(wo, wi, a);
        const J = eo * eo * Math.abs(oh) / (denom * denom);     // dω_h/dω_o
        const proj = Math.abs(ih) / Math.abs(ci * co);          // |i·h| / (|i·n||o·n|)
        return { h, ih, oh, ei, eo, denom, F, D, G, J, proj, f: proj * J * (1 - F) * D * G };
    };
    const evalT = (wi, wo, a, ior, mode) => {
        const t = termsT(wi, wo, a, ior);
        if (!t) return 0;
        return mode === 'importance' ? t.f * (t.ei * t.ei) / (t.eo * t.eo) : t.f;
    };
    const evalR = (wi, wo, a, ior) => {
        const ci = wi[2], co = wo[2];
        if (ci * co <= 0) return 0;
        const h = halfR(wi, wo);
        if (!h) return 0;
        const ih = dot(wi, h);
        if (ih * ci <= 0 || dot(wo, h) * co <= 0) return 0;
        return BX.fresnelDielectric(ih, ior) * BX.ggxD(h, a) * BX.smithG2(wo, wi, a) / (4 * Math.abs(ci * co));
    };
    const evalF = (wi, wo, a, ior, mode) => evalR(wi, wo, a, ior) + evalT(wi, wo, a, ior, mode);

    /** pbrt-v4 style sampler: given the fixed direction wo, generate wi. u0 picks R/T, (u1,u2) drive VNDF. */
    const sample = (wo, a, ior, u0, u1, u2, mode) => {
        const co = wo[2];
        if (Math.abs(co) < 1e-7) return null;
        const s = co > 0 ? 1 : -1, wos = [wo[0] * s, wo[1] * s, wo[2] * s];
        const h = BX.sampleVNDF(wos, a, a, u1, u2);
        const oh = dot(wo, h), F = BX.fresnelDielectric(oh, ior);
        const pdfH = BX.pdfVNDF(wos, h, a), G1o = BX.smithG1(wo, a);
        if (u0 < F) {
            const wi = [2 * oh * h[0] - wo[0], 2 * oh * h[1] - wo[1], 2 * oh * h[2] - wo[2]];
            if (wi[2] * co <= 0) return { wi, isT: false, weight: 0, pdf: 0, h, F };
            return { wi, isT: false, weight: BX.smithG2(wo, wi, a) / G1o, pdf: pdfH * F / (4 * Math.abs(oh)), h, F };
        }
        const eo = etaOf(wo, ior), ei = co > 0 ? ior : 1;
        const wi = BX.refract(wo, oh >= 0 ? h : neg(h), ei / eo);
        if (!wi || wi[2] * co >= 0) return { wi: wi || [0, 0, 0], isT: true, weight: 0, pdf: 0, h, F };
        const ih = dot(wi, h), denom = ei * ih + eo * oh;
        let weight = BX.smithG2(wo, wi, a) / G1o;
        if (mode !== 'importance') weight *= (eo * eo) / (ei * ei);
        return { wi, isT: true, weight, pdf: pdfH * (1 - F) * ei * ei * Math.abs(ih) / (denom * denom), h, F };
    };
    /** Density of `sample` over wi (solid angle) */
    const pdf = (wi, wo, a, ior) => {
        const ci = wi[2], co = wo[2];
        if (Math.abs(co) < 1e-7 || Math.abs(ci) < 1e-9) return 0;
        const s = co > 0 ? 1 : -1, wos = [wo[0] * s, wo[1] * s, wo[2] * s];
        if (ci * co > 0) {
            const h = halfR(wi, wo);
            if (!h) return 0;
            const oh = dot(wo, h);
            if (oh * co <= 0 || dot(wi, h) * ci <= 0) return 0;
            return BX.pdfVNDF(wos, h, a) * BX.fresnelDielectric(oh, ior) / (4 * Math.abs(oh));
        }
        const t = termsT(wi, wo, a, ior);
        if (!t) return 0;
        return BX.pdfVNDF(wos, t.h, a) * (1 - t.F) * t.ei * t.ei * Math.abs(t.ih) / (t.denom * t.denom);
    };
    /** Flux split for light arriving from w: [E_R, E_T] — both branches per VNDF sample (no Fresnel coin flip) */
    const albedoSplit = (w, a, ior, n) => {
        const co = w[2], s = co > 0 ? 1 : -1, ws = [w[0] * s, w[1] * s, w[2] * s];
        const G1 = BX.smithG1(w, a), eo = etaOf(w, ior), ei = co > 0 ? ior : 1;
        let R = 0, T = 0;
        for (let k = 0; k < n; k++) {
            const u = BX.hammersley(k, n), h = BX.sampleVNDF(ws, a, a, u[0], u[1]);
            const oh = dot(w, h), F = BX.fresnelDielectric(oh, ior);
            const wr = [2 * oh * h[0] - w[0], 2 * oh * h[1] - w[1], 2 * oh * h[2] - w[2]];
            if (wr[2] * co > 0) R += F * BX.smithG2(w, wr, a) / G1;
            if (F < 1) {
                const wt = BX.refract(w, oh >= 0 ? h : neg(h), ei / eo);
                if (wt && wt[2] * co < 0) T += (1 - F) * BX.smithG2(w, wt, a) / G1;
            }
        }
        return [R / n, T / n];
    };
    /** s / p polarized reflectance (cosI ≥ 0, eta = η_t/η_i) — teaching plot only */
    const fresnelSP = (cosI, eta) => {
        const sin2T = (1 - cosI * cosI) / (eta * eta);
        if (sin2T >= 1) return [1, 1];
        const cosT = Math.sqrt(1 - sin2T);
        const rs = (cosI - eta * cosT) / (cosI + eta * cosT), rp = (eta * cosI - cosT) / (eta * cosI + cosT);
        return [rs * rs, rp * rp];
    };
    /** Map a res×res equal-area image onto the TRANSMISSION hemisphere of wi (opposite side) */
    const transImage = (fn, wi, res, out) => BX.hemiImage((w) => fn(wi[2] >= 0 ? [w[0], w[1], -w[2]] : w), res, out);
    const transDir = (wi, x, y, res) => { const w = BX.hemiDir(x, y, res); return w ? (wi[2] >= 0 ? [w[0], w[1], -w[2]] : w) : null; };
    const tangents = (w) => {
        const t = Math.abs(w[2]) < 0.9 ? norm(BX.cross([0, 0, 1], w)) : norm(BX.cross([1, 0, 0], w));
        return [t, BX.cross(w, t)];
    };
    return { PI, etaOf, neg, halfT, halfR, jacobT, termsT, evalT, evalR, evalF, sample, pdf, albedoSplit, fresnelSP, transImage, transDir, tangents, clamp };
})();

// ── environment (equirect + box mips) shared by the path-traced reference and the real-time trick ──
const ENV_W = 256, ENV_H = 128, PT_RES = 96, PT_MAX = 384, EXPO = 0.85;
const L_KEY = BX.norm([-0.45, 0.55, -0.7]), L_FILL = BX.norm([0.55, 0.5, 0.67]);
function envRadiance(d) {
    const y = d[1], th = Math.acos(TR.clamp(y, -1, 1)), ph = Math.atan2(d[0], -d[2]);
    let c;
    if (Math.abs(th - Math.PI / 2) < 0.62) {
        const iu = Math.floor((ph + Math.PI) / (Math.PI / 16)), iv = Math.floor(th / (Math.PI / 20));
        c = ((iu + iv) & 1) ? [0.72, 0.52, 0.28] : [0.05, 0.09, 0.18];
    } else if (y > 0) c = [0.28 + 0.2 * y, 0.34 + 0.25 * y, 0.46 + 0.3 * y];
    else c = [0.05, 0.045, 0.04];
    const k1 = PL.smoothstep(0.982, 0.99, BX.dot(d, L_KEY)), k2 = PL.smoothstep(0.988, 0.994, BX.dot(d, L_FILL));
    return [c[0] + 9 * k1 + 2.6 * k2, c[1] + 8.4 * k1 + 2.9 * k2, c[2] + 7.4 * k1 + 3.4 * k2];
}
function buildEnvMips() {
    const mips = [];
    let w = ENV_W, h = ENV_H, data = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const ph = (x + 0.5) / w * 2 * Math.PI - Math.PI, th = (y + 0.5) / h * Math.PI;
        const c = envRadiance([Math.sin(th) * Math.sin(ph), Math.cos(th), -Math.sin(th) * Math.cos(ph)]), i = (y * w + x) * 3;
        data[i] = c[0]; data[i + 1] = c[1]; data[i + 2] = c[2];
    }
    mips.push({ w, h, data });
    while (h > 1) {
        const pw = w, src = data;
        w >>= 1; h >>= 1; data = new Float32Array(w * h * 3);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
            const a = ((2 * y) * pw + 2 * x) * 3 + c, b = ((2 * y + 1) * pw + 2 * x) * 3 + c;
            data[(y * w + x) * 3 + c] = 0.25 * (src[a] + src[a + 3] + src[b] + src[b + 3]);
        }
        mips.push({ w, h, data });
    }
    return mips;
}
function envFetch(m, u, v, out, k) {
    const x = u * m.w - 0.5, y = TR.clamp(v * m.h - 0.5, 0, m.h - 1);
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, y1 = Math.min(m.h - 1, y0 + 1);
    const xa = ((x0 % m.w) + m.w) % m.w, xb = (xa + 1) % m.w, d = m.data;
    for (let c = 0; c < 3; c++) {
        const t = d[(y0 * m.w + xa) * 3 + c] * (1 - fx) + d[(y0 * m.w + xb) * 3 + c] * fx;
        const b = d[(y1 * m.w + xa) * 3 + c] * (1 - fx) + d[(y1 * m.w + xb) * 3 + c] * fx;
        out[c] += k * (t * (1 - fy) + b * fy);
    }
}
function envSample(mips, d, lod, out, k = 1) {
    const u = Math.atan2(d[0], -d[2]) / (2 * Math.PI) + 0.5, v = Math.acos(TR.clamp(d[1], -1, 1)) / Math.PI;
    lod = TR.clamp(lod, 0, mips.length - 1);
    const l0 = Math.floor(lod), l1 = Math.min(mips.length - 1, l0 + 1), f = lod - l0;
    envFetch(mips[l0], u, v, out, k * (1 - f));
    if (f > 0) envFetch(mips[l1], u, v, out, k * f);
    return out;
}
const CAM_Z = 3.6, CAM_TAN = Math.tan(18 * Math.PI / 180);
const camRay = (px, py, W) => BX.norm([(2 * px / W - 1) * CAM_TAN, (1 - 2 * py / W) * CAM_TAN, -1]);
/** Ground truth: path trace a solid rough-glass unit sphere with the Walter BSDF (radiance mode) + Beer-Lambert */
function ptTrace(rd, S, rnd, out) {
    let ro = [0, 0, CAM_Z], tp = [1, 1, 1], inside = false;
    for (let depth = 0; depth < 10; depth++) {
        const b = BX.dot(ro, rd), c = BX.dot(ro, ro) - 1, disc = b * b - c;
        let t = -1;
        if (disc > 0) { const q = Math.sqrt(disc); t = inside ? -b + q : -b - q; }
        if (t <= 1e-6) {   // escaped (or numerically lost inside → drop)
            if (!inside) { const e = envSample(S.env, rd, 0, [0, 0, 0]); out[0] += tp[0] * e[0]; out[1] += tp[1] * e[1]; out[2] += tp[2] * e[2]; }
            return;
        }
        const p = [ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t];
        if (inside) for (let k = 0; k < 3; k++) tp[k] *= Math.exp(-S.sigma[k] * t);
        const n = BX.norm(p), [T, B] = TR.tangents(n);
        const wo = [-BX.dot(rd, T), -BX.dot(rd, B), -BX.dot(rd, n)];
        const s = TR.sample(wo, S.a, S.ior, rnd(), rnd(), rnd(), 'radiance');
        if (!s || !(s.weight > 0)) return;
        tp = [tp[0] * s.weight, tp[1] * s.weight, tp[2] * s.weight];
        const w = s.wi;
        rd = BX.norm([T[0] * w[0] + B[0] * w[1] + n[0] * w[2], T[1] * w[0] + B[1] * w[1] + n[1] * w[2], T[2] * w[0] + B[2] * w[1] + n[2] * w[2]]);
        inside = w[2] < 0;
        ro = [p[0] + rd[0] * 1e-4, p[1] + rd[1] * 1e-4, p[2] + rd[2] * 1e-4];
    }
}
const lodFor = (theta) => Math.log2(Math.max(1, theta / (2 * Math.PI / ENV_W)));
/** Real-time trick: macro-normal refraction through front + back face, mip-blurred lookup ∝ α, Beer-Lambert, no MC */
function rtShade(rd, S, out) {
    const ro = [0, 0, CAM_Z], b = BX.dot(ro, rd), c = BX.dot(ro, ro) - 1, disc = b * b - c;
    if (disc <= 0) return envSample(S.env, rd, 0, out);
    const t0 = -b - Math.sqrt(disc), p = [ro[0] + rd[0] * t0, ro[1] + rd[1] * t0, ro[2] + rd[2] * t0], n = BX.norm(p);
    const wo = TR.neg(rd), cv = BX.dot(n, wo), F = BX.fresnelDielectric(cv, S.ior);
    const rdir = [rd[0] + 2 * cv * n[0], rd[1] + 2 * cv * n[1], rd[2] + 2 * cv * n[2]];
    envSample(S.env, rdir, lodFor(2 * S.a * S.blurK), out, F);
    let d = BX.refract(wo, n, S.ior), q = p, len = 0, thr = 1 - F;
    for (let k = 0; k < 3 && d; k++) {
        const t = -2 * BX.dot(q, d), q2 = [q[0] + d[0] * t, q[1] + d[1] * t, q[2] + d[2] * t], n2 = BX.norm(q2);
        len += t;
        const back = TR.neg(d), nIn = TR.neg(n2), e = BX.refract(back, nIn, 1 / S.ior);
        if (e) {
            const F2 = BX.fresnelDielectric(BX.dot(back, nIn), 1 / S.ior), k3 = thr * (1 - F2);
            const blur = S.a * S.blurK * (S.ior - 1) * (1 + 1 / S.ior);
            const tmp = envSample(S.env, e, lodFor(blur), [0, 0, 0]);
            for (let ch = 0; ch < 3; ch++) out[ch] += k3 * tmp[ch] * Math.exp(-S.sigma[ch] * len);
            return out;
        }
        const dn = BX.dot(d, n2);
        d = [d[0] - 2 * dn * n2[0], d[1] - 2 * dn * n2[1], d[2] - 2 * dn * n2[2]];   // TIR: follow the internal reflection
        q = q2;
    }
    return out;
}

// ── small canvas helpers ──
function arrow(ctx, x0, y0, x1, y1, color, width, head = 6) {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0), hs = head + width;
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - hs * Math.cos(a - 0.4), y1 - hs * Math.sin(a - 0.4));
    ctx.lineTo(x1 - hs * Math.cos(a + 0.4), y1 - hs * Math.sin(a + 0.4));
    ctx.closePath(); ctx.fill();
}
function mediumBg(ctx, w, h, cy, S, insideLabel) {
    ctx.fillStyle = 'rgba(70,130,190,0.13)'; ctx.fillRect(0, cy, w, h - cy);
    ctx.strokeStyle = '#6a7a88'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
    ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'right';
    ctx.fillStyle = '#7a8a96'; ctx.fillText('空气 η = 1.00', w - 6, 4);
    ctx.fillStyle = '#7fb0dc'; ctx.fillText(`介质 η = ${S.ior.toFixed(2)}` + (insideLabel || ''), w - 6, h - 16);
}
const deg = (r) => r * 180 / Math.PI;
/** ray diagram for a smooth interface: incident (gold), reflected (orange, width ∝ F), refracted (blue, width ∝ 1−F) */
function drawSnell(canvas, S, inside) {
    const { ctx, w, h } = PL.draw.ctx(canvas);
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 14;
    mediumBg(ctx, w, h, cy, S);
    ctx.setLineDash([3, 3]); ctx.strokeStyle = '#46525c'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke(); ctx.setLineDash([]);
    const wi = inside ? S.wiIn : S.wiOut, n = inside ? [0, 0, -1] : [0, 0, 1], eta = inside ? 1 / S.ior : S.ior;
    const F = BX.fresnelDielectric(wi[2], S.ior), wr = [-wi[0], 0, wi[2]], wt = BX.refract(wi, n, eta);
    const P = (v) => [cx + v[0] * R, cy - v[2] * R];
    const [ix, iy] = P(wi), [rx, ry] = P(wr);
    arrow(ctx, ix, iy, cx, cy, '#e8c25a', 2);
    arrow(ctx, cx, cy, rx, ry, '#e0a060', 0.4 + 4 * F);
    ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
    ctx.fillStyle = '#e8c25a'; ctx.fillText(`θi = ${deg(S.thetaI).toFixed(0)}°`, 6, inside ? h - 30 : 4);
    ctx.fillStyle = '#e0a060'; ctx.fillText(`R = F = ${F.toFixed(3)}`, 6, inside ? h - 16 : 18);
    if (wt) {
        const [tx, ty] = P(wt);
        arrow(ctx, cx, cy, tx, ty, '#6ab0e0', 0.4 + 4 * (1 - F));
        ctx.fillStyle = '#6ab0e0';
        ctx.fillText(`θt = ${deg(Math.acos(Math.abs(wt[2]))).toFixed(1)}°  T = ${(1 - F).toFixed(3)}`, 6, inside ? 4 : h - 16);
    } else {
        ctx.fillStyle = '#e06c75'; ctx.fillText(`TIR：θi > θc = ${deg(S.thetaC).toFixed(1)}°`, 6, inside ? 4 : h - 16);
    }
}
/** overlay the smooth refraction direction (dashed) on a BX.lobe full-circle plot */
function lobeOverlay(canvas, S, wi) {
    const { ctx, w, h } = PL.draw.ctx(canvas, { clear: false });
    const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 14;
    ctx.fillStyle = 'rgba(70,130,190,0.08)'; ctx.fillRect(0, cy, w, h - cy);
    const n = wi[2] >= 0 ? [0, 0, 1] : [0, 0, -1], wt = BX.refract(wi, n, wi[2] >= 0 ? S.ior : 1 / S.ior);
    ctx.font = '10px Consolas, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    if (wt) {
        ctx.strokeStyle = 'rgba(90,208,122,0.7)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + wt[0] * R, cy - wt[2] * R); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = '#5ad07a'; ctx.fillText('┄ 光滑折射方向', w - 6, h - 4);
    } else { ctx.fillStyle = '#e06c75'; ctx.fillText('光滑界面此处 TIR', w - 6, h - 4); }
}
/** RGB image via an offscreen canvas so a text overlay does not reset the pixel canvas */
function drawImageRGB(canvas, W, H, fn, label) {
    const off = canvas._plOff || (canvas._plOff = document.createElement('canvas'));
    PL.draw.pixels(off, W, H, fn);
    const { ctx, w, h } = PL.draw.ctx(canvas);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, w, h);
    if (label) {
        ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'bottom'; ctx.textAlign = 'left';
        ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(0, h - 16, ctx.measureText(label).width + 10, 16);
        ctx.fillStyle = '#cdd'; ctx.fillText(label, 5, h - 3);
    }
}
/** ∫_{S²} fn(ω) dω with the pole on `axis` and θ′ = π t² (dense near the lobe peak) — independent check of Eval */
function quadAligned(fn, axis, nT, nP) {
    const [t1, t2] = TR.tangents(axis);
    let s = 0;
    for (let i = 0; i < nT; i++) {
        const t = (i + 0.5) / nT, th = Math.PI * t * t, st = Math.sin(th), ct = Math.cos(th), wgt = st * 2 * Math.PI * t / nT * (2 * Math.PI / nP);
        for (let j = 0; j < nP; j++) {
            const p = (j + 0.5) / nP * 2 * Math.PI, a = st * Math.cos(p), b = st * Math.sin(p);
            s += fn([a * t1[0] + b * t2[0] + ct * axis[0], a * t1[1] + b * t2[1] + ct * axis[1], a * t1[2] + b * t2[2] + ct * axis[2]]) * wgt;
        }
    }
    return s;
}
function textPanel(canvas, lines) {
    const { ctx } = PL.draw.ctx(canvas);
    ctx.font = '11px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
    lines.forEach(([t, c], k) => { ctx.fillStyle = c || '#9aa'; ctx.fillText(t, 8, 8 + k * 16); });
}
// ── 3D：反射 / 透射 lobe + 界面平面。mesh3d 是 y 朝上：W() 把局部 (x, y, z = n) 转成世界 (x, z, −y) ──
function trDrawLobe3d(c, S) {
    const { nu, nv, scale } = TR_L3D, NU = nu + 1, L = S.l3d, DEC = 3, W = (v) => [v[0], v[2], -v[1]];
    // 半径 = 对数刻度：r = scale · max(0, (log10(v / 两者峰值) + 3) / 3)，与右侧 log 图一致（线性刻度下透射 lobe 只是一根针）
    const grid = (P) => (u, v) => {
        const q = 3 * (Math.round(v * nv) * NU + Math.round(u * nu)), x = P[q], y = P[q + 1], z = P[q + 2], m = Math.hypot(x, y, z);
        if (!(m > 0)) return [0, 0, 0];
        const r = scale * Math.max(0, (Math.log10(m / L.max) + DEC) / DEC);
        return W([x / m * r, y / m * r, z / m * r]);
    };
    const at = (d, r) => W([d[0] * r, d[1] * r, d[2] * r]), wi = S.wi, up = wi[2] >= 0;
    const lines = [
        { pts: [[0, 0, 0], at([0, 0, up ? 1 : -1], 1.45)], color: '#9aa4ae', width: 1.2, arrow: true, label: 'n' },
        { pts: [[0, 0, 0], at(wi, 1.7)], color: '#e8c25a', width: 2.4, arrow: true, label: 'ωi' },
        { pts: [[0, 0, 0], at(L.mir, 1.55)], color: '#e0a060', width: 1.2, dash: [4, 3], label: '镜面反射' },
    ];
    if (L.wt) lines.push({ pts: [[0, 0, 0], at(L.wt, 1.55)], color: '#5ad07a', width: 1.2, dash: [4, 3], label: '光滑折射' });
    const labels = [
        { p: [-1.25, 0.16, 1.1], text: '空气 η = 1.00', color: '#9aa4ae' },
        { p: [-1.25, -0.2, 1.1], text: `介质 η = ${S.ior.toFixed(2)}`, color: '#7fb0dc' },
    ];
    if (!L.wt) labels.push({ p: at(BX.norm([-wi[0], -wi[1], up ? -0.35 : 0.35]), 1.2), text: `TIR（θi > θc = ${deg(S.thetaC).toFixed(1)}°）`, color: '#e06c75' });
    PL.draw.mesh3d(c, {
        meshes: [
            PL.mesh.plane({ c: [0, 0, 0], size: 2.6, surface: false, wire: true, wireColor: 'rgba(120,170,220,0.35)' }),
            PL.mesh.param(grid(S.l3dR), [nu, nv], { closedU: true, rgb: [0.88, 0.63, 0.38], gloss: 24 }),
            PL.mesh.param(grid(S.l3dT), [nu, nv], { closedU: true, rgb: [0.42, 0.69, 0.88], gloss: 24 }),
        ],
        lines, labels,
        grid: { y: 0, size: 2.6 },
        orbit: { yaw: 0.5, pitch: 0.2, dist: 5.4, target: [0, 0, 0] },
    });
}
const thetaDeg = (k, n) => Math.min(89, 90 * k / (n - 1));
const EN_N = 31, EN_RS = [0.1, 0.25, 0.4, 0.6, 0.8, 1.0], EN_COLORS = ['#56b6c2', '#6ab0e0', '#5ad07a', '#e8c25a', '#e0a060', '#e06c75'];
const NS = 4096, NBIN = 60, NSUB = 6, NPHI = 64, HRES = 96;

// ════════════════════════════════════════════════════════════════════════════
// HLSL
// ════════════════════════════════════════════════════════════════════════════
const HLSL_TR_COMMON = `
    // ── 03 透射模块 · 上半球 (z > 0) = 外部空气 η = 1，下半球 = 介质内部 η = Ior ──
    float SideEta(float3 w, float ior) { return w.z >= 0.0 ? 1.0 : ior; }

    // 折射：wi 背离表面、n 与 wi 同侧、eta = η_t / η_i；TIR 返回 0
    // HLSL 内置 refract(I, N, eta) 的 I 指向表面、eta = η_i / η_t ⇒ RefractDir(wi, n, e) == refract(-wi, n, 1 / e)
    float3 RefractDir(float3 wi, float3 n, float eta) {
        float cosI  = dot(n, wi);
        float sin2T = (1.0 - cosI * cosI) / (eta * eta);
        if (sin2T >= 1.0) return 0;
        return -wi / eta + (cosI / eta - sqrt(1.0 - sin2T)) * n;
    }

    // 精确非偏振 Fresnel；cosI < 0 = 从另一侧入射（η 取倒数）；TIR → 1
    float F_Dielectric(float cosI, float eta) {
        if (cosI < 0.0) { eta = 1.0 / eta; cosI = -cosI; }
        float sin2T = (1.0 - cosI * cosI) / (eta * eta);
        if (sin2T >= 1.0) return 1.0;
        float cosT = sqrt(1.0 - sin2T);
        float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
        float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
        return 0.5 * (rs * rs + rp * rp);
    }

    // Lambda_GGX 用 NoX²，所以下半球方向也能直接用
    float G1_Smith(float NoX, float a)            { return 1.0 / (1.0 + Lambda_GGX(NoX, a)); }
    float G2_Smith(float NoI, float NoO, float a) { return 1.0 / (1.0 + Lambda_GGX(NoI, a) + Lambda_GGX(NoO, a)); }

    // Heitz 2018 VNDF 采样；要求 wo.z ≥ 0，返回 h.z > 0
    float3 SampleVNDF_GGX(float3 wo, float a, float2 u) {
        float3 vh = normalize(float3(a * wo.x, a * wo.y, wo.z));
        float  lensq = vh.x * vh.x + vh.y * vh.y;
        float3 t1 = lensq > 0.0 ? float3(-vh.y, vh.x, 0.0) * rsqrt(lensq) : float3(1, 0, 0);
        float3 t2 = cross(vh, t1);
        float  r = sqrt(u.x), phi = 2.0 * PI * u.y;
        float  p1 = r * cos(phi), p2 = r * sin(phi);
        float  s = 0.5 * (1.0 + vh.z);
        p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;
        float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
        return normalize(float3(a * nh.x, a * nh.y, max(1e-6, nh.z)));
    }`;

const HLSL_DELTA = `
    // 光滑电介质 = delta BSDF：Eval 恒为 0，只能采样。wo → 相机，返回 wi，weight = f·|cosθi| / pdf
    float3 Sample_SmoothDielectric(float3 wo, float ior, float u, out float weight) {
        float F = F_Dielectric(wo.z, ior);                  // wo.z < 0：从介质内部看；TIR 时 F = 1
        if (u < F) {                                        // 以概率 F 反射：f·cos/pdf = F / F = 1
            weight = 1.0;
            return float3(-wo.x, -wo.y, wo.z);
        }
        float etaO = SideEta(wo, ior), etaI = wo.z >= 0.0 ? ior : 1.0;
        float3 n = float3(0, 0, wo.z >= 0.0 ? 1.0 : -1.0);  // 与 wo 同侧的法线
        weight = (etaO * etaO) / (etaI * etaI);              // 以概率 1−F 折射：(1−F)(η_o/η_i)² / (1−F)
        return RefractDir(wo, n, etaI / etaO);
    }`;

const HLSL_HALF = `
    // Walter 2007 Eq.16：广义半向量 h_t = −(η_i·wi + η_o·wo)，再统一翻到 +n 侧
    float3 HalfVectorT(float3 wi, float3 wo, float ior, out float etaI, out float etaO) {
        etaI = SideEta(wi, ior);
        etaO = SideEta(wo, ior);
        float3 h = normalize(-(etaI * wi + etaO * wo));
        return h.z < 0.0 ? -h : h;          // D(h) 只定义在 +n 半球；Ior < 1 时这一步才真正生效
    }

    // 有效透射配置：两个方向各自位于微面元的"正确"一侧（χ⁺），且分处宏观表面两侧
    bool ValidT(float3 wi, float3 wo, float3 h) {
        return wi.z * wo.z < 0.0 && dot(wi, h) * wi.z > 0.0 && dot(wo, h) * wo.z > 0.0;
    }`;

const HLSL_JACOB = `
    // Walter 2007 Eq.17：固定 wi 时 h_t 相对 wo 的立体角雅可比
    float JacobianT(float3 wi, float3 wo, float3 h, float etaI, float etaO) {
        float denom = etaI * dot(wi, h) + etaO * dot(wo, h);
        return etaO * etaO * abs(dot(wo, h)) / (denom * denom);
    }
    // 对照：反射 h_r = normalize(wi + wo)
    float JacobianR(float3 wo, float3 h) { return 1.0 / (4.0 * abs(dot(wo, h))); }`;

const HLSL_BSDF = `
    // 粗糙电介质 BSDF（radiance 模式）。wi → 光源、wo → 相机，局部坐标 n = +Z
    float Eval_RoughDielectric(float3 wi, float3 wo, float a, float ior) {
        float NoI = wi.z, NoO = wo.z;
        if (NoI * NoO > 0.0) {                                     // ── 同侧：反射 f_r ──
            float3 h = normalize(wi + wo);  h *= sign(h.z);
            float IoH = dot(wi, h);
            if (IoH * NoI <= 0.0 || dot(wo, h) * NoO <= 0.0) return 0.0;
            return F_Dielectric(IoH, ior) * D_GGX(h.z, a) * G2_Smith(NoI, NoO, a) / (4.0 * abs(NoI * NoO));
        }
        float  etaI, etaO;                                         // ── 异侧：透射 f_t（Walter Eq.21）──
        float3 h = HalfVectorT(wi, wo, ior, etaI, etaO);
        if (!ValidT(wi, wo, h)) return 0.0;
        float IoH = dot(wi, h), OoH = dot(wo, h);
        float denom = etaI * IoH + etaO * OoH;
        float F = F_Dielectric(IoH, ior);                          // = F(OoH)：两侧透射率相同
        float D = D_GGX(h.z, a);
        float G = G2_Smith(NoI, NoO, a);
        return abs(IoH * OoH / (NoI * NoO)) * etaO * etaO * (1.0 - F) * D * G / (denom * denom);
    }`;

const HLSL_ETA2 = `
    // radiance（相机路径）↔ importance（光源路径 / 光子）只差常数 (η_i/η_o)²：f*(wi,wo) = f(wo,wi)
    float BTDF_ForMode(float ftRadiance, float etaI, float etaO, bool importance) {
        return importance ? ftRadiance * (etaI * etaI) / (etaO * etaO) : ftRadiance;
    }
    // 典型 bug：两种模式都用了 radiance 版 → 双向方法（BDPT / 光子）在玻璃里能量偏 η² 倍`;

const HLSL_SAMPLE = `
    // 给定 wo（相机侧），采样 wi。u.x 选 R/T，u.yz 采 VNDF。返回 weight = f·|cosθi| / pdf
    float3 Sample_RoughDielectric(float3 wo, float a, float ior, float3 u, out float weight, out float pdf) {
        float  s    = wo.z >= 0.0 ? 1.0 : -1.0;
        float3 h    = SampleVNDF_GGX(wo * s, a, u.yz);           // 介质内部看：翻到上半球采样
        float  OoH  = dot(wo, h);
        float  F    = F_Dielectric(OoH, ior);                    // TIR → F = 1 → 必然反射
        float  G1o  = G1_Smith(wo.z, a);
        float  pdfH = G1o * abs(OoH) * D_GGX(h.z, a) / abs(wo.z);   // D_wo(h)
        float3 wi;
        weight = 0.0; pdf = 0.0;
        if (u.x < F) {                                            // ── 反射 ──
            wi = 2.0 * OoH * h - wo;
            if (wi.z * wo.z <= 0.0) return wi;                     // 反射到宏观表面另一侧：丢弃
            pdf    = pdfH * F / (4.0 * abs(OoH));
            weight = G2_Smith(wi.z, wo.z, a) / G1o;
        } else {                                                  // ── 折射 ──
            float etaO = SideEta(wo, ior), etaI = wo.z >= 0.0 ? ior : 1.0;
            wi = RefractDir(wo, OoH >= 0.0 ? h : -h, etaI / etaO);
            if (wi.z * wo.z >= 0.0) return wi;                     // 折回同侧：丢弃
            float IoH = dot(wi, h), denom = etaI * IoH + etaO * OoH;
            pdf    = pdfH * (1.0 - F) * etaI * etaI * abs(IoH) / (denom * denom);   // × dω_h/dω_i
            weight = G2_Smith(wi.z, wo.z, a) / G1o * (etaO * etaO) / (etaI * etaI);
        }
        return wi;
    }`;

const HLSL_ENERGY = `
    // 白炉（flux 版）：固定入射方向 w，VNDF 采 h，R、T 两支都累加（不掷 Fresnel 硬币）→ 零方差地分开 E_R / E_T
    float2 Furnace_RT(float3 w, float a, float ior, uint N) {
        float  s = w.z >= 0.0 ? 1.0 : -1.0;
        float  etaO = SideEta(w, ior), etaI = w.z >= 0.0 ? ior : 1.0;
        float  G1 = G1_Smith(w.z, a);
        float2 E = 0;
        for (uint k = 0; k < N; k++) {
            float3 h   = SampleVNDF_GGX(w * s, a, Hammersley(k, N));
            float  OoH = dot(w, h), F = F_Dielectric(OoH, ior);
            float3 wr  = 2.0 * OoH * h - w;
            if (wr.z * w.z > 0.0) E.x += F * G2_Smith(wr.z, w.z, a) / G1;
            float3 wt  = RefractDir(w, OoH >= 0.0 ? h : -h, etaI / etaO);
            if (F < 1.0 && wt.z * w.z < 0.0) E.y += (1.0 - F) * G2_Smith(wt.z, w.z, a) / G1;
        }
        return E / N;          // E_R + E_T ≤ 1；缺口 = 被 G2 截掉的多次散射能量
    }`;

const HLSL_PS = `
    // 实时粗糙折射（前向透明 pass，排在不透明之后）。SceneColorMips = 不透明结果 + mip 链
    Texture2D<float4> SceneColorMips;   Texture2D<float> BackFaceDepth;   SamplerState LinearClamp;

    float4 PS_RoughRefraction(VSOut i) : SV_Target {
        float3 N = normalize(i.normalWS), V = normalize(CameraPos - i.posWS);
        float  a = Roughness * Roughness;
        float  F = F_Dielectric(dot(N, V), Ior);                       // 宏观法线，单次 Fresnel
        float3 T = RefractDir(V, N, Ior);                              // 只折一次（或前后两面）
        float  thick = max(0.0, LinearDepth(BackFaceDepth[i.pos.xy]) - i.viewZ);
        float2 uv    = WorldToScreenUV(i.posWS + T * thick);           // 折射偏移
        float  blur  = BlurK * a * (Ior - 1.0) * (1.0 + 1.0 / Ior);    // 经验：透射角宽 ∝ α·(η−1)
        float  lod   = log2(max(1.0, blur * ScreenHeight / (2.0 * i.viewZ * TanHalfFov)));
        float3 bg    = SceneColorMips.SampleLevel(LinearClamp, uv, lod).rgb;
        float3 absorb = exp(-SigmaA * thick);                          // Beer-Lambert
        float3 refl  = PrefilteredEnv(reflect(-V, N), Roughness);      // split-sum 反射
        return float4(F * refl + (1.0 - F) * absorb * bg, 1.0);       // 不做 η² —— 两次界面相消
    }`;

const HLSL_PT = `
    // 参考：路径追踪实心玻璃球。每帧每像素 1 条路径，累加到 PathTraceAccum
    [shader("raygeneration")] void RayGen_PathTrace() {
        RayDesc ray = CameraRay(DispatchRaysIndex().xy, Jitter(FrameIndex));
        float3 tp = 1, L = 0;
        for (uint depth = 0; depth < 10; depth++) {
            Hit hit;
            if (!TraceScene(ray, hit)) { L += tp * EnvMap.SampleLevel(LinearWrap, DirToEquirect(ray.Direction), 0).rgb; break; }
            if (hit.inside) tp *= exp(-SigmaA * hit.t);                // 介质内走过的距离
            float3x3 TBN = Basis(hit.normalOut);                       // n = 外法线：上半球 = 空气
            float3 wo = mul(TBN, -ray.Direction), u = Rand3();
            float  w, pdf;
            float3 wi = Sample_RoughDielectric(wo, Roughness * Roughness, Ior, u, w, pdf);
            if (w <= 0.0) break;
            tp *= w;                                                   // 进出各乘 1/η²、η²，相消
            ray.Origin = hit.pos; ray.Direction = mul(wi, TBN);
        }
        PathTraceAccum[DispatchRaysIndex().xy] += float4(L, 1);
    }`;

// ── 附录：正文代码调用、步骤里没有展开的辅助函数 ──
const HLSL_LIB_SEQ = `
    // Hammersley 点集：x = (i + 0.5)/N，y = 以 2 为底的 radical inverse（reversebits）
    float2 Hammersley(uint i, uint N) {
        return float2((i + 0.5) / N, reversebits(i) * 2.3283064365386963e-10);   // 2⁻³²
    }`;

const HLSL_LIB_RT = `
    cbuffer ViewCB : register(b1) {
        float4x4 WorldToClip;
        float4x4 ClipToWorld;
        float3   CameraPos;
        float    NearZ, FarZ;                 // 标准 D3D 投影：设备深度 0 = 近平面，1 = 远平面
        float    ScreenHeight, TanHalfFov;
    };
    TextureCube<float4> PrefilteredEnvCube;   // 02 的 split-sum 预滤波链：mip = r · (ENV_MIP_COUNT − 1)
    #define ENV_MIP_COUNT 6

    // 透明物体的 VS 输出（前向 pass）
    struct VSOut {
        float4 pos      : SV_Position;
        float3 posWS    : TEXCOORD0;
        float3 normalWS : TEXCOORD1;
        float  viewZ    : TEXCOORD2;          // 视空间线性深度（到相机平面的距离）
    };

    // 设备深度 → 视空间线性深度（标准透视投影的逆：z = n·f / (f − d·(f − n))）
    float LinearDepth(float deviceZ) {
        return NearZ * FarZ / (FarZ - deviceZ * (FarZ - NearZ));
    }

    // 世界坐标 → 屏幕 uv（[0, 1]²，v 向下）：透视除法后把 NDC 映射到纹理坐标
    float2 WorldToScreenUV(float3 posWS) {
        float4 clip = mul(WorldToClip, float4(posWS, 1.0));
        float2 ndc  = clip.xy / clip.w;
        return float2(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y);
    }

    // split-sum 的预滤波环境（02 的 CS_PrefilterEnv 产物）：按感知粗糙度选 mip
    float3 PrefilteredEnv(float3 R, float roughness) {
        return PrefilteredEnvCube.SampleLevel(LinearClamp, R, roughness * (ENV_MIP_COUNT - 1)).rgb;
    }`;

const HLSL_LIB_PT = `
    cbuffer PathTraceCB : register(b2) { uint FrameIndex; };
    Texture2D<float4>   EnvMap;               // 等距柱状环境图（页面右图）
    SamplerState        LinearWrap;
    RWTexture2D<float4> PathTraceAccum;       // rgb = 累加辐亮度，a = 样本数

    // 一次求交的结果；inside = 这段光线在介质内部（从球内射向球面）
    struct Hit {
        float3 pos;
        float3 normalOut;                     // 外法线：局部上半球 = 空气
        float  t;
        bool   inside;
    };

    // 相机光线：像素中心 + 抖动，经 ClipToWorld 反投影出方向（z = 1 在两种深度约定下都是有限远的点）
    RayDesc CameraRay(uint2 pix, float2 jitter) {
        float2 uv = (pix + 0.5 + jitter) / DispatchRaysDimensions().xy;
        float4 p  = mul(ClipToWorld, float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0));
        RayDesc ray;
        ray.Origin    = CameraPos;
        ray.Direction = normalize(p.xyz / p.w - CameraPos);
        ray.TMin      = 1e-4;                 // 后续弹射从球面出发，TMin 跳过自身
        ray.TMax      = 1e30;
        return ray;
    }

    // 以 b 为底的 radical inverse（Halton 序列的一维）
    float Halton(uint i, uint b) {
        float f = 1.0, r = 0.0;
        for (; i > 0; i /= b) { f /= b; r += f * (i % b); }
        return r;
    }
    // 第 frame 帧的像素内抖动 ∈ [−0.5, 0.5)²：Halton(2, 3)，渐进累加时逐帧覆盖整个像素
    float2 Jitter(uint frame) {
        return float2(Halton(frame + 1, 2), Halton(frame + 1, 3)) - 0.5;
    }

    // 参考场景 = 原点处的单位玻璃球（与页面里的 JS 路径追踪相同），解析求交；一般场景换成 TraceRay + closest hit
    bool TraceScene(RayDesc ray, out Hit hit) {
        hit = (Hit)0;
        float b = dot(ray.Origin, ray.Direction), c = dot(ray.Origin, ray.Origin) - 1.0, disc = b * b - c;
        if (disc <= 0.0) return false;
        float q = sqrt(disc), t0 = -b - q, t1 = -b + q;
        if (t0 > ray.TMin)      { hit.t = t0; hit.inside = false; }   // 从外面打到球
        else if (t1 > ray.TMin) { hit.t = t1; hit.inside = true;  }   // 起点在球内（或在球面上朝里）
        else return false;
        if (hit.t > ray.TMax) return false;
        hit.pos       = ray.Origin + hit.t * ray.Direction;
        hit.normalOut = normalize(hit.pos);
        return true;
    }

    // 方向 → 等距柱状 uv（与页面环境图相同：u = atan2(x, −z)/2π + 0.5，v = acos(y)/π，y 朝上）
    float2 DirToEquirect(float3 d) {
        return float2(atan2(d.x, -d.z) * (0.5 * INV_PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) * INV_PI);
    }

    // 以 n 为 z 轴的正交基，按行 (t, b, n)：mul(TBN, v) = 世界 → 局部，mul(v, TBN) = 局部 → 世界（Duff et al. 2017）
    float3x3 Basis(float3 n) {
        float  s = n.z >= 0.0 ? 1.0 : -1.0;
        float  a = -1.0 / (s + n.z), c = n.x * n.y * a;
        float3 t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
        float3 b = float3(c, s + n.y * n.y * a, -n.y);
        return float3x3(t, b, n);
    }

    // PCG 哈希（Jarzynski & Olano 2020, JCGT 9(3)）
    uint PcgHash(uint v) {
        uint state = v * 747796405u + 2891336453u;
        uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
    // 线程私有的随机数状态：第一次调用时用 (像素, FrameIndex) 播种，之后每次推进
    static uint RngState = 0;
    float Rand() {
        if (RngState == 0) {
            uint2 p = DispatchRaysIndex().xy;
            RngState = PcgHash(p.x + PcgHash(p.y + PcgHash(FrameIndex))) | 1u;
        }
        RngState = PcgHash(RngState);
        return (RngState >> 8) * (1.0 / 16777216.0);                  // 24 位 → [0, 1)
    }
    // 三个独立的 [0, 1) 随机数：x 选 R / T，yz 驱动 VNDF（即 Sample_RoughDielectric 的 u）
    float3 Rand3() {
        return float3(Rand(), Rand(), Rand());
    }`;

// ── 3D lobe：以 lobe 轴为极点的 (φ, ψ) 网格，ψ = π·t² 向极点加密（窄 lobe 也有足够顶点） ──
const TR_L3D = { nu: 72, nv: 48, scale: 1.3 };
/** fn(ω) ≥ 0 over the sphere; writes local (z-up) vertices r·ω into P (unnormalized), returns the max */
function trLobeGrid(fn, axis, P) {
    const { nu, nv } = TR_L3D, NU = nu + 1, [e1, e2] = TR.tangents(axis);
    let vmax = 0;
    for (let j = 0; j <= nv; j++) {
        const ps = Math.PI * (j / nv) ** 2, cp = Math.cos(ps), sp = Math.sin(ps);
        for (let i = 0; i <= nu; i++) {
            const ph = 2 * Math.PI * i / nu, c = Math.cos(ph) * sp, s = Math.sin(ph) * sp, k = 3 * (j * NU + i);
            const d = [cp * axis[0] + c * e1[0] + s * e2[0], cp * axis[1] + c * e1[1] + s * e2[1], cp * axis[2] + c * e1[2] + s * e2[2]];
            const v = Math.max(0, fn(d) || 0);
            P[k] = d[0] * v; P[k + 1] = d[1] * v; P[k + 2] = d[2] * v;
            if (v > vmax) vmax = v;
        }
    }
    return vmax;
}

// ════════════════════════════════════════════════════════════════════════════
PipelineLab.mount({
    paramSteps: { rough: ['half', 'btdf', 'lobe', 'eta2', 'sample', 'energy', 'prod'], ior: ['snell', 'fresnel', 'half', 'jacob', 'btdf', 'lobe', 'eta2', 'sample', 'energy', 'prod'], thetaI: ['snell', 'fresnel', 'half', 'jacob', 'btdf', 'lobe', 'eta2', 'sample', 'energy'], side: ['half', 'jacob', 'btdf', 'lobe', 'sample'], tint: ['prod'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: 'BxDF 03 · 粗糙透射 BTDF（Walter 2007）',
    lang: 'zh-CN', codeLang: 'hlsl', seed: 3,
    tagColors: { SNELL: 'teal', FRESNEL: 'blue', HALF: 'magenta', JACOB: 'indigo', BTDF: 'green', LOBE: 'olive', ETA2: 'rust', SAMPLE: 'magenta', ENERGY: 'teal', PROD: 'gray' },

    params: [
        {
            id: 'rough', label: 'roughness r', min: 0.1, max: 1, step: 0.01, value: 0.4, digits: 2, title: '感知粗糙度 r，GGX α = r²',
            desc: '感知粗糙度 r（GGX α = r²），反射与透射共用同一个微表面分布。越大两个 lobe 越宽、透射 lobe 越偏向掠射一侧：拖到 0.1 看 Step 06 的 3D lobe 收成指向镜面方向与折射方向的两根尖刺，拖到 1 看 Step 09 白炉里 E<sub>R</sub> + E<sub>T</sub> 的缺口变大。',
        },
        {
            id: 'ior', label: 'IOR η', min: 1.05, max: 2.5, step: 0.01, value: 1.5, digits: 2, title: '介质折射率（外部为空气 1.0）：相对 IOR η = η_int / η_ext',
            desc: '介质折射率 η（外部按空气 1.0）。它决定折射角、F0 = ((η − 1)/(η + 1))² 和临界角 θ<sub>c</sub> = asin(1/η)：η 越大折射越强、透射 lobe 越贴近法线、TIR 区越宽（Step 01 / 02 / 06），也进入 Step 07 的 η² 辐亮度缩放。',
        },
        {
            id: 'thetaI', label: 'θi', min: 0, max: 89, step: 1, value: 40, unit: '°', map: (d) => d * Math.PI / 180, title: '入射角（lobe / 热图 / 采样步骤共用）',
            desc: '入射角 θ<sub>i</sub>（ω<sub>i</sub> 与它所在一侧法线的夹角），所有 lobe、热图与采样图共用。切到"介质 → 外部"后拖过临界角 θ<sub>c</sub>（玻璃约 41.8°），看透射 lobe 塌成贴着界面的细尾巴（Step 01 / 06）；外部入射拖到 70° 以上，看透射 lobe 变成偏向掠射的"彗星"（Step 05）。',
        },
        {
            id: 'side', label: '入射侧', options: [{ value: 'out', label: '外部 → 介质' }, { value: 'in', label: '介质 → 外部' }], value: 'out', title: '光线从哪一侧打到界面（影响 HALF / JACOB / BTDF / SAMPLE 步骤）',
            desc: '光从哪一侧打到界面：外部 → 介质（ω<sub>i</sub> 在上半球的空气里）或介质 → 外部（ω<sub>i</sub> 在下半球）。切换后 Step 03–05 的热图、Step 06 的 3D lobe 和 Step 08 的采样都换到另一侧；介质 → 外部且 θ<sub>i</sub> &gt; θ<sub>c</sub> 时出现全内反射，反射 lobe 成为主角。',
        },
        {
            id: 'tint', label: '吸收色', value: '#bfe3d0', title: 'Beer-Lambert：光在介质里走 1 个球半径后剩下的颜色（只影响最后一步的球）',
            desc: 'Beer-Lambert 吸收色：光在介质里走 1 个球半径后剩下的颜色，σ<sub>a</sub> = −ln(tint)。只影响 Step 10 的两个玻璃球（路径追踪参考与实时近似）：颜色越深吸收越强，穿过的路径越长（球心）颜色越饱和。',
        },
    ],

    stages: [
        {
            id: 'env', lifetime: 'resident',
            run(S) { S.env = buildEnvMips(); },
        },
        {
            id: 'setup', lifetime: 'param',
            run(S) {
                S.a = BX.alpha(S.rough);
                const st = Math.sin(S.thetaI), ct = Math.cos(S.thetaI);
                S.wiOut = [-st, 0, ct];
                S.wiIn = [-st, 0, -ct];
                S.wi = S.side === 'in' ? S.wiIn : S.wiOut;
                S.thetaC = Math.asin(1 / S.ior);
                S.sigma = S.tint.map((c) => -Math.log(Math.max(1e-3, c)));
                S.blurK = 1.6;
            },
        },
        {
            id: 'maps', lifetime: 'param',
            run(S) {
                const wi = S.wi, a = S.a, ior = S.ior;
                S.mapTh = TR.transImage((wo) => { const t = TR.termsT(wi, wo, a, ior); return t ? deg(Math.acos(t.h[2])) : NaN; }, wi, HRES, S.mapTh);
                const keys = ['D', 'G', 'T', 'J', 'f'];
                S.maps = S.maps || {};
                keys.forEach((k) => { S.maps[k] = PL.buf(S, 'map_' + k, HRES * HRES); });
                for (let y = 0; y < HRES; y++) for (let x = 0; x < HRES; x++) {
                    const i = y * HRES + x, wo = TR.transDir(wi, x, y, HRES), t = wo && TR.termsT(wi, wo, a, ior);
                    if (!t) { keys.forEach((k) => { S.maps[k][i] = NaN; }); continue; }
                    S.maps.D[i] = t.D; S.maps.G[i] = t.G; S.maps.T[i] = 1 - t.F; S.maps.J[i] = t.proj * t.J * Math.abs(wo[2]);
                    S.maps.f[i] = t.f * Math.abs(wo[2]);
                }
                for (const k of ['D', 'f']) {   // 10·log10(v / max), NaN (invalid) stays black
                    const src = S.maps[k], dst = S.maps[k + 'db'] = PL.buf(S, 'map_' + k + 'db', HRES * HRES);
                    let mx = 0;
                    for (let i = 0; i < src.length; i++) if (src[i] > mx) mx = src[i];
                    for (let i = 0; i < src.length; i++) dst[i] = isNaN(src[i]) ? NaN : Math.max(-40, 10 * Math.log10(src[i] / (mx || 1) + 1e-30));
                }
            },
        },
        {
            id: 'jacob', lifetime: 'param',
            run(S) {
                const wi = S.wi, ior = S.ior, n = 179, eps = 1e-4, below = wi[2] >= 0;
                const th = [], ana = [], rel = [], refl = [];
                for (let k = 0; k < n; k++) {
                    const t = (-89 + k) * Math.PI / 180;
                    const wo = [Math.sin(t), 0, below ? -Math.cos(t) : Math.cos(t)];
                    const J = TR.jacobT(wi, wo, ior);
                    const [t1, t2] = TR.tangents(wo);
                    const P = (u, v) => BX.norm([wo[0] + u * t1[0] + v * t2[0], wo[1] + u * t1[1] + v * t2[1], wo[2] + u * t1[2] + v * t2[2]]);
                    const ha = TR.halfT(wi, P(eps, 0), ior), hb = TR.halfT(wi, P(-eps, 0), ior), hc = TR.halfT(wi, P(0, eps), ior), hd = TR.halfT(wi, P(0, -eps), ior);
                    let Jn = NaN;
                    if (ha && hb && hc && hd) {
                        const c = BX.cross([ha[0] - hb[0], ha[1] - hb[1], ha[2] - hb[2]], [hc[0] - hd[0], hc[1] - hd[1], hc[2] - hd[2]]);
                        Jn = Math.hypot(c[0], c[1], c[2]) / (4 * eps * eps);
                    }
                    const wr = [Math.sin(t), 0, below ? Math.cos(t) : -Math.cos(t)], hr = TR.halfR(wi, wr);
                    th.push(deg(t)); ana.push(J); rel.push(Math.max(1e-12, Math.abs(Jn / J - 1))); refl.push(hr ? 1 / (4 * Math.abs(BX.dot(wr, hr))) : NaN);
                }
                S.jac = { th, ana, rel, refl };
            },
        },
        {
            id: 'samples', lifetime: 'param',
            run(S) {
                const wo = S.wi, a = S.a, ior = S.ior, rnd = PL.rng(S.seed ^ 0x9e3779b9);
                const px = PL.buf(S, 'smpX', NS), pz = PL.buf(S, 'smpZ', NS), kind = PL.buf(S, 'smpK', NS, Int8Array);
                const hist = new Float64Array(NBIN);
                let eR = 0, eT = 0, eR2 = 0, eT2 = 0, eImp = 0, nR = 0, nT = 0, nBad = 0;
                for (let k = 0; k < NS; k++) {
                    const u = BX.hammersley(k, NS), s = TR.sample(wo, a, ior, rnd(), u[0], u[1], 'radiance');
                    if (!s || !(s.weight > 0)) { kind[k] = 0; nBad++; continue; }
                    kind[k] = s.isT ? 2 : 1; s.isT ? nT++ : nR++;
                    px[k] = s.wi[0]; pz[k] = s.wi[2];
                    if (s.isT) { eT += s.weight; eT2 += s.weight * s.weight; } else { eR += s.weight; eR2 += s.weight * s.weight; }
                    eImp += s.isT ? s.weight * (TR.etaOf(s.wi, ior) ** 2) / (TR.etaOf(wo, ior) ** 2) : s.weight;
                    const b = Math.min(NBIN - 1, Math.floor(deg(Math.acos(TR.clamp(s.wi[2], -1, 1))) / (180 / NBIN)));
                    hist[b] += 1;
                }
                const binW = 180 / NBIN, histD = [], anaD = [], xs = [];
                for (let b = 0; b < NBIN; b++) {
                    histD.push(hist[b] / (NS * binW));
                    let acc = 0;
                    for (let sub = 0; sub < NSUB; sub++) {
                        const t = (b + (sub + 0.5) / NSUB) * binW * Math.PI / 180, st = Math.sin(t), ct = Math.cos(t);
                        for (let j = 0; j < NPHI; j++) {   // φ = π·s³：样本集中在入射平面 φ = 0（两个 lobe 的中心）
                            const u = -1 + (j + 0.5) * 2 / NPHI, ph = Math.PI * u * u * u, dph = 3 * Math.PI * u * u * (2 / NPHI);
                            acc += TR.pdf([st * Math.cos(ph), st * Math.sin(ph), ct], wo, a, ior) * st * dph;
                        }
                    }
                    anaD.push(acc / NSUB * Math.PI / 180);
                    xs.push((b + 0.5) * binW);
                }
                const mir = [-wo[0], -wo[1], wo[2]], up = wo[2] >= 0;
                const axT = BX.refract(wo, [0, 0, up ? 1 : -1], up ? ior : 1 / ior) || BX.norm([-wo[0], 0, up ? -0.05 : 0.05]);
                const qR = quadAligned((w) => TR.evalR(w, wo, a, ior) * Math.abs(w[2]), mir, 160, 96);
                const qT = quadAligned((w) => TR.evalT(w, wo, a, ior, 'radiance') * Math.abs(w[2]), axT, 160, 96);
                const sd = (m1, m2) => Math.sqrt(Math.max(0, m2 / NS - (m1 / NS) ** 2) / NS);   // MC 标准误差
                S.smp = { nR, nT, nBad, eR: eR / NS, eT: eT / NS, sR: sd(eR, eR2), sT: sd(eT, eT2), eImp: eImp / NS, qR, qT, hx: xs, hist: histD, ana: anaD };
            },
        },
        {
            id: 'energy', lifetime: 'param',
            run(S) {
                const ior = S.ior, dirO = (t) => [Math.sin(t), 0, Math.cos(t)], dirI = (t) => [Math.sin(t), 0, -Math.cos(t)];
                const ths = Array.from({ length: EN_N }, (_, k) => thetaDeg(k, EN_N));
                const curve = (a, dirF, n) => ths.map((t) => TR.albedoSplit(dirF(t * Math.PI / 180), a, ior, n));
                const key = ior.toFixed(3);
                if (S._enKey !== key) {   // fixed-roughness family depends only on η → memoize
                    S.enFam = EN_RS.map((r) => ({ r, out: curve(BX.alpha(r), dirO, 128), inn: curve(BX.alpha(r), dirI, 128) }));
                    S._enKey = key;
                }
                const cur = { out: curve(S.a, dirO, 256), inn: curve(S.a, dirI, 256) };
                const rs = Array.from({ length: 19 }, (_, k) => 0.1 + 0.05 * k);
                const tI = S.thetaI;
                S.en = {
                    ths, cur, rs,
                    vsR: rs.map((r) => [TR.albedoSplit(dirO(tI), BX.alpha(r), ior, 128), TR.albedoSplit(dirI(tI), BX.alpha(r), ior, 128)]),
                };
            },
        },
        {
            // 3D lobe（Step 06）：固定 ωi，f·|cos θo| 在整个 ωo 球面上；反射网格以镜面方向为极点，透射网格以光滑折射方向为极点
            id: 'lobe3d', lifetime: 'param', lazy: true, run(S) {
                const wi = S.wi, a = S.a, ior = S.ior, up = wi[2] >= 0, n = (TR_L3D.nu + 1) * (TR_L3D.nv + 1) * 3;
                const mir = [-wi[0], -wi[1], wi[2]], wt = BX.refract(wi, [0, 0, up ? 1 : -1], up ? ior : 1 / ior);
                const axT = wt || BX.norm([-wi[0], -wi[1], up ? -0.05 : 0.05]);   // TIR：没有光滑折射方向，网格极点取贴着界面的方向
                const mR = trLobeGrid((w) => TR.evalR(wi, w, a, ior) * Math.abs(w[2]), mir, PL.buf(S, 'l3dR', n));
                const mT = trLobeGrid((w) => TR.evalT(wi, w, a, ior) * Math.abs(w[2]), axT, PL.buf(S, 'l3dT', n));
                S.l3d = { mR, mT, mir, wt, max: Math.max(mR, mT) || 1 };
            },
        },
        {
            id: 'approx', lifetime: 'param',
            run(S) {
                const N = PT_RES, img = PL.buf(S, 'rtImg', N * N * 3);
                for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
                    const c = rtShade(camRay(x + 0.5, y + 0.5, N), S, [0, 0, 0]), i = (y * N + x) * 3;
                    img[i] = c[0]; img[i + 1] = c[1]; img[i + 2] = c[2];
                }
            },
        },
        {
            id: 'pt', lifetime: 'history',
            run(S) {
                const N = PT_RES, acc = PL.buf(S, 'ptAcc', N * N * 3);
                if (S._ptEpoch !== S.epoch) { acc.fill(0); S.ptSpp = 0; S._ptEpoch = S.epoch; }
                if (S.ptSpp >= PT_MAX) return;
                const rnd = PL.rng(((S.seed * 7919) ^ (S.ptSpp * 2654435761)) >>> 0), c = [0, 0, 0];
                for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
                    c[0] = c[1] = c[2] = 0;
                    ptTrace(camRay(x + rnd(), y + rnd(), N), S, rnd, c);
                    const i = (y * N + x) * 3;
                    acc[i] += c[0]; acc[i + 1] += c[1]; acc[i + 2] += c[2];
                }
                S.ptSpp++;
            },
        },
    ],

    intro: {
        title: '粗糙电介质：光滑界面 → Walter 2007 BTDF → 采样 → 能量 → 实时',
        flow: ['Snell / TIR', '精确 Fresnel R/T', '广义半向量 h_t', '雅可比 dω_h/dω_o', 'Walter BTDF', '内外两侧 lobe', 'η² 非互易', 'VNDF + R/T 采样', '白炉能量', '实时粗糙折射'],
        html: `<p>玻璃、水、宝石、清漆层的底层都是同一个对象：<strong>电介质界面</strong>——入射光按 Fresnel 一分为二，一部分反射、其余折射进另一侧介质，几乎不吸收。本页先把光滑界面（delta BSDF）讲透，再把它推广到 GGX 微表面上，得到 Walter et al. 2007 的粗糙透射 BTDF，最后讲重要性采样、能量检查和实时渲染里的近似。</p>
        <p>约定沿用系列：局部坐标 <code>n = +Z</code>，<code>ωi</code> 指向光源、<code>ωo</code> 指向相机，二者都背离表面；α = r²。本模块额外规定<strong>上半球 = 外部空气（η = 1）、下半球 = 介质内部（η = IOR）</strong>，<code>η_i / η_o</code> 是 ωi / ωo 所在一侧的折射率。顶栏 <code>θi</code> 与“入射侧”控制所有 lobe、热图与采样图；每张图都由页面内的 JS 参考实现（与 HLSL 同式）实时算出，悬停热图可读数。</p>
        <p>D、Λ、G 与 VNDF 来自 <a href="bxdf_02_microfacet.html">02 微表面</a>（<code>bxdf_common.js</code>）；本模块的透射项会被 <a href="bxdf_06_layering.html">06 多层混合</a> 的 clearcoat / 分层 BSDF 复用。</p>` + BX.nav('transmission'),
    },

    steps: [
        // ───────────────────────────── 01 SNELL
        {
            id: 'snell', title: 'Snell 定律、折射方向与全内反射', tag: 'SNELL',
            desc: '光滑界面上，光线方向的<strong>切向分量</strong>乘以折射率在界面两侧连续（相位匹配 / 光子切向动量守恒），由此得到 Snell 定律。从密介质射向疏介质（η < 1）时，sin θt 可能 > 1 —— 没有折射解，全部能量反射，这就是<strong>全内反射 TIR</strong>。玻璃（η = 1.5）的临界角约 41.8°：从水下或玻璃内部往外看，只有一个 ±θc 的“Snell 窗”能看到外面。',
            formula: [
                'η_i sin θ_i = η_t sin θ_t                          // η = η_t / η_i（本系列的相对 IOR 定义）',
                'cos θ_t = √(1 − (1 − cos^2 θ_i) / η^2)',
                'ω_t = −ω_i / η + (cos θ_i / η − cos θ_t) · n       // ω_i 背离表面，n 与 ω_i 同侧 = BX.refract',
                'sin θ_c = 1 / η     (η < 1 才存在)                 // sin^2 θ_t ≥ 1 ⇒ TIR，返回 null / 0',
            ],
            code: { title: 'HLSL · RefractDir（与 BX.refract 同约定）', src: `
                // wi 背离表面、n 与 wi 同侧、eta = η_t / η_i；TIR 返回 0
                float3 RefractDir(float3 wi, float3 n, float eta) {
                    float cosI  = dot(n, wi);
                    float sin2T = (1.0 - cosI * cosI) / (eta * eta);
                    if (sin2T >= 1.0) return 0;                  // 全内反射
                    return -wi / eta + (cosI / eta - sqrt(1.0 - sin2T)) * n;
                }
                // 等价写法：refract(-wi, n, 1.0 / eta) —— 内置函数的入射向量指向表面、eta 是 η_i/η_t` },
            previews: [
                { id: 'snellOut', label: '外部入射 · 空气 → 介质', sub: '线宽 ∝ 能量份额（R = F，T = 1 − F）', size: [270, 210], render: (c, S) => drawSnell(c, S, false) },
                { id: 'snellIn', label: '内部入射 · 介质 → 空气', sub: 'θi 超过 θc 时折射光消失', size: [270, 210], render: (c, S) => drawSnell(c, S, true) },
                {
                    id: 'snellCurve', label: 'θt(θi)', sub: '两个方向的折射角；虚线 = 不折射', size: [330, 210],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, 90], y: [0, 90], xLabel: 'θi (°)', yLabel: 'θt (°)',
                        series: [
                            { fn: (t) => deg(Math.asin(Math.sin(t * Math.PI / 180) / S.ior)), color: '#6ab0e0', label: '空气 → 介质' },
                            { fn: (t) => { const s = Math.sin(t * Math.PI / 180) * S.ior; return s < 1 ? deg(Math.asin(s)) : NaN; }, color: '#e0a060', label: '介质 → 空气' },
                            { fn: (t) => t, color: '#444', dash: [3, 3] },
                        ],
                        marks: [{ x: deg(S.thetaC), label: `θc ${deg(S.thetaC).toFixed(1)}°`, color: '#e06c75' }, { x: deg(S.thetaI), label: 'θi', color: '#e8c25a' }],
                    }),
                },
            ],
            notes: '拖 <code>θi</code>：外部入射时折射角总小于入射角，并压缩到 [0, θc]；内部入射超过 θc 立即 TIR（右图橙线断开）。拖 <code>IOR</code>：钻石（2.42）的临界角只有 24°，所以它的“火彩”来自大量内部全反射。',
            warn: 'HLSL / GLSL 内置 <code>refract(I, N, eta)</code> 的 <code>I</code> 指向表面、<code>eta = η_i/η_t</code>，并要求 <code>dot(N, I) &lt; 0</code>；与本系列（ω 背离表面、η = η_t/η_i）正好相反。混用时最常见的症状是：玻璃球看起来像放大镜反了、或者内部永远不发生 TIR。',
        },
        // ───────────────────────────── 02 FRESNEL
        {
            id: 'fresnel', title: '精确 Fresnel：R / T 能量分配与光滑 delta BSDF', tag: 'FRESNEL',
            desc: '界面不吸收能量，所以 <strong>T = 1 − F</strong>。Fresnel 方程按偏振分别给出 s / p 的振幅反射率；渲染器通常假设非偏振光，取二者平均。p 偏振在 Brewster 角完全透射；从内部入射到 θc 时 R 迅速升到 1。光滑界面的 BSDF 是两个 delta：只有恰好的镜面反射方向和折射方向有值，所以它<strong>不能求值，只能采样</strong>——以概率 F 选反射、1 − F 选折射，权重恰好约掉成 1（折射多一个 η² 因子，见 ETA2）。',
            formula: [
                'r_s = (cos θ_i − η cos θ_t) / (cos θ_i + η cos θ_t)',
                'r_p = (η cos θ_i − cos θ_t) / (η cos θ_i + cos θ_t)',
                'F = (r_s^2 + r_p^2) / 2,    T = 1 − F                  // TIR 时 F = 1',
                'tan θ_B = η   // Brewster：r_p = 0',
                { label: '光滑电介质 = delta BSDF', text: 'f_R(ω_i, ω_o) = F · δ(ω_i − R(ω_o)) / |cos θ_i|\nf_T(ω_i, ω_o) = (1 − F) · (η_o/η_i)^2 · δ(ω_i − T(ω_o)) / |cos θ_i|   // radiance 模式' },
            ],
            code: [
                { title: 'HLSL · F_Dielectric（精确，含 TIR）', src: `
                    // cosI < 0 = 从另一侧入射（η 取倒数）；TIR → 1。与 BX.fresnelDielectric 一致
                    float F_Dielectric(float cosI, float eta) {
                        if (cosI < 0.0) { eta = 1.0 / eta; cosI = -cosI; }
                        float sin2T = (1.0 - cosI * cosI) / (eta * eta);
                        if (sin2T >= 1.0) return 1.0;
                        float cosT = sqrt(1.0 - sin2T);
                        float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
                        float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
                        return 0.5 * (rs * rs + rp * rp);
                    }` },
                { title: 'HLSL · Sample_SmoothDielectric（delta BSDF）', src: HLSL_DELTA },
            ],
            previews: [
                {
                    id: 'fresOut', label: '空气 → 介质', sub: 's / p / 平均 F 与 T = 1 − F', size: [330, 210],
                    render: (c, S) => {
                        const sp = (t) => TR.fresnelSP(Math.cos(t * Math.PI / 180), S.ior);
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, 1], xLabel: 'θi (°)',
                            series: [
                                { fn: (t) => sp(t)[0], color: '#c678dd', dash: [4, 3], label: 'R_s' },
                                { fn: (t) => sp(t)[1], color: '#56b6c2', dash: [4, 3], label: 'R_p' },
                                { fn: (t) => BX.fresnelDielectric(Math.cos(t * Math.PI / 180), S.ior), color: '#e0a060', width: 2, label: 'R = F' },
                                { fn: (t) => 1 - BX.fresnelDielectric(Math.cos(t * Math.PI / 180), S.ior), color: '#6ab0e0', width: 2, label: 'T = 1 − F' },
                            ],
                            marks: [{ x: deg(Math.atan(S.ior)), label: 'θB', color: '#56b6c2' }, { x: deg(S.thetaI), label: 'θi', color: '#e8c25a' }],
                        });
                    },
                },
                {
                    id: 'fresIn', label: '介质 → 空气', sub: '虚线 = Schlick(cos θi)：内部入射时缺 TIR', size: [330, 210],
                    render: (c, S) => {
                        const F0 = BX.f0FromIor(S.ior);
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, 1], xLabel: 'θi (°)',
                            series: [
                                { fn: (t) => BX.fresnelDielectric(-Math.cos(t * Math.PI / 180), S.ior), color: '#e0a060', width: 2, label: 'R = F（精确）' },
                                { fn: (t) => 1 - BX.fresnelDielectric(-Math.cos(t * Math.PI / 180), S.ior), color: '#6ab0e0', width: 2, label: 'T = 1 − F' },
                                { fn: (t) => BX.schlick(F0, Math.cos(t * Math.PI / 180)), color: '#e06c75', dash: [3, 3], label: 'Schlick(cos θi) ✗' },
                                {
                                    fn: (t) => { const s = Math.sin(t * Math.PI / 180) * S.ior; return s < 1 ? BX.schlick(F0, Math.sqrt(1 - s * s)) : 1; },
                                    color: '#5ad07a', dash: [1, 2], label: 'Schlick(cos θt) ✓',
                                },
                            ],
                            marks: [{ x: deg(S.thetaC), label: 'θc', color: '#e06c75' }, { x: deg(Math.atan(1 / S.ior)), label: 'θB', color: '#56b6c2' }, { x: deg(S.thetaI), label: 'θi', color: '#e8c25a' }],
                        });
                    },
                },
            ],
            notes: '正入射 F0 = ((η−1)/(η+1))²：玻璃 4%、水 2%、钻石 17%。外部入射时 F 在掠射角才明显上升；内部入射则在 θc 之前就上升、过了 θc 恒为 1。拖 <code>IOR</code> → 1 时两条曲线都趋近 0（界面“消失”）。',
            warn: [
                '内部入射时直接用 <code>Schlick(F0, cos θi)</code> 是错的（红虚线）：它在 θc 之后仍然 &lt; 1，玻璃内部的 TIR 会漏光。要么用精确式，要么改用<strong>透射角</strong> <code>Schlick(F0, cos θt)</code>，并在 TIR 时返回 1（绿点线）。',
                '光滑分支的权重：反射 1、折射 (η_o/η_i)²。不要把 delta 分支塞进普通的 Eval / MIS——它的 pdf 是 delta，Eval 必须返回 0，否则光源采样会给它加上一份错误的能量。',
            ],
        },
        // ───────────────────────────── 03 HALF
        {
            id: 'half', title: '广义折射半向量 h_t 及其朝向', tag: 'HALF',
            desc: '粗糙界面上每个微面元都是一块光滑小镜面；要把 (ωi, ωo) 这一对方向归因到“哪个微面元把光折过去的”，就需要透射版本的半向量。把 Snell 定律写成向量：<strong>η_i·ωi 与 η_o·ωo 的切向分量大小相等、方向相反</strong>（两者都背离表面），所以二者之和只剩沿微法线的分量 —— 和就平行于 h。直觉上这是光子动量 η·d 在界面上只能改变法向分量。Walter 选 <code>h_t = −(η_i ωi + η_o ωo)</code>：这个符号让 h_t 总指向<strong>折射率较低的一侧</strong>；IOR &gt; 1 时就是 +n，正好落在 D(h) 的定义域里。为了对 IOR &lt; 1（气泡）同样成立，本文再统一翻到 +n 侧。',
            formula: [
                'h_r = normalize(ω_i + ω_o)                        // 反射：ω_i、ω_o 同侧',
                'h_t = −normalize(η_i ω_i + η_o ω_o)               // 透射（Walter Eq.16）：指向低 IOR 一侧',
                'h_t ← h_t · sign(h_t · n)                         // 本文约定：统一翻到 +n 侧',
                'η_i (ω_i − (ω_i·h)h) = −η_o (ω_o − (ω_o·h)h)    ⇒    η_i ω_i + η_o ω_o = (η_i ω_i·h + η_o ω_o·h) · h',
                'valid ⇔ (ω_i·h)(ω_i·n) > 0  ∧  (ω_o·h)(ω_o·n) > 0   // 两个方向都在微面元的“正面”(χ⁺)',
            ],
            code: { title: 'HLSL · HalfVectorT / ValidT', src: HLSL_HALF },
            previews: [
                {
                    id: 'halfDiag', label: '向量构造', sub: '任取一个倾斜微面元 m，把 ωi 折过去得 ωo，再由 (ωi, ωo) 反求 h_t', size: [320, 250],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c);
                        const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 18, L = R * 0.55;
                        mediumBg(ctx, w, h, cy, S);
                        const wi = S.wi, ior = S.ior, tau = (12 + 14 * S.rough) * Math.PI / 180;
                        const m = [Math.sin(tau), 0, Math.cos(tau)], mS = wi[2] >= 0 ? m : TR.neg(m);
                        const ei = TR.etaOf(wi, ior), eo = wi[2] >= 0 ? ior : 1;
                        const wo = BX.refract(wi, mS, eo / ei);
                        const P = (v, s) => [cx + v[0] * s, cy - v[2] * s];
                        ctx.strokeStyle = '#8a9aa6'; ctx.lineWidth = 3;
                        ctx.beginPath(); ctx.moveTo(cx - Math.cos(tau) * 34, cy - Math.sin(tau) * 34); ctx.lineTo(cx + Math.cos(tau) * 34, cy + Math.sin(tau) * 34); ctx.stroke();
                        ctx.setLineDash([2, 3]); ctx.strokeStyle = '#46525c'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke(); ctx.setLineDash([]);
                        const [ix, iy] = P(wi, R);
                        arrow(ctx, ix, iy, cx, cy, '#e8c25a', 1.8);
                        const [mx, my] = P(m, R * 0.62);
                        arrow(ctx, cx, cy, mx, my, '#8a9aa6', 1);
                        ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
                        if (!wo) { ctx.fillStyle = '#e06c75'; ctx.fillText('该微面元上 TIR：没有透射解', 6, 4); return; }
                        const [ox, oy] = P(wo, R);
                        arrow(ctx, cx, cy, ox, oy, '#6ab0e0', 1.8);
                        // head-to-tail: A = −η_i ωi, then B = −η_o ωo, sum ∥ h_t
                        const A = [-ei * wi[0], 0, -ei * wi[2]], B = [-eo * wo[0], 0, -eo * wo[2]];
                        const [ax, ay] = P(A, L), [sx, sy] = P([A[0] + B[0], 0, A[2] + B[2]], L);
                        arrow(ctx, cx, cy, ax, ay, 'rgba(232,194,90,0.75)', 1.2, 4);
                        arrow(ctx, ax, ay, sx, sy, 'rgba(106,176,224,0.8)', 1.2, 4);
                        arrow(ctx, cx, cy, sx, sy, '#c678dd', 2.2, 5);
                        const ht = TR.halfT(wi, wo, ior), err = deg(Math.acos(TR.clamp(BX.dot(ht, m), -1, 1)));
                        ctx.fillStyle = '#e8c25a'; ctx.fillText('ωi', ix + 4, iy + 2);
                        ctx.fillText('−ηᵢωi', ax + 4, ay - 4);
                        ctx.fillStyle = '#6ab0e0'; ctx.fillText('ωo', ox + 4, oy - 12);
                        ctx.fillStyle = '#8a9aa6'; ctx.fillText('m', mx + 5, my - 2);
                        ctx.fillStyle = '#c678dd'; ctx.fillText('紫 = −(ηᵢωi + ηₒωo) ∥ hₜ', 6, 4);
                        ctx.fillStyle = '#9aa'; ctx.fillText(`∠(hₜ, m) = ${err.toExponential(1)}°   倾角 ${deg(tau).toFixed(0)}°`, 6, 17);
                        ctx.fillText(`ηᵢ·sinθ′ᵢ = ${(ei * Math.sqrt(Math.max(0, 1 - BX.dot(wi, m) ** 2))).toFixed(4)}   ηₒ·sinθ′ₒ = ${(eo * Math.sqrt(Math.max(0, 1 - BX.dot(wo, m) ** 2))).toFixed(4)}`, 6, 30);
                    },
                },
                {
                    id: 'halfMap', label: 'θ_h(ωo) 透射半球', sub: '固定 ωi，扫遍另一侧所有 ωo；黑 = 无效配置（χ⁺ 不满足）',
                    render: (c, S) => PL.draw.field(c, S.mapTh, HRES, HRES, { map: 'viridis', range: [0, 90] }),
                    probe: (x, y, S) => { const v = S.mapTh[y * HRES + x]; return isNaN(v) ? '无效：微面元背对某个方向' : `θ_h = ${v.toFixed(1)}°`; },
                },
            ],
            notes: '左图：灰色短线是微面元，金色 = ωi，蓝色 = 由 Snell 在 m 上折出的 ωo；紫色合成向量与 m 重合（夹角 ~1e-14°）。右图是从透射侧看的等面积半球投影（光从 −X 来）：θ_h = 0 的暗点就是光滑折射方向，越往外需要越倾斜的微面元；外圈黑色区域需要“背面朝向”的微面元，BTDF 在那里为 0：从空气进入介质时，任何微面元最多只能把光偏折 90° − θc（玻璃 48.2°），所以有效区域是绕“直穿方向 −ωi”的一块有限区域。切到<strong>介质 → 外部</strong>且 θi 较大时，黑区明显变大——这正是粗糙 TIR 的几何来源。',
            warn: '符号约定必须三处一致：h_t 的方向（本文 +n 侧）、F 的参数 <code>dot(ωi, h)</code> 的正负（负 = 从内部入射）、以及 χ⁺ 检查。只要其中一个用了 <code>abs</code> 而另一个没有，就会出现“背面微面元也在透光”的亮斑。',
        },
        // ───────────────────────────── 04 JACOB
        {
            id: 'jacob', title: '雅可比 dω_h / dω_o：从微法线空间换到出射方向', tag: 'JACOB',
            desc: 'D(h) 是<strong>微法线 h 的立体角密度</strong>，而 BTDF 是对 <strong>ωo 的立体角</strong>定义的；从一个换到另一个必须乘雅可比。反射时 h 是 ωi、ωo 的平分线，ωo 的立体角是 h 的 4|ωo·h| 倍；透射时 ωo 与 h 的关系受 η 拉伸，Walter 推出了 Eq.17。分母 <code>(η_i ωi·h + η_o ωo·h)²</code> 后面会直接出现在 BTDF 里——它就是这个换元的痕迹。右图把解析式与<strong>数值差分</strong>（在 ωo 周围取 ±ε 的小平行四边形，量出对应 h 平行四边形的面积比）对照，相对误差应 ≲ 1e-6。',
            formula: [
                'dω_h / dω_o = η_o^2 |ω_o·h_t| / (η_i (ω_i·h_t) + η_o (ω_o·h_t))^2     // 透射（Walter Eq.17）',
                'dω_h / dω_o = 1 / (4 |ω_o·h_r|)                                        // 反射（对照）',
                'dω_h / dω_i = η_i^2 |ω_i·h_t| / (η_i (ω_i·h_t) + η_o (ω_o·h_t))^2     // 采样时固定 ω_o、换 ω_i',
                'J_num = |Δh_1 × Δh_2| / |Δω_1 × Δω_2|                                  // 数值验证：ε = 1e−4',
            ],
            code: { title: 'HLSL · JacobianT / JacobianR', src: HLSL_JACOB },
            previews: [
                {
                    id: 'jacobPlot', label: 'dω_h/dω_o', sub: '透射（实线）vs 反射（虚线）；横轴 = ωo 相对其所在侧法线的带符号角', size: [340, 220],
                    render: (c, S) => {
                        const vmax = Math.max(...S.jac.ana.filter(isFinite));
                        const tOpt = S.wi[2] >= 0 ? BX.refract(S.wi, [0, 0, 1], S.ior) : BX.refract(S.wi, [0, 0, -1], 1 / S.ior);
                        PL.draw.plot(c, {
                            x: [-89, 89], y: [0, vmax * 1.08], xLabel: 'θo 带符号 (°)',
                            series: [
                                { data: S.jac.ana, x: S.jac.th, color: '#6ab0e0', width: 2, label: '透射 Eq.17' },
                                { data: S.jac.refl, x: S.jac.th, color: '#e0a060', dash: [4, 3], label: '反射 1/(4|o·h|)' },
                            ],
                            marks: [
                                ...(tOpt ? [{ x: deg(Math.asin(tOpt[0])), label: '光滑折射', color: '#5ad07a' }] : []),
                                { x: deg(S.thetaI), label: '直穿 −ωi', color: '#c678dd' },
                            ],
                        });
                    },
                },
                {
                    id: 'jacobErr', label: '|J_num / J − 1|', sub: '数值差分 vs 解析式（log）', size: [300, 220],
                    render: (c, S) => PL.draw.plot(c, { x: [-89, 89], y: [1e-12, 1], logY: true, xLabel: 'θo 带符号 (°)', series: [{ data: S.jac.rel, x: S.jac.th, color: '#c678dd', width: 1.5 }] }),
                },
            ],
            notes: '外部入射（η_o = 1.5）时透射雅可比远大于反射（光滑折射方向上约 5.9 vs 0.33）：同一片微法线 h 只对应很小一块 ωo 立体角，所以介质内的透射 lobe 很紧凑；最大值 η_o²/(η_o − η_i)² = 9 出现在“直穿方向 −ωi”（微面元法线平行于光线、不偏折）。切到<strong>介质 → 外部</strong>（η_o = 1、η_i = 1.5）：最大值降到 1/(1.5 − 1)² = 4，大部分方向上雅可比都小于外部入射的情形——同一片 h 被“摊开”到更大的出射立体角，所以从介质里出来的透射 lobe 更宽。差分误差图在 1e-8 量级的平台是 ε² 截断误差加浮点误差。',
            warn: '实现采样 pdf 时最容易用错的是<strong>该对谁求导</strong>：Eval 的推导固定 ωi、对 ωo 求导（分子 η_o²）；路径追踪从相机出发，固定 ωo、对 ωi 采样，pdf 里要用分子是 η_i² 的那一个（第三行）。',
        },
        // ───────────────────────────── 05 BTDF
        {
            id: 'btdf', title: 'Walter BTDF：(1 − F)·D·G₂ × 雅可比', tag: 'BTDF',
            desc: '思路与 Cook-Torrance 完全相同：先在微法线空间里数清“有多少可见的、朝向正确的微面元能把 ωi 折到 ωo”——<code>(1 − F)·D(h)·G₂</code>，再乘上 ωi 一侧的投影比 <code>|ωi·h| / |ωi·n|</code>，最后用 JACOB 的雅可比换到 ωo 的立体角，并除以 <code>|ωo·n|</code>（BSDF 的定义里带一个 cos）。把雅可比代进去就得到 Walter Eq.21。下面 5 张热图是同一个 (ωi, 透射半球 ωo) 上各因子<strong>真实逐像素</strong>的值，最后一张是它们的乘积 f_t·|cos θo| —— 也就是固定入射光时透射能量在出射方向上的分布。',
            formula: [
                'f_t(ω_i,ω_o) = (1 − F(ω_i·h)) · D(h) · G_2(ω_i,ω_o) · |ω_i·h| / |ω_i·n| · (dω_h/dω_o) / |ω_o·n|',
                '             = |ω_i·h||ω_o·h| / (|ω_i·n||ω_o·n|) · η_o^2 (1 − F) D G_2 / (η_i (ω_i·h) + η_o (ω_o·h))^2   // Walter Eq.21',
                'f_r(ω_i,ω_o) = F(ω_i·h_r) · D(h_r) · G_2 / (4 |ω_i·n||ω_o·n|)                                         // 同侧：普通微表面反射',
                'f = f_r + f_t     // 同侧 → f_r，异侧 → f_t；G_2 = 1/(1 + Λ(ω_i) + Λ(ω_o))',
            ],
            code: { title: 'HLSL · Eval_RoughDielectric（R + T）', src: HLSL_BSDF },
            previews: ['D', 'G', 'T', 'J', 'f'].map((k) => ({
                id: 'term' + k,
                label: { D: 'D(h_t)', G: 'G₂(ωi, ωo)', T: '1 − F(ωi·h_t)', J: '投影 × 雅可比', f: 'f_t · |cos θo|' }[k],
                sub: { D: '微法线密度（dB，4 个数量级）', G: '高度相关 Smith 遮蔽', T: 'Fresnel 透射份额', J: '|i·h|/|i·n| · dω_h/dω_o（sqrt）', f: '乘积 = 透射能量分布（dB）' }[k],
                render: (c, S) => (k === 'D' || k === 'f' ? PL.draw.field(c, S.maps[k + 'db'], HRES, HRES, { map: 'inferno', range: [-40, 0] })
                    : PL.draw.field(c, S.maps[k], HRES, HRES, k === 'J' ? { map: 'inferno', stretch: 'sqrt' } : { map: 'viridis', range: [0, 1] })),
                probe: (x, y, S) => {
                    const i = y * HRES + x, m = S.maps;
                    return isNaN(m.f[i]) ? '无效配置 → f_t = 0' : `D ${m.D[i].toPrecision(3)} · G₂ ${m.G[i].toFixed(3)} · (1−F) ${m.T[i].toFixed(3)} · J ${m.J[i].toPrecision(3)} → f·cos ${m.f[i].toPrecision(3)}`;
                },
            })),
            notes: '拖 <code>roughness</code>：D 的亮斑从一个点扩散到大半个半球，G₂ 只在边缘变暗。拖 <code>θi</code> 到 70° 以上：透射 lobe 变成偏向掠射方向的“彗星”，因为雅可比把靠近 θc 一侧的 h 挤到了更小的立体角里。切到<strong>介质 → 外部</strong>：1 − F 图在 TIR 区直接变黑，只剩足够倾斜、局部入射角 &lt; θc 的微面元还能透光。',
            warn: [
                'Walter 原文的 G 是可分离的 G₁(i)·G₁(o)；这里与全系列一致用高度相关的 <code>1/(1+Λi+Λo)</code>（pbrt-v4 同）。Heitz 2014 指出透射时 ωi、ωo 分处两侧、高度相关性的方向与反射相反，严格形式并不相同——数值上差别在高粗糙度掠射角才明显。',
                '分母 <code>η_i(ωi·h) + η_o(ωo·h)</code> 在 η → 1 且 ωo ≈ −ωi 时趋于 0（h_t 退化）。IOR 下限不要放到 1.0；需要“无界面”时直接跳过 BSDF。',
            ],
        },
        // ───────────────────────────── 06 LOBE
        {
            id: 'lobe', title: '全圆 lobe：外部 vs 内部入射', tag: 'LOBE',
            desc: '把 f·|cos θo| 画在入射平面的整个圆上（上半圆 = 空气、下半圆 = 介质，<code>BX.lobe full: true</code>）：橙色是反射 lobe，蓝色是透射 lobe，绿色虚线是光滑界面的折射方向。左边两张是线性刻度，右边两张是 3 个数量级的对数刻度（看尾部）。两种入射的对比很直观：从外部打进来时绝大部分能量进了介质；从内部打出去时，一旦 θi 超过 θc，透射 lobe 只靠倾斜的微面元“漏”出一点，反射 lobe 成为主角。',
            formula: [
                'lobe(ω_o) = f(ω_i, ω_o) · |cos θ_o|      // 固定入射光，出射能量的角分布；∫ lobe dω_o = E_R + E_T',
                'ω_i = (−sin θ_i, 0, ±cos θ_i)            // + 外部入射，− 内部入射',
            ],
            previews: [
                ['out', false], ['in', false], ['out', true], ['in', true],
            ].map(([side, log]) => ({
                id: `lobe_${side}${log ? '_log' : ''}`,
                label: (side === 'out' ? '外部入射' : '内部入射') + (log ? ' · log' : ''),
                sub: side === 'out' ? '光从空气一侧打来（金色）' : '光在介质内打向界面',
                size: [250, 250],
                render: (c, S) => {
                    const wi = side === 'out' ? S.wiOut : S.wiIn;
                    BX.lobe(c, {
                        wi, full: true, cos: true, log: log ? 3 : 0,
                        series: [
                            { fn: (wo) => TR.evalR(wi, wo, S.a, S.ior), color: '#e0a060', label: 'f_r·|cos|' },
                            { fn: (wo) => TR.evalT(wi, wo, S.a, S.ior), color: '#6ab0e0', label: 'f_t·|cos|' },
                        ],
                    });
                    lobeOverlay(c, S, wi);
                },
            })).concat([{
                id: 'lobe3d', needs: ['lobe3d'], large: true, label: '3D：界面上下的反射 lobe 与透射 lobe（整个 ω<sub>o</sub> 球面）',
                sub: '固定 ω<sub>i</sub>（金色箭头，"入射侧"决定它在哪一侧）。橙 = f<sub>r</sub>·|cosθo|，蓝 = f<sub>t</sub>·|cosθo|（radiance 模式）；半径用对数刻度（3 个数量级，与右侧 log 图相同，两者共用同一个峰值）——线性刻度下透射 lobe 只是一根针。线框平面 = 界面（上 = 空气，下 = 介质）；虚线 = 镜面反射 / 光滑折射方向（TIR 时没有折射方向）',
                render: (c, S) => trDrawLobe3d(c, S),
            }]),
            notes: '拖 <code>roughness</code> 到 0.1：两个 lobe 收缩成指向镜面方向与绿色折射方向的尖刺（线性图的 max 很大）。拖 <code>θi</code> 跨过 θc（玻璃 41.8°）：右下“内部入射”的蓝色 lobe 从主 lobe 迅速缩成一条贴着界面的细尾巴。拖 <code>IOR</code>：外部入射的透射 lobe 随 η 增大更贴近法线（折射更强）。',
            warn: '透射 lobe 的峰值并不在光滑折射方向上——粗糙时它系统地偏向掠射一侧（D 对称，但雅可比与 G₂ 不对称）。用“绕折射方向旋转的对称 lobe”（如 Phong 式近似）做实时粗糙折射会把模糊的中心放错位置。',
        },
        // ───────────────────────────── 07 ETA2
        {
            id: 'eta2', title: 'η² 辐亮度缩放与 BTDF 的非互易', tag: 'ETA2',
            desc: '光从空气进入玻璃时，同一束能量被压进更小的立体角（θc 锥），所以<strong>辐亮度放大 η² 倍</strong>；出来时再缩回去。守恒的是 <code>L/η²</code>（basic radiance）。Walter Eq.21 的分子 η_o² 正是这件事：f_t 是给“光从 ωi 来、相机在 ωo”的 radiance 传输写的，它<strong>不满足</strong> f(ωi,ωo) = f(ωo,ωi)，而满足广义互易 <code>f/η_o² 对称</code>（Veach 1997）。后果：从相机出发的路径要用 radiance 版；从光源出发（光子、BDPT 的 light subpath）要用伴随版 f* = f(ωo, ωi)，差一个常数 (η_i/η_o)²。左图验证互易关系，右图说明为什么“相机视角的 albedo”在介质内部会大于 1——这不是能量不守恒。',
            formula: [
                'L_t = (η_t / η_i)^2 · (1 − F) · L_i              // 光滑界面：radiance 随立体角压缩放大',
                'f_t(ω_i, ω_o) / η_o^2 = f_t(ω_o, ω_i) / η_i^2       // 广义互易（Veach 1997 第 5 章）',
                'f_t^*(ω_i, ω_o) = f_t(ω_o, ω_i) = (η_i/η_o)^2 · f_t(ω_i, ω_o)   // importance / 光子传输',
                'E_cam(ω_o) = ∫ f(ω_i,ω_o) |cos θ_i| dω_i = E_R + (η_o/η_i)^2 · E_T   // 相机视角：外部 ×1/η²、内部 ×η²',
            ],
            code: { title: 'HLSL · radiance / importance 两种模式', src: HLSL_ETA2 },
            previews: [
                {
                    id: 'recip', label: '互易检查（外部 ωi，扫下半圆 ωo）', sub: 'f_t(ωi→ωo) vs f_t(ωo→ωi)：差 η²；×η² 后重合', size: [340, 220],
                    render: (c, S) => {
                        const wi = S.wiOut, xs = [], A = [], B = [], C = [];
                        for (let k = 0; k <= 178; k++) {
                            const t = (-89 + k) * Math.PI / 180, wo = [Math.sin(t), 0, -Math.cos(t)];
                            xs.push(-89 + k);
                            const fa = TR.evalT(wi, wo, S.a, S.ior), fb = TR.evalT(wo, wi, S.a, S.ior);
                            A.push(fa > 0 ? fa : NaN); B.push(fb > 0 ? fb : NaN); C.push(fb > 0 ? fb * S.ior * S.ior : NaN);
                        }
                        PL.draw.plot(c, {
                            logY: true, x: [-89, 89], xLabel: 'θo 带符号 (°)',
                            series: [
                                { data: A, x: xs, color: '#6ab0e0', width: 2.5, label: 'f_t(ωi, ωo)' },
                                { data: B, x: xs, color: '#e0a060', width: 1.5, label: 'f_t(ωo, ωi)' },
                                { data: C, x: xs, color: '#e8c25a', width: 1.2, dash: [4, 3], label: 'f_t(ωo, ωi) · η²' },
                            ],
                        });
                    },
                },
                {
                    id: 'ecam', label: '相机视角 albedo vs 能量（当前 r）', sub: '实线 = E_R + E_T（flux，≤ 1）；虚线 = radiance 模式 E_cam', size: [340, 220],
                    render: (c, S) => {
                        const e = S.en, eta2 = S.ior * S.ior;
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, Math.max(2.4, eta2 * 1.02)], xLabel: 'θ (°)',
                            series: [
                                { data: e.cur.out.map((v) => v[0] + v[1]), x: e.ths, color: '#6ab0e0', width: 2, label: '外部 E_R+E_T' },
                                { data: e.cur.out.map((v) => v[0] + v[1] / eta2), x: e.ths, color: '#6ab0e0', dash: [4, 3], label: '外部 E_cam' },
                                { data: e.cur.inn.map((v) => v[0] + v[1]), x: e.ths, color: '#e0a060', width: 2, label: '内部 E_R+E_T' },
                                { data: e.cur.inn.map((v) => v[0] + v[1] * eta2), x: e.ths, color: '#e0a060', dash: [4, 3], label: '内部 E_cam' },
                            ],
                            hmarks: [{ y: 1, label: '1', color: '#555' }],
                            marks: [{ x: deg(S.thetaC), label: 'θc', color: '#e06c75' }],
                        });
                    },
                },
            ],
            notes: '左图：蓝线与金色虚线完全重合，说明本页的 f_t 精确满足广义互易；橙线比它低 η² 倍。右图：相机在介质内部（橙虚线）看到的 albedo 接近 η²(1 − F) ≈ 2.2——因为整个外部半球被压缩进了 θc 锥。正确的白炉要在内部放 η² 倍的辐亮度（热平衡时 L ∝ η²），或者直接看 flux 版（实线）。',
            warn: [
                '同一条路径上进、出介质的 η² 因子会相消（1/η² · η²），所以只关心最终图像的单向路径追踪器“忘了 η²”时常常看不出错——直到相机放进水下、或者用了光子映射 / BDPT。',
                'pbrt-v4 的 <code>TransportMode::Radiance</code> 在透射时除以 η_rel²（其 η_rel = η_i/η_o）；本页的 radiance 模式与之等价，只是写成了 Walter 的分子 η_o² 形式。',
            ],
        },
        // ───────────────────────────── 08 SAMPLE
        {
            id: 'sample', title: '重要性采样：VNDF 采 h → 按 Fresnel 选 R / T → 带雅可比的 pdf', tag: 'SAMPLE',
            desc: '采样器从固定方向出发（路径追踪里是相机方向 ωo；这里把它摆在顶栏 θi / 入射侧给定的位置），先用 Heitz 2018 的 <strong>VNDF</strong> 采一个从 ωo 可见的微法线 h（ωo 在介质内时先把它翻到上半球），再以概率 F(ωo·h) 选反射、否则折射——TIR 时 F = 1 自动只剩反射。pdf 是 “h 的密度 × 选择概率 × h→ωi 的雅可比”。代入后几乎所有项都约掉：<strong>权重 = G₂/G₁(ωo)</strong>，折射再乘 (η_o/η_i)²。下面散点是 4096 个真实样本（投影到 x–z 平面的侧视图），右边是按极角统计的直方图与解析 pdf 边缘分布的对照。',
            formula: [
                'h ~ D_{ω_o}(h) = G_1(ω_o) |ω_o·h| D(h) / |ω_o·n|              // VNDF（BX.sampleVNDF / pdfVNDF）',
                'P(R) = F(ω_o·h),   P(T) = 1 − F(ω_o·h)                         // TIR ⇒ F = 1',
                'ω_i = 2(ω_o·h)h − ω_o   (R)        ω_i = refract(ω_o, ±h, η_i/η_o)   (T)',
                'pdf_R = D_{ω_o}(h) · F / (4|ω_o·h|)',
                'pdf_T = D_{ω_o}(h) · (1 − F) · η_i^2 |ω_i·h| / (η_i (ω_i·h) + η_o (ω_o·h))^2',
                'weight = f |cos θ_i| / pdf = G_2 / G_1(ω_o)   × (η_o/η_i)^2 (仅 T，radiance 模式)',
            ],
            code: { title: 'HLSL · Sample_RoughDielectric', src: HLSL_SAMPLE },
            previews: [
                {
                    id: 'smpScatter', label: '采样方向散点（侧视）', sub: '橙 = 反射样本，蓝 = 折射样本；金线 = 固定方向 ωo', size: [270, 270],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c);
                        const cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 12;
                        ctx.fillStyle = 'rgba(70,130,190,0.10)'; ctx.fillRect(0, cy, w, h - cy);
                        ctx.strokeStyle = '#2a3238'; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
                        ctx.strokeStyle = '#4a5560'; ctx.beginPath(); ctx.moveTo(cx - R - 6, cy); ctx.lineTo(cx + R + 6, cy); ctx.stroke();
                        const K = S.smpK, X = S.smpX, Z = S.smpZ;
                        for (let k = 0; k < NS; k++) {
                            if (!K[k]) continue;
                            ctx.fillStyle = K[k] === 1 ? 'rgba(224,160,96,0.55)' : 'rgba(106,176,224,0.5)';
                            ctx.fillRect(cx + X[k] * R - 1, cy - Z[k] * R - 1, 2, 2);
                        }
                        const wo = S.wi;
                        arrow(ctx, cx, cy, cx + wo[0] * R, cy - wo[2] * R, '#e8c25a', 1.6);
                        const m = S.smp, tot = m.nR + m.nT + m.nBad;
                        ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
                        ctx.fillStyle = '#e0a060'; ctx.fillText(`R ${(100 * m.nR / tot).toFixed(1)}%`, 6, 4);
                        ctx.fillStyle = '#6ab0e0'; ctx.fillText(`T ${(100 * m.nT / tot).toFixed(1)}%`, 6, 17);
                        ctx.fillStyle = '#888'; ctx.fillText(`丢弃 ${(100 * m.nBad / tot).toFixed(1)}%`, 6, 30);
                    },
                },
                {
                    id: 'smpHist', label: '极角直方图 vs 解析 pdf', sub: '柱 = 样本密度（每度），线 = ∫ pdf(θ, φ) sinθ dφ', size: [360, 220],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, 180], xLabel: 'θ（从 +n 量起，°）· 90° = 界面',
                        series: [
                            { data: S.smp.hist, x: S.smp.hx, bar: true, color: 'rgba(106,176,224,0.55)', label: '样本' },
                            { data: S.smp.ana, x: S.smp.hx, color: '#e8c25a', width: 2, label: '解析 pdf' },
                        ],
                        marks: [{ x: 90, label: '界面', color: '#555' }],
                    }),
                },
                {
                    id: 'smpCheck', label: 'MC vs 求积', sub: '同一个 Eval 与 Sample 必须给出同一个积分（差值应在 ~2σ 内）', size: [340, 140],
                    render: (c, S) => {
                        const m = S.smp;
                        textPanel(c, [
                            ['           MC Σw/N ± 1σ       求积 ∫f|cosθi|dωi', '#888'],
                            [`E_R        ${m.eR.toFixed(4)} ± ${m.sR.toFixed(4)}   ${m.qR.toFixed(4)}`, '#e0a060'],
                            [`E_T (cam)  ${m.eT.toFixed(4)} ± ${m.sT.toFixed(4)}   ${m.qT.toFixed(4)}`, '#6ab0e0'],
                            [`E_R + E_T（importance 权重）= ${m.eImp.toFixed(4)}`, '#5ad07a'],
                            [`N = ${NS} · 丢弃 ${m.nBad} · 求积 = 绕 lobe 轴 160×96`, '#666'],
                        ]);
                    },
                },
            ],
            notes: '直方图与金线吻合说明 Sample 与 Pdf 一致；MC 与求积吻合说明 Sample 与 Eval 一致——两条都过，MIS 才是对的。切到<strong>介质 → 外部</strong>并把 θi 拉过 θc：蓝点急剧变少，橙点变多（TIR），但权重依然 ≈ G₂/G₁，不会出现萤火虫。“丢弃”是反射到宏观表面背面、或折回同侧的样本（粗糙 + 掠射时增多），它们就是单次散射模型的能量缺口。',
            warn: [
                'R/T 的选择概率必须用<strong>采样出的 h</strong> 上的 F(ωo·h)，而不是宏观法线上的 F(cos θo)；后者会让 pdf 与权重对不上（权重不再约成 G₂/G₁，出现萤火虫）。',
                'ωo 在介质内部时，VNDF 要用翻转后的 −ωo 采样，但 Fresnel 与 refract 要用原始 ωo 和对应一侧的 η；BX.sampleVNDF 只接受上半球输入。',
            ],
        },
        // ───────────────────────────── 09 ENERGY
        {
            id: 'energy', title: '白炉：E_R + E_T 随角度与粗糙度', tag: 'ENERGY',
            desc: '电介质不吸收，所以理想情况下固定入射方向的反射 + 透射能量应恰好为 1。单次散射的微表面模型把被 G₂ 遮挡的光直接丢掉（真实表面上它们会在微面元间多次弹跳后离开），所以 E_R + E_T &lt; 1，缺口随粗糙度和掠射角增大。这里用 VNDF 对每个 h 同时累加 R、T 两支（不掷 Fresnel 硬币，零方差地分开二者），权重 G₂/G₁。左两张是 6 档粗糙度（白线 = 当前 r）的外部 / 内部入射；第三张拆开当前 r 的 E_R 与 E_T；第四张是 θi 固定时随粗糙度的变化。',
            formula: [
                'E_R(ω) + E_T(ω) = ∫ f(ω, ω′) |cos θ′| dω′ ≤ 1        // 固定入射方向的 flux 守恒（importance 模式）',
                'E_R = E_h[ F · G_2/G_1 · 1_{反射有效} ],   E_T = E_h[ (1 − F) · G_2/G_1 · 1_{折射有效} ],   h ~ D_ω',
                'loss = 1 − E_R − E_T                                    // 缺失的多次散射，需要能量补偿',
            ],
            code: { title: 'HLSL · Furnace_RT（离线烘 LUT 也用它）', src: HLSL_ENERGY },
            previews: [
                ...['out', 'inn'].map((sd) => ({
                    id: 'en_' + sd, label: sd === 'out' ? '外部入射 E_R + E_T' : '内部入射 E_R + E_T', sub: '6 档 r = 0.1 / 0.25 / 0.4 / 0.6 / 0.8 / 1.0（青 → 红），白 = 当前 r', size: [330, 220],
                    render: (c, S) => {
                        const e = S.en;
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0.3, 1.2], xLabel: 'θ (°)',
                            series: [
                                ...S.enFam.map((f, k) => ({ data: f[sd].map((v) => v[0] + v[1]), x: e.ths, color: EN_COLORS[k], width: 1.2, label: k === 0 || k === EN_RS.length - 1 ? `r=${f.r}` : undefined })),
                                { data: e.cur[sd].map((v) => v[0] + v[1]), x: e.ths, color: '#ffffff', width: 2.2, label: `当前 ${S.rough.toFixed(2)}` },
                            ],
                            hmarks: [{ y: 1, color: '#444' }],
                            marks: sd === 'inn' ? [{ x: deg(S.thetaC), label: 'θc', color: '#e06c75' }] : [],
                        });
                    },
                })),
                {
                    id: 'enSplit', label: '当前 r：E_R 与 E_T', sub: '橙 = E_R，蓝 = E_T；实线 = 外部入射，虚线 = 内部入射', size: [330, 220],
                    render: (c, S) => {
                        const e = S.en;
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, 1.02], xLabel: 'θ (°)',
                            series: [
                                { data: e.cur.out.map((v) => v[0]), x: e.ths, color: '#e0a060', width: 2 },
                                { data: e.cur.out.map((v) => v[1]), x: e.ths, color: '#6ab0e0', width: 2 },
                                { data: e.cur.inn.map((v) => v[0]), x: e.ths, color: '#e0a060', dash: [4, 3] },
                                { data: e.cur.inn.map((v) => v[1]), x: e.ths, color: '#6ab0e0', dash: [4, 3] },
                            ],
                            marks: [{ x: deg(S.thetaC), label: 'θc', color: '#e06c75' }, { x: deg(S.thetaI), label: 'θi', color: '#e8c25a' }],
                        });
                    },
                },
                {
                    id: 'enRough', label: 'θi 处 E_R + E_T vs 粗糙度', sub: '白 = E_R + E_T，蓝 = E_T；实线 = 外部入射，虚线 = 内部入射', size: [330, 220],
                    render: (c, S) => {
                        const e = S.en;
                        PL.draw.plot(c, {
                            x: [0.1, 1], y: [0, 1.02], xLabel: 'roughness r',
                            series: [
                                { data: e.vsR.map((v) => v[0][0] + v[0][1]), x: e.rs, color: '#ffffff', width: 2 },
                                { data: e.vsR.map((v) => v[1][0] + v[1][1]), x: e.rs, color: '#ffffff', dash: [4, 3] },
                                { data: e.vsR.map((v) => v[0][1]), x: e.rs, color: '#6ab0e0' },
                                { data: e.vsR.map((v) => v[1][1]), x: e.rs, color: '#6ab0e0', dash: [4, 3] },
                            ],
                            marks: [{ x: S.rough, label: 'r', color: '#e8c25a' }],
                        });
                    },
                },
            ],
            notes: '外部入射时 r = 1 的掠射缺口可达 10–20%；内部入射更糟，因为 TIR 让光在微面元之间多弹几次的概率更高，单次散射模型把这些全部丢掉。粗糙玻璃因此看起来<strong>偏暗、偏灰</strong>。拖 <code>IOR</code>：η 越大，内部入射的缺口越大（θc 更小，更多微面元处于 TIR）。θ → 90° 时曲线又回升，这是高度相关 Smith 的性质而非数值误差（已用独立的 Eval 求积核对）：掠射时只有最高处的微面元可见，从那里出射几乎不再被遮挡，G₂/G₁ → 1。',
            warn: '不能直接把反射的 Kulla-Conty 补偿照搬到透射：缺失能量要按比例分给 R 和 T，而且内外两侧的 LUT 不同（需要额外的 η 维，通常是 (cos θ, r, η) 3D 表）。补偿后要重新跑一遍这张白炉图确认 ≈ 1。',
        },
        // ───────────────────────────── 10 PROD
        {
            id: 'prod', title: '教学版 → 生产版：实时粗糙折射、吸收、色散与常见坑', tag: 'PROD',
            desc: '左图是<strong>参考</strong>：用本页的 Walter BSDF（Sample_RoughDielectric，radiance 模式）对一个实心粗糙玻璃球做路径追踪，含内部多次反射、TIR 与 Beer-Lambert 吸收，每帧每像素 1 spp 渐进累加（LIVE，最多 384 spp）。中图是<strong>实时近似</strong>：只用宏观法线折射（前表面 + 背表面，背面深度来自 pre-pass），按粗糙度选环境 / SceneColor 的 mip 来模糊，Fresnel 只算宏观一次，没有蒙特卡洛。右图是两者共用的等距柱状环境图（相当于 SceneColor 的替身）。拖 <code>roughness</code> 对比：近似版的模糊中心位置、边缘 TIR 环、粗糙时的整体亮度都与参考有差距——这些就是生产里要调的经验参数。',
            formula: [
                'T_abs = exp(−σ_a · d),   σ_a = −ln(tint)                // Beer-Lambert：d = 背面深度 − 正面深度',
                'θ_blur ≈ k · α · (η − 1)(1 + 1/η),   lod = log_2(θ_blur / θ_texel)   // 粗糙折射 = 选 mip（k 为经验常数）',
                'L ≈ F · Env(R, α) + (1 − F)(1 − F_2) · T_abs · SceneColor(uv_refr, lod)   // 实时：单路径、无 TIR 多次弹射',
                'η(λ) = A + B / λ^2                                       // Cauchy 色散：R/G/B 各用一个 IOR 采样',
            ],
            code: [
                { title: 'HLSL · PS_RoughRefraction（前向透明 pass）', src: HLSL_PS },
                { title: 'HLSL · RayGen_PathTrace（参考）', src: HLSL_PT },
            ],
            previews: [
                {
                    id: 'ptRef', label: '参考：路径追踪（Walter BSDF）', sub: `${PT_RES}²，渐进 1 spp/帧（暂停时停止累加）`, live: true, size: [240, 240],
                    render: (c, S) => {
                        const N = PT_RES, acc = S.ptAcc, k = 1 / Math.max(1, S.ptSpp);
                        drawImageRGB(c, N, N, (x, y, out) => {
                            const i = (y * N + x) * 3, d = BX.display([acc[i] * k, acc[i + 1] * k, acc[i + 2] * k], EXPO);
                            out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
                        }, `${S.ptSpp} spp`);
                    },
                },
                {
                    id: 'rtImg', label: '实时近似', sub: '宏观法线 2 次折射 + mip 模糊 + 吸收', size: [240, 240],
                    render: (c, S) => {
                        const N = PT_RES, img = S.rtImg;
                        drawImageRGB(c, N, N, (x, y, out) => {
                            const i = (y * N + x) * 3, d = BX.display([img[i], img[i + 1], img[i + 2]], EXPO);
                            out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
                        }, 'analytic, 0 MC');
                    },
                },
                {
                    id: 'envTex', label: '环境图（SceneColor 替身）', sub: `${ENV_W}×${ENV_H} 等距柱状，box mip 链`, size: [300, 150], smooth: true,
                    render: (c, S) => {
                        const m = S.env[0];
                        PL.draw.pixels(c, m.w, m.h, (x, y, out) => {
                            const i = (y * m.w + x) * 3, d = BX.display([m.data[i], m.data[i + 1], m.data[i + 2]], EXPO);
                            out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
                        });
                    },
                },
            ],
            notes: [
                '<strong>生产里通常的改动</strong>：① 屏幕空间折射：SceneColor 在不透明 pass 后拷贝并建 mip（本页右图的角色），透明 pass 用折射偏移后的 uv 和由 α 推出的 lod 采样；② 厚度来自背面深度 pre-pass，或干脆用美术给的常数厚度；③ 薄壁（thin-walled，Burley 2015）：窗玻璃、泡泡不进入介质，出射方向 = 入射方向，只模糊不偏折，也没有 η² 和 TIR；④ 色散：R/G/B 三个 IOR 各采一次 SceneColor（Cauchy 式），代价 3 次采样；⑤ 能量补偿：按 (cos θ, r, η) 的 3D LUT 把 ENERGY 步骤的缺口按比例加回 R、T；⑥ 离线渲染器：保留本页完整的 Eval / Sample / Pdf，加上 MIS 与 Russian roulette。',
                '<strong>常见坑</strong>：HLSL <code>refract</code> 的方向与 η 约定（SNELL）；内部入射用 Schlick(cos θi) 漏掉 TIR（FRESNEL）；h_t 朝向与 χ⁺ 检查不一致（HALF）；pdf 用错雅可比的分子 η_o² / η_i²（JACOB）；radiance / importance 模式混用（ETA2）；用宏观 F 选 R/T（SAMPLE）；透明物体排序与双面渲染导致背面被画两次；法线贴图把着色法线翻到视线背面时，内外判断要用几何法线；IOR = 1 时 h_t 退化。',
            ],
            warn: '实时版不能用“同一张 SceneColor”同时代表物体背后和物体自身：屏幕空间折射只能看到屏幕上已有的不透明像素，折射偏移出屏幕或被前景遮挡时要回退到反射探针；多层透明（玻璃里的玻璃）需要按深度排序或 OIT，否则后画的层看不到先画的层。',
        },
    ],

    library: {
        title: '附录 · 辅助函数（HLSL）',
        intro: '正文代码调用、但步骤里没有展开的辅助部分：Step 09 白炉积分用的 Hammersley 点集，Step 10 实时折射 PS 的深度 / 屏幕坐标 / 预滤波环境查询，以及路径追踪参考的相机光线、抖动、场景求交（参考场景就是页面里那个单位玻璃球，所以直接解析求交）、等距柱状映射、正交基和随机数。',
        blocks: [
            { title: 'Hammersley（Step 09 · Furnace_RT）', code: HLSL_LIB_SEQ },
            { title: 'VSOut · LinearDepth · WorldToScreenUV · PrefilteredEnv（Step 10 · PS_RoughRefraction）', code: HLSL_LIB_RT },
            { title: 'CameraRay · Jitter · TraceScene · DirToEquirect · Basis · Rand3（Step 10 · RayGen_PathTrace）', code: HLSL_LIB_PT },
        ],
    },

    graph: {
        title: 'Rough Dielectric · 实时前向折射 + 路径追踪参考',
        intro: '<strong>RDG 风格</strong>：蓝色 INLINE 节点是本页的 BTDF 各项，它们不是独立 pass，而是被内联进两个真实 pass——前向透明 pass <code>PS_RoughRefraction</code>（实时）与 <code>RayGen_PathTrace</code>（参考，逐帧累加）。<code>SceneColorTex</code> 是不透明 pass 的结果、由引擎提供（external）。常量与共享函数在 Common 面板。',
        types: {
            cb: { color: '#e8c25a', label: 'cbuffer · 材质参数' },
            fn: { color: '#c678dd', label: 'inline · HLSL 函数（被内联）' },
            tex: { color: '#4a8aff', label: 'texture · 场景颜色 / 累加缓冲' },
            out: { color: '#5ae87a', label: 'output · 合成结果' },
        },
        common: {
            open: false,
            title: 'Common.hlsl · 系列共享 + 透射模块（SideEta / RefractDir / F_Dielectric / G2 / VNDF）',
            blocks: [
                { label: '━ 系列共享 ━', code: BX.HLSL_COMMON },
                { label: '━ 03 透射模块 ━', code: HLSL_TR_COMMON },
                { label: '━ cbuffer ━', code: 'cbuffer MaterialCB : register(b0) { float Roughness; float Ior; float3 SigmaA; float BlurK; };' },
            ],
        },
        nodes: [
            {
                id: 'matcb', title: 'MaterialCB', tag: 'CB', category: 'user', desc: 'Roughness · Ior · SigmaA · BlurK',
                outputs: [{ name: 'MaterialCB', format: 'cbuffer b0', type: 'cb' }],
                detail: { desc: '材质参数：感知粗糙度 r（shader 里 α = r²）、介质 IOR（外部按空气 1.0）、吸收系数 σ_a = −ln(tint)、实时模糊常数 k。', code: { src: 'cbuffer MaterialCB : register(b0) { float Roughness; float Ior; float3 SigmaA; float BlurK; };' }, preview: null },
            },
            {
                id: 'fresnel', title: 'F_Dielectric', tag: 'INLINE', category: 'function', step: 'fresnel', desc: '精确 Fresnel · TIR → 1',
                dispatch: { kind: 'INLINE', name: 'F_Dielectric(cosI, eta)', note: '内联进 BTDF、采样器和实时 PS' },
                outputs: [{ name: 'F_Dielectric', format: 'float(float,float)', type: 'fn' }],
                detail: { preview: 'fresIn' },
            },
            {
                id: 'halfT', title: 'HalfVectorT · JacobianT', tag: 'INLINE', category: 'function', step: 'half', desc: 'h_t = −(η_i ωi + η_o ωo) → +n',
                dispatch: { kind: 'INLINE', name: 'HalfVectorT / JacobianT', note: '纯几何，无纹理读取' },
                outputs: [{ name: 'HalfVectorT', format: 'float3(float3,float3,…)', type: 'fn' }],
                detail: { code: { title: 'HalfVectorT + JacobianT', src: HLSL_HALF + '\n' + HLSL_JACOB }, preview: 'halfMap' },
            },
            {
                id: 'btdf', title: 'Eval_RoughDielectric', tag: 'INLINE', category: 'function', step: 'btdf', desc: 'f_r + (1−F)·D·G₂·J',
                dispatch: { kind: 'INLINE', name: 'Eval_RoughDielectric(wi, wo, a, ior)', note: 'MIS 的光源采样分支调用' },
                inputs: [{ name: 'F_Dielectric', format: 'fn', type: 'fn' }, { name: 'HalfVectorT', format: 'fn', type: 'fn' }],
                outputs: [{ name: 'Eval_RoughDielectric', format: 'float(float3,float3,…)', type: 'fn' }],
                detail: { commonUsed: 'D_GGX  Lambda_GGX  G2_Smith  F_Dielectric  HalfVectorT   // Common', preview: 'termf' },
            },
            {
                id: 'sampler', title: 'Sample_RoughDielectric', tag: 'INLINE', category: 'function', step: 'sample', desc: 'VNDF h → F 选 R/T → pdf',
                dispatch: { kind: 'INLINE', name: 'Sample_RoughDielectric(wo, a, ior, u)', note: 'BSDF 采样分支；权重 = G₂/G₁ (×η² 仅 T)' },
                inputs: [{ name: 'F_Dielectric', format: 'fn', type: 'fn' }, { name: 'HalfVectorT', format: 'fn', type: 'fn' }],
                outputs: [{ name: 'Sample_RoughDielectric', format: 'float3(float3,…, out w, out pdf)', type: 'fn' }],
                detail: { commonUsed: 'SampleVNDF_GGX  G1_Smith  G2_Smith  RefractDir  F_Dielectric   // Common', preview: 'smpHist' },
            },
            {
                id: 'mips', title: 'CS_SceneColorMips', tag: 'CS', category: 'function', lifetime: 'frame', step: 'prod', desc: '拷贝 SceneColor + 2×2 box mip 链',
                dispatch: { kind: 'CS', name: 'CS_DownsampleSceneColor', threads: [8, 8, 1], groups: 'ceil(mipW/8) × ceil(mipH/8) × 1（每级一次）', idRange: 'ID.xy ∈ [0, mipSize)', note: '在不透明 pass 之后、透明 pass 之前' },
                inputs: [{ name: 'SceneColorTex', format: 'RGBA16F', type: 'tex', external: true }],
                outputs: [{ name: 'SceneColorMips', format: 'RGBA16F · mips', type: 'tex' }],
                detail: { desc: '把不透明 pass 的 SceneColor 拷出来并建 mip：粗糙折射按 α 选 mip 级别。本页用等距柱状环境图扮演它（右图）。', preview: 'envTex' },
            },
            {
                id: 'ps', title: 'PS_RoughRefraction', tag: 'PS', category: 'output', lifetime: 'frame', step: 'prod', desc: 'F·Env + (1−F)·T_abs·SceneColor(uv, lod)',
                dispatch: { kind: 'PS', name: 'PS_RoughRefraction', target: 'SceneColor（translucent，after opaque）', idRange: 'SV_Position.xy ∈ 透明物体覆盖的像素', note: '背面深度来自 pre-pass；按深度从后往前' },
                inputs: [
                    { name: 'SceneColorMips', format: 'RGBA16F · mips', type: 'tex' },
                    { name: 'MaterialCB', format: 'cbuffer', type: 'cb' },
                    { name: 'F_Dielectric', format: 'fn', type: 'fn' },
                    { name: 'BackFaceDepth', format: 'D32F', type: 'tex', external: true },
                ],
                outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'out' }],
                detail: { code: { title: 'PS_RoughRefraction', src: HLSL_PS }, preview: 'rtImg' },
            },
            {
                id: 'rt', title: 'RayGen_PathTrace', tag: 'RT', category: 'output', lifetime: 'history', step: 'prod', desc: '每帧 1 spp，累加参考',
                dispatch: { kind: 'RT', name: 'RayGen_PathTrace', count: () => `${PT_RES}×${PT_RES} 条路径 / 帧`, idRange: () => `DispatchRaysIndex().xy ∈ [0, ${PT_RES})²`, note: '最多 10 次界面事件；参数变化时清零' },
                inputs: [
                    { name: 'Sample_RoughDielectric', format: 'fn', type: 'fn' },
                    { name: 'Eval_RoughDielectric', format: 'fn', type: 'fn' },
                    { name: 'MaterialCB', format: 'cbuffer', type: 'cb' },
                    { name: 'EnvMap', format: 'RGBA16F equirect', type: 'tex', external: true },
                    { name: 'PathTraceAccum', format: 'RGBA32F', type: 'tex', history: true },
                ],
                outputs: [{ name: 'PathTraceAccum', format: 'RGBA32F', type: 'tex' }],
                detail: { code: { title: 'RayGen_PathTrace', src: HLSL_PT }, preview: 'ptRef' },
            },
        ],
    },

    references: [
        { title: 'Microfacet Models for Refraction through Rough Surfaces', author: 'Bruce Walter, Stephen R. Marschner, Hongsong Li, Kenneth E. Torrance', venue: 'EGSR (Rendering Techniques)', year: 2007, note: '广义半向量 h_t（Eq.16）、雅可比（Eq.17）、BTDF（Eq.21）与 GGX 的提出' },
        { title: 'Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs', author: 'Eric Heitz', venue: 'JCGT 3(2)', year: 2014, note: 'Smith Λ、高度相关 G2，以及透射情形下的遮蔽-阴影' },
        { title: 'Sampling the GGX Distribution of Visible Normals', author: 'Eric Heitz', venue: 'JCGT 7(4)', year: 2018, note: 'VNDF 采样（BX.sampleVNDF）' },
        { title: 'Robust Monte Carlo Methods for Light Transport Simulation', author: 'Eric Veach', venue: 'PhD thesis, Stanford University', year: 1997, note: '第 5 章：折射导致的非对称 BSDF、广义互易、adjoint BSDF' },
        { title: 'Physically Based Rendering: From Theory to Implementation (4th ed.)', author: 'Matt Pharr, Wenzel Jakob, Greg Humphreys', venue: 'MIT Press', year: 2023, url: 'https://pbr-book.org/', note: 'DielectricBxDF：粗糙电介质的 f / Sample_f / PDF 与 TransportMode' },
        { title: 'Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering', author: 'Brent Burley', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2015, note: 'specTrans、薄壁（thin-walled）与实心的区别' },
        { title: 'Revisiting Physically Based Shading at Imageworks', author: 'Christopher Kulla, Alejandro Conty', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2017, note: '微表面多次散射的能量补偿（含电介质）' },
        { title: 'Principles of Optics', author: 'Max Born, Emil Wolf', venue: 'Cambridge University Press', year: 1999, note: 'Fresnel 方程、Brewster 角、全内反射' },
    ],
});
