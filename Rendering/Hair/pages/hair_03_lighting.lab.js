// ════════════════════════════════════════════════════════════════════════════
// 毛发渲染 03 · 自阴影与多次散射：Deep Opacity → Dual Scattering
// Build (inside pages/):
//   node <skill>/scripts/build_doc.mjs hair_common.js hair_03_lighting.lab.js -o hair_03_lighting.html
// Conventions: see hair_common.js header (y up, cm, fiber frame x = tangent, wi → light, wo → viewer, L_o = S·E).
// Page-local conventions:
//   • Light-space depth z = dot(p − center, fwd) of the light ortho camera (grows away from the light).
//   • Every rendered strand stands for κ real fibers: hair count n(z) = Σ_{fragments z_f < z} c_f (texel coverage),
//     transmittance T(z) = exp(−κ·n(z)) (Beer–Lambert over sub-pixel fibers). All methods approximate this T.
// ════════════════════════════════════════════════════════════════════════════

const L3 = (() => {
    const { dot, norm, clamp } = HAIR;
    const VIEW_RES = 128;
    /** wi (toward the light) from azimuth / elevation; az = 0 → light on the viewer side (+z), 180° → backlight */
    const lightDir = (az, el) => [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
    /** ortho camera looking along −w (w = direction toward the camera / light), fitted to an AABB */
    const fitOrtho = (w, box, pad = 0.04, strands = null) => {
        const yaw = Math.atan2(w[0], w[2]), pitch = -Math.asin(clamp(w[1], -1, 1));
        const c0 = HAIR.ortho({ yaw, pitch, center: [0, 0, 0], size: 1 });
        const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        const add = (p) => { const q = [dot(p, c0.right), dot(p, c0.up), dot(p, c0.fwd)]; for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], q[a]); hi[a] = Math.max(hi[a], q[a]); } };
        if (strands) { for (const st of strands) for (let i = 0; i < st.p.length; i += 3) add([st.p[i], st.p[i + 1], st.p[i + 2]]); }   // tight fit to the strand vertices
        else for (let k = 0; k < 8; k++) add([box[k & 1 ? 3 : 0], box[k & 2 ? 4 : 1], box[k & 4 ? 5 : 2]]);
        const m = [0, 1, 2].map((a) => (lo[a] + hi[a]) / 2), B = [c0.right, c0.up, c0.fwd];
        const center = [0, 1, 2].map((i) => B[0][i] * m[0] + B[1][i] * m[1] + B[2][i] * m[2]);
        const size = Math.max(hi[0] - lo[0], hi[1] - lo[1]) * (1 + pad);
        return Object.assign(HAIR.ortho({ yaw, pitch, center, size }), { zRange: [lo[2] - m[2], hi[2] - m[2]] });
    };
    const aabb = (strands) => {
        const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
        for (const s of strands) for (let i = 0; i < s.p.length; i += 3) for (let a = 0; a < 3; a++) {
            b[a] = Math.min(b[a], s.p[i + a]); b[a + 3] = Math.max(b[a + 3], s.p[i + a]);
        }
        return b;
    };
    /** inverse of HAIR.project: pixel (x, y) + depth → world position */
    const unproject = (cam, x, y, z, W, H) => {
        const k = H / cam.size, r = (x + 0.5 - W / 2) / k, u = (H / 2 - y - 0.5) / k;
        return [0, 1, 2].map((i) => cam.center[i] + cam.right[i] * r + cam.up[i] * u + cam.fwd[i] * z);
    };

    /** Light-view fragment lists (= an uncompressed deep shadow map). Same projection, analytic coverage and
     *  per-(strand, pixel) max de-duplication as HAIR.raster, so Σ_texel c_f equals HAIR.raster(...).cov exactly —
     *  but every fragment keeps its own depth, which HAIR.raster does not expose. Lists are sorted by z (CSR). */
    const fragments = (strands, cam, W, H) => {
        const n = W * H, k = H / cam.size;
        let cap = 1 << 16, Z = new Float32Array(cap), C = new Float32Array(cap), P = new Int32Array(cap), I = new Int32Array(cap), m = 0;
        const grow = () => {
            cap *= 2;
            const z2 = new Float32Array(cap), c2 = new Float32Array(cap), p2 = new Int32Array(cap), i2 = new Int32Array(cap);
            z2.set(Z); c2.set(C); p2.set(P); i2.set(I); Z = z2; C = c2; P = p2; I = i2;
        };
        const last = new Int32Array(n).fill(-1), cur = new Int32Array(n);
        for (const s of strands) {
            const Q = s.p, segs = Q.length / 3 - 1;
            let a = HAIR.project(cam, [Q[0], Q[1], Q[2]], W, H);
            for (let j = 0; j < segs; j++) {
                const b = HAIR.project(cam, [Q[3 * j + 3], Q[3 * j + 4], Q[3 * j + 5]], W, H);
                const halfW = (s.w0 + (s.w1 - s.w0) * (j + 0.5) / segs) * k / 2;
                const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-9, r = halfW + 1;
                const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0]) - r)), x1 = Math.min(W - 1, Math.ceil(Math.max(a[0], b[0]) + r));
                const y0 = Math.max(0, Math.floor(Math.min(a[1], b[1]) - r)), y1 = Math.min(H - 1, Math.ceil(Math.max(a[1], b[1]) + r));
                for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
                    const u = clamp(((x - a[0]) * dx + (y - a[1]) * dy) / L2, 0, 1);
                    const d = Math.hypot(a[0] + u * dx - x, a[1] + u * dy - y);
                    const c = halfW >= 0.5 ? clamp(halfW + 0.5 - d, 0, 1) : 2 * halfW * Math.max(0, 1 - d);
                    if (c <= 0) continue;
                    const i = y * W + x, z = a[2] + u * (b[2] - a[2]);
                    if (last[i] !== s.id) { last[i] = s.id; if (m === cap) grow(); Z[m] = z; C[m] = c; P[m] = i; I[m] = s.id; cur[i] = m++; }
                    else if (c > C[cur[i]]) { C[cur[i]] = c; Z[cur[i]] = z; }
                }
                a = b;
            }
        }
        const off = new Int32Array(n + 1);
        for (let f = 0; f < m; f++) off[P[f] + 1]++;
        for (let i = 0; i < n; i++) off[i + 1] += off[i];
        const z = new Float32Array(m), c = new Float32Array(m), sid = new Int32Array(m), fill = off.slice(0, n);
        for (let f = 0; f < m; f++) { const d = fill[P[f]]++; z[d] = Z[f]; c[d] = C[f]; sid[d] = I[f]; }
        for (let i = 0; i < n; i++) for (let a = off[i] + 1; a < off[i + 1]; a++) {   // insertion sort by depth
            const zv = z[a], cv = c[a], iv = sid[a]; let b = a - 1;
            while (b >= off[i] && z[b] > zv) { z[b + 1] = z[b]; c[b + 1] = c[b]; sid[b + 1] = sid[b]; b--; }
            z[b + 1] = zv; c[b + 1] = cv; sid[b + 1] = iv;
        }
        return { W, H, off, z, c, sid, count: m };
    };
    /** exact hair count in front of depth zq in texel i (optionally ignoring one strand — the shaded strand itself) */
    const countAt = (F, i, zq, ex = -1) => { let s = 0; for (let f = F.off[i]; f < F.off[i + 1] && F.z[f] < zq; f++) if (F.sid[f] !== ex) s += F.c[f]; return s; };

    /** Lokovic–Veach style compression: greedy segments, each extended while some slope keeps every original vertex
     *  within ±eps (slope-range test). Input: polyline zs/vs (non-decreasing z). Returns {z, v} vertex arrays. */
    const compress = (zs, vs, eps) => {
        const oz = [zs[0]], ov = [vs[0]];
        let z0 = zs[0], v0 = vs[0], lo = -Infinity, hi = Infinity;
        for (let j = 1; j < zs.length; j++) {
            const dz = Math.max(1e-6, zs[j] - z0);
            const sl = (vs[j] - eps - v0) / dz, sh = (vs[j] + eps - v0) / dz;
            if (Math.max(lo, sl) > Math.min(hi, sh)) {          // window closed → emit a vertex at the previous sample
                const s = (lo + hi) / 2, zp = zs[j - 1], vp = v0 + s * (zp - z0);
                oz.push(zp); ov.push(vp); z0 = zp; v0 = vp;
                const dz2 = Math.max(1e-6, zs[j] - z0);
                lo = (vs[j] - eps - v0) / dz2; hi = (vs[j] + eps - v0) / dz2;
            } else { lo = Math.max(lo, sl); hi = Math.min(hi, sh); }
        }
        const zl = zs[zs.length - 1];
        if (zl > z0) { const s = isFinite(lo + hi) ? (lo + hi) / 2 : 0; oz.push(zl); ov.push(v0 + s * (zl - z0)); }
        return { z: oz, v: ov };
    };
    const polyAt = (pz, pv, z) => {
        if (z <= pz[0]) return pv[0];
        for (let k = 1; k < pz.length; k++) if (z <= pz[k]) return pv[k - 1] + (pv[k] - pv[k - 1]) * (z - pz[k - 1]) / Math.max(1e-9, pz[k] - pz[k - 1]);
        return pv[pv.length - 1];
    };
    /** bilinear 2×2 PCF-style filter of a per-texel transmittance function fn(i) at light-space position (lx, ly) */
    const filter2 = (W, H, lx, ly, fn) => {
        const x0 = Math.floor(lx), y0 = Math.floor(ly), fx = lx - x0, fy = ly - y0;
        let s = 0, ws = 0;
        for (let k = 0; k < 4; k++) {
            const x = x0 + (k & 1), y = y0 + (k >> 1);
            const w = (k & 1 ? fx : 1 - fx) * (k >> 1 ? fy : 1 - fy);
            if (x < 0 || y < 0 || x >= W || y >= H) { s += w; ws += w; continue; }   // outside the map: unshadowed
            s += w * fn(y * W + x); ws += w;
        }
        return s / Math.max(1e-9, ws);
    };
    /** DOM layer ends d_1..d_K (relative to z0): constant Δz·k, linear sizes Δz·k(k+1)/2 (Yuksel & Keyser's choice), powers of two Δz·(2^k − 1) */
    const layerEnds = (K, dz, dist) => Array.from({ length: K }, (_, k) => dz * (dist === 'linear' ? (k + 1) * (k + 2) / 2 : dist === 'pow2' ? 2 ** (k + 1) - 1 : k + 1));
    /** layer index containing offset t = z − z0 (fragments beyond the last end → last layer) */
    const layerOf = (ends, t) => { for (let k = 0; k < ends.length; k++) if (t < ends[k]) return k; return ends.length - 1; };
    /** interpolated cumulative Ω at offset t from a per-texel layer array O (offset o) */
    const domOmega = (O, o, ends, t) => {
        if (t <= 0) return 0;
        const K = ends.length; let k = 0; while (k < K && t >= ends[k]) k++;
        if (k >= K) return O[o + K - 1];
        const a = k === 0 ? 0 : ends[k - 1], lo = k === 0 ? 0 : O[o + k - 1];
        return lo + (O[o + k] - lo) * (t - a) / (ends[k] - a);
    };
    return { VIEW_RES, lightDir, fitOrtho, aabb, unproject, fragments, countAt, compress, polyAt, filter2, layerEnds, layerOf, domOmega };
})();

// ── dual scattering (Zinke et al. 2008) tables in this series' convention: energy fraction = ∫ S dωi ──
const DS = (() => {
    const PI = Math.PI, NT = 24, NP = 36, NH = 48;
    const wrapPi = (x) => x - 2 * PI * Math.floor((x + PI) / (2 * PI));
    /** mass of the wrapped trimmed logistic N_p (centre Phi, scale s) over φ ∈ [a, b] (b − a ≤ 2π) */
    const massArc = (Phi, a, b, s) => {
        const F = (x) => HAIR.logisticCDF(x, s), Z = F(PI) - F(-PI), lo = wrapPi(a - Phi), hi = lo + (b - a);
        return hi <= PI ? (F(hi) - F(lo)) / Z : (F(PI) - F(lo) + F(hi - 2 * PI) - F(-PI)) / Z;
    };
    const gauss = (x, v) => Math.exp(-x * x / (2 * v)) / Math.sqrt(2 * PI * v);
    /** longitudinal lobe p with extra variance (pbrt-v3 cuticle rotation) */
    const lobeM = (b, p, sinI, cosI, sinO, cosO, extraV) => {
        let sOp = sinO, cOp = cosO;
        if (p === 0) { sOp = sinO * b.cos2k[1] - cosO * b.sin2k[1]; cOp = cosO * b.cos2k[1] + sinO * b.sin2k[1]; }
        else if (p === 1) { sOp = sinO * b.cos2k[0] + cosO * b.sin2k[0]; cOp = cosO * b.cos2k[0] - sinO * b.sin2k[0]; }
        else if (p === 2) { sOp = sinO * b.cos2k[2] + cosO * b.sin2k[2]; cOp = cosO * b.cos2k[2] - sinO * b.sin2k[2]; }
        return HAIR.Mp(cosI, Math.abs(cOp), sinI, sOp, b.v[p] + extraV);
    };
    /** Bake every table of Table 1 (+ the per-lobe energies they come from) for one fibre (HAIR.bsdf object). */
    const bake = (b, withNG = true) => {
        const alpha = Math.asin(b.sin2k[0]), shift = [2 * alpha, -alpha, -4 * alpha, 0];   // signed tilt exactly as HAIR.bsdf rotates θo → lobe centres in (θi + θo)
        const t = { NT, NP, af: [], ab: [], alf: [], alb: [], bf2: [], bb2: [], Ab: [], Db: [], sb2: [], Ef: [], Eb: [], NG: new Float32Array(4 * NT * NP * 3) };
        for (let j = 0; j < NT; j++) {
            const th = (j + 0.5) / NT * PI / 2, sinO = Math.sin(th), cosO = Math.cos(th);
            const etap = Math.sqrt(b.eta * b.eta - sinO * sinO) / cosO, sinTt = sinO / b.eta, cosTt = Math.sqrt(1 - sinTt * sinTt);
            const Ef = [0, 1, 2, 3].map(() => [0, 0, 0]), Eb = [0, 1, 2, 3].map(() => [0, 0, 0]);
            for (let k = 0; k < NH; k++) {
                const h = -1 + (2 * k + 1) / NH, gO = Math.asin(h), gT = Math.asin(h / etap), cosGt = Math.cos(gT);
                const T = b.sigmaA.map((sa) => Math.exp(-sa * 2 * cosGt / cosTt)), A = HAIR.Ap(cosO, b.eta, h, T);
                for (let p = 0; p < 4; p++) {
                    const Phi = HAIR.Phi(p, gO, gT), fw = p < 3 ? massArc(Phi, PI / 2, 3 * PI / 2, b.s) : 0.5;
                    for (let c = 0; c < 3; c++) { Ef[p][c] += A[p][c] * fw / NH; Eb[p][c] += A[p][c] * (1 - fw) / NH; }
                    if (withNG) for (let q = 0; q < NP; q++) {    // N^G (Eq. 25): N_p averaged over a π-wide window of incident azimuths, A_p folded in
                        const phi = -PI + (q + 0.5) / NP * 2 * PI, ng = p < 3 ? massArc(Phi, phi - PI / 2, phi + PI / 2, b.s) / PI : 1 / (2 * PI);
                        for (let c = 0; c < 3; c++) t.NG[((p * NT + j) * NP + q) * 3 + c] += A[p][c] * ng / NH;
                    }
                }
            }
            const sum = (E, f) => [0, 1, 2].map((c) => E.reduce((acc, e, p) => acc + e[c] * f(p), 0));
            const af = sum(Ef, () => 1), ab = sum(Eb, () => 1);
            const alf = sum(Ef, (p) => shift[p]).map((x, c) => x / af[c]), alb = sum(Eb, (p) => shift[p]).map((x, c) => x / ab[c]);
            const bf2 = sum(Ef, (p) => b.v[p]).map((x, c) => x / af[c]), bb2 = sum(Eb, (p) => b.v[p]).map((x, c) => x / ab[c]);
            const Ab = [], Db = [], sb2 = [];
            for (let c = 0; c < 3; c++) {
                const f2 = af[c] * af[c], a = ab[c], a3 = a * a * a, bf = Math.sqrt(bf2[c]), bb = Math.sqrt(bb2[c]);
                Ab[c] = a * f2 / (1 - f2) + a3 * f2 / Math.pow(1 - f2, 3);                                        // Eq. 11 + 13
                Db[c] = alb[c] * (1 - 2 * a * a / ((1 - f2) * (1 - f2))) + alf[c] * (2 * (1 - f2) * (1 - f2) + 4 * f2 * a * a) / Math.pow(1 - f2, 3);   // Eq. 16
                const sb = (1 + 0.7 * f2) * (a * Math.sqrt(2 * bf * bf + bb * bb) + a3 * Math.sqrt(2 * bf * bf + 3 * bb * bb)) / (a + a3 * (2 * bf + 3 * bb));   // Eq. 17
                sb2[c] = sb * sb;
            }
            Object.entries({ af, ab, alf, alb, bf2, bb2, Ab, Db, sb2, Ef, Eb }).forEach(([key, val]) => t[key].push(val));
        }
        return t;
    };
    const row = (t, th) => { const j = Math.min(NT - 1, Math.max(0, Math.round(Math.abs(th) / (PI / 2) * NT - 0.5))); return j; };
    /** bilinear N^G lookup (A_p folded in), RGB → out */
    const NGat = (t, p, th, phi, c) => {
        const u = Math.min(NT - 1, Math.max(0, Math.abs(th) / (PI / 2) * NT - 0.5)), j0 = Math.floor(u), j1 = Math.min(NT - 1, j0 + 1), fu = u - j0;
        const w = (wrapPi(phi) + PI) / (2 * PI) * NP - 0.5, q0 = Math.floor(w), fw = w - q0, qa = (q0 + NP) % NP, qb = (q0 + 1) % NP;
        const G = (j, q) => t.NG[((p * NT + j) * NP + q) * 3 + c];
        return (G(j0, qa) * (1 - fw) + G(j0, qb) * fw) * (1 - fu) + (G(j1, qa) * (1 - fw) + G(j1, qb) * fw) * fu;
    };
    return { PI, NT, wrapPi, massArc, gauss, lobeM, bake, row, NGat };
})();

// texel/plot helpers shared by previews
const L3V = {
    mask: (S, arr) => { const n = L3.VIEW_RES ** 2, o = PL.buf(S, '_mask', n); for (let i = 0; i < n; i++) o[i] = S.view.id[i] >= 0 ? arr[i] : NaN; return o; },
    tProbe: (name, key) => (x, y, S) => {
        const i = y * L3.VIEW_RES + x;
        if (S.view.id[i] < 0) return '背景（无发丝）';
        return `${name} = ${S[key][i].toFixed(3)}\n精确 T = ${S.Tref[i].toFixed(3)}\n光源空间 Δz = ${(S.vz[i]).toFixed(2)} cm`;
    },
    lightProbe: (x, y, S) => {
        const i = y * S.smRes + x, F = S.frags, nf = F.off[i + 1] - F.off[i];
        if (!nf) return '空 texel';
        return `片元数 = ${nf}\nhair count n = ${S.lr.cov[i].toFixed(2)}\nz₀ = ${S.lr.depth[i].toFixed(2)} cm\nT(背面) = ${Math.exp(-S.kappa * S.lr.cov[i]).toFixed(3)}`;
    },
    /** linear radiance (per unit irradiance) of a visible sample for one shading mode */
    radiance: (S, i, mode, out) => {
        const inv = 1 / Math.max(1, S.accN);
        for (let c = 0; c < 3; c++) {
            const s = S.acc[3 * i + c] * inv, dF = S.Tdom[i], tf = S.Tf[3 * i + c];
            const ms = (tf - dF) * S.df * (S.Ss[3 * i + c] + S.Sbs[3 * i + c]) + dF * S.Sbd[3 * i + c];
            out[c] = mode === 'none' ? s : mode === 'dom' ? s * dF : mode === 'ref' ? s * S.Tref[i] : mode === 'bin' ? s * S.Tbin[i]
                : mode === 'vox' ? s * S.Tvox[i] : mode === 'ms' ? ms : mode === 'dual' ? s * dF + ms
                : mode === 'local' ? dF * S.Sbd[3 * i + c] + (tf - dF) * S.df * S.Sbs[3 * i + c] : mode === 'gms' ? (tf - dF) * S.df * S.Ss[3 * i + c] : 0;
        }
        return out;
    },
    shade: (mode, which) => (cv, S) => {
        const V = L3.VIEW_RES, bg = [0.010, 0.011, 0.014], L = [0, 0, 0], ex = (which === 'dual' ? S.expoDual : S.expo) || 1;
        PL.draw.pixels(cv, V, V, (x, y, out) => {
            const i = y * V + x, a = S.view.id[i] >= 0 ? 1 - Math.exp(-S.kappa * S.view.cov[i]) : 0;
            if (a > 0) L3V.radiance(S, i, mode, L); else L[0] = L[1] = L[2] = 0;
            const d = HAIR.display([0, 1, 2].map((c) => a * L[c] * ex + (1 - a) * bg[c]), 1);
            out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
        });
    },
    shadeProbe: (mode) => (x, y, S) => {
        const i = y * L3.VIEW_RES + x;
        if (S.view.id[i] < 0) return '背景';
        const L = L3V.radiance(S, i, mode, [0, 0, 0]);
        return `L/E = (${L.map((v) => v.toFixed(4)).join(', ')})\nT_DOM = ${S.Tdom[i].toFixed(3)} · κn = ${S.mCount[i].toFixed(2)}\n帧数 = ${S.accN}`;
    },
};
// 3D 视图（Step 04）：光源视角的扫描列数、每格高度（texel）、最多画多少根发丝
const DOM3D = { cols: 5, cell: 4, maxStrands: 80 };

PipelineLab.mount({
    paramSteps: { count: ['why', 'dsm', 'osm', 'dom', 'lookup', 'voxel', 'global', 'local', 'compare', 'engine', 'prod'], lightAz: ['why', 'dsm', 'osm', 'dom', 'lookup', 'voxel', 'global', 'local', 'compare', 'engine', 'prod'], lightEl: ['why', 'dsm', 'osm', 'dom', 'lookup', 'voxel', 'global', 'local', 'compare', 'engine', 'prod'], kappa: ['why', 'dsm', 'osm', 'dom', 'lookup', 'voxel', 'global', 'local', 'compare', 'engine', 'prod'], smRes: ['why', 'dsm', 'osm', 'dom', 'lookup', 'voxel', 'global', 'local', 'compare', 'engine', 'prod'], K: ['dom', 'lookup', 'global', 'local', 'compare', 'engine', 'prod'], dz: ['dom', 'lookup', 'global', 'local', 'compare', 'engine', 'prod'], dist: ['dom', 'lookup', 'global', 'local', 'compare', 'engine', 'prod'], osm: ['osm', 'dom'], eps: ['dsm', 'lookup', 'prod'], bias: ['why', 'osm', 'dom', 'lookup', 'global', 'local', 'compare', 'engine', 'prod'], ce: ['lookup', 'voxel', 'global', 'local', 'compare'], cp: ['lookup', 'voxel', 'global', 'local', 'compare', 'engine'], df: ['global', 'local', 'compare'], vox: ['voxel'], voxBias: ['voxel'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '毛发渲染 03 · 自阴影与多次散射：Deep Opacity → Dual Scattering',
    lang: 'zh-CN', codeLang: 'hlsl', seed: 3,
    params: [
        { id: 'count', label: '发丝数', options: [750, 1500, 3000], value: 1500, affects: 'resident', title: '渲染发丝数量（groom 资产）',
            desc: 'groom 里渲染发丝的根数（16 条 guide 插值）。同一 κ 下发丝越多，光路上的 hair count 越大、阴影越深、多次散射占比越高；改它会重建 groom、光源视角片元列表和体素，几乎所有步骤都随之变化。' },
        { id: 'lightAz', label: '光源方位', min: -180, max: 180, step: 5, value: 55, unit: '°', map: (d) => d * Math.PI / 180, title: '0° = 与相机同侧（正面光），180° = 逆光',
            desc: '方向光的方位角：0° = 与相机同侧（正面光），±90° = 侧光（光要横穿整片 8 cm 宽的发帘），180° = 逆光。拖到 90° 看 Step 01：普通 shadow map 几乎全黑，精确透射率仍是平滑渐变；Step 04 的 3D 视图里 z₀ 面与各层跟着光转。' },
        { id: 'lightEl', label: '光源仰角', min: -30, max: 80, step: 5, value: 30, unit: '°', map: (d) => d * Math.PI / 180,
            desc: '方向光的仰角，正值 = 从上方照射。仰角越大，光线与（大致竖直的）发丝夹角越小、斜着穿过发帘：阴影图、透射率和 ā_f(θ) 查表的角度一起变（Step 01–09）。' },
        { id: 'kappa', label: '密度 κ', min: 0.5, max: 12, step: 0.1, value: 6, digits: 1, title: '每根渲染发丝代表的真实发丝数 = 消光倍率，T = exp(−κ·n)',
            desc: '每根渲染发丝代表的真实发丝数 = 消光倍率，T = exp(−κ·n)（≈ UE 的 DensityScale）。调小 → 发丝稀疏，普通 shadow map 与精确透射率的差距更明显（Step 01）；调大 → 阴影更深、m = κ·n 变大，多次散射的比重上升（Step 07–09）。' },
        { id: 'smRes', label: '阴影图', options: [64, 96, 128, 160], value: 128, title: '光源视角分辨率（texel）',
            desc: '光源视角阴影图的分辨率（texel）。越低，一个 texel 里挤的发丝越多、z₀ 在相邻 texel 间跳得越厉害（Step 11 右图），各方法的误差都变大；DSM / OSM / DOM 和精确参考都在同一分辨率下求值，所以比较始终公平。' },
        { id: 'K', label: 'DOM 层数', options: [2, 3, 4, 6, 8], value: 3,
            desc: 'Deep Opacity Map 的层数 K（从 z₀ 起往后排，最后一层收纳其余片元）。原文通常取 3；Step 04 的 Ω_k 图、T(z) 曲线和 3D 层面随之变化，Step 05 的 RMS 曲线显示超过 3–4 层后误差基本不再下降。' },
        { id: 'dz', label: '首层厚 Δz', min: 0.05, max: 2, step: 0.05, value: 0.3, digits: 2, unit: ' cm',
            desc: 'DOM 第一层的厚度（cm），后面的层按"层分布"加厚。太薄 → 各层盖不住整团头发，后面的发丝挤进最后一层、远端过暗；太厚 → 第一层内的线性插值抹平了最陡的那段（Step 04 曲线上的竖线与 3D 层面、Step 05 误差图）。' },
        { id: 'dist', label: '层分布', options: [{ value: 'constant', label: '等厚' }, { value: 'linear', label: '线性递增' }, { value: 'pow2', label: '2 的幂' }], value: 'linear', title: '层厚：等厚 Δz / 线性递增 Δz·k（原文）/ 2 的幂',
            desc: 'DOM 各层末端 d_k 的增长方式：等厚 Δz·k、线性递增 Δz·k(k+1)/2（原文：层厚 s, 2s, 3s…）、2 的幂 Δz·(2<sup>k</sup> − 1)。同样层数下换成"等厚"误差变大——透射率在表层附近变化最快，层应当前薄后厚（Step 04 / 05）。' },
        { id: 'osm', label: 'OSM 切片', options: [4, 8, 16, 32, 64], value: 8,
            desc: 'Opacity Shadow Map 的均匀切片数 M，铺满整团头发在光源空间的深度范围。从 8 调到 64，Step 03 的误差图逐渐变白；Step 05 的 RMS 曲线里 OSM 要到约 64 张才追上 3 层 DOM。' },
        { id: 'eps', label: 'DSM ε', min: 0.002, max: 0.1, step: 0.002, value: 0.02, digits: 3, title: 'Deep Shadow Map 压缩容差',
            desc: 'Deep Shadow Map 分段线性压缩的容差。ε 越小顶点越多、压缩曲线贴得越紧（Step 02 中图变亮、V(z) 曲线的折点变多）；ε = 0.02 时多数 texel 只要个位数顶点。Step 05 / 11 曲线里的 DSM 误差线也随之变化。' },
        { id: 'bias', label: 'bias', min: 0, max: 1, step: 0.01, value: 0.02, digits: 2, unit: ' cm', title: '查询深度向光源偏移量',
            desc: '查询透射率时把深度朝光源推的距离（cm），用来跳过被着色发丝自己。DOM 在 bias = 0 时最好，DSM 需要约 0.05 cm，再大就漏光：看 Step 11 的 RMS–bias 曲线；Step 01、03–05 的透射率 / 误差图和 Step 07–10 的多次散射（都读同一个 T_DOM）也跟着变。' },
        { id: 'ce', label: '真黑素', min: 0, max: 3, step: 0.05, value: 0.3, digits: 2, title: 'eumelanin 浓度（pbrt-v3：0.3 金发，1.3 棕发，8 黑发）', steps: ['lookup', 'voxel'],
            desc: '真黑色素（eumelanin）浓度 → 吸收系数 σ<sub>a</sub>（pbrt-v3：0.3 金发，1.3 棕发，8 黑发）。它决定每穿过一根发丝还剩多少能量 ā_f：0.3 时多次散射铺满发帘内部，拉到 3 以上单次散射 + 吸收体阴影就够了（Step 07–09）；也改变 Step 05 / 06 的发色。' },
        { id: 'cp', label: '褐黑素', min: 0, max: 1.5, step: 0.05, value: 0.1, digits: 2, title: 'pheomelanin 浓度（偏红 / 橙）', steps: ['lookup', 'voxel'],
            desc: '褐黑色素（pheomelanin）浓度，偏红 / 橙，与真黑素一起决定 σ<sub>a</sub>。调高得到红发、铜色（Step 05 / 06 的着色）；多次散射表随之重烘焙（Step 07–09），Step 10 的两张 LUT 也整张重算。' },
        { id: 'df', label: 'd_f = d_b', min: 0, max: 1, step: 0.05, value: 0.7, digits: 2, title: 'Dual scattering 密度因子（原文取 0.7）',
            desc: 'Dual scattering 的密度因子（原文取 0.7），同时缩放全局前向散射与局部回散射的权重。0 = 退回单次散射 + 吸收体阴影；只影响 Step 07–09 的多次散射分量。' },
        { id: 'vox', label: '体素', options: [16, 24, 32, 48, 64], value: 32, title: '密度体素网格最长边分辨率',
            desc: '密度体素网格最长边的体素数。16 时阴影糊成一团、表层发丝也被自己所在体素的密度压暗（误差图偏蓝）；64 接近 DOM。只影响 Step 06。' },
        { id: 'voxBias', label: '体素 bias', min: 0, max: 3, step: 0.25, value: 1, digits: 2, unit: ' vox', title: '体素 ray-march 起点偏移（体素数）',
            desc: '体素 ray-march 的起点沿光线推出去的距离（以体素为单位）。0 → 每个点都被自己所在体素的密度遮挡（整体偏暗）；3 → 起点越过了真实遮挡物、表层附近漏光（偏红）。只影响 Step 06。' },
    ],

    stages: [
        {
            id: 'groom', lifetime: 'resident', run(S) {
                S.groom = HAIR.makeGroom({ seed: 7, perGuide: Math.round(S.count / 16) });
                S.box = L3.aabb(S.groom.strands);
                // fixed 3/4 viewing camera for the swatch
                S.vcam = L3.fitOrtho(L3.lightDir(0.35, 0.12), S.box, 0.06);
                S.view = HAIR.raster(S.groom.strands, S.vcam, L3.VIEW_RES, L3.VIEW_RES);
            },
        },
        {
            id: 'light', lifetime: 'param', run(S) {
                const key = [S.lightAz, S.lightEl, S.smRes, S.count].join();
                if (S._lightKey === key) return;
                S._lightKey = key;
                S.wi = L3.lightDir(S.lightAz, S.lightEl);
                S.lcam = L3.fitOrtho(S.wi, S.box, 0.04, S.groom.strands);
                // PASS 1 — shadow depth from the light (nearest strand per texel = first occluder z0)
                S.lr = HAIR.raster(S.groom.strands, S.lcam, S.smRes, S.smRes);
                // all fragments with their own depth (uncompressed deep shadow map / A-buffer)
                S.frags = L3.fragments(S.groom.strands, S.lcam, S.smRes, S.smRes);
                // key texel for the 1D plots: densest texel in the central half of the map
                const N = S.smRes; let best = -1, bi = 0;
                for (let y = N >> 2; y < N - (N >> 2); y++) for (let x = N >> 2; x < N - (N >> 2); x++) {
                    const i = y * N + x; if (S.lr.cov[i] > best) { best = S.lr.cov[i]; bi = i; }
                }
                S.pick = bi;
                // light-space position of each view sample
                const V = L3.VIEW_RES, n = V * V;
                S.lx = PL.buf(S, 'lx', n); S.ly = PL.buf(S, 'ly', n); S.lz = PL.buf(S, 'lz', n); S.vz = PL.buf(S, 'vz', n);
                for (let i = 0; i < n; i++) {
                    if (S.view.id[i] < 0) continue;
                    const p = L3.unproject(S.vcam, i % V, Math.floor(i / V), S.view.depth[i], V, V);
                    const q = HAIR.project(S.lcam, p, N, N);
                    S.lx[i] = q[0]; S.ly[i] = q[1]; S.lz[i] = q[2];
                    const t = Math.round(q[1]) * N + Math.round(q[0]);
                    S.vz[i] = t >= 0 && t < N * N && isFinite(S.lr.depth[t]) ? q[2] - S.lr.depth[t] : 0;
                }
            },
        },
        {
            id: 'dsm', lifetime: 'param', run(S) {
                // exact visibility V(z) = exp(−κ·n(z)) as a polyline through every step, then greedy compression (±ε)
                const F = S.frags, N = S.smRes, n = N * N;
                S.dsmVerts = PL.buf(S, 'dsmVerts', n); S.dsm = [];
                for (let i = 0; i < n; i++) {
                    const a = F.off[i], b = F.off[i + 1];
                    if (a === b) { S.dsm[i] = null; S.dsmVerts[i] = 0; continue; }
                    const zs = [F.z[a]], vs = [1]; let cnt = 0;
                    for (let f = a; f < b; f++) { cnt += F.c[f]; zs.push(F.z[f] + 1e-4); vs.push(Math.exp(-S.kappa * cnt)); }
                    S.dsm[i] = L3.compress(zs, vs, S.eps);
                    S.dsmVerts[i] = S.dsm[i].z.length;
                }
            },
        },
        {
            id: 'osm', lifetime: 'param', run(S) {
                // uniformly spaced slices over the groom's light-space depth range; Ω_j = Σ_{z_f < b_j} c_f (cumulative)
                const F = S.frags, N = S.smRes, n = N * N, M = S.osm;
                const z0 = S.lcam.zRange[0], z1 = S.lcam.zRange[1], dzs = (z1 - z0) / M;
                S.osmB = { z0, dz: dzs, M };
                const O = PL.buf(S, 'osmOmega', n * (M + 1));
                O.fill(0);
                for (let i = 0; i < n; i++) for (let f = F.off[i]; f < F.off[i + 1]; f++) {
                    const j = Math.max(0, Math.ceil((F.z[f] - z0) / dzs + 1e-6));   // first boundary behind the fragment
                    for (let q = Math.min(M, j); q <= M; q++) O[i * (M + 1) + q] += F.c[f];
                }
                S.osmOmega = O;
            },
        },
        {
            id: 'dom', lifetime: 'param', run(S) {
                // deep opacity map: layer k ends at z0 + k·Δz; Ω_k cumulative; fragments behind the last boundary → last layer
                const F = S.frags, N = S.smRes, n = N * N, K = S.K, ends = (S.domEnds = L3.layerEnds(K, S.dz, S.dist));
                const O = PL.buf(S, 'domOmega', n * K);
                O.fill(0);
                for (let i = 0; i < n; i++) {
                    const z0 = S.lr.depth[i];
                    if (!isFinite(z0)) continue;
                    for (let f = F.off[i]; f < F.off[i + 1]; f++) {
                        const k = L3.layerOf(ends, Math.max(0, F.z[f] - z0));   // fragments in front of z0 (sub-pixel misses of pass 1) → layer 1
                        for (let q = k; q < K; q++) O[i * K + q] += F.c[f];
                    }
                }
                S.domOmega = O;
            },
        },
        {
            id: 'viewT', lifetime: 'param', run(S) {
                const V = L3.VIEW_RES, n = V * V, N = S.smRes, F = S.frags, kap = S.kappa, K = S.K, M = S.osm;
                const T = (name) => (S[name] = PL.buf(S, name, n));
                const Tb = T('Tbin'), Tr = T('Tref'), Td = T('Tdom'), To = T('Tosm'), Ts = T('Tdsm'), Ed = T('Edom'), Eo = T('Eosm');
                const domAt = (i, z) => {
                    const z0 = S.lr.depth[i];
                    return isFinite(z0) ? Math.exp(-kap * L3.domOmega(S.domOmega, i * K, S.domEnds, z - z0)) : 1;
                };
                const osmAt = (i, z) => {
                    const u = (z - S.osmB.z0) / S.osmB.dz;
                    if (u <= 0) return 1;
                    const j = Math.min(M - 1, Math.floor(u)), f = Math.min(1, u - j);
                    const om = S.osmOmega[i * (M + 1) + j] * (1 - f) + S.osmOmega[i * (M + 1) + j + 1] * f;
                    return Math.exp(-kap * om);
                };
                for (let i = 0; i < n; i++) {
                    if (S.view.id[i] < 0) { Tb[i] = Tr[i] = Td[i] = To[i] = Ts[i] = 1; Ed[i] = Eo[i] = 0; continue; }
                    const lx = S.lx[i], ly = S.ly[i], zq = S.lz[i] - S.bias;
                    Tb[i] = L3.filter2(N, N, lx, ly, (t) => (isFinite(S.lr.depth[t]) && zq > S.lr.depth[t] ? 0 : 1));
                    const own = S.view.id[i];   // reference: no depth bias — every other strand in front counts, the shaded strand itself never does
                    Tr[i] = L3.filter2(N, N, lx, ly, (t) => Math.exp(-kap * L3.countAt(F, t, S.lz[i], own)));
                    Ts[i] = L3.filter2(N, N, lx, ly, (t) => (S.dsm[t] ? L3.polyAt(S.dsm[t].z, S.dsm[t].v, zq) : 1));
                    Td[i] = L3.filter2(N, N, lx, ly, (t) => domAt(t, zq));
                    To[i] = L3.filter2(N, N, lx, ly, (t) => osmAt(t, zq));
                    Ed[i] = Td[i] - Tr[i]; Eo[i] = To[i] - Tr[i];
                }
                S.domAt = domAt; S.osmAt = osmAt;
                const rms = (E) => { let s = 0, c = 0; for (let i = 0; i < n; i++) if (S.view.id[i] >= 0) { s += E(i) ** 2; c++; } return Math.sqrt(s / Math.max(1, c)); };
                S.rms = { bin: rms((i) => Tb[i] - Tr[i]), osm: rms((i) => Eo[i]), dom: rms((i) => Ed[i]), dsm: rms((i) => Ts[i] - Tr[i]) };
            },
        },
        {
            id: 'fiber', lifetime: 'param', run(S) {
                // single-fibre BSDF + every dual-scattering table (Zinke et al. Table 1) — re-baked only when absorption changes
                const key = [S.ce, S.cp].join();
                if (S._fiberKey === key) return;
                S._fiberKey = key;
                S.bsdf = HAIR.bsdf({ sigmaA: HAIR.sigmaAFromConcentration(S.ce, S.cp), betaM: 0.3, betaN: 0.3, alpha: 2 });
                S.ds = DS.bake(S.bsdf);
            },
        },
        {
            id: 'dualTerms', lifetime: 'param', run(S) {
                // per visible sample: fibre-local directions + the deterministic dual-scattering terms (Fig. 5 of the paper)
                const V = L3.VIEW_RES, n = V * V, wo = S.vcam.toViewer, b = S.bsdf, t = S.ds, PI = Math.PI;
                S.wiL = PL.buf(S, 'wiL', 3 * n); S.woL = PL.buf(S, 'woL', 3 * n);
                const Tf = (S.Tf = PL.buf(S, 'Tf', 3 * n)), Ss = (S.Ss = PL.buf(S, 'Ss', 3 * n)), Sbd = (S.Sbd = PL.buf(S, 'Sbd', 3 * n)), Sbs = (S.Sbs = PL.buf(S, 'Sbs', 3 * n));
                const M = (S.mCount = PL.buf(S, 'mCount', n));
                let thSum = 0, thN = 0;
                for (let i = 0; i < n; i++) {
                    if (S.view.id[i] < 0) continue;
                    const f = HAIR.fiberFrame([S.view.tan[3 * i], S.view.tan[3 * i + 1], S.view.tan[3 * i + 2]], wo);
                    const wi = HAIR.toFiber(S.wi, f), wv = HAIR.toFiber(wo, f);
                    S.wiL.set(wi, 3 * i); S.woL.set(wv, 3 * i);
                    const sinI = wi[0], cosI = Math.sqrt(Math.max(0, 1 - sinI * sinI)), sinO = wv[0], cosO = Math.sqrt(Math.max(0, 1 - sinO * sinO));
                    const thI = Math.asin(sinI), thO = Math.asin(sinO), ph = DS.wrapPi(Math.atan2(wi[2], wi[1]) - Math.atan2(wv[2], wv[1]));
                    const m = -Math.log(Math.max(1e-6, S.Tdom[i]));          // κ·n：从 deep opacity map 读回的期望发丝数
                    M[i] = m;
                    const jI = DS.row(t, thI), jD = DS.row(t, (thO - thI) / 2), cI = Math.max(0.15, cosI);
                    thSum += Math.abs(thI); thN++;
                    const back = Math.abs(ph) < PI / 2 ? 1 / PI : 0, tri = Math.max(0, PI - Math.abs(ph)) / (PI * PI);
                    for (let c = 0; c < 3; c++) {
                        const sf2 = m * t.bf2[jI][c];
                        Tf[3 * i + c] = Math.exp(-m * (1 - t.af[jI][c]));
                        let ss = 0;
                        for (let p = 0; p < 4; p++) ss += DS.lobeM(b, p, sinI, cosI, sinO, cosO, sf2) * DS.NGat(t, p, thO, ph, c);
                        Ss[3 * i + c] = ss;
                        const x = thI + thO - t.Db[jD][c];
                        Sbd[3 * i + c] = S.df * t.Ab[jD][c] * back * DS.gauss(x, t.sb2[jD][c]) / cI;
                        Sbs[3 * i + c] = S.df * t.Ab[jD][c] * tri * DS.gauss(x, t.sb2[jD][c] + sf2) / cI;
                    }
                }
                S.thMean = thSum / Math.max(1, thN);
                // display exposures (98th percentile → 0.8) from a quick h-quadrature on 400 visible samples:
                // expo = unshadowed single scattering (shadow comparisons), expoDual = dual scattering (shared by the single/dual comparison)
                const l1 = [], l2 = [], wi0 = [0, 0, 0], wv0 = [0, 0, 0], rnd = PL.rng(17);
                for (let k = 0; k < 400; k++) {
                    const i = Math.floor(rnd() * n); if (S.view.id[i] < 0) continue;
                    for (let a = 0; a < 3; a++) { wi0[a] = S.wiL[3 * i + a]; wv0[a] = S.woL[3 * i + a]; }
                    const sb = b.Sbar(wi0, wv0, 12), dF = S.Tdom[i];
                    l1.push(HAIR.lum(sb));
                    l2.push(HAIR.lum([0, 1, 2].map((c) => dF * (sb[c] + Sbd[3 * i + c]) + (Tf[3 * i + c] - dF) * S.df * (Ss[3 * i + c] + Sbs[3 * i + c]))));
                }
                const p98 = (a) => { a.sort((x, y) => x - y); return Math.max(1e-5, a[Math.floor(a.length * 0.98)] || 1e-3); };
                S.expo = 0.8 / p98(l1); S.expoDual = 0.8 / p98(l2);
            },
        },
        {
            id: 'accum', lifetime: 'history', run(S) {
                // single scattering S(wi, wo, h): one stochastic h per pixel per frame, accumulated while light / fibre are unchanged
                const V = L3.VIEW_RES, n = V * V, key = [S.lightAz, S.lightEl, S.ce, S.cp, S.count].join();
                S.acc = PL.buf(S, 'acc', 3 * n);
                if (S._accKey !== key) { S.acc.fill(0); S.accN = 0; S._accKey = key; }
                const b = S.bsdf, g = 0.6180339887 * S.accN, wi = [0, 0, 0], wv = [0, 0, 0];
                for (let i = 0; i < n; i++) {
                    if (S.view.id[i] < 0) continue;
                    const r = Math.sin(i * 12.9898 + 78.233) * 43758.5453, u = PL.fract(r - Math.floor(r) + g);
                    for (let a = 0; a < 3; a++) { wi[a] = S.wiL[3 * i + a]; wv[a] = S.woL[3 * i + a]; }
                    const s = b.S(wi, wv, 2 * u - 1);
                    S.acc[3 * i] += s[0]; S.acc[3 * i + 1] += s[1]; S.acc[3 * i + 2] += s[2];
                }
                S.accN++;
            },
        },
        {
            id: 'voxel', lifetime: 'param', lazy: true, run(S) {
                // density voxelization: every segment splats (length × width) into the voxel it passes; + width-weighted tangent
                const R = S.vox, key = [R, S.count].join();
                if (S._voxKey === key) return;
                S._voxKey = key;
                const B = S.box, pad = 0.3, lo = [B[0] - pad, B[1] - pad, B[2] - pad], ext = [0, 1, 2].map((a) => B[a + 3] - B[a] + 2 * pad);
                const vs = Math.max(ext[0], ext[1], ext[2]) / R, dims = ext.map((e) => Math.max(2, Math.ceil(e / vs))), nV = dims[0] * dims[1] * dims[2];
                const D = new Float32Array(nV), TT = new Float32Array(3 * nV);
                for (const s of S.groom.strands) {
                    const P = s.p, segs = P.length / 3 - 1;
                    for (let j = 0; j < segs; j++) {
                        const a = [P[3 * j], P[3 * j + 1], P[3 * j + 2]], d = [P[3 * j + 3] - a[0], P[3 * j + 4] - a[1], P[3 * j + 5] - a[2]];
                        const L = Math.hypot(d[0], d[1], d[2]), w = s.w0 + (s.w1 - s.w0) * (j + 0.5) / segs, ns = Math.max(1, Math.ceil(L / (vs / 2)));
                        for (let k = 0; k < ns; k++) {
                            const u = (k + 0.5) / ns, ix = [0, 1, 2].map((q) => Math.min(dims[q] - 1, Math.max(0, Math.floor((a[q] + u * d[q] - lo[q]) / vs))));
                            const v = (ix[2] * dims[1] + ix[1]) * dims[0] + ix[0], A = L / ns * w;
                            D[v] += A; TT[3 * v] += A * d[0] / L; TT[3 * v + 1] += A * d[1] / L; TT[3 * v + 2] += A * d[2] / L;
                        }
                    }
                }
                const inv = 1 / (vs * vs * vs);
                for (let v = 0; v < nV; v++) D[v] *= inv;          // ρ: projected fibre area per volume (1/cm)
                S.vgrid = { lo, vs, dims, D, TT };
            },
        },
        {
            id: 'voxT', lifetime: 'param', lazy: true, needs: ['voxel'], run(S) {
                // ray-march every visible sample toward the light through the density grid: τ = κ ∫ ρ·|t̄ × ωi| ds
                const g = S.vgrid, V = L3.VIEW_RES, n = V * V, wi = S.wi, Tv = (S.Tvox = PL.buf(S, 'Tvox', n)), Ev = (S.Evox = PL.buf(S, 'Evox', n));
                const [nx, ny, nz] = g.dims, ds = g.vs * 0.5;
                const tri = (x, y, z, out) => {   // trilinear ρ and orientation factor
                    const fx = (x - g.lo[0]) / g.vs - 0.5, fy = (y - g.lo[1]) / g.vs - 0.5, fz = (z - g.lo[2]) / g.vs - 0.5;
                    const x0 = Math.floor(fx), y0 = Math.floor(fy), z0 = Math.floor(fz);
                    let d = 0, t0 = 0, t1 = 0, t2 = 0;
                    for (let k = 0; k < 8; k++) {
                        const X = x0 + (k & 1), Y = y0 + ((k >> 1) & 1), Z = z0 + (k >> 2);
                        if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) continue;
                        const w = (k & 1 ? fx - x0 : 1 - fx + x0) * ((k >> 1) & 1 ? fy - y0 : 1 - fy + y0) * (k >> 2 ? fz - z0 : 1 - fz + z0), v = (Z * ny + Y) * nx + X;
                        d += w * g.D[v]; t0 += w * g.TT[3 * v]; t1 += w * g.TT[3 * v + 1]; t2 += w * g.TT[3 * v + 2];
                    }
                    const tl = Math.hypot(t0, t1, t2) || 1, c = (t0 * wi[0] + t1 * wi[1] + t2 * wi[2]) / tl;
                    out[0] = d * Math.sqrt(Math.max(0, 1 - c * c));
                };
                const tmp = [0], ext = Math.hypot(nx, ny, nz) * g.vs;
                for (let i = 0; i < n; i++) {
                    if (S.view.id[i] < 0) { Tv[i] = 1; Ev[i] = 0; continue; }
                    const p = L3.unproject(S.vcam, i % V, Math.floor(i / V), S.view.depth[i], V, V);
                    let tau = 0;
                    for (let s = g.vs * S.voxBias; s < ext; s += ds) {
                        const x = p[0] + wi[0] * s, y = p[1] + wi[1] * s, z = p[2] + wi[2] * s;
                        if (x < g.lo[0] || y < g.lo[1] || z < g.lo[2] || x > g.lo[0] + nx * g.vs || y > g.lo[1] + ny * g.vs || z > g.lo[2] + nz * g.vs) break;
                        tri(x, y, z, tmp); tau += tmp[0] * ds;
                    }
                    Tv[i] = Math.exp(-S.kappa * tau); Ev[i] = Tv[i] - S.Tref[i];
                }
                let s2 = 0, c = 0; for (let i = 0; i < n; i++) if (S.view.id[i] >= 0) { s2 += Ev[i] ** 2; c++; }
                S.rmsVox = Math.sqrt(s2 / Math.max(1, c));
            },
        },
        {
            id: 'sweep', lifetime: 'param', lazy: true, run(S) {
                // RMS(T − T_ref) over the visible samples for several layer / slice counts (same Δz, same bias)
                const F = S.frags, N = S.smRes, V = L3.VIEW_RES, n = V * V, kap = S.kappa;
                const Ks = [1, 2, 3, 4, 6, 8], Ms = [2, 4, 8, 16, 32, 64], z0g = S.lcam.zRange[0], zr = S.lcam.zRange[1] - z0g;
                const evalRms = (lookup) => { let s = 0, c = 0; for (let i = 0; i < n; i++) if (S.view.id[i] >= 0) { const e = L3.filter2(N, N, S.lx[i], S.ly[i], (t) => lookup(t, S.lz[i] - S.bias)) - S.Tref[i]; s += e * e; c++; } return Math.sqrt(s / Math.max(1, c)); };
                S.sweepDom = Ks.map((K) => {
                    const O = new Float32Array(N * N * K);
                    const ends = L3.layerEnds(K, S.dz, S.dist);
                    for (let i = 0; i < N * N; i++) { const z0 = S.lr.depth[i]; if (!isFinite(z0)) continue; for (let f = F.off[i]; f < F.off[i + 1]; f++) { const k = L3.layerOf(ends, Math.max(0, F.z[f] - z0)); for (let q = k; q < K; q++) O[i * K + q] += F.c[f]; } }
                    return evalRms((i, z) => { const z0 = S.lr.depth[i]; return isFinite(z0) ? Math.exp(-kap * L3.domOmega(O, i * K, ends, z - z0)) : 1; });
                });
                S.sweepOsm = Ms.map((M) => {
                    const dzs = zr / M, O = new Float32Array(N * N * (M + 1));
                    for (let i = 0; i < N * N; i++) for (let f = F.off[i]; f < F.off[i + 1]; f++) { const j = Math.max(0, Math.ceil((F.z[f] - z0g) / dzs + 1e-6)); for (let q = Math.min(M, j); q <= M; q++) O[i * (M + 1) + q] += F.c[f]; }
                    return evalRms((i, z) => { const u = (z - z0g) / dzs; if (u <= 0) return 1; const j = Math.min(M - 1, Math.floor(u)), fr = Math.min(1, u - j); return Math.exp(-kap * (O[i * (M + 1) + j] * (1 - fr) + O[i * (M + 1) + j + 1] * fr)); });
                });
                S.sweepKs = Ks; S.sweepMs = Ms;
            },
        },
        {
            id: 'biasSweep', lifetime: 'param', lazy: true, run(S) {
                // RMS(T − T_ref) vs depth bias; the reference itself never uses a bias (it skips the shaded strand by id)
                const N = S.smRes, V = L3.VIEW_RES, n = V * V, kap = S.kappa, bs = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1];
                const out = { b: bs, sm: [], dsm: [], dom: [] };
                for (const bias of bs) {
                    const acc = [0, 0, 0]; let c = 0;
                    for (let i = 0; i < n; i++) {
                        if (S.view.id[i] < 0) continue;
                        const zq = S.lz[i] - bias, r = S.Tref[i]; c++;
                        acc[0] += (L3.filter2(N, N, S.lx[i], S.ly[i], (t) => (isFinite(S.lr.depth[t]) && zq > S.lr.depth[t] ? 0 : 1)) - r) ** 2;
                        acc[1] += (L3.filter2(N, N, S.lx[i], S.ly[i], (t) => (S.dsm[t] ? L3.polyAt(S.dsm[t].z, S.dsm[t].v, zq) : 1)) - r) ** 2;
                        acc[2] += (L3.filter2(N, N, S.lx[i], S.ly[i], (t) => S.domAt(t, zq)) - r) ** 2;
                    }
                    out.sm.push(Math.sqrt(acc[0] / c)); out.dsm.push(Math.sqrt(acc[1] / c)); out.dom.push(Math.sqrt(acc[2] / c));
                }
                S.biasSw = out;
            },
        },
        {
            id: 'lut2d', lifetime: 'param', lazy: true, run(S) {
                // engine-style LUT: (|θ|, eumelanin) → ā_f, Ā_b (luminance); one energy-only bake per column
                const key = [S.cp].join();
                if (S._lutKey === key) return;
                S._lutKey = key;
                const NC = 24, NT = DS.NT, ces = Array.from({ length: NC }, (_, k) => 4 * (k / (NC - 1)) ** 2);
                const af = new Float32Array(NT * NC), Ab = new Float32Array(NT * NC);
                ces.forEach((ce, x) => {
                    const t = DS.bake(HAIR.bsdf({ sigmaA: HAIR.sigmaAFromConcentration(ce, S.cp), betaM: 0.3, betaN: 0.3, alpha: 2 }), false);
                    for (let j = 0; j < NT; j++) { af[(NT - 1 - j) * NC + x] = HAIR.lum(t.af[j]); Ab[(NT - 1 - j) * NC + x] = HAIR.lum(t.Ab[j]); }
                });
                S.lut = { NC, NT, ces, af, Ab };
            },
        },
        {
            id: 'melSweep', lifetime: 'param', lazy: true, run(S) {
                // energy budget vs eumelanin (energy tables only, no N^G): how much of the dual-scattering energy single scattering keeps
                const ces = [0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3, 4], ms = [0.5, 1, 2, 4], th = S.thMean || 0;
                S.melSw = { ce: ces, ms, ratio: ms.map(() => []), af: [] };
                for (const ce of ces) {
                    const t = DS.bake(HAIR.bsdf({ sigmaA: HAIR.sigmaAFromConcentration(ce, S.cp), betaM: 0.3, betaN: 0.3, alpha: 2 }), false), j = DS.row(t, th);
                    const af = HAIR.lum(t.af[j]), ab = HAIR.lum(t.ab[j]), Ab = HAIR.lum(t.Ab[j]);
                    S.melSw.af.push(af);
                    ms.forEach((m, k) => {
                        const direct = Math.exp(-m), tf = Math.exp(-m * (1 - af));
                        const single = direct * ab, dual = direct * (ab + S.df * Ab) + (tf - direct) * S.df * (ab + S.df * Ab);
                        S.melSw.ratio[k].push(single / Math.max(1e-9, dual));
                    });
                }
            },
        },
        {
            id: 'dom3d', lifetime: 'param', lazy: true, run(S) {
                // 3D 视图数据（只读已有结果）：① 一部分发丝每段中点的 T_DOM——与着色查询完全相同（2×2 双线性 + bias）；
                // ② 光源视角里几条竖直扫描列：每列按 4 texel 高的格子取第一遮挡深度 z₀（格内最近），再沿光线往里推 d_k
                const N = S.smRes, st = S.groom.strands, step = Math.max(1, Math.ceil(st.length / DOM3D.maxStrands)), strands = [];
                for (let i = 0; i < st.length; i += step) {
                    const P = st[i].p, n = P.length / 3, T = new Float32Array(n - 1);
                    for (let k = 0; k + 1 < n; k++) {
                        const q = HAIR.project(S.lcam, [(P[3 * k] + P[3 * k + 3]) / 2, (P[3 * k + 1] + P[3 * k + 4]) / 2, (P[3 * k + 2] + P[3 * k + 5]) / 2], N, N);
                        T[k] = L3.filter2(N, N, q[0], q[1], (t) => S.domAt(t, q[2] - S.bias));
                    }
                    strands.push({ p: P, T });
                }
                const cw = DOM3D.cell, nc = Math.floor(N / cw), ends = [0].concat(S.domEnds), D = S.lr.depth;
                let x0 = N, x1 = -1;
                for (let i = 0; i < N * N; i++) if (isFinite(D[i])) { const x = i % N; x0 = Math.min(x0, x); x1 = Math.max(x1, x); }
                const cols = [];
                for (let r = 0; r < DOM3D.cols; r++) {
                    const xc = Math.round(x0 + (x1 - x0) * (r + 0.5) / DOM3D.cols), L = ends.map(() => new Float32Array(3 * nc).fill(NaN));
                    for (let cy = 0; cy < nc; cy++) {
                        let z0 = Infinity;
                        for (let y = cy * cw; y < (cy + 1) * cw; y++) for (let x = Math.max(0, xc - 1); x <= Math.min(N - 1, xc + 1); x++) z0 = Math.min(z0, D[y * N + x]);
                        if (isFinite(z0)) ends.forEach((d, k) => L[k].set(L3.unproject(S.lcam, xc, (cy + 0.5) * cw - 0.5, z0 + d, N, N), 3 * cy));
                    }
                    cols.push(L);
                }
                S.dom3d = { strands, cols, nc, ends };
            },
        },
    ],

    intro: {
        title: '自阴影与多次散射 · 从"多少根头发挡在前面"到"光在发丝间怎么走"',
        flow: ['光源视角深度 z₀', 'Opacity 分层累积', '透射率查询 T(z)', '体素密度', 'Dual Scattering', '着色合成'],
        html: HAIR.nav('lighting') +
            '<p>上一篇解决了"一根发丝怎么散射"，这一篇解决"<strong>一缕头发里的光</strong>"：成千上万根半透明发丝互相遮挡（自阴影），而浅色发丝把大部分能量向前透射，光在发丝之间多次散射。页面里跑着一套完整的 JS 参考实现：从光源视角光栅化同一份 groom（<code>HAIR.raster</code>），逐 texel 记录所有片元，分别实现 Deep Shadow Map、Opacity Shadow Map、Deep Opacity Map、密度体素和 Dual Scattering，并和"精确透射率"逐像素对比。</p>' +
            '<p><strong>本页统一约定</strong>：每根渲染发丝代表 κ 根真实发丝；光源视角 texel 内片元覆盖率之和 = <em>hair count</em> n(z)，透射率 <code>T(z) = exp(−κ·n(z))</code>。所有方法都是在近似这条曲线；"精确参考" = 未压缩的逐 texel 片元列表，在与各方法相同的阴影图分辨率下求值。</p>',
    },

    steps: [
        {
            id: 'why', title: '为什么毛发需要"体积"阴影', tag: 'WHY',
            desc: '一缕头发 = 数千根 50–100 µm 的半透明圆柱，从光源看过去，一个阴影图 texel 里往往有几十根发丝，<strong>每根只遮住 texel 的百分之几</strong>。普通 shadow map 每个 texel 只存一个深度，回答的是"有没有东西挡住"——第一根发丝之后全部判为全黑，发丝又细又密，结果就是锯齿 + 闪烁 + 整缕头发发黑。我们真正需要的是<strong>沿光线方向穿过发丝密度后的透射率</strong>：前面每多一层头发，光就衰减一点。',
            formula: [
                'T(x) = Π_i (1 − c_i)^κ ≈ exp(−κ · n(x))        // c_i：第 i 根发丝在 texel 内的覆盖率，κ：每根渲染发丝代表的真实发丝数',
                'n(x) = Σ_{z_i < z(x)} c_i                       // hair count：x 与光源之间的累计覆盖',
                'T_SM(x) = z(x) > z_0 + bias ? 0 : 1             // 普通 shadow map：只剩一个比较',
            ],
            code: {
                title: 'HLSL · 普通 shadow map 用在毛发上（反例）', src: `
                Texture2D<float> HairShadowDepth;      // 光源视角最近发丝深度 z0
                SamplerComparisonState ShadowCmp;

                float HairShadow_Binary(float3 LightUVZ)
                {
                    // 2x2 硬件 PCF 只能把 0/1 的边缘抹软，第一根发丝之后仍是全黑
                    return HairShadowDepth.SampleCmpLevelZero(ShadowCmp, LightUVZ.xy, LightUVZ.z - DepthBias);
                }`,
            },
            previews: [
                { id: 'lalpha', label: '光源视角 alpha', sub: '1 − Π(1 − c) · HAIR.raster', render: (c, S) => PL.draw.field(c, S.lr.alpha, S.smRes, S.smRes, { range: [0, 1] }), probe: L3V.lightProbe },
                { id: 'tbin', label: 'T · 普通 shadow map', sub: '2×2 PCF，第一根之后全黑', render: (c, S) => PL.draw.field(c, L3V.mask(S, S.Tbin), L3.VIEW_RES, L3.VIEW_RES, { range: [0, 1], map: 'inferno' }), probe: L3V.tProbe('T_SM', 'Tbin') },
                { id: 'tref', label: 'T · 精确透射率', sub: 'exp(−κ·n)，逐片元求和', render: (c, S) => PL.draw.field(c, L3V.mask(S, S.Tref), L3.VIEW_RES, L3.VIEW_RES, { range: [0, 1], map: 'inferno' }), probe: L3V.tProbe('T_ref', 'Tref') },
            ],
            notes: '左：光源视角的 alpha——几乎没有 texel 是 1，hair 对光源是"雾"而不是"墙"。中、右：相机视角下每个可见发丝点的透射率（inferno：亮 = 受光）。拖<strong>光源方位</strong>到 90°（侧光，光要穿过整片 8 cm 宽的发帘），普通 shadow map 几乎全黑，而精确透射率仍是一个平滑的由亮到暗渐变；把 <strong>κ</strong> 调小（稀疏发丝）差距更明显。',
            warn: '普通 shadow map 在毛发上还有一个时域问题：发丝宽度远小于 texel，最近深度 z₀ 由哪根发丝"中奖"决定，光源或发丝一动，z₀ 就跳——这是毛发阴影闪烁的根源。',
        },
        {
            id: 'dsm', title: 'Deep Shadow Maps：每个 texel 一条可见度函数', tag: 'DSM',
            desc: 'Lokovic & Veach（SIGGRAPH 2000）的回答：shadow map 的每个 texel 不再存一个深度，而是存一条<strong>可见度函数</strong> V(z)——沿这条光线、深度 z 处还剩多少光。对发丝来说它是一串小台阶（每穿过一根发丝乘一次 1 − c）。原文用于离线渲染（RenderMan），把 texel 内所有样本的可见度函数平均后，用<strong>分段线性函数在容差 ε 内贪心压缩</strong>，只存顶点。这是"最正确"的表示，但每个 texel 顶点数不定、要排序——实时 GPU 上需要逐像素链表（per-pixel linked list）或 A-buffer，显存与带宽都不可控。',
            formula: [
                'V(z) = Π_{z_i < z} (1 − c_i)^κ ≈ exp(−κ · n(z))     // texel 的可见度（= 透射率）函数',
                '压缩：从当前顶点出发，维护可行斜率区间 [s_lo, s_hi]，使所有后续采样点落在 ±ε 内；区间为空 → 落一个新顶点',
                '查询：V(z) = lerp(V_k, V_{k+1}, (z − z_k)/(z_{k+1} − z_k))   // 在压缩后的顶点之间线性插值',
            ],
            code: {
                title: 'HLSL · 实时 DSM：逐像素链表收集片元（压缩 / 排序在后续 CS 中做）', src: `
                struct FHairFragment { float Depth; float Coverage; uint Next; };
                RWStructuredBuffer<FHairFragment> FragmentPool;
                RWTexture2D<uint>                 HeadPointer;      // 0xFFFFFFFF = 空
                RWBuffer<uint>                    PoolCounter;

                void PS_HairDeepShadowCollect(float4 SvPos : SV_Position, float LightZ : TEXCOORD0, float Coverage : TEXCOORD1)
                {
                    uint Index;
                    InterlockedAdd(PoolCounter[0], 1, Index);
                    if (Index >= MaxFragments) return;              // 池满：直接丢片元 → 阴影漏光
                    uint Prev;
                    InterlockedExchange(HeadPointer[uint2(SvPos.xy)], Index, Prev);
                    FragmentPool[Index].Depth = LightZ;
                    FragmentPool[Index].Coverage = Coverage;
                    FragmentPool[Index].Next = Prev;
                }`,
            },
            previews: [
                { id: 'fragcnt', label: '片元数 / texel', sub: '未压缩 DSM 的存储量', render: (c, S) => { const n = S.smRes ** 2, a = PL.buf(S, '_fc', n); for (let i = 0; i < n; i++) a[i] = S.frags.off[i + 1] - S.frags.off[i] || NaN; PL.draw.field(c, a, S.smRes, S.smRes, { map: 'viridis' }); }, probe: L3V.lightProbe },
                { id: 'dsmverts', label: '压缩后顶点数', sub: '容差 ε 下的分段线性', render: (c, S) => { const n = S.smRes ** 2, a = PL.buf(S, '_dv', n); for (let i = 0; i < n; i++) a[i] = S.dsmVerts[i] || NaN; PL.draw.field(c, a, S.smRes, S.smRes, { map: 'viridis' }); }, probe: (x, y, S) => `顶点数 = ${S.dsmVerts[y * S.smRes + x]}\n片元数 = ${S.frags.off[y * S.smRes + x + 1] - S.frags.off[y * S.smRes + x]}` },
                {
                    id: 'vplot', label: 'V(z) · 关键 texel', sub: '自动选中最密的 texel', size: [360, 180],
                    render: (c, S) => {
                        const F = S.frags, i = S.pick, z0 = S.lr.depth[i], xs = [], ys = [];
                        let cnt = 0; xs.push(-0.3); ys.push(1);
                        for (let f = F.off[i]; f < F.off[i + 1]; f++) { xs.push(F.z[f] - z0); ys.push(Math.exp(-S.kappa * cnt)); cnt += F.c[f]; xs.push(F.z[f] - z0); ys.push(Math.exp(-S.kappa * cnt)); }
                        const zEnd = xs[xs.length - 1] + 0.3; xs.push(zEnd); ys.push(ys[ys.length - 1]);
                        const d = S.dsm[i];
                        PL.draw.plot(c, {
                            x: [-0.3, zEnd], y: [0, 1.02], xLabel: 'z − z₀ (cm)', yLabel: 'V',
                            series: [
                                { data: ys, x: xs, color: '#e0e0e0', label: '精确（逐片元台阶）', width: 1 },
                                { data: [1].concat(d.v), x: [-0.3].concat(d.z.map((z) => z - z0)), color: '#e0a060', label: `DSM 压缩 · ${d.z.length} 顶点`, width: 2 },
                            ],
                        });
                    },
                },
            ],
            notes: '拖 <strong>DSM ε</strong>：ε 越小顶点越多（中图变亮），曲线贴得越紧；ε = 0.02 时多数 texel 只需要个位数的顶点——这就是原文"压缩后只比普通 shadow map 大几倍"的原因。左图的片元数才是 GPU 上逐像素链表真正要付出的存储。',
            warn: 'DSM 原文的 V(z) 来自 texel 内大量子样本的平均（自带抗锯齿）；这里用的是解析覆盖率 c_i，压缩算法沿用"可行斜率区间"的贪心思路，细节（误差度量、顶点放置）是本页的简化实现。',
        },
        {
            id: 'osm', title: 'Opacity Shadow Maps：均匀切片', tag: 'OSM',
            desc: 'Kim & Neumann（EGWR 2001）把可见度函数换成固定格式：沿光线方向在整团头发的深度范围内放 M 张<strong>均匀间隔的 opacity 切片</strong>，每张存"这个深度之前累计了多少不透明度 Ω"，查询时在相邻两张切片之间线性插值，再用 <code>exp(−κΩ)</code> 转成透射率。好处是格式固定、可以用加法混合一遍画完；坏处是切片只按<strong>全局深度</strong>均匀划分——头发表面离光源的距离逐 texel 不同，大部分切片落在空气里，真正有头发的地方只分到一两张。',
            formula: [
                'b_j = z_min + j · (z_max − z_min) / M,   j = 0..M        // 全局均匀切片边界',
                'Ω_j = Σ_{z_i < b_j} c_i                                   // 第 j 张切片：累计不透明度（覆盖率之和）',
                'Ω(z) = lerp(Ω_j, Ω_{j+1}, (z − b_j)/(b_{j+1} − b_j)),   T(z) = exp(−κ · Ω(z))',
            ],
            code: {
                title: 'HLSL · OSM：每个片元按全局深度落到切片，加法混合写累计值', src: `
                // M 张切片打包进 M/4 张 RGBA16F（MRT），混合模式 ONE + ONE
                float4 SliceBoundaries[M / 4];   // b_1..b_M（光源空间深度）

                void PS_HairOpacitySlices(float4 SvPos : SV_Position, float LightZ : TEXCOORD0, float Coverage : TEXCOORD1,
                                          out float4 Slices[M / 4] : SV_Target0)
                {
                    [unroll] for (int s = 0; s < M / 4; ++s)
                        Slices[s] = Coverage * float4(LightZ < SliceBoundaries[s]);   // 片元之后的所有切片都 +c
                }`,
            },
            previews: [
                {
                    id: 'osmslices', label: 'Ω_j · 4 张切片', sub: '左→右：j = M/4, M/2, 3M/4, M', size: [360, 96],
                    render: (c, S) => {
                        const N = S.smRes, M = S.osm, js = [M >> 2, M >> 1, (3 * M) >> 2, M].map((j) => Math.max(1, j));
                        const W4 = N * 4, a = PL.buf(S, '_osl', W4 * N);
                        for (let y = 0; y < N; y++) for (let q = 0; q < 4; q++) for (let x = 0; x < N; x++) a[y * W4 + q * N + x] = S.osmOmega[(y * N + x) * (M + 1) + js[q]];
                        PL.draw.field(c, a, W4, N, { map: 'viridis', range: [0, 2.5] });
                    },
                },
                {
                    id: 'osmplot', label: 'T(z) · 关键 texel', sub: 'OSM vs 精确', size: [360, 180],
                    render: (c, S) => {
                        const F = S.frags, i = S.pick, z0 = S.lr.depth[i], zEnd = F.z[F.off[i + 1] - 1] - z0 + 0.6;
                        const marks = [];
                        for (let j = 0; j <= S.osm; j++) { const b = S.osmB.z0 + j * S.osmB.dz - z0; if (b > -0.3 && b < zEnd) marks.push({ x: b, color: '#56b6c2' }); }
                        PL.draw.plot(c, {
                            x: [-0.3, zEnd], y: [0, 1.02], xLabel: 'z − z₀ (cm)', marks, samples: 400,
                            series: [
                                { fn: (z) => Math.exp(-S.kappa * L3.countAt(F, i, z + z0)), color: '#e0e0e0', label: '精确', width: 1 },
                                { fn: (z) => S.osmAt(i, z + z0), color: '#56b6c2', label: `OSM · ${S.osm} 切片`, width: 2 },
                            ],
                        });
                    },
                },
                { id: 'eosm', label: 'T_OSM − T_ref', sub: '相机视角误差（红 = 偏亮/漏光）', render: (c, S) => PL.draw.field(c, L3V.mask(S, S.Eosm), L3.VIEW_RES, L3.VIEW_RES, { range: [-0.5, 0.5], map: 'diverge' }), probe: L3V.tProbe('T_OSM', 'Tosm') },
            ],
            notes: '右上曲线里的竖线是切片边界：在头发真正开始的 z₀ 附近，T 从 1 掉下来的那一段可能只有一两张切片，插值把它拉成一条斜线——表层发丝被错误地"提前变暗"或"推后变暗"。把 <strong>OSM 切片</strong>从 8 调到 64，误差图逐渐变白；原文正是靠几十到上百张切片换精度。',
        },
        {
            id: 'dom', title: 'Deep Opacity Maps：从第一遮挡者开始分层', tag: 'DOM',
            desc: 'Yuksel & Keyser（EG 2008）的关键观察：OSM 浪费切片，是因为切片从一个<strong>全局</strong>平面开始。先用一遍普通深度 pass 得到每个 texel 的第一遮挡者深度 z₀，再把 K 个 opacity 层<strong>贴着 z₀ 往后排</strong>——每个 texel 的层都从它自己的头发表面开始，几层就能覆盖透射率变化最剧烈的那一段。第二遍用加法混合，把片元的不透明度加到<strong>它所在的层以及之后所有层</strong>（存的是累计值）。原文的经验：通常 3 层就够，层厚<strong>线性递增</strong>（s, 2s, 3s…）；一张 RGBA 里 R 存 z₀、GBA 存 3 层（n 个 draw buffer → 4n − 1 层）；落在最后一层之后的片元有三种处理（丢弃 / 并入最后一层 / 让最后一层覆盖整团头发），并入最后一层"通常效果合理"——本页也这样做。深度图只用来确定各层起点，不做二值判断，所以原文发现 8-bit 深度就够用。',
            formula: [
                'Pass 1:  z_0(x,y) = min_i z_i                               // 只写深度（与普通 shadow map 相同）',
                'Pass 2:  Ω_k = Σ_{z_i < z_0 + d_k} c_i,   k = 1..K              // 累计式；Ω_K 收纳所有剩余片元',
                'd_k = Δz · k（等厚） | Δz · k(k+1)/2（线性递增，原文） | Δz · (2^k − 1)',
                '查询:    Ω(z) = lerp(Ω_{k−1}, Ω_k, (z − z_0 − d_{k−1}) / (d_k − d_{k−1})),  Ω_0 = 0,   T = exp(−κ · Ω(z))',
            ],
            code: {
                title: 'HLSL · PS_HairOpacityLayers（第二遍：R = z₀，GBA = 3 层累计 Ω）', src: `
                Texture2D<float> HairShadowDepth;     // Pass 1 输出：z0
                float2 LayerEnds;                     // d1, d2（线性递增：Δz·(1, 3)）；第 3 层收纳其余全部

                // 混合 ONE + ONE，R 通道用 write mask 关掉（保留 Pass 1 写入的 z0）
                // 片元的不透明度加到"它所在的层 + 之后所有层" → 每个通道直接是累计值 Ω_k
                float4 PS_HairOpacityLayers(float4 SvPos : SV_Position, float LightZ : TEXCOORD0, float Coverage : TEXCOORD1) : SV_Target0
                {
                    float z0 = HairShadowDepth[uint2(SvPos.xy)];
                    float t  = max(LightZ - z0, 0);                     // 0 = 第一遮挡者
                    return float4(0, Coverage * float(t < LayerEnds.x), Coverage * float(t < LayerEnds.y), Coverage);
                }`,
            },
            previews: [
                {
                    id: 'dom3d', needs: ['dom3d'], large: true, label: '3D：光源空间里的 z₀ 面与 DOM 层',
                    sub: '拖动旋转（默认视线垂直于光线）。细线 = 发丝，亮度 = T_DOM（暖白 = 受光，深灰 = 被挡住；与着色 pass 的查询相同）；蓝线 = Pass 1 的第一遮挡深度 z₀（光源视角的 5 条竖直扫描列，4 texel 一格取最近），橙线 = 各层末端 z₀ + d_k（越深越淡）——每层都是 z₀ 曲线沿光线平移的副本，贴着发丝表面，而不是 OSM 那样的全局平面；黄虚线 = 几条光线，圆点 = 它与 z₀ 和各层末端的交点',
                    render: (c, S) => drawDom3d(c, S),
                },
                { id: 'z0', label: 'z₀ · Pass 1 深度', sub: 'HAIR.raster().depth（近 = 暗）', render: (c, S) => { const n = S.smRes ** 2, d = PL.buf(S, '_z0', n); for (let i = 0; i < n; i++) d[i] = isFinite(S.lr.depth[i]) ? S.lr.depth[i] : NaN; PL.draw.field(c, d, S.smRes, S.smRes, { map: 'viridis' }); }, probe: L3V.lightProbe },
                {
                    id: 'domlayers', label: 'Ω_k · 各层累计', sub: '左→右：k = 1..K（最多 4 张）', size: [360, 96],
                    render: (c, S) => {
                        const N = S.smRes, K = S.K, ks = K <= 4 ? Array.from({ length: K }, (_, k) => k) : [0, 1, Math.floor(K / 2), K - 1];
                        const W4 = N * ks.length, a = PL.buf(S, '_dl', W4 * N);
                        for (let y = 0; y < N; y++) for (let q = 0; q < ks.length; q++) for (let x = 0; x < N; x++) a[y * W4 + q * N + x] = S.domOmega[(y * N + x) * K + ks[q]];
                        PL.draw.field(c, a, W4, N, { map: 'viridis', range: [0, 2.5] });
                    },
                },
                {
                    id: 'domplot', label: 'T(z) · 关键 texel', sub: 'DOM vs OSM vs 精确', size: [360, 180],
                    render: (c, S) => {
                        const F = S.frags, i = S.pick, z0 = S.lr.depth[i], zEnd = F.z[F.off[i + 1] - 1] - z0 + 0.6;
                        const marks = [];
                        [0].concat(S.domEnds).forEach((d, k) => { if (d < zEnd) marks.push({ x: d, color: '#e0a060', label: k === 0 ? 'z₀' : '' }); });
                        PL.draw.plot(c, {
                            x: [-0.3, zEnd], y: [0, 1.02], xLabel: 'z − z₀ (cm)', marks, samples: 400,
                            series: [
                                { fn: (z) => Math.exp(-S.kappa * L3.countAt(F, i, z + z0)), color: '#e0e0e0', label: '精确', width: 1 },
                                { fn: (z) => S.osmAt(i, z + z0), color: '#56b6c2', label: `OSM · ${S.osm}`, width: 1.2, dash: [4, 3] },
                                { fn: (z) => S.domAt(i, z + z0), color: '#e0a060', label: `DOM · ${S.K} 层`, width: 2 },
                            ],
                        });
                    },
                },
            ],
            notes: '下图竖线 = DOM 层边界（从 z₀ 起）。3 层 DOM 在 z₀ 附近就贴住了精确曲线，而 8 张 OSM 切片还在"台阶状"地追赶。拖 <strong>首层厚 Δz</strong>：太薄 → 各层覆盖不到整团头发，后面的发丝全部挤进最后一层，在最后一层里被线性插值"提前"算进来（远端过暗，原文 Fig. 8 的暗斑）；太厚 → 第一层里线性插值抹平了最陡的那段。<strong>层分布</strong>换成"等厚"，同样层数下误差变大——透射率在表层附近变化最快，越往里越平，所以层应当前薄后厚。',
            warn: '原文只说"在头发体积内插值"，没有给出层内插值公式和 Ω → 阴影的函数；本页沿用 OSM 的 exp(−κΩ) 并在层边界之间线性插值。z₀ 来自 Pass 1（HAIR.raster 的最近深度），亚像素发丝在 Pass 1 里可能没写深度，这类落在 z₀ 之前的片元并入第 1 层。',
        },
        {
            id: 'lookup', title: '着色 pass 查询透射率：有 / 无自阴影的发帘', tag: 'LOOKUP',
            desc: '着色时把可见发丝点 P 变换到光源空间，得到 texel 坐标和深度 z，在 4 个相邻 texel 上各自按 DOM 求 T，再双线性混合（等价于对透射率做 2×2 PCF）。单次散射着色 = <code>S(ωi, ωo, h) · E · T</code>。注意 S 对 h 的积分不能用小样本数中点公式——这里每帧每像素随机取一个 h，由 HISTORY 累积收敛（实时引擎里由 TAA 完成同样的事）。',
            formula: [
                '(u, v, z) = WorldToLight · P,     T(P) = Σ_{4 texels} w_{uv} · exp(−κ · Ω_texel(z − bias))',
                'L_o = E · T(P) · ⟨S(ωi, ωo, h)⟩_h        // h ~ U(−1, 1)，每帧一个，history 平均',
                '误差 = T_DOM − T_ref,   T_ref = 同分辨率下逐片元精确求和（排除被着色发丝自身，不加 bias）',
            ],
            code: {
                title: 'HLSL · 着色 pass 中的 DOM 查询（R = z₀，GBA = 3 层）', src: `
                Texture2D<float4> DeepOpacityMap;       // R = z0，GBA = 累计 Ω_1..Ω_3
                float3 LayerEnds;                       // d1, d2, d3（相对 z0）
                float  Kappa, DepthBias;

                float DomHairCount(int2 Texel, float LightZ)
                {
                    float4 M = DeepOpacityMap[Texel];
                    float  t = max(LightZ - DepthBias - M.r, 0);
                    if (t < LayerEnds.x) return M.g * t / LayerEnds.x;
                    if (t < LayerEnds.y) return lerp(M.g, M.b, (t - LayerEnds.x) / (LayerEnds.y - LayerEnds.x));
                    return lerp(M.b, M.a, saturate((t - LayerEnds.y) / (LayerEnds.z - LayerEnds.y)));   // 超过最后一层 → Ω_3
                }

                float HairTransmittance_DOM(float3 WorldPos)
                {
                    float3 L = mul(float4(WorldPos, 1), WorldToLightTexel).xyz;  // texel 坐标 + 光源空间深度
                    int2   p = (int2)floor(L.xy);
                    float2 f = L.xy - p;
                    float  T00 = exp(-Kappa * DomHairCount(p,              L.z));
                    float  T10 = exp(-Kappa * DomHairCount(p + int2(1, 0), L.z));
                    float  T01 = exp(-Kappa * DomHairCount(p + int2(0, 1), L.z));
                    float  T11 = exp(-Kappa * DomHairCount(p + int2(1, 1), L.z));
                    return lerp(lerp(T00, T10, f.x), lerp(T01, T11, f.x), f.y);
                }

                // 单次散射着色：h 每帧随机，由 TAA / history 积分
                float3 ShadeHairSingle(FHairSample s, float3 Wi, float3 Wo, float E)
                {
                    float h = 2 * InterleavedGradientNoise(s.PixelPos + 5.588238 * (FrameIndex % 64)) - 1;   // Jimenez 2014：每帧平移 5.588238 px，64 帧一循环
                    return HairBSDF_S(Wi, Wo, s.Tangent, h) * E * HairTransmittance_DOM(s.WorldPos);
                }`,
            },
            previews: [
                { id: 'shNone', label: '无自阴影', sub: 'E · ⟨S⟩', live: true, render: L3V.shade('none'), probe: L3V.shadeProbe('none') },
                { id: 'shBin', label: '普通 shadow map', sub: 'E · ⟨S⟩ · T_SM', live: true, render: L3V.shade('bin'), probe: L3V.shadeProbe('bin') },
                { id: 'shDom', label: 'Deep Opacity Map', sub: 'E · ⟨S⟩ · T_DOM', live: true, render: L3V.shade('dom'), probe: L3V.shadeProbe('dom') },
                { id: 'shRef', label: '精确透射率', sub: 'E · ⟨S⟩ · T_ref', live: true, render: L3V.shade('ref'), probe: L3V.shadeProbe('ref') },
                { id: 'edom', label: 'T_DOM − T_ref', sub: '红 = 偏亮（漏光） · 蓝 = 偏暗', render: (c, S) => PL.draw.field(c, L3V.mask(S, S.Edom), L3.VIEW_RES, L3.VIEW_RES, { range: [-0.3, 0.3], map: 'diverge' }), probe: L3V.tProbe('T_DOM', 'Tdom') },
                {
                    id: 'rmsplot', label: 'RMS 误差 vs 层数', sub: 'DOM（层厚 Δz 固定）vs OSM（均匀切片）', size: [360, 180], needs: ['sweep'],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [1, 64], logX: true, y: [0, Math.max(0.05, S.rms.bin * 1.05)], xLabel: '层 / 切片数（log）', yLabel: 'RMS',
                        series: [
                            { data: S.sweepDom, x: S.sweepKs, color: '#e0a060', label: 'DOM', width: 2 },
                            { data: S.sweepOsm, x: S.sweepMs, color: '#56b6c2', label: 'OSM', width: 2 },
                            { data: [S.rms.dsm, S.rms.dsm], x: [1, 64], color: '#c678dd', label: `DSM · ε = ${S.eps}`, dash: [4, 3] },
                            { data: [S.rms.bin, S.rms.bin], x: [1, 64], color: '#e06c75', label: '普通 SM', dash: [4, 3] },
                        ],
                    }),
                },
            ],
            notes: '四张发帘共享同一份累积的 ⟨S⟩，只差一个透射率因子，所以差异完全来自阴影方法。普通 shadow map 把除最外层以外的发丝全部压黑并带出 texel 锯齿；DOM 与精确结果肉眼几乎一致——误差图里残留的红蓝主要出现在 z₀ 跳变的发帘边缘和层边界附近。右下（默认参数实测）：3 层 DOM 的 RMS ≈ 0.09，OSM 要到 64 张切片才追上（≈ 0.10）；DOM 超过 3–4 层后不再改善，因为剩余误差来自 z₀ 跳变和层内线性插值，而不是层数。',
            warn: '每像素只着色"最近的那根发丝"，像素 alpha 用 1 − exp(−κ·cov) 合成背景——这是教学简化；生产中的毛发通常每像素保留多个样本（visibility buffer / 多层 OIT）再分别查询阴影。',
        },
        {
            id: 'voxel', title: '密度体素化：一份给所有光源用的发丝体积', tag: 'VOXEL',
            desc: '换一个思路：不针对某个光源生成阴影图，而是把发丝直接"抹"进一个 3D 网格——每个体素存<strong>发丝投影面积 / 体积</strong>（密度 ρ，单位 1/cm）以及加权平均切线。任何方向的透射率都可以沿该方向 ray-march 这张体积得到：多个光源、天光 / 环境光的遮挡、AO 共用同一份数据，不再随光源数量线性增加 pass。代价是分辨率：一个体素往往有 2–5 mm，比阴影图 texel 粗得多，细节被抹平，而且样本自己所在的体素里就有自己的密度，必须把 ray-march 起点推出去一段（体素 bias）。',
            formula: [
                'ρ(v) = Σ_{segments ∈ v} (ℓ · w) / V_voxel       // ℓ：段长，w：发丝宽度',
                't̄(v) = normalize(Σ ℓ · w · t)                   // 体素内平均切线（同一缕发丝几乎平行）',
                'τ(P) = κ · ∫_{b}^{∞} ρ(P + s·ω_i) · |t̄ × ω_i| ds,   T_vox = exp(−τ)   // |t̄ × ωi|：发丝在光线方向上的投影截面',
            ],
            code: {
                title: 'HLSL · CS_HairVoxelize + 透射率 ray-march', src: `
                StructuredBuffer<float4> Positions;        // 发丝顶点：xyz = 位置（cm），w = 该点发丝宽度（根 80 µm → 梢 30 µm）
                static const uint PointsPerStrand = 25;    // 本页 groom：每根 24 段 / 25 点，所有发丝首尾相接
                cbuffer HairVoxelCB { float3 VolumeMin; float VoxelSize; uint3 VolumeDims; uint NumSegments; float DensityQuant; float VoxelBias; float MaxMarch; };   // NumSegments = 点数 − 1
                RWTexture3D<uint> HairDensityU;            // 定点累加：InterlockedAdd 只支持整数
                RWTexture3D<int>  HairTangentU[3];         // 带符号定点：Σ ℓ·w·t
                Texture3D<float4> HairVolume;              // 定点结果 resolve 后：x = ρ（1/cm），yzw = Σ ℓ·w·t / V

                // 段 p → p+1 是否跨过两根发丝的接缝（p 是某根发丝的最后一个点）
                bool IsStrandEnd(uint p) { return p % PointsPerStrand == PointsPerStrand - 1; }
                // 段 p → p+1 中点处的发丝宽度：宽度沿发丝线性变化，取两端点的平均
                float StrandWidth(uint p) { return 0.5 * (Positions[p].w + Positions[p + 1].w); }
                // 世界坐标 → 体积纹理坐标 [0,1]³（体素中心在 (i + ½)/dims，与三线性过滤的约定一致）
                float3 VolumeUVW(float3 P) { return (P - VolumeMin) / (VoxelSize * float3(VolumeDims)); }

                [numthreads(64, 1, 1)]
                void CS_HairVoxelize(uint SegmentId : SV_DispatchThreadID)
                {
                    if (SegmentId >= NumSegments || IsStrandEnd(SegmentId)) return;
                    float3 A = Positions[SegmentId].xyz, B = Positions[SegmentId + 1].xyz;
                    float  Len = length(B - A), W = StrandWidth(SegmentId);
                    float3 T = (B - A) / Len;
                    uint   Steps = max(1u, (uint)ceil(Len / (0.5 * VoxelSize)));
                    for (uint s = 0; s < Steps; ++s)
                    {
                        uint3 V = (uint3)floor((lerp(A, B, (s + 0.5) / Steps) - VolumeMin) / VoxelSize);
                        float Area = Len / Steps * W;
                        InterlockedAdd(HairDensityU[V], (uint)(Area * DensityQuant));
                        [unroll] for (int a = 0; a < 3; ++a)
                            InterlockedAdd(HairTangentU[a][V], (int)(Area * T[a] * DensityQuant));
                    }
                }

                // 从 P 沿 L 穿过密度体积的期望发丝数 m = κ·∫ ρ·|t̄ × L| ds（P 已是 ray-march 起点）；Step 10 把它直接交给 LUT
                float HairCount_Voxel(float3 P, float3 L)
                {
                    float Tau = 0, Step = 0.5 * VoxelSize;
                    for (float s = 0; s < MaxMarch; s += Step)
                    {
                        float4 v = HairVolume.SampleLevel(LinearClamp, VolumeUVW(P + L * s), 0);  // x = ρ, yzw = Σ ℓ·w·t
                        float  c = dot(v.yzw, L) * rsqrt(max(dot(v.yzw, v.yzw), 1e-12));
                        Tau += v.x * sqrt(saturate(1 - c * c)) * Step;
                    }
                    return Kappa * Tau;
                }

                float HairTransmittance_Voxel(float3 P, float3 Wi)
                {
                    return exp(-HairCount_Voxel(P + Wi * (VoxelBias * VoxelSize), Wi));   // 起点朝光源推出 VoxelBias 个体素
                }`,
            },
            previews: [
                {
                    id: 'voxproj', label: 'Σ_z ρ · 体素列密度', sub: '沿 z 积分（正视）', needs: ['voxel'],
                    render: (c, S) => {
                        const g = S.vgrid, [nx, ny, nz] = g.dims, a = PL.buf(S, '_vp', nx * ny);
                        for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { let s = 0; for (let z = 0; z < nz; z++) s += g.D[(z * ny + y) * nx + x]; a[(ny - 1 - y) * nx + x] = s * g.vs || NaN; }
                        PL.draw.field(c, a, nx, ny, { map: 'viridis' });
                    },
                    probe: (x, y, S) => `体素 ${S.vgrid.dims.join('×')} · 边长 ${(S.vgrid.vs * 10).toFixed(1)} mm`,
                },
                { id: 'tvox', label: 'T · 体素 ray-march', sub: 'exp(−κ∫ρ|t̄×ωi|ds)', needs: ['voxT'], render: (c, S) => PL.draw.field(c, L3V.mask(S, S.Tvox), L3.VIEW_RES, L3.VIEW_RES, { range: [0, 1], map: 'inferno' }), probe: L3V.tProbe('T_vox', 'Tvox') },
                { id: 'evox', label: 'T_vox − T_ref', sub: '红 = 偏亮 · 蓝 = 偏暗', needs: ['voxT'], render: (c, S) => PL.draw.field(c, L3V.mask(S, S.Evox), L3.VIEW_RES, L3.VIEW_RES, { range: [-0.3, 0.3], map: 'diverge' }), probe: (x, y, S) => `RMS(体素) = ${S.rmsVox.toFixed(3)}\nRMS(DOM) = ${S.rms.dom.toFixed(3)}` },
                { id: 'shVox', label: '体素透射率着色', sub: 'E · ⟨S⟩ · T_vox', live: true, needs: ['voxT'], render: L3V.shade('vox'), probe: L3V.shadeProbe('vox') },
            ],
            notes: '拖 <strong>体素</strong>分辨率：16³ 时阴影整体糊成一团、表层发丝也被自己所在体素的密度压暗（误差图偏蓝）；64³ 接近 DOM。拖 <strong>体素 bias</strong> 到 0：每个点都被自己的体素遮挡（整体偏暗）；拖到 3：起点越过了真实遮挡物，表层附近漏光（偏红）。',
            warn: '只存标量密度时，|t̄ × ωi| 只能取各向同性平均（π/4），光线顺着发丝方向时会严重高估遮挡；这里额外存了平均切线。体素网格也要跟着头发包围盒走（或用稀疏 / 分页体素），否则大部分体素是空的。',
        },
        {
            id: 'global', title: 'Dual Scattering ①：全局多次散射（沿阴影路径的前向散射）', tag: 'GLOBAL',
            desc: '以上所有阴影方法都把发丝当成<strong>吸收体</strong>：光碰到一根发丝就没了。可浅色发丝的 TT 瓣把大部分能量<strong>向前透射</strong>——金发（真黑素 0.3）垂直入射时每穿过一根发丝，红光还剩约 70 %、蓝光约 40 %，而且只在纵向稍微展宽、方位上很快变得各向同性。Zinke et al. 2008 的 dual scattering 只沿<strong>阴影路径</strong>这一条代表性路径统计：穿过 n 根发丝后剩多少能量（T_f）、纵向展宽了多少（σ̄_f²），再假设散射光在方位上均匀分布在前半圆。于是"全局多次散射"在实现上就是一张<strong>换了内容的 deep opacity map</strong>。',
            formula: [
                'Ψ(x, ω_d, ω_i) = Ψ^G (1 + Ψ^L),     Ψ^G ≈ T_f(x, ω_d) · S_f(x, ω_d, ω_i)         // Eq. 3–4',
                'T_f = d_f · Π_{k=1}^{n} ā_f(θ_d^k),     σ̄_f^2 = Σ_{k=1}^{n} β̄_f^2(θ_d^k)         // Eq. 5、8；n = 0 时 T_f = 1',
                'ā_f(θ) = ∫_{Ω_f} S(ω_i, ω_o) dω_i,   ā_b = ∫_{Ω_b} S dω_i   // 本系列归一化：σa = 0 时 ā_f + ā_b = 1',
                '本页（DOM 给出期望发丝数 m = κ·n）：  directFraction = e^{−m},   T_f = e^{−m(1 − ā_f)},   σ̄_f^2 = m · β̄_f^2',
            ],
            code: {
                title: 'HLSL · 由 DOM 读出的发丝数得到全局多次散射量', src: `
                Texture1D<float4> DS_ForwardLUT;     // .rgb = ā_f(θ)，.a = 亮度加权 β̄_f²(θ)；按 |θ| 归一化寻址

                struct FHairGlobalScattering { float DirectFraction; float3 Tf; float3 SigmaF2; };

                FHairGlobalScattering EvaluateGlobalScattering(float HairCount /* Ω，来自 DOM */, float SinThetaI)
                {
                    FHairGlobalScattering G;
                    float  m  = Kappa * HairCount;                         // 期望穿过的发丝数
                    float4 F  = DS_ForwardLUT.SampleLevel(LinearClamp, abs(asin(SinThetaI)) * (2 / PI), 0);
                    G.DirectFraction = exp(-m);                            // 一根都没碰到的比例（Poisson）
                    G.Tf      = exp(-m * (1 - F.rgb));                     // E[ā_f^N]，N ~ Poisson(m)
                    G.SigmaF2 = m * F.a;                                   // 每次前向散射叠加一次纵向方差
                    return G;
                }`,
            },
            previews: [
                {
                    id: 'afplot', label: 'ā_f(θ) / ā_b(θ)', sub: '实线 = 前向，虚线 = 后向（RGB）', size: [360, 180],
                    render: (c, S) => {
                        const t = S.ds, th = t.af.map((_, j) => (j + 0.5) / t.NT * 90), col = ['#e06c75', '#5ad07a', '#6ab0e0'];
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, 1], xLabel: '|θ| (°)',
                            series: [0, 1, 2].map((k) => ({ data: t.af.map((v) => v[k]), x: th, color: col[k], label: k === 0 ? 'ā_f' : '', width: 2 }))
                                .concat([0, 1, 2].map((k) => ({ data: t.ab.map((v) => v[k]), x: th, color: col[k], label: k === 0 ? 'ā_b' : '', dash: [4, 3] }))),
                        });
                    },
                },
                {
                    id: 'tfplot', label: 'T vs 穿过的发丝数 m', sub: 'θ = 光线在发丝上的平均倾角', size: [360, 180],
                    render: (c, S) => {
                        const t = S.ds, j = DS.row(t, S.thMean || 0), col = ['#e06c75', '#5ad07a', '#6ab0e0'];
                        PL.draw.plot(c, {
                            x: [0, 12], y: [0, 1], xLabel: 'm = κ · n',
                            series: [{ fn: (m) => Math.exp(-m), color: '#e0e0e0', label: 'directFraction = e^−m（吸收体阴影）', width: 2 }]
                                .concat([0, 1, 2].map((k) => ({ fn: (m) => Math.exp(-m * (1 - t.af[j][k])), color: col[k], label: k === 0 ? 'T_f（RGB）' : '', width: 2 }))),
                        });
                    },
                },
                {
                    id: 'tfmap', label: 'd_f · (T_f − directFraction)', sub: '前向散射送达的能量（RGB）',
                    render: (c, S) => { const V = L3.VIEW_RES; PL.draw.pixels(c, V, V, (x, y, out) => { const i = y * V + x; if (S.view.id[i] < 0) { out[0] = out[1] = out[2] = 0; return out; } for (let k = 0; k < 3; k++) out[k] = HAIR.toSrgb(Math.max(0, S.df * (S.Tf[3 * i + k] - S.Tdom[i]))); return out; }); },
                    probe: (x, y, S) => { const i = y * L3.VIEW_RES + x; if (S.view.id[i] < 0) return '背景'; return `m = ${S.mCount[i].toFixed(2)}\ndirect = ${S.Tdom[i].toFixed(3)}\nT_f = (${[0, 1, 2].map((k) => S.Tf[3 * i + k].toFixed(3)).join(', ')})`; },
                },
            ],
            notes: '左上：真黑素 0.3（金发）时，垂直入射的红光每穿过一根发丝还剩约 70 %，蓝光只剩约 40 %——这就是浅色头发内部偏暖的来源。右上：白线是所有"吸收体"阴影方法给出的直射比例，彩色线是还能以前向散射形式抵达的能量；把<strong>真黑素</strong>拉到 3 以上，两者几乎重合，多次散射可以忽略。',
            warn: '与原文的差异：① 原文 T_f 在射线上逐次累乘真实的 ā_f(θ_d^k)；这里用"各发丝倾角相同"（disciplined cluster）假设，并对 Poisson 分布的发丝数取期望 E[ā_f^N] = e^{−m(1−ā_f)}（原文 GPU 版逐层存 T_f、σ̄_f 和 directFraction 共 7 个值）。② ā_f 按本系列的 ∫S dωi 归一化，等价于原文 Eq. 6 的定义，但不含 1/π 与 cosθ 的写法差异。③ β̄_f² 取 pbrt-v3 各瓣方差 v_p 的能量加权平均。',
        },
        {
            id: 'local', title: 'Dual Scattering ②：局部多次散射（着色点附近的回散射）', tag: 'LOCAL',
            desc: '全局项只统计"一路向前"的路径；而 x 附近的光还可以先穿过几根发丝、被<strong>向后散射</strong>、再穿回来——这部分至少含一次后向散射，空间上很平滑，于是原文把它做成一个<strong>材质属性</strong>：发丝簇的回散射函数 f_back。它只依赖单根发丝的四个统计量 ā_f、ā_b、ᾱ（平均纵向偏移）、β̄²（平均纵向方差），对 1 次与 3 次后向散射的所有路径求和得到闭式：平均回散射衰减 Ā_b、平均偏移 Δ̄_b、平均展宽 σ̄_b。浅色头发的整体"亮度与颜色"很大一部分来自这一项——单次散射的 R / TRT 只占后向能量的一小部分。',
            formula: [
                'Ψ^L f_s ≈ d_b · f_back,     Ā_b = Ā_1 + Ā_3 = ā_b ā_f^2 / (1 − ā_f^2) + ā_b^3 ā_f^2 / (1 − ā_f^2)^3     // Eq. 9、11、13、14',
                'Δ̄_b ≈ ᾱ_b (1 − 2ā_b^2/(1 − ā_f^2)^2) + ᾱ_f (2(1 − ā_f^2)^2 + 4ā_f^2 ā_b^2)/(1 − ā_f^2)^3              // Eq. 16',
                'σ̄_b ≈ (1 + 0.7ā_f^2) · (ā_b √(2β̄_f^2 + β̄_b^2) + ā_b^3 √(2β̄_f^2 + 3β̄_b^2)) / (ā_b + ā_b^3 (2β̄_f + 3β̄_b))  // Eq. 17',
                '本页（能量归一化）：S_back = d_b · Ā_b(θ_d) · (1/π)·[|φ| < π/2] · g(θ_i + θ_o − Δ̄_b, σ̄_b^2) / cos θ_i   // ∫ S_back dω_i = d_b·Ā_b',
            ],
            code: {
                title: 'HLSL · 局部多次散射（回散射）项', src: `
                Texture1D<float4> DS_BackLUT;      // .rgb = Ā_b(θ)，.a = 亮度加权 Δ̄_b(θ)
                Texture1D<float4> DS_BackSpread;   // .rgb = σ̄_b²(θ)

                float Gauss(float x, float Var) { return exp(-x * x / (2 * Var)) * rsqrt(2 * PI * Var); }

                // 按本系列约定（S 已含 cosθi），积分到能量 d_b·Ā_b；SigmaF2 > 0 时是"散射光再回散射"的版本
                float3 HairLocalScattering(float ThetaI, float ThetaO, float Phi, float3 SigmaF2, float AzimuthWeight)
                {
                    float  ThetaD = 0.5 * abs(ThetaO - ThetaI);
                    float4 Back   = DS_BackLUT.SampleLevel(LinearClamp, ThetaD * (2 / PI), 0);
                    float3 SigB2  = DS_BackSpread.SampleLevel(LinearClamp, ThetaD * (2 / PI), 0).rgb;
                    float3 G;
                    [unroll] for (int c = 0; c < 3; ++c)
                        G[c] = Gauss(ThetaI + ThetaO - Back.a, SigB2[c] + SigmaF2[c]);
                    return DensityFactor * Back.rgb * AzimuthWeight * G / max(cos(ThetaI), 0.15);
                }
                // 直射光：AzimuthWeight = abs(Phi) < PI/2 ? 1/PI : 0
                // 前向散射光：入射方位已在前半圆均匀分布，与后半圆盒函数卷积 → (PI − |Phi|) / PI²`,
            },
            previews: [
                {
                    id: 'abplot', label: 'Ā_b(θ) vs ā_b(θ)', sub: '实线 = 局部多次回散射，虚线 = 单根发丝后向（RGB）', size: [360, 180],
                    render: (c, S) => {
                        const t = S.ds, th = t.af.map((_, j) => (j + 0.5) / t.NT * 90), col = ['#e06c75', '#5ad07a', '#6ab0e0'];
                        PL.draw.plot(c, {
                            x: [0, 90], xLabel: '|θ_d| (°)',
                            series: [0, 1, 2].map((k) => ({ data: t.Ab.map((v) => v[k]), x: th, color: col[k], label: k === 0 ? 'Ā_b' : '', width: 2 }))
                                .concat([0, 1, 2].map((k) => ({ data: t.ab.map((v) => v[k]), x: th, color: col[k], label: k === 0 ? 'ā_b' : '', dash: [4, 3] }))),
                        });
                    },
                },
                {
                    id: 'sbplot', label: 'Δ̄_b、σ̄_b（亮度加权）', sub: '纵向偏移与展宽 · 度', size: [360, 180],
                    render: (c, S) => {
                        const t = S.ds, th = t.af.map((_, j) => (j + 0.5) / t.NT * 90), d = 180 / Math.PI;
                        PL.draw.plot(c, {
                            x: [0, 90], xLabel: '|θ_d| (°)', yLabel: '°',
                            series: [
                                { data: t.Db.map((v) => HAIR.lum(v) * d), x: th, color: '#e0a060', label: 'Δ̄_b', width: 2 },
                                { data: t.sb2.map((v) => Math.sqrt(HAIR.lum(v)) * d), x: th, color: '#c678dd', label: 'σ̄_b', width: 2 },
                                { data: t.bf2.map((v) => Math.sqrt(HAIR.lum(v)) * d), x: th, color: '#56b6c2', label: 'β̄_f', dash: [4, 3] },
                            ],
                        });
                    },
                },
                { id: 'shLocal', label: '局部多次散射分量', sub: 'directFraction·S_back + 散射光的 S_back · dual 曝光', render: L3V.shade('local', 'dual'), probe: L3V.shadeProbe('local') },
            ],
            notes: '左：金发时 Ā_b 与单根发丝的后向能量 ā_b 同一量级——局部多次散射大约把后向散射"翻倍"，而且偏红（ā_f 越大，几何级数 1/(1 − ā_f²) 越大）。把<strong>真黑素</strong>拉到 2 以上，Ā_b 迅速衰减到接近 0。右图分量本身很柔和（σ̄_b 约 30°），正好是单次散射缺失的那层"底色"。',
            warn: '归一化差异：原文 Fig. 5 写作 f_back = 2Ā_b · g(…) / (π cos²θ)（Marschner 式 BCSDF，纵向高斯按半角定义）；本页按系列约定把它重新归一化为"积分能量 = d_b·Ā_b"的 S 形式。另外原文在散射光分支用 π·d_b·f_back；本页改用方位卷积（前半圆盒 ⊗ 后半圆盒 = 三角形 (π − |φ|)/π²），能量同样是 d_b·Ā_b。σ̄_b 公式（Eq. 17）分母量纲不统一，是原文的数值拟合，照抄。',
        },
        {
            id: 'compare', title: '单次散射 vs Dual Scattering：浅色头发为什么"不对"', tag: 'COMPARE',
            desc: '把两部分拼起来（原文 Fig. 5）：没被挡住的那部分光（directFraction）照常做单次散射，再加上局部回散射；被挡住的部分不再是"黑"，而是以 T_f 的能量、展宽后的纵向瓣和各向同性方位（N^G）照到 x。只用单次散射 + 吸收体阴影时，浅色头发内部能量大量缺失——画面发灰、发暗、颜色偏冷，美术只能加一个 ad-hoc 的 diffuse 项去补（原文 Fig. 1 正是这个对比）。',
            formula: [
                'F = directFraction · (S_direct + d_b S_back) + (T_f − directFraction) · d_f · (S_scatter + d_b S_back^G)    // Fig. 5（本页约定，L_o = E·F）',
                'S_scatter = Σ_p M_p(θ_i, θ_o; v_p + σ̄_f^2) · N_p^G(θ_o, φ),     N^G(φ) = (1/π) ∫_{−π/2}^{π/2} N(φ + φ′) dφ′     // Eq. 23、25',
                '单次散射对照：F_single = directFraction · S_direct',
            ],
            code: {
                title: 'HLSL · 完整 dual scattering 着色（对应原文 Fig. 5）', src: `
                Texture2DArray<float4> DS_NG;       // N^G_p(|θo|, φ)·A_p，slice = 瓣 p（CS_BakeDualScatteringLUT 的输出，24 × 36 × 4）
                SamplerState           NGSampler;   // U（|θo|）CLAMP，V（φ）WRAP：φ 是周期的，TT 的峰正好落在 ±π 的接缝上

                // N^G 表的纹理坐标：u = |θo| / (π/2)，v = (φ + π) / 2π；表格按单元中心烘焙（θ_j = (j + ½)·(π/2)/24，φ_q = −π + (q + ½)·2π/36）
                float2 NGUV(float ThetaO, float Phi) { return float2(abs(ThetaO) * (2 / PI), (Phi + PI) * (0.5 / PI)); }

                // 纵向瓣 M_p：与 HairBSDF_Chiang 相同的角质层倾角平移（R −2α′、TT +α′、TRT +4α′，p = 3 不平移），方差按 RGB 分别给出（v_p + σ̄_f²）
                float3 HairM(int p, float SinThetaI, float SinThetaO, float3 Var)
                {
                    float cosI = SafeSqrt(1 - SinThetaI * SinThetaI), cosO = SafeSqrt(1 - SinThetaO * SinThetaO);
                    float sinOp = SinThetaO, cosOp = cosO;
                    if (p == 0)      { sinOp = SinThetaO * Hair.cos2kAlpha[1] - cosO * Hair.sin2kAlpha[1]; cosOp = cosO * Hair.cos2kAlpha[1] + SinThetaO * Hair.sin2kAlpha[1]; }
                    else if (p == 1) { sinOp = SinThetaO * Hair.cos2kAlpha[0] + cosO * Hair.sin2kAlpha[0]; cosOp = cosO * Hair.cos2kAlpha[0] - SinThetaO * Hair.sin2kAlpha[0]; }
                    else if (p == 2) { sinOp = SinThetaO * Hair.cos2kAlpha[2] + cosO * Hair.sin2kAlpha[2]; cosOp = cosO * Hair.cos2kAlpha[2] - SinThetaO * Hair.sin2kAlpha[2]; }
                    return float3(Mp(cosI, abs(cosOp), SinThetaI, sinOp, Var.r),
                                  Mp(cosI, abs(cosOp), SinThetaI, sinOp, Var.g),
                                  Mp(cosI, abs(cosOp), SinThetaI, sinOp, Var.b));
                }

                float3 HairDualScattering(FHairSample s, float3 Wi, float3 Wo, float HairCount)
                {
                    FHairGlobalScattering G = EvaluateGlobalScattering(HairCount, s.SinThetaI);
                    float  h        = 2 * InterleavedGradientNoise(s.PixelPos + 5.588238 * (FrameIndex % 64)) - 1;
                    float3 SDirect  = HairBSDF_S(Wi, Wo, s.Tangent, h);                         // R + TT + TRT + 残余
                    float  AzBack   = abs(s.Phi) < 0.5 * PI ? 1 / PI : 0;
                    float  AzTri    = max(PI - abs(s.Phi), 0) / (PI * PI);
                    float3 BackD    = HairLocalScattering(s.ThetaI, s.ThetaO, s.Phi, 0, AzBack);
                    float3 BackG    = HairLocalScattering(s.ThetaI, s.ThetaO, s.Phi, G.SigmaF2, AzTri);

                    float3 SScatter = 0;                                                          // 纵向方差 + σ̄_f²，方位换 N^G
                    [unroll] for (int p = 0; p < 4; ++p)
                        SScatter += HairM(p, s.SinThetaI, s.SinThetaO, Hair.v[p] + G.SigmaF2)
                                  * DS_NG.SampleLevel(NGSampler, float3(NGUV(s.ThetaO, s.Phi), p), 0).rgb;

                    float3 FDirect  = G.DirectFraction * (SDirect + BackD);
                    float3 FScatter = (G.Tf - G.DirectFraction) * DensityFactor * (SScatter + BackG);
                    return FDirect + FScatter;                                                    // L_o = E · F
                }`,
            },
            previews: [
                { id: 'shSingle', label: '单次散射 + DOM 阴影', sub: 'directFraction · S', live: true, render: L3V.shade('dom', 'dual'), probe: L3V.shadeProbe('dom') },
                { id: 'shDual', label: 'Dual scattering', sub: '+ 全局 + 局部多次散射', live: true, render: L3V.shade('dual', 'dual'), probe: L3V.shadeProbe('dual') },
                { id: 'shMS', label: '仅多次散射分量', sub: 'F − directFraction·S', render: L3V.shade('ms', 'dual'), probe: L3V.shadeProbe('ms') },
                {
                    id: 'melplot', label: '单次散射保留的能量比例', sub: '深度 m 处的回散射能量：单次 / dual（亮度）', size: [360, 180], needs: ['melSweep'],
                    render: (c, S) => {
                        const w = S.melSw, col = ['#e0e0e0', '#e0a060', '#c678dd', '#6ab0e0'];
                        PL.draw.plot(c, {
                            x: [0, 4], y: [0, 1.02], xLabel: '真黑素浓度', marks: [{ x: S.ce, label: '当前', color: '#5ad07a' }],
                            series: w.ms.map((m, k) => ({ data: w.ratio[k], x: w.ce, color: col[k], label: `m = ${m}`, width: 2 })),
                        });
                    },
                },
            ],
            notes: '三张图同曝光（按 dual 结果定曝光）。金发（真黑素 0.3）里多次散射分量（右）铺满了整片发帘的内部和下半部分，补回的正是左图里"被阴影吃掉"的暖色能量；拖<strong>真黑素</strong>到 3 以上，三张图几乎一样——深色头发每穿过一根发丝就吸收掉大部分能量，单次散射 + 吸收体阴影已经足够。右下曲线：在 m = 2 根发丝深处，金发（真黑素 0.3）的单次散射只保留了 dual 能量的约 30 %，真黑素 1 时约 64 %，2 以上才超过 85 %；拖<strong>d_f</strong> 改变多次散射的整体权重。',
            warn: '这里的 directFraction 就是 DOM 透射率 exp(−κΩ)，<strong>不要再额外乘一次阴影</strong>——dual scattering 的全局项本身就是"阴影"。另外 S_direct 需要 h 的随机采样 + 时域累积，而 S_scatter / S_back 很平滑，可以每帧确定性地求值。',
        },
        {
            id: 'engine', title: '引擎里的实时近似：hair count → LUT', tag: 'ENGINE',
            desc: '生产引擎几乎都收敛到同一个结构：<strong>先估计光路上的发丝数</strong>（DOM 或密度体素），<strong>再查一张按角度 / 吸收（/ 粗糙度）预计算的表</strong>得到衰减与散射。以下只列有公开出处的事实（出处见文末）：' +
                '<ul style="margin:6px 0 0 18px;line-height:1.7">' +
                '<li><strong>UE5 Groom · Deep Shadow</strong>：Deep Opacity Map，Front Depth + 4 层、每层存发丝密度；按 hair cluster × 光源分别生成（pass <code>HairStrandsDeepShadowFrontDepth</code> / <code>HairStrandsDeepShadowLayers</code>）。<code>r.HairStrands.DeepShadow.Resolution</code> 默认 2048，<code>DepthBiasScale</code>（默认 0.05）用于透射率计算，<code>DensityScale</code> 用来"补偿发丝数量不足"——正是本页的 κ；<code>KernelType</code> 可选 linear / PCF2x2 / PCF6x4 / PCSS / PCF6x6Accurate。</li>' +
                '<li><strong>UE5 · 体素化</strong>：稀疏（virtual，两级 page）密度体素，<code>VoxelWorldSize</code> 以 cm 计；按 Epic 的 Hair &amp; Fur 白皮书，它是发丝阴影、AO 与环境光照的默认机制，关闭 Deep Shadow 时阴影走 voxel tracing；<code>r.HairStrands.Voxelization.DensityScale</code>（默认 2）可分别覆盖 Shadow / Transmittance / Environment / AO，<code>DepthBiasScale.Shadow</code> 把 ray 起点朝光源偏移——即本页的体素 bias。</li>' +
                '<li><strong>UE5 · 多次散射</strong>：<code>HairStrandsLUT</code> 的 cvar 描述了一张按入射角 / 粗糙度 / 吸收切片的 hair LUT，并有 <code>DualScatteringRoughness</code> 覆盖项；透射数据含 <code>OpaqueVisibility / LocalScattering / GlobalScattering</code>，社区帖子引用的着色代码按 <code>GlobalScattering·(Fs + LocalScattering)·OpaqueVisibility</code> 组合（与 Fig. 5 的结构不同，Epic 在该帖中也表示部分偏离的原因已难以考证）。旧的 <code>KajiyaKayDiffuseAttenuation</code>（"Hack approximation for multiple scattering"）只作用于非 strands 几何。</li>' +
                '<li><strong>UE4 · Karis 2016</strong>：多次散射 = wrapped Lambert + 按"由指数阴影值反推的光程"计算的吸收，作者自称 "a giant artistic hack"；shadow map 用指数衰减代替硬比较做 PCF。</li>' +
                '<li><strong>Frostbite · Tafuri 2019</strong>：Dual Scattering [Zinke08]；用 4 层 DOM 估计光路上的 hair count（同时用于阴影），散射造成的衰减预先计算、平均后存进 LUT；退路是常数发丝密度 + Beer–Lambert。</li></ul>',
            formula: [
                'n_light = DensityScale · Ω_DOM(z)   或   ∫ ρ_voxel ds        // 光路上的期望发丝数（本页 m = κ·Ω）',
                '(T_f, σ̄_f^2, Ā_b, Δ̄_b, σ̄_b^2) = LUT(|θ|, σ_a, β)             // 本页：Table 1 的 1D 表按吸收重烘焙；引擎：一张多维 LUT',
            ],
            code: {
                title: 'HLSL · hair count + LUT 的通用写法（本页示意，非引擎源码）', src: `
                Texture3D<float4> HairScatterLUT;    // (|θ|·2/π, 粗糙度, 吸收) → (ā_f, Ā_b, β̄_f², σ̄_b²)
                SamplerState      LinearClamp;

                float4 SampleHairLUT(float AbsTheta, float Roughness, float Absorption)
                {
                    return HairScatterLUT.SampleLevel(LinearClamp, float3(AbsTheta * (2 / PI), Roughness, Absorption), 0);
                }

                // 每个光源每像素：hair count 来自 DOM（有 deep shadow）或体素 ray-march（否则）
                float HairCountToLight(FHairSample s, FLight Light)
                {
                    return Light.bHasDeepShadow ? DensityScale * DomHairCount(s.LightTexel, s.LightZ)
                                                : HairCount_Voxel(s.WorldPos + Light.L * VoxelBias * VoxelSize, Light.L);
                }`,
            },
            previews: [
                { id: 'lutAf', label: 'LUT · ā_f(|θ|, 真黑素)', sub: '横轴：真黑素 0→4（平方间隔）· 纵轴：|θ| 90°（上）→ 0°', needs: ['lut2d'], size: [180, 180], render: (c, S) => PL.draw.field(c, S.lut.af, S.lut.NC, S.lut.NT, { range: [0, 1], map: 'viridis' }), probe: (x, y, S) => { const L = S.lut, cx = Math.min(L.NC - 1, Math.floor(x / 180 * L.NC)), cy = Math.min(L.NT - 1, Math.floor(y / 180 * L.NT)); return '真黑素 = ' + L.ces[cx].toFixed(2) + '\n|θ| ≈ ' + ((L.NT - 0.5 - cy) / L.NT * 90).toFixed(0) + '°\naf = ' + L.af[cy * L.NC + cx].toFixed(3); } },
                { id: 'lutAb', label: 'LUT · Ā_b(|θ|, 真黑素)', sub: '局部多次回散射能量', needs: ['lut2d'], size: [180, 180], render: (c, S) => PL.draw.field(c, S.lut.Ab, S.lut.NC, S.lut.NT, { map: 'inferno' }), probe: (x, y, S) => { const L = S.lut, cx = Math.min(L.NC - 1, Math.floor(x / 180 * L.NC)), cy = Math.min(L.NT - 1, Math.floor(y / 180 * L.NT)); return '真黑素 = ' + L.ces[cx].toFixed(2) + '\n|θ| ≈ ' + ((L.NT - 0.5 - cy) / L.NT * 90).toFixed(0) + '°\nAb = ' + L.Ab[cy * L.NC + cx].toFixed(4); } },
                { id: 'hcount', label: 'm = κ·Ω · 着色点的 hair count', sub: '引擎传给 LUT 的量', render: (c, S) => PL.draw.field(c, L3V.mask(S, S.mCount), L3.VIEW_RES, L3.VIEW_RES, { map: 'viridis', range: [0, 6] }), probe: (x, y, S) => { const i = y * L3.VIEW_RES + x; return S.view.id[i] < 0 ? '背景' : 'm = ' + S.mCount[i].toFixed(2) + ' 根'; } },
            ],
            notes: '两张 LUT 是本页 fiber 模型（pbrt-v3 / Chiang 2016，β_m = β_n = 0.3）在不同真黑素浓度下重新烘焙的真实结果：ā_f 随真黑素迅速下降，Ā_b 只在浅色区域显著——这就是引擎可以把"吸收"当成 LUT 一个轴的原因。拖<strong>褐黑素</strong>会重烘焙整张表。右图是每个着色点读到的发丝数，它乘进 LUT 的结果才是最终的透射与散射。',
            warn: 'UE 部分只陈述能从官方文档、cvar 帮助字符串、Epic 白皮书与公开讲座 / 论坛帖子中核实的内容；LUT 的具体存储内容和 shader 细节随版本变化，本页的 HLSL 是按论文结构写的示意代码。',
        },
        {
            id: 'prod', title: '教学版 → 生产版，以及常见坑', tag: 'PROD',
            desc: '本页为了能在浏览器里逐 texel 对照，把所有东西都做在 CPU 上的小分辨率单光源上。落到 GPU 引擎时的改动：' +
                '<ol style="margin:6px 0 0 18px;line-height:1.7">' +
                '<li><strong>阴影图</strong>：128² → 每个 hair cluster × 每个光源一张约 2k 的 DOM，按发丝包围盒紧密拟合；Front Depth + 3–4 层装进一张 RGBA16F；层厚前薄后厚。</li>' +
                '<li><strong>精确参考 / DSM</strong>：逐 texel 片元列表只用于离线或调试（per-pixel linked list 的显存不可控）。</li>' +
                '<li><strong>体素</strong>：32³ 稠密网格 → 稀疏 / 分页体素，跟随发丝包围盒；多光源、天光、AO 共用；在没有 deep shadow 的光源上用它做阴影。</li>' +
                '<li><strong>多次散射</strong>：每像素十几次 M_p + N^G 双线性 → 一张多维 LUT（角度 × 粗糙度 × 吸收），一次采样拿到 T_f、σ̄_f、Ā_b…</li>' +
                '<li><strong>时域</strong>：h 的随机采样、shadow 采样的抖动都交给 TAA / 时域累积（本页的 HISTORY stage）；每像素多样本（visibility buffer）分别查询阴影。</li></ol>',
            formula: [
                '漏光：bias 太大 / 体素 bias 太大 / 片元池溢出被丢        ↔        自遮挡变暗：bias 太小 / 体素 bias = 0',
                '层跳变：z_0 由最近的亚像素发丝决定，光源或发丝一动 z_0 就跳 → 所有层一起前后移动',
            ],
            previews: [
                {
                    id: 'biasplot', label: 'RMS 误差 vs depth bias', sub: '参考不加 bias（只排除被着色发丝自身）', size: [360, 180], needs: ['biasSweep'],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, 1], y: [0, 0.5], xLabel: 'bias (cm)', yLabel: 'RMS', marks: [{ x: S.bias, label: '当前', color: '#5ad07a' }],
                        series: [
                            { data: S.biasSw.sm, x: S.biasSw.b, color: '#e06c75', label: '普通 SM', width: 2 },
                            { data: S.biasSw.dsm, x: S.biasSw.b, color: '#c678dd', label: 'DSM', width: 2 },
                            { data: S.biasSw.dom, x: S.biasSw.b, color: '#e0a060', label: 'DOM', width: 2 },
                        ],
                    }),
                },
                {
                    id: 'z0jump', label: '|Δz₀| · 前向深度跳变', sub: '与 4 邻 texel 的最大深度差（cm）',
                    render: (c, S) => {
                        const N = S.smRes, D = S.lr.depth, a = PL.buf(S, '_zj', N * N);
                        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
                            const i = y * N + x;
                            if (!isFinite(D[i])) { a[i] = NaN; continue; }
                            let m = 0;
                            [[1, 0], [-1, 0], [0, 1], [0, -1]].forEach(([dx, dy]) => { const X = x + dx, Y = y + dy; if (X >= 0 && Y >= 0 && X < N && Y < N && isFinite(D[Y * N + X])) m = Math.max(m, Math.abs(D[Y * N + X] - D[i])); });
                            a[i] = m;
                        }
                        PL.draw.field(c, a, N, N, { map: 'inferno', range: [0, 2] });
                    },
                    probe: L3V.lightProbe,
                },
            ],
            notes: '左（默认参数实测）：DOM 在 bias = 0 时最好，随 bias 单调变差——第一层内的线性插值本身就把被着色发丝自己的贡献摊薄了，额外的 bias 只会漏光；DSM 需要约 0.05 cm 的 bias 才能跳过自身发丝那一级台阶；普通 shadow map 在 0.2 cm 附近出现一个"最优点"，那只是靠大面积漏光碰巧接近半透明阴影，误差仍是 DOM 的两倍多。右：亮的 texel 是前向深度在相邻 texel 间跳了 1 cm 以上的地方——DOM 的所有层都挂在 z₀ 上，这些位置在动画里最容易闪烁（原文也报告了线段光栅化走样带来的轻微闪烁）。',
            warn: ['缓解层跳变：提高阴影图分辨率、对 z₀ 做小范围 min / 模糊、对阴影查询做 PCF（UE 提供多种 KernelType）、配合 TAA；不要用单个最近深度做二值判断。', '密度倍率（κ / DensityScale）同时改变阴影与多次散射的发丝数：两者应该读同一个 hair count，不要分别 hack 阴影浓度和"多次散射亮度"。'],
        },
    ],

    graph: {
        title: 'Hair Shadow & Multiple Scattering · Pass Graph',
        intro: '每个节点 = 一次 Draw / Dispatch；pin = 资源名 · 格式，同名资源自动连线。光源视角两遍（Front Depth + Opacity Layers）每个光源各做一次；体素化与 LUT 与光源无关、可被所有光源 / 环境光共享；着色 pass 读取 hair count，经 dual scattering 得到最终颜色，随机 h 由 HISTORY 累积。',
        common: {
            title: 'Common.hlsl · 系列共享 + 本页', open: false, blocks: [
                { label: '━ 系列共享 ━', code: HAIR.HLSL_COMMON },
                { label: '━ 系列共享 · 毛发 BSDF（HLSL 参考实现）━', code: HAIR.HLSL_BSDF },
                { label: '━ 本页约定 ━', code: `
                    // hair count n(z) = Σ 片元覆盖率；κ = 每根渲染发丝代表的真实发丝数（≈ UE DensityScale）
                    // T(z) = exp(-κ·n(z))；directFraction = T；T_f = exp(-κ·n·(1 - ā_f))；σ̄_f² = κ·n·β̄_f²
                    static const float DensityFactor = 0.7;      // d_f = d_b（Zinke et al. 2008）
                    float Gauss(float x, float Var) { return exp(-x * x / (2 * Var)) * rsqrt(2 * PI * Var); }

                    // 单根发丝参数，每个材质在 CPU 上算一次：σa 来自两种黑色素，βm = βn = 0.3，α = 2°（与本页 JS 的 HAIR.bsdf 相同）
                    cbuffer HairMaterial { HairParams Hair; };   // Hair = MakeHairParams(SigmaAFromConcentration(真黑素, 褐黑素), 0.3, 0.3, 2)

                    // 着色样本：可见性 pass 的输出 + 着色前在纤维坐标里算好的角度（Step 05 / 09 / 10 共用）
                    struct FHairSample {
                        float2 PixelPos;   float3 WorldPos;   float3 Tangent;     // 像素中心、世界位置（cm）、切线（根 → 梢）
                        float  SinThetaI, SinThetaO, ThetaI, ThetaO, Phi;         // 纤维坐标下 ωi / ωo 的纵向角，φ = φi − φo
                        int2   LightTexel; float LightZ;                          // 光源空间 texel 与深度（DOM 查询）
                    };

                    // 世界空间入口：ωi、ωo 转到纤维坐标（x = T，y 在法平面内朝向观察者）后调用系列共享的 Chiang BSDF
                    float3 HairBSDF_S(float3 Wi, float3 Wo, float3 T, float h)
                    {
                        return HairBSDF_Chiang(ToFiberFrame(Wi, T, Wo), ToFiberFrame(Wo, T, Wo), h, Hair);
                    }` },
            ],
        },
        types: {
            strand: { color: '#c678dd', label: 'strand · 发丝顶点' },
            depth: { color: '#6ab0e0', label: 'depth · 光源空间深度' },
            opacity: { color: '#e0a060', label: 'opacity · 累计 hair count Ω' },
            volume: { color: '#56b6c2', label: 'volume · 3D 密度' },
            lut: { color: '#d19a66', label: 'lut · 预计算表' },
            gbuf: { color: '#98c379', label: 'gbuffer · 可见发丝样本' },
            scalar: { color: '#abb2bf', label: 'scalar · 每像素标量' },
            color: { color: '#5ad07a', label: 'color · HDR 颜色' },
        },
        nodes: [
            {
                id: 'groomNode', title: 'HairStrands', tag: 'INPUT', category: 'system', lifetime: 'resident', desc: 'Groom 顶点 + 宽度',
                outputs: [{ name: 'StrandVertices', format: 'float4 · pos + width', type: 'strand' }],
                detail: { desc: '发丝顶点缓冲（本页：HAIR.makeGroom，16 根 guide 插值出 750–3000 根，每根 24 段）。', preview: null },
            },
            {
                id: 'depthPass', title: 'PS_HairShadowDepth', tag: 'PS', lifetime: 'param', step: 'dom', desc: 'z₀ = 最近发丝（Front Depth）',
                dispatch: { kind: 'PS', target: (S) => S.smRes + '² R32F', count: (S) => S.count + ' strands × 24 segs', note: '每个光源一次（UE：HairStrandsDeepShadowFrontDepth）' },
                inputs: [{ name: 'StrandVertices', format: 'float4', type: 'strand' }],
                outputs: [{ name: 'HairShadowDepth', format: 'R32F', type: 'depth' }],
                detail: { formula: 0, code: { title: 'HLSL · 线段光栅化只写深度', src: `
                    // 发丝作为 line / 屏幕对齐四边形光栅化；深度测试 LESS，不写颜色
                    float PS_HairShadowDepth(float4 SvPos : SV_Position) : SV_Depth { return SvPos.z; }` }, preview: 'z0' },
            },
            {
                id: 'layerPass', title: 'PS_HairOpacityLayers', tag: 'PS', lifetime: 'param', step: 'dom', desc: 'ONE+ONE · Ω_k += c',
                dispatch: { kind: 'PS', target: (S) => S.smRes + '² RGBA16F', note: '加法混合；R = z₀，GBA = 3 层累计（UE：HairStrandsDeepShadowLayers）' },
                inputs: [{ name: 'StrandVertices', format: 'float4', type: 'strand' }, { name: 'HairShadowDepth', format: 'R32F', type: 'depth' }],
                outputs: [{ name: 'DeepOpacityMap', format: 'RGBA16F', type: 'opacity' }],
                detail: { formula: 1, code: 0, preview: 'domlayers' },
            },
            {
                id: 'voxelize', title: 'CS_HairVoxelize', tag: 'CS', lifetime: 'param', step: 'voxel', desc: 'ρ += ℓ·w / V（原子加）',
                dispatch: { kind: 'CS', threads: [64, 1, 1], groups: (S) => [Math.ceil(S.count * 25 / 64), 1, 1], idRange: 'ID.x ∈ [0, 点数 − 1)：段 p → p+1，跨发丝接缝的跳过', note: '与光源无关，所有光源 / 天光 / AO 共享' },
                inputs: [{ name: 'StrandVertices', format: 'float4', type: 'strand' }],
                outputs: [{ name: 'HairDensityVolume', format: 'R32_UINT ×4 → RGBA16F', type: 'volume' }],
                detail: { preview: 'voxproj' },
            },
            {
                id: 'lutBake', title: 'CS_BakeDualScatteringLUT', tag: 'CS', category: 'bake', lifetime: 'param', step: 'global', desc: 'ā_f ā_b Ā_b Δ̄_b σ̄_b² N^G',
                dispatch: { kind: 'CS', threads: [8, 8, 1], groups: [3, 5, 4], idRange: 'ID.x = |θ| 24 档，ID.y = φ 36 档，ID.z = 瓣 p', note: '吸收（黑素）改变时重烘焙；对 h 做 48 点积分' },
                inputs: [{ name: 'FiberParams', format: 'cbuffer · σa, β, α, η', type: 'scalar', external: true }],
                outputs: [{ name: 'DualScatteringLUT', format: 'R16G16B16A16F · 1D ×3 + 2D array', type: 'lut' }],
                detail: {
                    desc: '原文 Table 1 的全部表：Ā_b(θ)、Δ̄_b(θ)、σ̄_b²(θ)（局部多次散射）与 N^G(θ, φ)（散射光的方位函数），以及它们依赖的 ā_f、ā_b、ᾱ、β̄²。',
                    formula: 'ā_f(θ) = Σ_p ⟨A_p(θ, h) · ∫_{|φ|>π/2} N_p dφ⟩_h,   N^G_p(θ, φ) = ⟨A_p · (1/π) ∫_{φ−π/2}^{φ+π/2} N_p⟩_h',
                    preview: 'afplot',
                },
            },
            {
                id: 'visPass', title: 'PS_HairVisibility', tag: 'PS', lifetime: 'resident', step: 'lookup', desc: '相机视角：最近发丝 + 覆盖率',
                dispatch: { kind: 'PS', target: L3.VIEW_RES + '² · depth / tangent / coverage', note: '教学版每像素 1 个样本；生产版保留多样本' },
                inputs: [{ name: 'StrandVertices', format: 'float4', type: 'strand' }],
                outputs: [{ name: 'HairVisibilityBuffer', format: 'R32F + RGB10A2 + R16F', type: 'gbuf' }],
                detail: { desc: '本页用 HAIR.raster 从相机光栅化：depth、id、tangent 与 cov（像素 alpha = 1 − exp(−κ·cov)）。', preview: null },
            },
            {
                id: 'transPass', title: 'CS_HairLightTransmittance', tag: 'CS', lifetime: 'param', step: 'lookup', desc: 'm = κ·Ω(z) → directFraction',
                dispatch: { kind: 'CS', threads: [8, 8, 1], groups: () => [L3.VIEW_RES / 8, L3.VIEW_RES / 8, 1], idRange: 'ID.xy ∈ 屏幕像素', note: '每个光源一次；有 deep shadow 读 DOM，否则体素 ray-march' },
                inputs: [
                    { name: 'HairVisibilityBuffer', format: 'gbuffer', type: 'gbuf' },
                    { name: 'DeepOpacityMap', format: 'RGBA16F', type: 'opacity' },
                    { name: 'HairDensityVolume', format: 'RGBA16F', type: 'volume' },
                ],
                outputs: [{ name: 'HairCountMask', format: 'R16F', type: 'scalar' }],
                detail: { formula: 0, code: 0, preview: 'hcount' },
            },
            {
                id: 'shadePass', title: 'PS_HairDualScattering', tag: 'PS', lifetime: 'frame', step: 'compare', desc: 'F = direct·(S + S_back) + (T_f − direct)·d_f·(…)',
                dispatch: { kind: 'PS', target: L3.VIEW_RES + '² RGBA16F', note: '每像素每帧一个随机 h；S_scatter / S_back 确定性求值' },
                inputs: [
                    { name: 'HairVisibilityBuffer', format: 'gbuffer', type: 'gbuf' },
                    { name: 'HairCountMask', format: 'R16F', type: 'scalar' },
                    { name: 'DualScatteringLUT', format: 'RGBA16F', type: 'lut' },
                ],
                outputs: [{ name: 'HairLighting', format: 'RGBA16F', type: 'color' }],
                detail: { formula: 0, code: 0, preview: 'shDual' },
            },
            {
                id: 'accumPass', title: 'CS_HairTemporalAccumulate', tag: 'CS', category: 'output', lifetime: 'history', step: 'lookup', desc: '⟨S⟩_h 时域平均（≈ TAA）',
                dispatch: { kind: 'CS', threads: [8, 8, 1], groups: () => [L3.VIEW_RES / 8, L3.VIEW_RES / 8, 1], idRange: 'ID.xy ∈ 屏幕像素', note: '光源 / 发色改变时清零' },
                inputs: [
                    { name: 'HairLighting', format: 'RGBA16F', type: 'color' },
                    { name: 'HairHistory', format: 'RGBA32F', type: 'color', history: true },
                ],
                outputs: [{ name: 'HairHistory', format: 'RGBA32F', type: 'color' }],
                detail: { formula: 1, preview: 'shDom' },
            },
        ],
    },

    references: [
        { title: 'Deep Shadow Maps', author: 'Tom Lokovic, Eric Veach', venue: 'SIGGRAPH 2000', year: 2000, url: 'https://graphics.stanford.edu/papers/deepshadows/', note: '逐 texel 可见度函数；L∞ 误差下贪心画"最长线段"压缩' },
        { title: 'Opacity Shadow Maps', author: 'Tae-Yong Kim, Ulrich Neumann', venue: 'Eurographics Workshop on Rendering', year: 2001, note: '均匀切片 + 加法混合；τ = exp(−κΩ)，切片间线性插值' },
        { title: 'Photorealistic Hair Modeling, Animation, and Rendering（SIGGRAPH course notes）', author: 'Tae-Yong Kim et al.', venue: 'SIGGRAPH Course', url: 'http://www.graphics.stanford.edu/courses/cs448-05-winter/papers/course34.pdf', note: 'OSM 的 τ = e^{−κΩ} 与插值公式出处' },
        { title: 'Deep Opacity Maps', author: 'Cem Yuksel, John Keyser', venue: 'Computer Graphics Forum 27(2) · Eurographics 2008', year: 2008, url: 'http://www.cemyuksel.com/research/deepopacity/', note: '从 z₀ 起分层；通常 3 层、层厚线性递增；最后一层收纳剩余片元' },
        { title: 'Dual Scattering Approximation for Fast Multiple Scattering in Hair', author: 'Arno Zinke, Cem Yuksel, Andreas Weber, John Keyser', venue: 'ACM TOG 27(3) · SIGGRAPH 2008', year: 2008, url: 'http://www.cemyuksel.com/research/dualscattering/', note: 'Eq. 3–25、Table 1、Fig. 5 伪代码；GPU 版基于 deep opacity maps' },
        { title: 'Light Scattering from Human Hair Fibers', author: 'Stephen R. Marschner, Henrik Wann Jensen, Mike Cammarano, Steve Worley, Pat Hanrahan', venue: 'ACM TOG 22(3) · SIGGRAPH 2003', year: 2003 },
        { title: 'A Practical and Controllable Hair and Fur Model for Production Path Tracing', author: 'Matt Jen-Yuan Chiang, Benedikt Bitterli, Chuck Tappan, Brent Burley', venue: 'Computer Graphics Forum 35(2)', year: 2016, note: '本系列 HAIR.bsdf 的单根发丝模型（pbrt-v3 实现）' },
        { title: 'Physically Based Hair Shading in Unreal', author: 'Brian Karis', venue: 'SIGGRAPH 2016 · Physically Based Shading course', year: 2016, url: 'https://blog.selfshadow.com/publications/s2016-shading-course/karis/s2016_pbs_epic_hair.pdf' },
        { title: 'Strand-based Hair Rendering in Frostbite', author: 'Sebastian Tafuri', venue: 'SIGGRAPH 2019 · Advances in Real-Time Rendering', year: 2019, url: 'https://advances.realtimerendering.com/s2019/hair_presentation_final.pdf' },
        { title: 'Grooming for Real-Time Realism: Hair and Fur with Unreal Engine（白皮书，中文版）', author: 'Epic Games', venue: 'Unreal Engine white paper', url: 'https://cdn2.unrealengine.com/unreal-engine-hair-and-fur-whitepaper-zhcn-73d68a66a2a5.pdf', note: '体素密度用于发丝阴影、AO 与环境光照' },
    ],
});

// ── 3D view (Step 04)：光源空间里的 DOM —— 只读 dom3d stage 的结果 ──
function drawDom3d(c, S) {
    const D = S.dom3d, B = S.box, cb = [0, 1, 2].map((a) => (B[a] + B[a + 3]) / 2), s = 2 / Math.max(B[3] - B[0], B[4] - B[1], B[5] - B[2]);
    const V = (x, y, z) => [(x - cb[0]) * s, (y - cb[1]) * s, (z - cb[2]) * s];   // cm → 视图单位（最长边 = 2）
    const lines = [];
    // 发丝：每段一条线，亮度 = 该段中点的 T_DOM（暖白 = 受光，深灰 = 被前面的发丝挡住；不用彩色，免得和层线混淆）
    for (const st of D.strands) {
        const P = st.p;
        for (let k = 0; k < st.T.length; k++) {
            const t = st.T[k], g = (a, b) => Math.round(a + (b - a) * t);
            lines.push({ pts: [V(P[3 * k], P[3 * k + 1], P[3 * k + 2]), V(P[3 * k + 3], P[3 * k + 4], P[3 * k + 5])], color: `rgb(${g(46, 255)}, ${g(42, 238)}, ${g(40, 205)})`, width: 0.9 });
        }
    }
    // 每条扫描列：z₀（蓝）与各层末端 z₀ + d_k（橙，越深越淡）连成折线，遇到空格就断开——它们是同一条曲线沿光线平移的副本
    const nc = D.nc, nL = D.ends.length, layerCol = (k, a = 1) => (k === 0 ? `rgba(106, 176, 224, ${0.95 * a})` : `rgba(224, 160, 96, ${((0.95 - 0.5 * (k - 1) / Math.max(1, nL - 2)) * a).toFixed(2)})`);
    const points = [];
    D.cols.forEach((col, r) => {
        col.forEach((L, k) => {
            let run = [];
            const flush = () => { if (run.length > 1) lines.push({ pts: run, color: layerCol(k), width: k === 0 ? 2.2 : 1.6 }); run = []; };
            for (let i = 0; i < nc; i++) { if (Number.isNaN(L[3 * i])) { flush(); continue; } run.push(V(L[3 * i], L[3 * i + 1], L[3 * i + 2])); }
            flush();
        });
        if (r % 2) return;
        // 每隔一列画一条光线：从光源一侧穿过 z₀，依次经过各层末端（深度都沿光线、从 z₀ 量起）
        const valid = []; for (let i = 0; i < nc; i++) if (!Number.isNaN(col[0][3 * i])) valid.push(i);
        if (!valid.length) return;
        const i = valid[valid.length >> 1], at = (k) => V(col[k][3 * i], col[k][3 * i + 1], col[k][3 * i + 2]);
        const a = at(0), b = at(nL - 1);
        lines.push({ pts: [PL.v3.add(a, PL.v3.mul(S.wi, 0.35)), PL.v3.add(b, PL.v3.mul(S.wi, -0.12))], color: 'rgba(255, 216, 102, 0.75)', width: 1, dash: [3, 3] });
        for (let k = 0; k < nL; k++) points.push({ p: at(k), color: layerCol(k), r: k === 0 ? 3.5 : 3 });
    });
    const c0 = [0, 0, 0];
    lines.push({ pts: [PL.v3.mul(S.wi, 1.25), PL.v3.mul(S.wi, 0.78)], color: '#ffd866', width: 2.4, arrow: true, labelAt: 'start', label: 'ωi（光）' });
    PL.draw.mesh3d(c, {
        lines, points,
        grid: { y: (B[1] - cb[1]) * s - 0.04, size: 1.2 },
        orbit: { yaw: S.lightAz - 1.45, pitch: 0.42, dist: 4.4, target: c0 },   // 默认从垂直于光线的一侧斜上方看：层的间距沿光线展开
    });
}
