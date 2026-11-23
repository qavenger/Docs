// 毛发渲染 01 · 着色模型 —— build from inside pages/:
//   node <skill>/scripts/build_doc.mjs hair_common.js hair_01_shading.lab.js -o hair_01_shading.html
// Conventions: see hair_common.js (fiber frame x = tangent root → tip, sinθ = ω·t, φ in the normal plane,
// wi → light, wo → viewer, S normalized so ∫S dω = 1 for σa = 0, L_o = S·E for a directional light).

const SH = (() => {
    // camera looks at the groom from the side its tips drape toward (yaw = π), so strands lean toward the viewer
    // down their length like hair over the crown — that is what makes a horizontal highlight band
    const W = 132, H = 160, cam = HAIR.ortho({ yaw: Math.PI, center: [0, -7.2, 0], size: 17.5 }), BG = [0.03, 0.035, 0.045];
    const lightDir = (S) => {   // camera-relative: az = 0 → from the viewer, ±180 → from behind the hair; el > 0 → from above
        const az = S.lightAz * Math.PI / 180, el = S.lightEl * Math.PI / 180, ce = Math.cos(el), R = cam.right, V = cam.toViewer, U = cam.up;
        return HAIR.norm([0, 1, 2].map((k) => ce * (Math.sin(az) * R[k] + Math.cos(az) * V[k]) + Math.sin(el) * U[k]));
    };
    const sat = (x) => Math.min(1, Math.max(0, x));
    // Marschner-style Gaussian (UE's Hair_g) and Schlick hair Fresnel (η = 1.55)
    const gauss = (B, x) => Math.exp(-0.5 * x * x / (B * B)) / (Math.sqrt(2 * Math.PI) * B);
    const KK_K = 2.2 / 7;   // Kajiya-Kay scale so that L = S·E with the same E = 7 as the Chiang render
    const kkS = (T, L, V, tint) => { const k = HAIR.kajiyaKay(T, L, V, 60); return tint.map((t) => KK_K * (0.35 * t * k.diffuse + 0.25 * k.specular)); };
    const hairF = (c) => { const F0 = ((1 - 1.55) / (1 + 1.55)) ** 2; return F0 + (1 - F0) * (1 - c) ** 5; };
    // pbrt-v3's original LogI0 (correction term 1/(16x)) — for the comparison plot only
    const logI0pbrt = (x) => (x > 12 ? x + 0.5 * (-Math.log(2 * Math.PI) + Math.log(1 / x) + 1 / (8 * x)) : Math.log(HAIR.I0(x)));
    const MpPbrt = (cI, cO, sI, sO, v) => { const a = cI * cO / v, b = sI * sO / v; return v <= 0.1 ? Math.exp(logI0pbrt(a) - b - 1 / v + 0.6931 + Math.log(1 / (2 * v))) : Math.exp(-b) * HAIR.I0(a) / (Math.sinh(1 / v) * 2 * v); };
    const normM = (fn, thO) => { let s = 0; const n = 1200; for (let i = 0; i < n; i++) { const t = -Math.PI / 2 + Math.PI * (i + 0.5) / n; s += fn(t, thO) * Math.cos(t) * Math.PI / n; } return s; };
    // UE / Karis 2016-style real-time hair (world-space L, V, T; T = root → tip in this series)
    const ue = (L, V, T, P) => {
        const sinL = Math.max(-1, Math.min(1, HAIR.dot(T, L))), sinV = Math.max(-1, Math.min(1, HAIR.dot(T, V))), VoL = HAIR.dot(V, L);
        const cosTD = Math.cos(0.5 * Math.abs(Math.asin(sinV) - Math.asin(sinL)));
        const Lp = [L[0] - sinL * T[0], L[1] - sinL * T[1], L[2] - sinL * T[2]], Vp = [V[0] - sinV * T[0], V[1] - sinV * T[1], V[2] - sinV * T[2]];
        const cosPhi = HAIR.dot(Lp, Vp) / Math.sqrt(HAIR.dot(Lp, Lp) * HAIR.dot(Vp, Vp) + 1e-4), cosHalf = Math.sqrt(sat(0.5 + 0.5 * cosPhi));
        const nP = 1.19 / cosTD + 0.36 * cosTD, a = 1 / nP, sh = P.alpha * Math.PI / 180, A = [-2 * sh, sh, 4 * sh], r2 = P.rough * P.rough, B = [r2, r2 / 2, r2 * 2];
        const C = P.sigmaA.map((s) => Math.exp(-2 * s));   // transmittance through one fiber diameter
        const sa = Math.sin(A[0]), ca = Math.cos(A[0]), shR = 2 * sa * (ca * cosHalf * Math.sqrt(1 - sinV * sinV) + sa * sinV);
        const R = gauss(B[0] * Math.SQRT2 * Math.max(cosHalf, 1e-3), sinL + sinV - shR) * 0.25 * cosHalf * hairF(Math.sqrt(sat(0.5 + 0.5 * VoL)));
        const h = cosHalf * (1 + a * (0.6 - 0.8 * cosPhi)), fT = hairF(cosTD * Math.sqrt(sat(1 - h * h)));
        const TTk = gauss(B[1], sinL + sinV - A[1]) * Math.exp(-3.65 * cosPhi - 3.98) * (1 - fT) ** 2;
        const fR = hairF(cosTD * 0.5), TRTk = gauss(B[2], sinL + sinV - A[2]) * Math.exp(17 * cosPhi - 16.78) * (1 - fR) ** 2 * fR;
        const tt = C.map((c) => Math.pow(c, Math.sqrt(sat(1 - (h * a) ** 2)) / cosTD)), trt = C.map((c) => Math.pow(c, 0.8 / cosTD));
        return { R, TT: tt.map((t) => t * TTk), TRT: trt.map((t) => t * TRTk) };
    };
    // exact 2D ray paths through a circular fiber cross-section (normal plane), offset h, index eta
    const refract2 = (d, n, r) => { const ci = -(d[0] * n[0] + d[1] * n[1]), k = 1 - r * r * (1 - ci * ci); if (k < 0) return null; const s = r * ci - Math.sqrt(k); return [r * d[0] + s * n[0], r * d[1] + s * n[1]]; };
    const tracePaths = (h, eta) => {
        const P = [-Math.sqrt(Math.max(0, 1 - h * h)), h], d = [1, 0], n = [P[0], P[1]], dn = d[0] * n[0] + d[1] * n[1];
        const out = [{ pts: [[-2.2, h], P], dir: [d[0] - 2 * dn * n[0], d[1] - 2 * dn * n[1]] }];
        let cur = P, dir = refract2(d, n, 1 / eta);
        const pts = [[-2.2, h], P];
        for (let p = 1; p <= 2 && dir; p++) {
            const s = -2 * (cur[0] * dir[0] + cur[1] * dir[1]), Q = [cur[0] + s * dir[0], cur[1] + s * dir[1]];
            pts.push(Q);
            const ex = refract2(dir, [-Q[0], -Q[1]], eta);
            out.push({ pts: pts.slice(), dir: ex });
            const dq = dir[0] * Q[0] + dir[1] * Q[1];
            dir = [dir[0] - 2 * dq * Q[0], dir[1] - 2 * dq * Q[1]];
            cur = Q;
        }
        return out;
    };
    return { W, H, cam, BG, lightDir, gauss, hairF, MpPbrt, normM, ue, tracePaths, kkS };
})();
const etaOf = (eta, thO) => Math.sqrt(eta * eta - Math.sin(thO) ** 2) / Math.cos(thO);   // Bravais effective index η'(θ)
const LOBE_COL = ['#e8c25a', '#e06c75', '#5ad07a', '#9aa5b1'];
const LOBE_NAME = ['R', 'TT', 'TRT', 'p ≥ 3'];
const L3D = { nu: 48, nv: 32 };

PipelineLab.mount({
    paramSteps: { thetaO: ['geom', 'paths', 'mterm', 'shift', 'nterm'], hView: ['paths', 'nterm'], betaM: ['mterm', 'shift', 'nterm', 'energy', 'render', 'rt', 'prod'], betaN: ['nterm', 'render', 'rt', 'prod'], alpha: ['shift', 'nterm', 'energy', 'render', 'rt', 'prod'], melanin: ['kk', 'nterm', 'absorb', 'render', 'rt', 'prod'], pheo: ['kk', 'nterm', 'absorb', 'render', 'rt', 'prod'], lightAz: ['kk', 'render', 'rt', 'prod'], lightEl: ['kk', 'render', 'rt', 'prod'], lobe: ['nterm', 'render', 'prod'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '毛发渲染 01 · 着色模型：Kajiya-Kay → Marschner → d\'Eon / Chiang → 实时近似',
    lang: 'zh-CN',
    codeLang: 'hlsl',
    seed: 101,
    tagColors: { GEOM: 'gray', KK: 'olive', PATHS: 'blue', M: 'magenta', SHIFT: 'rust', N: 'indigo', ABSORB: 'green', ENERGY: 'teal', RENDER: 'blue', RT: 'olive', PROD: 'gray' },
    params: [
        { id: 'thetaO', label: 'θo', min: -70, max: 70, step: 1, value: 20, unit: '°', desc: '观察方向与发丝<strong>法平面</strong>的夹角（纵向角，正值偏向发梢）。光线固定在镜面方向 θi = −θo：Step 01 的镜面锥、Step 03 的 Bravais η′、Step 04 / 05 的纵向曲线都以它为参照。拖到 ±60° 以上看 Gaussian 纵向项在掠射角漏能量、TRT 焦散消失。' },
        { id: 'hView', label: 'h', min: -0.98, max: 0.98, step: 0.01, value: 0.45, digits: 2, desc: '光线在纤维截面上的偏移：−1 … 1 = 从一侧边缘到另一侧边缘，0 = 正中。只影响 Step 03 的截面光路和 Step 06 的"单个 h"方位曲线；渲染里的 h 是每帧每像素随机取的。' },
        { id: 'betaM', label: 'βm', min: 0.05, max: 0.8, step: 0.01, value: 0.25, desc: '<strong>纵向粗糙度</strong>（Chiang 2016 参数化，换算成 d\'Eon 的方差 v）。越大，高光沿发丝方向拉得越宽、越暗；TT、TRT 的宽度按 v/4、4v 跟着变。' },
        { id: 'betaN', label: 'βn', min: 0.1, max: 0.9, step: 0.01, value: 0.35, desc: '<strong>方位粗糙度</strong>：截断 logistic 的宽度 s。越大，TT 背光光晕和 TRT 闪点越模糊；越小，对 h 的积分越难算准（Step 09）。' },
        { id: 'alpha', label: 'α', min: 0, max: 6, step: 0.1, value: 2, unit: '°', desc: '<strong>角质层倾角</strong>。把 R 推向发根、TT / TRT 推向发梢，于是白色主高光和带发色的第二高光上下错开（Step 05）；0° 时三条 lobe 都以镜面锥为中心。' },
        { id: 'melanin', label: 'eumelanin', min: 0, max: 3, step: 0.05, value: 0.5, desc: '<strong>真黑色素</strong>浓度 → 吸收系数 σ<sub>a</sub>（偏黑褐）。0 ≈ 白 / 金发，0.5 ≈ 浅棕，1.3 ≈ 棕，3 ≈ 黑；它决定 TT、TRT 的颜色和强度，R 不受影响。' },
        { id: 'pheo', label: 'pheomelanin', min: 0, max: 2, step: 0.05, value: 0.1, desc: '<strong>褐黑色素</strong>浓度（偏红 / 橙）。与真黑色素相加得到 σ<sub>a</sub>；调高得到红发、铜色。' },
        { id: 'lightAz', label: '光源方位', min: -170, max: 170, step: 1, value: -30, unit: '°', desc: '方向光绕视线的方位角：0° = 从观察者一侧照射，±90° = 侧光，±170° = 背光（看 TT 透射出的金边）。只影响渲染类预览（Step 02 / 09 / 10 / 11）。', affects: 'param' },
        { id: 'lightEl', label: '光源仰角', min: -30, max: 75, step: 1, value: 22, unit: '°', desc: '方向光的仰角，正值来自上方。它决定高光带落在发帘的哪个高度——发丝朝观察者倾斜的角度恰好满足镜面条件的地方。', affects: 'param' },
        { id: 'lobe', label: '显示 lobe', options: [{ value: 'all', label: '全部' }, { value: '0', label: 'R' }, { value: '1', label: 'TT' }, { value: '2', label: 'TRT' }], value: 'all', desc: '只看某一条散射路径：R（白色主高光）、TT（背光透射）、TRT（带发色、位置错开的第二高光）。作用于 Step 06 的 3D lobe 和 Step 09 的渲染。' },
    ],
    stages: [
        { id: 'groom', lifetime: 'resident', run(S) { S.groom = HAIR.makeGroom({ seed: 21, perGuide: 70, curl: 0.1 }); S.rast = HAIR.raster(S.groom.strands, SH.cam, SH.W, SH.H, { widthScale: 1.6 }); } },
        {
            id: 'model', lifetime: 'param', run(S) {
                S.sigmaA = HAIR.sigmaAFromConcentration(S.melanin, S.pheo);
                S.hb = HAIR.bsdf({ sigmaA: S.sigmaA, betaM: S.betaM, betaN: S.betaN, alpha: S.alpha });
                // azimuthal profiles in the normal plane (θi = θo = 0): Chiang per lobe (h-averaged, 256 points) + UE fit
                const n = 91, phis = PL.buf(S, 'phis', n), lob = [0, 1, 2].map((p) => PL.buf(S, 'az' + p, n)), uel = [0, 1, 2].map((p) => PL.buf(S, 'ue' + p, n));
                // UE roughness r with B = r² = √v: same longitudinal σ as the d'Eon R lobe
                const wo = [0, 1, 0], T = [1, 0, 0], V = [0, 1, 0], P = { alpha: S.alpha, rough: Math.pow(S.hb.v[0], 0.25), sigmaA: S.sigmaA };
                for (let i = 0; i < n; i++) {
                    const phi = -Math.PI + 2 * Math.PI * i / (n - 1), wi = [0, Math.cos(phi), Math.sin(phi)];
                    phis[i] = phi * 180 / Math.PI;
                    const acc = [0, 0, 0];
                    for (let k = 0; k < 256; k++) { const l = S.hb.lobes(wi, wo, -1 + (2 * k + 1) / 256); for (let p = 0; p < 3; p++) acc[p] += HAIR.lum(l[p]) / 256; }
                    for (let p = 0; p < 3; p++) lob[p][i] = acc[p];
                    const u = SH.ue(wi, V, T, P); uel[0][i] = u.R; uel[1][i] = HAIR.lum(u.TT); uel[2][i] = HAIR.lum(u.TRT);
                }
                // static images: Kajiya-Kay and the UE-style approximation on the shared raster
                const r = S.rast, L = SH.lightDir(S), Vv = SH.cam.toViewer, E = 7, N = SH.W * SH.H;
                const kk = PL.buf(S, 'kk', 3 * N), ui = PL.buf(S, 'uei', 3 * N), tint = S.sigmaA.map((s) => Math.exp(-2 * s));
                for (let i = 0; i < N; i++) {
                    if (r.id[i] < 0) { for (let c = 0; c < 3; c++) { kk[3 * i + c] = SH.BG[c]; ui[3 * i + c] = SH.BG[c]; } continue; }
                    const Tt = [r.tan[3 * i], r.tan[3 * i + 1], r.tan[3 * i + 2]], a = r.alpha[i], k = SH.kkS(Tt, L, Vv, tint), u = SH.ue(L, Vv, Tt, P);
                    for (let c = 0; c < 3; c++) {
                        kk[3 * i + c] = a * E * k[c] + (1 - a) * SH.BG[c];
                        ui[3 * i + c] = a * E * (u.R + u.TT[c] + u.TRT[c]) + (1 - a) * SH.BG[c];
                    }
                }
            },
        },
        {
            id: 'shift', lifetime: 'param', run(S) {   // measure each lobe's longitudinal peak from HAIR.bsdf itself (checks the α sign)
                const h = 0.3, thO = S.thetaO * Math.PI / 180, wo = [Math.sin(thO), Math.cos(thO), 0], n = 361;
                const scan = (hb, p) => {   // φ = Φ(p, h): the azimuth where this lobe's N_p peaks, so only M_p varies with θi
                    const phi = HAIR.Phi(p, Math.asin(h), Math.asin(h / etaOf(hb.eta, thO))), out = new Float32Array(n);
                    for (let k = 0; k < n; k++) { const t = (-90 + 180 * k / (n - 1)) * Math.PI / 180; out[k] = HAIR.lum(hb.lobes([Math.sin(t), Math.cos(t) * Math.cos(phi), Math.cos(t) * Math.sin(phi)], wo, h)[p]); }
                    return out;
                };
                const peakOf = (a) => { let b = 1; for (let k = 2; k < n - 1; k++) if (a[k] > a[b]) b = k; const y0 = a[b - 1], y1 = a[b], y2 = a[b + 1], d = y0 - 2 * y1 + y2, off = d < 0 ? 0.5 * (y0 - y2) / d : 0; return -90 + 180 * (b + off) / (n - 1); };
                S.peaks = [[], [], []];
                for (let ai = 0; ai <= 6; ai++) {
                    const hb = HAIR.bsdf({ sigmaA: [0, 0, 0], betaM: S.betaM, betaN: S.betaN, alpha: ai });
                    for (let p = 0; p < 3; p++) S.peaks[p].push((peakOf(scan(hb, p)) + S.thetaO) / 2);   // θh = (θi + θo)/2
                }
                S.prof = [0, 1, 2].map((p) => { const a = scan(S.hb, p); let m = 0; for (const v of a) m = Math.max(m, v); return Array.from(a, (v) => v / (m || 1)); });
                S.profX = Array.from({ length: n }, (_, k) => -90 + 180 * k / (n - 1));
            },
        },
        {
            id: 'accum', lifetime: 'history', run(S) {   // Chiang 2016 on the raster: random h per pixel, running mean; checkerboard halves per frame
                const n = SH.W * SH.H, acc = PL.buf(S, 'acc', 3 * n), cnt = PL.buf(S, 'accN', n);
                let full = false;
                if (S._ep !== S.epoch) { acc.fill(0); cnt.fill(0); S._ep = S.epoch; full = true; }
                const r = S.rast, L = SH.lightDir(S), V = SH.cam.toViewer, rnd = PL.rng((S.frame * 2654435761) >>> 0), E = 7, sel = S.lobe === 'all' ? -1 : Number(S.lobe), par = S.frame & 1;
                for (let i = 0; i < n; i++) {
                    if (!full && ((i % SH.W) + ((i / SH.W) | 0) + par) & 1) continue;
                    let c = SH.BG;
                    if (r.id[i] >= 0) {
                        const f = HAIR.fiberFrame([r.tan[3 * i], r.tan[3 * i + 1], r.tan[3 * i + 2]], V), l = S.hb.lobes(HAIR.toFiber(L, f), HAIR.toFiber(V, f), 2 * rnd() - 1), a = r.alpha[i];
                        const s = sel < 0 ? [0, 1, 2].map((k) => l[0][k] + l[1][k] + l[2][k] + l[3][k]) : l[sel];
                        c = [0, 1, 2].map((k) => a * s[k] * E + (1 - a) * SH.BG[k]);
                    }
                    const w = 1 / (cnt[i] = Math.min(cnt[i] + 1, 64));
                    for (let k = 0; k < 3; k++) acc[3 * i + k] += (c[k] - acc[3 * i + k]) * w;
                }
            },
        },
        {
            id: 'furnace', lifetime: 'param', lazy: true, run(S) {   // ∫ S dωi over the sphere for σa = 0, vs θo — regular (sinθ, φ) grid
                // Chiang: brute-force over directions for 4 fixed h (each must give 1 since Σ_p A_p(h) = 1); UE / KK have no h
                const thetas = [0, 15, 30, 45, 60, 70, 80], nT = 60, nP = 72, hs = [-0.75, -0.25, 0.25, 0.75];
                const hb0 = HAIR.bsdf({ sigmaA: [0, 0, 0], betaM: S.betaM, betaN: S.betaN, alpha: S.alpha }), P0 = { alpha: S.alpha, rough: Math.pow(hb0.v[0], 0.25), sigmaA: [0, 0, 0] }, one = [1, 1, 1];
                S.fTheta = thetas;
                S.fChiang = []; S.fUE = []; S.fKK = [];
                for (const td of thetas) {
                    const t = td * Math.PI / 180, wo = [Math.sin(t), Math.cos(t), 0], T = [1, 0, 0];
                    let sc = 0, su = 0, sk = 0;
                    for (let i = 0; i < nT; i++) {
                        const st = -1 + 2 * (i + 0.5) / nT, ct = Math.sqrt(1 - st * st);
                        for (let j = 0; j < nP; j++) {
                            const p = -Math.PI + 2 * Math.PI * (j + 0.5) / nP, wi = [st, ct * Math.cos(p), ct * Math.sin(p)];
                            for (const h of hs) sc += hb0.S(wi, wo, h)[1] / hs.length;
                            const u = SH.ue(wi, wo, T, P0); su += u.R + u.TT[1] + u.TRT[1];
                            sk += SH.kkS(T, wi, wo, one)[1];
                        }
                    }
                    const dw = (2 / nT) * (2 * Math.PI / nP);
                    S.fChiang.push(sc * dw); S.fUE.push(su * dw); S.fKK.push(sk * dw);
                }
            },
        },
        {
            id: 'lobe3d', lifetime: 'param', lazy: true, run(S) {   // h-averaged R / TT / TRT over the sphere of ωo for a fixed ωi (fiber frame: x = T)
                const nu = L3D.nu, nv = L3D.nv, NU = nu + 1, NV = nv + 1, nh = 64, thI = -S.thetaO * Math.PI / 180, wi = [Math.sin(thI), Math.cos(thI), 0];
                const R = [0, 1, 2].map((p) => PL.buf(S, 'l3d' + p, NU * NV));
                for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) {
                    const ph = 2 * Math.PI * i / nu, th = -Math.PI / 2 + Math.PI * j / nv, wo = [Math.sin(th), Math.cos(th) * Math.cos(ph), Math.cos(th) * Math.sin(ph)];
                    const acc = [0, 0, 0];
                    for (let k = 0; k < nh; k++) { const l = S.hb.lobes(wi, wo, -1 + (2 * k + 1) / nh); for (let p = 0; p < 3; p++) acc[p] += HAIR.lum(l[p]) / nh; }
                    for (let p = 0; p < 3; p++) R[p][j * NU + i] = acc[p];
                }
                S.l3dMax = R.map((a) => { let m = 0, at = 0; for (let k = 0; k < a.length; k++) if (a[k] > m) { m = a[k]; at = k; } return { m: m || 1, at }; });
            },
        },
    ],
    intro: {
        title: '毛发着色 · 从一条经验高光到三条光路',
        flow: ['纤维几何', 'Kajiya-Kay', 'R / TT / TRT', '纵向 M_p', '角质层倾角', '方位 N_p', '吸收 A_p', '白炉', '发丝渲染', '实时近似'],
        html: `<p>头发不是表面，是一根根半透明的细圆柱。光打到发丝上有三条主要出路：在角质层表面<strong>直接反射</strong>（R，白色主高光）、<strong>穿过</strong>发丝从另一侧出来（TT，背光时的光晕）、进去后在内壁<strong>反射一次</strong>再出来（TRT，带发色的第二高光）。本页先用 Kajiya-Kay 这个"只有一条高光"的经验模型做对照，再按 Marschner 2003 的结构把三条光路拆成纵向项 M、方位项 N、衰减 A 三部分，换上 d'Eon 2011 的能量守恒 M 和 Chiang 2016 的 logistic N，最后看实时引擎怎么用闭式拟合替代这些积分。</p>
<p>所有曲线、热力图和渲染都由系列共享的 <code>HAIR.bsdf</code>（pbrt-v3 / Chiang 2016 的移植）实时算出。</p>${HAIR.nav('shading')}`,
    },
    steps: [
        {
            id: 'geom', title: '纤维几何：θ、φ、h 与纤维坐标系', tag: 'GEOM',
            desc: '发丝的局部坐标以<strong>切线 t</strong>（根 → 梢）为 x 轴：纵向角 θ 从法平面量起（sinθ = ω·t），方位角 φ 在法平面里量。光线在截面上的落点用 <strong>h ∈ [−1, 1]</strong> 表示（相对半径的偏移），对应入射角 γ = asin h。圆柱的镜面反射方向构成一个圆锥：θ<sub>o</sub> = −θ<sub>i</sub>，所以发丝高光是一条<strong>沿着发丝垂直方向延伸的亮带</strong>，而不是一个点。',
            formula: 'sinθ = ω · t      φ = atan2(ω·b, ω·n)      γ = asin(h)\n镜面锥：θ_o = −θ_i      θ_h = (θ_i + θ_o)/2,   θ_d = (θ_o − θ_i)/2,   φ = φ_i − φ_o',
            code: { title: 'HLSL · 纤维坐标', src: `
                // T：发丝切线（根 → 梢）。返回 sinθ 与法平面内的方位角差 cos(φ)
                void HairAngles(float3 L, float3 V, float3 T, out float sinThetaL, out float sinThetaV, out float cosPhi) {
                    sinThetaL = dot(T, L);
                    sinThetaV = dot(T, V);
                    cosPhi    = CosPhi(L, V, T);          // Common：投影到法平面后的夹角
                }` },
            previews: [
                { id: 'geomDiag', label: '纤维与镜面锥', sub: '黄 = 入射，蓝 = 镜面锥上的出射方向', size: [300, 200], render: (c, S) => drawCone(c, S) },
                { id: 'geomCov', label: '共享 groom 的屏幕空间覆盖率', sub: '后面所有渲染都着色这一张 raster', render: (c, S) => PL.draw.field(c, S.rast.alpha, SH.W, SH.H, { range: [0, 1], map: 'gray' }) },
                { id: 'geom3d', large: true, label: '3D：发丝、入射光与镜面锥', sub: '黄箭头 = 指向光源（θi = −θo）；蓝色线框 = 所有满足镜面条件的出射方向；粗蓝箭头 = 入射平面内的那一条', render: (c, S) => drawGeom3d(c, S) },
            ],
        },
        {
            id: 'kk', title: 'Kajiya-Kay：漫反射 sin(T, L) + 高光 sin(T, H)^p', tag: 'KK',
            desc: 'Kajiya &amp; Kay 1989 把发丝当成无限细的不透明圆柱：漫反射对圆柱表面积分得到 ∝ sin(T, L)；高光取镜面锥上的 Phong 式衰减，常写成半向量形式 sin(T, H)<sup>p</sup>。它便宜、可控，至今仍是插片（模块 04）的主力；但只有一条高光、没有透射，也谈不上能量守恒：总能量完全由手调的 k<sub>d</sub>、k<sub>s</sub> 决定（右图：本页取值下白炉在 1.13–1.29 之间随 θ<sub>o</sub> 漂移，且与 β 无关——它根本没有粗糙度的概念）。',
            formula: 'f_KK = k_d · sin(T, L) + k_s · sin(T, H)^p      sin(T, X) = √(1 − (T·X)^2)      H = normalize(L + V)',
            code: { title: 'HLSL · Kajiya-Kay', src: `
                float3 KajiyaKay(float3 T, float3 L, float3 V, float3 albedo, float specPower) {
                    float  TdotL  = dot(T, L);
                    float  TdotH  = dot(T, normalize(L + V));
                    float  diff   = sqrt(saturate(1.0 - TdotL * TdotL));
                    float  spec   = pow(sqrt(saturate(1.0 - TdotH * TdotH)), specPower);
                    return albedo * diff * 0.35 + spec * 0.25;
                }` },
            previews: [
                { id: 'kkImg', label: 'Kajiya-Kay 着色', smooth: true, render: (c, S) => drawRGB(c, S.kk) },
                { id: 'kkFurn', needs: ['furnace'], label: '白炉 ∫ f dω（同样的 k_d、k_s）', sub: '随 θo 变化 → 不守恒', size: [300, 170], render: (c, S) => PL.draw.plot(c, { x: [0, 80], y: [0, 1.4], xLabel: 'θo (°)', series: [{ data: S.fKK, x: S.fTheta, color: '#d19a66', label: 'Kajiya-Kay' }], hmarks: [{ y: 1, color: '#555' }] }) },
            ],
        },
        {
            id: 'paths', title: 'Marschner：R、TT、TRT 三条光路的几何', tag: 'PATHS',
            desc: 'Marschner et al. 2003 把发丝当成<strong>透明的介质圆柱</strong>，按光在内部走的段数 p 分类：p = 0 表面反射 R，p = 1 穿透 TT，p = 2 内部反射一次 TRT。在法平面里，出射方位只由偏移 h 决定：Φ(p, h) = 2pγ<sub>t</sub> − 2γ<sub>i</sub> + pπ，其中 γ<sub>t</sub> 由 Snell 定律给出。倾斜入射时把三维问题投影到法平面，用 <strong>Bravais 有效折射率</strong> η′ 代替 η，二维的折射公式仍然精确成立（左图就是按 η′(θ<sub>o</sub>) 追踪的）。TRT 的 Φ(h) 有极值点（dΦ/dh = 0），一片 h 汇聚到同一出射方向 —— 这就是头发上的<strong>闪点（glint / 焦散）</strong>；η′ 超过 2（θ ≳ 47°）时这对极值合并消失。',
            formula: 'γ_i = asin(h)      γ_t = asin(h / η\')      η\' = √(η^2 − sin^2 θ) / cosθ      // Bravais 有效折射率\nΦ(p, h) = 2p γ_t − 2γ_i + pπ      // 出射方位；右图画的是偏离名义方向的量 Φ − pπ\ndΦ/dh = 0  →  h_c^2 = (p^2 − η\'^2) / (p^2 − 1)      // TRT（p = 2）：η\' = 1.55 时 h_c ≈ ±0.73',
            previews: [
                { id: 'pathDiag', label: '截面上的三条光路（h = 参数）', sub: '黄 R · 红 TT · 绿 TRT；按 η′(θo) 精确追踪折射 / 反射', size: [280, 240], render: (c, S) => drawPaths(c, S) },
                {
                    id: 'phiPlot', label: 'Φ(p, h) − pπ：出射方位偏离 vs 偏移 h', sub: 'TRT 的极值点 = 焦散（闪点）', size: [320, 200],
                    render: (c, S) => {
                        const eta = etaOf(S.hb.eta, S.thetaO * Math.PI / 180), hc2 = (4 - eta * eta) / 3, marks = [{ x: S.hView, label: 'h', color: '#888' }];
                        if (hc2 > 0) marks.push({ x: Math.sqrt(hc2), label: '焦散', color: '#5ad07a' }, { x: -Math.sqrt(hc2), label: '焦散', color: '#5ad07a' });
                        PL.draw.plot(c, { x: [-1, 1], y: [-180, 180], xLabel: 'h', series: [0, 1, 2].map((p) => ({ fn: (h) => (HAIR.Phi(p, Math.asin(h), Math.asin(h / eta)) - p * Math.PI) * 180 / Math.PI, color: LOBE_COL[p], label: LOBE_NAME[p], samples: 300 })), marks });
                    },
                },
            ],
            notes: '拖 h：R 在截面边缘（|h| → 1）掠射，Fresnel 变大；TT 在 h ≈ 0 时几乎直穿（正背光方向）；TRT 的出射方向在 |h| ≈ 0.73 附近停住不动 —— 那就是闪点。把 θo 拖到 50° 以上，η′ > 2，焦散标记消失。',
        },
        {
            id: 'mterm', title: '纵向项 M_p：Gaussian → d\'Eon 归一化（Bessel I₀）', tag: 'M',
            desc: 'Marschner 用 Gaussian 描述沿发丝方向的粗糙散布（以 θ<sub>h</sub> 为变量），实时引擎也普遍沿用。Gaussian 是"平面上"的分布：θ<sub>o</sub> 接近掠射时，lobe 有一部分落到 θ<sub>i</sub> 的定义域 [−90°, 90°] 之外，积分不再为 1（中图虚线：默认 βm 下 θ<sub>o</sub> = 70° 时 ≈ 0.93，85° 时 ≈ 0.65）。d\'Eon et al. 2011 把镜面锥与球面上的 von Mises-Fisher 分布做卷积，得到在球面上严格归一化的 M<sub>p</sub>，含修正 Bessel 函数 I₀。数值上要小心：pbrt-v3 的 log I₀ 大参数渐近式把修正项写成了 1/(16x)（应为 1/(8x)），v ≈ 0.05 时 M<sub>p</sub> 只积分到 ≈ 0.996（右图），本系列的实现已改正。',
            formula: 'M_G(θ_i, θ_o) = g(β, θ_h − α)      // Marschner：Gaussian，未归一化\nM_p(θ_i, θ_o) = csch(1/v) / (2v) · exp(−sinθ_i sinθ_o / v) · I_0(cosθ_i cosθ_o / v)      ∫ M_p cosθ_i dθ_i = 1\nlog I_0(x) ≈ x − ½ log(2πx) + 1/(8x)      (x > 12)',
            code: { title: 'HLSL · d\'Eon M_p（小 v 时用对数形式避免溢出）', src: `
                float LogI0(float x) {
                    if (x > 12) return x - 0.5 * log(2 * PI * x) + log(1 + 1 / (8 * x));   // 注意：是 1/(8x)
                    float val = 0, x2i = 1, ifact = 1, i4 = 1;
                    [unroll] for (int i = 0; i < 16; ++i) { if (i > 1) ifact *= i; val += x2i / (i4 * ifact * ifact); x2i *= x * x; i4 *= 4; }
                    return log(val);
                }
                float Mp(float cosI, float cosO, float sinI, float sinO, float v) {
                    float a = cosI * cosO / v, b = sinI * sinO / v;
                    return (v <= 0.1) ? exp(LogI0(a) - b - 1 / v + 0.6931 + log(1 / (2 * v)))
                                      : exp(-b) * exp(LogI0(a)) / (sinh(1 / v) * 2 * v);
                }` },
            previews: [
                {
                    id: 'mCurve', label: 'M_p(θi)，θo = 参数', sub: '实线 d\'Eon，虚线 Gaussian（同宽度）', size: [320, 180],
                    render: (c, S) => { const tO = S.thetaO * Math.PI / 180, v = S.hb.v[0]; PL.draw.plot(c, { x: [-90, 90], xLabel: 'θi (°)', series: [{ fn: (d) => { const t = d * Math.PI / 180; return HAIR.Mp(Math.cos(t), Math.cos(tO), Math.sin(t), Math.sin(tO), v); }, color: '#c678dd', label: 'd\'Eon M_p', samples: 361 }, { fn: (d) => { const t = d * Math.PI / 180; return SH.gauss(Math.sqrt(v), t + tO) / Math.max(1e-3, Math.cos(t)); }, color: '#c678dd', dash: [4, 3], label: 'Gaussian', samples: 361 }], marks: [{ x: -S.thetaO, label: '−θo', color: '#666' }] }); },
                },
                {
                    id: 'mNorm', label: '归一化 ∫ M cosθi dθi vs θo', sub: '同一 v；Gaussian 在掠射处漏能量 / 多能量', size: [320, 180],
                    render: (c, S) => { const v = S.hb.v[0]; PL.draw.plot(c, { x: [0, 85], y: [0.5, 1.3], xLabel: 'θo (°)', series: [{ fn: (d) => SH.normM((t, o) => HAIR.Mp(Math.cos(t), Math.cos(o), Math.sin(t), Math.sin(o), v), d * Math.PI / 180), color: '#c678dd', label: 'd\'Eon', samples: 40 }, { fn: (d) => SH.normM((t, o) => SH.gauss(Math.sqrt(v), t + o) / Math.max(1e-3, Math.cos(t)), d * Math.PI / 180), color: '#c678dd', dash: [4, 3], label: 'Gaussian', samples: 40 }], hmarks: [{ y: 1, color: '#555' }] }); },
                },
                {
                    id: 'mPbrt', label: 'pbrt-v3 log I₀ 修正项的影响', sub: '∫ M_p cosθ dθ vs v（θo = 20°）', size: [320, 180],
                    render: (c) => { const tO = 20 * Math.PI / 180; PL.draw.plot(c, { x: [0.005, 1], logX: true, y: [0.99, 1.002], xLabel: 'v（方差）', series: [{ fn: (v) => SH.normM((t, o) => HAIR.Mp(Math.cos(t), Math.cos(o), Math.sin(t), Math.sin(o), v), tO), color: '#5ad07a', label: '本系列（1/(8x)）', samples: 60 }, { fn: (v) => SH.normM((t, o) => SH.MpPbrt(Math.cos(t), Math.cos(o), Math.sin(t), Math.sin(o), v), tO), color: '#e06c75', label: 'pbrt-v3（1/(16x)）', samples: 60 }], hmarks: [{ y: 1, color: '#555' }] }); },
                },
            ],
            notes: '拖 βm：v 越大 lobe 越宽；拖 θo 到 60° 以上看 Gaussian 的归一化偏差变大。中间图里 Gaussian 已除以 cosθi 以便和 M_p 同口径比较（M_p 的归一化测度是 cosθ dθ）。',
        },
        {
            id: 'shift', title: '角质层倾角 α：R 向发根、TT / TRT 向发梢', tag: 'SHIFT',
            desc: '发丝表面的角质层像屋瓦一样朝发梢翘起，倾角 α ≈ 2°–3°。瓦片表面的法线因此略向<strong>发根</strong>倾斜：R 的锥面整体偏向发根（θ<sub>h</sub> ≈ −α），而光穿过翘起的瓦片两次 / 三次，TT、TRT 偏向发梢（≈ +α/2、+2α）。结果就是屏幕上<strong>白色主高光和带颜色的第二高光分开</strong>。实现时最容易出错的是符号：它取决于切线指向发根还是发梢 —— pbrt-v3 的代码按其曲线切线施加 +α，若切线是"根 → 梢"，R 会被推到发梢一侧；本系列的 <code>HAIR.bsdf</code> 因此在内部取 −α。',
            formula: 'θ_o\' = θ_o − 2pα_p      α_R = −α,  α_TT = +α/2,  α_TRT = +2α      // 以 θ_h 计，t = 根 → 梢\nsin(θ_o − 2α) = sinθ_o cos2α − cosθ_o sin2α      // pbrt 用倍角递推一次算好 sin 2^k α',
            previews: [
                { id: 'shiftPeaks', label: '各 lobe 峰值 θh vs α（从 HAIR.bsdf 实测）', sub: '虚线 = 预期 −α、+α/2、+2α', size: [320, 180], render: (c, S) => PL.draw.plot(c, { x: [0, 6], y: [-8, 14], xLabel: 'α (°)', series: [0, 1, 2].map((p) => ({ data: S.peaks[p], x: [0, 1, 2, 3, 4, 5, 6], color: LOBE_COL[p], label: LOBE_NAME[p] })).concat([-1, 0.5, 2].map((k, p) => ({ fn: (a) => k * a, color: LOBE_COL[p], dash: [3, 3] }))), hmarks: [{ y: 0, color: '#555' }] }) },
                { id: 'shiftProf', label: '各 lobe 的纵向剖面（θo = 参数，各自归一化）', sub: '竖线 = 预期峰位 θi = 2θh − θo', size: [320, 180], render: (c, S) => PL.draw.plot(c, { x: [-90, 90], y: [0, 1.05], xLabel: 'θi (°)', series: [0, 1, 2].map((p) => ({ data: S.prof[p], x: S.profX, color: LOBE_COL[p], label: LOBE_NAME[p] })), marks: [-1, 0.5, 2].map((k, p) => ({ x: 2 * k * S.alpha - S.thetaO, label: LOBE_NAME[p], color: LOBE_COL[p] })) }) },
            ],
            warn: '换引擎 / 换资产时先确认切线方向（Groom 资产、插片的 flow 贴图、各家 shader 对"T 指向发根还是发梢"的约定不同），再决定 α 的符号；否则两条高光的上下位置会反过来，但单看一张截图很难发现。左图实测峰位与 −α、+α/2、+2α 的小偏差来自 lobe 宽度（cosθ 因子把宽 lobe 的峰拉向法平面）；βm 很大时 lobe 太宽，"峰位"本身就没有意义了。',
        },
        {
            id: 'nterm', title: '方位项 N_p：几何光学 → logistic（Chiang 2016）', tag: 'N',
            desc: 'N<sub>p</sub>(φ) 描述在法平面内出射的方位分布。对单个 h，光精确地沿 Φ(p, h) 出射；真实发丝截面不圆、表面粗糙，所以 Chiang et al. 2016 用截断 logistic 分布在 Φ 周围模糊，宽度由 β<sub>n</sub> 控制。远场（发丝比像素细）要对 h 积分：左图为单个 h，右图为 h 平均 —— TT 集中在背光方向 φ ≈ ±180°，TRT 在 φ ≈ 0 附近，R 很宽。',
            formula: 'N_p(φ; h) = logistic_trim(φ − Φ(p, h); s)      s = √(π/8) · (0.265 β_n + 1.194 β_n^2 + 5.372 β_n^{22})\nN̄_p(φ) = ½ ∫_{−1}^{1} A_p(h) N_p(φ; h) dh      // 远场：对发丝宽度积分',
            code: { title: 'HLSL · 截断 logistic 的 N_p', src: `
                float Logistic(float x, float s) { x = abs(x); float e = exp(-x / s); return e / (s * (1 + e) * (1 + e)); }
                float LogisticCDF(float x, float s) { return 1 / (1 + exp(-x / s)); }
                float Np(float phi, int p, float s, float gammaO, float gammaT) {
                    float dphi = phi - (2 * p * gammaT - 2 * gammaO + p * PI);
                    dphi = dphi - 2 * PI * floor((dphi + PI) / (2 * PI));            // 折回 [−π, π)
                    return Logistic(dphi, s) / (LogisticCDF(PI, s) - LogisticCDF(-PI, s));
                }` },
            previews: [
                { id: 'nSingle', label: 'N_p(φ)，单个 h = 参数', size: [320, 180], render: (c, S) => { const eta = S.hb.eta, g0 = Math.asin(S.hView), gt = Math.asin(S.hView / eta); PL.draw.plot(c, { x: [-180, 180], xLabel: 'φ (°)', series: [0, 1, 2].map((p) => ({ fn: (d) => HAIR.Np(d * Math.PI / 180, p, S.hb.s, g0, gt), color: LOBE_COL[p], label: LOBE_NAME[p], samples: 361 })) }); } },
                { id: 'nAvg', label: 'h 平均后的各 lobe（θi = θo = 0）', sub: '已含 A_p 与 M_p；对数纵轴', size: [320, 180], render: (c, S) => PL.draw.plot(c, { x: [-180, 180], xLabel: 'φ = φi − φo (°)', logY: true, y: [1e-3, 3], series: [0, 1, 2].map((p) => ({ data: S['az' + p], x: S.phis, color: LOBE_COL[p], label: LOBE_NAME[p] })) }) },
                { id: 'lobe3d', needs: ['lobe3d'], large: true, label: '3D：R / TT / TRT 的整体形状', sub: '固定入射方向（黄箭头，θi = −θo），画出每个出射方向上的散射强度（h 平均，64 个 h）；各 lobe 按自身最大值归一化。顶栏"显示 lobe"可只看一条', render: (c, S) => drawLobes3d(c, S) },
            ],
            notes: '拖 βn：logistic 变宽，TRT 的闪点被抹平；拖 eumelanin：TT、TRT 的整体高度下降（吸收在 A_p 里）。3D 图里转到沿发丝的方向看：R、TRT 是绕发丝的"喇叭口"（镜面锥被 M_p 加粗），TT 集中在光的正对面；拖 α 看 R 与 TRT 的喇叭口沿发丝错开。',
        },
        {
            id: 'absorb', title: '衰减 A_p：Fresnel + 吸收，黑色素决定发色', tag: 'ABSORB',
            desc: '每条路径的能量 = 入射 Fresnel 透射 (1 − f)，乘以每次穿过发丝内部的吸收 T = e<sup>−σa·2cosγt/cosθt</sup>，再乘以内部反射的 f：A₀ = f，A₁ = (1 − f)² T，A₂ = (1 − f)² f T²，p ≥ 3 的剩余项用几何级数合并。σ<sub>a</sub> 来自两种黑色素的浓度（真黑色素偏黑褐，褐黑色素偏红），这是 Chiang 2016 给美术的直观控制。',
            formula: 'f = F(cosθ_o · cosγ_o, η)      T = exp(−σ_a · 2cosγ_t / cosθ_t)\nA_0 = f      A_1 = (1 − f)^2 T      A_2 = (1 − f)^2 f T^2      A_{3+} = A_2 · f T / (1 − f T)\nσ_a = c_eu · (0.419, 0.697, 1.37) + c_pheo · (0.187, 0.4, 1.05)',
            previews: [
                { id: 'aCurve', label: 'A_p(h)（亮度，θo = 0）', size: [320, 180], render: (c, S) => PL.draw.plot(c, { x: [-1, 1], y: [0, 1], xLabel: 'h', series: [0, 1, 2, 3].map((p) => ({ fn: (h) => { const cGt = Math.sqrt(Math.max(0, 1 - (h / S.hb.eta) ** 2)), T = S.sigmaA.map((sa) => Math.exp(-sa * 2 * cGt)); return HAIR.lum(HAIR.Ap(1, S.hb.eta, h, T)[p]); }, color: LOBE_COL[p], label: LOBE_NAME[p], samples: 200 })) }) },
                { id: 'aStrip', label: '发色条：TT（上）与 TRT ×6（下）随真黑色素 0 → 3', sub: 'h = 0、θ = 0；白线 = 当前浓度，褐黑色素 = 参数', size: [320, 90], render: (c, S) => drawMelaninStrip(c, S) },
            ],
            notes: '浅色头发（低黑色素）的 TT、TRT 很强 —— 这时单次散射远不够，发丝之间的多次散射才是亮度的主体，见模块 03。',
        },
        {
            id: 'energy', title: '白炉：Chiang 守恒，实时近似与 Kajiya-Kay 不守恒', tag: 'ENERGY',
            desc: '无吸收（σ<sub>a</sub> = 0）时，能量守恒的发丝模型应满足 ∫ S dω<sub>i</sub> = 1，与视角无关。Chiang / pbrt 的模型做到了这一点：d\'Eon 的 M<sub>p</sub> 与截断 logistic N<sub>p</sub> 各自归一化，而 R、TT、TRT 加上 p ≥ 3 的剩余项后 Σ<sub>p</sub> A<sub>p</sub>(h) = 1 —— 对<strong>每一个</strong> h 都成立。实时近似把积分换成拟合，正视附近还接近 1，掠射时明显漏能量；Kajiya-Kay 则没有这个概念。',
            formula: 'E(θ_o) = ∫_{S^2} S(ω_i, ω_o) dω_i      // σ_a = 0；规则 (sinθ_i, φ) 网格，Chiang 对 4 个固定 h 分别积分再平均\nΣ_{p=0}^{3} A_p(h) = f + (1 − f)^2 [1 + f + f^2/(1 − f)] = 1      // T = 1 时逐 h 成立',
            previews: [{
                id: 'eFurn', needs: ['furnace'], label: '白炉 E(θo)', sub: '60 × 72 方向网格；Chiang 再对 4 个固定 h 平均', size: [380, 190],
                render: (c, S) => PL.draw.plot(c, { x: [0, 80], y: [0, 1.6], xLabel: 'θo (°)', series: [{ data: S.fChiang, x: S.fTheta, color: '#5ad07a', label: 'Chiang / pbrt-v3' }, { data: S.fUE, x: S.fTheta, color: '#6ab0e0', label: '实时近似（UE 风格）' }, { data: S.fKK, x: S.fTheta, color: '#d19a66', label: 'Kajiya-Kay' }], hmarks: [{ y: 1, color: '#555' }] }),
            }],
            notes: '默认参数下：Chiang 在所有 θo 上都是 1.000；实时近似在 θo ≤ 45° 时约 1.00–1.01，60° 起漏能量，80° 时只剩 ≈ 0.55。把 βm、βn 拖到 0.7 以上，实时近似连正视也只有 ≈ 0.92；拖到很小（≈ 0.1），Chiang 在掠射处出现 ±3% 的偏差——那是方向网格分辨率不够，不是模型的问题。这一步的积分较重，只在滚动到这里时计算（lazy）。',
        },
        {
            id: 'render', title: '发丝渲染：每像素随机 h + 逐帧累积', tag: 'RENDER',
            desc: '把散射函数用到成千上万根发丝上：共享光栅化器给出每个像素最近发丝的切线与覆盖率；h 在每帧、每像素随机取一个值，history 阶段逐帧取平均。这和路径追踪里"对发丝宽度随机采样"是同一个思路——比固定少量 h 的中点求和可靠得多：N<sub>p</sub>(φ; h) 是随 h 移动的窄 logistic，固定 h 会在某些 φ 上整片漏掉或重复计入（实测各 φ 上的最大相对误差：β<sub>n</sub> = 0.35 时 8 个 h ≈ 36%、64 个 h ≈ 4%；β<sub>n</sub> = 0.2 时 16 个 h 仍有 ≈ 96%）。随机 h 每帧有噪声，但均值无偏，累积后收敛到正确的远场结果。用顶部的"显示 lobe"只看某一条路径。',
            formula: 'L_pixel ≈ α · E · (1/N) Σ_frames S(ω_i, ω_o; h_k),   h_k ~ U(−1, 1)      // α = 亚像素覆盖率',
            code: { title: 'HLSL · 逐样本着色（随机 h，交给 TAA 累积）', src: `
                cbuffer HairMaterialCB { float Eumelanin, Pheomelanin, BetaM, BetaN, AlphaDeg; };   // = 顶栏参数

                // 每个材质只需算一次（CPU 端或常量初始化）：顶栏参数 → HairParams
                HairParams MaterialParams() { return MakeHairParams(SigmaAFromConcentration(Eumelanin, Pheomelanin), BetaM, BetaN, AlphaDeg); }

                float3 ShadeHairSample(float3 T, float3 L, float3 V, float3 E, HairParams P, uint2 pixel, uint frame) {
                    float  h  = 2.0 * Hash(pixel, frame) - 1.0;            // 每帧换一个 h
                    float3 wi = ToFiberFrame(L, T, V), wo = ToFiberFrame(V, T, V);
                    return HairBSDF_Chiang(wi, wo, h, P) * E;              // R + TT + TRT + 剩余
                }` },
            previews: [
                { id: 'rKK', label: 'Kajiya-Kay', smooth: true, render: (c, S) => drawRGB(c, S.kk) },
                { id: 'rCh', live: true, label: 'Chiang 2016（累积）', sub: '单次散射；显示 lobe = 参数', smooth: true, render: (c, S) => drawRGB(c, S.acc) },
            ],
            notes: '拖光源方位到 ±150° 以上（背光）：Chiang 的发梢一圈被 TT 点亮，Kajiya-Kay 没有这个效果。只看 TRT 时能看到偏发色、位置与 R 错开的第二高光。',
        },
        {
            id: 'rt', title: '实时近似：用闭式拟合代替对 h 的积分（UE / Karis 2016 风格）', tag: 'RT',
            desc: '实时着色负担不起对 h 的积分和 Bessel 函数。Karis 2016 一类的近似保留"R + TT + TRT"的结构，但：M 用以 sinθ<sub>i</sub> + sinθ<sub>o</sub> 为变量的 Gaussian；N<sub>R</sub> = ¼cos(φ/2)，N<sub>TT</sub>、N<sub>TRT</sub> 用 exp(a·cosφ + b) 拟合；Bravais 指数用 η′ ≈ 1.19/cosθ<sub>d</sub> + 0.36cosθ<sub>d</sub> 拟合；TT 的有效偏移 h 由 φ 估计。左下图把它的方位剖面和 Chiang 的 h 平均剖面放在一起。',
            formula: 'M ≈ g(B, sinθ_i + sinθ_o − α_p)      N_R = ¼ cos(φ/2)      N_TT ≈ e^{−3.65 cosφ − 3.98}      N_TRT ≈ e^{17 cosφ − 16.78}\nη\' ≈ 1.19 / cosθ_d + 0.36 cosθ_d      h_TT ≈ cos(φ/2) · (1 + (0.6 − 0.8 cosφ)/η\')',
            code: { title: 'HLSL · 实时毛发着色（结构示意，常数以引擎源码为准）', src: `
                float3 HairShadingRT(float3 L, float3 V, float3 T, float rough, float3 sigmaA, float alpha) {
                    float sinL = dot(T, L), sinV = dot(T, V), cosPhi = CosPhi(L, V, T);
                    float cosHalfPhi = sqrt(saturate(0.5 + 0.5 * cosPhi));
                    float cosThetaD  = cos(0.5 * abs(asin(sinV) - asin(sinL)));
                    float nP = 1.19 / cosThetaD + 0.36 * cosThetaD, a = 1.0 / nP, B = rough * rough;
                    float3 C = exp(-2.0 * sigmaA);                                   // 穿过一个直径的透射色

                    float  R   = Hair_G(B * sqrt(2.0) * cosHalfPhi, sinL + sinV + 2 * alpha) * 0.25 * cosHalfPhi
                               * Hair_F(sqrt(saturate(0.5 + 0.5 * dot(V, L))));
                    float  h   = cosHalfPhi * (1 + a * (0.6 - 0.8 * cosPhi));
                    float  fT  = Hair_F(cosThetaD * sqrt(saturate(1 - h * h)));
                    float3 TT  = Hair_G(B * 0.5, sinL + sinV - alpha) * exp(-3.65 * cosPhi - 3.98) * Pow2(1 - fT)
                               * pow(C, sqrt(saturate(1 - h * h * a * a)) / cosThetaD);
                    float  fR  = Hair_F(cosThetaD * 0.5);
                    float3 TRT = Hair_G(B * 2.0, sinL + sinV - 4 * alpha) * exp(17 * cosPhi - 16.78) * Pow2(1 - fR) * fR
                               * pow(C, 0.8 / cosThetaD);
                    return R + TT + TRT;
                }` },
            previews: [
                { id: 'rtProf', label: '方位剖面：实时近似（虚线） vs Chiang（实线）', sub: 'θi = θo = 0', size: [360, 190], render: (c, S) => PL.draw.plot(c, { x: [-180, 180], xLabel: 'φ (°)', logY: true, y: [1e-3, 3], series: [0, 1, 2].map((p) => ({ data: S['az' + p], x: S.phis, color: LOBE_COL[p], label: LOBE_NAME[p] })).concat([0, 1, 2].map((p) => ({ data: S['ue' + p], x: S.phis, color: LOBE_COL[p], dash: [4, 3] }))) }) },
                { id: 'rtImg', label: '实时近似着色', sub: '确定性，无需累积', smooth: true, render: (c, S) => drawRGB(c, S.uei) },
            ],
            notes: '默认参数下对照：TT 的背光峰与 Chiang 很接近；TRT 拟合在 φ = 0 处约高 2.6 倍、也更窄；虚线 R 在 φ → ±180° 抬升——实时近似的 R 纵向宽度 ∝ cos(φ/2)，在 θi = −θo 处收成尖峰（积分有限），而 Chiang 的 h 平均 R 几乎与 φ 无关。对比时实时近似的粗糙度取 r = v<sup>1/4</sup>，使 R 的纵向标准差与 d\'Eon lobe 相同。',
            warn: '这里的实现按公开的 Karis 2016 结构与常见引擎代码复现，衰减项改写成与本系列 σ<sub>a</sub> 一致的形式；具体常数和 TT/TRT 的颜色写法各版本引擎并不相同，移植时以目标引擎源码为准。',
        },
        {
            id: 'prod', title: '教学版 → 生产版：预计算、多次散射与常见坑', tag: 'PROD',
            desc: '<ul><li><strong>预计算</strong>：离线 / 高端实时常把远场 N̄<sub>p</sub>(φ, θ<sub>d</sub>; β<sub>n</sub>, η) 或整体散射函数烘成小 LUT，运行时只做查表 + 解析 M<sub>p</sub>。</li><li><strong>多次散射</strong>：单次散射模型对浅色头发严重偏暗（Step 07、09）；生产里一定要配合自阴影透射率与 dual scattering 一类的近似（模块 03）。</li><li><strong>采样</strong>：路径追踪按 A<sub>p</sub> 选 lobe，再分别采样 M<sub>p</sub>（解析逆）与 N<sub>p</sub>（logistic 逆 CDF），权重接近常数。</li><li><strong>插片</strong>：卡片上没有真实的 h 与截面，通常退回 Kajiya-Kay / Scheuermann 双高光，或把 h 平均后的模型直接用在 flow 切线上（模块 04）。</li></ul>',
            previews: [{ id: 'prodCmp', label: '同一光照下三种模型', sub: '左 KK · 中 实时近似 · 右 Chiang（累积）', size: [420, 170], live: true, render: (c, S) => drawTriptych(c, S) }],
            warn: '<strong>常见坑</strong>：① 切线方向与 α 符号不一致，主 / 次高光位置颠倒；② 用固定少量 h 求和代替远场积分，窄 β<sub>n</sub> 时闪点和 TT 严重失真；③ fp16 下 I₀ / exp 溢出 —— 小 v 必须用对数形式；④ 只做单次散射就调亮颜色"补"，结果颜色发灰；⑤ 把 Kajiya-Kay 的 p 当作物理粗糙度去和 β 对应；⑥ 忽略覆盖率 α，细发丝被画成实心线，整体过亮。',
        },
    ],
    graph: {
        intro: '散射函数的各项都是被着色 pass 内联调用的函数；真正的 pass 是"可见发丝样本 → 逐样本着色"，以及可选的 LUT 预计算。',
        types: {
            cb: { color: '#e8c25a', label: 'cbuffer · 毛发参数' },
            vis: { color: '#4a8aff', label: 'samples · 切线 / 覆盖率' },
            fn: { color: '#c678dd', label: 'fn · 内联函数' },
            lut: { color: '#d19a66', label: 'LUT · 远场 N̄ / 多次散射' },
            light: { color: '#9aa5b1', label: 'light · 透射率（模块 03）' },
            radiance: { color: '#5ad07a', label: 'radiance · HDR' },
        },
        common: {
            title: 'Common.hlsl · 系列共享 + 本模块函数',
            blocks: [
                { label: '━ 系列共享 ━', code: HAIR.HLSL_COMMON },
                { label: '━ 系列共享 · 毛发 BSDF（HLSL 参考实现）━', code: HAIR.HLSL_BSDF },
            ],
        },
        nodes: [
            { id: 'cb', title: 'HairMaterialCB', tag: 'CB', category: 'user', desc: 'βm · βn · α · 黑色素 → σa', outputs: [{ name: 'HairCB', format: 'cbuffer', type: 'cb' }], detail: { preview: 'aStrip' } },
            { id: 'samples', title: 'HairVisibleSamples', tag: 'RT', category: 'system', desc: '切线 · 覆盖率（模块 02）', outputs: [{ name: 'HairSamples', format: 'per-pixel', type: 'vis' }], detail: { preview: 'geomCov' } },
            { id: 'trans', title: 'HairTransmittance', tag: 'RT', category: 'system', desc: '自阴影透射率（模块 03）', outputs: [{ name: 'Transmittance', format: 'R16F', type: 'light' }], detail: { preview: null } },
            { id: 'mp', title: 'Hair_Mp', tag: 'FN', step: 'mterm', dispatch: { kind: 'INLINE', note: 'd\'Eon 2011，对数形式' }, inputs: [{ name: 'HairCB', type: 'cb' }], outputs: [{ name: 'MpFn', format: 'M_p(θi, θo)', type: 'fn' }], detail: { preview: 'mCurve' } },
            { id: 'np', title: 'Hair_Np', tag: 'FN', step: 'nterm', dispatch: { kind: 'INLINE', note: 'Chiang 2016 截断 logistic' }, inputs: [{ name: 'HairCB', type: 'cb' }], outputs: [{ name: 'NpFn', format: 'N_p(φ; h)', type: 'fn' }], detail: { preview: 'nAvg' } },
            { id: 'ap', title: 'Hair_Ap', tag: 'FN', step: 'absorb', dispatch: { kind: 'INLINE', note: 'Fresnel + 吸收' }, inputs: [{ name: 'HairCB', type: 'cb' }], outputs: [{ name: 'ApFn', format: 'A_p(θo, h)', type: 'fn' }], detail: { preview: 'aCurve' } },
            { id: 'lut', title: 'CS_BakeHairLUT', tag: 'CS', category: 'bake', lifetime: 'param', step: 'prod', desc: '可选：远场 N̄ / 多次散射表', dispatch: { name: 'CS_BakeHairLUT', threads: [8, 8, 1], idRange: 'ID.xy = (φ, θd) 表格' }, inputs: [{ name: 'NpFn', type: 'fn' }, { name: 'ApFn', type: 'fn' }], outputs: [{ name: 'HairLUT', format: 'R16F 64×32', type: 'lut' }], detail: { preview: null } },
            {
                id: 'shade', title: 'CS_HairShading', tag: 'CS', category: 'output', lifetime: 'frame', step: 'render', desc: '逐样本 R + TT + TRT（随机 h）',
                dispatch: { name: 'CS_HairShading', threads: [8, 8, 1], idRange: '每个可见发丝样本' },
                inputs: [{ name: 'HairSamples', type: 'vis' }, { name: 'MpFn', type: 'fn' }, { name: 'NpFn', type: 'fn' }, { name: 'ApFn', type: 'fn' }, { name: 'HairLUT', type: 'lut' }, { name: 'Transmittance', type: 'light' }],
                outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'radiance' }],
                detail: { preview: 'rCh' },
            },
        ],
    },
    references: [
        { title: 'Rendering Fur with Three Dimensional Textures', author: 'J. Kajiya, T. Kay', venue: 'SIGGRAPH', year: 1989 },
        { title: 'Light Scattering from Human Hair Fibers', author: 'S. Marschner, H. W. Jensen, M. Cammarano, S. Worley, P. Hanrahan', venue: 'SIGGRAPH', year: 2003 },
        { title: 'An Energy-Conserving Hair Reflectance Model', author: 'E. d\'Eon, G. Francois, M. Hill, J. Letteri, J.-M. Aubry', venue: 'EGSR', year: 2011 },
        { title: 'A Practical and Controllable Hair and Fur Model for Production Path Tracing', author: 'M. J.-Y. Chiang, B. Bitterli, C. Tappan, B. Burley', venue: 'Computer Graphics Forum (Eurographics)', year: 2016 },
        { title: 'The Implementation of a Hair Scattering Model（pbrt-v3 hair.cpp 的配套说明）', author: 'M. Pharr', venue: 'pbrt.org', year: 2016 },
        { title: 'Physically Based Hair Shading in Unreal', author: 'B. Karis', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2016 },
    ],
});

// ── preview drawing helpers (read-only views of stage data) ──
function drawRGB(c, buf) { PL.draw.pixels(c, SH.W, SH.H, (x, y, o) => { const i = y * SH.W + x, d = HAIR.display([buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]]); o[0] = d[0]; o[1] = d[1]; o[2] = d[2]; }); }
function drawTriptych(c, S) {
    const { ctx, w, h } = PL.draw.ctx(c), off = drawTriptych.off || (drawTriptych.off = document.createElement('canvas'));
    [S.kk, S.uei, S.acc].forEach((buf, k) => { drawRGB(off, buf); const pw = w / 3 - 6, ph = pw * SH.H / SH.W; ctx.drawImage(off, k * (w / 3) + 3, (h - ph) / 2, pw, ph); });
}
function drawCone(c, S) {
    const { ctx, w, h } = PL.draw.ctx(c), cx = w * 0.42, cy = h * 0.55, tO = S.thetaO * Math.PI / 180;
    ctx.strokeStyle = '#8a7a5a'; ctx.lineWidth = 7; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(cx - 120, cy + 60); ctx.lineTo(cx + 120, cy - 60); ctx.stroke();
    const t = HAIR.norm([240, -120, 0]), n = [t[1], -t[0]];
    ctx.lineWidth = 1.5; ctx.strokeStyle = '#e8c25a';
    const L = [Math.sin(-tO) * t[0] + Math.cos(-tO) * n[0], Math.sin(-tO) * t[1] + Math.cos(-tO) * n[1]];
    ctx.beginPath(); ctx.moveTo(cx + L[0] * 90, cy + L[1] * 90); ctx.lineTo(cx, cy); ctx.stroke();
    ctx.strokeStyle = 'rgba(106,176,224,0.8)';
    for (let k = 0; k < 9; k++) { const ph = k / 8 * Math.PI; const dir = [Math.sin(tO) * t[0] + Math.cos(tO) * Math.cos(ph) * n[0], Math.sin(tO) * t[1] + Math.cos(tO) * Math.cos(ph) * n[1]]; ctx.globalAlpha = 0.25 + 0.75 * Math.abs(Math.cos(ph)); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dir[0] * 80, cy + dir[1] * 80); ctx.stroke(); }
    ctx.globalAlpha = 1; ctx.fillStyle = '#999'; ctx.font = '10px Consolas, monospace';
    ctx.fillText('t（根 → 梢）', cx + 70, cy - 60); ctx.fillText(`θi = ${(-S.thetaO).toFixed(0)}°  →  θo = ${S.thetaO.toFixed(0)}°`, 8, 14); ctx.fillText('镜面锥：θo = −θi', 8, h - 8);
}
function drawPaths(c, S) {
    const { ctx, w, h } = PL.draw.ctx(c), cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.3, eta = etaOf(S.hb.eta, S.thetaO * Math.PI / 180);
    ctx.strokeStyle = '#3a4148'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    ctx.fillStyle = 'rgba(140,110,70,0.12)'; ctx.fill();
    const P = SH.tracePaths(S.hView, eta), X = (p) => cx + p[0] * R, Y = (p) => cy - p[1] * R;
    P.forEach((path, p) => {
        ctx.strokeStyle = LOBE_COL[p]; ctx.lineWidth = 1.6; ctx.beginPath();
        path.pts.forEach((q, i) => (i ? ctx.lineTo(X(q), Y(q)) : ctx.moveTo(X(q), Y(q))));
        const e = path.pts[path.pts.length - 1];
        if (path.dir) ctx.lineTo(X(e) + path.dir[0] * R * 1.4, Y(e) - path.dir[1] * R * 1.4);
        ctx.stroke();
    });
    ctx.fillStyle = '#999'; ctx.font = '10px Consolas, monospace'; ctx.fillText(`h = ${S.hView.toFixed(2)}   η′(θo = ${S.thetaO}°) = ${eta.toFixed(3)}`, 8, 14);
    LOBE_NAME.slice(0, 3).forEach((n, p) => { ctx.fillStyle = LOBE_COL[p]; ctx.fillText('— ' + n, 8, h - 40 + p * 13); });
}
function drawMelaninStrip(c, S) {
    const Wd = 160, Hd = 2;
    PL.draw.pixels(c, Wd, Hd, (x, y, o) => {
        const ce = 3 * x / (Wd - 1), sa = HAIR.sigmaAFromConcentration(ce, S.pheo), cGt = 1, T = sa.map((s) => Math.exp(-s * 2 * cGt));
        const a = HAIR.Ap(1, 1.55, 0, T)[y === 0 ? 1 : 2], k = y === 0 ? 1.0 : 6.0, d = HAIR.display(a.map((v) => v * k));
        const mark = Math.abs(ce - S.melanin) < 0.02;
        o[0] = mark ? 1 : d[0]; o[1] = mark ? 1 : d[1]; o[2] = mark ? 1 : d[2];
    });
}
// ── 3D views ──
function drawGeom3d(c, S) {
    const V3 = PL.v3, T = V3.norm([1, 0.3, 0]), n = [0, 0, 1], b = V3.cross(T, n), thO = S.thetaO * Math.PI / 180, thI = -thO;
    const L = V3.add(V3.mul(T, Math.sin(thI)), V3.mul(n, Math.cos(thI)));          // toward the light
    const dirOn = (ph) => V3.add(V3.mul(T, Math.sin(thO)), V3.mul(V3.add(V3.mul(n, Math.cos(ph)), V3.mul(b, Math.sin(ph))), Math.cos(thO)));
    const rays = [];
    for (let k = 1; k < 16; k++) rays.push({ pts: [[0, 0, 0], V3.mul(dirOn(2 * Math.PI * k / 16), 1.25)], color: 'rgba(106,176,224,0.55)', width: 1 });
    PL.draw.mesh3d(c, {
        meshes: [
            PL.mesh.cylinder({ a: V3.mul(T, -1.7), b: V3.mul(T, 1.7), r: 0.07, rgb: [0.62, 0.48, 0.32], gloss: 24 }),
            PL.mesh.cone({ apex: [0, 0, 0], axis: T, angle: 90 - S.thetaO, length: 1.25, res: [32, 4], surface: false, wire: true, wireColor: 'rgba(106,176,224,0.45)' }),
        ],
        lines: rays.concat([
            { pts: [V3.mul(L, 1.6), [0, 0, 0]], color: '#ffd866', width: 2.4, arrow: true, labelAt: 'start', label: 'L（θi = ' + (-S.thetaO) + '°）' },
            { pts: [[0, 0, 0], V3.mul(dirOn(0), 1.5)], color: '#6ab0e0', width: 2.6, arrow: true, label: '镜面方向 θo = ' + S.thetaO + '°' },
        ]),
        labels: [{ p: V3.mul(T, 1.85), text: 'T（根 → 梢）', color: '#d0a878' }],
        orbit: { yaw: 0.45, pitch: 0.42, dist: 5.8 },
    });
}
function drawLobes3d(c, S) {
    const nu = L3D.nu, nv = L3D.nv, NU = nu + 1, sel = S.lobe === 'all' ? [0, 1, 2] : [Number(S.lobe)];
    const rgb = [[0.91, 0.76, 0.35], [0.88, 0.42, 0.46], [0.35, 0.82, 0.48]];
    const dir = (i, j) => { const ph = 2 * Math.PI * i / nu, th = -Math.PI / 2 + Math.PI * j / nv; return [Math.sin(th), Math.cos(th) * Math.cos(ph), Math.cos(th) * Math.sin(ph)]; };
    const meshes = sel.map((p) => {
        const R = S['l3d' + p], m = S.l3dMax[p].m;
        return PL.mesh.param((u, v) => { const i = Math.round(u * nu), j = Math.round(v * nv), r = 1.3 * R[j * NU + i] / m, d = dir(i, j); return [d[0] * r, d[1] * r, d[2] * r]; }, [nu, nv], { closedU: true, rgb: rgb[p], gloss: 28 });
    });
    meshes.push(PL.mesh.cylinder({ a: [-1.6, 0, 0], b: [1.6, 0, 0], r: 0.045, rgb: [0.55, 0.42, 0.28] }));
    const thI = -S.thetaO * Math.PI / 180, wi = [Math.sin(thI), Math.cos(thI), 0];
    const labels = sel.map((p) => { const at = S.l3dMax[p].at, d = dir(at % NU, Math.floor(at / NU)); return { p: PL.v3.mul(d, 1.42), text: LOBE_NAME[p], color: LOBE_COL[p] }; });
    labels.push({ p: [1.7, 0, 0], text: 'T', color: '#d0a878' });
    PL.draw.mesh3d(c, { meshes, lines: [{ pts: [PL.v3.mul(wi, 1.7), [0, 0, 0]], color: '#ffd866', width: 2.2, arrow: true, labelAt: 'start', label: 'ωi（光）' }], labels, orbit: { yaw: 1.05, pitch: 0.38, dist: 5.2 } });
}
