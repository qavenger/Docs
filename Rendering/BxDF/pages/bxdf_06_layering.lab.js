// BxDF 06 · 多层混合与叠加 —— build with the shared helpers first:
//   node <skill>/scripts/build_doc.mjs bxdf_common.js bxdf_06_layering.lab.js -o bxdf_06_layering.html
// Conventions: see bxdf_common.js (n = +Z, wi → light, wo → viewer, α = r², η = η_t/η_i, linear RGB).

const L6 = (() => {
    const INV_PI = BX.INV_PI, Z3 = [0, 0, 0], N_MU = 48;
    const muAt = (i) => 0.02 + 0.98 * i / (N_MU - 1);
    const lookup = (tab, mu) => {   // linear interpolation in a μ-table built with muAt
        const x = BX.clamp((mu - 0.02) / 0.98, 0, 1) * (N_MU - 1), i = Math.min(N_MU - 2, Math.floor(x)), t = x - i;
        return tab[i] * (1 - t) + tab[i + 1] * t;
    };
    const avg = (tab) => { let s = 0, w = 0; for (let i = 0; i < N_MU; i++) { const m = muAt(i); s += tab[i] * m; w += m; } return s / w; };   // ≈ 2∫E μ dμ / 1
    // GGX reflection lobe of a dielectric interface (coat / base specular)
    const ggxSpec = (r, ior) => {
        const a = BX.alpha(r);
        return (wi, wo) => {
            if (wi[2] <= 0 || wo[2] <= 0) return 0;
            const h = BX.half(wi, wo);
            return BX.fresnelDielectric(BX.dot(wo, h), ior) * BX.ggxD(h, a) * BX.smithG2(wo, wi, a) / (4 * wi[2] * wo[2]);
        };
    };
    // Disney clearcoat: GTR1 distribution (Burley 2012), α_c = lerp(0.1, 0.001, gloss); fixed F0 = 0.04; Smith G with α = 0.25
    const gtr1D = (cosH, a) => { if (a >= 1) return INV_PI; const a2 = a * a; return (a2 - 1) / (Math.PI * Math.log(a2) * (1 + (a2 - 1) * cosH * cosH)); };
    const disneyCoat = (gloss) => {
        const ac = 0.1 + (0.001 - 0.1) * gloss;
        return (wi, wo) => {
            if (wi[2] <= 0 || wo[2] <= 0) return 0;
            const h = BX.half(wi, wo), F = BX.schlick(0.04, BX.dot(wo, h));
            return F * gtr1D(h[2], ac) * BX.smithG2(wo, wi, 0.25) / (4 * wi[2] * wo[2]);
        };
    };
    // directional albedo table of a GGX dielectric reflection lobe via VNDF sampling (weight = F · G2/G1)
    const specTable = (r, ior, n = 384) => {
        const a = BX.alpha(r), out = new Float32Array(N_MU);
        for (let i = 0; i < N_MU; i++) {
            const mu = muAt(i), wo = [Math.sqrt(1 - mu * mu), 0, mu];
            out[i] = BX.albedoMC((w, u1, u2) => { const h = BX.sampleVNDF(w, a, a, u1, u2), l = BX.reflect(w, h); return l[2] > 0 ? { wi: l, weight: BX.fresnelDielectric(BX.dot(w, h), ior) * BX.smithG2(w, l, a) / BX.smithG1(w, a) } : null; }, wo, n);
        }
        return out;
    };
    // hemispherical average of the inside Fresnel (includes TIR): F̄_in = 2∫ F(μ, 1/η) μ dμ
    const fresnelAvgInside = (ior) => { let s = 0; const n = 400; for (let i = 0; i < n; i++) { const m = (i + 0.5) / n; s += BX.fresnelDielectric(m, 1 / ior) * m; } return 2 * s / n; };
    const fresnelAvgOutside = (ior) => { let s = 0; const n = 400; for (let i = 0; i < n; i++) { const m = (i + 0.5) / n; s += BX.fresnelDielectric(m, ior) * m; } return 2 * s / n; };
    const cosT = (mu, ior) => Math.sqrt(Math.max(0, 1 - (1 - mu * mu) / (ior * ior)));   // refracted cosine inside the coat
    // coat transmittance for one crossing: tint^(thick / μt)  (tint = transmittance at normal incidence for thick = 1)
    const transmit = (tint, thick, mu, ior) => { const p = thick / Math.max(1e-3, cosT(mu, ior)); return tint.map((t) => Math.pow(Math.max(1e-4, t), p)); };

    // ── layering rules: coat lobe c(wi, wo) (scalar), base b(wi, wo) (RGB) → layered f (RGB) ──
    const rules = {
        add: (P) => (wi, wo) => { const c = P.coat(wi, wo), b = P.base(wi, wo); return [b[0] + c, b[1] + c, b[2] + c]; },
        filament: (P) => (wi, wo) => {   // Filament: base · (1 − Fc(v·h)) + coat
            const c = P.coat(wi, wo), b = P.base(wi, wo), k = 1 - BX.fresnelDielectric(BX.dot(wo, BX.half(wi, wo)), P.ior);
            return [b[0] * k + c, b[1] * k + c, b[2] * k + c];
        },
        fresnel2: (P) => (wi, wo) => {   // base · (1 − F(μi)) · (1 − F(μo)) + coat
            const c = P.coat(wi, wo), b = P.base(wi, wo), k = (1 - BX.fresnelDielectric(wi[2], P.ior)) * (1 - BX.fresnelDielectric(wo[2], P.ior));
            return [b[0] * k + c, b[1] * k + c, b[2] * k + c];
        },
        albedo: (P) => (wi, wo) => {   // albedo scaling, view-only: base · (1 − E_c(μo)) + coat
            const c = P.coat(wi, wo), b = P.base(wi, wo), k = Math.max(0, 1 - lookup(P.Ec, wo[2]));
            return [b[0] * k + c, b[1] * k + c, b[2] * k + c];
        },
        albedoMin: (P) => (wi, wo) => {   // reciprocal variant: base · min(1 − E_c(μi), 1 − E_c(μo)) + coat
            const c = P.coat(wi, wo), b = P.base(wi, wo), k = Math.max(0, Math.min(1 - lookup(P.Ec, wi[2]), 1 - lookup(P.Ec, wo[2])));
            return [b[0] * k + c, b[1] * k + c, b[2] * k + c];
        },
        // first-order refracted model (Weidlich-Wilkie structure): (1−Fi)(1−Fo)/η² · T(μi) T(μo) · f_base(wi', wo') — no inter-reflection
        ww: (P) => (wi, wo) => {
            const c = P.coat(wi, wo), b = P.baseIn(wi, wo), k = (1 - BX.fresnelDielectric(wi[2], P.ior)) * (1 - BX.fresnelDielectric(wo[2], P.ior)) / (P.ior * P.ior);
            const ti = transmit(P.tint, P.thick, wi[2], P.ior), to = transmit(P.tint, P.thick, wo[2], P.ior);
            return [0, 1, 2].map((ch) => b[ch] * k * ti[ch] * to[ch] + c);
        },
        // smooth coat + Lambert base with all internal reflections (geometric series); exact for a smooth, clear coat
        smooth: (P) => (wi, wo) => {
            const c = P.coat(wi, wo), k = (1 - BX.fresnelDielectric(wi[2], P.ior)) * (1 - BX.fresnelDielectric(wo[2], P.ior)) / (P.ior * P.ior * Math.PI);
            const ti = transmit(P.tint, P.thick, wi[2], P.ior), to = transmit(P.tint, P.thick, wo[2], P.ior);
            return [0, 1, 2].map((ch) => { const rho = P.rho[ch], Tm = P.Tavg[ch]; return rho * k * ti[ch] * to[ch] / (1 - rho * P.FinAvg * Tm * Tm) + c; });
        },
    };
    // total albedo of a rule for white base (or the given ρ): coat part from the table, base part by quadrature
    const ruleAlbedo = (name, P, mu) => {
        const wo = [Math.sqrt(1 - mu * mu), 0, mu], Pb = Object.assign({}, P, { coat: () => 0 });
        const base = BX.albedoQuad((wi) => BX.lum(rules[name](Pb)(wi, wo)), wo, 28);
        return lookup(P.Ec, mu) + base;
    };

    // ── ground truth: random walk between a rough dielectric coat (top) and a Lambert base (bottom) ──
    // Plane-parallel layers ⇒ lateral position is irrelevant (the "position-free" view): only directions and which interface we are at.
    const flip = (w) => [w[0], w[1], -w[2]];
    const walk = (winc, P, rnd, maxEvents = 48) => {
        const a = BX.alpha(P.coatRough), eta = P.ior, out = [0, 0, 0];
        let thr = [1, 1, 1];
        // 1) arrival from outside
        let h = BX.sampleVNDF(winc, a, a, rnd(), rnd());
        let F = BX.fresnelDielectric(BX.dot(winc, h), eta);
        if (rnd() < F) {
            const l = BX.reflect(winc, h);
            if (l[2] > 0) { const g = BX.smithG2(winc, l, a) / BX.smithG1(winc, a); return [g, g, g]; }
            return out;
        }
        let t = BX.refract(winc, h, eta);
        if (!t || t[2] >= 0) return out;
        let g = BX.smithG2(winc, t, a) / BX.smithG1(winc, a);
        thr = thr.map((v) => v * g);
        for (let ev = 0; ev < maxEvents; ev++) {
            // 2) travel down to the base, absorb, diffuse bounce
            const down = P.tint.map((c) => Math.pow(Math.max(1e-4, c), P.thick / Math.max(1e-3, -t[2])));
            thr = thr.map((v, ch) => v * down[ch] * P.rho[ch]);
            const u = BX.sampleCosine(rnd(), rnd());
            const up = P.tint.map((c) => Math.pow(Math.max(1e-4, c), P.thick / Math.max(1e-3, u[2])));
            thr = thr.map((v, ch) => v * up[ch]);
            if (Math.max(thr[0], thr[1], thr[2]) < 1e-4) return out;
            // 3) hit the coat from inside (work in the flipped frame where the inside is "up")
            const wf = flip([-u[0], -u[1], -u[2]]);   // points back into the layer, z > 0 in the flipped frame
            h = BX.sampleVNDF(wf, a, a, rnd(), rnd());
            F = BX.fresnelDielectric(BX.dot(wf, h), 1 / eta);
            if (rnd() < F) {   // internal reflection: back down
                const l = BX.reflect(wf, h);
                if (l[2] <= 0) return out;
                g = BX.smithG2(wf, l, a) / BX.smithG1(wf, a);
                thr = thr.map((v) => v * g);
                t = flip(l);   // travel direction in the real frame (z < 0)
                continue;
            }
            const e = BX.refract(wf, h, 1 / eta);
            if (!e || e[2] >= 0) return out;
            g = BX.smithG2(wf, e, a) / BX.smithG1(wf, a);
            return thr.map((v) => v * g);   // escaped to the outside
        }
        return out;
    };
    return { N_MU, muAt, lookup, avg, ggxSpec, gtr1D, disneyCoat, specTable, fresnelAvgInside, fresnelAvgOutside, cosT, transmit, rules, ruleAlbedo, walk };
})();

// Material bundle shared by stages and previews
const mat = (S, over) => {
    const P = {
        ior: S.coatIor, coatRough: S.coatRough, rho: S.baseColor, tint: S.coatTint, thick: S.thick,
        Ec: S.Ec, FinAvg: S.FinAvg, Tavg: S.Tavg,
    };
    P.coat = L6.ggxSpec(S.coatRough, S.coatIor);
    P.base = (wi, wo) => (wi[2] > 0 && wo[2] > 0 ? P.rho.map((c) => c * BX.INV_PI) : [0, 0, 0]);
    P.baseIn = P.base;   // a Lambert base is the same for refracted directions
    return Object.assign(P, over || {});
};
const thetaW = (S) => BX.dir(S.thetaI * Math.PI / 180, Math.PI);
const lights6 = (S) => BX.lightsFor(S.thetaI * Math.PI / 180, S.lightAz);   // series convention: pitch = θi, yaw = 光源方位
const sph = (rule, over) => (c, S) => BX.renderSphere(c, L6.rules[rule](mat(S, over)), { res: 128, lights: lights6(S) });
const RULE_LABEL = { add: '直接相加', filament: 'Filament：(1−F_c(v·h))', fresnel2: '(1−F(μi))(1−F(μo))', albedo: 'albedo scaling (1−E_c(μo))', albedoMin: 'min(1−E_c(μi), 1−E_c(μo))', ww: '一阶折射模型（W-W 结构）', smooth: '光滑解析（含内部反射）' };
const RULE_COLOR = { add: '#e06c75', filament: '#d19a66', fresnel2: '#e8c25a', albedo: '#5ad07a', albedoMin: '#56b6c2', ww: '#c678dd', smooth: '#6ab0e0' };
const L6_3D = { n: 9, H: 0.55, B: 1.5 };   // 3D 光路：路径数、示意层厚、显示盒半宽（局部单位）

const HLSL_LAYER = `
    // ── BxDF 06 · 分层工具 ──
    float CoatE(float NoV, float coatRough) { return CoatEnergyLUT.SampleLevel(samLinearClamp, float2(NoV, coatRough), 0); }

    float3 LayerAlbedoScaled(float3 fBase, float3 fCoat, float NoV, float coatRough) {
        return fCoat + fBase * saturate(1.0 - CoatE(NoV, coatRough));        // 视线方向的 albedo scaling
    }
    float3 CoatTransmittance(float3 tint, float thickness, float NoX, float eta) {
        float cosT = sqrt(saturate(1.0 - (1.0 - NoX * NoX) / (eta * eta)));  // 折射后的 cos
        return pow(max(tint, 1e-4), thickness / max(cosT, 1e-3));            // Beer-Lambert，路径长 = d / cosθt
    }`;

PipelineLab.mount({
    paramSteps: { thetaI: ['mix', 'coat', 'fresnel', 'smooth', 'absorb', 'ww', 'albedo', 'uber'], lightAz: ['mix', 'coat', 'fresnel', 'smooth', 'absorb', 'ww', 'albedo', 'uber'], baseColor: ['mix', 'coat', 'fresnel', 'smooth', 'absorb', 'ww', 'albedo', 'uber'], coatRough: ['coat', 'fresnel', 'smooth', 'absorb', 'ww', 'albedo', 'mc', 'uber', 'prod'], coatIor: ['coat', 'fresnel', 'smooth', 'absorb', 'ww', 'albedo', 'mc', 'uber', 'prod'], coatTint: ['smooth', 'ww', 'mc'], thick: ['absorb'], metal: ['mix'], sheenW: ['uber'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: 'BxDF 06 · 多层混合与叠加：Clearcoat → 分层 BSDF → 材质总装',
    lang: 'zh-CN',
    codeLang: 'hlsl',
    seed: 606,
    animated: false,
    tagColors: { MIX: 'gray', COAT: 'rust', FRESNEL: 'olive', SMOOTH: 'blue', ABSORB: 'magenta', WW: 'indigo', ALBEDO: 'green', MC: 'teal', UBER: 'teal', PROD: 'gray' },
    params: [
        { id: 'thetaI', label: 'θi', min: 0, max: 85, step: 1, value: 50, unit: '°', title: '入射角（波瓣、曲线用）；球体渲染里 = 主光俯仰角',
            desc: '入射角 θi（光在 x–z 入射面内）：Step 02 的 GTR1 / GGX 对比波瓣和 Step 04 的 3D 层间光路用它。拖到 75° 以上：涂层 Fresnel 接近 1，大部分光在涂层表面就被反射走，进入涂层的光变少。在球体渲染里它是主光的<strong>俯仰角</strong>（从视线方向向上抬 θi；偏航由光源方位控制，方位 = 0° 时球心处的入射角就是 θi）。' },
        { id: 'lightAz', label: '光源方位', min: -170, max: 170, step: 1, value: -35, unit: '°', desc: '球体渲染里主光的<strong>偏航角</strong>：绕竖直轴转动主光（0° = 光在视线正上方的竖直面内，正 = 右侧，±170° ≈ 逆光），俯仰由 θi 决定；轮廓光、补光跟着主光转。只影响球体渲染（含顶部的四种规则对比），波瓣与曲线始终在入射面内。' },
        { id: 'baseColor', label: 'base', value: '#a8322a', map: BX.hex2lin, title: 'Lambert 底层反照率 ρ（sRGB 输入）',
            desc: 'Lambert 底层反照率 ρ（sRGB 输入，内部转线性）。它决定涂层下颜色变深、变饱和的程度：ρ 小的通道被内部反射级数压得更狠（Step 04 的饱和度曲线与 3D 光路的颜色）；白炉曲线固定用 ρ = 1，不随它变。' },
        { id: 'coatRough', label: 'coat r', min: 0.02, max: 0.8, step: 0.01, value: 0.12, title: '涂层感知粗糙度（GGX α = r²）',
            desc: '涂层感知粗糙度 r（GGX α = r²）。越大 → 涂层高光越宽、E<sub>coat</sub> 的正面值被抬高（Step 07 的 LUT）、宏观 Fresnel 规则越偏离随机游走参考（Step 03 / 08）。拖到 0.5 以上看所有闭式规则与 MC 拉开差距；光滑解析解只在 r 很小时精确。' },
        { id: 'coatIor', label: 'coat η', min: 1.2, max: 2.0, step: 0.01, value: 1.5, digits: 2, title: '涂层折射率',
            desc: '涂层折射率 η：决定 F0、内侧平均反射率 F̄<sub>in</sub>（η = 1.5 时约 60%，大部分是全内反射）和出射时的 1/η² 立体角压缩。拖大看涂层下颜色更深更饱和、光在层间反弹更多次（Step 04），以及 E<sub>coat</sub> LUT 整体变亮（Step 07）。' },
        { id: 'coatTint', label: 'coat tint', value: '#ffffff', map: BX.hex2lin, title: '涂层颜色 = 垂直穿过一次（厚度 1）的透射率；白 = 透明',
            desc: '涂层颜色 = 厚度 1 时垂直穿过一次的透射率，白 = 透明。进入光滑解析、一阶折射模型和随机游走里的吸收项（Step 04 / 06 / 08；Step 05 用固定的琥珀色作示例）。调成橙色：掠射处颜色更深，每次层间往返都再吸收一次。白炉曲线固定用透明涂层。' },
        { id: 'thick', label: '厚度', min: 0, max: 3, step: 0.05, value: 1, title: '涂层光学厚度倍数',
            desc: '涂层光学厚度倍数 d：吸收路径长 = d / cosθ<sub>t</sub>，进、出各一段。0 = 不吸收；拖到 2–3 看 Step 05 的透射率曲线与琥珀球变深——折射把角度压向法线，变化比按入射角算温和得多。顶栏 tint 为白（透明）时，它只影响 Step 05 的琥珀示例。' },
        { id: 'metal', label: 'metalness', min: 0, max: 1, step: 0.01, value: 0.5, title: 'Step 01 的线性混合权重',
            desc: 'Step 01 线性混合的金属度 m：f = (1 − m)·f<sub>dielectric</sub> + m·f<sub>metal</sub>。拖动看混合球和白炉曲线始终夹在两个端点之间——凸组合不需要任何能量修正。只影响 Step 01。' },
        { id: 'sheenW', label: 'sheen', min: 0, max: 1, step: 0.01, value: 0.35, title: 'Step 09 uber 材质的 sheen 权重',
            desc: 'Step 09 uber 材质最外层 sheen 的权重。拖大看顶层在掠射角拿走更多能量（sheen 的 E(μ) 在掠射最大），下面各层在球的边缘一起变暗；能量去向图里每条带随之重新分配。只影响 Step 09。' },
    ],
    stages: [
        {
            id: 'coatLUT', lifetime: 'param', run(S) {
                S.Ec = L6.specTable(S.coatRough, S.coatIor);
                S.EcAvg = L6.avg(S.Ec);
                S.FinAvg = L6.fresnelAvgInside(S.coatIor);
                S.FoutAvg = L6.fresnelAvgOutside(S.coatIor);
                // cosine-weighted average transmittance of one crossing (used in the internal-reflection series)
                S.Tavg = [0, 1, 2].map((ch) => { let s = 0; const n = 64; for (let i = 0; i < n; i++) { const m = (i + 0.5) / n; s += Math.pow(Math.max(1e-4, S.coatTint[ch]), S.thick / Math.max(1e-3, m)) * 2 * m; } return s / n; });
            },
        },
        {
            id: 'lut2d', lifetime: 'param', run(S) {   // E_coat(μ, r) for the current IOR — what an engine bakes once per IOR preset
                const R = 32, img = PL.buf(S, 'lut2d', R * R);
                for (let y = 0; y < R; y++) {
                    const r = 0.02 + 0.98 * (R - 1 - y) / (R - 1), a = BX.alpha(r);
                    for (let x = 0; x < R; x++) {
                        const mu = 0.02 + 0.98 * x / (R - 1), wo = [Math.sqrt(1 - mu * mu), 0, mu];
                        img[y * R + x] = BX.albedoMC((w, u1, u2) => { const h = BX.sampleVNDF(w, a, a, u1, u2), l = BX.reflect(w, h); return l[2] > 0 ? { wi: l, weight: BX.fresnelDielectric(BX.dot(w, h), S.coatIor) * BX.smithG2(w, l, a) / BX.smithG1(w, a) } : null; }, wo, 96);
                    }
                }
            },
        },
        {
            id: 'furnace', lifetime: 'param', run(S) {   // white-base furnace curves of every rule (clear coat)
                const P = mat(S, { rho: [1, 1, 1], tint: [1, 1, 1], Tavg: [1, 1, 1] });
                S.mus = Float32Array.from({ length: 24 }, (_, i) => 0.03 + 0.97 * i / 23);
                S.furn = {};
                for (const k of ['add', 'filament', 'fresnel2', 'albedo', 'albedoMin', 'ww', 'smooth']) S.furn[k] = Float32Array.from(S.mus, (mu) => L6.ruleAlbedo(k, P, mu));
            },
        },
        {
            id: 'mc', lifetime: 'param', lazy: true, run(S) {   // random-walk reference albedo vs μ (white base, current coat roughness / tint)
                const rnd = PL.rng(S.seed ^ 0x51ed), P = mat(S, { rho: [1, 1, 1] }), n = 1600;
                S.mcMu = Float32Array.from({ length: 12 }, (_, i) => 0.05 + 0.95 * i / 11);
                S.mcE = Float32Array.from(S.mcMu, (mu) => {
                    const w = [Math.sqrt(1 - mu * mu), 0, mu]; let s = 0;
                    for (let k = 0; k < n; k++) s += BX.lum(L6.walk(w, P, rnd));
                    return s / n;
                });
                // same for a smooth clear coat to cross-check the analytic internal-reflection model
                const Pc = mat(S, { rho: [0.5, 0.5, 0.5], coatRough: 0.02, tint: [1, 1, 1] });
                S.mcSmoothHalf = Float32Array.from(S.mcMu, (mu) => { const w = [Math.sqrt(1 - mu * mu), 0, mu]; let s = 0; for (let k = 0; k < n; k++) s += BX.lum(L6.walk(w, Pc, rnd)); return s / n; });
            },
        },
        {
            id: 'walk3d', lifetime: 'param', lazy: true, run(S) {   // a few Step 08 walks with positions for the 3D view + bounce statistics (white base, clear coat)
                const rnd = PL.rng((S.seed ^ 0x3d06) >>> 0), wi = thetaW(S), P = mat(S), Pw = mat(S, { rho: [1, 1, 1], tint: [1, 1, 1] }), N = 4000;
                const paths = Array.from({ length: L6_3D.n }, () => layerPath(wi, P, rnd, L6_3D.H));
                let coat = 0, out = 0, bounces = 0;
                for (let k = 0; k < N; k++) { const r = layerPath(wi, Pw, rnd, L6_3D.H, true); if (r.kind === 'coat') coat++; else if (r.kind === 'out') { out++; bounces += r.bounces; } }
                S.walk3d = { paths, coat: coat / N, meanB: out ? bounces / out : 0 };
            },
        },
        {
            id: 'uber', lifetime: 'param', run(S) {   // albedo tables for the uber stack: sheen (Charlie + Neubelt), base specular (GGX η 1.5)
                const a = 0.25, sh = new Float32Array(L6.N_MU);
                for (let i = 0; i < L6.N_MU; i++) {
                    const mu = L6.muAt(i), wo = [Math.sqrt(1 - mu * mu), 0, mu];
                    sh[i] = BX.albedoQuad((wi) => { if (wi[2] <= 0) return 0; const h = BX.half(wi, wo), s = Math.sqrt(Math.max(0, 1 - h[2] * h[2])); return (2 + 1 / a) * Math.pow(s, 1 / a) / (2 * Math.PI) / (4 * (wi[2] + wo[2] - wi[2] * wo[2])); }, wo, 32);
                }
                S.Esheen = sh;
                S.Espec = L6.specTable(0.45, 1.5, 256);
            },
        },
    ],
    library: {
        title: '附录 · 辅助函数（HLSL）',
        intro: 'Step 04 / 07 / 08 的代码调用的采样、遮蔽、Fresnel 与随机数函数，以及随机游走用到的材质常量。约定与系列一致：局部系 n = +Z，方向都指离表面，η = η<sub>t</sub>/η<sub>i</sub>，α = r²；<code>Lambda_GGX</code>、<code>PI</code> 来自 Common。',
        blocks: [
            {
                title: '材质常量 · 采样 · Smith 遮蔽（Step 07 / 08）', code: `
                // Step 07 烘焙与 Step 08 随机游走用到的材质常量（引擎里来自材质 cbuffer）
                cbuffer LayerWalkCB : register(b2) {
                    float  aCoat;      // 涂层 GGX α = r²
                    float  eta;        // 涂层折射率（外 → 涂层的 η_t/η_i）
                    float  CoatEta;    // 烘焙 E_coat LUT 用的 η（每个 η 预设一张表）
                    float  d;          // 涂层光学厚度倍数
                    float3 tint;       // 厚度 1、垂直穿过一次的透射率
                    float  padL0;
                    float3 rho;        // Lambert 底层反照率（线性）
                    float  padL1;
                };

                // Hammersley 点集的第 i 个点（共 N 个）：x = (i + ½)/N，y = 以 2 为底的 radical inverse（位反转）；与 BX.hammersley 相同
                float2 Hammersley(uint i, uint N) { return float2((i + 0.5) / N, reversebits(i) * 2.3283064365386963e-10); }

                // Heitz 2018 可见法线采样：返回微表面法线 H，pdf(H) = G1(V)·max(0, V·H)·D(H) / V.z（V 在局部系上半球）
                float3 SampleGGXVNDF(float3 V, float a, float2 u) {
                    float3 Vh = normalize(float3(a * V.x, a * V.y, V.z));             // 拉伸成 α = 1 的半球
                    float  l2 = Vh.x * Vh.x + Vh.y * Vh.y;
                    float3 T1 = l2 > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(l2) : float3(1.0, 0.0, 0.0);
                    float3 T2 = cross(Vh, T1);
                    float  r  = sqrt(u.x), phi = 2.0 * PI * u.y;
                    float  t1 = r * cos(phi), t2 = r * sin(phi);
                    float  s  = 0.5 * (1.0 + Vh.z);
                    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;                     // 按投影面积压扁圆盘的一半
                    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
                    return normalize(float3(a * Nh.x, a * Nh.y, max(0.0, Nh.z)));    // 反拉伸回 α
                }

                // 余弦加权半球采样（Malley）：pdf = cosθ / π；与 BX.sampleCosine 相同
                float3 SampleCosine(float2 u) {
                    float r = sqrt(u.x), phi = 2.0 * PI * u.y;
                    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
                }

                // GGX Smith 遮蔽 G1 = 1 / (1 + Λ(ω))，α = aCoat；Λ 只依赖 cos²θ，涂层内外两侧通用
                float SmithG1(float3 w) { return 1.0 / (1.0 + Lambda_GGX(abs(w.z), aCoat)); }
                // 高度相关的遮蔽-阴影 G2 = 1 / (1 + Λ(ωa) + Λ(ωb))（Heitz 2014），反射、透射两种配置都用它
                float SmithG2(float3 wa, float3 wb) { return 1.0 / (1.0 + Lambda_GGX(abs(wa.z), aCoat) + Lambda_GGX(abs(wb.z), aCoat)); }`,
            },
            {
                title: 'Fresnel · 随机数（Step 04 / 07 / 08）', code: `
                // 非偏振介电 Fresnel（与 BX.fresnelDielectric 相同）：cosI = 入射侧余弦，eta = η_t/η_i；cosI < 0 表示从另一侧入射；全内反射 → 1
                float FresnelDielectric(float cosI, float eta) {
                    if (cosI < 0.0) { eta = 1.0 / eta; cosI = -cosI; }
                    cosI = min(cosI, 1.0);
                    float sin2T = (1.0 - cosI * cosI) / (eta * eta);
                    if (sin2T >= 1.0) return 1.0;
                    float cosT  = sqrt(1.0 - sin2T);
                    float rPar  = (eta * cosI - cosT) / (eta * cosI + cosT);
                    float rPerp = (cosI - eta * cosT) / (cosI + eta * cosT);
                    return 0.5 * (rPar * rPar + rPerp * rPerp);
                }

                // 每条路径一个 PCG 随机数状态
                struct RNG { uint state; };

                // PCG-RXS-M-XS 32 位：推进 LCG 状态并返回 [0, 1) 的均匀数（取高 24 位，保证严格 < 1）
                float Rand(inout RNG rng) {
                    rng.state = rng.state * 747796405u + 2891336453u;
                    uint w = ((rng.state >> ((rng.state >> 28u) + 4u)) ^ rng.state) * 277803737u;
                    w = (w >> 22u) ^ w;
                    return float(w >> 8) * (1.0 / 16777216.0);
                }
                // 两个独立的均匀数（VNDF / 余弦采样用）
                float2 Rand2(inout RNG rng) { float a = Rand(rng); return float2(a, Rand(rng)); }`,
            },
        ],
    },
    intro: {
        title: '多层混合与叠加 · 把各类 lobe 组成一个真实材质',
        flow: ['线性混合', 'clearcoat 相加', 'Fresnel 让能量', '光滑解析', '吸收涂层', '一阶折射模型', 'albedo scaling', '随机游走参考', 'uber 总装'],
        html: `<p>前五个模块各讲一种 lobe；真实材质是它们的组合。组合有两种本质不同的方式：<strong>混合</strong>（同一点上按比例取两种材质，权重和为 1）与<strong>叠加</strong>（一层盖在另一层上，光要先穿过上层）。本模块把叠加拆开：先看最朴素的"直接相加"为什么凭空产生能量，再一步步加入 Fresnel、内部反射、吸收、折射方向，最后用一个随机游走参考检验各种近似，并按 OpenPBR 式的层序总装 uber 材质。</p>
<p>全页统一：底层 = Lambert（颜色 ρ），涂层 = GGX 电介质反射（粗糙度、η 可调）。白炉曲线都用 ρ = 1 的白色底层，看总反照率是否 ≤ 1。</p>${BX.nav('layering')}`,
    },
    sections: [{
        id: 'hero', title: '同一涂层、同一底层 · 四种叠加规则', position: 'top',
        sub: '从左到右：直接相加（掠射处发亮）· (1−F)(1−F) · albedo scaling · 光滑解析（含内部反射，颜色更深更饱和）。改 base / coat 参数四个球同时重算。',
        html: `<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;max-width:860px">${['add', 'fresnel2', 'albedo', 'smooth'].map((k, i) => `<div style="text-align:center"><canvas class="h${i}" style="width:100%;aspect-ratio:1"></canvas><div style="font:0.68rem Consolas,monospace;color:#888;margin-top:4px">${RULE_LABEL[k]}</div></div>`).join('')}</div>`,
        frame(el, S) { ['add', 'fresnel2', 'albedo', 'smooth'].forEach((k, i) => BX.renderSphere(el.querySelector('.h' + i), L6.rules[k](mat(S)), { res: 144, lights: lights6(S) })); },
    }],
    steps: [
        {
            id: 'mix', title: '线性混合 vs 物理叠加：metalness 是"混合"', tag: 'MIX',
            desc: '金属度工作流里的 <code>metalness</code> 不是一层盖一层，而是<strong>在同一像素里按面积比例混合</strong>两种材质：f = (1 − m)·f<sub>dielectric</sub> + m·f<sub>metal</sub>。只要每个分量各自守恒，凸组合（权重和为 1）就自动守恒，不需要任何能量修正。叠加则不同：上层反射掉的能量到不了下层，下层出射的光还要再穿过上层——这才是后面所有步骤要处理的问题。',
            formula: 'f_mix = (1 − m) · f_dielectric + m · f_metal      // 凸组合：E_mix = (1 − m) E_d + m E_m ≤ 1\nf_layer = f_top + (穿过上层的那部分) · f_base · (再穿出上层)      // 叠加：耦合项才是难点',
            code: { title: 'HLSL · metalness 混合', src: `
                float3 EvalMetalnessMix(float3 baseColor, float metalness, float rough, float NoL, float NoV, float NoH, float VoH) {
                    float  a    = rough * rough;
                    float  spec = D_GGX(NoH, a) * V_SmithGGXCorrelated(NoV, NoL, a);
                    float3 Fm   = F_Schlick(baseColor, VoH);                 // 金属：F0 = baseColor，无漫反射
                    float3 Fd   = F_Schlick(0.04, VoH);                      // 电介质：F0 = 0.04
                    float3 dielectric = baseColor * INV_PI * (1.0 - Fd) + spec * Fd;
                    float3 metal      = spec * Fm;
                    return lerp(dielectric, metal, metalness);               // 同一点上的面积混合
                }` },
            previews: [
                { id: 'mixD', label: '电介质 (m = 0)', smooth: true, render: (c, S) => BX.renderSphere(c, mixBRDF(S, 0), { res: 128, lights: lights6(S) }) },
                { id: 'mixM', label: '金属 (m = 1)', smooth: true, render: (c, S) => BX.renderSphere(c, mixBRDF(S, 1), { res: 128, lights: lights6(S) }) },
                { id: 'mixX', label: '混合 (m = 参数)', sub: '凸组合', smooth: true, render: (c, S) => BX.renderSphere(c, mixBRDF(S, S.metal), { res: 128, lights: lights6(S) }) },
                { id: 'mixE', label: '白炉：E_mix 恒介于两者之间', size: [300, 160], render: (c, S) => { const mus = S.mus; const e = (m) => Array.from(mus, (mu) => BX.lum(BX.albedoQuad((wi) => mixBRDF(S, m, [1, 1, 1])(wi, [Math.sqrt(1 - mu * mu), 0, mu]), [Math.sqrt(1 - mu * mu), 0, mu], 24))); PL.draw.plot(c, { x: [0, 1], y: [0, 1.1], xLabel: 'μ = cosθo', series: [{ data: e(0), x: mus, label: 'm = 0', color: '#6ab0e0' }, { data: e(1), x: mus, label: 'm = 1', color: '#e0a060' }, { data: e(S.metal), x: mus, label: `m = ${S.metal.toFixed(2)}`, color: '#5ad07a' }], hmarks: [{ y: 1, color: '#555' }] }); } },
            ],
            notes: '白炉曲线用白色 baseColor：混合曲线永远夹在两条端点曲线之间。低于 1 的部分来自单次散射 GGX 的能量损失（见模块 02 的多次散射补偿），不是混合造成的。',
        },
        {
            id: 'coat', title: 'Clearcoat 直接相加：凭空多出的能量', tag: 'COAT',
            desc: 'Disney 2012 的 clearcoat 是一个额外的镜面 lobe：GTR1 分布（尾巴比 GGX 更长）、固定 F0 = 0.04（η = 1.5）、粗糙度只在很窄的范围里由 <code>clearcoatGloss</code> 控制。最省事的做法是把它直接加到底层上：<code>f = f_coat + f_base</code>。问题是涂层反射走的能量底层又全部用了一遍——在掠射角 Fresnel 接近 1 时，总反照率明显超过 1。',
            formula: 'D_GTR1(h) = (α^2 − 1) / (π ln(α^2) (1 + (α^2 − 1) cos^2 θ_h))      α = lerp(0.1, 0.001, gloss)\nf_add = f_coat + f_base      // 白炉：E_add(μ) = E_coat(μ) + ρ ≥ 1',
            code: { title: 'HLSL · Disney clearcoat（GTR1）与直接相加', src: `
                float D_GTR1(float NoH, float a) {
                    if (a >= 1.0) return INV_PI;
                    float a2 = a * a;
                    return (a2 - 1.0) / (PI * log(a2) * (1.0 + (a2 - 1.0) * NoH * NoH));
                }
                float3 ClearcoatAdditive(float3 fBase, float gloss, float NoL, float NoV, float NoH, float VoH) {
                    float a  = lerp(0.1, 0.001, gloss);
                    float Fc = F_Schlick(0.04, VoH).x;
                    float fc = Fc * D_GTR1(NoH, a) * V_SmithGGXCorrelated(NoV, NoL, 0.25);
                    return fBase + fc;                                         // ← 不守恒
                }` },
            previews: [
                { id: 'coatLobe', label: 'GTR1 vs GGX（对数）', sub: '同宽度下 GTR1 尾巴更长', size: [260, 160], render: (c, S) => BX.lobe(c, { wi: thetaW(S), cos: true, log: 4, series: [{ fn: (wo) => L6.disneyCoat(0.7)(thetaW(S), wo), color: '#e0a060', label: 'GTR1 · gloss 0.7' }, { fn: (wo) => L6.ggxSpec(0.18, 1.5)(thetaW(S), wo), color: '#6ab0e0', label: 'GGX · r 0.18', dash: [4, 3] }] }) },
                { id: 'coatAdd', label: '直接相加', smooth: true, render: sph('add') },
                { id: 'coatFurn', label: '白炉：直接相加 vs 守恒上限', sub: 'ρ = 1；虚线 = 1', size: [300, 160], render: (c, S) => PL.draw.plot(c, { x: [0, 1], y: [0.8, 1.6], xLabel: 'μ = cosθo', series: [{ data: S.furn.add, x: S.mus, color: RULE_COLOR.add, label: '直接相加' }], hmarks: [{ y: 1, color: '#666' }] }) },
            ],
            warn: '直接相加的超出量正好等于涂层的方向反照率 E<sub>coat</sub>(μ)：正面只多 4% 左右，看不出来；掠射角能多出 30%–60%，表现为边缘一圈不自然的"发光"。',
        },
        {
            id: 'fresnel', title: '用 Fresnel 给底层让出能量', tag: 'FRESNEL',
            desc: '第一种修正：底层只拿到没被涂层反射的那部分。两种常见写法——Filament 用微表面上的 <code>F_c(v·h)</code> 衰减一次；更"物理"一点的写法认为光进入、离开涂层各穿过界面一次，用宏观法线上的 (1 − F(μ<sub>i</sub>))(1 − F(μ<sub>o</sub>))。它们都把总反照率拉回到 1 附近，但只考虑了界面的<strong>镜面</strong>反射：粗糙涂层把能量散到别的方向，宏观 Fresnel 并不等于涂层真正拿走的能量。',
            formula: 'f_Filament = f_coat + f_base · (1 − F_c(v·h))\nf_2F = f_coat + f_base · (1 − F(μ_i)) · (1 − F(μ_o))',
            code: { title: 'HLSL · Filament 风格的 clearcoat', src: `
                float3 ClearcoatFilament(float3 Fd, float3 Fr, float clearCoat, float clearCoatRough, float NoL, float NoV, float NoH, float LoH) {
                    float a  = clearCoatRough * clearCoatRough;
                    float Fc = F_Schlick(0.04, LoH).x * clearCoat;
                    float Frc = D_GGX(NoH, a) * V_SmithGGXCorrelated(NoV, NoL, a) * Fc;
                    return (Fd + Fr) * (1.0 - Fc) + Frc;                       // 底层只衰减一次（v·h 上的 Fresnel）
                }` },
            previews: [
                { id: 'fFil', label: 'Filament (1−F_c)', smooth: true, render: sph('filament') },
                { id: 'f2F', label: '(1−F(μi))(1−F(μo))', smooth: true, render: sph('fresnel2') },
                { id: 'fFurn', label: '白炉：三种写法', size: [320, 170], render: (c, S) => PL.draw.plot(c, { x: [0, 1], y: [0.7, 1.4], xLabel: 'μ = cosθo', series: ['add', 'filament', 'fresnel2'].map((k) => ({ data: S.furn[k], x: S.mus, color: RULE_COLOR[k], label: RULE_LABEL[k] })), hmarks: [{ y: 1, color: '#666' }] }) },
            ],
            notes: '拖 coat r：涂层越粗糙，宏观 Fresnel 与涂层真实反照率的差距越大，两种 Fresnel 写法的白炉曲线越偏离 1。',
        },
        {
            id: 'smooth', title: '光滑涂层 + Lambert 底：带内部反射的解析解', tag: 'SMOOTH',
            desc: '光滑、透明的涂层下面是 Lambert 底时可以精确求解。进入涂层的能量 (1 − F(μ<sub>i</sub>)) 被底层漫反射；向上的漫射光有 F̄<sub>in</sub> 被界面反射回来（η = 1.5 时约 60%，大部分是全内反射），再被底层反射……这是一个几何级数。出射辐亮度还要乘 1/η²（离开致密介质时立体角展开）。结果：涂层下的颜色<strong>更深、更饱和</strong>——这正是"东西湿了颜色更深"的原因（Lekner &amp; Dorf 1988）。',
            formula: 'f = (1 − F(μ_i)) (1 − F(μ_o)) · ρ / (π η^2 (1 − ρ F̄_in))      F̄_in = 2 ∫_0^1 F(μ, 1/η) μ dμ\n白炉（ρ = 1）：E(μ) = F(μ) + (1 − F(μ)) · ρ (1 − F̄_in) / (1 − ρ F̄_in) = 1      // η^2 (1 − F̄_in) = 1 − F̄_out',
            code: { title: 'HLSL · 光滑涂层下的漫反射（含内部反射级数）', src: `
                float3 CoatedLambertSmooth(float3 rho, float eta, float NoL, float NoV, float FinAvg) {
                    float Fi = FresnelDielectric(NoL, eta), Fo = FresnelDielectric(NoV, eta);
                    return (1.0 - Fi) * (1.0 - Fo) * rho * INV_PI / (eta * eta * (1.0 - rho * FinAvg));
                }   // FinAvg 只依赖 η：常量或 1D LUT` },
            previews: [
                { id: 'smDry', label: '无涂层 Lambert', smooth: true, render: (c, S) => BX.renderSphere(c, (wi, wo) => (wi[2] > 0 && wo[2] > 0 ? S.baseColor.map((v) => v * BX.INV_PI) : [0, 0, 0]), { res: 128, lights: lights6(S) }) },
                { id: 'smWet', label: '光滑涂层（解析）', sub: '更深、更饱和', smooth: true, render: sph('smooth', null) },
                {
                    id: 'smCheck', needs: ['mc'], label: '解析解 vs 随机游走（ρ = 0.5，光滑涂层）', sub: '实线 = 解析；点 = MC', size: [320, 170],
                    render: (c, S) => {
                        const rho = 0.5, eta = S.coatIor, Fin = S.FinAvg;
                        const fn = (mu) => BX.fresnelDielectric(mu, eta) + (1 - BX.fresnelDielectric(mu, eta)) * rho * (1 - Fin) / (1 - rho * Fin);
                        PL.draw.plot(c, { x: [0, 1], y: [0, 1], xLabel: 'μ', series: [{ fn, color: '#6ab0e0', label: '解析', samples: 120 }, { data: S.mcSmoothHalf, x: S.mcMu, color: '#e8c25a', label: 'MC', bar: false, width: 0.01, dash: [1, 6] }], marks: [] });
                        const { ctx, w, h } = PL.draw.ctx(c, { keep: true, smooth: true });
                        const X = (v) => 46 + v * (w - 56), Y = (v) => h - 32 - v * (h - 42);
                        ctx.fillStyle = '#e8c25a'; S.mcMu.forEach((m, i) => { ctx.beginPath(); ctx.arc(X(m), Y(S.mcSmoothHalf[i]), 2.5, 0, 7); ctx.fill(); });
                    },
                },
                { id: 'smSat', label: '饱和度：ρ → 涂层下的有效反照率', sub: 'ρ_eff = ρ(1 − F̄_in)/(1 − ρF̄_in)；虚线 = 无涂层', size: [300, 170], render: (c, S) => PL.draw.plot(c, { x: [0, 1], y: [0, 1], xLabel: 'ρ', series: [{ fn: (r) => r * (1 - S.FinAvg) / (1 - r * S.FinAvg), color: '#5ad07a', label: `η = ${S.coatIor.toFixed(2)}` }, { fn: (r) => r, color: '#666', dash: [4, 3], label: 'ρ' }] }) },
                {
                    id: 'layer3d', needs: ['walk3d'], large: true, label: '3D：涂层与底层之间来回反弹的光',
                    sub: '同一入射点（黄箭头，θi），Step 08 随机游走的 9 条路径。线色 = 当前吞吐量的颜色，亮度随能量衰减：每碰一次底层就乘一次 ρ，越弹越深、越饱和。蓝色线框 = 涂层界面，平面 = Lambert 底层（层厚只是示意：平行层里横向位置不影响结果）；超出盒子的部分截掉',
                    render: (c, S) => drawLayer3d(c, S),
                },
            ],
            notes: '右图：ρ 小的颜色分量被压得更厉害（曲线在低 ρ 段远低于对角线），所以三个通道的比例被拉开——颜色变饱和。η 越大 F̄<sub>in</sub> 越大，效果越强。MC 点由本页 Step 08 的随机游走算出，和解析解吻合，说明两边的模型一致。3D 图把级数画成了真实的光路：进入涂层的光平均要碰底层 1/(1 − F̄<sub>in</sub>) 次（η = 1.5 时 ≈ 2.5 次）才逃出，每碰一次乘一次 ρ——分母 1 − ρF̄<sub>in</sub> 就是这么来的。拖 coat η 看反弹次数变多。',
        },
        {
            id: 'absorb', title: '有色涂层：Beer-Lambert 与折射路径长度', tag: 'ABSORB',
            desc: '有颜色的透明漆（汽车漆、指甲油、琥珀）在涂层里吸收。光在涂层里走的路径长 = 厚度 / cosθ<sub>t</sub>，θ<sub>t</sub> 是<strong>折射后</strong>的角度，进、出各一段。掠射时路径更长，颜色更深——但因为折射把角度压向法线，变化比"按入射角算"温和得多。',
            formula: 'T(μ) = tint^{d / cos θ_t}      cos θ_t = √(1 − (1 − μ^2)/η^2)\nf_base,coated = T(μ_i) · T(μ_o) · (…透射项…) · f_base      // tint = 厚度 1 时垂直穿过一次的透射率',
            code: { title: 'HLSL · 涂层吸收', src: HLSL_LAYER },
            previews: [
                { id: 'abSph', label: '有色涂层（tint = 琥珀）', sub: '光滑解析 + 吸收', smooth: true, render: (c, S) => BX.renderSphere(c, L6.rules.smooth(mat(S, { tint: BX.hex2lin('#e0a050'), Tavg: [0, 1, 2].map((ch) => avgT(BX.hex2lin('#e0a050')[ch], S.thick)) })), { res: 128, lights: lights6(S) }) },
                { id: 'abPlot', label: '单次穿过的透射率 vs 入射角', sub: '实线 = 按折射角；虚线 = 错误地按入射角', size: [320, 170], render: (c, S) => { const tint = BX.hex2lin('#e0a050'); PL.draw.plot(c, { x: [0, 89], y: [0, 1], xLabel: 'θ (°)', series: [0, 1, 2].map((ch) => ({ fn: (t) => L6.transmit(tint, S.thick, Math.cos(t * Math.PI / 180), S.coatIor)[ch], color: ['#e06c75', '#5ad07a', '#6ab0e0'][ch], label: 'RGB'[ch] })).concat([0, 1, 2].map((ch) => ({ fn: (t) => Math.pow(tint[ch], S.thick / Math.max(1e-3, Math.cos(t * Math.PI / 180))), color: ['#e06c75', '#5ad07a', '#6ab0e0'][ch], dash: [3, 3] }))) }); } },
            ],
            warn: '用入射角而不是折射角算路径长，是有色涂层最常见的错误：掠射角会黑掉。另外吸收也要进入内部反射级数（每次往返再乘一次 T）——本页解析式用余弦加权平均的 T̄ 近似这一项。',
        },
        {
            id: 'ww', title: '一阶折射模型（Weidlich-Wilkie 2007 的结构）', tag: 'WW',
            desc: 'Weidlich &amp; Wilkie 把"涂层 + 任意底层 BRDF"写成：涂层反射 + 透射进去（1 − F）→ 在<strong>折射后的方向</strong>上求底层 BRDF → 吸收 → 透射出来（1 − F）·1/η²。它能处理任意多层、任意底层，代价是只算<strong>一次</strong>往返：底层向上的光被界面反射回去的那部分（上一步的几何级数）丢掉了。对 Lambert 底层，丢掉的正好是 1 − (1 − F̄<sub>in</sub>) 的那一大块。',
            formula: 'f_WW ≈ f_coat + (1 − F(μ_i)) (1 − F(μ_o)) / η^2 · T(μ_i) T(μ_o) · f_base(ω_i^{′}, ω_o^{′})      // ′ = 折射后的方向\n白炉（Lambert, ρ = 1）：E_WW(μ) = F(μ) + (1 − F(μ)) (1 − F̄_in) < 1',
            previews: [
                { id: 'wwSph', label: '一阶折射模型', smooth: true, render: sph('ww') },
                { id: 'wwFurn', label: '白炉：一阶 vs 含内部反射', size: [320, 170], render: (c, S) => PL.draw.plot(c, { x: [0, 1], y: [0, 1.1], xLabel: 'μ = cosθo', series: ['ww', 'smooth'].map((k) => ({ data: S.furn[k], x: S.mus, color: RULE_COLOR[k], label: RULE_LABEL[k] })), hmarks: [{ y: 1, color: '#666' }] }) },
            ],
            notes: '对光泽底层（例如涂层下面还有一层 GGX），在折射方向上求值会把底层高光"压窄"并移向法线，这是一阶模型真正有价值的地方；对 Lambert 底层，它只剩下一个能量偏暗的问题。',
            warn: '这里只展示该模型的核心结构；原论文还讨论了多层递推与粗糙界面的处理。不要把"一阶"模型直接用在高反照率底层上——会系统性偏暗。',
        },
        {
            id: 'albedo', title: 'Albedo scaling：通用的叠加规则 + E(μ, r) LUT', tag: 'ALBEDO',
            desc: '实时与离线渲染器（Imageworks、Autodesk Standard Surface、OpenPBR、glTF 扩展）最常用的通用规则：<strong>上层在该方向的方向反照率 E<sub>top</sub>(μ) 被拿走，剩下的 1 − E<sub>top</sub>(μ) 交给下层</strong>。E<sub>top</sub> 预先按 (μ, r) 烘焙成一张 LUT（下图，依赖 η）。只按视线方向缩放最便宜、对任意上层 lobe 都能保证白炉 ≤ 1，但不互易；用 min(1 − E(μ<sub>i</sub>), 1 − E(μ<sub>o</sub>)) 可以恢复互易。',
            formula: 'f = f_top + f_base · (1 − E_top(μ_o))      // 视线方向 albedo scaling\nf = f_top + f_base · min(1 − E_top(μ_i), 1 − E_top(μ_o))      // 互易变体\nE_top(μ) = ∫ f_top(ω_i, ω_o) cos θ_i dω_i      // 烘焙成 LUT(μ, r)',
            code: { title: 'HLSL · 烘焙 E_coat(μ, r)', src: `
                RWTexture2D<float> CoatEnergyLUT;            // 32 × 32，x = μ，y = 粗糙度
                [numthreads(8, 8, 1)]
                void CS_BakeCoatEnergy(uint3 id : SV_DispatchThreadID) {
                    float mu = (id.x + 0.5) / 32.0, r = (id.y + 0.5) / 32.0, a = r * r;
                    float3 V = float3(sqrt(1.0 - mu * mu), 0.0, mu);
                    float E = 0.0;
                    for (uint i = 0; i < 256; ++i) {
                        float2 u = Hammersley(i, 256);
                        float3 H = SampleGGXVNDF(V, a, u);
                        float3 L = reflect(-V, H);
                        if (L.z <= 0.0) continue;
                        float G1 = 1.0 / (1.0 + Lambda_GGX(V.z, a)), G2 = 1.0 / (1.0 + Lambda_GGX(V.z, a) + Lambda_GGX(L.z, a));
                        E += FresnelDielectric(dot(V, H), CoatEta) * G2 / G1;   // VNDF 采样：权重 = F · G2/G1
                    }
                    CoatEnergyLUT[id.xy] = E / 256.0;
                }` },
            previews: [
                { id: 'lut', label: 'E_coat(μ, r) LUT', sub: 'x = μ（左 0 → 右 1），y = r（下 0 → 上 1）', render: (c, S) => PL.draw.field(c, S.lut2d, 32, 32, { map: 'inferno', range: [0, 0.6] }), probe: (x, y, S) => `μ = ${(0.02 + 0.98 * x / 31).toFixed(2)}  r = ${(0.02 + 0.98 * (31 - y) / 31).toFixed(2)}\nE = ${S.lut2d[y * 32 + x].toFixed(3)}` },
                { id: 'alSph', label: 'albedo scaling', smooth: true, render: sph('albedo') },
                { id: 'alFurn', label: '白炉：两种 albedo scaling', size: [320, 170], render: (c, S) => PL.draw.plot(c, { x: [0, 1], y: [0.7, 1.1], xLabel: 'μ = cosθo', series: ['albedo', 'albedoMin', 'fresnel2'].map((k) => ({ data: S.furn[k], x: S.mus, color: RULE_COLOR[k], label: RULE_LABEL[k], dash: k === 'fresnel2' ? [4, 3] : null })), hmarks: [{ y: 1, color: '#666' }] }) },
            ],
            notes: 'LUT 的掠射一侧（右下→左下）亮：Fresnel 在掠射角趋于 1。粗糙度越大，正面也被抬高。拖 coat η 看整张表变亮——同一份 LUT 只对一个 η 有效，引擎通常只为 η = 1.5 烘一张。',
        },
        {
            id: 'mc', title: 'Ground truth：两层之间的随机游走', tag: 'MC',
            desc: '无限大平行层里，横向位置无关紧要——只需要跟踪方向和"在哪个界面"（position-free 的思路，Guo et al. 2018）。从外部射入：在粗糙涂层上按 VNDF 采样微表面法线，按 Fresnel 选反射或折射；折射进去后按 cosθ<sub>t</sub> 吸收，到底层 Lambert 漫反射，向上再吸收，在界面内侧再按 Fresnel 选择透射出去或反射回去（含全内反射）……直到逃逸。白色底层下它的总反照率就是"真正的"E(μ)，可以拿来检验所有近似。',
            formula: '权重（VNDF + 按 F 选分支）：w ← w · G_2(ω, ω^{′}) / G_1(ω)      // 反射与透射分支相同\n吸收：w ← w · tint^{d / |cos θ_t|}      漫反射：w ← w · ρ，新方向 ~ cosθ / π',
            code: { title: 'HLSL · 随机游走参考（离线验证用，CPU / 光追 shader）', src: `
                // wi 指向光源（局部系 n = +Z）；返回逃逸出来的吞吐量。与页面 JS 的 L6.walk 一一对应；常量与辅助函数见附录
                float3 LayeredRandomWalk(float3 wi, inout RNG rng) {
                    float3 thr = 1;
                    float3 H = SampleGGXVNDF(wi, aCoat, Rand2(rng));
                    if (Rand(rng) < FresnelDielectric(dot(wi, H), eta)) {          // 涂层表面直接反射
                        float3 l = reflect(-wi, H);
                        return l.z > 0.0 ? SmithG2(wi, l) / SmithG1(wi) : 0.0;
                    }
                    float3 t = refract(-wi, H, 1.0 / eta);                          // 外 → 涂层：η_i/η_t = 1/η，t 朝下
                    if (t.z >= 0.0) return 0;
                    thr *= SmithG2(wi, -t) / SmithG1(wi);
                    for (int ev = 0; ev < 48; ++ev) {
                        thr *= pow(tint, d / abs(t.z)) * rho;                    // 下行吸收 + 漫反射
                        float3 u = SampleCosine(Rand2(rng));
                        thr *= pow(tint, d / u.z);                               // 上行吸收
                        float3 wf = float3(-u.x, -u.y, u.z);                     // 翻转坐标系：内侧为"上"
                        H = SampleGGXVNDF(wf, aCoat, Rand2(rng));
                        if (Rand(rng) < FresnelDielectric(dot(wf, H), 1.0 / eta)) {   // 内部反射（含全内反射）→ 回到底层
                            float3 l = reflect(-wf, H);
                            if (l.z <= 0.0) return 0;
                            thr *= SmithG2(wf, l) / SmithG1(wf);
                            t = float3(l.x, l.y, -l.z);                          // 翻回真实坐标系：向下走
                            continue;
                        }
                        float3 e = refract(-wf, H, eta);                         // 涂层 → 外：η_i/η_t = η
                        if (e.z >= 0.0) return 0;                                // 翻转系里朝下 = 真实系里朝上
                        return thr * SmithG2(wf, e) / SmithG1(wf);               // 透射出去
                    }
                    return 0;
                }` },
            previews: [
                {
                    id: 'mcPlot', needs: ['mc'], label: '白炉：各规则 vs 随机游走', sub: '点 = MC（当前 coat r / tint，白色底层）', size: [380, 190],
                    render: (c, S) => {
                        PL.draw.plot(c, { x: [0, 1], y: [0.4, 1.4], xLabel: 'μ = cosθo', series: ['add', 'fresnel2', 'albedo', 'ww', 'smooth'].map((k) => ({ data: S.furn[k], x: S.mus, color: RULE_COLOR[k], label: RULE_LABEL[k] })), hmarks: [{ y: 1, color: '#555' }] });
                        const { ctx, w, h } = PL.draw.ctx(c, { keep: true, smooth: true });
                        const X = (v) => 46 + v * (w - 56), Y = (v) => h - 32 - (v - 0.4) / 1.0 * (h - 42);
                        ctx.fillStyle = '#ffffff'; S.mcMu.forEach((m, i) => { ctx.beginPath(); ctx.arc(X(m), Y(S.mcE[i]), 2.6, 0, 7); ctx.fill(); });
                    },
                },
                {
                    id: 'mcErr', needs: ['mc'], label: '相对误差 vs MC', sub: '(E_rule − E_MC) / E_MC', size: [380, 170],
                    render: (c, S) => {
                        const at = (k, mu) => { const xs = S.mus, ys = S.furn[k]; let i = 0; while (i < xs.length - 2 && xs[i + 1] < mu) i++; const t = (mu - xs[i]) / (xs[i + 1] - xs[i]); return ys[i] * (1 - t) + ys[i + 1] * t; };
                        PL.draw.plot(c, { x: [0, 1], y: [-0.5, 0.5], xLabel: 'μ = cosθo', series: ['add', 'fresnel2', 'albedo', 'ww', 'smooth'].map((k) => ({ data: Array.from(S.mcMu, (mu, i) => at(k, mu) / S.mcE[i] - 1), x: S.mcMu, color: RULE_COLOR[k], label: RULE_LABEL[k] })), hmarks: [{ y: 0, color: '#555' }] });
                    },
                },
            ],
            notes: '默认参数下（光滑、透明涂层）"光滑解析"与 MC 几乎重合；把 coat r 拖到 0.5 以上、或给涂层加颜色，所有闭式规则都开始偏离——粗糙界面会把光散到别的方向，宏观 Fresnel 与几何级数都不再精确。这一步只在滚动到这里时计算（lazy），每个 μ 1600 条路径。',
            warn: '这个参考只模拟"单次散射的微表面界面 + Lambert 底 + 无散射介质"，界面本身仍有 GGX 单次散射的能量损失（模块 02），所以粗糙时 MC 曲线也会低于 1——比较的是各规则与同一物理模型的差距，而不是与 1 的差距。',
        },
        {
            id: 'uber', title: '材质总装：sheen / coat / specular / {metal | transmission | subsurface | diffuse}', tag: 'UBER',
            desc: 'OpenPBR、Autodesk Standard Surface 这类 uber 材质都按"从上到下"的层序组织：最外层 fuzz/sheen，下面 coat，再下面是电介质 specular 界面，最底层按权重<strong>混合</strong>金属、透射、次表面、漫反射。层与层之间用 albedo scaling 传递能量，同层之间用凸组合混合——前面几步的两种组合方式在这里各司其职。下图是一个"sheen + coat + 电介质 + 漫反射"材质的能量去向随视角的分布。',
            formula: 'f = f_sheen + (1 − E_sheen(μ_o)) · [ f_coat + (1 − E_coat(μ_o)) · ( f_spec + (1 − E_spec(μ_o)) · f_diffuse ) ]\n能量账：E_sheen + (1 − E_sheen)·E_coat + … + 被吸收 = 1',
            previews: [
                { id: 'uberStack', label: '层序（上 → 下）', size: [260, 200], render: (c) => drawStack(c) },
                { id: 'uberBudget', label: '能量去向 vs 视角', sub: '白色漫反射底；sheen / coat 权重 = 参数', size: [380, 200], render: (c, S) => drawBudget(c, S) },
                { id: 'uberSph', label: 'uber 材质', sub: 'sheen + coat + 电介质 + 漫反射', smooth: true, render: (c, S) => BX.renderSphere(c, uberBRDF(S), { res: 128, lights: lights6(S) }) },
            ],
            notes: '拖 sheen：顶层拿走的能量在掠射角最多（sheen 的 E(μ) 在掠射最大），下面所有层在边缘一起变暗。拖 coat r：coat 那条带在正面变宽。这里底层是白色（ρ = 1），所以四条带加起来恰好等于 1：albedo scaling 把每层没反射走的能量（包括它自身单次散射的损失）全部交给下一层；底层有颜色时，最下面一条才会按 ρ 缩小，差值就是被吸收的能量。',
        },
        {
            id: 'prod', title: '教学版 → 生产版：LUT 预算、引擎做法与常见坑', tag: 'PROD',
            desc: '<ul><li><strong>LUT 预算</strong>：每种上层 lobe 一张 E(μ, r)（GGX η=1.5 coat、sheen、电介质 specular），通常 32² R16F 放进一张 texture array；E<sub>avg</sub>(r) 一张 1D。运行时每层多一次采样。</li><li><strong>引擎做法</strong>：Filament 的 clearcoat 用 (1 − F<sub>c</sub>) 衰减底层；glTF 的 sheen / clearcoat 扩展用 albedo scaling；UE5 Substrate 把材质表示成可叠加的 slab，层间用方向反照率与覆盖率组合；OpenPBR 规范明确了层序与 albedo scaling 的近似。</li><li><strong>粗糙涂层会"模糊"底层</strong>：粗糙界面散射了进出底层的光，底层高光应变宽——生产里常用"底层粗糙度 ← 底层粗糙度与涂层粗糙度的组合"近似。</li><li><strong>法线</strong>：涂层与底层可以有各自的法线（橘皮 vs 底漆划痕），E(μ) 查表要用各自的 N·V。</li></ul>',
            formula: '粗糙度耦合（常见近似）：r_base\' = min(1, (r_base^4 + 2 r_coat^4)^{1/4})      // 经验式，不同引擎写法不同',
            previews: [
                { id: 'prodCmp', label: '全部规则的白炉一览', size: [420, 200], render: (c, S) => PL.draw.plot(c, { x: [0, 1], y: [0.5, 1.5], xLabel: 'μ = cosθo', series: Object.keys(RULE_LABEL).map((k) => ({ data: S.furn[k], x: S.mus, color: RULE_COLOR[k], label: RULE_LABEL[k] })), hmarks: [{ y: 1, color: '#555' }] }) },
            ],
            notes: '<strong>选型速查</strong>：要便宜且一定不发光 → 视线方向 albedo scaling；要互易（双向路径追踪、ReSTIR 交换 wi/wo）→ min 变体或对称写法；光滑透明涂层 + 漫反射底 → 带内部反射的解析式最准；需要验证 → 本页的随机游走。',
            warn: '<strong>常见坑</strong>：① 底层与涂层<strong>重复计算 Fresnel</strong>（底层自己的 specular 已含 F，再乘 (1 − F<sub>coat</sub>) 没问题，但别把涂层的 F 再算进底层的 F0）；② 把涂层导致的<strong>变暗、变饱和</strong>当成 bug 去"补亮"——它是物理的；③ 用入射角算吸收路径；④ albedo scaling 的 LUT 粗糙度轴用错映射（感知 r vs α）；⑤ 视线方向缩放不互易，用在双向算法里会出偏差；⑥ 涂层粗糙时仍用宏观 Fresnel 衰减，掠射处偏亮。',
        },
    ],
    graph: {
        intro: '分层材质在 GPU 上并没有"每层一个 pass"：所有 lobe 都在同一个光照 pass 里按层序内联求值，层间耦合靠预烘的方向反照率 LUT。随机游走参考只用于离线验证（CPU / 光追），不进运行时。',
        types: {
            cb: { color: '#e8c25a', label: 'cbuffer · 材质参数' },
            geom: { color: '#9aa5b1', label: 'G-buffer · N · V' },
            lut: { color: '#d19a66', label: 'LUT · 方向反照率 E(μ, r)' },
            lobe: { color: '#c678dd', label: 'lobe · 被内联的函数' },
            radiance: { color: '#4a8aff', label: 'radiance · RGBA16F' },
        },
        common: {
            title: 'Common.hlsl · 系列共享 + 分层工具',
            blocks: [
                { label: '━ 系列共享 ━', code: BX.HLSL_COMMON },
                { label: '━ 本模块 · 分层 ━', code: HLSL_LAYER },
            ],
        },
        nodes: [
            { id: 'matcb', title: 'LayeredMaterialCB', tag: 'CB', category: 'user', desc: 'baseColor · coat r/η/tint/厚度 · sheen', outputs: [{ name: 'MaterialCB', format: 'cbuffer', type: 'cb' }], detail: { preview: null } },
            { id: 'gbuf', title: 'GBuffer', tag: 'RT', category: 'system', desc: 'N · V（涂层 / 底层可各有法线）', outputs: [{ name: 'GBuffer', format: 'RGBA16F ×3', type: 'geom' }], detail: { preview: null } },
            {
                id: 'bake', title: 'CS_BakeLayerLUTs', tag: 'CS', category: 'bake', lifetime: 'param', step: 'albedo', desc: 'E_coat / E_sheen / E_spec (μ, r)',
                dispatch: { name: 'CS_BakeLayerLUTs', threads: [8, 8, 1], groups: [4, 4, 3], idRange: 'ID.xy = (μ, r) ∈ 32²，z = lobe', note: '依赖 η：换 η 预设才重烘' },
                outputs: [{ name: 'LayerLUTs', format: 'R16F 32×32×3', type: 'lut' }], detail: { preview: 'lut' },
            },
            { id: 'coatFn', title: 'Coat_GGX', tag: 'FN', step: 'coat', dispatch: { kind: 'INLINE', note: '涂层镜面 lobe' }, inputs: [{ name: 'MaterialCB', type: 'cb' }, { name: 'GBuffer', type: 'geom' }], outputs: [{ name: 'CoatLobe', format: 'f(ωi, ωo)', type: 'lobe' }], detail: { formula: 0, code: 0, preview: 'coatLobe' } },
            { id: 'baseFn', title: 'Base_Lobes', tag: 'FN', step: 'mix', dispatch: { kind: 'INLINE', note: 'specular + {metal | diffuse …} 的凸组合' }, inputs: [{ name: 'MaterialCB', type: 'cb' }, { name: 'GBuffer', type: 'geom' }], outputs: [{ name: 'BaseLobe', format: 'f(ωi, ωo)', type: 'lobe' }] },
            { id: 'layerFn', title: 'LayerCombine', tag: 'FN', step: 'albedo', dispatch: { kind: 'INLINE', note: 'f_top + f_base·(1 − E_top(μo))' }, inputs: [{ name: 'CoatLobe', type: 'lobe' }, { name: 'BaseLobe', type: 'lobe' }, { name: 'LayerLUTs', type: 'lut' }], outputs: [{ name: 'LayeredBSDF', format: 'f(ωi, ωo)', type: 'lobe' }], detail: { formula: 0, code: 0, preview: 'alFurn' } },
            {
                id: 'light', title: 'PS_DeferredLighting', tag: 'PS', category: 'output', lifetime: 'frame', step: 'uber', desc: 'Σ lights · 层序内联求值',
                dispatch: { kind: 'PS', name: 'PS_DeferredLighting', target: 'SceneColor（full-screen）', idRange: 'SV_Position.xy ∈ viewport' },
                inputs: [{ name: 'LayeredBSDF', type: 'lobe' }, { name: 'GBuffer', type: 'geom' }, { name: 'MaterialCB', type: 'cb' }], outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'radiance' }],
                detail: { preview: 'uberSph' },
            },
            {
                id: 'ref', title: 'CPU_RandomWalkReference', tag: 'CPU', lifetime: 'param', step: 'mc', desc: '离线验证：层间随机游走',
                dispatch: { kind: 'CPU', name: 'LayeredRandomWalk', count: '12 μ × 1600 路径', note: '不进运行时；用来给闭式规则打分' },
                inputs: [{ name: 'MaterialCB', type: 'cb' }], outputs: [{ name: 'ReferenceAlbedo', format: 'float[12]', type: 'lut' }],
                detail: { preview: 'mcPlot' },
            },
        ],
    },
    references: [
        { title: 'Physically-Based Shading at Disney', author: 'B. Burley', venue: 'SIGGRAPH Course', year: 2012, note: 'clearcoat lobe（GTR1，F0 = 0.04）' },
        { title: 'Arbitrarily Layered Micro-Facet Surfaces', author: 'A. Weidlich, A. Wilkie', venue: 'GRAPHITE', year: 2007 },
        { title: 'Why some things are darker when wet', author: 'J. Lekner, M. C. Dorf', venue: 'Applied Optics 27(7)', year: 1988, note: '涂层 / 水膜下的内部反射与变暗' },
        { title: 'Revisiting Physically Based Shading at Imageworks', author: 'C. Kulla, A. Conty', venue: 'SIGGRAPH Course', year: 2017, note: '方向反照率与能量守恒的分层' },
        { title: 'Position-Free Monte Carlo Method for Layered BSDFs', author: 'Y. Guo, M. Hašan, S. Zhao', venue: 'ACM TOG (SIGGRAPH Asia)', year: 2018 },
        { title: 'Efficient Rendering of Layered Materials using an Atomic Decomposition with Statistical Operators', author: 'L. Belcour', venue: 'ACM TOG (SIGGRAPH)', year: 2018 },
        { title: 'Physically Based Rendering in Filament', author: 'R. Guy, M. Agopian', venue: 'Google', note: 'clearcoat 模型' },
        { title: 'OpenPBR Surface Specification', author: 'Academy Software Foundation', year: 2024 },
    ],
});

// ── helpers used by previews (hoisted function declarations) ──
function avgT(t, thick) { let s = 0; const n = 64; for (let i = 0; i < n; i++) { const m = (i + 0.5) / n; s += Math.pow(Math.max(1e-4, t), thick / m) * 2 * m; } return s / n; }
function mixBRDF(S, m, colorOverride) {
    const base = colorOverride || S.baseColor, a = BX.alpha(0.35);
    return (wi, wo) => {
        if (wi[2] <= 0 || wo[2] <= 0) return [0, 0, 0];
        const h = BX.half(wi, wo), s = BX.ggxD(h, a) * BX.smithG2(wo, wi, a) / (4 * wi[2] * wo[2]), voh = BX.dot(wo, h);
        const Fd = BX.schlick(0.04, voh), Fm = BX.schlick(base, voh);
        return [0, 1, 2].map((c) => (1 - m) * (base[c] * BX.INV_PI * (1 - Fd) + s * Fd) + m * s * Fm[c]);
    };
}
function uberBRDF(S) {
    const shA = 0.25, P = mat(S), spec = L6.ggxSpec(0.45, 1.5);
    return (wi, wo) => {
        if (wi[2] <= 0 || wo[2] <= 0) return [0, 0, 0];
        const h = BX.half(wi, wo), sinH = Math.sqrt(Math.max(0, 1 - h[2] * h[2]));
        const fSheen = S.sheenW * (2 + 1 / shA) * Math.pow(sinH, 1 / shA) / (2 * Math.PI) / (4 * (wi[2] + wo[2] - wi[2] * wo[2]));
        const kSheen = 1 - S.sheenW * L6.lookup(S.Esheen, wo[2]), kCoat = Math.max(0, 1 - L6.lookup(S.Ec, wo[2])), kSpec = Math.max(0, 1 - L6.lookup(S.Espec, wo[2]));
        const fc = P.coat(wi, wo), fs = spec(wi, wo);
        return [0, 1, 2].map((c) => fSheen * 0.9 + kSheen * (fc + kCoat * (fs + kSpec * S.baseColor[c] * BX.INV_PI)));
    };
}
function drawStack(c) {
    const { ctx, w, h } = PL.draw.ctx(c);
    const layers = [['fuzz / sheen', '#c678dd'], ['coat', '#e8c25a'], ['specular（电介质界面）', '#6ab0e0'], ['metal | transm. | SSS | diffuse', '#5ad07a']];
    const top = 14, lh = (h - 30) / layers.length;
    ctx.font = '11px Consolas, monospace'; ctx.textBaseline = 'middle';
    layers.forEach(([name, col], i) => {
        const y = top + i * lh;
        ctx.fillStyle = col + '33'; ctx.fillRect(12, y, w - 24, lh - 6);
        ctx.strokeStyle = col; ctx.strokeRect(12.5, y + 0.5, w - 25, lh - 7);
        ctx.fillStyle = col; ctx.fillText(name, 20, y + (lh - 6) / 2);
    });
    ctx.fillStyle = '#777'; ctx.font = '10px Consolas, monospace';
    ctx.fillText('层间：albedo scaling   同层：凸组合', 14, h - 8);
}
function drawBudget(c, S) {
    const { ctx, w, h } = PL.draw.ctx(c);
    const padL = 40, padR = 10, padT = 10, padB = 26, n = 40;
    const X = (mu) => padL + mu * (w - padL - padR), Y = (v) => h - padB - v * (h - padT - padB);
    const bands = [];
    for (let i = 0; i < n; i++) {
        const mu = 0.03 + 0.97 * i / (n - 1);
        const eSh = S.sheenW * L6.lookup(S.Esheen, mu), eC = L6.lookup(S.Ec, mu), eSp = L6.lookup(S.Espec, mu);
        const sh = eSh, co = (1 - eSh) * eC, sp = (1 - eSh) * (1 - eC) * eSp, di = (1 - eSh) * (1 - eC) * (1 - eSp);
        bands.push([mu, sh, co, sp, di]);
    }
    const cols = ['#c678dd', '#e8c25a', '#6ab0e0', '#5ad07a'], names = ['sheen', 'coat', 'specular', 'diffuse'];
    for (let k = 3; k >= 0; k--) {
        ctx.beginPath();
        bands.forEach(([mu, ...v], i) => { const cum = v.slice(0, k + 1).reduce((a, b) => a + b, 0); if (i) ctx.lineTo(X(mu), Y(cum)); else ctx.moveTo(X(mu), Y(cum)); });
        ctx.lineTo(X(bands[n - 1][0]), Y(0)); ctx.lineTo(X(bands[0][0]), Y(0)); ctx.closePath();
        ctx.fillStyle = cols[k] + 'aa'; ctx.fill();
    }
    ctx.strokeStyle = '#333'; ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
    ctx.fillStyle = '#5a5a5a'; ctx.font = '10px Consolas, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [0, 0.5, 1].forEach((v) => ctx.fillText(String(v), padL - 4, Y(v)));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    [0, 0.5, 1].forEach((v) => ctx.fillText(String(v), X(v), h - padB + 3));
    ctx.fillText('μ = cosθo', padL + (w - padL - padR) / 2, h - 12);
    ctx.textAlign = 'left';
    names.forEach((nm, k) => { ctx.fillStyle = cols[k]; ctx.fillText('■ ' + nm, w - padR - 78, padT + 4 + k * 13); });
}
// ── 3D 层间光路 ──
/** Step 08 的随机游走（规则与 L6.walk 相同），额外记录位置：涂层界面 z = 0，底层 z = −H。
 *  pts = 折线顶点（局部系），seg[i] = 走 pts[i] → pts[i+1] 这段时的吞吐量 RGB；statsOnly 时只返回结果类型与底层反弹次数 */
function layerPath(winc, P, rnd, H, statsOnly) {
    const a = BX.alpha(P.coatRough), eta = P.ior, flip = (w) => [w[0], w[1], -w[2]];
    let x = [0, 0, 0], thr = [1, 1, 1];
    const pts = statsOnly ? null : [winc.map((v) => v * 1.3), [0, 0, 0]], seg = statsOnly ? null : [[1, 1, 1]];
    const push = (p) => { if (pts) { pts.push(p); seg.push(thr.slice()); } };
    const done = (kind, bounces) => ({ pts, seg, kind, bounces });
    let h = BX.sampleVNDF(winc, a, a, rnd(), rnd());
    if (rnd() < BX.fresnelDielectric(BX.dot(winc, h), eta)) {   // 涂层表面直接反射
        const l = BX.reflect(winc, h);
        if (l[2] <= 0) return done('lost', 0);
        push(l.map((v) => v * 1.1));
        return done('coat', 0);
    }
    let t = BX.refract(winc, h, eta);
    if (!t || t[2] >= 0) return done('lost', 0);
    let g = BX.smithG2(winc, t, a) / BX.smithG1(winc, a);
    thr = thr.map((v) => v * g);
    for (let ev = 0; ev < 48; ev++) {
        const down = P.tint.map((c) => Math.pow(Math.max(1e-4, c), P.thick / Math.max(1e-3, -t[2])));
        thr = thr.map((v, ch) => v * down[ch]);
        const s = H / -t[2]; x = [x[0] + t[0] * s, x[1] + t[1] * s, -H]; push(x);   // 下行到底层
        const u = BX.sampleCosine(rnd(), rnd());
        const up = P.tint.map((c) => Math.pow(Math.max(1e-4, c), P.thick / Math.max(1e-3, u[2])));
        thr = thr.map((v, ch) => v * P.rho[ch] * up[ch]);
        const s2 = H / Math.max(1e-3, u[2]); x = [x[0] + u[0] * s2, x[1] + u[1] * s2, 0]; push(x);   // 漫反射后上行到界面
        if (Math.max(thr[0], thr[1], thr[2]) < 1e-4) return done('lost', ev + 1);
        const wf = flip([-u[0], -u[1], -u[2]]);
        h = BX.sampleVNDF(wf, a, a, rnd(), rnd());
        if (rnd() < BX.fresnelDielectric(BX.dot(wf, h), 1 / eta)) {   // 内部反射（含全内反射）
            const l = BX.reflect(wf, h);
            if (l[2] <= 0) return done('lost', ev + 1);
            g = BX.smithG2(wf, l, a) / BX.smithG1(wf, a); thr = thr.map((v) => v * g);
            t = flip(l);
            continue;
        }
        const e = BX.refract(wf, h, 1 / eta);
        if (!e || e[2] >= 0) return done('lost', ev + 1);
        g = BX.smithG2(wf, e, a) / BX.smithG1(wf, a); thr = thr.map((v) => v * g);
        const eo = flip(e); push([x[0] + eo[0] * 1.1, x[1] + eo[1] * 1.1, eo[2] * 1.1]);   // 逃逸
        return done('out', ev + 1);
    }
    return done('lost', 48);
}
function drawLayer3d(c, S) {
    const W = S.walk3d, { H, B } = L6_3D, P = (p) => [p[0], p[2], -p[1]];   // 局部 n = +Z → mesh3d 的 y-up（绕 x 轴旋转）
    const enc = (v) => Math.round(255 * Math.pow(BX.clamp(v, 0, 1), 1 / 2.2));
    const col = (thr) => { const m = Math.max(thr[0], thr[1], thr[2], 1e-9), b = BX.clamp(Math.pow(m, 0.3), 0.3, 1); return `rgb(${enc(thr[0] / m * b)},${enc(thr[1] / m * b)},${enc(thr[2] / m * b)})`; };
    const inside = (p) => Math.max(Math.abs(p[0]), Math.abs(p[1])) <= B;
    const lines = [];
    for (const r of W.paths) {
        for (let i = 1; i + 1 < r.pts.length; i++) {   // i = 0 是共同的入射段，下面单独画
            const a = r.pts[i];
            let b = r.pts[i + 1], cut = false;
            if (!inside(a)) break;
            if (!inside(b)) {   // 截到盒子边界
                let tt = 1;
                for (const k of [0, 1]) { const dd = b[k] - a[k]; if (Math.abs(dd) > 1e-9) { const tk = ((dd > 0 ? B : -B) - a[k]) / dd; if (tk >= 0 && tk < tt) tt = tk; } }
                b = [0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * tt); cut = true;
            }
            const last = !cut && i + 2 === r.pts.length;
            lines.push({ pts: [P(a), P(b)], color: r.kind === 'coat' ? '#9ccfee' : col(r.seg[i]), width: 1.6, arrow: last });
            if (cut) break;
        }
    }
    const wi = thetaW(S);
    lines.push({ pts: [P(wi.map((v) => v * 1.3)), [0, 0, 0]], color: '#e8c25a', width: 2.4, arrow: true, labelAt: 'start', label: 'ωi' });
    const sq = (y) => [[-B, y, -B], [B, y, -B], [B, y, B], [-B, y, B], [-B, y, -B]];
    lines.push({ pts: sq(0), color: 'rgba(106,176,224,0.7)', width: 1.2 });
    const base = S.baseColor.map((v) => 0.6 * Math.pow(BX.clamp(v, 0, 1), 1 / 2.2));   // 压暗，让光路更显眼
    PL.draw.mesh3d(c, {
        meshes: [
            PL.mesh.plane({ c: [0, 0, 0], size: 2 * B, surface: false, wire: true, wireColor: 'rgba(106,176,224,0.25)' }),
            PL.mesh.plane({ c: [0, -H, 0], size: 2 * B, rgb: base }),
        ],
        lines,
        labels: [
            { p: [B, 0.06, B], text: `涂层界面 η = ${S.coatIor.toFixed(2)}`, color: '#8cc4ea' },
            { p: [B, -H - 0.02, B], text: 'Lambert 底层 ρ', color: '#c8a8a0' },
        ],
        orbit: { yaw: 0.35, pitch: 0.32, dist: 7.0 },
        overlay: (ctx) => {
            ctx.font = '11px Consolas, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.fillStyle = '#99a';
            ctx.fillText(`白底、透明涂层（4000 条）：涂层直接反射 ${(W.coat * 100).toFixed(0)} %`, 8, 30);
            ctx.fillText(`进入后平均碰底层 ${W.meanB.toFixed(2)} 次才逃出（1/(1 − F̄in) = ${(1 / (1 - S.FinAvg)).toFixed(2)}）`, 8, 45);
        },
    });
}
