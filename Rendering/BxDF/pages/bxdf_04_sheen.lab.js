// ════════════════════════════════════════════════════════════════════════════
// BxDF 04 · 织物 Sheen：Charlie / Ashikhmin
// build: node <skill>/scripts/build_doc.mjs bxdf_common.js bxdf_04_sheen.lab.js -o bxdf_04_sheen.html
// 约定全部沿用 bxdf_common.js（n = +Z，wi → 光源，wo → 观察者，线性 RGB）。本模块额外约定：
//   • sheen 粗糙度用感知值 r ∈ (0,1]，α = r²（glTF KHR_materials_sheen / Filament）。
//     Charlie 论文里的 "r" 就是这里的 α；Ashikhmin 的宽度 σ 取 α（Filament D_Ashikhmin 约定）。
//   • 所有 sheen 波瓣按 sheenColor = 1 的标量计算，着色时再乘 sheenColor；不带 Fresnel（F ≡ 1）。
// ════════════════════════════════════════════════════════════════════════════
const SH = (() => {
    const PI = Math.PI;
    const P0 = [25.3245, 3.32435, 0.16801, -1.27393, -4.85967];   // E&K 2017 Table 1, r = 0
    const P1 = [21.5473, 3.82987, 0.19823, -1.97760, -4.32054];   // E&K 2017 Table 1, r = 1
    const aOf = (r) => Math.max(1e-3, r * r);
    const max3 = (c) => Math.max(c[0], c[1], c[2]);

    // ── distributions D(cosθh) ──
    const dCharlie = (c, a) => { const s2 = Math.max(0, 1 - c * c); return (2 + 1 / a) * Math.pow(s2, 0.5 / a) / (2 * PI); };
    const dAsh = (c, a) => { const c2 = c * c, s2 = Math.max(1e-9, 1 - c2), a2 = a * a; return (1 + 4 * Math.exp(-c2 / (a2 * s2)) / (s2 * s2)) / (PI * (1 + 4 * a2)); };
    const dAshOffset = (c, a) => 1 / (PI * (1 + 4 * a * a));
    const dGGX = (c, a) => BX.ggxD([Math.sqrt(Math.max(0, 1 - c * c)), 0, c], a);

    // ── Charlie Λ fit (E&K 2017 Eq. 3) + terminator softening (Eq. 4) ──
    const lFit = (x, a) => {
        const t = (1 - a) * (1 - a), p = (k) => t * P0[k] + (1 - t) * P1[k];
        return p(0) / (1 + p(1) * Math.pow(x, p(2))) + p(3) * x + p(4);
    };
    const lambda = (c, a) => { c = Math.min(1, Math.abs(c)); return c < 0.5 ? Math.exp(lFit(c, a)) : Math.exp(2 * lFit(0.5, a) - lFit(1 - c, a)); };
    const lambdaSoft = (c, a) => Math.pow(lambda(c, a), 1 + 2 * Math.pow(1 - Math.min(1, Math.abs(c)), 8));
    /** Smith Λ from the projected-area identity (Heitz 2014): Λ(μ) = (1/μ)∫⟨ω·m⟩D(m)dm − 1; φ part in closed form */
    const lambdaNum = (D, mu, a, n = 256) => {
        const so = Math.sqrt(Math.max(0, 1 - mu * mu));
        let s = 0;
        for (let i = 0; i < n; i++) {
            const c = (i + 0.5) / n, A = so * Math.sqrt(1 - c * c), B = mu * c;
            const I = B >= A ? 2 * PI * B : 2 * (B * Math.acos(-B / A) + A * Math.sqrt(1 - (B / A) * (B / A)));
            s += D(c, a) * I;
        }
        return s / n / mu - 1;
    };
    /** ∫ D(m) cosθm dωm by 1D midpoint quadrature in θ (should be 1) */
    const normInt = (D, a, n = 4000) => { const dt = PI / 2 / n; let s = 0; for (let i = 0; i < n; i++) { const t = (i + 0.5) * dt, c = Math.cos(t); s += D(c, a) * c * Math.sin(t); } return 2 * PI * s * dt; };

    // ── visibility V = G / (4·NoL·NoV) ──
    const vCharlie = (nl, nv, a, soft) => 1 / ((1 + lambda(nv, a) + (soft ? lambdaSoft(nl, a) : lambda(nl, a))) * 4 * nl * nv);
    const vNeubelt = (nl, nv) => 1 / (4 * (nl + nv - nl * nv));
    const vNeubeltSat = (nl, nv) => Math.min(1, vNeubelt(nl, nv));
    const vNoG = (nl, nv) => 1 / (4 * nl * nv);

    // ── BRDF lobes (scalar, sheenColor = 1, local frame) ──
    const up = (wi, wo) => wi[2] > 1e-6 && wo[2] > 1e-6;
    const fCharlie = (wi, wo, a, soft) => (up(wi, wo) ? dCharlie(BX.half(wi, wo)[2], a) * vCharlie(wi[2], wo[2], a, soft) : 0);
    const fFilament = (wi, wo, a) => (up(wi, wo) ? dCharlie(BX.half(wi, wo)[2], a) * vNeubeltSat(wi[2], wo[2]) : 0);
    const fOrder = (wi, wo, a) => (up(wi, wo) ? dAsh(BX.half(wi, wo)[2], a) * vNeubelt(wi[2], wo[2]) : 0);
    const fDisney = (wi, wo) => (up(wi, wo) ? BX.pow5(1 - BX.dot(wi, BX.half(wi, wo))) : 0);
    const fGGX = (wi, wo, a, F0) => { if (!up(wi, wo)) return 0; const h = BX.half(wi, wo); return BX.schlick(F0, BX.dot(wi, h)) * BX.ggxD(h, a) * BX.smithG2(wo, wi, a) / (4 * wi[2] * wo[2]); };

    // ── directional albedo E(μo) = ∫ f cosθi dωi，各向同性 → φ ∈ [0,π] 镜像对称 ×2；midpoint in (μi, φ) ──
    const COS = {};
    const cosTab = (m) => COS[m] || (COS[m] = Float64Array.from({ length: m }, (_, j) => Math.cos((j + 0.5) / m * PI)));
    /** f = D(h·n) · V(μi, μo)：V 每个 μi 环只算一次 */
    const albedoDV = (D, V, mu, a, n = 48) => {
        const so = Math.sqrt(Math.max(0, 1 - mu * mu)), ct = cosTab(n);
        let s = 0;
        for (let i = 0; i < n; i++) {
            const mi = (i + 0.5) / n, si = Math.sqrt(1 - mi * mi);
            let ring = 0;
            for (let j = 0; j < n; j++) { const d = si * so * ct[j] + mi * mu; ring += D((mi + mu) / Math.sqrt(2 + 2 * d), a); }
            s += ring * V(mi, mu, a) * mi;
        }
        return s * (2 * PI) / (n * n);
    };
    /** 通用：g(μi, μo, cosθh, wi·h) */
    const albedoG = (g, mu, n = 48) => {
        const so = Math.sqrt(Math.max(0, 1 - mu * mu)), ct = cosTab(n);
        let s = 0;
        for (let i = 0; i < n; i++) {
            const mi = (i + 0.5) / n, si = Math.sqrt(1 - mi * mi);
            let ring = 0;
            for (let j = 0; j < n; j++) { const d = si * so * ct[j] + mi * mu, l = Math.sqrt(2 + 2 * d); ring += g(mi, mu, (mi + mu) / l, l / 2); }
            s += ring * mi;
        }
        return s * (2 * PI) / (n * n);
    };

    // ── tables ──
    const curveAt = (arr, mu) => { const n = arr.length, x = PL.clamp(mu * n - 0.5, 0, n - 1), i = Math.min(n - 2, Math.floor(x)), f = x - i; return arr[i] * (1 - f) + arr[i + 1] * f; };
    const lutAt = (lut, N, mu, r) => {
        const x = PL.clamp(mu * N - 0.5, 0, N - 1), y = PL.clamp(r * N - 0.5, 0, N - 1), x0 = Math.floor(x), y0 = Math.floor(y);
        const x1 = Math.min(N - 1, x0 + 1), y1 = Math.min(N - 1, y0 + 1), fx = x - x0, fy = y - y0;
        return (lut[y0 * N + x0] * (1 - fx) + lut[y0 * N + x1] * fx) * (1 - fy) + (lut[y1 * N + x0] * (1 - fx) + lut[y1 * N + x1] * fx) * fy;
    };

    return {
        aOf, max3, dCharlie, dAsh, dAshOffset, dGGX, lFit, lambda, lambdaSoft, lambdaNum, normInt,
        vCharlie, vNeubelt, vNeubeltSat, vNoG, fCharlie, fFilament, fOrder, fDisney, fGGX, albedoDV, albedoG, curveAt, lutAt,
    };
})();

const SHEEN_LUT_N = 32;
const SHEEN_HEMI = 80;
const SHEEN_L3D = { nu: 64, nv: 24 };   // 3D 波瓣网格：φ 64 段 × θ 24 段（0 … 89.5°：正好 90° 时 f 按定义为 0，会把波瓣边缘拉回原点）
const sheenL3dDir = (i, j) => BX.dir(0.4972 * Math.PI * j / SHEEN_L3D.nv, 2 * Math.PI * i / SHEEN_L3D.nu);
const SHEEN_COL = { ggx: '#6ab0e0', disney: '#c678dd', order: '#e0a060', charlie: '#5ad07a', soft: '#56b6c2', filament: '#e06c75', num: '#e8e8e8', lambert: '#7d8894' };

// light for sphere renders: angle between light and view (in the x–z view plane), 22° elevation
// key light of the sphere renders: pitch = θi, yaw = 光源方位 (series convention BX.keyLight), derived in the "keyLight" stage
const sheenLights = (S) => [{ dir: S.light, color: [3.2, 3.2, 3.2] }, { dir: BX.norm([-0.55, 0.25, 0.8]), color: [0.2, 0.2, 0.23] }];
const sheenWi = (S) => [Math.sin(S.thetaI), 0, Math.cos(S.thetaI)];

// scalar sheen lobes by name (sheenColor = 1)
const SHEEN_LOBE = {
    none: () => 0,
    ggx: (wi, wo, a) => SH.fGGX(wi, wo, a, 0.04),
    disney: (wi, wo) => SH.fDisney(wi, wo),
    order: (wi, wo, a) => SH.fOrder(wi, wo, a),
    charlie: (wi, wo, a) => SH.fCharlie(wi, wo, a, false),
    soft: (wi, wo, a) => SH.fCharlie(wi, wo, a, true),
    filament: (wi, wo, a) => SH.fFilament(wi, wo, a),
};
/** 完整布料 BRDF（RGB）：Lambert(baseColor) × scaling + tint × sheen lobe。Efn(μ) = sheen 方向反照率（sheenColor = 1） */
const sheenCloth = (S, kind, mode = 'none', Efn = null) => {
    const a = SH.aOf(S.rough), C = S.sheenColor, B = S.baseColor, s = SH.max3(C), lob = SHEEN_LOBE[kind];
    const tint = kind === 'ggx' ? [1, 1, 1] : C;
    return (wi, wo) => {
        const fs = lob(wi, wo, a);
        let k = 1;
        if (mode !== 'none' && Efn) { const eo = 1 - s * Efn(wo[2]); k = Math.max(0, mode === 'view' ? eo : Math.min(eo, 1 - s * Efn(wi[2]))); }   // E > 1 时 k < 0 → 钳到 0
        return [tint[0] * fs + B[0] * k / Math.PI, tint[1] * fs + B[1] * k / Math.PI, tint[2] * fs + B[2] * k / Math.PI];
    };
};
const sheenSphere = (c, S, f) => BX.renderSphere(c, f, { lights: sheenLights(S), res: 112 });
/** same geometry and lights as BX.renderSphere: luminance of one pixel at screen (u, v) ∈ disk, NaN off-sphere */
const sheenPixelLum = (f, lights, u, v) => {
    const r2 = u * u + v * v;
    if (r2 > 1) return NaN;
    const n = [u, v, Math.sqrt(1 - r2)];
    let t = BX.cross([0, 1, 0], n);
    if (Math.hypot(t[0], t[1], t[2]) < 1e-4) t = [1, 0, 0];
    t = BX.norm(t);
    const b = BX.cross(n, t), toL = (w) => [BX.dot(w, t), BX.dot(w, b), BX.dot(w, n)], wo = toL([0, 0, 1]);
    let L = 0;
    for (const l of lights) { const wi = toL(l.dir); if (wi[2] <= 0) continue; L += BX.lum(f(wi, wo)) * BX.lum(l.color) * wi[2]; }
    return L;
};
/** per-pixel luminance image (for difference images) */
const sheenSphereLum = (f, S, res, out) => {
    out = out && out.length === res * res ? out : new Float32Array(res * res);
    const lights = sheenLights(S);
    for (let y = 0; y < res; y++) for (let x = 0; x < res; x++) out[y * res + x] = sheenPixelLum(f, lights, 2 * (x + 0.5) / res - 1, -(2 * (y + 0.5) / res - 1));
    return out;
};
/** hemisphere heat map + optional marker dots (local directions) */
const sheenHemi = (c, data, o = {}) => {
    PL.draw.field(c, data, SHEEN_HEMI, SHEEN_HEMI, Object.assign({ map: 'inferno', stretch: 'sqrt' }, o.field || {}));
    const g = c.getContext('2d');
    for (const m of o.marks || []) {
        const [px, py] = BX.hemiPixel(m.w, SHEEN_HEMI);
        g.fillStyle = m.color || '#e8c25a';
        const s = m.size || 3;
        g.fillRect(Math.round(px) - (s >> 1), Math.round(py) - (s >> 1), s, s);
    }
};
const sheenHemiProbe = (key, unit = '') => (x, y, S) => {
    const v = S[key][y * SHEEN_HEMI + x], w = BX.hemiDir(x, y, SHEEN_HEMI);
    return w && isFinite(v) ? `θ = ${(Math.acos(w[2]) * 180 / Math.PI).toFixed(1)}°  φ = ${(Math.atan2(w[1], w[0]) * 180 / Math.PI).toFixed(0)}°  →  ${v.toPrecision(4)}${unit}` : '';
};
const sheenMuGrid = (n) => Float32Array.from({ length: n }, (_, i) => (i + 0.5) / n);
/** 白炉球：显示 E_total(μo) − 1（只看 sheenColor 最大的通道） */
const sheenFurnace = (c, S, mode) => {
    const res = 96, data = PL.buf(S, '_fur_' + mode + '_' + c.id, res * res), tot = S.furnace[mode];
    for (let y = 0; y < res; y++) for (let x = 0; x < res; x++) {
        const u = 2 * (x + 0.5) / res - 1, v = -(2 * (y + 0.5) / res - 1), r2 = u * u + v * v;
        data[y * res + x] = r2 > 1 ? NaN : SH.curveAt(tot, Math.sqrt(1 - r2)) - 1;
    }
    S['_furData_' + mode] = data;
    PL.draw.field(c, data, res, res, { map: 'diverge', range: [-0.6, 0.6] });
};

// ═══════════════════════════════ HLSL ═══════════════════════════════
const HLSL_SHEEN_D = `
    // Estevez & Kulla 2017 · "Charlie"：D = (2 + 1/a)·sin^(1/a)θh / 2π，a = sheenRoughness²
    float D_Charlie(float NoH, float a) {
        float invA  = 1.0 / a;
        float sin2h = max(1.0 - NoH * NoH, 0.0078125);    // 2^-7：保证 fp16 下 sin2h² 不下溢（Filament）
        return (2.0 + invA) * pow(sin2h, 0.5 * invA) / (2.0 * PI);
    }
    // Ashikhmin & Premoze 2007 倒置高斯 + 偏置 1，A = 4；σ = a（Filament D_Ashikhmin 约定）
    float D_Ashikhmin(float NoH, float a) {
        float a2 = a * a, cos2h = NoH * NoH;
        float sin2h = max(1.0 - cos2h, 0.0078125);
        float sin4h = sin2h * sin2h;
        float cot2  = -cos2h / (a2 * sin2h);
        return 1.0 / (PI * (1.0 + 4.0 * a2) * sin4h) * (4.0 * exp(cot2) + sin4h);
    }`;
const HLSL_SHEEN_V = `
    // Charlie Λ 拟合（E&K 2017 Eq.3 + Table 1）；插值权重 t = (1 − a)²：t = 1 ↔ r = 0，t = 0 ↔ r = 1
    float SheenL(float x, float a) {
        float t = (1.0 - a) * (1.0 - a);
        float A = lerp(21.5473, 25.3245, t);
        float B = lerp(3.82987, 3.32435, t);
        float C = lerp(0.19823, 0.16801, t);
        float D = lerp(-1.97760, -1.27393, t);
        float E = lerp(-4.32054, -4.85967, t);
        return A / (1.0 + B * pow(x, C)) + D * x + E;
    }
    float Lambda_Charlie(float NoX, float a) {
        float c = abs(NoX);
        return c < 0.5 ? exp(SheenL(c, a)) : exp(2.0 * SheenL(0.5, a) - SheenL(1.0 - c, a));
    }
    // 高度相关 G2 = 1 / (1 + Λv + Λl)；V = G2 / (4·NoV·NoL)
    float V_Charlie(float NoV, float NoL, float a, bool softTerminator) {
        float lv = Lambda_Charlie(NoV, a);
        float ll = Lambda_Charlie(NoL, a);
        if (softTerminator) {                             // E&K Eq.4：只改光源侧 → 不再互易
            float p = 1.0 - NoL; p *= p; p *= p; p *= p;  // (1 − NoL)^8
            ll = pow(ll, 1.0 + 2.0 * p);
        }
        return 1.0 / ((1.0 + lv + ll) * (4.0 * NoV * NoL));
    }
    // Neubelt & Pettineo 2013（The Order: 1886）：Ashikhmin 的平滑分母，没有 G
    float V_Neubelt(float NoV, float NoL) {
        return saturate(1.0 / (4.0 * (NoL + NoV - NoL * NoV)));   // saturate 来自 Filament
    }`;
const HLSL_SHEEN_E = `
    Texture2D<float> SheenE_LUT;      // 32×32 R16F：x = NoV，y = sheenRoughness（感知 r，不是 a）
    SamplerState     LinearClamp;
    float Max3(float3 v) { return max(v.x, max(v.y, v.z)); }
    float SheenE(float NoX, float r) { return SheenE_LUT.SampleLevel(LinearClamp, float2(NoX, r), 0); }
    // glTF KHR_materials_sheen / E&K §5：base 乘 min(1 − s·E(μo), 1 − s·E(μi))；Filament 只用 NoV 那一项
    float SheenAlbedoScaling(float NoV, float NoL, float3 sheenColor, float r) {
        float s = Max3(sheenColor);
        return min(1.0 - s * SheenE(NoV, r), 1.0 - s * SheenE(NoL, r));
    }`;
const HLSL_BAKE = `
    RWTexture2D<float> SheenE_LUT;                        // 32×32 R16F
    [numthreads(8, 8, 1)]
    void CS_BakeSheenE(uint3 id : SV_DispatchThreadID) {
        float  NoV = (id.x + 0.5) / 32.0;                 // texel 中心
        float  r   = (id.y + 0.5) / 32.0, a = r * r;
        float3 V   = float3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
        const uint N = 1024;
        float  E   = 0.0;
        for (uint i = 0; i < N; i++) {
            float2 u   = Hammersley(i, N);
            float  z   = u.x, s = sqrt(1.0 - z * z), p = 2.0 * PI * u.y;
            float3 L   = float3(s * cos(p), s * sin(p), z);   // 均匀半球，pdf = 1/(2π)（E&K 推荐）
            float3 H   = normalize(L + V);
            E += D_Charlie(H.z, a) * V_Charlie(NoV, L.z, a, false) * L.z * (2.0 * PI);
        }
        SheenE_LUT[id.xy] = E / N;
    }`;
const HLSL_LIGHT = `
    float4 PS_DeferredLighting(float4 pos : SV_Position) : SV_Target {
        Surface s = LoadGBuffer(pos.xy);                  // N, V, baseColor, sheenColor, sheenRoughness …
        float3 L = LightDir, H = normalize(L + s.V);
        float NoV = saturate(dot(s.N, s.V)) + 1e-4;
        float NoL = saturate(dot(s.N, L));
        float NoH = saturate(dot(s.N, H));
        float a   = max(s.sheenRoughness * s.sheenRoughness, 1e-3);

        float3 base  = BaseBRDF(s, L, H);                                  // 01 漫反射 + 02 GGX
        float3 sheen = s.sheenColor * D_Charlie(NoH, a) * V_Charlie(NoV, NoL, a, false);
        float  k     = SheenAlbedoScaling(NoV, NoL, s.sheenColor, s.sheenRoughness);
        return float4((sheen + base * k) * LightColor * NoL, 1.0);
    }`;

PipelineLab.mount({
    paramSteps: { rough: ['motiv', 'ashikhmin', 'neubelt', 'charlie', 'lambda', 'energy', 'layer', 'sample', 'prod'], thetaI: ['motiv', 'disney', 'neubelt', 'lambda', 'energy', 'layer', 'sample', 'prod'], lightAz: ['motiv', 'disney', 'neubelt', 'lambda', 'layer', 'prod'], sheenColor: ['motiv', 'disney', 'neubelt', 'lambda', 'layer', 'prod'], baseColor: ['motiv', 'disney', 'neubelt', 'lambda', 'layer', 'prod'], scaling: ['layer'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: 'BxDF 04 · 织物 Sheen：Charlie / Ashikhmin',
    lang: 'zh-CN', codeLang: 'hlsl', seed: 4,
    tagColors: { MOTIV: 'gray', BASE: 'magenta', 'D·ASH': 'olive', 'V·NEU': 'rust', 'D·CHARLIE': 'green', 'Λ·V': 'teal', ENERGY: 'indigo', LAYER: 'blue', SAMPLE: 'olive', PROD: 'rust' },

    params: [
        {
            id: 'rough', label: 'sheen r', min: 0.1, max: 1, step: 0.01, value: 0.5, digits: 2, title: '感知粗糙度 r；Charlie / Ashikhmin 都用 α = r²（Disney sheen 没有粗糙度）',
            desc: '<strong>sheen 粗糙度</strong>（感知值 r，D 与 Λ 内部用 α = r²）：纤维偏离竖直的程度。拖小 → D 的质量挤向 θh = 90°，波瓣贴着地平线、球的亮边更细更亮；拖到 0.3 以下看 Λ 拟合的误差（Step 06）和 E > 1 的红角（Step 07）。几乎所有 D / V / E / 采样预览都随它变，Disney sheen 没有粗糙度。',
        },
        {
            id: 'thetaI', label: 'θi', min: 0, max: 88, step: 1, value: 65, unit: '°', map: (d) => d * Math.PI / 180, title: '波瓣 / 半球热图 / 采样对比用的入射角（在 x–z 入射面内）；ENERGY 曲线上的竖线 μ = cos θi',
            desc: '波瓣图、半球热图、3D 波瓣所用的入射角（x–z 入射面内，0° = 正对，88° = 贴地平线）；Step 09 的采样对比把它当作 θo。拖到 80° 以上：sheen 的能量整个挤到掠射端，D 采样浪费过半样本。ENERGY 曲线上的竖线是 μ = cos θi。在球体渲染里它是主光的<strong>俯仰角</strong>（从视线方向向上抬 θi；偏航由光源方位控制，方位 = 0° 时球心处的入射角就是 θi）。',
        },
        {
            id: 'lightAz', label: '光源方位', min: -170, max: 170, step: 1, value: 135, unit: '°',
            desc: '球体渲染里主光的<strong>偏航角</strong>：绕竖直轴转动主光（0° = 光在视线正上方的竖直面内，正 = 右侧，±170° ≈ 逆光），俯仰由 θ<sub>i</sub> 决定。θ<sub>i</sub> 与光源方位都拖到 0° = <strong>头灯</strong>：Charlie 球中心暗、一圈逆反射亮边，GGX 高光缩到球心；方位 ±90°、θ<sub>i</sub> 较小 = 侧光，看明暗交界和 terminator softening（Step 06 的亮度剖面）；方位 ±150° 以上 = 逆光，看受光一侧的 rim。只影响球体渲染与亮度剖面。',
        },
        { id: 'sheenColor', label: 'sheenColor', value: '#eaa2bb', map: BX.hex2lin, desc: 'sheen 层的颜色（线性 RGB；不带 Fresnel 时就是 sheen 波瓣的权重）。它乘在每个 sheen 波瓣上，并以 <code>s = max3(sheenColor)</code> 进入 albedo scaling：调成白色，看 Step 08 白炉球（k = 1）的轮廓更橙、补偿时 base 被扣得更多。' },
        { id: 'baseColor', label: 'baseColor', value: '#4d0c22', map: BX.hex2lin, desc: 'sheen 之下的 Lambert base 颜色。深色（默认）让 sheen 的亮边更显眼；调亮后才看得出 albedo scaling 三种模式的差别（Step 08 的着色球）以及生产版与教学版在轮廓上的差（Step 10）。' },
        { id: 'scaling', label: 'albedo scaling', value: 'min', options: [{ value: 'none', label: 'none' }, { value: 'view', label: 'view (Filament)' }, { value: 'min', label: 'min (glTF / E&K)' }], title: 'sheen 叠在 base 上时 base 乘的能量补偿系数', desc: 'sheen 叠在 base 上时 base 乘的能量补偿 k：<strong>none</strong> = 1（不补偿，白炉里轮廓 > 1）；<strong>view</strong> = 1 − s·E(μo)（Filament，只看 NoV，也能乘到 IBL 上）；<strong>min</strong> = 入射 / 出射两项取更保守者（glTF / E&amp;K，正视略丢能量）。只作用于 Step 08 的白炉球和着色球。' },
    ],

    stages: [
        { id: 'keyLight', lifetime: 'param', run(S) { S.light = BX.keyLight(S.thetaI, S.lightAz); } },   // pitch = θi, yaw = 光源方位
        {
            id: 'lut', lifetime: 'resident', run(S) {
                // E(μ, r) 32×32：x = μo，y = 感知 r（texel 中心）。Charlie D + Charlie V（教学 / glTF），Charlie D + saturate(V_Neubelt)（Filament）
                const N = SHEEN_LUT_N, E = PL.buf(S, 'lutC', N * N), F = PL.buf(S, 'lutF', N * N), Q = PL.buf(S, 'lamErr', N * N);
                const vC = (mi, mu, a) => SH.vCharlie(mi, mu, a, false), vF = (mi, mu) => SH.vNeubeltSat(mi, mu);
                for (let y = 0; y < N; y++) {
                    const a = SH.aOf((y + 0.5) / N);
                    for (let x = 0; x < N; x++) {
                        const mu = (x + 0.5) / N, i = y * N + x;
                        E[i] = SH.albedoDV(SH.dCharlie, vC, mu, a, 24);
                        F[i] = SH.albedoDV(SH.dCharlie, vF, mu, a, 24);
                        Q[i] = SH.lambda(mu, a) / SH.lambdaNum(SH.dCharlie, mu, a, 256) - 1;   // 拟合 / 数值 − 1
                    }
                }
                let fMax = 0; for (let i = 0; i < N * N; i++) fMax = Math.max(fMax, F[i]);
                S.lutFMax = fMax;
                // 与参数无关的检查（只算一次）：归一化 ∫D cosθh dω 随 r；α = 1 的 Λ 参考曲线
                const n = 19, rs = Float32Array.from({ length: n }, (_, i) => 0.1 + 0.9 * i / (n - 1));
                S.normR = rs;
                S.normC = rs.map((r) => SH.normInt(SH.dCharlie, SH.aOf(r)));
                S.normA = rs.map((r) => SH.normInt(SH.dAsh, SH.aOf(r)));
                S.normG = rs.map((r) => SH.normInt(SH.dGGX, SH.aOf(r)));
                S.offShare = rs.map((r) => 1 / (1 + 4 * SH.aOf(r) ** 2));
                const mus = sheenMuGrid(64);
                S.lamFit1 = mus.map((m) => SH.lambda(m, 1));
                S.lamNum1 = mus.map((m) => SH.lambdaNum(SH.dCharlie, m, 1, 384));
            },
        },
        {
            id: 'ndf', lifetime: 'param', run(S) {
                const a = SH.aOf(S.rough), R = SHEEN_HEMI, wi = sheenWi(S);
                S.hDc = BX.hemiImage((h) => SH.dCharlie(h[2], a), R, S.hDc);
                S.hDa = BX.hemiImage((h) => SH.dAsh(h[2], a), R, S.hDa);
                S.hFd = BX.hemiImage((wo) => SH.fDisney(wi, wo), R, S.hFd);
                S.hFo = BX.hemiImage((wo) => SH.fOrder(wi, wo, a), R, S.hFo);
                S.hFc = BX.hemiImage((wo) => SH.fCharlie(wi, wo, a, false), R, S.hFc);
            },
        },
        {
            id: 'albedo', lifetime: 'param', run(S) {
                const n = 64, a = SH.aOf(S.rough), mus = sheenMuGrid(n);
                S.mus = mus;
                // Λ：拟合 vs 投影面积数值积分（当前 α；α = 1 参考在 resident 里）
                S.lamFit = mus.map((m) => SH.lambda(m, a));
                S.lamNum = mus.map((m) => SH.lambdaNum(SH.dCharlie, m, a, 384));
                S.lamSoft = mus.map((m) => SH.lambdaSoft(m, a));
                const lnTab = S.lamNum.map((v) => Math.log(Math.max(1e-12, v)));
                const lamN = (m) => Math.exp(SH.curveAt(lnTab, m));
                const E = (S.E = S.E || {}), Q = 32;   // (μi, φ) 各 32 段中点积分
                E.charlie = mus.map((m) => SH.albedoDV(SH.dCharlie, (mi, mo) => SH.vCharlie(mi, mo, a, false), m, a, Q));
                E.num = mus.map((m) => SH.albedoDV(SH.dCharlie, (mi, mo) => 1 / ((1 + lamN(mo) + lamN(mi)) * 4 * mi * mo), m, a, Q));
                E.soft = mus.map((m) => SH.albedoDV(SH.dCharlie, (mi, mo) => SH.vCharlie(mi, mo, a, true), m, a, Q));
                E.filament = mus.map((m) => SH.albedoDV(SH.dCharlie, (mi, mo) => SH.vNeubeltSat(mi, mo), m, a, Q));
                E.order = mus.map((m) => SH.albedoDV(SH.dAsh, (mi, mo) => SH.vNeubelt(mi, mo), m, a, Q));
                E.disney = mus.map((m) => SH.albedoG((mi, mo, nh, ih) => BX.pow5(1 - ih), m, Q));
                S.Eavg = {};
                for (const k of Object.keys(E)) { let s = 0; for (let i = 0; i < n; i++) s += E[k][i] * mus[i]; S.Eavg[k] = 2 * s / n; }
            },
        },
        {
            id: 'furnace', lifetime: 'param', run(S) {
                // 白炉：白色 Lambert base（ρ = 1）+ sheen，环境辐亮度 ≡ 1 → 出射 = E_total(μo)。只看 sheenColor 最大的通道。
                const n = S.mus.length, C = S.sheenColor, s = SH.max3(C), E = S.E.charlie, mus = S.mus;
                const fur = (S.furnace = S.furnace || {});
                fur.none = mus.map((m, i) => s * E[i] + 1);
                fur.view = mus.map((m, i) => s * E[i] + Math.max(0, 1 - s * E[i]));
                fur.min = mus.map((m, i) => {
                    let acc = 0;
                    for (let j = 0; j < n; j++) acc += Math.max(0, Math.min(1 - s * E[i], 1 - s * E[j])) * mus[j];
                    return s * E[i] + 2 * acc / n;
                });
            },
        },
        {
            id: 'sampling', lifetime: 'param', run(S) {
                const a = SH.aOf(S.rough), mu = Math.max(0.05, Math.cos(S.thetaI)), wo = [Math.sqrt(1 - mu * mu), 0, mu];
                const f = (wi) => SH.fCharlie(wi, wo, a, false);
                S.sTruth = SH.albedoDV(SH.dCharlie, (mi, mo) => SH.vCharlie(mi, mo, a, false), mu, a, 96);
                S.hInteg = BX.hemiImage((wi) => f(wi) * wi[2], SHEEN_HEMI, S.hInteg);
                const strat = {
                    uniform: (u1, u2) => { const wi = BX.sampleUniform(u1, u2); return { wi, w: f(wi) * wi[2] * 2 * Math.PI }; },
                    cosine: (u1, u2) => { const wi = BX.sampleCosine(u1, u2); return { wi, w: wi[2] > 1e-6 ? f(wi) * Math.PI : 0 }; },
                    ndf: (u1, u2) => {
                        // 按 D·cosθh 采 h：CDF(θh) = sin^{2+1/α}θh → sinθh = u^{α/(1+2α)}
                        const st = Math.pow(u1, a / (1 + 2 * a)), ct = Math.sqrt(Math.max(0, 1 - st * st)), p = 2 * Math.PI * u2;
                        const h = [st * Math.cos(p), st * Math.sin(p), ct], oh = BX.dot(wo, h);
                        if (oh <= 0) return { wi: null, w: 0 };
                        const wi = [2 * oh * h[0] - wo[0], 2 * oh * h[1] - wo[1], 2 * oh * h[2] - wo[2]];
                        if (wi[2] <= 0) return { wi: null, w: 0 };
                        const pdf = SH.dCharlie(ct, a) * ct / (4 * oh);
                        return { wi, w: pdf > 0 ? f(wi) * wi[2] / pdf : 0 };
                    },
                };
                const K = 32, NMAX = 256, checks = [1, 2, 4, 8, 16, 32, 64, 128, 256];
                S.sN = checks; S.sRmse = {}; S.sDots = {}; S.sLost = {};
                for (const [name, fn] of Object.entries(strat)) {
                    const rng = PL.rng(0x5eed ^ name.length * 7919), err = new Float64Array(checks.length), dots = [];
                    let lost = 0;
                    for (let k = 0; k < K; k++) {
                        let sum = 0, ci = 0;
                        for (let i = 1; i <= NMAX; i++) {
                            const r = fn(rng(), rng());
                            if (!r.wi || r.w === 0) lost++;
                            else if (k === 0 && dots.length < 200) dots.push(r.wi);
                            sum += r.w;
                            if (i === checks[ci]) { const e = sum / i - S.sTruth; err[ci] += e * e; ci++; }
                        }
                    }
                    S.sRmse[name] = Array.from(err, (e) => Math.sqrt(e / K) / Math.max(1e-9, S.sTruth));
                    S.sDots[name] = dots;
                    S.sLost[name] = lost / (K * NMAX);
                }
            },
        },
        {
            id: 'prodcmp', lifetime: 'param', run(S) {
                const res = 112, r = S.rough, s = SH.max3(S.sheenColor);
                const Et = (m) => SH.curveAt(S.E.charlie, m);
                S.fTeach = sheenCloth(S, 'charlie', 'min', Et);
                const C = S.sheenColor, B = S.baseColor, a = SH.aOf(r);
                S.fProd = (wi, wo) => {   // Filament 风格：D_Charlie · saturate(V_Neubelt)，base × (1 − s·LUT(NoV, r))
                    const fs = SH.fFilament(wi, wo, a), k = Math.max(0, 1 - s * SH.lutAt(S.lutF, SHEEN_LUT_N, wo[2], r));
                    return [C[0] * fs + B[0] * k / Math.PI, C[1] * fs + B[1] * k / Math.PI, C[2] * fs + B[2] * k / Math.PI];
                };
                const lt = (S.pLt = sheenSphereLum(S.fTeach, S, res, S.pLt)), lp = (S.pLp = sheenSphereLum(S.fProd, S, res, S.pLp));
                const d = PL.buf(S, 'pDiff', res * res);
                let peak = 0;
                for (let i = 0; i < res * res; i++) { d[i] = isFinite(lt[i]) ? lp[i] - lt[i] : NaN; if (isFinite(lt[i])) peak = Math.max(peak, lt[i]); }
                S.pPeak = peak || 1;
                // 球体水平直径上的亮度剖面（terminator softening 对比）
                const N = 241, lights = sheenLights(S), fa = sheenCloth(S, 'charlie'), fb = sheenCloth(S, 'soft');
                S.rowU = Float32Array.from({ length: N }, (_, i) => -1 + 2 * i / (N - 1));
                S.rowA = S.rowU.map((u) => sheenPixelLum(fa, lights, u, 0));
                S.rowB = S.rowU.map((u) => sheenPixelLum(fb, lights, u, 0));
                S.rowT = (() => { const l = S.light; for (let i = 0; i < N; i++) { const u = S.rowU[i]; if (u * l[0] + Math.sqrt(Math.max(0, 1 - u * u)) * l[2] > 0) return u; } return NaN; })();
            },
        },
        {
            id: 'lobe3d', lifetime: 'param', lazy: true, run(S) {   // f(wi, ωo) over the ωo hemisphere for Charlie sheen and GGX spec (same α, fixed wi = sheenWi)
                const { nu, nv } = SHEEN_L3D, NU = nu + 1, NV = nv + 1, a = SH.aOf(S.rough), wi = sheenWi(S);
                const C = PL.buf(S, 'l3dC', NU * NV), G = PL.buf(S, 'l3dG', NU * NV), M = { c: 0, g: 0, atC: 0, atG: 0 };
                for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) {
                    const wo = sheenL3dDir(i % nu, j), k = j * NU + i;
                    C[k] = SH.fCharlie(wi, wo, a, false);
                    G[k] = SH.fGGX(wi, wo, a, 0.04);
                    if (C[k] > M.c) { M.c = C[k]; M.atC = k; }
                    if (G[k] > M.g) { M.g = G[k]; M.atG = k; }
                }
                M.c = M.c || 1; M.g = M.g || 1;
                S.l3dMax = M;
            },
        },
    ],

    library: {
        title: '附录 · 辅助函数（HLSL）',
        intro: 'Step 10 的 <code>PS_DeferredLighting</code> 从 G-buffer 读取着色输入。下面是本页假定的布局与读取函数——真实引擎的打包格式各不相同（法线常用八面体编码、粗糙度与别的通道共用），但套路一样：逐像素 <code>Texture2D.Load</code>，再由深度反投影出世界坐标得到 V。',
        blocks: [
            {
                title: 'G-buffer 布局与 LoadGBuffer', code: `
                // 本页假定的 G-buffer：A = 世界法线（×0.5 + 0.5），B = baseColor，C = sheenColor + sheenRoughness
                Texture2D<float4> GBufferA   : register(t0);   // RGB10A2_UNORM
                Texture2D<float4> GBufferB   : register(t1);   // RGBA8_SRGB：读取时硬件已转成线性
                Texture2D<float4> GBufferC   : register(t2);   // RGBA8_SRGB：rgb = sheenColor（转线性），a = sheenRoughness（alpha 不做 sRGB 转换）
                Texture2D<float>  SceneDepth : register(t3);   // 设备深度（D3D 约定 0 … 1）
                cbuffer ViewCB : register(b0) { float4x4 InvViewProj; float3 CameraPos; float pad2; float2 InvViewportSize; float2 pad3; };

                // PS_DeferredLighting 的着色输入（世界空间，N、V 已归一化）
                struct Surface {
                    float3 N;                 // 着色法线
                    float3 V;                 // 指向相机
                    float3 baseColor;         // 线性
                    float3 sheenColor;        // 线性
                    float  sheenRoughness;    // 感知 r；α = r² 在用到时再算
                };

                // 读一个像素的 G-buffer：Load 取整数像素坐标（无过滤）；V 由深度反投影出世界坐标再指向相机
                Surface LoadGBuffer(float2 pixel) {
                    int3   p   = int3(pixel, 0);                    // SV_Position.xy 在像素中心，截断即整数坐标
                    float4 a   = GBufferA.Load(p);
                    float4 b   = GBufferB.Load(p);
                    float4 c   = GBufferC.Load(p);
                    float  z   = SceneDepth.Load(p);
                    float2 uv  = pixel * InvViewportSize;
                    float4 ndc = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
                    float4 wp  = mul(InvViewProj, ndc);
                    Surface s;
                    s.N              = normalize(a.xyz * 2.0 - 1.0);
                    s.V              = normalize(CameraPos - wp.xyz / wp.w);
                    s.baseColor      = b.rgb;
                    s.sheenColor     = c.rgb;
                    s.sheenRoughness = c.a;
                    return s;
                }`,
            },
        ],
    },

    intro: {
        title: '织物 Sheen：从 Disney sheen 到 Charlie',
        flow: ['为什么 GGX 做不出丝绒', 'Disney sheen 基线', 'Ashikhmin 倒置高斯', 'Neubelt V（The Order）', 'Charlie D', 'Λ 拟合 / V', 'E(μ, r) LUT', 'sheen over base', '采样', '生产版'],
        html: '<p>丝绒、天鹅绒、桃皮绒的"边缘发亮"来自竖立的纤维：微法线集中在<strong>水平方向</strong>，正好与 GGX 的假设相反。本模块把三种常用 sheen 模型逐项拆开——Disney sheen（Burley 2012）、Ashikhmin–Premože 2007 倒置高斯配 Neubelt–Pettineo 2013 分母、Charlie（Estevez &amp; Kulla 2017，glTF <code>KHR_materials_sheen</code> / Filament 使用）——再算它们的方向反照率 E(μ, r)，用 albedo scaling 把 sheen 叠到 base 上。</p>' +
            '<p>所有预览都由页面内的 JS 参考实现当场算出：球体是逐像素求值的 BRDF，热图是半球上逐方向的 D / f，反照率与 LUT 是数值积分。<strong>约定</strong>：n = +Z，wi → 光源，wo → 观察者；sheen 粗糙度用感知值 r，Charlie 与 Ashikhmin 都用 <code>α = r²</code>（论文里 Charlie 的 "r" 就是这里的 α）；sheen 波瓣按 sheenColor = 1 计算、不带 Fresnel，着色时再乘 sheenColor。</p>' +
            BX.nav('sheen'),
    },

    steps: [
        // ───────────────────────────── 01 MOTIV ─────────────────────────────
        {
            id: 'motiv', title: '为什么布料需要自己的模型：纤维让法线分布"倒过来"', tag: 'MOTIV',
            desc: '微表面模型假设表面由一片片随机朝向的小镜面组成，GGX 的 NDF 在 <code>h = n</code> 处最大——高光跟着镜面方向走，粗糙度只是把它摊开。丝绒是另一回事：短纤维<strong>竖立</strong>在底布上，纤维侧面的法线几乎都<strong>垂直于宏观法线</strong>。于是：① 观察方向掠射时（球的轮廓），half vector 也接近水平，正好命中这批纤维 → <strong>边缘发亮</strong>；② 光和视线同向时（头灯）h = wo，轮廓处 θh → 90° 仍然亮 → <strong>逆反射</strong>；③ 光从背后来时纤维前向散射，形成 rim。GGX 无论怎么调粗糙度，NDF 的峰都在 θh = 0，做不出这种"中心暗、边缘亮"。',
            formula: [
                'D_GGX(θ_h) 在 θ_h = 0 最大                 // 镜面槽：法线贴着 n',
                'D_sheen(θ_h) 在 θ_h → 90° 最大            // 竖立纤维：法线贴着地平线',
                '轮廓处 θ_o → 90°，头灯 wi = wo → h = wo → θ_h → 90°   // 逆反射也落在峰上',
            ],
            previews: [
                { id: 'mGGX', label: 'Lambert + GGX', sub: '同一 r，F0 = 0.04', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'ggx')) },
                { id: 'mCharlie', label: 'Lambert + Charlie sheen', sub: 'D_Charlie · V_Charlie · sheenColor', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'charlie')) },
                {
                    id: 'mNDF', label: 'D(θh) / max', sub: '同一 α = r² 下三种 NDF 的形状', size: [360, 180],
                    render: (c, S) => {
                        const a = SH.aOf(S.rough), rad = (d) => d * Math.PI / 180, mk = (D) => { let m = 0; for (let d = 0; d <= 90; d += 0.5) m = Math.max(m, D(Math.cos(rad(d)), a)); return m || 1; };
                        const mg = mk(SH.dGGX), mc = mk(SH.dCharlie), ma = mk(SH.dAsh);
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, 1.05], xLabel: 'θh (°)', series: [
                                { fn: (d) => SH.dGGX(Math.cos(rad(d)), a) / mg, color: SHEEN_COL.ggx, label: 'GGX' },
                                { fn: (d) => SH.dAsh(Math.cos(rad(d)), a) / ma, color: SHEEN_COL.order, label: 'Ashikhmin 倒置高斯' },
                                { fn: (d) => SH.dCharlie(Math.cos(rad(d)), a) / mc, color: SHEEN_COL.charlie, label: 'Charlie sin^(1/α)' },
                            ],
                        });
                    },
                },
                {
                    id: 'mLobe', label: 'f(wi, wo) · 入射面', sub: '黄线 = wi（θi），虚线 = 镜面方向；log 3 decades', size: [360, 200],
                    render: (c, S) => { const a = SH.aOf(S.rough), wi = sheenWi(S); BX.lobe(c, { wi, log: 3, series: [{ fn: (wo) => SH.fGGX(wi, wo, a, 0.04), color: SHEEN_COL.ggx, label: 'GGX spec' }, { fn: (wo) => SH.fCharlie(wi, wo, a, false), color: SHEEN_COL.charlie, label: 'Charlie sheen' }] }); },
                },
                {
                    id: 'mLobe3d', needs: ['lobe3d'], large: true, label: '3D：Charlie sheen 与 GGX 的整个波瓣',
                    sub: '固定 wi（黄箭头，指向光源，θi），半径 = f(wi, ωo)，64 × 24 个方向；两个波瓣<strong>各按自身最大值归一化</strong>。绿 = Charlie sheen（越亮越大），蓝 = GGX（同一 α，F0 = 0.04），灰圈 = 地平线。GGX 贴着镜面方向；Charlie 沿地平线摊成月牙，峰在光源一侧（逆反射）',
                    render: (c, S) => drawSheenLobe3d(c, S),
                },
            ],
            notes: '把 <strong>θi 与光源方位都拖到 0°</strong>（头灯）：GGX 的高光缩到球心，Charlie 球却是中心暗、一圈亮边——这就是逆反射。把光源方位拖到 ±150° 以上（逆光）：GGX 几乎看不到，Charlie 在受光一侧的轮廓上出现 rim。波瓣图里 GGX 的峰在虚线（镜面方向），Charlie 的能量堆在 wo 掠射的两端，包括 wi 自己所在的一侧（逆反射）。3D 图把整个半球画出来：Charlie 是一只<strong>向地平线张开的浅碗</strong>——除了镜面方向附近的缺口，从任何掠射方向看都亮，最高处在光源一侧；GGX 只是镜面方向上的一片。拖 θi 到 20° 左右，GGX 竖起来贴近 n，Charlie 的碗仍偏在光源一侧；只有 θi = 0 时它才绕成对 n 旋转对称的整圈（天顶处 θh = 0 ⇒ f = 0）。',
            warn: '把 GGX 粗糙度调到 1 并不能"变成布料"：峰仍在 θh = 0，只是更宽、更暗。也不要指望用 Fresnel 的掠射增亮代替 sheen——Fresnel 只在 wi·h → 0 时增亮，头灯下 wi·h = 1，完全没有贡献。',
        },
        // ───────────────────────────── 02 BASE ─────────────────────────────
        {
            id: 'disney', title: 'Disney sheen（Burley 2012）：一个 Schlick 形状的基线', tag: 'BASE',
            desc: 'Disney principled BRDF 给布料只加了一项便宜的经验项：用 Schlick Fresnel 的形状 <code>(1 − cosθd)⁵</code> 在 diffuse 上叠一点掠射亮度，<code>Csheen</code> 在白色和 baseColor 的色相之间插值。它没有 NDF、没有粗糙度，也不做能量守恒，只用来给 diffuse 的边缘补一点光。注意 θd 是 wi 与 h 的夹角：只有 wi 和 wo 分别在两侧、都接近掠射（前向散射）时 cosθd 才小；<strong>头灯下 wi = wo ⇒ θd = 0 ⇒ 这一项恒为 0</strong>，所以它做不出丝绒的逆反射。',
            formula: [
                'f_sheen = sheen · C_sheen · (1 − cosθ_d)^5      // cosθ_d = wi·h，没有 1/π',
                'C_sheen = lerp(1, C_tint, sheenTint),  C_tint = baseColor / lum(baseColor)',
                'f_d,total = baseColor/π · F_Burley + f_sheen    // 整体再乘 (1 − metallic)',
            ],
            code: {
                title: 'HLSL · Disney sheen', src: `
                // Burley 2012：sheen ∈ [0,1]，sheenTint ∈ [0,1]，直接加在 diffuse 上（不除 π）
                float3 Sheen_Disney(float LoH, float3 baseColor, float sheen, float sheenTint) {
                    float  lum    = dot(baseColor, float3(0.3, 0.6, 0.1));    // BRDF Explorer 里的近似亮度
                    float3 tint   = lum > 0.0 ? baseColor / lum : 1.0;       // 只保留色相 / 饱和度
                    float3 Csheen = lerp(1.0, tint, sheenTint);
                    return sheen * Csheen * Pow5(1.0 - LoH);
                }`,
            },
            previews: [
                { id: 'dSphere', label: 'Lambert + Disney sheen', sub: 'sheen = 1，C_sheen = sheenColor', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'disney')) },
                { id: 'dBase', label: '只有 Lambert', sub: '对照：同一 baseColor', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'none')) },
                {
                    id: 'dLobe', label: 'f_sheen · 入射面', sub: '灰虚线 = Lambert 1/π 参照', size: [360, 200],
                    render: (c, S) => { const wi = sheenWi(S); BX.lobe(c, { wi, series: [{ fn: (wo) => SH.fDisney(wi, wo), color: SHEEN_COL.disney, label: 'Disney (1 − cosθd)⁵' }, { fn: (wo) => (wo[2] > 0 ? 1 / Math.PI : 0), color: SHEEN_COL.lambert, dash: [4, 3], label: 'Lambert 1/π' }] }); },
                },
                {
                    id: 'dHemi', label: 'f_sheen(wi, ·) 半球', sub: '等面积投影，边缘 = 地平线；黄点 = wi', render: (c, S) => sheenHemi(c, S.hFd, { marks: [{ w: sheenWi(S) }] }),
                    probe: sheenHemiProbe('hFd', ' /sr'),
                },
            ],
            notes: '把 θi 拖到 80° 以上，热图里只有 wi 对面（φ = 180°）的地平线附近有值：这是一个<strong>前向掠射</strong>波瓣。θi 与光源方位都拖到 0°（头灯），Disney 球与纯 Lambert 球几乎一样——头灯下它没有贡献。它的最大值也只有 1/sr 量级、只在极窄的掠射区，方向反照率 E ≤ 0.08（见 ENERGY 步）。',
            warn: '代码里 sheen 项<strong>不除 π</strong>，直接与 <code>baseColor/π · F_d</code> 相加。移植到别的引擎时如果顺手除了 π，sheen 会弱 3 倍；如果再乘一次 sheenColor 和 tint，就重复着色了。本页把 <code>sheen · C_sheen</code> 合并为 sheenColor。',
        },
        // ───────────────────────────── 03 D·ASH ─────────────────────────────
        {
            id: 'ashikhmin', title: 'Ashikhmin–Premože 倒置高斯：把 Beckmann 的 tan 换成 cot', tag: 'D·ASH',
            desc: 'Ashikhmin &amp; Premože（2007，"Distribution-based BRDFs"）的思路是：BRDF 的形状主要由 D 决定，那就直接设计一个"法线贴着地平线"的 D。把 Beckmann 里的 <code>tan²θ</code> 换成 <code>cot²θ</code>、<code>cos⁴</code> 换成 <code>sin⁴</code>，高斯的中心就从 θh = 0 搬到了 θh = 90°。再加一个常数偏置 1：Neubelt &amp; Pettineo 指出它让正对光源时也有一点高光，相当于把一部分 diffuse 折进了 specular 波瓣。A = 4 是幅度（沿用 Ashikhmin），σ 控制纤维偏离竖直的程度。',
            formula: [
                'D_inv(h) = 1 / (π (1 + A σ^2)) · ( 1 + A · exp(−cot^2θ_h / σ^2) / sin^4θ_h ),   A = 4',
                '∫ exp(−cot^2θ/σ^2) / sin^4θ · cosθ dω = 2π ∫_0^∞ u e^{−u^2/σ^2} du = π σ^2     // u = cotθ',
                '⇒ ∫ D_inv cosθ_h dω = (π + A π σ^2) / (π (1 + A σ^2)) = 1                       // 投影面积归一化严格成立',
                'σ = α = r^2   // 本页约定（Filament D_Ashikhmin）；The Order 课程笔记直接写 α',
            ],
            code: { title: 'HLSL · D_Ashikhmin（与 Common 中相同）', src: HLSL_SHEEN_D.split('\n').slice(7).join('\n') },
            previews: [
                {
                    id: 'aD', label: 'D_inv(θh) 拆成两部分', sub: '常数偏置 + 倒置高斯（当前 σ = r²）', size: [360, 180],
                    render: (c, S) => {
                        const a = SH.aOf(S.rough), rad = (d) => Math.cos(d * Math.PI / 180);
                        let m = 0; for (let d = 0; d <= 90; d += 0.25) m = Math.max(m, SH.dAsh(rad(d), a));
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, m * 1.08], xLabel: 'θh (°)', series: [
                                { fn: (d) => SH.dAsh(rad(d), a), color: SHEEN_COL.order, label: 'D_inv 总和' },
                                { fn: (d) => SH.dAshOffset(rad(d), a), color: SHEEN_COL.lambert, dash: [4, 3], label: '偏置 1/(π(1+4σ²))' },
                                { fn: (d) => SH.dAsh(rad(d), a) - SH.dAshOffset(rad(d), a), color: '#e8c25a', dash: [2, 2], label: '倒置高斯部分' },
                            ],
                        });
                    },
                },
                { id: 'aHemi', label: 'D_inv(h) 半球', sub: 'h 的等面积投影：亮环 = 水平微法线', render: (c, S) => sheenHemi(c, S.hDa), probe: sheenHemiProbe('hDa') },
                {
                    id: 'aNorm', label: '数值检查 ∫D cosθh dω', sub: '19 个 r，1D 中点积分 4000 段', size: [360, 180],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0.1, 1], y: [0, 1.2], xLabel: 'r', series: [
                            { data: S.normA, x: S.normR, color: SHEEN_COL.order, label: '∫ D_inv cos dω' },
                            { data: S.offShare, x: S.normR, color: SHEEN_COL.lambert, dash: [4, 3], label: '其中偏置占比 1/(1+4σ²)' },
                        ], hmarks: [{ y: 1, color: '#3a4148' }],
                    }),
                },
            ],
            notes: '拖 r：σ 越小，倒置高斯越挤向 θh = 90°（热图上的亮环越细），偏置的占比 <code>1/(1+4σ²)</code> 越大——r = 0.1 时几乎全部投影面积都在偏置里，D 变成"常数 + 边缘尖峰"。r = 1 时偏置只占 20%。归一化曲线始终贴着 1，与上面的解析推导一致。',
            warn: '<strong>σ 的映射各家不同</strong>：The Order 笔记写 <code>exp(−cot²θ/α²)</code> 但没说 α 和美术粗糙度的关系；Filament 的 <code>D_Ashikhmin(roughness)</code> 传入的是 <code>roughness = r²</code>；UE4 Cloth 用的是另一个形状 <code>D_InvGGX</code>（倒置 GGX），不能互换参数。E&amp;K 2017 还提到 Ashikhmin–Shirley 早期的 velvet 分布在光源方向有"暗斑"，这是促使他们换成 sin 形分布的原因之一。',
        },
        // ───────────────────────────── 04 V·NEU ─────────────────────────────
        {
            id: 'neubelt', title: 'The Order: 1886 的布料 BRDF：去掉 G，换一个更平滑的分母', tag: 'V·NEU',
            desc: 'Neubelt &amp; Pettineo（2013）在 The Order 里用上一步的 velvet D，并做了两处取舍：① <strong>不要 Smith G</strong>——他们试过，发现 G 把掠射处的 sheen 压得太暗；纤维是半透明的，主要效果是散射而不是 V 形槽的遮挡（与 Ashikhmin 的建议一致）；② 用 Ashikhmin 的分母 <code>4(n·l + n·v − (n·l)(n·v))</code> 代替 <code>4(n·l)(n·v)</code>，它在掠射时不会爆掉。再加一个 <code>(1 − F)·c_diff/π</code> 的漫反射，就能覆盖丝绒以外的棉布、麂皮。这个分母后来就是 Filament / glTF 里的 "Ashikhmin visibility" <code>V_Neubelt</code>。',
            formula: [
                'f_cloth = (1 − F(v·h)) · c_diff / π + F(v·h) · D_velvet(h) / (4 (n·l + n·v − (n·l)(n·v)))',
                'V_Neubelt = 1 / (4 (NoL + NoV − NoL·NoV))      // NoL = 0 时 = 1/(4 NoV)，有界',
                'V_CookTorrance(G=1) = 1 / (4 NoL NoV)          // NoL → 0 时 → ∞',
            ],
            code: { title: 'HLSL · V_Neubelt（与 Common 中相同）', src: HLSL_SHEEN_V.split('\n').slice(-4).join('\n') },
            previews: [
                {
                    id: 'nV', label: 'V(NoL)，NoV = cos θi', sub: 'log Y：哪种分母在掠射时最"烫"', size: [360, 190],
                    render: (c, S) => {
                        const a = SH.aOf(S.rough), nv = Math.max(0.02, Math.cos(S.thetaI));
                        PL.draw.plot(c, {
                            x: [0.01, 1], y: [0.01, 30], logY: true, xLabel: 'NoL', series: [
                                { fn: (nl) => SH.vNoG(nl, nv), color: SHEEN_COL.lambert, dash: [4, 3], label: '1/(4·NoL·NoV)，G = 1' },
                                { fn: (nl) => SH.vNeubelt(nl, nv), color: SHEEN_COL.order, label: 'V_Neubelt' },
                                { fn: (nl) => SH.vCharlie(nl, nv, a, false), color: SHEEN_COL.charlie, label: 'V_Charlie（当前 α）' },
                            ], hmarks: [{ y: 1, color: '#3a4148' }],
                        });
                    },
                },
                {
                    id: 'nLobe', label: 'f = D_inv · V_Neubelt · 入射面', sub: '灰虚线 = Lambert 1/π 参照', size: [360, 200],
                    render: (c, S) => { const a = SH.aOf(S.rough), wi = sheenWi(S); BX.lobe(c, { wi, series: [{ fn: (wo) => SH.fOrder(wi, wo, a), color: SHEEN_COL.order, label: 'Ashikhmin · Neubelt' }, { fn: (wo) => (wo[2] > 0 ? 1 / Math.PI : 0), color: SHEEN_COL.lambert, dash: [4, 3], label: 'Lambert 1/π' }] }); },
                },
                { id: 'nHemi', label: 'f(wi, ·) 半球', sub: '黄点 = wi', render: (c, S) => sheenHemi(c, S.hFo, { marks: [{ w: sheenWi(S) }] }), probe: sheenHemiProbe('hFo', ' /sr') },
                { id: 'nSphere', label: 'Lambert + Ashikhmin·Neubelt', sub: 'F ≡ sheenColor（省略 Schlick）', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'order')) },
            ],
            notes: 'V 图里：G = 1 的 Cook-Torrance 分母在 NoL → 0 时发散；V_Neubelt 停在 <code>1/(4·NoV)</code>；V_Charlie 同样有界——拟合的 Λ 在掠射时大致按 1/NoL 增长，G 恰好抵消分母里的 NoL，数值比 V_Neubelt 略低。差别在 NoV 也掠射时被放大：把 θi 拖到 85° 以上，V_Neubelt 在 NoL → 0 时涨到 <code>1/(4·NoV)</code> ≈ 2.9（Filament 用 saturate 截到 1），这正是布料轮廓的亮度来源。热图里偏置项让整个半球都有一层底色（类似 diffuse），倒置高斯在 wo 掠射的一圈上叠出亮环。',
            warn: 'V_Neubelt 在 NoL、NoV <strong>同时</strong>趋于 0 时仍会发散，Filament 因此套了 <code>saturate</code>。不带 G 的 BRDF 本身不保证能量守恒：glTF 规范明确提醒，用 Ashikhmin visibility 再配 albedo scaling 时整体不再严格守恒（ENERGY 步里能看到 Charlie·V_Neubelt 的 E 在掠射处偏大）。',
        },
        // ───────────────────────────── 05 D·CHARLIE ─────────────────────────────
        {
            id: 'charlie', title: 'Charlie 分布（Estevez &amp; Kulla 2017）：指数化的 sin', tag: 'D·CHARLIE',
            desc: 'Sony Imageworks 想要一个比倒置高斯更柔和、更便宜的 D：直接用 sin<sup>1/α</sup>θh。α = 1 时 D ∝ sinθh，非常宽，接近 diffuse 的观感；α → 0 时峰越来越尖地贴到地平线。它的归一化常数可以手算——sin<sup>2+1/α</sup>θ 正好是 D·cosθ·sinθ 的原函数（差一个常数），这也让按 D 采样有了解析的反函数（SAMPLE 步）。论文对比图里，它没有 Ashikhmin–Shirley velvet 在光源方向的暗斑，过渡也更软。',
            formula: [
                'D_Charlie(h) = (2 + 1/α) · sin^{1/α}θ_h / (2π),    α = r^2 ∈ (0, 1]',
                '∫ D cosθ_h dω = (2 + 1/α) ∫_0^{π/2} sin^{1+1/α}θ cosθ dθ = [sin^{2+1/α}θ]_0^{π/2} = 1',
                'α = 1 → D = 3 sinθ_h / 2π;     α → 0 → 质量全部挤到 θ_h = 90°',
            ],
            code: { title: 'HLSL · D_Charlie（与 Common 中相同）', src: HLSL_SHEEN_D.split('\n').slice(0, 7).join('\n') },
            previews: [
                {
                    id: 'cD', label: 'D_Charlie(θh)', sub: '实线 = 当前 α；虚线 = r = 0.3 / 0.6 / 1.0', size: [360, 180],
                    render: (c, S) => {
                        const a = SH.aOf(S.rough), cs = (d) => Math.cos(d * Math.PI / 180);
                        PL.draw.plot(c, {
                            x: [0, 90], y: [0, Math.max(0.6, SH.dCharlie(0, a) * 1.08)], xLabel: 'θh (°)', series: [
                                { fn: (d) => SH.dCharlie(cs(d), 0.09), color: '#2f6d43', dash: [3, 3], label: 'r = 0.3' },
                                { fn: (d) => SH.dCharlie(cs(d), 0.36), color: '#3f8f58', dash: [3, 3], label: 'r = 0.6' },
                                { fn: (d) => SH.dCharlie(cs(d), 1), color: '#4fae6c', dash: [3, 3], label: 'r = 1.0' },
                                { fn: (d) => SH.dCharlie(cs(d), a), color: SHEEN_COL.charlie, width: 2.2, label: 'Charlie（当前）' },
                                { fn: (d) => SH.dAsh(cs(d), a), color: SHEEN_COL.order, label: 'Ashikhmin（当前）' },
                            ],
                        });
                    },
                },
                { id: 'cHemi', label: 'D_Charlie(h) 半球', sub: '与上一步 D_inv 同一投影、同一 α', render: (c, S) => sheenHemi(c, S.hDc), probe: sheenHemiProbe('hDc') },
                {
                    id: 'cNorm', label: '数值检查 ∫D cosθh dω', sub: 'Charlie / Ashikhmin / GGX 对照', size: [360, 180],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0.1, 1], y: [0.9, 1.1], xLabel: 'r', series: [
                            { data: S.normG, x: S.normR, color: SHEEN_COL.ggx, label: 'GGX' },
                            { data: S.normA, x: S.normR, color: SHEEN_COL.order, dash: [5, 3], label: 'Ashikhmin' },
                            { data: S.normC, x: S.normR, color: SHEEN_COL.charlie, dash: [2, 2], label: 'Charlie' },
                        ],
                    }),
                    probe: (x, y, S) => `max |∫ − 1|：Charlie ${Math.max(...S.normC.map((v) => Math.abs(v - 1))).toExponential(1)}，Ashikhmin ${Math.max(...S.normA.map((v) => Math.abs(v - 1))).toExponential(1)}`,
                },
            ],
            notes: '对比 D 曲线：同一 α 下 Charlie 在 θh = 0 处严格为 0（sin 0 = 0），Ashikhmin 因为偏置在中心仍有 <code>1/(π(1+4σ²))</code>；r 小时 Charlie 的峰比倒置高斯宽而软。r = 1 的 Charlie 几乎是一条斜线，这就是"最粗糙的 sheen 像一层柔光"的来源。',
            warn: 'Charlie 论文里的参数叫 <strong>r</strong>，但它就是 D 的指数参数，等于 glTF / Filament 的 <code>α = sheenRoughness²</code>。把感知 r 直接塞进 <code>1/r</code>，或者对已经平方过的值再平方一次，是最常见的移植错误——看起来只是"粗糙度不太对"，很难发现。α → 0 时 <code>pow(sin², 0.5/α)</code> 指数巨大，fp16 下要像 Filament 那样把 sin²θh 钳到 2⁻⁷。',
        },
        // ───────────────────────────── 06 Λ·V ─────────────────────────────
        {
            id: 'lambda', title: 'Charlie 的 Λ：投影面积归一化 → 曲线拟合 → V', tag: 'Λ·V',
            desc: '要配 Smith G，需要这个 D 的 Λ。E&amp;K 按 Heitz 2014 的投影面积归一化 <code>∫ G₁ ⟨ω·m⟩ D(m) dm = cosθ</code> 数值求 G，再对 <code>ln Λ</code> 拟合一个 5 参数曲线：r = 0 和 r = 1 各拟合一组，中间用 <code>(1 − α)²</code> 插值；cosθ ≥ 0.5 时用对称延拓 <code>2L(0.5) − L(1 − cosθ)</code>，保证 Λ(1) ≈ 0。G 用高度相关形式。最后一个非物理修改：只对<strong>光源侧</strong>的 Λ 加指数 <code>1 + 2(1 − cosθi)⁸</code>，让明暗交界线更柔（BRDF 因此不再互易）。本步用同一个恒等式把 Λ 数值积分出来，<strong>直接检验拟合</strong>。',
            formula: [
                'Λ_num(μ) = (1/μ) ∫_{H^2} max(0, ω·m) D(m) dm − 1          // Heitz 2014，φ 积分有闭式',
                'L(x) = a / (1 + b x^c) + d x + e',
                'Λ(μ) = e^{L(μ)}  (μ < 0.5);    e^{2L(0.5) − L(1 − μ)}  (μ ≥ 0.5)',
                'p = (1 − α)^2 p_{r=0} + (1 − (1 − α)^2) p_{r=1}            // Table 1：a b c d e',
                'V_Charlie = 1 / ((1 + Λ(μ_o) + Λ(μ_i)) · 4 μ_i μ_o),   Λ\'(μ_i) = Λ(μ_i)^{1 + 2(1 − μ_i)^8}',
            ],
            code: { title: 'HLSL · Λ fit + V_Charlie + V_Neubelt（与 Common 中相同）', src: HLSL_SHEEN_V },
            previews: [
                {
                    id: 'lLam', label: 'Λ(μ)：拟合 vs 数值积分', sub: '绿 = 当前 α，灰 = α = 1；实线拟合、虚线数值', size: [360, 200],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, 1], y: [1e-3, 100], logY: true, xLabel: 'μ = cosθ', series: [
                            { data: S.lamFit1, x: S.mus, color: '#8a949e', label: 'fit α=1' },
                            { data: S.lamNum1, x: S.mus, color: '#c8d0d8', dash: [4, 3], label: 'numeric α=1' },
                            { data: S.lamFit, x: S.mus, color: SHEEN_COL.charlie, width: 2, label: 'fit（当前 α）' },
                            { data: S.lamNum, x: S.mus, color: SHEEN_COL.num, dash: [4, 3], label: 'numeric（当前 α）' },
                            { data: S.lamSoft, x: S.mus, color: SHEEN_COL.soft, dash: [2, 2], label: 'Λ′ 光源侧 softening' },
                        ],
                    }),
                    probe: (x, y, S) => { const i = Math.round(S.mus.length / 2); return `μ = ${S.mus[i].toFixed(3)}：fit ${S.lamFit[i].toFixed(3)}，numeric ${S.lamNum[i].toFixed(3)}（${((S.lamFit[i] / S.lamNum[i] - 1) * 100).toFixed(1)} %）`; },
                },
                {
                    id: 'lErr', label: 'Λ_fit / Λ_num − 1', sub: 'x = μ，y = r（下 0 → 上 1）；蓝 = 低估，橙 = 高估，±50 %',
                    render: (c, S) => PL.draw.field(c, S.lamErr, SHEEN_LUT_N, SHEEN_LUT_N, { map: 'diverge', range: [-0.5, 0.5], flipY: true }),
                    probe: (x, y, S) => { const j = SHEEN_LUT_N - 1 - y; return `μ = ${((x + 0.5) / SHEEN_LUT_N).toFixed(3)}, r = ${((j + 0.5) / SHEEN_LUT_N).toFixed(3)} → ${(S.lamErr[j * SHEEN_LUT_N + x] * 100).toFixed(1)} %`; },
                },
                {
                    id: 'lLobe', label: 'f · 入射面：三种 V', sub: '同一 D_Charlie，只换 visibility', size: [360, 200],
                    render: (c, S) => {
                        const a = SH.aOf(S.rough), wi = sheenWi(S);
                        BX.lobe(c, { wi, series: [{ fn: (wo) => SH.fCharlie(wi, wo, a, false), color: SHEEN_COL.charlie, label: 'V_Charlie' }, { fn: (wo) => SH.fCharlie(wi, wo, a, true), color: SHEEN_COL.soft, dash: [4, 3], label: 'V_Charlie + softening' }, { fn: (wo) => SH.fFilament(wi, wo, a), color: SHEEN_COL.filament, label: 'saturate(V_Neubelt)' }] });
                    },
                },
                { id: 'lHemi', label: 'f_Charlie(wi, ·) 半球', sub: '黄点 = wi', render: (c, S) => sheenHemi(c, S.hFc, { marks: [{ w: sheenWi(S) }] }), probe: sheenHemiProbe('hFc', ' /sr') },
                { id: 'lSphere', label: 'Charlie · V_Charlie', sub: '原始模型', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'charlie')) },
                { id: 'lSoft', label: 'Charlie · softening', sub: 'E&K Eq.4（非互易）', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'soft')) },
                {
                    id: 'lRow', label: '球体水平直径上的亮度', sub: '左右两个球 y = 0 那一行；竖线 = 主光的明暗交界 NoL = 0', size: [360, 180],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [-1, 1], xLabel: '屏幕 x（−1 左 … +1 右）', series: [
                            { data: S.rowA, x: S.rowU, color: SHEEN_COL.charlie, label: 'V_Charlie' },
                            { data: S.rowB, x: S.rowU, color: SHEEN_COL.soft, dash: [4, 3], label: 'V_Charlie + softening' },
                        ], marks: isFinite(S.rowT) ? [{ x: S.rowT, color: '#e8c25a' }] : [],
                    }),
                },
            ],
            notes: '误差图给出拟合的真实精度（本页数值核对，不是论文给的数字）：r ≈ 0.3–0.35 和 r ≥ 0.8 时与数值 Λ 相差 ±8 % 以内；r ≈ 0.4–0.7 时<strong>高估 8–20 %</strong>；r < 0.3 时迅速变成<strong>严重低估</strong>（r = 0.2 约 −25…−50 %，r = 0.1 约 −60…−85 %）。原因是 Table 1 只拟合了两端，中间靠 <code>(1 − α)²</code> 对参数插值，而 α → 0 时真实 Λ 按 1/√α 发散。G₂ = 1/(1+Λo+Λi) 把误差压小了一些，但低 r 掠射处的反照率仍会被高估（ENERGY 步白色虚线 vs 绿线）。两个球和下方的亮度剖面对比 softening：竖线（NoL = 0）右侧，原始模型一过交界亮度就迅速抬起，softening 版在交界附近被压低、过渡更平缓；θi 拖小、光源方位拖到 ±90° 左右（侧光）最明显，这正是 Imageworks 美术要的。',
            warn: 'Λ 在 μ → 0 时是 e<sup>a+e</sup> ≈ 10⁷–10⁹ 级别，<code>1 + Λo + Λi</code> 在 fp16 里会溢出成 inf → V = 0，这恰好是想要的极限；但如果先算 <code>4·NoV·NoL</code> 再相乘，0 × inf 会得到 NaN。glTF Sample Viewer 对整个 V 做了 <code>clamp(·, 0, 1)</code>。softening 让 BRDF 非互易，双向路径追踪或 ReSTIR 里交换 wi / wo 时结果会不一致。',
        },
        // ───────────────────────────── 07 ENERGY ─────────────────────────────
        {
            id: 'energy', title: '方向反照率 E(μ, r)：sheen 吃掉了多少能量', tag: 'ENERGY',
            desc: 'sheen 是叠在 base 上面的一层：从视线方向看，它反射走的那部分能量 base 就不该再反射一次。衡量这部分的量是 sheen 波瓣的方向反照率 <code>E(μo, r) = ∫ f_sheen cosθi dωi</code>（sheenColor = 1）。它只依赖 μo 和 r，可以离线烘成一张 2D LUT——glTF 规范给的就是 Charlie D + Charlie V 的表（x = NoV，y = 感知 sheenRoughness），Filament 把 Charlie D + V_Neubelt 的 E 放在 DFG LUT 的第三个通道。',
            formula: [
                'E(μ_o, r) = ∫_{H^2} f_sheen(ω_i, ω_o) cosθ_i dω_i          // sheenColor = 1，μ_o = cosθ_o',
                'E_avg(r) = 2 ∫_0^1 E(μ, r) μ dμ',
                'LUT：32 × 32，texel 中心 μ = (i + ½)/32，r = (j + ½)/32       // 本页 JS 用 (μi, φ) 中点积分',
            ],
            code: { title: 'HLSL · CS_BakeSheenE（均匀半球 MC）', src: HLSL_BAKE },
            previews: [
                {
                    id: 'eCurves', label: 'E(μo)，当前 r', sub: '竖线 = cos θi；虚线 y = 1', size: [380, 220],
                    render: (c, S) => {
                        const E = S.E, ea = S.Eavg, f = (k) => ea[k].toFixed(3);
                        PL.draw.plot(c, {
                            x: [0, 1], y: [0, Math.max(1.05, ...E.filament, ...E.charlie)], xLabel: 'μo = cosθo', series: [
                                { data: E.charlie, x: S.mus, color: SHEEN_COL.charlie, width: 2, label: `Charlie·V_Charlie  avg ${f('charlie')}` },
                                { data: E.num, x: S.mus, color: SHEEN_COL.num, dash: [4, 3], label: `Charlie·V(Λ 数值)  avg ${f('num')}` },
                                { data: E.soft, x: S.mus, color: SHEEN_COL.soft, dash: [2, 2], label: `+ softening  avg ${f('soft')}` },
                                { data: E.filament, x: S.mus, color: SHEEN_COL.filament, label: `Charlie·sat(V_Neubelt)  avg ${f('filament')}` },
                                { data: E.order, x: S.mus, color: SHEEN_COL.order, label: `Ashikhmin·V_Neubelt  avg ${f('order')}` },
                                { data: E.disney, x: S.mus, color: SHEEN_COL.disney, label: `Disney  avg ${f('disney')}` },
                            ], marks: [{ x: Math.cos(S.thetaI), color: '#e8c25a' }], hmarks: [{ y: 1, color: '#3a4148' }],
                        });
                    },
                    probe: (x, y, S) => { const m = Math.cos(S.thetaI), g = (k) => SH.curveAt(S.E[k], m).toFixed(3); return `μ = ${m.toFixed(3)}：Charlie ${g('charlie')} · Λnum ${g('num')} · soft ${g('soft')} · Filament ${g('filament')} · Order ${g('order')} · Disney ${g('disney')}`; },
                },
                {
                    id: 'eLut', label: 'E_Charlie(μ, r) LUT', sub: '32×32，x = μ，y = r（上 = 1）；0 → 1',
                    render: (c, S) => PL.draw.field(c, S.lutC, SHEEN_LUT_N, SHEEN_LUT_N, { map: 'viridis', range: [0, 1], flipY: true }),
                    probe: (x, y, S) => { const j = SHEEN_LUT_N - 1 - y; return `μ = ${((x + 0.5) / SHEEN_LUT_N).toFixed(3)}, r = ${((j + 0.5) / SHEEN_LUT_N).toFixed(3)} → E = ${S.lutC[j * SHEEN_LUT_N + x].toFixed(4)}`; },
                },
                {
                    id: 'eLutF', label: 'E_Filament(μ, r) LUT', sub: 'Charlie D · saturate(V_Neubelt)；红 = E > 1',
                    render: (c, S) => PL.draw.field(c, S.lutF, SHEEN_LUT_N, SHEEN_LUT_N, {
                        range: [0, 1.25], flipY: true,
                        map: (t, out) => { if (t > 0.8) { out[0] = 235; out[1] = 70; out[2] = 70; return out; } return PL.cmap.viridis(t / 0.8, out); },
                    }),
                    probe: (x, y, S) => { const j = SHEEN_LUT_N - 1 - y; return `μ = ${((x + 0.5) / SHEEN_LUT_N).toFixed(3)}, r = ${((j + 0.5) / SHEEN_LUT_N).toFixed(3)} → E = ${S.lutF[j * SHEEN_LUT_N + x].toFixed(4)}（表内最大 ${S.lutFMax.toFixed(3)}）`; },
                },
            ],
            notes: '所有 sheen 的 E 在正视（μ → 1）时都很小、掠射时变大——这正是"边缘发亮"在能量上的体现，也意味着 base 在轮廓处要让出更多能量。r 越小，Charlie 的 E 越集中在掠射端；r = 1 时正视也有约 0.16。Ashikhmin 的偏置让它正视时就有 0.13–0.25 的反照率（"像 diffuse"）。Disney sheen 的 E 不到 0.08。白色虚线是把 V 里的 Λ 换成上一步的<strong>数值 Λ</strong>：r = 0.5 时拟合版比它低约 10 %，r = 0.2 时掠射端反而高约 35 %。把 r 拖到 0.3 以下：Filament 组合（Charlie·saturate(V_Neubelt)）在掠射端 E > 1（r = 0.2、μ = 0.02 时 ≈ 1.23），右侧 LUT 的红色角落就是这一块；拟合 Λ 的 Charlie·V_Charlie 在 r ≲ 0.2 的掠射端也会超过 1（r = 0.1、μ = 0.02 时 ≈ 1.6，数值 Λ 版不会）——这是 Λ 拟合在低 r 处低估的直接后果。',
            warn: 'LUT 的 y 轴是<strong>感知 r</strong>（glTF 明确要求用 sheenRoughness 查表），而 D、Λ 内部用 α = r²。烘焙和查表用不同的参数化，会在整个粗糙度范围内系统性偏移。每种 D × V 组合都要有自己的表：拿 Charlie·V_Charlie 的表去补偿 Charlie·V_Neubelt 的 BRDF，补偿量会差 10–50 %。E > 1 的区域会让 <code>1 − s·E</code> 变负，见下一步。本页 LUT 每个 texel 用 24 × 24 的 (μi, φ) 中点积分，r < 0.1 的最底几行误差可达 5–10 %（只影响 r 滑块范围以外的部分）。',
        },
        // ───────────────────────────── 08 LAYER ─────────────────────────────
        {
            id: 'layer', title: 'sheen over base：albedo scaling', tag: 'LAYER',
            desc: '把 sheen 当成最上面一层：<code>f = sheenColor · f_sheen + f_base · k</code>。如果 k = 1，sheen 反射的能量是凭空多出来的——在白炉里（均匀环境光 + 白色 base）球的边缘会比背景更亮。albedo scaling（Kelemen &amp; Szirmay-Kalos 2001 的思路）用 sheen 的反照率扣掉这部分：只看视线方向（Filament、glTF 的简化版），或者取入射、出射两个方向里更保守的那个（E&amp;K 2017 / glTF 正式版）。用 <code>max3(sheenColor)</code> 是为了让最亮的通道也不超过 1。分层的一般理论放在 06 模块，这里只处理 sheen-over-base 这一种情况。',
            formula: [
                'f = sheenColor · f_sheen + f_base · k',
                'k_view = 1 − max3(sheenColor) · E(μ_o)                                   // Filament',
                'k_min  = min(1 − max3(C) · E(μ_o),  1 − max3(C) · E(μ_i))                 // glTF / E&K §5',
                '白炉（ρ_base = 1）：E_total(μ_o) = s·E(μ_o) + 2 ∫_0^1 k(μ_o, μ_i) μ_i dμ_i,   s = max3(C)',
            ],
            code: { title: 'HLSL · SheenAlbedoScaling（与 Common 中相同）', src: HLSL_SHEEN_E },
            previews: [
                {
                    id: 'yFur', label: '白炉 E_total(μo)', sub: '白色 Lambert base + Charlie sheen，sheenColor 最亮通道', size: [360, 190],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, 1], y: [0.6, Math.max(1.4, ...S.furnace.none) + 0.02], xLabel: 'μo', series: [
                            { data: S.furnace.none, x: S.mus, color: SHEEN_COL.filament, label: 'k = 1（不补偿）' },
                            { data: S.furnace.view, x: S.mus, color: SHEEN_COL.ggx, label: 'k_view' },
                            { data: S.furnace.min, x: S.mus, color: SHEEN_COL.charlie, dash: [4, 3], label: 'k_min' },
                        ], hmarks: [{ y: 1, color: '#3a4148' }],
                    }),
                },
                { id: 'yFurNone', label: '白炉球 · k = 1', sub: 'E_total − 1：橙 = 多出能量，蓝 = 丢能量，±0.6', render: (c, S) => sheenFurnace(c, S, 'none'), probe: (x, y, S) => { const v = S._furData_none[y * 96 + x]; return isFinite(v) ? `E_total − 1 = ${v.toFixed(3)}` : ''; } },
                { id: 'yFurMode', label: '白炉球 · 当前 scaling', sub: '参数栏 albedo scaling 切换', render: (c, S) => sheenFurnace(c, S, S.scaling), probe: (x, y, S) => { const v = S['_furData_' + S.scaling][y * 96 + x]; return isFinite(v) ? `E_total − 1 = ${v.toFixed(3)}` : ''; } },
                { id: 'ySphereNone', label: 'base + sheen · k = 1', sub: '不补偿', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'charlie')) },
                { id: 'ySphere', label: 'base + sheen · 当前 scaling', sub: 'E 取当前 r 的 Charlie 曲线', render: (c, S) => sheenSphere(c, S, sheenCloth(S, 'charlie', S.scaling, (m) => SH.curveAt(S.E.charlie, m))) },
            ],
            notes: 'k = 1 时白炉球的轮廓发橙（能量 > 1），r 越大、sheenColor 越亮越明显。<code>k_view</code> 对白色 Lambert 恰好让 E_total ≡ 1（因为 k 与 wi 无关）；<code>k_min</code> 更保守，而且丢能量最多的是<strong>正视</strong>方向（球心偏蓝，默认参数约 −10 %）：μo 大时 E(μo) 很小，min 总是挑到掠射入射方向的 <code>1 − s·E(μi)</code>（E&amp;K："会损失一些能量，但美术不反感"）。下面两个着色球在深色 base 上差别很小——要看出来，把 baseColor 调亮，再把 θi 与光源方位都拖到 0°（头灯）。',
            warn: '<strong>k_view 只依赖 NoV</strong>，所以它也可以乘到 base 的 IBL / 环境光上（glTF 规范的简化版就是这个用途）；k_min 依赖 NoL，只能用在逐光源的直接光里。E 超过 1 的地方（低 r 掠射，见上一步）<code>1 − s·E</code> 为负，base 会变成负辐亮度——本页对 k 做了 <code>max(k, 0)</code>，引擎里同样要钳。sheenColor 若是 sRGB 贴图，<code>max3</code> 前要先转线性。补偿只作用于 base，不要把 emission、透射、clearcoat 等更上层的项也乘进去。',
        },
        // ───────────────────────────── 09 SAMPLE ─────────────────────────────
        {
            id: 'sample', title: '重要性采样：为什么 sheen 用均匀半球', tag: 'SAMPLE',
            desc: 'Charlie D 有解析的反 CDF，按 <code>D·cosθh</code> 采 half vector 很直接。但 E&amp;K 指出这对"贴地平线"的分布反而有害：① 权重里有 <code>1/cosθh</code>，h 越水平权重越大，方差爆炸；② 由接近水平的 h 反射出来的 wi 经常<strong>落到地平线以下</strong>，样本直接浪费。sheen 波瓣本来就宽，<strong>均匀半球采样</strong>反而最稳；余弦采样把样本堆在天顶，而 sheen 的能量在掠射，效果次之。下面对当前 wo（θo = θi）的 E 做 32 组独立估计，比较相对 RMSE。',
            formula: [
                'uniform:  pdf = 1/(2π),          weight = 2π · f · cosθ_i',
                'cosine:   pdf = cosθ_i / π,      weight = π · f',
                'D-sample: sinθ_h = u_1^{α/(1+2α)},  pdf(ω_i) = D(h) cosθ_h / (4 |ω_o·h|)',
                '          weight = f cosθ_i / pdf = 4 V |ω_o·h| cosθ_i / cosθ_h   // cosθ_h → 0 时 → ∞',
            ],
            code: {
                title: 'HLSL · 三种采样', src: `
                float3 SampleUniform(float2 u, out float pdf) {          // E&K 推荐
                    float z = u.x, s = sqrt(max(0.0, 1.0 - z * z)), p = 2.0 * PI * u.y;
                    pdf = 0.5 * INV_PI;
                    return float3(s * cos(p), s * sin(p), z);
                }
                float3 SampleCosine(float2 u, out float pdf) {
                    float s = sqrt(u.x), p = 2.0 * PI * u.y, z = sqrt(max(0.0, 1.0 - u.x));
                    pdf = z * INV_PI;
                    return float3(s * cos(p), s * sin(p), z);
                }
                // 按 D·cosθh 采 h（IBL 预滤波 Charlie 环境图时常用：那里 V = N = R，不存在掠射问题）
                float3 SampleCharlieH(float2 u, float a) {
                    float sinT = pow(u.x, a / (2.0 * a + 1.0));          // CDF(θ) = sin^(2+1/a)θ 的反函数
                    float cosT = sqrt(1.0 - sinT * sinT), p = 2.0 * PI * u.y;
                    return float3(sinT * cos(p), sinT * sin(p), cosT);   // pdf(h) = D(h)·cosθh
                }`,
            },
            previews: [
                {
                    id: 'sRmse', label: '相对 RMSE vs 样本数', sub: '32 组独立随机估计；灰虚线 ∝ 1/√N', size: [380, 210],
                    render: (c, S) => {
                        const N = S.sN, u0 = S.sRmse.uniform[0], pct = (k) => (S.sLost[k] * 100).toFixed(0);
                        PL.draw.plot(c, {
                            x: [1, 256], logX: true, logY: true, xLabel: 'N（每像素样本数）', series: [
                                { data: N.map((n) => u0 / Math.sqrt(n)), x: N, color: '#55606a', dash: [4, 3] },
                                { data: S.sRmse.uniform, x: N, color: SHEEN_COL.charlie, width: 2, label: `uniform（浪费 ${pct('uniform')}%）` },
                                { data: S.sRmse.cosine, x: N, color: SHEEN_COL.ggx, label: `cosine（浪费 ${pct('cosine')}%）` },
                                { data: S.sRmse.ndf, x: N, color: SHEEN_COL.filament, label: `D·cosθh（浪费 ${pct('ndf')}%）` },
                            ],
                        });
                    },
                    probe: (x, y, S) => `E 真值 = ${S.sTruth.toFixed(4)}；N = 64 时 RMSE：uniform ${(S.sRmse.uniform[6] * 100).toFixed(1)}% · cosine ${(S.sRmse.cosine[6] * 100).toFixed(1)}% · D ${(S.sRmse.ndf[6] * 100).toFixed(1)}%`,
                },
                { id: 'sUni', label: 'uniform 样本', sub: '底图 = f·cosθi（积分被积函数）', render: (c, S) => sheenHemi(c, S.hInteg, { marks: S.sDots.uniform.map((w) => ({ w, color: '#5ad07a', size: 1 })) }), probe: sheenHemiProbe('hInteg') },
                { id: 'sCos', label: 'cosine 样本', sub: '堆在天顶', render: (c, S) => sheenHemi(c, S.hInteg, { marks: S.sDots.cosine.map((w) => ({ w, color: '#6ab0e0', size: 1 })) }), probe: sheenHemiProbe('hInteg') },
                { id: 'sNdf', label: 'D·cosθh 样本（有效部分）', sub: '落到地平线下的样本不显示', render: (c, S) => sheenHemi(c, S.hInteg, { marks: S.sDots.ndf.map((w) => ({ w, color: '#e06c75', size: 1 })) }), probe: sheenHemiProbe('hInteg') },
            ],
            notes: '图例里的"浪费"是实测比例：默认 θo = 65° 时约 2/3 的 D 样本因为 wo·h ≤ 0 或反射到地平线以下而作废，剩下的又挤在边缘，RMSE 是 uniform 的数倍。uniform 与 cosine 相差不大，uniform 略好——被积函数（底图）的能量在边缘，而 cosine 把样本堆在天顶。把 θi 拖到 85°（掠射观察）：被积函数整个贴到地平线，cosine 变成最差，D 采样与 uniform 接近但仍浪费过半样本。没有哪个角度 D 采样明显胜过 uniform。生产渲染器（Imageworks）因此用均匀半球采样；Zeltner 等 2022 的 LTC sheen 则给出了既能精确求值、又能直接采样的替代形状。',
            warn: '<strong>按 GGX 的 VNDF 去采 sheen</strong>（只因为它和 specular 共用一套采样代码）是常见错误：GGX 把样本集中在镜面方向，sheen 的掠射能量几乎采不到，表现为布料边缘的萤火虫噪点。多波瓣 MIS 时 sheen 波瓣的选择概率可按 <code>max3(sheenColor) · E(μo)</code> 分配。',
        },
        // ───────────────────────────── 10 PROD ─────────────────────────────
        {
            id: 'prod', title: '教学版 → 生产版，以及常见坑', tag: 'PROD',
            desc: '教学版：Charlie D + 拟合 Λ 的 V_Charlie，E 由当前 r 的精确积分给出，base 乘 k_min。生产版（Filament 的标准材质 sheen）：D_Charlie 照用，V 换成更便宜的 <code>saturate(V_Neubelt)</code>，E 从 32×32 LUT 双线性查表，base 只乘 k_view——k_view 不依赖光源，可以整个 pixel 只算一次，并且同样乘到 IBL 上。环境光部分：用 Charlie D 重要性采样预滤波一张"sheen 环境图"（按 r 分 mip），再乘 <code>sheenColor · E(NoV, r)</code>。下面对同一盏光逐像素求两版的差。',
            blocks: [
                { type: 'formula', text: '教学版：  f = C · D_Charlie · V_Charlie            + base · min(1 − sE(μ_o), 1 − sE(μ_i))   // E 精确积分\n生产版：  f = C · D_Charlie · saturate(V_Neubelt) + base · (1 − s · LUT(NoV, r))          // Filament\nIBL：     L_sheen = C · E(NoV, r) · Prefiltered_Charlie(N, r),   base_IBL ×= 1 − s · E(NoV, r)' },
                { type: 'code', title: 'HLSL · PS_DeferredLighting（sheen 部分）', src: HLSL_LIGHT },
                {
                    type: 'previews', items: [
                        { id: 'pTeach', label: '教学版', sub: 'V_Charlie · k_min · 精确 E', render: (c, S) => sheenSphere(c, S, S.fTeach) },
                        { id: 'pProd', label: '生产版', sub: 'saturate(V_Neubelt) · k_view · LUT', render: (c, S) => sheenSphere(c, S, S.fProd) },
                        {
                            id: 'pDiff', label: '亮度差：生产 − 教学', sub: '色标 ±8 % × 教学版峰值亮度；橙 = 生产版更亮',
                            render: (c, S) => PL.draw.field(c, S.pDiff, 112, 112, { map: 'diverge', range: [-0.08 * S.pPeak, 0.08 * S.pPeak] }),
                            probe: (x, y, S) => { const i = y * 112 + x; return isFinite(S.pDiff[i]) ? `教学 ${S.pLt[i].toFixed(4)} · 生产 ${S.pLp[i].toFixed(4)} · Δ ${S.pDiff[i].toFixed(4)}` : ''; },
                        },
                    ],
                },
                { type: 'note', html: '生产版几乎处处略亮，差异集中在<strong>受光的轮廓</strong>：V_Neubelt 不带 G，掠射时比 V_Charlie 亮；k_view 不看 NoL，比 k_min 少扣一点 base。默认参数下最大差约为教学版峰值的 +7 %（受光一侧的上缘），球心 < 1 %。把 θi 与光源方位都拖到 0°（头灯）、baseColor 调亮：整圈轮廓 +6–8 %，球心仍 < 1 %。对实时渲染来说这个差距通常可以接受，换来的是每像素只查一次 LUT、不算 Λ。' },
                {
                    type: 'html', html: '<p><strong>演进</strong>：① <em>glTF KHR_materials_sheen</em>：Charlie D + Charlie Λ（或 Ashikhmin V），sheenColor / sheenRoughness 两个参数，albedo scaling 用 min 或 view-only 版；② <em>Filament</em>：标准材质的 sheen 用 Charlie D + V_Neubelt，E 存在 DFG LUT 第三通道；独立的 cloth 着色模型另有 subsurfaceColor 包裹漫反射，sheenColor 默认 √baseColor；③ <em>Zeltner, Burley &amp; Chiang 2022</em>：用 LTC 拟合体积纤维层的多次散射 sheen，求值和采样都是解析的，OpenPBR 的 fuzz 层与 Blender 4.x 的 sheen 采用了这条路线；④ UE 的 Cloth 着色模型用倒置 GGX <code>D_InvGGX</code> + <code>Vis_Cloth</code>（即 V_Neubelt）。</p>',
                },
                {
                    type: 'warn', html: '<strong>常见坑</strong>：① α 用错：把感知 r 当 α，或把 α 再平方（看 D·CHARLIE 步）；② LUT 用 α 当 y 轴查表，或者用别的 D × V 组合烘的表（看 ENERGY 步）；③ 忘了 albedo scaling → 布料轮廓在白炉里发光；把 k 乘到 emission / clearcoat 上 → 过暗；④ V_Charlie 在 NoV → 0 时 0 × inf = NaN，要先 clamp NoV（+1e-4）再算；⑤ fp16 下 <code>pow(sin²θh, 0.5/α)</code> 下溢：sin²θh 钳到 2⁻⁷；⑥ 用 GGX 的 VNDF 采样 sheen → 边缘萤火虫；⑦ sheenColor 贴图忘了 sRGB → 线性；⑧ terminator softening 让 BRDF 非互易，双向算法里要么关掉，要么只在光源侧一致地使用。',
                },
            ],
        },
    ],

    graph: {
        title: 'Sheen 着色 · Pass 视图',
        intro: '<strong>RDG 风格</strong>：真正的 pass 只有两个——一次性的 <code>CS_BakeSheenE</code>（烘 E LUT）和每帧的 <code>PS_DeferredLighting</code>。<code>D_Charlie</code>、<code>V_Charlie</code>、<code>SheenAlbedoScaling</code> 是 <strong>INLINE</strong> 函数节点：它们不是 pass，而是被内联进两个 pass 的 shader 代码；连线表示"这个 pass 的 shader 调用了它"。base BRDF 来自 01 / 02 模块。',
        types: {
            cb: { color: '#e8c25a', label: 'cb · cbuffer / G-buffer 参数' },
            fn: { color: '#6ab0e0', label: 'fn · 内联 HLSL 函数' },
            lut: { color: '#a45fa5', label: 'lut · E(μ, r) 纹理 R16F' },
            out: { color: '#5ae87a', label: 'out · 输出颜色' },
        },
        common: {
            title: 'Common.hlsl · 系列共享 + Sheen 函数', open: false,
            blocks: [
                { label: '━ 系列共享 ━', code: BX.HLSL_COMMON },
                { label: '━ Sheen · D（Charlie / Ashikhmin）━', code: HLSL_SHEEN_D },
                { label: '━ Sheen · Λ / V（Charlie 拟合 / Neubelt）━', code: HLSL_SHEEN_V },
                { label: '━ Sheen · 能量（LUT 查表 / albedo scaling）━', code: HLSL_SHEEN_E },
                {
                    label: '━ 本模块辅助：采样点 / 光源 cbuffer ━', code: `
                    float RadicalInverse2(uint b) {
                        b = (b << 16u) | (b >> 16u);
                        b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
                        b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
                        b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
                        b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
                        return float(b) * 2.3283064e-10;
                    }
                    float2 Hammersley(uint i, uint N) { return float2((i + 0.5) / N, RadicalInverse2(i)); }   // 与 BX.hammersley 相同
                    cbuffer LightCB : register(b1) { float3 LightDir; float pad0; float3 LightColor; float pad1; };`,
                },
            ],
        },
        nodes: [
            {
                id: 'mat', title: 'GBuffer · Sheen', tag: 'CB', category: 'user', desc: 'baseColor · sheenColor · sheenRoughness',
                outputs: [{ name: 'SheenParams', format: 'GBuffer RGBA8 ×2', type: 'cb' }],
                detail: { desc: '材质参数：sheenColor（线性 RGB）、sheenRoughness（感知 r），以及 base 的 baseColor / roughness / metallic。glTF 里来自 KHR_materials_sheen。', preview: null },
            },
            {
                id: 'light', title: 'LightCB', tag: 'CB', category: 'system', desc: 'LightDir · LightColor',
                outputs: [{ name: 'LightCB', format: 'cbuffer b1', type: 'cb' }],
                detail: { desc: '引擎注入的方向光。页面上由 θi（俯仰）与"光源方位"（偏航）驱动球体渲染。', preview: null },
            },
            {
                id: 'dch', title: 'D_Charlie', tag: 'INLINE', category: 'function', step: 'charlie', desc: '(2 + 1/α)·sin^(1/α)θh / 2π',
                dispatch: { kind: 'INLINE', name: 'D_Charlie(NoH, a)', note: '内联进 CS_BakeSheenE 与 PS_DeferredLighting' },
                outputs: [{ name: 'D_Charlie', format: 'float(NoH, a)', type: 'fn' }],
                detail: { preview: 'cHemi' },
            },
            {
                id: 'vch', title: 'V_Charlie · Λ fit', tag: 'INLINE', category: 'function', step: 'lambda', desc: '1 / ((1+Λv+Λl)·4·NoV·NoL)',
                dispatch: { kind: 'INLINE', name: 'V_Charlie(NoV, NoL, a)', note: 'Λ = exp(L(x))，5 参数 × 2 端点' },
                outputs: [{ name: 'V_Charlie', format: 'float(NoV, NoL, a)', type: 'fn' }],
                detail: { preview: 'lLam', code: 0 },
            },
            {
                id: 'bake', title: 'CS_BakeSheenE', tag: 'CS', category: 'bake', lifetime: 'resident', step: 'energy', desc: 'E(μ, r) = ∫ f_sheen cosθi dωi',
                dispatch: { name: 'CS_BakeSheenE', threads: [8, 8, 1], groups: [4, 4, 1], idRange: 'ID.xy ∈ [0,32)² → (NoV, r) texel 中心', note: '每 thread 1024 个均匀半球样本；离线或加载时一次' },
                inputs: [{ name: 'D_Charlie', format: 'float(NoH, a)', type: 'fn' }, { name: 'V_Charlie', format: 'float(NoV, NoL, a)', type: 'fn' }],
                outputs: [{ name: 'SheenE_LUT', format: 'R16F 32×32', type: 'lut' }],
                detail: { conversion: 'NoV = (id.x + 0.5) / 32\nr   = (id.y + 0.5) / 32,   a = r^2     // y 轴存感知 r，内部用 α', preview: 'eLut' },
            },
            {
                id: 'scale', title: 'SheenAlbedoScaling', tag: 'INLINE', category: 'function', step: 'layer', desc: 'min(1 − s·E(μo), 1 − s·E(μi))',
                dispatch: { kind: 'INLINE', name: 'SheenAlbedoScaling(NoV, NoL, C, r)', note: '两次 LUT 采样；view-only 版只采一次' },
                inputs: [{ name: 'SheenE_LUT', format: 'R16F 32×32', type: 'lut' }],
                outputs: [{ name: 'SheenScale', format: 'float(NoV, NoL, C, r)', type: 'fn' }],
                detail: { preview: 'yFur' },
            },
            {
                id: 'base', title: 'BaseBRDF', tag: 'INLINE', category: 'function', desc: 'Lambert/Burley + GGX（模块 01 / 02）',
                dispatch: { kind: 'INLINE', name: 'BaseBRDF(s, L, H)', note: '本页球体里 base = Lambert(baseColor)' },
                outputs: [{ name: 'BaseBRDF', format: 'float3(s, L, H)', type: 'fn' }],
                detail: { desc: 'sheen 之下的 base 层：漫反射（01）+ 微表面镜面（02）。sheen 模块只关心它会被乘上 SheenScale。', code: { title: 'BaseBRDF（示意签名）', src: 'float3 BaseBRDF(Surface s, float3 L, float3 H);   // 见 bxdf_01 / bxdf_02' }, preview: 'dBase' },
            },
            {
                id: 'lit', title: 'PS_DeferredLighting', tag: 'PS', category: 'output', lifetime: 'frame', step: 'prod', desc: 'C·D·V + base·k',
                dispatch: { kind: 'PS', name: 'PS_DeferredLighting', target: 'SceneColor RGBA16F（full-screen）', idRange: 'SV_Position.xy ∈ viewport', note: '每像素每光源一次；k_view 版可每像素只算一次' },
                inputs: [
                    { name: 'SheenParams', format: 'GBuffer RGBA8 ×2', type: 'cb' }, { name: 'LightCB', format: 'cbuffer b1', type: 'cb' },
                    { name: 'D_Charlie', format: 'float(NoH, a)', type: 'fn' }, { name: 'V_Charlie', format: 'float(NoV, NoL, a)', type: 'fn' },
                    { name: 'SheenScale', format: 'float(NoV, NoL, C, r)', type: 'fn' }, { name: 'BaseBRDF', format: 'float3(s, L, H)', type: 'fn' },
                ],
                outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'out' }],
                detail: { code: { title: 'PS_DeferredLighting', src: HLSL_LIGHT }, preview: 'pTeach' },
            },
        ],
    },

    references: [
        { title: 'Physically-Based Shading at Disney', author: 'Brent Burley', venue: 'SIGGRAPH Course: Practical Physically Based Shading in Film and Game Production', year: 2012, note: 'Disney sheen：sheen · Csheen · (1 − cosθd)⁵' },
        { title: 'A Microfacet-based BRDF Generator', author: 'Michael Ashikhmin, Simon Premože, Peter Shirley', venue: 'SIGGRAPH', year: 2000, note: '由任意 NDF 构造 BRDF；早期 velvet 分布' },
        { title: 'Distribution-based BRDFs', author: 'Michael Ashikhmin, Simon Premože', venue: 'Unpublished Technical Report, University of Utah', year: 2007, note: '倒置高斯 velvet 分布与平滑分母 4(n·l + n·v − (n·l)(n·v))' },
        { title: 'Crafting a Next-Gen Material Pipeline for The Order: 1886', author: 'David Neubelt, Matt Pettineo', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2013, url: 'https://blog.selfshadow.com/publications/s2013-shading-course/rad/s2013_pbs_rad_notes.pdf', note: 'Eq. 21–23：D_invgauss 归一化 1/(π(1+Aα²))，A = 4，去掉 G' },
        { title: 'Production Friendly Microfacet Sheen BRDF', author: 'Alejandro Conty Estevez, Christopher Kulla', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2017, url: 'https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_sheen.pdf', note: 'Charlie D、Λ 拟合 Table 1、terminator softening、均匀采样、min(αi, αo) 分层' },
        { title: 'Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs', author: 'Eric Heitz', venue: 'JCGT 3(2)', year: 2014, note: '投影面积归一化 → 由 D 数值求 Smith Λ' },
        { title: 'A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling', author: 'Csaba Kelemen, László Szirmay-Kalos', venue: 'Eurographics Short Presentations', year: 2001, note: 'albedo scaling 的来源' },
        { title: 'KHR_materials_sheen', author: 'Khronos Group', venue: 'glTF 2.0 extension specification', year: 2021, url: 'https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_sheen', note: 'α = sheenRoughness²、Λ 插值用 (1 − α)²、albedo scaling 的 min / view-only 两版、LUT 用感知 r 查' },
        { title: 'Physically Based Rendering in Filament', author: 'Romain Guy, Mathias Agopian', venue: 'Filament documentation', url: 'https://google.github.io/filament/Filament.html', note: 'cloth 模型、D_Charlie + V_Neubelt、sheen 的 DFG 第三通道' },
        { title: 'Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines', author: 'Tizian Zeltner, Brent Burley, Matt Jen-Yuan Chiang', venue: 'SIGGRAPH Talks', year: 2022, note: 'LTC 体积 sheen，可解析采样' },
        { title: 'The Secret of Velvety Skin', author: 'Jan Koenderink, Sylvia Pont', venue: 'Machine Vision and Applications 14(4)', year: 2003, note: '丝绒 / 绒毛表面的掠射散射与 terminator' },
    ],
});

// ── 3D view（局部系 n = +Z；mesh3d 是 y-up，绕 x 轴转 −90°：(x, y, z) → (x, z, −y)，保持右手系）──
function sheenZUp(p) { return [p[0], p[2], -p[1]]; }
function drawSheenLobe3d(c, S) {
    const { nu, nv } = SHEEN_L3D, NU = nu + 1, M = S.l3dMax, R = 1.25, V3 = PL.v3;
    const dirAt = (k) => sheenL3dDir((k % NU) % nu, Math.floor(k / NU));
    const val = (buf, m, u, v) => buf[Math.round(v * nv) * NU + Math.round(u * nu)] / m;
    const lobe = (buf, m, opt) => PL.mesh.param((u, v) => {
        const r = R * val(buf, m, u, v), d = sheenL3dDir(Math.round(u * nu) % nu, Math.round(v * nv));
        return sheenZUp([d[0] * r, d[1] * r, d[2] * r]);
    }, [nu, nv], Object.assign({ closedU: true, twoSided: true }, opt));
    const wi = sheenWi(S), mir = [-wi[0], -wi[1], wi[2]];
    const ring = Array.from({ length: 65 }, (_, k) => { const p = 2 * Math.PI * k / 64; return [1.3 * Math.cos(p), 0, 1.3 * Math.sin(p)]; });
    const lo = [0.10, 0.36, 0.22], hi = [0.62, 0.95, 0.55];   // Charlie：暗绿（小）→ 亮绿（峰）
    PL.draw.mesh3d(c, {
        meshes: [
            lobe(S.l3dC, M.c, { gloss: 20, wire: true, wireColor: 'rgba(10,30,18,0.28)', color: (u, v) => { const t = Math.sqrt(PL.clamp(val(S.l3dC, M.c, u, v), 0, 1)); return [0, 1, 2].map((q) => lo[q] + (hi[q] - lo[q]) * t); } }),
            lobe(S.l3dG, M.g, { rgb: [0.42, 0.69, 0.88], gloss: 40 }),
        ],
        lines: [
            { pts: ring, color: 'rgba(150,160,170,0.5)', width: 1 },
            { pts: [[0, 0, 0], [0, 1.6, 0]], color: '#c8d0d8', width: 1.6, arrow: true, label: 'n' },
            { pts: [[0, 0, 0], sheenZUp(V3.mul(wi, 1.6))], color: '#ffd866', width: 2.4, arrow: true, label: 'ωi' },
            { pts: [[0, 0, 0], sheenZUp(V3.mul(mir, 1.6))], color: '#6ab0e0', width: 1.4, dash: [5, 4], label: '镜面方向' },
        ],
        labels: [
            { p: sheenZUp(V3.mul(dirAt(M.atC), R * 1.08)), text: 'Charlie 峰', color: SHEEN_COL.charlie },
            { p: sheenZUp(V3.mul(dirAt(M.atG), R * 1.08)), text: 'GGX 峰', color: SHEEN_COL.ggx },
        ],
        orbit: { yaw: 0.3, pitch: 0.36, dist: 5.7 },
    });
}
