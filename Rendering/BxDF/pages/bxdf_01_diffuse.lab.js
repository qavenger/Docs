// BxDF 01 · 漫反射：Lambert → Oren-Nayar → Burley
// Build:    node <skill>/scripts/build_doc.mjs bxdf_common.js bxdf_01_diffuse.lab.js -o bxdf_01_diffuse.html
// Validate: node <skill>/scripts/validate_doc.mjs bxdf_01_diffuse.html
// 约定见 bxdf_common.js 头部：局部坐标 n = +Z，wi 指向光源、wo 指向观察者，线性 RGB，α = r²（本模块的漫反射都不用 α）。
// 本模块所有模型都写成 k(wi, wo) = π·f / ρ（"相对 Lambert 的倍数"），f = ρ·k/π；EON 的多次散射项对 ρ 非线性，逐通道算。

const D1 = (() => {
    const PI = Math.PI, IPI = 1 / Math.PI, p5 = BX.pow5, clamp = BX.clamp;
    const sTerm = (wi, wo) => wi[0] * wo[0] + wi[1] * wo[1];          // s = wi·wo − cosθi·cosθo = sinθi·sinθo·cos(φi−φo)

    // ── Oren-Nayar 1994 qualitative model (QON)，σ = facet 坡度角标准差（弧度） ──
    const qonAB = (sg) => { const s2 = sg * sg; return [1 - 0.5 * s2 / (s2 + 0.33), 0.45 * s2 / (s2 + 0.09)]; };
    const kQON = (wi, wo, sg) => {
        const ni = wi[2], no = wo[2];
        if (ni <= 0 || no <= 0) return 0;
        const [A, B] = qonAB(sg), s = sTerm(wi, wo);
        return A + B * (s > 0 ? s / Math.max(ni, no) : 0);              // = A + B·max(0,cosΔφ)·sinα·tanβ
    };
    // ── Fujii 2013 (FON) + EON (Portsmouth, Kutz, Hill 2025)，r ∈ [0,1] ──
    const C1 = 0.5 - 2 / (3 * PI), C2 = 2 / 3 - 28 / (15 * PI);
    const fonA = (r) => 1 / (1 + C1 * r);
    const kFON = (wi, wo, r) => {
        const ni = wi[2], no = wo[2];
        if (ni <= 0 || no <= 0) return 0;
        const s = sTerm(wi, wo), st = s > 0 ? s / Math.max(ni, no) : s;   // s ≤ 0 时 t = 1（不截断）
        return fonA(r) * (1 + r * st);
    };
    const EFON = (mu, r) => {                                          // FON 方向反照率（ρ = 1，Fujii 解析式）
        mu = clamp(mu, 1e-4, 1);
        const AF = fonA(r), BF = r * AF, Si = Math.sqrt(1 - mu * mu);
        const G = Si * (Math.acos(mu) - Si * mu) + (2 / 3) * ((Si / mu) * (1 - Si * Si * Si) - Si);
        return AF + BF * IPI * G;
    };
    const avgEFON = (r) => fonA(r) * (1 + C2 * r);
    const rhoMS = (rho, avg) => rho * rho * avg / (1 - rho * (1 - avg));
    const piEON = (wi, wo, r, rho) => {                                // π·f_EON，ρ 为标量（单通道）
        const ni = wi[2], no = wo[2];
        if (ni <= 0 || no <= 0) return 0;
        const avg = avgEFON(r), e = 1e-7;
        return rho * kFON(wi, wo, r) + rhoMS(rho, avg) * Math.max(e, 1 - EFON(no, r)) * Math.max(e, 1 - EFON(ni, r)) / Math.max(e, 1 - avg);
    };
    const EEON = (mu, r, rho) => { const EF = EFON(mu, r); return rho * EF + rhoMS(rho, avgEFON(r)) * (1 - EF); };
    // ── Burley 2012 / Frostbite 2014，r = 感知粗糙度 ──
    const schlick1 = (f90, u) => 1 + (f90 - 1) * p5(1 - u);            // f0 = 1 的 Schlick
    const LoHof = (wi, wo) => BX.dot(wi, BX.half(wi, wo));              // cosθd
    const kBurley = (wi, wo, r) => {
        const ni = wi[2], no = wo[2];
        if (ni <= 0 || no <= 0) return 0;
        const c = LoHof(wi, wo), F90 = 0.5 + 2 * r * c * c;
        return schlick1(F90, ni) * schlick1(F90, no);
    };
    const kFrost = (wi, wo, r) => {
        const ni = wi[2], no = wo[2];
        if (ni <= 0 || no <= 0) return 0;
        const c = LoHof(wi, wo), F90 = 0.5 * r + 2 * r * c * c;          // energyBias = lerp(0, 0.5, r)
        return schlick1(F90, ni) * schlick1(F90, no) * (1 + (1 / 1.51 - 1) * r);   // energyFactor = lerp(1, 1/1.51, r)
    };

    const M = {
        lambert: { name: 'Lambert', color: '#9aa4ae', k: () => 1 },
        qon: { name: 'QON', color: '#e0a060', k: (wi, wo, S) => kQON(wi, wo, S.sigma) },
        fon: { name: 'FON', color: '#c678dd', k: (wi, wo, S) => kFON(wi, wo, S.rF) },
        eon: { name: 'EON', color: '#5ad07a', nonlinear: true, k: (wi, wo, S, rho) => { const p = Math.max(1e-4, rho); return piEON(wi, wo, S.rF, p) / p; } },
        burley: { name: 'Burley', color: '#6ab0e0', k: (wi, wo, S) => kBurley(wi, wo, S.rough) },
        frost: { name: 'Frostbite', color: '#e06c75', k: (wi, wo, S) => kFrost(wi, wo, S.rough) },
    };
    /** f in 1/sr, linear RGB, current parameters */
    const fRGB = (m, wi, wo, S) => {
        const a = S.albedo;
        if (m.nonlinear) return [a[0] * m.k(wi, wo, S, a[0]) * IPI, a[1] * m.k(wi, wo, S, a[1]) * IPI, a[2] * m.k(wi, wo, S, a[2]) * IPI];
        const k = m.k(wi, wo, S, 1) * IPI;
        return [a[0] * k, a[1] * k, a[2] * k];
    };

    // ── V-cavity（flatland：槽垂直于入射平面），角度为入射平面内带符号的 θ（>0 朝 +x） ──
    /** facet 从槽口算起可见（或受光）的比例。side = +1：左 facet，法线 (sin a, cos a)；−1：右 facet */
    const gFrac = (a, side, t) => {
        if (Math.cos(t - side * a) <= 0) return 0;                       // 背对
        const u = side * t;
        return u <= 0 ? 1 : Math.min(1, 2 / (1 + Math.tan(a) * Math.tan(u)));
    };
    const vGroove = (a, ti, to) => {                                    // π·f/ρ，单次散射
        const ct = Math.cos(ti) * Math.cos(to);
        if (ct <= 1e-6) return 0;
        let sum = 0;
        for (const side of [1, -1]) {
            const ci = Math.cos(ti - side * a), co = Math.cos(to - side * a);
            if (ci > 0 && co > 0) sum += ci * co * Math.min(gFrac(a, side, ti), gFrac(a, side, to));
        }
        return sum / (2 * Math.cos(a) * ct);
    };
    /** 截断半正态坡度分布的积分节点 */
    const slopeNodes = (sg, J = 48) => {
        if (sg < 1e-3) return { a: [0], w: [1] };
        const a = [], w = [];
        let tot = 0;
        for (let j = 0; j < J; j++) { const x = (j + 0.5) / J * 1.53; a.push(x); const v = Math.exp(-x * x / (2 * sg * sg)); w.push(v); tot += v; }
        return { a, w: w.map((v) => v / tot) };
    };

    // ── 方向反照率表：θo 均匀 NT 份 ──
    const NT = 48, NQ = 20;
    const thetaOf = (j) => (j + 0.5) / NT * PI / 2;
    const thetaDeg = Array.from({ length: NT }, (_, j) => (j + 0.5) / NT * 90);
    const table = (f, out) => { for (let j = 0; j < NT; j++) out[j] = BX.albedoQuad(f, BX.dir(thetaOf(j)), NQ); return out; };
    const lookupE = (tab, mu) => {
        const t = Math.acos(clamp(mu, 0, 1)) / (PI / 2) * NT - 0.5, i = Math.floor(t), u = t - i;
        const a = tab[clamp(i, 0, NT - 1)], b = tab[clamp(i + 1, 0, NT - 1)];
        return a + (b - a) * u;
    };

    // ── 画图辅助 ──
    const lightDir = (th, S) => BX.keyLight(th, S ? S.lightAz : 0);   // 视空间：俯仰 = th（偏航 0 时 = 与视线的夹角），偏航 = 光源方位（系列统一约定）
    const sphere = (c, m, S, o = {}) => BX.renderSphere(c, (wi, wo) => fRGB(m, wi, wo, S), {
        res: 112, lights: o.lights || [{ dir: lightDir(o.theta === undefined ? S.thetaI : o.theta, S), color: [PI, PI, PI] }], exposure: o.exposure || 1,
    });
    const outline = (c, res) => {
        const ctx = c.getContext('2d');
        ctx.save(); ctx.strokeStyle = 'rgba(120,140,160,0.75)'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(res / 2, res / 2, res / 2 - 0.5, 0, 2 * PI); ctx.stroke(); ctx.restore();
    };
    /** 白炉球：环境 radiance ≡ 1。mode 'gray' = 真实外观（背景 = 1）；'dev' = E − 1 的偏差（蓝 = 丢能量，橙 = 造能量） */
    const furnace = (c, tab, o = {}) => {
        const res = 112, tint = o.tint;
        if (o.mode === 'dev') {
            const img = PL.buf(c, '_dev', res * res);
            for (let y = 0; y < res; y++) for (let x = 0; x < res; x++) {
                const u = 2 * (x + 0.5) / res - 1, v = 2 * (y + 0.5) / res - 1, r2 = u * u + v * v;
                img[y * res + x] = r2 > 1 ? 0 : lookupE(tab, Math.sqrt(1 - r2)) - 1;
            }
            PL.draw.field(c, img, res, res, { range: [-0.4, 0.4], map: 'diverge' });
        } else {
            PL.draw.pixels(c, res, res, (x, y, out) => {
                const u = 2 * (x + 0.5) / res - 1, v = 2 * (y + 0.5) / res - 1, r2 = u * u + v * v;
                const E = r2 > 1 ? 1 : lookupE(tab, Math.sqrt(1 - r2));
                for (let k = 0; k < 3; k++) out[k] = 0.5 * (r2 > 1 || !tint ? E : E * tint[k]);
                return out;
            }, { encode: 'srgb' });
        }
        outline(c, res);
    };
    const RES_H = 80;
    const markLight = (c, wi, res) => {
        const ctx = c.getContext('2d'), [px, py] = BX.hemiPixel(wi, res);
        ctx.save(); ctx.strokeStyle = '#e8c25a'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(px + 0.5, py + 0.5, 3.2, 0, 2 * PI); ctx.stroke(); ctx.restore();
    };
    const ratioMap = (c, img, S) => { PL.draw.field(c, img, RES_H, RES_H, { range: [-1, 1], map: 'diverge' }); markLight(c, S.wi, RES_H); };
    const hemiProbe = (img) => (x, y) => {
        const w = BX.hemiDir(x, y, RES_H);
        if (!w) return '';
        const th = Math.acos(w[2]) * 180 / PI, ph = Math.atan2(w[1], w[0]) * 180 / PI, v = img[y * RES_H + x];
        return `θo=${th.toFixed(0)}° φo=${ph.toFixed(0)}°  k=${Math.pow(2, v).toFixed(3)}`;
    };
    const plotE = (c, list, o = {}) => PL.draw.plot(c, {
        x: [0, 90], y: o.y || [0.4, 1.6], xLabel: 'θo (°)', yLabel: 'E(ωo)',
        series: list.map((s) => ({ data: Array.from(s.tab), x: thetaDeg, color: s.color, label: s.label, dash: s.dash, width: s.width })),
        hmarks: [{ y: 1, label: '白炉 = 1', color: '#4a5560' }], marks: o.marks,
    });
    const deg = (r) => `${(r * 180 / PI).toFixed(0)}°`;
    const minmax = (t) => { let a = Infinity, b = -Infinity; for (const v of t) { a = Math.min(a, v); b = Math.max(b, v); } return [a, b]; };

    return { PI, IPI, qonAB, kQON, kFON, EFON, avgEFON, EEON, piEON, kBurley, kFrost, schlick1, M, fRGB, gFrac, vGroove, slopeNodes,
        NT, NQ, thetaOf, thetaDeg, table, lookupE, lightDir, sphere, furnace, outline, RES_H, ratioMap, markLight, hemiProbe, plotE, deg, minmax };
})();

// ── V 槽剖面示意（数据 = 真实的可见 / 受光比例） ──
function drawGrooves(c, S, to, caption) {
    const { ctx, w, h } = PL.draw.ctx(c);
    const G = S.grooves, M = G.length, ti = S.thetaI, top = 40, avail = h - top - 8;
    let hw = (w - 96) / (2 * M);
    const maxT = Math.max(1e-6, ...G.map((a) => Math.tan(a)));
    if (hw * maxT > avail) hw = avail / maxT;
    const x0 = 8;
    const seg = (p, q, col, lw) => { ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); };
    ctx.setLineDash([3, 3]); seg([x0 - 8, top], [x0 + 2 * M * hw + 8, top], '#2a3138', 1); ctx.setLineDash([]);
    let vis = 0, litVis = 0, ksum = 0;
    G.forEach((a, k) => {
        const xa = x0 + 2 * k * hw, B = [xa + hw, top + hw * Math.tan(a)];
        for (const [rim, side] of [[[xa, top], 1], [[xa + 2 * hw, top], -1]]) {
            const fi = D1.gFrac(a, side, ti), fo = D1.gFrac(a, side, to);
            const ci = Math.max(0, Math.cos(ti - side * a)), co = Math.max(0, Math.cos(to - side * a));
            const at = (s) => [rim[0] + (B[0] - rim[0]) * s, rim[1] + (B[1] - rim[1]) * s];
            seg(rim, B, '#3a4148', 1);
            const m = Math.min(fi, fo);
            if (m > 0) seg(at(0), at(m), `rgb(${150 + 105 * ci | 0},${90 + 120 * ci | 0},${40 + 50 * ci | 0})`, 3.2);
            if (fo > fi) seg(at(fi), at(fo), '#3a78c8', 3.2);
            const len = 1 / Math.cos(a);
            vis += len * fo * co; litVis += len * m * co;
        }
        ksum += D1.vGroove(a, ti, to);
    });
    // 方向示意（右侧小坐标）：ωi 黄、ωo 青；回射时两者重合
    const cx = w - 42, cy = h - 16, L = Math.min(34, h - 34);
    seg([cx - 36, cy], [cx + 36, cy], '#3a4148', 1);
    const arrow = (t, col, lab, dy) => {
        const ex = cx + Math.sin(t) * L, ey = cy - Math.cos(t) * L;
        seg([cx, cy], [ex, ey], col, 1.6);
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(ex, ey, 2.6, 0, 2 * Math.PI); ctx.fill();
        ctx.font = '10px Consolas, monospace'; ctx.textAlign = Math.sin(t) >= 0 ? 'left' : 'right'; ctx.fillText(lab, ex + (Math.sin(t) >= 0 ? 5 : -5), ey + 3 + dy);
    };
    arrow(ti, '#e8c25a', 'ωi', 0); arrow(to, '#56d0e0', 'ωo', Math.abs(to - ti) < 1e-6 ? 11 : 0);
    ctx.font = '10px Consolas, monospace'; ctx.textAlign = 'left'; ctx.fillStyle = '#9aa4ae';
    ctx.fillText(caption, 6, 12);
    ctx.fillStyle = '#e8c25a';
    ctx.fillText(`看到的面积中受光 ${(100 * litVis / Math.max(1e-9, vis)).toFixed(0)}%   k = ${(ksum / M).toFixed(2)}`, 6, 25);
}

// ── HLSL ──
const HLSL_HELPERS = `
    // 本模块辅助：f0 = 1 的 Schlick 权重；Hammersley 点集
    float  F_Schlick90(float f90, float u) { return 1.0 + (f90 - 1.0) * Pow5(1.0 - u); }
    float2 Hammersley(uint i, uint N) { return float2((i + 0.5) / N, reversebits(i) * 2.3283064365386963e-10); }`;

const HLSL_LAMBERT = `
    // Lambert：f = ρ/π，与 ωi、ωo 都无关
    float3 Fd_Lambert(float3 albedo) {
        return albedo * INV_PI;
    }
    // 方向光（E = 法向入射照度）：Lo = f · E · NoL
    float3 ShadeLambert(float3 albedo, float3 N, float3 L, float3 E) {
        return Fd_Lambert(albedo) * E * saturate(dot(N, L));
    }`;

const HLSL_FURNACE = `
    // 白炉检查（Debug CS）：环境 radiance ≡ 1、albedo = 1，输出 E(θo)；守恒 ⇔ 处处 = 1
    cbuffer FurnaceCB { uint FurnaceModel; float FurnaceR; };   // 被测模型与粗糙度（同 EvalDiffuse 的 model / r）
    RWTexture1D<float> FurnaceOut;
    [numthreads(64, 1, 1)]
    void CS_WhiteFurnace(uint3 id : SV_DispatchThreadID) {
        float  thetaO = (id.x + 0.5) / 64.0 * (0.5 * PI);
        float3 V = float3(sin(thetaO), 0.0, cos(thetaO));
        float  E = 0.0;
        for (uint i = 0; i < 1024; ++i) {
            float  pdf;
            float3 L = SampleCosineHemisphere(Hammersley(i, 1024), pdf);
            // 局部坐标 n = +Z：NoL = L.z，NoV = V.z
            E += EvalDiffuse(FurnaceModel, float3(1, 1, 1), FurnaceR, L.z, V.z, dot(L, V)).x * L.z / pdf;   // f·cosθi / pdf
        }
        FurnaceOut[id.x] = E / 1024.0;
    }`;

const HLSL_SAMPLE = `
    // Malley：单位圆盘上均匀采样，再竖直投影到半球 ⇒ pdf = cosθ/π（局部坐标 n = +Z）
    float3 SampleCosineHemisphere(float2 u, out float pdf) {
        float r   = sqrt(u.x);
        float phi = 2.0 * PI * u.y;
        float3 w  = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
        pdf = w.z * INV_PI;
        return w;
    }
    // MC 权重 = f·cosθ / pdf = π·f；Lambert 时恒等于 albedo（零方差）`;

const HLSL_VGROOVE = `
    // V 槽（flatland、单次散射）参考实现：a = facet 坡度，ti / to = 入射平面内带符号的角
    float GrooveFrac(float a, float side, float t) {        // 从槽口往下：可见（或受光）的比例
        if (cos(t - side * a) <= 0.0) return 0.0;            // facet 背对该方向
        float u = side * t;                                  // u > 0：方向越过对面槽壁
        return u <= 0.0 ? 1.0 : min(1.0, 2.0 / (1.0 + tan(a) * tan(u)));
    }
    float VGrooveK(float a, float ti, float to) {           // 返回 π·f/ρ
        float sum = 0.0;
        [unroll] for (int k = 0; k < 2; ++k) {
            float side = k == 0 ? 1.0 : -1.0;                // 左 facet 法线 (sin a, cos a)，右 facet (−sin a, cos a)
            float ci = cos(ti - side * a), co = cos(to - side * a);
            if (ci > 0.0 && co > 0.0)                        // 受光段与可见段都从槽口开始 → 交集 = min
                sum += ci * co * min(GrooveFrac(a, side, ti), GrooveFrac(a, side, to));
        }
        return sum / (2.0 * cos(a) * cos(ti) * cos(to));     // facet 面积 / 槽口面积 = 1/(2cos a) 每侧
    }`;

const HLSL_QON = `
    // Oren-Nayar 1994 定性模型（QON）。sigma = facet 坡度角标准差（弧度）
    float3 Fd_OrenNayarQON(float3 albedo, float sigma, float NoL, float NoV, float LoV) {
        float s2 = sigma * sigma;
        float A  = 1.0 - 0.5 * s2 / (s2 + 0.33);
        float B  = 0.45 * s2 / (s2 + 0.09);
        float s  = LoV - NoL * NoV;                          // sinθi·sinθo·cos(φi − φo)
        float st = s > 0.0 ? s / max(NoL, NoV) : 0.0;        // = max(0, cosΔφ)·sinα·tanβ，无 acos / tan
        return albedo * INV_PI * (A + B * st);
    }`;

const HLSL_EON = `
    // Fujii 2013（FON）+ EON（Portsmouth, Kutz, Hill 2025）。r ∈ [0,1]；由论文 GLSL Listing 1 改写
    static const float FON_C1 = 0.5 - 2.0 / (3.0 * PI);
    static const float FON_C2 = 2.0 / 3.0 - 28.0 / (15.0 * PI);

    float E_FON(float mu, float r) {                         // FON 方向反照率（ρ = 1，解析）
        mu = max(mu, 1e-4);
        float AF = 1.0 / (1.0 + FON_C1 * r), BF = r * AF;
        float Si = sqrt(1.0 - mu * mu);
        float G  = Si * (acos(mu) - Si * mu) + (2.0 / 3.0) * ((Si / mu) * (1.0 - Si * Si * Si) - Si);
        return AF + BF * INV_PI * G;
    }
    float3 Fd_EON(float3 rho, float r, float NoL, float NoV, float LoV) {
        float  s      = LoV - NoL * NoV;
        float  st     = s > 0.0 ? s / max(NoL, NoV) : s;       // FON：s ≤ 0 时 t = 1（不截断）
        float  AF     = 1.0 / (1.0 + FON_C1 * r);
        float3 f_ss   = rho * INV_PI * AF * (1.0 + r * st);    // 单次散射（FON）
        float  EFo    = E_FON(NoV, r), EFi = E_FON(NoL, r);
        float  avgEF  = AF * (1.0 + FON_C2 * r);                // ⟨Ê_F⟩
        float3 rho_ms = rho * rho * avgEF / (1.0 - rho * (1.0 - avgEF));
        float3 f_ms   = rho_ms * INV_PI * max(1e-7, 1.0 - EFo) * max(1e-7, 1.0 - EFi) / max(1e-7, 1.0 - avgEF);
        return f_ss + f_ms;                                     // 逐通道：f_ms 对 ρ 非线性
    }
    float3 E_EON(float3 rho, float r, float mu) {             // EON 方向反照率（解析，IBL 用）
        float AF = 1.0 / (1.0 + FON_C1 * r), EF = E_FON(mu, r), avgEF = AF * (1.0 + FON_C2 * r);
        float3 rho_ms = rho * rho * avgEF / (1.0 - rho * (1.0 - avgEF));
        return rho * EF + rho_ms * (1.0 - EF);
    }`;

const HLSL_BURLEY = `
    // Burley 2012（Disney diffuse）。r = 感知粗糙度（不是 α = r²）
    float3 Fd_Burley(float3 albedo, float r, float NoL, float NoV, float LoH) {
        float FD90 = 0.5 + 2.0 * r * LoH * LoH;               // LoH = cosθd
        float FL   = F_Schlick90(FD90, NoL);                  // 1 + (FD90 − 1)(1 − NoL)^5
        float FV   = F_Schlick90(FD90, NoV);
        return albedo * INV_PI * FL * FV;
    }`;

const HLSL_FROST = `
    // Lagarde & de Rousiers 2014, Moving Frostbite to PBR · Listing 1（F_Schlick(f0=1, f90, u) 写成 F_Schlick90）
    float Fr_DisneyDiffuse(float NdotV, float NdotL, float LdotH, float linearRoughness) {
        float energyBias   = lerp(0.0, 0.5, linearRoughness);
        float energyFactor = lerp(1.0, 1.0 / 1.51, linearRoughness);
        float fd90         = energyBias + 2.0 * LdotH * LdotH * linearRoughness;
        float lightScatter = F_Schlick90(fd90, NdotL);
        float viewScatter  = F_Schlick90(fd90, NdotV);
        return lightScatter * viewScatter * energyFactor;
    }
    // 用法：Fd = albedo * Fr_DisneyDiffuse(NoV, NoL, LoH, r) / PI;   NoV = abs(dot(N, V)) + 1e-5`;

const HLSL_BAKE = `
    // 漫反射预积分 LUT：E_d(NoV, r) = ∫ π·f/ρ · cosθi dωi / π（cosine 采样，64 个 Hammersley 点）
    RWTexture2D<float> DiffuseDFG_LUT;                         // 32×32，x = NoV，y = r
    [numthreads(8, 8, 1)]
    void CS_BakeDiffuseDFG(uint3 id : SV_DispatchThreadID) {
        float  NoV = (id.x + 0.5) / 32.0, r = (id.y + 0.5) / 32.0;
        float3 V   = float3(sqrt(1.0 - NoV * NoV), 0.0, NoV);
        float  E   = 0.0;
        for (uint i = 0; i < 64; ++i) {
            float  pdf;
            float3 L   = SampleCosineHemisphere(Hammersley(i, 64), pdf);
            float  LoH = sqrt(saturate(0.5 + 0.5 * dot(L, V)));   // cos²θd = (1 + L·V) / 2
            E += Fr_DisneyDiffuse(NoV, L.z, LoH, r);           // f·cos/pdf = π·f/ρ = Fr
        }
        DiffuseDFG_LUT[id.xy] = E / 64.0;
    }`;

const HLSL_LIGHTING = `
    Texture2D<float4>       GBufferA;          // baseColor（sRGB 纹理 → 采样即线性）· metallic
    Texture2D<float4>       GBufferB;          // 法线（八面体编码）
    Texture2D<float4>       GBufferC;          // r · diffuseModel
    Texture2D<float>        DiffuseDFG_LUT;    // CS_BakeDiffuseDFG 一次烘焙
    StructuredBuffer<Light> LightBuffer;
    RWTexture2D<float4>     SceneColor;

    float3 EvalDiffuse(uint model, float3 albedo, float r, float NoL, float NoV, float LoV) {
        float LoH = sqrt(saturate(0.5 + 0.5 * LoV));          // cosθd，不必算 H
        if (model == 0) return Fd_Lambert(albedo);
        if (model == 1) return Fd_EON(albedo, r, NoL, NoV, LoV);
        return albedo * Fr_DisneyDiffuse(NoV, NoL, LoH, r) * INV_PI;
    }

    [numthreads(8, 8, 1)]
    void CS_DeferredLighting(uint3 id : SV_DispatchThreadID) {
        GBufferData g = DecodeGBuffer(id.xy);
        float3 V   = normalize(CameraPos - g.worldPos);
        float  NoV = abs(dot(g.N, V)) + 1e-5;                  // 法线贴图会让 NoV ≤ 0
        float3 diffuseColor = g.baseColor * (1.0 - g.metallic); // 金属没有漫反射
        float3 Lo = 0;
        for (uint i = 0; i < NumLights; ++i) {
            float3 L   = LightBuffer[i].dir;
            float  NoL = saturate(dot(g.N, L));
            if (NoL <= 0.0) continue;
            Lo += EvalDiffuse(g.model, diffuseColor, g.r, NoL, NoV, dot(L, V)) * LightBuffer[i].E * NoL;
            // + 镜面项（见 02 模块），能量分配见正文
        }
        // 环境漫反射：SH 辐照度 × E_d(NoV) —— Lambert 时 E_d ≡ 1
        float3 Ed = g.model == 0 ? diffuseColor
                  : g.model == 1 ? E_EON(diffuseColor, g.r, NoV)
                  : diffuseColor * DiffuseDFG_LUT.SampleLevel(LinearClamp, float2(NoV, g.r), 0);
        Lo += Ed * EvalSHIrradiance(g.N) * INV_PI;
        SceneColor[id.xy] = float4(Lo, 1);
    }`;

// ── 附录：CS_DeferredLighting 用到的 G-buffer 解码与 SH 辐照度（布局 = Graph 里 BasePass 的三张 RT + 深度） ──
const HLSL_GBUFFER = `
    // 方向光：dir 指向光源（世界空间单位向量），E = 法向入射照度（线性 RGB）
    struct Light {
        float3 dir;
        float3 E;
    };
    cbuffer ViewConstants {
        float4x4 ClipToWorld;      // (View · Proj)⁻¹
        float3   CameraPos;
        uint     NumLights;
        float2   InvViewSize;      // 1 / 分辨率
    };
    Texture2D<float> SceneDepth;   // 设备深度（BasePass 写出）
    SamplerState     LinearClamp;

    // 逐像素表面属性（GBufferA = baseColor · metallic，B = 八面体法线，C = r · model）
    struct GBufferData {
        float3 baseColor;          // 线性（SRGB 格式的 Load 也会解码）
        float  metallic;
        float3 N;                  // 世界空间单位法线
        float  r;                  // 感知粗糙度
        uint   model;              // 0 = Lambert，1 = EON，2 = Frostbite
        float3 worldPos;
    };

    // 八面体解码（Cigolle et al. 2014）：e ∈ [−1, 1]² → 单位法线；|x| + |y| > 1 的部分是折上来的下半球
    float3 OctDecode(float2 e) {
        float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
        float  t = saturate(-n.z);
        n.x += n.x >= 0.0 ? -t : t;
        n.y += n.y >= 0.0 ? -t : t;
        return normalize(n);
    }

    // 读三张 G-buffer + 深度（Load：整数坐标、不过滤），世界坐标由深度反投影
    GBufferData DecodeGBuffer(uint2 pix) {
        float4 a = GBufferA.Load(int3(pix, 0));
        float4 b = GBufferB.Load(int3(pix, 0));
        float4 c = GBufferC.Load(int3(pix, 0));
        GBufferData g;
        g.baseColor = a.rgb;
        g.metallic  = a.a;
        g.N         = OctDecode(b.xy * 2.0 - 1.0);
        g.r         = c.r;
        g.model     = (uint)round(c.g * 255.0);              // RGBA8 UNORM 里存 model / 255
        float2 uv   = (pix + 0.5) * InvViewSize;
        float4 clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, SceneDepth.Load(int3(pix, 0)), 1.0);
        float4 wp   = mul(ClipToWorld, clip);
        g.worldPos  = wp.xyz / wp.w;
        return g;
    }`;

const HLSL_SH = `
    // 天空 radiance 投影到实 SH（l ≤ 2）的 9 组 rgb 系数 L_lm；用 float4 避开 cbuffer 中 float3 数组的填充
    cbuffer SkyLight {
        float4 SkyIrradianceSH[9];  // 顺序 (l, m)：00 | 1−1 10 11 | 2−2 2−1 20 21 22
    };

    // E(N) = ∫ L(ω) max(0, N·ω) dω ≈ Σ_l Â_l Σ_m L_lm Y_lm(N)，Â_0 = π，Â_1 = 2π/3，Â_2 = π/4（Ramamoorthi & Hanrahan 2001）
    float3 EvalSHIrradiance(float3 N) {
        float  x = N.x, y = N.y, z = N.z;
        float3 E = PI * 0.282095 * SkyIrradianceSH[0].rgb;                                    // l = 0：Y = 1/(2√π)
        E += (2.0 * PI / 3.0) * 0.488603 * (y * SkyIrradianceSH[1].rgb + z * SkyIrradianceSH[2].rgb + x * SkyIrradianceSH[3].rgb);
        E += (PI / 4.0) * (1.092548 * (x * y * SkyIrradianceSH[4].rgb + y * z * SkyIrradianceSH[5].rgb + x * z * SkyIrradianceSH[7].rgb)
                         + 0.315392 * (3.0 * z * z - 1.0) * SkyIrradianceSH[6].rgb
                         + 0.546274 * (x * x - y * y) * SkyIrradianceSH[8].rgb);
        return max(E, 0.0);         // L2 截断在高对比天空下会振铃出负值
    }`;

// ── 3D lobe：(φ, θ) 网格，局部 n = +Z；mesh3d 是 y 朝上，W() 把局部坐标转成世界坐标（x → x，z → y，y → −z） ──
const L3D = {
    nu: 64, nv: 24, scale: 1.3,
    dir: (i, j) => BX.dir((j / L3D.nv) * Math.PI / 2, (i / L3D.nu) * 2 * Math.PI),
    W: (v) => [v[0], v[2], -v[1]],
    rgb: (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255),
};

PipelineLab.mount({
    paramSteps: { albedo: ['lambert', 'lfurnace', 'retro', 'eon', 'graze', 'prod'], thetaI: ['lambert', 'sample', 'vcavity', 'qon', 'retro', 'eon', 'burley', 'graze'], lightAz: ['lambert', 'retro', 'graze'], sigma: ['sample', 'vcavity', 'qon', 'retro', 'onenergy', 'eon', 'prod'], rough: ['sample', 'retro', 'burley', 'graze', 'benergy', 'frost', 'prod'], nSamp: ['sample'], l3dMode: ['qon', 'burley'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: 'BxDF 01 · 漫反射：Lambert → Oren-Nayar → Burley',
    lang: 'zh-CN',
    codeLang: 'hlsl',
    seed: 1994,
    regen: false,

    params: [
        {
            id: 'albedo', label: 'ρ', value: '#b06a48', map: BX.hex2lin, title: 'albedo（sRGB 拾色 → 线性 RGB）',
            desc: '漫反射反照率 ρ（拾色器取 sRGB，内部转成线性 RGB）：决定反射多少、什么颜色。除 EON 外各模型都与 ρ 成正比，k = π f/ρ 不变，只有球体渲染变色；EON 的多次散射项 f<sub>ms</sub> 对 ρ 非线性（∝ ρ²），调亮 albedo 时 Step 06 剖面和 Step 08 lobe 里的 EON 跟着抬高。',
        },
        {
            id: 'thetaI', label: 'θi', min: 0, max: 85, step: 1, value: 60, unit: '°', map: (d) => d * Math.PI / 180, title: 'lobe / 热图中的入射角 θi；球体渲染中 = 主光的俯仰角',
            desc: '入射角 θ<sub>i</sub>：在 lobe、热图和 3D 视图里是 ω<sub>i</sub> 与法线的夹角，在球体渲染里是主光的<strong>俯仰角</strong>（从视线方向向上抬 θ<sub>i</sub>；光源方位 = 0° 时它就是光线与视线的夹角，也是球心处的入射角）。拖到 70° 以上看 QON 回射瓣和 Burley 掠射亮边最强（Step 05 / 09 的 3D lobe），拖到 0°（顶光）看 Step 06 的 QON 球变"平"、Step 10 的 Burley 轮廓亮环。',
        },
        {
            id: 'lightAz', label: '光源方位', min: -170, max: 170, step: 1, value: 0, unit: '°',
            desc: '球体渲染里主光的<strong>偏航角</strong>：绕竖直轴转动主光（0° = 光在视线正上方的竖直面内，正 = 右侧，±170° ≈ 逆光），俯仰由 θ<sub>i</sub> 决定。只影响球体渲染和球面比值图；lobe、热图与曲线始终画在入射面内，不受它影响。',
        },
        {
            id: 'sigma', label: 'σ (ON)', min: 0, max: 90, step: 1, value: 45, unit: '°', map: (d) => d * Math.PI / 180, title: 'Oren-Nayar facet 坡度角标准差 σ；FON / EON 用 r = σ/(π/2)',
            desc: 'Oren-Nayar 的 facet 坡度角标准差 σ（弧度意义上的粗糙度，0 = Lambert；FON / EON 取 r = σ/(π/2)）。σ 越大 V 槽越深：回射越强、A 越小（正面越暗）——看 Step 04 的槽剖面、Step 05 的 3D lobe 向光源一侧倾斜，以及 Step 07 白炉里丢掉的能量。',
        },
        {
            id: 'rough', label: 'r (Burley)', min: 0, max: 1, step: 0.01, value: 0.7, digits: 2, title: 'Burley / Frostbite 的感知粗糙度 r（直接用 r，不是 α = r²）',
            desc: 'Burley / Frostbite 的感知粗糙度 r（直接用 r，不是 α = r²），决定掠射极限 F<sub>D90</sub> = 0.5 + 2r·cos²θ<sub>d</sub>。r ≈ 0.25 时回射配置下等于 Lambert；拖到 0 看掠射变暗，拖到 1 看轮廓亮边与白炉能量增益（Step 09–12）。',
        },
        {
            id: 'nSamp', label: 'N 样本', options: [64, 256, 1024], value: 256, title: '采样散点图的样本数',
            desc: '采样散点图的样本数，只影响 Step 03 的两张散点：点越多越容易看出 cosine 采样向中心（法线方向）聚集、均匀采样在等面积投影上铺满。收敛曲线固定扫 N = 2…2048，不受它影响。',
        },
        {
            id: 'l3dMode', label: '3D 半径', options: [{ value: 'k', label: 'k = πf/ρ' }, { value: 'kcos', label: 'k·cosθo' }], value: 'k',
            desc: 'Step 05 / 09 的 3D lobe 用什么量作半径：<code>k = π f/ρ</code> 与入射平面 lobe 一致，Lambert 是半径 1 的半球，最能看出贴近地平线一圈的回射抬高与压暗；<code>k·cosθo</code> 是常说的"余弦瓣"（出射 radiant intensity），Lambert 是与表面相切的球，QON 的瓣整体倒向光源一侧。',
        },
    ],

    stages: [
        {
            id: 'prep', lifetime: 'param', run(S) {
                S.rF = S.sigma / (Math.PI / 2);
                S.rhoL = BX.lum(S.albedo);
                S.wi = BX.dir(S.thetaI, 0);
            },
        },
        {
            // 当前参数下各模型的白炉曲线 E(θo)（ρ = 1）
            id: 'furnace', lifetime: 'param', run(S) {
                S.Et = {};
                for (const key of Object.keys(D1.M)) {
                    const m = D1.M[key];
                    S.Et[key] = D1.table((wi, wo) => m.k(wi, wo, S, 1) * D1.IPI, PL.buf(S, 'Et_' + key, D1.NT));
                }
                let err = 0;
                for (let j = 0; j < D1.NT; j++) err = Math.max(err, Math.abs(S.Et.fon[j] - D1.EFON(Math.cos(D1.thetaOf(j)), S.rF)));
                S.fonErr = err;
            },
        },
        {
            // 固定粗糙度的参照曲线（不随参数变）
            id: 'furnaceFixed', lifetime: 'resident', run(S) {
                const mk = (k) => D1.table(k, new Float32Array(D1.NT));
                S.Efix = {
                    qon: [20, 45, 90].map((d) => ({ d, tab: mk((wi, wo) => D1.kQON(wi, wo, d * Math.PI / 180) * D1.IPI) })),
                    burley: [0, 0.5, 1].map((r) => ({ r, tab: mk((wi, wo) => D1.kBurley(wi, wo, r) * D1.IPI) })),
                    frost: [0, 0.5, 1].map((r) => ({ r, tab: mk((wi, wo) => D1.kFrost(wi, wo, r) * D1.IPI) })),
                };
            },
        },
        {
            // flatland V 腔：坡度分布积分 + 剖面上的代表性槽
            id: 'vcav', lifetime: 'param', run(S) {
                const { a, w } = D1.slopeNodes(S.sigma), ti = S.thetaI;
                const th = [], kv = [], kq = [];
                for (let d = -88; d <= 88; d += 2) {
                    const to = d * Math.PI / 180;
                    let k = 0;
                    for (let j = 0; j < a.length; j++) k += w[j] * D1.vGroove(a[j], ti, to);
                    th.push(d); kv.push(k); kq.push(D1.kQON(S.wi, [Math.sin(to), 0, Math.cos(to)], S.sigma));
                }
                S.vc = { th, kv, kq };
                // 7 个代表性槽：按分布的分位数取坡度，固定打乱顺序
                const cdf = [];
                w.reduce((s, v, j) => (cdf[j] = s + v), 0);
                const q = [0, 1, 2, 3, 4, 5, 6].map((k) => { const p = (k + 0.5) / 7; const j = cdf.findIndex((c) => c >= p); return a[j < 0 ? a.length - 1 : j]; });
                S.grooves = [3, 0, 5, 1, 6, 2, 4].map((i) => q[i]);
            },
        },
        {
            // 半球热图（ωo 上，ωi 固定）与积分被积函数
            id: 'maps', lifetime: 'param', run(S) {
                const R = D1.RES_H, rho = S.rhoL, wi = S.wi;
                const L2 = (v) => Math.log2(Math.max(1e-3, v));
                S.hmQON = BX.hemiImage((wo) => L2(D1.M.qon.k(wi, wo, S, rho)), R, S.hmQON);
                S.hmEON = BX.hemiImage((wo) => L2(D1.M.eon.k(wi, wo, S, rho)), R, S.hmEON);
                S.hmBurley = BX.hemiImage((wo) => L2(D1.M.burley.k(wi, wo, S, rho)), R, S.hmBurley);
                S.hmCos = BX.hemiImage((w) => w[2], R, S.hmCos);
                let sum = 0, cnt = 0;
                for (const v of S.hmCos) if (v === v) { sum += v; cnt++; }
                S.cosIntegral = sum * (2 * Math.PI / cnt);              // 等面积投影：每个像素 Δω = 2π / 像素数
            },
        },
        {
            id: 'samp', lifetime: 'param', run(S) {
                const rnd = PL.rng((S.seed ^ 0x9e3779b9) >>> 0), n = S.nSamp;
                S.smpCos = []; S.smpUni = [];
                for (let i = 0; i < n; i++) { S.smpCos.push(BX.sampleCosine(rnd(), rnd())); S.smpUni.push(BX.sampleUniform(rnd(), rnd())); }
                // 收敛：每个 N 做 T 次独立估计，相对 RMS 误差；ωo 取 θ 参数
                const wo = BX.dir(S.thetaI), T = 8, Ns = [];
                for (let N = 2; N <= 2048; N *= 2) Ns.push(N);
                const rr = PL.rng(12345), out = [];
                for (const key of ['lambert', 'burley', 'qon']) {
                    const m = D1.M[key], k = (wi) => m.k(wi, wo, S, 1);
                    const ref = BX.albedoQuad((wi) => k(wi) * D1.IPI, wo, 48);
                    for (const mode of ['cos', 'uni']) {
                        const errs = Ns.map((N) => {
                            let se = 0;
                            for (let t = 0; t < T; t++) {
                                let e = 0;
                                for (let i = 0; i < N; i++) {
                                    if (mode === 'cos') e += k(BX.sampleCosine(rr(), rr()));
                                    else { const wi = BX.sampleUniform(rr(), rr()); e += 2 * k(wi) * wi[2]; }
                                }
                                se += (e / N - ref) ** 2;
                            }
                            return Math.max(1.5e-4, Math.sqrt(se / T) / ref);
                        });
                        out.push({ key, mode, errs });
                    }
                }
                S.conv = { Ns, out };
            },
        },
        {
            // 漫反射预积分 LUT（与 CS_BakeDiffuseDFG 相同：cosine Hammersley 64 点）
            id: 'lut', lifetime: 'resident', run(S) {
                const N = 32, n = 64, lutF = PL.buf(S, 'lutF', N * N), lutB = PL.buf(S, 'lutB', N * N);
                for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
                    const mu = (x + 0.5) / N, r = (y + 0.5) / N, wo = [Math.sqrt(1 - mu * mu), 0, mu];
                    let eF = 0, eB = 0;
                    for (let i = 0; i < n; i++) { const [u1, u2] = BX.hammersley(i, n), wi = BX.sampleCosine(u1, u2); eF += D1.kFrost(wi, wo, r); eB += D1.kBurley(wi, wo, r); }
                    lutF[y * N + x] = eF / n; lutB[y * N + x] = eB / n;
                }
            },
        },
        {
            // 3D lobe（Step 05 / 09）：ωi 固定，k(ωi, ωo)（或 k·cosθo）在 ωo 半球的 (φ, θ) 网格上；Lambert 的最大值 = 1
            id: 'lobe3d', lifetime: 'param', lazy: true, run(S) {
                const { nu, nv } = L3D, NU = nu + 1, NV = nv + 1, withCos = S.l3dMode === 'kcos';
                S.l3d = {};
                for (const key of ['qon', 'burley']) {
                    const m = D1.M[key], R = PL.buf(S, 'l3d_' + key, NU * NV);
                    let mx = 0, at = 0;
                    for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) {
                        const wo = L3D.dir(i, j), v = m.k(S.wi, wo, S, S.rhoL) * (withCos ? Math.max(0, wo[2]) : 1);
                        R[j * NU + i] = v;
                        if (v > mx) { mx = v; at = j * NU + i; }
                    }
                    S.l3d[key] = { R, max: mx, at };
                }
            },
        },
    ],

    intro: {
        title: '漫反射 BRDF 的三代：常数 → 微表面 V 腔 → 经验拟合',
        flow: ['Lambert ρ/π', '白炉 E = ρ', 'cosine 采样', 'V 腔 σ', 'QON A·B', '回射', 'FON → EON', 'Burley F_D90', 'Frostbite 归一化', '生产版'],
        html: '<p>每个模型拆成若干步：公式 + HLSL + <strong>由页面内 JS 参考实现实时算出的结果</strong>（球体渲染、入射平面 lobe、半球热图、白炉曲线、采样散点）。所有模型都写成 <code>k(ωi, ωo) = π·f/ρ</code> —— "相对 Lambert 的倍数"，热图里 1 = 与 Lambert 相同。' +
            '顶栏 <code>θi</code> 是 lobe / 热图的入射角，在球体渲染中是主光的俯仰角（<code>光源方位</code> 控制偏航，方位 = 0° 时 θi 就是光线与视线的夹角）；<code>σ</code> 驱动 Oren-Nayar（FON / EON 用 r = σ/(π/2)）；<code>r</code> 驱动 Burley / Frostbite。悬停热图可读数。</p>' +
            '<p>镜面（GGX）与 Phong / Blinn-Phong 的历史在 02 模块；漫反射与镜面的能量分配在最后一步与 06 模块。</p>' + BX.nav('diffuse'),
    },

    steps: [
        // ───────────────────────── Lambert ─────────────────────────
        {
            id: 'lambert', title: 'Lambert · 常数 BRDF 为什么是 ρ/π', tag: 'BASE',
            desc: 'Lambert 假设出射 radiance 与观察方向无关，所以 f 是常数。常数取多少由能量守恒决定：反射率 = ∫ f·cosθ dω。半球对 <code>cosθ</code> 的积分是 <strong>π</strong> 而不是 2π —— cos 权重把半球"压扁"成单位圆盘，投影立体角 = 圆盘面积 π。要让反射率恰好等于 albedo ρ，f 只能是 <code>ρ/π</code>。',
            formula: [
                'f_Lambert(ωi, ωo) = ρ / π',
                '∫_Ω cosθ dω = ∫_0^{2π} ∫_0^{π/2} cosθ sinθ dθ dφ = π   // 投影立体角 = 单位圆盘面积',
                'L_o = f · E · cosθi = (ρ/π) · E · cosθi   // 方向光，E = 法向入射照度',
            ],
            code: { title: 'HLSL · Fd_Lambert', src: HLSL_LAMBERT },
            previews: [
                { id: 'lambertSphere', label: 'Lambert 球', sub: '方向光：俯仰 = θi，偏航 = 光源方位，E = π', render: (c, S) => D1.sphere(c, D1.M.lambert, S) },
                {
                    id: 'lambertLobe', label: '入射平面 lobe', sub: '蓝：k = π f/ρ ≡ 1（半圆）；橙：k·cosθo', size: [240, 150],
                    render: (c, S) => BX.lobe(c, { wi: S.wi, scale: 1.05, series: [{ fn: () => 1, color: '#6ab0e0', label: 'π f/ρ' }, { fn: (wo) => wo[2], color: '#e0a060', label: 'π f/ρ · cosθo' }] }),
                },
                {
                    id: 'cosInteg', label: 'cosθ 被积函数（等面积投影）', sub: '每像素 Δω 相同 → Σ cosθ·Δω ≈ π',
                    render: (c, S) => PL.draw.field(c, S.hmCos, D1.RES_H, D1.RES_H, { map: 'viridis', range: [0, 1] }),
                    probe: (x, y, S) => { const w = BX.hemiDir(x, y, D1.RES_H); return w ? `cosθ = ${w[2].toFixed(3)}   Σ = ${S.cosIntegral.toFixed(4)}` : `Σ cosθ·Δω = ${S.cosIntegral.toFixed(4)}`; },
                },
            ],
            notes: '拖 <code>θi</code>：球上的亮区随光转动，但每个像素只取决于 cosθi，与视线无关。lobe 图里 f 是半圆（各向同性的 radiance），乘上 cosθo 才是常说的"余弦瓣"（radiant intensity）。第三张图悬停可读出数值积分 Σ cosθ·Δω（80² 像素，≈ 3.14）。',
            warn: '1/π 和光源单位是一对：有的引擎把 π 吸收进光源强度（"灯光 = π·E"），shader 里就看不到 1/π。两套约定混用会让漫反射差 π 倍（≈ 3.14×）——先确认光源单位，再决定 BRDF 里要不要 1/π。',
        },
        {
            id: 'lfurnace', title: 'Lambert · 白炉测试 E(ωo) = ρ', tag: 'ENERGY',
            desc: '白炉测试（white furnace）：把物体放进各方向 radiance 恒为 1 的环境，此时每个像素的 radiance 就是方向反照率 <code>E(ωo) = ∫ f cosθi dωi</code>。ρ = 1 的能量守恒 BRDF 在任何 ωo 下都应给出 1，球"消失"在背景里；E &lt; 1 是丢能量（偏暗），E &gt; 1 是凭空造能量（偏亮、多次反射时发散）。Lambert 的 E 恒等于 ρ —— 后面所有模型都拿这条线作参照。',
            formula: [
                'E(ωo) = ∫_Ω f(ωi, ωo) cosθi dωi = (ρ/π) · π = ρ   // 与 ωo 无关',
                '白炉：L_env ≡ 1  ⇒  L_o(ωo) = E(ωo)   // ρ = 1 时守恒 ⇔ E ≡ 1',
            ],
            code: { title: 'HLSL · CS_WhiteFurnace（Debug）', src: HLSL_FURNACE },
            previews: [
                {
                    id: 'lambertE', label: 'E(θo)', sub: '数值积分（albedoQuad 20×40）', size: [360, 180],
                    render: (c, S) => D1.plotE(c, [
                        { tab: S.Et.lambert, color: '#9aa4ae', label: 'Lambert ρ = 1' },
                        { tab: S.Et.lambert.map((v) => v * S.rhoL), color: '#e0a060', label: 'Lambert ρ = 当前 albedo 亮度' },
                        { tab: S.Et.lambert.map((v) => v * Math.PI), color: '#e06c75', dash: [4, 3], label: '忘了 1/π：E = π' },
                    ], { y: [0, 3.4] }),
                },
                { id: 'lambertFurnace', label: '白炉 · ρ = 1', sub: '背景 = 环境 1；球消失（虚线 = 轮廓）', render: (c, S) => D1.furnace(c, S.Et.lambert) },
                { id: 'lambertFurnaceRho', label: '白炉 · 当前 albedo', sub: '球的颜色 = ρ 本身', render: (c, S) => D1.furnace(c, S.Et.lambert, { tint: S.albedo }) },
            ],
            notes: '中点积分在 cosθ 上对线性函数精确，所以 Lambert 的数值 E 与 1 相差只有浮点误差。白炉球的显示：线性 radiance × 0.5 再 sRGB 编码，背景（环境 = 1）与 E = 1 的像素同色。',
        },
        {
            id: 'sample', title: 'Lambert · 余弦重要性采样（Malley）', tag: 'SAMPLE',
            desc: '用 Monte Carlo 估计 ∫ f·L·cosθ dω 时，按 <code>pdf ∝ cosθ</code> 抽样可以把 cosθ 整个消掉：权重 f·cosθ/pdf = π·f，对 Lambert 恒等于 ρ —— <strong>零方差</strong>。Malley 方法：在单位圆盘上均匀撒点，再竖直投影到半球；圆盘面积 π 正是上一步的投影立体角。ON 和 Burley 都是"Lambert × 缓变因子 k"，所以 cosine 采样对它们同样有效。',
            formula: [
                'pdf(ω) = cosθ / π',
                '(u_1, u_2) → r = √u_1, φ = 2π u_2 → ω = (r cosφ, r sinφ, √(1 − u_1))   // 圆盘均匀 → 投影到半球',
                'weight = f · cosθi / pdf = π f = ρ · k(ωi, ωo)   // Lambert：k ≡ 1 ⇒ 零方差',
            ],
            code: { title: 'HLSL · SampleCosineHemisphere', src: HLSL_SAMPLE },
            previews: [
                {
                    id: 'cosScatter', label: 'cosine 采样', sub: '背景 = 目标密度 cosθ/π（等面积投影）', size: [180, 180],
                    render: (c, S) => drawScatter(c, S, S.smpCos, true),
                },
                {
                    id: 'uniScatter', label: '均匀半球采样', sub: '等面积投影下均匀分布', size: [180, 180],
                    render: (c, S) => drawScatter(c, S, S.smpUni, false),
                },
                {
                    id: 'conv', label: '相对 RMS 误差 vs N', sub: 'ωo 取 θi 参数；8 次独立估计', size: [360, 200],
                    render: (c, S) => {
                        const col = { lambert: '#9aa4ae', burley: '#6ab0e0', qon: '#e0a060' }, nm = { lambert: 'Lambert', burley: 'Burley', qon: 'QON' };
                        PL.draw.plot(c, {
                            logX: true, logY: true, x: [2, 2048], y: [1e-4, 1], xLabel: 'N', yLabel: 'rel. RMS',
                            series: S.conv.out.map((s) => ({ data: s.errs, x: S.conv.Ns, color: col[s.key], dash: s.mode === 'uni' ? [4, 3] : undefined, label: `${nm[s.key]} · ${s.mode === 'cos' ? 'cos' : 'uniform'}${s.key === 'lambert' && s.mode === 'cos' ? '（≡0，画在下限）' : ''}` })),
                        });
                    },
                },
            ],
            notes: '散点画在等面积投影上（半径² = 1 − cosθ）：均匀立体角采样在圆盘上均匀，cosine 采样向中心聚集，密度正比于背景色。收敛图：实线 = cosine 采样，虚线 = 均匀采样；两者都按 N<sup>−1/2</sup> 下降，cosine 采样的常数小得多，Lambert 则直接为 0。改 <code>N 样本</code> 只影响散点。',
            warn: 'cosine 采样的公式只在 n = +Z 的局部坐标系里成立：世界空间要用 TBN 把样本转出去，pdf 里的 cosθ 必须是同一个法线（法线贴图 N）下的值。EON 论文另给了贴合掠射回射峰的 CLTC 采样：掠射处方差降低超过 100×，开销约为 cosine 采样的 2 倍。',
        },
        // ───────────────────────── Oren-Nayar ─────────────────────────
        {
            id: 'vcavity', title: 'Oren-Nayar · V 腔微表面与 σ', tag: 'MICRO',
            desc: 'Oren &amp; Nayar 把粗糙表面看成大量对称的 V 形槽（V-cavity），每个槽的两个 facet 都是 Lambert 面；facet 坡度角服从均值 0、标准差 <strong>σ</strong> 的高斯分布 —— σ 就是粗糙度，单位是弧度。为什么用 V 槽：所有槽口都在同一平面上，遮蔽（masking）和阴影（shadowing）只发生在槽内部，可以解析写出（与 Torrance-Sparrow 同款几何）。关键现象：从光源一侧看，看到的几乎都是受光的 facet；从对侧看，看到的多是背光的 facet —— 这就是粗糙表面的<strong>回射</strong>（retro-reflection）。',
            formula: [
                'θ_a ~ N(0, σ^2)，σ ∈ [0, π/2]   // facet 坡度角，弧度',
                'g(θ) = min(1, 2 / (1 + tanθ_a · tanθ))   // 从槽口往下的可见比例（θ 越过对面槽壁时；否则 = 1）',
                '受光且可见的比例 = min(g(θi), g(θo))   // 两段都从槽口开始 → 交集取 min',
                'k = Σ_{facet} ⟨n_f·ωi⟩⟨n_f·ωo⟩ · min(g_i, g_o) / (2 cosθ_a · cosθi · cosθo)   // 单次散射，π f/ρ',
            ],
            code: { title: 'HLSL · VGrooveK（flatland 参考实现）', src: HLSL_VGROOVE },
            previews: [
                { id: 'grooveRetro', label: '从光源一侧看：ωo = ωi', sub: '粗线 = 看得见的 facet；暖色 = 受光，蓝 = 背光；细灰线 = 被遮住', size: [360, 130], render: (c, S) => drawGrooves(c, S, S.thetaI, `θo = +${D1.deg(S.thetaI)}（回射方向）`) },
                { id: 'grooveFwd', label: '从对侧看：ωo 镜像', sub: '同一组槽，换个观察方向', size: [360, 130], render: (c, S) => drawGrooves(c, S, -S.thetaI, `θo = −${D1.deg(S.thetaI)}（前向 / 镜像侧）`) },
                {
                    id: 'vcPlot', label: '入射平面内 k(θo)', sub: 'V 腔数值积分 vs QON 公式 vs Lambert', size: [360, 190],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [-90, 90], y: [0, 2.2], xLabel: 'θo (°)，>0 = 光源一侧', yLabel: 'k = π f/ρ',
                        series: [
                            { data: S.vc.th.map(() => 1), x: S.vc.th, color: '#9aa4ae', label: 'Lambert' },
                            { data: S.vc.kv, x: S.vc.th, color: '#5ad07a', label: 'V 腔 flatland（单次散射，坡度分布积分）' },
                            { data: S.vc.kq, x: S.vc.th, color: '#e0a060', label: 'QON 公式' },
                        ],
                        marks: [{ x: S.thetaI * 180 / Math.PI, label: 'ωo = ωi', color: '#e8c25a' }, { x: -S.thetaI * 180 / Math.PI, label: '镜像', color: '#56d0e0' }],
                    }),
                },
            ],
            notes: '拖 <code>σ</code>：槽变深，回射侧的 k 抬高、对侧压低。拖 <code>θi</code>：光越斜，两侧差别越大。剖面上方两行字是这 7 个槽的统计：看到的投影面积中受光的比例、以及这组槽的 k。',
            warn: '这里的槽全部垂直于入射平面（flatland、单次散射、坡度取截断半正态 |θ_a| 的 48 点积分），比 ON 对槽方位做平均后的效果更强，只用来看机制；曲线量级与 QON 不必重合。ON 的完整模型还包含槽内二次反射（interreflection）项。',
        },
        {
            id: 'qon', title: 'Oren-Nayar · 定性模型 A、B、α、β 与 cos(φi − φo)', tag: 'MODEL',
            desc: '对 V 腔积分得到的完整模型较繁琐，论文同时给了一个"定性模型"（qualitative，下称 <strong>QON</strong>）：Lambert 乘以 <code>A + B·max(0, cos(φi − φo))·sinα·tanβ</code>。A &lt; 1 把正面整体压暗；B 项只在 ωo 与 ωi 位于同一侧（|φi − φo| &lt; 90°）时生效，并在两者都接近掠射时最大 —— 这就是回射瓣。σ = 0 时 A = 1、B = 0，退化为 Lambert。',
            formula: [
                'f_QON = (ρ/π) · (A + B · max(0, cos(φi − φo)) · sinα · tanβ)',
                'A = 1 − 0.5 · σ^2 / (σ^2 + 0.33)        B = 0.45 · σ^2 / (σ^2 + 0.09)',
                'α = max(θi, θo)        β = min(θi, θo)',
                's = ωi·ωo − cosθi cosθo = sinθi sinθo cos(φi − φo)   // Fujii 的向量写法',
                'max(0, cos(φi − φo)) · sinα · tanβ = s > 0 ? s / max(cosθi, cosθo) : 0   // 无 acos / tan',
            ],
            code: { title: 'HLSL · Fd_OrenNayarQON', src: HLSL_QON },
            previews: [
                {
                    id: 'qonAB', label: 'A(σ)、B(σ)', sub: '竖线 = 当前 σ', size: [300, 170],
                    render: (c, S) => PL.draw.plot(c, {
                        x: [0, 90], y: [0, 1.05], xLabel: 'σ (°)',
                        series: [
                            { fn: (d) => D1.qonAB(d * Math.PI / 180)[0], color: '#e0a060', label: 'A' },
                            { fn: (d) => D1.qonAB(d * Math.PI / 180)[1], color: '#c678dd', label: 'B' },
                        ],
                        marks: [{ x: S.sigma * 180 / Math.PI, label: `A=${D1.qonAB(S.sigma)[0].toFixed(2)} B=${D1.qonAB(S.sigma)[1].toFixed(2)}`, color: '#e8c25a' }],
                    }),
                },
                {
                    id: 'qonLobe', label: '入射平面 lobe', sub: 'k = π f/ρ；虚线 = 镜像方向', size: [240, 150],
                    render: (c, S) => BX.lobe(c, { wi: S.wi, series: [{ fn: () => 1, color: '#9aa4ae', label: 'Lambert' }, { fn: (wo) => D1.M.qon.k(S.wi, wo, S, 1), color: '#e0a060', label: 'QON' }] }),
                },
                {
                    id: 'qonMap', label: 'log₂ k(ωo)，ωi 固定', sub: '黄圈 = ωi；右半 = 光源一侧', render: (c, S) => D1.ratioMap(c, S.hmQON, S), probe: (x, y, S) => D1.hemiProbe(S.hmQON)(x, y),
                },
                {
                    id: 'qonLobe3d', needs: ['lobe3d'], large: true, label: '3D：QON 与 Lambert 的 lobe（整个 ωo 半球）',
                    sub: '实体 = QON（当前 σ），灰色线框 = Lambert。半径 = 顶栏"3D 半径"：k = π f/ρ（Lambert 为半球）或 k·cosθo（Lambert 为与表面相切的球）；两者同一比例尺，按 max(1, 峰值) 归一化。黄箭头 = ω<sub>i</sub>（指向光源），青色虚线 = 镜像方向。拖动旋转',
                    render: (c, S) => drawLobe3d(c, S, 'qon'),
                },
            ],
            notes: '热图是 ωo 的等面积投影，颜色 = log₂(π f/ρ)：橙 = 比 Lambert 亮，蓝 = 暗，范围 ±1（2× / 0.5×）。QON 的左半（s ≤ 0）是一块均匀的蓝 —— 那里 B 项被 max(0, ·) 截掉，只剩 A；下一步 Fujii 的修改正是针对这一点。',
            warn: 'σ 是弧度而不是 [0,1] 的"粗糙度"；把 GGX 的 r 或 α 直接当 σ 用没有物理依据（本页 FON / EON 用 r = σ/(π/2) 只是沿用 EON 论文的比较方式）。HLSL 里不要用 acos / tan：tanβ 在 θ → 90° 发散，<code>s / max(NoL, NoV)</code> 是同一个量且稳定。',
        },
        {
            id: 'retro', title: 'Oren-Nayar · 回射：顶光下粗糙球变"平"', tag: 'RETRO',
            desc: '把光源放在相机处（相位角 0°）：Lambert 球从中心到边缘按 cosθ 变暗；QON 球的每个像素都满足 θi = θo、φi = φo，B 项给出 <code>B·sinθ·tanθ</code>，把边缘抬亮，整个圆盘更"平"。满月几乎看不出边缘变暗就是这一类粗糙表面回射的典型例子（月面还有 opposition effect 等其它机制）。相位角增大后回射逐渐消失，QON 退化为"更暗的 Lambert"。',
            formula: [
                'θi = θo = θ, φi = φo   ⇒   k_QON = A + B · sinθ · tanθ',
                'L / E = (ρ/π) · cosθ · k = (ρ/π) · (A cosθ + B sin^2θ)   // 边缘 θ → 90°：Lambert → 0，QON → ρB/π',
            ],
            previews: [
                { id: 'retroLambert', label: 'Lambert · 头灯', sub: '光在相机处：相位角 0°', render: (c, S) => D1.sphere(c, D1.M.lambert, S, { lights: [{ dir: [0, 0, 1], color: [Math.PI, Math.PI, Math.PI] }] }) },
                { id: 'retroQON', label: 'QON · 头灯', sub: '相位角 0°，当前 σ', render: (c, S) => D1.sphere(c, D1.M.qon, S, { lights: [{ dir: [0, 0, 1], color: [Math.PI, Math.PI, Math.PI] }] }) },
                {
                    id: 'retroProfile', label: '顶光下赤道剖面 L/(ρE)', sub: '球面 x ∈ [−1, 1]', size: [300, 180],
                    render: (c, S) => {
                        const xs = [], ser = { lambert: [], qon: [], eon: [], burley: [] };
                        for (let i = 0; i <= 120; i++) {
                            const x = -1 + 2 * i / 120, ct = Math.sqrt(Math.max(0, 1 - x * x)), w = [Math.abs(x), 0, ct];
                            xs.push(x);
                            for (const k of Object.keys(ser)) ser[k].push(D1.M[k].k(w, w, S, S.rhoL) * ct / Math.PI);
                        }
                        PL.draw.plot(c, { x: [-1, 1], y: [0, 0.45], xLabel: 'x', series: Object.keys(ser).map((k) => ({ data: ser[k], x: xs, color: D1.M[k].color, label: D1.M[k].name })) });
                    },
                },
                { id: 'retroQONside', label: 'QON · 顶栏主光', sub: '俯仰 θi · 偏航 = 光源方位', render: (c, S) => D1.sphere(c, D1.M.qon, S) },
            ],
            notes: '剖面图把四个模型放在一起（EON 用 r = σ/(π/2)，Burley 用 r）：Lambert 是半圆形的 cosθ；QON / EON 边缘不归零；Burley 在 r 大时边缘先升后降（下一组步骤）。拖 <code>σ</code> = 0 时 QON 曲线与 Lambert 重合。',
        },
        {
            id: 'onenergy', title: 'Oren-Nayar · 能量：QON 为什么过不了白炉', tag: 'ENERGY',
            desc: 'QON 是 V 腔<strong>单次散射</strong>的拟合，不含槽间多次反射，而且 A 把正面整体压暗：ρ = 1 时方向反照率明显低于 1，σ 越大丢得越多；掠射时 B 项又会把 E 拉回来。直接后果：粗糙漫反射材质整体偏暗、偏"脏"，和 Lambert 材质并排时亮度对不上。Fujii 把完整 ON 的二次反射近似成 A 中额外的 <code>+0.17ρ·σ²/(σ²+0.13)</code>，QON 把它省掉了。',
            formula: [
                'E_QON(ωo) = ∫ f_QON cosθi dωi < 1   (ρ = 1, σ > 0)   // 数值积分',
                'θo = 0：E = A + B·∫ (s/t) cosθi dωi / π，而 s/t 在 θo = 0 时 ≡ 0  ⇒  E(0) = A',
            ],
            previews: [
                {
                    id: 'qonE', label: 'E(θo) · QON', sub: 'ρ = 1；虚线 = 固定 σ 参照', size: [360, 190],
                    render: (c, S) => D1.plotE(c, [
                        { tab: S.Et.lambert, color: '#9aa4ae', label: 'Lambert' },
                        ...S.Efix.qon.map((q, i) => ({ tab: q.tab, color: ['#7a5a3a', '#a07040', '#c08850'][i], dash: [4, 3], label: `QON σ=${q.d}°` })),
                        { tab: S.Et.qon, color: '#e0a060', width: 2.2, label: `QON σ=${D1.deg(S.sigma)}（当前）` },
                    ], { y: [0.4, 1.3] }),
                },
                {
                    id: 'qonFurnace', label: '白炉偏差 · QON', sub: 'E(ωo) − 1：蓝 = 丢能量，橙 = 多出来', render: (c, S) => D1.furnace(c, S.Et.qon, { mode: 'dev' }),
                    probe: (x, y, S) => { const u = 2 * (x + 0.5) / 112 - 1, v = 2 * (y + 0.5) / 112 - 1, r2 = u * u + v * v; return r2 > 1 ? '环境 = 1' : `E = ${D1.lookupE(S.Et.qon, Math.sqrt(1 - r2)).toFixed(3)}`; },
                },
            ],
            notes: '读数：σ = 45° 时 θo = 0 的 E 就是 A ≈ 0.67 —— 正面直接丢掉三分之一。白炉偏差图：球心偏蓝（丢能量），边缘回到接近 0。',
        },
        {
            id: 'eon', title: 'Oren-Nayar · 后续修正：Fujii FON → EON', tag: 'FIX',
            desc: 'Fujii（2013）做了两处小改：① s ≤ 0（ωo 在光的对侧）时不再截断为 0，保留负的 s，前向一侧也随粗糙度变暗，函数更连续；② 用 r ∈ [0,1] 重写 A、B（B = r·A），使方向反照率在 θo → 90° 时趋于 1。但 FON 仍是单次散射，高粗糙度时损失约五分之一能量。<strong>EON</strong>（Portsmouth, Kutz, Hill, JCGT 2025）在 FON 上加一个多次散射 lobe <code>f_ms</code>：形状由 1 − Ê<sub>F</sub>(μi) 与 1 − Ê<sub>F</sub>(μo) 决定，幅度 ρ<sub>ms</sub> 由 albedo 的几何级数得到 —— ρ = 1 时严格通过白炉；并给出解析的方向反照率和 CLTC 重要性采样。OpenPBR Surface 规范选用 EON 作为粗糙漫反射模型。',
            formula: [
                'f_FON = (ρ/π) · A_F · (1 + r · s/t)    t = s > 0 ? max(cosθi, cosθo) : 1    A_F = 1 / (1 + (1/2 − 2/(3π)) r)',
                'Ê_F(μ) = A_F + (r A_F / π) · G(θ)    G(θ) = sinθ (θ − sinθ cosθ) + (2/3) (tanθ (1 − sin^3θ) − sinθ)',
                '⟨Ê_F⟩ = A_F · (1 + (2/3 − 28/(15π)) r)   // 半球平均（cos 加权）',
                'f_ms = (ρ_ms/π) · (1 − Ê_F(μi)) (1 − Ê_F(μo)) / (1 − ⟨Ê_F⟩)    ρ_ms = ρ^2 ⟨Ê_F⟩ / (1 − ρ (1 − ⟨Ê_F⟩))',
                'f_EON = f_FON + f_ms   ⇒   E_EON(μ) = ρ Ê_F(μ) + ρ_ms (1 − Ê_F(μ))   // ρ = 1 ⇒ ≡ 1',
            ],
            code: { title: 'HLSL · Fd_EON / E_FON / E_EON', src: HLSL_EON },
            previews: [
                {
                    id: 'eonLobe', label: '入射平面 lobe', sub: 'k = π f/ρ（EON 用当前 albedo 亮度）', size: [240, 150],
                    render: (c, S) => BX.lobe(c, { wi: S.wi, series: [
                        { fn: () => 1, color: '#9aa4ae', label: 'Lambert' },
                        { fn: (wo) => D1.M.qon.k(S.wi, wo, S, 1), color: '#e0a060', label: 'QON' },
                        { fn: (wo) => D1.M.fon.k(S.wi, wo, S, 1), color: '#c678dd', label: 'FON' },
                        { fn: (wo) => D1.M.eon.k(S.wi, wo, S, S.rhoL), color: '#5ad07a', label: 'EON' },
                    ] }),
                },
                {
                    id: 'eonE', label: 'E(θo) · QON / FON / EON', sub: 'ρ = 1；实线 = 数值积分', size: [360, 190],
                    render: (c, S) => D1.plotE(c, [
                        { tab: S.Et.lambert, color: '#9aa4ae', label: 'Lambert' },
                        { tab: S.Et.qon, color: '#e0a060', label: `QON σ=${D1.deg(S.sigma)}` },
                        { tab: S.Et.fon, color: '#c678dd', label: `FON r=${S.rF.toFixed(2)}` },
                        { tab: D1.thetaDeg.map((d) => D1.EFON(Math.cos(d * Math.PI / 180), S.rF)), color: '#c678dd', dash: [2, 3], label: 'Fujii 解析 Ê_F' },
                        { tab: S.Et.eon, color: '#5ad07a', width: 2.2, label: `EON r=${S.rF.toFixed(2)}` },
                    ], { y: [0.4, 1.3] }),
                },
                { id: 'fonFurnace', label: '白炉偏差 · FON', sub: '单次散射仍丢能量', render: (c, S) => D1.furnace(c, S.Et.fon, { mode: 'dev' }) },
                { id: 'eonFurnace', label: '白炉偏差 · EON', sub: 'f_ms 补回 → 球消失', render: (c, S) => D1.furnace(c, S.Et.eon, { mode: 'dev' }) },
            ],
            notes: '核对：FON 的数值积分与 Fujii 的解析 Ê<sub>F</sub> 应当重合（紫色实线 vs 点线）。lobe 里 FON 在镜像侧低于 QON（保留了负的 s），EON 在 FON 上整体抬高一层平滑的 f_ms。本页按 EON 论文 Fig. 2 的比较方式取 r = σ/(π/2)（σ = 45° ↔ r = 0.5），两者不是物理等价。',
            warn: 'f_ms 对 ρ 非线性（ρ²/(1 − ρ(…))）：RGB 必须逐通道算，不能先按标量算好再乘 albedo。Ê<sub>F</sub>、⟨Ê<sub>F</sub>⟩ 都是解析式（论文另给了 4 次多项式近似），不需要 LUT；E_FON 在 μ → 0 时有 Si/μ，HLSL 里先 clamp μ。',
        },
        // ───────────────────────── Burley ─────────────────────────
        {
            id: 'burley', title: 'Burley · F_D90 = 0.5 + 2r·cos²θd', tag: 'MODEL',
            desc: 'Burley（Disney 2012）从 MERL 实测数据出发：真实漫反射在掠射时要么变暗（光滑表面：光进出界面各有一次 Fresnel 损失），要么出现回射亮边（粗糙表面）。他用两个 Schlick 形状的因子分别作用在入射和出射方向，掠射极限 <code>F_D90</code> 随粗糙度和 θd 变化：r = 0 时 F_D90 = 0.5（掠射压暗一半）；r = 1、θd = 0（回射配置）时 F_D90 = 2.5。这是经验模型，不是从微表面推导出来的。',
            formula: [
                'f_d = (ρ/π) · (1 + (F_D90 − 1)(1 − cosθl)^5) · (1 + (F_D90 − 1)(1 − cosθv)^5)',
                'F_D90 = 0.5 + 2 · r · cos^2θd   // θd = ∠(l, h)，h = normalize(l + v)；r = 感知粗糙度（不是 α）',
                'cos^2θd = (1 + l·v) / 2   // 不必真的算 h',
            ],
            code: { title: 'HLSL · Fd_Burley', src: HLSL_BURLEY },
            previews: [
                {
                    id: 'burleyF', label: '单方向因子 1 + (F_D90 − 1)(1 − μ)^5', sub: 'F_D90 = 0.5 / 1 / 2.5 与当前 r 的回射值', size: [300, 180],
                    render: (c, S) => {
                        const f = (F90) => (mu) => D1.schlick1(F90, mu), cur = 0.5 + 2 * S.rough;
                        PL.draw.plot(c, {
                            x: [0, 1], y: [0, 2.6], xLabel: 'μ = cosθ',
                            series: [
                                { fn: f(0.5), color: '#3a78c8', label: 'F_D90 = 0.5（r = 0）' },
                                { fn: f(1), color: '#9aa4ae', label: 'F_D90 = 1（= Lambert）' },
                                { fn: f(2.5), color: '#e06c75', label: 'F_D90 = 2.5（r = 1, θd = 0）' },
                                { fn: f(cur), color: '#e8c25a', dash: [4, 3], label: `当前 r，θd = 0：${cur.toFixed(2)}` },
                            ],
                        });
                    },
                },
                {
                    id: 'burleyLobe', label: '入射平面 lobe', sub: 'k = π f/ρ', size: [240, 150],
                    render: (c, S) => BX.lobe(c, { wi: S.wi, series: [{ fn: () => 1, color: '#9aa4ae', label: 'Lambert' }, { fn: (wo) => D1.M.burley.k(S.wi, wo, S, 1), color: '#6ab0e0', label: `Burley r=${S.rough.toFixed(2)}` }] }),
                },
                { id: 'burleyMap', label: 'log₂ k(ωo)，ωi 固定', sub: '黄圈 = ωi', render: (c, S) => D1.ratioMap(c, S.hmBurley, S), probe: (x, y, S) => D1.hemiProbe(S.hmBurley)(x, y) },
                {
                    id: 'burleyLobe3d', needs: ['lobe3d'], large: true, label: '3D：Burley 与 Lambert 的 lobe（整个 ωo 半球）',
                    sub: '实体 = Burley（当前 r），灰色线框 = Lambert；半径与归一化同 Step 05。k 模式下看贴近地平线的一圈：光源一侧被 F<sub>D90</sub> 抬高，镜像一侧被压低；切到 k·cosθo，这圈差别乘上 cosθo 后几乎消失——这也是 Burley 在普通光照下不显眼的原因',
                    render: (c, S) => drawLobe3d(c, S, 'burley'),
                },
            ],
            notes: '拖 <code>r</code>：r ≈ 0.25 时回射配置下 F_D90 = 1，正好退化为 Lambert；更小则掠射变暗，更大则掠射变亮。热图里 Burley 的亮区集中在<strong>边缘一圈</strong>（两个 (1 − μ)^5 因子），并偏向光源一侧（θd 小）—— 与 QON 的回射瓣形状不同。',
            warn: '这里的 r 是 Burley 的感知粗糙度，与 GGX 的 α = r² 不同；把 α 喂给 F_D90 会让中等粗糙度的回射明显偏弱。镜面项与漫反射用同一个 r 是 Disney 的设计选择，不是物理约束。',
        },
        {
            id: 'graze', title: 'Burley · 掠射：光滑变暗、粗糙回射变亮', tag: 'GRAZE',
            desc: '在球上看最直观：r = 0 时球的轮廓附近比 Lambert 暗（(1 − μ)^5 把 F_D90 = 0.5 的压暗集中在掠射）；r 大时轮廓附近出现亮边，光越靠近视线（相位角小）越明显，因为 θd 小、F_D90 大。比值图逐像素给出 Burley / Lambert。',
            formula: 'k_Burley = F(μl) · F(μv)，F(μ) = 1 + (F_D90 − 1)(1 − μ)^5\n边缘 μv → 0：k → F_D90 · F(μl)   // r = 0 → ≤ 0.5；r = 1、顶光 → 2.5 · F(μl)',
            previews: [
                { id: 'grazeR0', label: 'Burley r = 0', sub: '主光：俯仰 θi · 偏航 = 方位', render: (c, S) => D1.sphere(c, D1.M.burley, Object.assign({}, S, { rough: 0 })) },
                { id: 'grazeR', label: 'Burley r = 当前', sub: '主光：俯仰 θi · 偏航 = 方位', render: (c, S) => D1.sphere(c, D1.M.burley, S) },
                { id: 'grazeL', label: 'Lambert', sub: '同一光照', render: (c, S) => D1.sphere(c, D1.M.lambert, S) },
                { id: 'grazeRatio0', label: 'log₂(Burley / Lambert) · r = 0', sub: '球面逐像素，范围 ±1；黑 = 背光', render: (c, S) => sphereRatio(c, S, 0) },
                { id: 'grazeRatio', label: 'log₂(Burley / Lambert) · r = 当前', sub: '橙 = 比 Lambert 亮', render: (c, S) => sphereRatio(c, S, S.rough) },
            ],
            notes: '两张比值图对照着看：r = 0 时整个轮廓一圈偏蓝 —— 光滑表面在掠射处只剩不到一半的漫反射；r 大时轮廓（尤其是受光一侧的边缘和明暗交界线附近）偏橙。把 <code>θi</code> 拖到 0（顶光）再把 <code>r</code> 拖到 1：轮廓亮环最强，因为此时处处 θd = 0、F_D90 = 2.5。',
        },
        {
            id: 'benergy', title: 'Burley · 能量增益', tag: 'ENERGY',
            desc: 'Burley 的 f_d 没有做能量归一：r 大时掠射回射项让 E(ωo) 超过 1（凭空产生能量），r = 0 时又处处略低于 1。Frostbite 讲义指出这是 Burley 有意为之 —— 让艺术家调粗糙度时漫反射颜色保持一致 —— 但对于要求"总反照率不超过 1"的管线，这是需要处理的问题。',
            formula: 'E_Burley(ωo) = ∫ f_d cosθi dωi   // ρ = 1，数值积分\nθo → 90°：E → F_D90 的平均 > 1   (r 大)',
            previews: [
                {
                    id: 'burleyE', label: 'E(θo) · Burley', sub: 'ρ = 1；虚线 = r = 0 / 0.5 / 1', size: [360, 190],
                    render: (c, S) => D1.plotE(c, [
                        { tab: S.Et.lambert, color: '#9aa4ae', label: 'Lambert' },
                        ...S.Efix.burley.map((q, i) => ({ tab: q.tab, color: ['#2a5a90', '#4a80b8', '#7ab0e0'][i], dash: [4, 3], label: `Burley r=${q.r}` })),
                        { tab: S.Et.burley, color: '#6ab0e0', width: 2.2, label: `Burley r=${S.rough.toFixed(2)}（当前）` },
                    ], { y: [0.6, 1.6] }),
                },
                {
                    id: 'burleyFurnace', label: '白炉偏差 · Burley', sub: 'E(ωo) − 1：橙 = 多出来的能量', render: (c, S) => D1.furnace(c, S.Et.burley, { mode: 'dev' }),
                    probe: (x, y, S) => { const u = 2 * (x + 0.5) / 112 - 1, v = 2 * (y + 0.5) / 112 - 1, r2 = u * u + v * v; return r2 > 1 ? '环境 = 1' : `E = ${D1.lookupE(S.Et.burley, Math.sqrt(1 - r2)).toFixed(3)}`; },
                },
            ],
            notes: '白炉偏差图：r = 0.7 时边缘一圈偏橙（掠射能量增益），中心接近 0。拖 <code>r</code> 到 0：整球变成浅蓝。',
        },
        {
            id: 'frost', title: 'Burley · Frostbite 重新归一化（Lagarde & de Rousiers 2014）', tag: 'RENORM',
            desc: 'Frostbite 的做法是两处线性修正：F_D90 的常数项从 0.5 改为随 r 由 0 增到 0.5（<code>energyBias</code>），整体再乘 <code>energyFactor = lerp(1, 1/1.51, r)</code> 压掉粗糙端的增益。讲义的目标是让<strong>漫反射 + 镜面</strong>的总方向反照率不超过 1（Frostbite 只保证"不多出能量"），而不是让漫反射单独过白炉 —— 所以下图里它单独时整体低于 1：满足"不多出能量"，但 r 大时漫反射本身也确实少了约三成。',
            formula: [
                'energyBias = lerp(0, 0.5, r)        energyFactor = lerp(1, 1/1.51, r)',
                'F_D90 = energyBias + 2 · r · cos^2θd',
                'f_d = (ρ/π) · F(μl) · F(μv) · energyFactor   // r = 0：F_D90 = 0，掠射处漫反射归零',
            ],
            code: { title: 'HLSL · Fr_DisneyDiffuse（Frostbite）', src: HLSL_FROST },
            previews: [
                {
                    id: 'frostE', label: 'E(θo) · Burley vs Frostbite', sub: 'ρ = 1；虚线 = Frostbite r = 0 / 0.5 / 1', size: [360, 190],
                    render: (c, S) => D1.plotE(c, [
                        { tab: S.Et.lambert, color: '#9aa4ae', label: 'Lambert' },
                        ...S.Efix.frost.map((q, i) => ({ tab: q.tab, color: ['#803040', '#b04858', '#e07080'][i], dash: [4, 3], label: `Frostbite r=${q.r}` })),
                        { tab: S.Et.burley, color: '#6ab0e0', label: `Burley r=${S.rough.toFixed(2)}` },
                        { tab: S.Et.frost, color: '#e06c75', width: 2.2, label: `Frostbite r=${S.rough.toFixed(2)}` },
                    ], { y: [0, 1.6] }),
                },
                {
                    id: 'lutB', label: 'E(NoV, r) − 1 · Burley', sub: 'x = NoV →，y = r ↑；范围 ±0.4',
                    render: (c, S) => drawLut(c, S.lutB), probe: (x, y, S) => lutProbe(S.lutB, x, y),
                },
                {
                    id: 'lutF', label: 'E(NoV, r) − 1 · Frostbite', sub: '= CS_BakeDiffuseDFG 的输出（减 1 显示）',
                    render: (c, S) => drawLut(c, S.lutF), probe: (x, y, S) => lutProbe(S.lutF, x, y),
                },
                { id: 'frostFurnace', label: '白炉偏差 · Frostbite', sub: '当前 r', render: (c, S) => D1.furnace(c, S.Et.frost, { mode: 'dev' }) },
            ],
            notes: '两张 LUT 是 32×32 的方向反照率（cosine 采样 64 点，与 Graph 里的 <code>CS_BakeDiffuseDFG</code> 相同）：Burley 的右上角（高 r、NoV 大）其实接近 1，橙色集中在左上（高 r、掠射）；Frostbite 把整张表压到 ≤ 1：顶部（r → 1）被 energyFactor = 1/1.51 整体压低，左下角（r → 0 的掠射，F_D90 → 0）偏差最大（亮蓝）。',
            warn: 'Frostbite 函数里的 <code>linearRoughness</code> 就是感知粗糙度 r（镜面项另用 α = r²）。1/1.51 是经验常数：与不同的镜面模型或多次散射补偿搭配时，总能量要重新在白炉里检查。',
        },
        // ───────────────────────── Production ─────────────────────────
        {
            id: 'prod', title: '教学版 → 生产版 · 选型与常见坑', tag: 'PROD',
            desc: '前面每一步只看一个方向光下的 f。落到引擎里，漫反射函数被内联进延迟光照 pass，和镜面项、环境光一起求值；环境漫反射还需要 E(NoV, r) 预积分（或解析式）。下面是从教学版到生产版的具体改动，以及每个模型在主流管线里的位置。',
            blocks: [
                { type: 'formula', label: '(A) 选型（已公开的做法）', text: 'UE4（Karis 2013）：Lambert —— 评估过 Burley，差别小、不值得额外开销\nFrostbite（Lagarde & de Rousiers 2014）：重新归一化的 Disney diffuse\nFilament：默认 Lambert，Burley 作为可选项\nOpenPBR Surface：EON（能量守恒的 Oren-Nayar），diffuse roughness 与 specular roughness 分开' },
                { type: 'formula', label: '(B) 求值改动', text: 'cos^2θd = (1 + L·V) / 2              // 省掉 H 的归一化\nNoV = abs(dot(N, V)) + 1e−5           // 法线贴图导致的背面 NoV\ndiffuseColor = baseColor · (1 − metallic)\nL_amb = diffuseColor · E_d(NoV, r) · E_SH(N) / π   // E_d：LUT（Burley / Frostbite）或解析式（EON），Lambert ≡ 1' },
                { type: 'code', title: 'HLSL · CS_DeferredLighting（漫反射部分）', open: true, src: HLSL_LIGHTING },
                { type: 'code', title: 'HLSL · CS_BakeDiffuseDFG', src: HLSL_BAKE },
                {
                    type: 'previews', items: [
                        { id: 'prodLambert', label: 'Lambert', sub: 'studio 三灯', render: (c, S) => D1.sphere(c, D1.M.lambert, S, { lights: BX.LIGHTS.studio }) },
                        { id: 'prodQON', label: 'QON', sub: `σ 当前`, render: (c, S) => D1.sphere(c, D1.M.qon, S, { lights: BX.LIGHTS.studio }) },
                        { id: 'prodEON', label: 'EON', sub: 'r = σ/(π/2)', render: (c, S) => D1.sphere(c, D1.M.eon, S, { lights: BX.LIGHTS.studio }) },
                        { id: 'prodBurley', label: 'Burley', sub: 'r 当前', render: (c, S) => D1.sphere(c, D1.M.burley, S, { lights: BX.LIGHTS.studio }) },
                        { id: 'prodFrost', label: 'Frostbite', sub: 'r 当前', render: (c, S) => D1.sphere(c, D1.M.frost, S, { lights: BX.LIGHTS.studio }) },
                    ],
                },
                { type: 'note', html: '五个球同一套 studio 灯光（主光左上、轮廓光右后、补光右下）、同一 albedo。QON 比 Lambert 暗、明暗过渡更平（白炉丢能量 + 回射），EON 把丢掉的能量补回；Burley 在轮廓光一侧边缘更亮，Frostbite 整体略暗。漫反射与镜面的能量分配（直接相加、乘 1 − F、或分层模型）放在 02 / 06 模块。' },
                {
                    type: 'warn', html: '<strong>常见坑</strong>：① albedo 贴图没从 sRGB 转线性（整体偏亮、饱和度低）；② 1/π 与光源单位双重计入或漏掉（差 π 倍）；③ 金属仍保留漫反射；④ 把 GGX 的 α 当成 Burley / Frostbite 的 r，或把 [0,1] 粗糙度直接当 ON 的 σ（弧度）；⑤ QON 的 tanβ 在掠射发散 —— 用 <code>s / max(NoL, NoV)</code>；⑥ EON 的 f_ms 没有逐通道算；⑦ 环境光用 SH 辐照度 × albedo，却在直接光里用 Burley / ON —— 两边亮度对不上，要乘 E_d(NoV, r)；⑧ 只看单个灯不跑白炉：QON 偏暗、Burley 偏亮的问题在普通光照下很难看出来。',
                },
            ],
        },
    ],

    library: {
        title: '附录 · 辅助函数（HLSL）',
        intro: 'Step 13 的 <code>CS_DeferredLighting</code> 调用、正文没有展开的两部分：G-buffer 解码（布局 = Graph 里 BasePass 节点输出的 GBufferA/B/C，再加一张深度）与天空光的 L2 SH 辐照度。都是引擎里很常见的写法，换成自己引擎的布局即可。',
        blocks: [
            { title: 'G-buffer 布局 · DecodeGBuffer · OctDecode', code: HLSL_GBUFFER },
            { title: 'EvalSHIrradiance · L2 SH 辐照度（Ramamoorthi & Hanrahan 2001）', code: HLSL_SH },
        ],
    },

    graph: {
        intro: '<strong>RDG 风格</strong>：BxDF 本身不是 pass，而是被内联（<code>INLINE</code>）进真实 pass 的函数 —— 延迟光照 CS 逐像素调用漫反射函数，LUT 烘焙 CS 用 cosine 采样对同一个函数做预积分。同名资源自动连线；函数 pin 的"格式"写返回类型。',
        types: {
            gbuf: { color: '#4a8aff', label: 'gbuffer · 屏幕空间纹理' },
            scalar: { color: '#e8c25a', label: 'scalar · cbuffer / 结构化缓冲' },
            fn: { color: '#a45fa5', label: 'fn · 内联 BxDF 函数' },
            lut: { color: '#e8853a', label: 'lut · 预积分查找表' },
            output: { color: '#5ae87a', label: 'output · HDR 颜色' },
        },
        common: {
            title: 'Common.hlsl · 系列共享 + 本模块漫反射函数', open: false,
            blocks: [
                { label: '━ 系列共享 ━', code: BX.HLSL_COMMON },
                { label: '━ 本模块 · 辅助 ━', code: HLSL_HELPERS },
                { label: '━ Lambert ━', code: HLSL_LAMBERT },
                { label: '━ Cosine 采样 ━', code: HLSL_SAMPLE },
                { label: '━ Oren-Nayar QON ━', code: HLSL_QON },
                { label: '━ FON / EON ━', code: HLSL_EON },
                { label: '━ Burley ━', code: HLSL_BURLEY },
                { label: '━ Frostbite ━', code: HLSL_FROST },
            ],
        },
        nodes: [
            {
                id: 'gbuffer', title: 'BasePass · GBuffer', tag: 'PS', category: 'system', lifetime: 'frame', desc: 'baseColor · N · r · model',
                dispatch: { kind: 'PS', name: 'BasePassPS', target: 'GBufferA/B/C（MRT）', idRange: 'SV_Position.xy ∈ viewport', note: '材质图写出 albedo、法线、粗糙度、漫反射模型 id' },
                outputs: [{ name: 'GBufferA', format: 'RGBA8_SRGB', type: 'gbuf' }, { name: 'GBufferB', format: 'RGB10A2', type: 'gbuf' }, { name: 'GBufferC', format: 'RGBA8', type: 'gbuf' }],
                detail: { desc: '引擎 BasePass 产出。albedo 用 sRGB 格式存储，采样时硬件解码为线性。', preview: null },
            },
            {
                id: 'lights', title: 'LightBuffer · SkySH', tag: 'CB', category: 'system', lifetime: 'frame', desc: '方向光 E · 天空 SH 辐照度',
                outputs: [{ name: 'LightBuffer', format: 'StructuredBuffer<Light>', type: 'scalar' }, { name: 'SkyIrradianceSH', format: 'SH L2 float3×9', type: 'scalar' }],
                detail: { desc: '每帧由引擎上传。方向光的 E 是法向入射照度；SH 给出 E_SH(N) = ∫ L cosθ dω。', preview: null },
            },
            {
                id: 'fdLambert', title: 'Fd_Lambert', tag: 'INLINE', step: 'lambert', desc: 'ρ / π',
                dispatch: { kind: 'INLINE', name: 'Fd_Lambert', note: '被 CS_DeferredLighting 内联' },
                outputs: [{ name: 'Fd_Lambert()', format: 'float3', type: 'fn' }],
                detail: { preview: 'lambertSphere' },
            },
            {
                id: 'fdON', title: 'Fd_OrenNayar · QON / EON', tag: 'INLINE', step: 'eon', desc: 'FON + f_ms（解析 Ê_F）',
                dispatch: { kind: 'INLINE', name: 'Fd_EON', note: '生产用 EON；QON 仅作对照' },
                outputs: [{ name: 'Fd_EON()', format: 'float3', type: 'fn' }],
                detail: {
                    conversion: 's = L·V − NoL·NoV   // 不需要 φ\nr = diffuseRoughness ∈ [0,1]',
                    formula: 'f_EON = f_FON + f_ms\nf_FON = (ρ/π) · A_F · (1 + r · s/t)\nf_ms = (ρ_ms/π) · (1 − Ê_F(μi)) (1 − Ê_F(μo)) / (1 − ⟨Ê_F⟩)   // ρ = 1 时白炉 ≡ 1',
                    preview: 'eonLobe',
                },
            },
            {
                id: 'fdBurley', title: 'Fr_DisneyDiffuse', tag: 'INLINE', step: 'frost', desc: 'Burley × energyBias / energyFactor',
                dispatch: { kind: 'INLINE', name: 'Fr_DisneyDiffuse', note: '被光照 CS 与 LUT 烘焙 CS 共同内联' },
                outputs: [{ name: 'Fr_DisneyDiffuse()', format: 'float', type: 'fn' }],
                detail: { conversion: 'LoH = sqrt(saturate(0.5 + 0.5·L·V))   // cosθd\nlinearRoughness = r（感知粗糙度）', preview: 'frostE' },
            },
            {
                id: 'sampleCos', title: 'SampleCosineHemisphere', tag: 'INLINE', step: 'sample', desc: 'Malley · pdf = cosθ/π',
                dispatch: { kind: 'INLINE', name: 'SampleCosineHemisphere', note: 'LUT 烘焙与白炉检查共用' },
                outputs: [{ name: 'SampleCosine()', format: 'float3 + pdf', type: 'fn' }],
                detail: { preview: 'cosScatter' },
            },
            {
                id: 'bake', title: 'CS_BakeDiffuseDFG', tag: 'CS', category: 'bake', lifetime: 'resident', step: 'frost', desc: 'E_d(NoV, r) · 64 spp cosine',
                dispatch: { name: 'CS_BakeDiffuseDFG', threads: [8, 8, 1], groups: [4, 4, 1], idRange: 'ID.xy ∈ [0,32)² → (NoV, r)', note: '启动时一次；每 thread 64 个 cosine 样本' },
                inputs: [{ name: 'Fr_DisneyDiffuse()', format: 'float', type: 'fn' }, { name: 'SampleCosine()', format: 'float3 + pdf', type: 'fn' }],
                outputs: [{ name: 'DiffuseDFG_LUT', format: 'R16F 32×32', type: 'lut' }],
                detail: { code: { title: 'HLSL · CS_BakeDiffuseDFG', src: HLSL_BAKE }, formula: 'E_d(μ_o, r) = (1/N) Σ_j Fr(μ_o, cosθ_j, cosθ_d, r)   // ω_j ~ cosθ/π', preview: 'lutF' },
            },
            {
                id: 'lighting', title: 'CS_DeferredLighting', tag: 'CS', category: 'output', lifetime: 'frame', step: 'prod', desc: 'Σ_lights f_d·E·NoL + E_d·SH',
                dispatch: { name: 'CS_DeferredLighting', threads: [8, 8, 1], groups: [240, 135, 1], idRange: 'ID.xy ∈ [0, 1920)×[0, 1080)', note: '1920×1080 示例；漫反射模型按 GBufferC 的 model id 分支' },
                inputs: [
                    { name: 'GBufferA', format: 'RGBA8_SRGB', type: 'gbuf' }, { name: 'GBufferB', format: 'RGB10A2', type: 'gbuf' }, { name: 'GBufferC', format: 'RGBA8', type: 'gbuf' },
                    { name: 'LightBuffer', format: 'StructuredBuffer<Light>', type: 'scalar' }, { name: 'SkyIrradianceSH', format: 'SH L2 float3×9', type: 'scalar' },
                    { name: 'DiffuseDFG_LUT', format: 'R16F 32×32', type: 'lut' },
                    { name: 'Fd_Lambert()', format: 'float3', type: 'fn' }, { name: 'Fd_EON()', format: 'float3', type: 'fn' }, { name: 'Fr_DisneyDiffuse()', format: 'float', type: 'fn' },
                ],
                outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'output' }],
                detail: {
                    commonUsed: 'Fd_Lambert  Fd_EON  E_EON  Fr_DisneyDiffuse   // Common',
                    conversion: 'NoV = abs(dot(N, V)) + 1e−5\nNoL = saturate(dot(N, L))\nLoV = dot(L, V)   // → s 与 cos^2θd\ndiffuseColor = baseColor · (1 − metallic)',
                    code: { title: 'HLSL · CS_DeferredLighting', src: HLSL_LIGHTING }, preview: 'prodEON',
                },
            },
        ],
    },

    references: [
        { title: 'Photometria, sive de mensura et gradibus luminis, colorum et umbrae', author: 'J. H. Lambert', venue: 'Augsburg', year: 1760, note: 'Lambert 余弦定律' },
        { title: "Generalization of Lambert's Reflectance Model", author: 'Michael Oren, Shree K. Nayar', venue: 'SIGGRAPH', year: 1994, note: 'V 腔模型、完整模型与定性模型（A、B 系数）' },
        { title: 'A tiny improvement of Oren-Nayar reflectance model', author: 'Yasuhiro Fujii', venue: 'web article', year: 2013, url: 'https://mimosa-pudica.net/improved-oren-nayar.html', note: 's/t 写法、s ≤ 0 时 t = 1、r ∈ [0,1] 的 A、B' },
        { title: 'EON: A Practical Energy-Preserving Rough Diffuse BRDF', author: 'Jamie Portsmouth, Peter Kutz, Stephen Hill', venue: 'Journal of Computer Graphics Techniques 14(1)', year: 2025, url: 'https://jcgt.org/published/0014/01/06/', note: 'FON 解析反照率、多次散射 lobe、CLTC 采样；本页 HLSL 由其 GLSL Listing 1 改写' },
        { title: 'Physically-Based Shading at Disney', author: 'Brent Burley', venue: 'SIGGRAPH Course: Practical Physically Based Shading in Film and Game Production', year: 2012, note: 'Disney diffuse 与 F_D90' },
        { title: 'Moving Frostbite to Physically Based Rendering 3.0', author: 'Sébastien Lagarde, Charles de Rousiers', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2014, note: 'Disney diffuse 的能量分析与重新归一化（Listing 1）' },
        { title: 'Real Shading in Unreal Engine 4', author: 'Brian Karis', venue: 'SIGGRAPH Course: Physically Based Shading in Theory and Practice', year: 2013, note: 'UE4 选用 Lambert 漫反射的理由' },
        { title: 'Physically Based Rendering in Filament', author: 'Romain Guy, Mathias Agopian', venue: 'Google', year: 2018, url: 'https://google.github.io/filament/Filament.html', note: '漫反射 BRDF 的取舍' },
        { title: 'OpenPBR Surface Specification', author: 'Andersson et al.', venue: 'Academy Software Foundation', year: 2024, note: '采用 EON 作为粗糙漫反射' },
        { title: 'Physically Based Rendering: From Theory to Implementation (4th ed.)', author: 'Matt Pharr, Wenzel Jakob, Greg Humphreys', venue: 'MIT Press', year: 2023, note: 'Malley 方法 / cosine 采样' },
        { title: 'An Efficient Representation for Irradiance Environment Maps', author: 'Ravi Ramamoorthi, Pat Hanrahan', venue: 'SIGGRAPH', year: 2001, note: 'L2 SH 辐照度（附录 EvalSHIrradiance）' },
        { title: 'A Survey of Efficient Representations for Independent Unit Vectors', author: 'Zina H. Cigolle, Sam Donow, Daniel Evangelakos, Michael Mara, Morgan McGuire, Quirin Meyer', venue: 'Journal of Computer Graphics Techniques 3(2)', year: 2014, url: 'https://jcgt.org/published/0003/02/01/', note: '八面体法线编码（附录 OctDecode）' },
    ],
});

// ── 采样散点：背景 = 目标密度（等面积投影），点 = 实际样本 ──
function drawScatter(c, S, pts, isCos) {
    const { ctx, w, h } = PL.draw.ctx(c), R = D1.RES_H, s = Math.min(w, h) - 8, x0 = (w - s) / 2, y0 = (h - s) / 2;
    const off = drawScatter._off || (drawScatter._off = document.createElement('canvas'));
    const img = PL.buf(drawScatter, isCos ? '_pc' : '_pu', R * R);
    for (let i = 0; i < R * R; i++) img[i] = S.hmCos[i] === S.hmCos[i] ? (isCos ? S.hmCos[i] / Math.PI : 0.5 / Math.PI) : NaN;
    PL.draw.field(off, img, R, R, { map: 'viridis', range: [0, 1 / Math.PI * 1.15] });
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, x0, y0, s, s);
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    const dot = pts.length > 300 ? 1.1 : 1.6;
    for (const p of pts) {
        const [px, py] = BX.hemiPixel(p, R);
        ctx.beginPath(); ctx.arc(x0 + (px + 0.5) / R * s, y0 + (py + 0.5) / R * s, dot, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.font = '10px Consolas, monospace'; ctx.fillStyle = '#9aa4ae'; ctx.textAlign = 'left';
    ctx.fillText(`N = ${pts.length}`, 4, 12);
}
// ── LUT 显示（E − 1，diverge，r 轴朝上） ──
function drawLut(c, lut) {
    const N = 32, img = PL.buf(c, '_lut', N * N);
    for (let i = 0; i < N * N; i++) img[i] = lut[i] - 1;
    PL.draw.field(c, img, N, N, { range: [-0.4, 0.4], map: 'diverge', flipY: true });
}
function lutProbe(lut, x, y) {
    const N = 32, yy = N - 1 - y;
    return `NoV=${((x + 0.5) / N).toFixed(2)} r=${((yy + 0.5) / N).toFixed(2)}  E=${lut[yy * N + x].toFixed(3)}`;
}
// ── 球面逐像素 log₂(Burley / Lambert)：主光俯仰 = θi、偏航 = 光源方位（与 D1.sphere 相同的几何） ──
function sphereRatio(c, S, r) {
    const res = 112, img = PL.buf(c, '_ratio', res * res), L = D1.lightDir(S.thetaI, S);
    for (let y = 0; y < res; y++) for (let x = 0; x < res; x++) {
        const u = 2 * (x + 0.5) / res - 1, v = -(2 * (y + 0.5) / res - 1), r2 = u * u + v * v;
        if (r2 > 1) { img[y * res + x] = NaN; continue; }
        const n = [u, v, Math.sqrt(1 - r2)], t = BX.norm(BX.cross(Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], n)), b = BX.cross(n, t);
        const loc = (w) => [BX.dot(w, t), BX.dot(w, b), BX.dot(w, n)], wi = loc(L), wo = loc([0, 0, 1]);
        img[y * res + x] = wi[2] > 0 ? Math.log2(Math.max(1e-3, D1.kBurley(wi, wo, r))) : NaN;
    }
    PL.draw.field(c, img, res, res, { range: [-1, 1], map: 'diverge' });
}
// ── 3D：某个模型的 k·cosθo lobe（实体，读 lobe3d 网格）+ Lambert 参照（线框，解析） ──
function drawLobe3d(c, S, key) {
    const { nu, nv, W } = L3D, NU = nu + 1, m = D1.M[key], L = S.l3d[key], R = L.R, withCos = S.l3dMode === 'kcos';
    const SC = L3D.scale / Math.max(1, L.max);                          // 同一比例尺：模型与 Lambert（峰值 1）中较大者 → 半径 1.3
    const at = (d, r) => W([d[0] * r, d[1] * r, d[2] * r]);
    const pos = (u, v) => { const i = Math.round(u * nu), j = Math.round(v * nv); return at(L3D.dir(i, j), SC * R[j * NU + i]); };
    const lam = (u, v) => { const d = BX.dir(v * Math.PI / 2, u * 2 * Math.PI); return at(d, SC * (withCos ? d[2] : 1)); };
    const wi = S.wi, mir = [-wi[0], -wi[1], wi[2]];
    // 标签：模型名放在光源一侧 θ = 25° 的瓣面上方；Lambert 放在镜像一侧 θ = 35° 的线框外（两种模型在那里都低于 Lambert）
    const jM = Math.round(nv * 25 / 90), up = L3D.dir(0, jM), back = BX.dir(35 * Math.PI / 180, Math.PI);
    PL.draw.mesh3d(c, {
        meshes: [
            PL.mesh.param(pos, [nu, nv], { closedU: true, rgb: L3D.rgb(m.color), gloss: 18 }),
            PL.mesh.param(lam, [24, 8], { closedU: true, surface: false, wire: true, wireColor: 'rgba(190,200,210,0.5)' }),
        ],
        lines: [
            { pts: [[0, 0, 0], at([0, 0, 1], 1.7)], color: '#9aa4ae', width: 1.4, arrow: true, label: 'n' },
            { pts: [[0, 0, 0], at(wi, 1.7)], color: '#e8c25a', width: 2.4, arrow: true, label: 'ωi' },
            { pts: [[0, 0, 0], at(mir, 1.5)], color: '#56d0e0', width: 1.2, dash: [4, 3], label: '镜像' },
        ],
        labels: [
            { p: at(up, SC * R[jM * NU] + 0.3), text: key === 'qon' ? `QON σ = ${D1.deg(S.sigma)}` : `Burley r = ${S.rough.toFixed(2)}`, color: m.color },
            { p: at(back, SC * (withCos ? back[2] : 1) + 0.12), text: 'Lambert', color: '#b8c2cc' },
        ],
        grid: { y: 0, size: 2.4 },
        orbit: { yaw: 0.5, pitch: 0.42, dist: 5, target: [0.2, 0.5, 0] },
    });
}
