// BxDF series · index page. Build:
//   node <skill>/scripts/build_doc.mjs bxdf_common.js index.lab.js -o index.html
// Small local versions of each family's headline BxDF drive the comparison previews; the full
// step-by-step derivations live in the six modules.

const IDX = (() => {
    const INV_PI = BX.INV_PI, z3 = [0, 0, 0];
    const lambert = (alb) => () => alb.map((c) => c * INV_PI);
    // Oren-Nayar 1994 qualitative model, σ in radians
    const orenNayar = (alb, sigma) => {
        const s2 = sigma * sigma, A = 1 - 0.5 * s2 / (s2 + 0.33), B = 0.45 * s2 / (s2 + 0.09);
        return (wi, wo) => {
            const ci = BX.clamp(wi[2], 0, 1), co = BX.clamp(wo[2], 0, 1), si = Math.sqrt(1 - ci * ci), so = Math.sqrt(1 - co * co);
            const cphi = si > 1e-4 && so > 1e-4 ? (wi[0] * wo[0] + wi[1] * wo[1]) / (si * so) : 0;
            const sinA = ci < co ? si : so, tanB = ci < co ? so / Math.max(co, 1e-4) : si / Math.max(ci, 1e-4);
            const k = A + B * Math.max(0, cphi) * sinA * tanB;
            return alb.map((c) => c * INV_PI * k);
        };
    };
    // Burley 2012 diffuse (perceptual roughness r)
    const burley = (alb, r) => (wi, wo) => {
        const cd = BX.dot(wi, BX.half(wi, wo)), fd90 = 0.5 + 2 * r * cd * cd;
        const k = (1 + (fd90 - 1) * BX.pow5(1 - Math.max(0, wi[2]))) * (1 + (fd90 - 1) * BX.pow5(1 - Math.max(0, wo[2])));
        return alb.map((c) => c * INV_PI * k);
    };
    const specular = (r, fresnel) => {
        const a = BX.alpha(r);
        return (wi, wo) => {
            if (wi[2] <= 0 || wo[2] <= 0) return z3;
            const h = BX.half(wi, wo), s = BX.ggxD(h, a) * BX.smithG2(wo, wi, a) / (4 * wi[2] * wo[2]), F = fresnel(BX.dot(wo, h));
            return Array.isArray(F) ? F.map((f) => f * s) : [F * s, F * s, F * s];
        };
    };
    const conductor = (metal, r) => specular(r, (c) => BX.fresnelConductor(c, metal.eta, metal.k));
    // GGX dielectric reflection + Fresnel-weighted Lambert ("plastic")
    const plastic = (alb, r, ior = 1.5) => {
        const sp = specular(r, (c) => BX.fresnelDielectric(c, ior));
        return (wi, wo) => {
            if (wi[2] <= 0 || wo[2] <= 0) return z3;
            const s = sp(wi, wo), kd = (1 - BX.fresnelDielectric(wi[2], ior)) * (1 - BX.fresnelDielectric(wo[2], ior));
            return [alb[0] * INV_PI * kd + s[0], alb[1] * INV_PI * kd + s[1], alb[2] * INV_PI * kd + s[2]];
        };
    };
    // Charlie sheen (Estevez & Kulla 2017) with the Neubelt & Pettineo visibility, over a dark Lambert base
    const velvet = (sheen, base, r) => {
        const a = Math.max(0.07, r * r);
        return (wi, wo) => {
            if (wi[2] <= 0 || wo[2] <= 0) return z3;
            const h = BX.half(wi, wo), sinH = Math.sqrt(Math.max(0, 1 - h[2] * h[2]));
            const D = (2 + 1 / a) * Math.pow(sinH, 1 / a) / (2 * Math.PI), V = 1 / (4 * (wi[2] + wo[2] - wi[2] * wo[2]));
            return [0, 1, 2].map((c) => base[c] * INV_PI + sheen[c] * D * V);
        };
    };
    // clear coat (GGX, η = 1.5) over a base: naive sum vs. (1 − F)(1 − F) energy weighting
    const coated = (base, rCoat, conserve) => {
        const sp = specular(rCoat, (c) => BX.fresnelDielectric(c, 1.5));
        return (wi, wo) => {
            if (wi[2] <= 0 || wo[2] <= 0) return z3;
            const c = sp(wi, wo), b = base(wi, wo), w = conserve ? (1 - BX.fresnelDielectric(wi[2], 1.5)) * (1 - BX.fresnelDielectric(wo[2], 1.5)) : 1;
            return [b[0] * w + c[0], b[1] * w + c[1], b[2] * w + c[2]];
        };
    };
    // Burley normalized diffusion profile, shape d (mm): integrates to 1 over the plane
    const normalizedDiffusion = (r, d) => (Math.exp(-r / d) + Math.exp(-r / (3 * d))) / (8 * Math.PI * d * Math.max(r, 1e-4));
    return { lambert, orenNayar, burley, conductor, plastic, velvet, coated, normalizedDiffusion };
})();

const lightsFor = (S) => {
    const az = S.lightAz * Math.PI / 180, key = BX.norm([Math.sin(az) * 0.8, 0.55, Math.cos(az) * 0.8]);
    return [{ dir: key, color: [2.8, 2.7, 2.55] }, { dir: BX.norm([-key[0], 0.15, -key[2] - 0.2]), color: [1.0, 1.05, 1.2] }, { dir: BX.norm([0.4, -0.4, 0.8]), color: [0.2, 0.22, 0.26] }];
};
const wiOf = (S) => BX.dir(S.thetaI * Math.PI / 180, Math.PI);   // light from the left in the plane of incidence
const GALLERY = [
    ['Lambert', (S) => IDX.lambert(S.base)],
    ['Oren-Nayar', (S) => IDX.orenNayar(S.base, S.rough)],
    ['Burley', (S) => IDX.burley(S.base, S.rough)],
    ['GGX · 金', (S) => IDX.conductor(BX.METALS.gold, S.rough)],
    ['GGX · 塑料', (S) => IDX.plastic(S.base, S.rough)],
    ['Charlie Sheen', (S) => IDX.velvet([0.9, 0.55, 0.7], [0.05, 0.01, 0.03], S.rough)],
    ['Clearcoat', (S) => IDX.coated(IDX.plastic(S.base, 0.6), 0.12, true)],
    ['GGX · 铝 (r/2)', (S) => IDX.conductor(BX.METALS.aluminum, S.rough * 0.5)],
];
const link = (i) => { const m = BX.MODULES[i]; return `<a href="${m.file}">打开模块 ${m.no} · ${m.short} →</a>`; };
const sphere = (make, extra) => (c, S) => BX.renderSphere(c, make(S), Object.assign({ res: 128, lights: lightsFor(S) }, extra || {}));

PipelineLab.mount({
    title: 'BxDF 系列 · 常用 BxDF 算法索引',
    lang: 'zh-CN',
    codeLang: 'hlsl',
    animated: false,
    tagColors: { DIFFUSE: 'green', SPECULAR: 'blue', BTDF: 'teal', SHEEN: 'magenta', BSSRDF: 'rust', LAYER: 'olive' },
    params: [
        { id: 'rough', label: 'roughness', min: 0.02, max: 1, step: 0.01, value: 0.45, title: '感知粗糙度 r（GGX α = r²；Oren-Nayar 这里用 σ = r 弧度）' },
        { id: 'thetaI', label: 'θi', min: 0, max: 85, step: 1, value: 45, unit: '°', title: '入射角（波瓣图与曲线用）' },
        { id: 'lightAz', label: '光源方位', min: -90, max: 90, step: 1, value: -40, unit: '°', title: '球体渲染的主光方位（正 = 右侧）' },
        { id: 'base', label: 'base', value: '#b5553a', map: BX.hex2lin, title: '漫反射 / 底层颜色（sRGB 输入，内部转线性）' },
    ],
    stages: [
        {
            id: 'albedo', lifetime: 'param', run(S) {
                const mus = PL.buf(S, 'mus', 32);
                for (let i = 0; i < 32; i++) mus[i] = 0.03 + 0.97 * i / 31;
                // GGX white furnace (F = 1): E(μ) via VNDF sampling, weight = G2 / G1
                S.ggxE = [0.25, 0.5, 0.75, 1.0].map((r) => {
                    const a = BX.alpha(r), out = new Float32Array(32);
                    for (let i = 0; i < 32; i++) {
                        const wo = [Math.sqrt(1 - mus[i] * mus[i]), 0, mus[i]];
                        out[i] = BX.albedoMC((w, u1, u2) => { const h = BX.sampleVNDF(w, a, a, u1, u2), l = BX.reflect(w, h); return l[2] > 0 ? { wi: l, weight: BX.smithG2(w, l, a) / BX.smithG1(w, a) } : null; }, wo, 512);
                    }
                    return out;
                });
                // coat (η = 1.5, r = 0.2) over a white Lambert base: naive sum vs (1−F)(1−F) weighting
                const a = BX.alpha(0.2), add = PL.buf(S, 'coatAdd', 32), con = PL.buf(S, 'coatCon', 32);
                for (let i = 0; i < 32; i++) {
                    const wo = [Math.sqrt(1 - mus[i] * mus[i]), 0, mus[i]];
                    const eCoat = BX.albedoMC((w, u1, u2) => { const h = BX.sampleVNDF(w, a, a, u1, u2), l = BX.reflect(w, h); return l[2] > 0 ? { wi: l, weight: BX.fresnelDielectric(BX.dot(w, h), 1.5) * BX.smithG2(w, l, a) / BX.smithG1(w, a) } : null; }, wo, 512);
                    const eBase = BX.albedoQuad((wi) => BX.INV_PI * (1 - BX.fresnelDielectric(wi[2], 1.5)) * (1 - BX.fresnelDielectric(wo[2], 1.5)), wo, 48);
                    add[i] = eCoat + 1; con[i] = eCoat + eBase;
                }
            },
        },
    ],
    intro: {
        title: '常用 BxDF 算法 · 系列索引',
        flow: ['漫反射', '微表面镜面', '透射 BTDF', '织物 Sheen', '次表面 BSSRDF', '多层混合'],
        html: `<p>六个模块各自把一类 BxDF 拆成 step-by-step：公式 → HLSL → 页面内 JS 参考实现实时算出的中间结果（球体渲染、波瓣、半球热力图、白炉能量曲线、采样分布）。本页用每类的"代表 BxDF"并排对比，并给出它们在一个 uber 材质里如何组合（下方节点图）。</p>
<p><strong>全系列统一约定</strong>：局部着色坐标系 n = +Z；<code>wi</code> 指向光源、<code>wo</code> 指向观察者（都背离表面）；f 的单位是 1/sr；感知粗糙度 r，GGX α = r²；相对折射率 η = η<sub>t</sub>/η<sub>i</sub>；全程线性 RGB，只有显示时做 ACES + sRGB。</p>
<p>阅读顺序：${BX.MODULES.map((m) => `<a href="${m.file}">${m.no} ${m.short}</a>`).join(' → ')}</p>`,
    },
    sections: [{
        id: 'gallery', title: '代表 BxDF 对比 · 同一光照 / 同一粗糙度', position: 'top',
        sub: '拖顶部的 roughness / 光源方位 / base 颜色，八个球同时重算。全部是真实 BxDF 求值（三盏方向光），不是贴图。',
        html: `<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;max-width:820px">${GALLERY.map(([name], i) => `<div style="text-align:center"><canvas class="g${i}" style="width:100%;aspect-ratio:1"></canvas><div style="font:0.68rem Consolas,monospace;color:#888;margin-top:4px">${name}</div></div>`).join('')}</div>`,
        frame(el, S) { GALLERY.forEach(([, make], i) => BX.renderSphere(el.querySelector('.g' + i), make(S), { res: 144, lights: lightsFor(S) })); },
    }],
    steps: [
        {
            id: 'm01', title: '01 · 漫反射：Lambert → Oren-Nayar → Burley', tag: 'DIFFUSE',
            desc: '从"各方向等亮"的 Lambert 出发：为什么是 ρ/π、怎样做余弦重要性采样；Oren-Nayar 用 V 形微槽解释粗糙表面的<strong>逆反射</strong>（月球"满月不暗边"）；Burley/Disney 漫反射用 F<sub>D90</sub> 在掠射角加亮或压暗，以及 Frostbite 的能量重归一化。',
            formula: ['f_Lambert = ρ / π', 'f_ON = (ρ/π) · (A + B · max(0, cos(φ_i − φ_o)) · sin α · tan β)', 'f_Burley = (ρ/π) · (1 + (F_D90 − 1)(1 − cos θ_l)^5) · (1 + (F_D90 − 1)(1 − cos θ_v)^5)    F_D90 = 0.5 + 2 r cos^2 θ_d'],
            previews: [
                { id: 'p01a', label: 'Lambert', smooth: true, render: sphere((S) => IDX.lambert(S.base)) },
                { id: 'p01b', label: 'Oren-Nayar', sub: 'σ = r', smooth: true, render: sphere((S) => IDX.orenNayar(S.base, S.rough)) },
                { id: 'p01c', label: 'Burley', smooth: true, render: sphere((S) => IDX.burley(S.base, S.rough)) },
                { id: 'p01d', label: '入射面内 f·cosθo', sub: '黄 = 入射方向，虚线 = 镜面方向', size: [240, 150], render: (c, S) => BX.lobe(c, { wi: wiOf(S), cos: true, series: [{ fn: (wo) => IDX.lambert([1, 1, 1])(wiOf(S), wo)[0], color: '#888', label: 'Lambert' }, { fn: (wo) => IDX.orenNayar([1, 1, 1], S.rough)(wiOf(S), wo)[0], color: '#5ad07a', label: 'Oren-Nayar' }, { fn: (wo) => IDX.burley([1, 1, 1], S.rough)(wiOf(S), wo)[0], color: '#e0a060', label: 'Burley' }] }) },
            ],
            notes: `拖 roughness：Oren-Nayar 与 Burley 的波瓣都向光源一侧鼓起（逆反射），Lambert 不变。${link(0)}`,
        },
        {
            id: 'm02', title: '02 · 微表面镜面：Cook-Torrance · GGX', tag: 'SPECULAR',
            desc: '从经验模型 Phong / Blinn-Phong 讲到微表面理论：法线分布 D（GGX 的长尾）、Smith 遮蔽 G、Fresnel F 三项各自的物理意义与归一化；各向异性；NDF 与 VNDF 重要性采样的方差差异；高粗糙度时单次散射丢能量（下图白炉曲线 &lt; 1），用 Kulla-Conty 多次散射补偿找回。',
            formula: 'f_spec = D(h) · G_2(ω_i, ω_o) · F(ω_o·h) / (4 cos θ_i cos θ_o)\nD_GGX = α^2 / (π ((n·h)^2 (α^2 − 1) + 1)^2)      G_2 = 1 / (1 + Λ(ω_o) + Λ(ω_i))',
            previews: [
                { id: 'p02a', label: 'GGX · 金（精确导体 Fresnel）', smooth: true, render: sphere((S) => IDX.conductor(BX.METALS.gold, S.rough)) },
                { id: 'p02b', label: 'GGX · 塑料（η = 1.5）', smooth: true, render: sphere((S) => IDX.plastic(S.base, S.rough)) },
                { id: 'p02c', label: 'GGX 波瓣（对数）', size: [240, 150], render: (c, S) => BX.lobe(c, { wi: wiOf(S), cos: true, log: 3, series: [{ fn: (wo) => IDX.conductor(BX.METALS.silver, S.rough)(wiOf(S), wo)[1], color: '#6ab0e0', label: `GGX r = ${S.rough.toFixed(2)}` }] }) },
                { id: 'p02d', label: '白炉 E(μ)（F = 1）', sub: '单次散射丢失的能量 = 1 − E', size: [300, 150], render: (c, S) => PL.draw.plot(c, { x: [0, 1], y: [0, 1.05], xLabel: 'μ = cosθo', series: [0.25, 0.5, 0.75, 1].map((r, k) => ({ data: S.ggxE[k], x: S.mus, label: `r = ${r}`, color: ['#5ad07a', '#6ab0e0', '#e0a060', '#e06c75'][k] })), hmarks: [{ y: 1, color: '#555' }] }) },
            ],
            notes: link(1),
        },
        {
            id: 'm03', title: '03 · 粗糙透射 BTDF（Walter 2007）', tag: 'BTDF',
            desc: '光滑电介质的 Snell + Fresnel（含全内反射）是起点；粗糙界面用广义折射半向量 h<sub>t</sub> ∝ −(η<sub>i</sub>ω<sub>i</sub> + η<sub>o</sub>ω<sub>o</sub>)、半向量 Jacobian 和 (1 − F)·D·G 组成 BTDF；再讲按 Fresnel 在反射 / 折射间选择的重要性采样、η² 辐亮度缩放，以及 R + T 的白炉能量。下图是精确 Fresnel 的 R / T 随入射角变化。',
            formula: 'f_t = (|ω_i·h| |ω_o·h|) / (|cos θ_i| |cos θ_o|) · η_o^2 (1 − F) D G_2 / (η_i (ω_i·h) + η_o (ω_o·h))^2',
            previews: [{
                id: 'p03a', label: '精确 Fresnel：R 与 T = 1 − R', sub: '实线 = 从空气射入 η=1.5；虚线 = 从玻璃内部射出（≈41.8° 后全内反射）', size: [420, 170],
                render: (c) => PL.draw.plot(c, {
                    x: [0, 90], y: [0, 1.02], xLabel: 'θi (°)', marks: [{ x: Math.asin(1 / 1.5) * 180 / Math.PI, label: 'TIR', color: '#e06c75' }],
                    series: [
                        { fn: (t) => BX.fresnelDielectric(Math.cos(t * Math.PI / 180), 1.5), color: '#6ab0e0', label: 'R 外→内' },
                        { fn: (t) => 1 - BX.fresnelDielectric(Math.cos(t * Math.PI / 180), 1.5), color: '#5ad07a', label: 'T 外→内' },
                        { fn: (t) => BX.fresnelDielectric(Math.cos(t * Math.PI / 180), 1 / 1.5), color: '#6ab0e0', dash: [4, 3], label: 'R 内→外' },
                        { fn: (t) => 1 - BX.fresnelDielectric(Math.cos(t * Math.PI / 180), 1 / 1.5), color: '#5ad07a', dash: [4, 3], label: 'T 内→外' },
                    ],
                }),
            }],
            notes: link(2),
        },
        {
            id: 'm04', title: '04 · 织物 Sheen：Charlie / Ashikhmin', tag: 'SHEEN',
            desc: '绒布、天鹅绒的高光来自竖立纤维：掠射角最亮、正对时暗——GGX 这类"法线集中在 n 附近"的分布描述不了。模块对比 Disney sheen、Ashikhmin velvet（倒高斯分布）与 Charlie sheen（sin<sup>1/α</sup> 分布）及其可见性项，并讲 sheen 叠在底层上时的能量缩放。',
            formula: 'D_Charlie(h) = (2 + 1/α) · sin(θ_h)^{1/α} / (2π)      V_Neubelt = 1 / (4 (n·l + n·v − (n·l)(n·v)))',
            previews: [
                { id: 'p04a', label: 'Charlie sheen + 深色底', smooth: true, render: sphere((S) => IDX.velvet([0.9, 0.55, 0.7], [0.05, 0.01, 0.03], S.rough)) },
                { id: 'p04b', label: 'Charlie vs GGX 波瓣', sub: 'θi = 参数；sheen 在掠射方向更强', size: [240, 150], render: (c, S) => BX.lobe(c, { wi: wiOf(S), cos: true, series: [{ fn: (wo) => IDX.velvet([1, 1, 1], [0, 0, 0], S.rough)(wiOf(S), wo)[0], color: '#c678dd', label: 'Charlie' }, { fn: (wo) => IDX.conductor(BX.METALS.silver, S.rough)(wiOf(S), wo)[1], color: '#6ab0e0', label: 'GGX', dash: [4, 3] }] }) },
            ],
            notes: link(3),
        },
        {
            id: 'm05', title: '05 · 次表面散射 BSSRDF：Dipole → Normalized Diffusion', tag: 'BSSRDF',
            desc: '光从 x<sub>i</sub> 进入、在介质里多次散射后从 x<sub>o</sub> 射出，BRDF 的"同点进出"假设失效。模块从 BSSRDF 定义和体参数讲起，经典 dipole 扩散（Jensen 2001）、高斯和拟合（d\'Eon 2007）、Burley 归一化扩散（下图，红光散射最远），再到实时的屏幕空间可分离 SSS 与预积分皮肤 LUT。',
            formula: 'R(r) = A · (e^{−r/d} + e^{−r/(3d)}) / (8π d r)      ∫_0^∞ R(r) · 2πr dr = A',
            previews: [{
                id: 'p05a', label: 'Burley 归一化扩散剖面 R(r)', sub: '每通道 d = (1.0, 0.45, 0.25) mm，类皮肤', size: [420, 170],
                render: (c) => PL.draw.plot(c, {
                    x: [0.01, 4], logY: true, y: [1e-4, 30], xLabel: 'r (mm)',
                    series: [[1.0, '#e06c75', 'R'], [0.45, '#5ad07a', 'G'], [0.25, '#6ab0e0', 'B']].map(([d, color, label]) => ({ fn: (r) => IDX.normalizedDiffusion(r, d), color, label: `${label}: d = ${d}` })),
                }),
            }],
            notes: link(4),
        },
        {
            id: 'm06', title: '06 · 多层混合与叠加：Clearcoat → 分层 BSDF → 材质总装', tag: 'LAYER',
            desc: '把前面各类 lobe 组合成真实材质：线性混合与物理叠加的区别；clearcoat 直接相加会<strong>凭空产生能量</strong>（下图虚线在掠射角超过 1），按 (1 − F)·(1 − F) 或 1 − E<sub>coat</sub>(μ) 衰减底层才守恒；吸收涂层、Weidlich-Wilkie 分层、OpenPBR 式 albedo scaling，最后用随机游走做 ground truth 并总装 uber 材质。',
            formula: 'f = f_coat + (1 − F_coat(ω_i)) (1 − F_coat(ω_o)) · f_base      // 近似：底层只接收没被涂层反射的能量',
            previews: [
                { id: 'p06a', label: '直接相加', smooth: true, render: sphere(() => IDX.coated(IDX.plastic([0.55, 0.06, 0.05], 0.6), 0.12, false)) },
                { id: 'p06b', label: '(1−F)(1−F) 加权', smooth: true, render: sphere(() => IDX.coated(IDX.plastic([0.55, 0.06, 0.05], 0.6), 0.12, true)) },
                { id: 'p06c', label: '白炉：涂层 + 白色 Lambert 底', sub: '虚线 = 直接相加 > 1（不守恒）', size: [300, 150], render: (c, S) => PL.draw.plot(c, { x: [0, 1], y: [0, 1.25], xLabel: 'μ = cosθo', series: [{ data: S.coatAdd, x: S.mus, color: '#e06c75', dash: [4, 3], label: '直接相加' }, { data: S.coatCon, x: S.mus, color: '#5ad07a', label: '(1−F)(1−F)' }], hmarks: [{ y: 1, label: '1', color: '#555' }] }) },
            ],
            notes: link(5),
        },
    ],
    graph: {
        title: '材质总装 · 各模块在 uber 材质里的位置',
        intro: 'BxDF 本身是函数，所以各 lobe 画成 <strong>INLINE</strong> 节点，被真正的 pass 调用；次表面散射才有独立的屏幕空间 pass。点节点可跳到对应模块的介绍卡片。',
        types: {
            cb: { color: '#e8c25a', label: 'cbuffer · 材质参数' },
            geom: { color: '#9aa5b1', label: 'G-buffer · 几何 / 视线' },
            lobe: { color: '#c678dd', label: 'lobe · 被内联的 BxDF 项' },
            lut: { color: '#d19a66', label: 'LUT · 预积分方向反照率' },
            radiance: { color: '#4a8aff', label: 'radiance · 线性 RGBA16F' },
        },
        common: { title: 'Common.hlsl · 系列共享定义（每个模块的 Common 面板都以此开头）', blocks: [{ label: '━ 系列共享 ━', code: BX.HLSL_COMMON }] },
        nodes: [
            { id: 'mat', title: 'MaterialParams', tag: 'CB', category: 'user', desc: 'baseColor · roughness · metal · coat · sheen …', outputs: [{ name: 'MaterialCB', format: 'cbuffer', type: 'cb' }], detail: { preview: null } },
            { id: 'gbuf', title: 'GBuffer', tag: 'RT', category: 'system', desc: 'N · V · 深度 · 曲率', outputs: [{ name: 'GBuffer', format: 'RGBA16F ×3', type: 'geom' }], detail: { preview: null } },
            { id: 'lut', title: 'CS_BakeAlbedoLUTs', tag: 'CS', category: 'bake', lifetime: 'resident', step: 'm06', desc: 'E(μ, r) for GGX / sheen / coat', dispatch: { name: 'CS_BakeAlbedoLUTs', threads: [8, 8, 1], groups: [4, 4, 3], idRange: 'ID.xy = (μ, r) 32²', note: '多次散射补偿与 albedo scaling 共用' }, outputs: [{ name: 'AlbedoLUTs', format: 'R16F 32×32×3', type: 'lut' }], detail: { preview: 'p02d' } },
            ...[['diffuse', 'Diffuse_Burley', 'm01', 'DiffuseLobe'], ['spec', 'Specular_GGX', 'm02', 'SpecularLobe'], ['btdf', 'Transmission_BTDF', 'm03', 'TransmissionLobe'], ['sheen', 'Sheen_Charlie', 'm04', 'SheenLobe'], ['coat', 'Coat_GGX', 'm06', 'CoatLobe']].map(([id, title, step, out]) => ({
                id, title, tag: 'FN', step, dispatch: { kind: 'INLINE', note: '被 PS_DeferredLighting 内联调用' },
                inputs: [{ name: 'MaterialCB', type: 'cb' }, { name: 'GBuffer', type: 'geom' }], outputs: [{ name: out, format: 'f(ωi, ωo)', type: 'lobe' }],
            })),
            {
                id: 'light', title: 'PS_DeferredLighting', tag: 'PS', lifetime: 'frame', step: 'm06', desc: 'Σ lights · 分层叠加 · 能量缩放',
                dispatch: { kind: 'PS', name: 'PS_DeferredLighting', target: 'full-screen · 两个 RT', idRange: 'SV_Position.xy ∈ viewport' },
                inputs: ['DiffuseLobe', 'SpecularLobe', 'TransmissionLobe', 'SheenLobe', 'CoatLobe'].map((name) => ({ name, type: 'lobe' })).concat([{ name: 'AlbedoLUTs', type: 'lut' }, { name: 'GBuffer', type: 'geom' }]),
                outputs: [{ name: 'DiffuseLighting', format: 'RGBA16F', type: 'radiance' }, { name: 'SpecularLighting', format: 'RGBA16F', type: 'radiance' }],
            },
            {
                id: 'sss', title: 'CS_SeparableSSS', tag: 'CS', lifetime: 'frame', step: 'm05', desc: '按扩散剖面做 H + V 两次模糊',
                dispatch: [{ name: 'CS_SSS_BlurH', threads: [64, 1, 1], idRange: '每 thread 1 像素' }, { name: 'CS_SSS_BlurV', threads: [1, 64, 1], idRange: '每 thread 1 像素' }],
                inputs: [{ name: 'DiffuseLighting', type: 'radiance' }, { name: 'GBuffer', type: 'geom' }], outputs: [{ name: 'SSSDiffuse', format: 'RGBA16F', type: 'radiance' }],
                detail: { preview: 'p05a' },
            },
            {
                id: 'comp', title: 'CS_Composite', tag: 'CS', category: 'output', lifetime: 'frame', desc: 'diffuse(SSS) + specular',
                dispatch: { name: 'CS_Composite', threads: [8, 8, 1], idRange: 'ID.xy ∈ viewport' },
                inputs: [{ name: 'SSSDiffuse', type: 'radiance' }, { name: 'SpecularLighting', type: 'radiance' }], outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'radiance' }],
                detail: { preview: 'p06b' },
            },
        ],
    },
    references: [
        { title: 'A Reflectance Model for Computer Graphics', author: 'R. Cook, K. Torrance', venue: 'ACM TOG', year: 1982 },
        { title: 'Generalization of Lambert\'s Reflectance Model', author: 'M. Oren, S. Nayar', venue: 'SIGGRAPH', year: 1994 },
        { title: 'A Practical Model for Subsurface Light Transport', author: 'H. W. Jensen, S. Marschner, M. Levoy, P. Hanrahan', venue: 'SIGGRAPH', year: 2001 },
        { title: 'Microfacet Models for Refraction through Rough Surfaces', author: 'B. Walter, S. Marschner, H. Li, K. Torrance', venue: 'EGSR', year: 2007 },
        { title: 'Arbitrarily Layered Micro-Facet Surfaces', author: 'A. Weidlich, A. Wilkie', venue: 'GRAPHITE', year: 2007 },
        { title: 'Physically-Based Shading at Disney', author: 'B. Burley', venue: 'SIGGRAPH Course', year: 2012 },
        { title: 'Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs', author: 'E. Heitz', venue: 'JCGT', year: 2014 },
        { title: 'Approximate Reflectance Profiles for Efficient Subsurface Scattering', author: 'P. Christensen, B. Burley', venue: 'Pixar Technical Memo', year: 2015 },
        { title: 'Revisiting Physically Based Shading at Imageworks', author: 'C. Kulla, A. Conty', venue: 'SIGGRAPH Course', year: 2017 },
        { title: 'Production Friendly Microfacet Sheen BRDF', author: 'A. C. Estevez, C. Kulla', venue: 'Sony Pictures Imageworks', year: 2017 },
        { title: 'Sampling the GGX Distribution of Visible Normals', author: 'E. Heitz', venue: 'JCGT', year: 2018 },
        { title: 'OpenPBR Surface Specification', author: 'Academy Software Foundation', year: 2024 },
    ],
});
