// 毛发渲染 02 · Groom：guide → 插值 → 光栅化与覆盖 → LOD
// Build (inside pages/):  node <skill>/scripts/build_doc.mjs hair_common.js hair_02_groom.lab.js -o hair_02_groom.html
// Conventions: see hair_common.js header (cm, y up, fiber frame x = tangent, wi → light, wo → viewer, L_o = S·E).

const G2 = (() => {
    const DEG = Math.PI / 180;
    const BG = [0.004, 0.005, 0.007];                         // 背景（线性 RGB）
    const SEG_LEN = 0.6;                                      // 资产构建：按弧长重采样的目标段长（cm）
    const LIGHT_EL = 38 * DEG;                                // 方向光仰角的默认值（参数 lightEl）
    const GUIDE_COLORS = ['#e8c25a', '#4a8aff', '#5ae87a', '#e86a5a', '#a45fa5', '#5ad0e8', '#e89a3a', '#9ae85a'];
    const hashU = (i) => { let x = (i + 1) * 0x9e3779b1; x ^= x >>> 15; x = Math.imul(x, 0x85ebca6b); x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35); x ^= x >>> 16; return (x >>> 0) / 4294967296; };
    const camOf = (S, o = {}) => HAIR.ortho(Object.assign({ yaw: S.yaw, pitch: S.pitch, center: [0, -7, -0.6], size: 18 }, o));
    const lightDir = (S) => { const el = S.lightEl == null ? LIGHT_EL : S.lightEl; return HAIR.norm([Math.sin(S.lightAz) * Math.cos(el), Math.sin(el), Math.cos(S.lightAz) * Math.cos(el)]); };   // 偏航 = 光方位，俯仰 = 光仰角

    /** 资产构建：makeGroom 的 strand → 按弧长重采样 → Points / Curves 平铺缓冲（与 GPU 资源同布局） */
    const pack = (groom) => {
        const src = groom.strands, nC = src.length, lens = new Float32Array(nC), cnt = new Uint32Array(nC), off = new Uint32Array(nC);
        let total = 0;
        src.forEach((s, i) => {
            const P = s.p; let L = 0;
            for (let k = 3; k < P.length; k += 3) L += Math.hypot(P[k] - P[k - 3], P[k + 1] - P[k - 2], P[k + 2] - P[k - 1]);
            lens[i] = L; cnt[i] = Math.max(4, Math.ceil(L / SEG_LEN) + 1); off[i] = total; total += cnt[i];
        });
        const pos = new Float32Array(3 * total), rad = new Float32Array(total), coordU = new Float32Array(total);
        const rootUV = new Float32Array(2 * nC), clump = new Uint16Array(nC), strands = [];
        src.forEach((s, i) => {
            const P = s.p, n = cnt[i], o = off[i], step = lens[i] / (n - 1);
            let seg = 0, acc = 0, segL = Math.hypot(P[3] - P[0], P[4] - P[1], P[5] - P[2]);
            for (let k = 0; k < n; k++) {
                const target = Math.min(lens[i], k * step);
                while (acc + segL < target && seg < P.length / 3 - 2) { acc += segL; seg++; segL = Math.hypot(P[3 * seg + 3] - P[3 * seg], P[3 * seg + 4] - P[3 * seg + 1], P[3 * seg + 5] - P[3 * seg + 2]); }
                const f = segL > 0 ? HAIR.clamp((target - acc) / segL, 0, 1) : 0, j = 3 * (o + k);
                for (let c = 0; c < 3; c++) pos[j + c] = P[3 * seg + c] + f * (P[3 * seg + 3 + c] - P[3 * seg + c]);
                const u = k / (n - 1);
                coordU[o + k] = u; rad[o + k] = 0.5 * (s.w0 + (s.w1 - s.w0) * u);
            }
            rootUV[2 * i] = s.rootUV[0]; rootUV[2 * i + 1] = s.rootUV[1]; clump[i] = s.guide;
            strands.push({ p: pos.subarray(3 * o, 3 * (o + n)), w0: s.w0, w1: s.w1, id: i, guide: s.guide, rootUV: s.rootUV, len: lens[i] });
        });
        return { nC, total, pos, rad, coordU, cnt, off, lens, rootUV, clump, strands };
    };

    /** makeGroom 内部用的 4-guide 双线性权重（root UV → guide 网格），复现出来给 runtime 插值用 */
    const guideWeights = (O, u, w) => {
        const fx = HAIR.clamp(u * O.guidesX - 0.5, 0, O.guidesX - 1), fz = HAIR.clamp(w * O.guidesZ - 0.5, 0, O.guidesZ - 1);
        const x0 = Math.floor(fx), z0 = Math.floor(fz), x1 = Math.min(O.guidesX - 1, x0 + 1), z1 = Math.min(O.guidesZ - 1, z0 + 1), tx = fx - x0, tz = fz - z0;
        return [[z0 * O.guidesX + x0, (1 - tx) * (1 - tz)], [z0 * O.guidesX + x1, tx * (1 - tz)], [z1 * O.guidesX + x0, (1 - tx) * tz], [z1 * O.guidesX + x1, tx * tz]];
    };

    /** 画一组折线（世界坐标 → 正交相机 → canvas CSS 像素） */
    const drawPolys = (ctx, cam, w, h, list, style) => {
        for (const it of list) {
            const P = it.p, n = P.length / 3;
            ctx.beginPath();
            for (let k = 0; k < n; k++) {
                const q = HAIR.project(cam, [P[3 * k], P[3 * k + 1], P[3 * k + 2]], w, h);
                if (k === 0) ctx.moveTo(q[0] + 0.5, q[1] + 0.5); else ctx.lineTo(q[0] + 0.5, q[1] + 0.5);
            }
            ctx.strokeStyle = style.color(it); ctx.lineWidth = style.width; ctx.globalAlpha = style.alpha; ctx.stroke();
        }
        ctx.globalAlpha = 1;
    };
    const label = (ctx, x, y, text, color = '#cfd8e3') => { ctx.font = '11px ui-monospace, Consolas, monospace'; ctx.fillStyle = color; ctx.fillText(text, x, y); };

    // D3D11 标准 MSAA 样本位置（1/16 像素，y 向下）
    const MSAA4 = [[-2, -6], [6, -2], [-6, 2], [2, 6]].map(([x, y]) => [x / 16, y / 16]);
    const MSAA8 = [[1, -3], [-1, 3], [5, 1], [-3, -5], [-5, 5], [-7, -1], [3, 7], [7, -7]].map(([x, y]) => [x / 16, y / 16]);
    const popc = (m) => { m -= (m >>> 1) & 0x55555555; m = (m & 0x33333333) + ((m >>> 2) & 0x33333333); return (Math.imul((m + (m >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24); };

    /** 点采样光栅化（硬件三角形光栅化的行为）：每像素若干样本点，被任一 ribbon 盖住记 1。
     *  返回每像素"被盖住的样本比例"。o: { offsets, widthScale, minWidthPx, crop:{x,y,w,h}, filter } */
    const rasterSamples = (strands, cam, W, H, o) => {
        const offs = o.offsets, nS = offs.length, ws = o.widthScale || 1, k = H / cam.size;
        const C = o.crop || { x: 0, y: 0, w: W, h: H }, mask = new Uint32Array(C.w * C.h);
        for (const s of strands) {
            if (o.filter && !o.filter(s)) continue;
            const P = s.p, segs = P.length / 3 - 1;
            let a = HAIR.project(cam, [P[0], P[1], P[2]], W, H);
            for (let j = 0; j < segs; j++) {
                const b = HAIR.project(cam, [P[3 * j + 3], P[3 * j + 4], P[3 * j + 5]], W, H);
                const tm = (j + 0.5) / segs, hw = Math.max((o.minWidthPx || 0) / 2, (s.w0 + (s.w1 - s.w0) * tm) * ws * k / 2), hw2 = hw * hw;
                const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-9, r = hw + 1;
                const x0 = Math.max(C.x, Math.floor(Math.min(a[0], b[0]) - r)), x1 = Math.min(C.x + C.w - 1, Math.ceil(Math.max(a[0], b[0]) + r));
                const y0 = Math.max(C.y, Math.floor(Math.min(a[1], b[1]) - r)), y1 = Math.min(C.y + C.h - 1, Math.ceil(Math.max(a[1], b[1]) + r));
                for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
                    let m = 0;
                    for (let q = 0; q < nS; q++) {
                        const sx = x + offs[q][0], sy = y + offs[q][1], u = HAIR.clamp(((sx - a[0]) * dx + (sy - a[1]) * dy) / L2, 0, 1);
                        const ex = a[0] + u * dx - sx, ey = a[1] + u * dy - sy;
                        if (ex * ex + ey * ey < hw2) m |= 1 << q;
                    }
                    if (m) mask[(y - C.y) * C.w + (x - C.x)] |= m;
                }
                a = b;
            }
        }
        const out = new Float32Array(C.w * C.h);
        for (let i = 0; i < out.length; i++) out[i] = popc(mask[i]) / nS;
        return out;
    };

    /** 参考解：crop 内每像素 F×F 个规则子样本，二值覆盖（任一 ribbon 盖住），逐行扫描线求交 → 面积覆盖率 */
    const refCoverage = (strands, cam, W, H, C, F, ws) => {
        const fw = C.w * F, fh = C.h * F, bits = new Uint8Array(fw * fh), k = H / cam.size;
        const toF = (q) => [(q[0] - C.x + 0.5) * F - 0.5, (q[1] - C.y + 0.5) * F - 0.5];
        for (const s of strands) {
            const P = s.p, segs = P.length / 3 - 1;
            let a = toF(HAIR.project(cam, [P[0], P[1], P[2]], W, H));
            for (let j = 0; j < segs; j++) {
                const b = toF(HAIR.project(cam, [P[3 * j + 3], P[3 * j + 4], P[3 * j + 5]], W, H));
                const hw = (s.w0 + (s.w1 - s.w0) * (j + 0.5) / segs) * ws * k / 2 * F, hw2 = hw * hw;
                const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-12, L = Math.sqrt(L2);
                const ymin = Math.max(0, Math.ceil(Math.min(a[1], b[1]) - hw)), ymax = Math.min(fh - 1, Math.floor(Math.max(a[1], b[1]) + hw));
                const bx0 = Math.min(a[0], b[0]) - hw, bx1 = Math.max(a[0], b[0]) + hw;
                if (ymin <= ymax && bx1 >= 0 && bx0 <= fw - 1) {
                    const nx = -dy / L, ny = dx / L;
                    for (let y = ymin; y <= ymax; y++) {
                        let xa = bx0, xb = bx1;
                        if (Math.abs(nx) > 1e-6) { const c0 = a[0] - ny * (y - a[1]) / nx, e = hw / Math.abs(nx); xa = Math.max(xa, c0 - e); xb = Math.min(xb, c0 + e); }
                        const xs = Math.max(0, Math.ceil(xa)), xe = Math.min(fw - 1, Math.floor(xb));
                        for (let x = xs; x <= xe; x++) {
                            const u = HAIR.clamp(((x - a[0]) * dx + (y - a[1]) * dy) / L2, 0, 1), ex = a[0] + u * dx - x, ey = a[1] + u * dy - y;
                            if (ex * ex + ey * ey < hw2) bits[y * fw + x] = 1;
                        }
                    }
                }
                a = b;
            }
        }
        const out = new Float32Array(C.w * C.h), inv = 1 / (F * F);
        for (let cy = 0; cy < C.h; cy++) for (let cx = 0; cx < C.w; cx++) {
            let n = 0;
            for (let sy = 0; sy < F; sy++) { const row = (cy * F + sy) * fw + cx * F; for (let sx = 0; sx < F; sx++) n += bits[row + sx]; }
            out[cy * C.w + cx] = n * inv;
        }
        return out;
    };

    /** 自动挑 crop：α(1−α)³ 总和最大的 n×n 窗口（峰值在 α ≈ 0.25）——发丝彼此分离、最容易走样的稀疏区域 */
    const pickCrop = (alpha, W, H, n) => {
        const I = new Float64Array((W + 1) * (H + 1));
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const a = alpha[y * W + x], b = 1 - a; I[(y + 1) * (W + 1) + x + 1] = a * b * b * b + I[y * (W + 1) + x + 1] + I[(y + 1) * (W + 1) + x] - I[y * (W + 1) + x]; }
        let best = { x: 0, y: 0, w: n, h: n }, bs = -1;
        for (let y = 0; y + n <= H; y += 2) for (let x = 0; x + n <= W; x += 2) {
            const s = I[(y + n) * (W + 1) + x + n] - I[y * (W + 1) + x + n] - I[(y + n) * (W + 1) + x] + I[y * (W + 1) + x];
            if (s > bs) { bs = s; best = { x, y, w: n, h: n }; }
        }
        return best;
    };
    const crop = (buf, W, C) => { const o = new Float32Array(C.w * C.h); for (let y = 0; y < C.h; y++) for (let x = 0; x < C.w; x++) o[y * C.w + x] = buf[(C.y + y) * W + C.x + x]; return o; };
    /** 误差统计：RMSE 与平均偏差（只在参考或测试 > 0 的像素上） */
    const errStats = (test, ref) => {
        let s2 = 0, s1 = 0, n = 0, sr = 0, st = 0;
        for (let i = 0; i < ref.length; i++) { if (ref[i] <= 0 && test[i] <= 0) continue; const e = test[i] - ref[i]; s2 += e * e; s1 += e; n++; sr += ref[i]; st += test[i]; }
        return { rmse: Math.sqrt(s2 / Math.max(1, n)), bias: s1 / Math.max(1, n), ratio: st / Math.max(1e-9, sr) };
    };
    const diffMap = (test, ref) => test.map((v, i) => v - ref[i]);
    /** 在已画好的全图 field 上框出 crop */
    const cropBox = (c, S) => {
        const { ctx, w } = PL.draw.ctx(c, { keep: true }), k = w / S.res, C = S.crop;
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.2; ctx.strokeRect(C.x * k, C.y * k, C.w * k, C.h * k);
    };
    /** 在画布底部叠一行文字（统计量随参数变化，所以画在画布上而不是写死在 sub 里） */
    const setSub = (c, text) => {
        const { ctx, w, h } = PL.draw.ctx(c, { keep: true });
        ctx.fillStyle = 'rgba(0,0,0,0.62)'; ctx.fillRect(0, h - 17, w, 17);
        ctx.font = '11px ui-monospace, Consolas, monospace'; ctx.fillStyle = '#e8edf3'; ctx.fillText(text, 5, h - 5);
    };
    const statSub = (c, st) => setSub(c, `RMSE ${st.rmse.toFixed(3)} · Σα/Σref ${st.ratio.toFixed(2)}×`);
    const probeErr = (test, S, x, y) => { const i = y * S.crop.w + x; return `测试 ${test[i].toFixed(3)} · 参考 ${S.ref[i].toFixed(3)} · Δ = ${(test[i] - S.ref[i]).toFixed(3)}`; };
    /** 合成并显示：C = α·L + (1 − α)·bg，ACES + sRGB（HAIR.display） */
    const EXPOSURE = 1.6, E_LIGHT = 3;   // 方向光辐照度 E（L_o = S·E）
    const drawShade = (c, S, L, scale) => {
        const R = S.res, A = S.R.alpha, rgb = [0, 0, 0];
        PL.draw.pixels(c, R, R, (x, y, out) => {
            const i = y * R + x, a = A[i];
            for (let k = 0; k < 3; k++) rgb[k] = a * L[3 * i + k] * scale + (1 - a) * BG[k];
            const d = HAIR.display(rgb, EXPOSURE); out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out;
        });
    };

    return { DEG, BG, SEG_LEN, LIGHT_EL, GUIDE_COLORS, hashU, camOf, lightDir, pack, guideWeights, drawPolys, label, MSAA4, MSAA8, rasterSamples, refCoverage, pickCrop, crop, errStats, diffMap, cropBox, setSub, statSub, probeErr, drawShade, E_LIGHT };
})();

// ── HLSL snippets ──
const HLSL_BUFFERS = `
    // Groom 资产 → GPU：一条曲线 = Points 里连续的一段，Curves 记 (offset, count)
    struct FHairPoint { float3 Position; float Radius; float CoordU; };   // CoordU = 根 → 梢 ∈ [0,1]
    struct FHairCurve { uint PointOffset; uint PointCount; float2 RootUV; uint ClumpId; };

    StructuredBuffer<FHairPoint> Points;       // 所有曲线的点首尾相接
    StructuredBuffer<FHairCurve> Curves;       // 前缀和：Offset[i+1] = Offset[i] + Count[i]
    StructuredBuffer<uint>       PointToCurve; // 每个点反查所属曲线（VS / CS 按点并行时用）

    // 第 i 条曲线的第 k 个点
    FHairPoint GetPoint(uint i, uint k) { return Points[Curves[i].PointOffset + k]; }`;

const HLSL_INTERP = `
    // CS_HairInterpolation：每个 render 点一个 thread；权重在资产构建时算好（resident）
    StructuredBuffer<float4> RestPosition;        // render strand 静止姿态：xyz = 位置, w = 半径（cm）
    StructuredBuffer<float3> GuideRestPosition;   // guide 静止姿态
    StructuredBuffer<float3> GuideDeformedPosition; // guide 本帧（模拟输出）
    StructuredBuffer<uint4>  GuideIndex;          // 每个 render 点：4 条 guide 上对应点的索引
    StructuredBuffer<float4> GuideWeight;         // 双线性权重（已混入 clump：w' = (1−c)·w + c·δ_nearest）
    RWStructuredBuffer<float4> DeformedPosition;  // 与光栅化 / 剔除读到的格式一致：xyz = 位置, w = 半径

    [numthreads(64, 1, 1)]
    void CS_HairInterpolation(uint3 id : SV_DispatchThreadID) {
        uint p = id.x; if (p >= PointCount) return;
        uint4 g = GuideIndex[p]; float4 w = GuideWeight[p];
        float3 d = 0;                                      // 线性"位移转移"：只搬运 guide 的位移
        [unroll] for (int k = 0; k < 4; k++)
            d += w[k] * (GuideDeformedPosition[g[k]] - GuideRestPosition[g[k]]);
        float4 rest = RestPosition[p];
        DeformedPosition[p] = float4(rest.xyz + d, rest.w); // 生产版：按 guide 局部帧旋转 rest 偏移；半径原样带过去
    }`;

const HLSL_RIBBON = `
    // VS_HairStrands：每个控制点 → 2 个顶点，在 VS 里朝相机展开成带（ribbon）
    cbuffer HairView { float4x4 ViewProj; float3 CameraForward; float WorldToPixel; float WidthScale; float MinPixelRadius; };
    StructuredBuffer<float4> DeformedPosition;          // xyz = 位置, w = 半径（cm）

    // 同一条曲线里的下一个点；发梢钳在本曲线最后一个点，绝不跨到下一条曲线（Curves 的 offset / count）
    uint NextPoint(uint p) { FHairCurve c = Curves[PointToCurve[p]]; return min(p + 1, c.PointOffset + c.PointCount - 1); }
    // 同一条曲线里的上一个点；发根钳在本曲线第一个点 → 端点处中心差分自动退化成单侧差分
    uint PrevPoint(uint p) { FHairCurve c = Curves[PointToCurve[p]]; return max(p, c.PointOffset + 1) - 1; }

    struct VSOut { float4 Pos : SV_Position; nointerpolation uint CurveId : TEXCOORD0; float Side : TEXCOORD1; float Coverage : TEXCOORD2; float U : TEXCOORD3; };

    VSOut VS_HairStrands(uint vid : SV_VertexID) {
        uint p = vid >> 1;  float side = (vid & 1) ? 1.0 : -1.0;     // 低位选左 / 右
        float4 P  = DeformedPosition[p];
        float3 T  = normalize(DeformedPosition[NextPoint(p)].xyz - DeformedPosition[PrevPoint(p)].xyz);  // 中心差分切线
        float3 V  = -CameraForward;                                 // 正交；透视用 normalize(CameraPos − P)
        float3 B  = normalize(cross(T, V));                         // 屏幕内垂直于发丝
        float  r  = P.w * WidthScale;                               // 真实半径（cm）
        float  rR = max(r, MinPixelRadius / WorldToPixel);          // 光栅化半宽 ≥ r_min（0.5–1 px），否则 PS 根本不会在大多数像素上执行
        VSOut o;
        o.Pos      = mul(float4(P.xyz + side * rR * B, 1), ViewProj);
        o.Side     = side;                                          // 插值后 = 横向偏移 h ∈ [−1,1]
        o.Coverage = r / rR;                                        // 加宽了多少 → 覆盖率按比例缩回
        o.CurveId  = PointToCurve[p];
        o.U        = Points[p].CoordU;
        return o;
    }`;

PipelineLab.mount({
    paramSteps: { yaw: ['asset', 'interp', 'geom', 'alias', 'cover', 'vis', 'shade', 'lod'], pitch: ['asset', 'interp', 'geom', 'alias', 'cover', 'vis', 'shade', 'lod'], res: ['geom', 'alias', 'cover', 'vis', 'shade', 'lod'], perGuide: ['asset', 'interp', 'geom', 'alias', 'cover', 'vis', 'shade', 'lod'], clump: ['asset', 'interp', 'geom', 'alias', 'cover', 'vis', 'shade', 'lod'], curl: ['asset', 'interp', 'geom', 'alias', 'cover', 'vis', 'shade', 'lod'], frizz: ['asset', 'interp', 'geom', 'alias', 'cover', 'vis', 'shade', 'lod'], lengthVar: ['asset', 'interp', 'geom', 'alias', 'cover', 'vis', 'shade', 'lod'], widthScale: ['geom', 'alias', 'cover', 'vis', 'shade', 'lod'], keep: ['lod'], color: ['shade'], lightAz: ['shade'], lightEl: ['shade'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '毛发渲染 02 · Groom：guide → 插值 → 光栅化与覆盖 → LOD',
    lang: 'zh-CN',
    codeLang: 'hlsl',
    seed: 7,
    regen: false,

    params: [
        { id: 'yaw', label: 'yaw', min: -180, max: 180, step: 1, value: 22, unit: '°', map: (d) => d * G2.DEG, title: '相机绕 y 轴（正交相机）',
            desc: '正交相机绕竖直轴的方位角。0° 正对发帘，±90° 从侧面看：两排 guide 前后叠在一起，像素里重叠的发丝变多、α 更快饱和。所有用主相机的画面都跟着转（Step 01 / 02 的线框、Step 03–08 的光栅化缓冲）。' },
        { id: 'pitch', label: 'pitch', min: -60, max: 60, step: 1, value: 8, unit: '°', map: (d) => d * G2.DEG, title: '相机俯仰',
            desc: '相机俯仰角：正值 = 从略低处仰视，负值 = 从上往下看。|pitch| 越大，下垂的发丝在画面里被压得越短、彼此重叠越多；影响的步骤与 yaw 相同。' },
        { id: 'res', label: '分辨率', options: [160, 180, 220], value: 180, title: '光栅化分辨率（正方形）。画面高 18 cm，180 px → 1 px ≈ 1 mm',
            desc: '光栅化分辨率（正方形，画面高固定 18 cm）：180 px 时 1 px ≈ 1 mm，根部发丝只有 0.08 px 宽。调到 220，Step 03 的宽度曲线整体上移（越接近能直接光栅化的区间），Step 04–08 的覆盖、走样、着色与 LOD 缓冲全部按新分辨率重算。' },
        { id: 'perGuide', label: 'strands/guide', min: 10, max: 150, step: 5, value: 90, format: (v) => `${v} → ${v * 16}`, title: '每条 guide 生成多少 render strand（16 条 guide）',
            desc: '每条 guide 插值出多少根 render strand（16 条 guide，箭头后是总数）。guide 不变、发量变密：Step 01 的点数统计、Step 05 的 Σc 与 α 随之升高（越密越接近饱和），其余光栅化画面跟着变化。' },
        { id: 'clump', label: 'clump', min: 0, max: 1, step: 0.05, value: 0.6, title: '向最近 guide 聚拢的强度（沿长度增强）',
            desc: '向最近 guide 聚拢的强度，沿长度按 t<sup>0.7</sup> 增强。拖到 0：纯 4-guide 加权，发丝均匀铺开（Step 02 分解图中栏 = 左栏）；拖到 1：发梢全部收进 guide、成簇感最强，发丝在像素内互相遮挡，Step 05 的 α_OIT 开始偏高。' },
        { id: 'curl', label: 'curl', min: 0, max: 0.6, step: 0.02, value: 0.16, unit: ' cm', title: '卷曲半径（沿长度 √t 增长）',
            desc: '每根发丝的螺旋卷曲半径（cm，沿长度按 √t 增大）。只改 render strand、不动 guide：Step 02 分解图右栏和 3D 视图里看螺旋；卷发弧长变长，按弧长重采样后 Step 01 的每曲线点数（count 柱）变多。' },
        { id: 'frizz', label: 'frizz', min: 0, max: 0.5, step: 0.01, value: 0.06, unit: ' cm', title: '毛躁：低频随机扰动幅度',
            desc: '毛躁：每根发丝独立的低频正弦扰动幅度（cm，沿长度线性增大），让发丝彼此散开、不再整齐平行。单独效果看 Step 02 分解图右栏；其余画面随 groom 一起变化。' },
        { id: 'lengthVar', label: '长度随机', min: 0, max: 0.6, step: 0.05, value: 0.25, title: '长度随机：len = 1 − lengthVar · ξ',
            desc: '每根发丝的长度 len = 1 − lengthVar·ξ（ξ 均匀随机）。调大后发梢参差、下半部分变稀：Step 01 的 count 柱高低不齐，Step 05 的 α 在发梢一带下降。' },
        { id: 'widthScale', label: '宽度×', min: 0.25, max: 12, step: 0.25, value: 1, title: '发丝宽度缩放（根 80 µm、梢 30 µm）',
            desc: '发丝宽度缩放（真实值：根 80 µm、梢 30 µm）。拖到 8–12，发丝接近 1 px：Step 03 的 ribbon 变成实心带，Step 04 两种"直接画"的做法都逼近参考；同时 Step 08 的 f_min 右移——发丝越粗，能抽掉的越少。' },
        { id: 'keep', label: 'LOD 保留', min: 0.05, max: 1, step: 0.05, value: 0.35, title: 'LOD：保留的 strand 比例（其余剔除）',
            desc: 'LOD 保留比例 f：按每根 strand 的固定哈希保留这么多，其余整根剔除；加宽版再把宽度乘 1/f。往小拖看 Step 08：只剔除时发量明显变稀，加宽版在 f 不小于 f_min ≈ w_px 时仍保住覆盖率。' },
        { id: 'color', label: '发色', value: '#6b4a32', map: HAIR.hex2lin, title: '目标反照率（线性）→ σa（Chiang 2016 反推）',
            desc: '目标发色（线性反照率），按 Chiang 2016 的拟合反推吸收系数 σ<sub>a</sub>。只影响 Step 07 的着色：TT / TRT 带发色，R 高光保持白色；改动会清空 history 累积。' },
        { id: 'lightAz', label: '光方位', min: -180, max: 180, step: 5, value: 35, unit: '°', map: (d) => d * G2.DEG, title: '方向光的偏航（绕竖直轴）',
            desc: '方向光的<strong>偏航</strong>（绕竖直轴转动，0° = 从相机一侧照射），俯仰由“光仰角”控制。转到 ±180°（背光）：TT 透射让稀疏的发梢整体发亮；正面光时主要是白色的 R 高光和带发色的 TRT 二次高光。只影响 Step 07。' },
        { id: 'lightEl', label: '光仰角', min: -20, max: 85, step: 1, value: 38, unit: '°', map: (d) => d * G2.DEG,
            desc: '方向光的<strong>俯仰</strong>（正 = 从上方照来）。它决定高光带落在发丝的哪一段：发丝朝向满足镜面条件（θh ≈ −α）的位置才亮，抬高或压低光源，Step 07 的高光带沿发丝上下移动；接近 85° 时成为顶光。只影响 Step 07。' },
    ],

    stages: [
        {
            id: 'groom', lifetime: 'param', run(S) {
                S.groom = HAIR.makeGroom({ seed: 7, perGuide: S.perGuide, clump: S.clump, curl: S.curl, frizz: S.frizz, lengthVar: S.lengthVar });
                S.asset = G2.pack(S.groom);
                S.cam = G2.camOf(S);
            },
        },
        {
            // 插值分解：同一 seed、少量 strand（RNG 顺序相同 → 与完整 groom 的前 384 条一一对应）
            id: 'decomp', lifetime: 'param', lazy: true, run(S) {
                const base = { seed: 7, perGuide: 24, lengthVar: S.lengthVar };
                S.decomp = [
                    HAIR.makeGroom(Object.assign({}, base, { clump: 0, curl: 0, frizz: 0 })),
                    HAIR.makeGroom(Object.assign({}, base, { clump: S.clump, curl: 0, frizz: 0 })),
                    HAIR.makeGroom(Object.assign({}, base, { clump: S.clump, curl: S.curl, frizz: S.frizz })),
                ];
            },
        },
        {
            // runtime 插值：guide 用一个解析摆动代替模拟（每帧），render strand 通过资产里的权重跟随
            id: 'interp', lifetime: 'frame', lazy: true, run(S) {
                const G = S.groom, O = G.opts, N = O.segs, t = S.t;
                if (!S.strandW || S._grEpoch !== S.epoch) {
                    // 资产构建时一次：权重 + 每个点对应的 guide 点号 tk（makeGroom 取 round(t·len·N)；y 不受 clump 影响，可由 y 反推）
                    S.strandW = G.strands.map((s) => {
                        const w = G2.guideWeights(O, s.rootUV[0], s.rootUV[1]);
                        const Lw = w.reduce((a, [gi, wt]) => a + wt * -G.guides[gi].p[3 * N + 1], 0) || 1;
                        const tk = new Uint8Array(N + 1);
                        for (let k = 0; k <= N; k++) tk[k] = HAIR.clamp(Math.round(-s.p[3 * k + 1] * N / Lw), 0, N);
                        return { w, near: s.guide, tk };
                    });
                    S._grEpoch = S.epoch;
                }
                const gd = PL.buf(S, 'guideDef', G.guides.length * 3 * (N + 1));
                G.guides.forEach((g, gi) => {
                    const ph = gi * 0.7, ax = 1.6 * Math.sin(1.3 * t + ph), az = 0.9 * Math.sin(0.9 * t + 1.7 * ph);
                    for (let k = 0; k <= N; k++) {
                        const s = k / N, j = 3 * (gi * (N + 1) + k), sw = s * s;
                        gd[j] = g.p[3 * k] + ax * sw; gd[j + 1] = g.p[3 * k + 1] + 0.12 * (ax * ax + az * az) * sw * 0.2; gd[j + 2] = g.p[3 * k + 2] + az * sw;
                    }
                });
                const sd = PL.buf(S, 'strandDef', G.strands.length * 3 * (N + 1));
                G.strands.forEach((s, si) => {
                    const { w, near, tk } = S.strandW[si];
                    for (let k = 0; k <= N; k++) {
                        const tt = k / N, c = O.clump * Math.pow(tt, 0.7), j = 3 * (si * (N + 1) + k), q0 = tk[k];
                        let dx = 0, dy = 0, dz = 0;   // 位移按 makeGroom 取样的同一个 guide 点 tk 转移
                        const add = (gi, wt) => { const q = 3 * (gi * (N + 1) + q0), r = 3 * q0, gp = G.guides[gi].p; dx += wt * (gd[q] - gp[r]); dy += wt * (gd[q + 1] - gp[r + 1]); dz += wt * (gd[q + 2] - gp[r + 2]); };
                        for (const [gi, wt] of w) add(gi, wt * (1 - c));
                        add(near, c);
                        sd[j] = s.p[3 * k] + dx; sd[j + 1] = s.p[3 * k + 1] + dy; sd[j + 2] = s.p[3 * k + 2] + dz;
                    }
                });
            },
        },
        {
            // 主光栅化：解析亚像素覆盖（HAIR.raster）+ 由最近 strand 反推的"只留最近一根"覆盖率
            id: 'raster', lifetime: 'param', run(S) {
                const R = S.res, n = R * R, A = S.asset, k = R / S.cam.size;
                S.R = HAIR.raster(A.strands, S.cam, R, R, { widthScale: S.widthScale });
                const cn = PL.buf(S, 'covNear', n), cs = PL.buf(S, 'covSum', n);
                for (let i = 0; i < n; i++) {
                    cs[i] = Math.min(1, S.R.cov[i]);
                    const id = S.R.id[i]; if (id < 0) { cn[i] = 0; continue; }
                    const s = A.strands[id], hw = (s.w0 + (s.w1 - s.w0) * S.R.v[i]) * S.widthScale * k / 2, d = Math.abs(S.R.h[i]) * Math.max(hw, 0.5);
                    cn[i] = hw >= 0.5 ? HAIR.clamp(hw + 0.5 - d, 0, 1) : 2 * hw * Math.max(0, 1 - d);   // 与 HAIR.raster 同一覆盖核
                }
                S.crop = G2.pickCrop(S.R.alpha, R, R, 40);
            },
        },
        {
            // 走样对比：1 px 硬线 / 真实宽度 + 像素中心采样 / MSAA 8× / 16×16 超采样参考（crop 内）
            id: 'alias', lifetime: 'param', lazy: true, run(S) {
                const R = S.res, st = S.asset.strands, C = S.crop, ws = S.widthScale;
                S.aNaive = G2.rasterSamples(st, S.cam, R, R, { offsets: [[0, 0]], widthScale: ws, minWidthPx: 1 });
                S.aCenter = G2.rasterSamples(st, S.cam, R, R, { offsets: [[0, 0]], widthScale: ws });
                S.ref = G2.refCoverage(st, S.cam, R, R, C, 16, ws);
                S.cMsaa = G2.rasterSamples(st, S.cam, R, R, { offsets: G2.MSAA8, widthScale: ws, crop: C });
                S.cMsaaW = G2.rasterSamples(st, S.cam, R, R, { offsets: G2.MSAA8, widthScale: ws, minWidthPx: 1, crop: C });
                const cr = (b) => G2.crop(b, R, C);
                S.cNaive = cr(S.aNaive); S.cCenter = cr(S.aCenter); S.cAlpha = cr(S.R.alpha); S.cNear = cr(S.covNear); S.cSum = cr(S.covSum);
                const E = (t) => G2.errStats(t, S.ref);
                S.stats = { naive: E(S.cNaive), center: E(S.cCenter), alpha: E(S.cAlpha), near: E(S.cNear), sum: E(S.cSum), msaa: E(S.cMsaa), msaaW: E(S.cMsaaW) };
            },
        },
        {
            // TAA 思路：每帧 1 个抖动样本（Halton 2,3），真实宽度二值覆盖，逐帧累积（crop 内）
            id: 'taa', lifetime: 'history', lazy: true, needs: ['alias'], run(S) {
                const C = S.crop, n = C.w * C.h;
                if (S._taaEpoch !== S.epoch || !S.taaSum || S.taaSum.length !== n) { S.taaSum = new Float32Array(n); S.taaN = 0; S.taaErr = []; S._taaEpoch = S.epoch; }
                const hal = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };
                const j = [hal(S.taaN + 1, 2) - 0.5, hal(S.taaN + 1, 3) - 0.5];
                S.taaCur = G2.rasterSamples(S.asset.strands, S.cam, S.res, S.res, { offsets: [j], widthScale: S.widthScale, crop: C });
                S.taaN++;
                const mean = PL.buf(S, 'taaMean', n);
                for (let i = 0; i < n; i++) { S.taaSum[i] += S.taaCur[i]; mean[i] = S.taaSum[i] / S.taaN; }
                if (S.taaErr.length < 512) S.taaErr.push(G2.errStats(mean, S.ref).rmse);
            },
        },
        {
            // 着色准备：每个可见像素的发丝局部帧（切线来自光栅化缓冲），wi / wo 转到 fiber 坐标
            id: 'shadeSetup', lifetime: 'param', lazy: true, run(S) {
                const R = S.res, n = R * R, Rb = S.R, Ld = G2.lightDir(S), V = S.cam.toViewer;
                S.hairBsdf = HAIR.bsdf({ sigmaA: HAIR.sigmaAFromReflectance(S.color, 0.3), betaM: 0.3, betaN: 0.3, alpha: 2 });
                const wl = PL.buf(S, 'wLocal', 6 * n), src = PL.buf(S, 'shadeSrc', n, Int32Array), list = [];
                src.fill(-1);
                for (let i = 0; i < n; i++) if (Rb.id[i] >= 0) src[i] = i;
                // 有覆盖率但没有可见性样本的像素（离发丝 0.5–1 px）：借 3×3 邻域里的样本——否则边缘一圈是黑的
                for (let y = 0; y < R; y++) for (let x = 0; x < R; x++) {
                    const i = y * R + x; if (src[i] >= 0 || Rb.alpha[i] <= 1e-3) continue;
                    for (let d = 0; d < 9 && src[i] < 0; d++) { const xx = x + (d % 3) - 1, yy = y + Math.floor(d / 3) - 1; if (xx >= 0 && yy >= 0 && xx < R && yy < R && Rb.id[yy * R + xx] >= 0) src[i] = yy * R + xx; }
                }
                for (let i = 0; i < n; i++) {
                    const j = src[i]; if (j < 0) continue;
                    const f = HAIR.fiberFrame([Rb.tan[3 * j], Rb.tan[3 * j + 1], Rb.tan[3 * j + 2]], V), a = HAIR.toFiber(Ld, f), b = HAIR.toFiber(V, f);
                    wl[6 * i] = a[0]; wl[6 * i + 1] = a[1]; wl[6 * i + 2] = a[2]; wl[6 * i + 3] = b[0]; wl[6 * i + 4] = b[1]; wl[6 * i + 5] = b[2];
                    list.push(i);
                }
                S.hairPix = Int32Array.from(list);
            },
        },
        {
            // 每像素每帧一个随机 h ∈ [−1,1]（宽度方向的入射位置），history 里累积 → 收敛到沿宽度平均的远场 S̄
            id: 'accum', lifetime: 'history', lazy: true, needs: ['shadeSetup'], run(S) {
                const n = S.res * S.res, B = S.hairBsdf, wl = S.wLocal, E = G2.E_LIGHT;
                if (S._accEpoch !== S.epoch || !S.accSum || S.accSum.length !== 3 * n) { S.accSum = new Float32Array(3 * n); S.accN = 0; S._accEpoch = S.epoch; }
                const cur = PL.buf(S, 'shadeCur', 3 * n), rnd = PL.rng((S.seed ^ 0x9e3779b9) + 7919 * S.accN), wi = [0, 0, 0], wo = [0, 0, 0];
                for (const i of S.hairPix) {
                    wi[0] = wl[6 * i]; wi[1] = wl[6 * i + 1]; wi[2] = wl[6 * i + 2]; wo[0] = wl[6 * i + 3]; wo[1] = wl[6 * i + 4]; wo[2] = wl[6 * i + 5];
                    const s = B.S(wi, wo, 2 * rnd() - 1);
                    for (let c = 0; c < 3; c++) { cur[3 * i + c] = s[c] * E; S.accSum[3 * i + c] += s[c] * E; }
                }
                S.accN++;
            },
        },
        {
            // LOD：按 strand 哈希保留比例 f，其余剔除；加宽 1/f 维持覆盖率。曲线在 96² 上扫 f
            id: 'lod', lifetime: 'param', lazy: true, run(S) {
                const R = S.res, st = S.asset.strands, f = S.keep, keepF = (s) => G2.hashU(s.id) < f;
                S.lodPlain = HAIR.raster(st, S.cam, R, R, { widthScale: S.widthScale, filter: keepF }).alpha;
                S.lodWide = HAIR.raster(st, S.cam, R, R, { widthScale: S.widthScale / f, filter: keepF }).alpha;
                S.lodKept = st.filter(keepF).length;
                const r = 96, full = HAIR.raster(st, S.cam, r, r, { widthScale: S.widthScale }).alpha, fs = [0.02, 0.035, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1];
                const mean = (a) => { let s = 0, m = 0; for (let i = 0; i < a.length; i++) if (full[i] > 0.02) { s += a[i]; m++; } return s / Math.max(1, m); }, m0 = mean(full);
                S.lodFs = fs;
                S.lodCurves = [0, -1, -0.5].map((e) => fs.map((ff) => mean(HAIR.raster(st, S.cam, r, r, { widthScale: S.widthScale * Math.pow(ff, e), filter: (s) => G2.hashU(s.id) < ff }).alpha) / m0));
            },
        },
    ],

    intro: {
        title: 'Strand-based Hair（Groom）· 从资产到像素',
        flow: ['Groom 资产', 'guide → strand 插值', 'VS 展开 ribbon', '亚像素覆盖', 'visibility / OIT / AA', '着色 + 时域累积', 'LOD'],
        html: `<p>本篇讲<strong>逐根发丝</strong>（strand-based，UE 里叫 Groom）的渲染管线：资产里存几千到十几万条折线，少量 <em>guide</em> 负责模拟和造型，大量 <em>render strand</em> 由 guide 插值得到；
            渲染时每根发丝被展开成面向相机的细带，<strong>真实宽度只有 0.03–0.1 像素</strong>，于是"覆盖率"——而不是"有没有被画到"——成了光栅化的核心。</p>
            <p>每一步的预览都是页面里 JS 参考实现真实算出的缓冲：groom 由 <code>HAIR.makeGroom</code> 生成（16 条 guide × N 条 strand），光栅化用 <code>HAIR.raster</code>（解析亚像素覆盖），着色用 <code>HAIR.bsdf</code>（Chiang 2016 / pbrt-v3）。
            相机为正交，画面高 18 cm；默认 180 px 时 1 px ≈ 1 mm，而根部发丝直径 80 µm ≈ <strong>0.08 px</strong>。光照只用一盏方向光、不做阴影——自阴影与多次散射是 03 篇的主题。</p>
            <p>变体选择：strand 由 guide 程序化插值生成（clump / curl / frizz，类似 Houdini / XGen / HairWorks 的做法），运行时再按 UE Groom 的方式把 guide 的模拟位移转移给 render strand；覆盖率用解析亚像素核，可见性只留每像素最近的一根，AA 对照 MSAA 与 TAA。引擎专有的细节（cvar、样本数、数据打包）随版本变化，以参考资料里的源码与文档为准。</p>
            ${HAIR.nav('groom')}`,
    },

    steps: [
        {
            id: 'asset', title: 'Groom 资产：guide 曲线、render strand 与 Points / Curves 缓冲', tag: 'ASSET',
            desc: `一个 groom 资产里有两套曲线：<strong>guide</strong>（几十到几千条，给模拟和造型用，点数少）和 <strong>render strand</strong>（几万到几十万条，真正画出来的发丝）。
                每条曲线是一条折线；<em>每点</em>数据有位置、半径、沿发丝坐标 u（根 0 → 梢 1），<em>每曲线</em>数据有 root UV（发根在头皮网格上的 UV，用来采样头皮颜色 / 绑定到网格）和 clump id（属于哪一簇，这里 = 最近的 guide）。
                GPU 上不存"曲线对象"：所有点首尾相接放进一个 <strong>Points</strong> 缓冲，<strong>Curves</strong> 缓冲只记每条曲线的 <code>(offset, count)</code>——标准的 CSR / 前缀和布局，按点并行（VS、插值 CS）时再用一个 point → curve 反查表。`,
            formula: [
                'Points[j]  = { P_j, r_j, u_j }                 // 位置 (cm)、半径、根→梢坐标',
                'Curves[i]  = { offset_i, count_i, rootUV_i, clump_i }',
                'offset_{i+1} = offset_i + count_i             // 前缀和；第 i 条曲线 = Points[offset_i … offset_i + count_i − 1]',
                'count_i = ⌈ len_i / Δ ⌉ + 1,  Δ = 0.6 cm         // 构建时按弧长重采样：越长 / 越卷的发丝点越多',
            ],
            code: { title: 'HLSL · Points / Curves 缓冲布局', src: HLSL_BUFFERS },
            previews: [
                {
                    id: 'assetView', label: 'guide（左）vs render strand（右）', sub: '同一相机；strand 按 clump id（最近 guide）着色', size: [360, 200],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0e12' }), half = w / 2;
                        const cam = G2.camOf(S, { size: 19 });
                        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, half, h); ctx.clip(); ctx.translate(-half / 2, 0);
                        G2.drawPolys(ctx, cam, w, h, S.groom.guides, { color: (g) => G2.GUIDE_COLORS[(g.gx + g.gz * 3) % 8], width: 1.6, alpha: 0.95 });
                        ctx.restore();
                        ctx.save(); ctx.beginPath(); ctx.rect(half, 0, half, h); ctx.clip(); ctx.translate(half / 2, 0);
                        const gs = S.groom.guides;
                        G2.drawPolys(ctx, cam, w, h, S.asset.strands, { color: (s) => G2.GUIDE_COLORS[(gs[s.guide].gx + gs[s.guide].gz * 3) % 8], width: 0.5, alpha: 0.35 });
                        ctx.restore();
                        ctx.fillStyle = '#334'; ctx.fillRect(half - 0.5, 0, 1, h);
                        G2.label(ctx, 6, 14, `${S.groom.guides.length} guides`); G2.label(ctx, half + 6, 14, `${S.asset.nC} strands · ${S.asset.total} points`);
                    },
                },
                {
                    id: 'pointsBuf', label: 'Points 缓冲 · 半径 r (µm)', sub: '行 = 前 128 条曲线，列 = 曲线内点号；黑 = 超出 count', size: [360, 200],
                    render: (c, S) => {
                        const A = S.asset, rows = Math.min(128, A.nC); let maxC = 0;
                        for (let i = 0; i < rows; i++) maxC = Math.max(maxC, A.cnt[i]);
                        const img = PL.buf(S, '_ptsImg', rows * maxC);
                        for (let i = 0; i < rows; i++) for (let k = 0; k < maxC; k++) img[i * maxC + k] = k < A.cnt[i] ? A.rad[A.off[i] + k] * 1e4 : NaN;
                        S._ptsW = maxC;
                        PL.draw.field(c, img, maxC, rows, { map: 'viridis', range: [10, 40] });
                    },
                    probe: (x, y, S) => {
                        const A = S.asset, W = S._ptsW || 1, cx = Math.floor(x * W / 360), cy = Math.floor(y * Math.min(128, A.nC) / 200);
                        if (cy >= A.nC) return '';
                        const k = cx; if (k >= A.cnt[cy]) return `curve ${cy}: count = ${A.cnt[cy]}，offset = ${A.off[cy]}`;
                        const j = A.off[cy] + k;
                        return `curve ${cy} · point ${k} → Points[${j}]  r = ${(A.rad[j] * 1e4).toFixed(1)} µm  u = ${A.coordU[j].toFixed(2)}`;
                    },
                },
                {
                    id: 'curveBuf', label: 'Curves 缓冲 · count 与 offset', sub: '前 160 条曲线：柱 = count_i，线 = offset_i（按右端归一到柱高）', size: [360, 180],
                    render: (c, S) => {
                        const A = S.asset, n = Math.min(160, A.nC);
                        const counts = Array.from({ length: n }, (_, i) => A.cnt[i]), maxC = Math.max(...counts), so = maxC / A.off[n - 1];
                        const offs = Array.from({ length: n }, (_, i) => A.off[i] * so);
                        PL.draw.plot(c, {
                            x: [0, n - 1], y: [0, maxC * 1.15], xLabel: 'curve index i', yLabel: 'points',
                            series: [{ data: counts, bar: true, color: '#4a8aff', label: 'count_i' }, { data: offs, color: '#e8c25a', label: `offset_i × ${so.toFixed(3)}` }],
                            hmarks: [{ y: A.total / A.nC, label: `平均 ${(A.total / A.nC).toFixed(1)} 点/曲线`, color: '#5ae87a' }],
                        });
                    },
                },
            ],
            notes: '拖 <code>curl</code>：卷发的弧长变长，按弧长重采样后每条曲线的点数（柱）变多、offset 斜率变陡；拖 <code>长度随机</code>：count 参差变大。<code>strands/guide</code> 只改 render strand 数量，guide 数量（16）不变。<br>对照 UE：CPU 侧的 <code>FHairStrandsCurves</code> 就是 <code>CurvesCount</code> / <code>CurvesOffset</code> / <code>CurvesLength</code> / <code>CurvesRootUV</code> / <code>ClumpIDs</code>，<code>FHairStrandsPoints</code> 是 <code>PointsPosition</code> / <code>PointsRadius</code>（按最大半径归一化）/ <code>PointsCoordU</code> 以及颜色、粗糙度、AO。',
            warn: [
                '真实资产（Alembic / USD 导出）每条曲线点数本来就不同，所以一定要有 offset/count 这一层间接；按"每条固定 N 点"假设写的 shader 一遇到真实资产就错位。',
                'guide 不是"少画几根的 strand"：它的点数、分布和平滑程度都为模拟服务；用 render strand 去做模拟既贵又不稳定，反过来直接渲染 guide 则发量明显不足。',
            ],
        },
        {
            id: 'interp', title: '插值：guide → render strand（权重、clump、curl、frizz、长度随机）', tag: 'INTERP',
            desc: `render strand 的形状由附近几条 guide 决定。经典做法有两种（Yuksel &amp; Tariq 2010 的课程里称 single-strand / multi-strand interpolation）：
                <strong>最近 guide</strong>（clump 风格，一簇发丝跟一条 guide 走，成簇感强）和<strong>多 guide 加权</strong>（重心 / 双线性权重，过渡平滑但发丝会"糊成一片"）。
                <code>HAIR.makeGroom</code> 把两者混合：先按发根在 guide 网格里的位置取 4 条 guide 的双线性加权，再沿长度以 <code>c(t) = clump · t^{0.7}</code> 向最近 guide 拉拢；最后叠加每根独立的 curl（螺旋）、frizz（低频随机）与长度随机。
                运行时 guide 被模拟（UE 里是 Niagara），render strand 用<strong>同一组在资产构建时算好的权重</strong>把 guide 的位移转移过来——这就是每帧的插值 CS。`,
            formula: [
                'base(t) = Σ_k w_k · G_k(t)                       // 4 条 guide，双线性权重 Σ w_k = 1',
                'c(t) = clump · t^{0.7}                           // 越靠近发梢越往最近 guide 聚拢',
                'P(t) = (1 − c)·(base + Δroot) + c·G_near(t) + curl·√t·(cos ψ, 0, sin ψ) + frizz·t·ξ(t)',
                'ψ = 2π·f_curl·t·len + ψ_0,   len = 1 − lengthVar·ξ   // 每根随机相位与长度',
                '运行时：P_i(t, τ) = P_i^{rest}(t) + Σ_k w′_{ik}·( G_k(t, τ) − G_k^{rest}(t) )   // 位移转移，w′ 含 clump',
            ],
            code: { title: 'HLSL · CS_HairInterpolation（运行时位移转移）', src: HLSL_INTERP },
            previews: [
                {
                    id: 'groom3d', large: true, label: '3D：guide（粗）与插值出的 render strand（细）',
                    sub: '拖动旋转。粗线 = 16 条 guide；细线 = 按固定哈希抽出的约 160 根 render strand，颜色 = clump id（最近的 guide）；白线 = 发根图里那一根，虚线连向它的 4 条 guide（越亮权重越大）；褐色条 = 头皮上的发根区域，网格 = 发梢高度',
                    render: (c, S) => drawGroom3d(c, S),
                },
                {
                    id: 'rootPatch', label: '发根 UV 与 guide 网格', sub: '点 = strand 发根（按最近 guide 着色），圈 = guide 发根；白线 = 自动挑出的一根 strand 的 4 个权重', size: [360, 160],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0e12' }), O = S.groom.opts, m = 12;
                        const X = (u) => m + u * (w - 2 * m), Y = (v) => m + v * (h - 2 * m);
                        ctx.strokeStyle = '#2a3340'; ctx.lineWidth = 1;
                        for (let gx = 0; gx < O.guidesX; gx++) { ctx.beginPath(); ctx.moveTo(X((gx + 0.5) / O.guidesX), Y(0)); ctx.lineTo(X((gx + 0.5) / O.guidesX), Y(1)); ctx.stroke(); }
                        for (let gz = 0; gz < O.guidesZ; gz++) { ctx.beginPath(); ctx.moveTo(X(0), Y((gz + 0.5) / O.guidesZ)); ctx.lineTo(X(1), Y((gz + 0.5) / O.guidesZ)); ctx.stroke(); }
                        const gs = S.groom.guides;
                        for (const s of S.groom.strands) { ctx.fillStyle = G2.GUIDE_COLORS[(gs[s.guide].gx + gs[s.guide].gz * 3) % 8]; ctx.globalAlpha = 0.55; ctx.fillRect(X(s.rootUV[0]) - 0.8, Y(s.rootUV[1]) - 0.8, 1.6, 1.6); }
                        ctx.globalAlpha = 1;
                        gs.forEach((g) => { ctx.beginPath(); ctx.arc(X((g.gx + 0.5) / O.guidesX), Y((g.gz + 0.5) / O.guidesZ), 4, 0, 2 * Math.PI); ctx.strokeStyle = G2.GUIDE_COLORS[(g.gx + g.gz * 3) % 8]; ctx.lineWidth = 2; ctx.stroke(); });
                        // 自动挑一根：发根离 (0.47, 0.42) 最近的 strand（落在 4 个 guide 之间，权重各不相同）
                        let pick = S.groom.strands[0], bd = Infinity;
                        for (const s of S.groom.strands) { const d = Math.hypot(s.rootUV[0] - 0.47, (s.rootUV[1] - 0.42) * 0.4); if (d < bd) { bd = d; pick = s; } }
                        const wts = G2.guideWeights(O, pick.rootUV[0], pick.rootUV[1]), px = X(pick.rootUV[0]), py = Y(pick.rootUV[1]);
                        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
                        for (const [gi, wt] of wts) {
                            const g = gs[gi], gx = X((g.gx + 0.5) / O.guidesX), gy = Y((g.gz + 0.5) / O.guidesZ);
                            ctx.globalAlpha = 0.3 + 0.7 * wt; ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(gx, gy); ctx.stroke();
                            ctx.globalAlpha = 1; G2.label(ctx, px + 0.7 * (gx - px) + (gx > px ? 4 : -30), py + 0.7 * (gy - py) + (gy > py ? 12 : -4), wt.toFixed(2), '#ffffff');
                        }
                        ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(px, py, 3, 0, 2 * Math.PI); ctx.fill();
                    },
                },
                {
                    id: 'decompView', label: '插值分解：前排中间 3 条 guide 的 strand', sub: '左 → 右：4-guide 加权 · + clump · + curl / frizz（同一批发根）；白线 = guide', size: [360, 220], needs: ['decomp'],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0e12' }), third = w / 3;
                        const gsel = [3, 4, 5];   // gz = 0 行的中间三条 guide
                        const titles = ['Σ w_k·G_k', '+ clump', '+ curl / frizz'];
                        S.decomp.forEach((G, k) => {
                            const gs = G.guides.filter((g) => g.gz === 0 && gsel.includes(g.gx)), gi = gs.map((g) => G.guides.indexOf(g));
                            const cx = (gs[0].root[0] + gs[2].root[0]) / 2, cam = HAIR.ortho({ yaw: 0, pitch: 0, center: [cx, -7.2, 0], size: 16 });
                            ctx.save(); ctx.beginPath(); ctx.rect(k * third, 0, third, h); ctx.clip(); ctx.translate(k * third - w / 3, 0);
                            G2.drawPolys(ctx, cam, w, h, G.strands.filter((s) => gi.includes(s.guide)), { color: (s) => G2.GUIDE_COLORS[(G.guides[s.guide].gx) % 8], width: 0.7, alpha: 0.6 });
                            G2.drawPolys(ctx, cam, w, h, gs, { color: () => '#ffffff', width: 1.6, alpha: 0.9 });
                            ctx.restore();
                            G2.label(ctx, k * third + 6, 14, titles[k]);
                            if (k) { ctx.fillStyle = '#334'; ctx.fillRect(k * third - 0.5, 0, 1, h); }
                        });
                    },
                },
                {
                    id: 'interpLive', label: '运行时插值：guide 摆动 → strand 跟随', sub: 'guide 用解析摆动代替模拟；strand 每帧用同一组权重做位移转移', size: [360, 220], live: true, needs: ['interp'],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0e12' }), G = S.groom, N = G.opts.segs, cam = G2.camOf(S, { size: 21 });
                        const gs = G.guides, step = Math.max(1, Math.round(G.strands.length / 900));
                        const strands = [], guides = [];
                        for (let i = 0; i < G.strands.length; i += step) strands.push({ p: S.strandDef.subarray(3 * i * (N + 1), 3 * (i + 1) * (N + 1)), guide: G.strands[i].guide });
                        gs.forEach((g, i) => guides.push({ p: S.guideDef.subarray(3 * i * (N + 1), 3 * (i + 1) * (N + 1)), g }));
                        G2.drawPolys(ctx, cam, w, h, strands, { color: (s) => G2.GUIDE_COLORS[(gs[s.guide].gx + gs[s.guide].gz * 3) % 8], width: 0.5, alpha: 0.35 });
                        G2.drawPolys(ctx, cam, w, h, guides, { color: () => '#ffffff', width: 1.6, alpha: 0.9 });
                        G2.label(ctx, 6, 14, `τ = ${S.t.toFixed(2)} s · 显示 ${strands.length} / ${G.strands.length} strands`);
                    },
                },
            ],
            notes: '拖 <code>clump</code> 到 0：分解图中间一栏与左栏完全相同（纯加权，发丝均匀铺开）；拖到 1：发梢全部收向白色 guide，成簇感最强。<code>curl</code> / <code>frizz</code> 只改变右栏（每根独立的高频偏移），不影响 guide。运行时预览里 strand 的整体运动完全来自 guide，卷曲形状保持不变。<br>对照 UE：插值数据按<strong>点</strong>存最多 3 条最近 guide 的索引与权重（<code>FHairStrandsInterpolationDatas</code>；勾 <code>bUseUniqueGuide</code> 只用 1 条），Alembic for Grooms 里对应 <code>groom_closest_guides</code> / <code>groom_guide_weights</code>（各 3 个）；没有导入 guide 时可以按 Hair to Guide Density 从 strand 里生成。本页沿用 makeGroom 的 4-guide 双线性。',
            warn: [
                '位移转移这里用的是<strong>线性平移</strong>：guide 大幅弯曲时，rest 偏移（curl / 发根间距）不跟着旋转，卷发会被"剪切"变形。生产版按 guide 每段的局部帧（或 RBF / 三角形帧）旋转 rest 偏移。',
                '权重必须在资产构建时一次算好（resident），运行时只做加权；若每帧重新找最近 guide，guide 一动邻接关系就跳变，发丝会闪。',
            ],
        },
        {
            id: 'geom', title: '发丝几何：折线 → 面向相机的 ribbon；宽度 vs 像素', tag: 'GEOM',
            desc: `GPU 没有"带宽度的曲线"图元。常见两种画法：<strong>line 图元</strong>（每段一条 1 px 线，宽度由硬件决定，与真实粗细无关）和 <strong>camera-facing ribbon</strong>：每个控制点在 VS 里沿 <code>B = normalize(T × V)</code> 向两侧各推 r，相邻点连成三角形带，宽度是真实的世界尺寸。
                问题是尺度：根部直径 80 µm，在 1 px ≈ 1 mm 的画面里只有 <strong>0.08 px</strong>。硬件三角形光栅化只在像素中心采样，这么细的带几乎永远"擦不到"像素中心——整根发丝断成虚线甚至消失；而强制 ≥ 1 px 宽又让发量看起来粗十倍。
                所以 VS 通常把光栅化宽度钳到 ~1 px，同时把"加宽了多少倍"作为 <code>coverage = r / r_raster</code> 传下去，在后面的覆盖率 / 透明度里补偿回来（见下一步）。`,
            formula: [
                'B = normalize(T × V),   x_± = P ± r_raster · B        // VS：每点 2 顶点',
                'w_px = 2r · WidthScale · (H / size)                  // 正交：世界宽度 → 像素；透视再除以 view depth',
                'r_raster = max(r, r_min / k),  r_min ≈ 0.5–1 px,   coverage = r / r_raster  // 加宽了多少，覆盖率就按比例缩回',
            ],
            code: { title: 'HLSL · VS_HairStrands（ribbon 展开 + 最小像素宽度）', src: HLSL_RIBBON },
            previews: [
                {
                    id: 'ribbonView', label: 'ribbon 特写（VS 展开的真实三角形）', sub: '画面宽 1.2 cm，取景中心附近；三角形按真实半径 r（含宽度×）展开；灰格 = 主画面像素，点 = 像素中心（红 = 被某个三角形盖住）', size: [360, 220],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0b0e12' }), A = S.asset, cam0 = S.cam, R = S.res;
                        // 自动挑一个点：投影后离主画面 (0.5, 0.45) 最近的发丝中点
                        let best = null, bd = Infinity;
                        for (const s of A.strands) { const P = s.p, k = Math.floor(P.length / 6); const q = HAIR.project(cam0, [P[3 * k], P[3 * k + 1], P[3 * k + 2]], R, R); const d = Math.hypot(q[0] - R * 0.5, q[1] - R * 0.45); if (d < bd) { bd = d; best = [P[3 * k], P[3 * k + 1], P[3 * k + 2]]; } }
                        const cam = HAIR.ortho({ yaw: S.yaw, pitch: S.pitch, center: best, size: 1.2 * h / w }), V = cam.toViewer, kC = h / cam.size;
                        const qc = HAIR.project(cam0, best, R, R), pix = kC * (cam0.size / R);          // 主画面 1 px = pix 个特写像素
                        const toC = (X, Y) => [w / 2 - 0.5 + (X - qc[0]) * pix, h / 2 - 0.5 + (Y - qc[1]) * pix];   // 主画面像素坐标 → 特写坐标
                        const segs = [];   // 特写坐标下的中心线段 + 半径，用于像素中心命中测试
                        let drawn = 0;
                        for (const s of A.strands) {
                            if (drawn >= 60) break;
                            const P = s.p, n = P.length / 3, cl = [];
                            for (let k = 0; k < n; k++) cl.push(HAIR.project(cam, [P[3 * k], P[3 * k + 1], P[3 * k + 2]], w, h));
                            if (!cl.some((q) => q[0] > -10 && q[0] < w + 10 && q[1] > -10 && q[1] < h + 10)) continue;
                            drawn++;
                            const verts = [];
                            for (let k = 0; k < n; k++) {
                                const a = Math.max(0, k - 1), b = Math.min(n - 1, k + 1);
                                const T = HAIR.norm([P[3 * b] - P[3 * a], P[3 * b + 1] - P[3 * a + 1], P[3 * b + 2] - P[3 * a + 2]]);
                                const B = HAIR.norm(HAIR.cross(T, V)), r = (s.w0 + (s.w1 - s.w0) * k / (n - 1)) * 0.5 * S.widthScale;
                                for (const sd of [-1, 1]) verts.push(HAIR.project(cam, [P[3 * k] + sd * r * B[0], P[3 * k + 1] + sd * r * B[1], P[3 * k + 2] + sd * r * B[2]], w, h));
                                if (k < n - 1) segs.push([cl[k], cl[k + 1], r * kC]);
                            }
                            const col = G2.GUIDE_COLORS[s.guide % 8];
                            for (let k = 0; k + 3 < verts.length; k += 2) {
                                const [a, b, c2, d] = [verts[k], verts[k + 1], verts[k + 2], verts[k + 3]];
                                ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(d[0], d[1]); ctx.lineTo(c2[0], c2[1]); ctx.closePath();
                                ctx.globalAlpha = 0.55; ctx.fillStyle = col; ctx.fill(); ctx.globalAlpha = 0.9; ctx.strokeStyle = col; ctx.lineWidth = 0.5; ctx.stroke();
                                ctx.beginPath(); ctx.moveTo(b[0], b[1]); ctx.lineTo(c2[0], c2[1]); ctx.stroke();   // 对角线：每段 2 个三角形
                            }
                        }
                        ctx.globalAlpha = 1;
                        // 主画面像素网格与像素中心
                        const X0 = Math.floor(qc[0] - w / 2 / pix) - 1, X1 = Math.ceil(qc[0] + w / 2 / pix) + 1, Y0 = Math.floor(qc[1] - h / 2 / pix) - 1, Y1 = Math.ceil(qc[1] + h / 2 / pix) + 1;
                        ctx.strokeStyle = 'rgba(255,255,255,0.16)'; ctx.lineWidth = 1;
                        for (let X = X0; X <= X1; X++) { const x = toC(X + 0.5, 0)[0]; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
                        for (let Y = Y0; Y <= Y1; Y++) { const y = toC(0, Y + 0.5)[1]; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
                        let hit = 0, tot = 0;
                        for (let Y = Y0; Y <= Y1; Y++) for (let X = X0; X <= X1; X++) {
                            const [x, y] = toC(X, Y); if (x < 0 || y < 0 || x > w || y > h) continue;
                            tot++;
                            const on = segs.some(([a, b, r]) => { const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1e-9, u = HAIR.clamp(((x - a[0]) * dx + (y - a[1]) * dy) / L2, 0, 1); return Math.hypot(a[0] + u * dx - x, a[1] + u * dy - y) < r; });
                            if (on) hit++;
                            ctx.beginPath(); ctx.arc(x, y, on ? 3 : 2, 0, 2 * Math.PI); ctx.fillStyle = on ? '#ff4d4d' : 'rgba(200,210,225,0.55)'; ctx.fill();
                        }
                        ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(0, 0, w, 18);
                        G2.label(ctx, 6, 13, `${drawn} 根 · 1 px = ${(10 * cam0.size / R).toFixed(2)} mm · 像素中心命中 ${hit} / ${tot}`);
                    },
                },
                {
                    id: 'widthPlot', label: '发丝宽度（像素）vs 取景尺寸', sub: '正交相机：画面高 size cm、H 像素；竖线 = 当前取景', size: [360, 180],
                    render: (c, S) => {
                        const R = S.res, ws = S.widthScale;
                        PL.draw.plot(c, {
                            x: [1, 200], y: [0.005, 20], logX: true, logY: true, xLabel: '画面高 size (cm)', yLabel: '宽度 (px)',
                            series: [
                                { fn: (x) => 0.008 * ws * R / x, color: '#e8c25a', label: '根 80 µm' },
                                { fn: (x) => 0.003 * ws * R / x, color: '#4a8aff', label: '梢 30 µm' },
                            ],
                            hmarks: [{ y: 1, label: '1 px', color: '#e86a5a' }],
                            marks: [{ x: 18, label: '当前 18 cm', color: '#5ae87a' }, { x: 0.008 * ws * R, label: '根 = 1 px', color: '#8a96a8' }],
                        });
                    },
                },
            ],
            notes: '把 <code>宽度×</code> 拖到 12：ribbon 变成实心带，根部宽度接近 1 px——这才是"粗到硬件能稳定采样"的尺度。<code>分辨率</code>从 160 改到 220，宽度曲线整体上移：分辨率越高（或相机越近），越接近能直接光栅化的区间。',
            warn: [
                '<code>T × V</code> 在发丝几乎指向相机时退化（|T × V| → 0），B 方向乱跳；VS 里要对长度做下限保护或改用屏幕空间的垂线。',
                '每条曲线最后一个点不能再和下一条曲线的第一个点连三角形——索引缓冲 / strip restart 或按点标记"曲线末端"必须处理，否则会画出横跨两根发丝的长三角形。',
            ],
        },
        {
            id: 'alias', title: '核心问题：比像素细 10 倍的发丝怎么走样', tag: 'ALIAS',
            desc: `两种"直接画"的做法都错，而且错的方向相反：
                <strong>① 1 px 硬线</strong>（line 图元，或把 ribbon 钳到 1 px 却不补偿覆盖率）——每根发丝都被画成 1 px 宽且不透明，发量看起来厚 10 倍、边缘锯齿；
                <strong>② 真实宽度 + 像素中心采样</strong>（普通三角形光栅化）——0.08 px 宽的带只有约 8% 的概率盖住某个像素中心，发丝断成随机的点，期望值虽然对，方差巨大，发丝或相机一动就闪烁。
                参考解：在自动挑出的 40×40 crop（α ≈ 0.25 的稀疏区域，发丝彼此分离）里每像素 16×16 = 256 个子样本做二值覆盖，得到每像素真实被盖住的面积比例。`,
            formula: [
                'α_ref(p) = (1/256) · Σ_{s∈p} [ ∃ strand：dist(x_s, strand) < r_px ]     // 16×16 子样本，二值并集',
                'E[ α_center ] ≈ α_ref,   Var[ α_center ] = α_ref·(1 − α_ref)            // 中心采样：期望对，方差大',
                'α_1px ≈ 1 − Π_s (1 − 1)  ≈ 1  (只要擦到)                                // 1 px 硬线：系统性高估',
            ],
            previews: [
                {
                    id: 'naiveFull', label: '① 1 px 硬线（全图）', sub: '每根发丝画成 1 px、不透明；白框 = 自动挑出的 crop', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, S.aNaive, S.res, S.res, { range: [0, 1] }); G2.cropBox(c, S); },
                },
                {
                    id: 'centerFull', label: '② 真实宽度 + 中心采样（全图）', sub: '普通三角形光栅化：只有盖住像素中心才算', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, S.aCenter, S.res, S.res, { range: [0, 1] }); G2.cropBox(c, S); },
                },
                {
                    id: 'refCrop', label: '参考 α_ref（crop · 256 spp）', sub: '每像素 16×16 二值子样本的覆盖比例', needs: ['alias'],
                    render: (c, S) => PL.draw.field(c, S.ref, S.crop.w, S.crop.h, { range: [0, 1] }),
                    probe: (x, y, S) => `α_ref = ${S.ref[y * S.crop.w + x].toFixed(3)}`,
                },
                {
                    id: 'naiveErr', label: '① − 参考', sub: '', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, G2.diffMap(S.cNaive, S.ref), S.crop.w, S.crop.h, { symmetric: true, range: [-1, 1] }); G2.statSub(c, S.stats.naive); },
                    probe: (x, y, S) => G2.probeErr(S.cNaive, S, x, y),
                },
                {
                    id: 'centerErr', label: '② − 参考', sub: '', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, G2.diffMap(S.cCenter, S.ref), S.crop.w, S.crop.h, { symmetric: true, range: [-1, 1] }); G2.statSub(c, S.stats.center); },
                    probe: (x, y, S) => G2.probeErr(S.cCenter, S, x, y),
                },
            ],
            notes: '误差图（±1）：橙 = 覆盖高估，蓝 = 漏掉，深灰 = 准确。① 几乎全橙（覆盖总量是参考的数倍，见标签里的 Σ 比值）；② 橙蓝交错（个别像素被打满，其余全漏）。把 <code>宽度×</code> 拖到 8 以上，两种做法都逐渐接近参考——走样问题本质上是"宽度 ≪ 1 px"。',
            warn: '② 的"期望正确"只在静止画面里取平均才成立；实际每帧都是一张新的随机二值图，TAA 来不及平均就会看到严重闪烁（下一步会量化）。',
        },
        {
            id: 'cover', title: '解析亚像素覆盖：coverage / alpha 缓冲与顺序无关累积', tag: 'COVER',
            desc: `不再问"像素中心有没有被盖住"，而是让每根发丝<strong>按它真实盖住的面积</strong>给出覆盖率 c ∈ [0,1]：光栅化时把 ribbon 至少加宽到覆盖核的半径（1 px）让每个可能被影响的像素都跑一次 PS，PS 里用像素中心到发丝中线的距离 d 解析地算 c。
                <code>HAIR.raster</code> 用的核：宽于 1 px 时是盒滤波的线性边缘；窄于 1 px 时是半径 1 px 的帐篷核，横向积分恰好等于像素宽度 w_px——<strong>能量守恒</strong>，不论发丝落在像素的哪里。
                多根发丝叠加时，假设它们在像素内位置互不相关，透射率相乘：<code>α = 1 − Π(1 − c)</code>，乘法与顺序无关，可以直接用乘法混合（Dst ·= 1 − c）写进一张 R16F。
                对照两种常见近似：<strong>只留最近一根</strong>（单样本 visibility buffer 只存最前面那根 strand 的覆盖率）和 <strong>Σc 截断</strong>（加法混合后 saturate）。AMD TressFX 是同一类做法：VS 把 quad 额外外扩约 0.71 px，PS 里用 <code>ComputeCoverage</code> 解析求覆盖率。`,
            formula: [
                'c(d) = 2 r_px · max(0, 1 − d)          (r_px < 0.5)   // 帐篷核：∫ c dd = 2 r_px = w_px',
                'c(d) = saturate(r_px + 0.5 − d)        (r_px ≥ 0.5)   // 盒滤波边缘',
                'α_OIT = 1 − Π_s (1 − c_s)              // 每根 strand 取其各段的 max，再相乘（与顺序无关）',
                'α_near = c_{nearest}   ·   α_Σ = min(1, Σ_s c_s)',
            ],
            code: {
                title: 'HLSL · PS_HairCoverage（解析覆盖 + 透射率乘法混合）', src: `
                    // Blend：Dst = Dst · (1 − Src.a)，RT 清成 1 → 结果 = Π(1 − c_s)，与绘制顺序无关
                    // VS 把光栅化半宽钳到 ≥ 1 px（覆盖核半径），并传下真实像素半径 TrueRadiusPx
                    float4 PS_HairCoverage(VSOut i) : SV_Target {
                        float d   = abs(i.Side) * i.RasterRadiusPx;   // 像素中心到发丝中线（px）
                        float rPx = i.TrueRadiusPx;
                        float c   = rPx >= 0.5 ? saturate(rPx + 0.5 - d)       // 宽：盒滤波边缘
                                               : 2.0 * rPx * saturate(1.0 - d); // 窄：帐篷核，积分 = 宽度
                        return float4(0, 0, 0, c);
                    }
                    // Composition：α = 1 − Transmittance；SceneColor = α · HairColor + (1 − α) · Scene` },
            previews: [
                {
                    id: 'alphaFull', label: 'α_OIT = 1 − Π(1 − c)（全图）', sub: 'HAIR.raster 的 alpha 缓冲', smooth: false,
                    render: (c, S) => { PL.draw.field(c, S.R.alpha, S.res, S.res, { range: [0, 1] }); G2.cropBox(c, S); },
                    probe: (x, y, S) => { const i = y * S.res + x; return `α = ${S.R.alpha[i].toFixed(3)}  Σc = ${S.R.cov[i].toFixed(3)}  nearest id = ${S.R.id[i]}`; },
                },
                {
                    id: 'covFull', label: 'Σc（全图，未截断）', sub: '每像素覆盖率之和 ≈ 穿过的"发丝像素宽度"总和',
                    render: (c, S) => PL.draw.field(c, S.R.cov, S.res, S.res, { map: 'inferno', range: [0, 3] }),
                    probe: (x, y, S) => `Σc = ${S.R.cov[y * S.res + x].toFixed(3)}`,
                },
                {
                    id: 'alphaErr', label: 'α_OIT − 参考', sub: '', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, G2.diffMap(S.cAlpha, S.ref), S.crop.w, S.crop.h, { symmetric: true, range: [-1, 1] }); G2.statSub(c, S.stats.alpha); },
                    probe: (x, y, S) => G2.probeErr(S.cAlpha, S, x, y),
                },
                {
                    id: 'nearErr', label: 'α_near − 参考（只留最近一根）', sub: '', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, G2.diffMap(S.cNear, S.ref), S.crop.w, S.crop.h, { symmetric: true, range: [-1, 1] }); G2.statSub(c, S.stats.near); },
                    probe: (x, y, S) => G2.probeErr(S.cNear, S, x, y),
                },
                {
                    id: 'sumErr', label: 'α_Σ − 参考（Σc 截断）', sub: '', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, G2.diffMap(S.cSum, S.ref), S.crop.w, S.crop.h, { symmetric: true, range: [-1, 1] }); G2.statSub(c, S.stats.sum); },
                    probe: (x, y, S) => G2.probeErr(S.cSum, S, x, y),
                },
            ],
            notes: '同一色标（±1）下，α_OIT 的误差图几乎是灰的；α_near 整片偏蓝（后面的发丝全丢了，见 Σ 比值）；α_Σ 在密集处偏橙（把重叠部分算了两次）。把 <code>clump</code> 拖到 1：发丝在像素内高度相关（挤在一起互相遮挡），"互不相关"假设变差，α_OIT 也开始偏橙。',
            warn: [
                '同一根 strand 相邻两段的 ribbon 在关节处重叠，用硬件混合会对同一根发丝乘两次 (1 − c)，关节处偏暗；<code>HAIR.raster</code> 用"每根 strand 取 max"避免了这点，硬件路径通常靠 strip 拓扑 + 关节处斜接，或在 compute 光栅化里按 strand 去重。',
                '解析覆盖只解决了"盖住多少"，没有解决"看到谁"：颜色仍要按深度排序合成（OIT 的颜色部分），这正是下一步 visibility buffer / 样本压缩要做的事。',
            ],
        },
        {
            id: 'vis', title: '引擎做法：visibility buffer + 样本压缩、MSAA、TAA，硬件 vs 软件光栅化', tag: 'VIS',
            desc: `生产引擎把"看到谁"和"盖住多少"拆开：
                <strong>可见性 pass</strong> 用 MSAA（每像素 4–8 个样本）只写 strand / segment id 和深度，不着色；
                <strong>压缩（compaction）pass</strong> 逐像素把 8 个样本去重成"唯一 strand 列表 + 各自样本占比"，再乘上另外累积的总覆盖率，得到每像素少量（通常 1–3 个）带权重的着色样本——着色次数 ∝ 唯一样本数，而不是 MSAA 样本数或发丝片元数；
                也有 PPLL / K-buffer 路线（AMD TressFX：逐像素链表保留最近的 K 个片元做排序混合）。
                但 MSAA 样本本身仍是点采样：真实宽度 0.08 px 时 8 个样本照样大量漏掉，所以要么把光栅化宽度钳到 ≥ 1 px（再用覆盖率补偿），要么靠<strong>时域抖动 + 累积</strong>（TAA / TSR）把点采样平均成面积。
                对这种又细又长的三角形，硬件光栅化效率很低（2×2 quad 打包、每个三角形只盖几个像素），所以较新的实现把远处 / 细小的发丝放到 <strong>compute 软件光栅化</strong>：每 thread 一个 segment，直接用 64-bit atomic 写 visibility buffer。`,
            formula: [
                'α_MSAA = (1/8) · Σ_{s=1..8} [ 样本 s 被盖住 ]          // D3D 标准 8× 样本位置',
                'w_k = α · n_k / Σ_j n_j                                  // 压缩：唯一 strand k 占 n_k 个样本',
                'α_TAA(N) = (1/N) · Σ_{f=1..N} [ x_f 被盖住 ],  x_f = 像素中心 + Halton(2,3)_f',
                'RMSE(N) ≈ √( α(1 − α) / N )                               // 点采样的方差只按 1/√N 下降',
            ],
            code: {
                title: 'HLSL · CS_HairVisibilityCompaction（MSAA 样本 → 去重样本列表）', src: `
                    // 着色样本：哪一段发丝、哪个像素、占该像素的权重，外加着色 pass 要用的世界切线
                    struct FHairSample { uint SegmentId; uint2 Pixel; float Weight; float3 Tangent; };

                    Texture2DMS<uint, 8>          HairVisibilityTex;      // 每样本：segment id（~0u = 空）
                    Texture2D<float>              HairCoverageTex;        // 解析覆盖累积出的总 α（上一步）
                    StructuredBuffer<float4>      DeformedPosition;       // 本帧发丝点（插值 CS 的输出）
                    RWStructuredBuffer<FHairSample> HairSampleList;       // { segId, pixel, weight, tangent }
                    RWTexture2D<uint2>            HairSampleOffsetCount;  // 每像素 (offset, count)
                    RWBuffer<uint>                SampleCounter;

                    // segment id = 该段起点的全局点号 p：段 p → p+1 一定在同一条曲线内（VS 不跨曲线连三角形），切线 = 段方向（根 → 梢）
                    FHairSample MakeSample(uint segId, uint2 pixel, float weight) {
                        FHairSample s;
                        s.SegmentId = segId;
                        s.Pixel     = pixel;
                        s.Weight    = weight;
                        s.Tangent   = normalize(DeformedPosition[segId + 1].xyz - DeformedPosition[segId].xyz);
                        return s;
                    }

                    [numthreads(8, 8, 1)]
                    void CS_HairCompaction(uint3 id : SV_DispatchThreadID) {
                        uint ids[8], cnt[8], n = 0, total = 0;
                        [unroll] for (uint s = 0; s < 8; s++) {
                            uint v = HairVisibilityTex.Load(id.xy, s);
                            if (v == ~0u) continue;
                            uint k = 0; while (k < n && ids[k] != v) k++;    // 去重
                            if (k == n) { ids[n] = v; cnt[n] = 0; n++; }
                            cnt[k]++; total++;
                        }
                        uint base; InterlockedAdd(SampleCounter[0], n, base);
                        float alpha = HairCoverageTex[id.xy];
                        for (uint k = 0; k < n; k++)
                            HairSampleList[base + k] = MakeSample(ids[k], id.xy, alpha * cnt[k] / max(total, 1));
                        HairSampleOffsetCount[id.xy] = uint2(base, n);
                    }` },
            previews: [
                {
                    id: 'msaaErr', label: 'MSAA 8×（真实宽度）− 参考', sub: '', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, G2.diffMap(S.cMsaa, S.ref), S.crop.w, S.crop.h, { symmetric: true, range: [-1, 1] }); G2.statSub(c, S.stats.msaa); },
                    probe: (x, y, S) => G2.probeErr(S.cMsaa, S, x, y),
                },
                {
                    id: 'msaaWErr', label: 'MSAA 8× + 钳到 1 px（不补偿）− 参考', sub: '', needs: ['alias'],
                    render: (c, S) => { PL.draw.field(c, G2.diffMap(S.cMsaaW, S.ref), S.crop.w, S.crop.h, { symmetric: true, range: [-1, 1] }); G2.statSub(c, S.stats.msaaW); },
                    probe: (x, y, S) => G2.probeErr(S.cMsaaW, S, x, y),
                },
                {
                    id: 'taaCur', label: 'TAA · 本帧（1 个抖动样本）', sub: '二值：盖住 / 没盖住', live: true, needs: ['taa'],
                    render: (c, S) => PL.draw.field(c, S.taaCur, S.crop.w, S.crop.h, { range: [0, 1] }),
                },
                {
                    id: 'taaMean', label: 'TAA · 累积均值', sub: '', live: true, needs: ['taa'],
                    render: (c, S) => { PL.draw.field(c, S.taaMean, S.crop.w, S.crop.h, { range: [0, 1] }); G2.setSub(c, `N = ${S.taaN} 帧 · RMSE = ${S.taaErr[S.taaErr.length - 1].toFixed(3)}`); },
                    probe: (x, y, S) => `α_TAA = ${S.taaMean[y * S.crop.w + x].toFixed(3)}  α_ref = ${S.ref[y * S.crop.w + x].toFixed(3)}`,
                },
                {
                    id: 'taaPlot', label: 'RMSE vs 累积帧数', sub: '对照：解析覆盖 α_OIT（一帧）与 MSAA 8×', live: true, needs: ['taa'], size: [360, 180],
                    render: (c, S) => {
                        const e = S.taaErr, n = e.length, a0 = S.stats.alpha.rmse;
                        PL.draw.plot(c, {
                            x: [1, Math.max(16, n)], y: [0.005, 1], logX: true, logY: true, xLabel: '帧数 N', yLabel: 'RMSE',
                            series: [
                                { data: e.slice(), x: e.map((_, i) => i + 1), color: '#e8c25a', label: 'TAA 1 spp 累积' },
                                { fn: (x) => e[0] / Math.sqrt(x), color: '#8a96a8', dash: [4, 3], label: '∝ 1/√N' },
                            ],
                            hmarks: [{ y: a0, label: `α_OIT ${a0.toFixed(3)}`, color: '#5ae87a' }, { y: S.stats.msaa.rmse, label: `MSAA 8× ${S.stats.msaa.rmse.toFixed(3)}`, color: '#4a8aff' }],
                        });
                    },
                },
            ],
            notes: 'MSAA 8×（真实宽度）比中心采样好但依然噪声很大——8 个点对 0.08 px 的带仍然是"彩票"；钳到 1 px 不补偿则整体偏橙（等价于 ① 的 8 样本版）。TAA 预览随播放逐帧累积：误差按 1/√N 慢慢下降，要 (RMSE₁ / RMSE_OIT)² ≈ 几十帧才追上<strong>一帧</strong>的解析覆盖（图中绿线）——而且只在完全静止时成立，发丝一动 history 就失效。所以引擎的组合是：<strong>解析覆盖率管"多少"</strong>，<strong>少量可见性样本管"谁"</strong>，TAA 只负责收拾残余噪声。',
            warn: [
                'UE 的 Groom 渲染（Hair Strands）就是这一类。UE 5.4 文档里 primary visibility 有三种模式：MSAA 可见性缓冲（默认；<code>r.HairStrands.Visibility.MSAA.SamplePerPixel</code> = 2 / 4 / 8，代码默认 8，scalability 配置按画质改成 1 / 4 / 8）、PPLL（<code>r.HairStrands.Visibility.PPLL</code>，每像素最多着色若干节点、最后一个节点聚合更远的发丝）和 compute 软件光栅化（<code>r.HairStrands.Visibility.ComputeRaster</code>，5.4 中标为 Experimental）；可见样本随后被分配成供光照使用的 hair sample。Frostbite（Tafuri 2019）同样是 visibility buffer + MSAA，并且"相似样本只着色一次"，约快 2×。cvar 随版本变化，以所用版本的源码为准。',
                '软件光栅化写 visibility buffer 时，深度 + id 必须打包进一个 64-bit 值再 <code>InterlockedMin</code>，否则多个 thread 同时写同一像素会出现 id 与深度不匹配。',
            ],
        },
        {
            id: 'shade', title: '着色：光栅化缓冲里的切线 + Chiang BSDF，随机 h + 时域累积', tag: 'SHADE',
            desc: `发丝没有法线，着色只需要<strong>切线 T</strong>：可见性 / 光栅化 pass 顺手把最近那根 strand 的世界切线写进缓冲（这里是 <code>HAIR.raster</code> 的 <code>tan</code>），着色 pass 用它建 fiber 帧（x = T，n ⟂ T 且朝向 wo），把 wi、wo 转到局部坐标后调 <code>HAIR.bsdf</code>。
                Chiang / pbrt-v3 的 S(wi, wo; h) 依赖光线在发丝横截面上的入射偏移 h。发丝 ≪ 1 px 时，一个像素看到的是<strong>整个宽度上的平均</strong>（远场）；在着色里直接做 h 的数值积分要几十上百次求值，所以每像素每帧只取<strong>一个随机 h</strong>，交给时域累积去平均——和引擎里让 TAA / TSR 消化随机采样噪声是同一个思路。
                光照只用一盏方向光（L_o = S·E，系列约定里投影宽度已折进 S），没有阴影：自阴影与多次散射是 03 篇的主题。`,
            formula: [
                'f = fiberFrame(T, wo):  x = T,  n ⟂ T 朝向 wo,  b = T × n',
                'L_f(p) = S( toFiber(wi), toFiber(wo); h_f ) · E,    h_f ~ U(−1, 1)  // 每像素每帧一个',
                'L̄_N(p) = (1/N) Σ_{f=1..N} L_f(p)  →  ½∫_{−1}^{1} S(wi, wo; h) dh · E   // history 累积 = 远场平均',
                'C(p) = α(p) · L̄_N(p) + (1 − α(p)) · C_bg      // α 来自解析覆盖（第 05 步）',
            ],
            code: {
                title: 'HLSL · CS_HairShading（每个压缩样本一个 thread）', src: `
                    StructuredBuffer<FHairSample>   HairSampleList;
                    RWStructuredBuffer<float3>      HairSampleLighting;
                    cbuffer HairLight { float3 LightDir; float3 LightIrradiance; float3 CameraForward; uint FrameIndex; float3 HairColor; };   // HairColor = 顶栏"发色"（线性反照率）

                    [numthreads(64, 1, 1)]
                    void CS_HairShading(uint3 id : SV_DispatchThreadID) {
                        FHairSample s  = HairSampleList[id.x];
                        float3 T  = s.Tangent;                         // 光栅化时写入的世界切线（根 → 梢）
                        float3 wo = -CameraForward, wi = LightDir;
                        float3 n  = normalize(wo - dot(wo, T) * T);    // fiber 帧：n ⟂ T 且朝向 wo（φo = 0）
                        float3 b  = cross(T, n);
                        float3 wiL = float3(dot(wi, T), dot(wi, n), dot(wi, b));
                        float3 woL = float3(dot(wo, T), dot(wo, n), dot(wo, b));
                        float  h  = 2.0 * Hash(s.Pixel, FrameIndex) - 1.0;   // 每像素每帧一个 h（系列共享的 PCG 哈希）
                        // 与 JS 的 HAIR.bsdf 同一组参数：发色 → σa，βm = βn = 0.3，α = 2°；输入全是 uniform，生产里每个材质在 CPU 上算一次放进 cbuffer
                        HairParams P = MakeHairParams(SigmaAFromReflectance(HairColor, 0.3), 0.3, 0.3, 2.0);
                        HairSampleLighting[id.x] = HairBSDF_Chiang(wiL, woL, h, P) * LightIrradiance;  // L_o = S·E
                    }` },
            previews: [
                {
                    id: 'tanBuf', label: '切线缓冲 T（最近 strand）', sub: 'RGB = 0.5 + 0.5·T；黑 = 没有可见性样本', needs: ['shadeSetup'],
                    render: (c, S) => { const R = S.res, Rb = S.R; PL.draw.pixels(c, R, R, (x, y, out) => { const i = y * R + x; if (Rb.id[i] < 0) { out[0] = out[1] = out[2] = 0; return out; } for (let k = 0; k < 3; k++) out[k] = 0.5 + 0.5 * Rb.tan[3 * i + k]; return out; }); },
                    probe: (x, y, S) => { const i = y * S.res + x, t = S.R.tan; return S.R.id[i] < 0 ? '无样本' : `T = (${t[3 * i].toFixed(2)}, ${t[3 * i + 1].toFixed(2)}, ${t[3 * i + 2].toFixed(2)})  strand ${S.R.id[i]}  v = ${S.R.v[i].toFixed(2)}`; },
                },
                {
                    id: 'shadeOne', label: '本帧：每像素 1 个随机 h', sub: 'C = α·L_f + (1 − α)·bg', live: true, needs: ['accum'],
                    render: (c, S) => G2.drawShade(c, S, S.shadeCur, 1),
                },
                {
                    id: 'shadeAcc', label: 'history 累积 L̄_N', sub: '', live: true, needs: ['accum'], large: true,
                    render: (c, S) => { G2.drawShade(c, S, S.accSum, 1 / Math.max(1, S.accN)); G2.setSub(c, `N = ${S.accN} 帧`); },
                    probe: (x, y, S) => { const i = y * S.res + x, n = Math.max(1, S.accN), a = S.accSum; return `α = ${S.R.alpha[i].toFixed(3)}  L̄ = (${(a[3 * i] / n).toFixed(3)}, ${(a[3 * i + 1] / n).toFixed(3)}, ${(a[3 * i + 2] / n).toFixed(3)})  N = ${S.accN}`; },
                },
            ],
            notes: '播放时"本帧"每一帧都是新的噪声图（同一根发丝上相邻像素的 h 不同，R / TRT 高光一闪一闪），累积图则稳定收敛。把 <code>光方位</code> 转到背后（±180°）：背光的 TT（透射）让稀疏的发梢整体发亮；正面光时主要是白色的 R 高光和带发色的 TRT 二次高光。改 <code>发色</code> 会重建 σa 并清空 history。',
            warn: [
                '发丝比像素宽时（特写、<code>宽度×</code> 很大）h 是有意义的近场量：应该用光栅化得到的真实横向偏移（<code>HAIR.raster</code> 的 <code>h</code> 缓冲），不能再随机。两者之间要按像素宽度过渡。',
                '有覆盖率但没有可见性样本的像素（离发丝 0.5–1 px）也必须有颜色：这里借用 3×3 邻域里最近的样本，否则 α 合成后发丝边缘会出现一圈黑边。',
            ],
        },
        {
            id: 'lod', title: 'LOD：减少 strand、加宽剩下的以保住覆盖率；远处换 card / mesh', tag: 'LOD',
            desc: `远处一个像素里挤着几十根发丝，逐根光栅化纯属浪费。<strong>strand decimation</strong>：按每根 strand 的固定哈希保留比例 f，其余整根剔除（哈希固定 → f 连续变化时发丝按固定顺序消失，不会整片跳变）；
                剩下的宽度乘 1/f，使总覆盖 Σc 在期望上不变。只要加宽后的发丝仍 ≪ 1 px，α = 1 − Π(1 − c) 也近似不变；当 w_px / f 超过 ~1 px，发丝就不再是亚像素的：画面变成"更少、更粗"的发丝，簇结构被抹平，覆盖统计也随之偏离（下图在 f 很小时反而偏高）——这时就该切换到 <strong>hair card / mesh</strong>（见 <a href="hair_04_cards.html">04 篇</a>）。
                引擎里通常还会对每条曲线做点数抽稀（vertex decimation，按弯曲角阈值保点），并把 LOD 选择做在 GPU 上的簇（cluster）粒度。`,
            formula: [
                'keep(s) = [ hash(s) < f ],    r′ = r / f              // 固定哈希 → 稳定的剔除顺序',
                'E[ Σc′ ] = f · N · (w_px / f) = N · w_px = Σc          // 帐篷核：c ∝ 宽度，总覆盖守恒',
                'α′ = 1 − (1 − c/f)^{fN}  ≈  1 − e^{−N c} = α          (c/f ≪ 1)',
                'f_min ≈ w_px   // 再小：加宽后的发丝 > 1 px，不再是亚像素（"更少更粗"，簇结构被抹平）',
            ],
            previews: [
                {
                    id: 'lodPlain', label: '只剔除（不加宽）', sub: '', needs: ['lod'],
                    render: (c, S) => { PL.draw.field(c, S.lodPlain, S.res, S.res, { range: [0, 1] }); G2.setSub(c, `${S.lodKept} / ${S.asset.nC} strands`); },
                },
                {
                    id: 'lodWide', label: '剔除 + 宽度 × 1/f', sub: '', needs: ['lod'],
                    render: (c, S) => { PL.draw.field(c, S.lodWide, S.res, S.res, { range: [0, 1] }); G2.setSub(c, `宽度 × ${(1 / S.keep).toFixed(2)}`); },
                },
                {
                    id: 'lodErr', label: '加宽版 − 完整 groom（α）', sub: '', needs: ['lod'],
                    render: (c, S) => { const d = G2.diffMap(S.lodWide, S.R.alpha); PL.draw.field(c, d, S.res, S.res, { symmetric: true, range: [-0.5, 0.5] }); G2.statSub(c, G2.errStats(S.lodWide, S.R.alpha)); },
                },
                {
                    id: 'lodPlot', label: '覆盖率保持：mean α(f) / mean α(1)', sub: '96² 上扫 f；竖线 = 当前 LOD 保留与 f_min ≈ w_px', needs: ['lod'], size: [360, 180],
                    render: (c, S) => {
                        const fs = S.lodFs, wpx = 0.0055 * S.widthScale * 96 / S.cam.size;   // 平均宽度（根 80 / 梢 30 µm）在 96² 上的像素宽
                        PL.draw.plot(c, {
                            x: [0.02, 1], y: [0, 1.2], logX: true, xLabel: '保留比例 f', yLabel: '相对 α',
                            series: [
                                { data: S.lodCurves[0], x: fs, color: '#e86a5a', label: '不加宽' },
                                { data: S.lodCurves[2], x: fs, color: '#4a8aff', label: '× 1/√f' },
                                { data: S.lodCurves[1], x: fs, color: '#5ae87a', label: '× 1/f' },
                            ],
                            hmarks: [{ y: 1, color: '#8a96a8' }],
                            marks: [{ x: S.keep, label: `f = ${S.keep}`, color: '#e8c25a' }, { x: Math.max(0.02, Math.min(1, wpx)), label: `f_min ≈ w_px = ${wpx.toFixed(3)}`, color: '#a45fa5' }],
                        });
                    },
                },
            ],
            notes: '拖 <code>LOD 保留</code>：红线（只剔除）几乎正比于 f 下降，发量肉眼可见变稀；绿线（× 1/f）在大部分区间贴着 1，只在 f 很小（加宽后超过 ~1 px）时偏离——这里是偏高：变宽的发丝把覆盖从簇里摊到了簇与簇之间，而完整 groom 的覆盖集中在簇里、彼此重叠。<code>宽度×</code> 调大会让 f_min 右移——发丝越粗（或离得越近），能抽掉的越少。<br>对照 UE：<code>FHairLODSettings</code> 正是这套参数——<code>CurveDecimation</code>（保留比例 f）、<code>VertexDecimation</code> + <code>AngularThreshold</code>（点抽稀）、<code>ThicknessScale</code>（补偿曲线减少的加粗系数，对应这里的 1/f）、<code>ScreenSize</code>，以及 <code>GeometryType</code> = Strands / Cards / Meshes。',
            warn: [
                '剔除必须按<strong>固定的 per-strand 哈希</strong>，并且 guide 插值、模拟都在剔除之前完成；若每帧随机挑 strand 或按屏幕位置挑，LOD 过渡会闪烁。',
                '加宽保住的是覆盖率，不是外观：剩下的发丝更粗，自阴影 / 透射（03 篇）和高光的细碎感都会变。f 很小时干脆切 card（04 篇），比把 strand 加宽到 1 px 以上更省也更稳。',
            ],
        },
        {
            id: 'prod', title: '教学版 → 生产版，与常见坑', tag: 'PROD',
            blocks: [
                {
                    type: 'html', html: `<table style="width:100%;border-collapse:collapse;font-size:0.82rem;line-height:1.5">
                        <tr style="color:#8a96a8;text-align:left"><th style="padding:4px 8px">环节</th><th style="padding:4px 8px">本页（教学版）</th><th style="padding:4px 8px">生产版</th></tr>
                        <tr><td style="padding:4px 8px">guide 与模拟</td><td style="padding:4px 8px">16 条 guide，解析摆动；线性位移转移</td><td style="padding:4px 8px">guide 在 GPU 上解 position-based / Cosserat 约束（UE：Niagara），带碰撞与风；render strand 按 guide 局部帧旋转转移，并绑定到蒙皮网格（root UV → 三角形）</td></tr>
                        <tr><td style="padding:4px 8px">资产</td><td style="padding:4px 8px">float32 位置，每点 3 个属性</td><td style="padding:4px 8px">位置量化（相对包围盒 16-bit 或 half）、半径 / u 打包进一个 uint；曲线按簇（cluster）分组，供剔除与 LOD</td></tr>
                        <tr><td style="padding:4px 8px">剔除 / LOD</td><td style="padding:4px 8px">全局保留比例 f + 宽度 1/f</td><td style="padding:4px 8px">GPU 上按簇做视锥剔除与屏幕尺寸 LOD，曲线和点同时抽稀，输出间接绘制参数；远处切 card / mesh</td></tr>
                        <tr><td style="padding:4px 8px">光栅化</td><td style="padding:4px 8px">JS 逐段距离测试，正交相机</td><td style="padding:4px 8px">近处 VS ribbon + MSAA 可见性；远处 / 细小的发丝走 compute 软件光栅化（64-bit atomic 写 id + 深度）</td></tr>
                        <tr><td style="padding:4px 8px">覆盖与可见性</td><td style="padding:4px 8px">每像素 1 个最近样本 + 解析 α</td><td style="padding:4px 8px">MSAA 样本压缩成每像素 1–3 个唯一样本（或 PPLL / K-buffer），权重 = 样本占比 × 总覆盖</td></tr>
                        <tr><td style="padding:4px 8px">光照</td><td style="padding:4px 8px">一盏方向光，无阴影</td><td style="padding:4px 8px">deep opacity map / 体素化密度做自阴影与透射，dual scattering 近似多次散射（03 篇）；环境光从体素或探针取</td></tr>
                        <tr><td style="padding:4px 8px">时域</td><td style="padding:4px 8px">静止画面里求平均</td><td style="padding:4px 8px">TAA / TSR 消化随机 h 与抖动噪声；需要正确的发丝运动矢量（插值后的位置差），否则拖影</td></tr>
                        <tr><td style="padding:4px 8px">预算</td><td style="padding:4px 8px">1.4k strands，180²</td><td style="padding:4px 8px">主角发型约 5–15 万根 render strand、几百到几千条 guide；整套发丝 pass 通常要控制在几 ms 内，靠 LOD 与 card 兜底</td></tr>
                    </table>`,
                },
                {
                    type: 'warn', html: `<strong>常见坑</strong>：
                        ① 把光栅化宽度钳到 1 px 却忘了乘 <code>r / r_raster</code> → 远处发型变成实心头盔；
                        ② 只用最近一根做可见性、又把它当不透明 → 发丝重叠处丢光（第 05 步的 α_near）；
                        ③ 着色样本按 MSAA 样本数而不是唯一 strand 数调度 → 着色成本 ×8；
                        ④ 运动矢量用 rest pose 或只用骨骼动画算 → 模拟中的发丝在 TAA 下糊成一片；
                        ⑤ LOD 剔除用逐帧随机 / 屏幕空间选择 → 过渡闪烁；
                        ⑥ 曲线首尾点之间误连三角形（offset / count 处理错）→ 横跨画面的长三角形；
                        ⑦ 半径单位混用（直径 / 半径、cm / mm）→ 覆盖率整体差 2 倍或 10 倍，且看起来"也还行"，很难发现。`,
                },
                { type: 'note', html: `下一篇：<a href="hair_03_lighting.html">03 自阴影与多次散射</a>——在本篇的发丝几何上做 deep opacity map 与 dual scattering。` },
            ],
        },
    ],

    library: {
        title: '附录 · 辅助函数（HLSL）',
        intro: 'pass 节点 <code>CS_HairCullLOD</code> 调用的小函数：每条曲线的固定哈希、曲线包围盒和 6 平面视锥测试。Points / Curves 的缓冲布局见 Common.hlsl；本页 JS 参考实现里对应的是 <code>G2.hashU</code>（LOD 保留）——相机是静止的正交相机，整个 groom 都在视锥内，所以没有单独的视锥剔除预览。',
        blocks: [
            {
                title: 'LOD 与剔除：CurveHash · CurveBounds · FrustumTest', code: `
                    // 每条曲线一个固定的 [0,1) 哈希（(c+1)·黄金比 + murmur3 fmix32，与本页 JS 的 G2.hashU 同一个函数）：与帧无关 → LOD 剔除顺序稳定
                    float CurveHash(uint c) {
                        uint x = (c + 1u) * 0x9e3779b1u;
                        x ^= x >> 15; x *= 0x85ebca6bu;
                        x ^= x >> 13; x *= 0xc2b2ae35u;
                        x ^= x >> 16;
                        return x * (1.0 / 4294967296.0);
                    }

                    // 轴对齐包围盒（世界空间，cm）
                    struct FAABB { float3 Min; float3 Max; };

                    // 一条曲线的包围盒：遍历它的点取 min / max，再外扩最大半径（DeformedPosition.w）。生产版在插值 CS 里按簇（cluster）用原子操作顺手累积
                    FAABB CurveBounds(uint c) {
                        FHairCurve cv = Curves[c];
                        float3 lo = 1e30, hi = -1e30;
                        float  r  = 0;
                        for (uint k = 0; k < cv.PointCount; k++) {
                            float4 P = DeformedPosition[cv.PointOffset + k];
                            lo = min(lo, P.xyz); hi = max(hi, P.xyz); r = max(r, P.w);
                        }
                        FAABB b; b.Min = lo - r; b.Max = hi + r;
                        return b;
                    }

                    // 6 平面视锥测试（FrustumPlanes 来自 cbuffer HairCull，法线朝内：dot(n, x) + d ≥ 0 为内侧）：盒子完全在任一平面外侧 → 剔除
                    bool FrustumTest(FAABB b) {
                        float3 c = 0.5 * (b.Max + b.Min), e = 0.5 * (b.Max - b.Min);   // 中心 + 半边长
                        [unroll] for (int i = 0; i < 6; i++) {
                            float4 pl = FrustumPlanes[i];
                            if (dot(pl.xyz, c) + pl.w < -dot(abs(pl.xyz), e)) return false;   // 中心到平面的距离 < −盒子在法线上的投影半径
                        }
                        return true;   // 与视锥相交或在内部（保守：角落处可能多留一些）
                    }`,
            },
        ],
    },

    graph: {
        title: 'Hair Strands · RDG pass 视图',
        intro: '每个节点 = 一次 Dispatch / Draw；pin = <strong>资源名 · 格式</strong>，同名资源自动连线。lifetime 按<strong>引擎里的真实频率</strong>标注：发丝每帧被模拟、插值、剔除、光栅化；本页的 JS 参考实现里相机和发丝静止，所以光栅化类 stage 只在改参数时重算。',
        types: {
            curve: { color: '#e8c25a', label: 'curve · 点 / 曲线 StructuredBuffer' },
            vis: { color: '#a45fa5', label: 'vis · 可见性 / 覆盖率纹理' },
            light: { color: '#4a8aff', label: 'light · 线性 HDR 颜色' },
            scalar: { color: '#8a96a8', label: 'scalar · cbuffer' },
        },
        common: {
            title: 'Common.hlsl · 系列共享 + 本篇的缓冲布局',
            open: false,
            blocks: [
                { label: '━ 系列共享 ━', code: HAIR.HLSL_COMMON },
                { label: '━ 系列共享 · 毛发 BSDF（HLSL 参考实现）━', code: HAIR.HLSL_BSDF },
                { label: '━ Groom 缓冲 ━', code: HLSL_BUFFERS },
            ],
        },
        nodes: [
            {
                id: 'view', title: 'View / HairParams', tag: 'CB', category: 'user', desc: 'ViewProj · WidthScale · LOD',
                outputs: [{ name: 'HairView', format: 'cbuffer', type: 'scalar' }],
                detail: { desc: '相机矩阵、宽度缩放、LOD 保留比例、光源方向。', preview: null },
            },
            {
                id: 'assetN', title: 'GroomAsset（构建）', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'asset', desc: 'Points / Curves / 插值权重',
                dispatch: { kind: 'CPU', name: 'BuildGroom', count: (S) => `${S.asset ? S.asset.nC : 0} curves · ${S.asset ? S.asset.total : 0} points`, note: '导入时一次：重采样、算 guide 权重、量化上传' },
                outputs: [
                    { name: 'RestPosition', format: 'float4 SB', type: 'curve' }, { name: 'Curves', format: 'uint4 SB', type: 'curve' },
                    { name: 'GuideRestPosition', format: 'float3 SB', type: 'curve' }, { name: 'GuideWeights', format: 'uint4+float4 SB', type: 'curve' },
                ],
                detail: { preview: 'pointsBuf' },
            },
            {
                id: 'sim', title: 'Guide Simulation', tag: 'CS', category: 'function', lifetime: 'frame', step: 'interp', desc: '只模拟 guide（UE：Niagara）',
                dispatch: { name: 'CS_GuideSolver', threads: [64, 1, 1], count: '16 guides × 25 点', note: '本页用解析摆动代替' },
                inputs: [{ name: 'GuideRestPosition', format: 'float3 SB', type: 'curve' }],
                outputs: [{ name: 'GuideDeformedPosition', format: 'float3 SB', type: 'curve' }],
                detail: { preview: 'interpLive', formula: 'ΔG_k(t, τ) = ( 1.6·sin(1.3τ + φ_k),  ≈0,  0.9·sin(0.9τ + 1.7φ_k) ) · t^2   // 本页代替模拟的解析摆动，t = 根→梢', code: null },
            },
            {
                id: 'interpN', title: 'CS_HairInterpolation', tag: 'CS', lifetime: 'frame', step: 'interp', desc: 'P = P_rest + Σ w·ΔG',
                dispatch: { name: 'CS_HairInterpolation', threads: [64, 1, 1], groups: (S) => [Math.ceil((S.asset ? S.asset.total : 0) / 64), 1, 1], idRange: 'ID.x ∈ [0, PointCount)', note: '每 thread 一个 render 点' },
                inputs: [{ name: 'RestPosition', format: 'float4 SB', type: 'curve' }, { name: 'GuideWeights', format: 'uint4+float4 SB', type: 'curve' }, { name: 'GuideDeformedPosition', format: 'float3 SB', type: 'curve' }, { name: 'GuideRestPosition', format: 'float3 SB', type: 'curve' }],
                outputs: [{ name: 'DeformedPosition', format: 'float4 SB', type: 'curve' }],
                detail: { preview: 'interpLive', formula: 4 },
            },
            {
                id: 'cull', title: 'CS_HairCullLOD', tag: 'CS', lifetime: 'frame', step: 'lod', desc: '视锥剔除 + 保留 hash < f',
                dispatch: { name: 'CS_HairCullLOD', threads: [64, 1, 1], groups: (S) => [Math.ceil((S.asset ? S.asset.nC : 0) / 64), 1, 1], idRange: 'ID.x ∈ [0, CurveCount)', note: '生产版按簇（cluster）粒度；输出间接绘制参数' },
                inputs: [{ name: 'DeformedPosition', format: 'float4 SB', type: 'curve' }, { name: 'Curves', format: 'uint4 SB', type: 'curve' }, { name: 'HairView', format: 'cbuffer', type: 'scalar' }],
                outputs: [{ name: 'CulledCurves', format: 'uint SB + IndirectArgs', type: 'curve' }],
                detail: {
                    preview: 'lodWide', formula: 0,
                    code: { title: 'CS_HairCullLOD（核心）', src: `
                        cbuffer HairCull { float4 FrustumPlanes[6]; float LODKeep; uint CurveCount; };   // 平面法线朝内；LODKeep = 保留比例 f
                        StructuredBuffer<float4>  DeformedPosition;   // 本帧发丝点：xyz = 位置，w = 半径
                        RWBuffer<uint>            IndirectArgs;       // [0] = 顶点数（DrawInstancedIndirect）
                        RWBuffer<uint>            CulledCount;
                        RWStructuredBuffer<uint>  CulledCurves;

                        [numthreads(64, 1, 1)]
                        void CS_HairCullLOD(uint3 id : SV_DispatchThreadID) {
                            uint c = id.x; if (c >= CurveCount) return;
                            if (CurveHash(c) >= LODKeep) return;                       // 固定哈希：稳定的剔除顺序
                            if (!FrustumTest(CurveBounds(c))) return;
                            uint slot; InterlockedAdd(IndirectArgs[0], Curves[c].PointCount * 2, slot);   // 每点 2 顶点
                            uint k; InterlockedAdd(CulledCount[0], 1, k);
                            CulledCurves[k] = c;                                       // VS 里再乘 RadiusScale = 1 / LODKeep
                        }` },
                },
            },
            {
                id: 'raster', title: 'HairVisibility（VS ribbon + PS）', tag: 'VS', lifetime: 'frame', step: 'cover', desc: 'ribbon → MSAA id + 解析覆盖 α',
                dispatch: [
                    { kind: 'VS', name: 'VS_HairStrands', count: (S) => `${S.asset ? 2 * S.asset.total : 0} 顶点（每点 2 个）`, idRange: 'SV_VertexID ∈ [0, 2·PointCount)', note: '远处 / 细小发丝可改走 compute 软件光栅化' },
                    { kind: 'PS', name: 'PS_HairVisibility', target: (S) => `HairVisibilityTex ${S.res}×${S.res} · MSAA 8×`, note: '只写 segment id + 深度' },
                    { kind: 'PS', name: 'PS_HairCoverage', target: (S) => `HairCoverageTex ${S.res}×${S.res}`, note: '解析覆盖，乘法混合 Π(1 − c)' },
                ],
                inputs: [{ name: 'DeformedPosition', format: 'float4 SB', type: 'curve' }, { name: 'CulledCurves', format: 'uint SB + IndirectArgs', type: 'curve' }, { name: 'HairView', format: 'cbuffer', type: 'scalar' }],
                outputs: [{ name: 'HairVisibilityTex', format: 'R32_UINT MSAA 8×', type: 'vis' }, { name: 'HairCoverageTex', format: 'R16F', type: 'vis' }],
                detail: { preview: 'alphaFull', code: { title: 'VS_HairStrands', src: HLSL_RIBBON } },
            },
            {
                id: 'compact', title: 'CS_HairVisibilityCompaction', tag: 'CS', lifetime: 'frame', step: 'vis', desc: '8 样本 → 唯一 strand 列表',
                dispatch: { name: 'CS_HairCompaction', threads: [8, 8, 1], groups: (S) => [Math.ceil(S.res / 8), Math.ceil(S.res / 8), 1], idRange: 'ID.xy ∈ 视口', note: '着色样本数 ∝ 唯一 strand，而非 MSAA 样本' },
                inputs: [{ name: 'HairVisibilityTex', format: 'R32_UINT MSAA 8×', type: 'vis' }, { name: 'HairCoverageTex', format: 'R16F', type: 'vis' }],
                outputs: [{ name: 'HairSampleList', format: 'FHairSample SB', type: 'vis' }],
                detail: { preview: 'msaaErr' },
            },
            {
                id: 'shadeN', title: 'CS_HairShading', tag: 'CS', lifetime: 'frame', step: 'shade', desc: 'S(wi, wo; h~U)·E',
                dispatch: { name: 'CS_HairShading', threads: [64, 1, 1], count: (S) => `${S.hairPix ? S.hairPix.length : '≈ 可见发丝像素'} 个样本`, idRange: 'ID.x ∈ [0, SampleCount)', note: '间接 dispatch，样本数来自压缩 pass' },
                inputs: [{ name: 'HairSampleList', format: 'FHairSample SB', type: 'vis' }, { name: 'HairView', format: 'cbuffer', type: 'scalar' }],
                outputs: [{ name: 'HairSampleLighting', format: 'R11G11B10F SB', type: 'light' }],
                detail: { preview: 'shadeOne' },
            },
            {
                id: 'compose', title: 'PS_HairComposition', tag: 'PS', lifetime: 'frame', step: 'shade', desc: 'Σ w_k·L_k + (1 − α)·Scene',
                dispatch: { kind: 'PS', name: 'PS_HairComposition', target: 'SceneColor（全屏）', note: '每像素按样本权重累加，再与场景按 1 − α 混合' },
                inputs: [{ name: 'HairSampleLighting', format: 'R11G11B10F SB', type: 'light' }, { name: 'HairSampleList', format: 'FHairSample SB', type: 'vis' }, { name: 'HairCoverageTex', format: 'R16F', type: 'vis' }],
                outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'light' }],
                detail: { preview: 'shadeOne', formula: 3 },
            },
            {
                id: 'taaN', title: 'TAA / TSR', tag: 'PS', category: 'output', lifetime: 'history', step: 'shade', desc: 'history 累积随机 h 与抖动',
                dispatch: { kind: 'PS', name: 'TemporalAA', target: 'SceneColorAA', note: '本页：静止画面里的累积均值 L̄_N' },
                inputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'light' }, { name: 'SceneColorAA', format: 'RGBA16F', type: 'light', history: true }],
                outputs: [{ name: 'SceneColorAA', format: 'RGBA16F', type: 'light' }],
                detail: { preview: 'shadeAcc', formula: 2 },
            },
        ],
    },

    references: [
        { title: 'Advanced Techniques in Real-time Hair Rendering and Simulation', author: 'Cem Yuksel, Sarah Tariq', venue: 'SIGGRAPH Course', year: 2010, url: 'http://www.cemyuksel.com/courses/conferences/siggraph2010-hair/', note: 'single-strand / multi-strand（重心权重）插值、发丝几何生成' },
        { title: 'Strand-based Hair Rendering in Frostbite', author: 'Sebastian Tafuri', venue: 'SIGGRAPH Advances in Real-Time Rendering in Games', year: 2019, url: 'https://advances.realtimerendering.com/s2019/hair_presentation_final.pdf', note: 'visibility buffer + MSAA、相似样本只着色一次、按像素尺寸加宽的三角形条带、deep opacity map 与 dual scattering' },
        { title: 'TressFX（GPUOpen）', author: 'AMD', venue: 'GPUOpen Effects', url: 'https://gpuopen.com/tressfx/', note: 'VS 展开 camera-facing quad（外扩约 0.71 px）、PS ComputeCoverage 解析覆盖、PPLL + K-buffer（KBUFFER_SIZE 8）；3.1 起有 ShortCut' },
        { title: 'Real-Time Concurrent Linked List Construction on the GPU', author: 'Jason C. Yang, Justin Hensley, Holger Grün, Nicolas Thibieroz', venue: 'Computer Graphics Forum (EGSR)', year: 2010, note: 'PPLL：逐像素链表 OIT' },
        { title: 'The Visibility Buffer: A Cache-Friendly Approach to Deferred Shading', author: 'Christopher A. Burns, Warren A. Hunt', venue: 'Journal of Computer Graphics Techniques', year: 2013, note: '只写 id 的可见性 pass，着色延后' },
        { title: 'Groom Interpolation in Unreal Engine', author: 'Epic Games', venue: 'Unreal Engine 5.4 Documentation', url: 'https://dev.epicgames.com/documentation/unreal-engine/groom-interpolation-in-unreal-engine?application_version=5.4', note: 'guide 生成、插值质量 / 距离、Randomize Guide、Use Unique Guide' },
        { title: 'FHairLODSettings / FHairStrandsCurves（API 参考）', author: 'Epic Games', venue: 'Unreal Engine API Documentation', url: 'https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/HairStrandsCore/FHairLODSettings?application_version=5.6', note: 'CurveDecimation、VertexDecimation、AngularThreshold、ThicknessScale、GeometryType；曲线 offset / count 布局' },
        { title: 'Groom Performance and Debugging with Unreal Engine', author: 'Epic Games', venue: 'Unreal Engine 5.4 Documentation', url: 'https://dev.epicgames.com/documentation/unreal-engine/groom-performance-and-debugging-with-unreal-engine', note: 'primary visibility 的 MSAA / PPLL / compute raster 模式与相关 cvar' },
        { title: 'A Practical and Controllable Hair and Fur Model for Production Path Tracing', author: 'Matt Jen-Yuan Chiang, Benedikt Bitterli, Chuck Tappan, Brent Burley', venue: 'Computer Graphics Forum (EGSR)', year: 2016, note: '本系列 HAIR.bsdf 的模型（pbrt-v3 实现）' },
    ],
});

// ── 3D views（只读 stage 数据）──
const G3D = { s: 1 / 7, lift: 1, maxStrands: 160 };   // cm → 视图单位：发根 y = 1，发梢 ≈ −1；最多画 ~160 根 render strand
function drawGroom3d(c, S) {
    const G = S.groom, gs = G.guides, O = G.opts, s = G3D.s;
    const P3 = (p, k) => [p[3 * k] * s, p[3 * k + 1] * s + G3D.lift, p[3 * k + 2] * s];
    const poly = (p) => { const out = []; for (let k = 0; k < p.length / 3; k++) out.push(P3(p, k)); return out; };
    const colOf = (g) => G2.GUIDE_COLORS[(g.gx + g.gz * 3) % 8];
    const up = (q) => [q[0], q[1] + 0.008, q[2]];   // 发根标记略高于头皮平面，俯视时不被遮住
    const hw = 0.5 * O.width * s + 0.04, hd = 0.5 * O.depth * s + 0.04;
    const rgba = (hex, a) => `rgba(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)}, ${a})`;
    // render strand 子集：与 LOD 同一个固定哈希 → 改参数时抽中的是同一批发丝
    const lines = [], frac = Math.min(1, G3D.maxStrands / G.strands.length);
    for (const st of G.strands) if (G2.hashU(st.id) < frac) lines.push({ pts: poly(st.p), color: rgba(colOf(gs[st.guide]), 0.55), width: 0.8 });
    for (const g of gs) lines.push({ pts: poly(g.p), color: colOf(g), width: 2.8 });
    // 发根图（rootPatch）里自动挑的那一根，以及它到 4 条 guide 发根的权重连线
    let pick = G.strands[0], bd = Infinity;
    for (const st of G.strands) { const d = Math.hypot(st.rootUV[0] - 0.47, (st.rootUV[1] - 0.42) * 0.4); if (d < bd) { bd = d; pick = st; } }
    lines.push({ pts: poly(pick.p), color: '#ffffff', width: 1.8 });
    for (const [gi, wt] of G2.guideWeights(O, pick.rootUV[0], pick.rootUV[1])) lines.push({ pts: [up(P3(pick.p, 0)), up(P3(gs[gi].p, 0))], color: `rgba(255, 255, 255, ${(0.25 + 0.75 * wt).toFixed(2)})`, width: 1.2, dash: [3, 3] });
    PL.draw.mesh3d(c, {
        meshes: [PL.mesh.param((u, v) => [(2 * u - 1) * hw, G3D.lift - 0.004, (2 * v - 1) * hd], [8, 2], { rgb: [0.42, 0.33, 0.27], twoSided: true })],
        lines,
        points: gs.map((g) => ({ p: up(P3(g.p, 0)), color: colOf(g), r: 3 })),
        labels: [{ p: [-hw, G3D.lift + 0.12, hd], text: '发根区域（root UV）', color: '#c8b49c' }, { p: P3(gs[7].p, O.segs), text: 'guide', color: colOf(gs[7]) }],
        grid: { y: -1.05, size: 1.2 },
        orbit: { yaw: 0.8, pitch: 0.42, dist: 4.2, target: [0, -0.05, -0.15] },
    });
}
