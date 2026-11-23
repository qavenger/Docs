// 毛发渲染 04 · 插片 Hair Cards —— Pipeline Lab 内容文件（与 hair_common.js 一起构建，在 pages/ 里执行）：
//   node <skill>/scripts/build_doc.mjs hair_common.js hair_04_cards.lab.js -o hair_04_cards.html
// 约定全部沿用 hair_common.js（y 上、z 朝默认观察者、cm、T = 根 → 梢、sinθ = ω·T、wi → 光、wo → 观察者、L_o = S·E）。
// 插片 = 从同一个 HAIR.makeGroom() 发型聚类 → 拟合 ribbon → 在 card 空间用 HAIR.raster 烘焙 atlas；屏幕端是自写的三角形光栅器。

const HC = (() => {
    const TW = 24, TH = 192, NM = 4;          // atlas tile = 24 × 192 texel（宽 × 长），4 级 mip（24→12→6→3）
    const RW = 80, RH = 112;                   // 屏幕渲染分辨率（正交相机）
    const CAM_SIZE = 16, CAM_CENTER = [0, -7.4, -1.0];
    const BG = [0.16, 0.175, 0.2];             // 合成背景（线性）
    const NREF = [0, 0, 1];                    // 头皮外法线（本发片朝 +z 生长面）→ 插片朝向参考
    const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const norm = HAIR.norm, cross = HAIR.cross;
    const hash = (i) => { let x = Math.imul(i | 0, 0x9E3779B1); x ^= x >>> 15; x = Math.imul(x, 0x85EBCA77); x ^= x >>> 13; x = Math.imul(x, 0xC2B2AE3D); x ^= x >>> 16; return (x >>> 0) / 4294967296; };
    const hash3 = (a, b, c) => hash(Math.imul(a, 73856093) ^ Math.imul(b, 19349663) ^ Math.imul(c, 83492791));
    const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

    // ── 1. 聚类：k-means（特征 = 根 / 中点 / 梢 位置，k-means++ 初始化） ──
    function kmeans(strands, K, rand, iters = 12) {
        const n = strands.length, F = 9, X = new Float32Array(n * F), N = strands[0].p.length / 3 - 1;
        const Wt = [1.6, 1.6, 1.6, 1, 1, 1, 0.7, 0.7, 0.7];
        strands.forEach((s, i) => [0, N >> 1, N].forEach((k, j) => { for (let c = 0; c < 3; c++) X[i * F + 3 * j + c] = s.p[3 * k + c] * Wt[3 * j + c]; }));
        const C = new Float32Array(K * F), d2 = new Float32Array(n).fill(Infinity), assign = new Int32Array(n);
        const dist = (i, j) => { let d = 0; for (let f = 0; f < F; f++) { const e = X[i * F + f] - C[j * F + f]; d += e * e; } return d; };
        let pick = Math.floor(rand() * n);
        for (let j = 0; j < K; j++) {
            for (let f = 0; f < F; f++) C[j * F + f] = X[pick * F + f];
            let tot = 0;
            for (let i = 0; i < n; i++) { d2[i] = Math.min(d2[i], dist(i, j)); tot += d2[i]; }
            let r = rand() * tot; pick = n - 1;
            for (let i = 0; i < n; i++) { r -= d2[i]; if (r <= 0) { pick = i; break; } }
        }
        for (let it = 0; it < iters; it++) {
            for (let i = 0; i < n; i++) { let best = 0, bd = Infinity; for (let j = 0; j < K; j++) { const d = dist(i, j); if (d < bd) { bd = d; best = j; } } assign[i] = best; }
            const sum = new Float64Array(K * F), cnt = new Int32Array(K);
            for (let i = 0; i < n; i++) { const j = assign[i]; cnt[j]++; for (let f = 0; f < F; f++) sum[j * F + f] += X[i * F + f]; }
            for (let j = 0; j < K; j++) if (cnt[j]) for (let f = 0; f < F; f++) C[j * F + f] = sum[j * F + f] / cnt[j];
        }
        // 按根部 x 排序簇编号（atlas 里 tile 的顺序 = 从左到右），保证截图稳定
        const rootX = new Float64Array(K), cnt = new Int32Array(K);
        for (let i = 0; i < n; i++) { rootX[assign[i]] += strands[i].p[0] + 0.01 * strands[i].p[2]; cnt[assign[i]]++; }
        const order = [...Array(K).keys()].filter((j) => cnt[j] > 0).sort((a, b) => rootX[a] / cnt[a] - rootX[b] / cnt[b]);
        const remap = new Int32Array(K).fill(-1); order.forEach((j, r) => { remap[j] = r; });
        for (let i = 0; i < n; i++) assign[i] = remap[assign[i]];
        return { assign, K: order.length };
    }

    // ── 2. 拟合：沿 clump 中心线的 ribbon（等弧长采样 → 平滑 → 帧 T/B/N → 宽度 = P95|a|） ──
    function arcPoint(P, cum, s) {
        const N = P.length / 3 - 1;
        if (s <= 0) return [P[0], P[1], P[2]];
        for (let k = 0; k < N; k++) if (cum[k + 1] >= s) {
            const u = (s - cum[k]) / Math.max(1e-9, cum[k + 1] - cum[k]);
            return [P[3 * k] + u * (P[3 * k + 3] - P[3 * k]), P[3 * k + 1] + u * (P[3 * k + 4] - P[3 * k + 1]), P[3 * k + 2] + u * (P[3 * k + 5] - P[3 * k + 2])];
        }
        return [P[3 * N], P[3 * N + 1], P[3 * N + 2]];
    }
    function fitCard(strands, ids, cumAll) {
        const N = strands[0].p.length / 3 - 1;
        const lens = ids.map((i) => cumAll[i][N]);
        const sorted = lens.slice().sort((a, b) => a - b), Lc = sorted[Math.floor(0.9 * (sorted.length - 1))];
        const c = new Float32Array(3 * (N + 1));
        for (let k = 0; k <= N; k++) {
            const sk = Lc * k / N; let x = 0, y = 0, z = 0, m = 0;
            ids.forEach((i, q) => { if (k > 0 && lens[q] < sk * 0.98) return; const p = arcPoint(strands[i].p, cumAll[i], sk); x += p[0]; y += p[1]; z += p[2]; m++; });
            if (m) { c[3 * k] = x / m; c[3 * k + 1] = y / m; c[3 * k + 2] = z / m; }
            else for (let a = 0; a < 3; a++) c[3 * k + a] = 2 * c[3 * k - 3 + a] - c[3 * k - 6 + a];
        }
        for (let pass = 0; pass < 2; pass++) {   // 1-2-1 平滑，去掉逐样本抖动
            const o = c.slice();
            for (let k = 1; k < N; k++) for (let a = 0; a < 3; a++) c[3 * k + a] = 0.25 * o[3 * k - 3 + a] + 0.5 * o[3 * k + a] + 0.25 * o[3 * k + 3 + a];
        }
        const T = new Float32Array(3 * (N + 1)), B = new Float32Array(3 * (N + 1)), Nn = new Float32Array(3 * (N + 1)), s = new Float32Array(N + 1);
        for (let k = 0; k <= N; k++) {
            const k0 = Math.max(0, k - 1), k1 = Math.min(N, k + 1);
            const t = norm([c[3 * k1] - c[3 * k0], c[3 * k1 + 1] - c[3 * k0 + 1], c[3 * k1 + 2] - c[3 * k0 + 2]]);
            const b = norm(cross(NREF, t)), nn = cross(t, b);
            T.set(t, 3 * k); B.set(b, 3 * k); Nn.set(nn, 3 * k);
            if (k > 0) s[k] = s[k - 1] + Math.hypot(c[3 * k] - c[3 * k - 3], c[3 * k + 1] - c[3 * k - 2], c[3 * k + 2] - c[3 * k - 1]);
        }
        // 每个发丝点 → card 局部坐标 (a 横向, s 弧长, d 法向)：投影到中心线折线上最近的点
        const loc = [], absA = [];
        let dMax = 0;
        ids.forEach((i) => {
            const P = strands[i].p, L = new Float32Array(3 * (N + 1));
            for (let q = 0; q <= N; q++) {
                const p = [P[3 * q], P[3 * q + 1], P[3 * q + 2]];
                let best = Infinity, bk = 0, bu = 0;
                for (let k = 0; k < N; k++) {
                    const ax = c[3 * k], ay = c[3 * k + 1], az = c[3 * k + 2], dx = c[3 * k + 3] - ax, dy = c[3 * k + 4] - ay, dz = c[3 * k + 5] - az;
                    const u = clamp(((p[0] - ax) * dx + (p[1] - ay) * dy + (p[2] - az) * dz) / Math.max(1e-9, dx * dx + dy * dy + dz * dz), 0, 1);
                    const ex = ax + u * dx - p[0], ey = ay + u * dy - p[1], ez = az + u * dz - p[2], d = ex * ex + ey * ey + ez * ez;
                    if (d < best) { best = d; bk = k; bu = u; }
                }
                const fr = frameAt({ c, T, B, Nn, s, N }, bk + bu);
                const r = [p[0] - fr.p[0], p[1] - fr.p[1], p[2] - fr.p[2]];
                const a = dot(r, fr.B), d = dot(r, fr.N);
                L[3 * q] = a; L[3 * q + 1] = fr.s; L[3 * q + 2] = d;
                absA.push(Math.abs(a)); dMax = Math.max(dMax, Math.abs(d));
            }
            loc.push(L);
        });
        absA.sort((x, y) => x - y);
        const halfW = Math.max(0.15, 1.12 * absA[Math.floor(0.95 * (absA.length - 1))]);
        return { ids, c, T, B, Nn, s, N, Lc: s[N], halfW, dMax, loc };
    }
    function frameAt(cd, x) {   // 中心线参数 x ∈ [0, N]（段号 + 段内比例）→ 插值帧
        const N = cd.N, k = Math.min(N - 1, Math.max(0, Math.floor(x))), u = clamp(x - k, 0, 1);
        const L = (A) => [A[3 * k] + u * (A[3 * k + 3] - A[3 * k]), A[3 * k + 1] + u * (A[3 * k + 4] - A[3 * k + 1]), A[3 * k + 2] + u * (A[3 * k + 5] - A[3 * k + 2])];
        return { p: L(cd.c), T: norm(L(cd.T)), B: norm(L(cd.B)), N: norm(L(cd.Nn)), s: cd.s[k] + u * (cd.s[k + 1] - cd.s[k]) };
    }

    // ── 3. 烘焙：card 空间 (a·f, −s, d) 里用 HAIR.raster 光栅化该 clump 的发丝 → atlas tile ──
    function makeAtlas(K) {
        const cols = Math.min(K, 16), rows = Math.ceil(K / 16), mips = [];
        for (let m = 0; m < NM; m++) {
            const W = cols * (TW >> m), H = rows * (TH >> m), n = W * H;
            mips.push({ W, H, tw: TW >> m, th: TH >> m, A: new Float32Array(n), C: new Float32Array(n), D: new Float32Array(n), T: new Float32Array(3 * n), R: new Float32Array(n), I: new Float32Array(n), O: new Float32Array(n), M: new Uint8Array(n) });
        }
        return { cols, rows, K, mips };
    }
    function bakeTile(card, strands, atlas, j) {
        const { ids, Lc, halfW, loc } = card, N = card.N;
        const du = 2 * halfW / TW, dv = Lc / TH, f = dv / du;
        const copies = ids.map((i, q) => {
            const L = loc[q], p = new Float32Array(L.length);
            for (let k = 0; k <= N; k++) { p[3 * k] = L[3 * k] * f; p[3 * k + 1] = -L[3 * k + 1]; p[3 * k + 2] = L[3 * k + 2]; }
            return { p, w0: strands[i].w0, w1: strands[i].w1, id: i };
        });
        const cam = HAIR.ortho({ yaw: 0, pitch: 0, center: [0, -Lc / 2, 0], size: Lc });
        const R = HAIR.raster(copies, cam, TW, TH, { widthScale: f });
        const m0 = atlas.mips[0], ox = (j % atlas.cols) * TW, oy = Math.floor(j / atlas.cols) * TH;
        card.tile = [j % atlas.cols, Math.floor(j / atlas.cols)];
        const n = TW * TH, has = new Uint8Array(n), tu = new Float32Array(n), tv = new Float32Array(n), tn = new Float32Array(n), rt = new Float32Array(n), idc = new Float32Array(n), dp = new Float32Array(n);
        for (let i = 0; i < n; i++) if (R.id[i] >= 0) {
            has[i] = 1;
            const t = norm([R.tan[3 * i] / f, -R.tan[3 * i + 1], R.tan[3 * i + 2]]);
            tu[i] = t[0]; tv[i] = t[1]; tn[i] = t[2]; rt[i] = R.v[i]; idc[i] = hash(R.id[i]); dp[i] = -R.depth[i];
        }
        // dilation：非 alpha 通道向空 texel 外扩（否则 bilinear / mip 会把"无发丝"的 0 混进切线与 ID）
        const filled = has.slice();
        for (let it = 0; it < 6; it++) {
            const nf = filled.slice();
            for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
                const i = y * TW + x; if (filled[i]) continue;
                let cnt = 0, a = 0, b = 0, c = 0, r = 0, q = 0, d = 0;
                for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
                    const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= TW || Y >= TH) continue;
                    const k = Y * TW + X; if (!filled[k]) continue;
                    cnt++; a += tu[k]; b += tv[k]; c += tn[k]; r += rt[k]; q += idc[k]; d += dp[k];
                }
                if (cnt) { const t = norm([a, b, c]); tu[i] = t[0]; tv[i] = t[1]; tn[i] = t[2]; rt[i] = r / cnt; idc[i] = q / cnt; dp[i] = d / cnt; nf[i] = 1; }
            }
            filled.set(nf);
        }
        // AO ≈ 团内密度（模糊后的累计覆盖 cov）造成的遮挡 × 根部头皮遮挡（廉价近似，生产工具会对整个 groom 做光线追踪）
        const blur = new Float32Array(n);
        for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
            let s = 0, w = 0;
            for (let dy = -3; dy <= 3; dy++) for (let dx = -2; dx <= 2; dx++) { const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= TW || Y >= TH) continue; s += R.cov[Y * TW + X]; w++; }
            blur[y * TW + x] = s / w;
        }
        for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
            const i = y * TW + x, o = (oy + y) * m0.W + ox + x;
            m0.A[o] = R.alpha[i]; m0.C[o] = R.cov[i]; m0.M[o] = has[i];
            if (!filled[i]) { tu[i] = 0; tv[i] = 1; tn[i] = 0; rt[i] = (y + 0.5) / TH; idc[i] = 0.5; dp[i] = 0; }
            m0.T[3 * o] = tu[i]; m0.T[3 * o + 1] = tv[i]; m0.T[3 * o + 2] = tn[i];
            m0.R[o] = rt[i]; m0.I[o] = idc[i]; m0.D[o] = dp[i];
            m0.O[o] = Math.exp(-1.1 * blur[i]) * (0.55 + 0.45 * smooth(0, 0.22, (y + 0.5) / TH));
        }
        return R;
    }
    function buildMips(atlas) {   // 2×2 box：tile 尺寸都能被 8 整除，所以不会跨 tile 取样
        for (let m = 1; m < NM; m++) {
            const S0 = atlas.mips[m - 1], S1 = atlas.mips[m];
            for (let y = 0; y < S1.H; y++) for (let x = 0; x < S1.W; x++) {
                const o = y * S1.W + x, idx = [(2 * y) * S0.W + 2 * x, (2 * y) * S0.W + 2 * x + 1, (2 * y + 1) * S0.W + 2 * x, (2 * y + 1) * S0.W + 2 * x + 1];
                let t0 = 0, t1 = 0, t2 = 0;
                for (const ch of ['A', 'C', 'D', 'R', 'I', 'O']) S1[ch][o] = (S0[ch][idx[0]] + S0[ch][idx[1]] + S0[ch][idx[2]] + S0[ch][idx[3]]) / 4;
                S1.M[o] = S0.M[idx[0]] | S0.M[idx[1]] | S0.M[idx[2]] | S0.M[idx[3]];
                for (const i of idx) { t0 += S0.T[3 * i]; t1 += S0.T[3 * i + 1]; t2 += S0.T[3 * i + 2]; }
                const t = norm([t0, t1, t2]); S1.T[3 * o] = t[0]; S1.T[3 * o + 1] = t[1]; S1.T[3 * o + 2] = t[2];
            }
        }
    }
    // Castaño 2010：每个 tile、每级 mip 找缩放 s，使 frac(s·α_m > τ) = frac(α_0 > τ)
    function coverageScales(atlas, tau) {
        const K = atlas.K, sc = new Float32Array(K * NM), stats = [];
        for (let m = 0; m < NM; m++) stats.push({ mean: 0, naive: 0, cp: 0, n: 0 });
        for (let j = 0; j < K; j++) {
            const tx = j % atlas.cols, ty = Math.floor(j / atlas.cols);
            const tileVals = (m) => { const M = atlas.mips[m], out = new Float32Array(M.tw * M.th); let q = 0; for (let y = 0; y < M.th; y++) for (let x = 0; x < M.tw; x++) out[q++] = M.A[(ty * M.th + y) * M.W + tx * M.tw + x]; return out; };
            const frac = (v, s) => { let c = 0; for (let i = 0; i < v.length; i++) if (Math.min(1, v[i] * s) > tau) c++; return c / v.length; };
            const v0 = tileVals(0), c0 = frac(v0, 1);
            for (let m = 0; m < NM; m++) {
                const v = m ? tileVals(m) : v0;
                let lo = 0.05, hi = 32, s = 1;
                if (m) { for (let it = 0; it < 28; it++) { s = Math.sqrt(lo * hi); if (frac(v, s) < c0) lo = s; else hi = s; } s = Math.sqrt(lo * hi); }
                sc[j * NM + m] = s;
                let mean = 0; for (let i = 0; i < v.length; i++) mean += v[i];
                stats[m].mean += mean / v.length; stats[m].naive += frac(v, 1); stats[m].cp += frac(v, s); stats[m].n++;
            }
        }
        stats.forEach((s) => { s.mean /= s.n; s.naive /= s.n; s.cp /= s.n; });
        return { sc, stats };
    }

    // ── 4. 屏幕端：插片 = ribbon 三角形，正交投影，逐像素收集片元（A-buffer），之后按不同透明模式 resolve ──
    function rasterCards(cards, cam, W, H) {
        let cap = W * H * 4, F = 0;
        let pix = new Int32Array(cap), cid = new Int16Array(cap), fu = new Float32Array(cap), fv = new Float32Array(cap), fz = new Float32Array(cap);
        const grow = () => { cap *= 2; const g = (A, T) => { const B = new T(cap); B.set(A); return B; }; pix = g(pix, Int32Array); cid = g(cid, Int16Array); fu = g(fu, Float32Array); fv = g(fv, Float32Array); fz = g(fz, Float32Array); };
        const slot = new Int32Array(W * H).fill(-1);
        cards.forEach((cd, j) => {
            const N = cd.N, P = [];
            for (let k = 0; k <= N; k++) for (const sgn of [-1, 1]) {
                const q = [cd.c[3 * k] + sgn * cd.halfW * cd.B[3 * k], cd.c[3 * k + 1] + sgn * cd.halfW * cd.B[3 * k + 1], cd.c[3 * k + 2] + sgn * cd.halfW * cd.B[3 * k + 2]];
                const pr = HAIR.project(cam, q, W, H);
                P.push([pr[0], pr[1], pr[2], sgn < 0 ? 0 : 1, k / N]);
            }
            const touched = [];
            const tri = (A, B, C) => {
                const area = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]);
                if (Math.abs(area) < 1e-9) return;
                const x0 = Math.max(0, Math.ceil(Math.min(A[0], B[0], C[0]))), x1 = Math.min(W - 1, Math.floor(Math.max(A[0], B[0], C[0])));
                const y0 = Math.max(0, Math.ceil(Math.min(A[1], B[1], C[1]))), y1 = Math.min(H - 1, Math.floor(Math.max(A[1], B[1], C[1])));
                for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
                    const w0 = ((B[0] - x) * (C[1] - y) - (B[1] - y) * (C[0] - x)) / area, w1 = ((C[0] - x) * (A[1] - y) - (C[1] - y) * (A[0] - x)) / area, w2 = 1 - w0 - w1;
                    if (w0 < -1e-7 || w1 < -1e-7 || w2 < -1e-7) continue;
                    const z = w0 * A[2] + w1 * B[2] + w2 * C[2], i = y * W + x;
                    let f = slot[i];
                    if (f >= 0 && fz[f] <= z) continue;           // 同一张 card 每像素只留最近的片元
                    if (f < 0) { if (F >= cap) grow(); f = F++; slot[i] = f; touched.push(i); }
                    pix[f] = i; cid[f] = j; fz[f] = z;
                    fu[f] = w0 * A[3] + w1 * B[3] + w2 * C[3]; fv[f] = w0 * A[4] + w1 * B[4] + w2 * C[4];
                }
            };
            for (let k = 0; k < N; k++) { const a = P[2 * k], b = P[2 * k + 1], c = P[2 * k + 2], d = P[2 * k + 3]; tri(a, b, d); tri(a, d, c); }
            touched.forEach((i) => { slot[i] = -1; });
        });
        // 每像素片元链表，按深度排序（近 → 远）
        const start = new Int32Array(W * H + 1), list = new Int32Array(F);
        for (let f = 0; f < F; f++) start[pix[f] + 1]++;
        for (let i = 0; i < W * H; i++) start[i + 1] += start[i];
        const fill = start.slice(0, W * H);
        for (let f = 0; f < F; f++) list[fill[pix[f]]++] = f;
        for (let i = 0; i < W * H; i++) {
            for (let a = start[i] + 1; a < start[i + 1]; a++) { const f = list[a]; let b = a - 1; while (b >= start[i] && fz[list[b]] > fz[f]) { list[b + 1] = list[b]; b--; } list[b + 1] = f; }
        }
        return { W, H, F, pix: pix.subarray(0, F), card: cid.subarray(0, F), u: fu.subarray(0, F), v: fv.subarray(0, F), z: fz.subarray(0, F), start, list };
    }
    // 片元采样：bilinear（tile 内 clamp —— 真实 atlas 要留 gutter，否则会串 tile）
    function sampleFrags(fr, cards, atlas, mip, scales) {
        const F = fr.F, M = atlas.mips[mip], out = { a: new Float32Array(F), acp: new Float32Array(F), T: new Float32Array(3 * F), Nw: new Float32Array(3 * F), R: new Float32Array(F), I: new Float32Array(F), O: new Float32Array(F), D: new Float32Array(F) };
        for (let f = 0; f < F; f++) {
            const j = fr.card[f], cd = cards[j], tx = cd.tile[0] * M.tw, ty = cd.tile[1] * M.th;
            const x = clamp(fr.u[f] * M.tw - 0.5, 0, M.tw - 1), y = clamp(fr.v[f] * M.th - 0.5, 0, M.th - 1);
            const x0 = Math.min(M.tw - 2, Math.floor(x)), y0 = Math.min(M.th - 2, Math.floor(y)), ax = x - x0, ay = y - y0;
            const i00 = (ty + y0) * M.W + tx + x0, i10 = i00 + 1, i01 = i00 + M.W, i11 = i01 + 1;
            const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
            const bl = (A) => A[i00] * w00 + A[i10] * w10 + A[i01] * w01 + A[i11] * w11;
            const a = bl(M.A);
            out.a[f] = a; out.acp[f] = Math.min(1, a * scales[j * NM + mip]);
            out.R[f] = bl(M.R); out.I[f] = bl(M.I); out.O[f] = bl(M.O); out.D[f] = bl(M.D);
            const tu = M.T[3 * i00] * w00 + M.T[3 * i10] * w10 + M.T[3 * i01] * w01 + M.T[3 * i11] * w11;
            const tv = M.T[3 * i00 + 1] * w00 + M.T[3 * i10 + 1] * w10 + M.T[3 * i01 + 1] * w01 + M.T[3 * i11 + 1] * w11;
            const tn = M.T[3 * i00 + 2] * w00 + M.T[3 * i10 + 2] * w10 + M.T[3 * i01 + 2] * w01 + M.T[3 * i11 + 2] * w11;
            const fm = frameAt(cd, fr.v[f] * cd.N);
            const Tw = norm([tu * fm.B[0] + tv * fm.T[0] + tn * fm.N[0], tu * fm.B[1] + tv * fm.T[1] + tn * fm.N[1], tu * fm.B[2] + tv * fm.T[2] + tn * fm.N[2]]);
            out.T[3 * f] = Tw[0]; out.T[3 * f + 1] = Tw[1]; out.T[3 * f + 2] = Tw[2];
            out.Nw[3 * f] = fm.N[0]; out.Nw[3 * f + 1] = fm.N[1]; out.Nw[3 * f + 2] = fm.N[2];
        }
        return out;
    }
    // 不打光的基色：根 → 梢渐变 × 每根发丝随机明暗（发丝参考与插片共用同一个函数）
    const baseColor = (col, root, id, out, o = 0) => { const k = (0.5 + 0.5 * root) * (0.8 + 0.4 * id); out[o] = col[0] * k; out[o + 1] = col[1] * k; out[o + 2] = col[2] * k; return out; };

    // ── 绘制辅助 ──
    function offscreen(W, H, fn) {
        const c = document.createElement('canvas'); c.width = W; c.height = H;
        const g = c.getContext('2d'), im = g.createImageData(W, H), px = [0, 0, 0];
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { fn(x, y, px); const o = 4 * (y * W + x); im.data[o] = 255 * clamp(px[0], 0, 1); im.data[o + 1] = 255 * clamp(px[1], 0, 1); im.data[o + 2] = 255 * clamp(px[2], 0, 1); im.data[o + 3] = 255; }
        g.putImageData(im, 0, 0); return c;
    }
    const disp = (rgb, i, out, exposure = 1) => { const d = HAIR.display([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]], exposure); out[0] = d[0]; out[1] = d[1]; out[2] = d[2]; return out; };

    // ── 5. 着色：Scheuermann 2004（KK + 两条平移高光）与 HAIR.bsdf（Chiang 2016）在插片切线上 ──
    const lightDir = (yaw, el) => [Math.sin(yaw) * Math.cos(el), Math.sin(el), Math.cos(yaw) * Math.cos(el)];
    const shiftT = (T, N, s) => norm([T[0] + s * N[0], T[1] + s * N[1], T[2] + s * N[2]]);
    const strandSpec = (T, V, L, e) => { const H = norm([L[0] + V[0], L[1] + V[1], L[2] + V[2]]), th = dot(T, H); return smooth(-1, 0, th) * Math.pow(Math.sqrt(Math.max(0, 1 - th * th)), e); };
    const KK = { shift1: 0.07, shift2: -0.09, exp1: 320, exp2: 48, spec1: 0.24, spec2: 0.2, jitter: 0.12, E: 2.2 };
    function shadeKK(T, Nc, V, L, root, id, ao, col, out, o, shifted = true) {
        const Ns = dot(Nc, V) < 0 ? [-Nc[0], -Nc[1], -Nc[2]] : Nc, base = baseColor(col, root, id, [0, 0, 0]);
        let r = 0, g = 0, b = 0;
        if (shifted) {
            const sh = (id - 0.5) * KK.jitter, t1 = shiftT(T, Ns, KK.shift1 + sh), t2 = shiftT(T, Ns, KK.shift2 + sh);
            const diff = clamp(0.25 + 0.75 * dot(Ns, L), 0, 1);                     // lerp(0.25, 1, N·L)
            const s1 = KK.spec1 * strandSpec(t1, V, L, KK.exp1), mask = 0.25 + 0.75 * ((id * 13.37) % 1), s2 = KK.spec2 * mask * strandSpec(t2, V, L, KK.exp2);
            r = diff * base[0] + s1 + s2 * Math.sqrt(col[0]); g = diff * base[1] + s1 + s2 * Math.sqrt(col[1]); b = diff * base[2] + s1 + s2 * Math.sqrt(col[2]);
            r *= ao; g *= ao; b *= ao;
        } else {                                                                     // 经典 Kajiya-Kay：sin(T,L) + sin(T,H)^p
            const k = HAIR.kajiyaKay(T, L, V, KK.exp1);
            r = k.diffuse * base[0] + KK.spec1 * k.specular; g = k.diffuse * base[1] + KK.spec1 * k.specular; b = k.diffuse * base[2] + KK.spec1 * k.specular;
        }
        out[o] = r * KK.E; out[o + 1] = g * KK.E; out[o + 2] = b * KK.E;
        return out;
    }
    function bsdfSample(bs, T, V, L, h, E, out, o) {   // L_o = S(wi, wo; h) · E，h 随机 → 时域累积 = 对发丝宽度积分
        const f = HAIR.fiberFrame(T, V), s = bs.S(HAIR.toFiber(L, f), HAIR.toFiber(V, f), h);
        out[o] = s[0] * E; out[o + 1] = s[1] * E; out[o + 2] = s[2] * E;
        return out;
    }

    // ── 6. 透明 resolve：同一份片元链表，五种做法 ──
    function cardOrder(cards, cam) {   // 按卡中心视深从远到近（每物体排序，不是每像素）
        const d = cards.map((cd) => { const k = cd.N >> 1; return HAIR.project(cam, [cd.c[3 * k], cd.c[3 * k + 1], cd.c[3 * k + 2]], 1, 1)[2]; });
        const rank = new Int32Array(cards.length);
        [...cards.keys()].sort((a, b) => d[b] - d[a]).forEach((j, r) => { rank[j] = r; });
        return rank;
    }
    const A2C_D = [0.125, 0.625, 0.875, 0.375];
    function resolve(mode, fr, alpha, col, rank, tau, W, H, out, outA) {
        const n = W * H, tmp = [];
        for (let i = 0; i < n; i++) {
            const s0 = fr.start[i], s1 = fr.start[i + 1];
            let r = BG[0], g = BG[1], b = BG[2], A = 0;
            if (mode === 'test') {
                for (let q = s0; q < s1; q++) { const f = fr.list[q]; if (alpha[f] > tau) { r = col[3 * f]; g = col[3 * f + 1]; b = col[3 * f + 2]; A = 1; break; } }
            } else if (mode === 'blendZ' || mode === 'blendCard') {
                tmp.length = 0; for (let q = s0; q < s1; q++) tmp.push(fr.list[q]);
                if (mode === 'blendZ') tmp.reverse(); else tmp.sort((x, y) => rank[fr.card[x]] - rank[fr.card[y]]);   // 远 → 近
                let T = 1;
                for (const f of tmp) { const a = alpha[f]; r = a * col[3 * f] + (1 - a) * r; g = a * col[3 * f + 1] + (1 - a) * g; b = a * col[3 * f + 2] + (1 - a) * b; T *= 1 - a; }
                A = 1 - T;
            } else if (mode === 'blendUnsorted') {   // 按卡编号画（与深度无关）+ 写深度：被先画的近片元挡住的远片元直接丢失
                tmp.length = 0; for (let q = s0; q < s1; q++) tmp.push(fr.list[q]);
                tmp.sort((x, y) => fr.card[x] - fr.card[y]);
                let zb = Infinity, T = 1;
                for (const f of tmp) { const a = alpha[f]; if (a < 1 / 255 || fr.z[f] > zb) continue; r = a * col[3 * f] + (1 - a) * r; g = a * col[3 * f + 1] + (1 - a) * g; b = a * col[3 * f + 2] + (1 - a) * b; T *= 1 - a; zb = fr.z[f]; }
                A = 1 - T;
            } else if (mode === 'a2c') {   // 4× MSAA：n = ⌊4α + dither⌋ 个样本；同一像素所有层用同一掩码 → 层间相关
                const x = i % W, y = (i / W) | 0, pat = (x & 1) + 2 * (y & 1);
                let cr = 0, cg = 0, cb = 0, cov = 0;
                for (let sIdx = 0; sIdx < 4; sIdx++) {
                    let hit = -1;
                    for (let q = s0; q < s1 && hit < 0; q++) { const f = fr.list[q], m = Math.min(4, Math.floor(alpha[f] * 4 + A2C_D[pat])); if (((sIdx - pat + 4) % 4) < m) hit = f; }
                    if (hit >= 0) { cr += col[3 * hit]; cg += col[3 * hit + 1]; cb += col[3 * hit + 2]; cov++; } else { cr += BG[0]; cg += BG[1]; cb += BG[2]; }
                }
                r = cr / 4; g = cg / 4; b = cb / 4; A = cov / 4;
            }
            out[3 * i] = r; out[3 * i + 1] = g; out[3 * i + 2] = b; outA[i] = A;
        }
    }
    function errStats(A, Aref, n) { let s = 0, d = 0, m = 0; for (let i = 0; i < n; i++) { if (A[i] < 0.004 && Aref[i] < 0.004) continue; s += Math.abs(A[i] - Aref[i]); d += A[i] - Aref[i]; m++; } return { mae: s / Math.max(1, m), bias: d / Math.max(1, m) }; }
    return { EB: 5, TW, TH, NM, RW, RH, CAM_SIZE, CAM_CENTER, BG, KK, clamp, dot, norm, hash, hash3, smooth, kmeans, fitCard, frameAt, makeAtlas, bakeTile, buildMips, coverageScales, rasterCards, sampleFrags, baseColor, offscreen, disp, lightDir, shiftT, strandSpec, shadeKK, bsdfSample, cardOrder, resolve, errStats };
})();

// ── 预览辅助：只画 stage 已算好的缓冲 ──
const showRGB = (c, rgb, W = HC.RW, H = HC.RH, exposure = 1) => PL.draw.pixels(c, W, H, (x, y, o) => HC.disp(rgb, y * W + x, o, exposure));
const showFront = (c, S, col) => PL.draw.pixels(c, HC.RW, HC.RH, (x, y, o) => { const f = S.front[y * HC.RW + x]; if (f < 0) { const d = HAIR.display(HC.BG.map((v) => v * 0.15)); o[0] = d[0]; o[1] = d[1]; o[2] = d[2]; return o; } return HC.disp(col, f, o); });
const GRID_COLS = [['ref', '发丝参考'], ['test', 'alpha test'], ['blendZ', 'blend·逐像素序'], ['blendCard', 'blend·按卡排序'], ['blendUnsorted', '未排序+写深度'], ['a2c', 'A2C 4×'], ['dither', 'dither+TAA']];
function gridData(S, key) {
    if (key === 'ref') return { rgb: S.refUnlit, A: S.R.alpha, err: null };
    if (key === 'dither') return { rgb: S.histU, A: S.histA, err: S.ditherErr };
    return S.grid[key];
}
function drawGrid(c, S) {
    const { ctx } = PL.draw.ctx(c, { bg: '#0d1014' }), W = HC.RW, H = HC.RH, gap = 9, x0 = 6, y0 = 34, tmp = [0, 0, 0];
    ctx.imageSmoothingEnabled = false; ctx.font = '10px Consolas, monospace'; ctx.textBaseline = 'alphabetic';
    GRID_COLS.forEach(([key, name], k) => {
        const d = gridData(S, key), X = x0 + k * (W + gap);
        ctx.drawImage(HC.offscreen(W, H, (x, y, o) => HC.disp(d.rgb, y * W + x, o)), X, y0, W, H);
        ctx.drawImage(HC.offscreen(W, H, (x, y, o) => { const i = y * W + x, e = HC.clamp(d.A[i] - S.R.alpha[i], -1, 1); PL.cmap.diverge(0.5 + 0.5 * e, tmp); o[0] = tmp[0] / 255; o[1] = tmp[1] / 255; o[2] = tmp[2] / 255; return o; }), X, y0 + H + 4, W, H);
        ctx.fillStyle = '#9aa'; ctx.fillText(name, X, 28);
        ctx.fillStyle = '#c8b46a'; ctx.fillText(d.err ? `${d.err.mae.toFixed(3)} / ${d.err.bias >= 0 ? '+' : ''}${d.err.bias.toFixed(3)}` : '—', X, y0 + 2 * H + 18);
    });
}
function pbCardMeans(S) {
    const F = S.fr.F, col = new Float32Array(3 * F);
    for (let f = 0; f < F; f++) { const k = S.pbCardN && S.pbCardN[f] ? 1 / S.pbCardN[f] : 0; col[3 * f] = S.pbCard[3 * f] * k; col[3 * f + 1] = S.pbCard[3 * f + 1] * k; col[3 * f + 2] = S.pbCard[3 * f + 2] * k; }
    return col;
}
function pbCardComposite(S, mode) {
    const n = HC.RW * HC.RH, rgb = new Float32Array(3 * n), A = new Float32Array(n);
    HC.resolve(mode, S.fr, mode === 'test' ? S.aTest : S.fs.a, pbCardMeans(S), S.rank, S.tau, HC.RW, HC.RH, rgb, A);
    return rgb;
}
function pbRefComposite(S) {
    const n = HC.RW * HC.RH, rgb = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { const a = S.R.id[i] >= 0 && S.pbRefCnt[i] ? S.R.alpha[i] : 0, k = a ? 1 / S.pbRefCnt[i] : 0; for (let c = 0; c < 3; c++) rgb[3 * i + c] = a * S.pbRef[3 * i + c] * k + (1 - a) * HC.BG[c]; }
    return rgb;
}
function drawComposite(c, S) {
    if (S.mode === 'dither') return showRGB(c, S.histS);
    if (S.shade === 'kk') return showRGB(c, S.comp);
    return showRGB(c, pbCardComposite(S, S.mode));
}
// ── 3D：烘焙出的 atlas（mip 0）贴回 ribbon，alpha test + Scheuermann 着色；右侧并排原始发丝 ──
function drawCards3d(c, S) {
    const M = S.atlas.mips[0], k3 = 0.12, ctr = HC.CAM_CENTER, DX = 0.8, V3 = PL.v3, col = [0, 0, 0];
    const P = (x, y, z, dx) => [(x - ctr[0]) * k3 + dx, (y - ctr[1]) * k3, (z - ctr[2]) * k3];   // cm → 视图单位，中心移到原点
    const meshes = S.cards.map((cd) => {
        const N = cd.N, ox = cd.tile[0] * M.tw, oy = cd.tile[1] * M.th, cP = [0, 0, 0], cB = [0, 0, 0], cT = [0, 0, 0], cN = [0, 0, 0];
        let i00 = 0, w00 = 0, w10 = 0, w01 = 0, w11 = 0;
        const tap = (u, v) => {   // tile 内 bilinear，寻址同 HC.sampleFrags：u → 横向 texel，v → 根 → 梢
            const x = HC.clamp(u * M.tw - 0.5, 0, M.tw - 1), y = HC.clamp(v * M.th - 0.5, 0, M.th - 1);
            const x0 = Math.min(M.tw - 2, Math.floor(x)), y0 = Math.min(M.th - 2, Math.floor(y)), ax = x - x0, ay = y - y0;
            i00 = (oy + y0) * M.W + ox + x0; w00 = (1 - ax) * (1 - ay); w10 = ax * (1 - ay); w01 = (1 - ax) * ay; w11 = ax * ay;
        };
        const bl = (A, st = 1, o = 0) => A[st * i00 + o] * w00 + A[st * (i00 + 1) + o] * w10 + A[st * (i00 + M.W) + o] * w01 + A[st * (i00 + M.W + 1) + o] * w11;
        const lerp3 = (A, v, out) => { const x = v * N, k = Math.min(N - 1, Math.floor(x)), f = x - k; for (let a = 0; a < 3; a++) out[a] = A[3 * k + a] + f * (A[3 * k + 3 + a] - A[3 * k + a]); return out; };
        return PL.mesh.param((u, v) => {   // ribbon：c(v) + (2u − 1)·w_half·B(v)，与 rasterCards 的顶点 / uv 相同
            lerp3(cd.c, v, cP); lerp3(cd.B, v, cB); const a = (2 * u - 1) * cd.halfW;
            return P(cP[0] + a * cB[0], cP[1] + a * cB[1], cP[2] + a * cB[2], -DX);
        }, [2, N], {
            twoSided: true,
            mask: (u, v) => { tap(u, v); return bl(M.A) > S.tau; },          // alpha test：clip(α − τ)
            shade: (px) => {                                                  // = PS_HairCard：flow 切线 + 平移高光 × AO
                tap(px.u, px.v); lerp3(cd.B, px.v, cB); lerp3(cd.T, px.v, cT); lerp3(cd.Nn, px.v, cN);
                const fx = bl(M.T, 3, 0), fy = bl(M.T, 3, 1), fz = bl(M.T, 3, 2);
                const T = HC.norm([fx * cB[0] + fy * cT[0] + fz * cN[0], fx * cB[1] + fy * cT[1] + fz * cN[1], fx * cB[2] + fy * cT[2] + fz * cN[2]]);
                HC.shadeKK(T, HC.norm(cN), px.view, S.L, bl(M.R), bl(M.I), bl(M.O), S.color, col, 0, true);
                return HAIR.display(col);
            },
        });
    });
    const lines = [];
    S.groom.strands.forEach((s, i) => {   // 每 8 根取 1 根原始发丝，平移到右侧
        if (i % 8) return;
        const pts = []; for (let k = 0; k < s.p.length / 3; k++) pts.push(P(s.p[3 * k], s.p[3 * k + 1], s.p[3 * k + 2], DX));
        const d = HAIR.display(HC.baseColor(S.color, 0.6, HC.hash(s.id != null ? s.id : i), [0, 0, 0]));
        lines.push({ pts, color: `rgba(${(255 * d[0]) | 0},${(255 * d[1]) | 0},${(255 * d[2]) | 0},0.85)`, width: 0.8 });
    });
    const o = [-DX, 0, 0];
    lines.push({ pts: [V3.add(o, V3.mul(S.L, 1.5)), V3.add(o, V3.mul(S.L, 0.6))], color: '#ffd866', width: 2.2, arrow: true, labelAt: 'start', label: 'L（光）' });
    PL.draw.mesh3d(c, {
        meshes, lines, light: S.L,
        labels: [{ p: [-DX, 1.08, 0], text: `插片 × ${S.Kc}（alpha test）`, color: '#8fc8ff' }, { p: [DX, 1.08, 0], text: '原始发丝（1/8）', color: '#d0a878' }],
        orbit: { yaw: 0.35, pitch: 0.12, dist: 4.3 },
    });
}

PipelineLab.mount({
    paramSteps: { K: ['why', 'cluster', 'fit', 'bake', 'atest', 'kk', 'pb', 'blend', 'shadow', 'lod'], yaw: ['why', 'fit', 'atest', 'kk', 'pb', 'blend', 'shadow', 'lod'], pitch: ['why', 'fit', 'atest', 'kk', 'pb', 'blend', 'shadow', 'lod'], color: ['why', 'bake', 'atest', 'kk', 'pb', 'blend', 'shadow', 'lod'], mode: ['why', 'blend'], mip: ['why', 'atest', 'kk', 'pb', 'blend', 'shadow', 'lod'], tau: ['bake', 'atest', 'blend', 'shadow', 'lod'], cp: ['blend', 'shadow'], shade: ['why', 'blend'], lyaw: ['why', 'bake', 'kk', 'pb', 'blend', 'shadow'], lel: ['why', 'bake', 'kk', 'pb', 'blend', 'shadow'] },   // generated by validate_doc --deps (param → steps whose previews change)
    title: '毛发渲染 04 · 插片 Hair Cards：烘焙 → 着色 → 透明 → LOD',
    lang: 'zh-CN',
    codeLang: 'hlsl',
    seed: 4,
    regen: false,

    params: [
        { id: 'K', label: 'cards', options: [8, 16, 32], value: 16, affects: 'resident', desc: '插片数量 = k-means 簇数，每簇一张卡、一个 24 × 192 的 atlas tile。改它会重新聚类、拟合、烘焙（Step 02–04）：K = 8 时卡宽、横向 texel 粗；K = 32 时卡窄、更贴合发丝，但三角形数翻倍（Step 01 的几何量对比）。' },
        { id: 'yaw', label: '相机 yaw', min: -90, max: 90, step: 1, value: 0, unit: '°', map: (d) => d * Math.PI / 180, desc: '正交相机绕 y 轴的转角，0° = 正对发片。拖到 ±60° 看 Step 08 的插片列整体变蓝——平卡的投影覆盖随 cos(视角) 缩小，圆柱形的发丝不会；±90° 时卡侧对相机，几乎只剩一条线（Step 03 的线框最直观）。' },
        { id: 'pitch', label: 'pitch', min: -40, max: 40, step: 1, value: 0, unit: '°', map: (d) => d * Math.PI / 180, desc: '相机俯仰。它和"光仰角"一起决定半向量 H 与发丝切线的夹角：拖动时 Step 06 / 07 的两条高光带沿发片上下移动、始终保持错开；卡面倾斜后各透明做法的覆盖也跟着变（Step 08）。' },
        { id: 'color', label: '发色', value: '#7a5236', map: HAIR.hex2lin, desc: '发色（sRGB 拾色 → 线性）：Kajiya-Kay / Scheuermann 的漫反射基色（再乘根 → 梢渐变与每根发丝的随机明暗）；物理 BSDF 的 σ<sub>a</sub> 由它反推（<code>sigmaAFromReflectance</code>），所以两种着色的色相一致（Step 06 / 07）。' },
        { id: 'mode', label: '透明', options: [{ value: 'test', label: 'alpha test' }, { value: 'blendCard', label: 'blend（按卡排序）' }, { value: 'a2c', label: 'alpha-to-coverage 4×' }, { value: 'dither', label: 'dither + TAA' }], value: 'dither', desc: '"着色合成"图用哪种透明做法：alpha test、按卡排序的 blend、alpha-to-coverage 4×、dither + TAA。只影响 Step 01 的中图和 Step 08 的着色合成；Step 08 的对照网格五种同时画，可以直接比较。' },
        { id: 'mip', label: 'mip / 距离', options: [{ value: 0, label: '0（近）' }, { value: 1, label: '1（≈1:1）' }, { value: 2, label: '2（2× 远）' }, { value: 3, label: '3（4× 远）' }], value: 1, desc: '屏幕分辨率固定，用 mip 级别等效"距离"：本视图 1 像素 ≈ 2 texel，mip 1 才是正确的级别，0 是欠采样（发丝断续、闪烁），2 / 3 相当于 2× / 4× 远。拖到 2、3 看 Step 05、08 的 alpha test 变稀（erosion），而 blend / A2C / dither 保持平均覆盖。' },
        { id: 'tau', label: 'α_ref', min: 0.1, max: 0.9, step: 0.05, value: 0.5, digits: 2, desc: 'alpha test 阈值 τ：<code>clip(α − τ)</code>。它同时决定保覆盖 mip 要维持的通过率、shadow map 里哪些 texel 算遮挡、Step 04 的 3D 插片丢掉哪些 texel。τ 越大，naive mip 的 erosion 出现得越早（Step 05 的红线）；也看 Step 08–10。' },
        { id: 'cp', label: '保覆盖 mip', value: true, desc: '<strong>保覆盖 mip</strong>（Castaño 2010）：每级 mip 把 α 乘一个缩放 s，使 alpha test 的通过率与 mip 0 一致。关掉后把 mip 拖到 2、3，Step 08 的 alpha test 列和 Step 09 的 shadow map 明显变稀；Step 05 把两种并排画，不受它影响。' },
        { id: 'shade', label: '着色', options: [{ value: 'kk', label: 'KK · Scheuermann' }, { value: 'bsdf', label: 'BSDF · Chiang' }], value: 'kk', desc: '"着色合成"用的模型：KK · Scheuermann（两条平移高光，Step 06）或 BSDF · Chiang（系列共享的物理 BSDF，每帧每片元随机 h、时域累积，Step 07）。只影响 Step 01 中图和 Step 08 的着色合成；切到 BSDF 后先是噪声，累积几十帧后收敛。' },
        { id: 'lyaw', label: '光 yaw', min: -90, max: 90, step: 1, value: 35, unit: '°', map: (d) => d * Math.PI / 180, desc: '方向光绕 y 轴的方位角，0° = 从观察者一侧照射。它移动高光、改变 shadow map 的投影方向：看 Step 06 / 07 的着色、Step 09 的阴影，以及 Step 04 的 3D 插片（同一盏光）。' },
        { id: 'lel', label: '光仰角', min: -60, max: 60, step: 1, value: -20, unit: '°', map: (d) => d * Math.PI / 180, desc: '方向光仰角，正值 = 来自上方。默认 −20°（略低于视线）时两条高光带落在发片中段；拖向 ±60° 看高光带沿发片上下移动（Step 06 / 07 与 Step 04 的 3D 插片），阴影随之变化（Step 09）。' },
    ],

    stages: [
        {
            id: 'groom', lifetime: 'resident', run(S) {
                S.groom = HAIR.makeGroom({});
                const N = S.groom.opts.segs;
                S.cum = S.groom.strands.map((s) => { const c = new Float32Array(N + 1), P = s.p; for (let k = 0; k < N; k++) c[k + 1] = c[k] + Math.hypot(P[3 * k + 3] - P[3 * k], P[3 * k + 4] - P[3 * k + 1], P[3 * k + 5] - P[3 * k + 2]); return c; });
            },
        },
        {
            id: 'cluster', lifetime: 'resident', run(S) {
                const r = HC.kmeans(S.groom.strands, S.K, PL.rng(1234 + S.K));
                S.assign = r.assign; S.Kc = r.K;
                S.clusterSize = new Int32Array(S.Kc); for (let i = 0; i < r.assign.length; i++) S.clusterSize[r.assign[i]]++;
            },
        },
        {
            id: 'fit', lifetime: 'resident', run(S) {
                const ids = [...Array(S.Kc)].map(() => []);
                S.assign.forEach((j, i) => ids[j].push(i));
                S.cards = ids.map((l) => HC.fitCard(S.groom.strands, l, S.cum));
            },
        },
        {
            id: 'bake', lifetime: 'resident', run(S) {
                S.atlas = HC.makeAtlas(S.Kc);
                S.cards.forEach((cd, j) => HC.bakeTile(cd, S.groom.strands, S.atlas, j));
                HC.buildMips(S.atlas);
            },
        },
        {
            id: 'view', lifetime: 'param', run(S) {
                S.cam = HAIR.ortho({ yaw: S.yaw, pitch: S.pitch, center: HC.CAM_CENTER, size: HC.CAM_SIZE });
                S.R = HAIR.raster(S.groom.strands, S.cam, HC.RW, HC.RH);
                const n = HC.RW * HC.RH, ref = PL.buf(S, 'refUnlit', 3 * n), c = [0, 0, 0];
                for (let i = 0; i < n; i++) {
                    const a = S.R.alpha[i];
                    if (S.R.id[i] >= 0) HC.baseColor(S.color, S.R.v[i], HC.hash(S.R.id[i]), c); else c.fill(0);
                    for (let k = 0; k < 3; k++) ref[3 * i + k] = a * c[k] + (1 - a) * HC.BG[k];
                }
            },
        },
        { id: 'cpmips', lifetime: 'param', run(S) { S.cpS = HC.coverageScales(S.atlas, S.tau); } },
        {
            id: 'frags', lifetime: 'param', run(S) {
                S.L = HC.lightDir(S.lyaw, S.lel); S.V = S.cam.toViewer;
                S.bs = HAIR.bsdf({ sigmaA: HAIR.sigmaAFromReflectance(S.color, 0.3), betaM: 0.3, betaN: 0.3, alpha: 2 });
                S.fr = HC.rasterCards(S.cards, S.cam, HC.RW, HC.RH);
                S.fs = HC.sampleFrags(S.fr, S.cards, S.atlas, S.mip, S.cpS.sc);
                S.aTest = S.cp ? S.fs.acp : S.fs.a;
                S.rank = HC.cardOrder(S.cards, S.cam);
                const n = HC.RW * HC.RH, front = PL.buf(S, 'front', n, Int32Array);
                for (let i = 0; i < n; i++) { front[i] = -1; for (let q = S.fr.start[i]; q < S.fr.start[i + 1]; q++) { const f = S.fr.list[q]; if (S.fs.a[f] > 0.02) { front[i] = f; break; } } }   // 第一层可见片元（只看着色）
            },
        },
        {
            id: 'shade', lifetime: 'param', run(S) {
                const F = S.fr.F, fs = S.fs, kk = PL.buf(S, 'colKK', 3 * F), kk0 = PL.buf(S, 'colKK0', 3 * F), un = PL.buf(S, 'colU', 3 * F);
                for (let f = 0; f < F; f++) {
                    const T = [fs.T[3 * f], fs.T[3 * f + 1], fs.T[3 * f + 2]], N = [fs.Nw[3 * f], fs.Nw[3 * f + 1], fs.Nw[3 * f + 2]];
                    HC.shadeKK(T, N, S.V, S.L, fs.R[f], fs.I[f], fs.O[f], S.color, kk, 3 * f, true);
                    HC.shadeKK(T, N, S.V, S.L, fs.R[f], fs.I[f], 1, S.color, kk0, 3 * f, false);
                    HC.baseColor(S.color, fs.R[f], fs.I[f], un, 3 * f);
                }
                // 发丝参考的 KK（同一模型，用发丝真实切线）——用于检查"插片切线"带来的差别
                const n = HC.RW * HC.RH, rk = PL.buf(S, 'refKK', 3 * n);
                for (let i = 0; i < n; i++) {
                    if (S.R.id[i] < 0) { rk[3 * i] = HC.BG[0]; rk[3 * i + 1] = HC.BG[1]; rk[3 * i + 2] = HC.BG[2]; continue; }
                    HC.shadeKK([S.R.tan[3 * i], S.R.tan[3 * i + 1], S.R.tan[3 * i + 2]], [0, 0, 1], S.V, S.L, S.R.v[i], HC.hash(S.R.id[i]), 1, S.color, rk, 3 * i, true);
                }
            },
        },
        {
            id: 'resolve', lifetime: 'param', run(S) {
                const n = HC.RW * HC.RH;
                S.grid = {};
                for (const m of ['test', 'blendZ', 'blendCard', 'blendUnsorted', 'a2c']) {
                    const rgb = PL.buf(S, 'g_' + m, 3 * n), A = PL.buf(S, 'gA_' + m, n);
                    HC.resolve(m, S.fr, m === 'test' ? S.aTest : S.fs.a, S.colU, S.rank, S.tau, HC.RW, HC.RH, rgb, A);
                    S.grid[m] = { rgb, A, err: HC.errStats(A, S.R.alpha, n) };
                }
                for (const [key, arr] of [['testNaive', S.fs.a], ['testCP', S.fs.acp]]) {
                    const rgb = PL.buf(S, key + '_rgb', 3 * n), A = PL.buf(S, key + '_A', n);
                    HC.resolve('test', S.fr, arr, S.colU, S.rank, S.tau, HC.RW, HC.RH, rgb, A);
                    S[key] = { rgb, A, err: HC.errStats(A, S.R.alpha, n) };
                }
                // 着色预览用"逐像素排序 blend"合成，只比较着色
                for (const [key, col] of [['kkZ', S.colKK], ['kk0Z', S.colKK0]]) { const rgb = PL.buf(S, key + '_rgb', 3 * n), A = PL.buf(S, key + '_A', n); HC.resolve('blendZ', S.fr, S.fs.a, col, S.rank, S.tau, HC.RW, HC.RH, rgb, A); S[key] = rgb; }
                const rkc = PL.buf(S, 'refKKc', 3 * n);
                for (let i = 0; i < n; i++) { const a = S.R.alpha[i]; for (let k = 0; k < 3; k++) rkc[3 * i + k] = S.R.id[i] >= 0 ? a * S.refKK[3 * i + k] + (1 - a) * HC.BG[k] : HC.BG[k]; }
                // 着色合成（KK）——非 dither 模式在这里一次算完
                if (S.mode !== 'dither') { S.comp = PL.buf(S, 'compKK', 3 * n); S.compA = PL.buf(S, 'compKKA', n); HC.resolve(S.mode, S.fr, S.mode === 'test' ? S.aTest : S.fs.a, S.colKK, S.rank, S.tau, HC.RW, HC.RH, S.comp, S.compA); }
            },
        },
        {
            // 阴影：光源正交相机下画插片（alpha test → shadow map）与发丝（参考不透明度）；屏幕前层片元回投到光空间
            id: 'shadow', lifetime: 'param', run(S) {
                const L = S.L, SW = 64, SH = 88, camL = HAIR.ortho({ yaw: Math.atan2(L[0], L[2]), pitch: Math.asin(-L[1]), center: HC.CAM_CENTER, size: HC.CAM_SIZE });
                const frL = HC.rasterCards(S.cards, camL, SW, SH), fsL = HC.sampleFrags(frL, S.cards, S.atlas, S.mip, S.cpS.sc), aT = S.cp ? fsL.acp : fsL.a;
                const smZ = PL.buf(S, 'smZ', SW * SH), z0 = PL.buf(S, 'smZ0', SW * SH), opC = PL.buf(S, 'smOpC', SW * SH);
                for (let i = 0; i < SW * SH; i++) {
                    smZ[i] = Infinity; z0[i] = Infinity; let T = 1;
                    for (let q = frL.start[i]; q < frL.start[i + 1]; q++) { const f = frL.list[q]; if (smZ[i] === Infinity && aT[f] > S.tau) smZ[i] = frL.z[f]; if (z0[i] === Infinity && fsL.a[f] > 0.02) z0[i] = frL.z[f]; T *= 1 - fsL.a[f]; }
                    opC[i] = 1 - T;
                }
                S.RL = HAIR.raster(S.groom.strands, camL, SW, SH);
                S.SW = SW; S.SH = SH; S.camL = camL;
                // 屏幕前层片元 → 世界 → 光空间
                const n = HC.RW * HC.RH, sb = PL.buf(S, 'shBin', n), se = PL.buf(S, 'shExp', n), k = HC.RH / HC.CAM_SIZE, cam = S.cam, sigma = 0.9;
                for (let i = 0; i < n; i++) {
                    const f = S.front[i]; if (f < 0) { sb[i] = NaN; se[i] = NaN; continue; }
                    const x = i % HC.RW, y = (i / HC.RW) | 0, a = (x - HC.RW / 2 + 0.5) / k, b = (HC.RH / 2 - y - 0.5) / k, z = S.fr.z[f];
                    const P = [0, 1, 2].map((c) => cam.center[c] + a * cam.right[c] + b * cam.up[c] + z * cam.fwd[c]);
                    const pl = HAIR.project(camL, P, SW, SH), lx = HC.clamp(Math.round(pl[0]), 0, SW - 1), ly = HC.clamp(Math.round(pl[1]), 0, SH - 1), li = ly * SW + lx;
                    sb[i] = pl[2] <= smZ[li] + 0.15 ? 1 : 0;                                  // 二值 alpha-tested shadow map
                    se[i] = Math.exp(-sigma * Math.max(0, pl[2] - z0[li] - 0.05));           // 深度差指数衰减（体积感近似）
                }
                const shc = PL.buf(S, 'shComp', 3 * n);
                for (let i = 0; i < n; i++) { const f = S.front[i], v = f < 0 ? 1 : 0.25 + 0.75 * se[i]; for (let c = 0; c < 3; c++) shc[3 * i + c] = HC.BG[c] + (S.kkZ[3 * i + c] - HC.BG[c]) * v; }
            },
        },
        {
            // LOD：同一相机、分辨率 ÷ z（= 距离 × z），比较总覆盖 Σα·z²（以 z = 1 的发丝为 1）
            id: 'lod', lifetime: 'param', lazy: true, run(S) {
                const Z = [1, 2, 4, 8], st = { strand: [], strandMin: [], strandDec: [], blend: [], testN: [], testCP: [] };
                const sum = (A) => { let s = 0; for (let i = 0; i < A.length; i++) s += A[i]; return s; };
                Z.forEach((z) => {
                    const W = Math.round(HC.RW / z), H = Math.round(HC.RH / z), cam = HAIR.ortho({ yaw: S.yaw, pitch: S.pitch, center: HC.CAM_CENTER, size: HC.CAM_SIZE }), z2 = (HC.RW * HC.RH) / (W * H);
                    st.strand.push(sum(HAIR.raster(S.groom.strands, cam, W, H).alpha) * z2);
                    st.strandMin.push(sum(HAIR.raster(S.groom.strands, cam, W, H, { minWidthPx: 1 }).alpha) * z2);
                    st.strandDec.push(sum(HAIR.raster(S.groom.strands, cam, W, H, { filter: (s) => s.id % z === 0, widthScale: z }).alpha) * z2);
                    const mip = Math.min(HC.NM - 1, 1 + Math.round(Math.log2(z))), fr = HC.rasterCards(S.cards, cam, W, H), fs = HC.sampleFrags(fr, S.cards, S.atlas, mip, S.cpS.sc);
                    let b = 0, tn = 0, tc = 0;
                    for (let i = 0; i < W * H; i++) {
                        let T = 1, gotN = 0, gotC = 0;
                        for (let q = fr.start[i]; q < fr.start[i + 1]; q++) { const f = fr.list[q]; T *= 1 - fs.a[f]; if (fs.a[f] > S.tau) gotN = 1; if (fs.acp[f] > S.tau) gotC = 1; }
                        b += 1 - T; tn += gotN; tc += gotC;
                    }
                    st.blend.push(b * z2); st.testN.push(tn * z2); st.testCP.push(tc * z2);
                });
                const ref = st.strand[0] || 1;
                Object.keys(st).forEach((k) => { st[k] = st[k].map((v) => v / ref); });
                S.lodZ = Z; S.lodStats = st;
                // 颜色匹配：头发区域平均（预乘）颜色，发丝参考 vs 插片 blend（不打光基色）
                const n = HC.RW * HC.RH, mean = (rgb, A) => { const m = [0, 0, 0]; let w = 0; for (let i = 0; i < n; i++) { w += A[i]; for (let c = 0; c < 3; c++) m[c] += rgb[3 * i + c] - (1 - A[i]) * HC.BG[c]; } return m.map((v) => v / Math.max(1e-6, w)); };   // 去预乘：Σ(C − (1−α)·BG) / Σα
                S.lodColor = { strand: mean(S.refUnlit, S.R.alpha), cards: mean(S.grid.blendZ.rgb, S.grid.blendZ.A) };
            },
        },
        {
            // dither + TAA：每帧每片元一个随机阈值（按 像素 × 卡 × 帧 hash，层间独立）→ 最近的通过者胜出 → 指数滑动平均
            id: 'taa', lifetime: 'history', run(S) {
                const n = HC.RW * HC.RH, hU = PL.buf(S, 'histU', 3 * n), hA = PL.buf(S, 'histA', n), hS = PL.buf(S, 'histS', 3 * n), cur = PL.buf(S, 'curU', 3 * n), curA = PL.buf(S, 'curA', n);
                if (S._taaEpoch !== S.epoch) { S._taaEpoch = S.epoch; S.taaN = 0; }
                const w = Math.max(1 / (S.taaN + 1), 0.1), fr = S.fr, fs = S.fs, fi = S.frame | 0, rnd = PL.rng(0x9e3779b9 ^ (fi * 7919 + 13)), tmp = [0, 0, 0];
                for (let i = 0; i < n; i++) {
                    let hit = -1;
                    for (let q = fr.start[i]; q < fr.start[i + 1]; q++) { const f = fr.list[q]; if (fs.a[f] > HC.hash3(i, fr.card[f], fi)) { hit = f; break; } }
                    let u0 = HC.BG[0], u1 = HC.BG[1], u2 = HC.BG[2], s0 = u0, s1 = u1, s2 = u2;
                    if (hit >= 0) {
                        u0 = S.colU[3 * hit]; u1 = S.colU[3 * hit + 1]; u2 = S.colU[3 * hit + 2];
                        if (S.shade === 'bsdf') { HC.bsdfSample(S.bs, [fs.T[3 * hit], fs.T[3 * hit + 1], fs.T[3 * hit + 2]], S.V, S.L, 2 * rnd() - 1, HC.EB, tmp, 0); s0 = tmp[0]; s1 = tmp[1]; s2 = tmp[2]; }
                        else { s0 = S.colKK[3 * hit]; s1 = S.colKK[3 * hit + 1]; s2 = S.colKK[3 * hit + 2]; }
                    }
                    cur[3 * i] = u0; cur[3 * i + 1] = u1; cur[3 * i + 2] = u2; curA[i] = hit >= 0 ? 1 : 0;
                    if (S.taaN === 0) { hU[3 * i] = u0; hU[3 * i + 1] = u1; hU[3 * i + 2] = u2; hA[i] = curA[i]; hS[3 * i] = s0; hS[3 * i + 1] = s1; hS[3 * i + 2] = s2; continue; }
                    hU[3 * i] += w * (u0 - hU[3 * i]); hU[3 * i + 1] += w * (u1 - hU[3 * i + 1]); hU[3 * i + 2] += w * (u2 - hU[3 * i + 2]);
                    hS[3 * i] += w * (s0 - hS[3 * i]); hS[3 * i + 1] += w * (s1 - hS[3 * i + 1]); hS[3 * i + 2] += w * (s2 - hS[3 * i + 2]);
                    hA[i] += w * (curA[i] - hA[i]);
                }
                S.taaN++;
                S.ditherErr = HC.errStats(hA, S.R.alpha, n);
            },
        },
        {
            // BSDF 累积（lazy）：发丝参考逐像素随机 h；插片逐片元随机 h（轮询 1/3 片元/帧）
            id: 'pbr', lifetime: 'history', lazy: true, run(S) {
                const n = HC.RW * HC.RH, F = S.fr.F, ref = PL.buf(S, 'pbRef', 3 * n), refN = PL.buf(S, 'pbRefCnt', n, Int32Array), cardS = PL.buf(S, 'pbCard', 3 * F), cardN = PL.buf(S, 'pbCardN', F, Int32Array), tmp = [0, 0, 0];
                if (S._pbEpoch !== S.epoch) { S._pbEpoch = S.epoch; S.pbN = 0; ref.fill(0); refN.fill(0); cardS.fill(0); cardN.fill(0); }
                if (S.pbN >= 192) return;
                const rnd = PL.rng(0x51ed27 ^ (S.pbN * 7919 + 3)), E = HC.EB;
                if (S.pbN < 128) for (let i = (S.pbN & 1); i < n; i += 2) {   // 轮询：每帧一半像素
                    if (S.R.id[i] < 0) continue;
                    refN[i]++;
                    HC.bsdfSample(S.bs, [S.R.tan[3 * i], S.R.tan[3 * i + 1], S.R.tan[3 * i + 2]], S.V, S.L, 2 * rnd() - 1, E, tmp, 0);
                    ref[3 * i] += tmp[0]; ref[3 * i + 1] += tmp[1]; ref[3 * i + 2] += tmp[2];
                }
                const fs = S.fs, ph = S.pbN % 3;
                for (let f = ph; f < F; f += 3) {
                    HC.bsdfSample(S.bs, [fs.T[3 * f], fs.T[3 * f + 1], fs.T[3 * f + 2]], S.V, S.L, 2 * rnd() - 1, E, tmp, 0);
                    cardS[3 * f] += tmp[0]; cardS[3 * f + 1] += tmp[1]; cardS[3 * f + 2] += tmp[2]; cardN[f]++;
                }
                S.pbN++;
                S.pbRefN = Math.min(S.pbN, 128) >> 1;
            },
        },
    ],

    intro: {
        title: '插片 Hair Cards · 从发丝 groom 烘焙到实时 LOD',
        flow: ['groom 发丝', '聚类 clump', '拟合 ribbon', 'card 空间烘焙 atlas', '切线 / KK / BSDF 着色', 'alpha test · blend · A2C · dither+TAA', '阴影', 'LOD 衔接'],
        html: '<p>插片（hair cards）= 一组带 alpha 纹理的四边形条带，每条代替一簇发丝。它是实时毛发的标准做法，也是 strand groom 远处的 LOD。本页所有插片都从系列共享的同一个 <code>HAIR.makeGroom()</code> 发型烘焙而来，所以每一步都能直接和发丝参考（<code>HAIR.raster</code>）对照。</p>' + HAIR.nav('cards'),
    },

    steps: [
        {
            id: 'why', title: '为什么用插片：成本、平台预算与 LOD 链', tag: 'WHY',
            desc: '一头真人发型约 10 万根发丝 × 每根数十段；即使只画 1440 根的这片 demo 也有 3.5 万段亚像素线段——每段都要覆盖 / 深度 / 着色，且宽度远小于像素，必须靠大量采样或解析覆盖才不闪。插片把一簇发丝预积分进纹理：几何降到每簇几十个三角形，透明与锯齿问题转移到 alpha 纹理上。代价是体积感、视差和自阴影都变成了"画出来的"。',
            notes: '左：发丝参考（HAIR.raster 的 OIT 覆盖 α，不打光基色）；中：同一发型烘焙出的插片，按参数栏的透明 / 着色 / mip 渲染；右：两者的真实几何量。插片的代价转移到了每像素：每个像素 2–4 层卡的 overdraw + 纹理采样。',
            previews: [
                { id: 'why_ref', label: '发丝参考', sub: 'α = 1 − Π(1 − c)，基色不打光', render: (c, S) => PL.draw.pixels(c, HC.RW, HC.RH, (x, y, o) => HC.disp(S.refUnlit, y * HC.RW + x, o)), probe: (x, y, S) => `α = ${S.R.alpha[y * HC.RW + x].toFixed(3)}` },
                { id: 'why_cards', label: '插片（参数栏设置）', sub: '透明 × 着色 × mip，见第 08 步', live: true, needs: ['pbr'], render: (c, S) => drawComposite(c, S) },
                {
                    id: 'why_cost', label: '几何量对比', sub: '发丝段数 vs 插片三角形（对数轴）', size: [360, 180],
                    render: (c, S) => {
                        const segs = S.groom.strands.length * S.groom.opts.segs, tris = S.cards.length * S.cards[0].N * 2;
                        PL.draw.plot(c, { logY: true, y: [10, 1e6], x: [0, 4], xLabel: '', yLabel: '个数', series: [
                            { data: [segs], x: [1], color: '#e8a25a', label: `发丝线段 ${segs}`, bar: true },
                            { data: [tris], x: [3], color: '#5ab0e8', label: `插片三角形 ${tris}`, bar: true },
                        ] });
                    },
                },
            ],
        },
        {
            id: 'cluster', title: '聚类：把发丝分成 clump', tag: 'CLUSTER',
            desc: '每张插片要代表一簇"长得差不多"的发丝，所以先聚类。groom 本身已有 guide / clump 结构（<code>strand.guide</code>），这里用更通用的 k-means：特征 = 根、中点、梢三处位置（根部权重更大，保证一张卡的发丝从头皮同一处长出）。簇数 K 就是插片数，参数栏可切 8 / 16 / 32。',
            formula: ['x_i = (1.6·p_root, p_mid, 0.7·p_tip) ∈ ℝ^9     // 每根发丝的特征', 'c_j = argmin Σ_i |x_i − μ_{c_i}|^2              // Lloyd 迭代 12 次，k-means++ 初始化'],
            previews: [
                {
                    id: 'cl_front', label: '簇着色（正视）', sub: '每种颜色 = 一张插片的发丝', size: [240, 336],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0d1014' }), cam = HAIR.ortho({ yaw: 0, pitch: 0, center: HC.CAM_CENTER, size: HC.CAM_SIZE });
                        ctx.globalAlpha = 0.5; ctx.lineWidth = 0.6;
                        S.groom.strands.forEach((s, i) => {
                            const j = S.assign[i], hue = (j * 137.508) % 360; ctx.strokeStyle = `hsl(${hue},70%,60%)`; ctx.beginPath();
                            for (let k = 0; k < s.p.length / 3; k++) { const p = HAIR.project(cam, [s.p[3 * k], s.p[3 * k + 1], s.p[3 * k + 2]], w, h); k ? ctx.lineTo(p[0] + 0.5, p[1] + 0.5) : ctx.moveTo(p[0] + 0.5, p[1] + 0.5); }
                            ctx.stroke();
                        });
                        ctx.globalAlpha = 1;   // 复位：画布上下文在重绘间保留状态，否则下次的背景填充只盖住一半（残影）
                    },
                },
                {
                    id: 'cl_size', label: '每簇发丝数', sub: 'k-means 不保证均匀', size: [360, 180],
                    render: (c, S) => PL.draw.plot(c, { x: [-0.5, S.Kc - 0.5], y: [0, Math.max(...S.clusterSize) * 1.15], xLabel: '簇 j', yLabel: '发丝数', series: [{ data: [...S.clusterSize], x: [...S.clusterSize].map((_, j) => j), color: '#7ac0a0', bar: true, label: 'N_j' }] }),
                },
            ],
        },
        {
            id: 'fit', title: '拟合插片几何：沿 clump 中心线的 ribbon', tag: 'FIT',
            desc: '对每簇：按弧长 s 等距采样每根发丝，取平均得中心线 c(s)（短发丝在超出长度后退出平均），1-2-1 平滑两遍；切线 T = dc/ds，宽度方向 B = normalize(N_scalp × T)，法线 N = T × B——卡面朝头皮外法线（这片发型是 +z）。半宽取簇内 |a|（发丝点到中心线的横向距离）的 95 分位 × 1.12；N 方向的偏移 d 不在几何里，而是烘焙进深度纹理。',
            formula: ['c(s) = mean_i p_i(s)                      // s ∈ [0, L_c]，L_c = 簇内发丝长度 90 分位', 'B = normalize(N_scalp × T),   N = T × B', 'a = (p − c)·B,   d = (p − c)·N             // card 空间坐标 (a, s, d)', 'w_half = 1.12 · P95(|a|)'],
            previews: [
                {
                    id: 'fit_overlay', label: '插片线框 × 发丝', sub: '当前相机；拖 yaw 看侧面', size: [240, 336],
                    render: (c, S) => {
                        const { ctx, w, h } = PL.draw.ctx(c, { bg: '#0d1014' });
                        ctx.globalAlpha = 0.35; ctx.lineWidth = 0.5; ctx.strokeStyle = '#c9a27a';
                        S.groom.strands.forEach((s, i) => { if (i % 3) return; ctx.beginPath(); for (let k = 0; k < s.p.length / 3; k++) { const p = HAIR.project(S.cam, [s.p[3 * k], s.p[3 * k + 1], s.p[3 * k + 2]], w, h); k ? ctx.lineTo(p[0] + 0.5, p[1] + 0.5) : ctx.moveTo(p[0] + 0.5, p[1] + 0.5); } ctx.stroke(); });
                        ctx.globalAlpha = 0.9; ctx.lineWidth = 1;
                        S.cards.forEach((cd, j) => {
                            ctx.strokeStyle = `hsl(${(j * 137.508) % 360},80%,62%)`;
                            const P = (k, sg) => HAIR.project(S.cam, [0, 1, 2].map((a) => cd.c[3 * k + a] + sg * cd.halfW * cd.B[3 * k + a]), w, h);
                            ctx.beginPath();
                            for (let k = 0; k <= cd.N; k++) { const a = P(k, -1), b = P(k, 1); ctx.moveTo(a[0] + 0.5, a[1] + 0.5); ctx.lineTo(b[0] + 0.5, b[1] + 0.5); }
                            for (const sg of [-1, 1]) for (let k = 0; k <= cd.N; k++) { const a = P(k, sg); k ? ctx.lineTo(a[0] + 0.5, a[1] + 0.5) : ctx.moveTo(a[0] + 0.5, a[1] + 0.5); }
                            ctx.stroke();
                        });
                        ctx.globalAlpha = 1;   // 复位（同上）
                    },
                },
                {
                    id: 'fit_dims', label: '每张卡的尺寸', sub: '半宽 w_half 与发丝离卡面最大距离 |d|max（cm）', size: [360, 180],
                    render: (c, S) => PL.draw.plot(c, { x: [-0.5, S.Kc - 0.5], y: [0, Math.max(...S.cards.map((cd) => Math.max(cd.halfW, cd.dMax))) * 1.2], xLabel: '卡 j', yLabel: 'cm', series: [
                        { data: S.cards.map((cd) => cd.halfW), x: S.cards.map((_, j) => j), color: '#5ab0e8', label: 'w_half' },
                        { data: S.cards.map((cd) => cd.dMax), x: S.cards.map((_, j) => j), color: '#e8a25a', label: '|d|max' },
                    ] }),
                },
            ],
        },
        {
            id: 'bake', title: '烘焙：在 card 空间光栅化簇内发丝 → atlas', tag: 'BAKE',
            desc: '把簇内每个发丝点换到 card 空间 (a, s, d)，横向乘 f = dv/du 使 tile 的 texel 成方形，再用系列共享的 <code>HAIR.raster</code>（解析亚像素覆盖、OIT α）以正交相机"从卡面正前方"光栅化——这就是烘焙器做的事。一次光栅化同时得到：覆盖 α、深度 d、切线（flow）、根→梢参数、发丝随机 ID；AO 由模糊后的累计覆盖近似。非 alpha 通道要做 dilation（向空 texel 外扩），否则 bilinear 和 mip 会把"空"的 0 混进来。',
            formula: ['p_card = (a·f, −s, d),   f = (L_c/TH) / (2 w_half/TW)   // tile 24 × 192 texel', 'α = 1 − Π_i (1 − c_i)          // HAIR.raster：每根发丝的解析覆盖 c_i', 'flow = (T·B, T·T_c, T·N)       // card 空间切线，存 RGB = 0.5 + 0.5·flow', 'AO ≈ exp(−1.1 · blur(Σ c_i)) · scalp(v)'],
            previews: [
                { id: 'atlas_a', label: 'Coverage α', sub: 'mip 0，每列一张卡', size: [384, 192], render: (c, S) => { const M = S.atlas.mips[0]; PL.draw.field(c, M.A, M.W, M.H, { range: [0, 1], map: 'gray' }); }, probe: (x, y, S) => { const M = S.atlas.mips[0]; return `α = ${M.A[y * M.W + x].toFixed(3)}  Σc = ${M.C[y * M.W + x].toFixed(2)}`; } },
                { id: 'atlas_d', label: 'Depth d', sub: '发丝离卡面距离（cm，蓝 = 卡后）', size: [384, 192], render: (c, S) => { const M = S.atlas.mips[0]; PL.draw.field(c, M.D.map((d, i) => (M.M[i] ? d : NaN)), M.W, M.H, { symmetric: true }); } },
                { id: 'atlas_t', label: 'Tangent / flow', sub: 'RGB = 0.5 + 0.5·(T·B, T·T_c, T·N)', size: [384, 192], render: (c, S) => { const M = S.atlas.mips[0]; PL.draw.pixels(c, M.W, M.H, (x, y, o) => { const i = y * M.W + x; o[0] = 0.5 + 0.5 * M.T[3 * i]; o[1] = 0.5 + 0.5 * M.T[3 * i + 1]; o[2] = 0.5 + 0.5 * M.T[3 * i + 2]; const a = 0.25 + 0.75 * Math.min(1, M.A[i] * 3); o[0] *= a; o[1] *= a; o[2] *= a; return o; }); } },
                { id: 'atlas_r', label: 'Root → tip', sub: '最近发丝的 v ∈ [0,1]', size: [384, 192], render: (c, S) => { const M = S.atlas.mips[0]; PL.draw.field(c, M.R, M.W, M.H, { range: [0, 1], map: 'viridis' }); } },
                { id: 'atlas_i', label: 'Strand ID', sub: 'hash(strand.id) ∈ [0,1)', size: [384, 192], render: (c, S) => { const M = S.atlas.mips[0]; PL.draw.field(c, M.I, M.W, M.H, { range: [0, 1], map: 'inferno' }); } },
                { id: 'atlas_o', label: 'AO', sub: '团内密度 × 根部头皮遮挡', size: [384, 192], render: (c, S) => { const M = S.atlas.mips[0]; PL.draw.field(c, M.O, M.W, M.H, { range: [0, 1], map: 'gray' }); } },
                { id: 'cards3d', large: true, label: '3D：烘焙后的插片 vs 原始发丝', sub: '左：每张卡一条 ribbon，贴 mip 0 atlas，α > α_ref 才保留（alpha test），flow 切线 + Scheuermann 着色；右：每 8 根取 1 的原始发丝。拖动旋转，侧看时卡是平的', render: (c, S) => drawCards3d(c, S) },
            ],
            notes: ['悬停 Coverage 可读 α 和累计覆盖 Σc。每张卡固定分到一个 24 × 192 的 tile，所以 atlas 随卡数增大（K = 8 → 192 × 192，16 → 384 × 192，32 → 384 × 384）；K = 8 时每张卡更宽，同样多的 texel 摊到更大的面积上，texel 更粗，K = 32 时卡窄、texel 更密。生产里通常反过来固定 atlas 预算——卡越多，每张卡分到的 texel 越少，这就是"卡数 ↔ 纹理密度"的预算交换。', '3D 视图把这份 atlas 贴回插片几何（与屏幕端 <code>rasterCards</code> 同一套顶点与 uv）：转到侧面，卡是一张平面，发丝的体积只剩 Depth 通道里的 d；alpha test 丢掉 α < α_ref 的 texel，发梢那段细发丝覆盖低，比右侧原始发丝短、稀。拖"α_ref"看卡变稀或变实。'],
            warn: 'tile 之间这里没有留 gutter（采样时在 tile 内 clamp 代替）；真实 atlas 必须留 ≥ 2^mip 个 texel 的边并做 dilation，否则 bilinear 与 mip 会串到隔壁 tile。',
        },
        {
            id: 'atest', title: 'Alpha test 与 mip 的 alpha erosion', tag: 'A-TEST',
            desc: '最便宜的透明：<code>clip(α − τ)</code>，写深度、无需排序。问题有两个：①二值化——发丝的半透明全部丢失，边缘锯齿、运动时闪烁；②<strong>alpha erosion</strong>：box 滤波的 mip 保住了平均 α，但细发丝的 α 被摊薄到 τ 以下，通过 alpha test 的 texel 越来越少——远处的头发"变秃"。Castaño 2010 的修正：生成 mip 时为每级找一个缩放 s，使 <code>frac(s·α_m > τ)</code> 等于 mip 0 的通过率。',
            formula: ['c_test(m) = frac(α_m > τ)                 // 随 m 增大而下降 = erosion', 'mean(α_m) = mean(α_0)                     // box 滤波保平均', 's_m : frac(min(1, s_m·α_m) > τ) = c_test(0) // 每 tile 每级二分查找'],
            code: {
                title: 'HLSL · CS_AlphaMipScale（coverage-preserving，每 tile 一个 group）', src: `
                groupshared uint gPass;
                [numthreads(64, 1, 1)]
                void CS_AlphaMipScale(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
                    uint2 tile = uint2(gid.x % AtlasCols, gid.x / AtlasCols); // 每个 group 处理一个 tile 的一级 mip
                    uint  mip  = gid.z + 1;                                   // mip 0 不缩放（s = 1）
                    float lo = 0.05, hi = 32;
                    for (int it = 0; it < 20; it++) {                      // 二分查找缩放 s
                        float s = sqrt(lo * hi);
                        if (gi == 0) gPass = 0;
                        GroupMemoryBarrierWithGroupSync();
                        for (uint t = gi; t < TexelsPerTile(mip); t += 64)
                            if (min(1, s * LoadAlpha(tile, mip, t)) > AlphaRef) InterlockedAdd(gPass, 1);
                        GroupMemoryBarrierWithGroupSync();
                        float cov = gPass / float(TexelsPerTile(mip));
                        if (cov < TargetCoverage[tile]) lo = s; else hi = s;   // Target = mip 0 的通过率
                        GroupMemoryBarrierWithGroupSync();
                    }
                    if (gi == 0) MipAlphaScale[uint3(tile, mip)] = sqrt(lo * hi);  // 写回：生成 mip 时 α *= s
                }` },
            previews: [
                {
                    id: 'mip_ladder', label: 'mip 0–3 · 同一张卡', sub: '左组 α · 中组 naive test · 右组保覆盖 test', size: [420, 236],
                    render: (c, S) => {
                        const { ctx, w } = PL.draw.ctx(c, { bg: '#0d1014' });
                        let j = 0; S.clusterSize.forEach((v, k) => { if (v > S.clusterSize[j]) j = k; });
                        const tx = j % S.atlas.cols, ty = Math.floor(j / S.atlas.cols), gap = 6, cw = 24, x0 = 8;
                        ctx.imageSmoothingEnabled = false; ctx.font = '10px Consolas, monospace'; ctx.fillStyle = '#888';
                        ['α', 'test', 'test · 保覆盖'].forEach((g, gi) => {
                            for (let m = 0; m < HC.NM; m++) {
                                const M = S.atlas.mips[m], s = S.cpS.sc[j * HC.NM + m];
                                const im = HC.offscreen(M.tw, M.th, (x, y, o) => { const a = M.A[(ty * M.th + y) * M.W + tx * M.tw + x], v = gi === 0 ? a : gi === 1 ? (a > S.tau ? 1 : 0) : (Math.min(1, a * s) > S.tau ? 1 : 0); o[0] = o[1] = o[2] = v; return o; });
                                const X = x0 + gi * (4 * (cw + 3) + gap * 2) + m * (cw + 3);
                                ctx.drawImage(im, X, 18, cw, 192);
                                ctx.fillText('m' + m, X + 3, 222);
                            }
                            ctx.fillText(g, x0 + gi * (4 * (cw + 3) + gap * 2), 12);
                        });
                        ctx.fillText(`卡 ${j}`, w - 40, 12);
                    },
                },
                {
                    id: 'mip_plot', label: 'alpha erosion 曲线', sub: '全部 tile 平均（atlas 空间）', size: [360, 200],
                    render: (c, S) => {
                        const st = S.cpS.stats, xs = st.map((_, m) => m);
                        PL.draw.plot(c, { x: [0, HC.NM - 1], y: [0, Math.max(0.6, ...st.map((s) => Math.max(s.mean, s.naive, s.cp)) ) * 1.1], xLabel: 'mip 级别', yLabel: '覆盖率', series: [
                            { data: st.map((s) => s.mean), x: xs, color: '#e8c25a', label: 'mean α（blend / A2C 保持）' },
                            { data: st.map((s) => s.naive), x: xs, color: '#e06c75', label: `frac(α > ${S.tau.toFixed(2)})` },
                            { data: st.map((s) => s.cp), x: xs, color: '#5ad07a', label: '保覆盖缩放后', dash: [4, 3] },
                        ] });
                    },
                },
                { id: 'test_naive', label: 'alpha test · naive mip', sub: '当前 mip 与 τ', render: (c, S) => showRGB(c, S.testNaive.rgb), probe: (x, y, S) => `cov ${S.testNaive.A[y * HC.RW + x]} · |Δα|均值 ${S.testNaive.err.mae.toFixed(3)}` },
                { id: 'test_cp', label: 'alpha test · 保覆盖 mip', sub: '同一 τ', render: (c, S) => showRGB(c, S.testCP.rgb), probe: (x, y, S) => `|Δα|均值 ${S.testCP.err.mae.toFixed(3)} · 偏差 ${S.testCP.err.bias.toFixed(3)}` },
            ],
            notes: ['把 mip 拖到 2、3：naive 那张图明显变稀，保覆盖那张基本不变；erosion 曲线红线下坠、绿线持平。τ 越大 erosion 越早出现。', '注意绿线保的是 <em>mip 0 的 alpha test 通过率</em>，不是平均 α（黄线）——对细发丝两者可能差很多，这也是 alpha test 头发普遍"偏稀或偏实"的原因之一。'],
            warn: 'mip 0 在本视图里是欠采样（1 像素 ≈ 2–3 texel）：切到 mip 0 会看到发丝断续、闪烁——这是 alpha test 的锯齿，不是烘焙问题。',
        },
        {
            id: 'kk', title: '插片着色 ①：flow 切线 + Kajiya-Kay / Scheuermann 平移高光', tag: 'SHADE',
            desc: '插片的几何法线对毛发没有意义，着色用的是从 flow 纹理解出的<strong>切线</strong>：T = flow.x·B + flow.y·T_c + flow.z·N。Kajiya-Kay 只有一条在 T ⟂ H 处的高光；Scheuermann 2004 借用 Marschner 的观察做了两处改动：沿法线平移切线得到两条错开的高光——主高光（表面反射，白、窄）移向发梢，次高光（带色素颜色、宽、被噪声遮罩调制成"闪点"）移向发根；漫反射用 lerp(0.25, 1, N·L) 代替 sin(T,L)（原文：没有正确自阴影时 sin(T,L) 过亮）。平移量再叠加每根发丝的随机值（这里取自 ID 通道），高光就不会是一条整齐的线。',
            formula: ['T\' = normalize(T + s·N)                               // ShiftTangent', 'spec(T\') = smoothstep(−1, 0, T\'·H) · (1 − (T\'·H)^2)^{p/2}', 'diffuse = saturate(lerp(0.25, 1, N·L))', 'C = (diffuse·base + k_1·spec(T_1) + k_2·mask·spec(T_2)·hair) · AO · E', 's_1 = +0.07 + ξ,  s_2 = −0.09 + ξ,  ξ = 0.12·(id − ½)   // T = 根 → 梢、N 朝外：s > 0 → 高光移向发梢；p_1 = 320, p_2 = 48'],
            code: {
                title: 'HLSL · PS_HairCard（Scheuermann 2004 风格）', src: `
                float3 ShiftTangent(float3 T, float3 N, float shift) { return normalize(T + shift * N); }
                float StrandSpecular(float3 T, float3 V, float3 L, float exponent) {
                    float3 H = normalize(L + V);
                    float dotTH = dot(T, H), sinTH = sqrt(1 - dotTH * dotTH);
                    float dirAtten = smoothstep(-1.0, 0.0, dotTH);           // 背向时衰减
                    return dirAtten * pow(sinTH, exponent);
                }
                float4 PS_HairCard(VSOut i, bool front : SV_IsFrontFace) : SV_Target {
                    float3 N = normalize(front ? i.N : -i.N);                 // 双面卡
                    float3 flow = CardTangent.Sample(BilinearClamp, i.uv) * 2 - 1;
                    float3 T = normalize(flow.x * i.B + flow.y * i.Tc + flow.z * i.N);
                    float4 attr = CardAttr.Sample(BilinearClamp, i.uv);       // root, id, ao
                    float  jit = (attr.y - 0.5) * ShiftJitter;
                    float3 t1 = ShiftTangent(T, N, PrimaryShift + jit);
                    float3 t2 = ShiftTangent(T, N, SecondaryShift + jit);
                    float3 base = HairColor * (0.5 + 0.5 * attr.x) * (0.8 + 0.4 * attr.y);
                    float3 diffuse = saturate(lerp(0.25, 1.0, dot(N, L))) * base;
                    float  mask = frac(attr.y * 13.37) * 0.75 + 0.25;        // 次高光闪点遮罩
                    float3 spec = Spec1 * StrandSpecular(t1, V, L, Exp1)
                                + Spec2 * mask * StrandSpecular(t2, V, L, Exp2) * sqrt(HairColor);
                    return float4((diffuse + spec) * attr.z * LightColor, CardCoverage.Sample(BilinearClamp, i.uv));
                }` },
            previews: [
                { id: 'kk_tan', label: '屏幕切线（前层）', sub: 'RGB = 0.5 + 0.5·T_world', render: (c, S) => PL.draw.pixels(c, HC.RW, HC.RH, (x, y, o) => { const f = S.front[y * HC.RW + x]; if (f < 0) { o[0] = o[1] = o[2] = 0.08; return o; } o[0] = 0.5 + 0.5 * S.fs.T[3 * f]; o[1] = 0.5 + 0.5 * S.fs.T[3 * f + 1]; o[2] = 0.5 + 0.5 * S.fs.T[3 * f + 2]; return o; }) },
                { id: 'kk_classic', label: '经典 Kajiya-Kay', sub: '一条高光，sin(T,L) 漫反射', render: (c, S) => showRGB(c, S.kk0Z) },
                { id: 'kk_sch', label: 'Scheuermann 两条平移高光', sub: '× AO，次高光带发色与闪点', render: (c, S) => showRGB(c, S.kkZ) },
                { id: 'kk_ref', label: '同模型 · 发丝真实切线', sub: '发丝参考：最近发丝 × α_ref', render: (c, S) => showRGB(c, S.refKKc) },
                {
                    id: 'kk_profile', label: '高光随切线角的剖面', sub: 'β = asin(T·H)；平移 s → 峰移到 β ≈ −atan(s)', size: [360, 200],
                    render: (c, S) => {
                        const prof = (s, p, k) => (bdeg) => { const b = bdeg * Math.PI / 180, th = (Math.sin(b) + s * Math.cos(b)) / Math.sqrt(1 + s * s); return k * HC.smooth(-1, 0, th) * Math.pow(Math.max(0, 1 - th * th), p / 2); };
                        PL.draw.plot(c, { x: [-40, 40], y: [0, 0.6], xLabel: 'β (°)', yLabel: '高光', series: [
                            { fn: prof(0, HC.KK.exp1, HC.KK.spec1), color: '#888', label: 'KK s = 0', dash: [4, 3] },
                            { fn: prof(HC.KK.shift1, HC.KK.exp1, HC.KK.spec1), color: '#e8e8e8', label: `主 s₁ = +${HC.KK.shift1}` },
                            { fn: prof(HC.KK.shift2, HC.KK.exp2, HC.KK.spec2), color: '#e0a060', label: `次 s₂ = ${HC.KK.shift2}` },
                        ], samples: 361 });
                    },
                },
            ],
            notes: ['拖"光仰角"或"相机 pitch"：两条高光带沿发片上下移动、始终保持错开。凸面上 T·H 由根到梢递减，峰值在 T·H ≈ −s，所以 s > 0（主高光）落在偏发梢处、s < 0（次高光）落在偏发根处——与 Marschner 2003 "次高光出现在主高光上方"的屏幕观察一致（Marschner 说的"R 偏根、TRT 偏梢"指的是散射锥的角度偏移，不是屏幕位置）。', '第 3、4 张对比：同一个着色模型，插片切线来自 flow 纹理的 bilinear 采样，比真实发丝切线平滑——插片高光天然更"整齐"，所以才需要 ID 抖动和闪点遮罩把它打散。'],
            warn: ['平移方向取决于 T 的朝向：本系列 T = 根 → 梢、N 朝外。Scheuermann 的 GDC 2004 幻灯片写的是"T 根→梢时正 shift 移向发根"，同年的 SIGGRAPH sketch 写的是"正 shift 移向发梢"——按凸面几何推导后者成立，本页按后者实现。', '与原文的差别：原 demo 的 shift 来自单独的 shift 纹理、AO 是逐顶点预计算、base 纹理也乘在高光上；这里 shift 抖动取自 ID 通道、AO 用烘焙通道、base 只乘漫反射，次高光单独乘 √发色。'],
        },
        {
            id: 'pb', title: '插片着色 ②：物理 BSDF 用在卡的切线上 + 深度偏移', tag: 'BSDF',
            desc: '把系列共享的 <code>HAIR.bsdf</code>（pbrt-v3 / Chiang 2016）直接用在 flow 切线上：每帧每片元随机取 h ∈ [−1,1]，时域累积 = 对发丝宽度积分（远场）。和发丝参考用同一个 BSDF、同一个光，差别只剩切线（flow 纹理 vs 真实发丝）——这就是插片 + 物理着色能达到的上限。插片是平的：与头部网格或其他卡相交时会出现生硬的直线交界，常用 <strong>pixel depth offset</strong>（PDO）把深度纹理 d 沿视线推回发丝真实位置，让交界变成随发丝起伏的锯齿状。',
            formula: ['L_o = E · ⟨ S(ω_i, ω_o; h) ⟩_h,   h ~ U[−1, 1]        // L_o = S·E，逐帧累积', 'frame = fiberFrame(T_flow, ω_o)   // n 朝 ω_o → φ_o = 0', 'z\' = z − d · (N·V) · DepthScale              // pixel depth offset（SV_DepthGreaterEqual 保住 early-Z）'],
            code: {
                title: 'HLSL · 插片上的物理 BSDF + PDO', src: `
                struct PSOut { float4 color : SV_Target; float depth : SV_DepthGreaterEqual; };
                PSOut PS_HairCardPB(VSOut i, bool front : SV_IsFrontFace) {
                    float3 N = front ? i.N : -i.N;
                    float3 T = DecodeFlow(CardTangent.Sample(BilinearClamp, i.uv), i.B, i.Tc, i.N);
                    float2 jit = float(FrameIndex % 64) * float2(47, 17) * 0.695;          // IGN 逐帧平移（UE 写法）
                    float  h = 2 * InterleavedGradientNoise(i.pos.xy + jit) - 1;            // 每帧换 h，TAA 负责积分
                    HairParams P = MakeHairParams(SigmaAFromReflectance(HairColor, BetaN), BetaM, BetaN, Alpha);  // 每材质常量（实际在 CPU 上算好）
                    float3 wi = ToFiberFrame(L, T, V), wo = ToFiberFrame(V, T, V);         // 纤维坐标：x = T，φo = 0
                    float3 Lo = HairBSDF_Chiang(wi, wo, h, P) * LightIrradiance;            // L_o = S·E
                    float  d = CardDepth.Sample(BilinearClamp, i.uv) * DepthScale;          // 发丝离卡面距离
                    PSOut o;
                    o.color = float4(Lo, 1);
                    o.depth = i.pos.z + max(0, -d * dot(N, V)) * DepthToNDC;                // 只往远推，保守深度
                    return o;
                }` },
            previews: [
                { id: 'pb_card', label: '插片 · HAIR.bsdf', sub: '逐片元随机 h 累积 → 逐像素排序 blend', live: true, needs: ['pbr'], render: (c, S) => showRGB(c, pbCardComposite(S, 'blendZ')) },
                { id: 'pb_ref', label: '发丝参考 · HAIR.bsdf', sub: '最近发丝切线，逐像素随机 h 累积 × α_ref', live: true, needs: ['pbr'], render: (c, S) => showRGB(c, pbRefComposite(S)), probe: (x, y, S) => `spp = ${S.pbRefN || 0} · α_ref = ${S.R.alpha[y * HC.RW + x].toFixed(2)}` },
                {
                    id: 'pb_pdo', label: 'PDO 偏移量', sub: '前层：−d·(N·V)（cm，红 = 推远）',
                    render: (c, S) => { const n = HC.RW * HC.RH, o = new Float32Array(n); for (let i = 0; i < n; i++) { const f = S.front[i]; if (f < 0) { o[i] = NaN; continue; } const N = [S.fs.Nw[3 * f], S.fs.Nw[3 * f + 1], S.fs.Nw[3 * f + 2]]; o[i] = -S.fs.D[f] * Math.abs(HC.dot(N, S.V)); } PL.draw.field(c, o, HC.RW, HC.RH, { symmetric: true }); },
                },
            ],
            notes: ['BSDF 的 σa 由"发色"反推，所以两种模型的整体色相一致；BSDF 里两条高光的错开来自鳞片倾角 α = 2°（pbrt-v3 写法：R 用 θ_o − 2α，TT 用 θ_o + α，TRT 用 θ_o + 4α），不需要手调 shift。', '刚改参数时两张图是噪声，约 100 帧后收敛（为控制每帧开销：发丝每帧更新一半像素、插片每帧更新 1/3 片元）。'],
            warn: '这里 BSDF 路径没有乘烘焙 AO，也没有阴影 / 多次散射——单根纤维的 S 只描述一次散射；插片要好看必须叠加阴影近似（第 09 步）和多次散射近似（见 03 篇）。',
        },
        {
            id: 'blend', title: '透明对照：test · blend · A2C · dither + TAA vs 发丝参考', tag: 'ALPHA',
            desc: '同一份片元链表（每像素 2–4 层卡），五种 resolve。<strong>Blend</strong> 需要从远到近排序，但引擎只能按物体（每张卡）排序——卡互相穿插、发片弯曲时每像素顺序就错了；若为了配合后续 pass 打开深度写，先画的近卡的透明 texel 会把后画的远卡挡掉（"空洞"）。<strong>Alpha-to-coverage</strong> 把 α 变成 MSAA 样本掩码，深度测试逐样本、不用排序，但只有 4 级，且同一像素各层用同一掩码 → 两层 α = 0.5 只得 0.5 而不是 0.75。<strong>Dither + TAA</strong>：每层用独立随机阈值，期望正好等于正确排序的 blend，噪声交给 TAA 历史累积。经典的折中是 Scheuermann 2004 的 4 pass：①alpha test 只放过不透明像素、写 Z、不写颜色（prime Z）②不透明区 Z-equal 着色 ③透明背面（剔除正面、不写 Z）④透明正面（剔除背面、写 Z）；发片在预处理时按离头部由内到外排进静态 index buffer，代替运行时排序。本页的节点图就是这条路线。',
            formula: ['C_blend = Σ_i α_i c_i Π_{j<i}(1 − α_j)          // 需要逐像素正确顺序', 'A2C:  mask_i = first ⌊4α_i + dither⌋ samples  // 各层同一掩码 → 相关', 'dither:  pass_i = α_i > ξ_i,  ξ_i 独立  ⇒  E[C] = C_blend', 'TAA:  H_t = lerp(H_{t−1}, C_t, max(1/(t+1), 0.1))'],
            code: {
                title: 'HLSL · 三种不排序的写法', src: `
                // ① alpha test（预深度 pass 与 shadow pass 同用）
                clip(CardCoverage.Sample(BilinearClamp, i.uv) - AlphaRef);
                // ② alpha-to-coverage：PSO 里 AlphaToCoverageEnable = TRUE，MSAA 4×；shader 照常输出 α
                return float4(color, alpha);
                // ③ hashed / dithered：阈值按 (像素, 物体, 帧) 取值，层间独立，期望 = 正确混合
                float xi = Hash3(uint3(i.pos.xy, PrimitiveSeed ^ FrameIndex));
                clip(alpha - xi);                                     // 之后 TAA 累积出半透明`,
            },
            previews: [
                { id: 'grid', label: '五种做法 × 发丝参考（不打光基色）', sub: '上：合成；下：Δα = α_方法 − α_发丝（红多蓝少）；数字 = 平均 |Δα| / 偏差', size: [660, 312], live: true, render: (c, S) => drawGrid(c, S) },
                { id: 'comp', label: '着色合成（参数栏的透明 × 着色）', sub: '当前 mip / τ / 模型', live: true, needs: ['pbr'], render: (c, S) => drawComposite(c, S) },
            ],
            notes: ['把 yaw 拖到 ±60°：所有插片列一起变蓝（偏差 ≈ −0.2）——平的卡投影覆盖随 cos(视角) 缩小，而圆柱形的发丝不会；这就是插片要多层交叉、按视角扭转的根本原因。同时"未排序 + 写深度"的空洞变多，A2C 的蓝点集中在多层重叠处。', 'mip 调大：alpha test 列迅速变蓝（erosion，见上一步），blend / A2C / dither 保持平均覆盖。dither 列刚改参数时是噪声，十几帧后收敛到"逐像素排序"那列。'],
            warn: ['Dither + TAA 的代价在运动：历史需要重投影，发丝这种亚像素细节最容易 ghosting / 拖影；静止截图里看不出来。', '对照里的"发丝参考"颜色只取最近一根发丝的基色，所以颜色误差只供参考；α 才是可比的量。'],
        },
        {
            id: 'shadow', title: '插片的阴影：alpha-tested shadow map 与自阴影近似', tag: 'SHADOW',
            desc: '阴影 pass 里插片也用 <code>clip(α − τ)</code> 写深度——这得到的是"硬"的、被 alpha 二值化的遮挡：一张卡要么全挡要么不挡，发丝间漏光的效果没了，自阴影在卡与卡的边界上一刀切。常见的廉价修补：用接收点与第一个遮挡者的<strong>深度差</strong>做指数衰减（越深入发团越暗），再乘烘焙 AO。真正的体积自阴影（Deep Opacity Maps 等）见 03 篇；插片同样可以写进 deep opacity，只是每张卡贡献的是 α 而不是单根发丝。',
            formula: ['V_bin = [ z_L ≤ SM_test(x_L) + bias ]                 // alpha-tested 深度', 'V_exp = exp(−σ · max(0, z_L − z_first(x_L)))         // σ ≈ 0.9 /cm，深度差衰减', 'C = C_shaded · (0.25 + 0.75 · V)                      // 0.25 = 环境项'],
            code: {
                title: 'HLSL · PS_ShadowDepth（alpha test）+ 接收端深度差衰减', src: `
                void PS_ShadowDepth(VSOut i) { clip(CardCoverage.Sample(BilinearClamp, i.uv) - AlphaRef); }   // 只写深度
                float HairCardShadow(float3 posWS) {
                    float3 l = mul(float4(posWS, 1), LightViewProj).xyz;              // 光空间
                    float zFirst = FirstOccluderDepth.SampleLevel(PointClamp, l.xy, 0);   // α > ε 的最近深度
                    return exp(-ShadowDensity * max(0, l.z - zFirst - ShadowBias));
                }` },
            previews: [
                { id: 'sh_sm', label: 'Shadow map（alpha test）', sub: '光空间深度，黑 = 没有通过 τ 的 texel', render: (c, S) => PL.draw.field(c, S.smZ.map((z) => (z === Infinity ? NaN : z)), S.SW, S.SH, { map: 'viridis' }) },
                { id: 'sh_op', label: '光空间不透明度', sub: '左：插片 1 − Π(1 − α)；右：发丝 HAIR.raster α', size: [264, 182], render: (c, S) => { const { ctx } = PL.draw.ctx(c, { bg: '#0d1014' }); ctx.imageSmoothingEnabled = false; ctx.drawImage(HC.offscreen(S.SW, S.SH, (x, y, o) => { o[0] = o[1] = o[2] = S.smOpC[y * S.SW + x]; return o; }), 4, 2, 124, 171); ctx.drawImage(HC.offscreen(S.SW, S.SH, (x, y, o) => { o[0] = o[1] = o[2] = S.RL.alpha[y * S.SW + x]; return o; }), 136, 2, 124, 171); } },
                { id: 'sh_bin', label: 'V_bin（屏幕前层）', sub: '二值：卡边界一刀切', render: (c, S) => PL.draw.field(c, S.shBin, HC.RW, HC.RH, { range: [0, 1] }) },
                { id: 'sh_exp', label: 'V_exp（深度差衰减）', sub: '越深入发团越暗', render: (c, S) => PL.draw.field(c, S.shExp, HC.RW, HC.RH, { range: [0, 1] }) },
                { id: 'sh_comp', label: 'KK × V_exp', sub: '逐像素排序 blend 合成', render: (c, S) => showRGB(c, S.shComp) },
            ],
            notes: '拖"光 yaw / 光仰角"：shadow map 里卡的轮廓随光转动；V_bin 只在卡与卡重叠处出现整块阴影，V_exp 随深入发团连续变暗。光空间不透明度两图的差别，就是 alpha 纹理丢掉的"发丝间漏光"。',
            warn: ['shadow pass 和主 pass 的 α_ref、mip 选择必须一致，否则阴影和可见头发对不上（远处主 pass 用保覆盖 mip、阴影用 naive mip 是常见 bug）。', '双面卡在阴影里也要关背面剔除；bias 要比卡的 |d|max 大，否则同一张卡自遮挡出条纹。'],
        },
        {
            id: 'lod', title: 'LOD 衔接：strands ↔ cards 的覆盖与颜色匹配', tag: 'LOD',
            desc: 'LOD 切换时最显眼的不是形状而是<strong>密度和亮度跳变</strong>。这里在同一相机下把分辨率除以 z（= 距离 × z），统计头发的总覆盖 Σα·z²（以近处发丝为 1）。解析覆盖的发丝在远处仍守恒；引擎为了不闪通常给发丝设最小像素宽度——远处反而<strong>变厚</strong>；按 1/z 抽稀并把宽度 × z 是 strand LOD 的标准补偿。插片若用 naive mip 的 alpha test 就会变稀；blend 或保覆盖 mip 才能和发丝的曲线对齐——两条曲线在切换距离处的比值就是需要的 α 缩放。',
            formula: ['cov(z) = z^2 · Σ_pixels α(z) / cov_strand(1)', 'strand LOD:  keep 1/z strands,  width × z     // 期望覆盖守恒', 'α_card ← α_card · cov_strand(z_switch) / cov_card(z_switch)'],
            previews: [
                {
                    id: 'lod_cov', label: '总覆盖 vs 距离', sub: '1 = 近处发丝；插片 mip = 1 + log₂z', size: [380, 220], needs: ['lod'],
                    render: (c, S) => { const st = S.lodStats, xs = S.lodZ; PL.draw.plot(c, { logX: true, x: [1, 8], y: [0, Math.max(1.6, ...Object.values(st).flat()) * 1.05], xLabel: '距离倍数 z', yLabel: '相对覆盖', hmarks: [{ y: 1, label: '' }], series: [
                        { data: st.strand, x: xs, color: '#e8e8e8', label: '发丝（解析覆盖）' },
                        { data: st.strandMin, x: xs, color: '#e06c75', label: '发丝 · 最小 1 px' },
                        { data: st.strandDec, x: xs, color: '#c678dd', label: '发丝 · 抽稀 1/z × 宽 z', dash: [4, 3] },
                        { data: st.blend, x: xs, color: '#5ab0e8', label: '插片 blend' },
                        { data: st.testN, x: xs, color: '#e0a060', label: '插片 test · naive mip' },
                        { data: st.testCP, x: xs, color: '#5ad07a', label: '插片 test · 保覆盖', dash: [4, 3] },
                    ] }); },
                },
                {
                    id: 'lod_width', label: '屏幕上的宽度 vs 距离', sub: '卡宽 < 1 px 时该合并插片（K ÷ 2）', size: [360, 220],
                    render: (c, S) => { const k = HC.RH / HC.CAM_SIZE, wc = 2 * S.cards.reduce((a, cd) => a + cd.halfW, 0) / S.cards.length * k, ws = 0.008 * k; PL.draw.plot(c, { logX: true, logY: true, x: [1, 64], y: [0.001, 30], xLabel: '距离倍数 z', yLabel: '像素', hmarks: [{ y: 1, label: '1 px' }], series: [
                        { fn: (z) => wc / z, color: '#5ab0e8', label: `卡宽（K=${S.Kc}）` }, { fn: (z) => wc * 2 / z, color: '#5ab0e8', dash: [4, 3], label: '卡宽（K/2 合并后≈×2）' }, { fn: (z) => ws / z, color: '#e8e8e8', label: '单根发丝宽' },
                    ] }); },
                },
                {
                    id: 'lod_color', label: '颜色匹配（z = 1）', sub: '去预乘的平均发色 Σ(C − (1−α)·BG) / Σα', size: [320, 190], needs: ['lod'],
                    render: (c, S) => { const a = S.lodColor.strand, b = S.lodColor.cards; PL.draw.plot(c, { x: [-0.5, 6.5], y: [0, Math.max(...a, ...b) * 1.5], xLabel: 'R · G · B（左发丝 / 右插片）', yLabel: '线性', series: [
                        { data: a, x: [0, 2.5, 5], color: '#e8e8e8', bar: true, label: '发丝' }, { data: b, x: [1, 3.5, 6], color: '#5ab0e8', bar: true, label: '插片 blend' },
                    ] }); },
                },
            ],
            notes: ['红线（最小像素宽）远处覆盖暴涨——头发远看"变粗变实"；紫线抽稀 + 加宽基本守恒；橙线 naive mip 下坠，绿线保覆盖但保的是 mip 0 的 test 通过率，不一定等于发丝的真实覆盖。', '颜色条若明显不等，切 LOD 时要在插片材质里补一个 tint / 亮度系数（生产中常按 LOD 存一组匹配参数）。'],
            warn: '只匹配覆盖还不够：发丝的高光来自每根发丝的真实切线，插片的切线被 flow 纹理平滑过，高光会更"整齐"更亮——LOD 切换处常要把插片的高光强度单独压一点。',
        },
        {
            id: 'prod', title: '教学版 → 生产版：预算、工具链与常见坑', tag: 'PROD',
            desc: '本页把每一步都做成了能看见的最小实现；生产里的差别主要在规模、自动化和"美术可控"。',
            blocks: [
                { type: 'html', html: `<table style="width:100%;border-collapse:collapse;font-size:0.84rem;line-height:1.5">
<tr style="color:#9ab"><th style="text-align:left;padding:4px 8px">环节</th><th style="text-align:left;padding:4px 8px">本页</th><th style="text-align:left;padding:4px 8px">生产版</th></tr>
<tr><td style="padding:4px 8px">卡片生成</td><td style="padding:4px 8px">k-means + 平均中心线</td><td style="padding:4px 8px">美术手摆卡（最常见）或由 groom 自动生成后人工修整。例：UE 的 Groom 资源每级 LOD 可选 strands / cards / meshes；Hair Card Generator 插件（5.4 起，Experimental）从 groom 生成卡与 atlas，可设卡数、纹理数（相似的卡共用纹理）、atlas 尺寸、"从上一级 LOD 精简"等</td></tr>
<tr><td style="padding:4px 8px">纹理</td><td style="padding:4px 8px">每卡唯一 tile，24×192，无 gutter</td><td style="padding:4px 8px">2K–4K atlas，少量发簇变体被很多张卡复用；tile 间留 gutter + dilation；通道打包后压缩。UE 的 cards 纹理槽与本页烘焙的通道几乎一一对应：Depth、Coverage、Tangent、Attributes（RootUV / CoordU / Seed）、Material（Color / Roughness）、Auxiliary</td></tr>
<tr><td style="padding:4px 8px">层次</td><td style="padding:4px 8px">一层 16 张</td><td style="padding:4px 8px">内层（不透明、遮头皮）→ 中层 → 外层碎发，由内向外分层摆放；绘制顺序固定成"由内到外"代替运行时排序</td></tr>
<tr><td style="padding:4px 8px">透明</td><td style="padding:4px 8px">五种 resolve 并列</td><td style="padding:4px 8px">组合使用：alpha test 预深度（大 τ）写不透明核心 + 半透明边缘用 dither/TAA 或排序 blend；保覆盖 mip 或 A2C 防 erosion</td></tr>
<tr><td style="padding:4px 8px">着色</td><td style="padding:4px 8px">KK-Scheuermann / 逐帧随机 h</td><td style="padding:4px 8px">引擎的毛发 shading model（近似 Marschner，含多次散射近似）；h 积分用解析近似或 LUT 而不是蒙特卡罗</td></tr>
<tr><td style="padding:4px 8px">阴影</td><td style="padding:4px 8px">二值 SM + 深度差</td><td style="padding:4px 8px">alpha-tested CSM + 屏幕空间接触阴影 + 卡也写入 deep opacity / 体素密度</td></tr>
<tr><td style="padding:4px 8px">LOD</td><td style="padding:4px 8px">覆盖统计</td><td style="padding:4px 8px">strands → cards → 简化网格，每级存覆盖 / 颜色 / 高光的匹配系数，切换时短时间交叉淡化（dither fade）</td></tr>
</table>` },
                { type: 'warn', html: '<strong>常见坑</strong>：① atlas 无 gutter / 非 alpha 通道不 dilation → 远处发丝边缘发黑或串色；② mip 用 box 滤波直接下采样 alpha → 远处变秃；③ 插片切线方向（根 → 梢）在 DCC 里翻转 → 平移高光跑反；④ 卡侧对相机时细成一条线——卡要按视角做 billboard 式扭转或多层交叉；⑤ 透明部分不写深度 → DOF、SSR、运动矢量与 TAA 都把头发当成背景，产生拖影；⑥ 阴影与主 pass 的 α_ref 不一致。' },
                { type: 'note', html: '想看 strand 路线本身的光栅化与覆盖，见 02 篇；自阴影与多次散射见 03 篇；着色模型细节见 01 篇。' },
            ],
        },
    ],

    graph: {
        intro: '<strong>RDG 风格</strong>：烘焙（橙色）离线做一次、常驻；运行时每帧 = 阴影 → 预深度（alpha test）→ base pass（不透明部分）→ 前向半透明 / dither → TAA。同名资源自动连线。',
        types: {
            mesh: { color: '#e8a25a', label: 'mesh · 插片 VB/IB' },
            tex: { color: '#a45fa5', label: 'tex · atlas 纹理' },
            depth: { color: '#4a8aff', label: 'depth · 深度' },
            color: { color: '#5ae87a', label: 'color · 颜色 RT' },
            cb: { color: '#e8c25a', label: 'cb · cbuffer' },
        },
        common: {
            title: 'Common.hlsl · 系列共享 + 插片 atlas 布局',
            blocks: [
                { label: '━ 系列共享 ━', code: HAIR.HLSL_COMMON },
                { label: '━ 系列共享 · 毛发 BSDF（HLSL 参考实现）━', code: HAIR.HLSL_BSDF },
                { label: '━ 插片 atlas ━', code: `
                    // atlas：每张卡一个 tile，第 j 张在 (j % AtlasCols, j / AtlasCols)；通道 = Coverage R8 · Depth R16F · Tangent RGB10A2 · Attr RGBA8(root, id, ao, –)
                    static const uint TileW = 24;    // mip 0 的 tile 宽：横向 a（u = 0 → 1）
                    static const uint TileH = 192;   // mip 0 的 tile 高：沿发丝 s（第 0 行 = 发根）
                    Texture2D<float>  CardCoverage;  Texture2D<float> CardDepth;
                    Texture2D<float3> CardTangent;   Texture2D<float4> CardAttr;
                    SamplerState      BilinearClamp;
                    cbuffer CardCB : register(b2) { float AlphaRef; float DepthScale; uint AtlasCols; uint _pad; };   // AtlasCols = min(K, 16)` },
            ],
        },
        nodes: [
            {
                id: 'viewcb', title: 'View / Light CB', tag: 'CB', category: 'system', desc: 'ViewProj · L · E',
                outputs: [{ name: 'ViewCB', format: 'cbuffer b0', type: 'cb' }],
                detail: { desc: '引擎注入的相机与方向光。', preview: null },
            },
            {
                id: 'cluster', title: 'CPU_ClusterFitCards', tag: 'CPU', category: 'bake', lifetime: 'resident', step: 'cluster', desc: 'k-means → ribbon',
                dispatch: { kind: 'CPU', name: 'ClusterFitCards', count: (S) => `${S.groom.strands.length} strands → ${S.Kc} cards`, note: '离线工具 / 编辑器里执行' },
                outputs: [{ name: 'CardVB', format: 'float3 pos · float2 uv · frame', type: 'mesh' }],
                detail: { preview: 'fit_overlay' },
            },
            {
                id: 'bake', title: 'PS_BakeCardAtlas', tag: 'PS', category: 'bake', lifetime: 'resident', step: 'bake', desc: 'card 空间光栅化发丝',
                dispatch: { kind: 'PS', name: 'PS_BakeCardAtlas', target: (S) => `${S.Kc} tiles × ${HC.TW}×${HC.TH}（MRT）`, note: '每 tile 画该簇发丝；RT0 乘法混合累计 Π(1−c)，其余 RT 深度测试取最近发丝' },
                inputs: [{ name: 'CardVB', format: 'card frames', type: 'mesh' }],
                outputs: [{ name: 'CardAtlas', format: 'Coverage R8 · Depth R16F · Tangent RGB10A2 · Attr RGBA8', type: 'tex' }],
                detail: { preview: 'atlas_a' },
            },
            {
                id: 'mips', title: 'CS_AlphaMipScale', tag: 'CS', category: 'bake', lifetime: 'param', step: 'atest', desc: 'box mip + 保覆盖缩放 s_m',
                dispatch: { name: 'CS_AlphaMipScale', threads: [64, 1, 1], groups: (S) => [S.Kc, 1, HC.NM - 1], idRange: 'SV_GroupID = (tile, –, mip)', note: '每 group 二分查找一个 tile 一级 mip 的缩放；离线随 atlas 一起做（τ 改了才重做）' },
                inputs: [{ name: 'CardAtlas', format: 'mip 0', type: 'tex' }],
                outputs: [{ name: 'CardAtlasMips', format: '4 mips · α 已缩放', type: 'tex' }],
                detail: { preview: 'mip_plot' },
            },
            {
                id: 'shadow', title: 'PS_ShadowDepth', tag: 'PS', lifetime: 'frame', step: 'shadow', desc: 'clip(α − τ) 只写深度',
                dispatch: { kind: 'PS', name: 'PS_ShadowDepth', target: (S) => `ShadowDepth ${S.SW}×${S.SH}`, count: (S) => `${S.Kc} cards × ${S.cards[0].N * 2} tris`, note: '双面、无颜色写入' },
                inputs: [{ name: 'CardVB', format: 'card frames', type: 'mesh' }, { name: 'CardAtlasMips', format: 'Coverage', type: 'tex' }, { name: 'ViewCB', format: 'LightViewProj', type: 'cb' }],
                outputs: [{ name: 'ShadowDepth', format: 'D32F', type: 'depth' }],
                detail: { preview: 'sh_sm' },
            },
            {
                id: 'prepass', title: 'PS_DepthPrepass', tag: 'PS', lifetime: 'frame', step: 'atest', desc: 'alpha test 写不透明核心深度',
                dispatch: { kind: 'PS', name: 'PS_DepthPrepass', target: 'SceneDepth（viewport）', count: (S) => `${S.fr.F} fragments`, note: 'clip(α − τ)；dither 模式下用 hashed 阈值代替 τ' },
                inputs: [{ name: 'CardVB', format: 'card frames', type: 'mesh' }, { name: 'CardAtlasMips', format: 'Coverage', type: 'tex' }, { name: 'ViewCB', format: 'ViewProj', type: 'cb' }],
                outputs: [{ name: 'SceneDepth', format: 'D32F', type: 'depth' }],
                detail: { preview: 'test_cp' },
            },
            {
                id: 'base', title: 'PS_HairCardBase', tag: 'PS', lifetime: 'frame', step: 'kk', desc: 'Z-equal · KK 平移高光 / BSDF',
                dispatch: { kind: 'PS', name: 'PS_HairCard', target: 'SceneColor（viewport）', idRange: 'SV_Position.xy ∈ viewport', note: 'depth test EQUAL：只着色预深度留下的片元（零 overdraw）' },
                inputs: [{ name: 'CardVB', format: 'card frames', type: 'mesh' }, { name: 'CardAtlasMips', format: 'all channels', type: 'tex' }, { name: 'SceneDepth', format: 'D32F', type: 'depth' }, { name: 'ShadowDepth', format: 'D32F', type: 'depth' }, { name: 'ViewCB', format: 'View · Light', type: 'cb' }],
                outputs: [{ name: 'SceneColor', format: 'RGBA16F', type: 'color' }],
                detail: { preview: 'kk_sch' },
            },
            {
                id: 'transl', title: 'PS_HairCardTranslucent', tag: 'PS', lifetime: 'frame', step: 'blend', desc: '按卡排序 blend α < τ 的边缘',
                dispatch: { kind: 'PS', name: 'PS_HairCardTranslucent', target: 'SceneColor（blend: SrcAlpha, InvSrcAlpha）', note: 'CPU 按卡中心深度从远到近排序；depth test LESS、不写深度' },
                inputs: [{ name: 'CardVB', format: 'sorted', type: 'mesh' }, { name: 'CardAtlasMips', format: 'all channels', type: 'tex' }, { name: 'SceneDepth', format: 'D32F', type: 'depth' }, { name: 'SceneColor', format: 'RGBA16F', type: 'color' }],
                outputs: [{ name: 'SceneColorHair', format: 'RGBA16F', type: 'color' }],
                detail: { preview: 'grid' },
            },
            {
                id: 'taa', title: 'PS_TAAResolve', tag: 'PS', category: 'output', lifetime: 'history', step: 'blend', desc: 'lerp(history, current, 0.1)',
                dispatch: { kind: 'PS', name: 'PS_TAAResolve', target: 'TAAHistory（viewport）', note: 'dither 透明与随机 h 的噪声都在这里被平均掉；运动时需重投影 + 邻域 clamp' },
                inputs: [{ name: 'SceneColorHair', format: 'RGBA16F', type: 'color' }, { name: 'TAAHistory', format: 'RGBA16F', type: 'color', history: true }],
                outputs: [{ name: 'TAAHistory', format: 'RGBA16F', type: 'color' }],
                detail: { preview: 'comp', code: { title: 'HLSL · PS_TAAResolve（静止相机简化版）', src: `
                    float4 PS_TAAResolve(float4 pos : SV_Position) : SV_Target {
                        float3 cur  = SceneColorHair[pos.xy].rgb;
                        float3 hist = TAAHistory.SampleLevel(BilinearClamp, Reproject(pos.xy), 0).rgb;
                        hist = clamp(hist, NeighborhoodMin(pos.xy), NeighborhoodMax(pos.xy));   // 运动时防拖影
                        return float4(lerp(hist, cur, max(1.0 / (FrameCount + 1), 0.1)), 1);
                    }` } },
            },
        ],
    },

    library: {
        title: '附录 · 辅助函数（HLSL）',
        intro: '步骤代码与节点代码里调用、但不属于某一步主线的小函数。atlas 布局与资源声明见 <code>Common.hlsl · 插片 atlas</code>，毛发 BSDF 与随机数（<code>PcgHash</code>、<code>InterleavedGradientNoise</code>、<code>HairBSDF_Chiang</code>）见系列共享块；这里的实现与本页 JS 参考实现的约定一一对应（tile 寻址同 <code>sampleFrags</code>，flow 编码同烘焙，BSDF 参数化同 <code>HAIR.bsdf</code>）。',
        blocks: [
            {
                title: 'atlas 寻址 · CS_AlphaMipScale（Step 05）', code: `
                // 第 mip 级一个 tile 的尺寸：mip 0 为 TileW × TileH（24 × 192），每级各边减半
                uint2 TileSize(uint mip) { return uint2(TileW, TileH) >> mip; }
                // 一个 tile 在第 mip 级的 texel 数（24·192 → 12·96 → 6·48 → 3·24）
                uint TexelsPerTile(uint mip) { uint2 s = TileSize(mip); return s.x * s.y; }
                // tile 内第 t 个 texel 的 α（行优先：t % 宽 = 横向，t / 宽 = 沿发丝，第 0 行 = 发根）；读的是 box mip，尚未乘缩放 s
                float LoadAlpha(uint2 tile, uint mip, uint t) {
                    uint2 s = TileSize(mip);
                    uint2 texel = tile * s + uint2(t % s.x, t / s.x);
                    return CardCoverage.Load(int3(texel, mip));
                }`,
            },
            {
                title: 'flow 解码与 BSDF 参数 · PS_HairCardPB（Step 07）', code: `
                // flow 纹理 → 世界空间切线（根 → 梢）：RGB = 0.5 + 0.5·(T·B, T·T_c, T·N)，B / T_c / N = 顶点插值的卡帧
                float3 DecodeFlow(float3 rgb, float3 B, float3 Tc, float3 N) {
                    float3 f = rgb * 2 - 1;
                    return normalize(f.x * B + f.y * Tc + f.z * N);
                }
                // 由目标发色反推吸收系数 σa（Chiang 2016 的拟合，同 HAIR.sigmaAFromReflectance）
                float3 SigmaAFromReflectance(float3 c, float betaN) {
                    float b = betaN, d = 5.969 - 0.215 * b + 2.532 * b * b - 10.73 * pow(b, 3) + 5.574 * pow(b, 4) + 0.245 * pow(b, 5);
                    float3 l = log(max(c, 1e-4)) / d;
                    return l * l;
                }
                // Chiang 2016 的美术参数 → HairParams（同 HAIR.bsdf：v = v0·(1, ¼, 4, 4)，s = √(π/8)·(…)，α′ = −α）
                HairParams MakeHairParams(float3 sigmaA, float betaM, float betaN, float alphaDeg) {
                    HairParams P;
                    P.eta = 1.55;
                    P.sigmaA = sigmaA;
                    float v0 = Pow2(0.726 * betaM + 0.812 * betaM * betaM + 3.7 * pow(betaM, 20));
                    P.v[0] = v0; P.v[1] = 0.25 * v0; P.v[2] = 4 * v0; P.v[3] = 4 * v0;
                    P.s = 0.626657069 * (0.265 * betaN + 1.194 * betaN * betaN + 5.372 * pow(betaN, 22));
                    P.sin2kAlpha[0] = sin(-alphaDeg * PI / 180);          // 切线 根 → 梢：R 偏向发根
                    P.cos2kAlpha[0] = SafeSqrt(1 - Pow2(P.sin2kAlpha[0]));
                    [unroll] for (int k = 1; k < 3; ++k) {                 // 倍角：sin 2x = 2 sin x cos x，cos 2x = cos²x − sin²x
                        P.sin2kAlpha[k] = 2 * P.cos2kAlpha[k - 1] * P.sin2kAlpha[k - 1];
                        P.cos2kAlpha[k] = Pow2(P.cos2kAlpha[k - 1]) - Pow2(P.sin2kAlpha[k - 1]);
                    }
                    return P;
                }`,
            },
            {
                title: '随机阈值 · hashed / dithered alpha（Step 08）', code: `
                // (像素 x, 像素 y, 物体种子 ^ 帧号) → [0,1) 的 alpha 阈值：三级 PCG 级联，层间、帧间互相独立（JS：HC.hash3）
                float Hash3(uint3 v) { return PcgHash(v.x + PcgHash(v.y + PcgHash(v.z))) * (1.0 / 4294967296.0); }`,
            },
            {
                title: 'TAA 辅助 · PS_TAAResolve（节点详情）', code: `
                cbuffer TAACB : register(b3) { float2 ViewportSize; float2 InvViewportSize; uint FrameCount; };
                Texture2D<float4> SceneColorHair;   Texture2D<float4> TAAHistory;
                Texture2D<float2> SceneVelocity;    // 运动矢量 = uv_cur − uv_prev（静止相机时为 0）
                // 像素中心（SV_Position.xy）→ 同一表面点在上一帧的 UV，用来采样历史
                float2 Reproject(float2 pixel) {
                    return pixel * InvViewportSize - SceneVelocity[uint2(pixel)];
                }
                // 当前帧 3×3 邻域的逐通道最小值（边界 clamp）：和 NeighborhoodMax 一起构成历史颜色的夹取盒
                float3 NeighborhoodMin(float2 pixel) {
                    float3 m = 1e30;
                    [unroll] for (int y = -1; y <= 1; ++y)
                        [unroll] for (int x = -1; x <= 1; ++x)
                            m = min(m, SceneColorHair[clamp(int2(pixel) + int2(x, y), 0, int2(ViewportSize) - 1)].rgb);
                    return m;
                }
                // 当前帧 3×3 邻域的逐通道最大值：历史落在 [min, max] 外说明这里的内容变了（运动 / 遮挡），夹回来防拖影
                float3 NeighborhoodMax(float2 pixel) {
                    float3 m = -1e30;
                    [unroll] for (int y = -1; y <= 1; ++y)
                        [unroll] for (int x = -1; x <= 1; ++x)
                            m = max(m, SceneColorHair[clamp(int2(pixel) + int2(x, y), 0, int2(ViewportSize) - 1)].rgb);
                    return m;
                }`,
            },
        ],
    },

    references: [
        { title: 'Practical Real-Time Hair Rendering and Shading', author: 'Thorsten Scheuermann', venue: 'SIGGRAPH Sketches', year: 2004, url: 'http://shaderwrangler.com/publications/hairsketch/', note: '插片 + 平移切线的两条高光、lerp(0.25,1,N·L) 漫反射、4 pass 近似排序' },
        { title: 'Hair Rendering and Shading', author: 'Thorsten Scheuermann', venue: 'GDC 2004（ATI 讲座；另见 ShaderX3 同名章节）', year: 2004, note: 'ShiftTangent / StrandSpecular 原始 HLSL；注意其中 shift 方向的表述与 sketch 相反' },
        { title: 'Computing Alpha Mipmaps', author: 'Ignacio Castaño', venue: 'The Witness 开发博客', year: 2010, url: 'http://the-witness.net/news/2010/09/computing-alpha-mipmaps/', note: '每级 mip 二分查找 alpha 缩放，保持 alpha test 覆盖率（NVTT 中有实现）' },
        { title: 'Anti-aliased Alpha Test: The Esoteric Alpha To Coverage', author: 'Ben Golus', venue: 'Medium', url: 'https://bgolus.medium.com/anti-aliased-alpha-test-the-esoteric-alpha-to-coverage-8b177335ae4f', note: 'alpha test、A2C 与 mip 缩放的对比' },
        { title: 'Hashed Alpha Testing', author: 'Chris Wyman, Morgan McGuire', venue: 'I3D', year: 2017, url: 'https://research.nvidia.com/labs/rtr/publication/wyman2017hashed/', note: '按物体空间坐标 hash 的随机 alpha 阈值 + TAA' },
        { title: 'Stochastic Transparency', author: 'Eric Enderton, Erik Sintorn, Peter Shirley, David Luebke', venue: 'I3D', year: 2010, note: '随机覆盖的期望 = 正确排序的混合' },
        { title: 'Setting Up Cards and Meshes for Grooms in Unreal Engine', author: 'Epic Games', venue: 'Unreal Engine 文档', url: 'https://dev.epicgames.com/documentation/unreal-engine/setting-up-cards-and-meshes-for-grooms-in-unreal-engine', note: 'Groom LOD = strands / cards / meshes；cards 纹理槽 Depth / Coverage / Tangent / Attributes / Material / Auxiliary' },
        { title: 'Hair Card Generator for Grooms in Unreal Engine', author: 'Epic Games', venue: 'Unreal Engine 文档', url: 'https://dev.epicgames.com/documentation/unreal-engine/hair-card-generator-for-grooms-in-unreal-engine', note: '从 groom 自动生成卡与 atlas（5.4 起 Experimental）' },
        { title: 'Rendering Fur with Three Dimensional Textures', author: 'James T. Kajiya, Timothy L. Kay', venue: 'SIGGRAPH', year: 1989 },
        { title: 'Light Scattering from Human Hair Fibers', author: 'Stephen R. Marschner, Henrik Wann Jensen, Mike Cammarano, Steve Worley, Pat Hanrahan', venue: 'ACM TOG (SIGGRAPH)', year: 2003, url: 'https://graphics.stanford.edu/papers/hair/hair-sg03final.pdf', note: 'R 偏根、TRT 偏梢（散射锥角度）；头上次高光出现在主高光上方' },
        { title: 'A Practical and Controllable Hair and Fur Model for Production Path Tracing', author: 'Matt Jen-Yuan Chiang, Benedikt Bitterli, Chuck Tappan, Brent Burley', venue: 'Computer Graphics Forum (EGSR)', year: 2016 },
    ],
});
