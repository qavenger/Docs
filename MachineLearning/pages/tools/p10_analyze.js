// 为页面 note 里的每一句「实测」取数：页面默认设置（PAGE_SEED）下的 CLIP（2000 步）与 LLaVA（3000 步）
// 用法：node pages/tools/p10_analyze.js
const { LM, P07 } = require('./p10_load.js');
const pct = (v) => (v * 100).toFixed(0) + '%';
const data = P07.evalData(), G = data.gallery, n = G.length;

// ── CLIP（与页面 Step 04 默认设置完全相同的运行） ──
const cr = P07.newClipRun({ seed: P07.PAGE_SEED, N: 8, tau: 0.05, hard: false, hold: 'none' }, data);
while (P07.clipWork(cr));
const g = cr.gal, z = cr.zs;
console.log(`CLIP @${g.it}: gallery i2t ${pct(g.i2t)} t2i ${pct(g.t2i)} mean cos pos ${g.pos.toFixed(3)} neg ${g.neg.toFixed(3)}; zero-shot ${z.acc.map(pct).join(' / ')}`);
[[8, 9], [10, 11]].forEach(([a, b]) => console.log(`  swap pair ${a}/${b}: cos(I${a},T${a}) ${g.sim[a * n + a].toFixed(3)} vs cos(I${a},T${b}) ${g.sim[a * n + b].toFixed(3)}; cos(I${b},T${b}) ${g.sim[b * n + b].toFixed(3)} vs cos(I${b},T${a}) ${g.sim[b * n + a].toFixed(3)}`));
// 双图形图片的 zero-shot：前 3 名
let mixed = 0;
for (let i = 8; i < 16; i++) {
    const s = z.galSims[0].subarray(i * 12, (i + 1) * 12), p = LM.softmax(Array.from(s), new Float32Array(12), 0.05), top = LM.topk(p, 3);
    const real = new Set(G[i].objs.map((o) => o.c * 3 + o.s)); if (!real.has(top[0])) mixed++;
    console.log(`  zs img ${i} ${G[i].caption}: ${top.map((k) => `${P07.clsName(k)} ${p[k].toFixed(2)}${real.has(k) ? '✓' : ''}`).join(' | ')}`);
}
console.log(`  two-object images whose top-1 prompt is NOT one of their two objects: ${mixed}/8`);
// ⟨eos⟩ 每个头看哪个字
const role = (cap, j) => { const ch = cap[j - 1]; if (j === 0) return 'bos'; if (j === cap.length + 1) return 'eos'; const colors = P07.COLORS.map((c) => c.name[0]), shapes = P07.SHAPES.map((s) => s[0]); const k = cap.indexOf('在'); const second = k >= 0 && j - 1 > k; if (colors.includes(ch)) return (second ? 'color2' : 'color1'); if (shapes.includes(ch)) return (second ? 'shape2' : 'shape1'); return 'other:' + ch; };
for (let h = 0; h < 4; h++) {
    const cnt = {};
    G.forEach((sc, b) => { const a = g.eosAtt[b][h]; const j = LM.argmax(a); const r = role(sc.caption, j); cnt[r] = (cnt[r] || 0) + 1; });
    console.log(`  text tower head ${h}: argmax role over 16 captions`, JSON.stringify(cnt));
}

// ── 冻结的 ViT（内嵌权重 = 上面这次运行的图像塔，f16）：[CLS] 注意力 ──
const vit = P07.bakedViT(), rv = P07.embedImages(vit, G, { trace: true }), att = rv.tr[0].att;
let md = 0; vit.params.forEach((p, i) => { for (let k = 0; k < p.size; k++) md = Math.max(md, Math.abs(p.data[k] - cr.tr.clip.img.params[i].data[k])); });
console.log(`baked ViT vs live run at step 2000: max |Δw| = ${md.toExponential(2)} (float16 rounding)`);
const covers = (sc, p) => { const gy = Math.floor(p / 4), gx = p % 4; let c = 0; for (let y = gy * 6; y < gy * 6 + 6; y++) for (let x = gx * 6; x < gx * 6 + 6; x++) { const i = (y * 24 + x) * 3; const bg = sc.img[i] + sc.img[i + 1] + sc.img[i + 2]; if (bg > 0.9) c++; } return c > 0; };
for (let h = 0; h < 4; h++) {
    let mx = 0, onObj = 0, self = 0, leftRight = { L: 0, R: 0 };
    G.forEach((sc, b) => { const o = ((b * 4 + h) * 17) * 17, row = Array.from(att.subarray(o + 1, o + 17)); const m = Math.max(...row), p = row.indexOf(m); mx += m / n; self += att[o] / n; if (covers(sc, p)) onObj++; if (!sc.single) { if (p % 4 < 2) leftRight.L++; else leftRight.R++; } });
    console.log(`  ViT head ${h}: mean max ${pct(mx)}, mean self ${pct(self)}, argmax patch on an object ${onObj}/16, two-object images: argmax left ${leftRight.L} / right ${leftRight.R}`);
}

// ── LLaVA（页面默认：CLIP 编码器） ──
const lr = P07.newLlavaRun({ seed: P07.PAGE_SEED, vit }, data);
while (P07.llavaWork(lr));
const last = lr.accHist[lr.accHist.length - 1];
console.log(`LLaVA @${last.it}: exact ${pct(last.exact)} (single ${pct(last.exact1)}, double ${pct(last.exact2)}), attr ${pct(last.attr)}`);
console.log('  acc history', lr.accHist.filter((r) => [150, 300, 600, 900, 1200, 1800, 2400, 3000].includes(r.it)).map((r) => `${r.it}:${pct(r.exact)}`).join(' '));
const mm = lr.tr.mm, agg = {};
for (let i = 8; i < 16; i++) {
    const vis = P07.visFeatures(vit, [G[i]], 0), pv = LM.AG.noGrad(() => P07.projFwd(mm.proj, vis)), gen = P07.decode(mm.lm, pv.data, 16, 16, { attn: true });
    const cap = gen.text, ids = gen.ids.concat([P07.EOS]);
    ids.forEach((id, t) => {
        const r = t === gen.ids.length ? 'eos' : role(cap, t + 1);
        if (!['color1', 'shape1', 'color2', 'shape2'].includes(r)) return;
        const heads = gen.att[t][1], per = heads.map((pr) => { let L = 0, R = 0; for (let j = 0; j < 16; j++) (j % 4 < 2 ? (L += pr[j]) : (R += pr[j])); return [L, R]; });
        const mean = per.reduce((a, b) => [a[0] + b[0] / 4, a[1] + b[1] / 4], [0, 0]);
        (agg[r] = agg[r] || []).push({ mean, per });
    });
}
for (const [r, l] of Object.entries(agg)) {
    const mL = l.reduce((s, x) => s + x.mean[0], 0) / l.length, mR = l.reduce((s, x) => s + x.mean[1], 0) / l.length;
    const hs = [0, 1, 2, 3].map((h) => { const a = l.reduce((s, x) => s + x.per[h][0], 0) / l.length, b = l.reduce((s, x) => s + x.per[h][1], 0) / l.length; return `h${h} L${pct(a)}/R${pct(b)}`; });
    console.log(`  LLaVA layer-2 attention when predicting ${r} (8 two-object gallery images): mean over heads L ${pct(mL)} / R ${pct(mR)} | ${hs.join(' ')}`);
}
