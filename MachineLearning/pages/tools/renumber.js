// renumber.js — 一次性：按「分类顺序」重新编号（共享基础 01–05、通用架构 06–07、大语言模型 08–09、分化领域 10–15）。
// 改：文件名（.lab.js；旧 .html 删除后重建）、tools/pNN_* 脚本名、正文里「第 NN 篇」（含列表 / 区间）、LM.navHtml('NN')、「下一篇（NN）」、文件引用。
// 用法（在 pages/ 下）：node tools/renumber.js [--apply]
const fs = require('fs');
const path = require('path');
const APPLY = process.argv.includes('--apply');
const pages = path.join(__dirname, '..'), tools = __dirname;
const MAP = { '03': '06', '04': '07', '05': '08', '06': '09', '07': '10', '08': '03', '09': '04', '10': '05' };
const mp = (n) => MAP[n] || n;
const SLUGS = { '03': 'tokenizer_embedding', '04': 'transformer', '05': 'pretraining', '06': 'alignment', '07': 'multimodal', '08': 'cnn_unet', '09': 'latent_vae', '10': 'reinforcement_learning' };
const flags = [], stats = {};
const bump = (f, k) => { stats[f] = stats[f] || {}; stats[f][k] = (stats[f][k] || 0) + 1; };

/** 把一组篇号压缩成「03–05、11–15」这样的写法 */
function compress(nums) {
    const a = [...new Set(nums.map(Number))].sort((x, y) => x - y), runs = [];
    for (const v of a) { const r = runs[runs.length - 1]; if (r && v === r[1] + 1) r[1] = v; else runs.push([v, v]); }
    const f = (v) => String(v).padStart(2, '0');
    return runs.map(([s, e]) => (s === e ? f(s) : e === s + 1 ? `${f(s)}、${f(e)}` : `${f(s)}–${f(e)}`)).join('、');
}
function rewrite(file, src) {
    let s = src;
    // R1 文件名
    s = s.replace(/\b(0[3-9]|10)_(tokenizer_embedding|transformer|pretraining|alignment|multimodal|cnn_unet|latent_vae|reinforcement_learning)\b/g, (all, n, slug) => {
        if (SLUGS[n] !== slug) { flags.push(`${file}: 文件名不匹配 ${all}`); return all; }
        bump(file, 'file'); return `${mp(n)}_${slug}`;
    });
    // R2 工具脚本名、截图目录
    s = s.replace(/\bp(0[3-9]|10)_([a-z0-9_]+)/g, (all, n, rest) => { bump(file, 'tool'); return `p${mp(n)}_${rest}`; });
    s = s.replace(/\bshots(0[3-9]|10)\b/g, (all, n) => { bump(file, 'shots'); return `shots${mp(n)}`; });
    // R3 「第 NN 篇」（列表、区间）
    s = s.replace(/第(\s*)(\d\d)((?:\s*[、/–\-~～,，和与及至到]\s*(?:第\s*)?\d\d)*)(\s*)篇/g, (all, sp1, first, rest, sp2) => {
        const body = first + rest, nums = body.match(/\d\d/g), seps = body.split(/\d\d/).slice(1, -1).map((x) => x.replace(/第/g, '').trim());
        if (nums.length === 1) { bump(file, 'ch'); return `第${sp1}${mp(first)}${sp2}篇`; }
        const isRange = seps.some((x) => /[–\-~～至到]/.test(x));
        if (!isRange) { bump(file, 'list'); let k = 0; return all.replace(/\d\d/g, (n) => (k++, mp(n))); }
        if (nums.length !== 2 || seps.length !== 1) { flags.push(`${file}: 复杂区间 ${all}`); return all; }
        const a = +nums[0], b = +nums[1], img = []; for (let v = a; v <= b; v++) img.push(mp(String(v).padStart(2, '0')));
        const out = compress(img);
        if (out !== `${mp(nums[0])}–${mp(nums[1])}`) flags.push(`${file}: 区间 ${all} → 第 ${out} 篇（不再连续，已改写成列表）`);
        bump(file, 'range'); return `第${sp1}${out}${sp2}篇`;
    });
    // R4 导航
    s = s.replace(/LM\.navHtml\('(\d\d)'/g, (all, n) => { bump(file, 'nav'); return `LM.navHtml('${mp(n)}'`; });
    // R5 「下一篇（NN）」
    s = s.replace(/下一篇（(\d\d)）/g, (all, n) => { bump(file, 'next'); flags.push(`${file}: 请人工核对「下一篇（${n}）」→（${mp(n)}）`); return `下一篇（${mp(n)}）`; });
    return s;
}
const targets = [];
for (const f of fs.readdirSync(pages)) if (/\.lab\.js$/.test(f) || f === 'llm_common.js') targets.push(path.join(pages, f));
for (const f of fs.readdirSync(path.join(pages, 'cpp'))) if (/\.(h|cpp)$/.test(f)) targets.push(path.join(pages, 'cpp', f));
for (const f of fs.readdirSync(tools)) if (/\.(js|mjs|md)$/.test(f) && f !== 'renumber.js') targets.push(path.join(tools, f));
const changed = [];
for (const f of targets) {
    const src = fs.readFileSync(f, 'utf8'), out = rewrite(path.relative(pages, f), src);
    if (out !== src) { changed.push(f); if (APPLY) fs.writeFileSync(f, out); }
}
// 文件改名（两步：先改成临时名，避免 03 ↔ 06 这样的互换冲突）
const renames = [];
for (const [o, slug] of Object.entries(SLUGS)) renames.push([path.join(pages, `${o}_${slug}.lab.js`), path.join(pages, `${mp(o)}_${slug}.lab.js`)]);
for (const f of fs.readdirSync(tools)) { const m = /^p(0[3-9]|10)_(.+)$/.exec(f); if (m) renames.push([path.join(tools, f), path.join(tools, `p${mp(m[1])}_${m[2]}`)]); }
console.log(`${changed.length} files with text changes; ${renames.length} renames`);
for (const [f, st] of Object.entries(stats)) console.log('  ', f.padEnd(42), JSON.stringify(st));
flags.forEach((x) => console.log('  ⚑', x));
if (APPLY) {
    const tmp = renames.map(([a, b]) => { const t = a + '.__ren__'; fs.renameSync(a, t); return [t, b]; });
    for (const [t, b] of tmp) { if (fs.existsSync(b)) throw new Error('target exists: ' + b); fs.renameSync(t, b); }
    for (const [o, slug] of Object.entries(SLUGS)) { const h = path.join(pages, `${o}_${slug}.html`); if (fs.existsSync(h)) fs.unlinkSync(h); }
    for (const d of fs.readdirSync(tools)) if (/^shots\d\d$/.test(d)) fs.rmSync(path.join(tools, d), { recursive: true, force: true });
    console.log('applied');
}
