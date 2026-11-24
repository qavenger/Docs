// p02_save_eval.js — 从 validate_doc.mjs --eval 的输出里取出 JSON：截图写成 PNG，其余数值打印出来
// 用法：node tools/p02_save_eval.js <validator 输出文件> <截图目录>
const fs = require('fs'), path = require('path');
const [, , file, dir] = process.argv;
const txt = fs.readFileSync(file, 'utf8'), at = txt.indexOf('eval →');
if (at < 0) { console.log(txt.slice(-3000)); throw new Error('no eval output'); }
let i = txt.indexOf('{', at), depth = 0, j = i, inStr = false;
for (; j < txt.length; j++) {
    const ch = txt[j];
    if (inStr) { if (ch === '\\') j++; else if (ch === '"') inStr = false; continue; }
    if (ch === '"') inStr = true; else if (ch === '{') depth++; else if (ch === '}') { depth--; if (!depth) break; }
}
const out = JSON.parse(txt.slice(i, j + 1));
fs.mkdirSync(dir, { recursive: true });
for (const [id, url] of Object.entries(out.imgs || {})) fs.writeFileSync(path.join(dir, `eval-${id}.png`), Buffer.from(url.split(',')[1], 'base64'));
delete out.imgs;
console.log(JSON.stringify(out, null, 1));
