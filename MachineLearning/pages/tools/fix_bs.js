// 一次性：把文件里误写进去的退格符（0x08，本该是正则里的 \b）换回 \b
const fs = require('fs');
const path = require('path');
for (const f of (process.argv.length > 2 ? process.argv.slice(2) : ['tools_common.js', 'tools_catalog.js', 'tools_cpp.js'])) {
    const p = path.join(__dirname, '..', f);
    const s = fs.readFileSync(p, 'utf8'), n = (s.match(/\x08/g) || []).length;
    if (n) fs.writeFileSync(p, s.replace(/\x08/g, String.fromCharCode(92) + 'b'));
    console.log(f, 'fixed', n);
}
