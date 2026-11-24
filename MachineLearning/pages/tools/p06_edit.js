// p06_edit.js — 对 06_alignment.lab.js 做精确字符串替换（任何一处找不到就报错退出）。用法：node tools/p06_edit.js <edits.json>
const fs = require('fs'), path = require('path');
const LAB = path.join(__dirname, '..', '06_alignment.lab.js');
const edits = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
let s = fs.readFileSync(LAB, 'utf8');
edits.forEach(([a, b], i) => { const n = s.split(a).length - 1; if (n !== 1) { console.error(`edit #${i}: found ${n} matches for: ${a.slice(0, 100)}`); process.exit(1); } s = s.replace(a, () => b); });
fs.writeFileSync(LAB, s);
console.log(`applied ${edits.length} edits`);
