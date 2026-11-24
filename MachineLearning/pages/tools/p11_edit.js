// 小工具：对 11_diffusion.lab.js 做精确的字符串替换（node tools/p11_edit.js <edits.json>）
const fs = require('fs'), path = require('path');
const file = path.join(__dirname, '..', '11_diffusion.lab.js');
let s = fs.readFileSync(file, 'utf8');
const edits = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
for (const [a, b] of edits) { if (!s.includes(a)) { console.error('missing:', a.slice(0, 90)); process.exit(1); } s = s.replace(a, b); }
fs.writeFileSync(file, s); console.log('applied', edits.length);
